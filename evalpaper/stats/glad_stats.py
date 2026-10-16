"""Statistics, leaderboards and the automatic PREREG / amendment evaluation (v2) for the label-free GLAD
re-evaluation.

  export PYTHONDONTWRITEBYTECODE=1
  /N/slate/yujlu/envs/grail-prelim/bin/python -B prelim/evalpaper/stats/glad_stats.py [--final] [--B 4000]

v2 (2026-10-13) implements PREREG.md and PREREG_AMENDMENTS.md A0-A24. v1 (2026-10-12) is archived unchanged in
stats/archive/glad_stats_v1.py; its outputs (prelim/results/eval_leaderboard_*.csv without "v2") are never touched.
The input schemas and the frozen hash are in stats/FROZEN.md.

INPUTS (missing files are reported, never fatal, unless noted)
  <res>/eval_folds.csv (+ eval_folds_extra.csv)   frozen 12 deep detectors, written by harness/collect.py
  <res>/deep_rerun_parts/*.json                   MUTAG runs of the old 4 (A10 §5; collect.py skips MUTAG)
  harness/parts/*.json + scores npz               per-run trajectories and per-graph scores (A15 audit, A21 ties)
  ext/parts/*.json + scores npz                   E5 (MUSE LGKDE GLADPro GLADMamba SEGO) and E4 (GLASS ProtoGLAD
                                                  HC-GLAD SUL-GUL); protocol rows are produced by each adapter's
                                                  own collect/derive code, imported read-only and run with its
                                                  outputs redirected to a temporary directory (stdout discarded)
  <res>/eval_kernels_folds.csv                    KernelGLAD (stratum kernel)
  <res>/grail0_folds.csv                          GRAIL-0 and a-priori controls (stratum reference)
  <res>/m1_selection_folds.csv                    M1 label-free selectors (old 4; descriptive)
  <res>/eval_typed*.csv (+ m2 GX controls)        A9 typed test (G2(c)); schema in FROZEN.md
  <res>/eval_version_a96137b_folds.csv            A17 version-sensitivity runs; schema in FROZEN.md
  stats/diverged_registry.csv + '# DIVERGED' lines in harness/ext task files   A12 registry

OUTPUTS  <out>/eval_leaderboard_v2_*.csv, <out>/eval_leaderboard_v2_prereg.json, stats/REPORT.md (the previous
REPORT.md is copied to stats/archive/ first; nothing is deleted).

RULES (where each amendment is implemented)
  A5   GraphDE: paper orientation is the main row; UB-GOLD orientation rows -> repro_only table.
  A9   G2(c) = PASS iff H4 CONFIRMED on the 8 confirmatory methods (g2c_a9); H4s scope test; 4-method M2
       computation kept as descriptive only.
  A10/A11  strata frozen12 / E5 / E4 / kernel / reference; G1 and G2 on the frozen 12 only; E5, E4, recent
       (E5+E4) and pooled views are secondary.  ORACLE of E5/E4 = the method's own test-label rule where it has
       one, else UB-GOLD's rule (oracle_rule column); the UB-GOLD-rule ORACLE is the secondary protocol
       oracle_ubgold for every iterative method.
  A12  DIVERGED runs: no metric, never in aggregates, coverage shows them; diverged UB9 cell => no G1 coverage.
  A13/A16/A18/A19/A22/A23/A24  per-method row labels (hp_label, method_label, oracle_rule); GLASS is NOT
       transductive (A24 withdraws the A23 §1 flag); identical-config noise estimate computed from the grid.
  A14  G1 under VAL-NORMAL and under LAST; primary G1 = PASS only if both pass, else "SELECTOR-DEPENDENT"
       (not met for GO); VAL-NORMAL-degenerate flag per method x hp setting; holdout_obj sensitivity.
  A15  re-audited on every protocol row from the per-run trajectories (a15_audit): ORACLE never at a non-finite
       point, VAL-NORMAL never at a non-finite held-out mean, non-finite LAST -> chance (50 / anomaly rate / 95)
       flagged nonfinite_selected, all-non-finite run -> chance under every protocol; sensitivity drops flagged runs.
  A17  version sensitivity GLocalKD/GLADC a96137b vs c940457 (paired), never part of G1/G2.
  A20  checkpoint-free methods (ProtoGLAD, KernelGLAD, GRAIL-0 + controls) excluded from pooled checkpoint
       inflation; they enter the HP axis and leaderboards.
  A21  identical-config methods (declared: SUL-GUL, GLASS; plus automatic detection) excluded from pooled HP leak;
       tie_collapsed flag (distinct finite test scores < max(10, 5% of test size) at a selected point).
"""
import argparse
import contextlib
import glob
import hashlib
import importlib.util
import io
import json
import os
import os.path as osp
import re
import shutil
import sys
import tempfile
import time
from itertools import combinations

sys.dont_write_bytecode = True   # importing the adapters must never write __pycache__ into their folders

import numpy as np
import pandas as pd
from scipy import stats

VERSION = "v2"
HERE = osp.dirname(osp.abspath(__file__))
EVALPAPER = osp.dirname(HERE)
PRELIM = osp.dirname(EVALPAPER)
RES = osp.join(PRELIM, "results")
REPORT = osp.join(HERE, "REPORT.md")
HARNESS = osp.join(EVALPAPER, "harness")
EXT = osp.join(EVALPAPER, "ext")
PROTOGLAD_GRID = "/N/slate/yujlu/grail_data/ext_protoglad/grid"
MS_MANIFEST = "/N/slate/yujlu/grail_data/molswap/manifest.json"     # A9.1: frozen MolSwap yields per seed

UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
SECONDARY = ["DD", "COLLAB", "Tox21_p53", "Tox21_MMP", "MUTAG"]          # MUTAG added by A10 §5
SETS = {"UB9": UB9, "secondary": SECONDARY}
ALL_DS = UB9 + SECONDARY
NFOLD = 5
DS_ALIASES = {"PROTEINS": "PROTEINS_full", "IMDB-B": "IMDB-BINARY", "REDDIT-B": "REDDIT-BINARY",
              "p53": "Tox21_p53", "MMP": "Tox21_MMP"}

DEEP12 = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD", "GLADC", "CVTGAD", "GOOD-D", "GraphDE",
          "GraphCL_IF", "GraphCL_OCSVM", "InfoGraph_IF", "InfoGraph_OCSVM"]
OLD4 = DEEP12[:4]
NEW8 = DEEP12[4:]
E5 = ["MUSE", "LGKDE", "GLADPro", "GLADMamba", "SEGO"]
E4 = ["GLASS", "ProtoGLAD", "HC-GLAD", "SUL-GUL"]
KERNELS = ["WL-IF", "WL-OCSVM", "PK-IF", "PK-OCSVM"]
REF_MAP = {"grail0_sumz": "GRAIL-0", "ctrl_size_lw": "size-only", "ctrl_gc_lw": "graphlet-count+LW",
           "ctrl_gc_iforest": "graphlet-count+iForest", "ctrl_wl_iforest": "WL-hist+iForest (ctrl)"}
REFERENCES = list(REF_MAP.values())
STRATA = ["frozen12", "E5", "E4", "kernel", "reference"]
POOLS = {"frozen12": ["frozen12"], "E5": ["E5"], "E4": ["E4"], "recent(E5+E4)": ["E5", "E4"],
         "frozen12+E5": ["frozen12", "E5"], "frozen12+E5+E4": ["frozen12", "E5", "E4"],
         "kernel": ["kernel"], "reference": ["reference"], "all": STRATA}
BASES = ["oracle", "oracle_ubgold", "last", "valnormal"]
PROTOCOLS = BASES + ["hpfree_" + b for b in BASES]
METRICS = ("auroc", "auprc", "fpr95")
PATIENCE = 30
CHANCE_FPR95 = 95.0

# A20: no checkpoint axis (their ORACLE = LAST = VAL-NORMAL by construction)
CHECKPOINT_FREE = set(["ProtoGLAD"] + KERNELS + REFERENCES)
# A21 §2 (SUL-GUL), A23 §3 / A24 §2 (GLASS): 'ubgold' and 'hpfree' resolve to identical options
IDENTICAL_CONFIG_DECLARED = {"SUL-GUL": "A21 §2", "GLASS": "A23 §3, A24 §2"}
NO_HP_AXIS = set(REFERENCES)      # a-priori references: no hyperparameters at all
# ORACLE rule of the primary 'oracle' row (A10 §2, A11 §2, A14, A15, A16 §3, A23 §2)
ORACLE_RULE = {**{m: "ubgold_patience30" for m in DEEP12},
               "MUSE": "ubgold_patience30 (MUSE has no own test-label rule; A10 §2)",
               "SEGO": "ubgold_patience30 (SEGO's own reporting rule is LAST; A16 §3)",
               "LGKDE": "own: max (AUROC+AUPR)/2 with LGKDE EarlyStopping (A10 §2)",
               "GLADPro": "own: epoch with max 5-fold mean test AUROC (A10 §2, A15)",
               "GLADMamba": "own: common epoch with max 5-fold mean test AUROC (A14)",
               "HC-GLAD": "own: common epoch with max 5-fold mean test AUROC (A11 §2)",
               "SUL-GUL": "own: GUL best test epoch over 5-fold mean (A11 §2)",
               "GLASS": "own: test-selected checkpoint + best of 24 rules (A23 §2)",
               "ProtoGLAD": "none: no checkpoint ('tuned' cell; A11 §2, A20)",
               **{m: "none: no checkpoint (test-tuned HP 'tuned')" for m in KERNELS},
               **{m: "none: a-priori reference" for m in REFERENCES}}
UBGOLD_RULE_METHODS = set(DEEP12) | {"MUSE", "SEGO"}          # their 'oracle' IS the UB-GOLD rule
COMMON_EPOCH_OWN = {"GLADPro", "GLADMamba", "HC-GLAD", "SUL-GUL"}  # own rule = one epoch for all 5 folds
EF_FILTER = set(DEEP12) | {"MUSE", "SEGO", "GLADMamba", "HC-GLAD", "LGKDE"}  # ORACLE scans epoch % eval_freq == 0
METHOD_LABEL = {"SUL-GUL": "SUL-GUL (GUL only; SUL not released)", "GraphDE": "GraphDE (paper orientation, A5)",
                "SEGO": "SEGO ('ubgold' = released-modal)", "LGKDE": "LGKDE ('ubgold' = MUTAG.yaml for all)"}
METHOD_NOTES = {
    "MUSE": "A13: modal anom_0 config for COX2/ENZYMES/MUTAG, 'protein' config for PROTEINS_full; 10 snapshots; "
            "detector grid rows (oracle_grid) descriptive only; labelled-validation protocol not emulated.",
    "LGKDE": "A18: 'ubgold' rows = MUTAG.yaml for all datasets; HP-FREE = config-reader defaults; reproduction "
             "criterion (A10 §4) not met (1/4), port bitwise identical.",
    "GLADPro": "A15 ratifications: README lines, repository defaults; ORACLE-own + UB-GOLD-rule ORACLE.",
    "GLADMamba": "A14: batch sizes per dataset in HP-FREE; modal config for PROTEINS_full/MUTAG; VAL-NORMAL scale "
                 "artifact (A14 trigger).",
    "SEGO": "A16: 'ubgold' = released-modal (not author-tuned); ORACLE = UB-GOLD rule; own reporting rule = LAST.",
    "GLASS": "A23/A24: label-free columns use rcsm_full_test_cdf (label-free AND inductive, A24 §1); ORACLE-own = "
             "test-selected checkpoint + best of 24 rules; ubgold == hpfree (A21 identical config).",
    "ProtoGLAD": "A20: no checkpoint axis; UB9 'tuned' = test-grid-best (test-label-selected upper bound), MUTAG "
                 "'tuned' = README line; CPU rows.",
    "HC-GLAD": "A19: memory-feasibility rule (modal fallback); ORACLE-own = common best epoch.",
    "SUL-GUL": "A21/A22: GUL only (SUL not released); normal-only (poll_ratio 0); 'ubgold' = released code defaults "
               "(500 epochs; paper states 1000); ubgold == hpfree (identical config).",
}
# adapters (E5/E4): script, how to run it, output attribute and files. All outputs go to a temporary directory.
ADAPTERS = {
    "MUSE": dict(script="MUSE/collect_muse.py", mode="main", outattr="RES", folds="muse_folds.csv",
                 extra=["muse_extra.csv"]),
    "LGKDE": dict(script="LGKDE/derive.py", mode="rows_of"),
    "GLADPro": dict(script="GLADPro/derive.py", mode="main", outattr="OUT", folds="gladpro_folds.csv",
                    extra=["gladpro_extra.csv"]),
    "GLADMamba": dict(script="GLADMamba/collect.py", mode="main", outattr="RES", folds="ext_gladmamba_folds.csv",
                      extra=["ext_gladmamba_folds_extra.csv"]),
    "SEGO": dict(script="SEGO/collect_sego.py", mode="main", outattr="OUT", folds="sego_folds.csv",
                 extra=["sego_folds_extra.csv"]),
    "GLASS": dict(script="GLASS/derive.py", mode="main", outattr="OUT", folds="glass_folds.csv",
                  extra=["glass_extra.csv"]),
    "ProtoGLAD": dict(script="ProtoGLAD/derive.py", mode="protoglad"),
    "HC-GLAD": dict(script="HC-GLAD/collect.py", mode="main", outattr="RES", folds="ext_hcglad_folds.csv",
                    extra=["ext_hcglad_folds_extra.csv"]),
    "SUL-GUL": dict(script="SUL-GUL/derive.py", mode="main", outattr="OUT", folds="sulgul_folds.csv",
                    extra=["sulgul_extra.csv"]),
}
HP_IGNORE = {"hp", "gpu", "device", "data_root", "ckpt_root", "result_root", "save_dir", "base.save_dir",
             "base.gpu_id", "text_device", "use_cache", "threads"}
TRAJ_ID_KEYS = ["epoch", "auroc", "auprc", "fpr95", "holdout_score", "holdout_loss", "holdout_obj", "train_loss"]

# A9 typed test
GX_TYPES = ["T1a", "T3a", "T3b"]
MS_TYPES = ["MS1", "MS2", "MS3"]
CTRL_ALIASES = {"gc_lw": {"GRAIL0:ctrl_gc_lw", "ctrl_gc_lw", "graphlet-count+LW"},
                "size_lw": {"GRAIL0:ctrl_size_lw", "ctrl_size_lw", "size-only"},
                "elp_lw": {"POSTHOC:ctrl_edgelab_lw", "ctrl_edgelab_lw", "edge-label-pair+LW"}}
A9 = dict(valid_gc_min=85.0, valid_size_max=60.0, confound_elp_min=85.0, blind_max=60.0, min_anomalies=100,
          pooled_blind_min=0.75, ms_blind_min=0.60, ms_methods_min=5, refuted_ms_max=0.40, min_valid_ms=2,
          h4s_drop_pp=25.0)
# A9 §2 as amended by A25 (user decision, 2026-10-13 19:48:37 EDT): a (type, variant) is infeasible if ANY seed yields
# fewer than 100 anomalies, for size-matched and size-changing variants alike. Yields are read from the frozen MolSwap
# manifest (counts_per_seed; its 'status' field encodes the superseded 150 floor and is NOT used) or from n_anomalies.
A9_FEASIBILITY_READING = "any_seed_below_100_infeasible (A25)"
SIZE_VARIANT_ALIASES = {"sizechg": "changing", "size-changing": "changing", "size_changing": "changing",
                        "size-matched": "matched", "size_matched": "matched"}
G1_RULE = dict(n_methods_min=8, mean_min=3.0, per_method_min=2.0, frac_min=0.75)


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def canon_method(m):
    key = str(m).strip().replace("-", "_").lower()
    for d in DEEP12 + E5 + E4:
        if d.replace("-", "_").lower() == key:
            return d
    return str(m).strip()


def stratum_of(m):
    if m in DEEP12:
        return "frozen12"
    if m in E5:
        return "E5"
    if m in E4:
        return "E4"
    if m in KERNELS:
        return "kernel"
    if m in REFERENCES:
        return "reference"
    return "other"


def scale100(df, col="auroc"):
    """Percent scale factor for a file of unknown scale, decided on its AUROC column."""
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return 100.0 if len(v) and v.abs().max() <= 1.0 + 1e-9 else 1.0


def sha256_file(p):
    if not osp.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def quiet():
    """Adapters print their own per-dataset tables; those are discarded unread."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def finite(x):
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def hp_of(protocol):
    return "hpfree" if str(protocol).startswith("hpfree_") else "ubgold"


def base_of(protocol):
    p = str(protocol)
    return p[len("hpfree_"):] if p.startswith("hpfree_") else p


def hp_label(method, dataset, hp, hp_config=""):
    cfg = str(hp_config or "")
    if method in REFERENCES:
        return "a-priori (no hyperparameters)"
    if method in KERNELS:
        return "test-tuned grid cell ('tuned')" if hp == "ubgold" else "UB-GOLD defaults ('default')"
    if hp == "hpfree":
        return {"LGKDE": "config-reader defaults; KeyError options from MUTAG.yaml (A18 §1)",
                "SEGO": "treecl.py defaults (A16 §2)", "GLASS": "repository defaults (== ubgold; A23 §3)",
                "SUL-GUL": "repository argparse defaults (== ubgold; A21 §2)",
                "GLADMamba": "repository defaults; per-dataset batch sizes (A14)",
                "HC-GLAD": "repository defaults; memory-feasibility fallback (A19 §2)"}.get(
            method, "label-free: mymain defaults (A1)" if method in DEEP12 else "repository defaults (A10 §3)")
    if method in DEEP12:
        return ("modal UB-GOLD configuration (MUTAG has no UB-GOLD line; A10 §5)" if dataset == "MUTAG"
                else "UB-GOLD per-dataset script (test-tuned)")
    if method == "MUSE":
        if dataset in ("COX2", "ENZYMES", "MUTAG"):
            return "modal over MUSE's released anom_0 entries (A13 §1)"
        if dataset == "PROTEINS_full":
            return "MUSE 'protein' configuration (A13 §2)"
        return "MUSE released per-dataset configuration"
    if method == "LGKDE":
        return "MUTAG.yaml for all datasets (A18 §2)"
    if method == "SEGO":
        return "released-modal (A16 §1)"
    if method == "GLASS":
        return "single released configuration (README line; A23 §3)"
    if method == "SUL-GUL":
        return "released code defaults (500 epochs; paper states 1000) (A22 §3)"
    if method == "ProtoGLAD":
        return ("README evaluation line (A20 §1)" if dataset == "MUTAG"
                else "test-grid-best replay: TEST-LABEL-SELECTED (A11 §2)")
    if method == "GLADPro":
        return "released README configuration (A15)"
    if method in ("GLADMamba", "HC-GLAD"):
        return ("modal released configuration (no released line for this dataset)" if "|modal|" in cfg
                else "released per-dataset script")
    return "released configuration"


# ----------------------------------------------------------------------------------------------
# per-run trajectories (A15 audit, A21 ties, A14 flag, identical-config check)
# ----------------------------------------------------------------------------------------------
def _arr(traj, key):
    return np.array([np.nan if (r.get(key) is None or not isinstance(r.get(key), (int, float)))
                     else float(r.get(key)) for r in traj], dtype=float)


def run_info(path, group):
    J = json.load(open(path))
    meta, traj = J["meta"], J["traj"]
    m = canon_method(meta["method"])
    hpar = meta.get("hparams", {}) or {}
    hold_key = "holdout_score" if "holdout_score" in traj[0] else "holdout_loss"
    au = _arr(traj, "auroc")
    nf = _arr(traj, "nonfinite")
    nf = np.where(np.isfinite(nf), nf, 0)
    hold = _arr(traj, hold_key)
    ne = hpar.get("num_epoch", None)
    try:
        ne = int(ne)
    except (TypeError, ValueError):
        ne = int(max(r["epoch"] for r in traj))
    ident = json.dumps({"train_loss": J.get("train_loss"),
                        "traj": [[r.get(k) for k in TRAJ_ID_KEYS] for r in traj]}, sort_keys=True, default=str)
    return dict(method=m, dataset=DS_ALIASES.get(meta["dataset"], meta["dataset"]), fold=int(meta["fold"]),
                hp=meta.get("hp", "ubgold"), run_id=meta["run_id"], path=path, group=group,
                epochs=np.array([int(r["epoch"]) for r in traj]), auroc=au, auprc=_arr(traj, "auprc"),
                fpr95=_arr(traj, "fpr95"), test_ok=np.isfinite(au) & (nf == 0), hold=hold,
                hold_ok=np.isfinite(hold), sel=_arr(traj, "auroc_sel") if m == "CVTGAD" else au,
                eval_freq=int(hpar.get("eval_freq", 1) or 1), num_epoch=ne,
                n_test=int(meta.get("n_test", 0) or 0), n_anom=int(meta.get("n_test_anomalous", 0) or 0),
                npz=meta.get("scores_npz"), hparams={k: v for k, v in hpar.items() if k not in HP_IGNORE},
                digest=hashlib.sha256(ident.encode()).hexdigest(), tagged=meta.get("max_epochs_override") is not None,
                r0_rule=meta.get("r0_rule"))


def untagged(meta, method_name=None):
    m = method_name or meta["method"]
    stem = f"{m}_{meta['dataset']}_fold{meta['fold']}"
    return meta["run_id"] in (stem, stem + "_hpfree") and meta.get("max_epochs_override") is None


def index_runs(dirs, methods=None):
    """{(group, run_id): run_info}. dirs: list of (group, directory, filter); only untagged runs."""
    idx = {}
    for group, d, kind in dirs:
        for fn in sorted(glob.glob(osp.join(d, "*.json"))):
            try:
                J = json.load(open(fn))
            except (OSError, ValueError):
                continue
            meta = J.get("meta", {})
            if "run_id" not in meta or not J.get("traj"):
                continue
            m = canon_method(meta["method"])
            if methods is not None and m not in methods:
                continue
            if kind == "deep_rerun":
                if meta["run_id"] != f"{meta['method']}_{meta['dataset']}_fold{meta['fold']}" or m not in OLD4:
                    continue
            elif not untagged(meta):
                continue
            key = (group, meta["run_id"])
            if key not in idx:
                idx[key] = run_info(fn, group)
    return idx


def chance(row, run, action):
    rate = 100.0 * run["n_anom"] / run["n_test"] if run and run["n_test"] else np.nan
    row.update(auroc=50.0, auprc=rate, fpr95=CHANCE_FPR95, nonfinite_selected=1, a15_action=action)
    return row


def set_point(row, run, j, action):
    row.update(auroc=100 * run["auroc"][j], auprc=100 * run["auprc"][j], fpr95=100 * run["fpr95"][j],
               epoch_selected=int(run["epochs"][j]), a15_action=action)
    return row


def ubgold_select(run, use_ef):
    """UB-GOLD fit() replayed with A15 rule 1: only finite points can improve; patience 30."""
    max_auc, best, counter = 0.0, None, 0
    for i, e in enumerate(run["epochs"]):
        if use_ef and e % run["eval_freq"] != 0:
            continue
        a = run["sel"][i]
        if run["test_ok"][i] and np.isfinite(a) and a > max_auc:
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE:
            break
    return best


def valnormal_select(run):
    """A3 + A15 rule 2: minimum finite held-out-normal mean; ties -> earliest."""
    h = np.where(run["hold_ok"], run["hold"], np.inf)
    return None if not run["hold_ok"].any() else int(np.argmin(h))


def point_index(run, epoch):
    if run is None or epoch is None or not finite(epoch) or int(epoch) < 0:
        return None
    w = np.flatnonzero(run["epochs"] == int(epoch))
    return int(w[0]) if len(w) else None


def a15_audit(rows, runs):
    """rows: list of dict protocol rows (main). Applies A15 rules 1-4 in place; returns list of group re-selections
    (method, dataset, hp) needed for common-epoch own rules."""
    regroup = set()
    for r in rows:
        r.setdefault("nonfinite_selected", 0)
        r.setdefault("a15_action", "")
        run = runs.get(r.get("_runkey"))
        if run is None:
            continue
        b, m = r["base"], r["method"]
        i = point_index(run, r.get("epoch_selected"))
        row_bad = not finite(r.get("auroc"))
        if not run["test_ok"].any():                       # rule 4
            chance(r, run, "A15.4_all_points_nonfinite")
            continue
        if b in ("oracle", "oracle_ubgold"):
            glass_own = (m == "GLASS" and b == "oracle")
            bad = row_bad or (not glass_own and (i is None or not run["test_ok"][i]))
            if not bad:
                continue
            if b == "oracle_ubgold" or m in UBGOLD_RULE_METHODS:   # rule 1, UB-GOLD rule
                j = ubgold_select(run, m in EF_FILTER)
                if j is None:
                    chance(r, run, "A15.1_no_finite_point")
                else:
                    set_point(r, run, j, "A15.1_reselected")
            elif m in COMMON_EPOCH_OWN:
                regroup.add((m, r["dataset"], r["hp"]))
            else:
                chance(r, run, "A15.1_unresolved_own_rule")      # listed for a coordinator ruling
        elif b == "last":
            if row_bad or i is None or not run["test_ok"][i]:  # rule 3
                chance(r, run, "A15.3_last_nonfinite")
        elif b == "valnormal":
            act = ""
            if i is None or not run["hold_ok"][i]:             # rule 2
                j = valnormal_select(run)
                if j is None:
                    chance(r, run, "A15.2_no_finite_holdout")
                    continue
                set_point(r, run, j, "A15.2_reselected")
                i, act = j, "A15.2_reselected+"
            if not run["test_ok"][i] or not finite(r.get("auroc")):
                chance(r, run, act + "A15.3_selected_point_test_nonfinite")
    return regroup


def common_epoch_reselect(rows, runs, groups):
    """A15 rule 1 for own rules that pick one epoch for all 5 folds: max 5-fold mean test AUROC over epochs that are
    finite in every fold (first max)."""
    for (m, ds, hp) in groups:
        G = [r for r in rows if r["method"] == m and r["dataset"] == ds and r["hp"] == hp and r["base"] == "oracle"]
        R = [runs.get(r.get("_runkey")) for r in G]
        if not G or any(x is None for x in R):
            for r, run in zip(G, R):
                chance(r, run, "A15.1_unresolved_own_rule")
            continue
        common = set(R[0]["epochs"].tolist())
        for run in R[1:]:
            common &= set(run["epochs"].tolist())
        best, best_e = -np.inf, None
        for e in sorted(common):
            if m in EF_FILTER and e % R[0]["eval_freq"] != 0:
                continue
            idx = [point_index(run, e) for run in R]
            if all(run["test_ok"][i] for run, i in zip(R, idx)):
                v = float(np.mean([run["auroc"][i] for run, i in zip(R, idx)]))
                if v > best:
                    best, best_e = v, e
        for r, run in zip(G, R):
            if best_e is None:
                chance(r, run, "A15.1_no_epoch_finite_in_all_folds")
            else:
                set_point(r, run, point_index(run, best_e), "A15.1_reselected_common_epoch")


def synth_missing(rows, runs, ingested_groups):
    """Protocol rows an adapter did not write for an existing run (e.g. no finite point) are derived here with the
    A15 rules, so that a run is never silently missing from a protocol."""
    have = {(r["method"], r["dataset"], r["fold"], r["hp"], r["base"]) for r in rows}
    new = []
    for key, run in runs.items():
        m, ds, f, hp = run["method"], run["dataset"], run["fold"], run["hp"]
        if (m, ds, hp) not in ingested_groups or m in CHECKPOINT_FREE:
            continue
        need = ["last", "valnormal", "oracle_ubgold"] + (["oracle"] if m in UBGOLD_RULE_METHODS else [])
        for b in need:
            if (m, ds, f, hp, b) in have:
                continue
            r = dict(method=m, dataset=ds, fold=f, hp=hp, base=b, protocol=("hpfree_" if hp == "hpfree" else "") + b,
                     run_id=run["run_id"], source="stats_v2_A15_synth", _runkey=key, epoch_selected=-1,
                     auroc=np.nan, auprc=np.nan, fpr95=np.nan, oracle_rule="ubgold_patience30" if "oracle" in b else "")
            if not run["test_ok"].any():
                chance(r, run, "A15.4_all_points_nonfinite")
            elif b == "last":
                j = int(np.flatnonzero(run["epochs"] == run["epochs"].max())[0])
                set_point(r, run, j, "synth")
                if not run["test_ok"][j]:
                    chance(r, run, "A15.3_last_nonfinite")
            elif b == "valnormal":
                j = valnormal_select(run)
                if j is None:
                    chance(r, run, "A15.2_no_finite_holdout")
                else:
                    set_point(r, run, j, "synth")
                    if not run["test_ok"][j]:
                        chance(r, run, "A15.3_selected_point_test_nonfinite")
            else:
                j = ubgold_select(run, m in EF_FILTER)
                if j is None:
                    chance(r, run, "A15.1_no_finite_point")
                else:
                    set_point(r, run, j, "synth")
            r["a15_action"] = "synth:" + r.get("a15_action", "")
            new.append(r)
    return new


def tie_flags(rows, runs):
    """A21 §3: distinct finite test scores at the selected point < max(10, 5% of the test-set size)."""
    by_run = {}
    for r in rows:
        r.update(n_distinct=np.nan, tie_threshold=np.nan, tie_collapsed=np.nan, tie_source="unavailable")
        run = runs.get(r.get("_runkey"))
        if run is not None:
            by_run.setdefault(r["_runkey"], []).append(r)
    for key, rs in by_run.items():
        run = runs[key]
        p = run.get("npz")
        if not p or not osp.exists(p):
            continue
        try:
            z = np.load(p, allow_pickle=False)
            ep = z["epochs"] if "epochs" in z.files else run["epochs"]
            T = z["test"] if "test" in z.files else None
        except (OSError, ValueError, KeyError):
            continue
        thr = max(10.0, 0.05 * run["n_test"])
        for r in rs:
            v = None
            if r["method"] == "GLASS" and r["base"] == "oracle" and r.get("scorer"):
                k = "own_ref__" + str(r["scorer"])
                v = z[k] if k in z.files else None
            elif T is not None and finite(r.get("epoch_selected")) and int(r["epoch_selected"]) >= 0:
                w = np.flatnonzero(np.asarray(ep) == int(r["epoch_selected"]))
                v = T[w[0]] if len(w) else None
            if v is None:
                continue
            v = np.asarray(v, dtype=float)
            nd = int(np.unique(v[np.isfinite(v)]).size)
            r.update(n_distinct=nd, tie_threshold=thr, tie_collapsed=int(nd < thr), tie_source="npz")


# ----------------------------------------------------------------------------------------------
# ingestion
# ----------------------------------------------------------------------------------------------
ROWCOLS = ["method", "method_label", "stratum", "dataset", "fold", "hp", "protocol", "base", "auroc", "auprc", "fpr95",
           "epoch_selected", "run_id", "source", "oracle_rule", "hp_label", "transductive", "nonfinite_selected",
           "a15_action", "n_distinct", "tie_threshold", "tie_collapsed", "tie_source", "scorer", "hp_alias"]


def records(df, method=None, source=None, run_group=None, scale=1.0):
    out = []
    for r in df.to_dict("records"):
        m = canon_method(method or r["method"])
        p = str(r["protocol"])
        d = dict(method=m, dataset=DS_ALIASES.get(r["dataset"], r["dataset"]), fold=int(r["fold"]), hp=hp_of(p),
                 base=base_of(p), protocol=p, auroc=scale * float(r["auroc"]) if finite(r.get("auroc")) else np.nan,
                 auprc=scale * float(r["auprc"]) if finite(r.get("auprc")) else np.nan,
                 fpr95=scale * float(r["fpr95"]) if finite(r.get("fpr95")) else np.nan,
                 epoch_selected=r.get("epoch_selected", np.nan), run_id=r.get("run_id", ""),
                 source=source or r.get("source", ""), hp_config=r.get("hp_config", ""),
                 adapter_oracle_rule=r.get("oracle_rule", r.get("rule", "")), scorer=r.get("scorer", ""),
                 hp_alias=int(r["hp_alias"]) if finite(r.get("hp_alias")) else 0,
                 adapter_nonfinite=int(r["nonfinite_selected"]) if finite(r.get("nonfinite_selected")) else 0)
        if run_group:
            d["_runkey"] = (run_group, d["run_id"])
        out.append(d)
    return out


def load_frozen12(args, notes, prov):
    p = args.eval_folds or osp.join(args.res, "eval_folds.csv")
    prov["eval_folds.csv"] = sha256_file(p)
    if not osp.exists(p):
        notes.append(f"eval_folds.csv: not found at {p} (frozen-12 rows missing)")
        return [], [], []
    raw = pd.read_csv(p)
    raw["method"] = raw["method"].map(canon_method)
    repro = raw[raw.get("orientation", pd.Series("", index=raw.index)).astype(str) == "ubgold"]
    raw = raw[~raw.index.isin(repro.index)]
    main = []
    for r in records(raw):
        r["_runkey"] = ("deep_rerun" if r["source"] == "deep_rerun" else "harness", r["run_id"])
        main.append(r)
    # MUTAG of the old 4 (A10 §5): collect.py skips it; derived with the harness's own run_rows
    have = {(r["method"], r["dataset"], r["fold"], r["hp"]) for r in main}
    hc = load_module(osp.join(HARNESS, "collect.py"), "_gs_harness_collect")
    prov["harness/collect.py"] = sha256_file(osp.join(HARNESS, "collect.py"))
    n_mut = 0
    for fn in sorted(glob.glob(osp.join(args.dr_parts, "*_MUTAG_fold*.json"))):
        J = json.load(open(fn))
        meta = J["meta"]
        if meta["method"] not in OLD4 or meta["run_id"] != f"{meta['method']}_MUTAG_fold{meta['fold']}":
            continue
        if (meta["method"], "MUTAG", int(meta["fold"]), "ubgold") in have:
            continue
        meta.setdefault("hp", "ubgold")
        rs, _ = hc.run_rows(J, "deep_rerun_mutag")
        for r in records(pd.DataFrame(rs)):
            r["_runkey"] = ("deep_rerun", r["run_id"])
            main.append(r)
            n_mut += 1
    extra = []
    px = osp.join(osp.dirname(p), "eval_folds_extra.csv")
    if osp.exists(px):
        ex = pd.read_csv(px)
        ex["method"] = ex["method"].map(canon_method)
        ex = ex[ex.get("orientation", pd.Series("", index=ex.index)).astype(str) != "ubgold"]
        extra = records(ex)
    # the frozen-12 ORACLE is the UB-GOLD rule: oracle_ubgold is the same row
    for r in [r for r in main if r["base"] == "oracle"]:
        c = dict(r, base="oracle_ubgold", protocol=r["protocol"].replace("oracle", "oracle_ubgold"))
        main.append(c)
    notes.append(f"eval_folds.csv: {len(raw)} main rows (GraphDE UB-GOLD-orientation rows: {len(repro)} -> "
                 f"reproduction only, A5); MUTAG rows of the old 4 from deep_rerun parts: {n_mut}; extra rows {len(extra)}")
    return main, extra, records(repro) if len(repro) else []


def read_csv_safe(p):
    try:
        return pd.read_csv(p)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def run_adapter(method, cfg, args, tmp, prov):
    script = osp.join(EXT, cfg["script"])
    prov["ext/" + cfg["script"]] = sha256_file(script)
    mod = load_module(script, "_gs_adapter_" + re.sub(r"\W", "_", method))
    mod.PARTS = args.ext_parts
    # reproduction diagnostics that these adapters compute from other files and that the stats never consume
    if method == "MUSE":
        mod.r1_results = lambda: pd.DataFrame()           # R1 job-log parsing
    if method == "SUL-GUL":
        mod.NATIVE_OUT = osp.join(tmp, "_native_not_used")  # native-protocol diagnostic
    folds, extras = pd.DataFrame(), []
    if cfg["mode"] == "main":
        out = osp.join(tmp, re.sub(r"\W", "_", method))
        os.makedirs(out, exist_ok=True)
        setattr(mod, cfg["outattr"], out)
        with quiet():
            mod.main()
        folds = read_csv_safe(osp.join(out, cfg["folds"]))
        extras = [read_csv_safe(osp.join(out, e)) for e in cfg["extra"]]
    elif cfg["mode"] == "rows_of":
        rows = []
        for fn in sorted(glob.glob(osp.join(args.ext_parts, f"{method}_*.json"))):
            J = json.load(open(fn))
            if untagged(J["meta"]):
                rows += mod.rows_of(J)
        folds = pd.DataFrame(rows)
    elif cfg["mode"] == "protoglad":
        runs = mod.load_parts(args.ext_parts)
        rows = []
        for (ds, hp, fold), J in sorted(runs.items()):
            m, r = J["meta"], J["traj"][0]
            pre = "" if hp == "ubgold" else "hpfree_"
            for proto in ("oracle", "last", "valnormal"):        # verbatim from ProtoGLAD/derive.py main()
                rows.append(dict(method="ProtoGLAD", dataset=ds, fold=fold, protocol=pre + proto,
                                 auroc=100 * r["auroc"], auprc=100 * r["auprc"], fpr95=100 * r["fpr95"],
                                 epoch_selected=1, hp_config=("ubgold|" if hp == "ubgold" else "hpfree|") + m["ubgold_sh_line"],
                                 source="ext_protoglad", run_id=m["run_id"], hp_meaning=m.get("hp_meaning", "")))
        folds = pd.DataFrame(rows)
        ex = []                                                  # A20 §1 secondary row oracle_own_gridbest
        mod.GRID = args.protoglad_grid
        G = mod.grid_df("ubgold") if osp.isdir(args.protoglad_grid) else pd.DataFrame()
        for ds in mod.DATASETS:
            gb = osp.join(mod.GRID_RES, f"{ds}_gridbest.json")
            if not osp.exists(gb) or not len(G):
                continue
            bp = json.load(open(gb))["best_params"]
            g = G[G.dataset == ds]
            sel = g[(g.h == bp["h"]) & (g.psi1 == bp["psi1"]) & (g.psi2 == bp["psi2"]) & (g.tau == bp["tau"])]
            for _, x in sel.sort_values("fold").iterrows():
                ex.append(dict(method="ProtoGLAD", dataset=ds, fold=int(x.fold), protocol="oracle_own_gridbest",
                               auroc=100 * x.auroc, auprc=100 * x.auprc, fpr95=100 * x.fpr95, run_id=""))
        extras = [pd.DataFrame(ex)]
    return folds, extras


EXT_RENAME = {"LGKDE": {"oracle_own": "oracle"}}


def load_ext(args, notes, prov, methods):
    main, extra, failures = [], [], []
    tmp = tempfile.mkdtemp(prefix="glad_stats_v2_", dir=args.tmp)
    try:
        for method in E5 + E4:
            if method not in methods:
                continue
            try:
                folds, extras = run_adapter(method, ADAPTERS[method], args, tmp, prov)
            except Exception as e:  # noqa: BLE001 - reported, fatal unless --allow-adapter-failure
                failures.append(f"{method}: {type(e).__name__}: {e}")
                if not args.allow_adapter_failure:
                    raise
                continue
            n0 = len(main)
            for frame, is_extra in [(folds, False)] + [(x, True) for x in extras]:
                if frame is None or not len(frame) or "protocol" not in frame:
                    continue
                for r in records(frame, method=method, run_group="ext"):
                    b = EXT_RENAME.get(method, {}).get(r["base"], r["base"])
                    r["base"] = b
                    r["protocol"] = ("hpfree_" if r["hp"] == "hpfree" else "") + b
                    if not is_extra and b in ("oracle", "oracle_ubgold", "last", "valnormal"):
                        main.append(r)
                        if b == "oracle" and (method in UBGOLD_RULE_METHODS or method == "ProtoGLAD"):
                            main.append(dict(r, base="oracle_ubgold",
                                             protocol=r["protocol"].replace("oracle", "oracle_ubgold")))
                    else:
                        extra.append(r)
            notes.append(f"{method}: adapter {ADAPTERS[method]['script']} -> {len(main) - n0} protocol rows")
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return main, extra, failures


def load_kernels(args, notes, prov):
    p = osp.join(args.res, "eval_kernels_folds.csv")
    prov["eval_kernels_folds.csv"] = sha256_file(p)
    if not osp.exists(p):
        notes.append("eval_kernels_folds.csv: not present")
        return [], None
    kr = pd.read_csv(p)
    raw = kr.copy()
    kr = kr[~kr["method"].astype(str).str.contains("@")]   # @testfit = UB-GOLD transductive path, not ranked
    s = scale100(kr)
    out = []
    for proto, hp in (("tuned", "ubgold"), ("default", "hpfree")):
        sub = kr[kr.protocol == proto]
        for b in BASES:
            t = sub.copy()
            t["protocol"] = ("hpfree_" if hp == "hpfree" else "") + b
            t["source"] = "eval_kernels"
            out += records(t, scale=s)
    notes.append(f"eval_kernels_folds.csv: {len(kr)} train-fit rows (KernelGLAD 'tuned' -> ubgold protocols, "
                 f"'default' -> hpfree protocols; @testfit rows not ranked)")
    return out, raw


def load_references(args, notes, prov):
    p = osp.join(args.res, "grail0_folds.csv")
    prov["grail0_folds.csv"] = sha256_file(p)
    if not osp.exists(p):
        notes.append("grail0_folds.csv: not present")
        return []
    g0 = pd.read_csv(p)
    g0 = g0[g0.method.isin(REF_MAP)].copy()
    g0["method"] = g0["method"].map(REF_MAP)
    s = scale100(g0)
    out = []
    for proto in PROTOCOLS:
        t = g0.copy()
        t["protocol"] = proto
        t["source"] = "grail0"
        out += records(t, scale=s)
    notes.append(f"grail0_folds.csv: references {sorted(g0.method.unique())} (a priori; identical in every protocol)")
    return out


def finalize(rows):
    df = pd.DataFrame(rows)
    if not len(df):
        return pd.DataFrame(columns=ROWCOLS)
    df["stratum"] = df["method"].map(stratum_of)
    df["method_label"] = df["method"].map(lambda m: METHOD_LABEL.get(m, m))
    aor = df.get("adapter_oracle_rule", pd.Series("", index=df.index)).fillna("").astype(str)
    df["oracle_rule"] = [ORACLE_RULE.get(m, "") if b == "oracle" else
                         (ORACLE_RULE.get(m, "") if m in CHECKPOINT_FREE else "ubgold_patience30")
                         if b == "oracle_ubgold" else (a if str(b).startswith("oracle") else "")
                         for m, b, a in zip(df.method, df.base, aor)]
    df["hp_label"] = [hp_label(m, d, h, c) for m, d, h, c in
                      zip(df.method, df.dataset, df.hp, df.get("hp_config", pd.Series("", index=df.index)))]
    df["transductive"] = False      # A24 §1: GLASS's default rule is inductive; no ranked row is transductive
    for c in ROWCOLS:
        if c not in df:
            df[c] = np.nan
    df["fold"] = df["fold"].astype(int)
    return df


# ----------------------------------------------------------------------------------------------
# A12 registry, coverage, flags
# ----------------------------------------------------------------------------------------------
DIV_RE = re.compile(r"^\s*#\s*DIVERGED\b[^:]*:\s*(\S+)\s+(\S+)\s+(\d+)\s+(ubgold|hpfree)\b")


def load_diverged(args, notes):
    rows = []
    p = args.diverged_registry
    if p and osp.exists(p):
        d = pd.read_csv(p)
        for r in d.to_dict("records"):
            rows.append(dict(method=canon_method(r["method"]), dataset=r["dataset"], fold=int(r["fold"]), hp=r["hp"],
                             source=osp.basename(p), evidence=r.get("evidence", ""), amendment=r.get("amendment", "")))
    pats = [osp.join(args.harness_dir, "jobs", "tasks*.txt"), osp.join(args.ext_dir, "*", "jobs", "tasks*.txt"),
            osp.join(args.ext_dir, "*", "tasks*.txt")]
    for pat in pats:
        for fn in sorted(glob.glob(pat)):
            if ".bak" in osp.basename(fn):
                continue
            for line in open(fn, errors="replace"):
                mm = DIV_RE.match(line)
                if mm:
                    rows.append(dict(method=canon_method(mm.group(1)), dataset=mm.group(2), fold=int(mm.group(3)),
                                     hp=mm.group(4), source=osp.relpath(fn, PRELIM), evidence=line.strip(), amendment="A12"))
    df = pd.DataFrame(rows, columns=["method", "dataset", "fold", "hp", "source", "evidence", "amendment"])
    if len(df):
        df = df.groupby(["method", "dataset", "fold", "hp"], as_index=False).agg(
            source=("source", lambda s: "|".join(sorted(set(s)))), evidence=("evidence", "first"),
            amendment=("amendment", "first"))
    notes.append(f"A12 DIVERGED registry: {len(df)} runs ({p} + '# DIVERGED' lines in task files)")
    return df


def coverage_table(long, diverged, methods):
    div = {}
    for r in diverged.itertuples():
        div.setdefault((r.method, r.dataset, r.hp), set()).add(int(r.fold))
    rows = []
    for m in methods:
        st = stratum_of(m)
        hps = ["ubgold", "hpfree"] if m not in REFERENCES else ["ubgold"]
        sub = long[long.method == m]
        for ds in ALL_DS:
            for hp in hps:
                s = sub[(sub.dataset == ds) & (sub.hp == hp)]
                need = ["oracle", "last", "valnormal"]
                per = {b: set(s[s.base == b].fold) for b in need}
                done = set.intersection(*per.values()) if per else set()
                any_ = set(s.fold)
                dv = div.get((m, ds, hp), set())
                nff = sorted(s[s.nonfinite_selected == 1].fold.unique()) if len(s) else []
                tcf = sorted(s[s.tie_collapsed == 1].fold.unique()) if len(s) else []
                nf, tc = len(nff), len(tcf)
                st_lbl = (f"{len(done)}/5" + (f" + DIVERGED {len(dv)}" if dv else "") +
                          (f" (partial protocols: {sorted(any_ - done)})" if any_ - done else "") +
                          (f" [nonfinite_selected: f{','.join(map(str, nff))}]" if nff else "") +
                          (f" [tie_collapsed: f{','.join(map(str, tcf))}]" if tcf else ""))
                if not done and not dv and not any_:
                    st_lbl = "missing"
                rows.append(dict(method=m, method_label=METHOD_LABEL.get(m, m), stratum=st, dataset=ds,
                                 set="UB9" if ds in UB9 else "secondary", hp=hp, folds_done=len(done),
                                 folds_diverged=len(dv), diverged_folds="|".join(map(str, sorted(dv))),
                                 folds_missing=NFOLD - len(done) - len(dv), runs_nonfinite_selected=nf,
                                 nonfinite_selected_folds="|".join(map(str, nff)), runs_tie_collapsed=tc,
                                 tie_collapsed_folds="|".join(map(str, tcf)), status=st_lbl,
                                 resolved=bool(len(done) + len(dv) >= NFOLD)))
    return pd.DataFrame(rows)


def valnormal_degenerate(long, runs):
    """A14 §3, per method x hp setting: >= 80% of runs select a point in the first 10% of training AND mean
    VAL-NORMAL AUROC >= 5 below LAST (same runs). Scope 'all' = every dataset; 'UB9' reported alongside."""
    out = []
    for (m, hp), g in long[long.base.isin(["valnormal", "last"])].groupby(["method", "hp"]):
        if m in CHECKPOINT_FREE:
            continue
        for scope, dss in (("all", ALL_DS), ("UB9", UB9)):
            v = g[(g.base == "valnormal") & g.dataset.isin(dss)].set_index(["dataset", "fold"])
            l_ = g[(g.base == "last") & g.dataset.isin(dss)].set_index(["dataset", "fold"])
            common = v.index.intersection(l_.index)
            if not len(common):
                continue
            v, l_ = v.loc[common], l_.loc[common]
            early = []
            for (_, _), r in v.iterrows():
                run = runs.get(r.get("_runkey")) if isinstance(r.get("_runkey"), tuple) else None
                ne = run["num_epoch"] if run else np.nan
                e = r["epoch_selected"]
                if finite(e) and e >= 0 and finite(ne):
                    early.append(float(e) <= 0.1 * float(ne))
            fe = float(np.mean(early)) if early else np.nan
            gap = float(l_.auroc.mean() - v.auroc.mean())
            out.append(dict(method=m, hp=hp, scope=scope, runs=len(common), runs_with_epoch=len(early),
                            frac_valnormal_first10pct=fe, mean_last_minus_valnormal=gap,
                            valnormal_degenerate=bool(finite(fe) and fe >= 0.8 and gap >= 5.0)))
    return pd.DataFrame(out)


def identical_config(runs, methods):
    """A21 §2: per (dataset, fold), do the 'ubgold' and 'hpfree' runs resolve to identical options? Also records
    bit-identity of the pair (train_loss + trajectory; A24 §2)."""
    by = {}
    for run in runs.values():
        if run["method"] in methods:
            by.setdefault((run["method"], run["dataset"], run["fold"]), {})[run["hp"]] = run
    rows = []
    for (m, ds, f), d in sorted(by.items()):
        if "ubgold" in d and "hpfree" in d:
            a, b = d["ubgold"], d["hpfree"]
            keys = set(a["hparams"]) | set(b["hparams"])
            same = all(str(a["hparams"].get(k)) == str(b["hparams"].get(k)) for k in keys)
            rows.append(dict(method=m, dataset=ds, fold=f, hparams_identical=same,
                             bit_identical=bool(same and a["digest"] == b["digest"]),
                             differing_options="|".join(sorted(k for k in keys if str(a["hparams"].get(k)) !=
                                                               str(b["hparams"].get(k))))[:300]))
    det = pd.DataFrame(rows, columns=["method", "dataset", "fold", "hparams_identical", "bit_identical",
                                      "differing_options"])
    summ = []
    for m in methods:
        g = det[det.method == m]
        auto = bool(len(g) and g.hparams_identical.all())
        summ.append(dict(method=m, pairs_compared=len(g), pairs_hparams_identical=int(g.hparams_identical.sum()),
                         auto_identical_config=auto, declared=IDENTICAL_CONFIG_DECLARED.get(m, ""),
                         excluded_from_pooled_hp_leak=bool(auto or m in IDENTICAL_CONFIG_DECLARED)))
    return det, pd.DataFrame(summ)


def noise_estimate(long, det, ident_methods):
    """A23 §3 / A24 §2 for identical-config methods: count of bit-identical ubgold/hpfree pairs and |dAUROC| of the
    differing pairs at LAST and VAL-NORMAL. Label: lower bound on seed-to-seed variation."""
    rows = []
    for m in ident_methods:
        g = det[det.method == m]
        if not len(g):
            continue
        diff_pairs = g[~g.bit_identical]
        r = dict(method=m, label="lower bound on seed-to-seed variation (seed, split and initialisation fixed)",
                 pairs=len(g), bit_identical=int(g.bit_identical.sum()), differing=len(diff_pairs))
        for b in ("last", "valnormal"):
            s = long[(long.method == m) & (long.base == b)]
            u = s[s.hp == "ubgold"].set_index(["dataset", "fold"]).auroc
            h = s[(s.hp == "hpfree") & (s.hp_alias.fillna(0) == 0)].set_index(["dataset", "fold"]).auroc
            keys = [(d, f) for d, f in zip(diff_pairs.dataset, diff_pairs.fold) if (d, f) in u.index and (d, f) in h.index]
            x = np.abs(np.array([u[k] - h[k] for k in keys], dtype=float))
            r[f"{b}_n"] = len(x)
            for q, nm in ((None, "mean"), (0, "min"), (50, "median"), (90, "p90"), (100, "max")):
                r[f"{b}_absdiff_{nm}"] = (float(np.mean(x)) if q is None else float(np.percentile(x, q))) if len(x) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# arrays, bootstrap, rank statistics
# ----------------------------------------------------------------------------------------------
class Cubes:
    """(method, protocol, metric) -> (len(ALL_DS), 5) arrays; NaN where missing."""

    def __init__(self, long):
        self.di = {d: i for i, d in enumerate(ALL_DS)}
        self.arr = {}
        for (m, p), g in long.groupby(["method", "protocol"]):
            for k in METRICS:
                A = np.full((len(ALL_DS), NFOLD), np.nan)
                for d, f, v in zip(g.dataset, g.fold, g[k]):
                    if d in self.di and 0 <= int(f) < NFOLD:
                        A[self.di[d], int(f)] = v
                self.arr[(m, p, k)] = A
        self.methods = sorted(long.method.unique()) if len(long) else []

    def get(self, methods, dss, proto, k="auroc"):
        idx = [self.di[d] for d in dss]
        empty = np.full((len(ALL_DS), NFOLD), np.nan)
        return np.stack([self.arr.get((m, proto, k), empty)[idx] for m in methods]) if methods else \
            np.zeros((0, len(dss), NFOLD))

    def complete(self, m, dss, protos):
        return all(np.isfinite(self.get([m], dss, p)).all() for p in protos)

    def nfolds(self, m, dss, p):
        return np.isfinite(self.get([m], dss, p)[0]).sum(-1)


class Boot:
    def __init__(self, n_ds, B, seed):
        rng = np.random.default_rng(seed)
        self.B = B
        self.fold_idx = rng.integers(0, NFOLD, size=(B, n_ds, NFOLD))
        self.ds_idx = rng.integers(0, n_ds, size=(B, n_ds))
        self.D = n_ds

    def fold(self, A):
        g = A[:, np.arange(self.D)[None, :, None], self.fold_idx]
        return g.mean(-1).mean(-1)

    def ds(self, A):
        return A.mean(-1)[:, self.ds_idx].mean(-1)

    def ds_means(self, dm):
        """dm: (M, D) dataset means (may contain NaN for dropped datasets) -> (M, B)."""
        return np.nanmean(dm[:, self.ds_idx], -1)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def tau_b(x, y):
    i, j = np.triu_indices(x.shape[-1], 1)
    sx = np.sign(x[..., i] - x[..., j])
    sy = np.sign(y[..., i] - y[..., j])
    num = (sx * sy).sum(-1)
    den = np.sqrt((sx != 0).sum(-1) * (sy != 0).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def top_k(means, names, k=3):
    order = np.argsort(-np.asarray(means), kind="stable")
    return [names[i] for i in order[:k]]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def cliques(names, ranks, cd):
    order = np.argsort(ranks, kind="stable")
    r = np.asarray(ranks)[order]
    groups = []
    for i in range(len(r)):
        j = i
        while j + 1 < len(r) and r[j + 1] - r[i] <= cd + 1e-12:
            j += 1
        if j > i and not any(g[0] <= i and j <= g[1] for g in groups):
            groups.append((i, j))
    return [[names[order[k]] for k in range(i, j + 1)] for i, j in groups]


def pool_methods(C, pool, present):
    return [m for m in present if stratum_of(m) in POOLS[pool]]


# ----------------------------------------------------------------------------------------------
# analyses
# ----------------------------------------------------------------------------------------------
def leaderboards(C, long, B, seed):
    rows, perds = [], []
    present = sorted(long.method.unique())
    for sname, dss in SETS.items():
        boot = Boot(len(dss), B, seed)
        for proto in PROTOCOLS:
            ms_all = [m for m in present if np.isfinite(C.get([m], dss, proto)).any()]
            if not ms_all:
                continue
            A = {k: C.get(ms_all, dss, proto, k) for k in METRICS}
            for pool in POOLS:
                ms = pool_methods(C, pool, ms_all)
                if not ms:
                    continue
                cm = [m for m in ms if C.complete(m, dss, [proto])]
                rank, fb, db = {}, None, None
                if cm:
                    Ac = C.get(cm, dss, proto)
                    fb, db = boot.fold(Ac), boot.ds(Ac)
                    rk = stats.rankdata(-Ac.mean(-1).mean(-1), method="min")
                    rank = {m: int(r) for m, r in zip(cm, rk)}
                for m in ms:
                    i = ms_all.index(m)
                    dm = np.nanmean(A["auroc"][i], -1) if np.isfinite(A["auroc"][i]).any() else np.nan
                    r = dict(set=sname, pool=pool, protocol=proto, method=m, method_label=METHOD_LABEL.get(m, m),
                             stratum=stratum_of(m), oracle_rule=ORACLE_RULE.get(m, "") if "oracle" in proto else "",
                             n_datasets=int((C.nfolds(m, dss, proto) > 0).sum()), n_datasets_total=len(dss),
                             complete=m in cm, mean_auroc=float(np.nanmean(dm)),
                             mean_auprc=float(np.nanmean(np.nanmean(A["auprc"][i], -1))),
                             mean_fpr95=float(np.nanmean(np.nanmean(A["fpr95"][i], -1))))
                    if m in cm:
                        j = cm.index(m)
                        r.update(rank_in_pool=rank[m], fold_ci_lo=ci(fb[j])[0], fold_ci_hi=ci(fb[j])[1],
                                 ds_ci_lo=ci(db[j])[0], ds_ci_hi=ci(db[j])[1])
                    rows.append(r)
            for i, m in enumerate(ms_all):
                for k, d in enumerate(dss):
                    v = A["auroc"][i, k]
                    if np.isfinite(v).any():
                        perds.append(dict(set=sname, protocol=proto, method=m, stratum=stratum_of(m), dataset=d,
                                          n_folds=int(np.isfinite(v).sum()), auroc_mean=float(np.nanmean(v)),
                                          auroc_std_ddof0=float(np.nanstd(v)),
                                          auprc_mean=float(np.nanmean(A["auprc"][i, k])),
                                          fpr95_mean=float(np.nanmean(A["fpr95"][i, k]))))
    return pd.DataFrame(rows), pd.DataFrame(perds)


# (label, a, b, axis): axis 'checkpoint' (A20 exclusion) or 'hp' (A21 exclusion) or 'total'/'other'
COMPARISONS = [
    ("checkpoint: ORACLE - VAL-NORMAL", "oracle", "valnormal", "checkpoint"),
    ("checkpoint: ORACLE - LAST", "oracle", "last", "checkpoint"),
    ("checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL", "oracle_ubgold", "valnormal", "checkpoint"),
    ("checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST", "oracle_ubgold", "last", "checkpoint"),
    ("checkpoint, HP-FREE: ORACLE - VAL-NORMAL", "hpfree_oracle", "hpfree_valnormal", "checkpoint"),
    ("checkpoint, HP-FREE: ORACLE - LAST", "hpfree_oracle", "hpfree_last", "checkpoint"),
    ("HP leak: VAL-NORMAL - HP-FREE", "valnormal", "hpfree_valnormal", "hp"),
    ("HP leak at LAST: LAST - HP-FREE LAST", "last", "hpfree_last", "hp"),
    ("total: ORACLE - HP-FREE", "oracle", "hpfree_valnormal", "total"),
    ("A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj)", "oracle", "valnormal_obj", "checkpoint"),
    ("early stop: ORACLE - max over run", "oracle", "oracle_nostop", "other"),
]


def method_axes(m, ident_excl):
    return dict(checkpoint_free=m in CHECKPOINT_FREE, identical_config=m in ident_excl, no_hp=m in NO_HP_AXIS)


def excluded(m, axis, ident_excl):
    if axis == "checkpoint" and m in CHECKPOINT_FREE:
        return "A20: no checkpoint axis (0 by construction)"
    if axis == "hp" and m in ident_excl:
        return "A21: identical ubgold/hpfree configuration (0 by construction)"
    if axis == "hp" and m in NO_HP_AXIS:
        return "a-priori reference: no hyperparameters"
    if axis == "total" and (m in CHECKPOINT_FREE and (m in ident_excl or m in NO_HP_AXIS)):
        return "no checkpoint and no HP axis"
    return ""


def inflation(C, Cx, long, B, seed, ident_excl):
    """Per-method differences with paired fold/dataset bootstrap CIs (complete methods only), and pooled means over
    methods per pool with the A20/A21 exclusions."""
    rows, pooled = [], []
    present = sorted(long.method.unique())
    for sname, dss in SETS.items():
        boot = Boot(len(dss), B, seed)
        per = {}
        for label, a, b, axis in COMPARISONS:
            for m in present:
                X = Cx if b in ("valnormal_obj", "oracle_nostop") else C
                Aa, Ab = C.get([m], dss, a), X.get([m], dss, b)
                if not (np.isfinite(Aa).all() and np.isfinite(Ab).all()):
                    continue
                D = Aa - Ab
                dm = D[0].mean(-1)
                fb, db = boot.fold(D)[0], boot.ds(D)[0]
                ex = excluded(m, axis, ident_excl)
                rows.append(dict(set=sname, method=m, method_label=METHOD_LABEL.get(m, m), stratum=stratum_of(m),
                                 comparison=label, a=a, b=b, axis=axis,
                                 oracle_rule=ORACLE_RULE.get(m, "") if "oracle" in a else "",
                                 mean_diff=float(dm.mean()), fold_ci_lo=ci(fb)[0], fold_ci_hi=ci(fb)[1],
                                 ds_ci_lo=ci(db)[0], ds_ci_hi=ci(db)[1], n_datasets=len(dss),
                                 n_datasets_diff_ge2=int((dm >= 2.0).sum()), max_dataset_diff=float(dm.max()),
                                 max_fold_diff=float(np.nanmax(D)), excluded_from_pooled=ex))
                per[(label, m)] = (D, ex)
        for label, a, b, axis in COMPARISONS:
            for pool in POOLS:
                ms = [m for m in present if stratum_of(m) in POOLS[pool] and (label, m) in per]
                inc = [m for m in ms if not per[(label, m)][1]]
                exc = [m for m in ms if per[(label, m)][1]]
                if not inc:
                    continue
                D = np.concatenate([per[(label, m)][0] for m in inc])
                v = D.mean(-1).mean(-1)
                pooled.append(dict(set=sname, pool=pool, comparison=label, axis=axis, n_methods=len(inc),
                                   mean_over_methods=float(v.mean()), fold_ci_lo=ci(boot.fold(D).mean(0))[0],
                                   fold_ci_hi=ci(boot.fold(D).mean(0))[1], ds_ci_lo=ci(boot.ds(D).mean(0))[0],
                                   ds_ci_hi=ci(boot.ds(D).mean(0))[1], n_methods_ge2=int((v >= 2.0).sum()),
                                   methods="|".join(inc), excluded="|".join(f"{m} ({per[(label, m)][1]})" for m in exc)))
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def hp_leak_table(infl, cov, ident_summary, ident_excl, methods):
    """Per method: HP axis status and VAL-NORMAL - HP-FREE on UB9 (complete methods), with the exclusion reason."""
    rows = []
    s = infl[(infl.set == "UB9") & (infl.comparison == "HP leak: VAL-NORMAL - HP-FREE")].set_index("method") \
        if len(infl) else pd.DataFrame()
    for m in methods:
        axis = ("no hyperparameters (a-priori reference)" if m in NO_HP_AXIS else
                "no HP axis: identical ubgold/hpfree configuration (A21; 'no per-dataset configuration released')"
                if m in ident_excl else "tuned vs label-free HP")
        r = dict(method=m, method_label=METHOD_LABEL.get(m, m), stratum=stratum_of(m), hp_axis=axis,
                 tuned_label=hp_label(m, "AIDS", "ubgold"), hpfree_label=hp_label(m, "AIDS", "hpfree"),
                 excluded_from_pooled=excluded(m, "hp", ident_excl))
        if len(s) and m in s.index:
            x = s.loc[m]
            r.update(ub9_mean_hp_leak=x.mean_diff, fold_ci_lo=x.fold_ci_lo, fold_ci_hi=x.fold_ci_hi,
                     ds_ci_lo=x.ds_ci_lo, ds_ci_hi=x.ds_ci_hi)
        g = ident_summary[ident_summary.method == m] if len(ident_summary) else pd.DataFrame()
        if len(g):
            r.update(pairs_compared=int(g.pairs_compared.iloc[0]),
                     pairs_hparams_identical=int(g.pairs_hparams_identical.iloc[0]))
        rows.append(r)
    return pd.DataFrame(rows)


KENDALL_PAIRS = [("oracle", "valnormal"), ("oracle", "last"), ("valnormal", "last"), ("oracle_ubgold", "valnormal"),
                 ("oracle", "hpfree_valnormal"), ("valnormal", "hpfree_valnormal"), ("hpfree_oracle", "hpfree_valnormal")]


def kendall(C, long, B, seed):
    rows = []
    present = sorted(long.method.unique())
    for sname, dss in SETS.items():
        boot = Boot(len(dss), B, seed)
        for a, b in KENDALL_PAIRS:
            for pool in POOLS:
                ms = [m for m in pool_methods(C, pool, present) if C.complete(m, dss, [a, b])]
                if len(ms) < 3:
                    continue
                Aa, Ab = C.get(ms, dss, a), C.get(ms, dss, b)
                if np.allclose(Aa, Ab, equal_nan=True):
                    continue
                ma, mb = Aa.mean(-1).mean(-1), Ab.mean(-1).mean(-1)
                fa, fbb = boot.fold(Aa).T, boot.fold(Ab).T
                da, dbb = boot.ds(Aa).T, boot.ds(Ab).T
                ta, tb = top_k(ma, ms), top_k(mb, ms)
                k = min(3, len(ms))
                rows.append(dict(set=sname, pool=pool, protocol_a=a, protocol_b=b, n_methods=len(ms),
                                 tau_b=float(tau_b(ma, mb)), fold_ci_lo=ci(tau_b(fa, fbb))[0],
                                 fold_ci_hi=ci(tau_b(fa, fbb))[1], ds_ci_lo=ci(tau_b(da, dbb))[0],
                                 ds_ci_hi=ci(tau_b(da, dbb))[1], top3_a="|".join(ta), top3_b="|".join(tb),
                                 top3_changed=set(ta) != set(tb),
                                 p_top3_changed_foldboot=float(np.mean([set(np.argsort(-fa[i])[:k]) != set(np.argsort(-fbb[i])[:k]) for i in range(len(fa))])),
                                 p_top3_changed_dsboot=float(np.mean([set(np.argsort(-da[i])[:k]) != set(np.argsort(-dbb[i])[:k]) for i in range(len(da))])),
                                 methods="|".join(ms)))
    return pd.DataFrame(rows)


def friedman(C, long, alpha=0.05):
    frows, prows, crows = [], [], []
    present = sorted(long.method.unique())
    for sname, dss in SETS.items():
        for proto in PROTOCOLS:
            for pool in POOLS:
                ms = [m for m in pool_methods(C, pool, present) if C.complete(m, dss, [proto])]
                k, N = len(ms), len(dss)
                if k < 3:
                    continue
                DM = C.get(ms, dss, proto).mean(-1).T
                if np.allclose(DM, DM[:, :1]):
                    continue
                R = np.vstack([stats.rankdata(-row) for row in DM])
                avg = R.mean(0)
                chi2, p = stats.friedmanchisquare(*[DM[:, j] for j in range(k)])
                ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) - chi2 > 0 else np.inf
                pff = float(stats.f.sf(ff, k - 1, (k - 1) * (N - 1))) if np.isfinite(ff) else 0.0
                q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
                cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
                frows.append(dict(set=sname, pool=pool, protocol=proto, n_methods=k, n_datasets=N,
                                  friedman_chi2=float(chi2), friedman_p=float(p), iman_davenport_F=float(ff),
                                  iman_davenport_p=pff, nemenyi_CD=float(cd)))
                pr = []
                for i, j in combinations(range(k), 2):
                    d = DM[:, i] - DM[:, j]
                    try:
                        pw = float(stats.wilcoxon(DM[:, i], DM[:, j]).pvalue) if np.any(d != 0) else 1.0
                    except ValueError:
                        pw = 1.0
                    pr.append(dict(set=sname, pool=pool, protocol=proto, method_a=ms[i], method_b=ms[j],
                                   mean_a=float(DM[:, i].mean()), mean_b=float(DM[:, j].mean()),
                                   datasets_a_wins=int((d > 0).sum()), datasets_b_wins=int((d < 0).sum()),
                                   wilcoxon_p=pw, avg_rank_a=float(avg[i]), avg_rank_b=float(avg[j]),
                                   nemenyi_sig=bool(abs(avg[i] - avg[j]) > cd)))
                adj = holm([x["wilcoxon_p"] for x in pr])
                for x, a_ in zip(pr, adj):
                    x["wilcoxon_p_holm"] = float(a_)
                    x["holm_sig_0.05"] = bool(a_ < alpha)
                prows += pr
                letters = {m: "" for m in ms}
                for gi, g in enumerate(cliques(ms, avg, cd)):
                    for m in g:
                        letters[m] += chr(ord("a") + gi) if gi < 26 else "*"
                for j in np.argsort(avg, kind="stable"):
                    crows.append(dict(set=sname, pool=pool, protocol=proto, method=ms[j], avg_rank=float(avg[j]),
                                      mean_auroc=float(DM[:, j].mean()), nemenyi_CD=float(cd),
                                      cd_groups=letters[ms[j]] or "-",
                                      not_sig_diff_from_best=bool(avg[j] - avg.min() <= cd)))
    return pd.DataFrame(frows), pd.DataFrame(prows), pd.DataFrame(crows)


M1_CRIT = ["oracle", "valnormal", "last", "pseudo", "pseudo_ensemble", "ckpt_ensemble", "consensus", "aucp",
           "oracle_nostop"]


def selectors(args, B, seed, notes):
    """M1 label-free selectors on the old 4 (unchanged from v1; descriptive)."""
    p = osp.join(args.res, "m1_selection_folds.csv")
    if not osp.exists(p):
        return pd.DataFrame()
    m1 = pd.read_csv(p).rename(columns={"criterion": "protocol"})
    m1["method"] = m1["method"].map(canon_method)
    s = scale100(m1)
    for k in METRICS:
        m1[k] = pd.to_numeric(m1[k], errors="coerce") * s
    m1["hp"], m1["base"] = "ubgold", m1["protocol"]
    C = Cubes(m1)
    rows, per_method = [], {}
    for m in OLD4:
        protos = set(m1[m1.method == m].protocol)
        dss = [d for d in UB9 if all(np.isfinite(C.get([m], [d], c)).all() for c in M1_CRIT if c in protos)]
        per_method[m] = dss
        for c in M1_CRIT:
            if c not in protos or not dss:
                continue
            A, Av, Ao = (C.get([m], dss, x)[0] for x in (c, "valnormal", "oracle"))
            gap = (Ao - Av).mean(-1).mean()
            gain = (A - Av).mean(-1)
            rows.append(dict(method=m, criterion=c, n_datasets=len(dss), mean_auroc=float(A.mean(-1).mean()),
                             minus_valnormal=float(gain.mean()),
                             recovered_fraction_of_oracle_gap=float(gain.mean() / gap) if gap else np.nan))
    df = pd.DataFrame(rows)
    for c in M1_CRIT:
        sub = df[df.criterion == c] if len(df) else df
        if len(sub) == 4:
            rows.append(dict(method="POOLED(4)", criterion=c, n_datasets=-1, mean_auroc=float(sub.mean_auroc.mean()),
                             minus_valnormal=float(sub.minus_valnormal.mean())))
    notes.append(f"m1_selection_folds.csv: {len(m1)} rows (old-4 label-free selectors; descriptive)")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# PREREG gates (frozen 12) with A12/A14/A15/A21
# ----------------------------------------------------------------------------------------------
def status(met, resolved, final):
    if met is None:
        return "FAIL" if final else "PENDING"
    if final or resolved:
        return "PASS" if met else "FAIL"
    return "PENDING"


def resolved_all(cov, methods, hp):
    if not len(cov):
        return False
    c = cov[cov.method.isin(methods) & (cov.set == "UB9") & (cov.hp == hp)]
    return bool(len(c) == len(methods) * len(UB9) and c.resolved.all())


SEL_NAME = {"valnormal": "VAL-NORMAL", "last": "LAST"}


def g1_selector(C, cov, selector, B, seed, final, methods=DEEP12):
    dss = UB9
    comp = [m for m in methods if C.complete(m, dss, ["oracle", selector])]
    res = resolved_all(cov, methods, "ubgold")
    out = dict(selector=selector, comparison=f"ORACLE - {SEL_NAME[selector]}", complete_methods=comp,
               n_complete=len(comp), n_required=G1_RULE["n_methods_min"], all_methods_resolved=res)
    if not comp:
        out["status"] = status(None, res, final)
        return out
    D = C.get(comp, dss, "oracle") - C.get(comp, dss, selector)
    per = D.mean(-1).mean(-1)
    boot = Boot(len(dss), B, seed)
    cond = {"n_complete>=8": len(comp) >= G1_RULE["n_methods_min"],
            "mean>=3.0": bool(per.mean() >= G1_RULE["mean_min"]),
            "frac(>=2.0)>=0.75": bool((per >= G1_RULE["per_method_min"]).mean() >= G1_RULE["frac_min"])}
    out.update(inflation_per_method={m: float(v) for m, v in zip(comp, per)}, mean_inflation=float(per.mean()),
               mean_fold_ci=ci(boot.fold(D).mean(0)), mean_ds_ci=ci(boot.ds(D).mean(0)),
               n_methods_ge2=int((per >= 2.0).sum()), frac_methods_ge2=float((per >= 2.0).mean()),
               conditions=cond, currently_met=bool(all(cond.values())))
    out["status"] = status(all(cond.values()), res, final)
    return out


def g1_combined(vn, la):
    """A14 §2: PASS only if it passes under BOTH selectors; one-of-two = SELECTOR-DEPENDENT (not met for GO)."""
    a, b = vn["status"], la["status"]
    if a == "PASS" and b == "PASS":
        return "PASS"
    if {a, b} == {"PASS", "FAIL"}:
        return "SELECTOR-DEPENDENT"
    if a == "FAIL" and b == "FAIL":
        return "FAIL"
    if "PENDING" in (a, b):
        if "FAIL" in (a, b):    # cannot become PASS any more
            return "PENDING (cannot PASS: one selector FAIL)"
        return "PENDING"
    return "FAIL"


def g2a(C, B, seed, final, cov, methods=DEEP12):
    comp = [m for m in methods if C.complete(m, UB9, ["oracle", "valnormal"])]
    res = resolved_all(cov, methods, "ubgold")
    out = dict(pool="frozen-12 methods with complete UB9 ORACLE and VAL-NORMAL", methods=comp)
    if len(comp) < 2:
        out["status"] = status(None, res, final)
        return out
    mo = C.get(comp, UB9, "oracle").mean(-1).mean(-1)
    mv = C.get(comp, UB9, "valnormal").mean(-1).mean(-1)
    t = float(tau_b(mo, mv))
    to, tv = top_k(mo, comp), top_k(mv, comp)
    met = (t <= 0.8) or (set(to) != set(tv))
    out.update(tau_b=t, top3_oracle=to, top3_valnormal=tv, top3_changed=set(to) != set(tv), currently_met=bool(met))
    out["status"] = status(met, res, final)
    return out


def g2b(C, B, seed, final, cov, ident_excl, methods=DEEP12):
    comp = [m for m in methods if C.complete(m, UB9, ["valnormal", "hpfree_valnormal"])]
    excl = [m for m in comp if m in ident_excl]
    use = [m for m in comp if m not in ident_excl]
    res = resolved_all(cov, methods, "ubgold") and resolved_all(cov, methods, "hpfree")
    out = dict(pool="frozen-12 methods with complete UB9 VAL-NORMAL and HP-FREE, minus A21 identical-config",
               methods=use, excluded_A21=excl, n_complete=len(use))
    if not use:
        out["status"] = status(None, res, final)
        return out
    D = C.get(use, UB9, "valnormal") - C.get(use, UB9, "hpfree_valnormal")
    per = D.mean(-1).mean(-1)
    boot = Boot(len(UB9), B, seed)
    met = bool(per.mean() >= 2.0)
    out.update(hp_leak_per_method={m: float(v) for m, v in zip(use, per)}, mean_hp_leak=float(per.mean()),
               mean_fold_ci=ci(boot.fold(D).mean(0)), mean_ds_ci=ci(boot.ds(D).mean(0)), currently_met=met)
    out["status"] = status(met, res, final)
    return out


def sensitivity_drop(long, B, seed, ident_excl):
    """A15 §5 / A21 §3: drop flagged runs (all protocols of a flagged (method, dataset, fold, hp) run); dataset means
    over the remaining folds; a method needs >= 1 fold per dataset. Reports G1 (both selectors), G2(b) and per-method
    checkpoint inflation."""
    rows = []
    runkey = list(zip(long.method, long.dataset, long.fold, long.hp))
    nf = long.groupby(["method", "dataset", "fold", "hp"]).nonfinite_selected.max()
    tc = long.groupby(["method", "dataset", "fold", "hp"]).tie_collapsed.max()
    variants = {"drop nonfinite_selected": set(nf[nf == 1].index),
                "drop tie_collapsed": set(tc[tc == 1].index)}
    variants["drop any flag"] = variants["drop nonfinite_selected"] | variants["drop tie_collapsed"]
    for vname, flagged in variants.items():
        keep = np.array([k not in flagged for k in runkey])
        L = long[keep]
        C = Cubes(L)
        boot = Boot(len(UB9), B, seed)
        for a, b, lab in (("oracle", "valnormal", "G1/VAL-NORMAL"), ("oracle", "last", "G1/LAST"),
                          ("valnormal", "hpfree_valnormal", "G2b HP leak")):
            pool = [m for m in DEEP12 if not (lab.startswith("G2b") and m in ident_excl)]
            per = {}
            for m in pool:
                D = C.get([m], UB9, a)[0] - C.get([m], UB9, b)[0]
                dmean = np.array([np.nanmean(x) if np.isfinite(x).any() else np.nan for x in D])
                if np.isfinite(dmean).all():
                    per[m] = dmean
            if not per:
                continue
            ms = sorted(per)
            dm = np.vstack([per[m] for m in ms])
            v = dm.mean(-1)
            r = dict(variant=vname, n_runs_dropped=len(flagged), quantity=lab, n_methods=len(ms),
                     mean_over_methods=float(v.mean()), ds_ci_lo=ci(boot.ds_means(dm).mean(0))[0],
                     ds_ci_hi=ci(boot.ds_means(dm).mean(0))[1], n_methods_ge2=int((v >= 2.0).sum()),
                     per_method="; ".join(f"{m} {x:.2f}" for m, x in zip(ms, v)))
            if lab.startswith("G1"):
                r["g1_conditions_met"] = bool(len(ms) >= 8 and v.mean() >= 3.0 and (v >= 2.0).mean() >= 0.75)
            rows.append(r)
        # every stratum: per-method checkpoint inflation after dropping
        for m in sorted(L.method.unique()):
            if m in CHECKPOINT_FREE:
                continue
            for a, b in (("oracle", "valnormal"), ("oracle", "last")):
                D = C.get([m], UB9, a)[0] - C.get([m], UB9, b)[0]
                dmean = np.array([np.nanmean(x) if np.isfinite(x).any() else np.nan for x in D])
                if np.isfinite(dmean).all():
                    rows.append(dict(variant=vname, n_runs_dropped=len(flagged), quantity=f"per-method {a}-{b}",
                                     method=m, stratum=stratum_of(m), n_methods=1, mean_over_methods=float(dmean.mean())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# A9: confirmatory typed test (G2(c)) and the H4s scope test
# ----------------------------------------------------------------------------------------------
def load_typed(args, notes, prov):
    files = sorted(glob.glob(osp.join(args.res, "eval_typed*.csv")))
    frames = []
    for p in files:
        prov[osp.basename(p)] = sha256_file(p)
        t = pd.read_csv(p)
        if "criterion" not in t and "protocol" in t:
            t = t.rename(columns={"protocol": "criterion"})
        miss = {"method", "dataset", "anomaly_type", "seed", "criterion", "auroc"} - set(t.columns)
        if miss:
            raise SystemExit(f"{p}: missing required typed columns {sorted(miss)} (FROZEN.md)")
        t = t.copy()
        t["auroc"] = pd.to_numeric(t["auroc"], errors="coerce") * scale100(t)
        t["src"] = osp.basename(p)
        frames.append(t)
    if not frames:
        notes.append("eval_typed*.csv: none yet (A9 typed runs PENDING)")
        return None
    t = pd.concat(frames, ignore_index=True)
    t["method"] = t["method"].map(canon_method)
    t["dataset"] = t["dataset"].astype(str).str.upper().replace({"MOLSWAP": "MS"})
    if "size_variant" not in t:
        t["size_variant"] = np.where(t.dataset == "GX", "matched", np.nan)
    t["size_variant"] = t["size_variant"].fillna(pd.Series(np.where(t.dataset == "GX", "matched", ""), index=t.index))
    if "feature_variant" not in t:
        t["feature_variant"] = np.where(t.dataset == "GX", "F2", "MS")
    if "orientation" in t:
        t = t[t.orientation.astype(str) != "ubgold"]     # A5: GraphDE paper orientation only
    if "n_anomalies" not in t:
        t["n_anomalies"] = np.nan
    t["seed"] = t["seed"].astype(int)
    t["size_variant"] = t["size_variant"].astype(str).replace(SIZE_VARIANT_ALIASES)
    bad = t[(t.dataset == "MS") & ~t.size_variant.isin(["matched", "changing"])]
    if len(bad):
        raise SystemExit("eval_typed: MS rows need size_variant matched|changing (FROZEN.md)")
    key = ["method", "dataset", "anomaly_type", "size_variant", "feature_variant", "seed", "criterion"]
    dup = t.duplicated(key, keep=False)
    if dup.any():
        raise SystemExit(f"eval_typed: duplicate rows for key {key}: {t[dup][key].head(5).values.tolist()}")
    notes.append(f"eval_typed*.csv: {len(t)} rows from {', '.join(osp.basename(f) for f in files)}")
    return t


def load_m2_gx_controls(args):
    """GX controls already computed in M2 (same generator, seeds 0-4): used when eval_typed has none for GX."""
    out = []
    p = osp.join(args.res, "m2_typed_runs.csv")
    if osp.exists(p):
        d = pd.read_csv(p)
        d = d[(d.dataset == "GX") & d.method.isin(CTRL_ALIASES["gc_lw"] | CTRL_ALIASES["size_lw"])]
        d = d[d.feature_variant.astype(str) == "F2"]
        out.append(d.assign(src="m2_typed_runs.csv"))
    p = osp.join(args.res, "m2_posthoc_controls.csv")
    if osp.exists(p):
        d = pd.read_csv(p)
        d = d[(d.dataset == "GX") & d.method.isin(CTRL_ALIASES["elp_lw"])]
        out.append(d.assign(src="m2_posthoc_controls.csv", criterion="training-free", feature_variant="F2"))
    if not out:
        return None
    d = pd.concat(out, ignore_index=True)
    d["auroc"] = pd.to_numeric(d["auroc"], errors="coerce") * scale100(d)
    d["size_variant"] = "matched"
    d["n_anomalies"] = np.nan
    d["seed"] = d["seed"].astype(int)
    return d


def ctrl_kind(m):
    for k, al in CTRL_ALIASES.items():
        if m in al:
            return k
    return None


def control_means(t):
    """mean AUROC over seeds per (generator, type, size_variant, control); controls are protocol-invariant."""
    t = t.copy()
    t["ctrl"] = t.method.map(ctrl_kind)
    t = t[t.ctrl.notna()]
    rows = []
    for (g, a, v, c), s in t.groupby(["dataset", "anomaly_type", "size_variant", "ctrl"]):
        crit = s.criterion.astype(str)
        pick = "training-free" if (crit == "training-free").any() else ("valnormal" if (crit == "valnormal").any()
                                                                       else crit.iloc[0])
        s = s[crit == pick]
        rows.append(dict(dataset=g, anomaly_type=a, size_variant=v, ctrl=c, mean=float(s.auroc.mean()),
                         n_seeds=int(s.seed.nunique()), src="|".join(sorted(set(s.src)))))
    return pd.DataFrame(rows, columns=["dataset", "anomaly_type", "size_variant", "ctrl", "mean", "n_seeds", "src"])


def load_ms_manifest(path):
    """A9.1: per-seed anomaly yields of the frozen MolSwap generator {(type, variant): [counts]}. Only
    counts_per_seed is read; the manifest's 'status'/'min_yield' fields reflect the superseded 150 floor (A25)."""
    if not path or not osp.exists(path):
        return {}
    m = json.load(open(path))
    out = {}
    for a, d in (m.get("status") or {}).items():
        for v, x in d.items():
            out[(a, SIZE_VARIANT_ALIASES.get(v, v))] = [int(c) for c in x.get("counts_per_seed", [])]
    return out


def feasible_from_counts(counts):
    """A9 §2 + A25: infeasible if any seed yields < 100 anomalies; None while fewer than 5 seeds are known and no seed
    is below the floor yet."""
    if not counts:
        return None
    if any(c < A9["min_anomalies"] for c in counts):
        return False
    return True if len(counts) >= NFOLD else None


def type_table(t, m2ctrl, manifest=None):
    """A9 §2 feasibility, §4 validity (controls only) and confounding, per (generator, type, size variant)."""
    manifest = manifest or {}
    ct = control_means(t) if t is not None else pd.DataFrame()
    if m2ctrl is not None:
        m2 = control_means(m2ctrl)
        have = set(zip(ct.dataset, ct.anomaly_type, ct.size_variant, ct.ctrl)) if len(ct) else set()
        m2 = m2[[k not in have for k in zip(m2.dataset, m2.anomaly_type, m2.size_variant, m2.ctrl)]]
        ct = pd.concat([ct, m2], ignore_index=True)
    rows = []
    for g, types in (("GX", GX_TYPES), ("MS", MS_TYPES)):
        for a in types:
            for v in (("matched",) if g == "GX" else ("matched", "changing")):
                c = ct[(ct.dataset == g) & (ct.anomaly_type == a) & (ct.size_variant == v)] if len(ct) else ct
                get = {k: (float(c[c.ctrl == k]["mean"].iloc[0]) if len(c) and (c.ctrl == k).any() else np.nan)
                       for k in CTRL_ALIASES}
                nan_ = {k: not finite(x) for k, x in get.items()}
                counts, csrc = [], ""
                if g == "MS" and (a, v) in manifest:
                    counts, csrc = manifest[(a, v)], "manifest"
                elif g == "MS" and t is not None:
                    na = t[(t.dataset == g) & (t.anomaly_type == a) & (t.size_variant == v)].groupby("seed").n_anomalies.max()
                    counts, csrc = [int(x) for x in na.dropna()], "eval_typed n_anomalies"
                min_anom = float(min(counts)) if counts else np.nan
                n_seed_anom = len(counts)
                feasible = True if g == "GX" else (feasible_from_counts(counts) if counts else None)
                valid = None if (nan_["gc_lw"] or nan_["size_lw"]) else bool(
                    get["gc_lw"] >= A9["valid_gc_min"] and get["size_lw"] <= A9["valid_size_max"])
                confounded = None if nan_["elp_lw"] else bool(get["elp_lw"] >= A9["confound_elp_min"])
                rows.append(dict(generator=g, anomaly_type=a, size_variant=v, graphlet_count_lw=get["gc_lw"],
                                 size_only=get["size_lw"], edge_label_pair_lw=get["elp_lw"], valid=valid,
                                 confounded=confounded, min_anomalies_per_seed=min_anom,
                                 seeds_with_anomaly_count=n_seed_anom, counts_per_seed="|".join(map(str, counts)),
                                 counts_src=csrc, feasibility_reading=A9_FEASIBILITY_READING if g == "MS" else "",
                                 feasible=feasible,
                                 controls_src="|".join(sorted(set(c.src)) if len(c) else [])))
    T = pd.DataFrame(rows)
    T["in_decision"] = [(v == "matched") and (va is True) and (co is False) and (fe is True)
                        for v, va, co, fe in zip(T.size_variant, T.valid, T.confounded, T.feasible)]
    T["undetermined"] = [any(x is None for x in (va, co, fe)) for va, co, fe in zip(T.valid, T.confounded, T.feasible)]
    return T


def typed_cells(t, methods, gen_types, fv_gx, B, seed):
    """VAL-NORMAL mean over seeds per (method, generator, type, size variant) with seed-bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in methods:
        for (g, a, v) in gen_types:
            s = t[(t.method == m) & (t.dataset == g) & (t.anomaly_type == a) & (t.size_variant == v) &
                  (t.criterion.astype(str) == "valnormal")] if t is not None else pd.DataFrame()
            if g == "GX" and len(s):
                s = s[s.feature_variant.astype(str) == fv_gx]
            if g == "MS" and len(s) and s.feature_variant.nunique() > 1:
                raise SystemExit(f"eval_typed: {m} {a} {v}: several MS feature variants {sorted(s.feature_variant.unique())}")
            x = s.sort_values("seed").auroc.to_numpy(float) if len(s) else np.array([])
            r = dict(method=m, stratum=stratum_of(m), generator=g, anomaly_type=a, size_variant=v,
                     feature_variant=fv_gx if g == "GX" else "MS", n_seeds=len(x),
                     valnormal_mean=float(x.mean()) if len(x) else np.nan, complete=len(x) == 5)
            if len(x):
                bs = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
                r.update(seed_ci_lo=ci(bs)[0], seed_ci_hi=ci(bs)[1], blind=bool(x.mean() <= A9["blind_max"]))
            else:
                r["blind"] = None
            rows.append(r)
    return pd.DataFrame(rows)


def a9_verdict(cells, T, methods, final):
    """A9 §5 on (valid, unconfounded, feasible, size-matched) pairs of `methods`. Precedence (conservative):
    INCONCLUSIVE (< 2 valid MS types) > CONFIRMED > REFUTED > PARTIAL. Missing cells: PENDING unless final,
    where they count as not blind."""
    D = T[T.in_decision]
    dec_types = list(zip(D.generator, D.anomaly_type))
    ms_types = [x for x in dec_types if x[0] == "MS"]
    out = dict(decision_types=[f"{g}:{a}" for g, a in dec_types], n_valid_ms_types=len(ms_types),
               undetermined_types=[f"{g}:{a}:{v}" for g, a, v, u in zip(T.generator, T.anomaly_type, T.size_variant,
                                                                     T.undetermined) if u])
    c = cells[cells.size_variant == "matched"].set_index(["method", "generator", "anomaly_type"]) if len(cells) else None

    def blind(m, g, a):
        if c is None or (m, g, a) not in c.index:
            return None
        r = c.loc[(m, g, a)]
        return None if r["n_seeds"] == 0 else bool(r["valnormal_mean"] <= A9["blind_max"])

    pairs = [(m, g, a, blind(m, g, a)) for m in methods for (g, a) in dec_types]
    missing = [p for p in pairs if p[3] is None]
    incomplete = [(m, g, a) for m in methods for (g, a) in dec_types
                  if c is None or (m, g, a) not in c.index or not bool(c.loc[(m, g, a)]["complete"])]
    val = [(m, g, a, bool(b)) for m, g, a, b in pairs if b is not None or final]
    pooled = [b for _, _, _, b in val]
    ms_b = [b for _, g, _, b in val if g == "MS"]
    frac_p = float(np.mean(pooled)) if pooled else np.nan
    frac_ms = float(np.mean(ms_b)) if ms_b else np.nan
    maj = {}
    for m in methods:
        bs = [b for mm, g, _, b in val if mm == m and g == "MS"]
        maj[m] = bool(len(ms_types) and sum(bs) > len(ms_types) / 2.0)
    n_maj = int(sum(maj.values()))
    cond_i = bool(finite(frac_p) and frac_p >= A9["pooled_blind_min"])
    cond_ii = bool(finite(frac_ms) and frac_ms >= A9["ms_blind_min"] and n_maj >= A9["ms_methods_min"])
    if len(ms_types) < A9["min_valid_ms"]:
        verdict = "INCONCLUSIVE"
    elif cond_i and cond_ii:
        verdict = "CONFIRMED"
    elif finite(frac_ms) and frac_ms < A9["refuted_ms_max"]:
        verdict = "REFUTED"
    else:
        verdict = "PARTIAL"
    out.update(n_pairs=len(pairs), n_pairs_missing=len(missing), n_cells_incomplete=len(incomplete),
               frac_blind_pooled=frac_p, frac_blind_ms=frac_ms, methods_blind_on_majority_of_ms=n_maj,
               per_method_majority_ms=maj, cond_i=cond_i, cond_ii=cond_ii, verdict=verdict,
               complete=bool(not incomplete and not (T.undetermined & (T.size_variant == "matched")).any()))
    return out


def h4s_scope(cells, T, methods):
    """A9 §6: blind rate size-matched vs size-changing MS variants, on the MS types valid (and feasible) in their
    size-matched variant whose size-changing variant is feasible."""
    D = T[(T.generator == "MS") & T.in_decision]
    chg = T[(T.generator == "MS") & (T.size_variant == "changing")].set_index("anomaly_type")
    types = [a for a in D.anomaly_type if a in chg.index and chg.loc[a, "feasible"] == True]  # noqa: E712 (None-safe)
    if not types or not len(cells):
        return dict(types=types, status="PENDING (no eligible MS type with both variants)")
    c = cells.set_index(["method", "generator", "anomaly_type", "size_variant"])
    rates = {}
    for v in ("matched", "changing"):
        bs = [bool(c.loc[(m, "MS", a, v)]["blind"]) for m in methods for a in types
              if (m, "MS", a, v) in c.index and c.loc[(m, "MS", a, v)]["n_seeds"] > 0]
        rates[v] = (float(np.mean(bs)) if bs else np.nan, len(bs))
    drop = 100.0 * (rates["matched"][0] - rates["changing"][0])
    return dict(types=types, blind_rate_matched=rates["matched"][0], n_matched=rates["matched"][1],
                blind_rate_changing=rates["changing"][0], n_changing=rates["changing"][1], drop_pp=drop,
                scope_to_size_preserving=bool(finite(drop) and drop >= A9["h4s_drop_pp"]),
                status="computed" if finite(drop) else "PENDING")


def g2c_descriptive_m2(args):
    """The PREREG 4-method G2(c) computation on M2 data (descriptive only, A9 §5)."""
    p = osp.join(args.res, "m2_typed_summary.csv")
    if not osp.exists(p):
        return None
    t = pd.read_csv(p)
    t["method"] = t["method"].map(canon_method)
    dm = t[(t.feature_variant == "F2") & (t.criterion == "valnormal") & (t.dataset == "GX")
           & t.anomaly_type.isin(GX_TYPES) & t.method.isin(DEEP12)]
    per = dm.pivot_table(index="method", columns="anomaly_type", values="auroc_mean", aggfunc="mean")
    if not set(GX_TYPES) <= set(per.columns):
        return None
    ok = (per[GX_TYPES] <= 60).all(axis=1)
    return dict(label="descriptive only (A9 §5): PREREG's post-hoc 4-method computation on M2",
                per_method={m: {k: float(per.loc[m, k]) for k in GX_TYPES} for m in per.index},
                frac_all_le60=float(ok.mean()))


def g2c_a9(args, B, seed, final, notes, prov):
    t = load_typed(args, notes, prov)
    m2c = load_m2_gx_controls(args)
    manifest = load_ms_manifest(args.ms_manifest)
    prov["molswap manifest.json"] = sha256_file(args.ms_manifest) if args.ms_manifest else None
    T = type_table(t, m2c, manifest)
    gen_types = [(g, a, v) for g, a, v in zip(T.generator, T.anomaly_type, T.size_variant)]
    others = OLD4 + E5 + E4
    cells = typed_cells(t, NEW8 + others, gen_types, "F2", B, seed)
    cells_f1 = typed_cells(t, NEW8, [x for x in gen_types if x[0] == "GX"], "F1", B, seed)
    out = dict(label="hypothesis generated post hoc; tested out of sample per A9",
               confirmatory_methods=NEW8, rule=A9, feasibility_reading=A9_FEASIBILITY_READING,
               feasibility_note="A9 §2 as amended by A25: infeasible iff any seed yields < 100 anomalies")
    if t is None or not (t.method.isin(NEW8)).any():
        out.update(status="PENDING", verdict="PENDING (no typed results of the 8 confirmatory methods)")
    v = a9_verdict(cells[cells.method.isin(NEW8)], T, NEW8, final)
    out["decision"] = v
    if t is not None and t.method.isin(NEW8).any():
        out["verdict"] = v["verdict"]
        if v["complete"] or final:
            out["status"] = "PASS" if v["verdict"] == "CONFIRMED" else "FAIL"
        else:
            out["status"] = "PENDING"
            out["verdict"] = f"PENDING (currently {v['verdict']} on the available cells)"
    # F1 sensitivity: GX cells with F1, MS unchanged
    if len(cells_f1) and cells_f1.n_seeds.sum():
        cf = pd.concat([cells_f1, cells[(cells.method.isin(NEW8)) & (cells.generator == "MS")]], ignore_index=True)
        out["sensitivity_F1"] = a9_verdict(cf, T, NEW8, final)
    out["discovery_methods_descriptive"] = {g: a9_verdict(cells[cells.method.isin(ms)], T, ms, True)
                                            for g, ms in (("old4 (discovery)", OLD4), ("E5", E5), ("E4", E4))
                                            if len(cells) and cells[cells.method.isin(ms)].n_seeds.sum()}
    out["H4s_scope"] = h4s_scope(cells[cells.method.isin(NEW8)], T, NEW8)
    out["descriptive_4method_m2"] = g2c_descriptive_m2(args)
    return out, T, cells


# ----------------------------------------------------------------------------------------------
# A17 version sensitivity
# ----------------------------------------------------------------------------------------------
def version_sensitivity(args, C, B, seed, notes, prov):
    p = osp.join(args.res, "eval_version_a96137b_folds.csv")
    prov[osp.basename(p)] = sha256_file(p)
    if not osp.exists(p):
        notes.append("A17: eval_version_a96137b_folds.csv not present (version-sensitivity runs PENDING)")
        return pd.DataFrame(), "PENDING (no a96137b runs yet)"
    v = pd.read_csv(p)
    v["method"] = v["method"].map(canon_method)
    if "source" in v:
        v = v[v.source.astype(str) == "evalpaper_harness"]    # collect.py would add deep_rerun rows otherwise
    if "ubgold_commit" in v:
        v = v[v.ubgold_commit.astype(str).str.startswith("a96137b")]
    v = v[v.method.isin(["GLocalKD", "GLADC"]) & v.protocol.isin(["oracle", "last", "valnormal"])]
    rows = records(v)
    for r in rows:
        r["protocol"] = "v_" + r["protocol"]
    V = Cubes(pd.DataFrame(rows)) if rows else None
    out = []
    boot = Boot(len(UB9), B, seed)
    for m in ["GLocalKD", "GLADC"]:
        for p_ in ["oracle", "last", "valnormal"]:
            if V is None:
                continue
            A1, A0 = V.get([m], UB9, "v_" + p_), C.get([m], UB9, p_)
            ok = np.isfinite(A1).all() and np.isfinite(A0).all()
            r = dict(method=m, quantity=f"{p_}: a96137b - c940457", complete=bool(ok),
                     n_paired_folds=int((np.isfinite(A1) & np.isfinite(A0)).sum()))
            if ok:
                D = A1 - A0
                r.update(mean_a96137b=float(A1.mean()), mean_c940457=float(A0.mean()), mean_diff=float(D.mean()), fold_ci_lo=ci(boot.fold(D)[0])[0], fold_ci_hi=ci(boot.fold(D)[0])[1],
                         ds_ci_lo=ci(boot.ds(D)[0])[0], ds_ci_hi=ci(boot.ds(D)[0])[1])
            out.append(r)
        for sel in ("valnormal", "last"):
            if V is None:
                continue
            I1 = V.get([m], UB9, "v_oracle") - V.get([m], UB9, "v_" + sel)
            I0 = C.get([m], UB9, "oracle") - C.get([m], UB9, sel)
            if np.isfinite(I1).all() and np.isfinite(I0).all():
                out.append(dict(method=m, quantity=f"inflation ORACLE-{sel.upper()}: a96137b vs c940457", complete=True,
                                inflation_a96137b=float(I1.mean()), inflation_c940457=float(I0.mean()),
                                mean_diff=float((I1 - I0).mean())))
    st = "computed" if out and all(r.get("complete") for r in out) else "PARTIAL/PENDING"
    notes.append(f"A17: {len(v)} a96137b rows (GLocalKD/GLADC, harness source only)")
    return pd.DataFrame(out), st


# ----------------------------------------------------------------------------------------------
# amendment status block
# ----------------------------------------------------------------------------------------------
def amendment_status(pr, long, flags_runs, cov, deg, ident_summary, noise, version_status, failures, diverged, repro):
    def n(base=None, col=None, val=1):
        s = long if base is None else long[long.base == base]
        return int((s[col] == val).sum())
    a15 = long[long.a15_action.astype(str) != ""]
    rows = [
        ("A0-A4, A6-A8", "harness-level definitions (HP-FREE, VAL-NORMAL signal, CVTGAD score, two-stage alignment, "
                         "environment, secondary datasets)", "applied upstream in harness/run_eval.py + collect.py; "
                         "stats consumes eval_folds.csv"),
        ("A5", "GraphDE paper orientation main", f"{len(repro)} UB-GOLD-orientation rows kept as reproduction only"),
        ("A9", "G2(c) out-of-sample typed test", f"G2(c) {pr['G2c']['status']}; verdict: {pr['G2c'].get('verdict', '-')}"),
        ("A10/A11", "E5 / E4 strata, MUTAG secondary", "rows per stratum: " + ", ".join(
            f"{s} {int((long.stratum == s).sum())}" for s in STRATA)),
        ("A12", "DIVERGED runs", f"{len(diverged)} registered; excluded from aggregates; coverage shows "
                                 f"{int(cov.folds_diverged.sum()) if len(cov) else 0} diverged folds"),
        ("A13/A16/A18/A19/A22", "per-method labels (MUSE, SEGO released-modal, LGKDE MUTAG.yaml, HC-GLAD memory rule, "
                                "SUL-GUL GUL only + 500 epochs)", "hp_label / method_label columns"),
        ("A14", "dual-selector G1; VAL-NORMAL-degenerate flag",
         f"G1 VAL-NORMAL {pr['G1']['valnormal']['status']}, LAST {pr['G1']['last']['status']} -> primary "
         f"{pr['G1']['primary']}; degenerate (scope all): " + (", ".join(
             f"{r.method}/{r.hp}" for r in deg[(deg.scope == 'all') & deg.valnormal_degenerate].itertuples()) or "none")
         if len(deg) else "-"),
        ("A15", "non-finite rules", f"{len(a15)} protocol rows changed by the audit; nonfinite_selected rows "
                                    f"{n(col='nonfinite_selected')}; runs flagged {int(flags_runs.nonfinite_selected.sum()) if len(flags_runs) else 0}"),
        ("A17", "version sensitivity GLocalKD/GLADC a96137b", version_status),
        ("A20", "checkpoint-free excluded from pooled checkpoint inflation", ", ".join(sorted(
            m for m in CHECKPOINT_FREE if m in set(long.method)))),
        ("A21", "identical-config excluded from pooled HP leak; tie_collapsed flag",
         "excluded: " + ", ".join(ident_summary[ident_summary.excluded_from_pooled_hp_leak].method) +
         f"; tie_collapsed runs {int((flags_runs.tie_collapsed == 1).sum()) if len(flags_runs) else 0}; rows without "
         f"per-graph scores: {int((long.tie_source == 'unavailable').sum())}" if len(ident_summary) else "-"),
        ("A23/A24", "GLASS: inductive default rule (A24 withdraws the transductive flag); ubgold==hpfree noise estimate",
         ("; ".join(f"{r.method}: {r.bit_identical}/{r.pairs} pairs bit-identical, {r.differing} differ" for r in noise.itertuples())
          if len(noise) else "no identical-config pairs") + "; transductive rows: " + str(int(long.transductive.sum()))),
        ("adapters", "E5/E4 adapter runs", "all ran" if not failures else "FAILED: " + "; ".join(failures)),
    ]
    return pd.DataFrame(rows, columns=["amendment", "rule", "state"])


# ----------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------
def f2(x, col=""):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "–"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (float, np.floating)):
        if col.endswith("_p") or col.endswith("p_holm"):
            return f"{x:.2g}" if x >= 1e-3 else f"{x:.1e}"
        return f"{x:.2f}"
    return str(x)


def md_table(df, cols, heads=None):
    heads = heads or cols
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f2(r.get(c), c) for c in cols) + " |")
    return "\n".join(out)


def write_report(path, ctx):
    a = ctx["args"]
    pr = ctx["prereg"]
    L = ["# Label-free GLAD re-evaluation: statistics report v2 (AUTO-GENERATED)", ""]
    L.append(f"Generated {pr['generated']} by `prelim/evalpaper/stats/glad_stats.py` {VERSION} (sha256 "
             f"`{ctx['prov'].get('glad_stats.py')}`; B = {a.B}, seed {a.seed}{', FINAL mode' if a.final else ''}). "
             "Do not edit by hand. AUROC × 100; a leaderboard value is the mean over the datasets of a set of the 5-fold "
             "means. CIs: 95% percentile intervals of paired fold- and dataset-bootstraps (fold CIs are optimistic).")
    if a.methods:
        L += ["", f"**DEVELOPMENT RUN restricted to: {a.methods}. Not a study result.**"]
    L += ["", "## 1. PREREG and amendment status", ""]
    g1 = pr["G1"]
    L.append(f"- **G1 primary (A14 dual selector): {g1['primary']}** — VAL-NORMAL: {g1['valnormal']['status']}, "
             f"LAST: {g1['last']['status']}. PASS needs both; one of two = SELECTOR-DEPENDENT (not met for GO).")
    for s in ("valnormal", "last"):
        d = g1[s]
        if "mean_inflation" in d:
            L.append(f"  - ORACLE − {SEL_NAME[s]}: complete {d['n_complete']}/12 (≥ 8), mean {d['mean_inflation']:.2f} "
                     f"(fold CI {d['mean_fold_ci'][0]:.2f}–{d['mean_fold_ci'][1]:.2f}; ds CI {d['mean_ds_ci'][0]:.2f}–"
                     f"{d['mean_ds_ci'][1]:.2f}; ≥ 3.0), methods ≥ 2.0: {d['n_methods_ge2']}/{d['n_complete']} (≥ 75%); "
                     f"conditions {d['conditions']}; per method: " +
                     ", ".join(f"{m} {v:.2f}" for m, v in d["inflation_per_method"].items()))
        else:
            L.append(f"  - ORACLE − {SEL_NAME[s]}: no complete method.")
    ga, gb, gc = pr["G2a"], pr["G2b"], pr["G2c"]
    L.append(f"- **G2(a): {ga['status']}** — " + (f"τ-b(ORACLE, VAL-NORMAL) = {ga['tau_b']:.2f} over {len(ga['methods'])} "
             f"methods (≤ 0.8); top-3 {ga['top3_oracle']} → {ga['top3_valnormal']} (changed {ga['top3_changed']})."
                                                  if "tau_b" in ga else "no data."))
    L.append(f"- **G2(b): {gb['status']}** — " + (f"mean VAL-NORMAL − HP-FREE = {gb['mean_hp_leak']:.2f} over "
             f"{gb['n_complete']} methods (≥ 2.0); A21 exclusions: {gb['excluded_A21'] or 'none'}."
                                                  if "mean_hp_leak" in gb else "no complete method."))
    dv = gc.get("decision", {})
    L.append(f"- **G2(c) per A9: {gc['status']}** ({gc['label']}) — verdict {gc.get('verdict')}; decision types "
             f"{dv.get('decision_types')}; valid MS types {dv.get('n_valid_ms_types')}; blind fraction pooled "
             f"{f2(dv.get('frac_blind_pooled'))}, MS {f2(dv.get('frac_blind_ms'))}; methods blind on a majority of MS "
             f"types {dv.get('methods_blind_on_majority_of_ms')}; H4s: {gc['H4s_scope']}.")
    L.append(f"- **G2 overall: {pr['G2']}; decision: {pr['GO']}.**")
    L += ["", md_table(ctx["amend"], ["amendment", "rule", "state"]), ""]
    L += ["## 2. Inputs and provenance", ""] + [f"- {n}" for n in ctx["notes"]]
    L += ["", "sha256 of code and inputs used:", ""] + [f"- `{k}`: `{v}`" for k, v in sorted(ctx["prov"].items())]
    L += ["", "## 3. Coverage (DIVERGED cells shown; A12)", ""]
    cov = ctx["cov"]
    if len(cov):
        w = cov.pivot_table(index=["stratum", "method", "hp"], columns="dataset", values="status", aggfunc="first")
        w = w[[d for d in ALL_DS if d in w.columns]]
        L.append("| stratum | method | hp | " + " | ".join(w.columns) + " |")
        L.append("|" + "---|" * (len(w.columns) + 3))
        for (s, m, h), r in w.iterrows():
            L.append(f"| {s} | {METHOD_LABEL.get(m, m)} | {h} | " + " | ".join(str(r[c]) for c in w.columns) + " |")
    L += ["", "## 4. Flags (A12, A14, A15, A21)", ""]
    fm = ctx["flags_method"]
    if len(fm):
        L.append(md_table(fm, list(fm.columns)))
    deg = ctx["deg"]
    if len(deg):
        L += ["", "A14 §3 VAL-NORMAL-degenerate (descriptive; not a gate):", "",
              md_table(deg, ["method", "hp", "scope", "runs", "frac_valnormal_first10pct", "mean_last_minus_valnormal",
                             "valnormal_degenerate"])]
    fl = ctx["flags_rows"]
    if len(fl):
        L += ["", f"Flagged protocol rows ({len(fl)}; full list in `eval_leaderboard_v2_flags.csv`):", "",
              md_table(fl.head(60), ["method", "dataset", "fold", "protocol", "a15_action", "nonfinite_selected",
                                     "n_distinct", "tie_threshold", "tie_collapsed"])]
    L += ["", "Sensitivity analysis (A15 §5, A21 §3): flagged runs dropped.", ""]
    sd = ctx["sens"]
    if len(sd):
        L.append(md_table(sd[sd.method.isna()] if "method" in sd else sd,
                          ["variant", "n_runs_dropped", "quantity", "n_methods", "mean_over_methods", "ds_ci_lo",
                           "ds_ci_hi", "n_methods_ge2", "g1_conditions_met"]))
    L += ["", "## 5. Leaderboards (per stratum and pooled)", ""]
    lb = ctx["lb"]
    for sname in SETS:
        for pool in ["frozen12", "E5", "E4", "recent(E5+E4)", "frozen12+E5+E4", "all"]:
            sub = lb[(lb.set == sname) & (lb.pool == pool)] if len(lb) else lb
            if not len(sub):
                continue
            protos = [p for p in PROTOCOLS if p in set(sub.protocol)]
            L += [f"### {sname} / {pool}", "", "| method | stratum | " + " | ".join(protos) + " |",
                  "|" + "---|" * (len(protos) + 2)]
            order = sub[sub.protocol == "valnormal"].sort_values("mean_auroc", ascending=False).method.tolist()
            order += [m for m in sorted(sub.method.unique()) if m not in order]
            for m in order:
                cells = []
                for p in protos:
                    e = sub[(sub.method == m) & (sub.protocol == p)]
                    if not len(e):
                        cells.append("–")
                        continue
                    e = e.iloc[0]
                    cells.append(f2(e.mean_auroc) + ("" if e.complete else "*") +
                                 (f" (#{int(e.rank_in_pool)})" if e.complete and finite(e.get("rank_in_pool")) else ""))
                L.append(f"| {METHOD_LABEL.get(m, m)} | {stratum_of(m)} | " + " | ".join(cells) + " |")
            L.append("")
    L.append("`*` = incomplete coverage of the set (not ranked). ORACLE = own test-label rule where the method has one "
             "(oracle_rule in the CSV), else UB-GOLD's rule; oracle_ubgold = UB-GOLD's rule for every iterative method.")
    L += ["", "## 6. Inflation per method (both selectors; A14) and pooled (A20 exclusions)", ""]
    inf = ctx["infl"]
    if len(inf):
        s = inf[(inf.set == "UB9") & (inf.axis == "checkpoint")]
        L.append(md_table(s.sort_values(["comparison", "stratum", "method"]),
                          ["method", "stratum", "comparison", "mean_diff", "fold_ci_lo", "fold_ci_hi", "ds_ci_lo",
                           "ds_ci_hi", "n_datasets_diff_ge2", "excluded_from_pooled"]))
    ip = ctx["infl_pooled"]
    if len(ip):
        L += ["", "Pooled over methods (UB9):", "",
              md_table(ip[ip.set == "UB9"], ["pool", "comparison", "n_methods", "mean_over_methods", "fold_ci_lo",
                                               "fold_ci_hi", "ds_ci_lo", "ds_ci_hi", "excluded"])]
    L += ["", "## 7. Hyperparameter leak (tuned − label-free) with A21 exclusions", ""]
    hp = ctx["hpleak"]
    if len(hp):
        L.append(md_table(hp, ["method_label", "stratum", "hp_axis", "ub9_mean_hp_leak", "fold_ci_lo", "fold_ci_hi",
                               "excluded_from_pooled"]))
    nz = ctx["noise"]
    if len(nz):
        L += ["", "Identical-configuration pairs (A23 §3, A24 §2): ubgold − hpfree differences are run-to-run noise, a "
                  "**lower bound on seed-to-seed variation**; counts computed from the grid at run time:", "",
              md_table(nz, list(nz.columns))]
    L += ["", "## 8. G2(c) per A9 (typed test)", ""]
    T = ctx["typed_types"]
    if len(T):
        L.append(md_table(T, ["generator", "anomaly_type", "size_variant", "graphlet_count_lw", "size_only",
                              "edge_label_pair_lw", "valid", "confounded", "feasible", "in_decision", "controls_src"]))
    tc = ctx["typed_cells"]
    if len(tc) and tc.n_seeds.sum():
        s = tc[tc.n_seeds > 0]
        L += ["", md_table(s, ["method", "stratum", "generator", "anomaly_type", "size_variant", "n_seeds",
                               "valnormal_mean", "seed_ci_lo", "seed_ci_hi", "blind"])]
    L += ["", "```", json.dumps({k: gc[k] for k in gc if k not in ("rule",)}, indent=1, default=str)[:6000], "```"]
    L += ["", "## 9. Version sensitivity (A17; never part of G1/G2)", ""]
    vs = ctx["version"]
    L.append(md_table(vs, list(vs.columns)) if len(vs) else f"{ctx['version_status']}.")
    L += ["", "## 10. Agreement between protocol leaderboards (Kendall τ-b)", ""]
    kd = ctx["kendall"]
    if len(kd):
        L.append(md_table(kd[kd.set == "UB9"], ["pool", "protocol_a", "protocol_b", "n_methods", "tau_b", "fold_ci_lo",
                                                 "fold_ci_hi", "ds_ci_lo", "ds_ci_hi", "top3_changed"]))
    L += ["", "## 11. Friedman / Iman–Davenport and Nemenyi CD (UB9)", ""]
    fr = ctx["friedman"]
    if len(fr):
        L.append(md_table(fr[fr.set == "UB9"], ["pool", "protocol", "n_methods", "friedman_chi2", "friedman_p",
                                                 "iman_davenport_p", "nemenyi_CD"]))
    L += ["", "## 12. Label-free selectors of M1 (old 4; descriptive)", ""]
    sel = ctx["selectors"]
    if len(sel):
        L.append(md_table(sel, ["method", "criterion", "n_datasets", "mean_auroc", "minus_valnormal",
                                "recovered_fraction_of_oracle_gap"]))
    L += ["", "## 13. KernelGLAD (UB-GOLD's WL/PK + IF/OCSVM)", ""] + kernel_lines(a, ctx.get("kernels_raw"))
    L += ["", "## 14. Output files", ""] + [f"- `{f}`" for f in ctx["outputs"]]
    if osp.exists(path):
        os.makedirs(osp.join(osp.dirname(path), "archive"), exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(osp.getmtime(path)))
        dst = osp.join(osp.dirname(path), "archive", f"REPORT_generated_{stamp}.md")
        if not osp.exists(dst):
            shutil.copy2(path, dst)
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")



def kernel_lines(args, kernels_raw):
    """KernelGLAD details (ported from v1 §8): per-dataset 5-fold means, UB-GOLD's verbatim pipeline, published
    values against the chance band, score orientation."""
    L = []
    if kernels_raw is None:
        return L
    k = kernels_raw.copy()
    k["auroc"] = pd.to_numeric(k["auroc"], errors="coerce") * scale100(k)
    k["constant"] = k["hp_config"].astype(str).str.contains('"kernel_constant": true')
    L.append("5-fold means per dataset. Rows without `@testfit`: kernel and detector fitted on the normal training "
             "graphs (ranked). `@testfit`: UB-GOLD's own code path, fitted on the unlabelled test set (transductive; "
             "never ranked). 'tuned' = best mean TEST AUROC over UB-GOLD's search space; 'default' = UB-GOLD defaults. "
             "§ = constant kernel matrix (AUROC 50 by construction).")
    L.append("")
    for sname, dss in SETS.items():
        s_ = k[k.dataset.isin(dss)]
        if s_.empty:
            continue
        w = s_.groupby(["method", "protocol", "dataset"]).auroc.mean().unstack("dataset")
        cst = s_.groupby(["method", "protocol", "dataset"]).constant.all().unstack("dataset")
        cols = [d for d in dss if d in w.columns]
        L += [f"*{sname}*:", "", "| method | protocol | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
        for (m, p), r in w.iterrows():
            L.append(f"| {m} | {p} | " + " | ".join(f2(r.get(d, np.nan)) + ("§" if d in cst.columns and
                                                                              bool(cst.loc[(m, p), d] is True) else "")
                                                     for d in cols) + " |")
        L.append("")
    vp = osp.join(args.res, "eval_kernels_ubgold_verbatim.csv")
    if osp.exists(vp):
        v = pd.read_csv(vp)
        for c in ("auroc_released", "auroc_aligned", "frac_label_mismatch", "kernel_maxabs_vs_ours"):
            v[c] = pd.to_numeric(v[c], errors="coerce")
        g = v.groupby("dataset").agg(folds=("fold", "size"), released=("auroc_released", "mean"),
                                     aligned=("auroc_aligned", "mean"), mismatch=("frac_label_mismatch", "mean"),
                                     kmax=("kernel_maxabs_vs_ours", "max")).reset_index()
        L += ["UB-GOLD's verbatim WL-IF pipeline (released label pairing vs correct pairing of the same scores):", "",
              md_table(g, ["dataset", "folds", "released", "aligned", "mismatch", "kmax"]), ""]
    pp = osp.join(args.res, "eval_kernels_published.csv")
    if osp.exists(pp):
        pub = pd.read_csv(pp)
        pub["absz"] = pd.to_numeric(pub["z_vs_chance"], errors="coerce").abs()
        L.append("UB-GOLD's published kernel AUROCs against the chance band of a 5-fold mean: " + "; ".join(
            f"{fam}: {int((g.absz <= 3).sum())}/{len(g)} cells within |z| <= 3"
            for fam, g in pub.groupby(pub.method.str.contains("OCSVM").map({True: "OCSVM", False: "IF"}))) + ".")
    return L

# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def expand_methods(spec):
    if not spec:
        return None
    groups = {"old4": OLD4, "new8": NEW8, "frozen12": DEEP12, "E5": E5, "E4": E4, "kernel": KERNELS,
              "reference": REFERENCES}
    out = set()
    for tok in spec.split(","):
        tok = tok.strip()
        out |= set(groups.get(tok, [canon_method(tok)]))
    return out


def build(args):
    """Ingestion + audit. Returns (long, extra, repro, ctx)."""
    notes, prov = [], {"glad_stats.py": sha256_file(osp.abspath(__file__))}
    keep = expand_methods(args.methods)
    allm = set(DEEP12 + E5 + E4 + KERNELS + REFERENCES)
    methods = keep if keep is not None else allm
    f_main, f_extra, repro = load_frozen12(args, notes, prov)
    e_main, e_extra, failures = load_ext(args, notes, prov, methods)
    k_main, kernels_raw = load_kernels(args, notes, prov)
    r_main = load_references(args, notes, prov)
    main = [r for r in f_main + e_main + k_main + r_main if r["method"] in methods]
    extra = [r for r in f_extra + e_extra if r["method"] in methods]
    runs = index_runs([("harness", args.harness_parts, "harness"), ("deep_rerun", args.dr_parts, "deep_rerun"),
                       ("ext", args.ext_parts, "ext")], methods=methods)
    notes.append(f"trajectories indexed for the audit: {len(runs)} runs")
    regroup = a15_audit(main, runs)
    common_epoch_reselect(main, runs, regroup)
    groups = {(r["method"], r["dataset"], r["hp"]) for r in main}
    main += synth_missing(main, runs, groups)
    for r in extra:                                   # secondary rows: non-finite -> chance, flagged (A15 by analogy)
        r.setdefault("nonfinite_selected", 0)
        r.setdefault("a15_action", "")
        if not finite(r.get("auroc")) and r.get("_runkey") in runs:
            chance(r, runs[r["_runkey"]], "A15.3_secondary_row_nonfinite")
    tie_flags(main, runs)
    diverged = load_diverged(args, notes)
    dk = set(zip(diverged.method, diverged.dataset, diverged.fold, diverged.hp)) if len(diverged) else set()
    n_div = sum((r["method"], r["dataset"], r["fold"], r["hp"]) in dk for r in main)
    main = [r for r in main if (r["method"], r["dataset"], r["fold"], r["hp"]) not in dk]   # A12 §4
    extra = [r for r in extra if (r["method"], r["dataset"], r["fold"], r["hp"]) not in dk]
    if n_div:
        notes.append(f"A12: {n_div} rows of DIVERGED runs removed from every aggregate")
    long, ex = finalize(main), finalize(extra)
    dup = long.duplicated(["method", "dataset", "fold", "protocol"], keep=False)
    if dup.any():
        bad = long[dup][["method", "dataset", "protocol"]].drop_duplicates().head(10).values.tolist()
        raise SystemExit(f"duplicate (method, dataset, fold, protocol) rows: {bad}")
    ctx = dict(notes=notes, prov=prov, runs=runs, diverged=diverged, failures=failures, kernels_raw=kernels_raw,
               methods=sorted(methods & set(long.method)) if len(long) else [])
    return long, ex, finalize(repro), ctx


def analyse(long, ex, repro, ctx, args):
    B, seed = args.B, args.seed
    runs, notes, prov = ctx["runs"], ctx["notes"], ctx["prov"]
    methods = ctx["methods"]
    C = Cubes(long)
    Cx = Cubes(ex) if len(ex) else Cubes(long.iloc[0:0])
    cov = coverage_table(long, ctx["diverged"], methods)
    deg = valnormal_degenerate(long, runs)
    det, ident_summary = identical_config(runs, set(methods) - CHECKPOINT_FREE | ({"ProtoGLAD"} & set(methods)))
    ident_excl = set(ident_summary[ident_summary.excluded_from_pooled_hp_leak].method) | \
        (set(IDENTICAL_CONFIG_DECLARED) & set(methods))
    noise = noise_estimate(long, det, sorted(ident_excl))
    lb, perds = leaderboards(C, long, B, seed)
    infl, infl_pooled = inflation(C, Cx, long, B, seed, ident_excl)
    hpl = hp_leak_table(infl, cov, ident_summary, ident_excl, methods)
    kd = kendall(C, long, B, seed)
    fr, ph, cdt = friedman(C, long)
    sel = selectors(args, B, seed, notes)
    sens = sensitivity_drop(long, B, seed, ident_excl)
    vs, vstatus = version_sensitivity(args, C, B, seed, notes, prov)
    gc, T, tcells = g2c_a9(args, B, seed, args.final, notes, prov)
    vn = g1_selector(C, cov, "valnormal", B, seed, args.final)
    la = g1_selector(C, cov, "last", B, seed, args.final)
    pr = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S %Z"), version=VERSION, final_mode=bool(args.final),
              rule="PASS/FAIL once every frozen-12 UB9 cell is resolved (5 folds done or DIVERGED), or with --final; "
                   "otherwise PENDING with the current value",
              G1=dict(valnormal=vn, last=la, primary=g1_combined(vn, la)),
              G2a=g2a(C, B, seed, args.final, cov), G2b=g2b(C, B, seed, args.final, cov, ident_excl), G2c=gc)
    sts = [pr[k]["status"] for k in ("G2a", "G2b", "G2c")]
    pr["G2"] = "PASS" if "PASS" in sts else ("FAIL" if all(s == "FAIL" for s in sts) else "PENDING")
    g1p = pr["G1"]["primary"]
    pr["GO"] = ("GO" if g1p == "PASS" and pr["G2"] == "PASS" else
                "NO-GO" if g1p in ("FAIL", "SELECTOR-DEPENDENT") or pr["G2"] == "FAIL" else "PENDING")
    if args.methods:
        pr["DEVELOPMENT_RESTRICTION"] = args.methods
    # flags
    fcols = ["method", "stratum", "dataset", "fold", "hp", "protocol", "run_id", "a15_action", "nonfinite_selected",
             "n_distinct", "tie_threshold", "tie_collapsed", "tie_source"]
    flags_rows = long[(long.nonfinite_selected == 1) | (long.tie_collapsed == 1) |
                      (long.a15_action.astype(str) != "")][fcols]
    runs_tab = long.groupby(["method", "dataset", "fold", "hp"]).agg(
        nonfinite_selected=("nonfinite_selected", "max"), tie_collapsed=("tie_collapsed", "max")).reset_index()
    fm = []
    for m in methods:
        g = runs_tab[runs_tab.method == m]
        lm = long[long.method == m]
        fm.append(dict(method=m, stratum=stratum_of(m), runs=len(g), runs_nonfinite_selected=int((g.nonfinite_selected == 1).sum()),
                       runs_tie_collapsed=int((g.tie_collapsed == 1).sum()),
                       rows_tie_unavailable=int((lm.tie_source == "unavailable").sum()),
                       rows_changed_by_A15=int((lm.a15_action.astype(str) != "").sum()),
                       diverged_runs=int(cov[cov.method == m].folds_diverged.sum()) if len(cov) else 0,
                       valnormal_degenerate="|".join(f"{r.hp}" for r in deg[(deg.method == m) & (deg.scope == "all") &
                                                                         deg.valnormal_degenerate].itertuples()) if len(deg) else "",
                       transductive=False, identical_config=m in ident_excl, checkpoint_free=m in CHECKPOINT_FREE))
    fm = pd.DataFrame(fm)
    amend = amendment_status(pr, long, runs_tab, cov, deg, ident_summary, noise, vstatus, ctx["failures"],
                             ctx["diverged"], repro)
    labels = pd.DataFrame([dict(method=m, method_label=METHOD_LABEL.get(m, m), stratum=stratum_of(m),
                                oracle_rule=ORACLE_RULE.get(m, ""), tuned_label=hp_label(m, "AIDS", "ubgold"),
                                hpfree_label=hp_label(m, "AIDS", "hpfree"), transductive=False,
                                notes=METHOD_NOTES.get(m, "")) for m in methods])
    tabs = {"folds": long.drop(columns=["_runkey"], errors="ignore"), "extra_rows": ex.drop(columns=["_runkey"], errors="ignore"),
            "repro_only": repro.drop(columns=["_runkey"], errors="ignore"), "leaderboard": lb, "per_dataset": perds,
            "inflation": infl, "inflation_pooled": infl_pooled, "hpleak": hpl, "identical_config": det,
            "identical_config_summary": ident_summary, "noise_estimate": noise, "flags": flags_rows,
            "flags_method": fm, "valnormal_degenerate": deg, "coverage": cov, "diverged": ctx["diverged"],
            "kendall": kd, "friedman": fr, "posthoc": ph, "cd": cdt, "selectors": sel, "sensitivity": sens,
            "version": vs, "typed_types": T, "typed_cells": tcells, "amendments": amend, "labels": labels}
    return dict(prereg=pr, tabs=tabs, cov=cov, deg=deg, flags_method=fm, flags_rows=flags_rows, sens=sens, lb=lb,
                infl=infl, infl_pooled=infl_pooled, hpleak=hpl, noise=noise, typed_types=T, typed_cells=tcells,
                version=vs, version_status=vstatus, kendall=kd, friedman=fr, selectors=sel, amend=amend)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--final", action="store_true", help="decision-date mode: PENDING becomes PASS/FAIL")
    ap.add_argument("--res", default=RES, help="input results directory")
    ap.add_argument("--out", default=None, help="output directory for eval_leaderboard_v2_* (default: --res)")
    ap.add_argument("--report", default=REPORT)
    ap.add_argument("--eval-folds", dest="eval_folds", default=None, help="eval_folds.csv (default: <res>/eval_folds.csv)")
    ap.add_argument("--harness-parts", dest="harness_parts", default=osp.join(HARNESS, "parts"))
    ap.add_argument("--dr-parts", dest="dr_parts", default=None, help="default: <res>/deep_rerun_parts")
    ap.add_argument("--ext-parts", dest="ext_parts", default=osp.join(EXT, "parts"))
    ap.add_argument("--harness-dir", dest="harness_dir", default=HARNESS, help="for '# DIVERGED' task-file lines")
    ap.add_argument("--ext-dir", dest="ext_dir", default=EXT)
    ap.add_argument("--diverged-registry", dest="diverged_registry", default=osp.join(HERE, "diverged_registry.csv"))
    ap.add_argument("--protoglad-grid", dest="protoglad_grid", default=PROTOGLAD_GRID)
    ap.add_argument("--ms-manifest", dest="ms_manifest", default=MS_MANIFEST)
    ap.add_argument("--tmp", default=None, help="parent of the temporary adapter-output directory")
    ap.add_argument("--keep-tmp", dest="keep_tmp", action="store_true")
    ap.add_argument("--allow-adapter-failure", dest="allow_adapter_failure", action="store_true")
    ap.add_argument("--methods", default=None, help="DEVELOPMENT ONLY: restrict to methods/groups "
                    "(old4,new8,frozen12,E5,E4,kernel,reference or names); the report is marked")
    ap.add_argument("--ingest-only", dest="ingest_only", action="store_true",
                    help="ingest + audit, print structural counts only (no metric values), write nothing")
    args = ap.parse_args(argv)
    args.dr_parts = args.dr_parts or osp.join(args.res, "deep_rerun_parts")
    args.out = args.out or args.res
    long, ex, repro, ctx = build(args)
    if args.ingest_only:
        print(f"rows: {len(long)} protocol, {len(ex)} secondary, {len(repro)} reproduction-only")
        print(long.groupby(["stratum", "base"]).size().unstack(fill_value=0).to_string())
        print(long.groupby(["method", "hp"]).size().unstack(fill_value=0).to_string())
        print("adapter failures:", ctx["failures"] or "none")
        return None
    res = analyse(long, ex, repro, ctx, args)
    os.makedirs(args.out, exist_ok=True)
    outputs = []
    for k, df in res["tabs"].items():
        p = osp.join(args.out, f"eval_leaderboard_{VERSION}_{k}.csv")
        df.to_csv(p, index=False, float_format="%.4f")
        outputs.append(p)
    p = osp.join(args.out, f"eval_leaderboard_{VERSION}_prereg.json")
    with open(p, "w") as fh:
        json.dump(res["prereg"], fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    outputs.append(p)
    ctx2 = dict(res, args=args, notes=ctx["notes"], prov=ctx["prov"], prereg=res["prereg"], kernels_raw=ctx["kernels_raw"],
                outputs=[osp.relpath(o, PRELIM) if o.startswith(PRELIM) else o for o in outputs])
    write_report(args.report, ctx2)
    pr = res["prereg"]
    print(f"G1 primary (A14): {pr['G1']['primary']}  [VAL-NORMAL {pr['G1']['valnormal']['status']}, "
          f"LAST {pr['G1']['last']['status']}]")
    print(f"G2(a) {pr['G2a']['status']}  G2(b) {pr['G2b']['status']}  G2(c)/A9 {pr['G2c']['status']}  "
          f"-> G2 {pr['G2']}; decision {pr['GO']}")
    print(f"wrote {args.report} and {args.out}/eval_leaderboard_{VERSION}_*")
    return res


if __name__ == "__main__":
    main()
