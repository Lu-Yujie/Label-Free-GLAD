"""Post hoc robustness and mechanism analyses of PREREG_AMENDMENTS A32 (label-free GLAD re-evaluation).

These analyses are POST HOC. They are reported as robustness or mechanism analyses, never as confirmatory tests, and
they change no decision criterion (G1, G2, A9). The specification is A32 (written 2026-10-14 20:55:17 EDT, before
any of these analyses was computed). The frozen statistics code (stats/glad_stats.py, v2) is imported read-only and
never modified; its selection rules (UB-GOLD ORACLE replay with patience 30 and the EF filter, VAL-NORMAL, LAST, the
A15 non-finite rules) and its dataset bootstrap are reused unchanged.

Login-node workflow (no SLURM job is needed; every invocation stays under 2 minutes and uses <= 4 threads):

  export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  PY=/N/slate/yujlu/envs/grail-prelim/bin/python
  $PY -B prelim/evalpaper/stats/posthoc_robustness.py prep        # per (dataset, fold): size matching, size-only ref
  $PY -B prelim/evalpaper/stats/posthoc_robustness.py compute     # per run; repeat until 'status' shows no pending run
  $PY -B prelim/evalpaper/stats/posthoc_robustness.py status
  $PY -B prelim/evalpaper/stats/posthoc_robustness.py aggregate   # tables, prelim/results/eval_posthoc_*.csv, report
  /N/slate/yujlu/envs/paper-figs/bin/python -B prelim/evalpaper/stats/posthoc_robustness.py figures

Per-run and per-fold intermediate results are cached as one small JSON file each under
/N/slate/yujlu/grail_data/posthoc_cache/ (runs/, folds/), so the computation is resumable across short invocations.
A cached file records the sha256 of the part JSON and of the score npz it was computed from; 'compute' recomputes a
run whose inputs changed.

A32 items implemented
  1  seven label-free selectors S1-S7, Delta_s = ORACLE - s per detector, pooled mean, dataset-bootstrap 95% CIs
     (4000 resamples, glad_stats' Boot with its default seed 20261012, i.e. the same resamples as REPORT.md)
  2a permutation null (200 label permutations per run, keeping prevalence, seed 20261014 + run index; fast rank-sum
     AUROC: each checkpoint's scores ranked once with average ranks for ties, then one matrix product with the
     permutation label matrix)
  2b Spearman correlations across runs of ORACLE - LAST with the number of evaluation points and with the SD of the
     test-AUROC trajectory (descriptive)
  3  leave-one-dataset-out G1 means, per-dataset detector-mean inflation (forest plot), crossed detector x dataset
     bootstrap (4000 resamples)
  4  size-matched evaluation (1:1 nearest neighbour on standardised (log n, log(m+1)) within each test fold, caliper
     0.2, without replacement, anomalies in test-index order; a dataset is reported only if >= 30% of its anomalies
     are matched in every fold) with the size-only reference (Ledoit-Wolf Mahalanobis on (log n, log(m+1), density)
     fitted on the training normals, verbatim from grail0/channels.py + grail0/controls.py) on the matched subset
  5  reproduction-fidelity tiers of the 9 recent methods and the pooled recent-method inflation without the
     "reproduction not met" methods
Interpretations where A32 is silent are listed in INTERPRETATIONS below and in the report.
"""
import argparse
import hashlib
import io
import json
import math
import os
import os.path as osp
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
EVALPAPER = osp.dirname(HERE)
PRELIM = osp.dirname(EVALPAPER)
RES = osp.join(PRELIM, "results")
HARNESS_PARTS = osp.join(EVALPAPER, "harness", "parts")
DR_PARTS = osp.join(RES, "deep_rerun_parts")
EXT_PARTS = osp.join(EVALPAPER, "ext", "parts")
GLAD_STATS = osp.join(HERE, "glad_stats.py")
GLAD_STATS_FROZEN_SHA = "2731b02f07519f8bbd97305018fd04ff3771ca3f71e4b1d86b6d7a34d5802c78"
FOLDS_CSV = osp.join(RES, "eval_leaderboard_v2_folds.csv")          # glad_stats output (for the sanity check)
POOLED_CSV = osp.join(RES, "eval_leaderboard_v2_inflation_pooled.csv")
GRAIL0_CSV = osp.join(RES, "grail0_folds.csv")
CACHE = "/N/slate/yujlu/grail_data/posthoc_cache"
CACHE_RUNS = osp.join(CACHE, "runs")
CACHE_FOLDS = osp.join(CACHE, "folds")
FIGDIR = osp.join(EVALPAPER, "paper", "figures")
REPORT = osp.join(HERE, "POSTHOC_REPORT.md")
OUT = osp.join(RES, "eval_posthoc_{}.csv")

UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
DEEP12 = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD", "GLADC", "CVTGAD", "GOOD-D", "GraphDE",
          "GraphCL_IF", "GraphCL_OCSVM", "InfoGraph_IF", "InfoGraph_OCSVM"]
OLD4, NEW8 = DEEP12[:4], DEEP12[4:]
E5 = ["MUSE", "LGKDE", "GLADPro", "GLADMamba", "SEGO"]
E4 = ["GLASS", "ProtoGLAD", "HC-GLAD", "SUL-GUL"]
METHODS_ALL = DEEP12 + E5 + E4                       # canonical order; defines the run index (permutation seeds)
CHECKPOINT_FREE_RECENT = {"ProtoGLAD"}               # A20: no checkpoint axis -> no selector / inflation analysis
RECENT_IT = [m for m in E5 + E4 if m not in CHECKPOINT_FREE_RECENT]
NFOLD = 5

B = 4000
BOOT_SEED = 20261012          # glad_stats --seed default: identical dataset resamples to REPORT.md
PERM_SEED0 = 20261014         # A32 2(a): seed = 20261014 + run index
N_PERM = 200
CALIPER = 0.2
MATCH_MIN = 0.30
CHANCE = 50.0

SELECTORS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
SEL_LABEL = {"S1": "LAST", "S2": "VAL-NORMAL (min mean held-out score)", "S3": "min median held-out score",
             "S4": "min training loss", "S5": "point closest to half the epoch budget",
             "S6": "uniform checkpoint (trajectory mean)", "S7": "min mean standardised held-out score"}
SEL_TICK = {"S1": "LAST\n ", "S2": "VAL-\nNORMAL", "S3": "median\nheld-out", "S4": "min train\nloss",
            "S5": "mid-\nbudget", "S6": "uniform\nckpt", "S7": "stand.\nheld-out"}
SEL_SHORT = {"S1": "LAST", "S2": "VAL-NORMAL", "S3": "Median held-out", "S4": "Min train loss",
             "S5": "Mid-budget", "S6": "Uniform ckpt", "S7": "Standardised held-out"}
# S7 ("holdout_score_norm or its equivalent"): a held-out score standardised with statistics of the normal training
# graphs at the same checkpoint. Recorded separately only for CVTGAD (harness: adaptive normalisation with the
# training-loss mean/std of the epoch). For three recent methods the method's own per-graph score already IS such a
# standardised score, so S7 coincides with S2 by construction. No other run records one -> S7 unavailable there.
S7_SOURCE = {"CVTGAD": ("holdout_score_norm", "separately recorded adaptive-normalised held-out score"),
             "GLADMamba": ("holdout_score", "own score z-normalised with the epoch's training-loss mean/std (= S2)"),
             "SEGO": ("holdout_score", "own score standardised with the epoch's training-loss means/stds (= S2)"),
             "GLASS": ("holdout_score", "default rule rcsm_full_test_cdf: CDF-calibrated against training-reference "
                                        "scores of the same checkpoint (= S2)")}

INTERPRETATIONS = [
    "Run set: UB9 x 'ubgold' x 5 folds. The 12 UB-GOLD detectors use the same runs as eval_folds.csv (harness parts; "
    "for SIGNET/OCGTL/OCGIN/GLocalKD on UB9 the deep_rerun parts, which have per-point trajectories but no per-graph "
    "scores). Recent methods: the 8 iterative E5/E4 methods; ProtoGLAD has no checkpoint axis (A20) and enters only "
    "item 4 (its VAL-NORMAL and ORACLE points coincide) and item 5's tier table.",
    "ORACLE = UB-GOLD rule (patience 30) for every method (A32 'unless stated otherwise'; A26 for pooled numbers): "
    "glad_stats.ubgold_select with the EF filter set of glad_stats, CVTGAD selected on auroc_sel (A4).",
    "A15 carried over to the new selectors: a signal-based selector (S2, S3, S4, S7) never picks a point whose signal "
    "is non-finite (S3's median is non-finite if any held-out score is); no finite signal, or a selected point with "
    "non-finite test scores, gives AUROC 50 and a flag (glad_stats interpretation 2a/2b). S5 at a non-finite point "
    "gives 50. S6 counts a non-finite point as 50 (a uniformly drawn non-finite checkpoint ranks at chance).",
    "Ties: S2-S5 take the earliest evaluation point among ties (as glad_stats.valnormal_select).",
    "S4: 'mean training loss' = the epoch's mean training loss over batches as recorded at each evaluation point "
    "(traj train_loss, equal to the per-epoch train_loss list).",
    "S5: epoch budget = num_epoch of the run's resolved configuration; distance |epoch - num_epoch/2|.",
    "S7: 'holdout_score_norm or its equivalent' = a held-out score standardised with statistics of the normal "
    "training graphs at the same checkpoint. Only CVTGAD records it separately; GLADMamba, SEGO and GLASS score with "
    "such a standardised score already (S7 = S2 there). No other run records one, so S7 is not available for the "
    "other 11 UB-GOLD detectors and 5 recent methods; no substitute is computed.",
    "Item 1 pooled means are over the detectors with complete UB9 coverage for that selector (as glad_stats). Because "
    "S3 and S7 have smaller coverage, the table also gives S1-S6 pooled over the S3-covered detectors (coverage-"
    "matched view; same estimand, same bootstrap).",
    "Item 2(a): 'max over evaluation points' is the plain maximum over all finite points (not the patience-30 rule), "
    "as A32 states; this can only make the null larger than a null of the patience rule. The trajectory mean counts "
    "non-finite points as 50, a non-finite LAST is 50 (A15 rule 3). Null values are aggregated like the observed "
    "values (5-fold mean per dataset, mean over UB9, mean over detectors). Ratios: null(max - LAST) / observed "
    "(ORACLE - LAST) as A32 states; null(max - mean) is reported against observed ORACLE - LAST (literal) and "
    "against observed ORACLE - S6 (its observed analogue).",
    "Run index (permutation seed): position in the canonical list METHODS_ALL x UB9 x folds, i.e. "
    "index = (method_pos * 9 + dataset_pos) * 5 + fold with METHODS_ALL = 12 UB-GOLD detectors, E5, E4 in the "
    "glad_stats order.",
    "Item 2(b): Spearman correlation over all runs of a pool (frozen-12; recent; both); SD of the test-AUROC "
    "trajectory = population SD (ddof 0) over finite evaluation points, in AUROC points.",
    "Item 3 forest plot: per-dataset detector-mean inflation with a paired fold-bootstrap 95% CI (4000 resamples; the "
    "5 folds are resampled jointly for all detectors, glad_stats' fold resamples), consistent with the stated target "
    "(the evaluated detectors fixed, datasets varying); the detector range (min, max) is reported beside it. Crossed "
    "bootstrap: detectors and datasets resampled independently with replacement in each of 4000 replicates "
    "(seed 20261012) on the 12 x 9 matrix of 5-fold means.",
    "Item 4: features standardised with the mean and population SD (ddof 0) over all test graphs of the fold; "
    "distance = Euclidean distance in the standardised plane, matched if <= 0.2; nearest-distance ties go to the "
    "normal with the lower test index; m = number of undirected edges (fold cache, equal to the npz test_n_edges); "
    "the size-only reference is refitted per fold on all normal training graphs of the fold, as in grail0 "
    "(verified against grail0_folds.csv on the full test sets). Detector AUROC on the matched subset at the run's "
    "VAL-NORMAL and ORACLE (UB-GOLD rule) points; a non-finite selected point is 50 (A15).",
    "Item 5 tiers: each method gets the lowest-fidelity tier that applies, in A32's order (native < minimal "
    "compatibility patch < framework port < incomplete released configuration < reproduction not met). 'Native' = no "
    "edit to the method's model, loss or scoring code beyond harness instrumentation (data root, folds, seeds, "
    "held-out carve-out, no_grad, recording, resume); 'incomplete released configuration' = no released "
    "configuration for at least one UB9 dataset, or a released component that cannot run in the one-class protocol; "
    "'reproduction not met' = the pre-registered A10 §4 check reported as not met. Primary exclusion = LGKDE (the "
    "only method documented as 'not met'). Because the sources conflict for GLADPro (NOTES: criterion met, 2 of 7, "
    "both matches on datasets whose anomaly class likely differs; Appendix B caption: such datasets are not counted) "
    "and SUL/GUL is 'not evaluable' (A22), a second, labelled row also drops GLADPro and SUL-GUL.",
]

# Item 5: fidelity facets, transcribed from Appendix B (paper/sections/app_ports.tex, tab:ports and paragraphs) and
# ext/<M>/NOTES.md (deviation lists and reproduction sections).
FIDELITY = [
    dict(method="MUSE", code="minimal compatibility patch",
         code_evidence="adjacency target sized by node count (released loss crashes on AIDS/ENZYMES isolated nodes); "
                       "training loop copied into the adapter, equivalence-tested (MUSE NOTES 1.6, 4)",
         config="incomplete released configuration",
         config_evidence="second-stage width/lr chosen with labelled validation anomalies in the released code, not "
                         "available in one-class folds (fixed h32/lr1e-3); modal values for COX2, ENZYMES (A13)",
         repro="met", repro_evidence="own code+splits 4/5; UB-GOLD folds 2/7 within +-5 (app_ports tab:ports)"),
    dict(method="LGKDE", code="framework port",
         code_evidence="DGL -> PyTorch Geometric port, bit-identical on MUTAG/BZR (LGKDE NOTES 2-3)",
         config="incomplete released configuration",
         config_evidence="only MUTAG.yaml released, used for all datasets (A18)",
         repro="not met", repro_evidence="1 of 4 comparable datasets within +-5, also under its own pipeline "
                                         "(LGKDE NOTES 4.3; app_ports tab:ports)"),
    dict(method="GLADPro", code="minimal compatibility patch",
         code_evidence="D1 PyG>=2.5 propagate, D2 out-of-place eps for torch>=2 autograd; forward values unchanged "
                       "(GLADPro NOTES deviations)",
         config="incomplete released configuration",
         config_evidence="README lines only for MUTAG, PROTEINS, IMDB-B; modal (= defaults with 500 epochs) elsewhere",
         repro="met (contested)",
         repro_evidence="NOTES: criterion met, COX2 +3.34 and ENZYMES +4.64 vs M-GLAD, both with a likely different "
                        "anomaly/normal class; 0 of 3 direction-matched datasets within +-5 (MUTAG +5.40, IMDB-B "
                        "-11.03, AIDS -22.75); paper text not retrievable"),
    dict(method="GLADMamba", code="native",
         code_evidence="only data root, seeds, carve-out, no_grad, recording, resume (GLADMamba NOTES deviations)",
         config="incomplete released configuration",
         config_evidence="no released script for PROTEINS_full (modal) (A14; app_ports tab:ports)",
         repro="met", repro_evidence="6 of 8 within +-5 (app_ports tab:ports)"),
    dict(method="SEGO", code="minimal compatibility patch",
         code_evidence="three PyG 2.0->2.6 API patches (one checked against an independent reference), follow_batch "
                       "for the AD loader (SEGO NOTES deviations 1, 5)",
         config="incomplete released configuration",
         config_evidence="no anomaly-detection configuration released; 'released-modal' over 10 pair lines (A16)",
         repro="met", repro_evidence="2 of 5 within +-5 (SEGO NOTES verdict; app_ports tab:ports)"),
    dict(method="GLASS", code="native",
         code_evidence="instrumentation only; evaluate_all_scorers copied line by line to keep per-graph scores; "
                       "early stopping replayed (GLASS NOTES D1-D8)",
         config="released configuration complete",
         config_evidence="one released configuration for all datasets (README line; A23 §3)",
         repro="met", repro_evidence="5 of 7 direction-matched datasets within +-5 (GLASS NOTES; tab:ports)"),
    dict(method="ProtoGLAD", code="minimal compatibility patch",
         code_evidence="CPU fallback copy of IK_inne_gpu.__init__; memory-saving copy of compute_graph_embeddings "
                       "(identical values) (ProtoGLAD NOTES D1, D3)",
         config="released configuration complete",
         config_evidence="replay of its released test-AUROC grid search (README line for MUTAG) (A20)",
         repro="met", repro_evidence="3 of 4 within +-5 (app_ports tab:ports)"),
    dict(method="HC-GLAD", code="native",
         code_evidence="instrumentation only; REDDIT-B hyperedges cached from the verbatim function (HC-GLAD NOTES "
                       "deviations)",
         config="incomplete released configuration",
         config_evidence="no released script for NCI1, IMDB-B (modal values) (A19; app_ports tab:ports)",
         repro="met", repro_evidence="7 of 7 within +-5 of the preprint (app_ports tab:ports)"),
    dict(method="SUL-GUL", code="minimal compatibility patch",
         code_evidence="topk capped at the training-set size where the released code would crash (SUL-GUL NOTES D7)",
         config="incomplete released configuration",
         config_evidence="SUL not released; paper's 1000 epochs and grid-searched hyperparameters not released "
                         "(A21, A22)",
         repro="not evaluable", repro_evidence="paper reports GUL only at 100% contamination (A22)"),
]
TIER_ORDER = ["native", "minimal compatibility patch", "framework port", "incomplete released configuration",
              "reproduction not met"]


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    if not osp.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_json_atomic(path, obj):
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as fh:
        json.dump(obj, fh, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


_GS = None


def gs():
    """glad_stats v2, imported read-only (its hash is checked against FROZEN.md)."""
    global _GS
    if _GS is None:
        import importlib.util
        h = sha256_file(GLAD_STATS)
        if h != GLAD_STATS_FROZEN_SHA:
            raise SystemExit(f"glad_stats.py sha256 {h} differs from the frozen hash {GLAD_STATS_FROZEN_SHA}")
        spec = importlib.util.spec_from_file_location("_posthoc_glad_stats", GLAD_STATS)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.UB9 == UB9 and mod.DEEP12 == DEEP12 and mod.E5 == E5 and mod.E4 == E4
        assert CHECKPOINT_FREE_RECENT == set(E5 + E4) & mod.CHECKPOINT_FREE
        _GS = mod
    return _GS


def rank_auc(R, y):
    """AUROC from rows of average ranks R (k x n) for labels y (1 = anomaly)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (R[..., y == 1].sum(-1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auc(s, y):
    from scipy.stats import rankdata
    return float(rank_auc(rankdata(np.asarray(s, float))[None, :], np.asarray(y))[0])


def size_feats(n, m):
    """grail0/channels.py size_feats, verbatim formula: [log n, log(m+1), density]."""
    out = np.zeros((len(n), 3))
    for i in range(len(n)):
        nn, mm = int(n[i]), int(m[i])
        out[i] = [math.log(max(nn, 1)), math.log(mm + 1.0), (2.0 * mm / (nn * (nn - 1))) if nn > 1 else 0.0]
    return out


def size_only_scores(Ftr, Fte, ridge=1e-6):
    """grail0/controls.py mahal(size_feats(train), size_feats(test), CONFIG['ridge_S0']) with channels._lw."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf(assume_centered=False).fit(Ftr)
    S = lw.covariance_ + ridge * np.eye(Ftr.shape[1])
    mu, P = lw.location_, np.linalg.inv(S)
    D = Fte - mu
    return np.sum(D @ P * D, 1)


def nm_digest(n, m):
    return sha256_bytes(np.asarray(n, np.int64).tobytes() + np.asarray(m, np.int64).tobytes())


# ----------------------------------------------------------------------------------------------------------------
# run enumeration
# ----------------------------------------------------------------------------------------------------------------
def _untagged(path, stem):
    if not osp.exists(path):
        return False
    with open(path) as fh:
        meta = json.load(fh)["meta"]
    return meta.get("run_id") == stem and meta.get("max_epochs_override") is None and \
        meta.get("hp", "ubgold") == "ubgold"


def enumerate_runs():
    """All (method, UB9 dataset, fold) 'ubgold' runs of the canonical list, with their part file (or None)."""
    out = []
    for mi, m in enumerate(METHODS_ALL):
        for di, ds in enumerate(UB9):
            for k in range(NFOLD):
                stem = f"{m}_{ds}_fold{k}"
                path, group = None, None
                if m in DEEP12:
                    p = osp.join(HARNESS_PARTS, stem + ".json")
                    if _untagged(p, stem):
                        path, group = p, "harness"
                    else:                                   # collect.py: deep_rerun parts for the old 4 (UB-GOLD hp)
                        p = osp.join(DR_PARTS, stem + ".json")
                        if m in OLD4 and _untagged(p, stem):
                            path, group = p, "deep_rerun"
                else:
                    p = osp.join(EXT_PARTS, stem + ".json")
                    if _untagged(p, stem):
                        path, group = p, "ext"
                out.append(dict(method=m, dataset=ds, fold=k, run_index=(mi * len(UB9) + di) * NFOLD + k,
                                path=path, group=group, stem=stem))
    return out


def cache_path(spec):
    return osp.join(CACHE_RUNS, f"{spec['stem']}.json")


def fold_cache_path(ds, k):
    return osp.join(CACHE_FOLDS, f"{ds}_fold{k}.json")


# ----------------------------------------------------------------------------------------------------------------
# prep: per (dataset, fold) size matching and size-only reference
# ----------------------------------------------------------------------------------------------------------------
def match_fold(n, m, y, test_idx, caliper=CALIPER):
    X = np.column_stack([np.log(np.maximum(np.asarray(n, float), 1.0)), np.log(np.asarray(m, float) + 1.0)])
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / np.where(sd > 0, sd, 1.0)
    order = np.argsort(np.asarray(test_idx), kind="stable")     # test-index order
    anoms = [int(i) for i in order if y[i] == 1]
    norms = np.array([int(i) for i in order if y[i] == 0], dtype=int)
    avail = np.ones(len(norms), bool)
    pa, pn, dist = [], [], []
    for a in anoms:
        cand = np.flatnonzero(avail)
        if not len(cand):
            break
        d = np.sqrt(((Z[norms[cand]] - Z[a]) ** 2).sum(1))
        jj = int(np.argmin(d))                                   # first minimum = lowest test index among ties
        if d[jj] <= caliper:
            pa.append(a)
            pn.append(int(norms[cand[jj]]))
            dist.append(float(d[jj]))
            avail[cand[jj]] = False
    return pa, pn, dist, Z


def smd(x, y):
    """standardised mean difference anomalies vs normals (pooled SD)."""
    a, b = x[y == 1], x[y == 0]
    s = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0) if len(a) > 1 and len(b) > 1 else float("nan")
    return float((a.mean() - b.mean()) / s) if s and s > 0 else float("nan")


def cmd_prep(args):
    sys.path.insert(0, PRELIM)
    from common.data import load_folds
    g0 = pd.read_csv(GRAIL0_CSV)
    g0 = g0[g0.method == "ctrl_size_lw"].set_index(["dataset", "fold"])["auroc"]
    t0 = time.time()
    for ds in UB9:
        if all(osp.exists(fold_cache_path(ds, k)) for k in range(NFOLD)) and not args.force:
            continue
        if time.time() - t0 > args.budget:
            print("[prep] time budget reached; run again")
            return
        for F in load_folds(ds):
            k = int(F["fold"])
            te, tr = F["test"], F["train"]
            n = np.array([g["n"] for g in te], np.int64)
            m = np.array([len(g["edges"]) for g in te], np.int64)
            y = np.asarray(F["y_test"], np.int64)
            tidx = np.asarray(F["meta"]["test_idx"], np.int64)
            ntr = np.array([g["n"] for g in tr], np.int64)
            mtr = np.array([len(g["edges"]) for g in tr], np.int64)
            s_full = size_only_scores(size_feats(ntr, mtr), size_feats(n, m))
            pa, pn, dist, Z = match_fold(n, m, y, tidx)
            sub = np.array(pa + pn, int)
            Fte = size_feats(n, m)
            rec = dict(dataset=ds, fold=k, n_test=len(y), n_anom=int(y.sum()), n_norm=int((y == 0).sum()),
                       n_train=len(tr), test_idx_sorted=bool(np.all(np.diff(tidx) > 0)), nm_sha=nm_digest(n, m),
                       n=n.tolist(), m=m.tolist(), y=y.tolist(),
                       matched_anom_pos=pa, matched_norm_pos=pn, match_dist=dist, n_pairs=len(pa),
                       frac_anom_matched=len(pa) / max(1, int(y.sum())),
                       sizeonly_auc_full=100 * auc(s_full, y),
                       sizeonly_auc_matched=100 * auc(s_full[sub], y[sub]) if len(pa) else None,
                       sizeonly_auc_grail0=100 * float(g0.get((ds, k), np.nan)),
                       smd_before={c: smd(Fte[:, i], y) for i, c in enumerate(["log_n", "log_m1", "density"])},
                       smd_after=({c: smd(Fte[sub, i], y[sub]) for i, c in enumerate(["log_n", "log_m1", "density"])}
                                  if len(pa) > 1 else None),
                       fold_cache=osp.join("/N/slate/yujlu/grail_data/folds", ds + ".pkl"),
                       fold_cache_sha256=sha256_file(osp.join("/N/slate/yujlu/grail_data/folds", ds + ".pkl")),
                       computed=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
            write_json_atomic(fold_cache_path(ds, k), rec)
            print(f"[prep] {ds} fold {k}: anomalies {rec['n_anom']}, normals {rec['n_norm']}, matched {len(pa)} "
                  f"({rec['frac_anom_matched']:.2f}); size-only full {rec['sizeonly_auc_full']:.4f} "
                  f"(grail0 {rec['sizeonly_auc_grail0']:.4f})")
    print("[prep] done")


# ----------------------------------------------------------------------------------------------------------------
# compute: per run
# ----------------------------------------------------------------------------------------------------------------
def compute_run(spec, fold_rec):
    from scipy.stats import rankdata
    G = gs()
    m = spec["method"]
    raw = open(spec["path"], "rb").read()
    J = json.loads(raw)
    traj = J["traj"]
    run = G.run_info(spec["path"], spec["group"])
    assert run["method"] == m and run["dataset"] == spec["dataset"] and run["fold"] == spec["fold"]
    ep = run["epochs"]
    au = 100.0 * run["auroc"]
    ok = run["test_ok"]
    ne = int(run["num_epoch"])
    T_ = len(ep)

    def at(j, none_flag):
        if j is None:
            return dict(auroc=CHANCE, epoch=None, flag=none_flag)
        j = int(j)
        if not ok[j]:
            return dict(auroc=CHANCE, epoch=int(ep[j]), flag="A15_selected_point_test_nonfinite")
        return dict(auroc=float(au[j]), epoch=int(ep[j]), flag="")

    def argmin_finite(v):
        v = np.asarray(v, float)
        f = np.isfinite(v)
        return int(np.argmin(np.where(f, v, np.inf))) if f.any() else None

    sel, idx = {}, {}
    idx["ORACLE"] = G.ubgold_select(run, m in G.EF_FILTER)
    sel["ORACLE"] = at(idx["ORACLE"], "A15.1_no_finite_point")
    jl = int(np.flatnonzero(ep == ep.max())[0])
    idx["S1"] = jl
    sel["S1"] = at(jl, "")
    idx["S2"] = G.valnormal_select(run)
    sel["S2"] = at(idx["S2"], "A15.2_no_finite_holdout")
    tl = G._arr(traj, "train_loss")
    idx["S4"] = argmin_finite(tl)
    sel["S4"] = at(idx["S4"], "no_finite_train_loss")
    idx["S5"] = int(np.argmin(np.abs(ep - ne / 2.0)))
    sel["S5"] = at(idx["S5"], "")
    sel["S6"] = dict(auroc=float(np.mean(np.where(ok, au, CHANCE))), epoch=None,
                     flag=("A15_nonfinite_points_as_50:%d" % int((~ok).sum())) if (~ok).any() else "")
    if m in S7_SOURCE:
        key = S7_SOURCE[m][0]
        v = G._arr(traj, key) if key != "holdout_score" else run["hold"]
        idx["S7"] = argmin_finite(v)
        sel["S7"] = at(idx["S7"], "A15.2_no_finite_holdout")
        sel["S7"]["source"] = key
    else:
        sel["S7"] = None
    sel["S3"] = None
    finite_au = au[ok]
    rec = dict(method=m, dataset=spec["dataset"], fold=spec["fold"], hp="ubgold", group=spec["group"],
               run_id=J["meta"]["run_id"], run_index=spec["run_index"], part=spec["path"], part_sha256=sha256_bytes(raw),
               n_eval=int(T_), num_epoch=ne, eval_freq=int(run["eval_freq"]), n_test=int(run["n_test"]),
               n_anom=int(run["n_anom"]), n_nonfinite_points=int((~ok).sum()),
               traj_sd=float(np.std(finite_au)) if len(finite_au) else None, idx={k: (None if v is None else int(v))
                                                                              for k, v in idx.items()},
               npz=None, npz_sha256=None, perm=None, sizematch=None, auroc_check_maxdiff=None)
    p = run.get("npz")
    if p and osp.exists(p):
        zb = open(p, "rb").read()
        z = np.load(io.BytesIO(zb), allow_pickle=False)
        rec["npz"], rec["npz_sha256"] = p, sha256_bytes(zb)
        Tm = np.asarray(z["test"], float)
        H = np.asarray(z["hold"], float)
        y = np.asarray(z["y_test"]).astype(np.int64)
        assert list(np.asarray(z["epochs"]).tolist()) == ep.tolist(), "npz epochs differ from the trajectory"
        assert Tm.shape[0] == T_ and H.shape[0] == T_
        rowfin = np.isfinite(Tm).all(1)
        rec["rows_finite_equal_test_ok"] = bool(np.array_equal(rowfin, ok))
        # S3: minimum median held-out score (non-finite median if any held-out score is non-finite)
        med = np.where(np.isfinite(H).all(1), np.median(np.where(np.isfinite(H), H, 0.0), 1), np.nan)
        idx["S3"] = argmin_finite(med)
        sel["S3"] = at(idx["S3"], "A15.2_no_finite_holdout")
        rec["idx"]["S3"] = idx["S3"]
        # fast rank-sum AUROC: rank each finite checkpoint once (average ranks for ties)
        R = np.full(Tm.shape, np.nan)
        if rowfin.any():
            R[rowfin] = rankdata(Tm[rowfin], axis=1)
        a_obs = np.full(T_, np.nan)
        a_obs[rowfin] = rank_auc(R[rowfin], y)
        both = rowfin & ok
        rec["auroc_check_maxdiff"] = float(np.max(np.abs(100 * a_obs[both] - au[both]))) if both.any() else None
        # 2(a) permutation null
        n, n1 = len(y), int(y.sum())
        rng = np.random.default_rng(PERM_SEED0 + spec["run_index"])
        P = np.empty((n, N_PERM))
        for b in range(N_PERM):
            P[:, b] = y[rng.permutation(n)]
        A = np.full((T_, N_PERM), np.nan)
        if rowfin.any():
            A[rowfin] = ((R[rowfin] @ P) - n1 * (n1 + 1) / 2.0) / (n1 * (n - n1))
        A *= 100.0
        okp = rowfin & ok
        mx = np.max(np.where(okp[:, None], A, -np.inf), 0) if okp.any() else np.full(N_PERM, CHANCE)
        last = A[jl] if okp[jl] else np.full(N_PERM, CHANCE)
        mean = np.where(okp[:, None], A, CHANCE).mean(0)
        d1, d2 = mx - last, mx - mean
        rec["perm"] = dict(seed=PERM_SEED0 + spec["run_index"], n_perm=N_PERM,
                           null_max_minus_last_mean=float(d1.mean()), null_max_minus_last_sd=float(d1.std(ddof=1)),
                           null_max_minus_last_q95=float(np.quantile(d1, 0.95)),
                           null_max_minus_mean_mean=float(d2.mean()), null_max_minus_mean_sd=float(d2.std(ddof=1)),
                           null_max_minus_mean_q95=float(np.quantile(d2, 0.95)))
        # 4 size-matched AUROC at VAL-NORMAL and ORACLE points
        if fold_rec is not None:
            nn_, mm_ = np.asarray(z["test_n_nodes"], np.int64), np.asarray(z["test_n_edges"], np.int64)
            assert nm_digest(nn_, mm_) == fold_rec["nm_sha"], "npz graph sizes differ from the fold cache"
            assert np.array_equal(y, np.asarray(fold_rec["y"])), "npz labels differ from the fold cache"
            sub = np.array(fold_rec["matched_anom_pos"] + fold_rec["matched_norm_pos"], int)
            sm = dict(n_pairs=len(fold_rec["matched_anom_pos"]))
            for name, key, nf in (("valnormal", "S2", "A15.2_no_finite_holdout"),
                                  ("oracle", "ORACLE", "A15.1_no_finite_point")):
                j = idx[key]
                if j is None or not ok[int(j)] or not len(sub):
                    sm[name] = dict(full=CHANCE if (j is None or not ok[int(j)]) else float(au[int(j)]),
                                    matched=CHANCE if (j is None or not ok[int(j)]) else None,
                                    epoch=None if j is None else int(ep[int(j)]))
                else:
                    sm[name] = dict(full=float(au[int(j)]), matched=100 * auc(Tm[int(j), sub], y[sub]),
                                    epoch=int(ep[int(j)]))
            rec["sizematch"] = sm
    rec["sel"] = sel
    rec["idx"] = {k: (None if v is None else int(v)) for k, v in idx.items()}
    rec["computed"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    return rec


def run_is_current(spec):
    p = cache_path(spec)
    if not osp.exists(p):
        return False
    try:
        c = json.load(open(p))
    except (OSError, ValueError):
        return False
    if c.get("part") != spec["path"] or c.get("part_sha256") != sha256_file(spec["path"]):
        return False
    if c.get("npz") and c.get("npz_sha256") != sha256_file(c["npz"]):
        return False
    return True


def todo_runs(check_hash=True):
    specs = [s for s in enumerate_runs() if s["path"]]       # ProtoGLAD included: item 4 only (A20)
    return specs, [s for s in specs if not (run_is_current(s) if check_hash else osp.exists(cache_path(s)))]


def cmd_compute(args):
    t0 = time.time()
    specs, todo = todo_runs(check_hash=args.recheck)
    if args.only:
        keep = set(args.only.split(","))
        todo = [s for s in todo if s["method"] in keep or s["dataset"] in keep]
    # longest first (score matrices with many evaluation points), so the tail of short invocations is short
    todo.sort(key=lambda s: -(osp.getsize(s["path"]) if s["path"] else 0))
    done = 0
    fold_recs = {}
    for s in todo:
        if time.time() - t0 > args.budget:
            break
        key = (s["dataset"], s["fold"])
        if key not in fold_recs:
            fp = fold_cache_path(*key)
            fold_recs[key] = json.load(open(fp)) if osp.exists(fp) else None
            if fold_recs[key] is None:
                raise SystemExit(f"fold cache {fp} missing: run 'prep' first")
        rec = compute_run(s, fold_recs[key])
        write_json_atomic(cache_path(s), rec)
        done += 1
    left = len(todo) - done
    print(f"[compute] {done} runs in {time.time() - t0:.1f} s; {left} pending of {len(specs)} runs with parts")


def cmd_status(args):
    specs = enumerate_runs()
    have = [s for s in specs if s["path"]]
    miss = [s for s in specs if not s["path"]]
    cached = [s for s in have if osp.exists(cache_path(s))]
    folds = sum(osp.exists(fold_cache_path(d, k)) for d in UB9 for k in range(NFOLD))
    print(f"runs in canonical list {len(specs)}; with part {len(have)}; without part {len(miss)}; "
          f"cached {len(cached)}; fold caches {folds}/45")
    if miss:
        print("missing parts:", sorted({(s['method'], s['dataset']) for s in miss})[:20])


# ----------------------------------------------------------------------------------------------------------------
# aggregate
# ----------------------------------------------------------------------------------------------------------------
def load_cache():
    rows = []
    specs = enumerate_runs()
    for s in specs:
        if not s["path"]:
            continue
        p = cache_path(s)
        if not osp.exists(p):
            raise SystemExit(f"run cache missing for {s['stem']}: run 'compute' until no run is pending")
        c = json.load(open(p))
        if c.get("part") != s["path"] or c.get("part_sha256") != sha256_file(s["path"]) or \
                (c.get("npz") and c.get("npz_sha256") != sha256_file(c["npz"])):
            raise SystemExit(f"stale run cache for {s['stem']} (input changed): run 'compute --recheck'")
        r = dict(method=c["method"], dataset=c["dataset"], fold=c["fold"], group=c["group"], run_id=c["run_id"],
                 run_index=c["run_index"], n_eval=c["n_eval"], num_epoch=c["num_epoch"], eval_freq=c["eval_freq"],
                 n_test=c["n_test"], n_anom=c["n_anom"], n_nonfinite_points=c["n_nonfinite_points"],
                 traj_sd=c["traj_sd"], has_scores=c["npz"] is not None, auroc_check_maxdiff=c["auroc_check_maxdiff"],
                 rows_finite_equal_test_ok=c.get("rows_finite_equal_test_ok"), part_sha256=c["part_sha256"],
                 npz_sha256=c["npz_sha256"])
        for k in ["ORACLE"] + SELECTORS:
            v = c["sel"].get(k)
            r[k] = np.nan if v is None else v["auroc"]
            r["ep_" + k] = np.nan if (v is None or v.get("epoch") is None) else v["epoch"]
            r["flag_" + k] = "" if v is None else v.get("flag", "")
        for kk in ("null_max_minus_last_mean", "null_max_minus_last_sd", "null_max_minus_last_q95",
                   "null_max_minus_mean_mean", "null_max_minus_mean_sd", "null_max_minus_mean_q95", "seed"):
            r["perm_" + kk] = (c["perm"] or {}).get(kk, np.nan)
        sm = c.get("sizematch") or {}
        r["sm_n_pairs"] = sm.get("n_pairs", np.nan)
        for nm in ("valnormal", "oracle"):
            x = sm.get(nm) or {}
            r[f"sm_{nm}_full"] = x.get("full", np.nan)
            r[f"sm_{nm}_matched"] = np.nan if x.get("matched") is None else x.get("matched")
        rows.append(r)
    R = pd.DataFrame(rows)
    R["stratum"] = np.where(R.method.isin(DEEP12), "frozen12", "recent")
    return R, specs


def cube(R, methods, col):
    """(M, 9, 5) array of a run-level column; NaN where missing."""
    A = np.full((len(methods), len(UB9), NFOLD), np.nan)
    mi = {m: i for i, m in enumerate(methods)}
    di = {d: i for i, d in enumerate(UB9)}
    for m, d, f, v in zip(R.method, R.dataset, R.fold, R[col]):
        if m in mi and d in di:
            A[mi[m], di[d], int(f)] = v
    return A


def pool_ci(D, boot):
    """pooled mean over methods of D (M, 9, 5) and its dataset-bootstrap CI (glad_stats inflation())."""
    G = gs()
    v = D.mean(-1).mean(-1)
    lo, hi = G.ci(boot.ds(D).mean(0))
    return float(v.mean()), lo, hi


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def md_table(df, cols, heads=None, nd=2):
    heads = heads or cols
    L = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(fmt(float(v), nd))
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v))
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def cmd_aggregate(args):
    from scipy import stats
    G = gs()
    R, specs = load_cache()
    out, rep = {}, {}
    boot = G.Boot(len(UB9), B, BOOT_SEED)
    pools = {"frozen12": DEEP12, "recent": RECENT_IT, "frozen12+recent": DEEP12 + RECENT_IT}
    for k in SELECTORS:
        R["D_" + k] = R["ORACLE"] - R[k]

    # ---------------- coverage ----------------
    miss = [s for s in specs if not s["path"]]
    cov = []
    for m in METHODS_ALL:
        g = R[R.method == m]
        sp = [s for s in specs if s["method"] == m]
        cov.append(dict(method=m, stratum="frozen12" if m in DEEP12 else ("E5" if m in E5 else "E4"),
                        runs_listed=len(sp), runs_with_part=sum(bool(s["path"]) for s in sp),
                        runs_analysed=len(g), runs_with_scores=int(g.has_scores.sum()) if len(g) else 0,
                        source=("deep_rerun parts (trajectory only, no per-graph scores)" if m in OLD4 else
                                "ext parts + npz; A20: no checkpoint axis, item 4 only" if m in CHECKPOINT_FREE_RECENT
                                else "harness parts + npz" if m in NEW8 else "ext parts + npz"),
                        **{f"cov_{k}": int(np.isfinite(g[k]).sum()) if len(g) and m not in CHECKPOINT_FREE_RECENT else 0
                           for k in ["ORACLE"] + SELECTORS},
                        cov_perm=int(np.isfinite(g.perm_null_max_minus_last_mean).sum())
                        if len(g) and m not in CHECKPOINT_FREE_RECENT else 0,
                        cov_sizematch=int(np.isfinite(g.sm_valnormal_matched).sum()) if len(g) else 0))
    out["coverage"] = pd.DataFrame(cov)

    # ---------------- sanity: reproduce glad_stats ----------------
    F = pd.read_csv(FOLDS_CSV, low_memory=False)
    F = F[(F.hp == "ubgold") & F.dataset.isin(UB9) & F.base.isin(["oracle_ubgold", "last", "valnormal"])]
    F = F.pivot_table(index=["method", "dataset", "fold"], columns="base", values=["auroc", "epoch_selected"])
    san = []
    for m in DEEP12 + RECENT_IT:
        g = R[R.method == m].set_index(["method", "dataset", "fold"])
        for mine, theirs in (("ORACLE", "oracle_ubgold"), ("S1", "last"), ("S2", "valnormal")):
            j = g[[mine, "ep_" + mine]].join(F[("auroc", theirs)].rename("gs_auroc"), how="left") \
                .join(F[("epoch_selected", theirs)].rename("gs_epoch"), how="left")
            d = (j[mine] - j.gs_auroc).abs()
            de = ~((j["ep_" + mine] == j.gs_epoch) | (j["ep_" + mine].isna() & j.gs_epoch.isna()))
            san.append(dict(method=m, protocol=theirs, selector=mine, runs=len(j), runs_missing_in_glad_stats=int(j.gs_auroc.isna().sum()),
                            max_abs_auroc_diff=float(d.max()), runs_auroc_diff_gt_1e4=int((d > 1e-4).sum()),
                            runs_epoch_differs=int(de.sum())))
    S = pd.DataFrame(san)
    out["sanity"] = S
    sanity_pool = []
    for comp, a, b in (("ORACLE - VAL-NORMAL", "ORACLE", "S2"), ("ORACLE - LAST", "ORACLE", "S1")):
        D = cube(R, DEEP12, a) - cube(R, DEEP12, b)
        v, lo, hi = pool_ci(D, boot)
        Dn = cube(R, NEW8, a) - cube(R, NEW8, b)
        v8, lo8, hi8 = pool_ci(Dn, boot)
        sanity_pool.append(dict(comparison=comp, detectors="12 (all runs, trajectories)", value=v, ds_ci_lo=lo,
                                ds_ci_hi=hi))
        sanity_pool.append(dict(comparison=comp, detectors="8 with per-graph scores (NEW8)", value=v8, ds_ci_lo=lo8,
                                ds_ci_hi=hi8))
    P = pd.read_csv(POOLED_CSV)
    P = P[(P.set == "UB9") & (P.pool.isin(["frozen12", "recent(E5+E4)"]))]
    rep["report_pooled"] = P
    out["sanity_pooled"] = pd.DataFrame(sanity_pool)
    auc_check = float(R.auroc_check_maxdiff.max())

    # ---------------- item 1: selectors ----------------
    per, pooled = [], []
    s3_methods = {p: [m for m in ms if np.isfinite(cube(R, [m], "D_S3")).all()] for p, ms in pools.items()}
    for m in DEEP12 + RECENT_IT:
        for k in SELECTORS:
            D = cube(R, [m], "D_" + k)
            g = R[R.method == m]
            if not np.isfinite(D).all():
                per.append(dict(method=m, stratum="frozen12" if m in DEEP12 else "recent", selector=k,
                                selector_label=SEL_LABEL[k], complete=False, n_runs=int(np.isfinite(D).sum()),
                                note=("no per-graph scores (deep_rerun)" if k == "S3" and m in OLD4 else
                                      "no standardised held-out score recorded" if k == "S7" else "incomplete")))
                continue
            dm = D[0].mean(-1)
            lo, hi = G.ci(boot.ds(D)[0])
            flags = int((g["flag_" + k].fillna("").astype(str) != "").sum())
            per.append(dict(method=m, stratum="frozen12" if m in DEEP12 else "recent", selector=k,
                            selector_label=SEL_LABEL[k], complete=True, n_runs=45,
                            mean_oracle=float(cube(R, [m], "ORACLE")[0].mean(-1).mean()),
                            mean_selected=float(cube(R, [m], k)[0].mean(-1).mean()), delta=float(dm.mean()),
                            ds_ci_lo=lo, ds_ci_hi=hi, n_datasets_delta_gt0=int((dm > 0).sum()),
                            runs_flagged_A15=flags,
                            note=(S7_SOURCE[m][1] if k == "S7" else "")))
    per = pd.DataFrame(per)
    for pname, ms in pools.items():
        for k in SELECTORS:
            comp = [m for m in ms if np.isfinite(cube(R, [m], "D_" + k)).all()]
            if not comp:
                pooled.append(dict(pool=pname, selector=k, selector_label=SEL_LABEL[k], view="primary", n_methods=0))
                continue
            v, lo, hi = pool_ci(cube(R, comp, "D_" + k), boot)
            pooled.append(dict(pool=pname, selector=k, selector_label=SEL_LABEL[k], view="primary",
                               n_methods=len(comp), delta=v, ds_ci_lo=lo, ds_ci_hi=hi, ci_above_0=bool(lo > 0),
                               methods="|".join(comp)))
            cm = s3_methods[pname]
            if cm and set(cm) != set(comp) and set(cm) <= set(comp):
                v2, lo2, hi2 = pool_ci(cube(R, cm, "D_" + k), boot)
                pooled.append(dict(pool=pname, selector=k, selector_label=SEL_LABEL[k],
                                   view="coverage-matched (S3-covered detectors)", n_methods=len(cm), delta=v2,
                                   ds_ci_lo=lo2, ds_ci_hi=hi2, ci_above_0=bool(lo2 > 0), methods="|".join(cm)))
    pooled = pd.DataFrame(pooled)
    out["selectors"] = per
    out["selectors_pooled"] = pooled

    # ---------------- item 2a: permutation null ----------------
    perm, permp = [], []
    for m in DEEP12 + RECENT_IT:
        g = R[R.method == m]
        Dn = cube(R, [m], "perm_null_max_minus_last_mean")
        if not np.isfinite(Dn).all():
            perm.append(dict(method=m, stratum="frozen12" if m in DEEP12 else "recent", complete=False,
                             runs_with_scores=int(g.has_scores.sum()),
                             note="no per-graph scores (deep_rerun)" if m in OLD4 else "incomplete"))
            continue
        obs_l = float(cube(R, [m], "D_S1")[0].mean(-1).mean())
        obs_6 = float(cube(R, [m], "D_S6")[0].mean(-1).mean())
        n1 = float(Dn[0].mean(-1).mean())
        n2 = float(cube(R, [m], "perm_null_max_minus_mean_mean")[0].mean(-1).mean())
        q1 = float(cube(R, [m], "perm_null_max_minus_last_q95")[0].mean(-1).mean())
        perm.append(dict(method=m, stratum="frozen12" if m in DEEP12 else "recent", complete=True, runs_with_scores=45,
                         obs_oracle_minus_last=obs_l, null_max_minus_last=n1, ratio_null_to_obs=n1 / obs_l if obs_l else np.nan,
                         null_max_minus_last_q95_mean=q1,
                         obs_oracle_minus_S6=obs_6, null_max_minus_mean=n2,
                         ratio_nullmean_to_obs_oracle_minus_last=n2 / obs_l if obs_l else np.nan,
                         ratio_nullmean_to_obs_oracle_minus_S6=n2 / obs_6 if obs_6 else np.nan,
                         mean_n_eval=float(g.n_eval.mean())))
    perm = pd.DataFrame(perm)
    for pname, ms in pools.items():
        g = perm[perm.method.isin(ms) & perm.complete]
        if not len(g):
            continue
        comp = list(g.method)
        o1, o6 = g.obs_oracle_minus_last.mean(), g.obs_oracle_minus_S6.mean()
        n1, n2 = g.null_max_minus_last.mean(), g.null_max_minus_mean.mean()
        lo1, hi1 = G.ci(boot.ds(cube(R, comp, "perm_null_max_minus_last_mean")).mean(0))
        lo2, hi2 = G.ci(boot.ds(cube(R, comp, "perm_null_max_minus_mean_mean")).mean(0))
        _, olo, ohi = pool_ci(cube(R, comp, "D_S1"), boot)
        permp.append(dict(pool=pname, n_methods=len(comp), obs_oracle_minus_last=o1, obs_ds_ci_lo=olo, obs_ds_ci_hi=ohi,
                          null_max_minus_last=n1, null_ds_ci_lo=lo1, null_ds_ci_hi=hi1, ratio_null_to_obs=n1 / o1,
                          obs_oracle_minus_S6=o6, null_max_minus_mean=n2, null2_ds_ci_lo=lo2, null2_ds_ci_hi=hi2,
                          ratio_nullmean_to_obs_oracle_minus_last=n2 / o1, ratio_nullmean_to_obs_oracle_minus_S6=n2 / o6,
                          methods="|".join(comp)))
    out["permnull"] = perm
    out["permnull_pooled"] = pd.DataFrame(permp)

    # ---------------- item 2b: Spearman across runs ----------------
    sp = []
    for pname, ms in pools.items():
        g = R[R.method.isin(ms)].copy()
        g = g[np.isfinite(g.D_S1)]
        for xcol, xlab in (("n_eval", "number of evaluation points"), ("traj_sd", "SD of the test-AUROC trajectory")):
            gg = g[np.isfinite(g[xcol].astype(float))]
            rho, pv = stats.spearmanr(gg[xcol].astype(float), gg.D_S1)
            sp.append(dict(pool=pname, x=xlab, y="ORACLE - LAST (per run)", n_runs=len(gg), spearman_rho=float(rho),
                           p_value=float(pv), n_methods=gg.method.nunique()))
    out["spearman"] = pd.DataFrame(sp)

    # ---------------- item 3: dataset robustness ----------------
    lodo, forest, crossed = [], [], []
    DV = cube(R, DEEP12, "D_S2")
    DL = cube(R, DEEP12, "D_S1")
    assert np.isfinite(DV).all() and np.isfinite(DL).all()
    mV, mL = DV.mean(-1), DL.mean(-1)                                 # (12, 9) 5-fold means
    lodo.append(dict(left_out="none (all 9)", g1_mean_vs_valnormal=float(mV.mean()), g1_mean_vs_last=float(mL.mean()),
                     n_methods=12, n_datasets=9))
    for d, ds in enumerate(UB9):
        keep = [i for i in range(len(UB9)) if i != d]
        lodo.append(dict(left_out=ds, g1_mean_vs_valnormal=float(mV[:, keep].mean(1).mean()),
                         g1_mean_vs_last=float(mL[:, keep].mean(1).mean()), n_methods=12, n_datasets=8))
    fidx = boot.fold_idx                                              # (B, 9, 5) glad_stats fold resamples
    for d, ds in enumerate(UB9):
        r = dict(dataset=ds, n_methods=12)
        for lab, Dm in (("valnormal", DV), ("last", DL)):
            X = Dm[:, d, :]                                           # (12, 5)
            bs = X[:, fidx[:, d, :]].mean(-1).mean(0)                 # (B,)
            lo, hi = G.ci(bs)
            per_m = X.mean(-1)
            r.update({f"mean_vs_{lab}": float(per_m.mean()), f"fold_ci_lo_{lab}": lo, f"fold_ci_hi_{lab}": hi,
                      f"min_detector_{lab}": float(per_m.min()), f"max_detector_{lab}": float(per_m.max()),
                      f"n_detectors_gt0_{lab}": int((per_m > 0).sum())})
        forest.append(r)
    rng = np.random.default_rng(BOOT_SEED)
    det_idx = rng.integers(0, len(DEEP12), size=(B, len(DEEP12)))
    ds_idx = rng.integers(0, len(UB9), size=(B, len(UB9)))
    for lab, Mx, D in (("ORACLE - VAL-NORMAL", mV, DV), ("ORACLE - LAST", mL, DL)):
        bs = Mx[det_idx[:, :, None], ds_idx[:, None, :]].mean((1, 2))
        lo, hi = G.ci(bs)
        dlo, dhi = G.ci(boot.ds(D).mean(0))
        crossed.append(dict(comparison=lab, g1_mean=float(Mx.mean()), ds_boot_ci_lo=dlo, ds_boot_ci_hi=dhi,
                            crossed_ci_lo=lo, crossed_ci_hi=hi, B=B, seed=BOOT_SEED, n_methods=12, n_datasets=9))
    out["lodo"] = pd.DataFrame(lodo)
    out["forest"] = pd.DataFrame(forest)
    out["crossed"] = pd.DataFrame(crossed)

    # ---------------- item 4: size-matched ----------------
    fr = []
    for ds in UB9:
        for k in range(NFOLD):
            c = json.load(open(fold_cache_path(ds, k)))
            fr.append(dict(dataset=ds, fold=k, n_test=c["n_test"], n_anom=c["n_anom"], n_norm=c["n_norm"],
                           n_pairs=c["n_pairs"], frac_anom_matched=c["frac_anom_matched"],
                           max_match_dist=max(c["match_dist"]) if c["match_dist"] else np.nan,
                           sizeonly_auc_full=c["sizeonly_auc_full"], sizeonly_auc_grail0=c["sizeonly_auc_grail0"],
                           sizeonly_auc_matched=np.nan if c["sizeonly_auc_matched"] is None else c["sizeonly_auc_matched"],
                           smd_log_n_before=c["smd_before"]["log_n"], smd_log_m1_before=c["smd_before"]["log_m1"],
                           smd_density_before=c["smd_before"]["density"],
                           smd_log_n_after=(c["smd_after"] or {}).get("log_n", np.nan),
                           smd_log_m1_after=(c["smd_after"] or {}).get("log_m1", np.nan),
                           smd_density_after=(c["smd_after"] or {}).get("density", np.nan),
                           test_idx_sorted=c["test_idx_sorted"]))
    FR = pd.DataFrame(fr)
    FR["sizeonly_matches_grail0"] = (FR.sizeonly_auc_full - FR.sizeonly_auc_grail0).abs() < 1e-9
    dsum = FR.groupby("dataset", sort=False).agg(min_frac_matched=("frac_anom_matched", "min"),
                                                 mean_frac_matched=("frac_anom_matched", "mean"),
                                                 pairs=("n_pairs", "sum"), anomalies=("n_anom", "sum"),
                                                 sizeonly_full=("sizeonly_auc_full", "mean"),
                                                 sizeonly_matched=("sizeonly_auc_matched", "mean")).reset_index()
    dsum["reported"] = dsum.min_frac_matched >= MATCH_MIN
    rep_ds = list(dsum[dsum.reported].dataset)
    out["sizematch_folds"] = FR
    out["sizematch_datasets"] = dsum
    smm, smp = [], []
    sm_methods = NEW8 + RECENT_IT + sorted(CHECKPOINT_FREE_RECENT)
    for m in sm_methods:
        for ds in UB9:
            g = R[(R.method == m) & (R.dataset == ds)]
            if len(g) != 5 or not np.isfinite(g.sm_valnormal_full).all():
                continue
            smm.append(dict(method=m, stratum="frozen12" if m in DEEP12 else "recent", dataset=ds,
                            reported=ds in rep_ds, n_pairs_total=int(g.sm_n_pairs.sum()),
                            valnormal_full=g.sm_valnormal_full.mean(), valnormal_matched=g.sm_valnormal_matched.mean(),
                            oracle_full=g.sm_oracle_full.mean(), oracle_matched=g.sm_oracle_matched.mean()))
    SM = pd.DataFrame(smm)
    for c in ("valnormal", "oracle"):
        SM[f"{c}_change"] = SM[f"{c}_matched"] - SM[f"{c}_full"]
    SM["inflation_full"] = SM.oracle_full - SM.valnormal_full
    SM["inflation_matched"] = SM.oracle_matched - SM.valnormal_matched
    out["sizematch"] = SM
    so_rep = dsum[dsum.reported]
    for m in sm_methods:
        g = SM[(SM.method == m) & SM.reported]
        if len(g) != len(rep_ds) or not len(g):
            continue
        smp.append(dict(level="method", name=m, stratum="frozen12" if m in DEEP12 else "recent", n_datasets=len(g),
                        valnormal_full=g.valnormal_full.mean(), valnormal_matched=g.valnormal_matched.mean(),
                        oracle_full=g.oracle_full.mean(), oracle_matched=g.oracle_matched.mean(),
                        inflation_full=g.inflation_full.mean(), inflation_matched=g.inflation_matched.mean()))
    SMP = pd.DataFrame(smp)
    for pname, ms in (("frozen12 (8 with scores)", NEW8), ("recent", RECENT_IT)):
        g = SMP[SMP["name"].isin(ms)] if len(SMP) else SMP
        if len(g):
            smp.append(dict(level="pool", name=pname, n_methods=len(g), n_datasets=len(rep_ds),
                            valnormal_full=g.valnormal_full.mean(), valnormal_matched=g.valnormal_matched.mean(),
                            oracle_full=g.oracle_full.mean(), oracle_matched=g.oracle_matched.mean(),
                            inflation_full=g.inflation_full.mean(), inflation_matched=g.inflation_matched.mean()))
    for ds in UB9:                                        # per-dataset pool means (disaggregation of the above)
        for pname, ms in (("frozen12 (8 with scores)", NEW8), ("recent", RECENT_IT)):
            g = SM[SM.method.isin(ms) & (SM.dataset == ds)]
            if len(g):
                smp.append(dict(level="dataset pool", name=f"{ds}: {pname}", dataset=ds, reported=ds in rep_ds,
                                n_methods=len(g), n_datasets=1, valnormal_full=g.valnormal_full.mean(),
                                valnormal_matched=g.valnormal_matched.mean(), oracle_full=g.oracle_full.mean(),
                                oracle_matched=g.oracle_matched.mean(), inflation_full=g.inflation_full.mean(),
                                inflation_matched=g.inflation_matched.mean()))
        d0 = dsum[dsum.dataset == ds].iloc[0]
        smp.append(dict(level="dataset pool", name=f"{ds}: size-only reference", dataset=ds,
                        reported=bool(d0.reported), n_datasets=1, valnormal_full=d0.sizeonly_full,
                        valnormal_matched=d0.sizeonly_matched, oracle_full=d0.sizeonly_full,
                        oracle_matched=d0.sizeonly_matched))
    if len(so_rep):
        smp.append(dict(level="reference", name="size-only (LW Mahalanobis)", n_datasets=len(so_rep),
                        valnormal_full=so_rep.sizeonly_full.mean(), valnormal_matched=so_rep.sizeonly_matched.mean(),
                        oracle_full=so_rep.sizeonly_full.mean(), oracle_matched=so_rep.sizeonly_matched.mean()))
    out["sizematch_pooled"] = pd.DataFrame(smp)

    # ---------------- item 5: fidelity tiers ----------------
    fid = []
    for f in FIDELITY:
        facets = [f["code"]]
        if f["config"] == "incomplete released configuration":
            facets.append(f["config"])
        if f["repro"] == "not met":
            facets.append("reproduction not met")
        tier = max(facets, key=TIER_ORDER.index)
        alt = "reproduction not met" if f["repro"] in ("met (contested)", "not evaluable") else tier
        fid.append(dict(method=f["method"], stratum="E5" if f["method"] in E5 else "E4", tier=tier,
                        tier_alternative_reading=alt, code_facet=f["code"], config_facet=f["config"],
                        reproduction_check=f["repro"], code_evidence=f["code_evidence"],
                        config_evidence=f["config_evidence"], reproduction_evidence=f["repro_evidence"],
                        in_checkpoint_pool=f["method"] in RECENT_IT))
    FID = pd.DataFrame(fid)
    out["fidelity"] = FID
    not_met = list(FID[FID.tier == "reproduction not met"].method)
    not_met_alt = list(FID[FID.tier_alternative_reading == "reproduction not met"].method)
    fp = []
    for lab, drop in (("all 8 iterative recent methods (= REPORT.md recent(E5+E4))", []),
                      ("without 'reproduction not met' (primary tier reading)", not_met),
                      ("without 'reproduction not met' or contested/not evaluable (alternative reading)", not_met_alt)):
        ms = [m for m in RECENT_IT if m not in drop]
        for comp, k in (("ORACLE_ubgold - VAL-NORMAL", "D_S2"), ("ORACLE_ubgold - LAST", "D_S1")):
            v, lo, hi = pool_ci(cube(R, ms, k), boot)
            fp.append(dict(pool=lab, comparison=comp, n_methods=len(ms), mean_over_methods=v, ds_ci_lo=lo,
                           ds_ci_hi=hi, dropped="|".join(drop), methods="|".join(ms)))
    out["fidelity_pooled"] = pd.DataFrame(fp)

    # ---------------- run-level table ----------------
    out["runs"] = R
    for k, df in out.items():
        df.to_csv(OUT.format(k), index=False, float_format="%.6f")
    write_report(R, out, rep, dict(auc_check=auc_check, rep_ds=rep_ds, not_met=not_met, not_met_alt=not_met_alt,
                                   miss=miss, s3_methods=s3_methods))
    print(f"wrote {REPORT} and {len(out)} CSVs ({OUT.format('*')})")


# ----------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------
def env_versions():
    import platform
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    for mod in ("scipy", "sklearn"):
        try:
            v[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return v


def summary_sentences(out, ctx):
    """One conservative sentence per A32 item, generated from the computed tables (wording branches on the values)."""
    S = []
    pp = out["selectors_pooled"]
    pr = pp[pp.view == "primary"]
    f12, rc = pr[pr.pool == "frozen12"], pr[pr.pool == "recent"]
    allpos = bool((f12.n_methods > 0).all() and f12.ci_above_0.fillna(False).all())
    allpos_r = bool((rc.n_methods > 0).all() and rc.ci_above_0.fillna(False).all())
    n3 = int(f12[f12.selector == "S3"].n_methods.iloc[0])
    n7 = int(f12[f12.selector == "S7"].n_methods.iloc[0])
    s7 = f12[f12.selector == "S7"].iloc[0]
    f6 = f12[f12.selector != "S7"]
    lo_r, hi_r = f6.loc[f6.delta.idxmin()], f6.loc[f6.delta.idxmax()]
    S.append(("1", f"For the UB-GOLD detectors the pooled ORACLE − selector gap was {lo_r.delta:.2f} ({lo_r.selector}) "
                   f"to {hi_r.delta:.2f} ({hi_r.selector}) over S1–S6 (S3 on {n3} of the 12 detectors) and "
                   f"{s7.delta:.2f} for S7 (only {n7} detector records a standardised held-out score), and "
                   f"{'every' if allpos else 'not every'} pooled 95% CI of the seven selectors lay above 0 (recent "
                   f"methods: {rc.delta.min():.2f}–{rc.delta.max():.2f}, {'every' if allpos_r else 'not every'} CI "
                   f"above 0); the gap is therefore not specific to VAL-NORMAL or LAST among these pre-specified "
                   f"selectors, which does not show that no better label-free selector exists."))
    pn = out["permnull_pooled"].set_index("pool")
    perm = out["permnull"]
    pc = perm[perm.complete]
    nge = int((pc.ratio_null_to_obs >= 1.0).sum())
    a, b = pn.loc["frozen12"], pn.loc["recent"]
    S.append(("2(a)", f"Under permuted test labels, taking the maximum over checkpoints gave on average "
                      f"{a.null_max_minus_last:.2f} AUROC points over LAST for the {int(a.n_methods)} UB-GOLD detectors "
                      f"with stored scores (observed ORACLE − LAST {a.obs_oracle_minus_last:.2f}, ratio "
                      f"{a.ratio_null_to_obs:.2f}) and {b.null_max_minus_last:.2f} for the {int(b.n_methods)} recent "
                      f"methods (observed {b.obs_oracle_minus_last:.2f}, ratio {b.ratio_null_to_obs:.2f}), and for "
                      f"{nge} of {len(pc)} methods the null was at least as large as the observed gap, so a substantial "
                      f"part of the observed gap is of the size that test-label selection produces on noise alone "
                      f"(the null uses the plain maximum, which can only overstate the null of the patience-30 rule)."))
    sp = out["spearman"].set_index(["pool", "x"])
    r_ne = sp.loc[("frozen12", "number of evaluation points")]
    r_sd = sp.loc[("frozen12", "SD of the test-AUROC trajectory")]
    q_ne = sp.loc[("recent", "number of evaluation points")]
    q_sd = sp.loc[("recent", "SD of the test-AUROC trajectory")]
    def strength(r):
        a_ = abs(r.spearman_rho)
        return ("no clear association" if (a_ < 0.1 or r.p_value >= 0.05) else
                ("a weak association" if a_ < 0.3 else "a moderate association"))

    S.append(("2(b)", f"Across runs, ORACLE − LAST rose with the spread of the test-AUROC trajectory (Spearman ρ = "
                      f"{r_sd.spearman_rho:.2f} for UB-GOLD detectors, {q_sd.spearman_rho:.2f} for recent methods), "
                      f"and showed {strength(r_ne)} with the number of evaluation points for UB-GOLD detectors (ρ = "
                      f"{r_ne.spearman_rho:.2f}) and {strength(q_ne)} for recent methods (ρ = {q_ne.spearman_rho:.2f}); "
                      f"this is descriptive and mixes between- and within-method variation."))
    lo = out["lodo"].iloc[1:]
    cr = out["crossed"].set_index("comparison")
    fo = out["forest"]
    above = bool((lo.g1_mean_vs_valnormal >= 3.0).all() and (lo.g1_mean_vs_last >= 3.0).all())
    cv, cl = cr.loc["ORACLE - VAL-NORMAL"], cr.loc["ORACLE - LAST"]
    S.append(("3", f"Leaving out any one UB9 dataset kept the two G1 means within {lo.g1_mean_vs_valnormal.min():.2f}–"
                   f"{lo.g1_mean_vs_valnormal.max():.2f} (vs VAL-NORMAL) and {lo.g1_mean_vs_last.min():.2f}–"
                   f"{lo.g1_mean_vs_last.max():.2f} (vs LAST), {'all' if above else 'not all'} above 3.0, and the "
                   + (f"detector-mean inflation was positive on all 9 datasets for both comparisons (smallest on "
                      if (fo.mean_vs_valnormal > 0).all() and (fo.mean_vs_last > 0).all() else
                      f"detector-mean inflation was positive on {int((fo.mean_vs_valnormal > 0).sum())} (vs VAL-NORMAL) "
                      f"and {int((fo.mean_vs_last > 0).sum())} (vs LAST) of 9 datasets (smallest on ")
                   + f"{NAMES.get(fo.loc[fo.mean_vs_valnormal.idxmin(), 'dataset'], fo.loc[fo.mean_vs_valnormal.idxmin(), 'dataset'])}, "
                   f"{fo.mean_vs_valnormal.min():.2f}); treating the detectors as a sample as well widens the 95% CIs "
                   f"to [{cv.crossed_ci_lo:.2f}, {cv.crossed_ci_hi:.2f}] and [{cl.crossed_ci_lo:.2f}, "
                   f"{cl.crossed_ci_hi:.2f}], so the claim should be stated for the evaluated detectors across the UB9 "
                   f"datasets."))
    ds_ = out["sizematch_datasets"]
    rep_ = ds_[ds_.reported]
    not_rep = list(ds_[~ds_.reported].dataset)
    smp = out["sizematch_pooled"]
    pl = smp[(smp.level == "pool") & (smp["name"].str.startswith("frozen12"))]
    if len(rep_) and len(pl):
        p0 = pl.iloc[0]
        S.append(("4", f"{len(rep_)} of 9 UB9 datasets met the 30% matching rule ({', '.join(NAMES.get(d, d) for d in not_rep)} "
                       f"did not); on their matched subsets the size-only reference (the matching check) gave "
                       + ", ".join(f"{NAMES.get(d, d)} {v:.1f}" for d, v in zip(rep_.dataset, rep_.sizeonly_matched))
                       + f" AUROC, and the mean AUROC of the 8 UB-GOLD detectors with scores fell "
                       f"from {p0.valnormal_full:.1f} to {p0.valnormal_matched:.1f} at VAL-NORMAL and from "
                       f"{p0.oracle_full:.1f} to {p0.oracle_matched:.1f} at ORACLE (ORACLE − VAL-NORMAL "
                       f"{p0.inflation_full:.1f} → {p0.inflation_matched:.1f}); matched subsets are not random samples "
                       f"of the test sets, so these changes are descriptive."))
    else:
        S.append(("4", "No dataset met the 30% matching rule; no size-matched detector result is reported."))
    fp = out["fidelity_pooled"]
    full = fp[fp.pool.str.startswith("all")].set_index("comparison")
    prim = fp[fp.pool.str.contains("primary")].set_index("comparison")
    alt = fp[fp.pool.str.contains("alternative")].set_index("comparison")
    kv, kl = "ORACLE_ubgold - VAL-NORMAL", "ORACLE_ubgold - LAST"
    S.append(("5", f"Only {', '.join(ctx['not_met']) or 'no method'} falls in the 'reproduction not met' tier; without "
                   f"it the pooled recent-method inflation is {prim.loc[kv, 'mean_over_methods']:.2f} "
                   f"[{prim.loc[kv, 'ds_ci_lo']:.2f}, {prim.loc[kv, 'ds_ci_hi']:.2f}] vs VAL-NORMAL and "
                   f"{prim.loc[kl, 'mean_over_methods']:.2f} [{prim.loc[kl, 'ds_ci_lo']:.2f}, "
                   f"{prim.loc[kl, 'ds_ci_hi']:.2f}] vs LAST (all eight: {full.loc[kv, 'mean_over_methods']:.2f} and "
                   f"{full.loc[kl, 'mean_over_methods']:.2f}), and also dropping "
                   f"{' and '.join(m for m in ctx['not_met_alt'] if m not in ctx['not_met'])} (contested or not "
                   f"evaluable) gives {alt.loc[kv, 'mean_over_methods']:.2f} [{alt.loc[kv, 'ds_ci_lo']:.2f}, "
                   f"{alt.loc[kv, 'ds_ci_hi']:.2f}] and {alt.loc[kl, 'mean_over_methods']:.2f} "
                   f"[{alt.loc[kl, 'ds_ci_lo']:.2f}, {alt.loc[kl, 'ds_ci_hi']:.2f}]."))
    return S


def paper_notes(out, ctx):
    """Issues found while computing A32 (the coordinator appends them to PAPER_NOTES.md; this script does not)."""
    N = []
    R = out["runs"]
    c = R[R.method == "CVTGAD"]
    early = float((c.ep_S7 <= 0.1 * c.num_epoch).mean()) if len(c) else float("nan")
    N.append("S7 coverage (A32 item 1): `holdout_score_norm` is recorded only for CVTGAD (harness/run_eval.py, "
             "fit_cvtgad); the other harness detectors and the old-4 deep_rerun runs record no standardised held-out "
             "score, so S7 is pooled over 1 UB-GOLD detector (+ GLADMamba, SEGO, GLASS, where S7 = S2 by construction). "
             f"CVTGAD's S7 picks a point in the first 10% of training in {100 * early:.0f}% of its runs, the same "
             "training-statistics normalisation artefact that A14 describes for GLADMamba. Status: verified "
             "(eval_posthoc_runs.csv).")
    pc = out["permnull"]
    pc = pc[pc.complete]
    ge = pc[pc.ratio_null_to_obs >= 1.0]
    N.append("Permutation null (A32 2a): for " + ", ".join(f"{r.method} ({r.ratio_null_to_obs:.2f})" for r in ge.itertuples())
             + " the null max − LAST is at least the observed ORACLE − LAST; the largest nulls occur for the methods "
             "with the most evaluation points (" + ", ".join(f"{r.method} {r.mean_n_eval:.0f} points: {r.null_max_minus_last:.2f}"
                                                       for r in pc.sort_values('null_max_minus_last').tail(2).itertuples())
             + "). A reviewer may read the inflation of these methods as selection on noise. Status: verified "
             "(eval_posthoc_permnull.csv).")
    ds_ = out["sizematch_datasets"]
    FR = out["sizematch_folds"]
    mp = (np.minimum(FR.n_anom, FR.n_norm) / FR.n_anom).groupby(FR.dataset).min()
    imp = [d for d in mp.index if mp[d] < MATCH_MIN]
    N.append("Size matching (A32 4): " + ", ".join(f"{d} (max possible {100 * mp[d]:.0f}% matched)" for d in imp)
             + " cannot reach the 30% rule because normals are the minority of their UB-GOLD test folds (UB-GOLD's "
             "anomaly-class convention, comment #7); AIDS fails because its anomalies and normals barely overlap in "
             f"size (size-only AUROC {float(ds_.set_index('dataset').loc['AIDS', 'sizeonly_full']):.1f} on the full test "
             "sets). Status: verified (eval_posthoc_sizematch_folds.csv).")
    far = ds_[ds_.reported & ((ds_.sizeonly_matched - 50).abs() > 5)]
    if len(far):
        fx = FR[FR.dataset.isin(far.dataset)]
        smax = float(fx[["smd_log_n_after", "smd_log_m1_after", "smd_density_after"]].abs().max().max())
        N.append("Matching check (A32 4): on " + ", ".join(f"{d} the size-only reference keeps AUROC {v:.1f}"
                                                         for d, v in zip(far.dataset, far.sizeonly_matched))
                 + f" on the matched subset although every fold's |standardised mean difference| of log n, log(m+1) "
                 f"and density is at most {smax:.2f} after matching; the 0.2 caliper on (log n, log(m+1)) leaves "
                 "residual size information that the Ledoit-Wolf Mahalanobis reference (two-sided distance from the "
                 "training-normal centre) still uses. Size-matched results on that dataset are not size-free. "
                 "Status: verified (eval_posthoc_sizematch_folds.csv).")
    cr = out["crossed"].set_index("comparison")
    if cr.loc["ORACLE - VAL-NORMAL", "crossed_ci_lo"] < 3.0:
        N.append(f"Crossed detector × dataset bootstrap (A32 3): the 95% CI of ORACLE − VAL-NORMAL is "
                 f"[{cr.loc['ORACLE - VAL-NORMAL', 'crossed_ci_lo']:.2f}, {cr.loc['ORACLE - VAL-NORMAL', 'crossed_ci_hi']:.2f}], "
                 "its lower end below the G1 threshold of 3.0; the paper should state the inferential target (the "
                 "evaluated detectors across UB9) and not generalise to other detectors. Status: verified "
                 "(eval_posthoc_crossed.csv).")
    N.append("Reproduction status of GLADPro (A32 5): ext/GLADPro/NOTES.md calls the A10 §4 criterion met (COX2, ENZYMES "
             "vs M-GLAD), but both matches are on datasets whose anomaly or normal classes likely differ, and the "
             "Appendix B table caption says such datasets are not counted; under that rule GLADPro meets 0 of 3. The "
             "table entry and the caption should be made consistent. Status: verified (app_ports.tex tab:ports, "
             "GLADPro NOTES reproduction section).")
    return N


def write_report(R, out, rep, ctx):
    G = gs()
    L = []
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    L += ["# Post hoc robustness analyses (A32): report (AUTO-GENERATED)", "",
          f"Generated {now} by `prelim/evalpaper/stats/posthoc_robustness.py` (sha256 `{sha256_file(osp.abspath(__file__))}`). "
          f"Do not edit by hand. Specification: PREREG_AMENDMENTS.md A32 (written 2026-10-14 20:55:17 EDT, before any "
          f"of these analyses was computed). **All analyses here are post hoc robustness or mechanism analyses, not "
          f"confirmatory tests, and none changes G1, G2 or A9.**", "",
          f"Frozen `stats/glad_stats.py` imported read-only (sha256 `{sha256_file(GLAD_STATS)}`, equal to FROZEN.md). "
          f"AUROC × 100. Dataset-bootstrap CIs: 95% percentile, B = {B}, glad_stats `Boot` with seed {BOOT_SEED} "
          f"(the same dataset resamples as REPORT.md). Environment: {env_versions()}. Cache: `{CACHE}` "
          f"(one JSON per run with the sha256 of its part JSON and score npz).", ""]
    # sanity
    sp = out["sanity_pooled"]
    S = out["sanity"]
    rp = rep["report_pooled"]

    def rpv(pool, comp):
        x = rp[(rp.pool == pool) & (rp.comparison == comp)]
        return (float(x.mean_over_methods.iloc[0]), float(x.ds_ci_lo.iloc[0]), float(x.ds_ci_hi.iloc[0])) if len(x) \
            else (np.nan,) * 3

    L += ["## 0. Sanity check against glad_stats (REPORT.md)", ""]
    a = rpv("frozen12", "checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL")
    b = rpv("frozen12", "checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST")
    x12 = sp[sp.detectors.str.startswith("12")].set_index("comparison")
    x8 = sp[sp.detectors.str.startswith("8")].set_index("comparison")
    L += [f"- Re-derived from the parts trajectories with glad_stats' own selection functions, 12 UB-GOLD detectors, "
          f"UB9: ORACLE − VAL-NORMAL = **{x12.loc['ORACLE - VAL-NORMAL', 'value']:.4f}** "
          f"[{x12.loc['ORACLE - VAL-NORMAL', 'ds_ci_lo']:.2f}, {x12.loc['ORACLE - VAL-NORMAL', 'ds_ci_hi']:.2f}] "
          f"(REPORT.md: {a[0]:.4f} [{a[1]:.2f}, {a[2]:.2f}]); ORACLE − LAST = **{x12.loc['ORACLE - LAST', 'value']:.4f}** "
          f"[{x12.loc['ORACLE - LAST', 'ds_ci_lo']:.2f}, {x12.loc['ORACLE - LAST', 'ds_ci_hi']:.2f}] "
          f"(REPORT.md: {b[0]:.4f} [{b[1]:.2f}, {b[2]:.2f}]).",
          f"- Run by run, against `eval_leaderboard_v2_folds.csv` (oracle_ubgold, last, valnormal; written with 4 "
          f"decimals): max |ΔAUROC| = {S.max_abs_auroc_diff.max():.2e}, runs with |Δ| > 1e-4: "
          f"{int(S.runs_auroc_diff_gt_1e4.sum())}, runs whose selected epoch differs: {int(S.runs_epoch_differs.sum())}, "
          f"runs missing in glad_stats: {int(S.runs_missing_in_glad_stats.sum())} (12 UB-GOLD detectors and 8 iterative "
          f"recent methods, {len(R[R.method.isin(DEEP12 + RECENT_IT)])} runs; ProtoGLAD has no checkpoint axis).",
          f"- On the runs that have per-graph scores (the 8 harness detectors; SIGNET, OCGTL, OCGIN, GLocalKD use deep_rerun "
          f"runs without scores) the same pooled values are ORACLE − VAL-NORMAL {x8.loc['ORACLE - VAL-NORMAL', 'value']:.2f} "
          f"[{x8.loc['ORACLE - VAL-NORMAL', 'ds_ci_lo']:.2f}, {x8.loc['ORACLE - VAL-NORMAL', 'ds_ci_hi']:.2f}] and "
          f"ORACLE − LAST {x8.loc['ORACLE - LAST', 'value']:.2f} [{x8.loc['ORACLE - LAST', 'ds_ci_lo']:.2f}, "
          f"{x8.loc['ORACLE - LAST', 'ds_ci_hi']:.2f}]; the difference from 4.13 / 5.12 is coverage (4 detectors fewer), "
          f"not computation.",
          f"- Rank-sum AUROC from the stored per-graph scores reproduces the recorded trajectory AUROC at every finite "
          f"point of every scored run: max |Δ| = {ctx['auc_check']:.2e} AUROC points.", ""]
    # coverage
    cov = out["coverage"]
    L += ["## Coverage", "",
          "Primary set (A32): UB9 × 'ubgold' × 5 folds; 12 UB-GOLD detectors and the recent methods. Runs without "
          "per-graph scores are listed, never imputed.", "",
          md_table(cov, ["method", "stratum", "runs_with_part", "runs_analysed", "runs_with_scores", "cov_S1", "cov_S2",
                         "cov_S3", "cov_S4", "cov_S5", "cov_S6", "cov_S7", "cov_perm", "cov_sizematch", "source"],
                   ["method", "stratum", "parts", "analysed", "with scores", "S1", "S2", "S3", "S4", "S5", "S6", "S7",
                    "perm. null", "size-match", "source"], nd=0), "",
          "- Runs without per-graph scores (not imputed): SIGNET, OCGTL, OCGIN, GLocalKD × all 9 UB9 datasets × 5 folds "
          "= 180 deep_rerun runs (GPU runs of the pilot; trajectories only). They enter items 1 (S1, S2, S4, S5, S6), "
          "2(b) and 3, but not S3, S7, 2(a) or 4.",
          f"- Runs without a part file in the canonical list: {len(ctx['miss'])}.",
          "- ProtoGLAD (no checkpoint axis, A20) is outside items 1–3 and 2(a); its scores enter item 4.", ""]
    # item 1
    per, pooled = out["selectors"], out["selectors_pooled"]
    L += ["## 1. Label-free selector robustness", "",
          "Δ_s = ORACLE (UB-GOLD rule) − selector s, per run; per detector the mean over UB9 of 5-fold means; pooled = "
          "mean over detectors with complete UB9 coverage for s. 95% dataset-bootstrap CIs.", ""]
    pr = pooled[pooled.view == "primary"].copy()
    for pname in ("frozen12", "recent", "frozen12+recent"):
        g = pr[pr.pool == pname]
        L += [f"**Pool {pname}**", "", md_table(g, ["selector", "selector_label", "n_methods", "delta", "ds_ci_lo",
                                                     "ds_ci_hi", "ci_above_0"],
                                                 ["s", "selector", "detectors", "Δ_s", "CI lo", "CI hi", "CI > 0"]), ""]
        cm = pooled[(pooled.pool == pname) & (pooled.view != "primary")]
        if len(cm):
            L += [f"Coverage-matched view (same {int(cm.n_methods.iloc[0])} detectors as S3: "
                  f"{cm.methods.iloc[0].replace('|', ', ')}):", "",
                  md_table(cm, ["selector", "n_methods", "delta", "ds_ci_lo", "ds_ci_hi"],
                           ["s", "detectors", "Δ_s", "CI lo", "CI hi"]), ""]
    f12 = pr[pr.pool == "frozen12"]
    allpos = bool(f12.ci_above_0.fillna(False).all() and len(f12) == 7 and (f12.n_methods > 0).all())
    rc = pr[pr.pool == "recent"]
    allpos_r = bool(rc.ci_above_0.fillna(False).all() and len(rc) == 7 and (rc.n_methods > 0).all())
    L += [f"A32 rule for the summary sentence (all seven pooled Δ_s have CIs above 0): UB-GOLD detectors "
          f"**{'met' if allpos else 'NOT met'}**; recent methods **{'met' if allpos_r else 'NOT met'}**. S7 is pooled "
          f"over {int(f12[f12.selector == 'S7'].n_methods.iloc[0])} UB-GOLD detector(s) and "
          f"{int(rc[rc.selector == 'S7'].n_methods.iloc[0])} recent methods only (see coverage).", ""]
    pw = per[per.complete].pivot_table(index="method", columns="selector", values="delta")
    pw = pw.reindex([m for m in DEEP12 + RECENT_IT if m in pw.index]).reset_index()
    L += ["Per detector Δ_s (CIs in `eval_posthoc_selectors.csv`; '–' = selector not available):", "",
          md_table(pw.fillna(np.nan), ["method"] + [k for k in SELECTORS if k in pw.columns]), ""]
    neg = per[per.complete & (per.ds_ci_hi < 0)]
    nz = per[per.complete & (per.ds_ci_lo <= 0) & (per.ds_ci_hi >= 0)]
    L += [f"Per-detector CIs entirely below 0: {', '.join(f'{r.method}/{r.selector} ({r.delta:.2f})' for r in neg.itertuples()) or 'none'}. "
          f"Per-detector CIs including 0: {', '.join(f'{r.method}/{r.selector} ({r.delta:.2f})' for r in nz.itertuples()) or 'none'}.", ""]
    rep["item1"] = (pr, allpos, allpos_r)
    # item 2
    perm, permp = out["permnull"], out["permnull_pooled"]
    L += ["## 2. Selection on noise", "", "### 2(a) Permutation null", "",
          f"Per run, the test labels are permuted {N_PERM} times (prevalence kept; seed 20261014 + run index); in each "
          "permutation, max over evaluation points of AUROC minus AUROC at LAST, and max minus the trajectory mean. "
          "Values are averaged over permutations, then aggregated like the observed values.", "",
          md_table(perm[perm.complete], ["method", "obs_oracle_minus_last", "null_max_minus_last", "ratio_null_to_obs",
                                         "null_max_minus_last_q95_mean", "obs_oracle_minus_S6", "null_max_minus_mean",
                                         "ratio_nullmean_to_obs_oracle_minus_S6", "mean_n_eval"],
                   ["method", "obs. ORACLE − LAST", "null max − LAST", "ratio", "null q95 (mean)",
                    "obs. ORACLE − S6", "null max − mean", "ratio to obs. ORACLE − S6", "eval points"]), "",
          md_table(permp, ["pool", "n_methods", "obs_oracle_minus_last", "null_max_minus_last", "null_ds_ci_lo",
                           "null_ds_ci_hi", "ratio_null_to_obs", "obs_oracle_minus_S6", "null_max_minus_mean",
                           "ratio_nullmean_to_obs_oracle_minus_last", "ratio_nullmean_to_obs_oracle_minus_S6"],
                   ["pool", "detectors", "obs. ORACLE − LAST", "null max − LAST", "CI lo", "CI hi", "ratio",
                    "obs. ORACLE − S6", "null max − mean", "ratio to obs. ORACLE − LAST", "ratio to obs. ORACLE − S6"]),
          "", "Not computable (no per-graph scores): SIGNET, OCGTL, OCGIN, GLocalKD on UB9.", "",
          "### 2(b) Opportunity and spread (descriptive)", "",
          md_table(out["spearman"], ["pool", "x", "n_runs", "n_methods", "spearman_rho", "p_value"],
                   ["pool", "x (vs observed ORACLE − LAST per run)", "runs", "methods", "Spearman ρ", "p"], nd=3), "",
          "The number of evaluation points is set by the configuration (method and dataset), so these correlations mix "
          "between-method and within-method variation; they are descriptive only.", ""]
    # item 3
    L += ["## 3. Dataset robustness and inference target", "", "### Leave one dataset out (12 UB-GOLD detectors)", "",
          md_table(out["lodo"], ["left_out", "g1_mean_vs_valnormal", "g1_mean_vs_last"],
                   ["left out", "ORACLE − VAL-NORMAL", "ORACLE − LAST"]), ""]
    lo_ = out["lodo"].iloc[1:]
    L += [f"Range over the 9 leave-one-out values: vs VAL-NORMAL {lo_.g1_mean_vs_valnormal.min():.2f}–"
          f"{lo_.g1_mean_vs_valnormal.max():.2f}; vs LAST {lo_.g1_mean_vs_last.min():.2f}–{lo_.g1_mean_vs_last.max():.2f}.",
          "", "### Per-dataset detector-mean inflation (forest plot `paper/figures/posthoc_forest.pdf`)", "",
          md_table(out["forest"], ["dataset", "mean_vs_valnormal", "fold_ci_lo_valnormal", "fold_ci_hi_valnormal",
                                   "min_detector_valnormal", "max_detector_valnormal", "n_detectors_gt0_valnormal",
                                   "mean_vs_last", "fold_ci_lo_last", "fold_ci_hi_last", "min_detector_last",
                                   "max_detector_last", "n_detectors_gt0_last"],
                   ["dataset", "vs VAL-NORMAL", "fold CI lo", "fold CI hi", "min det.", "max det.", "det. > 0",
                    "vs LAST", "fold CI lo", "fold CI hi", "min det.", "max det.", "det. > 0"]), "",
          "Fold-bootstrap CIs hold the 12 detectors fixed and resample the 5 folds (optimistic, as in REPORT.md).", "",
          "### Crossed detector × dataset bootstrap", "",
          md_table(out["crossed"], ["comparison", "g1_mean", "ds_boot_ci_lo", "ds_boot_ci_hi", "crossed_ci_lo",
                                    "crossed_ci_hi"],
                   ["comparison", "G1 mean", "dataset-bootstrap CI lo", "hi", "crossed CI lo", "hi"]), "",
          "**Inferential target (for the paper):** the 12 evaluated UB-GOLD detectors, as configured by UB-GOLD, under "
          "variation across the UB9 datasets (dataset bootstrap). The crossed bootstrap additionally treats the detectors "
          "as a sample, which is a wider target than the one the main analysis claims.", ""]
    # item 4
    ds_ = out["sizematch_datasets"]
    FR = out["sizematch_folds"]
    L += ["## 4. Size-matched evaluation on the original test sets", "",
          f"1:1 nearest neighbour on standardised (log n, log(m+1)) within each test fold, caliper {CALIPER}, without "
          f"replacement, anomalies in test-index order. A dataset is reported only if ≥ {int(100 * MATCH_MIN)}% of its "
          f"anomalies are matched in every fold. Size-only reference = Ledoit–Wolf Mahalanobis on (log n, log(m+1), "
          f"density) fitted on the fold's normal training graphs; on the full test sets it reproduces grail0_folds.csv "
          f"in {int(FR.sizeonly_matches_grail0.sum())}/45 folds (|Δ| < 1e-9).", "",
          md_table(ds_, ["dataset", "anomalies", "pairs", "min_frac_matched", "mean_frac_matched", "reported",
                         "sizeonly_full", "sizeonly_matched"],
                   ["dataset", "anomalies (5 folds)", "pairs (5 folds)", "min frac. matched", "mean frac. matched",
                    "reported", "size-only AUROC, full", "size-only AUROC, matched"]), ""]
    bal = FR[FR.dataset.isin(ctx["rep_ds"])]
    if len(bal):
        L += [f"Balance on the reported datasets (mean |SMD| over folds, before → after): log n "
              f"{bal.smd_log_n_before.abs().mean():.2f} → {bal.smd_log_n_after.abs().mean():.2f}; log(m+1) "
              f"{bal.smd_log_m1_before.abs().mean():.2f} → {bal.smd_log_m1_after.abs().mean():.2f}; density "
              f"{bal.smd_density_before.abs().mean():.2f} → {bal.smd_density_after.abs().mean():.2f}.", ""]
    smp = out["sizematch_pooled"]
    if len(smp):
        top = smp[smp.level != "dataset pool"]
        dsp = smp[smp.level == "dataset pool"]
        L += ["Mean over the reported datasets (detector AUROC at its VAL-NORMAL and ORACLE points; full test set vs "
              "matched subset). Pools: the 8 UB-GOLD detectors with per-graph scores; the 8 iterative recent methods "
              "(ProtoGLAD, whose VAL-NORMAL and ORACLE points coincide, is shown as a row but not pooled, A20):", "",
              md_table(top, ["level", "name", "n_datasets", "valnormal_full", "valnormal_matched", "oracle_full",
                             "oracle_matched", "inflation_full", "inflation_matched"],
                       ["level", "detector / pool", "datasets", "VAL-NORMAL full", "VAL-NORMAL matched", "ORACLE full",
                        "ORACLE matched", "ORACLE − VN full", "ORACLE − VN matched"]), "",
              "Per dataset (pool means; datasets below the 30% rule are shown for completeness only and are not "
              "reported results):", "",
              md_table(dsp, ["name", "reported", "valnormal_full", "valnormal_matched", "oracle_full",
                             "oracle_matched"],
                       ["dataset: pool", "reported", "VAL-NORMAL full", "VAL-NORMAL matched", "ORACLE full",
                        "ORACLE matched"]), ""]
    else:
        L += ["No dataset meets the reporting threshold; no detector result is reported (A32 item 4).", ""]
    L += ["Per method × dataset values (all datasets, with the `reported` flag) are in `eval_posthoc_sizematch.csv`; "
          "per-fold matching statistics in `eval_posthoc_sizematch_folds.csv`.", ""]
    # item 5
    FID, FP = out["fidelity"], out["fidelity_pooled"]
    L += ["## 5. Reproduction-fidelity sensitivity", "",
          md_table(FID, ["method", "tier", "tier_alternative_reading", "code_facet", "config_facet", "reproduction_check"],
                   ["method", "tier", "tier (alternative reading)", "code", "configuration", "A10 §4 check"]), "",
          "Evidence for each facet is in `eval_posthoc_fidelity.csv` (Appendix B tab:ports and ext/<M>/NOTES.md).", "",
          md_table(FP, ["pool", "comparison", "n_methods", "mean_over_methods", "ds_ci_lo", "ds_ci_hi", "dropped"],
                   ["pool", "comparison", "methods", "mean", "CI lo", "CI hi", "dropped"]), ""]
    rr = rpv("recent(E5+E4)", "checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL")
    rl = rpv("recent(E5+E4)", "checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST")
    L += [f"Sanity: the full recent pool equals REPORT.md recent(E5+E4), UB-GOLD-rule ORACLE: − VAL-NORMAL "
          f"{rr[0]:.4f} [{rr[1]:.2f}, {rr[2]:.2f}], − LAST {rl[0]:.4f} [{rl[1]:.2f}, {rl[2]:.2f}].", ""]
    # one-sentence results, inserted after the header
    summ = summary_sentences(out, ctx)
    hdr_end = L.index("## 0. Sanity check against glad_stats (REPORT.md)")
    L[hdr_end:hdr_end] = ["## Summary: one sentence per A32 item (post hoc, descriptive; numbers from the tables below)",
                          ""] + [f"- **Item {k}.** {v}" for k, v in summ] + [""]
    L += ["## Points for PAPER_NOTES.md (found while computing; for the coordinator to append)", ""] + \
        [f"- {t}" for t in paper_notes(out, ctx)] + [""]
    # interpretations
    L += ["## Interpretations where A32 is silent", ""] + [f"{i + 1}. {t}" for i, t in enumerate(INTERPRETATIONS)] + [""]
    L += ["## Outputs", "", "`prelim/results/eval_posthoc_{" + ",".join(sorted(out)) + "}.csv`; figures "
          "`prelim/evalpaper/paper/figures/posthoc_forest.{pdf,png}` and `posthoc_selectors.{pdf,png}` (written by "
          "`posthoc_robustness.py figures` with the paper-figs environment).", ""]
    rep["L"] = L
    with open(REPORT, "w") as fh:
        fh.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------------------------------------------
# figures (paper-figs environment: numpy, pandas, matplotlib only)
# ----------------------------------------------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
NAMES = {"GraphCL_IF": "GraphCL-IF", "GraphCL_OCSVM": "GraphCL-OCSVM", "InfoGraph_IF": "InfoGraph-IF",
         "InfoGraph_OCSVM": "InfoGraph-OCSVM", "SUL-GUL": "SUL/GUL", "PROTEINS_full": "PROTEINS",
         "IMDB-BINARY": "IMDB-B", "REDDIT-BINARY": "REDDIT-B"}


def cmd_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False})

    def save(fig, stem):
        for ext in ("pdf", "png"):
            fig.savefig(osp.join(FIGDIR, f"{stem}.{ext}"), bbox_inches="tight", dpi=220)
        plt.close(fig)

    # ---- forest plot: per-dataset detector-mean inflation + leave-one-dataset-out ----
    fo = pd.read_csv(OUT.format("forest"))
    lo = pd.read_csv(OUT.format("lodo"))
    cr = pd.read_csv(OUT.format("crossed")).set_index("comparison")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.75, 2.75), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.50))
    n = len(fo)
    off = {"valnormal": 0.17, "last": -0.17}
    mk = {"valnormal": "o", "last": "^"}
    key = {"valnormal": "ORACLE - VAL-NORMAL", "last": "ORACLE - LAST"}

    def marker(axx, x, y, c, s=18):
        axx.scatter(x, y, marker=mk[c], s=s, zorder=4, facecolor=BLUE if c == "valnormal" else "white",
                    edgecolor=BLUE, linewidth=0.9)

    for c in ("valnormal", "last"):
        y = np.arange(n)[::-1] + off[c]
        ax.scatter(fo[f"min_detector_{c}"], y, marker="|", s=12, color=MUTED, linewidth=0.7, zorder=2)
        ax.scatter(fo[f"max_detector_{c}"], y, marker="|", s=12, color=MUTED, linewidth=0.7, zorder=2)
        ax.hlines(y, fo[f"fold_ci_lo_{c}"], fo[f"fold_ci_hi_{c}"], color=BLUE, linewidth=1.0, zorder=3)
        marker(ax, fo[f"mean_vs_{c}"], y, c)
    ax.axvline(0, color=INK, linewidth=0.6, zorder=1)
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([NAMES.get(d, d) for d in fo.dataset])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Detector-mean inflation (AUROC points)")
    ax.set_title("a  Per dataset (mean of 12 UB-GOLD detectors)", loc="left", fontsize=7, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    # panel b: all 9 datasets (dataset- and crossed-bootstrap CIs) and the 9 leave-one-out values
    sub = lo[~lo.left_out.str.startswith("none")].reset_index(drop=True)
    full = lo[lo.left_out.str.startswith("none")].iloc[0]
    rows = ["All 9 datasets"] + ["without " + NAMES.get(d, d) for d in sub.left_out]
    yy = np.arange(len(rows))[::-1]
    for c in ("valnormal", "last"):
        y0 = yy[0] + off[c]
        bx.hlines(y0, cr.loc[key[c], "crossed_ci_lo"], cr.loc[key[c], "crossed_ci_hi"], color=BLUE, linewidth=0.8,
                  alpha=0.55, zorder=2)
        bx.hlines(y0, cr.loc[key[c], "ds_boot_ci_lo"], cr.loc[key[c], "ds_boot_ci_hi"], color=BLUE, linewidth=2.2,
                  zorder=3)
        marker(bx, [full[f"g1_mean_vs_{c}"]], [y0], c, s=20)
        marker(bx, sub[f"g1_mean_vs_{c}"], yy[1:] + off[c], c)
    bx.axhline(yy[0] - 0.55, color=GRID, linewidth=0.8, zorder=0)
    bx.axvline(3.0, color=MUTED, linewidth=0.7, linestyle=(0, (2, 1.5)), zorder=1)
    bx.set_yticks(yy)
    bx.set_yticklabels(rows)
    bx.set_ylim(-0.6, len(rows) - 0.4)
    bx.set_xlabel("G1 mean inflation (AUROC points)")
    bx.set_title("b  Robustness of the two G1 means", loc="left", fontsize=7, color=INK)
    bx.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.set_xlim(min(2.5, cr.crossed_ci_lo.min() - 0.3), max(sub.g1_mean_vs_last.max(), cr.crossed_ci_hi.max()) + 0.3)
    h = [Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=4, label="ORACLE − VAL-NORMAL"),
         Line2D([], [], marker="^", markerfacecolor="white", markeredgecolor=BLUE, linestyle="", markersize=4,
                label="ORACLE − LAST"),
         Line2D([], [], color=BLUE, linewidth=1.0, label="Fold-bootstrap 95% CI (a)"),
         Line2D([], [], marker="|", color=MUTED, linestyle="", markersize=5, label="Detector min / max (a)"),
         Line2D([], [], color=BLUE, linewidth=2.2, label="Dataset-bootstrap 95% CI (b)"),
         Line2D([], [], color=BLUE, linewidth=0.8, alpha=0.55, label="Crossed detector × dataset 95% CI (b)"),
         Line2D([], [], color=MUTED, linewidth=0.7, linestyle=(0, (2, 1.5)), label="G1 threshold 3.0 (b)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=6, handletextpad=0.4,
               columnspacing=1.2, borderaxespad=0.0)
    save(fig, "posthoc_forest")

    # ---- selector robustness ----
    pp = pd.read_csv(OUT.format("selectors_pooled"))
    pp = pp[pp.view == "primary"]
    per = pd.read_csv(OUT.format("selectors"))
    per = per[per.complete == True]  # noqa: E712
    pn = pd.read_csv(OUT.format("permnull_pooled")).set_index("pool")
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    xs = np.arange(len(SELECTORS))
    groups = (("frozen12", "UB-GOLD detectors", BLUE, "o", -0.15, DEEP12),
              ("recent", "Recent methods", ORANGE, "s", 0.15, RECENT_IT))
    rng = np.random.default_rng(0)
    for pname, label, col, mkr, dx, ms in groups:
        g = pp[pp.pool == pname].set_index("selector").reindex(SELECTORS)
        d = per[per.method.isin(ms)]
        for i, k in enumerate(SELECTORS):
            v = d[d.selector == k].delta.to_numpy()
            if len(v):
                ax.scatter(i + dx + rng.uniform(-0.045, 0.045, len(v)), v, s=4, color=col, alpha=0.35, linewidth=0,
                           marker=mkr, zorder=2)
        ok = np.isfinite(g.delta.to_numpy(float))
        ax.vlines(xs[ok] + dx, g.ds_ci_lo[ok], g.ds_ci_hi[ok], color=col, linewidth=1.1, zorder=3)
        ax.scatter(xs[ok] + dx, g.delta[ok], marker=mkr, s=20, color=col, edgecolor="white", linewidth=0.5, zorder=4,
                   label=f"{label}: pooled Δ, 95% CI")
    # permutation-null references (detectors with per-graph scores): at S1 (max - LAST) and S6 (max - mean)
    for pname, dx in (("frozen12", -0.40), ("recent", 0.40)):
        if pname in pn.index:
            for xi, col_ in ((0, "null_max_minus_last"), (5, "null_max_minus_mean")):
                ax.scatter([xi + dx], [pn.loc[pname, col_]], marker="D", s=13, facecolor="white", edgecolor=AQUA,
                           linewidth=1.0, zorder=5)
    ax.scatter([], [], marker="D", s=13, facecolor="white", edgecolor=AQUA, linewidth=1.0,
               label="Permutation null, 8 | 8 detectors with scores\n(at S1: max − LAST; at S6: max − mean)")
    ax.scatter([], [], s=6, color=MUTED, alpha=0.5, linewidth=0, label="Single detector")
    ax.axhline(0, color=INK, linewidth=0.6, zorder=1)
    ax.set_xticks(xs)
    nn = pp.pivot_table(index="selector", columns="pool", values="n_methods").reindex(SELECTORS).fillna(0).astype(int)
    ax.set_xticklabels([f"{k}\n{SEL_TICK[k]}\n{nn.loc[k, 'frozen12']} | {nn.loc[k, 'recent']}" for k in SELECTORS],
                       fontsize=5.3, linespacing=1.05)
    ax.text(-0.62, -0.168, "n (UB-GOLD | recent)", transform=ax.get_xaxis_transform(), fontsize=5,
            color=MUTED, ha="right", va="center")
    ax.set_xlim(-0.6, len(SELECTORS) - 0.4)
    ax.set_ylabel("ORACLE − selector (AUROC points)")
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(min(-0.5, float(per.delta.min()) - 0.8), float(max(pp.ds_ci_hi.max(), per.delta.max())) + 1.2)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=5.6, handletextpad=0.3, borderaxespad=0.2,
              ncol=1)
    save(fig, "posthoc_selectors")
    print("wrote", osp.join(FIGDIR, "posthoc_forest.{pdf,png}"), "and posthoc_selectors.{pdf,png}")


# ----------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--budget", type=float, default=80.0)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("compute")
    p.add_argument("--budget", type=float, default=80.0, help="stop starting new runs after this many seconds")
    p.add_argument("--only", default=None, help="comma-separated methods or datasets (testing)")
    p.add_argument("--recheck", action="store_true",
                   help="recompute runs whose part JSON or score npz changed since caching (re-hashes all inputs; "
                        "'aggregate' always verifies the hashes)")
    sub.add_parser("status")
    sub.add_parser("aggregate")
    sub.add_parser("figures")
    args = ap.parse_args(argv)
    {"prep": cmd_prep, "compute": cmd_compute, "status": cmd_status, "aggregate": cmd_aggregate,
     "figures": cmd_figures}[args.cmd](args)


if __name__ == "__main__":
    main()
