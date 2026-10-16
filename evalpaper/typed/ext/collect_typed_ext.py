"""Collect finished Phase B runs (the 9 E5/E4 methods on GX / MS-v2) -> results/eval_typed_ext.csv in the
stats/FROZEN.md typed schema (RUN_NOTES_ext.md §0.1 items 7 and 11).

One row per (method, generator, anomaly type, size variant, feature variant, seed, criterion):
  oracle       UB-GOLD's rule (A26 §1: uniform rule) replayed on that group's own test-AUROC trajectory: candidates are
               points with epoch % eval_freq == 0 for the methods glad_stats lists in EF_FILTER (MUSE, SEGO, GLADMamba,
               HC-GLAD, LGKDE) and every point otherwise (GLADPro, SUL-GUL, GLASS incl. epoch 0, ProtoGLAD); keep a point
               if its AUROC is strictly above the running max (start 0); stop after 30 candidates without improvement.
  oracle_own   the method's own test-label rule, replayed per group (oracle_rule column names it):
                 LGKDE      EarlyStopping (its own class, read from the pinned source) on (sklearn AUROC + AP)/2, first max
                            among epochs 1..stop (per run; = the port's derive.py);
                 GLADPro, SUL-GUL   common epoch with the max 5-seed mean test AUROC (first max);
                 GLADMamba, HC-GLAD common epoch (epoch % eval_freq == 0) with the max 5-seed mean, ties lower std
                            (ddof 0), then earlier;
                 GLASS      per seed GLASS's test-selected checkpoint (in-loop ensemble AUROC, strict improvement from the
                            epoch-0 baseline, +5 per non-improving evaluation, stop at >= patience), then the rule (of 24)
                            with the max 5-seed mean test AUROC at those checkpoints (first max, evaluate_reference.py
                            l.383); every seed reports that rule;
                 ProtoGLAD  first cell (product order) whose 5-seed mean test AUROC is strictly larger than the running
                            best from 0.0 (run_detection.py l.184/259); every seed reports that cell.
               Cross-seed rules need all 5 seeds of (method, generator, feature variant); A15.1: only points finite in all
               5 seeds can be chosen (glad_stats common_epoch_reselect); MUSE and SEGO have no own rule (no row).
  last         final epoch;  valnormal  minimum finite held-out-normal mean, earliest on ties (A3, A15.2).
  last_r0, valnormal_r0   GLASS only: the fallback rule fixed_test_raw (A23 §1), held-out mean of that rule.
A15 mechanics and the A21 tie flag exactly as typed/collect_typed.py. ProtoGLAD rows carry checkpoint_free = 1 (A20).
Values in percent. Only complete, untagged runs are collected unless --include-tagged (tests; never into results).
Prints counts only, never a metric value (blinding).

  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -B collect_typed_ext.py          # -> results
  ... collect_typed_ext.py --parts <smoke parts> --include-tagged --out <tmpdir>/eval_typed_ext.csv          # tests
"""
import sys

sys.dont_write_bytecode = True

import argparse
import ast
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
TYPED = osp.dirname(HERE)
EVALPAPER = osp.dirname(TYPED)
PRELIM = osp.dirname(EVALPAPER)
RES = osp.join(PRELIM, "results")
PARTS = osp.join(HERE, "parts")
TYPED_ROOT = os.environ.get("TYPED_ROOT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/typed"))
sys.path.insert(0, TYPED)
import typed_data as TD  # noqa: E402

PATIENCE = 30
METHODS = ["MUSE", "LGKDE", "GLADPro", "GLADMamba", "SEGO", "GLASS", "ProtoGLAD", "HC-GLAD", "SUL-GUL"]
EF_FILTER = {"MUSE", "SEGO", "GLADMamba", "HC-GLAD", "LGKDE"}          # = glad_stats.EF_FILTER ∩ the 9
COMMON_FIRST_MAX = {"GLADPro", "SUL-GUL"}
COMMON_TIE_STD = {"GLADMamba", "HC-GLAD"}
CHECKPOINT_FREE = {"ProtoGLAD"}
GLASS_R0 = "fixed_test_raw"
OWN_RULE = {
    "LGKDE": "own: LGKDE EarlyStopping on (AUROC+AP)/2, first max among epochs 1..stop (A10 §2, A18)",
    "GLADPro": "own: common epoch with max 5-seed mean test AUROC, first max; finite in all seeds (A15)",
    "SUL-GUL": "own: GUL common epoch with max 5-seed mean test AUROC, first max; finite in all seeds (A11 §2, A21)",
    "GLADMamba": "own: common epoch (epoch % eval_freq == 0) with max 5-seed mean test AUROC, ties lower std then "
                 "earlier; finite in all seeds (A14)",
    "HC-GLAD": "own: common epoch (epoch % eval_freq == 0) with max 5-seed mean test AUROC, ties lower std then "
               "earlier; finite in all seeds (A11 §2, A27 §1)",
    "GLASS": "own: GLASS test-selected checkpoint (in-loop ensemble AUROC, patience) + best of 24 reference rules by "
             "5-seed mean (A23 §2)",
    "ProtoGLAD": "own: test-grid-best of 144 cells by 5-seed mean test AUROC, first strictly larger (A11 §2, A20 §1)",
}
COLS = ["method", "dataset", "anomaly_type", "size_variant", "feature_variant", "seed", "criterion", "auroc", "auprc",
        "fpr95", "n_anomalies", "n_normals", "orientation", "oracle_rule", "epoch_selected", "ubgold_stop_epoch",
        "own_detail", "nonfinite_selected", "a15_action", "n_distinct", "tie_collapsed", "checkpoint_free", "device",
        "hp_config", "run_id", "n_eval", "generator", "source"]
KEY = ["method", "dataset", "anomaly_type", "size_variant", "feature_variant", "seed", "criterion"]


def finite(x):
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def ubgold_oracle(vals, epochs, eval_freq, ok):
    """= collect_typed.ubgold_oracle (harness collect.py rule with A15.1); eval_freq 1 = every point a candidate."""
    max_auc, best, counter, stop_epoch = 0.0, None, 0, None
    for i, (a, ep) in enumerate(zip(vals, epochs)):
        if ep % eval_freq != 0:
            continue
        if ok[i] and finite(a) and a > max_auc:
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE:
            stop_epoch = ep
            break
    return best, stop_epoch


def argmin_finite(vals):
    v = np.array([x if finite(x) else np.nan for x in vals], dtype=float)
    return None if np.all(np.isnan(v)) else int(np.nanargmin(v))


def device_of(meta):
    d = meta.get("device_all") or meta.get("device", "")
    return "+".join(sorted({("cuda" if "cuda" in x else "cpu") for x in d.split("+") if x}))


def lgkde_early_stopping():
    """LGKDE's own EarlyStopping class, extracted verbatim from the pinned source (as the port's derive.py)."""
    path = osp.join(EVALPAPER, "ext", "LGKDE", "src", "experiment.py")
    tree = ast.parse(open(path).read())
    node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EarlyStopping"][0]
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns["EarlyStopping"]


# ----------------------------------------------------------------------------------------------------------------
# rows
# ----------------------------------------------------------------------------------------------------------------
class Run:
    def __init__(self, J, src):
        self.J, self.src = J, src
        self.meta, self.traj = J["meta"], J["traj"]
        m = self.meta
        self.method, self.generator, self.seed, self.fv = m["method"], m["generator"], int(m["seed"]), m["feature_variant"]
        h = m["hparams"]
        self.ef, self.ne = int(h["eval_freq"]), int(h["num_epoch"])
        self.epochs = [int(r["epoch"]) for r in self.traj]
        self.groups = m["groups"]
        self.cand_ef = self.ef if self.method in EF_FILTER else 1

    def check_complete(self):
        if self.method == "GLASS":
            expect = [0, 1] + list(range(5, self.ne + 1, 5))
        elif self.method in ("GLADPro", "SUL-GUL", "LGKDE", "ProtoGLAD"):
            expect = list(range(1, self.ne + 1))
        else:
            expect = sorted(set(range(self.ef, self.ne + 1, self.ef)) | {self.ne})
        if self.epochs != expect:
            raise ValueError(f"{self.meta['run_id']}: incomplete trajectory ({len(self.epochs)} of {len(expect)} points)")
        return self

    def base(self, key):
        cell = self.groups[key]
        n_an, n_no = int(cell["n_anomalies"]), int(cell["n_normals"])
        return dict(method=self.method, dataset=cell["dataset"], anomaly_type=cell["anomaly_type"],
                    size_variant=cell["size_variant"], feature_variant=cell["feature_variant"], seed=self.seed,
                    n_anomalies=n_an, n_normals=n_no, orientation="native", device=device_of(self.meta),
                    hp_config="ubgold|" + self.meta.get("hp_line", ""), run_id=self.meta["run_id"], n_eval=len(self.traj),
                    generator=self.generator, checkpoint_free=int(self.method in CHECKPOINT_FREE), source=self.src)

    def mets(self, key, i, which="main", rule=None):
        """(auroc, auprc, fpr95, n_distinct) fractions at point i; which = main | rule."""
        t = self.traj[i]["types"][key]
        if which == "rule":
            a, p, f, nf, nd = t["rules"][rule]
            return (a if nf == 0 else float("nan")), p, f, nd
        if which == "cell":
            a, p, f, nf, nd = self.traj[i]["grid"][rule][key]
            return (a if nf == 0 else float("nan")), p, f, nd
        return t["auroc"], t["auprc"], t["fpr95"], t.get("n_distinct")


def make_row(run, key, crit, i, which="main", rule=None, action="", stop=None, oracle_rule="", own_detail=""):
    r = dict(run.base(key), criterion=crit, ubgold_stop_epoch=stop, a15_action=action, oracle_rule=oracle_rule,
             own_detail=own_detail)
    n_an, n_no = r["n_anomalies"], r["n_normals"]
    share = n_an / float(n_an + n_no)
    tie_thr = max(10, 0.05 * (n_an + n_no))
    if i is not None:
        a, p, f, nd = run.mets(key, i, which, rule)
    if i is None or not finite(a):
        default = {"last": "A15.3_last_nonfinite", "oracle": "A15.1_selected_nonfinite",
                   "valnormal": "A15.2a_valnormal_nonfinite_test", "oracle_own": "A15.1_unresolved_own_rule",
                   "last_r0": "A15.3_last_nonfinite", "valnormal_r0": "A15.2a_valnormal_nonfinite_test"}[crit]
        r.update(auroc=50.0, auprc=100.0 * share, fpr95=95.0, nonfinite_selected=1,
                 epoch_selected=(run.epochs[i] if i is not None else None), a15_action=action or default,
                 n_distinct=(nd if i is not None else None))
    else:
        r.update(auroc=100.0 * a, auprc=100.0 * p, fpr95=100.0 * f, nonfinite_selected=0, epoch_selected=run.epochs[i],
                 n_distinct=nd)
    nd = r["n_distinct"]
    r["tie_collapsed"] = int(nd < tie_thr) if nd is not None else None
    return r


def run_rows(run):
    """Per-run rows: oracle (UB-GOLD rule), last, valnormal (+ GLASS r0 rows, + LGKDE's per-run own rule)."""
    rows = []
    ep, ne = run.epochs, run.ne
    i_last = ep.index(ne)
    holds = [r.get("holdout_score", np.nan) for r in run.traj]
    for key in run.groups:
        met = [r["types"][key] for r in run.traj]
        ok = [finite(m["auroc"]) for m in met]
        if not any(ok):                                     # A15 rule 4
            for crit in ("oracle", "last", "valnormal"):
                rows.append(make_row(run, key, crit, None, action="A15.4_all_points_nonfinite",
                                     stop=ne if crit == "oracle" else None,
                                     oracle_rule="ubgold_patience30" if crit == "oracle" else ""))
        else:
            b, stop = ubgold_oracle([m["auroc"] for m in met], ep, run.cand_ef, ok)
            rows.append(make_row(run, key, "oracle", b, action="" if b is not None else "A15.1_no_finite_point",
                                 stop=stop if stop is not None else ne, oracle_rule="ubgold_patience30"))
            rows.append(make_row(run, key, "last", i_last))
            v = argmin_finite(holds)
            rows.append(make_row(run, key, "valnormal", v, action="" if v is not None else "A15.2b_no_finite_holdout"))
        if run.method == "GLASS":
            rows.append(make_row(run, key, "last_r0", i_last, "rule", GLASS_R0, oracle_rule="",
                                 own_detail="fallback rule fixed_test_raw (A23 §1)"))
            h0 = [r["rule_holdout"][GLASS_R0] for r in run.traj]
            v0 = argmin_finite(h0)
            rows.append(make_row(run, key, "valnormal_r0", v0, "rule", GLASS_R0,
                                 action="" if v0 is not None else "A15.2b_no_finite_holdout",
                                 own_detail="fallback rule fixed_test_raw (A23 §1)"))
        if run.method == "LGKDE":
            ES = lgkde_early_stopping()
            rule = run.meta["flags"]["own_rule"]
            es = ES(patience=rule["patience"], min_delta=rule["min_delta"], min_epochs=rule["min_epochs"])
            best_perf, best_i, stop_e = 0, None, None
            for i, r in enumerate(run.traj):
                perf = r["types"][key]["own_sel"]
                if perf > best_perf:                        # derive.py / save_results (NaN never wins)
                    best_perf, best_i = perf, i
                if es(r["epoch"], perf):
                    stop_e = r["epoch"]
                    break
            rows.append(make_row(run, key, "oracle_own", best_i, oracle_rule=OWN_RULE["LGKDE"],
                                 stop=stop_e if stop_e is not None else ne,
                                 own_detail=json.dumps(dict(stop_epoch=stop_e if stop_e is not None else ne))))
    return rows


def cross_seed_rows(method, runs):
    """Own rules that choose one point (or rule / cell) for all 5 seeds of (method, generator, feature variant)."""
    rows, unresolved = [], []
    runs = sorted(runs, key=lambda r: r.seed)
    keys = list(runs[0].groups)
    rule = OWN_RULE[method]
    for key in keys:
        if method in COMMON_FIRST_MAX | COMMON_TIE_STD:
            common = set(runs[0].epochs)
            for r in runs[1:]:
                common &= set(r.epochs)
            cands = []
            for e in sorted(common):
                if method in COMMON_TIE_STD and e % runs[0].ef != 0:
                    continue
                idx = [r.epochs.index(e) for r in runs]
                v = [r.traj[i]["types"][key]["auroc"] for r, i in zip(runs, idx)]
                okf = [r.traj[i]["types"][key].get("nonfinite", 0) == 0 for r, i in zip(runs, idx)]
                if all(okf) and all(finite(x) for x in v):
                    cands.append((float(np.mean(v)), float(np.std(v)), e))
            if not cands:
                for r in runs:
                    rows.append(make_row(r, key, "oracle_own", None, action="A15.1_no_epoch_finite_in_all_seeds",
                                         oracle_rule=rule))
                continue
            if method in COMMON_TIE_STD:
                best = sorted(cands, key=lambda c: (-c[0], c[1], c[2]))[0]
            else:
                best = cands[0]
                for c in cands[1:]:
                    if c[0] > best[0]:
                        best = c
            e = best[2]
            for r in runs:
                rows.append(make_row(r, key, "oracle_own", r.epochs.index(e), oracle_rule=rule,
                                     own_detail=json.dumps(dict(common_epoch=e))))
        elif method == "GLASS":
            sel = []
            for r in runs:
                pat = int(r.meta["flags"]["patience"])
                ens = [t["types"][key]["glass_ens_auroc"] for t in r.traj]
                best, best_i, counter, stopped = ens[0], 0, 0, False
                for i in range(1, len(ens)):
                    if stopped:
                        break
                    if ens[i] > best:
                        best, best_i, counter = ens[i], i, 0
                    else:
                        counter += 5
                    if counter >= pat:
                        stopped = True
                sel.append(best_i)
            means = {}
            for name in runs[0].meta["flags"]["rules"]:
                v = [r.mets(key, i, "rule", name)[0] for r, i in zip(runs, sel)]
                if all(finite(x) for x in v):
                    means[name] = float(np.mean(v))
            if not means:
                for r, i in zip(runs, sel):
                    rows.append(make_row(r, key, "oracle_own", None, action="A15.1_unresolved_own_rule", oracle_rule=rule))
                continue
            best_rule = max(means, key=lambda n: means[n])      # first max in enumeration order (dict order)
            for r, i in zip(runs, sel):
                rows.append(make_row(r, key, "oracle_own", i, "rule", best_rule, oracle_rule=rule,
                                     own_detail=json.dumps(dict(checkpoint_epoch=r.epochs[i], rule=best_rule))))
        elif method == "ProtoGLAD":
            cells = runs[0].meta["flags"]["grid_cells"]
            best_avg, best_cell = 0.0, None
            for c in cells:
                v = [r.mets(key, 0, "cell", c)[0] for r in runs]
                mv = float(np.mean(v)) if all(finite(x) for x in v) else float("nan")
                if finite(mv) and mv > best_avg:
                    best_avg, best_cell = mv, c
            for r in runs:
                if best_cell is None:
                    rows.append(make_row(r, key, "oracle_own", None, action="A15.1_unresolved_own_rule", oracle_rule=rule))
                else:
                    rows.append(make_row(r, key, "oracle_own", 0, "cell", best_cell, oracle_rule=rule,
                                         own_detail=json.dumps(dict(cell=best_cell))))
    return rows, unresolved


# ----------------------------------------------------------------------------------------------------------------
def load_runs(parts_dir, include_tagged, only_tag=None):
    out, skipped, bad = [], [], []
    for fn in sorted(glob.glob(osp.join(parts_dir, "*.json"))):
        J = json.load(open(fn))
        meta = J["meta"]
        if meta.get("method") not in METHODS:
            skipped.append(osp.basename(fn))
            continue
        stem = f"{meta['method']}_{meta['generator']}_{meta['feature_variant']}_s{meta['seed']}"
        tagged = bool(meta.get("tag")) or meta.get("max_epochs_override") is not None or meta["run_id"] != stem \
            or bool(meta.get("smoke")) or bool(meta.get("flags", {}).get("dry_n"))
        if (tagged and not include_tagged) or (only_tag is not None and meta.get("tag") != only_tag):
            skipped.append(meta["run_id"])
            continue
        try:
            out.append(Run(J, "typed/ext/" + meta["method"] + "_typed.py").check_complete())
        except ValueError as e:
            bad.append(str(e))
    return out, skipped, bad


def coverage(runs):
    have = {(r.method, r.generator, r.seed, r.fv) for r in runs}
    expected = [(m, g, s, fv) for m in METHODS for (g, s, fv) in TD.UNITS_A28]
    return len(expected), [e for e in expected if e not in have]


def collect(runs):
    rows = []
    for r in runs:
        rows += run_rows(r)
    groups = {}
    for r in runs:
        if r.method in COMMON_FIRST_MAX | COMMON_TIE_STD | {"GLASS", "ProtoGLAD"}:
            groups.setdefault((r.method, r.generator, r.fv), []).append(r)
    incomplete_own = []
    for (m, g, fv), rs in sorted(groups.items()):
        if sorted(r.seed for r in rs) != list(TD.SEEDS):
            incomplete_own.append(f"{m}_{g}_{fv} ({len(rs)}/5 seeds)")
            continue
        rr, _ = cross_seed_rows(m, rs)
        rows += rr
    return rows, incomplete_own


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=PARTS)
    ap.add_argument("--out", default=osp.join(RES, "eval_typed_ext.csv"))
    ap.add_argument("--include-tagged", action="store_true", help="tests only: also collect tagged smoke runs")
    ap.add_argument("--only-tag", default=None, help="tests only: collect only runs with this tag")
    a = ap.parse_args()
    a.include_tagged = a.include_tagged or a.only_tag is not None
    if a.include_tagged and osp.realpath(osp.dirname(osp.abspath(a.out))) == osp.realpath(RES):
        sys.exit("tagged (smoke) runs are never written into prelim/results")
    runs, skipped, bad = load_runs(a.parts, a.include_tagged, a.only_tag)
    rows, incomplete_own = collect(runs)
    df = pd.DataFrame(rows, columns=COLS)
    if len(df):
        dup = df.duplicated(KEY, keep=False)
        assert not dup.any(), df[dup][KEY].head().values.tolist()
        ok = np.isfinite(df.auroc) & df.auroc.between(0, 100) & np.isfinite(df.auprc) & df.auprc.between(0, 100) \
            & np.isfinite(df.fpr95) & df.fpr95.between(0, 100)
        assert ok.all(), f"{int((~ok).sum())} rows fail the finite/range check"
        df = df.sort_values(KEY).reset_index(drop=True)
    # the duplicate-key check of glad_stats spans every eval_typed*.csv: our method names must not occur elsewhere
    out_dir = osp.dirname(osp.abspath(a.out))
    for other in sorted(glob.glob(osp.join(out_dir, "eval_typed*.csv"))):
        if osp.realpath(other) == osp.realpath(a.out):
            continue
        meths = set(pd.read_csv(other, usecols=["method"])["method"].astype(str))
        clash = meths & set(METHODS)
        assert not clash, f"{osp.basename(other)} already holds rows of {sorted(clash)}"
    os.makedirs(out_dir, exist_ok=True)
    tmp = a.out + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, a.out)
    n_exp, missing = coverage(runs)
    crash = sorted(glob.glob(osp.join(TYPED_ROOT, "ext_ckpt", "crash", "*.attempt*.json")))
    print(f"runs collected {len(runs)} (skipped {len(skipped)}, incomplete {len(bad)}); rows {len(df)} "
          f"(A15-flagged {int(df.nonfinite_selected.sum()) if len(df) else 0}, tie-flagged "
          f"{int((df.tie_collapsed == 1).sum()) if len(df) else 0}); criteria "
          f"{dict(df.criterion.value_counts()) if len(df) else {}}; grid coverage {n_exp - len(missing)}/{n_exp}; "
          f"own-rule cells without 5 seeds {len(incomplete_own)}; crash records {len(crash)} -> {a.out}")
    for b in bad:
        print("  incomplete:", b)
    if incomplete_own and len(incomplete_own) <= 20:
        print("  oracle_own pending (needs 5 seeds):", ", ".join(incomplete_own))


if __name__ == "__main__":
    main()
