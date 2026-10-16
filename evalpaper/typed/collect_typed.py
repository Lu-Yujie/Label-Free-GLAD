"""Collect finished typed runs -> results/eval_typed.csv in the stats/FROZEN.md typed schema (RUN_NOTES §0).

One row per (method, generator, anomaly type, size variant, feature variant, seed, criterion[, orientation]):
  oracle     UB-GOLD's rule replayed on that type's own test-AUROC trajectory (= harness collect.ubgold_oracle): only
             points with epoch % eval_freq == 0; keep a point if its AUROC > running max (start 0); stop after 30 points
             without improvement. Selection key: the main score's AUROC, except CVTGAD (adaptive-normalised score,
             A4) and GraphDE/ubgold (UB-GOLD orientation, A5). AUPRC/FPR95 at the same point.
  last       the final epoch (num_epoch).
  valnormal  the point with the minimum mean held-out-normal score (A3), earliest on ties; one point for all types.
A15 (mechanical): ORACLE never selects a non-finite point; VAL-NORMAL ignores non-finite held-out means; a selected point
whose scores for that type are non-finite -> AUROC 50, AUPRC = anomaly share of that type's test set, FPR95 95,
nonfinite_selected = 1; no finite candidate -> the same chance values, flagged. A21 tie flag (descriptive): distinct
finite scores of that type's test set at the selected point < max(10, 5% of its size).
GraphDE rows carry orientation 'paper' (main) and 'ubgold' (reproduction; glad_stats drops them, A5). Values in percent.

Only complete, untagged runs are collected (smoke/timing runs carry a tag). Prints counts only, never metric values.
Usage:
  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -B collect_typed.py            # grid -> results
  ... collect_typed.py --parts <smoke parts> --include-tagged --out <tmpdir>/eval_typed.csv           # tests
"""
import sys

sys.dont_write_bytecode = True

import argparse
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PRELIM = osp.dirname(osp.dirname(HERE))
RES = osp.join(PRELIM, "results")
PARTS = osp.join(HERE, "parts")
TYPED_ROOT = os.environ.get("TYPED_ROOT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/typed"))
sys.path.insert(0, HERE)
import typed_data as TD  # noqa: E402

PATIENCE = 30
ALL12 = ["GLADC", "CVTGAD", "GOOD-D", "GraphDE", "GraphCL_IF", "GraphCL_OCSVM", "InfoGraph_IF", "InfoGraph_OCSVM",
         "SIGNET", "OCGTL", "OCGIN", "GLocalKD"]
COLS = ["method", "dataset", "anomaly_type", "size_variant", "feature_variant", "seed", "criterion", "auroc", "auprc",
        "fpr95", "n_anomalies", "n_normals", "orientation", "epoch_selected", "ubgold_stop_epoch", "nonfinite_selected",
        "a15_action", "n_distinct", "tie_collapsed", "device", "hp_config", "run_id", "n_eval", "source"]
KEY = ["method", "dataset", "anomaly_type", "size_variant", "feature_variant", "seed", "criterion", "orientation"]


def finite(x):
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def ubgold_oracle(vals, epochs, eval_freq, ok=None):
    """harness/collect.py ubgold_oracle with A15 rule 1 as stats/glad_stats.ubgold_select: a point can improve only if
    its selection AUROC is finite and (ok) the main scores of that type are finite; others count as non-improvement."""
    max_auc, best, counter, stop_epoch = 0.0, None, 0, None
    ok = ok if ok is not None else [True] * len(vals)
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
    return None if np.all(np.isnan(v)) else int(np.nanargmin(v))   # first occurrence = earliest on ties


def device_of(meta):
    d = meta.get("device_all") or meta.get("device", "")
    return "+".join(sorted({("cuda" if "cuda" in x else "cpu") for x in d.split("+") if x}))


def run_rows(J, src):
    meta, traj = J["meta"], J["traj"]
    m, gen, seed, fv = meta["method"], meta["generator"], int(meta["seed"]), meta["feature_variant"]
    h = meta["hparams"]
    ef, ne = int(h["eval_freq"]), int(h["num_epoch"])
    epochs = [int(r["epoch"]) for r in traj]
    expect = sorted(set(range(ef, ne + 1, ef)) | {ne})
    if epochs != expect:
        raise ValueError(f"{meta['run_id']}: incomplete trajectory ({len(epochs)} of {len(expect)} points)")
    i_last = epochs.index(ne)
    base = dict(method=m, seed=seed, device=device_of(meta), hp_config="modal|" + meta.get("hp_line", ""),
                run_id=meta["run_id"], n_eval=len(traj), source=src)
    if m == "GraphDE":
        orients = [("paper", "types", "types", "holdout_score"), ("ubgold", "types_ubgold", "types_ubgold",
                                                                   "holdout_score_ubgold")]
    elif m == "CVTGAD":
        orients = [("native", "types", "types_sel", "holdout_score")]
    else:
        orients = [("native", "types", "types", "holdout_score")]
    rows = []
    for key, cell in meta["groups"].items():
        n_an, n_no = int(cell["n_anomalies"]), int(cell["n_normals"])
        share = n_an / float(n_an + n_no)
        tie_thr = max(10, 0.05 * (n_an + n_no))
        common = dict(base, dataset=cell["dataset"], anomaly_type=cell["anomaly_type"],
                      size_variant=cell["size_variant"], feature_variant=cell["feature_variant"], n_anomalies=n_an,
                      n_normals=n_no)
        for orient, kmet, ksel, khold in orients:
            met = [r[kmet][key] for r in traj]

            def row(crit, i, action="", stop=None):
                r = dict(common, criterion=crit, orientation=orient, ubgold_stop_epoch=stop, a15_action=action)
                if i is None or not finite(met[i]["auroc"]):
                    r.update(auroc=50.0, auprc=100.0 * share, fpr95=95.0, nonfinite_selected=1,
                             epoch_selected=(epochs[i] if i is not None else None),
                             a15_action=action or {"last": "A15.3_last_nonfinite", "oracle": "A15.1_selected_nonfinite",
                                                   "valnormal": "A15.2a_valnormal_nonfinite_test"}[crit],
                             n_distinct=(met[i].get("n_distinct") if i is not None else None))
                else:
                    r.update(auroc=100.0 * met[i]["auroc"], auprc=100.0 * met[i]["auprc"], fpr95=100.0 * met[i]["fpr95"],
                             nonfinite_selected=0, epoch_selected=epochs[i], n_distinct=met[i].get("n_distinct"))
                nd = r["n_distinct"]
                r["tie_collapsed"] = int(nd is not None and nd < tie_thr) if nd is not None else None
                return r

            ok = [finite(x["auroc"]) for x in met]
            if not any(ok):                                  # A15 rule 4: every point non-finite for this type
                for crit in ("oracle", "last", "valnormal"):
                    rows.append(row(crit, None, "A15.4_all_points_nonfinite", ne if crit == "oracle" else None))
                continue
            b, stop = ubgold_oracle([r[ksel][key]["auroc"] for r in traj], epochs, ef, ok)
            rows.append(row("oracle", b, "" if b is not None else "A15.1_no_finite_point",
                            stop if stop is not None else ne))
            rows.append(row("last", i_last))
            v = argmin_finite([r.get(khold, np.nan) for r in traj])
            rows.append(row("valnormal", v, "" if v is not None else "A15.2b_no_finite_holdout"))
    return rows


def load_runs(parts_dir, include_tagged, only_tag=None):
    out, skipped = [], []
    for fn in sorted(glob.glob(osp.join(parts_dir, "*.json"))):
        J = json.load(open(fn))
        meta = J["meta"]
        stem = f"{meta['method']}_{meta['generator']}_{meta['feature_variant']}_s{meta['seed']}"
        tagged = bool(meta.get("tag")) or meta.get("max_epochs_override") is not None or meta["run_id"] != stem
        if (tagged and not include_tagged) or (only_tag is not None and meta.get("tag") != only_tag):
            skipped.append(meta["run_id"])
            continue
        out.append(J)
    return out, skipped


def coverage(runs):
    have = {(J["meta"]["method"], J["meta"]["generator"], int(J["meta"]["seed"]), J["meta"]["feature_variant"])
            for J in runs}
    expected = [(m, g, s, fv) for m in ALL12 for (g, s, fv) in TD.UNITS_A28]   # CHANGE-A28.1: MS-v2 units
    missing = [e for e in expected if e not in have]
    return len(expected), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=PARTS)
    ap.add_argument("--out", default=osp.join(RES, "eval_typed.csv"))
    ap.add_argument("--include-tagged", action="store_true", help="tests only: also collect tagged smoke runs")
    ap.add_argument("--only-tag", default=None, help="tests only: collect only runs with this tag (implies --include-tagged)")
    a = ap.parse_args()
    a.include_tagged = a.include_tagged or a.only_tag is not None
    if a.include_tagged and osp.realpath(osp.dirname(a.out)) == osp.realpath(RES):
        sys.exit("tagged (smoke) runs are never written into prelim/results")
    runs, skipped = load_runs(a.parts, a.include_tagged, a.only_tag)
    rows, bad = [], []
    for J in runs:
        try:
            rows += run_rows(J, "typed/run_typed.py")
        except ValueError as e:
            bad.append(str(e))
    df = pd.DataFrame(rows, columns=COLS)
    if len(df):
        dup = df.duplicated(KEY, keep=False)
        assert not dup.any(), df[dup][KEY].head().values.tolist()
        ok = np.isfinite(df.auroc) & df.auroc.between(0, 100) & np.isfinite(df.auprc) & df.auprc.between(0, 100) \
            & np.isfinite(df.fpr95) & df.fpr95.between(0, 100)
        assert ok.all(), f"{int((~ok).sum())} rows fail the finite/range check"
        df = df.sort_values(KEY).reset_index(drop=True)
    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, a.out)
    n_exp, missing = coverage(runs)
    crash = sorted(glob.glob(osp.join(TYPED_ROOT, "crash", "*.attempt*.json")))
    print(f"runs collected {len(runs)} (tagged skipped {len(skipped)}, incomplete {len(bad)}); rows {len(df)} "
          f"(A15-flagged {int(df.nonfinite_selected.sum()) if len(df) else 0}, tie-flagged "
          f"{int((df.tie_collapsed == 1).sum()) if len(df) else 0}); grid coverage {n_exp - len(missing)}/{n_exp}; "
          f"crash records {len(crash)} -> {a.out}")
    for b in bad:
        print("  incomplete:", b)


if __name__ == "__main__":
    main()
