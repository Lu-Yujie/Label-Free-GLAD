"""Aggregate every finished run into prelim/results/eval_folds.csv (schema read by the statistics code).

Run on the login node at any time (reads only finished runs; safe while jobs are running):
    PYTHONDONTWRITEBYTECODE=1 python3 -B prelim/evalpaper/harness/collect.py

eval_folds.csv: one row per (method, dataset, fold, protocol, orientation)
  method        UB-GOLD name: GLADC, CVTGAD, GOOD-D, GraphDE, GraphCL_IF, GraphCL_OCSVM, InfoGraph_IF,
                InfoGraph_OCSVM, SIGNET, OCGTL, OCGIN, GLocalKD
  dataset       AIDS BZR COX2 DHFR ENZYMES NCI1 PROTEINS_full IMDB-BINARY REDDIT-BINARY (UB9);
                DD COLLAB Tox21_p53 Tox21_MMP (secondary)
  fold          0-4 (UB-GOLD trial i = fold i, seed 3407+i; Tox21: fixed split, trial i)
  protocol      oracle | last | valnormal            (UB-GOLD per-dataset hyperparameters)
                hpfree_oracle | hpfree_last | hpfree_valnormal   (label-free hyperparameters, A1)
  auroc auprc fpr95   in PERCENT (0-100); higher-is-better for auroc/auprc, lower for fpr95
  epoch_selected      training epoch of the selected checkpoint
  device        cuda | cpu | cpu+cuda (a resumed run that used both)
  hp_config     'ubgold|<UB-GOLD .sh options>' or 'hpfree|<resolved options>'
  orientation   native (all methods) | paper / ubgold (GraphDE only: paper = -infer_e_gx is the MAIN
                column fixed by PREREG; ubgold = UB-GOLD's unnegated score, reproduction only)
  source        evalpaper_harness | deep_rerun (SIGNET/OCGTL/OCGIN/GLocalKD with UB-GOLD hp, GPU runs of
                prelim/deep_rerun, same folds/seeds/held-out split; recomputed here from their JSONs)
  run_id, n_eval, ubgold_stop_epoch (oracle rows: epoch at which UB-GOLD's patience-30 stop fires)

Protocol definitions (PREREG + PREREG_AMENDMENTS):
  oracle     replay of UB-GOLD's fit(): max_AUC=0, keep checkpoint if test AUROC > max_AUC, stop after
             30 evaluations without improvement; only evaluation points with epoch % eval_freq == 0.
             Selection key: test AUROC of the reported score, except CVTGAD (normalised-score AUROC,
             A4) and GraphDE/ubgold (UB-GOLD orientation). AUPRC/FPR95 at the same checkpoint.
  last       final epoch (num_epoch) of the fixed budget.
  valnormal  evaluation point with the minimum mean held-out-normal anomaly score (A3); ties -> earliest.

Also written: eval_folds_extra.csv (oracle_nostop, valnormal_obj, two-stage misaligned-label replica,
CVTGAD normalised-score variants), eval_coverage.csv, eval_repro.csv (ORACLE vs UB-GOLD Table 8).
"""
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PRELIM = osp.dirname(osp.dirname(HERE))
RES = os.environ.get("EP_RES", osp.join(PRELIM, "results"))
PARTS = os.environ.get("EP_PARTS", osp.join(HERE, "parts"))
DR_PARTS = osp.join(PRELIM, "results", "deep_rerun_parts")
PATIENCE = 30
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
SEC = ["DD", "COLLAB", "Tox21_p53", "Tox21_MMP"]
NEW8 = ["GLADC", "CVTGAD", "GOOD-D", "GraphDE", "GraphCL_IF", "GraphCL_OCSVM", "InfoGraph_IF", "InfoGraph_OCSVM"]
OLD4 = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD"]

# UB-GOLD Table 8 (arXiv 2406.15523v2), AUROC %, read from the paper text
# (scratchpad verify_sota/2406.15523.txt; first block of Table 8; columns IG-SVM, IG-iF, GCL-SVM, GCL-iF,
# OCGIN, GLocalKD, OCGTL, SIGNET, GLADC, CVTGAD, GOOD-D, GraphDE). Old-4 values match deep_rerun/aggregate.py.
_T8 = {  # dataset: (IG-SVM, IG-iF, GCL-SVM, GCL-iF, OCGIN, GLocalKD, OCGTL, SIGNET, GLADC, CVTGAD, GOOD-D, GraphDE)
    "Tox21_p53": (68.11, 60.85, 68.61, 64.60, 68.35, 65.43, 67.58, 68.10, 65.82, 69.40, 67.82, 62.59),
    "Tox21_MMP": (57.72, 52.58, 69.91, 71.31, 69.37, 68.12, 67.51, 71.23, 70.03, 70.58, 71.41, 60.12),
    "COLLAB": (36.47, 38.18, 44.91, 45.44, 60.58, 51.85, 48.13, 72.45, 54.32, 71.01, 69.34, 46.77),
    "IMDB-BINARY": (40.89, 45.64, 68.00, 63.88, 61.47, 53.31, 65.27, 70.12, 65.94, 69.82, 66.68, 59.25),
    "REDDIT-BINARY": (60.32, 52.51, 84.49, 82.64, 82.10, 80.32, 89.92, 85.24, 78.87, 87.43, 89.43, 63.42),
    "ENZYMES": (60.97, 53.94, 62.73, 63.09, 62.44, 61.75, 63.59, 63.12, 63.44, 68.56, 64.58, 52.10),
    "PROTEINS_full": (61.15, 52.78, 72.61, 72.60, 76.46, 77.29, 72.89, 75.86, 77.43, 76.49, 76.02, 68.81),
    "DD": (70.33, 42.67, 76.43, 65.41, 79.08, 80.76, 77.76, 74.53, 76.54, 78.84, 79.91, 60.49),
    "BZR": (41.50, 45.42, 68.93, 67.81, 69.13, 68.55, 51.89, 80.79, 68.23, 77.69, 73.28, 65.94),
    "AIDS": (87.20, 97.96, 95.44, 98.80, 96.89, 96.93, 99.36, 97.60, 98.02, 99.21, 97.10, 70.82),
    "COX2": (49.11, 48.61, 59.68, 59.38, 57.81, 58.93, 59.81, 72.35, 64.13, 64.36, 63.19, 54.73),
    "NCI1": (45.11, 61.88, 43.33, 46.44, 69.46, 65.29, 75.75, 74.32, 68.32, 69.13, 61.58, 58.74),
    "DHFR": (45.58, 63.15, 58.21, 57.01, 61.09, 61.79, 59.82, 72.87, 61.25, 63.23, 64.48, 53.23),
}
_COLS = ["InfoGraph_OCSVM", "InfoGraph_IF", "GraphCL_OCSVM", "GraphCL_IF", "OCGIN", "GLocalKD", "OCGTL", "SIGNET",
         "GLADC", "CVTGAD", "GOOD-D", "GraphDE"]
PUBLISHED = {(m, ds): v[i] for ds, v in _T8.items() for i, m in enumerate(_COLS)}


def ubgold_oracle(traj, eval_freq, key):
    max_auc, best, counter, stop_epoch = 0.0, None, 0, None
    for i, r in enumerate(traj):
        if r["epoch"] % eval_freq != 0:
            continue
        a = r.get(key, float("nan"))
        if a == a and a > max_auc:
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE:
            stop_epoch = r["epoch"]
            break
    return best, stop_epoch


def pick(traj, key, mode):
    vals = np.array([r.get(key, np.nan) for r in traj], dtype=float)
    if np.all(np.isnan(vals)):
        return None
    return int(np.nanargmin(vals) if mode == "min" else np.nanargmax(vals))


def device_of(meta):
    d = meta.get("device_all") or meta.get("device", "")
    parts = sorted({("cuda" if "cuda" in x else "cpu") for x in d.split("+") if x})
    return "+".join(parts)


def hp_config(meta, hp):
    if hp == "ubgold":
        line = meta.get("ubgold_sh_line", "")
        return "ubgold|" + " ".join(line.split()[2:]).split("#")[0].strip()
    h = meta["hparams"]
    keys = ["num_epoch", "eval_freq", "batch_size", "batch_size_test", "lr", "hidden_dim", "num_layer", "dropout",
            "alpha", "num_cluster", "rw_dim", "dg_dim", "output_dim", "encoder_layers", "GNN_Encoder", "detector"]
    return "hpfree|" + " ".join(f"{k}={h[k]}" for k in keys if k in h)


def run_rows(J, source):
    meta, traj = J["meta"], J["traj"]
    m, ds, fo = meta["method"], meta["dataset"], int(meta["fold"])
    hp = meta.get("hp", "ubgold")
    pre = "" if hp == "ubgold" else "hpfree_"
    h = meta["hparams"]
    ef, ne = int(h["eval_freq"]), int(h["num_epoch"])
    base = dict(method=m, dataset=ds, fold=fo, device=device_of(meta), hp_config=hp_config(meta, hp), source=source,
                run_id=meta["run_id"], n_eval=len(traj))
    if m == "GraphDE":
        orients = [("paper", ("auroc", "auprc", "fpr95"), "auroc", "holdout_score"),
                   ("ubgold", ("auroc_ubgold", "auprc_ubgold", "fpr95_ubgold"), "auroc_ubgold", "holdout_score_ubgold")]
    elif m == "CVTGAD":
        orients = [("native", ("auroc", "auprc", "fpr95"), "auroc_sel", "holdout_score")]
    else:
        orients = [("native", ("auroc", "auprc", "fpr95"), "auroc", "holdout_score" if "holdout_score" in traj[0] else "holdout_loss")]
    rows, extra = [], []
    last_i = [i for i, r in enumerate(traj) if r["epoch"] == ne]
    assert last_i, (meta["run_id"], ne)
    for orient, (ka, kp, kf), ksel, khold in orients:
        def mk(proto, i, more=None, tgt=rows):
            r = traj[i]
            d = dict(base, protocol=proto, auroc=100 * r[ka], auprc=100 * r[kp], fpr95=100 * r[kf],
                     epoch_selected=int(r["epoch"]), orientation=orient)
            d.update(more or {})
            tgt.append(d)
        b, stop = ubgold_oracle(traj, ef, ksel)
        mk(pre + "oracle", b, dict(ubgold_stop_epoch=stop if stop is not None else ne))
        mk(pre + "last", last_i[0])
        mk(pre + "valnormal", pick(traj, khold, "min"))
        o2 = pick(traj, ka, "max")
        mk(pre + "oracle_nostop", o2, tgt=extra)
        if "holdout_obj" in traj[0] and orient != "ubgold":
            v2 = pick(traj, "holdout_obj", "min")
            if v2 is not None:
                mk(pre + "valnormal_obj", v2, tgt=extra)
        if m == "CVTGAD":
            mk(pre + "valnormal_norm", pick(traj, "holdout_score_norm", "min"), tgt=extra)
            b2, _ = ubgold_oracle(traj, ef, "auroc")
            mk(pre + "oracle_rawsel", b2, tgt=extra)
        if "auroc_ubgold_misaligned" in traj[0]:
            b3, _ = ubgold_oracle(traj, ef, "auroc_ubgold_misaligned")
            mk(pre + "oracle_misaligned_sel", b3, dict(auroc_misaligned=100 * traj[b3]["auroc_ubgold_misaligned"]), tgt=extra)
    return rows, extra


def load_runs():
    out = []
    for fn in sorted(glob.glob(osp.join(PARTS, "*.json"))):
        J = json.load(open(fn))
        meta = J["meta"]
        stem = f"{meta['method']}_{meta['dataset']}_fold{meta['fold']}"
        if meta["run_id"] not in (stem, stem + "_hpfree") or meta.get("max_epochs_override") is not None:
            continue   # tagged smoke / timing / test runs
        out.append((J, "evalpaper_harness"))
    have = {(J["meta"]["method"], J["meta"]["dataset"], J["meta"]["fold"], J["meta"].get("hp", "ubgold")) for J, _ in out}
    for fn in sorted(glob.glob(osp.join(DR_PARTS, "*.json"))):
        J = json.load(open(fn))
        meta = J["meta"]
        if meta["run_id"] != f"{meta['method']}_{meta['dataset']}_fold{meta['fold']}" or meta["method"] not in OLD4:
            continue   # '_full' (no held-out carve-out) and other variants are not part of this table
        if meta["dataset"] not in UB9 + SEC:
            continue   # MUTAG (not a UB-GOLD dataset)
        if (meta["method"], meta["dataset"], meta["fold"], "ubgold") in have:
            continue
        meta.setdefault("hp", "ubgold")
        out.append((J, "deep_rerun"))
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    rows, extra = [], []
    for J, src in load_runs():
        r, e = run_rows(J, src)
        rows += r
        extra += e
    cols = ["method", "dataset", "fold", "protocol", "auroc", "auprc", "fpr95", "epoch_selected", "device",
            "hp_config", "orientation", "source", "run_id", "n_eval", "ubgold_stop_epoch"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values(["method", "dataset", "protocol", "orientation", "fold"]).reset_index(drop=True)
    df.to_csv(osp.join(RES, "eval_folds.csv"), index=False)
    dx = pd.DataFrame(extra)
    if len(dx):
        dx = dx.sort_values(["method", "dataset", "protocol", "orientation", "fold"]).reset_index(drop=True)
    dx.to_csv(osp.join(RES, "eval_folds_extra.csv"), index=False)
    # coverage: folds done per (method, dataset, hp)
    cov = []
    for m in NEW8 + OLD4:
        for ds in UB9 + SEC:
            for hp, proto in (("ubgold", "oracle"), ("hpfree", "hpfree_oracle")):
                sub = df[(df.method == m) & (df.dataset == ds) & (df.protocol == proto) &
                         (df.orientation != "ubgold")]
                cov.append(dict(method=m, dataset=ds, hp=hp, set=("UB9" if ds in UB9 else "secondary"),
                                folds_done=int(sub.fold.nunique()), sources="+".join(sorted(set(sub.source)))))
    pd.DataFrame(cov).to_csv(osp.join(RES, "eval_coverage.csv"), index=False)
    # reproduction check: ORACLE (UB-GOLD hp) 5-fold mean vs UB-GOLD Table 8
    rep = []
    for (m, ds), g in df[df.protocol == "oracle"].groupby(["method", "dataset"]):
        for orient, gg in g.groupby("orientation"):
            pub = PUBLISHED.get((m, ds))
            rep.append(dict(method=m, dataset=ds, orientation=orient, folds=int(gg.fold.nunique()),
                            oracle_mean=gg.auroc.mean(), oracle_std=gg.auroc.std(ddof=0), published=pub,
                            diff=(gg.auroc.mean() - pub) if pub is not None else np.nan,
                            last_mean=df[(df.method == m) & (df.dataset == ds) & (df.protocol == "last") & (df.orientation == orient)].auroc.mean(),
                            valnormal_mean=df[(df.method == m) & (df.dataset == ds) & (df.protocol == "valnormal") & (df.orientation == orient)].auroc.mean()))
    pd.DataFrame(rep).to_csv(osp.join(RES, "eval_repro.csv"), index=False)
    n_new = int((df.source == "evalpaper_harness").sum())
    print(f"eval_folds.csv: {len(df)} rows ({n_new} from the new harness); extra {len(dx)} rows; "
          f"coverage -> eval_coverage.csv; reproduction -> eval_repro.csv")


if __name__ == "__main__":
    main()
