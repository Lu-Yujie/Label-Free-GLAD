"""POST HOC (designed after seeing that OCGTL / OCGIN detect T2; not pre-registered; does not affect
the verdict): are T2 anomalies visible to trivial label-aware 1-hop / 1-WL statistics that the
original synthetic study did not include among its controls?

Controls (fitted on the 500 training normals only, fixed parameters, nothing tuned):
  ctrl_wl_iforest      grail0/controls.py's WL-subtree kernel + IsolationForest (3 WL iterations;
                       initial colour = node label on GX, degree on Syn-Com / BM; WL relabeling run
                       jointly over train + test exactly as grail0/run.py does on the TU folds).
  ctrl_edgelab_lw      log(1 + counts of the 10 unordered node-label pairs over edges) -> Ledoit-Wolf
                       Mahalanobis with channel A's ridge (GX only).
  ctrl_edgelab_iforest same features -> IsolationForest (GX only).
Also descriptive: mean number of edges per graph for every label pair, normals vs each GX type
(seed 0..4 pooled).
Output: results/m2_posthoc_controls.csv (dataset, anomaly_type, seed, method, auroc, auprc,
fpr95) and results/m2_posthoc_edgelabel_counts.csv.
Usage (CPU venv, PYTHONPATH=prelim): python -B m2_typed/posthoc_controls.py
"""
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PRELIM = osp.dirname(HERE)
RES = osp.join(PRELIM, "results")
sys.path.insert(0, PRELIM)
sys.path.insert(0, HERE)
import m2io  # noqa: E402

PAIRS = [(a, b) for a in range(4) for b in range(a, 4)]


def edgelab(graphs):
    out = np.zeros((len(graphs), len(PAIRS)))
    idx = {p: i for i, p in enumerate(PAIRS)}
    for i, g in enumerate(graphs):
        lab = np.asarray(g["node_label"])
        e = np.asarray(g["edges"]).reshape(-1, 2)
        a, b = np.minimum(lab[e[:, 0]], lab[e[:, 1]]), np.maximum(lab[e[:, 0]], lab[e[:, 1]])
        for x, y in zip(a, b):
            out[i, idx[(int(x), int(y))]] += 1
    return out


def run(ds, seed):
    from common.metrics import evaluate
    from grail0 import controls as C
    from grail0.channels import CONFIG
    train, test, group, names = m2io.load(ds, seed)
    allg = train + test
    labs, off = C.wl_labels(allg, h=3)
    Xs = C.wl_sparse(labs, off)
    Ktr, Kte = C.wl_kernel_feats(Xs, list(range(len(train))), list(range(len(train), len(allg))), seed)
    meth = {"ctrl_wl_iforest": C.iforest(Ktr, Kte, seed)}
    cnt_rows = []
    if ds == "GX":
        Etr, Ete = edgelab(train), edgelab(test)
        meth["ctrl_edgelab_lw"] = C.mahal(np.log1p(Etr), np.log1p(Ete), CONFIG["ridge_A"])
        meth["ctrl_edgelab_iforest"] = C.iforest(np.log1p(Etr), np.log1p(Ete), seed)
        for k, nm in enumerate(names):
            mu = Ete[group == k].mean(0)
            cnt_rows.append({"seed": seed, "group": nm, **{"%d-%d" % p: float(v) for p, v in zip(PAIRS, mu)}})
    rows = []
    for k in range(1, len(names)):
        sel = (group == 0) | (group == k)
        y = (group[sel] == k).astype(int)
        for m, s in meth.items():
            rows.append({"dataset": ds, "anomaly_type": names[k], "seed": seed, "method": m,
                         **evaluate(np.asarray(s)[sel], y)})
    return rows, cnt_rows


def main():
    rows, cnts = [], []
    for ds in m2io.DATASETS:
        for s in m2io.SEEDS:
            r, c = run(ds, s)
            rows += r; cnts += c
            print("[posthoc] %s s%d" % (ds, s), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(osp.join(RES, "m2_posthoc_controls.csv"), index=False)
    cdf = pd.DataFrame(cnts)
    cdf.to_csv(osp.join(RES, "m2_posthoc_edgelabel_counts.csv"), index=False)
    s = df.groupby(["dataset", "anomaly_type", "method"]).auroc.agg(["mean", "std"]).mul(100).round(1)
    print(s.to_string())
    print(cdf.groupby("group").mean(numeric_only=True).drop(columns="seed").round(2).to_string())


if __name__ == "__main__":
    main()
