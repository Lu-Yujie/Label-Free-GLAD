"""Score-orientation check for KernelGLAD's OCSVM / IF (fold 0, train-fit, UB-GOLD defaults).

The orientation is fixed from code and documentation, not from test AUROC:
  * UB-GOLD GAOOD/detector/KernelGLAD.py:139  anomaly_scores = -detector.decision_function(K)
  * sklearn OneClassSVM.decision_function: "Signed distance is positive for an inlier and negative
    for an outlier"; IsolationForest.decision_function: "The lower, the more abnormal".
This script adds LABEL-FREE confirmations (columns without 'expl_'):
  * frac_train_decision_lt0: share of training normals with decision < 0 (OCSVM: should be ~nu = 0.1;
    IF with contamination='auto': offset -0.5 on score_samples);
  * spearman_decision_vs_meansim: rank correlation, over test graphs, between the decision value and
    the graph's mean kernel similarity to the training graphs (an inlier score must increase with it).
and EXPLANATORY columns (prefix 'expl_', they use test labels and only explain a result, they never
choose anything): mean similarity to the training graphs for test anomalies vs test normals, and the
AUROC of the anomaly score, of -mean similarity and of the node count.

  PYTHONPATH=$GLAD_ENVS/evalpaper-kernels-site $GLAD_ENVS/ubgold-gpu/bin/python -B orientation_check.py
"""
import csv
import os.path as osp
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import kglad  # noqa: E402
from kglad_data import load_folds  # noqa: E402

OUT = osp.join(osp.dirname(osp.dirname(HERE)), "results", "eval_kernels_orientation_check.csv")
DATASETS = sys.argv[1:] or ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "DD"]


def main():
    rows = []
    for ds in DATASETS:
        f = load_folds(ds)[0]
        y = np.asarray(f["y_test"])
        trg, teg = kglad.to_grakel(f["train"]), kglad.to_grakel(f["test"])
        n_te = np.array([g["n"] for g in f["test"]])
        kernels = ["WL"] + (["PK"] if f["test"][0]["node_label"] is not None else [])
        for kern in kernels:
            Kf, Ks, st, _ = kglad.kernel_matrices(kern, 5, trg, teg, "train", 3407)
            msim = Ks.mean(1)
            for det in ("OCSVM", "IF"):
                np.random.set_state(st)
                d = (OneClassSVM(kernel="precomputed", nu=0.1) if det == "OCSVM"
                     else IsolationForest(n_estimators=200, max_samples=0.5, contamination="auto")).fit(Kf)
                dtr, dte = d.decision_function(Kf), d.decision_function(Ks)
                rows.append({"dataset": ds, "fold": 0, "kernel": kern, "detector": det,
                             "frac_train_decision_lt0": round(float((dtr < 0).mean()), 4),
                             "spearman_decision_vs_meansim": round(float(spearmanr(dte, msim)[0]), 4),
                             "expl_meansim_test_anomalies": round(float(msim[y == 1].mean()), 4),
                             "expl_meansim_test_normals": round(float(msim[y == 0].mean()), 4),
                             "expl_auroc_score": round(100 * roc_auc_score(y, -dte), 2),
                             "expl_auroc_neg_meansim": round(100 * roc_auc_score(y, -msim), 2),
                             "expl_auroc_n_nodes": round(100 * roc_auc_score(y, n_te), 2)})
                print(rows[-1], flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
