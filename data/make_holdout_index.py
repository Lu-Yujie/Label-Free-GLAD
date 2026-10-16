"""Write data/holdout/<DS>.json: the 20% held-out normal training graphs of every fold (the VAL-NORMAL set).

  python -B data/make_holdout_index.py [--splits data/ubgold_splits] [--out data/holdout] [DS ...]

Rule (evalpaper/harness/run_eval.py, identical in every runner of the study): the normal training graphs of a fold
are taken in UB-GOLD's loader order (train_idx of data/ubgold_splits/<DS>.json for TU datasets; the training-file
graphs for Tox21, the same in every fold), n = their number, n_hold = round(0.2 * n),
perm = numpy.random.RandomState(2026 + 1000 * fold).permutation(n), hold = sorted(perm[:n_hold]),
fit = sorted(perm[n_hold:]). Positions index that loader order; for TU datasets the graph indices into the TU
dataset are train_idx[pos]. Detectors train on the fit positions only.
"""
import argparse
import hashlib
import json
import os
import os.path as osp

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
HOLDOUT_SEED, FRAC, NFOLD = 2026, 0.2, 5


def split(n, fold):
    n_hold = int(round(FRAC * n))
    perm = np.random.RandomState(HOLDOUT_SEED + 1000 * fold).permutation(n)
    return sorted(perm[n_hold:].tolist()), sorted(perm[:n_hold].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=osp.join(HERE, "ubgold_splits"))
    ap.add_argument("--out", default=osp.join(HERE, "holdout"))
    ap.add_argument("datasets", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = a.datasets or sorted(f[:-5] for f in os.listdir(a.splits) if f.endswith(".json") and "check" not in f)
    for ds in names:
        p = osp.join(a.splits, ds + ".json")
        raw = open(p, "rb").read()
        S = json.loads(raw)
        folds = []
        for k in range(NFOLD):
            if S["kind"] == "TU":
                train_idx = [int(i) for i in S["folds"][k]["train_idx"]]
                n = len(train_idx)
            else:                      # Tox21: one fixed split, the training file in loader order
                train_idx, n = None, int(S["n_train"])
            fit, hold = split(n, k)
            rec = dict(fold=k, holdout_seed=HOLDOUT_SEED + 1000 * k, n_train_normal=n, n_fit=len(fit),
                       n_hold=len(hold), hold_pos=hold)
            if train_idx is not None:
                rec["hold_graph_idx"] = [train_idx[i] for i in hold]
            folds.append(rec)
        out = dict(DS=ds, kind=S["kind"], source=f"ubgold_splits/{ds}.json",
                   source_sha256=hashlib.sha256(raw).hexdigest(),
                   rule="n_hold = round(0.2 n); perm = RandomState(2026 + 1000 fold).permutation(n); "
                        "hold = sorted(perm[:n_hold]); fit = the other positions",
                   folds=folds)
        with open(osp.join(a.out, ds + ".json"), "w") as f:
            json.dump(out, f, indent=None, separators=(",", ":"))
            f.write("\n")
        print(ds, [r["n_hold"] for r in folds])


if __name__ == "__main__":
    main()
