"""Compare two finished LGKDE runs (part JSON + score npz) bitwise: per-epoch trajectory values and per-graph
scores. Usage: python -B tests/compare_runs.py PARTS_DIR SCORES_DIR RUN_ID_A RUN_ID_B"""
import json
import os.path as osp
import sys

import numpy as np

KEYS = ["auroc", "auprc", "fpr95", "lgkde_auroc", "lgkde_aupr", "own_sel", "holdout_score", "train_loss",
        "test_density_mean", "kde_alpha"]


def flat(traj):
    return np.array([v for r in traj for k in KEYS for v in np.ravel(r[k])], dtype=np.float64)


def main():
    parts, scores, a, b = sys.argv[1:5]
    Ja, Jb = (json.load(open(osp.join(parts, x + ".json"))) for x in (a, b))
    Za, Zb = (np.load(osp.join(scores, x + ".npz")) for x in (a, b))
    fa, fb = flat(Ja["traj"]), flat(Jb["traj"])
    out = dict(a=a, b=b, epochs=(len(Ja["traj"]), len(Jb["traj"])),
               traj_max_abs_diff=float(np.max(np.abs(fa - fb))) if fa.shape == fb.shape else "shape mismatch",
               test_scores_max_abs_diff=float(np.max(np.abs(Za["test"] - Zb["test"]))),
               hold_scores_max_abs_diff=float(np.max(np.abs(Za["hold"] - Zb["hold"]))),
               segments_b=[s["start_epoch"] for s in Jb["meta"]["segments"]])
    out["bitwise_identical"] = (out["traj_max_abs_diff"] == 0.0 and out["test_scores_max_abs_diff"] == 0.0
                                and out["hold_scores_max_abs_diff"] == 0.0)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
