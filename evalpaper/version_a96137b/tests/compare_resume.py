"""A17 smoke S2: a stop-and-resume run must equal the straight run bit for bit (NOTES.md §3.5).

  python -B tests/compare_resume.py <straight_rid>:<resumed_rid> [...]   (smoke parts / scores)
  python -B tests/compare_resume.py <rid_1>=<rid_2> [...]                 (repeatability of two straight runs, e.g. GPU)

Compares every trajectory field, the train-loss list and every array of the per-graph score npz. Prints only
IDENTICAL/DIFFERENT, max |diff| per array and the resume bookkeeping (segments, loader-fingerprint checks); no metric
value is printed.
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import os.path as osp  # noqa: E402

import numpy as np  # noqa: E402

SMOKE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper_a96137b/smoke")


def same(x, y):
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return x == y


def main():
    allok = True
    for pair in sys.argv[1:]:
        repeat = "=" in pair
        ra, rb = pair.split("=" if repeat else ":")
        Ja = json.load(open(osp.join(SMOKE, "parts", ra + ".json")))
        Jb = json.load(open(osp.join(SMOKE, "parts", rb + ".json")))
        ta, tb = Ja["traj"], Jb["traj"]
        skip = {"t"}   # wall-clock time of the evaluation point
        traj_ok = len(ta) == len(tb) and all(set(x) == set(y) and all(same(x[k], y[k]) for k in x if k not in skip)
                                             for x, y in zip(ta, tb))
        loss_ok = Ja["train_loss"] == Jb["train_loss"]
        za, zb = np.load(Ja["meta"]["scores_npz"]), np.load(Jb["meta"]["scores_npz"])
        diffs = {}
        npz_ok = set(za.files) == set(zb.files)
        for k in za.files:
            A, B = za[k], zb[k]
            if A.shape != B.shape:
                diffs[k] = "shape"
                npz_ok = False
                continue
            eq = np.array_equal(A, B, equal_nan=A.dtype.kind == "f")
            npz_ok &= eq
            if A.dtype.kind == "f" and A.size:
                d = np.abs(A - B)
                diffs[k] = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
        segs = [(s["device"], s["start_epoch"]) for s in Jb["meta"]["segments"]]
        checks = Jb["meta"].get("a17", {}).get("probe_checks")
        if repeat:
            ok = traj_ok and loss_ok and npz_ok
        else:
            ok = traj_ok and loss_ok and npz_ok and len(segs) >= 2 and checks and all(c["ok"] for c in checks)
        allok &= bool(ok)
        print(f"{ra} vs {rb}: {'IDENTICAL' if ok else 'DIFFERENT'} (traj {traj_ok}, train_loss {loss_ok}, npz {npz_ok}; "
              f"n_eval {len(ta)}; resumed segments {segs}; loader-fingerprint checks {checks}; max|diff| {diffs})")
    print("S2 resume/repeat:", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
