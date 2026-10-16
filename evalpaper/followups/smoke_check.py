"""A33 smoke checker (grail-prelim env): every smoke part under $GLAD_DATA_ROOT/followups/*/smoke must carry
the A33 marker, have finite test/held-out scores (except x_s=0 runs, which are reported, A33 NOTES §7), flipped labels
equal to the fold cache (orientation), x_s checks > 0 (ablation), and every <x>full / <x>res pair must agree exactly
(training losses and every score array). Prints PASS/FAIL lines only, never a metric.
  $GLAD_ENVS/grail-prelim/bin/python -B evalpaper/followups/smoke_check.py [--tag_prefix gsmoke]
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import argparse
import glob
import json
import os.path as osp
import sys

import numpy as np

D = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/followups")
ap = argparse.ArgumentParser()
ap.add_argument("--tag_prefix", default="")
a = ap.parse_args()
fails, n = [], 0
parts = sorted(glob.glob(f"{D}/orientation/smoke/parts/*.json") + glob.glob(f"{D}/ablation_xs/smoke/*/*/parts/*.json"))
byrid = {}
for f in parts:
    J = json.load(open(f))
    m = J["meta"]
    if not str(m.get("tag") or m["run_id"].rsplit("_", 1)[-1]).startswith(a.tag_prefix):
        continue
    n += 1
    rid, a33 = m["run_id"], m.get("a33", {})
    z = np.load(m["scores_npz"])
    keys = [k for k in z.files if k == "test" or k == "hold" or k.startswith("test__") or k.startswith("testnorm__")]
    finite = all(np.isfinite(z[k]).all() for k in keys)
    dev = "+".join(sorted({"cuda" if "cuda" in s["device"] else "cpu" for s in m["segments"]}))
    ok = bool(a33) and dev == a33.get("planned_device", dev)
    note = []
    if "orientation" in f:
        cache = json.load(open(f"{D}/orientation/folds/{m['dataset']}.json"))["folds"][m["fold"]]
        ok &= list(z["y_test"]) == cache["y_test"] and bool(a33.get("fold_check", {}).get("ok"))
        ok &= finite
    else:
        ok &= (a33.get("xs_zero_checks") or 0) > 0 and bool(a33.get("xs_shape_check"))
        if a33.get("variant") == "x_s=1":
            ok &= finite
        else:
            note.append(f"x_s=0 finite={finite}")
    print(f"{'PASS' if ok else 'FAIL'} {f.split('/followups/')[1]} dev={dev} segs={[s['start_epoch'] for s in m['segments']]} {' '.join(note)}")
    if not ok:
        fails.append(rid)
    byrid[osp.join(osp.dirname(f), rid)] = (J, z, keys)
for k, (J, z, keys) in byrid.items():
    if k.endswith("full"):
        r = k[:-4] + "res"
        if r in byrid:
            J2, z2, _ = byrid[r]
            same = J["train_loss"] == J2["train_loss"] and all(np.array_equal(z[x], z2[x], equal_nan=True) for x in keys)
            print(f"{'PASS' if same else 'FAIL'} resume identical: {osp.basename(k)} vs {osp.basename(r)}")
            if not same:
                fails.append(osp.basename(r))
print(f"{n} smoke parts checked; failures: {fails}")
sys.exit(1 if fails else 0)
