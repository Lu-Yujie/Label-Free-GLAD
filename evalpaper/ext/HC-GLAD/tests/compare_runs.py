"""Compare two finished runs (parts JSON + scores npz) for bit-identity. Usage: compare_runs.py PARTS SCORES runA runB"""
import json, sys, os.path as osp
import numpy as np
parts, scores, a, b = sys.argv[1:5]
A = json.load(open(osp.join(parts, a + ".json"))); B = json.load(open(osp.join(parts, b + ".json")))
ta, tb = A["traj"], B["traj"]
assert [r["epoch"] for r in ta] == [r["epoch"] for r in tb], "eval epochs differ"
md = 0.0
for ra, rb in zip(ta, tb):
    for k in ra:
        if k == "t":
            continue
        va, vb = ra[k], rb[k]
        if isinstance(va, float) and isinstance(vb, float):
            if not (np.isnan(va) and np.isnan(vb)):
                md = max(md, abs(va - vb))
        else:
            assert va == vb, (k, va, vb)
tl = max(abs(x - y) for x, y in zip(A["train_loss"], B["train_loss"]))
na, nb = np.load(osp.join(scores, a + ".npz")), np.load(osp.join(scores, b + ".npz"))
mds = 0.0
for k in na.files:
    if na[k].dtype.kind == "f":
        mds = max(mds, float(np.nanmax(np.abs(na[k] - nb[k]))) if na[k].size else 0.0)
    else:
        assert np.array_equal(na[k], nb[k]), k
print(f"{a} vs {b}: n_eval={len(ta)} max|diff| traj={md:.3g} train_loss={tl:.3g} scores={mds:.3g} "
      f"segments={[s['start_epoch'] for s in B['meta']['segments']]} auroc={[round(r['auroc'],4) for r in ta]} "
      f"auroc_own==auroc: {all(abs(r['auroc']-r['auroc_own'])<1e-12 for r in ta)}")
