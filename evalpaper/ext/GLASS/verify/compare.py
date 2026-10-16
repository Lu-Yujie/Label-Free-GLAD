"""Compare two GLASS runs (part JSON + scores npz): trajectories, per-graph scores and own-rule results.
  python -B compare.py <run_id_a> <run_id_b> [parts_dir] [scores_dir] [--out file.json]"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import json, os.path as osp, sys
import numpy as np
SM = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_glass/smoke")
args = [x for x in sys.argv[1:] if not x.startswith("--")]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
if out in args:
    args.remove(out)
a, b = args[:2]
parts = args[2] if len(args) > 2 else SM + "/parts"
scores = args[3] if len(args) > 3 else SM + "/scores"
JA, JB = json.load(open(osp.join(parts, a + ".json"))), json.load(open(osp.join(parts, b + ".json")))
NA, NB = np.load(osp.join(scores, a + ".npz")), np.load(osp.join(scores, b + ".npz"))
res = dict(a=a, b=b)
TA, TB = JA["traj"], JB["traj"]
res["n_eval"] = [len(TA), len(TB)]
keys = ["auroc", "auprc", "fpr95", "holdout_score", "train_loss", "r0_auroc", "r0_holdout", "glass_ens_auroc",
        "glass_best_ens_auroc", "glass_best_epoch", "glass_patience_counter"]
md = 0.0
for ra, rb in zip(TA, TB):
    for k in keys:
        x, y = ra[k], rb[k]
        if not (x != x and y != y):
            md = max(md, abs(x - y))
res["traj_max_abs_diff"] = md
res["train_loss_max_abs_diff"] = float(np.nanmax(np.abs(np.array(JA["train_loss"]) - np.array(JB["train_loss"]))))
assert sorted(NA.files) == sorted(NB.files)
sd = {}
for k in NA.files:
    x, y = NA[k], NB[k]
    if x.dtype.kind in "fi" and x.shape == y.shape and x.size:
        d = np.atleast_1d(np.abs(x.astype(np.float64) - y.astype(np.float64)))
        both_nan = np.atleast_1d(np.isnan(x.astype(float)) & np.isnan(y.astype(float)))
        d[both_nan] = 0
        sd[k] = float(np.nanmax(d)) if d.size else 0.0
    else:
        sd[k] = None if x.shape == y.shape else f"shape {x.shape} vs {y.shape}"
res["score_arrays"] = len(sd)
res["scores_max_abs_diff"] = max(v for v in sd.values() if isinstance(v, float))
res["nonzero_score_diffs"] = {k: v for k, v in sd.items() if v not in (0.0, None)}
oa, ob = JA["meta"]["own"], JB["meta"]["own"]
res["own_equal"] = json.dumps(oa, sort_keys=True) == json.dumps(ob, sort_keys=True)
res["own_a"] = dict(stop_epoch=oa["stop_epoch"], best_epoch=oa["best_epoch"], best_method=oa["best_method"])
res["segments_b"] = [(s["start_epoch"], s.get("job")) for s in JB["meta"]["segments"]]
print(json.dumps(res, indent=1))
if out:
    json.dump(res, open(out, "w"), indent=1)
