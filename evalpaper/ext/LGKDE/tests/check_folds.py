"""Folds used by the LGKDE adapter == harness folds (login node, <= 2 min per call).

For each (dataset, fold): load the fold through ubgold_data.load_fold (the harness's own UB-GOLD loader code,
which also asserts the split indices against $GLAD_DATA_ROOT/ubgold_splits) and compare with
  (a) every harness score file for that (dataset, fold) in $GLAD_DATA_ROOT/evalpaper/scores
      (y_test, test_index, hold_pos, per-graph node/edge counts of test and held-out graphs);
  (b) the common fold cache ($GLAD_DATA_ROOT/folds): test graph ids/labels and the set of
      normal training graphs.
Usage: python -B tests/check_folds.py --datasets BZR,COX2 [--folds 0,1,2,3,4] --out tests/folds_<x>.json
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import argparse
import glob
import json
import os.path as osp
import pickle
import sys
import time

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import ubgold_data

HSCORES = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper/scores")
FOLDCACHE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_lgkde/foldcache_idx")   # JSON export of $GLAD_DATA_ROOT/folds/<DS>.pkl (numpy-2 pickles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    H = ubgold_data.load_harness()
    res = {}
    for DS in a.datasets.split(","):
        cache = dict(folds=json.load(open(osp.join(FOLDCACHE, DS + ".json"))))
        for fo in map(int, a.folds.split(",")):
            t0 = time.time()
            fd = ubgold_data.load_fold(DS, fo, H=H)
            ours = dict(y_test=fd["y_test"], test_index=fd["split_test_index"], hold_pos=fd["hold_pos"],
                        test_n_nodes=np.array([g["n"] for g in fd["test"]]),
                        test_n_edges=np.array([g["edge_index"].size(1) // 2 for g in fd["test"]]),
                        hold_n_nodes=np.array([g["n"] for g in fd["hold"]]),
                        hold_n_edges=np.array([g["edge_index"].size(1) // 2 for g in fd["hold"]]))
            r = dict(split_assert_vs_ubgold_splits="passed (asserted in load_fold)", n_fit=len(fd["fit"]),
                     n_hold=len(fd["hold"]), n_test=len(fd["test"]), harness_files=[])
            for fn in sorted(glob.glob(osp.join(HSCORES, f"*_{DS}_fold{fo}.npz")) +
                             sorted(glob.glob(osp.join(HSCORES, f"*_{DS}_fold{fo}_hpfree.npz")))):
                z = np.load(fn)
                ok = {k: bool(np.array_equal(np.asarray(z[k]), ours[k])) for k in ours if k in z.files}
                r["harness_files"].append(dict(file=osp.basename(fn), equal=ok, all_equal=all(ok.values())))
            cf = cache["folds"][fo]
            # cache: test gids in UB-GOLD test order, y_test; normal training gids
            ctest = np.asarray(cf.get("test_idx", cf.get("test_gids", [])))
            cy = np.asarray(cf.get("y_test", []))
            ctrain = np.asarray(cf.get("train_idx", cf.get("train_gids", [])))
            r["cache_keys"] = sorted(cf.keys())
            r["cache_test_idx_equal"] = bool(np.array_equal(ctest, ours["test_index"])) if ctest.size else None
            r["cache_y_test_equal"] = bool(np.array_equal(cy, ours["y_test"])) if cy.size else None
            r["cache_n_train_normal_equal"] = bool(ctrain.size == fd["n_all"]) if ctrain.size else None
            r["seconds"] = round(time.time() - t0, 1)
            res[f"{DS}_fold{fo}"] = r
            print(DS, fo, json.dumps({k: v for k, v in r.items() if k != "harness_files"}),
                  "harness:", [(h["file"], h["all_equal"]) for h in r["harness_files"]], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
