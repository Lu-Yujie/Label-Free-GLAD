"""[copy of ext/GLADPro/verify/check_fold_cache.py for the SUL-GUL adapter] Compare the adapter's folds (verify/folds_*.json from verify_login.py) with
  (a) the common fold cache $GLAD_DATA_ROOT/folds/<DS>.pkl (train normals, test, y_test);
  (b) m1_selection score files $GLAD_DATA_ROOT/m1_scores/*_<DS>_fold<k>.npz
      (test_index, y_test, hold_pos: the deep_rerun/m1/harness held-out carve-out).
Run with the grail-prelim env (the cache pickles need numpy 2):
  $GLAD_ENVS/grail-prelim/bin/python -B verify/check_fold_cache.py
"""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import glob
import json
import os.path as osp
import pickle

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
res = {}
for fj in sorted(glob.glob(osp.join(HERE, "folds_*.json"))):
    for key, r in json.load(open(fj)).items():
        DS, k = key.split("/")
        k = int(k)
        cache = pickle.load(open((os.environ.get("GLAD_DATA_ROOT", "glad_data") + f"/folds/{DS}.pkl"), "rb"))
        cf = cache["folds"][k]
        assert np.array_equal(np.asarray(cf["test_idx"]), np.asarray(r["split_test_index"])), key
        assert np.array_equal(np.asarray(cf["y_test"]), np.asarray(r["y_test"])), key
        tr = set(map(int, cf["train_idx"]))
        ours_normals = [i for i in r["split_train_index"] if i in tr]
        assert ours_normals == sorted(map(int, cf["train_idx"])), key
        assert len(ours_normals) == r["n_fit"] + r["n_hold"], key
        m1 = []
        for f in sorted(glob.glob((os.environ.get("GLAD_DATA_ROOT", "glad_data") + f"/m1_scores/*_{DS}_fold{k}.npz"))):
            z = np.load(f)
            assert np.array_equal(z["test_index"], np.asarray(r["split_test_index"])), f
            assert np.array_equal(z["y_test"], np.asarray(r["y_test"])), f
            assert np.array_equal(z["hold_pos"], np.asarray(r["hold_pos"])), f
            m1.append(osp.basename(f))
        res[key] = dict(cache="ok", n_train_normals=len(ours_normals), m1_npz_checked=len(m1),
                        harness_npz_checked=len(r["harness_npz_checked"]))
for key, v in res.items():
    print(key, v)
json.dump(res, open(osp.join(HERE, "fold_check_summary.json"), "w"), indent=1)
print(f"ALL OK: {len(res)} (dataset, fold) pairs")
