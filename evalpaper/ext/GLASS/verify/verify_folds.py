"""Login-node check: the GLASS adapter's own load_fold() gives exactly the harness folds.
  python -B verify_folds.py DS [DS ...]   -> appends one JSON line per (DS, fold) to verify/fold_check.jsonl
For each (DS, fold): load_fold asserts train/test indices and y_test against ubgold_splits/<DS>.json and the graph
identity of every training/test graph against the full TU list; here we additionally compare test_index, y_test,
hold_pos and test/held-out graph sizes with every harness (frozen-12) and ext (E5) score file of that fold."""
import glob
import json
import os
import os.path as osp
import sys
import time

import numpy as np

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, HERE)
import run_method as RM  # noqa: E402

SCORE_DIRS = [(os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper/scores")] + sorted(glob.glob((os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_*/scores")))
SCORE_DIRS = [d for d in SCORE_DIRS if "ext_glass" not in d]


def main():
    H = RM.load_harness()
    out = osp.join(HERE, "verify", "fold_check.jsonl")
    for DS in sys.argv[1:]:
        for fold in range(5):
            t0 = time.time()
            F = RM.load_fold(H, DS, fold)
            full = F["full"]
            test_index = np.asarray(F["test_orig"], dtype=np.int64)
            hold_pos = np.asarray(F["hold_pos"], dtype=np.int64)
            test_nn = np.array([full[i].num_nodes for i in F["test_orig"]])
            hold_nn = np.array([full[i].num_nodes for i in F["hold_orig"]])
            test_ne = np.array([full[i].edge_index.size(1) // 2 for i in F["test_orig"]])
            files = []
            for d in SCORE_DIRS:
                files += sorted(glob.glob(osp.join(d, f"*_{DS}_fold{fold}.npz")) + glob.glob(osp.join(d, f"*_{DS}_fold{fold}_hpfree.npz")))
            ok, bad = 0, []
            for f in files:
                z = np.load(f)
                checks = dict(test_index=np.array_equal(z["test_index"], test_index),
                              y_test=np.array_equal(z["y_test"], F["y_test"]),
                              hold_pos=np.array_equal(z["hold_pos"], hold_pos))
                if "test_n_nodes" in z.files:
                    checks["test_n_nodes"] = np.array_equal(z["test_n_nodes"], test_nn)
                    checks["test_n_edges"] = np.array_equal(z["test_n_edges"], test_ne)
                if "hold_n_nodes" in z.files:
                    checks["hold_n_nodes"] = np.array_equal(z["hold_n_nodes"], hold_nn)
                if all(checks.values()):
                    ok += 1
                else:
                    bad.append((osp.basename(f), {k: v for k, v in checks.items() if not v}))
            rec = dict(DS=DS, fold=fold, n_fit=len(F["fit_orig"]), n_hold=F["n_hold"], n_test=len(test_index),
                       n_test_anom=int(F["y_test"].sum()), files_checked=len(files), files_identical=ok, mismatches=bad,
                       ubgold_splits_and_graph_identity="asserted in load_fold", seconds=round(time.time() - t0, 1),
                       checked=time.strftime("%F %T"))
            print(json.dumps(rec), flush=True)
            with open(out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
