"""Diagnostic only (port fidelity, NOTES.md "Reproduction targets"): GUL's own code path on its own data loader,
splits, anomaly convention and reporting rule (src/main.py, src/data_loader.py), in our environment.
Not part of the grid; its numbers are never mixed with the UB-GOLD-fold results.

  python -B native_main.py --dataset MUTAG --poll_ratio 0 --split K      (any other GUL argparse option may follow)

* Runs GUL's split K only (main.py:86 loops over all 5 splits in one process; one process per split lets the splits
  run in parallel). For split K it calls GUL's own load_data(args, K) and run_split(args, graphs, split_idx, K)
  from src/main.py (imported as a module; its __main__ block is not executed) and stores the per-epoch test AUC list
  (GUL's compute_metrics) in $GLAD_DATA_ROOT/ext_sulgul/native/out/<DS>_poll<p>_s<K>.json.
  native_summary.py then applies GUL's own rule (main.py:94-104: argmax over epochs of the 5-split mean AUC).
* Only overrides: matplotlib stub (D6), seeding (python/numpy/torch with 3407 + 1000*poll_flag + K before the
  split, for repeatability; GUL sets no seed), deterministic algorithms on CPU (D8). The data root is
  $GLAD_DATA_ROOT/ext_sulgul/native/data (main.py reads ./data/<DS> relative to the working directory);
  the TU raw files are copied there (MUTAG: the repo's own shipped raw files) and processed by the login-node
  `--prepare` call before any job, so parallel splits never race on processing.
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]

import json
import os
import os.path as osp
import random
import shutil
import sys
import time

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
NATIVE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_sulgul/native")
UB_DATA = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_run/data")


def prepare_raw(DS):
    raw_dst = osp.join(NATIVE, "data", DS, "raw")
    if not osp.isdir(raw_dst):
        src = osp.join(HERE, "src", "data", DS, "raw") if DS == "MUTAG" else osp.join(UB_DATA, DS, DS, "raw")
        shutil.copytree(src, raw_dst)
    return raw_dst


def main():
    argv = sys.argv[1:]
    prepare = "--prepare" in argv
    argv = [a for a in argv if a != "--prepare"]
    K = int(argv[argv.index("--split") + 1])
    i = argv.index("--split")
    argv = argv[:i] + argv[i + 2:]
    DS = argv[argv.index("--dataset") + 1]
    prepare_raw(DS)
    sys.path.insert(0, HERE)
    import run_method as RM
    utils, layers, model = RM.load_gul()          # D6 stub; modules from src/
    sys.path.insert(0, RM.GUL_SRC)
    import main as gul_main                        # src/main.py (guarded __main__ block not executed)
    import data_loader as gul_dl
    for mod in (gul_main, gul_dl):
        assert osp.realpath(mod.__file__).startswith(osp.realpath(RM.GUL_SRC)), mod.__file__
    if not torch.cuda.is_available():
        torch.use_deterministic_algorithms(True, warn_only=True)   # D8
    torch.set_num_threads(int(os.environ.get("EP_THREADS", "4")))
    os.makedirs(osp.join(NATIVE, "out"), exist_ok=True)
    os.chdir(NATIVE)
    sys.argv = ["main.py"] + argv
    args = utils.load_args()
    if prepare:                                    # login node: process the TU raw files once, no training
        graphs, split_idx = gul_dl.load_data(args, K)
        print("prepared", DS, len(graphs), "train", len(split_idx[0]), "test", len(split_idx[1]))
        return
    seed = 3407 + 1000 * int(args.poll_ratio != 0) + K
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    # verbatim body of main.py's __main__ loop for one split
    print(f'Start fold:{K}')
    graphs, split_idx = gul_dl.load_data(args, K)
    auc_fold = gul_main.run_split(args, graphs, split_idx, K)
    print(f'Finish fold:{K}')
    print(args)
    out = dict(dataset=DS, split=K, poll_ratio=args.poll_ratio, seed=seed, n_train=len(split_idx[0]),
               n_test=len(split_idx[1]), n_test_anom=int(sum(int(graphs[j].y) for j in split_idx[1])),
               n_train_anom=int(sum(int(graphs[j].y) for j in split_idx[0])), argv=argv,
               args={k: str(v) for k, v in vars(args).items()}, auc=[float(a) for a in auc_fold],
               seconds=time.time() - t0, threads=torch.get_num_threads(), gul_commit=RM.GUL_COMMIT,
               completed=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = osp.join(NATIVE, "out", f"{DS}_poll{args.poll_ratio:g}_s{K}.json")
    json.dump(out, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    print("wrote", p, f"last auc {auc_fold[-1]:.4f} max {max(auc_fold):.4f}")


if __name__ == "__main__":
    main()
