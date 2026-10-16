"""Diagnostic only: run GLADPro's own src/main.py (its own splits, anomaly convention and reporting rule,
main.py:201-211) in our environment, with only the runtime compatibility overrides D1, D2, D6 (and D8 on
CPU) of run_method.py. Used to separate porting errors from protocol differences in the reproduction check
(NOTES.md). Not part of the grid; its numbers are never mixed with the UB-GOLD-fold results.

  python -B native_main.py --dataset MUTAG --epochs 500 --lr 1e-4        (README line arguments)
Data: TU raw files copied to $GLAD_DATA_ROOT/ext_gladpro/native/data/<DS>/raw (main.py reads
./data/<DS> relative to the working directory).
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]

import os
import os.path as osp
import runpy
import shutil
import sys

import torch

HERE = osp.dirname(osp.abspath(__file__))
NATIVE = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_gladpro/native")
UB_DATA = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_run/data")

if __name__ == "__main__":
    argv = sys.argv[1:]
    DS = argv[argv.index("--dataset") + 1]
    raw_dst = osp.join(NATIVE, "data", DS, "raw")
    if not osp.isdir(raw_dst):
        shutil.copytree(osp.join(UB_DATA, DS, DS, "raw"), raw_dst)
    sys.path.insert(0, HERE)
    import run_method as RM
    RM.load_gladpro()                                   # D1, D2, D6 overrides; modules from src/
    if not torch.cuda.is_available():
        torch.use_deterministic_algorithms(True, warn_only=True)   # D8
    torch.set_num_threads(int(os.environ.get("EP_THREADS", "4")))
    os.chdir(NATIVE)
    sys.argv = ["main.py"] + argv
    init = {}
    if os.environ.get("NATIVE_SPLIT"):
        # run only GLADPro split k (main.py:161 `for split in range(args.n_split)`, default n_split 5); every
        # other range() call keeps the builtin. Splits run as parallel processes; native_summary.py applies
        # GLADPro's own rule (argmax of the 5-split mean AUC) to the per-epoch log lines.
        import builtins
        K, N = int(os.environ["NATIVE_SPLIT"]), 5

        def _range(*a):
            return [K] if a == (N,) else builtins.range(*a)
        init["range"] = _range
    runpy.run_path(osp.join(RM.GP_SRC, "main.py"), init_globals=init, run_name="__main__")
