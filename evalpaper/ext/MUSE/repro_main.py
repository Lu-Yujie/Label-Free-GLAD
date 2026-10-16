"""R1 reproduction (NOTES.md 1.5): run MUSE's own src/main.py UNMODIFIED (its own splits, features, labelled
validation and selection) and print its result. Two environment shims only, no change to MUSE's code:
  * the TUDataset root '/tmp/ENZYMES' hard-coded in src/data_loader.py is redirected to
    $GLAD_DATA_ROOT/ext_muse/tu (raw TU files symlinked from the UB-GOLD data dir; PyG writes its
    processed/ files there);
  * PyG 2.6.1 has no module `torch_geometric.utils.subgraph` (data_loader.py imports it but never uses it);
    the name is aliased to `torch_geometric.utils._subgraph`.

  python -B repro_main.py -data bzr -anom_type 0 -lr 0.0001 -dim 32 -n_layers 4 -gamma 2.0 -device cpu
"""
import os.path as _osp, sys as _sys
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]
_sys.dont_write_bytecode = True

import os
import os.path as osp
import runpy
import sys
import time

SRC = osp.join(_HERE_, "src")
TU_ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ext_muse/tu")
UB_DATA = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_run/data")
TU_NAME = {"bzr": "BZR", "dhfr": "DHFR", "aids": "AIDS", "nci1": "NCI1", "imdb": "IMDB-BINARY",
           "reddit": "REDDIT-BINARY", "dd": "DD"}


def link_raw(name):
    """ext_muse/tu/<name>/raw/* -> symlinks to UB-GOLD's raw TU files (read-only source)."""
    src = osp.join(UB_DATA, name, name, "raw")
    dst = osp.join(TU_ROOT, name, "raw")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if not osp.exists(osp.join(dst, f)):
            os.symlink(osp.join(src, f), osp.join(dst, f))


def r1_done(data):
    """True if a finished R1 log for this dataset exists (any job): a later job then skips it (no duplicate work)."""
    import glob
    for fn in glob.glob(osp.join(_HERE_, "logs", f"r1_{data}_*.log")):
        txt = open(fn).read()
        if "Average of AUROC" in txt and "== R1 done" in txt:
            return fn
    return None


def main():
    argv = sys.argv[1:]
    data = argv[argv.index("-data") + 1]
    done = r1_done(data)
    if done:
        print(f"== R1 {data}: already finished in {osp.basename(done)}; skipped", flush=True)
        return
    link_raw(TU_NAME[data])
    import torch_geometric.utils._subgraph as _sg
    sys.modules.setdefault("torch_geometric.utils.subgraph", _sg)      # PyG >= 2.3 module rename (shim)
    sys.path.insert(0, SRC)
    import data_loader                                                 # MUSE's src/data_loader.py
    from torch_geometric.datasets import TUDataset as _TU

    def TUDataset(root, name, **kw):                                   # shim: redirect the dataset root only
        assert root == "/tmp/ENZYMES", root
        return _TU(TU_ROOT, name, **kw)
    data_loader.TUDataset = TUDataset
    sys.argv = [osp.join(SRC, "main.py")] + argv
    t0 = time.time()
    print(f"== R1 MUSE main.py {' '.join(argv)} start {time.strftime('%F %T')}", flush=True)
    runpy.run_path(osp.join(SRC, "main.py"), run_name="__main__")
    print(f"== R1 done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
