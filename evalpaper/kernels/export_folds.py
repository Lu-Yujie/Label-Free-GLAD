"""Export the verified UB-GOLD fold caches to a numpy-version-independent pickle.

The caches in $GLAD_DATA_ROOT/folds were pickled under numpy 2.x (grail-prelim env);
UB-GOLD's pinned GraKeL 0.1.10 only runs under numpy 1.x (ubgold-gpu env), which cannot
unpickle numpy-2 arrays. This script (run in grail-prelim) re-serialises exactly what
common.data.load_folds(name) returns (features='ubgold') using builtins only: arrays become
(dtype, shape, raw bytes). kglad_data.load_folds() rebuilds them and the SHA-256 written next
to every export is re-checked there.

  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -B export_folds.py [DS ...]
"""
import hashlib
import os
import os.path as osp
import pickle
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(osp.dirname(HERE)))  # 
from common.data import load_folds, list_datasets  # noqa: E402

OUT = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/evalpaper_kernels/folds")


def pack(a):
    if a is None:
        return None
    a = np.ascontiguousarray(a)
    return (str(a.dtype), tuple(int(s) for s in a.shape), a.tobytes())


def digest(folds):
    h = hashlib.sha256()
    for f in folds:
        h.update(str(f["fold"]).encode())
        h.update(np.asarray(f["y_test"], dtype=np.int64).tobytes())
        for part in ("train", "test"):
            for g in f[part]:
                h.update(np.int64(g["n"]).tobytes())
                h.update(np.int64(g["gid"]).tobytes())
                h.update(np.ascontiguousarray(g["edges"], dtype=np.int64).tobytes())
                if g["node_label"] is not None:
                    h.update(np.ascontiguousarray(g["node_label"], dtype=np.int64).tobytes())
    return h.hexdigest()


def main():
    names = sys.argv[1:] or list_datasets()
    os.makedirs(OUT, exist_ok=True)
    for ds in names:
        folds = load_folds(ds)
        graphs = {}
        fl = []
        for f in folds:
            ids = {}
            for part in ("train", "test"):
                ids[part] = []
                for g in f[part]:
                    gid = int(g["gid"])
                    ids[part].append(gid)
                    if gid not in graphs:
                        graphs[gid] = {"n": int(g["n"]), "edges": pack(np.asarray(g["edges"], dtype=np.int64)),
                                       "node_label": pack(None if g["node_label"] is None
                                                          else np.asarray(g["node_label"], dtype=np.int64)),
                                       "gid": gid}
            meta = {k: v for k, v in f["meta"].items() if isinstance(v, (int, float, str, bool)) or v is None}
            fl.append({"fold": int(f["fold"]), "train": ids["train"], "test": ids["test"],
                       "y_test": pack(np.asarray(f["y_test"], dtype=np.int64)), "meta": meta})
        obj = {"dataset": ds, "graphs": graphs, "folds": fl, "sha256": digest(folds)}
        with open(osp.join(OUT, ds + ".pkl"), "wb") as fh:
            pickle.dump(obj, fh, protocol=4)
        print(ds, len(graphs), "graphs", [len(f["test"]) for f in fl], obj["sha256"][:16], flush=True)


if __name__ == "__main__":
    main()
