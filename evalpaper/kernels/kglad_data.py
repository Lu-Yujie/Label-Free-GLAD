"""Load the portable fold exports written by export_folds.py (works under numpy 1.x and 2.x).

Returns the same structure as common.data.load_folds(name) (features='ubgold'):
a list of {'train': [graph], 'test': [graph], 'y_test': int64 array, 'fold': int, 'meta': dict},
graph = {'n', 'edges' (m,2) int64 i<j, 'node_label' int64 or None, 'gid'}. The SHA-256 of the
original cache content is recomputed and checked on load.
"""
import os.path as osp
import pickle

import numpy as np

from export_folds import OUT, digest  # same constants / hash as the exporter


def unpack(t):
    if t is None:
        return None
    dt, shape, raw = t
    a = np.frombuffer(raw, dtype=np.dtype(dt)).reshape(shape)
    a.flags.writeable = False
    return a


def load_folds(ds, folds=None, check=True):
    with open(osp.join(OUT, ds + ".pkl"), "rb") as fh:
        obj = pickle.load(fh)
    graphs = {gid: {"n": g["n"], "edges": unpack(g["edges"]), "node_label": unpack(g["node_label"]),
                    "gid": gid, "x": None} for gid, g in obj["graphs"].items()}
    out = []
    for f in obj["folds"]:
        out.append({"train": [graphs[i] for i in f["train"]], "test": [graphs[i] for i in f["test"]],
                    "y_test": np.array(unpack(f["y_test"])), "fold": f["fold"], "meta": f["meta"]})
    if check and folds is None:
        assert digest(out) == obj["sha256"], f"{ds}: export hash mismatch"
    if folds is not None:
        out = [f for f in out if f["fold"] in folds]
    return out
