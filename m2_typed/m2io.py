"""Read the M2 data sets written by export_data.py (plain .npz, no pickle; numpy-version agnostic)."""
import os  # release: GLAD_DATA_ROOT / GLAD_ENVS
import os.path as osp

import numpy as np

DATA = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/m2_data")
SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["GX", "SC20-a0.60", "SC20-a0.80", "SC20-a0.20", "BM-MT", "BM-MN", "BM-MS"]


def npz_path(ds, seed):
    return osp.join(DATA, "%s_s%d.npz" % (ds, seed))


def unpack(Z, pre):
    """Inverse of export_data.pack(): list of graph dicts in the common.data / gens format."""
    n, m, gid = Z[pre + "n"], Z[pre + "m"], Z[pre + "gid"]
    no = np.r_[0, np.cumsum(n)]
    mo = np.r_[0, np.cumsum(m)]
    out = []
    for i in range(len(n)):
        g = {"n": int(n[i]), "edges": np.ascontiguousarray(Z[pre + "edges"][mo[i]:mo[i + 1]]).astype(np.int64),
             "x": None, "node_label": None, "gid": int(gid[i])}
        if pre + "x" in Z:
            g["x"] = np.ascontiguousarray(Z[pre + "x"][no[i]:no[i + 1]]).astype(np.float32)
        if pre + "node_label" in Z:
            g["node_label"] = np.ascontiguousarray(Z[pre + "node_label"][no[i]:no[i + 1]]).astype(np.int64)
        for k, dt in (("role", np.int64), ("gt_node", bool)):
            if pre + k in Z:
                g[k] = np.ascontiguousarray(Z[pre + k][no[i]:no[i + 1]]).astype(dt)
        out.append(g)
    return out


def load(ds, seed):
    """-> train (list of graph dicts), test (list), group (test group codes, 0 = normal), names"""
    with np.load(npz_path(ds, seed), allow_pickle=False) as Z:
        Z = {k: Z[k] for k in Z.files}
    return unpack(Z, "tr_"), unpack(Z, "te_"), Z["te_group"].astype(np.int64), [str(s) for s in Z["group_names"]]
