"""Run UB-GOLD's OWN split/loader code (PyG route) and export the index lists.

The functions ``get_ad_split_TU``, ``get_ad_dataset_TU``, ``read_graph_file``,
``get_ad_dataset_Tox21`` and ``init_structural_encoding`` are extracted
verbatim (via ``ast``) from the official repository's
``dataloader/data_loader.py`` (cloned to $GLAD_DATA_ROOT/UB-GOLD)
and executed unchanged. The only instrumentation is a TUDataset subclass that
tags every Data object with its original index (``orig_idx``) so the filtered
train list can be mapped back to dataset indices. ``need_str_enc=False`` is
passed (structural encodings only add the ``x_s`` field, they do not affect
membership).

Every graph UB-GOLD produces is compared against our native reader
(common.tu_io): number of nodes, undirected edge set, and node-label one-hot
(argmax) / constant features.

Usage (run with -I because it reads downloaded data / third-party source):
  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -I -B \
      common/ubgold_xcheck.py AIDS BZR ...
Output: $GLAD_DATA_ROOT/ubgold_splits/<DS>.json
"""
import ast
import json
import os
import os.path as osp
import random
import re
import sys
import time
import types

PRELIM = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, PRELIM)

import numpy as np
import networkx as nx
import torch
from scipy import sparse as sp
from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx

from common.tu_io import read_tu_pyg_like, read_tox21_ubgold_like

DATA_ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data"))
UBG_SRC = osp.join(DATA_ROOT, "UB-GOLD", "dataloader", "data_loader.py")
RUN_DIR = osp.join(DATA_ROOT, "ubgold_run")       # cwd for UB-GOLD code (expects ./data/<DS>)
OUT_DIR = osp.join(DATA_ROOT, "ubgold_splits")
RAW = osp.join(DATA_ROOT, "raw")
WANT = ["read_graph_file", "init_structural_encoding", "get_ad_split_TU",
        "get_ad_dataset_TU", "get_ad_dataset_Tox21"]


class TaggedTUDataset(TUDataset):
    """TUDataset that records the original graph index on each Data object."""

    def get(self, idx):
        d = super().get(idx)
        d.orig_idx = int(idx)
        return d


def load_ubgold_functions():
    src = open(UBG_SRC).read()
    tree = ast.parse(src)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANT]
    assert sorted(f.name for f in funcs) == sorted(WANT), [f.name for f in funcs]
    mod = ast.Module(body=funcs, type_ignores=[])
    code = compile(mod, UBG_SRC, "exec")
    ns = dict(os=os, osp=osp, re=re, sp=sp, torch=torch, np=np, nx=nx, random=random,
              DataLoader=DataLoader, Constant=Constant, TUDataset=TaggedTUDataset,
              to_scipy_sparse_matrix=to_scipy_sparse_matrix, degree=degree,
              from_networkx=from_networkx, StratifiedKFold=StratifiedKFold)
    exec(code, ns)
    return ns


def setup_run_dir(DS):
    """Lay out ./data/<DS>/<DS>/raw (PyG) or ./data/<DS>_training (Tox21) via symlinks."""
    os.makedirs(osp.join(RUN_DIR, "data"), exist_ok=True)
    if DS.startswith("Tox21"):
        for part in ("training", "testing"):
            link = osp.join(RUN_DIR, "data", f"{DS}_{part}")
            if not osp.exists(link):
                os.symlink(osp.join(RAW, f"{DS}_{part}"), link)
    else:
        rawdir = osp.join(RUN_DIR, "data", DS, DS, "raw")
        os.makedirs(rawdir, exist_ok=True)
        for f in os.listdir(osp.join(RAW, DS)):
            if f.startswith(DS + "_") and f.endswith(".txt"):
                dst = osp.join(rawdir, f)
                if not osp.exists(dst):
                    os.symlink(osp.join(RAW, DS, f), dst)


def undirected_set(edge_index):
    ei = edge_index.numpy() if hasattr(edge_index, "numpy") else np.asarray(edge_index)
    if ei.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    lo = np.minimum(ei[0], ei[1]); hi = np.maximum(ei[0], ei[1])
    keep = lo != hi
    pairs = np.unique(np.stack([lo[keep], hi[keep]], 1), axis=0)
    return pairs.astype(np.int64)


def compare_graph(d, g, has_node_labels):
    """Return list of mismatch strings (empty if identical)."""
    bad = []
    if int(d.num_nodes) != g["n"]:
        bad.append(f"n {int(d.num_nodes)} vs {g['n']}")
        return bad
    e_ub = undirected_set(d.edge_index)
    e_nat = g["edges"]
    if e_ub.shape != e_nat.shape or not np.array_equal(e_ub, e_nat):
        bad.append(f"edges {e_ub.shape} vs {e_nat.shape}")
    x = d.x.numpy()
    if has_node_labels:
        if not np.array_equal(x.argmax(1), g["node_label"]) or not np.all(x.sum(1) == 1):
            bad.append("node-label one-hot mismatch")
    else:
        if not np.all(x == 1) or x.shape[1] != 1:
            bad.append("expected constant x")
    return bad


def run_tu(ns, DS):
    t0 = time.time()
    graphs, y, info = read_tu_pyg_like(osp.join(RAW, DS), DS)
    has_nl = DS not in ("IMDB-BINARY", "REDDIT-BINARY", "COLLAB")
    args = types.SimpleNamespace(DS=DS, batch_size=128, batch_size_test=9999, rw_dim=16, dg_dim=16)
    splits = ns["get_ad_split_TU"](args, fold=5)
    folds, mism = [], []
    num_feat = None
    for k in range(5):
        out = ns["get_ad_dataset_TU"](args, splits[k], need_str_enc=False)
        dl_train, dl_test, meta = out[3], out[5], out[6]
        num_feat = meta["num_feat"]
        tr = [int(d.orig_idx) for d in dl_train.dataset]
        te = [int(d.orig_idx) for d in dl_test.dataset]
        yt = [int(d.y) for d in dl_test.dataset]
        for d in list(dl_train.dataset) + list(dl_test.dataset):
            b = compare_graph(d, graphs[int(d.orig_idx)], has_nl)
            if b:
                mism.append((k, int(d.orig_idx), b))
        folds.append({"fold": k,
                      "split_train_index": [int(i) for i in splits[k][0]],
                      "split_test_index": [int(i) for i in splits[k][1]],
                      "train_idx": tr, "test_idx": te, "y_test": yt,
                      "num_train_meta": int(meta["num_train"])})
    res = {"DS": DS, "kind": "TU", "num_feat": int(num_feat), "n_graphs": len(graphs),
           "native_info": info, "n_graph_mismatches": len(mism), "mismatch_examples": mism[:20],
           "folds": folds, "seconds": round(time.time() - t0, 1),
           "versions": {"torch": torch.__version__, "torch_geometric": __import__("torch_geometric").__version__,
                        "sklearn": __import__("sklearn").__version__}}
    return res


def run_tox21(ns, DS):
    t0 = time.time()
    args = types.SimpleNamespace(DS=DS, batch_size=128, batch_size_test=9999, rw_dim=16, dg_dim=16)
    out = ns["get_ad_dataset_Tox21"](args, need_str_enc=False)
    data_train, data_test, meta = out[0], out[2], out[6]
    g_tr, y_tr_m, _, info_tr = read_tox21_ubgold_like(osp.join(RAW, f"{DS}_training"), f"{DS}_training")
    g_te, y_te_m, _, info_te = read_tox21_ubgold_like(osp.join(RAW, f"{DS}_testing"), f"{DS}_testing")
    nat_train = [g for g, yy in zip(g_tr, y_tr_m) if yy == 1]
    nat_test = g_te
    nat_ytest = [1 if yy == 1 else 0 for yy in y_te_m]
    mism = []
    if len(nat_train) != len(data_train):
        mism.append(("train_len", len(data_train), len(nat_train)))
    if len(nat_test) != len(data_test):
        mism.append(("test_len", len(data_test), len(nat_test)))
    for i, (d, g) in enumerate(zip(data_train, nat_train)):
        b = compare_graph(d, g, has_node_labels=False)
        if b:
            mism.append(("train", i, b))
    yt = [int(d.y) for d in data_test]
    for i, (d, g) in enumerate(zip(data_test, nat_test)):
        b = compare_graph(d, g, has_node_labels=False)
        if b:
            mism.append(("test", i, b))
    if yt != nat_ytest:
        mism.append(("y_test differs",))
    res = {"DS": DS, "kind": "Tox21", "num_feat": int(meta["num_feat"]),
           "train_gid": [g["gid"] for g in nat_train], "test_gid": [g["gid"] for g in nat_test],
           "y_test": yt, "n_train": len(data_train), "n_test": len(data_test),
           "n_test_anom": int(sum(yt)), "native_info_train": info_tr, "native_info_test": info_te,
           "n_graph_mismatches": len(mism), "mismatch_examples": mism[:20],
           "seconds": round(time.time() - t0, 1)}
    return res


def main(names):
    os.makedirs(OUT_DIR, exist_ok=True)
    ns = load_ubgold_functions()
    for DS in names:
        setup_run_dir(DS)
        cwd = os.getcwd()
        os.chdir(RUN_DIR)                     # UB-GOLD resolves ./data/<DS> from cwd
        try:
            res = run_tox21(ns, DS) if DS.startswith("Tox21") else run_tu(ns, DS)
        finally:
            os.chdir(cwd)
        with open(osp.join(OUT_DIR, DS + ".json"), "w") as f:
            json.dump(res, f)
        print(f"[ubgold_xcheck] {DS}: mismatches={res['n_graph_mismatches']} "
              f"num_feat={res['num_feat']} time={res['seconds']}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
