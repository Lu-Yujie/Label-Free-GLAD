"""Shared data interface for the Grail pilot (UB-GOLD-faithful GLAD folds).

    from common.data import list_datasets, load_folds
    for name in list_datasets():
        for fold in load_folds(name):
            fold['train']   # list of NORMAL training graphs
            fold['test']    # list of test graphs (normal + anomalous)
            fold['y_test']  # np.ndarray int64, 1 = anomaly, 0 = normal
            fold['fold']    # int (0..4); UB-GOLD trial i uses fold i
            fold['meta']    # dict (dataset, seed = 3407+fold, protocol, class semantics, ...)

Each graph is a dict:
    'n'          int
    'edges'      np.ndarray int64 (m,2), undirected, each edge once with i<j, no self-loops
    'x'          np.ndarray float32 (n,d) of continuous node attributes, or None
    'node_label' np.ndarray int64 (n,) in 0..meta['node_label_dim']-1, or None
    'gid'        int, index of the graph in the dataset cache (identical object id across
                 folds; train/test gids never overlap within a fold)

Feature modes (``features=``):
    'ubgold' (default) -- exactly what UB-GOLD's models see. node_label = TU node labels on
        the labelled TU datasets (UB-GOLD one-hot encodes them, dim = meta['node_label_dim']);
        node_label = None for IMDB-BINARY / REDDIT-BINARY / COLLAB (UB-GOLD: Constant(1)) and
        for Tox21 (UB-GOLD sets x = ones); x = None everywhere (UB-GOLD never uses
        continuous node attributes: TUDataset(use_node_attr=False)).
    'all' -- additionally exposes continuous node attributes as 'x' where the raw TU data
        has them (AIDS 4-d, BZR/COX2/DHFR 3-d, ENZYMES 18-d, PROTEINS_full 29-d) and the
        atom-type node labels for Tox21. NOT comparable to published UB-GOLD numbers.

Arrays are shared between folds and marked read-only; copy before modifying in place.
Graph dicts are fresh per fold, so annotating a dict in one fold does not leak into another.
"""
import os
import os.path as osp
import pickle

import numpy as np

DATA_ROOT = (os.environ.get("GLAD_DATA_ROOT", "glad_data"))
FOLD_DIR = osp.join(DATA_ROOT, "folds")

# canonical order: group 1, group 2, group 3
DATASET_ORDER = [
    "AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "MUTAG", "PROTEINS_full",
    "IMDB-BINARY", "DD", "COLLAB", "REDDIT-BINARY",
    "Tox21_p53", "Tox21_MMP",
]
ALIASES = {
    "PROTEINS": "PROTEINS_full", "PROTEINS-full": "PROTEINS_full", "IMDB-B": "IMDB-BINARY",
    "IMDB": "IMDB-BINARY", "REDDIT-B": "REDDIT-BINARY", "REDDIT": "REDDIT-BINARY",
    "p53": "Tox21_p53", "MMP": "Tox21_MMP",
}
TU12 = ["MUTAG", "PROTEINS_full", "DD", "ENZYMES", "DHFR", "BZR", "COX2", "AIDS",
        "IMDB-BINARY", "NCI1", "COLLAB", "REDDIT-BINARY"]   # the LGKDE/GLASS 12-TU set


def _path(name):
    return osp.join(FOLD_DIR, ALIASES.get(name, name) + ".pkl")


def list_datasets(include_tox21=True):
    """Datasets whose fold cache exists, in canonical order."""
    out = [d for d in DATASET_ORDER if osp.exists(_path(d))]
    if not include_tox21:
        out = [d for d in out if not d.startswith("Tox21")]
    return out


def load_raw_cache(name):
    """The pickled cache (graphs, folds with index arrays, meta). For auditing only."""
    with open(_path(name), "rb") as f:
        return pickle.load(f)


def _ro(a):
    if a is not None:
        a.flags.writeable = False
    return a


def load_folds(name, features="ubgold", folds=None):
    """Return the list of UB-GOLD folds for ``name`` (see module docstring)."""
    if features not in ("ubgold", "all"):
        raise ValueError("features must be 'ubgold' or 'all'")
    obj = load_raw_cache(name)
    meta0 = obj["meta"]
    use_nl = features == "all" or meta0["ubgold_uses_node_labels"]
    use_x = features == "all"
    for g in obj["graphs"]:
        _ro(g["edges"]); _ro(g["node_label"]); _ro(g["x_attr"])

    def view(g):
        return {"n": g["n"], "edges": g["edges"],
                "x": g["x_attr"] if use_x else None,
                "node_label": g["node_label"] if use_nl else None,
                "gid": g["gid"]}

    meta_public = {k: v for k, v in meta0.items() if k not in ("reader_info", "file_ids")}
    meta_public["features"] = features
    out = []
    for f in obj["folds"]:
        if folds is not None and f["fold"] not in folds:
            continue
        tr, te = f["train_idx"], f["test_idx"]
        y = np.asarray(f["y_test"], dtype=np.int64).copy()
        meta = dict(meta_public)
        meta.update({"fold": int(f["fold"]), "seed": int(f["seed"]), "n_train": int(len(tr)),
                     "n_test": int(len(te)), "n_test_anomalies": int(y.sum()),
                     "train_idx": np.asarray(tr).copy(), "test_idx": np.asarray(te).copy()})
        out.append({"train": [view(obj["graphs"][i]) for i in tr],
                    "test": [view(obj["graphs"][i]) for i in te],
                    "y_test": y, "fold": int(f["fold"]), "meta": meta})
    return out
