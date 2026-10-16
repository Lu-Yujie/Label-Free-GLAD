"""Native (numpy/pandas) readers for TU-format graph datasets.

Two readers, each mimicking the exact code path UB-GOLD uses:

* ``read_tu_pyg_like(folder, name)`` mimics PyG 2.6.1 ``TUDataset`` /
  ``read_tu_data`` (used by UB-GOLD ``get_ad_dataset_TU``):
    - nodes per graph = counts in ``graph_indicator`` (isolated nodes kept),
    - graph labels remapped with ``unique(sorted=True, return_inverse=True)``,
    - node labels: column minus its global min (PyG then one-hots them),
    - self-loops removed, duplicate edges removed (coalesce).
  We additionally symmetrise edges into an undirected list with i<j.

* ``read_tox21_ubgold_like(folder, name)`` mimics UB-GOLD ``read_graph_file``
  (used by ``get_ad_dataset_Tox21``):
    - graph labels remapped by ORDER OF FIRST APPEARANCE in the file,
    - each graph is ``nx.from_edgelist`` of its A.txt lines, so isolated
      nodes are DROPPED and node order = first appearance in the edge list,
    - graphs with no edges (0 nodes after that) are DROPPED,
    - UB-GOLD sets x = ones (node labels are read but never used).

All returned graphs are dicts: n, edges (int64 (m,2), i<j, unique, no
self-loops), node_label (int64 (n,) or None), x_attr (float32 (n,d) or None),
gid (int, index of the graph in the raw file, 0-based).
"""
import os
import numpy as np
import pandas as pd


def _read_int_matrix(path):
    df = pd.read_csv(path, header=None, sep=",", engine="c", dtype=np.int64,
                     skipinitialspace=True)
    return df.to_numpy()


def _read_float_matrix(path):
    df = pd.read_csv(path, header=None, sep=",", engine="c", dtype=np.float64,
                     skipinitialspace=True)
    return df.to_numpy()


def _undirected_unique(src, dst):
    """Return (edges (m,2) int64 with i<j, unique, no self loops, sorted),
    number of self-loop entries removed, number of directed entries whose
    reverse is missing (asymmetry count)."""
    keep = src != dst
    n_self = int((~keep).sum())
    src, dst = src[keep], dst[keep]
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    if lo.size == 0:
        return np.zeros((0, 2), dtype=np.int64), n_self, 0
    # asymmetry: count directed pairs (u,v) whose (v,u) is absent
    big = int(max(lo.max(), hi.max())) + 1
    key_dir = np.unique(src.astype(np.int64) * big + dst)
    key_rev = dst.astype(np.int64) * big + src
    key_rev = np.unique(key_rev)
    n_asym = int(np.setdiff1d(key_dir, key_rev, assume_unique=True).size)
    key = np.unique(lo.astype(np.int64) * big + hi)
    edges = np.stack([key // big, key % big], axis=1).astype(np.int64)
    return edges, n_self, n_asym


def read_tu_pyg_like(folder, name):
    """Read a standard TU dataset folder (``<folder>/<name>_A.txt`` ...)."""
    pre = os.path.join(folder, name)
    A = _read_int_matrix(pre + "_A.txt") - 1                    # (E,2) 0-based global ids
    batch = _read_int_matrix(pre + "_graph_indicator.txt")[:, 0] - 1
    N = int(batch.max()) + 1
    assert np.all(np.diff(batch) >= 0), "graph_indicator must be sorted"
    counts = np.bincount(batch, minlength=N)
    start = np.concatenate([[0], np.cumsum(counts)])

    y_raw = _read_int_matrix(pre + "_graph_labels.txt")[:, 0]
    assert y_raw.shape[0] == N
    uniq, y = np.unique(y_raw, return_inverse=True)            # == torch.unique(sorted=True, return_inverse=True)

    node_label = None
    nl_min = None
    if os.path.exists(pre + "_node_labels.txt"):
        nl = _read_int_matrix(pre + "_node_labels.txt")
        assert nl.shape[1] == 1, "multi-column node labels not supported"
        nl = nl[:, 0]
        nl_min = int(nl.min())
        node_label = (nl - nl_min).astype(np.int64)
    x_attr = None
    if os.path.exists(pre + "_node_attributes.txt"):
        x_attr = _read_float_matrix(pre + "_node_attributes.txt").astype(np.float32)
        assert x_attr.shape[0] == batch.shape[0]

    # assign edges to graphs
    gsrc = batch[A[:, 0]]
    gdst = batch[A[:, 1]]
    assert np.all(gsrc == gdst), "edge crosses graphs"
    order = np.argsort(gsrc, kind="stable")
    A = A[order]
    gsrc = gsrc[order]
    ecounts = np.bincount(gsrc, minlength=N)
    estart = np.concatenate([[0], np.cumsum(ecounts)])

    graphs = []
    tot_self = tot_asym = 0
    for g in range(N):
        s, e = estart[g], estart[g + 1]
        loc = A[s:e] - start[g]
        edges, n_self, n_asym = _undirected_unique(loc[:, 0], loc[:, 1])
        tot_self += n_self
        tot_asym += n_asym
        n = int(counts[g])
        if edges.size:
            assert edges.max() < n
        gd = {
            "n": n,
            "edges": edges,
            "node_label": None if node_label is None else node_label[start[g]:start[g + 1]].copy(),
            "x_attr": None if x_attr is None else x_attr[start[g]:start[g + 1]].copy(),
            "gid": g,
        }
        graphs.append(gd)

    info = {
        "n_graphs": N,
        "graph_label_raw_values": uniq.tolist(),            # remapped label k <-> raw value uniq[k]
        "class_counts": np.bincount(y).tolist(),
        "node_label_min_raw": nl_min,
        "node_label_dim": None if node_label is None else int(node_label.max()) + 1,
        "x_attr_dim": None if x_attr is None else int(x_attr.shape[1]),
        "self_loops_removed": int(tot_self),
        "asymmetric_directed_entries": int(tot_asym),
    }
    return graphs, y.astype(np.int64), info


def read_tox21_ubgold_like(folder, name):
    """Mimic UB-GOLD ``read_graph_file(name, path)`` for a TU Tox21 split folder."""
    pre = os.path.join(folder, name)
    gi = _read_int_matrix(pre + "_graph_indicator.txt")[:, 0]      # node i (1-based) -> graph id (1-based)
    y_raw = _read_int_matrix(pre + "_graph_labels.txt")[:, 0]
    label_vals = []
    for v in y_raw.tolist():                                          # first-appearance order
        if v not in label_vals:
            label_vals.append(v)
    lmap = {v: i for i, v in enumerate(label_vals)}
    y_mapped = np.array([lmap[v] for v in y_raw.tolist()], dtype=np.int64)
    nl = None
    if os.path.exists(pre + "_node_labels.txt"):
        nl = _read_int_matrix(pre + "_node_labels.txt")[:, 0]
    A = _read_int_matrix(pre + "_A.txt")                              # 1-based global ids
    N = y_raw.shape[0]
    g_of_edge = gi[A[:, 0] - 1]
    order = np.argsort(g_of_edge, kind="stable")                      # keeps file order within graph
    A = A[order]
    g_of_edge = g_of_edge[order]
    ecounts = np.bincount(g_of_edge, minlength=N + 1)
    estart = np.concatenate([[0], np.cumsum(ecounts)])

    graphs, ys, kept_gid = [], [], []
    n_dropped_empty = 0
    n_isolated_dropped = 0
    n_self = 0
    node_counts_file = np.bincount(gi, minlength=N + 1)
    for g in range(1, N + 1):
        E = A[estart[g]:estart[g + 1]]
        if E.shape[0] == 0:
            n_dropped_empty += 1
            continue
        # node order = first appearance scanning (e0, e1) pairs in file order (nx.from_edgelist)
        flat = E.reshape(-1)
        _, first_idx = np.unique(flat, return_index=True)
        nodes_in_order = flat[np.sort(first_idx)]
        mapping = {int(u): i for i, u in enumerate(nodes_in_order.tolist())}
        loc = np.vectorize(mapping.__getitem__, otypes=[np.int64])(E)
        n = len(mapping)
        n_isolated_dropped += int(node_counts_file[g]) - n
        n_self += int((loc[:, 0] == loc[:, 1]).sum())
        edges, _, _ = _undirected_unique(loc[:, 0], loc[:, 1])
        node_label = None
        if nl is not None:
            node_label = nl[nodes_in_order - 1].astype(np.int64)
        graphs.append({"n": n, "edges": edges, "node_label": node_label,
                       "x_attr": None, "gid": g - 1})
        ys.append(y_mapped[g - 1])
    info = {
        "n_graphs_file": int(N),
        "n_graphs_kept": len(graphs),
        "n_dropped_no_edges": int(n_dropped_empty),
        "n_isolated_nodes_dropped": int(n_isolated_dropped),
        "self_loop_entries_in_file": int(n_self),
        "first_appearance_label_order": [int(v) for v in label_vals],  # mapped k <-> raw label_vals[k]
        "raw_label_counts_file": {int(v): int((y_raw == v).sum()) for v in label_vals},
    }
    return graphs, np.array(ys, dtype=np.int64), y_raw, info
