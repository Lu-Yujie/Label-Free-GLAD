"""Synthetic graph dicts (gens.py format, stored by export_data.py) -> PyG Data objects in the form
UB-GOLD's get_ad_dataset_TU hands to SIGNET / OCGTL / OCGIN / GLocalKD.

What a UB-GOLD TU graph looks like (dataloader/data_loader.py + torch_geometric TUDataset):
  x          float32 (n, F): one-hot node labels (TUDataset default use_node_attr=False, i.e. continuous
             node attributes are NOT used by UB-GOLD), or Constant(1, cat=False) = ones (n, 1) for the
             unlabelled datasets (IMDB-BINARY, REDDIT-BINARY, COLLAB);
  edge_index int64 (2, 2m): both directions, coalesced (sorted), no self loops;
  edge_attr  None (get_ad_dataset_TU sets data.edge_attr = None);
  y          int64 (1,): training graphs y = 0 and get an 'idx' attribute; test graphs y = 1 for an
             anomaly, 0 for a normal graph.
Here test graphs additionally carry 'gtype' (int64 (1,), the generator group: 0 normal, k = anomaly
type k) and 'tidx' (position in the test list), used only to slice metrics per type after the
shuffled test DataLoader; no model reads them.

Feature variants
  F1     one-hot(node_label, 4)                      UB-GOLD style (labels only)         GLAD-X-lite
  F2     [one-hot(node_label, 4), x (2-d, raw)]      every node input GRAIL-0 had        GLAD-X-lite
  C1     ones (n, 1) = UB-GOLD's Constant(1)         featureless sets                    Syn-Com, BM
GRAIL-0's channel C on GLAD-X-lite used node_label AND the continuous x (attrib.py calls api.fit with
n_labels=4; api.fit sets use_x=True whenever every graph has x), so F2 is the primary variant.
"""
import numpy as np
import torch

N_LABELS_GX = 4


def node_features(g, variant):
    n = g["n"]
    if variant == "C1":
        return np.ones((n, 1), np.float32)
    lab = np.asarray(g["node_label"], np.int64)
    assert lab.min() >= 0 and lab.max() < N_LABELS_GX
    oh = np.zeros((n, N_LABELS_GX), np.float32)
    oh[np.arange(n), lab] = 1.0
    if variant == "F1":
        return oh
    if variant == "F2":
        return np.concatenate([oh, np.asarray(g["x"], np.float32)], 1)
    raise ValueError(variant)


def to_data(g, variant, y, **extra):
    from torch_geometric.data import Data
    from torch_geometric.utils import coalesce
    e = np.asarray(g["edges"], np.int64).reshape(-1, 2)
    ei = torch.from_numpy(np.concatenate([e, e[:, ::-1]], 0).T.copy())
    ei = coalesce(ei, num_nodes=g["n"])
    d = Data(x=torch.from_numpy(node_features(g, variant)), edge_index=ei,
             y=torch.tensor([int(y)], dtype=torch.long))
    d.num_nodes = g["n"]
    for k, v in extra.items():
        d[k] = torch.tensor([int(v)], dtype=torch.long)
    return d


def variants_for(ds):
    return ["F1", "F2"] if ds == "GX" else ["C1"]


def build(train, test, group, variant):
    """-> data_train (y=0, idx), data_test (y in {0,1}, gtype, tidx). Mirrors get_ad_dataset_TU."""
    data_train = [to_data(g, variant, 0, idx=i) for i, g in enumerate(train)]
    data_test = [to_data(g, variant, int(group[i] > 0), gtype=int(group[i]), tidx=i) for i, g in enumerate(test)]
    meta = {"num_feat": int(data_train[0].x.shape[1]), "num_train": len(data_train),
            "max_nodes_num": int(max(g["n"] for g in list(train) + list(test))), "num_edge_feat": 0}
    return data_train, data_test, meta


def selftest(train, test, group, variant):
    """Checks used by the smoke test: edge sets, symmetry, features, labels."""
    dtr, dte, meta = build(train, test, group, variant)
    for g, d in zip(list(train) + list(test), dtr + dte):
        ei = d.edge_index.numpy()
        assert ei.shape[1] == 2 * len(g["edges"])
        assert not np.any(ei[0] == ei[1])
        s = set(map(tuple, ei.T.tolist()))
        assert all((b, a) in s for a, b in s)
        und = {(min(a, b), max(a, b)) for a, b in s}
        assert und == set(map(tuple, np.asarray(g["edges"]).tolist()))
        assert d.x.shape[0] == g["n"] and d.x.dtype == torch.float32
    if variant in ("F1", "F2"):
        for g, d in zip(train[:20], dtr[:20]):
            assert np.array_equal(d.x[:, :N_LABELS_GX].argmax(1).numpy(), g["node_label"])
            if variant == "F2":
                assert np.allclose(d.x[:, N_LABELS_GX:].numpy(), g["x"])
    return meta
