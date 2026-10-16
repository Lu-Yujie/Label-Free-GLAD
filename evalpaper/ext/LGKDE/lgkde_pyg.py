"""PyG/pure-torch port of LGKDE (github.com/MathAdventurer/LGKDE @ 1bba5ef, MIT; files model.py and ng.py).

The model math is unchanged. Every deviation from the DGL original is marked "# CHANGE" and listed in
NOTES.md ("Port"). losses.py / evaluation.py are imported verbatim from ./src (they do not use DGL).

Graph container
---------------
A batch of graphs is a `GraphBatch`: node features `x` (N, F) float32, `edge_index` (2, E) int64 with
row 0 = source and row 1 = destination (PyG convention; DGL's (src, dst)), and `ptr` (G+1,) node offsets
(nodes of graph i are ptr[i]:ptr[i+1], as in a PyG Batch and in a DGL batched graph). Self-loops are
part of `edge_index` exactly as dgl.add_self_loop would add them (appended after the existing edges,
one per node, even if a self-loop already exists).

Two implementations of the deep-MMD distance matrix are provided:
  * DGMMD._compute_distance_matrix_node_level_loop  -- line-by-line copy of the original per-pair loop
    (reference; used by the equivalence tests).
  * DGMMD._compute_distance_matrix_node_level       -- vectorized, blocked version (used for runs).
    It computes the SAME quantity: for every pair of graphs (i, j) and every bandwidth bw,
        MMD^2_ij(bw) = mean_{a,a' in i} k(a,a') + mean_{b,b' in j} k(b,b') - 2 mean_{a in i, b in j} k(a,b),
        k(a,b) = exp(-clamp(|a|^2 + |b|^2 - 2 a.b, min=0) / bw),
    then clamp(MMD^2, 1e-8), sqrt, and the running torch.max over bandwidths starting from 0, with the
    same diagonal (0) and symmetry (one computation mirrored to [j,i]) semantics as the original.
"""
import importlib.util
import math
import os.path as osp
import warnings
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from scipy.sparse import csr_matrix
from scipy.linalg import svd

HERE = osp.dirname(osp.abspath(__file__))
SRC = osp.join(HERE, "src")


def _load_src(name):
    """Import a DGL-free file of the original repo verbatim (losses.py, evaluation.py)."""
    spec = importlib.util.spec_from_file_location("lgkde_src_" + name, osp.join(SRC, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================================
# graph container (CHANGE: replaces dgl.DGLGraph / dgl.batch / dgl.unbatch)
# ============================================================================================
class GraphBatch:
    __slots__ = ("x", "edge_index", "ptr", "gidx")

    def __init__(self, x, edge_index, ptr, gidx=None):
        self.x, self.edge_index, self.ptr, self.gidx = x, edge_index, ptr, gidx

    @property
    def num_graphs(self):
        return int(self.ptr.numel() - 1)

    def batch_num_nodes(self):
        return self.ptr[1:] - self.ptr[:-1]

    def to(self, device):
        return GraphBatch(self.x.to(device), self.edge_index.to(device), self.ptr.to(device),
                          None if self.gidx is None else self.gidx.to(device))


def add_self_loop(edge_index, n):
    """dgl.add_self_loop: append one (i, i) edge per node after the existing edges (no dedup)."""
    loops = torch.arange(n, dtype=torch.long).repeat(2, 1)
    return torch.cat([edge_index.long(), loops], 1)


def remove_self_loop(edge_index):
    """dgl.remove_self_loop: drop every (i, i) edge, keep the order of the others."""
    return edge_index[:, edge_index[0] != edge_index[1]]


def batch_graphs(xs, eis, gidx=None):
    """dgl.batch for a list of graphs given as (x_i, edge_index_i): node/edge concatenation."""
    sizes = [int(x.size(0)) for x in xs]
    ptr = torch.zeros(len(sizes) + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(torch.tensor(sizes, dtype=torch.long), 0)
    ei = torch.cat([e + ptr[i] for i, e in enumerate(eis)], 1) if eis else torch.zeros(2, 0, dtype=torch.long)
    return GraphBatch(torch.cat(xs, 0), ei, ptr,
                      None if gidx is None else torch.as_tensor(gidx, dtype=torch.long))


def collate(samples):
    """GraphDataLoader collate for (graph, label) samples; a graph is a dict(x, edge_index, gidx)."""
    graphs, labels = map(list, zip(*samples))
    bg = batch_graphs([g["x"] for g in graphs], [g["edge_index"] for g in graphs],
                      [g["gidx"] for g in graphs])
    return bg, torch.utils.data.default_collate(labels)


class CustomDataset(torch.utils.data.Dataset):          # verbatim (dataprocessing.py)
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


# ============================================================================================
# GraphConv: CHANGE re-implementation of dgl.nn.GraphConv(in, out, norm='both', bias=True)
# (weight=True, activation=None, allow_zero_in_degree=False), DGL 2.4 semantics.
# ============================================================================================
class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm='both', bias=True):
        super().__init__()
        assert norm == 'both'
        self._in_feats, self._out_feats, self._norm = in_feats, out_feats, norm
        # same parameter shapes/creation order and init as DGL (weight (in, out) xavier_uniform_, bias zeros)
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, g, feat):
        src, dst = g.edge_index[0], g.edge_index[1]
        n = feat.size(0)
        in_deg = torch.bincount(dst, minlength=n)
        if bool((in_deg == 0).any()):     # DGL raises DGLError for 0-in-degree nodes
            raise RuntimeError("There are 0-in-degree nodes in the graph (DGL GraphConv would raise)")
        out_deg = torch.bincount(src, minlength=n).to(feat).clamp(min=1)
        feat_src = feat * torch.pow(out_deg, -0.5).unsqueeze(-1)
        if self._in_feats > self._out_feats:      # DGL: multiply W first, then aggregate
            feat_src = torch.matmul(feat_src, self.weight)
            rst = torch.zeros(n, feat_src.size(1), dtype=feat.dtype, device=feat.device).index_add_(0, dst, feat_src[src])
        else:                                     # DGL: aggregate first, then multiply W
            rst = torch.zeros(n, feat_src.size(1), dtype=feat.dtype, device=feat.device).index_add_(0, dst, feat_src[src])
            rst = torch.matmul(rst, self.weight)
        in_norm = torch.pow(in_deg.to(feat).clamp(min=1), -0.5).unsqueeze(-1)
        rst = rst * in_norm
        if self.bias is not None:
            rst = rst + self.bias
        return rst


# ============================================================================================
# model.py (KDE class unused by the Trainer and omitted; DGMMD, LGKDE)
# ============================================================================================
def _chunks_by_size(sizes, max_padded, ratio=1.2):
    """CHANGE (vectorization helper): group graphs (sorted by size) into chunks whose padded node count
    (count x max size) stays <= max_padded and whose sizes differ by at most `ratio` (bounded padding)."""
    order = np.argsort(sizes, kind="stable")
    chunks, cur, cur_min, cur_max = [], [], None, None
    for i in order:
        s = int(sizes[i])
        if cur:
            nmax = max(cur_max, s)
            if (len(cur) + 1) * nmax > max_padded or nmax > ratio * cur_min:
                chunks.append(cur)
                cur = []
        if not cur:
            cur, cur_min, cur_max = [int(i)], s, s
        else:
            cur.append(int(i))
            cur_max = max(cur_max, s)
    if cur:
        chunks.append(cur)
    return chunks


def _pad(h, ptr, idx, device):
    """Padded (g, m, d) embeddings of graphs idx and a (g, m) 0/1 mask (float)."""
    sizes = (ptr[1:] - ptr[:-1])[idx]
    m = int(sizes.max())
    g = len(idx)
    starts = ptr[:-1][idx]
    ar = torch.arange(m, device=device)
    valid = ar[None, :] < sizes[:, None]                                   # (g, m)
    gather = (starts[:, None] + ar[None, :]).clamp(max=h.size(0) - 1)       # (g, m)
    hp = h[gather.reshape(-1)].reshape(g, m, h.size(1)) * valid[..., None].to(h.dtype)
    return hp, valid.to(h.dtype)


def _block_dist(hA, mA, hB, mB):
    """Padded node-pair squared distances, same arithmetic as DGMMD._pairwise_distances(_cross):
    clamp((|a|^2 + |b|^2) - 2.0 * (a.b), min=0); padded pairs -> +inf (kernel value exactly 0).
    Returns the (gA, ma, gB, mb) distance block and the mask of pairs that the clamp zeroed."""
    gA, ma, d = hA.shape
    gB, mb, _ = hB.shape
    nA = (hA ** 2).sum(dim=2).reshape(-1, 1)
    nB = (hB ** 2).sum(dim=2).reshape(1, -1)
    ab = torch.mm(hA.reshape(gA * ma, d), hB.reshape(gB * mb, d).t())
    dist = nA + nB
    dist.sub_(ab, alpha=2.0)                     # (|a|^2 + |b|^2) - 2 a.b  (2 a.b is exact)
    del ab
    neg = dist < 0
    dist.clamp_(min=0.0)
    valid = (mA.reshape(-1, 1) > 0) & (mB.reshape(1, -1) > 0)
    dist.masked_fill_(~valid, float("inf"))
    return dist.view(gA, ma, gB, mb), neg.view(gA, ma, gB, mb)


EXP_FLOOR = -87.33   # ln(FLT_MIN); see _kexp


def _kexp(dist, bw):
    """exp(-dist / bw) (division by bw as in the original). CHANGE (speed only): the exponent is floored at
    ln(FLT_MIN) = -87.33 because the AVX2 exp of torch/MKL takes a 15-30x slower path for arguments that
    underflow; kernel values below 1.18e-38 (incl. padded pairs, dist=inf) become ~1.18e-38 instead of a
    denormal/0. Effect on any kernel mean (>= 1/n^2 >= 7e-8) or on MMD^2 (clamped at 1e-8): < 1e-30."""
    return torch.div(dist, -bw).clamp_(min=EXP_FLOOR).exp_()


class _BlockKernelSums(torch.autograd.Function):
    """sum_{a in i, b in j} exp(-dist_ab / bw) for padded chunks -> (gA, gB, K).
    Forward: exp(-dist/bw) with dist as above (division by bw as in the original). Backward: the
    gradient autograd would give for the same expression (exp, division, clamp mask, |a|^2+|b|^2-2ab),
    with the block recomputed instead of stored (bounded memory)."""

    @staticmethod
    def forward(ctx, hA, mA, hB, mB, bandwidths):
        ctx.bandwidths = tuple(bandwidths)
        ctx.save_for_backward(hA, mA, hB, mB)
        with torch.no_grad():
            dist, _ = _block_dist(hA, mA, hB, mB)
            out = [_kexp(dist, bw).sum(dim=(1, 3)) for bw in ctx.bandwidths]
        return torch.stack(out, -1)

    @staticmethod
    def backward(ctx, grad_out):
        hA, mA, hB, mB = ctx.saved_tensors
        gA, ma, d = hA.shape
        gB, mb, _ = hB.shape
        with torch.no_grad():
            dist, neg = _block_dist(hA, mA, hB, mB)
            W = torch.zeros_like(dist)
            for k, bw in enumerate(ctx.bandwidths):
                E = _kexp(dist, bw)
                W.addcmul_(E, (grad_out[:, :, k] / (-bw))[:, None, :, None])
                del E
            W.masked_fill_(neg, 0.0)              # clamp(min=0) passes no gradient where dist < 0
            rs = W.sum(dim=(2, 3))                # d/d|a|^2
            cs = W.sum(dim=(0, 1))                # d/d|b|^2
            W2 = W.view(gA * ma, gB * mb)
            g_hA = 2.0 * hA * rs[..., None] - 2.0 * torch.mm(W2, hB.reshape(gB * mb, d)).view(gA, ma, d)
            g_hB = 2.0 * hB * cs[..., None] - 2.0 * torch.mm(W2.t(), hA.reshape(gA * ma, d)).view(gB, mb, d)
        return g_hA, None, g_hB, None, None


def _block_kernel_sums(hA, mA, hB, mB, bandwidths):
    return _BlockKernelSums.apply(hA, mA, hB, mB, tuple(bandwidths))


def _block_kernel_sums_autograd(hA, mA, hB, mB, bandwidths):
    """Plain-autograd version of the same computation (reference for tests)."""
    gA, ma, d = hA.shape
    gB, mb, _ = hB.shape
    nA = (hA ** 2).sum(dim=2)
    nB = (hB ** 2).sum(dim=2)
    ab = torch.mm(hA.reshape(gA * ma, d), hB.reshape(gB * mb, d).t()).reshape(gA, ma, gB, mb)
    dist = (nA[:, :, None, None] + nB[None, None, :, :]) - 2.0 * ab
    dist = torch.clamp(dist, min=0.0)
    out = []
    for bw in bandwidths:
        k = torch.exp(-dist / bw)
        k = k * mA[:, :, None, None]
        k = k * mB[None, None, :, :]
        out.append(k.sum(dim=(1, 3)))
    return torch.stack(out, -1)


def _self_kernel_sums(hA, mA, bandwidths):
    """Within-graph sums sum_{a,a' in i} exp(-dist/bw) for a padded chunk -> (g, K)."""
    nA = (hA ** 2).sum(dim=2)
    ab = torch.bmm(hA, hA.transpose(1, 2))
    dist = torch.clamp(nA[:, :, None] + nA[:, None, :] - 2.0 * ab, min=0.0)
    mm = mA[:, :, None] * mA[:, None, :]
    return torch.stack([(torch.exp(-dist / bw) * mm).sum(dim=(1, 2)) for bw in bandwidths], -1)


class DGMMD(nn.Module):
    """
    A GNN-based model that computes pairwise graph distances using
    an MMD-based kernel approach.
    """
    # CHANGE: vectorization knobs (not hyperparameters; they do not change the computed values beyond
    # float32 summation order). block_elems bounds the padded node-pair block; use_checkpoint recomputes
    # each block in backward instead of storing it (exact gradients, bounded memory).
    block_elems = 2 ** 25
    use_checkpoint = True
    vectorized = True
    block_impl = "function"      # "function" (custom backward) | "autograd" (checkpointed autograd)

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        bandwidths: List[float] = [1.0],
        dropout: float = 0.1,
        batch_norm: bool = True
    ):
        super().__init__()

        self.bandwidths = bandwidths

        # GNN Layers
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GraphConv(in_dim, out_dim, norm='both', bias=True))   # CHANGE: dglnn.GraphConv
            if batch_norm:
                self.bns = nn.ModuleList([nn.BatchNorm1d(out_dim)])
            else:
                self.bns = None
        else:
            self.layers.append(GraphConv(in_dim, hidden_dim, norm='both', bias=True))
            for _ in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, norm='both', bias=True))
            self.layers.append(GraphConv(hidden_dim, out_dim, norm='both', bias=True))

            if batch_norm:
                self.bns = nn.ModuleList(
                    [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)]
                    + [nn.BatchNorm1d(out_dim)]
                )
            else:
                self.bns = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if self.bns is not None:
                h = self.bns[i](h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def compute_distance_matrix(self, graphs, graphs_b=None):
        """
        Compute graph distances with node-level distribution MMD.
        """
        if self.vectorized:     # CHANGE: vectorized/blocked evaluation of the same matrix
            return self._compute_distance_matrix_node_level(graphs, graphs_b)
        return self._compute_distance_matrix_node_level_loop(graphs, graphs_b)

    # ---------------------------------------------------------------------------------------
    # original per-pair loop (verbatim except the embedding helper); reference implementation
    # ---------------------------------------------------------------------------------------
    def _compute_distance_matrix_node_level_loop(self, graphs, graphs_b=None):
        node_embeddings_A, graph_ids_A, num_graphs_A = self._compute_embeddings_node_level(graphs)
        dist_sq_A = self._pairwise_distances(node_embeddings_A)

        # Precompute indices for each graph in A
        graph_indices_A = [(graph_ids_A == i).nonzero(as_tuple=True)[0]
                           for i in range(num_graphs_A)]

        if graphs_b is None:
            # Intra-set (N x N)
            distance_matrix = torch.zeros(num_graphs_A, num_graphs_A, device=node_embeddings_A.device)
            for i in range(num_graphs_A):
                idx_i = graph_indices_A[i]
                dist_ii = dist_sq_A[idx_i][:, idx_i]
                for j in range(i, num_graphs_A):
                    if j == i:
                        distance_matrix[i, j] = 0.0
                        continue
                    idx_j = graph_indices_A[j]
                    dist_jj = dist_sq_A[idx_j][:, idx_j]
                    dist_ij = dist_sq_A[idx_i][:, idx_j]
                    mmd_ij = self._compute_mmd_sup(dist_ii, dist_jj, dist_ij)
                    distance_matrix[i, j] = distance_matrix[j, i] = mmd_ij
            return distance_matrix
        else:
            # Cross-set (N x M)
            node_embeddings_B, graph_ids_B, num_graphs_B = self._compute_embeddings_node_level(graphs_b)
            dist_sq_B = self._pairwise_distances(node_embeddings_B)
            dist_sq_cross = self._pairwise_distances_cross(node_embeddings_A, node_embeddings_B)

            graph_indices_B = [(graph_ids_B == j).nonzero(as_tuple=True)[0]
                               for j in range(num_graphs_B)]

            distance_matrix = torch.zeros(num_graphs_A, num_graphs_B, device=node_embeddings_A.device)
            for i in range(num_graphs_A):
                idx_i = graph_indices_A[i]
                dist_ii = dist_sq_A[idx_i][:, idx_i]
                for j in range(num_graphs_B):
                    idx_j = graph_indices_B[j]
                    dist_jj = dist_sq_B[idx_j][:, idx_j]
                    dist_ij = dist_sq_cross[idx_i][:, idx_j]
                    mmd_ij = self._compute_mmd_sup(dist_ii, dist_jj, dist_ij)
                    distance_matrix[i, j] = mmd_ij
            return distance_matrix

    # ---------------------------------------------------------------------------------------
    # CHANGE: vectorized evaluation (same embeddings calls in the same order: A first, then B)
    # ---------------------------------------------------------------------------------------
    def _compute_distance_matrix_node_level(self, graphs, graphs_b=None):
        hA, ptrA, GA = self._embed_ptr(graphs)
        if graphs_b is None:
            K, kA = self._kernel_means(hA, ptrA, None, None)          # K: (GA, GA, nbw) upper-tri valid
            return self._mmd_sup_from_means(kA, kA, K, intra=True)
        hB, ptrB, GB = self._embed_ptr(graphs_b)
        K, kA, kB = self._kernel_means(hA, ptrA, hB, ptrB)
        return self._mmd_sup_from_means(kA, kB, K, intra=False)

    def _embed_ptr(self, graphs):
        h = self.forward(graphs, graphs.x)
        return h, graphs.ptr.to(h.device), graphs.num_graphs

    def _mmd_sup_from_means(self, kA, kB, K, intra):
        """Same arithmetic as _compute_mmd_sup, vectorized over pairs: mmd_sq = K_ii + K_jj - 2 K_ij,
        clamp 1e-8, sqrt, running torch.max over bandwidths from a zero start."""
        GA, GB = K.shape[0], K.shape[1]
        mmd_sup = torch.zeros(1, device=K.device)
        for k in range(len(self.bandwidths)):
            mmd_sq = kA[:, k][:, None] + kB[:, k][None, :] - 2 * K[:, :, k]
            mmd_sq = torch.clamp(mmd_sq, min=1e-8)
            mmd = torch.sqrt(mmd_sq)
            mmd_sup = torch.max(mmd_sup, mmd)
        if intra:   # diagonal exactly 0; [j, i] is the same computation as [i, j] (i < j)
            up = torch.triu(mmd_sup, diagonal=1)
            return up + up.t()
        return mmd_sup

    def _kernel_means(self, hA, ptrA, hB, ptrB):
        dev = hA.device
        bws = list(self.bandwidths)
        nbw = len(bws)
        szA = (ptrA[1:] - ptrA[:-1]).cpu().numpy()
        GA = len(szA)
        cap = int(max(1, math.sqrt(self.block_elems)))
        chA = _chunks_by_size(szA, cap)
        padA = [_pad(hA, ptrA, torch.as_tensor(c, device=dev), dev) for c in chA]
        cntA = torch.as_tensor(szA, dtype=hA.dtype, device=dev)

        def ck_(fn, *args):
            if self.use_checkpoint and torch.is_grad_enabled() and any(a.requires_grad for a in args if torch.is_tensor(a)):
                from torch.utils.checkpoint import checkpoint
                return checkpoint(fn, *args, use_reentrant=False)
            return fn(*args)

        if self.block_impl == "function":
            blk = lambda a, ma, b, mb: _block_kernel_sums(a, ma, b, mb, bws)
            ck = lambda fn, *args: fn(*args) if fn is blk else ck_(fn, *args)
        else:
            blk = lambda a, ma, b, mb: _block_kernel_sums_autograd(a, ma, b, mb, bws)
            ck = ck_
        slf = lambda a, ma: _self_kernel_sums(a, ma, bws)
        idxA = [torch.as_tensor(c, device=dev, dtype=torch.long) for c in chA]
        if hB is None:
            # intra: blocks (p, q) with p <= q only; K[i, j] for i != j is read from the block that
            # contains the unordered pair, the diagonal blocks also give the within-graph means
            parts = []
            for p in range(len(chA)):
                for q in range(p, len(chA)):
                    s = ck(blk, padA[p][0], padA[p][1], padA[q][0], padA[q][1])
                    parts.append((p, q, s))
            S = _assemble(GA, GA, nbw, idxA, idxA, parts, symmetric=True, dtype=hA.dtype, device=dev)
            K = S / (cntA[:, None, None] * cntA[None, :, None])
            kA = torch.diagonal(K, dim1=0, dim2=1).t()       # (GA, nbw): within-graph means
            return K, kA
        szB = (ptrB[1:] - ptrB[:-1]).cpu().numpy()
        GB = len(szB)
        chB = _chunks_by_size(szB, cap)
        padB = [_pad(hB, ptrB, torch.as_tensor(c, device=dev), dev) for c in chB]
        idxB = [torch.as_tensor(c, device=dev, dtype=torch.long) for c in chB]
        cntB = torch.as_tensor(szB, dtype=hB.dtype, device=dev)
        parts = []
        for p in range(len(chA)):
            for q in range(len(chB)):
                parts.append((p, q, ck(blk, padA[p][0], padA[p][1], padB[q][0], padB[q][1])))
        S = _assemble(GA, GB, nbw, idxA, idxB, parts, symmetric=False, dtype=hA.dtype, device=dev)
        K = S / (cntA[:, None, None] * cntB[None, :, None])
        sA = torch.zeros(GA, nbw, dtype=hA.dtype, device=dev).index_put(
            (torch.cat(idxA),), torch.cat([ck(slf, *padA[p]) for p in range(len(chA))], 0))
        sB = torch.zeros(GB, nbw, dtype=hB.dtype, device=dev).index_put(
            (torch.cat(idxB),), torch.cat([ck(slf, *padB[q]) for q in range(len(chB))], 0))
        kA = sA / (cntA[:, None] ** 2)
        kB = sB / (cntB[:, None] ** 2)
        return K, kA, kB

    def _compute_embeddings_node_level(self, graphs):
        """

        """
        # CHANGE: graphs is a GraphBatch (dgl.batch of a list is not used by the Trainer)
        batched_graph = graphs
        num_graphs = graphs.num_graphs

        h = batched_graph.x
        h = self.forward(batched_graph, h)

        # Build graph_ids
        num_nodes_per_graph = batched_graph.batch_num_nodes().tolist()
        graph_ids_list = []
        for i, n in enumerate(num_nodes_per_graph):
            graph_ids_list.extend([i]*n)
        graph_ids = torch.tensor(graph_ids_list, device=h.device)
        return h, graph_ids, num_graphs

    def _pairwise_distances(self, embeddings):
        norm = (embeddings ** 2).sum(dim=1, keepdim=True)
        dist_sq = norm + norm.t() - 2.0 * torch.mm(embeddings, embeddings.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)
        return dist_sq

    def _pairwise_distances_cross(self, embA, embB):
        normA = (embA ** 2).sum(dim=1, keepdim=True)  # [N,1]
        normB = (embB ** 2).sum(dim=1, keepdim=True)  # [M,1]
        dist_sq_cross = normA + normB.t() - 2.0 * (embA @ embB.t())
        dist_sq_cross = torch.clamp(dist_sq_cross, min=0.0)
        return dist_sq_cross

    def _compute_mmd_sup(self, dist_ii, dist_jj, dist_ij):
        device = dist_ii.device
        mmd_sup = torch.zeros(1, device=device)

        for bw in self.bandwidths:
            K_ii = torch.exp(-dist_ii / bw).mean()
            K_jj = torch.exp(-dist_jj / bw).mean()
            K_ij = torch.exp(-dist_ij / bw).mean()
            mmd_sq = K_ii + K_jj - 2 * K_ij
            mmd_sq = torch.clamp(mmd_sq, min=1e-8)
            mmd = torch.sqrt(mmd_sq)
            mmd_sup = torch.max(mmd_sup, mmd)
        return mmd_sup


def _assemble(GA, GB, nbw, idxA, idxB, parts, symmetric, dtype, device):
    """CHANGE (vectorization helper): place block sums (gp, gq, nbw) at [idxA[p], idxB[q]] of a
    (GA, GB, nbw) matrix (autograd-friendly); with symmetric=True also mirror off-diagonal blocks."""
    rows_i, cols_j, vals = [], [], []
    for p, q, s in parts:
        ii = idxA[p][:, None].expand(-1, idxB[q].numel())
        jj = idxB[q][None, :].expand(idxA[p].numel(), -1)
        rows_i.append(ii.reshape(-1)); cols_j.append(jj.reshape(-1)); vals.append(s.reshape(-1, nbw))
        if symmetric and p != q:
            rows_i.append(jj.reshape(-1)); cols_j.append(ii.reshape(-1)); vals.append(s.reshape(-1, nbw))
    out = torch.zeros(GA, GB, nbw, dtype=dtype, device=device)
    return out.index_put((torch.cat(rows_i), torch.cat(cols_j)), torch.cat(vals, 0))


class LGKDE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        bandwidths: List[float] = [1.0],
        dropout: float = 0.1,
        batch_norm: bool = True,
        kde_dimension: int = 1,
        learn_kde_weights: bool = True
    ):
        super().__init__()

        # Core DGMMD model.
        self.dgmmd = DGMMD(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            bandwidths=bandwidths,
            dropout=dropout,
            batch_norm=batch_norm
        )

        # Learnable weights for multi-scale KDE
        self.kde_logits = nn.Parameter(torch.ones(len(bandwidths))/len(bandwidths))
        self.kde_dimension = kde_dimension

        # Optionally freeze KDE weights
        if not learn_kde_weights:
            self.kde_logits.requires_grad_(False)

    def compute_kde_scores(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        alpha = F.softmax(self.kde_logits, dim=0)
        M, N = dist_matrix.size()

        total_kde = torch.zeros(M, device=dist_matrix.device, dtype=dist_matrix.dtype)
        for k, bw in enumerate(self.dgmmd.bandwidths):
            exponent = -0.5 * (dist_matrix / bw)**2
            kernel_vals = torch.exp(exponent)
            partial_kde = (1.0 / N) * kernel_vals.sum(dim=1) / ((2*math.pi*(bw**2)) ** (0.5*self.kde_dimension))
            total_kde += alpha[k] * partial_kde

        return total_kde

    def get_reference_scores(self, reference_graphs) -> torch.Tensor:
        ref_dist = self.dgmmd.compute_distance_matrix(reference_graphs, graphs_b=None)
        return self.compute_kde_scores(ref_dist)

    def get_query_scores(self, query_graphs, reference_graphs) -> torch.Tensor:
        query_dist = self.dgmmd.compute_distance_matrix(query_graphs, graphs_b=reference_graphs)
        return self.compute_kde_scores(query_dist)


# ============================================================================================
# ng.py: NegativeSampleGenerator (SpCo branch not ported: unused by every configuration here)
# ============================================================================================
class NegativeSampleGenerator:
    def __init__(self,
                 perturbation_methods=('feature_swap','feature_noise','structure'),
                 attr_swap_ratio=0.2,
                 spectral_perturb_ratio=0.2,  # deprecated
                 feature_noise_std=0.1,
                 threshold=0.5,
                 weighted_graph=False,
                 verbose=False,
                 big_energy_threshold=0.5,
                 mid_energy_threshold=0.75,
                 max_ratio=10.0,
                 spco_theta=1.0,
                 spco_epsilon=0.1,
                 spco_lam=0.5,
                 spco_sinkhorn_iter=3
                 ):
        self.perturbation_methods = perturbation_methods
        self.attr_swap_ratio = attr_swap_ratio
        self.spectral_perturb_ratio = spectral_perturb_ratio  # deprecated
        self.feature_noise_std = feature_noise_std
        self.threshold = threshold
        self.weighted_graph = weighted_graph
        self.verbose = verbose

        self.big_energy_threshold = big_energy_threshold
        self.mid_energy_threshold = mid_energy_threshold
        self.max_ratio = max_ratio

        # SpCo
        self.spco_theta = spco_theta
        self.spco_epsilon = spco_epsilon
        self.spco_lam = spco_lam
        self.spco_sinkhorn_iter = spco_sinkhorn_iter
        # CHANGE: SpCo is not ported (no configuration uses it); weighted graphs are not ported
        assert 'spco' not in self.perturbation_methods and not self.weighted_graph
        # CHANGE: cache of the structure perturbation, a deterministic function of the graph's
        # adjacency (SVD + thresholding, no randomness); key = (graph id, increase_edges)
        self._struct_cache = {}

    def generate_negative_graph(self, g, increase_edges=True):
        """g: dict(x=<view of the batch's CPU feature copy>, edge_index=<with self-loops>, gidx=<id>).
        Returns (x_neg, edge_index_neg). Mirrors dgl: clone -> feature_perturbation -> structure ->
        remove_self_loop -> add_self_loop."""
        g_neg = dict(g)        # CHANGE: g.clone() (DGL clone shares the feature tensor storage)

        if any(m in ['feature_swap', 'feature_noise'] for m in self.perturbation_methods):
            g_neg = self.feature_perturbation(g_neg)

        if 'structure' in self.perturbation_methods:
            g_neg = self.structure_perturbation_svd(g_neg, increase_edges)

        n = g_neg["x"].size(0)
        ei = remove_self_loop(g_neg["edge_index"])
        ei = add_self_loop(ei, n)

        if self.verbose:
            print(f"Generated negative graph: {ei.size(1)} edges (increase={increase_edges})")

        return g_neg["x"], ei

    def feature_perturbation(self, g):
        g_neg = dict(g)        # CHANGE: g.clone() -- shares the feature storage, exactly like DGL
        features = g_neg["x"]
        num_nodes = features.size(0)

        if 'feature_swap' in self.perturbation_methods:
            num_swaps = int(num_nodes * self.attr_swap_ratio)
            if num_swaps > 1:
                idx = torch.randperm(num_nodes)[:num_swaps]
                swapped_idx = torch.roll(idx, shifts=1)
                features[idx] = features[swapped_idx]          # in place (as in the original)

        if 'feature_noise' in self.perturbation_methods:
            noise = torch.randn_like(features) * self.feature_noise_std
            features = features + noise

        g_neg["x"] = features
        return g_neg

    def structure_perturbation_svd(self, g, increase_edges=True):
        key = (g.get("gidx"), bool(increase_edges))
        if key[0] is not None and key in self._struct_cache:     # CHANGE: deterministic -> cached
            ei = self._struct_cache[key]
        elif key[0] is not None:
            # CHANGE: one SVD for both directions (the original computes the identical SVD twice)
            e_sub, e_add = self._structure_edges(g["edge_index"], g["x"].size(0), (False, True))
            self._struct_cache[(key[0], False)] = e_sub
            self._struct_cache[(key[0], True)] = e_add
            ei = self._struct_cache[key]
        else:
            ei = self._structure_edges(g["edge_index"], g["x"].size(0), (bool(increase_edges),))[0]
        # CHANGE: the original builds a new graph whose features are a clone of the (perturbed) features
        return dict(x=g["x"].clone(), edge_index=ei, gidx=g.get("gidx"))

    def _structure_edges(self, edge_index, n, directions):
        """structure_perturbation_svd for each increase_edges flag in `directions` (one SVD)."""
        # CHANGE: dense adjacency as DGLGraph.adj().to_dense() (row = src, col = dst, float32)
        A = np.zeros((n, n), dtype=np.float32)
        np.add.at(A, (edge_index[0].numpy(), edge_index[1].numpy()), 1.0)

        try:
            U, S, Vt = svd(A, full_matrices=False)
        except Exception as e:
            warnings.warn(f"[WARN] SVD: {e}")
            return [edge_index.clone() for _ in directions]

        out = []
        for increase_edges in directions:
            S_new = self.spectral_perturbation(S, increase_edges)

            A_perturbed = (U * S_new) @ Vt
            A_perturbed = 0.5*(A_perturbed + A_perturbed.T)
            A_perturbed[A_perturbed<0] = 0

            A_perturbed = (A_perturbed>self.threshold).astype(float)

            A_sp = csr_matrix(A_perturbed)
            src, dst = A_sp.nonzero()
            out.append(torch.stack([torch.tensor(src), torch.tensor(dst)]).long())
        return out

    def spectral_perturbation(self, S, increase_edges=True):     # verbatim
        sorted_idx_desc = np.argsort(-S)
        S_desc = S[sorted_idx_desc]
        cum_energy = np.cumsum(S_desc**2)
        total_energy = cum_energy[-1] if len(cum_energy)>0 else 1e-9

        def find_cut_index(thres):

            for i in range(len(cum_energy)):
                if cum_energy[i]/total_energy >= thres:
                    return i+1
            return len(cum_energy)

        big_cut = find_cut_index(self.big_energy_threshold)
        mid_cut = find_cut_index(self.mid_energy_threshold)

        seg_big   = np.arange(0, big_cut)
        seg_mid   = np.arange(big_cut, mid_cut)
        seg_small = np.arange(mid_cut, len(S_desc))

        mean_big   = S_desc[seg_big].mean()   if len(seg_big)>0 else 1.0
        mean_small = S_desc[seg_small].mean() if len(seg_small)>0 else 1.0

        ratio_raw = mean_big/(mean_small+1e-10)
        ratio_clamped = min(ratio_raw, self.max_ratio)

        if self.verbose:
            print(f" [EnergySplit] big=[0..{big_cut-1}], mid=[{big_cut}..{mid_cut-1}], small=[{mid_cut}..end]")
            print(f"   mean_big={mean_big:.4f}, mean_small={mean_small:.4f}, ratio={ratio_raw:.4f} => clamp={ratio_clamped:.4f}")

        S_new = S.copy()

        seg_big_orig   = sorted_idx_desc[seg_big]
        seg_mid_orig   = sorted_idx_desc[seg_mid]
        seg_small_orig = sorted_idx_desc[seg_small]

        if increase_edges:

            if len(seg_small_orig)>0:
                S_new[seg_small_orig] *= ratio_clamped
        else:

            if len(seg_big_orig)>0:
                S_new[seg_big_orig] /= (ratio_clamped+1e-10)
        return S_new
