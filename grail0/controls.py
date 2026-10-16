"""Controls for the GRAIL-0 Gate-1 pilot (same folds, fitted on normal training graphs only,
fixed hyper-parameters and seeds; nothing is tuned).

  ctrl_size_lw        size only: [log n, log(m+1), density] -> Ledoit-Wolf Mahalanobis.
  ctrl_gc_iforest     same-vocabulary graphlet counts, raw (no size adjustment):
                      [log(1+n), log(1+m), log(1+c_k) for every graphlet type] -> IsolationForest.
  ctrl_gc_lw          same features -> Ledoit-Wolf Mahalanobis (+ the same fixed ridge as channel A).
  ctrl_gclab_iforest  graphlet counts + node-label histogram log(1+count_l) -> IsolationForest.
  ctrl_gclab_lw       graphlet counts + node-label histogram -> Ledoit-Wolf Mahalanobis.
  ctrl_wl_iforest     WL-subtree kernel + IsolationForest (the kernel+iF baseline of Zhao & Akoglu /
                      GLocalKD / UB-GOLD): 3 WL iterations (initial label = node label, or degree if
                      the dataset has none), cosine-normalized count vectors over all iterations;
                      features = kernel values against <= 1000 training graphs (fixed seed).
"""
import math

import numpy as np
from sklearn.ensemble import IsolationForest

from grail0.channels import size_feats, _lw, CONFIG

IF_TREES = 300


def gc_feats(F):
    out = []
    for f in F:
        out.append(np.concatenate([[math.log1p(f["n"]), math.log1p(f["m"])],
                                   np.log1p(f["counts"].astype(np.float64))]))
    return np.stack(out)


def label_hist(graphs, L):
    H = np.zeros((len(graphs), L + 1))
    for i, g in enumerate(graphs):
        lab = np.minimum(np.asarray(g["node_label"], dtype=np.int64), L)
        H[i] = np.bincount(lab, minlength=L + 1)[: L + 1]
    return np.log1p(H)


def mahal(Xtr, Xte, ridge):
    mu, P = _lw(Xtr, ridge)
    D = Xte - mu
    return np.sum(D @ P * D, 1)


def iforest(Xtr, Xte, seed):
    clf = IsolationForest(n_estimators=IF_TREES, random_state=seed, n_jobs=1).fit(Xtr)
    return -clf.score_samples(Xte)


# ----------------------------------------------------------------------------- WL
def _splitmix(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def wl_labels(graphs, h=3):
    """WL relabeling on all graphs jointly (label-free). Returns list over iterations 0..h of
    int64 label arrays over the stacked nodes, plus node offsets. Multiset hashing:
    new(v) = H(old(v), sum_{u in N(v)} H'(old(u))) with 64-bit splitmix hashes."""
    sizes = np.array([g["n"] for g in graphs], dtype=np.int64)
    off = np.zeros(len(graphs) + 1, dtype=np.int64); np.cumsum(sizes, out=off[1:])
    tot = int(off[-1])
    src, dst = [], []
    for i, g in enumerate(graphs):
        e = np.asarray(g["edges"], dtype=np.int64).reshape(-1, 2) + off[i]
        src.append(e[:, 0]); dst.append(e[:, 1]); src.append(e[:, 1]); dst.append(e[:, 0])
    src = np.concatenate(src) if src else np.zeros(0, np.int64)
    dst = np.concatenate(dst) if dst else np.zeros(0, np.int64)
    order = np.argsort(src, kind="stable"); src, dst = src[order], dst[order]
    deg = np.bincount(src, minlength=tot)
    indptr = np.zeros(tot + 1, dtype=np.int64); np.cumsum(deg, out=indptr[1:])
    if all(g.get("node_label") is not None for g in graphs):
        lab0 = np.concatenate([np.asarray(g["node_label"], dtype=np.int64) for g in graphs])
    else:
        lab0 = deg.astype(np.int64)
    _, cur = np.unique(lab0, return_inverse=True)
    out = [cur.astype(np.int64)]
    with np.errstate(over="ignore"):
        for it in range(h):
            hv = _splitmix(cur.astype(np.uint64) + np.uint64(1000003 * (it + 1)))
            nb = hv[dst]
            s = np.zeros(tot, dtype=np.uint64)
            nz = deg > 0
            if len(nb):
                red = np.add.reduceat(nb, indptr[:-1][nz]) if nz.any() else np.zeros(0, np.uint64)
                s[nz] = red
            sig = _splitmix(_splitmix(cur.astype(np.uint64) * np.uint64(0x100000001B3)) ^ s)
            _, cur = np.unique(sig, return_inverse=True)
            # make labels unique per iteration across iterations
            out.append(cur.astype(np.int64))
    return out, off


def wl_sparse(wl, off):
    """Sparse WL count matrix (graphs x WL labels of all iterations 0..h)."""
    import scipy.sparse as sp
    G = len(off) - 1
    rows = np.repeat(np.arange(G), np.diff(off))
    mats = []
    for lab in wl:
        mats.append(sp.csr_matrix((np.ones(len(lab)), (rows, lab)), shape=(G, int(lab.max()) + 1)))
    return sp.hstack(mats).tocsr()


def wl_kernel_feats(Xs, pos_tr, pos_te, seed, max_ref=1000):
    """Normalized WL-subtree kernel rows against (<= max_ref sampled) training graphs."""
    from sklearn.preprocessing import normalize
    rng = np.random.default_rng(seed)
    ref = np.asarray(pos_tr)
    if len(ref) > max_ref:
        ref = np.sort(rng.choice(ref, max_ref, replace=False))
    Xn = normalize(Xs, norm="l2", axis=1)
    R = Xn[ref]
    Ktr = (Xn[np.asarray(pos_tr)] @ R.T).toarray()
    Kte = (Xn[np.asarray(pos_te)] @ R.T).toarray()
    return Ktr, Kte


def run_controls(Ftr, Fte, train, test, seed, L=None, wl=None):
    """Returns dict method -> test scores (higher = more anomalous)."""
    out = {}
    out["ctrl_size_lw"] = mahal(size_feats(Ftr), size_feats(Fte), CONFIG["ridge_S0"])
    Xtr, Xte = gc_feats(Ftr), gc_feats(Fte)
    out["ctrl_gc_iforest"] = iforest(Xtr, Xte, seed)
    out["ctrl_gc_lw"] = mahal(Xtr, Xte, CONFIG["ridge_A"])
    if L is not None and all(g.get("node_label") is not None for g in train):
        Htr, Hte = label_hist(train, L), label_hist(test, L)
        Xtr2, Xte2 = np.hstack([Xtr, Htr]), np.hstack([Xte, Hte])
        out["ctrl_gclab_iforest"] = iforest(Xtr2, Xte2, seed)
        out["ctrl_gclab_lw"] = mahal(Xtr2, Xte2, CONFIG["ridge_A"])
    if wl is not None:
        Xs, pos_tr, pos_te = wl
        Ktr, Kte = wl_kernel_feats(Xs, pos_tr, pos_te, seed)
        out["ctrl_wl_iforest"] = iforest(Ktr, Kte, seed)
    return out
