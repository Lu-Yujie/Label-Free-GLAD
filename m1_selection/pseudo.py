"""Typed pseudo-anomaly generator for label-free checkpoint selection (M1 kill test).

FIXED A PRIORI (see NOTES.md, pre-registration). Built ONLY from the held-out normal graphs of
a (dataset, fold) -- the same 20% carve-out as deep_rerun -- plus, for P-label-noise, the
node-label marginal of the fold's normal training graphs (no test graph is ever touched).
Every family is size-preserving (identical node count n and edge count m):

  rewire      degree-preserving double-edge swaps touching ~15% of the edges
              (n_swaps = max(1, round(0.15 m / 2))); if fewer swaps succeed (e.g. stars,
              near-complete graphs) the remainder is done by random rewiring that keeps m
              (delete a random edge, add a random non-edge).
  motif       plant K4 cliques on random connected 4-node sets that are not already cliques
              (k_plant = max(1, round(m / 40)), disjoint node sets when possible); the
              missing edges are added and the SAME number of random edges outside the
              planted sets is deleted (preferring edges whose endpoints keep degree >= 1),
              so m is unchanged. Fallback when no connected non-clique 4-set exists: close
              open triangles (u-v-w, u !~ w) instead.
  roleswap    (node-labelled datasets only) swap the labels of k = max(1, round(0.05 n))
              node pairs whose structural roles differ: u from the high-degree nodes
              (deg >= 75th percentile), v from the low-degree nodes (deg <= 25th percentile),
              deg(u) > deg(v) and label(u) != label(v); relaxed to any pair with
              deg(u) > deg(v) and different labels when needed. The label multiset is kept.
  labelnoise  (node-labelled datasets only) resample the labels of k = max(1, round(0.10 n))
              uniformly chosen nodes from the fold's training node-label marginal (all normal
              training graphs); redrawn (<= 50 times) until at least one label changes.

Each held-out normal graph yields exactly one perturbed copy per applicable family. The RNG of
copy i of family f is np.random.default_rng([BASE_SEED, crc32(dataset), fold, f_id, i]); no
global RNG (python/numpy/torch) is touched, so building P never changes model training.

Usage:
  python -B pseudo.py build --dataset BZR --fold 0       # writes $GLAD_DATA_ROOT/m1_pseudo/BZR_fold0.pt
  python -B pseudo.py build --all --workers 16
  python -B pseudo.py sanity                              # size-only AUROC etc. -> results/m1_pseudo_sanity.csv
"""
import os.path as _osp, sys as _sys  # M1: this directory holds select.py, which would shadow the
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))  # stdlib module `select` (used by
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]  # subprocess): drop it from sys.path

import argparse
import math
import os
import os.path as osp
import sys
import zlib

import numpy as np

BASE_SEED = 20261012
FAMILIES = ["rewire", "motif", "roleswap", "labelnoise"]
FAM_ID = {f: i for i, f in enumerate(FAMILIES)}
STRUCT_FAMILIES = ["rewire", "motif"]
PSEUDO_DIR = os.environ.get("M1_PSEUDO", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/m1_pseudo"))
DATASETS = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]

REWIRE_FRAC = 0.15
MOTIF_EDGES_PER_PLANT = 40
ROLESWAP_FRAC = 0.05
LABELNOISE_FRAC = 0.10


# ------------------------------------------------------------------------------------------
# graph helpers (undirected simple graphs as sorted lists of (i, j) with i < j)
# ------------------------------------------------------------------------------------------
def edges_from_index(edge_index, n):
    """edge_index (2, E) array-like with both directions -> sorted unique (i<j) list, no loops."""
    ei = np.asarray(edge_index)
    if ei.size == 0:
        return []
    a, b = ei[0].astype(np.int64), ei[1].astype(np.int64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keep = lo != hi
    s = set(zip(lo[keep].tolist(), hi[keep].tolist()))
    assert all(0 <= i < n and 0 <= j < n for i, j in s)
    return sorted(s)


def index_from_edges(edges):
    """sorted (i<j) list -> (2, 2m) int64 array, both directions, sorted by (row, col)."""
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    e = np.asarray(edges, dtype=np.int64)
    both = np.concatenate([e, e[:, ::-1]], 0)
    order = np.lexsort((both[:, 1], both[:, 0]))
    return both[order].T.copy()


def _key(i, j):
    return (i, j) if i < j else (j, i)


def adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def degrees(n, edges):
    d = np.zeros(n, dtype=np.int64)
    for i, j in edges:
        d[i] += 1
        d[j] += 1
    return d


def _random_nonedge(n, eset, rng, max_tries=200):
    """Uniform random non-adjacent pair (rejection sampling, exhaustive fallback)."""
    if n < 2 or len(eset) >= n * (n - 1) // 2:
        return None
    for _ in range(max_tries):
        i, j = rng.integers(0, n, size=2)
        if i != j and _key(int(i), int(j)) not in eset:
            return _key(int(i), int(j))
    cand = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in eset]
    return cand[int(rng.integers(len(cand)))] if cand else None


# ------------------------------------------------------------------------------------------
# families
# ------------------------------------------------------------------------------------------
def p_rewire(n, edges, rng, frac=REWIRE_FRAC):
    edges = list(edges)
    m = len(edges)
    info = dict(swaps=0, fallback_edges=0, target_swaps=0)
    if m == 0:
        return edges, info
    target = max(1, int(round(frac * m / 2.0)))
    info["target_swaps"] = target
    eset = set(edges)
    tries = 0
    while info["swaps"] < target and tries < 100 * target and m >= 2:
        tries += 1
        k1, k2 = rng.choice(m, size=2, replace=False)
        a, b = edges[k1]
        c, d = edges[k2]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        e1, e2 = _key(a, d), _key(c, b)
        if e1 in eset or e2 in eset:
            continue
        eset.discard(edges[k1]); eset.discard(edges[k2])
        eset.add(e1); eset.add(e2)
        edges[k1], edges[k2] = e1, e2
        info["swaps"] += 1
    # fallback: random rewiring keeping m (only for the swaps that could not be done)
    rem = 2 * (target - info["swaps"])
    for _ in range(rem):
        new = _random_nonedge(n, eset, rng)
        if new is None:
            break
        k = int(rng.integers(m))
        eset.discard(edges[k])
        eset.add(new)
        edges[k] = new
        info["fallback_edges"] += 1
    out = sorted(eset)
    assert len(out) == m
    return out, info


def _connected_set(adj, size, rng, excluded, max_tries=60):
    """Random connected node set of `size` nodes that is NOT already a clique; nodes in
    `excluded` are avoided (dropped on a second pass if needed)."""
    n = len(adj)
    for avoid in (excluded, set()):
        starts = [v for v in range(n) if adj[v] and v not in avoid]
        if not starts:
            continue
        for _ in range(max_tries):
            s = starts[int(rng.integers(len(starts)))]
            S = [s]
            ok = True
            while len(S) < size:
                front = sorted({u for v in S for u in adj[v]} - set(S) - avoid)
                if not front:
                    ok = False
                    break
                S.append(front[int(rng.integers(len(front)))])
            if not ok:
                continue
            missing = [(S[a], S[b]) for a in range(size) for b in range(a + 1, size)
                       if S[b] not in adj[S[a]]]
            if missing:
                return S, [_key(i, j) for i, j in missing]
    return None, None


def p_motif(n, edges, rng, edges_per_plant=MOTIF_EDGES_PER_PLANT):
    m = len(edges)
    info = dict(plants=0, triangle_closures=0, added=0, deleted=0, k_plant=0)
    if m == 0 or n < 3:
        return list(edges), info
    k_plant = max(1, int(round(m / float(edges_per_plant))))
    info["k_plant"] = k_plant
    eset = set(edges)
    adj = adjacency(n, edges)
    protected = set()      # edges inside planted sets (never deleted)
    used_nodes = set()
    added = []
    for _ in range(k_plant):
        S, missing = _connected_set(adj, 4, rng, used_nodes) if n >= 4 else (None, None)
        if S is None:
            # fallback: close an open triangle u - v - w (u, w non-adjacent)
            wedges = [(u, v, w) for v in range(n) for u in adj[v] for w in adj[v] if u < w and w not in adj[u]]
            if not wedges:
                break
            u, v, w = wedges[int(rng.integers(len(wedges)))]
            S, missing = [u, v, w], [_key(u, w)]
            info["triangle_closures"] += 1
        else:
            info["plants"] += 1
        for i, j in missing:
            eset.add((i, j)); adj[i].add(j); adj[j].add(i)
            added.append((i, j))
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                protected.add(_key(S[a], S[b]))
        used_nodes |= set(S)
    # delete the same number of edges outside the planted sets
    n_del = len(added)
    deg = degrees(n, sorted(eset))
    for _ in range(n_del):
        cand = [e for e in sorted(eset) if e not in protected]
        if not cand:
            break
        safe = [e for e in cand if deg[e[0]] >= 2 and deg[e[1]] >= 2]
        pool = safe if safe else cand
        e = pool[int(rng.integers(len(pool)))]
        eset.discard(e); adj[e[0]].discard(e[1]); adj[e[1]].discard(e[0])
        deg[e[0]] -= 1; deg[e[1]] -= 1
        info["deleted"] += 1
    info["added"] = len(added)
    out = sorted(eset)
    if len(out) != m:  # could not delete enough (everything protected): undo -> unchanged copy
        return list(edges), dict(info, undone=1)
    return out, info


def p_roleswap(n, edges, labels, rng, frac=ROLESWAP_FRAC):
    labels = np.asarray(labels).copy()
    info = dict(k=0, swaps=0, relaxed=0)
    if n < 2:
        return labels, info
    k = max(1, int(round(frac * n)))
    info["k"] = k
    deg = degrees(n, edges)
    hi_thr, lo_thr = np.percentile(deg, 75), np.percentile(deg, 25)
    used = set()
    for _ in range(k):
        pair = None
        H = [u for u in range(n) if deg[u] >= hi_thr and u not in used]
        L = [v for v in range(n) if deg[v] <= lo_thr and v not in used]
        for u in rng.permutation(H).tolist() if H else []:
            cand = [v for v in L if deg[v] < deg[u] and labels[v] != labels[u] and v != u]
            if cand:
                pair = (u, cand[int(rng.integers(len(cand)))])
                break
        if pair is None:  # relaxed: any role-differing pair with different labels
            free = [u for u in range(n) if u not in used]
            cands = [(u, v) for u in free for v in free if deg[u] > deg[v] and labels[u] != labels[v]]
            if not cands:
                break
            pair = cands[int(rng.integers(len(cands)))]
            info["relaxed"] += 1
        u, v = pair
        labels[u], labels[v] = labels[v], labels[u]
        used |= {u, v}
        info["swaps"] += 1
    return labels, info


def p_labelnoise(n, labels, marginal, rng, frac=LABELNOISE_FRAC, max_redraw=50):
    labels = np.asarray(labels)
    info = dict(k=0, changed=0, redraws=0)
    if n == 0:
        return labels.copy(), info
    k = max(1, int(round(frac * n)))
    info["k"] = k
    p = np.asarray(marginal, dtype=np.float64)
    p = p / p.sum()
    for r in range(max_redraw):
        new = labels.copy()
        idx = rng.choice(n, size=k, replace=False)
        new[idx] = rng.choice(len(p), size=k, p=p)
        ch = int((new != labels).sum())
        if ch > 0:
            info.update(changed=ch, redraws=r)
            return new, info
    info["redraws"] = max_redraw
    return labels.copy(), info


def copy_rng(ds, fold, fam, i):
    return np.random.default_rng([BASE_SEED, zlib.crc32(ds.encode()), int(fold), FAM_ID[fam], int(i)])


def perturb(ds, fold, fam, i, n, edges, labels, marginal):
    """One perturbed copy -> (edges, labels, info). labels may be None (unlabelled datasets)."""
    rng = copy_rng(ds, fold, fam, i)
    if fam == "rewire":
        e, info = p_rewire(n, edges, rng)
        return e, labels, info
    if fam == "motif":
        e, info = p_motif(n, edges, rng)
        return e, labels, info
    if fam == "roleswap":
        lab, info = p_roleswap(n, edges, labels, rng)
        return list(edges), lab, info
    if fam == "labelnoise":
        lab, info = p_labelnoise(n, labels, marginal, rng)
        return list(edges), lab, info
    raise ValueError(fam)


def load_sibling(name):
    """import a module of this directory by file path (the directory is NOT on sys.path)."""
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, osp.join(_HERE_, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------
# build the cache for one (dataset, fold) from the UB-GOLD pipeline (identical held-out set)
# ------------------------------------------------------------------------------------------
def labels_from_x(x):
    """UB-GOLD x on labelled TU datasets = one-hot node labels. Returns (labels, d) or (None, d)
    if x is not an exact one-hot matrix with >1 columns (e.g. Constant(1) on IMDB/REDDIT)."""
    x = np.asarray(x)
    d = x.shape[1]
    if d <= 1:
        return None, d
    if not (np.all((x == 0) | (x == 1)) and np.all(x.sum(1) == 1)):
        return None, d
    return x.argmax(1).astype(np.int64), d


def build(ds, fold, force=False):
    import torch
    os.makedirs(PSEUDO_DIR, exist_ok=True)
    out = osp.join(PSEUDO_DIR, f"{ds}_fold{fold}.pt")
    if osp.exists(out) and not force:
        return out
    run_m1 = load_sibling("run_m1")
    F = run_m1.prepare_fold(ds, fold)          # same code path as training (no RNG consumed)
    train_all, hold_pos = F["train_all"], F["hold_pos"]
    hold = [train_all[i] for i in hold_pos]
    # training node-label marginal (all normal training graphs of the fold)
    lab0, d = labels_from_x(train_all[0].x.numpy())
    labelled = lab0 is not None
    marginal = None
    if labelled:
        cnt = np.zeros(d, dtype=np.float64)
        for g in train_all:
            lab, _ = labels_from_x(g.x.numpy())
            assert lab is not None, "mixed one-hot / non-one-hot x"
            cnt += np.bincount(lab, minlength=d)
        marginal = cnt / cnt.sum()
    fams = FAMILIES if labelled else STRUCT_FAMILIES
    P = {f: [] for f in fams}
    diag = []
    sig = []
    for i, g in enumerate(hold):
        n = int(g.num_nodes)
        edges = edges_from_index(g.edge_index.numpy(), n)
        # UB-GOLD TU graphs are simple and symmetric: check that round-tripping is lossless
        ei_rt = index_from_edges(edges)
        ei0 = g.edge_index.numpy()
        sym_ok = ei_rt.shape == ei0.shape and set(map(tuple, ei_rt.T.tolist())) == set(map(tuple, ei0.T.tolist()))
        labels = labels_from_x(g.x.numpy())[0] if labelled else None
        sig.append(dict(pos=int(hold_pos[i]), n=n, m=len(edges), ei_hash=zlib.crc32(np.ascontiguousarray(ei0).tobytes()),
                        simple_symmetric=bool(sym_ok)))
        for f in fams:
            e2, lab2, info = perturb(ds, fold, f, i, n, edges, labels, marginal)
            assert len(e2) == len(edges), (ds, fold, f, i)
            ei2 = index_from_edges(e2)
            if labelled:
                x2 = np.zeros((n, d), dtype=np.float32)
                x2[np.arange(n), lab2] = 1.0
                assert np.array_equal(np.sort(lab2), np.sort(labels)) or f == "labelnoise"
            else:
                x2 = g.x.numpy().copy()
            changed_e = len(set(e2) - set(edges))
            changed_l = int((lab2 != labels).sum()) if labelled else 0
            P[f].append(dict(edge_index=torch.from_numpy(ei2), x=torch.from_numpy(x2.astype(g.x.numpy().dtype))))
            diag.append(dict(dataset=ds, fold=fold, family=f, i=i, n=n, m=len(edges), changed_edges=changed_e,
                             changed_labels=changed_l, identical=int(changed_e == 0 and changed_l == 0),
                             **{k: v for k, v in info.items() if v is not None}))
    payload = dict(dataset=ds, fold=fold, families=fams, labelled=labelled, label_dim=d,
                   marginal=marginal, hold_pos=list(map(int, hold_pos)), hold_sig=sig, P=P, diag=diag,
                   base_seed=BASE_SEED,
                   params=dict(REWIRE_FRAC=REWIRE_FRAC, MOTIF_EDGES_PER_PLANT=MOTIF_EDGES_PER_PLANT,
                               ROLESWAP_FRAC=ROLESWAP_FRAC, LABELNOISE_FRAC=LABELNOISE_FRAC))
    tmp = out + f".tmp{os.getpid()}"  # unique: CPU and GPU jobs may build concurrently
    torch.save(payload, tmp)
    os.replace(tmp, out)
    return out


def _build_one(t):
    ds, fold, force = t
    try:
        return build(ds, fold, force), None
    except Exception as e:  # noqa
        import traceback
        return None, f"{ds} {fold}: {traceback.format_exc()}"


def sanity(out_csv):
    """Size-only / edges-only AUROC on P (must be 50: P is size-preserving) plus simple
    structural statistics, and generator diagnostics, per dataset x family."""
    import torch
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    rows, drows = [], []
    for ds in DATASETS:
        for fold in range(5):
            fn = osp.join(PSEUDO_DIR, f"{ds}_fold{fold}.pt")
            if not osp.exists(fn):
                continue
            C = torch.load(fn, weights_only=False)
            drows += C["diag"]
            n0 = np.array([s["n"] for s in C["hold_sig"]])
            m0 = np.array([s["m"] for s in C["hold_sig"]])
            for f in C["families"]:
                n1 = np.array([int(p["x"].shape[0]) for p in C["P"][f]])
                m1 = np.array([int(p["edge_index"].shape[1] // 2) for p in C["P"][f]])
                y = np.r_[np.zeros(len(n0)), np.ones(len(n1))]
                rows.append(dict(dataset=ds, fold=fold, family=f, n_pairs=len(n0),
                                 size_only_auroc=100 * roc_auc_score(y, np.r_[n0, n1]),
                                 edges_only_auroc=100 * roc_auc_score(y, np.r_[m0, m1]),
                                 size_plus_edges_auroc=100 * roc_auc_score(y, np.r_[n0 + m0, n1 + m1])))
    R = pd.DataFrame(rows)
    D = pd.DataFrame(drows)
    R.to_csv(out_csv, index=False)
    agg = D.groupby(["dataset", "family"]).agg(n_copies=("i", "size"), identical_frac=("identical", "mean"),
                                               mean_changed_edges=("changed_edges", "mean"),
                                               mean_m=("m", "mean"), mean_changed_labels=("changed_labels", "mean"),
                                               mean_n=("n", "mean")).reset_index()
    for extra in ("fallback_edges", "relaxed", "triangle_closures"):
        if extra in D:
            agg = agg.merge(D.groupby(["dataset", "family"])[extra].apply(lambda s: float(np.nanmean(s)) if s.notna().any() else np.nan)
                            .rename(f"mean_{extra}").reset_index(), on=["dataset", "family"], how="left")
    agg.to_csv(out_csv.replace(".csv", "_generator.csv"), index=False)
    return R, agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "sanity"])
    ap.add_argument("--dataset")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        tasks = [(ds, f, a.force) for ds in DATASETS for f in range(5)] if a.all else [(a.dataset, a.fold, a.force)]
        if a.workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.workers) as pool:
                res = pool.map(_build_one, tasks, chunksize=1)
        else:
            res = [_build_one(t) for t in tasks]
        bad = [e for _, e in res if e]
        for r, e in res:
            print(r if r else e, flush=True)
        sys.exit(1 if bad else 0)
    else:
        here = osp.dirname(osp.abspath(__file__))
        R, agg = sanity(osp.join(osp.dirname(here), "results", "m1_pseudo_sanity.csv"))
        print(R.groupby("family")[["size_only_auroc", "edges_only_auroc", "size_plus_edges_auroc"]].agg(["mean", "min", "max"]))
        print(agg.to_string())
