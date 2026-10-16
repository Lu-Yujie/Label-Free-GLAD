"""GRAIL-0 graphlet engine: exact, permutation-invariant graphlet and orbit counts.

Vocabulary (fixed): all connected graphlets on 3, 4, 5 nodes (2 + 6 + 21 = 29 types) and
their automorphism orbits (1 edge orbit + 3 + 11 + 58 = 73 orbits; orbit 0 = degree).
Numbering: graphlets sorted by (size, #edges, descending degree sequence, canonical mask);
orbits inside a graphlet sorted by (degree inside the graphlet, canonical vertex). For 3-4
nodes this coincides with the Przulj / ORCA numbering (orbits 0..14); the 5-node numbering
(orbits 15..72) is our own (documented in NOTES.md, names in GRAPHLET_NAMES).

Engines (both exact, hence exactly permutation invariant):
  'esu'   : C implementation of ESU enumeration (every connected induced k-subset visited
            once, k = 3..kmax) + bitmask -> (type, orbit) lookup. Used with kmax = 5.
  'comb4' : closed-form non-induced 2-4-node orbit counts in C, converted to induced counts
            with the inclusion matrix (kmax = 4). Used for social graphs with hubs.
Engine/kmax are fixed a priori by dataset family (ENGINE_BY_FAMILY), never by labels.
"""
import ctypes
import itertools
import os
import os.path as osp
import pickle
import subprocess
import time

import numpy as np
import networkx as nx

HERE = osp.dirname(osp.abspath(__file__))
SO_PATH = osp.join(HERE, "_graphlets_c.so")
C_PATH = osp.join(HERE, "graphlets_c.c")
CACHE_DIR = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/cache/graphlets")
CACHE_VERSION = 1

# fixed a-priori engine choice by dataset family (never by labels)
FAMILY = {
    "AIDS": "molecule", "BZR": "molecule", "COX2": "molecule", "DHFR": "molecule",
    "NCI1": "molecule", "MUTAG": "molecule", "Tox21_p53": "molecule", "Tox21_MMP": "molecule",
    "ENZYMES": "protein", "PROTEINS_full": "protein", "DD": "protein",
    "IMDB-BINARY": "social", "COLLAB": "social", "REDDIT-BINARY": "social",
}
ENGINE_BY_FAMILY = {"molecule": ("esu", 5), "protein": ("esu", 5), "social": ("comb4", 4)}


def engine_for(dataset):
    return ENGINE_BY_FAMILY[FAMILY.get(dataset, "molecule")]


# ----------------------------------------------------------------------------- tables
def _bit(i, j):
    if i > j:
        i, j = j, i
    return j * (j - 1) // 2 + i


def _mask_edges(s, mask):
    return [(i, j) for j in range(s) for i in range(j) if (mask >> _bit(i, j)) & 1]


def _named_graphs():
    """Human-readable names for graphlets (matched by isomorphism)."""
    G = {}
    G["P3 (wedge)"] = nx.path_graph(3)
    G["triangle"] = nx.complete_graph(3)
    G["P4 (path)"] = nx.path_graph(4)
    G["3-star (claw)"] = nx.star_graph(3)
    G["C4 (square)"] = nx.cycle_graph(4)
    G["paw (tailed triangle)"] = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    G["diamond (K4-e)"] = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)])
    G["K4 (4-clique)"] = nx.complete_graph(4)
    G["P5 (path)"] = nx.path_graph(5)
    G["4-star"] = nx.star_graph(4)
    G["fork (chair)"] = nx.Graph([(0, 1), (0, 2), (0, 3), (3, 4)])
    G["C5 (5-ring)"] = nx.cycle_graph(5)
    G["K5 (5-clique)"] = nx.complete_graph(5)
    G["house"] = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)])
    G["bowtie (butterfly)"] = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)])
    G["bull"] = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (1, 4)])
    G["cricket"] = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (0, 4)])
    G["banner (C4+tail)"] = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4)])
    G["long paw (triangle+P2 tail)"] = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)])
    G["kite (diamond+tail at deg-2)"] = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (3, 4)])
    G["dart (diamond+tail at deg-3)"] = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4)])
    G["K4+tail"] = nx.Graph(list(nx.complete_graph(4).edges()) + [(0, 4)])
    G["K2,3"] = nx.complete_bipartite_graph(2, 3)
    G["gem (fan)"] = nx.Graph([(0, 1), (1, 2), (2, 3)] + [(4, i) for i in range(4)])
    G["W4 (wheel)"] = nx.wheel_graph(5)
    G["K5-e"] = nx.Graph([e for e in nx.complete_graph(5).edges() if e != (0, 1)])
    G["book (K2+3K1)"] = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)])
    G["K2,3+side-edge"] = nx.Graph([(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 4)])
    G["K5-P3"] = nx.Graph([e for e in nx.complete_graph(5).edges() if e not in ((0, 1), (1, 2))])
    return G


def build_tables(verbose=False):
    """Lookup tables for sizes 3..5: type id and orbit per position for every mask."""
    tab_type = np.full((6, 1024), -1, dtype=np.int16)
    tab_orb = np.full((6, 1024, 5), -1, dtype=np.int16)
    graphlets = []          # list of dicts
    orbit_graphlet = [-1]   # orbit 0 = edge (degree), graphlet id -1
    orbit_names = ["edge / degree"]
    named = _named_graphs()
    next_orbit = 1
    for s in (3, 4, 5):
        nb = s * (s - 1) // 2
        perms = list(itertools.permutations(range(s)))
        per_mask = {}
        canon_set = {}
        for mask in range(1 << nb):
            edges = _mask_edges(s, mask)
            g = nx.Graph(); g.add_nodes_from(range(s)); g.add_edges_from(edges)
            if not nx.is_connected(g):
                continue
            best, bestp = None, None
            for p in perms:
                pm = 0
                for (i, j) in edges:
                    pm |= 1 << _bit(p[i], p[j])
                if best is None or pm < best:
                    best, bestp = pm, p
            per_mask[mask] = (best, bestp)
            canon_set[best] = True
        cands = []
        for cm in canon_set:
            edges = _mask_edges(s, cm)
            deg = [0] * s
            for i, j in edges:
                deg[i] += 1; deg[j] += 1
            cands.append(((len(edges), tuple(sorted(deg, reverse=True)), cm), cm, edges, deg))
        cands.sort(key=lambda t: t[0])
        canon_info = {}
        for key, cm, edges, deg in cands:
            # automorphism orbits of the canonical graphlet
            auts = []
            for p in perms:
                pm = 0
                for (i, j) in edges:
                    pm |= 1 << _bit(p[i], p[j])
                if pm == cm:
                    auts.append(p)
            orb_of = list(range(s))
            for v in range(s):
                orb_of[v] = min(p[v] for p in auts)
            reps = sorted(set(orb_of), key=lambda r: (deg[r], r))
            gid = len(graphlets)
            g = nx.Graph(); g.add_nodes_from(range(s)); g.add_edges_from(edges)
            name = None
            for nm, H in named.items():
                if H.number_of_nodes() == s and nx.is_isomorphic(g, H):
                    name = nm
                    break
            if name is None:
                name = "g%d-%de-#%d" % (s, len(edges), gid)
            rep2orbit = {}
            degcount = {}
            for r in reps:
                rep2orbit[r] = next_orbit
                d = deg[r]
                k = degcount.get(d, 0); degcount[d] = k + 1
                orbit_graphlet.append(gid)
                orbit_names.append("%s|deg%d%s" % (name, d, "" if k == 0 else chr(ord("a") + k)))
                next_orbit += 1
            canon_info[cm] = (gid, [rep2orbit[orb_of[v]] for v in range(s)])
            graphlets.append({"id": gid, "size": s, "n_edges": len(edges), "canon_mask": cm,
                              "edges": edges, "name": name, "n_auts": len(auts),
                              "orbits": sorted(set(rep2orbit.values()))})
        for mask, (cm, p) in per_mask.items():
            gid, vorb = canon_info[cm]
            tab_type[s, mask] = gid
            for pos in range(s):
                tab_orb[s, mask, pos] = vorb[p[pos]]
    # fix names that are still duplicated (shouldn't happen) and sanity checks
    names = [g["name"] for g in graphlets]
    assert len(set(names)) == len(names), names
    n3 = sum(g["size"] == 3 for g in graphlets); n4 = sum(g["size"] == 4 for g in graphlets)
    n5 = sum(g["size"] == 5 for g in graphlets)
    assert (n3, n4, n5) == (2, 6, 21), (n3, n4, n5)
    assert next_orbit == 73, next_orbit
    return {"tab_type": tab_type, "tab_orb": tab_orb, "graphlets": graphlets,
            "orbit_graphlet": np.array(orbit_graphlet), "orbit_names": orbit_names,
            "n_types": {3: 2, 4: 8, 5: 29}, "n_orbits": {3: 4, 4: 15, 5: 73}}


_TABLES = None


def tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = build_tables()
    return _TABLES


# ----------------------------------------------------------------------------- C lib
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        if (not osp.exists(SO_PATH)) or osp.getmtime(SO_PATH) < osp.getmtime(C_PATH):
            subprocess.check_call(["gcc", "-O3", "-fPIC", "-shared", "-o", SO_PATH, C_PATH])
        lib = ctypes.CDLL(SO_PATH)
        P = ctypes.c_void_p
        lib.esu_count.restype = ctypes.c_int64
        lib.esu_count.argtypes = [ctypes.c_int64, P, P, ctypes.c_int, P, P, P, P, ctypes.c_int]
        lib.orbit4_noninduced.restype = ctypes.c_int
        lib.orbit4_noninduced.argtypes = [ctypes.c_int64, P, P, P]
        _LIB = lib
    return _LIB


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def to_csr(n, edges):
    """Symmetric CSR with sorted rows from an (m,2) undirected edge list (i<j, no loops)."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(e):
        e = e[e[:, 0] != e[:, 1]]
        e = np.unique(np.sort(e, axis=1), axis=0)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, np.ascontiguousarray(dst.astype(np.int32)), len(e)


# ----------------------------------------------------------------------------- engines
def esu_counts(n, edges, kmax=5):
    T = tables()
    indptr, idx, m = to_csr(n, edges)
    n_types, n_orb = T["n_types"][kmax], T["n_orbits"][kmax]
    counts = np.zeros(29, dtype=np.int64)
    gdv = np.zeros((n, 73), dtype=np.int64)
    tt = np.ascontiguousarray(T["tab_type"]); to = np.ascontiguousarray(T["tab_orb"])
    if n > 0:
        r = _lib().esu_count(n, _ptr(indptr), _ptr(idx), kmax, _ptr(tt), _ptr(to),
                             _ptr(counts), _ptr(gdv), 73)
        if r < 0:
            raise RuntimeError("esu_count failed with code %d" % r)
        gdv[:, 0] = np.diff(indptr)
    return counts[:n_types].copy(), np.ascontiguousarray(gdv[:, :n_orb]), m


_MINV4 = None


def inclusion_matrix(sizes=(3, 4)):
    """M[j, i] = for an induced occurrence in which the node sits in orbit i, the number of
    connected spanning sub-edge-sets of that occurrence in which the node sits in orbit j.
    Then non-induced N_j(v) = sum_i M[j, i] * induced O_i(v). Brute force over the tables."""
    T = tables()
    M = np.zeros((73, 73), dtype=np.int64)
    for g in T["graphlets"]:
        s = g["size"]
        if s not in sizes:
            continue
        cm = g["canon_mask"]
        bits = [b for b in range(s * (s - 1) // 2) if (cm >> b) & 1]
        for i in g["orbits"]:
            pos = [p for p in range(s) if T["tab_orb"][s, cm, p] == i][0]  # representative
            for sub in range(1 << len(bits)):
                m2 = 0
                for k, b in enumerate(bits):
                    if (sub >> k) & 1:
                        m2 |= 1 << b
                if T["tab_type"][s, m2] < 0:
                    continue  # sub-edge-set not connected on all s nodes
                M[T["tab_orb"][s, m2, pos], i] += 1
    return M


def _minv4():
    global _MINV4
    if _MINV4 is None:
        M = inclusion_matrix((3, 4))[1:15, 1:15].astype(np.float64)
        Minv = np.linalg.inv(M)
        R = np.rint(Minv)
        assert np.allclose(Minv, R, atol=1e-9)
        _MINV4 = R.astype(np.int64)
    return _MINV4


def comb4_counts(n, edges):
    """Exact 2-4-node orbit counts (n,15) and 3-4-node graphlet counts (8,) via closed forms."""
    T = tables()
    indptr, idx, m = to_csr(n, edges)
    N = np.zeros((n, 15), dtype=np.int64)
    if n > 0:
        r = _lib().orbit4_noninduced(n, _ptr(indptr), _ptr(idx), _ptr(N))
        if r < 0:
            raise RuntimeError("orbit4_noninduced failed")
    O = np.zeros((n, 15), dtype=np.int64)
    O[:, 0] = N[:, 0]
    O[:, 1:15] = N[:, 1:15] @ _minv4().T
    if (O < 0).any():
        raise RuntimeError("negative induced orbit count (bug)")
    counts = np.zeros(8, dtype=np.int64)
    for g in T["graphlets"]:
        if g["size"] > 4:
            continue
        tot = O[:, g["orbits"]].sum()
        assert tot % g["size"] == 0
        counts[g["id"]] = tot // g["size"]
    return counts, O, m


def graph_features(g, engine="esu", kmax=5):
    """Label-free structural features of one graph dict (common.data format)."""
    n = int(g["n"])
    if engine == "esu":
        counts, gdv, m = esu_counts(n, g["edges"], kmax)
    elif engine == "comb4":
        assert kmax == 4
        counts, gdv, m = comb4_counts(n, g["edges"])
    else:
        raise ValueError(engine)
    return {"n": n, "m": int(m), "counts": counts, "gdv": gdv}


def _feat_chunk(graphs, engine, kmax):
    return [(g["gid"], graph_features(g, engine, kmax)) for g in graphs]


def dataset_features(name, graphs_by_gid, n_jobs=1, verbose=True):
    """Compute (or load) the per-graph graphlet cache for a dataset.
    graphs_by_gid: dict gid -> graph dict. Cache: CACHE_DIR/<name>.pkl."""
    engine, kmax = engine_for(name)
    path = osp.join(CACHE_DIR, name + ".pkl")
    if osp.exists(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if obj.get("version") == CACHE_VERSION and set(graphs_by_gid) <= set(obj["feats"]):
            return obj
    t0 = time.time()
    gids = sorted(graphs_by_gid)
    # balance chunks by an edge-count proxy
    cost = np.array([graphs_by_gid[i]["n"] + 5 * len(graphs_by_gid[i]["edges"]) for i in gids], float)
    order = np.argsort(-cost)
    n_chunks = max(1, min(len(gids), n_jobs * 8))
    chunks = [[] for _ in range(n_chunks)]
    load = np.zeros(n_chunks)
    for k in order:
        c = int(np.argmin(load)); chunks[c].append(graphs_by_gid[gids[k]]); load[c] += cost[k]
    if n_jobs > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_feat_chunk)(ch, engine, kmax) for ch in chunks if ch)
    else:
        res = [_feat_chunk(ch, engine, kmax) for ch in chunks if ch]
    feats = {}
    for r in res:
        for gid, f in r:
            feats[gid] = f
    obj = {"version": CACHE_VERSION, "dataset": name, "engine": engine, "kmax": kmax,
           "feats": feats, "seconds": time.time() - t0}
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    if verbose:
        print("[graphlets] %s: %d graphs, engine=%s kmax=%d, %.1fs" %
              (name, len(feats), engine, kmax, obj["seconds"]), flush=True)
    return obj
