"""MS-v2: replacement molecular typed-anomaly generator (PREREG amendment A28; replaces the retired MolSwap v1).

Every rule is documented in prelim/evalpaper/typed/NOTES_MSv2.md (§3 written before this code existed; the final
design is selected there by a pre-declared rule from control scores only). Fully deterministic given the NCI1 fold
cache and the library versions.

    python molswap_v2.py generate [--out DIR] [--procs 4]   # write data + manifest.json (the frozen DESIGN)
    python molswap_v2.py validate [--out DIR] [--report PATH]
    python molswap_v2.py negcheck [--out DIR] [--report PATH] # validator negative controls
    python molswap_v2.py hashes   [--out DIR]

Loader (same interface as v1's load_ms):
    train, test, y_test = load_ms_v2(seed, "MS1", "matched")     # test = test normals + anomalies

Graph dicts follow prelim/common/data.py: n, edges (int64 (m,2), i<j, unique, lexsorted), x (None), node_label
(int64 (n,)), gid; anomalies carry evaluation-only extra keys (NOTES_MSv2.md). Node labels are used for one thing
only: the valence rule (every node keeps a (label, degree) pair that occurs in real NCI1 normals). They always stay
on their nodes.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import os.path as osp
import pickle
import platform
from collections import Counter

import networkx as nx
import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
PRELIM = osp.dirname(osp.dirname(HERE))
if PRELIM not in sys.path:
    sys.path.insert(0, PRELIM)
from common.data import load_folds  # noqa: E402

OUT_DIR = "/N/slate/yujlu/grail_data/molswap_v2"
MAGIC = 5067570                    # 0x4D5332 = "MS2"
SEEDS = [0, 1, 2, 3, 4]
FAMILY_CODE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
FAMILY_NAME = {"A": "ring contraction 6->5 + pendant (5-ring over-use)",
               "B": "ring expansion 5->6 (5-ring under-use)",
               "C": "ring contraction 6->4 + 2-chain (rare 4-ring)",
               "D": "ring relocation (open one ring, close a 4-6 ring in an acyclic part; v1 MS2 rule)",
               "E": "fused -> spiro ring junction (6-ring of an ortho-fused pair becomes a 5-ring sharing one atom)",
               "F": "ring contraction 6->3 + chain (rare 3-ring)"}
POOLS = ("P0", "P56")
VARIANTS = ["matched", "sizechg"]
VARIANT_CODE = {"matched": 0, "sizechg": 1}
MIN_YIELD = 100                    # A25: a (type, variant) with < 100 anomalies in any seed is infeasible
KCAP = 50                          # safety cap for k = "all"
PICKLE_PROTOCOL = 4
D_DIST = (3, 4, 5)
WL_ITERS = 3

# The frozen design: the mechanical result of the pre-declared selection rules (NOTES_MSv2.md §3.5 and §5).
# Each type: (name, family, k). The pool P0 is the v1 pool.
DESIGN = {"pool": "P0", "types": [("MS1", "C", 1), ("MS2", "D", 2), ("MS3", "F", 1)]}


# ============================================================================ graph primitives
def nx_graph(n, eset):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(sorted(eset))
    return G


def eset_of(edges):
    return set((int(a), int(b)) for a, b in edges)


def ekey(a, b):
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def edges_array(eset):
    if not eset:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(eset), dtype=np.int64).reshape(-1, 2)


_HIST_CACHE = {}
_BLOCK_CACHE = {}


def _block_lens(block):
    """MCB cycle lengths of one biconnected block (sorted tuple of its edges); cached per block."""
    r = _BLOCK_CACHE.get(block)
    if r is None:
        nodes = set(x for e in block for x in e)
        rank = len(block) - len(nodes) + 1
        if rank == 1:
            r = (len(block),)
        else:
            H = nx.Graph()
            H.add_edges_from(block)
            r = tuple(len(c) for c in nx.minimum_cycle_basis(H))
        if len(_BLOCK_CACHE) > 200000:
            _BLOCK_CACHE.clear()
        _BLOCK_CACHE[block] = r
    return r


def ring_hist(n, eset):
    """Sorted tuple of cycle lengths of a minimum cycle basis, computed per biconnected block (v1 rule; the cycle
    space is the direct sum over blocks). Blocks are cached individually (speed only; same values)."""
    key = (n, tuple(sorted(eset)))
    h = _HIST_CACHE.get(key)
    if h is None:
        G = nx_graph(n, eset)
        lens = []
        for block in nx.biconnected_component_edges(G):
            block = tuple(sorted(ekey(a, b) for a, b in block))
            if len(block) < 3:
                continue
            lens.extend(_block_lens(block))
        h = tuple(sorted(lens))
        if len(_HIST_CACHE) > 200000:
            _HIST_CACHE.clear()
        _HIST_CACHE[key] = h
    return h


def hist_replace(h, remove, add):
    lst = list(h)
    for r in remove:
        if r not in lst:
            return None
        lst.remove(r)
    return tuple(sorted(lst + list(add)))


def bridge_set(G):
    return set(ekey(a, b) for a, b in nx.bridges(G))


def ring_nodes(G, br):
    out = set()
    for a, b in G.edges():
        if ekey(a, b) not in br:
            out.add(a)
            out.add(b)
    return out


def canon_cycle(c):
    c = [int(x) for x in c]
    i = c.index(min(c))
    c = c[i:] + c[:i]
    if c[-1] < c[1]:
        c = [c[0]] + c[1:][::-1]
    return tuple(c)


def chordless_rings(G, size):
    return sorted(set(canon_cycle(c) for c in nx.chordless_cycles(G, length_bound=size) if len(c) == size))


def is_chordless_cycle(G, nodes):
    H = G.subgraph(nodes)
    return (H.number_of_nodes() == len(nodes) and H.number_of_edges() == len(nodes)
            and all(d == 2 for _, d in H.degree()) and nx.is_connected(H))


def is_simple_edges(n, edges):
    e = np.asarray(edges)
    if len(e) == 0:
        return True
    if e.shape[1] != 2 or (e[:, 0] >= e[:, 1]).any() or e.min() < 0 or e.max() >= n:
        return False
    return len(np.unique(e, axis=0)) == len(e)


def labelled_nx(n, eset, labels):
    G = nx_graph(n, eset)
    for i, l in enumerate(labels):
        G.nodes[i]["l"] = str(int(l))
    return G


def wl_hash(G):
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="l", iterations=WL_ITERS)


def label_isomorphic(G1, G2):
    return nx.is_isomorphic(G1, G2, node_match=lambda a, b: a["l"] == b["l"])


def degrees(n, eset):
    d = np.zeros(n, dtype=np.int64)
    for a, b in eset:
        d[a] += 1
        d[b] += 1
    return d


# ============================================================================ pool, split, reference sets
_P0 = None


def load_p0():
    """Connected normal-class NCI1 graphs (UB-GOLD convention), sorted by gid (the v1 pool, same code)."""
    global _P0
    if _P0 is not None:
        return _P0
    folds = load_folds("NCI1", features="ubgold")
    meta = folds[0]["meta"]
    assert meta["dataset"] == "NCI1" and meta["anomaly_class"] == 0 and meta["normal_classes"] == [1]
    assert meta["ubgold_uses_node_labels"] and meta["node_label_dim"] == 37
    f0 = folds[0]
    normals = {g["gid"]: g for g in f0["train"]}
    normals.update({g["gid"]: g for g, y in zip(f0["test"], f0["y_test"]) if int(y) == 0})
    union = set()
    for f in folds:
        union |= set(g["gid"] for g in f["train"])
        union |= set(g["gid"] for g, y in zip(f["test"], f["y_test"]) if int(y) == 0)
    assert union == set(normals), "fold-0 normals differ from the union over folds"
    pool, excluded = [], []
    for gid in sorted(normals):
        g = normals[gid]
        G = nx_graph(g["n"], eset_of(g["edges"]))
        if g["n"] >= 1 and nx.is_connected(G):
            pool.append({"n": int(g["n"]), "edges": np.array(g["edges"], dtype=np.int64, copy=True), "x": None,
                         "node_label": np.array(g["node_label"], dtype=np.int64, copy=True), "gid": int(gid)})
        else:
            excluded.append(int(gid))
    _P0 = (pool, excluded, len(normals))
    return _P0


def in_p56(g):
    h = set(ring_hist(g["n"], eset_of(g["edges"])))
    return 5 in h and 6 in h


def load_pool(name):
    pool, excluded, n_normal = load_p0()
    if name == "P0":
        return list(pool)
    if name == "P56":
        return [g for g in pool if in_p56(g)]
    raise ValueError(name)


def split_pool(pool, seed):
    N = len(pool)
    rng = np.random.default_rng(np.random.SeedSequence([MAGIC, int(seed), 0]))
    perm = rng.permutation(N)
    n_tr = int(round(0.6 * N))
    n_te = int(round(0.2 * N))
    tr = sorted(int(i) for i in perm[:n_tr])
    te = sorted(int(i) for i in perm[n_tr:n_tr + n_te])
    pa = sorted(int(i) for i in perm[n_tr + n_te:])
    return [pool[i] for i in tr], [pool[i] for i in te], [pool[i] for i in pa]


_REF = None


def reference():
    """VAL (set of (label, degree) on P0 nodes) and the WL(3) index of all P0 graphs (labels included)."""
    global _REF
    if _REF is None:
        pool, _, _ = load_p0()
        val = set()
        idx = {}
        for g in pool:
            d = degrees(g["n"], eset_of(g["edges"]))
            val |= set(zip(g["node_label"].tolist(), d.tolist()))
            G = labelled_nx(g["n"], eset_of(g["edges"]), g["node_label"])
            idx.setdefault(wl_hash(G), []).append(g["gid"])
        _REF = {"VAL": val, "WL": idx, "by_gid": {g["gid"]: g for g in pool}}
    return _REF


def valence_ok(labels, deg, nodes):
    val = reference()["VAL"]
    return all((int(labels[v]), int(deg[v])) in val for v in nodes)


def iso_to_p0(n, eset, labels):
    """gid of a P0 graph that is label-isomorphic to (n, eset, labels), else None."""
    R = reference()
    G = labelled_nx(n, eset, labels)
    for gid in R["WL"].get(wl_hash(G), []):
        P = R["by_gid"][gid]
        if P["n"] == n and len(P["edges"]) == len(eset) and \
                label_isomorphic(G, labelled_nx(P["n"], eset_of(P["edges"]), P["node_label"])):
            return gid
    return None


# ============================================================================ one edit step per family
def _state(n, E):
    G = nx_graph(n, E)
    br = bridge_set(G)
    return {"n": n, "E": E, "G": G, "br": br, "rn": ring_nodes(G, br), "hist": ring_hist(n, E),
            "deg": dict(G.degree())}


def _swap(st, labels, rem, add):
    """Common step acceptance: simple, connected, valence. Returns (E2, G2) or None."""
    E = st["E"]
    if rem not in E or add in E or add[0] == add[1]:
        return None
    E2 = (E - {rem}) | {add}
    G2 = nx_graph(st["n"], E2)
    if not nx.is_connected(G2):
        return None
    deg2 = dict(G2.degree())
    if not valence_ok(labels, deg2, set(rem) | set(add)):
        return None
    return E2, G2


def step_A(st, labels, rng):
    G, deg = st["G"], st["deg"]
    cands = []
    for R in chordless_rings(G, 6):
        for i in range(6):
            v = R[i]
            if deg[v] != 2:
                continue
            prev, nxt = R[i - 1], R[(i + 1) % 6]
            cands.append((R, v, prev, nxt))
            cands.append((R, v, nxt, prev))
    target = hist_replace(st["hist"], [6], [5])
    if not cands or target is None:
        return None
    for k in rng.permutation(len(cands)):
        R, v, p, q = cands[int(k)]
        r = _swap(st, labels, ekey(v, q), ekey(p, q))
        if r is None:
            continue
        E2, G2 = r
        if ring_hist(st["n"], E2) != target:
            continue
        if G2.degree(v) != 1 or not is_chordless_cycle(G2, [x for x in R if x != v]):
            continue
        return {"E": E2, "rem": ekey(v, q), "add": ekey(p, q), "gt": list(R)}
    return None


def step_B(st, labels, rng):
    G, rn = st["G"], st["rn"]
    cands = []
    for R in chordless_rings(G, 5):
        for i in range(5):
            p = R[i]
            for q in sorted((R[i - 1], R[(i + 1) % 5])):
                for v in sorted(G.neighbors(p)):
                    if v not in rn and v not in R:
                        cands.append((R, p, q, v))
    target = hist_replace(st["hist"], [5], [6])
    if not cands or target is None:
        return None
    for k in rng.permutation(len(cands)):
        R, p, q, v = cands[int(k)]
        r = _swap(st, labels, ekey(p, q), ekey(v, q))
        if r is None:
            continue
        E2, G2 = r
        if ring_hist(st["n"], E2) != target:
            continue
        if not is_chordless_cycle(G2, list(R) + [v]):
            continue
        return {"E": E2, "rem": ekey(p, q), "add": ekey(v, q), "gt": list(R) + [v]}
    return None


def step_C(st, labels, rng):
    G, deg = st["G"], st["deg"]
    cands = []
    for R in chordless_rings(G, 6):
        for i in range(6):
            b, c = R[(i + 1) % 6], R[(i + 2) % 6]
            if deg[b] == 2 and deg[c] == 2:
                cands.append((R, i, 0))
                cands.append((R, i, 1))
    target = hist_replace(st["hist"], [6], [4])
    if not cands or target is None:
        return None
    for k in rng.permutation(len(cands)):
        R, i, dr = cands[int(k)]
        a, b, c, d, e, f = (R[(i + j) % 6] for j in range(6))
        rem = ekey(c, d) if dr == 0 else ekey(a, b)
        r = _swap(st, labels, rem, ekey(a, d))
        if r is None:
            continue
        E2, G2 = r
        if ring_hist(st["n"], E2) != target:
            continue
        if not is_chordless_cycle(G2, [a, d, e, f]):
            continue
        return {"E": E2, "rem": rem, "add": ekey(a, d), "gt": list(R)}
    return None


def _d_deletions(st):
    n, E, br, h = st["n"], st["E"], st["br"], st["hist"]
    out = []
    for e in sorted(E):
        if e in br:
            continue
        h2 = ring_hist(n, E - {e})
        if len(h2) != len(h) - 1:
            continue
        diff = Counter(h) - Counter(h2)
        if sum(diff.values()) == 1 and not (Counter(h2) - Counter(h)):
            out.append((e, next(iter(diff))))
    return out


def step_D(st, labels, rng):
    G, rn = st["G"], st["rn"]
    acyc = sorted(v for v in G.nodes() if v not in rn and G.degree(v) > 0)
    F = G.subgraph(acyc)
    pairs = []
    for u in acyc:
        dist = nx.single_source_shortest_path_length(F, u, cutoff=max(D_DIST))
        for w, d in dist.items():
            if w > u and d in D_DIST:
                pairs.append((u, w, d))
    pairs.sort()
    if not pairs:
        return None
    dels = _d_deletions(st)
    if not dels:
        return None
    pair_order = rng.permutation(len(pairs))
    del_order = rng.permutation(len(dels))
    for pk in pair_order:
        u, w, d = pairs[int(pk)]
        path = nx.shortest_path(F, u, w)
        for dk in del_order:
            e, s = dels[int(dk)]
            r = _swap(st, labels, e, ekey(u, w))
            if r is None:
                continue
            E2, G2 = r
            if ring_hist(st["n"], E2) != hist_replace(st["hist"], [s], [d + 1]):
                continue
            if not is_chordless_cycle(G2, path):
                continue
            rn2 = ring_nodes(G2, bridge_set(G2))
            if not all(x in rn2 for x in path):
                continue
            Ge = G.copy()
            Ge.remove_edge(*e)
            opened = nx.shortest_path(Ge, e[0], e[1])
            return {"E": E2, "rem": e, "add": ekey(u, w), "gt": list(opened) + list(path)}
    return None


def step_E(st, labels, rng):
    """Fused -> spiro junction (NOTES_MSv2.md §5)."""
    G, deg = st["G"], st["deg"]
    r6 = chordless_rings(G, 6)
    rk_all = chordless_rings(G, 5) + r6
    cands = []
    for Rs in r6:
        es = set(ekey(Rs[i], Rs[(i + 1) % 6]) for i in range(6))
        for Rk in rk_all:
            if Rk == Rs:
                continue
            shared = set(Rs) & set(Rk)
            if len(shared) != 2:
                continue
            c0, c1 = sorted(shared)
            if ekey(c0, c1) not in es or deg[c0] != 3 or deg[c1] != 3:
                continue
            for c, o in ((c0, c1), (c1, c0)):
                i = Rs.index(o)
                nb = [Rs[i - 1], Rs[(i + 1) % 6]]
                a1 = [x for x in nb if x != c][0]
                cands.append((Rs, Rk, c, o, a1))
    target = hist_replace(st["hist"], [6], [5])
    if not cands or target is None:
        return None
    for k in rng.permutation(len(cands)):
        Rs, Rk, c, o, a1 = cands[int(k)]
        r = _swap(st, labels, ekey(o, a1), ekey(c, a1))
        if r is None:
            continue
        E2, G2 = r
        if ring_hist(st["n"], E2) != target:
            continue
        r5 = [x for x in Rs if x != o]
        if not is_chordless_cycle(G2, r5) or not is_chordless_cycle(G2, list(Rk)):
            continue
        if set(r5) & set(Rk) != {c} or G2.degree(c) != 4:
            continue
        return {"E": E2, "rem": ekey(o, a1), "add": ekey(c, a1), "gt": sorted(set(Rs) | set(Rk))}
    return None


def step_F(st, labels, rng):
    """6-ring -> 3-ring + chain (NOTES_MSv2.md §5)."""
    G, deg = st["G"], st["deg"]
    cands = []
    for R in chordless_rings(G, 6):
        for i in range(6):
            if deg[R[(i + 1) % 6]] == 2:
                cands.append((R, i, 0))
                cands.append((R, i, 1))
    target = hist_replace(st["hist"], [6], [3])
    if not cands or target is None:
        return None
    for k in rng.permutation(len(cands)):
        R, i, dr = cands[int(k)]
        a, b, c = R[i], R[(i + 1) % 6], R[(i + 2) % 6]
        rem = ekey(c, R[(i + 3) % 6]) if dr == 0 else ekey(a, R[i - 1])
        r = _swap(st, labels, rem, ekey(a, c))
        if r is None:
            continue
        E2, G2 = r
        if ring_hist(st["n"], E2) != target:
            continue
        if not is_chordless_cycle(G2, [a, b, c]):
            continue
        return {"E": E2, "rem": rem, "add": ekey(a, c), "gt": list(R)}
    return None


STEP = {"A": step_A, "B": step_B, "C": step_C, "D": step_D, "E": step_E, "F": step_F}


def edit_sequence(P, family, seed, kmax):
    """Sequential accepted steps (up to kmax) of `family` on parent P. Step j uses its own RNG, independent of kmax."""
    n, E = P["n"], eset_of(P["edges"])
    labels = P["node_label"]
    seq = []
    for j in range(kmax):
        st = _state(n, E)
        rng = np.random.default_rng(np.random.SeedSequence([MAGIC, int(seed), FAMILY_CODE[family], int(P["gid"]), j]))
        res = STEP[family](st, labels, rng)
        if res is None:
            break
        E = res["E"]
        seq.append(res)
    return seq


def finalize(P, seq, k):
    """Size-matched anomaly from the first min(k, len(seq)) steps, or (None, reason)."""
    if not seq:
        return None, "no_step_accepted"
    kk = len(seq) if k == "all" else min(int(k), len(seq))
    steps = seq[:kk]
    n = P["n"]
    E = steps[-1]["E"]
    lab = P["node_label"]
    if label_isomorphic(labelled_nx(n, E, lab), labelled_nx(n, eset_of(P["edges"]), lab)):
        return None, "isomorphic_to_parent"
    if iso_to_p0(n, E, lab) is not None:
        return None, "isomorphic_to_a_normal"
    gt = np.zeros(n, dtype=bool)
    for s in steps:
        gt[s["gt"]] = True
    return {"E": E, "gt": gt, "removed": [s["rem"] for s in steps], "added": [s["add"] for s in steps],
            "n_steps": kk}, None


def attach_donor(P, res, parents, seed, family):
    """Size-changing variant: the size-matched anomaly plus one unmodified donor molecule joined by one edge."""
    n = P["n"]
    lab = P["node_label"]
    lo = -(-n // 5)
    hi = max(lo, n)
    elig = [D for D in parents if D["gid"] != P["gid"] and lo <= D["n"] <= hi]
    if not elig:
        return None, "no_donor_in_size_window"
    rng = np.random.default_rng(np.random.SeedSequence([MAGIC, int(seed), 10 + FAMILY_CODE[family], int(P["gid"])]))
    val = reference()["VAL"]
    deg = degrees(n, res["E"])
    hosts = [i for i in range(n) if not res["gt"][i] and deg[i] <= 2 and (int(lab[i]), int(deg[i]) + 1) in val]
    if not hosts:
        return None, "no_host_node"
    for di in rng.permutation(len(elig)):
        D = elig[int(di)]
        nD = D["n"]
        degD = degrees(nD, eset_of(D["edges"]))
        tails = [j for j in range(nD) if degD[j] <= 2 and (int(D["node_label"][j]), int(degD[j]) + 1) in val]
        if not tails:
            continue
        h = hosts[int(rng.integers(len(hosts)))]
        t = tails[int(rng.integers(len(tails)))]
        E3 = set(res["E"]) | set((int(a) + n, int(b) + n) for a, b in D["edges"]) | {(h, n + t)}
        lab3 = np.concatenate([np.asarray(lab, dtype=np.int64), np.asarray(D["node_label"], dtype=np.int64)])
        n3 = n + nD
        if iso_to_p0(n3, E3, lab3) is not None:
            continue
        return {"n": n3, "E": E3, "labels": lab3, "donor_gid": int(D["gid"]), "attach": (h, n + t)}, None
    return None, "no_donor_attachment"


# ============================================================================ generation
def anomaly_gid(slot, variant, parent_gid):
    return 1_000_000 * int(slot) + 100_000 * VARIANT_CODE[variant] + int(parent_gid)


def _normal_copy(g):
    return {"n": int(g["n"]), "edges": np.array(g["edges"], dtype=np.int64, copy=True), "x": None,
            "node_label": np.array(g["node_label"], dtype=np.int64, copy=True), "gid": int(g["gid"])}


def make_cell(parents, seed, family, k, slot, name):
    """Anomalies of one (type, seed): (matched list, sizechg list, skip counters, null-edit parents)."""
    matched, sizechg, used = [], [], []
    sk_m, sk_s = Counter(), Counter()
    kmax = KCAP if k == "all" else int(k)
    for P in parents:
        seq = edit_sequence(P, family, seed, kmax)
        res, why = finalize(P, seq, k)
        if res is None:
            sk_m[why] += 1
            sk_s["matched_infeasible"] += 1
            continue
        n = P["n"]
        lab = np.array(P["node_label"], dtype=np.int64, copy=True)
        gm = {"n": n, "edges": edges_array(res["E"]), "x": None, "node_label": lab,
              "gid": anomaly_gid(slot, "matched", P["gid"]), "gt_node": res["gt"].copy(),
              "parent_gid": int(P["gid"]), "ms_type": name, "family": family, "variant": "matched",
              "n_edits": int(res["n_steps"]), "edit_removed": np.asarray(res["removed"], dtype=np.int64).reshape(-1, 2),
              "edit_added": np.asarray(res["added"], dtype=np.int64).reshape(-1, 2)}
        matched.append(gm)
        used.append(P)
        sc, why = attach_donor(P, res, parents, seed, family)
        if sc is None:
            sk_s[why] += 1
            continue
        gt3 = np.zeros(sc["n"], dtype=bool)
        gt3[:n] = res["gt"]
        frag = np.zeros(sc["n"], dtype=bool)
        frag[n:] = True
        sizechg.append({"n": int(sc["n"]), "edges": edges_array(sc["E"]), "x": None, "node_label": sc["labels"],
                        "gid": anomaly_gid(slot, "sizechg", P["gid"]), "gt_node": gt3, "parent_gid": int(P["gid"]),
                        "ms_type": name, "family": family, "variant": "sizechg", "n_edits": int(res["n_steps"]),
                        "edit_removed": gm["edit_removed"].copy(), "edit_added": gm["edit_added"].copy(),
                        "donor_gid": sc["donor_gid"], "frag_node": frag,
                        "attach_edge": edges_array({ekey(*sc["attach"])})})
    return matched, sizechg, sk_m, sk_s, used


def design_types(design):
    return [t[0] for t in design["types"]]


def file_names(design):
    names = ["train", "test_normal"]
    for t in design_types(design):
        for v in VARIANTS:
            names.append(f"anom_{t}_{v}")
    return names


def generate_seed(design, seed):
    pool = load_pool(design["pool"])
    train, test_norm, parents = split_pool(pool, seed)
    out = {"train": [_normal_copy(g) for g in train], "test_normal": [_normal_copy(g) for g in test_norm]}
    skips = {}
    for slot, (name, family, k) in enumerate(design["types"], start=1):
        m, s, sk_m, sk_s, _ = make_cell(parents, seed, family, k, slot, name)
        out[f"anom_{name}_matched"] = m
        out[f"anom_{name}_sizechg"] = s
        skips[name] = {"matched": dict(sorted(sk_m.items())), "sizechg": dict(sorted(sk_s.items()))}
    split = {"train": [g["gid"] for g in train], "test_normal": [g["gid"] for g in test_norm],
             "parents": [g["gid"] for g in parents]}
    return out, split, skips


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _size_summary(graphs):
    if not graphs:
        return {"n_mean": None, "m_mean": None, "n_min": None, "n_max": None}
    ns = np.array([g["n"] for g in graphs], dtype=np.float64)
    ms = np.array([len(g["edges"]) for g in graphs], dtype=np.float64)
    return {"n_mean": round(float(ns.mean()), 3), "m_mean": round(float(ms.mean()), 3),
            "n_min": int(ns.min()), "n_max": int(ns.max())}


def generate(out_dir, design=None, procs=1):
    design = design or DESIGN
    assert design is not None, "DESIGN is not set"
    pool = load_pool(design["pool"])
    p0, excluded, n_normal = load_p0()
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "generator": "MS-v2 (PREREG_AMENDMENTS A28); rules in prelim/evalpaper/typed/NOTES_MSv2.md",
        "code": "prelim/evalpaper/typed/molswap_v2.py",
        "code_sha256": sha256_file(osp.abspath(__file__)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "networkx": nx.__version__,
                     "pickle_protocol": PICKLE_PROTOCOL},
        "source": "NCI1 fold cache /N/slate/yujlu/grail_data/folds/NCI1.pkl via prelim/common/data.py "
                  "load_folds('NCI1', features='ubgold'); normal class = class 1 (raw 1), UB-GOLD convention",
        "node_label_dim": 37, "x": None,
        "design": {"pool": design["pool"],
                   "types": [{"name": t, "family": f, "family_description": FAMILY_NAME[f], "k": k}
                             for t, f, k in design["types"]]},
        "pool": {"n_normal_class": n_normal, "n_p0_connected": len(p0), "n_excluded_disconnected": len(excluded),
                 "name": design["pool"], "n_pool": len(pool), "pool_gids": [g["gid"] for g in pool]},
        "valence_set": sorted([list(x) for x in reference()["VAL"]]),
        "min_yield": MIN_YIELD, "types": design_types(design), "variants": VARIANTS,
        "file_format": "pickle protocol 4: list of graph dicts (prelim/common format + eval-only keys)",
        "seeds": {}, "status": {},
    }
    counts = {t: {v: [] for v in VARIANTS} for t in design_types(design)}
    reference()
    if procs > 1:   # per-seed work is independent and deterministic; files are written here in seed order
        from multiprocessing import get_context
        with get_context("fork").Pool(min(procs, len(SEEDS))) as pp:
            per_seed = pp.starmap(generate_seed, [(design, s) for s in SEEDS])
    else:
        per_seed = [generate_seed(design, s) for s in SEEDS]
    for s, (data, split, skips) in zip(SEEDS, per_seed):
        sd = osp.join(out_dir, f"seed{s}")
        os.makedirs(sd, exist_ok=True)
        files = {}
        for name in file_names(design):
            path = osp.join(sd, name + ".pkl")
            with open(path, "wb") as f:
                pickle.dump(data[name], f, protocol=PICKLE_PROTOCOL)
            files[f"seed{s}/{name}.pkl"] = {"sha256": sha256_file(path), "bytes": osp.getsize(path),
                                            "n_graphs": len(data[name]), **_size_summary(data[name])}
        cnt = {name: len(data[name]) for name in file_names(design)}
        for t in design_types(design):
            for v in VARIANTS:
                counts[t][v].append(cnt[f"anom_{t}_{v}"])
        edits = {t: dict(sorted(Counter(int(g["n_edits"]) for g in data[f"anom_{t}_matched"]).items()))
                 for t in design_types(design)}
        manifest["seeds"][str(s)] = {"split": split, "counts": cnt, "skips": skips, "n_edits_hist": edits,
                                     "files": files}
        print(f"seed {s}: " + ", ".join(f"{k}={v}" for k, v in cnt.items()), flush=True)
    for t in design_types(design):
        manifest["status"][t] = {}
        for v in VARIANTS:
            c = counts[t][v]
            manifest["status"][t][v] = {"counts_per_seed": c, "min": int(min(c)),
                                        "status": "FEASIBLE" if min(c) >= MIN_YIELD else "INFEASIBLE"}
    mpath = osp.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=False)
        f.write("\n")
    return manifest


# ============================================================================ loader
def load_ms_v2(seed, ms_type, variant, root=OUT_DIR, allow_infeasible=False):
    """(train normals, test = test normals + anomalies, y_test) for one MS-v2 (type, variant, seed).
    Same interface as v1's load_ms. A (type, variant) with < 100 anomalies in some seed (A25) raises unless
    allow_infeasible=True."""
    with open(osp.join(root, "manifest.json")) as f:
        man = json.load(f)
    assert ms_type in man["types"] and variant in VARIANTS, (ms_type, variant)
    st = man["status"][ms_type][variant]["status"]
    if st != "FEASIBLE" and not allow_infeasible:
        raise ValueError(f"{ms_type}/{variant} is {st} (fewer than {MIN_YIELD} anomalies in a seed; A25)")
    sd = osp.join(root, f"seed{int(seed)}")
    with open(osp.join(sd, "train.pkl"), "rb") as f:
        train = pickle.load(f)
    with open(osp.join(sd, "test_normal.pkl"), "rb") as f:
        tn = pickle.load(f)
    with open(osp.join(sd, f"anom_{ms_type}_{variant}.pkl"), "rb") as f:
        an = pickle.load(f)
    y = np.r_[np.zeros(len(tn), dtype=np.int64), np.ones(len(an), dtype=np.int64)]
    return train, tn + an, y


# ============================================================================ validation (no detector)
def _blocks_state(n, E):
    G = nx_graph(n, E)
    br = bridge_set(G)
    return G, br, ring_nodes(G, br), ring_hist(n, E)


def check_step(family, n, E0, E1, labels):
    """Independent re-check of one edge swap E0 -> E1 against the family signature (NOTES_MSv2.md §3.2, §5).
    Written separately from the generator's step functions: it only sees the two edge sets."""
    rem, add = sorted(E0 - E1), sorted(E1 - E0)
    if len(rem) != 1 or len(add) != 1:
        return False
    rem, add = rem[0], add[0]
    G0, br0, rn0, h0 = _blocks_state(n, E0)
    G1, br1, rn1, h1 = _blocks_state(n, E1)
    if not nx.is_connected(G1):
        return False
    d1 = dict(G1.degree())
    if not valence_ok(labels, d1, set(rem) | set(add)):
        return False
    x, y = add
    if family == "C":
        if h1 != hist_replace(h0, [6], [4]):
            return False
        # a chordless 6-cycle of G0 containing both endpoints of the removed edge and x, y at ring distance 3
        for R in chordless_rings(G0, 6):
            if x not in R or y not in R or rem[0] not in R or rem[1] not in R:
                continue
            i, j = R.index(x), R.index(y)
            if (i - j) % 6 != 3:
                continue
            four = None
            for cand in ([R[(i + t) % 6] for t in (0, 3, 4, 5)], [R[(i + t) % 6] for t in (0, 3, 2, 1)]):
                if is_chordless_cycle(G1, cand):
                    four = cand
            if four is None:
                continue
            chain = [v for v in R if v not in four]
            if (len(chain) == 2 and all(v not in rn1 for v in chain) and G1.has_edge(*chain)
                    and all(G0.degree(v) == 2 for v in chain)
                    and len(set(rem) & set(chain)) == 1 and len(set(rem) & {x, y}) == 1):
                return True
        return False
    if family == "F":
        if h1 != hist_replace(h0, [6], [3]):
            return False
        common = set(G1.neighbors(x)) & set(G1.neighbors(y))
        for R in chordless_rings(G0, 6):
            if x not in R or y not in R or rem[0] not in R or rem[1] not in R:
                continue
            i, j = R.index(x), R.index(y)
            if (i - j) % 6 not in (2, 4):
                continue
            b = R[(i + 1) % 6] if (j - i) % 6 == 2 else R[(i - 1) % 6]
            if b not in common or G0.degree(b) != 2:
                continue
            # the removed edge is the ring edge leaving the a-b-c segment at a or at c
            Redges = set(ekey(R[t], R[(t + 1) % 6]) for t in range(6))
            if rem in Redges and len(set(rem) & {x, y}) == 1 and b not in rem and is_chordless_cycle(G1, [x, y, b]):
                return True
        return False
    if family == "D":
        e = rem
        if e in br0:
            return False
        h0e = ring_hist(n, E0 - {e})
        if len(h0e) != len(h0) - 1 or (Counter(h0e) - Counter(h0)):
            return False
        acyc0 = set(v for v in G0.nodes() if v not in rn0 and G0.degree(v) > 0)
        F0 = G0.subgraph(acyc0)
        if x not in acyc0 or y not in acyc0 or not nx.has_path(F0, x, y):
            return False
        path = nx.shortest_path(F0, x, y)
        d = len(path) - 1
        if d not in D_DIST or nx.shortest_path_length(G0, x, y) != d:
            return False
        if not is_chordless_cycle(G1, path) or not all(v in rn1 for v in path):
            return False
        return h1 == tuple(sorted(h0e + (d + 1,))) and rn0 != rn1
    raise ValueError(family)


def check_anomaly(g, P, family, k):
    """Replay the recorded swaps from the parent; every step must pass check_step; the result must equal g."""
    n = P["n"]
    E = eset_of(P["edges"])
    rems = [ekey(*r) for r in np.asarray(g["edit_removed"]).reshape(-1, 2)]
    adds = [ekey(*a) for a in np.asarray(g["edit_added"]).reshape(-1, 2)]
    if len(rems) != len(adds) or not (1 <= len(rems) <= k) or len(rems) != int(g["n_edits"]):
        return False
    for r, a in zip(rems, adds):
        if r not in E or a in E:
            return False
        E1 = (E - {r}) | {a}
        if not check_step(family, n, E, E1, P["node_label"]):
            return False
        if not (g["gt_node"][list(r)].all() and g["gt_node"][list(a)].all()):
            return False
        E = E1
    return E == eset_of(g["edges"])


def check_sizechg(g, gm, by_gid, parent_set):
    n = gm["n"]
    D = by_gid.get(g["donor_gid"])
    if D is None or g["donor_gid"] == g["parent_gid"] or g["donor_gid"] not in parent_set:
        return False
    if g["n"] != n + D["n"] or not (5 * D["n"] >= n and D["n"] <= max(-(-n // 5), n)):
        return False
    E = eset_of(g["edges"])
    core = set((a, b) for a, b in E if a < n and b < n)
    frag = set((a - n, b - n) for a, b in E if a >= n and b >= n)
    cross = [(a, b) for a, b in E if (a < n) != (b < n)]
    deg = degrees(g["n"], E)
    return (core == eset_of(gm["edges"]) and frag == eset_of(D["edges"]) and len(cross) == 1
            and np.array_equal(g["node_label"][:n], gm["node_label"])
            and np.array_equal(g["node_label"][n:], D["node_label"])
            and not gm["gt_node"][cross[0][0]] and np.array_equal(g["gt_node"][:n], gm["gt_node"])
            and bool(g["frag_node"][n:].all()) and not bool(g["frag_node"][:n].any())
            and valence_ok(g["node_label"], deg, cross[0]))


def validate(out_dir, design=None):
    design = design or DESIGN
    p0, _, _ = load_p0()
    by_gid = {g["gid"]: g for g in p0}
    with open(osp.join(out_dir, "manifest.json")) as f:
        man = json.load(f)
    assert man["design"]["pool"] == design["pool"]
    pool_gids = set(man["pool"]["pool_gids"])
    assert pool_gids == set(g["gid"] for g in load_pool(design["pool"]))
    types = {t: (f, k) for t, f, k in design["types"]}
    rep = {"checks": {}, "per_seed": {}, "failures": []}
    fails = rep["failures"]

    def fail(msg):
        if len(fails) < 200:
            fails.append(msg)

    def ok_struct(g):
        G = nx_graph(g["n"], eset_of(g["edges"]))
        return (is_simple_edges(g["n"], g["edges"]) and g["edges"].dtype == np.int64
                and g["node_label"].dtype == np.int64 and len(g["node_label"]) == g["n"]
                and g["x"] is None and nx.is_connected(G))

    tot = Counter()
    for s in SEEDS:
        sd = osp.join(out_dir, f"seed{s}")
        data = {}
        for name in file_names(design):
            path = osp.join(sd, name + ".pkl")
            if sha256_file(path) != man["seeds"][str(s)]["files"][f"seed{s}/{name}.pkl"]["sha256"]:
                fail(f"seed{s} {name}: sha256 differs from manifest")
            with open(path, "rb") as f:
                data[name] = pickle.load(f)
        tr = [g["gid"] for g in data["train"]]
        te = [g["gid"] for g in data["test_normal"]]
        pa = man["seeds"][str(s)]["split"]["parents"]
        disjoint = (len(set(tr) | set(te) | set(pa)) == len(tr) + len(te) + len(pa)
                    and set(tr) | set(te) | set(pa) == pool_gids)
        tot["split_ok"] += int(disjoint)
        if not disjoint:
            fail(f"seed{s}: split not a disjoint cover of the pool")
        exp_tr, exp_te, exp_pa = split_pool(load_pool(design["pool"]), s)
        if [g["gid"] for g in exp_tr] != tr or [g["gid"] for g in exp_te] != te or [g["gid"] for g in exp_pa] != pa:
            fail(f"seed{s}: split differs from split_pool")
        for g in data["train"] + data["test_normal"]:
            P = by_gid[g["gid"]]
            if not (np.array_equal(g["edges"], P["edges"]) and np.array_equal(g["node_label"], P["node_label"])
                    and g["n"] == P["n"] and ok_struct(g)):
                fail(f"seed{s}: normal {g['gid']} differs from the cache graph")
                tot["normal_bad"] += 1
        wl_train = set(wl_hash(labelled_nx(g["n"], eset_of(g["edges"]), g["node_label"])) for g in data["train"])
        wl_testn = set(wl_hash(labelled_nx(g["n"], eset_of(g["edges"]), g["node_label"])) for g in data["test_normal"])
        ps = {"wl_dup_with_train": {}, "wl_dup_with_test_normal": {}, "n_edits": {}, "new_ring_sizes_D": {}}
        par_set = set(pa)
        for t, (family, k) in types.items():
            matched = {g["parent_gid"]: g for g in data[f"anom_{t}_matched"]}
            for v in VARIANTS:
                key = f"anom_{t}_{v}"
                gl = data[key]
                gids = [g["gid"] for g in gl]
                if len(set(gids)) != len(gids):
                    fail(f"seed{s} {key}: duplicate gids")
                if len(gl) != man["seeds"][str(s)]["counts"][key]:
                    fail(f"seed{s} {key}: count differs from manifest")
                if len(set(g["parent_gid"] for g in gl)) != len(gl):
                    fail(f"seed{s} {key}: a parent used twice")
                wl_dup = wl_dup_te = 0
                for g in gl:
                    tot[f"{t}_{v}_total"] += 1
                    P = by_gid[g["parent_gid"]]
                    good = (ok_struct(g) and g["parent_gid"] in par_set and g["ms_type"] == t and g["variant"] == v
                            and g["family"] == family)
                    tot[f"{t}_{v}_simple_connected_parent_ok"] += int(good)
                    if not good:
                        fail(f"seed{s} {key} {g['gid']}: not simple/connected or bad parent/type")
                    Gl = labelled_nx(g["n"], eset_of(g["edges"]), g["node_label"])
                    hg = wl_hash(Gl)
                    wl_dup += int(hg in wl_train)
                    wl_dup_te += int(hg in wl_testn)
                    no_iso = iso_to_p0(g["n"], eset_of(g["edges"]), g["node_label"]) is None
                    tot[f"{t}_{v}_not_isomorphic_to_any_P0_graph"] += int(no_iso)
                    deg = degrees(g["n"], eset_of(g["edges"]))
                    ok_val = valence_ok(g["node_label"], deg, range(g["n"]))
                    tot[f"{t}_{v}_valence_ok"] += int(ok_val)
                    if v == "matched":
                        ok_nm = g["n"] == P["n"] and len(g["edges"]) == len(P["edges"])
                        ok_lab = np.array_equal(g["node_label"], P["node_label"])
                        ok_iso = not label_isomorphic(Gl, labelled_nx(P["n"], eset_of(P["edges"]), P["node_label"]))
                        ok_sig = check_anomaly(g, P, family, k)
                        tot[f"{t}_{v}_nm_equal"] += int(ok_nm)
                        tot[f"{t}_{v}_labels_identical"] += int(ok_lab)
                        tot[f"{t}_{v}_not_isomorphic_to_parent"] += int(ok_iso)
                        tot[f"{t}_{v}_signature_ok"] += int(ok_sig)
                        ps["n_edits"].setdefault(t, Counter())[int(g["n_edits"])] += 1
                        if family == "D":
                            E0 = eset_of(P["edges"])
                            h0, h1 = Counter(ring_hist(P["n"], E0)), Counter(ring_hist(g["n"], eset_of(g["edges"])))
                            for sz, c in (h1 - h0).items():
                                ps["new_ring_sizes_D"][sz] = ps["new_ring_sizes_D"].get(sz, 0) + c
                        if not (ok_nm and ok_lab and ok_iso and ok_sig and no_iso and ok_val and good):
                            fail(f"seed{s} {key} {g['gid']}: nm={ok_nm} lab={ok_lab} iso={ok_iso} sig={ok_sig} "
                                 f"p0iso={no_iso} val={ok_val}")
                    else:
                        n0 = P["n"]
                        ok_dn = 5 * (g["n"] - n0) >= n0 and g["n"] - n0 <= n0
                        gm = matched.get(g["parent_gid"])
                        ok_core = gm is not None and check_sizechg(g, gm, by_gid, par_set)
                        tot[f"{t}_{v}_dn_20_to_100pct"] += int(ok_dn)
                        tot[f"{t}_{v}_core_eq_matched_donor_intact_one_bridge"] += int(ok_core)
                        if not (ok_dn and ok_core and no_iso and ok_val and good):
                            fail(f"seed{s} {key} {g['gid']}: dn={ok_dn} core={ok_core} p0iso={no_iso} val={ok_val}")
                ps["wl_dup_with_train"][key] = wl_dup
                ps["wl_dup_with_test_normal"][key] = wl_dup_te
        ps["n_edits"] = {t: dict(sorted(c.items())) for t, c in ps["n_edits"].items()}
        ps["new_ring_sizes_D"] = dict(sorted(ps["new_ring_sizes_D"].items()))
        rep["per_seed"][str(s)] = ps
    rep["checks"] = dict(sorted(tot.items()))
    rep["n_failures"] = len(fails)
    return rep


def negative_controls(out_dir, design=None, n_random=400):
    """Validator negative controls (no detector): each type's checker on the other types' anomalies, and on random
    single edge swaps of pool graphs (seed-0 parents)."""
    design = design or DESIGN
    p0, _, _ = load_p0()
    by_gid = {g["gid"]: g for g in p0}
    types = {t: (f, k) for t, f, k in design["types"]}
    an = {}
    for t in types:
        with open(osp.join(out_dir, "seed0", f"anom_{t}_matched.pkl"), "rb") as f:
            an[t] = pickle.load(f)
    out = {"cross_type": {}, "random_swaps": {}}
    for t, (fam, k) in types.items():
        for t2 in types:
            if t2 == t:
                continue
            acc = 0
            for g in an[t2]:
                P = by_gid[g["parent_gid"]]
                gg = dict(g)
                acc += int(check_anomaly(gg, P, fam, max(k, len(np.asarray(g["edit_added"]).reshape(-1, 2)))))
            out["cross_type"][f"{t}_checker_on_{t2}"] = f"{acc}/{len(an[t2])}"
    rng = np.random.default_rng(np.random.SeedSequence([MAGIC, 999]))
    pool = load_pool(design["pool"])
    swaps = []
    while len(swaps) < n_random:
        P = pool[int(rng.integers(len(pool)))]
        n, E = P["n"], eset_of(P["edges"])
        if n < 4 or not E:
            continue
        el = sorted(E)
        r = el[int(rng.integers(len(el)))]
        a = ekey(*rng.choice(n, 2, replace=False))
        if a in E:
            continue
        E1 = (E - {r}) | {a}
        if not nx.is_connected(nx_graph(n, E1)):
            continue
        swaps.append((P, E, E1))
    for t, (fam, k) in types.items():
        out["random_swaps"][f"{t}_{fam}"] = f"{sum(check_step(fam, P['n'], E, E1, P['node_label']) for P, E, E1 in swaps)}/{len(swaps)}"
    return out


# ============================================================================ CLI
def hashes(out_dir):
    out = {}
    for root, _, fs in os.walk(out_dir):
        for fn in sorted(fs):
            p = osp.join(root, fn)
            out[osp.relpath(p, out_dir)] = sha256_file(p)
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "validate", "negcheck", "hashes"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--report", default=None)
    ap.add_argument("--procs", type=int, default=1, help="processes for generate (one per seed; <= 4 on a login node)")
    a = ap.parse_args()
    if a.cmd == "generate":
        man = generate(a.out, procs=a.procs)
        print(json.dumps(man["status"], indent=1))
    elif a.cmd in ("validate", "negcheck"):
        rep = validate(a.out) if a.cmd == "validate" else negative_controls(a.out)
        txt = json.dumps(rep, indent=1, default=str)
        if a.report:
            with open(a.report, "w") as f:
                f.write(txt + "\n")
        print(txt)
    else:
        for k, v in hashes(a.out).items():
            print(f"{v}  {k}")


if __name__ == "__main__":
    main()
