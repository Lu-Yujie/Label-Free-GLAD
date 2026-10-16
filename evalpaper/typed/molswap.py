"""MolSwap (MS) typed-anomaly generator, PREREG amendment A9 section 2 "MS".

Every rule is documented in prelim/evalpaper/typed/NOTES.md (written before this code was first run).
The output is fully deterministic given the fold cache and library versions.

    python molswap.py generate [--out DIR]          # write data + manifest.json
    python molswap.py validate [--out DIR] [--report PATH]
    python molswap.py hashes   [--out DIR]          # sha256 of every file under DIR

Loader for downstream runs:
    train, test, y_test = load_ms(seed, "MS1", "matched")     # test = test normals + anomalies

Graph dicts follow prelim/common/data.py: n, edges (int64 (m,2), i<j, unique, lexsorted), x (None),
node_label (int64 (n,)), gid; anomalies carry evaluation-only extra keys (see NOTES.md section 6).
Node labels are never used to choose anything.
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

OUT_DIR = "/N/slate/yujlu/grail_data/molswap"
MAGIC = 19795                      # 0x4D53 = "MS"
SEEDS = [0, 1, 2, 3, 4]
TYPES = ["MS1", "MS2", "MS3"]
TYPE_CODE = {"MS1": 1, "MS2": 2, "MS3": 3}
VARIANTS = ["matched", "sizechg"]
VARIANT_CODE = {"matched": 0, "sizechg": 1}
MIN_YIELD = 150                    # A9: a type yielding < 150 anomalies per seed is infeasible
PICKLE_PROTOCOL = 4
MS2_DIST = (3, 4, 5)
WL_ITERS = 3


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


def ring_hist(n, eset):
    """Sorted tuple of cycle lengths of a minimum cycle basis (unique as a multiset).

    Computed per biconnected block (the cycle space is the direct sum over blocks): a block of cycle
    rank 1 is a single ring whose length is its edge count; fused blocks use
    networkx.minimum_cycle_basis on that block only.
    """
    key = (n, tuple(sorted(eset)))
    h = _HIST_CACHE.get(key)
    if h is None:
        G = nx_graph(n, eset)
        lens = []
        for block in nx.biconnected_component_edges(G):
            block = sorted(ekey(a, b) for a, b in block)
            if len(block) < 3:                      # a bridge
                continue
            nodes = set(x for e in block for x in e)
            rank = len(block) - len(nodes) + 1
            if rank == 1:
                lens.append(len(block))
            else:
                H = nx.Graph()
                H.add_edges_from(block)
                lens.extend(len(c) for c in nx.minimum_cycle_basis(H))
        h = tuple(sorted(lens))
        _HIST_CACHE[key] = h
    return h


def ring_hist_reference(n, eset):
    """Direct networkx.minimum_cycle_basis on the whole graph (slow; used only to test ring_hist)."""
    return tuple(sorted(len(c) for c in nx.minimum_cycle_basis(nx_graph(n, eset))))


def hist_replace(h, remove, add):
    """Multiset h minus the elements of `remove` plus those of `add`; None if `remove` not contained."""
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


def is_chordless_cycle(G, nodes):
    """True iff `nodes` induce a single cycle (connected, every node degree 2 inside)."""
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


def labelled_nx(g):
    G = nx_graph(g["n"], eset_of(g["edges"]))
    for i, l in enumerate(g["node_label"]):
        G.nodes[i]["l"] = str(int(l))
    return G


def wl_hash(G):
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="l", iterations=WL_ITERS)


def label_isomorphic(G1, G2):
    if wl_hash(G1) != wl_hash(G2):
        return False
    return nx.is_isomorphic(G1, G2, node_match=lambda a, b: a["l"] == b["l"])


# ============================================================================ pool and split
def load_pool():
    """Connected normal-class NCI1 graphs (UB-GOLD convention), sorted by gid; plus excluded gids."""
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
            pool.append({"n": int(g["n"]),
                         "edges": np.array(g["edges"], dtype=np.int64, copy=True),
                         "x": None,
                         "node_label": np.array(g["node_label"], dtype=np.int64, copy=True),
                         "gid": int(gid)})
        else:
            excluded.append(int(gid))
    return pool, excluded, len(normals)


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


# ============================================================================ per-parent static info
_INFO = {}


def parent_info(P):
    gid = P["gid"]
    info = _INFO.get(gid)
    if info is not None:
        return info
    n = P["n"]
    E = eset_of(P["edges"])
    G = nx_graph(n, E)
    br = bridge_set(G)
    rn = ring_nodes(G, br)
    info = {"n": n, "E": E, "G": G, "br": br, "rn": rn, "hist": ring_hist(n, E),
            "deg": dict(G.degree())}
    _INFO[gid] = info
    return info


def ms1_candidates(info):
    G, deg = info["G"], info["deg"]
    rings = set()
    for c in nx.chordless_cycles(G, length_bound=6):
        if len(c) == 6:
            rings.add(canon_cycle(c))
    cands = []
    for R in sorted(rings):
        for i in range(6):
            v = R[i]
            if deg[v] != 2:
                continue
            prev, nxt = R[i - 1], R[(i + 1) % 6]
            cands.append((R, v, prev, nxt))      # keep (v,prev); remove (v,nxt); add (prev,nxt)
            cands.append((R, v, nxt, prev))      # keep (v,nxt);  remove (v,prev); add (nxt,prev)
    return sorted(cands)


def ms2_pairs(info):
    G, rn = info["G"], info["rn"]
    acyc = sorted(v for v in G.nodes() if v not in rn and G.degree(v) > 0)
    F = G.subgraph(acyc)
    pairs = []
    for u in acyc:
        dist = nx.single_source_shortest_path_length(F, u, cutoff=max(MS2_DIST))
        for v, d in dist.items():
            if v > u and d in MS2_DIST:
                pairs.append((u, v, d))
    return sorted(pairs), F


def ms2_deletions(info):
    """Ring edges whose removal opens exactly one ring: hist(G-e) = hist(G) minus one element s."""
    if "ms2_del" in info:
        return info["ms2_del"]
    n, E, br, h = info["n"], info["E"], info["br"], info["hist"]
    out = []
    for e in sorted(E):
        if e in br:
            continue
        h2 = ring_hist(n, E - {e})
        if len(h2) != len(h) - 1:
            continue
        diff = Counter(h) - Counter(h2)
        if sum(diff.values()) == 1 and not (Counter(h2) - Counter(h)):
            s = next(iter(diff))
            out.append((e, s))
    info["ms2_del"] = out
    return out


def ms3_candidates(info):
    G, br, deg = info["G"], info["br"], info["deg"]
    cands = []
    for (b, c) in sorted(info["E"]):
        if deg[b] != 2 or deg[c] != 2:
            continue
        a = [x for x in G.neighbors(b) if x != c][0]
        d = [x for x in G.neighbors(c) if x != b][0]
        if a == d:
            continue
        if ekey(a, b) in br and ekey(b, c) in br and ekey(c, d) in br:
            cands.append((b, c, a, d, b))      # centre b
            cands.append((b, c, a, d, c))      # centre c
    return sorted(cands)


# ============================================================================ the three edits
def _common_ok(P, info, E2):
    n = info["n"]
    if len(E2) != len(info["E"]):
        return False
    if any(a == b for a, b in E2):
        return False
    G2 = nx_graph(n, E2)
    if not nx.is_connected(G2):
        return False
    g_new = {"n": n, "edges": edges_array(E2), "node_label": P["node_label"]}
    if label_isomorphic(labelled_nx(P), labelled_nx(g_new)):
        return False
    return True


def make_ms1(P, info, rng):
    cands = ms1_candidates(info)
    if not cands:
        return None, "no_6ring_with_deg2_node"
    n, E = info["n"], info["E"]
    target = hist_replace(info["hist"], [6], [5])
    for k in rng.permutation(len(cands)):
        R, v, p, q = cands[int(k)]
        rem, add = ekey(v, q), ekey(p, q)
        if rem not in E or add in E:
            continue
        E2 = (E - {rem}) | {add}
        if target is None or ring_hist(n, E2) != target:
            continue
        G2 = nx_graph(n, E2)
        if G2.degree(v) != 1 or not is_chordless_cycle(G2, [x for x in R if x != v]):
            continue
        if not _common_ok(P, info, E2):
            continue
        gt = np.zeros(n, dtype=bool)
        gt[list(R)] = True
        return {"E2": E2, "removed": [rem], "added": [add], "gt": gt}, None
    return None, "no_candidate_accepted"


def make_ms2(P, info, rng):
    pairs, F = ms2_pairs(info)
    if not pairs:
        return None, "no_acyclic_pair_at_dist_3_5"
    dels = ms2_deletions(info)
    if not dels:
        return None, "no_ring_edge_opening_exactly_one_ring"
    n, E, G = info["n"], info["E"], info["G"]
    pair_order = rng.permutation(len(pairs))
    del_order = rng.permutation(len(dels))
    for pk in pair_order:
        u, v, d = pairs[int(pk)]
        path = nx.shortest_path(F, u, v)
        assert len(path) == d + 1
        for dk in del_order:
            e, s = dels[int(dk)]
            add = ekey(u, v)
            if add in E:
                continue
            E2 = (E - {e}) | {add}
            if ring_hist(n, E2) != hist_replace(info["hist"], [s], [d + 1]):
                continue
            G2 = nx_graph(n, E2)
            if not is_chordless_cycle(G2, path):
                continue
            rn2 = ring_nodes(G2, bridge_set(G2))
            if not all(x in rn2 for x in path):
                continue
            if not _common_ok(P, info, E2):
                continue
            Ge = G.copy()
            Ge.remove_edge(*e)
            opened = nx.shortest_path(Ge, e[0], e[1])
            gt = np.zeros(n, dtype=bool)
            gt[list(opened)] = True
            gt[list(path)] = True
            return {"E2": E2, "removed": [e], "added": [add], "gt": gt}, None
    return None, "no_candidate_accepted"


def make_ms3(P, info, rng):
    cands = ms3_candidates(info)
    if not cands:
        return None, "no_linear_chain_segment"
    n, E = info["n"], info["E"]
    deg_hist = Counter(info["deg"].values())
    for k in rng.permutation(len(cands)):
        b, c, a, d, centre = cands[int(k)]
        if centre == b:
            rem, add = ekey(c, d), ekey(b, d)
        else:
            rem, add = ekey(a, b), ekey(a, c)
        if rem not in E or add in E:
            continue
        E2 = (E - {rem}) | {add}
        if ring_hist(n, E2) != info["hist"]:
            continue
        G2 = nx_graph(n, E2)
        if ring_nodes(G2, bridge_set(G2)) != info["rn"]:
            continue
        dh2 = Counter(dict(G2.degree()).values())
        want = Counter(deg_hist)
        want[2] -= 2
        want[1] += 1
        want[3] += 1
        if +dh2 != +want:
            continue
        H = G2.subgraph([a, b, c, d])
        if H.number_of_edges() != 3 or H.degree(centre) != 3:
            continue
        if not _common_ok(P, info, E2):
            continue
        gt = np.zeros(n, dtype=bool)
        gt[[a, b, c, d]] = True
        return {"E2": E2, "removed": [rem], "added": [add], "gt": gt}, None
    return None, "no_candidate_accepted"


MAKERS = {"MS1": make_ms1, "MS2": make_ms2, "MS3": make_ms3}


def attach_donor(Pgid, n, E2, labels, gt, parents, rng):
    """Size-changing variant: attach an unmodified donor molecule (anomaly-parent set) by one edge."""
    lo = -(-n // 5)                 # ceil(0.2 n), exact integer arithmetic
    hi = max(lo, n // 2)            # floor(0.5 n)
    elig = [D for D in parents if D["gid"] != Pgid and lo <= D["n"] <= hi]
    if not elig:
        return None, "no_donor_in_size_window"
    D = elig[int(rng.integers(len(elig)))]
    deg = np.zeros(n, dtype=np.int64)
    for a, b in E2:
        deg[a] += 1
        deg[b] += 1
    hosts = [i for i in range(n) if not gt[i] and deg[i] <= 2]
    if not hosts:
        return None, "no_host_node"
    h = hosts[int(rng.integers(len(hosts)))]
    nD = D["n"]
    degD = np.zeros(nD, dtype=np.int64)
    for a, b in D["edges"]:
        degD[a] += 1
        degD[b] += 1
    tails = [j for j in range(nD) if degD[j] <= 2]
    if not tails:
        return None, "no_donor_node"
    t = tails[int(rng.integers(len(tails)))]
    E3 = set(E2) | set((int(a) + n, int(b) + n) for a, b in D["edges"]) | {(h, n + t)}
    n3 = n + nD
    lab3 = np.concatenate([np.asarray(labels, dtype=np.int64), np.asarray(D["node_label"], dtype=np.int64)])
    return {"n": n3, "E3": E3, "labels": lab3, "donor_gid": int(D["gid"]), "attach": (h, n + t)}, None


# ============================================================================ generation
def anomaly_gid(ms_type, variant, parent_gid):
    return 1_000_000 * TYPE_CODE[ms_type] + 100_000 * VARIANT_CODE[variant] + int(parent_gid)


def _normal_copy(g):
    return {"n": int(g["n"]), "edges": np.array(g["edges"], dtype=np.int64, copy=True), "x": None,
            "node_label": np.array(g["node_label"], dtype=np.int64, copy=True), "gid": int(g["gid"])}


def generate_seed(pool, seed):
    train, test_norm, parents = split_pool(pool, seed)
    out = {"train": [_normal_copy(g) for g in train], "test_normal": [_normal_copy(g) for g in test_norm]}
    skips = {}
    for t in TYPES:
        code = TYPE_CODE[t]
        matched, sizechg = [], []
        sk_m, sk_s = Counter(), Counter()
        for P in parents:
            info = parent_info(P)
            rng = np.random.default_rng(np.random.SeedSequence([MAGIC, int(seed), code, P["gid"]]))
            res, why = MAKERS[t](P, info, rng)
            if res is None:
                sk_m[why] += 1
                sk_s["matched_infeasible"] += 1
                continue
            n = info["n"]
            lab = np.array(P["node_label"], dtype=np.int64, copy=True)
            gm = {"n": n, "edges": edges_array(res["E2"]), "x": None, "node_label": lab,
                  "gid": anomaly_gid(t, "matched", P["gid"]),
                  "gt_node": res["gt"].copy(), "parent_gid": int(P["gid"]), "ms_type": t,
                  "variant": "matched",
                  "edit_removed": edges_array(set(res["removed"])),
                  "edit_added": edges_array(set(res["added"]))}
            matched.append(gm)
            rng_d = np.random.default_rng(np.random.SeedSequence([MAGIC, int(seed), 10 + code, P["gid"]]))
            sc, why = attach_donor(P["gid"], n, res["E2"], lab, res["gt"], parents, rng_d)
            if sc is None:
                sk_s[why] += 1
                continue
            gt3 = np.zeros(sc["n"], dtype=bool)
            gt3[:n] = res["gt"]
            frag = np.zeros(sc["n"], dtype=bool)
            frag[n:] = True
            gs = {"n": int(sc["n"]), "edges": edges_array(sc["E3"]), "x": None, "node_label": sc["labels"],
                  "gid": anomaly_gid(t, "sizechg", P["gid"]),
                  "gt_node": gt3, "parent_gid": int(P["gid"]), "ms_type": t, "variant": "sizechg",
                  "edit_removed": edges_array(set(res["removed"])),
                  "edit_added": edges_array(set(res["added"])),
                  "donor_gid": sc["donor_gid"], "frag_node": frag,
                  "attach_edge": edges_array({ekey(*sc["attach"])})}
            sizechg.append(gs)
        out[f"anom_{t}_matched"] = matched
        out[f"anom_{t}_sizechg"] = sizechg
        skips[t] = {"matched": dict(sorted(sk_m.items())), "sizechg": dict(sorted(sk_s.items()))}
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


def file_names():
    names = ["train", "test_normal"]
    for t in TYPES:
        for v in VARIANTS:
            names.append(f"anom_{t}_{v}")
    return names


def generate(out_dir):
    pool, excluded, n_normal = load_pool()
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "generator": "MolSwap (MS), PREREG_AMENDMENTS A9 section 2; rules in prelim/evalpaper/typed/NOTES.md",
        "code": "prelim/evalpaper/typed/molswap.py",
        "code_sha256": sha256_file(osp.abspath(__file__)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "networkx": nx.__version__, "pickle_protocol": PICKLE_PROTOCOL},
        "source": "NCI1 fold cache /N/slate/yujlu/grail_data/folds/NCI1.pkl via prelim/common/data.py "
                  "load_folds('NCI1', features='ubgold'); normal class = class 1 (raw 1), UB-GOLD convention",
        "node_label_dim": 37, "x": None,
        "pool": {"n_normal_class": n_normal, "n_pool_connected": len(pool),
                 "n_excluded_disconnected": len(excluded), "excluded_gids": excluded,
                 "pool_gids": [g["gid"] for g in pool]},
        "min_yield": MIN_YIELD, "types": TYPES, "variants": VARIANTS,
        "file_format": "pickle protocol 4: list of graph dicts (prelim/common format + eval-only keys)",
        "seeds": {}, "status": {},
    }
    counts = {t: {v: [] for v in VARIANTS} for t in TYPES}
    for s in SEEDS:
        sd = osp.join(out_dir, f"seed{s}")
        os.makedirs(sd, exist_ok=True)
        data, split, skips = generate_seed(pool, s)
        files = {}
        for name in file_names():
            path = osp.join(sd, name + ".pkl")
            with open(path, "wb") as f:
                pickle.dump(data[name], f, protocol=PICKLE_PROTOCOL)
            files[f"seed{s}/{name}.pkl"] = {"sha256": sha256_file(path), "bytes": osp.getsize(path),
                                            "n_graphs": len(data[name]), **_size_summary(data[name])}
        cnt = {name: len(data[name]) for name in file_names()}
        for t in TYPES:
            for v in VARIANTS:
                counts[t][v].append(cnt[f"anom_{t}_{v}"])
        manifest["seeds"][str(s)] = {"split": split, "counts": cnt, "skips": skips, "files": files}
        print(f"seed {s}: " + ", ".join(f"{k}={v}" for k, v in cnt.items()), flush=True)
    for t in TYPES:
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
def load_ms(seed, ms_type, variant, root=OUT_DIR, allow_infeasible=False):
    """(train normals, test = test normals + anomalies, y_test) for one MS (type, variant, seed).

    A (type, variant) marked INFEASIBLE (A9: < 150 anomalies in some seed) raises unless
    allow_infeasible=True, which is meant only for a use that a timestamped amendment authorises.
    """
    assert ms_type in TYPES and variant in VARIANTS
    with open(osp.join(root, "manifest.json")) as f:
        st = json.load(f)["status"][ms_type][variant]["status"]
    if st != "FEASIBLE" and not allow_infeasible:
        raise ValueError(f"{ms_type}/{variant} is {st} per A9 (fewer than {MIN_YIELD} anomalies in a seed)")
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
def validate(out_dir):
    pool, _, _ = load_pool()
    by_gid = {g["gid"]: g for g in pool}
    with open(osp.join(out_dir, "manifest.json")) as f:
        man = json.load(f)
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
    pool_gids = set(by_gid)
    for s in SEEDS:
        sd = osp.join(out_dir, f"seed{s}")
        data = {}
        for name in file_names():
            path = osp.join(sd, name + ".pkl")
            assert sha256_file(path) == man["seeds"][str(s)]["files"][f"seed{s}/{name}.pkl"]["sha256"]
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
        for g in data["train"] + data["test_normal"]:
            P = by_gid[g["gid"]]
            if not (np.array_equal(g["edges"], P["edges"]) and np.array_equal(g["node_label"], P["node_label"])
                    and g["n"] == P["n"] and ok_struct(g)):
                fail(f"seed{s}: normal {g['gid']} differs from the cache graph")
                tot["normal_bad"] += 1
        wl_train = set(wl_hash(labelled_nx(g)) for g in data["train"])
        wl_testn = set(wl_hash(labelled_nx(g)) for g in data["test_normal"])
        ps = {"wl_dup_with_train": {}, "wl_dup_with_test_normal": {}, "ms2_opened_vs_new_ring_size": {}}
        par_set = set(pa)
        for t in TYPES:
            matched = {g["parent_gid"]: g for g in data[f"anom_{t}_matched"]}
            for v in VARIANTS:
                key = f"anom_{t}_{v}"
                gl = data[key]
                gids = [g["gid"] for g in gl]
                if len(set(gids)) != len(gids):
                    fail(f"seed{s} {key}: duplicate gids")
                if len(gl) != man["seeds"][str(s)]["counts"][key]:
                    fail(f"seed{s} {key}: count differs from manifest")
                wl_dup = 0
                wl_dup_te = 0
                for g in gl:
                    if v == "matched" and t == "MS2":
                        P = by_gid[g["parent_gid"]]
                        e = tuple(int(x) for x in g["edit_removed"][0])
                        E0 = eset_of(P["edges"])
                        s_open = Counter(ring_hist(P["n"], E0)) - Counter(ring_hist(P["n"], E0 - {e}))
                        u, w = (int(x) for x in g["edit_added"][0])
                        G0 = nx_graph(P["n"], E0)
                        k = f"{next(iter(s_open))}->{nx.shortest_path_length(G0, u, w) + 1}"
                        dd = ps["ms2_opened_vs_new_ring_size"]
                        dd[k] = dd.get(k, 0) + 1
                    tot[f"{t}_{v}_total"] += 1
                    P = by_gid[g["parent_gid"]]
                    good = ok_struct(g) and g["parent_gid"] in par_set and g["ms_type"] == t and g["variant"] == v
                    if not good:
                        fail(f"seed{s} {key} {g['gid']}: not simple/connected or bad parent")
                    else:
                        tot[f"{t}_{v}_simple_connected"] += 1
                    hg = wl_hash(labelled_nx(g))
                    wl_dup += int(hg in wl_train)
                    wl_dup_te += int(hg in wl_testn)
                    if v == "matched":
                        ok_nm = g["n"] == P["n"] and len(g["edges"]) == len(P["edges"])
                        ok_lab = np.array_equal(g["node_label"], P["node_label"])
                        ok_iso = not label_isomorphic(labelled_nx(g), labelled_nx(P))
                        ok_sig = _check_signature(t, P, g)
                        tot[f"{t}_{v}_nm_equal"] += int(ok_nm)
                        tot[f"{t}_{v}_labels_identical"] += int(ok_lab)
                        tot[f"{t}_{v}_not_isomorphic_to_parent"] += int(ok_iso)
                        tot[f"{t}_{v}_signature_ok"] += int(ok_sig)
                        if not (ok_nm and ok_lab and ok_iso and ok_sig):
                            fail(f"seed{s} {key} {g['gid']}: nm={ok_nm} lab={ok_lab} iso={ok_iso} sig={ok_sig}")
                    else:
                        n0 = P["n"]
                        ok_dn = 5 * (g["n"] - n0) >= n0
                        gm = matched.get(g["parent_gid"])
                        ok_core = gm is not None and _check_sizechg(g, gm, by_gid)
                        tot[f"{t}_{v}_dn_ge_20pct"] += int(ok_dn)
                        tot[f"{t}_{v}_core_eq_matched_and_donor_intact"] += int(ok_core)
                        if not (ok_dn and ok_core):
                            fail(f"seed{s} {key} {g['gid']}: dn={ok_dn} core={ok_core}")
                ps["wl_dup_with_train"][key] = wl_dup
                ps["wl_dup_with_test_normal"][key] = wl_dup_te
        ps["ms2_opened_vs_new_ring_size"] = dict(sorted(ps["ms2_opened_vs_new_ring_size"].items()))
        rep["per_seed"][str(s)] = ps
    rep["checks"] = dict(sorted(tot.items()))
    rep["n_failures"] = len(fails)
    return rep


def _check_signature(t, P, g):
    """Recompute the intended structural change of type t from parent P and anomaly g only."""
    n = P["n"]
    E0, E1 = eset_of(P["edges"]), eset_of(g["edges"])
    rem, add = sorted(E0 - E1), sorted(E1 - E0)
    if len(rem) != 1 or len(add) != 1:
        return False
    G0, G1 = nx_graph(n, E0), nx_graph(n, E1)
    h0, h1 = ring_hist(n, E0), ring_hist(n, E1)
    br0, br1 = bridge_set(G0), bridge_set(G1)
    rn0, rn1 = ring_nodes(G0, br0), ring_nodes(G1, br1)
    (r0, r1), (a0, a1) = rem[0], add[0]
    if t == "MS1":
        if h1 != hist_replace(h0, [6], [5]):
            return False
        shared = {r0, r1} & {a0, a1}
        if len(shared) != 1:
            return False
        q = shared.pop()
        v = r0 if r1 == q else r1
        p = a0 if a1 == q else a1
        if not (G0.degree(v) == 2 and G1.degree(v) == 1 and G1.has_edge(v, p)):
            return False
        ring6 = sorted(np.flatnonzero(g["gt_node"]).tolist())   # claimed parent 6-ring
        if len(ring6) != 6 or not {v, p, q} <= set(ring6) or not is_chordless_cycle(G0, ring6):
            return False
        ring5 = [x for x in ring6 if x != v]
        return is_chordless_cycle(G1, ring5)
    if t == "MS2":
        e = ekey(r0, r1)
        if e in br0:                                  # deleted edge must be a ring edge
            return False
        h0e = ring_hist(n, E0 - {e})                  # must open exactly one ring (size s)
        if len(h0e) != len(h0) - 1 or (Counter(h0e) - Counter(h0)):
            return False
        acyc0 = set(G0.nodes()) - rn0
        F0 = G0.subgraph(acyc0)
        if a0 not in acyc0 or a1 not in acyc0 or not nx.has_path(F0, a0, a1):
            return False
        path = nx.shortest_path(F0, a0, a1)
        d = len(path) - 1
        if d not in MS2_DIST or nx.shortest_path_length(G0, a0, a1) != d:
            return False
        if not is_chordless_cycle(G1, path) or not all(x in rn1 for x in path):
            return False
        return h1 == tuple(sorted(h0e + (d + 1,))) and rn0 != rn1
    if t == "MS3":
        if h1 != h0 or rn0 != rn1:
            return False
        dh0, dh1 = Counter(dict(G0.degree()).values()), Counter(dict(G1.degree()).values())
        want = Counter(dh0)
        want[2] -= 2
        want[1] += 1
        want[3] += 1
        if +dh1 != +want:
            return False
        nodes = sorted(np.flatnonzero(g["gt_node"]).tolist())
        if len(nodes) != 4 or not {r0, r1, a0, a1} <= set(nodes):
            return False
        H0, H1 = G0.subgraph(nodes), G1.subgraph(nodes)
        is_p4 = (H0.number_of_edges() == 3 and sorted(d for _, d in H0.degree()) == [1, 1, 2, 2]
                 and nx.is_connected(H0) and all(ekey(x, y) in br0 for x, y in H0.edges()))
        inner = [x for x, dd in H0.degree() if dd == 2]
        is_star = H1.number_of_edges() == 3 and max(d for _, d in H1.degree()) == 3
        return is_p4 and is_star and all(G0.degree(x) == 2 for x in inner)
    return False


def _check_sizechg(g, gm, by_gid):
    n = gm["n"]
    D = by_gid.get(g["donor_gid"])
    if D is None or g["n"] != n + D["n"] or g["donor_gid"] == g["parent_gid"]:
        return False
    E = eset_of(g["edges"])
    core = set((a, b) for a, b in E if a < n and b < n)
    frag = set((a - n, b - n) for a, b in E if a >= n and b >= n)
    cross = [(a, b) for a, b in E if (a < n) != (b < n)]
    return (core == eset_of(gm["edges"]) and frag == eset_of(D["edges"]) and len(cross) == 1
            and np.array_equal(g["node_label"][:n], gm["node_label"])
            and np.array_equal(g["node_label"][n:], D["node_label"])
            and not gm["gt_node"][cross[0][0]] and np.array_equal(g["gt_node"][:n], gm["gt_node"]))


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
    ap.add_argument("cmd", choices=["generate", "validate", "hashes"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    if a.cmd == "generate":
        man = generate(a.out)
        print(json.dumps(man["status"], indent=1))
    elif a.cmd == "validate":
        rep = validate(a.out)
        txt = json.dumps(rep, indent=1)
        if a.report:
            with open(a.report, "w") as f:
                f.write(txt + "\n")
        print(txt)
    else:
        for k, v in hashes(a.out).items():
            print(f"{v}  {k}")


if __name__ == "__main__":
    main()
