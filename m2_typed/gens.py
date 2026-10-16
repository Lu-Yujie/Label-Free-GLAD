"""Synthetic graph generators for the GRAIL-0 synthetic validation (all deterministic given a seed).

Graph dicts follow the common.data format: {'n', 'edges' (m,2) int64 with i<j, unique, no loops,
'x' (n,d) float32 or None, 'node_label' (n,) int64 or None, 'gid'}; extra keys used only for
evaluation: 'gt_node' (n,) bool ground-truth anomalous/motif nodes, 'role' (n,) int64 generator role.

1. Syn-Com (MUSE, NeurIPS'24, App. C.1, verbatim): n nodes (even), communities
   Va = {0..n/2-1}, Vb = {n/2..n-1}; A_ij ~ Bernoulli((1+tau)/2) inside a community,
   Bernoulli((1-tau)/2) across; X = I_n in MUSE (we keep x=None: GRAIL-0 is structure-only here;
   the GAE baselines use X = I_n exactly as MUSE).

2. GLAD-X-lite (channel attribution; frozen before any result was seen):
   base = preferential-attachment tree (BA, m=1) with n_b ~ U{20..30} nodes; 10 motifs attached by
   one edge each to 10 distinct random base nodes: 2 stars (center + 4 leaves, attached via the
   center) and 8 ring motifs, each a house (p_house) or a bowtie (1-p_house); both have 5 nodes and
   6 edges, so n = n_b + 50 and m = n_b + 65 for every graph of every type (equal size).
   Generator roles: 0 base, 1 star center, 2 star leaf, 3 house node, 4 bowtie node, 5 K2,3 node.
   node_label: base 0, center 1, leaf 2, ring motif (house/bowtie/K2,3) 3; with prob. 0.05 the label
   is replaced by a uniform draw from {0,1,2,3}.
   x (d=2): center ~ N(+1, I), leaf ~ N(-1, I), all other nodes ~ N(0, I).
   Types: 'normal' (p_house=0.5);
     'T1a' structural novelty: 2 of the 8 ring motifs are K2,3 (5 nodes, 6 edges; never in normals);
     'T1b' structural rewiring: 4 random edges removed, 4 random non-edges added (n, m unchanged);
     'T2'  attribute-given-role: in each star, (x, label) of the center is swapped with a random
           leaf -> per-graph multiset of attributes and labels is exactly unchanged;
     'T3a' composition over-use: p_house = 0.9;  'T3b' composition under-use: p_house = 0.1.

3. BM-MT / BM-MN / BM-MS (SIGNET NeurIPS'23 Sec. 5 / App. F.1, "following [DIR, GSAT]").
   SIGNET never released its BM generator (datasets/__init__ imports only Mutag, MNIST75sp; the BM
   import is commented out), so we regenerate with DIR's synthetic_structsim builders
   (github.com/Wuyxin/DIR-GNN, spmotif_gen; re-implemented here, not imported):
   base = nx.balanced_tree(r in {2,3}, 2) | nx.ladder_graph(w in {4,5,6}) | nx.wheel_graph(w in 8..12),
   uniform over the three types (base sizes chosen to match SIGNET Table 4's ~14 nodes per graph);
   each motif attached by one edge from its first node to a distinct random base node (DIR
   build_graph with rdm_basis_plugins=True); DIR house edges (1,2),(4,3),(3,2),(4,1),(0,1),(0,4);
   cycle(L) = 0-1-...-(L-1)-0. GT nodes = motif nodes; GT edges = both endpoints in a motif (DIR find_gd).
     BM-MT: normal 1 house, anomaly 1 five-cycle.  BM-MN: normal 1-2 houses, anomaly 3-4 houses.
     BM-MS: normal one cycle with 3-5 nodes, anomaly one cycle with 6-9 nodes.
"""
import numpy as np
import networkx as nx


def _finish(n, edge_list, gid, x=None, node_label=None, role=None, gt_node=None):
    e = np.asarray(edge_list, dtype=np.int64).reshape(-1, 2)
    if len(e):
        e = np.sort(e, axis=1)
        e = e[e[:, 0] != e[:, 1]]
        e = np.unique(e, axis=0)
    g = {"n": int(n), "edges": e, "x": x, "node_label": node_label, "gid": int(gid)}
    if role is not None:
        g["role"] = np.asarray(role, dtype=np.int64)
    if gt_node is not None:
        g["gt_node"] = np.asarray(gt_node, dtype=bool)
    return g


def dense_adj(g):
    A = np.zeros((g["n"], g["n"]), dtype=np.float64)
    if len(g["edges"]):
        A[g["edges"][:, 0], g["edges"][:, 1]] = 1.0
        A[g["edges"][:, 1], g["edges"][:, 0]] = 1.0
    return A


# ----------------------------------------------------------------------------- 1. Syn-Com
def syncom_P(n, tau):
    h = n // 2
    P = np.full((n, n), (1.0 - tau) / 2.0)
    P[:h, :h] = (1.0 + tau) / 2.0
    P[h:, h:] = (1.0 + tau) / 2.0
    np.fill_diagonal(P, 0.0)
    return P


def syncom(n, tau, rng, gid=0):
    assert n % 2 == 0 and n > 2
    P = syncom_P(n, tau)
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < P[iu]
    edges = np.stack([iu[0][keep], iu[1][keep]], 1)
    return _finish(n, edges, gid)


def syncom_set(n, tau0, taua, seed, n_train=500, n_test_norm=250, n_test_anom=250):
    """Train (normal, tau0), test normals (tau0) and test anomalies (taua); fully deterministic."""
    ss = np.random.SeedSequence([int(seed), int(n), int(round(tau0 * 1000)), int(round(taua * 1000)), 1])
    rng = np.random.default_rng(ss)
    train = [syncom(n, tau0, rng, gid=i) for i in range(n_train)]
    test = [syncom(n, tau0, rng, gid=100000 + i) for i in range(n_test_norm)]
    test += [syncom(n, taua, rng, gid=200000 + i) for i in range(n_test_anom)]
    y = np.r_[np.zeros(n_test_norm, int), np.ones(n_test_anom, int)]
    return train, test, y


# ----------------------------------------------------------------------------- 2. GLAD-X-lite
GX_TYPES = ["T1a", "T1b", "T2", "T3a", "T3b"]
GX_LABEL_NOISE = 0.05
GX_D = 2
GX_N_STARS = 2
GX_N_RINGS = 8
GX_P_HOUSE = 0.5

# motif templates: (n_nodes, edges, attach node, generator role)
_STAR = (5, [(0, 1), (0, 2), (0, 3), (0, 4)], 0, None)                    # 0 = center
_HOUSE5 = (5, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 0), (4, 1)], 2, 3)    # square 0123 + roof 4
_BOWTIE = (5, [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)], 1, 4)    # center 0
_K23 = (5, [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)], 2, 5)


def gx_graph(rng, kind="normal", gid=0):
    assert kind in ["normal"] + GX_TYPES
    n_b = int(rng.integers(20, 31))
    base = nx.barabasi_albert_graph(n_b, 1, seed=int(rng.integers(2 ** 31 - 1)))
    edges = [tuple(e) for e in base.edges()]
    role = [0] * n_b
    p_house = {"T3a": 0.9, "T3b": 0.1}.get(kind, GX_P_HOUSE)
    rings = [_HOUSE5 if rng.random() < p_house else _BOWTIE for _ in range(GX_N_RINGS)]
    if kind == "T1a":
        for j in rng.choice(GX_N_RINGS, 2, replace=False):
            rings[j] = _K23
    motifs = [("star", _STAR)] * GX_N_STARS + [("ring", r) for r in rings]
    order = rng.permutation(len(motifs))
    motifs = [motifs[i] for i in order]
    plugins = rng.choice(n_b, len(motifs), replace=False)
    start = n_b
    stars = []
    for (kindm, (k, me, att, r)), pl in zip(motifs, plugins):
        edges += [(start + a, start + b) for a, b in me]
        edges.append((start + att, int(pl)))
        if kindm == "star":
            role += [1] + [2] * (k - 1)
            stars.append(start)
        else:
            role += [r] * k
        start += k
    n = start
    role = np.array(role, dtype=np.int64)
    lab = np.where(role == 0, 0, np.where(role == 1, 1, np.where(role == 2, 2, 3))).astype(np.int64)
    flip = rng.random(n) < GX_LABEL_NOISE
    lab[flip] = rng.integers(0, 4, int(flip.sum()))
    mu = np.zeros((n, GX_D))
    mu[role == 1] = 1.0
    mu[role == 2] = -1.0
    x = (mu + rng.standard_normal((n, GX_D))).astype(np.float32)
    gt = np.zeros(n, dtype=bool)
    if kind == "T1a":
        gt[role == 5] = True
    elif kind == "T1b":
        E = set(tuple(sorted(e)) for e in edges)
        El = sorted(E)
        rem = [El[i] for i in rng.choice(len(El), 4, replace=False)]
        for e in rem:
            E.discard(e)
        added = 0
        while added < 4:
            u, v = (int(t) for t in rng.integers(0, n, 2))
            e = (min(u, v), max(u, v))
            if u == v or e in E or e in rem:
                continue
            E.add(e); added += 1
            gt[[u, v]] = True
        for a, b in rem:
            gt[[a, b]] = True
        edges = sorted(E)
    elif kind == "T2":
        for c in stars:
            leaf = c + 1 + int(rng.integers(0, 4))
            x[[c, leaf]] = x[[leaf, c]]
            lab[[c, leaf]] = lab[[leaf, c]]
            gt[[c, leaf]] = True
    return _finish(n, edges, gid, x=x, node_label=lab, role=role, gt_node=gt)


def gx_set(seed, n_train=500, n_test_norm=250, n_test_anom=250):
    ss = np.random.SeedSequence([int(seed), 7, 2])
    rng = np.random.default_rng(ss)
    train = [gx_graph(rng, "normal", gid=i) for i in range(n_train)]
    test_norm = [gx_graph(rng, "normal", gid=100000 + i) for i in range(n_test_norm)]
    anom = {}
    for t_i, t in enumerate(GX_TYPES):
        anom[t] = [gx_graph(rng, t, gid=200000 + 10000 * t_i + i) for i in range(n_test_anom)]
    return train, test_norm, anom


# ----------------------------------------------------------------------------- 3. BM-MT/MN/MS
def _dir_house():
    return 5, [(1, 2), (4, 3), (3, 2), (4, 1), (0, 1), (0, 4)]


def _dir_cycle(L):
    return L, [(i, i + 1) for i in range(L - 1)] + [(L - 1, 0)]


def _bm_base(rng):
    t = int(rng.integers(0, 3))
    if t == 0:
        G = nx.balanced_tree(int(rng.integers(2, 4)), 2)
    elif t == 1:
        G = nx.ladder_graph(int(rng.integers(4, 7)))
    else:
        G = nx.wheel_graph(int(rng.integers(8, 13)))
    G = nx.convert_node_labels_to_integers(G)
    return G.number_of_nodes(), [tuple(e) for e in G.edges()], ["tree", "ladder", "wheel"][t]


def bm_graph(rng, dataset, anomalous, gid=0):
    n_b, edges, btype = _bm_base(rng)
    if dataset == "BM-MT":
        motifs = [_dir_cycle(5)] if anomalous else [_dir_house()]
    elif dataset == "BM-MN":
        k = int(rng.integers(3, 5)) if anomalous else int(rng.integers(1, 3))
        motifs = [_dir_house() for _ in range(k)]
    elif dataset == "BM-MS":
        L = int(rng.integers(6, 10)) if anomalous else int(rng.integers(3, 6))
        motifs = [_dir_cycle(L)]
    else:
        raise ValueError(dataset)
    plugins = rng.choice(n_b, len(motifs), replace=False)
    start = n_b
    role = [0] * n_b
    for (k, me), pl in zip(motifs, plugins):
        edges += [(start + a, start + b) for a, b in me]
        edges.append((start, int(pl)))
        role += [1] * k
        start += k
    role = np.array(role)
    g = _finish(start, edges, gid, role=role, gt_node=role > 0)
    g["base"] = btype
    return g


def bm_set(dataset, seed, n_train=500, n_test_norm=100, n_test_anom=100):
    ss = np.random.SeedSequence([int(seed), {"BM-MT": 1, "BM-MN": 2, "BM-MS": 3}[dataset], 3])
    rng = np.random.default_rng(ss)
    train = [bm_graph(rng, dataset, False, gid=i) for i in range(n_train)]
    test = [bm_graph(rng, dataset, False, gid=100000 + i) for i in range(n_test_norm)]
    test += [bm_graph(rng, dataset, True, gid=200000 + i) for i in range(n_test_anom)]
    y = np.r_[np.zeros(n_test_norm, int), np.ones(n_test_anom, int)]
    return train, test, y
