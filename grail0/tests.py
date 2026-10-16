"""Unit tests for the GRAIL-0 graphlet engine and model.

Run: cd <repo> && PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -m grail0.tests
"""
import itertools
import sys
import time

import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from grail0 import graphlets as GL


def _edges(G):
    e = np.array([(min(u, v), max(u, v)) for u, v in G.edges() if u != v], dtype=np.int64)
    return e.reshape(-1, 2)


def _gdict(G):
    G = nx.convert_node_labels_to_integers(G)
    return {"n": G.number_of_nodes(), "edges": _edges(G), "x": None, "node_label": None, "gid": 0}


def brute_force(G, kmax):
    """Reference: itertools over all k-subsets + networkx isomorphism to the named graphlets
    + automorphism orbits via VF2. Independent of the ESU code path and of the bit tables."""
    T = GL.tables()
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    canon = []
    for g in T["graphlets"]:
        H = nx.Graph(); H.add_nodes_from(range(g["size"])); H.add_edges_from(g["edges"])
        canon.append(H)
    counts = np.zeros(T["n_types"][kmax], dtype=np.int64)
    gdv = np.zeros((n, T["n_orbits"][kmax]), dtype=np.int64)
    gdv[:, 0] = [G.degree(v) for v in range(n)]
    for k in range(3, kmax + 1):
        for S in itertools.combinations(range(n), k):
            H = G.subgraph(S)
            if not nx.is_connected(H):
                continue
            for gid, C in enumerate(canon):
                if C.number_of_nodes() != k or C.number_of_edges() != H.number_of_edges():
                    continue
                gm = GraphMatcher(H, C)
                if gm.is_isomorphic():
                    counts[gid] += 1
                    iso = gm.mapping  # H node -> C node
                    cm = T["graphlets"][gid]["canon_mask"]
                    for v in S:
                        gdv[v, T["tab_orb"][k, cm, iso[v]]] += 1
                    break
            else:
                raise AssertionError("no graphlet matched")
    return counts, gdv


def check(cond, msg):
    if not cond:
        print("FAIL:", msg); sys.exit(1)


def test_tables():
    T = GL.tables()
    check(len(T["graphlets"]) == 29, "29 graphlets")
    check(len(T["orbit_names"]) == 73, "73 orbits")
    check(all(not g["name"].startswith("g5-") for g in T["graphlets"]), "all graphlets named")
    # orbit tables agree with VF2 automorphism orbits
    for g in T["graphlets"]:
        s = g["size"]; H = nx.Graph(); H.add_nodes_from(range(s)); H.add_edges_from(g["edges"])
        auts = list(GraphMatcher(H, H).isomorphisms_iter())
        check(len(auts) == g["n_auts"], "automorphism count %s" % g["name"])
        for u in range(s):
            for v in range(s):
                same = any(a[u] == v for a in auts)
                check(same == (T["tab_orb"][s, g["canon_mask"], u] == T["tab_orb"][s, g["canon_mask"], v]),
                      "orbit partition of %s" % g["name"])
    print("tables ok: 29 graphlets, 73 orbits, orbit partitions = VF2 automorphism orbits")


def test_known():
    T = GL.tables()
    name2id = {g["name"]: g["id"] for g in T["graphlets"]}
    cases = {
        "triangle": (nx.complete_graph(3), {"triangle": 1}),
        "C4": (nx.cycle_graph(4), {"P3 (wedge)": 4, "C4 (square)": 1}),
        "K4": (nx.complete_graph(4), {"triangle": 4, "K4 (4-clique)": 1}),
        "star4": (nx.star_graph(4), {"P3 (wedge)": 6, "3-star (claw)": 4, "4-star": 1}),
        "P5": (nx.path_graph(5), {"P3 (wedge)": 3, "P4 (path)": 2, "P5 (path)": 1}),
        "C5": (nx.cycle_graph(5), {"P3 (wedge)": 5, "P4 (path)": 5, "C5 (5-ring)": 1}),
        "K5": (nx.complete_graph(5), {"triangle": 10, "K4 (4-clique)": 5, "K5 (5-clique)": 1}),
        "house": (nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)]),
                  {"house": 1, "triangle": 1, "C4 (square)": 1, "P3 (wedge)": 6,
                   "paw (tailed triangle)": 2, "P4 (path)": 2}),
    }
    for nm, (G, exp) in cases.items():
        f = GL.graph_features(_gdict(G), "esu", 5)
        want = np.zeros(29, dtype=np.int64)
        for k, v in exp.items():
            want[name2id[k]] = v
        check(np.array_equal(f["counts"], want), "known counts %s: %s" % (nm, f["counts"]))
    # node orbits: star center = orbit 7 (claw center) count C(4,3) = 4; leaves orbit 6 = 3 each
    f = GL.graph_features(_gdict(nx.star_graph(4)), "esu", 5)
    check(f["gdv"][0, 7] == 4 and np.all(f["gdv"][1:, 6] == 3), "star orbits")
    check(f["gdv"][0, 2] == 6 and np.all(f["gdv"][1:, 1] == 3), "star P3 orbits")
    print("known graphs ok (triangle, C4, K4, 4-star, P5, C5, K5, house)")


def random_graphs(rng, k):
    out = []
    for _ in range(k):
        n = int(rng.integers(5, 13)); p = float(rng.uniform(0.15, 0.7))
        out.append(nx.gnp_random_graph(n, p, seed=int(rng.integers(1 << 30))))
    out.append(nx.barbell_graph(4, 2)); out.append(nx.wheel_graph(7)); out.append(nx.petersen_graph())
    out.append(nx.complete_bipartite_graph(3, 4)); out.append(nx.lollipop_graph(5, 3))
    return out


def test_bruteforce(rng):
    t = time.time()
    for G in random_graphs(rng, 25):
        f = GL.graph_features(_gdict(G), "esu", 5)
        c, g = brute_force(G, 5)
        check(np.array_equal(f["counts"], c), "ESU counts vs brute force")
        check(np.array_equal(f["gdv"], g), "ESU GDV vs brute force")
        # consistency: counts = sum of orbit counts / size
        T = GL.tables()
        for gr in T["graphlets"]:
            check(f["gdv"][:, gr["orbits"]].sum() == gr["size"] * f["counts"][gr["id"]], "gdv/count")
    print("ESU (k<=5) == brute force (itertools + VF2) on 30 random/structured graphs (%.1fs)" % (time.time() - t))


def test_permutation(rng):
    for G in random_graphs(rng, 10):
        g = _gdict(G)
        f = GL.graph_features(g, "esu", 5)
        for _ in range(5):
            perm = rng.permutation(g["n"])
            e = perm[g["edges"]]
            e = np.sort(e, axis=1)
            f2 = GL.graph_features({"n": g["n"], "edges": e}, "esu", 5)
            check(np.array_equal(f["counts"], f2["counts"]), "perm counts")
            check(np.array_equal(f["gdv"], f2["gdv"][perm]), "perm gdv")
            f3 = GL.graph_features({"n": g["n"], "edges": e}, "comb4", 4)
            f4 = GL.graph_features(g, "comb4", 4)
            check(np.array_equal(f4["gdv"], f3["gdv"][perm]), "perm gdv comb4")
    print("permutation invariance ok (ESU k<=5 and comb4), 50 random relabelings")


def test_comb4(rng):
    graphs = random_graphs(rng, 60)
    for n, p in [(60, 0.5), (40, 0.9), (80, 0.05)]:
        graphs.append(nx.gnp_random_graph(n, p, seed=int(rng.integers(1 << 30))))
    graphs.append(nx.complete_graph(12))
    H = nx.star_graph(60); H.add_edges_from([(1, 2), (2, 3), (5, 6), (61, 1), (61, 7)]); graphs.append(H)
    graphs.append(nx.barabasi_albert_graph(200, 3, seed=1))
    for G in graphs:
        g = _gdict(G)
        a = GL.graph_features(g, "esu", 4)
        b = GL.graph_features(g, "comb4", 4)
        check(np.array_equal(a["counts"], b["counts"]), "comb4 counts vs ESU")
        check(np.array_equal(a["gdv"], b["gdv"]), "comb4 gdv vs ESU")
    print("comb4 (closed forms + inclusion-matrix inversion) == ESU k<=4 on %d graphs" % len(graphs))


def test_model(rng):
    """api: fit/score/explain run, scores are permutation invariant, explanation is additive."""
    from grail0 import api
    tr = []
    for i in range(60):
        G = nx.gnp_random_graph(int(rng.integers(12, 25)), 0.2, seed=int(rng.integers(1 << 30)))
        g = _gdict(G); g["gid"] = i
        g["node_label"] = rng.integers(0, 3, size=g["n"]).astype(np.int64)
        tr.append(g)
    te = []
    for i in range(10):
        G = nx.gnp_random_graph(int(rng.integers(12, 25)), 0.2 if i < 5 else 0.5, seed=int(rng.integers(1 << 30)))
        g = _gdict(G); g["gid"] = 1000 + i
        g["node_label"] = rng.integers(0, 3, size=g["n"]).astype(np.int64)
        te.append(g)
    model = api.fit(tr, engine="esu", kmax=5, seed=0, n_labels=3)
    s = api.score(model, te)
    check(np.all(np.isfinite(s["sum_z"])), "finite scores")
    # permutation invariance of the whole pipeline
    te2 = []
    for g in te:
        perm = rng.permutation(g["n"]); inv = np.argsort(perm)
        e = np.sort(perm[g["edges"]], axis=1)
        nl = np.empty_like(g["node_label"]); nl[perm] = g["node_label"]
        te2.append({"n": g["n"], "edges": e, "x": None, "node_label": nl, "gid": g["gid"] + 500})
    s2 = api.score(model, te2)
    check(np.allclose(s["sum_z"], s2["sum_z"], atol=1e-8), "pipeline permutation invariance")
    ex = api.explain(model, te[7])
    check(abs(ex["check_sum_minus_score"]) < 1e-6, "explanation additivity %g" % ex["check_sum_minus_score"])
    print("api ok: fit/score/explain; pipeline scores invariant to node relabeling; "
          "explanation sums to the score (|delta| = %.1e)" % abs(ex["check_sum_minus_score"]))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    test_tables()
    test_known()
    test_bruteforce(rng)
    test_permutation(rng)
    test_comb4(rng)
    if "--no-model" not in sys.argv:
        test_model(rng)
    print("ALL TESTS PASSED")
