"""Unit tests for pseudo.py on small hand-made graphs (pure numpy; no torch needed).

  PYTHONDONTWRITEBYTECODE=1 python -B test_pseudo.py
"""
import os.path as _osp, sys as _sys  # M1: this directory holds select.py, which would shadow the
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))  # stdlib module `select` (used by
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]  # subprocess): drop it from sys.path

import itertools
import os.path as osp
import sys

import numpy as np

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("pseudo", osp.join(_HERE_, "pseudo.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def path(n):
    return [(i, i + 1) for i in range(n - 1)]


def cycle(n):
    return sorted([(i, i + 1) for i in range(n - 1)] + [(0, n - 1)])


def star(n):
    return [(0, i) for i in range(1, n)]


def complete(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def molecule():
    """benzene ring with three substituents and a 2-atom tail (12 nodes, 12 edges)."""
    e = cycle(6) + [(0, 6), (2, 7), (4, 8), (8, 9), (9, 10), (9, 11)]
    lab = np.array([0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 1])
    return 12, sorted(e), lab


def random_graph(n, p, seed):
    r = np.random.default_rng(seed)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if r.random() < p]


def check_simple(n, e):
    assert all(0 <= i < j < n for i, j in e), "not (i<j) / out of range / self-loop"
    assert len(set(e)) == len(e), "multi-edge"


def has_k4_on(edges, nodes):
    s = set(edges)
    return all(P._key(a, b) in s for a, b in itertools.combinations(nodes, 2))


def count_k4(n, edges):
    adj = P.adjacency(n, edges)
    return sum(1 for q in itertools.combinations(range(n), 4)
               if all(b in adj[a] for a, b in itertools.combinations(q, 2)))


def test_roundtrip():
    n, e, _ = molecule()
    ei = P.index_from_edges(e)
    assert ei.shape == (2, 2 * len(e))
    assert P.edges_from_index(ei, n) == e
    # symmetric and sorted by (row, col)
    pairs = list(zip(ei[0].tolist(), ei[1].tolist()))
    assert pairs == sorted(pairs) and set(pairs) == {(j, i) for i, j in pairs}


def test_rewire():
    graphs = [(12,) + (molecule()[1],), (30, random_graph(30, 0.15, 1)), (20, cycle(20)), (10, path(10)),
              (8, star(8)), (5, complete(5)), (4, path(4)), (2, [(0, 1)])]
    for n, e in graphs:
        for seed in range(5):
            r = np.random.default_rng(seed)
            e2, info = P.p_rewire(n, e, r)
            check_simple(n, e2)
            assert len(e2) == len(e), "m changed"
            if info["fallback_edges"] == 0:
                assert np.array_equal(P.degrees(n, e2), P.degrees(n, e)), "degree sequence changed"
    # a random graph: swaps succeed, ~15% of edges touched, degrees exact
    n, e = 40, random_graph(40, 0.2, 7)
    e2, info = P.p_rewire(n, e, np.random.default_rng(0))
    assert info["swaps"] == info["target_swaps"] == max(1, round(0.15 * len(e) / 2)) and info["fallback_edges"] == 0
    assert 0 < len(set(e2) - set(e)) <= 2 * info["swaps"]
    # star: no valid swap exists -> fallback random rewiring keeps m (degrees may change)
    e2, info = P.p_rewire(8, star(8), np.random.default_rng(0))
    assert info["swaps"] == 0 and info["fallback_edges"] > 0 and len(e2) == 7 and set(e2) != set(star(8))
    # complete graph: nothing can change
    e2, info = P.p_rewire(5, complete(5), np.random.default_rng(0))
    assert set(e2) == set(complete(5))


def test_motif():
    n, e, _ = molecule()
    for seed in range(10):
        e2, info = P.p_motif(n, e, np.random.default_rng(seed))
        check_simple(n, e2)
        assert len(e2) == len(e), "m changed"
        assert info["plants"] == 1 and info["added"] == info["deleted"] >= 1
        assert count_k4(n, e2) >= 1 and count_k4(n, e) == 0, "no planted 4-clique"
    # larger sparse graph: k_plant = round(m/40) cliques
    n, e = 120, random_graph(120, 0.03, 3)
    e2, info = P.p_motif(n, e, np.random.default_rng(0))
    check_simple(n, e2)
    assert len(e2) == len(e) and info["plants"] == max(1, round(len(e) / 40))
    assert count_k4(n, e2) >= info["plants"] - 0  # disjoint plants -> at least that many K4s
    # complete graph: nothing to add -> unchanged
    e2, info = P.p_motif(6, complete(6), np.random.default_rng(0))
    assert set(e2) == set(complete(6)) and info["added"] == 0
    # 3-node path: no 4-set -> triangle-closure fallback cannot keep m (all edges protected) -> unchanged
    e2, info = P.p_motif(3, path(3), np.random.default_rng(0))
    assert len(e2) == 2
    # deletions avoid isolating nodes when possible
    n, e = 30, cycle(30) + [(0, 15), (5, 20)]
    e = sorted(set(P._key(*x) for x in e))
    for seed in range(10):
        e2, _ = P.p_motif(n, e, np.random.default_rng(seed))
        assert P.degrees(n, e2).min() >= 1


def test_roleswap():
    n, e, lab = molecule()
    deg = P.degrees(n, e)
    for seed in range(10):
        lab2, info = P.p_roleswap(n, e, lab, np.random.default_rng(seed))
        assert np.array_equal(np.sort(lab2), np.sort(lab)), "label multiset changed"
        assert info["swaps"] == max(1, round(0.05 * n))
        ch = np.where(lab2 != lab)[0]
        assert len(ch) == 2 * info["swaps"]
        # swapped pair: different degrees (structural roles) and different labels
        assert deg[ch[0]] != deg[ch[1]]
    # all labels equal -> nothing to swap -> unchanged
    lab2, info = P.p_roleswap(n, e, np.zeros(n, dtype=int), np.random.default_rng(0))
    assert info["swaps"] == 0 and np.array_equal(lab2, np.zeros(n))
    # regular graph (all degrees equal) -> no role difference -> unchanged
    lab2, info = P.p_roleswap(10, cycle(10), np.arange(10) % 3, np.random.default_rng(0))
    assert info["swaps"] == 0
    # high-degree <-> low-degree preference: star centre swaps with a leaf
    lab = np.array([0] + [1] * 9)
    lab2, info = P.p_roleswap(10, star(10), lab, np.random.default_rng(0))
    assert lab2[0] == 1 and lab2.sum() == 9 and info["relaxed"] == 0


def test_labelnoise():
    n, e, lab = molecule()
    marg = np.array([0.7, 0.2, 0.1, 0.0])
    for seed in range(20):
        lab2, info = P.p_labelnoise(n, lab, marg, np.random.default_rng(seed))
        assert info["k"] == max(1, round(0.1 * n))
        assert 1 <= (lab2 != lab).sum() <= info["k"]
        assert lab2.max() <= 2, "drew a label with zero marginal probability"
    # input is not modified in place
    lab0 = lab.copy()
    P.p_labelnoise(n, lab, marg, np.random.default_rng(0))
    assert np.array_equal(lab, lab0)


def test_determinism_and_no_global_rng():
    n, e, lab = molecule()
    marg = np.array([0.6, 0.3, 0.1])
    st_np = np.random.get_state()
    import random
    st_py = random.getstate()
    a = [P.perturb("BZR", 0, f, 3, n, e, lab, marg) for f in P.FAMILIES]
    b = [P.perturb("BZR", 0, f, 3, n, e, lab, marg) for f in P.FAMILIES]
    for (e1, l1, _), (e2, l2, _) in zip(a, b):
        assert e1 == e2 and np.array_equal(l1, l2)
    c = P.perturb("BZR", 1, "rewire", 3, n, e, lab, marg)
    d = P.perturb("BZR", 0, "rewire", 4, n, e, lab, marg)
    assert np.array_equal(np.random.get_state()[1], st_np[1]) and random.getstate() == st_py, "global RNG touched"
    assert (c[0] != a[0][0]) or (d[0] != a[0][0])


def test_size_preserving_all_families():
    n, e, lab = molecule()
    marg = np.array([0.6, 0.3, 0.1])
    for i in range(30):
        for f in P.FAMILIES:
            e2, l2, _ = P.perturb("X", 0, f, i, n, e, lab, marg)
            assert len(e2) == len(e) and len(l2) == n


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"all {len(tests)} tests passed")
