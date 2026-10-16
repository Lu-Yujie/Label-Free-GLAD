"""Descriptive structure statistics of the NCI1 normal pool for MS-v2 design (no control, no detector)."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __import__('os').path.dirname(__file__))
import ctrllib as CL  # noqa
from typed import molswap as V1
import numpy as np, networkx as nx
from collections import Counter

pool, _, _ = V1.load_pool()


def rings_of(G, size):
    return sorted(set(V1.canon_cycle(c) for c in nx.chordless_cycles(G, length_bound=size) if len(c) == size))


rows = []
for g in pool:
    n, E = g["n"], V1.eset_of(g["edges"])
    G = V1.nx_graph(n, E); br = V1.bridge_set(G); rn = V1.ring_nodes(G, br); deg = dict(G.degree())
    lab = g["node_label"]
    h = V1.ring_hist(n, E); hs = set(h)
    r6 = rings_of(G, 6); r5 = rings_of(G, 5)
    # contractible 6-rings: has a deg-2 atom
    c6 = [R for R in r6 if any(deg[v] == 2 for v in R)]
    c6lab = [R for R in r6 if any(deg[v] == 2 and any(lab[v] == lab[p] for p in (R[i - 1], R[(i + 1) % 6]))
                                  for i, v in enumerate(R))]
    # expandable 5-rings: ring atom p with an acyclic (non-ring) neighbour v
    e5 = [R for R in r5 if any(any(u not in rn for u in G.neighbors(p)) for p in R)]
    e5lab = [R for R in r5 if any(any((u not in rn and lab[u] == lab[p]) for u in G.neighbors(p)) for p in R)]
    rows.append(dict(n=n, hs=tuple(sorted(hs)), n6=len(r6), n5=len(r5), c6=len(c6), c6lab=len(c6lab), e5=len(e5),
                     e5lab=len(e5lab)))
import pandas as pd
d = pd.DataFrame(rows)
d["both56"] = d.hs.apply(lambda s: 5 in s and 6 in s)
d["exact56"] = d.hs.apply(lambda s: set(s) == {5, 6})
print(len(d), "both56", d.both56.sum(), "exact56", d.exact56.sum())
for nm, sub in (("all", d), ("both56", d[d.both56]), ("exact56", d[d.exact56])):
    print(nm, len(sub), "c6>=1", (sub.c6 >= 1).mean().round(3), "c6lab>=1", (sub.c6lab >= 1).mean().round(3),
          "e5>=1", (sub.e5 >= 1).mean().round(3), "e5lab>=1", (sub.e5lab >= 1).mean().round(3),
          "both c6&e5", ((sub.c6 >= 1) & (sub.e5 >= 1)).mean().round(3),
          "n median", sub.n.median(), "n6 mean", sub.n6.mean().round(2), "n5 mean", sub.n5.mean().round(2))
print(d[d.both56].groupby(["n5", "n6"]).size().sort_values(ascending=False).head(12))
