"""MS-v2 design work, diagnosis D4 (NOTES_MSv2.md §1): the v1 edits applied sequentially to up to k sites of the same
parent (k = 1, 2, 3, all; MS2 k = 1, 2), each step passing v1's own per-type acceptance on the current graph.
Edits per (type, parent) are computed once (RNG keyed by type, parent gid and step, not by seed) and the v1 per-seed
parent sets are used. Output msv2_work/diag_v1_multi.json. No detector of any kind."""
import sys

sys.dont_write_bytecode = True
import json
import os.path as osp
import time
from multiprocessing import Pool

import numpy as np
import networkx as nx
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import ctrllib as CL  # noqa: E402
import typed_data as TD  # noqa: E402
from typed import molswap as V1  # noqa: E402

KMAX = {"MS1": 12, "MS2": 2, "MS3": 12}
MAGIC = 77001


def fresh_info(n, E):
    G = V1.nx_graph(n, E)
    br = V1.bridge_set(G)
    return {"n": n, "E": set(E), "G": G, "br": br, "rn": V1.ring_nodes(G, br), "hist": V1.ring_hist(n, E),
            "deg": dict(G.degree())}


def chain(args):
    t, P = args
    code = V1.TYPE_CODE[t]
    n, E = P["n"], V1.eset_of(P["edges"])
    out = []
    for step in range(KMAX[t]):
        info = fresh_info(n, E)
        cur = {"n": n, "edges": V1.edges_array(E), "node_label": P["node_label"], "gid": -1}
        rng = np.random.default_rng(np.random.SeedSequence([MAGIC, code, P["gid"], step]))
        res, why = V1.MAKERS[t](cur, info, rng)
        if res is None:
            break
        E = res["E2"]
        out.append(V1.edges_array(E))
    return t, P["gid"], out


if __name__ == "__main__":
    t0 = time.time()
    pool, _, _ = V1.load_pool()
    by_gid = {g["gid"]: g for g in pool}
    man = json.load(open(osp.join(V1.OUT_DIR, "manifest.json")))
    par_union = sorted(set(x for s in TD.SEEDS for x in man["seeds"][str(s)]["split"]["parents"]))
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if lim:
        par_union = par_union[:lim]
    jobs = [(t, by_gid[g]) for t in V1.TYPES for g in par_union]
    with Pool(4) as pp:
        res = pp.map(chain, jobs, chunksize=8)
    seq = {(t, gid): e for t, gid, e in res}
    print(f"edits done {time.time() - t0:.1f}s", flush=True)
    rows = []
    for s in TD.SEEDS:
        td = TD.load_typed("MS", s)
        pa = man["seeds"][str(s)]["split"]["parents"]
        if lim:
            pa = [g for g in pa if g in set(par_union)]
        for t in V1.TYPES:
            for k in (1, 2, 3, 5, "all"):
                if t == "MS2" and k not in (1, 2):
                    continue
                an, nsites = [], []
                for g in pa:
                    e = seq[(t, g)]
                    if not e:
                        continue
                    kk = len(e) if k == "all" else min(k, len(e))
                    P = by_gid[g]
                    an.append({"n": P["n"], "edges": e[kk - 1], "x": None, "node_label": P["node_label"], "gid": -g})
                    nsites.append(kk)
                if len(an) < 10:
                    continue
                a = CL.cell_aurocs(td.train, td.test_normals, an, 37)
                rows.append(dict(seed=s, type=t, k=str(k), n_anom=len(an), mean_edits=float(np.mean(nsites)), **a))
    df = pd.DataFrame(rows)
    df.to_json(osp.join(HERE, "diag_v1_multi.json" if not lim else "diag_v1_multi_test.json"), orient="records", indent=1)
    dist = {t: np.bincount([len(seq[(t, g)]) for g in par_union]).tolist() for t in V1.TYPES}
    print("sites-available histogram (number of accepted sequential edits per parent):", dist)
    print(df.groupby(["type", "k"]).mean(numeric_only=True).drop(columns="seed").round(1).to_string())
    print(f"total {time.time() - t0:.1f}s")
