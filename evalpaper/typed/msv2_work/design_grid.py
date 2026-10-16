"""MS-v2 iteration-1 candidate grid (NOTES_MSv2.md §3): for one pool, generate every candidate (family, k) in memory
with molswap_v2's own functions and score the A9 §4 controls + the null-edit check per seed. No detector of any kind.
    python design_grid.py P0|P56 [--limit N]     -> msv2_work/grid_iter1_<pool>.json
"""
import sys

sys.dont_write_bytecode = True
import argparse
import json
import os.path as osp
import time
from collections import Counter
from multiprocessing import get_context

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import ctrllib as CL  # noqa: E402
import molswap_v2 as M  # noqa: E402

FAMS_ITER = {"iter1": {"A": [1, 2, "all"], "B": [1, 2, "all"], "C": [1, 2], "D": [1, 2]},
             "iter2": {"E": [1, 2, "all"], "F": [1, 2, "all"]}}
FAMS = None
L = 37
PARENTS = None
SEED = None


def work(args):
    fam, P = args
    ks = FAMS[fam]
    kmax = M.KCAP if "all" in ks else max(ks)
    seq = M.edit_sequence(P, fam, SEED, kmax)
    out = {}
    for k in ks:
        res, why = M.finalize(P, seq, k)
        if res is None:
            out[str(k)] = (None, why, None, "matched_infeasible", 0)
            continue
        gm = {"n": P["n"], "edges": M.edges_array(res["E"]), "x": None, "node_label": P["node_label"], "gid": -1}
        sc, why2 = M.attach_donor(P, res, PARENTS, SEED, fam)
        gs = None if sc is None else {"n": sc["n"], "edges": M.edges_array(sc["E"]), "x": None,
                                      "node_label": sc["labels"], "gid": -2}
        out[str(k)] = (gm, None, gs, why2, res["n_steps"])
    return fam, P["gid"], out


def main():
    global PARENTS, SEED, FAMS
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", choices=M.POOLS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--iter", default="iter1", choices=sorted(FAMS_ITER))
    ap.add_argument("--seeds", default=None, help="comma list (default all); output gets a _s<list> suffix")
    a = ap.parse_args()
    FAMS = FAMS_ITER[a.iter]
    t0 = time.time()
    M.reference()
    pool = M.load_pool(a.pool)
    by_gid = {g["gid"]: g for g in pool}
    rows = []
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else M.SEEDS
    for s in seeds:
        train, tn, parents = M.split_pool(pool, s)
        PARENTS, SEED = parents, s
        todo = parents[: a.limit] if a.limit else parents
        jobs = [(f, P) for f in FAMS for P in todo]
        with get_context("fork").Pool(4) as pp:
            res = pp.map(work, jobs, chunksize=4)
        for fam in FAMS:
            for k in FAMS[fam]:
                key = str(k)
                an, sc, nullp, sk_m, sk_s, nst = [], [], [], Counter(), Counter(), []
                for f, gid, out in res:
                    if f != fam:
                        continue
                    gm, why, gs, why2, ns = out[key]
                    if gm is None:
                        sk_m[why] += 1
                        continue
                    an.append(gm)
                    nst.append(ns)
                    nullp.append(by_gid[gid])
                    if gs is None:
                        sk_s[why2] += 1
                    else:
                        sc.append(gs)
                r = dict(pool=a.pool, family=fam, k=key, seed=s, n_pool=len(pool), n_train=len(train),
                         n_test_normal=len(tn), n_parents=len(todo), n_matched=len(an), n_sizechg=len(sc),
                         mean_edits=float(np.mean(nst)) if nst else None,
                         edits_hist=dict(sorted(Counter(nst).items())), skips_matched=dict(sk_m),
                         skips_sizechg=dict(sk_s))
                if len(an) >= 5:
                    r.update({f"matched_{c}": v for c, v in CL.cell_aurocs(train, tn, an, L).items()})
                    r.update({f"null_{c}": v for c, v in CL.cell_aurocs(train, tn, nullp, L).items()})
                if len(sc) >= 5:
                    r.update({f"sizechg_{c}": v for c, v in CL.cell_aurocs(train, tn, sc, L).items()})
                rows.append(r)
        print(f"{a.pool} seed {s} done {time.time() - t0:.1f}s", flush=True)
    out = osp.join(HERE, f"grid_{a.iter}_{a.pool}{'_test' if a.limit else ''}{'_s' + a.seeds.replace(',', '') if a.seeds else ''}.json")
    json.dump(dict(rows=rows, seconds=time.time() - t0, molswap_v2_sha256=M.sha256_file(M.__file__),
                   threads=__import__("os").environ.get("OMP_NUM_THREADS")), open(out, "w"), indent=1)
    print("wrote", out, f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
