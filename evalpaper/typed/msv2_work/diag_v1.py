"""MS-v2 design work, diagnosis of MolSwap v1 with controls only (NOTES_MSv2.md §1: D0, D1, D2, D3, D5).
D4 (edit multiplicity) is in diag_v1_multi.py. Output: msv2_work/diag_v1.json. No detector of any kind."""
import sys

sys.dont_write_bytecode = True
import json
import os.path as osp
import time

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import ctrllib as CL  # noqa: E402
import typed_data as TD  # noqa: E402
from typed import molswap as MS1  # noqa: E402  (v1, read-only import)
from grail0 import controls as C  # noqa: E402
from grail0.channels import CONFIG, _lw, comp_design, comp_y  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

L = 37
t0 = time.time()
ref = pd.read_csv(osp.join(CL.PRELIM, "results", "eval_typed_controls.csv"))
ref = ref[ref.dataset == "MS"]
pool, _, _ = MS1.load_pool()
by_gid = {g["gid"]: g for g in pool}
pool_n = np.array([g["n"] for g in pool])
terc = np.quantile(pool_n, [1 / 3, 2 / 3])


def n_band(n):
    return int(np.searchsorted(terc, n, side="right"))


def rank_cls(g):
    r = len(g["edges"]) - g["n"] + 1
    return 0 if r <= 1 else (1 if r == 2 else 2)


C5 = None
for gr in GL_T if False else []:
    pass
from grail0 import graphlets as GL  # noqa: E402
T = GL.tables()
C5 = [x["id"] for x in T["graphlets"] if x["name"] == "C5 (5-ring)"][0]


def has5(g):
    return int(CL.feats(g)["counts"][C5] > 0)


CTRL_NAME = {"gc_lw": "GRAIL0:ctrl_gc_lw", "size_lw": "GRAIL0:ctrl_size_lw", "elp_lw": "POSTHOC:ctrl_edgelab_lw"}
out = {"D0": [], "D1": [], "D2": [], "D3": [], "D5": []}


def alt_controls(train, test):
    """D5: generic alternatives (diagnostic only)."""
    Ftr, Fte = CL.F(train), CL.F(test)
    res = {}
    # (a) GRAIL-0 channel A: residualised graphlet log-counts -> LW
    Xd, Y = comp_design(Ftr), comp_y(Ftr)
    beta = np.linalg.lstsq(Xd, Y, rcond=None)[0]
    R = Y - Xd @ beta
    _, P = _lw(R, CONFIG["ridge_A"], centered=True)
    Rt = comp_y(Fte) - comp_design(Fte) @ beta
    res["chanA_lw"] = np.sum(Rt @ P * Rt, 1)
    # (b) kNN (k=10) in LW-whitened gc space
    Xtr, Xte = C.gc_feats(Ftr), C.gc_feats(Fte)
    mu, Pg = _lw(Xtr, CONFIG["ridge_A"])
    w, V = np.linalg.eigh(Pg)
    W = V * np.sqrt(np.maximum(w, 0))
    Ztr, Zte = (Xtr - mu) @ W, (Xte - mu) @ W
    D = ((Zte[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
    res["gc_knn10"] = np.sqrt(np.sort(D, 1)[:, :10]).mean(1)
    # (c) gc + 73 orbit totals -> LW
    def orb(Fs):
        return np.stack([np.log1p(f["gdv"].sum(0).astype(float)) for f in Fs])
    X2tr, X2te = np.hstack([Xtr, orb(Ftr)]), np.hstack([Xte, orb(Fte)])
    res["gc_orb_lw"] = C.mahal(X2tr, X2te, CONFIG["ridge_A"])
    return res


for s in TD.SEEDS:
    td = TD.load_typed("MS", s)
    train, tn = td.train, td.test_normals
    nn = len(tn)
    # D0
    for key, an in td.groups.items():
        a = CL.cell_aurocs(train, tn, an, L)
        t, v = key.split(":")
        for c, val in a.items():
            r = ref[(ref.method == CTRL_NAME[c]) & (ref.anomaly_type == t) & (ref.size_variant == TD.SIZE_VARIANT_CSV[v])
                    & (ref.seed == s)]
            out["D0"].append(dict(seed=s, cell=key, ctrl=c, auroc=val, ref=float(r.auroc.iloc[0]),
                                  absdiff=abs(val - float(r.auroc.iloc[0]))))
    # fitted gc model pieces for D2
    Ftr = CL.F(train)
    Xtr = C.gc_feats(Ftr)
    mu, Pg = _lw(Xtr, CONFIG["ridge_A"])
    rng = np.random.default_rng(1000 + s)
    i, j = rng.integers(0, len(train), 4000), rng.integers(0, len(train), 4000)
    keep = i != j
    dX = Xtr[i[keep]] - Xtr[j[keep]]
    pair_med = float(np.median(np.sqrt(np.sum(dX @ Pg * dX, 1))))
    for t in MS1.TYPES:
        an = td.groups[f"{t}:matched"]
        parents = [by_gid[g["parent_gid"]] for g in an]
        # D1 null edit
        a = CL.cell_aurocs(train, tn, parents, L)
        out["D1"].append(dict(seed=s, type=t, n_parents=len(parents), **{f"null_{k}": v for k, v in a.items()}))
        # D2 edit magnitude
        Fa, Fp = C.gc_feats(CL.F(an)), C.gc_feats(CL.F(parents))
        sa, sp = np.sum((Fa - mu) @ Pg * (Fa - mu), 1), np.sum((Fp - mu) @ Pg * (Fp - mu), 1)
        d = Fa - Fp
        disp = np.sqrt(np.sum(d @ Pg * d, 1))
        Fn = C.gc_feats(CL.F(tn))
        sn = np.sum((Fn - mu) @ Pg * (Fn - mu), 1)
        out["D2"].append(dict(seed=s, type=t, frac_anom_gt_parent=float(np.mean(sa > sp)),
                              median_score_delta=float(np.median(sa - sp)), median_disp=float(np.median(disp)),
                              median_pair_dist_normals=pair_med, ratio=float(np.median(disp)) / pair_med,
                              median_score_test_normals=float(np.median(sn)), median_score_anom=float(np.median(sa)),
                              median_score_parents=float(np.median(sp))))
        # D3 strata: (i) global model, (ii) refit within stratum
        S_all = CL.scores(train, list(tn) + list(an), L, ("gc_lw",))["gc_lw"]
        for sname, fn in (("n_band", lambda g: n_band(g["n"])), ("rank_cls", rank_cls), ("has5", has5)):
            st_tn = np.array([fn(g) for g in tn])
            st_an = np.array([fn(by_gid[g["parent_gid"]]) for g in an])
            st_tr = np.array([fn(g) for g in train])
            for b in sorted(set(st_tn) | set(st_an)):
                mn, ma, mt = st_tn == b, st_an == b, st_tr == b
                rec = dict(seed=s, type=t, strat=sname, stratum=int(b), n_test_normals=int(mn.sum()),
                           n_anom=int(ma.sum()), n_train=int(mt.sum()))
                if mn.sum() >= 10 and ma.sum() >= 10:
                    rec["global_gc"] = CL.auroc(S_all[:nn][mn], S_all[nn:][ma])
                    if mt.sum() >= 50:
                        tr_b = [g for g, k in zip(train, mt) if k]
                        a2 = CL.cell_aurocs(tr_b, [g for g, k in zip(tn, mn) if k], [g for g, k in zip(an, ma) if k], L)
                        rec.update({f"refit_{k}": v for k, v in a2.items()})
                out["D3"].append(rec)
        # D5 alternative controls
        alt = alt_controls(train, list(tn) + list(an))
        out["D5"].append(dict(seed=s, type=t, **{k: CL.auroc(v[:nn], v[nn:]) for k, v in alt.items()}))
    print(f"seed {s} done {time.time() - t0:.1f}s", flush=True)

out["meta"] = dict(seconds=time.time() - t0, n_terciles=terc.tolist())
json.dump(out, open(osp.join(HERE, "diag_v1.json"), "w"), indent=1)
# summaries
d0 = pd.DataFrame(out["D0"]); print("D0 max absdiff", d0.absdiff.max())
for k in ("D1", "D2", "D5"):
    df = pd.DataFrame(out[k]); print(k); print(df.groupby("type").mean(numeric_only=True).drop(columns="seed").round(2).to_string())
d3 = pd.DataFrame(out["D3"])
print("D3"); print(d3.groupby(["type", "strat", "stratum"]).mean(numeric_only=True).drop(columns="seed").round(1).to_string())
