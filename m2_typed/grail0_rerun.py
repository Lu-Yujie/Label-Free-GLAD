"""GRAIL-0 and its controls on the M2 sets AS EXPORTED (round-tripped through the .npz files the deep
models read), compared with the original numbers in results/synthetic_*.csv.

Purpose
  1. Identity check: if every AUROC equals the original one, the deep models and GRAIL-0 were
     evaluated on exactly the same generated graphs, and the summary may quote the original CSVs.
  2. An F1-matched GRAIL-0 reference on GLAD-X-lite: channel C fitted on node labels only
     (api.fit(..., use_x=False)); method names carry the prefix 'labonly_'. NEW numbers, not in the
     original CSVs; reference only.
The calls replicate synthetic/attrib.py (GX), synthetic/flip.py (Syn-Com) and synthetic/bm.py (BM)
exactly: GRAIL-0 via grail0/api.py unchanged (engine esu, kmax 5, seed = data seed, n_labels = 4 on
GX), controls via grail0/controls.py with the same parameters; attr_pool is copied from attrib.py.
Nothing in grail0/, common/ or synthetic/ is modified.

Output: results/m2_grail0_rerun.csv
Usage (CPU venv, PYTHONPATH=prelim):  python -B m2_typed/grail0_rerun.py --n_jobs 16
"""
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PRELIM = osp.dirname(HERE)
RES = osp.join(PRELIM, "results")
sys.path.insert(0, PRELIM)
sys.path.insert(0, HERE)
import m2io  # noqa: E402

ENGINE = ("esu", 5)


def attr_pool(graphs, L=4):  # verbatim copy of synthetic/attrib.py:attr_pool
    out = []
    for g in graphs:
        h = np.bincount(np.minimum(g["node_label"], L), minlength=L + 1)[: L + 1]
        x = np.asarray(g["x"], np.float64)
        out.append(np.r_[np.log1p(h), x.mean(0), x.std(0)])
    return np.stack(out)


def run_cfg(ds, seed):
    from common.metrics import evaluate
    from grail0 import api, graphlets as GL
    from grail0 import controls as C
    t0 = time.time()
    train, test, group, names = m2io.load(ds, seed)
    Ftr = [GL.graph_features(g, *ENGINE) for g in train]
    Fte = [GL.graph_features(g, *ENGINE) for g in test]
    methods = {}
    if ds == "GX":
        for tag, use_x in (("", None), ("labonly_", False)):
            model = api.fit(train, engine=ENGINE[0], kmax=ENGINE[1], seed=seed, n_labels=4, feats=Ftr, use_x=use_x)
            assert model["channels"] == ["S0", "A", "B", "C"], model["channels"]
            assert model["use_x"] == (use_x is None)
            out = api.score(model, test, feats=Fte)
            methods.update({"grail0_" + tag + k: out[k] for k in ("sum_z", "max_z", "sum_z_noS0", "cauchy", "stouffer")})
            methods.update({"ch_" + tag + c: out[c] for c in model["channels"]})
        Str, Ste = C.size_feats(Ftr), C.size_feats(Fte)
        methods["ctrl_size_lw"] = C.mahal(Str, Ste, C.CONFIG["ridge_S0"])
        Gtr, Gte = C.gc_feats(Ftr), C.gc_feats(Fte)
        methods["ctrl_gc_iforest"] = C.iforest(Gtr, Gte, seed)
        methods["ctrl_gc_lw"] = C.mahal(Gtr, Gte, C.CONFIG["ridge_A"])
        Htr, Hte = C.label_hist(train, 4), C.label_hist(test, 4)
        methods["ctrl_gclab_iforest"] = C.iforest(np.c_[Gtr, Htr], np.c_[Gte, Hte], seed)
        methods["ctrl_gclab_lw"] = C.mahal(np.c_[Gtr, Htr], np.c_[Gte, Hte], C.CONFIG["ridge_A"])
        methods["ctrl_attrpool_lw"] = C.mahal(attr_pool(train), attr_pool(test), 1e-6)
    else:
        model = api.fit(train, engine=ENGINE[0], kmax=ENGINE[1], seed=seed, feats=Ftr)
        out = api.score(model, test, feats=Fte)
        methods.update({"grail0_" + k: out[k] for k in ("sum_z", "max_z", "sum_z_noS0", "cauchy", "stouffer")})
        methods.update({"ch_" + c: out[c] for c in model["channels"]})
        methods["ctrl_size_lw"] = C.mahal(C.size_feats(Ftr), C.size_feats(Fte), C.CONFIG["ridge_S0"])
        Gtr, Gte = C.gc_feats(Ftr), C.gc_feats(Fte)
        methods["ctrl_gc_iforest"] = C.iforest(Gtr, Gte, seed)
        methods["ctrl_gc_lw"] = C.mahal(Gtr, Gte, C.CONFIG["ridge_A"])
    rows = []
    for k in range(1, len(names)):
        sel = (group == 0) | (group == k)
        y = (group[sel] == k).astype(int)
        for m, s in methods.items():
            rows.append({"dataset": ds, "anomaly_type": names[k], "seed": seed, "method": m,
                         **evaluate(np.asarray(s)[sel], y)})
    print("[grail0_rerun] %s s%d %.1fs" % (ds, seed, time.time() - t0), flush=True)
    return rows


def originals():
    """Original per-seed rows, keyed like run_cfg's rows."""
    out = []
    a = pd.read_csv(osp.join(RES, "synthetic_attrib_detection.csv"))
    for r in a.itertuples():
        out.append(("GX", r.type, int(r.seed), r.method, float(r.auroc)))
    f = pd.read_csv(osp.join(RES, "synthetic_flip.csv"))
    f = f[(f.n == 20) & np.isclose(f.tau0, 0.4) & f.taua.round(2).isin([0.6, 0.8, 0.2])]
    for r in f.itertuples():
        out.append(("SC20-a%.2f" % r.taua, "taua%.2f" % r.taua, int(r.seed), r.method, float(r.auroc)))
    b = pd.read_csv(osp.join(RES, "synthetic_bm_detection.csv"))
    for r in b.itertuples():
        out.append((r.dataset, r.dataset.replace("BM-", ""), int(r.seed), r.method, float(r.auroc)))
    return pd.DataFrame(out, columns=["dataset", "anomaly_type", "seed", "method", "auroc_original"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=1)
    a = ap.parse_args()
    cfgs = [(ds, s) for ds in m2io.DATASETS for s in m2io.SEEDS]
    cfgs.sort(key=lambda c: c[0] != "GX")  # heaviest first
    if a.n_jobs > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(run_cfg)(d, s) for d, s in cfgs)
    else:
        res = [run_cfg(d, s) for d, s in cfgs]
    df = pd.DataFrame([r for rr in res for r in rr])
    df = df.merge(originals(), on=["dataset", "anomaly_type", "seed", "method"], how="left")
    df["abs_diff_auroc"] = (df["auroc"] - df["auroc_original"]).abs()
    out = osp.join(RES, "m2_grail0_rerun.csv")
    df.to_csv(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    cmp_ = df.dropna(subset=["auroc_original"])
    print("rows %d; compared with originals: %d; max |diff| AUROC = %.3g; rows without an original: %s" % (
        len(df), len(cmp_), cmp_["abs_diff_auroc"].max(),
        sorted(df[df.auroc_original.isna()].method.unique().tolist())))
    exp = originals()
    exp = exp[exp.method.isin(df.method.unique())]
    missing = exp.merge(df[["dataset", "anomaly_type", "seed", "method"]], how="left", indicator=True)
    print("original rows (same methods) not reproduced:", int((missing["_merge"] == "left_only").sum()))


if __name__ == "__main__":
    main()
