"""GRAIL-0 Gate-1 pilot driver (resumable).

  cd <repo> && PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -m grail0.run \
      --datasets AIDS BZR --n_jobs 32
  ... -m grail0.run --aggregate      # merge per-fold parts into the two result CSVs

Per dataset: (1) label-free graphlet cache over all graphs (fold-independent), (2) WL labels,
(3) the 5 folds in parallel; each fold writes results/grail0_parts/<DS>__fold<k>.csv and is
skipped if that file exists.
"""
import argparse
import glob
import json
import os
import os.path as osp
import pickle
import time

import numpy as np
import pandas as pd

PRELIM = osp.dirname(osp.dirname(osp.abspath(__file__)))
RES = osp.join(PRELIM, "results")
PARTS = os.environ.get("GRAIL0_PARTS", osp.join(RES, "grail0_parts"))
WL_DIR = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/cache/wl")
COLS = ["dataset", "fold", "method", "auroc", "auprc", "fpr95", "conformal_fpr_at_0.05"]
GRAIL_METHODS = {"sum_z": "grail0_sumz", "max_z": "grail0_maxz", "sum_z_noS0": "grail0_noS0_sumz",
                 "S0": "ch_S0", "A": "ch_A", "B": "ch_B", "C": "ch_C",
                 "cauchy": "grail0_cauchy", "stouffer": "grail0_stouffer"}
ATTR_DATASETS = {"AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "PROTEINS_full", "Tox21_p53", "Tox21_MMP"}


def _graphs_by_gid(name):
    from common.data import load_folds
    gb = {}
    for f in load_folds(name):
        for g in f["train"] + f["test"]:
            gb[g["gid"]] = g
    return gb


def prepare(name, n_jobs):
    from grail0 import graphlets as GL
    from grail0.controls import wl_labels
    gb = _graphs_by_gid(name)
    GL.dataset_features(name, gb, n_jobs=n_jobs)
    os.makedirs(WL_DIR, exist_ok=True)
    p = osp.join(WL_DIR, name + ".pkl")
    if not osp.exists(p):
        t = time.time()
        gids = sorted(gb)
        labs, off = wl_labels([gb[i] for i in gids], h=3)
        tmp = p + ".tmp%d" % os.getpid()
        with open(tmp, "wb") as f:
            pickle.dump({"gids": gids, "labels": labs, "off": off}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)
        print("[wl] %s: %d graphs %.1fs" % (name, len(gids), time.time() - t), flush=True)


def run_fold(name, k, threads=4):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=threads):
        return _run_fold(name, k)


def _run_fold(name, k):
    from common.data import load_folds
    from common.metrics import evaluate
    from grail0 import api
    from grail0 import graphlets as GL
    from grail0.controls import run_controls
    out_path = osp.join(PARTS, "%s__fold%d.csv" % (name, k))
    if osp.exists(out_path):
        return out_path, None
    t0 = time.time()
    fold = load_folds(name, folds=[k])[0]
    with open(osp.join(GL.CACHE_DIR, name + ".pkl"), "rb") as f:
        cache = pickle.load(f)
    with open(osp.join(WL_DIR, name + ".pkl"), "rb") as f:
        wlc = pickle.load(f)
    feats = cache["feats"]
    engine, kmax = cache["engine"], cache["kmax"]
    meta = fold["meta"]; seed = int(meta["seed"]); y = fold["y_test"]
    L = meta.get("node_label_dim") if fold["train"][0]["node_label"] is not None else None
    Ftr = [feats[g["gid"]] for g in fold["train"]]
    Fte = [feats[g["gid"]] for g in fold["test"]]
    rows = []
    info = {"dataset": name, "fold": k, "seed": seed, "engine": engine, "kmax": kmax,
            "n_train": len(Ftr), "n_test": len(Fte), "n_anom": int(y.sum()), "node_labels": L}

    def add(method, s, p=None):
        r = evaluate(s, y)
        cf = float(np.mean(p[y == 0] <= 0.05)) if p is not None else float("nan")
        rows.append([name, k, method, r["auroc"], r["auprc"], r["fpr95"], cf])

    t = time.time()
    model = api.fit(fold["train"], engine=engine, kmax=kmax, seed=seed, n_labels=L, feats=Ftr)
    sc = api.score(model, fold["test"], feats=Fte)
    info["grail0_seconds"] = time.time() - t
    for key, meth in GRAIL_METHODS.items():
        if key in sc:
            add(meth, sc[key], sc.get("p_" + key))
    # controls
    t = time.time()
    from grail0.controls import wl_sparse
    pos = {g: i for i, g in enumerate(wlc["gids"])}
    wl = (wl_sparse(wlc["labels"], wlc["off"]), [pos[g["gid"]] for g in fold["train"]],
          [pos[g["gid"]] for g in fold["test"]])
    ctr = run_controls(Ftr, Fte, fold["train"], fold["test"], seed, L=L, wl=wl)
    info["controls_seconds"] = time.time() - t
    for meth, s in ctr.items():
        add(meth, s)
    # secondary variant: continuous attributes / Tox21 atom types (features='all'; NOT comparable
    # to published UB-GOLD numbers)
    if name in ATTR_DATASETS:
        fa = load_folds(name, features="all", folds=[k])[0]
        La = fa["meta"].get("node_label_dim") if fa["train"][0]["node_label"] is not None else None
        ma = api.fit(fa["train"], engine=engine, kmax=kmax, seed=seed, n_labels=La, feats=Ftr)
        sa = api.score(ma, fa["test"], feats=Fte)
        add("grail0attr_sumz", sa["sum_z"], sa["p_sum_z"])
        if "C" in sa:
            add("grail0attr_ch_C", sa["C"], sa["p_C"])
        info["attr_dims"] = None if fa["train"][0]["x"] is None else int(fa["train"][0]["x"].shape[1])
    info["seconds"] = time.time() - t0
    df = pd.DataFrame(rows, columns=COLS)
    os.makedirs(PARTS, exist_ok=True)
    tmp = out_path + ".tmp%d" % os.getpid()
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    with open(out_path.replace(".csv", ".json"), "w") as f:
        json.dump(info, f, indent=1)
    return out_path, info


TU12 = ["MUTAG", "PROTEINS_full", "DD", "ENZYMES", "DHFR", "BZR", "COX2", "AIDS",
        "IMDB-BINARY", "NCI1", "COLLAB", "REDDIT-BINARY"]


def aggregate():
    files = sorted(glob.glob(osp.join(PARTS, "*__fold*.csv")))
    if not files:
        print("no parts"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    from common.data import DATASET_ORDER
    order = {d: i for i, d in enumerate(DATASET_ORDER)}
    df["_o"] = df["dataset"].map(order)
    df = df.sort_values(["_o", "fold", "method"]).drop(columns="_o")
    df.to_csv(osp.join(RES, "grail0_folds.csv"), index=False)
    rows = []
    for (ds, meth), g in df.groupby(["dataset", "method"], sort=False):
        r = {"dataset": ds, "method": meth, "n_folds": len(g)}
        for c in ["auroc", "auprc", "fpr95", "conformal_fpr_at_0.05"]:
            v = g[c].to_numpy(float)
            mu, sd = np.nanmean(v) if np.isfinite(v).any() else np.nan, np.nanstd(v) if np.isfinite(v).any() else np.nan
            r[c] = "" if not np.isfinite(mu) else "%.2f ± %.2f" % (100 * mu, 100 * sd)
            r[c + "_mean"] = mu; r[c + "_std"] = sd
        rows.append(r)
    S = pd.DataFrame(rows)
    # cross-dataset averages (mean of per-dataset means), only over datasets complete for that method
    avg_rows = []
    for meth, g in S.groupby("method", sort=False):
        for tag, dsets in [("AVG_TU12", TU12), ("AVG_ALL", list(order))]:
            gg = g[g["dataset"].isin(dsets) & (g["n_folds"] == 5)]
            if len(gg) == 0:
                continue
            r = {"dataset": "%s(%d)" % (tag, len(gg)), "method": meth, "n_folds": int(gg["n_folds"].sum())}
            for c in ["auroc", "auprc", "fpr95", "conformal_fpr_at_0.05"]:
                mu = gg[c + "_mean"].mean()
                r[c] = "" if not np.isfinite(mu) else "%.2f" % (100 * mu)
                r[c + "_mean"] = mu; r[c + "_std"] = np.nan
            avg_rows.append(r)
    S = pd.concat([S, pd.DataFrame(avg_rows)], ignore_index=True)
    S["_o"] = S["dataset"].map(lambda d: order.get(d, 99))
    S = S.sort_values(["_o", "dataset", "method"], kind="stable").drop(columns="_o")
    S.to_csv(osp.join(RES, "grail0_summary.csv"), index=False)
    print("wrote %s (%d rows) and grail0_summary.csv (%d rows)" % (osp.join(RES, "grail0_folds.csv"), len(df), len(S)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    from joblib import Parallel, delayed
    for ds in a.datasets:
        t = time.time()
        todo = [k for k in a.folds if not osp.exists(osp.join(PARTS, "%s__fold%d.csv" % (ds, k)))]
        if not todo:
            print("[skip] %s: all folds done" % ds, flush=True); continue
        prepare(ds, a.n_jobs)
        thr = max(1, a.n_jobs // len(todo))
        res = Parallel(n_jobs=min(len(todo), a.n_jobs), backend="loky")(
            delayed(run_fold)(ds, k, thr) for k in todo)
        for p, info in res:
            if info:
                print("[fold] %s fold %d: %.1fs (grail0 %.1fs, controls %.1fs)" %
                      (ds, info["fold"], info["seconds"], info["grail0_seconds"], info["controls_seconds"]), flush=True)
        print("[done] %s %.1fs" % (ds, time.time() - t), flush=True)
    if a.aggregate or a.datasets:
        aggregate()


if __name__ == "__main__":
    main()
