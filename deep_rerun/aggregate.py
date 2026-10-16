"""Aggregate run_rerun.py trajectories into the deep-rerun result tables.

Protocols per (method, dataset, fold), all from ONE training run:
  oracle    UB-GOLD's protocol: running max of test AUROC over evaluation epochs
            (epoch % eval_freq == 0), with UB-GOLD's early stopping (stop after 30 consecutive
            evaluations without a strict improvement); AUPRC/FPR95 at that checkpoint.
  last      final epoch (num_epoch) of a fixed budget; label-free.
  valnormal checkpoint with the minimum mean loss (= the method's own anomaly score) on the
            held-out 20% of normal training graphs; label-free. Ties -> earliest.
Extra diagnostics (deep_rerun_folds_extra.csv):
  oracle_nostop  max test AUROC over the whole trajectory (no early stopping)
  trainloss      checkpoint minimizing the mean training loss of the epoch (label-free)

Outputs (results/): deep_rerun_folds.csv, deep_rerun_folds_extra.csv,
deep_rerun_summary.csv, deep_rerun_averages.csv, deep_rerun_trajectories.csv
"""
import glob
import json
import math
import os
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
RES0 = osp.join(osp.dirname(HERE), "results")
RES = os.environ.get("DR_RES", RES0)            # output dir
PARTS = os.environ.get("DR_PARTS", osp.join(RES0, "deep_rerun_parts"))
PATIENCE = 30

# Published AUROC (%) — UB-GOLD Table 8 (arXiv 2406.15523v2), as transcribed in
# research_notes/.../sota_glad.md (12-TU table, copied verbatim by LGKDE/GLASS) and checked in
# verify_sota_glad.md (row 4: the only LGKDE/GLASS deviation is OCGTL REDDIT-B 88.03 vs
# UB-GOLD 89.92; we use UB-GOLD's 89.92). MUTAG is not a UB-GOLD dataset (LGKDE's own runs).
PUBLISHED = {
    "SIGNET": dict(MUTAG=88.84, PROTEINS_full=75.86, DD=74.53, ENZYMES=63.12, DHFR=72.87, BZR=80.79,
                   COX2=72.35, AIDS=97.60, **{"IMDB-BINARY": 70.12}, NCI1=74.32, COLLAB=72.45,
                   **{"REDDIT-BINARY": 85.24}),
    "OCGTL": dict(MUTAG=88.02, PROTEINS_full=72.89, DD=77.76, ENZYMES=63.59, DHFR=59.82, BZR=51.89,
                  COX2=59.81, AIDS=99.36, **{"IMDB-BINARY": 65.27}, NCI1=75.75, COLLAB=48.13,
                  **{"REDDIT-BINARY": 89.92}),
    "OCGIN": dict(MUTAG=79.55, PROTEINS_full=76.46, DD=79.08, ENZYMES=62.44, DHFR=61.09, BZR=69.13,
                  COX2=57.81, AIDS=96.89, **{"IMDB-BINARY": 61.47}, NCI1=69.46, COLLAB=60.58,
                  **{"REDDIT-BINARY": 82.10}),
    "GLocalKD": dict(MUTAG=86.25, PROTEINS_full=77.29, DD=80.76, ENZYMES=61.75, DHFR=61.79, BZR=68.55,
                     COX2=58.93, AIDS=96.93, **{"IMDB-BINARY": 53.31}, NCI1=65.29, COLLAB=51.85,
                     **{"REDDIT-BINARY": 80.32}),
}
DS_ORDER = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY",
            "REDDIT-BINARY", "MUTAG", "DD", "COLLAB"]
METHODS = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD"]


def pick(traj, key, mode):
    vals = np.array([r.get(key, np.nan) for r in traj], dtype=float)
    if np.all(np.isnan(vals)):
        return None
    return int(np.nanargmin(vals) if mode == "min" else np.nanargmax(vals))


def ubgold_oracle(traj, eval_freq):
    """Replicates the detectors' fit(): max_AUC=0; save if val_auc > max_AUC; patience 30."""
    max_auc, best, counter, stop_epoch = 0.0, None, 0, None
    for i, r in enumerate(traj):
        if r["epoch"] % eval_freq != 0:
            continue
        a = r["auroc"]
        if a == a and a > max_auc:  # NaN never improves (UB-GOLD would crash on NaN scores)
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE:
            stop_epoch = r["epoch"]
            break
    return best, stop_epoch


def row(meta, proto, r, extra=None):
    d = dict(method=meta["method"], dataset=meta["dataset"], fold=meta["fold"], protocol=proto,
             auroc=r["auroc"], auprc=r["auprc"], fpr95=r["fpr95"], epoch_selected=r["epoch"])
    if extra:
        d.update(extra)
    return d


def main():
    files = sorted(glob.glob(osp.join(PARTS, "*.json")))
    rows, extra_rows, traj_rows, hp_rows = [], [], [], []
    full_rows = []
    for fn in files:
        J = json.load(open(fn))
        meta, traj = J["meta"], J["traj"]
        tag = meta["run_id"].replace(f"{meta['method']}_{meta['dataset']}_fold{meta['fold']}", "").lstrip("_")
        if tag == "full":  # exact UB-GOLD training set (no held-out carve-out): oracle/last only
            hp = meta["hparams"]
            b, stop_epoch = ubgold_oracle(traj, int(hp["eval_freq"]))
            last = [r for r in traj if r["epoch"] == int(hp["num_epoch"])][0]
            o2 = pick(traj, "auroc", "max")
            for proto, r in (("oracle", traj[b]), ("last", last), ("oracle_nostop", traj[o2])):
                full_rows.append(row(meta, proto, r, dict(device=meta["device"])))
            continue
        if tag:
            continue  # other tagged (smoke/variant) runs are not part of the main table
        hp = meta["hparams"]
        ef, ne = int(hp["eval_freq"]), int(hp["num_epoch"])
        b, stop_epoch = ubgold_oracle(traj, ef)
        last = [r for r in traj if r["epoch"] == ne][0]
        v = pick(traj, "holdout_loss", "min")
        info = dict(ubgold_stop_epoch=stop_epoch if stop_epoch is not None else ne)
        rows.append(row(meta, "oracle", traj[b], info))
        rows.append(row(meta, "last", last))
        rows.append(row(meta, "valnormal", traj[v]))
        o2 = pick(traj, "auroc", "max")
        extra_rows.append(row(meta, "oracle_nostop", traj[o2]))
        t = pick(traj, "train_loss", "min")
        extra_rows.append(row(meta, "trainloss", traj[t]))
        for r in traj:
            traj_rows.append(dict(method=meta["method"], dataset=meta["dataset"], fold=meta["fold"], **r))
        hp_rows.append(dict(method=meta["method"], dataset=meta["dataset"], fold=meta["fold"],
                            ubgold_sh_line=meta["ubgold_sh_line"], n_fit=meta["n_fit"], n_holdout=meta["n_holdout"],
                            n_train_normal_ubgold=meta["n_train_normal_ubgold"], device=meta["device"],
                            torch=meta["torch"], scatter_impl=meta["scatter_impl"], seconds=round(meta["seconds"], 1),
                            **{k: hp[k] for k in ("num_epoch", "lr", "batch_size", "hidden_dim", "num_layer", "eval_freq")
                               if k in hp},
                            **{k: hp[k] for k in ("encoder_layers", "readout", "output_dim", "num_trans") if k in hp}))
    if not rows:
        print("no runs yet")
        return
    F = pd.DataFrame(rows)
    E = pd.DataFrame(extra_rows)
    T = pd.DataFrame(traj_rows)
    H = pd.DataFrame(hp_rows)
    key = lambda df: df.assign(_d=df.dataset.map({d: i for i, d in enumerate(DS_ORDER)}),
                               _m=df.method.map({m: i for i, m in enumerate(METHODS)}))
    F = key(F).sort_values(["_m", "_d", "fold", "protocol"]).drop(columns=["_d", "_m"])
    E = key(E).sort_values(["_m", "_d", "fold", "protocol"]).drop(columns=["_d", "_m"])
    F.to_csv(osp.join(RES, "deep_rerun_folds.csv"), index=False)
    E.to_csv(osp.join(RES, "deep_rerun_folds_extra.csv"), index=False)
    T.to_csv(osp.join(RES, "deep_rerun_trajectories.csv"), index=False)
    # how informative is the label-free signal? Spearman(-holdout_loss, test AUROC) over eval points
    from scipy.stats import spearmanr
    C = []
    for (m, ds, f), g in T.groupby(["method", "dataset", "fold"]):
        g = g.dropna(subset=["holdout_loss", "auroc"])
        rho = spearmanr(-g.holdout_loss, g.auroc).correlation if len(g) > 2 else np.nan
        rho_t = spearmanr(-g.train_loss, g.auroc).correlation if len(g) > 2 else np.nan
        C.append(dict(method=m, dataset=ds, fold=f, n_points=len(g), spearman_neg_holdout_loss_vs_auroc=rho,
                      spearman_neg_train_loss_vs_auroc=rho_t))
    C = pd.DataFrame(C)
    C.to_csv(osp.join(RES, "deep_rerun_signal_corr.csv"), index=False)
    H.to_csv(osp.join(RES, "deep_rerun_hparams.csv"), index=False)

    # ---- summary: mean ± std over folds (std = np.std, ddof=0, as common.metrics.summarize) ----
    allF = pd.concat([F.assign(table="main"), E.assign(table="extra")])
    S = []
    for (m, ds, p), g in allF.groupby(["method", "dataset", "protocol"]):
        d = dict(method=m, dataset=ds, protocol=p, n_folds=len(g))
        for k in ("auroc", "auprc", "fpr95"):
            x = 100 * g[k].astype(float).values
            d[k] = f"{np.mean(x):.2f} ± {np.std(x):.2f}"
            d[k + "_mean"] = float(np.mean(x))
            d[k + "_std"] = float(np.std(x))
        d["epoch_selected_mean"] = float(g.epoch_selected.mean())
        pub = PUBLISHED.get(m, {}).get(ds)
        d["published_auroc"] = pub
        d["auroc_minus_published"] = (d["auroc_mean"] - pub) if pub is not None else None
        S.append(d)
    S = key(pd.DataFrame(S))
    S["_p"] = S.protocol.map({"oracle": 0, "last": 1, "valnormal": 2, "oracle_nostop": 3, "trainloss": 4})
    S = S.sort_values(["_m", "_d", "_p"]).drop(columns=["_d", "_m", "_p"])
    # oracle-minus-label-free gaps (per method, dataset)
    piv = S.pivot_table(index=["method", "dataset"], columns="protocol", values="auroc_mean")
    gaps = {}
    for (m, ds), r in piv.iterrows():
        gaps[(m, ds)] = dict(gap_oracle_minus_last=r.get("oracle", np.nan) - r.get("last", np.nan),
                             gap_oracle_minus_valnormal=r.get("oracle", np.nan) - r.get("valnormal", np.nan))
    S["gap_oracle_minus_last"] = [gaps[(m, d)]["gap_oracle_minus_last"] for m, d in zip(S.method, S.dataset)]
    S["gap_oracle_minus_valnormal"] = [gaps[(m, d)]["gap_oracle_minus_valnormal"] for m, d in zip(S.method, S.dataset)]
    S.to_csv(osp.join(RES, "deep_rerun_summary.csv"), index=False)

    # ---- averages over the datasets each method covers with all 5 folds ----
    g0 = pd.read_csv(osp.join(RES0, "grail0_summary.csv"))
    g0 = g0[g0.method.isin(["grail0_sumz", "ctrl_size_lw"])].set_index(["method", "dataset"])["auroc_mean"]
    A = []
    UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
    for m, (setname, allowed) in [(m, x) for m in METHODS for x in (("UB-GOLD datasets", UB9),
                                                                    ("UB-GOLD datasets + MUTAG", UB9 + ["MUTAG"]))]:
        Sm = S[(S.method == m) & (S.protocol == "oracle") & (S.n_folds == 5)]
        dsets = [d for d in DS_ORDER if d in set(Sm.dataset) and d in allowed]
        if not dsets or (setname.endswith("MUTAG") and "MUTAG" not in dsets):
            continue
        d = dict(method=m, dataset_set=setname, n_datasets=len(dsets), datasets=" ".join(dsets))
        for p in ("oracle", "last", "valnormal", "oracle_nostop", "trainloss"):
            v = S[(S.method == m) & (S.protocol == p) & S.dataset.isin(dsets)].set_index("dataset").auroc_mean
            d[p] = float(v.reindex(dsets).mean())
        d["published"] = float(np.mean([PUBLISHED[m][x] for x in dsets]))
        d["oracle_minus_published"] = d["oracle"] - d["published"]
        d["oracle_minus_last"] = d["oracle"] - d["last"]
        d["oracle_minus_valnormal"] = d["oracle"] - d["valnormal"]
        d["grail0_same_datasets"] = float(np.mean([100 * g0[("grail0_sumz", x)] for x in dsets]))
        d["size_only_same_datasets"] = float(np.mean([100 * g0[("ctrl_size_lw", x)] for x in dsets]))
        A.append(d)
    A = pd.DataFrame(A)
    A.to_csv(osp.join(RES, "deep_rerun_averages.csv"), index=False)

    # ---- exact-UB-GOLD-training-set runs (tag 'full') vs the 80%-trained main runs ----
    if full_rows:
        FU = pd.DataFrame(full_rows)
        FU.to_csv(osp.join(RES, "deep_rerun_fulldata_folds.csv"), index=False)
        U = []
        for (m, ds, p), g in FU.groupby(["method", "dataset", "protocol"]):
            x = 100 * g.auroc.values
            main_ = S[(S.method == m) & (S.dataset == ds) & (S.protocol == p)]
            U.append(dict(method=m, dataset=ds, protocol=p, n_folds=len(g), auroc=f"{x.mean():.2f} ± {x.std():.2f}",
                          auroc_mean=x.mean(), auroc_std=x.std(), epoch_selected_mean=g.epoch_selected.mean(),
                          published_auroc=PUBLISHED.get(m, {}).get(ds),
                          main80_auroc_mean=float(main_.auroc_mean.iloc[0]) if len(main_) else np.nan))
        U = pd.DataFrame(U)
        U["minus_published"] = U.auroc_mean - U.published_auroc
        U["full_minus_main80"] = U.auroc_mean - U.main80_auroc_mean
        U.to_csv(osp.join(RES, "deep_rerun_fulldata_summary.csv"), index=False)
        print(U[["method", "dataset", "protocol", "n_folds", "auroc", "published_auroc", "minus_published",
                 "full_minus_main80"]].round(2).to_string(index=False))

    # ---- CPU-vs-GPU check (SIGNET BZR/COX2 runs finished on the CPU node before we switched) ----
    cpu = sorted(glob.glob(osp.join(HERE, "cpu_parts", "*.json")))
    if cpu:
        C = []
        for fn in cpu:
            J = json.load(open(fn))
            meta, traj = J["meta"], J["traj"]
            hp = meta["hparams"]
            b, _ = ubgold_oracle(traj, int(hp["eval_freq"]))
            last = [r for r in traj if r["epoch"] == int(hp["num_epoch"])][0]
            v = pick(traj, "holdout_loss", "min")
            for proto, r in (("oracle", traj[b]), ("last", last), ("valnormal", traj[v])):
                C.append(row(meta, proto, r, dict(device="cpu")))
        C = pd.DataFrame(C)
        G_ = F.merge(C, on=["method", "dataset", "fold", "protocol"], suffixes=("_gpu", "_cpu"))
        G_.to_csv(osp.join(RES, "deep_rerun_cpu_vs_gpu.csv"), index=False)
        print("\nCPU vs GPU (same seeds/folds), mean AUROC %:")
        print((G_.groupby(["dataset", "protocol"])[["auroc_gpu", "auroc_cpu"]].mean() * 100).round(2).to_string())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(S[S.protocol.isin(["oracle", "last", "valnormal"])][
        ["method", "dataset", "protocol", "n_folds", "auroc", "epoch_selected_mean", "published_auroc",
         "auroc_minus_published"]].to_string(index=False))
    print()
    print(A.drop(columns=["datasets"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
