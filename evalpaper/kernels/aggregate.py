"""Aggregate KernelGLAD unit JSONs (parts/) into the result CSVs.

  python -B aggregate.py   (any env with numpy + pandas)

Writes (results/):
  eval_kernels_folds.csv    same columns as eval_folds.csv; protocol in {tuned, default};
                            method in {WL-IF, WL-OCSVM, PK-IF, PK-OCSVM} (fit on normal training graphs)
                            and the same with suffix '@testfit' (UB-GOLD's code path: fit on the test set);
                            auroc/auprc/fpr95 as fractions in [0, 1] (UB-GOLD metric functions' scale)
  eval_kernels_grid.csv     every evaluated configuration x fold (for audit / alternative selections)
  eval_kernels_summary.csv  per (method, protocol, dataset): 5-fold mean/std (ddof 0), chosen config,
                            kernel degeneracy flag
and evalpaper/kernels/tables.md.

'default' = UB-GOLD mymain defaults (WL_iter 5, IF 200 trees / sample ratio 0.5, OCSVM nu 0.1).
'tuned'   = per (method, fit mode, dataset) the configuration of UB-GOLD's KernelGLAD search space
            with the highest MEAN TEST AUROC over the 5 folds (test-selected oracle; ties -> default,
            then smaller WL_iter / n_trees / sample ratio).
"""
import glob
import json
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PRELIM = osp.dirname(osp.dirname(HERE))
RES = osp.join(PRELIM, "results")
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
SECONDARY = ["DD", "COLLAB", "Tox21_p53", "Tox21_MMP"]
DATASETS = UB9 + SECONDARY
DEF = dict(WL_iter=5, IF_n_trees=200, IF_sample_ratio=0.5, nuOCSVM=0.1)


def load_grid():
    rows = []
    for p in sorted(glob.glob(osp.join(HERE, "parts", "*.json"))):
        d = json.load(open(p))
        for r in d["results"]:
            det = r["detector"]
            name = f"{d['kernel']}-{det}" + ("" if d["mode"] == "train" else "@testfit")
            rows.append({"method": name, "kernel": d["kernel"], "detector": det, "fit": d["mode"],
                         "dataset": d["dataset"], "fold": d["fold"], "WL_iter": d["WL_iter"],
                         "IF_n_trees": r.get("IF_n_trees", np.nan), "IF_sample_ratio": r.get("IF_sample_ratio", np.nan),
                         "nuOCSVM": r.get("nuOCSVM", np.nan), "auroc": r["auroc"], "auprc": r["auprc"],
                         "fpr95": r["fpr95"], "score_std": r["score_std"], "n_fit": d["n_fit"], "n_test": d["n_test"],
                         "kernel_offdiag_std": d["kernel_offdiag_std"], "kernel_seconds": d["kernel_seconds"],
                         "seconds": r["seconds"], "host": d.get("host"), "job": d.get("job")})
    return pd.DataFrame(rows)


def cfg_key(r):
    if r["detector"] == "IF":
        return f"WL_iter={int(r['WL_iter'])},IF_n_trees={int(r['IF_n_trees'])},IF_sample_ratio={r['IF_sample_ratio']}"
    return f"WL_iter={int(r['WL_iter'])},nuOCSVM={r['nuOCSVM']}"


def is_default(r):
    if int(r["WL_iter"]) != DEF["WL_iter"]:
        return False
    if r["detector"] == "IF":
        return int(r["IF_n_trees"]) == DEF["IF_n_trees"] and abs(r["IF_sample_ratio"] - DEF["IF_sample_ratio"]) < 1e-9
    return abs(r["nuOCSVM"] - DEF["nuOCSVM"]) < 1e-9


def main():
    g = load_grid()
    if g.empty:
        raise SystemExit("no parts")
    g["config"] = g.apply(cfg_key, axis=1)
    g["is_default"] = g.apply(is_default, axis=1)
    g.to_csv(osp.join(RES, "eval_kernels_grid.csv"), index=False, float_format="%.6f")
    folds, summ = [], []
    for (m, ds), sub in g.groupby(["method", "dataset"]):
        n_cfg_folds = sub.groupby("config").fold.nunique()
        full = n_cfg_folds[n_cfg_folds == 5].index
        means = sub[sub.config.isin(full)].groupby("config").agg(auroc=("auroc", "mean"), is_default=("is_default", "first"),
                                                                 WL_iter=("WL_iter", "first"), nt=("IF_n_trees", "first"),
                                                                 sr=("IF_sample_ratio", "first"))
        choices = {}
        if len(means[means.is_default]):
            choices["default"] = means[means.is_default].index[0]
        # tuned is only defined on the complete search grid (all configs x 5 folds)
        n_expected = 60 if sub.detector.iloc[0] == "IF" else 5
        if len(full) == n_expected:
            mm = means.reset_index()
            mm["neg"] = -mm["auroc"].round(12)
            mm = mm.sort_values(["neg", "is_default", "WL_iter", "nt", "sr"], ascending=[True, False, True, True, True])
            choices["tuned"] = mm.iloc[0]["config"]
        for proto, cfg in choices.items():
            rows = sub[sub.config == cfg].sort_values("fold")
            degenerate = bool((rows.kernel_offdiag_std.fillna(1) < 1e-9).all())
            for r in rows.itertuples():
                hp = {"config": cfg, "fit": r.fit, "kernel": r.kernel, "detector": r.detector,
                      "selection": ("UB-GOLD mymain defaults (= released WL_IF.sh line)" if proto == "default" else
                                    f"max mean test AUROC over {n_expected} configs of UB-GOLD's KernelGLAD search space"),
                      "kernel_constant": degenerate, "PK_bin_width": 1}
                folds.append({"method": m, "dataset": ds, "fold": int(r.fold), "protocol": proto, "auroc": r.auroc,
                              "auprc": r.auprc, "fpr95": r.fpr95, "epoch_selected": "", "device": "cpu",
                              "hp_config": json.dumps(hp, sort_keys=True), "orientation": "ubgold"})
            summ.append({"method": m, "dataset": ds, "protocol": proto, "config": cfg, "n_folds": len(rows),
                         "auroc_mean": 100 * rows.auroc.mean(), "auroc_std_ddof0": 100 * rows.auroc.std(ddof=0),
                         "auprc_mean": 100 * rows.auprc.mean(), "fpr95_mean": 100 * rows.fpr95.mean(),
                         "kernel_constant": degenerate, "n_configs_complete": len(full), "n_configs_expected": n_expected})
    F = pd.DataFrame(folds)
    order = {d: i for i, d in enumerate(DATASETS)}
    F = F.sort_values(["method", "protocol", "dataset", "fold"], key=lambda s: s.map(order) if s.name == "dataset" else s)
    F.to_csv(osp.join(RES, "eval_kernels_folds.csv"), index=False, float_format="%.6f")
    S = pd.DataFrame(summ)
    S.to_csv(osp.join(RES, "eval_kernels_summary.csv"), index=False, float_format="%.4f")
    # tables.md
    L = ["# KernelGLAD tables (auto-generated by aggregate.py; AUROC x 100, 5-fold means)", ""]
    for setname, dss in (("UB9", UB9), ("secondary", SECONDARY)):
        for fitname, suffix in (("fit on normal training graphs (primary)", ""), ("UB-GOLD code path: fit on the test set", "@testfit")):
            sub = S[S.method.str.endswith(suffix) if suffix else ~S.method.str.contains("@")]
            sub = sub[sub.dataset.isin(dss)]
            if sub.empty:
                continue
            w = sub.pivot_table(index=["method", "protocol"], columns="dataset", values="auroc_mean")
            w = w[[d for d in dss if d in w.columns]]
            cnt = sub.pivot_table(index=["method", "protocol"], columns="dataset", values="n_folds")
            w["mean"] = w.mean(axis=1).where(w.notna().all(axis=1))
            L.append(f"## {setname}, {fitname}")
            L.append("")
            L.append("| method | protocol | " + " | ".join(w.columns) + " |")
            L.append("|" + "---|" * (len(w.columns) + 2))
            for (m, p), r in w.iterrows():
                cells = []
                for c in w.columns:
                    v = r[c]
                    flag = ""
                    if c != "mean":
                        k = sub[(sub.method == m) & (sub.protocol == p) & (sub.dataset == c)]
                        if len(k) and bool(k.kernel_constant.iloc[0]):
                            flag = "§"
                    cells.append("–" if pd.isna(v) else f"{v:.2f}{flag}")
                L.append(f"| {m} | {p} | " + " | ".join(cells) + " |")
            L.append("")
    L.append("§ = constant kernel matrix (PK on graphs whose UB-GOLD node features are constant): every graph "
             "gets the same score, AUROC = 50 by construction.")
    L.append("")
    tuned = S[S.protocol == "tuned"][["method", "dataset", "config", "auroc_mean"]]
    L.append("## Test-selected ('tuned') configurations")
    L.append("")
    L.append("| method | dataset | config | AUROC |")
    L.append("|---|---|---|---|")
    for r in tuned.itertuples():
        L.append(f"| {r.method} | {r.dataset} | {r.config} | {r.auroc_mean:.2f} |")
    open(osp.join(HERE, "tables.md"), "w").write("\n".join(L) + "\n")
    exp = 4 * 2 * 13 * 2
    have = S.groupby(["method", "protocol"]).dataset.nunique()
    print(f"grid rows {len(g)}; folds rows {len(F)}; (method, protocol) cells {len(S)} of {exp} "
          f"(complete = 5 folds: {(S.n_folds == 5).sum()})")
    print(have.to_string())


if __name__ == "__main__":
    main()
