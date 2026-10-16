"""Paired comparison of GRAIL-0 / size-only vs the re-run deep baselines on identical UB-GOLD folds,
and ranking stability of the deep methods across protocols.

Writes results/deep_rerun_vs_grail0.csv and prints markdown tables.
GRAIL-0 = grail0_sumz, size-only = ctrl_size_lw (results/grail0_folds.csv).
"""
import os.path as osp

import numpy as np
import pandas as pd

RES = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
F = pd.read_csv(osp.join(RES, "deep_rerun_folds.csv"))
F = F[F.dataset.isin(UB9)]  # MUTAG (not a UB-GOLD dataset) is reported separately
G = pd.read_csv(osp.join(RES, "grail0_folds.csv"))
G = G[G.method.isin(["grail0_sumz", "ctrl_size_lw", "ctrl_wl_iforest", "ctrl_gc_iforest"])]
rng = np.random.default_rng(0)
rows = []
for m in sorted(F.method.unique()):
    Fm = F[F.method == m]
    # datasets with all 5 folds for this method
    full = [d for d, g in Fm[Fm.protocol == "oracle"].groupby("dataset") if g.fold.nunique() == 5]
    for proto in ["oracle", "last", "valnormal"]:
        a = Fm[(Fm.protocol == proto) & Fm.dataset.isin(full)].set_index(["dataset", "fold"]).auroc
        for ref in ["grail0_sumz", "ctrl_size_lw", "ctrl_wl_iforest", "ctrl_gc_iforest"]:
            b = G[(G.method == ref) & G.dataset.isin(full)].set_index(["dataset", "fold"]).auroc
            j = a.index.intersection(b.index)
            diff = (a.loc[j] - b.loc[j]) * 100   # deep minus reference, per (dataset, fold)
            # dataset-level means, then bootstrap over datasets x folds (stratified by dataset)
            per_ds = diff.groupby(level=0).mean()
            boots = []
            groups = [diff.loc[d].values for d in per_ds.index]
            for _ in range(2000):
                boots.append(np.mean([rng.choice(g, len(g)).mean() for g in groups]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(method=m, protocol=proto, reference=ref, n_datasets=len(per_ds), n_pairs=len(j),
                             mean_diff=per_ds.mean(), ci_lo=lo, ci_hi=hi,
                             datasets_deep_wins=int((per_ds > 0).sum()),
                             deep_mean=a.loc[j].groupby(level=0).mean().mean() * 100,
                             ref_mean=b.loc[j].groupby(level=0).mean().mean() * 100))
R = pd.DataFrame(rows)
R.to_csv(osp.join(RES, "deep_rerun_vs_grail0.csv"), index=False)
name = {"grail0_sumz": "GRAIL-0", "ctrl_size_lw": "size-only", "ctrl_wl_iforest": "WL+iForest", "ctrl_gc_iforest": "graphlet+iForest"}
print("| deep method | protocol | reference | #datasets | deep mean | ref mean | deep − ref [95% CI, fold bootstrap] | datasets deep wins |")
print("|---|---|---|---|---|---|---|---|")
for _, r in R.iterrows():
    print(f"| {r.method} | {r.protocol} | {name[r.reference]} | {r.n_datasets} | {r.deep_mean:.2f} | {r.ref_mean:.2f} | "
          f"{r.mean_diff:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] | {r.datasets_deep_wins}/{r.n_datasets} |")

# ranking stability among deep methods on datasets covered by all methods
S = pd.read_csv(osp.join(RES, "deep_rerun_summary.csv"))
S = S[(S.n_folds == 5) & S.protocol.isin(["oracle", "last", "valnormal"]) & S.dataset.isin(UB9)]
meths = sorted(S.method.unique())
common = sorted(set.intersection(*[set(S[S.method == m].dataset) for m in meths])) if meths else []
if len(meths) > 1 and common:
    print(f"\nRanking on {len(common)} datasets covered by all of {meths}:")
    g0 = G.groupby(["method", "dataset"]).auroc.mean().mul(100)
    refs = {"GRAIL-0": "grail0_sumz", "size-only": "ctrl_size_lw"}
    print("| protocol | " + " | ".join(meths) + " | GRAIL-0 | size-only | best method per dataset (deep + GRAIL-0) |")
    print("|---|" + "---|" * (len(meths) + 2) + "---|")
    for p in ["oracle", "last", "valnormal"]:
        piv = S[(S.protocol == p) & S.dataset.isin(common)].pivot(index="dataset", columns="method", values="auroc_mean")
        for k, v in refs.items():
            piv[k] = [g0[(v, d)] for d in piv.index]
        means = piv.mean()
        best = piv.drop(columns=["size-only"]).idxmax(axis=1).value_counts().to_dict()
        print(f"| {p} | " + " | ".join(f"{means[m]:.2f}" for m in meths) +
              f" | {means['GRAIL-0']:.2f} | {means['size-only']:.2f} | {best} |")
    print("datasets:", " ".join(common))
