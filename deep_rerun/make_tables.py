"""Print markdown tables for NOTES.md from results/deep_rerun_*.csv (run aggregate.py first)."""
import os.path as osp

import numpy as np
import pandas as pd

RES = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
S = pd.read_csv(osp.join(RES, "deep_rerun_summary.csv"))
A = pd.read_csv(osp.join(RES, "deep_rerun_averages.csv"))
F = pd.read_csv(osp.join(RES, "deep_rerun_folds.csv"))
DS = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY", "MUTAG"]
SHORT = {"PROTEINS_full": "PROTEINS", "IMDB-BINARY": "IMDB-B", "REDDIT-BINARY": "REDDIT-B"}

for m in ["SIGNET", "OCGTL", "OCGIN", "GLocalKD"]:
    Sm = S[S.method == m]
    if Sm.empty:
        continue
    print(f"\n#### {m}\n")
    print("| dataset | folds | published (UB-GOLD) | ORACLE (ours) | ORACLE − published | LAST | VAL-NORMAL | ORACLE − LAST | ORACLE − VAL-NORMAL | mean epoch: oracle / valnormal | max-over-run (no early stop) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in DS:
        g = Sm[Sm.dataset == d].set_index("protocol")
        if g.empty:
            continue
        o, l, v = g.loc["oracle"], g.loc["last"], g.loc["valnormal"]
        on = g.loc["oracle_nostop"] if "oracle_nostop" in g.index else None
        pub = o.published_auroc
        print(f"| {SHORT.get(d, d)} | {int(o.n_folds)} | {pub:.2f} | {o.auroc} | {o.auroc_mean - pub:+.2f} | {l.auroc} | {v.auroc} | "
              f"{o.auroc_mean - l.auroc_mean:+.2f} | {o.auroc_mean - v.auroc_mean:+.2f} | "
              f"{o.epoch_selected_mean:.0f} / {v.epoch_selected_mean:.0f} | {on.auroc if on is not None else ''} |")

print("\n#### Averages over the datasets each method covers (5/5 folds), AUROC %\n")
print("| method | dataset set | #datasets | published | ORACLE | LAST | VAL-NORMAL | ORACLE − published | ORACLE − LAST | ORACLE − VAL-NORMAL | GRAIL-0 (same datasets) | size-only (same datasets) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in A.iterrows():
    print(f"| {r.method} | {r.dataset_set} | {r.n_datasets} | {r.published:.2f} | {r.oracle:.2f} | {r['last']:.2f} | {r.valnormal:.2f} | "
          f"{r.oracle_minus_published:+.2f} | {r.oracle_minus_last:+.2f} | {r.oracle_minus_valnormal:+.2f} | "
          f"{r.grail0_same_datasets:.2f} | {r.size_only_same_datasets:.2f} |")

# fold-level paired differences (oracle - label-free)
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
F = F[F.dataset.isin(UB9)]
print("\n#### Fold-level oracle − label-free differences (AUROC points; all folds of the 9 UB-GOLD datasets pooled)\n")
print("| method | n runs | oracle − last: mean [min, max] | oracle − valnormal: mean [min, max] | runs where valnormal ≥ last |")
print("|---|---|---|---|---|")
for m in ["SIGNET", "OCGTL", "OCGIN", "GLocalKD"]:
    Fm = F[F.method == m]
    if Fm.empty:
        continue
    P = Fm.pivot_table(index=["dataset", "fold"], columns="protocol", values="auroc") * 100
    dl = P["oracle"] - P["last"]
    dv = P["oracle"] - P["valnormal"]
    print(f"| {m} | {len(P)} | {dl.mean():.2f} [{dl.min():.2f}, {dl.max():.2f}] | {dv.mean():.2f} [{dv.min():.2f}, {dv.max():.2f}] | "
          f"{int((P['valnormal'] >= P['last']).sum())}/{len(P)} |")

# SIGNET trained on UB-GOLD's exact training set (no 20% carve-out; tag 'full')
fu = osp.join(RES, "deep_rerun_fulldata_summary.csv")
if osp.exists(fu):
    U = pd.read_csv(fu)
    print("\n#### SIGNET on UB-GOLD's exact training set (100% of normal training graphs; ORACLE/LAST only)\n")
    print("| dataset | published | ORACLE (100%) | LAST (100%) | ORACLE − LAST (100%) | ORACLE: 100% − 80% | LAST: 100% − 80% | max-over-run (100%) |")
    print("|---|---|---|---|---|---|---|---|")
    av = {"pub": [], "o": [], "l": [], "o80": [], "l80": [], "n": []}
    for d in UB9:
        g = U[(U.method == "SIGNET") & (U.dataset == d)].set_index("protocol")
        if g.empty or g.loc["oracle"].n_folds < 5:
            continue
        o, l, n = g.loc["oracle"], g.loc["last"], g.loc["oracle_nostop"]
        print(f"| {SHORT.get(d, d)} | {o.published_auroc:.2f} | {o.auroc} | {l.auroc} | {o.auroc_mean - l.auroc_mean:+.2f} | "
              f"{o.full_minus_main80:+.2f} | {l.full_minus_main80:+.2f} | {n.auroc} |")
        av["pub"].append(o.published_auroc); av["o"].append(o.auroc_mean); av["l"].append(l.auroc_mean)
        av["o80"].append(o.main80_auroc_mean); av["l80"].append(l.main80_auroc_mean); av["n"].append(n.auroc_mean)
    if av["o"]:
        m_ = {k: np.mean(v) for k, v in av.items()}
        print(f"| **mean ({len(av['o'])})** | {m_['pub']:.2f} | {m_['o']:.2f} | {m_['l']:.2f} | {m_['o'] - m_['l']:+.2f} | "
              f"{m_['o'] - m_['o80']:+.2f} | {m_['l'] - m_['l80']:+.2f} | {m_['n']:.2f} |")
