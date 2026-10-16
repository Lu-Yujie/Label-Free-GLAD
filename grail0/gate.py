"""Gate-1 numbers for GRAIL-0, computed only from results/grail0_summary.csv.

  cd <repo> && PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -m grail0.gate

Published reference numbers (AUROC x100, 5 seeds, UB-GOLD protocol; copied from
research_notes/.../sota_glad.md, table B: SIGNET = UB-GOLD Table 8 values as reused by LGKDE;
LGKDE = LGKDE ICML'26 Table 1). NOTE: UB-GOLD baselines were hyper-parameter-tuned on the
test set (verify_sota_glad.md #1); GRAIL-0 uses no labels at all.
"""
import os.path as osp

import numpy as np
import pandas as pd

RES = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
TU12 = ["MUTAG", "PROTEINS_full", "DD", "ENZYMES", "DHFR", "BZR", "COX2", "AIDS",
        "IMDB-BINARY", "NCI1", "COLLAB", "REDDIT-BINARY"]
SIGNET = dict(zip(TU12, [88.84, 75.86, 74.53, 63.12, 72.87, 80.79, 72.35, 97.60, 70.12, 74.32, 72.45, 85.24]))
LGKDE = dict(zip(TU12, [91.63, 78.97, 79.84, 71.04, 82.58, 81.11, 66.69, 99.06, 68.77, 76.67, 67.94, 88.12]))
LABELLED = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "MUTAG", "PROTEINS_full", "DD"]
PILOT8 = ["AIDS", "BZR", "COX2", "DHFR", "NCI1", "MUTAG", "PROTEINS_full", "IMDB-BINARY"]
ALL14 = TU12 + ["Tox21_p53", "Tox21_MMP"]
METHODS = ["grail0_sumz", "grail0_maxz", "grail0_noS0_sumz", "grail0_cauchy", "grail0_stouffer",
           "ch_S0", "ch_A", "ch_B", "ch_C", "ctrl_size_lw", "ctrl_gc_iforest", "ctrl_gc_lw",
           "ctrl_gclab_iforest", "ctrl_gclab_lw", "ctrl_wl_iforest", "grail0attr_sumz", "grail0attr_ch_C"]


def main():
    S = pd.read_csv(osp.join(RES, "grail0_summary.csv"))
    S = S[S["n_folds"] == 5]
    A = S.pivot(index="dataset", columns="method", values="auroc_mean") * 100
    D = S.pivot(index="dataset", columns="method", values="auroc_std") * 100
    lines = []
    lines.append("| dataset | " + " | ".join(METHODS) + " | SIGNET (pub.) | LGKDE (pub.) |")
    lines.append("|---" * (len(METHODS) + 3) + "|")
    for ds in ALL14:
        if ds not in A.index:
            continue
        cells = []
        for m in METHODS:
            v = A.loc[ds].get(m, np.nan)
            cells.append("" if not np.isfinite(v) else "%.1f±%.1f" % (v, D.loc[ds, m]))
        lines.append("| %s | %s | %s | %s |" % (ds, " | ".join(cells),
                                               SIGNET.get(ds, ""), LGKDE.get(ds, "")))

    def avg(m, dsets):
        v = [A.loc[d, m] for d in dsets if d in A.index and m in A.columns and np.isfinite(A.loc[d, m])]
        return (np.mean(v), len(v)) if len(v) == len(dsets) else (np.nan, len(v))

    lines.append("")
    lines.append("| mean AUROC over | " + " | ".join(METHODS) + " | SIGNET | LGKDE |")
    lines.append("|---" * (len(METHODS) + 3) + "|")
    for tag, ds in [("TU12", TU12), ("all 14", ALL14), ("9 labelled", LABELLED), ("pilot-8", PILOT8)]:
        cells = []
        for m in METHODS:
            mu, k = avg(m, ds)
            cells.append("" if not np.isfinite(mu) else "%.2f" % mu)
        sg = [SIGNET[d] for d in ds if d in SIGNET]; lg = [LGKDE[d] for d in ds if d in LGKDE]
        lines.append("| %s | %s | %s | %s |" % (tag, " | ".join(cells),
                                               "%.2f" % np.mean(sg) if len(sg) == len(ds) else "",
                                               "%.2f" % np.mean(lg) if len(lg) == len(ds) else ""))
    print("\n".join(lines))
    print()
    g = "grail0_sumz"
    for tag, ds in [("TU12", TU12), ("all 14", ALL14), ("9 labelled", LABELLED)]:
        mu = avg(g, ds)[0]
        msg = ["%s: GRAIL-0 sum-z %.2f" % (tag, mu)]
        for c in ["ctrl_size_lw", "ctrl_gc_iforest", "ctrl_gc_lw", "ctrl_gclab_iforest", "ctrl_gclab_lw",
                  "ctrl_wl_iforest"]:
            cm = avg(c, ds)[0]
            if np.isfinite(cm):
                wins = sum(A.loc[d, g] > A.loc[d, c] for d in ds)
                msg.append("%s %.2f (delta %+.2f, GRAIL-0 wins %d/%d)" % (c, cm, mu - cm, wins, len(ds)))
        if tag == "TU12":
            msg.append("SIGNET %.2f (delta %+.2f), LGKDE %.2f (delta %+.2f)" % (
                np.mean(list(SIGNET.values())), mu - np.mean(list(SIGNET.values())),
                np.mean(list(LGKDE.values())), mu - np.mean(list(LGKDE.values()))))
        print("- " + "; ".join(msg))
    # size dominance check (judge_advisor Gate-1 (b)) on the pilot-8 set
    r = [(d, A.loc[d, "ctrl_size_lw"] / A.loc[d, g]) for d in PILOT8]
    k = sum(x < 0.95 for _, x in r)
    print("- size-only < 95%% of GRAIL-0 on %d/8 pilot datasets (%s)" % (
        k, ", ".join("%s %.2f" % (d, x) for d, x in r)))
    print("- dropping S0 changes the TU12 mean by %+.2f (sum-z without S0 minus with S0)" % (
        avg("grail0_noS0_sumz", TU12)[0] - avg(g, TU12)[0]))


if __name__ == "__main__":
    main()
