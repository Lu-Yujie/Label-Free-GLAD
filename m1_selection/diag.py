"""M1 post-verdict diagnostics (descriptive only; not part of the pre-registered rule).
Reads results/m1_selection_folds.csv and m1_corr.csv (written by select.py) and writes
results/m1_selection_diag.csv plus a printed summary used in NOTES.md:
  - where each criterion lands in training (selected epoch / num_epoch; fraction = final epoch),
  - level / saturation of the pseudo-validation AUROC (max over eval points; ties at the max),
  - fraction of runs with positive Spearman(signal, test AUROC),
  - fold-level wins/ties/losses of PSEUDO vs VAL-NORMAL.
  python -B diag.py
"""
import os.path as _osp, sys as _sys  # M1: this directory holds select.py, which would shadow the
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))  # stdlib module `select` (used by
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]  # subprocess): drop it from sys.path
import os.path as osp

import numpy as np
import pandas as pd

RES = osp.join(osp.dirname(_HERE_), "results")
METHODS = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD"]
NUM_EPOCH = {"SIGNET": 1000, "OCGTL": 500, "OCGIN": 500, "GLocalKD": 150}


def main():
    F = pd.read_csv(osp.join(RES, "m1_selection_folds.csv"))
    C = pd.read_csv(osp.join(RES, "m1_corr.csv"))
    rows = []
    for m in METHODS:
        r = dict(method=m)
        for c in ["oracle", "valnormal", "pseudo", "consensus", "aucp"]:
            g = F[(F.criterion == c) & (F.method == m)]
            rel = g.epoch_selected / NUM_EPOCH[m]
            r[f"{c}_median_rel_epoch"] = float(rel.median())
            r[f"{c}_frac_final_epoch"] = float((g.epoch_selected == NUM_EPOCH[m]).mean())
        g = C[C.method == m]
        r["pauc_mean_max_median"] = float(g.pauc_mean_at_pseudo.median())
        r["frac_runs_pauc_max_ge_99"] = float((g.pauc_mean_at_pseudo >= 99).mean())
        r["frac_runs_pauc_max_lt_60"] = float((g.pauc_mean_at_pseudo < 60).mean())
        r["median_ties_at_pauc_max"] = float(g.n_pauc_ties_at_max.median())
        r["frac_rho_pseudo_pos"] = float((g.spearman_pauc_mean_vs_auroc > 0).mean())
        r["frac_rho_neg_holdout_pos"] = float((g.spearman_neg_holdout_vs_auroc > 0).mean())
        r["frac_rho_aucp_pos"] = float((g.spearman_aucp_vs_auroc > 0).mean())
        w = F[(F.method == m) & F.criterion.isin(["pseudo", "valnormal"])].pivot_table(
            index=["dataset", "fold"], columns="criterion", values="auroc")
        d = w.pseudo - w.valnormal
        r.update(fold_wins=int((d > 0).sum()), fold_ties=int((d == 0).sum()), fold_losses=int((d < 0).sum()),
                 fold_diff_min=float(d.min()), fold_diff_max=float(d.max()))
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(osp.join(RES, "m1_selection_diag.csv"), index=False)
    pd.set_option("display.width", 250)
    print(D.T.to_string())
    print("\nmedian max pseudo-AUROC per dataset:",
          C.groupby("dataset").pauc_mean_at_pseudo.median().round(1).to_dict())
    print("devices:", F.drop_duplicates(["method", "dataset", "fold"]).device.value_counts().to_dict())


if __name__ == "__main__":
    main()
