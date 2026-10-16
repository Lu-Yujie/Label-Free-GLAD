"""DIAGNOSTIC ONLY (uses test labels). Summarise results/diag/*.json (labelled_selection_diag.py) with MUSE's own
main.py selection (lines 147-175), the test set standing in for MUSE's validation set:
  sel_main_py   : per fold and snapshot the best MLP cell (strict '>' over the cell loop order), then ONE snapshot
                  for all folds = argmax of the fold-mean; reported = fold-mean test AUROC at that snapshot.
  es_main_cell  : MLP cell fixed to the adapter's main cell (h=32, lr=1e-3) but early-stopped on test AUROC;
                  snapshot chosen as in main.py.
Compared with the adapter's protocols (results/muse_repro.csv) and the paper's Table 3. Writes results/diag_summary.csv.
  python -B diag/diag_summary.py
"""
import glob
import json
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
D = osp.join(HERE, "results", "diag")


def main():
    by = {}
    for f in sorted(glob.glob(osp.join(D, "MUSE_*.json"))):
        r = json.load(open(f))
        by.setdefault((r["dataset"], r["hp"]), []).append(r)
    rep = pd.read_csv(osp.join(HERE, "results", "muse_repro.csv")).set_index("dataset")
    rows = []
    for (ds, hp), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["fold"])
        A = np.array([r["test_auroc"] for r in rs])            # folds x snapshots x cells
        main = rs[0]["cells"].index("h32_lr0.001")
        best = A.max(2)                                        # first max over cells == strict '>' loop
        snap = int(np.argmax(best.mean(0)))
        snap_es = int(np.argmax(A[:, :, main].mean(0)))
        row = dict(dataset=ds, hp=hp, folds=len(rs), sel_main_py=100 * best[:, snap].mean(),
                   sel_main_py_std=100 * best[:, snap].std(), sel_snapshot_epoch=rs[0]["epochs"][snap],
                   es_main_cell=100 * A[:, snap_es, main].mean())
        if ds in rep.index:
            rr = rep.loc[ds]
            row.update(paper_t3=rr.paper_t3, r1_muse_own=rr.get("r1_auroc", np.nan))
            if hp == "ubgold":
                row.update(adapter_oracle=rr.oracle_mean, adapter_oracle_grid=rr.oracle_grid_mean,
                           adapter_valnormal=rr.valnormal_mean)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(osp.join(HERE, "results", "diag_summary.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
