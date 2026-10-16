"""Aggregate finished MUSE runs (ext/parts/MUSE_*.json) into ext/MUSE/results/ (never into prelim/results).

Protocol rows come from the harness's own derivation (prelim/evalpaper/harness/collect.py: run_rows, imported
read-only), so ORACLE / LAST / VAL-NORMAL are computed exactly as for the frozen 12 detectors:
  results/muse_folds.csv   eval_folds.csv schema (oracle, last, valnormal, hpfree_*; orientation native)
  results/muse_extra.csv   oracle_nostop, valnormal_obj (harness) + oracle_grid (max test AUROC over
                           snapshots x 9 MLP cells; labelled upper bound of MUSE's validation selection)
  results/muse_grid.csv    per MLP cell: oracle (UB-GOLD rule), last, valnormal (cell's own held-out mean)
  results/muse_coverage.csv, results/muse_repro.csv (R2: ORACLE ubgold vs paper Table 3; R1 from job logs)
    python -B collect_muse.py
"""
import glob
import importlib.util
import json
import os
import os.path as osp
import re
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PARTS = os.environ.get("EP_PARTS", osp.join(osp.dirname(HERE), "parts"))
RES = osp.join(HERE, "results")
HARNESS_COLLECT = osp.join(osp.dirname(osp.dirname(HERE)), "harness", "collect.py")
DATASETS = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY", "MUTAG"]
# MUSE paper (arXiv 2410.20366 / NeurIPS'24), MUSE row, test AUROC x100 (std). Table 3: class 0 anomalous
# (= UB-GOLD's direction, MUSE anom_type 0); Table 4: class 1 anomalous; Table 1: mean of both.
# 'Protein' = PROTEINS (same graphs as PROTEINS_full); MUSE's 'MUTAG' is Mutagenicity (not comparable).
PAPER = {  # dataset: (table3, table3_std, table4, table1)
    "AIDS": (100.0, 0.0, 99.3, 99.7), "BZR": (68.2, 11.3, 70.8, 69.5), "DHFR": (64.3, 8.0, 63.2, 63.8),
    "NCI1": (72.5, 0.3, 69.6, 71.1), "PROTEINS_full": (78.7, 2.2, 78.0, 78.4), "IMDB-BINARY": (69.6, 3.7, 68.7, 69.2),
    "REDDIT-BINARY": (84.7, 4.4, 72.1, 78.4)}
R1_KEY = {"aids": "AIDS", "bzr": "BZR", "dhfr": "DHFR", "nci1": "NCI1", "imdb": "IMDB-BINARY", "reddit": "REDDIT-BINARY"}


def harness():
    spec = importlib.util.spec_from_file_location("ep_collect", HARNESS_COLLECT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load():
    out = []
    for fn in sorted(glob.glob(osp.join(PARTS, "MUSE_*.json"))):
        J = json.load(open(fn))
        m = J["meta"]
        stem = f"MUSE_{m['dataset']}_fold{m['fold']}"
        if m["run_id"] not in (stem, stem + "_hpfree") or m.get("max_epochs_override") is not None:
            continue
        out.append(J)
    return out


def r1_results():
    rows = []
    for fn in sorted(glob.glob(osp.join(HERE, "logs", "r1_*_*.log"))):
        txt = open(fn).read()
        d = re.search(r"-data (\w+)", txt)
        a = re.search(r"Average of AUROC: ([0-9.]+) \| STD of AUROC: ([0-9.]+)", txt)
        if d and a:
            rows.append(dict(dataset=R1_KEY.get(d.group(1), d.group(1)), r1_auroc=100 * float(a.group(1)),
                             r1_std=100 * float(a.group(2)), r1_log=osp.basename(fn)))
    return pd.DataFrame(rows)


def main():
    H = harness()
    os.makedirs(RES, exist_ok=True)
    runs = load()
    rows, extra, grid = [], [], []
    for J in runs:
        r, e = H.run_rows(J, "ext_muse")
        rows += r
        extra += e
        meta, traj = J["meta"], J["traj"]
        pre = "" if meta["hp"] == "ubgold" else "hpfree_"
        base = dict(method="MUSE", dataset=meta["dataset"], fold=int(meta["fold"]), run_id=meta["run_id"],
                    orientation="native", source="ext_muse")
        best = max(((c, i, traj[i]["grid"][c]["auroc"]) for i in range(len(traj)) for c in traj[i]["grid"]),
                   key=lambda t: t[2])
        extra.append(dict(base, protocol=pre + "oracle_grid", auroc=100 * best[2],
                          auprc=100 * traj[best[1]]["grid"][best[0]]["auprc"],
                          fpr95=100 * traj[best[1]]["grid"][best[0]]["fpr95"],
                          epoch_selected=traj[best[1]]["epoch"], mlp_cell=best[0]))
        ef = int(meta["hparams"]["eval_freq"])
        for c in traj[0]["grid"]:
            sub = [dict(r["grid"][c], epoch=r["epoch"]) for r in traj]
            b, _ = H.ubgold_oracle(sub, ef, "auroc")
            v = H.pick(sub, "holdout_score", "min")
            for proto, i in (("oracle", b), ("last", len(sub) - 1), ("valnormal", v)):
                grid.append(dict(base, hp=meta["hp"], mlp_cell=c, protocol=proto, auroc=100 * sub[i]["auroc"],
                                 auprc=100 * sub[i]["auprc"], fpr95=100 * sub[i]["fpr95"], epoch_selected=sub[i]["epoch"]))
    cols = ["method", "dataset", "fold", "protocol", "auroc", "auprc", "fpr95", "epoch_selected", "device",
            "hp_config", "orientation", "source", "run_id", "n_eval", "ubgold_stop_epoch"]
    df = pd.DataFrame(rows, columns=cols).sort_values(["dataset", "protocol", "fold"]).reset_index(drop=True)
    df.to_csv(osp.join(RES, "muse_folds.csv"), index=False)
    pd.DataFrame(extra).to_csv(osp.join(RES, "muse_extra.csv"), index=False)
    pd.DataFrame(grid).to_csv(osp.join(RES, "muse_grid.csv"), index=False)
    cov = [dict(method="MUSE", dataset=ds, hp=hp, folds_done=int(df[(df.dataset == ds) & (df.protocol == p)].fold.nunique()))
           for ds in DATASETS for hp, p in (("ubgold", "oracle"), ("hpfree", "hpfree_oracle"))]
    pd.DataFrame(cov).to_csv(osp.join(RES, "muse_coverage.csv"), index=False)
    # reproduction table
    dx = pd.DataFrame(extra)
    r1 = r1_results()
    rep = []
    for ds in DATASETS:
        g = lambda p, d=df: d[(d.dataset == ds) & (d.protocol == p)].auroc
        o = g("oracle")
        row = dict(dataset=ds, folds=int(o.size), oracle_mean=o.mean(), oracle_std=o.std(ddof=0), last_mean=g("last").mean(),
                   valnormal_mean=g("valnormal").mean(), hpfree_valnormal_mean=g("hpfree_valnormal").mean(),
                   oracle_grid_mean=(dx[(dx.dataset == ds) & (dx.protocol == "oracle_grid")].auroc.mean() if len(dx) else np.nan))
        p = PAPER.get(ds)
        row.update(paper_t3=p[0] if p else np.nan, paper_t3_std=p[1] if p else np.nan, paper_t4=p[2] if p else np.nan,
                   paper_t1=p[3] if p else np.nan)
        row["r2_diff_oracle_vs_t3"] = row["oracle_mean"] - row["paper_t3"]
        if len(r1) and ds in set(r1.dataset):
            rr = r1[r1.dataset == ds].iloc[-1]
            row.update(r1_auroc=rr.r1_auroc, r1_std=rr.r1_std, r1_diff_vs_t3=rr.r1_auroc - row["paper_t3"])
        rep.append(row)
    pd.DataFrame(rep).to_csv(osp.join(RES, "muse_repro.csv"), index=False)
    print(f"{len(runs)} MUSE runs -> {RES}/muse_folds.csv ({len(df)} rows), muse_extra.csv, muse_grid.csv, "
          f"muse_coverage.csv, muse_repro.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(pd.DataFrame(rep).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
