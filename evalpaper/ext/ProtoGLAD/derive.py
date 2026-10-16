"""Derive the ProtoGLAD tables from the parts, grid units and native runs (login node, any time).

  /N/slate/yujlu/envs/ubgold-gpu/bin/python -B prelim/evalpaper/ext/ProtoGLAD/derive.py

Writes to prelim/evalpaper/ext/ProtoGLAD/results/:
  protoglad_folds.csv     one row per (dataset, fold, protocol); same columns as prelim/results/eval_folds.csv plus
                          hp_meaning. ProtoGLAD has one evaluation point per run, so within each setting
                          oracle = last = valnormal (A11 §2):
                            oracle / last / valnormal                 'tuned' (UB9: test-grid-best of the replayed
                                                                      search = TEST-LABEL-SELECTED upper bound;
                                                                      MUTAG: README eval line)
                            hpfree_oracle / hpfree_last / hpfree_valnormal   repository defaults
  protoglad_extra.csv     secondary labelled rows (never protocol rows):
                            oracle_own_gridbest   the method's own reporting rule (test-grid-best cell, all folds)
                            grid_perfold_best     per-fold max test AUROC over the 144 cells (descriptive)
                            grid_cell_mean        mean over the 144 cells (expected AUROC of a random grid cell)
  protoglad_coverage.csv  folds done per (dataset, hp)
  protoglad_repro.csv     R1 (UB-GOLD folds, own rule) and R2 (native protocol, verbatim run.py) vs the paper
  protoglad_diag_native_features.csv   diagnostic: ProtoGLAD's own features on UB-GOLD folds (never a protocol row)
  runtime_table.csv       seconds per grid unit / part / native search
"""
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PARTS = os.environ.get("EP_PARTS", osp.join(osp.dirname(HERE), "parts"))
DIAG_PARTS = "/N/slate/yujlu/grail_data/ext_protoglad/diag_parts"
GRID = "/N/slate/yujlu/grail_data/ext_protoglad/grid"
GRID_RES = osp.join(HERE, "results", "grid")
NATIVE = "/N/slate/yujlu/grail_data/ext_protoglad/native"
OUT = os.environ.get("EP_OUT", osp.join(HERE, "results"))
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
DATASETS = UB9 + ["MUTAG"]
# ProtoGLAD paper (arXiv 2602.10708v1) Table 1, AUC x 100 (mean, std); PROTEINS -> PROTEINS_full (NOTES.md)
PAPER = {"AIDS": (98.0, 0.5), "BZR": (85.6, 0.5), "COX2": (70.9, 1.5), "DD": (78.8, 0.6), "MUTAG": (89.8, 0.5),
         "PROTEINS_full": (79.4, 2.6), "NCI1": (70.1, 2.4), "DHFR": (62.4, 5.6)}
# UB-GOLD anomaly class (class 0) == ProtoGLAD's default (minority) anomaly class?
SAME_DIRECTION = {"AIDS": True, "DHFR": True, "NCI1": True, "MUTAG": True, "BZR": False, "COX2": False,
                  "PROTEINS_full": False, "DD": False}


def load_parts(parts, suffix=""):
    runs = {}
    for f in sorted(glob.glob(osp.join(parts, "ProtoGLAD_*.json"))):
        J = json.load(open(f))
        m = J["meta"]
        stem = f"ProtoGLAD_{m['dataset']}_fold{m['fold']}" + ("_hpfree" if m["hp"] == "hpfree" else "") + suffix
        if m["run_id"] != stem:
            continue                                   # smoke / tagged runs never enter the tables
        runs[(m["dataset"], m["hp"], int(m["fold"]))] = J
    return runs


def grid_df(features="ubgold"):
    rows = []
    for f in glob.glob(osp.join(GRID, features, "*", "fold*_psi1_*.json")):
        rows += json.load(open(f))["rows"]
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load_parts(PARTS)
    rows = []
    for (ds, hp, fold), J in sorted(runs.items()):
        m, r = J["meta"], J["traj"][0]
        pre = "" if hp == "ubgold" else "hpfree_"
        for proto in ("oracle", "last", "valnormal"):
            rows.append(dict(method="ProtoGLAD", dataset=ds, fold=fold, protocol=pre + proto, auroc=100 * r["auroc"],
                             auprc=100 * r["auprc"], fpr95=100 * r["fpr95"], epoch_selected=1, device=m["device"],
                             hp_config=("ubgold|" if hp == "ubgold" else "hpfree|") + m["ubgold_sh_line"],
                             orientation="native", source="ext_protoglad", run_id=m["run_id"], n_eval=1,
                             ubgold_stop_epoch=1 if proto == "oracle" else np.nan, hp_meaning=m["hp_meaning"],
                             holdout_score=r["holdout_score"], n_prototypes=r["n_prototypes"],
                             nonfinite=r.get("nonfinite", 0)))
    df = pd.DataFrame(rows)
    df.to_csv(osp.join(OUT, "protoglad_folds.csv"), index=False)

    # secondary rows from the grid replay
    extra, rep = [], []
    G = grid_df("ubgold")
    for ds in DATASETS:
        gb = osp.join(GRID_RES, f"{ds}_gridbest.json")
        if not osp.exists(gb) or not len(G):
            continue
        B = json.load(open(gb))
        bp = B["best_params"]
        g = G[G.dataset == ds]
        sel = g[(g.h == bp["h"]) & (g.psi1 == bp["psi1"]) & (g.psi2 == bp["psi2"]) & (g.tau == bp["tau"])]
        for _, x in sel.sort_values("fold").iterrows():
            extra.append(dict(method="ProtoGLAD", dataset=ds, fold=int(x.fold), protocol="oracle_own_gridbest",
                              auroc=100 * x.auroc, auprc=100 * x.auprc, fpr95=100 * x.fpr95,
                              cell=f"h={bp['h']} psi1={bp['psi1']} psi2={bp['psi2']} tau={bp['tau']}",
                              rule="own: first max of 5-fold mean test AUC over 144 cells (test-label selected)"))
        for fold, gf in g.groupby("fold"):
            i = gf.auroc.idxmax()
            extra.append(dict(method="ProtoGLAD", dataset=ds, fold=int(fold), protocol="grid_perfold_best",
                              auroc=100 * gf.auroc.max(), auprc=100 * gf.loc[i, "auprc"], fpr95=100 * gf.loc[i, "fpr95"],
                              cell=f"psi1={gf.loc[i, 'psi1']} psi2={gf.loc[i, 'psi2']} tau={gf.loc[i, 'tau']}",
                              rule="descriptive: per-fold max over cells"))
            extra.append(dict(method="ProtoGLAD", dataset=ds, fold=int(fold), protocol="grid_cell_mean",
                              auroc=100 * gf.auroc.mean(), auprc=100 * gf.auprc.mean(), fpr95=100 * gf.fpr95.mean(),
                              cell="mean of 144 cells", rule="descriptive"))
        own = 100 * sel.auroc.values
        if ds in PAPER:
            pm, ps = PAPER[ds]
            rep.append(dict(check="R1 UB-GOLD folds, own rule (test-grid-best)", dataset=ds, folds=len(own),
                            ours_mean=own.mean(), ours_std=own.std(), paper=pm, paper_std=ps, diff=own.mean() - pm,
                            within5=bool(abs(own.mean() - pm) <= 5), same_anomaly_class=SAME_DIRECTION[ds],
                            cell=str(bp)))
    dx = pd.DataFrame(extra)
    dx.to_csv(osp.join(OUT, "protoglad_extra.csv"), index=False)

    # R2: native protocol (verbatim run.py search)
    for ds in sorted(PAPER):
        rd = osp.join(NATIVE, f"run_{ds}")
        csvp = osp.join(rd, "sensitivity_plots", ds, f"{ds}_param_results.csv")
        if not osp.exists(csvp):
            continue
        meta = json.load(open(osp.join(rd, "native_meta.json")))
        if not meta.get("completed"):
            continue                                    # an interrupted search is not the own rule (RNG stream)
        c = pd.read_csv(csvp)
        best, bi = 0.0, None
        for i, x in c.iterrows():                       # hyperparam_search's rule on its own CSV order
            if x.auc > best:
                best, bi = x.auc, i
        pm, ps = PAPER[ds]
        rep.append(dict(check="R2 native protocol (verbatim run.py search, seed 42, minority anomaly, own features)",
                        dataset=ds, folds=5, ours_mean=100 * c.loc[bi, "auc"], ours_std=100 * c.loc[bi, "auc_std"],
                        paper=pm, paper_std=ps, diff=100 * c.loc[bi, "auc"] - pm,
                        within5=bool(abs(100 * c.loc[bi, "auc"] - pm) <= 5), same_anomaly_class=True,
                        cell=f"psi1={c.loc[bi, 'psi1']} psi2={c.loc[bi, 'psi2']} tau={c.loc[bi, 'tau']}",
                        cells_done=len(c), completed=bool(meta.get("completed")), device=meta.get("device"),
                        attr=meta.get("attr"), cell_auc_median=100 * c.auc.median(), cell_auc_min=100 * c.auc.min()))
    pd.DataFrame(rep).to_csv(osp.join(OUT, "protoglad_repro.csv"), index=False)

    # coverage
    cov = [dict(dataset=ds, hp=hp, folds_done=sum((ds, hp, f) in runs for f in range(5)),
                grid_units_done=len(glob.glob(osp.join(GRID, "ubgold", ds, "fold*_psi1_*.json"))))
           for ds in DATASETS for hp in ("ubgold", "hpfree")]
    pd.DataFrame(cov).to_csv(osp.join(OUT, "protoglad_coverage.csv"), index=False)

    # diagnostic: native features on UB-GOLD folds
    druns = load_parts(DIAG_PARTS, "_nativefeat") if osp.isdir(DIAG_PARTS) else {}
    drows = []
    for (ds, hp, fold), J in sorted(druns.items()):
        r = J["traj"][0]
        base = runs.get((ds, hp, fold))
        drows.append(dict(dataset=ds, fold=fold, hp=hp, features=J["meta"]["features"], auroc=100 * r["auroc"],
                          auprc=100 * r["auprc"], auroc_ubgold_features=(100 * base["traj"][0]["auroc"]
                                                                         if base else np.nan),
                          cell=J["meta"]["ubgold_sh_line"].split("|")[-1].strip()))
    pd.DataFrame(drows).to_csv(osp.join(OUT, "protoglad_diag_native_features.csv"), index=False)

    # runtime
    rt = []
    for f in glob.glob(osp.join(GRID, "*", "*", "fold*_psi1_*.json")):
        J = json.load(open(f))
        rt.append(dict(kind="grid_unit", features=J["features"], dataset=J["dataset"], fold=J["fold"], psi1=J["psi1"],
                       seconds=J["seconds"], maxrss_gb=J["memory"]["maxrss_gb"], device=J["device"],
                       seconds_stage1=J["rows"][0]["seconds_stage1"],
                       seconds_stage2_mean=float(np.mean([x["seconds_stage2"] for x in J["rows"]])),
                       seconds_stage2_max=float(np.max([x["seconds_stage2"] for x in J["rows"]]))))
    for (ds, hp, fold), J in runs.items():
        rt.append(dict(kind="part", features="ubgold", dataset=ds, fold=fold, hp=hp, seconds=J["meta"]["seconds"],
                       maxrss_gb=J["meta"]["memory"]["maxrss_gb"], device=J["meta"]["device"]))
    for d in glob.glob(osp.join(NATIVE, "run_*", "native_meta.json")):
        m = json.load(open(d))
        rt.append(dict(kind="native_search", dataset=m["dataset"], seconds=m.get("seconds"), device=m["device"],
                       completed=bool(m.get("completed"))))
    pd.DataFrame(rt).to_csv(osp.join(HERE, "runtime_table.csv"), index=False)

    # console summary
    if len(df):
        s = df[df.protocol.isin(["oracle", "hpfree_oracle"])].groupby(["dataset", "protocol"]).auroc.agg(["mean", "std", "count"])
        print(s.round(2).to_string())
    if rep:
        print(pd.DataFrame(rep)[["check", "dataset", "ours_mean", "paper", "diff", "within5"]].round(2).to_string())


if __name__ == "__main__":
    main()
