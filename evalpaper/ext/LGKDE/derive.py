"""Protocol rows for LGKDE runs (NOTES.md 0.4) from the part JSONs in ext/parts (or EP_PARTS).

Writes ext/LGKDE/results/lgkde_folds.csv (schema of prelim/results/eval_folds.csv plus `rule`) and
ext/LGKDE/results/lgkde_repro.csv (own-rule ORACLE 5-fold mean vs LGKDE Table 1).
  oracle_own     LGKDE's own rule: EarlyStopping(patience, min_delta, min_epochs) (verbatim class from
                 src/experiment.py) on (test AUROC + test AUPR)/2, checkpoint = first max of that key among
                 epochs 1..stop (save_results / plot_results).
  oracle_ubgold  UB-GOLD rule: max test AUROC, patience 30 (harness collect.py ubgold_oracle, eval_freq 1).
  last           epoch num_epoch (100).
  valnormal      min mean held-out-normal anomaly score (= max mean density), ties -> earliest.
hpfree runs get the prefix hpfree_. AUROC/AUPRC/FPR95 in percent (UB-GOLD metric functions; FPR95 = first
threshold with TPR >= 0.95). LGKDE's own FPR95 (nearest TPR to 0.95) is in column fpr95_lgkde.
Usage: python -B derive.py [--parts DIR] [--include_tagged]
"""
import argparse
import ast
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PARTS = os.environ.get("EP_PARTS", osp.normpath(osp.join(HERE, "..", "parts")))
OUT = osp.join(HERE, "results")
PATIENCE_UBGOLD = 30
PAPER = dict(MUTAG=91.63, PROTEINS_full=78.97, DD=79.84, ENZYMES=71.04, DHFR=82.58, BZR=81.11, COX2=66.69,
             AIDS=99.06, **{"IMDB-BINARY": 68.77}, NCI1=76.67, COLLAB=67.94, **{"REDDIT-BINARY": 88.12})
SAME_DIRECTION = {"MUTAG", "AIDS", "NCI1", "DHFR"}   # NOTES.md 0.5


def _early_stopping_cls():
    tree = ast.parse(open(osp.join(HERE, "src", "experiment.py")).read())
    node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EarlyStopping"][0]
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "experiment.py", "exec"), ns)
    return ns["EarlyStopping"]


EarlyStopping = _early_stopping_cls()


def oracle_own(traj, rule):
    es = EarlyStopping(patience=rule["patience"], min_delta=rule["min_delta"], min_epochs=rule["min_epochs"])
    best_perf, best_i, stop = 0, None, None
    for i, r in enumerate(traj):
        perf = r["own_sel"]
        if perf > best_perf:                      # save_results
            best_perf, best_i = perf, i
        if es(r["epoch"], perf):                  # train(): metrics recorded, then the stop check
            stop = r["epoch"]
            break
    return best_i, stop


def oracle_ubgold(traj, eval_freq=1, key="auroc"):
    max_auc, best, counter, stop = 0.0, None, 0, None
    for i, r in enumerate(traj):
        if r["epoch"] % eval_freq != 0:
            continue
        a = r.get(key, float("nan"))
        if a == a and a > max_auc:
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE_UBGOLD:
            stop = r["epoch"]
            break
    return best, stop


def pick(traj, key, mode):
    v = np.array([r.get(key, np.nan) for r in traj], dtype=float)
    if np.all(np.isnan(v)):
        return None
    return int(np.nanargmin(v) if mode == "min" else np.nanargmax(v))


def rows_of(J):
    meta, traj = J["meta"], J["traj"]
    hp = meta["hp"]
    pre = "" if hp == "ubgold" else "hpfree_"
    ne = int(meta["hparams"]["num_epoch"])
    base = dict(method=meta["method"], dataset=meta["dataset"], fold=int(meta["fold"]),
                device=meta.get("device_all", meta.get("device")), hp_config=f"{hp}|{meta['released_config']}",
                orientation="native", source="ext_lgkde", run_id=meta["run_id"], n_eval=len(traj))
    out = []

    def mk(proto, i, rule, more=None):
        r = traj[i]
        d = dict(base, protocol=pre + proto, rule=rule, auroc=100 * r["auroc"], auprc=100 * r["auprc"],
                 fpr95=100 * r["fpr95"], fpr95_lgkde=100 * r["lgkde_fpr95"], epoch_selected=int(r["epoch"]))
        d.update(more or {})
        out.append(d)
    b, stop = oracle_own(traj, meta["own_rule"])
    mk("oracle_own", b, "LGKDE own: max (AUROC+AUPR)/2 with its EarlyStopping " + json.dumps(meta["own_rule"]),
       dict(stop_epoch=stop if stop is not None else ne))
    b2, stop2 = oracle_ubgold(traj)
    mk("oracle_ubgold", b2, "UB-GOLD: max AUROC, patience 30", dict(stop_epoch=stop2 if stop2 is not None else ne))
    li = [i for i, r in enumerate(traj) if r["epoch"] == ne]
    assert li, meta["run_id"]
    mk("last", li[0], "final epoch")
    mk("valnormal", pick(traj, "holdout_score", "min"), "min mean held-out-normal anomaly score")
    mk("oracle_nostop", pick(traj, "auroc", "max"), "max AUROC over all epochs (diagnostic)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=PARTS)
    ap.add_argument("--include_tagged", action="store_true")
    a = ap.parse_args()
    rows = []
    for fn in sorted(glob.glob(osp.join(a.parts, "LGKDE_*.json"))):
        J = json.load(open(fn))
        m = J["meta"]
        stem = f"LGKDE_{m['dataset']}_fold{m['fold']}"
        if not a.include_tagged and (m["run_id"] not in (stem, stem + "_hpfree") or m.get("max_epochs_override") is not None):
            continue
        rows += rows_of(J)
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(rows)
    if not len(df):
        print("no finished LGKDE runs")
        return
    df = df.sort_values(["dataset", "protocol", "fold"]).reset_index(drop=True)
    df.to_csv(osp.join(OUT, "lgkde_folds.csv"), index=False)
    rep = []
    for ds, g in df.groupby("dataset"):
        row = dict(dataset=ds, paper_auroc=PAPER.get(ds), same_anomaly_class_as_paper=ds in SAME_DIRECTION)
        for proto in ("oracle_own", "oracle_ubgold", "last", "valnormal", "hpfree_valnormal", "hpfree_oracle_own"):
            gg = g[g.protocol == proto]
            row[f"{proto}_folds"] = int(gg.fold.nunique())
            row[f"{proto}_auroc"] = gg.auroc.mean() if len(gg) else np.nan
        row["oracle_own_std"] = g[g.protocol == "oracle_own"].auroc.std(ddof=0)
        row["diff_vs_paper"] = row["oracle_own_auroc"] - row["paper_auroc"] if row["paper_auroc"] else np.nan
        row["within_5"] = bool(abs(row["diff_vs_paper"]) <= 5) if row["diff_vs_paper"] == row["diff_vs_paper"] else None
        rep.append(row)
    pd.DataFrame(rep).to_csv(osp.join(OUT, "lgkde_repro.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(rep)[["dataset", "oracle_own_folds", "oracle_own_auroc", "oracle_own_std", "paper_auroc",
                                 "diff_vs_paper", "within_5", "same_anomaly_class_as_paper", "oracle_ubgold_auroc",
                                 "last_auroc", "valnormal_auroc", "hpfree_valnormal_auroc"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
