"""Aggregate finished SEGO runs (ext/parts/SEGO_*.json) into fold-level protocol rows and the reproduction
check. Login node, any time (reads only finished runs).

    PYTHONDONTWRITEBYTECODE=1 /N/slate/yujlu/envs/grail-prelim/bin/python -B prelim/evalpaper/ext/SEGO/collect_sego.py

Writes ext/SEGO/results/:
  sego_folds.csv   one row per (dataset, fold, protocol); same columns as prelim/results/eval_folds.csv.
                   protocol: oracle | last | valnormal | hpfree_oracle | hpfree_last | hpfree_valnormal
                   - oracle   = UB-GOLD's rule (max test AUROC, patience 30 evaluation points; LABELLED; SEGO
                                has no test-label rule of its own, A10 §2)
                   - last     = SEGO's own reporting rule (final epoch)
                   - valnormal= min mean held-out-normal SEGO score (A3), ties -> earliest
  sego_folds_extra.csv  oracle_nostop (max test AUROC over all points), valnormal_obj (min held-out training
                   objective, secondary)
  sego_repro.csv   SEGO's own rule (LAST), 'ubgold' (released-modal) setting, 5-fold mean vs paper Table 2
  sego_summary.csv per (dataset, protocol): folds, mean, std
The row logic is copied from harness/collect.py (ubgold_oracle, pick, run_rows).
"""
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PARTS = os.environ.get("EP_PARTS", osp.join(osp.dirname(HERE), "parts"))
OUT = osp.join(HERE, "results")
PATIENCE = 30
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
# SEGO paper (arXiv 2503.03241, AAAI'25) Table 2, anomaly detection AUC %, mean over 5 runs
PAPER_T2 = {"ENZYMES": 76.62, "DHFR": 65.31, "BZR": 89.21, "NCI1": 70.34, "IMDB-BINARY": 66.48}


def ubgold_oracle(traj, eval_freq, key):
    max_auc, best, counter, stop_epoch = 0.0, None, 0, None
    for i, r in enumerate(traj):
        if r["epoch"] % eval_freq != 0:
            continue
        a = r.get(key, float("nan"))
        if a == a and a > max_auc:
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE:
            stop_epoch = r["epoch"]
            break
    return best, stop_epoch


def pick(traj, key, mode):
    vals = np.array([r.get(key, np.nan) if r.get(key) is not None else np.nan for r in traj], dtype=float)
    if np.all(np.isnan(vals)):
        return None
    return int(np.nanargmin(vals) if mode == "min" else np.nanargmax(vals))


def run_rows(J):
    meta, traj = J["meta"], J["traj"]
    hp = meta.get("hp", "ubgold")
    pre = "" if hp == "ubgold" else "hpfree_"
    h = meta["hparams"]
    ef, ne = int(h["eval_freq"]), int(h["num_epoch"])
    dev = "+".join(sorted({("cuda" if "cuda" in x else "cpu") for x in meta.get("device_all", meta["device"]).split("+")}))
    cfg = ("ubgold|" + " ".join(meta["ubgold_sh_line"].split()[2:]).split("#")[0].strip()) if hp == "ubgold" else (
        "hpfree|" + " ".join(f"{k}={h[k]}" for k in ("num_epoch", "eval_freq", "batch_size", "batch_size_test", "lr",
                                                     "hidden_dim", "num_layer", "alpha", "num_cluster", "rw_dim", "dg_dim")))
    base = dict(method="SEGO", dataset=meta["dataset"], fold=int(meta["fold"]), device=dev, hp_config=cfg,
                source="ext_sego", run_id=meta["run_id"], n_eval=len(traj), orientation="native")
    rows, extra = [], []

    def mk(proto, i, more=None, tgt=rows):
        r = traj[i]
        d = dict(base, protocol=proto, auroc=100 * r["auroc"], auprc=100 * r["auprc"], fpr95=100 * r["fpr95"],
                 epoch_selected=int(r["epoch"]))
        d.update(more or {})
        tgt.append(d)
    last_i = [i for i, r in enumerate(traj) if r["epoch"] == ne]
    assert last_i, meta["run_id"]
    b, stop = ubgold_oracle(traj, ef, "auroc")
    mk(pre + "oracle", b, dict(ubgold_stop_epoch=stop if stop is not None else ne))
    mk(pre + "last", last_i[0])
    mk(pre + "valnormal", pick(traj, "holdout_score", "min"))
    mk(pre + "oracle_nostop", pick(traj, "auroc", "max"), tgt=extra)
    v2 = pick(traj, "holdout_obj", "min")
    if v2 is not None:
        mk(pre + "valnormal_obj", v2, tgt=extra)
    return rows, extra


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, extra = [], []
    for fn in sorted(glob.glob(osp.join(PARTS, "SEGO_*.json"))):
        J = json.load(open(fn))
        m = J["meta"]
        stem = f"SEGO_{m['dataset']}_fold{m['fold']}"
        if m["run_id"] not in (stem, stem + "_hpfree") or m.get("max_epochs_override") is not None:
            continue
        r, e = run_rows(J)
        rows += r
        extra += e
    cols = ["method", "dataset", "fold", "protocol", "auroc", "auprc", "fpr95", "epoch_selected", "device",
            "hp_config", "orientation", "source", "run_id", "n_eval", "ubgold_stop_epoch"]
    df = pd.DataFrame(rows, columns=cols).sort_values(["dataset", "protocol", "fold"]).reset_index(drop=True)
    df.to_csv(osp.join(OUT, "sego_folds.csv"), index=False)
    pd.DataFrame(extra).to_csv(osp.join(OUT, "sego_folds_extra.csv"), index=False)
    summ = df.groupby(["dataset", "protocol"]).agg(folds=("fold", "nunique"), auroc_mean=("auroc", "mean"),
                                                   auroc_std=("auroc", lambda x: float(np.std(x, ddof=0)))).reset_index()
    summ.to_csv(osp.join(OUT, "sego_summary.csv"), index=False)
    rep = []
    for ds, pub in PAPER_T2.items():
        g = df[(df.dataset == ds) & (df.protocol == "last")]
        o = df[(df.dataset == ds) & (df.protocol == "oracle")]
        v = df[(df.dataset == ds) & (df.protocol == "valnormal")]
        mean = g.auroc.mean() if len(g) else np.nan
        rep.append(dict(dataset=ds, folds=int(g.fold.nunique()), last_mean=mean,
                        last_std=float(np.std(g.auroc, ddof=0)) if len(g) else np.nan, paper=pub, diff=mean - pub,
                        within5=bool(abs(mean - pub) <= 5) if len(g) == 5 else None,
                        oracle_mean=o.auroc.mean() if len(o) else np.nan,
                        valnormal_mean=v.auroc.mean() if len(v) else np.nan))
    pd.DataFrame(rep).to_csv(osp.join(OUT, "sego_repro.csv"), index=False)
    n = df.run_id.nunique()
    print(f"{n} finished SEGO runs -> {OUT}/sego_folds.csv ({len(df)} rows)")
    print(pd.DataFrame(rep).round(2).to_string(index=False))
    if len(summ):
        print(summ.pivot(index="dataset", columns="protocol", values="auroc_mean").round(2).to_string())


if __name__ == "__main__":
    main()
