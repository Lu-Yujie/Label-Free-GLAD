"""Derive the protocol rows for GLADPro from prelim/evalpaper/ext/parts/GLADPro_*.json (login node, any time).

  /N/slate/yujlu/envs/grail-prelim/bin/python -B prelim/evalpaper/ext/GLADPro/derive.py

Writes to prelim/evalpaper/ext/GLADPro/results/:
  gladpro_folds.csv   one row per (dataset, fold, protocol); same columns as prelim/results/eval_folds.csv plus
                      oracle_rule and nonfinite_selected. Protocols (NOTES.md, pre-registered):
                        oracle            GLADPro's own rule (main.py:201-211): epoch = argmax over epochs of the
                                          5-fold mean test AUROC (NaN -> 0, as its compute_metrics); all folds
                                          report that epoch. oracle_rule = gladpro_own. Needs all 5 folds.
                        oracle_ubgold     UB-GOLD's per-fold rule (max test AUROC, patience 30 evaluations) on
                                          GLADPro's native schedule (every epoch). oracle_rule = ubgold_patience30.
                        last              final epoch.
                        valnormal         min mean held-out-normal anomaly score (ties -> earliest; NaN skipped).
                        hpfree_<same>     the same rows for the hpfree runs.
                      A selected point with non-finite test scores is reported at chance (AUROC 50, AUPRC =
                      anomaly prevalence, FPR95 100) with nonfinite_selected = 1 (NOTES.md rule).
  gladpro_extra.csv   oracle_nostop (per-fold max test AUROC over all epochs) and hpfree_oracle_nostop.
  gladpro_coverage.csv folds done per (dataset, hp).
  gladpro_repro.csv   own-rule ORACLE (ubgold hp), 5-fold mean +- std, vs third-party GLADPro numbers.
"""
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PARTS = os.environ.get("EP_PARTS", osp.join(osp.dirname(HERE), "parts"))
OUT = os.environ.get("EP_OUT", osp.join(HERE, "results"))
PATIENCE = 30
DATASETS = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY",
            "MUTAG", "DD"]
# third-party GLADPro AUROC (%) (NOTES.md): M-GLAD IJCAI'26 Table 1 (5 trials, UB-GOLD-sized folds) and
# ProtoGLAD arXiv 2602.10708 Table 1 (5-fold CV). The paper's own table could not be retrieved.
MGLAD = {"PROTEINS_full": (65.97, 5.35), "AIDS": (78.00, 8.35), "BZR": (67.04, 5.29), "COX2": (55.67, 5.93),
         "ENZYMES": (52.78, 4.49), "DD": (65.96, 3.21), "IMDB-BINARY": (70.62, 3.55), "MUTAG": (82.08, 7.67)}
PROTOGLAD = {"AIDS": 52.6, "BZR": 60.6, "COX2": 63.7, "DD": 76.6, "MUTAG": 89.1, "PROTEINS_full": 73.6,
             "NCI1": 59.2, "DHFR": 60.9}


def load_parts():
    runs = {}
    for f in sorted(glob.glob(osp.join(PARTS, "GLADPro_*.json"))):
        J = json.load(open(f))
        m = J["meta"]
        if m.get("max_epochs_override") is not None or m["run_id"] != (
                f"GLADPro_{m['dataset']}_fold{m['fold']}" + ("_hpfree" if m["hp"] == "hpfree" else "")):
            continue                                   # smoke / tagged runs never enter the tables
        runs[(m["dataset"], m["hp"], int(m["fold"]))] = J
    return runs


def ubgold_oracle(aucs):
    max_auc, best, counter, stop = 0.0, None, 0, None
    for i, a in enumerate(aucs):
        if a == a and a > max_auc:
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE:
            stop = i
            break
    return best, stop


def row(J, i, proto, rule, extra=None):
    m, r = J["meta"], J["traj"][i]
    nonfin = int(r.get("nonfinite", 0) > 0 or not np.isfinite(r["auroc"]))
    prev = m["n_test_anomalous"] / m["n_test"]
    d = dict(method="GLADPro", dataset=m["dataset"], fold=int(m["fold"]), protocol=proto,
             auroc=50.0 if nonfin else 100 * r["auroc"], auprc=100 * prev if nonfin else 100 * r["auprc"],
             fpr95=100.0 if nonfin else 100 * r["fpr95"], epoch_selected=int(r["epoch"]),
             device="+".join(sorted({("cuda" if "cuda" in s["device"] else "cpu") for s in m["segments"]})),
             hp_config=f"{m['hp']}|{m['ubgold_sh_line']}", orientation="native", source="ext_gladpro",
             run_id=m["run_id"], n_eval=len(J["traj"]), oracle_rule=rule, nonfinite_selected=nonfin,
             n_prot_selected=int(r.get("n_prot", -1)))
    d.update(extra or {})
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load_parts()
    rows, extra, cov = [], [], []
    for DS in DATASETS:
        for hp in ("ubgold", "hpfree"):
            pre = "" if hp == "ubgold" else "hpfree_"
            folds = {k: runs[(DS, hp, k)] for k in range(5) if (DS, hp, k) in runs}
            cov.append(dict(method="GLADPro", dataset=DS, hp=hp, folds_done=len(folds),
                            folds=",".join(map(str, sorted(folds)))))
            for k, J in folds.items():
                T = J["traj"]
                aucs = [r["auroc"] for r in T]
                b, stop = ubgold_oracle(aucs)
                if b is not None:
                    rows.append(row(J, b, pre + "oracle_ubgold", "ubgold_patience30",
                                    dict(ubgold_stop_epoch=(T[stop]["epoch"] if stop is not None else T[-1]["epoch"]))))
                rows.append(row(J, len(T) - 1, pre + "last", ""))
                hold = np.array([r["holdout_score"] for r in T], dtype=float)
                if np.isfinite(hold).any():
                    rows.append(row(J, int(np.nanargmin(np.where(np.isfinite(hold), hold, np.nan))), pre + "valnormal", ""))
                a = np.array(aucs, dtype=float)
                if np.isfinite(a).any():
                    extra.append(row(J, int(np.nanargmax(a)), pre + "oracle_nostop", "per_fold_max"))
            if len(folds) == 5:
                L = {len(J["traj"]) for J in folds.values()}
                assert len(L) == 1, (DS, hp, L)
                A = np.array([[r["auroc"] if np.isfinite(r["auroc"]) else 0.0 for r in folds[k]["traj"]]
                              for k in range(5)])           # NaN -> 0 as GLADPro's compute_metrics
                idx = int(np.argmax(A.mean(0)))
                for k in range(5):
                    rows.append(row(folds[k], idx, pre + "oracle", "gladpro_own"))
    F = pd.DataFrame(rows)
    F.to_csv(osp.join(OUT, "gladpro_folds.csv"), index=False)
    pd.DataFrame(extra).to_csv(osp.join(OUT, "gladpro_extra.csv"), index=False)
    pd.DataFrame(cov).to_csv(osp.join(OUT, "gladpro_coverage.csv"), index=False)
    rep = []
    if len(F):
        for DS in DATASETS:
            g = F[(F.dataset == DS)]
            if not len(g):
                continue
            d = dict(dataset=DS)
            for p in ("oracle", "oracle_ubgold", "last", "valnormal", "hpfree_oracle", "hpfree_last", "hpfree_valnormal"):
                x = g[g.protocol == p].auroc
                d[p] = f"{x.mean():.2f} ± {x.std(ddof=0):.2f} (n={len(x)})" if len(x) else ""
            x = g[g.protocol == "oracle"].auroc
            if DS in MGLAD and len(x) == 5:
                d["mglad"] = f"{MGLAD[DS][0]:.2f} ± {MGLAD[DS][1]:.2f}"
                d["diff_vs_mglad"] = round(x.mean() - MGLAD[DS][0], 2)
                d["within5_mglad"] = bool(abs(x.mean() - MGLAD[DS][0]) <= 5)
            if DS in PROTOGLAD and len(x) == 5:
                d["protoglad"] = PROTOGLAD[DS]
                d["diff_vs_protoglad"] = round(x.mean() - PROTOGLAD[DS], 2)
            rep.append(d)
    R = pd.DataFrame(rep)
    R.to_csv(osp.join(OUT, "gladpro_repro.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(cov).pivot(index="dataset", columns="hp", values="folds_done"))
        print(R.to_string(index=False))


if __name__ == "__main__":
    main()
