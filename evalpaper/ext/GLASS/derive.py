"""Derive the protocol rows for GLASS from prelim/evalpaper/ext/parts/GLASS_*.json (login node, any time).

  /N/slate/yujlu/envs/ubgold-gpu/bin/python -B prelim/evalpaper/ext/GLASS/derive.py

Writes to prelim/evalpaper/ext/GLASS/results/ (all AUROC/AUPRC/FPR95 in %; NOTES.md Part 1, pre-registered):
  glass_folds.csv     one row per (dataset, fold, protocol); columns of prelim/results/eval_folds.csv plus
                      oracle_rule, scorer, nonfinite_selected, hp_alias.
                        oracle         ORACLE-own (primary own rule, labelled): GLASS's test-selected checkpoint with its
                                       replayed early stopping, then evaluate_reference.py's best of 24 rules by 5-fold
                                       mean test AUROC (oracle_rule glass_ref_best24; needs all 5 folds)
                        oracle_ubgold  UB-GOLD rule (max test AUROC, patience 30 evaluations) on the default scorer
                        last           epoch 150, default scorer rcsm_full_test_cdf
                        valnormal      min held-out-normal mean of the default scorer (ties earliest, non-finite skipped)
                        hpfree_<same>  from the _hpfree runs; if a run is missing, the ubgold run is used (hp_alias = 1)
  glass_extra.csv     oracle_own_train (train_single.py: best of 6 scorers per fold at the same checkpoint),
                      oracle_own_max (higher 5-fold mean of the two own reports), oracle_nostop (max test AUROC of the
                      default scorer), last_r0 / valnormal_r0 (fallback rule fixed_test_raw), and hpfree_ versions.
  glass_coverage.csv  folds done per (dataset, hp).
  glass_repro.csv     ORACLE-own (ubgold) 5-fold mean vs the paper (v2 main table), direction flag.
Non-finite (A15): ORACLE never selects a non-finite point; VAL-NORMAL skips non-finite held-out means; a non-finite
LAST is scored AUROC 50, AUPRC = test anomaly rate, FPR95 95 and flagged nonfinite_selected = 1.
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
DATASETS = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY", "MUTAG"]
# paper v2 (2026-10-01) main table, GLASS row (AUROC %, 5 seeds, GLASS's own split)
PAPER = {"MUTAG": 91.42, "PROTEINS_full": 80.63, "ENZYMES": 66.20, "DHFR": 69.38, "BZR": 80.93, "COX2": 68.70,
         "AIDS": 99.32, "IMDB-BINARY": 77.20, "NCI1": 75.86, "REDDIT-BINARY": 88.80}
SAME_DIRECTION = {"MUTAG", "AIDS", "DHFR", "NCI1", "ENZYMES", "IMDB-BINARY", "REDDIT-BINARY"}   # NOTES Part 1


def load_parts():
    runs = {}
    for f in sorted(glob.glob(osp.join(PARTS, "GLASS_*.json"))):
        J = json.load(open(f))
        m = J["meta"]
        if m.get("max_epochs_override") is not None or m["run_id"] != (
                f"GLASS_{m['dataset']}_fold{m['fold']}" + ("_hpfree" if m["hp"] == "hpfree" else "")):
            continue                                   # smoke / tagged runs never enter the tables
        runs[(m["dataset"], m["hp"], int(m["fold"]))] = J
    return runs


def finite(x):
    return x is not None and x == x and np.isfinite(x)


def base(J, proto, hp_alias):
    m = J["meta"]
    return dict(method="GLASS", dataset=m["dataset"], fold=int(m["fold"]), protocol=proto,
                device="+".join(sorted({("cuda" if "cuda" in s["device"] else "cpu") for s in m["segments"]})),
                hp_config=f"{m['hp']}|{m['ubgold_sh_line']}", orientation="native", source="ext_glass",
                run_id=m["run_id"], n_eval=len(J["traj"]), hp_alias=int(hp_alias))


def point_row(J, i, proto, scorer, rule, hp_alias, prefix=""):
    """Metrics of evaluation point i under the default scorer (prefix '') or the fallback rule (prefix 'r0_')."""
    m, r = J["meta"], J["traj"][i]
    au, ap, fp = r[prefix + "auroc"], r[prefix + "auprc"], r[prefix + "fpr95"]
    nonfin = int(not finite(au))
    prev = m["n_test_anomalous"] / m["n_test"]
    d = base(J, proto, hp_alias)
    d.update(auroc=50.0 if nonfin else 100 * au, auprc=100 * prev if nonfin else 100 * ap,
             fpr95=95.0 if nonfin else 100 * fp, epoch_selected=int(r["epoch"]), oracle_rule=rule, scorer=scorer,
             nonfinite_selected=nonfin)
    return d


def ubgold_oracle(aucs):
    max_auc, best, counter, stop = 0.0, None, 0, None
    for i, a in enumerate(aucs):
        if finite(a) and a > max_auc:
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE:
            stop = i
            break
    return best, stop


def own_row(J, proto, rule, scorer, metrics, hp_alias):
    o = J["meta"]["own"]
    d = base(J, proto, hp_alias)
    d.update(auroc=100 * metrics[0], auprc=100 * metrics[1], fpr95=100 * metrics[2], epoch_selected=int(o["best_epoch"]),
             oracle_rule=rule, scorer=scorer, nonfinite_selected=0, own_stop_epoch=int(o["stop_epoch"]))
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load_parts()
    rows, extra, cov = [], [], []
    for DS in DATASETS:
        for hp in ("ubgold", "hpfree"):
            pre = "" if hp == "ubgold" else "hpfree_"
            folds, alias = {}, {}
            for k in range(5):
                if (DS, hp, k) in runs:
                    folds[k], alias[k] = runs[(DS, hp, k)], 0
                elif hp == "hpfree" and (DS, "ubgold", k) in runs:
                    folds[k], alias[k] = runs[(DS, "ubgold", k)], 1        # NOTES Part 1: identical configuration
            cov.append(dict(method="GLASS", dataset=DS, hp=hp, folds_done=sum(1 for k in folds if not alias[k]),
                            folds_aliased=sum(alias.values()), folds=",".join(map(str, sorted(folds)))))
            for k, J in folds.items():
                T = J["traj"]
                hal = alias[k]
                aucs = [r["auroc"] for r in T]
                b, stop = ubgold_oracle(aucs)
                if b is not None:
                    d = point_row(J, b, pre + "oracle_ubgold", J["meta"]["default_rule"], "ubgold_patience30", hal)
                    d["ubgold_stop_epoch"] = T[stop]["epoch"] if stop is not None else T[-1]["epoch"]
                    rows.append(d)
                rows.append(point_row(J, len(T) - 1, pre + "last", J["meta"]["default_rule"], "", hal))
                for prefix, tag, hkey in (("", "", "holdout_score"), ("r0_", "_r0", "r0_holdout")):
                    hold = np.array([r[hkey] if finite(r[hkey]) else np.nan for r in T], dtype=float)
                    scorer = J["meta"]["default_rule"] if not prefix else J["meta"]["r0_rule"]
                    if np.isfinite(hold).any():
                        i = int(np.nanargmin(hold))
                        (rows if not prefix else extra).append(point_row(J, i, pre + "valnormal" + tag, scorer, "", hal, prefix))
                    if prefix:
                        extra.append(point_row(J, len(T) - 1, pre + "last" + tag, scorer, "", hal, prefix))
                a = np.array([x if finite(x) else np.nan for x in aucs], dtype=float)
                if np.isfinite(a).any():
                    extra.append(point_row(J, int(np.nanargmax(a)), pre + "oracle_nostop", J["meta"]["default_rule"],
                                           "per_fold_max", hal))
                o = J["meta"]["own"]
                bm = o["best_method"]
                extra.append(own_row(J, pre + "oracle_own_train", "glass_train_best6", bm, o["train_metrics"][bm], hal))
            if len(folds) == 5:
                rule_names = folds[0]["meta"]["rules"]
                means = {r: np.mean([folds[k]["meta"]["own"]["ref_metrics"][r][0] for k in range(5)]) for r in rule_names}
                best_rule = max(means, key=lambda r: means[r])          # evaluate_reference.py l.383 (first max wins)
                ref_rows = [own_row(folds[k], pre + "oracle", "glass_ref_best24", best_rule,
                                    folds[k]["meta"]["own"]["ref_metrics"][best_rule], alias[k]) for k in range(5)]
                rows += ref_rows
                tr = [e for e in extra if e["dataset"] == DS and e["protocol"] == pre + "oracle_own_train"]
                m_ref = np.mean([r["auroc"] for r in ref_rows])
                m_tr = np.mean([r["auroc"] for r in tr]) if len(tr) == 5 else -1
                pick = ref_rows if m_ref >= m_tr else tr
                for r in pick:
                    r2 = dict(r, protocol=pre + "oracle_own_max", oracle_rule="glass_own_max:" + r["oracle_rule"])
                    extra.append(r2)
    F = pd.DataFrame(rows)
    F.to_csv(osp.join(OUT, "glass_folds.csv"), index=False)
    pd.DataFrame(extra).to_csv(osp.join(OUT, "glass_extra.csv"), index=False)
    pd.DataFrame(cov).to_csv(osp.join(OUT, "glass_coverage.csv"), index=False)
    E = pd.DataFrame(extra)
    rep = []
    for DS in DATASETS:
        if not len(F) or not (F.dataset == DS).any():
            continue
        d = dict(dataset=DS, direction_vs_paper=("same" if DS in SAME_DIRECTION else "opposite"))
        for p in ("oracle", "oracle_ubgold", "last", "valnormal", "hpfree_oracle", "hpfree_last", "hpfree_valnormal"):
            x = F[(F.dataset == DS) & (F.protocol == p)].auroc
            d[p] = f"{x.mean():.2f} ± {x.std(ddof=0):.2f} (n={len(x)})" if len(x) else ""
        for p in ("oracle_own_train", "oracle_own_max"):
            x = E[(E.dataset == DS) & (E.protocol == p)].auroc if len(E) else []
            d[p] = f"{np.mean(x):.2f} (n={len(x)})" if len(x) else ""
        g = F[(F.dataset == DS) & (F.protocol == "oracle")]
        if len(g) == 5:
            d["oracle_rule_selected"] = g.scorer.iloc[0]
            d["paper"] = PAPER[DS]
            d["diff_vs_paper"] = round(g.auroc.mean() - PAPER[DS], 2)
            d["within5"] = bool(abs(g.auroc.mean() - PAPER[DS]) <= 5)
        rep.append(d)
    R = pd.DataFrame(rep)
    R.to_csv(osp.join(OUT, "glass_repro.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(cov).pivot(index="dataset", columns="hp", values="folds_done"))
        if len(R):
            print(R.to_string(index=False))


if __name__ == "__main__":
    main()
