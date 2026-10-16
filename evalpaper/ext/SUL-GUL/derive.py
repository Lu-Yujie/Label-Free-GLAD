"""Derive the protocol rows for SUL/GUL from prelim/evalpaper/ext/parts/SUL-GUL_*.json (login node, any time).

  /N/slate/yujlu/envs/grail-prelim/bin/python -B prelim/evalpaper/ext/SUL-GUL/derive.py

Writes to prelim/evalpaper/ext/SUL-GUL/results/:
  sulgul_folds.csv     one row per (dataset, fold, protocol); same columns as prelim/results/eval_folds.csv plus
                       oracle_rule and nonfinite_selected. Protocols (NOTES.md Part 1, pre-registered):
                         oracle          ORACLE-own, GUL's rule (main.py:94-104): epoch = argmax over epochs of the
                                         5-fold mean test AUROC (first on ties); every fold reports that epoch.
                                         A15 rule 1: only epochs at which all 5 folds have finite test scores.
                                         oracle_rule = gul_own (labelled). Needs all 5 folds.
                         oracle_ubgold   UB-GOLD's per-fold rule (max test AUROC, patience 30 evaluations) on GUL's
                                         every-epoch schedule; non-finite points never improve (A15 rule 1).
                         last            final epoch (A15 rule 3 if non-finite).
                         valnormal       min mean held-out-normal anomaly score; ties -> earliest; non-finite held-out
                                         means are skipped (A15 rule 2).
                         hpfree_<same>   the same rows for the hpfree runs.
                       Non-finite selected point (A15 rules 3-4): AUROC 50, AUPRC = test anomaly rate, FPR95 95,
                       nonfinite_selected = 1. A run with no finite point gets that under every protocol.
  sulgul_extra.csv     oracle_nostop (per-fold max test AUROC), oracle_verbatim (GUL's rule with non-finite -> AUC 0,
                       as its compute_metrics), valnormal_obj (min holdout_obj; A14 §4 sensitivity only); hpfree_ too.
  sulgul_coverage.csv  folds done per (dataset, hp); non-finite counts.
  sulgul_hp_identity.csv  ubgold vs hpfree run of each (dataset, fold): bit-identical trajectories and scores?
  sulgul_repro.csv     own-rule ORACLE (ubgold), 5-fold mean +- std, vs the paper's Table 1 GUL values (contamination
                       100%; not comparable with our normal-only runs, descriptive) and vs the native diagnostic
                       (poll 0 = our setting on GUL's splits; poll 1 = the paper's setting, like-for-like with PAPER).
  sulgul_native_diag.csv  GUL's own code on its own splits (native_main.py outputs): own rule, LAST, per-split max.
"""
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PARTS = os.environ.get("EP_PARTS", osp.join(osp.dirname(HERE), "parts"))
SCORES = os.environ.get("EP_SCORES", "/N/slate/yujlu/grail_data/ext_sulgul/scores")
NATIVE_OUT = "/N/slate/yujlu/grail_data/ext_sulgul/native/out"
OUT = os.environ.get("EP_OUT", osp.join(HERE, "results"))
METHOD = "SUL-GUL"
PATIENCE = 30
DATASETS = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY",
            "MUTAG"]
# Paper (WWW'26, PDF supplied by the user 2026-10-13, ext/SUL-GUL/"SUL GUL.pdf"): GUL row of Table 1 (p. 907),
# AUC % mean +- std, 5 folds. Setting: FULLY UNSUPERVISED = contamination 100% (all training-split anomalies kept,
# section 5.1 p. 906, footnote 4). The paper reports NO normal-only (contamination 0%) GUL number (Fig. 4, p. 908,
# covers 10-100% and is graphical only). Per the pre-registered rule (NOTES.md Part 1) these values are therefore
# "not comparable" with our normal-only runs, and the A10 §4 check is not evaluable; they are compared descriptively,
# and like-for-like with the native poll_ratio 1 diagnostic (GUL's own code in the paper's setting).
# PROTEINS (paper) = PROTEINS_full (same 1113 graphs, same 3 node-label features). Not in the paper: ENZYMES.
PAPER_SETTING = "fully unsupervised (contamination 100%), Table 1 GUL row, p. 907"
PAPER = {"MUTAG": (93.38, 1.02), "NCI1": (69.13, 2.20), "PROTEINS_full": (79.56, 2.60), "AIDS": (99.63, 0.33),
         "BZR": (75.28, 3.30), "COX2": (72.75, 6.77), "DHFR": (63.87, 2.81), "IMDB-BINARY": (69.84, 5.90),
         "REDDIT-BINARY": (83.66, 1.17), "DD": (78.13, 2.78)}
# anomaly class of GUL's own loader vs UB-GOLD's (data_loader.py:15-16): opposite on these
DIRECTION_MISMATCH = {"BZR", "COX2", "PROTEINS_full", "DD"}


def load_parts():
    runs = {}
    for f in sorted(glob.glob(osp.join(PARTS, f"{METHOD}_*.json"))):
        J = json.load(open(f))
        m = J["meta"]
        if m.get("max_epochs_override") is not None or m["run_id"] != (
                f"{METHOD}_{m['dataset']}_fold{m['fold']}" + ("_hpfree" if m["hp"] == "hpfree" else "")):
            continue                                   # smoke / tagged runs never enter the tables
        runs[(m["dataset"], m["hp"], int(m["fold"]))] = J
    return runs


def finite_pt(r):
    return int(r.get("nonfinite", 0)) == 0 and np.isfinite(r["auroc"])


def ubgold_oracle(T):
    max_auc, best, counter, stop = 0.0, None, 0, None
    for i, r in enumerate(T):
        a = r["auroc"]
        if finite_pt(r) and a > max_auc:
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE:
            stop = i
            break
    return best, stop


def row(J, i, proto, rule="", extra=None, force_chance=False):
    m, r = J["meta"], J["traj"][i]
    nonfin = int(force_chance or not finite_pt(r))
    prev = m["n_test_anomalous"] / m["n_test"]
    d = dict(method=METHOD, dataset=m["dataset"], fold=int(m["fold"]), protocol=proto,
             auroc=50.0 if nonfin else 100 * r["auroc"], auprc=100 * prev if nonfin else 100 * r["auprc"],
             fpr95=95.0 if nonfin else 100 * r["fpr95"], epoch_selected=int(r["epoch"]),
             device="+".join(sorted({("cuda" if "cuda" in s["device"] else "cpu") for s in m["segments"]})),
             hp_config=f"{m['hp']}|{m['ubgold_sh_line']}|topk_eff={m['hparams']['topk_effective']}",
             orientation="native", source="ext_sulgul", run_id=m["run_id"], n_eval=len(J["traj"]), oracle_rule=rule,
             nonfinite_selected=nonfin, variant=m.get("variant", "GUL"))
    d.update(extra or {})
    return d


def identical(Ja, Jb):
    keys = ["auroc", "auprc", "fpr95", "holdout_score", "holdout_obj", "train_loss"]
    same = len(Ja["traj"]) == len(Jb["traj"]) and all(
        (x[k] == y[k]) or (x[k] != x[k] and y[k] != y[k]) for x, y in zip(Ja["traj"], Jb["traj"]) for k in keys)
    za = np.load(Ja["meta"]["scores_npz"]) if osp.exists(Ja["meta"]["scores_npz"]) else None
    zb = np.load(Jb["meta"]["scores_npz"]) if osp.exists(Jb["meta"]["scores_npz"]) else None
    sc = None
    if za is not None and zb is not None:
        sc = all(np.array_equal(za[k], zb[k], equal_nan=True) for k in ("test", "hold"))
    return same, sc


def native_summary():
    rows = []
    for f in sorted(glob.glob(osp.join(NATIVE_OUT, "*_poll*_s*.json"))):
        J = json.load(open(f))
        rows.append(J)
    out = []
    groups = {}
    for J in rows:
        groups.setdefault((J["dataset"], J["poll_ratio"]), {})[J["split"]] = J
    for (DS, p), G in sorted(groups.items()):
        d = dict(dataset=DS, poll_ratio=p, splits_done=len(G))
        if len(G) == 5:
            A = np.array([G[k]["auc"] for k in range(5)])           # GUL's compute_metrics (failures -> 0)
            mean = A.mean(0)
            idx = int(np.argmax(mean))
            d.update(own_rule=round(100 * mean[idx], 2), own_rule_std=round(100 * np.std(A[:, idx]), 2),
                     own_rule_epoch=idx + 1, last=round(100 * A[:, -1].mean(), 2),
                     last_std=round(100 * A[:, -1].std(), 2), per_split_max=round(100 * A.max(1).mean(), 2),
                     n_train=G[0]["n_train"], n_train_anom=G[0]["n_train_anom"], n_test=G[0]["n_test"],
                     n_test_anom=G[0]["n_test_anom"], epochs=A.shape[1])
        out.append(d)
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load_parts()
    rows, extra, cov, ident = [], [], [], []
    for DS in DATASETS:
        for hp in ("ubgold", "hpfree"):
            pre = "" if hp == "ubgold" else "hpfree_"
            folds = {k: runs[(DS, hp, k)] for k in range(5) if (DS, hp, k) in runs}
            nf = {k: sum(1 for r in J["traj"] if not finite_pt(r)) for k, J in folds.items()}
            cov.append(dict(method=METHOD, dataset=DS, hp=hp, folds_done=len(folds),
                            folds=",".join(map(str, sorted(folds))),
                            nonfinite_points=",".join(f"{k}:{v}" for k, v in sorted(nf.items()))))
            for k, J in folds.items():
                T = J["traj"]
                all_bad = not any(finite_pt(r) for r in T)
                b, stop = ubgold_oracle(T)
                if b is None:   # no finite point (A15 rule 4)
                    rows.append(row(J, len(T) - 1, pre + "oracle_ubgold", "ubgold_patience30", force_chance=True))
                else:
                    rows.append(row(J, b, pre + "oracle_ubgold", "ubgold_patience30",
                                    dict(ubgold_stop_epoch=(T[stop]["epoch"] if stop is not None else T[-1]["epoch"]))))
                rows.append(row(J, len(T) - 1, pre + "last", force_chance=all_bad))
                hold = np.array([r["holdout_score"] for r in T], dtype=float)
                if np.isfinite(hold).any() and not all_bad:
                    rows.append(row(J, int(np.nanargmin(np.where(np.isfinite(hold), hold, np.nan))), pre + "valnormal"))
                else:
                    rows.append(row(J, len(T) - 1, pre + "valnormal", force_chance=True))
                a = np.array([r["auroc"] if finite_pt(r) else np.nan for r in T], dtype=float)
                if np.isfinite(a).any():
                    extra.append(row(J, int(np.nanargmax(a)), pre + "oracle_nostop", "per_fold_max"))
                obj = np.array([r.get("holdout_obj", np.nan) for r in T], dtype=float)
                if np.isfinite(obj).any():
                    extra.append(row(J, int(np.nanargmin(np.where(np.isfinite(obj), obj, np.nan))), pre + "valnormal_obj"))
            if len(folds) == 5:
                L = {len(J["traj"]) for J in folds.values()}
                assert len(L) == 1, (DS, hp, L)
                A = np.array([[r["auroc"] if finite_pt(r) else np.nan for r in folds[k]["traj"]] for k in range(5)])
                ok = np.isfinite(A).all(0)
                if ok.any():
                    idx = int(np.argmax(np.where(ok, A.mean(0), -np.inf)))
                    for k in range(5):
                        rows.append(row(folds[k], idx, pre + "oracle", "gul_own"))
                else:
                    for k in range(5):
                        rows.append(row(folds[k], len(folds[k]["traj"]) - 1, pre + "oracle", "gul_own", force_chance=True))
                Av = np.array([[r.get("auroc_gul", 0.0) for r in folds[k]["traj"]] for k in range(5)])
                idv = int(np.argmax(Av.mean(0)))
                for k in range(5):
                    extra.append(row(folds[k], idv, pre + "oracle_verbatim", "gul_own_verbatim"))
        for k in range(5):
            if (DS, "ubgold", k) in runs and (DS, "hpfree", k) in runs:
                Ja, Jb = runs[(DS, "ubgold", k)], runs[(DS, "hpfree", k)]
                s, sc = identical(Ja, Jb)
                ident.append(dict(dataset=DS, fold=k, traj_identical=s, scores_identical=sc,
                                  threads=(Ja["meta"]["threads"], Jb["meta"]["threads"]),
                                  hparams_identical=Ja["meta"]["hparams"] == Jb["meta"]["hparams"]))
    F = pd.DataFrame(rows)
    F.to_csv(osp.join(OUT, "sulgul_folds.csv"), index=False)
    pd.DataFrame(extra).to_csv(osp.join(OUT, "sulgul_extra.csv"), index=False)
    pd.DataFrame(cov).to_csv(osp.join(OUT, "sulgul_coverage.csv"), index=False)
    pd.DataFrame(ident).to_csv(osp.join(OUT, "sulgul_hp_identity.csv"), index=False)
    N = native_summary()
    N.to_csv(osp.join(OUT, "sulgul_native_diag.csv"), index=False)
    rep = []
    X = pd.DataFrame(extra)
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
            if len(X):
                xm = X[(X.dataset == DS) & (X.protocol == "oracle_nostop")].auroc
                d["per_fold_max"] = round(xm.mean(), 2) if len(xm) else None
            d["own_rule_epoch"] = int(g[g.protocol == "oracle"].epoch_selected.iloc[0]) if len(x) else None
            d["direction_vs_gul_loader"] = "opposite" if DS in DIRECTION_MISMATCH else "same"
            if DS in PAPER and len(x) == 5:
                d["paper_c100"] = f"{PAPER[DS][0]:.2f} ± {PAPER[DS][1]:.2f}"
                d["diff_ours_minus_paper_c100"] = round(x.mean() - PAPER[DS][0], 2)   # descriptive only
                d["a10_check"] = "not evaluable (paper has no normal-only number; settings not comparable)"
            else:
                d["paper_c100"] = "not in paper"
            if len(N):
                gul = {"IMDB-BINARY": "IMDB-BINARY"}.get(DS, DS)
                for pr in (0, 1):
                    n = N[(N.dataset == gul) & (N.poll_ratio == pr)]
                    if len(n) and "own_rule" in n and n.own_rule.notna().all():
                        d[f"native_poll{pr}_own_rule"] = float(n.own_rule.iloc[0])
                        if pr == 1 and DS in PAPER:      # like-for-like: GUL's own code in the paper's setting
                            d["native_poll1_minus_paper"] = round(float(n.own_rule.iloc[0]) - PAPER[DS][0], 2)
            rep.append(d)
    R = pd.DataFrame(rep)
    R.to_csv(osp.join(OUT, "sulgul_repro.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(cov).pivot(index="dataset", columns="hp", values="folds_done"))
        print(R.to_string(index=False))
        if len(N):
            print(N.to_string(index=False))
        if len(ident):
            I = pd.DataFrame(ident)
            print("ubgold==hpfree bit-identical:", int(I.traj_identical.sum()), "/", len(I), "trajectories;",
                  int((I.scores_identical == True).sum()), "/", len(I), "score files")


if __name__ == "__main__":
    main()
