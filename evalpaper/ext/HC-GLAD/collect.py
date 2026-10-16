"""Aggregate finished HC-GLAD runs (prelim/evalpaper/ext/parts/HC-GLAD_*.json) into the harness's eval_folds.csv
schema. Run on the login node at any time (reads only finished runs):
    PYTHONDONTWRITEBYTECODE=1 /N/slate/yujlu/envs/grail-prelim/bin/python -B prelim/evalpaper/ext/HC-GLAD/collect.py

Writes to prelim/evalpaper/ext/HC-GLAD/results/ (override with EP_RES):
  ext_hcglad_folds.csv   one row per (dataset, fold, protocol); columns of harness eval_folds.csv + oracle_rule,
                         oracle_folds, nonfinite_selected
     protocol  oracle          ORACLE, the method's OWN rule (main.py:193-199): per (dataset, hp) the common epoch
                               with max mean test AUROC over the folds (ties: lower std ddof 0, then earlier); only
                               epoch % eval_freq == 0 and only epochs finite in every fold (A15.1); oracle_rule 'own'
               oracle_ubgold   ORACLE, UB-GOLD rule: per fold max test AUROC, patience 30, non-finite points skipped
               last            final epoch (A15.3: non-finite test scores -> chance, flagged)
               valnormal       min mean held-out-normal anomaly score (A3), ties -> earliest; non-finite held-out
                               means never selected (A15.2)
               hpfree_<p>      the same four for hp = hpfree
     auroc/auprc/fpr95 in percent; orientation 'native'; source 'ext_hcglad'
     A15 chance level: AUROC 50, AUPRC = anomaly rate of the test fold, FPR95 95; nonfinite_selected = 1
  ext_hcglad_folds_extra.csv  oracle_nostop (per-fold max), valnormal_obj (held-out training objective, chunks of
                         batch_size), last_<component> (test AUROC of L_g, L_n, L_g^hyper, L_n^hyper at LAST)
  ext_hcglad_coverage.csv     folds done per (dataset, hp), runs with non-finite points / own-code-crash points
  ext_hcglad_repro.csv        own-rule ORACLE (ubgold hp) 5-fold mean +- std vs the paper table (arXiv 2407.02057v2)
  ext_hcglad_flags.csv        A14 "VAL-NORMAL-degenerate" flag per hp (descriptive)
"""
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
PARTS = os.environ.get("EP_PARTS", osp.join(HERE, "..", "parts"))
RES = os.environ.get("EP_RES", osp.join(HERE, "results"))
METHOD = "HC-GLAD"
PATIENCE = 30
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
SEC = ["MUTAG"]
# HC-GLAD paper (arXiv 2407.02057v2, main table), AUC % mean, std (own reporting rule)
PAPER = {"PROTEINS_full": (77.51, 2.58), "ENZYMES": (65.39, 6.23), "AIDS": (99.51, 0.38), "DHFR": (61.43, 4.27),
         "BZR": (75.75, 9.11), "COX2": (59.98, 7.44), "REDDIT-BINARY": (79.09, 2.52), "IMDB-BINARY": (60.92, 3.39)}
REPRO_CRITERION = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "PROTEINS_full", "REDDIT-BINARY"]   # released lines
COLS = ["method", "dataset", "fold", "protocol", "auroc", "auprc", "fpr95", "epoch_selected", "device", "hp_config",
        "orientation", "source", "run_id", "n_eval", "ubgold_stop_epoch", "oracle_rule", "oracle_folds",
        "nonfinite_selected"]


def fin(x):
    return x is not None and x == x and np.isfinite(x)


def ubgold_oracle(traj, eval_freq, key="auroc"):
    max_auc, best, counter, stop_epoch = 0.0, None, 0, None
    for i, r in enumerate(traj):
        if r["epoch"] % eval_freq != 0:
            continue
        a = r.get(key, float("nan"))
        if fin(a) and a > max_auc:
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE:
            stop_epoch = r["epoch"]
            break
    return best, stop_epoch


def pick(traj, key, mode):
    vals = np.array([np.nan if not fin(r.get(key)) else r.get(key) for r in traj], dtype=float)
    if np.all(np.isnan(vals)):
        return None
    return int(np.nanargmin(vals) if mode == "min" else np.nanargmax(vals))


def device_of(meta):
    d = meta.get("device_all") or meta.get("device", "")
    return "+".join(sorted({("cuda" if "cuda" in x else "cpu") for x in d.split("+") if x}))


def hp_config(meta, hp):
    h = meta["hparams"]
    keys = ["num_epoch", "eval_freq", "batch_size", "batch_size_test", "lr", "optimizer", "weight_decay", "momentum",
            "hidden_dim", "num_layers", "rw_dim", "dg_dim", "a", "b", "GNN_Encoder", "pooling", "readout", "c"]
    line = meta.get("ubgold_sh_line", "")
    src = "modal" if line.startswith("MODAL") else line.split(": ", 1)[0]
    pre = "ubgold|" + src + "|" if hp == "ubgold" else "hpfree|"
    return pre + " ".join(f"{k}={h[k]}" for k in keys if k in h)


def load_runs():
    out = []
    for fn in sorted(glob.glob(osp.join(PARTS, f"{METHOD}_*.json"))):
        J = json.load(open(fn))
        m = J["meta"]
        stem = f"{METHOD}_{m['dataset']}_fold{m['fold']}"
        if os.environ.get("EP_INCLUDE_TAGGED") != "1" and (
                m["run_id"] not in (stem, stem + "_hpfree") or m.get("max_epochs_override") is not None):
            continue   # tagged smoke / timing / diagnostic runs (EP_INCLUDE_TAGGED=1: self-test only)
        out.append(J)
    return out


def own_rule(runs):
    """main.py:193-199 replayed: common epoch maximising the mean test AUROC over the available folds; only epochs
    finite in every fold (A15.1)."""
    ef = int(runs[0]["meta"]["hparams"]["eval_freq"])
    per = {}
    for J in runs:
        for r in J["traj"]:
            if r["epoch"] % ef == 0:
                per.setdefault(r["epoch"], []).append(r["auroc"])
    n = len(runs)
    cand = [(np.mean(v), np.std(v), e) for e, v in per.items() if len(v) == n and all(fin(x) for x in v)]
    cand.sort(key=lambda x: (-x[0], x[1], x[2]))
    return cand[0][2] if cand else None


def main():
    os.makedirs(RES, exist_ok=True)
    runs = load_runs()
    rows, extra, cov_flags = [], [], {}
    groups = {}
    for J in runs:
        groups.setdefault((J["meta"]["dataset"], J["meta"].get("hp", "ubgold")), []).append(J)
    for (ds, hp), G in sorted(groups.items()):
        pre = "" if hp == "ubgold" else "hpfree_"
        e_own = own_rule(G)
        for J in sorted(G, key=lambda J: J["meta"]["fold"]):
            meta, traj = J["meta"], J["traj"]
            h = meta["hparams"]
            ef, ne = int(h["eval_freq"]), int(h["num_epoch"])
            rate = float(meta["n_test_anomalous"]) / float(meta["n_test"])
            base = dict(method=METHOD, dataset=ds, fold=int(meta["fold"]), device=device_of(meta),
                        hp_config=hp_config(meta, hp), orientation="native", source="ext_hcglad",
                        run_id=meta["run_id"], n_eval=len(traj))
            n_nonfin = sum(1 for r in traj if not fin(r.get("auroc")))
            n_crash = sum(int(r.get("own_code_would_crash", 0)) for r in traj)
            n_hold_nonfin = sum(1 for r in traj if not fin(r.get("holdout_score")))
            cov_flags.setdefault((ds, hp), []).append((n_nonfin, n_crash, n_hold_nonfin))

            def mk(proto, i, more=None, tgt=rows, ka="auroc", kp="auprc", kf="fpr95"):
                if i is None:      # A15.4: no usable point -> chance, flagged
                    d = dict(base, protocol=proto, auroc=50.0, auprc=100 * rate, fpr95=95.0, epoch_selected=-1,
                             nonfinite_selected=1)
                else:
                    r = traj[i]
                    if fin(r.get(ka)):
                        d = dict(base, protocol=proto, auroc=100 * r[ka], auprc=100 * r[kp] if kp in r else np.nan,
                                 fpr95=100 * r[kf] if kf in r else np.nan, epoch_selected=int(r["epoch"]),
                                 nonfinite_selected=0)
                    else:          # A15.3: selected point with non-finite test scores -> chance, flagged
                        d = dict(base, protocol=proto, auroc=50.0, auprc=100 * rate, fpr95=95.0,
                                 epoch_selected=int(r["epoch"]), nonfinite_selected=1)
                d.update(more or {})
                tgt.append(d)
            if e_own is not None:
                i_own = [i for i, r in enumerate(traj) if r["epoch"] == e_own][0]
                mk(pre + "oracle", i_own, dict(oracle_rule="own", oracle_folds=len(G)))
            else:
                mk(pre + "oracle", None, dict(oracle_rule="own", oracle_folds=len(G)))
            b, stop = ubgold_oracle(traj, ef)
            mk(pre + "oracle_ubgold", b, dict(oracle_rule="ubgold", ubgold_stop_epoch=stop if stop is not None else ne))
            last_i = [i for i, r in enumerate(traj) if r["epoch"] == ne]
            assert last_i, (meta["run_id"], ne)
            mk(pre + "last", last_i[0])
            mk(pre + "valnormal", pick(traj, "holdout_score", "min"))
            mk(pre + "oracle_nostop", pick(traj, "auroc", "max"), tgt=extra)
            mk(pre + "valnormal_obj", pick(traj, "holdout_obj", "min"), tgt=extra)
            for c in ("g", "n", "g_hyper", "n_hyper"):
                mk(pre + "last_" + c, last_i[0], tgt=extra, ka="auroc_" + c, kp="-", kf="-")
    df = pd.DataFrame(rows, columns=COLS)
    df = df.sort_values(["dataset", "protocol", "fold"]).reset_index(drop=True)
    df.to_csv(osp.join(RES, "ext_hcglad_folds.csv"), index=False)
    dx = pd.DataFrame(extra)
    if len(dx):
        dx = dx.sort_values(["dataset", "protocol", "fold"]).reset_index(drop=True)
    dx.to_csv(osp.join(RES, "ext_hcglad_folds_extra.csv"), index=False)
    cov = []
    for ds in UB9 + SEC:
        for hp in ("ubgold", "hpfree"):
            fl = cov_flags.get((ds, hp), [])
            cov.append(dict(method=METHOD, dataset=ds, hp=hp, set=("UB9" if ds in UB9 else "secondary"),
                            folds_done=len(groups.get((ds, hp), [])),
                            runs_with_nonfinite_test=sum(1 for x in fl if x[0] > 0),
                            runs_with_own_code_crash_point=sum(1 for x in fl if x[1] > 0),
                            runs_with_nonfinite_holdout=sum(1 for x in fl if x[2] > 0),
                            nonfinite_selected_rows=int(df[(df.dataset == ds) & (df.protocol.str.startswith("hpfree_") == (hp == "hpfree"))].nonfinite_selected.sum()) if len(df) else 0))
    pd.DataFrame(cov).to_csv(osp.join(RES, "ext_hcglad_coverage.csv"), index=False)
    rep = []
    for ds in UB9 + SEC:
        g = df[(df.dataset == ds)]
        if not len(g):
            continue
        row = dict(method=METHOD, dataset=ds)
        for p in ("oracle", "oracle_ubgold", "last", "valnormal", "hpfree_oracle", "hpfree_oracle_ubgold",
                  "hpfree_last", "hpfree_valnormal"):
            gg = g[g.protocol == p]
            row[p + "_mean"] = gg.auroc.mean() if len(gg) else np.nan
            row[p + "_std"] = gg.auroc.std(ddof=0) if len(gg) else np.nan
            if p == "oracle":
                row["folds"] = int(gg.fold.nunique())
                row["oracle_epoch"] = int(gg.epoch_selected.iloc[0]) if len(gg) else None
        pub = PAPER.get(ds)
        row["paper_mean"], row["paper_std"] = (pub if pub else (np.nan, np.nan))
        row["diff_own_oracle"] = row["oracle_mean"] - row["paper_mean"] if pub else np.nan
        row["within_5"] = bool(abs(row["diff_own_oracle"]) <= 5) if pub and row["oracle_mean"] == row["oracle_mean"] else None
        row["counts_for_criterion"] = ds in REPRO_CRITERION and row.get("folds") == 5
        rep.append(row)
    pd.DataFrame(rep).to_csv(osp.join(RES, "ext_hcglad_repro.csv"), index=False)
    # A14 descriptive flag: >= 80% of runs select a point in the first 10% of training AND mean VAL-NORMAL >= 5 below LAST
    fl = []
    for hp, pre in (("ubgold", ""), ("hpfree", "hpfree_")):
        v = df[df.protocol == pre + "valnormal"]
        l = df[df.protocol == pre + "last"]
        if not len(v):
            continue
        ne = {r.run_id: int(json.load(open(osp.join(PARTS, r.run_id + ".json")))["meta"]["hparams"]["num_epoch"]) for r in v.itertuples()}
        early = np.mean([r.epoch_selected <= 0.1 * ne[r.run_id] for r in v.itertuples()])
        gap = l.auroc.mean() - v.auroc.mean()
        fl.append(dict(method=METHOD, hp=hp, runs=len(v), frac_valnormal_first10pct=early,
                       mean_last_minus_valnormal=gap, valnormal_degenerate=bool(early >= 0.8 and gap >= 5)))
    pd.DataFrame(fl).to_csv(osp.join(RES, "ext_hcglad_flags.csv"), index=False)
    print(f"{len(runs)} runs; ext_hcglad_folds.csv: {len(df)} rows; extra {len(dx)} rows -> {RES}")


if __name__ == "__main__":
    main()
