"""Aggregate finished GLADMamba runs (prelim/evalpaper/ext/parts/GLADMamba_*.json) into the harness's
eval_folds.csv schema. Run on the login node at any time (reads only finished runs):
    PYTHONDONTWRITEBYTECODE=1 /N/slate/yujlu/envs/grail-prelim/bin/python -B prelim/evalpaper/ext/GLADMamba/collect.py

Writes to prelim/evalpaper/ext/GLADMamba/results/ (override with EP_RES):
  ext_gladmamba_folds.csv   one row per (dataset, fold, protocol); columns of harness eval_folds.csv + oracle_rule
     protocol  oracle          ORACLE, the method's OWN rule (main.py:165-170): per (dataset, hp) the common epoch
                               with max mean test AUROC over the folds (ties: lower std ddof 0, then earlier);
                               only epoch % eval_freq == 0; oracle_rule = 'own'; oracle_folds = folds used
               oracle_ubgold   ORACLE, UB-GOLD rule: per fold max test AUROC, patience 30 (oracle_rule = 'ubgold')
               last            final epoch
               valnormal       min mean held-out-normal anomaly score (A3), ties -> earliest
               hpfree_<p>      the same four for hp = hpfree
     auroc/auprc/fpr95 in percent; orientation 'native'; source 'ext_gladmamba'
  ext_gladmamba_folds_extra.csv  oracle_nostop (per-fold max), valnormal_raw (held-out raw L_g+L_n),
                               valnormal_obj (weighted training objective on held-out), last_raw (raw-score AUROC)
  ext_gladmamba_coverage.csv     folds done per (dataset, hp)
  ext_gladmamba_repro.csv        own-rule ORACLE (ubgold hp) 5-fold mean +- std vs paper Table 2
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
METHOD = "GLADMamba"
PATIENCE = 30
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
SEC = ["MUTAG"]
# GLADMamba paper (arXiv 2503.17903) Table 2, AUC % mean, std over 5 trials (own reporting rule)
PAPER = {"ENZYMES": (68.39, 4.55), "AIDS": (99.29, 0.47), "DHFR": (63.79, 4.16), "BZR": (77.25, 4.62),
         "COX2": (66.38, 1.40), "NCI1": (73.06, 1.87), "IMDB-BINARY": (69.63, 2.70), "REDDIT-BINARY": (86.12, 0.41)}
COLS = ["method", "dataset", "fold", "protocol", "auroc", "auprc", "fpr95", "epoch_selected", "device", "hp_config",
        "orientation", "source", "run_id", "n_eval", "ubgold_stop_epoch", "oracle_rule", "oracle_folds"]


def ubgold_oracle(traj, eval_freq, key="auroc"):
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
    vals = np.array([np.nan if r.get(key) is None else r.get(key) for r in traj], dtype=float)
    if np.all(np.isnan(vals)):
        return None
    return int(np.nanargmin(vals) if mode == "min" else np.nanargmax(vals))


def device_of(meta):
    d = meta.get("device_all") or meta.get("device", "")
    return "+".join(sorted({("cuda" if "cuda" in x else "cpu") for x in d.split("+") if x}))


def hp_config(meta, hp):
    h = meta["hparams"]
    keys = ["num_epoch", "eval_freq", "batch_size", "batch_size_test", "lr", "hidden_dim", "num_layer", "rw_dim",
            "dg_dim", "alpha", "is_adaptive", "GNN_Encoder", "graph_level_pool", "d_state", "dt_rank", "d_conv", "l"]
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
    """main.py:165-170 replayed: common epoch maximising the mean test AUROC over the available folds."""
    ef = int(runs[0]["meta"]["hparams"]["eval_freq"])
    per = {}
    for J in runs:
        for r in J["traj"]:
            if r["epoch"] % ef == 0:
                per.setdefault(r["epoch"], []).append(r["auroc"])
    n = len(runs)
    cand = [(np.mean(v), np.std(v), e) for e, v in per.items() if len(v) == n and np.all(np.isfinite(v))]
    cand.sort(key=lambda x: (-x[0], x[1], x[2]))
    return cand[0][2] if cand else None


def main():
    os.makedirs(RES, exist_ok=True)
    runs = load_runs()
    rows, extra = [], []
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
            base = dict(method=METHOD, dataset=ds, fold=int(meta["fold"]), device=device_of(meta),
                        hp_config=hp_config(meta, hp), orientation="native", source="ext_gladmamba",
                        run_id=meta["run_id"], n_eval=len(traj))

            def mk(proto, i, more=None, tgt=rows, ka="auroc", kp="auprc", kf="fpr95"):
                r = traj[i]
                d = dict(base, protocol=proto, auroc=100 * r[ka], auprc=100 * r[kp],
                         fpr95=100 * r[kf] if kf in r else np.nan, epoch_selected=int(r["epoch"]))
                d.update(more or {})
                tgt.append(d)
            if e_own is not None:
                i_own = [i for i, r in enumerate(traj) if r["epoch"] == e_own][0]
                mk(pre + "oracle", i_own, dict(oracle_rule="own", oracle_folds=len(G)))
            b, stop = ubgold_oracle(traj, ef)
            mk(pre + "oracle_ubgold", b, dict(oracle_rule="ubgold", ubgold_stop_epoch=stop if stop is not None else ne))
            last_i = [i for i, r in enumerate(traj) if r["epoch"] == ne]
            assert last_i, (meta["run_id"], ne)
            mk(pre + "last", last_i[0])
            mk(pre + "valnormal", pick(traj, "holdout_score", "min"))
            mk(pre + "oracle_nostop", pick(traj, "auroc", "max"), tgt=extra)
            for key, nm in (("holdout_raw", "valnormal_raw"), ("holdout_obj", "valnormal_obj")):
                v = pick(traj, key, "min")
                if v is not None:
                    mk(pre + nm, v, tgt=extra)
            mk(pre + "last_raw", last_i[0], tgt=extra, ka="auroc_raw", kp="auprc_raw", kf="-")
    df = pd.DataFrame(rows, columns=COLS)
    df = df.sort_values(["dataset", "protocol", "fold"]).reset_index(drop=True)
    df.to_csv(osp.join(RES, "ext_gladmamba_folds.csv"), index=False)
    dx = pd.DataFrame(extra)
    if len(dx):
        dx = dx.sort_values(["dataset", "protocol", "fold"]).reset_index(drop=True)
    dx.to_csv(osp.join(RES, "ext_gladmamba_folds_extra.csv"), index=False)
    cov = []
    for ds in UB9 + SEC:
        for hp in ("ubgold", "hpfree"):
            n = len(groups.get((ds, hp), []))
            cov.append(dict(method=METHOD, dataset=ds, hp=hp, set=("UB9" if ds in UB9 else "secondary"), folds_done=n))
    pd.DataFrame(cov).to_csv(osp.join(RES, "ext_gladmamba_coverage.csv"), index=False)
    rep = []
    for ds in UB9 + SEC:
        g = df[(df.dataset == ds)]
        if not len(g):
            continue
        row = dict(method=METHOD, dataset=ds)
        for p in ("oracle", "oracle_ubgold", "last", "valnormal", "hpfree_oracle", "hpfree_last", "hpfree_valnormal"):
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
        rep.append(row)
    pd.DataFrame(rep).to_csv(osp.join(RES, "ext_gladmamba_repro.csv"), index=False)
    print(f"{len(runs)} runs; ext_gladmamba_folds.csv: {len(df)} rows; extra {len(dx)} rows -> {RES}")


if __name__ == "__main__":
    main()
