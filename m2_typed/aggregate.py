"""Aggregate the M2 typed kill test.

Inputs
  m2_typed/parts/*.json                      one deep training run each (run_m2.py; tagged runs ignored)
  results/m2_grail0_rerun.csv         GRAIL-0 + controls re-scored on the exported sets
  results/synthetic_{attrib_detection,flip,bm_detection}.csv   original GRAIL-0 numbers
Protocols per (method, data set, feature variant, seed, anomaly type), from ONE training run:
  oracle     UB-GOLD's rule replayed on THIS type's test-AUROC trajectory: running max over evaluation
             points with epoch % eval_freq == 0, strict improvement, stop after 30 evaluations without
             improvement (deep_rerun/aggregate.py:ubgold_oracle, copied); AUPRC/FPR95 at that point.
  last       final epoch (num_epoch).
  valnormal  minimum held-out-normal loss (earliest on ties); same checkpoint for every type.
  extras (m2_typed_runs_extra.csv): oracle_nostop (max over the run), trainloss (min training loss),
             and for SIGNET on GX the combined-batch scores (all 1500 test graphs in one InfoNCE batch)
             at the valnormal / last checkpoints.
Outputs (results/)
  m2_typed_runs.csv      method, feature_variant, dataset, anomaly_type, seed, criterion, auroc, auprc,
                         fpr95, epoch_selected (+ ubgold_stop_epoch, source); metrics as fractions.
                         Deep methods: criteria oracle / last / valnormal. GRAIL-0 and controls:
                         criterion 'training-free'.
  m2_typed_runs_extra.csv
  m2_typed_summary.csv   profile matrix, long format: 5-seed mean +- sample std (ddof = 1) in % (x100).
  m2_typed_verdict.json  the pre-registered T2 verdict (F2 primary, F1 secondary).
  m2_typed/tables.md     markdown tables pasted into NOTES.md.
"""
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
RES0 = osp.join(osp.dirname(HERE), "results")
RES = os.environ.get("M2_RES", RES0)  # output dir (tests only override it)
PARTS = os.environ.get("M2_PARTS", osp.join(HERE, "parts"))
PATIENCE = 30
METHODS = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD"]
COLS = [("GX", "T1a"), ("GX", "T1b"), ("GX", "T2"), ("GX", "T3a"), ("GX", "T3b"),
        ("SC20-a0.60", "taua0.60"), ("SC20-a0.80", "taua0.80"), ("SC20-a0.20", "taua0.20"),
        ("BM-MT", "MT"), ("BM-MN", "MN"), ("BM-MS", "MS")]
COLNAME = {c: (c[1] if c[0] == "GX" else {"SC20-a0.60": "SC a0.6", "SC20-a0.80": "SC a0.8",
                                           "SC20-a0.20": "SC a0.2"}.get(c[0], c[0])) for c in COLS}
CRIT = ["valnormal", "last", "oracle"]
CRITNAME = {"valnormal": "VAL-NORMAL", "last": "LAST", "oracle": "ORACLE", "training-free": "-"}


def pick(traj, key, mode):  # deep_rerun/aggregate.py, copied
    vals = np.array([r.get(key, np.nan) for r in traj], dtype=float)
    if np.all(np.isnan(vals)):
        return None
    return int(np.nanargmin(vals) if mode == "min" else np.nanargmax(vals))


def ubgold_oracle(traj, eval_freq):  # deep_rerun/aggregate.py, copied
    """Replicates the detectors' fit(): max_AUC=0; save if val_auc > max_AUC; patience 30."""
    max_auc, best, counter, stop_epoch = 0.0, None, 0, None
    for i, r in enumerate(traj):
        if r["epoch"] % eval_freq != 0:
            continue
        a = r["auroc"]
        if a == a and a > max_auc:  # NaN never improves (UB-GOLD would crash on NaN scores)
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE:
            stop_epoch = r["epoch"]
            break
    return best, stop_epoch


def deep_rows():
    rows, extra = [], []
    files = sorted(glob.glob(osp.join(PARTS, "*.json")))
    for fn in files:
        J = json.load(open(fn))
        meta, traj = J["meta"], J["traj"]
        base = f"{meta['method']}_{meta['dataset']}_{meta['variant']}_s{meta['data_seed']}"
        if meta["run_id"] != base:
            continue  # tagged (smoke) runs
        hp = meta["hparams"]
        ef, ne = int(hp["eval_freq"]), int(hp["num_epoch"])
        il = [i for i, r in enumerate(traj) if r["epoch"] == ne]
        assert len(il) == 1, fn
        il = il[0]
        iv = pick(traj, "holdout_loss", "min")
        it = pick(traj, "train_loss", "min")
        key = dict(method=meta["method"], feature_variant=meta["variant"], dataset=meta["dataset"],
                   seed=int(meta["data_seed"]))
        for T in list(traj[0]["types"].keys()):
            seq = [dict(epoch=r["epoch"], auroc=r["types"][T]["auroc"]) for r in traj]
            ib, stop = ubgold_oracle(seq, ef)
            io = pick([dict(a=r["types"][T]["auroc"]) for r in traj], "a", "max")

            def rec(i, crit, src="types", **kw):
                m = traj[i][src][T]
                return dict(key, anomaly_type=T, criterion=crit, auroc=m["auroc"], auprc=m["auprc"],
                            fpr95=m["fpr95"], epoch_selected=traj[i]["epoch"], **kw)
            target = rows if T != "ALL" else extra
            target.append(rec(ib, "oracle", ubgold_stop_epoch=stop if stop is not None else ne))
            target.append(rec(il, "last"))
            target.append(rec(iv, "valnormal"))
            extra.append(rec(io, "oracle_nostop"))
            extra.append(rec(it, "trainloss"))
            if "types_combined" in traj[0]:
                extra.append(rec(iv, "valnormal_combinedbatch", src="types_combined"))
                extra.append(rec(il, "last_combinedbatch", src="types_combined"))
    df = pd.DataFrame(rows)
    if len(df):
        df["source"] = "m2_typed/parts (run_m2.py)"
    return df, pd.DataFrame(extra)


def grail_rows():
    """GRAIL-0 + controls. Original CSV numbers if the rerun reproduced every original AUROC
    on the exported sets; the rerun's numbers otherwise (and always for labonly_*)."""
    p = osp.join(os.environ.get("M2_G0", RES0), "m2_grail0_rerun.csv")
    if not osp.exists(p):
        return pd.DataFrame(), {"verified": False, "reason": "no rerun file"}
    rr = pd.read_csv(p)
    cmp_ = rr.dropna(subset=["auroc_original"])
    nonlab = rr[~rr.method.str.contains("labonly_")]
    info = {"rows_rerun": int(len(rr)), "rows_compared": int(len(cmp_)),
            "max_abs_diff_auroc": float(cmp_["abs_diff_auroc"].max()) if len(cmp_) else None,
            "rows_without_original": sorted(rr[rr.auroc_original.isna()].method.unique().tolist())}
    info["verified"] = bool(len(cmp_) == len(nonlab) and len(cmp_) > 0 and info["max_abs_diff_auroc"] < 1e-9)
    out = rr.copy()
    out["source"] = np.where(out.auroc_original.notna() & info["verified"],
                             "original synthetic_*.csv (rerun identical)", "m2_grail0_rerun.csv")
    # report the original AUROC where verified (identical anyway); AUPRC/FPR95 from the rerun
    out.loc[out.auroc_original.notna() & info["verified"], "auroc"] = out["auroc_original"]
    out["feature_variant"] = np.where(out.dataset != "GX", "C1",
                                      np.where(out.method.str.contains("labonly_"), "F1", "F2"))
    out["method"] = "GRAIL0:" + out["method"]
    out["criterion"] = "training-free"
    out["epoch_selected"] = np.nan
    cols = ["method", "feature_variant", "dataset", "anomaly_type", "seed", "criterion", "auroc", "auprc",
            "fpr95", "epoch_selected", "source"]
    out = out[cols]
    # one-sided flip references on Syn-Com (original synthetic_flip.csv; same sets, see above)
    f = pd.read_csv(osp.join(RES0, "synthetic_flip.csv"))
    f = f[(f.n == 20) & np.isclose(f.tau0, 0.4) & f.taua.round(2).isin([0.6, 0.8, 0.2])
          & f.method.isin(["oracle_resid", "true_nll", "gae_bce", "gae_frob", "lgae_frob", "muse_lw"])]
    ref = pd.DataFrame({"method": "REF:" + f.method, "feature_variant": "I_n/none",
                        "dataset": ["SC20-a%.2f" % t for t in f.taua], "anomaly_type": ["taua%.2f" % t for t in f.taua],
                        "seed": f.seed.astype(int), "criterion": "training-free", "auroc": f.auroc, "auprc": f.auprc,
                        "fpr95": f.fpr95, "epoch_selected": np.nan,
                        "source": "original synthetic_flip.csv" + (" (sets verified identical)" if info["verified"] else "")})
    parts = [out, ref]
    ph = osp.join(os.environ.get("M2_G0", RES0), "m2_posthoc_controls.csv")
    if osp.exists(ph):  # POST HOC label-aware controls (posthoc_controls.py; not pre-registered)
        q = pd.read_csv(ph)
        parts.append(pd.DataFrame({"method": "POSTHOC:" + q.method,
                                   "feature_variant": np.where(q.dataset == "GX", "F1", "C1"),
                                   "dataset": q.dataset, "anomaly_type": q.anomaly_type, "seed": q.seed.astype(int),
                                   "criterion": "training-free", "auroc": q.auroc, "auprc": q.auprc, "fpr95": q.fpr95,
                                   "epoch_selected": np.nan, "source": "m2_posthoc_controls.csv (POST HOC)"}))
    return pd.concat(parts, ignore_index=True), info


def summarize(df):
    g = df.groupby(["method", "feature_variant", "criterion", "dataset", "anomaly_type"], dropna=False)
    s = g.agg(n_seeds=("seed", "nunique"), auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
              auroc_min=("auroc", "min"), auroc_max=("auroc", "max"), auprc_mean=("auprc", "mean"),
              auprc_std=("auprc", "std"), fpr95_mean=("fpr95", "mean"),
              epoch_selected_mean=("epoch_selected", "mean")).reset_index()
    for c in ("auroc_mean", "auroc_std", "auroc_min", "auroc_max", "auprc_mean", "auprc_std", "fpr95_mean"):
        s[c] = 100.0 * s[c]
    s["auroc"] = [f"{m:.1f} ± {sd:.1f}" if n > 1 else f"{m:.1f}" for m, sd, n in
                  zip(s.auroc_mean, s.auroc_std, s.n_seeds)]
    return s


def verdict(runs, variant):
    d = runs[(runs.dataset == "GX") & (runs.anomaly_type == "T2") & (runs.feature_variant == variant)
             & runs.method.isin(METHODS)]
    A = {}
    for m in METHODS:
        for c in ("valnormal", "oracle", "last"):
            v = d[(d.method == m) & (d.criterion == c)].sort_values("seed")
            A[(m, c)] = dict(n=int(len(v)), mean=float(100 * v.auroc.mean()) if len(v) else None,
                             sd=float(100 * v.auroc.std()) if len(v) > 1 else None,
                             min=float(100 * v.auroc.min()) if len(v) else None,
                             max=float(100 * v.auroc.max()) if len(v) else None,
                             per_seed=[round(100 * x, 2) for x in v.auroc])
    complete = all(A[(m, "valnormal")]["n"] == 5 and A[(m, "oracle")]["n"] == 5 for m in METHODS)
    dead = [m for m in METHODS if A[(m, "valnormal")]["n"] >= 3 and A[(m, "valnormal")]["mean"] >= 80]
    if dead:
        v = "DEAD"
    elif complete and all(A[(m, "valnormal")]["mean"] <= 60 for m in METHODS) and \
            sum(A[(m, "oracle")]["mean"] <= 70 for m in METHODS) >= 3:
        v = "ALIVE"
    else:
        v = "INCONCLUSIVE"
    inverted = [m for m in METHODS if A[(m, "valnormal")]["mean"] is not None and A[(m, "valnormal")]["mean"] <= 30]
    return {"variant": variant, "verdict": v, "complete": complete, "dead_by": dead,
            "valnormal_le60": {m: (A[(m, 'valnormal')]["mean"] is not None and A[(m, 'valnormal')]["mean"] <= 60) for m in METHODS},
            "oracle_le70": {m: (A[(m, 'oracle')]["mean"] is not None and A[(m, 'oracle')]["mean"] <= 70) for m in METHODS},
            "inversion_flag": inverted,
            "A": {f"{m}|{c}": A[(m, c)] for (m, c) in A}}


def md_matrix(summ, row_specs, title):
    cells = {(r.method, r.feature_variant, r.criterion, r.dataset, r.anomaly_type): r.auroc for r in summ.itertuples()}
    head = "| " + " | ".join(["method", "criterion"] + [COLNAME[c] for c in COLS]) + " |"
    sep = "|" + "---|" * (len(COLS) + 2)
    out = [f"**{title}**", "", head, sep]
    for label, method, crit, var_gx, var_other in row_specs:
        vals = []
        for (ds, t) in COLS:
            v = var_gx if ds == "GX" else var_other
            vals.append(cells.get((method, v, crit, ds, t), "") if v else "")
        if any(vals):
            out.append("| " + " | ".join([label, CRITNAME.get(crit, crit)] + vals) + " |")
    return "\n".join(out)


def main():
    deep, extra = deep_rows()
    gr, ginfo = grail_rows()
    runs = pd.concat([deep, gr], ignore_index=True)
    cols = ["method", "feature_variant", "dataset", "anomaly_type", "seed", "criterion", "auroc", "auprc", "fpr95",
            "epoch_selected", "ubgold_stop_epoch", "source"]
    for c in cols:
        if c not in runs:
            runs[c] = np.nan
    runs = runs[cols].sort_values(["method", "feature_variant", "dataset", "anomaly_type", "criterion", "seed"])
    runs.to_csv(osp.join(RES, "m2_typed_runs.csv"), index=False)
    if len(extra):
        extra.sort_values(["method", "feature_variant", "dataset", "anomaly_type", "criterion", "seed"]).to_csv(
            osp.join(RES, "m2_typed_runs_extra.csv"), index=False)
    summ = summarize(runs)
    summ.to_csv(osp.join(RES, "m2_typed_summary.csv"), index=False)
    esumm = summarize(extra) if len(extra) else pd.DataFrame()
    V = {"F2_primary": verdict(runs, "F2"), "F1_secondary": verdict(runs, "F1"), "grail0_identity": ginfo,
         "n_deep_runs": int(deep[["method", "feature_variant", "dataset", "seed"]].drop_duplicates().shape[0]) if len(deep) else 0}
    with open(osp.join(RES, "m2_typed_verdict.json"), "w") as f:
        json.dump(V, f, indent=1)

    # ------------------------------------------------------------------ markdown tables
    md = []
    deep_specs = []
    for crit in CRIT:
        for m in METHODS:
            deep_specs.append((m, m, crit, "F2", "C1"))
    md.append(md_matrix(summ, deep_specs, "Deep methods, primary variant (GX columns: F2 = labels + x; Syn-Com/BM: Constant(1)). AUROC x100, mean ± sd over 5 seeds."))
    f1_specs = [(m, m, crit, "F1", None) for crit in CRIT for m in METHODS]
    md.append(md_matrix(summ, f1_specs, "Deep methods, F1 (labels only, UB-GOLD TU style); GX columns only."))
    gspecs = [("GRAIL-0 sum-z", "GRAIL0:grail0_sum_z", "training-free", "F2", "C1"),
              ("GRAIL-0 max-z", "GRAIL0:grail0_max_z", "training-free", "F2", "C1"),
              ("ch S0 (size)", "GRAIL0:ch_S0", "training-free", "F2", "C1"),
              ("ch A (composition)", "GRAIL0:ch_A", "training-free", "F2", "C1"),
              ("ch B (node role)", "GRAIL0:ch_B", "training-free", "F2", "C1"),
              ("ch C (attr. given role)", "GRAIL0:ch_C", "training-free", "F2", None),
              ("GRAIL-0 sum-z, labels-only C", "GRAIL0:grail0_labonly_sum_z", "training-free", "F1", None),
              ("ch C, labels only", "GRAIL0:ch_labonly_C", "training-free", "F1", None),
              ("ctrl size LW", "GRAIL0:ctrl_size_lw", "training-free", "F2", "C1"),
              ("ctrl graphlet counts iForest", "GRAIL0:ctrl_gc_iforest", "training-free", "F2", "C1"),
              ("ctrl graphlet counts LW", "GRAIL0:ctrl_gc_lw", "training-free", "F2", "C1"),
              ("ctrl counts + label hist iForest", "GRAIL0:ctrl_gclab_iforest", "training-free", "F2", None),
              ("ctrl counts + label hist LW", "GRAIL0:ctrl_gclab_lw", "training-free", "F2", None),
              ("ctrl attribute pooling LW", "GRAIL0:ctrl_attrpool_lw", "training-free", "F2", None),
              ("POST HOC ctrl WL-subtree iForest", "POSTHOC:ctrl_wl_iforest", "training-free", "F1", "C1"),
              ("POST HOC ctrl edge-label pairs LW", "POSTHOC:ctrl_edgelab_lw", "training-free", "F1", None),
              ("POST HOC ctrl edge-label pairs iForest", "POSTHOC:ctrl_edgelab_iforest", "training-free", "F1", None),
              ("ref GIN-GAE BCE (one-sided)", "REF:gae_bce", "training-free", None, "I_n/none"),
              ("ref GIN-GAE Frob (one-sided)", "REF:gae_frob", "training-free", None, "I_n/none"),
              ("ref exact NLL (one-sided)", "REF:true_nll", "training-free", None, "I_n/none"),
              ("ref oracle residual (one-sided)", "REF:oracle_resid", "training-free", None, "I_n/none")]
    md.append(md_matrix(summ, gspecs, "GRAIL-0, channels, controls and one-sided references on the same sets (training-free)."))
    # T2 verdict table
    vt = ["**Pre-registered T2 statistic A(m, c): mean T2 AUROC x100 over seeds 0-4 [min, max]**", "",
          "| variant | method | VAL-NORMAL | ORACLE | LAST | per-seed VAL-NORMAL |", "|---|---|---|---|---|---|"]
    for vk in ("F2_primary", "F1_secondary"):
        A = V[vk]["A"]
        for m in METHODS:
            def f(c):
                a = A[f"{m}|{c}"]
                return "" if a["mean"] is None else f"{a['mean']:.1f} ± {a['sd'] or 0:.1f} [{a['min']:.1f}, {a['max']:.1f}] (n={a['n']})"
            vt.append(f"| {V[vk]['variant']} | {m} | {f('valnormal')} | {f('oracle')} | {f('last')} | {A[f'{m}|valnormal']['per_seed']} |")
    vt.append("")
    for vk in ("F2_primary", "F1_secondary"):
        vt.append(f"- {vk}: **{V[vk]['verdict']}** (complete={V[vk]['complete']}; dead_by={V[vk]['dead_by']}; "
                  f"VAL-NORMAL<=60: {V[vk]['valnormal_le60']}; ORACLE<=70: {V[vk]['oracle_le70']}; "
                  f"inversion flag (VAL-NORMAL<=30): {V[vk]['inversion_flag']})")
    md.append("\n".join(vt))
    # diagnostics
    if len(esumm):
        dg = ["**Diagnostics (m2_typed_runs_extra.csv): AUROC x100, mean ± sd**", "",
              "| method | variant | criterion | " + " | ".join(COLNAME[c] for c in COLS) + " | GX ALL |",
              "|" + "---|" * (len(COLS) + 4)]
        cells = {(r.method, r.feature_variant, r.criterion, r.dataset, r.anomaly_type): r.auroc for r in esumm.itertuples()}
        for crit in ("oracle_nostop", "trainloss", "valnormal_combinedbatch", "valnormal", "last"):
            for m in METHODS:
                for v in ("F2", "F1"):
                    vals = [cells.get((m, v if ds == "GX" else "C1", crit, ds, t), "") for ds, t in COLS]
                    vals.append(cells.get((m, v, crit, "GX", "ALL"), ""))
                    if any(vals):
                        dg.append(f"| {m} | {v} | {crit} | " + " | ".join(vals) + " |")
        md.append("\n".join(dg))
        ep = summ[summ.method.isin(METHODS)].groupby(["method", "criterion"]).epoch_selected_mean.mean().unstack()
        lines = ["| method | " + " | ".join(ep.columns) + " |", "|" + "---|" * (len(ep.columns) + 1)]
        lines += ["| " + m + " | " + " | ".join(f"{v:.0f}" for v in ep.loc[m]) + " |" for m in ep.index]
        md.append("**Mean selected epoch (over all data sets/types/seeds)**\n\n" + "\n".join(lines))
    md.append("GRAIL-0 identity check: " + json.dumps(ginfo))
    txt = "\n\n".join(md) + "\n"
    open(osp.join(RES if RES != RES0 else HERE, "tables.md"), "w").write(txt)
    notes = osp.join(HERE, "NOTES.md")
    if RES == RES0 and osp.exists(notes):  # fill NOTES.md between markers (if present)
        n = open(notes).read()
        a, b = "<!-- BEGIN:TABLES -->", "<!-- END:TABLES -->"
        if a in n and b in n:
            n = n[:n.index(a) + len(a)] + "\n" + txt + n[n.index(b):]
            open(notes, "w").write(n)
    print("\n\n".join(md))
    print("\nruns:", V["n_deep_runs"], "deep runs;", len(runs), "rows")


if __name__ == "__main__":
    main()
