"""M1: label-free checkpoint-selection criteria from the saved per-graph scores, metrics,
paired bootstrap CIs and the pre-registered verdict (see NOTES.md).

Criteria (all label-free; test labels are read ONLY by `metrics()` for reporting):
  last            final epoch of UB-GOLD's fixed budget (num_epoch)
  valnormal       argmin over eval points of the mean held-out-normal score (= the method's own
                  loss), earliest on ties  (deep_rerun's VAL-NORMAL)
  pseudo          argmax of the mean over applicable families of the pseudo-validation AUROC
                  (held-out normals vs their perturbed copies), later epoch on ties
  ckpt_ensemble   mean over ALL eval points of rank-normalised test scores (ROBOD / temporal
                  ensemble style)
  consensus       checkpoint whose test-score ranking has the highest Spearman correlation with
                  the mean rank over all checkpoints (GLAM-style; transductive), later on ties
  pseudo_ensemble mean of rank-normalised test scores of the top-3 checkpoints by pseudo AUROC
                  (ties -> later epochs first)
Reference (uses test labels): oracle = UB-GOLD's protocol (max test AUROC, patience 30, only
eval points with epoch % eval_freq == 0).
Diagnostics (not part of the verdict): pseudo_<family> (select by one family), oracle_nostop,
aucp (AUCp, arXiv 2606.08742: argmax AUROC of held-out normals vs all unlabeled test graphs).

Outputs: results/m1_selection_folds.csv, m1_selection_summary.csv, m1_selection_by_dataset.csv,
m1_trajectories.csv, m1_corr.csv, m1_family_diag.csv, m1_repro_vs_deep_rerun.csv;
m1_selection/verdict.json and tables.md.

  python -B select.py [--boot 10000]
"""
import os.path as _osp, sys as _sys  # M1: this directory holds select.py, which would shadow the
_HERE_ = _osp.realpath(_osp.dirname(_osp.abspath(__file__)))  # stdlib module `select` (used by
_sys.path[:] = [p for p in _sys.path if _osp.realpath(p or ".") != _HERE_]  # subprocess): drop it from sys.path

import argparse
import warnings
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

HERE = osp.dirname(osp.abspath(__file__))
RES = osp.join(osp.dirname(HERE), "results")
RES0 = RES  # deep_rerun results (read-only)
PARTS = os.environ.get("M1_PARTS", osp.join(HERE, "parts"))
SCORES = os.environ.get("M1_SCORES", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/m1_scores"))
CACHE = os.environ.get("M1_SELECT_CACHE", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/m1_select_cache"))  # per-run results
PATIENCE = 30
METHODS = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD"]
DATASETS = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
FAMILIES = ["rewire", "motif", "roleswap", "labelnoise"]
CRITERIA = ["oracle", "last", "valnormal", "pseudo", "ckpt_ensemble", "consensus", "pseudo_ensemble"]
LABEL_FREE = ["last", "valnormal", "pseudo", "ckpt_ensemble", "consensus", "pseudo_ensemble"]
DIAG = ["oracle_nostop", "aucp"] + [f"pseudo_{f}" for f in FAMILIES]


# ------------------------------------------------------------------------------------------
# metrics (same sklearn calls as UB-GOLD's GAOOD/metric; in %)
# ------------------------------------------------------------------------------------------
def fpr95_sk(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return fpr[np.where(tpr >= 0.95)[0][0]]


def metrics_sklearn(y, s):
    """reference implementation (UB-GOLD's GAOOD/metric calls); slow per call (input validation)."""
    s = np.asarray(s, dtype=np.float64)
    if not np.all(np.isfinite(s)):
        return dict(auroc=np.nan, auprc=np.nan, fpr95=np.nan)
    return dict(auroc=100 * roc_auc_score(y, s), auprc=100 * average_precision_score(y, s), fpr95=100 * fpr95_sk(y, s))


def _binary_clf_curve(y, s):
    """sklearn.metrics._ranking._binary_clf_curve (pos_label=1, no weights)."""
    order = np.argsort(s, kind="mergesort")[::-1]
    s, y = s[order], y[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = 1 + idx - tps
    return fps.astype(np.float64), tps.astype(np.float64)


def metrics(y, s):
    """AUROC / AUPRC (average precision) / FPR@95%TPR in %, numerically identical to sklearn's
    roc_auc_score, average_precision_score and roc_curve(drop_intermediate=True) (checked
    against metrics_sklearn on every run in main())."""
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y).astype(np.int64)
    if not np.all(np.isfinite(s)):
        return dict(auroc=np.nan, auprc=np.nan, fpr95=np.nan)
    fps, tps = _binary_clf_curve(y, s)
    # average precision
    prec = tps / (tps + fps)
    rec = tps / tps[-1]
    prec_r, rec_r = np.r_[prec[::-1], 1.0], np.r_[rec[::-1], 0.0]
    ap = -np.sum(np.diff(rec_r) * prec_r[:-1])
    # ROC with sklearn's drop_intermediate=True
    if fps.size > 2:
        keep = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps_d, tps_d = fps[keep], tps[keep]
    else:
        fps_d, tps_d = fps, tps
    fpr = np.r_[0.0, fps_d] / fps[-1]
    tpr = np.r_[0.0, tps_d] / tps[-1]
    f95 = fpr[np.where(tpr >= 0.95)[0][0]]
    au = test_auroc_rows(s[None, :], y)[0]
    return dict(auroc=au, auprc=100 * ap, fpr95=100 * f95)


def auc_rows(neg, pos):
    """vectorised AUROC (%) per row: neg (E, a) negatives vs pos (E, b) positives; average ranks
    for ties (identical to sklearn's roc_auc_score); NaN for rows with non-finite scores."""
    neg, pos = np.atleast_2d(neg).astype(np.float64), np.atleast_2d(pos).astype(np.float64)
    S = np.concatenate([neg, pos], 1)
    ok = np.all(np.isfinite(S), 1)
    out = np.full(S.shape[0], np.nan)
    if ok.any():
        R = rankdata(S[ok], axis=1)
        a, b = neg.shape[1], pos.shape[1]
        out[ok] = 100 * (R[:, a:].sum(1) - b * (b + 1) / 2.0) / (a * b)
    return out


def test_auroc_rows(T, y):
    y = np.asarray(y).astype(bool)
    return auc_rows(T[:, ~y], T[:, y])


def auc(neg, pos):
    s = np.r_[neg, pos]
    if not np.all(np.isfinite(s)):
        return np.nan
    return 100 * roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], s)


# ------------------------------------------------------------------------------------------
# label-free criteria (inputs never include test labels)
# ------------------------------------------------------------------------------------------
def argmax_later(v):
    v = np.asarray(v, dtype=np.float64)
    if np.all(np.isnan(v)):
        return None
    m = np.nanmax(v)
    return int(np.where(v == m)[0][-1])


def argmin_earlier(v):
    v = np.asarray(v, dtype=np.float64)
    if np.all(np.isnan(v)):
        return None
    return int(np.nanargmin(v))


def rank_norm(T):
    """rank-normalise each checkpoint's test scores (average ranks for ties) to (0, 1]."""
    return np.stack([rankdata(r) / len(r) for r in T], 0)


def pseudo_auc_matrix(Z, fams):
    """pseudo-validation AUROC per eval point and family -> (E, F)."""
    return np.stack([auc_rows(Z["pneg_" + f], Z["ppos_" + f]) for f in fams], 1)


def select_all(Z, fams):
    """All label-free selections for one run. Returns dict criterion -> ('ckpt', idx) or
    ('scores', array) plus diagnostics, and the per-eval-point label-free signals."""
    T, H, ep = Z["test"], Z["hold"], Z["epochs"]
    E = len(ep)
    num_epoch = int(Z["num_epoch"])
    valid = np.all(np.isfinite(T), 1)                         # checkpoints with finite test scores
    hold_mean = np.array([h.mean() if np.all(np.isfinite(h)) else np.nan for h in H])
    PA = pseudo_auc_matrix(Z, fams)
    pmean = np.where(np.all(np.isfinite(PA), 1), PA.mean(1), np.nan)
    out = {}
    out["last"] = ("ckpt", int(np.where(ep == num_epoch)[0][0]))
    out["valnormal"] = ("ckpt", argmin_earlier(hold_mean))
    out["pseudo"] = ("ckpt", argmax_later(pmean))
    R = np.full(T.shape, np.nan)
    R[valid] = rank_norm(T[valid])
    out["ckpt_ensemble"] = ("scores", np.nanmean(R[valid], 0))
    rbar = np.nanmean(R[valid], 0)
    # Spearman(R_e, rbar) = Pearson correlation of the ranks of R_e and of rbar
    rb = rankdata(rbar)
    RR = np.full(T.shape, np.nan)
    RR[valid] = rankdata(R[valid], axis=1)
    rho = np.array([np.corrcoef(RR[e], rb)[0, 1] if valid[e] else np.nan for e in range(E)])
    out["consensus"] = ("ckpt", argmax_later(rho))
    order = sorted([e for e in range(E) if np.isfinite(pmean[e]) and valid[e]], key=lambda e: (-pmean[e], -ep[e]))
    top = order[:3]
    out["pseudo_ensemble"] = ("scores", R[top].mean(0) if top else np.full(T.shape[1], np.nan))
    out["_pseudo_top3_epochs"] = [int(ep[e]) for e in top]
    for k, f in enumerate(fams):
        out["pseudo_" + f] = ("ckpt", argmax_later(PA[:, k]))
    # AUCp (Rahman Siddiquee et al., arXiv 2606.08742): held-out normals (neg) vs ALL unlabeled
    # test graphs (pseudo-positive); transductive, label-free; later epoch on ties. Diagnostic.
    aucp = auc_rows(H, T)
    out["aucp"] = ("ckpt", argmax_later(aucp))
    return out, dict(hold_mean=hold_mean, pauc=PA, pauc_mean=pmean, consensus_rho=rho, aucp=aucp)


def ubgold_oracle(auroc, epochs, eval_freq):
    max_auc, best, counter, stop = 0.0, None, 0, None
    for i, (a, e) in enumerate(zip(auroc, epochs)):
        if e % eval_freq != 0:
            continue
        if a == a and a > max_auc:
            max_auc, best, counter = a, i, 0
        else:
            counter += 1
        if counter >= PATIENCE:
            stop = e
            break
    return best, stop


# ------------------------------------------------------------------------------------------
# per-run evaluation
# ------------------------------------------------------------------------------------------
def load_run(json_path):
    J = json.load(open(json_path))
    meta = J["meta"]
    Z = dict(np.load(meta["scores_npz"], allow_pickle=False))
    return meta, J["traj"], Z


def evaluate_run(meta, traj, Z):
    fams = [str(f) for f in Z["families"]]
    y = Z["y_test"]
    ep = Z["epochs"]
    E = len(ep)
    # per-eval-point test AUROC from the saved scores (reporting only); AUPRC/FPR95 lazily
    au = test_auroc_rows(Z["test"], y)
    _M = {}

    class _Lazy:
        def __getitem__(self, e):
            if e not in _M:
                _M[e] = metrics(y, Z["test"][e])
            return _M[e]
    M = _Lazy()
    # consistency with the metrics recorded online by run_m1.py
    online = np.array([100 * r["auroc"] for r in traj])
    assert np.allclose(au, online, atol=1e-6, equal_nan=True), (meta["run_id"], np.nanmax(np.abs(au - online)))
    sel, sig = select_all(Z, fams)
    rows = []
    base = dict(method=meta["method"], dataset=meta["dataset"], fold=meta["fold"])
    dev = meta.get("device_all", meta.get("device", "?"))
    b, stop = ubgold_oracle(au, ep, int(Z["eval_freq"]))
    rows.append(dict(base, criterion="oracle", **M[b], epoch_selected=int(ep[b]),
                     ubgold_stop_epoch=stop if stop is not None else int(Z["num_epoch"])))
    for c in LABEL_FREE + [d for d in DIAG if d.startswith("pseudo_") or d == "aucp"]:
        if c not in sel:
            continue
        kind, v = sel[c]
        if kind == "ckpt":
            if v is None:
                rows.append(dict(base, criterion=c, auroc=np.nan, auprc=np.nan, fpr95=np.nan, epoch_selected=np.nan))
            else:
                rows.append(dict(base, criterion=c, **M[v], epoch_selected=int(ep[v])))
        else:
            rows.append(dict(base, criterion=c, **metrics(y, v), epoch_selected=np.nan))
    o2 = int(np.nanargmax(au))
    rows.append(dict(base, criterion="oracle_nostop", **M[o2], epoch_selected=int(ep[o2])))
    # exactness check of the numpy metrics against sklearn (UB-GOLD's metric calls) on 2 checkpoints
    for e in (b, len(ep) - 1):
        ref, mine = metrics_sklearn(y, Z["test"][e]), M[e]
        assert all(abs(ref[k] - mine[k]) < 1e-9 for k in ref), (meta["run_id"], e, ref, mine)
    for r in rows:
        r["device"] = dev
    # correlations across eval points
    corr = dict(base, n_points=E, device=dev)
    ok = np.isfinite(au) & np.isfinite(sig["pauc_mean"])
    corr["spearman_pauc_mean_vs_auroc"] = spearmanr(sig["pauc_mean"][ok], au[ok]).correlation if ok.sum() > 2 else np.nan
    for k, f in enumerate(fams):
        okf = np.isfinite(au) & np.isfinite(sig["pauc"][:, k])
        corr[f"spearman_pauc_{f}_vs_auroc"] = spearmanr(sig["pauc"][okf, k], au[okf]).correlation if okf.sum() > 2 else np.nan
    okh = np.isfinite(au) & np.isfinite(sig["hold_mean"])
    corr["spearman_neg_holdout_vs_auroc"] = spearmanr(-sig["hold_mean"][okh], au[okh]).correlation if okh.sum() > 2 else np.nan
    okc = np.isfinite(au) & np.isfinite(sig["consensus_rho"])
    corr["spearman_consensus_rho_vs_auroc"] = spearmanr(sig["consensus_rho"][okc], au[okc]).correlation if okc.sum() > 2 else np.nan
    oka = np.isfinite(au) & np.isfinite(sig["aucp"])
    corr["spearman_aucp_vs_auroc"] = spearmanr(sig["aucp"][oka], au[oka]).correlation if oka.sum() > 2 else np.nan
    corr["pauc_mean_at_pseudo"] = float(np.nanmax(sig["pauc_mean"])) if np.any(np.isfinite(sig["pauc_mean"])) else np.nan
    corr["pauc_mean_first"] = float(sig["pauc_mean"][0])
    corr["pauc_mean_last"] = float(sig["pauc_mean"][-1])
    corr["n_pauc_ties_at_max"] = int(np.sum(sig["pauc_mean"] == np.nanmax(sig["pauc_mean"]))) if ok.any() else 0
    corr["pseudo_top3_epochs"] = " ".join(map(str, sel["_pseudo_top3_epochs"]))
    for k, f in enumerate(fams):
        corr[f"pauc_{f}_median"] = float(np.nanmedian(sig["pauc"][:, k]))
    corr["nonfinite_test_ckpts"] = int((~np.all(np.isfinite(Z["test"]), 1)).sum())
    # trajectory rows
    tr = []
    for e in range(E):
        r = dict(base, epoch=int(ep[e]), auroc=au[e], auprc=100 * traj[e]["auprc"], fpr95=100 * traj[e]["fpr95"],
                 holdout_loss=sig["hold_mean"][e], train_loss=traj[e]["train_loss"], pauc_mean=sig["pauc_mean"][e],
                 consensus_rho=sig["consensus_rho"][e])
        for k, f in enumerate(fams):
            r["pauc_" + f] = sig["pauc"][e, k]
        tr.append(r)
    return rows, corr, tr


# ------------------------------------------------------------------------------------------
# aggregation + bootstrap
# ------------------------------------------------------------------------------------------
def method_means(F, crit, metric="auroc"):
    """per-method mean over datasets of the 5-fold mean (all datasets have 5 folds)."""
    g = F[F.criterion == crit].groupby(["method", "dataset"])[metric].mean().reset_index()
    return g.groupby("method")[metric].mean()


def cube(F, crit, metric="auroc"):
    """array [method, dataset, fold] (NaN where missing)."""
    A = np.full((len(METHODS), len(DATASETS), 5), np.nan)
    sub = F[F.criterion == crit]
    for r in sub.itertuples():
        A[METHODS.index(r.method), DATASETS.index(r.dataset), int(r.fold)] = getattr(r, metric)
    return A


def stat_from_cube(D):
    """D [M, Dset, F] differences -> (per-method means [M], 4-method average)."""
    per_m = np.array([np.nanmean(np.nanmean(D[m], 1)) for m in range(D.shape[0])])
    return per_m, float(np.nanmean(per_m))


def fold_bootstrap(D, B, rng):
    """paired fold bootstrap: within every dataset resample the 5 folds with replacement (same
    indices for all methods). Returns arrays [B, M] and [B]."""
    M_, Dn, Fn = D.shape
    idx = rng.integers(0, Fn, size=(B, Dn, Fn))
    per_m = np.empty((B, M_))
    for m in range(M_):
        X = D[m]                                              # [Dn, Fn]
        Xb = np.take_along_axis(np.broadcast_to(X, (B, Dn, Fn)), idx, axis=2)
        per_m[:, m] = np.nanmean(np.nanmean(Xb, 2), 1)
    return per_m, np.nanmean(per_m, 1)


def dataset_bootstrap(D, B, rng):
    """resample datasets with replacement (fold means kept)."""
    M_, Dn, _ = D.shape
    Dm = np.nanmean(D, 2)                                     # [M, Dn]
    idx = rng.integers(0, Dn, size=(B, Dn))
    per_m = np.stack([np.nanmean(Dm[m][idx], 1) for m in range(M_)], 1)
    return per_m, np.nanmean(per_m, 1)


def ci(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", choices=["all", "runs", "aggregate"], default="all",
                    help="runs: evaluate (and cache) per-run results within --budget seconds; "
                         "aggregate: tables/bootstrap/verdict from the cache (login-node friendly)")
    ap.add_argument("--budget", type=float, default=1e9)
    a = ap.parse_args()
    import pickle
    import time as _time
    t0 = _time.time()
    os.makedirs(CACHE, exist_ok=True)
    files = sorted(glob.glob(osp.join(PARTS, "*.json")))
    rows, corrs, trs = [], [], []
    pending = 0
    for fn in files:
        rid = osp.basename(fn)[:-5]
        cpath = osp.join(CACHE, rid + ".pkl")
        if osp.exists(cpath) and osp.getmtime(cpath) >= osp.getmtime(fn):
            if a.stage != "runs":
                r, c, t = pickle.load(open(cpath, "rb"))
                rows += r; corrs += c; trs += t
            continue
        if a.stage == "aggregate" or _time.time() - t0 > a.budget:
            pending += 1
            continue
        meta, traj, Z = load_run(fn)
        if meta["run_id"] != f"{meta['method']}_{meta['dataset']}_fold{meta['fold']}":
            res = ([], [], [])  # tagged smoke/variant runs are not part of the main table
        else:
            r, c, t = evaluate_run(meta, traj, Z)
            res = (r, [c], t)
        pickle.dump(res, open(cpath + ".tmp", "wb"))
        os.replace(cpath + ".tmp", cpath)
        rows += res[0]; corrs += res[1]; trs += res[2]
    if a.stage == "runs" or pending:
        print(f"per-run stage: {len(files)} runs, {pending} not yet evaluated", flush=True)
        if pending:
            return
    if a.stage == "runs":
        return
    analyze(pd.DataFrame(rows), pd.DataFrame(corrs), pd.DataFrame(trs), a.boot, a.seed, RES, HERE)


def key(df):
    return df.assign(_m=df.method.map(METHODS.index), _d=df.dataset.map(DATASETS.index))


def analyze(F, C, T, boot, seed, res_dir, notes_dir):
    """aggregation, bootstrap CIs, verdict, tables (no data loading; testable)."""
    RES, HERE_ = res_dir, notes_dir
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    F = key(F).sort_values(["_m", "_d", "fold"]).drop(columns=["_m", "_d"])
    C = key(C).sort_values(["_m", "_d", "fold"]).drop(columns=["_m", "_d"])
    T = key(T).sort_values(["_m", "_d", "fold", "epoch"]).drop(columns=["_m", "_d"])
    main_cols = ["method", "dataset", "fold", "criterion", "auroc", "auprc", "fpr95", "epoch_selected"]
    F[main_cols + [c for c in F.columns if c not in main_cols]].to_csv(osp.join(RES, "m1_selection_folds.csv"), index=False)
    C.to_csv(osp.join(RES, "m1_corr.csv"), index=False)
    T.to_csv(osp.join(RES, "m1_trajectories.csv"), index=False)
    coverage = F[F.criterion == "oracle"].groupby(["method", "dataset"]).size().unstack(fill_value=0)

    # ---------------- per-dataset means
    BD = F.groupby(["method", "dataset", "criterion"]).agg(n_folds=("auroc", "size"), auroc=("auroc", "mean"),
                                                           auroc_std=("auroc", lambda s: s.std(ddof=0)),
                                                           auprc=("auprc", "mean"), fpr95=("fpr95", "mean"),
                                                           epoch_selected=("epoch_selected", "mean")).reset_index()
    BD.to_csv(osp.join(RES, "m1_selection_by_dataset.csv"), index=False)

    # ---------------- summary with paired bootstrap CIs
    rng = np.random.default_rng(seed)
    allc = CRITERIA + DIAG
    means = {c: method_means(F, c) for c in allc}
    means_pr = {c: method_means(F, c, "auprc") for c in allc}
    means_fp = {c: method_means(F, c, "fpr95") for c in allc}
    VN = cube(F, "valnormal")
    OR = cube(F, "oracle")
    gap = OR - VN
    summ = []
    boot_cache = {}
    for c in allc:
        D = cube(F, c) - VN
        pm, avg = stat_from_cube(D)
        fb_m, fb = fold_bootstrap(D, boot, rng)
        db_m, db = dataset_bootstrap(D, boot, rng)
        boot_cache[c] = (fb_m, fb, db_m, db)
        for k, m in enumerate(METHODS + ["ALL4"]):
            if m == "ALL4":
                mean_au = float(np.mean([means[c].get(x, np.nan) for x in METHODS]))
                mean_pr = float(np.mean([means_pr[c].get(x, np.nan) for x in METHODS]))
                mean_fp = float(np.mean([means_fp[c].get(x, np.nan) for x in METHODS]))
                g, gf, gd = avg, fb, db
                gap_m = float(np.mean([stat_from_cube(gap)[0]]))
                ndat = int(np.sum(np.any(np.isfinite(cube(F, c)), axis=2)))
                wins = int(np.sum(np.nanmean(D, 2) > 0))
            else:
                mean_au, mean_pr, mean_fp = float(means[c].get(m, np.nan)), float(means_pr[c].get(m, np.nan)), float(means_fp[c].get(m, np.nan))
                g, gf, gd = pm[k], fb_m[:, k], db_m[:, k]
                gap_m = stat_from_cube(gap)[0][k]
                ndat = int(np.sum(np.any(np.isfinite(cube(F, c)[k]), axis=1)))
                wins = int(np.sum(np.nanmean(D[k], 1) > 0))
            lo, hi = ci(gf)
            dlo, dhi = ci(gd)
            summ.append(dict(method=m, criterion=c, auroc=mean_au, auprc=mean_pr, fpr95=mean_fp, gain_vs_valnormal=g,
                             gain_ci_fold_lo=lo, gain_ci_fold_hi=hi, gain_ci_dataset_lo=dlo, gain_ci_dataset_hi=dhi,
                             oracle_minus_valnormal=gap_m,
                             recovered_frac_of_oracle_gap=(g / gap_m if gap_m and np.isfinite(gap_m) else np.nan),
                             n_datasets=ndat, datasets_better_than_valnormal=wins))
    S = pd.DataFrame(summ)
    # PSEUDO vs the better literature-style baseline, and vs each baseline
    all_mean = {c: float(np.mean([means[c].get(x, np.nan) for x in METHODS])) for c in allc}
    better = "ckpt_ensemble" if all_mean["ckpt_ensemble"] >= all_mean["consensus"] else "consensus"
    extra = []
    for ref in ["ckpt_ensemble", "consensus", "last"]:
        D = cube(F, "pseudo") - cube(F, ref)
        pm, avg = stat_from_cube(D)
        fb_m, fb = fold_bootstrap(D, boot, rng)
        db_m, db = dataset_bootstrap(D, boot, rng)
        for k, m in enumerate(METHODS + ["ALL4"]):
            g = avg if m == "ALL4" else pm[k]
            gf = fb if m == "ALL4" else fb_m[:, k]
            gd = db if m == "ALL4" else db_m[:, k]
            extra.append(dict(method=m, criterion=f"pseudo_minus_{ref}", auroc=np.nan, gain_vs_valnormal=np.nan,
                              diff=g, diff_ci_fold_lo=ci(gf)[0], diff_ci_fold_hi=ci(gf)[1],
                              diff_ci_dataset_lo=ci(gd)[0], diff_ci_dataset_hi=ci(gd)[1]))
    S = pd.concat([S, pd.DataFrame(extra)], ignore_index=True)
    S.to_csv(osp.join(RES, "m1_selection_summary.csv"), index=False)

    # ---------------- verdict (pre-registered rule, NOTES.md)
    ps = S[(S.criterion == "pseudo")].set_index("method")
    gain_avg = float(ps.loc["ALL4", "gain_vs_valnormal"])
    n_improved = int(sum(ps.loc[m, "gain_vs_valnormal"] > 0 for m in METHODS))
    vs_better = all_mean["pseudo"] - all_mean[better]
    beats_both = all_mean["pseudo"] > all_mean["ckpt_ensemble"] and all_mean["pseudo"] > all_mean["consensus"]
    if gain_avg >= 2.0 and n_improved >= 3 and vs_better >= 1.0:
        verdict = "PASS"
    elif gain_avg < 1.0 or not beats_both:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    V = dict(verdict=verdict, gain_pseudo_vs_valnormal_avg4=gain_avg,
             gain_ci_fold=[float(ps.loc["ALL4", "gain_ci_fold_lo"]), float(ps.loc["ALL4", "gain_ci_fold_hi"])],
             gain_ci_dataset=[float(ps.loc["ALL4", "gain_ci_dataset_lo"]), float(ps.loc["ALL4", "gain_ci_dataset_hi"])],
             methods_improved=n_improved, per_method_gain={m: float(ps.loc[m, "gain_vs_valnormal"]) for m in METHODS},
             better_literature_baseline=better, pseudo_minus_better_baseline=vs_better,
             pseudo_beats_both_baselines=bool(beats_both),
             pass_not_robust=bool(verdict == "PASS" and float(ps.loc["ALL4", "gain_ci_fold_lo"]) <= 0),
             mean_auroc_avg4={c: all_mean[c] for c in allc},
             coverage={m: {d: int(v) for d, v in r.items()} for m, r in coverage.T.to_dict().items()},
             n_runs=int(len(F[F.criterion == "oracle"])), boot=boot)
    json.dump(V, open(osp.join(HERE_, "verdict.json"), "w"), indent=1, default=float)

    # ---------------- reproduction check against deep_rerun
    DR = pd.read_csv(osp.join(RES0, "deep_rerun_folds.csv"))
    DR = DR[DR.dataset.isin(DATASETS)]
    rep = []
    for proto in ["oracle", "last", "valnormal"]:
        a1 = F[F.criterion == proto][["method", "dataset", "fold", "auroc", "epoch_selected"]]
        a0 = DR[DR.protocol == proto][["method", "dataset", "fold", "auroc", "epoch_selected"]].copy()
        a0["auroc"] *= 100
        J = a1.merge(a0, on=["method", "dataset", "fold"], suffixes=("_m1", "_dr"))
        for m in METHODS + ["ALL4"]:
            j = J if m == "ALL4" else J[J.method == m]
            if not len(j):
                continue
            if m == "ALL4":
                m1v = np.mean([j[j.method == x].groupby("dataset").auroc_m1.mean().mean() for x in METHODS if (j.method == x).any()])
                drv = np.mean([j[j.method == x].groupby("dataset").auroc_dr.mean().mean() for x in METHODS if (j.method == x).any()])
            else:
                m1v = j.groupby("dataset").auroc_m1.mean().mean()
                drv = j.groupby("dataset").auroc_dr.mean().mean()
            rep.append(dict(method=m, protocol=proto, n_runs=len(j), m1_mean=m1v, deep_rerun_mean=drv, diff=m1v - drv,
                            mean_abs_fold_diff=float(np.mean(np.abs(j.auroc_m1 - j.auroc_dr))),
                            max_abs_fold_diff=float(np.max(np.abs(j.auroc_m1 - j.auroc_dr))),
                            same_epoch_frac=float(np.mean(j.epoch_selected_m1 == j.epoch_selected_dr))))
    REP = pd.DataFrame(rep)
    REP.to_csv(osp.join(RES, "m1_repro_vs_deep_rerun.csv"), index=False)

    # ---------------- per-family diagnostics
    fam_rows = []
    for (m, ds), g in C.groupby(["method", "dataset"]):
        r = dict(method=m, dataset=ds, spearman_pauc_mean=g.spearman_pauc_mean_vs_auroc.median(),
                 spearman_neg_holdout=g.spearman_neg_holdout_vs_auroc.median())
        for f in FAMILIES:
            col = f"spearman_pauc_{f}_vs_auroc"
            if col in g and g[col].notna().any():
                r[f"spearman_{f}"] = g[col].median()
                r[f"pauc_{f}_median"] = g[f"pauc_{f}_median"].median()
                fsel = F[(F.method == m) & (F.dataset == ds) & (F.criterion == f"pseudo_{f}")].auroc.mean()
                vn = F[(F.method == m) & (F.dataset == ds) & (F.criterion == "valnormal")].auroc.mean()
                r[f"gain_pseudo_{f}_vs_valnormal"] = fsel - vn
        fam_rows.append(r)
    FD = pd.DataFrame(fam_rows)
    FD = key(FD).sort_values(["_m", "_d"]).drop(columns=["_m", "_d"])
    FD.to_csv(osp.join(RES, "m1_family_diag.csv"), index=False)

    write_tables(F, S, C, FD, REP, BD, V, coverage, all_mean, HERE_)
    print(json.dumps(V, indent=1, default=float))
    return V


def md(df, floatfmt=".2f", index=False):
    """minimal markdown table (no tabulate dependency)."""
    if index:
        df = df.reset_index()
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "–" if not np.isfinite(v) else format(v, floatfmt)
        return str(v)
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(cell(v) for v in r) + " |")
    return "\n".join(out)


def fmt(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def write_tables(F, S, C, FD, REP, BD, V, coverage, all_mean, notes_dir):
    L = []
    L.append(f"Generated by select.py; {V['n_runs']} runs; bootstrap B = {V['boot']}. AUROC in %.\n")
    L.append("### Coverage (folds per method x dataset)\n")
    L.append(md(coverage, index=True) + "\n")
    L.append("### Mean AUROC per criterion (mean over datasets of 5-fold means)\n")
    L.append("| criterion | " + " | ".join(METHODS) + " | 4-method avg | gain vs VAL-NORMAL (avg4) [95% fold-boot CI] | dataset-boot CI | recovered frac of ORACLE−VALN gap |")
    L.append("|---" * (len(METHODS) + 5) + "|")
    for c in CRITERIA + DIAG:
        s = S[S.criterion == c].set_index("method")
        L.append(f"| {c} | " + " | ".join(fmt(s.loc[m, 'auroc']) for m in METHODS) + f" | {fmt(s.loc['ALL4', 'auroc'])} | "
                 f"{fmt(s.loc['ALL4', 'gain_vs_valnormal'], sign=True)} [{fmt(s.loc['ALL4', 'gain_ci_fold_lo'], sign=True)}, {fmt(s.loc['ALL4', 'gain_ci_fold_hi'], sign=True)}] | "
                 f"[{fmt(s.loc['ALL4', 'gain_ci_dataset_lo'], sign=True)}, {fmt(s.loc['ALL4', 'gain_ci_dataset_hi'], sign=True)}] | {fmt(s.loc['ALL4', 'recovered_frac_of_oracle_gap'])} |")
    L.append("\n### Per-method gain over VAL-NORMAL [95% paired fold-bootstrap CI] (datasets where better / covered)\n")
    L.append("| criterion | " + " | ".join(METHODS) + " |")
    L.append("|---" * (len(METHODS) + 1) + "|")
    for c in [x for x in CRITERIA + DIAG if x != "valnormal"]:
        s = S[S.criterion == c].set_index("method")
        L.append(f"| {c} | " + " | ".join(
            f"{fmt(s.loc[m, 'gain_vs_valnormal'], sign=True)} [{fmt(s.loc[m, 'gain_ci_fold_lo'], sign=True)}, {fmt(s.loc[m, 'gain_ci_fold_hi'], sign=True)}] ({int(s.loc[m, 'datasets_better_than_valnormal'])}/{int(s.loc[m, 'n_datasets'])})"
            for m in METHODS) + " |")
    L.append("\n### PSEUDO minus literature-style baselines [95% fold-bootstrap CI; dataset-bootstrap CI]\n")
    L.append("| comparison | " + " | ".join(METHODS) + " | 4-method avg |")
    L.append("|---" * (len(METHODS) + 2) + "|")
    for ref in ["ckpt_ensemble", "consensus", "last"]:
        s = S[S.criterion == f"pseudo_minus_{ref}"].set_index("method")
        L.append(f"| PSEUDO − {ref} | " + " | ".join(
            f"{fmt(s.loc[m, 'diff'], sign=True)} [{fmt(s.loc[m, 'diff_ci_fold_lo'], sign=True)}, {fmt(s.loc[m, 'diff_ci_fold_hi'], sign=True)}]" for m in METHODS)
            + f" | {fmt(s.loc['ALL4', 'diff'], sign=True)} [{fmt(s.loc['ALL4', 'diff_ci_fold_lo'], sign=True)}, {fmt(s.loc['ALL4', 'diff_ci_fold_hi'], sign=True)}]; "
              f"[{fmt(s.loc['ALL4', 'diff_ci_dataset_lo'], sign=True)}, {fmt(s.loc['ALL4', 'diff_ci_dataset_hi'], sign=True)}] |")
    L.append("\n### Per-dataset mean AUROC (5 folds)\n")
    for m in METHODS:
        L.append(f"#### {m}\n")
        L.append("| dataset | " + " | ".join(CRITERIA) + " | pseudo − valnormal |")
        L.append("|---" * (len(CRITERIA) + 2) + "|")
        b = BD[BD.method == m].set_index(["dataset", "criterion"])
        for ds in DATASETS:
            if (ds, "oracle") not in b.index:
                L.append(f"| {ds} | " + " | ".join("–" for _ in CRITERIA) + " | – |")
                continue
            L.append(f"| {ds} | " + " | ".join(fmt(b.loc[(ds, c), 'auroc']) for c in CRITERIA)
                     + f" | {fmt(b.loc[(ds, 'pseudo'), 'auroc'] - b.loc[(ds, 'valnormal'), 'auroc'], sign=True)} |")
        L.append("")
    L.append("### Spearman (across eval points) between label-free signals and test AUROC: median over runs [IQR]\n")
    L.append("| method | pseudo mean | " + " | ".join(FAMILIES) + " | −held-out loss | consensus rho | AUCp | frac runs pseudo-mean rho > 0 |")
    L.append("|---" * (len(FAMILIES) + 6) + "|")
    for m in METHODS + ["ALL4"]:
        g = C if m == "ALL4" else C[C.method == m]
        def q(col):
            if col not in g or g[col].notna().sum() == 0:
                return "–"
            v = g[col].dropna()
            return f"{v.median():+.2f} [{v.quantile(.25):+.2f}, {v.quantile(.75):+.2f}]"
        L.append(f"| {m} | {q('spearman_pauc_mean_vs_auroc')} | " + " | ".join(q(f'spearman_pauc_{f}_vs_auroc') for f in FAMILIES)
                 + f" | {q('spearman_neg_holdout_vs_auroc')} | {q('spearman_consensus_rho_vs_auroc')} | {q('spearman_aucp_vs_auroc')} | "
                 f"{fmt(float((g.spearman_pauc_mean_vs_auroc > 0).mean()))} |")
    L.append("\n### Per-family diagnostics per method x dataset (median Spearman of family pseudo-AUROC vs test AUROC; gain of selecting by that family alone vs VAL-NORMAL)\n")
    cols = ["method", "dataset", "spearman_pauc_mean", "spearman_neg_holdout"] + [c for f in FAMILIES for c in (f"spearman_{f}", f"pauc_{f}_median", f"gain_pseudo_{f}_vs_valnormal")]
    cols = [c for c in cols if c in FD]
    L.append(md(FD[cols], floatfmt="+.2f") + "\n")
    L.append("### Reproduction of deep_rerun (same protocol definitions, new runs)\n")
    L.append(md(REP) + "\n")
    open(osp.join(notes_dir, "tables.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
