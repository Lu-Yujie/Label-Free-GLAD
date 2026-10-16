"""Statistics, leaderboards and the automatic PREREG evaluation for the label-free GLAD re-evaluation.

  PYTHONDONTWRITEBYTECODE=1 /N/slate/yujlu/envs/grail-prelim/bin/python -B glad_stats.py [--final] [--B 4000]

Inputs (prelim/results/; missing files are skipped and reported):
  deep_rerun_folds.csv (+ _extra)  SIGNET/OCGTL/OCGIN/GLocalKD, oracle/last/valnormal, UB-GOLD per-dataset HP
  m1_selection_folds.csv           same 4 methods re-run on CPU with 6 label-free selectors (+ AUCp)
  eval_folds.csv                   12 deep detectors (week-1 harness): oracle/last/valnormal/hpfree_*
  eval_kernels_folds.csv           KernelGLAD WL/PK x IF/OCSVM, protocol tuned/default
  grail0_folds.csv                 GRAIL-0 and the a-priori controls (size-only, graphlet counts, WL+iForest)
  m2_typed_summary.csv, eval_typed*.csv   typed synthetic suite (for the post-hoc G2(c))

Outputs: prelim/results/eval_leaderboard_*.csv, eval_leaderboard_prereg.json and
prelim/evalpaper/stats/REPORT.md (auto-generated; do not edit by hand).

Conventions
  * AUROC/AUPRC/FPR95 in percent. A leaderboard entry = mean over the datasets of a set of the 5-fold
    means. Only methods with all datasets x 5 folds of a set ("complete") are ranked and bootstrapped.
  * Paired bootstraps (B replicates, fixed seed): the same resampled folds (fold bootstrap: folds
    resampled with replacement within every dataset) or the same resampled datasets (dataset
    bootstrap) are applied to every method and protocol, so differences, ranks and Kendall tau are
    computed on matched replicates. Folds share training data, so fold-bootstrap CIs are optimistic.
  * Protocols: oracle, last, valnormal (UB-GOLD per-dataset HP, which UB-GOLD tuned on test) and
    hpfree_oracle, hpfree_last, hpfree_valnormal (label-free HP, PREREG_AMENDMENTS A1).
    Non-iterative methods have no checkpoint: KernelGLAD enters oracle/last/valnormal with its
    test-tuned HP ('tuned') and hpfree_* with UB-GOLD defaults ('default'); GRAIL-0 and the
    controls are a-priori (no HP, no checkpoint) and enter every protocol with the same value.
    Each leaderboard therefore compares methods that had the same access to test labels.
  * PREREG (prelim/evalpaper/PREREG.md) is evaluated exactly; G1/G2 use deep methods only.
    Status is PASS/FAIL once all 12 deep detectors are complete for the criterion, or with --final
    (decision date 2026-11-08: whatever exists then is final, missing methods count as missing);
    before that it is PENDING with the current value and whether it is currently met.
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

HERE = osp.dirname(osp.abspath(__file__))
PRELIM = osp.dirname(osp.dirname(HERE))
RES = osp.join(PRELIM, "results")
REPORT = osp.join(HERE, "REPORT.md")

UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
SECONDARY = ["DD", "COLLAB", "Tox21_p53", "Tox21_MMP"]
SETS = {"UB9": UB9, "secondary": SECONDARY}
PROTOCOLS = ["oracle", "last", "valnormal", "hpfree_oracle", "hpfree_last", "hpfree_valnormal"]
DEEP12 = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD", "GLADC", "CVTGAD", "GOOD-D", "GraphDE",
          "GraphCL_IF", "GraphCL_OCSVM", "InfoGraph_IF", "InfoGraph_OCSVM"]
OLD4 = DEEP12[:4]
NFOLD = 5
REF_MAP = {"grail0_sumz": "GRAIL-0", "ctrl_size_lw": "size-only", "ctrl_gc_lw": "graphlet-count+LW",
           "ctrl_gc_iforest": "graphlet-count+iForest", "ctrl_wl_iforest": "WL-hist+iForest (ctrl)"}
KERNEL_PROTO = {"tuned": ["oracle", "last", "valnormal"], "default": ["hpfree_oracle", "hpfree_last", "hpfree_valnormal"]}
TYPED_TYPES = ["T1a", "T3a", "T3b"]
DS_ALIASES = {"PROTEINS": "PROTEINS_full", "IMDB-B": "IMDB-BINARY", "REDDIT-B": "REDDIT-BINARY",
              "p53": "Tox21_p53", "MMP": "Tox21_MMP"}


def canon_method(m):
    key = str(m).strip().replace("-", "_").lower()
    for d in DEEP12:
        if d.replace("-", "_").lower() == key:
            return d
    return str(m).strip()


def pct(s):
    s = pd.to_numeric(s, errors="coerce")
    return s * 100.0 if s.dropna().abs().max() <= 1.0 + 1e-9 else s


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------
def load_all(args):
    notes, frames = [], []
    cols = ["method", "family", "dataset", "fold", "protocol", "auroc", "auprc", "fpr95", "source"]

    def std(df, family, source):
        df = df.copy()
        df["method"] = df["method"].map(canon_method)
        df["dataset"] = df["dataset"].map(lambda d: DS_ALIASES.get(d, d))
        df["fold"] = pd.to_numeric(df["fold"]).astype(int)
        for c in ("auroc", "auprc", "fpr95"):
            df[c] = pct(df[c]) if c in df else np.nan
        df["family"] = family
        df["source"] = source
        return df[cols]

    # 1. deep_rerun (canonical for the old 4, UB-GOLD HP, oracle/last/valnormal)
    p = osp.join(RES, "deep_rerun_folds.csv")
    dr = None
    if osp.exists(p):
        dr = std(pd.read_csv(p), "deep", "deep_rerun")
        dr = dr[dr.dataset != "MUTAG"]
        notes.append(f"deep_rerun_folds.csv: {len(dr)} rows, methods {sorted(dr.method.unique())}")
        px = osp.join(RES, "deep_rerun_folds_extra.csv")
        if osp.exists(px):
            ex = std(pd.read_csv(px), "deep", "deep_rerun")
            ex = ex[ex.dataset != "MUTAG"]
            ex["protocol"] = "x_" + ex["protocol"]
            dr = pd.concat([dr, ex])
    # 2. eval_folds (week-1 harness)
    p = osp.join(RES, "eval_folds.csv")
    ev = None
    if osp.exists(p):
        raw = pd.read_csv(p)
        raw["method"] = raw["method"].map(canon_method)
        if "orientation" in raw:
            keep = []
            for m, g in raw.groupby("method"):
                ors = set(g["orientation"].fillna("").astype(str))
                if len(ors) > 1 and "ubgold" in ors:  # PREREG: UB-GOLD orientation = reproduction column only
                    notes.append(f"eval_folds.csv: {m}: orientation 'ubgold' rows kept as reproduction only "
                                 f"(main: {sorted(ors - {'ubgold'})})")
                    g = g[g["orientation"].astype(str) != "ubgold"]
                keep.append(g)
            raw = pd.concat(keep) if keep else raw
        ev = std(raw, "deep", "eval_folds")
        ev.loc[~ev.method.isin(DEEP12), "family"] = "other"
        notes.append(f"eval_folds.csv: {len(ev)} rows, methods {sorted(ev.method.unique())}, "
                     f"protocols {sorted(ev.protocol.unique())}")
    else:
        notes.append("eval_folds.csv: not present yet (8 additional detectors and all HP-FREE runs PENDING)")
    # merge deep sources: checkpoint protocols of one (method, dataset, fold) from ONE source
    deep = []
    if dr is not None:
        deep.append(dr)
    if ev is not None:
        ck = ["oracle", "last", "valnormal"]
        prefer_dr = args.prefer != "eval_folds"
        if dr is not None and prefer_dr:
            have = set(map(tuple, dr[dr.protocol.isin(ck)][["method", "dataset", "fold"]].drop_duplicates().values))
            mask = ev.protocol.isin(ck) & ev[["method", "dataset", "fold"]].apply(tuple, axis=1).isin(have)
            if mask.any():
                notes.append(f"eval_folds.csv: {int(mask.sum())} oracle/last/valnormal rows of cells already in "
                             "deep_rerun ignored (deep_rerun is canonical for those cells)")
            ev = ev[~mask]
        elif dr is not None:
            have = set(map(tuple, ev[ev.protocol.isin(ck)][["method", "dataset", "fold"]].drop_duplicates().values))
            dr = dr[~(dr.protocol.isin(ck) & dr[["method", "dataset", "fold"]].apply(tuple, axis=1).isin(have))]
            deep = [dr]
        deep.append(ev)
    if deep:
        frames.append(pd.concat(deep))
    # 3. kernels
    p = osp.join(RES, "eval_kernels_folds.csv")
    if osp.exists(p):
        kr = pd.read_csv(p)
        kr = kr[~kr["method"].astype(str).str.contains("@")]  # @testfit = UB-GOLD transductive path, not ranked
        rows = []
        for proto, targets in KERNEL_PROTO.items():
            sub = kr[kr.protocol == proto]
            for t in targets:
                s = sub.copy()
                s["protocol"] = t
                rows.append(s)
        if rows:
            frames.append(std(pd.concat(rows), "kernel", "eval_kernels"))
        notes.append(f"eval_kernels_folds.csv: {len(kr)} train-fit rows, methods {sorted(kr.method.unique())}")
    else:
        notes.append("eval_kernels_folds.csv: not present")
    # 4. a-priori references
    p = osp.join(RES, "grail0_folds.csv")
    if osp.exists(p):
        g0 = pd.read_csv(p)
        g0 = g0[g0.method.isin(REF_MAP)].copy()
        g0["method"] = g0["method"].map(REF_MAP)
        rows = []
        for t in PROTOCOLS:
            s = g0.copy()
            s["protocol"] = t
            rows.append(s)
        frames.append(std(pd.concat(rows), "reference", "grail0"))
        notes.append(f"grail0_folds.csv: references {sorted(g0.method.unique())}")
    long = pd.concat(frames, ignore_index=True)
    dup = long.duplicated(["method", "dataset", "fold", "protocol"], keep=False)
    if dup.any():
        bad = long[dup][["method", "dataset", "protocol"]].drop_duplicates().head(10).values.tolist()
        raise SystemExit(f"duplicate (method, dataset, fold, protocol) rows: {bad}")
    # 5. M1 selectors (separate table; own runs)
    m1 = None
    p = osp.join(RES, "m1_selection_folds.csv")
    if osp.exists(p):
        m1 = pd.read_csv(p).rename(columns={"criterion": "protocol"})
        m1 = std(m1, "deep", "m1")
        notes.append(f"m1_selection_folds.csv: {len(m1)} rows, criteria {sorted(m1.protocol.unique())}")
    return long, m1, notes


# ----------------------------------------------------------------------------------------------
# array helpers and bootstrap
# ----------------------------------------------------------------------------------------------
def cube(long, methods, datasets, protocol, metric="auroc"):
    """(M, D, 5) array; NaN where missing."""
    A = np.full((len(methods), len(datasets), NFOLD), np.nan)
    sub = long[(long.protocol == protocol) & long.method.isin(methods) & long.dataset.isin(datasets)]
    mi = {m: i for i, m in enumerate(methods)}
    di = {d: i for i, d in enumerate(datasets)}
    for m, d, f, v in sub[["method", "dataset", "fold", metric]].itertuples(index=False):
        if 0 <= f < NFOLD:
            A[mi[m], di[d], f] = v
    return A


class Boot:
    def __init__(self, n_ds, B, seed):
        rng = np.random.default_rng(seed)
        self.B = B
        self.fold_idx = rng.integers(0, NFOLD, size=(B, n_ds, NFOLD))
        self.ds_idx = rng.integers(0, n_ds, size=(B, n_ds))
        self.D = n_ds

    def fold(self, A):
        """A: (M, D, 5) -> (M, B) set means under the fold bootstrap."""
        g = A[:, np.arange(self.D)[None, :, None], self.fold_idx]  # (M, B, D, 5)
        return g.mean(-1).mean(-1)

    def ds(self, A):
        dm = A.mean(-1)  # (M, D)
        return dm[:, self.ds_idx].mean(-1)  # (M, B)


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def tau_b(x, y):
    """Kendall tau-b along the last axis (x, y: (..., M))."""
    i, j = np.triu_indices(x.shape[-1], 1)
    sx = np.sign(x[..., i] - x[..., j])
    sy = np.sign(y[..., i] - y[..., j])
    num = (sx * sy).sum(-1)
    den = np.sqrt((sx != 0).sum(-1) * (sy != 0).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def top_k(means, names, k=3):
    order = np.argsort(-np.asarray(means), kind="stable")
    return [names[i] for i in order[:k]]


def coverage(long, method, datasets, protocol):
    sub = long[(long.method == method) & (long.protocol == protocol) & long.dataset.isin(datasets)]
    cnt = sub.groupby("dataset").fold.nunique()
    return {d: int(cnt.get(d, 0)) for d in datasets}


def complete(long, method, datasets, protocols):
    return all(all(v == NFOLD for v in coverage(long, method, datasets, p).values()) for p in protocols)


# ----------------------------------------------------------------------------------------------
# analyses
# ----------------------------------------------------------------------------------------------
def leaderboards(long, B, seed):
    rows, perds = [], []
    fam = long.drop_duplicates("method").set_index("method")["family"].to_dict()
    for sname, dss in SETS.items():
        boot = Boot(len(dss), B, seed)
        for proto in PROTOCOLS:
            methods = sorted(long[(long.protocol == proto) & long.dataset.isin(dss)].method.unique())
            if not methods:
                continue
            cm = [m for m in methods if complete(long, m, dss, [proto])]
            A = {k: cube(long, methods, dss, proto, k) for k in ("auroc", "auprc", "fpr95")}
            idx = {m: i for i, m in enumerate(methods)}
            if cm:
                Ac = A["auroc"][[idx[m] for m in cm]]
                fb, db = boot.fold(Ac), boot.ds(Ac)
                means_c = Ac.mean(-1).mean(-1)
                rank = stats.rankdata(-means_c, method="min")
            for m in methods:
                i = idx[m]
                cov = coverage(long, m, dss, proto)
                dm = np.nanmean(A["auroc"][i], -1)
                r = {"set": sname, "protocol": proto, "method": m, "family": fam.get(m, ""),
                     "n_datasets": int(sum(v > 0 for v in cov.values())), "n_datasets_total": len(dss),
                     "complete": m in cm, "mean_auroc": float(np.nanmean(dm)),
                     "mean_auprc": float(np.nanmean(np.nanmean(A["auprc"][i], -1))),
                     "mean_fpr95": float(np.nanmean(np.nanmean(A["fpr95"][i], -1)))}
                if m in cm:
                    j = cm.index(m)
                    r.update(rank=int(rank[j]), fold_ci_lo=ci(fb[j])[0], fold_ci_hi=ci(fb[j])[1],
                             ds_ci_lo=ci(db[j])[0], ds_ci_hi=ci(db[j])[1])
                rows.append(r)
                for k, d in enumerate(dss):
                    v = A["auroc"][i, k]
                    if np.isfinite(v).any():
                        perds.append({"set": sname, "protocol": proto, "method": m, "family": fam.get(m, ""),
                                      "dataset": d, "n_folds": int(np.isfinite(v).sum()),
                                      "auroc_mean": float(np.nanmean(v)), "auroc_std_ddof0": float(np.nanstd(v)),
                                      "auprc_mean": float(np.nanmean(A["auprc"][i, k])),
                                      "fpr95_mean": float(np.nanmean(A["fpr95"][i, k]))})
    return pd.DataFrame(rows), pd.DataFrame(perds)


INFL = [("oracle", "valnormal", "checkpoint: ORACLE - VAL-NORMAL"),
        ("oracle", "last", "checkpoint: ORACLE - LAST"),
        ("valnormal", "hpfree_valnormal", "HP leak: VAL-NORMAL - HP-FREE"),
        ("oracle", "hpfree_valnormal", "total: ORACLE - HP-FREE"),
        ("oracle", "x_oracle_nostop", "early stop: ORACLE - max-over-run"),
        ]


def inflation(long, B, seed, kernels_raw):
    rows = []
    fam = long.drop_duplicates("method").set_index("method")["family"].to_dict()
    for sname, dss in SETS.items():
        boot = Boot(len(dss), B, seed)
        for m in sorted(long.method.unique()):
            if fam.get(m) not in ("deep",):
                continue
            for a, b, label in INFL:
                if not complete(long, m, dss, [a, b]):
                    continue
                D = cube(long, [m], dss, a) - cube(long, [m], dss, b)
                dm = D[0].mean(-1)
                fb, db = boot.fold(D)[0], boot.ds(D)[0]
                rows.append({"set": sname, "method": m, "family": "deep", "comparison": label, "a": a, "b": b,
                             "mean_diff": float(dm.mean()), "fold_ci_lo": ci(fb)[0], "fold_ci_hi": ci(fb)[1],
                             "ds_ci_lo": ci(db)[0], "ds_ci_hi": ci(db)[1], "n_datasets": len(dss),
                             "n_datasets_diff_ge2": int((dm >= 2.0).sum()), "max_dataset_diff": float(dm.max()),
                             "max_fold_diff": float(np.nanmax(D))})
        # kernels: test-tuned HP vs UB-GOLD default (the only test-label use a kernel detector can have)
        if kernels_raw is not None:
            kr = kernels_raw[~kernels_raw["method"].astype(str).str.contains("@")].copy()
            kr["auroc"] = pct(kr["auroc"])
            for m in sorted(kr.method.unique()):
                k2 = kr[kr.method == m]
                A = {}
                for proto in ("tuned", "default"):
                    s = k2[k2.protocol == proto]
                    X = np.full((1, len(dss), NFOLD), np.nan)
                    for d, f, v in s[["dataset", "fold", "auroc"]].itertuples(index=False):
                        if d in dss:
                            X[0, dss.index(d), int(f)] = v
                    A[proto] = X
                D = A["tuned"] - A["default"]
                if not np.isfinite(D).all():
                    continue
                dm = D[0].mean(-1)
                fb, db = boot.fold(D)[0], boot.ds(D)[0]
                rows.append({"set": sname, "method": m, "family": "kernel", "comparison": "HP leak: tuned - default",
                             "a": "tuned", "b": "default", "mean_diff": float(dm.mean()),
                             "fold_ci_lo": ci(fb)[0], "fold_ci_hi": ci(fb)[1], "ds_ci_lo": ci(db)[0],
                             "ds_ci_hi": ci(db)[1], "n_datasets": len(dss), "n_datasets_diff_ge2": int((dm >= 2.0).sum()),
                             "max_dataset_diff": float(dm.max()), "max_fold_diff": float(np.nanmax(D))})
    return pd.DataFrame(rows)


PAIRS = [("oracle", "valnormal"), ("oracle", "last"), ("valnormal", "last"), ("oracle", "hpfree_valnormal"),
         ("valnormal", "hpfree_valnormal"), ("hpfree_oracle", "hpfree_valnormal")]


def pools(long, dss, protos):
    fam = long.drop_duplicates("method").set_index("method")["family"].to_dict()
    ms = sorted(m for m in long.method.unique() if complete(long, m, dss, protos))
    return {"deep": [m for m in ms if fam.get(m) == "deep"], "all": ms}


def kendall(long, B, seed):
    rows = []
    for sname, dss in SETS.items():
        boot = Boot(len(dss), B, seed)
        for a, b in PAIRS:
            for pool, ms in pools(long, dss, [a, b]).items():
                if len(ms) < 3:
                    continue
                Aa, Ab = cube(long, ms, dss, a), cube(long, ms, dss, b)
                if np.allclose(Aa, Ab, equal_nan=True):
                    continue  # identical by construction (only protocol-invariant methods)
                ma, mb = Aa.mean(-1).mean(-1), Ab.mean(-1).mean(-1)
                t = float(tau_b(ma, mb))
                fa, fbb = boot.fold(Aa).T, boot.fold(Ab).T  # (B, M)
                da, dbb = boot.ds(Aa).T, boot.ds(Ab).T
                tf, td = tau_b(fa, fbb), tau_b(da, dbb)
                ta, tb = top_k(ma, ms), top_k(mb, ms)
                ch = set(ta) != set(tb)
                k = min(3, len(ms))
                chf = np.mean([set(np.argsort(-fa[i])[:k]) != set(np.argsort(-fbb[i])[:k]) for i in range(len(fa))])
                chd = np.mean([set(np.argsort(-da[i])[:k]) != set(np.argsort(-dbb[i])[:k]) for i in range(len(da))])
                rows.append({"set": sname, "pool": pool, "protocol_a": a, "protocol_b": b, "n_methods": len(ms),
                             "tau_b": t, "fold_ci_lo": ci(tf)[0], "fold_ci_hi": ci(tf)[1],
                             "ds_ci_lo": float(np.nanpercentile(td, 2.5)), "ds_ci_hi": float(np.nanpercentile(td, 97.5)),
                             "top3_a": "|".join(ta), "top3_b": "|".join(tb), "top3_changed": ch,
                             "p_top3_changed_foldboot": float(chf), "p_top3_changed_dsboot": float(chd),
                             "methods": "|".join(ms)})
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def cliques(names, ranks, cd):
    """Maximal groups of rank-adjacent methods whose average ranks differ by <= CD (CD-diagram bars)."""
    order = np.argsort(ranks, kind="stable")
    r = np.asarray(ranks)[order]
    groups = []
    for i in range(len(r)):
        j = i
        while j + 1 < len(r) and r[j + 1] - r[i] <= cd + 1e-12:
            j += 1
        if j > i and not any(g[0] <= i and j <= g[1] for g in groups):
            groups.append((i, j))
    return [[names[order[k]] for k in range(i, j + 1)] for i, j in groups]


def friedman(long, alpha=0.05):
    frows, prows, crows = [], [], []
    for sname, dss in SETS.items():
        for proto in PROTOCOLS:
            for pool, ms in pools(long, dss, [proto]).items():
                k, N = len(ms), len(dss)
                if k < 3:
                    continue
                DM = cube(long, ms, dss, proto).mean(-1).T  # (N, k)
                R = np.vstack([stats.rankdata(-row) for row in DM])
                avg = R.mean(0)
                chi2, p = stats.friedmanchisquare(*[DM[:, j] for j in range(k)])
                ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) - chi2 > 0 else np.inf
                pff = float(stats.f.sf(ff, k - 1, (k - 1) * (N - 1))) if np.isfinite(ff) else 0.0
                q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
                cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
                frows.append({"set": sname, "pool": pool, "protocol": proto, "n_methods": k, "n_datasets": N,
                              "friedman_chi2": float(chi2), "friedman_p": float(p), "iman_davenport_F": float(ff),
                              "iman_davenport_p": pff, "nemenyi_q_0.05": float(q), "nemenyi_CD": float(cd)})
                pr = []
                for i, j in combinations(range(k), 2):
                    d = DM[:, i] - DM[:, j]
                    try:
                        w = stats.wilcoxon(DM[:, i], DM[:, j], zero_method="wilcox") if np.any(d != 0) else None
                        pw = float(w.pvalue) if w is not None else 1.0
                    except ValueError:
                        pw = 1.0
                    pr.append({"set": sname, "pool": pool, "protocol": proto, "method_a": ms[i], "method_b": ms[j],
                               "mean_a": float(DM[:, i].mean()), "mean_b": float(DM[:, j].mean()),
                               "datasets_a_wins": int((d > 0).sum()), "datasets_b_wins": int((d < 0).sum()),
                               "wilcoxon_p": pw, "avg_rank_a": float(avg[i]), "avg_rank_b": float(avg[j]),
                               "nemenyi_sig": bool(abs(avg[i] - avg[j]) > cd)})
                if pr:
                    adj = holm([x["wilcoxon_p"] for x in pr])
                    for x, a in zip(pr, adj):
                        x["wilcoxon_p_holm"] = float(a)
                        x["holm_sig_0.05"] = bool(a < alpha)
                prows += pr
                cl = cliques(ms, avg, cd)
                letters = {m: "" for m in ms}
                for gi, g in enumerate(cl):
                    for m in g:
                        letters[m] += chr(ord("a") + gi) if gi < 26 else "*"
                for j in np.argsort(avg, kind="stable"):
                    m = ms[j]
                    crows.append({"set": sname, "pool": pool, "protocol": proto, "method": m, "avg_rank": float(avg[j]),
                                  "mean_auroc": float(DM[:, j].mean()), "nemenyi_CD": float(cd),
                                  "cd_groups": letters[m] or "-",
                                  "not_sig_diff_from_best": bool(avg[j] - avg.min() <= cd)})
    return pd.DataFrame(frows), pd.DataFrame(prows), pd.DataFrame(crows)


M1_CRIT = ["oracle", "valnormal", "last", "pseudo", "pseudo_ensemble", "ckpt_ensemble", "consensus", "aucp",
           "oracle_nostop"]


def selectors(m1, B, seed):
    """Per method (own complete datasets) and pooled over methods (method-average), M1 runs."""
    if m1 is None:
        return pd.DataFrame()
    rows = []
    boot = Boot(len(UB9), B, seed)
    per_method = {}
    for m in OLD4:
        sub = m1[m1.method == m]
        dss = [d for d in UB9 if all(coverage(sub, m, [d], c)[d] == NFOLD for c in M1_CRIT if c in set(sub.protocol))]
        per_method[m] = dss
        for c in M1_CRIT:
            if c not in set(sub.protocol):
                continue
            A = cube(sub, [m], UB9, c)
            Av = cube(sub, [m], UB9, "valnormal")
            Ao = cube(sub, [m], UB9, "oracle")
            mask = np.array([d in dss for d in UB9])
            dm = A[0, mask].mean(-1)
            gap = (Ao - Av)[0, mask].mean(-1).mean()
            gain = (A - Av)[0, mask].mean(-1)
            rows.append({"method": m, "criterion": c, "n_datasets": int(mask.sum()), "mean_auroc": float(dm.mean()),
                         "minus_valnormal": float(gain.mean()),
                         "recovered_fraction_of_oracle_gap": float(gain.mean() / gap) if gap else np.nan})
    # pooled: method-average of per-method dataset means (GLocalKD over its 8 datasets), paired bootstrap
    for c in M1_CRIT:
        fb_all, db_all, vals = [], [], []
        for m in OLD4:
            sub = m1[m1.method == m]
            if c not in set(sub.protocol):
                continue
            mask = np.array([d in per_method[m] for d in UB9])
            D = (cube(sub, [m], UB9, c) - cube(sub, [m], UB9, "valnormal"))
            Dm = np.where(mask[None, :, None], D, np.nan)
            g = Dm[:, np.arange(len(UB9))[None, :, None], boot.fold_idx]  # (1,B,D,5)
            fb_all.append(np.nanmean(g.mean(-1), -1)[0])
            dmm = Dm[0].mean(-1)
            # dataset bootstrap restricted to the method's datasets
            didx = np.flatnonzero(mask)
            rs = np.random.default_rng(seed + 7).integers(0, len(didx), size=(B, len(didx)))
            db_all.append(dmm[didx][rs].mean(-1))
            vals.append(np.nanmean(dmm))
        if len(vals) == 4:
            fb, db = np.mean(fb_all, 0), np.mean(db_all, 0)
            rows.append({"method": "POOLED(4)", "criterion": c, "n_datasets": -1,
                         "mean_auroc": float(np.mean([r["mean_auroc"] for r in rows if r["criterion"] == c and r["method"] in OLD4])),
                         "minus_valnormal": float(np.mean(vals)), "fold_ci_lo": ci(fb)[0], "fold_ci_hi": ci(fb)[1],
                         "ds_ci_lo": ci(db)[0], "ds_ci_hi": ci(db)[1]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# PREREG
# ----------------------------------------------------------------------------------------------
def load_typed():
    frames = []
    p = osp.join(RES, "m2_typed_summary.csv")
    if osp.exists(p):
        t = pd.read_csv(p)
        frames.append(t.assign(src="m2_typed_summary.csv"))
    for p in sorted(glob.glob(osp.join(RES, "eval_typed*.csv"))):
        t = pd.read_csv(p)
        if "auroc_mean" not in t and "auroc" in t:
            t["auroc"] = pct(t["auroc"])
            keys = [c for c in ("method", "feature_variant", "criterion", "dataset", "anomaly_type") if c in t]
            t = t.groupby(keys, as_index=False).agg(auroc_mean=("auroc", "mean"), n_seeds=("auroc", "size"))
        frames.append(t.assign(src=osp.basename(p)))
    if not frames:
        return None
    t = pd.concat(frames, ignore_index=True)
    t["method"] = t["method"].map(canon_method)
    t["auroc_mean"] = pd.to_numeric(t["auroc_mean"])
    return t


def status(met, n_have, n_need_final, final):
    if met is None:
        return "PENDING"
    if final or n_have >= n_need_final:
        return "PASS" if met else "FAIL"
    return "PENDING"


def prereg(long, B, seed, final):
    dss = UB9
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "final_mode": bool(final),
           "rule": "PASS/FAIL once all 12 deep detectors are complete for the criterion, or with --final "
                   "(on the 2026-11-08 decision date); otherwise PENDING with the current value"}
    deep_cov = {}
    for m in DEEP12:
        deep_cov[m] = {p: sum(v == NFOLD for v in coverage(long, m, dss, p).values()) for p in
                       ("oracle", "valnormal", "last", "hpfree_valnormal")}
    out["deep_coverage_UB9_datasets_complete"] = deep_cov
    # ---- G1
    c1 = [m for m in DEEP12 if complete(long, m, dss, ["oracle", "valnormal"])]
    boot = Boot(len(dss), B, seed)
    g1 = {"complete_methods": c1, "n_complete": len(c1), "n_required": 8}
    if c1:
        D = cube(long, c1, dss, "oracle") - cube(long, c1, dss, "valnormal")
        per = D.mean(-1).mean(-1)
        fb, db = boot.fold(D).mean(0), boot.ds(D).mean(0)
        g1.update(inflation_per_method={m: float(v) for m, v in zip(c1, per)},
                  mean_inflation=float(per.mean()), mean_inflation_fold_ci=ci(fb), mean_inflation_ds_ci=ci(db),
                  frac_methods_ge2=float((per >= 2.0).mean()), n_methods_ge2=int((per >= 2.0).sum()))
        cond = (len(c1) >= 8, per.mean() >= 3.0, (per >= 2.0).mean() >= 0.75)
        g1["conditions"] = {"n_complete>=8": cond[0], "mean>=3.0": bool(cond[1]), "frac>=2.0 >= 0.75": bool(cond[2])}
        g1["currently_met"] = bool(all(cond))
        g1["inflation_conditions_met_on_current_methods"] = bool(cond[1] and cond[2])
        g1["coverage_condition_met"] = bool(cond[0])
        g1["status"] = status(all(cond), len(c1), 12, final)
    else:
        g1["status"] = status(None if not final else False, 0, 12, final)
    out["G1"] = g1
    # ---- G2(a)
    ga = {"pool": "deep methods with complete UB9 ORACLE and VAL-NORMAL", "methods": c1}
    if len(c1) >= 2:
        mo = cube(long, c1, dss, "oracle").mean(-1).mean(-1)
        mv = cube(long, c1, dss, "valnormal").mean(-1).mean(-1)
        t = float(tau_b(mo, mv))
        to, tv = top_k(mo, c1), top_k(mv, c1)
        ga.update(tau_b=t, top3_oracle=to, top3_valnormal=tv, top3_changed=set(to) != set(tv),
                  leaderboard_oracle={m: float(v) for m, v in zip(c1, mo)},
                  leaderboard_valnormal={m: float(v) for m, v in zip(c1, mv)})
        met = (t <= 0.8) or (set(to) != set(tv))
        if len(c1) <= 3:
            ga["note"] = "top-3 set of <= 3 methods cannot change"
        ga["currently_met"] = bool(met)
        ga["status"] = status(met, len(c1), 12, final)
    else:
        ga["status"] = status(None if not final else False, len(c1), 12, final)
    out["G2a"] = ga
    # ---- G2(b)
    cb = [m for m in DEEP12 if complete(long, m, dss, ["valnormal", "hpfree_valnormal"])]
    gb = {"pool": "deep methods with complete UB9 VAL-NORMAL and HP-FREE (hpfree_valnormal)", "methods": cb,
          "n_complete": len(cb)}
    if cb:
        D = cube(long, cb, dss, "valnormal") - cube(long, cb, dss, "hpfree_valnormal")
        per = D.mean(-1).mean(-1)
        gb.update(hp_leak_per_method={m: float(v) for m, v in zip(cb, per)}, mean_hp_leak=float(per.mean()),
                  mean_fold_ci=ci(boot.fold(D).mean(0)), mean_ds_ci=ci(boot.ds(D).mean(0)))
        met = per.mean() >= 2.0
        gb["currently_met"] = bool(met)
        gb["status"] = status(met, len(cb), 12, final)
    else:
        gb["status"] = status(None if not final else False, 0, 12, final)
    out["G2b"] = gb
    # ---- G2(c) (post-hoc)
    gc = {"label": "POST-HOC criterion (revised after M2 showed T2 is detectable; not an independent confirmation)",
          "variant": "F2 (M2 primary: one-hot labels + raw x)", "criterion": "valnormal"}
    t = load_typed()
    if t is not None:
        dm = t[(t.feature_variant == "F2") & (t.criterion == "valnormal") & (t.dataset == "GX")
               & t.anomaly_type.isin(TYPED_TYPES) & t.method.isin(DEEP12)]
        per = dm.pivot_table(index="method", columns="anomaly_type", values="auroc_mean", aggfunc="mean")
        per = per.dropna(subset=[c for c in TYPED_TYPES if c in per.columns]) if set(TYPED_TYPES) <= set(per.columns) else per.iloc[0:0]
        ok = (per[TYPED_TYPES] <= 60).all(axis=1) if len(per) else pd.Series(dtype=bool)
        ctrl = t[(t.method == "GRAIL0:ctrl_gc_lw") & (t.feature_variant == "F2") & (t.dataset == "GX")
                 & t.anomaly_type.isin(TYPED_TYPES)].groupby("anomaly_type").auroc_mean.mean()
        ctrl_ok = bool(len(ctrl) == 3 and (ctrl >= 85).all())
        gc.update(deep_methods=list(per.index), n_methods=int(len(per)),
                  per_method={m: {k: float(per.loc[m, k]) for k in TYPED_TYPES} for m in per.index},
                  methods_all_le60=[m for m in per.index if ok[m]],
                  frac_le60=float(ok.mean()) if len(per) else None,
                  graphlet_count_lw={k: float(v) for k, v in ctrl.items()}, graphlet_count_lw_all_ge85=ctrl_ok)
        # sensitivity: F1 (labels only)
        d1 = t[(t.feature_variant == "F1") & (t.criterion == "valnormal") & (t.dataset == "GX")
               & t.anomaly_type.isin(TYPED_TYPES) & t.method.isin(DEEP12)]
        p1 = d1.pivot_table(index="method", columns="anomaly_type", values="auroc_mean", aggfunc="mean")
        if set(TYPED_TYPES) <= set(p1.columns) and len(p1):
            gc["sensitivity_F1_frac_le60"] = float((p1[TYPED_TYPES] <= 60).all(axis=1).mean())
            gc["sensitivity_F1_methods_failing"] = [m for m in p1.index if not (p1.loc[m, TYPED_TYPES] <= 60).all()]
        # denominator: deep methods in the study = typed-measured methods U methods counting toward G1;
        # a study method without typed results cannot count as <= 60 (conservative)
        study = sorted(set(per.index) | set(c1))
        n_ok = int(ok.sum()) if len(per) else 0
        gc["study_methods"] = study
        gc["frac_le60_over_study_methods"] = n_ok / len(study) if study else None
        if len(per):
            frac = (n_ok / len(study)) if final else float(ok.mean())
            met = bool(frac >= 0.75 and ctrl_ok)
            gc["currently_met"] = met
            gc["status"] = status(met, len(per), 12, final)
        else:
            gc["status"] = status(None if not final else False, 0, 12, final)
        gc["sources"] = sorted(t.src.unique())
    else:
        gc["status"] = "PENDING"
    out["G2c"] = gc
    sts = [out[k]["status"] for k in ("G2a", "G2b", "G2c")]
    g2 = "PASS" if "PASS" in sts else ("FAIL" if all(s == "FAIL" for s in sts) else "PENDING")
    out["G2"] = g2
    out["GO"] = ("GO" if (g1["status"] == "PASS" and g2 == "PASS") else
                 "NO-GO" if (g1["status"] == "FAIL" or g2 == "FAIL") else "PENDING")
    return out


# ----------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------
def f2(x, col=""):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "–"
    if col.endswith("_p") or col.endswith("p_holm") or col == "wilcoxon_p":
        return f"{x:.2g}" if x >= 1e-3 else f"{x:.1e}"
    return f"{x:.2f}"


def md_table(df, cols, heads=None):
    heads = heads or cols
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f2(r[c], c) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_report(notes, lb, perds, infl, kd, fr, ph, cdt, sel, pr, kernels_raw, verbatim, args):
    L = []
    L.append("# Label-free GLAD re-evaluation: statistics report (AUTO-GENERATED)")
    L.append("")
    L.append(f"Generated {pr['generated']} by `prelim/evalpaper/stats/glad_stats.py` (B = {args.B} bootstrap "
             f"replicates, seed {args.seed}{', FINAL mode' if args.final else ''}). Do not edit by hand; rerun the "
             "script. All numbers are AUROC × 100 unless noted; a leaderboard value is the mean over the datasets "
             "of the set of the 5-fold means. CIs are 95% percentile intervals from paired fold- and "
             "dataset-bootstraps (fold-bootstrap CIs are optimistic because folds share training data; Tox21 "
             "'folds' are 5 seeds on one fixed split).")
    L.append("")
    L.append("## 1. Inputs and coverage")
    for n in notes:
        L.append(f"- {n}")
    L.append("- Protocol mapping for methods without checkpoints: KernelGLAD 'tuned' (test-selected HP) → "
             "oracle/last/valnormal, 'default' (UB-GOLD defaults) → hpfree_*; GRAIL-0 and controls are a-priori "
             "and identical in every protocol. Only train-fit KernelGLAD rows are ranked (`@testfit` = UB-GOLD's "
             "transductive code path, reported in §8).")
    cov = []
    for m, c in pr["deep_coverage_UB9_datasets_complete"].items():
        cov.append(f"{m}: oracle {c['oracle']}/9, valnormal {c['valnormal']}/9, last {c['last']}/9, "
                   f"hpfree_valnormal {c['hpfree_valnormal']}/9")
    L.append("- Deep-detector UB9 coverage (datasets with all 5 folds):")
    for c in cov:
        L.append(f"  - {c}")
    L.append("")
    # PREREG
    L.append("## 2. Automatic PREREG evaluation (prelim/evalpaper/PREREG.md, go/no-go 2026-11-08)")
    L.append("")
    L.append(f"Rule: {pr['rule']}.")
    L.append("")
    g1, ga, gb, gc = pr["G1"], pr["G2a"], pr["G2b"], pr["G2c"]
    def cm(d):
        return "" if "currently_met" not in d else (" — currently MET" if d["currently_met"] else " — currently NOT met")
    g1msg = ""
    if "coverage_condition_met" in g1:
        g1msg = (f" — coverage condition {'met' if g1['coverage_condition_met'] else 'NOT met'}; inflation conditions "
                 f"{'met' if g1['inflation_conditions_met_on_current_methods'] else 'NOT met'} on the complete methods")
    L.append(f"- **G1: {g1['status']}**{g1msg}. Complete deep methods on UB9: {g1['n_complete']}/12 "
             f"(≥ 8 required): {', '.join(g1['complete_methods']) or 'none'}.")
    if "mean_inflation" in g1:
        L.append(f"  - Mean ORACLE − VAL-NORMAL over these methods: {g1['mean_inflation']:.2f} "
                 f"(fold CI {g1['mean_inflation_fold_ci'][0]:.2f}–{g1['mean_inflation_fold_ci'][1]:.2f}; dataset CI "
                 f"{g1['mean_inflation_ds_ci'][0]:.2f}–{g1['mean_inflation_ds_ci'][1]:.2f}); threshold ≥ 3.0.")
        L.append(f"  - Methods with ≥ 2.0: {g1['n_methods_ge2']}/{g1['n_complete']} "
                 f"({100*g1['frac_methods_ge2']:.0f}%; threshold ≥ 75%). Per method: " +
                 ", ".join(f"{m} {v:.2f}" for m, v in g1["inflation_per_method"].items()) + ".")
    L.append(f"- **G2(a): {ga['status']}**{cm(ga)}. " + (
        f"Kendall τ-b(ORACLE, VAL-NORMAL) on UB9 = {ga['tau_b']:.2f} over {len(ga['methods'])} deep methods "
        f"(threshold ≤ 0.8); top-3 ORACLE {ga['top3_oracle']} vs VAL-NORMAL {ga['top3_valnormal']} "
        f"(changed: {ga['top3_changed']})." if "tau_b" in ga else "no data."))
    L.append(f"- **G2(b): {gb['status']}**{cm(gb)}. " + (
        f"Mean VAL-NORMAL − HP-FREE over {gb['n_complete']} deep methods = {gb['mean_hp_leak']:.2f} "
        f"(threshold ≥ 2.0)." if "mean_hp_leak" in gb else
        "No deep method has complete UB9 HP-FREE (hpfree_valnormal) runs yet (0/12)."))
    if "per_method" in gc:
        L.append(f"- **G2(c) [POST-HOC]: {gc['status']}**{cm(gc)}. {gc['label']}. Deep methods with VAL-NORMAL ≤ 60 "
                 f"on all of T1a/T3a/T3b (F2): {len(gc['methods_all_le60'])}/{gc['n_methods']} measured "
                 f"({len(gc['methods_all_le60'])}/{len(gc['study_methods'])} of the study's deep methods; with --final "
                 f"the latter is used) "
                 f"({100*(gc['frac_le60'] or 0):.0f}%; threshold ≥ 75%); graphlet-count + LW: " +
                 ", ".join(f"{k} {v:.1f}" for k, v in gc["graphlet_count_lw"].items()) +
                 f" (all ≥ 85: {gc['graphlet_count_lw_all_ge85']}). Per method: " +
                 "; ".join(f"{m} " + "/".join(f"{v:.1f}" for v in d.values()) for m, d in gc["per_method"].items()) + ".")
        if "sensitivity_F1_frac_le60" in gc:
            L.append(f"  - Sensitivity (F1, labels only; not the criterion): {100*gc['sensitivity_F1_frac_le60']:.0f}% "
                     f"of methods ≤ 60 on all three; failing: {gc['sensitivity_F1_methods_failing'] or 'none'}.")
    else:
        L.append(f"- **G2(c) [POST-HOC]: {gc['status']}** (no typed results found).")
    L.append(f"- **G2 overall: {pr['G2']}; decision: {pr['GO']}.**")
    L.append("")
    # leaderboards
    for sname in SETS:
        L.append(f"## 3{'a' if sname == 'UB9' else 'b'}. Leaderboards on {sname} "
                 f"({', '.join(SETS[sname])})")
        sub = lb[lb.set == sname]
        if sub.empty:
            L.append("No data.")
            continue
        w = sub.pivot_table(index="method", columns="protocol", values="mean_auroc")
        w = w[[p for p in PROTOCOLS if p in w.columns]]
        fam = sub.drop_duplicates("method").set_index("method")["family"]
        comp = sub.groupby("method").complete.all()
        w = w.assign(family=fam, complete=comp).sort_values(["family", w.columns[0]], ascending=[True, False])
        L.append("")
        L.append("Mean AUROC per protocol (methods with incomplete coverage of the set are marked `*` and are "
                 "not ranked; their value averages only the datasets they cover).")
        L.append("")
        heads = ["method", "family"] + [p for p in PROTOCOLS if p in w.columns]
        L.append("| " + " | ".join(heads) + " |")
        L.append("|" + "---|" * len(heads))
        for m, r in w.iterrows():
            cells = []
            for p in heads[2:]:
                e = sub[(sub.method == m) & (sub.protocol == p)]
                if e.empty:
                    cells.append("–")
                    continue
                e = e.iloc[0]
                cells.append(f"{e.mean_auroc:.2f}" + ("" if e.complete else "*") +
                             (f" (#{int(e['rank'])})" if e.complete and np.isfinite(e.get("rank", np.nan)) else ""))
            L.append(f"| {m} | {r.family} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("95% CIs for the label-free main protocol (valnormal; fold bootstrap / dataset bootstrap):")
        L.append("")
        s2 = sub[(sub.protocol == "valnormal") & sub.complete].sort_values("mean_auroc", ascending=False)
        if len(s2):
            L.append(md_table(s2, ["method", "family", "mean_auroc", "fold_ci_lo", "fold_ci_hi", "ds_ci_lo", "ds_ci_hi", "mean_auprc", "mean_fpr95"],
                              ["method", "family", "AUROC", "fold lo", "fold hi", "ds lo", "ds hi", "AUPRC", "FPR95"]))
        L.append("")
    # inflation
    L.append("## 4. Oracle-minus-label-free inflation per method")
    L.append("")
    if len(infl):
        L.append(md_table(infl.sort_values(["set", "comparison", "method"]),
                          ["set", "method", "comparison", "mean_diff", "fold_ci_lo", "fold_ci_hi", "ds_ci_lo", "ds_ci_hi",
                           "n_datasets_diff_ge2", "max_dataset_diff", "max_fold_diff"],
                          ["set", "method", "comparison", "mean", "fold lo", "fold hi", "ds lo", "ds hi", "#ds ≥ 2", "max ds", "max fold"]))
    L.append("")
    # kendall
    L.append("## 5. Agreement between protocol leaderboards (Kendall τ-b, top-3 sets)")
    L.append("")
    if len(kd):
        L.append(md_table(kd, ["set", "pool", "protocol_a", "protocol_b", "n_methods", "tau_b", "fold_ci_lo", "fold_ci_hi",
                               "ds_ci_lo", "ds_ci_hi", "top3_changed", "p_top3_changed_foldboot", "p_top3_changed_dsboot"],
                          ["set", "pool", "A", "B", "k", "τ-b", "fold lo", "fold hi", "ds lo", "ds hi", "top-3 changed",
                           "P(change) fold", "P(change) ds"]))
        L.append("")
        L.append("Top-3 sets: " + "; ".join(f"{r.set}/{r.pool} {r.protocol_a}: {r.top3_a} → {r.protocol_b}: {r.top3_b}"
                                             for r in kd.itertuples() if r.protocol_a == "oracle" and r.protocol_b == "valnormal"))
    else:
        L.append("Fewer than 3 methods complete in every pair of protocols.")
    L.append("")
    # friedman
    L.append("## 6. Friedman / Iman–Davenport, Holm-corrected Wilcoxon and Nemenyi critical differences")
    L.append("")
    if len(fr):
        L.append(md_table(fr, ["set", "pool", "protocol", "n_methods", "n_datasets", "friedman_chi2", "friedman_p",
                               "iman_davenport_p", "nemenyi_CD"],
                          ["set", "pool", "protocol", "k", "N", "χ²", "p", "Iman–Davenport p", "CD (α=0.05)"]))
        L.append("")
        L.append("Critical-difference tables (average rank over datasets, 1 = best; methods sharing a letter are "
                 "not separated by the Nemenyi CD):")
        for (sname, pool, proto), g in cdt.groupby(["set", "pool", "protocol"], sort=False):
            if proto not in ("oracle", "valnormal", "hpfree_valnormal"):
                continue
            L.append("")
            L.append(f"*{sname} / {pool} / {proto}* (CD = {g.nemenyi_CD.iloc[0]:.2f}): " +
                     ", ".join(f"{r.method} {r.avg_rank:.2f}[{r.cd_groups}]" for r in g.itertuples()))
        sig = ph[ph.get("holm_sig_0.05", pd.Series(dtype=bool)) == True] if len(ph) else ph  # noqa: E712
        L.append("")
        L.append(f"Holm-significant pairwise Wilcoxon differences (α = 0.05): {len(sig)} of {len(ph)} pairs "
                 "(full list in `eval_leaderboard_posthoc.csv`; with N = 9 datasets the smallest attainable "
                 "two-sided Wilcoxon p is 0.0039, so Holm over many pairs rarely rejects).")
    L.append("")
    # selectors
    L.append("## 7. Label-free checkpoint selectors (M1 runs, UB-GOLD HP; GLocalKD on 8 datasets)")
    L.append("")
    if len(sel):
        pooled = sel[sel.method == "POOLED(4)"]
        L.append(md_table(pooled, ["criterion", "mean_auroc", "minus_valnormal", "fold_ci_lo", "fold_ci_hi", "ds_ci_lo", "ds_ci_hi"],
                          ["criterion", "AUROC (4-method mean)", "− VAL-NORMAL", "fold lo", "fold hi", "ds lo", "ds hi"]))
        L.append("")
        w = sel[sel.method != "POOLED(4)"].pivot_table(index="criterion", columns="method", values="recovered_fraction_of_oracle_gap")
        L.append("Fraction of each method's ORACLE − VAL-NORMAL gap recovered (1 = ORACLE, 0 = VAL-NORMAL): " +
                 "; ".join(f"{c}: " + ", ".join(f"{m} {w.loc[c, m]:+.2f}" for m in w.columns) for c in w.index
                           if c not in ("oracle", "valnormal")))
    L.append("")
    # kernels
    L.append("## 8. KernelGLAD (UB-GOLD's WL/PK + IF/OCSVM)")
    L.append("")
    if kernels_raw is not None:
        k = kernels_raw.copy()
        k["auroc"] = pct(k["auroc"])
        k["constant"] = k["hp_config"].astype(str).str.contains('"kernel_constant": true')
        L.append("5-fold means per dataset. Rows without `@testfit`: kernel and detector fitted on the normal "
                 "training graphs (same protocol as every other method; these rows are ranked above). `@testfit`: "
                 "UB-GOLD's own code path, which fits kernel and detector on the unlabelled test set (scores paired "
                 "with the correct labels). 'tuned' = configuration with the best mean TEST AUROC over UB-GOLD's "
                 "KernelGLAD search space (oracle HP); 'default' = UB-GOLD defaults = the released `WL_IF.sh` line. "
                 "§ = constant kernel matrix (PK on graphs with constant UB-GOLD node features), AUROC 50 by construction.")
        L.append("")
        for sname, dss in SETS.items():
            s_ = k[k.dataset.isin(dss)]
            if s_.empty:
                continue
            w = s_.groupby(["method", "protocol", "dataset"]).auroc.mean().unstack("dataset")
            cst = s_.groupby(["method", "protocol", "dataset"]).constant.all().unstack("dataset")
            nf = s_.groupby(["method", "protocol", "dataset"]).fold.nunique().unstack("dataset")
            cols = [d for d in dss if d in w.columns]
            L.append(f"*{sname}*:")
            L.append("")
            L.append("| method | protocol | " + " | ".join(cols) + " | mean |")
            L.append("|" + "---|" * (len(cols) + 3))
            for (m, p), r in w.iterrows():
                cells = []
                for d in cols:
                    v = r.get(d, np.nan)
                    flag = "§" if (d in cst.columns and bool(cst.loc[(m, p), d] == True)) else ""  # noqa: E712
                    flag += "" if (d in nf.columns and nf.loc[(m, p), d] == NFOLD) or pd.isna(v) else "*"
                    cells.append(f2(v) + flag)
                full = all(np.isfinite(r.get(d, np.nan)) for d in dss)
                L.append(f"| {m} | {p} | " + " | ".join(cells) + f" | {f2(np.mean([r[d] for d in dss])) if full else '–'} |")
            L.append("")
    if verbatim is not None and len(verbatim):
        v = verbatim.copy()
        for c in ("auroc_released", "auroc_aligned", "frac_label_mismatch", "kernel_maxabs_vs_ours"):
            v[c] = pd.to_numeric(v[c])
        g = v.groupby("dataset").agg(folds=("fold", "size"), released=("auroc_released", "mean"),
                                     aligned=("auroc_aligned", "mean"), mismatch=("frac_label_mismatch", "mean"),
                                     kmax=("kernel_maxabs_vs_ours", "max")).reset_index()
        L.append("UB-GOLD's verbatim WL-IF pipeline (`WL_IF.sh`; scores fitted on the test set) with the released "
                 "label pairing vs the correct pairing of the same scores (`eval_kernels_ubgold_verbatim.csv`):")
        L.append("")
        L.append(md_table(g, ["dataset", "folds", "released", "aligned", "mismatch", "kmax"],
                          ["dataset", "folds", "AUROC released pairing", "AUROC aligned", "share of labels mismatched",
                           "max |K_ubgold − K_ours|"]))
    pp = osp.join(RES, "eval_kernels_published.csv")
    if osp.exists(pp):
        pub = pd.read_csv(pp)
        pub["absz"] = pd.to_numeric(pub["z_vs_chance"], errors="coerce").abs()
        L.append("")
        L.append("UB-GOLD's published kernel AUROCs (Table 3, arXiv v2; `eval_kernels_published.csv`) against the chance "
                 "band of a 5-fold mean under random score/label pairing: " + "; ".join(
                     f"{fam}: {int((g.absz <= 3).sum())}/{len(g)} cells within |z| ≤ 3"
                     for fam, g in pub.groupby(pub.method.str.contains("OCSVM").map({True: "OCSVM", False: "IF"}))) +
                 ". Cells with |z| > 3: " + ", ".join(f"{r.dataset} {r.method} {r.published_auroc:.2f} (z {r.z_vs_chance:+.1f})"
                                                      for r in pub[pub.absz > 3].itertuples()) + ".")
    po = osp.join(RES, "eval_kernels_orientation_check.csv")
    if osp.exists(po):
        o = pd.read_csv(po)
        oc = o[o.detector == "OCSVM"]
        L.append("")
        L.append("Score orientation (fixed from UB-GOLD KernelGLAD.py:139 `-decision_function` and the sklearn docs; "
                 "`eval_kernels_orientation_check.csv`, fold 0, train-fit, defaults): label-free, OCSVM flags "
                 f"{oc.frac_train_decision_lt0.min():.3f}–{oc.frac_train_decision_lt0.max():.3f} of the training normals "
                 f"(nu = 0.1) and its decision value rises with a test graph's mean kernel similarity to the training "
                 f"graphs (Spearman {oc.spearman_decision_vs_meansim.min():.2f}–{oc.spearman_decision_vs_meansim.max():.2f}). "
                 "Sub-50 OCSVM AUROCs therefore mean that the anomalous class is MORE similar to the training normals "
                 "(explanatory, label-using columns: OCSVM AUROC vs AUROC of −mean similarity, Pearson r = "
                 f"{np.corrcoef(oc.expl_auroc_score, oc.expl_auroc_neg_meansim)[0, 1]:.2f} over {len(oc)} dataset × kernel cells), "
                 "not a sign error.")
    L.append("")
    L.append("## 9. Output files")
    L.append("")
    for f in sorted(glob.glob(osp.join(RES, "eval_leaderboard_*"))):
        L.append(f"- `prelim/results/{osp.basename(f)}`")
    with open(REPORT, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    global RES, REPORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--final", action="store_true", help="decision-date mode: PENDING becomes PASS/FAIL")
    ap.add_argument("--prefer", default="deep_rerun", choices=["deep_rerun", "eval_folds"],
                    help="source for oracle/last/valnormal cells present in both")
    ap.add_argument("--res", default=RES, help="results directory (tests only)")
    ap.add_argument("--report", default=REPORT, help="report path (tests only)")
    args = ap.parse_args()
    RES, REPORT = args.res, args.report
    long, m1, notes = load_all(args)
    kp = osp.join(RES, "eval_kernels_folds.csv")
    kernels_raw = pd.read_csv(kp) if osp.exists(kp) else None
    vp = osp.join(RES, "eval_kernels_ubgold_verbatim.csv")
    verbatim = pd.read_csv(vp) if osp.exists(vp) else None
    lb, perds = leaderboards(long, args.B, args.seed)
    infl = inflation(long, args.B, args.seed, kernels_raw)
    kd = kendall(long, args.B, args.seed)
    fr, ph, cdt = friedman(long)
    sel = selectors(m1, args.B, args.seed)
    pr = prereg(long, args.B, args.seed, args.final)
    out = {"long": lb, "per_dataset": perds, "inflation": infl, "kendall": kd, "friedman": fr, "posthoc": ph,
           "cd": cdt, "selectors": sel}
    for k, df in out.items():
        df.to_csv(osp.join(RES, f"eval_leaderboard_{k}.csv"), index=False, float_format="%.4f")
    cov_rows = []
    for (m, p, d), g in long.groupby(["method", "protocol", "dataset"]):
        cov_rows.append({"method": m, "protocol": p, "dataset": d, "n_folds": g.fold.nunique(),
                         "source": "|".join(sorted(g.source.unique()))})
    pd.DataFrame(cov_rows).to_csv(osp.join(RES, "eval_leaderboard_coverage.csv"), index=False)
    with open(osp.join(RES, "eval_leaderboard_prereg.json"), "w") as fh:
        json.dump(pr, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_report(notes, lb, perds, infl, kd, fr, ph, cdt, sel, pr, kernels_raw, verbatim, args)
    # console summary
    print("PREREG (prelim/evalpaper/PREREG.md):")
    for k in ("G1", "G2a", "G2b", "G2c"):
        d = pr[k]
        extra = ""
        if k == "G1" and "mean_inflation" in d:
            extra = f"n_complete={d['n_complete']}/12 mean(ORACLE-VALNORMAL)={d['mean_inflation']:.2f} frac>=2: {d['frac_methods_ge2']:.2f}"
        elif k == "G1":
            extra = f"n_complete={d['n_complete']}/12"
        if k == "G2a" and "tau_b" in d:
            extra = f"tau_b={d['tau_b']:.2f} top3_changed={d['top3_changed']} (k={len(d['methods'])})"
        if k == "G2b":
            extra = (f"mean HP leak={d['mean_hp_leak']:.2f} (k={d['n_complete']})" if "mean_hp_leak" in d
                     else f"no HP-FREE data (k={d['n_complete']})")
        if k == "G2c" and "frac_le60" in d:
            extra = f"[POST-HOC] frac<=60={d['frac_le60']:.2f} (k={d['n_methods']}) gc+LW>=85: {d['graphlet_count_lw_all_ge85']}"
        cmet = "" if "currently_met" not in d else (" currently MET" if d["currently_met"] else " currently NOT met")
        if k == "G1" and "coverage_condition_met" in d:
            cmet = (f" (coverage {'met' if d['coverage_condition_met'] else 'NOT met'}: {d['n_complete']}/12 complete, >= 8 needed;"
                    f" inflation conditions {'met' if d['inflation_conditions_met_on_current_methods'] else 'NOT met'})")
        print(f"  {k}: {d['status']}{cmet}  {extra}")
    print(f"  G2: {pr['G2']}   decision: {pr['GO']}")
    print(f"wrote {REPORT} and {RES}/eval_leaderboard_*.csv")


if __name__ == "__main__":
    main()
