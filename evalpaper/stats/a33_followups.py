"""A33 descriptive follow-ups, implemented as fixed in PREREG_AMENDMENTS A35 (written before any A33 metric was read).

A33.1 anomaly-class orientation: the 12 UB-GOLD detectors on BZR, COX2, PROTEINS_full with class 1 (minority) as the
      anomaly class, against the main orientation on the same 15 (dataset, fold) cells; training-free references.
A33.2 x_s ablation (A34: x_s = 1): CVTGAD and GOOD-D on UB9 (45 cells) and on the typed suite, against unablated runs.

Descriptive only: no verdict, threshold or imputation. glad_stats.py (frozen) is imported read-only for canon_method,
Boot, ci and scale100, so intervals use the same bootstrap as the official run.

  cd <repo> && PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -B evalpaper/stats/a33_followups.py
"""
import hashlib
import importlib.util
import os.path as osp
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = osp.dirname(osp.abspath(__file__))
PRELIM = osp.dirname(osp.dirname(HERE))
RES = osp.join(PRELIM, "results")
REPORT = osp.join(HERE, "A33_REPORT.md")
B, SEED = 4000, 20261012
DS3 = ["BZR", "COX2", "PROTEINS_full"]
PROTOS = ["oracle", "last", "valnormal"]
LABEL = {"oracle": "ORACLE", "last": "LAST", "valnormal": "VAL-NORMAL"}
REFS = {"ctrl_size_lw": "size only", "ctrl_gc_lw": "graphlet count + LW", "grail0_sumz": "GRAIL-0"}
ABL = ["CVTGAD", "GOOD-D"]


def load_gs():
    p = osp.join(HERE, "glad_stats.py")
    spec = importlib.util.spec_from_file_location("_a33_glad_stats", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, p


GS, GS_PATH = load_gs()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def need(name):
    p = osp.join(RES, name)
    if not osp.exists(p):
        sys.exit(f"missing input {p} (run the A33 collect scripts first)")
    return p


def cube(df, methods, dss, proto):
    """(M, D, 5) AUROC array from long rows (method, dataset, fold, protocol, auroc); NaN where missing."""
    A = np.full((len(methods), len(dss), 5), np.nan)
    mi, di = {m: i for i, m in enumerate(methods)}, {d: i for i, d in enumerate(dss)}
    g = df[df.protocol == proto]
    for m, d, f, v in zip(g.method, g.dataset, g.fold, g.auroc):
        if m in mi and d in di and 0 <= int(f) < 5:
            A[mi[m], di[d], int(f)] = v
    return A


def boot_ci(boot, A):
    """A: (M, D, 5) with no NaN -> (fold CI, dataset CI) of the mean over all M, D, 5."""
    if not np.isfinite(A).all():
        return (np.nan, np.nan), (np.nan, np.nan)
    return GS.ci(boot.fold(A).mean(0)), GS.ci(boot.ds(A).mean(0))


def main_rows(prov):
    p = need("eval_leaderboard_v2_folds.csv")
    prov["eval_leaderboard_v2_folds.csv"] = sha(p)
    f = pd.read_csv(p, low_memory=False)
    f = f[(f.hp == "ubgold") & f.protocol.isin(PROTOS)].copy()
    f["auroc"] = pd.to_numeric(f.auroc, errors="coerce")
    return f


def a33_rows(name, prov):
    p = need(name)
    prov[name] = sha(p)
    d = pd.read_csv(p, low_memory=False)
    d["method"] = d["method"].map(GS.canon_method)
    d["dataset"] = d["dataset"].map(lambda x: GS.DS_ALIASES.get(x, x))
    if "orientation" in d:                                   # A5: GraphDE paper orientation only
        d = d[d.orientation.astype(str) != "ubgold"]
    if "hp_config" in d:
        d = d[~d.hp_config.astype(str).str.startswith("hpfree")]
    d = d[d.protocol.isin(PROTOS)].copy()
    d["auroc"] = pd.to_numeric(d.auroc, errors="coerce") * GS.scale100(d)
    return d


def orientation(prov, notes):
    main = main_rows(prov)
    flip = a33_rows("eval_orientation_folds.csv", prov)
    det = [m for m in GS.DEEP12]
    boot = GS.Boot(len(DS3), B, SEED)
    nonfin = int((~np.isfinite(flip.auroc)).sum())
    notes.append(f"A33.1: {len(flip)} flipped rows ({flip.method.nunique()} detectors), non-finite AUROC cells {nonfin}")
    rows, gaps = [], {}
    for conv, src in (("main (class 0 anomalous)", main), ("flipped (class 1 anomalous)", flip)):
        C = {p: cube(src, det, DS3, p) for p in PROTOS}
        for i, m in enumerate(det):
            r = dict(convention=conv, detector=m, n_cells=int(np.isfinite(C["oracle"][i]).sum()))
            for p in PROTOS:
                r[LABEL[p]] = float(np.nanmean(C[p][i]))
            r["ORACLE-VAL-NORMAL"] = float(np.nanmean(C["oracle"][i] - C["valnormal"][i]))
            r["ORACLE-LAST"] = float(np.nanmean(C["oracle"][i] - C["last"][i]))
            rows.append(r)
        for sel in ("valnormal", "last"):
            G = C["oracle"] - C[sel]
            ok = np.isfinite(G).all(axis=(1, 2))
            (flo, fhi), (dlo, dhi) = boot_ci(boot, G[ok])
            gaps[(conv, sel)] = dict(convention=conv, gap=f"ORACLE-{LABEL[sel]}", n_detectors=int(ok.sum()),
                                     mean=float(G[ok].mean()), fold_ci_lo=flo, fold_ci_hi=fhi, ds_ci_lo=dlo,
                                     ds_ci_hi=dhi, n_ge_2=int((G[ok].mean(axis=(1, 2)) >= 2.0).sum()))
    per = pd.DataFrame(rows)
    pooled = pd.DataFrame(list(gaps.values()))
    # per dataset (pooled over detectors, both orientations)
    pd_rows = []
    for conv, src in (("main (class 0 anomalous)", main), ("flipped (class 1 anomalous)", flip)):
        C = {p: cube(src, det, DS3, p) for p in PROTOS}
        for j, d in enumerate(DS3):
            r = dict(convention=conv, dataset=d)
            for p in PROTOS:
                r[LABEL[p]] = float(np.nanmean(C[p][:, j]))
            r["ORACLE-VAL-NORMAL"] = float(np.nanmean(C["oracle"][:, j] - C["valnormal"][:, j]))
            r["ORACLE-LAST"] = float(np.nanmean(C["oracle"][:, j] - C["last"][:, j]))
            pd_rows.append(r)
    per_ds = pd.DataFrame(pd_rows)
    # references
    pg, pr = need("grail0_folds.csv"), need("eval_orientation_refs_folds.csv")
    prov["grail0_folds.csv"], prov["eval_orientation_refs_folds.csv"] = sha(pg), sha(pr)
    ref_rows = []
    for conv, p in (("main (class 0 anomalous)", pg), ("flipped (class 1 anomalous)", pr)):
        g = pd.read_csv(p)
        g["dataset"] = g["dataset"].map(lambda x: GS.DS_ALIASES.get(x, x))
        g["auroc"] = pd.to_numeric(g.auroc, errors="coerce") * GS.scale100(g)
        g = g[g.method.isin(REFS) & g.dataset.isin(DS3)]
        for (m, d), x in g.groupby(["method", "dataset"]):
            ref_rows.append(dict(convention=conv, reference=REFS[m], dataset=d, n_folds=len(x), auroc=float(x.auroc.mean())))
    refs = pd.DataFrame(ref_rows)
    return per, pooled, per_ds, refs


def ablation(prov, notes):
    main = main_rows(prov)
    abl = a33_rows("eval_ablation_xs_ones_folds.csv", prov)
    UB9 = GS.UB9
    boot = GS.Boot(len(UB9), B, SEED)
    notes.append(f"A33.2 UB9: {len(abl)} ablated rows, non-finite AUROC cells {int((~np.isfinite(abl.auroc)).sum())}")
    rows = []
    for m in ABL:
        A1 = {p: cube(abl, [m], UB9, p) for p in PROTOS}
        A0 = {p: cube(main, [m], UB9, p) for p in PROTOS}
        for p in PROTOS:
            D = A1[p] - A0[p]
            ok = np.isfinite(D)
            (flo, fhi), (dlo, dhi) = boot_ci(boot, D) if ok.all() else ((np.nan, np.nan), (np.nan, np.nan))
            rows.append(dict(detector=m, quantity=LABEL[p], n_paired=int(ok.sum()), unablated=float(np.nanmean(A0[p])),
                             ablated=float(np.nanmean(A1[p])), diff=float(np.nanmean(D)), fold_ci_lo=flo,
                             fold_ci_hi=fhi, ds_ci_lo=dlo, ds_ci_hi=dhi))
        for sel in ("valnormal", "last"):
            I1, I0 = A1["oracle"] - A1[sel], A0["oracle"] - A0[sel]
            rows.append(dict(detector=m, quantity=f"ORACLE-{LABEL[sel]}", n_paired=int((np.isfinite(I1) & np.isfinite(I0)).sum()),
                             unablated=float(np.nanmean(I0)), ablated=float(np.nanmean(I1)),
                             diff=float(np.nanmean(I1 - I0))))
    ub9 = pd.DataFrame(rows)
    # per dataset VAL-NORMAL (descriptive)
    dsr = []
    for m in ABL:
        a1, a0 = cube(abl, [m], UB9, "valnormal")[0], cube(main, [m], UB9, "valnormal")[0]
        for j, d in enumerate(UB9):
            dsr.append(dict(detector=m, dataset=d, unablated=float(np.nanmean(a0[j])), ablated=float(np.nanmean(a1[j])),
                            diff=float(np.nanmean(a1[j] - a0[j]))))
    ub9_ds = pd.DataFrame(dsr)
    # typed
    pt, pc = need("eval_ablation_xs_ones_typed.csv"), need("eval_leaderboard_v2_typed_cells.csv")
    prov["eval_ablation_xs_ones_typed.csv"], prov["eval_leaderboard_v2_typed_cells.csv"] = sha(pt), sha(pc)
    t = pd.read_csv(pt)
    t["method"] = t["method"].map(GS.canon_method)
    t["auroc"] = pd.to_numeric(t.auroc, errors="coerce") * GS.scale100(t)
    t = t[(t.criterion == "valnormal") & t.method.isin(ABL)]
    t = t[((t.anomaly_type.str.startswith("T")) & (t.feature_variant == "F2")) | (t.anomaly_type.str.startswith("MS"))]
    notes.append(f"A33.2 typed: {len(t)} ablated VAL-NORMAL rows, non-finite {int((~np.isfinite(t.auroc)).sum())}")
    c = pd.read_csv(pc)
    c = c[c.method.isin(ABL) & (((c.generator == "GX") & (c.feature_variant == "F2")) | (c.generator == "MS"))]
    trow = []
    for (m, a, s), x in t.groupby(["method", "anomaly_type", "size_variant"]):
        u = c[(c.method == m) & (c.anomaly_type == a) & (c.size_variant == s)]
        un = float(u.valnormal_mean.iloc[0]) if len(u) else np.nan
        ab = float(x.auroc.mean())
        trow.append(dict(detector=m, generator="GX" if a.startswith("T") else "MS-v2", anomaly_type=a, size_variant=s,
                         n_seeds=int(x.seed.nunique()), unablated=un, ablated=ab, diff=ab - un,
                         blind_unablated=bool(un <= 60.0) if np.isfinite(un) else None, blind_ablated=bool(ab <= 60.0)))
    typed = pd.DataFrame(trow).sort_values(["detector", "generator", "anomaly_type", "size_variant"])
    return ub9, ub9_ds, typed


def md(df, nd=2):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    prov, notes = {"glad_stats.py": sha(GS_PATH), "a33_followups.py": sha(__file__)}, []
    per, pooled, per_ds, refs = orientation(prov, notes)
    ub9, ub9_ds, typed = ablation(prov, notes)
    outs = dict(orientation=per, orientation_pooled=pooled, orientation_datasets=per_ds, orientation_refs=refs,
                ablation_ub9=ub9, ablation_ub9_datasets=ub9_ds, ablation_typed=typed)
    for k, df in outs.items():
        df.to_csv(osp.join(RES, f"eval_a33_{k}.csv"), index=False)
    L = ["# A33 descriptive follow-ups (AUTO-GENERATED by a33_followups.py; A35)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}. Descriptive only; never part of G1/G2. AUROC in points.",
         "", "## Notes", ""] + [f"- {n}" for n in notes] + ["", "## A33.1 orientation: pooled checkpoint leakage", "",
         md(pooled), "", "## A33.1 per detector (means over 15 cells)", "", md(per), "",
         "## A33.1 per dataset (means over detectors)", "", md(per_ds), "", "## A33.1 references", "", md(refs), "",
         "## A33.2 ablation, UB9 (45 cells)", "", md(ub9), "", "## A33.2 ablation, UB9 VAL-NORMAL per dataset", "",
         md(ub9_ds), "", "## A33.2 ablation, typed VAL-NORMAL (mean over 5 seeds)", "", md(typed), "",
         "## Inputs (sha256)", ""] + [f"- `{k}`: `{v}`" for k, v in prov.items()]
    open(REPORT, "w").write("\n".join(L) + "\n")
    print(f"wrote {REPORT} and {len(outs)} eval_a33_*.csv files; " + "; ".join(notes))


if __name__ == "__main__":
    main()
