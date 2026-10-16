"""LaTeX tables for Appendices D (typed suite), E (kernel baselines) and F (additional results).

    PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/paper-figs/bin/python -B evalpaper/paper/tables/make_tables.py

Every number is read from the frozen-stats outputs (results/eval_leaderboard_v2_*.csv, written by
stats/glad_stats.py) or from the other result files named in SOURCES; nothing is typed by hand. Each output file holds
only a tabular environment (the float, caption and label live in sections/app_*.tex), plus a header comment with
the sha256 of every source it read. Re-run after the official `glad_stats.py --final` run: tables whose inputs are
still empty (A17 version runs, typed rows of the recent methods) are written with \\tbd{} cells and fill in on
regeneration. The few aggregates computed here (a 9- or 4-dataset mean of per-dataset kernel means, the 65-fold mean
of UB-GOLD's verbatim kernel pipeline) are checked against the values already printed in REPORT.md / kernels/NOTES.md.
"""
import hashlib
import json
import math
import os.path as osp
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
EVAL = osp.abspath(osp.join(HERE, "..", ".."))                 # evalpaper
RES = osp.abspath(osp.join(EVAL, "..", "results"))             # results
TYPED = osp.join(EVAL, "typed")


def src(name):
    return osp.join(RES, name)


SOURCES = {
    "typed_cells": src("eval_leaderboard_v2_typed_cells.csv"),
    "typed_types": src("eval_leaderboard_v2_typed_types.csv"),
    "prereg": src("eval_leaderboard_v2_prereg.json"),
    "msv1": src("retired_msv1/eval_typed_controls_msv1.csv"),
    "msv2_iter1": osp.join(TYPED, "msv2_work", "select_iter1.json"),
    "msv2_iter2": osp.join(TYPED, "msv2_work", "select_iter2.json"),
    "k_summary": src("eval_kernels_summary.csv"),
    "k_verbatim": src("eval_kernels_ubgold_verbatim.csv"),
    "k_published": src("eval_kernels_published.csv"),
    "leaderboard": src("eval_leaderboard_v2_leaderboard.csv"),
    "per_dataset": src("eval_leaderboard_v2_per_dataset.csv"),
    "inflation": src("eval_leaderboard_v2_inflation.csv"),
    "inflation_pooled": src("eval_leaderboard_v2_inflation_pooled.csv"),
    "kendall": src("eval_leaderboard_v2_kendall.csv"),
    "sensitivity": src("eval_leaderboard_v2_sensitivity.csv"),
    "flags_method": src("eval_leaderboard_v2_flags_method.csv"),
    "noise": src("eval_leaderboard_v2_noise_estimate.csv"),
    "selectors": src("eval_leaderboard_v2_selectors.csv"),
    "version": src("eval_leaderboard_v2_version.csv"),
    "a33_orient": src("eval_a33_orientation.csv"),
    "a33_orient_pooled": src("eval_a33_orientation_pooled.csv"),
    "a33_orient_refs": src("eval_a33_orientation_refs.csv"),
    "a33_abl_ub9": src("eval_a33_ablation_ub9.csv"),
    "a33_abl_typed": src("eval_a33_ablation_typed.csv"),
}

# ---------------------------------------------------------------------------------------------- names and order
NAMES = {"GraphCL_IF": "GraphCL-IF", "GraphCL_OCSVM": "GraphCL-OCSVM", "InfoGraph_IF": "InfoGraph-IF",
         "InfoGraph_OCSVM": "InfoGraph-OCSVM", "SUL-GUL": "SUL/GUL", "graphlet-count+LW": "Graphlet count + LW",
         "graphlet-count+iForest": "Graphlet count + IF", "size-only": "Size only",
         "WL-hist+iForest (ctrl)": "WL histogram + IF"}
GROUP_NAME = {"frozen12": "UB-GOLD", "E5": "recent", "E4": "recent", "kernel": "kernel", "reference": "reference"}
GROUP_ORDER = ["frozen12", "recent", "kernel", "reference"]
CONFIRMATORY = ["GLADC", "CVTGAD", "GOOD-D", "GraphDE", "GraphCL_IF", "GraphCL_OCSVM", "InfoGraph_IF",
                "InfoGraph_OCSVM"]
DISCOVERY = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD"]
RECENT = ["MUSE", "LGKDE", "GLADPro", "GLADMamba", "SEGO", "GLASS", "ProtoGLAD", "HC-GLAD", "SUL-GUL"]
UB9 = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "NCI1", "PROTEINS_full", "IMDB-BINARY", "REDDIT-BINARY"]
SEC = ["DD", "COLLAB", "Tox21_p53", "Tox21_MMP", "MUTAG"]
DS_SHORT = {"ENZYMES": "ENZ.", "PROTEINS_full": "PROT.", "IMDB-BINARY": "IMDB-B", "REDDIT-BINARY": "RDT-B",
            "Tox21_p53": "p53", "Tox21_MMP": "MMP"}
KERNELS = ["WL-IF", "WL-OCSVM", "PK-IF", "PK-OCSVM"]


def name(m):
    return NAMES.get(m, m)


def ds(d):
    return DS_SHORT.get(d, d)


def isnum(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and not pd.isna(x)


def rnd(x, nd):
    """Round half up on the decimal string of x. The stats CSVs store 4 decimals, so this reproduces the rounding of
    the unrounded values that REPORT.md prints (e.g. 5.68505 -> CSV 5.6850 -> 5.69), which float formatting
    (round-half-even on the binary value) would print as 5.68."""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def f1(x):
    return rnd(x, 1) if isnum(x) else "--"


def f2(x):
    return rnd(x, 2) if isnum(x) else "--"


def sgn2(x):
    if not isnum(x):
        return "--"
    s = rnd(x, 2)
    return "$-$" + s[1:] if s.startswith("-") else "+" + s


def f2_full(x):
    """For means computed here from full-precision per-fold values: plain float formatting, as REPORT.md."""
    return f"{x:.2f}" if isnum(x) else "--"


def neg(s):
    """Typeset a leading minus sign of an already formatted number as a math minus."""
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def ci(lo, hi, fmt=f2):
    return f"[{neg(fmt(lo))}, {neg(fmt(hi))}]" if isnum(lo) and isnum(hi) else "--"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(stem, body, used):
    lines = [f"% {stem}.tex: generated by tables/make_tables.py; do not edit by hand.", "% sources (sha256):"]
    for k in used:
        p = SOURCES[k]
        rel = osp.relpath(p, osp.join(EVAL, ".."))
        lines.append(f"%   {rel}  {sha256(p) if osp.exists(p) else 'MISSING'}")
    with open(osp.join(HERE, f"{stem}.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n" + body.rstrip() + "\n")
    print(f"wrote tables/{stem}.tex")


def check(value, expected, tol, what):
    if not (isnum(value) and abs(value - expected) <= tol):
        raise SystemExit(f"cross-check failed: {what}: got {value}, expected {expected}")


def lb_frame():
    lb = pd.read_csv(SOURCES["leaderboard"])
    return lb[(lb.set == "UB9") & (lb.pool == "all")]


def method_order():
    """Fixed row order for the per-method tables: group, then UB9 VAL-NORMAL mean (descending)."""
    lb = lb_frame()
    v = lb[lb.protocol == "valnormal"].copy()
    v["grp"] = v.stratum.map(lambda s: "recent" if s in ("E5", "E4") else s)
    v["g"] = v.grp.map(GROUP_ORDER.index)
    v = v.sort_values(["g", "mean_auroc"], ascending=[True, False])
    return list(zip(v.method, v.stratum))


# ============================================================================================ Appendix D (typed)
TYPED_COLS = [("GX", "T1a", "matched"), ("GX", "T3a", "matched"), ("GX", "T3b", "matched"),
              ("MS", "MS1", "matched"), ("MS", "MS2", "matched"), ("MS", "MS3", "matched"),
              ("MS", "MS1", "changing"), ("MS", "MS2", "changing"), ("MS", "MS3", "changing")]
TYPED_HEAD = (r"& \multicolumn{3}{c}{GX, size-preserving} & \multicolumn{3}{c}{MS-v2, size-preserving} "
              r"& \multicolumn{3}{c}{MS-v2, size-changing} \\" "\n"
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" "\n"
              r"Detector & T1a & T3a & T3b & MS1 & MS2 & MS3 & MS1 & MS2 & MS3 \\")


def typed_rows(cells, methods, cell_fmt, compact_missing=False):
    out = []
    missing = [m for m in methods if cells[cells.method == m].n_seeds.fillna(0).eq(0).all()]
    if compact_missing and len(missing) == len(methods):
        return [r"\multicolumn{10}{c}{\tbd{intervals for the nine recent methods after their typed runs}} \\"]
    for m in methods:
        c = cells[cells.method == m]
        if c.empty or (c.n_seeds.fillna(0) == 0).all():
            out.append(f"{name(m)} & \\multicolumn{{9}}{{c}}{{\\tbd{{typed runs in progress}}}} \\\\")
            continue
        vals = []
        for g, t, v in TYPED_COLS:
            r = c[(c.generator == g) & (c.anomaly_type == t) & (c.size_variant == v)]
            vals.append(cell_fmt(r.iloc[0]) if len(r) and r.iloc[0].n_seeds > 0 else "--")
        out.append(f"{name(m)} & " + " & ".join(vals) + r" \\")
    return out


def typed_mean_cell(r):
    s = f1(r.valnormal_mean)
    if not bool(r.complete):
        s += r"$^\dagger$"
    return r"\textbf{" + s + "}" if str(r.blind) == "False" else s


def typed_ci_cell(r):
    return f"{f1(r.seed_ci_lo)}--{f1(r.seed_ci_hi)}" if isnum(r.seed_ci_lo) else "--"


def typed_tables():
    cells = pd.read_csv(SOURCES["typed_cells"])
    types = pd.read_csv(SOURCES["typed_types"])
    for stem, fmt, with_controls in (("typed_cells", typed_mean_cell, True), ("typed_ci", typed_ci_cell, False)):
        L = [r"\begin{tabular}{l*{9}{c}}", r"\toprule", TYPED_HEAD, r"\midrule",
             r"\multicolumn{10}{l}{\emph{Confirmatory detectors (enter the decision)}} \\"]
        L += typed_rows(cells, CONFIRMATORY, fmt)
        L += [r"\midrule", r"\multicolumn{10}{l}{\emph{Discovery detectors (hypothesis formed on them; descriptive)}} \\"]
        L += typed_rows(cells, DISCOVERY, fmt)
        L += [r"\midrule", r"\multicolumn{10}{l}{\emph{Recent methods (descriptive)}} \\"]
        L += typed_rows(cells, RECENT, fmt, compact_missing=not with_controls)
        if with_controls:
            L += [r"\midrule", r"\multicolumn{10}{l}{\emph{Controls (training-free; decide validity)}} \\"]
            for col, lab in (("graphlet_count_lw", "Graphlet count + LW"), ("size_only", "Size only"),
                             ("edge_label_pair_lw", "Edge-label pair + LW")):
                vals = []
                for g, t, v in TYPED_COLS:
                    r = types[(types.generator == g) & (types.anomaly_type == t) & (types.size_variant == v)]
                    vals.append(f1(r.iloc[0][col]) if len(r) else "--")
                L.append(f"{lab} & " + " & ".join(vals) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        write(stem, "\n".join(L), ["typed_cells", "typed_types"])


def typed_validity():
    types = pd.read_csv(SOURCES["typed_types"])
    it1, it2 = (json.load(open(SOURCES[k])) for k in ("msv2_iter1", "msv2_iter2"))
    final = {t: (f, k) for t, f, k in it2["final"]}
    null = {}
    for t, (f, k) in final.items():
        for row in it1["table"] + it2["table"]:
            if row["pool"] == it2["pool"] and row["family"] == f and str(row["k"]) == str(k):
                null[t] = row["null_gc"]
    L = [r"\begin{tabular}{llcccccccl}", r"\toprule",
         r"Generator & Type & Size & Graphlet & Size & Edge-label & Null edit & Valid & In & Anomalies per seed \\",
         r" & & & count + LW & only & pair + LW & (graphlet) & & decision & (seeds 0--4) \\", r"\midrule"]
    for _, r in types.iterrows():
        gen = "GX" if r.generator == "GX" else "MS-v2"
        var = "preserving" if r.size_variant == "matched" else "changing"
        nl = f1(null.get(r.anomaly_type)) if (gen == "MS-v2" and var == "preserving") else "--"
        if gen == "GX":
            counts = "250 each"
        else:
            counts = ", ".join(str(r.counts_per_seed).split("|"))
        yn = lambda b: "yes" if str(b) == "True" else "no"
        L.append(f"{gen} & {r.anomaly_type} & {var} & {f1(r.graphlet_count_lw)} & {f1(r.size_only)} & "
                 f"{f1(r.edge_label_pair_lw)} & {nl} & {yn(r.valid)} & {yn(r.in_decision)} & {counts} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("typed_validity", "\n".join(L), ["typed_types", "msv2_iter1", "msv2_iter2"])


def typed_decision():
    g = json.load(open(SOURCES["prereg"]))["G2c"]
    rows = [("Confirmatory, GX features F2 (primary)", 8, g["decision"]),
            ("Confirmatory, GX features F1 (sensitivity)", 8, g["sensitivity_F1"]),
            ("Discovery (descriptive)", 4, g["discovery_methods_descriptive"]["old4 (discovery)"])]
    check(g["decision"]["frac_blind_pooled"], 0.8125, 1e-9, "G2c pooled blind fraction (REPORT: 0.81)")
    L = [r"\begin{tabular}{lcccccl}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Share of pairs blind} & Detectors blind on & \\",
         r"\cmidrule(lr){4-5}",
         r"Group & Detectors & Pairs & GX and MS-v2 & MS-v2 & most MS-v2 types & Verdict \\",
         r"\midrule"]
    for lab, n_m, d in rows:
        L.append(f"{lab} & {n_m} & {d['n_pairs']} & {d['frac_blind_pooled']:.2f} & {d['frac_blind_ms']:.2f} & "
                 f"{d['methods_blind_on_majority_of_ms']} & {d['verdict'].lower()} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("typed_decision", "\n".join(L), ["prereg"])


def typed_msv1():
    d = pd.read_csv(SOURCES["msv1"])
    d = d[d.dataset == "MS"].drop_duplicates(["anomaly_type", "size_variant"])
    L = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Type (v1) & Size & Graphlet count + LW & Size only & Edge-label pair + LW & Valid & Min.\ anomalies per seed \\",
         r"\midrule"]
    for t in ("MS1", "MS2", "MS3"):
        for v in ("matched", "changing"):
            r = d[(d.anomaly_type == t) & (d.size_variant == v)].iloc[0]
            L.append(f"{t} & {'preserving' if v == 'matched' else v} & {f1(r.type_gc_lw_mean)} & {f1(r.type_size_lw_mean)} & {f1(r.type_elp_lw_mean)} & "
                     f"{'yes' if bool(r.type_valid) else 'no'} & {int(r.type_min_anomalies)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("typed_msv1", "\n".join(L), ["msv1"])


def typed_msv2_design():
    it1, it2 = (json.load(open(SOURCES[k])) for k in ("msv2_iter1", "msv2_iter2"))
    chosen = {(f, str(k)) for _, f, k in it2["final"]}
    korder = {"1": 0, "2": 1, "all": 2}
    L = [r"\begin{tabular}{clcc*{8}{c}l}", r"\toprule",
         r"It. & Pool & Fam. & $k$ & Min.\ yield & Edits & Graphlet count + LW & Size & Edge-label & Null & "
         r"\multicolumn{2}{c}{Size-changing} & Pass \\",
         r"\cmidrule(lr){11-12}",
         r" & & & & m / sc & (mean) & mean [min--max] & only & pair + LW & edit & graphlet & size & \\", r"\midrule"]
    first = True
    for it, js in ((1, it1), (2, it2)):
        rows = sorted(js["table"], key=lambda r: (["P0", "P56"].index(r["pool"]), r["family"], korder[str(r["k"])]))
        if not first:
            L.append(r"\midrule")
        first = False
        prev_pool = None
        for r in rows:
            if prev_pool is not None and r["pool"] != prev_pool:
                L.append(r"\cmidrule(lr){2-13}")
            prev_pool = r["pool"]
            k = str(r["k"])
            sel = (r["pool"] == it2["pool"] and (r["family"], k) in chosen)
            if r["PASS"]:
                ps = r"\textbf{yes, selected}" if sel else "yes"
            else:
                ps = "no (yield)" if not r["feasible"] else "no"
            L.append(f"{it} & {r['pool']} & {r['family']} & {k} & {r['min_matched']} / {r['min_sizechg']} & "
                     f"{r['mean_edits']:.1f} & {f1(r['gc'])} [{f1(r['gc_min'])}--{f1(r['gc_max'])}] & {f1(r['size'])} & "
                     f"{f1(r['elp'])} & {f1(r['null_gc'])} & {f1(r['sc_gc'])} & {f1(r['sc_size'])} & {ps} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("typed_msv2_design", "\n".join(L), ["msv2_iter1", "msv2_iter2"])


# ============================================================================================ Appendix E (kernels)
def kernel_tables():
    s = pd.read_csv(SOURCES["k_summary"])
    lb = lb_frame()
    for stem, dsets in (("kernels_ub9", UB9), ("kernels_secondary", SEC[:4])):
        L = [r"\begin{tabular}{llc" + "c" * len(dsets) + "c}", r"\toprule",
             "Detector & Fit & HP & " + " & ".join(ds(d) for d in dsets) + r" & Mean \\", r"\midrule"]
        for i, m in enumerate(KERNELS):
            if i:
                L.append(r"\midrule")
            for fit, mm in (("train", m), ("test", m + "@testfit")):
                for hp in ("default", "tuned"):
                    vals, xs = [], []
                    for d in dsets:
                        r = s[(s.method == mm) & (s.dataset == d) & (s.protocol == hp)]
                        if len(r) != 1 or int(r.iloc[0].n_folds) != 5:
                            raise SystemExit(f"kernel summary: missing {mm} {d} {hp}")
                        x = float(r.iloc[0].auroc_mean)
                        xs.append(x)
                        vals.append(f1(x) + (r"$^\S$" if bool(r.iloc[0].kernel_constant) else ""))
                    mean = sum(xs) / len(xs)
                    if fit == "train" and dsets is UB9:      # same number as the leaderboard (kernel rows)
                        proto = "valnormal" if hp == "tuned" else "hpfree_valnormal"
                        check(mean, float(lb[(lb.method == m) & (lb.protocol == proto)].mean_auroc.iloc[0]), 0.006,
                              f"kernel UB9 mean {m} {hp}")
                    L.append(f"{m} & {fit} & {hp} & " + " & ".join(vals) + f" & {f1(mean)} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        write(stem, "\n".join(L), ["k_summary", "leaderboard"])


def kernel_verbatim():
    v = pd.read_csv(SOURCES["k_verbatim"])
    if not (v.kernel_maxabs_vs_ours == 0).all():
        raise SystemExit("verbatim kernel matrices differ from ours")
    g = v.groupby("dataset").agg(folds=("fold", "count"), rel=("auroc_released", "mean"),
                                 ali=("auroc_aligned", "mean"), mis=("frac_label_mismatch", "mean"))
    order = UB9 + SEC[:4]
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Dataset & Folds & Released pairing & Aligned pairing & Labels out of place \\", r"\midrule"]
    for d in order:
        r = g.loc[d]
        L.append(f"{ds(d)} & {int(r.folds)} & {f2_full(r.rel)} & {f2_full(r.ali)} & {f2_full(r.mis)} \\\\")
    rel, ali = v.auroc_released.mean(), v.auroc_aligned.mean()
    check(rel, 49.83, 0.005, "verbatim released 65-fold mean (PAPER_NOTES U5: 49.83)")
    check(ali, 50.48, 0.005, "verbatim aligned 65-fold mean (kernels/NOTES.md: 50.48)")
    L += [r"\midrule", f"All folds & {len(v)} & {f2_full(rel)} & {f2_full(ali)} & -- \\\\", r"\bottomrule", r"\end{tabular}"]
    write("kernels_verbatim", "\n".join(L), ["k_verbatim"])


def kernel_published():
    p = pd.read_csv(SOURCES["k_published"])
    order = UB9 + SEC[:4]
    L = [r"\begin{tabular}{lc" + "c" * 4 + "}", r"\toprule",
         r"Dataset & Chance s.d. & " + " & ".join(KERNELS) + r" \\", r"\midrule"]
    n_in = {"IF": 0, "OCSVM": 0}
    for d in order:
        vals, sd = [], None
        for m in KERNELS:
            r = p[(p.dataset == d) & (p.method == m)].iloc[0]
            sd = r.chance_sd_of_5fold_mean
            z = float(r.z_vs_chance)
            cell = f"{f2(r.published_auroc)} ({neg(f'{z:.1f}')})"
            if abs(z) > 3:
                cell = r"\textbf{" + cell + "}"
            else:
                n_in["IF" if m.endswith("IF") else "OCSVM"] += 1
            vals.append(cell)
        L.append(f"{ds(d)} & {f2(sd)} & " + " & ".join(vals) + r" \\")
    if n_in != {"IF": 21, "OCSVM": 24}:
        raise SystemExit(f"published-vs-chance counts changed: {n_in} (REPORT: IF 21/26, OCSVM 24/26)")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("kernels_published", "\n".join(L), ["k_published"])


# ============================================================================================ Appendix F (extra)
def leaderboard_full():
    lb = lb_frame()
    order = method_order()
    protos = ["oracle", "oracle_ubgold", "last", "valnormal", "hpfree_valnormal"]
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & \oracle{} & \oracle{} & \last{} & \valnormal{} & \valnormal{} & \hpfree{} \\",
         r"Detector & Group & own rule & UB-GOLD rule & & & 95\% CI & \\", r"\midrule"]
    prev = None
    for m, st in order:
        grp = GROUP_NAME[st]
        if prev is not None and grp != prev:
            L.append(r"\midrule")
        prev = grp
        cells = []
        for p_ in protos:
            r = lb[(lb.method == m) & (lb.protocol == p_)].iloc[0]
            c = f"{f1(r.mean_auroc)} ({int(r.rank_in_pool)})"
            cells.append(c)
            if p_ == "valnormal":
                cells.append(ci(r.ds_ci_lo, r.ds_ci_hi, f1))
        L.append(f"{name(m)} & {grp} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_leaderboard", "\n".join(L), ["leaderboard"])


def constant_kernel_cells():
    """(method, dataset) cells where the PK kernel matrix is constant (AUROC 50 by construction)."""
    k = pd.read_csv(SOURCES["k_summary"])
    k = k[k.kernel_constant.astype(str) == "True"]
    return set(zip(k.method, k.dataset))


def pd_cell(r, m, d, const):
    if not len(r):
        return "--"
    s = f1(r.iloc[0].auroc_mean)
    if r.iloc[0].n_folds != 5:
        s += r"$^\ddagger$"
    if (m, d) in const:
        s += r"$^\S$"
    return s


def per_dataset_tables():
    pdx = pd.read_csv(SOURCES["per_dataset"])
    lb = lb_frame()
    order = method_order()
    const = constant_kernel_cells()
    for proto, stem in (("oracle", "extra_ub9_oracle"), ("last", "extra_ub9_last"),
                        ("valnormal", "extra_ub9_valnormal"), ("hpfree_valnormal", "extra_ub9_hpfree")):
        x = pdx[(pdx.set == "UB9") & (pdx.protocol == proto)]
        L = [r"\begin{tabular}{l" + "c" * (len(UB9) + 1) + "}", r"\toprule",
             "Detector & " + " & ".join(ds(d) for d in UB9) + r" & Mean \\", r"\midrule"]
        prev = None
        for m, st in order:
            grp = GROUP_NAME[st]
            if prev is not None and grp != prev:
                L.append(r"\midrule")
            prev = grp
            vals = []
            for d in UB9:
                vals.append(pd_cell(x[(x.method == m) & (x.dataset == d)], m, d, const))
            mean = lb[(lb.method == m) & (lb.protocol == proto)].mean_auroc
            L.append(f"{name(m)} & " + " & ".join(vals) + f" & {f1(mean.iloc[0]) if len(mean) else '--'} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        write(stem, "\n".join(L), ["per_dataset", "leaderboard", "k_summary"])
    for pair, stem in ((("oracle", "valnormal"), "extra_sec_a"), (("last", "hpfree_valnormal"), "extra_sec_b")):
        head = {"oracle": r"\oracle{} (own rule)", "valnormal": r"\valnormal{}", "last": r"\last{}",
                "hpfree_valnormal": r"\hpfree{}"}
        L = [r"\begin{tabular}{l" + "c" * (2 * len(SEC)) + "}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{{len(SEC)}}}{{c}}{{{head[p_]}}}" for p_ in pair) + r" \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
             "Detector & " + " & ".join(ds(d) for d in SEC * 2) + r" \\", r"\midrule"]
        prev = None
        for m, st in order:
            grp = GROUP_NAME[st]
            if prev is not None and grp != prev:
                L.append(r"\midrule")
            prev = grp
            vals = []
            for p_ in pair:
                x = pdx[(pdx.set == "secondary") & (pdx.protocol == p_)]
                for d in SEC:
                    vals.append(pd_cell(x[(x.method == m) & (x.dataset == d)], m, d, const))
            L.append(f"{name(m)} & " + " & ".join(vals) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        write(stem, "\n".join(L), ["per_dataset", "k_summary"])


def inflation_table():
    inf = pd.read_csv(SOURCES["inflation"])
    inf = inf[inf.set == "UB9"]
    pool = pd.read_csv(SOURCES["inflation_pooled"])
    pool = pool[pool.set == "UB9"]
    order = [(m, st) for m, st in method_order() if st in ("frozen12", "E5", "E4")]
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{\oracle{} $-$ \valnormal{}} & \multicolumn{2}{c}{\oracle{} $-$ \last{}} & "
         r"Own rule \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Detector & Mean [95\% CI] & $\geq 2$ & Mean [95\% CI] & $\geq 2$ & $-$ \valnormal{} \\", r"\midrule"]
    prev = None
    for m, st in order:
        grp = GROUP_NAME[st]
        if prev is not None and grp != prev:
            L.append(r"\midrule")
        prev = grp
        x = inf[inf.method == m]
        if st == "frozen12":
            vn = x[x.comparison == "checkpoint: ORACLE - VAL-NORMAL"].iloc[0]
            la = x[x.comparison == "checkpoint: ORACLE - LAST"].iloc[0]
            own = "--"
        else:
            vn = x[x.comparison == "checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL"].iloc[0]
            la = x[x.comparison == "checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST"].iloc[0]
            o = x[x.comparison == "checkpoint: ORACLE - VAL-NORMAL"].iloc[0]
            own = sgn2(o.mean_diff)
        if isinstance(vn.excluded_from_pooled, str) and vn.excluded_from_pooled.startswith("A20"):
            L.append(f"{name(m)} & \\multicolumn{{5}}{{c}}{{no checkpoints, one model per run, "
                     f"excluded from the pooled rows}} \\\\")
            continue
        L.append(f"{name(m)} & {sgn2(vn.mean_diff)} {ci(vn.ds_ci_lo, vn.ds_ci_hi)} & {int(vn.n_datasets_diff_ge2)}/9 & "
                 f"{sgn2(la.mean_diff)} {ci(la.ds_ci_lo, la.ds_ci_hi)} & {int(la.n_datasets_diff_ge2)}/9 & {own} \\\\")
    L.append(r"\midrule")
    pooled = [("UB-GOLD detectors", "frozen12", "checkpoint: ORACLE - VAL-NORMAL", "checkpoint: ORACLE - LAST"),
              ("Recent methods", "recent(E5+E4)", "checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL",
               "checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST"),
              ("All checkpointed", "frozen12+E5+E4", "checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL",
               "checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST")]
    for lab, pl, cv, cl in pooled:
        a = pool[(pool.pool == pl) & (pool.comparison == cv)].iloc[0]
        b = pool[(pool.pool == pl) & (pool.comparison == cl)].iloc[0]
        L.append(f"{lab} (pooled) & {sgn2(a.mean_over_methods)} {ci(a.ds_ci_lo, a.ds_ci_hi)} & "
                 f"{int(a.n_methods_ge2)}/{int(a.n_methods)} & {sgn2(b.mean_over_methods)} {ci(b.ds_ci_lo, b.ds_ci_hi)} & "
                 f"{int(b.n_methods_ge2)}/{int(b.n_methods)} & -- \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_inflation", "\n".join(L), ["inflation", "inflation_pooled", "leaderboard"])


def kendall_table():
    k = pd.read_csv(SOURCES["kendall"])
    k = k[k.set == "UB9"]
    pools = [("frozen12", "UB-GOLD (12)"), ("recent(E5+E4)", "Recent (9)"), ("frozen12+E5+E4", "All detectors (21)"),
             ("all", "All rows (30)")]
    head = {"oracle": r"\oracle{}", "oracle_ubgold": r"\oracle{}$_\text{UB-GOLD}$", "valnormal": r"\valnormal{}",
            "last": r"\last{}", "hpfree_valnormal": r"\hpfree{}", "hpfree_oracle": r"\oracle{}$_\text{defaults}$"}
    pairs = [("oracle", "valnormal"), ("oracle", "last"), ("valnormal", "last"), ("oracle_ubgold", "valnormal"),
             ("oracle", "hpfree_valnormal"), ("valnormal", "hpfree_valnormal"), ("hpfree_oracle", "hpfree_valnormal")]
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Rows ranked & Protocols compared & $\tau_b$ & Fold-bootstrap CI & Dataset-bootstrap CI & Top 3 changed \\",
         r"\midrule"]
    for i, (pl, lab) in enumerate(pools):
        if i:
            L.append(r"\midrule")
        for a, b in pairs:
            r = k[(k.pool == pl) & (k.protocol_a == a) & (k.protocol_b == b)]
            if not len(r):
                continue
            r = r.iloc[0]
            L.append(f"{lab} & {head[a]} vs {head[b]} & {f2(r.tau_b)} & {ci(r.fold_ci_lo, r.fold_ci_hi)} & "
                     f"{ci(r.ds_ci_lo, r.ds_ci_hi)} & {'yes' if bool(r.top3_changed) else 'no'} \\\\")
            lab = ""
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_kendall", "\n".join(L), ["kendall"])


def sensitivity_table():
    s = pd.read_csv(SOURCES["sensitivity"])
    pool = pd.read_csv(SOURCES["inflation_pooled"])
    pool = pool[(pool.set == "UB9") & (pool.pool == "frozen12")]
    full = {"G1/VAL-NORMAL": pool[pool.comparison == "checkpoint: ORACLE - VAL-NORMAL"].iloc[0],
            "G1/LAST": pool[pool.comparison == "checkpoint: ORACLE - LAST"].iloc[0],
            "G2b HP leak": pool[pool.comparison == "HP leak: VAL-NORMAL - HP-FREE"].iloc[0]}
    labels = {"G1/VAL-NORMAL": r"Inflation, \oracle{} $-$ \valnormal{}", "G1/LAST": r"Inflation, \oracle{} $-$ \last{}",
              "G2b HP leak": r"Hyperparameter leakage"}
    variants = ["drop nonfinite_selected", "drop tie_collapsed", "drop any flag"]
    nrun = {v: int(s[s.variant == v].n_runs_dropped.iloc[0]) for v in variants}
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Quantity (12 UB-GOLD detectors) & All runs & Drop non-finite & Drop tie-collapsed & Drop either \\",
         f" & & ({nrun[variants[0]]} runs) & ({nrun[variants[1]]} runs) & ({nrun[variants[2]]} runs) \\\\",
         r"\midrule"]
    for q in ("G1/VAL-NORMAL", "G1/LAST", "G2b HP leak"):
        f = full[q]
        cells = [f"{sgn2(f.mean_over_methods)} {ci(f.ds_ci_lo, f.ds_ci_hi)}"]
        for v in variants:
            r = s[(s.variant == v) & (s.quantity == q)].iloc[0]
            cells.append(f"{sgn2(r.mean_over_methods)} {ci(r.ds_ci_lo, r.ds_ci_hi)}")
        L.append(f"{labels[q]} & " + " & ".join(cells) + r" \\")
        if q.startswith("G1"):
            cnt = [f"{int(f.n_methods_ge2)}/12"] + [f"{int(s[(s.variant == v) & (s.quantity == q)].iloc[0].n_methods_ge2)}/12"
                                                   for v in variants]
            L.append(r"\quad detectors with $\geq 2$ points & " + " & ".join(cnt) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_sensitivity", "\n".join(L), ["sensitivity", "inflation_pooled"])


def flags_table():
    f = pd.read_csv(SOURCES["flags_method"])
    f = f[(f.runs_nonfinite_selected > 0) | (f.runs_tie_collapsed > 0) | (f.diverged_runs > 0)]
    order = [m for m, _ in method_order()]
    f = f.assign(o=f.method.map(order.index)).sort_values("o")
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r" & & Non-finite at & Tie- & Diverged \\",
         r"Detector & Runs & selected point & collapsed & (excluded) \\", r"\midrule"]
    for _, r in f.iterrows():
        L.append(f"{name(r.method)} & {int(r.runs)} & {int(r.runs_nonfinite_selected)} & {int(r.runs_tie_collapsed)} & "
                 f"{int(r.diverged_runs)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_flags", "\n".join(L), ["flags_method", "leaderboard"])


def noise_table():
    n = pd.read_csv(SOURCES["noise"])
    L = [r"\begin{tabular}{lccccccccc}", r"\toprule",
         r" & & & & \multicolumn{3}{c}{$|\Delta|$ at \last{}} & \multicolumn{3}{c}{$|\Delta|$ at \valnormal{}} \\",
         r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r"Method & Pairs & Bit-identical & Differ & Median & 90th pct. & Max & Median & 90th pct. & Max \\",
         r"\midrule"]
    for _, r in n.iterrows():
        L.append(f"{name(r.method)} & {int(r.pairs)} & {int(r.bit_identical)} & {int(r.differing)} & "
                 f"{f2(r.last_absdiff_median)} & {f2(r.last_absdiff_p90)} & {f2(r.last_absdiff_max)} & "
                 f"{f2(r.valnormal_absdiff_median)} & {f2(r.valnormal_absdiff_p90)} & {f2(r.valnormal_absdiff_max)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_noise", "\n".join(L), ["noise"])


def selectors_table():
    s = pd.read_csv(SOURCES["selectors"])
    crit = [("oracle", r"\oracle{} (test labels, early stop)"), ("oracle_nostop", r"\oracle{} without early stop"),
            ("valnormal", r"\valnormal{} (reference)"), ("last", r"\last{}"), ("pseudo", "Pseudo-anomaly AUROC"),
            ("pseudo_ensemble", "Pseudo-anomaly top-3 ensemble"), ("ckpt_ensemble", "Checkpoint ensemble"),
            ("consensus", "Consensus checkpoint"), ("aucp", "AUCp (transductive)")]
    meths = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD", "POOLED(4)"]
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Selection rule & SIGNET & OCGTL & OCGIN & GLocalKD$^\ast$ & Mean of 4 & $-$ \valnormal{} \\", r"\midrule"]
    for c, lab in crit:
        vals = [f1(s[(s.method == m) & (s.criterion == c)].iloc[0].mean_auroc) for m in meths]
        gap = s[(s.method == "POOLED(4)") & (s.criterion == c)].iloc[0].minus_valnormal
        L.append(f"{lab} & " + " & ".join(vals) + f" & {'0' if c == 'valnormal' else sgn2(gap)} \\\\")
        if c == "valnormal":
            L.append(r"\midrule")
        if c == "oracle_nostop":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_selectors", "\n".join(L), ["selectors"])


def version_table():
    p = SOURCES["version"]
    v = pd.read_csv(p) if osp.exists(p) and osp.getsize(p) > 2 else pd.DataFrame()
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Detector & Quantity & c940457 & a96137b & Paired difference & 95\% CI (dataset bootstrap) \\", r"\midrule"]
    if v.empty:
        for m in ("GLocalKD", "GLADC"):
            for q in (r"\oracle{}", r"\last{}", r"\valnormal{}", r"\oracle{} $-$ \valnormal{}", r"\oracle{} $-$ \last{}"):
                L.append(f"{m} & {q} & \\tbd{{}} & \\tbd{{}} & \\tbd{{}} & \\tbd{{}} \\\\")
            if m == "GLocalKD":
                L.append(r"\midrule")
    else:
        qlab = {"oracle: a96137b - c940457": r"\oracle{}", "last: a96137b - c940457": r"\last{}",
                "valnormal: a96137b - c940457": r"\valnormal{}",
                "inflation ORACLE-VALNORMAL: a96137b vs c940457": r"\oracle{} $-$ \valnormal{}",
                "inflation ORACLE-LAST: a96137b vs c940457": r"\oracle{} $-$ \last{}"}
        for i, m in enumerate(("GLocalKD", "GLADC")):
            if i:
                L.append(r"\midrule")
            for q, lab in qlab.items():
                r = v[(v.method == m) & (v.quantity == q)]
                if not len(r):
                    L.append(f"{m} & {lab} & \\multicolumn{{4}}{{c}}{{incomplete}} \\\\")
                    continue
                r = r.iloc[0]
                if str(r.get("complete")) == "False":
                    L.append(f"{m} & {lab} & \\multicolumn{{4}}{{c}}{{incomplete ({int(r.get('n_paired_folds') or 0)} "
                             f"paired folds)}} \\\\")
                    continue
                if q.startswith("inflation"):
                    a0, a1, cis = r.get("inflation_c940457"), r.get("inflation_a96137b"), "--"
                else:
                    a0, a1 = r.get("mean_c940457"), r.get("mean_a96137b")
                    cis = ci(r.get("ds_ci_lo"), r.get("ds_ci_hi"))
                L.append(f"{m} & {lab} & {f2(a0)} & {f2(a1)} & {sgn2(r.get('mean_diff'))} & {cis} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_version", "\n".join(L), ["version"])


# ============================================================================================ A33 follow-ups (A35)
MAIN_CONV, FLIP_CONV = "main (class 0 anomalous)", "flipped (class 1 anomalous)"


def bold60(x):
    return f"\\textbf{{{f1(x)}}}" if isnum(x) and x > 60 else f1(x)


def a33_orientation_table():
    o = pd.read_csv(SOURCES["a33_orient"])
    pl = pd.read_csv(SOURCES["a33_orient_pooled"])
    cols = ["ORACLE", "VAL-NORMAL", "LAST", "ORACLE-VAL-NORMAL", "ORACLE-LAST"]
    L = [r"\begin{tabular}{lccccc@{\hspace{10pt}}ccccc}", r"\toprule",
         r" & \multicolumn{5}{c}{UB-GOLD's anomaly class (class 0)} & \multicolumn{5}{c}{Flipped (class 1)} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
         r"Detector & \oracle{} & \valnormal{} & \last{} & $\Delta_{\mathrm{VN}}$ & $\Delta_{\mathrm{L}}$ & "
         r"\oracle{} & \valnormal{} & \last{} & $\Delta_{\mathrm{VN}}$ & $\Delta_{\mathrm{L}}$ \\", r"\midrule"]
    for m in DISCOVERY + CONFIRMATORY:
        a = o[(o.convention == MAIN_CONV) & (o.detector == m)].iloc[0]
        b = o[(o.convention == FLIP_CONV) & (o.detector == m)].iloc[0]
        L.append(" & ".join([name(m)] + [f2_full(a[c]) for c in cols] + [f2_full(b[c]) for c in cols]) + r" \\")
    L.append(r"\midrule")
    def pooled(conv, gap):
        return pl[(pl.convention == conv) & (pl.gap == gap)].iloc[0]
    for lab, key in (("Mean of 12", "mean"), ("95\\% CI, fold bootstrap", "fold"), ("95\\% CI, dataset bootstrap", "ds"),
                     ("Detectors $\\geq 2$ points", "n")):
        cells = []
        for conv in (MAIN_CONV, FLIP_CONV):
            cells += ["", "", ""]
            for gap in ("ORACLE-VAL-NORMAL", "ORACLE-LAST"):
                r = pooled(conv, gap)
                cells.append(f2_full(r["mean"]) if key == "mean" else
                             ci(r["fold_ci_lo"], r["fold_ci_hi"], f2_full) if key == "fold" else
                             ci(r["ds_ci_lo"], r["ds_ci_hi"], f2_full) if key == "ds" else f"{int(r['n_ge_2'])}/12")
        L.append(" & ".join([lab] + cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_orientation", "\n".join(L), ["a33_orient", "a33_orient_pooled"])
    r = pd.read_csv(SOURCES["a33_orient_refs"])
    L = [r"\begin{tabular}{lccc@{\hspace{10pt}}ccc}", r"\toprule",
         r" & \multicolumn{3}{c}{UB-GOLD's anomaly class} & \multicolumn{3}{c}{Flipped} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Reference & BZR & COX2 & PROTEINS\_full & BZR & COX2 & PROTEINS\_full \\", r"\midrule"]
    for ref in ("size only", "graphlet count + LW", "GRAIL-0"):
        cells = []
        for conv in (MAIN_CONV, FLIP_CONV):
            for d in ("BZR", "COX2", "PROTEINS_full"):
                x = r[(r.convention == conv) & (r.reference == ref) & (r.dataset == d)]
                cells.append(f1(x.auroc.iloc[0]) if len(x) else "--")
        L.append(" & ".join([ref[0].upper() + ref[1:]] + cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_orientation_refs", "\n".join(L), ["a33_orient_refs"])


def a33_ablation_tables():
    u = pd.read_csv(SOURCES["a33_abl_ub9"])
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Detector & Quantity & Unablated & $x_s = 1$ & Paired difference & 95\% CI (dataset bootstrap) \\", r"\midrule"]
    lab = {"ORACLE": r"\oracle{}", "LAST": r"\last{}", "VAL-NORMAL": r"\valnormal{}",
           "ORACLE-VAL-NORMAL": r"\oracle{} $-$ \valnormal{}", "ORACLE-LAST": r"\oracle{} $-$ \last{}"}
    for i, m in enumerate(("CVTGAD", "GOOD-D")):
        if i:
            L.append(r"\midrule")
        for q, ql in lab.items():
            r = u[(u.detector == m) & (u.quantity == q)].iloc[0]
            L.append(f"{m} & {ql} & {f2_full(r['unablated'])} & {f2_full(r['ablated'])} & {sgn2(r['diff'])} & "
                     f"{ci(r.get('ds_ci_lo'), r.get('ds_ci_hi'), f2_full)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_ablation_ub9", "\n".join(L), ["a33_abl_ub9"])
    t = pd.read_csv(SOURCES["a33_abl_typed"])
    keys = [("T1a", "matched"), ("T3a", "matched"), ("T3b", "matched"), ("MS1", "matched"), ("MS2", "matched"),
            ("MS3", "matched"), ("MS1", "changing"), ("MS2", "changing"), ("MS3", "changing")]
    L = [r"\begin{tabular}{llccc@{\hspace{8pt}}ccc@{\hspace{8pt}}ccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{GX, size kept} & \multicolumn{3}{c}{MS-v2, size kept} & "
         r"\multicolumn{3}{c}{MS-v2, size changed} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"Detector & Input & T1a & T3a & T3b & MS1 & MS2 & MS3 & MS1 & MS2 & MS3 \\", r"\midrule"]
    for i, m in enumerate(("CVTGAD", "GOOD-D")):
        if i:
            L.append(r"\midrule")
        for col, lab_ in (("unablated", "encodings"), ("ablated", "$x_s = 1$")):
            cells = []
            for a, sv in keys:
                x = t[(t.detector == m) & (t.anomaly_type == a) & (t.size_variant == sv)]
                cells.append(bold60(x[col].iloc[0]) if len(x) else "--")
            L.append(" & ".join([m if col == "unablated" else "", lab_] + cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_ablation_typed", "\n".join(L), ["a33_abl_typed"])


def main():
    typed_tables()
    typed_validity()
    typed_decision()
    typed_msv1()
    typed_msv2_design()
    kernel_tables()
    kernel_verbatim()
    kernel_published()
    leaderboard_full()
    per_dataset_tables()
    inflation_table()
    kendall_table()
    sensitivity_table()
    flags_table()
    noise_table()
    selectors_table()
    version_table()
    a33_orientation_table()
    a33_ablation_tables()


if __name__ == "__main__":
    main()
