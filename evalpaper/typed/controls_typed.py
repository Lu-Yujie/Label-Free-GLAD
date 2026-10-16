"""A9 §4 controls for the typed test -> prelim/results/eval_typed_controls.csv (RUN_NOTES.md §0 item 13).

MS (new): per seed, fitted on the 982 training normals only, scored on test normals + each (type, variant) cell:
  GRAIL0:ctrl_gc_lw        graphlet counts (grail0.graphlets ESU kmax 5) -> grail0.controls.gc_feats -> Ledoit-Wolf
                           Mahalanobis + ridge_A                                   (= m2_typed/grail0_rerun.py)
  GRAIL0:ctrl_size_lw      grail0.channels.size_feats -> LW + ridge_S0             (= m2_typed/grail0_rerun.py)
  POSTHOC:ctrl_edgelab_lw  log1p(counts of every unordered node-label pair over edges) -> LW + ridge_A
                           (= m2_typed/posthoc_controls.py; label alphabet 4 on GX -> 10 pairs, 37 on MS -> 703 pairs)
GX: the M2 controls (same exported sets and seeds) are reused verbatim (stats/FROZEN.md interpretation 11) and copied
  here with their source file; this script recomputes them with the code above only as an implementation check.

Validity table (A9 §4 + A25, mechanical, 5-seed means in percent): valid iff gc_lw >= 85 and size_lw <= 60;
confounded iff edgelab_lw >= 85; feasible iff no seed has < 100 anomalies. Carried in extra 'type_*' columns (the file
matches glad_stats' eval_typed*.csv glob, so it follows the FROZEN.md typed schema) and cross-checked against the
frozen glad_stats.type_table.

Usage (CPU env):  PYTHONDONTWRITEBYTECODE=1 /N/slate/yujlu/envs/grail-prelim/bin/python -B controls_typed.py
          [--out prelim/results/eval_typed_controls.csv] [--check-only]

CHANGE-MSv2 (A28): `--generator MSV2` computes the same three controls on MS-v2 (typed/molswap_v2.py, FROZEN_MSv2.md)
and writes prelim/results/eval_typed_controls_msv2_PENDING.csv (meta typed/controls_meta_msv2.json). Differences from
the v1 path: (i) the GX rows are copied verbatim from the v1 controls file (--gx-source, default
prelim/results/eval_typed_controls.csv; they are M2's values), and the GX implementation check compares the
recomputation with those rows, so no prelim/results/m2_* file is read; (ii) a null-edit check (unedited parents of each
size-matched cell scored as if anomalous) goes to the meta file only; (iii) the glad_stats cross-check uses the MS-v2
manifest; (iv) BLAS threads must be 1 or 2 (exact score ties are otherwise broken by thread-dependent floating-point
noise; NOTES_MSv2.md §2 D0). Rows get an extra column generator_version. The default (v1) path is unchanged.
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        /N/slate/yujlu/envs/grail-prelim/bin/python -B controls_typed.py --generator MSV2
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import os.path as osp
import platform
import shutil
import tempfile
import time
import types

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
EVALPAPER = osp.dirname(HERE)
PRELIM = osp.dirname(EVALPAPER)
RES = osp.join(PRELIM, "results")
for _p in (PRELIM, EVALPAPER, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import typed_data as TD  # noqa: E402

ENGINE = ("esu", 5)
A9_VALID_GC, A9_VALID_SIZE, A9_CONFOUND, A25_FLOOR = 85.0, 60.0, 85.0, 100
CTRL = {"gc_lw": "GRAIL0:ctrl_gc_lw", "size_lw": "GRAIL0:ctrl_size_lw", "elp_lw": "POSTHOC:ctrl_edgelab_lw"}
CODE_FILES = [osp.join(HERE, "controls_typed.py"), osp.join(HERE, "typed_data.py"),
              osp.join(PRELIM, "grail0", "graphlets.py"), osp.join(PRELIM, "grail0", "controls.py"),
              osp.join(PRELIM, "grail0", "channels.py"), osp.join(PRELIM, "common", "metrics.py"),
              osp.join(PRELIM, "m2_typed", "grail0_rerun.py"), osp.join(PRELIM, "m2_typed", "posthoc_controls.py")]


def sha(p):
    return TD.sha256_file(p) if osp.exists(p) else None


def edgelab(graphs, L):
    """m2_typed/posthoc_controls.edgelab with the pair list over the label alphabet 0..L-1 (CHANGE-TYPED: L)."""
    pairs = [(a, b) for a in range(L) for b in range(a, L)]
    out = np.zeros((len(graphs), len(pairs)))
    idx = {p: i for i, p in enumerate(pairs)}
    for i, g in enumerate(graphs):
        lab = np.asarray(g["node_label"])
        assert lab.min() >= 0 and lab.max() < L
        e = np.asarray(g["edges"]).reshape(-1, 2)
        a, b = np.minimum(lab[e[:, 0]], lab[e[:, 1]]), np.maximum(lab[e[:, 0]], lab[e[:, 1]])
        for x, y in zip(a, b):
            out[i, idx[(int(x), int(y))]] += 1
    return out


def control_scores(train, test, L):
    """dict control -> scores of `test` (higher = more anomalous); fitted on `train` only."""
    from grail0 import graphlets as GL
    from grail0 import controls as C
    from grail0.channels import CONFIG
    Ftr = [GL.graph_features(g, *ENGINE) for g in train]
    Fte = [GL.graph_features(g, *ENGINE) for g in test]
    out = {}
    out["size_lw"] = C.mahal(C.size_feats(Ftr), C.size_feats(Fte), C.CONFIG["ridge_S0"])
    out["gc_lw"] = C.mahal(C.gc_feats(Ftr), C.gc_feats(Fte), C.CONFIG["ridge_A"])
    Etr, Ete = edgelab(train, L), edgelab(test, L)
    out["elp_lw"] = C.mahal(np.log1p(Etr), np.log1p(Ete), CONFIG["ridge_A"])
    return out


def score_unit(td, L):
    """Rows (one per control x group) for one TypedData; controls use ALL training normals (td.train)."""
    from common.metrics import evaluate
    graphs, y, grp = td.combined()
    S = control_scores(td.train, graphs, L)
    grp = np.asarray(grp)
    rows = []
    for key in td.groups:
        sel = (grp == "normal") | (grp == key)
        yy = (grp[sel] == key).astype(int)
        for c, s in S.items():
            m = evaluate(np.asarray(s)[sel], yy)
            rows.append(dict(method=CTRL[c], **td.cell(key), seed=td.seed, criterion="training-free",
                             auroc=100 * m["auroc"], auprc=100 * m["auprc"], fpr95=100 * m["fpr95"]))
    return rows


def m2_gx_rows():
    """The M2 GX controls (verbatim values), in percent."""
    a = pd.read_csv(osp.join(RES, "m2_typed_runs.csv"))
    a = a[(a.dataset == "GX") & (a.feature_variant == "F2") & a.anomaly_type.isin(TD.GX_TYPES)
          & a.method.isin([CTRL["gc_lw"], CTRL["size_lw"]])].copy()
    a["source"] = "prelim/results/m2_typed_runs.csv (M2, feature_variant F2 rows)"
    b = pd.read_csv(osp.join(RES, "m2_posthoc_controls.csv"))
    b = b[(b.dataset == "GX") & (b.method == "ctrl_edgelab_lw") & b.anomaly_type.isin(TD.GX_TYPES)].copy()
    b["method"] = CTRL["elp_lw"]
    b["source"] = "prelim/results/m2_posthoc_controls.csv (M2 post hoc)"
    d = pd.concat([a, b], ignore_index=True)
    scale = 100.0 if d.auroc.abs().max() <= 1.0 + 1e-9 else 1.0
    for k in ("auroc", "auprc", "fpr95"):
        d[k] = d[k].astype(float) * scale
    d["seed"] = d.seed.astype(int)
    d["criterion"] = "training-free"
    d["size_variant"] = "matched"
    d["feature_variant"] = "F2"
    d["n_anomalies"] = 250
    d["n_normals"] = 250
    return d[["method", "dataset", "anomaly_type", "size_variant", "feature_variant", "seed", "criterion", "auroc",
              "auprc", "fpr95", "n_anomalies", "n_normals", "source"]]


def validity(df, ms_counts):
    rows = []
    for (g, a, v), s in df.groupby(["dataset", "anomaly_type", "size_variant"], sort=False):
        mean = {c: float(s[s.method == CTRL[c]].auroc.mean()) for c in CTRL}
        nseed = {c: int(s[s.method == CTRL[c]].seed.nunique()) for c in CTRL}
        counts = ms_counts.get((a, v), []) if g == "MS" else [250] * 5
        feasible = bool(len(counts) == 5 and min(counts) >= A25_FLOOR)
        valid = bool(mean["gc_lw"] >= A9_VALID_GC and mean["size_lw"] <= A9_VALID_SIZE)
        conf = bool(mean["elp_lw"] >= A9_CONFOUND)
        rows.append(dict(dataset=g, anomaly_type=a, size_variant=v, type_gc_lw_mean=mean["gc_lw"],
                         type_size_lw_mean=mean["size_lw"], type_elp_lw_mean=mean["elp_lw"],
                         type_seeds=min(nseed.values()), type_valid=valid, type_confounded=conf,
                         type_feasible=feasible, type_min_anomalies=min(counts) if counts else np.nan,
                         type_counts_per_seed="|".join(map(str, counts)),
                         type_in_decision=bool(v == "matched" and valid and not conf and feasible)))
    return pd.DataFrame(rows)


def glad_stats_crosscheck(csv_path, V, ms_manifest=None):
    """Run the FROZEN glad_stats.type_table on a temporary results dir holding only this file; compare flags.
    CHANGE-MSv2: ms_manifest overrides glad_stats.MS_MANIFEST (the MS-v2 manifest for --generator MSV2)."""
    import importlib.util
    p = osp.join(EVALPAPER, "stats", "glad_stats.py")
    spec = importlib.util.spec_from_file_location("glad_stats_frozen", p)
    gs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs)
    tmp = tempfile.mkdtemp(prefix="typed_ctrl_check_")
    try:
        shutil.copy(csv_path, osp.join(tmp, osp.basename(csv_path)))
        args = types.SimpleNamespace(res=tmp)
        notes, prov = [], {}
        t = gs.load_typed(args, notes, prov)
        T = gs.type_table(t, gs.load_m2_gx_controls(args), gs.load_ms_manifest(ms_manifest or gs.MS_MANIFEST))
        sv = {"matched": "matched", "changing": "changing"}
        bad = []
        for r in V.itertuples():
            q = T[(T.generator == r.dataset) & (T.anomaly_type == r.anomaly_type) & (T.size_variant == sv[r.size_variant])]
            assert len(q) == 1, (r.dataset, r.anomaly_type, r.size_variant)
            q = q.iloc[0]
            for mine, theirs in (("type_valid", "valid"), ("type_confounded", "confounded"),
                                 ("type_feasible", "feasible"), ("type_in_decision", "in_decision")):
                if bool(getattr(r, mine)) != bool(q[theirs]):
                    bad.append((r.dataset, r.anomaly_type, r.size_variant, mine, getattr(r, mine), q[theirs]))
            for mine, theirs in (("type_gc_lw_mean", "graphlet_count_lw"), ("type_size_lw_mean", "size_only"),
                                 ("type_elp_lw_mean", "edge_label_pair_lw")):
                if abs(getattr(r, mine) - float(q[theirs])) > 1e-9:
                    bad.append((r.dataset, r.anomaly_type, r.size_variant, mine, getattr(r, mine), q[theirs]))
        return dict(glad_stats_sha256=sha(p), n_types=len(T), mismatches=bad, notes=notes,
                    ms_manifest=ms_manifest or gs.MS_MANIFEST)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def gx_rows_from_v1_file(path):                                                       # CHANGE-MSv2
    """The GX control rows of the v1 controls file (M2 values copied verbatim there), in the M2-row columns."""
    d = pd.read_csv(path)
    d = d[d.dataset == "GX"].copy()
    cols = ["method", "dataset", "anomaly_type", "size_variant", "feature_variant", "seed", "criterion", "auroc",
            "auprc", "fpr95", "n_anomalies", "n_normals", "source"]
    d = d[cols].reset_index(drop=True)
    d["source"] = d["source"] + f" [copied verbatim via {osp.relpath(path, PRELIM)}]"
    assert len(d) == 45 and set(d.method) == set(CTRL.values()), (len(d), set(d.method))
    return d


def null_edit(td, L):                                                                  # CHANGE-MSv2
    """Unedited parents of each size-matched cell vs the test normals (a parent-selection check; meta only)."""
    from common.metrics import evaluate
    from typed import molswap_v2 as M2
    by_gid = {g["gid"]: g for g in M2.load_p0()[0]}
    out = {}
    for key, an in td.groups.items():
        if not key.endswith(":matched"):
            continue
        parents = [by_gid[g["parent_gid"]] for g in an]
        S = control_scores(td.train, list(td.test_normals) + parents, L)
        y = np.r_[np.zeros(len(td.test_normals), int), np.ones(len(parents), int)]
        out[key] = {CTRL[c]: 100 * evaluate(np.asarray(v), y)["auroc"] for c, v in S.items()}
    return out


def main_msv2(a):                                                                      # CHANGE-MSv2
    thr = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    if any(v not in ("1", "2") for v in thr.values()):
        raise SystemExit(f"set OMP/MKL/OPENBLAS_NUM_THREADS to 1 or 2 (got {thr}); see NOTES_MSv2.md §2 D0")
    out = a.out or osp.join(RES, "eval_typed_controls_msv2_PENDING.csv")
    meta_path = a.meta or osp.join(HERE, "controls_meta_msv2.json")
    t0 = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    gx = gx_rows_from_v1_file(a.gx_source)
    chk = []
    for s in TD.SEEDS:
        for r in score_unit(TD.load_typed("GX", s, "F2"), TD.N_LABELS_GX):
            ref = gx[(gx.method == r["method"]) & (gx.anomaly_type == r["anomaly_type"]) & (gx.seed == s)]
            assert len(ref) == 1, (r["method"], r["anomaly_type"], s)
            chk.append(abs(r["auroc"] - float(ref.auroc.iloc[0])) / 100.0)
    gx_check = dict(rows=len(chk), max_abs_diff_auroc_fraction=float(max(chk)), against=a.gx_source,
                    against_sha256=sha(a.gx_source))
    print(f"GX recomputation vs the GX rows of {osp.basename(a.gx_source)}: {gx_check}", flush=True)
    if gx_check["max_abs_diff_auroc_fraction"] > 1e-9:
        raise SystemExit("GX control recomputation does not reproduce the reference GX rows; stop")
    if a.check_only:
        return
    man = TD._msv2_manifest(verify=True)
    L = int(man["node_label_dim"])
    rows, nulls = [], {}
    for s in TD.SEEDS:
        td = TD.load_typed("MSV2", s)
        rows += score_unit(td, L)
        nulls[s] = null_edit(td, L)
        print(f"MS-v2 seed {s}: {len(td.train)} train normals, {len(td.test_normals)} test normals, "
              f"groups {td.summary()['groups']}", flush=True)
    ms = pd.DataFrame(rows)
    ms["source"] = "typed/controls_typed.py --generator MSV2 (A28 MS-v2, A9 §4 controls)"
    ms_counts = {(t, TD.SIZE_VARIANT_CSV[v]): list(map(int, man["status"][t][v]["counts_per_seed"]))
                 for t, v in TD.MSV2_CELLS}
    df = pd.concat([gx, ms[gx.columns]], ignore_index=True)
    df["generator_version"] = np.where(df.dataset == "GX", "GX (M2 export, A9)", "MS-v2 (A28)")
    V = validity(df, ms_counts)
    df = df.merge(V, on=["dataset", "anomaly_type", "size_variant"], how="left")
    assert not df.duplicated(["method", "dataset", "anomaly_type", "size_variant", "feature_variant", "seed",
                              "criterion"]).any()
    assert np.isfinite(df.auroc).all() and df.auroc.between(0, 100).all()
    os.makedirs(osp.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
    xc = glad_stats_crosscheck(out, V, ms_manifest=osp.join(TD.MSV2_ROOT, "manifest.json"))
    null_mean = {}
    for key in nulls[0]:
        null_mean[key] = {c: float(np.mean([nulls[s][key][c] for s in TD.SEEDS])) for c in CTRL.values()}
    import grail0, sklearn, scipy, networkx
    meta = dict(written=time.strftime("%Y-%m-%d %H:%M:%S %Z"), started=started, seconds=time.time() - t0,
                generator="MSV2", out=out, out_sha256=sha(out), rows=len(df), gx_recomputation_check=gx_check,
                glad_stats_crosscheck=xc, ms_label_alphabet=L, n_edge_label_pairs_ms=L * (L + 1) // 2,
                null_edit_auroc_per_seed={str(s): v for s, v in nulls.items()}, null_edit_auroc_mean=null_mean,
                engine=list(ENGINE), host=platform.node(), python=platform.python_version(), blas_threads=thr,
                versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                              scipy=scipy.__version__, networkx=networkx.__version__),
                code_sha256={osp.relpath(p, PRELIM): sha(p) for p in CODE_FILES + [osp.join(HERE, "molswap_v2.py")]},
                msv2_manifest_sha256=TD.MSV2_MANIFEST_SHA256, m2_manifest_sha256=sha(TD.M2_MANIFEST),
                gx_source_sha256=sha(a.gx_source))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    cols = ["dataset", "anomaly_type", "size_variant", "type_gc_lw_mean", "type_size_lw_mean", "type_elp_lw_mean",
            "type_min_anomalies", "type_valid", "type_confounded", "type_feasible", "type_in_decision"]
    print(V[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("null-edit (5-seed mean):", json.dumps(null_mean))
    print(f"wrote {out} ({len(df)} rows) sha256 {meta['out_sha256']}; glad_stats cross-check mismatches: "
          f"{xc['mismatches'] or 'none'}; meta {meta_path}")
    if xc["mismatches"]:
        raise SystemExit("validity table disagrees with the frozen glad_stats.type_table")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="MS", choices=["MS", "MSV2"])                # CHANGE-MSv2
    ap.add_argument("--gx-source", default=osp.join(RES, "eval_typed_controls.csv"))     # CHANGE-MSv2 (MSV2 only)
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--check-only", action="store_true", help="GX recomputation check only; write nothing")
    a = ap.parse_args()
    if a.generator == "MSV2":                                                          # CHANGE-MSv2
        return main_msv2(a)
    a.out = a.out or osp.join(RES, "eval_typed_controls.csv")                          # CHANGE-MSv2 (same default)
    a.meta = a.meta or osp.join(HERE, "controls_meta.json")                            # CHANGE-MSv2 (same default)
    t0 = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    # 1. implementation check on GX against M2 (same code path as MS, label alphabet 4)
    m2 = m2_gx_rows()
    chk = []
    for s in TD.SEEDS:
        for r in score_unit(TD.load_typed("GX", s, "F2"), TD.N_LABELS_GX):
            ref = m2[(m2.method == r["method"]) & (m2.anomaly_type == r["anomaly_type"]) & (m2.seed == s)]
            assert len(ref) == 1, (r["method"], r["anomaly_type"], s)
            chk.append(abs(r["auroc"] - float(ref.auroc.iloc[0])) / 100.0)
    gx_check = dict(rows=len(chk), max_abs_diff_auroc_fraction=float(max(chk)))
    print(f"GX recomputation vs M2: {gx_check}", flush=True)
    if gx_check["max_abs_diff_auroc_fraction"] > 1e-9:
        raise SystemExit("GX control recomputation does not reproduce M2; stop (implementation check failed)")
    if a.check_only:
        return
    # 2. MS controls
    man = TD._ms_manifest(verify=True)
    L = int(man["node_label_dim"])
    rows = []
    for s in TD.SEEDS:
        td = TD.load_typed("MS", s)
        rows += score_unit(td, L)
        print(f"MS seed {s}: {len(td.train)} train normals, {len(td.test_normals)} test normals, "
              f"groups {td.summary()['groups']}", flush=True)
    ms = pd.DataFrame(rows)
    ms["source"] = "typed/controls_typed.py (new, A9 §4)"
    ms_counts = {(t, TD.SIZE_VARIANT_CSV[v]): list(map(int, man["status"][t][v]["counts_per_seed"]))
                 for t, v in TD.MS_CELLS}
    df = pd.concat([m2_gx_rows(), ms[m2.columns]], ignore_index=True)
    V = validity(df, ms_counts)
    df = df.merge(V, on=["dataset", "anomaly_type", "size_variant"], how="left")
    assert not df.duplicated(["method", "dataset", "anomaly_type", "size_variant", "feature_variant", "seed",
                              "criterion"]).any()
    assert np.isfinite(df.auroc).all() and df.auroc.between(0, 100).all()
    os.makedirs(osp.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    df.to_csv(tmp, index=False)   # full precision (repr)
    os.replace(tmp, a.out)
    xc = glad_stats_crosscheck(a.out, V)
    import grail0, sklearn, scipy, networkx
    meta = dict(written=time.strftime("%Y-%m-%d %H:%M:%S %Z"), started=started, seconds=time.time() - t0,
                out=a.out, out_sha256=sha(a.out), rows=len(df), gx_recomputation_check=gx_check,
                glad_stats_crosscheck=xc, ms_label_alphabet=L, n_edge_label_pairs_ms=L * (L + 1) // 2,
                engine=list(ENGINE), host=platform.node(), python=platform.python_version(),
                versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                              scipy=scipy.__version__, networkx=networkx.__version__),
                code_sha256={osp.relpath(p, PRELIM): sha(p) for p in CODE_FILES},
                ms_manifest_sha256=TD.MS_MANIFEST_SHA256, m2_manifest_sha256=sha(TD.M2_MANIFEST),
                m2_sources_sha256={f: sha(osp.join(RES, f)) for f in ("m2_typed_runs.csv", "m2_posthoc_controls.csv")})
    with open(a.meta, "w") as f:
        json.dump(meta, f, indent=1)
    cols = ["dataset", "anomaly_type", "size_variant", "type_gc_lw_mean", "type_size_lw_mean", "type_elp_lw_mean",
            "type_min_anomalies", "type_valid", "type_confounded", "type_feasible", "type_in_decision"]
    print(V[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"wrote {a.out} ({len(df)} rows) sha256 {meta['out_sha256']}; glad_stats cross-check mismatches: "
          f"{xc['mismatches'] or 'none'}; meta {a.meta}")
    if xc["mismatches"]:
        raise SystemExit("validity table disagrees with the frozen glad_stats.type_table")


if __name__ == "__main__":
    main()
