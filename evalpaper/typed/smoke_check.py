"""Smoke-test analysis for the A9 typed infrastructure. Prints NO metric value of any confirmatory method.

Checks (on $GLAD_DATA_ROOT/typed/smoke/, tags 'smoke', 'resume', 'time4'):
  1. automatic sanity: complete trajectory, every per-type AUROC/AUPRC/FPR95 finite and within [0, 1] (counts only);
  2. timings and peak memory per method x generator -> typed/runtime_smoke.csv;
  3. resume equivalence (stop at epoch 4 + resume vs straight run): max |diff| over trajectories and per-graph scores;
  4. identity with M2 for the discovery methods (old 4, GX F2 seed 0, first evaluation points): max |diff| only;
  5. batch dependence: max |score of the same test normal in different group sets| (0 = per-graph scoring);
  6. collector on the 'smoke' runs -> temp dir; frozen glad_stats typed functions (g2c_a9 = what analyse() calls) on a
     BLINDED copy (every deep AUROC/AUPRC/FPR95 replaced by U(0,1) noise; controls kept), structure printed only.
Usage: PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -B smoke_check.py [--report PATH]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import glob
import importlib.util
import json
import os
import os.path as osp
import shutil
import subprocess
import tempfile
import types

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
EVALPAPER = osp.dirname(HERE)
PRELIM = osp.dirname(EVALPAPER)
RES = osp.join(PRELIM, "results")
SMOKE = osp.join(os.environ.get("TYPED_ROOT", (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/typed")), "smoke")
M2PARTS = osp.join(PRELIM, "m2_typed", "parts")
OLD4 = ["SIGNET", "OCGTL", "OCGIN", "GLocalKD"]


def load(tag):
    out = {}
    for fn in sorted(glob.glob(osp.join(SMOKE, "parts", f"*_{tag}.json"))):
        J = json.load(open(fn))
        if J["meta"].get("tag") == tag:
            m = J["meta"]
            out[(m["method"], m["generator"], m["feature_variant"], int(m["seed"]))] = J
    return out


def numeric_leaves(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "t":
                continue
            yield from numeric_leaves(v, prefix + "/" + str(k))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from numeric_leaves(v, prefix + f"[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix, float(x)


def maxdiff_struct(a, b):
    la, lb = dict(numeric_leaves(a)), dict(numeric_leaves(b))
    if set(la) != set(lb):
        return float("inf"), sorted(set(la) ^ set(lb))[:5]
    d = 0.0
    for k in la:
        x, y = la[k], lb[k]
        if np.isnan(x) and np.isnan(y):
            continue
        d = max(d, abs(x - y))
    return d, []


def sanity(J):
    tr, h = J["traj"], J["meta"]["hparams"]
    ef, ne = int(h["eval_freq"]), int(h["num_epoch"])
    complete = [r["epoch"] for r in tr] == sorted(set(range(ef, ne + 1, ef)) | {ne})
    bad = 0
    for r in tr:
        for kk in ("types", "types_sel", "types_ubgold"):
            for v in (r.get(kk) or {}).values():
                for x in ("auroc", "auprc", "fpr95"):
                    bad += int(not (np.isfinite(v[x]) and 0.0 <= v[x] <= 1.0))
    hold_ok = all(np.isfinite(r["holdout_score"]) for r in tr)
    return complete, bad, hold_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=osp.join(SMOKE, "smoke_report.json"))
    ap.add_argument("--runtime_csv", default=osp.join(HERE, "runtime_smoke.csv"))
    a = ap.parse_args()
    rep = {}
    smoke, resume, time4, gpu8 = load("smoke"), load("resume"), load("time4"), load("gpu8")
    # 1 + 2
    rows, san = [], {}
    for tag, D in (("smoke", smoke), ("resume", resume), ("time4", time4), ("gpu8", gpu8)):
        for (m, g, fv, s), J in sorted(D.items()):
            c, bad, hok = sanity(J)
            san[f"{tag}:{m}:{g}:{fv}"] = dict(complete=c, bad_metric_values=bad, holdout_finite=hok)
            mt = J["meta"]
            if tag != "resume":
                rows.append(dict(tag=tag, method=m, generator=g, feature_variant=fv, threads=mt["threads"],
                                 device=("gpu" if "cuda" in mt["device"] else "cpu"),
                                 cuda_max_reserved_gb=mt["memory"].get("cuda_max_reserved_gb"),
                                 epochs=mt["epochs_this_segment"], n_eval=mt["n_eval"],
                                 sec_per_epoch_train=mt["sec_per_epoch_train"], sec_per_eval=mt["sec_per_eval"],
                                 seconds_total=mt["seconds"], maxrss_gb=mt["memory"]["maxrss_gb"],
                                 per_type_mode=mt["per_type_mode"], n_test_combined=mt["n_test_combined"],
                                 num_epoch_modal=None))
    rep["sanity"] = san
    rep["sanity_all_ok"] = all(v["complete"] and v["bad_metric_values"] == 0 and v["holdout_finite"] for v in san.values())
    rt = pd.DataFrame(rows)
    if len(rt):
        rt.to_csv(a.runtime_csv, index=False)
    print(f"[1] sanity: {len(san)} runs, all ok = {rep['sanity_all_ok']}; failing: "
          f"{[k for k, v in san.items() if not (v['complete'] and v['bad_metric_values'] == 0 and v['holdout_finite'])]}")
    print(f"[2] timings -> {a.runtime_csv}")
    if len(rt):
        print(rt[["tag", "method", "generator", "feature_variant", "device", "threads", "sec_per_epoch_train",
                  "sec_per_eval", "maxrss_gb", "per_type_mode"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # 3 resume
    res = {}
    for key, J in resume.items():
        ref = smoke.get(key)
        if ref is None:
            res["|".join(map(str, key))] = "no straight run"
            continue
        d1, miss = maxdiff_struct(J["traj"], ref["traj"])
        d2, _ = maxdiff_struct(J["train_loss"], ref["train_loss"])
        Za, Zb = np.load(J["meta"]["scores_npz"]), np.load(ref["meta"]["scores_npz"])
        d3 = max(float(np.nanmax(np.abs(Za[k] - Zb[k]))) if Za[k].size else 0.0 for k in Za.files
                 if k.startswith(("test", "hold")))
        res["|".join(map(str, key))] = dict(traj=d1, train_loss=d2, scores=d3, segments=len(J["meta"]["segments"]),
                                            key_mismatch=miss)
    rep["resume"] = res
    print("[3] resume equivalence (max |diff|; 0.0 = identical):")
    for k, v in res.items():
        print("   ", k, v)
    # 4 M2 identity (old 4 only: discovery methods, not confirmatory)
    ident = {}
    for m in OLD4:
        J = smoke.get((m, "GX", "F2", 0))
        p = osp.join(M2PARTS, f"{m}_GX_F2_s0.json")
        if J is None or not osp.exists(p):
            continue
        R = json.load(open(p))
        rm = {r["epoch"]: r for r in R["traj"]}
        d = dict(types=0.0, holdout=0.0, train_loss=0.0, points=0)
        for r in J["traj"]:
            q = rm.get(r["epoch"])
            if q is None or r["epoch"] == J["meta"]["hparams"]["num_epoch"] and r["epoch"] % 4:
                continue
            d["points"] += 1
            for t in ("T1a", "T3a", "T3b"):
                for x in ("auroc", "auprc", "fpr95"):
                    d["types"] = max(d["types"], abs(r["types"][t][x] - q["types"][t][x]))
            d["holdout"] = max(d["holdout"], abs(r["holdout_score"] - q["holdout_loss"]))
        n = len(J["train_loss"])
        d["train_loss"] = float(np.max(np.abs(np.array(J["train_loss"]) - np.array(R["train_loss"][:n]))))
        ident[m] = d
    rep["m2_identity_old4"] = ident
    print("[4] identity with M2 (old 4, GX F2 seed 0, first points; max |diff| of per-type metrics / held-out / train loss):")
    for k, v in ident.items():
        print("   ", k, v)
    # 5 batch dependence: the same test normals are scored in every group's set; per-graph scorers give identical scores
    bd = {}
    for (m, g, fv, s), J in sorted(smoke.items()):
        if J["meta"]["per_type_mode"] != "own_set":
            bd[f"{m}:{g}:{fv}"] = "slice (batch_size_test 1: one pass, per-graph)"
            continue
        Z = np.load(J["meta"]["scores_npz"])
        keys = list(J["meta"]["groups"])
        nn = int(J["meta"]["n_test_normals"])
        ref = Z["test__" + keys[0].replace(":", "-")][:, :nn]
        dd = max(float(np.nanmax(np.abs(Z["test__" + k.replace(":", "-")][:, :nn] - ref))) for k in keys[1:])
        bd[f"{m}:{g}:{fv}"] = dd
    rep["normals_across_sets_maxabs"] = bd
    print("[5] max |score of the same test normal in different group sets| (0 = batch-independent scoring):")
    for k, v in bd.items():
        print("   ", k, v)
    # 6 collector + frozen glad_stats on a blinded temporary copy
    tmp = tempfile.mkdtemp(prefix="typed_ingest_")
    try:
        out = osp.join(tmp, "eval_typed.csv")
        py = sys.executable
        r = subprocess.run([py, "-B", osp.join(HERE, "collect_typed.py"), "--parts", osp.join(SMOKE, "parts"),
                            "--only-tag", "smoke", "--out", out], capture_output=True, text=True)
        print("[6] collector:", r.stdout.strip(), r.stderr.strip()[-500:])
        df = pd.read_csv(out)
        rng = np.random.default_rng(0)
        for c in ("auroc", "auprc", "fpr95"):                       # blind mirror (FROZEN.md practice)
            df[c] = rng.uniform(0, 100, len(df))
        df.to_csv(out, index=False)
        shutil.copy(osp.join(RES, "eval_typed_controls.csv"), tmp)
        spec = importlib.util.spec_from_file_location("glad_stats_frozen", osp.join(EVALPAPER, "stats", "glad_stats.py"))
        gs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gs)
        args = types.SimpleNamespace(res=tmp, ms_manifest=gs.MS_MANIFEST)
        notes, prov = [], {}
        g2c, T, cells = gs.g2c_a9(args, 200, 1, False, notes, prov)
        info = dict(rows_in_file=len(df), notes=[n for n in notes if "eval_typed" in n],
                    typed_cells=len(cells), cells_with_seeds=int((cells.n_seeds > 0).sum()),
                    methods_with_cells=sorted(cells[cells.n_seeds > 0].method.unique().tolist()),
                    feature_variants=sorted(df.feature_variant.unique().tolist()),
                    criteria=sorted(df.criterion.unique().tolist()), status=g2c.get("status"),
                    verdict=g2c.get("verdict"), decision_types=g2c["decision"]["decision_types"],
                    n_valid_ms_types=g2c["decision"]["n_valid_ms_types"],
                    has_F1_sensitivity="sensitivity_F1" in g2c, h4s=g2c["H4s_scope"].get("status"),
                    glad_stats_sha256=prov.get("glad_stats.py") or gs.sha256_file(osp.join(EVALPAPER, "stats", "glad_stats.py")))
        rep["glad_stats_ingest"] = info
        print("[6] frozen glad_stats.g2c_a9 on the BLINDED smoke copy (structure only):", json.dumps(info, default=str))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(a.report, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print("report ->", a.report)


if __name__ == "__main__":
    main()
