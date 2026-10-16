"""Unit tests of collect_typed_ext.py on SYNTHETIC parts (made-up numbers in a temp dir; no real run is read).

Checks the selection rules: UB-GOLD patience-30 rule with the eval_freq filter, VAL-NORMAL ties/non-finite, A15 chance
rows, LGKDE's EarlyStopping replay, the common-epoch rules (first max vs lower-std tie break, finite in all seeds),
GLASS's checkpoint replay (+5 per non-improving evaluation, patience) + best-of-24 rule, ProtoGLAD's first strictly
larger grid cell, the r0 rows, the checkpoint_free flag, and that tagged runs never reach results.
  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -B test_collect_typed_ext.py
"""
import sys

sys.dont_write_bytecode = True

import json
import os
import os.path as osp
import subprocess
import tempfile

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import collect_typed_ext as CX  # noqa: E402

KEYS = ["T1a", "T3a", "T3b"]
CELL = {k: dict(dataset="GX", anomaly_type=k, size_variant="matched", feature_variant="F2", n_anomalies=250,
                n_normals=250) for k in KEYS}


def tmet(a, nd=400, nf=0):
    return dict(auroc=a, auprc=0.5, fpr95=0.5, nonfinite=nf, n_distinct=nd)


def part(method, seed, epochs, ef, ne, auc, hold, extra_types=None, extra_rec=None, flags=None):
    traj = []
    for i, e in enumerate(epochs):
        types = {}
        for k in KEYS:
            a = auc(k, seed, i, e)
            t = tmet(a, nf=(0 if np.isfinite(a) else 7))
            if extra_types:
                t.update(extra_types(k, seed, i, e))
            types[k] = t
        rec = dict(epoch=e, types=types, holdout_score=hold(seed, i, e))
        if extra_rec:
            rec.update(extra_rec(seed, i, e))
        traj.append(rec)
    meta = dict(method=method, generator="GX", seed=seed, feature_variant="F2", run_id=f"{method}_GX_F2_s{seed}",
                tag="", max_epochs_override=None, hparams=dict(eval_freq=ef, num_epoch=ne), groups=CELL,
                device_all="cpu", hp_line="synthetic", flags=flags or {})
    return dict(meta=meta, traj=traj, train_loss=[])


def write(d, J):
    with open(osp.join(d, J["meta"]["run_id"] + ".json"), "w") as f:
        json.dump(J, f)


def rows_of(d):
    runs, skipped, bad = CX.load_runs(d, False)
    assert not bad, bad
    rows, inc = CX.collect(runs)
    return {(r["method"], r["anomaly_type"], r["seed"], r["criterion"]): r for r in rows}, inc


def main():
    tmp = tempfile.mkdtemp(prefix="test_cx_", dir=os.environ.get("TMPDIR"))
    d = osp.join(tmp, "parts")
    os.makedirs(d)
    nan = float("nan")
    # MUSE: eval every 20; UB-GOLD rule over snapshots; last nonfinite for T3b seed 0 -> A15.3 chance
    for s in range(5):
        ep = list(range(20, 201, 20))
        write(d, part("MUSE", s, ep, 20, 200,
                      lambda k, s_, i, e: (nan if (k == "T3b" and s_ == 0 and e == 200) else 0.5 + 0.01 * i),
                      lambda s_, i, e: 1.0 - 0.01 * min(i, 4)))
    # GLADMamba: ef 5, epochs 5..15; epoch 10 and 15 equal mean, 15 lower std -> 15; epoch 5 best but NaN in seed 4
    tab = {5: [0.9] * 4 + [nan], 10: [0.6, 0.8, 0.6, 0.8, 0.7], 15: [0.7] * 5}
    for s in range(5):
        write(d, part("GLADMamba", s, [5, 10, 15], 5, 15, lambda k, s_, i, e: tab[e][s_],
                      lambda s_, i, e: [3.0, 2.0, 2.0][i]))
    # GLADPro: every epoch 1..3; equal max mean at 2 and 3 -> first (2)
    tabp = {1: [0.5] * 5, 2: [0.6, 0.8, 0.7, 0.7, 0.7], 3: [0.7] * 5}
    for s in range(5):
        write(d, part("GLADPro", s, [1, 2, 3], 1, 3, lambda k, s_, i, e: tabp[e][s_], lambda s_, i, e: 1.0))
    # LGKDE: own_sel peaks at epoch 2 then flat -> EarlyStopping(patience 2, min_delta 0.001, min_epochs 1) stops at 4
    sel = [0.5, 0.7, 0.69, 0.69, 0.95]
    for s in range(5):
        write(d, part("LGKDE", s, [1, 2, 3, 4, 5], 1, 5, lambda k, s_, i, e: 0.6,
                      lambda s_, i, e: 1.0, extra_types=lambda k, s_, i, e: dict(own_sel=sel[i]),
                      flags=dict(own_rule=dict(patience=2, min_delta=0.001, min_epochs=1))))
    # GLASS: points 0,1,5,...,50; ens best at point 2 (epoch 5) then 8 non-improving evaluations -> stop (patience 40)
    gep = [0, 1] + list(range(5, 51, 5))
    ens = [50, 55, 60] + [59] * 8 + [99]
    rules = ["fixed_test_raw", "rcsm_full_test_cdf", "domain_test_z"]

    def grules(k, s_, i, e):
        return dict(glass_ens_auroc=ens[i], rules={"fixed_test_raw": [0.55, 0.5, 0.5, 0, 300],
                                                   "rcsm_full_test_cdf": [0.6, 0.5, 0.5, 0, 300],
                                                   "domain_test_z": [0.6 if i != 2 else 0.9, 0.5, 0.5, 0, 300]})
    for s in range(5):
        write(d, part("GLASS", s, gep, 5, 50, lambda k, s_, i, e: 0.6, lambda s_, i, e: 1.0 + (i == 3) * -1,
                      extra_types=grules,
                      extra_rec=lambda s_, i, e: dict(rule_holdout={r: 1.0 + (i == 4) * -1 for r in rules}),
                      flags=dict(patience=40, rules=rules)))
    # ProtoGLAD: one point; cells c1 < c2 = c3 -> c2 (first strictly larger)
    cells = ["c1", "c2", "c3"]
    for s in range(5):
        write(d, part("ProtoGLAD", s, [1], 1, 1, lambda k, s_, i, e: 0.6, lambda s_, i, e: 1.0,
                      extra_rec=lambda s_, i, e: dict(grid={"c1": {k: [0.6, .5, .5, 0, 300] for k in KEYS},
                                                            "c2": {k: [0.7, .5, .5, 0, 300] for k in KEYS},
                                                            "c3": {k: [0.7, .5, .5, 0, 300] for k in KEYS}}),
                      flags=dict(grid_cells=cells)))
    # a tagged smoke part: must be skipped by default
    J = part("SUL-GUL", 0, [1, 2], 1, 2, lambda k, s_, i, e: 0.6, lambda s_, i, e: 1.0)
    J["meta"].update(tag="smoke", run_id="SUL-GUL_GX_F2_s0_smoke", smoke=True)
    write(d, J)

    R, inc = rows_of(d)
    ok = []

    def check(name, cond):
        ok.append(cond)
        print(("PASS " if cond else "FAIL ") + name)

    check("MUSE oracle = best snapshot (epoch 200 except NaN case)", R[("MUSE", "T1a", 1, "oracle")]["epoch_selected"] == 200)
    check("MUSE no oracle_own", ("MUSE", "T1a", 1, "oracle_own") not in R)
    r = R[("MUSE", "T3b", 0, "last")]
    check("A15.3 chance for non-finite LAST", r["auroc"] == 50.0 and r["nonfinite_selected"] == 1 and r["fpr95"] == 95.0
          and abs(r["auprc"] - 50.0) < 1e-9)
    check("MUSE oracle skips NaN point", R[("MUSE", "T3b", 0, "oracle")]["epoch_selected"] == 180)
    check("VAL-NORMAL earliest minimum", R[("MUSE", "T1a", 2, "valnormal")]["epoch_selected"] == 100)
    check("GLADMamba own: lower-std tie -> 15, NaN epoch excluded",
          all(R[("GLADMamba", "T1a", s, "oracle_own")]["epoch_selected"] == 15 for s in range(5)))
    check("GLADMamba valnormal earliest tie", R[("GLADMamba", "T1a", 0, "valnormal")]["epoch_selected"] == 10)
    check("GLADPro own: first max -> 2", all(R[("GLADPro", "T3a", s, "oracle_own")]["epoch_selected"] == 2 for s in range(5)))
    check("LGKDE own: EarlyStopping stop before the late peak -> epoch 2",
          R[("LGKDE", "T1a", 0, "oracle_own")]["epoch_selected"] == 2)
    g = R[("GLASS", "T1a", 3, "oracle_own")]
    check("GLASS own: checkpoint epoch 5 (stop before the late ens peak), rule domain_test_z",
          g["epoch_selected"] == 5 and json.loads(g["own_detail"])["rule"] == "domain_test_z" and abs(g["auroc"] - 90.0) < 1e-9)
    check("GLASS UB-GOLD oracle considers epoch 0", R[("GLASS", "T1a", 0, "oracle")]["epoch_selected"] == 0)
    check("GLASS valnormal_r0 uses the r0 held-out mean", R[("GLASS", "T1a", 0, "valnormal_r0")]["epoch_selected"] == 15)
    check("GLASS valnormal uses the default held-out mean", R[("GLASS", "T1a", 0, "valnormal")]["epoch_selected"] == 10)
    p = R[("ProtoGLAD", "T3b", 4, "oracle_own")]
    check("ProtoGLAD own: first strictly larger cell c2", json.loads(p["own_detail"])["cell"] == "c2" and abs(p["auroc"] - 70) < 1e-9)
    check("ProtoGLAD checkpoint_free flag", R[("ProtoGLAD", "T3b", 4, "last")]["checkpoint_free"] == 1)
    check("tagged smoke run skipped", not any(k[0] == "SUL-GUL" for k in R))
    out_ok = subprocess.run([sys.executable, "-B", osp.join(HERE, "collect_typed_ext.py"), "--parts", d,
                             "--include-tagged", "--out", osp.join(CX.RES, "eval_typed_ext_TEST.csv")],
                            capture_output=True, text=True)
    check("refuses tagged runs into prelim/results", out_ok.returncode != 0 and
          not osp.exists(osp.join(CX.RES, "eval_typed_ext_TEST.csv")))
    print(f"{sum(ok)}/{len(ok)} checks passed (temp dir {tmp})")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
