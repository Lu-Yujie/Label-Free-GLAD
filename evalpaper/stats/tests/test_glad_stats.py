"""Unit tests for prelim/evalpaper/stats/glad_stats.py (v2). Synthetic fixtures only; no real result is read.

  export PYTHONDONTWRITEBYTECODE=1
  /N/slate/yujlu/envs/grail-prelim/bin/python -B -m unittest discover -s prelim/evalpaper/stats/tests -v
"""
import json
import os
import os.path as osp
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import fixtures as F  # noqa: E402

G = F.G
UB9 = G.UB9


def mkrow(run, base, epoch, auroc=None, hp=None, method=None, **kw):
    hp = hp or run["hp"]
    i = list(run["epochs"]).index(epoch) if epoch in list(run["epochs"]) else None
    a = auroc if auroc is not None else (100 * run["auroc"][i] if i is not None else np.nan)
    r = dict(method=method or run["method"], dataset=run["dataset"], fold=run["fold"], hp=hp, base=base,
             protocol=("hpfree_" if hp == "hpfree" else "") + base, epoch_selected=epoch, auroc=a, auprc=a,
             fpr95=100 - a if a == a else np.nan, run_id=run["run_id"], _runkey=("ext", run["run_id"]))
    r.update(kw)
    return r


TMP_BASE = os.environ.get("GS_TEST_TMP") or None     # optional parent directory for test temp dirs


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gs_test_", dir=TMP_BASE)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_of(self, method="SEGO", ds="AIDS", fold=0, hp="ubgold", **kw):
        p = F.write_part(self.tmp, method, ds, fold, hp, **kw)
        return G.run_info(p, "ext")


# ------------------------------------------------------------------------------------------------ A15
class TestA15(Tmp):
    def test_last_nonfinite_scored_chance(self):
        run = self.run_of(auroc=[0.6, 0.7, float("nan")], hold=[0.5, 0.4, 0.3], nonfinite=[0, 0, 3], n_test=40, n_anom=8)
        r = mkrow(run, "last", 3)
        G.a15_audit([r], {r["_runkey"]: run})
        self.assertEqual((r["auroc"], r["auprc"], r["fpr95"], r["nonfinite_selected"]), (50.0, 20.0, 95.0, 1))
        self.assertEqual(r["a15_action"], "A15.3_last_nonfinite")

    def test_adapter_chance_row_normalised_to_fpr95_95(self):
        run = self.run_of(method="GLADPro", auroc=[0.6, float("nan")], hold=[0.5, 0.4], nonfinite=[0, 2])
        r = mkrow(run, "last", 2, auroc=50.0, fpr95=100.0)       # GLADPro derive.py writes FPR95 100
        G.a15_audit([r], {r["_runkey"]: run})
        self.assertEqual((r["auroc"], r["fpr95"], r["nonfinite_selected"]), (50.0, 95.0, 1))

    def test_oracle_never_selects_nonfinite_point(self):
        run = self.run_of(auroc=[0.6, 0.9, 0.7], hold=[0.5, 0.4, 0.3], nonfinite=[0, 1, 0])
        self.assertEqual(G.ubgold_select(run, True), 2)
        r = mkrow(run, "oracle_ubgold", 2, auroc=np.nan)
        r2 = mkrow(run, "oracle", 2, auroc=np.nan, method="SIGNET")
        G.a15_audit([r, r2], {r["_runkey"]: run})
        for x in (r, r2):
            self.assertEqual(x["epoch_selected"], 3)
            self.assertAlmostEqual(x["auroc"], 70.0)
            self.assertEqual(x["a15_action"], "A15.1_reselected")
            self.assertEqual(x["nonfinite_selected"], 0)

    def test_valnormal_skips_nonfinite_holdout_ties_earliest(self):
        run = self.run_of(auroc=[0.6, 0.65, 0.7, 0.8], hold=[0.5, float("-inf"), 0.2, 0.2])
        self.assertEqual(G.valnormal_select(run), 2)
        r = mkrow(run, "valnormal", 2)                          # a nanargmin-based adapter would pick the -inf point
        G.a15_audit([r], {r["_runkey"]: run})
        self.assertEqual((r["epoch_selected"], r["auroc"], r["a15_action"]), (3, 70.0, "A15.2_reselected"))

    def test_valnormal_point_with_nonfinite_test_scores_is_chance(self):
        run = self.run_of(auroc=[0.6, float("nan"), 0.7], hold=[0.5, 0.1, 0.3], nonfinite=[0, 4, 0])
        r = mkrow(run, "valnormal", 2)
        G.a15_audit([r], {r["_runkey"]: run})
        self.assertEqual((r["auroc"], r["nonfinite_selected"]), (50.0, 1))
        self.assertIn("A15.3_selected_point_test_nonfinite", r["a15_action"])

    def test_valnormal_without_finite_holdout_is_chance(self):
        run = self.run_of(auroc=[0.6, 0.7], hold=[float("nan"), float("inf")])
        r = mkrow(run, "valnormal", 1)
        G.a15_audit([r], {r["_runkey"]: run})
        self.assertEqual((r["auroc"], r["a15_action"]), (50.0, "A15.2_no_finite_holdout"))

    def test_all_points_nonfinite_every_protocol_chance(self):
        run = self.run_of(auroc=[float("nan")] * 3, hold=[0.1, 0.2, 0.3], nonfinite=[1, 1, 1])
        rows = [mkrow(run, b, 3, auroc=np.nan) for b in ("oracle", "oracle_ubgold", "last", "valnormal")]
        G.a15_audit(rows, {rows[0]["_runkey"]: run})
        for r in rows:
            self.assertEqual((r["auroc"], r["nonfinite_selected"], r["a15_action"]),
                             (50.0, 1, "A15.4_all_points_nonfinite"))

    def test_common_epoch_own_rule_reselected_over_all_finite_epochs(self):
        runs, rows = {}, []
        for f in range(5):
            a = [0.6, 0.8, 0.99, 0.7]
            nf = [0, 0, 0, 0]
            if f == 2:
                a[2], nf[2] = float("nan"), 3
            run = self.run_of(method="GLADPro", fold=f, auroc=a, hold=[0.4, 0.3, 0.2, 0.1], nonfinite=nf)
            runs[("ext", run["run_id"])] = run
            rows.append(mkrow(run, "oracle", 3, auroc=50.0 if f == 2 else 99.0))  # GLADPro's NaN->0 rule picked 3
        regroup = G.a15_audit(rows, runs)
        self.assertEqual(regroup, {("GLADPro", "AIDS", "ubgold")})
        G.common_epoch_reselect(rows, runs, regroup)
        for r in rows:
            self.assertEqual((r["epoch_selected"], r["auroc"], r["a15_action"]),
                             (2, 80.0, "A15.1_reselected_common_epoch"))

    def test_own_rule_that_cannot_be_replayed_is_listed_unresolved(self):
        run = self.run_of(method="LGKDE", auroc=[0.6, float("nan")], hold=[0.5, 0.4], nonfinite=[0, 1])
        r = mkrow(run, "oracle", 2, auroc=np.nan)
        G.a15_audit([r], {r["_runkey"]: run})
        self.assertEqual((r["auroc"], r["a15_action"]), (50.0, "A15.1_unresolved_own_rule"))

    def test_missing_protocol_rows_are_synthesised(self):
        run = self.run_of(auroc=[0.6, 0.9, float("nan")], hold=[0.5, 0.1, 0.3], nonfinite=[0, 0, 2])
        key = ("ext", run["run_id"])
        rows = [mkrow(run, "last", 3, auroc=np.nan)]
        G.a15_audit(rows, {key: run})
        new = G.synth_missing(rows, {key: run}, {("SEGO", "AIDS", "ubgold")})
        got = {r["base"]: r for r in new}
        self.assertEqual(set(got), {"valnormal", "oracle_ubgold", "oracle"})   # SEGO's ORACLE is the UB-GOLD rule
        self.assertEqual((got["valnormal"]["epoch_selected"], got["valnormal"]["auroc"]), (2, 90.0))
        self.assertTrue(all(r["a15_action"].startswith("synth") for r in new))


# ------------------------------------------------------------------------------------------------ A21 ties
class TestTies(Tmp):
    def _scores(self, n_pts, n_test, distinct_at_last, nan_at_last=0):
        rng = np.random.default_rng(1)
        T = rng.normal(size=(n_pts, n_test))
        T[-1] = np.arange(n_test) % distinct_at_last
        if nan_at_last:
            T[-1, :nan_at_last] = np.nan
        return T

    def test_threshold_is_max_10_and_5pct(self):
        for n_test, d, expect in ((400, 19, 1), (400, 20, 0), (100, 9, 1), (100, 10, 0)):
            run = self.run_of(fold=d % 5, ds=f"D{n_test}_{d}", auroc=[0.6, 0.7], hold=[0.2, 0.3], n_test=n_test,
                              test_scores=self._scores(2, n_test, d))
            r = mkrow(run, "last", 2)
            G.tie_flags([r], {r["_runkey"]: run})
            self.assertEqual((r["n_distinct"], r["tie_threshold"], r["tie_collapsed"]),
                             (d, max(10.0, 0.05 * n_test), expect), (n_test, d))

    def test_nonfinite_scores_not_counted(self):
        run = self.run_of(auroc=[0.6, 0.7], hold=[0.2, 0.3], n_test=100,
                          test_scores=self._scores(2, 100, 10, nan_at_last=10))
        r = mkrow(run, "last", 2)
        G.tie_flags([r], {r["_runkey"]: run})
        self.assertEqual(r["tie_collapsed"], 0)       # 10 distinct finite values remain (NaN rows are values 0..9)

    def test_glass_own_oracle_uses_own_ref_scores(self):
        run = self.run_of(method="GLASS", auroc=[0.6, 0.7], hold=[0.2, 0.3], n_test=100,
                          npz_extra={"own_ref__rule_x": np.repeat([0.0, 1.0], 50)})
        r = mkrow(run, "oracle", 2, scorer="rule_x")
        G.tie_flags([r], {r["_runkey"]: run})
        self.assertEqual((r["n_distinct"], r["tie_collapsed"]), (2, 1))

    def test_rows_without_scores_are_unavailable_not_flagged(self):
        run = self.run_of(auroc=[0.6, 0.7], hold=[0.2, 0.3], write_npz=False)
        r = mkrow(run, "last", 2)
        G.tie_flags([r], {r["_runkey"]: run})
        self.assertEqual(r["tie_source"], "unavailable")
        self.assertTrue(np.isnan(r["tie_collapsed"]))


# ------------------------------------------------------------------------------------------------ long-table helpers
def long_frame(spec, datasets=UB9, folds=range(5)):
    """spec: {method: {protocol: value or callable(ds, fold)}} -> finalized long table."""
    rows = []
    for m, protos in spec.items():
        for p, v in protos.items():
            for ds in datasets:
                for f in folds:
                    val = v(ds, f) if callable(v) else v
                    if val is None:
                        continue
                    rows.append(dict(method=m, dataset=ds, fold=f, hp=G.hp_of(p), base=G.base_of(p), protocol=p,
                                     auroc=val, auprc=val, fpr95=100 - val, epoch_selected=1, run_id=f"{m}_{ds}_{f}",
                                     nonfinite_selected=0, a15_action="", tie_collapsed=np.nan, hp_alias=0))
    return G.finalize(rows)


def frozen_spec(gv, gl, n=12, hp_leak=3.0):
    return {m: {"oracle": 80.0, "oracle_ubgold": 80.0, "valnormal": 80.0 - gv(i), "last": 80.0 - gl(i),
                "hpfree_oracle": 77.0, "hpfree_valnormal": 80.0 - gv(i) - hp_leak, "hpfree_last": 75.0}
            for i, m in enumerate(G.DEEP12[:n])}


EMPTY_DIV = pd.DataFrame(columns=["method", "dataset", "fold", "hp"])


# ------------------------------------------------------------------------------------------------ A14 / G1
class TestG1(unittest.TestCase):
    def g1(self, spec, final=False, div=EMPTY_DIV):
        long = long_frame(spec)
        C = G.Cubes(long)
        cov = G.coverage_table(long, div, sorted(set(long.method)))
        vn = G.g1_selector(C, cov, "valnormal", 50, 1, final)
        la = G.g1_selector(C, cov, "last", 50, 1, final)
        return vn, la, G.g1_combined(vn, la)

    def test_pass_under_both(self):
        vn, la, comb = self.g1(frozen_spec(lambda i: 4.0, lambda i: 4.0))
        self.assertEqual((vn["status"], la["status"], comb), ("PASS", "PASS", "PASS"))

    def test_selector_dependent(self):
        vn, la, comb = self.g1(frozen_spec(lambda i: 4.0, lambda i: 1.0))
        self.assertEqual((vn["status"], la["status"], comb), ("PASS", "FAIL", "SELECTOR-DEPENDENT"))
        vn, la, comb = self.g1(frozen_spec(lambda i: 1.0, lambda i: 4.0))
        self.assertEqual(comb, "SELECTOR-DEPENDENT")

    def test_fail_under_both(self):
        self.assertEqual(self.g1(frozen_spec(lambda i: 1.0, lambda i: 1.0))[2], "FAIL")

    def test_75pct_rule(self):
        # mean >= 3 but only 8/12 methods >= 2.0 -> FAIL
        vn, _, _ = self.g1(frozen_spec(lambda i: 6.0 if i < 8 else 0.0, lambda i: 6.0))
        self.assertGreaterEqual(vn["mean_inflation"], 3.0)
        self.assertFalse(vn["conditions"]["frac(>=2.0)>=0.75"])
        self.assertEqual(vn["status"], "FAIL")

    def test_pending_until_resolved_and_final(self):
        spec = frozen_spec(lambda i: 4.0, lambda i: 4.0, n=7)
        vn, la, comb = self.g1(spec)
        self.assertEqual((vn["status"], comb), ("PENDING", "PENDING"))
        vn, la, comb = self.g1(spec, final=True)
        self.assertEqual((vn["status"], comb), ("FAIL", "FAIL"))      # 7 complete < 8

    def test_diverged_ub9_cell_removes_coverage_but_resolves(self):
        spec = frozen_spec(lambda i: 4.0, lambda i: 4.0)
        drop = lambda ds, f: None if ds == "AIDS" and f >= 3 else 80.0  # noqa: E731
        spec["GLADC"] = {p: (drop if p in ("oracle", "oracle_ubgold", "valnormal", "last") else v)
                         for p, v in spec["GLADC"].items()}
        div = pd.DataFrame([dict(method="GLADC", dataset="AIDS", fold=f, hp="ubgold") for f in (3, 4)])
        vn, la, comb = self.g1(spec, div=div)
        self.assertNotIn("GLADC", vn["complete_methods"])
        self.assertEqual(vn["n_complete"], 11)
        self.assertTrue(vn["all_methods_resolved"])
        self.assertEqual(vn["status"], "PASS")                        # resolved without --final

    def test_go_decision_no_go_when_selector_dependent(self):
        # GO logic in analyse(): replicate the rule on the combined label
        for g1p, g2, expect in (("SELECTOR-DEPENDENT", "PASS", "NO-GO"), ("PASS", "PASS", "GO"),
                                ("PASS", "PENDING", "PENDING"), ("FAIL", "PASS", "NO-GO")):
            go = ("GO" if g1p == "PASS" and g2 == "PASS" else
                  "NO-GO" if g1p in ("FAIL", "SELECTOR-DEPENDENT") or g2 == "FAIL" else "PENDING")
            self.assertEqual(go, expect)


class TestDegenerate(unittest.TestCase):
    def make(self, n_early, gap, n=10):
        rows, runs = [], {}
        for f in range(n):
            key = ("ext", f"X{f}")
            runs[key] = dict(num_epoch=100)
            rows.append(dict(method="SEGO", dataset=UB9[f % 9], fold=f // 9, hp="ubgold", base="valnormal",
                             protocol="valnormal", auroc=70.0 - gap, epoch_selected=5 if f < n_early else 50, _runkey=key))
            rows.append(dict(method="SEGO", dataset=UB9[f % 9], fold=f // 9, hp="ubgold", base="last",
                             protocol="last", auroc=70.0, epoch_selected=100, _runkey=key))
        return G.valnormal_degenerate(pd.DataFrame(rows), runs)

    def test_flag_rule(self):
        d = self.make(8, 5.0)
        self.assertTrue(bool(d[(d.scope == "all")].valnormal_degenerate.iloc[0]))
        d = self.make(7, 5.0)
        self.assertFalse(bool(d[(d.scope == "all")].valnormal_degenerate.iloc[0]))   # 70% < 80%
        d = self.make(10, 4.99)
        self.assertFalse(bool(d[(d.scope == "all")].valnormal_degenerate.iloc[0]))   # gap < 5

    def test_epoch_at_exactly_10pct_counts_as_early(self):
        rows, runs = [], {("e", "r"): dict(num_epoch=400)}
        for b, e, a in (("valnormal", 40, 60.0), ("last", 400, 70.0)):
            rows.append(dict(method="SEGO", dataset="AIDS", fold=0, hp="ubgold", base=b, protocol=b, auroc=a,
                             epoch_selected=e, _runkey=("e", "r")))
        d = G.valnormal_degenerate(pd.DataFrame(rows), runs)
        self.assertEqual(float(d.frac_valnormal_first10pct.iloc[0]), 1.0)


# ------------------------------------------------------------------------------------------------ A20 / A21
class TestExclusions(Tmp):
    def test_checkpoint_free_excluded_from_pooled_checkpoint_inflation_only(self):
        spec = {"GLADPro": {"oracle": 80.0, "valnormal": 76.0, "last": 77.0, "hpfree_valnormal": 70.0},
                "SEGO": {"oracle": 80.0, "valnormal": 78.0, "last": 78.0, "hpfree_valnormal": 77.0},
                "ProtoGLAD": {"oracle": 75.0, "valnormal": 75.0, "last": 75.0, "hpfree_valnormal": 71.0}}
        long = long_frame(spec)
        C = G.Cubes(long)
        per, pooled = G.inflation(C, G.Cubes(long.iloc[0:0]), long, 50, 1, set())
        p = pooled[(pooled.set == "UB9") & (pooled.pool == "recent(E5+E4)")].set_index("comparison")
        ck = p.loc["checkpoint: ORACLE - VAL-NORMAL"]
        self.assertEqual(ck.n_methods, 2)
        self.assertAlmostEqual(ck.mean_over_methods, 3.0)
        self.assertIn("ProtoGLAD (A20", ck.excluded)
        hp = p.loc["HP leak: VAL-NORMAL - HP-FREE"]
        self.assertEqual(hp.n_methods, 3)                              # ProtoGLAD enters the HP axis
        self.assertAlmostEqual(hp.mean_over_methods, (6 + 1 + 4) / 3)

    def test_identical_config_detected_and_excluded_from_hp_leak(self):
        runs = {}
        for m, same in (("LGKDE", True), ("GLADPro", False)):
            for f in range(2):
                for hp in ("ubgold", "hpfree"):
                    hpar = {"lr": 1e-3} if (same or hp == "ubgold") else {"lr": 1e-4}
                    run = self.run_of(method=m, fold=f, hp=hp, auroc=[0.6, 0.7], hold=[0.2, 0.3], hparams=hpar)
                    runs[("ext", run["run_id"])] = run
        det, summ = G.identical_config(runs, {"LGKDE", "GLADPro", "SUL-GUL"})
        s = summ.set_index("method")
        self.assertTrue(s.loc["LGKDE", "auto_identical_config"])
        self.assertFalse(s.loc["GLADPro", "auto_identical_config"])
        self.assertTrue(s.loc["SUL-GUL", "excluded_from_pooled_hp_leak"])     # declared (A21 §2) even without pairs
        spec = {"LGKDE": {"oracle": 80.0, "valnormal": 78.0, "hpfree_valnormal": 78.0},
                "GLADPro": {"oracle": 80.0, "valnormal": 78.0, "hpfree_valnormal": 74.0}}
        long = long_frame(spec)
        _, pooled = G.inflation(G.Cubes(long), G.Cubes(long.iloc[0:0]), long, 50, 1, {"LGKDE"})
        hp = pooled[(pooled.set == "UB9") & (pooled.pool == "E5") &
                    (pooled.comparison == "HP leak: VAL-NORMAL - HP-FREE")].iloc[0]
        self.assertEqual((hp.n_methods, hp.mean_over_methods), (1, 4.0))
        self.assertIn("LGKDE (A21", hp.excluded)

    def test_g2b_excludes_identical_config_frozen_method(self):
        long = long_frame(frozen_spec(lambda i: 4.0, lambda i: 4.0))
        C = G.Cubes(long)
        cov = G.coverage_table(long, EMPTY_DIV, G.DEEP12)
        out = G.g2b(C, 50, 1, False, cov, {"SIGNET"})
        self.assertEqual(out["excluded_A21"], ["SIGNET"])
        self.assertNotIn("SIGNET", out["methods"])
        self.assertEqual(out["status"], "PASS")

    def test_noise_estimate_counts_from_grid(self):
        runs, rows = {}, []
        deltas = {0: 0.0, 1: 0.0, 2: 1.0, 3: 3.0}
        for f, dlt in deltas.items():
            for hp in ("ubgold", "hpfree"):
                last = 0.7 + (dlt / 100 if hp == "hpfree" else 0.0)
                run = self.run_of(method="SUL-GUL", fold=f, hp=hp, auroc=[0.6, last], hold=[0.1, 0.3])
                runs[("ext", run["run_id"])] = run
                for b, e in (("last", 2), ("valnormal", 1)):
                    rows.append(mkrow(run, b, e, hp_alias=0))
        det, _ = G.identical_config(runs, {"SUL-GUL"})
        long = G.finalize(rows)
        nz = G.noise_estimate(long, det, ["SUL-GUL"]).iloc[0]
        self.assertEqual((nz.pairs, nz.bit_identical, nz.differing), (4, 2, 2))
        self.assertAlmostEqual(nz.last_absdiff_max, 3.0)
        self.assertAlmostEqual(nz.last_absdiff_median, 2.0)
        self.assertAlmostEqual(nz.valnormal_absdiff_max, 0.0)
        self.assertIn("lower bound on seed-to-seed variation", nz.label)


# ------------------------------------------------------------------------------------------------ sensitivity
class TestSensitivity(unittest.TestCase):
    def test_flagged_runs_dropped(self):
        spec = frozen_spec(lambda i: 4.0, lambda i: 4.0)
        long = long_frame(spec)
        m = long.method == "SIGNET"
        hit = m & (long.dataset == "AIDS") & (long.fold == 0) & (long.hp == "ubgold")
        long.loc[hit & (long.base == "valnormal"), "auroc"] = 30.0     # a chance-like outlier
        long.loc[hit & (long.base == "last"), "nonfinite_selected"] = 1
        s = G.sensitivity_drop(long, 50, 1, set())
        base = s[(s.variant == "drop nonfinite_selected") & (s.quantity == "per-method oracle-valnormal") &
                 (s.method == "SIGNET")].iloc[0]
        self.assertAlmostEqual(base.mean_over_methods, 4.0)            # the flagged fold is gone
        self.assertEqual(int(base.n_runs_dropped), 1)
        keep = G.sensitivity_drop(long.assign(nonfinite_selected=0), 50, 1, set())
        k = keep[(keep.quantity == "per-method oracle-valnormal") & (keep.method == "SIGNET")].iloc[0]
        self.assertGreater(k.mean_over_methods, 4.0)


# ------------------------------------------------------------------------------------------------ A9
class TestA9(unittest.TestCase):
    NEW8 = G.NEW8

    def verdict(self, deep, ctrl=None, counts=None, final=False, variants=("matched",), methods=None):
        t = F.typed_frame(deep, ctrl or {}, methods or self.NEW8, ms_variants=variants)
        t["auroc"] = t["auroc"] * 100
        t["src"] = "fixture"
        man = {(a, G.SIZE_VARIANT_ALIASES.get(v, v)): c for (a, v), c in (counts or {}).items()}
        for a in G.MS_TYPES:
            for v in ("matched", "changing"):
                man.setdefault((a, v), [200] * 5)
        T = G.type_table(t, None, man)
        gt = list(zip(T.generator, T.anomaly_type, T.size_variant))
        cells = G.typed_cells(t, self.NEW8, gt, "F2", 50, 1)
        return G.a9_verdict(cells, T, self.NEW8, final), T, cells

    def test_confirmed(self):
        v, T, _ = self.verdict({})
        self.assertEqual(v["verdict"], "CONFIRMED")
        self.assertEqual(v["n_valid_ms_types"], 3)
        self.assertTrue(v["complete"])

    def test_refuted(self):
        deep = {(m, "MS", t, "matched"): 70.0 for m in self.NEW8 for t in G.MS_TYPES}
        self.assertEqual(self.verdict(deep)[0]["verdict"], "REFUTED")

    def test_partial(self):
        deep = {(m, "MS", t, "matched"): 70.0 for m in self.NEW8[:4] for t in G.MS_TYPES}
        v = self.verdict(deep)[0]
        self.assertAlmostEqual(v["frac_blind_ms"], 0.5)
        self.assertEqual(v["verdict"], "PARTIAL")

    def test_confirmed_needs_5_of_8_methods_on_majority(self):
        # MS fraction 0.625 >= 0.60, but only 4 methods blind on a majority (2 of 3) of MS types
        deep = {}
        for m in self.NEW8[:4]:
            deep[(m, "MS", "MS1", "matched")] = 70.0
        for m in self.NEW8[4:]:
            deep[(m, "MS", "MS1", "matched")] = 70.0
            deep[(m, "MS", "MS2", "matched")] = 70.0
        v = self.verdict(deep)[0]
        self.assertAlmostEqual(v["frac_blind_ms"], (4 * 2 + 4 * 1) / 24)
        self.assertEqual(v["methods_blind_on_majority_of_ms"], 4)
        self.assertEqual(v["verdict"], "PARTIAL")

    def test_inconclusive_takes_precedence_over_confirmed(self):
        counts = {("MS2", "matched"): [156, 159, 162, 99, 167], ("MS3", "matched"): [142, 141, 98, 137, 153]}
        v, T, _ = self.verdict({}, counts=counts)
        self.assertEqual(v["n_valid_ms_types"], 1)
        self.assertEqual(v["verdict"], "INCONCLUSIVE")                  # all pairs blind, still INCONCLUSIVE
        self.assertEqual(T.set_index(["anomaly_type", "size_variant"]).loc[("MS2", "matched"), "feasible"], False)

    def test_feasibility_floor_a25(self):
        self.assertEqual(G.A9["min_anomalies"], 100)
        self.assertTrue(G.feasible_from_counts([156, 159, 162, 145, 167]))   # was infeasible under the old 150
        self.assertTrue(G.feasible_from_counts([127, 131, 148, 101, 138]))
        self.assertTrue(G.feasible_from_counts([100, 100, 100, 100, 100]))   # floor is inclusive
        self.assertFalse(G.feasible_from_counts([300, 300, 99, 300, 300]))   # any seed < 100
        self.assertIsNone(G.feasible_from_counts([200, 200]))               # fewer than 5 seeds known
        self.assertFalse(G.feasible_from_counts([200, 90]))

    def test_manifest_counts_used_status_field_ignored(self):
        import tempfile as _t
        d = _t.mkdtemp(dir=os.environ.get("GS_TEST_TMP") or None)
        try:
            yields = {("MS1", "matched"): [303, 307, 302, 305, 311], ("MS1", "sizechg"): [270, 277, 295, 222, 275],
                      ("MS2", "matched"): [156, 159, 162, 145, 167], ("MS2", "sizechg"): [143, 150, 160, 117, 155],
                      ("MS3", "matched"): [142, 141, 153, 137, 153], ("MS3", "sizechg"): [127, 131, 148, 101, 138]}
            p = F.manifest(osp.join(d, "m.json"), yields, stale_status=True)
            self.assertIn("INFEASIBLE", open(p).read())                      # frozen file still says so (150 floor)
            man = G.load_ms_manifest(p)
            self.assertEqual(man[("MS3", "changing")], [127, 131, 148, 101, 138])
            t = F.typed_frame({}, {}, self.NEW8, ms_variants=("matched", "changing"))
            t["auroc"] *= 100
            t["src"] = "fixture"
            T = G.type_table(t, None, man)
            ms = T[T.generator == "MS"]
            self.assertTrue((ms.feasible == True).all())                    # noqa: E712  all six cells feasible
            self.assertTrue((ms.counts_src == "manifest").all())
            cells = G.typed_cells(t, self.NEW8, list(zip(T.generator, T.anomaly_type, T.size_variant)), "F2", 50, 1)
            self.assertEqual(G.a9_verdict(cells, T, self.NEW8, False)["n_valid_ms_types"], 3)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_typed_n_anomalies_used_without_manifest(self):
        t = F.typed_frame({}, {}, self.NEW8, n_anom=99)
        t["auroc"] *= 100
        t["src"] = "fixture"
        T = G.type_table(t, None, {})
        m = T[(T.generator == "MS") & (T.size_variant == "matched")]
        self.assertTrue((m.feasible == False).all())                        # noqa: E712
        self.assertTrue((m.counts_src == "eval_typed n_anomalies").all())
        c = T[(T.generator == "MS") & (T.size_variant == "changing")]
        self.assertTrue(c.feasible.isna().all())                             # no rows, no manifest: undetermined

    def test_validity_and_confounding_controls(self):
        ctrl = {("MS", "MS2", "matched"): (95.0, 50.0, 90.0),      # edge-label-pair >= 85: confounded
                ("MS", "MS3", "matched"): (95.0, 65.0, 50.0),      # size-only > 60: invalid
                ("GX", "T1a", "matched"): (84.9, 50.0, 50.0),      # graphlet-count < 85: invalid
                ("GX", "T3a", "matched"): (85.0, 60.0, 84.9)}      # boundaries: valid and unconfounded
        v, T, _ = self.verdict({}, ctrl=ctrl)
        Ti = T.set_index(["generator", "anomaly_type", "size_variant"])
        self.assertTrue(Ti.loc[("MS", "MS2", "matched"), "confounded"])
        self.assertFalse(Ti.loc[("MS", "MS3", "matched"), "valid"])
        self.assertFalse(Ti.loc[("GX", "T1a", "matched"), "valid"])
        self.assertTrue(Ti.loc[("GX", "T3a", "matched"), "in_decision"])
        self.assertEqual(sorted(v["decision_types"]), ["GX:T3a", "GX:T3b", "MS:MS1"])
        self.assertEqual(v["verdict"], "INCONCLUSIVE")

    def test_blind_boundary_is_inclusive(self):
        deep = {(m, g, t, "matched"): 60.0 for m in self.NEW8 for g, ts in (("GX", G.GX_TYPES), ("MS", G.MS_TYPES))
                for t in ts}
        self.assertEqual(self.verdict(deep)[0]["frac_blind_pooled"], 1.0)
        deep = {k: 60.01 for k in deep}
        self.assertEqual(self.verdict(deep)[0]["frac_blind_pooled"], 0.0)

    def test_missing_cells_pending_vs_final(self):
        t = F.typed_frame({}, {}, self.NEW8[:7])               # InfoGraph_OCSVM has no typed cells
        t["auroc"] *= 100
        t["src"] = "fixture"
        man = {(a, v): [200] * 5 for a in G.MS_TYPES for v in ("matched", "changing")}
        T = G.type_table(t, None, man)
        cells = G.typed_cells(t, self.NEW8, list(zip(T.generator, T.anomaly_type, T.size_variant)), "F2", 50, 1)
        v = G.a9_verdict(cells, T, self.NEW8, False)
        self.assertFalse(v["complete"])
        self.assertEqual(v["n_pairs_missing"], 6)
        self.assertEqual(v["verdict"], "CONFIRMED")            # on the available cells; status would be PENDING
        vf = G.a9_verdict(cells, T, self.NEW8, True)
        self.assertAlmostEqual(vf["frac_blind_pooled"], 42 / 48)  # missing counted as not blind
        self.assertEqual(vf["methods_blind_on_majority_of_ms"], 7)

    def test_h4s_scope(self):
        deep = {(m, "MS", t, "changing"): 70.0 for m in self.NEW8 for t in G.MS_TYPES}
        v, T, cells = self.verdict(deep, variants=("matched", "changing"))
        h = G.h4s_scope(cells, T, self.NEW8)
        self.assertAlmostEqual(h["drop_pp"], 100.0)
        self.assertTrue(h["scope_to_size_preserving"])
        deep = {(m, "MS", t, "changing"): 70.0 for m in self.NEW8[:1] for t in G.MS_TYPES}
        v, T, cells = self.verdict(deep, variants=("matched", "changing"))
        h = G.h4s_scope(cells, T, self.NEW8)
        self.assertAlmostEqual(h["drop_pp"], 12.5)
        self.assertFalse(h["scope_to_size_preserving"])


# ------------------------------------------------------------------------------------------------ A17, A24, A12
class TestMisc(Tmp):
    def test_version_sensitivity_ignores_deep_rerun_rows(self):
        main = long_frame({"GLocalKD": {"oracle": 80.0, "last": 76.0, "valnormal": 77.0}})
        rows = []
        for ds in UB9:
            for f in range(5):
                for p, v, src in (("oracle", 81.0, "evalpaper_harness"), ("last", 76.0, "evalpaper_harness"),
                                  ("valnormal", 77.0, "evalpaper_harness"), ("oracle", 10.0, "deep_rerun")):
                    rows.append(dict(method="GLocalKD", dataset=ds, fold=f, protocol=p, auroc=v, auprc=v, fpr95=1,
                                     epoch_selected=1, run_id="x", source=src))
        pd.DataFrame(rows).to_csv(osp.join(self.tmp, "eval_version_a96137b_folds.csv"), index=False)
        args = types.SimpleNamespace(res=self.tmp)
        v, st = G.version_sensitivity(args, G.Cubes(main), 50, 1, [], {})
        o = v[(v.method == "GLocalKD") & (v.quantity == "oracle: a96137b - c940457")].iloc[0]
        self.assertAlmostEqual(o.mean_diff, 1.0)
        i = v[v.quantity == "inflation ORACLE-VALNORMAL: a96137b vs c940457"].iloc[0]
        self.assertAlmostEqual(i.mean_diff, 1.0)

    def test_glass_rows_are_not_transductive(self):
        long = long_frame({"GLASS": {"oracle": 80.0, "valnormal": 70.0}})
        self.assertFalse(long.transductive.any())
        self.assertIn("inductive", G.METHOD_NOTES["GLASS"])

    def test_diverged_registry_union_and_bak_ignored(self):
        os.makedirs(osp.join(self.tmp, "h", "jobs"))
        with open(osp.join(self.tmp, "h", "jobs", "tasks_gpu.txt"), "w") as fh:
            fh.write("GLADC AIDS 0 ubgold mem=1\n# DIVERGED per PREREG_AMENDMENTS A12: GLADC AIDS 3 ubgold mem=1\n")
        with open(osp.join(self.tmp, "h", "jobs", "tasks_gpu.txt.bak_before_A12"), "w") as fh:
            fh.write("# DIVERGED per PREREG_AMENDMENTS A12: GLADC AIDS 1 ubgold mem=1\n")
        reg = osp.join(self.tmp, "reg.csv")
        pd.DataFrame([dict(method="GLADC", dataset="AIDS", fold=4, hp="ubgold", evidence="x", amendment="A12")]).to_csv(reg, index=False)
        args = types.SimpleNamespace(diverged_registry=reg, harness_dir=osp.join(self.tmp, "h"),
                                     ext_dir=osp.join(self.tmp, "nonexistent"))
        d = G.load_diverged(args, [])
        self.assertEqual(sorted(d.fold), [3, 4])
        long = long_frame({"GLADC": {"oracle": 80.0, "last": 70.0, "valnormal": 72.0}}, datasets=["AIDS"],
                          folds=range(3))
        cov = G.coverage_table(long, d, ["GLADC"])
        c = cov[(cov.dataset == "AIDS") & (cov.hp == "ubgold")].iloc[0]
        self.assertEqual(c.status, "3/5 + DIVERGED 2")
        self.assertTrue(bool(c.resolved))


# ------------------------------------------------------------------------------------------------ end to end
class TestEndToEnd(unittest.TestCase):
    """A synthetic world through main(): frozen 12 (harness-format parts -> eval_folds via collect.py's run_rows),
    four real adapters on fixture parts (SEGO, GLADPro, SUL-GUL, ProtoGLAD; the other five run on zero parts),
    kernels, references, typed results (CONFIRMED), MolSwap manifest, A17 file, A12 registry."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="gs_e2e_", dir=TMP_BASE)
        t = cls.tmp
        hp_dir, ext, res, out = (osp.join(t, x) for x in ("harness_parts", "ext_parts", "res", "out"))
        for d in (res, out, osp.join(t, "dr"), osp.join(t, "hdir", "jobs")):
            os.makedirs(d, exist_ok=True)
        # frozen 12: VAL-NORMAL gap 4 (G1 passes), LAST gap 1 (fails) -> SELECTOR-DEPENDENT; HP-FREE 3 lower
        for i, m in enumerate(G.DEEP12):
            for ds in UB9:
                for f in range(5):
                    for hp, base in (("ubgold", 0.80 - 0.01 * i), ("hpfree", 0.77 - 0.01 * i)):
                        if m == "GLADC" and ds == "AIDS" and f >= 3 and hp == "ubgold":
                            continue                                      # DIVERGED (A12)
                        kw = dict(hparams={"lr": 1e-4 if hp == "ubgold" else 1e-3})
                        if m == "OCGIN" and ds == "DHFR" and f == 1 and hp == "hpfree":
                            a, h = F.shaped(base, 4.0, 1.0)
                            a[-1] = float("nan")
                            F.write_part(hp_dir, m, ds, f, hp, a, h, nonfinite=[0] * 9 + [5], seed=i, **kw)
                            continue
                        F.frozen_part(hp_dir, m, ds, f, hp, base, 4.0, 1.0, seed=i, **kw)
        F.write_eval_folds(hp_dir, res)
        for f in range(5):                                   # old-4 MUTAG run (A10 §5), deep_rerun format
            F.frozen_part(osp.join(t, "dr"), "SIGNET", "MUTAG", f, "ubgold", 0.8, 4.0, 1.0, run_id=f"SIGNET_MUTAG_fold{f}")
        with open(osp.join(t, "hdir", "jobs", "tasks_gpu.txt"), "w") as fh:
            fh.write("# DIVERGED per PREREG_AMENDMENTS A12: GLADC AIDS 3 ubgold mem=1\n"
                     "# DIVERGED per PREREG_AMENDMENTS A12: GLADC AIDS 4 ubgold mem=1\n")
        # E5/E4 fixtures through the real adapters
        for f in range(5):
            for hp in ("ubgold", "hpfree"):
                hpar = {"lr": 1e-4 if hp == "ubgold" else 1e-3}
                for ds in ("AIDS", "BZR"):
                    a, h = F.shaped(0.75, 3.0, 2.0)
                    F.write_part(ext, "SEGO", ds, f, hp, a, h, eval_freq=1, seed=f, hparams=hpar)
                for ds in UB9:
                    F.write_part(ext, "ProtoGLAD", ds, f, hp, [0.66 if hp == "ubgold" else 0.62], [0.2],
                                 traj_extra={"n_prototypes": 8}, seed=f, hparams=hpar)
                a = [0.6, 0.75, 0.99, 0.7]          # GLADPro's NaN->0 own rule still picks epoch 3 (mean 0.792)
                nf = [0, 0, 0, 0]
                if f == 2 and hp == "ubgold":
                    a[2], nf[2] = float("nan"), 3
                F.write_part(ext, "GLADPro", "AIDS", f, hp, a, [0.4, 0.3, 0.2, 0.1], nonfinite=nf,
                             traj_extra={"n_prot": 4}, seed=f, hparams=hpar)
                same = f < 3
                last = 0.7 if (hp == "ubgold" or same) else 0.72
                F.write_part(ext, "SUL-GUL", "AIDS", f, hp, [0.6, last], [0.1, 0.3], traj_extra={"auroc_gul": 0.5},
                             seed=f if same else f + (100 if hp == "hpfree" else 0), hparams={"lr": 1e-3})
        F.kernels_csv(res)
        F.grail0_csv(res)
        typed = F.typed_frame({}, {}, G.NEW8)
        typed = pd.concat([typed, typed[typed.method == "GraphDE"].assign(orientation="ubgold", auroc=0.99)])
        typed.loc[typed.orientation.isna(), "orientation"] = "paper"
        typed.to_csv(osp.join(res, "eval_typed_fixture.csv"), index=False)
        F.manifest(osp.join(t, "manifest.json"))
        vrows = [dict(method="GLocalKD", dataset=ds, fold=f, protocol=p, auroc=v, auprc=v, fpr95=10, epoch_selected=1,
                      run_id="v", source="evalpaper_harness") for ds in UB9 for f in range(5)
                 for p, v in (("oracle", 81.0), ("last", 76.0), ("valnormal", 77.0))]
        pd.DataFrame(vrows).to_csv(osp.join(res, "eval_version_a96137b_folds.csv"), index=False)
        reg = osp.join(t, "reg.csv")
        pd.DataFrame(columns=["method", "dataset", "fold", "hp", "evidence", "amendment"]).to_csv(reg, index=False)
        # previous outputs that must survive
        with open(osp.join(out, "eval_leaderboard_long.csv"), "w") as fh:
            fh.write("v1 output\n")
        with open(osp.join(out, "REPORT.md"), "w") as fh:
            fh.write("previous report\n")
        cls.res = G.main(["--res", res, "--out", out, "--report", osp.join(out, "REPORT.md"), "--harness-parts", hp_dir,
                          "--ext-parts", ext, "--dr-parts", osp.join(t, "dr"), "--harness-dir", osp.join(t, "hdir"),
                          "--ext-dir", osp.join(t, "noext"), "--diverged-registry", reg,
                          "--protoglad-grid", osp.join(t, "nogrid"), "--ms-manifest", osp.join(t, "manifest.json"),
                          "--tmp", t, "--B", "50"])
        cls.out = out

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def tab(self, k):
        return pd.read_csv(osp.join(self.out, f"eval_leaderboard_v2_{k}.csv"))

    def test_gates(self):
        pr = json.load(open(osp.join(self.out, "eval_leaderboard_v2_prereg.json")))
        self.assertEqual(pr["G1"]["valnormal"]["status"], "PASS")
        self.assertEqual(pr["G1"]["last"]["status"], "FAIL")
        self.assertEqual(pr["G1"]["primary"], "SELECTOR-DEPENDENT")
        self.assertEqual(pr["G1"]["valnormal"]["n_complete"], 11)       # GLADC: diverged UB9 cell
        self.assertEqual(pr["G2b"]["status"], "PASS")
        self.assertEqual(pr["G2c"]["status"], "PASS")
        self.assertEqual(pr["G2c"]["verdict"], "CONFIRMED")
        self.assertEqual(pr["G2"], "PASS")
        self.assertEqual(pr["GO"], "NO-GO")

    def test_outputs_versioned_previous_kept(self):
        self.assertEqual(open(osp.join(self.out, "eval_leaderboard_long.csv")).read(), "v1 output\n")
        arch = os.listdir(osp.join(self.out, "archive"))
        self.assertTrue(any(a.startswith("REPORT_generated_") for a in arch))
        rep = open(osp.join(self.out, "REPORT.md")).read()
        self.assertIn("statistics report v2", rep)
        self.assertIn("SELECTOR-DEPENDENT", rep)

    def test_coverage_shows_diverged(self):
        c = self.tab("coverage")
        r = c[(c.method == "GLADC") & (c.dataset == "AIDS") & (c.hp == "ubgold")].iloc[0]
        self.assertEqual(r.status, "3/5 + DIVERGED 2")
        r = c[(c.method == "OCGIN") & (c.dataset == "DHFR") & (c.hp == "hpfree")].iloc[0]
        self.assertEqual(r.status, "5/5 [nonfinite_selected: f1]")                 # A15 §5: listed in coverage

    def test_a15_rows(self):
        f = self.tab("folds")
        r = f[(f.method == "OCGIN") & (f.dataset == "DHFR") & (f.fold == 1) & (f.protocol == "hpfree_last")].iloc[0]
        self.assertEqual((r.auroc, r.fpr95, r.nonfinite_selected), (50.0, 95.0, 1))
        g = f[(f.method == "GLADPro") & (f.protocol == "oracle")]
        self.assertEqual(len(g), 5)
        self.assertTrue((g.a15_action == "A15.1_reselected_common_epoch").all())
        self.assertTrue((g.epoch_selected == 2).all())
        self.assertTrue((g.auroc.round(6) == 75.0).all())
        h = f[(f.method == "GLADPro") & (f.protocol == "hpfree_oracle")]
        self.assertTrue((h.a15_action.fillna("") == "").all() and (h.epoch_selected == 3).all())  # no violation
        fl = self.tab("flags")
        self.assertTrue(((fl.method == "OCGIN") & (fl.nonfinite_selected == 1)).any())

    def test_sensitivity_reports_dropped_runs(self):
        s = self.tab("sensitivity")
        self.assertTrue((s[s.variant == "drop nonfinite_selected"].n_runs_dropped >= 1).all())

    def test_a20_a21_and_noise(self):
        ip = self.tab("inflation_pooled")
        r = ip[(ip.set == "UB9") & (ip.pool == "frozen12+E5+E4") & (ip.comparison == "checkpoint: ORACLE - VAL-NORMAL")]
        self.assertIn("ProtoGLAD (A20", r.excluded.iloc[0])
        self.assertNotIn("ProtoGLAD", r.methods.iloc[0])
        h = ip[(ip.set == "UB9") & (ip.pool == "E4") & (ip.comparison == "HP leak: VAL-NORMAL - HP-FREE")].iloc[0]
        self.assertIn("ProtoGLAD", h.methods)                             # A20: HP axis only
        s = self.tab("identical_config_summary").set_index("method")
        self.assertTrue(bool(s.loc["SUL-GUL", "excluded_from_pooled_hp_leak"]))
        nz = self.tab("noise_estimate").set_index("method")
        self.assertEqual((nz.loc["SUL-GUL", "pairs"], nz.loc["SUL-GUL", "bit_identical"]), (5, 3))
        self.assertAlmostEqual(nz.loc["SUL-GUL", "last_absdiff_max"], 2.0, places=3)
        hp = self.tab("hpleak").set_index("method")
        self.assertIn("A21", hp.loc["SUL-GUL", "excluded_from_pooled"])

    def test_labels_and_orientation(self):
        lab = self.tab("labels").set_index("method")
        self.assertEqual(lab.loc["SUL-GUL", "method_label"], "SUL-GUL (GUL only; SUL not released)")
        self.assertFalse(lab.transductive.any())
        f = self.tab("folds")
        self.assertIn("released-modal", f[f.method == "SEGO"].hp_label.iloc[0] + "|" +
                      "|".join(f[(f.method == "SEGO") & (f.hp == "ubgold")].hp_label.unique()))
        self.assertTrue((self.tab("repro_only").method == "GraphDE").all())
        o = f[(f.method == "GLADPro") & (f.base == "oracle")].oracle_rule.iloc[0]
        self.assertTrue(o.startswith("own"))

    def test_mutag_old4_from_deep_rerun_parts(self):
        f = self.tab("folds")
        m = f[(f.method == "SIGNET") & (f.dataset == "MUTAG")]
        self.assertEqual(sorted(m.protocol.unique()), ["last", "oracle", "oracle_ubgold", "valnormal"])
        self.assertTrue(m.hp_label.str.startswith("modal UB-GOLD configuration").all())
        self.assertTrue((m.source == "deep_rerun_mutag").all())

    def test_version_rows(self):
        v = self.tab("version")
        self.assertTrue(((v.method == "GLocalKD") & v.complete).any())


if __name__ == "__main__":
    unittest.main()
