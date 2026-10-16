"""Unit tests of collect_typed.run_rows on synthetic trajectories (no real data, no detector).
  PYTHONDONTWRITEBYTECODE=1 $GLAD_ENVS/grail-prelim/bin/python -B -m unittest test_collect_typed  (in typed/)
"""
import sys

sys.dont_write_bytecode = True

import math
import os.path as osp
import unittest

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import collect_typed as CT  # noqa: E402

NAN = float("nan")


def met(a):
    return dict(auroc=a, auprc=(a if a == a else NAN), fpr95=(1 - a if a == a else NAN), nonfinite=0 if a == a else 5,
                n_distinct=500)


def part(method, aurocs, holds, sel=None, ub=None, hold_ub=None, ne=12, ef=4, gen="GX", key="T1a"):
    epochs = sorted(set(range(ef, ne + 1, ef)) | {ne})
    assert len(epochs) == len(aurocs) == len(holds)
    traj = []
    for i, ep in enumerate(epochs):
        r = dict(epoch=ep, types={key: met(aurocs[i])}, holdout_score=holds[i], train_loss=1.0)
        if sel is not None:
            r["types_sel"] = {key: met(sel[i])}
        if ub is not None:
            r["types_ubgold"] = {key: met(ub[i])}
            r["holdout_score_ubgold"] = hold_ub[i]
        traj.append(r)
    cell = dict(dataset=gen, anomaly_type=key, size_variant="matched", feature_variant="F2", n_anomalies=100,
                n_normals=300)
    meta = dict(method=method, generator=gen, seed=0, feature_variant="F2", run_id=f"{method}_{gen}_F2_s0",
                hparams=dict(eval_freq=ef, num_epoch=ne), groups={key: cell}, hp_line="x", device="cpu")
    return dict(meta=meta, traj=traj, train_loss=[1.0] * ne)


def rows_by(rows, crit, orient=None):
    out = [r for r in rows if r["criterion"] == crit and (orient is None or r["orientation"] == orient)]
    assert len(out) == 1
    return out[0]


class TestCollect(unittest.TestCase):
    def test_plain(self):
        R = CT.run_rows(part("OCGIN", [0.6, 0.8, 0.7], [3.0, 1.0, 2.0]), "t")
        self.assertAlmostEqual(rows_by(R, "oracle")["auroc"], 80.0)
        self.assertEqual(rows_by(R, "oracle")["epoch_selected"], 8)
        self.assertEqual(rows_by(R, "last")["epoch_selected"], 12)
        self.assertEqual(rows_by(R, "valnormal")["epoch_selected"], 8)
        self.assertTrue(all(r["nonfinite_selected"] == 0 for r in R))

    def test_last_nonfinite_is_chance(self):     # A15 rule 3
        r = rows_by(CT.run_rows(part("OCGIN", [0.6, 0.8, NAN], [3.0, 1.0, 2.0]), "t"), "last")
        self.assertEqual((r["auroc"], r["fpr95"], r["nonfinite_selected"]), (50.0, 95.0, 1))
        self.assertAlmostEqual(r["auprc"], 25.0)    # anomaly share 100 / 400

    def test_oracle_skips_nonfinite(self):       # A15 rule 1
        r = rows_by(CT.run_rows(part("OCGIN", [0.6, NAN, 0.7], [3.0, 1.0, 2.0]), "t"), "oracle")
        self.assertEqual(r["epoch_selected"], 12)

    def test_valnormal_nonfinite_test_is_chance(self):   # FROZEN interp. 2a
        r = rows_by(CT.run_rows(part("OCGIN", [0.6, NAN, 0.7], [3.0, 1.0, 2.0]), "t"), "valnormal")
        self.assertEqual((r["epoch_selected"], r["auroc"], r["nonfinite_selected"]), (8, 50.0, 1))

    def test_valnormal_ignores_nonfinite_holdout(self):  # A15 rule 2, ties -> earliest
        r = rows_by(CT.run_rows(part("OCGIN", [0.6, 0.8, 0.7], [NAN, 2.0, 2.0]), "t"), "valnormal")
        self.assertEqual(r["epoch_selected"], 8)

    def test_all_nonfinite(self):                # A15 rule 4
        R = CT.run_rows(part("OCGIN", [NAN, NAN, NAN], [1.0, 2.0, 3.0]), "t")
        self.assertTrue(all(r["auroc"] == 50.0 and r["nonfinite_selected"] == 1 for r in R))
        self.assertTrue(all(r["a15_action"] == "A15.4_all_points_nonfinite" for r in R))

    def test_cvtgad_selects_on_normalised(self):  # A4
        P = part("CVTGAD", [0.9, 0.6, 0.7], [3.0, 1.0, 2.0], sel=[0.5, 0.95, 0.6])
        r = rows_by(CT.run_rows(P, "t"), "oracle")
        self.assertEqual((r["epoch_selected"], r["auroc"]), (8, 60.0))

    def test_cvtgad_sel_point_with_nonfinite_raw_skipped(self):
        P = part("CVTGAD", [0.9, NAN, 0.7], [3.0, 1.0, 2.0], sel=[0.5, 0.95, 0.6])
        r = rows_by(CT.run_rows(P, "t"), "oracle")
        self.assertEqual(r["epoch_selected"], 12)

    def test_graphde_two_orientations(self):     # A5
        P = part("GraphDE", [0.7, 0.8, 0.6], [3.0, 1.0, 2.0], ub=[0.3, 0.2, 0.4], hold_ub=[0.1, 0.5, 0.9])
        R = CT.run_rows(P, "t")
        self.assertEqual(rows_by(R, "valnormal", "paper")["epoch_selected"], 8)
        self.assertEqual(rows_by(R, "valnormal", "ubgold")["epoch_selected"], 4)
        self.assertAlmostEqual(rows_by(R, "oracle", "ubgold")["auroc"], 40.0)

    def test_patience_and_eval_freq_filter(self):
        ne, ef = 130, 1
        a = [0.5 + 0.001 * i if i < 10 else 0.4 for i in range(ne)]
        r = rows_by(CT.run_rows(part("OCGIN", a, [1.0] * ne, ne=ne, ef=ef), "t"), "oracle")
        self.assertEqual((r["epoch_selected"], r["ubgold_stop_epoch"]), (10, 40))
        # final epoch not a multiple of eval_freq is never an ORACLE candidate
        r = rows_by(CT.run_rows(part("OCGIN", [0.5, 0.6, 0.99], [1.0, 1.0, 1.0], ne=10, ef=4), "t"), "oracle")
        self.assertEqual(r["epoch_selected"], 8)

    def test_incomplete_raises(self):
        P = part("OCGIN", [0.6, 0.8, 0.7], [3.0, 1.0, 2.0])
        P["traj"] = P["traj"][:2]
        with self.assertRaises(ValueError):
            CT.run_rows(P, "t")


if __name__ == "__main__":
    unittest.main()
