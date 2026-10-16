# Results

All tables are CSV files with one row per fold, run or aggregate. Most files give AUROC, AUPRC and FPR95 in percent;
a few inputs (for example `grail0_folds.csv`) use fractions, which the statistics code rescales. The files are
byte-identical to the ones the study produced, except that `eval_leaderboard_v2_diverged.csv` shows job identifiers
as `<jobid>`.

| Files | Written by | Contents |
| --- | --- | --- |
| `eval_folds.csv`, `eval_folds_extra.csv`, `eval_coverage.csv`, `eval_repro.csv` | `evalpaper/harness/collect.py` | Protocol rows of the 12 UB-GOLD detectors per (dataset, fold, setting), extra rows, coverage and reproduction columns |
| `eval_kernels_*.csv` | `evalpaper/kernels/` | Kernel baselines: per-fold rows, summary, UB-GOLD's own transductive path and published numbers |
| `grail0_folds.csv`, `grail0_summary.csv` | `grail0/run.py` | Training-free references |
| `eval_typed.csv`, `eval_typed_ext.csv`, `eval_typed_controls_msv2.csv` | `evalpaper/typed/` | Typed suite: 12 UB-GOLD detectors, nine recent methods, controls |
| `retired_msv1/` | `evalpaper/typed/controls_typed.py` | Controls of the retired MolSwap v1 suite, kept as a record |
| `eval_version_a96137b_folds.csv` | `evalpaper/version_a96137b/` | A17 version-sensitivity runs |
| `eval_orientation_*.csv`, `eval_ablation_xs_ones_*.csv` | `evalpaper/followups/` | A33 runs (flipped orientation, x_s ablation) |
| `deep_rerun_*.csv`, `m1_selection_folds.csv`, `m2_*.csv` | `deep_rerun/`, `m1_selection/`, `m2_typed/` | Pilot re-run, first label-free selectors, typed pilot |
| `eval_leaderboard_v2_*.csv`, `eval_leaderboard_v2_prereg.json` | `evalpaper/stats/glad_stats.py --final` | Leaderboards per protocol, inflation, hyperparameter leak, typed test, flags, coverage and the pre-registered decisions |
| `eval_posthoc_*.csv` | `evalpaper/stats/posthoc_robustness.py` | Post hoc robustness analyses (A32) |
| `eval_a33_*.csv` | `evalpaper/stats/a33_followups.py` | A33 follow-up summaries (A35) |

The ORACLE columns use test labels and are diagnostic upper bounds only.
