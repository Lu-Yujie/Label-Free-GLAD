# Label-free GLAD re-evaluation: statistics report (AUTO-GENERATED)

Generated 2026-10-12 22:10:23 by `prelim/evalpaper/stats/glad_stats.py` (B = 4000 bootstrap replicates, seed 20261012). Do not edit by hand; rerun the script. All numbers are AUROC × 100 unless noted; a leaderboard value is the mean over the datasets of the set of the 5-fold means. CIs are 95% percentile intervals from paired fold- and dataset-bootstraps (fold-bootstrap CIs are optimistic because folds share training data; Tox21 'folds' are 5 seeds on one fixed split).

## 1. Inputs and coverage
- deep_rerun_folds.csv: 540 rows, methods ['GLocalKD', 'OCGIN', 'OCGTL', 'SIGNET']
- eval_folds.csv: not present yet (8 additional detectors and all HP-FREE runs PENDING)
- eval_kernels_folds.csv: 520 train-fit rows, methods ['PK-IF', 'PK-OCSVM', 'WL-IF', 'WL-OCSVM']
- grail0_folds.csv: references ['GRAIL-0', 'WL-hist+iForest (ctrl)', 'graphlet-count+LW', 'graphlet-count+iForest', 'size-only']
- m1_selection_folds.csv: 2205 rows, criteria ['aucp', 'ckpt_ensemble', 'consensus', 'last', 'oracle', 'oracle_nostop', 'pseudo', 'pseudo_ensemble', 'pseudo_labelnoise', 'pseudo_motif', 'pseudo_rewire', 'pseudo_roleswap', 'valnormal']
- Protocol mapping for methods without checkpoints: KernelGLAD 'tuned' (test-selected HP) → oracle/last/valnormal, 'default' (UB-GOLD defaults) → hpfree_*; GRAIL-0 and controls are a-priori and identical in every protocol. Only train-fit KernelGLAD rows are ranked (`@testfit` = UB-GOLD's transductive code path, reported in §8).
- Deep-detector UB9 coverage (datasets with all 5 folds):
  - SIGNET: oracle 9/9, valnormal 9/9, last 9/9, hpfree_valnormal 0/9
  - OCGTL: oracle 9/9, valnormal 9/9, last 9/9, hpfree_valnormal 0/9
  - OCGIN: oracle 9/9, valnormal 9/9, last 9/9, hpfree_valnormal 0/9
  - GLocalKD: oracle 9/9, valnormal 9/9, last 9/9, hpfree_valnormal 0/9
  - GLADC: oracle 0/9, valnormal 0/9, last 0/9, hpfree_valnormal 0/9
  - CVTGAD: oracle 0/9, valnormal 0/9, last 0/9, hpfree_valnormal 0/9
  - GOOD-D: oracle 0/9, valnormal 0/9, last 0/9, hpfree_valnormal 0/9
  - GraphDE: oracle 0/9, valnormal 0/9, last 0/9, hpfree_valnormal 0/9
  - GraphCL_IF: oracle 0/9, valnormal 0/9, last 0/9, hpfree_valnormal 0/9
  - GraphCL_OCSVM: oracle 0/9, valnormal 0/9, last 0/9, hpfree_valnormal 0/9
  - InfoGraph_IF: oracle 0/9, valnormal 0/9, last 0/9, hpfree_valnormal 0/9
  - InfoGraph_OCSVM: oracle 0/9, valnormal 0/9, last 0/9, hpfree_valnormal 0/9

## 2. Automatic PREREG evaluation (prelim/evalpaper/PREREG.md, go/no-go 2026-11-08)

Rule: PASS/FAIL once all 12 deep detectors are complete for the criterion, or with --final (on the 2026-11-08 decision date); otherwise PENDING with the current value.

- **G1: PENDING** — coverage condition NOT met; inflation conditions met on the complete methods. Complete deep methods on UB9: 4/12 (≥ 8 required): SIGNET, OCGTL, OCGIN, GLocalKD.
  - Mean ORACLE − VAL-NORMAL over these methods: 3.88 (fold CI 3.24–4.54; dataset CI 2.69–5.20); threshold ≥ 3.0.
  - Methods with ≥ 2.0: 4/4 (100%; threshold ≥ 75%). Per method: SIGNET 3.15, OCGTL 3.21, OCGIN 5.34, GLocalKD 3.84.
- **G2(a): PENDING** — currently NOT met. Kendall τ-b(ORACLE, VAL-NORMAL) on UB9 = 1.00 over 4 deep methods (threshold ≤ 0.8); top-3 ORACLE ['SIGNET', 'OCGTL', 'OCGIN'] vs VAL-NORMAL ['SIGNET', 'OCGTL', 'OCGIN'] (changed: False).
- **G2(b): PENDING**. No deep method has complete UB9 HP-FREE (hpfree_valnormal) runs yet (0/12).
- **G2(c) [POST-HOC]: PENDING** — currently MET. POST-HOC criterion (revised after M2 showed T2 is detectable; not an independent confirmation). Deep methods with VAL-NORMAL ≤ 60 on all of T1a/T3a/T3b (F2): 4/4 measured (4/4 of the study's deep methods; with --final the latter is used) (100%; threshold ≥ 75%); graphlet-count + LW: T1a 100.0, T3a 90.1, T3b 91.2 (all ≥ 85: True). Per method: GLocalKD 50.1/51.5/51.6; OCGIN 48.1/52.9/51.6; OCGTL 52.8/54.0/54.8; SIGNET 49.8/50.5/52.9.
  - Sensitivity (F1, labels only; not the criterion): 75% of methods ≤ 60 on all three; failing: ['OCGIN'].
- **G2 overall: PENDING; decision: PENDING.**

## 3a. Leaderboards on UB9 (AIDS, BZR, COX2, DHFR, ENZYMES, NCI1, PROTEINS_full, IMDB-BINARY, REDDIT-BINARY)

Mean AUROC per protocol (methods with incomplete coverage of the set are marked `*` and are not ranked; their value averages only the datasets they cover).

| method | family | oracle | last | valnormal | hpfree_oracle | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|
| SIGNET | deep | 74.72 (#1) | 70.71 (#1) | 71.57 (#1) | – | – | – |
| OCGTL | deep | 72.06 (#2) | 68.65 (#2) | 68.85 (#2) | – | – | – |
| OCGIN | deep | 70.15 (#3) | 64.47 (#3) | 64.81 (#3) | – | – | – |
| GLocalKD | deep | 66.36 (#4) | 62.51 (#4) | 62.52 (#4) | – | – | – |
| WL-IF | kernel | 58.65 (#7) | 58.65 (#7) | 58.65 (#7) | 56.79 (#5) | 56.79 (#5) | 56.79 (#5) |
| PK-IF | kernel | 54.53 (#11) | 54.53 (#11) | 54.53 (#11) | 50.93 (#7) | 50.93 (#7) | 50.93 (#7) |
| PK-OCSVM | kernel | 48.81 (#12) | 48.81 (#12) | 48.81 (#12) | 47.19 (#8) | 47.19 (#8) | 47.19 (#8) |
| WL-OCSVM | kernel | 43.22 (#13) | 43.22 (#13) | 43.22 (#13) | 41.97 (#9) | 41.97 (#9) | 41.97 (#9) |
| GRAIL-0 | reference | 62.10 (#5) | 62.10 (#5) | 62.10 (#5) | 62.10 (#1) | 62.10 (#1) | 62.10 (#1) |
| graphlet-count+LW | reference | 60.39 (#6) | 60.39 (#6) | 60.39 (#6) | 60.39 (#2) | 60.39 (#2) | 60.39 (#2) |
| graphlet-count+iForest | reference | 58.19 (#8) | 58.19 (#8) | 58.19 (#8) | 58.19 (#3) | 58.19 (#3) | 58.19 (#3) |
| size-only | reference | 57.25 (#9) | 57.25 (#9) | 57.25 (#9) | 57.25 (#4) | 57.25 (#4) | 57.25 (#4) |
| WL-hist+iForest (ctrl) | reference | 55.81 (#10) | 55.81 (#10) | 55.81 (#10) | 55.81 (#6) | 55.81 (#6) | 55.81 (#6) |

95% CIs for the label-free main protocol (valnormal; fold bootstrap / dataset bootstrap):

| method | family | AUROC | fold lo | fold hi | ds lo | ds hi | AUPRC | FPR95 |
|---|---|---|---|---|---|---|---|---|
| SIGNET | deep | 71.57 | 70.16 | 72.87 | 65.36 | 78.35 | 67.95 | 71.24 |
| OCGTL | deep | 68.85 | 67.50 | 70.21 | 59.35 | 79.58 | 69.38 | 70.22 |
| OCGIN | deep | 64.81 | 63.39 | 66.21 | 55.65 | 75.63 | 65.94 | 79.29 |
| GLocalKD | deep | 62.52 | 60.57 | 64.39 | 54.56 | 71.41 | 62.03 | 78.00 |
| GRAIL-0 | reference | 62.10 | 60.52 | 63.77 | 52.61 | 73.59 | 63.99 | 80.18 |
| graphlet-count+LW | reference | 60.39 | 59.14 | 61.66 | 49.59 | 72.38 | 61.94 | 81.36 |
| WL-IF | kernel | 58.65 | 57.76 | 59.53 | 48.97 | 69.48 | 59.55 | 84.97 |
| graphlet-count+iForest | reference | 58.19 | 56.59 | 59.82 | 48.13 | 69.47 | 58.94 | 82.62 |
| size-only | reference | 57.25 | 55.69 | 58.77 | 47.63 | 69.75 | 61.55 | 83.37 |
| WL-hist+iForest (ctrl) | reference | 55.81 | 54.86 | 56.81 | 47.60 | 64.52 | 55.37 | 89.45 |
| PK-IF | kernel | 54.53 | 52.98 | 56.09 | 51.24 | 58.21 | 52.18 | 91.88 |
| PK-OCSVM | kernel | 48.81 | 47.43 | 50.32 | 42.33 | 54.47 | 49.66 | 94.68 |
| WL-OCSVM | kernel | 43.22 | 42.13 | 44.38 | 31.02 | 54.20 | 50.02 | 93.54 |

## 3b. Leaderboards on secondary (DD, COLLAB, Tox21_p53, Tox21_MMP)

Mean AUROC per protocol (methods with incomplete coverage of the set are marked `*` and are not ranked; their value averages only the datasets they cover).

| method | family | oracle | last | valnormal | hpfree_oracle | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|
| WL-IF | kernel | 57.65 (#1) | 57.65 (#1) | 57.65 (#1) | 54.49 (#1) | 54.49 (#1) | 54.49 (#1) |
| PK-IF | kernel | 49.42 (#6) | 49.42 (#6) | 49.42 (#6) | 47.82 (#6) | 47.82 (#6) | 47.82 (#6) |
| PK-OCSVM | kernel | 44.84 (#8) | 44.84 (#8) | 44.84 (#8) | 44.69 (#8) | 44.69 (#8) | 44.69 (#8) |
| WL-OCSVM | kernel | 41.19 (#9) | 41.19 (#9) | 41.19 (#9) | 37.77 (#9) | 37.77 (#9) | 37.77 (#9) |
| WL-hist+iForest (ctrl) | reference | 53.92 (#2) | 53.92 (#2) | 53.92 (#2) | 53.92 (#2) | 53.92 (#2) | 53.92 (#2) |
| graphlet-count+iForest | reference | 51.39 (#3) | 51.39 (#3) | 51.39 (#3) | 51.39 (#3) | 51.39 (#3) | 51.39 (#3) |
| size-only | reference | 50.73 (#4) | 50.73 (#4) | 50.73 (#4) | 50.73 (#4) | 50.73 (#4) | 50.73 (#4) |
| GRAIL-0 | reference | 49.81 (#5) | 49.81 (#5) | 49.81 (#5) | 49.81 (#5) | 49.81 (#5) | 49.81 (#5) |
| graphlet-count+LW | reference | 45.44 (#7) | 45.44 (#7) | 45.44 (#7) | 45.44 (#7) | 45.44 (#7) | 45.44 (#7) |

95% CIs for the label-free main protocol (valnormal; fold bootstrap / dataset bootstrap):

| method | family | AUROC | fold lo | fold hi | ds lo | ds hi | AUPRC | FPR95 |
|---|---|---|---|---|---|---|---|---|
| WL-IF | kernel | 57.65 | 56.88 | 58.41 | 54.20 | 62.56 | 40.21 | 94.06 |
| WL-hist+iForest (ctrl) | reference | 53.92 | 53.08 | 54.75 | 50.40 | 58.33 | 37.67 | 95.16 |
| graphlet-count+iForest | reference | 51.39 | 50.54 | 52.35 | 45.59 | 57.20 | 36.16 | 91.65 |
| size-only | reference | 50.73 | 50.05 | 51.37 | 41.88 | 59.58 | 35.06 | 89.52 |
| GRAIL-0 | reference | 49.81 | 49.02 | 50.60 | 44.87 | 52.54 | 33.98 | 92.00 |
| PK-IF | kernel | 49.42 | 48.10 | 50.76 | 48.26 | 50.00 | 32.93 | 96.87 |
| graphlet-count+LW | reference | 45.44 | 44.47 | 46.26 | 36.34 | 54.53 | 30.81 | 92.98 |
| PK-OCSVM | kernel | 44.84 | 44.06 | 45.42 | 34.51 | 50.00 | 31.04 | 99.59 |
| WL-OCSVM | kernel | 41.19 | 40.57 | 41.93 | 27.18 | 55.21 | 29.45 | 91.62 |

## 4. Oracle-minus-label-free inflation per method

| set | method | comparison | mean | fold lo | fold hi | ds lo | ds hi | #ds ≥ 2 | max ds | max fold |
|---|---|---|---|---|---|---|---|---|---|---|
| UB9 | PK-IF | HP leak: tuned - default | 3.60 | 2.49 | 4.78 | 0.79 | 7.30 | 3 | 15.69 | 29.04 |
| UB9 | PK-OCSVM | HP leak: tuned - default | 1.62 | 0.40 | 2.79 | 0.41 | 3.07 | 2 | 5.39 | 12.78 |
| UB9 | WL-IF | HP leak: tuned - default | 1.86 | 1.33 | 2.35 | 1.18 | 2.64 | 3 | 3.90 | 6.62 |
| UB9 | WL-OCSVM | HP leak: tuned - default | 1.24 | 0.94 | 1.55 | 0.68 | 2.07 | 1 | 4.18 | 4.70 |
| UB9 | GLocalKD | checkpoint: ORACLE - LAST | 3.85 | 2.73 | 5.00 | 1.61 | 6.33 | 5 | 10.72 | 21.45 |
| UB9 | OCGIN | checkpoint: ORACLE - LAST | 5.69 | 4.70 | 6.66 | 3.66 | 7.86 | 8 | 12.51 | 17.56 |
| UB9 | OCGTL | checkpoint: ORACLE - LAST | 3.41 | 2.07 | 4.81 | 1.56 | 5.41 | 5 | 9.28 | 25.92 |
| UB9 | SIGNET | checkpoint: ORACLE - LAST | 4.01 | 2.52 | 5.42 | 0.52 | 7.52 | 4 | 13.34 | 22.14 |
| UB9 | GLocalKD | checkpoint: ORACLE - VAL-NORMAL | 3.84 | 2.71 | 4.98 | 1.61 | 6.32 | 5 | 10.72 | 21.45 |
| UB9 | OCGIN | checkpoint: ORACLE - VAL-NORMAL | 5.34 | 4.38 | 6.31 | 3.25 | 7.54 | 7 | 12.40 | 17.56 |
| UB9 | OCGTL | checkpoint: ORACLE - VAL-NORMAL | 3.21 | 1.86 | 4.62 | 1.29 | 5.34 | 4 | 9.28 | 25.92 |
| UB9 | SIGNET | checkpoint: ORACLE - VAL-NORMAL | 3.15 | 1.81 | 4.45 | -0.19 | 6.70 | 5 | 13.38 | 17.96 |
| UB9 | GLocalKD | early stop: ORACLE - max-over-run | -0.00 | -0.01 | -0.00 | -0.01 | -0.00 | 0 | 0.00 | 0.00 |
| UB9 | OCGIN | early stop: ORACLE - max-over-run | -0.21 | -0.39 | -0.06 | -0.42 | -0.06 | 0 | 0.00 | 0.00 |
| UB9 | OCGTL | early stop: ORACLE - max-over-run | -0.46 | -0.79 | -0.17 | -0.92 | -0.14 | 0 | 0.00 | 0.00 |
| UB9 | SIGNET | early stop: ORACLE - max-over-run | -2.18 | -3.18 | -1.26 | -3.51 | -0.98 | 0 | 0.00 | 0.00 |
| secondary | PK-IF | HP leak: tuned - default | 1.60 | 0.72 | 2.33 | 0.00 | 4.81 | 1 | 6.41 | 10.46 |
| secondary | PK-OCSVM | HP leak: tuned - default | 0.14 | -0.26 | 0.47 | 0.00 | 0.43 | 0 | 0.57 | 2.48 |
| secondary | WL-IF | HP leak: tuned - default | 3.15 | 2.25 | 3.96 | 2.18 | 4.21 | 3 | 4.68 | 7.08 |
| secondary | WL-OCSVM | HP leak: tuned - default | 3.43 | 3.32 | 3.53 | 0.42 | 6.44 | 2 | 7.68 | 7.68 |

## 5. Agreement between protocol leaderboards (Kendall τ-b, top-3 sets)

| set | pool | A | B | k | τ-b | fold lo | fold hi | ds lo | ds hi | top-3 changed | P(change) fold | P(change) ds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UB9 | deep | oracle | valnormal | 4 | 1.00 | 0.67 | 1.00 | 0.33 | 1.00 | False | 0.02 | 0.13 |
| UB9 | all | oracle | valnormal | 13 | 1.00 | 0.97 | 1.00 | 0.82 | 1.00 | False | 0.02 | 0.27 |
| UB9 | deep | oracle | last | 4 | 1.00 | 0.67 | 1.00 | 0.33 | 1.00 | False | 0.04 | 0.17 |
| UB9 | all | oracle | last | 13 | 1.00 | 0.95 | 1.00 | 0.79 | 1.00 | False | 0.04 | 0.33 |
| UB9 | deep | valnormal | last | 4 | 1.00 | 0.67 | 1.00 | 0.67 | 1.00 | False | 0.02 | 0.05 |
| UB9 | all | valnormal | last | 13 | 1.00 | 0.97 | 1.00 | 0.95 | 1.00 | False | 0.02 | 0.06 |
| UB9 | all | oracle | hpfree_valnormal | 9 | 0.89 | 0.83 | 1.00 | 0.61 | 1.00 | True | 0.64 | 0.41 |
| UB9 | all | valnormal | hpfree_valnormal | 9 | 0.89 | 0.83 | 1.00 | 0.61 | 1.00 | True | 0.64 | 0.41 |
| secondary | all | oracle | hpfree_valnormal | 9 | 1.00 | 0.83 | 1.00 | 0.43 | 1.00 | False | 0.01 | 0.20 |
| secondary | all | valnormal | hpfree_valnormal | 9 | 1.00 | 0.83 | 1.00 | 0.43 | 1.00 | False | 0.01 | 0.20 |

Top-3 sets: UB9/deep oracle: SIGNET|OCGTL|OCGIN → valnormal: SIGNET|OCGTL|OCGIN; UB9/all oracle: SIGNET|OCGTL|OCGIN → valnormal: SIGNET|OCGTL|OCGIN

## 6. Friedman / Iman–Davenport, Holm-corrected Wilcoxon and Nemenyi critical differences

| set | pool | protocol | k | N | χ² | p | Iman–Davenport p | CD (α=0.05) |
|---|---|---|---|---|---|---|---|---|
| UB9 | deep | oracle | 4 | 9 | 4.07 | 0.25 | 0.26 | 1.56 |
| UB9 | all | oracle | 13 | 9 | 49.64 | 1.6e-06 | 9.8e-09 | 6.08 |
| UB9 | deep | last | 4 | 9 | 6.73 | 0.081 | 0.071 | 1.56 |
| UB9 | all | last | 13 | 9 | 34.67 | 5.3e-04 | 1.0e-04 | 6.08 |
| UB9 | deep | valnormal | 4 | 9 | 6.73 | 0.081 | 0.071 | 1.56 |
| UB9 | all | valnormal | 13 | 9 | 34.12 | 6.5e-04 | 1.4e-04 | 6.08 |
| UB9 | all | hpfree_oracle | 9 | 9 | 8.82 | 0.36 | 0.36 | 4.00 |
| UB9 | all | hpfree_last | 9 | 9 | 8.82 | 0.36 | 0.36 | 4.00 |
| UB9 | all | hpfree_valnormal | 9 | 9 | 8.82 | 0.36 | 0.36 | 4.00 |
| secondary | all | oracle | 9 | 4 | 8.44 | 0.39 | 0.41 | 6.01 |
| secondary | all | last | 9 | 4 | 8.44 | 0.39 | 0.41 | 6.01 |
| secondary | all | valnormal | 9 | 4 | 8.44 | 0.39 | 0.41 | 6.01 |
| secondary | all | hpfree_oracle | 9 | 4 | 10.52 | 0.23 | 0.22 | 6.01 |
| secondary | all | hpfree_last | 9 | 4 | 10.52 | 0.23 | 0.22 | 6.01 |
| secondary | all | hpfree_valnormal | 9 | 4 | 10.52 | 0.23 | 0.22 | 6.01 |

Critical-difference tables (average rank over datasets, 1 = best; methods sharing a letter are not separated by the Nemenyi CD):

*UB9 / deep / oracle* (CD = 1.56): OCGIN 2.11[a], SIGNET 2.22[a], OCGTL 2.44[a], GLocalKD 3.22[a]

*UB9 / all / oracle* (CD = 6.08): SIGNET 2.78[a], OCGTL 3.00[a], OCGIN 3.22[a], GLocalKD 5.33[ab], GRAIL-0 6.56[ab], PK-IF 7.56[ab], graphlet-count+LW 7.78[ab], graphlet-count+iForest 8.44[ab], size-only 8.44[ab], WL-IF 8.56[ab], PK-OCSVM 8.67[ab], WL-hist+iForest (ctrl) 10.22[b], WL-OCSVM 10.44[b]

*UB9 / deep / valnormal* (CD = 1.56): OCGTL 1.89[a], SIGNET 2.11[a], OCGIN 2.67[a], GLocalKD 3.33[a]

*UB9 / all / valnormal* (CD = 6.08): OCGTL 3.00[a], SIGNET 3.44[a], OCGIN 5.22[ab], GRAIL-0 5.78[ab], GLocalKD 6.44[ab], graphlet-count+LW 7.11[ab], PK-IF 7.33[ab], WL-IF 8.00[ab], graphlet-count+iForest 8.11[ab], size-only 8.22[ab], PK-OCSVM 8.33[ab], WL-hist+iForest (ctrl) 9.67[b], WL-OCSVM 10.33[b]

*UB9 / all / hpfree_valnormal* (CD = 4.00): GRAIL-0 3.22[a], graphlet-count+LW 4.44[a], PK-IF 4.67[a], size-only 4.67[a], graphlet-count+iForest 4.89[a], PK-OCSVM 5.22[a], WL-hist+iForest (ctrl) 5.44[a], WL-IF 5.78[a], WL-OCSVM 6.67[a]

*secondary / all / oracle* (CD = 6.01): WL-IF 2.00[a], WL-hist+iForest (ctrl) 4.00[a], graphlet-count+LW 4.50[a], graphlet-count+iForest 5.00[a], size-only 5.00[a], GRAIL-0 5.75[a], WL-OCSVM 6.00[a], PK-IF 6.12[a], PK-OCSVM 6.62[a]

*secondary / all / valnormal* (CD = 6.01): WL-IF 2.00[a], WL-hist+iForest (ctrl) 4.00[a], graphlet-count+LW 4.50[a], graphlet-count+iForest 5.00[a], size-only 5.00[a], GRAIL-0 5.75[a], WL-OCSVM 6.00[a], PK-IF 6.12[a], PK-OCSVM 6.62[a]

*secondary / all / hpfree_valnormal* (CD = 6.01): WL-IF 3.00[a], WL-hist+iForest (ctrl) 3.25[a], graphlet-count+LW 4.00[a], graphlet-count+iForest 5.00[a], size-only 5.00[a], GRAIL-0 5.25[a], PK-IF 5.38[a], PK-OCSVM 5.88[a], WL-OCSVM 8.25[a]

Holm-significant pairwise Wilcoxon differences (α = 0.05): 2 of 576 pairs (full list in `eval_leaderboard_posthoc.csv`; with N = 9 datasets the smallest attainable two-sided Wilcoxon p is 0.0039, so Holm over many pairs rarely rejects).

## 7. Label-free checkpoint selectors (M1 runs, UB-GOLD HP; GLocalKD on 8 datasets)

| criterion | AUROC (4-method mean) | − VAL-NORMAL | fold lo | fold hi | ds lo | ds hi |
|---|---|---|---|---|---|---|
| oracle | 70.37 | 4.26 | 3.50 | 5.00 | 2.77 | 5.88 |
| valnormal | 66.11 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| last | 65.72 | -0.39 | -0.60 | -0.20 | -0.66 | -0.16 |
| pseudo | 66.01 | -0.10 | -0.76 | 0.55 | -1.23 | 0.94 |
| pseudo_ensemble | 66.11 | -0.00 | -0.50 | 0.51 | -1.01 | 0.97 |
| ckpt_ensemble | 65.78 | -0.32 | -0.57 | -0.07 | -0.90 | 0.24 |
| consensus | 65.60 | -0.51 | -0.76 | -0.25 | -1.03 | 0.03 |
| aucp | 65.84 | -0.27 | -1.33 | 0.83 | -2.95 | 2.24 |
| oracle_nostop | 71.15 | 5.04 | 4.35 | 5.72 | 3.64 | 6.48 |

Fraction of each method's ORACLE − VAL-NORMAL gap recovered (1 = ORACLE, 0 = VAL-NORMAL): aucp: GLocalKD +0.66, OCGIN +0.40, OCGTL -0.05, SIGNET -1.96; ckpt_ensemble: GLocalKD +0.14, OCGIN +0.10, OCGTL -0.53, SIGNET -0.28; consensus: GLocalKD +0.13, OCGIN +0.06, OCGTL -0.51, SIGNET -0.44; last: GLocalKD +0.00, OCGIN -0.06, OCGTL -0.10, SIGNET -0.28; oracle_nostop: GLocalKD +1.00, OCGIN +1.03, OCGTL +1.25, SIGNET +1.65; pseudo: GLocalKD +0.16, OCGIN +0.10, OCGTL -0.23, SIGNET -0.33; pseudo_ensemble: GLocalKD +0.20, OCGIN +0.13, OCGTL -0.22, SIGNET -0.34

## 8. KernelGLAD (UB-GOLD's WL/PK + IF/OCSVM)

5-fold means per dataset. Rows without `@testfit`: kernel and detector fitted on the normal training graphs (same protocol as every other method; these rows are ranked above). `@testfit`: UB-GOLD's own code path, which fits kernel and detector on the unlabelled test set (scores paired with the correct labels). 'tuned' = configuration with the best mean TEST AUROC over UB-GOLD's KernelGLAD search space (oracle HP); 'default' = UB-GOLD defaults = the released `WL_IF.sh` line. § = constant kernel matrix (PK on graphs with constant UB-GOLD node features), AUROC 50 by construction.

*UB9*:

| method | protocol | AIDS | BZR | COX2 | DHFR | ENZYMES | NCI1 | PROTEINS_full | IMDB-BINARY | REDDIT-BINARY | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PK-IF | default | 38.48 | 39.79 | 57.82 | 45.90 | 62.06 | 58.23 | 56.14 | 50.00§ | 50.00§ | 50.93 |
| PK-IF | tuned | 54.16 | 49.00 | 58.44 | 47.07 | 62.19 | 60.21 | 59.74 | 50.00§ | 50.00§ | 54.53 |
| PK-IF@testfit | default | 28.30 | 41.17 | 46.60 | 43.23 | 59.02 | 56.39 | 32.46 | 50.00§ | 50.00§ | 45.24 |
| PK-IF@testfit | tuned | 32.61 | 45.50 | 52.56 | 45.11 | 60.64 | 56.39 | 34.14 | 50.00§ | 50.00§ | 47.44 |
| PK-OCSVM | default | 29.98 | 48.48 | 54.51 | 47.68 | 60.15 | 54.33 | 29.58 | 50.00§ | 50.00§ | 47.19 |
| PK-OCSVM | tuned | 35.38 | 53.71 | 56.16 | 47.68 | 60.50 | 55.82 | 30.02 | 50.00§ | 50.00§ | 48.81 |
| PK-OCSVM@testfit | default | 23.46 | 41.07 | 44.81 | 43.29 | 58.90 | 55.34 | 28.42 | 50.00§ | 50.00§ | 43.92 |
| PK-OCSVM@testfit | tuned | 29.24 | 42.52 | 44.81 | 44.17 | 59.92 | 56.13 | 29.56 | 50.00§ | 50.00§ | 45.15 |
| WL-IF | default | 87.90 | 34.82 | 54.26 | 39.81 | 60.25 | 48.94 | 56.66 | 55.32 | 73.13 | 56.79 |
| WL-IF | tuned | 91.80 | 37.58 | 55.15 | 41.16 | 61.90 | 50.09 | 60.17 | 56.36 | 73.61 | 58.65 |
| WL-IF@testfit | default | 72.01 | 45.67 | 58.71 | 43.13 | 58.82 | 45.07 | 32.53 | 57.17 | 74.56 | 54.19 |
| WL-IF@testfit | tuned | 78.77 | 46.26 | 60.62 | 44.20 | 60.99 | 46.30 | 34.71 | 58.95 | 75.18 | 56.22 |
| WL-OCSVM | default | 4.21 | 51.14 | 47.03 | 32.73 | 57.56 | 66.30 | 18.95 | 46.96 | 52.89 | 41.97 |
| WL-OCSVM | tuned | 8.39 | 51.70 | 47.99 | 34.49 | 58.83 | 67.20 | 19.49 | 47.52 | 53.33 | 43.22 |
| WL-OCSVM@testfit | default | 3.76 | 41.81 | 41.02 | 32.66 | 57.07 | 66.40 | 19.94 | 44.34 | 52.53 | 39.95 |
| WL-OCSVM@testfit | tuned | 7.41 | 43.39 | 41.39 | 34.19 | 58.62 | 67.35 | 20.08 | 44.34 | 52.74 | 41.06 |

*secondary*:

| method | protocol | DD | COLLAB | Tox21_p53 | Tox21_MMP | mean |
|---|---|---|---|---|---|---|
| PK-IF | default | 41.27 | 50.00§ | 50.00§ | 50.00§ | 47.82 |
| PK-IF | tuned | 47.68 | 50.00§ | 50.00§ | 50.00§ | 49.42 |
| PK-IF@testfit | default | 30.77 | 50.00§ | 50.00§ | 50.00§ | 45.19 |
| PK-IF@testfit | tuned | 32.44 | 50.00§ | 50.00§ | 50.00§ | 45.61 |
| PK-OCSVM | default | 28.77 | 50.00§ | 50.00§ | 50.00§ | 44.69 |
| PK-OCSVM | tuned | 29.34 | 50.00§ | 50.00§ | 50.00§ | 44.84 |
| PK-OCSVM@testfit | default | 28.12 | 50.00§ | 50.00§ | 50.00§ | 44.53 |
| PK-OCSVM@testfit | tuned | 29.72 | 50.00§ | 50.00§ | 50.00§ | 44.93 |
| WL-IF | default | 63.24 | 54.33 | 51.82 | 48.58 | 54.49 |
| WL-IF | tuned | 65.04 | 57.15 | 55.14 | 53.26 | 57.65 |
| WL-IF@testfit | default | 31.55 | 49.55 | 45.94 | 44.02 | 42.76 |
| WL-IF@testfit | tuned | 32.93 | 52.23 | 46.79 | 44.65 | 44.15 |
| WL-OCSVM | default | 18.18 | 35.34 | 49.71 | 47.83 | 37.77 |
| WL-OCSVM | tuned | 18.59 | 35.77 | 54.90 | 55.51 | 41.19 |
| WL-OCSVM@testfit | default | 18.76 | 34.71 | 46.07 | 39.84 | 34.84 |
| WL-OCSVM@testfit | tuned | 18.93 | 34.95 | 49.01 | 44.62 | 36.88 |

UB-GOLD's verbatim WL-IF pipeline (`WL_IF.sh`; scores fitted on the test set) with the released label pairing vs the correct pairing of the same scores (`eval_kernels_ubgold_verbatim.csv`):

| dataset | folds | AUROC released pairing | AUROC aligned | share of labels mismatched | max |K_ubgold − K_ours| |
|---|---|---|---|---|---|
| AIDS | 5 | 50.63 | 71.50 | 0.33 | 0.00 |
| BZR | 5 | 46.31 | 45.68 | 0.33 | 0.00 |
| COLLAB | 5 | 49.55 | 50.49 | 0.50 | 0.00 |
| COX2 | 5 | 54.64 | 53.75 | 0.36 | 0.00 |
| DD | 5 | 48.20 | 30.90 | 0.48 | 0.00 |
| DHFR | 5 | 49.69 | 43.64 | 0.43 | 0.00 |
| ENZYMES | 5 | 53.15 | 58.19 | 0.26 | 0.00 |
| IMDB-BINARY | 5 | 51.45 | 57.46 | 0.47 | 0.00 |
| NCI1 | 5 | 50.41 | 45.46 | 0.50 | 0.00 |
| PROTEINS_full | 5 | 49.48 | 32.41 | 0.48 | 0.00 |
| REDDIT-BINARY | 5 | 45.89 | 74.81 | 0.51 | 0.00 |
| Tox21_MMP | 5 | 50.00 | 44.46 | 0.27 | 0.00 |
| Tox21_p53 | 5 | 48.33 | 47.47 | 0.19 | 0.00 |

UB-GOLD's published kernel AUROCs (Table 3, arXiv v2; `eval_kernels_published.csv`) against the chance band of a 5-fold mean under random score/label pairing: IF: 21/26 cells within |z| ≤ 3; OCSVM: 24/26 cells within |z| ≤ 3. Cells with |z| > 3: COLLAB WL-OCSVM 54.62 (z +5.7), PROTEINS_full PK-IF 61.24 (z +6.4), PROTEINS_full WL-IF 65.75 (z +8.9), DD PK-IF 75.29 (z +14.8), DD WL-IF 70.49 (z +12.0), AIDS WL-IF 61.42 (z +7.1), NCI1 WL-OCSVM 54.18 (z +4.6).

Score orientation (fixed from UB-GOLD KernelGLAD.py:139 `-decision_function` and the sklearn docs; `eval_kernels_orientation_check.csv`, fold 0, train-fit, defaults): label-free, OCSVM flags 0.068–0.203 of the training normals (nu = 0.1) and its decision value rises with a test graph's mean kernel similarity to the training graphs (Spearman 0.65–0.99). Sub-50 OCSVM AUROCs therefore mean that the anomalous class is MORE similar to the training normals (explanatory, label-using columns: OCSVM AUROC vs AUROC of −mean similarity, Pearson r = 0.95 over 17 dataset × kernel cells), not a sign error.

## 9. Output files

- `prelim/results/eval_leaderboard_cd.csv`
- `prelim/results/eval_leaderboard_coverage.csv`
- `prelim/results/eval_leaderboard_friedman.csv`
- `prelim/results/eval_leaderboard_inflation.csv`
- `prelim/results/eval_leaderboard_kendall.csv`
- `prelim/results/eval_leaderboard_long.csv`
- `prelim/results/eval_leaderboard_per_dataset.csv`
- `prelim/results/eval_leaderboard_posthoc.csv`
- `prelim/results/eval_leaderboard_prereg.json`
- `prelim/results/eval_leaderboard_selectors.csv`
