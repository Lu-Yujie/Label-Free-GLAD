# Post hoc robustness analyses (A32): report (AUTO-GENERATED)

Generated 2026-10-14 21:28:08 EDT by `evalpaper/stats/posthoc_robustness.py` (sha256 `b771868b94459777f884ce1e5bea015ae13e2aa91e08e32621052824b01ac580`). Do not edit by hand. Specification: PREREG_AMENDMENTS.md A32 (written 2026-10-14 20:55:17 EDT, before any of these analyses was computed). **All analyses here are post hoc robustness or mechanism analyses, not confirmatory tests, and none changes G1, G2 or A9.**

Frozen `stats/glad_stats.py` imported read-only (sha256 `2731b02f07519f8bbd97305018fd04ff3771ca3f71e4b1d86b6d7a34d5802c78`, equal to FROZEN.md). AUROC × 100. Dataset-bootstrap CIs: 95% percentile, B = 4000, glad_stats `Boot` with seed 20261012 (the same dataset resamples as REPORT.md). Environment: {'python': '3.12.6', 'numpy': '2.5.4', 'pandas': '3.0.6', 'scipy': '1.18.1', 'sklearn': '1.9.1'}. Cache: `$GLAD_DATA_ROOT/posthoc_cache` (one JSON per run with the sha256 of its part JSON and score npz).

## Summary: one sentence per A32 item (post hoc, descriptive; numbers from the tables below)

- **Item 1.** For the UB-GOLD detectors the pooled ORACLE − selector gap was 4.13 (S2) to 5.22 (S6) over S1–S6 (S3 on 8 of the 12 detectors) and 10.81 for S7 (only 1 detector records a standardised held-out score), and every pooled 95% CI of the seven selectors lay above 0 (recent methods: 6.69–7.97, every CI above 0); the gap is therefore not specific to VAL-NORMAL or LAST among these pre-specified selectors, which does not show that no better label-free selector exists.
- **Item 2(a).** Under permuted test labels, taking the maximum over checkpoints gave on average 3.28 AUROC points over LAST for the 8 UB-GOLD detectors with stored scores (observed ORACLE − LAST 5.56, ratio 0.59) and 5.34 for the 8 recent methods (observed 6.69, ratio 0.80), and for 6 of 16 methods the null was at least as large as the observed gap, so a substantial part of the observed gap is of the size that test-label selection produces on noise alone (the null uses the plain maximum, which can only overstate the null of the patience-30 rule).
- **Item 2(b).** Across runs, ORACLE − LAST rose with the spread of the test-AUROC trajectory (Spearman ρ = 0.42 for UB-GOLD detectors, 0.28 for recent methods), and showed no clear association with the number of evaluation points for UB-GOLD detectors (ρ = -0.05) and a weak association for recent methods (ρ = 0.17); this is descriptive and mixes between- and within-method variation.
- **Item 3.** Leaving out any one UB9 dataset kept the two G1 means within 3.96–4.41 (vs VAL-NORMAL) and 4.85–5.46 (vs LAST), all above 3.0, and the detector-mean inflation was positive on all 9 datasets for both comparisons (smallest on REDDIT-B, 1.86); treating the detectors as a sample as well widens the 95% CIs to [2.92, 5.36] and [3.63, 6.90], so the claim should be stated for the evaluated detectors across the UB9 datasets.
- **Item 4.** 6 of 9 UB9 datasets met the 30% matching rule (AIDS, BZR, COX2 did not); on their matched subsets the size-only reference (the matching check) gave DHFR 47.4, ENZYMES 51.1, NCI1 51.1, PROTEINS 51.8, IMDB-B 51.0, REDDIT-B 66.7 AUROC, and the mean AUROC of the 8 UB-GOLD detectors with scores fell from 61.1 to 53.8 at VAL-NORMAL and from 65.5 to 56.2 at ORACLE (ORACLE − VAL-NORMAL 4.4 → 2.4); matched subsets are not random samples of the test sets, so these changes are descriptive.
- **Item 5.** Only LGKDE falls in the 'reproduction not met' tier; without it the pooled recent-method inflation is 8.16 [6.71, 9.58] vs VAL-NORMAL and 7.32 [5.73, 8.93] vs LAST (all eight: 7.47 and 6.69), and also dropping GLADPro and SUL-GUL (contested or not evaluable) gives 6.38 [5.39, 7.43] and 5.05 [3.65, 6.69].

## 0. Sanity check against glad_stats (REPORT.md)

- Re-derived from the parts trajectories with glad_stats' own selection functions, 12 UB-GOLD detectors, UB9: ORACLE − VAL-NORMAL = **4.1303** [3.32, 4.79] (REPORT.md: 4.1303 [3.32, 4.79]); ORACLE − LAST = **5.1195** [3.97, 6.24] (REPORT.md: 5.1195 [3.97, 6.24]).
- Run by run, against `eval_leaderboard_v2_folds.csv` (oracle_ubgold, last, valnormal; written with 4 decimals): max |ΔAUROC| = 5.00e-05, runs with |Δ| > 1e-4: 0, runs whose selected epoch differs: 0, runs missing in glad_stats: 0 (12 UB-GOLD detectors and 8 iterative recent methods, 900 runs; ProtoGLAD has no checkpoint axis).
- On the runs that have per-graph scores (the 8 harness detectors; SIGNET, OCGTL, OCGIN, GLocalKD use deep_rerun runs without scores) the same pooled values are ORACLE − VAL-NORMAL 4.25 [3.31, 5.05] and ORACLE − LAST 5.56 [4.22, 6.90]; the difference from 4.13 / 5.12 is coverage (4 detectors fewer), not computation.
- Rank-sum AUROC from the stored per-graph scores reproduces the recorded trajectory AUROC at every finite point of every scored run: max |Δ| = 2.84e-14 AUROC points.

## Coverage

Primary set (A32): UB9 × 'ubgold' × 5 folds; 12 UB-GOLD detectors and the recent methods. Runs without per-graph scores are listed, never imputed.

| method | stratum | parts | analysed | with scores | S1 | S2 | S3 | S4 | S5 | S6 | S7 | perm. null | size-match | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SIGNET | frozen12 | 45 | 45 | 0 | 45 | 45 | 0 | 45 | 45 | 45 | 0 | 0 | 0 | deep_rerun parts (trajectory only, no per-graph scores) |
| OCGTL | frozen12 | 45 | 45 | 0 | 45 | 45 | 0 | 45 | 45 | 45 | 0 | 0 | 0 | deep_rerun parts (trajectory only, no per-graph scores) |
| OCGIN | frozen12 | 45 | 45 | 0 | 45 | 45 | 0 | 45 | 45 | 45 | 0 | 0 | 0 | deep_rerun parts (trajectory only, no per-graph scores) |
| GLocalKD | frozen12 | 45 | 45 | 0 | 45 | 45 | 0 | 45 | 45 | 45 | 0 | 0 | 0 | deep_rerun parts (trajectory only, no per-graph scores) |
| GLADC | frozen12 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | harness parts + npz |
| CVTGAD | frozen12 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 43 | harness parts + npz |
| GOOD-D | frozen12 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | harness parts + npz |
| GraphDE | frozen12 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | harness parts + npz |
| GraphCL_IF | frozen12 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | harness parts + npz |
| GraphCL_OCSVM | frozen12 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | harness parts + npz |
| InfoGraph_IF | frozen12 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | harness parts + npz |
| InfoGraph_OCSVM | frozen12 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | harness parts + npz |
| MUSE | E5 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | ext parts + npz |
| LGKDE | E5 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | ext parts + npz |
| GLADPro | E5 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | ext parts + npz |
| GLADMamba | E5 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 43 | ext parts + npz |
| SEGO | E5 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 43 | ext parts + npz |
| GLASS | E4 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 43 | ext parts + npz |
| ProtoGLAD | E4 | 45 | 45 | 45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 43 | ext parts + npz; A20: no checkpoint axis, item 4 only |
| HC-GLAD | E4 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | ext parts + npz |
| SUL-GUL | E4 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 45 | 0 | 45 | 43 | ext parts + npz |

- Runs without per-graph scores (not imputed): SIGNET, OCGTL, OCGIN, GLocalKD × all 9 UB9 datasets × 5 folds = 180 deep_rerun runs (GPU runs of the pilot; trajectories only). They enter items 1 (S1, S2, S4, S5, S6), 2(b) and 3, but not S3, S7, 2(a) or 4.
- Runs without a part file in the canonical list: 0.
- ProtoGLAD (no checkpoint axis, A20) is outside items 1–3 and 2(a); its scores enter item 4.

## 1. Label-free selector robustness

Δ_s = ORACLE (UB-GOLD rule) − selector s, per run; per detector the mean over UB9 of 5-fold means; pooled = mean over detectors with complete UB9 coverage for s. 95% dataset-bootstrap CIs.

**Pool frozen12**

| s | selector | detectors | Δ_s | CI lo | CI hi | CI > 0 |
|---|---|---|---|---|---|---|
| S1 | LAST | 12 | 5.12 | 3.97 | 6.24 | yes |
| S2 | VAL-NORMAL (min mean held-out score) | 12 | 4.13 | 3.32 | 4.79 | yes |
| S3 | min median held-out score | 8 | 4.15 | 3.22 | 4.98 | yes |
| S4 | min training loss | 12 | 5.10 | 3.88 | 6.25 | yes |
| S5 | point closest to half the epoch budget | 12 | 5.21 | 4.00 | 6.30 | yes |
| S6 | uniform checkpoint (trajectory mean) | 12 | 5.22 | 4.34 | 6.08 | yes |
| S7 | min mean standardised held-out score | 1 | 10.81 | 6.28 | 16.34 | yes |

Coverage-matched view (same 8 detectors as S3: GLADC, CVTGAD, GOOD-D, GraphDE, GraphCL_IF, GraphCL_OCSVM, InfoGraph_IF, InfoGraph_OCSVM):

| s | detectors | Δ_s | CI lo | CI hi |
|---|---|---|---|---|
| S1 | 8 | 5.56 | 4.22 | 6.90 |
| S2 | 8 | 4.25 | 3.31 | 5.05 |
| S4 | 8 | 5.58 | 4.20 | 6.97 |
| S5 | 8 | 5.53 | 4.20 | 6.77 |
| S6 | 8 | 5.31 | 4.36 | 6.30 |

**Pool recent**

| s | selector | detectors | Δ_s | CI lo | CI hi | CI > 0 |
|---|---|---|---|---|---|---|
| S1 | LAST | 8 | 6.69 | 5.75 | 7.61 | yes |
| S2 | VAL-NORMAL (min mean held-out score) | 8 | 7.47 | 6.15 | 8.83 | yes |
| S3 | min median held-out score | 8 | 7.42 | 6.30 | 8.65 | yes |
| S4 | min training loss | 8 | 6.75 | 5.70 | 7.71 | yes |
| S5 | point closest to half the epoch budget | 8 | 7.67 | 6.06 | 9.33 | yes |
| S6 | uniform checkpoint (trajectory mean) | 8 | 7.34 | 6.08 | 8.66 | yes |
| S7 | min mean standardised held-out score | 3 | 7.97 | 6.45 | 9.60 | yes |

**Pool frozen12+recent**

| s | selector | detectors | Δ_s | CI lo | CI hi | CI > 0 |
|---|---|---|---|---|---|---|
| S1 | LAST | 20 | 5.75 | 4.75 | 6.67 | yes |
| S2 | VAL-NORMAL (min mean held-out score) | 20 | 5.47 | 4.83 | 6.02 | yes |
| S3 | min median held-out score | 16 | 5.79 | 5.17 | 6.36 | yes |
| S4 | min training loss | 20 | 5.76 | 4.78 | 6.64 | yes |
| S5 | point closest to half the epoch budget | 20 | 6.19 | 5.15 | 7.20 | yes |
| S6 | uniform checkpoint (trajectory mean) | 20 | 6.07 | 5.27 | 6.88 | yes |
| S7 | min mean standardised held-out score | 4 | 8.68 | 6.48 | 11.18 | yes |

Coverage-matched view (same 16 detectors as S3: GLADC, CVTGAD, GOOD-D, GraphDE, GraphCL_IF, GraphCL_OCSVM, InfoGraph_IF, InfoGraph_OCSVM, MUSE, LGKDE, GLADPro, GLADMamba, SEGO, GLASS, HC-GLAD, SUL-GUL):

| s | detectors | Δ_s | CI lo | CI hi |
|---|---|---|---|---|
| S1 | 16 | 6.13 | 5.14 | 7.04 |
| S2 | 16 | 5.86 | 5.08 | 6.55 |
| S4 | 16 | 6.17 | 5.20 | 7.03 |
| S5 | 16 | 6.60 | 5.52 | 7.64 |
| S6 | 16 | 6.33 | 5.49 | 7.14 |

A32 rule for the summary sentence (all seven pooled Δ_s have CIs above 0): UB-GOLD detectors **met**; recent methods **met**. S7 is pooled over 1 UB-GOLD detector(s) and 3 recent methods only (see coverage).

Per detector Δ_s (CIs in `eval_posthoc_selectors.csv`; '–' = selector not available):

| method | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|---|---|---|---|---|---|---|---|
| SIGNET | 4.01 | 3.15 | – | 4.22 | 5.11 | 6.33 | – |
| OCGTL | 3.41 | 3.21 | – | 3.13 | 4.65 | 5.35 | – |
| OCGIN | 5.69 | 5.34 | – | 5.33 | 5.10 | 5.10 | – |
| GLocalKD | 3.85 | 3.84 | – | 3.86 | 3.44 | 3.40 | – |
| GLADC | 5.07 | 4.82 | 4.68 | 4.99 | 4.47 | 4.16 | – |
| CVTGAD | 5.12 | 5.02 | 5.28 | 5.02 | 6.17 | 6.56 | 10.81 |
| GOOD-D | 4.55 | 4.65 | 4.73 | 4.91 | 6.70 | 6.23 | – |
| GraphDE | 3.38 | 2.15 | 2.09 | 3.36 | 2.54 | 3.01 | – |
| GraphCL_IF | 3.99 | 3.10 | 3.51 | 4.43 | 4.10 | 3.93 | – |
| GraphCL_OCSVM | 6.37 | 6.00 | 5.30 | 6.04 | 6.71 | 5.75 | – |
| InfoGraph_IF | 9.67 | 5.32 | 5.24 | 9.59 | 8.34 | 7.74 | – |
| InfoGraph_OCSVM | 6.33 | 2.96 | 2.40 | 6.33 | 5.18 | 5.11 | – |
| MUSE | 7.68 | 6.57 | 6.27 | 8.31 | 6.66 | 6.24 | – |
| LGKDE | 2.29 | 2.65 | 2.13 | 2.13 | 6.03 | 4.79 | – |
| GLADPro | 18.16 | 18.68 | 18.36 | 20.41 | 22.39 | 20.19 | – |
| GLADMamba | 4.38 | 10.44 | 10.10 | 4.50 | 4.82 | 5.20 | 10.44 |
| SEGO | 4.38 | 8.25 | 6.64 | 4.22 | 6.57 | 6.58 | 8.25 |
| GLASS | 6.26 | 5.23 | 6.78 | 5.19 | 6.69 | 6.18 | 5.23 |
| HC-GLAD | 2.56 | 1.44 | 2.04 | 1.45 | 2.06 | 2.22 | – |
| SUL-GUL | 7.85 | 6.49 | 7.00 | 7.77 | 6.14 | 7.31 | – |

Per-detector CIs entirely below 0: none. Per-detector CIs including 0: SIGNET/S2 (3.15), LGKDE/S1 (2.29), LGKDE/S2 (2.65), LGKDE/S3 (2.13), LGKDE/S4 (2.13), HC-GLAD/S2 (1.44), HC-GLAD/S3 (2.04), HC-GLAD/S4 (1.45), SUL-GUL/S2 (6.49), SUL-GUL/S3 (7.00), SUL-GUL/S5 (6.14).

## 2. Selection on noise

### 2(a) Permutation null

Per run, the test labels are permuted 200 times (prevalence kept; seed 20261014 + run index); in each permutation, max over evaluation points of AUROC minus AUROC at LAST, and max minus the trajectory mean. Values are averaged over permutations, then aggregated like the observed values.

| method | obs. ORACLE − LAST | null max − LAST | ratio | null q95 (mean) | obs. ORACLE − S6 | null max − mean | ratio to obs. ORACLE − S6 | eval points |
|---|---|---|---|---|---|---|---|---|
| GLADC | 5.07 | 2.28 | 0.45 | 5.49 | 4.16 | 2.27 | 0.55 | 25.00 |
| CVTGAD | 5.12 | 5.47 | 1.07 | 11.39 | 6.56 | 5.47 | 0.83 | 147.22 |
| GOOD-D | 4.55 | 5.35 | 1.18 | 12.23 | 6.23 | 5.36 | 0.86 | 91.11 |
| GraphDE | 3.38 | 2.15 | 0.64 | 4.59 | 3.01 | 2.14 | 0.71 | 63.11 |
| GraphCL_IF | 3.99 | 2.53 | 0.63 | 6.40 | 3.93 | 2.55 | 0.65 | 8.00 |
| GraphCL_OCSVM | 6.37 | 3.56 | 0.56 | 8.74 | 5.75 | 3.57 | 0.62 | 15.00 |
| InfoGraph_IF | 9.67 | 3.30 | 0.34 | 8.17 | 7.74 | 3.29 | 0.42 | 16.67 |
| InfoGraph_OCSVM | 6.33 | 1.61 | 0.25 | 5.23 | 5.11 | 1.61 | 0.31 | 15.00 |
| MUSE | 7.68 | 4.32 | 0.56 | 10.93 | 6.24 | 4.27 | 0.68 | 10.00 |
| LGKDE | 2.29 | 4.14 | 1.80 | 9.85 | 4.79 | 4.11 | 0.86 | 100.00 |
| GLADPro | 18.16 | 9.05 | 0.50 | 16.87 | 20.19 | 8.98 | 0.44 | 455.56 |
| GLADMamba | 4.38 | 4.41 | 1.01 | 10.30 | 5.20 | 4.39 | 0.84 | 108.67 |
| SEGO | 4.38 | 5.78 | 1.32 | 12.46 | 6.58 | 5.75 | 0.87 | 40.00 |
| GLASS | 6.26 | 2.98 | 0.48 | 7.06 | 6.18 | 2.96 | 0.48 | 32.00 |
| HC-GLAD | 2.56 | 4.48 | 1.75 | 9.06 | 2.22 | 4.41 | 1.99 | 240.00 |
| SUL-GUL | 7.85 | 7.54 | 0.96 | 14.91 | 7.31 | 7.52 | 1.03 | 500.00 |

| pool | detectors | obs. ORACLE − LAST | null max − LAST | CI lo | CI hi | ratio | obs. ORACLE − S6 | null max − mean | ratio to obs. ORACLE − LAST | ratio to obs. ORACLE − S6 |
|---|---|---|---|---|---|---|---|---|---|---|
| frozen12 | 8 | 5.56 | 3.28 | 2.20 | 4.45 | 0.59 | 5.31 | 3.28 | 0.59 | 0.62 |
| recent | 8 | 6.69 | 5.34 | 4.02 | 6.77 | 0.80 | 7.34 | 5.30 | 0.79 | 0.72 |
| frozen12+recent | 16 | 6.13 | 4.31 | 3.13 | 5.57 | 0.70 | 6.33 | 4.29 | 0.70 | 0.68 |

Not computable (no per-graph scores): SIGNET, OCGTL, OCGIN, GLocalKD on UB9.

### 2(b) Opportunity and spread (descriptive)

| pool | x (vs observed ORACLE − LAST per run) | runs | methods | Spearman ρ | p |
|---|---|---|---|---|---|
| frozen12 | number of evaluation points | 540 | 12 | -0.046 | 0.291 |
| frozen12 | SD of the test-AUROC trajectory | 540 | 12 | 0.419 | 0.000 |
| recent | number of evaluation points | 360 | 8 | 0.175 | 0.001 |
| recent | SD of the test-AUROC trajectory | 360 | 8 | 0.282 | 0.000 |
| frozen12+recent | number of evaluation points | 900 | 20 | 0.056 | 0.094 |
| frozen12+recent | SD of the test-AUROC trajectory | 900 | 20 | 0.357 | 0.000 |

The number of evaluation points is set by the configuration (method and dataset), so these correlations mix between-method and within-method variation; they are descriptive only.

## 3. Dataset robustness and inference target

### Leave one dataset out (12 UB-GOLD detectors)

| left out | ORACLE − VAL-NORMAL | ORACLE − LAST |
|---|---|---|
| none (all 9) | 4.13 | 5.12 |
| AIDS | 4.33 | 5.10 |
| BZR | 4.10 | 5.12 |
| COX2 | 4.07 | 5.10 |
| DHFR | 4.02 | 4.85 |
| ENZYMES | 4.09 | 5.29 |
| NCI1 | 4.14 | 5.46 |
| PROTEINS_full | 3.96 | 4.89 |
| IMDB-BINARY | 4.04 | 4.86 |
| REDDIT-BINARY | 4.41 | 5.41 |

Range over the 9 leave-one-out values: vs VAL-NORMAL 3.96–4.41; vs LAST 4.85–5.46.

### Per-dataset detector-mean inflation (forest plot `paper/figures/posthoc_forest.pdf`)

| dataset | vs VAL-NORMAL | fold CI lo | fold CI hi | min det. | max det. | det. > 0 | vs LAST | fold CI lo | fold CI hi | min det. | max det. | det. > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AIDS | 2.50 | 1.96 | 3.08 | -2.69 | 8.25 | 11 | 5.24 | 4.25 | 6.23 | -2.40 | 13.10 | 11 |
| BZR | 4.40 | 3.01 | 5.85 | -0.12 | 9.28 | 11 | 5.10 | 3.29 | 7.05 | -0.12 | 9.28 | 11 |
| COX2 | 4.61 | 3.41 | 5.69 | -0.30 | 9.48 | 11 | 5.30 | 4.23 | 6.36 | 0.32 | 12.64 | 12 |
| DHFR | 5.00 | 4.58 | 5.71 | -3.71 | 11.06 | 11 | 7.30 | 6.64 | 7.84 | -1.72 | 14.48 | 11 |
| ENZYMES | 4.44 | 3.17 | 6.03 | 0.85 | 10.72 | 12 | 3.79 | 2.90 | 4.60 | 0.86 | 10.72 | 12 |
| NCI1 | 4.05 | 2.89 | 5.03 | -0.22 | 14.12 | 11 | 2.41 | 2.04 | 2.77 | -0.05 | 5.98 | 11 |
| PROTEINS_full | 5.48 | 4.38 | 6.80 | 0.13 | 12.40 | 12 | 6.95 | 5.67 | 8.37 | 0.11 | 12.51 | 12 |
| IMDB-BINARY | 4.82 | 3.42 | 6.14 | 0.08 | 13.38 | 12 | 7.19 | 6.39 | 8.16 | -0.47 | 25.57 | 11 |
| REDDIT-BINARY | 1.86 | 1.53 | 2.11 | -0.01 | 6.49 | 11 | 2.80 | 2.07 | 3.30 | 0.00 | 8.75 | 12 |

Fold-bootstrap CIs hold the 12 detectors fixed and resample the 5 folds (optimistic, as in REPORT.md).

### Crossed detector × dataset bootstrap

| comparison | G1 mean | dataset-bootstrap CI lo | hi | crossed CI lo | hi |
|---|---|---|---|---|---|
| ORACLE - VAL-NORMAL | 4.13 | 3.32 | 4.79 | 2.92 | 5.36 |
| ORACLE - LAST | 5.12 | 3.97 | 6.24 | 3.63 | 6.90 |

**Inferential target (for the paper):** the 12 evaluated UB-GOLD detectors, as configured by UB-GOLD, under variation across the UB9 datasets (dataset bootstrap). The crossed bootstrap additionally treats the detectors as a sample, which is a wider target than the one the main analysis claims.

## 4. Size-matched evaluation on the original test sets

1:1 nearest neighbour on standardised (log n, log(m+1)) within each test fold, caliper 0.2, without replacement, anomalies in test-index order. A dataset is reported only if ≥ 30% of its anomalies are matched in every fold. Size-only reference = Ledoit–Wolf Mahalanobis on (log n, log(m+1), density) fitted on the fold's normal training graphs; on the full test sets it reproduces grail0_folds.csv in 45/45 folds (|Δ| < 1e-9).

| dataset | anomalies (5 folds) | pairs (5 folds) | min frac. matched | mean frac. matched | reported | size-only AUROC, full | size-only AUROC, matched |
|---|---|---|---|---|---|---|---|
| AIDS | 400 | 5 | 0.00 | 0.01 | no | 98.72 | 50.00 |
| BZR | 319 | 64 | 0.16 | 0.20 | no | 42.47 | 42.30 |
| COX2 | 365 | 88 | 0.22 | 0.24 | no | 50.93 | 50.00 |
| DHFR | 295 | 259 | 0.80 | 0.88 | yes | 41.92 | 47.38 |
| ENZYMES | 100 | 89 | 0.80 | 0.89 | yes | 43.73 | 51.14 |
| NCI1 | 2053 | 1503 | 0.70 | 0.73 | yes | 53.67 | 51.07 |
| PROTEINS_full | 663 | 296 | 0.41 | 0.45 | yes | 57.78 | 51.77 |
| IMDB-BINARY | 500 | 237 | 0.45 | 0.47 | yes | 51.98 | 51.01 |
| REDDIT-BINARY | 1000 | 424 | 0.39 | 0.42 | yes | 74.08 | 66.71 |

Balance on the reported datasets (mean |SMD| over folds, before → after): log n 0.61 → 0.01; log(m+1) 0.65 → 0.02; density 0.56 → 0.03.

Mean over the reported datasets (detector AUROC at its VAL-NORMAL and ORACLE points; full test set vs matched subset). Pools: the 8 UB-GOLD detectors with per-graph scores; the 8 iterative recent methods (ProtoGLAD, whose VAL-NORMAL and ORACLE points coincide, is shown as a row but not pooled, A20):

| level | detector / pool | datasets | VAL-NORMAL full | VAL-NORMAL matched | ORACLE full | ORACLE matched | ORACLE − VN full | ORACLE − VN matched |
|---|---|---|---|---|---|---|---|---|
| method | GLADC | 6 | 59.94 | 50.49 | 65.02 | 54.64 | 5.07 | 4.15 |
| method | CVTGAD | 6 | 70.00 | 62.10 | 73.69 | 63.91 | 3.70 | 1.81 |
| method | GOOD-D | 6 | 66.03 | 59.20 | 70.90 | 61.42 | 4.87 | 2.21 |
| method | GraphDE | 6 | 52.21 | 49.67 | 53.41 | 50.24 | 1.20 | 0.57 |
| method | GraphCL_IF | 6 | 62.19 | 53.58 | 65.05 | 54.93 | 2.85 | 1.35 |
| method | GraphCL_OCSVM | 6 | 58.30 | 52.12 | 65.47 | 55.12 | 7.17 | 3.00 |
| method | InfoGraph_IF | 6 | 59.60 | 51.10 | 65.76 | 54.90 | 6.16 | 3.80 |
| method | InfoGraph_OCSVM | 6 | 60.69 | 52.12 | 64.81 | 54.12 | 4.12 | 2.00 |
| method | MUSE | 6 | 56.97 | 54.21 | 63.72 | 55.73 | 6.75 | 1.52 |
| method | LGKDE | 6 | 56.43 | 56.31 | 58.50 | 58.83 | 2.07 | 2.52 |
| method | GLADPro | 6 | 48.98 | 50.57 | 66.69 | 58.77 | 17.72 | 8.20 |
| method | GLADMamba | 6 | 61.49 | 53.35 | 72.32 | 62.08 | 10.83 | 8.73 |
| method | SEGO | 6 | 64.25 | 57.85 | 71.70 | 61.95 | 7.45 | 4.10 |
| method | GLASS | 6 | 66.20 | 61.39 | 70.75 | 64.28 | 4.55 | 2.89 |
| method | HC-GLAD | 6 | 59.16 | 51.76 | 61.72 | 53.44 | 2.55 | 1.68 |
| method | SUL-GUL | 6 | 53.19 | 51.92 | 63.01 | 54.14 | 9.82 | 2.22 |
| method | ProtoGLAD | 6 | 60.68 | 55.17 | 60.68 | 55.17 | 0.00 | 0.00 |
| pool | frozen12 (8 with scores) | 6 | 61.12 | 53.80 | 65.51 | 56.16 | 4.39 | 2.36 |
| pool | recent | 6 | 58.33 | 54.67 | 66.05 | 58.65 | 7.72 | 3.98 |
| reference | size-only (LW Mahalanobis) | 6 | 53.86 | 53.18 | 53.86 | 53.18 | – | – |

Per dataset (pool means; datasets below the 30% rule are shown for completeness only and are not reported results):

| dataset: pool | reported | VAL-NORMAL full | VAL-NORMAL matched | ORACLE full | ORACLE matched |
|---|---|---|---|---|---|
| AIDS: frozen12 (8 with scores) | no | 91.15 | 44.79 | 93.32 | 50.00 |
| AIDS: recent | no | 73.34 | 33.85 | 78.15 | 50.00 |
| AIDS: size-only reference | no | 98.72 | 50.00 | 98.72 | 50.00 |
| BZR: frozen12 (8 with scores) | no | 60.46 | 50.45 | 64.88 | 55.74 |
| BZR: recent | no | 50.59 | 53.30 | 60.72 | 59.06 |
| BZR: size-only reference | no | 42.47 | 42.30 | 42.47 | 42.30 |
| COX2: frozen12 (8 with scores) | no | 55.53 | 47.67 | 60.86 | 53.97 |
| COX2: recent | no | 50.74 | 46.32 | 56.70 | 51.55 |
| COX2: size-only reference | no | 50.93 | 50.00 | 50.93 | 50.00 |
| DHFR: frozen12 (8 with scores) | yes | 55.97 | 50.59 | 61.61 | 54.26 |
| DHFR: recent | yes | 51.07 | 51.89 | 57.57 | 55.57 |
| DHFR: size-only reference | yes | 41.92 | 47.38 | 41.92 | 47.38 |
| ENZYMES: frozen12 (8 with scores) | yes | 59.88 | 57.04 | 64.49 | 60.95 |
| ENZYMES: recent | yes | 56.34 | 54.63 | 62.16 | 59.11 |
| ENZYMES: size-only reference | yes | 43.73 | 51.14 | 43.73 | 51.14 |
| NCI1: frozen12 (8 with scores) | yes | 45.60 | 56.35 | 51.26 | 59.37 |
| NCI1: recent | yes | 55.52 | 56.79 | 63.26 | 62.67 |
| NCI1: size-only reference | yes | 53.67 | 51.07 | 53.67 | 51.07 |
| PROTEINS_full: frozen12 (8 with scores) | yes | 66.48 | 48.40 | 70.52 | 49.33 |
| PROTEINS_full: recent | yes | 59.76 | 50.78 | 65.67 | 54.13 |
| PROTEINS_full: size-only reference | yes | 57.78 | 51.77 | 57.78 | 51.77 |
| IMDB-BINARY: frozen12 (8 with scores) | yes | 58.79 | 51.50 | 63.56 | 52.48 |
| IMDB-BINARY: recent | yes | 56.80 | 53.03 | 66.50 | 55.65 |
| IMDB-BINARY: size-only reference | yes | 51.98 | 51.01 | 51.98 | 51.01 |
| REDDIT-BINARY: frozen12 (8 with scores) | yes | 80.00 | 58.90 | 81.65 | 60.56 |
| REDDIT-BINARY: recent | yes | 70.53 | 60.90 | 81.17 | 64.78 |
| REDDIT-BINARY: size-only reference | yes | 74.08 | 66.71 | 74.08 | 66.71 |

Per method × dataset values (all datasets, with the `reported` flag) are in `eval_posthoc_sizematch.csv`; per-fold matching statistics in `eval_posthoc_sizematch_folds.csv`.

## 5. Reproduction-fidelity sensitivity

| method | tier | tier (alternative reading) | code | configuration | A10 §4 check |
|---|---|---|---|---|---|
| MUSE | incomplete released configuration | incomplete released configuration | minimal compatibility patch | incomplete released configuration | met |
| LGKDE | reproduction not met | reproduction not met | framework port | incomplete released configuration | not met |
| GLADPro | incomplete released configuration | reproduction not met | minimal compatibility patch | incomplete released configuration | met (contested) |
| GLADMamba | incomplete released configuration | incomplete released configuration | native | incomplete released configuration | met |
| SEGO | incomplete released configuration | incomplete released configuration | minimal compatibility patch | incomplete released configuration | met |
| GLASS | native | native | native | released configuration complete | met |
| ProtoGLAD | minimal compatibility patch | minimal compatibility patch | minimal compatibility patch | released configuration complete | met |
| HC-GLAD | incomplete released configuration | incomplete released configuration | native | incomplete released configuration | met |
| SUL-GUL | incomplete released configuration | reproduction not met | minimal compatibility patch | incomplete released configuration | not evaluable |

Evidence for each facet is in `eval_posthoc_fidelity.csv` (Appendix B tab:ports and ext/<M>/NOTES.md).

| pool | comparison | methods | mean | CI lo | CI hi | dropped |
|---|---|---|---|---|---|---|
| all 8 iterative recent methods (= REPORT.md recent(E5+E4)) | ORACLE_ubgold - VAL-NORMAL | 8 | 7.47 | 6.15 | 8.83 |  |
| all 8 iterative recent methods (= REPORT.md recent(E5+E4)) | ORACLE_ubgold - LAST | 8 | 6.69 | 5.75 | 7.61 |  |
| without 'reproduction not met' (primary tier reading) | ORACLE_ubgold - VAL-NORMAL | 7 | 8.16 | 6.71 | 9.58 | LGKDE |
| without 'reproduction not met' (primary tier reading) | ORACLE_ubgold - LAST | 7 | 7.32 | 5.73 | 8.93 | LGKDE |
| without 'reproduction not met' or contested/not evaluable (alternative reading) | ORACLE_ubgold - VAL-NORMAL | 5 | 6.38 | 5.39 | 7.43 | LGKDE|GLADPro|SUL-GUL |
| without 'reproduction not met' or contested/not evaluable (alternative reading) | ORACLE_ubgold - LAST | 5 | 5.05 | 3.65 | 6.69 | LGKDE|GLADPro|SUL-GUL |

Sanity: the full recent pool equals REPORT.md recent(E5+E4), UB-GOLD-rule ORACLE: − VAL-NORMAL 7.4679 [6.15, 8.83], − LAST 6.6947 [5.75, 7.61].

## Points noted while computing

- S7 coverage (A32 item 1): `holdout_score_norm` is recorded only for CVTGAD (harness/run_eval.py, fit_cvtgad); the other harness detectors and the old-4 deep_rerun runs record no standardised held-out score, so S7 is pooled over 1 UB-GOLD detector (+ GLADMamba, SEGO, GLASS, where S7 = S2 by construction). CVTGAD's S7 picks a point in the first 10% of training in 91% of its runs, the same training-statistics normalisation artefact that A14 describes for GLADMamba. Status: verified (eval_posthoc_runs.csv).
- Permutation null (A32 2a): for CVTGAD (1.07), GOOD-D (1.18), LGKDE (1.80), GLADMamba (1.01), SEGO (1.32), HC-GLAD (1.75) the null max − LAST is at least the observed ORACLE − LAST; the largest nulls occur for the methods with the most evaluation points (SUL-GUL 500 points: 7.54, GLADPro 456 points: 9.05). A reviewer may read the inflation of these methods as selection on noise. Status: verified (eval_posthoc_permnull.csv).
- Size matching (A32 4): BZR (max possible 27% matched), COX2 (max possible 27% matched) cannot reach the 30% rule because normals are the minority of their UB-GOLD test folds (UB-GOLD's anomaly-class convention, comment #7); AIDS fails because its anomalies and normals barely overlap in size (size-only AUROC 98.7 on the full test sets). Status: verified (eval_posthoc_sizematch_folds.csv).
- Matching check (A32 4): on REDDIT-BINARY the size-only reference keeps AUROC 66.7 on the matched subset although every fold's |standardised mean difference| of log n, log(m+1) and density is at most 0.09 after matching; the 0.2 caliper on (log n, log(m+1)) leaves residual size information that the Ledoit-Wolf Mahalanobis reference (two-sided distance from the training-normal centre) still uses. Size-matched results on that dataset are not size-free. Status: verified (eval_posthoc_sizematch_folds.csv).
- Crossed detector × dataset bootstrap (A32 3): the 95% CI of ORACLE − VAL-NORMAL is [2.92, 5.36], its lower end below the G1 threshold of 3.0; the paper should state the inferential target (the evaluated detectors across UB9) and not generalise to other detectors. Status: verified (eval_posthoc_crossed.csv).
- Reproduction status of GLADPro (A32 5): ext/GLADPro/NOTES.md calls the A10 §4 criterion met (COX2, ENZYMES vs M-GLAD), but both matches are on datasets whose anomaly or normal classes likely differ, and the Appendix B table caption says such datasets are not counted; under that rule GLADPro meets 0 of 3. The table entry and the caption should be made consistent. Status: verified (app_ports.tex tab:ports, GLADPro NOTES reproduction section).

## Interpretations where A32 is silent

1. Run set: UB9 x 'ubgold' x 5 folds. The 12 UB-GOLD detectors use the same runs as eval_folds.csv (harness parts; for SIGNET/OCGTL/OCGIN/GLocalKD on UB9 the deep_rerun parts, which have per-point trajectories but no per-graph scores). Recent methods: the 8 iterative E5/E4 methods; ProtoGLAD has no checkpoint axis (A20) and enters only item 4 (its VAL-NORMAL and ORACLE points coincide) and item 5's tier table.
2. ORACLE = UB-GOLD rule (patience 30) for every method (A32 'unless stated otherwise'; A26 for pooled numbers): glad_stats.ubgold_select with the EF filter set of glad_stats, CVTGAD selected on auroc_sel (A4).
3. A15 carried over to the new selectors: a signal-based selector (S2, S3, S4, S7) never picks a point whose signal is non-finite (S3's median is non-finite if any held-out score is); no finite signal, or a selected point with non-finite test scores, gives AUROC 50 and a flag (glad_stats interpretation 2a/2b). S5 at a non-finite point gives 50. S6 counts a non-finite point as 50 (a uniformly drawn non-finite checkpoint ranks at chance).
4. Ties: S2-S5 take the earliest evaluation point among ties (as glad_stats.valnormal_select).
5. S4: 'mean training loss' = the epoch's mean training loss over batches as recorded at each evaluation point (traj train_loss, equal to the per-epoch train_loss list).
6. S5: epoch budget = num_epoch of the run's resolved configuration; distance |epoch - num_epoch/2|.
7. S7: 'holdout_score_norm or its equivalent' = a held-out score standardised with statistics of the normal training graphs at the same checkpoint. Only CVTGAD records it separately; GLADMamba, SEGO and GLASS score with such a standardised score already (S7 = S2 there). No other run records one, so S7 is not available for the other 11 UB-GOLD detectors and 5 recent methods; no substitute is computed.
8. Item 1 pooled means are over the detectors with complete UB9 coverage for that selector (as glad_stats). Because S3 and S7 have smaller coverage, the table also gives S1-S6 pooled over the S3-covered detectors (coverage-matched view; same estimand, same bootstrap).
9. Item 2(a): 'max over evaluation points' is the plain maximum over all finite points (not the patience-30 rule), as A32 states; this can only make the null larger than a null of the patience rule. The trajectory mean counts non-finite points as 50, a non-finite LAST is 50 (A15 rule 3). Null values are aggregated like the observed values (5-fold mean per dataset, mean over UB9, mean over detectors). Ratios: null(max - LAST) / observed (ORACLE - LAST) as A32 states; null(max - mean) is reported against observed ORACLE - LAST (literal) and against observed ORACLE - S6 (its observed analogue).
10. Run index (permutation seed): position in the canonical list METHODS_ALL x UB9 x folds, i.e. index = (method_pos * 9 + dataset_pos) * 5 + fold with METHODS_ALL = 12 UB-GOLD detectors, E5, E4 in the glad_stats order.
11. Item 2(b): Spearman correlation over all runs of a pool (frozen-12; recent; both); SD of the test-AUROC trajectory = population SD (ddof 0) over finite evaluation points, in AUROC points.
12. Item 3 forest plot: per-dataset detector-mean inflation with a paired fold-bootstrap 95% CI (4000 resamples; the 5 folds are resampled jointly for all detectors, glad_stats' fold resamples), consistent with the stated target (the evaluated detectors fixed, datasets varying); the detector range (min, max) is reported beside it. Crossed bootstrap: detectors and datasets resampled independently with replacement in each of 4000 replicates (seed 20261012) on the 12 x 9 matrix of 5-fold means.
13. Item 4: features standardised with the mean and population SD (ddof 0) over all test graphs of the fold; distance = Euclidean distance in the standardised plane, matched if <= 0.2; nearest-distance ties go to the normal with the lower test index; m = number of undirected edges (fold cache, equal to the npz test_n_edges); the size-only reference is refitted per fold on all normal training graphs of the fold, as in grail0 (verified against grail0_folds.csv on the full test sets). Detector AUROC on the matched subset at the run's VAL-NORMAL and ORACLE (UB-GOLD rule) points; a non-finite selected point is 50 (A15).
14. Item 5 tiers: each method gets the lowest-fidelity tier that applies, in A32's order (native < minimal compatibility patch < framework port < incomplete released configuration < reproduction not met). 'Native' = no edit to the method's model, loss or scoring code beyond harness instrumentation (data root, folds, seeds, held-out carve-out, no_grad, recording, resume); 'incomplete released configuration' = no released configuration for at least one UB9 dataset, or a released component that cannot run in the one-class protocol; 'reproduction not met' = the pre-registered A10 §4 check reported as not met. Primary exclusion = LGKDE (the only method documented as 'not met'). Because the sources conflict for GLADPro (NOTES: criterion met, 2 of 7, both matches on datasets whose anomaly class likely differs; Appendix B caption: such datasets are not counted) and SUL/GUL is 'not evaluable' (A22), a second, labelled row also drops GLADPro and SUL-GUL.

## Outputs

`results/eval_posthoc_{coverage,crossed,fidelity,fidelity_pooled,forest,lodo,permnull,permnull_pooled,runs,sanity,sanity_pooled,selectors,selectors_pooled,sizematch,sizematch_datasets,sizematch_folds,sizematch_pooled,spearman}.csv`; figures `evalpaper/paper/figures/posthoc_forest.{pdf,png}` and `posthoc_selectors.{pdf,png}` (written by `posthoc_robustness.py figures` with the paper-figs environment).

