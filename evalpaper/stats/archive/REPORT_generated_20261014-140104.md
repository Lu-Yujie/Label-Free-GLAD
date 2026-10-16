# Label-free GLAD re-evaluation: statistics report v2 (AUTO-GENERATED)

Generated 2026-10-14 14:01:02 EDT by `prelim/evalpaper/stats/glad_stats.py` v2 (sha256 `2731b02f07519f8bbd97305018fd04ff3771ca3f71e4b1d86b6d7a34d5802c78`; B = 4000, seed 20261012). Do not edit by hand. AUROC × 100; a leaderboard value is the mean over the datasets of a set of the 5-fold means. CIs: 95% percentile intervals of paired fold- and dataset-bootstraps (fold CIs are optimistic).

## 1. PREREG and amendment status

- **G1 primary (A14 dual selector): PASS** — VAL-NORMAL: PASS, LAST: PASS. PASS needs both; one of two = SELECTOR-DEPENDENT (not met for GO).
  - ORACLE − VAL-NORMAL: complete 12/12 (≥ 8), mean 4.13 (fold CI 3.77–4.50; ds CI 3.32–4.79; ≥ 3.0), methods ≥ 2.0: 12/12 (≥ 75%); conditions {'n_complete>=8': True, 'mean>=3.0': True, 'frac(>=2.0)>=0.75': True}; per method: SIGNET 3.15, OCGTL 3.21, OCGIN 5.34, GLocalKD 3.84, GLADC 4.82, CVTGAD 5.02, GOOD-D 4.65, GraphDE 2.15, GraphCL_IF 3.10, GraphCL_OCSVM 6.00, InfoGraph_IF 5.32, InfoGraph_OCSVM 2.96
  - ORACLE − LAST: complete 12/12 (≥ 8), mean 5.12 (fold CI 4.77–5.48; ds CI 3.97–6.24; ≥ 3.0), methods ≥ 2.0: 12/12 (≥ 75%); conditions {'n_complete>=8': True, 'mean>=3.0': True, 'frac(>=2.0)>=0.75': True}; per method: SIGNET 4.01, OCGTL 3.41, OCGIN 5.69, GLocalKD 3.85, GLADC 5.07, CVTGAD 5.12, GOOD-D 4.55, GraphDE 3.38, GraphCL_IF 3.99, GraphCL_OCSVM 6.37, InfoGraph_IF 9.67, InfoGraph_OCSVM 6.33
- **G2(a): PASS** — τ-b(ORACLE, VAL-NORMAL) = 0.76 over 12 methods (≤ 0.8); top-3 ['CVTGAD', 'SIGNET', 'GOOD-D'] → ['SIGNET', 'CVTGAD', 'OCGTL'] (changed True).
- **G2(b): FAIL** — mean VAL-NORMAL − HP-FREE = 1.53 over 12 methods (≥ 2.0); A21 exclusions: none.
- **G2(c) per A9: PASS** (hypothesis generated post hoc; tested out of sample per A9) — verdict CONFIRMED; decision types ['GX:T1a', 'GX:T3a', 'GX:T3b', 'MS:MS1', 'MS:MS2', 'MS:MS3']; valid MS types 3; blind fraction pooled 0.81, MS 0.75; methods blind on a majority of MS types 6; H4s: {'types': ['MS1', 'MS2', 'MS3'], 'blind_rate_matched': 0.75, 'n_matched': 24, 'blind_rate_changing': 0.125, 'n_changing': 24, 'drop_pp': 62.5, 'scope_to_size_preserving': True, 'status': 'computed'}.
- **G2 overall: PASS; decision: GO.**

| amendment | rule | state |
|---|---|---|
| A0-A4, A6-A8 | harness-level definitions (HP-FREE, VAL-NORMAL signal, CVTGAD score, two-stage alignment, environment, secondary datasets) | applied upstream in harness/run_eval.py + collect.py; stats consumes eval_folds.csv |
| A5 | GraphDE paper orientation main | 390 UB-GOLD-orientation rows kept as reproduction only |
| A9 | G2(c) out-of-sample typed test | G2(c) PASS; verdict: CONFIRMED |
| A10/A11 | E5 / E4 strata, MUTAG secondary | rows per stratum: frozen12 6256, E5 2000, E4 1600, kernel 2080, reference 2800 |
| A12 | DIVERGED runs | 6 registered; excluded from aggregates; coverage shows 6 diverged folds |
| A13/A16/A18/A19/A22 | per-method labels (MUSE, SEGO released-modal, LGKDE MUTAG.yaml, HC-GLAD memory rule, SUL-GUL GUL only + 500 epochs) | hp_label / method_label columns |
| A14 | dual-selector G1; VAL-NORMAL-degenerate flag | G1 VAL-NORMAL PASS, LAST PASS -> primary PASS; degenerate (scope all): GLADMamba/hpfree |
| A15 | non-finite rules | 12 protocol rows changed by the audit; nonfinite_selected rows 12; runs flagged 12 |
| A17 | version sensitivity GLocalKD/GLADC a96137b | PENDING (no a96137b runs yet) |
| A20 | checkpoint-free excluded from pooled checkpoint inflation | GRAIL-0, PK-IF, PK-OCSVM, ProtoGLAD, WL-IF, WL-OCSVM, WL-hist+iForest (ctrl), graphlet-count+LW, graphlet-count+iForest, size-only |
| A21 | identical-config excluded from pooled HP leak; tie_collapsed flag | excluded: GLASS, SUL-GUL; tie_collapsed runs 71; rows without per-graph scores: 5680 |
| A23/A24 | GLASS: inductive default rule (A24 withdraws the transductive flag); ubgold==hpfree noise estimate | GLASS: 18/50 pairs bit-identical, 32 differ; SUL-GUL: 50/50 pairs bit-identical, 0 differ; transductive rows: 0 |
| adapters | E5/E4 adapter runs | all ran |

## 2. Inputs and provenance

- eval_folds.csv: 4632 main rows (GraphDE UB-GOLD-orientation rows: 390 -> reproduction only, A5); MUTAG rows of the old 4 from deep_rerun parts: 60; extra rows 3208
- MUSE: adapter MUSE/collect_muse.py -> 400 protocol rows
- LGKDE: adapter LGKDE/derive.py -> 400 protocol rows
- GLADPro: adapter GLADPro/derive.py -> 400 protocol rows
- GLADMamba: adapter GLADMamba/collect.py -> 400 protocol rows
- SEGO: adapter SEGO/collect_sego.py -> 400 protocol rows
- GLASS: adapter GLASS/derive.py -> 400 protocol rows
- ProtoGLAD: adapter ProtoGLAD/derive.py -> 400 protocol rows
- HC-GLAD: adapter HC-GLAD/collect.py -> 400 protocol rows
- SUL-GUL: adapter SUL-GUL/derive.py -> 400 protocol rows
- eval_kernels_folds.csv: 520 train-fit rows (KernelGLAD 'tuned' -> ubgold protocols, 'default' -> hpfree protocols; @testfit rows not ranked)
- grail0_folds.csv: references ['GRAIL-0', 'WL-hist+iForest (ctrl)', 'graphlet-count+LW', 'graphlet-count+iForest', 'size-only'] (a priori; identical in every protocol)
- trajectories indexed for the audit: 2464 runs
- A12 DIVERGED registry: 6 runs (/geode3/home/u110/yujlu/BigRed200/research/GAD/Grail/prelim/evalpaper/stats/diverged_registry.csv + '# DIVERGED' lines in task files)
- m1_selection_folds.csv: 2205 rows (old-4 label-free selectors; descriptive)
- A17: eval_version_a96137b_folds.csv not present (version-sensitivity runs PENDING)
- eval_typed*.csv: 2295 rows from eval_typed.csv, eval_typed_controls_msv2.csv

sha256 of code and inputs used:

- `eval_folds.csv`: `db861f0d0585aadf45af57ce010a8d86b87bb07ad5481c315720ecf8b0d63338`
- `eval_kernels_folds.csv`: `115df3afa5be31c8c9c6ea46a420bf625b6fe74b5256dac63c99f9b7cb5d8582`
- `eval_typed.csv`: `94539dbfa9f66c66e57586b545eacd67aa30f1c209cf252cb8925760bd7b6d98`
- `eval_typed_controls_msv2.csv`: `8d148b82a315a2fa4238c97a12f6a414204b4c6d368e91968b21a15a5facbf8e`
- `eval_version_a96137b_folds.csv`: `None`
- `ext/GLADMamba/collect.py`: `4373d7d0e9ed6023a038649a21e00424aa10547aff291577bdcbc8ac088e6fba`
- `ext/GLADPro/derive.py`: `b8072f09949b73d35c54f9b3174daf199b0e145e5a5b72ee65c32b8bbb82c3f8`
- `ext/GLASS/derive.py`: `9215f15f133014a8e147b3d18661224af2aa37fe7a386f413d0669a96077c17d`
- `ext/HC-GLAD/collect.py`: `813a7b84b51cc8411430126b04a348557a06ae079d1724d31ab6b8a62deb9a38`
- `ext/LGKDE/derive.py`: `79b104fbd2ebe9f2619d139f7f4d9d2628624f9d7c6844c19e9b3c776a0511ff`
- `ext/MUSE/collect_muse.py`: `4933393e56280917f853f3fa88855f539e15a9bb4deb68283781e46ac9346f0d`
- `ext/ProtoGLAD/derive.py`: `2db86f4d4535fc5c316b36f919d8f24c91a94d0a4a8fb981e0a4885dd4418edf`
- `ext/SEGO/collect_sego.py`: `67182983f17e9b0fbe96a36c2baf52b789c90c5c62a7a4dc5bdd93eeec76f3be`
- `ext/SUL-GUL/derive.py`: `31917bb946cfb9bb661f5b12008c0fd40f6ed56ed0272531da4782e1b0fd3138`
- `glad_stats.py`: `2731b02f07519f8bbd97305018fd04ff3771ca3f71e4b1d86b6d7a34d5802c78`
- `grail0_folds.csv`: `79622dca28c3402f6198d469bb286173407edcd98d4075ba69ad5ccb0e7ec3dd`
- `harness/collect.py`: `792d8ab6ec16778ac39ab921bf96566ad5ce09cccb3c75a4fc7a06ae76e4317e`
- `molswap manifest.json`: `7d7ddf76749425a74bed8d7e5c667b510318f6ef7b755a1224ee8d8dd73f2658`

## 3. Coverage (DIVERGED cells shown; A12)

| stratum | method | hp | AIDS | BZR | COX2 | DHFR | ENZYMES | NCI1 | PROTEINS_full | IMDB-BINARY | REDDIT-BINARY | DD | COLLAB | Tox21_p53 | Tox21_MMP | MUTAG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E4 | GLASS | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E4 | GLASS | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E4 | HC-GLAD | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E4 | HC-GLAD | ubgold | 5/5 | 5/5 [nonfinite_selected: f0] [tie_collapsed: f0] | 5/5 | 5/5 | 5/5 | 5/5 [nonfinite_selected: f4] [tie_collapsed: f4] | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E4 | ProtoGLAD | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E4 | ProtoGLAD | ubgold | 5/5 | 5/5 | 5/5 [tie_collapsed: f0,1,2] | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E4 | SUL-GUL (GUL only; SUL not released) | hpfree | 5/5 [nonfinite_selected: f4] [tie_collapsed: f0,4] | 5/5 | 5/5 | 5/5 | 5/5 [tie_collapsed: f0,1,2,3,4] | 5/5 [nonfinite_selected: f1,2,3,4] [tie_collapsed: f0,1,2,3,4] | 5/5 [tie_collapsed: f1,3,4] | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E4 | SUL-GUL (GUL only; SUL not released) | ubgold | 5/5 [nonfinite_selected: f4] [tie_collapsed: f0,4] | 5/5 | 5/5 | 5/5 | 5/5 [tie_collapsed: f0,1,2,3,4] | 5/5 [nonfinite_selected: f1,2,3,4] [tie_collapsed: f0,1,2,3,4] | 5/5 [tie_collapsed: f1,3,4] | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E5 | GLADMamba | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E5 | GLADMamba | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E5 | GLADPro | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 [tie_collapsed: f0,1,2,3,4] | 5/5 [tie_collapsed: f1,2,3] | missing | missing | missing | missing | 5/5 |
| E5 | GLADPro | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 [tie_collapsed: f0,1,2,3,4] | 5/5 [tie_collapsed: f1,2] | missing | missing | missing | missing | 5/5 |
| E5 | LGKDE ('ubgold' = MUTAG.yaml for all) | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E5 | LGKDE ('ubgold' = MUTAG.yaml for all) | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E5 | MUSE | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E5 | MUSE | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E5 | SEGO ('ubgold' = released-modal) | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| E5 | SEGO ('ubgold' = released-modal) | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | missing | missing | missing | 5/5 |
| frozen12 | CVTGAD | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | CVTGAD | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | GLADC | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | 2/5 + DIVERGED 3 | 5/5 | 5/5 | missing |
| frozen12 | GLADC | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 2/5 + DIVERGED 3 | 5/5 | 5/5 | missing |
| frozen12 | GLocalKD | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | GLocalKD | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| frozen12 | GOOD-D | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | GOOD-D | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | GraphCL_IF | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | GraphCL_IF | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | GraphCL_OCSVM | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 [tie_collapsed: f3] | 5/5 | 5/5 | missing |
| frozen12 | GraphCL_OCSVM | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 [tie_collapsed: f2,3] | 5/5 | missing |
| frozen12 | GraphDE (paper orientation, A5) | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 [tie_collapsed: f0,1,2,3] | 5/5 | 5/5 | 5/5 | 5/5 [tie_collapsed: f3] | 5/5 [tie_collapsed: f1,2] | missing |
| frozen12 | GraphDE (paper orientation, A5) | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 [tie_collapsed: f3] | 5/5 | 5/5 | 5/5 | 5/5 [tie_collapsed: f2,3] | 5/5 [tie_collapsed: f2,3,4] | missing |
| frozen12 | InfoGraph_IF | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | InfoGraph_IF | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | InfoGraph_OCSVM | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | InfoGraph_OCSVM | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | OCGIN | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | OCGIN | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| frozen12 | OCGTL | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | OCGTL | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 [tie_collapsed: f0,1,2,3,4] | 5/5 | 5/5 |
| frozen12 | SIGNET | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| frozen12 | SIGNET | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| kernel | PK-IF | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| kernel | PK-IF | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| kernel | PK-OCSVM | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| kernel | PK-OCSVM | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| kernel | WL-IF | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| kernel | WL-IF | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| kernel | WL-OCSVM | hpfree | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| kernel | WL-OCSVM | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | missing |
| reference | GRAIL-0 | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| reference | WL-hist+iForest (ctrl) | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| reference | graphlet-count+LW | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| reference | graphlet-count+iForest | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| reference | size-only | ubgold | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

## 4. Flags (A12, A14, A15, A21)

| method | stratum | runs | runs_nonfinite_selected | runs_tie_collapsed | rows_tie_unavailable | rows_changed_by_A15 | diverged_runs | valnormal_degenerate | transductive | identical_config | checkpoint_free |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CVTGAD | frozen12 | 130 | 0 | 0 | 0 | 0 | 0 |  | False | False | False |
| GLADC | frozen12 | 119 | 0 | 0 | 0 | 0 | 6 |  | False | False | False |
| GLADMamba | E5 | 100 | 0 | 0 | 0 | 0 | 0 | hpfree | False | False | False |
| GLADPro | E5 | 100 | 0 | 15 | 0 | 0 | 0 |  | False | False | False |
| GLASS | E4 | 100 | 0 | 0 | 0 | 0 | 0 |  | False | True | False |
| GLocalKD | frozen12 | 130 | 0 | 0 | 200 | 0 | 0 |  | False | False | False |
| GOOD-D | frozen12 | 130 | 0 | 0 | 0 | 0 | 0 |  | False | False | False |
| GRAIL-0 | reference | 140 | 0 | 0 | 560 | 0 | 0 |  | False | False | True |
| GraphCL_IF | frozen12 | 130 | 0 | 0 | 0 | 0 | 0 |  | False | False | False |
| GraphCL_OCSVM | frozen12 | 130 | 0 | 3 | 0 | 0 | 0 |  | False | False | False |
| GraphDE | frozen12 | 130 | 0 | 13 | 0 | 0 | 0 |  | False | False | False |
| HC-GLAD | E4 | 100 | 2 | 2 | 0 | 2 | 0 |  | False | False | False |
| InfoGraph_IF | frozen12 | 130 | 0 | 0 | 0 | 0 | 0 |  | False | False | False |
| InfoGraph_OCSVM | frozen12 | 130 | 0 | 0 | 0 | 0 | 0 |  | False | False | False |
| LGKDE | E5 | 100 | 0 | 0 | 0 | 0 | 0 |  | False | False | False |
| MUSE | E5 | 100 | 0 | 0 | 0 | 0 | 0 |  | False | False | False |
| OCGIN | frozen12 | 135 | 0 | 0 | 200 | 0 | 0 |  | False | False | False |
| OCGTL | frozen12 | 135 | 0 | 5 | 200 | 0 | 0 |  | False | False | False |
| PK-IF | kernel | 130 | 0 | 0 | 520 | 0 | 0 |  | False | False | True |
| PK-OCSVM | kernel | 130 | 0 | 0 | 520 | 0 | 0 |  | False | False | True |
| ProtoGLAD | E4 | 100 | 0 | 3 | 0 | 0 | 0 |  | False | False | True |
| SEGO | E5 | 100 | 0 | 0 | 0 | 0 | 0 |  | False | False | False |
| SIGNET | frozen12 | 135 | 0 | 0 | 200 | 0 | 0 |  | False | False | False |
| SUL-GUL | E4 | 100 | 10 | 30 | 0 | 10 | 0 |  | False | True | False |
| WL-IF | kernel | 130 | 0 | 0 | 520 | 0 | 0 |  | False | False | True |
| WL-OCSVM | kernel | 130 | 0 | 0 | 520 | 0 | 0 |  | False | False | True |
| WL-hist+iForest (ctrl) | reference | 140 | 0 | 0 | 560 | 0 | 0 |  | False | False | True |
| graphlet-count+LW | reference | 140 | 0 | 0 | 560 | 0 | 0 |  | False | False | True |
| graphlet-count+iForest | reference | 140 | 0 | 0 | 560 | 0 | 0 |  | False | False | True |
| size-only | reference | 140 | 0 | 0 | 560 | 0 | 0 |  | False | False | True |

A14 §3 VAL-NORMAL-degenerate (descriptive; not a gate):

| method | hp | scope | runs | frac_valnormal_first10pct | mean_last_minus_valnormal | valnormal_degenerate |
|---|---|---|---|---|---|---|
| CVTGAD | hpfree | all | 65 | 0.00 | -0.07 | False |
| CVTGAD | hpfree | UB9 | 45 | 0.00 | -0.10 | False |
| CVTGAD | ubgold | all | 65 | 0.00 | -0.01 | False |
| CVTGAD | ubgold | UB9 | 45 | 0.00 | -0.10 | False |
| GLADC | hpfree | all | 57 | 0.04 | 0.01 | False |
| GLADC | hpfree | UB9 | 45 | 0.02 | 0.03 | False |
| GLADC | ubgold | all | 62 | 0.00 | -0.15 | False |
| GLADC | ubgold | UB9 | 45 | 0.00 | -0.25 | False |
| GLADMamba | hpfree | all | 50 | 0.90 | 6.68 | True |
| GLADMamba | hpfree | UB9 | 45 | 0.89 | 3.77 | False |
| GLADMamba | ubgold | all | 50 | 0.76 | 9.74 | False |
| GLADMamba | ubgold | UB9 | 45 | 0.73 | 6.06 | False |
| GLADPro | hpfree | all | 50 | 0.06 | -0.09 | False |
| GLADPro | hpfree | UB9 | 45 | 0.07 | 1.53 | False |
| GLADPro | ubgold | all | 50 | 0.04 | 0.50 | False |
| GLADPro | ubgold | UB9 | 45 | 0.04 | 0.52 | False |
| GLASS | hpfree | all | 50 | 0.60 | -0.43 | False |
| GLASS | hpfree | UB9 | 45 | 0.62 | -0.78 | False |
| GLASS | ubgold | all | 50 | 0.56 | -0.66 | False |
| GLASS | ubgold | UB9 | 45 | 0.58 | -1.03 | False |
| GLocalKD | hpfree | all | 60 | 0.12 | 0.02 | False |
| GLocalKD | hpfree | UB9 | 45 | 0.11 | 0.03 | False |
| GLocalKD | ubgold | all | 70 | 0.00 | -0.01 | False |
| GLocalKD | ubgold | UB9 | 45 | 0.00 | -0.01 | False |
| GOOD-D | hpfree | all | 65 | 0.00 | 0.11 | False |
| GOOD-D | hpfree | UB9 | 45 | 0.00 | 0.02 | False |
| GOOD-D | ubgold | all | 65 | 0.00 | 0.04 | False |
| GOOD-D | ubgold | UB9 | 45 | 0.00 | 0.10 | False |
| GraphCL_IF | hpfree | all | 65 | 0.35 | -0.48 | False |
| GraphCL_IF | hpfree | UB9 | 45 | 0.40 | -0.51 | False |
| GraphCL_IF | ubgold | all | 65 | 0.00 | -0.72 | False |
| GraphCL_IF | ubgold | UB9 | 45 | 0.00 | -0.89 | False |
| GraphCL_OCSVM | hpfree | all | 65 | 0.35 | 1.06 | False |
| GraphCL_OCSVM | hpfree | UB9 | 45 | 0.47 | 1.29 | False |
| GraphCL_OCSVM | ubgold | all | 65 | 0.45 | -0.98 | False |
| GraphCL_OCSVM | ubgold | UB9 | 45 | 0.60 | -0.37 | False |
| GraphDE | hpfree | all | 65 | 0.17 | -0.55 | False |
| GraphDE | hpfree | UB9 | 45 | 0.07 | -1.81 | False |
| GraphDE | ubgold | all | 65 | 0.09 | -0.13 | False |
| GraphDE | ubgold | UB9 | 45 | 0.09 | -1.23 | False |
| HC-GLAD | hpfree | all | 50 | 0.00 | 0.02 | False |
| HC-GLAD | hpfree | UB9 | 45 | 0.00 | 0.03 | False |
| HC-GLAD | ubgold | all | 50 | 0.04 | -1.35 | False |
| HC-GLAD | ubgold | UB9 | 45 | 0.04 | -1.12 | False |
| InfoGraph_IF | hpfree | all | 65 | 0.91 | -1.64 | False |
| InfoGraph_IF | hpfree | UB9 | 45 | 0.98 | -2.23 | False |
| InfoGraph_IF | ubgold | all | 65 | 0.57 | -3.51 | False |
| InfoGraph_IF | ubgold | UB9 | 45 | 0.67 | -4.35 | False |
| InfoGraph_OCSVM | hpfree | all | 65 | 0.69 | -3.57 | False |
| InfoGraph_OCSVM | hpfree | UB9 | 45 | 0.82 | -4.38 | False |
| InfoGraph_OCSVM | ubgold | all | 65 | 0.51 | -3.21 | False |
| InfoGraph_OCSVM | ubgold | UB9 | 45 | 0.53 | -3.37 | False |
| LGKDE | hpfree | all | 50 | 0.14 | 1.29 | False |
| LGKDE | hpfree | UB9 | 45 | 0.16 | 0.70 | False |
| LGKDE | ubgold | all | 50 | 0.20 | 0.16 | False |
| LGKDE | ubgold | UB9 | 45 | 0.22 | 0.35 | False |
| MUSE | hpfree | all | 50 | 0.20 | 1.77 | False |
| MUSE | hpfree | UB9 | 45 | 0.18 | 1.82 | False |
| MUSE | ubgold | all | 50 | 0.20 | -0.52 | False |
| MUSE | ubgold | UB9 | 45 | 0.22 | -1.10 | False |
| OCGIN | hpfree | all | 65 | 0.00 | 0.22 | False |
| OCGIN | hpfree | UB9 | 45 | 0.00 | -0.01 | False |
| OCGIN | ubgold | all | 70 | 0.00 | 0.22 | False |
| OCGIN | ubgold | UB9 | 45 | 0.00 | -0.35 | False |
| OCGTL | hpfree | all | 65 | 0.00 | -0.02 | False |
| OCGTL | hpfree | UB9 | 45 | 0.00 | -0.03 | False |
| OCGTL | ubgold | all | 70 | 0.00 | -1.14 | False |
| OCGTL | ubgold | UB9 | 45 | 0.00 | -0.19 | False |
| SEGO | hpfree | all | 50 | 0.42 | 9.36 | False |
| SEGO | hpfree | UB9 | 45 | 0.36 | 3.76 | False |
| SEGO | ubgold | all | 50 | 0.46 | 9.71 | False |
| SEGO | ubgold | UB9 | 45 | 0.40 | 3.86 | False |
| SIGNET | hpfree | all | 65 | 0.00 | -0.55 | False |
| SIGNET | hpfree | UB9 | 45 | 0.00 | -0.46 | False |
| SIGNET | ubgold | all | 70 | 0.00 | -0.57 | False |
| SIGNET | ubgold | UB9 | 45 | 0.00 | -0.86 | False |
| SUL-GUL | hpfree | all | 50 | 0.00 | -1.24 | False |
| SUL-GUL | hpfree | UB9 | 45 | 0.00 | -1.36 | False |
| SUL-GUL | ubgold | all | 50 | 0.00 | -1.24 | False |
| SUL-GUL | ubgold | UB9 | 45 | 0.00 | -1.36 | False |

Flagged protocol rows (127; full list in `eval_leaderboard_v2_flags.csv`):

| method | dataset | fold | protocol | a15_action | nonfinite_selected | n_distinct | tie_threshold | tie_collapsed |
|---|---|---|---|---|---|---|---|---|
| GraphCL_OCSVM | COLLAB | 3 | hpfree_oracle |  | 0 | 30.00 | 50.00 | 1.00 |
| GraphCL_OCSVM | Tox21_p53 | 2 | last |  | 0 | 2.00 | 13.45 | 1.00 |
| GraphCL_OCSVM | Tox21_p53 | 3 | last |  | 0 | 8.00 | 13.45 | 1.00 |
| GraphDE | IMDB-BINARY | 3 | hpfree_oracle |  | 0 | 3.00 | 10.00 | 1.00 |
| GraphDE | IMDB-BINARY | 0 | hpfree_valnormal |  | 0 | 2.00 | 10.00 | 1.00 |
| GraphDE | IMDB-BINARY | 1 | hpfree_valnormal |  | 0 | 3.00 | 10.00 | 1.00 |
| GraphDE | IMDB-BINARY | 2 | hpfree_valnormal |  | 0 | 3.00 | 10.00 | 1.00 |
| GraphDE | IMDB-BINARY | 3 | hpfree_valnormal |  | 0 | 3.00 | 10.00 | 1.00 |
| GraphDE | IMDB-BINARY | 3 | valnormal |  | 0 | 3.00 | 10.00 | 1.00 |
| GraphDE | Tox21_MMP | 1 | hpfree_valnormal |  | 0 | 1.00 | 11.90 | 1.00 |
| GraphDE | Tox21_MMP | 2 | hpfree_valnormal |  | 0 | 1.00 | 11.90 | 1.00 |
| GraphDE | Tox21_MMP | 2 | valnormal |  | 0 | 1.00 | 11.90 | 1.00 |
| GraphDE | Tox21_MMP | 3 | valnormal |  | 0 | 1.00 | 11.90 | 1.00 |
| GraphDE | Tox21_MMP | 4 | valnormal |  | 0 | 1.00 | 11.90 | 1.00 |
| GraphDE | Tox21_p53 | 3 | hpfree_valnormal |  | 0 | 1.00 | 13.45 | 1.00 |
| GraphDE | Tox21_p53 | 3 | oracle |  | 0 | 1.00 | 13.45 | 1.00 |
| GraphDE | Tox21_p53 | 2 | valnormal |  | 0 | 1.00 | 13.45 | 1.00 |
| GraphDE | Tox21_p53 | 3 | valnormal |  | 0 | 1.00 | 13.45 | 1.00 |
| OCGTL | Tox21_p53 | 0 | last |  | 0 | 3.00 | 13.45 | 1.00 |
| OCGTL | Tox21_p53 | 1 | last |  | 0 | 2.00 | 13.45 | 1.00 |
| OCGTL | Tox21_p53 | 2 | last |  | 0 | 2.00 | 13.45 | 1.00 |
| OCGTL | Tox21_p53 | 3 | last |  | 0 | 2.00 | 13.45 | 1.00 |
| OCGTL | Tox21_p53 | 4 | last |  | 0 | 3.00 | 13.45 | 1.00 |
| GraphCL_OCSVM | COLLAB | 3 | hpfree_oracle_ubgold |  | 0 | 30.00 | 50.00 | 1.00 |
| GraphDE | IMDB-BINARY | 3 | hpfree_oracle_ubgold |  | 0 | 3.00 | 10.00 | 1.00 |
| GraphDE | Tox21_p53 | 3 | oracle_ubgold |  | 0 | 1.00 | 13.45 | 1.00 |
| GLADPro | IMDB-BINARY | 0 | last |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 0 | valnormal |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 1 | last |  | 0 | 4.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 1 | valnormal |  | 0 | 4.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 2 | last |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 2 | valnormal |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 3 | last |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 3 | valnormal |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 4 | last |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 4 | valnormal |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 2 | oracle |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 0 | hpfree_last |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 0 | hpfree_valnormal |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 1 | hpfree_last |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 1 | hpfree_valnormal |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 2 | hpfree_last |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 2 | hpfree_valnormal |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 3 | hpfree_last |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 3 | hpfree_valnormal |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 4 | hpfree_last |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | IMDB-BINARY | 4 | hpfree_valnormal |  | 0 | 1.00 | 10.00 | 1.00 |
| GLADPro | REDDIT-BINARY | 1 | last |  | 0 | 1.00 | 20.00 | 1.00 |
| GLADPro | REDDIT-BINARY | 1 | valnormal |  | 0 | 1.00 | 20.00 | 1.00 |
| GLADPro | REDDIT-BINARY | 2 | valnormal |  | 0 | 1.00 | 20.00 | 1.00 |
| GLADPro | REDDIT-BINARY | 1 | oracle |  | 0 | 1.00 | 20.00 | 1.00 |
| GLADPro | REDDIT-BINARY | 1 | hpfree_last |  | 0 | 2.00 | 20.00 | 1.00 |
| GLADPro | REDDIT-BINARY | 1 | hpfree_valnormal |  | 0 | 1.00 | 20.00 | 1.00 |
| GLADPro | REDDIT-BINARY | 2 | hpfree_last |  | 0 | 2.00 | 20.00 | 1.00 |
| GLADPro | REDDIT-BINARY | 2 | hpfree_valnormal |  | 0 | 1.00 | 20.00 | 1.00 |
| GLADPro | REDDIT-BINARY | 3 | hpfree_last |  | 0 | 2.00 | 20.00 | 1.00 |
| GLADPro | REDDIT-BINARY | 1 | hpfree_oracle |  | 0 | 1.00 | 20.00 | 1.00 |
| ProtoGLAD | COX2 | 0 | oracle |  | 0 | 3.00 | 10.00 | 1.00 |
| ProtoGLAD | COX2 | 0 | oracle_ubgold |  | 0 | 3.00 | 10.00 | 1.00 |
| ProtoGLAD | COX2 | 0 | last |  | 0 | 3.00 | 10.00 | 1.00 |

Sensitivity analysis (A15 §5, A21 §3): flagged runs dropped.

| variant | n_runs_dropped | quantity | n_methods | mean_over_methods | ds_ci_lo | ds_ci_hi | n_methods_ge2 | g1_conditions_met |
|---|---|---|---|---|---|---|---|---|
| drop nonfinite_selected | 12 | G1/VAL-NORMAL | 12 | 4.13 | 3.32 | 4.79 | 12.00 | True |
| drop nonfinite_selected | 12 | G1/LAST | 12 | 5.12 | 3.97 | 6.24 | 12.00 | True |
| drop nonfinite_selected | 12 | G2b HP leak | 12 | 1.53 | 0.64 | 2.42 | 3.00 | – |
| drop tie_collapsed | 71 | G1/VAL-NORMAL | 12 | 4.13 | 3.32 | 4.79 | 12.00 | True |
| drop tie_collapsed | 71 | G1/LAST | 12 | 5.11 | 3.97 | 6.23 | 12.00 | True |
| drop tie_collapsed | 71 | G2b HP leak | 12 | 1.54 | 0.65 | 2.43 | 3.00 | – |
| drop any flag | 71 | G1/VAL-NORMAL | 12 | 4.13 | 3.32 | 4.79 | 12.00 | True |
| drop any flag | 71 | G1/LAST | 12 | 5.11 | 3.97 | 6.23 | 12.00 | True |
| drop any flag | 71 | G2b HP leak | 12 | 1.54 | 0.65 | 2.43 | 3.00 | – |

## 5. Leaderboards (per stratum and pooled)

### UB9 / frozen12

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| SIGNET | frozen12 | 74.72 (#2) | 74.72 (#2) | 70.71 (#2) | 71.57 (#1) | 72.93 (#1) | 72.93 (#1) | 68.82 (#1) | 69.28 (#1) |
| CVTGAD | frozen12 | 76.08 (#1) | 76.08 (#1) | 70.96 (#1) | 71.05 (#2) | 69.83 (#4) | 69.83 (#4) | 66.43 (#3) | 66.53 (#3) |
| OCGTL | frozen12 | 72.06 (#4) | 72.06 (#4) | 68.65 (#3) | 68.85 (#3) | 69.38 (#6) | 69.38 (#6) | 60.07 (#10) | 60.09 (#11) |
| GOOD-D | frozen12 | 72.27 (#3) | 72.27 (#3) | 67.71 (#4) | 67.62 (#4) | 70.94 (#2) | 70.94 (#2) | 66.84 (#2) | 66.82 (#2) |
| OCGIN | frozen12 | 70.15 (#5) | 70.15 (#5) | 64.47 (#5) | 64.81 (#5) | 67.93 (#8) | 67.93 (#8) | 63.84 (#6) | 63.85 (#7) |
| InfoGraph_OCSVM | frozen12 | 67.67 (#9) | 67.67 (#9) | 61.35 (#10) | 64.72 (#6) | 66.89 (#10) | 66.89 (#10) | 59.77 (#11) | 64.15 (#5) |
| InfoGraph_IF | frozen12 | 69.46 (#6) | 69.46 (#6) | 59.79 (#11) | 64.14 (#7) | 69.69 (#5) | 69.69 (#5) | 61.20 (#8) | 63.43 (#8) |
| GraphCL_IF | frozen12 | 67.12 (#10) | 67.12 (#10) | 63.12 (#7) | 64.02 (#8) | 68.94 (#7) | 68.94 (#7) | 63.41 (#7) | 63.92 (#6) |
| GLADC | frozen12 | 68.22 (#8) | 68.22 (#8) | 63.15 (#6) | 63.40 (#9) | 67.75 (#9) | 67.75 (#9) | 64.27 (#4) | 64.23 (#4) |
| GraphCL_OCSVM | frozen12 | 68.90 (#7) | 68.90 (#7) | 62.53 (#8) | 62.89 (#10) | 69.86 (#3) | 69.86 (#3) | 64.01 (#5) | 62.72 (#9) |
| GLocalKD | frozen12 | 66.36 (#11) | 66.36 (#11) | 62.51 (#9) | 62.52 (#11) | 63.20 (#11) | 63.20 (#11) | 61.08 (#9) | 61.05 (#10) |
| GraphDE (paper orientation, A5) | frozen12 | 54.41 (#12) | 54.41 (#12) | 51.03 (#12) | 52.26 (#12) | 55.32 (#12) | 55.32 (#12) | 51.56 (#12) | 53.36 (#12) |

### UB9 / E5

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| SEGO ('ubgold' = released-modal) | E5 | 72.74 (#2) | 72.74 (#2) | 68.36 (#2) | 64.49 (#1) | 74.23 (#1) | 74.23 (#1) | 70.51 (#1) | 66.75 (#1) |
| GLADMamba | E5 | 73.19 (#1) | 74.17 (#1) | 69.80 (#1) | 63.74 (#2) | 71.96 (#2) | 73.61 (#2) | 67.08 (#2) | 63.31 (#2) |
| MUSE | E5 | 65.58 (#3) | 65.58 (#3) | 57.91 (#3) | 59.01 (#3) | 64.29 (#3) | 64.29 (#4) | 56.64 (#3) | 54.81 (#3) |
| LGKDE ('ubgold' = MUTAG.yaml for all) | E5 | 52.84 (#5) | 56.76 (#5) | 54.47 (#4) | 54.11 (#4) | 53.21 (#5) | 57.09 (#5) | 52.18 (#4) | 51.48 (#4) |
| GLADPro | E5 | 59.01 (#4) | 64.94 (#4) | 46.79 (#5) | 46.26 (#5) | 61.18 (#4) | 64.69 (#3) | 50.42 (#5) | 48.89 (#5) |

### UB9 / E4

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| GLASS | E4 | 72.77 (#1) | 70.09 (#1) | 63.83 (#1) | 64.86 (#1) | 72.89 (#1) | 70.23 (#1) | 63.77 (#2) | 64.54 (#1) |
| HC-GLAD | E4 | 66.52 (#2) | 65.68 (#2) | 63.12 (#3) | 64.24 (#2) | 64.73 (#2) | 65.64 (#2) | 64.32 (#1) | 64.29 (#2) |
| ProtoGLAD | E4 | 63.43 (#3) | 63.43 (#3) | 63.43 (#2) | 63.43 (#3) | 59.57 (#3) | 59.57 (#3) | 59.57 (#3) | 59.57 (#3) |
| SUL-GUL (GUL only; SUL not released) | E4 | 58.25 (#4) | 56.15 (#4) | 48.30 (#4) | 49.66 (#4) | 58.25 (#4) | 56.15 (#4) | 48.30 (#4) | 49.66 (#4) |

### UB9 / recent(E5+E4)

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| GLASS | E4 | 72.77 (#2) | 70.09 (#3) | 63.83 (#3) | 64.86 (#1) | 72.89 (#2) | 70.23 (#3) | 63.77 (#4) | 64.54 (#2) |
| SEGO ('ubgold' = released-modal) | E5 | 72.74 (#3) | 72.74 (#2) | 68.36 (#2) | 64.49 (#2) | 74.23 (#1) | 74.23 (#1) | 70.51 (#1) | 66.75 (#1) |
| HC-GLAD | E4 | 66.52 (#4) | 65.68 (#4) | 63.12 (#5) | 64.24 (#3) | 64.73 (#4) | 65.64 (#4) | 64.32 (#3) | 64.29 (#3) |
| GLADMamba | E5 | 73.19 (#1) | 74.17 (#1) | 69.80 (#1) | 63.74 (#4) | 71.96 (#3) | 73.61 (#2) | 67.08 (#2) | 63.31 (#4) |
| ProtoGLAD | E4 | 63.43 (#6) | 63.43 (#7) | 63.43 (#4) | 63.43 (#5) | 59.57 (#7) | 59.57 (#7) | 59.57 (#5) | 59.57 (#5) |
| MUSE | E5 | 65.58 (#5) | 65.58 (#5) | 57.91 (#6) | 59.01 (#6) | 64.29 (#5) | 64.29 (#6) | 56.64 (#6) | 54.81 (#6) |
| LGKDE ('ubgold' = MUTAG.yaml for all) | E5 | 52.84 (#9) | 56.76 (#8) | 54.47 (#7) | 54.11 (#7) | 53.21 (#9) | 57.09 (#8) | 52.18 (#7) | 51.48 (#7) |
| SUL-GUL (GUL only; SUL not released) | E4 | 58.25 (#8) | 56.15 (#9) | 48.30 (#8) | 49.66 (#8) | 58.25 (#8) | 56.15 (#9) | 48.30 (#9) | 49.66 (#8) |
| GLADPro | E5 | 59.01 (#7) | 64.94 (#6) | 46.79 (#9) | 46.26 (#9) | 61.18 (#6) | 64.69 (#5) | 50.42 (#8) | 48.89 (#9) |

### UB9 / frozen12+E5+E4

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| SIGNET | frozen12 | 74.72 (#2) | 74.72 (#2) | 70.71 (#2) | 71.57 (#1) | 72.93 (#2) | 72.93 (#3) | 68.82 (#2) | 69.28 (#1) |
| CVTGAD | frozen12 | 76.08 (#1) | 76.08 (#1) | 70.96 (#1) | 71.05 (#2) | 69.83 (#7) | 69.83 (#7) | 66.43 (#5) | 66.53 (#4) |
| OCGTL | frozen12 | 72.06 (#7) | 72.06 (#6) | 68.65 (#4) | 68.85 (#3) | 69.38 (#9) | 69.38 (#9) | 60.07 (#14) | 60.09 (#15) |
| GOOD-D | frozen12 | 72.27 (#6) | 72.27 (#5) | 67.71 (#6) | 67.62 (#4) | 70.94 (#5) | 70.94 (#4) | 66.84 (#4) | 66.82 (#2) |
| GLASS | E4 | 72.77 (#4) | 70.09 (#8) | 63.83 (#8) | 64.86 (#5) | 72.89 (#3) | 70.23 (#5) | 63.77 (#10) | 64.54 (#5) |
| OCGIN | frozen12 | 70.15 (#8) | 70.15 (#7) | 64.47 (#7) | 64.81 (#6) | 67.93 (#11) | 67.93 (#11) | 63.84 (#9) | 63.85 (#10) |
| InfoGraph_OCSVM | frozen12 | 67.67 (#12) | 67.67 (#12) | 61.35 (#15) | 64.72 (#7) | 66.89 (#13) | 66.89 (#13) | 59.77 (#15) | 64.15 (#8) |
| SEGO ('ubgold' = released-modal) | E5 | 72.74 (#5) | 72.74 (#4) | 68.36 (#5) | 64.49 (#8) | 74.23 (#1) | 74.23 (#1) | 70.51 (#1) | 66.75 (#3) |
| HC-GLAD | E4 | 66.52 (#14) | 65.68 (#15) | 63.12 (#12) | 64.24 (#9) | 64.73 (#14) | 65.64 (#14) | 64.32 (#6) | 64.29 (#6) |
| InfoGraph_IF | frozen12 | 69.46 (#9) | 69.46 (#9) | 59.79 (#16) | 64.14 (#10) | 69.69 (#8) | 69.69 (#8) | 61.20 (#12) | 63.43 (#11) |
| GraphCL_IF | frozen12 | 67.12 (#13) | 67.12 (#13) | 63.12 (#11) | 64.02 (#11) | 68.94 (#10) | 68.94 (#10) | 63.41 (#11) | 63.92 (#9) |
| GLADMamba | E5 | 73.19 (#3) | 74.17 (#3) | 69.80 (#3) | 63.74 (#12) | 71.96 (#4) | 73.61 (#2) | 67.08 (#3) | 63.31 (#12) |
| ProtoGLAD | E4 | 63.43 (#17) | 63.43 (#18) | 63.43 (#9) | 63.43 (#13) | 59.57 (#18) | 59.57 (#18) | 59.57 (#16) | 59.57 (#16) |
| GLADC | frozen12 | 68.22 (#11) | 68.22 (#11) | 63.15 (#10) | 63.40 (#14) | 67.75 (#12) | 67.75 (#12) | 64.27 (#7) | 64.23 (#7) |
| GraphCL_OCSVM | frozen12 | 68.90 (#10) | 68.90 (#10) | 62.53 (#13) | 62.89 (#15) | 69.86 (#6) | 69.86 (#6) | 64.01 (#8) | 62.72 (#13) |
| GLocalKD | frozen12 | 66.36 (#15) | 66.36 (#14) | 62.51 (#14) | 62.52 (#16) | 63.20 (#16) | 63.20 (#17) | 61.08 (#13) | 61.05 (#14) |
| MUSE | E5 | 65.58 (#16) | 65.58 (#16) | 57.91 (#17) | 59.01 (#17) | 64.29 (#15) | 64.29 (#16) | 56.64 (#17) | 54.81 (#17) |
| LGKDE ('ubgold' = MUTAG.yaml for all) | E5 | 52.84 (#21) | 56.76 (#19) | 54.47 (#18) | 54.11 (#18) | 53.21 (#21) | 57.09 (#19) | 52.18 (#18) | 51.48 (#19) |
| GraphDE (paper orientation, A5) | frozen12 | 54.41 (#20) | 54.41 (#21) | 51.03 (#19) | 52.26 (#19) | 55.32 (#20) | 55.32 (#21) | 51.56 (#19) | 53.36 (#18) |
| SUL-GUL (GUL only; SUL not released) | E4 | 58.25 (#19) | 56.15 (#20) | 48.30 (#20) | 49.66 (#20) | 58.25 (#19) | 56.15 (#20) | 48.30 (#21) | 49.66 (#20) |
| GLADPro | E5 | 59.01 (#18) | 64.94 (#17) | 46.79 (#21) | 46.26 (#21) | 61.18 (#17) | 64.69 (#15) | 50.42 (#20) | 48.89 (#21) |

### UB9 / all

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| SIGNET | frozen12 | 74.72 (#2) | 74.72 (#2) | 70.71 (#2) | 71.57 (#1) | 72.93 (#2) | 72.93 (#3) | 68.82 (#2) | 69.28 (#1) |
| CVTGAD | frozen12 | 76.08 (#1) | 76.08 (#1) | 70.96 (#1) | 71.05 (#2) | 69.83 (#7) | 69.83 (#7) | 66.43 (#5) | 66.53 (#4) |
| OCGTL | frozen12 | 72.06 (#7) | 72.06 (#6) | 68.65 (#4) | 68.85 (#3) | 69.38 (#9) | 69.38 (#9) | 60.07 (#16) | 60.09 (#17) |
| GOOD-D | frozen12 | 72.27 (#6) | 72.27 (#5) | 67.71 (#6) | 67.62 (#4) | 70.94 (#5) | 70.94 (#4) | 66.84 (#4) | 66.82 (#2) |
| GLASS | E4 | 72.77 (#4) | 70.09 (#8) | 63.83 (#8) | 64.86 (#5) | 72.89 (#3) | 70.23 (#5) | 63.77 (#10) | 64.54 (#5) |
| OCGIN | frozen12 | 70.15 (#8) | 70.15 (#7) | 64.47 (#7) | 64.81 (#6) | 67.93 (#11) | 67.93 (#11) | 63.84 (#9) | 63.85 (#10) |
| InfoGraph_OCSVM | frozen12 | 67.67 (#12) | 67.67 (#12) | 61.35 (#16) | 64.72 (#7) | 66.89 (#13) | 66.89 (#13) | 59.77 (#17) | 64.15 (#8) |
| SEGO ('ubgold' = released-modal) | E5 | 72.74 (#5) | 72.74 (#4) | 68.36 (#5) | 64.49 (#8) | 74.23 (#1) | 74.23 (#1) | 70.51 (#1) | 66.75 (#3) |
| HC-GLAD | E4 | 66.52 (#14) | 65.68 (#15) | 63.12 (#12) | 64.24 (#9) | 64.73 (#14) | 65.64 (#14) | 64.32 (#6) | 64.29 (#6) |
| InfoGraph_IF | frozen12 | 69.46 (#9) | 69.46 (#9) | 59.79 (#18) | 64.14 (#10) | 69.69 (#8) | 69.69 (#8) | 61.20 (#13) | 63.43 (#11) |
| GraphCL_IF | frozen12 | 67.12 (#13) | 67.12 (#13) | 63.12 (#11) | 64.02 (#11) | 68.94 (#10) | 68.94 (#10) | 63.41 (#11) | 63.92 (#9) |
| GLADMamba | E5 | 73.19 (#3) | 74.17 (#3) | 69.80 (#3) | 63.74 (#12) | 71.96 (#4) | 73.61 (#2) | 67.08 (#3) | 63.31 (#12) |
| ProtoGLAD | E4 | 63.43 (#17) | 63.43 (#18) | 63.43 (#9) | 63.43 (#13) | 59.57 (#20) | 59.57 (#20) | 59.57 (#18) | 59.57 (#18) |
| GLADC | frozen12 | 68.22 (#11) | 68.22 (#11) | 63.15 (#10) | 63.40 (#14) | 67.75 (#12) | 67.75 (#12) | 64.27 (#7) | 64.23 (#7) |
| GraphCL_OCSVM | frozen12 | 68.90 (#10) | 68.90 (#10) | 62.53 (#13) | 62.89 (#15) | 69.86 (#6) | 69.86 (#6) | 64.01 (#8) | 62.72 (#13) |
| GLocalKD | frozen12 | 66.36 (#15) | 66.36 (#14) | 62.51 (#14) | 62.52 (#16) | 63.20 (#16) | 63.20 (#17) | 61.08 (#14) | 61.05 (#15) |
| GRAIL-0 | reference | 62.10 (#18) | 62.10 (#19) | 62.10 (#15) | 62.10 (#17) | 62.10 (#17) | 62.10 (#18) | 62.10 (#12) | 62.10 (#14) |
| graphlet-count+LW | reference | 60.39 (#19) | 60.39 (#20) | 60.39 (#17) | 60.39 (#18) | 60.39 (#19) | 60.39 (#19) | 60.39 (#15) | 60.39 (#16) |
| MUSE | E5 | 65.58 (#16) | 65.58 (#16) | 57.91 (#21) | 59.01 (#19) | 64.29 (#15) | 64.29 (#16) | 56.64 (#22) | 54.81 (#23) |
| WL-IF | kernel | 58.65 (#21) | 58.65 (#21) | 58.65 (#19) | 58.65 (#20) | 56.79 (#24) | 56.79 (#24) | 56.79 (#21) | 56.79 (#21) |
| graphlet-count+iForest | reference | 58.19 (#23) | 58.19 (#22) | 58.19 (#20) | 58.19 (#21) | 58.19 (#22) | 58.19 (#21) | 58.19 (#19) | 58.19 (#19) |
| size-only | reference | 57.25 (#24) | 57.25 (#23) | 57.25 (#22) | 57.25 (#22) | 57.25 (#23) | 57.25 (#22) | 57.25 (#20) | 57.25 (#20) |
| WL-hist+iForest (ctrl) | reference | 55.81 (#25) | 55.81 (#26) | 55.81 (#23) | 55.81 (#23) | 55.81 (#25) | 55.81 (#26) | 55.81 (#23) | 55.81 (#22) |
| PK-IF | kernel | 54.53 (#26) | 54.53 (#27) | 54.53 (#24) | 54.53 (#24) | 50.93 (#28) | 50.93 (#28) | 50.93 (#26) | 50.93 (#26) |
| LGKDE ('ubgold' = MUTAG.yaml for all) | E5 | 52.84 (#28) | 56.76 (#24) | 54.47 (#25) | 54.11 (#25) | 53.21 (#27) | 57.09 (#23) | 52.18 (#24) | 51.48 (#25) |
| GraphDE (paper orientation, A5) | frozen12 | 54.41 (#27) | 54.41 (#28) | 51.03 (#26) | 52.26 (#26) | 55.32 (#26) | 55.32 (#27) | 51.56 (#25) | 53.36 (#24) |
| SUL-GUL (GUL only; SUL not released) | E4 | 58.25 (#22) | 56.15 (#25) | 48.30 (#28) | 49.66 (#27) | 58.25 (#21) | 56.15 (#25) | 48.30 (#28) | 49.66 (#27) |
| PK-OCSVM | kernel | 48.81 (#29) | 48.81 (#29) | 48.81 (#27) | 48.81 (#28) | 47.19 (#29) | 47.19 (#29) | 47.19 (#29) | 47.19 (#29) |
| GLADPro | E5 | 59.01 (#20) | 64.94 (#17) | 46.79 (#29) | 46.26 (#29) | 61.18 (#18) | 64.69 (#15) | 50.42 (#27) | 48.89 (#28) |
| WL-OCSVM | kernel | 43.22 (#30) | 43.22 (#30) | 43.22 (#30) | 43.22 (#30) | 41.97 (#30) | 41.97 (#30) | 41.97 (#30) | 41.97 (#30) |

### secondary / frozen12

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| GOOD-D | frozen12 | 71.03* | 71.03* | 66.14* | 66.24* | 71.20* | 71.20* | 65.47* | 65.14* |
| OCGIN | frozen12 | 68.95 (#2) | 68.95 (#2) | 66.40 (#1) | 65.16 (#1) | 67.45* | 67.45* | 62.03* | 61.29* |
| CVTGAD | frozen12 | 69.94* | 69.94* | 65.32* | 65.12* | 69.48* | 69.48* | 66.46* | 66.46* |
| InfoGraph_OCSVM | frozen12 | 66.69* | 66.69* | 62.24* | 65.09* | 66.93* | 66.93* | 62.46* | 64.20* |
| OCGTL | frozen12 | 68.62 (#3) | 68.62 (#3) | 61.86 (#3) | 64.72 (#2) | 66.45* | 66.45* | 56.46* | 56.46* |
| InfoGraph_IF | frozen12 | 66.98* | 66.98* | 62.63* | 64.25* | 67.75* | 67.75* | 62.95* | 63.26* |
| SIGNET | frozen12 | 75.10 (#1) | 75.10 (#1) | 63.02 (#2) | 63.07 (#3) | 68.28* | 68.28* | 61.58* | 62.34* |
| GraphCL_IF | frozen12 | 65.96* | 65.96* | 62.40* | 62.74* | 67.27* | 67.27* | 62.38* | 62.77* |
| GraphCL_OCSVM | frozen12 | 68.03* | 68.03* | 60.14* | 62.49* | 69.03* | 69.03* | 61.60* | 61.05* |
| GLADC | frozen12 | 63.46* | 63.46* | 62.51* | 62.42* | 59.14* | 59.14* | 59.03* | 59.10* |
| GraphDE (paper orientation, A5) | frozen12 | 52.32* | 52.32* | 49.70* | 47.36* | 52.82* | 52.82* | 48.71* | 46.41* |
| GLocalKD | frozen12 | 44.54 (#4) | 44.54 (#4) | 41.00 (#4) | 41.00 (#4) | 59.94* | 59.94* | 59.89* | 59.89* |

### secondary / E5

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| GLADPro | E5 | 87.48* | 73.57* | 86.71* | 86.39* | 86.75* | 85.35* | 48.31* | 62.97* |
| LGKDE ('ubgold' = MUTAG.yaml for all) | E5 | 88.21* | 88.84* | 70.33* | 71.95* | 88.19* | 88.32* | 73.42* | 66.74* |
| MUSE | E5 | 82.41* | 82.41* | 75.74* | 71.05* | 87.56* | 87.56* | 80.51* | 79.23* |
| GLADMamba | E5 | 67.36* | 73.26* | 66.81* | 23.90* | 68.05* | 69.14* | 68.05* | 35.24* |
| SEGO ('ubgold' = released-modal) | E5 | 83.46* | 83.46* | 76.81* | 14.50* | 82.88* | 82.88* | 79.13* | 19.38* |

### secondary / E4

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| GLASS | E4 | 91.13* | 91.49* | 89.57* | 86.92* | 91.13* | 91.49* | 89.57* | 86.92* |
| ProtoGLAD | E4 | 80.87* | 80.87* | 80.87* | 80.87* | 62.16* | 62.16* | 62.16* | 62.16* |
| SUL-GUL (GUL only; SUL not released) | E4 | 91.74* | 90.89* | 78.24* | 78.44* | 91.74* | 90.89* | 78.24* | 78.44* |
| HC-GLAD | E4 | 32.62* | 15.55* | 28.10* | 31.54* | 15.73* | 16.78* | 13.20* | 13.20* |

### secondary / recent(E5+E4)

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| GLASS | E4 | 91.13* | 91.49* | 89.57* | 86.92* | 91.13* | 91.49* | 89.57* | 86.92* |
| GLADPro | E5 | 87.48* | 73.57* | 86.71* | 86.39* | 86.75* | 85.35* | 48.31* | 62.97* |
| ProtoGLAD | E4 | 80.87* | 80.87* | 80.87* | 80.87* | 62.16* | 62.16* | 62.16* | 62.16* |
| SUL-GUL (GUL only; SUL not released) | E4 | 91.74* | 90.89* | 78.24* | 78.44* | 91.74* | 90.89* | 78.24* | 78.44* |
| LGKDE ('ubgold' = MUTAG.yaml for all) | E5 | 88.21* | 88.84* | 70.33* | 71.95* | 88.19* | 88.32* | 73.42* | 66.74* |
| MUSE | E5 | 82.41* | 82.41* | 75.74* | 71.05* | 87.56* | 87.56* | 80.51* | 79.23* |
| HC-GLAD | E4 | 32.62* | 15.55* | 28.10* | 31.54* | 15.73* | 16.78* | 13.20* | 13.20* |
| GLADMamba | E5 | 67.36* | 73.26* | 66.81* | 23.90* | 68.05* | 69.14* | 68.05* | 35.24* |
| SEGO ('ubgold' = released-modal) | E5 | 83.46* | 83.46* | 76.81* | 14.50* | 82.88* | 82.88* | 79.13* | 19.38* |

### secondary / frozen12+E5+E4

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| GLASS | E4 | 91.13* | 91.49* | 89.57* | 86.92* | 91.13* | 91.49* | 89.57* | 86.92* |
| GLADPro | E5 | 87.48* | 73.57* | 86.71* | 86.39* | 86.75* | 85.35* | 48.31* | 62.97* |
| ProtoGLAD | E4 | 80.87* | 80.87* | 80.87* | 80.87* | 62.16* | 62.16* | 62.16* | 62.16* |
| SUL-GUL (GUL only; SUL not released) | E4 | 91.74* | 90.89* | 78.24* | 78.44* | 91.74* | 90.89* | 78.24* | 78.44* |
| LGKDE ('ubgold' = MUTAG.yaml for all) | E5 | 88.21* | 88.84* | 70.33* | 71.95* | 88.19* | 88.32* | 73.42* | 66.74* |
| MUSE | E5 | 82.41* | 82.41* | 75.74* | 71.05* | 87.56* | 87.56* | 80.51* | 79.23* |
| GOOD-D | frozen12 | 71.03* | 71.03* | 66.14* | 66.24* | 71.20* | 71.20* | 65.47* | 65.14* |
| OCGIN | frozen12 | 68.95 (#2) | 68.95 (#2) | 66.40 (#1) | 65.16 (#1) | 67.45* | 67.45* | 62.03* | 61.29* |
| CVTGAD | frozen12 | 69.94* | 69.94* | 65.32* | 65.12* | 69.48* | 69.48* | 66.46* | 66.46* |
| InfoGraph_OCSVM | frozen12 | 66.69* | 66.69* | 62.24* | 65.09* | 66.93* | 66.93* | 62.46* | 64.20* |
| OCGTL | frozen12 | 68.62 (#3) | 68.62 (#3) | 61.86 (#3) | 64.72 (#2) | 66.45* | 66.45* | 56.46* | 56.46* |
| InfoGraph_IF | frozen12 | 66.98* | 66.98* | 62.63* | 64.25* | 67.75* | 67.75* | 62.95* | 63.26* |
| SIGNET | frozen12 | 75.10 (#1) | 75.10 (#1) | 63.02 (#2) | 63.07 (#3) | 68.28* | 68.28* | 61.58* | 62.34* |
| GraphCL_IF | frozen12 | 65.96* | 65.96* | 62.40* | 62.74* | 67.27* | 67.27* | 62.38* | 62.77* |
| GraphCL_OCSVM | frozen12 | 68.03* | 68.03* | 60.14* | 62.49* | 69.03* | 69.03* | 61.60* | 61.05* |
| GLADC | frozen12 | 63.46* | 63.46* | 62.51* | 62.42* | 59.14* | 59.14* | 59.03* | 59.10* |
| GraphDE (paper orientation, A5) | frozen12 | 52.32* | 52.32* | 49.70* | 47.36* | 52.82* | 52.82* | 48.71* | 46.41* |
| GLocalKD | frozen12 | 44.54 (#4) | 44.54 (#4) | 41.00 (#4) | 41.00 (#4) | 59.94* | 59.94* | 59.89* | 59.89* |
| HC-GLAD | E4 | 32.62* | 15.55* | 28.10* | 31.54* | 15.73* | 16.78* | 13.20* | 13.20* |
| GLADMamba | E5 | 67.36* | 73.26* | 66.81* | 23.90* | 68.05* | 69.14* | 68.05* | 35.24* |
| SEGO ('ubgold' = released-modal) | E5 | 83.46* | 83.46* | 76.81* | 14.50* | 82.88* | 82.88* | 79.13* | 19.38* |

### secondary / all

| method | stratum | oracle | oracle_ubgold | last | valnormal | hpfree_oracle | hpfree_oracle_ubgold | hpfree_last | hpfree_valnormal |
|---|---|---|---|---|---|---|---|---|---|
| GLASS | E4 | 91.13* | 91.49* | 89.57* | 86.92* | 91.13* | 91.49* | 89.57* | 86.92* |
| GLADPro | E5 | 87.48* | 73.57* | 86.71* | 86.39* | 86.75* | 85.35* | 48.31* | 62.97* |
| ProtoGLAD | E4 | 80.87* | 80.87* | 80.87* | 80.87* | 62.16* | 62.16* | 62.16* | 62.16* |
| SUL-GUL (GUL only; SUL not released) | E4 | 91.74* | 90.89* | 78.24* | 78.44* | 91.74* | 90.89* | 78.24* | 78.44* |
| LGKDE ('ubgold' = MUTAG.yaml for all) | E5 | 88.21* | 88.84* | 70.33* | 71.95* | 88.19* | 88.32* | 73.42* | 66.74* |
| MUSE | E5 | 82.41* | 82.41* | 75.74* | 71.05* | 87.56* | 87.56* | 80.51* | 79.23* |
| GOOD-D | frozen12 | 71.03* | 71.03* | 66.14* | 66.24* | 71.20* | 71.20* | 65.47* | 65.14* |
| OCGIN | frozen12 | 68.95 (#2) | 68.95 (#2) | 66.40 (#1) | 65.16 (#1) | 67.45* | 67.45* | 62.03* | 61.29* |
| CVTGAD | frozen12 | 69.94* | 69.94* | 65.32* | 65.12* | 69.48* | 69.48* | 66.46* | 66.46* |
| InfoGraph_OCSVM | frozen12 | 66.69* | 66.69* | 62.24* | 65.09* | 66.93* | 66.93* | 62.46* | 64.20* |
| OCGTL | frozen12 | 68.62 (#3) | 68.62 (#3) | 61.86 (#3) | 64.72 (#2) | 66.45* | 66.45* | 56.46* | 56.46* |
| InfoGraph_IF | frozen12 | 66.98* | 66.98* | 62.63* | 64.25* | 67.75* | 67.75* | 62.95* | 63.26* |
| SIGNET | frozen12 | 75.10 (#1) | 75.10 (#1) | 63.02 (#2) | 63.07 (#3) | 68.28* | 68.28* | 61.58* | 62.34* |
| GraphCL_IF | frozen12 | 65.96* | 65.96* | 62.40* | 62.74* | 67.27* | 67.27* | 62.38* | 62.77* |
| GraphCL_OCSVM | frozen12 | 68.03* | 68.03* | 60.14* | 62.49* | 69.03* | 69.03* | 61.60* | 61.05* |
| GLADC | frozen12 | 63.46* | 63.46* | 62.51* | 62.42* | 59.14* | 59.14* | 59.03* | 59.10* |
| graphlet-count+iForest | reference | 58.57 (#4) | 58.57 (#4) | 58.57 (#4) | 58.57 (#4) | 58.57 (#1) | 58.57 (#1) | 58.57 (#1) | 58.57 (#1) |
| size-only | reference | 57.77 (#5) | 57.77 (#5) | 57.77 (#5) | 57.77 (#5) | 57.77 (#2) | 57.77 (#2) | 57.77 (#2) | 57.77 (#2) |
| WL-IF | kernel | 57.65* | 57.65* | 57.65* | 57.65* | 54.49* | 54.49* | 54.49* | 54.49* |
| WL-hist+iForest (ctrl) | reference | 57.61 (#6) | 57.61 (#6) | 57.61 (#6) | 57.61 (#6) | 57.61 (#3) | 57.61 (#3) | 57.61 (#3) | 57.61 (#3) |
| GRAIL-0 | reference | 57.02 (#7) | 57.02 (#7) | 57.02 (#7) | 57.02 (#7) | 57.02 (#4) | 57.02 (#4) | 57.02 (#4) | 57.02 (#4) |
| graphlet-count+LW | reference | 53.63 (#8) | 53.63 (#8) | 53.63 (#8) | 53.63 (#8) | 53.63 (#5) | 53.63 (#5) | 53.63 (#5) | 53.63 (#5) |
| PK-IF | kernel | 49.42* | 49.42* | 49.42* | 49.42* | 47.82* | 47.82* | 47.82* | 47.82* |
| GraphDE (paper orientation, A5) | frozen12 | 52.32* | 52.32* | 49.70* | 47.36* | 52.82* | 52.82* | 48.71* | 46.41* |
| PK-OCSVM | kernel | 44.84* | 44.84* | 44.84* | 44.84* | 44.69* | 44.69* | 44.69* | 44.69* |
| WL-OCSVM | kernel | 41.19* | 41.19* | 41.19* | 41.19* | 37.77* | 37.77* | 37.77* | 37.77* |
| GLocalKD | frozen12 | 44.54 (#9) | 44.54 (#9) | 41.00 (#9) | 41.00 (#9) | 59.94* | 59.94* | 59.89* | 59.89* |
| HC-GLAD | E4 | 32.62* | 15.55* | 28.10* | 31.54* | 15.73* | 16.78* | 13.20* | 13.20* |
| GLADMamba | E5 | 67.36* | 73.26* | 66.81* | 23.90* | 68.05* | 69.14* | 68.05* | 35.24* |
| SEGO ('ubgold' = released-modal) | E5 | 83.46* | 83.46* | 76.81* | 14.50* | 82.88* | 82.88* | 79.13* | 19.38* |

`*` = incomplete coverage of the set (not ranked). ORACLE = own test-label rule where the method has one (oracle_rule in the CSV), else UB-GOLD's rule; oracle_ubgold = UB-GOLD's rule for every iterative method.

## 6. Inflation per method (both selectors; A14) and pooled (A20 exclusions)

| method | stratum | comparison | mean_diff | fold_ci_lo | fold_ci_hi | ds_ci_lo | ds_ci_hi | n_datasets_diff_ge2 | excluded_from_pooled |
|---|---|---|---|---|---|---|---|---|---|
| HC-GLAD | E4 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 2.45 | 1.13 | 3.79 | 1.10 | 4.16 | 4 |  |
| SUL-GUL | E4 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 8.94 | 6.29 | 11.48 | 6.26 | 11.51 | 8 |  |
| GLADMamba | E5 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 3.65 | 2.41 | 5.05 | 1.43 | 6.12 | 3 |  |
| MUSE | E5 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 6.02 | 4.69 | 7.37 | 4.63 | 7.43 | 9 |  |
| SEGO | E5 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 4.23 | 3.38 | 5.13 | 3.01 | 5.86 | 7 |  |
| CVTGAD | frozen12 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 5.02 | 4.20 | 5.90 | 2.77 | 7.43 | 7 |  |
| GraphCL_IF | frozen12 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 4.23 | 3.32 | 5.22 | 2.56 | 6.18 | 6 |  |
| GraphCL_OCSVM | frozen12 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 6.10 | 5.19 | 7.05 | 4.75 | 7.53 | 9 |  |
| GraphDE | frozen12 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 3.19 | 2.08 | 4.32 | 1.35 | 5.26 | 4 |  |
| InfoGraph_IF | frozen12 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 6.76 | 5.70 | 7.89 | 3.14 | 11.57 | 7 |  |
| InfoGraph_OCSVM | frozen12 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 3.91 | 3.07 | 4.86 | 1.06 | 8.25 | 4 |  |
| GLASS | E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 6.26 | 5.41 | 7.07 | 2.31 | 11.22 | 5 |  |
| HC-GLAD | E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 2.56 | 0.94 | 4.31 | 0.93 | 4.08 | 6 |  |
| ProtoGLAD | E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| SUL-GUL | E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 7.85 | 5.68 | 10.23 | 2.48 | 13.36 | 7 |  |
| GLADMamba | E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 4.38 | 3.04 | 5.81 | 2.21 | 6.71 | 6 |  |
| GLADPro | E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 18.16 | 15.85 | 20.42 | 12.58 | 23.64 | 9 |  |
| LGKDE | E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 2.29 | -0.96 | 5.30 | -3.70 | 7.84 | 5 |  |
| MUSE | E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 7.68 | 5.34 | 10.29 | 5.13 | 10.52 | 9 |  |
| SEGO | E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 4.38 | 3.49 | 5.33 | 3.12 | 6.01 | 9 |  |
| CVTGAD | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 5.12 | 4.31 | 6.02 | 2.85 | 7.46 | 7 |  |
| GLADC | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 5.07 | 4.19 | 5.90 | 2.59 | 7.46 | 6 |  |
| GLocalKD | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 3.85 | 2.73 | 5.00 | 1.61 | 6.33 | 5 |  |
| GOOD-D | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 4.55 | 3.34 | 5.80 | 2.74 | 6.38 | 6 |  |
| GraphCL_IF | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 3.99 | 3.06 | 4.95 | 2.81 | 5.37 | 8 |  |
| GraphCL_OCSVM | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 6.37 | 5.37 | 7.40 | 4.71 | 8.20 | 9 |  |
| GraphDE | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 3.38 | 2.34 | 4.43 | 1.43 | 5.49 | 4 |  |
| InfoGraph_IF | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 9.67 | 8.65 | 10.75 | 5.39 | 14.87 | 9 |  |
| InfoGraph_OCSVM | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 6.33 | 5.57 | 7.08 | 2.37 | 11.05 | 4 |  |
| OCGIN | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 5.69 | 4.70 | 6.66 | 3.66 | 7.86 | 8 |  |
| OCGTL | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 3.41 | 2.07 | 4.81 | 1.56 | 5.41 | 5 |  |
| SIGNET | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 4.01 | 2.52 | 5.42 | 0.52 | 7.52 | 4 |  |
| PK-IF | kernel | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| PK-OCSVM | kernel | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-IF | kernel | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-OCSVM | kernel | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| GRAIL-0 | reference | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-hist+iForest (ctrl) | reference | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+LW | reference | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+iForest | reference | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| size-only | reference | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| GLASS | E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 5.23 | 3.66 | 6.94 | 3.88 | 6.93 | 8 |  |
| HC-GLAD | E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 1.44 | -0.17 | 3.06 | -1.01 | 3.47 | 6 |  |
| ProtoGLAD | E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| SUL-GUL | E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 6.49 | 4.34 | 8.73 | -0.62 | 12.80 | 7 |  |
| GLADMamba | E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 10.44 | 9.00 | 11.94 | 5.68 | 15.95 | 7 |  |
| GLADPro | E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 18.68 | 15.86 | 21.34 | 10.87 | 26.20 | 9 |  |
| LGKDE | E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 2.65 | -0.44 | 5.55 | -3.57 | 8.26 | 6 |  |
| MUSE | E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 6.57 | 4.53 | 8.97 | 3.63 | 9.77 | 6 |  |
| SEGO | E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 8.25 | 6.92 | 9.67 | 5.76 | 10.49 | 8 |  |
| CVTGAD | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 5.02 | 4.20 | 5.91 | 2.79 | 7.41 | 7 |  |
| GLADC | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 4.82 | 3.91 | 5.72 | 2.55 | 7.03 | 6 |  |
| GLocalKD | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 3.84 | 2.71 | 4.98 | 1.61 | 6.32 | 5 |  |
| GOOD-D | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 4.65 | 3.32 | 6.07 | 2.69 | 6.74 | 7 |  |
| GraphCL_IF | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 3.10 | 2.27 | 3.96 | 1.55 | 4.82 | 5 |  |
| GraphCL_OCSVM | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 6.00 | 4.51 | 7.55 | 3.65 | 8.46 | 8 |  |
| GraphDE | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 2.15 | 1.55 | 2.78 | 0.84 | 3.68 | 3 |  |
| InfoGraph_IF | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 5.32 | 4.22 | 6.45 | 2.62 | 8.37 | 6 |  |
| InfoGraph_OCSVM | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 2.96 | 2.05 | 3.89 | 1.03 | 5.29 | 4 |  |
| OCGIN | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 5.34 | 4.38 | 6.31 | 3.25 | 7.54 | 7 |  |
| OCGTL | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 3.21 | 1.86 | 4.62 | 1.29 | 5.34 | 4 |  |
| SIGNET | frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 3.15 | 1.81 | 4.45 | -0.19 | 6.70 | 5 |  |
| PK-IF | kernel | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| PK-OCSVM | kernel | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-IF | kernel | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-OCSVM | kernel | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| GRAIL-0 | reference | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-hist+iForest (ctrl) | reference | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+LW | reference | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+iForest | reference | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| size-only | reference | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| GLASS | E4 | checkpoint, HP-FREE: ORACLE - LAST | 9.12 | 7.98 | 10.24 | 5.82 | 13.28 | 9 |  |
| HC-GLAD | E4 | checkpoint, HP-FREE: ORACLE - LAST | 0.41 | -0.27 | 1.05 | 0.13 | 0.75 | 0 |  |
| ProtoGLAD | E4 | checkpoint, HP-FREE: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| SUL-GUL | E4 | checkpoint, HP-FREE: ORACLE - LAST | 9.95 | 7.44 | 12.58 | 5.62 | 14.57 | 8 |  |
| GLADMamba | E5 | checkpoint, HP-FREE: ORACLE - LAST | 4.87 | 3.73 | 6.04 | 1.90 | 8.25 | 5 |  |
| GLADPro | E5 | checkpoint, HP-FREE: ORACLE - LAST | 10.76 | 8.02 | 13.49 | 7.52 | 13.86 | 9 |  |
| LGKDE | E5 | checkpoint, HP-FREE: ORACLE - LAST | 1.03 | -1.07 | 3.18 | -9.88 | 9.19 | 7 |  |
| MUSE | E5 | checkpoint, HP-FREE: ORACLE - LAST | 7.66 | 6.26 | 9.07 | 5.60 | 10.31 | 9 |  |
| SEGO | E5 | checkpoint, HP-FREE: ORACLE - LAST | 3.72 | 3.03 | 4.49 | 1.80 | 6.46 | 6 |  |
| CVTGAD | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 3.39 | 2.07 | 4.66 | 2.47 | 4.22 | 7 |  |
| GLADC | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 3.48 | 2.88 | 4.10 | 1.56 | 5.81 | 6 |  |
| GLocalKD | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 2.11 | 1.25 | 3.10 | 0.78 | 3.69 | 3 |  |
| GOOD-D | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 4.10 | 3.05 | 5.15 | 2.60 | 5.54 | 7 |  |
| GraphCL_IF | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 5.53 | 4.40 | 6.66 | 3.54 | 7.56 | 8 |  |
| GraphCL_OCSVM | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 5.85 | 4.73 | 6.99 | 4.01 | 7.65 | 8 |  |
| GraphDE | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 3.76 | 2.67 | 4.99 | 1.05 | 7.35 | 3 |  |
| InfoGraph_IF | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 8.49 | 7.02 | 9.96 | 5.29 | 11.95 | 8 |  |
| InfoGraph_OCSVM | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 7.12 | 5.86 | 8.40 | 4.03 | 10.30 | 7 |  |
| OCGIN | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 4.09 | 3.37 | 4.83 | 2.20 | 6.14 | 6 |  |
| OCGTL | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 9.31 | 7.60 | 11.10 | 4.34 | 14.45 | 7 |  |
| SIGNET | frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 4.11 | 2.49 | 5.73 | 1.26 | 7.20 | 5 |  |
| PK-IF | kernel | checkpoint, HP-FREE: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| PK-OCSVM | kernel | checkpoint, HP-FREE: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-IF | kernel | checkpoint, HP-FREE: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-OCSVM | kernel | checkpoint, HP-FREE: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| GRAIL-0 | reference | checkpoint, HP-FREE: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-hist+iForest (ctrl) | reference | checkpoint, HP-FREE: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+LW | reference | checkpoint, HP-FREE: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+iForest | reference | checkpoint, HP-FREE: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| size-only | reference | checkpoint, HP-FREE: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| GLASS | E4 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 8.35 | 6.58 | 10.23 | 6.27 | 10.65 | 9 |  |
| HC-GLAD | E4 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 0.43 | -0.24 | 1.07 | 0.16 | 0.77 | 0 |  |
| ProtoGLAD | E4 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| SUL-GUL | E4 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 8.59 | 6.55 | 10.87 | 5.05 | 12.70 | 8 |  |
| GLADMamba | E5 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 8.64 | 6.95 | 10.65 | 3.25 | 14.36 | 5 |  |
| GLADPro | E5 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 12.30 | 8.54 | 15.75 | 8.29 | 16.60 | 8 |  |
| LGKDE | E5 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 1.73 | -0.30 | 3.80 | -9.05 | 9.57 | 7 |  |
| MUSE | E5 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 9.48 | 6.87 | 12.20 | 5.03 | 14.97 | 8 |  |
| SEGO | E5 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 7.48 | 5.89 | 9.11 | 4.53 | 10.51 | 8 |  |
| CVTGAD | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 3.30 | 1.90 | 4.60 | 2.34 | 4.18 | 7 |  |
| GLADC | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 3.52 | 2.83 | 4.21 | 1.54 | 5.81 | 6 |  |
| GLocalKD | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 2.15 | 1.31 | 3.13 | 0.73 | 3.78 | 3 |  |
| GOOD-D | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 4.12 | 2.96 | 5.39 | 2.57 | 5.76 | 7 |  |
| GraphCL_IF | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 5.02 | 4.13 | 5.91 | 3.22 | 6.96 | 8 |  |
| GraphCL_OCSVM | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 7.13 | 5.81 | 8.49 | 4.17 | 10.30 | 8 |  |
| GraphDE | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 1.95 | 1.39 | 2.56 | 0.89 | 3.18 | 4 |  |
| InfoGraph_IF | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 6.26 | 4.74 | 7.79 | 3.46 | 9.06 | 7 |  |
| InfoGraph_OCSVM | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 2.74 | 1.69 | 3.91 | 1.62 | 3.82 | 6 |  |
| OCGIN | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 4.08 | 3.36 | 4.82 | 2.17 | 6.13 | 6 |  |
| OCGTL | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 9.29 | 7.57 | 11.10 | 4.28 | 14.43 | 7 |  |
| SIGNET | frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 3.65 | 2.05 | 5.30 | 0.93 | 6.57 | 5 |  |
| PK-IF | kernel | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| PK-OCSVM | kernel | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-IF | kernel | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-OCSVM | kernel | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| GRAIL-0 | reference | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-hist+iForest (ctrl) | reference | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+LW | reference | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+iForest | reference | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| size-only | reference | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| GLASS | E4 | checkpoint: ORACLE - LAST | 8.95 | 7.88 | 9.97 | 5.45 | 13.26 | 9 |  |
| HC-GLAD | E4 | checkpoint: ORACLE - LAST | 3.40 | 1.84 | 5.11 | 2.01 | 4.82 | 7 |  |
| ProtoGLAD | E4 | checkpoint: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| SUL-GUL | E4 | checkpoint: ORACLE - LAST | 9.95 | 7.44 | 12.58 | 5.62 | 14.57 | 8 |  |
| GLADMamba | E5 | checkpoint: ORACLE - LAST | 3.39 | 2.18 | 4.78 | 1.05 | 6.00 | 4 |  |
| GLADPro | E5 | checkpoint: ORACLE - LAST | 12.22 | 9.77 | 14.59 | 7.06 | 17.56 | 8 |  |
| LGKDE | E5 | checkpoint: ORACLE - LAST | -1.62 | -3.40 | 0.19 | -12.36 | 6.27 | 5 |  |
| MUSE | E5 | checkpoint: ORACLE - LAST | 7.68 | 5.34 | 10.29 | 5.13 | 10.52 | 9 |  |
| SEGO | E5 | checkpoint: ORACLE - LAST | 4.38 | 3.49 | 5.33 | 3.12 | 6.01 | 9 |  |
| CVTGAD | frozen12 | checkpoint: ORACLE - LAST | 5.12 | 4.31 | 6.02 | 2.85 | 7.46 | 7 |  |
| GLADC | frozen12 | checkpoint: ORACLE - LAST | 5.07 | 4.19 | 5.90 | 2.59 | 7.46 | 6 |  |
| GLocalKD | frozen12 | checkpoint: ORACLE - LAST | 3.85 | 2.73 | 5.00 | 1.61 | 6.33 | 5 |  |
| GOOD-D | frozen12 | checkpoint: ORACLE - LAST | 4.55 | 3.34 | 5.80 | 2.74 | 6.38 | 6 |  |
| GraphCL_IF | frozen12 | checkpoint: ORACLE - LAST | 3.99 | 3.06 | 4.95 | 2.81 | 5.37 | 8 |  |
| GraphCL_OCSVM | frozen12 | checkpoint: ORACLE - LAST | 6.37 | 5.37 | 7.40 | 4.71 | 8.20 | 9 |  |
| GraphDE | frozen12 | checkpoint: ORACLE - LAST | 3.38 | 2.34 | 4.43 | 1.43 | 5.49 | 4 |  |
| InfoGraph_IF | frozen12 | checkpoint: ORACLE - LAST | 9.67 | 8.65 | 10.75 | 5.39 | 14.87 | 9 |  |
| InfoGraph_OCSVM | frozen12 | checkpoint: ORACLE - LAST | 6.33 | 5.57 | 7.08 | 2.37 | 11.05 | 4 |  |
| OCGIN | frozen12 | checkpoint: ORACLE - LAST | 5.69 | 4.70 | 6.66 | 3.66 | 7.86 | 8 |  |
| OCGTL | frozen12 | checkpoint: ORACLE - LAST | 3.41 | 2.07 | 4.81 | 1.56 | 5.41 | 5 |  |
| SIGNET | frozen12 | checkpoint: ORACLE - LAST | 4.01 | 2.52 | 5.42 | 0.52 | 7.52 | 4 |  |
| PK-IF | kernel | checkpoint: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| PK-OCSVM | kernel | checkpoint: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-IF | kernel | checkpoint: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-OCSVM | kernel | checkpoint: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| GRAIL-0 | reference | checkpoint: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-hist+iForest (ctrl) | reference | checkpoint: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+LW | reference | checkpoint: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+iForest | reference | checkpoint: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| size-only | reference | checkpoint: ORACLE - LAST | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| GLASS | E4 | checkpoint: ORACLE - VAL-NORMAL | 7.91 | 6.24 | 9.67 | 5.97 | 9.89 | 9 |  |
| HC-GLAD | E4 | checkpoint: ORACLE - VAL-NORMAL | 2.28 | 1.05 | 3.60 | 1.17 | 3.45 | 5 |  |
| ProtoGLAD | E4 | checkpoint: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| SUL-GUL | E4 | checkpoint: ORACLE - VAL-NORMAL | 8.59 | 6.55 | 10.87 | 5.05 | 12.70 | 8 |  |
| GLADMamba | E5 | checkpoint: ORACLE - VAL-NORMAL | 9.45 | 7.96 | 11.11 | 4.81 | 14.97 | 7 |  |
| GLADPro | E5 | checkpoint: ORACLE - VAL-NORMAL | 12.75 | 9.17 | 16.27 | 6.42 | 19.34 | 7 |  |
| LGKDE | E5 | checkpoint: ORACLE - VAL-NORMAL | -1.27 | -2.95 | 0.52 | -12.47 | 6.75 | 6 |  |
| MUSE | E5 | checkpoint: ORACLE - VAL-NORMAL | 6.57 | 4.53 | 8.97 | 3.63 | 9.77 | 6 |  |
| SEGO | E5 | checkpoint: ORACLE - VAL-NORMAL | 8.25 | 6.92 | 9.67 | 5.76 | 10.49 | 8 |  |
| CVTGAD | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 5.02 | 4.20 | 5.91 | 2.79 | 7.41 | 7 |  |
| GLADC | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 4.82 | 3.91 | 5.72 | 2.55 | 7.03 | 6 |  |
| GLocalKD | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 3.84 | 2.71 | 4.98 | 1.61 | 6.32 | 5 |  |
| GOOD-D | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 4.65 | 3.32 | 6.07 | 2.69 | 6.74 | 7 |  |
| GraphCL_IF | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 3.10 | 2.27 | 3.96 | 1.55 | 4.82 | 5 |  |
| GraphCL_OCSVM | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 6.00 | 4.51 | 7.55 | 3.65 | 8.46 | 8 |  |
| GraphDE | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 2.15 | 1.55 | 2.78 | 0.84 | 3.68 | 3 |  |
| InfoGraph_IF | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 5.32 | 4.22 | 6.45 | 2.62 | 8.37 | 6 |  |
| InfoGraph_OCSVM | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 2.96 | 2.05 | 3.89 | 1.03 | 5.29 | 4 |  |
| OCGIN | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 5.34 | 4.38 | 6.31 | 3.25 | 7.54 | 7 |  |
| OCGTL | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 3.21 | 1.86 | 4.62 | 1.29 | 5.34 | 4 |  |
| SIGNET | frozen12 | checkpoint: ORACLE - VAL-NORMAL | 3.15 | 1.81 | 4.45 | -0.19 | 6.70 | 5 |  |
| PK-IF | kernel | checkpoint: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| PK-OCSVM | kernel | checkpoint: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-IF | kernel | checkpoint: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-OCSVM | kernel | checkpoint: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| GRAIL-0 | reference | checkpoint: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| WL-hist+iForest (ctrl) | reference | checkpoint: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+LW | reference | checkpoint: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| graphlet-count+iForest | reference | checkpoint: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |
| size-only | reference | checkpoint: ORACLE - VAL-NORMAL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | A20: no checkpoint axis (0 by construction) |

Pooled over methods (UB9):

| pool | comparison | n_methods | mean_over_methods | fold_ci_lo | fold_ci_hi | ds_ci_lo | ds_ci_hi | excluded |
|---|---|---|---|---|---|---|---|---|
| frozen12 | checkpoint: ORACLE - VAL-NORMAL | 12 | 4.13 | 3.77 | 4.50 | 3.32 | 4.79 |  |
| E5 | checkpoint: ORACLE - VAL-NORMAL | 5 | 7.15 | 6.37 | 7.96 | 5.43 | 8.73 |  |
| E4 | checkpoint: ORACLE - VAL-NORMAL | 3 | 6.26 | 5.13 | 7.51 | 4.74 | 7.99 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| recent(E5+E4) | checkpoint: ORACLE - VAL-NORMAL | 8 | 6.82 | 6.08 | 7.61 | 5.49 | 8.19 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| frozen12+E5 | checkpoint: ORACLE - VAL-NORMAL | 17 | 5.02 | 4.66 | 5.39 | 4.16 | 5.72 |  |
| frozen12+E5+E4 | checkpoint: ORACLE - VAL-NORMAL | 20 | 5.20 | 4.82 | 5.60 | 4.51 | 5.81 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| all | checkpoint: ORACLE - VAL-NORMAL | 20 | 5.20 | 4.82 | 5.60 | 4.51 | 5.81 | GRAIL-0 (A20: no checkpoint axis (0 by construction))|PK-IF (A20: no checkpoint axis (0 by construction))|PK-OCSVM (A20: no checkpoint axis (0 by construction))|ProtoGLAD (A20: no checkpoint axis (0 by construction))|WL-IF (A20: no checkpoint axis (0 by construction))|WL-OCSVM (A20: no checkpoint axis (0 by construction))|WL-hist+iForest (ctrl) (A20: no checkpoint axis (0 by construction))|graphlet-count+LW (A20: no checkpoint axis (0 by construction))|graphlet-count+iForest (A20: no checkpoint axis (0 by construction))|size-only (A20: no checkpoint axis (0 by construction)) |
| frozen12 | checkpoint: ORACLE - LAST | 12 | 5.12 | 4.77 | 5.48 | 3.97 | 6.24 |  |
| E5 | checkpoint: ORACLE - LAST | 5 | 5.21 | 4.54 | 5.90 | 3.73 | 6.54 |  |
| E4 | checkpoint: ORACLE - LAST | 3 | 7.43 | 6.39 | 8.48 | 5.70 | 9.25 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| recent(E5+E4) | checkpoint: ORACLE - LAST | 8 | 6.04 | 5.48 | 6.63 | 5.22 | 6.98 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| frozen12+E5 | checkpoint: ORACLE - LAST | 17 | 5.15 | 4.77 | 5.53 | 4.03 | 6.20 |  |
| frozen12+E5+E4 | checkpoint: ORACLE - LAST | 20 | 5.49 | 5.12 | 5.88 | 4.55 | 6.36 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| all | checkpoint: ORACLE - LAST | 20 | 5.49 | 5.12 | 5.88 | 4.55 | 6.36 | GRAIL-0 (A20: no checkpoint axis (0 by construction))|PK-IF (A20: no checkpoint axis (0 by construction))|PK-OCSVM (A20: no checkpoint axis (0 by construction))|ProtoGLAD (A20: no checkpoint axis (0 by construction))|WL-IF (A20: no checkpoint axis (0 by construction))|WL-OCSVM (A20: no checkpoint axis (0 by construction))|WL-hist+iForest (ctrl) (A20: no checkpoint axis (0 by construction))|graphlet-count+LW (A20: no checkpoint axis (0 by construction))|graphlet-count+iForest (A20: no checkpoint axis (0 by construction))|size-only (A20: no checkpoint axis (0 by construction)) |
| frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 12 | 4.13 | 3.77 | 4.50 | 3.32 | 4.79 |  |
| E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 5 | 9.32 | 8.56 | 10.10 | 7.97 | 10.78 |  |
| E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 3 | 4.39 | 3.11 | 5.74 | 2.38 | 6.25 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| recent(E5+E4) | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 8 | 7.47 | 6.91 | 8.06 | 6.15 | 8.83 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| frozen12+E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 17 | 5.66 | 5.29 | 6.02 | 5.09 | 6.23 |  |
| frozen12+E5+E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 20 | 5.47 | 5.16 | 5.78 | 4.83 | 6.02 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| all | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - VAL-NORMAL | 20 | 5.47 | 5.16 | 5.78 | 4.83 | 6.02 | GRAIL-0 (A20: no checkpoint axis (0 by construction))|PK-IF (A20: no checkpoint axis (0 by construction))|PK-OCSVM (A20: no checkpoint axis (0 by construction))|ProtoGLAD (A20: no checkpoint axis (0 by construction))|WL-IF (A20: no checkpoint axis (0 by construction))|WL-OCSVM (A20: no checkpoint axis (0 by construction))|WL-hist+iForest (ctrl) (A20: no checkpoint axis (0 by construction))|graphlet-count+LW (A20: no checkpoint axis (0 by construction))|graphlet-count+iForest (A20: no checkpoint axis (0 by construction))|size-only (A20: no checkpoint axis (0 by construction)) |
| frozen12 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 12 | 5.12 | 4.77 | 5.48 | 3.97 | 6.24 |  |
| E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 5 | 7.38 | 6.45 | 8.32 | 6.07 | 8.54 |  |
| E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 3 | 5.56 | 4.49 | 6.69 | 4.05 | 7.11 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| recent(E5+E4) | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 8 | 6.69 | 6.06 | 7.37 | 5.75 | 7.61 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| frozen12+E5 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 17 | 5.78 | 5.34 | 6.24 | 4.65 | 6.83 |  |
| frozen12+E5+E4 | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 20 | 5.75 | 5.35 | 6.17 | 4.75 | 6.67 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| all | checkpoint (UB-GOLD-rule ORACLE): ORACLE_ubgold - LAST | 20 | 5.75 | 5.35 | 6.17 | 4.75 | 6.67 | GRAIL-0 (A20: no checkpoint axis (0 by construction))|PK-IF (A20: no checkpoint axis (0 by construction))|PK-OCSVM (A20: no checkpoint axis (0 by construction))|ProtoGLAD (A20: no checkpoint axis (0 by construction))|WL-IF (A20: no checkpoint axis (0 by construction))|WL-OCSVM (A20: no checkpoint axis (0 by construction))|WL-hist+iForest (ctrl) (A20: no checkpoint axis (0 by construction))|graphlet-count+LW (A20: no checkpoint axis (0 by construction))|graphlet-count+iForest (A20: no checkpoint axis (0 by construction))|size-only (A20: no checkpoint axis (0 by construction)) |
| frozen12 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 12 | 4.43 | 4.10 | 4.77 | 3.21 | 5.52 |  |
| E5 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 5 | 7.93 | 6.64 | 9.17 | 6.10 | 9.80 |  |
| E4 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 3 | 5.79 | 4.81 | 6.92 | 4.13 | 7.65 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| recent(E5+E4) | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 8 | 7.12 | 6.11 | 8.11 | 5.78 | 8.62 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| frozen12+E5 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 17 | 5.46 | 4.99 | 5.95 | 4.31 | 6.48 |  |
| frozen12+E5+E4 | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 20 | 5.51 | 5.04 | 5.98 | 4.62 | 6.38 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| all | checkpoint, HP-FREE: ORACLE - VAL-NORMAL | 20 | 5.51 | 5.04 | 5.98 | 4.62 | 6.38 | GRAIL-0 (A20: no checkpoint axis (0 by construction))|PK-IF (A20: no checkpoint axis (0 by construction))|PK-OCSVM (A20: no checkpoint axis (0 by construction))|ProtoGLAD (A20: no checkpoint axis (0 by construction))|WL-IF (A20: no checkpoint axis (0 by construction))|WL-OCSVM (A20: no checkpoint axis (0 by construction))|WL-hist+iForest (ctrl) (A20: no checkpoint axis (0 by construction))|graphlet-count+LW (A20: no checkpoint axis (0 by construction))|graphlet-count+iForest (A20: no checkpoint axis (0 by construction))|size-only (A20: no checkpoint axis (0 by construction)) |
| frozen12 | checkpoint, HP-FREE: ORACLE - LAST | 12 | 5.11 | 4.63 | 5.57 | 3.94 | 6.21 |  |
| E5 | checkpoint, HP-FREE: ORACLE - LAST | 5 | 5.61 | 4.85 | 6.37 | 4.10 | 6.94 |  |
| E4 | checkpoint, HP-FREE: ORACLE - LAST | 3 | 6.49 | 5.55 | 7.47 | 4.65 | 8.46 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| recent(E5+E4) | checkpoint, HP-FREE: ORACLE - LAST | 8 | 5.94 | 5.32 | 6.57 | 5.19 | 6.87 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| frozen12+E5 | checkpoint, HP-FREE: ORACLE - LAST | 17 | 5.26 | 4.81 | 5.69 | 4.15 | 6.31 |  |
| frozen12+E5+E4 | checkpoint, HP-FREE: ORACLE - LAST | 20 | 5.44 | 5.02 | 5.85 | 4.59 | 6.32 | ProtoGLAD (A20: no checkpoint axis (0 by construction)) |
| all | checkpoint, HP-FREE: ORACLE - LAST | 20 | 5.44 | 5.02 | 5.85 | 4.59 | 6.32 | GRAIL-0 (A20: no checkpoint axis (0 by construction))|PK-IF (A20: no checkpoint axis (0 by construction))|PK-OCSVM (A20: no checkpoint axis (0 by construction))|ProtoGLAD (A20: no checkpoint axis (0 by construction))|WL-IF (A20: no checkpoint axis (0 by construction))|WL-OCSVM (A20: no checkpoint axis (0 by construction))|WL-hist+iForest (ctrl) (A20: no checkpoint axis (0 by construction))|graphlet-count+LW (A20: no checkpoint axis (0 by construction))|graphlet-count+iForest (A20: no checkpoint axis (0 by construction))|size-only (A20: no checkpoint axis (0 by construction)) |
| frozen12 | HP leak: VAL-NORMAL - HP-FREE | 12 | 1.53 | 1.16 | 1.90 | 0.64 | 2.42 |  |
| E5 | HP leak: VAL-NORMAL - HP-FREE | 5 | 0.47 | -0.98 | 1.67 | -0.64 | 1.67 |  |
| E4 | HP leak: VAL-NORMAL - HP-FREE | 2 | 1.91 | 0.81 | 2.96 | 0.32 | 3.67 | GLASS (A21: identical ubgold/hpfree configuration (0 by construction))|SUL-GUL (A21: identical ubgold/hpfree configuration (0 by construction)) |
| recent(E5+E4) | HP leak: VAL-NORMAL - HP-FREE | 7 | 0.88 | -0.24 | 1.79 | -0.12 | 1.95 | GLASS (A21: identical ubgold/hpfree configuration (0 by construction))|SUL-GUL (A21: identical ubgold/hpfree configuration (0 by construction)) |
| frozen12+E5 | HP leak: VAL-NORMAL - HP-FREE | 17 | 1.22 | 0.75 | 1.65 | 0.46 | 1.99 |  |
| frozen12+E5+E4 | HP leak: VAL-NORMAL - HP-FREE | 19 | 1.29 | 0.84 | 1.71 | 0.60 | 2.04 | GLASS (A21: identical ubgold/hpfree configuration (0 by construction))|SUL-GUL (A21: identical ubgold/hpfree configuration (0 by construction)) |
| kernel | HP leak: VAL-NORMAL - HP-FREE | 4 | 2.08 | 1.65 | 2.51 | 0.85 | 3.67 |  |
| all | HP leak: VAL-NORMAL - HP-FREE | 23 | 1.43 | 1.04 | 1.78 | 0.78 | 2.10 | GLASS (A21: identical ubgold/hpfree configuration (0 by construction))|GRAIL-0 (a-priori reference: no hyperparameters)|SUL-GUL (A21: identical ubgold/hpfree configuration (0 by construction))|WL-hist+iForest (ctrl) (a-priori reference: no hyperparameters)|graphlet-count+LW (a-priori reference: no hyperparameters)|graphlet-count+iForest (a-priori reference: no hyperparameters)|size-only (a-priori reference: no hyperparameters) |
| frozen12 | HP leak at LAST: LAST - HP-FREE LAST | 12 | 1.22 | 0.67 | 1.71 | 0.04 | 2.58 |  |
| E5 | HP leak at LAST: LAST - HP-FREE LAST | 5 | 0.10 | -0.97 | 1.07 | -1.78 | 1.87 |  |
| E4 | HP leak at LAST: LAST - HP-FREE LAST | 2 | 1.33 | -0.05 | 2.57 | -0.03 | 2.72 | GLASS (A21: identical ubgold/hpfree configuration (0 by construction))|SUL-GUL (A21: identical ubgold/hpfree configuration (0 by construction)) |
| recent(E5+E4) | HP leak at LAST: LAST - HP-FREE LAST | 7 | 0.45 | -0.39 | 1.18 | -0.74 | 1.69 | GLASS (A21: identical ubgold/hpfree configuration (0 by construction))|SUL-GUL (A21: identical ubgold/hpfree configuration (0 by construction)) |
| frozen12+E5 | HP leak at LAST: LAST - HP-FREE LAST | 17 | 0.89 | 0.43 | 1.33 | -0.02 | 2.00 |  |
| frozen12+E5+E4 | HP leak at LAST: LAST - HP-FREE LAST | 19 | 0.94 | 0.51 | 1.33 | 0.09 | 1.93 | GLASS (A21: identical ubgold/hpfree configuration (0 by construction))|SUL-GUL (A21: identical ubgold/hpfree configuration (0 by construction)) |
| kernel | HP leak at LAST: LAST - HP-FREE LAST | 4 | 2.08 | 1.65 | 2.51 | 0.85 | 3.67 |  |
| all | HP leak at LAST: LAST - HP-FREE LAST | 23 | 1.14 | 0.77 | 1.48 | 0.45 | 1.90 | GLASS (A21: identical ubgold/hpfree configuration (0 by construction))|GRAIL-0 (a-priori reference: no hyperparameters)|SUL-GUL (A21: identical ubgold/hpfree configuration (0 by construction))|WL-hist+iForest (ctrl) (a-priori reference: no hyperparameters)|graphlet-count+LW (a-priori reference: no hyperparameters)|graphlet-count+iForest (a-priori reference: no hyperparameters)|size-only (a-priori reference: no hyperparameters) |
| frozen12 | total: ORACLE - HP-FREE | 12 | 5.66 | 5.25 | 6.11 | 4.27 | 6.86 |  |
| E5 | total: ORACLE - HP-FREE | 5 | 7.62 | 6.06 | 9.00 | 6.15 | 8.99 |  |
| E4 | total: ORACLE - HP-FREE | 4 | 5.73 | 4.78 | 6.71 | 4.30 | 7.31 |  |
| recent(E5+E4) | total: ORACLE - HP-FREE | 9 | 6.78 | 5.73 | 7.77 | 5.62 | 7.97 |  |
| frozen12+E5 | total: ORACLE - HP-FREE | 17 | 6.24 | 5.72 | 6.77 | 4.94 | 7.36 |  |
| frozen12+E5+E4 | total: ORACLE - HP-FREE | 21 | 6.14 | 5.64 | 6.64 | 5.13 | 7.13 |  |
| kernel | total: ORACLE - HP-FREE | 4 | 2.08 | 1.65 | 2.51 | 0.85 | 3.67 |  |
| all | total: ORACLE - HP-FREE | 25 | 5.49 | 5.06 | 5.91 | 4.68 | 6.32 | GRAIL-0 (no checkpoint and no HP axis)|WL-hist+iForest (ctrl) (no checkpoint and no HP axis)|graphlet-count+LW (no checkpoint and no HP axis)|graphlet-count+iForest (no checkpoint and no HP axis)|size-only (no checkpoint and no HP axis) |
| frozen12 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 6 | 4.87 | 4.50 | 5.28 | 3.70 | 6.34 |  |
| E5 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 3 | 4.63 | 3.93 | 5.42 | 3.54 | 5.81 |  |
| E4 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 2 | 5.69 | 4.20 | 7.17 | 4.86 | 6.66 |  |
| recent(E5+E4) | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 5 | 5.06 | 4.17 | 5.97 | 4.32 | 5.78 |  |
| frozen12+E5 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 9 | 4.79 | 4.42 | 5.18 | 3.88 | 5.82 |  |
| frozen12+E5+E4 | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 11 | 4.95 | 4.48 | 5.43 | 4.15 | 5.91 |  |
| all | A14 §4 sensitivity: ORACLE - VAL-NORMAL(holdout_obj) | 11 | 4.95 | 4.48 | 5.43 | 4.15 | 5.91 |  |
| frozen12 | early stop: ORACLE - max over run | 12 | -0.37 | -0.48 | -0.28 | -0.55 | -0.21 |  |
| E5 | early stop: ORACLE - max over run | 5 | -4.01 | -4.38 | -3.65 | -5.87 | -2.55 |  |
| E4 | early stop: ORACLE - max over run | 3 | -1.84 | -2.32 | -1.37 | -2.91 | -0.82 |  |
| recent(E5+E4) | early stop: ORACLE - max over run | 8 | -3.19 | -3.43 | -2.95 | -4.62 | -2.11 |  |
| frozen12+E5 | early stop: ORACLE - max over run | 17 | -1.44 | -1.57 | -1.32 | -1.99 | -1.01 |  |
| frozen12+E5+E4 | early stop: ORACLE - max over run | 20 | -1.50 | -1.63 | -1.38 | -2.07 | -1.08 |  |
| all | early stop: ORACLE - max over run | 20 | -1.50 | -1.63 | -1.38 | -2.07 | -1.08 |  |

## 7. Hyperparameter leak (tuned − label-free) with A21 exclusions

| method_label | stratum | hp_axis | ub9_mean_hp_leak | fold_ci_lo | fold_ci_hi | excluded_from_pooled |
|---|---|---|---|---|---|---|
| CVTGAD | frozen12 | tuned vs label-free HP | 4.53 | 2.73 | 6.27 |  |
| GLADC | frozen12 | tuned vs label-free HP | -0.83 | -1.94 | 0.29 |  |
| GLADMamba | E5 | tuned vs label-free HP | 0.43 | -0.94 | 1.83 |  |
| GLADPro | E5 | tuned vs label-free HP | -2.62 | -5.24 | -0.16 |  |
| GLASS | E4 | no HP axis: identical ubgold/hpfree configuration (A21; 'no per-dataset configuration released') | 0.32 | -0.16 | 0.78 | A21: identical ubgold/hpfree configuration (0 by construction) |
| GLocalKD | frozen12 | tuned vs label-free HP | 1.47 | -0.87 | 3.83 |  |
| GOOD-D | frozen12 | tuned vs label-free HP | 0.79 | -0.37 | 1.90 |  |
| GRAIL-0 | reference | no hyperparameters (a-priori reference) | 0.00 | 0.00 | 0.00 | a-priori reference: no hyperparameters |
| GraphCL_IF | frozen12 | tuned vs label-free HP | 0.09 | -0.69 | 0.93 |  |
| GraphCL_OCSVM | frozen12 | tuned vs label-free HP | 0.17 | -1.02 | 1.40 |  |
| GraphDE (paper orientation, A5) | frozen12 | tuned vs label-free HP | -1.10 | -1.89 | -0.32 |  |
| HC-GLAD | E4 | tuned vs label-free HP | -0.05 | -1.75 | 1.72 |  |
| InfoGraph_IF | frozen12 | tuned vs label-free HP | 0.71 | -0.39 | 1.83 |  |
| InfoGraph_OCSVM | frozen12 | tuned vs label-free HP | 0.57 | -0.49 | 1.69 |  |
| LGKDE ('ubgold' = MUTAG.yaml for all) | E5 | tuned vs label-free HP | 2.63 | 0.60 | 4.67 |  |
| MUSE | E5 | tuned vs label-free HP | 4.19 | -0.06 | 7.93 |  |
| OCGIN | frozen12 | tuned vs label-free HP | 0.96 | -0.28 | 2.17 |  |
| OCGTL | frozen12 | tuned vs label-free HP | 8.76 | 7.45 | 9.93 |  |
| PK-IF | kernel | tuned vs label-free HP | 3.60 | 2.49 | 4.78 |  |
| PK-OCSVM | kernel | tuned vs label-free HP | 1.62 | 0.40 | 2.79 |  |
| ProtoGLAD | E4 | tuned vs label-free HP | 3.86 | 2.59 | 5.19 |  |
| SEGO ('ubgold' = released-modal) | E5 | tuned vs label-free HP | -2.26 | -3.47 | -1.12 |  |
| SIGNET | frozen12 | tuned vs label-free HP | 2.29 | 0.63 | 3.98 |  |
| SUL-GUL (GUL only; SUL not released) | E4 | no HP axis: identical ubgold/hpfree configuration (A21; 'no per-dataset configuration released') | 0.00 | 0.00 | 0.00 | A21: identical ubgold/hpfree configuration (0 by construction) |
| WL-IF | kernel | tuned vs label-free HP | 1.86 | 1.33 | 2.35 |  |
| WL-OCSVM | kernel | tuned vs label-free HP | 1.24 | 0.94 | 1.55 |  |
| WL-hist+iForest (ctrl) | reference | no hyperparameters (a-priori reference) | 0.00 | 0.00 | 0.00 | a-priori reference: no hyperparameters |
| graphlet-count+LW | reference | no hyperparameters (a-priori reference) | 0.00 | 0.00 | 0.00 | a-priori reference: no hyperparameters |
| graphlet-count+iForest | reference | no hyperparameters (a-priori reference) | 0.00 | 0.00 | 0.00 | a-priori reference: no hyperparameters |
| size-only | reference | no hyperparameters (a-priori reference) | 0.00 | 0.00 | 0.00 | a-priori reference: no hyperparameters |

Identical-configuration pairs (A23 §3, A24 §2): ubgold − hpfree differences are run-to-run noise, a **lower bound on seed-to-seed variation**; counts computed from the grid at run time:

| method | label | pairs | bit_identical | differing | last_n | last_absdiff_mean | last_absdiff_min | last_absdiff_median | last_absdiff_p90 | last_absdiff_max | valnormal_n | valnormal_absdiff_mean | valnormal_absdiff_min | valnormal_absdiff_median | valnormal_absdiff_p90 | valnormal_absdiff_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLASS | lower bound on seed-to-seed variation (seed, split and initialisation fixed) | 50 | 18 | 32 | 32 | 1.01 | 0.00 | 0.45 | 2.45 | 5.20 | 32 | 1.23 | 0.00 | 0.00 | 4.04 | 8.33 |
| SUL-GUL | lower bound on seed-to-seed variation (seed, split and initialisation fixed) | 50 | 50 | 0 | 0 | – | – | – | – | – | 0 | – | – | – | – | – |

## 8. G2(c) per A9 (typed test)

| generator | anomaly_type | size_variant | graphlet_count_lw | size_only | edge_label_pair_lw | valid | confounded | feasible | in_decision | controls_src |
|---|---|---|---|---|---|---|---|---|---|---|
| GX | T1a | matched | 100.00 | 50.96 | 51.24 | True | False | True | True | eval_typed_controls_msv2.csv |
| GX | T3a | matched | 90.14 | 49.07 | 50.19 | True | False | True | True | eval_typed_controls_msv2.csv |
| GX | T3b | matched | 91.23 | 50.99 | 50.74 | True | False | True | True | eval_typed_controls_msv2.csv |
| MS | MS1 | matched | 95.07 | 48.82 | 52.30 | True | False | True | True | eval_typed_controls_msv2.csv |
| MS | MS1 | changing | 95.53 | 67.41 | 76.55 | False | False | True | False | eval_typed_controls_msv2.csv |
| MS | MS2 | matched | 86.40 | 51.72 | 59.04 | True | False | True | True | eval_typed_controls_msv2.csv |
| MS | MS2 | changing | 88.33 | 72.21 | 81.06 | False | False | True | False | eval_typed_controls_msv2.csv |
| MS | MS3 | matched | 94.23 | 49.34 | 51.16 | True | False | True | True | eval_typed_controls_msv2.csv |
| MS | MS3 | changing | 95.07 | 68.03 | 76.29 | False | False | True | False | eval_typed_controls_msv2.csv |

| method | stratum | generator | anomaly_type | size_variant | n_seeds | valnormal_mean | seed_ci_lo | seed_ci_hi | blind |
|---|---|---|---|---|---|---|---|---|---|
| GLADC | frozen12 | GX | T1a | matched | 5 | 51.09 | 46.98 | 55.20 | True |
| GLADC | frozen12 | GX | T3a | matched | 5 | 50.46 | 48.58 | 52.34 | True |
| GLADC | frozen12 | GX | T3b | matched | 5 | 54.60 | 51.73 | 56.94 | True |
| GLADC | frozen12 | MS | MS1 | matched | 5 | 53.39 | 52.07 | 54.74 | True |
| GLADC | frozen12 | MS | MS1 | changing | 5 | 85.68 | 84.57 | 86.67 | False |
| GLADC | frozen12 | MS | MS2 | matched | 5 | 56.01 | 54.08 | 57.87 | True |
| GLADC | frozen12 | MS | MS2 | changing | 5 | 87.00 | 85.60 | 88.31 | False |
| GLADC | frozen12 | MS | MS3 | matched | 5 | 52.59 | 50.86 | 54.16 | True |
| GLADC | frozen12 | MS | MS3 | changing | 5 | 85.66 | 84.99 | 86.40 | False |
| CVTGAD | frozen12 | GX | T1a | matched | 5 | 69.94 | 66.57 | 73.47 | False |
| CVTGAD | frozen12 | GX | T3a | matched | 5 | 53.20 | 49.39 | 56.75 | True |
| CVTGAD | frozen12 | GX | T3b | matched | 5 | 56.34 | 52.66 | 59.43 | True |
| CVTGAD | frozen12 | MS | MS1 | matched | 5 | 68.87 | 66.88 | 70.86 | False |
| CVTGAD | frozen12 | MS | MS1 | changing | 5 | 85.04 | 83.83 | 86.31 | False |
| CVTGAD | frozen12 | MS | MS2 | matched | 5 | 82.03 | 80.46 | 83.60 | False |
| CVTGAD | frozen12 | MS | MS2 | changing | 5 | 88.46 | 87.67 | 89.03 | False |
| CVTGAD | frozen12 | MS | MS3 | matched | 5 | 76.93 | 75.77 | 78.02 | False |
| CVTGAD | frozen12 | MS | MS3 | changing | 5 | 87.45 | 87.27 | 87.62 | False |
| GOOD-D | frozen12 | GX | T1a | matched | 5 | 92.65 | 91.33 | 93.97 | False |
| GOOD-D | frozen12 | GX | T3a | matched | 5 | 56.96 | 54.03 | 59.35 | True |
| GOOD-D | frozen12 | GX | T3b | matched | 5 | 60.70 | 58.99 | 61.99 | False |
| GOOD-D | frozen12 | MS | MS1 | matched | 5 | 69.34 | 66.09 | 72.52 | False |
| GOOD-D | frozen12 | MS | MS1 | changing | 5 | 85.34 | 84.25 | 86.83 | False |
| GOOD-D | frozen12 | MS | MS2 | matched | 5 | 75.82 | 72.69 | 80.39 | False |
| GOOD-D | frozen12 | MS | MS2 | changing | 5 | 86.60 | 85.08 | 88.43 | False |
| GOOD-D | frozen12 | MS | MS3 | matched | 5 | 69.88 | 68.28 | 71.47 | False |
| GOOD-D | frozen12 | MS | MS3 | changing | 5 | 84.77 | 83.44 | 86.06 | False |
| GraphDE | frozen12 | GX | T1a | matched | 5 | 49.95 | 48.16 | 51.74 | True |
| GraphDE | frozen12 | GX | T3a | matched | 5 | 48.62 | 46.48 | 51.20 | True |
| GraphDE | frozen12 | GX | T3b | matched | 5 | 50.01 | 48.55 | 51.47 | True |
| GraphDE | frozen12 | MS | MS1 | matched | 5 | 51.71 | 49.77 | 53.63 | True |
| GraphDE | frozen12 | MS | MS1 | changing | 5 | 47.98 | 46.28 | 49.98 | True |
| GraphDE | frozen12 | MS | MS2 | matched | 5 | 44.50 | 42.21 | 46.49 | True |
| GraphDE | frozen12 | MS | MS2 | changing | 5 | 43.49 | 40.22 | 46.43 | True |
| GraphDE | frozen12 | MS | MS3 | matched | 5 | 49.98 | 47.96 | 51.62 | True |
| GraphDE | frozen12 | MS | MS3 | changing | 5 | 46.11 | 44.49 | 47.70 | True |
| GraphCL_IF | frozen12 | GX | T1a | matched | 5 | 50.29 | 47.68 | 52.70 | True |
| GraphCL_IF | frozen12 | GX | T3a | matched | 5 | 50.06 | 48.95 | 51.16 | True |
| GraphCL_IF | frozen12 | GX | T3b | matched | 5 | 49.05 | 47.62 | 50.66 | True |
| GraphCL_IF | frozen12 | MS | MS1 | matched | 5 | 52.76 | 51.60 | 54.11 | True |
| GraphCL_IF | frozen12 | MS | MS1 | changing | 5 | 77.89 | 76.57 | 79.14 | False |
| GraphCL_IF | frozen12 | MS | MS2 | matched | 5 | 59.09 | 58.71 | 59.47 | True |
| GraphCL_IF | frozen12 | MS | MS2 | changing | 5 | 80.92 | 79.38 | 82.42 | False |
| GraphCL_IF | frozen12 | MS | MS3 | matched | 5 | 52.59 | 51.77 | 53.62 | True |
| GraphCL_IF | frozen12 | MS | MS3 | changing | 5 | 77.83 | 76.43 | 79.42 | False |
| GraphCL_OCSVM | frozen12 | GX | T1a | matched | 5 | 51.11 | 48.38 | 53.29 | True |
| GraphCL_OCSVM | frozen12 | GX | T3a | matched | 5 | 50.61 | 48.57 | 52.46 | True |
| GraphCL_OCSVM | frozen12 | GX | T3b | matched | 5 | 49.65 | 47.74 | 51.79 | True |
| GraphCL_OCSVM | frozen12 | MS | MS1 | matched | 5 | 51.56 | 50.11 | 53.01 | True |
| GraphCL_OCSVM | frozen12 | MS | MS1 | changing | 5 | 74.76 | 72.15 | 76.75 | False |
| GraphCL_OCSVM | frozen12 | MS | MS2 | matched | 5 | 57.21 | 55.24 | 59.19 | True |
| GraphCL_OCSVM | frozen12 | MS | MS2 | changing | 5 | 78.02 | 75.17 | 80.64 | False |
| GraphCL_OCSVM | frozen12 | MS | MS3 | matched | 5 | 50.62 | 49.42 | 51.99 | True |
| GraphCL_OCSVM | frozen12 | MS | MS3 | changing | 5 | 74.69 | 72.79 | 76.51 | False |
| InfoGraph_IF | frozen12 | GX | T1a | matched | 5 | 47.23 | 46.55 | 47.94 | True |
| InfoGraph_IF | frozen12 | GX | T3a | matched | 5 | 48.54 | 46.70 | 50.43 | True |
| InfoGraph_IF | frozen12 | GX | T3b | matched | 5 | 47.19 | 45.99 | 48.34 | True |
| InfoGraph_IF | frozen12 | MS | MS1 | matched | 5 | 51.04 | 49.05 | 52.86 | True |
| InfoGraph_IF | frozen12 | MS | MS1 | changing | 5 | 86.08 | 83.95 | 88.22 | False |
| InfoGraph_IF | frozen12 | MS | MS2 | matched | 5 | 58.19 | 53.77 | 62.31 | True |
| InfoGraph_IF | frozen12 | MS | MS2 | changing | 5 | 88.06 | 85.48 | 90.65 | False |
| InfoGraph_IF | frozen12 | MS | MS3 | matched | 5 | 51.10 | 49.06 | 52.63 | True |
| InfoGraph_IF | frozen12 | MS | MS3 | changing | 5 | 85.61 | 83.05 | 87.93 | False |
| InfoGraph_OCSVM | frozen12 | GX | T1a | matched | 5 | 49.41 | 48.00 | 51.80 | True |
| InfoGraph_OCSVM | frozen12 | GX | T3a | matched | 5 | 49.60 | 48.24 | 51.04 | True |
| InfoGraph_OCSVM | frozen12 | GX | T3b | matched | 5 | 48.41 | 45.88 | 51.66 | True |
| InfoGraph_OCSVM | frozen12 | MS | MS1 | matched | 5 | 49.85 | 48.45 | 51.17 | True |
| InfoGraph_OCSVM | frozen12 | MS | MS1 | changing | 5 | 84.91 | 83.35 | 85.90 | False |
| InfoGraph_OCSVM | frozen12 | MS | MS2 | matched | 5 | 57.08 | 53.36 | 59.71 | True |
| InfoGraph_OCSVM | frozen12 | MS | MS2 | changing | 5 | 87.30 | 85.26 | 88.51 | False |
| InfoGraph_OCSVM | frozen12 | MS | MS3 | matched | 5 | 50.60 | 48.74 | 51.94 | True |
| InfoGraph_OCSVM | frozen12 | MS | MS3 | changing | 5 | 84.62 | 82.13 | 86.07 | False |
| SIGNET | frozen12 | GX | T1a | matched | 5 | 49.81 | 48.97 | 50.92 | True |
| SIGNET | frozen12 | GX | T3a | matched | 5 | 50.48 | 48.77 | 52.49 | True |
| SIGNET | frozen12 | GX | T3b | matched | 5 | 52.92 | 49.53 | 55.74 | True |
| SIGNET | frozen12 | MS | MS1 | matched | 5 | 54.89 | 53.13 | 57.53 | True |
| SIGNET | frozen12 | MS | MS1 | changing | 5 | 74.99 | 73.95 | 76.14 | False |
| SIGNET | frozen12 | MS | MS2 | matched | 5 | 62.76 | 61.17 | 64.35 | False |
| SIGNET | frozen12 | MS | MS2 | changing | 5 | 70.21 | 66.17 | 74.25 | False |
| SIGNET | frozen12 | MS | MS3 | matched | 5 | 53.35 | 52.47 | 54.81 | True |
| SIGNET | frozen12 | MS | MS3 | changing | 5 | 75.75 | 74.28 | 76.95 | False |
| OCGTL | frozen12 | GX | T1a | matched | 5 | 52.77 | 49.78 | 55.95 | True |
| OCGTL | frozen12 | GX | T3a | matched | 5 | 53.98 | 51.09 | 56.43 | True |
| OCGTL | frozen12 | GX | T3b | matched | 5 | 54.75 | 53.83 | 55.68 | True |
| OCGTL | frozen12 | MS | MS1 | matched | 5 | 60.43 | 59.29 | 61.71 | False |
| OCGTL | frozen12 | MS | MS1 | changing | 5 | 89.65 | 88.23 | 90.95 | False |
| OCGTL | frozen12 | MS | MS2 | matched | 5 | 68.30 | 66.18 | 70.60 | False |
| OCGTL | frozen12 | MS | MS2 | changing | 5 | 91.91 | 90.50 | 93.34 | False |
| OCGTL | frozen12 | MS | MS3 | matched | 5 | 63.03 | 61.37 | 64.74 | False |
| OCGTL | frozen12 | MS | MS3 | changing | 5 | 90.22 | 87.96 | 92.20 | False |
| OCGIN | frozen12 | GX | T1a | matched | 5 | 48.07 | 45.75 | 50.76 | True |
| OCGIN | frozen12 | GX | T3a | matched | 5 | 52.91 | 50.29 | 55.26 | True |
| OCGIN | frozen12 | GX | T3b | matched | 5 | 51.61 | 49.28 | 54.99 | True |
| OCGIN | frozen12 | MS | MS1 | matched | 5 | 50.13 | 48.31 | 51.81 | True |
| OCGIN | frozen12 | MS | MS1 | changing | 5 | 77.67 | 76.41 | 78.92 | False |
| OCGIN | frozen12 | MS | MS2 | matched | 5 | 61.79 | 59.63 | 63.74 | False |
| OCGIN | frozen12 | MS | MS2 | changing | 5 | 82.90 | 81.34 | 84.28 | False |
| OCGIN | frozen12 | MS | MS3 | matched | 5 | 51.05 | 49.86 | 52.38 | True |
| OCGIN | frozen12 | MS | MS3 | changing | 5 | 78.14 | 76.07 | 80.04 | False |
| GLocalKD | frozen12 | GX | T1a | matched | 5 | 50.11 | 48.16 | 52.82 | True |
| GLocalKD | frozen12 | GX | T3a | matched | 5 | 51.54 | 49.34 | 53.75 | True |
| GLocalKD | frozen12 | GX | T3b | matched | 5 | 51.59 | 50.10 | 53.73 | True |
| GLocalKD | frozen12 | MS | MS1 | matched | 5 | 50.34 | 48.44 | 52.20 | True |
| GLocalKD | frozen12 | MS | MS1 | changing | 5 | 83.36 | 80.15 | 86.57 | False |
| GLocalKD | frozen12 | MS | MS2 | matched | 5 | 58.82 | 54.09 | 61.62 | True |
| GLocalKD | frozen12 | MS | MS2 | changing | 5 | 86.97 | 82.96 | 90.89 | False |
| GLocalKD | frozen12 | MS | MS3 | matched | 5 | 50.76 | 48.84 | 52.54 | True |
| GLocalKD | frozen12 | MS | MS3 | changing | 5 | 83.79 | 80.23 | 87.35 | False |

```
{
 "label": "hypothesis generated post hoc; tested out of sample per A9",
 "confirmatory_methods": [
  "GLADC",
  "CVTGAD",
  "GOOD-D",
  "GraphDE",
  "GraphCL_IF",
  "GraphCL_OCSVM",
  "InfoGraph_IF",
  "InfoGraph_OCSVM"
 ],
 "feasibility_reading": "any_seed_below_100_infeasible (A25)",
 "feasibility_note": "A9 \u00a72 as amended by A25: infeasible iff any seed yields < 100 anomalies",
 "decision": {
  "decision_types": [
   "GX:T1a",
   "GX:T3a",
   "GX:T3b",
   "MS:MS1",
   "MS:MS2",
   "MS:MS3"
  ],
  "n_valid_ms_types": 3,
  "undetermined_types": [],
  "n_pairs": 48,
  "n_pairs_missing": 0,
  "n_cells_incomplete": 0,
  "frac_blind_pooled": 0.8125,
  "frac_blind_ms": 0.75,
  "methods_blind_on_majority_of_ms": 6,
  "per_method_majority_ms": {
   "GLADC": true,
   "CVTGAD": false,
   "GOOD-D": false,
   "GraphDE": true,
   "GraphCL_IF": true,
   "GraphCL_OCSVM": true,
   "InfoGraph_IF": true,
   "InfoGraph_OCSVM": true
  },
  "cond_i": true,
  "cond_ii": true,
  "verdict": "CONFIRMED",
  "complete": true
 },
 "verdict": "CONFIRMED",
 "status": "PASS",
 "sensitivity_F1": {
  "decision_types": [
   "GX:T1a",
   "GX:T3a",
   "GX:T3b",
   "MS:MS1",
   "MS:MS2",
   "MS:MS3"
  ],
  "n_valid_ms_types": 3,
  "undetermined_types": [],
  "n_pairs": 48,
  "n_pairs_missing": 0,
  "n_cells_incomplete": 0,
  "frac_blind_pooled": 0.7708333333333334,
  "frac_blind_ms": 0.75,
  "methods_blind_on_majority_of_ms": 6,
  "per_method_majority_ms": {
   "GLADC": true,
   "CVTGAD": false,
   "GOOD-D": false,
   "GraphDE": true,
   "GraphCL_IF": true,
   "GraphCL_OCSVM": true,
   "InfoGraph_IF": true,
   "InfoGraph_OCSVM": true
  },
  "cond_i": true,
  "cond_ii": true,
  "verdict": "CONFIRMED",
  "complete": true
 },
 "discovery_methods_descriptive": {
  "old4 (discovery)": {
   "decision_types": [
    "GX:T1a",
    "GX:T3a",
    "GX:T3b",
    "MS:MS1",
    "MS:MS2",
    "MS:MS3"
   ],
   "n_valid_ms_types": 3,
   "undetermined_types": [],
   "n_pairs": 24,
   "n_pairs_missing": 0,
   "n_cells_incomplete": 0,
   "frac_blind_pooled": 0.7916666666666666,
   "frac_blind_ms": 0.5833333333333334,
   "methods_blind_on_majority_of_ms": 3,
   "per_method_majority_ms": {
    "SIGNET": true,
    "OCGTL": false,
    "OCGIN": true,
    "GLocalKD": true
   },
   "cond_i": true,
   "cond_ii": false,
   "verdict": "PARTIAL",
   "complete": true
  }
 },
 "H4s_scope": {
  "types": [
   "MS1",
   "MS2",
   "MS3"
  ],
  "blind_rate_matched": 0.75,
  "n_matched": 24,
  "blind_rate_changing": 0.125,
  "n_changing": 24,
  "drop_pp": 62.5,
  "scope_to_size_preserving": true,
  "status": "computed"
 },
 "descriptive_4method_m2": {
  "label": "descriptive only (A9 \u00a75): PREREG's post-hoc 4-method computation on M2",
  "per_method": {
   "GLocalKD": {
    "T1a": 50.10527999999999,
    "T3a": 51.53536,
    "T3b": 51.58736
   },
   "OCGIN": {
    "T1a": 48.07487999999999,
    "T3a": 52.90592000000001,
    "T3b": 51.60928
   },
   "OCGTL": {
    "T1a": 52.77088,
    "T3a": 53.979200000000006,
    "T3b": 54.7544
   },
   "SIGNET": {
    "T1a": 49.81088,
    "T3a": 50.478080000000006,
    "T3b": 52.922880000000006
   }
  },
  "frac_all_le60": 1.0
 }
}
```

## 9. Version sensitivity (A17; never part of G1/G2)

PENDING (no a96137b runs yet).

## 10. Agreement between protocol leaderboards (Kendall τ-b)

| pool | protocol_a | protocol_b | n_methods | tau_b | fold_ci_lo | fold_ci_hi | ds_ci_lo | ds_ci_hi | top3_changed |
|---|---|---|---|---|---|---|---|---|---|
| frozen12 | oracle | valnormal | 12 | 0.76 | 0.61 | 0.85 | 0.36 | 0.85 | True |
| E5 | oracle | valnormal | 5 | 0.60 | 0.60 | 0.80 | 0.40 | 1.00 | False |
| E4 | oracle | valnormal | 4 | 1.00 | 0.33 | 1.00 | 0.00 | 1.00 | False |
| recent(E5+E4) | oracle | valnormal | 9 | 0.61 | 0.39 | 0.72 | 0.33 | 0.78 | True |
| frozen12+E5 | oracle | valnormal | 17 | 0.68 | 0.57 | 0.76 | 0.50 | 0.78 | True |
| frozen12+E5+E4 | oracle | valnormal | 21 | 0.65 | 0.52 | 0.72 | 0.49 | 0.73 | True |
| all | oracle | valnormal | 30 | 0.77 | 0.70 | 0.81 | 0.59 | 0.80 | True |
| frozen12 | oracle | last | 12 | 0.70 | 0.58 | 0.79 | 0.42 | 0.85 | True |
| E5 | oracle | last | 5 | 0.80 | 0.60 | 0.80 | 0.40 | 0.80 | False |
| E4 | oracle | last | 4 | 0.67 | 0.00 | 1.00 | 0.00 | 1.00 | False |
| recent(E5+E4) | oracle | last | 9 | 0.67 | 0.56 | 0.83 | 0.44 | 0.78 | False |
| frozen12+E5 | oracle | last | 17 | 0.79 | 0.71 | 0.82 | 0.59 | 0.87 | False |
| frozen12+E5+E4 | oracle | last | 21 | 0.70 | 0.62 | 0.75 | 0.51 | 0.80 | False |
| all | oracle | last | 30 | 0.77 | 0.71 | 0.81 | 0.60 | 0.81 | False |
| frozen12 | valnormal | last | 12 | 0.70 | 0.58 | 0.85 | 0.42 | 0.97 | False |
| E5 | valnormal | last | 5 | 0.80 | 0.80 | 1.00 | 0.40 | 1.00 | False |
| E4 | valnormal | last | 4 | 0.67 | 0.00 | 1.00 | 0.33 | 1.00 | False |
| recent(E5+E4) | valnormal | last | 9 | 0.72 | 0.56 | 0.89 | 0.56 | 0.94 | True |
| frozen12+E5 | valnormal | last | 17 | 0.71 | 0.60 | 0.82 | 0.56 | 0.88 | True |
| frozen12+E5+E4 | valnormal | last | 21 | 0.70 | 0.60 | 0.82 | 0.60 | 0.88 | True |
| all | valnormal | last | 30 | 0.83 | 0.77 | 0.89 | 0.75 | 0.92 | True |
| frozen12 | oracle_ubgold | valnormal | 12 | 0.76 | 0.61 | 0.85 | 0.36 | 0.85 | True |
| E5 | oracle_ubgold | valnormal | 5 | 0.60 | 0.40 | 0.80 | 0.20 | 0.80 | False |
| E4 | oracle_ubgold | valnormal | 4 | 1.00 | 0.33 | 1.00 | 0.00 | 1.00 | False |
| recent(E5+E4) | oracle_ubgold | valnormal | 9 | 0.56 | 0.22 | 0.67 | 0.17 | 0.67 | True |
| frozen12+E5 | oracle_ubgold | valnormal | 17 | 0.69 | 0.57 | 0.76 | 0.46 | 0.76 | True |
| frozen12+E5+E4 | oracle_ubgold | valnormal | 21 | 0.65 | 0.52 | 0.71 | 0.46 | 0.71 | True |
| all | oracle_ubgold | valnormal | 30 | 0.77 | 0.69 | 0.79 | 0.55 | 0.79 | True |
| frozen12 | oracle | hpfree_valnormal | 12 | 0.45 | 0.33 | 0.58 | 0.27 | 0.67 | False |
| E5 | oracle | hpfree_valnormal | 5 | 0.60 | 0.60 | 0.80 | 0.40 | 1.00 | False |
| E4 | oracle | hpfree_valnormal | 4 | 1.00 | 0.67 | 1.00 | 0.33 | 1.00 | False |
| recent(E5+E4) | oracle | hpfree_valnormal | 9 | 0.56 | 0.50 | 0.78 | 0.44 | 0.83 | True |
| frozen12+E5 | oracle | hpfree_valnormal | 17 | 0.57 | 0.51 | 0.69 | 0.43 | 0.69 | True |
| frozen12+E5+E4 | oracle | hpfree_valnormal | 21 | 0.58 | 0.50 | 0.68 | 0.45 | 0.69 | True |
| kernel | oracle | hpfree_valnormal | 4 | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 | False |
| all | oracle | hpfree_valnormal | 30 | 0.69 | 0.65 | 0.76 | 0.53 | 0.75 | True |
| frozen12 | valnormal | hpfree_valnormal | 12 | 0.52 | 0.39 | 0.70 | 0.39 | 0.76 | True |
| E5 | valnormal | hpfree_valnormal | 5 | 1.00 | 0.80 | 1.00 | 0.60 | 1.00 | False |
| E4 | valnormal | hpfree_valnormal | 4 | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 | False |
| recent(E5+E4) | valnormal | hpfree_valnormal | 9 | 0.94 | 0.61 | 1.00 | 0.61 | 0.94 | False |
| frozen12+E5 | valnormal | hpfree_valnormal | 17 | 0.66 | 0.56 | 0.75 | 0.54 | 0.81 | True |
| frozen12+E5+E4 | valnormal | hpfree_valnormal | 21 | 0.70 | 0.57 | 0.75 | 0.57 | 0.81 | True |
| kernel | valnormal | hpfree_valnormal | 4 | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 | False |
| all | valnormal | hpfree_valnormal | 30 | 0.79 | 0.73 | 0.83 | 0.67 | 0.85 | True |
| frozen12 | hpfree_oracle | hpfree_valnormal | 12 | 0.42 | 0.33 | 0.58 | 0.33 | 0.85 | True |
| E5 | hpfree_oracle | hpfree_valnormal | 5 | 0.80 | 0.80 | 1.00 | 0.60 | 1.00 | False |
| E4 | hpfree_oracle | hpfree_valnormal | 4 | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 | False |
| recent(E5+E4) | hpfree_oracle | hpfree_valnormal | 9 | 0.67 | 0.50 | 0.78 | 0.33 | 0.78 | True |
| frozen12+E5 | hpfree_oracle | hpfree_valnormal | 17 | 0.54 | 0.50 | 0.68 | 0.43 | 0.75 | True |
| frozen12+E5+E4 | hpfree_oracle | hpfree_valnormal | 21 | 0.56 | 0.49 | 0.65 | 0.43 | 0.70 | True |
| all | hpfree_oracle | hpfree_valnormal | 30 | 0.70 | 0.66 | 0.74 | 0.57 | 0.77 | True |

## 11. Friedman / Iman–Davenport and Nemenyi CD (UB9)

| pool | protocol | n_methods | friedman_chi2 | friedman_p | iman_davenport_p | nemenyi_CD |
|---|---|---|---|---|---|---|
| frozen12 | oracle | 12 | 34.54 | 3.0e-04 | 4.2e-05 | 5.55 |
| E5 | oracle | 5 | 22.49 | 1.6e-04 | 1.7e-06 | 2.03 |
| E4 | oracle | 4 | 11.27 | 0.01 | 0.0042 | 1.56 |
| recent(E5+E4) | oracle | 9 | 33.16 | 5.8e-05 | 1.9e-06 | 4.00 |
| frozen12+E5 | oracle | 17 | 65.34 | 6.4e-08 | 9.4e-11 | 8.23 |
| frozen12+E5+E4 | oracle | 21 | 76.63 | 1.5e-08 | 1.8e-11 | 10.44 |
| kernel | oracle | 4 | 4.16 | 0.24 | 0.25 | 1.56 |
| reference | oracle | 5 | 6.04 | 0.2 | 0.19 | 2.03 |
| all | oracle | 30 | 134.19 | 1.8e-15 | 4.4e-23 | 15.56 |
| frozen12 | oracle_ubgold | 12 | 34.54 | 3.0e-04 | 4.2e-05 | 5.55 |
| E5 | oracle_ubgold | 5 | 18.84 | 8.4e-04 | 6.6e-05 | 2.03 |
| E4 | oracle_ubgold | 4 | 8.60 | 0.035 | 0.025 | 1.56 |
| recent(E5+E4) | oracle_ubgold | 9 | 26.64 | 8.2e-04 | 1.5e-04 | 4.00 |
| frozen12+E5 | oracle_ubgold | 17 | 57.87 | 1.2e-06 | 1.4e-08 | 8.23 |
| frozen12+E5+E4 | oracle_ubgold | 21 | 69.09 | 2.6e-07 | 2.1e-09 | 10.44 |
| kernel | oracle_ubgold | 4 | 4.16 | 0.24 | 0.25 | 1.56 |
| reference | oracle_ubgold | 5 | 6.04 | 0.2 | 0.19 | 2.03 |
| all | oracle_ubgold | 30 | 129.84 | 1.0e-14 | 1.4e-21 | 15.56 |
| frozen12 | last | 12 | 26.57 | 0.0053 | 0.0024 | 5.55 |
| E5 | last | 5 | 29.51 | 6.2e-06 | 1.8e-11 | 2.03 |
| E4 | last | 4 | 9.93 | 0.019 | 0.011 | 1.56 |
| recent(E5+E4) | last | 9 | 42.22 | 1.2e-06 | 7.6e-10 | 4.00 |
| frozen12+E5 | last | 17 | 63.93 | 1.1e-07 | 2.5e-10 | 8.23 |
| frozen12+E5+E4 | last | 21 | 75.23 | 2.5e-08 | 4.6e-11 | 10.44 |
| kernel | last | 4 | 4.16 | 0.24 | 0.25 | 1.56 |
| reference | last | 5 | 6.04 | 0.2 | 0.19 | 2.03 |
| all | last | 30 | 96.37 | 3.7e-09 | 7.9e-12 | 15.56 |
| frozen12 | valnormal | 12 | 21.65 | 0.027 | 0.019 | 5.55 |
| E5 | valnormal | 5 | 8.44 | 0.077 | 0.066 | 2.03 |
| E4 | valnormal | 4 | 10.33 | 0.016 | 0.0081 | 1.56 |
| recent(E5+E4) | valnormal | 9 | 21.54 | 0.0058 | 0.0025 | 4.00 |
| frozen12+E5 | valnormal | 17 | 50.26 | 2.1e-05 | 1.2e-06 | 8.23 |
| frozen12+E5+E4 | valnormal | 21 | 58.88 | 1.1e-05 | 6.0e-07 | 10.44 |
| kernel | valnormal | 4 | 4.16 | 0.24 | 0.25 | 1.56 |
| reference | valnormal | 5 | 6.04 | 0.2 | 0.19 | 2.03 |
| all | valnormal | 30 | 79.90 | 1.2e-06 | 4.3e-08 | 15.56 |
| frozen12 | hpfree_oracle | 12 | 29.36 | 0.002 | 6.4e-04 | 5.55 |
| E5 | hpfree_oracle | 5 | 17.51 | 0.0015 | 2.1e-04 | 2.03 |
| E4 | hpfree_oracle | 4 | 13.93 | 0.003 | 4.9e-04 | 1.56 |
| recent(E5+E4) | hpfree_oracle | 9 | 28.36 | 4.1e-04 | 5.1e-05 | 4.00 |
| frozen12+E5 | hpfree_oracle | 17 | 48.95 | 3.4e-05 | 2.5e-06 | 8.23 |
| frozen12+E5+E4 | hpfree_oracle | 21 | 63.88 | 1.8e-06 | 4.2e-08 | 10.44 |
| kernel | hpfree_oracle | 4 | 3.34 | 0.34 | 0.36 | 1.56 |
| reference | hpfree_oracle | 5 | 6.04 | 0.2 | 0.19 | 2.03 |
| all | hpfree_oracle | 30 | 112.88 | 7.7e-12 | 3.0e-16 | 15.56 |
| frozen12 | hpfree_oracle_ubgold | 12 | 29.36 | 0.002 | 6.4e-04 | 5.55 |
| E5 | hpfree_oracle_ubgold | 5 | 20.62 | 3.8e-04 | 1.2e-05 | 2.03 |
| E4 | hpfree_oracle_ubgold | 4 | 8.33 | 0.04 | 0.029 | 1.56 |
| recent(E5+E4) | hpfree_oracle_ubgold | 9 | 26.73 | 7.9e-04 | 1.4e-04 | 4.00 |
| frozen12+E5 | hpfree_oracle_ubgold | 17 | 46.84 | 7.2e-05 | 7.6e-06 | 8.23 |
| frozen12+E5+E4 | hpfree_oracle_ubgold | 21 | 57.41 | 1.8e-05 | 1.3e-06 | 10.44 |
| kernel | hpfree_oracle_ubgold | 4 | 3.34 | 0.34 | 0.36 | 1.56 |
| reference | hpfree_oracle_ubgold | 5 | 6.04 | 0.2 | 0.19 | 2.03 |
| all | hpfree_oracle_ubgold | 30 | 112.80 | 7.9e-12 | 3.1e-16 | 15.56 |
| frozen12 | hpfree_last | 12 | 22.74 | 0.019 | 0.012 | 5.55 |
| E5 | hpfree_last | 5 | 29.78 | 5.4e-06 | 9.0e-12 | 2.03 |
| E4 | hpfree_last | 4 | 9.40 | 0.024 | 0.015 | 1.56 |
| recent(E5+E4) | hpfree_last | 9 | 38.46 | 6.2e-06 | 2.6e-08 | 4.00 |
| frozen12+E5 | hpfree_last | 17 | 52.23 | 1.0e-05 | 4.0e-07 | 8.23 |
| frozen12+E5+E4 | hpfree_last | 21 | 62.38 | 3.0e-06 | 9.5e-08 | 10.44 |
| kernel | hpfree_last | 4 | 3.34 | 0.34 | 0.36 | 1.56 |
| reference | hpfree_last | 5 | 6.04 | 0.2 | 0.19 | 2.03 |
| all | hpfree_last | 30 | 81.08 | 8.1e-07 | 2.5e-08 | 15.56 |
| frozen12 | hpfree_valnormal | 12 | 18.64 | 0.068 | 0.056 | 5.55 |
| E5 | hpfree_valnormal | 5 | 15.91 | 0.0031 | 7.1e-04 | 2.03 |
| E4 | hpfree_valnormal | 4 | 8.60 | 0.035 | 0.025 | 1.56 |
| recent(E5+E4) | hpfree_valnormal | 9 | 26.37 | 9.1e-04 | 1.7e-04 | 4.00 |
| frozen12+E5 | hpfree_valnormal | 17 | 44.50 | 1.7e-04 | 2.5e-05 | 8.23 |
| frozen12+E5+E4 | hpfree_valnormal | 21 | 52.62 | 9.2e-05 | 1.3e-05 | 10.44 |
| kernel | hpfree_valnormal | 4 | 3.34 | 0.34 | 0.36 | 1.56 |
| reference | hpfree_valnormal | 5 | 6.04 | 0.2 | 0.19 | 2.03 |
| all | hpfree_valnormal | 30 | 70.46 | 2.6e-05 | 3.1e-06 | 15.56 |

## 12. Label-free selectors of M1 (old 4; descriptive)

| method | criterion | n_datasets | mean_auroc | minus_valnormal | recovered_fraction_of_oracle_gap |
|---|---|---|---|---|---|
| SIGNET | oracle | 9 | 74.61 | 3.33 | 1.00 |
| SIGNET | valnormal | 9 | 71.29 | 0.00 | 0.00 |
| SIGNET | last | 9 | 70.35 | -0.94 | -0.28 |
| SIGNET | pseudo | 9 | 70.19 | -1.10 | -0.33 |
| SIGNET | pseudo_ensemble | 9 | 70.17 | -1.12 | -0.34 |
| SIGNET | ckpt_ensemble | 9 | 70.34 | -0.95 | -0.28 |
| SIGNET | consensus | 9 | 69.83 | -1.46 | -0.44 |
| SIGNET | aucp | 9 | 64.77 | -6.52 | -1.96 |
| SIGNET | oracle_nostop | 9 | 76.77 | 5.49 | 1.65 |
| OCGTL | oracle | 9 | 72.13 | 3.06 | 1.00 |
| OCGTL | valnormal | 9 | 69.06 | 0.00 | 0.00 |
| OCGTL | last | 9 | 68.77 | -0.29 | -0.10 |
| OCGTL | pseudo | 9 | 68.37 | -0.69 | -0.23 |
| OCGTL | pseudo_ensemble | 9 | 68.40 | -0.67 | -0.22 |
| OCGTL | ckpt_ensemble | 9 | 67.43 | -1.63 | -0.53 |
| OCGTL | consensus | 9 | 67.49 | -1.57 | -0.51 |
| OCGTL | aucp | 9 | 68.91 | -0.15 | -0.05 |
| OCGTL | oracle_nostop | 9 | 72.89 | 3.82 | 1.25 |
| OCGIN | oracle | 9 | 70.40 | 5.33 | 1.00 |
| OCGIN | valnormal | 9 | 65.07 | 0.00 | 0.00 |
| OCGIN | last | 9 | 64.72 | -0.34 | -0.06 |
| OCGIN | pseudo | 9 | 65.58 | 0.51 | 0.10 |
| OCGIN | pseudo_ensemble | 9 | 65.78 | 0.72 | 0.13 |
| OCGIN | ckpt_ensemble | 9 | 65.62 | 0.55 | 0.10 |
| OCGIN | consensus | 9 | 65.38 | 0.31 | 0.06 |
| OCGIN | aucp | 9 | 67.18 | 2.11 | 0.40 |
| OCGIN | oracle_nostop | 9 | 70.58 | 5.51 | 1.03 |
| GLocalKD | oracle | 8 | 64.33 | 5.32 | 1.00 |
| GLocalKD | valnormal | 8 | 59.01 | 0.00 | 0.00 |
| GLocalKD | last | 8 | 59.01 | 0.00 | 0.00 |
| GLocalKD | pseudo | 8 | 59.89 | 0.87 | 0.16 |
| GLocalKD | pseudo_ensemble | 8 | 60.07 | 1.06 | 0.20 |
| GLocalKD | ckpt_ensemble | 8 | 59.74 | 0.73 | 0.14 |
| GLocalKD | consensus | 8 | 59.70 | 0.69 | 0.13 |
| GLocalKD | aucp | 8 | 62.50 | 3.49 | 0.66 |
| GLocalKD | oracle_nostop | 8 | 64.35 | 5.33 | 1.00 |
| POOLED(4) | oracle | -1 | 70.37 | 4.26 | – |
| POOLED(4) | valnormal | -1 | 66.11 | 0.00 | – |
| POOLED(4) | last | -1 | 65.72 | -0.39 | – |
| POOLED(4) | pseudo | -1 | 66.01 | -0.10 | – |
| POOLED(4) | pseudo_ensemble | -1 | 66.11 | -0.00 | – |
| POOLED(4) | ckpt_ensemble | -1 | 65.78 | -0.32 | – |
| POOLED(4) | consensus | -1 | 65.60 | -0.51 | – |
| POOLED(4) | aucp | -1 | 65.84 | -0.27 | – |
| POOLED(4) | oracle_nostop | -1 | 71.15 | 5.04 | – |

## 13. KernelGLAD (UB-GOLD's WL/PK + IF/OCSVM)

5-fold means per dataset. Rows without `@testfit`: kernel and detector fitted on the normal training graphs (ranked). `@testfit`: UB-GOLD's own code path, fitted on the unlabelled test set (transductive; never ranked). 'tuned' = best mean TEST AUROC over UB-GOLD's search space; 'default' = UB-GOLD defaults. § = constant kernel matrix (AUROC 50 by construction).

*UB9*:

| method | protocol | AIDS | BZR | COX2 | DHFR | ENZYMES | NCI1 | PROTEINS_full | IMDB-BINARY | REDDIT-BINARY |
|---|---|---|---|---|---|---|---|---|---|---|
| PK-IF | default | 38.48 | 39.79 | 57.82 | 45.90 | 62.06 | 58.23 | 56.14 | 50.00 | 50.00 |
| PK-IF | tuned | 54.16 | 49.00 | 58.44 | 47.07 | 62.19 | 60.21 | 59.74 | 50.00 | 50.00 |
| PK-IF@testfit | default | 28.30 | 41.17 | 46.60 | 43.23 | 59.02 | 56.39 | 32.46 | 50.00 | 50.00 |
| PK-IF@testfit | tuned | 32.61 | 45.50 | 52.56 | 45.11 | 60.64 | 56.39 | 34.14 | 50.00 | 50.00 |
| PK-OCSVM | default | 29.98 | 48.48 | 54.51 | 47.68 | 60.15 | 54.33 | 29.58 | 50.00 | 50.00 |
| PK-OCSVM | tuned | 35.38 | 53.71 | 56.16 | 47.68 | 60.50 | 55.82 | 30.02 | 50.00 | 50.00 |
| PK-OCSVM@testfit | default | 23.46 | 41.07 | 44.81 | 43.29 | 58.90 | 55.34 | 28.42 | 50.00 | 50.00 |
| PK-OCSVM@testfit | tuned | 29.24 | 42.52 | 44.81 | 44.17 | 59.92 | 56.13 | 29.56 | 50.00 | 50.00 |
| WL-IF | default | 87.90 | 34.82 | 54.26 | 39.81 | 60.25 | 48.94 | 56.66 | 55.32 | 73.13 |
| WL-IF | tuned | 91.80 | 37.58 | 55.15 | 41.16 | 61.90 | 50.09 | 60.17 | 56.36 | 73.61 |
| WL-IF@testfit | default | 72.01 | 45.67 | 58.71 | 43.13 | 58.82 | 45.07 | 32.53 | 57.17 | 74.56 |
| WL-IF@testfit | tuned | 78.77 | 46.26 | 60.62 | 44.20 | 60.99 | 46.30 | 34.71 | 58.95 | 75.18 |
| WL-OCSVM | default | 4.21 | 51.14 | 47.03 | 32.73 | 57.56 | 66.30 | 18.95 | 46.96 | 52.89 |
| WL-OCSVM | tuned | 8.39 | 51.70 | 47.99 | 34.49 | 58.83 | 67.20 | 19.49 | 47.52 | 53.33 |
| WL-OCSVM@testfit | default | 3.76 | 41.81 | 41.02 | 32.66 | 57.07 | 66.40 | 19.94 | 44.34 | 52.53 |
| WL-OCSVM@testfit | tuned | 7.41 | 43.39 | 41.39 | 34.19 | 58.62 | 67.35 | 20.08 | 44.34 | 52.74 |

*secondary*:

| method | protocol | DD | COLLAB | Tox21_p53 | Tox21_MMP |
|---|---|---|---|---|---|
| PK-IF | default | 41.27 | 50.00 | 50.00 | 50.00 |
| PK-IF | tuned | 47.68 | 50.00 | 50.00 | 50.00 |
| PK-IF@testfit | default | 30.77 | 50.00 | 50.00 | 50.00 |
| PK-IF@testfit | tuned | 32.44 | 50.00 | 50.00 | 50.00 |
| PK-OCSVM | default | 28.77 | 50.00 | 50.00 | 50.00 |
| PK-OCSVM | tuned | 29.34 | 50.00 | 50.00 | 50.00 |
| PK-OCSVM@testfit | default | 28.12 | 50.00 | 50.00 | 50.00 |
| PK-OCSVM@testfit | tuned | 29.72 | 50.00 | 50.00 | 50.00 |
| WL-IF | default | 63.24 | 54.33 | 51.82 | 48.58 |
| WL-IF | tuned | 65.04 | 57.15 | 55.14 | 53.26 |
| WL-IF@testfit | default | 31.55 | 49.55 | 45.94 | 44.02 |
| WL-IF@testfit | tuned | 32.93 | 52.23 | 46.79 | 44.65 |
| WL-OCSVM | default | 18.18 | 35.34 | 49.71 | 47.83 |
| WL-OCSVM | tuned | 18.59 | 35.77 | 54.90 | 55.51 |
| WL-OCSVM@testfit | default | 18.76 | 34.71 | 46.07 | 39.84 |
| WL-OCSVM@testfit | tuned | 18.93 | 34.95 | 49.01 | 44.62 |

UB-GOLD's verbatim WL-IF pipeline (released label pairing vs correct pairing of the same scores):

| dataset | folds | released | aligned | mismatch | kmax |
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

UB-GOLD's published kernel AUROCs against the chance band of a 5-fold mean: IF: 21/26 cells within |z| <= 3; OCSVM: 24/26 cells within |z| <= 3.

## 14. Output files

- `results/eval_leaderboard_v2_folds.csv`
- `results/eval_leaderboard_v2_extra_rows.csv`
- `results/eval_leaderboard_v2_repro_only.csv`
- `results/eval_leaderboard_v2_leaderboard.csv`
- `results/eval_leaderboard_v2_per_dataset.csv`
- `results/eval_leaderboard_v2_inflation.csv`
- `results/eval_leaderboard_v2_inflation_pooled.csv`
- `results/eval_leaderboard_v2_hpleak.csv`
- `results/eval_leaderboard_v2_identical_config.csv`
- `results/eval_leaderboard_v2_identical_config_summary.csv`
- `results/eval_leaderboard_v2_noise_estimate.csv`
- `results/eval_leaderboard_v2_flags.csv`
- `results/eval_leaderboard_v2_flags_method.csv`
- `results/eval_leaderboard_v2_valnormal_degenerate.csv`
- `results/eval_leaderboard_v2_coverage.csv`
- `results/eval_leaderboard_v2_diverged.csv`
- `results/eval_leaderboard_v2_kendall.csv`
- `results/eval_leaderboard_v2_friedman.csv`
- `results/eval_leaderboard_v2_posthoc.csv`
- `results/eval_leaderboard_v2_cd.csv`
- `results/eval_leaderboard_v2_selectors.csv`
- `results/eval_leaderboard_v2_sensitivity.csv`
- `results/eval_leaderboard_v2_version.csv`
- `results/eval_leaderboard_v2_typed_types.csv`
- `results/eval_leaderboard_v2_typed_cells.csv`
- `results/eval_leaderboard_v2_amendments.csv`
- `results/eval_leaderboard_v2_labels.csv`
- `results/eval_leaderboard_v2_prereg.json`
