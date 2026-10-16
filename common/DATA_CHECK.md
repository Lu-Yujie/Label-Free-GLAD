# DATA_CHECK: UB-GOLD-faithful folds for the Grail pilot

Generated 2026-10-12 03:47 by `common/check_data.py`. Machine-readable copy: `common/data_check.json`. Caches: `$GLAD_DATA_ROOT/folds/<DS>.pkl`.

## Protocol as implemented (UB-GOLD `-exp_type ad`)
- Source of truth: the official UB-GOLD repository (github.com/UB-GOLD/UB-GOLD, commit c940457, 2026-06-03), cloned to `$GLAD_DATA_ROOT/UB-GOLD`. Its functions `get_ad_split_TU`, `get_ad_dataset_TU`, `read_graph_file` and `get_ad_dataset_Tox21` were extracted verbatim from `dataloader/data_loader.py` and executed unchanged with PyG 2.6.1 (UB-GOLD's pinned version) + CPU torch (`common/ubgold_xcheck.py`). The exported index lists are what the caches use, and `common/build_folds.py` asserts they equal an independent sklearn re-implementation.
- TU datasets: graph labels remapped to 0..C-1 (sorted raw values, as PyG). `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` over all graphs; UB-GOLD trial i = fold i with seed 3407+i (`meta['seed']`). Train = training-fold graphs with label != 0 (normal only). Test = the entire test fold, y=1 iff label == 0. No anomaly down-sampling and no validation set (UB-GOLD sets val = test; we expose no validation split: label-free calibration must cross-fit inside `fold['train']`).
- Features (`load_folds(name)` default `features='ubgold'`): what UB-GOLD's models see. Labelled TU sets: `node_label` = TU node label (UB-GOLD one-hot, dim = `meta['node_label_dim']`); IMDB-BINARY/REDDIT-BINARY/COLLAB: no node features (UB-GOLD `Constant(1)`); `x` = None everywhere (UB-GOLD uses `TUDataset(use_node_attr=False)`). `features='all'` additionally exposes continuous node attributes / Tox21 atom types; results with it are NOT comparable to published UB-GOLD numbers.
- Tox21 (Type I): fixed TU `_training`/`_testing` files, UB-GOLD's per-file first-appearance label remap, isolated nodes and edge-less graphs dropped (as `nx.from_edgelist` does), constant node features; the same split is returned 5 times (`meta['fixed_split']=True`, seeds 3407..3411).
- Metrics (`common.metrics.evaluate`) mirror UB-GOLD `GAOOD/metric/metric.py`: sklearn AUROC, average precision, FPR at the first ROC point with TPR >= 0.95. Anomaly = positive class; higher score = more anomalous.

## Verification summary
| Dataset | UB-GOLD code export == cache (split + every graph: n, edge set, node-label one-hot) | UB-GOLD num_feat | sklearn 1.0.2 folds == cache | train∩test (indices) | all train graphs normal / y_test consistent |
|---|---|---|---|---|---|
| AIDS | yes (0 mismatches) | 38 | yes | none | yes |
| BZR | yes (0 mismatches) | 53 | yes | none | yes |
| COX2 | yes (0 mismatches) | 35 | yes | none | yes |
| DHFR | yes (0 mismatches) | 53 | yes | none | yes |
| ENZYMES | yes (0 mismatches) | 3 | yes | none | yes |
| NCI1 | yes (0 mismatches) | 37 | yes | none | yes |
| MUTAG | yes (0 mismatches) | 7 | yes | none | yes |
| PROTEINS_full | yes (0 mismatches) | 3 | yes | none | yes |
| IMDB-BINARY | yes (0 mismatches) | 1 | yes | none | yes |
| DD | yes (0 mismatches) | 89 | yes | none | yes |
| COLLAB | yes (0 mismatches) | 1 | yes | none | yes |
| REDDIT-BINARY | yes (0 mismatches) | 1 | yes | none | yes |
| Tox21_p53 | yes (0 mismatches) | 1 | n/a (fixed files) | none | n/a (Tox21: by construction) |
| Tox21_MMP | yes (0 mismatches) | 1 | n/a (fixed files) | none | n/a (Tox21: by construction) |

## Dataset summary vs. UB-GOLD Table 1 (arXiv 2406.15523v2)
Counts are (# normal train, # normal test, # anomalous test). UB-GOLD Table 1 lists one fold; ours = fold 0 and the per-fold range.

| Dataset | #graphs | class counts (raw label: count) | anomaly class (raw) | UB-GOLD features | node_label_dim | cont. attr dim (`features='all'`) | avg n / avg m / max n | UB-GOLD Table 1 | ours fold 0 | ours range over folds | match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AIDS | 2000 | 0: 400, 1: 1600 | 0 (class 0) | node_label_onehot | 38 | 4 | 15.7 / 16.2 / 95 | 1280/320/80 | 1280/320/80 | 1280-1280/320-320/80-80 | exact |
| BZR | 405 | -1: 319, 1: 86 | -1 (class 0) | node_label_onehot | 53 | 3 | 35.8 / 38.4 / 57 | 69/17/64 | 69/17/64 | 68-69/17-18/63-64 | exact |
| COX2 | 467 | -1: 365, 1: 102 | -1 (class 0) | node_label_onehot | 35 | 3 | 41.2 / 43.4 / 56 | 81/21/73 | 81/21/73 | 81-82/20-21/73-73 | exact |
| DHFR | 756 | -1: 295, 1: 461 | -1 (class 0) | node_label_onehot | 53 | 3 | 42.4 / 44.5 / 71 | 368/93/59 | 368/93/59 | 368-369/92-93/59-59 | exact |
| ENZYMES | 600 | 1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100 | 1 (class 0) | node_label_onehot | 3 | 18 | 32.6 / 62.1 / 126 | 400/100/20 | 400/100/20 | 400-400/100-100/20-20 | exact |
| NCI1 | 4110 | 0: 2053, 1: 2057 | 0 (class 0) | node_label_onehot | 37 | - | 29.9 / 32.3 / 111 | 1646/411/411 | 1646/411/411 | 1645-1646/411-412/410-411 | exact |
| MUTAG | 188 | -1: 63, 1: 125 | -1 (class 0) | node_label_onehot | 7 | - | 17.9 / 19.8 / 28 | - | 100/25/13 | 100-100/25-25/12-13 | n/a (not in UB-GOLD) |
| PROTEINS_full | 1113 | 1: 663, 2: 450 | 1 (class 0) | node_label_onehot | 3 | 29 | 39.1 / 72.8 / 620 | 360/90/133 | 360/90/133 | 360-360/90-90/132-133 | exact |
| IMDB-BINARY | 1000 | 0: 500, 1: 500 | 0 (class 0) | constant | None | - | 19.8 / 96.5 / 136 | 400/100/100 | 400/100/100 | 400-400/100-100/100-100 | exact |
| DD | 1178 | 1: 691, 2: 487 | 1 (class 0) | node_label_onehot | 89 | - | 284.3 / 715.7 / 5748 | 390/97/139 | 390/97/139 | 389-390/97-98/138-139 | exact |
| COLLAB | 5000 | 1: 2600, 2: 775, 3: 1625 | 1 (class 0) | constant | None | - | 74.5 / 2457.2 / 492 | 1920/480/520 | 1920/480/520 | 1920-1920/480-480/520-520 | exact |
| REDDIT-BINARY | 2000 | -1: 1000, 1: 1000 | -1 (class 0) | constant | None | - | 429.6 / 497.8 / 3782 | 800/200/200 | 800/200/200 | 800-800/200-200/200-200 | exact |
| Tox21_p53 | 8357 | train file {1: 537, 0: 8097}; test file {0: 241, 1: 28} | test raw 1; train normal raw 0 | constant | 51 | - | 17.4 / 17.9 / 132 | 8088/241/28 | 8088/241/28 | 8088-8088/241-241/28-28 | exact |
| Tox21_MMP | 6408 | train file {1: 1142, 0: 6178}; test file {0: 200, 1: 38} | test raw 1; train normal raw 0 | constant | 51 | - | 16.7 / 17.1 / 109 | 6170/200/38 | 6170/200/38 | 6170-6170/200-200/38-38 | exact |

## Deviations from UB-GOLD and caveats (read before reporting numbers)
1. **Edge format.** UB-GOLD/PyG keep both directions in `edge_index`; we store each undirected edge once (i<j). Equivalent here: every TU set has 0 asymmetric `A.txt` entries; the only self-loops are 5674 `A.txt` entries in COLLAB, which PyG's `read_tu_data` also removes (checked: 0 self-loops and 24,572,158 = 2 x 12,286,079 directed edges in UB-GOLD's PyG COLLAB data, matching our undirected count). Node order is UB-GOLD's (TU file order; for Tox21, first appearance in the edge list).
2. **No validation split (same as UB-GOLD).** UB-GOLD passes the test loader as the validation loader and tunes hyperparameters on test, so its published numbers are test-selected upper bounds. Our folds expose only `train` (normal) and `test`; any calibration, model selection or score orientation must use training normals only (e.g. cross-fitting inside `fold['train']`).
3. **PROTEINS = `PROTEINS_full`.** UB-GOLD's run scripts (`benchmark/Source/*.sh`) use `PROTEINS_full` (same 1113 graphs and labels as PROTEINS; under UB-GOLD features both give the same 3-d node-label one-hot). Its 29-d continuous attributes are only exposed with `features='all'`. Aliases `PROTEINS`, `IMDB-B`, `REDDIT-B` are accepted by `load_folds`.
4. **MUTAG is not a UB-GOLD dataset.** It is in the LGKDE/GLASS 12-TU set. We ran UB-GOLD's own `get_ad_dataset_TU` on it unchanged: anomaly = class 0 = raw label -1 (non-mutagenic, 63 graphs), normal = raw 1 (mutagenic, 125). We did not verify that LGKDE/GLASS used the same direction for MUTAG.
5. **The normal class is often the minority (UB-GOLD code ≠ its own text).** Anomaly = class 0 everywhere. That is the MAJORITY class for BZR, COX2, PROTEINS_full, DD and COLLAB, contrary to UB-GOLD App. B ("minority class is treated as anomalies"). ENZYMES and COLLAB have several normal classes (5 and 2). This is what UB-GOLD's published numbers use, so we keep it.
6. **Tox21: HSE and PPAR-gamma are inverted in UB-GOLD and not cached.** UB-GOLD remaps labels by first appearance per file and trains on mapped label 1. For p53 and MMP the training file starts with an active (raw 1), so training = inactives (raw 0) and test anomalies = actives (raw 1), which is consistent. For HSE and PPAR-gamma the training file starts with an inactive, so UB-GOLD trains on the ACTIVE molecules (423 and 219 graphs, exactly its Table 1 counts) while still scoring actives as test anomalies. Those two tasks are internally inconsistent, so we only cross-checked them (`ubgold_splits/Tox21_HSE.json`, `Tox21_PPAR-gamma.json`) and did not add them to `list_datasets()`. Tox21 features in UB-GOLD are constant; atom types are only exposed with `features='all'`. Tox21 test sets are tiny (28 and 38 anomalies), and the split is fixed, so the std over the 5 'folds' only reflects method randomness.
7. **Variance vs std.** UB-GOLD's CSVs (and probably its Table 8 '±') are `statistics.variance` values ×100. `common.metrics.summarize` reports the std (`np.std`, ddof=0).
8. **Library versions.** UB-GOLD pinned scikit-learn 1.0.2 and PyG 2.6.1; we ran PyG 2.6.1 with torch 2.14.1+cpu and scikit-learn 1.9.1. We also ran the 1.0.2 `StratifiedKFold._make_test_folds` source on every TU label vector (`common/skl_version_check.py`). It reproduces the cached folds exactly (verification table above).
9. **Isomorphic duplicates.** Some TU sets (notably IMDB-BINARY) contain many WL-identical graphs, including across the normal/anomaly classes (see `wl-dup` and 'WL-identical to a graph of the other class' below). This is a property of the data that UB-GOLD keeps, not leakage. No graph index appears in both train and test of any fold.
10. **Graph size alone is strongly informative, and its direction varies.** See the `size AUROC` column (score = n): the anomaly class is much larger on AIDS and smaller on MUTAG and NCI1. A fixed one-sided size-like score therefore flips on some datasets. Channels should be two-sided or calibrated on training normals only.

## Per-fold details
`wl-dup` = test graphs whose WL(3) fingerprint (node labels included when UB-GOLD uses them) equals that of some training graph, split by test normal / test anomaly. Duplicates are a property of the raw TU data (isomorphic copies), not index leakage; anomalous test graphs that are WL-identical to a normal training graph cap the attainable AUROC of any WL-bounded detector. `size AUROC` = AUROC of the raw node count n as the score (a data diagnostic, not a baseline).

### AIDS
- graphs WL-identical to a graph of the other class (whole dataset): 0
- graphs with no edges: 0; graphs with isolated nodes: 163
- self-loop entries removed: 0; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 1280 | 320 | 80 | 0.200 | 18/0 | 0.999 |
| 1 | 3408 | 1280 | 320 | 80 | 0.200 | 22/0 | 1.000 |
| 2 | 3409 | 1280 | 320 | 80 | 0.200 | 23/0 | 0.996 |
| 3 | 3410 | 1280 | 320 | 80 | 0.200 | 21/0 | 0.992 |
| 4 | 3411 | 1280 | 320 | 80 | 0.200 | 27/0 | 1.000 |

### BZR
- graphs WL-identical to a graph of the other class (whole dataset): 0
- graphs with no edges: 0; graphs with isolated nodes: 0
- self-loop entries removed: 0; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 69 | 17 | 64 | 0.790 | 0/0 | 0.722 |
| 1 | 3408 | 69 | 17 | 64 | 0.790 | 0/0 | 0.635 |
| 2 | 3409 | 69 | 17 | 64 | 0.790 | 0/0 | 0.593 |
| 3 | 3410 | 69 | 17 | 64 | 0.790 | 0/0 | 0.620 |
| 4 | 3411 | 68 | 18 | 63 | 0.778 | 0/0 | 0.497 |

### COX2
- graphs WL-identical to a graph of the other class (whole dataset): 0
- graphs with no edges: 0; graphs with isolated nodes: 0
- self-loop entries removed: 0; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 81 | 21 | 73 | 0.777 | 0/0 | 0.510 |
| 1 | 3408 | 81 | 21 | 73 | 0.777 | 0/0 | 0.576 |
| 2 | 3409 | 82 | 20 | 73 | 0.785 | 0/0 | 0.655 |
| 3 | 3410 | 82 | 20 | 73 | 0.785 | 0/0 | 0.549 |
| 4 | 3411 | 82 | 20 | 73 | 0.785 | 0/0 | 0.485 |

### DHFR
- graphs WL-identical to a graph of the other class (whole dataset): 2
- graphs with no edges: 0; graphs with isolated nodes: 0
- self-loop entries removed: 0; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 368 | 93 | 59 | 0.388 | 5/0 | 0.620 |
| 1 | 3408 | 369 | 92 | 59 | 0.391 | 2/1 | 0.614 |
| 2 | 3409 | 369 | 92 | 59 | 0.391 | 3/0 | 0.624 |
| 3 | 3410 | 369 | 92 | 59 | 0.391 | 2/0 | 0.594 |
| 4 | 3411 | 369 | 92 | 59 | 0.391 | 8/0 | 0.543 |

### ENZYMES
- graphs WL-identical to a graph of the other class (whole dataset): 0
- graphs with no edges: 0; graphs with isolated nodes: 8
- self-loop entries removed: 0; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 400 | 100 | 20 | 0.167 | 1/0 | 0.395 |
| 1 | 3408 | 400 | 100 | 20 | 0.167 | 1/0 | 0.622 |
| 2 | 3409 | 400 | 100 | 20 | 0.167 | 2/0 | 0.490 |
| 3 | 3410 | 400 | 100 | 20 | 0.167 | 1/0 | 0.553 |
| 4 | 3411 | 400 | 100 | 20 | 0.167 | 3/0 | 0.629 |

### NCI1
- graphs WL-identical to a graph of the other class (whole dataset): 15
- graphs with no edges: 0; graphs with isolated nodes: 399
- self-loop entries removed: 0; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 1646 | 411 | 411 | 0.500 | 21/1 | 0.323 |
| 1 | 3408 | 1646 | 411 | 411 | 0.500 | 18/2 | 0.295 |
| 2 | 3409 | 1646 | 411 | 411 | 0.500 | 19/0 | 0.315 |
| 3 | 3410 | 1645 | 412 | 410 | 0.499 | 13/1 | 0.349 |
| 4 | 3411 | 1645 | 412 | 410 | 0.499 | 20/1 | 0.298 |

### MUTAG
- graphs WL-identical to a graph of the other class (whole dataset): 0
- graphs with no edges: 0; graphs with isolated nodes: 0
- self-loop entries removed: 0; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 100 | 25 | 13 | 0.342 | 6/0 | 0.103 |
| 1 | 3408 | 100 | 25 | 13 | 0.342 | 4/0 | 0.178 |
| 2 | 3409 | 100 | 25 | 13 | 0.342 | 4/0 | 0.106 |
| 3 | 3410 | 100 | 25 | 12 | 0.324 | 4/0 | 0.112 |
| 4 | 3411 | 100 | 25 | 12 | 0.324 | 4/0 | 0.080 |

### PROTEINS_full
- graphs WL-identical to a graph of the other class (whole dataset): 15
- graphs with no edges: 0; graphs with isolated nodes: 5
- self-loop entries removed: 0; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 360 | 90 | 133 | 0.596 | 14/1 | 0.765 |
| 1 | 3408 | 360 | 90 | 133 | 0.596 | 6/0 | 0.731 |
| 2 | 3409 | 360 | 90 | 133 | 0.596 | 9/1 | 0.764 |
| 3 | 3410 | 360 | 90 | 132 | 0.595 | 15/0 | 0.773 |
| 4 | 3411 | 360 | 90 | 132 | 0.595 | 12/0 | 0.773 |

### IMDB-BINARY
- graphs WL-identical to a graph of the other class (whole dataset): 318
- graphs with no edges: 0; graphs with isolated nodes: 0
- self-loop entries removed: 0; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 400 | 100 | 100 | 0.500 | 57/33 | 0.436 |
| 1 | 3408 | 400 | 100 | 100 | 0.500 | 56/22 | 0.550 |
| 2 | 3409 | 400 | 100 | 100 | 0.500 | 61/38 | 0.519 |
| 3 | 3410 | 400 | 100 | 100 | 0.500 | 57/28 | 0.497 |
| 4 | 3411 | 400 | 100 | 100 | 0.500 | 53/31 | 0.493 |

### DD
- graphs WL-identical to a graph of the other class (whole dataset): 0
- graphs with no edges: 0; graphs with isolated nodes: 0
- self-loop entries removed: 0; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 390 | 97 | 139 | 0.589 | 0/0 | 0.774 |
| 1 | 3408 | 389 | 98 | 138 | 0.585 | 0/0 | 0.803 |
| 2 | 3409 | 389 | 98 | 138 | 0.585 | 0/0 | 0.803 |
| 3 | 3410 | 390 | 97 | 138 | 0.587 | 0/0 | 0.816 |
| 4 | 3411 | 390 | 97 | 138 | 0.587 | 0/0 | 0.829 |

### COLLAB
- graphs WL-identical to a graph of the other class (whole dataset): 369
- graphs with no edges: 0; graphs with isolated nodes: 0
- self-loop entries removed: 5674; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 1920 | 480 | 520 | 0.520 | 135/20 | 0.474 |
| 1 | 3408 | 1920 | 480 | 520 | 0.520 | 145/19 | 0.492 |
| 2 | 3409 | 1920 | 480 | 520 | 0.520 | 134/24 | 0.481 |
| 3 | 3410 | 1920 | 480 | 520 | 0.520 | 137/25 | 0.490 |
| 4 | 3411 | 1920 | 480 | 520 | 0.520 | 127/33 | 0.465 |

### REDDIT-BINARY
- graphs WL-identical to a graph of the other class (whole dataset): 0
- graphs with no edges: 0; graphs with isolated nodes: 21
- self-loop entries removed: 0; directed A.txt entries without reverse: 0

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 800 | 200 | 200 | 0.500 | 1/0 | 0.795 |
| 1 | 3408 | 800 | 200 | 200 | 0.500 | 0/0 | 0.771 |
| 2 | 3409 | 800 | 200 | 200 | 0.500 | 1/0 | 0.821 |
| 3 | 3410 | 800 | 200 | 200 | 0.500 | 1/0 | 0.770 |
| 4 | 3411 | 800 | 200 | 200 | 0.500 | 1/0 | 0.750 |

### Tox21_p53
- graphs with no edges: 0; graphs with isolated nodes: 0
- training file: 8634 graphs, dropped (no edges) 11, isolated nodes dropped 1506, first-appearance label order [1, 0]
- testing file: 269 graphs, dropped (no edges) 0, isolated nodes dropped 119, first-appearance label order [0, 1]

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 8088 | 241 | 28 | 0.104 | 44/2 | 0.643 |
| 1 | 3408 | 8088 | 241 | 28 | 0.104 | 44/2 | 0.643 |
| 2 | 3409 | 8088 | 241 | 28 | 0.104 | 44/2 | 0.643 |
| 3 | 3410 | 8088 | 241 | 28 | 0.104 | 44/2 | 0.643 |
| 4 | 3411 | 8088 | 241 | 28 | 0.104 | 44/2 | 0.643 |

### Tox21_MMP
- graphs with no edges: 0; graphs with isolated nodes: 0
- training file: 7320 graphs, dropped (no edges) 9, isolated nodes dropped 1242, first-appearance label order [1, 0]
- testing file: 238 graphs, dropped (no edges) 0, isolated nodes dropped 107, first-appearance label order [0, 1]

| fold | seed | normal train | test normal | test anomalous | test anomaly ratio | wl-dup (normal/anom) | size AUROC |
|---|---|---|---|---|---|---|---|
| 0 | 3407 | 6170 | 200 | 38 | 0.160 | 37/1 | 0.676 |
| 1 | 3408 | 6170 | 200 | 38 | 0.160 | 37/1 | 0.676 |
| 2 | 3409 | 6170 | 200 | 38 | 0.160 | 37/1 | 0.676 |
| 3 | 3410 | 6170 | 200 | 38 | 0.160 | 37/1 | 0.676 |
| 4 | 3411 | 6170 | 200 | 38 | 0.160 | 37/1 | 0.676 |

