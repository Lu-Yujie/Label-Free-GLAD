# Data

This folder is laid out like the data root of the study, so `export GLAD_DATA_ROOT=$PWD/data` lets the code find
everything released here. Downloaded graphs, rebuilt caches, checkpoints and per-graph scores go into the same root.

| Path | Contents |
| --- | --- |
| `ubgold_splits/<DS>.json` | Verified fold index lists for 16 datasets, exported from UB-GOLD's own loader (`common/ubgold_xcheck.py`) and matched graph by graph to an independent rebuild (`common/build_folds.py`). For TU datasets each of the 5 folds has `split_train_index`, `split_test_index`, the normal training graphs `train_idx`, the test graphs `test_idx` and `y_test` (1 = anomaly). Tox21 has one fixed split (`train_gid`, `test_gid`, `y_test`). |
| `holdout/<DS>.json` | The 20% held-out normal training graphs of each fold (`hold_pos`, positions in the loader order, and for TU datasets `hold_graph_idx`). Detectors train on the other 80%. Rebuilt by `make_holdout_index.py` |
| `followups/orientation/folds/` | Flipped-orientation folds of amendment A33.1 for BZR, COX2 and PROTEINS_full (class 1 is the anomaly), with their check file |
| `molswap_v2/` | The MS-v2 typed suite: frozen manifest and the generated graphs for seeds 0 to 4 (`evalpaper/typed/molswap_v2.py`, see `evalpaper/typed/README.md`) |
| `molswap/` | The retired first molecular generator's data (MolSwap v1, amendment A28), kept as a record |
| `m2_data/` | The synthetic GX suite (`GX_s<seed>.npz`) and the other synthetic sets of the typed pilot (`m2_typed/export_data.py`) |
| `raw_zip_sha256.txt` | sha256 of the TU archives we downloaded |
| `download_tu.sh` | Downloads the TU and Tox21 files into `raw/` and `ubgold_run/data/` and checks them |
| `make_holdout_index.py` | Recomputes `holdout/` from `ubgold_splits/` |

Datasets used in the evaluation: AIDS, BZR, COX2, DHFR, ENZYMES, NCI1, PROTEINS_full, IMDB-BINARY and REDDIT-BINARY
(the main set), and DD, COLLAB, Tox21_p53, Tox21_MMP and MUTAG (secondary). Tox21_HSE and Tox21_PPAR-gamma are
included because the paper shows that their released splits train on the class that the test set labels as
anomalous. They are not used in the evaluation.

## Rebuilding the caches

```bash
export GLAD_DATA_ROOT=$PWD/data
bash data/download_tu.sh                                      # raw/ and ubgold_run/data/
python -I -B common/build_folds.py AIDS BZR COX2 DHFR ENZYMES NCI1 PROTEINS_full IMDB-BINARY REDDIT-BINARY \
    DD COLLAB MUTAG Tox21_p53 Tox21_MMP                       # folds/<DS>.pkl, checked against ubgold_splits/
python -B data/make_holdout_index.py                          # holdout/<DS>.json
```

The fold caches are pickles of graph dictionaries (`common/data.py`). They are not released because they are large
and fully determined by the downloads and the index lists.

## Typed suite

The typed data are loaded and checked against their frozen hashes by `evalpaper/typed/typed_data.py`. That file is
frozen, so run code that uses it through `scripts/portable_run.py`, which points its data paths to
`$GLAD_DATA_ROOT`. For example, `python -B scripts/portable_run.py evalpaper/typed/typed_data.py selftest` checks
every typed data file.
