# Adapters for the nine recent methods

Each folder holds one adapter that runs a method from its official code on UB-GOLD's folds, with the same run
interface and the same per-run record as the harness for the 12 UB-GOLD detectors (`evalpaper/harness/run_eval.py`).

```bash
python -B evalpaper/ext/<method>/run_method.py --method <method> --dataset BZR --fold 0 [--hp hpfree]
python -B evalpaper/ext/<method>/run_pool.py --tasks <task list> ...     # many runs, shared locks, resumable
```

Run records go to `evalpaper/ext/parts/`, per-graph scores and checkpoints to `$GLAD_DATA_ROOT/ext_<method>/`. Each
folder's collector (`collect*.py` or `derive.py`) turns the records into the protocol rows that
`evalpaper/stats/glad_stats.py` reads.

## What every adapter does the same way

- Folds, labels and node features come from UB-GOLD's own loaders (commit c940457) and are checked against
  `data/ubgold_splits/`. Trial `k` uses UB-GOLD's seed `3407 + k`.
- 20% of the normal training graphs are held out (`data/holdout/`), and the method trains on the other 80%.
- At every evaluation point the adapter records the test metrics and the anomaly scores of the held-out normals and
  of every test graph. Test labels are used only to compute metrics.
- Two settings per (dataset, fold). `ubgold` is the method's released per-dataset configuration, which may have been
  tuned on test data. `hpfree` uses the repository defaults; an option whose default crashes or does not fit in memory
  takes a fallback fixed before any test result was seen.
- Protocol rows: ORACLE (the method's own test-label rule where its code has one, otherwise UB-GOLD's rule),
  `oracle_ubgold` (UB-GOLD's rule for every method), LAST, VAL-NORMAL, and the same rows for `hpfree`. HP-FREE in the
  paper is VAL-NORMAL of the `hpfree` runs.
- The upstream code in `src/` is never edited. Changes are runtime overrides or marked copies in the adapter, each
  marked `# CHANGE` in the code.

## Per method

| Method | Venue | Code | `ubgold` setting | Notes |
| --- | --- | --- | --- | --- |
| MUSE | NeurIPS 2024 | fetched | released per-dataset configuration | Early stopping on labelled validation anomalies is removed; snapshots are scored at every evaluation point |
| LGKDE | ICML 2026 | included (MIT) | released configuration | The DGL original is re-implemented in PyTorch Geometric; `equiv/` checks the port against the original on 7 datasets |
| GLADPro | KDD 2025 | included (MIT) | released configuration | `hpfree` uses the argparse defaults (1000 epochs) |
| GLADMamba | ECML-PKDD 2025 | fetched | released configuration | `hpfree` keeps the released per-dataset batch sizes; needs the `ext-gladmamba` overlay |
| SEGO | AAAI 2025 | included (MIT) | most common released configuration | Three PyTorch Geometric 2.0 to 2.6 API fixes are applied at runtime; REDDIT-BINARY `hpfree` uses test batch 128 for memory |
| GLASS | preprint | included (MIT) | repository defaults | `ubgold` and `hpfree` are the same configuration, so their difference measures run-to-run noise; text embeddings come from Qwen3-Embedding-0.6B |
| ProtoGLAD | arXiv 2602.10708 | fetched | replay of its test-label grid search (144 cells); MUTAG uses the README setting | Training-free, so it has no checkpoints; a CPU fallback replaces a GPU-only class |
| HC-GLAD | Neural Networks 2026 | fetched | released configuration | Memory fallbacks for `hpfree`: NCI1 test batch 16; REDDIT-BINARY hidden size 8 with 2 layers; REDDIT-BINARY hyperedges are cached |
| SUL/GUL | WWW 2026 | fetched | repository defaults (the released configuration) | Runs the GUL variant; `ubgold` and `hpfree` coincide |

## Fetched methods

MUSE, GLADMamba, HC-GLAD, ProtoGLAD and SUL/GUL have no licence, so their code is not in this repository. Run
`bash evalpaper/ext/<method>/fetch_src.sh` (or `bash scripts/fetch_third_party.sh` for all five). It clones the
repository at the commit we evaluated into `src/` and rebuilds the adapter files that reuse upstream lines from the
recipes in `rebuild/` (`scripts/rebuild_from_upstream.py`). A recipe holds our own lines and line references into the
upstream files, and the rebuild checks the sha256 of every file it writes. These files exist only after the fetch:

| Method | Rebuilt files |
| --- | --- |
| MUSE | `run_method.py`, `evalpaper/typed/ext/MUSE_typed.py` |
| GLADMamba | `run_method.py`, `evalpaper/typed/ext/GLADMamba_typed.py` |
| HC-GLAD | `run_method.py`, `tests/check_training_only.py`, `evalpaper/typed/ext/HC-GLAD_typed.py` |
| ProtoGLAD | `run_method.py`, `native_repro.py` |
| SUL/GUL | `run_method.py`, `evalpaper/typed/ext/SUL-GUL_typed.py` |
