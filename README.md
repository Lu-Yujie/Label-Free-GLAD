# Label-Free GLAD

This repository accompanies the paper

> **What Remains Without Test Labels? Re-evaluating a Graph-Level Anomaly Detection Benchmark**
> Yujie Lu (yujlu@iu.edu) and Da Yan (yanda@iu.edu), Luddy School of Informatics, Computing, and Engineering,
> Indiana University Bloomington.

UB-GOLD is a widely used benchmark for graph-level anomaly detection (GLAD). Its code picks each detector's checkpoint
and hyperparameters with the test labels. We re-ran the benchmark's detectors and nine more recent methods on
UB-GOLD's own folds while keeping test labels out of every choice, and we measured how much of the published
performance remains. The analysis plan was fixed before the results were seen, as the paper describes.

The repository holds what is needed to check the paper's numbers and to run the evaluation on new detectors.

- The verified fold index lists for every dataset, with the held-out normal graphs of each fold.
- The typed-anomaly suite: the GX generator, the MS-v2 molecule generator with its frozen manifest, and the data.
- The evaluation harness for UB-GOLD's 12 deep detectors and the adapters for the nine recent methods.
- The kernel baselines and the training-free references.
- The frozen statistics code, the result tables (CSV) and the scripts that make the paper's tables and figures.

Per-run records and per-graph scores are too large for git. They are released as archives (see below).

## Protocols in one paragraph

Each detector is trained on the normal training graphs of a fold, minus 20% that we hold out. We then pick the
checkpoint in four ways. ORACLE uses the test labels as UB-GOLD's code does, so it is a diagnostic upper bound only.
LAST takes the final checkpoint. VAL-NORMAL takes the checkpoint whose held-out normal graphs look most normal to the
detector. HP-FREE repeats VAL-NORMAL with the method's default hyperparameters instead of UB-GOLD's tuned ones. The
paper compares these protocols across detectors and datasets, and tests on the typed suite whether a detector responds
to specific structural changes.

## Layout

| Path | Contents |
| --- | --- |
| `data/` | Fold and held-out index lists, flipped-orientation folds, typed-suite data, download and rebuild scripts (`data/README.md`) |
| `results/` | The result tables that the statistics, paper tables and figures read, and the statistics outputs (`results/README.md`) |
| `evalpaper/harness/` | The instrumented UB-GOLD harness (`run_eval.py`), the pool runner (`run_pool.py`), the collector, and UB-GOLD itself (`ubgold_src/`, MIT) |
| `evalpaper/ext/<method>/` | One adapter per recent method: MUSE, LGKDE, GLADPro, GLADMamba, SEGO, GLASS, ProtoGLAD, HC-GLAD, SUL-GUL (`evalpaper/ext/README.md`) |
| `evalpaper/typed/` | The typed-suite generators and data module, and the typed runners (`evalpaper/typed/README.md`) |
| `evalpaper/version_a96137b/` | Version-sensitivity runs of GLocalKD and GLADC with an older UB-GOLD commit (amendment A17) |
| `evalpaper/followups/` | Flipped anomaly-class orientation and the x_s ablation (amendment A33) |
| `evalpaper/kernels/` | UB-GOLD's WL and propagation kernels with Isolation Forest and OCSVM, fitted on normal training graphs |
| `evalpaper/stats/` | Frozen statistics (`glad_stats.py`), post hoc analyses, A33 follow-ups, their tests and reports |
| `evalpaper/paper/` | `tables/make_tables.py` and `figures/make_figures.py` |
| `grail0/`, `common/` | Training-free references (GRAIL-0 and simple controls) and shared data code (fold caches, metrics) |
| `deep_rerun/`, `m1_selection/`, `m2_typed/` | Earlier stages whose outputs the statistics read: the pilot re-run, the first label-free selectors, and the GX generator with the typed pilot |
| `environments/` | Package versions of every Python environment we used |
| `scripts/` | `fetch_third_party.sh`, `rebuild_from_upstream.py`, `portable_run.py` and `localize_records.py` |

## Setup

We used Python 3.12. The statistics, tables and data scripts need only NumPy, SciPy and pandas (environment
`grail-prelim`), and the figures also need Matplotlib (`paper-figs`). Training needs PyTorch 2.5.1 and PyTorch
Geometric 2.6.1 (`ubgold-gpu`) plus small per-method overlays. `environments/` lists the exact versions.

Two environment variables point the code to your files.

```bash
export PYTHONDONTWRITEBYTECODE=1
export GLAD_DATA_ROOT=$PWD/data     # data, caches, checkpoints and per-graph scores
export GLAD_ENVS=$HOME/envs         # only for training: the folder that holds the overlays in environments/
```

Five of the recent methods have no licence, so their code is not included. Run `bash scripts/fetch_third_party.sh`
once before you use their adapters. It clones each repository at the commit we evaluated and rebuilds our adapter
files that reuse upstream lines, checking each file's sha256 (`evalpaper/ext/README.md`).

## Check the paper's results without retraining

These steps read only the released CSVs and finish in seconds.

```bash
python -B evalpaper/paper/tables/make_tables.py     # writes evalpaper/paper/tables/*.tex
python -B evalpaper/paper/figures/make_figures.py   # writes the figures next to the script
python -B evalpaper/stats/a33_followups.py          # A33 follow-ups: results/eval_a33_*.csv and A33_REPORT.md
```

The table script writes one `.tex` file per paper table, each with a header that names the CSV it read and its sha256.

To rebuild the statistics themselves, download the run archives (next section), unpack them and run the frozen
statistics code. It reads the per-run records and the result tables, runs each adapter's collector, and writes the
leaderboards, the pre-registered decisions and `REPORT.md`. It takes a few minutes on four cores and about 32 GB of
memory.

```bash
mkdir -p /tmp/stats_out
python -B scripts/portable_run.py evalpaper/stats/glad_stats.py --final \
    --ms-manifest data/molswap_v2/manifest.json --out /tmp/stats_out --report /tmp/stats_out/REPORT.md
for f in /tmp/stats_out/eval_leaderboard_v2_*; do cmp "$f" "results/$(basename "$f")"; done
```

The post hoc analyses run the same way after one step that points their cache at your copy of the records:
`python -B scripts/localize_records.py`, then `python -B scripts/portable_run.py evalpaper/stats/posthoc_robustness.py
aggregate`. The unit tests of the statistics code run with `python -B -m unittest discover -s evalpaper/stats/tests`.

### Files kept byte for byte

The following files are released exactly as they were used, because their sha256 is fixed in the paper or in the
study's frozen records. Some of them contain absolute data paths of the machine the study ran on.
`scripts/portable_run.py` maps those paths to `$GLAD_DATA_ROOT` while it runs a script, without changing the file.

- `evalpaper/stats/glad_stats.py` (sha256 `2731b02f07519f8bbd97305018fd04ff3771ca3f71e4b1d86b6d7a34d5802c78`), its
  tests, `evalpaper/stats/diverged_registry.csv` and the archived first version and reports in `evalpaper/stats/archive/`;
- `evalpaper/stats/posthoc_robustness.py`;
- `evalpaper/harness/collect.py` and the collectors of the nine adapters (`collect*.py`, `derive.py`);
- the typed-suite files `evalpaper/typed/molswap.py`, `molswap_v2.py`, `typed_data.py`, `controls_typed.py`, the
  validation and control JSON files and `msv2_work/`, and the data in `data/molswap/`, `data/molswap_v2/`,
  `data/m2_data/` and `data/ubgold_splits/`.

The task lists in `evalpaper/harness/jobs/` and `evalpaper/ext/*/` also keep their original identifiers, because
`glad_stats.py` copies their `# DIVERGED` lines into `results/eval_leaderboard_v2_diverged.csv`. In the released copy
of that CSV, job identifiers are replaced by `<jobid>`.

## Run outputs (archives)

| Archive | Unpack into | Contents |
| --- | --- | --- |
| `lfglad_runs_v1.tar.gz` | the repository root | One JSON record per run: seeds, device, library versions, configuration, code hash, pinned third-party commits and the metrics at every evaluation point. Covers the 12 UB-GOLD detectors, the nine recent methods, the typed suite, A17, A33, the kernels, the pilot re-run and the references |
| `lfglad_aux_v1.tar.gz` | `$GLAD_DATA_ROOT` | ProtoGLAD's grid results and the post hoc analysis cache |
| `lfglad_scores_ubgold_v1.tar` | `$GLAD_DATA_ROOT` | Per-graph scores of the 12 UB-GOLD detectors at every evaluation point (main grid and A17) |
| `lfglad_scores_recent_v1.tar` | `$GLAD_DATA_ROOT` | Per-graph scores of the nine recent methods |
| `lfglad_scores_typed_v1.tar` | `$GLAD_DATA_ROOT` | Per-graph scores on the typed suite |
| `lfglad_scores_followups_v1.tar` | `$GLAD_DATA_ROOT` | Per-graph scores of the A33 follow-ups |

Download: `<OSF link, added at release>`. `SHA256SUMS` lists the checksum of each archive. The archives hold no
model checkpoints. Paths inside the records are written as `$GLAD_DATA_ROOT/...`, which `scripts/portable_run.py`
resolves. The 180 pilot runs (four UB-GOLD detectors on the nine main datasets) have trajectories only.

## Re-run the experiments

**Data.** `bash data/download_tu.sh` downloads the TU and Tox21 files and checks them against the copies we used.
`common/build_folds.py` then rebuilds the fold caches (`$GLAD_DATA_ROOT/folds/<DS>.pkl`) and asserts that they match
the released UB-GOLD index lists in `data/ubgold_splits/`. `common/ubgold_xcheck.py` re-exports those lists from
UB-GOLD's own loader if you want to check them from scratch. `data/README.md` has the details.

**One run.** A run is one (method, dataset, fold, hyperparameter setting). For the UB-GOLD detectors:

```bash
python -B evalpaper/harness/run_eval.py --method GLADC --dataset BZR --fold 0            # UB-GOLD's tuned setting
python -B evalpaper/harness/run_eval.py --method GLADC --dataset BZR --fold 0 --hp hpfree
```

Each recent method has the same interface in `evalpaper/ext/<method>/run_method.py`. A run writes its JSON record to
a `parts/` folder next to the runner and its per-graph scores under `$GLAD_DATA_ROOT`. Long runs checkpoint and
resume.

**Many runs.** `run_pool.py` runs a task list inside one allocation. Tasks share lock files, so several pools can work
on one list without running a task twice, and unfinished runs resume in the next pool. The task lists of our grids
are in `evalpaper/harness/jobs/` and `evalpaper/ext/<method>/`, with the scripts that generated them, longest first.

```bash
python -B evalpaper/harness/run_pool.py --tasks evalpaper/harness/jobs/tasks_gpu.txt --workers 24 --gpus 4 \
    --minutes 600 --mem_gb 200
```

**From runs to tables.** `evalpaper/harness/collect.py` and each adapter's collector turn the run records into the
CSVs in `results/`. The typed suite, A17, A33 and the kernels have their own runners and collectors in their folders;
the docstring at the top of each script gives its usage. `grail0/run.py` computes the training-free references.

**A new detector.** Train it on the fit graphs of each fold (`data/holdout/`), score the held-out normal graphs and the
test graphs at every evaluation point, and record the scores in the same JSON format. The protocols and the
statistics then apply unchanged. The adapters for the nine recent methods are worked examples.

## Licence and citation

Our code is under the MIT licence (`LICENSE`). Third-party code keeps its own licence (`THIRD_PARTY_NOTICES.md`). The
datasets belong to their creators. We redistribute only fold indices and data we derived.

If you use this repository, please cite the paper.

```bibtex
@article{lu2026labelfree,
  title   = {What Remains Without Test Labels? Re-evaluating a Graph-Level Anomaly Detection Benchmark},
  author  = {Lu, Yujie and Yan, Da},
  journal = {arXiv preprint arXiv:<id to be added>},
  year    = {2026}
}
```
