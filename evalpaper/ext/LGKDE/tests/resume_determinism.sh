#!/bin/bash
# Login-node test (CPU, 4 threads, ~1 min per call): determinism and resume equivalence of run_method.py.
#  A: straight run with a fresh negative-structure cache;  B: identical straight run (cache read from disk);
#  C: stop after epoch STOP (checkpoint, rc 3), then resume to the end.  Compares A-B and A-C bitwise.
# Usage: bash tests/resume_determinism.sh DATASET HP [EPOCHS] [STOP]     (HP = ubgold | hpfree)
# Results of 2026-10-13 (MUTAG fold 1, 8 epochs, stop 4): tests/resume_determinism_MUTAG.jsonl (all bitwise).
set -e
DS=${1:-MUTAG}; HP=${2:-ubgold}; E=${3:-8}; STOP=${4:-4}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=$GLAD_ENVS/ubgold-gpu/bin/python
R=$GLAD_DATA_ROOT/ext_lgkde/smoke/restest_${DS}_${HP}
rm -rf $R; mkdir -p $R
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
       EP_PARTS=$R/parts EP_SCORES=$R/scores EP_CKPT=$R/ckpt LGKDE_NEGCACHE=$R/negcache
cd /tmp
run() { $PY -u -B $HERE/run_method.py --method LGKDE --dataset $DS --fold 1 --hp $HP --max_epochs $E --threads 4 "$@"; }
run --tag detA > $R/A.log 2>&1
run --tag detB > $R/B.log 2>&1
set +e; EP_TEST_STOP_EPOCH=$STOP run --tag resC > $R/C1.log 2>&1; echo "first segment rc=$? (3 = checkpointed)"; set -e
run --tag resC > $R/C2.log 2>&1
SFX=$([ "$HP" = hpfree ] && echo "_hpfree" || echo "")
for t in detB resC; do
  $PY -B $HERE/tests/compare_runs.py $R/parts $R/scores LGKDE_${DS}_fold1${SFX}_detA LGKDE_${DS}_fold1${SFX}_$t
done
