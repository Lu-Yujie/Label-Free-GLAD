#!/bin/bash
# Bit-identity of the default run (per-batch gc.collect/empty_cache skipped) vs the verbatim run (EP_VERBATIM_GC=1)
# on CPU: usage tests/gc_equiv.sh DS HP EPOCHS
set -u
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
G=<repo>/evalpaper/ext/HC-GLAD
P=$GLAD_ENVS/ubgold-gpu/bin/python
SM=$GLAD_DATA_ROOT/ext_hcglad/smoke
export EP_PARTS=$SM/parts EP_SCORES=$SM/scores EP_CKPT=$SM/ckpt
DS=$1; HP=$2; E=$3
sfx=$([ "$HP" = hpfree ] && echo _hpfree || echo "")
rm -f $SM/parts/HC-GLAD_${DS}_fold1${sfx}_gcV.json $SM/parts/HC-GLAD_${DS}_fold1${sfx}_gcN.json
EP_VERBATIM_GC=1 $P -B $G/run_method.py --dataset $DS --fold 1 --hp $HP --max_epochs $E --tag gcV --threads 4 > $SM/gcV_${DS}_$HP.log 2>&1
$P -B $G/run_method.py --dataset $DS --fold 1 --hp $HP --max_epochs $E --tag gcN --threads 4 > $SM/gcN_${DS}_$HP.log 2>&1
$P -B $G/tests/compare_runs.py $SM/parts $SM/scores HC-GLAD_${DS}_fold1${sfx}_gcV HC-GLAD_${DS}_fold1${sfx}_gcN
