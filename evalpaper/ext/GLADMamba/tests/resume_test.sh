#!/bin/bash
# Resume-equivalence + determinism on CPU: straight run (A), stop at epoch 6 + resume (B), straight again (C).
# usage: tests/resume_test.sh DS HP EPOCHS THREADS
set -u
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
G=<repo>/evalpaper/ext/GLADMamba
P=$GLAD_ENVS/ubgold-gpu/bin/python
SM=$GLAD_DATA_ROOT/ext_gladmamba/smoke
export EP_PARTS=$SM/parts EP_SCORES=$SM/scores EP_CKPT=$SM/ckpt
DS=$1; HP=$2; E=$3; TH=${4:-4}
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH
sfx=$([ "$HP" = hpfree ] && echo _hpfree || echo "")
for t in rtA rtB rtC; do rm -f $SM/parts/GLADMamba_${DS}_fold1${sfx}_$t.json; done
$P -B $G/run_method.py --dataset $DS --fold 1 --hp $HP --max_epochs $E --tag rtA --threads $TH > $SM/rtA_${DS}_$HP.log 2>&1
EP_TEST_STOP_EPOCH=6 $P -B $G/run_method.py --dataset $DS --fold 1 --hp $HP --max_epochs $E --tag rtB --threads $TH > $SM/rtB_${DS}_$HP.log 2>&1; echo "rtB first segment rc=$?"
$P -B $G/run_method.py --dataset $DS --fold 1 --hp $HP --max_epochs $E --tag rtB --threads $TH >> $SM/rtB_${DS}_$HP.log 2>&1; echo "rtB resume rc=$?"
$P -B $G/run_method.py --dataset $DS --fold 1 --hp $HP --max_epochs $E --tag rtC --threads $TH > $SM/rtC_${DS}_$HP.log 2>&1
$P -B $G/tests/compare_runs.py $SM/parts $SM/scores GLADMamba_${DS}_fold1${sfx}_rtA GLADMamba_${DS}_fold1${sfx}_rtB
$P -B $G/tests/compare_runs.py $SM/parts $SM/scores GLADMamba_${DS}_fold1${sfx}_rtA GLADMamba_${DS}_fold1${sfx}_rtC
