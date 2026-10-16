#!/bin/bash
# Login-node resume-equivalence tests (4 threads, CPU, smoke dirs, fixed random text embeddings).
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SM=$GLAD_DATA_ROOT/ext_glass/smoke
export EP_GLASS_EMB=$SM/embedding_cache_rand EP_PARTS=$SM/parts EP_SCORES=$SM/scores EP_CKPT=$SM/ckpt
P=$GLAD_ENVS/ubgold-gpu/bin/python
G=<repo>/evalpaper/ext/GLASS
cd /tmp
run() { $P -B $G/run_method.py --dataset $1 --fold $2 --max_epochs $3 --tag $4 --threads 4 2>&1 | grep -E "done|deadline|resume|Early|Error|error" ; }
# MUTAG fold 1, 50 epochs: replayed early stop at epoch 40 (own final evaluation), resumed after it (45) and before it (30)
run MUTAG 1 50 rsStraight
EP_TEST_STOP_EPOCH=45 run MUTAG 1 50 rsStop45; run MUTAG 1 50 rsStop45
EP_TEST_STOP_EPOCH=30 run MUTAG 1 50 rsStop30; run MUTAG 1 50 rsStop30
# ENZYMES fold 0, 30 epochs (minibatches, negatives, prototype init at 20); stop at 22
run ENZYMES 0 30 rsStraight
EP_TEST_STOP_EPOCH=22 run ENZYMES 0 30 rsStop22; run ENZYMES 0 30 rsStop22
for p in "MUTAG_fold1_rsStraight MUTAG_fold1_rsStop45" "MUTAG_fold1_rsStraight MUTAG_fold1_rsStop30" "ENZYMES_fold0_rsStraight ENZYMES_fold0_rsStop22"; do
  set -- $p
  $P -B $G/verify/compare.py GLASS_$1 GLASS_$2 --out $G/verify/resume_${2}.json | grep -E '"(traj_max_abs_diff|scores_max_abs_diff|train_loss_max_abs_diff|own_equal)"'
done
$P -B $G/verify/compare.py GLASS_ENZYMES_fold0_detA GLASS_ENZYMES_fold0_detB --out $G/verify/determinism_ENZYMES_fold0.json > /dev/null
$P -B $G/verify/compare.py GLASS_ENZYMES_fold0_nd_detA GLASS_ENZYMES_fold0_nd_detB --out $G/verify/nondeterminism_ENZYMES_fold0_nodetalgo.json > /dev/null
$P -B $G/verify/compare.py GLASS_MUTAG_fold1_nd_detA GLASS_MUTAG_fold1_nd_detB --out $G/verify/determinism_MUTAG_fold1_nodetalgo.json > /dev/null
echo ALLDONE
