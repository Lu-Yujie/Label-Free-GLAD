# SIGNET

python benchmark/mymain.py -exp_type ad -DS MUTAG -batch_size 128 -batch_size_test 9999 -num_epoch 1000 -lr 0.0001 -hidden_dim 128 -model SIGNET SIGNE | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f


# WL-IF
python benchmark/mymain.py -exp_type ad -DS MUTAG -batch_size 128 -batch_size_test 1 -eval_freq 1 -num_e
poch 30  -model KernelGLAD KernelGLAD -detector IF -kernel WL -IF_n_trees 200  -IF_sample_ratio 0.5



# WL-OCSVM
python benchmark/mymain.py -exp_type ad -DS MUTAG -batch_size 128 -batch_size_test 1 -eval_freq 1 -num_epoch 30  -model KernelGLAD KernelGLAD -kernel WL -detector OCSVM -nuOCSVM 0.1 | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f


# PK-IF 
python benchmark/mymain.py -exp_type ad -DS MUTAG -batch_size 128 -batch_size_test 1 -eval_freq 1 -num_epoch 30  -model KernelGLAD KernelGLAD -detector IF -kernel PK -IF_n_trees 200  -IF_sample_ratio 0.5 | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f



# PK-OCSVM
python benchmark/mymain.py -exp_type ad -DS MUTAG -batch_size 128 -batch_size_test 1 -eval_freq 1 -num_epoch 30  -model KernelGLAD KernelGLAD -kernel PK -detector OCSVM -nuOCSVM 0.1 | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f



# OCGIN

python benchmark/mymain.py -exp_type ad -DS MUTAG -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.0001 -hidden_dim 32 -num_layer 3 -model OCGIN OCGIN | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f


# GLocalKD
python benchmark/mymain.py -exp_type ad -DS MUTAG -num_epoch 500 -batch_size 128 -batch_size_test 1 -hidden_dim 32 -num_layer 3  -model GLocalKD GLocalKD -output_dim 256 | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f



CUDA_VISIBLE_DEVICES=2 python benchmark/mymain.py -exp_type ad -DS MUTAG -num_epoch 500 -batch_size 128 -batch_size_test 1 -hidden_dim 64 -num_layer 3  -model GLocalKD GLocalKD -output_dim 32 | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f


# OCGTL
python benchmark/mymain.py -exp_type ad -DS MUTAG -batch_size 128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 3 -model OCGTL OCGTL | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f

CUDA_VISIBLE_DEVICES=3 python benchmark/mymain.py -exp_type ad -DS MUTAG -batch_size 128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 3 -model OCGTL OCGTL | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f



# GLADC 

python benchmark/mymain.py -exp_type ad -DS MUTAG -num_epoch 500 -batch_size 128 -batch_size_test 1 -hidden_dim 128 -dropout 0.1  -lr 0.001 -model GLADC GLADC | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f-output_dim 64

CUDA_VISIBLE_DEVICES=1 python benchmark/mymain.py -exp_type ad -DS MUTAG -num_epoch 500 -batch_size 128 -batch_size_test 1 -hidden_dim 128 -dropout 0.1 -lr 0.001 -model GLADC GLADC | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f


CUDA_VISIBLE_DEVICES=1 python benchmark/mymain.py -exp_type ad -DS MUTAG -num_epoch 500 -batch_size 128 -batch_size_test 1 -hidden_dim 32 -dropout 0.1 -lr 0.001 -model GLADC GLADC | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f



# CVTGAD
CUDA_VISIBLE_DEVICES=3 python3 benchmark/mymain.py -exp_type ad -DS MUTAG -batch_size 128 -rw_dim 16 -dg_dim 16 -hidden_dim 32 -num_epoch 500 -num_cluster 3 -alpha 1.0 -num_layer 2  -lr 0.001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f


CUDA_VISIBLE_DEVICES=3 python3 benchmark/mymain.py -exp_type ad -DS MUTAG -batch_size 128 -rw_dim 16 -dg_dim 16 -hidden_dim 32 -num_epoch 500 -num_cluster 3 -alpha 1.0 -num_layer 2  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool | tee >(ts=$(date +"%Y%m%d%H%M%S"); logfile="Logs/${ts}.log"; tee "$logfile") | tail -f






