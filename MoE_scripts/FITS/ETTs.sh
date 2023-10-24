#!/bin/bash

source /jet/home/rni/.bashrc
conda  activate /jet/home/rni/.local/lib/.conda/envs/min-entropy
cd /jet/home/rni/ECE_STORAGE/min-entropy/Fraug-more-results-1785/FrAug

# add for DLinear-I

seq_len=720
model_name=FITS
m=2
log_dir=FITS_all

cut_freq=15

pred_len=$1
t_dim=$2
ett_type=$3


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETT$ett_type.csv \
  --model_id ETT$ett_type'_'$seq_len'_j'$pred_len'_CF'$cut_freq'_T'$t_dim \
  --model $model_name \
  --data ETT$ett_type \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --cut_freq $cut_freq \
  --train_epochs 50\
  --t_dim $t_dim \
  --aug_rate 0\
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/$log_dir/$m'j_'$model_name'_'ETT$ett_type'_'$seq_len'_'$pred_len'_CF'$cut_freq'_T'$t_dim.log

  echo "Done with $m'j_'$model_name'_'ETT$ett_type'_'$seq_len'_'$pred_len'_CF'$cut_freq.log"