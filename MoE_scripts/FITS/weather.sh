#!/bin/bash

source /jet/home/rni/.bashrc
conda  activate /jet/home/rni/.local/lib/.conda/envs/min-entropy
cd /jet/home/rni/ECE_STORAGE/min-entropy/Fraug-more-results-1785/FrAug

# add for DLinear-I

seq_len=720
model_name=FITS
m=2
log_dir=FITS_all

H_order=10

pred_len=$1
t_dim=$2


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id Weather_$seq_len'_j'$pred_len'_H'$H_order'_T'$t_dim \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 144 \
  --train_epochs 50\
  --t_dim $t_dim \
  --aug_rate 0\
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/$log_dir/$m'j_'$model_name'_'Weather_$seq_len'_'$pred_len'_H'$H_order'_T'$t_dim.log

  echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'$pred_len'_H'$H_order.log"
