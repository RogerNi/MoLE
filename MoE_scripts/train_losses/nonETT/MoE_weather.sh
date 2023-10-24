#!/bin/bash

set -x
source /jet/home/rni/.bashrc
conda  activate /jet/home/rni/.local/lib/.conda/envs/min-entropy
cd /jet/home/rni/IDL_STORAGE/min-entropy/Fraug-more-results-1785/FrAug

# mutli-tasking
(
# for pred_len in 96 192 336 720
# do
pred_len=720
# for batch_size in 4 8 16 32 64 128 256
# do
batch_size=$1
    for aug_rate in 0
    do
        for aug_method in f_mask
        do
            for lr in 0.01 0.05 0.005
            do
                for t_dim in 3
                do

                # Weather
                python -u run_longExp.py \
                --is_training 1 \
                --root_path ./dataset/ \
                --data_path weather.csv \
                --model_id weather_336'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim \
                --model TDLinear \
                --data custom \
                --features M \
                --seq_len 336 \
                --pred_len $pred_len \
                --enc_in 21 \
                --des 'Exp' \
                --itr 1 \
                --batch_size $batch_size  \
                --in_batch_augmentation \
                --aug_method  $aug_method \
                --aug_rate $aug_rate \
                --t_dim $t_dim \
                --learning_rate $lr >logs/train_losses/weather_720_3_$batch_size'_'$lr.log

                done
            done
        done
    done
# done
# done
)
