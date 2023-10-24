#!/bin/bash

set -x
source /jet/home/rni/.bashrc
conda  activate /jet/home/rni/.local/lib/.conda/envs/min-entropy
cd /jet/home/rni/IDL_STORAGE/min-entropy/Fraug-more-results-1785/FrAug

(
for pred_len in 96
do
    for aug_rate in 0
    do
        for aug_method in f_mask
        do
            for lr in 0.05
            do
                for t_dim in 5
                do

                # for batch_size in 4 8 16 32 64 128 256 512
                for batch_size in 640 768 896 1024
                do

                # Weather
                python -u run_longExp.py \
                --is_training 1 \
                --root_path ./dataset/ \
                --data_path ETT$1.csv \
                --model_id ETT$1'_'336'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim \
                --model TDLinear \
                --data ETT$1 \
                --features M \
                --seq_len 336 \
                --pred_len $pred_len \
                --enc_in 7 \
                --des 'Exp' \
                --itr 1 \
                --batch_size $batch_size  \
                --in_batch_augmentation \
                --aug_method  $aug_method \
                --aug_rate $aug_rate \
                --t_dim $t_dim \
                --learning_rate $lr >logs/train_losses/ETT$1'_'336'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim'_'$batch_size.log

                done
                done
            done
        done
    done
done
)
