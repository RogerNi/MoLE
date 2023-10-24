#!/bin/bash

source /jet/home/rni/.bashrc
conda  activate /jet/home/rni/.local/lib/.conda/envs/min-entropy
cd /jet/home/rni/ECE_STORAGE/min-entropy/Fraug-more-results-1785/FrAug

# mutli-tasking
(
pred_len=$1
lr=$2
t_dim=1
seed=$3
# for pred_len in 96 192 336 720
# do
    # for aug_rate in 0 0.25 0.5 0.75
    for aug_rate in 0
    do
        # for aug_method in f_mask f_mix
        for aug_method in f_mask
        do
            # for lr in 0.005 0.01 0.05
            # do
            #     for t_dim in 1 2 3 4 5 6
            #     do

                # Weather
                python -u run_longExp.py \
                --is_training 1 \
                --root_path ./dataset/ \
                --data_path weather.csv \
                --model_id weather_336'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim'_'$seed \
                --model TDLinear \
                --data custom \
                --features M \
                --seq_len 336 \
                --pred_len $pred_len \
                --enc_in 21 \
                --des 'Exp' \
                --itr 1 \
                --batch_size 8  \
                --in_batch_augmentation \
                --aug_method  $aug_method \
                --aug_rate $aug_rate \
                --t_dim $t_dim \
                --seed $seed \
                --learning_rate $lr >logs/ablations/ensemble/DLinear/weather_336'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim'_'$seed.log
                # done
            # done
        done
    done
# done
)
