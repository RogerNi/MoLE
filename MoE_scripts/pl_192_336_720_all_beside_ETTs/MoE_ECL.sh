#!/bin/bash

set -x
source /jet/home/rni/.bashrc
conda  activate /jet/home/rni/.local/lib/.conda/envs/min-entropy
cd /jet/home/rni/IDL_STORAGE/min-entropy/Fraug-more-results-1785/FrAug

# mutli-tasking
(
# for pred_len in 96 192 336 720
# do
pred_len=$1
    for aug_rate in 0 0.25 0.5 0.75
    do
        for aug_method in f_mask f_mix
        do
            for lr in 0.005 0.01 0.05
            do
                for t_dim in 1 2 3 4 5 6
                do

                # Weather
                python -u run_longExp.py \
                --is_training 1 \
                --root_path ./dataset/ \
                --data_path ECL.csv \
                --model_id ECL_336'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim \
                --model TDLinear \
                --data custom \
                --features M \
                --seq_len 336 \
                --pred_len $pred_len \
                --enc_in 321 \
                --des 'Exp' \
                --itr 1 \
                --batch_size 8  \
                --in_batch_augmentation \
                --aug_method  $aug_method \
                --aug_rate $aug_rate \
                --t_dim $t_dim \
                --learning_rate $lr >logs/MoE_pl_192_336_720/ECL_336'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim.log &

                done
                wait
            done
        done
    done
# done
)
