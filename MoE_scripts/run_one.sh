#!/bin/bash

cd $(dirname $0)/..

(
for pred_len in 96
do
    for aug_rate in 0
    do
        for aug_method in f_mask
        do
            for lr in 0.005
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
                --batch_size 8  \
                --in_batch_augmentation \
                --aug_method  $aug_method \
                --aug_rate $aug_rate \
                --t_dim $t_dim \
                --learning_rate $lr

                done
            done
        done
    done
done
)