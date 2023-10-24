#!/bin/bash

cd $(dirname $0)/../../../..


if [[ -z "${SEQ_LEN}" ]]; then
  MY_SEQ_LEN="336"
else
  MY_SEQ_LEN="${SEQ_LEN}"
fi

head_dropout=$4

# mutli-tasking
(
pred_len=$1
lr=$2
t_dim=$3
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

                # traffic
                python -u run_longExp.py \
                --is_training 1 \
                --root_path ./dataset/ \
                --data_path traffic.csv \
                --model_id traffic_$MY_SEQ_LEN'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim \
                --model TRLinear \
                --data custom \
                --features M \
                --seq_len $MY_SEQ_LEN \
                --pred_len $pred_len \
                --enc_in 862 \
                --des 'Exp' \
                --itr 1 \
                --batch_size 8  \
                --in_batch_augmentation \
                --aug_method  $aug_method \
                --aug_rate $aug_rate \
                --t_dim $t_dim \
                --head_dropout $head_dropout \
                --learning_rate $lr >logs/ablations/head_dropout/RLinear/$head_dropout/traffic_$MY_SEQ_LEN'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim.log
                # done
            # done
        done
    done
# done
)
