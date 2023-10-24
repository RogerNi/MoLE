#!/bin/bash

source /jet/home/rni/.bashrc
conda  activate /jet/home/rni/.local/lib/.conda/envs/min-entropy
cd /jet/home/rni/G_STORAGE/min-entropy/Fraug-more-results-1785/FrAug


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

                if [[ -z "${SEED}" ]]; then
                  MY_SEED="2021"
                  LOG_FILE=logs/ablations/original_routing/DLinear/traffic_$MY_SEQ_LEN'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim.log
                else
                  MY_SEED="${SEED}"
                  mkdir -p logs/ablations/original_routing/DLinear/$MY_SEED
                  LOG_FILE=logs/ablations/original_routing/DLinear/$MY_SEED/traffic_$MY_SEQ_LEN'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim.log
                fi

                python -u run_longExp.py \
                --is_training 1 \
                --root_path ./dataset/ \
                --data_path traffic.csv \
                --model_id traffic_$MY_SEQ_LEN'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim \
                --model TDLinear \
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
                --seed $MY_SEED \
                --learning_rate $lr >$LOG_FILE
                # done
            # done
        done
    done
# done
)
