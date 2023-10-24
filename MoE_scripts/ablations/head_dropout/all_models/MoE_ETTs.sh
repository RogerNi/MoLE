#!/bin/bash

cd $(dirname $0)/../../../..


if [[ -z "${SEQ_LEN}" ]]; then
  MY_SEQ_LEN="336"
else
  MY_SEQ_LEN="${SEQ_LEN}"
fi

if [[ -z "${MODEL}" ]]; then
  MY_MODEL="DLinear"
else
  MY_MODEL="${MODEL}"
fi


head_dropout=$5
ett=$4
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

                # Weather

                if [[ -z "${SEED}" ]]; then
                  MY_SEED="2021"
                  LOG_FILE=logs/ablations/head_dropout/$MY_MODEL/$head_dropout/ETT$ett'_'$MY_SEQ_LEN'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim.log
                else
                  MY_SEED="${SEED}"
                  mkdir -p logs/ablations/head_dropout/$MY_MODEL/$head_dropout/$MY_SEED
                  LOG_FILE=logs/ablations/head_dropout/$MY_MODEL/$head_dropout/$MY_SEED/ETT$ett'_'$MY_SEQ_LEN'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim.log
                fi

                if [ -e "$LOG_FILE" ]; then
                    echo "File exists"
                    exit 1
                fi

                mkdir -p "$(dirname "$LOG_FILE")"

                if [ "$TEST_MODE" = true ]; then
                    echo "TEST_MODE" >> $LOG_FILE
                    echo $(date) >> $LOG_FILE
                    continue
                fi
                
                python -u run_longExp.py \
                --is_training 1 \
                --root_path ./dataset/ \
                --data_path ETT$ett.csv \
                --model_id ETT$ett'_'$MY_SEQ_LEN'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim \
                --model T$MY_MODEL \
                --data ETT$ett \
                --features M \
                --seq_len $MY_SEQ_LEN \
                --pred_len $pred_len \
                --enc_in 7 \
                --des 'Exp' \
                --itr 1 \
                --batch_size 8  \
                --in_batch_augmentation \
                --aug_method  $aug_method \
                --aug_rate $aug_rate \
                --t_dim $t_dim \
                --head_dropout $head_dropout \
                --seed $MY_SEED \
                --learning_rate $lr >> $LOG_FILE
                # done
            # done
        done
    done
# done
)
