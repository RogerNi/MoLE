#!/bin/bash



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

head_dropout=$4

(
pred_len=$1
lr=$2
t_dim=$3
aug_rate=0
aug_method=f_mask


if [[ -z "${SEED}" ]]; then
  MY_SEED="2021"
  LOG_FILE=logs/ablations/head_dropout/$MY_MODEL/$head_dropout/ECL_$MY_SEQ_LEN'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim.log
else
  MY_SEED="${SEED}"
  mkdir -p logs/ablations/head_dropout/$MY_MODEL/$head_dropout/$MY_SEED
  LOG_FILE=logs/ablations/head_dropout/$MY_MODEL/$head_dropout/$MY_SEED/ECL_$MY_SEQ_LEN'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim.log
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
--data_path ECL.csv \
--model_id ECL_$MY_SEQ_LEN'_'$pred_len'_'$aug_method'_'$aug_rate'_'$lr'_'$t_dim \
--model MoLE_$MY_MODEL \
--data custom \
--features M \
--seq_len $MY_SEQ_LEN \
--pred_len $pred_len \
--enc_in 321 \
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
)
