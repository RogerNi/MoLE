#!/bin/bash

script_full_path=$(dirname "$0")


for seed in 2021 2022 2023
do
    for model in DLinear RLinear RMLP
    do
        for pred_len in 96 192 336 720
        do

            for lr in 0.005 0.01 0.05
            do

                for t_dim in 1 2 3 4 5 6
                do

                    for head_dropout in 0 0.2
                    do
                    
                        SEED=$seed MODEL=$model $script_full_path/MoE_ECL.sh $pred_len $lr $t_dim $head_dropout
                        SEED=$seed MODEL=$model $script_full_path/MoE_traffic.sh $pred_len $lr $t_dim $head_dropout
                        SEED=$seed MODEL=$model $script_full_path/MoE_weather.sh $pred_len $lr $t_dim $head_dropout

                        for ett in h1 h2 m1 m2
                        do
                        SEED=$seed MODEL=$model $script_full_path/MoE_ETTs.sh $pred_len $lr $t_dim $ett $head_dropout
                        done

                    done
                done
            done
        done
    done
done
