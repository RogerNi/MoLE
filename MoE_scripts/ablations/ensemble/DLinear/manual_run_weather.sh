simple_hypersearch "./MoE_weather.sh {pred_len} {lr} {seed}" -p pred_len 96 192 336 720 -p lr 0.005 0.01 0.05 -p seed 1 2 3 4 5 6 | simple_gpu_scheduler --gpus 0,1,2,3
