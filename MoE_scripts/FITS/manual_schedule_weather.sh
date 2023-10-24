simple_hypersearch "./weather.sh {pred_len} {t_dim}" -p pred_len 96 192 336 720 -p t_dim 1 2 3 4 5 | simple_gpu_scheduler --gpus 0,1,2,3
