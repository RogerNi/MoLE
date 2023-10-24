script_full_path=$(dirname "$0")
for pred_len in 96 192 336 720
do

for lr in 0.005 0.01 0.05
do

# for t_dim in 1 2 3 4 5
for t_dim in 1 2 3 4 5 6
do

for data_idx in 79 114 850 1786
do

sbatch -p GPU-shared -t 48:00:00 --gpus=v100-16:1 -A cis230033p $script_full_path/MoE_weather.sh $pred_len $lr $t_dim $data_idx

done
done
done
done
