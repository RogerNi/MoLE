script_full_path=$(dirname "$0")
for pred_len in 96 192 336 720
do

for lr in 0.005 0.01 0.05
do

# for t_dim in 1 2 3 4 5
for t_dim in 1 2 3 4 5 6
do

for head_dropout in 0.2
do

sbatch -p GPU-shared -t 00:40:00 --gpus=v100-16:1 -A cis230033p $script_full_path/MoE_weather.sh $pred_len $lr $t_dim $head_dropout

done

done
done
done
