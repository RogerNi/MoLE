script_full_path=$(dirname "$0")
for pred_len in 96 192 336 720
do

for t_dim in 1 2 3 4 5
do

sbatch -p GPU-shared -t 48:00:00 --gpus=v100-32:1 -A cis230033p --mail-type=ALL $script_full_path/weather.sh $pred_len $t_dim

done
done