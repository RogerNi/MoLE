script_full_path=$(dirname "$0")

for batch_size in 4 8 16 32 64 128 256
do

sbatch -p GPU-shared -t 48:00:00 --gpus=v100-32:1 -A cis230033p --mail-type=ALL $script_full_path/MoE_weather.sh $batch_size

done
