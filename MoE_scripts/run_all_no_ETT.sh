script_full_path=$(dirname "$0")

for script in MoE_traffic MoE_weather MoE_ECL
do

sbatch -p GPU-shared -t 48:00:00 --gpus=v100-32:1 -A cis230033p --mail-type=ALL $script_full_path/$script.sh

done
