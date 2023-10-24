script_full_path=$(dirname "$0")

for script in MoE_traffic MoE_ECL
do
for pred_len in 96 192 336 720
do
for aug_rate in 0.5 0.75
do

sbatch -p GPU-shared -t 48:00:00 --gpus=v100-32:1 -A cis230033p --mail-type=ALL $script_full_path/$script.sh $pred_len $aug_rate

done
done
done
