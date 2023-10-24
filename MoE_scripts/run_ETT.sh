script_full_path=$(dirname "$0")

for ett in h1 h2 m1 m2
do

sbatch -p GPU-shared -t 48:00:00 --gpus=v100-32:1 -A cis230033p --mail-type=ALL $script_full_path/MoE_ETT.sh $ett

done
