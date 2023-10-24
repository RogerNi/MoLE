script_full_path=$(dirname "$0")

GRAM=32

PRED_LEN=100

seq_len_list="6 88 170 254"
# seq_len_list="6 254"


# Weather (no dropout)
for ablation in original_routing
do

for pred_len in $PRED_LEN
do

for lr in 0.005 0.01 0.05
do

# for t_dim in 1 2 3 4 5
for t_dim in 1 2 3 4 5 6
do

for seq_len in $seq_len_list
do


sbatch -p GPU-shared -t 02:00:00 --gpus=v100-$GRAM:1 -A cis230033p --export=SEQ_LEN=$seq_len $script_full_path/$ablation/DLinear/MoE_ECL.sh $pred_len $lr $t_dim
sbatch -p GPU-shared -t 04:45:00 --gpus=v100-$GRAM:1 -A cis230033p --export=SEQ_LEN=$seq_len $script_full_path/$ablation/DLinear/MoE_traffic.sh $pred_len $lr $t_dim

done
done
done
done
done

