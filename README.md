# Mixture-of-Linear-Experts for Long-term Time Series Forecasting (MoLE)

This is the official implementation of the paper "Mixture-of-Linear-Experts for Long-term Time Series Forecasting".

## Requirements

Please refer to the `requirements.txt` file for the required packages.

## Datasets

All datasets we used in our experiments are available at this [Google Drive's shared folder](https://drive.google.com/drive/folders/1ZhaQwLYcnhT5zEEZhBTo03-jDwl7bt7v). These datasets were first provided in Autoformer. Please download the datasets and put them in the `dataset` folder. Each dataset is an `.csv` file.

## Usage

### Main Experiments

To run the main experiments, please run the following command:

```bash
MoE_scripts/run_all_3_seeds.sh
```

This scripts sequentially runs the main experiments on all datasets with 3 different random seeds. The results will be saved in the `logs` folder.

### RandomIn/RandomOut Experiments

To run the RandomIn/RandomOut experiments, please refer to `MoE_scripts/ablations/[experiments]`. The experiments includes:
 - original_routing: Original experiments
 - head_dropout: Original experiments with head dropout
 - random_bypass: RandomOut experiments
 - random_bypass_dropout: RandomOut experiments with head dropout
 - random_routing: RandomIn experiments
 - random_routing_dropout: RandomIn experiments with head dropout

### Short Input Length Experiments

To run the short input length experiments, please refer to `MoE_scripts/ablations/run_all_short_seq_len.sh`.

## Acknowledgement

We thank the authors of the following repositories for their open-source code, which we used in our experiments:

https://github.com/zhouhaoyi/Informer2020

https://github.com/cure-lab/LTSF-Linear

https://github.com/plumprc/RTSF

