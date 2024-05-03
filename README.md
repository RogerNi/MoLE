# MoLE (AISTATS 2024)

This is the official implementation of the paper "Mixture-of-Linear-Experts for Long-term Time Series Forecasting". [[arXiv]](https://arxiv.org/abs/2312.06786) [[PMLR]](https://proceedings.mlr.press/v238/ni24a.html)

## Requirements

Please refer to the `requirements.txt` file for the required packages.

## Datasets

All datasets we used in our experiments (except Weather2K) are available at this [Google Drive's shared folder](https://drive.google.com/drive/folders/1ZhaQwLYcnhT5zEEZhBTo03-jDwl7bt7v). These datasets were first provided in Autoformer. Please download the datasets and put them in the `dataset` folder. Each dataset is an `.csv` file.

Weather2K dataset is available at this [GitHub repository](https://github.com/bycnfz/weather2k).

## Usage

### Main Experiments

To run the main experiments, please run the following command:

```bash
scripts/run_all_3_seeds.sh
```

This scripts sequentially runs the main experiments on all datasets with 3 different random seeds. The results will be saved in the `logs` folder.

### Additional Experiments

The repository has been cleaned up for easier usage. If you want to run ablation experiments, please refer to earlier commits.

## Acknowledgement

We thank the authors of the following repositories for their open-source code or dataset, which we used in our experiments:

https://github.com/zhouhaoyi/Informer2020

https://github.com/cure-lab/LTSF-Linear

https://github.com/plumprc/RTSF

https://github.com/bycnfz/weather2k

## Citation
If you find our work useful, please consider citing our paper using the following BibTeX:
```
@inproceedings{ni2024mixture,
  title={Mixture-of-Linear-Experts for Long-term Time Series Forecasting},
  author={Ni, Ronghao and Lin, Zinan and Wang, Shuaiqi and Fanti, Giulia},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={4672--4680},
  year={2024},
  organization={PMLR}
}
```
