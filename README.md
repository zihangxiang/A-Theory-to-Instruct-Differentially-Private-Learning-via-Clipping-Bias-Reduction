# DPSGD-VRD: A Theory to Instruct Differentially-Private Learning via Clipping Bias Reduction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Published at IEEE S&P 2023**  
- [IEEE Xplore](https://ieeexplore.ieee.org/document/10179409)

## Overview

DPSGD-VRD provides an implementation of differentially private stochastic gradient descent (DP-SGD) with advanced bias reduction techniques for gradient clipping, as proposed in our IEEE S&P 2023 paper. This repository enables reproducible research and experimentation with state-of-the-art DP learning algorithms.

## Features

- Differentially private SGD with clipping bias reduction


## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run experiments using the provided shell script or directly via Python.

**Example: Run DP-SGD with default parameters**
```bash
python main.py --expected_batchsize 5000 --epsilon 8 --EPOCH 40 --lr 0.1 --log_dir logs
```

Or use the batch script for the recommended experiment setup:
```bash
bash run.sh
```

## Arguments

- `--expected_batchsize`: Training batch size (e.g., `5000`)
- `--EPOCH`: Number of training epochs (e.g., `40`)
- `--epsilon`: Privacy budget (e.g., `8`)
- `--lr`: Learning rate (e.g., `0.1`)
- `--log_dir`: Directory to save logs (default: `logs`)
- `--beta`: (Optional) Momentum beta (default: `0.9`)
- `--which_norm`: (Optional) Clipping norm type (default: `2`)
- `--C`: (Optional) Clipping constant (default: `1`)

## Citing

If you use this code, please cite:

```tex
@inproceedings{DBLP:conf/sp/XiaoXWD23,
  author       = {Hanshen Xiao and Zihang Xiang and Di Wang and Srinivas Devadas},
  title        = {A Theory to Instruct Differentially-Private Learning via Clipping Bias Reduction},
  booktitle    = {44th {IEEE} Symposium on Security and Privacy, {SP} 2023, San Francisco, CA, USA, May 21-25, 2023},
  pages        = {2170--2189},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/SP46215.2023.10179409}
}
```