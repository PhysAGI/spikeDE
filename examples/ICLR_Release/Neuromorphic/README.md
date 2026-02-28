# ICLR Release: Neuromorphic Data Classification Tasks

This directory provides an open-source-friendly categorization of two training tasks without altering the original locations of the training scripts.

## Directory Structure

- `file/scripts/`: Execution scripts (categorized by task)
- `file/logs/`: Training logs (named by task)

## Task Categorization

1. **N101 Transformer**
   - Training Script: `train_n101_transformer.py`
   - Launch Script: `file/scripts/run_n101_transformer.sh`
   - Log File: `file/logs/train_n101_alpha0.1_b32.log`

2. **HardVS CNN**
   - Training Script: `train_hardvs_cnn.py`
   - Launch Script: `file/scripts/run_hardvs_cnn.sh`
   - Log File: `file/logs/train_dvs_alpha0.8.log`

## Execution Methods

Run the following commands from the `iclr_release` directory:

```bash
bash file/scripts/run_n101_transformer.sh
bash file/scripts/run_hardvs_cnn.sh
```

Or launch all tasks at once:

```bash
bash file/scripts/run_all.sh
```

## Corresponding Original Commands

- `nohup python train_n101_transformer.py --alpha 0.1 --device cuda:5 -b 32 > file/logs/train_n101_alpha0.1_b32.log 2>&1 &`
- `nohup python train_hardvs_cnn.py --alpha 0.8 -device cuda:7 -b 32 > file/logs/train_dvs_alpha0.8.log 2>&1 &`

## Dataset Download Information

### 1. N-Caltech101

Base URL: `https://www.garrickorchard.com/datasets/n-caltech101`

- `Caltech101.zip` (MD5: `66201824eabb0239c7ab992480b50ba3`)
- `Caltech101_annotations.zip` (MD5: `25e64cea645291e368db1e70f214988e`)
- `ReadMe(Caltech101)-SINAPSE-G.txt` (MD5: `d464b81684e0af3b5773555eb1d5b95c`)
- `ReadMe(Caltech101).txt` (MD5: `33632a7a5c46074c70509f960d0dd5e5`)

### 2. HARDVS

Base URL: `https://github.com/Event-AHU/HARDVS`

- `test_label.txt` (MD5: `5b664af5843f9b476a9c22626f7f5a59`)
- `train_label.txt` (MD5: `0d642b6e6871034f151b2649a89d8d3c`)
- `val_label.txt` (MD5: `cd2cebcba80e4552102bbacf2b5df812`)
- `MINI_HARDVS_files.zip` (MD5: `9c4cc0d9ba043faa17f6f1a9e9aff982`)
