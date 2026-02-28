# ICLR Release: Graph Learning Tasks

This folder is a minimal, clean release for Cora, Citeseer, Pubmed, Photo, Computers and Arxiv training. It keeps only the files required to run the code.

## Directory Structure

```
.
├── README.md
├── models.py
├── run.py
├── scripts/
│   ├── run_drsgnn_citeseer.sh
│   ├── run_drsgnn_cora.sh
│   ├── run_sgcn_citeseer.sh
│   └── run_sgcn_cora.sh
└── utils.py
```

## Running Guide

### Prerequisites

Ensure you have the following installed:
- Python ≥ 3.8
- PyTorch ≥ 1.12
- torch-geometric
- ogb
- spikingjelly
- spikeDE

### Basic Usage

Run the default experiment on Cora with SpikingGCN:

```bash
python run.py
```

### Command-Line Arguments

You can customize the experiment using the following arguments:

```bash
python run.py \
  --backbone SGCN \                         # or DRSGNN
  --dataset Cora \                          # Cora, Citeseer, Pubmed, Photo, Computers, arxiv
  --split ratio \                           # public, random, or ratio
  --ratio 0.7 0.2 0.1 \                     # train/val/test split (if --split ratio)
  --batch_size 512 \
  --epochs 100 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --time_steps 100 \
  --tau 0.5 \
  --threshold 1.0 \
  --alpha 0.5 \                             # float, [0.5, 0.6], or [[0.5],[0.6]]
  --coefficients "[0.9, 0.1]" \             # list format as string
  --learn_alpha \
  --learn_coefficient \
  --early_stop \
  --patience 50 \
  --device cuda
```

> **Note**: For `--alpha` and `--coefficients`, pass complex structures as JSON-like strings (e.g., `"[[0.5, 0.5], [0.6, 0.4]]"`). The parser supports `float`, `List[float]`, and `List[List[float]]`.

Pre-configured scripts are available in the `scripts/` directory for quick reproduction:

```bash
bash scripts/run_sgcn_cora.sh
```

### Output

Results are saved under:
```
checkpoints/
└── {backbone}_GLMulti_{dataset}/
    └── {timestamp}/
        ├── best_model_*.pth
        └── log.json
```

The `log.json` contains training/validation metrics, test accuracy (mean ± std over 20 runs), and hyperparameter settings.

### Example Commands

**Train DRSGNN on PubMed with custom alpha per layer:**
```bash
python run.py --backbone DRSGNN --dataset Pubmed --alpha "[[0.4], [0.6]]" --learn_alpha
```

**Reproduce results on OGB-Arxiv with fixed coefficients:**
```bash
python run.py --dataset arxiv --coefficients "[0.8, 0.15, 0.05]" --time_steps 200
```