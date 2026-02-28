# Graph Learning Tasks

This guide provides an overview of the graph learning experiments conducted with **spikeDE**. These experiments demonstrate the effectiveness of our fractional-order Spiking Neural Networks (f-SNNs) in capturing long-range dependencies on graph-structured data, outperforming traditional integer-order SNNs.

All source code for these experiments is open-source and reproducible. You can access the specific implementation scripts and configurations in our :material-github: [GitHub repository](#).

## Overview

Traditional Spiking Neural Networks (SNNs) typically model neuron dynamics using first-order Ordinary Differential Equations (ODEs), which assume a Markovian property where the current state depends only on the immediate past. This limits their ability to capture complex temporal correlations often present in graph data sequences.

Our **f-SNN** framework integrates **fractional-order differential equations (f-ODEs)** into Spiking Graph Neural Networks (SGNNs). By replacing standard integer-order neurons with fractional-order neurons, our models introduce a power-law memory kernel. This allows the network to retain information from distant past time steps, leading to:

- **Higher Node Classification Accuracy:** Consistently outperforming baselines like SpikingJelly and snnTorch across multiple datasets.
- **Enhanced Robustness:** Superior stability against feature masking and structural perturbations (edge dropping).
- **Energy Efficiency:** Comparable or lower energy consumption due to optimized firing rates, despite the added expressivity of fractional dynamics.

## Datasets

We evaluated our framework on six mainstream graph learning benchmarks, covering citation networks, co-purchase graphs, and large-scale academic graphs:

| Dataset | Description | Task | # Nodes | # Classes |
| :--- | :--- | :--- | :--- | :--- |
| **Cora** | Citation network of machine learning papers. | Node Classification | 2,708 | 7 |
| **Citeseer** | Citation network of scientific publications. | Node Classification | 3,327 | 6 |
| **Pubmed** | Citation network of biomedical articles. | Node Classification | 19,717 | 3 |
| **Photo** | Amazon co-purchase graph (cameras). | Node Classification | 7,650 | 8 |
| **Computers** | Amazon co-purchase graph (computers). | Node Classification | 13,752 | 10 |
| **ogbn-arxiv** | Large-scale citation network of arXiv papers. | Node Classification | 169,343 | 40 |

## Experimental Setup

### Architecture & Baselines
To ensure fair comparison, we integrated our `f-LIF` neurons into two established Spiking Graph Neural Network architectures:

- **SGCN** (Spiking Graph Convolutional Network)
- **DRSGNN** (Dynamic Reactive Spiking Graph Neural Network)

**Baselines:** We compared our results against the original implementations of these models using standard `LIF` neurons from popular frameworks including [SpikingJelly](https://github.com/fangwei123456/spikingjelly) and [snnTorch](https://github.com/jeshraghian/snntorch).

### Key Hyperparameters

- **Encoding:** Poisson spike encoding.
- **Time Steps ($T$):** 100 steps for all graph tasks.
- **Batch Size:** 32.
- **Data Split:** Training/Validation/Test ratio of 0.7/0.2/0.1.
- **Fractional Order ($\alpha$):** Tuned per dataset. Setting $\alpha=1$ recovers the standard integer-order model.
- **Positional Encoding (for DRSGNN):** Dimension 32, using Laplacian Eigenvectors (LSPE) or Random Walk (RWPE).

## Key Results

Our fractional adaptations of SGCN and DRSGNN achieved superior accuracy across all datasets. For example, on the **Cora** dataset, **SGCN (f-SNN)** achieved **88.08%** accuracy, significantly outperforming the SpikingJelly baseline (81.81%).

| Method | Cora | Citeseer | Pubmed | Photo | Computers | ogbn-arxiv |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SGCN (SpikingJelly)** | 81.81 | 71.83 | 86.79 | 87.72 | 70.86 | 50.26 |
| **SGCN (snnTorch)** | 83.12 | 71.68 | 59.82 | 83.34 | 74.88 | 21.55 |
| **SGCN (f-SNN)** | **88.08** | **73.80** | **87.17** | **92.49** | **89.12** | **51.10** |
| **DRSGNN (SpikingJelly)** | 83.30 | 72.72 | 87.13 | 88.31 | 76.55 | 50.13 |
| **DRSGNN (snnTorch)** | 80.98 | 68.00 | 59.56 | 82.28 | 76.78 | 28.46 |
| **DRSGNN (f-SNN)** | **88.51** | **75.11** | **87.29** | **91.93** | **88.77** | **53.13** |

!!! note "Robustness Experiments"
    The f-SNN framework demonstrated exceptional robustness under feature masking and edge dropping scenarios, maintaining high accuracy even when significant portions of graph information were corrupted. Detailed robustness analysis can be found in our [ICLR 2026 Paper](https://arxiv.org/abs/2507.16937).

## Reproducing the Results

The experiments are organized in the `examples/ICLR_Release/Graph` directory of our repository.

### Directory Structure
```text
examples/ICLR_Release/Graph/
├── README.md
├── models.py             # SGCN and DRSGNN definitions with f-LIF support
├── run.py                # Main entry point for training and evaluation
├── utils.py              # Data loading and preprocessing utilities
└── scripts/
    ├── run_sgcn_cora.sh
    ├── run_drsgnn_citeseer.sh
    └── ...
```

### Running the Experiments
You can reproduce the results using the provided shell scripts or by running the main python script directly.

**Using Shell Scripts:**
```bash
# Run SGCN on Cora
bash scripts/run_sgcn_cora.sh

# Run DRSGNN on Citeseer
bash scripts/run_drsgnn_citeseer.sh
```

**Using Command Line:**
```bash
python run.py \
  --backbone SGCN \
  --dataset Cora \
  --time_steps 100 \
  --alpha 0.3 \
  --epochs 100
```

For detailed dataset download links and specific configuration parameters, please refer to the `README.md` inside the `examples/ICLR_Release/Graph` folder.

!!! tip
    For more theoretical details on why fractional calculus improves graph learning, please refer to our [ICLR 2026 Paper](https://arxiv.org/abs/2507.16937) or other tutorials in this documentation.