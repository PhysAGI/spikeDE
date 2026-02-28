# spikeDE: Fractional-Order Spiking Neural Networks in PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.8-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-ICLR%202026-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.16937)

**spikeDE** is a powerful, native **PyTorch** library designed for building, training, and deploying **Fractional-Order Spiking Neural Networks (*f*-SNNs)**.

Traditional SNNs are often limited by integer-order differential equations (e.g., standard LIF models), assuming Markovian dynamics where the current state depends only on the immediate past. **spikeDE** introduces a generalized fractional calculus framework utilizing Caputo fractional derivatives ($D^\alpha$). This endows neural units with inherent **long-term memory** and **non-Markovian properties**, allowing for a closer simulation of the complex temporal dynamics and fractal structures observed in biological neurons.

## 🌟 Key Highlights

*   **🧠 Capturing Long-Range Dependencies**
    Unlike the exponential decay of traditional LIF neurons, our *f*-LIF neurons utilize the Mittag-Leffler function to achieve power-law relaxation. This enables membrane potentials to retain a "heavy-tailed" memory of past inputs, naturally simulating the complex temporal correlations found in biological neurons.

*   **🛡️ Superior Robustness and Stability**
    Theoretical proofs demonstrate that fractional-order dynamics sub-linearly suppress perturbation accumulation ($t^\alpha$ vs $t$). Experiments show that spikeDE models maintain higher accuracy than integer-order baselines under heavy noise injection, occlusion, and temporal jitter.

*   **🧩 Strict Generalization Capability**
    spikeDE is a strict superset of traditional SNNs. By setting the fractional order $\alpha=1$, it fully recovers standard IF/LIF dynamics. This means it can seamlessly integrate with existing CNN, Transformer, and GNN architectures simply by replacing the neuron modules.

*   **⚡ Efficient and Uncompromised**
    Despite increased expressiveness, our optimized solvers (leveraging the short-memory principle and FFT-based convolution) ensure that *f*-SNNs maintain energy efficiency comparable to traditional SNNs while delivering state-of-the-art performance.

> **Key Theoretical Insight**: A single fractional-order neuron represents a continuous spectrum of time scales, which would require an infinite number of integer-order units to equate precisely. This irreducibility grants *f*-SNNs fundamentally richer expressive power.

## 📦 Installation Guide

spikeDE supports Python 3.9 through 3.13. Installation in a virtual environment is recommended.

### 1. Install Dependencies

First, install PyTorch (refer to the [PyTorch Official Site](https://pytorch.org/get-started/locally/) for platform-specific commands), then install the necessary differential equation solvers:

```bash
# Install ODE solver (for integer-order)
pip install torchdiffeq

# Install FDE solver (for fractional-order, core dependency)
pip install git+https://github.com/kangqiyu/torchfde.git
```

### 2. Install spikeDE

Install directly from source:

```bash
pip install git+https://github.com/PhysAGI/spikeDE.git
```

## 🚀 Quick Start

Use `SNNWrapper` to easily convert existing discrete-time SNNs into continuous-time fractional-order systems.

### Basic Usage Example

```python
import torch
import torch.nn as nn
from spikeDE import SNNWrapper, LIFNeuron

# 1. Define a standard PyTorch SNN model
class MySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.lif1 = LIFNeuron(tau=2.0, threshold=1.0)
        self.fc2 = nn.Linear(20, 5)
        self.lif2 = LIFNeuron(tau=2.0, threshold=1.0)

    def forward(self, x):
        x = self.lif1(self.fc1(x))
        x = self.lif2(self.fc2(x))
        return x

model = MySNN()

# 2. Wrap the model with SNNWrapper to enable fractional-order dynamics
net = SNNWrapper(
    base=model,
    integrator='fdeint',      # Use fractional differential equation solver
    alpha=0.7,                # Set fractional order (0 < alpha <= 1)
    method='gl',              # Use Grünwald-Letnikov method
    learn_alpha=False         # Whether to make alpha a learnable parameter
)

# 3. [IMPORTANT] Initialize neuron shapes (Must be called before the first forward pass)
# input_shape should match the input data shape (excluding the time dimension)
net._set_neuron_shapes(input_shape=(1, 10)) 

# 4. Prepare data and run
# Input shape: [Time_Steps, Batch_Size, Features]
time_steps = 50
batch_size = 4
x_time = torch.linspace(0, 1, time_steps)
x_input = torch.randn(time_steps, batch_size, 10)

output = net(x_input, x_time)
print(f"Output shape: {output.shape}")
```

### Advanced Feature: Layer-wise Custom Memory Dynamics

You can set different $\alpha$ values for each layer of the network, or even let the network automatically learn the optimal memory depth:

```python
# Set alpha for a two-layer network: Strong memory for layer 1 (0.3), Weak memory for layer 2 (0.8)
net = SNNWrapper(
    base=model,
    integrator='fdeint',
    alpha=[0.3, 0.8],          # List length must match the number of neuron layers
    alpha_mode='per_layer',    # Explicitly declare per-layer mode
    learn_alpha=True           # Enable gradient updates to let the network optimize alpha automatically
)
net._set_neuron_shapes(input_shape=(1, 10))
```

## 🧪 Performance

spikeDE demonstrates superior performance compared to traditional SNN frameworks (such as SpikingJelly and snnTorch) across multiple benchmarks:

| Task | Dataset | Baseline (SpikingJelly) | **spikeDE (*f*-SNN)** | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Neuromorphic** | HarDVS | 46.10% | **47.66%** | +1.55% |
| **Neuromorphic** | N-Caltech101 | 72.63% | **76.27%** | +3.64% |
| **Graph Learning** | Cora (SGCN) | 81.81% | **88.08%** | +6.27% |
| **Graph Learning** | Photo (DRSGNN) | 88.31% | **91.93%** | +3.62% |

> For more detailed experimental data and robustness analysis, please refer to our [ICLR 2026 paper](https://arxiv.org/abs/2507.16937).*

## 📚 Documentation & Resources

*   **[Full Documentation](https://physagi.github.io/spikeDE/)**: Includes detailed API references, tutorials, and mathematical principle explanations.
*   **[Example Code](examples/)**: Contains neuromorphic vision and graph learning scripts required to reproduce paper results.
*   **[Paper Link](https://arxiv.org/abs/2507.16937)**: "Fractional-order Spiking Neural Network", ICLR 2026.

## 🔖 Citation

If you use spikeDE in your research or applications, please cite our paper:

```bibtex
@misc{ge2026fractionalorderspikingneuralnetwork,
      title={Fractional-order Spiking Neural Network}, 
      author={Chengjie Ge and Yufeng Peng and Zihao Li and Qiyu Kang and Xueyang Fu and Xuhao Li and Qixin Zhang and Junhao Ren and Zheng-Jun Zha},
      year={2026},
      eprint={2507.16937},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2507.16937}, 
}
```

## 🤝 Contributing

spikeDE is an open-source project licensed under the [MIT License](LICENSE). We welcome contributions from the community!
Whether you are reporting bugs, requesting new features, improving documentation, or contributing new fractional-order solvers/neuron models, please feel free to submit an Issue or Pull Request.

Let's build the future of Spiking Neural Networks together.
