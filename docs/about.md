# About spikeDE

Traditional SNNs, governed by integer-order differential equations, often struggle to capture long-range temporal dependencies without resorting to deep, computationally expensive architectures. Our team recognized that biological neurons operate with complex, non-Markovian dynamics—properties naturally described by **fractional calculus**.

The goal of **spikeDE** is to bridge the gap between advanced mathematical theory and practical deep learning applications. By providing a robust, :simple-pytorch: PyTorch-native implementation of Fractional-Order SNNs (f-SNNs), we aim to empower researchers and engineers to build more expressive, robust, and biologically plausible neural models with minimal effort.

## The Research Behind spikeDE

The core algorithms and theoretical foundations of **spikeDE** are based on our paper, **"Fractional-order Spiking Neural Network"**, which has been accepted by **ICLR 2026** (International Conference on Learning Representations).

### Paper Highlights

Our work introduces a paradigm shift from Markovian integer-order dynamics to non-Markovian fractional-order dynamics. Here are the key contributions:

<div class="grid cards" markdown>

-   :material-brain: **Capturing Long-Range Dependencies**

    ---

    Unlike traditional LIF neurons that rely on exponential decay, our **f-LIF** neurons utilize power-law relaxation via the Mittag-Leffler function. This allows the membrane potential to retain a "heavy-tailed" memory of past inputs, naturally modeling the complex temporal correlations observed in biological neurons.

-   :material-shield: **Proven Robustness & Stability**

    ---

    We provide theoretical guarantees showing that fractional-order dynamics suppress perturbation accumulation sub-linearly ($t^\alpha$ vs $t$). Experiments confirm that **spikeDE** models maintain superior accuracy under heavy noise injection, occlusion, and temporal jitter compared to integer-order baselines.

-   :material-puzzle: **Strict Generalization**

    ---

    The **spikeDE** framework is a strict superset of traditional SNNs. By setting the fractional order $\alpha=1$, it recovers standard IF/LIF dynamics. This ensures seamless compatibility with existing architectures like CNNs, Transformers, and GNNs, requiring only a drop-in replacement of the neuron module.

-   :octicons-zap-24: **Efficiency Without Compromise**

    ---

    Despite the added expressivity, our optimized solvers (using short-memory principles and FFT-based convolution) ensure that **f-SNNs** achieve **comparable energy efficiency** to traditional SNNs while delivering state-of-the-art performance on neuromorphic vision and graph learning tasks.

</div>

!!! quote "Key Theoretical Insight"
    A single fractional-order neuron represents a continuum of timescales that would require infinitely many integer-order units for exact equivalence. This irreducibility grants f-SNNs fundamentally richer expressive power.

### Citation

If you use **spikeDE** in your research or applications, please consider citing our paper.

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
<div class="grid" markdown>

[:simple-arxiv: Read the full paper on arXiv](https://arxiv.org/abs/2507.16937){ .md-button }

[:material-eye: View on OpenReview (ICLR 2026)](https://openreview.net/forum?id=NJhBSLJ0nL){ .md-button }

</div>

## Community & Contribution

**spikeDE** is an open-source project licensed under the [MIT License](https://opensource.org/licenses/MIT). We welcome contributions from the community!

Whether you find a bug, have a feature request, want to improve documentation, or wish to contribute new fractional solvers/neuron models, please feel free to open an :material-bug: issue or submit a :material-source-pull: Pull Request on our :material-github: [GitHub Repository](https://github.com/PhysAGI/spikeDE).

Let's build the future of Spiking Neural Networks together.
