# Welcome to spikeDE's documentation

**spikeDE** is a powerful, :simple-pytorch: [PyTorch](https://pytorch.org)-native library designed to build, train, and deploy **Fractional-Order Spiking Neural Networks (f-SNNs)**.

While traditional Spiking Neural Networks (SNNs) rely on integer-order differential equations (e.g., standard Leaky Integrate-and-Fire models) that assume Markovian dynamics—where the current state depends solely on the immediate past—**spikeDE** introduces a generalized **fractional calculus framework**. By utilizing Caputo fractional derivatives ($D^\alpha$), our library enables neural units with inherent **long-term memory** and **non-Markovian properties**, closely mimicking the complex temporal dynamics and fractal structures observed in biological neurons.

## Why Fractional?

The core innovation of **spikeDE** lies in its ability to treat fractional order as a strict generalization of integer order. It is not merely an alternative model; it is a comprehensive superset framework.

| Feature | Traditional Integer-Order SNNs ($\alpha = 1$) | **spikeDE** Fractional-Order SNNs ($0 < \alpha \le 1$) |
| :--- | :--- | :--- |
| **Dynamics** | Forgets history rapidly; strictly Markovian. | Integrates history over long time horizons with heavy-tailed memory. |
| **Memory Mechanism** | Short-term memory with a fixed time constant. Simulating long dependencies requires large, deep networks. | A single fractional neuron naturally captures multi-scale temporal correlations without extra depth. |
| **Expressivity** | Limited by fixed decay rates and finite timescales. | Theoretical analysis proves that one fractional neuron cannot be equated to any finite ensemble of integer-order neurons. |
| **Robustness** | Often sensitive to input noise and parameter perturbations. | Fractional dynamics provide enhanced robustness against disturbances and noise. |

In **spikeDE**, setting the fractional order $\alpha = 1$ seamlessly recovers standard SNN behavior, ensuring backward compatibility. Conversely, setting $0 < \alpha < 1$ unlocks the power of fractional dynamics, offering superior performance in tasks requiring complex temporal processing, such as neuromorphic vision, speech recognition, and event-based graph learning.

---

## Documentation Overview

Navigate through our comprehensive guides to get started, master the core concepts, or dive into advanced API usage.

<div class="grid cards" markdown>

- :material-rocket-launch: **Getting Started**

    ---
    New to **spikeDE**? Start here to install the package and launch your first fractional spiking network with our step-by-step quickstart guide.

    [:material-arrow-right: Installation & Quickstart](./get_start/index.md)

- :material-school: **Tutorials**

    ---
    From core concepts to advanced mastery. Deepen your understanding of the mathematical foundations and learn how to optimize your **spikeDE** workflows.

    [:material-arrow-right: Explore Tutorials](./tutorials/index.md)

- :material-code-tags: **API Reference**

    ---
    Comprehensive documentation for **spikeDE's** core modules. Build custom architectures with full control.

    [:material-arrow-right: Browse API Docs](./api/index.md)

- :material-rss: **Blog & Updates**

    ---
    Stay updated with the latest posts on new features, performance benchmarks, and insights from the community.

    [:material-arrow-right: Visit Blog](./blog/index.md)

</div>

Happy spiking with fractional dynamics!
