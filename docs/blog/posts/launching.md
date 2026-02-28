---
date: 2026-02-28
categories:
  - Release
  - Research
authors:
  - team
comments: true
draft: false
slug: launching-spikede-fractional-snn
title: "Introducing spikeDE: Bridging Fractional Calculus and Spiking Neural Networks"
description: "We are thrilled to announce the open-source release of spikeDE, a PyTorch based library that empowers Spiking Neural Networks with long-range memory via fractional-order dynamics."
pin: true
---

# Introducing spikeDE: Where Fractional Calculus Meets Spiking Neural Networks

Today marks a significant milestone for our research team. We are incredibly proud to announce the public release of **spikeDE**, an open-source PyTorch based library designed to bring **Fractional-Order Dynamics** to the world of Spiking Neural Networks (SNNs).

For years, SNNs have been celebrated for their biological plausibility and energy efficiency. However, traditional models like the Leaky Integrate-and-Fire (LIF) neuron rely on integer-order differential equations ($\alpha=1$), which assume **Markovian dynamics**—meaning the neuron's current state depends **only** on its immediate past. This simplification fails to capture the rich, complex temporal dependencies observed in real biological neurons, which exhibit **long-term memory** and **power-law relaxation**.

With **spikeDE**, we change the paradigm.

<!-- more -->

## The Core Idea: Beyond Integer Orders

As detailed in our upcoming paper at **ICLR 2026**, [*"Fractional-order Spiking Neural Network"*](https://arxiv.org/abs/2507.16937), biological systems often operate on multiple time scales simultaneously. A single integer-order neuron cannot efficiently represent this spectrum without stacking infinite layers.

**spikeDE** introduces the **Caputo fractional derivative** ($D^\alpha$, where $0 < \alpha \leq 1$) into the membrane potential dynamics:

$$ \tau D^\alpha U(t) = f(t, U(t), I(t)) $$

By tuning the fractional order $\alpha$, our *f*-LIF neurons naturally exhibit:

1.  **Heavy-tailed Memory**: Past inputs influence the current state via a Mittag-Leffler function decay, not just a simple exponential.
2.  **Non-Markovian Behavior**: The system inherently "remembers" its history, capturing long-range dependencies crucial for processing temporal data like event-based vision or dynamic graphs.
3.  **Enhanced Robustness**: Our theoretical analysis shows that fractional dynamics suppress noise accumulation sub-linearly ($t^\alpha$ vs $t$), making *f*-SNNs significantly more robust to input perturbations.

## What's New in spikeDE?

The initial release of **spikeDE** is built from the ground up to be flexible, efficient, and strictly compatible with the PyTorch ecosystem.

### Native Fractional Solvers
We integrate optimized solvers directly into the computational graph. This allows for **end-to-end training**, even with non-local fractional operators.

### Per-Layer Customization
Not all layers need the same memory depth. **spikeDE** allows you to set distinct $\alpha$ values for each layer or even make $\alpha$ a **learnable parameter**, letting the network discover the optimal time-scale spectrum for your specific task.

```python
from spikeDE import SNNWrapper, LIFNeuron

# Make alpha learnable! The network decides how much memory it needs.
net = SNNWrapper(
    base=my_snn_model,
    integrator='fdeint',
    alpha=[0.5, 0.8, 0.9], # Different memory depths per layer
    learn_alpha=True       # Enable gradient updates for alpha
)
```

### Strict Generalization
**spikeDE** is a strict superset of traditional SNNs. Setting $\alpha=1.0$ recovers the standard LIF dynamics exactly. This means you can seamlessly migrate existing CNN-to-SNN or direct-training workflows to the fractional domain with minimal code changes.

## Early Results: State-of-the-Art Performance

Our experiments, documented in the [ICLR 2026 paper](https://arxiv.org/abs/2507.16937), demonstrate that *f*-SNNs consistently outperform their integer-order counterparts:

- **Neuromorphic Vision**: On the **HarDVS** dataset, our *f*-SNN achieved **47.66%** accuracy, surpassing the best integer-order baseline by **+1.55%**.
- **Graph Learning**: In dynamic graph tasks (e.g., **Cora**), the fractional Spiking Graph Convolutional Network showed a remarkable **+6.2%** improvement in node classification, proving the power of long-range temporal aggregation on graph structures.
- **Robustness**: Under heavy noise injection and time-jitter attacks, *f*-SNNs maintained stable performance where traditional SNNs degraded rapidly.

## Getting Started

Getting started with **spikeDE** is easy. Whether you are a neuroscientist modeling biological circuits or a machine learning engineer building low-power AI, our documentation has you covered.

## Summary

We believe that **Fractional Calculus** holds the key to unlocking the next generation of efficient, brain-inspired AI. By open-sourcing **spikeDE**, we hope to lower the barrier for researchers to explore this fascinating intersection of mathematics and neuroscience.

<div class="grid" markdown>

[:simple-arxiv: Full Paper](https://arxiv.org/abs/2507.16937){ .md-button }
[:material-file-document: Documentation](../index.md){ .md-button }
[:material-github: Source Code](https://github.com/PhysAGI/spikeDE){ .md-button }

</div>

We welcome contributions, issues, and discussions. Let's build the future of memory-rich neural networks together! Happy Spiking!
