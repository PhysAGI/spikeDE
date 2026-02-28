# SpikeDE.snn

This module provides the `SNNWrapper` class, which converts standard Spiking Neural Networks (SNNs)
into Fractional Differential Equation (FDE) systems. It supports flexible configuration of 
fractional orders ($\alpha$) per neuron layer, including single-term, multi-term (distributed order),
and learnable parameters.

## Key Features

- Per-layer fractional orders (single-term or multi-term).
- Learnable $\alpha$ and coefficients via backpropagation.
- Automatic shape inference and parameter registration.
- Support for various FDE solvers (Grunwald-Letnikov, L1, etc.).

---

::: spikeDE.snn
    options:
        members:
        - PerLayerAlphaConfig
        - SNNWrapper
        filters: public
        group_by_category: true
        show_submodules: false
