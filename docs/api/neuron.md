# spikeDE.neuron

This module provides a flexible and extensible framework for building Spiking Neural Networks (SNNs) that seamlessly bridge standard integer-order dynamics with advanced fractional-order calculus. Unlike traditional frameworks that rely on discrete step-by-step updates, **spikeDE** reimagines neurons as continuous dynamical systems. This architectural shift allows users to upgrade standard models into Fractional-Order Spiking Neurons, endowing them with infinite memory and complex temporal dependencies without altering core logic.

At the heart of this module is the separation of concerns: neuron classes define instantaneous dynamics (computing derivatives), while external solvers (via `SNNWrapper`) handle state evolution and fractional integration. This design supports a wide range of models, from classic Integrate-and-Fire variants to sophisticated noisy-threshold and hard-reset mechanisms, all compatible with surrogate gradient learning.

## Key Features

- **Modular Architecture**: Stateless neuron modules compute derivatives ($dv/dt$) independently of state history, allowing them to work interchangeably with standard (`odeint`) and fractional (`fdeint`) solvers.
- **Learnable Parameters**: Supports learnable membrane time constants ($\tau$) via exponential reparameterization and customizable surrogate gradient functions (e.g., arctan, sigmoid) for effective backpropagation through non-differentiable spikes.
- **Extensibility**: Provides a clear `BaseNeuron` interface for defining custom dynamics, ensuring that user-defined neurons automatically inherit fractional capabilities when wrapped in the appropriate solver.

---

::: spikeDE.neuron
    options:
        members:
        - BaseNeuron
        - IFNeuron
        - LIFNeuron
        filters: public
        group_by_category: true
        show_submodules: false
