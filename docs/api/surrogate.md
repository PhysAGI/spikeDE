# SpikeDE.surrogate

This module provides a comprehensive collection of **surrogate gradient functions** and **stochastic spiking mechanisms** designed for training Spiking Neural Networks (SNNs) using backpropagation.

Since the spiking operation (Heaviside step function) is non-differentiable, this library implements various smooth approximations to estimate gradients during the backward pass while maintaining discrete binary spikes in the forward pass. Additionally, it offers a noisy threshold approach that enables stochastic firing during training for improved regularization and biological plausibility.

## Key Features

- **Multiple Surrogate Gradients**: Includes Sigmoid, Arctangent, Piecewise Linear, and Gaussian derivatives, each with distinct mathematical properties suited for different network depths and convergence requirements.
- **Stochastic Spiking**: Implements `NoisyThresholdSpike`, which injects logistic noise into the threshold to create a differentiable soft-spike mechanism during training, automatically reverting to hard spikes during inference.
- **Flexible API**: Available as both reusable `torch.autograd.Function` classes for custom layer integration and functional wrappers for concise usage.

---

::: spikeDE.surrogate
    options:
        filters: public
        members:
        - SigmoidSurrogate
        - ArctanSurrogate
        - PiecewiseLinearSurrogate
        - GaussianSurrogate
        - NoisyThresholdSpikeModule
        - sigmoid_surrogate
        - arctan_surrogate
        - piecewise_linear_surrogate
        - gaussian_surrogate
        - noisy_threshold_spike
        group_by_category: true
        show_submodules: false
