# Surrogate Gradient Learning in spikeDE: Differentiating the Non-Differentiable

Training Spiking Neural Networks (SNNs) presents a fundamental challenge: the spiking mechanism is inherently non-differentiable. The generation of a spike is typically modeled by the Heaviside step function, which has a zero gradient almost everywhere and an undefined gradient at the threshold. This discontinuity prevents the direct application of standard backpropagation algorithms.

**spikeDE** overcomes this barrier using **Surrogate Gradient (SG)** learning. This technique decouples the forward and backward passes:

- **Forward Pass:** The network uses the true, hard Heaviside step function to generate discrete spikes, preserving the event-driven nature and energy efficiency of SNNs.
- **Backward Pass:** During gradient computation, the undefined derivative of the step function is replaced by a smooth, differentiable **surrogate function**.

This approach allows gradients to flow through the spiking nonlinearity, enabling end-to-end training of fractional-order spiking networks using standard optimizers like Adam or SGD.

## The Mathematical Formulation

Let $U(t)$ be the membrane potential and $\theta$ be the firing threshold. The spike output $S(t)$ in the forward pass is defined as:

$$ S(t) = H(U(t) - \theta) = \begin{cases} 1 & \text{if } U(t) \geq \theta \\ 0 & \text{if } U(t) < \theta \end{cases} $$

Since $\frac{\partial S}{\partial U}$ is undefined, we approximate it with a surrogate gradient function $\sigma'(U - \theta)$ during backpropagation. The chain rule then becomes:

$$ \frac{\partial \mathcal{L}}{\partial U} = \frac{\partial \mathcal{L}}{\partial S} \cdot \underbrace{\sigma'(U - \theta)}_{\text{Surrogate Gradient}} $$

where $\mathcal{L}$ is the loss function. The choice of $\sigma'$ significantly impacts convergence speed, stability, and final accuracy.

## Available Surrogate Functions

spikeDE provides a suite of built-in surrogate gradient functions, each with distinct mathematical properties and hyperparameters. You can select them via the `surrogate` argument in your neuron configuration.

### Sigmoid Surrogate
The most widely used surrogate, derived from the derivative of the scaled sigmoid function. It provides a smooth, bell-shaped gradient centered at the threshold.

- **Formula:**

    $$ \sigma'(x) = \kappa \cdot \text{sigmoid}(\kappa x) \cdot (1 - \text{sigmoid}(\kappa x)) $$

- **Hyperparameter:** `scale` ($\kappa$, default: 5.0). Controls the sharpness. Larger $\kappa$ approximates the true step function more closely but may lead to vanishing gradients.
- **Best For:** General-purpose training; robust baseline for most architectures.

### Arctangent Surrogate
Based on the derivative of the arctangent function. It features heavier tails compared to the sigmoid, allowing gradients to propagate even when the membrane potential is far from the threshold.

- **Formula:**

    $$ \sigma'(x) = \frac{\kappa}{1 + (\kappa x)^2} $$

- **Hyperparameter:** `scale` ($\kappa$, default: 2.0).
- **Best For:** Deep networks where gradient vanishing is a concern; scenarios requiring broader credit assignment.

!!! note
    spikeDE implements a normalized variant to ensure stable gradient magnitudes.

### Piecewise Linear Surrogate
A computationally efficient approximation that defines a triangular window around the threshold. Gradients are constant within the window and zero outside.

- **Formula:**

    $$ \sigma'(x) = \begin{cases} \frac{1}{2\gamma} & \text{if } |x| \leq \gamma \\ 0 & \text{otherwise} \end{cases} $$

- **Hyperparameter:** `gamma` ($\gamma$, default: 1.0). Defines the width of the active region.
- **Best For:** High-speed training on resource-constrained hardware; models where sparse gradient updates are preferred.

### Gaussian Surrogate
Uses a normalized Gaussian function to approximate the derivative. It offers the smoothest profile with exponential decay, providing very localized gradient updates.

- **Formula:**

    $$ \sigma'(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{x^2}{2\sigma^2}} $$

- **Hyperparameter:** `sigma` ($\sigma$, default: 1.0). Controls the spread of the gradient.
- **Best For:** Precision tasks where only neurons very close to firing should receive updates.

### Noisy Threshold Spike
Instead of a hard spike in the forward pass, this method injects logistic noise into the threshold, creating a stochastic soft spike during training while reverting to a hard spike during inference.

- **Mechanism:** $S(t) = \text{sigmoid}(\kappa(U(t) - \theta) + \epsilon)$, where $\epsilon \sim \text{Logistic}(0, 1)$.
- **Best For:** Improving exploration during training and enhancing robustness to input noise.

## Configuration and Usage

Configuring the surrogate gradient is straightforward within the `spikeDE` neuron definitions. You can specify the surrogate function by name when initializing your neurons:

```python
from spikeDE import LIFNeuron

neuron = LIFNeuron(
    tau=2.0, 
    threshold=1.0, 
    surrogate='arctan',      # Options: 'sigmoid', 'arctan', 'linear', 'gaussian', 'noisy'
    surrogate_scale=2.0      # Specific hyperparameter for the chosen surrogate
)
```

## Summary

| Method | Formula Shape | Tail Behavior | Computational Cost | Best Use Case |
| :--- | :--- | :--- | :---: | :--- |
| **`sigmoid`** | Bell-shaped | Exponential decay | Low | **Default**; balanced performance. |
| **`arctan`** | Bell-shaped | Heavy (polynomial) decay | Low | Deep networks; avoiding vanishing gradients. |
| **`linear`** | Rectangular | Zero (hard cutoff) | **Lowest** | Fast training; sparse updates. |
| **`gaussian`** | Bell-shaped | Exponential decay | Medium | Precision tuning; localized updates. |
| **`noisy`** | Stochastic Sigmoid | Exponential decay | Medium | Robustness; exploration during training. |

For most applications in **spikeDE**, the **`sigmoid`** surrogate with a scale of 5.0 provides an excellent trade-off between accuracy and convergence speed. 