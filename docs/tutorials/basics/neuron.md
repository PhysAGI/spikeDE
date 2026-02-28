# Neurons in spikeDE: Bridging Standard Dynamics and Fractional Calculus

In **spikeDE**, the neuron is reimagined. We move away from the traditional *step-by-step* update rule found in frameworks like [SpikingJelly](https://github.com/fangwei123456/spikingjelly) and towards a **continuous dynamical system** perspective. This shift allows us to seamlessly upgrade standard integer-order neurons into **Fractional-Order Spiking Neurons**, endowing them with infinite memory and complex temporal dynamics without rewriting your core logic.

This tutorial introduces the core neuron models available in **spikeDE**, focusing on how fractional-order dynamics extend traditional spiking neural networks (SNNs). We will explore the theoretical motivation behind fractional calculus in neuroscience, examine the mathematical formulations of Integrate-and-Fire (IF) and Leaky Integrate-and-Fire (LIF) neurons within the **spikeDE** framework, and guide you through configuring, using, and customizing these neurons for your own research.

---

## From Integer to Fractional Order

Traditional SNNs model neuronal membrane potential dynamics using **integer-order ordinary differential equations (ODEs)**. In these models, the rate of change of the membrane potential $v(t)$ at any instant depends solely on its current state and input, typically expressed as:

$$
\frac{dv(t)}{dt} = f(t, v(t), I_{\text{in}}(t))
$$

where $f(t, v, I_{\text{in}})$ represents the specific dynamics (e.g., leakage, input current). This formulation assumes the **Markov property**: the future state depends only on the present, with no memory of the past beyond the current value. While computationally efficient, this assumption limits the model's ability to capture complex temporal dependencies observed in biological neurons, such as long-range correlations and fractal dendritic structures.

**Fractional calculus** offers a powerful generalization. By replacing the integer-order derivative $\frac{d}{dt}$ with a **fractional-order derivative** $D^\alpha$ (where $0 < \alpha \leq 1$), we introduce **non-locality** and **memory** into the system:

$$
D^\alpha v(t) = f(t, v(t), I_{\text{in}}(t))
$$

The Caputo fractional derivative, commonly used in **spikeDE**, is defined as:

$$
D^\alpha v(t) = \frac{1}{\Gamma(1-\alpha)} \int_0^t (t-\tau)^{-\alpha} v'(\tau) d\tau
$$

Here, the current rate of change depends on a weighted integral of the entire history of the function $v(\tau)$, with weights decaying as a power law $(t-\tau)^{-\alpha}$. 

- When $\alpha = 1$, the model recovers the standard integer-order ODE (memoryless).
- When $0 < \alpha < 1$, the system exhibits **long-term memory**: past states influence the present, enabling richer temporal patterns and improved robustness to noise.

This transition from integer to fractional order allows **spikeDE** to model biological phenomena like spike-frequency adaptation and heavy-tailed relaxation processes that integer-order models cannot capture.

---

## Available Neurons in spikeDE

**spikeDE** currently provides two foundational neuron models, both extended to support fractional-order dynamics: the **Integrate-and-Fire (IF)** and **Leaky Integrate-and-Fire (LIF)** neurons. Below are their mathematical formulations.

### Fractional Integrate-and-Fire

The standard IF neuron integrates input current without leakage. Its fractional-order extension is governed by:

$$
\tau D^\alpha v(t) = I_{\text{in}}(t)
$$

where:

- $v(t)$ is the membrane potential.
- $I_{\text{in}}(t)$ is the input current.
- $\tau$ is the membrane time constant.
- $D^\alpha$ is the Caputo fractional derivative of order $\alpha$.

When $\alpha=1$, this reduces to the classic IF equation $\tau \frac{dv}{dt} = I_{\text{in}}(t)$.

### Fractional Leaky Integrate-and-Fire

The LIF neuron includes a leakage term that drives the potential toward zero. Its fractional dynamics are described by:

$$
\tau D^\alpha v(t) = -v(t) + I_{\text{in}}(t)
$$

Key properties:

- **Memory Effect**: The term $D^\alpha v(t)$ incorporates the history of $v(t)$, leading to power-law relaxation (Mittag-Leffler decay) instead of exponential decay.
- **Generalization**: Setting $\alpha=1$ recovers the standard LIF equation $\tau \frac{dv}{dt} = -v(t) + I_{\text{in}}(t)$.

Both models employ the same spiking mechanism: a spike $S(t)$ is emitted when $v(t)$ crosses a threshold $\theta$, followed by a reset (soft or hard). The fractional dynamics specifically govern the **charging phase** between spikes.

---

## Configuration and Usage

In **spikeDE**, neurons are implemented as stateless modules that compute the derivative $dv/dt$ (or the fractional equivalent) given the current membrane potential and input. The actual time integration and state management are handled by the `SNNWrapper` and its associated solvers.

### Basic Neuron Initialization

You can instantiate neurons directly from the **spikeDE** library. 

```python
from spikeDE import LIFNeuron, IFNeuron

# Initialize a standard LIF neuron
lif_neuron = LIFNeuron(
    tau=2.0, 
    threshold=1.0, 
    surrogate_opt='arctan',
    tau_learnable=False
)

# Initialize an IF neuron with learnable tau
if_neuron = IFNeuron(
    tau=1.5, 
    threshold=0.8, 
    surrogate_opt='sigmoid',
    tau_learnable=True
)
```

Key hyperparameters include:

- `tau`: Membrane time constant ($\tau$).
- `threshold`: Firing threshold ($\theta$).
- `surrogate_opt`: Surrogate gradient function for backpropagation (e.g., `'arctan'`, `'sigmoid'`).
- `tau_learnable`: Whether $\tau$ is a learnable parameter.

### Integrating with SNNWrapper

To leverage fractional-order dynamics, you must wrap your network using `SNNWrapper`. This wrapper handles the fractional integration (using methods like Grünwald-Letnikov) and manages the memory history required for $D^\alpha$.

```python
import torch.nn as nn
from spikeDE import SNNWrapper, LIFNeuron

class MySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.lif1 = LIFNeuron(tau=2.0)
        self.fc2 = nn.Linear(20, 5)
        self.lif2 = LIFNeuron(tau=2.0)

    def forward(self, x):
        x = self.lif1(self.fc1(x))
        x = self.lif2(self.fc2(x))
        return x

# Wrap the network with fractional dynamics
net = SNNWrapper(
    MySNN(),
    integrator='fdeint',      # Use fractional ODE solver
    alpha=0.7,                # Fractional order α (0 < α ≤ 1)
)

# Critical: Initialize shapes before training
net._set_neuron_shapes(input_shape=(1, 10))
```

**Key Hyperparameters in `SNNWrapper`:**

- `integrator`: Set to `'fdeint'` for fractional dynamics or `'odeint'` for integer-order.
- `alpha`: The fractional order $\alpha$. Can be a scalar (same for all layers) or a list (per-layer).

---

## Define Your Own Neuron

One of the core design principles of **spikeDE** is the separation of **neuron dynamics** from **state evolution**. 

- **Neuron Module**: Stateless. It accepts the current membrane potential $v_{mem}$ and input current, then returns the derivative $dv/dt$ and the spike signal. It does **not** store history or update state.
- **Solver (`SNNWrapper`)**: Stateful. It maintains the history of $v_{mem}$ required for fractional integration and updates the state over time using numerical schemes (e.g., Grünwald-Letnikov, Adams-Bashforth-Moulton).

This architecture allows you to easily define custom neurons by simply implementing the `forward` method to compute the derivative. To create a custom neuron, inherit from `BaseNeuron` and override the `forward` method. You must return a tuple `(dv_dt, spike)`.

```python
import torch
import torch.nn as nn
from spikeDE.neuron import BaseNeuron

class CustomNeuron(BaseNeuron):
    def forward(self, v_mem, current_input):
        if current_input is None:
            return v_mem
        
        tau = self.get_tau()
        
        # Define your custom dynamics here
        # Example: A quadratic leak term instead of linear
        dv_no_reset = (-v_mem**2 + current_input) / tau
        
        # Compute post-charge potential (for spike generation)
        v_post_charge = v_mem + dv_no_reset  # Assuming dt=1.0
        
        # Generate spike using surrogate gradient
        spike = self.surrogate_f(
            v_post_charge - self.threshold, 
            self.surrogate_grad_scale
        )
        
        # Compute final derivative including reset effect
        dv_dt = dv_no_reset - (spike.detach() * self.threshold) / tau
        
        return dv_dt, spike
```

### How Fractional Dynamics Are Applied

Once you define your custom neuron, `SNNWrapper` automatically applies fractional integration to it. You do **not** need to implement the fractional derivative logic inside the neuron itself.

When `integrator='fdeint'` is set in `SNNWrapper`:

1. The solver collects the history of $v_{mem}$ for each layer.
2. At each time step, it computes the fractional derivative approximation (e.g., using Grünwald-Letnikov coefficients) based on this history.
3. It combines this with the $dv/dt$ returned by your neuron to update the state.

This means your custom neuron works seamlessly with both integer-order (`odeint`) and fractional-order (`fdeint`) solvers without modification. The `SNNWrapper` acts as the bridge, injecting the memory effects required for fractional calculus while keeping the neuron definition clean and focused on instantaneous dynamics.

---

## Summary

In this tutorial, we explored how **spikeDE** extends traditional spiking neurons with fractional-order dynamics:

- **From Integer to Fractional**: We discussed how replacing $\frac{d}{dt}$ with $D^\alpha$ introduces memory and non-locality, enabling the modeling of complex temporal dependencies.
- **Available Neurons**: **spikeDE** supports ***f*-IF** and ***f*-LIF** neurons, which generalize their integer-order counterparts via fractional differential equations.
- **Configuration**: Neurons are configured with standard hyperparameters (`tau`, `threshold`), while fractional behavior is controlled via `SNNWrapper` arguments like `alpha` and `integrator`.
- **Customization**: By separating dynamics (neuron) from integration (solver), **spikeDE** allows users to define custom neurons easily. The `SNNWrapper` handles the complexity of fractional history management, ensuring your custom models benefit from fractional dynamics automatically.

With these tools, you can build powerful, biologically plausible SNNs capable of capturing long-range temporal correlations and achieving superior robustness in real-world tasks.
