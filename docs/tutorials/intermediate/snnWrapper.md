# SNNWrapper: Building Continuous-Time Spiking Neural Networks

In the **spikeDE** framework, the `SNNWrapper` is the central orchestrator that transforms your standard, discrete-time PyTorch Spiking Neural Network (SNN) into a continuous-depth dynamical system. 

While other components handle specific roles—**Neurons** define local dynamics, **Solvers** handle numerical integration, and **ODEFunc** rewires the graph—the `SNNWrapper` brings them all together. It manages the lifecycle of the simulation, from inferring network architecture to configuring fractional memory dynamics and executing the time-stepping loop.

This tutorial guides you through the philosophy, configuration, and usage of `SNNWrapper`, helping you unlock adaptive step-size solving, fractional-order calculus, and precise temporal modeling with minimal code changes.

---

## The Core Philosophy: Separation of Concerns

Traditional SNN frameworks often couple the neuron's state update logic directly with the time-stepping loop. This makes it difficult to swap integration methods or introduce complex dynamics like fractional derivatives.

`SNNWrapper` adopts a modular architecture based on three distinct responsibilities:

1.  **Architecture Inference**: Automatically detecting neuron layers and output shapes without manual specification.
2.  **Dynamics Configuration**: Managing fractional orders ($\alpha$) and memory coefficients per layer.
3.  **Execution Engine**: Delegating the actual integration to specialized solvers while handling data interpolation and boundary routing.

This separation allows you to write standard PyTorch `nn.Module` code while gaining access to advanced continuous-time features.

---

## Quick Start

Wrapping your existing model is straightforward. The only requirement is that your network uses neurons compatible with spikeDE.

```python
import torch.nn as nn
from spikeDE import SNNWrapper, LIFNeuron

# 1. Define your standard SNN
class MySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.neuron = LIFNeuron(tau=2.0)
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        # Standard discrete forward pass
        x = self.neuron(self.conv(x))
        return self.fc(x)

model = MySNN()

# 2. Wrap with SNNWrapper
net = SNNWrapper(
    base=model,
    integrator='fdeint',   # Use fractional solver
    alpha=0.8,             # Global fractional order
    method='gl'            # Grünwald-Letnikov scheme
)

# 3. Initialize shapes (Critical Step)
# This triggers architecture inference
net._set_neuron_shapes(input_shape=(1, 3, 32, 32))

# 4. Run simulation
# Input shape: [Time_Steps, Batch, Channels, Height, Width]
x_time = torch.linspace(0, 1, 100)
x_input = torch.randn(100, 1, 3, 32, 32)

output = net(x_input, x_time)
```

!!! warning "Remember to Initialize the Neuron Shape"
    The call to `_set_neuron_shapes` is mandatory before the first forward pass. It performs a dry run to detect the number of neuron layers and their output dimensions, which are required to initialize the fractional memory states.

---

## Key Features

### Automatic Architecture Inference

You do not need to manually count neuron layers or specify tensor shapes. `SNNWrapper` uses PyTorch FX to trace your model graph:

- **Neuron Detection**: Identifies all instances of `BaseNeuron` in your network.
- **Shape Recording**: Runs a dummy forward pass to record the exact output shape of each neuron layer.
- **Boundary Detection**: Intelligently splits the graph into the **ODE part** and the **Post-Neuron part**, handling complex cases like skip connections automatically.

### Flexible Fractional Configuration

`SNNWrapper` supports heterogeneous memory dynamics, allowing you to configure the fractional order $\alpha$ globally, per layer, or as a multi-term sum. For detailed configuration guidelines and mode explanations, please refer to [**Per-Layer Alpha: Customizing Memory Dynamics in Fractional SNNs**](./per_layer_alpha.md).

---

## Internal Workflow

Understanding the data flow helps in debugging and advanced customization.

### Initialization

Before any simulation runs, the wrapper must understand the network's geometry.

1. **Tracing**: It traces the `base` model using FX.
2. **Dry Run**: Executes a forward pass with dummy data (zeros) to record:
    - `neuron_shapes`: Output dimensions of every neuron layer.
    - `boundary_shapes`: Dimensions of tensors passed to the post-processing module.
3. **Parameter Registration**: Based on the detected layer count, it parses your `alpha` configuration and registers $\alpha$ and coefficients as `nn.Parameter` objects if `learn_alpha=True`.
4.  **Compilation**: Optionally compiles the ODE function using `torch.compile` for accelerated inference.

### Forward Pass

During the actual simulation:

1. **Input Interpolation**: The input $x(t)$ is reconstructed from discrete samples using the specified method (linear, cubic, etc.) to allow evaluation at arbitrary time points $t$.
2. **State Initialization**: Membrane potentials $v_{mem}$ and boundary states are initialized to zero (or custom values).
3. **Integration Loop**: The selected solver (e.g., `gl_integrate_tuple`) iterates through time steps:
    - Calls the ODE function to compute derivatives $(dv/dt)$.
    - Updates states using the fractional history buffer.
4. **Boundary Processing**: After integration completes, the final boundary outputs (spikes) are stacked and passed through the `post_neuron_module` (e.g., a classification head) to produce the final result.

---

## Advanced Usage

### Custom Interpolation
By default, inputs are interpolated linearly. For smoother sensory data (like audio), you can use cubic splines:
```python
net = SNNWrapper(model, interpolation_method='cubic')
```
Supported methods: `'linear'`, `'nearest'`, `'cubic'`, `'akima'`.

### Solver and Method Selection

You can seamlessly switch between integer-order and fractional-order dynamics by selecting the appropriate `integrator`. For fractional solving, `SNNWrapper` supports various discretization schemes via the `method` argument.

```python
# Integer-order ODE solving
net_ode = SNNWrapper(model, integrator='odeint', method='euler')

# Fractional solving with Grünwald-Letnikov scheme
net_gl = SNNWrapper(model, integrator='fdeint', method='gl')

# Fractional solving with Predictor-Corrector scheme
net_pred = SNNWrapper(model, integrator='fdeint', method='pred')

# Fractional solving with L1 scheme
net_l1 = SNNWrapper(model, integrator='fdeint', method='l1')
```

Common `method` options include `'euler'` (exclusive to `'odeint'`), `'gl'`, `'pred'`, `'l1'`, and `'trap'`. For a deep dive into the mathematical foundations and performance characteristics of each solver, please refer to [**Solvers in spikeDE: Powering Temporal Memory**](../basics/solver.md).

### Memory Truncation
Fractional calculus theoretically requires infinite history. For long sequences, you can truncate the memory to improve speed:
```python
net = SNNWrapper(
    model, 
    alpha=0.7, 
    method='gl',
    options={'memory': 50}  # Only look back 50 steps
)
```

---

## Summary

The `SNNWrapper` is the bridge between static PyTorch definitions and dynamic continuous-time simulations. By automating architecture inference, managing complex fractional configurations, and delegating heavy lifting to optimized solvers, it allows you to focus on designing neural architectures rather than wrestling with differential equation solvers.

Whether you need simple integer-order ODEs or complex, learnable, multi-term fractional dynamics, `SNNWrapper` provides a unified, pythonic interface to power your next-generation Spiking Neural Networks.