# Per-Layer Alpha: Customizing Memory Dynamics in Fractional SNNs

In the realm of Fractional Spiking Neural Networks (*f*-SNNs), the fractional order $\alpha$ is not just a hyperparameter; it is the **control knob for memory**. It dictates how much historical information a neuron retains when computing its current state.

While setting a global $\alpha$ for the entire network is a good starting point, biological neural systems and complex temporal tasks often require heterogeneous dynamics. Some layers might need to act as short-term buffers (high $\alpha$, close to 1.0), while others function as long-term integrators (low $\alpha$, closer to 0.0).

The **Per-Layer Alpha** configuration in **spikeDE** empowers you to define these dynamics with surgical precision. You can assign unique fractional orders to each layer, enable multi-term distributed orders for complex time-scale modeling, and even make these orders **learnable** during training.

## The Physics of Alpha: Why Per-Layer Matters

Before diving into code, it's crucial to understand what changing $\alpha$ actually does to your network's behavior.

The membrane potential $V(t)$ in an *f*-SNN evolves according to the Caputo fractional derivative:

$$ D^{\alpha} V(t) = f(t, V(t)) $$

- **$\alpha \approx 1.0$ (Integer-like)**: The system has **short memory**. It behaves almost like a standard ODE, reacting quickly to recent inputs but forgetting the past rapidly. Ideal for layers detecting fast transients or edges.
- **$\alpha \ll 1.0$ (Strongly Fractional)**: The system has **long memory**. The current state is heavily influenced by the entire history of inputs due to the power-law kernel of the fractional derivative. Ideal for layers responsible for context accumulation, integration, or pattern recognition over long windows.

By configuring `alpha` per layer, you allow your network to automatically specialize: early layers might process rapid signal changes, while deeper layers integrate evidence over time.

## Configuration Modes

**spikeDE** supports four distinct modes for configuring fractional orders, ranging from simple global settings to highly complex, layer-specific distributed orders.

### Global Scalar
Every neuron layer in your network shares the exact same fractional order $\alpha$. This is equivalent to standard *f*-SNN behavior.

```python
# All layers use α = 0.6
net = SNNWrapper(
    my_model, 
    integrator='fdeint', 
    alpha=0.6, 
    learn_alpha=False
)
```

### Per-Layer Single-Term
Each layer $\ell$ is assigned its own specific order $\alpha_\ell$. This is the most common advanced use case, allowing you to manually tune the "memory depth" of each stage in your network.

$$ D^{\alpha_\ell} V_\ell(t) = f_\ell(t, V_\ell(t)) $$

**Usage:**
```python
# Layer 0: Strong memory (α=0.3)
# Layer 1: Weak memory (α=0.8)
net = SNNWrapper(
    my_model, 
    integrator='fdeint', 
    alpha=[0.3, 0.8],       # List length must match number of neuron layers
    alpha_mode='per_layer', # Crucial: Explicitly declare mode
    learn_alpha=True        # Optional: Let the network optimize these values
)
```

### Multi-Term Broadcast
Instead of a single derivative, each layer computes a weighted sum of multiple fractional derivatives. This allows a single layer to model dynamics across multiple time scales simultaneously. The same set of orders and weights is applied to **all** layers.

$$ \sum_{j} w_j D^{\alpha_j} V_\ell(t) = f_\ell(t, V_\ell(t)) $$

**Usage:**
```python
# All layers compute: 1.0*D^0.3 V + 0.5*D^0.5 V + 0.2*D^0.7 V
net = SNNWrapper(
    my_model, 
    integrator='fdeint', 
    alpha=[0.3, 0.5, 0.7],             # The set of orders
    multi_coefficient=[1.0, 0.5, 0.2], # Corresponding weights
    alpha_mode='multiterm',            # Crucial: Declare multi-term mode
    learn_coefficient=True             # Optional: Learn the weights w_j
)
```

### Per-Layer Multi-Term
This is the most powerful configuration. Each layer can have its own unique set of fractional orders and weighting coefficients. This effectively gives every layer its own custom differential equation structure.

**Usage:**
```python
net = SNNWrapper(
    my_model, 
    integrator='fdeint', 
    # Layer 0: 2-term, Layer 1: 3-term
    alpha=[
        [0.3, 0.5],           # Orders for Layer 0
        [0.4, 0.6, 0.8]       # Orders for Layer 1
    ],
    multi_coefficient=[
        [1.0, 0.5],           # Weights for Layer 0
        [1.0, 0.3, 0.1]       # Weights for Layer 1
    ],
    # Note: alpha_mode is auto-detected here due to nested list structure
)
```

## Disambiguating Configuration

A common point of confusion arises when passing a flat list to `alpha`. For a 2-layer network, does `alpha=[0.3, 0.7]` mean:

1. Layer 0 gets 0.3, Layer 1 gets 0.7?
2. Both layers get a 2-term equation with orders [0.3, 0.7]?

To prevent silent errors, **spikeDE** requires you to be explicit using the `alpha_mode` argument whenever ambiguity exists.

| Input Structure | `alpha_mode` Setting | Resulting Behavior |
| :--- | :--- | :--- |
| `0.5` (float) | (ignored) | **Global Scalar**: All layers $\alpha=0.5$. |
| `[0.3, 0.7]` | `'per_layer'` | **Per-Layer**: Layer 0: $\alpha_0$, Layer 1: $\alpha_1$. |
| `[0.3, 0.7]` | `'multiterm'` | **Broadcast Multi-Term**: All layers use terms $\{0.3, 0.7\}$. |
| `[[0.3], [0.7]]` | (auto) | **Per-Layer Multi-Term**: Detected automatically via nesting. |

!!! warning "Best Practice"
    Always explicitly set `alpha_mode='per_layer'` or `alpha_mode='multiterm'` when passing lists. Relying on heuristics (`alpha_mode='auto'`) may trigger warnings or unexpected behavior.

## Making Dynamics Learnable

One of the most compelling features of this framework is the ability to treat $\alpha$ and the multi-term coefficients as **trainable parameters**. Instead of manually searching for the optimal memory depth, you can let gradient descent find it.

Simply set the `learn_alpha` or `learn_coefficient` flags. These parameters are automatically registered with your PyTorch module and included in `model.parameters()`.

```python
net = SNNWrapper(
    my_model,
    integrator='fdeint',
    alpha=[0.5, 0.5],          # Initial guess
    alpha_mode='per_layer',
    learn_alpha=True,          # Enable gradient updates for alpha
    learn_coefficient=False
)

# Standard PyTorch optimization loop
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(net(batch.x, batch.t), batch.y)
    loss.backward()
    optimizer.step()

# Inspect learned values
print("Learned Alphas:", net.get_per_layer_alpha())
```

During training, the network might discover that early layers perform better with $\alpha \approx 0.9$ (fast reaction) while deeper layers converge to $\alpha \approx 0.4$ (strong integration), adapting the mathematical structure of the network to the data distribution.

## Summary

The **Per-Layer Alpha** configuration transforms your SNN from a static architecture into a **dynamically adaptable system**. By decoupling the memory characteristics of each layer, you unlock:

- **Specialization**: Layers can optimize for different temporal frequencies.
- **Expressivity**: Multi-term configurations capture complex, non-Markovian dynamics.
- **Automation**: Learnable $\alpha$ removes the burden of manual hyperparameter tuning for fractional orders.

Whether you are modeling biological plausibility or pushing the boundaries of temporal deep learning, per-layer control is the key to unlocking the full potential of fractional calculus in neural networks.