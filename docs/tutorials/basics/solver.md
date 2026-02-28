# Solvers in spikeDE: Powering Temporal Memory

In **spikeDE**, the solver is the engine that transforms standard neural dynamics into fractional-order systems. While your neuron defines the *local* instantaneous behavior (the derivative $f(t, v)$), the solver handles the *global* time evolution, enforcing the power-law memory kernel that characterizes fractional calculus.

This guide details the numerical methods available in spikeDE, explains how to configure them for single-term or multi-term fractional orders, and demonstrates how to balance accuracy with computational efficiency using memory truncation.

---

## The Solver's Role: From Local Derivative to Global History

Traditional SNN frameworks typically employ simple Euler integration, where the state at time $t$ depends solely on the state at $t-\Delta t$. This Markovian property limits the network's ability to retain long-term temporal context.

spikeDE replaces this with rigorous Fractional Differential Equation (FDE) solvers. Mathematically, given a neuron defined by:

$$ D^\alpha v(t) = f(t, v(t)) $$

The solver computes the state $v(t_k)$ not just from the immediate past, but as a weighted convolution of the entire history:

$$ v(t_k) = v(0) + \frac{1}{\Gamma(\alpha)} \int_0^{t_k} (t_k - \tau)^{\alpha-1} f(\tau, v(\tau)) d\tau $$

**Key Responsibilities of the spikeDE Solver:**

1. **History Management:** Automatically maintains the buffer of past states $[v_0, \dots, v_{k-1}]$.
2. **Weight Computation:** Calculates the specific coefficients (e.g., Grünwald-Letnikov, L1, Adams-Bashforth) based on the fractional order $\alpha$.
3. **Convolution:** Performs the weighted sum efficiently at every time step.
4. **Differentiation:** Supports backpropagation through the integration steps (via Adjoint methods or direct unrolling), allowing $\alpha$ and network weights to be learned jointly.

You do not need to implement these complex summations. You simply select a `method` and set `alpha`; the solver handles the rest.

---

## Available Numerical Methods

spikeDE implements four distinct numerical schemes, each rooted in a specific discretization strategy for fractional operators. Understanding their mathematical basis helps in selecting the right tool for your theoretical framework and accuracy requirements.

### Grünwald-Letnikov

The Grünwald-Letnikov (GL) definition is often considered the most natural discrete analog of the fractional derivative. It derives directly from the limit definition of the derivative extended to non-integer orders.

Mathematically, the $\alpha$-th order derivative is approximated by a finite difference scheme utilizing **generalized binomial coefficients**:

$$ D^\alpha v(t) \approx \frac{1}{h^\alpha} \sum_{j=0}^{k} (-1)^j \binom{\alpha}{j} v(t - jh) $$

where $\binom{\alpha}{j} = \frac{\Gamma(\alpha+1)}{j!\Gamma(\alpha-j+1)}$.

- **Why it matters:** The coefficients $(-1)^j \binom{\alpha}{j}$ decay algebraically, naturally encoding the power-law memory without requiring complex integral approximations.
- **Multi-Term Capability:** Because the GL definition is linear, it seamlessly extends to multi-term equations ($\sum c_i D^{\alpha_i} v = f$) by simply summing the weighted histories of different orders. This makes it the only native solver in spikeDE for multi-term dynamics.
- **Convergence:** It offers first-order accuracy $O(h)$, which is sufficient for most deep learning applications where stochastic gradient noise dominates numerical error.

### Product Trapezoidal

The Product Trapezoidal method reformulates the FDE as a Volterra integral equation of the second kind and applies the trapezoidal rule to approximate the integral term.

Instead of differencing the state $v$, it integrates the function $f(t, v)$:

$$ v(t_k) = v(0) + \frac{1}{\Gamma(\alpha)} \int_0^{t_k} (t_k - \tau)^{\alpha-1} f(\tau, v(\tau)) d\tau $$

The integral is discretized by piecewise linear interpolation of $f$, leading to weights that involve terms like $(j+1)^{\alpha+1} - 2j^{\alpha+1} + (j-1)^{\alpha+1}$.

- **Why it matters:** By integrating rather than differencing, this method achieves **second-order accuracy** $O(h^2)$ for smooth functions. It effectively smooths out high-frequency oscillations that might arise in the GL scheme.
- **Limitation:** The derivation assumes a single fractional order $\alpha$ for the kernel $(t-\tau)^{\alpha-1}$. Consequently, it cannot natively resolve multi-term sums with distinct exponents and will fallback to `gl` if such a configuration is detected.

### L1 Scheme

The L1 scheme is specifically tailored for the **Caputo derivative**, which is preferred in physical modeling because it allows for standard initial conditions ($v(0)=v_0$) rather than fractional ones.

It approximates the Caputo derivative by assuming the function $v(t)$ is piecewise linear over each time interval $[t_j, t_{j+1}]$. The resulting discretization uses weights derived from the integral of the slope:

$$ D^\alpha_C v(t_k) \approx \frac{1}{\Gamma(2-\alpha) h^\alpha} \sum_{j=0}^{k-1} b_j (v_{k-j} - v_{k-j-1}) $$

where $b_j = (j+1)^{1-\alpha} - j^{1-\alpha}$.

- **Why it matters:** The L1 scheme provides an accuracy of $O(h^{2-\alpha})$, which is superior to GL when $\alpha$ is close to 1. It is the standard choice for problems where the physical interpretation of the initial state is critical.
- **Limitation:** Like the trapezoidal method, the standard L1 formulation is derived for a single order $\alpha$. Multi-term configurations trigger an automatic fallback to the `gl` solver.

### Adams-Bashforth Predictor

This method utilizes the explicit Adams-Bashforth formulation applied to the equivalent Volterra integral equation. Unlike the previous methods which may implicitly depend on the current state (requiring iteration for implicit schemes), this is a purely **explicit predictor**.

It stores the history of the **function evaluations** $f_j = f(t_j, v_j)$ rather than the states $v_j$ themselves:

$$ v_{k+1} = v_0 + \frac{h^\alpha}{\Gamma(\alpha+2)} \left( f_{k+1}^P + \sum_{j=0}^{k} a_{j,k} f_j \right) $$

!!! note
    In the explicit predictor variant used here, $f_{k+1}$ is estimated or omitted depending on the specific variant implementation.

- **Why it matters:** It decouples the history storage from the state values, storing only the derivatives. This can be advantageous for specific stiff systems or when the function $f$ is computationally cheaper to store than the full state vector in certain architectures.
- **Usage:** Primarily useful for fast, explicit stepping in Caputo-based systems where stability constraints are manageable.

---

## Configuration and Usage

Configuring the solver is done via the `SNNWrapper`. You can specify the method globally or let the system auto-select based on your $\alpha$ configuration.

### Basic Single-Term Configuration

For standard fractional neurons where each layer has a single $\alpha$:

```python
from spikeDE import SNNWrapper

# Wrap your base model
net = SNNWrapper(
    base=my_model,
    alpha=0.85,          # Fractional order (0 < alpha <= 1)
    method='gl',         # Options: 'gl', 'trap', 'l1', 'pred'
    memory=-1            # Use full history (-1 or None)
)
```

### Multi-Term Fractional Orders

spikeDE uniquely supports **multi-term** fractional derivatives, where a layer's dynamics are governed by a sum of fractional operators:

$$ \sum_{i} c_i D^{\alpha_i} v(t) = f(t, v) $$

To enable this, pass a list of orders (and optional coefficients) to `alpha`. The solver will automatically switch to the robust **Multi-Term GL** backend.

```python
# Define multi-term alpha for a specific layer
# Format: [alpha_1, alpha_2, ...]
layer_alpha = [0.3, 0.7] 
coefficients = [1.0, 0.5] # Optional weights c_i

net = SNNWrapper(
    base=my_model,
    per_layer_alpha=[layer_alpha], # List of alphas per integrated layer
    per_layer_coefficient=[coefficients],
    method='glmulti' # multi-term support
)
```

!!! note "Automatic Fallback"
    If you request `method='trap'` or `method='l1'` but provide a multi-term $\alpha$, spikeDE will issue a warning and automatically switch to the `gl` (Grünwald-Letnikov) solver, as it is the only method currently supporting multi-term formulations.

---

## Managing Memory Complexity

A defining feature of fractional calculus is **infinite memory**. However, storing the entire history $[v_0, \dots, v_k]$ becomes computationally expensive ($O(N^2)$ complexity) for long time sequences.

spikeDE addresses this with **Short-Memory Principle** truncation.

### The `memory` Parameter

You can limit the history length used in the convolution sum by setting the `memory` argument.

```python
net = SNNWrapper(
    base=my_model,
    alpha=0.7,
    method='gl',
    memory=50  # Only look back 50 time steps
)
```

- `memory=None` or `-1`: **Full Memory**. Uses all historical steps since $t=0$. Highest accuracy, highest cost.
- `memory=M` (int): **Truncated Memory**. Only the last $M$ time steps are considered. The convolution sum becomes:
    
    $$ v_k \approx \text{Initial Terms} + \sum_{j=k-M}^{k} w_j f(t_j, v_j) $$

!!! tip "Choosing Memory Length"
    For many physical and biological systems, the power-law kernel $(t-\tau)^{\alpha-1}$ decays sufficiently fast that old history contributes negligibly. A memory length of 50–200 steps often provides an excellent trade-off between speed and accuracy for inference. For training or highly sensitive chaotic systems, consider using full memory.

---

## Summary

| Method | Type | Formulation | Multi-Term Support | Accuracy |
| :--- | :--- | :--- | :---: | :---: |
| **`gl`** | Explicit | Riemann-Liouville | :material-checkbox-marked-outline: **Yes** | $O(h)$ |
| **`trap`** | Implicit-like | Riemann-Liouville | :material-checkbox-blank-off-outline: (Fallback) | $O(h^2)$ |
| **`l1`** | Explicit | Caputo | :material-checkbox-blank-off-outline: (Fallback) | $O(h^{2-\alpha})$ |
| **`pred`** | Explicit | Caputo | :material-checkbox-blank-off-outline: (Fallback) | $O(h)$ |

For most applications in Spiking Neural Networks, **`gl`** is recommended due to its stability, support for learnable multi-term $\alpha$, and efficient implementation. Use `trap` or `l1` if you require higher precision for specific scientific modeling tasks and are working with single-term orders.
