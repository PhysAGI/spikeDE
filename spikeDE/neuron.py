import torch
import torch.nn as nn
from .surrogate import (
    sigmoid_surrogate,
    arctan_surrogate,
    piecewise_linear_surrogate,
    gaussian_surrogate,
)
from typing import Optional, Tuple


surrogate_f = {
    "sigmoid_surrogate": sigmoid_surrogate,
    "arctan_surrogate": arctan_surrogate,
    "piecewise_linear_surrogate": piecewise_linear_surrogate,
    "gaussian_surrogate": gaussian_surrogate,
}


class BaseNeuron(nn.Module):
    r"""Base class for spiking neuron models with configurable membrane time constant and surrogate gradients.

    This abstract class provides the foundational structure for spiking neurons.
    It supports learnable or fixed membrane time constant ($\tau$) and
    customizable surrogate gradient functions for backpropagation through non-differentiable spikes.

    The effective membrane time constant is computed as:
    
    $$
    \tau =
    \begin{cases}
        \tau_0 \cdot (1 + e^{\theta}) & \text{if } \texttt{tau_learnable=True} \\
        \tau_0 & \text{otherwise}
    \end{cases}
    $$

    where $\tau_0$ is the initial value and $\theta$ is a learnable parameter.

    Subclasses must implement the `forward` method to define specific dynamics.

    Attributes:
        initial_tau (float): Initial value of the membrane time constant $\tau_0$.
        tau_param (Optional[torch.nn.parameter.Parameter]): Learnable parameter $\theta$ if `tau_learnable=True`;
            otherwise `None`.
        tau (float): Fixed $\tau$ used when `tau_learnable=False`.
        threshold (float): Firing threshold $V_{\text{th}}$.
        surrogate_grad_scale (float): Scaling factor for surrogate gradient steepness.
        surrogate_f (typing.Callable): Surrogate gradient function (e.g., arctan-based).
        tau_learnable (bool): Whether $\tau$ is trainable.
    """

    def __init__(
        self,
        tau: float = 0.5,
        threshold: float = 1.0,
        surrogate_grad_scale: float = 5.0,
        surrogate_opt: str = "arctan_surrogate",
        tau_learnable: bool = False,
    ) -> None:
        r"""Initializes the BaseNeuron module.

        Args:
            tau: The base membrane time constant $\tau$. Used directly if `tau_learnable=False`,
                or as a scaling factor if `tau_learnable=True`.
            threshold: The membrane potential threshold at which the neuron fires a spike.
            surrogate_grad_scale: Scaling factor applied inside the surrogate gradient function
                to control gradient magnitude during backpropagation.
            surrogate_opt: Name of the surrogate gradient function to use.
                Must be a key in the global `surrogate_f` dictionary (e.g., `"arctan_surrogate"`).
            tau_learnable: If `True`, $\tau$ becomes a learnable parameter.
                If `False`, $\tau$ remains fixed.
        """
        super(BaseNeuron, self).__init__()
        self.initial_tau = tau

        if tau_learnable:
            self.tau_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            self.tau_param = None
            self.tau = tau

        self.threshold = threshold
        self.surrogate_grad_scale = surrogate_grad_scale
        self.surrogate_f = surrogate_f[surrogate_opt]
        self.tau_learnable = tau_learnable

    def get_tau(self) -> float:
        r"""Returns the effective membrane time constant $\tau$.

        Ensures positivity via exponential reparameterization when learnable.

        Returns:
            Scalar tensor representing $\tau$.
        """
        if self.tau_learnable:
            return self.initial_tau * (1 + torch.exp(self.tau_param))
        else:
            return self.tau

    def forward(
        self, v_mem: torch.Tensor, current_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Performs one step of neuron state update.

        Must be overridden by subclasses to implement specific spiking dynamics.

        Args:
            v_mem: Membrane potential tensor of shape `(batch_size, ...)`.
            current_input: Input current tensor, same shape as `v_mem`.

        Returns:
            A tuple `(dv_dt, spike)` where:

                - `dv_dt`: Effective derivative of membrane potential.
                - `spike`: Continuous spike approximation in [0, 1].

        Raises:
            NotImplementedError: Always raised here; subclass must implement.
        """
        raise NotImplementedError("Neuron forward method must be overridden.")


class IFNeuron(BaseNeuron):
    r"""Integrate-and-Fire (IF) spiking neuron model with surrogate gradients.

    This model integrates input without leakage. The dynamics follow:

    $$
        \tau\frac{\text{d}v}{\text{d}t} = I(t), \quad
        \text{spike} = \sigma(v - V_{\text{th}})
    $$

    where $\sigma$ is a differentiable surrogate.

    Note:
        Despite inheriting `tau`, this model behaves as a pure integrator
        when leakage is disabled (i.e., no decay term on `v_mem`).
    """

    def forward(
        self, v_mem: torch.Tensor, current_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for IF neuron dynamics (discrete-time, `dt=1.0`).

        Args:
            v_mem: Current membrane potential.
            current_input: Input current (same shape as `v_mem`).

        Returns:
            Tuple `(dv_dt, spike)` representing effective derivative and spike output.
        """
        if current_input is None:
            return v_mem
        tau = self.get_tau()
        dt = 1.0
        dv_no_reset = (current_input) / tau
        v_post_charge = v_mem + dt * dv_no_reset
        spike = self.surrogate_f(
            v_post_charge - self.threshold, self.surrogate_grad_scale
        )
        dv_dt = dv_no_reset - (spike.detach() * self.threshold) / tau
        return dv_dt, spike


class LIFNeuron(BaseNeuron):
    r"""Leaky Integrate-and-Fire (LIF) spiking neuron model with surrogate gradients.

    Implements classic leaky dynamics governed by:

    $$
        \tau \frac{\text{d}v}{\text{d}t} = -v + I(t), \quad
        \text{spike} = \sigma(v - V_{\text{th}})
    $$

    where $\sigma$ is a differentiable surrogate.
    """

    def forward(
        self, v_mem: torch.Tensor, current_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for LIF neuron dynamics (discrete-time, `dt=1.0`).

        Args:
            v_mem: Current membrane potential.
            current_input: Input current (same shape as `v_mem`).

        Returns:
            Tuple `(dv_dt, spike)` representing effective derivative and spike output.
        """
        if current_input is None:
            return v_mem
        tau = self.get_tau()
        dt = 1.0
        dv_no_reset = (-v_mem + current_input) / tau
        v_post_charge = v_mem + dt * dv_no_reset
        spike = self.surrogate_f(
            v_post_charge - self.threshold, self.surrogate_grad_scale
        )
        dv_dt = dv_no_reset - (spike.detach() * self.threshold) / tau
        return dv_dt, spike


class LIFNeuron_OrderOne(BaseNeuron):
    def forward(
        self, v_mem: torch.Tensor, current_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if current_input is None:
            return v_mem

        tau = self.get_tau()
        dv_no_reset = (-v_mem + current_input) / tau
        v_post_charge = v_mem + dv_no_reset

        spike = self.surrogate_f(
            v_post_charge - self.threshold, self.surrogate_grad_scale
        )
        v_mem = v_post_charge - (spike.detach() * self.threshold)
        return v_mem, spike


class LIFNeuronFDE(BaseNeuron):
    r"""Leaky Integrate-and-Fire neuron with Fractional-Order Dynamics (FDE).

    This model extends the classic LIF neuron by incorporating fractional-order dynamics,
    which introduces memory effects into the membrane potential evolution.
    The governing equation is approximated in discrete time using a Grünwald–Letnikov (GL)
    scheme:

    $$
        \tau \frac{d^\beta v}{dt^\beta} = -v + I(t), \quad
        \text{spike} = \sigma(v - V_{\text{th}})
    $$

    where $\beta \in (0,1]$ controls the degree of memory (fractional order),
    and $\sigma$ is a differentiable surrogate for the spiking nonlinearity.

    Memory is implemented via a sliding window over past voltage states,
    with coefficients precomputed from the GL expansion.
    The effective update becomes:

    $$
        v^{(k+1)} = h^\beta \cdot \frac{-v^{(k)} + I^{(k)}}{\tau}
                     - \sum_{j=0}^{k-1} c_{k-j} v^{(j)}
    $$

    where $c_j$ are the reversed GL coefficients.

    Attributes:
        T (int): Total number of timesteps to simulate; used to precompute coefficient array.
        name (str): Optional identifier for debugging or logging.
        h (float): Time step size ($\Delta t$).
        beta (float): Fractional order parameter $\beta \in (0, 1]$.
        batchsize (int): Batch dimension expected during simulation.
        c_rev (torch.Tensor): Reversed Grünwald–Letnikov coefficients of length `T+1`.
        h_beta (torch.Tensor): Scalar tensor representing $h^\beta$, computed once.
        mem_hist (torch.Tensor): Buffer storing past membrane potentials of shape `(T, batchsize, ...)`.
        memory (Optional[int]): If not `None`, limits the history window to this many steps.
        k (int): Current timestep index (modulo handled externally).
    """

    def __init__(
        self,
        tau: float = 0.5,
        threshold: float = 1.0,
        surrogate_grad_scale: float = 5.0,
        surrogate_opt: str = "arctan_surrogate",
        tau_learnable: bool = False,
        method: str = "gl",
        beta: float = 0.5,
        batchsize: int = 32,
        T: int = 16,
        step_size: float = 1.0,
        memory: int = -1,
        name: str = "",
    ) -> None:
        r"""Initializes the LIFNeuronFDE module.

        Args:
            tau: Base membrane time constant $\tau$. Used directly if `tau_learnable=False`,
                or as scaling factor if learnable.
            threshold: Firing threshold $V_{\text{th}}$.
            surrogate_grad_scale: Scaling factor for surrogate gradient steepness.
            surrogate_opt: Name of surrogate function; must be key in global `surrogate_f`.
            tau_learnable: Whether $\tau$ is trainable via exponential reparameterization.
            method: Numerical method for fractional derivative (currently only `"gl"` supported).
            beta: Fractional order $\beta \in (0, 1]$. Lower values increase memory effect.
            batchsize: Expected batch dimension for internal buffer allocation.
            T: Maximum sequence length; determines size of coefficient array and history buffer.
            step_size: Discrete time step $h = \Delta t$.
            memory: If >= 0, restricts history window to `memory` steps; -1 means full history.
            name: Optional string identifier (useful for debugging multiple neurons).
        """
        super().__init__(
            tau=tau,
            threshold=threshold,
            surrogate_grad_scale=surrogate_grad_scale,
            surrogate_opt=surrogate_opt,
            tau_learnable=tau_learnable,
        )
        self.T = T
        self.name = name
        self.h = step_size
        self.beta = beta
        self.batchsize = batchsize

        self.c_rev = None
        self.h_beta = None

        self.mem_hist = None
        if memory == -1:
            self.memory = None
        else:
            self.memory = memory

        self.k = 0
        self._shape = None
        self._device = None
        self._dtype = None
        self._initialized = False

    def _reset(self) -> None:
        r"""Resets internal timestep counter and reinitializes memory buffer."""
        self.k = 0
        if self.mem_hist is not None:
            del self.mem_hist
        self.mem_hist = torch.empty(
            (self.T, self.batchsize) + self._shape[1:],
            device=self._device,
            dtype=self._dtype,
        )

    def _get_cj_(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        r"""Precomputes reversed Grünwald–Letnikov coefficients for fractional derivative.

        Computes $c_j = \prod_{i=1}^j \left(1 - \frac{1+\beta}{i}\right)$ for $j=0,\dots,T$,
        then reverses the array for efficient convolution-like access.

        Args:
            T: Number of coefficients to compute.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Reversed coefficient tensor of shape `(T+1,)`.
        """
        c = torch.empty(T + 1, device=device, dtype=dtype)
        c[0] = 1
        for j in range(1, T + 1):
            c[j] = (1 - (1 + self.beta) / j) * c[j - 1]
        c_rev = torch.flip(c, dims=[0])
        return c_rev

    def initialize(self, v_mem: torch.Tensor) -> torch.Tensor:
        r"""Lazy initialization of internal buffers based on input shape/device/dtype.

        Called automatically on first forward pass when `current_input is None`.

        Args:
            v_mem: Initial membrane potential tensor.

        Returns:
            Unmodified `v_mem` (enables chaining).
        """
        self._shape = v_mem.shape
        self._device = v_mem.device
        self._dtype = v_mem.dtype

        self.c_rev = self._get_cj_(self.T, self._device, self._dtype)
        self.h_beta = torch.pow(
            torch.tensor(self.h, device=self._device, dtype=self._dtype), self.beta
        )

        self.mem_hist = torch.empty(
            (self.T, self.batchsize) + self._shape[1:],
            device=self._device,
            dtype=self._dtype,
        )
        self._initialized = True
        return v_mem

    @torch.compile(mode="reduce-overhead")
    def _forward_compiled(
        self, v_mem: torch.Tensor, current_input: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compiled core computation for fractional LIF dynamics at timestep `k`.

        Separated to enable TorchDynamo compilation while keeping mutable state (`self.k`)
        outside the compiled region.

        Args:
            v_mem: Current membrane potential.
            current_input: Input current (same shape as `v_mem`).
            k: Current timestep index (used to slice history).

        Returns:
            Tuple `(v_mem_new, spike)` where:
                `v_mem_new`: Updated membrane potential after fractional update and reset;
                `spike`: Continuous spike approximation via surrogate gradient.
        """
        tau = self.get_tau()

        dv_no_reset = (-v_mem + current_input) / tau

        if self.memory is None:
            history_len = k
            start = 0
        else:
            start = max(0, k - self.memory)
            history_len = k - start

        if history_len > 0:
            w = self.c_rev[self.T - history_len : self.T].view(history_len, 1)
            win = self.mem_hist[start:k].reshape(history_len, -1)
            hist = torch.matmul(w.T, win).reshape(v_mem.shape)
        else:
            hist = torch.zeros_like(v_mem)

        v_post_charge = self.h_beta * dv_no_reset - hist

        spike = self.surrogate_f(
            v_post_charge - self.threshold, self.surrogate_grad_scale
        )

        v_mem_new = v_post_charge - (spike.detach() * self.threshold)

        return v_mem_new, spike

    def forward(
        self, v_mem: torch.Tensor, current_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass implementing fractional-order LIF dynamics.

        On first call with `current_input=None`, performs lazy initialization.
        Subsequent calls update the neuron state using fractional memory.

        Args:
            v_mem: Membrane potential tensor of shape `(batch_size, ...)`.
            current_input: Input current tensor of same shape; if `None`, triggers initialization.

        Returns:
            Tuple `(v_mem_new, spike)` where:
                `v_mem_new`: Updated membrane potential after fractional update and reset;
                `spike`: Continuous spike approximation via surrogate gradient.
        """
        if current_input is None:
            return self.initialize(v_mem)

        if not self._initialized:
            self.initialize(v_mem)

        v_mem_new, spike = self._forward_compiled(v_mem, current_input, self.k)

        self.mem_hist[self.k].copy_(v_mem_new)
        self.k = self.k + 1

        return v_mem_new, spike


class HardResetLIFNeuron(BaseNeuron):
    r"""Leaky Integrate-and-Fire (LIF) neuron with **approximate hard reset** via strong negative feedback.

    This model implements a continuous-time approximation of the classic *hard reset* mechanism:
    when a spike is emitted, the membrane potential $v$ is instantaneously clamped to zero (or baseline).
    Since true discontinuities are incompatible with gradient-based learning,
    this class approximates the reset using a large negative derivative impulse proportional to the current voltage.

    The dynamics are governed by:

    $$
        \tau \frac{dv}{dt} = -v + I(t) - \kappa \cdot s(t) \cdot v,
    $$

    where:

    - $s(t) = \sigma(v - V_{\text{th}})$ is a differentiable surrogate spike,
    - $\kappa > 0$ is the **reset strength** controlling how aggressively $v$ is driven toward zero upon spiking.

    In the limit $\kappa \to \infty$, this recovers the ideal hard reset: $v \leftarrow 0$ after a spike.

    Note:
        Unlike standard soft-reset LIF models that subtract $V_{\text{th}}$,
        this formulation drives $v$ toward **zero**, mimicking biological reset to resting potential.

    Attributes:
        reset_strength (float): Positive scalar $\kappa$ that scales the reset-induced decay.
            Larger values yield behavior closer to ideal hard reset.
            Typical values range from 10.0 to 100.0; default is 50.0.
    """

    def __init__(
        self,
        tau: float = 0.5,
        threshold: float = 1.0,
        surrogate_grad_scale: float = 5.0,
        reset_strength: float = 50.0,
    ) -> None:
        r"""Initializes the HardResetLIFNeuron module.

        Args:
            tau: Membrane time constant $\tau$. Can be fixed or learnable (via parent class).
            threshold: Firing threshold $V_{\text{th}}$.
            surrogate_grad_scale: Scaling factor for the steepness of the surrogate gradient.
            reset_strength: Strength $\kappa$ of the reset feedback term.
                Higher values enforce stronger suppression of $v$ post-spike,
                better approximating a true hard reset.
        """
        super(HardResetLIFNeuron, self).__init__(tau, threshold, surrogate_grad_scale)
        self.reset_strength = reset_strength

    def forward(
        self, v_mem: float, current_input: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass implementing LIF dynamics with hard-reset approximation.

        Args:
            v_mem: Current membrane potential tensor of shape `(batch_size, ...)`.
            current_input: Input current tensor of same shape as `v_mem`.
                If `None`, returns `v_mem` unchanged (used for initialization in some frameworks).

        Returns:
            Tuple `(dv_dt, spike)` representing effective derivative and spike output.

        Warning:
            As currently implemented, the `reset_dv_dt` term is **not added** to the returned `dv_dt`
            (it is computed but unused). This means the reset effect is **not applied during state update**
            unless an external solver explicitly incorporates it.
            For a fully functional hard-reset approximation, uncomment the line:
            ```python
            dv_dt = standard_dv_dt + reset_dv_dt
            ```
        """
        if current_input is None:
            return v_mem
        tau = self.get_tau()
        v_scaled = v_mem - self.threshold
        spike_out = self.surrogate_f(v_scaled, self.surrogate_grad_scale)
        standard_dv_dt = (-v_mem + current_input) / tau
        reset_dv_dt = -spike_out * self.reset_strength * v_mem / tau
        dv_dt = standard_dv_dt
        return dv_dt, spike_out


# --------------------------------------------------


class NoisyLIFNeuron_OrderOne(BaseNeuron):
    """
    Discrete-time noisy-threshold LIF neuron.

    Theory:
        S_t = H(v - θ + ξ),  ξ ~ Logistic(0, γ)

        Taking expectation:
        p(v) = E[S|v] = P(ξ ≥ θ-v) = σ((v-θ)/γ)

        Gradient (the key insight):
        p'(v) = f(θ-v) = (1/γ) · σ(z) · (1-σ(z))

        where f is the logistic PDF, replacing the Dirac delta.

    Training modes:
        - "meanfield":   S = σ(z)              Deterministic, E[spike]
        - "soft_noisy":  S = σ(z + g)          Stochastic, differentiable (NEW)
        - "hard_st":     H(z+g) fwd, σ(z) bwd  Binary spikes, smooth gradients
        - "concrete":    S = σ((z+g)/τ)        Gumbel-softmax style relaxation
        - "hard_concrete":  S = H(z+g) fwd   a strict Gumbel-softmax

    where z = (v-θ)/γ, g ~ Logistic(0,1).
    """

    def __init__(
        self,
        tau: float = 10.0,
        threshold: float = 1.0,
        gamma: float = 0.2,  # Noise scale γ (was surrogate_grad_scale)
        train_spike_mode: str = "hard_concrete",
        concrete_temp: float = 0.25,  # Temperature for concrete relaxation
        z_clip: float = 10.0,  # Numerical stability clamp
        hard_eval: bool = True,  # Use hard spikes at eval time
        detach_reset: bool = True,  # Whether to detach spike in reset
    ):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.gamma = gamma  # Clearer name than surrogate_grad_scale
        self.train_spike_mode = train_spike_mode
        self.concrete_temp = concrete_temp
        self.z_clip = z_clip
        self.hard_eval = hard_eval
        self.detach_reset = detach_reset

    def get_tau(self):
        return self.tau

    @staticmethod
    def _sample_logistic(shape, device, dtype, eps=1e-7):
        """Sample g ~ Logistic(0, 1) via inverse CDF."""
        u = torch.empty(shape, device=device, dtype=dtype).uniform_(eps, 1 - eps)
        return u.log() - (-u).log1p()  # Slightly faster than log(u) - log(1-u)

    def spike_fn(self, v_pre: torch.Tensor) -> torch.Tensor:
        """
        Compute spike from pre-reset membrane potential.

        This is the core noisy threshold implementation.
        """
        # Eval mode: hard threshold
        if not self.training and self.hard_eval:
            return (v_pre >= self.threshold).to(v_pre.dtype)

        # Normalized distance from threshold: z = (v - θ) / γ
        z = (v_pre - self.threshold) / self.gamma

        if self.z_clip is not None:
            z = z.clamp(-self.z_clip, self.z_clip)

        mode = self.train_spike_mode.lower()

        # === Mean-field: deterministic expected spike ===
        if mode == "meanfield":
            return torch.sigmoid(z)

        # === Sample logistic noise for stochastic modes ===
        g = self._sample_logistic(z.shape, z.device, z.dtype)

        # === Soft noisy: differentiable stochastic spike ===
        # This is the "true" noisy threshold with reparameterization
        if mode == "soft_noisy":
            return torch.sigmoid(z + g)

        # === Hard straight-through: binary forward, smooth backward ===
        # Most common choice - gets exact binary spikes but learns
        if mode == "hard_st":
            hard = ((z + g) >= 0).to(v_pre.dtype)
            soft = torch.sigmoid(z)
            return (hard - soft).detach() + soft

        # === Concrete / Gumbel-softmax relaxation ===
        if mode == "concrete":
            return torch.sigmoid((z + g) / self.concrete_temp)

        # === Hard Concrete (Straight-Through) ===
        # This matches gumbel_softmax(hard=True).
        # Forward: Binary (0 or 1) based on stochastic sampling
        # Backward: Gradient of the Sigmoid relaxation
        if mode == "hard_concrete":
            # 1. Compute soft relaxation
            y_soft = torch.sigmoid((z + g))

            # 2. Compute hard spike (using the same noise g is important for consistency)
            # Note: (z+g) > 0 is equivalent to sigmoid((z+g)/tau) > 0.5
            y_hard = ((z + g) >= 0).to(v_pre.dtype)

            # 3. Straight-Through trick
            return (y_hard - y_soft).detach() + y_soft

        raise ValueError(
            f"Unknown train_spike_mode='{self.train_spike_mode}'. "
            f"Options: meanfield, soft_noisy, hard_st, concrete, hard_noisy"
        )

    def forward(self, v_mem: torch.Tensor, current_input: torch.Tensor = None):
        """
        Single timestep LIF dynamics with noisy threshold.

        Args:
            v_mem: Membrane potential at time t
            current_input: Synaptic input current

        Returns:
            v_next: Membrane potential at time t+1 (after reset)
            spike: Spike output
        """
        if current_input is None:
            return v_mem

        tau = self.get_tau()

        # Integrate (Euler step with dt=1)
        dv = (-v_mem + current_input) / tau
        v_post_charge = v_mem + dv

        # Spike via noisy threshold
        spike = self.spike_fn(v_post_charge)

        # Reset: v_next = v - spike * (v - v_reset), assuming v_reset = 0
        # Simplifies to: v_next = v - spike * threshold (for reset-by-subtraction)
        if self.detach_reset:
            # Your original: gradient doesn't flow through reset
            v_next = v_post_charge - spike.detach() * self.threshold
        else:
            # Fully differentiable reset (theoretically correct per the math)
            v_next = v_post_charge - spike * self.threshold

        return v_next, spike


class NoisyLIFNeuron(BaseNeuron):
    """
    Discrete-time noisy-threshold LIF neuron.

    Theory:
        S_t = H(v - θ + ξ),  ξ ~ Logistic(0, γ)

        Taking expectation:
        p(v) = E[S|v] = P(ξ ≥ θ-v) = σ((v-θ)/γ)

        Gradient (the key insight):
        p'(v) = f(θ-v) = (1/γ) · σ(z) · (1-σ(z))

        where f is the logistic PDF, replacing the Dirac delta.

    Training modes:
        - "meanfield":   S = σ(z)              Deterministic, E[spike]
        - "soft_noisy":  S = σ(z + g)          Stochastic, differentiable (NEW)
        - "hard_st":     H(z+g) fwd, σ(z) bwd  Binary spikes, smooth gradients
        - "concrete":    S = σ((z+g)/τ)        Gumbel-softmax style relaxation
        - "hard_concrete":  S = H(z+g) fwd   a strict Gumbel-softmax

    where z = (v-θ)/γ, g ~ Logistic(0,1).
    """

    def __init__(
        self,
        tau: float = 10.0,
        threshold: float = 1.0,
        gamma: float = 0.2,  # Noise scale γ (was surrogate_grad_scale)
        train_spike_mode: str = "hard_concrete",
        concrete_temp: float = 0.25,  # Temperature for concrete relaxation
        z_clip: float = 10.0,  # Numerical stability clamp
        hard_eval: bool = True,  # Use hard spikes at eval time
        detach_reset: bool = True,  # Whether to detach spike in reset
    ):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.gamma = gamma  # Clearer name than surrogate_grad_scale
        self.train_spike_mode = train_spike_mode
        self.concrete_temp = concrete_temp
        self.z_clip = z_clip
        self.hard_eval = hard_eval
        self.detach_reset = detach_reset

    def get_tau(self):
        return self.tau

    @staticmethod
    def _sample_logistic(shape, device, dtype, eps=1e-7):
        """Sample g ~ Logistic(0, 1) via inverse CDF."""
        u = torch.empty(shape, device=device, dtype=dtype).uniform_(eps, 1 - eps)
        return u.log() - (-u).log1p()  # Slightly faster than log(u) - log(1-u)

    def spike_fn(self, v_pre: torch.Tensor) -> torch.Tensor:
        """
        Compute spike from pre-reset membrane potential.

        This is the core noisy threshold implementation.
        """
        # Eval mode: hard threshold
        if not self.training and self.hard_eval:
            return (v_pre >= self.threshold).to(v_pre.dtype)

        # Normalized distance from threshold: z = (v - θ) / γ
        z = (v_pre - self.threshold) / self.gamma

        if self.z_clip is not None:
            z = z.clamp(-self.z_clip, self.z_clip)

        mode = self.train_spike_mode.lower()

        # === Mean-field: deterministic expected spike ===
        if mode == "meanfield":
            return torch.sigmoid(z)

        # === Sample logistic noise for stochastic modes ===
        g = self._sample_logistic(z.shape, z.device, z.dtype)

        # === Soft noisy: differentiable stochastic spike ===
        # This is the "true" noisy threshold with reparameterization
        if mode == "soft_noisy":
            return torch.sigmoid(z + g)

        # === Hard straight-through: binary forward, smooth backward ===
        # Most common choice - gets exact binary spikes but learns
        if mode == "hard_st":
            hard = ((z + g) >= 0).to(v_pre.dtype)
            soft = torch.sigmoid(z)
            return (hard - soft).detach() + soft

        # === Concrete / Gumbel-softmax relaxation ===
        if mode == "concrete":
            return torch.sigmoid((z + g) / self.concrete_temp)

        # === Hard Concrete (Straight-Through) ===
        # This matches gumbel_softmax(hard=True).
        # Forward: Binary (0 or 1) based on stochastic sampling
        # Backward: Gradient of the Sigmoid relaxation
        if mode == "hard_concrete":
            # 1. Compute soft relaxation
            y_soft = torch.sigmoid((z + g))

            # 2. Compute hard spike (using the same noise g is important for consistency)
            # Note: (z+g) > 0 is equivalent to sigmoid((z+g)/tau) > 0.5
            y_hard = ((z + g) >= 0).to(v_pre.dtype)

            # 3. Straight-Through trick
            return (y_hard - y_soft).detach() + y_soft

        raise ValueError(
            f"Unknown train_spike_mode='{self.train_spike_mode}'. "
            f"Options: meanfield, soft_noisy, hard_st, concrete, hard_noisy"
        )

    def forward(self, v_mem: torch.Tensor, current_input: torch.Tensor = None):
        """
        Single timestep LIF dynamics with noisy threshold.

        Args:
            v_mem: Membrane potential at time t
            current_input: Synaptic input current

        Returns:
            v_next: Membrane potential at time t+1 (after reset)
            spike: Spike output
        """
        if current_input is None:
            return v_mem

        tau = self.get_tau()

        dt = 1.0  # 若外部 step_size 可变，这里用传入的 dt 这边我们强制dt=1.0进行膜电位预估
        # 1) 先按 LIF 漂移（不含 reset）
        dv_no_reset = (-v_mem + current_input) / tau  # + 1
        # print("*"*36,v_mem.shape, current_input.shape)
        v_post_charge = v_mem + dt * dv_no_reset  # fire-after-charge 的判阈值点
        # 2) 在充电后的电压上“开火”
        spike = self.spike_fn(v_post_charge)

        # 3) Reset: v_next = v - spike * (v - v_reset), assuming v_reset = 0
        # Simplifies to: v_next = v - spike * threshold (for reset-by-subtraction)
        if self.detach_reset:
            # Your original: gradient doesn't flow through reset
            dv_dt = dv_no_reset - spike.detach() * self.threshold
        else:
            # Fully differentiable reset (theoretically correct per the math)
            dv_dt = dv_no_reset - spike * self.threshold

        return dv_dt, spike
