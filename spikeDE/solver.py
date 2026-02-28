import torch.nn as nn
import torch
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Tuple, List, Union


def euler_integrate_tuple(
    ode_func: Callable[
        [torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]
    ],
    y0_tuple: Tuple[torch.Tensor, ...],
    t_grid: torch.Tensor,
    neuron_count: int,
) -> List[List[torch.Tensor]]:
    r"""
    Performs standard explicit Euler integration for integer-order ODEs ($D^1 y = f(t, y)$).

    This function distinguishes between dynamic state variables (neurons) which are integrated,
    and boundary outputs (e.g., spike outputs) which are treated as pass-through values computed
    directly from the derivative without accumulation.

    The update rule for integrated components is:

    $$y_{k+1} = y_k + \Delta t \cdot f(t_k, y_k)$$

    Args:
        ode_func: A callable `f(t, y_tuple)` returning a tuple of derivatives.
                  Expected format: `(dy_1, ..., dy_N, boundary_1, ...)`.
        y0_tuple: A tuple of initial state tensors `(y_1, ..., y_N, boundary_1, ...)`.
        t_grid: A 1D tensor of time points `[t_0, t_1, ..., t_N]`. Step sizes can be non-uniform.
        neuron_count: The number of components in the state tuple representing dynamic neurons
                      to be integrated. Components beyond this index are treated as pass-through boundaries.

    Returns:
        A list of lists, where `history[i][k]` is the state of component `i` at time step `k+1`.
        The length of each inner list is `len(t_grid) - 1`.

    Raises:
        AssertionError: If `y0_tuple` is not a tuple or `t_grid` has fewer than 2 points.
    """
    assert isinstance(y0_tuple, tuple)
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    n_integrate = neuron_count
    # n_integrate is the number of neurons,
    # i.e., length of (dv1/dt, dv2/dt, ..., dvN/dt)
    n_components = len(y0_tuple)
    # length of (dv1/dt, dv2/dt, ..., dvN/dt, boundary_1, boundary_2, ...)

    # Initialize history lists for each component
    y_current = list(y0_tuple)
    y_history = [[] for _ in y0_tuple]

    # Euler integration: y_{k+1} = y_k + dt * f(t_k, y_k)
    for k in range(N - 1):
        tk = t_grid[k]
        dt = t_grid[k + 1] - t_grid[k]  # Scalar tensor, will broadcast automatically
        dy = ode_func(
            tk, tuple(y_current)
        )  # Expect tuple return, consistent with y structure
        # assert isinstance(dy, tuple) and len(dy) == len(y)

        # Update all integrated components except the last one
        for i in range(n_integrate):
            y_current[i] = y_current[i] + dt * dy[i]
            y_history[i].append(y_current[i])

        # Pass-through boundary output e.g. final spike output
        # See odefunc_fx.md
        for i in range(n_integrate, n_components):
            y_current[i] = dy[i]
            y_history[i].append(y_current[i])

    return y_history


"""
Unified Fractional Differential Equation Solvers for Spiking Neural Networks
with Per-Layer Alpha Support.

Each layer can have its own fractional order (alpha), which can be:
- Single-term: a scalar float
- Multi-term: a list/tensor of floats

Caputo formulation:
- pred_integrate_tuple (Adams-Bashforth predictor)
- l1_integrate_tuple (L1 scheme)

Riemann-Liouville formulation:
- gl_integrate_tuple (Grünwald-Letnikov)
- trap_integrate_tuple (Product Trapezoidal)
- glmethod_multiterm_integrate_tuple (Multi-term GL)
"""


@dataclass
class PerLayerAlphaInfo:
    r"""
    Metadata container for the fractional order ($\alpha$) configuration of a single layer.

    Stores the fractional order(s) and precomputed constants required for numerical integration.
    To ensure gradient flow during backpropagation when $\alpha$ is learnable, all values are
    stored as `torch.Tensor` objects rather than Python floats.

    Attributes:
        alpha: A tensor containing the fractional order(s).
               Shape `(1,)` for single-term, shape `(M,)` for multi-term with $M$ terms.
        is_multi_term: Boolean flag indicating if the layer has multiple fractional terms ($M > 1$).
        coefficient: Optional tensor of coefficients $[c_1, ..., c_M]$ for multi-term equations.
                     Defaults to ones if not provided.
        h_alpha: Precomputed $h^\alpha$ (Single-term only).
        h_alpha_gamma: Precomputed $h^\alpha \cdot \Gamma(2-\alpha)$ (Single-term only).
        h_alpha_over_alpha_gamma: Precomputed $h^\alpha / (\alpha \cdot \Gamma(\alpha))$ (Single-term only).
    """

    alpha: (
        torch.Tensor
    )  # Always tensor: 1-element for single-term, n-element for multi-term
    is_multi_term: bool
    coefficient: Optional[torch.Tensor] = None  # Coefficients (defaults to ones)

    # Precomputed constants for single-term (populated during config creation)
    h_alpha: Optional[torch.Tensor] = None
    h_alpha_gamma: Optional[torch.Tensor] = None
    h_alpha_over_alpha_gamma: Optional[torch.Tensor] = None


@dataclass
class SNNSolverConfig:
    r"""
    Central configuration object for SNN fractional solvers.

    Aggregates simulation parameters, device information, and per-layer fractional metadata
    to streamline the solver execution loop.

    Attributes:
        N: Number of time points in the grid.
        h: Step size tensor (scalar), assumed uniform $h = t_{k+1} - t_k$.
        device: Torch device for computation.
        dtype: Torch data type for computation.
        n_components: Total number of state components (neurons + boundaries).
        n_integrate: Number of components to integrate (excludes boundary outputs).
        per_layer_info: List of `PerLayerAlphaInfo` objects, one per integrated layer.
    """

    N: int  # Number of time points
    h: torch.Tensor  # Step size
    device: torch.device
    dtype: torch.dtype
    n_components: int  # Total number of state components
    n_integrate: int  # Number of components to integrate (excludes spike)

    # Per-layer alpha information
    per_layer_info: List[PerLayerAlphaInfo] = field(default_factory=list)

    @classmethod
    def from_inputs(
        cls,
        y0_tuple: Tuple[torch.Tensor, ...],
        per_layer_alpha: List[Any],
        t_grid: torch.Tensor,
        per_layer_coefficient: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> "SNNSolverConfig":
        r"""
        Constructs a solver configuration from user inputs.

        Processes raw alpha inputs (scalars, lists, or tensors) into standardized
        `PerLayerAlphaInfo` objects. Precomputes constants involving the Gamma function
        for single-term solvers to optimize the main integration loop.

        Args:
            y0_tuple: Tuple of initial state tensors. Used to infer device and dtype.
            per_layer_alpha: List of alpha values. Each element can be:

                - `float`: Single-term scalar.
                - `torch.Tensor`: 1-element (single-term) or M-element (multi-term).
                - `list`: Converted to tensor.
            t_grid: Time grid tensor.
            per_layer_coefficient: Optional list of coefficient tensors for multi-term layers.

        Returns:
            A configured `SNNSolverConfig` instance.

        Raises:
            AssertionError: If `t_grid` has fewer than 2 points or coefficient dimensions mismatch alpha.
        """
        device = y0_tuple[0].device
        dtype = y0_tuple[0].dtype

        N = len(t_grid)
        assert N >= 2, "t_grid must have at least 2 points"

        t_grid = t_grid.to(device=device, dtype=dtype)
        h = t_grid[-1] - t_grid[-2]

        n_components = len(y0_tuple)
        # length of (dv1/dt, dv2/dt, ..., dvN/dt, boundary_1, boundary_2, ...)

        n_integrate = len(per_layer_alpha)
        # n_integrate is the number of neurons,
        # i.e., length of (dv1/dt, dv2/dt, ..., dvN/dt)

        # Build per-layer info
        per_layer_info = []
        for i, alpha in enumerate(per_layer_alpha):
            # Determine if this layer is multi-term based on number of elements
            if isinstance(alpha, torch.Tensor):
                is_multi_term = alpha.numel() > 1
                alpha_tensor = alpha.to(device=device, dtype=dtype)
            elif isinstance(alpha, (list, tuple)):
                is_multi_term = len(alpha) > 1
                alpha_tensor = torch.tensor(alpha, dtype=dtype, device=device)
            else:
                # Scalar float
                is_multi_term = False
                alpha_tensor = torch.tensor([alpha], dtype=dtype, device=device)

            # Get coefficient for this layer
            coeff = None
            if per_layer_coefficient is not None and i < len(per_layer_coefficient):
                coeff = per_layer_coefficient[i]
                if coeff is not None and not isinstance(coeff, torch.Tensor):
                    coeff = torch.tensor(coeff, dtype=dtype, device=device)
                elif coeff is not None:
                    coeff = coeff.to(device=device, dtype=dtype)

                    # Now compare dimensions when coeff exists
                assert (
                    coeff.numel() == alpha_tensor.numel()
                ), f"Coefficient tensor size mismatch: {coeff.numel()} vs {alpha_tensor.numel()}"

            # Default coefficient to ones if not provided (for both single and multi-term)
            if coeff is None:
                coeff = torch.ones(alpha_tensor.numel(), dtype=dtype, device=device)

            # For single-term, precompute constants (used by non-multiterm solvers)
            if not is_multi_term:
                alpha_val = (
                    alpha_tensor.squeeze()
                )  # Keep as 0-dim tensor for gradient flow
                h_alpha = torch.pow(h, alpha_val)
                gamma_2_minus_alpha = math.gamma(
                    2 - alpha_val.item()
                )  # gamma needs float
                gamma_alpha = math.gamma(alpha_val.item())

                info = PerLayerAlphaInfo(
                    alpha=alpha_tensor,  # Keep as tensor for gradient flow
                    is_multi_term=False,
                    coefficient=coeff,
                    h_alpha=h_alpha,
                    h_alpha_gamma=h_alpha * gamma_2_minus_alpha,
                    h_alpha_over_alpha_gamma=h_alpha / (alpha_val * gamma_alpha),
                )
            else:
                info = PerLayerAlphaInfo(
                    alpha=alpha_tensor,
                    is_multi_term=True,
                    coefficient=coeff,
                )

            per_layer_info.append(info)

        return cls(
            N=N,
            h=h,
            device=device,
            dtype=dtype,
            n_components=n_components,
            n_integrate=n_integrate,
            per_layer_info=per_layer_info,
        )


def get_memory_bounds(k: int, memory: Optional[int]) -> Tuple[int, int]:
    r"""
    Calculates the range of history indices to include in the convolution sum.

    Supports memory truncation for long sequences to reduce computational complexity
    from $O(N^2)$ to $O(N \cdot M)$, where $M$ is the memory length.

    Args:
        k: Current time step index.
        memory: Maximum number of history steps to retain. If `None` or `-1`, uses full history.

    Returns:
        A tuple `(start_idx, memory_length)` defining the slice of history to use.

            - `start_idx`: The starting index in the history list;
            - `memory_length`: The number of elements to include.
    """
    if memory is None or memory == -1:
        memory_length = k + 1
    else:
        memory_length = min(memory, k + 1)
        assert memory_length > 0, "memory must be greater than 0"

    start_idx = max(0, k + 1 - memory_length)
    return start_idx, memory_length


class SNNFractionalMethod(ABC):
    r"""
    Abstract Base Class (ABC) for SNN fractional differential equation solvers.

    Defines the interface for various numerical methods (GL, L1, Trapezoidal, etc.).
    Implementations must define how weights are computed, how convolutions are performed,
    and how the state update is calculated.

    Subclasses distinguish themselves by:

    1. The formulation used (Riemann-Liouville vs. Caputo).
    2. The type of history stored ($y$ values vs. $f(t,y)$ values).
    3. Support for single-term vs. multi-term equations.
    """

    @property
    @abstractmethod
    def stores_f_history(self) -> bool:
        r"""
        Indicates whether the method stores function evaluations $f(t, y)$ or state values $y$ in history.

        Returns:
            `True` if the method (e.g., Adams-Bashforth) relies on $f$-history, `False` if the method (e.g., GL, L1) relies on $y$-history.
        """
        pass

    @abstractmethod
    def compute_weights_for_layer(
        self, k: int, start_idx: int, config: SNNSolverConfig, layer_idx: int
    ) -> Any:
        r"""
        Computes the convolution weights for a specific layer at time step $k$.

        Args:
            k: Current time step index.
            start_idx: Start index of the history window.
            config: Solver configuration containing layer metadata.
            layer_idx: Index of the layer being processed.

        Returns:
            A tensor or structure containing the weights $w_j$ for the convolution sum.
        """
        pass

    @abstractmethod
    def compute_update_for_layer(
        self,
        f_k_i: torch.Tensor,
        convolution_sum: Any,
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> torch.Tensor:
        r"""
        Computes the next state $y_{k+1}$ for a specific layer.

        Combines the current derivative $f_k$ and the convolution sum according to the method's formula.
        Note: The method name in the original code was slightly misleading; this function computes the update,
        while `compute_convolution` computes the sum. Based on usage in `snn_solve`, this function
        applies the final formula. However, looking at the implementation in subclasses,
        `compute_convolution` actually performs the summation loop, and this function applies the scaling.

        Correction based on code analysis:
        `compute_convolution` returns the sum $\sum w_j h_j$.
        `compute_update_for_layer` takes that sum and $f_k$ to return $y_{k+1}$.

        Args:
            f_k_i: The derivative/value $f(t_k, y_k)$ for this layer.
            convolution_sum: The result of the history convolution.
            config: Solver configuration.
            layer_idx: Index of the layer.

        Returns:
            The updated state tensor $y_{k+1}$.
        """
        pass

    @abstractmethod
    def compute_convolution(
        self,
        k: int,
        start_idx: int,
        weights: Any,
        history_i: List[torch.Tensor],
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> Any:
        r"""
        Computes the weighted sum (convolution) of history values.

        Calculates $\sum w_j \cdot h_j$, where $h_j$ is either $y_j$ or $f_j$ depending on `stores_f_history`.

        Args:
            k: Current time step index.
            start_idx: Start index of the history window.
            weights: Weights computed by `compute_weights_for_layer`.
            history_i: List of historical tensors for the current component.
            config: Solver configuration.
            layer_idx: Index of the layer.

        Returns:
            The result of the convolution sum (tensor).
        """
        pass

    def initialize(self, config: SNNSolverConfig) -> None:
        r"""
        Optional hook for method-specific precomputation before the time loop.

        Used to precompute static coefficients (e.g., GL binomial coefficients) that depend
        on $\alpha$ and $N$ but not on the state $y$.

        Args:
            config: Solver configuration.
        """
        pass


class GrunwaldLetnikovSNN(SNNFractionalMethod):
    r"""
    Grünwald-Letnikov (GL) solver for single-term Riemann-Liouville Fractional Differential Equations (FDEs).

    This class implements the standard GL discretization scheme, which approximates the
    Riemann-Liouville fractional derivative $D^\alpha y(t)$ using a finite difference convolution.

    Mathematical Formulation:
    The update rule for the state $y$ at step $k+1$ is given by:

    $$y_{k+1} = h^\alpha f(t_k, y_k) - \sum_{j=0}^{k} c_{k-j}^{(\alpha)} y_j$$

    where $h$ is the step size, $f(t, y)$ is the ODE function, and $c_j^{(\alpha)}$ are the
    Grünwald-Letnikov coefficients generated recursively:

    $$c_0^{(\alpha)} = 1, \quad c_j^{(\alpha)} = \left(1 - \frac{1+\alpha}{j}\right)c_{j-1}^{(\alpha)} \quad \text{for } j \ge 1$$

    Key Characteristics:

    - **Formulation**: Riemann-Liouville.
    - **Accuracy**: First-order $O(h)$.
    - **Memory**: Requires full history of states $y$ unless truncated.
    - **Constraint**: Strictly supports single-term fractional orders ($\alpha$ is a scalar per layer).
      Attempting to use multi-term $\alpha$ will raise a `ValueError`. For multi-term support,
      use `GrunwaldLetnikovMultitermSNN`.
    """

    def __init__(self):
        """Initializes the solver with empty coefficient storage."""
        self._c_per_layer: Optional[List[torch.Tensor]] = None

    @property
    def stores_f_history(self) -> bool:
        return False

    def initialize(self, config: SNNSolverConfig) -> None:
        self._c_per_layer = []

        for i, layer_info in enumerate(config.per_layer_info):
            if layer_info.is_multi_term:
                raise ValueError(
                    f"Layer {i} has multi-term alpha but GrunwaldLetnikovSNN only supports "
                    f"single-term. Use GrunwaldLetnikovMultitermSNN instead."
                )

            # Single-term: alpha is a 1-element tensor
            alpha = layer_info.alpha.squeeze()  # Get scalar tensor

            # Avoid in-place operations to preserve gradient flow when alpha is learnable
            # Build coefficients using list and stack instead of in-place assignment
            c_list = [torch.ones(1, dtype=config.dtype, device=config.device)]
            for j in range(1, config.N + 1):
                c_next = (1 - (1 + alpha) / j) * c_list[j - 1]
                c_list.append(c_next.unsqueeze(0) if c_next.dim() == 0 else c_next)
            c = torch.cat(c_list, dim=0)
            self._c_per_layer.append(c)

    def compute_weights_for_layer(
        self, k: int, start_idx: int, config: SNNSolverConfig, layer_idx: int
    ) -> torch.Tensor:
        return self._c_per_layer[layer_idx]

    def compute_convolution(
        self,
        k: int,
        start_idx: int,
        weights: torch.Tensor,
        history_i: List[torch.Tensor],
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> Any:
        if k > 0:
            convolution_sum = 0
            for j in range(start_idx, k):
                convolution_sum = convolution_sum + weights[k - j] * history_i[j]
        else:
            convolution_sum = 0
        return convolution_sum

    def compute_update_for_layer(
        self,
        f_k_i: torch.Tensor,
        convolution_sum: Any,
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> torch.Tensor:
        layer_info = config.per_layer_info[layer_idx]
        return layer_info.h_alpha * f_k_i - convolution_sum


class ProductTrapezoidalSNN(SNNFractionalMethod):
    r"""
    Product Trapezoidal solver for single-term Riemann-Liouville FDEs.

    This method offers higher accuracy ($O(h^2)$) compared to the Grünwald-Letnikov scheme
    by using a piecewise linear interpolation of the integrand. It is particularly effective
    for smooth solutions.

    The update rule is:

    $$y_{k+1} = \frac{h^\alpha}{\Gamma(2-\alpha)} f(t_k, y_k) - \sum_{j=0}^{k} A_{j,k+1} y_j$$

    The weights $A_{j,k+1}$ are position-dependent and defined as:

    - For $j=0$:

      $$A_{0,k+1} = k^{1-\alpha} - (k+\alpha)(k+1)^{-\alpha}$$

    - For $j \ge 1$:

      $$A_{j,k+1} = (k+2-j)^{1-\alpha} + (k-j)^{1-\alpha} - 2(k+1-j)^{1-\alpha}$$

    Key Characteristics:

    - **Formulation**: Riemann-Liouville.
    - **Accuracy**: Second-order $O(h^2)$.
    - **Constraint**: Supports single-term $\alpha$ only.
    """

    @property
    def stores_f_history(self) -> bool:
        return False

    def initialize(self, config: SNNSolverConfig) -> None:
        for i, layer_info in enumerate(config.per_layer_info):
            if layer_info.is_multi_term:
                raise ValueError(
                    f"Layer {i} has multi-term alpha but ProductTrapezoidalSNN only supports "
                    f"single-term. Use GrunwaldLetnikovMultitermSNN instead."
                )

    def compute_weights_for_layer(
        self, k: int, start_idx: int, config: SNNSolverConfig, layer_idx: int
    ) -> torch.Tensor:
        layer_info = config.per_layer_info[layer_idx]
        alpha = layer_info.alpha.squeeze()  # Get scalar tensor for single-term
        one_minus_alpha = 1 - alpha

        j_vals = torch.arange(
            start_idx, k + 1, dtype=config.dtype, device=config.device
        )

        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        A_j_kp1 = kjp2 + kj - 2 * kjp1

        if start_idx == 0:
            k_power = torch.pow(
                torch.tensor(k, dtype=config.dtype, device=config.device),
                one_minus_alpha,
            )
            kp1_neg_alpha = torch.pow(
                torch.tensor(k + 1, dtype=config.dtype, device=config.device), -alpha
            )
            # Avoid in-place assignment to preserve gradient flow
            A_0_value = k_power - (k + alpha) * kp1_neg_alpha
            # Replace first element without in-place operation
            A_j_kp1 = torch.cat([A_0_value.unsqueeze(0), A_j_kp1[1:]], dim=0)

        return A_j_kp1

    def compute_convolution(
        self,
        k: int,
        start_idx: int,
        weights: torch.Tensor,
        history_i: List[torch.Tensor],
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> Any:
        if k > 0:
            convolution_sum = 0
            for j in range(start_idx, k):
                local_idx = j - start_idx + 1
                convolution_sum = convolution_sum + weights[local_idx] * history_i[j]
        else:
            convolution_sum = 0
        return convolution_sum

    def compute_update_for_layer(
        self,
        f_k_i: torch.Tensor,
        convolution_sum: Any,
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> torch.Tensor:
        layer_info = config.per_layer_info[layer_idx]
        return layer_info.h_alpha_gamma * f_k_i - convolution_sum


class L1MethodSNN(SNNFractionalMethod):
    r"""
    L1 scheme solver for single-term Caputo Fractional Differential Equations.

    The L1 method is the most widely used numerical scheme for Caputo derivatives,
    offering an accuracy of $O(h^{2-\alpha})$ for smooth solutions. It approximates
    the fractional derivative using piecewise linear interpolation of the function.

    Mathematical Formulation:
    The update rule is:

    $$y_{k+1} = \frac{h^\alpha}{\Gamma(2-\alpha)} f(t_k, y_k) - \sum_{j=0}^{k} c_j^{(k)} y_j$$

    The coefficients $c_j^{(k)}$ are defined as:

    - For $j=0$:

      $$c_0^{(k)} = -\left((k+1)^{1-\alpha} - k^{1-\alpha}\right)$$

    - For $j \ge 1$:

      $$c_j^{(k)} = (k-j+2)^{1-\alpha} - 2(k-j+1)^{1-\alpha} + (k-j)^{1-\alpha}$$

    Key Characteristics:

    - **Formulation**: Caputo.
    - **Accuracy**: $O(h^{2-\alpha})$.
    - **Constraint**: Single-term $\alpha$ only.
    """

    @property
    def stores_f_history(self) -> bool:
        return False

    def initialize(self, config: SNNSolverConfig) -> None:
        for i, layer_info in enumerate(config.per_layer_info):
            if layer_info.is_multi_term:
                raise ValueError(
                    f"Layer {i} has multi-term alpha but L1MethodSNN only supports "
                    f"single-term. Use GrunwaldLetnikovMultitermSNN instead."
                )

    def compute_weights_for_layer(
        self, k: int, start_idx: int, config: SNNSolverConfig, layer_idx: int
    ) -> torch.Tensor:
        layer_info = config.per_layer_info[layer_idx]
        alpha = layer_info.alpha.squeeze()  # Get scalar tensor for single-term
        one_minus_alpha = 1 - alpha

        j_vals = torch.arange(
            start_idx, k + 1, dtype=config.dtype, device=config.device
        )

        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        c_j_k = kjp2 - 2 * kjp1 + kj

        if start_idx == 0:
            kp1_power = torch.pow(
                torch.tensor(k + 1, dtype=config.dtype, device=config.device),
                one_minus_alpha,
            )
            k_power = torch.pow(
                torch.tensor(k, dtype=config.dtype, device=config.device),
                one_minus_alpha,
            )
            # Avoid in-place assignment to preserve gradient flow
            c_0_value = -(kp1_power - k_power)
            # Replace first element without in-place operation
            c_j_k = torch.cat([c_0_value.unsqueeze(0), c_j_k[1:]], dim=0)

        return c_j_k

    def compute_convolution(
        self,
        k: int,
        start_idx: int,
        weights: torch.Tensor,
        history_i: List[torch.Tensor],
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> Any:
        if k > 0:
            convolution_sum = 0
            for j in range(start_idx, k):
                local_idx = j - start_idx + 1
                convolution_sum = convolution_sum + weights[local_idx] * history_i[j]
        else:
            convolution_sum = 0
        return convolution_sum

    def compute_update_for_layer(
        self,
        f_k_i: torch.Tensor,
        convolution_sum: Any,
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> torch.Tensor:
        layer_info = config.per_layer_info[layer_idx]
        return layer_info.h_alpha_gamma * f_k_i - convolution_sum


class AdamsBashforthSNN(SNNFractionalMethod):
    r"""
    Adams-Bashforth predictor method for single-term Caputo FDEs.

    This method serves as a predictor step in predictor-corrector schemes (like PECE).
    Unlike the other methods which convolve state history $y_j$, Adams-Bashforth convolves
    the history of function evaluations $f(t_j, y_j)$.

    The update rule is:

    $$y_{k+1} = \sum_{j=0}^{k} b_{j,k+1} f(t_j, y_j)$$

    where the weights are:

    $$b_{j,k+1} = \frac{h^\alpha}{\alpha \Gamma(\alpha)} \left[ (k+1-j)^\alpha - (k-j)^\alpha \right]$$

    Key Characteristics:

    - **Formulation**: Caputo (Predictor).
    - **History Type**: Stores $f(t, y)$ instead of $y$.
    - **Constraint**: Single-term $\alpha$ only.
    """

    @property
    def stores_f_history(self) -> bool:
        return True

    def initialize(self, config: SNNSolverConfig) -> None:
        for i, layer_info in enumerate(config.per_layer_info):
            if layer_info.is_multi_term:
                raise ValueError(
                    f"Layer {i} has multi-term alpha but AdamsBashforthSNN only supports "
                    f"single-term. Use GrunwaldLetnikovMultitermSNN instead."
                )

    def compute_weights_for_layer(
        self, k: int, start_idx: int, config: SNNSolverConfig, layer_idx: int
    ) -> torch.Tensor:
        layer_info = config.per_layer_info[layer_idx]
        alpha = layer_info.alpha.squeeze()  # Get scalar tensor for single-term

        j_vals = torch.arange(
            start_idx, k + 1, dtype=config.dtype, device=config.device
        )
        b_j_kp1 = layer_info.h_alpha_over_alpha_gamma * (
            torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
        )
        return b_j_kp1

    def compute_convolution(
        self,
        k: int,
        start_idx: int,
        weights: torch.Tensor,
        history_i: List[torch.Tensor],
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> Any:
        convolution_sum = 0
        for j in range(start_idx, k + 1):
            local_idx = j - start_idx
            convolution_sum = convolution_sum + weights[local_idx] * history_i[j]
        return convolution_sum

    def compute_update_for_layer(
        self,
        f_k_i: torch.Tensor,
        convolution_sum: Any,
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> torch.Tensor:
        return convolution_sum


class GrunwaldLetnikovMultitermSNN(SNNFractionalMethod):
    r"""
    Unified Grünwald-Letnikov solver for multi-term Riemann-Liouville FDEs.

    This solver handles distributed-order or multi-term equations of the form:

    $$\sum_{m=1}^{M} c_m D^{\alpha_m} y(t) = f(t, y(t))$$

    It generalizes the single-term GL method by aggregating the coefficients from each term
    into a single effective convolution kernel.

    The discretization leads to the update rule:

    $$y_{k+1} = \frac{1}{\tilde{c}_0} \left( f(t_k, y_k) - \sum_{j=0}^{k} \tilde{c}_{k-j} y_j \right)$$

    where the aggregated coefficients $\tilde{c}_m$ are computed as:

    $$\tilde{c}_m = \sum_{i=1}^{M} c_i h^{-\alpha_i} c_m^{(\alpha_i)}$$

    Here, $c_i$ are the user-defined equation coefficients, $h^{-\alpha_i}$ scales by step size,
    and $c_m^{(\alpha_i)}$ are the standard GL coefficients for order $\alpha_i$.

    Key Characteristics:

    - **Formulation**: Riemann-Liouville (Multi-term).
    - **Flexibility**: Supports both single-term (as a 1-term case) and multi-term layers.
    - **Gradient Flow**: Fully differentiable with respect to $\alpha_m$ and coefficients $c_m$.
    """

    def __init__(self):
        self._c_tilde_per_layer: Optional[List[torch.Tensor]] = None
        self._c_tilde_0_inv_per_layer: Optional[List[torch.Tensor]] = None

    @property
    def stores_f_history(self) -> bool:
        return False

    def initialize(self, config: SNNSolverConfig) -> None:
        self._c_tilde_per_layer = []
        self._c_tilde_0_inv_per_layer = []

        for layer_info in config.per_layer_info:
            # Both single-term and multi-term now have alpha as tensor and coefficient stored
            alpha = layer_info.alpha  # Always a tensor now
            coefficient = layer_info.coefficient  # Always available (defaults to ones)

            n_terms = alpha.numel()

            # Compute GL coefficients c_m^{(α_j)} for each fractional order
            # Avoid in-place operations to preserve gradient flow when alpha is learnable
            # Build columns iteratively using list and stack
            c_columns = [torch.ones(n_terms, dtype=config.dtype, device=config.device)]
            for m in range(1, config.N + 1):
                c_next = (1 - (1 + alpha) / m) * c_columns[m - 1]
                c_columns.append(c_next)
            c = torch.stack(c_columns, dim=1)  # Shape: (n_terms, config.N + 1)

            # Compute h^{-α_j} for each term
            h_neg_power = torch.pow(config.h, -alpha)

            # Compute distributed GL weights
            # coefficient maintains gradient flow if it's a learnable Parameter
            weighted_h = coefficient * h_neg_power
            c_tilde = torch.sum(weighted_h.unsqueeze(1) * c, dim=0)
            c_tilde_0_inv = 1.0 / c_tilde[0]

            self._c_tilde_per_layer.append(c_tilde)
            self._c_tilde_0_inv_per_layer.append(c_tilde_0_inv)

    def compute_weights_for_layer(
        self, k: int, start_idx: int, config: SNNSolverConfig, layer_idx: int
    ) -> torch.Tensor:
        return self._c_tilde_per_layer[layer_idx]

    def compute_convolution(
        self,
        k: int,
        start_idx: int,
        weights: torch.Tensor,
        history_i: List[torch.Tensor],
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> Any:
        if k > 0:
            convolution_sum = 0
            for j in range(start_idx, k):
                coeff_idx = k - j
                convolution_sum = convolution_sum + weights[coeff_idx] * history_i[j]
        else:
            convolution_sum = 0
        return convolution_sum

    def compute_update_for_layer(
        self,
        f_k_i: torch.Tensor,
        convolution_sum: Any,
        config: SNNSolverConfig,
        layer_idx: int,
    ) -> torch.Tensor:
        return self._c_tilde_0_inv_per_layer[layer_idx] * (f_k_i - convolution_sum)


def snn_solve(
    ode_func: Callable[[torch.Tensor, Tuple], Tuple],
    y0_tuple: Tuple[torch.Tensor, ...],
    per_layer_alpha: List[Any],
    t_grid: torch.Tensor,
    method: SNNFractionalMethod,
    memory: Optional[int] = None,
    per_layer_coefficient: Optional[List[Optional[torch.Tensor]]] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Unified driver function for solving SNN fractional differential equations.

    Orchestrates the time-stepping loop, managing state history, memory truncation,
    and dispatching to the specific numerical method provided.

    Args:
        ode_func: Function `f(t, y_tuple)` returning derivatives.
        y0_tuple: Initial state tuple.
        per_layer_alpha: List of fractional orders per layer.
        t_grid: Time points tensor.
        method: Instance of `SNNFractionalMethod` (e.g., `GrunwaldLetnikovSNN`).
        memory: Optional integer to limit history length for convolution.
        per_layer_coefficient: Coefficients for multi-term layers.

    Returns:
        List of lists containing the trajectory of each state component.
    """
    assert isinstance(y0_tuple, tuple), "y0_tuple must be a tuple"

    # Create configuration with per-layer alpha
    config = SNNSolverConfig.from_inputs(
        y0_tuple, per_layer_alpha, t_grid, per_layer_coefficient
    )

    # Move t_grid to correct device/dtype
    t_grid = t_grid.to(device=config.device, dtype=config.dtype)

    # Initialize method
    method.initialize(config)

    # Initialize state
    y_current = list(y0_tuple)

    # History
    y_history = [[] for _ in y0_tuple]

    # For predictor method, we need f-history
    if method.stores_f_history:
        fhistory = [[] for _ in range(config.n_integrate)]
    else:
        fhistory = None

    # Main loop
    for k in range(config.N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Store function evaluations if needed
        if method.stores_f_history:
            for i in range(config.n_integrate):
                fhistory[i].append(f_k[i])

        # Determine memory range
        start_idx, _ = get_memory_bounds(k, memory)

        # Update each integrated component with its own alpha
        for i in range(config.n_integrate):
            # Compute weights for this specific layer
            weights = method.compute_weights_for_layer(
                k, start_idx, config, layer_idx=i
            )

            # Get appropriate history
            history_i = fhistory[i] if method.stores_f_history else y_history[i]

            # Compute convolution sum
            convolution_sum = method.compute_convolution(
                k, start_idx, weights, history_i, config, layer_idx=i
            )

            # Compute update for this layer
            y_current[i] = method.compute_update_for_layer(
                f_k[i], convolution_sum, config, layer_idx=i
            )

            # Store in history
            if not method.stores_f_history:
                y_history[i].append(y_current[i])

        # Pass-through boundary output e.g. the final spike output
        for i in range(config.n_integrate, config.n_components):
            y_current[i] = f_k[i]
            y_history[i].append(y_current[i])

    # Cleanup
    if fhistory is not None:
        del fhistory

    return y_history


# ============================================================================
# Public API - Modified for per-layer alpha
# ============================================================================


def _has_any_multiterm(per_layer_alpha: List[Any]) -> bool:
    """
    Check if any layer has multi-term alpha.

    Args:
        per_layer_alpha: List of alpha values (tensors, floats, or lists)

    Returns:
        True if any layer has more than one alpha value
    """
    for alpha in per_layer_alpha:
        if isinstance(alpha, torch.Tensor):
            if alpha.numel() > 1:
                return True
        elif isinstance(alpha, (list, tuple)):
            if len(alpha) > 1:
                return True
        # Scalar float is single-term
    return False


def _get_solver_with_multiterm_fallback(
    method_name: str,
    per_layer_alpha: List[Any],
    per_layer_coefficient: Optional[List[Optional[torch.Tensor]]] = None,
):
    """
    Get the appropriate solver, falling back to GL multiterm if needed.

    Args:
        method_name: Requested method (`'gl'`, `'trap'`, `'l1'`, `'pred'`)
        per_layer_alpha: List of alpha values
        per_layer_coefficient: Optional coefficients

    Returns:
        Tuple of (solver_instance, actually_used_method_name)
    """
    import warnings

    has_multiterm = _has_any_multiterm(per_layer_alpha)

    if has_multiterm:
        if method_name != "gl":
            warnings.warn(
                f"Multi-term alpha detected but method='{method_name}' was requested. "
                f"Currently only 'gl' (Grünwald-Letnikov) supports multi-term. "
                f"Automatically switching to GrunwaldLetnikovMultitermSNN.",
                UserWarning,
            )
        return GrunwaldLetnikovMultitermSNN(), "glmulti"
    else:
        # All single-term, use requested solver
        solvers_map = {
            "gl": GrunwaldLetnikovSNN,
            "trap": ProductTrapezoidalSNN,
            "l1": L1MethodSNN,
            "pred": AdamsBashforthSNN,
        }
        if method_name not in solvers_map:
            raise ValueError(
                f"Unknown method: {method_name}. Choose from {list(solvers_map.keys())}"
            )
        return solvers_map[method_name](), method_name


def gl_integrate_tuple(
    ode_func: Callable,
    y0_tuple: Tuple[torch.Tensor, ...],
    per_layer_alpha: List[Any],
    t_grid: torch.Tensor,
    memory: Optional[int] = None,
    per_layer_coefficient: Optional[List[Optional[torch.Tensor]]] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Solves FDEs using the Grünwald-Letnikov (GL) method with per-layer alpha support.

    Automatically switches to `GrunwaldLetnikovMultitermSNN` if any layer has multi-term alpha.
    Suitable for Riemann-Liouville formulations.

    Args:
        ode_func: Function `f(t, y_tuple)` returning derivatives.
        y0_tuple: Initial state tuple.
        per_layer_alpha: List of alpha values, one per integrated component.
                         Each can be scalar (single-term) or list/tensor (multi-term).
        t_grid: Time points tensor.
        memory: Optional memory truncation length.
        per_layer_coefficient: Coefficients for multi-term layers.

    Returns:
        List of lists containing the state trajectory.
    """
    solver, _ = _get_solver_with_multiterm_fallback(
        "gl", per_layer_alpha, per_layer_coefficient
    )
    return snn_solve(
        ode_func,
        y0_tuple,
        per_layer_alpha,
        t_grid,
        solver,
        memory=memory,
        per_layer_coefficient=per_layer_coefficient,
    )


def trap_integrate_tuple(
    ode_func: Callable,
    y0_tuple: Tuple[torch.Tensor, ...],
    per_layer_alpha: List[Any],
    t_grid: torch.Tensor,
    memory: Optional[int] = None,
    per_layer_coefficient: Optional[List[Optional[torch.Tensor]]] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Solves FDEs using the Product Trapezoidal method with per-layer alpha support.

    Note:
        If any layer has multi-term alpha, automatically falls back to GL multiterm
        with a warning. Offers higher accuracy ($O(h^2)$) for single-term Riemann-Liouville equations.

    Args:
        ode_func: Function `f(t, y_tuple)` returning derivatives.
        y0_tuple: Initial state tuple.
        per_layer_alpha: List of alpha values.
        t_grid: Time points tensor.
        memory: Optional memory truncation length.
        per_layer_coefficient: Coefficients for multi-term layers.

    Returns:
        List of lists containing the state trajectory.
    """
    solver, _ = _get_solver_with_multiterm_fallback(
        "trap", per_layer_alpha, per_layer_coefficient
    )
    return snn_solve(
        ode_func,
        y0_tuple,
        per_layer_alpha,
        t_grid,
        solver,
        memory=memory,
        per_layer_coefficient=per_layer_coefficient,
    )


def l1_integrate_tuple(
    ode_func: Callable,
    y0_tuple: Tuple[torch.Tensor, ...],
    per_layer_alpha: List[Any],
    t_grid: torch.Tensor,
    memory: Optional[int] = None,
    per_layer_coefficient: Optional[List[Optional[torch.Tensor]]] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Solves FDEs using the L1 scheme with per-layer alpha support.

    Note:
        If any layer has multi-term alpha, automatically falls back to GL multiterm
        with a warning. Commonly used for Caputo formulations with accuracy $O(h^{2-\alpha})$.

    Args:
        ode_func: Function `f(t, y_tuple)` returning derivatives.
        y0_tuple: Initial state tuple.
        per_layer_alpha: List of alpha values.
        t_grid: Time points tensor.
        memory: Optional memory truncation length.
        per_layer_coefficient: Coefficients for multi-term layers.

    Returns:
        List of lists containing the state trajectory.
    """
    solver, _ = _get_solver_with_multiterm_fallback(
        "l1", per_layer_alpha, per_layer_coefficient
    )
    return snn_solve(
        ode_func,
        y0_tuple,
        per_layer_alpha,
        t_grid,
        solver,
        memory=memory,
        per_layer_coefficient=per_layer_coefficient,
    )


def pred_integrate_tuple(
    ode_func: Callable,
    y0_tuple: Tuple[torch.Tensor, ...],
    per_layer_alpha: List[Any],
    t_grid: torch.Tensor,
    memory: Optional[int] = None,
    per_layer_coefficient: Optional[List[Optional[torch.Tensor]]] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Solves FDEs using the Adams-Bashforth predictor with per-layer alpha support.

    Note:
        If any layer has multi-term alpha, automatically falls back to GL multiterm
        with a warning. Uses $f$-history instead of $y$-history.

    Args:
        ode_func: Function `f(t, y_tuple)` returning derivatives.
        y0_tuple: Initial state tuple.
        per_layer_alpha: List of alpha values.
        t_grid: Time points tensor.
        memory: Optional memory truncation length.
        per_layer_coefficient: Coefficients for multi-term layers.

    Returns:
        List of lists containing the state trajectory.
    """
    solver, _ = _get_solver_with_multiterm_fallback(
        "pred", per_layer_alpha, per_layer_coefficient
    )
    return snn_solve(
        ode_func,
        y0_tuple,
        per_layer_alpha,
        t_grid,
        solver,
        memory=memory,
        per_layer_coefficient=per_layer_coefficient,
    )


def glmethod_multiterm_integrate_tuple(
    ode_func: Callable,
    y0_tuple: Tuple[torch.Tensor, ...],
    per_layer_alpha: List[Any],
    t_grid: torch.Tensor,
    memory: Optional[int] = None,
    per_layer_coefficient: Optional[List[Optional[torch.Tensor]]] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Solves FDEs using the Multi-term GL method with per-layer alpha support.

    This solver handles both single-term and multi-term layers uniformly.
    Single-term layers are treated as 1-element multi-term.

    Args:
        ode_func: Function `f(t, y_tuple)` returning derivatives.
        y0_tuple: Initial state tuple.
        per_layer_alpha: List where each element is either:
                         - float/1-element tensor (single-term for that layer)
                         - list/multi-element tensor (multi-term for that layer)
        t_grid: Time points tensor.
        memory: Optional memory truncation length.
        per_layer_coefficient: List of coefficient tensors/lists, one per layer.
                               For single-term layers, defaults to `[1.0]`.
                               For multi-term layers, defaults to ones if None.

    Returns:
        List of lists containing the state trajectory.
    """
    return snn_solve(
        ode_func,
        y0_tuple,
        per_layer_alpha,
        t_grid,
        GrunwaldLetnikovMultitermSNN(),
        memory=memory,
        per_layer_coefficient=per_layer_coefficient,
    )


# Backward compatibility wrapper for single alpha (broadcasts to all layers)
def _broadcast_alpha_to_layers(alpha, n_layers):
    """
    Convert single alpha to per-layer format.

    Args:
        alpha: float, list, or tensor
        n_layers: number of integrated components

    Returns:
        list: per-layer alpha values
    """
    # Check if alpha is already per-layer
    if isinstance(alpha, (list, tuple)):
        if len(alpha) == n_layers:
            # Could be per-layer single-term or single multi-term alpha
            # Check if first element is also a list/tuple (per-layer multi-term)
            if all(isinstance(a, (list, tuple, torch.Tensor)) for a in alpha):
                return alpha  # Already per-layer multi-term
            else:
                # Check if it looks like a multi-term alpha (all floats) vs per-layer single-term
                # Heuristic: if all elements are scalars, treat as per-layer single-term
                return alpha
        else:
            # Single multi-term alpha, broadcast to all layers
            return [alpha] * n_layers
    elif isinstance(alpha, torch.Tensor):
        if alpha.numel() == 1:
            # Single scalar, broadcast
            return [alpha.item()] * n_layers
        elif alpha.numel() == n_layers:
            # Per-layer single-term
            return alpha.tolist()
        else:
            # Multi-term, broadcast to all layers
            return [alpha] * n_layers
    else:
        # Single float, broadcast to all layers
        return [alpha] * n_layers


SOLVERS = {
    "gl": gl_integrate_tuple,
    "trap": trap_integrate_tuple,
    "l1": l1_integrate_tuple,
    "pred": pred_integrate_tuple,
}

# ----------------------------------------------------------------


def step_dynamics(
    ode_func: Callable[[torch.Tensor, Tuple], Tuple],
    y0_tuple: Tuple[torch.Tensor, ...],
    t_grid: torch.Tensor,
) -> List[torch.Tensor]:
    r"""
    Steps through a discrete-time dynamical system, collecting boundary outputs.

    This function drives the for-loop of an SNN/RNN without numerical integration scaling (no $dt$).
    The update function directly computes the next state: $y_{k+1} = f(t_k, y_k)$.

    Args:
        ode_func: Callable `(t, y_tuple) -> tuple`. State update function.
        y0_tuple: Tuple of initial state tensors.
        t_grid: 1D tensor of time points (length T+1).

    Returns:
        List of spike outputs (last component of state) at each time step.
    """
    assert isinstance(y0_tuple, tuple)
    N = len(t_grid)
    assert N >= 2

    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    y = list(y0_tuple)
    spike_history = []

    for k in range(N - 1):
        t_k = t_grid[k]
        y = ode_func(t_k, tuple(y))
        spike_history.append(y[-1])
        ###here we only assume one boundary term.
        ###will update to (boundary_1, boundary_2, ...) if necessary.

    return spike_history


# ------------------------------------ the following is adjoint training code ------------------------------------#


def fdeint_adjoint(
    func: Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
    y0_tuple: Tuple[torch.Tensor, ...],
    alpha: Union[float, torch.Tensor, List[float]],
    t_grid: torch.Tensor,
    method: str,
    memory: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    r"""
    Solves a Fractional Differential Equation (FDE) with adjoint sensitivity analysis.

    This function enables gradient-based optimization of both the initial states $y_0$ and
    the parameters of the ODE function `func` (e.g., neural network weights) with respect to
    a loss function defined on the solution trajectory. It uses the continuous adjoint method
    adapted for fractional calculus.

    The workflow involves:

    1. **Forward Pass**: Solving $D^\alpha y(t) = f(t, y(t), \theta)$ to obtain $y(T)$.
    2. **Backward Pass**: Solving the augmented adjoint equation to compute $\frac{\partial L}{\partial y_0}$
       and $\frac{\partial L}{\partial \theta}$.

    Args:
        func: The ODE function $f(t, y, \theta)$. Must accept `(t, y_tuple)` and return a tuple of tensors.
              Parameters $\theta$ are implicitly captured from the function's scope or registered modules.
        y0_tuple: A tuple of initial state tensors $(y_1^0, \dots, y_N^0)$.
        alpha: The fractional order(s). Can be a scalar, a tensor, or a list depending on the solver configuration.
        t_grid: A 1D tensor of time points $[t_0, t_1, \dots, t_T]$ defining the integration interval.
        method: The numerical integration scheme identifier (e.g., `'gl-f'`, `'trap-f'`, `'l1-f'`).
                Suffixes `-f` indicate full history storage required for adjoint, `-o` for optimized/no-history.
        memory: Optional integer to limit the memory length for convolution sums (short-memory principle).
                If `None`, full history is used.

    Returns:
        A tuple of tensors representing the solution at the final time point $y(t_T)$, compatible with `torch.autograd` for backpropagation.

    Note:
        This function wraps `FDEAdjointMethod.apply`. Ensure `func` contains parameters that require gradients if parameter optimization is desired.
    """
    # params = tuple(p for p in func.parameters())  # 或只取 requires_grad=True 的
    params = find_parameters(func)
    n_state = len(y0_tuple)
    n_params = len(params)
    # 注意：apply 不接受关键字参数；把计数放在前两位最简单
    return FDEAdjointMethod.apply(
        func, n_state, n_params, *y0_tuple, alpha, t_grid, method, *params, memory
    )


class FDEAdjointMethod(torch.autograd.Function):
    r"""
    Custom Autograd Function for Fractional Differential Equations with Adjoint Sensitivity.

    This class implements the forward and backward passes required for differentiating through
    FDE solvers. It supports various numerical schemes (GL, Trapezoidal, L1, Adams-Bashforth)
    and handles the complexity of fractional memory terms during backpropagation.

    Mathematical Formulation:
    The adjoint state $\lambda(t)$ satisfies the fractional adjoint equation:

    $$D^\alpha \lambda(t) = -\left(\frac{\partial f}{\partial y}\right)^T \lambda(t)$$

    solved backwards from $t=T$ to $t=0$. Parameter gradients are computed via:

    $$\frac{dL}{d\theta} = \int_0^T \lambda(t)^T \frac{\partial f}{\partial \theta} dt$$
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        ode_func: Callable,
        n_state: int,
        n_params: int,
        *args: Any,
    ) -> Tuple[torch.Tensor, ...]:
        r"""
        Performs the forward integration of the FDE.

        Unpacks arguments, selects the appropriate solver based on `method`, and computes
        the state trajectory. Saves necessary context for the backward pass.

        Args:
            ctx: Context object to save tensors for backward pass.
            ode_func: The ODE function $f(t, y)$.
            n_state: Number of state components in `y0_tuple`.
            n_params: Number of learnable parameters in `ode_func`.
            *args: Packed arguments containing:

                - `y0_tuple`: Initial states (n_state tensors).
                - `alpha`: Fractional order.
                - `t_grid`: Time grid.
                - `method`: Solver method string.
                - `func_params`: Model parameters (n_params tensors).
                - `memory`: Memory truncation limit.

        Returns:
            A tuple of tensors representing the final state $y(t_{end})$.
        """
        n_state = int(n_state)
        n_params = int(n_params)

        # 解析位置参数： y0_1,...,y0_n, alpha, t_grid, method, p1,...,pm, memory
        y0_tuple = tuple(args[:n_state])  # Tensors
        alpha = args[n_state]  # Tensor 或 float（若要学习，必须是 Tensor）
        t_grid = args[n_state + 1]  # Tensor
        method = args[n_state + 2]  # str / enum（非 Tensor）
        func_params = tuple(
            args[n_state + 3 : n_state + 3 + n_params]
        )  # Tensors (Parameters)
        memory = args[n_state + 3 + n_params]  # 任意对象（非 Tensor）

        with torch.no_grad():
            yhistory = SOLVERS_Forward[method](
                ode_func=ode_func,
                y0_tuple=y0_tuple,
                alpha=alpha,
                t_grid=t_grid,
                memory=memory,
            )

        # 检查是否需要梯度
        y0_needs_grad = any(t.requires_grad for t in y0_tuple)
        params_need_grad = (
            any(p.requires_grad for p in func_params) if func_params else False
        )

        ctx.n_state = n_state
        ctx.n_params = n_params
        if y0_needs_grad or params_need_grad:
            # 保存必要的张量用于反向传播
            # 注意：保存整个 tuple 对象，而不是展开的张量
            ctx.save_for_backward(t_grid)
            ctx.func_params = func_params  # 同样保存为属性
            ctx.yhistory = yhistory  # 保存最终状态
            ctx.ode_func = ode_func
            ctx.alpha = alpha
            ctx.method = method
            ctx.memory = memory

        # 返回结果
        # if method == 'euler':
        #     outs = tuple(yhistory)
        # else:
        #     # 如果 yhistory 是嵌套列表，取最后一个
        outs = tuple([y[-1] for y in yhistory])

        return outs

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_output: torch.Tensor
    ) -> Tuple[Optional[Any], ...]:
        r"""
        Performs the backward adjoint integration to compute gradients.

        Reconstructs the augmented dynamics system and solves it backwards in time to obtain
        gradients with respect to initial states ($y_0$) and model parameters ($\theta$).

        Args:
            ctx: Context object containing saved tensors from forward pass.
            *grad_output: Gradients of the loss with respect to the output states $y(t_{end})$.

        Returns:
            A tuple of gradients corresponding to the inputs of `forward`: `(grad_func, grad_n_state, grad_n_params, grad_y0..., grad_alpha, grad_t_grid, grad_method, grad_params..., grad_memory)`. Non-tensor inputs return `None`.
        """
        # 早退：不需要反传时，返回正确数量的 None
        if not hasattr(ctx, "yhistory"):
            n_state = ctx.n_state
            n_params = ctx.n_params
            grads = []
            grads.append(None)  # ode_func
            grads.append(None)  # n_state
            grads.append(None)  # n_params
            grads.extend([None] * n_state)  # y0_1,...,y0_n
            grads.append(None)  # alpha
            grads.append(None)  # t_grid
            grads.append(None)  # method
            grads.extend([None] * n_params)  # p1,...,pm
            grads.append(None)  # memory
            return tuple(grads)

        # 恢复保存的张量和属性
        t_grid = ctx.saved_tensors[0]
        func_params = ctx.func_params
        yhistory = ctx.yhistory
        # yhistory is the last states for euler and the full history for other methods

        func = ctx.ode_func
        alpha = ctx.alpha
        method = ctx.method
        memory = ctx.memory
        n_tensors = len(yhistory)

        # 创建 augmented dynamics
        class AugDynamics:
            def __init__(self, func, n_tensors, func_params):
                self.func = func
                self.n_tensors = n_tensors
                self.f_params = func_params  # 使用传入的参数

            def __call__(self, t, y_aug):
                y, adj_y, adj_params = y_aug

                with torch.set_grad_enabled(True):
                    # detach 并设置 requires_grad
                    y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                    func_eval = self.func(t, y)

                    # 计算 VJP
                    vjp_y_and_params = torch.autograd.grad(
                        func_eval,
                        y + self.f_params,
                        tuple(adj_y),
                        allow_unused=True,
                        retain_graph=False,  # 不保留图
                        create_graph=False,
                    )

                vjp_y = vjp_y_and_params[: self.n_tensors]
                vjp_params = vjp_y_and_params[self.n_tensors :]

                # 处理 None 梯度
                vjp_y = tuple(
                    torch.zeros_like(y_) if vjp_y_ is None else vjp_y_
                    for vjp_y_, y_ in zip(vjp_y, y)
                )

                vjp_params = tuple(
                    torch.zeros_like(p) if vp is None else vp
                    for vp, p in zip(vjp_params, self.f_params)
                )

                return (func_eval, vjp_y, vjp_params)

        # 创建 augmented dynamics 实例
        augmented_dynamics = AugDynamics(func, n_tensors, func_params)
        t_grid_flip = t_grid.flip(0)

        with torch.no_grad():
            adj_y = grad_output

            # 初始化参数梯度
            if func_params:
                adj_params = tuple(torch.zeros_like(p) for p in func_params)
            else:
                adj_params = ()

            # 设置初始增广状态
            # 注意：yhistory 作为初始 y 状态
            aug_y0 = ([], adj_y, adj_params)

            # 调用反向求解器
            adj_y, adj_params = SOLVERS_Backward[method](
                augmented_dynamics,
                aug_y0,
                alpha,
                t_grid_flip,
                yhistory,  # 传递最终状态
                memory,
            )

        # 清理
        # del augmented_dynamics
        # del ctx.yhistory

        # 在最后，确保清理所有局部变量
        del augmented_dynamics
        del yhistory  # 也要删除局部变量
        del func
        del func_params
        del ctx.yhistory
        del ctx.ode_func
        del ctx.func_params
        del ctx.alpha
        del ctx.method

        # 准备返回值
        # 返回格式：(grad_func, grad_y0_tuple, grad_alpha, grad_t_grid, grad_method, grad_func_params, grad_memory)
        grad_y0 = adj_y
        grad_params = adj_params

        return None, None, None, *grad_y0, None, None, None, *grad_params, None


def forward_euler_wo_history(
    ode_func: Callable,
    y0_tuple: Tuple[torch.Tensor, ...],
    alpha: Any,
    t_grid: torch.Tensor,
    memory: Optional[int] = None,
) -> List[torch.Tensor]:
    r"""
    Explicit Euler integration without storing full history.

    Solves $y_{k+1} = y_k + h \cdot f(t_k, y_k)$.
    This variant is memory-efficient ($O(1)$) but insufficient for methods requiring
    history-dependent adjoints unless combined with checkpointing. Used primarily for
    integer-order baselines or specific optimized paths.

    Args:
        ode_func: Function $f(t, y)$.
        y0_tuple: Initial state tuple.
        alpha: Fractional order (unused in standard Euler, kept for signature compatibility).
        t_grid: Time grid tensor.
        memory: Unused.

    Returns:
        List of final state tensors (not a list of lists).
    """
    assert isinstance(y0_tuple, tuple)
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    # Initialize history lists for each component

    # Clone initial values
    y = list(y0_tuple)

    # Euler integration: y_{k+1} = y_k + dt * f(t_k, y_k)
    for k in range(N - 1):
        tk = t_grid[k]
        dt = t_grid[k + 1] - t_grid[k]  # Scalar tensor, will broadcast automatically
        dy = ode_func(tk, tuple(y))  # Expect tuple return, consistent with y structure
        # assert isinstance(dy, tuple) and len(dy) == len(y)
        # Update all integrated components except the last one
        # for i in range(len(y)):
        #     y[i] = y[i] + dt * dy[i]
        for i in range(len(y)):
            y[i].add_(dy[i], alpha=dt)
    return y


def backward_euler_wo_history(
    ode_func: Callable,
    y_aug: Tuple[List, List, List],
    alpha: Any,
    t_grid: torch.Tensor,
    y_finalstate: List[torch.Tensor],
    memory: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    r"""
    Backward integration for Euler method without full history dependency.

    Since Euler has no memory term, the backward pass simply integrates the adjoint
    equation using the reconstructed forward trajectory (or re-evaluation).

    Args:
        ode_func: Augmented dynamics function.
        y_aug: Initial augmented state `(dummy_y, adj_y0, adj_params0)`.
        alpha: Unused.
        t_grid: Flipped time grid.
        y_finalstate: Final state from forward pass (used as starting point for reconstruction if needed).
        memory: Unused.

    Returns:
        Tuple of `(final_adj_y, final_adj_params)`.
    """
    with torch.no_grad():
        N = len(t_grid)
        h = torch.abs((t_grid[-1] - t_grid[-2]))  # uniform step size
        h = float(h)

        _, adj_y0, adj_params0 = y_aug
        # 初始化
        # y_state = [x.detach().clone() for x in y_finalstate]
        # adj_y = [x.detach().clone() for x in adj_y0]
        # adj_params = tuple(p.detach().clone() for p in adj_params0) if adj_params0 else ()
        y_state = list(y_finalstate)
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

        # return [x.detach().clone() for x in adj_y0], [x.detach().clone() for x in adj_params0]

        for k in range(N - 1):
            tk = t_grid[k]

            # 调用 augmented dynamics
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # 更新状态
            for i in range(len(adj_y)):
                # adj_y[i] = adj_y[i] + h * vjp_y[i]
                # y_state[i] = y_state[i] - h * func_eval[i]
                adj_y[i].add_(vjp_y[i], alpha=h)
                y_state[i].add_(func_eval[i], alpha=-h)  # 注意这里是 -h（减法）

            # 更新参数梯度
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)  # 直接修改 tuple 中的张量

    return adj_y, adj_params


def forward_euler_w_history(
    ode_func: Callable,
    y0_tuple: Tuple[torch.Tensor, ...],
    alpha: Any,
    t_grid: torch.Tensor,
    memory: Optional[int] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Explicit Euler integration storing full history.

    Required for adjoint methods that expect a history list structure consistent with
    fractional solvers, even if the method itself is memory-less.

    Args:
        ode_func: Function $f(t, y)$.
        y0_tuple: Initial state tuple.
        alpha: Unused.
        t_grid: Time grid.
        memory: Unused.

    Returns:
        List of lists, where each inner list contains the trajectory of one state component.
    """
    assert isinstance(y0_tuple, tuple)
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    # Initialize history lists for each component

    # Clone initial values
    y_current = list(y0_tuple)
    y_history = [[] for _ in y0_tuple]

    # Euler integration: y_{k+1} = y_k + dt * f(t_k, y_k)
    for k in range(N - 1):
        tk = t_grid[k]
        dt = t_grid[k + 1] - t_grid[k]  # Scalar tensor, will broadcast automatically
        dy = ode_func(
            tk, tuple(y_current)
        )  # Expect tuple return, consistent with y structure
        # assert isinstance(dy, tuple) and len(dy) == len(y)

        # Update all integrated components except the last one
        for i in range(len(y_current)):
            y_current[i] = y_current[i] + dt * dy[i]
            # Final element is the output spike (pass-through)
            y_history[i].append(y_current[i])
    return y_history


def backward_euler_w_history(
    ode_func: Callable,
    y_aug: Tuple[List, List, List],
    alpha: Any,
    t_grid: torch.Tensor,
    yhistory: List[List[torch.Tensor]],
    memory: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    r"""
    Backward integration for Euler method using stored history.

    Iterates backwards through the provided `yhistory` to compute adjoint updates.

    Args:
        ode_func: Augmented dynamics.
        y_aug: Initial augmented state.
        alpha: Unused.
        t_grid: Flipped time grid.
        yhistory: Full forward trajectory.
        memory: Unused.

    Returns:
        Tuple of `(final_adj_y, final_adj_params)`.
    """
    with torch.no_grad():
        N = len(t_grid)
        h = torch.abs((t_grid[-1] - t_grid[-2]))  # uniform step size
        h = float(h)

        _, adj_y0, adj_params0 = y_aug

        adj_y = list(adj_y0)  # [y0.clone() for y0 in y0_tuple]
        adj_params = list(adj_params0)

        # return tuple(y_i.clone() for y_i in adj_y0), tuple(y_i.clone() for y_i in adj_params0)

        # for k in range(N - 1):
        #     tk = t_grid[k]
        #     y_state = list([y[-1-k] for y in yhistory])
        for k in range(N - 2):
            tk = t_grid[k + 1]
            # t_grid_flip = t_grid.flip(0) recal that t has been flipped already
            # y_state = list([y[-k - 1] for y in yhistory])
            y_state = list([y[-k - 2] for y in yhistory])
            # 调用 augmented dynamics
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))
            # vjp_y = tuple(torch.zeros_like(y_i) for y_i in adj_y)
            # vjp_params = tuple(torch.zeros_like(y_i) for y_i in adj_params)
            # 更新状态
            # for i in range(len(adj_y)):
            #     adj_y[i] = adj_y[i] + h * vjp_y[i]
            for i in range(len(adj_y)):
                adj_y[i].add_(vjp_y[i], alpha=h)

            # 更新参数梯度
            # if adj_params and vjp_params:
            #     adj_params = tuple(
            #         ap + h * vp for ap, vp in zip(adj_params, vjp_params)
            #     )
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)  # 直接修改 tuple 中的张量

    del yhistory, vjp_params, func_eval, vjp_y

    return adj_y, adj_params


def forward_gl(
    ode_func: Callable,
    y0_tuple: Tuple[torch.Tensor, ...],
    alpha: Union[float, torch.Tensor],
    t_grid: torch.Tensor,
    memory: Optional[int] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Forward Grünwald-Letnikov (GL) integration.

    Implements the Riemann-Liouville approximation:

    $$y_{k+1} = h^\alpha f(t_k, y_k) - \sum_{j=0}^{k} c_{k-j}^{(\alpha)} y_j$$

    where coefficients $c_j^{(\alpha)}$ are computed recursively.

    Args:
        ode_func: Function $f(t, y)$.
        y0_tuple: Initial state tuple.
        alpha: Fractional order $\alpha \in (0, 1)$.
        t_grid: Uniform time grid.
        memory: Max history length for truncation.

    Returns:
        List of lists containing the state trajectory.
    """
    assert isinstance(y0_tuple, tuple)
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = t_grid[-1] - t_grid[-2]  # uniform step size
    h_alpha = torch.pow(h, alpha)

    # GL coefficients: need up to c[N]
    c = torch.zeros(N + 1, dtype=dtype, device=device)
    c[0] = 1
    for j in range(1, N + 1):
        c[j] = (1 - (1 + alpha) / j) * c[j - 1]

    # Initialize with y_0 (clone to avoid modifying input)
    y_current = list(y0_tuple)  # tuple(y.clone() for y in y0_tuple)
    # History: y_history[i][j] stores y_j for component i
    # Initialize with y_0
    # y_history = [[y, ] for y in y0_tuple]
    y_history = [[] for y in y0_tuple]
    # y_history = [[y.clone()] for y in y0_tuple]

    for k in range(N - 1):
        t_k = t_grid[k]
        # Evaluate f(t_k, y_k)

        dy = ode_func(t_k, tuple(y_current))
        # assert isinstance(dy, tuple) and len(dy) == len(y_current)

        # Determine memory range
        if memory is None or memory == -1:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)

        start_idx = max(0, k + 1 - memory_length)

        # Update all integrated components
        for i in range(len(y_current)):
            # Accumulate: Σ c_{k+1-j} * y_j for j from start_idx to k
            if k > 0:
                convolution_sum = 0  # torch.zeros_like(y[i])
                for j in range(start_idx, k):
                    # GL coefficient for this lag
                    # convolution_sum = convolution_sum + c[k+1-j] * y_history[i][j]
                    convolution_sum = convolution_sum + c[k - j] * y_history[i][j]
                # here we assume at time k, we have k elements (without y0=0)
                # the most restrict formulation should be convolution_sum + c[k-j] * y_history[i][j]
                # which however seems do have have good numerical stability
            else:
                convolution_sum = 0

            # convolution_sum = None
            # for j in range(start_idx, k+1):
            # # # GL coefficient for this lag
            #     if convolution_sum is None:
            #         convolution_sum = c[k+1-j] * y_history[i][j]
            #     else:
            #         convolution_sum = convolution_sum + c[k+1-j] * y_history[i][j]

            # y_{k+1} = h^alpha * f_k - convolution_sum
            y_current[i] = h_alpha * dy[i] - convolution_sum
            y_history[i].append(y_current[i])

    return y_history


def backward_gl(
    ode_func: Callable,
    y_aug: Tuple[List, List, List],
    alpha: Union[float, torch.Tensor],
    t_grid: torch.Tensor,
    yhistory: List[List[torch.Tensor]],
    memory: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    r"""
    Backward Grünwald-Letnikov integration for adjoint sensitivity.

    Solves the adjoint equation using the same GL discretization structure,
    accumulating gradients from the future (which is the past in reversed time).

    Args:
        ode_func: Augmented dynamics.
        y_aug: Initial augmented state (at reversed $t=0$, i.e., forward $t=T$).
        alpha: Fractional order.
        t_grid: Flipped time grid.
        yhistory: Forward trajectory (accessed in reverse).
        memory: Memory truncation limit.

    Returns:
        Tuple of `(final_adj_y, final_adj_params)`.
    """
    with torch.no_grad():
        N = len(t_grid)
        h = torch.abs(t_grid[-1] - t_grid[-2])
        h_alpha = torch.pow(h, alpha)

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype

        # GL coefficients
        c = torch.zeros(N + 1, dtype=dtype, device=device)
        c[0] = 1
        for j in range(1, N + 1):
            c[j] = (1 - (1 + alpha) / j) * c[j - 1]

        # Initialize adjoint history lists for each component
        adjy_history = [
            [
                xx,
            ]
            for xx in adj_y0
        ]

        # Clone initial adjoint values
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

        for k in range(N - 2):
            tk = t_grid[k + 1]
            # t_grid_flip = t_grid.flip(0) recal that t has been flipped already
            # y_state = list([y[-k - 1] for y in yhistory])
            y_state = list([y[-k - 2] for y in yhistory])

            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # Determine memory range
            if memory is None or memory == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(memory, k + 1)

            start_idx = max(0, k + 1 - memory_length)

            # Update all adjoint components
            for i in range(len(adj_y)):
                # Calculate history sum

                if True:  # k > 0:
                    convolution_sum = 0  # torch.zeros_like(y[i])
                    for j in range(start_idx, k + 1):
                        # GL coefficient for this lag
                        convolution_sum = (
                            convolution_sum + c[k + 1 - j] * adjy_history[i][j]
                        )
                    # here we assume at time k, we have k elements (without y0=0)
                    # the most restrict formulation should be convolution_sum + c[k-j] * y_history[i][j]
                    # which however seems do have have good numerical stability
                else:
                    convolution_sum = 0

                # convolution_sum = None
                # for j in range(start_idx, k+1):
                #     # # GL coefficient for this lag
                #     if convolution_sum is None:
                #         convolution_sum = c[k + 1 - j] * adjy_history[i][j]
                #     else:
                #         convolution_sum = convolution_sum + c[k + 1  - j] * adjy_history[i][j]

                # Update adjoint state
                adj_y[i] = h_alpha * vjp_y[i] - convolution_sum
                adjy_history[i].append(adj_y[i])

            # 更新参数梯度
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)  # 直接修改 tuple 中的张量

    del adjy_history, yhistory

    return adj_y, adj_params


def forward_trap(
    ode_func: Callable,
    y0_tuple: Tuple[torch.Tensor, ...],
    alpha: Union[float, torch.Tensor],
    t_grid: torch.Tensor,
    memory: Optional[int] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Forward Product Trapezoidal method.

    Provides $O(h^2)$ accuracy for Riemann-Liouville FDEs.
    Formula:

    $$y_{k+1} = \frac{h^\alpha}{\Gamma(2-\alpha)} f_k - \sum_{j=0}^{k} A_{j,k+1} y_j$$

    where weights $A_{j,k+1}$ depend on the distance from the current step.

    Args:
        ode_func: Function $f(t, y)$.
        y0_tuple: Initial state.
        alpha: Fractional order.
        t_grid: Time grid.
        memory: Memory limit.

    Returns:
        State trajectory.
    """
    assert isinstance(y0_tuple, tuple)

    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    h = t_grid[-1] - t_grid[-2]  # uniform step size
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    # Initialize with y_0
    y_current = list(y0_tuple)
    y_history = [[] for y in y0_tuple]

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Determine memory range
        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute A_{j,k+1} weights for indices from start_idx to k
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

        # General formula for j >= 1:
        # A_{j,k+1} = (k+2-j)^{1-α} + (k-j)^{1-α} - 2(k+1-j)^{1-α}
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        A_j_kp1 = kjp2 + kj - 2 * kjp1

        # Special handling for j=0 if it's in the range:
        # A_{0,k+1} = k^{1-α} - (k+α)(k+1)^{-α}
        if start_idx == 0:
            k_power = torch.pow(
                torch.tensor(k, dtype=dtype, device=device), one_minus_alpha
            )
            kp1_neg_alpha = torch.pow(
                torch.tensor(k + 1, dtype=dtype, device=device), -alpha
            )
            A_j_kp1[0] = k_power - (k + alpha) * kp1_neg_alpha

        # Update ALL state components (forward integrates all, not len-1)
        for i in range(len(y_current)):
            # Compute convolution sum: sum_{j=start_idx}^{k-1} A_{j,k+1} * y_j[i]
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):
                    # local_idx = j - start_idx
                    local_idx = j - start_idx + 1
                    # the most restrict formulation should be local_idx = j - start_idx + 1
                    convolution_sum = (
                        convolution_sum + A_j_kp1[local_idx] * y_history[i][j]
                    )
            else:
                convolution_sum = 0

            # y_{k+1} = Γ(2-α) * h^α * f_k - convolution_sum
            y_current[i] = h_alpha_gamma * f_k[i] - convolution_sum
            y_history[i].append(y_current[i])

    return y_history


def backward_trap(
    ode_func: Callable,
    y_aug: Tuple[List, List, List],
    alpha: Union[float, torch.Tensor],
    t_grid: torch.Tensor,
    yhistory: List[List[torch.Tensor]],
    memory: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    r"""
    Backward Product Trapezoidal method for adjoint sensitivity.

    Args:
        ode_func: Augmented dynamics.
        y_aug: Initial augmented state.
        alpha: Fractional order.
        t_grid: Flipped time grid.
        yhistory: Forward trajectory.
        memory: Memory limit.

    Returns:
        Final adjoint states and parameter gradients.
    """
    with torch.no_grad():
        N = len(t_grid)
        h = torch.abs(t_grid[-1] - t_grid[-2])
        h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
        one_minus_alpha = 1 - alpha

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype

        # Initialize adjoint history lists for each component (with initial values)
        adjy_history = [
            [
                xx,
            ]
            for xx in adj_y0
        ]

        # Clone initial adjoint values
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

        for k in range(N - 2):
            tk = t_grid[k + 1]
            # y_state = list([y[-k - 1] for y in yhistory])
            y_state = list([y[-k - 2] for y in yhistory])

            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # Determine memory range
            if memory is None:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)

            start_idx = max(0, k + 1 - memory_length)

            # Compute A_{j,k+1} weights for indices from start_idx to k
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

            # General formula for j >= 1:
            # A_{j,k+1} = (k+2-j)^{1-α} + (k-j)^{1-α} - 2(k+1-j)^{1-α}
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
            kj = torch.pow(k - j_vals, one_minus_alpha)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
            A_j_kp1 = kjp2 + kj - 2 * kjp1

            # Special handling for j=0 if it's in the range:
            # A_{0,k+1} = k^{1-α} - (k+α)(k+1)^{-α}
            if start_idx == 0:
                k_power = torch.pow(
                    torch.tensor(k, dtype=dtype, device=device), one_minus_alpha
                )
                kp1_neg_alpha = torch.pow(
                    torch.tensor(k + 1, dtype=dtype, device=device), -alpha
                )
                A_j_kp1[0] = k_power - (k + alpha) * kp1_neg_alpha

            # Update all adjoint components
            for i in range(len(adj_y)):
                # Calculate history sum - note: range goes to k+1 (one more than forward)
                convolution_sum = 0
                for j in range(start_idx, k + 1):
                    local_idx = j - start_idx
                    convolution_sum = (
                        convolution_sum + A_j_kp1[local_idx] * adjy_history[i][j]
                    )

                # Update adjoint state
                adj_y[i] = h_alpha_gamma * vjp_y[i] - convolution_sum
                adjy_history[i].append(adj_y[i])

            # Update parameter gradients
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

    del adjy_history, yhistory

    return adj_y, adj_params


def forward_l1(
    ode_func: Callable,
    y0_tuple: Tuple[torch.Tensor, ...],
    alpha: Union[float, torch.Tensor],
    t_grid: torch.Tensor,
    memory: Optional[int] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Forward L1 scheme for Caputo FDEs.

    Accuracy $O(h^{2-\alpha})$.
    Formula:

    $$y_{k+1} = \frac{h^\alpha}{\Gamma(2-\alpha)} f_k - \sum_{j=0}^{k} c_j^{(k)} y_j$$

    Args:
        ode_func: Function $f(t, y)$.
        y0_tuple: Initial state.
        alpha: Fractional order.
        t_grid: Time grid.
        memory: Memory limit.

    Returns:
        State trajectory.
    """
    assert isinstance(y0_tuple, tuple)
    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = t_grid[-1] - t_grid[-2]  # uniform step size
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]
    # Current state
    y_current = list(y0_tuple)

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]
        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, y_current)

        # Determine memory range
        if memory is None:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute c_j^(k) weights for indices from start_idx to k
        # General formula for j >= 1: c_j^(k) = (k-j+2)^{1-α} - 2(k-j+1)^{1-α} + (k-j)^{1-α}
        # Special case for j = 0: c_0^(k) = -[(k+1)^{1-α} - k^{1-α}]
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        c_j_k = kjp2 - 2 * kjp1 + kj

        # Special handling for j=0 if it's in the range
        if start_idx == 0:
            kp1_power = torch.pow(
                torch.tensor(k + 1, dtype=dtype, device=device), one_minus_alpha
            )
            k_power = torch.pow(
                torch.tensor(k, dtype=dtype, device=device), one_minus_alpha
            )
            c_j_k[0] = -(kp1_power - k_power)

        # Update ALL state components (forward integrates all, not len-1)
        for i in range(len(y_current)):
            # Compute convolution sum: sum_{j=start_idx}^{k-1} c_j^(k) * y_j[i]
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):
                    # local_idx = j - start_idx
                    local_idx = j - start_idx + 1
                    # the most restrict formulation should be local_idx = j - start_idx + 1
                    convolution_sum = (
                        convolution_sum + c_j_k[local_idx] * y_history[i][j]
                    )
            else:
                convolution_sum = 0

            # y_{k+1} = h^α * Γ(2-α) * f_k - convolution_sum
            y_current[i] = h_alpha_gamma * f_k[i] - convolution_sum
            y_history[i].append(y_current[i])

    return y_history


def backward_l1(
    ode_func: Callable,
    y_aug: Tuple[List, List, List],
    alpha: Union[float, torch.Tensor],
    t_grid: torch.Tensor,
    yhistory: List[List[torch.Tensor]],
    memory: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    r"""
    Backward L1 scheme for adjoint sensitivity.

    Args:
        ode_func: Augmented dynamics.
        y_aug: Initial augmented state.
        alpha: Fractional order.
        t_grid: Flipped time grid.
        yhistory: Forward trajectory.
        memory: Memory limit.

    Returns:
        Final adjoint states and parameter gradients.
    """
    with torch.no_grad():
        N = len(t_grid)
        h = torch.abs(t_grid[-1] - t_grid[-2])
        h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
        one_minus_alpha = 1 - alpha

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype

        # Initialize adjoint history lists for each component (with initial values)
        adjy_history = [
            [
                xx,
            ]
            for xx in adj_y0
        ]

        # Clone initial adjoint values
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

        for k in range(N - 2):
            tk = t_grid[k + 1]
            y_state = list([y[-k - 2] for y in yhistory])

            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # Determine memory range
            if memory is None:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)

            start_idx = max(0, k + 1 - memory_length)

            # Compute c_j^(k) weights for indices from start_idx to k
            # General formula for j >= 1: c_j^(k) = (k-j+2)^{1-α} - 2(k-j+1)^{1-α} + (k-j)^{1-α}
            # Special case for j = 0: c_0^(k) = -[(k+1)^{1-α} - k^{1-α}]
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
            kj = torch.pow(k - j_vals, one_minus_alpha)
            c_j_k = kjp2 - 2 * kjp1 + kj

            # Special handling for j=0 if it's in the range
            if start_idx == 0:
                kp1_power = torch.pow(
                    torch.tensor(k + 1, dtype=dtype, device=device), one_minus_alpha
                )
                k_power = torch.pow(
                    torch.tensor(k, dtype=dtype, device=device), one_minus_alpha
                )
                c_j_k[0] = -(kp1_power - k_power)

            # Update all adjoint components
            for i in range(len(adj_y)):
                # Calculate history sum - note: range goes to k+1 (one more than forward)
                convolution_sum = 0
                for j in range(start_idx, k + 1):
                    local_idx = j - start_idx
                    convolution_sum = (
                        convolution_sum + c_j_k[local_idx] * adjy_history[i][j]
                    )

                # Update adjoint state
                adj_y[i] = h_alpha_gamma * vjp_y[i] - convolution_sum
                adjy_history[i].append(adj_y[i])

            # Update parameter gradients
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

    del adjy_history, yhistory

    return adj_y, adj_params


def forward_pred(
    ode_func: Callable,
    y0_tuple: Tuple[torch.Tensor, ...],
    alpha: Union[float, torch.Tensor],
    t_grid: torch.Tensor,
    memory: Optional[int] = None,
) -> List[List[torch.Tensor]]:
    r"""
    Forward Adams-Bashforth predictor method.

    Uses history of function evaluations $f(t, y)$ instead of states $y$.
    Formula:

    $$y_{k+1} = \sum_{j=0}^{k} b_{j,k+1} f(t_j, y_j)$$

    where $b_{j,k+1} = \frac{h^\alpha}{\alpha \Gamma(\alpha)} [(k+1-j)^\alpha - (k-j)^\alpha]$.

    Args:
        ode_func: Function $f(t, y)$.
        y0_tuple: Initial state.
        alpha: Fractional order.
        t_grid: Time grid.
        memory: Memory limit.

    Returns:
        State trajectory.
    """
    assert isinstance(y0_tuple, tuple)
    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = t_grid[-1] - t_grid[-2]  # uniform step size
    # gamma_alpha = 1 / math.gamma(alpha)
    h_alpha_over_alpha = torch.pow(h, alpha) / (alpha * math.gamma(alpha))

    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]
    # History for function evaluations (for ALL components in forward)
    fhistory = [[] for _ in y0_tuple]
    # Current state
    y_current = list(y0_tuple)

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]
        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, y_current)

        # Store function evaluations for ALL components
        for i in range(len(y_current)):
            fhistory[i].append(f_k[i])

        # Determine memory range
        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute weights b_{j,k+1} for indices from start_idx to k
        # b_{j,k+1} = (h^α / α) * [(k+1-j)^α - (k-j)^α]
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
        b_j_kp1 = h_alpha_over_alpha * (
            torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
        )

        # Update ALL state components
        for i in range(len(y_current)):
            # Compute convolution sum: sum_{j=start_idx}^{k} b_{j,k+1} * f_j[i]
            convolution_sum = 0
            for j in range(start_idx, k + 1):
                local_idx = j - start_idx
                convolution_sum = convolution_sum + b_j_kp1[local_idx] * fhistory[i][j]

            # y_{k+1} = (1/Γ(α)) * convolution_sum
            y_current[i] = convolution_sum
            y_history[i].append(y_current[i])

    del fhistory
    return y_history


def backward_pred(
    ode_func: Callable,
    y_aug: Tuple[List, List, List],
    alpha: Union[float, torch.Tensor],
    t_grid: torch.Tensor,
    yhistory: List[List[torch.Tensor]],
    memory: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    r"""
    Backward Adams-Bashforth predictor method.

    Args:
        ode_func: Augmented dynamics.
        y_aug: Initial augmented state.
        alpha: Fractional order.
        t_grid: Flipped time grid.
        yhistory: Forward trajectory.
        memory: Memory limit.

    Returns:
        Final adjoint states and parameter gradients.
    """
    with torch.no_grad():
        N = len(t_grid)
        h = torch.abs(t_grid[-1] - t_grid[-2])
        # gamma_alpha = 1 / math.gamma(alpha)
        h_alpha_over_alpha = torch.pow(h, alpha) / (alpha * math.gamma(alpha))

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype

        # Initialize adjoint history lists with initial values
        adjf_history = [
            [
                xx,
            ]
            for xx in adj_y0
        ]

        # Clone initial adjoint values
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

        for k in range(N - 2):
            tk = t_grid[k + 1]
            y_state = list([y[-k - 2] for y in yhistory])

            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # Store adjoint of function evaluation
            for i in range(len(adj_y)):
                adjf_history[i].append(vjp_y[i])

            # Determine memory range
            if memory is None:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)

            start_idx = max(0, k + 1 - memory_length)

            # Compute weights b_{j,k+1}
            # b_{j,k+1} = (h^α / α) * [(k+1-j)^α - (k-j)^α]
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
            b_j_kp1 = h_alpha_over_alpha * (
                torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
            )

            # Update all adjoint components
            for i in range(len(adj_y)):
                # Compute convolution sum over adjoint history
                convolution_sum = 0
                for j in range(start_idx, k + 1):
                    local_idx = j - start_idx
                    convolution_sum = (
                        convolution_sum + b_j_kp1[local_idx] * adjf_history[i][j]
                    )

                # Update adjoint state
                adj_y[i] = adj_y0[i] + convolution_sum

            # Update parameter gradients
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

    del adjf_history, yhistory

    return adj_y, adj_params


def find_parameters(module: nn.Module) -> List[torch.Tensor]:
    r"""
    Extracts all trainable parameters from a PyTorch module.

    Handles special cases such as `DataParallel` replicas where parameters might not
    be registered in the standard `.parameters()` iterator.

    Args:
        module: The `nn.Module` to inspect.

    Returns:
        A list of `torch.Tensor` parameters that require gradients.
    """

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, "_is_replica", False):

        def find_tensor_attributes(module):
            tuples = [
                (k, v)
                for k, v in module.__dict__.items()
                if torch.is_tensor(v) and v.requires_grad
            ]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


#### it seems torch compile cannot speed up the methods

forward_gl_compiled = torch.compile(forward_gl)
backward_gl_compiled = torch.compile(backward_gl)
forward_trap_compiled = torch.compile(forward_trap)
backward_trap_compiled = torch.compile(backward_trap)
forward_l1_compiled = torch.compile(forward_l1)
backward_l1_compiled = torch.compile(backward_l1)
forward_pred_compiled = torch.compile(forward_pred)
backward_pred_compiled = torch.compile(backward_pred)


forward_euler_w_history_compiled = torch.compile(forward_euler_w_history)
forward_euler_wo_history_compiled = torch.compile(forward_euler_wo_history)
backward_euler_w_history_compiled = torch.compile(backward_euler_w_history)
backward_euler_wo_history_compiled = torch.compile(backward_euler_wo_history)


SOLVERS_Forward = {
    "gl-f": forward_gl_compiled,
    "gl-o": forward_gl_compiled,
    "trap-f": forward_trap_compiled,
    "trap-o": forward_trap_compiled,
    "l1-f": forward_l1_compiled,
    "l1-o": forward_l1_compiled,
    "pred-f": forward_pred_compiled,
    "pred-o": forward_pred_compiled,
    "euler": forward_euler_wo_history_compiled,
}

SOLVERS_Backward = {
    "gl-f": backward_gl_compiled,
    "gl-o": backward_euler_w_history_compiled,
    "trap-f": backward_trap_compiled,
    "trap-o": backward_euler_w_history_compiled,
    "l1-f": backward_trap_compiled,
    "l1-o": backward_euler_w_history_compiled,
    "pred-f": backward_pred_compiled,
    "pred-o": backward_euler_w_history_compiled,
    "euler": backward_euler_wo_history_compiled,
}
