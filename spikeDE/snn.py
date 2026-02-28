import torch
import torch.nn as nn
from functools import wraps
from typing import Optional, List, Union, Any, Tuple, Callable, Dict
from torch.fx.interpreter import Interpreter

# Import solvers
from .solver import (
    euler_integrate_tuple,
    step_dynamics,
    gl_integrate_tuple,
    trap_integrate_tuple,
    pred_integrate_tuple,
    l1_integrate_tuple,
    glmethod_multiterm_integrate_tuple,
    _broadcast_alpha_to_layers,
    fdeint_adjoint,
    SOLVERS,
)
from .neuron import BaseNeuron, LIFNeuron, IFNeuron
from .odefunc import ODEFuncFromFX

euler_integrate_tuple_compiled = torch.compile(euler_integrate_tuple)


# euler_integrate_any_compiled = torch.compile(euler_integrate_any)

"""
Configuration for per-layer alpha values.

See docs/per_layer_alpha.md and examples/example_per_layer_alpha.py for more details.

Three configuration modes are supported:

A) Per-layer Single-term Mode:
   Each layer has a single alpha value.
   Example: alpha = [0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.5]  # Length = n_layers
   Required: alpha_mode = 'per_layer'

B) Multi-term Broadcast Mode:
   The same multi-term configuration applies to all layers.
   Example: alpha = [0.3, 0.5, 0.7]  # Broadcast to all layers
   Required: alpha_mode = 'multiterm'
   Optional: multi_coefficient = [1.0, 0.5, 0.2] (defaults to ones)

C) Per-layer Multi-term Mode (Mixed):
   Each layer has its own multi-term configuration.
   Example: alpha = [[0.3, 0.5], [0.4, 0.6, 0.8], 0.5, ...]  # May contain nested lists
   Note: Detected automatically when alpha contains nested structures.
   Optional: multi_coefficient follows the same nested structure (defaults to ones if set as None).

The alpha_mode parameter explicitly disambiguates Case A vs Case B when
alpha is a flat list. This is necessary because len(alpha) could equal
n_layers in both cases.
"""


class PerLayerAlphaConfig:
    r"""
    Configuration parser and validator for per-layer fractional orders ($\alpha$).

    This class normalizes user inputs into a consistent internal format based on a
    decision table logic. It handles three primary configuration cases:

    1. **Case A (Per-layer Single-term)**: Each layer has a single $\alpha$ value.
       Example: `alpha=[0.3, 0.5]` with `alpha_mode='per_layer'`.

    2. **Case B (Multi-term Broadcast)**: The same multi-term configuration applies to all layers.
       Example: `alpha=[0.3, 0.5, 0.7]` with `alpha_mode='multiterm'`.

    3. **Case C (Per-layer Multi-term)**: Each layer has its own multi-term configuration.
       Example: `alpha=[[0.3, 0.5], [0.4, 0.6]]` (auto-detected).

    Attributes:
        n_layers (int): Number of neuron layers in the network.
        case (str): Detected configuration case.
        per_layer_alpha (List[List[float]]): Normalized alpha values per layer.
        per_layer_coefficient (List[List[float]]): Normalized coefficients per layer.
        per_layer_is_multi_term (List[bool]): Boolean flag per layer indicating multi-term usage.
        per_layer_learn_alpha (List[bool]): Learnable flag for alpha per layer.
        per_layer_learn_coefficient (List[bool]): Learnable flag for coefficients per layer.
    """

    def __init__(
        self,
        alpha: Union[float, List[float], List[List[float]]],
        n_layers: int,
        multi_coefficient: Optional[Union[List[float], List[List[float]]]] = None,
        learn_alpha: Union[bool, List[bool]] = False,
        learn_coefficient: Union[bool, List[bool]] = False,
        alpha_mode: str = "auto",
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        r"""
        Initialize the Per-Layer Alpha Configuration.

        Args:
            alpha: Fractional order(s). Accepts:

                - `float`: Single value for all layers.
                - `List[float]`: Interpretation depends on `alpha_mode`.
                - `List[List[float]]`: Per-layer multi-term configuration.
                - `torch.Tensor`: Will be converted to list.
            n_layers: Total number of neuron layers in the SNN.
            multi_coefficient: Coefficients for multi-term FDEs. Accepts:

                - `None`: Defaults to ones.
                - `List[float]`: Shared coefficients for all layers (Case B).
                - `List[List[float]]`: Per-layer coefficients (Case C).
            learn_alpha: Whether $\alpha$ values are learnable parameters.

                - `bool`: Applied globally to all layers.
                - `List[bool]`: Per-layer learnable flags.
            learn_coefficient: Whether coefficients are learnable parameters.

                - `bool`: Applied globally.
                - `List[bool]`: Per-layer learnable flags.
            alpha_mode: Disambiguation mode for flat list inputs.

                - `'auto'`: Automatic detection (default).
                - `'per_layer'`: Force Case A.
                - `'multiterm'`: Force Case B.
            device: Target device for tensors.
            dtype: Data type for tensors (default: float32).

        Raises:
            ValueError: If input dimensions mismatch, invalid modes are provided,
                        or coefficient lengths do not match alpha lengths.
        """
        self.n_layers = n_layers
        self.device = device
        self.dtype = dtype

        if alpha_mode not in ("auto", "per_layer", "multiterm"):
            raise ValueError(
                f"alpha_mode must be 'auto', 'per_layer', or 'multiterm', got '{alpha_mode}'"
            )

        # Parse using decision table
        self.case, self.per_layer_alpha, self.per_layer_coefficient = self._parse(
            alpha, multi_coefficient, n_layers, alpha_mode
        )

        # Determine is_multi_term per layer
        self.per_layer_is_multi_term = [len(a) > 1 for a in self.per_layer_alpha]

        # Parse learnable flags
        self._parse_learn_flags(learn_alpha, learn_coefficient)

    # =========================================================================
    # Step 1: Type detection helpers
    # =========================================================================

    def _get_alpha_type(self, alpha: Any) -> Tuple[str, Any]:
        """
        Determine the type of alpha input and normalize it.

        Args:
            alpha: Raw input alpha value.

        Returns:
            tuple: `(alpha_type, normalized_alpha)` where `alpha_type` is
                   'scalar', 'nested', or 'flat'.
        """
        # Scalar float/int
        if isinstance(alpha, (int, float)):
            return "scalar", float(alpha)

        # Tensor
        if isinstance(alpha, torch.Tensor):
            if alpha.numel() == 1:
                return "scalar", float(alpha.item())
            else:
                return self._get_alpha_type(alpha.tolist())  # Recurse as list

        # List/tuple
        if isinstance(alpha, (list, tuple)):
            if len(alpha) == 0:
                raise ValueError("alpha cannot be empty")

            # Check if nested (contains any list/tuple)
            if any(isinstance(x, (list, tuple)) for x in alpha):
                return "nested", alpha
            else:
                # Flat list - normalize to floats
                return "flat", [float(x) for x in alpha]

        raise ValueError(f"alpha must be float, list, or tensor, got {type(alpha)}")

    def _get_coef_type(self, multi_coefficient: Any) -> Tuple[str, Any]:
        """
        Determine the type of coefficient input and normalize it.

        Args:
            multi_coefficient: Raw input coefficient value.

        Returns:
            tuple: `(coef_type, normalized_coef)` where `coef_type` is
                   'none', 'nested', or 'flat'.
        """
        if multi_coefficient is None:
            return "none", None

        if isinstance(multi_coefficient, torch.Tensor):
            multi_coefficient = multi_coefficient.tolist()

        if not isinstance(multi_coefficient, (list, tuple)):
            raise ValueError(
                f"multi_coefficient must be None or list, got {type(multi_coefficient)}"
            )

        if len(multi_coefficient) == 0:
            raise ValueError("multi_coefficient cannot be empty")

        # Check if nested
        if any(isinstance(x, (list, tuple)) for x in multi_coefficient):
            return "nested", multi_coefficient
        else:
            return "flat", [float(x) for x in multi_coefficient]

    # =========================================================================
    # Step 2: Main parsing logic (follows decision table)
    # =========================================================================

    def _parse(
        self, alpha: Any, multi_coefficient: Any, n_layers: int, alpha_mode: str
    ) -> Tuple[str, List[List[float]], List[List[float]]]:
        """
        Main parsing logic following the configuration decision table.

        Routes the input to the appropriate builder method based on input types
        and the specified alpha_mode.

        Returns:
            tuple: `(case_name, per_layer_alpha, per_layer_coefficient)`
        """
        import warnings

        # Step 1: Get types
        alpha_type, alpha_normalized = self._get_alpha_type(alpha)
        coef_type, coef_normalized = self._get_coef_type(multi_coefficient)

        # Step 2: Apply decision table

        # Row 1: scalar alpha → always A_scalar
        if alpha_type == "scalar":
            return self._build_case_a_scalar(alpha_normalized, n_layers)

        # Row 2: nested alpha → always Case C
        if alpha_type == "nested":
            return self._build_case_c(alpha_normalized, coef_normalized, n_layers)

        # Row 3+: flat alpha - depends on alpha_mode and coef_type
        assert alpha_type == "flat"
        alpha_list = alpha_normalized

        if alpha_mode == "per_layer":
            # Force Case A
            if len(alpha_list) != n_layers:
                raise ValueError(
                    f"alpha_mode='per_layer' requires len(alpha)==n_layers. "
                    f"Got {len(alpha_list)} alphas but {n_layers} layers."
                )
            return self._build_case_a(alpha_list, n_layers)

        elif alpha_mode == "multiterm":
            # Force Case B (with or without per-layer coefficients)
            if coef_type == "nested":
                return self._build_case_b_per_layer_coef(
                    alpha_list, coef_normalized, n_layers
                )
            else:
                # coef_type is 'none' or 'flat'
                if coef_type == "flat" and len(coef_normalized) != len(alpha_list):
                    raise ValueError(
                        f"alpha_mode='multiterm': coefficient length ({len(coef_normalized)}) "
                        f"must match alpha length ({len(alpha_list)})"
                    )
                return self._build_case_b(alpha_list, coef_normalized, n_layers)

        else:  # alpha_mode == 'auto'
            return self._parse_auto(
                alpha_list, coef_type, coef_normalized, n_layers, warnings
            )

    def _parse_auto(
        self,
        alpha_list: List[float],
        coef_type: str,
        coef_normalized: Any,
        n_layers: int,
        warnings: Any,
    ) -> Tuple[str, List[List[float]], List[List[float]]]:
        """
        Auto-detection logic for flat alpha inputs.
        """
        if coef_type == "flat":
            # Flat coefficient provided → Case B
            if len(coef_normalized) != len(alpha_list):
                raise ValueError(
                    f"multi_coefficient length ({len(coef_normalized)}) must match "
                    f"alpha length ({len(alpha_list)})"
                )
            return self._build_case_b(alpha_list, coef_normalized, n_layers)

        elif coef_type == "nested":
            # Nested coefficient with flat alpha → Case B
            return self._build_case_b_per_layer_coef(
                alpha_list, coef_normalized, n_layers
            )

        else:  # coef_type == 'none'
            # No coefficient - use length heuristic
            if len(alpha_list) == n_layers:
                warnings.warn(
                    f"Ambiguous: len(alpha)={len(alpha_list)} == n_layers={n_layers}. "
                    f"Defaulting to Case A (per-layer single-term). "
                    f"Use alpha_mode='multiterm' for Case B.",
                    UserWarning,
                )
                return self._build_case_a(alpha_list, n_layers)
            else:
                warnings.warn(
                    f"len(alpha)={len(alpha_list)} != n_layers={n_layers}. "
                    f"Treating as Case B (multi-term broadcast). "
                    f"Use alpha_mode='multiterm' to suppress this warning.",
                    UserWarning,
                )
                return self._build_case_b(alpha_list, None, n_layers)

    # =========================================================================
    # Step 3: Case builders
    # =========================================================================

    def _build_case_a_scalar(
        self, alpha_val: float, n_layers: int
    ) -> Tuple[str, List[List[float]], List[List[float]]]:
        """Case A_scalar: single alpha broadcast to all layers."""
        per_layer_alpha = [[alpha_val]] * n_layers
        per_layer_coef = [
            [1.0]
        ] * n_layers  ## however this coef will not be used in solvers

        print(f"[Alpha Config] Case A_scalar: Single alpha broadcast")
        print(f"  Alpha = {alpha_val} for all {n_layers} layers")

        return "A_scalar", per_layer_alpha, per_layer_coef

    def _build_case_a(
        self, alpha_list: List[float], n_layers: int
    ) -> Tuple[str, List[List[float]], List[List[float]]]:
        """Case A: per-layer single-term."""
        per_layer_alpha = [[a] for a in alpha_list]
        per_layer_coef = [
            [1.0]
        ] * n_layers  ## however this coef will not be used in solvers

        print(f"[Alpha Config] Case A: Per-layer single-term")
        print(f"  Alphas: {alpha_list}")

        return "A", per_layer_alpha, per_layer_coef

    def _build_case_b(
        self, alpha_list: List[float], coef_list: Optional[List[float]], n_layers: int
    ) -> Tuple[str, List[List[float]], List[List[float]]]:
        """Case B: multi-term broadcast to all layers."""
        n_terms = len(alpha_list)

        if coef_list is None:
            coef_list = [1.0] * n_terms

        per_layer_alpha = [list(alpha_list)] * n_layers
        per_layer_coef = [list(coef_list)] * n_layers

        print(f"[Alpha Config] Case B: Multi-term broadcast")
        print(f"  Alpha (all layers): {alpha_list}")
        print(f"  Coefficient (all layers): {coef_list}")

        return "B", per_layer_alpha, per_layer_coef

    def _build_case_b_per_layer_coef(
        self, alpha_list: List[float], coef_nested: List[Any], n_layers: int
    ) -> Tuple[str, List[List[float]], List[List[float]]]:
        """Case B variant: multi-term alpha broadcast, per-layer coefficients."""
        n_terms = len(alpha_list)

        if len(coef_nested) != n_layers:
            raise ValueError(
                f"Nested multi_coefficient must have {n_layers} elements, got {len(coef_nested)}"
            )

        per_layer_alpha = [list(alpha_list)] * n_layers
        per_layer_coef = []

        for i, coef in enumerate(coef_nested):
            if coef is None:
                per_layer_coef.append([1.0] * n_terms)
            elif isinstance(coef, (int, float)):
                per_layer_coef.append([float(coef)] * n_terms)
            else:
                if len(coef) != n_terms:
                    raise ValueError(
                        f"multi_coefficient[{i}] has {len(coef)} elements, "
                        f"expected {n_terms} to match alpha"
                    )
                per_layer_coef.append([float(c) for c in coef])

        print(f"[Alpha Config] Case B (per-layer coef): Multi-term broadcast")
        print(f"  Alpha (all layers): {alpha_list}")
        for i, coef in enumerate(per_layer_coef):
            print(f"  Layer {i} coef: {coef}")

        return "B", per_layer_alpha, per_layer_coef

    def _build_case_c(
        self, alpha_nested: List[Any], coef_input: Any, n_layers: int
    ) -> Tuple[str, List[List[float]], List[List[float]]]:
        """Case C: per-layer multi-term (mixed)."""
        if len(alpha_nested) != n_layers:
            raise ValueError(
                f"Nested alpha must have {n_layers} elements, got {len(alpha_nested)}"
            )

        # Normalize alpha: scalars → [scalar]
        per_layer_alpha = []
        for i, a in enumerate(alpha_nested):
            if isinstance(a, (int, float)):
                per_layer_alpha.append([float(a)])
            elif isinstance(a, (list, tuple)):
                per_layer_alpha.append([float(x) for x in a])
            elif isinstance(a, torch.Tensor):
                per_layer_alpha.append(
                    [float(x) for x in a.tolist()]
                    if a.numel() > 1
                    else [float(a.item())]
                )
            else:
                raise ValueError(f"Invalid alpha[{i}] type: {type(a)}")

        # Parse coefficients
        per_layer_coef = []
        if coef_input is None:
            # Default: ones matching each layer's alpha length
            for i in range(n_layers):
                per_layer_coef.append([1.0] * len(per_layer_alpha[i]))
        else:
            if len(coef_input) != n_layers:
                raise ValueError(
                    f"multi_coefficient must have {n_layers} elements for Case C, got {len(coef_input)}"
                )
            for i, coef in enumerate(coef_input):
                n_terms_i = len(per_layer_alpha[i])
                if coef is None:
                    per_layer_coef.append([1.0] * n_terms_i)
                elif isinstance(coef, (int, float)):
                    per_layer_coef.append([float(coef)] * n_terms_i)
                elif isinstance(coef, (list, tuple)):
                    if len(coef) != n_terms_i:
                        raise ValueError(
                            f"multi_coefficient[{i}] has {len(coef)} elements, "
                            f"expected {n_terms_i} to match alpha[{i}]"
                        )
                    per_layer_coef.append([float(c) for c in coef])
                else:
                    raise ValueError(
                        f"Invalid multi_coefficient[{i}] type: {type(coef)}"
                    )

        print(f"[Alpha Config] Case C: Per-layer multi-term")
        for i in range(n_layers):
            print(f"  Layer {i}: α={per_layer_alpha[i]}, coef={per_layer_coef[i]}")

        return "C", per_layer_alpha, per_layer_coef

    # =========================================================================
    # Learnable flags parsing
    # =========================================================================

    def _parse_learn_flags(self, learn_alpha: Any, learn_coefficient: Any) -> None:
        """Parse learnable flags into per-layer format."""
        if isinstance(learn_alpha, bool):
            self.per_layer_learn_alpha = [learn_alpha] * self.n_layers
        else:
            if len(learn_alpha) != self.n_layers:
                raise ValueError(f"learn_alpha list must have {self.n_layers} elements")
            self.per_layer_learn_alpha = list(learn_alpha)

        if isinstance(learn_coefficient, bool):
            self.per_layer_learn_coefficient = [learn_coefficient] * self.n_layers
        else:
            if len(learn_coefficient) != self.n_layers:
                raise ValueError(
                    f"learn_coefficient list must have {self.n_layers} elements"
                )
            self.per_layer_learn_coefficient = list(learn_coefficient)

    # =========================================================================
    # Parameter registration
    # =========================================================================

    def register_parameters(self, module: nn.Module) -> None:
        """
        Register alpha and coefficient as parameters or buffers in the given module.

        Args:
            module: The nn.Module to register parameters into.
        """
        module.per_layer_alpha_params = nn.ParameterList()
        module.per_layer_coefficient_params = nn.ParameterList()
        module._alpha_is_param = []
        module._coef_is_param = []

        for i in range(self.n_layers):
            alpha_vals = self.per_layer_alpha[i]
            coef_vals = self.per_layer_coefficient[i]
            learn_alpha = self.per_layer_learn_alpha[i]
            learn_coef = self.per_layer_learn_coefficient[i]

            # Create tensors
            alpha_tensor = torch.tensor(alpha_vals, dtype=self.dtype)
            coef_tensor = torch.tensor(coef_vals, dtype=self.dtype)

            # Register alpha
            if learn_alpha:
                module.per_layer_alpha_params.append(nn.Parameter(alpha_tensor))
                #
                module._alpha_is_param.append(True)
            else:
                module.per_layer_alpha_params.append(
                    nn.Parameter(alpha_tensor, requires_grad=False)
                )
                module._alpha_is_param.append(False)

            # Register coefficient
            if learn_coef:
                module.per_layer_coefficient_params.append(nn.Parameter(coef_tensor))
                module._coef_is_param.append(True)
            else:
                module.per_layer_coefficient_params.append(
                    nn.Parameter(coef_tensor, requires_grad=False)
                )
                module._coef_is_param.append(False)

        # Store metadata
        module._per_layer_is_multi_term = self.per_layer_is_multi_term.copy()
        module._alpha_case = self.case

    def print_config(self):
        """Print configuration summary to stdout."""
        print(f"\n[Per-Layer Alpha Configuration]")
        print(f"  Case: {self.case}")
        print(f"  Layers: {self.n_layers}")
        for i in range(self.n_layers):
            alpha = self.per_layer_alpha[i]
            coef = self.per_layer_coefficient[i]
            is_multi = self.per_layer_is_multi_term[i]
            learn_a = self.per_layer_learn_alpha[i]
            learn_c = self.per_layer_learn_coefficient[i]

            if is_multi:
                print(
                    f"  Layer {i}: {len(alpha)}-term, α={alpha}, coef={coef}, "
                    f"learn_α={learn_a}, learn_coef={learn_c}"
                )
            else:
                print(f"  Layer {i}: single-term, α={alpha[0]}, learn_α={learn_a}")


def get_integrator(integrator_name: str) -> Callable:
    """
    Factory function to retrieve the integrator function based on name.

    Args:
        integrator_name: Name of the integrator ('odeint', 'fdeint', etc.).

    Returns:
        Callable: The integrator function.

    Raises:
        ValueError: If the integrator name is unknown.
    """
    if integrator_name == "odeint_adjoint":
        from torchdiffeq import odeint_adjoint as odeint

        return odeint
    elif integrator_name == "odeint":
        from torchdiffeq import odeint

        return odeint
    elif integrator_name == "fdeint_adjoint":
        from torchfde import fdeint_adjoint as fdeint

        return fdeint
    elif integrator_name == "fdeint":
        from torchfde import fdeint

        return fdeint
    elif integrator_name == "fdeint_mem" or integrator_name == "odeint_mem":
        return step_dynamics
    else:
        raise ValueError(f"Unknown integrator: {integrator_name}")


class BoundaryShapeRecorder(Interpreter):
    """
    FX Interpreter to record boundary output shapes from the ODE graph.

    The ODE graph returns a tuple: (dv1/dt, ..., dvN/dt, boundary_1, boundary_2, ...).
    This recorder extracts the shapes of the boundary outputs (indices >= neuron_count).
    """

    def __init__(self, gm: torch.fx.GraphModule, neuron_count: int):
        super().__init__(gm)
        self.neuron_count = neuron_count
        self.boundary_shapes = []

    def run(self, *args: Any) -> Any:
        result = super().run(*args)
        # Output: (dv1/dt, ..., dvN/dt, boundary_1, boundary_2, ...)
        if isinstance(result, tuple):
            for i in range(self.neuron_count, len(result)):
                self.boundary_shapes.append(tuple(result[i].shape))
        return result


class ShapeRecorder(Interpreter):
    """
    FX Interpreter to record output shapes of neuron modules.

    Iterates through the traced graph and captures the output shape of every
    instance of `BaseNeuron`.
    """

    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__(gm)
        self.shapes = []
        self.neuron_instances = []

    def run_node(self, node: torch.fx.Node) -> Any:
        result = super().run_node(node)
        # whenever we hit a neuron, record its output tensor shape
        if node.op == "call_module":
            submod = self.module.get_submodule(node.target)
            if isinstance(submod, BaseNeuron):
                # result is (dv_dt, spike_out)
                spike_out = result
                # record the shape of either (they match)
                self.shapes.append(tuple(spike_out.shape))
                self.neuron_instances.append(submod)
        return result


def requires_initialization(func: Callable) -> Callable:
    """
    Decorator to ensure the SNNWrapper is initialized before method execution.

    Checks for the `_is_initialized` flag. If False, raises a RuntimeError
    with instructions to call `_set_neuron_shapes`.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_is_initialized") or not self._is_initialized:
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"ERROR: {self.__class__.__name__} not initialized!\n"
                f"You must call '_set_neuron_shapes(input_shape)' before calling '{func.__name__}'.\n"
                f"Example:\n"
                f"  net._set_neuron_shapes(input_shape=(1, 2, 128, 128)), # (batch=1, channels=2, H=128, W=128)\n"
                f"{'='*60}"
            )
        return func(self, *args, **kwargs)

    return wrapper


class SNNWrapper(nn.Module):
    r"""
    SNN Wrapper with per-layer fractional order support.

    This class wraps a standard PyTorch SNN model, converting its forward pass
    into a numerical integration of Fractional Differential Equations (FDEs).
    It supports flexible configuration of fractional orders ($\alpha$) per layer.

    Supported Features:

    - Per-layer alpha (single-term or multi-term).
    - Per-layer learnable alpha and coefficients.
    - Multiple FDE solvers (Grunwald-Letnikov, L1, etc.).
    - Automatic shape inference via FX tracing.
    """

    def __init__(
        self,
        base: nn.Module,
        integrator: str = "odeint",
        interpolation_method: str = "linear",
        alpha: Union[float, List[float], List[List[float]]] = 0.5,
        multi_coefficient: Optional[Union[List[float], List[List[float]]]] = None,
        learn_alpha: Union[bool, List[bool]] = False,
        learn_coefficient: Union[bool, List[bool]] = False,
        alpha_mode: str = "auto",
    ):
        """
        Initialize SNNWrapper with per-layer alpha support.

        Args:
            base: Base neural network model (nn.Module) containing neuron layers.
            integrator: Integrator type (`'odeint'`, `'fdeint'`, etc.).
            interpolation_method: Input interpolation method.
            alpha: Fractional order(s). Can be:

                - `float`: same alpha for all layers (single-term).
                - `List[float]`: interpretation depends on alpha_mode.
                - `List[List[float]]`: per-layer multi-term (Case C).
            multi_coefficient: Coefficients for multi-term FDE. Can be:

                - `None`: auto-fill with ones.
                - `List[float]`: same for all multi-term layers (Case B).
                - `List[List[float]]`: per-layer (Case C).
            learn_alpha: Whether alpha is learnable. Can be:

                - `bool`: applies to all layers.
                - `List[bool]`: per-layer.
            learn_coefficient: Whether coefficients are learnable. Can be:

                - `bool`: applies to all layers.
                - `List[bool]`: per-layer.
            alpha_mode: How to interpret flat list alpha. Options:

                - `'auto'`: Try to detect based on length and multi_coefficient.
                - `'per_layer'`: Force Case A (each element is one layer's alpha).
                - `'multiterm'`: Force Case B (broadcast multi-term to all layers).
                Ignored if alpha contains nested lists (always Case C).
        """
        super().__init__()

        self.integrator_indicator = integrator
        self.interpolation_method = interpolation_method
        self.integrator = get_integrator(integrator)

        # Store alpha config (will be finalized in _set_neuron_shapes)
        self._alpha_spec = alpha
        self._multi_coefficient_spec = multi_coefficient
        self._learn_alpha_spec = learn_alpha
        self._learn_coefficient_spec = learn_coefficient
        self._alpha_mode_spec = alpha_mode

        # Build FX-based ODEFunc
        self.ode_func = ODEFuncFromFX(base, interpolation_method=interpolation_method)
        self.traced_backbone = self.ode_func.traced
        self.post_neuron_module = self.ode_func.get_post_neuron_module()
        self.neuron_instances = None
        # Initialize as None to track initialization status
        self.neuron_shapes = None

        self._is_initialized = False  # Add initialization flag
        # we must call _set_neuron_shapes before run
        self._alpha_config = None

        # Store direct references BEFORE compiling
        self._ode_gm = self.ode_func.ode_gm
        self._ode_func_uncompiled = self.ode_func  # Add this for set_inputs

    def train(self, mode: bool = True) -> "SNNWrapper":
        """Sets the module in training mode and propagates to submodules."""
        super().train(mode)
        self.traced_backbone.train(mode)
        self._ode_gm.train(mode)  # Use stored reference
        if hasattr(self, "post_neuron_module") and self.post_neuron_module is not None:
            self.post_neuron_module.train(mode)
        return self

    def eval(self) -> "SNNWrapper":
        """Sets the module in evaluation mode and propagates to submodules."""
        super().eval()
        self.traced_backbone.eval()
        self._ode_gm.eval()  # Use stored reference
        if hasattr(self, "post_neuron_module") and self.post_neuron_module is not None:
            self.post_neuron_module.eval()
        return self

    def _set_neuron_shapes(self, input_shape: Optional[Tuple[int, ...]] = None):
        """
        Pre-compute neuron output shapes and finalize alpha configuration.

        Must be called before `forward()` to:
        1. Determine the number of neuron layers via FX tracing.
        2. Determine boundary output shapes.
        3. Register per-layer alpha parameters based on the detected layer count.

        Args:
            input_shape: Tuple representing the input shape (Batch, Channels, Height, Width).

        Raises:
            NotImplementedError: If input_shape is not provided.
        """
        self._is_initialized = True

        if input_shape is None:
            raise NotImplementedError(
                "Neuron membrane shape must be obtained using an input shape."
            )

        device = next(self.parameters()).device
        dummy_input = torch.zeros(input_shape, device=device)

        training_mode = self.training
        self.eval()

        with torch.no_grad():
            # Step 1: Get neuron shapes from traced backbone
            recorder = ShapeRecorder(self.traced_backbone)
            recorder.run(dummy_input)
            self.neuron_shapes = recorder.shapes
            self.neuron_instances = recorder.neuron_instances
            n_layers = len(self.neuron_shapes)
            self.neuron_count = n_layers

            # Step 2: Get boundary shapes from ode_gm
            batch_size = input_shape[0]
            adjusted_shapes = [(batch_size, *shape[1:]) for shape in self.neuron_shapes]

            dummy_t = torch.tensor(0.0, device=device)
            dummy_v_mems = tuple(torch.zeros(s, device=device) for s in adjusted_shapes)
            dummy_x = dummy_input.unsqueeze(0)  # [1, batch, ...]
            dummy_x_time = torch.tensor([0.0], device=device)

            # Use _get_ode_func_module to access ode_gm
            ode_gm = self._ode_gm
            boundary_recorder = BoundaryShapeRecorder(ode_gm, n_layers)
            boundary_recorder.run(dummy_t, dummy_v_mems, dummy_x, dummy_x_time)
            self.boundary_shapes = boundary_recorder.boundary_shapes
            self.n_boundaries = len(self.boundary_shapes)

        # Now we know n_layers, finalize alpha configuration
        print(f"\nDetected {n_layers} neuron layers")

        self._alpha_config = PerLayerAlphaConfig(
            alpha=self._alpha_spec,
            n_layers=n_layers,
            multi_coefficient=self._multi_coefficient_spec,
            learn_alpha=self._learn_alpha_spec,
            learn_coefficient=self._learn_coefficient_spec,
            alpha_mode=self._alpha_mode_spec,
            device=next(self.parameters()).device,
            dtype=torch.float32,
        )

        # Register parameters
        self._alpha_config.register_parameters(self)
        self._alpha_config.print_config()
        self.train(training_mode)

        # Now compile
        self.ode_func = torch.compile(self._ode_func_uncompiled)
        print(self.neuron_instances)

    def get_per_layer_alpha(self) -> List[torch.Tensor]:
        """
        Get current per-layer alpha values as tensors.
        Always returns tensors to maintain gradient flow for learnable alphas.

        Returns:
            List of alpha tensors for each layer.
        """
        result = []
        for i in range(len(self.per_layer_alpha_params)):
            alpha_param = self.per_layer_alpha_params[i]
            # Always return tensor to maintain gradient flow
            # Don't call .item() as it detaches from computation graph!
            result.append(alpha_param)
        return result

    def get_per_layer_coefficient(self) -> List[torch.Tensor]:
        """
        Get current per-layer coefficient values as tensors.
        Always returns tensors to maintain gradient flow.

        Note: 
            For single-term layers (Case A), coefficients are technically unused by solvers but returned as `[1.0]` for interface consistency.

        Returns:
            List of coefficient tensors for each layer.
        """
        result = []
        for i in range(len(self.per_layer_coefficient_params)):
            # Always return the coefficient tensor
            # Don't return None as it breaks gradient flow
            result.append(self.per_layer_coefficient_params[i])
        return result

    def _reset_mem(self):
        """Reset internal memory states of all neuron instances."""
        for neuron in self.neuron_instances:
            neuron._reset()

    ## this api is not used in this version.

    # net._set_neuron_shapes(input_shape=(1, 2, 128, 128))
    # we must call _set_neuron_shapes before running the f-SNN.
    # input_shape=(batch=1, data_size)
    @requires_initialization
    def forward(
        self,
        x: torch.Tensor,
        x_time: torch.Tensor,
        output_time: Optional[torch.Tensor] = None,
        method: str = "euler",
        options: Dict[str, Any] = {"step_size": 0.1},
    ) -> torch.Tensor:
        """
        Perform the forward pass of the Fractional SNN.

        Args:
            x: Input tensor of shape [Time_Steps, Batch, ...].
            x_time: Time points corresponding to input steps, shape [Time_Steps,].
            output_time: Optional time points for output. If None, auto-generated.
            method: Integration method (`'euler'`, `'gl'`, `'trap'`, `'l1'`, etc.).
            options: Dictionary of solver options (e.g., `{'step_size': 0.1}`).

        Returns:
            Output tensor processed by the post-neuron module.
        """
        time_steps, batch_size = x.shape[:2]
        if len(x_time) != time_steps:
            x_time = x_time[0:time_steps]
        self._ode_func_uncompiled.set_inputs(x, x_time)

        # Initialize neuron membrane potentials
        adjusted_neuron_shapes = [
            (batch_size, *shape[1:]) for shape in self.neuron_shapes
        ]
        v_mems = [torch.zeros(s, device=x.device) for s in adjusted_neuron_shapes]
        # Initialize boundary outputs with correct shapes
        boundary_inits = [
            torch.zeros((batch_size, *shape[1:]), device=x.device)
            for shape in self.boundary_shapes
        ]

        # Initial state:
        initial_state = (*v_mems, *boundary_inits)

        # Helper function to avoid code duplication
        def process_boundaries(v_mem_all_time_and_final_spike):
            if self.n_boundaries == 1:
                finalspike_out = torch.stack(v_mem_all_time_and_final_spike[-1], dim=0)
                return self.post_neuron_module(finalspike_out)
            else:
                boundary_outputs = tuple(
                    torch.stack(
                        v_mem_all_time_and_final_spike[self.neuron_count + i], dim=0
                    )
                    for i in range(self.n_boundaries)
                )
                return self.post_neuron_module(boundary_outputs)

        # 2) now run the real integration
        if True:  # output_time is None:
            # create output_time to add one more element to x_time
            if len(x_time) > 1:
                dt = x_time[1] - x_time[0]
            else:
                dt = options.get("step_size", 1.0)

            # Create the next time point on the same device
            next_t = (x_time[-1] + dt).unsqueeze(0)

            # Concatenate to form the full output time vector
            output_time = torch.cat((x_time, next_t), dim=0)

        # Get current per-layer alpha and coefficient
        per_layer_alpha = self.get_per_layer_alpha()
        per_layer_coefficient = self.get_per_layer_coefficient()

        # check if it is fdeint or odeint
        if self.integrator_indicator == "odeint" and method == "euler":
            v_mem_all_time_and_final_spike = euler_integrate_tuple_compiled(
                self.ode_func, initial_state, output_time, self.neuron_count
            )
            return process_boundaries(v_mem_all_time_and_final_spike)

        elif self.integrator_indicator == "fdeint":
            memory = None if options.get("memory", -1) == -1 else options["memory"]

            # Determine solver based on alpha case:
            # Case A: per-layer single-term → use requested solver
            # Case B/C: multi-term involved → always use multiterm solver
            use_multiterm = self._alpha_case in ("B", "C") or method == "glmulti"

            if use_multiterm:
                # Use multiterm solver for Case B, C, or explicit request
                if self._alpha_case in ("B", "C") and method not in ("gl", "glmulti"):
                    import warnings

                    warnings.warn(
                        f"Alpha configuration is Case {self._alpha_case} (multi-term). "
                        f"Method '{method}' not supported for multi-term, "
                        f"using GrunwaldLetnikovMultitermSNN instead.",
                        UserWarning,
                    )

                v_mem_all_time_and_final_spike = glmethod_multiterm_integrate_tuple(
                    self.ode_func,
                    initial_state,
                    per_layer_alpha,
                    output_time,
                    memory=memory,
                    per_layer_coefficient=per_layer_coefficient,
                )
            else:
                # Case A (per-layer single-term): use requested solver
                integrate_method = SOLVERS[method]
                v_mem_all_time_and_final_spike = integrate_method(
                    self.ode_func,
                    initial_state,
                    per_layer_alpha,
                    output_time,
                    memory=memory,
                    per_layer_coefficient=per_layer_coefficient,
                )

            return process_boundaries(v_mem_all_time_and_final_spike)

        # ----------------------- please ignore the following temporarily-----------------------

        if self.integrator_indicator == "odeint_mem":
            v_mem_all_time_and_final_spike = step_dynamics(
                self.ode_func, initial_state, output_time
            )
            finalspike_out = torch.stack(v_mem_all_time_and_final_spike, dim=0)
            final_output = self.post_neuron_module(finalspike_out)
            return final_output

        elif (
            self.integrator_indicator == "odeint_adjoint"
            or self.integrator_indicator == "odeint"
        ):
            v_mem_all_time_and_cumulated_spike = self.integrator(
                self.ode_func,
                initial_state,
                output_time,
                method=method,
                options=options,
            )
            finalspike_out_sum = v_mem_all_time_and_cumulated_spike[-1][-1:, ...]

            final_output = self.post_neuron_module(finalspike_out_sum)
            return final_output

        elif self.integrator_indicator == "fdeint_mem":

            v_mem_all_time_and_final_spike = step_dynamics(
                self.ode_func, initial_state, output_time
            )
            # finalspike_out = v_mem_all_time_and_final_spike
            finalspike_out = torch.stack(v_mem_all_time_and_final_spike, dim=0)
            final_output = self.post_neuron_module(finalspike_out)
            return final_output

        elif self.integrator_indicator == "fdeint_adjoint":
            memory = None if options.get("memory", -1) == -1 else options["memory"]

            v_mem_all_time_and_cumulated_spike = fdeint_adjoint(
                self.ode_func,
                initial_state,
                per_layer_alpha[0],
                output_time,
                method=method,
                memory=memory,
            )
            # print('v_mem_all_time_and_cumulated_spike.shape: ', v_mem_all_time_and_cumulated_spike[-1].shape)
            finalspike_out_sum = v_mem_all_time_and_cumulated_spike[-1].unsqueeze(0)
            final_output = self.post_neuron_module(finalspike_out_sum)
            return final_output

        elif (
            False
        ):  # self.integrator_indicator == "fdeint_adjoint" or self.integrator_indicator == "fdeint":
            # print(f"output_time: {output_time}")# print('using', self.integrator_indicator, 'for integration')
            # raise NotImplementedError("please only use odeint+euler or fdeint+gl")
            T = output_time[-1]
            step_size = output_time[-1] - output_time[-2]
            v_mem_all_time_and_cumulated_spike = self.integrator(
                self.ode_func,
                (*v_mems, spike_sum_init),
                torch.tensor(self.alpha),
                t=T,
                step_size=step_size,
                method=method,
                options=options,
            )
            # print(v_mem_all_time_and_cumulated_spike[-1].shape)
            finalspike_out_sum = v_mem_all_time_and_cumulated_spike[-1].unsqueeze(0)
            # v_scaled = final_v_mem - self.ode_func.finalneuron_threshold
            # finalspike_out = self.ode_func.finalneuron_surrogate_f(v_scaled, self.ode_func.finalneuron_surrogate_grad_scale)
            final_output = self.post_neuron_module(finalspike_out_sum)
            return final_output

    def print_alpha_info(self):
        """Print current alpha values and their learnable status."""
        if not self._is_initialized:
            print("Not initialized yet. Call _set_neuron_shapes first.")
            return

        print("\n[Current Alpha Values]")
        for i in range(len(self.per_layer_alpha_params)):
            alpha = self.per_layer_alpha_params[i]
            is_multi = self._per_layer_is_multi_term[i]
            is_learnable = self._alpha_is_param[i]

            if is_multi:
                coef = self.per_layer_coefficient_params[i]
                coef_learnable = self._coef_is_param[i]
                print(f"  Layer {i} (multi-term):")
                print(f"    Alpha: {alpha.data.tolist()} (learnable: {is_learnable})")
                print(
                    f"    Coefficient: {coef.data.tolist()} (learnable: {coef_learnable})"
                )
            else:
                print(
                    f"  Layer {i} (single-term): alpha = {alpha.item():.4f} (learnable: {is_learnable})"
                )
