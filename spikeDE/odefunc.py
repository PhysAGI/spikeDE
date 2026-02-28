import numpy as np
import torch
from torch import fx, nn
from .neuron import BaseNeuron
from .layer import VotingLayer, ClassificationHead
import operator
from typing import Dict, Tuple, Any, Union
from torch.fx import Tracer

"""
Torch-based ODEFunc with Interpolation Methods

This module provides an implementation of ODEFunc for neural ODE systems with
various interpolation methods implemented purely in PyTorch. The interpolation
is applied to input data at arbitrary time points during integration.

Supported interpolation methods:

- 'linear': Linear interpolation between adjacent points
- 'nearest': Nearest neighbor interpolation (no actual interpolation)
- 'cubic': Cubic spline interpolation using Catmull-Rom formula
- 'akima': Akima interpolation that reduces oscillations in cubic splines

All methods are implemented using PyTorch operations to maintain GPU acceleration
without requiring data transfers between devices.
"""


################# the following is for interpolate################
## See examples/function_test/check_interpolate.py, check_interpolate.py
# and spikeDE/odefunc_fx.md for more understanding.


def interpolate(
    x: torch.Tensor,
    x_time: torch.Tensor,
    t: Union[float, torch.Tensor],
    method: str = "linear",
) -> torch.Tensor:
    """
    Interpolate batched input data at arbitrary query time(s) `t`.

    This function reconstructs continuous-time input signals from discrete samples
    to support numerical ODE solvers that evaluate the vector field at non-integer time steps.

    Args:
        x (torch.Tensor): Input tensor of shape `[T, B, ...]`, where `T` is the number
            of time points and `B` is the batch size.
        x_time (torch.Tensor): Time points tensor. Can be:
            - `[T]`: Shared timestamps for all batches.
            - `[B, T]` or `[T, B]`: Batch-specific timestamps.
        t (Union[float, torch.Tensor]): Query time(s). Can be:
            - `float` or `scalar tensor`: Same time for all batches.
            - `[B]` tensor: Different time per batch.
        method (str): Interpolation algorithm. Options:
            - `'linear'`: Linear interpolation (default).
            - `'nearest'`: Nearest neighbor.
            - `'cubic'`: Catmull-Rom cubic spline (requires T >= 4).
            - `'akima'`: Akima spline, robust against oscillations (requires T >= 5).

    Returns:
        torch.Tensor: Interpolated tensor of shape `[B, ...]`.

    Raises:
        ValueError: If an unsupported interpolation method is specified.
        AssertionError: If input shapes are inconsistent.

    Note:
        Values outside the time range `[min(x_time), max(x_time)]` are clamped
        to the boundary values.
    """
    x, x_time, t, B, T = _prepare_inputs(x, x_time, t)

    t_min, t_max = x_time[:, 0], x_time[:, -1]
    at_min, at_max = t <= t_min, t >= t_max
    in_range = ~at_min & ~at_max

    result = torch.zeros((B,) + x.shape[2:], dtype=x.dtype, device=x.device)

    if at_min.any():
        result[at_min] = x[0, at_min]
    if at_max.any():
        result[at_max] = x[-1, at_max]

    if in_range.any():
        methods = {
            "linear": linear_interpolate_batched,
            "nearest": nearest_interpolate_batched,
            "cubic": cubic_interpolate_batched,
            "akima": akima_interpolate_batched,
        }
        if method not in methods:
            raise ValueError(
                f"Unknown method: {method}. Choose from {list(methods.keys())}"
            )
        result[in_range] = methods[method](
            x[:, in_range], x_time[in_range], t[in_range]
        )

    return result


def _prepare_inputs(
    x: torch.Tensor, x_time: torch.Tensor, t: Union[float, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    Prepare and validate inputs, normalizing shapes for batched processing.

    Args:
        x (torch.Tensor): Input tensor of shape `[T, B, ...]`.
        x_time (torch.Tensor): Timestamps, either `[T]` (shared) or `[B, T]`/`[T, B]` (batched).
        t (Union[float, torch.Tensor]): Query time, scalar or `[B]`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
            A tuple containing:
            - `x`: Unchanged input `[T, B, ...]`.
            - `x_time`: Normalized to `[B, T]`.
            - `t`: Normalized to `[B]`.
            - `B`: Batch size.
            - `T`: Number of time points.

    Raises:
        ValueError: If `x_time` or `t` shapes are incompatible with `x`.
        AssertionError: If dimensions do not match expected constraints.
    """
    T, B = x.shape[:2]

    # Handle x_time shape
    if x_time.dim() == 1:
        assert x_time.shape[0] == T, f"x_time length {x_time.shape[0]} != T={T}"
        x_time_batched = x_time.unsqueeze(0).expand(B, -1)  # [T] -> [B, T]
    elif x_time.shape == (B, T):
        x_time_batched = x_time
    elif x_time.shape == (T, B):
        x_time_batched = x_time.T  # [T, B] -> [B, T]
    else:
        raise ValueError(
            f"x_time shape {x_time.shape} incompatible with x shape {x.shape}"
        )

    # Handle t shape
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=x_time.dtype, device=x_time.device)
    else:
        t = t.to(dtype=x_time.dtype, device=x_time.device)

    if t.dim() == 0:
        t_batched = t.expand(B)  # scalar -> [B]
    elif t.dim() == 1:
        assert t.shape[0] == B, f"t length {t.shape[0]} != B={B}"
        t_batched = t
    else:
        raise ValueError(
            f"t must be scalar or 1D tensor of length B={B}, got shape {t.shape}"
        )

    return x, x_time_batched, t_batched, B, T


def _expand_to(scalar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Expand a 1D tensor `[B]` to match the dimensions of a target tensor `[B, ...]` for broadcasting.

    Args:
        scalar (torch.Tensor): 1D tensor of shape `[B]`.
        target (torch.Tensor): Target tensor of shape `[B, D1, D2, ...]`.

    Returns:
        torch.Tensor: Reshaped tensor of shape `[B, 1, 1, ...]` compatible with `target`.
    """
    return scalar.view(-1, *([1] * (target.dim() - 1)))


def linear_interpolate_batched(
    x: torch.Tensor, x_time: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Perform batched linear interpolation.

    Args:
        x (torch.Tensor): Input tensor `[T, B, ...]`.
        x_time (torch.Tensor): Time points `[B, T]`.
        t (torch.Tensor): Query times `[B]`.

    Returns:
        torch.Tensor: Interpolated values `[B, ...]`.
    """
    B, T = x_time.shape
    batch_idx = torch.arange(B, device=x.device)

    idx = torch.searchsorted(x_time, t.unsqueeze(1)).squeeze(1)
    idx = (idx - 1).clamp(0, T - 2)

    t0 = x_time[batch_idx, idx]
    t1 = x_time[batch_idx, idx + 1]
    x0 = x[idx, batch_idx]
    x1 = x[idx + 1, batch_idx]

    ratio = _expand_to((t - t0) / (t1 - t0 + 1e-10), x0)
    return x0 + ratio * (x1 - x0)


def nearest_interpolate_batched(
    x: torch.Tensor, x_time: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Perform batched nearest neighbor interpolation.

    Args:
        x (torch.Tensor): Input tensor `[T, B, ...]`.
        x_time (torch.Tensor): Time points `[B, T]`.
        t (torch.Tensor): Query times `[B]`.

    Returns:
        torch.Tensor: Values at nearest time points `[B, ...]`.
    """
    B = x_time.shape[0]
    batch_idx = torch.arange(B, device=x.device)

    idx = torch.abs(x_time - t.unsqueeze(1)).argmin(dim=1)
    return x[idx, batch_idx]


def cubic_interpolate_batched(
    x: torch.Tensor, x_time: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Perform batched cubic (Catmull-Rom) interpolation.

    Requires at least 4 time points (T >= 4).

    Args:
        x (torch.Tensor): Input tensor `[T, B, ...]`.
        x_time (torch.Tensor): Time points `[B, T]`.
        t (torch.Tensor): Query times `[B]`.

    Returns:
        torch.Tensor: Interpolated values `[B, ...]`.
    """
    B, T = x_time.shape
    batch_idx = torch.arange(B, device=x.device)

    idx = torch.searchsorted(x_time, t.unsqueeze(1)).squeeze(1)
    idx = (idx - 1).clamp(0, T - 2)

    i1 = idx
    i0 = (i1 - 1).clamp(0, T - 1)
    i2 = (i1 + 1).clamp(0, T - 1)
    i3 = (i1 + 2).clamp(0, T - 1)

    p0, p1, p2, p3 = (
        x[i0, batch_idx],
        x[i1, batch_idx],
        x[i2, batch_idx],
        x[i3, batch_idx],
    )
    t1, t2 = x_time[batch_idx, i1], x_time[batch_idx, i2]

    dt = torch.where(t2 == t1, torch.ones_like(t2), t2 - t1)
    u = _expand_to(((t - t1) / dt).clamp(0, 1), p0)

    u2, u3 = u * u, u * u * u
    return (
        (-0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3) * u3
        + (p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3) * u2
        + (-0.5 * p0 + 0.5 * p2) * u
        + p1
    )


def akima_interpolate_batched(
    x: torch.Tensor, x_time: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Perform batched Akima interpolation.

    Akima interpolation uses local slopes to reduce oscillations common in cubic splines.
    Requires at least 5 time points (T >= 5).

    Args:
        x (torch.Tensor): Input tensor `[T, B, ...]`.
        x_time (torch.Tensor): Time points `[B, T]`.
        t (torch.Tensor): Query times `[B]`.

    Returns:
        torch.Tensor: Interpolated values `[B, ...]`.
    """
    B, T = x_time.shape
    batch_idx = torch.arange(B, device=x.device)

    # FIX: Corrected clamp range to allow interpolation in boundary intervals
    idx = torch.searchsorted(x_time, t.unsqueeze(1)).squeeze(1)
    idx = (idx - 1).clamp(0, T - 2)

    # Clamping indices handles the "ghost points" by duplicating boundary values
    i0 = (idx - 2).clamp(0, T - 1)
    i1 = (idx - 1).clamp(0, T - 1)
    i2 = idx
    i3 = (idx + 1).clamp(0, T - 1)
    i4 = (idx + 2).clamp(0, T - 1)

    x0, x1, x2, x3, x4 = (
        x[i0, batch_idx],
        x[i1, batch_idx],
        x[i2, batch_idx],
        x[i3, batch_idx],
        x[i4, batch_idx],
    )
    t0, t1, t2, t3, t4 = (
        x_time[batch_idx, i0],
        x_time[batch_idx, i1],
        x_time[batch_idx, i2],
        x_time[batch_idx, i3],
        x_time[batch_idx, i4],
    )

    def safe_slope(xa, xb, ta, tb):
        dt = torch.where(tb == ta, torch.full_like(tb, 1e-10), tb - ta)
        return (xb - xa) / _expand_to(dt, xa)

    m0, m1, m2, m3 = (
        safe_slope(x0, x1, t0, t1),
        safe_slope(x1, x2, t1, t2),
        safe_slope(x2, x3, t2, t3),
        safe_slope(x3, x4, t3, t4),
    )

    dm0, dm1, dm2 = torch.abs(m1 - m0), torch.abs(m2 - m1), torch.abs(m3 - m2)
    eps = 1e-10

    denom_left = dm1 + dm0 + eps
    s1 = torch.where(denom_left > eps, (dm1 * m0 + dm0 * m1) / denom_left, m1)

    denom_right = dm1 + dm2 + eps
    s2 = torch.where(denom_right > eps, (dm1 * m2 + dm2 * m1) / denom_right, m2)

    h = torch.where(t3 == t2, torch.ones_like(t3), t3 - t2)
    # Ensure u is calculated relative to the correct interval start (t2)
    u = _expand_to(((t - t2) / h).clamp(0, 1), x2)
    h = _expand_to(h, x2)

    a = x2
    b = s1 * h
    c = 3 * (x3 - x2) - h * (2 * s1 + s2)
    d = 2 * (x2 - x3) + h * (s1 + s2)

    u2, u3 = u * u, u * u * u
    return a + b * u + c * u2 + d * u3


########################################################################################
"""
ODEFuncFromFX from odefunc.py is a sophisticated module that transforms a SNN into 
an ODE-compatible form using PyTorch FX graph tracing. This enables the use of numerical
ODE/FDE solvers to integrate the membrane potential dynamics continuously over time, rather than 
using discrete time-stepping. See odefunc_fx.md for more details.
"""


class SNNLeafTracer(Tracer):
    """
    Custom FX Tracer that treats specific modules as leaf nodes.

    This tracer ensures that `BaseNeuron`, `VotingLayer`, and `ClassificationHead`
    modules are not decomposed during symbolic tracing, preserving their internal
    logic as single graph nodes.
    """

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        """
        Determine if a module should be treated as a leaf node.

        Args:
            m (nn.Module): The module instance being traced.
            module_qualified_name (str): The qualified name of the module.

        Returns:
            bool: True if the module is a leaf node (should not be traced internally).
        """
        if isinstance(m, BaseNeuron):
            return True
        if isinstance(m, VotingLayer):
            return True
        if isinstance(m, ClassificationHead):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def symbolic_trace_leaf_neurons(module: nn.Module) -> fx.GraphModule:
    """
    Symbolically trace a module using the custom `SNNLeafTracer`.

    Args:
        module (nn.Module): The PyTorch module to trace.

    Returns:
        fx.GraphModule: The traced graph module with leaf neurons preserved.
    """
    tracer = SNNLeafTracer()
    graph = tracer.trace(module)
    return fx.GraphModule(module, graph)


def remove_dead_code(m: nn.Module) -> nn.Module:
    """
    Remove dead code from a traced FX graph.

    Args:
        m (nn.Module): A traced GraphModule.

    Returns:
        nn.Module: A new GraphModule with unused nodes eliminated.
    """
    graph = fx.Tracer().trace(m)

    # 自动删除没有被使用的节点
    graph.eliminate_dead_code()

    return fx.GraphModule(m, graph)


class ODEFuncFromFX(nn.Module):
    r"""
    Wrapper that converts a Spiking Neural Network (SNN) into an ODE-compatible form.

    This class leverages PyTorch FX to symbolically trace the input `backbone` SNN and
    restructure its computation graph into two distinct parts:

    1. **ODE Graph (`ode_gm`)**: Contains all operations up to and including the last
       spiking neuron layer. It outputs:
       - The time derivatives of all neuronal membrane potentials ($dv/dt$).
       - Boundary values (spikes or intermediate tensors) required by downstream layers.

    2. **Post-Neuron Module (`post_neuron_module`)**: Contains all operations occurring
       after the last neuron (e.g., voting layers, classifiers). This module is decoupled
       from the ODE integration loop and applied only after solving the ODE system.

    Input signals are assumed to be sampled at discrete time points. To support continuous-time
    ODE solvers, inputs are interpolated on-the-fly using the specified `interpolation_method`.

    The resulting object can be passed directly to numerical ODE solvers (e.g., `torchdiffeq`)
    as the vector field function $f(t, v) = \text{d}v/\text{d}t$.

    Attributes:
        interpolation_method (str): Interpolation scheme for continuous input reconstruction.
        neuron_count (int): Number of `BaseNeuron` instances detected in the backbone.
        x (torch.Tensor): Cached input tensor (shape: `(T, ...)`).
        x_time (torch.Tensor): Time stamps corresponding to input samples (shape: `(T,)`).
        nfe (int): Number of function evaluations performed (useful for profiling solver cost).
        ode_gm (fx.GraphModule): The ODE-compatible computation graph.
        post_neuron_module (nn.Module): Module containing post-neuron operations.
        traced (fx.GraphModule): The original traced backbone for reference.
    """

    def __init__(
        self, backbone: nn.Module, interpolation_method: str = "linear"
    ) -> None:
        r"""
        Initializes the ODE-compatible wrapper from a spiking neural network.

        Args:
            backbone (nn.Module): The original SNN model containing `BaseNeuron` layers.
                Must be FX-traceable and contain at least one neuron layer.
            interpolation_method (str): Method used to interpolate discrete inputs to continuous time.
                Supported options:

                - `'linear'`: Linear interpolation between adjacent samples.
                - `'nearest'`: Hold last value (zero-order hold).
                - `'cubic'`: Catmull-Rom cubic spline interpolation.
                - `'akima'`: Akima spline interpolation (reduces overshoot).

        Raises:
            ValueError: If an unsupported node operation is encountered during graph rewriting.
        """
        super().__init__()
        self.interpolation_method = interpolation_method
        self.neuron_count = 0
        self.x = None
        self.x_time = None
        self.nfe = 0

        # Step 1: Symbolically trace
        traced: fx.GraphModule = symbolic_trace_leaf_neurons(backbone)
        modules = dict(traced.named_modules())

        # Step 2: Create New ODE-Compatible Graph
        new_graph = fx.Graph()
        node_map: Dict[fx.Node, fx.Node] = {}  # Old Node -> New Node

        # Create ODE specific inputs
        t_node = new_graph.placeholder("t")
        v_mems_node = new_graph.placeholder("v_mems")
        x_node = new_graph.placeholder("x")
        x_time_node = new_graph.placeholder("x_time")

        # Step 3: Input Interpolation
        # Robustly find the first placeholder for input mapping, ignoring name 'x'
        first_placeholder = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                first_placeholder = node
                break

        current_input = new_graph.call_function(
            interpolate,  # Ensure this function is in scope or imported
            args=(x_node, x_time_node, t_node, self.interpolation_method),
        )

        if first_placeholder:
            node_map[first_placeholder] = current_input

        # Initialize variables
        current_output = current_input
        dv_dt_list = []
        neuron_index = 0

        # Step 4: Find Last Neuron (Pre-scan)
        last_neuron_node = None
        neuron_nodes = []
        for node in traced.graph.nodes:
            if node.op == "call_module" and isinstance(
                modules[node.target], BaseNeuron
            ):
                last_neuron_node = node
                neuron_nodes.append(node)
                # (Optional) Save threshold/surrogate here if needed

        # Step 5: FIRST PASS - Build ODE Graph Content
        # We need to distinguish between nodes that MUST be in ODE (up to last neuron)
        # and nodes that MIGHT be in ODE (dependencies of post-neuron nodes)

        nodes_in_ode_graph = []  # To track topological order in new graph

        for node in traced.graph.nodes:
            # Skip the original placeholder as we handled it manually
            if node.op == "placeholder":
                continue

            if node.op == "call_module":
                submodule = modules[node.target]
                if isinstance(submodule, BaseNeuron):
                    # --- NEURON LOGIC ---
                    v_mem_current = new_graph.call_function(
                        operator.getitem, args=(v_mems_node, neuron_index)
                    )

                    # Assume BaseNeuron.forward returns (dv/dt, spike) in this context
                    mapped_input = self._map_arguments(node.args, node_map)
                    out_node = new_graph.call_module(
                        node.target,
                        args=(
                            v_mem_current,
                            (
                                mapped_input[0]
                                if isinstance(mapped_input, tuple)
                                else mapped_input
                            ),
                        ),
                    )

                    dv_dt = new_graph.call_function(
                        operator.getitem, args=(out_node, 0)
                    )
                    spike_output = new_graph.call_function(
                        operator.getitem, args=(out_node, 1)
                    )

                    dv_dt_list.append(dv_dt)

                    # For the graph flow, the neuron output is the spike
                    node_map[node] = spike_output
                    nodes_in_ode_graph.append(node)
                    neuron_index += 1
                    continue

            # --- STANDARD NODE LOGIC ---
            # Map arguments using the node_map
            mapped_args = self._map_arguments(node.args, node_map)
            mapped_kwargs = self._map_arguments(node.kwargs, node_map)

            # Recreate node in new graph
            if node.op == "call_module":
                new_node = new_graph.call_module(
                    node.target, args=mapped_args, kwargs=mapped_kwargs
                )
            elif node.op == "call_function":
                new_node = new_graph.call_function(
                    node.target, args=mapped_args, kwargs=mapped_kwargs
                )
            elif node.op == "call_method":
                new_node = new_graph.call_method(
                    node.target, args=mapped_args, kwargs=mapped_kwargs
                )
            elif node.op == "get_attr":
                new_node = new_graph.get_attr(node.target)
                node_map[node] = new_node
                # Don't add to nodes_in_ode_graph - it's not part of main data flow
                continue
            elif node.op == "output":
                continue
            else:
                # Should never happen, but just in case
                raise ValueError(f"Unexpected node op: {node.op}")

            node_map[node] = new_node
            nodes_in_ode_graph.append(node)

        # Step 6: INTELLIGENT CUT LOGIC
        # We need to determine which nodes act as the "Boundary" between ODE and Post-Process.
        # Ideally, everything after `last_neuron_node` should be post-process.
        # However, if a post-process node depends on a node `N` inside the ODE graph,
        # `N` must be an output of the ODE graph.

        post_neuron_nodes = []
        ode_graph_outputs = {}  # Map original_node -> new_graph_node to be outputted

        # Identify nodes strictly after the last neuron
        start_collecting = False
        for node in traced.graph.nodes:
            if start_collecting and node.op != "output":
                post_neuron_nodes.append(node)
            if node == last_neuron_node:
                start_collecting = True

        # Check dependencies of post-neuron nodes
        # If a post-node depends on a node NOT in `post_neuron_nodes`, that dependency is a boundary.
        boundary_nodes = set()

        # Check dependencies of post-neuron nodes
        post_neuron_set = set(post_neuron_nodes)
        for p_node in post_neuron_nodes:

            def register_boundary(arg):
                if isinstance(arg, fx.Node):
                    if arg not in post_neuron_set and arg.op != "placeholder":
                        boundary_nodes.add(arg)

            fx.map_arg(p_node.args, register_boundary)
            fx.map_arg(p_node.kwargs, register_boundary)

        # If no last neuron found, or no post nodes, boundary is just the last non-output node
        # ============================================================
        if post_neuron_nodes and not boundary_nodes:
            # Post-neuron nodes exist but don't depend on anything from ODE graph
            # This is unusual, but we should still return the last neuron's spike
            if last_neuron_node:
                boundary_nodes.add(last_neuron_node)

        if not post_neuron_nodes:
            # No post-neuron nodes at all - ODE graph IS the whole network
            # Return the last meaningful output
            if last_neuron_node:
                boundary_nodes.add(last_neuron_node)

        # Step 7: Finalize ODE Graph Output
        # The output is: (*dv_dt_list, boundary_val_1, boundary_val_2, ...)

        # Create order mapping from ORIGINAL traced graph (always complete and correct)
        node_order = {node: i for i, node in enumerate(traced.graph.nodes)}
        sorted_boundary_nodes = sorted(
            list(boundary_nodes), key=lambda n: node_order.get(n, float("inf"))
        )

        boundary_values = [node_map[n] for n in sorted_boundary_nodes]
        output_tuple = tuple(dv_dt_list) + tuple(boundary_values)
        new_graph.output(output_tuple)

        print("=" * 50)
        print("ODE Graph:")
        print(new_graph)

        self.ode_gm = fx.GraphModule(traced, new_graph)
        self.ode_gm.graph.eliminate_dead_code()  # This will clean up unused nodes in ODE graph!
        self.ode_gm.recompile()  # Add this line

        # Step 8: Build Post-Neuron Graph
        self.boundary_map = {}  # To remember which index corresponds to which node

        if post_neuron_nodes:
            post_graph = fx.Graph()
            post_node_map = {}

            # Create placeholders for boundary inputs
            # Input to post-net is the tuple of boundary values returned by ODE
            # But usually we wrap this. Let's assume input is unpacked or we use index.
            # Strategy: The post-module input will be the tuple of boundary values.

            if len(sorted_boundary_nodes) == 1:
                # SINGLE BOUNDARY: Use direct input (preserves original interface)
                post_input = post_graph.placeholder("x")
                post_node_map[sorted_boundary_nodes[0]] = post_input
            else:
                # MULTIPLE BOUNDARIES: Use tuple unpacking
                post_input_tuple = post_graph.placeholder("boundary_inputs")
                for idx, orig_node in enumerate(sorted_boundary_nodes):
                    val = post_graph.call_function(
                        operator.getitem, args=(post_input_tuple, idx)
                    )
                    post_node_map[orig_node] = val

            # Recreate post nodes
            current_post_out = None
            for node in post_neuron_nodes:
                mapped_args = self._map_arguments(node.args, post_node_map)
                mapped_kwargs = self._map_arguments(node.kwargs, post_node_map)

                if node.op == "call_module":
                    new_node = post_graph.call_module(
                        node.target, args=mapped_args, kwargs=mapped_kwargs
                    )
                elif node.op == "call_function":
                    new_node = post_graph.call_function(
                        node.target, args=mapped_args, kwargs=mapped_kwargs
                    )
                elif node.op == "call_method":
                    new_node = post_graph.call_method(
                        node.target, args=mapped_args, kwargs=mapped_kwargs
                    )
                elif node.op == "get_attr":
                    new_node = post_graph.get_attr(node.target)
                    post_node_map[node] = new_node
                    continue  # ← Don't update current_post_out
                else:
                    raise ValueError(
                        f"Unexpected node op in post_neuron_nodes: {node.op}"
                    )

                post_node_map[node] = new_node
                current_post_out = new_node

            post_graph.output(current_post_out)
            self.post_neuron_module = fx.GraphModule(traced, post_graph)
            print("-" * 50)
            print("Post-Neuron Graph:")
            print(post_graph)

        else:
            # Identity or specialized handler if purely ODE
            self.post_neuron_module = nn.Identity()
            print("-" * 50)
            print("No post-neuron operations found, set nn.Identity()")

        self.neuron_count = neuron_index
        self.traced = traced
        self.nfe = 0

    def _depends_on_nodes(self, node: fx.Node, target_nodes: set) -> bool:
        """
        Check if a node's arguments reference any nodes in the target set.

        Args:
            node (fx.Node): The node to check.
            target_nodes (set): Set of target nodes to check dependency against.

        Returns:
            bool: True if `node` depends on any node in `target_nodes`.
        """

        def check_args(args):
            if isinstance(args, fx.Node):
                return args in target_nodes
            elif isinstance(args, (tuple, list)):
                return any(check_args(arg) for arg in args)
            elif isinstance(args, dict):
                return any(check_args(v) for v in args.values())
            return False

        # Check both args and kwargs
        depends = check_args(node.args) or check_args(node.kwargs)
        return depends

    def _map_arguments(self, args: Any, node_map: Dict[fx.Node, fx.Node]) -> Any:
        """
        Recursively map node references in arguments to new graph nodes.

        Handles nested structures such as tuples within tuples, lists, and dictionaries.

        Args:
            args (Any): Original arguments (can be tuple, dict, list, or primitive).
            node_map (Dict[fx.Node, fx.Node]): Mapping table from old nodes to new nodes.

        Returns:
            Any: Mapped arguments with updated node references.
        """
        if isinstance(args, tuple):
            return tuple(self._map_arguments(a, node_map) for a in args)  # Recursive!
        elif isinstance(args, list):
            return [self._map_arguments(a, node_map) for a in args]
        elif isinstance(args, dict):
            return {k: self._map_arguments(v, node_map) for k, v in args.items()}
        elif isinstance(args, fx.Node):
            return node_map.get(args, args)
        else:
            return args

    def set_inputs(self, x: torch.Tensor, x_time: torch.Tensor) -> None:
        r"""
        Caches the input signal and its sampling timestamps for interpolation.

        These inputs are used during the ODE solve to reconstruct $x(t)$ at arbitrary times.

        Args:
            x (torch.Tensor): Input tensor of shape `(T, batch_size, ...)` where `T`
                is the number of time steps.
            x_time (torch.Tensor): Corresponding time stamps of shape `(T,)` or
                `(batch_size, T)`, typically monotonically increasing.
        """
        self.x = x
        self.x_time = x_time

    def forward(self, t: float, v_mems: Tuple) -> Tuple[torch.Tensor, ...]:
        r"""
        Computes the vector field $f(t, v) = \text{d}v/\text{d}t$ for ODE solvers.

        This method is called repeatedly by the ODE integrator. It evaluates the ODE graph
        at time `t` using the current membrane potentials `v_mems` and interpolated input.

        Args:
            t (float): Current time (scalar).
            v_mems (Tuple[torch.Tensor, ...]): Tuple of membrane potential tensors,
                one per neuron layer, each of shape `(batch_size, ...)`.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing:
                - `dv_dt_i`: Time derivative of membrane potential for the *i*-th neuron.
                - `boundary_val_j`: Intermediate values needed by the post-neuron module,
                  in topological order.

            Total length is `neuron_count + num_boundary_values`.
        """
        x = self.x
        x_time = self.x_time
        self.nfe += 1

        return self.ode_gm(t, v_mems, x, x_time)

    def get_post_neuron_module(self) -> nn.Module:
        r"""
        Returns the module that processes outputs after the last spiking neuron.

        This module should be applied to the boundary values returned by the ODE solver
        to produce the final network prediction.

        Returns:
            nn.Module: The post-neuron computation path.
        """
        return self.post_neuron_module

    def get_ode_module(self) -> fx.GraphModule:
        r"""
        Returns the FX graph module implementing the ODE vector field.

        This module encapsulates the entire ODE-compatible computation graph and can be
        inspected, saved, or modified independently.

        Returns:
            fx.GraphModule: The internal ODE evaluation graph.
        """
        return self.ode_gm
