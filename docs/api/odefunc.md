# SpikeDE.odefunc

This module serves as the core engine for **Spiking Neural ODEs** within the SpikeDE framework. It provides the `ODEFuncFromFX` class, which automatically transforms standard Spiking Neural Networks (SNNs) into continuous-time neural ODE systems suitable for integration with adaptive solvers (e.g., `torchdiffeq`).

## Key Features

- **Automatic Graph Transformation**: Leverages PyTorch FX to symbolically trace SNN backbones, separating the continuous dynamics (membrane potential evolution) from discrete post-processing layers (e.g., voting or classification).
- **Continuous Input Reconstruction**: Supports high-precision reconstruction of discrete input spike trains at arbitrary time steps $t$ during integration.
- **Pure PyTorch Interpolation**: Implements four interpolation strategies entirely in PyTorch to ensure seamless GPU acceleration without CPU-GPU data transfer overhead:
    - `linear`: Standard linear interpolation.
    - `nearest`: Zero-order hold (nearest neighbor).
    - `cubic`: Catmull-Rom cubic splines for smooth trajectories.
    - `akima`: Akima splines for robustness against oscillations.
- **Solver Compatibility**: The generated vector field functions are fully compatible with standard ODE solvers, enabling efficient backpropagation through the integration process.

---

::: spikeDE.odefunc
    options:
        members:
        - ODEFuncFromFX
        - SNNLeafTracer
        - linear_interpolate_batched
        - nearest_interpolate_batched
        - cubic_interpolate_batched
        - akima_interpolate_batched
        - interpolate
        - remove_dead_code
        - symbolic_trace_leaf_neurons
        filters: public
        group_by_category: true
        show_submodules: false