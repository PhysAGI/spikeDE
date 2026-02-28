# SpikeDE.solver

This module delivers a comprehensive, differentiable numerical engine designed to simulate Spiking Neural Networks (SNNs) governed by Fractional Differential Equations (FDEs). Bridging the gap between fractional calculus and deep learning, this module supports both **Riemann-Liouville** and **Caputo** formulations through a diverse array of high-order discretization schemes, including **Grünwald-Letnikov (GL)**, **Product Trapezoidal**, **L1**, and **Adams-Bashforth** methods. It enables precise modeling of complex temporal dynamics while maintaining full compatibility with gradient-based optimization.

Whether used for forward inference via `snn_solve` or for training sophisticated fractional SNNs, the module provides a mathematically rigorous foundation for next-generation neural dynamics. Its architecture is built to handle advanced requirements such as per-layer fractional orders, multi-term distributed-order equations, and efficient memory management, ensuring scalability for long-sequence modeling.

## Key Features

- **Diverse Discretization Schemes**: Implements multiple high-precision numerical methods (Grünwald-Letnikov, Product Trapezoidal, L1, Adams-Bashforth) to solve FDEs under both Riemann-Liouville and Caputo definitions.
- **Advanced Fractional Configurations**: Natively supports **per-layer fractional orders**, allowing different layers to exhibit distinct memory properties, and handles **multi-term distributed-order equations** for complex dynamical systems.
- **Flexible Solver Interface**: Provides a unified API (`snn_solve`) alongside low-level integration primitives (`gl_integrate_tuple`, `l1_integrate_tuple`, etc.) for custom solver development and fine-grained control over state evolution.

---

::: spikeDE.solver
    options:
        filters: public
        members:
        - PerLayerAlphaInfo
        - SNNSolverConfig
        - SNNFractionalMethod
        - GrunwaldLetnikovSNN
        - ProductTrapezoidalSNN
        - L1MethodSNN
        - AdamsBashforthSNN
        - GrunwaldLetnikovMultitermSNN
        - FDEAdjointMethod
        - snn_solve
        - euler_integrate_tuple
        - gl_integrate_tuple
        - trap_integrate_tuple
        - l1_integrate_tuple
        - pred_integrate_tuple
        - glmethod_multiterm_integrate_tuple\
        - fdeint_adjoint
        - forward_euler_wo_history
        - backward_euler_wo_history
        - forward_euler_w_history
        - backward_euler_w_history
        - forward_gl
        - backward_gl
        - forward_trap
        - backward_trap
        - forward_l1
        - backward_l1
        - forward_pred
        - backward_pred
        - find_parameters
        - get_memory_bounds
        - step_dynamics
        group_by_category: true
        show_submodules: false