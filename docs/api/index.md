# API References

Welcome to the **spikeDE** API Reference. This documentation provides a comprehensive guide to the internal modules that power our continuous-time Spiking Neural Network framework.

Designed with a **Continuous Dynamics First** philosophy, spikeDE bridges standard integer-order SNNs with advanced Fractional-Order Calculus, enabling infinite memory and complex temporal dependencies without altering your core model logic.

## Core Modules

<div class="grid cards" markdown>

-   :material-brain: **spikeDE.neuron**

    ---

    The foundation of continuous spiking dynamics. Defines stateless neuron modules that compute instantaneous derivatives independent of history.

    [:octicons-arrow-right-24: View Module](./neuron.md)

-   :material-function: **spikeDE.odefunc**

    ---

    The engine for Spiking Neural ODEs. Uses PyTorch FX to symbolically trace and transform discrete SNNs into continuous vector field functions.

    [:octicons-arrow-right-24: View Module](./odefunc.md)

-   :material-calculator: **spikeDE.solver**

    ---

    A differentiable numerical engine for Fractional Differential Equations (FDEs). Implements high-order discretization schemes like Grünwald-Letnikov.

    [:octicons-arrow-right-24: View Module](./solver.md)

-   :material-package: **spikeDE.snn**

    ---

    The high-level wrapper (`SNNWrapper`) that unifies the ecosystem. Converts standard PyTorch SNNs into fractional systems with flexible configuration.

    [:octicons-arrow-right-24: View Module](./snn.md)

-   :fontawesome-solid-road-spikes: **spikeDE.surrogate**

    ---

    Essential tools for training SNNs via backpropagation. Provides smooth approximations for the non-differentiable spiking operation.

    [:octicons-arrow-right-24: View Module](./surrogate.md)

- :material-layers: **spikeDE.layer**

    ---

    Specialized output modules for aggregating spatiotemporal spiking activities. Provides layers to prepare task-ready predictions.

    [:octicons-arrow-right-24: View Module](./layer.md)

</div>

---

## Design Philosophy

The **spikeDE** architecture is built on the separation of concerns:

1.  **Dynamics Definition**: Neurons define *what* changes ($dv/dt$).
2.  **State Evolution**: Solvers define *how* it changes over time (integration).
3.  **Graph Transformation**: FX traces bridge the gap between discrete PyTorch modules and continuous mathematical systems.

This modular design allows you to upgrade any standard SNN to a Fractional-Order SNN simply by wrapping it, unlocking powerful temporal modeling capabilities with minimal code changes.
