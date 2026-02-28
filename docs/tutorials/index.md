# Tutorials

Welcome to the **spikeDE** Tutorials section. These guides are designed to take you from understanding the core concepts of fractional spiking neurons to mastering advanced configurations for complex temporal dynamics.

Whether you are building your first Spiking Neural Network (SNN) or researching novel fractional-order architectures, these tutorials provide the theoretical background and practical code examples you need.

## Basics
Foundational concepts for building and training fractional SNNs.

<div class="grid cards" markdown>

-   :material-brain: **Neuron**

    ---

    Learn how spikeDE reimagines neurons as continuous dynamical systems. Discover how to upgrade standard Integrate-and-Fire models into Fractional-Order neurons with infinite memory.

    [:octicons-arrow-right-24: Read Tutorial](./basics/neuron.md)

-   :material-lightning-bolt: **Surrogate Gradient**

    ---

    Overcome the non-differentiable nature of spiking. Explore various surrogate functions (Sigmoid, Arctan, etc.) that enable end-to-end backpropagation in SNNs.

    [:octicons-arrow-right-24: Read Tutorial](./basics/surrogate.md)

-   :material-calculator: **Solver**

    ---

    Understand the numerical engines powering temporal memory. Compare methods like Grünwald-Letnikov, L1, and Product Trapezoidal for solving Fractional Differential Equations.

    [:octicons-arrow-right-24: Read Tutorial](./basics/solver.md)

</div>

## Intermediate
Advanced techniques for customizing network dynamics and architecture.

<div class="grid cards" markdown>

-   :material-function: **ODE Function**

    ---

    Dive into the graph transformation process. See how `ODEFuncFromFX` uses PyTorch FX to convert discrete networks into continuous vector fields compatible with ODE solvers.

    [:octicons-arrow-right-24: Read Tutorial](./intermediate/odefunc_fx.md)

-   :material-package: **SNN Wrapper**

    ---

    Master the central orchestrator. Learn to configure `SNNWrapper` for automatic architecture inference, input interpolation, and seamless switching between integer and fractional modes.

    [:octicons-arrow-right-24: Read Tutorial](./intermediate/snnWrapper.md)

-   :material-tune: **Per-Layer Alpha**

    ---

    Customize memory dynamics with surgical precision. Configure heterogeneous fractional orders ($\alpha$) per layer, enable multi-term derivatives, and make memory depth learnable.

    [:octicons-arrow-right-24: Read Tutorial](./intermediate/per_layer_alpha.md)

</div>

## Advanced
Real-world applications and complex task implementations directly adapted from our [published research](https://arxiv.org/abs/2507.16937).

<div class="grid cards" markdown>

-   :material-eye: **Neuromorphic Task**  
    
    ---

    Explore event-driven vision experiments. Learn how *f*-SNNs outperform traditional models on neuromorphic datasets like DVS128 Gesture and N-Caltech101 by capturing long-range temporal correlations.  

    [:octicons-arrow-right-24: Read Tutorial](./advanced/neuromorphic.md)

-   :material-graph: **Graph Learning Task**  
    
    ---

    Dive into graph-structured data processing. Discover how fractional-order dynamics enhance node classification accuracy and robustness on citation and co-purchase networks compared to integer-order baselines.  
    
    [:octicons-arrow-right-24: Read Tutorial](./advanced/graph.md)

</div>

## What's Next?

Now that you have explored the core components and advanced configurations of **spikeDE**, you are ready to build your own models.

*   :material-rocket-launch: **Ready to code?** Check out the [Introduction by Example](../get_start/introduction.md) in the **Getting Started** section for a complete end-to-end workflow.
*   :material-code-tags: **Need detailed specs?** Visit the [API Reference](../api/index.md) for comprehensive documentation on classes, methods, and parameters.

Happy spiking!