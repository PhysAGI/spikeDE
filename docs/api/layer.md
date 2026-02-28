# spikeDE.layer

This module provides the essential building blocks for constructing the output stages of Spiking Neural Networks (SNNs) within the **spikeDE** framework. While the core neuron dynamics and ODE solvers handle continuous temporal evolution, this layer focuses on aggregating high-dimensional spiking activities into compact, task-ready representations.

## Key Features

- **Structured Voting:** Implements `VotingLayer` to perform group-wise averaging along the feature dimension, effectively reducing noise and compressing information based on predefined voting sizes.
- **Spatiotemporal Aggregation:** The `ClassificationHead` automatically handles multi-dimensional inputs, performing intelligent averaging across patch and time dimensions before applying the final linear projection.

---

::: spikeDE.layer
    options:
        members:
        - VotingLayer
        - ClassificationHead
        filters: public
        group_by_category: true
        show_submodules: false
