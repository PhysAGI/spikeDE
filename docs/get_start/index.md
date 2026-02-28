# Getting Started

**spikeDE** empowers you to build **Fractional-Order Spiking Neural Networks (*f*-SNNs)** using an API that aligns closely with :simple-pytorch: PyTorch. This design ensures a seamless transition for :simple-pytorch: PyTorch users, allowing you to leverage existing skills while exploring advanced fractional dynamics with minimal learning curve.

We support a diverse range of modern neural architectures, including **Multilayer Perceptrons (MLPs)**, **Convolutional Neural Networks (CNNs)**, **Residual Networks (ResNets)** and **Transformers**.

Whether you are researching neuromorphic vision, modeling complex time-series, or developing energy-efficient AI, **spikeDE** provides the robust tools needed to construct, train, and deploy high-performance spiking models.

In this section, you will find step-by-step guides ranging from installation to building and training your first functional network. Choose the path that best fits your current needs:

<div class="grid cards" markdown>

- :material-download: **Installation**

    ---
    Get up and running quickly. Install spikeDE via `pip` or from source, with detailed instructions for setting up all necessary dependencies.

    [:material-arrow-right: Go to Installation Guide](./installation.md)

- :material-lightbulb-on: **Introduction by Example**

    ---
    Dive straight into code. Follow a complete walkthrough to define a fractional spiking model, encode inputs, train on a dataset, and evaluate performance.

    [:material-arrow-right: Try Your First *f*-SNN](./introduction.md)

</div>

!!! tip "New to Spiking Neural Networks?" 
    If you are unfamiliar with SNN concepts, start with the **[Introduction by Example](./introduction.md)**. It assumes only basic familiarity with PyTorch and gently introduces core concepts such as spike encoding, fractional leaky integrate-and-fire (*f*-LIF) neurons, and surrogate gradients.

Happy spiking!