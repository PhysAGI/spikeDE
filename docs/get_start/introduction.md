# Introduction by Example

We shortly introduce the fundamental concepts of **spikeDE** through a simple example: training a SNN on the MNIST dataset using fractional-order dynamics. This tutorial assumes no prior knowledge of SNNs or differential equation solvers—everything you need will be explained along the way.

!!! tip "Recommend Reading"
    For an introduction to SNNs, we refer the interested reader to [Training Spiking Neural Networks Using Lessons From Deep Learning](https://ieeexplore.ieee.org/document/10242251).

---

## What is spikeDE?

**spikeDE** is a :simple-pytorch: PyTorch-based library designed to implement the **Fractional-Order Spiking Neural Network (*f*-SNN)** framework. Unlike traditional SNN libraries that rely on first-order Ordinary Differential Equations (ODEs) with Markovian properties—where the current state depends only on the immediate past—**spikeDE** governs neuron dynamics using **Fractional-Order Differential Equations (FDEs)**. This approach is grounded in the observation that biological neurons often exhibit non-Markovian behaviors, such as power-law relaxation and long-range temporal correlations, which cannot be captured by integer-order models.

Crucially, spikeDE serves as a **generalized framework** that strictly encompasses traditional integer-order SNNs. By setting the fractional order $\alpha = 1$, the library naturally recovers standard Leaky Integrate-and-Fire (LIF) and Integrate-and-Fire (IF) models, making it a superset of existing approaches rather than an alternative solver. When $0 < \alpha < 1$, the Caputo fractional derivative introduces a **power-law memory kernel**, allowing the membrane potential to depend on its entire history. This capability enables the modeling of complex phenomena like persistent memory, fractal dendritic structures, and enhanced robustness to input perturbations, offering a more biologically plausible and mathematically rich foundation for spiking networks.

At its core, spikeDE provides:

- **Fractional Neuron Models**: Implementations of *f*-LIF and *f*-IF neurons that naturally encode long-term dependencies via fractional calculus.
- **Generalized Wrapper (`SNNWrapper`)**: A flexible interface that converts any standard PyTorch network into an *f*-SNN, supporting both **single-term** and **multi-term** fractional dynamics.
- **Advanced Numerical Solvers**: Efficient discretization methods (e.g., fractional Adams–Bashforth–Moulton, Grünwald–Letnikov) tailored for non-local fractional operators.
- **Trainable Fractional Orders**: Options to learn the fractional order $\alpha$ and memory coefficients end-to-end, allowing the network to adapt its temporal memory span automatically.

This allows researchers to move beyond simple recurrence and explore how **non-Markovian dynamics**, **history-dependent evolution**, and **fractional temporal scaling** enhance learning in spiking networks across vision, graph, and sequence tasks.

---

## Step-by-Step Walkthrough: MNIST Classification with spikeDE

Below, we walk through the key components of the provided example script. You can run this code after installing spikeDE and :simple-pytorch: PyTorch.

!!! Note
    The full script is designed as a standalone example. Only classes/functions imported from `spikeDE` (e.g., `SNN`, `SNNWrapper`, `LIFNeuron`) are part of the package. Everything else—data loading, model definitions like `CNNExample`, utility functions like `spike_converter`—are **user-defined helpers** written specifically for this demo.

### Importing Required Modules

```python title="Importing requried modules",linenums="1",hl_lines="10 11"
# Core pytorch components
import torch
import torch.nn as nn

# Dataset loading components
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Core spikeDE components
from spikeDE import SNN, SNNWrapper
from spikeDE import LIFNeuron, IFNeuron
```

Here, only the last two lines involve **spikeDE**. The rest are standard :simple-pytorch: PyTorch utilities for data handling and training loops.

---

### Defining Your Base Network

Before wrapping a network with spikeDE, you define a **standard** :simple-pytorch: **PyTorch model** using regular layers—but insert **spiking neurons** at activation points.

```python title="CNN based network",linenums="1",hl_lines="5 9 14 16"
class CNNExample(nn.Module):
    def __init__(self, tau, threshold, surrogate_grad_scale):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.lif1 = LIFNeuron(tau, threshold, surrogate_grad_scale)  # ← Spiking neuron!
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.lif3 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.fc2 = nn.Linear(128, 10)
        self.lif4 = LIFNeuron(tau, threshold, surrogate_grad_scale)

    def forward(self, x):
        out = self.lif1(self.conv1(x))
        out = self.pool1(out)
        out = self.lif2(self.conv2(out))
        out = self.pool2(out)
        out = self.lif3(self.fc1(self.flatten(out)))
        out = self.lif4(self.fc2(out))
        return out
```

!!! tip "Key Insight"
    - This looks like a normal CNN—but instead of ReLU, we use `LIFNeuron`.
    - Each `LIFNeuron` maintains internal membrane potential and emits spikes based on dynamics defined by **tau (time constant)**, **threshold**, and a **surrogate gradient** for backpropagation.  
    - The actual spiking behavior is **not computed here directly**—it’s handled later by `SNNWrapper` during time integration.

??? quote "MLP Based Network"
    You could similarly define an MLP:
    ```python title="MLP based network",linenums="1",hl_lines="6 8"
    class MLPExample(nn.Module):
        def __init__(self, tau, threshold, surrogate_grad_scale):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(28*28, 2560, bias=False)
            self.lif1 = LIFNeuron(tau, threshold, surrogate_grad_scale)
            self.fc2 = nn.Linear(2560, 10, bias=False)
            self.lif2 = LIFNeuron(tau, threshold, surrogate_grad_scale)

        def forward(self, x):
            x = self.flatten(x)
            x = self.lif1(self.fc1(x))
            x = self.lif2(self.fc2(x))
            return x
    ```

---

### Converting Static Inputs to Spike Trains

SNNs process **temporal spike sequences**, not static images. So we must convert each MNIST image into a series of spikes over time.

```python title="Spike converting",linenums="1",hl_lines="6 10"
def spike_converter(x, time_steps=100, flatten=False):
    batch_size = x.size(0)
    if flatten:
        x = x.view(batch_size, -1)
        p = x.unsqueeze(1).repeat(1, time_steps, 1)
        spikes = torch.bernoulli(p)
        return spikes.permute(1, 0, 2)  # [T, B, N]
    else:
        p = x.unsqueeze(1).repeat(1, time_steps, 1, 1, 1)
        spikes = torch.bernoulli(p)
        return spikes.permute(1, 0, 2, 3, 4)  # [T, B, C, H, W]
```

In the training loop, inputs are scaled (`data = 10 * data`) to increase spike rates—this is a common heuristic.

---

### Wrapping Your Model with `SNNWrapper`

This is where the **fractional framework** is applied. The `SNNWrapper` transforms your static network into a dynamical system driven by FDEs.

```python title="Wrapping the base model",linenums="1",hl_lines="14 15"
# Initialize the base CNN
base_network = CNNExample(tau=2.0, threshold=1.0, surrogate_grad_scale=0.3)

# Wrap with fractional dynamics
snn_model = SNNWrapper(
    base_network,
    integrator="fdeint",       # Use fractional solver
    alpha=0.8,                 # Fractional order (0 < alpha <= 1)
    multi_coefficient=None,    # None for single-term FDE
    learn_alpha=True,          # Optionally learn the fractional order
    learn_coefficient=False
)

# Initialize internal buffers based on input shape (C, H, W)
snn_model._set_neuron_shapes(input_shape=(1, 28, 28))
```

!!! note "Key Parameters"
    - `integrator`: Chooses the solver type:
        - `'odeint'` / `'odeint_adjoint'` for classical ODEs (integer-order);
        - `'fdeint'` / `'fdeint_adjoint'` for FDEs.
    - `alpha`: The fractional order (e.g., `0.5` for single alpha, `[0.3, 0.4, 0.5]` for multi-alpha).
    - `multi_coefficient`: Weights for each term (required if `alpha` has multiple values).
    - `learn_coefficient`: If `True`, coefficient(s) become **trainable parameter(s)**.
    - `learn_alpha`: If `True`, $\alpha(s)$ become **trainable parameter(s)**.

---

### Training Loop: Time Integration Over Spikes

During training, static inputs are first encoded into temporal spike trains with shape `[T, B, ...]`. These sequences are then passed to the model alongside a **time grid** that defines the evolution interval for the fractional solver:

```python title="Training loop",linenums="1"
# Define time grid: from 0 to T_end with (T+1) points
data_time = torch.linspace(0, 0.01 * 100, 100 + 1, device=device).float()

# Forward pass through the fractional dynamics solver
output = model(
    data,
    data_time,
    method="gl",
    options={"step_size": 1.0, "memory": -1},
)

# Aggregate temporal outputs (e.g., average pooling) for final classification
output = output.mean(0)
```

!!! note "Key arguments explained"
    - **`data_time`**: Specifies the discrete time points $t_0, t_1, \dots, t_T$ over which the differential equation is solved.
    - **`method`**: Selects the numerical integration scheme (e.g., `'gl'` for the Grünwald–Letnikov formula, suitable for capturing long-range memory).
    - **`options`**: Configures solver-specific parameters. For instance, `memory=-1` instructs the solver to utilize the **full history** of the state, which is essential for accurate fractional-order simulation.

The model returns a sequence of outputs corresponding to each time step. To obtain a single prediction for classification, we typically aggregate these temporal responses (e.g., via averaging or summing).

??? quote "Full Training Pipeline"
    The following block combines data loading, model instantiation, and the training loop. It demonstrates how to pass the time grid to the solver and handle the temporal outputs of the *f*-SNN.
    ```python title="Standalone Code",linenums="1"
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from spikeDE import SNNWrapper, LIFNeuron

    # --- 1. Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TIME_STEPS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    FRACTIONAL_ORDER = 0.8  # Alpha < 1 enables long-term memory
    EPOCHS = 5

    # --- 2. Data Loading ---
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            lambda x: x.clamp(0, 1),  # Ensure values are in [0, 1]
        ]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # --- 3. Model Definition ---
    class CNNExample(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 32, 3, padding=1)
            self.lif1 = LIFNeuron(tau=2.0, threshold=1.0, surrogate_grad_scale=0.3)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(32 * 14 * 14, 10)
            self.lif2 = LIFNeuron(tau=2.0, threshold=1.0, surrogate_grad_scale=0.3)

        def forward(self, x):
            x = self.lif1(self.conv(x))
            x = self.pool(x)
            x = x.flatten(1)
            x = self.lif2(self.fc(x))
            return x


    base_net = CNNExample().to(DEVICE)
    model = SNNWrapper(
        base_net, integrator="fdeint", alpha=FRACTIONAL_ORDER, learn_alpha=False
    ).to(DEVICE)

    # Initialize shapes (C, H, W)
    model._set_neuron_shapes(input_shape=(1, 28, 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # --- 4. Helper: Spike Encoding ---
    def spike_converter(x, time_steps=100, flatten=False):
        batch_size = x.size(0)
        if flatten:
            x = x.view(batch_size, -1)
            p = x.unsqueeze(1).repeat(1, time_steps, 1)
            spikes = torch.bernoulli(p)
            return spikes.permute(1, 0, 2)  # [T, B, N]
        else:
            p = x.unsqueeze(1).repeat(1, time_steps, 1, 1, 1)
            spikes = torch.bernoulli(p)
            return spikes.permute(1, 0, 2, 3, 4)  # [T, B, C, H, W]


    # --- 5. Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Convert static images to spike trains [T, B, ...]
            spike_input = spike_converter(data, TIME_STEPS).to(DEVICE)
            spike_input = spike_input * 10

            # Define time grid for the solver
            # Time goes from 0 to T_end. Step size depends on your physical time scaling.
            data_time = torch.linspace(0, 1.0, TIME_STEPS + 1).to(DEVICE)

            optimizer.zero_grad()

            # Forward pass through fractional solver
            # Output shape: [T, B, Classes]
            output_seq = model(
                spike_input, data_time, method="gl", options={"step_size": 1.0}
            )

            # Decision strategy: Sum or Average spikes over time
            output = output_seq.mean(dim=0)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
        torch.cuda.empty_cache()

    # --- 6. Evaluation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            spike_input = spike_converter(data, TIME_STEPS)
            data_time = torch.linspace(0, 1.0, TIME_STEPS + 1).to(DEVICE)

            output_seq = model(
                spike_input, data_time, method="gl", options={"step_size": 1.0}
            )
            output = output_seq.mean(dim=0)

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        torch.cuda.empty_cache()

    print(f"Test Accuracy: {100. * correct / total:.2f}%")

    ```

---

## Next Steps

- **Different Neuron Types**: Try different neuron types (`IFNeuron`).
- **Experiment with $\alpha$**: Try setting `alpha=1.0` to compare against standard LIF, or `alpha=0.6` for stronger memory effects.
- **Learnable Orders**: Enable `learn_alpha=True` to let the network discover the optimal memory depth per layer.
- **Multi-term Dynamics**: Explore `multi_coefficient` to simulate complex biological relaxation processes.
- **Visualization**: Plot the membrane potential over time to observe the power-law decay characteristic of fractional systems.

spikeDE opens the door to **physics-informed spiking networks**—where neural dynamics obey principled mathematical laws beyond simple recurrence. Happy spiking!