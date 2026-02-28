import torch
import torch.nn as nn
from spikeDE import SNNWrapper, LIFNeuron


# 1. Define your SNN (2 neuron layers)
class MySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.lif1 = LIFNeuron()  # Layer 0
        self.fc2 = nn.Linear(20, 5)
        self.lif2 = LIFNeuron()  # Layer 1

    def forward(self, x):
        x = self.lif1(self.fc1(x))
        x = self.lif2(self.fc2(x))
        return x


# 2. Wrap with SNNWrapper (per-layer α)
net = SNNWrapper(
    MySNN(),
    integrator='fdeint',
    alpha=[0.3, 0.7],  # Layer 0: α=0.3, Layer 1: α=0.7
    alpha_mode='per_layer',
    learn_alpha=True,
)

# 3. Initialize shapes (required!)
net._set_neuron_shapes(input_shape=(1, 10))

# 4. Create optimizer (alpha parameters are automatically included)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# 5. Training loop
for epoch in range(30):
    # Create dummy data: (time_steps, batch, features)
    x = torch.randn(8, 4, 10)
    x_time = torch.linspace(0, 1, 8)
    target = torch.randint(0, 5, (4,))

    # Forward pass
    output = net(x, x_time, method='gl')
    output = output.mean(0)
    loss = nn.CrossEntropyLoss()(output.squeeze(0), target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Monitor alpha values
    if epoch % 5 == 0:
        alphas = net.get_per_layer_alpha()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        print(f"  Alpha: {[a.tolist() for a in alphas]}")