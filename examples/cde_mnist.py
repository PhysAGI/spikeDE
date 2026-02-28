import argparse
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikeDE import SNNWrapper
from spikeDE import LIFNeuron
import time
from tqdm import tqdm

# Use the updated LIFNeuron
LIFNeuron = LIFNeuron1111


def parse_args():
    parser = argparse.ArgumentParser(description='SNN Training for MNIST')

    # System parameters
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA device ID (default: 0)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for storing input data (default: ./data)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log training status every n batches (default: 100)')

    # Model parameters
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'],
                        help='Model architecture: mlp or cnn (default: cnn)')
    parser.add_argument('--neuron_type', type=str, default='LIF',
                        help='Neuron type for SNN (default: LIF)')
    parser.add_argument('--tau', type=float, default=2.0,
                        help='Membrane time constant for LIF neurons (default: 2.0)')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Firing threshold (default: 1.0)')
    parser.add_argument('--surrogate_grad_scale', type=float, default=5.0,
                        help='Scale parameter for surrogate gradient (default: 5.0)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--time_steps', type=int, default=4,
                        help='Number of time steps for SNN simulation (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')

    # Integrator parameters
    parser.add_argument('--integrator', type=str, default='fdeint',
                        choices=['odeint_adjoint', 'odeint', 'odeint_mem', 'fdeint_adjoint', 'fdeint', 'fdeint_mem'],
                        help='Differential equation integrator type (default: fdeint)')

    parser.add_argument('--method', type=str, default='gl', choices=[
        'euler',  # odeint only supports euler method
        'gl-f', 'gl-o', 'trap-f', 'trap-o',
        'l1-f', 'l1-o', 'pred-f', 'pred-o',  # fdeint_adjoint methods
        'pred', 'gl', 'trap', 'l1', 'glmulti'  # fdeint methods
    ], help='Method for solver (default: gl)')

    parser.add_argument('--step_size', type=float, default=1.0,
                        help='Integration step size (default: 1.0)')
    parser.add_argument('--time_interval', type=float, default=1.0,
                        help='Time interval between steps in ms (default: 1.0)')
    parser.add_argument('--memory', type=int, default=-1,
                        help='Memory window size for FDE solver. '
                             '-1 means full memory (default: -1)')

    # Fractional order parameters (unified alpha)
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5],
                        help='Fractional order(s). Single value for single-term FDE, '
                             'multiple values for multi-term FDE. '
                             'Examples: --alpha 0.5 (single) or --alpha 0.3 0.7 0.9 (multi)')

    # Multi-term coefficient (only needed when alpha has multiple values)
    parser.add_argument('--multi_coefficient', type=float, nargs='+', default=None,
                        help='Coefficients for multi-term FDE. Required when --alpha has multiple values. '
                             'Example: --multi_coefficient 1.0 0.5 0.2')

    parser.add_argument('--learn_coefficient', action='store_true',
                        help='Make multi-term coefficients learnable')

    return parser.parse_args()


def process_alpha_args(args):
    """
    Process alpha arguments and validate multi-term configuration.

    Returns:
        tuple: (alpha, multi_coefficient) where alpha is float or list
    """
    if len(args.alpha) == 1:
        # Single-term FDE
        alpha = args.alpha[0]
        if args.multi_coefficient is not None:
            print("Warning: --multi_coefficient is ignored for single-term FDE (single alpha value)")
        multi_coefficient = None
        print(f"[Single-term FDE] Fractional order (alpha): {alpha}")
    else:
        # Multi-term FDE
        alpha = args.alpha
        multi_coefficient = args.multi_coefficient

        # Validate
        if multi_coefficient is None:
            raise ValueError(
                f"--multi_coefficient is required when --alpha has multiple values. "
                f"Got alpha={alpha}"
            )
        if len(alpha) != len(multi_coefficient):
            raise ValueError(
                f"--alpha (len={len(alpha)}) and --multi_coefficient "
                f"(len={len(multi_coefficient)}) must have the same length"
            )
        print(f"[Multi-term FDE] Enabled:")
        print(f"  - Fractional orders (alpha): {alpha}")
        print(f"  - Coefficients: {multi_coefficient}")
        print(f"  - Learnable: {args.learn_coefficient}")

    return alpha, multi_coefficient


def spike_converter(x, time_steps=100, flatten=False):
    """
    Convert input to spike sequence.

    Args:
        x: Input data [batch_size, channels, height, width] or [batch_size, features]
        time_steps: Number of time steps
        flatten: Whether to flatten input (for MLP model)

    Returns:
        Spike sequence [time_steps, batch_size, ...]
    """
    batch_size = x.size(0)

    if flatten:
        # For MLP model, flatten input
        x = x.view(batch_size, -1)
        p = x.unsqueeze(1).repeat(1, time_steps, 1)
        spikes = torch.bernoulli(p)
        return spikes.permute(1, 0, 2)  # [time_steps, batch_size, features]
    else:
        # For CNN model, keep image dimensions
        if len(x.shape) == 4:  # [batch_size, channels, height, width]
            p = x.unsqueeze(1).repeat(1, time_steps, 1, 1, 1)
            spikes = torch.bernoulli(p)
            return spikes.permute(1, 0, 2, 3, 4)  # [time_steps, batch_size, channels, height, width]
        elif len(x.shape) == 3:  # [batch_size, height, width]
            p = x.unsqueeze(1).unsqueeze(1).repeat(1, time_steps, 1, 1, 1)
            spikes = torch.bernoulli(p)
            return spikes.permute(1, 0, 2, 3, 4)
        else:  # Already flattened vector
            x = x.view(batch_size, -1)
            p = x.unsqueeze(1).repeat(1, time_steps, 1)
            spikes = torch.bernoulli(p)
            return spikes.permute(1, 0, 2)


class CNNTest(nn.Module):
    """
    CNN with LIFNeuron example
    """

    def __init__(self, tau: float, threshold: float, surrogate_grad_scale: float):
        super().__init__()
        # First conv layer + spiking neuron + pooling
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.lif1 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Second conv layer + spiking neuron + pooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.lif2 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layers + spiking neurons
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.lif3 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.fc2 = nn.Linear(128, 10)
        self.lif4 = LIFNeuron(tau, threshold, surrogate_grad_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 28, 28]
        out = self.conv1(x)
        out = self.lif1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.lif2(out)
        out = self.pool2(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.lif3(out)
        out = self.fc2(out)
        out = self.lif4(out)

        return out


class MLPTest(nn.Module):
    """
    MLP with LIFNeuron example
    """

    def __init__(self, tau: float, threshold: float, surrogate_grad_scale: float):
        super().__init__()
        # Flatten input
        self.flatten = nn.Flatten()
        # First FC layer + spiking neuron
        self.fc1 = nn.Linear(28 * 28, 2560, bias=False)
        self.lif1 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        # Second FC layer + spiking neuron
        self.fc2 = nn.Linear(2560, 10, bias=False)
        self.lif2 = LIFNeuron(tau, threshold, surrogate_grad_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 28, 28] or [batch, 28*28]
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.lif1(out)
        out = self.fc2(out)
        out = self.lif2(out)
        return out


def print_total_parameters(model):
    """Print total number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def train(args, model, device, train_loader, optimizer, criterion, epoch, alpha, multi_coefficient):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0

    # Set up time grid
    data_time = torch.linspace(0, args.time_interval * args.time_steps,
                               args.time_steps + 1, device=device).float()

    # Integration options
    options = {'step_size': args.step_size, 'memory': args.memory}

    # Progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)

        # Convert data to spike sequence
        use_flatten = args.model == 'mlp'
        data = spike_converter(data, time_steps=args.time_steps, flatten=use_flatten).to(device)
        data = 10 * data  # Scale spike intensity

        optimizer.zero_grad()

        # Forward pass
        output = model(data, data_time, output_time=data_time, method=args.method, options=options)
        output = output.mean(0)

        # Compute loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update statistics
        train_samples += target.numel()
        train_loss += loss.item() * target.numel()
        train_acc += (output.argmax(1) == target).float().sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{train_loss / train_samples:.4f}',
            'Acc': f'{100. * train_acc / train_samples:.2f}%'
        })

        # Reset NFE counter
        model.ode_func.nfe = 0

    # Clean GPU cache
    torch.cuda.empty_cache()

    return train_loss / train_samples, train_acc / train_samples


def test(args, model, device, test_loader, criterion):
    """Test the model."""
    model.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0

    # Set up time grid
    data_time = torch.linspace(0, args.time_interval * args.time_steps,
                               args.time_steps + 1, device=device).float()

    # Integration options
    options = {'step_size': args.step_size, 'memory': args.memory}

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            data, target = data.to(device), target.to(device)

            # Convert data to spike sequence
            use_flatten = args.model == 'mlp'
            data = spike_converter(data, time_steps=args.time_steps, flatten=use_flatten).to(device)
            data = 10 * data  # Scale spike intensity

            # Forward pass
            output = model(data, data_time, output_time=data_time, method=args.method, options=options)
            output = output.mean(0)

            # Compute loss
            test_loss += criterion(output, target).item() * target.numel()

            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            test_acc += pred.eq(target.view_as(pred)).sum().item()
            test_samples += target.numel()

    test_loss /= test_samples
    accuracy = 100. * test_acc / test_samples

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {test_acc}/{test_samples} ({accuracy:.2f}%)')

    # Clean GPU cache
    torch.cuda.empty_cache()

    return accuracy


def main():
    # Parse command line arguments
    args = parse_args()

    # Process alpha arguments
    alpha, multi_coefficient = process_alpha_args(args)

    print("=" * 60)
    print("Configuration:")
    print(args)
    print("=" * 60)

    # Set CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.clamp(0, 1)  # Ensure values are in [0, 1]
    ])

    # Load datasets
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create base network
    if args.model == 'cnn':
        base_network = CNNTest(args.tau, args.threshold, args.surrogate_grad_scale)
        input_shape = (1, 1, 28, 28)  # (batch, channels, height, width)
    else:
        base_network = MLPTest(args.tau, args.threshold, args.surrogate_grad_scale)
        input_shape = (1, 1, 28, 28)

    # Create SNNWrapper with unified alpha
    snn_model = SNNWrapper(
        base_network,
        integrator=args.integrator,
        alpha=alpha,
        multi_coefficient=multi_coefficient,
        learn_coefficient=args.learn_coefficient
    ).to(device)

    # Initialize neuron shapes
    snn_model._set_neuron_shapes(input_shape=input_shape)

    print("\n" + "=" * 60)
    print("Model Architecture:")
    print(snn_model)
    print("=" * 60)
    print_total_parameters(snn_model)
    print("=" * 60)

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(snn_model.parameters(), lr=args.lr)

    # Training and testing
    best_accuracy = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train(args, snn_model, device, train_loader,
                                      optimizer, criterion, epoch, alpha, multi_coefficient)

        # Test
        accuracy = test(args, snn_model, device, test_loader, criterion)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {100. * train_acc:.2f}%, "
              f"Test Acc: {accuracy:.2f}%, Time: {epoch_time:.2f}s")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(snn_model.state_dict(), f"snn_{args.model}_best.pt")
            print(f"  -> New best model saved!")

    print("=" * 60)
    print(f"Training complete! Best test accuracy: {best_accuracy:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()