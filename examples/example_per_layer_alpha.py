"""
Per-Layer Alpha Configuration Examples for _f_-SNN

See also docs/per_layer_alpha.md and spikeDE/snn.py

This file demonstrates all alpha configuration cases for the fractional SNN framework.
It uses a simple 2-layer network with 3-term multi-term examples for consistency.

=============================================================================
Mathematical Background (2-layer network examples)
=============================================================================

The membrane potential V_ℓ(t) of neuron layer ℓ evolves according to:

Config 1 - Same α for all layers (2 layers, scalar α):
    D^α V_ℓ(t) = f_ℓ(t, V_ℓ(t)),  ∀ℓ ∈ {0, 1}

Config 2 - Per-layer α (2 layers, different α):
    D^{α_ℓ} V_ℓ(t) = f_ℓ(t, V_ℓ(t)),  α₀=0.3, α₁=0.7

Config 3 - Multi-term broadcast (2 layers, 3-term):
    Σⱼ wⱼ D^{αⱼ} V_ℓ(t) = f_ℓ(t, V_ℓ(t)),  α=[0.3, 0.5, 0.7]

Config 4 - Per-layer multi-term (2 layers, Layer 0: 2-term, Layer 1: 3-term):
    Σⱼ w^ℓ_j D^{α^ℓ_j} V_ℓ(t) = f_ℓ(t, V_ℓ(t))

=============================================================================
Usage
=============================================================================
    python example_alpha_cases.py --list      # List all cases
    python example_alpha_cases.py --case 1    # Run Config 1
    python example_alpha_cases.py --case all  # Run all cases
"""

import torch
import torch.nn as nn
import argparse
import warnings
warnings.filterwarnings("ignore")

# Import from spikeDE package
# Adjust import path as needed for your setup
try:
    from spikeDE import SNNWrapper
    from spikeDE import LIFNeuron
except ImportError:
    print("Note: spikeDE package not found. Using local imports.")


# =============================================================================
# Simple 2-Layer SNN for Testing
# =============================================================================

class SimpleSNN(nn.Module):
    """
    A minimal SNN with 2 neuron layers for demonstrating alpha configurations.

    This is the standard example network used throughout this file.
    All examples use n_layers = 2 for consistency.
    Multi-term examples use 3 terms for consistency.

    Architecture:
        input (10) → Linear(10, 20) → LIF₀ → Linear(20, 10) → LIF₁ → output (10)
    """

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=10):
        super().__init__()

        # Layer 0: input → hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lif1 = LIFNeuron(tau=2.0, threshold=1.0)

        # Layer 1: hidden → output
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.lif2 = LIFNeuron(tau=2.0, threshold=1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.lif1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.lif2(x)

        return x


# =============================================================================
# Helper Functions
# =============================================================================

def create_dummy_data(batch_size=4, time_steps=8, input_dim=10, device='cpu'):
    """Create dummy input data for testing."""
    x = torch.randn(time_steps, batch_size, input_dim, device=device)
    x_time = torch.linspace(0, 1, time_steps, device=device)
    return x, x_time


def run_forward_pass(net, x, x_time, method='gl'):
    """Run a forward pass through the network."""
    output = net(x, x_time, method=method, options={'step_size': 0.1, 'memory': -1})
    return output


def print_separator(title):
    """Print a formatted separator."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(net):
    """Print configuration result."""
    print(f"\nDetected case: {net._alpha_case}")
    print(f"Per-layer alpha: {[p.tolist() for p in net.per_layer_alpha_params]}")
    if hasattr(net, 'per_layer_coefficient_params'):
        print(f"Per-layer coef:  {[p.tolist() for p in net.per_layer_coefficient_params]}")


# =============================================================================
# Configuration 1: Same α for All Layers
# =============================================================================

def config_1_scalar_alpha():
    """
    Configuration 1: Same α for All Layers (2-layer network)

    Mathematical formulation:
        D^α V_ℓ(t) = f_ℓ(t, V_ℓ(t)),  ∀ℓ ∈ {0, 1}

    Both layers share the same fractional order α = 0.5.
    This is the simplest configuration.
    """
    print_separator("Config 1: Same α for All Layers")
    print("2-layer network, both layers use α = 0.5")
    print("Math: D^α V_ℓ(t) = f_ℓ(t, V_ℓ(t)), same α for all layers")
    print("\nCode:")
    print("  alpha = 0.5              # Scalar")
    print("  learn_alpha = False")

    network = SimpleSNN()
    net = SNNWrapper(
        network,
        integrator='fdeint',
        alpha=0.5,
        learn_alpha=False,
    )

    net._set_neuron_shapes(input_shape=(1, 10))
    print_result(net)

    # Test forward pass
    x, x_time = create_dummy_data()
    output = run_forward_pass(net, x, x_time)
    print(f"Output shape: {output.shape}")

    return net


# =============================================================================
# Configuration 2: Per-Layer α
# =============================================================================

def config_2_per_layer_alpha():
    """
    Configuration 2: Per-Layer α (2-layer network)

    Mathematical formulation:
        D^{α_ℓ} V_ℓ(t) = f_ℓ(t, V_ℓ(t)),  ℓ ∈ {0, 1}

    Layer 0: α₀=0.3, Layer 1: α₁=0.7
    Different fractional orders allow different memory characteristics per layer:
    - Lower α_ℓ → Stronger memory effect
    - Higher α_ℓ → Weaker memory effect (closer to ODE)
    """
    print_separator("Config 2: Per-Layer α")
    print("2-layer network: Layer 0 α=0.3, Layer 1 α=0.7")
    print("Math: D^{α_ℓ} V_ℓ(t) = f_ℓ(t, V_ℓ(t)), different α per layer")
    print("\nCode:")
    print("  alpha = [0.3, 0.7]       # Layer 0: α=0.3, Layer 1: α=0.7")
    print("  alpha_mode = 'per_layer'")
    print("  learn_alpha = [True, False]  # Only layer 0 is learnable")

    network = SimpleSNN()
    net = SNNWrapper(
        network,
        integrator='fdeint',
        alpha=[0.3, 0.7],
        alpha_mode='per_layer',
        learn_alpha=[True, False],
    )

    net._set_neuron_shapes(input_shape=(1, 10))
    print_result(net)
    print(f"Alpha requires_grad: {[p.requires_grad for p in net.per_layer_alpha_params]}")

    x, x_time = create_dummy_data()
    output = run_forward_pass(net, x, x_time)
    print(f"Output shape: {output.shape}")

    return net


# =============================================================================
# Configuration 3: Multi-Term (Distributed Order)
# =============================================================================

def config_3_multiterm_broadcast():
    """
    Configuration 3: Multi-Term / Distributed Order (2-layer network, 3-term)

    Mathematical formulation:
        Σⱼ wⱼ D^{αⱼ} V_ℓ(t) = f_ℓ(t, V_ℓ(t)),  ∀ℓ

    Both layers use: w₁·D^0.3 V + w₂·D^0.5 V + w₃·D^0.7 V = f(t, V)
    Distributed-order FDE: weighted sum of multiple fractional derivatives.
    This captures dynamics spanning multiple time scales.
    """
    print_separator("Config 3: Multi-Term (Distributed Order)")
    print("2-layer network, 3-term broadcast to all layers")
    print("Math: w₁·D^0.3 V + w₂·D^0.5 V + w₃·D^0.7 V = f(t, V)")
    print("\nCode:")
    print("  alpha = [0.3, 0.5, 0.7]           # 3-term")
    print("  multi_coefficient = [1.0, 0.5, 0.2]")
    print("  alpha_mode = 'multiterm'")
    print("  learn_coefficient = True          # Learn the weights")

    network = SimpleSNN()
    net = SNNWrapper(
        network,
        integrator='fdeint',
        alpha=[0.3, 0.5, 0.7],
        multi_coefficient=[1.0, 0.5, 0.2],
        alpha_mode='multiterm',
        learn_alpha=False,
        learn_coefficient=True,
    )

    net._set_neuron_shapes(input_shape=(1, 10))
    print_result(net)
    print(f"Coef requires_grad: {[p.requires_grad for p in net.per_layer_coefficient_params]}")

    x, x_time = create_dummy_data()
    output = run_forward_pass(net, x, x_time)
    print(f"Output shape: {output.shape}")

    return net


# =============================================================================
# Configuration 4: Per-Layer Multi-Term
# =============================================================================

def config_4_per_layer_multiterm():
    """
    Configuration 4: Per-Layer Multi-Term (2-layer network)

    Mathematical formulation:
        Σⱼ w^ℓ_j D^{α^ℓ_j} V_ℓ(t) = f_ℓ(t, V_ℓ(t)),  ℓ ∈ {0, 1}

    Layer 0: 2-term (α=[0.3, 0.5])
    Layer 1: 3-term (α=[0.4, 0.6, 0.8])
    Each layer has its own set of fractional orders and weights.
    This is the most flexible configuration.
    """
    print_separator("Config 4: Per-Layer Multi-Term")
    print("2-layer network: Layer 0 has 2-term, Layer 1 has 3-term")
    print("Math: Σⱼ w^ℓ_j D^{α^ℓ_j} V_ℓ(t) = f_ℓ(t, V_ℓ(t)), per-layer multi-term")
    print("\nCode:")
    print("  alpha = [[0.3, 0.5], [0.4, 0.6, 0.8]]  # Nested structure")
    print("  multi_coefficient = [[1.0, 0.5], [1.0, 0.3, 0.1]]")
    print("  # alpha_mode is ignored when alpha is nested")

    network = SimpleSNN()
    net = SNNWrapper(
        network,
        integrator='fdeint',
        alpha=[
            [0.3, 0.5],        # Layer 0: 2-term
            [0.4, 0.6, 0.8],   # Layer 1: 3-term
        ],
        multi_coefficient=[
            [1.0, 0.5],        # Layer 0 coefficients
            [1.0, 0.3, 0.1],   # Layer 1 coefficients
        ],
        learn_alpha=[True, False],
        learn_coefficient=[True, True],
    )

    net._set_neuron_shapes(input_shape=(1, 10))
    print_result(net)

    x, x_time = create_dummy_data()
    output = run_forward_pass(net, x, x_time)
    print(f"Output shape: {output.shape}")

    return net


# =============================================================================
# Advanced: Multi-term with Per-Layer Coefficients
# =============================================================================

def advanced_multiterm_per_layer_coef():
    """
    Advanced: Multi-term with Per-Layer Coefficients (2-layer network, 3-term)

    Same alpha values broadcast to all layers, but different coefficients per layer.
    Useful when you want the same fractional orders but different weightings.
    """
    print_separator("Advanced: Multi-term with Per-Layer Coefficients")
    print("2-layer network, 3-term: same α, different coefficients per layer")
    print("\nCode:")
    print("  alpha = [0.3, 0.5, 0.7]  # 3-term, broadcast to all")
    print("  multi_coefficient = [[1.0, 0.5, 0.2], [0.8, 0.3, 0.1]]  # Nested, per-layer")
    print("  alpha_mode = 'multiterm'  # Required!")

    network = SimpleSNN()
    net = SNNWrapper(
        network,
        integrator='fdeint',
        alpha=[0.3, 0.5, 0.7],               # 3-term
        multi_coefficient=[
            [1.0, 0.5, 0.2],  # Layer 0 coefficients
            [0.8, 0.3, 0.1],  # Layer 1 coefficients
        ],
        alpha_mode='multiterm',
    )

    net._set_neuron_shapes(input_shape=(1, 10))
    print_result(net)

    x, x_time = create_dummy_data()
    output = run_forward_pass(net, x, x_time)
    print(f"Output shape: {output.shape}")

    return net


# =============================================================================
# Auto-Detection Cases
# =============================================================================

def auto_detect_with_coef():
    """
    Auto-detection: Flat alpha + flat coefficient → Mode B (2-layer network, 3-term)

    When multi_coefficient is provided with matching length,
    auto mode correctly detects this as multi-term broadcast.
    """
    print_separator("Auto-detect: Flat alpha + flat coef → Mode B")
    print("2-layer network, 3-term with auto detection")
    print("\nCode:")
    print("  alpha = [0.3, 0.5, 0.7]           # 3-term")
    print("  multi_coefficient = [1.0, 0.5, 0.2]")
    print("  alpha_mode = 'auto'               # Will detect as Mode B")

    network = SimpleSNN()
    net = SNNWrapper(
        network,
        integrator='fdeint',
        alpha=[0.3, 0.5, 0.7],
        multi_coefficient=[1.0, 0.5, 0.2],
        alpha_mode='auto',
    )

    net._set_neuron_shapes(input_shape=(1, 10))
    print_result(net)

    x, x_time = create_dummy_data()
    output = run_forward_pass(net, x, x_time)
    print(f"Output shape: {output.shape}")

    return net


def auto_detect_per_layer_coef():
    """
    Auto-detection: Flat alpha + nested coefficient → Mode B

    """
    print_separator("Auto-detect: Flat alpha + nested coef → Mode B")
    print("Code:")
    print("  alpha = [0.3, 0.5]  # Flat")
    print("  multi_coefficient = [[1.0, 0.5], [0.8, 0.3]]  # Nested")
    print("  alpha_mode = 'auto'  # Detect as Mode B!")

    network = SimpleSNN()

    net = SNNWrapper(
            network,
            integrator='fdeint',
            alpha=[0.3, 0.5],
            multi_coefficient=[
                [1.0, 0.5],
                [0.8, 0.3],
            ],
            alpha_mode='auto',
        )
    net._set_neuron_shapes(input_shape=(1, 10))
    print_result(net)

    x, x_time = create_dummy_data()
    output = run_forward_pass(net, x, x_time)
    print(f"Output shape: {output.shape}")

    return net



def auto_detect_heuristic_a():
    """
    Heuristic: len(alpha) == n_layers → defaults to Mode A

    When alpha_mode='auto' and len(alpha) matches n_layers,
    the system defaults to per-layer single-term (Mode A) with a warning.
    """
    print_separator("Heuristic: len(alpha) == n_layers → Mode A")
    print("Code:")
    print("  alpha = [0.3, 0.7]  # 2 values, 2 layers")
    print("  alpha_mode = 'auto'  # Ambiguous, will warn")

    network = SimpleSNN()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        net = SNNWrapper(
            network,
            integrator='fdeint',
            alpha=[0.3, 0.7],
            alpha_mode='auto',
        )

        net._set_neuron_shapes(input_shape=(1, 10))

        if w:
            print(f"\n⚠ Warning issued:")
            print(f"  {w[-1].message}")

    print_result(net)

    x, x_time = create_dummy_data()
    output = run_forward_pass(net, x, x_time)
    print(f"Output shape: {output.shape}")

    return net


def auto_detect_heuristic_b():
    """
    Heuristic: len(alpha) != n_layers → defaults to Mode B (2-layer network, 3-term)

    When alpha_mode='auto' and len(alpha) doesn't match n_layers,
    the system defaults to multi-term broadcast (Mode B) with a warning.
    Here: 3 alpha values for 2 layers → clearly Mode B.
    """
    print_separator("Heuristic: len(alpha) != n_layers → Mode B")
    print("2-layer network, 3-term (3 values ≠ 2 layers → detected as Mode B)")
    print("\nCode:")
    print("  alpha = [0.3, 0.5, 0.7]  # 3 values, but 2 layers")
    print("  alpha_mode = 'auto'      # Will detect as Mode B, with warning")

    network = SimpleSNN()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        net = SNNWrapper(
            network,
            integrator='fdeint',
            alpha=[0.3, 0.5, 0.7],
            alpha_mode='auto',
        )

        net._set_neuron_shapes(input_shape=(1, 10))

        if w:
            print(f"\n⚠ Warning issued:")
            print(f"  {w[-1].message}")

    print_result(net)

    x, x_time = create_dummy_data()
    output = run_forward_pass(net, x, x_time)
    print(f"Output shape: {output.shape}")

    return net


# =============================================================================
# Learnable Parameters
# =============================================================================

def learnable_alpha():
    """
    Learnable Alpha: Gradient-based optimization of α

    The fractional order α can be learned during training.
    Gradients flow through the alpha parameters.
    """
    print_separator("Learnable Alpha")
    print("Code:")
    print("  alpha = [0.3, 0.7]")
    print("  alpha_mode = 'per_layer'")
    print("  learn_alpha = [True, False]  # Layer 0 learnable, Layer 1 fixed")

    network = SimpleSNN()
    net = SNNWrapper(
        network,
        integrator='fdeint',
        alpha=[0.3, 0.7],
        alpha_mode='per_layer',
        learn_alpha=[True, False],
    )

    net._set_neuron_shapes(input_shape=(1, 10))
    print_result(net)
    print(f"Alpha requires_grad: {[p.requires_grad for p in net.per_layer_alpha_params]}")

    # Verify gradient flow
    x, x_time = create_dummy_data()
    output = run_forward_pass(net, x, x_time)
    loss = output.sum()
    loss.backward()

    print(f"\nAfter backward pass:")
    for i, p in enumerate(net.per_layer_alpha_params):
        if p.grad is not None:
            print(f"  Layer {i}: grad exists (learnable)")
        else:
            print(f"  Layer {i}: grad=None (fixed)")

    return net


def learnable_coefficient():
    """
    Learnable Coefficients: Learn the mixing weights (2-layer network, 3-term)

    In multi-term mode, the coefficients w_j can be learned
    while keeping the fractional orders fixed.
    """
    print_separator("Learnable Coefficients")
    print("2-layer network, 3-term with learnable coefficients")
    print("\nCode:")
    print("  alpha = [0.3, 0.5, 0.7]           # 3-term, fixed")
    print("  multi_coefficient = [1.0, 0.5, 0.2]  # Learnable")
    print("  learn_alpha = False")
    print("  learn_coefficient = True")

    network = SimpleSNN()
    net = SNNWrapper(
        network,
        integrator='fdeint',
        alpha=[0.3, 0.5, 0.7],               # 3-term
        multi_coefficient=[1.0, 0.5, 0.2],
        alpha_mode='multiterm',
        learn_alpha=False,
        learn_coefficient=True,
    )

    net._set_neuron_shapes(input_shape=(1, 10))
    print_result(net)
    print(f"Coef requires_grad: {[p.requires_grad for p in net.per_layer_coefficient_params]}")

    # Verify gradient flow
    x, x_time = create_dummy_data()
    output = run_forward_pass(net, x, x_time)
    loss = output.sum()
    loss.backward()

    print(f"\nAfter backward pass:")
    for i, p in enumerate(net.per_layer_coefficient_params):
        if p.grad is not None:
            print(f"  Layer {i} coef: grad exists (learnable)")
        else:
            print(f"  Layer {i} coef: grad=None (fixed)")

    return net


# =============================================================================
# Case Registry
# =============================================================================

ALL_CASES = {
    # Main Configurations (1-4)
    1: ("Config 1: Same α for all layers", config_1_scalar_alpha),
    2: ("Config 2: Per-layer α", config_2_per_layer_alpha),
    3: ("Config 3: Multi-term (distributed order)", config_3_multiterm_broadcast),
    4: ("Config 4: Per-layer multi-term", config_4_per_layer_multiterm),

    # Advanced
    5: ("Advanced: Multi-term with per-layer coef", advanced_multiterm_per_layer_coef),

    # Auto-detection
    6: ("Auto-detect: flat + flat coef → Mode B", auto_detect_with_coef),
    7: ("Auto-detect: flat + nested coef → Mode B", auto_detect_per_layer_coef),
    '8a': ("Heuristic: len==n_layers → Mode A", auto_detect_heuristic_a),
    '8b': ("Heuristic: len!=n_layers → Mode B", auto_detect_heuristic_b),

    # Learnable
    9: ("Learnable: Alpha parameters", learnable_alpha),
    10: ("Learnable: Coefficient parameters", learnable_coefficient),
}


def list_cases():
    """Print list of all available cases."""
    print("\n" + "=" * 70)
    print("  PER-LAYER ALPHA CONFIGURATION EXAMPLES (2-layer network)")
    print("=" * 70)

    print("\n  MAIN CONFIGURATIONS")
    print("  " + "-" * 66)
    print("  Case 1:  Same α for all layers       D^α V_ℓ = f_ℓ(t, V_ℓ)")
    print("  Case 2:  Per-layer α                 D^{α_ℓ} V_ℓ = f_ℓ(t, V_ℓ)")
    print("  Case 3:  Multi-term broadcast (3-term) Σⱼ wⱼ D^{αⱼ} V_ℓ = f_ℓ(t, V_ℓ)")
    print("  Case 4:  Per-layer multi-term        Σⱼ w^ℓ_j D^{α^ℓ_j} V_ℓ = f_ℓ(t, V_ℓ)")

    print("\n  ADVANCED CONFIGURATION")
    print("  " + "-" * 66)
    print("  Case 5:  Multi-term with per-layer coefficients")

    print("\n  AUTO-DETECTION BEHAVIOR")
    print("  " + "-" * 66)
    print("  Case 6:  auto + flat coef → detected as Mode B")
    print("  Case 7:  auto + nested coef → detected as Mode B")
    print("  Case 8a: auto + len==n_layers → defaults to Mode A (warning)")
    print("  Case 8b: auto + len!=n_layers → defaults to Mode B (warning)")

    print("\n  LEARNABLE PARAMETERS")
    print("  " + "-" * 66)
    print("  Case 9:  Learnable alpha")
    print("  Case 10: Learnable coefficients")

    print("\n" + "=" * 70)
    print("  USAGE")
    print("  " + "-" * 66)
    print("  python example_alpha_cases.py --case 1     # Run Case 1")
    print("  python example_alpha_cases.py --case 8a    # Run Case 8a")
    print("  python example_alpha_cases.py --case all   # Run all cases")
    print("  python example_alpha_cases.py --list       # Show this list")
    print("=" * 70 + "\n")


def run_case(case_key):
    """Run a specific case."""
    if case_key not in ALL_CASES:
        print(f"Unknown case: {case_key}")
        list_cases()
        return

    name, func = ALL_CASES[case_key]
    func()


def run_all_cases():
    """Run all cases."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  RUNNING ALL ALPHA CONFIGURATION CASES".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    for key in [1, 2, 3, 4, 5, 6, 7, '8a', '8b', 9, 10]:
        name, func = ALL_CASES[key]
        try:
            func()
        except Exception as e:
            print(f"\n❌ Case {key} failed: {e}")

    print("\n" + "=" * 70)
    print("  ALL CASES COMPLETED")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Per-Layer Alpha Configuration Examples for _f_-SNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python example_alpha_cases.py --list        # List all cases
  python example_alpha_cases.py --case 1      # Config 1: Same α for all
  python example_alpha_cases.py --case 2      # Config 2: Per-layer α
  python example_alpha_cases.py --case 3      # Config 3: Multi-term
  python example_alpha_cases.py --case 4      # Config 4: Per-layer multi-term
  python example_alpha_cases.py --case all    # Run all cases
        """
    )
    parser.add_argument('--case', type=str, default='all',
                        help='Case number to run (1-10, 8a, 8b) or "all"')
    parser.add_argument('--list', action='store_true',
                        help='List all available cases')

    args = parser.parse_args()

    if args.list:
        list_cases()
        return

    if args.case.lower() == 'all':
        run_all_cases()
    else:
        try:
            case_key = int(args.case)
        except ValueError:
            case_key = args.case
        run_case(case_key)


if __name__ == '__main__':
    main()