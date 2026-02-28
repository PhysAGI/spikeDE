"""
Test file for ODEFuncFromFX

This file tests the core functionality described in odefunc_fx.md:
1. Single Boundary Case: Sequential SNN → post_neuron_module receives tensor
2. Multi-Boundary Case: Branching SNN → post_neuron_module receives tuple
3. ODE graph output structure: (dv1/dt, ..., dvN/dt, boundary_1, ...)
4. Interpolation methods
5. Neuron counting

Run with: python test_odefunc.py
"""

import torch
import torch.nn as nn
from spikeDE import LIFNeuron, IFNeuron
from spikeDE.odefunc import ODEFuncFromFX, interpolate

print("=" * 70)
print("ODEFuncFromFX Test Suite")
print("=" * 70)


# ============================================================================
# Test Case 1: Single Boundary (Sequential SNN)
# ============================================================================
# Network: input → fc1 → lif1 → fc2 → lif2 → fc3 (post-processing)
# Expected: 1 boundary (output of lif2), post_neuron_module receives tensor

class SequentialSNN(nn.Module):
    """Simple sequential SNN with post-processing layer."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.lif1 = LIFNeuron()
        self.fc2 = nn.Linear(20, 15)
        self.lif2 = LIFNeuron()
        self.fc3 = nn.Linear(15, 5)  # Post-neuron processing

    def forward(self, x):
        x = self.lif1(self.fc1(x))
        x = self.lif2(self.fc2(x))
        x = self.fc3(x)  # After last neuron
        return x


def test_single_boundary():
    print("\n" + "-" * 70)
    print("TEST 1: Single Boundary (Sequential SNN)")
    print("-" * 70)

    model = SequentialSNN()
    ode_func = ODEFuncFromFX(model)

    # Check neuron count
    print(f"\n[Check] neuron_count = {ode_func.neuron_count}")
    assert ode_func.neuron_count == 2, f"Expected 2 neurons, got {ode_func.neuron_count}"
    print("  ✓ Correct neuron count")

    # Prepare dummy inputs
    batch_size = 4
    dummy_t = torch.tensor(0.5)
    dummy_v_mems = (
        torch.randn(batch_size, 20),  # v_mem for lif1
        torch.randn(batch_size, 15),  # v_mem for lif2
    )
    dummy_x = torch.randn(8, batch_size, 10)  # [T, B, features]
    dummy_x_time = torch.linspace(0, 1, 8)

    # Run ode_gm
    ode_func.set_inputs(dummy_x, dummy_x_time)
    result = ode_func.ode_gm(dummy_t, dummy_v_mems, dummy_x, dummy_x_time)

    print(f"\n[Check] ode_gm output structure:")
    print(f"  Output is tuple: {isinstance(result, tuple)}")
    print(f"  Number of outputs: {len(result)}")

    # Expected: (dv1/dt, dv2/dt, boundary_1)
    n_outputs = len(result)
    n_boundaries = n_outputs - ode_func.neuron_count
    print(f"  neuron_count: {ode_func.neuron_count}")
    print(f"  n_boundaries: {n_boundaries}")

    assert n_boundaries == 1, f"Expected 1 boundary, got {n_boundaries}"
    print("  ✓ Correct number of boundaries")

    # Check shapes
    print(f"\n[Check] Output shapes:")
    for i, out in enumerate(result):
        if i < ode_func.neuron_count:
            print(f"  dv{i+1}/dt shape: {tuple(out.shape)}")
        else:
            print(f"  boundary_{i - ode_func.neuron_count + 1} shape: {tuple(out.shape)}")

    # Check post_neuron_module
    print(f"\n[Check] post_neuron_module:")
    post_module = ode_func.get_post_neuron_module()
    print(f"  Type: {type(post_module).__name__}")

    # Test post_neuron_module with single tensor input
    boundary_output = result[-1]  # Last output is boundary
    post_result = post_module(boundary_output)
    print(f"  Input shape: {tuple(boundary_output.shape)}")
    print(f"  Output shape: {tuple(post_result.shape)}")
    assert post_result.shape == (batch_size, 5), f"Expected (4, 5), got {post_result.shape}"
    print("  ✓ post_neuron_module works with single tensor")

    print("\n✓ TEST 1 PASSED: Single Boundary Case")
    return True


# ============================================================================
# Test Case 2: Multi-Boundary (Branching SNN with Skip Connection)
# ============================================================================
# Network with skip connection:
#   input → fc1 → lif1 → fc2 → lif2 ─┬─→ concat → fc_out
#                    └────────────────┘
# The concat layer needs BOTH lif1 output AND lif2 output
# Expected: 2 boundaries, post_neuron_module receives tuple

class BranchingSNN(nn.Module):
    """SNN with skip connection requiring multiple boundaries."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.lif1 = LIFNeuron()
        self.fc2 = nn.Linear(20, 20)  # Same size for easy concat
        self.lif2 = LIFNeuron()
        self.fc_out = nn.Linear(40, 5)  # 20 + 20 from concat

    def forward(self, x):
        x1 = self.lif1(self.fc1(x))
        x2 = self.lif2(self.fc2(x1))
        # Skip connection: concat lif1 and lif2 outputs
        x_cat = torch.cat([x1, x2], dim=-1)
        out = self.fc_out(x_cat)
        return out


def test_multi_boundary():
    print("\n" + "-" * 70)
    print("TEST 2: Multi-Boundary (Branching SNN with Skip Connection)")
    print("-" * 70)

    model = BranchingSNN()
    ode_func = ODEFuncFromFX(model)

    # Check neuron count
    print(f"\n[Check] neuron_count = {ode_func.neuron_count}")
    assert ode_func.neuron_count == 2, f"Expected 2 neurons, got {ode_func.neuron_count}"
    print("  ✓ Correct neuron count")

    # Prepare dummy inputs
    batch_size = 4
    dummy_t = torch.tensor(0.5)
    dummy_v_mems = (
        torch.randn(batch_size, 20),  # v_mem for lif1
        torch.randn(batch_size, 20),  # v_mem for lif2
    )
    dummy_x = torch.randn(8, batch_size, 10)
    dummy_x_time = torch.linspace(0, 1, 8)

    # Run ode_gm
    ode_func.set_inputs(dummy_x, dummy_x_time)
    result = ode_func.ode_gm(dummy_t, dummy_v_mems, dummy_x, dummy_x_time)

    print(f"\n[Check] ode_gm output structure:")
    print(f"  Output is tuple: {isinstance(result, tuple)}")
    print(f"  Number of outputs: {len(result)}")

    n_outputs = len(result)
    n_boundaries = n_outputs - ode_func.neuron_count
    print(f"  neuron_count: {ode_func.neuron_count}")
    print(f"  n_boundaries: {n_boundaries}")

    # For skip connection, we expect 2 boundaries (lif1 and lif2 outputs)
    assert n_boundaries == 2, f"Expected 2 boundaries for skip connection, got {n_boundaries}"
    print("  ✓ Correct number of boundaries (2 for skip connection)")

    # Check shapes
    print(f"\n[Check] Output shapes:")
    for i, out in enumerate(result):
        if i < ode_func.neuron_count:
            print(f"  dv{i+1}/dt shape: {tuple(out.shape)}")
        else:
            print(f"  boundary_{i - ode_func.neuron_count + 1} shape: {tuple(out.shape)}")

    # Check post_neuron_module with tuple input
    print(f"\n[Check] post_neuron_module with tuple input:")
    post_module = ode_func.get_post_neuron_module()

    # Extract boundary outputs
    boundary_outputs = tuple(result[ode_func.neuron_count + i] for i in range(n_boundaries))
    print(f"  Input: tuple of {len(boundary_outputs)} tensors")
    for i, b in enumerate(boundary_outputs):
        print(f"    boundary_{i+1} shape: {tuple(b.shape)}")

    post_result = post_module(boundary_outputs)
    print(f"  Output shape: {tuple(post_result.shape)}")
    assert post_result.shape == (batch_size, 5), f"Expected (4, 5), got {post_result.shape}"
    print("  ✓ post_neuron_module works with tuple input")

    print("\n✓ TEST 2 PASSED: Multi-Boundary Case")
    return True


# ============================================================================
# Test Case 3: No Post-Processing (Pure ODE)
# ============================================================================
# Network: input → fc1 → lif1 → fc2 → lif2 (no layers after last neuron)
# Expected: post_neuron_module is nn.Identity

class PureODESNN(nn.Module):
    """SNN with no post-processing layers."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.lif1 = LIFNeuron()
        self.fc2 = nn.Linear(20, 5)
        self.lif2 = LIFNeuron()  # Last layer is neuron

    def forward(self, x):
        x = self.lif1(self.fc1(x))
        x = self.lif2(self.fc2(x))
        return x


def test_no_post_processing():
    print("\n" + "-" * 70)
    print("TEST 3: No Post-Processing (Pure ODE)")
    print("-" * 70)

    model = PureODESNN()
    ode_func = ODEFuncFromFX(model)

    print(f"\n[Check] neuron_count = {ode_func.neuron_count}")
    assert ode_func.neuron_count == 2, f"Expected 2 neurons, got {ode_func.neuron_count}"
    print("  ✓ Correct neuron count")

    # Check post_neuron_module is Identity
    print(f"\n[Check] post_neuron_module type:")
    post_module = ode_func.get_post_neuron_module()
    print(f"  Type: {type(post_module).__name__}")

    is_identity = isinstance(post_module, nn.Identity)
    print(f"  Is nn.Identity: {is_identity}")
    assert is_identity, "Expected nn.Identity for no post-processing"
    print("  ✓ post_neuron_module is nn.Identity")

    print("\n✓ TEST 3 PASSED: No Post-Processing Case")
    return True


# ============================================================================
# Test Case 4: Interpolation Methods
# ============================================================================

def test_interpolation():
    print("\n" + "-" * 70)
    print("TEST 4: Interpolation Methods")
    print("-" * 70)

    # Create test data
    T, B, F = 5, 3, 4  # time, batch, features
    x = torch.randn(T, B, F)
    x_time = torch.linspace(0, 1, T)

    methods = ['linear', 'nearest', 'cubic', 'akima']

    for method in methods:
        print(f"\n[Check] Method: {method}")

        # Test at various time points
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t_tensor = torch.tensor(t)
            result = interpolate(x, x_time, t_tensor, method=method)
            print(f"  t={t}: output shape {tuple(result.shape)}")
            assert result.shape == (B, F), f"Expected ({B}, {F}), got {result.shape}"

        print(f"  ✓ {method} interpolation works")

    print("\n✓ TEST 4 PASSED: All Interpolation Methods")
    return True


# ============================================================================
# Test Case 5: Different Neuron Types
# ============================================================================

class MixedNeuronSNN(nn.Module):
    """SNN with different neuron types."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.lif = LIFNeuron()
        self.fc2 = nn.Linear(20, 15)
        self.if_neuron = IFNeuron()
        self.fc3 = nn.Linear(15, 5)

    def forward(self, x):
        x = self.lif(self.fc1(x))
        x = self.if_neuron(self.fc2(x))
        x = self.fc3(x)
        return x


def test_mixed_neurons():
    print("\n" + "-" * 70)
    print("TEST 5: Mixed Neuron Types (LIF + IF)")
    print("-" * 70)

    model = MixedNeuronSNN()
    ode_func = ODEFuncFromFX(model)

    print(f"\n[Check] neuron_count = {ode_func.neuron_count}")
    assert ode_func.neuron_count == 2, f"Expected 2 neurons, got {ode_func.neuron_count}"
    print("  ✓ Correct neuron count (LIF + IF)")

    # Run forward pass
    batch_size = 4
    dummy_t = torch.tensor(0.5)
    dummy_v_mems = (
        torch.randn(batch_size, 20),
        torch.randn(batch_size, 15),
    )
    dummy_x = torch.randn(8, batch_size, 10)
    dummy_x_time = torch.linspace(0, 1, 8)

    ode_func.set_inputs(dummy_x, dummy_x_time)
    result = ode_func.ode_gm(dummy_t, dummy_v_mems, dummy_x, dummy_x_time)

    print(f"\n[Check] ode_gm output:")
    print(f"  Number of outputs: {len(result)}")
    print(f"  Expected: {ode_func.neuron_count} dv/dt + boundaries")

    # Check shapes
    for i, out in enumerate(result):
        print(f"  Output {i} shape: {tuple(out.shape)}")

    print("\n✓ TEST 5 PASSED: Mixed Neuron Types")
    return True


# ============================================================================
# Test Case 6: Forward Pass Equivalence
# ============================================================================

def test_forward_pass():
    print("\n" + "-" * 70)
    print("TEST 6: ODE Forward Pass")
    print("-" * 70)

    model = SequentialSNN()
    ode_func = ODEFuncFromFX(model)

    batch_size = 4
    dummy_x = torch.randn(8, batch_size, 10)
    dummy_x_time = torch.linspace(0, 1, 8)

    # Set inputs
    ode_func.set_inputs(dummy_x, dummy_x_time)

    # Test forward at multiple time points
    print("\n[Check] Forward pass at different time points:")
    for t in [0.0, 0.3, 0.7, 1.0]:
        dummy_t = torch.tensor(t)
        dummy_v_mems = (
            torch.zeros(batch_size, 20),
            torch.zeros(batch_size, 15),
        )

        # Use ode_func.forward() which is called by ODE solvers
        result = ode_func.forward(dummy_t, dummy_v_mems)

        print(f"  t={t}:")
        print(f"    Number of outputs: {len(result)}")
        print(f"    dv1/dt shape: {tuple(result[0].shape)}")
        print(f"    dv2/dt shape: {tuple(result[1].shape)}")
        if len(result) > 2:
            print(f"    boundary shape: {tuple(result[2].shape)}")

    # Check NFE counter
    print(f"\n[Check] NFE (Number of Function Evaluations): {ode_func.nfe}")
    assert ode_func.nfe == 4, f"Expected 4 NFE, got {ode_func.nfe}"
    print("  ✓ NFE counter works")

    print("\n✓ TEST 6 PASSED: Forward Pass")
    return True


# ============================================================================
# Test Case 7: Deeper Network
# ============================================================================

class DeepSNN(nn.Module):
    """Deeper SNN with 4 neuron layers."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.lif1 = LIFNeuron()
        self.fc2 = nn.Linear(32, 32)
        self.lif2 = LIFNeuron()
        self.fc3 = nn.Linear(32, 16)
        self.lif3 = LIFNeuron()
        self.fc4 = nn.Linear(16, 16)
        self.lif4 = LIFNeuron()
        self.fc_out = nn.Linear(16, 5)

    def forward(self, x):
        x = self.lif1(self.fc1(x))
        x = self.lif2(self.fc2(x))
        x = self.lif3(self.fc3(x))
        x = self.lif4(self.fc4(x))
        x = self.fc_out(x)
        return x


def test_deep_network():
    print("\n" + "-" * 70)
    print("TEST 7: Deep Network (4 Neuron Layers)")
    print("-" * 70)

    model = DeepSNN()
    ode_func = ODEFuncFromFX(model)

    print(f"\n[Check] neuron_count = {ode_func.neuron_count}")
    assert ode_func.neuron_count == 4, f"Expected 4 neurons, got {ode_func.neuron_count}"
    print("  ✓ Correct neuron count")

    # Run forward
    batch_size = 4
    dummy_t = torch.tensor(0.5)
    dummy_v_mems = (
        torch.randn(batch_size, 32),
        torch.randn(batch_size, 32),
        torch.randn(batch_size, 16),
        torch.randn(batch_size, 16),
    )
    dummy_x = torch.randn(8, batch_size, 10)
    dummy_x_time = torch.linspace(0, 1, 8)

    ode_func.set_inputs(dummy_x, dummy_x_time)
    result = ode_func.ode_gm(dummy_t, dummy_v_mems, dummy_x, dummy_x_time)

    print(f"\n[Check] ode_gm output:")
    print(f"  Total outputs: {len(result)}")
    print(f"  Expected: 4 dv/dt + 1 boundary = 5")
    assert len(result) == 5, f"Expected 5 outputs, got {len(result)}"

    for i in range(4):
        print(f"  dv{i+1}/dt shape: {tuple(result[i].shape)}")
    print(f"  boundary shape: {tuple(result[4].shape)}")

    print("\n✓ TEST 7 PASSED: Deep Network")
    return True


# ============================================================================
# Test Case 8: Temporal Equivalence of Multi-Boundary Cat Operation
# ============================================================================
# Verify that cat([x1_stacked, x2_stacked], dim=-1) followed by fc
# is equivalent to doing cat and fc at each time step

def test_temporal_cat_equivalence():
    print("\n" + "-" * 70)
    print("TEST 8: Temporal Equivalence of Multi-Boundary Cat Operation")
    print("-" * 70)

    T, B, F1, F2 = 8, 4, 20, 20  # time, batch, features
    out_features = 5

    # Create test data
    torch.manual_seed(42)
    x1_list = [torch.randn(B, F1) for _ in range(T)]  # Per-time tensors
    x2_list = [torch.randn(B, F2) for _ in range(T)]

    # Shared fc layer
    fc = nn.Linear(F1 + F2, out_features)

    # Method 1: Cat and fc at each time step, then stack
    print("\n[Method 1] Cat → fc at each time step, then stack:")
    results_per_time = []
    for t in range(T):
        x_cat_t = torch.cat([x1_list[t], x2_list[t]], dim=-1)  # (B, 40)
        out_t = fc(x_cat_t)  # (B, 5)
        results_per_time.append(out_t)
    output_method1 = torch.stack(results_per_time, dim=0)  # (T, B, 5)
    print(f"  Output shape: {tuple(output_method1.shape)}")

    # Method 2: Stack first, then cat, then fc (current implementation)
    print("\n[Method 2] Stack → cat → fc (current implementation):")
    x1_stacked = torch.stack(x1_list, dim=0)  # (T, B, 20)
    x2_stacked = torch.stack(x2_list, dim=0)  # (T, B, 20)
    x_cat_stacked = torch.cat([x1_stacked, x2_stacked], dim=-1)  # (T, B, 40)
    output_method2 = fc(x_cat_stacked)  # (T, B, 5)
    print(f"  Output shape: {tuple(output_method2.shape)}")

    # Check equivalence
    print("\n[Check] Equivalence:")
    is_equal = torch.allclose(output_method1, output_method2, atol=1e-6)
    max_diff = (output_method1 - output_method2).abs().max().item()
    print(f"  torch.allclose: {is_equal}")
    print(f"  Max difference: {max_diff:.2e}")

    assert is_equal, f"Methods not equivalent! Max diff: {max_diff}"
    print("  ✓ Both methods produce identical results")

    # Also verify with the actual post_neuron_module from BranchingSNN
    print("\n[Check] With actual BranchingSNN post_neuron_module:")
    model = BranchingSNN()
    ode_func = ODEFuncFromFX(model)
    post_module = ode_func.get_post_neuron_module()

    # Simulate accumulated boundaries over time
    torch.manual_seed(123)
    boundary_1_list = [torch.randn(B, 20) for _ in range(T)]
    boundary_2_list = [torch.randn(B, 20) for _ in range(T)]

    # Method 1: Apply post_module at each time
    results_per_time = []
    for t in range(T):
        out_t = post_module((boundary_1_list[t], boundary_2_list[t]))
        results_per_time.append(out_t)
    output_real_method1 = torch.stack(results_per_time, dim=0)

    # Method 2: Stack then apply (how SNNWrapper does it)
    boundary_1_stacked = torch.stack(boundary_1_list, dim=0)
    boundary_2_stacked = torch.stack(boundary_2_list, dim=0)
    output_real_method2 = post_module((boundary_1_stacked, boundary_2_stacked))

    is_equal_real = torch.allclose(output_real_method1, output_real_method2, atol=1e-6)
    max_diff_real = (output_real_method1 - output_real_method2).abs().max().item()
    print(f"  torch.allclose: {is_equal_real}")
    print(f"  Max difference: {max_diff_real:.2e}")

    assert is_equal_real, f"Real post_module not equivalent! Max diff: {max_diff_real}"
    print("  ✓ post_neuron_module is temporally equivalent")

    print("\n✓ TEST 8 PASSED: Temporal Equivalence Verified")
    return True


# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == "__main__":
    tests = [
        ("Single Boundary", test_single_boundary),
        ("Multi-Boundary", test_multi_boundary),
        ("No Post-Processing", test_no_post_processing),
        ("Interpolation", test_interpolation),
        ("Mixed Neurons", test_mixed_neurons),
        ("Forward Pass", test_forward_pass),
        ("Deep Network", test_deep_network),
        ("Temporal Cat Equivalence", test_temporal_cat_equivalence),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n✗ TEST FAILED: {name}")
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, passed_flag, error in results:
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")