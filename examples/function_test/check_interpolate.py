import torch
from spikeDE.odefunc import interpolate

torch.manual_seed(42)

print("=" * 60)
print("Test 1: Shared timestamps, scalar query time")
print("=" * 60)

T, B, C = 10, 4, 3
x = torch.randn(T, B, C)
x_time = torch.linspace(0, 1, T)
t = 0.35

for method in ['linear', 'nearest', 'cubic', 'akima']:
    result = interpolate(x, x_time, t, method=method)
    print(f"{method:8s}: shape={result.shape}, mean={result.mean():.4f}")

print("\n" + "=" * 60)
print("Test 2: Shared timestamps, batch query times")
print("=" * 60)

t_batch = torch.tensor([0.1, 0.35, 0.7, 0.95])  # [B]
for method in ['linear', 'nearest', 'cubic', 'akima']:
    result = interpolate(x, x_time, t_batch, method=method)
    print(f"{method:8s}: shape={result.shape}")

print("\n" + "=" * 60)
print("Test 3: Batch-specific timestamps")
print("=" * 60)

# Each batch has different time ranges
x_time_batch = torch.stack([
    torch.linspace(0, 1, T),
    torch.linspace(0, 2, T),
    torch.linspace(0.5, 1.5, T),
    torch.linspace(-1, 1, T),
])  # [B, T]

t_batch = torch.tensor([0.5, 1.0, 1.0, 0.0])  # [B]

for method in ['linear', 'nearest', 'cubic', 'akima']:
    result = interpolate(x, x_time_batch, t_batch, method=method)
    print(f"{method:8s}: shape={result.shape}")

print("\n" + "=" * 60)
print("Test 4: Gradient check")
print("=" * 60)

x = torch.randn(T, B, C, requires_grad=True)
x_time = torch.linspace(0, 1, T)
t = torch.tensor([0.25, 0.5, 0.75, 0.9], requires_grad=True)

result = interpolate(x, x_time, t, method='linear')
loss = result.sum()
loss.backward()

print(f"x.grad exists: {x.grad is not None}")
print(f"t.grad exists: {t.grad is not None}")
print(f"x.grad shape: {x.grad.shape}")
print(f"t.grad shape: {t.grad.shape}")

print("\n" + "=" * 60)
print("Test 5: Boundary handling")
print("=" * 60)

t_boundary = torch.tensor([-0.5, 0.0, 1.0, 1.5])  # out of [0,1] range
result = interpolate(x, x_time, t_boundary, method='linear')
print(f"Query times: {t_boundary.tolist()}")
print(f"Result shape: {result.shape}")

print("\n" + "=" * 60)
print("Test 6: Higher dimensional data [T, B, H, W]")
print("=" * 60)

T, B, H, W = 8, 3, 4, 4
x = torch.randn(T, B, H, W)
x_time = torch.linspace(0, 1, T)
t = torch.tensor([0.2, 0.5, 0.8])

result = interpolate(x, x_time, t, method='cubic')
print(f"Input shape: {x.shape}")
print(f"Output shape: {result.shape}")

print("\nAll tests passed!")