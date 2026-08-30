"""
Verifies the Vulkan-CUDA external memory interop actually works: Vulkan
writes known values (0..15) into a GPU buffer, exports it, CUDA imports it,
and this checks the returned torch tensor matches exactly.
"""
import torch
import vulkan_cuda_interop

result = vulkan_cuda_interop.test_interop()

print(f"Returned tensor: {result}")
print(f"Device: {result.device}")
print(f"Dtype: {result.dtype}")

expected = torch.arange(16, dtype=torch.float32, device=result.device)
matches = torch.equal(result, expected)

print(f"\nExpected: {expected}")
print(f"Match: {matches}")

if matches:
    print("\nSUCCESS -- Vulkan wrote this data, CUDA read it via external memory import, no CPU copy involved.")
else:
    print("\nMISMATCH -- something is wrong with the memory interop. Do not proceed to part 2 until this passes.")
