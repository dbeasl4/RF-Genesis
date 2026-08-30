"""
Benchmarks the full live Vulkan pipeline (pose -> SMPL -> vertex upload ->
BLAS/TLAS rebuild -> trace -> CUDA readback) across real motion frames,
using the same methodology as the Mitsuba/CUDA benchmark for a fair,
apples-to-apples comparison.
"""
import sys
import os
import time
import argparse
import numpy as np
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

parser = argparse.ArgumentParser()
parser.add_argument("motion_file", nargs="?",
                     default="../../../../output/hello_rfgen/obj_diff.npz")
parser.add_argument("--max_frames", type=int, default=None,
                     help="Limit to the first N frames (default: all)")
args = parser.parse_args()
motion_path = os.path.abspath(args.motion_file)

os.chdir(os.path.join(_repo_root, "genesis"))

from genesis.raytracing import smpl as smpl_utils
from genesis.raytracing.pathtracer import RayTracer as MitsubaRayTracer

sys.path.insert(0, os.path.join(_repo_root, "genesis", "raytracing", "VulkanCudaInterop"))
from live_raytracer import RayTracer as LiveVulkanRayTracer

data = np.load(motion_path, allow_pickle=True)
root_translation = data['root_translation']
n_frames = len(root_translation)
if args.max_frames:
    n_frames = min(n_frames, args.max_frames)
body_offset = np.array([0, 1, 3])

print(f"Benchmarking {n_frames} frames from {motion_path}\n")

# Mitsuba
mitsuba_tracer = MitsubaRayTracer()
t0 = time.perf_counter()
for i in range(n_frames):
    pose = data['pose'][i]
    shape = data['shape'][0]
    translation = (np.array(root_translation[i]) - body_offset).astype(np.float32)
    mitsuba_tracer.update_pose(pose, shape, translation)
    mitsuba_tracer.trace()
torch.cuda.synchronize()
mitsuba_time = time.perf_counter() - t0

# Live Vulkan (full pipeline: pose -> SMPL -> upload -> BLAS/TLAS -> trace -> readback)
body = smpl_utils.get_smpl_layer()
live_tracer = LiveVulkanRayTracer(body, resolution=128, fov=60.0)
t0 = time.perf_counter()
for i in range(n_frames):
    pose = data['pose'][i]
    shape = data['shape'][0]
    translation = (np.array(root_translation[i]) - body_offset).astype(np.float32)
    live_tracer.update_pose(pose, shape, translation)
    live_tracer.trace()
torch.cuda.synchronize()
live_vulkan_time = time.perf_counter() - t0

print(f"Mitsuba:            {mitsuba_time:.3f}s total, {mitsuba_time / n_frames * 1000:.2f} ms/frame")
print(f"Live Vulkan (full): {live_vulkan_time:.3f}s total, {live_vulkan_time / n_frames * 1000:.2f} ms/frame")
print(f"\nSpeedup vs Mitsuba: {mitsuba_time / live_vulkan_time:.2f}x")
print("\nFor reference, from the earlier CUDA kernel benchmark (same machine):")
print("  Custom CUDA (brute-force):  15.16 ms/frame  (1.27x vs Mitsuba)")
print("\nThis number includes the full per-frame cost: SMPL forward pass,")
print("vertex upload, BLAS/TLAS rebuild, trace dispatch, and CUDA readback.")