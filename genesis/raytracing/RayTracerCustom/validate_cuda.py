"""
Validates the custom CUDA ray tracer against Mitsuba, same methodology as
VulkanCudaInterop/validate_live.py. Exists to answer one specific
question: does the CUDA path's shadow ray (a much simpler brute-force
implementation, independent of Vulkan's pipeline/SBT machinery) show the
same ~50-60% intensity-mean gap Vulkan shows? If yes, the bug is shared
conceptually (not a Vulkan-specific pipeline bug). If no, it isolates the
bug to something specific to Vulkan's shadow ray implementation.

Run from genesis/raytracing/RayTracerCustom/:
    python validate_cuda.py --motion ../../../output/hello_rfgen/obj_diff.npz --frame 50
"""
import sys
import os
import argparse
import numpy as np
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

parser = argparse.ArgumentParser()
parser.add_argument("--motion", default=None)
parser.add_argument("--frame", type=int, default=0)
args = parser.parse_args()
motion_path = os.path.abspath(args.motion) if args.motion else None

os.chdir(os.path.join(_repo_root, "genesis"))

from genesis.raytracing.pathtracer import RayTracer as MitsubaRayTracer
from genesis.raytracing.RayTracerCustom.raytracer import RayTracer as CustomRayTracer

smpl_data = np.load(motion_path, allow_pickle=True)
pose = smpl_data['pose'][args.frame]
shape = smpl_data['shape'][0]
root_translation = np.array(smpl_data['root_translation'][args.frame])
body_offset = np.array([0, 1, 3])
translation = (root_translation - body_offset).astype(np.float32)
print(f"Using frame {args.frame} from {motion_path}")

# CUDA (same camera/light defaults as Mitsuba's get_deafult_scene() --
# no update_sensor() call, matching validate_live.py's setup, unlike the
# mismatched camera in RayTracerCustom/benchmark.py)
cuda_tracer = CustomRayTracer()
cuda_tracer.update_pose(pose, shape, translation)
cuda_PIR, _ = cuda_tracer.trace()
cuda_distance = cuda_PIR[:, :, 0]
cuda_intensity = cuda_PIR[:, :, 1]
cuda_hits = int((cuda_distance > 0).sum())

# Mitsuba
mitsuba_tracer = MitsubaRayTracer()
mitsuba_tracer.update_pose(pose, shape, translation)
mitsuba_PIR, _ = mitsuba_tracer.trace()
mitsuba_distance = mitsuba_PIR[:, :, 0]
mitsuba_intensity = mitsuba_PIR[:, :, 1]
mitsuba_hits = int((mitsuba_distance > 0).sum())

print(f"\n{'':20s} {'Mitsuba':>15s} {'CUDA':>15s}")
print(f"{'Hits':20s} {mitsuba_hits:>15d} {cuda_hits:>15d}")
if mitsuba_hits > 0 and cuda_hits > 0:
    print(f"{'Distance min':20s} {mitsuba_distance[mitsuba_distance>0].min():>15.5f} "
          f"{cuda_distance[cuda_distance>0].min():>15.5f}")
    print(f"{'Distance max':20s} {mitsuba_distance[mitsuba_distance>0].max():>15.5f} "
          f"{cuda_distance[cuda_distance>0].max():>15.5f}")
print(f"{'Intensity max':20s} {mitsuba_intensity.max():>15.5f} {cuda_intensity.max():>15.5f}")

mitsuba_mean = mitsuba_intensity[mitsuba_intensity > 0].mean() if mitsuba_hits > 0 else float('nan')
cuda_mean = cuda_intensity[cuda_distance > 0].mean() if cuda_hits > 0 else float('nan')
print(f"{'Intensity mean':20s} {mitsuba_mean:>15.5f} {cuda_mean:>15.5f}")

mitsuba_median = np.median(mitsuba_intensity[mitsuba_intensity > 0]) if mitsuba_hits > 0 else float('nan')
cuda_median = np.median(cuda_intensity[cuda_distance > 0]) if cuda_hits > 0 else float('nan')
print(f"{'Intensity median':20s} {mitsuba_median:>15.5f} {cuda_median:>15.5f}")

if cuda_mean and mitsuba_mean:
    print(f"\nCUDA/Mitsuba mean ratio: {cuda_mean / mitsuba_mean:.3f}x "
          f"({'MATCHES Vulkan gap -- shared bug' if abs(cuda_mean/mitsuba_mean - 1.5) < 0.3 else 'DIFFERS from Vulkan gap -- Vulkan-specific bug'})")
