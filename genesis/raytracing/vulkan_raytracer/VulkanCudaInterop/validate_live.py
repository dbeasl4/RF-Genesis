"""
Validates the live Vulkan ray tracer (persistent, PyTorch-driven) against
Mitsuba, exercising the real, integrated Python-callable class. See
GETTING_STARTED.md for how to run this.
"""
import sys
import os
import argparse
import numpy as np
from PIL import Image
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

parser = argparse.ArgumentParser()
parser.add_argument("--motion", default=None)
parser.add_argument("--frame", type=int, default=0)
args = parser.parse_args()
motion_path = os.path.abspath(args.motion) if args.motion else None

distance_out = os.path.abspath("live_distance.png")
intensity_out = os.path.abspath("live_intensity.png")

os.chdir(os.path.join(_repo_root, "genesis"))

from genesis.raytracing import smpl as smpl_utils
from genesis.raytracing.pathtracer import RayTracer as MitsubaRayTracer

sys.path.insert(0, os.path.join(_repo_root, "genesis", "raytracing", "VulkanCudaInterop"))
from live_raytracer import RayTracer as LiveVulkanRayTracer

body = smpl_utils.get_smpl_layer()

if motion_path:
    data = np.load(motion_path, allow_pickle=True)
    pose = data['pose'][args.frame].astype(np.float32)
    shape = data['shape'][0].astype(np.float32)
    root_translation = np.array(data['root_translation'][args.frame])
    body_offset = np.array([0, 1, 3])
    translation = (root_translation - body_offset).astype(np.float32)
    print(f"Using frame {args.frame} from {motion_path}")
else:
    pose = np.zeros(72, dtype=np.float32)
    shape = np.zeros(10, dtype=np.float32)
    translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    print("Using rest pose (no --motion given)")

# Live Vulkan. No update_sensor() call here: the constructor's default
# camera (origin (0,1,3), target (0,1,0)) already matches Mitsuba's scene
# camera correctly. Calling update_sensor() with the radar sensor's
# position (a different, unrelated value used elsewhere in the pipeline
# for signal generation) instead of the rendering camera would overwrite
# this correct default and produce badly wrong results.
live_tracer = LiveVulkanRayTracer(body, resolution=128, fov=60.0)
live_tracer.update_pose(pose, shape, translation)
live_PIR, _ = live_tracer.trace()
live_distance = live_PIR[:, :, 0]
live_intensity = live_PIR[:, :, 1]
live_hits = int((live_distance > 0).sum())

# Mitsuba
mitsuba_tracer = MitsubaRayTracer()
mitsuba_tracer.update_pose(pose, shape, translation)
mitsuba_PIR, _ = mitsuba_tracer.trace()
mitsuba_distance = mitsuba_PIR[:, :, 0]
mitsuba_intensity = mitsuba_PIR[:, :, 1]
mitsuba_hits = int((mitsuba_distance > 0).sum())

print(f"\n{'':20s} {'Mitsuba':>15s} {'Live Vulkan':>15s}")
print(f"{'Hits':20s} {mitsuba_hits:>15d} {live_hits:>15d}")
if mitsuba_hits > 0:
    print(f"{'Distance min':20s} {mitsuba_distance[mitsuba_distance>0].min():>15.5f} "
          f"{live_distance[live_distance>0].min() if live_hits > 0 else float('nan'):>15.5f}")
    print(f"{'Distance max':20s} {mitsuba_distance[mitsuba_distance>0].max():>15.5f} "
          f"{live_distance[live_distance>0].max() if live_hits > 0 else float('nan'):>15.5f}")
print(f"{'Intensity max':20s} {mitsuba_intensity.max():>15.5f} {live_intensity.max():>15.5f}")


def save_grayscale(arr, out_path, is_distance=False, hit_test=None):
    a = arr.copy()
    hit_mask = hit_test(a) if hit_test else (a > 0)
    if is_distance and hit_mask.any():
        hit_min, hit_max = a[hit_mask].min(), a[hit_mask].max()
        span = max(hit_max - hit_min, 1e-6)
        a = np.where(hit_mask, (a - hit_min) / span, 0.0)
    else:
        a = np.where(hit_mask, a, 0.0)
        max_val = a.max() if a.max() > 0 else 1.0
        a = a / max_val
    Image.fromarray((a * 255).astype(np.uint8)).save(out_path)


save_grayscale(live_distance, distance_out, is_distance=True)
save_grayscale(live_intensity, intensity_out, is_distance=False)
print(f"\nSaved {distance_out} and {intensity_out}")
