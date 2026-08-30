"""
Runs only the live Vulkan ray tracer, without loading Mitsuba, for a
lightweight standalone test that doesn't need the full comparison in
validate_live.py. See GETTING_STARTED.md for how to run this.

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
raw_out = os.path.abspath("live_output.npz")

os.chdir(os.path.join(_repo_root, "genesis"))

from genesis.raytracing import smpl as smpl_utils

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

live_tracer = LiveVulkanRayTracer(body, resolution=128, fov=60.0)
# NOTE: no update_sensor() call -- the constructor's default camera
# (origin (0,1,3), target (0,1,0)) already matches Mitsuba's scene camera.
live_tracer.update_pose(pose, shape, translation)
PIR, pointclouds = live_tracer.trace()

distance = PIR[:, :, 0]
intensity = PIR[:, :, 1]
hits = int((distance >= 0).sum())

print(f"\nLive Vulkan render: {distance.shape[1]}x{distance.shape[0]}")
print(f"Hits: {hits} / {distance.size}")
if hits > 0:
    print(f"Distance range:  [{distance[distance >= 0].min():.5f}, {distance[distance >= 0].max():.5f}]")
    print(f"Intensity range: [{intensity.min():.5f}, {intensity.max():.5f}]")

np.savez(raw_out, PIR=PIR, pointclouds=pointclouds)


def save_grayscale(arr, out_path, is_distance=False):
    a = arr.copy()
    hit_mask = a >= 0
    if is_distance and hit_mask.any():
        hit_min, hit_max = a[hit_mask].min(), a[hit_mask].max()
        span = max(hit_max - hit_min, 1e-6)
        a = np.where(hit_mask, (a - hit_min) / span, 0.0)
    else:
        a = np.where(hit_mask, a, 0.0)
        max_val = a.max() if a.max() > 0 else 1.0
        a = a / max_val
    Image.fromarray((a * 255).astype(np.uint8)).save(out_path)


save_grayscale(distance, distance_out, is_distance=True)
save_grayscale(intensity, intensity_out, is_distance=False)
print(f"\nSaved {distance_out}, {intensity_out}, and {raw_out}")
print("\nNow run compare_mitsuba.py separately (different process) with the")
print("SAME --motion/--frame arguments, and compare the printed stats.")