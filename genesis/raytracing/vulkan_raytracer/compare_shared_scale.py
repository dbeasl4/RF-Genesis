"""
Loads both vulkan_output.bin and a fresh Mitsuba render of the same
frame, then saves BOTH intensity images normalized against the SAME
reference value (Mitsuba's max), instead of each being independently
normalized to its own max the way view_output.py / compare_mitsuba.py
do. This makes brightness in the two saved PNGs directly comparable --
whichever one looks brighter here genuinely IS brighter, at the same
absolute scale.

Run from vulkan_raytracer/, after generating build/vulkan_output.bin at
the resolution/frame you want to compare (export_mesh.py + the compiled
executable), matching --motion/--frame/--resolution to that same run.
"""
import sys
import os
import argparse
import numpy as np
from PIL import Image

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

parser = argparse.ArgumentParser()
parser.add_argument("--motion", default=None)
parser.add_argument("--frame", type=int, default=0)
parser.add_argument("--resolution", type=int, default=128)
args = parser.parse_args()
motion_path = os.path.abspath(args.motion) if args.motion else None

with open("build/vulkan_output.bin", "rb") as f:
    w = int.from_bytes(f.read(4), "little")
    h = int.from_bytes(f.read(4), "little")
    vdata = np.frombuffer(f.read(), dtype=np.float32).reshape(h, w, 2)
vulkan_intensity = vdata[:, :, 1]
vulkan_hit = vdata[:, :, 0] > 0

os.chdir(os.path.join(_repo_root, "genesis"))
from genesis.raytracing.pathtracer import RayTracer, get_deafult_scene
import mitsuba as mi

tracer = RayTracer()
if args.resolution != tracer.PIR_resolution:
    tracer.PIR_resolution = args.resolution
    tracer.scene = mi.load_dict(get_deafult_scene(res=tracer.PIR_resolution))
    tracer.params_scene = mi.traverse(tracer.scene)

if motion_path:
    data = np.load(motion_path, allow_pickle=True)
    pose = data['pose'][args.frame].astype(np.float32)
    shape = data['shape'][0].astype(np.float32)
    root_translation = np.array(data['root_translation'][args.frame])
    body_offset = np.array([0, 1, 3])
    translation = (root_translation - body_offset).astype(np.float32)
else:
    pose = np.zeros(72, dtype=np.float32)
    shape = np.zeros(10, dtype=np.float32)
    translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)

tracer.update_pose(pose, shape, translation)
PIR, _ = tracer.trace()
mitsuba_intensity = PIR[:, :, 1]
mitsuba_hit = PIR[:, :, 0] > 0

shared_max = max(vulkan_intensity[vulkan_hit].max() if vulkan_hit.any() else 0,
                  mitsuba_intensity[mitsuba_hit].max() if mitsuba_hit.any() else 0)
print(f"Vulkan max:  {vulkan_intensity[vulkan_hit].max():.5f}")
print(f"Mitsuba max: {mitsuba_intensity[mitsuba_hit].max():.5f}")
print(f"Shared scale (both images normalized against this): {shared_max:.5f}")


def save_shared_scale(intensity, hit_mask, out_path):
    a = np.where(hit_mask, intensity, 0.0)
    a = np.clip(a / shared_max, 0.0, 1.0)
    Image.fromarray((a * 255).astype(np.uint8)).save(out_path)


out_dir = os.path.join(_repo_root, "genesis", "raytracing", "vulkan_raytracer")
save_shared_scale(vulkan_intensity, vulkan_hit, os.path.join(out_dir, "vulkan_intensity_shared_scale.png"))
save_shared_scale(mitsuba_intensity, mitsuba_hit, os.path.join(out_dir, "mitsuba_intensity_shared_scale.png"))
print("Saved vulkan_intensity_shared_scale.png and mitsuba_intensity_shared_scale.png")
