"""
Renders the same SMPL body pose (as exported by export_mesh.py) through
Mitsuba, using the same camera position/target/fov as the Vulkan program,
so the two outputs can be compared directly. Run from vulkan_raytracer/.
Pass the same arguments used with export_mesh.py so both programs render
the identical pose. See GETTING_STARTED.md for details.

Known expected difference: Mitsuba's light ('tx' in pathtracer.py's
get_deafult_scene()) is a spot light with a 40-degree cutoff angle. The
Vulkan shader currently models a simple omnidirectional point light with
no directional falloff, so shading near the edges of Mitsuba's spot cone
may differ somewhat. This is a known model difference, not a bug.
"""
import sys
import os
import argparse
import numpy as np
from PIL import Image
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

parser = argparse.ArgumentParser()
parser.add_argument("--motion", default=None, help="Path to obj_diff.npz to pull a real pose from")
parser.add_argument("--frame", type=int, default=0)
parser.add_argument("--resolution", type=int, default=128, help="Render resolution, must match the Vulkan program's WIDTH/HEIGHT for a fair comparison")
args = parser.parse_args()
motion_path = os.path.abspath(args.motion) if args.motion else None

# Resolve output paths to absolute before the chdir below, otherwise these
# bare filenames would resolve relative to genesis/ (the new cwd) instead of
# wherever this script was actually run from. Same lesson already learned
# with benchmark.py's motion-file path.
distance_out = os.path.abspath("mitsuba_distance.png")
intensity_out = os.path.abspath("mitsuba_intensity.png")

# Same relative-path requirement as run.py / benchmark.py / export_mesh.py:
# Mitsuba's scene definition needs cwd = RF-Genesis/genesis/
os.chdir(os.path.join(_repo_root, "genesis"))

from genesis.raytracing.pathtracer import RayTracer, get_deafult_scene
import mitsuba as mi

tracer = RayTracer()  # PIR_resolution=128 by default, matching the Vulkan program's WIDTH/HEIGHT

if args.resolution != tracer.PIR_resolution:
    # RayTracer's __init__ hardcodes resolution to 128 and builds the
    # scene from it immediately -- rebuild the same way, at the
    # requested resolution instead, rather than editing pathtracer.py.
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
    print(f"Using frame {args.frame} from {motion_path}")
else:
    pose = np.zeros(72, dtype=np.float32)
    shape = np.zeros(10, dtype=np.float32)
    # Explicit zero vector rather than translation=None: update_pose() has
    # an existing bug in this repo where passing None leaves a local
    # variable unbound in that code path (unrelated to anything we changed).
    translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    print("Using rest pose (no --motion given)")

tracer.update_pose(pose, shape, translation)
PIR, pointclouds = tracer.trace()

distance = PIR[:, :, 0]   # Mitsuba uses 0 as its miss sentinel (see pathtracer.py: t[t>9999]=0)
intensity = PIR[:, :, 1]


def save_grayscale(arr, out_path, is_distance=False):
    a = arr.copy()
    hit_mask = a > 0 if is_distance else a >= 0

    if is_distance and hit_mask.any():
        hit_min = a[hit_mask].min()
        hit_max = a[hit_mask].max()
        span = max(hit_max - hit_min, 1e-6)
        a = np.where(hit_mask, (a - hit_min) / span, 0.0)
    else:
        a[~hit_mask] = 0
        max_val = a.max() if a.max() > 0 else 1.0
        a = a / max_val

    img = (a * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


save_grayscale(distance, distance_out, is_distance=True)
save_grayscale(intensity, intensity_out, is_distance=False)

hits = int((distance > 0).sum())
print(f"Mitsuba render: {distance.shape[1]}x{distance.shape[0]}")
print(f"Hits: {hits} / {distance.size}")
if hits > 0:
    print(f"Distance range:  [{distance[distance > 0].min():.5f}, {distance[distance > 0].max():.5f}]")
    print(f"Intensity range: [{intensity.min():.5f}, {intensity.max():.5f}]")
print(f"Saved {distance_out} and {intensity_out}")
