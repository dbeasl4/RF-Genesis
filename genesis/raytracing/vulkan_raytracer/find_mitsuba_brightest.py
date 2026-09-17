"""
Diagnostic: finds Mitsuba's brightest pixel (coordinates + distance +
intensity), to compare directly against the Vulkan version's brightest
pixel and check whether the two renderers are actually reporting the
same physical point on the body, or two different points entirely.
"""
import sys
import os
import argparse
import numpy as np

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

parser = argparse.ArgumentParser()
parser.add_argument("--motion", default=None)
parser.add_argument("--frame", type=int, default=0)
args = parser.parse_args()
motion_path = os.path.abspath(args.motion) if args.motion else None

os.chdir(os.path.join(_repo_root, "genesis"))

from genesis.raytracing.pathtracer import RayTracer

tracer = RayTracer()

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

distance = PIR[:, :, 0]
intensity = PIR[:, :, 1]

y, x = np.unravel_index(np.argmax(intensity), intensity.shape)
print(f"Mitsuba brightest pixel: ({y}, {x})  distance: {distance[y,x]:.6f}  intensity: {intensity[y,x]:.6f}")
