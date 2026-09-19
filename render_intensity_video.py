"""
Renders an actual video from the ray tracer's real computed light output
(the intensity channel: real shading, spot-light falloff, everything the
shader computes) -- not the simple stick-figure debug plot run.py's own
visualize.py produces.

Uses the live Vulkan path directly, the same class pathtracer.py uses
internally for Step 2 -- so this is a real, faithful look at exactly what
the ray tracer computes each frame, stitched into a video.

Run from RF-Genesis/:
    python render_intensity_video.py output/real_mocap_test2/obj_diff.npz -o intensity_video.mp4
"""
import sys
import os
import argparse
import subprocess
import shutil
import numpy as np
from PIL import Image
import torch

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)

parser = argparse.ArgumentParser()
parser.add_argument("motion_file", help="Path to an obj_diff.npz file (from either MDM or amass_to_rfgen.py)")
parser.add_argument("-o", "--output", default="intensity_video.mp4")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--resolution", type=int, default=256, help="Render resolution (square)")
args = parser.parse_args()

if shutil.which("ffmpeg") is None:
    print("ffmpeg not found on this system -- required to stitch frames into a video.")
    print("Install it with: sudo apt install ffmpeg")
    sys.exit(1)

# Resolve these BEFORE the chdir below -- os.path.abspath() resolves
# relative paths against the CURRENT working directory at the moment
# it's called, so doing this after chdir would resolve them against
# genesis/ instead of wherever the script was actually run from.
motion_path = os.path.abspath(args.motion_file)
output_path = os.path.abspath(args.output)

os.chdir(os.path.join(_repo_root, "genesis"))
from genesis.raytracing import smpl as smpl_utils

sys.path.insert(0, os.path.join(_repo_root, "genesis", "raytracing", "vulkan_raytracer", "VulkanCudaInterop"))
from live_raytracer import RayTracer

data = np.load(motion_path, allow_pickle=True)
n_frames = len(data['root_translation'])
body_offset = np.array([0, 1, 3])

print(f"Rendering {n_frames} frames...")

body = smpl_utils.get_smpl_layer()
tracer = RayTracer(body, resolution=args.resolution, fov=60.0)


def render_frame(i):
    pose = data['pose'][i].astype(np.float32)
    shape = data['shape'][0].astype(np.float32)
    translation = (np.array(data['root_translation'][i]) - body_offset).astype(np.float32)
    tracer.update_pose(pose, shape, translation)
    PIR, _ = tracer.trace()
    intensity = PIR[:, :, 1]
    hitmask = PIR[:, :, 0] > 0
    return intensity, hitmask


# Pass 1: find the shared brightness scale across the whole sequence.
# Only keeps a single running scalar, not each frame's full image data --
# holding every frame in memory at once is what caused the OOM kill on
# long sequences (several GB at 720x720 x thousands of frames).
print("Pass 1/2: finding the shared brightness scale...")
global_max = 1.0
for i in range(n_frames):
    intensity, hitmask = render_frame(i)
    if hitmask.any():
        frame_max = float(intensity[hitmask].max())
        global_max = max(global_max, frame_max)
    if (i + 1) % 200 == 0 or i == n_frames - 1:
        print(f"  ...{i+1}/{n_frames}")
print(f"Global intensity max across sequence: {global_max:.4f}")

# Pass 2: render again (yes, this re-does the trace -- cheap relative to
# avoiding several GB of accumulated frame data), saving each frame
# straight to disk immediately instead of accumulating in memory.
print("Pass 2/2: rendering and saving frames...")
frame_dir = os.path.join(_repo_root, "_intensity_video_frames_tmp")
os.makedirs(frame_dir, exist_ok=True)
for i in range(n_frames):
    intensity, hitmask = render_frame(i)
    a = np.where(hitmask, intensity, 0.0)
    a = np.clip(a / global_max, 0.0, 1.0)
    Image.fromarray((a * 255).astype(np.uint8)).save(os.path.join(frame_dir, f"frame_{i:04d}.png"))
    if (i + 1) % 200 == 0 or i == n_frames - 1:
        print(f"  ...{i+1}/{n_frames}")

subprocess.run([
    "ffmpeg", "-y", "-framerate", str(args.fps),
    "-i", os.path.join(frame_dir, "frame_%04d.png"),
    "-pix_fmt", "yuv420p", output_path
], check=True)

shutil.rmtree(frame_dir)
print(f"\nSaved {output_path}")
