"""
Renders TWO bodies in one scene, side by side, using two independent
motion clips. Combines both bodies' vertex/normal/face data into ONE
mesh buffer per frame (with the second body's face indices offset past
the first body's vertex count) before handing it to the standalone
executable -- so from the renderer's point of view, it's just one
(bigger) mesh, no shader or pipeline changes needed at all.

Real synchronized 2-person interaction data isn't straightforwardly
available through AMASS (it only contains single-person fits, even for
originally-multi-person CMU capture sessions) -- this uses two
independent clips instead, offset apart in space so they don't overlap.
A real interaction dataset (CHI3D, Hi4D, etc.) would be a separate,
future data-source swap; this proves the actual two-body RENDERING
capability works, independent of where the motion data comes from.

Run from RF-Genesis/:
    python render_two_bodies.py obj_diff_real.npz obj_diff_real.npz -o two_bodies.mp4
    (same file twice is fine for a first test -- they'll walk in sync,
    just spatially offset apart)
"""
import sys
import os
import struct
import argparse
import subprocess
import shutil
import numpy as np
from PIL import Image
import torch

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)

parser = argparse.ArgumentParser()
parser.add_argument("motion_file_a", help="obj_diff.npz for the first body")
parser.add_argument("motion_file_b", help="obj_diff.npz for the second body")
parser.add_argument("-o", "--output", default="two_bodies.mp4")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--resolution", type=int, default=512, help="Render resolution (square)")
parser.add_argument("--separation", type=float, default=1.0,
                     help="Horizontal (X) offset applied to each body, in meters, so they don't overlap")
args = parser.parse_args()

if shutil.which("ffmpeg") is None:
    print("ffmpeg not found. Install it with: sudo apt install ffmpeg")
    sys.exit(1)

motion_path_a = os.path.abspath(args.motion_file_a)
motion_path_b = os.path.abspath(args.motion_file_b)
output_path = os.path.abspath(args.output)

_vulkan_dir = os.path.join(_repo_root, "genesis", "raytracing", "vulkan_raytracer")
_exe_path = os.path.join(_vulkan_dir, "build", "vulkan_raytracer")
_mesh_path = os.path.join(_vulkan_dir, "smpl_mesh.bin")
_output_bin_path = os.path.join(_vulkan_dir, "build", "vulkan_output.bin")

os.chdir(os.path.join(_repo_root, "genesis"))
from genesis.raytracing import smpl as smpl_utils


def compute_vertex_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vertex_normals = np.zeros_like(vertices)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (vertex_normals / norms).astype(np.float32)


body = smpl_utils.get_smpl_layer()
faces_np = body.th_faces.cpu().numpy().astype(np.uint32)
n_verts = int(faces_np.max()) + 1  # 6890 for standard SMPL, computed rather than hardcoded

data_a = np.load(motion_path_a, allow_pickle=True)
data_b = np.load(motion_path_b, allow_pickle=True)
n_frames = min(len(data_a['root_translation']), len(data_b['root_translation']))
body_offset = np.array([0, 1, 3])

print(f"Rendering {n_frames} frames, two bodies, {args.separation}m apart...")


def smpl_forward(data, i, x_offset):
    pose = data['pose'][i % len(data['pose'])].astype(np.float32)
    shape = data['shape'][0].astype(np.float32)
    translation = (np.array(data['root_translation'][i % len(data['root_translation'])]) - body_offset).astype(np.float32)
    translation[0] += x_offset  # spread the two bodies apart along X

    pose_t = torch.tensor(pose).view(1, -1).float()
    shape_t = torch.tensor(shape).view(1, -1).float()
    vertices, _ = body(pose_t, th_betas=shape_t)
    vertices = (vertices[0] + torch.as_tensor(translation)).detach().cpu().numpy().astype(np.float32)
    return vertices


def make_floor(floor_y, half_size=2.5):
    """
    A simple flat quad, normals pointing straight up -- gives the spot
    light an actual surface to illuminate besides the bodies, so the
    scene reads as a real lit space rather than figures floating in a
    void. Positioned empirically (floor_y is computed from the actual
    lowest vertex across a sample of real frames, not a hardcoded guess),
    so it's correctly placed under the feet regardless of body offset
    conventions or coordinate corrections applied to the source motion.
    """
    verts = np.array([
        [-half_size, floor_y, -half_size],
        [ half_size, floor_y, -half_size],
        [ half_size, floor_y,  half_size],
        [-half_size, floor_y,  half_size],
    ], dtype=np.float32)
    normals = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (4, 1))
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return verts, normals, faces


print("Finding floor height and scene centering (checking every frame -- cheap, pure SMPL math, no rendering)...")
floor_y = None
x_min = x_max = z_min = z_max = None
for i in range(n_frames):
    va = smpl_forward(data_a, i, -args.separation / 2)
    vb = smpl_forward(data_b, i, +args.separation / 2)
    all_v = np.concatenate([va, vb], axis=0)
    frame_min_y = all_v[:, 1].min()
    floor_y = frame_min_y if floor_y is None else min(floor_y, frame_min_y)
    fx_min, fx_max = all_v[:, 0].min(), all_v[:, 0].max()
    fz_min, fz_max = all_v[:, 2].min(), all_v[:, 2].max()
    x_min = fx_min if x_min is None else min(x_min, fx_min)
    x_max = fx_max if x_max is None else max(x_max, fx_max)
    z_min = fz_min if z_min is None else min(z_min, fz_min)
    z_max = fz_max if z_max is None else max(z_max, fz_max)

# Recenter the SCENE (not the camera) around the origin. The light is
# separately hardcoded to aim at the world origin -- moving the camera
# to frame real AMASS data (which can sit anywhere in world space) would
# leave the light pointed at the wrong place, correctly framed but
# unlit. Shifting the data instead keeps the camera and light exactly as
# originally (and correctly) set up.
recenter_x = (x_min + x_max) / 2.0
recenter_z = (z_min + z_max) / 2.0
print(f"Recentering scene by ({-recenter_x:.2f}, 0, {-recenter_z:.2f}) so it sits under the existing camera/light")

floor_y -= 0.02  # a small margin so feet don't visually clip through the floor
floor_verts, floor_normals, floor_faces_local = make_floor(floor_y)
print(f"Floor placed at y={floor_y:.4f}")


def render_frame(i):
    recenter = np.array([recenter_x, 0.0, recenter_z], dtype=np.float32)
    verts_a = smpl_forward(data_a, i, -args.separation / 2) - recenter
    verts_b = smpl_forward(data_b, i, +args.separation / 2) - recenter

    combined_verts = np.concatenate([verts_a, verts_b, floor_verts], axis=0)
    combined_normals = np.concatenate([
        compute_vertex_normals(verts_a, faces_np.astype(np.int64)),
        compute_vertex_normals(verts_b, faces_np.astype(np.int64)),
        floor_normals,
    ], axis=0)
    # Body B's faces are offset past body A's vertex count; the floor's
    # faces are offset past BOTH bodies' vertex counts.
    combined_faces = np.concatenate([
        faces_np,
        faces_np + n_verts,
        floor_faces_local + (2 * n_verts),
    ], axis=0)

    with open(_mesh_path, "wb") as f:
        f.write(struct.pack("<II", combined_verts.shape[0], combined_faces.shape[0]))
        f.write(combined_verts.tobytes())
        f.write(combined_normals.tobytes())
        f.write(combined_faces.tobytes())

    proc = subprocess.run([_exe_path, str(args.resolution)],
                           cwd=os.path.join(_vulkan_dir, "build"),
                           capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Frame {i}: standalone executable failed (exit {proc.returncode})")
        print(proc.stderr[-500:])
        return None, None

    with open(_output_bin_path, "rb") as f:
        w = int.from_bytes(f.read(4), "little")
        h = int.from_bytes(f.read(4), "little")
        vdata = np.frombuffer(f.read(), dtype=np.float32).reshape(h, w, 2)
    return vdata[:, :, 1], vdata[:, :, 0] > 0


def colorize(a):
    a_toned = np.power(np.clip(a, 0.0, 1.0), 0.6)
    stops = np.array([
        [0.00, 0.00, 0.00], [0.35, 0.05, 0.00], [0.85, 0.30, 0.00],
        [1.00, 0.80, 0.40], [1.00, 1.00, 1.00],
    ])
    stop_positions = np.linspace(0.0, 1.0, len(stops))
    r = np.interp(a_toned, stop_positions, stops[:, 0])
    g = np.interp(a_toned, stop_positions, stops[:, 1])
    b = np.interp(a_toned, stop_positions, stops[:, 2])
    return np.stack([r, g, b], axis=-1)


print("Pass 1/2: finding the shared brightness scale...")
global_max = 1.0
for i in range(n_frames):
    intensity, hitmask = render_frame(i)
    if intensity is not None and hitmask.any():
        global_max = max(global_max, float(intensity[hitmask].max()))
    if (i + 1) % 50 == 0 or i == n_frames - 1:
        print(f"  ...{i+1}/{n_frames}")
print(f"Global intensity max: {global_max:.4f}")

print("Pass 2/2: rendering and saving frames...")
frame_dir = os.path.join(_repo_root, "_two_bodies_frames_tmp")
os.makedirs(frame_dir, exist_ok=True)
BACKGROUND = np.array([0.03, 0.03, 0.05])
for i in range(n_frames):
    intensity, hitmask = render_frame(i)
    if intensity is None:
        intensity = np.zeros((args.resolution, args.resolution))
        hitmask = np.zeros((args.resolution, args.resolution), dtype=bool)
    a = np.clip(np.where(hitmask, intensity, 0.0) / global_max, 0.0, 1.0)
    rgb = colorize(a)
    rgb[~hitmask] = BACKGROUND
    Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(frame_dir, f"frame_{i:04d}.png"))
    if (i + 1) % 50 == 0 or i == n_frames - 1:
        print(f"  ...{i+1}/{n_frames}")

subprocess.run([
    "ffmpeg", "-y", "-framerate", str(args.fps),
    "-i", os.path.join(frame_dir, "frame_%04d.png"),
    "-pix_fmt", "yuv420p", output_path
], check=True)

shutil.rmtree(frame_dir)
print(f"\nSaved {output_path}")
