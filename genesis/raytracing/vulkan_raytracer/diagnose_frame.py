"""
Diagnostic: finds the exact pixel with the most extreme Vulkan/Mitsuba
intensity ratio at a given frame, and prints full details from both
renderers at that pixel -- distance, intensity, and world position --
to figure out what's actually going wrong there.
"""
import sys
import os
import struct
import argparse
import subprocess
import numpy as np
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

parser = argparse.ArgumentParser()
parser.add_argument("motion_file")
parser.add_argument("--frame", type=int, required=True)
args = parser.parse_args()
motion_path = os.path.abspath(args.motion_file)

_this_dir = os.path.dirname(os.path.abspath(__file__))


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


os.chdir(os.path.join(_repo_root, "genesis"))
from genesis.raytracing import smpl as smpl_utils
from genesis.raytracing.pathtracer import RayTracer as MitsubaRayTracer

body = smpl_utils.get_smpl_layer()
faces_np = body.th_faces.cpu().numpy().astype(np.uint32)

data = np.load(motion_path, allow_pickle=True)
pose = data['pose'][args.frame].astype(np.float32)
shape = data['shape'][0].astype(np.float32)
root_translation = np.array(data['root_translation'][args.frame])
body_offset = np.array([0, 1, 3])
translation = (root_translation - body_offset).astype(np.float32)

pose_t = torch.tensor(pose).view(1, -1).float()
shape_t = torch.tensor(shape).view(1, -1).float()
vertices, _ = body(pose_t, th_betas=shape_t)
vertices = (vertices[0] + torch.as_tensor(translation)).detach().cpu().numpy().astype(np.float32)
normals = compute_vertex_normals(vertices, faces_np.astype(np.int64))

mesh_path = os.path.join(_this_dir, "smpl_mesh.bin")
with open(mesh_path, "wb") as f:
    f.write(struct.pack("<II", vertices.shape[0], faces_np.shape[0]))
    f.write(vertices.tobytes())
    f.write(normals.tobytes())
    f.write(faces_np.tobytes())

subprocess.run([os.path.join(_this_dir, "build", "vulkan_raytracer")],
               cwd=os.path.join(_this_dir, "build"), check=True)

with open(os.path.join(_this_dir, "build", "vulkan_output.bin"), "rb") as f:
    vw = int.from_bytes(f.read(4), "little")
    vh = int.from_bytes(f.read(4), "little")
    vdata = np.frombuffer(f.read(), dtype=np.float32).reshape(vh, vw, 2)
v_distance = vdata[:, :, 0]
v_intensity = vdata[:, :, 1]

mitsuba_tracer = MitsubaRayTracer()
mitsuba_tracer.update_pose(pose, shape, translation)
PIR, _ = mitsuba_tracer.trace()
m_distance = PIR[:, :, 0]
m_intensity = PIR[:, :, 1]

both_hit = (v_distance > 0) & (m_distance > 0)
ratio = np.where(both_hit, v_intensity / np.maximum(m_intensity, 1e-6), 0.0)

y, x = np.unravel_index(np.argmax(ratio), ratio.shape)
print(f"Frame {args.frame}, worst pixel: ({y}, {x})")
print(f"  Vulkan:  distance={v_distance[y,x]:.6f}  intensity={v_intensity[y,x]:.6f}")
print(f"  Mitsuba: distance={m_distance[y,x]:.6f}  intensity={m_intensity[y,x]:.6f}")
print(f"  Ratio: {ratio[y,x]:.4f}")

# A few neighboring pixels too, for context on whether this is an
# isolated single-pixel spike or a small localized region.
print("\nNeighboring pixels (3x3 around worst pixel):")
for dy in [-1, 0, 1]:
    for dx in [-1, 0, 1]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < vh and 0 <= nx < vw:
            print(f"  ({ny},{nx}): V_dist={v_distance[ny,nx]:.4f} V_int={v_intensity[ny,nx]:.4f} "
                  f"M_dist={m_distance[ny,nx]:.4f} M_int={m_intensity[ny,nx]:.4f}")
