"""
Full-sequence validation: runs BOTH renderers across every frame (or a
strided subset) of a real motion capture sequence, and reports honest,
aggregate statistics -- not just a couple of hand-picked frames.

For each frame: writes the mesh, runs the standalone Vulkan executable
as a subprocess (safe, isolated -- no persistent live GPU state involved
at all), runs Mitsuba in-process, and compares hit counts, distance, and
intensity.

Run from vulkan_raytracer/, after building ./build/vulkan_raytracer at
128x128 (the pipeline's real resolution -- revert WIDTH/HEIGHT back down
first if you'd bumped them up for visual inspection).

    python validate_full_sequence.py ../../../output/hello_rfgen/obj_diff.npz
    python validate_full_sequence.py ../../../output/hello_rfgen/obj_diff.npz --stride 4
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
parser.add_argument("--stride", type=int, default=1, help="Test every Nth frame (default: all)")
args = parser.parse_args()
motion_path = os.path.abspath(args.motion_file)

_this_dir = os.path.dirname(os.path.abspath(__file__))
_mesh_path = os.path.join(_this_dir, "smpl_mesh.bin")
_exe_path = os.path.join(_this_dir, "build", "vulkan_raytracer")
_output_path = os.path.join(_this_dir, "build", "vulkan_output.bin")


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

mitsuba_tracer = MitsubaRayTracer()

data = np.load(motion_path, allow_pickle=True)
n_frames = len(data['root_translation'])
frame_indices = list(range(0, n_frames, args.stride))
body_offset = np.array([0, 1, 3])

print(f"Validating {len(frame_indices)} of {n_frames} frames (stride={args.stride})\n")

results = []
for idx in frame_indices:
    pose = data['pose'][idx].astype(np.float32)
    shape = data['shape'][0].astype(np.float32)
    root_translation = np.array(data['root_translation'][idx])
    translation = (root_translation - body_offset).astype(np.float32)

    # --- Vulkan: write mesh, run standalone executable, read output ---
    pose_t = torch.tensor(pose).view(1, -1).float()
    shape_t = torch.tensor(shape).view(1, -1).float()
    vertices, _ = body(pose_t, th_betas=shape_t)
    vertices = (vertices[0] + torch.as_tensor(translation)).detach().cpu().numpy().astype(np.float32)
    normals = compute_vertex_normals(vertices, faces_np.astype(np.int64))

    with open(_mesh_path, "wb") as f:
        f.write(struct.pack("<II", vertices.shape[0], faces_np.shape[0]))
        f.write(vertices.tobytes())
        f.write(normals.tobytes())
        f.write(faces_np.tobytes())

    proc = subprocess.run([_exe_path], cwd=os.path.join(_this_dir, "build"),
                           capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Frame {idx}: Vulkan executable failed (exit {proc.returncode}), skipping")
        print(proc.stderr[-500:])
        continue

    with open(_output_path, "rb") as f:
        vw = int.from_bytes(f.read(4), "little")
        vh = int.from_bytes(f.read(4), "little")
        vdata = np.frombuffer(f.read(), dtype=np.float32).reshape(vh, vw, 2)
    v_distance = vdata[:, :, 0]
    v_intensity = vdata[:, :, 1]
    v_hit = v_distance > 0

    # --- Mitsuba, same frame ---
    mitsuba_tracer.update_pose(pose, shape, translation)
    PIR, _ = mitsuba_tracer.trace()
    m_distance = PIR[:, :, 0]
    m_intensity = PIR[:, :, 1]
    m_hit = m_distance > 0

    both_hit = v_hit & m_hit
    n_both = int(both_hit.sum())

    row = {
        "frame": idx,
        "v_hits": int(v_hit.sum()),
        "m_hits": int(m_hit.sum()),
        "both_hits": n_both,
    }
    if n_both > 0:
        row["mean_abs_dist_err"] = float(np.abs(v_distance[both_hit] - m_distance[both_hit]).mean())
        ratio = v_intensity[both_hit] / np.maximum(m_intensity[both_hit], 1e-6)
        row["mean_intensity_ratio"] = float(ratio.mean())
        row["max_intensity_ratio"] = float(ratio.max())
        row["min_intensity_ratio"] = float(ratio.min())
    results.append(row)

    if len(results) % 20 == 0 or idx == frame_indices[-1]:
        print(f"  ...{len(results)}/{len(frame_indices)} frames done")

print(f"\n{'Frame':>6} {'V hits':>8} {'M hits':>8} {'Both':>6} {'DistErr':>9} {'MeanRatio':>10} {'MinRatio':>9} {'MaxRatio':>9}")
for r in results:
    if "mean_intensity_ratio" in r:
        print(f"{r['frame']:>6} {r['v_hits']:>8} {r['m_hits']:>8} {r['both_hits']:>6} "
              f"{r['mean_abs_dist_err']:>9.4f} {r['mean_intensity_ratio']:>10.4f} "
              f"{r['min_intensity_ratio']:>9.4f} {r['max_intensity_ratio']:>9.4f}")
    else:
        print(f"{r['frame']:>6} {r['v_hits']:>8} {r['m_hits']:>8} {r['both_hits']:>6}  (no overlapping hits)")

valid = [r for r in results if "mean_intensity_ratio" in r]
if valid:
    ratios = np.array([r["mean_intensity_ratio"] for r in valid])
    dist_errs = np.array([r["mean_abs_dist_err"] for r in valid])
    print(f"\n--- Summary across {len(valid)} frames ---")
    print(f"Mean distance error:     {dist_errs.mean():.5f}  (max: {dist_errs.max():.5f})")
    print(f"Mean intensity ratio:    {ratios.mean():.4f}  (i.e. Vulkan averages {ratios.mean()*100:.1f}% of Mitsuba)")
    print(f"Ratio range across seq:  [{ratios.min():.4f}, {ratios.max():.4f}]")
    worst = valid[np.argmin(ratios)]
    print(f"Worst frame (lowest ratio): frame {worst['frame']}, ratio {worst['mean_intensity_ratio']:.4f}")
