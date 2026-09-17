"""
Exports one frame of SMPL mesh data to a simple binary file the Vulkan
program can load directly. With no arguments, exports the rest pose
(all-zero pose/shape), useful for a first correctness test.

Binary format:
    uint32 vertexCount
    uint32 faceCount
    float32 vertices[vertexCount * 3]   x,y,z per vertex
    float32 normals[vertexCount * 3]    x,y,z per-vertex smooth normal
    uint32  indices[faceCount * 3]      3 indices per triangle

Run from vulkan_raytracer/. See GETTING_STARTED.md for details.
"""
import sys
import os
import struct
import argparse
import numpy as np
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from genesis.raytracing import smpl as smpl_utils


def compute_vertex_normals(vertices, faces):
    """
    Per-vertex smooth normals: accumulate each triangle's (unnormalized,
    area-weighted) face normal into its 3 vertices, then normalize. This
    is what lets the shader interpolate a continuously varying normal
    across a triangle instead of using one flat normal per face.
    vertices: (V, 3) float32 array. faces: (F, 3) int array of indices.
    Returns: (V, 3) float32 array of unit normals.
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="smpl_mesh.bin")
    parser.add_argument("--motion", default=None, help="Path to obj_diff.npz to pull a real pose from")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    # Resolve paths to absolute before changing directory below, otherwise
    # relative paths the user passed in would break once cwd changes.
    output_path = os.path.abspath(args.output)
    motion_path = os.path.abspath(args.motion) if args.motion else None

    # smpl.get_smpl_layer() uses a relative path ('../models/smpl_models')
    # that only resolves correctly if the working directory is
    # RF-Genesis/genesis/, exactly what run.py does with its own
    # os.chdir("genesis/") before calling into the ray tracer. Replicate
    # that here (same fix already applied to benchmark.py).
    os.chdir(os.path.join(_repo_root, "genesis"))

    body = smpl_utils.get_smpl_layer()

    if motion_path:
        data = np.load(motion_path, allow_pickle=True)
        pose = torch.tensor(data['pose'][args.frame]).view(1, -1).float()
        shape = torch.tensor(data['shape'][0]).view(1, -1).float()
        root_translation = np.array(data['root_translation'][args.frame])
        body_offset = np.array([0, 1, 3])
        translation = torch.tensor((root_translation - body_offset).astype(np.float32))
        print(f"Using frame {args.frame} from {motion_path}")
    else:
        pose = torch.zeros(1, 72)
        shape = torch.zeros(1, 10)
        translation = None
        print("Using rest pose (no --motion given)")

    vertices, _ = body(pose, th_betas=shape)
    vertices = vertices[0]
    if translation is not None:
        vertices = vertices + translation.to(vertices.device)
    vertices = vertices.detach().cpu().numpy().astype(np.float32)
    faces = body.th_faces.cpu().numpy().astype(np.uint32)
    normals = compute_vertex_normals(vertices, faces.astype(np.int64))

    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", vertices.shape[0], faces.shape[0]))
        f.write(vertices.tobytes())
        f.write(normals.tobytes())
        f.write(faces.tobytes())

    print(f"Exported {vertices.shape[0]} vertices, {faces.shape[0]} triangles to {output_path}")
    print(f"Vertex bounds: x[{vertices[:,0].min():.3f},{vertices[:,0].max():.3f}] "
          f"y[{vertices[:,1].min():.3f},{vertices[:,1].max():.3f}] "
          f"z[{vertices[:,2].min():.3f},{vertices[:,2].max():.3f}]")


if __name__ == "__main__":
    main()
