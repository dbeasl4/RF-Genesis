"""
Python wrapper for the live Vulkan ray tracer, matching the same interface
as pathtracer.RayTracer and the CUDA version's RayTracer: update_pose(),
update_sensor(), trace() returning (PIR, pointclouds).

Pose/shape parameters go through the SMPL model in PyTorch, the resulting
vertices get uploaded into a persistent Vulkan buffer, and the trace's
output comes back as a zero-copy CUDA tensor, converted to the numpy
(PIR, pointclouds) shape the rest of the pipeline expects.
"""
import os
import sys
import numpy as np
import torch

import vulkan_raytracer_live

_this_dir = os.path.dirname(os.path.abspath(__file__))
_shader_dir = os.path.abspath(os.path.join(_this_dir, "..", "build", "shaders"))

# Robust import of smpl regardless of how this module gets loaded (via
# sys.path.insert in a standalone script, or as a proper package submodule
# via pathtracer.py's relative import). Always resolve through the repo
# root absolutely, avoiding any relative-vs-absolute import ambiguity.
_repo_root = os.path.abspath(os.path.join(_this_dir, "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from genesis.raytracing import smpl as smpl_utils


def compute_vertex_normals(vertices, faces):
    """
    Per-vertex smooth normals, recomputed fresh from the CURRENT frame's
    deformed vertices -- topology (faces) is fixed, but positions (and
    therefore normals) change with pose every call. Torch equivalent of
    export_mesh.py's numpy version, used for the standalone executable's
    static case; this one runs every frame in the hot path here.
    vertices: (V, 3) float tensor. faces: (F, 3) int/long tensor.
    Returns: (V, 3) float tensor of unit normals.
    """
    faces_long = faces.long()
    v0 = vertices[faces_long[:, 0]]
    v1 = vertices[faces_long[:, 1]]
    v2 = vertices[faces_long[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)

    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.index_add_(0, faces_long[:, 0], face_normals)
    vertex_normals.index_add_(0, faces_long[:, 1], face_normals)
    vertex_normals.index_add_(0, faces_long[:, 2], face_normals)

    norms = vertex_normals.norm(dim=1, keepdim=True)
    norms = torch.where(norms < 1e-12, torch.ones_like(norms), norms)
    return vertex_normals / norms


class RayTracer:
    def __init__(self, body=None, resolution=128, fov=60.0):
        """
        body: an already-loaded SMPL layer, e.g. smpl.get_smpl_layer().
              If not given, loads one internally, making RayTracer()
              a true zero-argument drop-in, matching pathtracer.RayTracer
              and the CUDA version's RayTracer.
        """
        if body is None:
            body = smpl_utils.get_smpl_layer()
        self.body = body
        self.resolution = resolution
        self.faces = torch.as_tensor(body.th_faces, dtype=torch.int32)
        self._impl = vulkan_raytracer_live.RayTracer(self.faces, resolution, fov, _shader_dir)

    def update_sensor(self, origin, target):
        self._impl.update_sensor(list(map(float, origin)), list(map(float, target)))

    def update_pose(self, pose_params, shape_params, translation=None):
        pose_t = torch.as_tensor(pose_params).view(1, -1).float()
        shape_t = torch.as_tensor(shape_params).view(1, -1).float()

        vertices, _ = self.body(pose_t, th_betas=shape_t)
        vertices = vertices[0]

        if translation is not None:
            translation_t = torch.as_tensor(translation, dtype=torch.float32)
            vertices = vertices + translation_t

        normals = compute_vertex_normals(vertices, self.faces)
        self._impl.update_pose(vertices.cpu(), normals.cpu())

    def trace(self):
        PIR_gpu, pointclouds_gpu = self._impl.trace()
        PIR = PIR_gpu.cpu().numpy()
        pointclouds = pointclouds_gpu.cpu().numpy()
        return PIR, pointclouds


