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

# Avoiding any relative-vs-absolute import ambiguity.
_repo_root = os.path.abspath(os.path.join(_this_dir, "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from genesis.raytracing import smpl as smpl_utils


class RayTracer:
    def __init__(self, body=None, resolution=128, fov=60.0):
        """
        body: an already-loaded SMPL layer, e.g. smpl.get_smpl_layer().
              If not given, loads one internally, this makes RayTracer()
              a true zero-argument drop-in, matching pathtracer.RayTracer
              and the CUDA version's RayTracer.
        """
        if body is None:
            body = smpl_utils.get_smpl_layer()
        self.body = body
        self.resolution = resolution
        faces = torch.as_tensor(body.th_faces, dtype=torch.int32)
        self._impl = vulkan_raytracer_live.RayTracer(faces, resolution, fov, _shader_dir)

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

        self._impl.update_pose(vertices.cpu())

    def trace(self):
        PIR_gpu, pointclouds_gpu = self._impl.trace()
        PIR = PIR_gpu.cpu().numpy()
        pointclouds = pointclouds_gpu.cpu().numpy()
        return PIR, pointclouds

