"""
Drop-in replacement for genesis/raytracing/pathtracer.py's RayTracer,
backed by a custom brute-force CUDA kernel instead of Mitsuba.

Matches the real interface used by pathtracer.py's module-level trace():

    raytracer = RayTracer()
    raytracer.update_pose(pose_params, shape_params, translation)
    PIR, pointclouds = raytracer.trace()

Differences from the Mitsuba version (see raytracer_kernel.cu's NEXT STEPS
comment for how to close these gaps):
  - No shadow ray yet -> no shadows cast by the body onto itself/the wall.
  - Wall plane not included yet -> only renders the SMPL mesh.
"""

import torch
import numpy as np

from .. import smpl
import custom_raytracer_cuda


class RayTracer:

    def __init__(self, resolution=128, fov=60.0,
                 light_pos=(0.0, 0.0, 3.0), light_intensity=1000.0):
        self.PIR_resolution = resolution
        self.fov = fov
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.body = smpl.get_smpl_layer()
        self.faces = torch.as_tensor(self.body.th_faces, dtype=torch.int32, device=self.device).contiguous()

        self.vertices = None  # float32 [V, 3] cuda tensor, set per-frame

        self.cam_origin = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float32, device=self.device)
        self.cam_target = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)

        self.light_pos = torch.tensor(light_pos, dtype=torch.float32, device=self.device)
        self.light_intensity = float(light_intensity)

    def update_pose(self, pose_params, shape_params, translation=None):
        """Matches pathtracer.RayTracer.update_pose's real signature."""
        pose_t = torch.tensor(pose_params, device=self.device).view(1, -1).float()
        shape_t = torch.tensor(shape_params, device=self.device).view(1, -1).float()

        vertices, _ = self.body(pose_t, th_betas=shape_t)
        vertices = vertices[0]

        if translation is not None:
            translation_t = torch.as_tensor(translation, dtype=torch.float32, device=self.device)
            vertices = vertices + translation_t

        self.vertices = vertices.contiguous()

    def update_sensor(self, origin, target):
        self.cam_origin = torch.as_tensor(origin, dtype=torch.float32, device=self.device)
        self.cam_target = torch.as_tensor(target, dtype=torch.float32, device=self.device)

    def _camera_basis(self):
        forward = torch.nn.functional.normalize(self.cam_target - self.cam_origin, dim=0)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        right = torch.nn.functional.normalize(torch.cross(forward, world_up, dim=0), dim=0)
        up = torch.cross(right, forward, dim=0)
        return right, up, forward

    def trace(self):
        """Matches pathtracer.RayTracer.trace(): returns (PIR, pointclouds)."""
        if self.vertices is None:
            raise RuntimeError("Call update_pose() before trace().")

        right, up, forward = self._camera_basis()

        distance, intensity = custom_raytracer_cuda.raytrace(
            self.vertices, self.faces,
            self.cam_origin, right, up, forward,
            self.fov,
            self.PIR_resolution, self.PIR_resolution,
            self.light_pos, self.light_intensity
        )

        distance_np = distance.cpu().numpy()
        intensity_np = intensity.cpu().numpy()
        velocity_np = np.zeros_like(distance_np)
        PIR = np.stack([distance_np, intensity_np, velocity_np], axis=2)

        # Reconstruct world-space hit position per pixel, equivalent to
        # Mitsuba si.p, from distance and per-pixel ray direction, without
        # needing the kernel to output it separately: hit_pos = origin + dir * t
        res = self.PIR_resolution
        aspect = 1.0 
        tan_half_fov = torch.tan(torch.deg2rad(torch.tensor(self.fov / 2.0, device=self.device)))
        xs = (2.0 * (torch.arange(res, device=self.device).float() + 0.5) / res - 1.0) * tan_half_fov * aspect
        ys = (1.0 - 2.0 * (torch.arange(res, device=self.device).float() + 0.5) / res) * tan_half_fov
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')

        dirs = (grid_x.unsqueeze(-1) * right + grid_y.unsqueeze(-1) * up + forward)
        dirs = torch.nn.functional.normalize(dirs, dim=-1)

        hit_pos = self.cam_origin + dirs * distance.unsqueeze(-1)
        pointclouds = hit_pos.reshape(-1, 3).cpu().numpy()

        return PIR, pointclouds