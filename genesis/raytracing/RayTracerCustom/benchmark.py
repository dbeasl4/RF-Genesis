"""
Benchmark: custom CUDA ray tracer vs. Mitsuba, on the same motion data.
Prints per-frame and total timing for both renderers.
"""

import sys
import os
import time
import numpy as np
import torch

# Make the RF-Genesis repo root importable
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Import Mitsuba's RayTracer
from genesis.raytracing.pathtracer import RayTracer as MitsubaRayTracer
from genesis.raytracing.RayTracerCustom.raytracer import RayTracer as CustomRayTracer


def load_motion(motion_filename):
    data = np.load(motion_filename, allow_pickle=True)
    return data['pose'], data['shape'], data['root_translation']


def bench_mitsuba(pose, shape, root_translation, n_frames):
    tracer = MitsubaRayTracer()
    body_offset = np.array([0, 1, 3])
    t0 = time.perf_counter()
    for i in range(n_frames):
        tracer.update_pose(pose[i], shape[0], np.array(root_translation[i]) - body_offset)
        tracer.trace()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_custom(pose, shape, root_translation, n_frames):
    tracer = CustomRayTracer()
    tracer.update_sensor(origin=(0, 0, 0), target=(0, 0, -5))
    body_offset = np.array([0, 1, 3])
    t0 = time.perf_counter()
    for i in range(n_frames):
        tracer.update_pose(pose[i], shape[0], np.array(root_translation[i]) - body_offset)
        tracer.trace()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


if __name__ == "__main__":
    motion_file = sys.argv[1] if len(sys.argv) > 1 else "../../../output/hello_custom/obj_diff.npz"
    motion_file = os.path.abspath(motion_file)

    pose, shape, root_translation = load_motion(motion_file)
    n_frames = len(root_translation)

    os.chdir(os.path.join(_repo_root, "genesis"))

    print(f"Benchmarking {n_frames} frames from {motion_file}\n")

    mitsuba_time = bench_mitsuba(pose, shape, root_translation, n_frames)
    print(f"Mitsuba:        {mitsuba_time:.3f}s total, {mitsuba_time / n_frames * 1000:.2f} ms/frame")

    custom_time = bench_custom(pose, shape, root_translation, n_frames)
    print(f"Custom CUDA:    {custom_time:.3f}s total, {custom_time / n_frames * 1000:.2f} ms/frame")

    print(f"\nSpeedup: {mitsuba_time / custom_time:.2f}x")