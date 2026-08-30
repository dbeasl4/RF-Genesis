"""
Loads vulkan_output.bin (written by the Vulkan program) and saves the
distance and intensity channels as grayscale PNGs for a quick visual
check, same idea used for comparing the CUDA version against Mitsuba.
Run from the vulkan_raytracer/ folder.
"""

import sys
import struct
import numpy as np
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "build/vulkan_output.bin"

with open(path, "rb") as f:
    width, height = struct.unpack("<II", f.read(8))
    data = np.frombuffer(f.read(), dtype=np.float32).reshape(height, width, 2)

distance = data[:, :, 0]
intensity = data[:, :, 1]


def save_grayscale(arr, out_path, is_distance=False):
    a = arr.copy()
    hit_mask = a >= 0  # -1 is the miss sentinel

    if is_distance and hit_mask.any():
        # Normalize by the actual min/max among hits
        hit_min = a[hit_mask].min()
        hit_max = a[hit_mask].max()
        span = max(hit_max - hit_min, 1e-6)
        a = np.where(hit_mask, (a - hit_min) / span, 0.0)
    else:
        a[~hit_mask] = 0
        max_val = a.max() if a.max() > 0 else 1.0
        a = a / max_val

    img = (a * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


save_grayscale(distance, "vulkan_distance.png", is_distance=True)
save_grayscale(intensity, "vulkan_intensity.png", is_distance=False)

hits = int((distance >= 0).sum())
print(f"Loaded {width}x{height} output from {path}")
print(f"Hits: {hits} / {width * height}")
if hits > 0:
    hit_mask = distance >= 0
    print(f"Distance range:  [{distance[hit_mask].min():.5f}, {distance[hit_mask].max():.5f}]")
    print(f"Intensity range: [{intensity.min():.5f}, {intensity.max():.5f}]")
print("Saved vulkan_distance.png and vulkan_intensity.png")