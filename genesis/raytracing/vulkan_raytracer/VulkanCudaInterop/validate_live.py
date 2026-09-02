"""
Validates the live Vulkan ray tracer (persistent, PyTorch-driven) against
Mitsuba, exercising the real, integrated Python-callable class. See
GETTING_STARTED.md for how to run this.
"""
import sys
import os
import argparse
import numpy as np
from PIL import Image
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

parser = argparse.ArgumentParser()
parser.add_argument("--motion", default=None)
parser.add_argument("--frame", type=int, default=0)
args = parser.parse_args()
motion_path = os.path.abspath(args.motion) if args.motion else None

distance_out = os.path.abspath("live_distance.png")
intensity_out = os.path.abspath("live_intensity.png")
mitsuba_distance_out = os.path.abspath("mitsuba_distance.png")
mitsuba_intensity_out = os.path.abspath("mitsuba_intensity.png")
compare_distance_out = os.path.abspath("compare_distance.png")
compare_intensity_out = os.path.abspath("compare_intensity.png")
diff_intensity_out = os.path.abspath("diff_intensity.png")

os.chdir(os.path.join(_repo_root, "genesis"))

from genesis.raytracing import smpl as smpl_utils
from genesis.raytracing.pathtracer import RayTracer as MitsubaRayTracer

sys.path.insert(0, os.path.join(_repo_root, "genesis", "raytracing", "VulkanCudaInterop"))
from live_raytracer import RayTracer as LiveVulkanRayTracer

body = smpl_utils.get_smpl_layer()

if motion_path:
    data = np.load(motion_path, allow_pickle=True)
    pose = data['pose'][args.frame].astype(np.float32)
    shape = data['shape'][0].astype(np.float32)
    root_translation = np.array(data['root_translation'][args.frame])
    body_offset = np.array([0, 1, 3])
    translation = (root_translation - body_offset).astype(np.float32)
    print(f"Using frame {args.frame} from {motion_path}")
else:
    pose = np.zeros(72, dtype=np.float32)
    shape = np.zeros(10, dtype=np.float32)
    translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    print("Using rest pose (no --motion given)")

# Live Vulkan
live_tracer = LiveVulkanRayTracer(body, resolution=128, fov=60.0)
live_tracer.update_pose(pose, shape, translation)
live_PIR, _ = live_tracer.trace()
live_distance = live_PIR[:, :, 0]
live_intensity = live_PIR[:, :, 1]
live_hits = int((live_distance > 0).sum())

# Mitsuba
mitsuba_tracer = MitsubaRayTracer()
mitsuba_tracer.update_pose(pose, shape, translation)
mitsuba_PIR, _ = mitsuba_tracer.trace()
mitsuba_distance = mitsuba_PIR[:, :, 0]
mitsuba_intensity = mitsuba_PIR[:, :, 1]
mitsuba_hits = int((mitsuba_distance > 0).sum())

print(f"\n{'':20s} {'Mitsuba':>15s} {'Live Vulkan':>15s}")
print(f"{'Hits':20s} {mitsuba_hits:>15d} {live_hits:>15d}")
if mitsuba_hits > 0:
    print(f"{'Distance min':20s} {mitsuba_distance[mitsuba_distance>0].min():>15.5f} "
          f"{live_distance[live_distance>0].min() if live_hits > 0 else float('nan'):>15.5f}")
    print(f"{'Distance max':20s} {mitsuba_distance[mitsuba_distance>0].max():>15.5f} "
          f"{live_distance[live_distance>0].max() if live_hits > 0 else float('nan'):>15.5f}")
print(f"{'Intensity max':20s} {mitsuba_intensity.max():>15.5f} {live_intensity.max():>15.5f}")
# Max is a single-pixel statistic and can be noisy (e.g. filter-ringing on
# Mitsuba's gaussian-reconstruction side, or a single pixel right on the
# spot cone's boundary). Mean over actual hit pixels is a steadier signal
# for whether the whole image's brightness matches, not just one pixel.
mitsuba_mean = mitsuba_intensity[mitsuba_intensity > 0].mean() if mitsuba_hits > 0 else float('nan')
live_mean = live_intensity[live_distance > 0].mean() if live_hits > 0 else float('nan')
print(f"{'Intensity mean':20s} {mitsuba_mean:>15.5f} {live_mean:>15.5f}")
# Mean is sensitive to a handful of extreme outlier pixels; median isn't.
# If mean is way off but median is close, the gap is likely a few bad
# pixels, not a systematic brightness/shading bias across the whole body.
mitsuba_median = np.median(mitsuba_intensity[mitsuba_intensity > 0]) if mitsuba_hits > 0 else float('nan')
live_median = np.median(live_intensity[live_distance > 0]) if live_hits > 0 else float('nan')
print(f"{'Intensity median':20s} {mitsuba_median:>15.5f} {live_median:>15.5f}")
live_hit_vals = live_intensity[live_distance > 0]
if live_hit_vals.size > 0:
    p99 = np.percentile(live_hit_vals, 99)
    n_outliers = int((live_hit_vals > 3 * live_median).sum()) if live_median > 0 else 0
    print(f"{'Vulkan 99th pct':20s} {'':>15s} {p99:>15.5f}")
    print(f"{'Pixels > 3x median':20s} {'':>15s} {n_outliers:>15d}")

# Erode the hit masks so only pixels well inside the silhouette remain --
# excludes soft, anti-aliased edge pixels (Mitsuba's spp=32 + gaussian
# filter blends these toward the background) to isolate whatever gap is
# left once edge effects are removed.
try:
    import scipy.ndimage as ndi
    mitsuba_hit_mask = mitsuba_intensity > 0
    live_hit_mask = live_distance > 0
    core_mi = ndi.binary_erosion(mitsuba_hit_mask, iterations=2)
    core_lv = ndi.binary_erosion(live_hit_mask, iterations=2)
    if core_mi.sum() > 0 and core_lv.sum() > 0:
        print(f"{'Core-only mean':20s} {mitsuba_intensity[core_mi].mean():>15.5f} "
              f"{live_intensity[core_lv].mean():>15.5f}")
    else:
        print("(core region empty after erosion -- silhouette too small/thin for this check)")
except ImportError:
    print("(install scipy to run the core-only mean check: pip install scipy)")


def save_grayscale(arr, out_path, is_distance=False, hit_test=None, shared_range=None, scale=6):
    a = arr.copy()
    hit_mask = hit_test(a) if hit_test else (a > 0)
    if is_distance and hit_mask.any():
        lo, hi = shared_range if shared_range else (a[hit_mask].min(), a[hit_mask].max())
        span = max(hi - lo, 1e-6)
        a = np.where(hit_mask, (a - lo) / span, 0.0)
    else:
        lo, hi = shared_range if shared_range else (0.0, a.max() if a.max() > 0 else 1.0)
        span = max(hi - lo, 1e-6)
        a = np.where(hit_mask, np.clip((a - lo) / span, 0.0, 1.0), 0.0)
    img = Image.fromarray((a * 255).astype(np.uint8))
    if scale > 1:
        # NEAREST keeps pixels crisp/blocky rather than blurring a small
        # source image -- more honest for a 128x128 render than smoothing.
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(out_path)
    return out_path


def save_side_by_side(left_arr, right_arr, out_path, left_label, right_label,
                       is_distance=False, shared_range=None, scale=6):
    """Two grayscale images side by side, sharing one normalization range so
    real brightness/distance differences between them stay visible instead
    of each panel being auto-stretched to its own max."""
    def normalize(a):
        hit_mask = a > 0
        if is_distance and hit_mask.any():
            lo, hi = shared_range
            span = max(hi - lo, 1e-6)
            return np.where(hit_mask, np.clip((a - lo) / span, 0.0, 1.0), 0.0)
        lo, hi = shared_range
        span = max(hi - lo, 1e-6)
        return np.where(hit_mask, np.clip((a - lo) / span, 0.0, 1.0), 0.0)

    left_img = Image.fromarray((normalize(left_arr) * 255).astype(np.uint8)).convert("RGB")
    right_img = Image.fromarray((normalize(right_arr) * 255).astype(np.uint8)).convert("RGB")
    if scale > 1:
        left_img = left_img.resize((left_img.width * scale, left_img.height * scale), Image.NEAREST)
        right_img = right_img.resize((right_img.width * scale, right_img.height * scale), Image.NEAREST)

    label_h = 32
    w, h = left_img.size
    canvas = Image.new("RGB", (w * 2 + 8, h + label_h), (30, 30, 30))
    canvas.paste(left_img, (0, label_h))
    canvas.paste(right_img, (w + 8, label_h))
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = None
        draw.text((6, 4), left_label, fill=(255, 255, 255), font=font)
        draw.text((w + 14, 4), right_label, fill=(255, 255, 255), font=font)
    except Exception:
        pass
    canvas.save(out_path)
    return out_path


# Individual images (each normalized to its own hit range -- fine for
# eyeballing shape/silhouette, not for comparing absolute brightness).
save_grayscale(live_distance, distance_out, is_distance=True)
save_grayscale(live_intensity, intensity_out, is_distance=False)
save_grayscale(mitsuba_distance, mitsuba_distance_out, is_distance=True)
save_grayscale(mitsuba_intensity, mitsuba_intensity_out, is_distance=False)

# Side-by-side comparisons, sharing one normalization range across both
# panels so an actual brightness/distance gap between Mitsuba and Vulkan
# shows up visually instead of being normalized away.
dist_hit = (mitsuba_distance > 0) | (live_distance > 0)
if dist_hit.any():
    dist_lo = min(mitsuba_distance[mitsuba_distance > 0].min() if mitsuba_hits > 0 else 1e9,
                   live_distance[live_distance > 0].min() if live_hits > 0 else 1e9)
    dist_hi = max(mitsuba_distance.max(), live_distance.max())
    save_side_by_side(mitsuba_distance, live_distance, compare_distance_out,
                       "Mitsuba", "Vulkan", is_distance=True, shared_range=(dist_lo, dist_hi))

def save_diff_heatmap(mitsuba_arr, live_arr, out_path, scale=6):
    """abs(live - mitsuba) as a color heatmap, so the intensity gap we
    measured numerically (mean off by ~40-50%) actually shows up visually
    instead of being inferred from two panels that individually look
    similar at a glance."""
    diff = np.abs(live_arr - mitsuba_arr)
    mask = (mitsuba_arr > 0) | (live_arr > 0)
    diff = np.where(mask, diff, 0.0)
    vmax = diff.max() if diff.max() > 0 else 1.0
    norm = np.clip(diff / vmax, 0.0, 1.0)

    try:
        import matplotlib.cm as cm
        colored = (cm.get_cmap('inferno')(norm)[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        # black -> red -> yellow -> white ramp, no matplotlib dependency
        colored = np.zeros((*norm.shape, 3), dtype=np.uint8)
        colored[..., 0] = (np.clip(norm * 3, 0, 1) * 255).astype(np.uint8)
        colored[..., 1] = (np.clip(norm * 3 - 1, 0, 1) * 255).astype(np.uint8)
        colored[..., 2] = (np.clip(norm * 3 - 2, 0, 1) * 255).astype(np.uint8)

    img = Image.fromarray(colored)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    label_h = 32
    canvas = Image.new("RGB", (img.width, img.height + label_h), (30, 30, 30))
    canvas.paste(img, (0, label_h))
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = None
        draw.text((6, 4), f"|Vulkan - Mitsuba| intensity  (brightest = {vmax:.2f} diff)",
                   fill=(255, 255, 255), font=font)
    except Exception:
        pass
    canvas.save(out_path)
    return out_path


intens_hi = max(mitsuba_intensity.max(), live_intensity.max(), 1e-6)
save_side_by_side(mitsuba_intensity, live_intensity, compare_intensity_out,
                   "Mitsuba", "Vulkan", is_distance=False, shared_range=(0.0, intens_hi))
save_diff_heatmap(mitsuba_intensity, live_intensity, diff_intensity_out)

print(f"\nSaved {distance_out} and {intensity_out}")
print(f"Saved {mitsuba_distance_out} and {mitsuba_intensity_out}")
print(f"Saved side-by-side comparisons: {compare_distance_out}")
print(f"                                {compare_intensity_out}")
print(f"Saved difference heatmap:       {diff_intensity_out}")