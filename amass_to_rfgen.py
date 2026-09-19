"""
Converts a real AMASS motion capture file (SMPL+H parameterization) into
this pipeline's obj_diff.npz format, so real motion capture data can be
fed directly into run.py's Step 2/4 onward, skipping MDM's text-to-motion
generation (Step 1) entirely.

AMASS files contain more than this pipeline needs (hand articulation,
DMPL soft-tissue coefficients, extra shape dimensions) -- this script
extracts just the body pose, the first 10 shape coefficients, and the
root translation, and resamples to a fixed frame rate if the source
capture rate differs.

Get AMASS data (free, requires academic registration):
    https://amass.is.tue.mpg.de/

Usage:
    python amass_to_rfgen.py path/to/some_amass_file.npz -o obj_diff.npz
    python amass_to_rfgen.py path/to/some_amass_file.npz -o obj_diff.npz --target_fps 30

Known wrinkle worth checking after conversion: AMASS's sub-datasets can
vary slightly in floor-plane/up-axis convention. Render one frame with
export_mesh.py/view_output.py or watch the resulting video before
assuming this is perfectly correct -- if the body appears sideways or
upside-down, that's a coordinate-convention mismatch to fix, not
something guaranteed correct by construction.
"""
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("amass_file", help="Path to a downloaded AMASS .npz file")
parser.add_argument("-o", "--output", default="obj_diff_from_amass.npz")
parser.add_argument("--target_fps", type=float, default=30.0,
                     help="Resample to this frame rate (this pipeline's downstream radar signal generation assumes 30fps)")
args = parser.parse_args()

amass = np.load(args.amass_file, allow_pickle=True)

print(f"AMASS file keys: {list(amass.keys())}")

# AMASS's frame rate key name varies slightly between sub-datasets.
source_fps = None
for key in ("mocap_framerate", "mocap_frame_rate", "frame_rate"):
    if key in amass:
        source_fps = float(amass[key])
        break
if source_fps is None:
    print("WARNING: couldn't find a frame rate key in this file, assuming 120fps (a common AMASS default).")
    source_fps = 120.0

poses_full = amass["poses"]  # (n_frames, up to 156) -- SMPL+H, hands included
trans_full = amass["trans"]  # (n_frames, 3)
betas_full = amass["betas"]  # (16,) or similar, shared across the whole sequence

n_frames_source = poses_full.shape[0]
print(f"Source: {n_frames_source} frames at {source_fps:.1f} fps")

# Only the first 72 pose dimensions are standard SMPL body pose (3 root +
# 23 body joints x 3 axis-angle each) -- the rest is hand articulation
# this pipeline's SMPL model (10 shape coefficients, no hands) doesn't use.
pose_smpl = poses_full[:, :72].astype(np.float32)
shape_smpl = betas_full[:10].astype(np.float32)
translation = trans_full.astype(np.float32)

# Resample to the target frame rate by simple nearest-frame subsampling.
# A smoother option would interpolate between frames instead, but this
# pipeline's own frame interpolation (signal_generator.py) already blends
# between frames downstream, so simple subsampling is a reasonable,
# honest starting point rather than over-engineering this conversion.
if abs(source_fps - args.target_fps) > 0.5:
    n_frames_target = int(round(n_frames_source * args.target_fps / source_fps))
    indices = np.linspace(0, n_frames_source - 1, n_frames_target).round().astype(int)
    pose_smpl = pose_smpl[indices]
    translation = translation[indices]
    print(f"Resampled {source_fps:.1f}fps -> {args.target_fps:.1f}fps: "
          f"{n_frames_source} -> {len(indices)} frames")

# Match obj_diff.npz's existing shape convention: 'shape' has a leading
# dimension (this pipeline consistently indexes shape[0]), even though
# AMASS's betas apply to the whole sequence uniformly.
shape_out = shape_smpl[np.newaxis, :]

np.savez(args.output, pose=pose_smpl, shape=shape_out, root_translation=translation)

print(f"\nSaved {args.output}")
print(f"  pose: {pose_smpl.shape}")
print(f"  shape: {shape_out.shape}")
print(f"  root_translation: {translation.shape}")
print("\nNext: run.py's Step 1 (motion generation) checks for an existing")
print("obj_diff.npz and skips regenerating it if present -- so placing")
print(f"this file at the expected output/<name>/obj_diff.npz path lets")
print("Steps 2-4 run against this real motion capture data directly.")
