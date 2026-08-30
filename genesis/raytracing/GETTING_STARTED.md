# Getting Started: RF-Genesis Custom Ray Tracers (CUDA + Vulkan)

This is a step-by-step build guide. For the full story of *why* each step
exists (bugs hit, decisions made), see `RF-Genesis_Notes.md` and
`vulkan_raytracer/PLAN.md`: this file is just the commands.

**Prerequisite:** RF-Genesis's own base pipeline already cloned and working
(per its own README), with the `rfgen` conda environment set up and able
to run:
```bash
python run.py -o "a person walking back and forth" -e "a living room" -n "hello_rfgen"
```
If that doesn't work yet, fix that first; everything below builds on top
of it.

---

## Part 1: CUDA Ray Tracer

**Files needed** in `genesis/raytracing/RayTracerCustom/`:
`raytracer_kernel.cu`, `setup.py`, `raytracer.py`, `benchmark.py`

**One-time environment setup:**
```bash
conda activate rfgen

# Full CUDA toolkit (not just the driver); match the version PyTorch was built against:
python -c "import torch; print(torch.version.cuda)"
conda install -c nvidia/label/cuda-<VERSION> cuda-toolkit   # use the version printed above

# Make CUDA_HOME persist automatically on future `conda activate rfgen`:
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export CUDA_HOME=$CONDA_PREFIX' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda deactivate && conda activate rfgen   # reload with the new env var
```

**Build:**
```bash
cd genesis/raytracing/RayTracerCustom
pip install -e . --no-build-isolation
```

**Wire into the pipeline**: in `genesis/raytracing/pathtracer.py`, add near the top:
```python
import os
USE_CUSTOM_RAYTRACER = os.environ.get("RFGEN_CUSTOM_RT", "0") == "1"
```
and after the `class RayTracer:` block (Mitsuba's), before `def get_deafult_scene`:
```python
if USE_CUSTOM_RAYTRACER:
    from .RayTracerCustom.raytracer import RayTracer
```

**Run it:**
```bash
cd ~/RF-Genesis
RFGEN_CUSTOM_RT=1 python run.py -o "a person walking back and forth" -e "a living room" -n "hello_custom"
```

**Benchmark against Mitsuba:**
```bash
cd genesis/raytracing/RayTracerCustom
python benchmark.py ../../../output/hello_custom/obj_diff.npz
```

---

## Part 2: Vulkan Ray Tracer (standalone)

**One-time system setup:**
```bash
sudo apt install libvulkan-dev vulkan-tools glslang-tools spirv-tools

# Confirm hardware ray tracing support:
vulkaninfo | grep -iE "ray_tracing|acceleration_structure"
```

**Files needed** in `genesis/raytracing/vulkan_raytracer/`:
`CMakeLists.txt`, `src/main.cpp`, `shaders/raygen.rgen`,
`shaders/closesthit.rchit`, `shaders/miss.rmiss`

**Build** (must NOT be inside the `rfgen` conda env; its compilers
conflict with system library search paths):
```bash
cd genesis/raytracing/vulkan_raytracer
conda deactivate
mkdir build && cd build
cmake ..
cmake --build .
conda activate rfgen   # back in for everything else below
```

**Export a test mesh and run standalone (optional sanity check):**
```bash
cd genesis/raytracing/vulkan_raytracer
python export_mesh.py
cd build && ./vulkan_raytracer && cd ..
python view_output.py
```

**Validate against Mitsuba (optional):**
```bash
python compare_mitsuba.py --motion ../../../output/hello_rfgen/obj_diff.npz --frame 0
```

---

## Part 3: Vulkan-CUDA-PyTorch Live Integration

**Files needed** in `genesis/raytracing/vulkan_raytracer/VulkanCudaInterop/`:
`setup.py`, `interop_test.cpp`, `raytracer_live.cpp`, `live_raytracer.py`,
`test_interop.py`, `validate_live.py`, `run_live_only.py`,
`benchmark_live.py`, `__init__.py`

**One-time local header copy** (avoids a real conda-vs-system header
conflict; see notes if curious):
```bash
cd genesis/raytracing/vulkan_raytracer/VulkanCudaInterop
mkdir -p vulkan_headers
cp -r /usr/include/vulkan vulkan_headers/
cp -r /usr/include/vk_video vulkan_headers/
```

**Build:**
```bash
pip install -e . --no-build-isolation
```

**Test the core memory-sharing mechanism:**
```bash
python test_interop.py
```

**Run just the Vulkan tracer standalone** (no Mitsuba loaded, lighter weight than the full comparison below):
```bash
python run_live_only.py
python run_live_only.py --motion ../../../../output/hello_rfgen/obj_diff.npz --frame 0
```

**Validate the full live pipeline against Mitsuba:**
```bash
python validate_live.py --motion ../../../../output/hello_rfgen/obj_diff.npz --frame 0
```

**Benchmark the full live pipeline:**
```bash
python benchmark_live.py ../../../../output/hello_rfgen/obj_diff.npz
```

**Wire into the pipeline:**

1. Add `__init__.py` (empty file) in both:
   - `genesis/raytracing/vulkan_raytracer/`
   - `genesis/raytracing/vulkan_raytracer/VulkanCudaInterop/`

2. In `genesis/raytracing/pathtracer.py`, add near the top (alongside `USE_CUSTOM_RAYTRACER`):
```python
USE_VULKAN_RAYTRACER = os.environ.get("RFGEN_VULKAN_RT", "0") == "1"
```
and alongside the CUDA import swap:
```python
if USE_VULKAN_RAYTRACER:
    from .vulkan_raytracer.VulkanCudaInterop.live_raytracer import RayTracer
```

3. (Optional) create `run_vulkan.sh` at the RF-Genesis root:
```bash
#!/bin/bash
RFGEN_VULKAN_RT=1 python run.py "$@"
```
```bash
chmod +x run_vulkan.sh
```

---

## Running the Full Pipeline (all three options)

```bash
cd ~/RF-Genesis

# Mitsuba (default, unchanged)
python run.py -o "a person walking back and forth" -e "a living room" -n "hello_rfgen"

# CUDA ray tracer
RFGEN_CUSTOM_RT=1 python run.py -o "a person walking back and forth" -e "a living room" -n "hello_custom"

# Vulkan ray tracer
RFGEN_VULKAN_RT=1 python run.py -o "a person walking back and forth" -e "a living room" -n "hello_vulkan"
# or:
./run_vulkan.sh -o "a person walking back and forth" -e "a living room" -n "hello_vulkan"
```

Output for each lands in `output/<name>/output.mp4`.

---

## Troubleshooting Quick Reference

| Symptom | Likely cause |
|---|---|
| `CUDA_HOME environment variable is not set` | Full CUDA toolkit not installed, or env var not exported in this shell |
| `mismatches the version that was used to compile PyTorch` | Toolkit version doesn't match `torch.version.cuda`; install the matching version |
| `fatal error: bits/timesize.h` or `math.h` errors mixing conda + system headers | Don't add `/usr/include` as a plain include path inside a conda build; copy only the specific needed headers into a project-local folder instead |
| `Could NOT find Vulkan` in `cmake` | Not building from inside a plain shell (`conda deactivate` first); conda's compiler search path conflicts with system library discovery |
| Edited a file but behavior didn't change | Confirm the edit actually landed: `grep` for a string unique to the new version. If empty, the file wasn't really replaced; re-download/re-copy it |
| Shader edit has no effect after rebuild | Do a full clean rebuild (`rm -rf build/*`, `cmake ..` again). This machine has a recurring clock-skew issue that can make `make` skip recompiling changed files |
| Segfault with no clear cause | Add `fprintf(stderr, ...); fflush(stderr);` checkpoints through the suspect code and bisect. See `PLAN.md` for a full worked example |
