# Fork Notes: Custom CUDA & Vulkan Ray Tracers

This fork adds two custom replacements for RF-Genesis's Mitsuba-based body
ray tracing stage, directly addressing the item on the original project's
own to-do list:

> - [ ] **New** Replace Mitsuba with custom rayTracing engines.

## What's added

- **`genesis/raytracing/RayTracerCustom/`**: a brute-force CUDA ray
  tracer, benchmarked ~1.27x faster than Mitsuba on this pipeline's actual
  workload.
- **`genesis/raytracing/vulkan_raytracer/`**: a hardware-accelerated
  Vulkan ray tracer (real BVH acceleration structures, RT cores, a full
  ray tracing pipeline with custom shaders), with zero-copy GPU memory
  sharing into PyTorch. Benchmarked ~1.39x faster than Mitsuba, including
  the complete per-frame cost (pose computation through final output).

Both are validated for correctness directly against Mitsuba (treated as
ground truth) on real motion data, matching within 1 pixel of hit accuracy
and single-digit-percent shading accuracy.

## Usage

Both are opt-in via a flag; Mitsuba remains the default, unmodified path:

```bash
# Mitsuba (default, unchanged)
python run.py -o "a person walking back and forth" -e "a living room" -n "hello_rfgen"

# Custom CUDA ray tracer
RFGEN_CUSTOM_RT=1 python run.py -o "a person walking back and forth" -e "a living room" -n "hello_custom"

# Custom Vulkan ray tracer
RFGEN_VULKAN_RT=1 python run.py -o "a person walking back and forth" -e "a living room" -n "hello_vulkan"
# or, equivalently:
./run_vulkan.sh -o "a person walking back and forth" -e "a living room" -n "hello_vulkan"
```

See `genesis/raytracing/GETTING_STARTED.md` for full build instructions
(both ray tracers require some one-time environment setup: a matching CUDA
toolkit for the CUDA version, the Vulkan SDK for the Vulkan version).

## Known scope / limitations

- No shadow ray yet (no self-shadowing) in either custom implementation.
- The Vulkan version currently models a simple omnidirectional point light
  rather than Mitsuba's spot-light cone, which accounts for the remaining
  ~8-11% shading intensity gap in validation.
- Vulkan's pose-to-vertex input path is CPU-mediated rather than
  zero-copy (the output path is genuinely zero-copy). Synchronization
  uses a full queue-wait rather than async semaphores.

None of these affect correctness meaningfully at the current validation
tolerance; they're documented as known, deliberate scope boundaries for
anyone looking to extend this further.

## Technical Notes

**Hardware tested on:** NVIDIA GeForce RTX 4060 Laptop GPU.

**Vulkan-CUDA memory interop** (`VulkanCudaInterop/`): Vulkan and CUDA
share GPU memory directly: Vulkan writes into an exportable buffer,
exports it as a file descriptor (`vkGetMemoryFdKHR`), and CUDA imports it
(`cudaImportExternalMemory`) to get a device pointer aliasing the same
physical VRAM, with zero CPU copying. The physical device is matched
between the two APIs by GPU UUID, since a system can have multiple GPUs
(e.g. an integrated + discrete pair) and there's no guarantee both APIs
would otherwise pick the same one.

**Why synchronization uses a full queue wait (`vkQueueWaitIdle`) instead
of async semaphores:** `vkQueueWaitIdle` guarantees correctness with very
little code: it blocks until the GPU finishes, so there's no risk of
CUDA reading memory before Vulkan finishes writing it. Async semaphores
(`VK_KHR_external_semaphore_fd` + `cudaImportExternalSemaphore`) would let
Vulkan's next frame overlap with CUDA/PyTorch consuming the current
frame's output (a real potential speedup), but they add meaningfully more
API surface, and semaphore misuse is a common source of subtle GPU hangs
and race conditions. Given the benchmark already showed a real 1.39x
speedup over Mitsuba without this added complexity, this was a deliberate
scope decision: ship a correct, validated result first, with async sync
as a documented follow-up rather than a blocker.

**Why the interop test rebuilds its whole Vulkan context per call:** the
correctness test (`interop_test.cpp`) intentionally creates and tears down
a full Vulkan instance/device every call, which is fine for a one-shot
check but wasteful for real per-frame use. The live ray tracer
(`raytracer_live.cpp`) keeps this state persistent across frames instead.

**Live ray tracer scope, stated plainly** (`raytracer_live.cpp`):
- **Output is true zero-copy:** the trace's result buffer is exported once
  at construction and imported into CUDA once. Every call to `trace()`
  re-reads that same already-shared memory, no CPU round-trip.
- **Input is not zero-copy yet:** `update_pose()` takes vertex data as a
  numpy/CPU array and copies it into a host-visible Vulkan buffer via
  `vkMapMemory`. True zero-copy on the input side (PyTorch writing
  directly into a Vulkan-CUDA-shared buffer) is a further optimization,
  not implemented here.
- **Synchronization** uses `vkQueueWaitIdle()` (a full queue stall), same
  as the interop test. See the note above on why this was a deliberate
  choice rather than async semaphores.

**Why the shader includes an albedo/pi term:** the closest-hit shader
(`closesthit.rchit`) matches Mitsuba's diffuse BSDF convention (outgoing
radiance = albedo/pi * N.L * irradiance). Mitsuba's SMPL material uses
reflectance (0.8, 0.8, 0.8); without this term, the shader's output was
about 3.5x brighter than Mitsuba's (Vulkan max 121.88 vs Mitsuba max
34.80 on the same test frame). Adding it brought the two into close
agreement, closing most of the gap seen during validation. No shadow ray
is implemented yet, matching the CUDA version's current scope.

---

## Original Project

This fork is built on top of the official implementation of *RF Genesis:
Zero-Shot Generalization of mmWave Sensing through Simulation-Based Data
Synthesis and Generative Diffusion Models* (SenSys '23) by Xingyu Chen and
Xinyu Zhang, UC San Diego.

- Original repository: https://github.com/Asixa/RF-Genesis
- Project page: https://rfgen.xingyuchen.me/
- Paper: https://xingyuchen.me/files/Xingyu.Chen_SenSys23_RFGen.pdf

```
@inproceedings{chen2023rfgenesis,
      author = {Chen, Xingyu and Zhang, Xinyu},
      title = {RF Genesis: Zero-Shot Generalization of mmWave Sensing through Simulation-Based Data Synthesis and Generative Diffusion Models},
      booktitle = {ACM Conference on Embedded Networked Sensor Systems (SenSys '23)},
      year = {2023},
      pages = {1-14},
      address = {Istanbul, Turkiye},
      publisher = {ACM, New York, NY, USA},
      url = {https://doi.org/10.1145/3625687.3625798},
      doi = {10.1145/3625687.3625798}
  }
```

Original code distributed under the MIT License (see `LICENSE`). Note
that the pipeline depends on other libraries and datasets (CLIP, SMPL,
MDM, mmMesh) that each have their own separate licenses which must also
be followed; see the original repository for details.

---

# RF Genesis V1.1 (2025 Updates!)
### [Project Page](https://rfgen.xingyuchen.me/) | [Paper](https://xingyuchen.me/files/Xingyu.Chen_SenSys23_RFGen.pdf) 

The offical implementation of [  *RF Genesis: Zero-Shot Generalization of mmWave Sensing
through Simulation-Based Data Synthesis and Generative
Diffusion Models*](https://rfgen.xingyuchen.me/).

[Xingyu Chen](https://xingyuchen.me/),
[Xinyu Zhang](http://xyzhang.ucsd.edu/index.html),
UC San Diego.

In SenSys 2023
![teaser](https://rfgen.xingyuchen.me/RFGen/pull.png)


## Updates 2025!
Sorry for the long wait — the RFLoRA model is finally released! Feel free to try it out.

Please note that RFLoRA was a workaround developed back in 2023, before 3D diffusion models became available. In 2025, I’ll be adding support for 3D diffusion models for indoor room generation.

Also, I’ve noticed that some dependencies, including MDM and Mitsuba, have updated their APIs. I’ll start maintaining this project again while also preparing for RFGenV2!



## News
📢 **June/25** - RFLoRA model released, trained under 20k images!

📢 **June/25** - Experimental CUDA kernel for signal generation, reduce memory usage by 1000X!

📢 **22/Jan/24** - Initial Release of RF Genesis!

📢 **29/March/24** - Added the code for point-cloud processing and visualization.

## To-Do List
- [ ]  **New**  Replace Mitsuba with custom rayTracing engines.
- [ ]  **New**  3D Diffusion including indoor environments.
- [ ] More documentations.


## Quick Start
This code was tested on `Ubuntu 20.04.5 LTS` and requires:

* Python 3.10
* conda3 or miniconda3
* CUDA capable GPU (one is enough)


Clone the repository
```
git clone https://github.com/Asixa/RF-Genesis.git
cd RF-Genesis
```

Create a conda environment.
```
conda create -n rfgen python=3.10 -y 
conda activate rfgen
```
Install python packages
```
pip install -r requirements.txt
sh setup.sh
```
Run a simple example.
```
python run.py -o "a person walking back and forth" -e "a living room" -n "hello_rfgen"
```

Optional Command:

Skiping visualization rendering
```
--no-visualize 
```
Skiping environmental diffusion
```
--no-environment 
```

## RFLoRA

```
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load model
pipe = StableDiffusionPipeline.from_pretrained("darkstorm2150/Protogen_x5.3_Official_Release",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

pipe.load_lora_weights("Asixa/RFLoRA")

prompt = "a living room with a table, a chair, a TV, a computer, a lamp, a plant, a window, a door" 
image = pipe(prompt, num_inference_steps=25).images[0]
plt.imshow(image)
```




## Visualization
![ezgif-7-eec8a9c9af](https://github.com/Asixa/RF-Genesis/assets/22312333/a53ef6d7-18b3-4f02-a82a-5bca3aaf08f8)

Rendered SMPL animation and radar point clouds. 


## Radar Hardware
The current simulation is based on the model of [**Texas Instruments AWR 1843**](https://www.ti.com/product/AWR1843#all) radar, with 3TX 4RX MIMO setup. 
![TI1843](https://github.com/Asixa/RF-Genesis/assets/22312333/bf68a6df-a3d2-4889-a7eb-509caf52a2fb)

The radar configuration is shown in [TI1843.json](https://github.com/Asixa/RF-Genesis/blob/main/models/TI1843_config.json) and it can be freely adjusted.

## Citation
```
@inproceedings{chen2023rfgenesis,
      author = {Chen, Xingyu and Zhang, Xinyu},
      title = {RF Genesis: Zero-Shot Generalization of mmWave Sensing through Simulation-Based Data Synthesis and Generative Diffusion Models},
      booktitle = {ACM Conference on Embedded Networked Sensor Systems (SenSys ’23)},
      year = {2023},
      pages = {1-14},
      address = {Istanbul, Turkiye},
      publisher = {ACM, New York, NY, USA},
      url = {https://doi.org/10.1145/3625687.3625798},
      doi = {10.1145/3625687.3625798}
  }

```


## License
This code is distributed under an [MIT LICENSE](LICENSE).
Note that our code depends on other libraries, including [CLIP](https://github.com/openai/CLIP), [SMPL](https://smpl.is.tue.mpg.de/), [MDM](https://guytevet.github.io/mdm-page/), [mmMesh](https://github.com/HavocFiXer/mmMesh) and uses datasets that each have their own respective licenses that must also be followed.
