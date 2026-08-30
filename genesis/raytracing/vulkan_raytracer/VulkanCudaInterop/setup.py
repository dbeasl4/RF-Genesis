from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vulkan_cuda_interop',
    ext_modules=[
        # Part 1
        CUDAExtension(
            name='vulkan_cuda_interop',
            sources=['interop_test.cpp'],
            include_dirs=['vulkan_headers'],
            library_dirs=['/usr/lib/x86_64-linux-gnu'],
            libraries=['vulkan'],
        ),
        # Part 2
        CUDAExtension(
            name='vulkan_raytracer_live',
            sources=['raytracer_live.cpp'],
            include_dirs=['vulkan_headers'],
            library_dirs=['/usr/lib/x86_64-linux-gnu'],
            libraries=['vulkan'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)

# Build:
#   pip install -e . --no-build-isolation
#
# (same --no-build-isolation reason as the earlier CUDA ray tracer build:
# pip's isolated build sandbox doesn't have torch available for setup.py's
# own imports)