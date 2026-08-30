from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_raytracer',
    ext_modules=[
        CUDAExtension(
            name='custom_raytracer_cuda',
            sources=['raytracer_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
