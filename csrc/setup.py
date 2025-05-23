from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

__version__ = "0.0.1"

# Find all .cpp and .cu files in src directory
sources = []
for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith('.cpp') or file.endswith('.cu'):
            sources.append(os.path.join(root, file))

# Make sure we're including the new memory_manager.cpp file
if 'src/memory_manager.cpp' not in sources:
    sources.append('src/memory_manager.cpp')

setup(
    name="swiftllm_c",
    version=__version__,
    author="",
    url="",
    description="Some C++/CUDA sources.",
    long_description="",
    ext_modules=[
        CUDAExtension(
            name='swiftllm_c',
            sources=sources,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--use_fast_math', '-std=c++17']
            },
            include_dirs=['src']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
    python_requires=">=3.9",
)
