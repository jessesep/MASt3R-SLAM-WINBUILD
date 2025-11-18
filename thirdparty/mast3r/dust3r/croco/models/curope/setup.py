# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch import cuda
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess

# Get CUDA toolkit version (not PyTorch's CUDA runtime version)
# PyTorch cu128 reports 12.8, but system nvcc might be 12.4
try:
    nvcc_version = subprocess.check_output(['nvcc', '--version'], text=True)
    # Extract version like "release 12.4, V12.4.131"
    for line in nvcc_version.split('\n'):
        if 'release' in line:
            version_str = line.split('release')[1].split(',')[0].strip()
            cuda_major, cuda_minor = map(int, version_str.split('.')[:2])
            break
    else:
        raise ValueError("Could not parse nvcc version")
except:
    # Fallback to PyTorch's CUDA version if nvcc not available
    cuda_version = torch.version.cuda
    cuda_major, cuda_minor = map(int, cuda_version.split(".")[:2])

# Base architectures supported by CUDA 12.4
all_cuda_archs = [
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_75,code=sm_75',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86'
]

# Add sm_90 (Hopper) for CUDA 11.8+
if (cuda_major, cuda_minor) >= (11, 8):
    all_cuda_archs.extend(['-gencode', 'arch=compute_90,code=sm_90'])

# Add sm_120 (Blackwell) only for CUDA 12.8+ **TOOLKIT**
# For CUDA 12.4 toolkit, we compile for sm_90 and rely on forward compatibility at runtime
if (cuda_major, cuda_minor) >= (12, 8):
    all_cuda_archs.extend(['-gencode', 'arch=compute_120,code=sm_120'])

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=['-O3','--ptxas-options=-v',"--use_fast_math"]+all_cuda_archs, 
                    cxx=['-O3'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
