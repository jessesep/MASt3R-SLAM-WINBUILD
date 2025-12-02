from pathlib import Path
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
has_cuda = torch.cuda.is_available()

# Convert to absolute paths with proper native separators for Windows
include_dirs = [
    str(Path(ROOT) / "mast3r_slam" / "backend" / "include"),
    str(Path(ROOT) / "thirdparty" / "eigen"),
]

sources = [
    "mast3r_slam/backend/src/gn.cpp",
]
extra_compile_args = {
    "cores": ["j8"],
    "cxx": ["-O3"],
}

if has_cuda:
    from torch.utils.cpp_extension import CUDAExtension
    import subprocess

    # Get CUDA toolkit version (not PyTorch's runtime version)
    # PyTorch cu128 reports 12.8 but system toolkit might be 12.4
    try:
        nvcc_version = subprocess.check_output(['nvcc', '--version'], text=True)
        for line in nvcc_version.split('\n'):
            if 'release' in line:
                version_str = line.split('release')[1].split(',')[0].strip()
                cuda_major, cuda_minor = map(int, version_str.split('.')[:2])
                break
    except:
        # Fallback to PyTorch version if nvcc unavailable
        cuda_version = torch.version.cuda
        cuda_major, cuda_minor = map(int, cuda_version.split(".")[:2])

    sources.append("mast3r_slam/backend/src/gn_kernels.cu")
    sources.append("mast3r_slam/backend/src/matching_kernels.cu")
    extra_compile_args["nvcc"] = [
        "-O3",
        "-Xcudafe", "--diag_suppress=20014",  # Suppress __host__/__device__ warnings for Eigen
        "-Xcudafe", "--diag_suppress=177",     # Suppress unreferenced label warnings
        "-gencode=arch=compute_60,code=sm_60",
        "-gencode=arch=compute_61,code=sm_61",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
    ]

    if (cuda_major, cuda_minor) >= (11, 8):
        extra_compile_args["nvcc"].append("-gencode=arch=compute_90,code=sm_90")

    if (cuda_major, cuda_minor) >= (12, 8):
        extra_compile_args["nvcc"].append("-gencode=arch=compute_120,code=sm_120")
    ext_modules = [
        CUDAExtension(
            "mast3r_slam_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]
else:
    print("CUDA not found, cannot compile backend!")

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
