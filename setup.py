import os
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

from chamfer_distance.Method.setup import (
    getCppStandard,
    getSupportedComputeCapabilities,
)


cpp_standard = getCppStandard()
compute_capabilities = getSupportedComputeCapabilities()

print(f"Targeting C++ standard {cpp_standard}")

chamfer_root_path = os.getcwd() + "/chamfer_distance/Cpp/"
chamfer_lib_path = os.getcwd() + "/chamfer_distance/Lib/"
chamfer_src_path = chamfer_root_path + "src/"
chamfer_sources = glob.glob(chamfer_src_path + "*.cpp")
chamfer_include_dirs = [
    chamfer_root_path + "include",
    chamfer_lib_path + "cudaKDTree",
    chamfer_lib_path + "tiny-cuda-nn/include",
    chamfer_lib_path + "tiny-cuda-nn/dependencies",
    chamfer_lib_path + "tiny-cuda-nn/dependencies/fmt/include",
]

chamfer_extra_compile_args = [
    "-O3",
    f"-std=c++{cpp_standard}",
    "-DCMAKE_BUILD_TYPE Release",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
]

if len(compute_capabilities) > 0:
    # os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9"

    compute_capability = compute_capabilities[0]

    base_nvcc_flags = [
        f"-std=c++{cpp_standard}",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        # The following definitions must be undefined
        # since TCNN requires half-precision operation.
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-Xcompiler=-Wno-float-conversion",
        "-Xcompiler=-fno-strict-aliasing",
    ]

    nvcc_flags = base_nvcc_flags + [
        f"-gencode=arch=compute_{compute_capability},code={code}_{compute_capability}"
        for code in ["compute", "sm"]
    ]

    chamfer_sources += glob.glob(chamfer_src_path + "*.cu")

    extra_compile_args = {
        "cxx": chamfer_extra_compile_args
        + [
            "-DUSE_CUDA",
            "-DTCNN_NO_NETWORKS",
            f"-DTCNN_MIN_GPU_ARCH={compute_capability}",
            "-DTORCH_USE_CUDA_DSA",
        ],
        "nvcc": nvcc_flags
        + [
            "-O3",
            "-Xfatbin",
            "-compress-all",
            "-DUSE_CUDA",
            "-DTCNN_NO_NETWORKS",
            f"-DTCNN_MIN_GPU_ARCH={compute_capability}",
            "-DTORCH_USE_CUDA_DSA",
        ],
    }

    chamfer_module = CUDAExtension(
        name="chamfer_cpp",
        sources=chamfer_sources,
        include_dirs=chamfer_include_dirs,
        extra_compile_args=extra_compile_args,
        libraries=["cuda", "cudart"],
    )

else:
    chamfer_module = CppExtension(
        name="chamfer_cpp",
        sources=chamfer_sources,
        include_dirs=chamfer_include_dirs,
        extra_compile_args=chamfer_extra_compile_args,
    )

setup(
    name="Chamfer",
    version="1.0.0",
    author="Changhao Li, Ruichen Zheng",
    packages=find_packages(),
    ext_modules=[chamfer_module],
    zip_safe=False,
    python_requires=">=3.7",
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
