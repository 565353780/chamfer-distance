import os

import re
from setuptools import setup
from pkg_resources import parse_version
import subprocess
import shutil
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

package_name = "triton_chamfer"

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = SCRIPT_DIR


def min_supported_compute_capability(cuda_version):
    if cuda_version >= parse_version("12.0"):
        return 50
    else:
        return 20


def max_supported_compute_capability(cuda_version):
    if cuda_version < parse_version("11.0"):
        return 75
    elif cuda_version < parse_version("11.1"):
        return 80
    elif cuda_version < parse_version("11.8"):
        return 86
    elif cuda_version < parse_version("12.8"):
        return 90
    else:
        return 120


if "TCNN_CUDA_ARCHITECTURES" in os.environ and os.environ[
        "TCNN_CUDA_ARCHITECTURES"]:
    compute_capabilities = [
        int(x) for x in os.environ["TCNN_CUDA_ARCHITECTURES"].replace(
            ";", ",").split(",")
    ]
    print(
        f"Obtained compute capabilities {compute_capabilities} from environment variable TCNN_CUDA_ARCHITECTURES"
    )
elif torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    compute_capabilities = [major * 10 + minor]
    print(
        f"Obtained compute capability {compute_capabilities[0]} from PyTorch")
else:
    raise EnvironmentError(
        "Unknown compute capability. Specify the target compute capabilities in the TCNN_CUDA_ARCHITECTURES environment variable or install PyTorch with the CUDA backend to detect it automatically."
    )

cpp_standard = 14


# Get CUDA version and make sure the targeted compute capability is compatible
def _maybe_find_nvcc():
    # Try PATH first
    maybe_nvcc = shutil.which("nvcc")

    if maybe_nvcc is not None:
        return maybe_nvcc

    # Then try CUDA_HOME from torch (cpp_extension.CUDA_HOME is undocumented, which is why we only use
    # it as a fallback)
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        return None

    if not CUDA_HOME:
        return None

    return os.path.join(CUDA_HOME, "bin", "nvcc")


def _maybe_nvcc_version():
    maybe_nvcc = _maybe_find_nvcc()

    if maybe_nvcc is None:
        return None

    nvcc_version_result = subprocess.run(
        [maybe_nvcc, "--version"],
        text=True,
        check=False,
        stdout=subprocess.PIPE,
    )

    if nvcc_version_result.returncode != 0:
        return None

    cuda_version = re.search(r"release (\S+),", nvcc_version_result.stdout)

    if not cuda_version:
        return None

    return parse_version(cuda_version.group(1))


cuda_version = _maybe_nvcc_version()
if cuda_version is not None:
    print(f"Detected CUDA version {cuda_version}")
    if cuda_version >= parse_version("11.0"):
        cpp_standard = 17

    supported_compute_capabilities = [
        cc for cc in compute_capabilities
        if cc >= min_supported_compute_capability(cuda_version)
        and cc <= max_supported_compute_capability(cuda_version)
    ]

    if not supported_compute_capabilities:
        supported_compute_capabilities = [
            max_supported_compute_capability(cuda_version)
        ]

    if supported_compute_capabilities != compute_capabilities:
        print(
            f"WARNING: Compute capabilities {compute_capabilities} are not all supported by the installed CUDA version {cuda_version}. Targeting {supported_compute_capabilities} instead."
        )
        compute_capabilities = supported_compute_capabilities

min_compute_capability = min(compute_capabilities)

print(f"Targeting C++ standard {cpp_standard}")

base_nvcc_flags = [
    f"-std=c++{cpp_standard}",
    "--extended-lambda",
    "--expt-relaxed-constexpr",
    # The following definitions must be undefined
    # since TCNN requires half-precision operation.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

if os.name == "posix":
    base_cflags = [f"-std=c++{cpp_standard}"]
    base_nvcc_flags += [
        "-Xcompiler=-Wno-float-conversion",
        "-Xcompiler=-fno-strict-aliasing",
    ]

base_definitions = ["-DTCNN_NO_NETWORKS"]

compute_capability = compute_capabilities[0]
nvcc_flags = base_nvcc_flags + [
    f"-gencode=arch=compute_{compute_capability},code={code}_{compute_capability}"
    for code in ["compute", "sm"]
]
definitions = base_definitions + [f"-DTCNN_MIN_GPU_ARCH={compute_capability}"]

nvcc_flags = nvcc_flags + definitions
cflags = base_cflags + definitions

setup(
    name=package_name,
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name=f"{package_name}._C",
            sources=glob.glob(f"{package_name}/src/*.cu"),
            include_dirs=[
                os.path.abspath(f"{package_name}/include"),
                os.path.abspath(f"{package_name}/third_party/cudaKDTree"),
                os.path.abspath(
                    f"{package_name}/third_party/tiny-cuda-nn/include"),
                os.path.abspath(
                    f"{package_name}/third_party/tiny-cuda-nn/dependencies"),
                os.path.abspath(
                    f"{package_name}/third_party/tiny-cuda-nn/dependencies/fmt/include"
                )
            ],
            extra_compile_args={
                "cxx": cflags,
                "nvcc": nvcc_flags
            },
            libraries=["cuda"],
        )
    ],
    packages=[package_name],
    zip_safe=False,
    python_requires=">=3.7",
    cmdclass={"build_ext": BuildExtension})
