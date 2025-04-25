import os
import re
import torch
import shutil
import subprocess
from pkg_resources import parse_version


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

def getCppStandard() -> int:
    cuda_version = _maybe_nvcc_version()
    if cuda_version is None:
        return 17

    if cuda_version >= parse_version("11.0"):
        return 17

    return 14

def getSupportedComputeCapabilities() -> list:
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
        print(
            "Unknown compute capability. Specify the target compute capabilities in the TCNN_CUDA_ARCHITECTURES environment variable or install PyTorch with the CUDA backend to detect it automatically."
        )

        compute_capabilities = []

    cuda_version = _maybe_nvcc_version()
    if cuda_version is None:
        return []

    print(f"Detected CUDA version {cuda_version}")
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

    return compute_capabilities
