import torch
from typing import Tuple

import chamfer_cpp

from chamfer_distance.Method.chamfer_triton import sided_triton


def sided_forward_func(
    name: str,
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, n = xyz1.shape[:2]

    dist1 = torch.zeros((batch_size, n), device=xyz1.device, dtype=xyz1.dtype)
    idx1 = torch.zeros((batch_size, n), device=xyz1.device, dtype=torch.int32)

    if name == "triton":
        dist1, idx1 = sided_triton(xyz1, xyz2)
    elif name == "cuda":
        chamfer_cpp.sided_forward_cuda(xyz1, xyz2, dist1, idx1)
    elif name == "cukd":
        chamfer_cpp.sided_forward_cukd(xyz1, xyz2, dist1, idx1)
    elif name == "faiss":
        idx1 = idx1.type(torch.int64)
        chamfer_cpp.sided_forward_faiss(xyz1, xyz2, dist1, idx1)
        idx1 = idx1.type(torch.int32)
    else:
        print("[ERROR][forwards::sided_forward_func]")
        print("\t func not found!")
        print("\t name:", name)
        exit()

    return dist1, idx1
