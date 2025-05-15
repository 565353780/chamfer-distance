import torch
from typing import Tuple

import chamfer_cpp


def sided_forward_cuda(
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, n = xyz1.shape[0], xyz1.shape[1]

    dist1 = torch.zeros((batch_size, n), device=xyz1.device, dtype=xyz1.dtype)
    idx1 = torch.zeros((batch_size, n), device=xyz1.device, dtype=torch.int32)

    chamfer_cpp.sided_cuda_forward(xyz1, xyz2, dist1, idx1)

    return dist1, idx1
