import torch
from typing import Tuple

import chamfer_cpp

from chamfer_distance.Method.chamfer_torch import chamfer_torch
from chamfer_distance.Method.chamfer_triton import sided_triton


def sided_forward_func(
    name: str,
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, n = xyz1.shape[:2]

    # 计算每个子张量的最大大小b，使得b*M*3或b*N*3接近INT_MAX
    max_size = int(torch.iinfo(torch.int32).max / 3)
    sub_batch_size = max_size // max(n, xyz2.shape[1])

    # 将xyz1和xyz2按照sub_batch_size划分为局部张量
    local_dist1_list = []
    local_idx1_list = []

    for i in range(0, batch_size, sub_batch_size):
        end_idx = min(i + sub_batch_size, batch_size)
        local_xyz1 = xyz1[i:end_idx]
        local_xyz2 = xyz2[i:end_idx]

        local_dist1 = torch.zeros(
            (end_idx - i, n), device=local_xyz1.device, dtype=local_xyz1.dtype
        )
        local_idx1 = torch.zeros(
            (end_idx - i, n), device=local_xyz1.device, dtype=torch.int32
        )

        if name == "torch":
            local_dist1, _, local_idx1, _ = chamfer_torch(local_xyz1, local_xyz2)
        elif name == "triton":
            local_dist1, local_idx1 = sided_triton(local_xyz1, local_xyz2)
        elif name == "cuda":
            chamfer_cpp.sided_forward_cuda(
                local_xyz1, local_xyz2, local_dist1, local_idx1
            )
        elif name == "cukd":
            chamfer_cpp.sided_forward_cukd(
                local_xyz1, local_xyz2, local_dist1, local_idx1
            )
        else:
            print("[ERROR][forwards::sided_forward_func]")
            print("\t func not found!")
            print("\t name:", name)
            exit()

        local_dist1_list.append(local_dist1)
        local_idx1_list.append(local_idx1)

    dist1 = torch.cat(local_dist1_list)
    idx1 = torch.cat(local_idx1_list)

    return dist1, idx1
