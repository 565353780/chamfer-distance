import torch
from typing import List

from chamfer_distance.Method.ops import kd_closest_query
from chamfer_distance.Method.chamfer_triton import chamfer_distance_backward, chamfer_distance_setup_context


@torch.library.custom_op("chamfer::triton_kd", mutates_args=())
def chamfer_triton_kd(xyz1: torch.Tensor,
                        xyz2: torch.Tensor) -> List[torch.Tensor]:
    dist1, idx1 = kd_closest_query(xyz1, xyz2)
    dist2, idx2 = kd_closest_query(xyz2, xyz1)
    return dist1, idx1, dist2, idx2


chamfer_triton_kd.register_autograd(
    chamfer_distance_backward, setup_context=chamfer_distance_setup_context)
