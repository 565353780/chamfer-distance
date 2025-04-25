import torch
from chamfer_cpp import kd_closest_query_cuda, crude_nn_cuda

def kd_closest_query(xyz1: torch.Tensor, xyz2: torch.Tensor) -> list[torch.Tensor]:
    return kd_closest_query_cuda(xyz1, xyz2)

def crude_nn(xyz1: torch.Tensor, xyz2: torch.Tensor) -> torch.Tensor:
    return crude_nn_cuda(xyz1, xyz2)
