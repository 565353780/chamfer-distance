import torch

__all__ = ["kd_closest_query", "crude_nn"]

def kd_closest_query(xyz1: torch.Tensor, xyz2: torch.Tensor) -> list[torch.Tensor]:
    return torch.ops.triton_chamfer.kd_closest_query.default(xyz1, xyz2)

def crude_nn(xyz1: torch.Tensor, xyz2: torch.Tensor) -> torch.Tensor:
    return torch.ops.triton_chamfer.crude_nn.default(xyz1, xyz2)