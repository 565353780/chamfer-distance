import torch
from typing import Tuple


def sided_backward(
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
    idx1: torch.Tensor,
    graddist1: torch.Tensor,
) -> torch.Tensor:
    batch_size, n_points = xyz1.shape[0], xyz1.shape[1]

    batch_indices = (
        torch.arange(batch_size, device=xyz1.device).view(-1, 1).expand(-1, n_points)
    )

    selected_xyz2 = xyz2[batch_indices, idx1]

    gradxyz1 = 2.0 * graddist1.unsqueeze(-1) * (xyz1 - selected_xyz2)

    return gradxyz1


def chamfer_backward(
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
    graddist1: torch.Tensor,
    graddist2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_idxs1 = idx1.unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
    valid_idxs2 = idx2.unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)

    d_dist1 = graddist1.unsqueeze(-1) * 2 * (xyz1 - torch.gather(xyz2, 1, valid_idxs1))
    d_dist2 = graddist2.unsqueeze(-1) * 2 * (xyz2 - torch.gather(xyz1, 1, valid_idxs2))

    gradxyz1 = torch.scatter_add(d_dist1, 1, valid_idxs2, -d_dist2)
    gradxyz2 = torch.scatter_add(d_dist2, 1, valid_idxs1, -d_dist1)
    return gradxyz1, gradxyz2
