import torch


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
) -> torch.Tensor:
    batch_size, n_points = xyz1.shape[0], xyz1.shape[1]

    batch_indices = (
        torch.arange(batch_size, device=xyz1.device).view(-1, 1).expand(-1, n_points)
    )

    selected_xyz2 = xyz2[batch_indices, idx1]

    gradxyz1 = 2.0 * graddist1.unsqueeze(-1) * (xyz1 - selected_xyz2)

    return gradxyz1
