import torch
from typing import List

from chamfer_distance.Method.nm_dist import nm_dist


@torch.library.custom_op("chamfer::triton", mutates_args=())
def chamfer_triton(xyz1: torch.Tensor,
                     xyz2: torch.Tensor) -> List[torch.Tensor]:
    dist1_list = []
    dist2_list = []
    idx1_list = []
    idx2_list = []

    for i in range(xyz1.shape[0]):
        dist1, idx1 = nm_dist(xyz1[i], xyz2[i])
        dist2, idx2 = nm_dist(xyz2[i], xyz1[i])
        dist1_list.append(dist1)
        dist2_list.append(dist2)
        idx1_list.append(idx1)
        idx2_list.append(idx2)

    dists1 = torch.vstack(dist1_list)
    dists2 = torch.vstack(dist2_list)
    idxs1 = torch.vstack(idx1_list)
    idxs2 = torch.vstack(idx2_list)

    return dists1, dists2, idxs1, idxs2

@chamfer_triton.register_fake
def _(xyz1: torch.Tensor, xyz2: torch.Tensor):
    B, N = xyz1.shape[:2]
    M = xyz2.shape[1]
    return xyz1.new_empty((B, N, )), \
    xyz2.new_empty((B, M, )), \
    xyz1.new_empty((B, N, ), dtype=torch.int32), \
    xyz2.new_empty((B, M, ), dtype=torch.int32)


def chamfer_distance_setup_context(ctx, inputs, output):
    xyz1, xyz2 = inputs
    _, _, idxs1, idxs2 = output
    ctx.save_for_backward(xyz1, xyz2, idxs1, idxs2)


def chamfer_distance_backward(ctx, grad_out):
    xyz1, xyz2, idxs1, idxs2 = ctx.saved_tensors
    grad_dist1, grad_dist2, _, _ = grad_out

    valid_idxs1 = idxs1.unsqueeze(-1).expand(-1, -1, 3).type(torch.int64)
    valid_idxs2 = idxs2.unsqueeze(-1).expand(-1, -1, 3).type(torch.int64)

    d_dist1 = grad_dist1.unsqueeze(-1) * 2 * (xyz1 - torch.gather(xyz2, 1, valid_idxs1))
    d_dist2 = grad_dist2.unsqueeze(-1) * 2 * (xyz2 - torch.gather(xyz1, 1, valid_idxs2))

    grad_xyz1 = torch.scatter_add(d_dist1, 1,
                                  valid_idxs2, -d_dist2)
    grad_xyz2 = torch.scatter_add(d_dist2, 1,
                                  valid_idxs1, -d_dist1)
    return grad_xyz1, grad_xyz2


chamfer_triton.register_autograd(
    chamfer_distance_backward, setup_context=chamfer_distance_setup_context)
