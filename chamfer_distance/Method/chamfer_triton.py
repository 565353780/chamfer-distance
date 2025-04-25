import torch
from typing import List

from chamfer_distance.Method.nm_dist import nm_dist


@torch.library.custom_op("chamfer::triton", mutates_args=())
def chamfer_triton(xyz1: torch.Tensor,
                     xyz2: torch.Tensor) -> List[torch.Tensor]:
    dist1, idx1 = nm_dist(xyz1, xyz2)
    dist2, idx2 = nm_dist(xyz2, xyz1)
    return dist1, idx1, dist2, idx2

@chamfer_triton.register_fake
def _(xyz1: torch.Tensor, xyz2: torch.Tensor):
    N = xyz1.shape[0]
    M = xyz2.shape[0]
    return xyz1.new_empty((N, )), xyz1.new_empty(
        (N, ), dtype=torch.int32), xyz2.new_empty((M, )), xyz2.new_empty(
            (M, ), dtype=torch.int32)


def chamfer_distance_setup_context(ctx, inputs, output):
    xyz1, xyz2 = inputs
    _, idx1, _, idx2 = output
    ctx.save_for_backward(xyz1, idx1, xyz2, idx2)


def chamfer_distance_backward(ctx, grad_out):
    xyz1, idx1, xyz2, idx2 = ctx.saved_tensors
    grad_dist1, _, grad_dist2, _ = grad_out

    d_dist1 = grad_dist1[:, None] * 2 * (xyz1 - xyz2[idx1])
    d_dist2 = grad_dist2[:, None] * 2 * (xyz2 - xyz1[idx2])

    grad_xyz1 = torch.scatter_add(d_dist1, 0,
                                  idx2[:, None].expand(-1, 3).long(), -d_dist2)
    grad_xyz2 = torch.scatter_add(d_dist2, 0,
                                  idx1[:, None].expand(-1, 3).long(), -d_dist1)
    return grad_xyz1, grad_xyz2


chamfer_triton.register_autograd(
    chamfer_distance_backward, setup_context=chamfer_distance_setup_context)
