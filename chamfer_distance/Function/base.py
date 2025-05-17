import torch

from chamfer_distance.Method.backwards import sided_backward, chamfer_backward


class BaseSidedFunction(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_dist1, grad_idx1):
        xyz1, xyz2, idx1 = ctx.saved_tensors
        grad_xyz1 = sided_backward(xyz1, xyz2, idx1, grad_dist1)
        return grad_xyz1, None


class BaseChamferFunction(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        grad_xyz1, grad_xyz2 = chamfer_backward(
            xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2
        )
        return grad_xyz1, grad_xyz2, None, None


class BaseSearcherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, searcher, xyz1, xyz2):
        dist1, idx1 = searcher.query(xyz1)

        ctx.save_for_backward(xyz1, xyz2, idx1)

        return dist1, idx1

    @staticmethod
    def backward(ctx, grad_dist1, grad_idx1):
        xyz1, xyz2, idx1 = ctx.saved_tensors

        selected_xyz2 = xyz2[idx1]

        grad_xyz1 = 2.0 * grad_dist1.unsqueeze(-1) * (xyz1 - selected_xyz2)

        return None, grad_xyz1, None
