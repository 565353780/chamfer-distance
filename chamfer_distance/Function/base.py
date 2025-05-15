import torch

from chamfer_distance.Method.backwards import sided_backward, chamfer_backward


class BaseSidedFunction(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_dist1, grad_idx1):
        xyz1, xyz2, idx1 = ctx.saved_tensors
        grad_xyz1 = sided_backward(xyz1, xyz2, idx1, grad_dist1)
        return grad_xyz1


class BaseChamferFunction(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        grad_xyz1, grad_xyz2 = chamfer_backward(
            xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2
        )
        return grad_xyz1, grad_xyz2, None, None
