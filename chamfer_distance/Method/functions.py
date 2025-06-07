import torch

from chamfer_distance.Method.forwards import sided_forward_func


class BaseFunction(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        valid_idxs1 = idx1.unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
        valid_idxs2 = idx2.unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)

        d_dist1 = (
            grad_dist1.unsqueeze(-1) * 2 * (xyz1 - torch.gather(xyz2, 1, valid_idxs1))
        )
        d_dist2 = (
            grad_dist2.unsqueeze(-1) * 2 * (xyz2 - torch.gather(xyz1, 1, valid_idxs2))
        )

        grad_xyz1 = None
        grad_xyz2 = None

        if ctx.needs_input_grad[0]:
            grad_xyz1 = torch.scatter_add(d_dist1, 1, valid_idxs2, -d_dist2)
        if ctx.needs_input_grad[1]:
            grad_xyz2 = torch.scatter_add(d_dist2, 1, valid_idxs1, -d_dist1)

        return grad_xyz1, grad_xyz2, None, None


class ChamferFunction(BaseFunction):
    @staticmethod
    def forward(ctx, xyz1, xyz2, sided_forward_func_name: str = "cuda"):
        dist1, idx1 = sided_forward_func(sided_forward_func_name, xyz1, xyz2)
        dist2, idx2 = sided_forward_func(sided_forward_func_name, xyz2, xyz1)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2


class SearcherFunction(BaseFunction):
    @staticmethod
    def forward(ctx, xyz1, xyz2, searcher, sided_forward_func_name: str = "cuda"):
        dist1, idx1 = searcher.query(xyz1)
        dist2, idx2 = sided_forward_func(sided_forward_func_name, xyz2, xyz1)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2, idx1, idx2
