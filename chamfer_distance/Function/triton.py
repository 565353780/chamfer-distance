from chamfer_distance.Method.forwards import sided_forward_func
from chamfer_distance.Function.base import BaseSidedFunction, BaseChamferFunction


class SidedTriton(BaseSidedFunction):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, idx1 = sided_forward_func("triton", xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1)
        return dist1, idx1


class ChamferTriton(BaseChamferFunction):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, idx1 = sided_forward_func("triton", xyz1, xyz2)
        dist2, idx2 = sided_forward_func("triton", xyz2, xyz1)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2
