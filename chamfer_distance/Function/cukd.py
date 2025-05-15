from chamfer_distance.Method.forwards import sided_forward_cukd
from chamfer_distance.Function.base import BaseSidedFunction, BaseChamferFunction


class SidedCUKD(BaseSidedFunction):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, idx1 = sided_forward_cukd(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1)
        return dist1, idx1


class ChamferCUKD(BaseChamferFunction):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, idx1 = sided_forward_cukd(xyz1, xyz2)
        dist2, idx2 = sided_forward_cukd(xyz2, xyz1)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2
