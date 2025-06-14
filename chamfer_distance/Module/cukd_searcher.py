import torch
from typing import Tuple

try:
    import chamfer_cpp

    CHAMFER_CPP_LOADED = True
except:
    CHAMFER_CPP_LOADED = False

from chamfer_distance.Method.functions import SearcherFunction


class CUKDSearcher(object):
    def __init__(self):
        if not CHAMFER_CPP_LOADED:
            print("[ERROR][CUKDSearcher::__init__]")
            print("\t chamfer_cpp can not be imported! please compile it first!")
            exit()

        self.searcher = chamfer_cpp.CUKDSearcher()

        self.gt_points = None
        return

    def addPoints(
        self,
        points: torch.Tensor,
        THREAD_POOL: int = 16,
        BATCH_SIZE_B: int = 32,
        BATCH_SIZE_N: int = 16,
    ) -> bool:
        assert points.is_cuda, "输入点云必须是CUDA张量"
        if points.dim() == 2:
            points = points.unsqueeze(0)
        assert points.dim() == 3, "输入点云必须是[B, N, D]形状"
        assert points.size(2) == 3, "输入点云维度必须为3"

        self.searcher.addPoints(points, THREAD_POOL, BATCH_SIZE_B, BATCH_SIZE_N)
        self.gt_points = points
        return True

    def query(
        self,
        points: torch.Tensor,
        sided_forward_func_name: str = "cuda",
        BATCH_SIZE_B: int = 32,
        BATCH_SIZE_M: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.gt_points is not None, "请先调用addPoints添加点云数据"
        assert points.is_cuda, "输入点云必须是CUDA张量"
        if points.dim() == 2:
            points = points.unsqueeze(0)
        assert points.dim() == 3, "输入点云必须是[B, M, 3]形状"
        assert points.size(2) == 3, "输入点云维度必须为3"

        return SearcherFunction.apply(
            points,
            self.gt_points,
            self.searcher,
            sided_forward_func_name,
            BATCH_SIZE_B,
            BATCH_SIZE_M,
        )
