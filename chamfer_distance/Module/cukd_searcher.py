import torch
from typing import Tuple

import chamfer_cpp

from chamfer_distance.Method.functions import SearcherFunction


class CUKDSearcher(object):
    def __init__(self):
        self.searcher = chamfer_cpp.CUKDSearcher()

        self.gt_points = None
        return

    def addPoints(self, points: torch.Tensor) -> bool:
        assert points.is_cuda, "输入点云必须是CUDA张量"
        if points.dim() == 3 and points.shape[0] == 1:
            points = points.squeeze(0)
        assert points.dim() == 2, "输入点云必须是[N, D]形状"
        assert points.size(1) == 3, "输入点云维度必须为3"

        self.searcher.addPoints(points)
        self.gt_points = points
        return True

    def query(
        self, points: torch.Tensor, sided_forward_func_name: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.gt_points is not None, "请先调用addPoints添加点云数据"
        assert points.is_cuda, "输入点云必须是CUDA张量"
        if points.dim() == 3 and points.shape[0] == 1:
            points = points.squeeze(0)
        assert points.dim() == 2, "输入点云必须是[M, 3]形状"
        assert points.size(1) == 3, "输入点云维度必须为3"

        return SearcherFunction.apply(
            points, self.gt_points, self.searcher, sided_forward_func_name
        )
