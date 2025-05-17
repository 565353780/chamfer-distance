import torch
from typing import Tuple

from chamfer_distance.Function.base import BaseSearcherFunction


class BaseSearcher:
    def __init__(self):
        self.gt_points = None
        return

    def addPoints(self, points: torch.Tensor) -> bool:
        assert points.is_cuda, "输入点云必须是CUDA张量"
        assert points.dim() == 2, "输入点云必须是[N, D]形状"
        assert points.size(1) == 3, "输入点云维度必须为3"

        self.searcher.addPoints(points)
        self.gt_points = points
        return True

    def query(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.gt_points is not None, "请先调用addPoints添加点云数据"
        assert points.is_cuda, "输入点云必须是CUDA张量"
        assert points.dim() == 2, "输入点云必须是[M, 3]形状"
        assert points.size(1) == 3, "输入点云维度必须为3"

        return BaseSearcherFunction.apply(self.searcher, points, self.gt_points)
