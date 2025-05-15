import torch
from typing import Tuple

import chamfer_cpp


class FAISSSearcher:
    """
    FAISSSearcher类的Python包装器，用于在GPU上进行高效的最近邻搜索。
    该类允许添加静态点云数据，并对查询点进行最近邻搜索，所有操作都在GPU上完成。

    使用示例：
        searcher = ChamferDistances.FAISSSearcher()
        searcher.addPoints(xyz2)  # 添加静态点云
        dist, idx = searcher.query(xyz1)  # 查询最近邻点
    """

    def __init__(self):
        """
        初始化FAISSSearcher对象
        """
        self.searcher = chamfer_cpp.FAISSSearcher()

    def addPoints(self, points: torch.Tensor) -> None:
        """
        添加点云数据到索引

        参数:
            points: 形状为[N, D]的点云数据，必须是CUDA张量
        """
        assert points.is_cuda, "输入点云必须是CUDA张量"
        assert points.dim() == 2, "输入点云必须是[N, D]形状"
        self.searcher.addPoints(points)

    def query(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        查询最近邻点

        参数:
            points: 形状为[M, D]的查询点，必须是CUDA张量

        返回:
            dist: 形状为[M]的距离张量
            idx: 形状为[M]的索引张量
        """
        assert points.is_cuda, "输入点云必须是CUDA张量"
        assert points.dim() == 2, "输入点云必须是[M, D]形状"
        return self.searcher.query(points)
