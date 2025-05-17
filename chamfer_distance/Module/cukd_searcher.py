import torch
from typing import Tuple

import chamfer_cpp


class CUKDSearcher:
    """
    CUKDSearcher类的Python包装器，用于在GPU上进行高效的最近邻搜索。
    该类使用CUKD库构建KD树，允许添加静态点云数据，并对查询点进行最近邻搜索，所有操作都在GPU上完成。
    相比FAISS，CUKD在某些场景下可能提供更好的性能。

    使用示例：
        searcher = ChamferDistances.CUKDSearcher()
        searcher.addPoints(xyz2)  # 添加静态点云
        dist, idx = searcher.query(xyz1)  # 查询最近邻点
    """

    def __init__(self):
        """
        初始化CUKDSearcher对象
        """
        self.searcher = chamfer_cpp.CUKDSearcher()
        self.has_points = False

    def addPoints(self, points: torch.Tensor) -> None:
        """
        添加点云数据并构建KD树

        参数:
            points: 形状为[N, 3]的点云数据，必须是CUDA张量
        """
        assert points.is_cuda, "输入点云必须是CUDA张量"
        assert points.dim() == 2, "输入点云必须是[N, 3]形状"
        assert points.size(1) == 3, "输入点云维度必须为3"
        
        self.searcher.addPoints(points)
        self.has_points = True

    def query(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        查询最近邻点

        参数:
            points: 形状为[M, 3]的查询点，必须是CUDA张量

        返回:
            dist: 形状为[M]的距离张量，表示每个查询点到最近点的平方欧氏距离
            idx: 形状为[M]的索引张量，表示每个查询点的最近点在原始点云中的索引
        """
        assert self.has_points, "请先调用addPoints添加点云数据"
        assert points.is_cuda, "输入点云必须是CUDA张量"
        assert points.dim() == 2, "输入点云必须是[M, 3]形状"
        assert points.size(1) == 3, "输入点云维度必须为3"
        
        return self.searcher.query(points)