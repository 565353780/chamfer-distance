import time
import torch
from tqdm import trange

from chamfer_distance.Module.sided_distances import SidedDistances
from chamfer_distance.Module.faiss_searcher import FAISSSearcher


def demo():
    # 确保CUDA可用
    if not torch.cuda.is_available():
        print("CUDA不可用，FAISSSearcher需要CUDA支持")
        return

    # 创建随机点云数据
    n_points = 300000
    m_points = 400000

    # 创建两个点云
    xyz1 = torch.rand(1, n_points, 3, device="cuda").float()  # 查询点云
    xyz2 = torch.rand(1, m_points, 3, device="cuda").float()  # 静态点云

    print(f"点云1形状: {xyz1.shape}")
    print(f"点云2形状: {xyz2.shape}")

    # 使用传统方法计算Chamfer距离
    print("\n使用传统方法计算Chamfer距离...")
    start_time = time.time()
    for _ in trange(10):
        dists1, idxs1 = SidedDistances.faiss(xyz1, xyz2)
        mean = dists1.mean()
        assert mean >= 0
    traditional_time = time.time() - start_time
    traditional_time /= 10
    print(f"传统方法耗时: {traditional_time:.4f}秒")

    # 使用新的FAISSSearcher类
    print("\n使用新的FAISSSearcher类...")
    start_time = time.time()

    # 创建FAISSSearcher实例
    searcher = FAISSSearcher()

    # 添加静态点云
    searcher.addPoints(xyz2.reshape(-1, 3))  # 将批次维度展平

    # 查询最近邻点
    for _ in trange(10):
        dist1, idx1 = searcher.query(xyz1.reshape(-1, 3))  # 将批次维度展平
        mean = dist1.mean()
        assert mean >= 0
    new_method_time = time.time() - start_time
    new_method_time /= 10
    print(f"新方法耗时: {new_method_time:.4f}秒")
    return True
