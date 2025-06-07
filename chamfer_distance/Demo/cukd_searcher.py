import time
import torch
from tqdm import trange

from chamfer_distance.Module.cukd_searcher import CUKDSearcher
from chamfer_distance.Module.chamfer_distances import ChamferDistances


def demo(
    baseline_name: str = "cukd",
    xyz_shapes: list = [300000, 400000],
    iter_num: int = 100,
):
    # 确保CUDA可用
    if not torch.cuda.is_available():
        print("CUDA不可用，CUKDSearcher需要CUDA支持")
        return

    # 创建两个点云
    n_points, m_points = xyz_shapes
    xyz1 = torch.rand(1, n_points, 3, device="cuda").float()  # 查询点云
    xyz2 = torch.rand(1, m_points, 3, device="cuda").float()  # 静态点云

    xyz1.requires_grad_(True)

    print(f"点云1形状: {xyz1.shape}")
    print(f"点云2形状: {xyz2.shape}")

    # 使用传统方法计算Chamfer距离
    print("\n使用传统方法计算Chamfer距离...")
    start_time = time.time()
    for _ in trange(iter_num):
        dists1, dists2, idxs1, idxs2 = ChamferDistances.namedAlgo(baseline_name)(
            xyz1, xyz2
        )
        mean = dists1.mean() + dists2.mean()
        assert mean >= 0
    traditional_time = time.time() - start_time
    traditional_time /= iter_num
    print(f"传统方法耗时: {traditional_time:.4f}秒")

    # 使用新的CUKDSearcher类
    print("\n使用新的Searcher类...")
    start_time = time.time()

    searcher = CUKDSearcher()

    # 添加静态点云
    searcher.addPoints(xyz2)  # 将批次维度展平

    # 查询最近邻点
    for _ in trange(iter_num):
        dist1, dist2, idx1, idx2 = searcher.query(xyz1)  # 将批次维度展平
        mean = dist1.mean() + dist2.mean()
        assert mean >= 0
    new_method_time = time.time() - start_time
    new_method_time /= iter_num
    print(f"新方法耗时: {new_method_time:.4f}秒")
    print("speed up ratio:", traditional_time / new_method_time)
    return True
