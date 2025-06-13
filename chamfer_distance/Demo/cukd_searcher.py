import time
import torch
from tqdm import trange

from chamfer_distance.Module.cukd_searcher import CUKDSearcher
from chamfer_distance.Module.chamfer_distances import ChamferDistances


def testAlgoSpeed(
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
    algo_name: str,
    iter_num: int = 10,
) -> bool:
    print("start calculate chamfer via " + algo_name + "...")
    start_time = time.time()
    for _ in trange(iter_num):
        dists1, dists2, idxs1, idxs2 = ChamferDistances.namedAlgo(algo_name)(xyz1, xyz2)
        mean = dists1.mean() + dists2.mean()
        assert mean >= 0
    spend_time = time.time() - start_time
    spend_time /= iter_num

    print("time:", spend_time, "s; fps:", 1.0 / spend_time)
    return True


def demo(
    baseline_name: str = "cukd",
    xyz_shapes: list = [1600, 1000],
    iter_num: int = 10,
    device: str = "cuda:0",
):
    # 确保CUDA可用
    if not torch.cuda.is_available():
        print("CUDA不可用，CUKDSearcher需要CUDA支持")
        return

    # 创建两个点云
    n_points, m_points = xyz_shapes
    xyz1 = torch.rand(10000, n_points, 3, device=device).float()  # 查询点云
    xyz2 = torch.rand(10000, m_points, 3, device=device).float()  # 静态点云

    xyz1.requires_grad_(True)

    print(f"点云1形状: {xyz1.shape}")
    print(f"点云2形状: {xyz2.shape}")

    # testAlgoSpeed(xyz1, xyz2, "default", iter_num)
    testAlgoSpeed(xyz1, xyz2, "cuda", iter_num)
    # testAlgoSpeed(xyz1, xyz2, "triton", iter_num)
    testAlgoSpeed(xyz1, xyz2, "cukd", iter_num)

    print("\n使用新的Searcher类...")
    searcher = CUKDSearcher()
    searcher.addPoints(xyz2)

    start_time = time.time()
    for _ in trange(iter_num):
        dist1, dist2, idx1, idx2 = searcher.query(xyz1)
        mean = dist1.mean() + dist2.mean()
        assert mean >= 0
    new_method_time = time.time() - start_time
    new_method_time /= iter_num
    print("time:", new_method_time, "s; fps:", 1.0 / new_method_time)
    return True
