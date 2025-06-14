import time
import torch
import optuna
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


def testCUKDSearcherSpeed(
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
    iter_num: int = 10,
    THREAD_POOL: int = 16,
    BATCH_SIZE_B: int = 32,
    BATCH_SIZE_N: int = 16,
    BATCH_SIZE_M: int = 16,
) -> bool:
    print("start calculate chamfer via CUKDSearcher...")
    start_time = time.time()
    for _ in trange(iter_num):
        cukd_searcher = CUKDSearcher()
        cukd_searcher.addPoints(xyz2, THREAD_POOL, BATCH_SIZE_B, BATCH_SIZE_N)
        dists1, dists2, idxs1, idxs2 = cukd_searcher.query(
            xyz1, "cuda", BATCH_SIZE_B, BATCH_SIZE_M
        )
        mean = dists1.mean() + dists2.mean()
        assert mean >= 0
    spend_time = time.time() - start_time
    spend_time /= iter_num

    print(
        "THREAD_POOL:",
        THREAD_POOL,
        "BATCH_SIZE_B:",
        BATCH_SIZE_B,
        "BATCH_SIZE_N:",
        BATCH_SIZE_N,
        "BATCH_SIZE_M:",
        BATCH_SIZE_M,
    )
    print("time:", spend_time, "s; fps:", 1.0 / spend_time)
    return True


def testCUKDSearcherAddPointsSpeed(
    xyz2: torch.Tensor,
    iter_num: int = 10,
    THREAD_POOL: int = 16,
    BATCH_SIZE_B: int = 32,
    BATCH_SIZE_N: int = 16,
) -> float:
    start_time = time.time()
    for _ in range(iter_num):
        cukd_searcher = CUKDSearcher()
        cukd_searcher.addPoints(xyz2, THREAD_POOL, BATCH_SIZE_B, BATCH_SIZE_N)
    spend_time = time.time() - start_time
    return spend_time


def testCUKDSearcherQuerySpeed(
    cukd_searcher: CUKDSearcher,
    xyz1: torch.Tensor,
    iter_num: int = 10,
    BATCH_SIZE_B: int = 32,
    BATCH_SIZE_M: int = 16,
) -> float:
    start_time = time.time()
    for _ in range(iter_num):
        dists1, dists2, idxs1, idxs2 = cukd_searcher.query(
            xyz1, "cuda", BATCH_SIZE_B, BATCH_SIZE_M
        )
        mean = dists1.mean() + dists2.mean()
        assert mean >= 0
    spend_time = time.time() - start_time
    return spend_time


def demo_test_speed(
    xyz1_shapes: list = [4000, 1600, 3],
    xyz2_shapes: list = [4000, 1000, 3],
    iter_num: int = 10,
    device: str = "cuda:0",
):
    # 确保CUDA可用
    if not torch.cuda.is_available():
        print("CUDA不可用，CUKDSearcher需要CUDA支持")
        return

    # 创建两个点云
    xyz1 = torch.rand(*xyz1_shapes, device=device).float()  # 查询点云
    xyz2 = torch.rand(*xyz2_shapes, device=device).float()  # 静态点云

    xyz1.requires_grad_(True)

    print(f"点云1形状: {xyz1.shape}")
    print(f"点云2形状: {xyz2.shape}")

    # testAlgoSpeed(xyz1, xyz2, "default", iter_num)
    testAlgoSpeed(xyz1, xyz2, "cuda", iter_num)
    # testAlgoSpeed(xyz1, xyz2, "triton", iter_num)
    testAlgoSpeed(xyz1, xyz2, "cukd", iter_num)
    testCUKDSearcherSpeed(xyz1, xyz2, iter_num)
    return True


def demo_search_best_param(
    xyz1_shapes: list = [4000, 1600, 3],
    xyz2_shapes: list = [4000, 1000, 3],
    iter_num: int = 10,
    device: str = "cuda:0",
):
    # 确保CUDA可用
    if not torch.cuda.is_available():
        print("CUDA不可用，CUKDSearcher需要CUDA支持")
        return

    # 创建两个点云
    xyz1 = torch.rand(*xyz1_shapes, device=device).float()  # 查询点云
    xyz2 = torch.rand(*xyz2_shapes, device=device).float()  # 静态点云

    xyz1.requires_grad_(True)

    print(f"点云1形状: {xyz1.shape}")
    print(f"点云2形状: {xyz2.shape}")

    def objectiveAddPoints(trial):
        THREAD_POOL = trial.suggest_categorical("THREAD_POOL", [32, 64, 128, 256])
        BATCH_SIZE_B = trial.suggest_categorical("BATCH_SIZE_B", [1, 2, 4, 8, 16, 32])
        BATCH_SIZE_N = trial.suggest_categorical("BATCH_SIZE_N", [1, 2, 4, 8, 16, 32])
        # batch_size = trial.suggest_int("batch_size", 16, 128, step=16)

        time_cost = testCUKDSearcherAddPointsSpeed(
            xyz2,
            iter_num,
            THREAD_POOL,
            BATCH_SIZE_B,
            BATCH_SIZE_N,
        )
        return time_cost

    cukd_searcher = CUKDSearcher()
    cukd_searcher.addPoints(xyz2)

    def objectiveQuery(trial):
        BATCH_SIZE_B = trial.suggest_categorical("BATCH_SIZE_B", [1, 2, 4, 8, 16, 32])
        BATCH_SIZE_M = trial.suggest_categorical("BATCH_SIZE_M", [1, 2, 4, 8, 16, 32])

        time_cost = testCUKDSearcherQuerySpeed(
            cukd_searcher,
            xyz1,
            iter_num,
            BATCH_SIZE_B,
            BATCH_SIZE_M,
        )
        return time_cost

    study_add_points = optuna.create_study(direction="minimize")
    study_add_points.optimize(objectiveAddPoints, n_trials=50)

    study_query = optuna.create_study(direction="minimize")
    study_query.optimize(objectiveQuery, n_trials=50)

    print("Best params of addPoints:", study_add_points.best_params)
    print("Best params of query:", study_query.best_params)

    return True
