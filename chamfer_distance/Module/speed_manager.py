import torch
import numpy as np
from time import time

from chamfer_distance.Data.fps_map import FPSMap
from chamfer_distance.Module.chamfer_distances import ChamferDistances
from chamfer_distance.Module.timer import Timer


class SpeedManager(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def getAlgoFPS(
        algo_name: str,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        max_test_second: float = 1.0,
        warmup: int = 10,
        window_size: int = 10,
        rel_std_threshold: float = 0.01,
    ) -> float:
        algo_func = ChamferDistances.namedAlgo(algo_name)
        if algo_func is None:
            print('[ERROR][SpeedManager::getAlgoFPS]')
            print('\t namedAlgo failed!')
            return 0.0

        if algo_name == 'cpu':
            if xyz1.shape[0] * xyz1.shape[1] * xyz2.shape[0] * xyz2.shape[1] > 40000 ** 2:
                print('[WARN][SpeedManager::getAlgoFPS]')
                print('\t data too large for cpu calculation! will return fps = 0.0!')
                return 0.0

        print('[INFO][SpeedManager]')
        print('\t start test speed of [' + algo_name + ']...', end='')
        fps_list = []
        window = []

        timer = Timer()
        for i in range(100000000):
            start = time()

            dist1, dist2, = algo_func(xyz1, xyz2)[:2]
            if isinstance(dist1, torch.Tensor):
                mean = torch.mean(dist1) + torch.mean(dist2)
            else:
                mean = np.mean(dist1) + np.mean(dist2)

            assert mean >= 0

            end = time()

            fps = 1.0 / (end - start)
            fps_list.append(fps)

            if i < warmup:
                continue

            window.append(fps)
            if len(window) > window_size:
                window.pop(0)

            if len(window) == window_size:
                max_value = np.max(window)
                std = np.std(window)
                if std / max_value < rel_std_threshold:
                    break

            if timer.now() > max_test_second:
                break

        fps = float(np.mean(window))

        print('\t fps =', fps)

        return fps

    @staticmethod
    def getAlgoFPSDict(
        xyz1_shape: list = [1, 4000, 3],
        xyz2_shape: list = [1, 4000, 3],
        max_test_second: float = 1.0,
        warmup: int = 10,
        window_size: int = 10,
        rel_std_threshold: float = 0.01,
    ) -> dict:
        xyz1 = torch.randn(*xyz1_shape).cuda()
        xyz2 = torch.randn(*xyz2_shape).cuda()

        xyz1.requires_grad_(True)
        xyz2.requires_grad_(True)

        algo_fps_dict = {}

        algo_name_list = ChamferDistances.getAlgoNameList()

        for algo_name in algo_name_list:
            algo_fps = SpeedManager.getAlgoFPS(
                algo_name,
                xyz1,
                xyz2,
                max_test_second,
                warmup,
                window_size,
                rel_std_threshold,
            )

            algo_fps_dict[algo_name] = algo_fps

        return algo_fps_dict

    @staticmethod
    def getAlgoFPSMapDict(
        point_cloud_sizes_m: list,
        point_cloud_sizes_n: list,
    ) -> dict:
        algo_name_list = ChamferDistances.getAlgoNameList()

        algo_fps_map_dict = {}
        for algo_name in algo_name_list:
            algo_fps_map_dict[algo_name] = FPSMap()

        for m in point_cloud_sizes_m:
            for n in point_cloud_sizes_n:
                if m > n:
                    continue

                print('[INFO][SpeedManager::getAlgoFPSMapDict]')
                print(f"\t test point cloud sizes : P={m}, Q={n}")

                xyz1_shape = [1, m, 3]
                xyz2_shape = [1, n, 3]

                algo_fps_dict = SpeedManager.getAlgoFPSDict(xyz1_shape, xyz2_shape)

                for algo_name, algo_fps in algo_fps_dict.items():
                    algo_fps_map_dict[algo_name].addFPS(m, n, algo_fps)

        return algo_fps_map_dict
