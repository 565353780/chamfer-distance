import torch
import numpy as np
from math import sqrt
from time import time
from typing import Union
from scipy.optimize import brentq

from chamfer_distance.Config.path import ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH
from chamfer_distance.Data.fps_map import FPSMap
from chamfer_distance.Method.path import createFileFolder
from chamfer_distance.Module.chamfer_distances import ChamferDistances
from chamfer_distance.Module.timer import Timer


class ChamferSpeedManager(object):
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
            print("[ERROR][ChamferSpeedManager::getAlgoFPS]")
            print("\t namedAlgo failed!")
            return 0.0

        if algo_name == "cpu":
            if xyz1.shape[0] * xyz1.shape[1] * xyz2.shape[0] * xyz2.shape[1] > 40000**2:
                print("[WARN][ChamferSpeedManager::getAlgoFPS]")
                print("\t data too large for cpu calculation! will return fps = 0.0!")
                return 0.0

        print("[INFO][ChamferSpeedManager]")
        print("\t start test speed of [" + algo_name + "]...", end="")
        fps_list = []
        window = []

        timer = Timer()
        for i in range(100000000):
            start = time()

            (
                dist1,
                dist2,
            ) = algo_func(xyz1, xyz2)[:2]
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

        print("\t fps =", fps)

        return fps

    @staticmethod
    def getAlgosFPSDiff(
        algo_name_1: str,
        algo_name_2: str,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        max_test_second: float = 1.0,
        warmup: int = 10,
        window_size: int = 10,
        rel_std_threshold: float = 0.01,
    ) -> float:
        fps_1 = ChamferSpeedManager.getAlgoFPS(
            algo_name_1,
            xyz1,
            xyz2,
            max_test_second,
            warmup,
            window_size,
            rel_std_threshold,
        )
        fps_2 = ChamferSpeedManager.getAlgoFPS(
            algo_name_2,
            xyz1,
            xyz2,
            max_test_second,
            warmup,
            window_size,
            rel_std_threshold,
        )
        return fps_1 - fps_2

    @staticmethod
    def getAlgosFPSDiffSimple(
        algo_name_1: str,
        algo_name_2: str,
        calculation_num: Union[int, float],
        max_test_second: float = 1.0,
        warmup: int = 10,
        window_size: int = 10,
        rel_std_threshold: float = 0.01,
    ) -> float:
        x = int(sqrt(calculation_num))
        y = int(calculation_num) // x
        xyz1 = torch.randn(1, x, 3).cuda()
        xyz2 = torch.randn(1, y, 3).cuda()

        xyz1.requires_grad_(True)
        xyz2.requires_grad_(True)

        return ChamferSpeedManager.getAlgosFPSDiff(
            algo_name_1,
            algo_name_2,
            xyz1,
            xyz2,
            max_test_second,
            warmup,
            window_size,
            rel_std_threshold,
        )

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
            algo_fps = ChamferSpeedManager.getAlgoFPS(
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
    def getAlgoSimpleFPSMapDict(
        calculation_nums: list,
    ) -> dict:
        calculation_nums = [int(data) for data in calculation_nums]

        xy_pairs = []

        for calculation_num in calculation_nums:
            x = int(sqrt(calculation_num))
            y = calculation_num // x
            xy_pairs.append((x, y))

        algo_name_list = ChamferDistances.getAlgoNameList()

        algo_fps_map_dict = {}
        for algo_name in algo_name_list:
            algo_fps_map_dict[algo_name] = FPSMap()

        for m, n in xy_pairs:
            print("[INFO][ChamferSpeedManager::getAlgoFPSMapDict]")
            print(f"\t test point cloud sizes : P={m}, Q={n}")

            xyz1_shape = [1, m, 3]
            xyz2_shape = [1, n, 3]

            algo_fps_dict = ChamferSpeedManager.getAlgoFPSDict(xyz1_shape, xyz2_shape)

            for algo_name, algo_fps in algo_fps_dict.items():
                algo_fps_map_dict[algo_name].addFPS(m, n, algo_fps, False)

        return algo_fps_map_dict

    @staticmethod
    def getAlgoBalanceFPSMapDict(
        calculation_num: int = 10000**2,
        split_num: int = 10,
        max_unbalance_weight: float = 9.0,
        max_test_second: float = 1.0,
        warmup: int = 10,
        window_size: int = 10,
        rel_std_threshold: float = 0.01,
    ) -> dict:
        ratios = np.geomspace(
            1.0 / max_unbalance_weight, max_unbalance_weight, num=split_num
        )
        xy_pairs = []

        for r in ratios:
            x = int(round((calculation_num * r) ** 0.5))
            y = calculation_num // x
            xy_pairs.append((x, y))

        algo_name_list = ChamferDistances.getAlgoNameList()

        algo_fps_map_dict = {}
        for algo_name in algo_name_list:
            algo_fps_map_dict[algo_name] = FPSMap()

        for m, n in xy_pairs:
            print("[INFO][ChamferSpeedManager::getAlgoFPSMapDict]")
            print(f"\t test point cloud sizes : P={m}, Q={n}")

            xyz1_shape = [1, m, 3]
            xyz2_shape = [1, n, 3]

            algo_fps_dict = ChamferSpeedManager.getAlgoFPSDict(
                xyz1_shape,
                xyz2_shape,
                max_test_second,
                warmup,
                window_size,
                rel_std_threshold,
            )

            for algo_name, algo_fps in algo_fps_dict.items():
                algo_fps_map_dict[algo_name].addFPS(m, n, algo_fps, False)

        return algo_fps_map_dict

    @staticmethod
    def getAlgoFPSMapDict(
        point_cloud_sizes_m: list,
        point_cloud_sizes_n: list,
        max_test_second: float = 1.0,
        warmup: int = 10,
        window_size: int = 10,
        rel_std_threshold: float = 0.01,
    ) -> dict:
        point_cloud_sizes_m = [int(data) for data in point_cloud_sizes_m]
        point_cloud_sizes_n = [int(data) for data in point_cloud_sizes_n]

        algo_name_list = ChamferDistances.getAlgoNameList()

        algo_fps_map_dict = {}
        for algo_name in algo_name_list:
            algo_fps_map_dict[algo_name] = FPSMap()

        for m in point_cloud_sizes_m:
            for n in point_cloud_sizes_n:
                if m > n:
                    continue

                print("[INFO][ChamferSpeedManager::getAlgoFPSMapDict]")
                print(f"\t test point cloud sizes : P={m}, Q={n}")

                xyz1_shape = [1, m, 3]
                xyz2_shape = [1, n, 3]

                algo_fps_dict = ChamferSpeedManager.getAlgoFPSDict(
                    xyz1_shape,
                    xyz2_shape,
                    max_test_second,
                    warmup,
                    window_size,
                    rel_std_threshold,
                )

                for algo_name, algo_fps in algo_fps_dict.items():
                    algo_fps_map_dict[algo_name].addFPS(m, n, algo_fps)

        return algo_fps_map_dict

    @staticmethod
    def getAlgosEqualFPSPoint(
        algo_name_1: str,
        algo_name_2: str,
        min_calculation_num: Union[int, float] = 1e4,
        max_calculation_num: Union[int, float] = 1e10,
        max_test_second: float = 1.0,
        warmup: int = 10,
        window_size: int = 10,
        rel_std_threshold: float = 0.01,
    ) -> Union[float, None]:
        if not ChamferDistances.isAlgoNameValid(algo_name_1):
            print("[ERROR][ChamferSpeedManager::getAlgosEqualFPSPoint]")
            print("\t namedAlgo failed for algo name 1!")
            return None
        if not ChamferDistances.isAlgoNameValid(algo_name_2):
            print("[ERROR][ChamferSpeedManager::getAlgosEqualFPSPoint]")
            print("\t namedAlgo failed for algo name 2!")
            return None

        def fpsDiff(calculation_num: float) -> float:
            fps_diff = ChamferSpeedManager.getAlgosFPSDiffSimple(
                algo_name_1,
                algo_name_2,
                calculation_num,
                max_test_second,
                warmup,
                window_size,
                rel_std_threshold,
            )

            return fps_diff

        equal_fps_point = brentq(
            fpsDiff,
            min_calculation_num,
            max_calculation_num,
            xtol=1.0,
            rtol=1e-3,
            maxiter=100,
        )

        return equal_fps_point

    @staticmethod
    def saveEqualFPSPoint(
        algo_interval_dict: dict,
        save_equal_fps_point_txt_file_path: str = ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH,
    ) -> bool:
        createFileFolder(save_equal_fps_point_txt_file_path)

        with open(save_equal_fps_point_txt_file_path, "w") as f:
            for algo_name, algo_interval in algo_interval_dict.items():
                f.write(algo_name)
                f.write("|")
                f.write(str(algo_interval[0]))
                f.write("|")
                f.write(str(algo_interval[1]))
                f.write("\n")

        return True
