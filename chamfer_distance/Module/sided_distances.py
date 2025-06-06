import torch
from typing import Union, Tuple

from chamfer_distance.Config.path import SIDED_ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH
from chamfer_distance.Method.check import checkSidedResults
from chamfer_distance.Method.io import loadSidedAlgoIntervalDict
from chamfer_distance.Function.torch import sided_torch
from chamfer_distance.Function.triton import SidedTriton
from chamfer_distance.Function.cuda import SidedCUDA
from chamfer_distance.Function.cukd import SidedCUKD
from chamfer_distance.Module.cukd_searcher import CUKDSearcher


class SidedDistances(object):
    algo_interval_dict = loadSidedAlgoIntervalDict()

    @staticmethod
    def loadFusionAlgo(
        algo_equal_fps_point_txt_file_path: str = SIDED_ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH,
    ):
        SidedDistances.algo_interval_dict = loadSidedAlgoIntervalDict(
            algo_equal_fps_point_txt_file_path
        )

    def __init__(self, algo_equal_fps_point_txt_file_path: Union[str, None]) -> None:
        if algo_equal_fps_point_txt_file_path is not None:
            SidedDistances.loadFusionAlgo(algo_equal_fps_point_txt_file_path)
        return

    @staticmethod
    def default(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if xyz1.shape[1] * xyz2.shape[1] > 40000**2:
            print("[WARN][SidedDistances::default]")
            print("\t data are too large! will stop calculation!")
            return torch.empty(0), torch.empty(0)

        return sided_torch(xyz1, xyz2)

    @staticmethod
    def triton(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return SidedTriton.apply(xyz1, xyz2)

    @staticmethod
    def cuda(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return SidedCUDA.apply(xyz1, xyz2)

    @staticmethod
    def cukd(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return SidedCUKD.apply(xyz1, xyz2)

    @staticmethod
    def cukd_searcher(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cukd_searcher = CUKDSearcher()
        cukd_searcher.addPoints(xyz2)
        dist1, idx1 = cukd_searcher.query(xyz1)

        dist1 = dist1.unsqueeze(0)
        idx1 = idx1.unsqueeze(0)
        return dist1, idx1

    @staticmethod
    def getAlgoDict() -> dict:
        algo_dict = {
            "default": SidedDistances.default,
        }

        gpu_algo_dict = {
            "triton": SidedDistances.triton,
            "cuda": SidedDistances.cuda,
            "cukd": SidedDistances.cukd,
            "cukd_searcher": SidedDistances.cukd_searcher,
        }

        if SidedDistances.algo_interval_dict is not None:
            gpu_algo_dict["fusion"] = SidedDistances.fusion

        if torch.cuda.is_available():
            algo_dict.update(gpu_algo_dict)
            return algo_dict

        return algo_dict

    @staticmethod
    def namedAlgo(algo_name: str):
        algo_dict = SidedDistances.getAlgoDict()

        if algo_name not in algo_dict.keys():
            print("[ERROR][SidedDistances::namedAlgo]")
            print("\t algo name not valid!")
            print("\t algo_name:", algo_name)
            print("\t valid algo names are:")
            print("\t", algo_dict.keys())
            return None

        return algo_dict[algo_name]

    @staticmethod
    def fusion(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert SidedDistances.algo_interval_dict is not None

        calculation_num = xyz1.shape[0] * xyz1.shape[1] * xyz2.shape[0] * xyz2.shape[1]

        for algo_name, algo_interval in SidedDistances.algo_interval_dict.items():
            if calculation_num < algo_interval[0] or calculation_num > algo_interval[1]:
                continue

            algo_func = SidedDistances.namedAlgo(algo_name)
            assert algo_func is not None

            return algo_func(xyz1, xyz2)

    @staticmethod
    def getAlgoNameList() -> list:
        return list(SidedDistances.getAlgoDict().keys())

    @staticmethod
    def isAlgoNameValid(algo_name: str) -> bool:
        algo_name_list = SidedDistances.getAlgoNameList()
        return algo_name in algo_name_list

    @staticmethod
    def getBenchmarkAlgoName():
        return "default"

    @staticmethod
    def getBenchmarkAlgo():
        return SidedDistances.namedAlgo(SidedDistances.getBenchmarkAlgoName())

    @staticmethod
    def check(
        xyz1_shape: list = [1, 4000, 3],
        xyz2_shape: list = [1, 4000, 3],
    ) -> bool:
        xyz1 = torch.randn(*xyz1_shape).cuda()
        xyz2 = torch.randn(*xyz2_shape).cuda()

        xyz1.requires_grad_(True)

        algo_dict = SidedDistances.getAlgoDict()

        for algo_name, algo_func in algo_dict.items():
            print("[INFO][SidedDistances::check]")
            print("\t start check [" + algo_name + "]...", end="")
            checkSidedResults(algo_func, SidedDistances.getBenchmarkAlgo(), xyz1, xyz2)
            print("\t passed!")
        return True
