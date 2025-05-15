import torch
from typing import Union, Tuple
from kaolin.metrics.pointcloud import sided_distance

from chamfer_distance.Config.path import ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH
from chamfer_distance.Method.check import checkResults
from chamfer_distance.Method.io import loadAlgoIntervalDict
from chamfer_distance.Function.torch import chamfer_torch
from chamfer_distance.Function.triton import ChamferTriton
from chamfer_distance.Function.cuda import ChamferCUDA
from chamfer_distance.Function.cukd import ChamferCUKD
from chamfer_distance.Function.faiss import ChamferFAISS


class ChamferDistances(object):
    algo_interval_dict = loadAlgoIntervalDict()

    @staticmethod
    def loadFusionAlgo(
        algo_equal_fps_point_txt_file_path: str = ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH,
    ):
        ChamferDistances.algo_interval_dict = loadAlgoIntervalDict(
            algo_equal_fps_point_txt_file_path
        )

    def __init__(self, algo_equal_fps_point_txt_file_path: Union[str, None]) -> None:
        if algo_equal_fps_point_txt_file_path is not None:
            ChamferDistances.loadFusionAlgo(algo_equal_fps_point_txt_file_path)
        return

    @staticmethod
    def default(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if xyz1.shape[1] * xyz2.shape[1] > 40000**2:
            print("[WARN][ChamferDistances::default]")
            print("\t data are too large! will stop calculation!")
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

        return chamfer_torch(xyz1, xyz2)

    @staticmethod
    def triton(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return ChamferTriton.apply(xyz1, xyz2)

    @staticmethod
    def cuda(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return ChamferCUDA.apply(xyz1, xyz2)

    @staticmethod
    def cukd(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return ChamferCUKD.apply(xyz1, xyz2)

    @staticmethod
    def kaolin(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dists1, idxs1 = sided_distance(xyz1, xyz2)
        dists2, idxs2 = sided_distance(xyz2, xyz1)
        return dists1, dists2, idxs1, idxs2

    @staticmethod
    def faiss(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return ChamferFAISS.apply(xyz1, xyz2)

    @staticmethod
    def getAlgoDict() -> dict:
        algo_dict = {
            "default": ChamferDistances.default,
        }

        gpu_algo_dict = {
            "triton": ChamferDistances.triton,
            "cuda": ChamferDistances.cuda,
            "kaolin": ChamferDistances.kaolin,
            "faiss": ChamferDistances.faiss,
        }

        if ChamferDistances.algo_interval_dict is not None:
            gpu_algo_dict["fusion"] = ChamferDistances.fusion

        if torch.cuda.is_available():
            algo_dict.update(gpu_algo_dict)
            return algo_dict

        return algo_dict

    @staticmethod
    def namedAlgo(algo_name: str):
        algo_dict = ChamferDistances.getAlgoDict()

        if algo_name not in algo_dict.keys():
            print("[ERROR][ChamferDistances::namedAlgo]")
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert ChamferDistances.algo_interval_dict is not None

        calculation_num = xyz1.shape[0] * xyz1.shape[1] * xyz2.shape[0] * xyz2.shape[1]

        for algo_name, algo_interval in ChamferDistances.algo_interval_dict.items():
            if calculation_num < algo_interval[0] or calculation_num > algo_interval[1]:
                continue

            algo_func = ChamferDistances.namedAlgo(algo_name)
            assert algo_func is not None

            return algo_func(xyz1, xyz2)

    @staticmethod
    def getAlgoNameList() -> list:
        return list(ChamferDistances.getAlgoDict().keys())

    @staticmethod
    def isAlgoNameValid(algo_name: str) -> bool:
        algo_name_list = ChamferDistances.getAlgoNameList()
        return algo_name in algo_name_list

    @staticmethod
    def getBenchmarkAlgoName():
        return "default"

    @staticmethod
    def getBenchmarkAlgo():
        return ChamferDistances.namedAlgo(ChamferDistances.getBenchmarkAlgoName())

    @staticmethod
    def check(
        xyz1_shape: list = [1, 4000, 3],
        xyz2_shape: list = [1, 4000, 3],
    ) -> bool:
        xyz1 = torch.randn(*xyz1_shape).cuda()
        xyz2 = torch.randn(*xyz2_shape).cuda()

        xyz1.requires_grad_(True)
        xyz2.requires_grad_(True)

        algo_dict = ChamferDistances.getAlgoDict()

        for algo_name, algo_func in algo_dict.items():
            print("[INFO][ChamferDistances::check]")
            print("\t start check [" + algo_name + "]...", end="")
            checkResults(algo_func, ChamferDistances.getBenchmarkAlgo(), xyz1, xyz2)
            print("\t passed!")
        return True
