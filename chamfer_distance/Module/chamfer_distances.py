import torch
from typing import Tuple

from chamfer_distance.Method.check import checkChamferResults
from chamfer_distance.Method.chamfer_torch import chamfer_torch
from chamfer_distance.Method.functions import ChamferFunction


class ChamferDistances(object):
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
        return ChamferFunction.apply(xyz1, xyz2, "triton")

    @staticmethod
    def cuda(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return ChamferFunction.apply(xyz1, xyz2, "cuda")

    @staticmethod
    def cukd(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return ChamferFunction.apply(xyz1, xyz2, "cukd")

    @staticmethod
    def getAlgoDict() -> dict:
        algo_dict = {
            "default": ChamferDistances.default,
        }

        gpu_algo_dict = {
            "triton": ChamferDistances.triton,
            "cuda": ChamferDistances.cuda,
            "cukd": ChamferDistances.cukd,
        }

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
        xyz1_requires_grad: bool = True,
        xyz2_requires_grad: bool = True,
    ) -> bool:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        xyz1 = torch.randn(*xyz1_shape).to(device)
        xyz2 = torch.randn(*xyz2_shape).to(device)

        if xyz1_requires_grad:
            xyz1.requires_grad_(True)
        if xyz2_requires_grad:
            xyz2.requires_grad_(True)

        algo_dict = ChamferDistances.getAlgoDict()

        for algo_name, algo_func in algo_dict.items():
            print("[INFO][ChamferDistances::check]")
            print("\t start check [" + algo_name + "]...", end="")
            checkChamferResults(
                algo_func, ChamferDistances.getBenchmarkAlgo(), xyz1, xyz2
            )
            print("\t passed!")
        return True
