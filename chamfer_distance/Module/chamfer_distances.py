import torch
from typing import Tuple

import chamfer_cpp

from chamfer_distance.Method.chamfer_triton import chamfer_triton
from chamfer_distance.Method.check import checkResults


class ChamferDistances(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def cpu(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if xyz1.shape[1] * xyz2.shape[1] > 40000 ** 2:
            print('[WARN][ChamferDistances::cpu]')
            print('\t data are too large! will stop calculation!')
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

        if xyz1.device != 'cpu':
            xyz1 = xyz1.cpu()
        if xyz2.device != 'cpu':
            xyz2 = xyz2.cpu()
        dists1, dists2, idxs1, idxs2 = chamfer_cpp.chamfer_cpu(xyz1, xyz2)
        return dists1, dists2, idxs1, idxs2

    @staticmethod
    def cuda(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dists1, dists2, idxs1, idxs2 = chamfer_cpp.chamfer_cuda(xyz1, xyz2)
        return dists1, dists2, idxs1, idxs2

    @staticmethod
    def triton(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dists1, dists2, idxs1, idxs2 = chamfer_triton(xyz1, xyz2)
        return dists1, dists2, idxs1, idxs2

    @staticmethod
    def cuda_kd(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dists1, dists2, idxs1, idxs2 = chamfer_cpp.chamfer_cuda_kd(xyz1, xyz2)
        return dists1, dists2, idxs1, idxs2

    @staticmethod
    def cuda_kd_cub(
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dists1, dists2, idxs1, idxs2 = chamfer_cpp.chamfer_cuda_kd_cub(xyz1, xyz2)
        return dists1, dists2, idxs1, idxs2

    @staticmethod
    def getAlgoDict() -> dict:
        cpu_algo_dict = {
            'cpu': ChamferDistances.cpu,
        }

        gpu_algo_dict = {
            'cuda': ChamferDistances.cuda,
            'triton': ChamferDistances.triton,
            'cuda_kd': ChamferDistances.cuda_kd,
            'cuda_kd_cub': ChamferDistances.cuda_kd_cub,
        }

        if torch.cuda.is_available():
            return cpu_algo_dict | gpu_algo_dict

        return cpu_algo_dict

    @staticmethod
    def getAlgoNameList() -> list:
        return list(ChamferDistances.getAlgoDict().keys())

    @staticmethod
    def namedAlgo(algo_name: str):
        algo_dict = ChamferDistances.getAlgoDict()

        if algo_name not in algo_dict.keys():
            print('[ERROR][ChamferDistances::namedAlgo]')
            print('\t algo name not valid!')
            print('\t algo_name:', algo_name)
            print('\t valid algo names are:')
            print('\t', algo_dict.keys())
            return None

        return algo_dict[algo_name]

    @staticmethod
    def getBenchmarkAlgoName():
        return 'cuda'

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
            print('[INFO][ChamferDistances::check]')
            print('\t start check [' + algo_name + ']...', end='')
            checkResults(algo_func, ChamferDistances.getBenchmarkAlgo(), xyz1, xyz2)
            print('\t passed!')
        return True
