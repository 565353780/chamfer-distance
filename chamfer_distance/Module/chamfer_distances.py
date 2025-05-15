import torch
from typing import Union, Tuple, List
from kaolin.metrics.pointcloud import sided_distance

import chamfer_cpp

from chamfer_distance.Config.path import ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH
from chamfer_distance.Method.chamfer_triton import chamfer_triton
from chamfer_distance.Method.check import checkResults
from chamfer_distance.Method.io import loadAlgoIntervalDict


class ChamferDistances(object):
    algo_interval_dict = loadAlgoIntervalDict()

    @staticmethod
    def loadFusionAlgo(algo_equal_fps_point_txt_file_path: str = ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH):
        ChamferDistances.algo_interval_dict = loadAlgoIntervalDict(algo_equal_fps_point_txt_file_path)

    def __init__(self, algo_equal_fps_point_txt_file_path: Union[str, None]) -> None:
        if algo_equal_fps_point_txt_file_path is not None:
            ChamferDistances.loadFusionAlgo(algo_equal_fps_point_txt_file_path)
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
        dists1, dists2, idxs1, idxs2 = chamfer_cpp.chamfer_faiss(xyz1, xyz2)
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
        
    class FAISSSearcher:
        """
        FAISSSearcher类的Python包装器，用于在GPU上进行高效的最近邻搜索。
        该类允许添加静态点云数据，并对查询点进行最近邻搜索，所有操作都在GPU上完成。
        
        使用示例：
            searcher = ChamferDistances.FAISSSearcher()
            searcher.addPoints(xyz2)  # 添加静态点云
            dist, idx = searcher.query(xyz1)  # 查询最近邻点
        """
        def __init__(self):
            """
            初始化FAISSSearcher对象
            """
            self.searcher = chamfer_cpp.FAISSSearcher()
            
        def addPoints(self, points: torch.Tensor) -> None:
            """
            添加点云数据到索引
            
            参数:
                points: 形状为[N, D]的点云数据，必须是CUDA张量
            """
            assert points.is_cuda, "输入点云必须是CUDA张量"
            assert points.dim() == 2, "输入点云必须是[N, D]形状"
            self.searcher.addPoints(points)
            
        def query(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            查询最近邻点
            
            参数:
                points: 形状为[M, D]的查询点，必须是CUDA张量
                
            返回:
                dist: 形状为[M]的距离张量
                idx: 形状为[M]的索引张量
            """
            assert points.is_cuda, "输入点云必须是CUDA张量"
            assert points.dim() == 2, "输入点云必须是[M, D]形状"
            return self.searcher.query(points)

    @staticmethod
    def getAlgoDict() -> dict:
        cpu_algo_dict = {
            'cpu': ChamferDistances.cpu,
        }

        gpu_algo_dict = {
            'kaolin': ChamferDistances.kaolin,
            'faiss': ChamferDistances.faiss,
            'cuda': ChamferDistances.cuda,
            'triton': ChamferDistances.triton,
            'cuda_kd': ChamferDistances.cuda_kd,
        }

        if ChamferDistances.algo_interval_dict is not None:
            gpu_algo_dict['fusion'] = ChamferDistances.fusion

        if torch.cuda.is_available():
            return gpu_algo_dict

        return cpu_algo_dict

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
