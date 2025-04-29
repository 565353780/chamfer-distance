import torch

from chamfer_distance.Method.render import renderAlgoFPSMapDict
from chamfer_distance.Module.chamfer_distances import ChamferDistances
from chamfer_distance.Module.speed_manager import SpeedManager


def demo():
    torch.manual_seed(0)

    xyz1_shape = [1, 4000, 3]
    xyz2_shape = [1, 4000, 3]

    ChamferDistances.check(xyz1_shape, xyz2_shape)

    algo_fps_dict = SpeedManager.getAlgoFPSDict(xyz1_shape, xyz2_shape)

    print('fps:')
    for algo_name, algo_fps in algo_fps_dict.items():
        print(algo_name, ':', algo_fps)

    point_cloud_sizes_m = [100, 1000, 10000, 100000]
    point_cloud_sizes_n = [100, 1000, 10000, 100000]

    print("\n开始进行Chamfer算法性能对比测试...")
    algo_fps_map_dict = SpeedManager.getAlgoFPSMapDict(point_cloud_sizes_m, point_cloud_sizes_n)

    print("\n生成可视化结果...")
    renderAlgoFPSMapDict(algo_fps_map_dict)
    return True
