from chamfer_distance.Method.render import renderAlgoFPSMapDict
from chamfer_distance.Module.speed_manager import SpeedManager


def demo_fps():
    xyz1_shape = [1, 4000, 3]
    xyz2_shape = [1, 4000, 3]

    algo_fps_dict = SpeedManager.getAlgoFPSDict(xyz1_shape, xyz2_shape)

    print('fps:')
    for algo_name, algo_fps in algo_fps_dict.items():
        print(algo_name, ':', algo_fps)
    return True

def demo_balance():
    calculation_num = 10000 ** 2
    split_num = 10
    max_unbalance_weight = 9.0

    print("\n开始进行Chamfer算法平衡性测试...")
    algo_fps_map_dict = SpeedManager.getAlgoBalanceFPSMapDict(calculation_num, split_num, max_unbalance_weight)

    print("\n生成可视化结果...")
    renderAlgoFPSMapDict(algo_fps_map_dict)
    return True

def demo_speed():
    point_cloud_sizes_m = [100, 1000, 10000, 100000]
    point_cloud_sizes_n = [100, 1000, 10000, 100000]

    print("\n开始进行Chamfer算法性能对比测试...")
    algo_fps_map_dict = SpeedManager.getAlgoFPSMapDict(point_cloud_sizes_m, point_cloud_sizes_n)

    print("\n生成可视化结果...")
    renderAlgoFPSMapDict(algo_fps_map_dict)
    return True
