from chamfer_distance.Method.sample import sqrt_uniform_array
from chamfer_distance.Method.render import renderAlgoFPSMapDict, renderAlgoFPSMapDictCurve, renderBestAlgoFPSMapDict
from chamfer_distance.Module.chamfer_distances import ChamferDistances
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

def demo_best_speed():
    point_cloud_sizes_m = [100, 1000, 10000, 100000]
    point_cloud_sizes_n = [100, 1000, 10000, 100000]

    print("\n开始进行Chamfer算法性能对比测试...")
    algo_fps_map_dict = SpeedManager.getAlgoFPSMapDict(point_cloud_sizes_m, point_cloud_sizes_n)

    print("\n生成可视化结果...")
    renderBestAlgoFPSMapDict(algo_fps_map_dict)
    return True

def demo_best_speed_curve():
    min_calculation_num = 1e7
    max_calculation_num = 3e9

    algo_name_1 = 'cuda'
    algo_name_2 = 'triton'
    algo_name_3 = 'cuda_kd'

    while SpeedManager.getAlgosFPSDiffSimple(algo_name_1, algo_name_2, min_calculation_num) < 0:
        min_calculation_num /= 2

    while SpeedManager.getAlgosFPSDiffSimple(algo_name_2, algo_name_3, max_calculation_num) > 0:
        max_calculation_num *= 2

    if ChamferDistances.algo_interval_dict is None:
        print('current search interval: [', min_calculation_num, ',', max_calculation_num, ']')
        print('start search the equal fps point...')

        algo_12_equal_fps_point = SpeedManager.getAlgosEqualFPSPoint(
            algo_name_1, algo_name_2, min_calculation_num, max_calculation_num)

        print('algo_12_equal_fps_point:', algo_12_equal_fps_point)

        algo_23_equal_fps_point = SpeedManager.getAlgosEqualFPSPoint(
            algo_name_2, algo_name_3, min_calculation_num, max_calculation_num)

        print('algo_23_equal_fps_point:', algo_23_equal_fps_point)

        algo_interval_dict = {
            'cuda': [0, algo_12_equal_fps_point],
            'triton': [algo_12_equal_fps_point, algo_23_equal_fps_point],
            'cuda_kd': [algo_23_equal_fps_point, float('inf')],
        }

        SpeedManager.saveEqualFPSPoint(algo_interval_dict)
        ChamferDistances.loadFusionAlgo()
    else:
        algo_12_equal_fps_point = ChamferDistances.algo_interval_dict['triton'][0]
        algo_23_equal_fps_point = ChamferDistances.algo_interval_dict['cuda_kd'][0]

    calculation_nums_1 = sqrt_uniform_array(start=min_calculation_num, stop=algo_12_equal_fps_point, num=10).tolist()

    calculation_nums_2 = sqrt_uniform_array(start=algo_12_equal_fps_point, stop=algo_23_equal_fps_point, num=10).tolist()

    calculation_nums_3 = sqrt_uniform_array(start=algo_23_equal_fps_point, stop=max_calculation_num, num=10).tolist()

    calculation_nums = calculation_nums_1 + calculation_nums_2 + calculation_nums_3

    print("\n开始进行Chamfer算法性能对比测试...")
    algo_fps_map_dict = SpeedManager.getAlgoSimpleFPSMapDict(calculation_nums)

    print("\n生成可视化结果...")
    renderAlgoFPSMapDictCurve(algo_fps_map_dict)
    return True
