from chamfer_distance.Method.sample import sqrt_uniform_array
from chamfer_distance.Method.render import (
    renderAlgoFPSMapDict,
    renderAlgoFPSMapDictCurve,
    renderBestAlgoFPSMapDict,
)
from chamfer_distance.Module.chamfer_speed_manager import ChamferSpeedManager


def demo_fps():
    xyz1_shape = [1, 4000, 3]
    xyz2_shape = [1, 4000, 3]

    algo_fps_dict = ChamferSpeedManager.getAlgoFPSDict(xyz1_shape, xyz2_shape)

    print("fps:")
    for algo_name, algo_fps in algo_fps_dict.items():
        print(algo_name, ":", algo_fps)
    return True


def demo_balance():
    calculation_num = 10000**2
    split_num = 10
    max_unbalance_weight = 9.0

    print("\n开始进行Chamfer算法平衡性测试...")
    algo_fps_map_dict = ChamferSpeedManager.getAlgoBalanceFPSMapDict(
        calculation_num, split_num, max_unbalance_weight
    )

    print("\n生成可视化结果...")
    renderAlgoFPSMapDict(algo_fps_map_dict)
    return True


def demo_speed():
    point_cloud_sizes_m = [100, 1000, 10000, 100000]
    point_cloud_sizes_n = [100, 1000, 10000, 100000]

    print("\n开始进行Chamfer算法性能对比测试...")
    algo_fps_map_dict = ChamferSpeedManager.getAlgoFPSMapDict(
        point_cloud_sizes_m, point_cloud_sizes_n
    )

    print("\n生成可视化结果...")
    renderAlgoFPSMapDict(algo_fps_map_dict)
    return True


def demo_best_speed():
    point_cloud_sizes_m = [100, 1000, 10000, 100000]
    point_cloud_sizes_n = [100, 1000, 10000, 100000]

    print("\n开始进行Chamfer算法性能对比测试...")
    algo_fps_map_dict = ChamferSpeedManager.getAlgoFPSMapDict(
        point_cloud_sizes_m, point_cloud_sizes_n
    )

    print("\n生成可视化结果...")
    renderBestAlgoFPSMapDict(algo_fps_map_dict)
    return True


def demo_best_speed_curve():
    min_calculation_num = 1e7
    max_calculation_num = 3e9
    record_num = 10

    algo_name_1 = "cuda"
    algo_name_2 = "triton"
    algo_name_3 = "cukd"

    while (
        ChamferSpeedManager.getAlgosFPSDiffSimple(
            algo_name_1, algo_name_2, min_calculation_num
        )
        < 0
    ):
        min_calculation_num /= 2

    while (
        ChamferSpeedManager.getAlgosFPSDiffSimple(
            algo_name_2, algo_name_3, max_calculation_num
        )
        > 0
    ):
        max_calculation_num *= 2

    calculation_nums = sqrt_uniform_array(
        min_calculation_num, max_calculation_num, record_num
    )

    print("\n开始进行Chamfer算法性能对比测试...")
    algo_fps_map_dict = ChamferSpeedManager.getAlgoSimpleFPSMapDict(calculation_nums)

    print("\n生成可视化结果...")
    renderAlgoFPSMapDictCurve(algo_fps_map_dict)
    return True
