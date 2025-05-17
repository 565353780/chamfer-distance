import os
from typing import Union

from chamfer_distance.Config.path import (
    SIDED_ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH,
    CHAMFER_ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH,
)


def loadSidedAlgoIntervalDict(
    algo_equal_fps_point_txt_file_path: str = SIDED_ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH,
) -> Union[dict, None]:
    if not os.path.exists(algo_equal_fps_point_txt_file_path):
        if (
            algo_equal_fps_point_txt_file_path
            != SIDED_ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH
        ):
            print("[ERROR][io::loadSidedAlgoIntervalDict]")
            print("\t algo equal fps point txt file not exist!")
            print(
                "\t algo_equal_fps_point_txt_file_path:",
                algo_equal_fps_point_txt_file_path,
            )
        return None

    with open(algo_equal_fps_point_txt_file_path, "r") as f:
        lines = f.readlines()

    algo_interval_dict = {}
    for line in lines:
        if "|" not in line:
            continue

        data = line.split("\n")[0].split("|")
        algo_interval_dict[data[0]] = [float(data[1]), float(data[2])]

    return algo_interval_dict


def loadChamferAlgoIntervalDict(
    algo_equal_fps_point_txt_file_path: str = CHAMFER_ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH,
) -> Union[dict, None]:
    if not os.path.exists(algo_equal_fps_point_txt_file_path):
        if (
            algo_equal_fps_point_txt_file_path
            != CHAMFER_ALGO_EQUAL_FPS_POINT_TXT_FILE_PATH
        ):
            print("[ERROR][io::loadChamferAlgoIntervalDict]")
            print("\t algo equal fps point txt file not exist!")
            print(
                "\t algo_equal_fps_point_txt_file_path:",
                algo_equal_fps_point_txt_file_path,
            )
        return None

    with open(algo_equal_fps_point_txt_file_path, "r") as f:
        lines = f.readlines()

    algo_interval_dict = {}
    for line in lines:
        if "|" not in line:
            continue

        data = line.split("\n")[0].split("|")
        algo_interval_dict[data[0]] = [float(data[1]), float(data[2])]

    return algo_interval_dict
