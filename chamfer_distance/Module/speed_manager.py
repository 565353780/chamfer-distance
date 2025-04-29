import torch
import numpy as np
from tqdm import trange

from chamfer_distance.Module.chamfer_distances import ChamferDistances
from chamfer_distance.Module.timer import Timer


class SpeedManager(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def getAlgoFPS(
        algo_name: str,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        test_second: float = 1.0,
    ) -> float:
        algo_func = ChamferDistances.namedAlgo(algo_name)
        if algo_func is None:
            print('[ERROR][SpeedManager::getAlgoFPS]')
            print('\t namedAlgo failed!')
            return 0.0

        if algo_name == 'cpu':
            if xyz1.shape[0] * xyz1.shape[1] * xyz2.shape[0] * xyz2.shape[1] > 40000 ** 2:
                print('[WARN][SpeedManager::getAlgoFPS]')
                print('\t data too large for cpu calculation! will return fps = 0.0!')
                return 0.0

        print('[INFO][SpeedManager]')
        print('\t start test speed of [' + algo_name + ']...')
        timer = Timer()
        calculate_num = 0
        for _ in trange(10000000):
            dist1, dist2, idx1, idx2 = algo_func(xyz1, xyz2)
            if isinstance(dist1, torch.Tensor):
                mean = torch.mean(dist1)
            else:
                mean = np.mean(dist1)

            assert mean >= 0
            calculate_num += 1

            spend_second = timer.now()
            if spend_second >= test_second:
                break

        fps = calculate_num / timer.now()

        return fps
