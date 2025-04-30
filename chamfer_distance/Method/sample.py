import numpy as np


def sqrt_uniform_array(start: float, stop: float, num: int):
    sqrt_start = np.sqrt(start)
    sqrt_stop = np.sqrt(stop)
    sqrt_values = np.linspace(sqrt_start, sqrt_stop, num)
    return sqrt_values ** 2
