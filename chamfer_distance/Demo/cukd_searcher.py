from chamfer_distance.Demo.base_searcher import demo_test_searcher_speed
from chamfer_distance.Module.cukd_searcher import CUKDSearcher


def demo():
    demo_test_searcher_speed(
        CUKDSearcher,
        "cukd",
        [300000, 400000],
        100,
    )
    return True
