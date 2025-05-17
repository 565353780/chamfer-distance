from chamfer_distance.Demo.base_searcher import demo_test_searcher_speed
from chamfer_distance.Module.faiss_searcher import FAISSSearcher


def demo():
    demo_test_searcher_speed(
        FAISSSearcher,
        "faiss",
        [300000, 400000],
        100,
    )
    return True
