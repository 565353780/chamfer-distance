import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from chamfer_distance.Demo.cukd_searcher import demo_search_best_param, demo_test_speed

if __name__ == "__main__":
    demo_search_best_param(
        xyz1_shapes=[4000, 1600, 3],
        xyz2_shapes=[4000, 1000, 3],
        iter_num=10,
        device="cuda:0",
    )
    exit()

    demo_test_speed(
        xyz1_shapes=[1, 100000, 3],
        xyz2_shapes=[1, 400000, 3],
        iter_num=10,
        device="cuda:0",
    )

    demo_test_speed(
        xyz1_shapes=[2, 100000, 3],
        xyz2_shapes=[2, 400000, 3],
        iter_num=10,
        device="cuda:0",
    )

    demo_test_speed(
        xyz1_shapes=[4, 100000, 3],
        xyz2_shapes=[4, 400000, 3],
        iter_num=10,
        device="cuda:0",
    )

    demo_test_speed(
        xyz1_shapes=[8, 100000, 3],
        xyz2_shapes=[8, 400000, 3],
        iter_num=10,
        device="cuda:0",
    )

    demo_test_speed(
        xyz1_shapes=[4000, 1600, 3],
        xyz2_shapes=[4000, 1000, 3],
        iter_num=10,
        device="cuda:0",
    )
