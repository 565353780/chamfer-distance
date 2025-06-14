import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from chamfer_distance.Demo.cukd_searcher import demo

if __name__ == "__main__":
    demo(
        xyz1_shapes=[1, 100000, 3],
        xyz2_shapes=[1, 400000, 3],
        iter_num=10,
        device="cuda:0",
    )

    demo(
        xyz1_shapes=[2, 100000, 3],
        xyz2_shapes=[2, 400000, 3],
        iter_num=10,
        device="cuda:0",
    )

    demo(
        xyz1_shapes=[3, 100000, 3],
        xyz2_shapes=[3, 400000, 3],
        iter_num=10,
        device="cuda:0",
    )

    demo(
        xyz1_shapes=[4, 100000, 3],
        xyz2_shapes=[4, 400000, 3],
        iter_num=10,
        device="cuda:0",
    )

    exit()
    demo(
        xyz1_shapes=[4000, 1600, 3],
        xyz2_shapes=[4000, 1000, 3],
        iter_num=10,
        device="cuda:0",
    )
