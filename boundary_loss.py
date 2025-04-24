import jax
import time
import torch
import mash_cpp
from tqdm import trange

from chamfer_loss import chamfer_distance


class Timer:

    def __init__(self):
        self.reset()

    def log(self, msg):
        cur_time = time.time()
        print(f"{msg}: {cur_time - self.start_time}")
        self.start_time = cur_time

    def reset(self):
        self.start_time = time.time()


def test_loss(xyz1, xyz2, chamfer_impl):
    dist1, _, dist2, _ = chamfer_impl(xyz1, xyz2)
    return dist1.sum() + dist2.sum()


if __name__ == "__main__":
    torch.manual_seed(0)

    anchor_num = 800
    mask_boundary_sample_num = 90

    mask_boundary_phi_idxs = torch.repeat_interleave(
        torch.arange(anchor_num), mask_boundary_sample_num).cuda()
    boundary_pts = torch.randn(
        (anchor_num * mask_boundary_sample_num, 3)).cuda()

    # JIT it first for correct time measurement
    chamfer_distance(
        torch.randn((mask_boundary_sample_num, 3)).cuda(),
        torch.randn((mask_boundary_sample_num * (anchor_num - 1), 3)).cuda())

    def functor(i):
        mask_xyz1 = mask_boundary_phi_idxs == i
        mask_xyz2 = torch.logical_not(mask_xyz1)
        xyz1 = boundary_pts[mask_xyz1]
        xyz2 = boundary_pts[mask_xyz2]
        dist1, _, dist2, _ = chamfer_distance(xyz1, xyz2)
        return dist1.mean() + dist2.mean()

    timer = Timer()
    for _ in trange(100):
        mash_cpp.toBoundaryConnectLoss(anchor_num, boundary_pts,
                                    mask_boundary_phi_idxs)
    timer.log("MashCPP")

    for _ in trange(100):
        out = jax.tree_util.tree_map(lambda i: functor(i),
                                    list(torch.arange(anchor_num).cuda()))
    timer.log("JAX tree_map")

    for _ in trange(100):
        [functor(i) for i in range(anchor_num)]
    timer.log("List comprehension")

    for _ in trange(100):
        for i in range(anchor_num):
            functor(i)
    timer.log("For loop")
