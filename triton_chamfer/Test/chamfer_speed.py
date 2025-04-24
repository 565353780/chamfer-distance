import torch
import numpy as np
from tqdm import trange

import mash_cpp

from triton_chamfer.Method.chamfer_sp import chamfer_sp
from triton_chamfer.Method.chamfer_triton import chamfer_triton
from triton_chamfer.Method.chamfer_triton_kd import chamfer_triton_kd
from triton_chamfer.Module.timer import Timer


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x],
                               grad_outputs=grad_outputs,
                               create_graph=True)[0]
    return grad

def checkResults(func, xyz1: torch.Tensor, xyz2: torch.Tensor) -> bool:
    dist1, idx1, dist2, idx2 = func(xyz1, xyz2)
    dist1_mashcpp, dist2_mashcpp, idx1_mashcpp, idx2_mashcpp = mash_cpp.toChamferDistance(xyz1[None, ...], xyz2[None, ...])

    def test_loss(d1: torch.Tensor, d2: torch.Tensor):
        return d1.mean() - d2.sum()

    loss = test_loss(dist1, dist2)
    loss_mashcpp = test_loss(dist1_mashcpp, dist2_mashcpp)

    d_xyz1 = gradient(loss, xyz1)
    d_xyz2 = gradient(loss, xyz2)
    d_xyz1_mashcpp = gradient(loss_mashcpp, xyz1)
    d_xyz2_mashcpp = gradient(loss_mashcpp, xyz2)

    assert torch.allclose(d_xyz1_mashcpp, d_xyz1)
    assert torch.allclose(d_xyz2_mashcpp, d_xyz2)

    xyz1 = torch.randn(8192, 3).cuda()
    xyz2 = torch.randn(8192, 3).cuda()

    dist1, idx1, dist2, idx2 = chamfer_triton_kd(xyz1, xyz2)
    dist1_ref, idx1_ref, dist2_ref, idx2_ref = chamfer_sp(
        xyz1.cpu().numpy(),
        xyz2.cpu().numpy())

    assert np.isclose(dist1.cpu(), dist1_ref).sum() == len(xyz1)
    assert np.isclose(idx1.cpu(), idx1_ref).sum() == len(xyz1)
    assert np.isclose(dist2.cpu(), dist2_ref).sum() == len(xyz2)
    assert np.isclose(idx2.cpu(), idx2_ref).sum() == len(xyz2)
    return True

def get_func_fps(func_name: str, func, xyz1: torch.Tensor, xyz2: torch.Tensor, test_second: float = 3.0) -> float:
    print('start test speed of', func_name, '...')
    timer = Timer()
    calculate_num = 0
    for _ in trange(10000000):
        dist1, idx1, dist2, idx2 = func(xyz1, xyz2)
        if isinstance(dist1, torch.Tensor):
            mean = torch.mean(dist1)
        else:
            mean = np.mean(dist1)

        assert mean > 0
        calculate_num += 1

        spend_second = timer.now()
        if spend_second >= test_second:
            break

    fps = calculate_num / timer.now()

    print("dist1 mean = ", mean)
    return fps

def test():
    torch.manual_seed(0)

    xyz1 = torch.randn(3800, 3).cuda()
    xyz2 = torch.randn(4000, 3).cuda()
    test_second = 3.0

    xyz1.requires_grad_(True)
    xyz2.requires_grad_(True)

    print('start check results of chamfer_triton...')
    checkResults(chamfer_triton, xyz1, xyz2)
    print('checkResults passed!')

    print('start check results of chamfer_triton_kd...')
    checkResults(chamfer_triton_kd, xyz1, xyz2)
    print('checkResults passed!')

    chamfer_mashcpp_fps = get_func_fps('chamfer_mashcpp', mash_cpp.toChamferDistance, xyz1[None, ...], xyz2[None, ...], test_second)
    chamfer_sp_fps = get_func_fps('chamfer_sp', chamfer_sp, xyz1.cpu().detach().numpy(), xyz2.cpu().detach().numpy(), test_second)
    chamfer_triton_fps = get_func_fps('chamfer_triton', chamfer_triton, xyz1, xyz2, test_second)
    chamfer_triton_kd_fps = get_func_fps('chamfer_triton_kd', chamfer_triton_kd, xyz1, xyz2, test_second)

    print('fps list:')
    print('mashcpp:\t', chamfer_mashcpp_fps)
    print('sp:\t\t', chamfer_sp_fps)
    print('triton:\t\t', chamfer_triton_fps)
    print('triton_kd:\t', chamfer_triton_kd_fps)
    return True
