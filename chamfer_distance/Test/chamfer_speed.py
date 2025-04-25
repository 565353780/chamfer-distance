import torch
import numpy as np
from tqdm import trange

import chamfer_cpp

from chamfer_distance.Method.chamfer_triton import chamfer_triton
from chamfer_distance.Method.chamfer_triton_kd import chamfer_triton_kd
from chamfer_distance.Module.timer import Timer


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x],
                               grad_outputs=grad_outputs,
                               create_graph=True)[0]
    return grad

def checkResults(func, xyz1: torch.Tensor, xyz2: torch.Tensor) -> bool:
    dist1, idx1, dist2, idx2 = func(xyz1, xyz2)
    dist1_cuda, dist2_cuda, idx1_cuda, idx2_cuda = chamfer_cpp.chamfer_cuda(xyz1[None, ...], xyz2[None, ...])

    def test_loss(d1: torch.Tensor, d2: torch.Tensor):
        return d1.mean() - d2.sum()

    loss = test_loss(dist1, dist2)
    loss_cuda = test_loss(dist1_cuda, dist2_cuda)

    d_xyz1 = gradient(loss, xyz1)
    d_xyz2 = gradient(loss, xyz2)
    d_xyz1_cuda = gradient(loss_cuda, xyz1)
    d_xyz2_cuda = gradient(loss_cuda, xyz2)

    assert torch.allclose(d_xyz1_cuda, d_xyz1, atol=1e-5), torch.max(torch.abs(d_xyz1_cuda - d_xyz1))
    assert torch.allclose(d_xyz2_cuda, d_xyz2, atol=1e-5), torch.max(torch.abs(d_xyz2_cuda - d_xyz2))

    xyz1 = torch.randn(8192, 3).cuda()
    xyz2 = torch.randn(8192, 3).cuda()

    dist1, idx1, dist2, idx2 = chamfer_triton_kd(xyz1, xyz2)
    dist1_ref, dist2_ref, idx1_ref, idx2_ref = chamfer_cpp.chamfer_cpu(
        xyz1[None, ...].cpu(),
        xyz2[None, ...].cpu())

    assert torch.allclose(dist1.cpu(), dist1_ref[0], atol=1e-5)
    assert torch.allclose(dist2.cpu(), dist2_ref[0], atol=1e-5)
    assert torch.all(idx1.cpu() == idx1_ref[0])
    assert torch.all(idx2.cpu() == idx2_ref[0])
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

        assert mean >= 0
        calculate_num += 1

        spend_second = timer.now()
        if spend_second >= test_second:
            break

    fps = calculate_num / timer.now()

    return fps

def test():
    torch.manual_seed(0)

    xyz1 = torch.randn(38000, 3).cuda()
    xyz2 = torch.randn(400000, 3).cuda()
    test_second = 3.0

    xyz1.requires_grad_(True)
    xyz2.requires_grad_(True)

    print('start check results of chamfer_triton...')
    checkResults(chamfer_triton, xyz1, xyz2)
    print('checkResults passed!')

    print('start check results of chamfer_triton_kd...')
    checkResults(chamfer_triton_kd, xyz1, xyz2)
    print('checkResults passed!')

    chamfer_cpu_fps = 0
    if xyz1.shape[0] * xyz2.shape[0] > 40000 ** 2:
        print('the size of input xyz are too large! will ignored the chamfer_cpu speed test!')
    else:
        try:
            chamfer_cpu_fps = get_func_fps('chamfer_cpu', chamfer_cpp.chamfer_cpu, xyz1[None, ...].cpu(), xyz2[None, ...].cpu(), test_second)
        except Exception as e:
            print('get_func_fps failed for chamfer_cpu! maybe the size of input xyz are too large!')
            print(e)
            pass
    chamfer_cuda_fps = get_func_fps('chamfer_cuda', chamfer_cpp.chamfer_cuda, xyz1[None, ...], xyz2[None, ...], test_second)
    chamfer_triton_fps = get_func_fps('chamfer_triton', chamfer_triton, xyz1, xyz2, test_second)
    chamfer_triton_kd_fps = get_func_fps('chamfer_triton_kd', chamfer_triton_kd, xyz1, xyz2, test_second)

    print('fps list:')
    print('cpu:\t\t', chamfer_cpu_fps)
    print('cuda:\t\t', chamfer_cuda_fps)
    print('triton:\t\t', chamfer_triton_fps)
    print('triton_kd:\t', chamfer_triton_kd_fps)
    return True
