import torch

import chamfer_cpp

from chamfer_distance.Method.grad import gradient


def checkResults(func1, func2, xyz1: torch.Tensor, xyz2: torch.Tensor) -> bool:
    dist11, dist12, idx11, idx12 = func1(xyz1, xyz2)
    dist21, dist22, idx21, idx22 = func2(xyz1[None, ...], xyz2[None, ...])

    def test_loss(d1: torch.Tensor, d2: torch.Tensor):
        return d1.mean() - d2.sum()

    loss = test_loss(dist11, dist12)
    loss_cuda = test_loss(dist21, dist22)

    d_xyz1 = gradient(loss, xyz1)
    d_xyz2 = gradient(loss, xyz2)
    d_xyz1_cuda = gradient(loss_cuda, xyz1)
    d_xyz2_cuda = gradient(loss_cuda, xyz2)

    assert torch.allclose(d_xyz1_cuda, d_xyz1, atol=1e-5), torch.max(torch.abs(d_xyz1_cuda - d_xyz1))
    assert torch.allclose(d_xyz2_cuda, d_xyz2, atol=1e-5), torch.max(torch.abs(d_xyz2_cuda - d_xyz2))

    xyz1 = torch.randn(8192, 3).cuda()
    xyz2 = torch.randn(8192, 3).cuda()

    dist1, dist2, idx1, idx2 = func1(xyz1, xyz2)
    dist1_ref, dist2_ref, idx1_ref, idx2_ref = chamfer_cpp.chamfer_cpu(
        xyz1[None, ...].cpu(),
        xyz2[None, ...].cpu())

    assert torch.allclose(dist1.cpu(), dist1_ref[0], atol=1e-5)
    assert torch.allclose(dist2.cpu(), dist2_ref[0], atol=1e-5)
    assert torch.all(idx1.cpu() == idx1_ref[0])
    assert torch.all(idx2.cpu() == idx2_ref[0])
    return True
