import torch

from chamfer_distance.Method.grad import gradient
from chamfer_distance.Function.torch import chamfer_torch


def checkResults(func1, func2, xyz1: torch.Tensor, xyz2: torch.Tensor) -> bool:
    dist11, dist12, idx11, idx12 = func1(xyz1, xyz2)
    dist21, dist22, idx21, idx22 = func2(xyz1, xyz2)

    if dist11.shape[0] == 0:
        print("[ERROR][check::checkResults]")
        print("\t func1 call failed!")
        return False
    if dist21.shape[0] == 0:
        print("[ERROR][check::checkResults]")
        print("\t func2 call failed!")
        return False

    def test_loss(d1: torch.Tensor, d2: torch.Tensor):
        return d1.mean() - d2.sum()

    loss_1 = test_loss(dist11, dist12)
    loss_2 = test_loss(dist21, dist22)

    d_xyz11 = gradient(loss_1, xyz1)
    d_xyz12 = gradient(loss_1, xyz2)
    d_xyz21 = gradient(loss_2, xyz1)
    d_xyz22 = gradient(loss_2, xyz2)

    assert torch.allclose(idx21, idx11, atol=1e-5), torch.max(torch.abs(idx21 - idx11))
    assert torch.allclose(idx22, idx12, atol=1e-5), torch.max(torch.abs(idx22 - idx12))

    assert torch.allclose(d_xyz21, d_xyz11, atol=1e-5), torch.max(
        torch.abs(d_xyz21 - d_xyz11)
    )
    assert torch.allclose(d_xyz22, d_xyz12, atol=1e-5), torch.max(
        torch.abs(d_xyz22 - d_xyz12)
    )

    xyz1 = torch.randn(1, 8192, 3).cuda()
    xyz2 = torch.randn(1, 8192, 3).cuda()

    dist1, dist2, idx1, idx2 = func1(xyz1, xyz2)
    dist1_ref, dist2_ref, idx1_ref, idx2_ref = chamfer_torch(xyz1, xyz2)

    assert torch.allclose(dist1, dist1_ref, atol=1e-5)
    assert torch.allclose(dist2, dist2_ref, atol=1e-5)
    assert torch.all(idx1 == idx1_ref)
    assert torch.all(idx2 == idx2_ref)
    return True
