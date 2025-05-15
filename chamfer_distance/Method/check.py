import torch

import chamfer_cpp

from chamfer_distance.Method.grad import gradient


def to_valid_tensor(source_tensor):
    valid_tensor = torch.nan_to_num(source_tensor, 0.0)
    return valid_tensor


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

    assert torch.allclose(d_xyz21, d_xyz11, atol=1e-5), torch.max(
        torch.abs(d_xyz21 - d_xyz11)
    )
    assert torch.allclose(d_xyz22, d_xyz12, atol=1e-5), torch.max(
        torch.abs(d_xyz22 - d_xyz12)
    )

    xyz1 = torch.randn(1, 8192, 3).cuda()
    xyz2 = torch.randn(1, 8192, 3).cuda()

    dist1, dist2, idx1, idx2 = func1(xyz1, xyz2)
    dist1_ref, dist2_ref, idx1_ref, idx2_ref = chamfer_cpp.chamfer_cpu(
        xyz1.cpu(), xyz2.cpu()
    )

    assert torch.allclose(dist1.cpu(), dist1_ref[0], atol=1e-5)
    assert torch.allclose(dist2.cpu(), dist2_ref[0], atol=1e-5)
    assert torch.all(idx1.cpu() == idx1_ref[0])
    assert torch.all(idx2.cpu() == idx2_ref[0])
    return True
