import torch

from chamfer_distance.Method.grad import gradient


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

    assert torch.all(idx21 == idx11), print(
        idx11, "\n", idx21, "\n", torch.where(idx21 != idx11), "\n", idx11.shape
    )
    assert torch.all(idx22 == idx12), print(
        idx12, "\n", idx22, "\n", torch.where(idx22 != idx12), "\n", idx12.shape
    )

    assert torch.allclose(dist11, dist21, atol=1e-5), torch.max(
        torch.abs(dist11 - dist21)
    )
    assert torch.allclose(dist12, dist22, atol=1e-5), torch.max(
        torch.abs(dist12 - dist22)
    )

    assert torch.allclose(d_xyz21, d_xyz11, atol=1e-5), torch.max(
        torch.abs(d_xyz21 - d_xyz11)
    )
    assert torch.allclose(d_xyz22, d_xyz12, atol=1e-5), torch.max(
        torch.abs(d_xyz22 - d_xyz12)
    )
    return True
