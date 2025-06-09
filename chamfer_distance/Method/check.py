import torch

from chamfer_distance.Method.grad import gradient


def checkChamferResults(func1, func2, xyz1: torch.Tensor, xyz2: torch.Tensor) -> bool:
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

    # FIXME: check dist1, dist2 grads about xyz1, xyz2 seperately
    loss_11 = dist11.mean()
    loss_12 = dist12.mean()
    loss_21 = dist21.mean()
    loss_22 = dist22.mean()

    assert loss_11 >= 0, print(loss_11)
    assert loss_12 >= 0, print(loss_12)
    assert loss_21 >= 0, print(loss_21)
    assert loss_22 >= 0, print(loss_22)

    assert torch.allclose(dist11, dist21, atol=1e-5), torch.max(
        torch.abs(dist11 - dist21)
    )

    assert torch.allclose(dist12, dist22, atol=1e-5), torch.max(
        torch.abs(dist12 - dist22)
    )

    not_match_idxs = torch.where(idx21 != idx11)
    if not_match_idxs[0].shape[0] > 0:
        assert torch.allclose(
            dist11[not_match_idxs], dist21[not_match_idxs], atol=1e-5
        ), torch.max(torch.abs(dist11[not_match_idxs] - dist21[not_match_idxs]))

    not_match_idxs = torch.where(idx22 != idx12)
    if not_match_idxs[0].shape[0] > 0:
        assert torch.allclose(
            dist12[not_match_idxs], dist22[not_match_idxs], atol=1e-5
        ), torch.max(torch.abs(dist12[not_match_idxs] - dist22[not_match_idxs]))

    if xyz1.requires_grad:
        d_xyz11 = gradient(loss_11, xyz1)
        d_xyz12 = gradient(loss_12, xyz1)
        d_xyz21 = gradient(loss_21, xyz1)
        d_xyz22 = gradient(loss_22, xyz1)

        assert torch.allclose(d_xyz21, d_xyz11, atol=1e-5), torch.max(
            torch.abs(d_xyz21 - d_xyz11)
        )

        assert torch.allclose(d_xyz22, d_xyz12, atol=1e-5), torch.max(
            torch.abs(d_xyz22 - d_xyz12)
        )

    if xyz2.requires_grad:
        d_xyz11 = gradient(loss_11, xyz2)
        d_xyz12 = gradient(loss_12, xyz2)
        d_xyz21 = gradient(loss_21, xyz2)
        d_xyz22 = gradient(loss_22, xyz2)

        assert torch.allclose(d_xyz21, d_xyz11, atol=1e-5), torch.max(
            torch.abs(d_xyz21 - d_xyz11)
        )

        assert torch.allclose(d_xyz22, d_xyz12, atol=1e-5), torch.max(
            torch.abs(d_xyz22 - d_xyz12)
        )

    return True
