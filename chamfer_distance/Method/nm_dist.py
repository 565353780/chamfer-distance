import torch
import triton

from chamfer_distance.Method.kernel import nm_dist_kernel


def nm_dist(xyz1: torch.Tensor, xyz2: torch.Tensor):
    assert xyz1.shape[-1] == xyz2.shape[-1], "Incompatible dimensions"
    assert xyz1.is_contiguous(), "Matrix xyz1 must be contiguous"
    assert xyz2.is_contiguous(), "Matrix xyz2 must be contiguous"

    N = xyz1.shape[0]
    M = xyz2.shape[0]

    dists = torch.zeros((N,), device=xyz1.device, dtype=xyz1.dtype)
    indices = torch.zeros((N,), device=xyz1.device, dtype=torch.int32)
    # FIXME: The lock size is overkill
    lock = torch.zeros((N,), device=xyz1.device, dtype=torch.int32)

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
    )

    configs = {
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_M": 512,
        "GROUP_SIZE": 16,
        "num_warps": 2,
        "num_stages": 3,
    }

    nm_dist_kernel[grid](
        xyz1,
        xyz2,
        lock,
        dists,
        indices,
        N,
        M,
        xyz1.stride(0),
        xyz1.stride(1),
        xyz2.stride(0),
        xyz2.stride(1),
        dists.stride(0),
        indices.stride(0),
        lock.stride(0),
        **configs,
    )

    return dists, indices
