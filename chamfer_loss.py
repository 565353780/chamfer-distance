import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from scipy.spatial.distance import cdist
from typing import List

import triton
import triton.language as tl
import triton_chamfer

from icecream import ic


def chamfer_distance_sp(ref_pts, query_pts):
    dist_mat = cdist(ref_pts, query_pts, metric='sqeuclidean')
    ref_closest_index_sp = np.argmin(dist_mat, axis=1)
    ref_closest_dist_sp = np.min(dist_mat, axis=1)
    query_closest_index_sp = np.argmin(dist_mat, axis=0)
    query_closest_dist_sp = np.min(dist_mat, axis=0)

    return ref_closest_dist_sp, ref_closest_index_sp, query_closest_dist_sp, query_closest_index_sp


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                'BLOCK_SIZE_N': 16,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 1
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 16,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 1
            },
            num_warps=2,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 16,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 16,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=2,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=2,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 32
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 32
            },
            num_warps=2,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=2,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 32
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 32
            },
            num_warps=2,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 64
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 64
            },
            num_warps=2,
            num_stages=3),
    ]


# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=['M', 'N']
#     )
@triton.jit
def nm_dist_kernel(xyz1_ptr, xyz2_ptr, lock_ptr, dists_ptr, indices_ptr, N, M,
                   xyz1_stride_n, xyz1_stride_d, xyz2_stride_m, xyz2_stride_d,
                   dist_stride_n, indices_stride_n, lock_stride,
                   BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                   GROUP_SIZE: tl.constexpr):

    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # More L2 cache friendly launch
    num_pid_n = tl.num_programs(axis=0)
    num_pid_m = tl.num_programs(axis=1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_SIZE)

    base_n = pid_n * BLOCK_SIZE_N
    base_m = pid_m * BLOCK_SIZE_M

    batch_base_n = base_n + tl.arange(0, BLOCK_SIZE_N)
    batch_n_mask = batch_base_n < N
    xyz1_x = tl.load(xyz1_ptr + batch_base_n * xyz1_stride_n,
                     mask=batch_n_mask,
                     other=100)
    xyz1_y = tl.load(xyz1_ptr + batch_base_n * xyz1_stride_n + xyz1_stride_d,
                     mask=batch_n_mask,
                     other=100)
    xyz1_z = tl.load(xyz1_ptr + batch_base_n * xyz1_stride_n +
                     2 * xyz1_stride_d,
                     mask=batch_n_mask,
                     other=100)

    batch_base_m = base_m + tl.arange(0, BLOCK_SIZE_M)
    batch_m_mask = batch_base_m < M
    xyz2_x = tl.load(xyz2_ptr + batch_base_m * xyz2_stride_m,
                     mask=batch_m_mask,
                     other=-100)
    xyz2_y = tl.load(xyz2_ptr + batch_base_m * xyz2_stride_m + xyz2_stride_d,
                     mask=batch_m_mask,
                     other=-100)
    xyz2_z = tl.load(xyz2_ptr + batch_base_m * xyz2_stride_m +
                     2 * xyz2_stride_d,
                     mask=batch_m_mask,
                     other=-100)

    x2 = xyz1_x[:, None] - xyz2_x[None, :]
    y2 = xyz1_y[:, None] - xyz2_y[None, :]
    z2 = xyz1_z[:, None] - xyz2_z[None, :]
    d = x2 * x2 + y2 * y2 + z2 * z2

    best_d = tl.min(d, axis=1)

    # ​TODO: sqrt depends on SLU. Let pytorch handle it for now
    # best_d = tl.sqrt(best_d)
    best_idx = tl.argmin(d, axis=1) + base_m

    lock = lock_ptr + pid_n * lock_stride
    while tl.atomic_cas(lock, 0, 1) == 1:
        pass

    cur_best_d = tl.load(dists_ptr + batch_base_n * dist_stride_n,
                         mask=batch_n_mask)
    # Handle zero initialization in JAX
    # FIXME: The safer option is to use another lock for first occuring pid_n initialization
    out_mask = ((best_d < cur_best_d) | (cur_best_d == 0)) & batch_n_mask
    tl.store(dists_ptr + batch_base_n * dist_stride_n, best_d, mask=out_mask)
    tl.store(indices_ptr + batch_base_n * indices_stride_n,
             best_idx,
             mask=out_mask)

    # Release lock
    tl.atomic_xchg(lock, 0)


def nm_dist(xyz1: torch.Tensor, xyz2: torch.Tensor):
    assert xyz1.shape[-1] == xyz2.shape[-1], "Incompatible dimensions"
    assert xyz1.is_contiguous(), "Matrix xyz1 must be contiguous"
    assert xyz2.is_contiguous(), "Matrix xyz2 must be contiguous"

    N, D = xyz1.shape
    M, D = xyz2.shape

    dists = torch.zeros((N, ), device=xyz1.device, dtype=xyz1.dtype)
    indices = torch.zeros((N, ), device=xyz1.device, dtype=torch.int32)
    # FIXME: The lock size is overkill
    lock = torch.zeros((N, ), device=xyz1.device, dtype=torch.int32)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']),
                         triton.cdiv(M, META['BLOCK_SIZE_M']))

    configs = {
        'BLOCK_SIZE_N': 16,
        'BLOCK_SIZE_M': 512,
        'GROUP_SIZE': 16,
        "num_warps": 2,
        "num_stages": 3
    }

    nm_dist_kernel[grid](xyz1, xyz2, lock, dists, indices, N, M,
                         xyz1.stride(0), xyz1.stride(1), xyz2.stride(0),
                         xyz2.stride(1), dists.stride(0), indices.stride(0),
                         lock.stride(0), **configs)

    return dists, indices


@torch.library.custom_op("mash::chamfer_distance", mutates_args=())
def chamfer_distance(xyz1: torch.Tensor,
                     xyz2: torch.Tensor) -> List[torch.Tensor]:
    dist1, idx1 = nm_dist(xyz1, xyz2)
    dist2, idx2 = nm_dist(xyz2, xyz1)
    return dist1, idx1, dist2, idx2


@chamfer_distance.register_fake
def _(xyz1: torch.Tensor, xyz2: torch.Tensor):
    N = xyz1.shape[0]
    M = xyz2.shape[0]
    return xyz1.new_empty((N, )), xyz1.new_empty(
        (N, ), dtype=torch.int32), xyz2.new_empty((M, )), xyz2.new_empty(
            (M, ), dtype=torch.int32)


def chamfer_distance_setup_context(ctx, inputs, output):
    xyz1, xyz2 = inputs
    _, idx1, _, idx2 = output
    ctx.save_for_backward(xyz1, idx1, xyz2, idx2)


def chamfer_distance_backward(ctx, grad_out):
    xyz1, idx1, xyz2, idx2 = ctx.saved_tensors
    grad_dist1, _, grad_dist2, _ = grad_out

    d_dist1 = grad_dist1[:, None] * 2 * (xyz1 - xyz2[idx1])
    d_dist2 = grad_dist2[:, None] * 2 * (xyz2 - xyz1[idx2])

    grad_xyz1 = torch.scatter_add(d_dist1, 0,
                                  idx2[:, None].expand(-1, 3).long(), -d_dist2)
    grad_xyz2 = torch.scatter_add(d_dist2, 0,
                                  idx1[:, None].expand(-1, 3).long(), -d_dist1)
    return grad_xyz1, grad_xyz2


chamfer_distance.register_autograd(
    chamfer_distance_backward, setup_context=chamfer_distance_setup_context)


@torch.library.custom_op("mash::chamfer_distance_kd", mutates_args=())
def chamfer_distance_kd(xyz1: torch.Tensor,
                        xyz2: torch.Tensor) -> List[torch.Tensor]:
    dist1, idx1 = triton_chamfer.ops.kd_closest_query(xyz1, xyz2)
    dist2, idx2 = triton_chamfer.ops.kd_closest_query(xyz2, xyz1)
    return dist1, idx1, dist2, idx2


chamfer_distance_kd.register_autograd(
    chamfer_distance_backward, setup_context=chamfer_distance_setup_context)


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x],
                               grad_outputs=grad_outputs,
                               create_graph=True)[0]
    return grad


if __name__ == "__main__":
    import mash_cpp

    torch.manual_seed(0)

    xyz1 = torch.randn(868, 3).cuda()
    xyz1.requires_grad_(True)
    xyz2 = torch.randn(2976, 3).cuda()
    xyz2.requires_grad_(True)

    dist1, idx1, dist2, idx2 = chamfer_distance_kd(xyz1, xyz2)
    dist1_mashcpp, dist2_mashcpp, idx1_mashcpp, idx2_mashcpp = mash_cpp.toChamferDistance(
        xyz1[None, ...], xyz2[None, ...])

    def test_loss(d1: torch.Tensor, d2: torch.Tensor):
        return d1.mean() - d2.sum()

    loss = test_loss(dist1, dist2)
    loss_mashcpp = test_loss(dist1_mashcpp, dist2_mashcpp)

    d_xyz1 = gradient(loss, xyz1)
    d_xyz2 = gradient(loss, xyz2)
    d_xyz1_mashcpp = gradient(loss_mashcpp, xyz1)
    d_xyz2_mashcpp = gradient(loss_mashcpp, xyz2)

    ic(torch.allclose(d_xyz1_mashcpp, d_xyz1))
    ic(torch.allclose(d_xyz2_mashcpp, d_xyz2))
    exit()

    # xyz1 = torch.randn(8192, 3).cuda()
    # xyz2 = torch.randn(8192, 3).cuda()

    # dist1, idx1, dist2, idx2 = chamfer_distance_kd(xyz1, xyz2)
    # dist1_ref, idx1_ref, dist2_ref, idx2_ref = chamfer_distance_sp(
    #     xyz1.cpu().numpy(),
    #     xyz2.cpu().numpy())

    # ic(np.isclose(dist1.cpu(), dist1_ref).sum() == len(xyz1))
    # ic(np.isclose(idx1.cpu(), idx1_ref).sum() == len(xyz1))
    # ic(np.isclose(dist2.cpu(), dist2_ref).sum() == len(xyz2))
    # ic(np.isclose(idx2.cpu(), idx2_ref).sum() == len(xyz2))

    # exit()

    # num_pts = 50000

    # xyz1 = torch.randn(num_pts, 3).cuda()
    # xyz2 = torch.randn(num_pts, 3).cuda()

    # mash_cpp.toChamferDistance(xyz1[None, ...], xyz2[None, ...])

    # triton_out = nm_dist(xyz1, xyz2)
    # triton_out2 = nm_dist(xyz2, xyz1)

    # exit()

    # num_pts = 400000

    # with profile(activities=[ProfilerActivity.CUDA],
    #              record_shapes=True) as prof:
    #     xyz1 = torch.randn(num_pts, 3).cuda()
    #     xyz1.requires_grad_(True)
    #     xyz2 = torch.randn(num_pts, 3).cuda()

    #     with record_function("Triton"):
    #         triton_out = chamfer_distance(xyz1, xyz2)

    #     with record_function("KD"):
    #         kd_out = chamfer_distance_kd(xyz1, xyz2)

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["N", "M"],
            x_vals=[4 * np.power(10, i) for i in range(1, 6)],
            line_arg="provider",
            line_vals=["Triton", "KD"],
            line_names=["Triton", "KD"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name="NMDist Performance",
            args={}))

    @triton.testing.perf_report(configs)
    def benchmark(M, N, provider):
        xyz1 = torch.randn(M, 3).cuda()
        xyz2 = torch.randn(N, 3).cuda()
        quantiles = [0.5, 0.2, 0.8]
        if provider == "KD":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: chamfer_distance_kd(xyz1, xyz2), quantiles=quantiles)
        if provider == "Triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: chamfer_distance(xyz1, xyz2), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True)
