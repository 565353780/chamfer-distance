import triton
import triton.language as tl

from chamfer_distance.Config.autotune import get_cuda_autotune_config


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
