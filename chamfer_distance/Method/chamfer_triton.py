import torch
import triton
import triton.language as tl

from chamfer_distance.Method.nm_dist import nm_dist


@triton.jit
def batch_nm_dist_kernel(
    xyz1_ptr,
    xyz2_ptr,
    lock_ptr,
    dists_ptr,
    indices_ptr,
    B,
    N,
    M,
    xyz1_stride_b,
    xyz1_stride_n,
    xyz1_stride_d,
    xyz2_stride_b,
    xyz2_stride_m,
    xyz2_stride_d,
    dists_stride_b,
    dists_stride_n,
    indices_stride_b,
    indices_stride_n,
    lock_stride_b,
    lock_stride_n,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # 获取程序ID
    pid_b = tl.program_id(axis=0)  # 批次维度
    pid_n = tl.program_id(axis=1)  # 点云1维度
    pid_m = tl.program_id(axis=2)  # 点云2维度

    # 计算基础索引
    base_b = pid_b * BLOCK_SIZE_B
    base_n = pid_n * BLOCK_SIZE_N
    base_m = pid_m * BLOCK_SIZE_M

    # 批次索引和掩码
    batch_base_b = base_b + tl.arange(0, BLOCK_SIZE_B)
    batch_b_mask = batch_base_b < B

    # 点云1索引和掩码
    batch_base_n = base_n + tl.arange(0, BLOCK_SIZE_N)
    batch_n_mask = batch_base_n < N

    # 点云2索引和掩码
    batch_base_m = base_m + tl.arange(0, BLOCK_SIZE_M)
    batch_m_mask = batch_base_m < M

    # 对每个批次和点云1中的点进行处理
    for bb in range(BLOCK_SIZE_B):
        b = batch_base_b[bb]
        if b >= B:
            break

        # 加载点云1的坐标
        xyz1_offset = b * xyz1_stride_b
        xyz1_x_list = []
        xyz1_y_list = []
        xyz1_z_list = []

        for nn in range(BLOCK_SIZE_N):
            n = batch_base_n[nn]
            if n >= N:
                continue

            # 计算点云1中点的偏移量
            offset = xyz1_offset + n * xyz1_stride_n
            xyz1_x_list.append(tl.load(xyz1_ptr + offset, other=100))
            xyz1_y_list.append(tl.load(xyz1_ptr + offset + xyz1_stride_d, other=100))
            xyz1_z_list.append(
                tl.load(xyz1_ptr + offset + 2 * xyz1_stride_d, other=100)
            )

        if len(xyz1_x_list) == 0:
            continue

        xyz1_x = tl.make_block(xyz1_x_list)
        xyz1_y = tl.make_block(xyz1_y_list)
        xyz1_z = tl.make_block(xyz1_z_list)

        # 加载点云2的坐标
        xyz2_offset = b * xyz2_stride_b
        xyz2_x_list = []
        xyz2_y_list = []
        xyz2_z_list = []

        for mm in range(BLOCK_SIZE_M):
            m = batch_base_m[mm]
            if m >= M:
                continue

            # 计算点云2中点的偏移量
            offset = xyz2_offset + m * xyz2_stride_m
            xyz2_x_list.append(tl.load(xyz2_ptr + offset, other=-100))
            xyz2_y_list.append(tl.load(xyz2_ptr + offset + xyz2_stride_d, other=-100))
            xyz2_z_list.append(
                tl.load(xyz2_ptr + offset + 2 * xyz2_stride_d, other=-100)
            )

        if len(xyz2_x_list) == 0:
            continue

        xyz2_x = tl.make_block(xyz2_x_list)
        xyz2_y = tl.make_block(xyz2_y_list)
        xyz2_z = tl.make_block(xyz2_z_list)

        # 计算距离
        for nn in range(len(xyz1_x_list)):
            best_d = float("inf")
            best_idx = 0

            for mm in range(len(xyz2_x_list)):
                x_diff = xyz1_x[nn] - xyz2_x[mm]
                y_diff = xyz1_y[nn] - xyz2_y[mm]
                z_diff = xyz1_z[nn] - xyz2_z[mm]
                d = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff

                if d < best_d:
                    best_d = d
                    best_idx = base_m + mm

            # 获取锁
            n = batch_base_n[nn]
            lock = lock_ptr + b * lock_stride_b + n * lock_stride_n
            while tl.atomic_cas(lock, 0, 1) == 1:
                pass

            # 更新最佳距离和索引
            dists_offset = b * dists_stride_b + n * dists_stride_n
            indices_offset = b * indices_stride_b + n * indices_stride_n

            cur_best_d = tl.load(dists_ptr + dists_offset)
            if best_d < cur_best_d or cur_best_d == 0:
                tl.store(dists_ptr + dists_offset, best_d)
                tl.store(indices_ptr + indices_offset, best_idx)

            # 释放锁
            tl.atomic_xchg(lock, 0)


def batch_nm_dist(xyz1: torch.Tensor, xyz2: torch.Tensor):
    """
    使用triton并行计算批量点云的最近邻距离

    Args:
        xyz1: 形状为BxNx3的点云
        xyz2: 形状为BxMx3的点云

    Returns:
        dists: 形状为BxN的距离张量
        indices: 形状为BxN的索引张量
    """
    assert xyz1.shape[-1] == xyz2.shape[-1] == 3, "点云维度必须为3"
    assert xyz1.is_contiguous(), "xyz1必须是连续的"
    assert xyz2.is_contiguous(), "xyz2必须是连续的"

    B = xyz1.shape[0]  # 批量大小
    N = xyz1.shape[1]  # 每个批次中xyz1的点数
    M = xyz2.shape[1]  # 每个批次中xyz2的点数

    # 创建输出张量
    dists = torch.zeros((B, N), device=xyz1.device, dtype=xyz1.dtype)
    indices = torch.zeros((B, N), device=xyz1.device, dtype=torch.int32)
    lock = torch.zeros((B, N), device=xyz1.device, dtype=torch.int32)

    # 设置grid和配置
    BLOCK_SIZE_B = min(8, B)  # 每个block处理的批次数
    BLOCK_SIZE_N = min(16, N)  # 每个block处理的点云1点数
    BLOCK_SIZE_M = min(32, M)  # 每个block处理的点云2点数
    GROUP_SIZE = 8

    grid = (
        triton.cdiv(B, BLOCK_SIZE_B),
        triton.cdiv(N, BLOCK_SIZE_N),
        triton.cdiv(M, BLOCK_SIZE_M),
    )

    # 调用triton kernel
    batch_nm_dist_kernel[grid](
        xyz1.contiguous(),
        xyz2.contiguous(),
        lock,
        dists,
        indices,
        B,
        N,
        M,
        xyz1.stride(0),
        xyz1.stride(1),
        xyz1.stride(2),
        xyz2.stride(0),
        xyz2.stride(1),
        xyz2.stride(2),
        dists.stride(0),
        dists.stride(1),
        indices.stride(0),
        indices.stride(1),
        lock.stride(0),
        lock.stride(1),
        BLOCK_SIZE_B,
        BLOCK_SIZE_N,
        BLOCK_SIZE_M,
        GROUP_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return dists, indices


def sided_triton(xyz1: torch.Tensor, xyz2: torch.Tensor):
    """
    计算从点云xyz1到点云xyz2的单侧Chamfer距离

    Args:
        xyz1: 形状为BxNx3的点云
        xyz2: 形状为BxMx3的点云

    Returns:
        dists1: 形状为BxN的距离张量
        idxs1: 形状为BxN的索引张量
    """
    # 使用批量并行计算
    return batch_nm_dist(xyz1, xyz2)
