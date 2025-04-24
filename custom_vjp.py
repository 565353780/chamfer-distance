import numpy as np
import jax
from jax import numpy as jnp, vmap, jit, custom_jvp, grad, jvp
import jax_triton as jt
from chamfer_loss import nm_dist, chamfer_distance, closest_neighbour_sp, gradient
from jax import custom_vjp
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import triton
import triton.language as tl

from icecream import ic


@triton.jit
def nm_dist_kernel_loop(xyz1_ptr, xyz2_ptr, dist_ptr, indices_ptr, N, M,
                        xyz1_stride_n, xyz1_stride_d, xyz2_stride_m,
                        xyz2_stride_d, dist_stride_n, indices_stride_n,
                        BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_M: tl.constexpr):

    # TODO: More L2 cache friendly launch
    pid_n = tl.program_id(axis=0)
    base_n = pid_n * BLOCK_SIZE_N

    for pid_m in range((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M):
        base_m = pid_m * BLOCK_SIZE_M

        batch_base_n = base_n + tl.arange(0, BLOCK_SIZE_N)
        batch_n_mask = batch_base_n < N
        xyz1_x = tl.load(xyz1_ptr + batch_base_n * xyz1_stride_n,
                         mask=batch_n_mask,
                         other=100)
        xyz1_y = tl.load(xyz1_ptr + batch_base_n * xyz1_stride_n +
                         xyz1_stride_d,
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
        xyz2_y = tl.load(xyz2_ptr + batch_base_m * xyz2_stride_m +
                         xyz2_stride_d,
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
        best_idx = (tl.argmin(d, axis=1) + base_m).cast(tl.int32)

        if pid_m == 0:
            tl.store(dist_ptr + batch_base_n * dist_stride_n,
                     best_d,
                     mask=batch_n_mask)
            tl.store(indices_ptr + batch_base_n * indices_stride_n,
                     best_idx,
                     mask=batch_n_mask)
        else:
            cur_best_d = tl.load(dist_ptr + batch_base_n * dist_stride_n,
                                 mask=batch_n_mask)
            out_mask = (best_d < cur_best_d) & batch_n_mask
            tl.store(dist_ptr + batch_base_n * dist_stride_n,
                     best_d,
                     mask=out_mask)
            tl.store(indices_ptr + batch_base_n * indices_stride_n,
                     best_idx,
                     mask=out_mask)


def nm_dist_triton(xyz1: jnp.ndarray, xyz2: jnp.ndarray):
    N, D = xyz1.shape
    M, D = xyz2.shape

    lock = jnp.zeros((N, ), dtype=jnp.int32)

    dist_shape = jax.ShapeDtypeStruct(shape=(xyz1.shape[0], ),
                                      dtype=xyz1.dtype)
    idx_shape = jax.ShapeDtypeStruct(shape=(xyz1.shape[0], ), dtype=jnp.int32)

    metaparams = dict(BLOCK_SIZE_N=16,
                      BLOCK_SIZE_M=512,
                      num_warps=2,
                      num_stages=3)

    grid = (jt.cdiv(N, metaparams['BLOCK_SIZE_N']),
            jt.cdiv(M, metaparams['BLOCK_SIZE_M']))

    xyz1_stride = jt.strides_from_shape(xyz1.shape)
    xyz2_stride = jt.strides_from_shape(xyz2.shape)
    lock_stride = jt.strides_from_shape(lock.shape)

    return jt.triton_call(xyz1,
                          xyz2,
                          N=N,
                          M=M,
                          xyz1_stride_n=xyz1_stride[0],
                          xyz1_stride_d=xyz1_stride[1],
                          xyz2_stride_m=xyz2_stride[0],
                          xyz2_stride_d=xyz2_stride[1],
                          dist_stride_n=lock_stride[0],
                          indices_stride_n=lock_stride[0],
                          kernel=nm_dist_kernel_loop,
                          out_shape=[dist_shape, idx_shape],
                          grid=grid,
                          **metaparams)


@custom_vjp
@jit
def chamfer_distance_jax(xyz1: jax.Array, xyz2: jax.Array):
    dist1, idx1 = nm_dist_triton(xyz1, xyz2)
    dist2, idx2 = nm_dist_triton(xyz2, xyz1)
    return dist1, idx1, dist2, idx2


@jit
def chamfer_distance_jax_fwd(xyz1: jax.Array, xyz2: jax.Array):
    dist1, idx1, dist2, idx2 = chamfer_distance_jax(xyz1, xyz2)
    return (dist1, idx1, dist2, idx2), (xyz1, idx1, xyz2, idx2)


@jit
def chamfer_distance_jax_bwd(res, dz):
    xyz1, idx1, xyz2, idx2 = res
    dz1, _, dz2, _ = dz

    d_dist1 = dz1[:, None] * 2 * (xyz1 - xyz2[idx1])
    d_dist2 = dz2[:, None] * 2 * (xyz2 - xyz1[idx2])

    dxyz1 = d_dist1.at[idx2, :].add(-d_dist2)
    dxyz2 = d_dist2.at[idx1, :].add(-d_dist1)

    return (dxyz1, dxyz2)


chamfer_distance_jax.defvjp(chamfer_distance_jax_fwd, chamfer_distance_jax_bwd)

if __name__ == "__main__":

    N = 8192
    M = 8192

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    xyz1 = jax.random.normal(key1, (N, 3))
    xyz2 = jax.random.normal(key2, (M, 3))

    xyz1_torch = torch.from_dlpack(xyz1)
    xyz1_torch.requires_grad_(True)
    xyz2_torch = torch.from_dlpack(xyz2)
    xyz2_torch.requires_grad_(True)

    # dist1, idx1, dist2, idx2 = chamfer_distance_jax(xyz1, xyz2)
    # dist1_ref, idx1_ref, dist2_ref, idx2_ref = chamfer_distance_sp(xyz1, xyz2)

    # ic(np.isclose(dist1, dist1_ref).sum() == len(xyz1))
    # ic(np.isclose(idx1, idx1_ref).sum() == len(xyz1))
    # ic(np.isclose(dist2, dist2_ref).sum() == len(xyz2))
    # ic(np.isclose(idx2, idx2_ref).sum() == len(xyz2))

    # exit()

    # def test_loss(xyz1, xyz2, chamfer_impl):
    #     dist1, idx1, dist2, idx2 = chamfer_impl(xyz1, xyz2)
    #     return dist1.sum() + dist2.sum()

    # d_xyz1, d_xyz2 = grad(test_loss, argnums=(0, 1))(xyz1, xyz2,
    #                                                  chamfer_distance_jax)

    # loss = test_loss(xyz1_torch, xyz2_torch, chamfer_distance)
    # d_xyz1_torch = gradient(loss, xyz1_torch)
    # d_xyz2_torch = gradient(loss, xyz2_torch)

    # ic(jnp.allclose(d_xyz1, jax.dlpack.from_dlpack(d_xyz1_torch.detach())))
    # ic(jnp.allclose(d_xyz2, jax.dlpack.from_dlpack(d_xyz2_torch.detach())))

    # exit()

    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=True) as prof:

        with record_function("Jax-Triton"):
            chamfer_distance_jax(xyz1, xyz2)

        with record_function("Pytorch-Triton"):
            chamfer_distance(xyz1_torch, xyz2_torch)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
