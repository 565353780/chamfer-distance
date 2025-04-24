import numpy as np
from scipy.spatial.distance import cdist

import jax
from jax import numpy as jnp, jit, vmap
from jax.experimental import pallas as pl

from icecream import ic


def chamfer_distance_sp(ref_pts, query_pts):
    dist_mat = cdist(ref_pts, query_pts, metric='sqeuclidean')
    ref_closest_index_sp = np.argmin(dist_mat, axis=1)
    ref_closest_dist_sp = np.min(dist_mat, axis=1)
    query_closest_index_sp = np.argmin(dist_mat, axis=0)
    query_closest_dist_sp = np.min(dist_mat, axis=0)

    return ref_closest_dist_sp, ref_closest_index_sp, query_closest_dist_sp, query_closest_index_sp


BLOCK_SIZE_N = 16
BLOCK_SIZE_M = 16

@jit
def nmdist_kernel_pallas(
    xyz1_ref,
    xyz2_ref,
    lock_ref,
    dist_ref,
    idx_ref
):
    pid_n = pl.program_id(axis=0)
    pid_m = pl.program_id(axis=1)

    base_n = pid_n * BLOCK_SIZE_N
    base_m = pid_m * BLOCK_SIZE_M

    batch_base_n = base_n + jnp.arange(0, BLOCK_SIZE_N)
    batch_n_mask = (batch_base_n < N)[:, None]

    batch_base_m = base_m + jnp.arange(0, BLOCK_SIZE_M)
    batch_m_mask = (batch_base_m < M)[:, None]

    xyz1_x = pl.load(xyz1_ref,
                     (pl.dslice(base_n, BLOCK_SIZE_N), pl.dslice(0, 1)),
                     mask=batch_n_mask,
                     other=100)
    xyz1_y = pl.load(xyz1_ref,
                     (pl.dslice(base_n, BLOCK_SIZE_N), pl.dslice(1, 1)),
                     mask=batch_n_mask,
                     other=100)
    xyz1_z = pl.load(xyz1_ref,
                     (pl.dslice(base_n, BLOCK_SIZE_N), pl.dslice(2, 1)),
                     mask=batch_n_mask,
                     other=100)

    xyz2_x = pl.load(xyz2_ref,
                     (pl.dslice(base_m, BLOCK_SIZE_M), pl.dslice(0, 1)),
                     mask=batch_m_mask,
                     other=-100)

    xyz2_y = pl.load(xyz2_ref,
                     (pl.dslice(base_m, BLOCK_SIZE_M), pl.dslice(1, 1)),
                     mask=batch_m_mask,
                     other=-100)

    xyz2_z = pl.load(xyz2_ref,
                     (pl.dslice(base_m, BLOCK_SIZE_M), pl.dslice(2, 1)),
                     mask=batch_m_mask,
                     other=-100)

    x2 = xyz1_x[:, None, :] - xyz2_x[None, :, :]
    y2 = xyz1_y[:, None, :] - xyz2_y[None, :, :]
    z2 = xyz1_z[:, None, :] - xyz2_z[None, :, :]
    d = x2 * x2 + y2 * y2 + z2 * z2

    best_d = jnp.min(d, axis=1)
    best_idx = jnp.argmin(d, axis=1) + base_m

    # FIXME: Waiting for upstream fix: https://github.com/jax-ml/jax/issues/18909
    def _cond(_):
        return pl.atomic_cas(lock_ref, 0, 1) == 1
    jax.lax.while_loop(_cond, lambda a: a, 0)

    cur_best_d = pl.load(dist_ref, (pl.dslice(base_n, BLOCK_SIZE_N), ),
                             mask=batch_n_mask)
    out_mask = ((best_d < cur_best_d) | (cur_best_d == 0)) & batch_n_mask
    pl.store(dist_ref, (pl.dslice(base_n, BLOCK_SIZE_N), ),
                best_d,
                mask=out_mask)
    pl.store(idx_ref, (pl.dslice(base_n, BLOCK_SIZE_N), ),
                best_idx,
                mask=out_mask)

    pl.atomic_xchg(lock_ref, (), 0)


key1, key2 = jax.random.split(jax.random.PRNGKey(0))
xyz1 = jax.random.normal(key1, (17, 3))
xyz2 = jax.random.normal(key2, (33, 3))

N, D = xyz1.shape
M, D = xyz2.shape

lock = jnp.zeros((pl.cdiv(N, BLOCK_SIZE_N), ), dtype=jnp.int32)

dist, idx = pl.pallas_call(
    nmdist_kernel_pallas,
    grid=(pl.cdiv(N, BLOCK_SIZE_N), pl.cdiv(M, BLOCK_SIZE_M)),
    # FIXME: Why would I split input given I need to pl.load anywhere?
    # in_specs=[
    #     pl.BlockSpec((BLOCK_SIZE_N, D), lambda i, j: (i, j)),
    #     pl.BlockSpec((BLOCK_SIZE_M, D), lambda i, j: (i, j)),
    #     pl.BlockSpec((None, ), lambda i, j: i)
    # ],
    in_specs=[
        pl.BlockSpec(xyz1.shape),
        pl.BlockSpec(xyz2.shape),
        pl.BlockSpec((None, ), lambda i, j: i)
    ],
    # out_specs=[pl.BlockSpec((BLOCK_SIZE_N, )),
    #            pl.BlockSpec((BLOCK_SIZE_N, ))],
    out_shape=[
        jax.ShapeDtypeStruct((N, 1), jnp.float32),
        jax.ShapeDtypeStruct((N, 1), jnp.int32)
    ],
    # interpret=True,
    # debug=True,
)(xyz1, xyz2, lock)

dist = dist[:, 0]
idx = idx[:, 0]

ic(dist)
ic(idx)
