import triton


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
