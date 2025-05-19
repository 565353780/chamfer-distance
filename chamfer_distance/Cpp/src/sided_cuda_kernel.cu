#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

// NM_KERNEL_TILE_SIZE controls the tile size used in shared memory for the
// NmDistanceKernel. This value impacts performance and can be tuned based on
// the specific GPU architecture and problem size. A larger tile size can
// increase parallelism but may also lead to register spilling or reduced
// occupancy if it exceeds available shared memory or registers. It's
// recommended to experiment with different values (e.g., powers of 2 like 256,
// 512, 1024) to find the optimal setting for your use case.
const int NM_KERNEL_TILE_SIZE = 512;

// THREADS_PER_BLOCK_VAL defines the number of threads per CUDA block.
// This value is crucial for balancing parallelism and resource utilization
// (registers, shared memory per block). It's often a power of 2 (e.g., 256,
// 512, 1024). The optimal value depends on the kernel's complexity and the
// target GPU.
const int THREADS_PER_BLOCK_VAL = 512;

// GRID_DIM_X_VAL specifies the number of blocks in the x-dimension of the CUDA
// grid for the NmDistanceKernel. The kernel uses a grid-stride loop for the
// batch dimension 'b', meaning gridDim.x determines the initial number of
// parallel batch computations. A common value for such loops is 32, but this
// can be tuned based on 'b' and GPU characteristics.
const unsigned int GRID_DIM_X_VAL = 32;

__global__ void NmDistanceKernel(const int b, const int n, const float *xyz,
                                 const int m, const float *xyz2, float *result,
                                 int *result_i) {
  const int batch = NM_KERNEL_TILE_SIZE;
  __shared__ float buf[NM_KERNEL_TILE_SIZE * 3];
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int k2 = 0; k2 < m; k2 += NM_KERNEL_TILE_SIZE) {
      int end_k = min(m, k2 + NM_KERNEL_TILE_SIZE) - k2;
      for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
        buf[j] = xyz2[(i * m + k2) * 3 + j];
      }
      __syncthreads();
      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
           j += blockDim.x * gridDim.y) {
        float x1 = xyz[(i * n + j) * 3 + 0];
        float y1 = xyz[(i * n + j) * 3 + 1];
        float z1 = xyz[(i * n + j) * 3 + 2];
        int best_i = 0;
        float best = 0;
        int end_ka = end_k - (end_k & 3);
        if (end_ka == NM_KERNEL_TILE_SIZE) {
          for (int k = 0; k < NM_KERNEL_TILE_SIZE; k += 4) {
            {
              float x2 = buf[k * 3 + 0] - x1;
              float y2 = buf[k * 3 + 1] - y1;
              float z2 = buf[k * 3 + 2] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }
            {
              float x2 = buf[k * 3 + 3] - x1;
              float y2 = buf[k * 3 + 4] - y1;
              float z2 = buf[k * 3 + 5] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }
            {
              float x2 = buf[k * 3 + 6] - x1;
              float y2 = buf[k * 3 + 7] - y1;
              float z2 = buf[k * 3 + 8] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }
            {
              float x2 = buf[k * 3 + 9] - x1;
              float y2 = buf[k * 3 + 10] - y1;
              float z2 = buf[k * 3 + 11] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
            {
              float x2 = buf[k * 3 + 0] - x1;
              float y2 = buf[k * 3 + 1] - y1;
              float z2 = buf[k * 3 + 2] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }
            {
              float x2 = buf[k * 3 + 3] - x1;
              float y2 = buf[k * 3 + 4] - y1;
              float z2 = buf[k * 3 + 5] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }
            {
              float x2 = buf[k * 3 + 6] - x1;
              float y2 = buf[k * 3 + 7] - y1;
              float z2 = buf[k * 3 + 8] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }
            {
              float x2 = buf[k * 3 + 9] - x1;
              float y2 = buf[k * 3 + 10] - y1;
              float z2 = buf[k * 3 + 11] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          float x2 = buf[k * 3 + 0] - x1;
          float y2 = buf[k * 3 + 1] - y1;
          float z2 = buf[k * 3 + 2] - z1;
          float d = x2 * x2 + y2 * y2 + z2 * z2;
          if (k == 0 || d < best) {
            best = d;
            best_i = k + k2;
          }
        }
        if (k2 == 0 || result[(i * n + j)] > best) {
          result[(i * n + j)] = best;
          result_i[(i * n + j)] = best_i;
        }
      }
      __syncthreads();
    }
  }
}

void sided_forward_cuda(const torch::Tensor &xyz1, const torch::Tensor &xyz2,
                        torch::Tensor &dist1, torch::Tensor &idx1) {

  const auto batch_size = xyz1.size(0);
  const auto n = xyz1.size(1); // N points in cloud1
  const auto m = xyz2.size(1); // M points in cloud2

  if (batch_size == 0 || n == 0 || m == 0) {
    // If any dimension is zero, there's no work to do.
    // Assuming output tensors are already correctly sized (e.g., to 0) or will
    // be handled by the caller.
    return;
  }

  const int threads_per_block_val =
      THREADS_PER_BLOCK_VAL; // Number of threads per block, matching original
                             // launch.

  // For grid_dim_x, the kernel uses a grid-stride loop: for (int i =
  // blockIdx.x; i < b; i += gridDim.x) This means grid_dim_x is the total
  // number of blocks launched in the x-dimension. 32 is a common fixed value
  // for such loops, as in the original launch.
  unsigned int grid_dim_x = GRID_DIM_X_VAL;

  // For grid_dim_y, it's related to n (number of points in the first cloud).
  // The kernel loop for j is: for (int j = threadIdx.x + blockIdx.y *
  // blockDim.x; j < n; j += blockDim.x * gridDim.y) blockDim.x is
  // threads_per_block_val. gridDim.y is the total number of blocks in the
  // y-dimension. Calculate the number of blocks needed in y-dim to cover all
  // 'n' points once by the threads in these blocks, then cap this value at 16
  // (the original fixed value for gridDim.y).
  unsigned int grid_dim_y_calculated =
      (unsigned int)((n + threads_per_block_val - 1) / threads_per_block_val);
  // Since n > 0 (due to the check above) and threads_per_block_val > 0,
  // grid_dim_y_calculated will be at least 1.
  unsigned int grid_dim_y = std::min(16u, grid_dim_y_calculated);
  // grid_dim_y will be at least 1 if n > 0.

  dim3 num_cuda_blocks(grid_dim_x, grid_dim_y, 1);
  // The kernel is structured for 1D blocks (blockDim.x = threads, blockDim.y =
  // 1, blockDim.z = 1).
  dim3 threads_per_block_dim(threads_per_block_val, 1, 1);

  NmDistanceKernel<<<num_cuda_blocks, threads_per_block_dim>>>(
      batch_size, n, xyz1.data_ptr<float>(), m, xyz2.data_ptr<float>(),
      dist1.data_ptr<float>(), idx1.data_ptr<int>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
    // THError("aborting");
  }
}
