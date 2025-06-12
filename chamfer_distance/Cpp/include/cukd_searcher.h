#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <torch/extension.h>
#include <unordered_map>
#include <vector>

// 前向声明
namespace cukd {
template <typename T> struct box_t;
}

// 定义OrderedPoint结构体
template <typename PointT> struct OrderedPoint {
  PointT position;
  int idx;
};

template <typename FloatT = float, typename PointT = float3,
          uint32_t THREAD_POOL = 16, uint32_t BATCH_SIZE_B = 32,
          uint32_t BATCH_SIZE_N = 16, uint32_t BATCH_SIZE_M = 16>
void buildKDTree(const torch::Tensor &input, void **d_nodes, void **d_bounds);

template <typename FloatT = float, typename PointT = float3,
          uint32_t THREAD_POOL = 16, uint32_t BATCH_SIZE_B = 32,
          uint32_t BATCH_SIZE_N = 16, uint32_t BATCH_SIZE_M = 16>
std::vector<torch::Tensor> queryKDTree(void *d_nodes, void *d_bounds,
                                       const torch::Tensor &query,
                                       const uint32_t &n_points);

class CUKDSearcher {
public:
  CUKDSearcher();

  ~CUKDSearcher();

  void addPoints(const torch::Tensor &points);

  std::vector<torch::Tensor> query(const torch::Tensor &points);

private:
  void releaseResources();

  // 释放CUDA资源
  void freeKDTreeResources(void *d_input, void *d_bounds);

private:
  bool initialized;

  int batch_size;
  u_int32_t n_points;
  void *d_nodes;
  void *d_bounds;
};
