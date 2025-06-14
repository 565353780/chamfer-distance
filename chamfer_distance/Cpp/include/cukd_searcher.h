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

template <typename FloatT, typename PointT>
void buildKDTree(const torch::Tensor &input, void **d_nodes, void **d_bounds,
                 const uint32_t &THREAD_POOL, const uint32_t &BATCH_SIZE_B,
                 const uint32_t &BATCH_SIZE_N);

template <typename FloatT, typename PointT>
void queryKDTree(void *d_nodes, void *d_bounds, const torch::Tensor &query,
                 const uint32_t &n_points, torch::Tensor &dists,
                 torch::Tensor &idxs, const uint32_t &BATCH_SIZE_B,
                 const uint32_t &BATCH_SIZE_N);

class CUKDSearcher {
public:
  CUKDSearcher();

  ~CUKDSearcher();

  void addPoints(const torch::Tensor &points, const uint32_t &THREAD_POOL = 16,
                 const uint32_t &BATCH_SIZE_B = 32,
                 const uint32_t &BATCH_SIZE_N = 16);

  std::vector<torch::Tensor> query(const torch::Tensor &points,
                                   const uint32_t &BATCH_SIZE_B = 32,
                                   const uint32_t &BATCH_SIZE_N = 16);

private:
  void releaseResources();

  // 释放CUDA资源
  void freeKDTreeResources(void *d_input, void *d_bounds);

private:
  bool initialized = false;

  int batch_size = 0;
  u_int32_t n_points = 0;
  void *d_nodes = nullptr;
  void *d_bounds = nullptr;
};
