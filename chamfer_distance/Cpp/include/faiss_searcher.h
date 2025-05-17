#pragma once

#include <memory>
#include <torch/extension.h>

#ifdef USE_CUDA
// 前向声明
namespace faiss {
namespace gpu {
class StandardGpuResources;
class GpuIndexFlatL2;
} // namespace gpu
} // namespace faiss

// FAISSSearcher类，用于在GPU上进行最近邻搜索
class FAISSSearcher {
public:
  FAISSSearcher();
  ~FAISSSearcher();

  // 添加点云数据到索引
  void addPoints(const torch::Tensor &points); // [N, D]

  // 查询最近邻点
  std::vector<torch::Tensor>
  query(const torch::Tensor &points); // [M, D] -> [dist, idx]

private:
  std::unique_ptr<faiss::gpu::StandardGpuResources> res;
  std::unique_ptr<faiss::gpu::GpuIndexFlatL2> index;
  int dimension;
  bool initialized;
};
#endif
