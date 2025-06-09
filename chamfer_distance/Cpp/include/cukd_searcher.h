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

// 使用模板参数定义点类型
using PointT = float3;

// 定义OrderedPoint结构体
template <typename T> struct OrderedPoint {
  T position;
  int idx;
};

/**
 * CUKDSearcher类 - 在GPU上进行高效的最近邻搜索
 *
 * 该类使用CUDA KD-Tree实现高效的点云最近邻搜索，支持批处理
 */
class CUKDSearcher {
public:
  /**
   * 构造函数
   */
  CUKDSearcher();

  /**
   * 析构函数
   */
  ~CUKDSearcher();

  /**
   * 添加点云数据并构建KD树
   *
   * @param points 形状为[B N 3]的CUDA张量，表示B批次的N个3D点
   */
  void addPoints(const torch::Tensor &points);

  /**
   * 查询最近邻点
   *
   * @param points 形状为[B M 3]的CUDA张量，表示B批次的M个查询点
   * @return 包含两个张量的vector：距离和索引
   */
  std::vector<torch::Tensor> query(const torch::Tensor &points);

private:
  // 释放资源的私有方法
  void releaseResources();

  // 在CUDA设备上分配内存并构建KD树
  void *allocateAndBuildKDTree(const torch::Tensor &points, void **d_bounds,
                               int batchIdx);

  // 在CUDA设备上执行KD树查询
  std::vector<torch::Tensor> queryKDTree(const std::vector<void *> &d_inputs,
                                         const std::vector<void *> &d_bounds,
                                         const torch::Tensor &points,
                                         const std::vector<int> &numInputs);

  // 释放CUDA资源
  void freeKDTreeResources(void *d_input, void *d_bounds);

private:
  // 点云维度
  int dimension;
  // 是否已初始化
  bool initialized;
  // 批次大小
  int batchSize;
  // 每个批次的输入点数量
  std::vector<int> numInputs;
  // 每个批次的点云数据
  std::vector<void *> d_inputs;
  // 每个批次的边界盒
  std::vector<void *> d_bounds;
};
