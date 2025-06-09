#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <torch/extension.h>
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
 * 该类使用CUDA KD-Tree实现高效的点云最近邻搜索
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
   * @param points 形状为[N, 3]的CUDA张量，表示N个3D点
   */
  void addPoints(const torch::Tensor &points);

  /**
   * 查询最近邻点
   *
   * @param points 形状为[M, 3]的CUDA张量，表示M个查询点
   * @return 包含两个张量的vector：距离和索引
   */
  std::vector<torch::Tensor> query(const torch::Tensor &points);

private:
  // 释放资源的私有方法
  void releaseResources();

  // 在CUDA设备上分配内存并构建KD树
  void *allocateAndBuildKDTree(const torch::Tensor &points, void **d_bounds);

  // 在CUDA设备上执行KD树查询
  std::vector<torch::Tensor> queryKDTree(void *d_input, void *d_bounds,
                                         const torch::Tensor &points,
                                         int numInput);

  // 释放CUDA资源
  void freeKDTreeResources(void *d_input, void *d_bounds);

private:
  // 点云维度
  int dimension;
  // 是否已初始化
  bool initialized;
  // 输入点数量
  uint32_t numInput;
  // 点云数据
  void *d_input;
  // 边界盒
  void *d_bounds;
};
