#pragma once

#include <torch/extension.h>
#include <vector>

// 前向声明
template <typename PointT> struct OrderedPoint;

// CUKDSearcher类，用于在GPU上进行高效的最近邻搜索
class CUKDSearcher {
public:
  CUKDSearcher();
  ~CUKDSearcher();

  // 添加点云数据并构建KD树
  void addPoints(const torch::Tensor &points);

  // 查询最近邻点
  std::vector<torch::Tensor> query(const torch::Tensor &points);

private:
  // 释放资源
  void releaseResources();

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
