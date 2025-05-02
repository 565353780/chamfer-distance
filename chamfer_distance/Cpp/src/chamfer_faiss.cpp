#include "chamfer_faiss.h"
#include "chamfer_backward.h"
#include "chamfer_cpu.h"
#include <ATen/cuda/CUDAContext.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

#ifdef USE_CUDA
int chamfer_faiss_forward(const torch::Tensor &xyz1, // [B, N, 3]
                          const torch::Tensor &xyz2, // [B, M, 3]
                          torch::Tensor &dist1,      // [B, N]
                          torch::Tensor &dist2,      // [B, M]
                          torch::Tensor &idx1,       // [B, N]
                          torch::Tensor &idx2) {     // [B, M]
  TORCH_CHECK(xyz1.is_cuda() && xyz2.is_cuda(), "Input must be CUDA");
  TORCH_CHECK(xyz1.dim() == 3 && xyz2.dim() == 3, "Input must be [B, N, 3]");

  const int B = xyz1.size(0);
  const int N = xyz1.size(1);
  const int M = xyz2.size(1);
  const int D = xyz1.size(2); // = 3

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 创建 FAISS 资源
  faiss::gpu::StandardGpuResources res;
  res.setDefaultStream(0, stream);

  // Flatten inputs
  auto xyz1_flat = xyz1.reshape({B * N, D}).contiguous();
  auto xyz2_flat = xyz2.reshape({B * M, D}).contiguous();

  auto dist1_flat = dist1.reshape({B * N}).contiguous();
  auto dist2_flat = dist2.reshape({B * M}).contiguous();
  auto idx1_flat = idx1.reshape({B * N}).contiguous();
  auto idx2_flat = idx2.reshape({B * M}).contiguous();

  // Build xyz2 as DB index
  faiss::gpu::GpuIndexFlatL2 index_x2(&res, D);
  index_x2.add(B * M, xyz2_flat.data_ptr<float>());

  // Query xyz1 against xyz2
  index_x2.search(
      B * N, xyz1_flat.data_ptr<float>(), 1, dist1_flat.data_ptr<float>(),
      reinterpret_cast<faiss::idx_t *>(idx1_flat.data_ptr<int64_t>()));

  // Build xyz1 as DB index
  faiss::gpu::GpuIndexFlatL2 index_x1(&res, D);
  index_x1.add(B * N, xyz1_flat.data_ptr<float>());

  // Query xyz2 against xyz1
  index_x1.search(
      B * M, xyz2_flat.data_ptr<float>(), 1, dist2_flat.data_ptr<float>(),
      reinterpret_cast<faiss::idx_t *>(idx2_flat.data_ptr<int64_t>()));

  return 0;
}

std::vector<torch::Tensor>
chamfer_FAISS_3DFunction::forward(torch::autograd::AutogradContext *ctx,
                                  const torch::Tensor &xyz1,
                                  const torch::Tensor &xyz2) {
  int batchsize = xyz1.size(0);
  int n = xyz1.size(1);
  int m = xyz2.size(1);

  torch::TensorOptions opts =
      torch::TensorOptions().dtype(xyz1.dtype()).device(xyz1.device());

  torch::TensorOptions idx_opts =
      torch::TensorOptions().dtype(torch::kInt64).device(xyz1.device());

  torch::Tensor dist1 = torch::zeros({batchsize, n}, opts);
  torch::Tensor dist2 = torch::zeros({batchsize, m}, opts);

  torch::Tensor idx1 = torch::zeros({batchsize, n}, idx_opts);
  torch::Tensor idx2 = torch::zeros({batchsize, m}, idx_opts);

  chamfer_faiss_forward(xyz1, xyz2, dist1, dist2, idx1, idx2);

  ctx->save_for_backward({xyz1, xyz2, idx1, idx2});

  std::vector<torch::Tensor> dists_with_idxs({dist1, dist2, idx1, idx2});

  return dists_with_idxs;
}

std::vector<torch::Tensor>
chamfer_FAISS_3DFunction::backward(torch::autograd::AutogradContext *ctx,
                                   std::vector<torch::Tensor> &grad_outputs) {
  std::vector<torch::Tensor> dists_with_idxs = ctx->get_saved_variables();

  torch::Tensor xyz1 = dists_with_idxs[0];
  torch::Tensor xyz2 = dists_with_idxs[1];
  torch::Tensor idx1 = dists_with_idxs[2];
  torch::Tensor idx2 = dists_with_idxs[3];

  torch::Tensor &graddist1 = grad_outputs[0];
  torch::Tensor &graddist2 = grad_outputs[1];

  torch::Tensor gradxyz1 = torch::zeros_like(xyz1);
  torch::Tensor gradxyz2 = torch::zeros_like(xyz2);

  chamfer_torch_backward(xyz1, xyz2, graddist1, graddist2, idx1, idx2, gradxyz1,
                         gradxyz2);

  std::vector<torch::Tensor> grads({gradxyz1, gradxyz2});

  return grads;
}

const std::vector<torch::Tensor> chamfer_faiss(const torch::Tensor &input1,
                                               const torch::Tensor &input2) {
  const torch::Tensor contiguous_input1 = input1.contiguous();
  const torch::Tensor contiguous_input2 = input2.contiguous();

  const std::vector<torch::Tensor> dists_with_idxs =
      chamfer_FAISS_3DFunction::apply(contiguous_input1, contiguous_input2);

  const torch::Tensor valid_dists1 = toValidTensor(dists_with_idxs[0]);
  const torch::Tensor valid_dists2 = toValidTensor(dists_with_idxs[1]);
  const torch::Tensor idxs1 = dists_with_idxs[2].toType(torch::kInt);
  const torch::Tensor idxs2 = dists_with_idxs[3].toType(torch::kInt);

  const std::vector<torch::Tensor> valid_dists_with_idxs(
      {valid_dists1, valid_dists2, idxs1, idxs2});

  return valid_dists_with_idxs;
}
#endif
