import torch
import faiss
import numpy as np
from tqdm import trange

# 假设静态数据库是一个 400000 x 3 的 numpy 数组
db = np.random.rand(400000, 3).astype("float32")

# 构建 GPU FlatL2 索引
res = faiss.StandardGpuResources()
index_flat = faiss.IndexFlatL2(3)  # L2 距离
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
gpu_index.add(db)  # 一次性添加静态点

for i in trange(10000):
    queries = np.random.rand(100000, 3).astype("float32")

    # 查询最近的 1 个点（返回 distances 和 indices）
    D, I = gpu_index.search(queries, 1)  # D: [100000, 1], I: [100000, 1]
