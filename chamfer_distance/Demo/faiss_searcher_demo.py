import torch
import time
from chamfer_distance.Module.chamfer_distances import ChamferDistances


def demo_faiss_searcher():
    # 确保CUDA可用
    if not torch.cuda.is_available():
        print("CUDA不可用，FAISSSearcher需要CUDA支持")
        return

    # 创建随机点云数据
    batch_size = 1
    n_points = 10000
    m_points = 5000
    dim = 3

    # 创建两个点云
    xyz1 = torch.rand(batch_size, n_points, dim, device="cuda").float()  # 查询点云
    xyz2 = torch.rand(batch_size, m_points, dim, device="cuda").float()  # 静态点云

    print(f"点云1形状: {xyz1.shape}")
    print(f"点云2形状: {xyz2.shape}")

    # 使用传统方法计算Chamfer距离
    print("\n使用传统方法计算Chamfer距离...")
    start_time = time.time()
    dists1, dists2, idxs1, idxs2 = ChamferDistances.faiss(xyz1, xyz2)
    traditional_time = time.time() - start_time
    print(f"传统方法耗时: {traditional_time:.4f}秒")

    # 使用新的FAISSSearcher类
    print("\n使用新的FAISSSearcher类...")
    start_time = time.time()
    
    # 创建FAISSSearcher实例
    searcher = ChamferDistances.FAISSSearcher()
    
    # 添加静态点云
    searcher.addPoints(xyz2.reshape(-1, dim))  # 将批次维度展平
    
    # 查询最近邻点
    dist1, idx1 = searcher.query(xyz1.reshape(-1, dim))  # 将批次维度展平
    
    # 重塑结果以匹配传统方法的输出
    dist1 = dist1.reshape(batch_size, n_points)
    idx1 = idx1.reshape(batch_size, n_points)
    
    new_method_time = time.time() - start_time
    print(f"新方法耗时: {new_method_time:.4f}秒")

    # 验证结果是否一致
    print("\n验证结果:")
    dist_diff = torch.abs(dists1 - dist1).mean().item()
    print(f"距离差异: {dist_diff:.6f}")

    # 演示多次查询的效率
    print("\n演示多次查询的效率:")
    query_times = 10
    
    # 传统方法多次查询
    start_time = time.time()
    for _ in range(query_times):
        dists1, dists2, idxs1, idxs2 = ChamferDistances.faiss(xyz1, xyz2)
    traditional_multi_time = time.time() - start_time
    print(f"传统方法{query_times}次查询耗时: {traditional_multi_time:.4f}秒")
    
    # 新方法多次查询
    start_time = time.time()
    for _ in range(query_times):
        dist1, idx1 = searcher.query(xyz1.reshape(-1, dim))
    new_method_multi_time = time.time() - start_time
    print(f"新方法{query_times}次查询耗时: {new_method_multi_time:.4f}秒")
    print(f"加速比: {traditional_multi_time / new_method_multi_time:.2f}倍")


if __name__ == "__main__":
    demo_faiss_searcher()