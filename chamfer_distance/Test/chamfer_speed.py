import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from chamfer_distance.Data.fps_map import FPSMap
from chamfer_distance.Module.chamfer_distances import ChamferDistances
from chamfer_distance.Module.speed_manager import SpeedManager

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def recordChamferAlgoSpeed(point_cloud_sizes_m, point_cloud_sizes_n):
    algo_name_list = ChamferDistances.getAlgoNameList()

    results = {}
    for algo_name in algo_name_list:
        results[algo_name] = FPSMap()

    for m in point_cloud_sizes_m:
        for n in point_cloud_sizes_n:
            if m > n:
                continue

            print('[INFO][chamfer_speed::recordChamferAlgoSpeed]')
            print(f"\t test point cloud sizes : P={m}, Q={n}")

            xyz1_shape = [1, m, 3]
            xyz2_shape = [1, n, 3]

            algo_fps_dict = SpeedManager.getAlgoFPSDict(xyz1_shape, xyz2_shape)

            for algo_name, algo_fps in algo_fps_dict.items():
                results[algo_name].addFPS(m, n, algo_fps)

    results['cuda'].render()
    exit()

    return results


def visualizeChamferSpeedResults(results, point_cloud_sizes_m, point_cloud_sizes_n):
    """
    可视化不同算法在各种点云大小下的性能表现
    支持优化后的结果矩阵（只测试m>=n的情况）
    
    Args:
        results: recordChamferAlgoSpeed函数返回的结果字典
        point_cloud_sizes_m: 第一个点云的点数列表
        point_cloud_sizes_n: 第二个点云的点数列表
    """
    algorithms = ['cpu', 'cuda', 'triton', 'cuda_kd', 'cuda_kd_cub']
    algo_names = {'cpu': 'CPU', 'cuda': 'CUDA', 'triton': 'Triton', 'cuda_kd': 'CUDA KD', 'cuda_kd_cub': 'CUDA KD CUB'}
    
    # 创建一个2x2的子图布局
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 为每个算法创建热力图
    for i, algo in enumerate(algorithms):
        ax = axes[i]
        
        # 创建带注释的热力图
        data = results[algo]
        # 创建注释矩阵
        annot = []
        for i_m in range(len(point_cloud_sizes_m)):
            row = []
            for i_n in range(len(point_cloud_sizes_n)):
                # 对于m<n的情况，添加星号标记
                if point_cloud_sizes_m[i_m] < point_cloud_sizes_n[i_n]:
                    row.append(f"{data[i_m, i_n]:.2f}*")
                else:
                    row.append(f"{data[i_m, i_n]:.2f}")
            annot.append(row)
        
        # 绘制热力图
        sns.heatmap(data, annot=annot, fmt="", cmap='viridis', 
                   xticklabels=point_cloud_sizes_n, yticklabels=point_cloud_sizes_m, ax=ax)
        
        ax.set_title(f'{algo_names[algo]} 算法性能 (FPS)')
        ax.set_xlabel('点云Q大小')
        ax.set_ylabel('点云P大小')
    
    # 添加说明文字
    plt.figtext(0.5, 0.01, "注: 带*标记的数据是通过对称性(P与Q交换)得到的结果", 
                ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig('chamfer_speed_comparison.png', dpi=300)
    plt.show()
    
    # 创建条形图比较不同算法在每种点云大小组合下的性能
    n_combinations = len(point_cloud_sizes_m) * len(point_cloud_sizes_n)
    fig, axes = plt.subplots(n_combinations, 1, figsize=(10, 4 * n_combinations))
    
    # 如果只有一种组合，确保axes是可迭代的
    if n_combinations == 1:
        axes = [axes]
    
    # 为每种点云大小组合创建条形图
    idx = 0
    for i, m in enumerate(point_cloud_sizes_m):
        for j, n in enumerate(point_cloud_sizes_n):
            ax = axes[idx]
            fps_values = [results[algo][i, j] for algo in algorithms]
            
            # 创建条形图
            bars = ax.bar(algo_names.values(), fps_values, color=['blue', 'orange', 'green', 'red'])
            
            # 在条形上方添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # 为通过对称性得到的结果添加特殊标记
            title = f'点云大小: P={m}, Q={n}'
            if m < n:
                title += ' (通过对称性得到)'
                # 为条形图添加特殊背景色以标识对称性结果
                ax.set_facecolor('#fff8e8')  # 浅黄色背景
            
            ax.set_title(title)
            ax.set_ylabel('FPS (每秒帧数)')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            idx += 1
    
    plt.tight_layout()
    plt.savefig('chamfer_speed_bars.png', dpi=300)
    plt.show()
    
    # 创建一个表格，显示每种算法在不同点云大小下的最佳性能
    best_algo = np.empty((len(point_cloud_sizes_m), len(point_cloud_sizes_n)), dtype='U10')
    
    for i in range(len(point_cloud_sizes_m)):
        for j in range(len(point_cloud_sizes_n)):
            fps_values = {algo: results[algo][i, j] for algo in algorithms}
            best_algo[i, j] = max(fps_values, key=fps_values.get)
    
    # 创建热力图显示最佳算法
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.zeros_like(best_algo, dtype=float), annot=best_algo, fmt='', cmap='viridis',
               xticklabels=point_cloud_sizes_n, yticklabels=point_cloud_sizes_m)
    plt.title('不同点云大小下的最佳算法')
    plt.xlabel('点云Q大小')
    plt.ylabel('点云P大小')
    plt.tight_layout()
    plt.savefig('chamfer_best_algorithm.png', dpi=300)
    plt.show()

def test():
    torch.manual_seed(0)

    xyz1_shape = [1, 4000, 3]
    xyz2_shape = [1, 4000, 3]

    ChamferDistances.check(xyz1_shape, xyz2_shape)

    algo_fps_dict = SpeedManager.getAlgoFPSDict(xyz1_shape, xyz2_shape)

    print('fps:')
    for algo_name, algo_fps in algo_fps_dict.items():
        print(algo_name, ':', algo_fps)

    point_cloud_sizes_m = [100, 500, 1000, 5000]
    point_cloud_sizes_n = [100, 500, 1000, 5000]

    print("\n开始进行Chamfer算法性能对比测试...")
    results = recordChamferAlgoSpeed(point_cloud_sizes_m, point_cloud_sizes_n)

    print("\n性能测试结果矩阵:")
    for algo, matrix in results.items():
        print(f"\n{algo} 算法:")
        print(matrix)
    
    # 可视化结果
    print("\n生成可视化结果...")
    visualizeChamferSpeedResults(results, point_cloud_sizes_m, point_cloud_sizes_n)
    print("\n可视化结果已保存为PNG文件")
    
    # 找出整体最佳算法
    all_fps = []
    for algo in ['cpu', 'cuda', 'triton', 'triton_kd']:
        all_fps.append(np.mean(results[algo]))
    
    best_overall = ['cpu', 'cuda', 'triton', 'triton_kd'][np.argmax(all_fps)]
    print(f"\n综合所有测试情况，平均性能最佳的算法是: {best_overall}")
    print(f"平均FPS: {np.max(all_fps):.2f}")
    return True
