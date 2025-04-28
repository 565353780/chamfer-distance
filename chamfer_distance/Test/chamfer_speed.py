import torch
import numpy as np
import seaborn as sns
from tqdm import trange
from matplotlib import pyplot as plt

import chamfer_cpp

from chamfer_distance.Method.check import checkResults
from chamfer_distance.Method.chamfer_triton import chamfer_triton
from chamfer_distance.Module.timer import Timer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_func_fps(func_name: str, func, xyz1: torch.Tensor, xyz2: torch.Tensor, test_second: float = 3.0) -> float:
    print('start test speed of', func_name, '...')
    timer = Timer()
    calculate_num = 0
    for _ in trange(10000000):
        dist1, dist2, idx1, idx2 = func(xyz1, xyz2)
        if isinstance(dist1, torch.Tensor):
            mean = torch.mean(dist1)
        else:
            mean = np.mean(dist1)

        assert mean >= 0
        calculate_num += 1

        spend_second = timer.now()
        if spend_second >= test_second:
            break

    fps = calculate_num / timer.now()

    return fps

def recordChamferAlgoSpeed(point_cloud_sizes_m, point_cloud_sizes_n, test_second=1.0):
    """
    记录不同点云大小组合下四种Chamfer距离算法的性能
    由于m和n的size是对称的，因此仅测试m>=n时的情况
    
    Args:
        point_cloud_sizes_m: 第一个点云的点数列表
        point_cloud_sizes_n: 第二个点云的点数列表
        test_second: 每个测试运行的秒数
        
    Returns:
        results: 字典，包含四种算法在不同点云大小组合下的FPS矩阵
    """
    torch.manual_seed(0)
    
    # 初始化结果矩阵
    results = {
        'cpu': np.zeros((len(point_cloud_sizes_m), len(point_cloud_sizes_n))),
        'cuda': np.zeros((len(point_cloud_sizes_m), len(point_cloud_sizes_n))),
        'triton': np.zeros((len(point_cloud_sizes_m), len(point_cloud_sizes_n))),
        'cuda_kd': np.zeros((len(point_cloud_sizes_m), len(point_cloud_sizes_n)))
    }
    
    # 遍历点云大小组合，只测试m>=n的情况
    for i, m in enumerate(point_cloud_sizes_m):
        for j, n in enumerate(point_cloud_sizes_n):
            # 只测试m>=n的情况
            if m < n:
                # 对于m<n的情况，使用对称性，直接使用已测试的结果
                print(f"\n跳过测试点云大小: P={m}, Q={n}，使用P={n}, Q={m}的结果")
                # 找到对应的索引
                n_idx = point_cloud_sizes_m.index(n) if n in point_cloud_sizes_m else -1
                m_idx = point_cloud_sizes_n.index(m) if m in point_cloud_sizes_n else -1
                
                # 如果找到对应的索引，则使用对称结果
                if n_idx >= 0 and m_idx >= 0 and results['cpu'][n_idx, m_idx] != 0:
                    for algo in results.keys():
                        results[algo][i, j] = results[algo][n_idx, m_idx]
                    continue
            
            print(f"\n测试点云大小: P={m}, Q={n}")
            
            # 创建随机点云
            xyz1 = torch.randn(m, 3).cuda()
            xyz2 = torch.randn(n, 3).cuda()
            
            # 设置梯度追踪
            xyz1.requires_grad_(True)
            xyz2.requires_grad_(True)
            
            # 测试CPU版本（如果点云不太大）
            if m * n <= 40000 ** 2:
                try:
                    cpu_fps = get_func_fps('chamfer_cpu', chamfer_cpp.chamfer_cpu, 
                                          xyz1[None, ...].cpu(), xyz2[None, ...].cpu(), test_second)
                    results['cpu'][i, j] = cpu_fps
                except Exception as e:
                    print(f'CPU版本测试失败: {e}')
                    results['cpu'][i, j] = 0
            else:
                print('点云规模过大，跳过CPU版本测试')
                results['cpu'][i, j] = 0
            
            # 测试CUDA版本
            cuda_fps = get_func_fps('chamfer_cuda', chamfer_cpp.chamfer_cuda, 
                                   xyz1[None, ...], xyz2[None, ...], test_second)
            results['cuda'][i, j] = cuda_fps
            
            # 测试Triton版本
            triton_fps = get_func_fps('chamfer_triton', chamfer_triton, 
                                     xyz1, xyz2, test_second)
            results['triton'][i, j] = triton_fps
            
            # 测试Triton KD版本
            cuda_kd_fps = get_func_fps('chamfer_cuda_kd', chamfer_cpp.chamfer_cuda_kd, 
                                        xyz1, xyz2, test_second)
            results['cuda_kd'][i, j] = cuda_kd_fps
            
            # 打印当前组合的结果
            print(f"点云大小 P={m}, Q={n} 的FPS结果:")
            print(f"CPU: {results['cpu'][i, j]:.2f}")
            print(f"CUDA: {results['cuda'][i, j]:.2f}")
            print(f"Triton: {results['triton'][i, j]:.2f}")
            print(f"CUDA KD: {results['cuda_kd'][i, j]:.2f}")
    
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
    algorithms = ['cpu', 'cuda', 'triton', 'cuda_kd']
    algo_names = {'cpu': 'CPU', 'cuda': 'CUDA', 'triton': 'Triton', 'cuda_kd': 'CUDA KD'}
    
    # 创建一个2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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

    xyz1 = torch.randn(1, 38000, 3).cuda()
    xyz2 = torch.randn(1, 400000, 3).cuda()
    test_second = 1.0

    xyz1.requires_grad_(True)
    xyz2.requires_grad_(True)

    print('start check results of chamfer_triton...')
    checkResults(chamfer_triton, chamfer_cpp.chamfer_cuda, xyz1, xyz2)
    print('checkResults passed!')

    print('start check results of chamfer_cuda_kd...')
    checkResults(chamfer_cpp.chamfer_cuda_kd, chamfer_cpp.chamfer_cuda, xyz1, xyz2)
    print('checkResults passed!')

    # print('start check results of chamfer_cuda_kd_cub...')
    # checkResults(chamfer_cpp.chamfer_cuda_kd_cub, chamfer_cpp.chamfer_cuda, xyz1, xyz2)
    # print('checkResults passed!')

    chamfer_cpu_fps = 0
    if xyz1.shape[1] * xyz2.shape[1] > 40000 ** 2:
        print('the size of input xyz are too large! will ignored the chamfer_cpu speed test!')
    else:
        try:
            chamfer_cpu_fps = get_func_fps('chamfer_cpu', chamfer_cpp.chamfer_cpu, xyz1.cpu(), xyz2.cpu(), test_second)
        except Exception as e:
            print('get_func_fps failed for chamfer_cpu! maybe the size of input xyz are too large!')
            print(e)
            pass
    chamfer_cuda_fps = get_func_fps('chamfer_cuda', chamfer_cpp.chamfer_cuda, xyz1, xyz2, test_second)
    chamfer_triton_fps = get_func_fps('chamfer_triton', chamfer_triton, xyz1, xyz2, test_second)
    chamfer_cuda_kd_fps = get_func_fps('chamfer_cuda_kd', chamfer_cpp.chamfer_cuda_kd, xyz1, xyz2, test_second)
    # chamfer_cuda_kd_cub_fps = get_func_fps('chamfer_cuda_kd_cub', chamfer_cpp.chamfer_cuda_kd_cub, xyz1, xyz2, test_second)

    print('fps list:')
    print('cpu:\t\t', chamfer_cpu_fps)
    print('cuda:\t\t', chamfer_cuda_fps)
    print('triton:\t\t', chamfer_triton_fps)
    print('cuda_kd:\t', chamfer_cuda_kd_fps)
    # print('cuda_kd_cub:\t', chamfer_cuda_kd_cub_fps)
    
    # 测试recordChamferAlgoSpeed函数
    print("\n是否运行Chamfer算法性能对比测试? (y/n)")
    run_benchmark = input().strip().lower() == 'y'
    
    if run_benchmark:
        # 定义不同的点云大小
        point_cloud_sizes_m = [1000, 5000, 10000]
        point_cloud_sizes_n = [1000, 5000, 10000]
        
        print("\n开始进行Chamfer算法性能对比测试...")
        results = recordChamferAlgoSpeed(point_cloud_sizes_m, point_cloud_sizes_n, test_second=1.0)
        
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
    else:
        print("跳过性能对比测试。")
    
    return True
