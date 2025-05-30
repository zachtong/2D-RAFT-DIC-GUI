import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from pathlib import Path

def load_displacement_field(file_path):
    """
    加载位移场数据，支持.npy和.mat格式
    """
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.mat'):
        data = sio.loadmat(file_path)
        # 假设.mat文件中的位移场存储为'U'和'V'
        return np.stack([data['U'], data['V']], axis=-1)

def calculate_metrics(pred, gt, valid_mask=None, center_ratio=0.9):
    """
    计算各种评估指标
    Args:
        pred: 预测的位移场
        gt: 真实位移场
        valid_mask: 可选的有效区域mask
        center_ratio: 中心区域的比例，默认0.9表示中间90%区域
    """
    if valid_mask is None:
        # 创建中心区域的mask
        h, w = pred.shape[:2]
        margin = (1 - center_ratio) / 2
        h_start, h_end = int(h * margin), int(h * (1 - margin))
        w_start, w_end = int(w * margin), int(w * (1 - margin))
        center_mask = np.zeros_like(pred[..., 0], dtype=bool)
        center_mask[h_start:h_end, w_start:w_end] = True
        valid_mask = center_mask & ~np.isnan(pred[..., 0]) & ~np.isnan(gt[..., 0])
    
    metrics = {}
    
    # 计算U和V分量的误差
    for i, component in enumerate(['U', 'V']):
        diff = pred[..., i][valid_mask] - gt[..., i][valid_mask]
        metrics[f'{component}_MAE'] = np.mean(np.abs(diff))
        metrics[f'{component}_RMSE'] = np.sqrt(np.mean(diff**2))
        metrics[f'{component}_MAX'] = np.max(np.abs(diff))
    
    # 计算位移场整体的误差
    total_diff = np.sqrt(np.sum((pred - gt)**2, axis=-1))[valid_mask]
    metrics['Total_MAE'] = np.mean(np.abs(total_diff))
    metrics['Total_RMSE'] = np.sqrt(np.mean(total_diff**2))
    metrics['Total_MAX'] = np.max(total_diff)
    
    return metrics

def visualize_comparison(pred, gt, output_path, center_ratio=0.9):
    """
    可视化比较结果
    Args:
        pred: 预测的位移场
        gt: 真实位移场
        output_path: 输出路径
        center_ratio: 中心区域的比例，默认0.9表示中间90%区域
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 设置颜色映射范围
    vmin_u = min(np.nanmin(pred[..., 0]), np.nanmin(gt[..., 0]))
    vmax_u = max(np.nanmax(pred[..., 0]), np.nanmax(gt[..., 0]))
    vmin_v = min(np.nanmin(pred[..., 1]), np.nanmin(gt[..., 1]))
    vmax_v = max(np.nanmax(pred[..., 1]), np.nanmax(gt[..., 1]))
    
    # U分量比较
    im0 = axes[0,0].imshow(pred[..., 0], cmap='jet', vmin=vmin_u, vmax=vmax_u)
    axes[0,0].set_title('Predicted U')
    plt.colorbar(im0, ax=axes[0,0])
    
    im1 = axes[0,1].imshow(gt[..., 0], cmap='jet', vmin=vmin_u, vmax=vmax_u)
    axes[0,1].set_title('Ground Truth U')
    plt.colorbar(im1, ax=axes[0,1])
    
    diff_u = pred[..., 0] - gt[..., 0]
    im2 = axes[0,2].imshow(diff_u, cmap='RdBu')
    axes[0,2].set_title('U Difference')
    plt.colorbar(im2, ax=axes[0,2])
    
    # V分量比较
    im3 = axes[1,0].imshow(pred[..., 1], cmap='jet', vmin=vmin_v, vmax=vmax_v)
    axes[1,0].set_title('Predicted V')
    plt.colorbar(im3, ax=axes[1,0])
    
    im4 = axes[1,1].imshow(gt[..., 1], cmap='jet', vmin=vmin_v, vmax=vmax_v)
    axes[1,1].set_title('Ground Truth V')
    plt.colorbar(im4, ax=axes[1,1])
    
    diff_v = pred[..., 1] - gt[..., 1]
    im5 = axes[1,2].imshow(diff_v, cmap='RdBu')
    axes[1,2].set_title('V Difference')
    plt.colorbar(im5, ax=axes[1,2])
    
    # 在绘图后添加中心区域的边界标记
    h, w = pred.shape[:2]
    margin = (1 - center_ratio) / 2
    h_start, h_end = int(h * margin), int(h * (1 - margin))
    w_start, w_end = int(w * margin), int(w * (1 - margin))
    
    for ax in axes.flat:
        ax.axvline(x=w_start, color='white', linestyle='--', alpha=0.5)
        ax.axvline(x=w_end, color='white', linestyle='--', alpha=0.5)
        ax.axhline(y=h_start, color='white', linestyle='--', alpha=0.5)
        ax.axhline(y=h_end, color='white', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def load_ground_truth(u_path, v_path):
    """
    加载分开存储的U和V ground truth数据
    """
    u_data = sio.loadmat(u_path)['u_displacement']  # 根据实际变量名调整
    v_data = sio.loadmat(v_path)['v_displacement']  # 根据实际变量名调整
    return np.stack([u_data, v_data], axis=-1)

def plot_error_statistics(pred_results, gt, output_dir, center_ratio=0.9):
    """
    绘制误差的均值和标准差对比图
    Args:
        pred_results: 预测结果字典
        gt: 真实位移场
        output_dir: 输出目录
        center_ratio: 中心区域的比例，默认0.9表示中间90%区域
    """
    # 创建中心区域的mask
    h, w = gt.shape[:2]
    margin = (1 - center_ratio) / 2
    h_start, h_end = int(h * margin), int(h * (1 - margin))
    w_start, w_end = int(w * margin), int(w * (1 - margin))
    center_mask = np.zeros_like(gt[..., 0], dtype=bool)
    center_mask[h_start:h_end, w_start:w_end] = True
    
    # 计算统计信息
    stats = {}
    for name, pred in pred_results.items():
        valid_mask = center_mask & ~np.isnan(pred[..., 0]) & ~np.isnan(gt[..., 0])
        u_diff = pred[..., 0][valid_mask] - gt[..., 0][valid_mask]
        v_diff = pred[..., 1][valid_mask] - gt[..., 1][valid_mask]
        
        stats[name] = {
            'u_mean': np.mean(np.abs(u_diff)),  # 使用绝对值的均值
            'v_mean': np.mean(np.abs(v_diff)),
            'u_std': np.std(u_diff),
            'v_std': np.std(v_diff)
        }
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 准备数据
    names = list(stats.keys())
    x = np.arange(len(names))
    width = 0.35
    
    # 绘制均值对比图
    ax1.bar(x - width/2, [stats[name]['u_mean'] for name in names], width, label='U')
    ax1.bar(x + width/2, [stats[name]['v_mean'] for name in names], width, label='V')
    ax1.set_title('Mean Absolute Error Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制标准差对比图
    ax2.bar(x - width/2, [stats[name]['u_std'] for name in names], width, label='U')
    ax2.bar(x + width/2, [stats[name]['v_std'] for name in names], width, label='V')
    ax2.set_title('Standard Deviation Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_statistics_comparison.png'))
    plt.close()
    
    # 保存统计信息到CSV
    import pandas as pd
    df_stats = pd.DataFrame(stats).T
    df_stats.to_csv(os.path.join(output_dir, 'error_statistics.csv'))
    
    return stats

def compare_results(gt_u_path, gt_v_path, results_dirs, output_dir, center_ratio=0.9):
    """
    比较多个结果与ground truth
    Args:
        gt_u_path: ground truth U分量路径
        gt_v_path: ground truth V分量路径
        results_dirs: 结果目录列表
        output_dir: 输出目录
        center_ratio: 中心区域的比例，默认0.9表示中间90%区域
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载ground truth
    gt = load_ground_truth(gt_u_path, gt_v_path)
    
    # 存储所有结果的指标和预测结果
    all_metrics = {}
    pred_results = {}
    
    # 对每个结果进行评估
    for result_path in results_dirs:
        dir_name = Path(result_path).parent.parent.name
        print(f"\nProcessing {dir_name}...")
        
        # 加载预测结果
        pred = load_displacement_field(result_path)
        pred_results[dir_name] = pred
        
        # 计算指标
        metrics = calculate_metrics(pred, gt, center_ratio=center_ratio)
        all_metrics[dir_name] = metrics
        
        # 生成可视化比较
        output_path = os.path.join(output_dir, f"{dir_name}_comparison.png")
        visualize_comparison(pred, gt, output_path, center_ratio=center_ratio)
        
        # 打印指标
        print(f"\nMetrics for {dir_name}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")
    
    # 生成误差统计对比图
    error_stats = plot_error_statistics(pred_results, gt, output_dir, center_ratio=center_ratio)
    
    # 保存所有指标到CSV文件
    import pandas as pd
    df = pd.DataFrame(all_metrics).T
    df.to_csv(os.path.join(output_dir, 'comparison_metrics.csv'))
    
    return all_metrics, error_stats

# 使用示例
if __name__ == "__main__":
    # 设置路径
    gt_u_path = "examples/Quadratic/u_displacement_1.mat"
    gt_v_path = "examples/Quadratic/v_displacement_1.mat"
    results_dirs = [
        "Results_no_crop/displacement_results_mat/displacement_field_1.mat",
        "Results_window_256_step128/displacement_results_mat/displacement_field_1.mat",
        "Results_window_256_step64/displacement_results_mat/displacement_field_1.mat",
        "Results_window_256_step128_smooth/displacement_results_mat/displacement_field_1.mat",
        "Results_window_256_step64_smooth/displacement_results_mat/displacement_field_1.mat"
    ]
    output_dir = "comparison_results_80%"
    
    # 运行比较
    metrics, error_stats = compare_results(gt_u_path, gt_v_path, results_dirs, output_dir,0.8)