import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
import cv2
import time
import torch
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import matplotlib.cm as cm
import warnings
import scipy.io as sio
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置matplotlib使用Agg后端，避免字体问题
plt.switch_backend('Agg')

sys.path.append('./core')  # 将 raft 文件夹添加到路径中
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

# A sketchy class to pass to RAFT (adjust parameters as needed)
class Args():
    def __init__(self, model='', path='', small=False, mixed_precision=True, alternate_corr=False):
        self.model = model
        self.path = path
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration
    
def process_img(img, device):
    """Converts a numpy image (H, W, 3) to a torch tensor of shape [1, 3, H, W]."""
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)

def load_model(weights_path, args, weights_only=True):
    """Loads the RAFT model with given weights and arguments."""
    model = RAFT(args)
    pretrained_weights = torch.load(weights_path, map_location=torch.device("cpu"))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(pretrained_weights)
    model.to("cuda")
    return model

def inference(model, frame1, frame2, device, pad_mode='sintel',
              iters=12, flow_init=None, upsample=True, test_mode=True):
    """Runs RAFT inference on a pair of images."""
    model.eval()
    with torch.no_grad():
        frame1 = process_img(frame1, device)
        frame2 = process_img(frame2, device)
        
        # 记录原始尺寸
        original_size = (frame1.shape[2], frame1.shape[3])
        
        # 使用padder进行padding
        padder = InputPadder(frame1.shape, mode=pad_mode)
        frame1, frame2 = padder.pad(frame1, frame2)
        
        # 使用新的 autocast 语法
        with torch.amp.autocast('cuda', enabled=True):
            if test_mode:
                flow_low, flow_up = model(frame1, frame2, iters=iters, 
                                        flow_init=flow_init,
                                        upsample=upsample, 
                                        test_mode=test_mode)
                
                # 裁剪回原始尺寸
                flow_up = flow_up[:, :, :original_size[0], :original_size[1]]
                flow_low = flow_low[:, :, :original_size[0]//8, :original_size[1]//8]
                
                return flow_low, flow_up
            else:
                flow_iters = model(frame1, frame2, iters=iters, 
                                 flow_init=flow_init,
                                 upsample=upsample, 
                                 test_mode=test_mode)
                return flow_iters

def get_viz(flo):
    """Converts flow to a visualization image."""
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    return flow_viz.flow_to_image(flo)

def load_and_convert_image(img_path):
    """从本地加载图片并进行颜色空间和位深度转换
    
    Args:
        img_path: 图片路径
        
    Returns:
        frame_rgb: RGB格式的8bit图像
    """
    # 读取图片
    frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if frame is None:
        raise Exception(f"Failed to load image from {img_path}")

    # 检查图像位深度并转换为8bit
    if frame.dtype != np.uint8:
        if frame.dtype == np.uint16:
            # 16bit转8bit，保持相对亮度关系
            frame = (frame / 256).astype(np.uint8)
        else:
            # 其他位深度，归一化到0-255范围
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)

    # BGR转RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame_rgb

# ============================
# Cutting and Flow Reconstruction with Fixed-Stride Cropping
# ============================
def calculate_window_positions(image_size, crop_size, stride):
    """
    计算窗口位置，确保完整覆盖且使用完整尺寸的窗口
    
    Args:
        image_size: 图像尺寸
        crop_size: 裁剪窗口尺寸
        stride: 滑动步长
    
    Returns:
        positions: 窗口起始位置列表
    """
    positions = []
    current = 0
    
    while current <= image_size - crop_size:
        positions.append(current)
        current += stride
    
    # 如果最后一个位置未覆盖到边缘，添加一个确保覆盖边缘的位置
    if current < image_size - crop_size:
        positions.append(image_size - crop_size)
    elif positions[-1] + crop_size < image_size:
        positions.append(image_size - crop_size)
    
    return positions

def cut_image_pair_with_flow(ref_image, def_image, output_dir, model, device,
                           crop_size=(128, 128), stride=64, maxDisplacement=0, 
                           plot_windows=False, roi_mask=None):
    """
    修改后的切割函数，支持ROI
    
    Args:
        ref_image: 参考图像
        def_image: 变形图像
        output_dir: 输出目录
        model: RAFT模型
        device: 计算设备
        crop_size: 切割窗口大小
        stride: 滑动步长
        maxDisplacement: 最大位移
        plot_windows: 是否绘制窗口布局
        roi_mask: ROI掩码，与输入图像同尺寸的二值数组
    """
    # 创建必要的子目录
    crops_dir = os.path.join(output_dir, "crops")
    windows_dir = os.path.join(output_dir, "windows")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(windows_dir, exist_ok=True)

    H, W, _ = ref_image.shape
    crop_h, crop_w = crop_size

    # 计算x和y方向的窗口位置
    x_positions = calculate_window_positions(W, crop_w, stride)
    y_positions = calculate_window_positions(H, crop_h, stride)

    windows = []
    global_flow = np.zeros((H, W, 2), dtype=np.float64)
    count_field = np.zeros((H, W), dtype=np.float64)

    # 如果提供了ROI掩码，确保其类型为boolean
    if roi_mask is not None:
        roi_mask = roi_mask.astype(bool)
    else:
        # 如果没有提供掩码，创建一个全True的掩码
        roi_mask = np.ones((H, W), dtype=bool)

    # Valid mask (保持原有的有效区域计算)
    confidenceRange_y = [maxDisplacement, crop_h-maxDisplacement]
    confidenceRange_x = [maxDisplacement, crop_w-maxDisplacement]
    valid_mask = np.zeros((crop_h, crop_w), dtype=np.float64)
    valid_mask[confidenceRange_y[0]:confidenceRange_y[1], 
              confidenceRange_x[0]:confidenceRange_x[1]] = 1.0

    count = 0
    for y in y_positions:
        for x in x_positions:
            # 现在所有窗口都是完整的crop_size大小
            ref_window = ref_image[y:y+crop_h, x:x+crop_w, :]
            def_window = def_image[y:y+crop_h, x:x+crop_w, :]
            
            # 保存切割的子图
            # cv2.imwrite(os.path.join(crops_dir, f"ref_cropped_{count}.png"),
            #            cv2.cvtColor(ref_window, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(os.path.join(crops_dir, f"def_cropped_{count}.png"),
            #            cv2.cvtColor(def_window, cv2.COLOR_RGB2BGR))

            windows.append({
                'index': count,
                'position': (x, y, x+crop_w, y+crop_h)
            })

            # RAFT inference
            flow_low, flow_up = inference(model, ref_window, def_window, device, test_mode=True)
            flow_up = flow_up.squeeze(0)
            
            # 由于我们现在确保了完整的窗口尺寸，不需要额外的尺寸调整
            u = flow_up[0].cpu().numpy()
            v = flow_up[1].cpu().numpy()
            window_flow = np.stack((u, v), axis=-1)

            global_flow[y:y+crop_h, x:x+crop_w, :] += window_flow * valid_mask[..., None]
            count_field[y:y+crop_h, x:x+crop_w] += valid_mask

            count += 1

    # 计算平均位移场
    final_flow = np.where(count_field[..., None] > 0,
                         global_flow / count_field[..., None],
                         np.nan)
    
    # 应用boolean掩码
    final_flow[~roi_mask] = np.nan

    # 对位移场进行平滑插值
    smoothed_flow = smooth_displacement_field(final_flow)

    print(f"Total window pairs processed: {count}")

    # 保存窗口布局图
    if plot_windows:
        try:
            # 使用 Agg 后端确保线程安全
            plt.switch_backend('Agg')
            
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(ref_gray, cmap='gray')
            ax.axis('off')
            colormap = cm.get_cmap('hsv', len(windows))
            
            # 一次性创建所有矩形，而不是逐个添加
            patches_list = []
            for window in windows:
                x_start, y_start, x_end, y_end = window['position']
                w = x_end - x_start
                h = y_end - y_start
                color = colormap(window['index'])
                rect = patches.Rectangle((x_start, y_start), w, h, linewidth=2,
                                      edgecolor=color, facecolor='none')
                patches_list.append(rect)
            
            # 批量添加所有矩形
            ax.add_collection(collections.PatchCollection(patches_list, match_original=True))
            
            plt.title("Sliding Windows Layout")
            # 使用 bbox_inches='tight' 确保图像完整保存
            plt.savefig(os.path.join(windows_dir, "windows_layout.png"), 
                       bbox_inches='tight', dpi=300)
            plt.close(fig)  # 确保关闭图形
            
        except Exception as e:
            print(f"Warning: Failed to save windows layout: {str(e)}")
            # 继续执行，不让绘图错误影响主要处理流程
    
    return smoothed_flow, count_field

def save_displacement_results(displacement_field, output_dir, index):
    """保存位移场结果为npy和mat格式"""
    # 创建目录结构
    npy_dir = os.path.join(output_dir, "displacement_results_npy")
    mat_dir = os.path.join(output_dir, "displacement_results_mat")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)
    
    # 保存.npy格式
    npy_file = os.path.join(npy_dir, f"displacement_field_{index}.npy")
    np.save(npy_file, displacement_field)
    
    # 保存.mat格式
    mat_file = os.path.join(mat_dir, f"displacement_field_{index}.mat")
    # 分离U和V分量
    u = displacement_field[:, :, 0]
    v = displacement_field[:, :, 1]
    sio.savemat(mat_file, {'U': u, 'V': v})

def smooth_displacement_field(displacement_field):
    """
    对位移场进行平滑处理，减少窗口重叠区域的突变
    
    Args:
        displacement_field: shape为(H, W, 2)的位移场数据
    
    Returns:
        smoothed_field: 平滑后的位移场数据
    """
    # 创建输出数组
    smoothed_field = np.zeros_like(displacement_field)
    
    # 分别处理U和V分量
    for i in range(2):
        component = displacement_field[..., i]
        
        # 处理NaN值
        valid_mask = ~np.isnan(component)
        temp_component = component.copy()
        # 将NaN暂时替换为周围有效值的平均值
        if not np.all(valid_mask):
            from scipy.ndimage import binary_dilation
            kernel = np.ones((3, 3))
            while not np.all(valid_mask):
                dilated = binary_dilation(valid_mask, kernel)
                new_valid = dilated & ~valid_mask
                for y, x in zip(*np.where(new_valid)):
                    valid_neighbors = component[max(0,y-1):y+2, max(0,x-1):x+2][
                        ~np.isnan(component[max(0,y-1):y+2, max(0,x-1):x+2])
                    ]
                    if len(valid_neighbors) > 0:
                        temp_component[y, x] = np.mean(valid_neighbors)
                valid_mask = dilated
        
        # 应用高斯滤波
        smoothed = gaussian_filter(temp_component, sigma=2.0)
        
        # 可选：使用双边滤波进一步改善（保持边缘特征）
        # smoothed = cv2.bilateralFilter(smoothed.astype(np.float32), 
        #                               d=5,  # 邻域直径
        #                               sigmaColor=0.1,  # 颜色空间标准差
        #                               sigmaSpace=5.0)  # 坐标空间标准差
        
        # 恢复原始的NaN位置
        smoothed[np.isnan(component)] = np.nan
        
        smoothed_field[..., i] = smoothed
    
    return smoothed_field