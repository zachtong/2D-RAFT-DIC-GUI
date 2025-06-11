import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk

def save_displacement_visualization(final_flow, output_path):
    """保存位移场可视化结果"""
    u = final_flow[:, :, 0]
    v = final_flow[:, :, 1]
    
    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(16, 8))
    im_u = ax_u.imshow(u, cmap='jet')
    ax_u.set_title("Displacement Field U")
    fig.colorbar(im_u, ax=ax_u)
    
    im_v = ax_v.imshow(v, cmap='jet')
    ax_v.set_title("Displacement Field V")
    fig.colorbar(im_v, ax=ax_v)
    
    plt.savefig(output_path)
    plt.close()

def create_preview_image(image, crop_size=None, shift=None):
    """创建预览图像，优化窗口显示效果"""
    # 转换为PIL图像用于GUI显示
    preview_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(preview_pil)
    
    h, w = image.shape[:2]
    
    if crop_size and shift:
        crop_h, crop_w = crop_size
        
        # 计算网格
        x_positions = range(0, w-crop_w+shift, shift)
        y_positions = range(0, h-crop_h+shift, shift)
        
        # 绘制所有窗口，使用统一的样式
        for y in y_positions:
            for x in x_positions:
                x_end = min(x + crop_w, w)
                y_end = min(y + crop_h, h)
                
                # 使用浅色线条绘制所有窗口
                draw.rectangle(
                    [(x, y), (x_end, y_end)],
                    outline='lightblue',
                    width=1
                )
        
        # 突出显示三个关键窗口，但使用较细的线条
        # 第一个窗口（蓝色）
        draw.rectangle([(0, 0), (crop_w, crop_h)], 
                      outline='blue', width=2)
        
        # 右邻居（黄色）
        if shift < w:
            draw.rectangle(
                [(shift, 0), 
                 (min(shift + crop_w, w), crop_h)],
                outline='yellow', width=2
            )
        
        # 下邻居（绿色）
        if shift < h:
            draw.rectangle(
                [(0, shift), 
                 (crop_w, min(shift + crop_h, h))],
                outline='green', width=2
            )
    
    return preview_pil

def update_preview(canvas, image, crop_size=None, shift=None):
    """更新GUI中的预览图像"""
    # 获取canvas的当前大小
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    
    if canvas_width <= 1 or canvas_height <= 1:  # Canvas尚未完全初始化
        return
    
    preview = create_preview_image(image, crop_size, shift)
    
    # 计算缩放比例以适应canvas
    w, h = preview.size
    scale = min(canvas_width/w, canvas_height/h)
    display_size = (int(w*scale), int(h*scale))
    
    # 居中显示的偏移量
    x_offset = (canvas_width - display_size[0]) // 2
    y_offset = (canvas_height - display_size[1]) // 2
    
    preview = preview.resize(display_size, Image.LANCZOS)
    
    # 更新canvas
    photo = ImageTk.PhotoImage(preview)
    # 清除之前的内容
    canvas.delete("all")
    # 在canvas上创建图像
    canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
    # 保持引用
    canvas.image = photo