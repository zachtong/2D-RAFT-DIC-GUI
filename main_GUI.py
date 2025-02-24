import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
from PIL import Image, ImageTk
import helpFunctions as hf
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置backend
import matplotlib.pyplot as plt
import visualization as vis
import io
import scipy.io as sio

class RAFTDICGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAFT-DIC Displacement Field Calculator")
        
        # 设置最小窗口大小
        self.root.minsize(1200, 800)
        
        # 初始化变量
        self.init_variables()
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        # 创建上下分栏
        top_frame = ttk.Frame(main_frame)
        bottom_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        bottom_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 设置权重
        main_frame.grid_rowconfigure(1, weight=1)  # bottom_frame可以扩展
        main_frame.grid_columnconfigure(0, weight=1)
        
        # 顶部控制面板（路径选择、参数设置等）
        self.create_control_panel(top_frame)
        
        # 底部预览区域（ROI选择、裁剪预览、位移场预览）
        self.create_preview_panel(bottom_frame)
        
        self.displacement_results = []  # 只保留文件路径列表
        self.current_displacement = None  # 可以移除这行
        self.displacement_cache = {}  # 可以移除这行
        self.cache_size = 5  # 可以移除这行
        
    def init_variables(self):
        """初始化变量"""
        # 路径变量
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        
        # 处理参数变量
        self.mode = tk.StringVar(value="accumulative")
        self.crop_size_h = tk.StringVar(value="128")
        self.crop_size_w = tk.StringVar(value="128")
        self.stride = tk.StringVar(value="64")
        self.max_displacement = tk.StringVar(value="30")
        
        # ROI相关变量
        self.roi_points = []          # 存储ROI多边形的顶点
        self.drawing_roi = False      # ROI绘制状态标志
        self.roi_mask = None         # ROI的二值掩码
        self.roi_rect = None         # ROI的外接矩形坐标
        self.roi_scale_factor = 1.0  # ROI显示的缩放因子
        self.display_size = (400, 400)
        self.is_cutting_mode = False  # 是否处于裁剪模式
        
        # 图像和结果相关变量
        self.current_image = None
        self.displacement_results = []
        
    def create_control_panel(self, parent):
        """创建控制面板"""
        # 使用LabelFrame包装控制面板
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 路径选择区域
        path_frame = ttk.Frame(control_frame)
        path_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # 输入路径
        ttk.Label(path_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(path_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        
        # 输出路径
        ttk.Label(path_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(path_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_output).grid(row=1, column=2)
        
        # 处理模式和参数设置放在同一行
        settings_frame = ttk.Frame(control_frame)
        settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # 处理模式（左侧）
        mode_frame = ttk.LabelFrame(settings_frame, text="Processing Mode", padding="5")
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=(0, 5))
        ttk.Radiobutton(mode_frame, text="Accumulative", variable=self.mode, 
                        value="accumulative").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(mode_frame, text="Incremental", variable=self.mode, 
                        value="incremental").grid(row=1, column=0, padx=5)
        
        # 参数设置（右侧）
        param_frame = ttk.LabelFrame(settings_frame, text="Parameters", padding="5")
        param_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 网格布局参数
        param_frame.grid_columnconfigure(1, weight=1)
        
        # Crop Size设置
        ttk.Label(param_frame, text="Crop Size (H×W):").grid(row=0, column=0, sticky=tk.W)
        size_frame = ttk.Frame(param_frame)
        size_frame.grid(row=0, column=1, sticky=tk.W)
        ttk.Entry(size_frame, textvariable=self.crop_size_h, width=5).grid(row=0, column=0)
        ttk.Label(size_frame, text="×").grid(row=0, column=1, padx=2)
        ttk.Entry(size_frame, textvariable=self.crop_size_w, width=5).grid(row=0, column=2)
        
        # Stride设置
        ttk.Label(param_frame, text="Stride:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.stride, width=10).grid(row=1, column=1, sticky=tk.W)
        
        # Max Displacement设置
        ttk.Label(param_frame, text="Max Displacement:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.max_displacement, width=10).grid(row=2, column=1, sticky=tk.W)
        
        # 运行按钮和进度条
        run_frame = ttk.Frame(control_frame)
        run_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(run_frame, text="Run", command=self.run).grid(row=0, column=0, pady=5)
        self.progress = ttk.Progressbar(run_frame, length=400, mode='determinate')
        self.progress.grid(row=1, column=0, pady=5)
        
        # 添加图片信息显示
        self.image_info = ttk.Label(parent, text="未选择图片")
        self.image_info.grid(row=6, column=0, columnspan=3, pady=5)
        
        # 参数变化时自动更新预览
        self.crop_size_h.trace('w', self.update_preview)
        self.crop_size_w.trace('w', self.update_preview)
        self.stride.trace('w', self.update_preview)
        
    def create_preview_panel(self, parent):
        """创建预览面板"""
        preview_frame = ttk.LabelFrame(parent, text="Preview", padding="5")
        preview_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(2, weight=1)
        
        # ROI选择区域
        roi_frame = ttk.LabelFrame(preview_frame, text="ROI Selection", padding="5")
        roi_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        roi_frame.grid_rowconfigure(0, weight=1)
        roi_frame.grid_columnconfigure(0, weight=1)
        
        # 创建Canvas用于ROI选择
        self.roi_canvas = tk.Canvas(roi_frame, width=400, height=400)
        self.roi_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 添加滚动条
        x_scrollbar = ttk.Scrollbar(roi_frame, orient=tk.HORIZONTAL, command=self.roi_canvas.xview)
        y_scrollbar = ttk.Scrollbar(roi_frame, orient=tk.VERTICAL, command=self.roi_canvas.yview)
        x_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        y_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.roi_canvas.configure(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
        
        # ROI控制按钮
        self.roi_controls = ttk.Frame(roi_frame)
        self.roi_controls.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Button(self.roi_controls, text="Draw ROI", 
                   command=self.start_roi_drawing).grid(row=0, column=0, padx=2)
        ttk.Button(self.roi_controls, text="Cut ROI", 
                   command=self.start_cutting_roi).grid(row=0, column=1, padx=2)
        ttk.Button(self.roi_controls, text="Clear ROI", 
                   command=self.clear_roi).grid(row=0, column=2, padx=2)
        
        # 确认按钮在初始化时创建，但不显示
        self.confirm_roi_btn = ttk.Button(self.roi_controls, text="Confirm ROI", 
                                         command=self.confirm_roi)
        self.confirm_roi_btn.grid(row=0, column=3, padx=2)
        self.confirm_roi_btn.grid_remove()  # 初始时隐藏
        
        # 裁剪预览区域
        crop_frame = ttk.LabelFrame(preview_frame, text="Crop Windows Preview", padding="5")
        crop_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        crop_frame.grid_rowconfigure(0, weight=1)
        crop_frame.grid_columnconfigure(0, weight=1)
        
        # 创建Canvas而不是Label来显示预览
        self.preview_canvas = tk.Canvas(crop_frame)
        self.preview_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 绑定大小变化事件
        self.preview_canvas.bind('<Configure>', self.on_preview_canvas_resize)
        
        # 位移场预览区域
        disp_frame = ttk.LabelFrame(preview_frame, text="Displacement Field Preview", padding="5")
        disp_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        disp_frame.grid_rowconfigure(0, weight=1)
        disp_frame.grid_columnconfigure(0, weight=1)
        
        self.displacement_label = ttk.Label(disp_frame)
        self.displacement_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建帧选择滑动条
        slider_frame = ttk.Frame(disp_frame)
        slider_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(slider_frame, text="Frame:").grid(row=0, column=0, padx=5)
        self.frame_slider = ttk.Scale(slider_frame, 
                                     from_=1, 
                                     to=1,
                                     orient=tk.HORIZONTAL,
                                     command=self.update_displacement_preview)
        self.frame_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        slider_frame.grid_columnconfigure(1, weight=1)
        
        # 图像信息显示
        self.image_info = ttk.Label(preview_frame, text="")
        self.image_info.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
    def browse_input(self):
        """浏览输入目录"""
        directory = filedialog.askdirectory()
        if directory:
            self.input_path.set(directory)
            # 清除之前的ROI和预览
            self.clear_roi()
            # 更新图像信息
            self.update_image_info(directory)
            
    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_path.set(directory)
    
    def update_image_info(self, directory):
        """更新图像信息"""
        try:
            # 获取图像文件列表
            image_files = sorted([f for f in os.listdir(directory) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            
            if not image_files:
                messagebox.showerror("Error", "No valid image files found in the directory")
                return
            
            # 读取第一张图片
            img_path = os.path.join(directory, image_files[0])
            img = cv2.imread(img_path)
            if img is None:
                raise Exception(f"Failed to load image: {img_path}")
            
            # 存储当前图像
            self.current_image = img
            
            # 更新ROI预览
            self.update_roi_label(img)
            
            # 更新信息显示
            h, w = img.shape[:2]
            info_text = f"Image size: {w}x{h}\nNumber of images: {len(image_files)}"
            self.image_info.config(text=info_text)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def on_preview_canvas_resize(self, event):
        """响应预览canvas大小变化"""
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.update_preview()

    def update_preview(self, *args):
        """更新预览"""
        if self.current_image is None:
            return
        
        try:
            # 获取ROI矩形尺寸
            if self.roi_rect:
                xmin, ymin, xmax, ymax = self.roi_rect
                roi_h, roi_w = ymax - ymin, xmax - xmin
                
                # 获取用户输入的参数值，但不强制限制在ROI尺寸范围内
                try:
                    crop_h = int(self.crop_size_h.get() or "128")
                    crop_w = int(self.crop_size_w.get() or "128")
                    stride = int(self.stride.get() or "64")
                except ValueError:
                    crop_h, crop_w = 128, 128
                    stride = 64
                
                # 仅用于预览显示时调整尺寸
                preview_crop_h = min(crop_h, roi_h)
                preview_crop_w = min(crop_w, roi_w)
                preview_stride = min(stride, min(preview_crop_h, preview_crop_w))
                
                # 更新预览
                preview_image = self.current_image[ymin:ymax, xmin:xmax].copy()
                vis.update_preview(
                    self.preview_canvas,
                    preview_image,
                    crop_size=(preview_crop_h, preview_crop_w),
                    stride=preview_stride
                )
        except Exception as e:
            print(f"Error in update_preview: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def validate_inputs(self):
        """验证输入参数并调整为有效值"""
        try:
            if not os.path.exists(self.input_path.get()):
                raise ValueError("Input directory does not exist")
            
            # 获取ROI尺寸
            if self.roi_rect is None:
                raise ValueError("Please select ROI first")
            
            xmin, ymin, xmax, ymax = self.roi_rect
            roi_h, roi_w = ymax - ymin, xmax - xmin
            
            # 获取用户输入
            try:
                crop_h = int(self.crop_size_h.get())
                crop_w = int(self.crop_size_w.get())
                stride = int(self.stride.get())
            except ValueError:
                raise ValueError("Invalid crop size or stride value")
            
            # 调整裁剪尺寸
            if crop_h < 128 or crop_w < 128:
                print("Warning: Crop size too small, using minimum size of 128")
                crop_h = max(128, crop_h)
                crop_w = max(128, crop_w)
                self.crop_size_h.set(str(crop_h))
                self.crop_size_w.set(str(crop_w))
            
            if crop_h > roi_h or crop_w > roi_w:
                print("Warning: Crop size larger than ROI, using ROI dimensions")
                crop_h = min(crop_h, roi_h)
                crop_w = min(crop_w, roi_w)
                self.crop_size_h.set(str(crop_h))
                self.crop_size_w.set(str(crop_w))
            
            # 调整步长
            if stride < 32:
                print("Warning: Stride too small, using minimum value of 32")
                stride = 32
                self.stride.set(str(stride))
            
            if stride > min(crop_h, crop_w):
                print("Warning: Stride larger than crop size, using crop size")
                stride = min(crop_h, crop_w)
                self.stride.set(str(stride))
            
            # 验证最大位移
            max_displacement = int(self.max_displacement.get())
            if max_displacement < 0:
                raise ValueError("Max displacement must be non-negative")
            
            return True
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return False
    
    def process_images(self, args):
        """处理所有图像"""
        # 获取图像文件列表
        image_files = sorted([f for f in os.listdir(args.img_dir) 
                          if f.endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])

        if len(image_files) < 2:
            raise Exception("至少需要2张图片")

        # 确保已经设置了ROI
        if self.roi_rect is None:
            raise Exception("请先选择ROI区域")

        # 获取ROI矩形区域和掩码
        xmin, ymin, xmax, ymax = self.roi_rect

        # 创建结果目录 # 这里创建是为了之后读取结果用于展示
        results_dir = os.path.join(args.project_root, "displacement_results_npy")
        os.makedirs(results_dir, exist_ok=True)

        # 加载参考图像
        ref_img_path = os.path.join(args.img_dir, image_files[0])
        ref_image = hf.load_and_convert_image(ref_img_path)
        
        # 提取ROI区域
        ref_roi = ref_image[ymin:ymax, xmin:xmax]

        # 处理每对图像
        total_pairs = len(image_files) - 1
        for i in range(1, len(image_files)):
            # 更新进度条
            self.progress['value'] = (i / total_pairs) * 100
            self.root.update()

            # 加载变形图像
            def_img_path = os.path.join(args.img_dir, image_files[i])
            def_image = hf.load_and_convert_image(def_img_path)
            def_roi = def_image[ymin:ymax, xmin:xmax]

            # 提取ROI掩码对应区域
            roi_mask_crop = self.roi_mask[ymin:ymax, xmin:xmax] if self.roi_mask is not None else None

            # 处理图像对
            displacement_field, _ = hf.cut_image_pair_with_flow(
                ref_roi, def_roi,
                args.project_root,
                args.model,
                args.device,
                crop_size=args.crop_size,
                stride=args.stride,
                maxDisplacement=args.max_displacement,
                plot_windows=(i == 1),  # 仅为第一对图像绘制窗口布局
                roi_mask=roi_mask_crop
            )

            # 保存位移场结果
            hf.save_displacement_results(displacement_field, args.project_root, i)

            # 如果是增量模式，更新参考图像
            if args.mode == "incremental":
                ref_roi = def_roi.copy() # TBD

        # 保存结果路径
        self.displacement_results = [
            os.path.join(results_dir, f"displacement_field_{i}.npy")
            for i in range(1, len(image_files))
        ]
        
        # 更新滑动条范围
        if self.displacement_results:
            self.frame_slider.configure(to=len(self.displacement_results))
            self.frame_slider.set(1)
            self.update_displacement_preview()

    def update_displacement_preview(self, *args):
        """更新位移场预览"""
        if not self.displacement_results:
            return
        
        try:
            # 获取当前帧索引
            current_frame = int(self.frame_slider.get()) - 1
            if current_frame < 0 or current_frame >= len(self.displacement_results):
                return
            
            # 直接从文件加载位移场
            displacement = np.load(self.displacement_results[current_frame])
            
            # 创建位移场可视化
            fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 6))
            
            # 获取位移场的有效范围（排除NaN）
            u = displacement[:, :, 0]
            v = displacement[:, :, 1]
            valid_mask = ~np.isnan(u)
            if np.any(valid_mask):
                vmin_u = np.nanmin(u)
                vmax_u = np.nanmax(u)
                vmin_v = np.nanmin(v)
                vmax_v = np.nanmax(v)
            else:
                vmin_u = vmax_u = vmin_v = vmax_v = 0
            
            # 使用相同的色标范围
            im_u = ax_u.imshow(u, cmap='jet', vmin=vmin_u, vmax=vmax_u)
            ax_u.set_title("Displacement Field U")
            fig.colorbar(im_u, ax=ax_u)
            
            im_v = ax_v.imshow(v, cmap='jet', vmin=vmin_v, vmax=vmax_v)
            ax_v.set_title("Displacement Field V")
            fig.colorbar(im_v, ax=ax_v)
            
            # 保存到内存中的缓冲区
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            # 更新预览
            preview = Image.open(buf)
            w, h = preview.size
            max_size = (800, 400)
            scale = min(max_size[0]/w, max_size[1]/h)
            display_size = (int(w*scale), int(h*scale))
            preview = preview.resize(display_size, Image.LANCZOS)
            
            # 更新标签
            photo = ImageTk.PhotoImage(preview)
            self.displacement_label.configure(image=photo)
            self.displacement_label.image = photo
            
        except Exception as e:
            print(f"Error in update_displacement_preview: {str(e)}")
            import traceback
            traceback.print_exc()

    def fig2img(self, fig):
        """将matplotlib图像转换为PhotoImage"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        
        img = Image.open(buf)
        # 设置统一的显示尺寸
        display_size = (400, 400)  # 与原始图像预览保持一致
        img = img.resize(display_size, Image.LANCZOS)
        
        return ImageTk.PhotoImage(img)
    
    def run(self):
        """运行处理程序"""
        if not self.validate_inputs():
            return
        
        # 创建参数对象
        class Args:
            pass
        
        args = Args()
        args.img_dir = self.input_path.get()
        args.project_root = self.output_path.get()
        args.mode = self.mode.get()
        args.scale_factor = 1.0  # 固定为1.0，不再使用可变的缩放
        args.crop_size = (int(self.crop_size_h.get()), int(self.crop_size_w.get()))
        args.stride = int(self.stride.get())
        args.max_displacement = int(self.max_displacement.get())
        
        # 加载模型
        model_args = hf.Args()
        args.model = hf.load_model("models/raft-dic_v1.pth", args=model_args)
        args.device = 'cuda'
        
        # 确保输出目录存在
        os.makedirs(args.project_root, exist_ok=True)
        
        try:
            # 禁用界面
            self.root.config(cursor="watch")
            for child in self.root.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Radiobutton)):
                        widget.configure(state='disabled')
            
            # 重置进度条
            self.progress['value'] = 0
            
            # 运行处理
            self.process_images(args)
            
            messagebox.showinfo("Success", "Processing completed!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"Error details: {str(e)}")  # 添加详细错误输出
        finally:
            # 恢复界面
            self.root.config(cursor="")
            for child in self.root.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Radiobutton)):
                        widget.configure(state='normal')
            self.progress['value'] = 0

    def start_cutting_roi(self):
        """开始ROI裁剪模式"""
        if self.roi_mask is None:
            messagebox.showerror("Error", "Please draw initial ROI first")
            return
        
        self.is_cutting_mode = True
        self.drawing_roi = True
        self.roi_points = []
        
        # 绑定鼠标事件
        self.roi_canvas.bind('<Button-1>', self.add_roi_point)
        self.roi_canvas.bind('<Motion>', self.update_roi_preview)
        self.roi_canvas.bind('<Double-Button-1>', self.finish_roi_drawing)
        
        # 保持现有ROI的显示
        preview = self.current_image.copy()
        overlay = np.zeros_like(preview)
        overlay[self.roi_mask] = [0, 255, 0]  # 绿色表示选中区域
        preview = cv2.addWeighted(preview, 1, overlay, 0.3, 0)
        
        # 更新显示
        self.update_roi_label(preview)
        
        h, w = self.current_image.shape[:2]
        self.roi_scale_factor = min(400/w, 400/h)
        self.display_size = (int(w * self.roi_scale_factor), int(h * self.roi_scale_factor))

    def start_roi_drawing(self):
        """开始ROI绘制"""
        if self.current_image is None:
            messagebox.showerror("Error", "Please select input images first")
            return
        
        self.drawing_roi = True
        self.roi_points = []
        
        # 如果在裁剪模式下，保持现有ROI的显示
        if self.is_cutting_mode and self.roi_mask is not None:
            preview = self.current_image.copy()
            overlay = np.zeros_like(preview)
            overlay[self.roi_mask] = [0, 255, 0]  # 绿色表示选中区域
            preview = cv2.addWeighted(preview, 1, overlay, 0.3, 0)
            self.update_roi_label(preview)
        else:
            # 正常模式下，显示原始图像
            self.update_roi_label(self.current_image)
        
        # 绑定鼠标事件
        self.roi_canvas.bind('<Button-1>', self.add_roi_point)
        self.roi_canvas.bind('<Motion>', self.update_roi_preview)
        self.roi_canvas.bind('<Double-Button-1>', self.finish_roi_drawing)
        
        h, w = self.current_image.shape[:2]
        self.roi_scale_factor = min(400/w, 400/h)
        self.display_size = (int(w * self.roi_scale_factor), int(h * self.roi_scale_factor))

    def add_roi_point(self, event):
        """添加ROI点"""
        if not self.drawing_roi:
            return
        
        # 将画布坐标转换为图像坐标
        x = event.x / self.roi_scale_factor
        y = event.y / self.roi_scale_factor
        
        # 添加点到列表
        self.roi_points.append((x, y))
        
        # 如果是第一个点，直接画点
        if len(self.roi_points) == 1:
            self.roi_canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, 
                                      fill='red')
        else:
            # 画点
            self.roi_canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, 
                                      fill='red')
            # 连接线段
            prev_x = int(self.roi_points[-2][0] * self.roi_scale_factor)
            prev_y = int(self.roi_points[-2][1] * self.roi_scale_factor)
            self.roi_canvas.create_line(prev_x, prev_y, event.x, event.y, 
                                      fill='green', width=2)

    def update_roi_preview(self, event):
        """更新ROI预览"""
        if not self.drawing_roi or len(self.roi_points) == 0:
            return
        
        # 获取最后一个点的坐标
        last_x = int(self.roi_points[-1][0] * self.roi_scale_factor)
        last_y = int(self.roi_points[-1][1] * self.roi_scale_factor)
        
        # 删除之前的预览线
        self.roi_canvas.delete('preview_line')
        
        # 绘制新的预览线
        self.roi_canvas.create_line(last_x, last_y, event.x, event.y,
                                  fill='yellow', width=2, tags='preview_line')

    def finish_roi_drawing(self, event):
        """完成ROI绘制"""
        if len(self.roi_points) < 3:
            messagebox.showerror("Error", "Please select at least 3 points")
            return
        
        self.drawing_roi = False
        
        # 绘制闭合线段
        first_x = int(self.roi_points[0][0] * self.roi_scale_factor)
        first_y = int(self.roi_points[0][1] * self.roi_scale_factor)
        last_x = int(self.roi_points[-1][0] * self.roi_scale_factor)
        last_y = int(self.roi_points[-1][1] * self.roi_scale_factor)
        self.roi_canvas.create_line(last_x, last_y, first_x, first_y, 
                                  fill='green', width=2)
        
        # 解绑鼠标事件
        self.roi_canvas.unbind('<Button-1>')
        self.roi_canvas.unbind('<Motion>')
        self.roi_canvas.unbind('<Double-Button-1>')
        
        # 创建或更新ROI掩码
        if self.is_cutting_mode and self.roi_mask is not None:
            # 在裁剪模式下，从现有ROI中减去新绘制的区域
            new_mask = np.zeros_like(self.roi_mask, dtype=bool)
            points = np.array(self.roi_points, dtype=np.int32)
            cv2.fillPoly(new_mask.view(np.uint8), [points], 1)
            self.roi_mask = self.roi_mask & ~new_mask
        else:
            # 正常模式下创建新的ROI
            self.create_roi_mask()
        
        # 重置裁剪模式
        self.is_cutting_mode = False
        
        # 更新显示，包括半透明遮罩
        self.update_roi_display()
        
        # 显示确认按钮并启用
        self.confirm_roi_btn.grid()
        self.confirm_roi_btn.configure(state='normal')

    def create_roi_mask(self):
        """创建ROI掩码"""
        if not self.roi_points or len(self.roi_points) < 3:
            return
        
        # 获取图像尺寸
        h, w = self.current_image.shape[:2]
        
        # 创建掩码
        if self.roi_mask is None:
            self.roi_mask = np.zeros((h, w), dtype=bool)
        
        # 转换点坐标为整数数组
        points = np.array(self.roi_points, dtype=np.int32)
        
        # 填充多边形
        cv2.fillPoly(self.roi_mask.view(np.uint8), [points], 1)

    def extract_roi_rectangle(self):
        """提取包含ROI的最小矩形"""
        if self.roi_mask is None:
            return
        
        # 找到非零点的坐标
        rows, cols = np.nonzero(self.roi_mask)
        if len(rows) == 0 or len(cols) == 0:
            return
        
        # 获取边界
        ymin, ymax = rows.min(), rows.max()
        xmin, xmax = cols.min(), cols.max()
        
        # 存储矩形坐标
        self.roi_rect = (xmin, ymin, xmax+1, ymax+1)

    def update_roi_display(self):
        """更新ROI显示，包括半透明遮罩"""
        if self.current_image is None or self.roi_mask is None:
            return
        
        # 创建带有半透明遮罩的预览图像
        preview = self.current_image.copy()
        overlay = np.zeros_like(preview)
        overlay[self.roi_mask] = [0, 255, 0]  # 绿色表示选中区域
        preview = cv2.addWeighted(preview, 1, overlay, 0.3, 0)
        
        # 更新ROI预览
        self.update_roi_label(preview)

    def clear_roi(self):
        """清除ROI选择"""
        self.roi_points = []
        self.roi_mask = None
        self.roi_rect = None
        self.is_cutting_mode = False
        
        # 重置预览
        if self.current_image is not None:
            self.update_roi_label(self.current_image)
        
        # 重置主预览区域
        self.update_preview()
        
        # 隐藏确认按钮
        self.confirm_roi_btn.grid_remove()

    def confirm_roi(self):
        """确认ROI选择并更新预览"""
        if self.roi_mask is None:
            messagebox.showerror("Error", "Please draw ROI first")
            return
        
        # 提取包含ROI的最大矩形
        self.extract_roi_rectangle()
        
        # 获取ROI矩形区域
        if self.roi_rect:
            xmin, ymin, xmax, ymax = self.roi_rect
            
            # 更新裁剪区域的预览
            self.update_preview()
            
            # 更新信息显示
            roi_info = f"\nROI size: {xmax-xmin}x{ymax-ymin}"
            self.image_info.config(text=self.image_info.cget("text") + roi_info)

    def update_roi_label(self, image):
        """更新ROI预览标签"""
        if image is None:
            return
        
        # 计算缩放比例
        h, w = image.shape[:2]
        scale = min(400/w, 400/h)
        
        # 调整图像大小
        display_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)
        
        # 转换为PhotoImage
        if len(resized.shape) == 2:
            # 如果是灰度图像，转换为RGB
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            # 如果是RGBA图像，转换为RGB
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
        
        # 更新画布大小
        self.roi_canvas.config(width=display_size[0], height=display_size[1])
        
        # 创建PhotoImage
        image_pil = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(image_pil)
        
        # 清除画布并显示新图像
        self.roi_canvas.delete("all")
        self.roi_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.roi_canvas.image = photo  # 保持引用
        
        # 更新缩放因子
        self.roi_scale_factor = scale

if __name__ == '__main__':
    root = tk.Tk()
    app = RAFTDICGUI(root)
    root.mainloop()
