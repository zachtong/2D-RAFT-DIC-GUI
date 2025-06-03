"""
RAFT-DIC GUI Application
------------------------
This module implements a graphical user interface for the RAFT-DIC displacement field calculator.
It provides tools for:
- Image loading and ROI selection
- Displacement field calculation using RAFT
- Result visualization and analysis
- Interactive parameter adjustment

Author: [Your Name]
Date: [Current Date]
Version: 1.0

Dependencies:
- tkinter
- OpenCV
- PIL
- NumPy
- Matplotlib
- SciPy
- RAFT-DIC core modules

Usage:
Run this script directly to launch the GUI application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
from PIL import Image, ImageTk
import helpFunctions as hf
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import visualization as vis
import io
import scipy.io as sio

class RAFTDICGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAFT-DIC Displacement Field Calculator")
        
        # Set minimum window size
        self.root.minsize(1200, 800)
        
        # Initialize variables
        self.init_variables()
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        # Create top and bottom panels
        top_frame = ttk.Frame(main_frame)
        bottom_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        bottom_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Set weights for frame expansion
        main_frame.grid_rowconfigure(1, weight=1)  # bottom_frame can expand
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Top control panel (path selection, parameter settings, etc.)
        self.create_control_panel(top_frame)
        
        # Bottom preview area (ROI selection, crop preview, displacement field preview)
        self.create_preview_panel(bottom_frame)
        
        self.displacement_results = []  # Store only file paths
        self.current_displacement = None  # Can remove this line
        self.displacement_cache = {}  # Can remove this line
        self.cache_size = 5  # Can remove this line
        
    def init_variables(self):
        """Initialize variables"""
        # Path variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        
        # Processing parameter variables
        self.mode = tk.StringVar(value="accumulative")
        self.crop_size_h = tk.StringVar(value="512")
        self.crop_size_w = tk.StringVar(value="512")
        self.stride = tk.StringVar(value="256")
        self.max_displacement = tk.StringVar(value="5")
        
        # ROI related variables
        self.roi_points = []          # Store ROI polygon vertices
        self.drawing_roi = False      # ROI drawing state flag
        self.roi_mask = None         # ROI binary mask
        self.roi_rect = None         # ROI bounding rectangle coordinates
        self.roi_scale_factor = 1.0  # ROI display scaling factor
        self.display_size = (400, 400)
        self.is_cutting_mode = False  # Whether in cutting mode
        
        # Image and result related variables
        self.current_image = None
        self.displacement_results = []
        
        # Add scaling related variables
        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.current_photo = None  # Save current displayed PhotoImage
        
        # Add playback related variables
        self.is_playing = False
        self.play_after_id = None
        self.play_interval = 100  # Playback interval (ms)
        
        # Modify smoothing processing related variables
        self.use_smooth = tk.BooleanVar(value=True)  # Whether to use smoothing
        self.sigma = tk.StringVar(value="2.0")  # Gaussian smoothing sigma parameter
        
    def create_control_panel(self, parent):
        """Create control panel"""
        # Use LabelFrame to wrap control panel
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Path selection area
        path_frame = ttk.Frame(control_frame)
        path_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Input path
        ttk.Label(path_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(path_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        
        # Output path
        ttk.Label(path_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(path_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_output).grid(row=1, column=2)
        
        # Processing mode and parameter settings in the same row
        settings_frame = ttk.Frame(control_frame)
        settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Processing mode (left)
        mode_frame = ttk.LabelFrame(settings_frame, text="Processing Mode", padding="5")
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=(0, 5))
        ttk.Radiobutton(mode_frame, text="Accumulative", variable=self.mode, 
                        value="accumulative").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(mode_frame, text="Incremental", variable=self.mode, 
                        value="incremental").grid(row=1, column=0, padx=5)
        
        # Parameter settings (right)
        param_frame = ttk.LabelFrame(settings_frame, text="Parameters", padding="5")
        param_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Grid layout parameters
        param_frame.grid_columnconfigure(1, weight=1)
        
        # Crop Size settings
        ttk.Label(param_frame, text="Crop Size (H×W):").grid(row=0, column=0, sticky=tk.W)
        size_frame = ttk.Frame(param_frame)
        size_frame.grid(row=0, column=1, sticky=tk.W)
        
        # Create Entry widgets
        self.crop_h_entry = ttk.Entry(size_frame, textvariable=self.crop_size_h, width=5)
        self.crop_h_entry.grid(row=0, column=0)
        ttk.Label(size_frame, text="×").grid(row=0, column=1, padx=2)
        self.crop_w_entry = ttk.Entry(size_frame, textvariable=self.crop_size_w, width=5)
        self.crop_w_entry.grid(row=0, column=2)
        
        # Stride settings
        ttk.Label(param_frame, text="Stride:").grid(row=1, column=0, sticky=tk.W)
        self.stride_entry = ttk.Entry(param_frame, textvariable=self.stride, width=10)
        self.stride_entry.grid(row=1, column=1, sticky=tk.W)
        
        # Bind events
        self.crop_h_entry.bind('<Return>', self.on_param_change)
        self.crop_h_entry.bind('<FocusOut>', self.on_param_change)
        self.crop_w_entry.bind('<Return>', self.on_param_change)
        self.crop_w_entry.bind('<FocusOut>', self.on_param_change)
        self.stride_entry.bind('<Return>', self.on_param_change)
        self.stride_entry.bind('<FocusOut>', self.on_param_change)
        
        # Max Displacement settings
        ttk.Label(param_frame, text="Max Displacement:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.max_displacement, width=10).grid(row=2, column=1, sticky=tk.W)
        
        # Add new smoothing processing options in parameter settings frame
        smooth_frame = ttk.LabelFrame(param_frame, text="Smoothing Options", padding="5")
        smooth_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Use smoothing options
        ttk.Checkbutton(smooth_frame, text="Use Smoothing", 
                        variable=self.use_smooth).grid(row=0, column=0, sticky=tk.W)
        
        # Sigma value settings
        ttk.Label(smooth_frame, text="Sigma:").grid(row=1, column=0, sticky=tk.W)
        sigma_entry = ttk.Entry(smooth_frame, textvariable=self.sigma, width=8)
        sigma_entry.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(smooth_frame, text="(0.5-5.0)").grid(row=1, column=2, sticky=tk.W)
        
        # Run button and progress bar
        run_frame = ttk.Frame(control_frame)
        run_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(run_frame, text="Run", command=self.run).grid(row=0, column=0, pady=5)
        self.progress = ttk.Progressbar(run_frame, length=400, mode='determinate')
        self.progress.grid(row=1, column=0, pady=5)
        
        # Add image information display
        self.image_info = ttk.Label(parent, text="No image selected")
        self.image_info.grid(row=6, column=0, columnspan=3, pady=5)
        
        # Automatically update preview when parameters change
        # self.crop_size_h.trace('w', self.update_preview)
        # self.crop_size_w.trace('w', self.update_preview)
        # self.stride.trace('w', self.update_preview)
        
    def create_preview_panel(self, parent):
        """Create preview panel"""
        preview_frame = ttk.LabelFrame(parent, text="Preview", padding="5")
        preview_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(2, weight=1)
        
        # ROI selection area
        roi_frame = ttk.LabelFrame(preview_frame, text="ROI Selection", padding="5")
        roi_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        roi_frame.grid_rowconfigure(0, weight=1)
        roi_frame.grid_columnconfigure(0, weight=1)
        
        # Create Canvas for ROI selection
        self.roi_canvas = tk.Canvas(roi_frame, width=400, height=400)
        self.roi_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbars
        x_scrollbar = ttk.Scrollbar(roi_frame, orient=tk.HORIZONTAL, command=self.roi_canvas.xview)
        y_scrollbar = ttk.Scrollbar(roi_frame, orient=tk.VERTICAL, command=self.roi_canvas.yview)
        x_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        y_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.roi_canvas.configure(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
        
        # Bind mouse wheel events
        self.roi_canvas.bind('<MouseWheel>', self.on_mousewheel)  # Windows
        self.roi_canvas.bind('<Button-4>', self.on_mousewheel)    # Linux up scroll
        self.roi_canvas.bind('<Button-5>', self.on_mousewheel)    # Linux down scroll
        
        # Add scaling control buttons
        zoom_frame = ttk.Frame(roi_frame)
        zoom_frame.grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(zoom_frame, text="Zoom in", command=lambda: self.zoom(1.2)).grid(row=0, column=0, padx=2)
        ttk.Button(zoom_frame, text="Zomm out", command=lambda: self.zoom(0.8)).grid(row=0, column=1, padx=2)
        ttk.Button(zoom_frame, text="Reset", command=self.reset_zoom).grid(row=0, column=2, padx=2)
        
        # ROI control buttons
        self.roi_controls = ttk.Frame(roi_frame)
        self.roi_controls.grid(row=3, column=0, columnspan=2, pady=5)
        
        ttk.Button(self.roi_controls, text="Draw ROI", 
                   command=self.start_roi_drawing).grid(row=0, column=0, padx=2)
        ttk.Button(self.roi_controls, text="Cut ROI", 
                   command=self.start_cutting_roi).grid(row=0, column=1, padx=2)
        ttk.Button(self.roi_controls, text="Clear ROI", 
                   command=self.clear_roi).grid(row=0, column=2, padx=2)
        
        # Confirm button is created at initialization but not displayed
        self.confirm_roi_btn = ttk.Button(self.roi_controls, text="Confirm ROI", 
                                         command=self.confirm_roi)
        self.confirm_roi_btn.grid(row=0, column=3, padx=2)
        self.confirm_roi_btn.grid_remove()  # Initially hidden
        
        # Crop preview area
        crop_frame = ttk.LabelFrame(preview_frame, text="Crop Windows Preview", padding="5")
        crop_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        crop_frame.grid_rowconfigure(0, weight=1)
        crop_frame.grid_columnconfigure(0, weight=1)
        
        # Create Canvas instead of Label to display preview
        self.preview_canvas = tk.Canvas(crop_frame)
        self.preview_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind size change events
        self.preview_canvas.bind('<Configure>', self.on_preview_canvas_resize)
        
        # Displacement field preview area
        disp_frame = ttk.LabelFrame(preview_frame, text="Displacement Field Preview", padding="5")
        disp_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        disp_frame.grid_rowconfigure(0, weight=1)
        disp_frame.grid_columnconfigure(0, weight=1)
        
        self.displacement_label = ttk.Label(disp_frame)
        self.displacement_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Modify frame selection control area
        slider_frame = ttk.Frame(disp_frame)
        slider_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Add playback control buttons
        control_frame = ttk.Frame(slider_frame)
        control_frame.grid(row=0, column=0, padx=5)
        
        # Add play/pause button
        self.play_icon = "▶"  # Play icon
        self.pause_icon = "⏸"  # Pause icon
        self.is_playing = False
        self.play_button = ttk.Button(control_frame, 
                                     text=self.play_icon,
                                     width=3,
                                     command=self.toggle_play)
        self.play_button.grid(row=0, column=0, padx=2)
        
        # Add frame number display and input
        frame_control = ttk.Frame(slider_frame)
        frame_control.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Label(frame_control, text="Frame:").grid(row=0, column=0, padx=5)
        
        # Add frame number input box
        self.frame_entry = ttk.Entry(frame_control, width=5)
        self.frame_entry.grid(row=0, column=1, padx=2)
        
        # Add total frame number display
        self.total_frames_label = ttk.Label(frame_control, text="/1")
        self.total_frames_label.grid(row=0, column=2, padx=2)
        
        # Add jump button
        ttk.Button(frame_control, text="Go", command=self.jump_to_frame).grid(row=0, column=3, padx=5)
        
        # Add current image name display
        self.current_image_name = ttk.Label(frame_control, text="")
        self.current_image_name.grid(row=0, column=4, padx=5)
        
        # Slider
        self.frame_slider = ttk.Scale(slider_frame, 
                                    from_=1, 
                                    to=1,
                                    orient=tk.HORIZONTAL,
                                    command=self.update_displacement_preview)
        self.frame_slider.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        slider_frame.grid_columnconfigure(1, weight=1)
        
        # Image information display
        self.image_info = ttk.Label(preview_frame, text="")
        self.image_info.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Add playback speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.grid(row=0, column=1, padx=5)
        
        ttk.Label(speed_frame, text="Speed:").grid(row=0, column=0)
        self.speed_var = tk.StringVar(value="1x")
        speed_menu = ttk.OptionMenu(speed_frame, self.speed_var, "1x",
                                   "0.25x", "0.5x", "1x", "2x", "4x",
                                   command=self.change_play_speed)
        speed_menu.grid(row=0, column=1)
        
    def browse_input(self):
        """Browse input directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.input_path.set(directory)
            # Get image file list
            self.image_files = sorted([f for f in os.listdir(directory) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
            # Clear previous ROI and preview
            self.clear_roi()
            # Update image information
            self.update_image_info(directory)
            
    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_path.set(directory)
    
    def update_image_info(self, directory):
        """Update image information"""
        try:
            # Get image file list
            image_files = sorted([f for f in os.listdir(directory) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
            
            if not image_files:
                messagebox.showerror("Error", "No valid image files found in the directory")
                return
            
            # Save image file list for later use
            self.image_files = image_files
            
            # Read first image, use function from helpFunctions
            img_path = os.path.join(directory, image_files[0])
            img = hf.load_and_convert_image(img_path)
            if img is None:
                raise Exception(f"Failed to load image: {img_path}")
            
            # Store current image
            self.current_image = img
            
            # Update ROI preview
            self.update_roi_label(img)
            
            # Update information display
            h, w = img.shape[:2]
            info_text = f"Image size: {w}x{h}\nNumber of images: {len(image_files)}"
            self.image_info.config(text=info_text)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def on_preview_canvas_resize(self, event):
        """Respond to preview canvas size change"""
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.update_preview()

    def update_preview(self, *args):
        """Update preview"""
        if self.current_image is None:
            return
        
        try:
            # Get ROI rectangle dimensions
            if self.roi_rect:
                xmin, ymin, xmax, ymax = self.roi_rect
                roi_h, roi_w = ymax - ymin, xmax - xmin
                
                # Get user input parameters, but do not force them within ROI dimensions
                try:
                    crop_h = int(self.crop_size_h.get() or "128")
                    crop_w = int(self.crop_size_w.get() or "128")
                    stride = int(self.stride.get() or "64")
                except ValueError:
                    crop_h, crop_w = 128, 128
                    stride = 64
                
                # Only used for preview display when adjusting size
                preview_crop_h = min(crop_h, roi_h)
                preview_crop_w = min(crop_w, roi_w)
                preview_stride = min(stride, min(preview_crop_h, preview_crop_w))
                
                # Update preview
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
        """Validate input parameters and adjust to valid values"""
        try:
            if not os.path.exists(self.input_path.get()):
                raise ValueError("Input directory does not exist")
            
            # Get ROI dimensions
            if self.roi_rect is None:
                raise ValueError("Please select ROI first")
            
            xmin, ymin, xmax, ymax = self.roi_rect
            roi_h, roi_w = ymax - ymin, xmax - xmin
            
            # Get user input
            try:
                crop_h = int(self.crop_size_h.get())
                crop_w = int(self.crop_size_w.get())
                stride = int(self.stride.get())
            except ValueError:
                raise ValueError("Invalid crop size or stride value")
            
            # Adjust crop size
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
            
            # Adjust stride
            if stride < 32:
                print("Warning: Stride too small, using minimum value of 32")
                stride = 32
                self.stride.set(str(stride))
            
            if stride > min(crop_h, crop_w):
                print("Warning: Stride larger than crop size, using crop size")
                stride = min(crop_h, crop_w)
                self.stride.set(str(stride))
            
            # Verify max displacement
            max_displacement = int(self.max_displacement.get())
            if max_displacement < 0:
                raise ValueError("Max displacement must be non-negative")
            
            # Verify smoothing processing parameters
            try:
                sigma = float(self.sigma.get())
                if not 0.5 <= sigma <= 5.0:
                    print("Warning: Sigma value should be between 0.5 and 5.0")
                    sigma = np.clip(sigma, 0.5, 5.0)
                    self.sigma.set(f"{sigma:.2f}")
            except ValueError:
                print("Warning: Invalid sigma value, using default 2.0")
                self.sigma.set("2.0")
            
            return True
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return False
    
    def process_images(self, args):
        """Process all images"""
        # Get image file list
        image_files = sorted([f for f in os.listdir(args.img_dir) 
                          if f.endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'))])

        if len(image_files) < 2:
            raise Exception("At least 2 images are needed")

        # Ensure ROI is set
        if self.roi_rect is None:
            raise Exception("Please select ROI area first")

        # Get ROI rectangle area and mask
        xmin, ymin, xmax, ymax = self.roi_rect

        # Create results directory # This is created for later use to read results for display
        results_dir = os.path.join(args.project_root, "displacement_results_npy")
        os.makedirs(results_dir, exist_ok=True)

        # Load reference image
        ref_img_path = os.path.join(args.img_dir, image_files[0])
        ref_image = hf.load_and_convert_image(ref_img_path)
        
        # Extract ROI area
        ref_roi = ref_image[ymin:ymax, xmin:xmax]

        # Process each image pair
        total_pairs = len(image_files) - 1
        for i in range(1, len(image_files)):
            # Update progress bar
            self.progress['value'] = (i / total_pairs) * 100
            self.root.update()

            # Load deformed image
            def_img_path = os.path.join(args.img_dir, image_files[i])
            def_image = hf.load_and_convert_image(def_img_path)
            def_roi = def_image[ymin:ymax, xmin:xmax]

            # Extract ROI mask corresponding area
            roi_mask_crop = self.roi_mask[ymin:ymax, xmin:xmax] if self.roi_mask is not None else None

            # Process image pair, add smoothing parameters
            displacement_field, _ = hf.cut_image_pair_with_flow(
                ref_roi, def_roi,
                args.project_root,
                args.model,
                args.device,
                crop_size=args.crop_size,
                stride=args.stride,
                maxDisplacement=args.max_displacement,
                plot_windows=(i == 1),
                roi_mask=roi_mask_crop,
                use_smooth=args.use_smooth,
                sigma=args.sigma
            )

            # Save displacement field results
            hf.save_displacement_results(displacement_field, args.project_root, i)

            # If incremental mode, update reference image
            if args.mode == "incremental":
                ref_roi = def_roi.copy() # TBD

        # Save results path
        self.displacement_results = [
            os.path.join(results_dir, f"displacement_field_{i}.npy")
            for i in range(1, len(image_files))
        ]
        
        # Update slider range
        if self.displacement_results:
            self.frame_slider.configure(to=len(self.displacement_results))
            self.frame_slider.set(1)
            self.total_frames_label.configure(text=f"/{len(self.displacement_results)}")
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, "1")
            self.update_displacement_preview()

    def update_displacement_preview(self, *args):
        """Update displacement field preview"""
        if not self.displacement_results:
            return
        
        try:
            # Get current frame index
            current_frame = int(self.frame_slider.get()) - 1
            if current_frame < 0 or current_frame >= len(self.displacement_results):
                return
            
            # Update frame number input box
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, str(current_frame + 1))
            
            # Update current image name display
            if hasattr(self, 'image_files') and len(self.image_files) > current_frame + 1:
                self.current_image_name.configure(text=self.image_files[current_frame + 1])
            
            # Load and display displacement field
            displacement = np.load(self.displacement_results[current_frame])
            
            # Create displacement field visualization
            fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Get displacement field valid range (exclude NaN)
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
            
            # Use same color scale range
            im_u = ax_u.imshow(u, cmap='jet', vmin=vmin_u, vmax=vmax_u)
            ax_u.set_title("Displacement Field U")
            fig.colorbar(im_u, ax=ax_u)
            
            im_v = ax_v.imshow(v, cmap='jet', vmin=vmin_v, vmax=vmax_v)
            ax_v.set_title("Displacement Field V")
            fig.colorbar(im_v, ax=ax_v)
            
            # Save to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            # Update preview
            preview = Image.open(buf)
            w, h = preview.size
            max_size = (800, 400)
            scale = min(max_size[0]/w, max_size[1]/h)
            display_size = (int(w*scale), int(h*scale))
            preview = preview.resize(display_size, Image.LANCZOS)
            
            # Update label
            photo = ImageTk.PhotoImage(preview)
            self.displacement_label.configure(image=photo)
            self.displacement_label.image = photo
            
            # If last frame and playing, reset to first frame
            if self.is_playing and current_frame == len(self.displacement_results) - 1:
                self.frame_slider.set(1)
            
        except Exception as e:
            print(f"Error in update_displacement_preview: {str(e)}")
            import traceback
            traceback.print_exc()

    def fig2img(self, fig):
        """Convert matplotlib image to PhotoImage"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        
        img = Image.open(buf)
        # Set uniform display size
        display_size = (400, 400)  # Keep consistent with original image preview
        img = img.resize(display_size, Image.LANCZOS)
        
        return ImageTk.PhotoImage(img)
    
    def run(self):
        """Run processing program"""
        if not self.validate_inputs():
            return
        
        # Create parameter object
        class Args:
            pass
        
        args = Args()
        args.img_dir = self.input_path.get()
        args.project_root = self.output_path.get()
        args.mode = self.mode.get()
        args.scale_factor = 1.0  # Fixed at 1.0, no longer use variable scaling
        args.crop_size = (int(self.crop_size_h.get()), int(self.crop_size_w.get()))
        args.stride = int(self.stride.get())
        args.max_displacement = int(self.max_displacement.get())
        
        # Add smoothing parameters
        args.use_smooth = self.use_smooth.get()
        args.sigma = float(self.sigma.get())
        
        # Load model
        model_args = hf.Args()
        args.model = hf.load_model("models/raft-dic_v1.pth", args=model_args)
        args.device = 'cuda'
        
        # Ensure output directory exists
        os.makedirs(args.project_root, exist_ok=True)
        
        try:
            # Disable interface
            self.root.config(cursor="watch")
            for child in self.root.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Radiobutton)):
                        widget.configure(state='disabled')
            
            # Reset progress bar
            self.progress['value'] = 0
            
            # Run processing
            self.process_images(args)
            
            # Update slider and frame number display
            total_frames = len(self.displacement_results)
            self.frame_slider.configure(to=total_frames)
            self.frame_slider.set(1)
            self.total_frames_label.configure(text=f"/{total_frames}")
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, "1")
            self.update_displacement_preview()
            
            messagebox.showinfo("Success", "Processing completed!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"Error details: {str(e)}")  # Add detailed error output
        finally:
            # Restore interface
            self.root.config(cursor="")
            for child in self.root.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Radiobutton)):
                        widget.configure(state='normal')
            self.progress['value'] = 0

    def start_cutting_roi(self):
        """Start ROI cutting mode"""
        if self.roi_mask is None:
            messagebox.showerror("Error", "Please draw initial ROI first")
            return
        
        self.is_cutting_mode = True
        self.drawing_roi = True
        self.roi_points = []
        
        # Bind mouse events
        self.roi_canvas.bind('<Button-1>', self.add_roi_point)
        self.roi_canvas.bind('<Motion>', self.update_roi_preview)
        self.roi_canvas.bind('<Double-Button-1>', self.finish_roi_drawing)
        
        # Keep existing ROI display
        preview = self.current_image.copy()
        overlay = np.zeros_like(preview)
        overlay[self.roi_mask] = [0, 255, 0]  # Green indicates selected area
        preview = cv2.addWeighted(preview, 1, overlay, 0.3, 0)
        
        # Update display
        self.update_roi_label(preview)
        
        h, w = self.current_image.shape[:2]
        self.roi_scale_factor = min(400/w, 400/h)
        self.display_size = (int(w * self.roi_scale_factor), int(h * self.roi_scale_factor))

    def start_roi_drawing(self):
        """Start ROI drawing"""
        if self.current_image is None:
            messagebox.showerror("Error", "Please select input images first")
            return
        
        self.drawing_roi = True
        self.roi_points = []
        
        # If in cutting mode, keep existing ROI display
        if self.is_cutting_mode and self.roi_mask is not None:
            preview = self.current_image.copy()
            overlay = np.zeros_like(preview)
            overlay[self.roi_mask] = [0, 255, 0]  # Green indicates selected area
            preview = cv2.addWeighted(preview, 1, overlay, 0.3, 0)
            self.update_roi_label(preview)
        else:
            # Normal mode, display original image
            self.update_roi_label(self.current_image)
        
        # Bind mouse events
        self.roi_canvas.bind('<Button-1>', self.add_roi_point)
        self.roi_canvas.bind('<Motion>', self.update_roi_preview)
        self.roi_canvas.bind('<Double-Button-1>', self.finish_roi_drawing)
        
        h, w = self.current_image.shape[:2]
        self.roi_scale_factor = min(400/w, 400/h)
        self.display_size = (int(w * self.roi_scale_factor), int(h * self.roi_scale_factor))

    def add_roi_point(self, event):
        """Add ROI point"""
        if not self.drawing_roi:
            return
        
        # Get actual coordinates on canvas (considering scroll position)
        canvas_x = self.roi_canvas.canvasx(event.x)
        canvas_y = self.roi_canvas.canvasy(event.y)
        
        # Convert canvas coordinates to image coordinates (considering scaling)
        x = canvas_x / self.zoom_factor
        y = canvas_y / self.zoom_factor
        
        # Add point to list
        self.roi_points.append((x, y))
        
        # Draw point on canvas
        self.roi_canvas.create_oval(
            canvas_x-2, canvas_y-2,
            canvas_x+2, canvas_y+2,
            fill='red'
        )
        
        # If not first point, draw connecting line
        if len(self.roi_points) > 1:
            prev_x = self.roi_points[-2][0] * self.zoom_factor
            prev_y = self.roi_points[-2][1] * self.zoom_factor
            self.roi_canvas.create_line(
                prev_x, prev_y,
                canvas_x, canvas_y,
                fill='green', width=2
            )

    def update_roi_preview(self, event):
        """Update ROI preview"""
        if not self.drawing_roi or len(self.roi_points) == 0:
            return
        
        # Get coordinates of last point
        last_x = int(self.roi_points[-1][0] * self.roi_scale_factor)
        last_y = int(self.roi_points[-1][1] * self.roi_scale_factor)
        
        # Delete previous preview line
        self.roi_canvas.delete('preview_line')
        
        # Draw new preview line
        self.roi_canvas.create_line(last_x, last_y, event.x, event.y,
                                  fill='yellow', width=2, tags='preview_line')

    def finish_roi_drawing(self, event):
        """Finish ROI drawing"""
        if len(self.roi_points) < 3:
            messagebox.showerror("Error", "Please select at least 3 points")
            return
        
        self.drawing_roi = False
        
        # Draw closed line segment
        first_x = int(self.roi_points[0][0] * self.roi_scale_factor)
        first_y = int(self.roi_points[0][1] * self.roi_scale_factor)
        last_x = int(self.roi_points[-1][0] * self.roi_scale_factor)
        last_y = int(self.roi_points[-1][1] * self.roi_scale_factor)
        self.roi_canvas.create_line(last_x, last_y, first_x, first_y, 
                                  fill='green', width=2)
        
        # Unbind mouse events
        self.roi_canvas.unbind('<Button-1>')
        self.roi_canvas.unbind('<Motion>')
        self.roi_canvas.unbind('<Double-Button-1>')
        
        # Create or update ROI mask
        if self.is_cutting_mode and self.roi_mask is not None:
            # In cutting mode, subtract new drawn area from existing ROI
            new_mask = np.zeros_like(self.roi_mask, dtype=bool)
            points = np.array(self.roi_points, dtype=np.int32)
            cv2.fillPoly(new_mask.view(np.uint8), [points], 1)
            self.roi_mask = self.roi_mask & ~new_mask
        else:
            # Normal mode create new ROI
            self.create_roi_mask()
        
        # Reset cutting mode
        self.is_cutting_mode = False
        
        # Update display, including semi-transparent mask
        self.update_roi_display()
        
        # Show confirm button and enable
        self.confirm_roi_btn.grid()
        self.confirm_roi_btn.configure(state='normal')

    def create_roi_mask(self):
        """Create ROI mask"""
        if not self.roi_points or len(self.roi_points) < 3:
            return
        
        # Get image dimensions
        h, w = self.current_image.shape[:2]
        
        # Create mask
        if self.roi_mask is None:
            self.roi_mask = np.zeros((h, w), dtype=bool)
        
        # Convert point coordinates to integer array
        points = np.array(self.roi_points, dtype=np.int32)
        
        # Fill polygon
        cv2.fillPoly(self.roi_mask.view(np.uint8), [points], 1)

    def extract_roi_rectangle(self):
        """Extract smallest rectangle containing ROI"""
        if self.roi_mask is None:
            return
        
        # Find non-zero point coordinates
        rows, cols = np.nonzero(self.roi_mask)
        if len(rows) == 0 or len(cols) == 0:
            return
        
        # Get boundaries
        ymin, ymax = rows.min(), rows.max()
        xmin, xmax = cols.min(), cols.max()
        
        # Store rectangle coordinates
        self.roi_rect = (xmin, ymin, xmax+1, ymax+1)

    def update_roi_display(self):
        """Update ROI display, including semi-transparent mask"""
        if self.current_image is None or self.roi_mask is None:
            return
        
        # Create preview image with semi-transparent mask
        preview = self.current_image.copy()
        overlay = np.zeros_like(preview)
        overlay[self.roi_mask] = [0, 255, 0]  # Green indicates selected area
        preview = cv2.addWeighted(preview, 1, overlay, 0.3, 0)
        
        # Update ROI preview
        self.update_roi_label(preview)

    def clear_roi(self):
        """Clear ROI selection"""
        self.roi_points = []
        self.roi_mask = None
        self.roi_rect = None
        self.is_cutting_mode = False
        
        # Reset preview
        if self.current_image is not None:
            self.update_roi_label(self.current_image)
        
        # Reset main preview area
        self.update_preview()
        
        # Hide confirm button
        self.confirm_roi_btn.grid_remove()
        
        # Stop playback
        self.is_playing = False
        if hasattr(self, 'play_button'):
            self.play_button.configure(text=self.play_icon)
        if self.play_after_id:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None

    def confirm_roi(self):
        """Confirm ROI selection and update preview"""
        if self.roi_mask is None:
            messagebox.showerror("Error", "Please draw ROI first")
            return
        
        # Extract largest rectangle containing ROI
        self.extract_roi_rectangle()
        
        # Get ROI rectangle area
        if self.roi_rect:
            xmin, ymin, xmax, ymax = self.roi_rect
            
            # Update crop area preview
            self.update_preview()
            
            # Update information display
            roi_info = f"\nROI size: {xmax-xmin}x{ymax-ymin}"
            self.image_info.config(text=self.image_info.cget("text") + roi_info)

    def update_roi_label(self, image):
        """Update ROI preview label"""
        if image is None:
            return
        
        # Get original image dimensions
        h, w = image.shape[:2]
        
        # Calculate display size
        display_w = int(w * self.zoom_factor)
        display_h = int(h * self.zoom_factor)
        
        # Adjust image size
        resized = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        
        # Convert to RGB format
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
        elif resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Create PhotoImage
        image_pil = Image.fromarray(resized)
        self.current_photo = ImageTk.PhotoImage(image_pil)
        
        # Clear canvas and display new image
        self.roi_canvas.delete("all")
        self.roi_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_photo)
        
        # Redraw existing ROI points and lines
        if self.roi_points:
            for i, point in enumerate(self.roi_points):
                x, y = point
                canvas_x = x * self.zoom_factor
                canvas_y = y * self.zoom_factor
                
                # Draw point
                self.roi_canvas.create_oval(
                    canvas_x-2, canvas_y-2,
                    canvas_x+2, canvas_y+2,
                    fill='red'
                )
                
                # Draw line
                if i > 0:
                    prev_x = self.roi_points[i-1][0] * self.zoom_factor
                    prev_y = self.roi_points[i-1][1] * self.zoom_factor
                    self.roi_canvas.create_line(
                        prev_x, prev_y,
                        canvas_x, canvas_y,
                        fill='green', width=2
                    )
        
        # Update scroll area
        self.roi_canvas.configure(scrollregion=(0, 0, display_w, display_h))

    def on_mousewheel(self, event):
        """Handle mouse wheel events"""
        if self.current_image is None:
            return
            
        # Get current mouse position (relative to canvas)
        x = self.roi_canvas.canvasx(event.x)
        y = self.roi_canvas.canvasy(event.y)
        
        # Windows platform
        if event.delta:
            if event.delta > 0:
                self.zoom(1.1, x, y)
            else:
                self.zoom(0.9, x, y)
        # Linux/Mac platform
        else:
            if event.num == 4:
                self.zoom(1.1, x, y)
            elif event.num == 5:
                self.zoom(0.9, x, y)

    def zoom(self, factor, x=None, y=None):
        """Scale image"""
        if self.current_image is None:
            return
            
        # Update scaling factor
        old_zoom = self.zoom_factor
        self.zoom_factor *= factor
        
        # Limit scaling range
        min_zoom = 0.1
        max_zoom = 5.0
        self.zoom_factor = max(min_zoom, min(max_zoom, self.zoom_factor))
        
        # If scaling factor did not change, return immediately
        if self.zoom_factor == old_zoom:
            return
        
        # Save current scroll position
        old_x = self.roi_canvas.canvasx(0)
        old_y = self.roi_canvas.canvasy(0)
        
        # Update display
        self.update_roi_label(self.current_image)
        
        # If mouse position provided, adjust scroll position to keep point under mouse
        if x is not None and y is not None:
            # Calculate new scroll position
            new_x = x * (self.zoom_factor / old_zoom) - event.x
            new_y = y * (self.zoom_factor / old_zoom) - event.y
            
            # Set new scroll position
            self.roi_canvas.xview_moveto(new_x / self.roi_canvas.winfo_width())
            self.roi_canvas.yview_moveto(new_y / self.roi_canvas.winfo_height())

    def reset_zoom(self):
        """Reset scaling"""
        if self.current_image is None:
            return
            
        self.zoom_factor = 1.0
        self.update_roi_label(self.current_image)
        
        # Reset scrollbar position
        self.roi_canvas.xview_moveto(0)
        self.roi_canvas.yview_moveto(0)

    def jump_to_frame(self):
        """Jump to specified frame"""
        try:
            frame_num = int(self.frame_entry.get())
            total_frames = len(self.displacement_results)
            if 1 <= frame_num <= total_frames:
                self.frame_slider.set(frame_num)
                self.update_displacement_preview()
            else:
                messagebox.showerror("Error", f"Frame number must be between 1 and {total_frames}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid frame number")

    def toggle_play(self):
        """Toggle play/pause state"""
        if not self.displacement_results:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            # Update button display to pause icon
            self.play_button.configure(text=self.pause_icon)
            # Start playback
            self.play_next_frame()
        else:
            # Update button display to play icon
            self.play_button.configure(text=self.play_icon)
            # Stop playback
            if self.play_after_id:
                self.root.after_cancel(self.play_after_id)
                self.play_after_id = None

    def play_next_frame(self):
        """Play next frame"""
        if not self.is_playing:
            return
        
        current_frame = int(self.frame_slider.get())
        total_frames = len(self.displacement_results)
        
        # Calculate next frame
        next_frame = current_frame + 1
        if next_frame > total_frames:
            next_frame = 1  # Loop playback
        
        # Update slider position
        self.frame_slider.set(next_frame)
        
        # Schedule next frame playback
        self.play_after_id = self.root.after(self.play_interval, self.play_next_frame)

    def change_play_speed(self, *args):
        """Change playback speed"""
        speed_map = {
            "0.25x": 400,
            "0.5x": 200,
            "1x": 100,
            "2x": 50,
            "4x": 25
        }
        self.play_interval = speed_map[self.speed_var.get()]
        
        # If playing, restart playback to apply new speed
        if self.is_playing:
            if self.play_after_id:
                self.root.after_cancel(self.play_after_id)
            self.play_next_frame()

    def on_param_change(self, event=None):
        """Update preview when parameter input is complete"""
        self.update_preview()



if __name__ == '__main__':
    root = tk.Tk()
    app = RAFTDICGUI(root)
    root.mainloop()