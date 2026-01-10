import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from raft_dic_gui.ui_components import CTkCollapsibleFrame as CollapsibleFrame, Tooltip
import raft_dic_gui.model as mdl

from raft_dic_gui.views.post_processing_panel import PostProcessingPanel

class ControlPanel(ttk.Frame):
    def __init__(self, parent, callbacks=None, config=None):
        super().__init__(parent)
        self.callbacks = callbacks or {}
        self.config = config
        
        # Initialize variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.model_summary_text = tk.StringVar(value="No checkpoint selected.")
        
        # Processing parameter variables
        self.mode = tk.StringVar(value="accumulative")
        self.use_crop = tk.BooleanVar(value=False)
        self.crop_size_w = tk.StringVar(value="0")
        self.crop_size_h = tk.StringVar(value="0")
        self.shift_size = tk.StringVar(value="1800")
        
        # Visualization settings
        self.use_fixed_colorbar = tk.BooleanVar(value=False)
        self.colorbar_u_min = tk.StringVar(value="-1")
        self.colorbar_u_max = tk.StringVar(value="1")
        self.colorbar_v_min = tk.StringVar(value="-1")
        self.colorbar_v_max = tk.StringVar(value="1")
        self.symmetric_colorbar = tk.BooleanVar(value=False)
        self.colormap = tk.StringVar(value="turbo")
        self.overlay_alpha = tk.StringVar(value="0.5")
        self.fixed_range_frame = tk.StringVar(value="1")
        
        # Physical Units
        self.use_physical_units = tk.BooleanVar(value=False)
        self.physical_ratio = tk.StringVar(value="1.0")
        self.physical_unit = tk.StringVar(value="mm")
        
        # Time Step
        self.time_step = tk.StringVar(value="1.0")
        # Smoothing
        self.use_smooth = tk.BooleanVar(value=True)
        self.sigma = tk.StringVar(value="2.0")
        
        # Tiling
        self.context_padding = tk.StringVar(value="32")
        self.tile_overlap = tk.StringVar(value="32")
        self.safety_factor = tk.StringVar(value="0.55")
        self.p_max_pixels = tk.StringVar(value="1100*1100")
        self.show_tiles = tk.BooleanVar(value=False)
        
        # Background Image Mode
        self.background_mode = tk.StringVar(value="reference")
        self.use_smooth_interpolation = tk.BooleanVar(value=True)
        self.deform_display_mode = tk.StringVar(value="heatmap")
        self.deform_interp = tk.StringVar(value="linear")
        self.show_deformed_boundary = tk.BooleanVar(value=True)
        self.quiver_step = tk.StringVar(value="20")
        
        # Tooltips
        self.tooltips = {
            "path": "Select input image directory and output directory for results",
            "model": "Choose the RAFT checkpoint to run; settings are inferred automatically per file",
            "mode": "Accumulative: Calculate displacement relative to first frame\nIncremental: Calculate displacement relative to previous frame",
            "crop": "Enable/disable image cropping for processing",
            "crop_size": "Size of the cropping window (Width x Height)",
            "shift": "Step size for moving the cropping window",
            "smooth": "Gaussian smoothing (sigma in pixels). Larger sigma = smoother, less detail. Typical 0.5-5.0.",
            "vis": "Visualization settings including colorbar range and colormap",
            "run": "Start processing the image sequence",
            "tiling": "Tiling Settings:\nContext Padding: Extra pixels around ROI for context (default 32).\nTile Overlap: Overlap between tiles for smooth fusion (default 32).",
            "safety": "Safety Factor: Controls VRAM usage aggressiveness.\n0.2: Very Safe (Slow)\n0.4: Default\n0.55: Aggressive (Fast)\n0.7: Extreme (Risk of OOM)",
            "pmax": "Pixel budget per RAFT call (tile area). This is a limit on Total Pixels (W*H). E.g., 140*140 (~20k px) allows 200x100. Lower to reduce VRAM usage; higher risks OOM.",
            "tiles_overlay": "Draw tiles and their valid interiors over ROI preview for debugging."
        }
        
        self.create_widgets()

    def _trigger(self, key, *args):
        """Safely trigger a callback if it exists."""
        if self.callbacks.get(key):
            self.callbacks[key](*args)

    def select_tab(self, index):
        """Programmatically switch tabs."""
        try:
            self.notebook.select(index)
        except Exception:
            pass

    def create_widgets(self):
        # Create Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        
        # Tab 1: DIC Parameters
        self.dic_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dic_tab, text="DIC Parameters")
        
        # Tab 2: Post-Processing
        self.post_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.post_tab, text="Post-Processing")
        
        # Initialize PostProcessingPanel inside the tab
        self.post_processing_panel = PostProcessingPanel(self.post_tab, config=self.config, callbacks=self.callbacks)
        self.post_processing_panel.pack(fill="both", expand=True)
        
        # --- DIC Tab Content (Existing Scrollable Canvas) ---
        # Main control frame with scrollbar, parented to dic_tab
        control_canvas = tk.Canvas(self.dic_tab)
        scrollbar = ttk.Scrollbar(self.dic_tab, orient="vertical", command=control_canvas.yview)
        control_frame = ttk.Frame(control_canvas)
        
        # Configure scrolling
        control_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Grid layout for scrollbar and canvas within dic_tab
        control_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure parent grid (dic_tab)
        self.dic_tab.grid_rowconfigure(0, weight=1)
        self.dic_tab.grid_columnconfigure(0, weight=1)
        
        # Create window in canvas
        canvas_frame = control_canvas.create_window((0, 0), window=control_frame, anchor="nw")
        
        # Configure control frame
        control_frame.grid_columnconfigure(0, weight=1)
        
        # Path selection section
        path_frame = CollapsibleFrame(control_frame, text="Path Settings")
        path_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Add help button to header
        help_btn = self.create_help_button(path_frame.header_frame, "path")
        help_btn.grid(row=0, column=1, padx=2)
        
        path_content = path_frame.get_content_frame()
        path_content.grid_columnconfigure(1, weight=1)
        
        # Input path
        ttk.Label(path_content, text="Input Directory:").grid(row=0, column=0, sticky="w", padx=5)
        input_entry = ttk.Entry(path_content, textvariable=self.input_path)
        input_entry.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(path_content, text="Browse", command=lambda: self._trigger('browse_input'), width=8).grid(row=0, column=2, padx=5)
        
        # Model selection
        ttk.Label(path_content, text="Model:").grid(row=1, column=0, sticky="w", padx=5, pady=(5, 0))
        self.model_combobox = ttk.Combobox(path_content, textvariable=self.selected_model, values=(), width=38, state="readonly")
        self.model_combobox.grid(row=1, column=1, sticky="ew", padx=5, pady=(5, 0))
        try:
            self.model_combobox.bind('<<ComboboxSelected>>', lambda e: self.on_model_selected())
        except Exception:
            pass
        ttk.Button(path_content, text="Refresh", command=self.refresh_model_list, width=8).grid(row=1, column=2, padx=5, pady=(5, 0))
        ttk.Label(path_content, textvariable=self.model_summary_text, justify="left", wraplength=360).grid(
            row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(2, 6)
        )
        self.refresh_model_list(initial=True)
        
        # Add separator
        ttk.Separator(control_frame, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=5)
        
        # Processing mode section
        mode_frame = CollapsibleFrame(control_frame, text="Processing Mode")
        mode_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Add help button to header
        help_btn = self.create_help_button(mode_frame.header_frame, "mode")
        help_btn.grid(row=0, column=1, padx=2)
        
        mode_content = mode_frame.get_content_frame()
        mode_content.grid_columnconfigure(0, weight=1)
        
        # Mode radio buttons laid out horizontally
        ttk.Radiobutton(mode_content, text="Accumulative", variable=self.mode,
                       value="accumulative", command=self._on_mode_change).grid(row=0, column=0, sticky="w", padx=20, pady=2)
        ttk.Radiobutton(mode_content, text="Incremental", variable=self.mode,
                       value="incremental", command=self._on_mode_change).grid(row=0, column=1, sticky="w", padx=10, pady=2)
        
        # Key Frame Selector (shown only for incremental mode)
        self.key_frame_container = ttk.LabelFrame(mode_content, text="Key Frames (Reference Update Points)", padding=5)
        self.key_frame_container.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.key_frame_container.grid_columnconfigure(0, weight=1)
        
        # Info label
        ttk.Label(self.key_frame_container, text="Select frames to use as new references:").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))
        
        # Scrollable frame for checkboxes (will be populated when images are loaded)
        self.key_frame_scroll_frame = ttk.Frame(self.key_frame_container)
        self.key_frame_scroll_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        
        # Placeholder label (shown when no images loaded)
        self.key_frame_placeholder = ttk.Label(self.key_frame_scroll_frame, 
                                                text="Load images to see available frames",
                                                foreground="gray")
        self.key_frame_placeholder.pack(pady=10)
        
        # Quick selection buttons
        btn_frame = ttk.Frame(self.key_frame_container)
        btn_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=(5, 0))
        ttk.Button(btn_frame, text="Every N Frames:", width=14, 
                  command=self._select_every_n_frames).pack(side="left")
        self.every_n_entry = ttk.Entry(btn_frame, width=5)
        self.every_n_entry.insert(0, "50")
        self.every_n_entry.pack(side="left", padx=5)
        
        # Initially hide key frame selector (show only when incremental mode)
        self.key_frame_container.grid_remove()
        
        # Storage for key frame checkboxes
        self.key_frame_vars = {}  # frame_index -> BooleanVar
        
        # Add separator
        ttk.Separator(control_frame, orient="horizontal").grid(row=3, column=0, sticky="ew", pady=5)
        
        # Parameters section
        param_frame = CollapsibleFrame(control_frame, text="Parameters")
        param_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        
        param_content = param_frame.get_content_frame()
        param_content.grid_columnconfigure(0, weight=1)

        # Advanced parameters (collapsible, hidden by default)
        advanced_cf = CollapsibleFrame(param_content, text="Advanced Parameters")
        advanced_cf.grid(row=1, column=0, sticky="ew", pady=5)
        adv_content = advanced_cf.get_content_frame()
        adv_content.grid_columnconfigure(1, weight=1)
        try:
            advanced_cf.toggle() # Start collapsed
        except Exception:
            pass

        # Tiling Settings
        tile_frame = ttk.LabelFrame(adv_content, text="Tiling Settings", padding=5)
        tile_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=4)
        tile_frame.grid_columnconfigure(1, weight=1)

        # Context Padding
        ttk.Label(tile_frame, text="Context Padding (px):").grid(row=0, column=0, sticky="w", padx=5)
        pad_combo = ttk.Combobox(tile_frame, textvariable=self.context_padding, 
                                values=("16", "32", "64", "128"), width=10)
        pad_combo.grid(row=0, column=1, sticky="w")
        
        # Tile Overlap
        ttk.Label(tile_frame, text="Tile Overlap (px):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        overlap_combo = ttk.Combobox(tile_frame, textvariable=self.tile_overlap,
                                    values=("16", "32", "64"), width=10)
        overlap_combo.grid(row=1, column=1, sticky="w", pady=2)
        overlap_combo.bind('<<ComboboxSelected>>', lambda e: self._trigger('on_overlap_change'))

        # Safety Factor
        ttk.Label(tile_frame, text="Safety Factor:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.safety_combo = ttk.Combobox(tile_frame, textvariable=self.safety_factor,
                                    values=("0.2", "0.4", "0.55", "0.7", "1.0"), width=10)
        self.safety_combo.grid(row=2, column=1, sticky="w", pady=2)
        self.safety_combo.bind('<<ComboboxSelected>>', lambda e: self._trigger('on_safety_factor_change'))
        
        # Help button for safety
        hp_safe = self.create_help_button(tile_frame, "safety")
        hp_safe.grid(row=2, column=2, padx=6)

        # Show Tiles Checkbox
        ttk.Checkbutton(tile_frame, text="Show Tiles", variable=self.show_tiles,
                       command=lambda: self._trigger('on_show_tiles_change')).grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # Help button for tiling
        help_btn = self.create_help_button(tile_frame, "tiling")
        help_btn.grid(row=0, column=2, padx=6)

        # Smoothing options
        smooth_frame = ttk.Frame(adv_content)
        smooth_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        
        ttk.Checkbutton(smooth_frame, text="Use Smoothing",
                        variable=self.use_smooth,
                        command=self.update_smoothing_state).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(smooth_frame, text="Sigma (px):").grid(row=0, column=1, sticky="w", padx=(10,0))
        self.sigma_entry = ttk.Entry(smooth_frame, textvariable=self.sigma, width=5)
        self.sigma_entry.grid(row=0, column=2, padx=5)
        self.sigma_hint = ttk.Label(smooth_frame, text="0.5-5.0")
        self.sigma_hint.grid(row=0, column=3, sticky="w")
        # Slider for sigma
        self.sigma_scale = ttk.Scale(smooth_frame, from_=0.5, to=5.0, orient=tk.HORIZONTAL,
                                     command=self.on_sigma_scale_change, length=100)
        self.sigma_scale.grid(row=0, column=4, padx=5)
        
        # Visualization settings
        vis_frame = ttk.LabelFrame(param_content, text="Visualization Settings", padding=5)
        vis_frame.grid(row=2, column=0, sticky="ew", pady=5)
        vis_frame.grid_columnconfigure(0, weight=1)
        
        # Colorbar settings
        ttk.Checkbutton(vis_frame, text="Use Fixed Range", 
                       variable=self.use_fixed_colorbar,
                       command=self.callbacks.get('update_preview', lambda: None)).grid(row=0, column=0, sticky="w", padx=5)
        
        range_frame = ttk.Frame(vis_frame)
        range_frame.grid(row=1, column=0, sticky="ew", pady=5, padx=5)
        range_frame.grid_columnconfigure(1, weight=1)
        range_frame.grid_columnconfigure(3, weight=1)
        
        # U Range
        ttk.Label(range_frame, text="U:").grid(row=0, column=0, sticky="w")
        umin_entry = ttk.Entry(range_frame, textvariable=self.colorbar_u_min, width=8)
        umin_entry.grid(row=0, column=1, padx=2)
        ttk.Label(range_frame, text="to").grid(row=0, column=2, padx=2)
        umax_entry = ttk.Entry(range_frame, textvariable=self.colorbar_u_max, width=8)
        umax_entry.grid(row=0, column=3, padx=2)
        
        # V Range
        ttk.Label(range_frame, text="V:").grid(row=1, column=0, sticky="w", pady=2)
        vmin_entry = ttk.Entry(range_frame, textvariable=self.colorbar_v_min, width=8)
        vmin_entry.grid(row=1, column=1, padx=2)
        ttk.Label(range_frame, text="to").grid(row=1, column=2, padx=2)
        vmax_entry = ttk.Entry(range_frame, textvariable=self.colorbar_v_max, width=8)
        vmax_entry.grid(row=1, column=3, padx=2)
        
        # Bind range entries
        def _on_range_edit(event=None):
            self.use_fixed_colorbar.set(True)
            self._trigger('on_param_change')

        for ent in (umin_entry, umax_entry, vmin_entry, vmax_entry):
            try:
                ent.bind('<FocusOut>', _on_range_edit)
                ent.bind('<Return>', _on_range_edit)
            except Exception:
                pass
        
        # Colormap selection
        ttk.Label(vis_frame, text="Colormap:").grid(row=2, column=0, sticky="w", padx=5, pady=(5,0))
        colormap_combo = ttk.Combobox(vis_frame, 
                                    textvariable=self.colormap,
                                    values=["turbo", "viridis", "jet", "magma", "plasma", "inferno", 
                                           "cividis", "RdYlBu", "coolwarm"],
                                    width=15,
                                    state="readonly")
        colormap_combo.grid(row=3, column=0, sticky="w", padx=5, pady=(0,5))
        colormap_combo.bind('<<ComboboxSelected>>', lambda e: self._trigger('on_param_change'))

        # Transparency
        alpha_row = ttk.Frame(vis_frame)
        alpha_row.grid(row=4, column=0, sticky='w', padx=5, pady=(0,5))
        ttk.Label(alpha_row, text="Transparency:").grid(row=0, column=0, padx=(0,6))
        alpha_entry = ttk.Entry(alpha_row, textvariable=self.overlay_alpha, width=6)
        alpha_entry.grid(row=0, column=1, padx=(0,8))
        ttk.Button(alpha_row, text="Update", command=lambda: self._trigger('update_preview')).grid(row=0, column=2)

        # Physical Units
        phys_frame = ttk.Frame(vis_frame)
        phys_frame.grid(row=5, column=0, sticky='ew', padx=5, pady=(0,5))
        ttk.Checkbutton(phys_frame, text="Physical Units", variable=self.use_physical_units,
                        command=lambda: self._trigger('update_preview')).pack(side="left")
        ttk.Label(phys_frame, text="Ratio:").pack(side="left", padx=(5, 2))
        ratio_entry = ttk.Entry(phys_frame, textvariable=self.physical_ratio, width=5)
        ratio_entry.pack(side="left")
        
        unit_cb = ttk.Combobox(phys_frame, textvariable=self.physical_unit, 
                              values=["mm", "um", "nm", "m", "inch"], width=4, state="readonly")
        unit_cb.pack(side="left", padx=(2,0))
        unit_cb.bind("<<ComboboxSelected>>", lambda e: self._trigger('update_preview'))

        ttk.Label(phys_frame, text="dt(s):").pack(side="left", padx=(5, 2))
        dt_entry = ttk.Entry(phys_frame, textvariable=self.time_step, width=5)
        dt_entry.pack(side="left")

        # Fixed range from specific frame
        fixed_from_frame = ttk.Frame(vis_frame)
        fixed_from_frame.grid(row=6, column=0, sticky='w', padx=5, pady=(0,5))
        ttk.Label(fixed_from_frame, text="Range from Frame:").grid(row=0, column=0, padx=(0,6))
        ttk.Entry(fixed_from_frame, textvariable=self.fixed_range_frame, width=5).grid(row=0, column=1, padx=(0,8))
        ttk.Button(fixed_from_frame, text="Apply", command=lambda: self._trigger('set_fixed_colorbar')).grid(row=0, column=2)

        # Background Image Mode selection (Global display mode)
        bg_mode_frame = ttk.LabelFrame(vis_frame, text="Display Background", padding=5)
        bg_mode_frame.grid(row=7, column=0, sticky="ew", pady=5)
        bg_mode_frame.grid_columnconfigure(0, weight=1)
        
        # Mode radio buttons - simple toggle
        ttk.Radiobutton(bg_mode_frame, text="Reference Frame (Frame 1)", variable=self.background_mode, 
                       value="reference", command=self._on_display_mode_change).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.deformed_radio = ttk.Radiobutton(bg_mode_frame, text="Current Deformed Frame", variable=self.background_mode, 
                       value="deformed", command=self._on_display_mode_change, state='disabled')
        self.deformed_radio.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        
        # Info label for deformed mode (shown when deformed is selected)
        self.deformed_info_label = ttk.Label(bg_mode_frame, 
            text="ℹ️ Values mapped to deformed coordinates.\n   Underlying data remains in Lagrangian definition.",
            foreground="gray", font=("TkDefaultFont", 8))
        self.deformed_info_label.grid(row=2, column=0, sticky="w", padx=20, pady=2)
        self.deformed_info_label.grid_remove()  # Hidden by default
        
        # Update info label visibility when mode changes
        def _update_info_visibility(*_):
            try:
                if self.background_mode.get() == 'deformed':
                    self.deformed_info_label.grid()
                else:
                    self.deformed_info_label.grid_remove()
            except Exception:
                pass
        self.background_mode.trace_add('write', _update_info_visibility)
        
        # Add separator
        ttk.Separator(control_frame, orient="horizontal").grid(row=5, column=0, sticky="ew", pady=5)
        
        # Run section
        run_frame = CollapsibleFrame(control_frame, text="Run Control")
        run_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
        
        run_content = run_frame.get_content_frame()
        run_content.grid_columnconfigure(0, weight=1)
        
        # Run/Stop controls and progress
        self.run_button = ttk.Button(run_content, text="Run", command=lambda: self._trigger('run'), width=12)
        self.run_button.grid(row=0, column=0, pady=5, padx=(0,6))
        self.stop_button = ttk.Button(run_content, text="Stop", command=lambda: self._trigger('stop'), width=12, state='disabled')
        self.stop_button.grid(row=0, column=1, pady=5)
        self.progress = ttk.Progressbar(run_content, length=220, mode='determinate')
        self.progress.grid(row=1, column=0, columnspan=2, pady=5, sticky='ew')
        self.progress_text = ttk.Label(run_content, text="0/0")
        self.progress_text.grid(row=2, column=0, columnspan=2)
        self.time_log = tk.Text(run_content, height=5, wrap="word", state="disabled")
        self.time_log.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        
        # Status Label
        self.status_label = ttk.Label(run_content, text="Ready", anchor="w")
        self.status_label.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        run_content.grid_columnconfigure(0, weight=1)
        run_content.grid_columnconfigure(1, weight=1)
        
        # Configure canvas scrolling
        def configure_scroll_region(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))

        def configure_canvas_width(event):
            control_canvas.itemconfig(canvas_frame, width=event.width)
        
        control_frame.bind("<Configure>", configure_scroll_region)
        control_canvas.bind("<Configure>", configure_canvas_width)
        
        # Bind mouse wheel
        def on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind mouse wheel only to this canvas to avoid affecting other panes
        control_canvas.bind("<MouseWheel>", on_mousewheel)

        # Initialize UI states
        self.update_max_disp_hint()
        self.update_smoothing_state()

    def create_help_button(self, parent, key):
        """Create a help button with tooltip"""
        btn = ttk.Button(parent, text="?", width=2)
        if key in self.tooltips:
            Tooltip(btn, self.tooltips[key])
        return btn

    def update_crop_state(self):
        """Enable/disable crop inputs based on checkbox"""
        state = "normal" if self.use_crop.get() else "disabled"
        self.crop_w_entry.configure(state=state)
        self.crop_h_entry.configure(state=state)
        self.shift_entry.configure(state=state)

    def update_max_disp_hint(self):
        """Update hint text for max displacement"""
        pass # Logic removed in original

    def update_smoothing_state(self):
        """Enable/disable smoothing inputs"""
        state = "normal" if self.use_smooth.get() else "disabled"
        self.sigma_entry.configure(state=state)
        self.sigma_scale.configure(state=state)

    def on_sigma_scale_change(self, value):
        """Update sigma entry when scale moves"""
        self.sigma.set(f"{float(value):.2f}")



    def on_sigma_entry_change(self, event=None):
        """Update sigma scale when entry changes"""
        try:
            val = float(self.sigma.get())
            self.sigma_scale.set(val)
        except ValueError:
            pass

    def refresh_model_list(self, *_args, initial: bool = False):
        """Populate available RAFT checkpoints in the model selector."""
        try:
            entries = mdl.discover_models()
        except Exception as exc:
            entries = []
            print(f"Error discovering models: {exc}")
        
        self.available_models = entries
        self.model_lookup = {e.label: e for e in entries}
        
        names = [e.label for e in entries]
        self.model_combobox['values'] = names
        
        if initial and names:
            self.model_combobox.current(0)
            self.selected_model.set(names[0]) # Explicitly update StringVar
            self.on_model_selected()

    # Obsolete methods removed: _map_disp_selection, on_disp_preset_change

    def update_config(self, config):
        """Update the configuration object with current UI values."""
        config.img_dir = self.input_path.get()
        # Output directory is deprecated in UI, defaulting to input or empty
        config.project_root = self.input_path.get()
        config.mode = self.mode.get()
        config.crop_size = (int(self.crop_size_h.get() or "0"), int(self.crop_size_w.get() or "0"))
        config.shift = int(self.shift_size.get() or "0")
        
        config.use_smooth = self.use_smooth.get()
        config.sigma = float(self.sigma.get())
        
        try:
            config.context_padding = int(self.context_padding.get())
        except Exception:
            config.context_padding = 32
            
        try:
            config.tile_overlap = int(self.tile_overlap.get())
        except Exception:
            config.tile_overlap = 32
            
        try:
            config.safety_factor = float(self.safety_factor.get())
        except Exception:
            config.safety_factor = 0.55

        # Parse pixel budget input
        try:
            s = (self.p_max_pixels.get() or "").lower().replace(' ', '')
            if '*' in s:
                a, b = s.split('*', 1)
                config.p_max_pixels = int(float(a)) * int(float(b))
            elif 'x' in s:
                a, b = s.split('x', 1)
                config.p_max_pixels = int(float(a)) * int(float(b))
            else:
                config.p_max_pixels = int(float(s))
        except Exception:
            config.p_max_pixels = 1100 * 1100
        
        # Device
        # Device
        config.device = getattr(mdl, "DEFAULT_DEVICE", "cuda")
        
        # Model Path
        entry = self.get_selected_model_entry()
        if entry:
            config.model_path = entry.path
            config.model_label = entry.label
        else:
            # If no model selected, we should probably fail validation or set empty
            config.model_path = ""
            config.model_label = ""

    def get_selected_model_entry(self):
        selected_label = self.selected_model.get()
        return self.model_lookup.get(selected_label)

    def on_model_selected(self, *_args, show_errors: bool = False):
        """Update model summary and cached metadata when selection changes."""
        label = self.selected_model.get()
        entry = self.model_lookup.get(label)
        if not entry:
            self.model_summary_text.set("Select a model checkpoint.")
            if show_errors and self.available_models:
                tk.messagebox.showwarning("Model Selection", "Please choose a valid checkpoint before running.")
            return

        try:
            metadata = mdl.describe_checkpoint(entry.path)
            summary = mdl.metadata_summary(metadata)
            self.model_summary_text.set(summary)
            
            # Auto-estimate safe pixel budget (Calculation only, display in ROI panel)
            # safe_pmax = mdl.estimate_safe_pmax(metadata)
            # self.p_max_pixels.set(f"{safe_pmax}*{safe_pmax}")
            
            # Notify controller if needed
            if self.callbacks.get('on_model_selected'):
                self.callbacks['on_model_selected']()
        except Exception as exc:
            self.model_summary_text.set(f"Failed to inspect checkpoint: {exc}")
            if show_errors:
                tk.messagebox.showerror("Model Inspection Failed", f"Could not read checkpoint metadata:\n{exc}")

    def _on_display_mode_change(self):
        """Handle display mode (reference/deformed) change."""
        mode = self.background_mode.get()
        print(f"[DeformedView] Display mode changed to: {mode}")
        self._trigger('update_preview')
    
    def enable_deformed_mode(self):
        """Enable deformed mode option after displacement computed."""
        try:
            self.deformed_radio.configure(state="normal")
            print("[DeformedView] Deformed mode enabled")
        except Exception as e:
            print(f"[DeformedView] Error enabling deformed mode: {e}")
    
    def disable_deformed_mode(self):
        """Disable and reset to reference mode."""
        try:
            self.background_mode.set("reference")
            self.deformed_radio.configure(state="disabled")
            print("[DeformedView] Deformed mode disabled, reset to reference")
        except Exception as e:
            print(f"[DeformedView] Error disabling deformed mode: {e}")

    def _on_mode_change(self):
        """Handle processing mode change (Accumulative/Incremental)."""
        mode = self.mode.get()
        if mode == "incremental":
            self.key_frame_container.grid()
        else:
            self.key_frame_container.grid_remove()
    
    def populate_key_frames(self, num_frames: int):
        """
        Populate key frame checkboxes based on loaded image count.
        
        Args:
            num_frames: Total number of frames in the sequence
        """
        # Clear existing checkboxes
        for widget in self.key_frame_scroll_frame.winfo_children():
            widget.destroy()
        self.key_frame_vars.clear()
        
        if num_frames <= 0:
            self.key_frame_placeholder = ttk.Label(self.key_frame_scroll_frame, 
                                                    text="Load images to see available frames",
                                                    foreground="gray")
            self.key_frame_placeholder.pack(pady=10)
            return
        
        # Create scrollable frame with checkboxes
        # Use a fixed-width container for horizontal wrapping
        container = ttk.Frame(self.key_frame_scroll_frame)
        container.pack(fill="x", expand=True)
        
        # Create checkboxes (wrap every 10 frames to a new row)
        row_frame = None
        frames_per_row = 8
        
        for i in range(1, num_frames + 1):
            if (i - 1) % frames_per_row == 0:
                row_frame = ttk.Frame(container)
                row_frame.pack(fill="x", pady=1)
            
            var = tk.BooleanVar(value=(i == 1))  # Frame 1 is default checked
            self.key_frame_vars[i] = var
            
            cb = ttk.Checkbutton(row_frame, text=str(i), variable=var, width=4)
            cb.pack(side="left", padx=2)
    
    def _select_every_n_frames(self):
        """Select key frames at regular intervals."""
        try:
            n = int(self.every_n_entry.get())
            if n <= 0:
                return
        except ValueError:
            return
        
        # Clear all except frame 1
        for frame_idx, var in self.key_frame_vars.items():
            if frame_idx == 1:
                var.set(True)  # Always keep frame 1
            elif (frame_idx - 1) % n == 0:
                var.set(True)
            else:
                var.set(False)
    
    def get_selected_key_frames(self) -> list:
        """
        Get list of selected key frame indices.
        
        Returns:
            List of frame indices (1-indexed) that are selected as key frames
        """
        selected = [frame_idx for frame_idx, var in self.key_frame_vars.items() if var.get()]
        return sorted(selected)

