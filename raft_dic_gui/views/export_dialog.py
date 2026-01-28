"""
Export Configuration Dialog

Modal dialog for configuring batch image export settings.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import customtkinter as ctk
from typing import List, Dict, Callable, Optional
import os


def _log(msg: str):
    """Print debug message with standard prefix."""
    print(f"[ImageExport] {msg}")


class ExportDialog(ctk.CTkToplevel):
    """Modal dialog for batch image export configuration."""
    
    def __init__(self, 
                 parent,
                 available_components: List[str],
                 total_frames: int,
                 default_settings: Dict,
                 current_component_ranges: Dict,
                 on_export: Callable[[Dict], None]):
        """
        Initialize the export dialog.
        
        Args:
            parent: Parent window
            available_components: List of component names ['u', 'v', 'exx', ...]
            total_frames: Total number of frames available
            default_settings: Dict with colormap, alpha, display_mode
            current_component_ranges: {comp_name: (vmin, vmax)} from current view
            on_export: Callback function receiving settings dict when Export clicked
        """
        super().__init__(parent)
        
        self.available_components = available_components
        self.total_frames = total_frames
        self.default_settings = default_settings
        self.current_component_ranges = current_component_ranges
        self.on_export = on_export
        
        _log(f"Dialog opened: {len(available_components)} components, {total_frames} frames")
        
        # Configure window
        self.title("Export Visualization Images")
        self.geometry("600x700")
        self.resizable(True, True)
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 600) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 700) // 2
        self.geometry(f"+{x}+{y}")
        
        # Component checkboxes and entries storage
        self.comp_vars = {}  # {name: BooleanVar}
        self.comp_vmin = {}  # {name: StringVar}
        self.comp_vmax = {}  # {name: StringVar}
        
        self._create_widgets()
        
        # Focus
        self.focus_set()
    
    def _create_widgets(self):
        """Create all dialog widgets."""
        # Main scrollable container
        main_frame = ctk.CTkScrollableFrame(self, width=560, height=600)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # === Output Directory ===
        self._create_output_section(main_frame)
        
        # === General Settings ===
        self._create_general_section(main_frame)
        
        # === Frame Selection ===
        self._create_frame_section(main_frame)
        
        # === Component Selection ===
        self._create_component_section(main_frame)
        
        # === Preview ===
        self._create_preview_section(main_frame)
        
        # === Buttons ===
        self._create_buttons()
    
    def _create_output_section(self, parent):
        """Create output directory selection section."""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(frame, text="Output Directory", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 10))
        
        self.output_dir = tk.StringVar(value=os.path.expanduser("~"))
        entry = ctk.CTkEntry(row, textvariable=self.output_dir, width=400)
        entry.pack(side="left", fill="x", expand=True)
        
        ctk.CTkButton(row, text="Browse...", width=80, 
                     command=self._browse_output).pack(side="right", padx=(10, 0))
    
    def _create_general_section(self, parent):
        """Create general settings section."""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(frame, text="General Settings", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        content = ctk.CTkFrame(frame, fg_color="transparent")
        content.pack(fill="x", padx=10, pady=(0, 10))
        
        # Display Mode
        row1 = ctk.CTkFrame(content, fg_color="transparent")
        row1.pack(fill="x", pady=2)
        ctk.CTkLabel(row1, text="Display Mode:", width=120, anchor="w").pack(side="left")
        self.display_mode = tk.StringVar(value=self.default_settings.get('display_mode', 'reference'))
        ctk.CTkRadioButton(row1, text="Reference", variable=self.display_mode, 
                          value="reference").pack(side="left", padx=10)
        ctk.CTkRadioButton(row1, text="Deformed", variable=self.display_mode, 
                          value="deformed").pack(side="left")
        
        # Colormap
        row2 = ctk.CTkFrame(content, fg_color="transparent")
        row2.pack(fill="x", pady=2)
        ctk.CTkLabel(row2, text="Colormap:", width=120, anchor="w").pack(side="left")
        self.colormap = tk.StringVar(value=self.default_settings.get('colormap', 'turbo'))
        colormaps = ["turbo", "viridis", "jet", "magma", "plasma", "inferno", "cividis", "RdYlBu", "coolwarm"]
        ctk.CTkComboBox(row2, variable=self.colormap, values=colormaps, width=150).pack(side="left")
        
        # Transparency
        row3 = ctk.CTkFrame(content, fg_color="transparent")
        row3.pack(fill="x", pady=2)
        ctk.CTkLabel(row3, text="Transparency:", width=120, anchor="w").pack(side="left")
        self.alpha = tk.StringVar(value=str(self.default_settings.get('alpha', 0.7)))
        ctk.CTkEntry(row3, textvariable=self.alpha, width=80).pack(side="left")
        
        # Image Format
        row4 = ctk.CTkFrame(content, fg_color="transparent")
        row4.pack(fill="x", pady=2)
        ctk.CTkLabel(row4, text="Image Format:", width=120, anchor="w").pack(side="left")
        self.format = tk.StringVar(value="png")
        formats = ["png", "jpg", "pdf", "svg"]
        ctk.CTkComboBox(row4, variable=self.format, values=formats, width=100).pack(side="left")
        
        # Checkboxes
        row5 = ctk.CTkFrame(content, fg_color="transparent")
        row5.pack(fill="x", pady=(5, 0))
        self.include_colorbar = tk.BooleanVar(value=True)
        self.include_title = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(row5, text="Include colorbar", variable=self.include_colorbar).pack(side="left", padx=(0, 20))
        ctk.CTkCheckBox(row5, text="Include title", variable=self.include_title).pack(side="left")
    
    def _create_frame_section(self, parent):
        """Create frame selection section."""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(frame, text="Frame Selection", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        content = ctk.CTkFrame(frame, fg_color="transparent")
        content.pack(fill="x", padx=10, pady=(0, 10))
        
        # Radio buttons
        self.frame_mode = tk.StringVar(value="all")
        
        row1 = ctk.CTkFrame(content, fg_color="transparent")
        row1.pack(fill="x", pady=2)
        ctk.CTkRadioButton(row1, text=f"All frames (1 to {self.total_frames})", 
                          variable=self.frame_mode, value="all").pack(side="left")
        
        row2 = ctk.CTkFrame(content, fg_color="transparent")
        row2.pack(fill="x", pady=2)
        ctk.CTkRadioButton(row2, text="Custom range:", variable=self.frame_mode, 
                          value="custom").pack(side="left")
        ctk.CTkLabel(row2, text="From").pack(side="left", padx=(10, 5))
        self.frame_start = tk.StringVar(value="1")
        ctk.CTkEntry(row2, textvariable=self.frame_start, width=60).pack(side="left")
        ctk.CTkLabel(row2, text="To").pack(side="left", padx=(10, 5))
        self.frame_end = tk.StringVar(value=str(self.total_frames))
        ctk.CTkEntry(row2, textvariable=self.frame_end, width=60).pack(side="left")
    
    def _create_component_section(self, parent):
        """Create component selection section with color ranges."""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 10))
        
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(header, text="Component Selection & Color Ranges", 
                    font=("", 14, "bold")).pack(side="left")
        
        # Buttons
        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right")
        ctk.CTkButton(btn_frame, text="Auto-fill", width=80, height=24,
                     command=self._autofill_ranges).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="Clear All", width=80, height=24,
                     command=self._clear_ranges).pack(side="left", padx=2)
        
        # Scrollable component list
        comp_frame = ctk.CTkScrollableFrame(frame, height=150)
        comp_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Header row
        header_row = ctk.CTkFrame(comp_frame, fg_color="transparent")
        header_row.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(header_row, text="Component", width=100, anchor="w").pack(side="left")
        ctk.CTkLabel(header_row, text="Min", width=100, anchor="w").pack(side="left", padx=(20, 0))
        ctk.CTkLabel(header_row, text="Max", width=100, anchor="w").pack(side="left", padx=(10, 0))
        
        # Component rows
        for comp in self.available_components:
            self._create_component_row(comp_frame, comp)
    
    def _create_component_row(self, parent, comp_name: str):
        """Create a row for a single component."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=1)
        
        # Checkbox
        var = tk.BooleanVar(value=(comp_name in ['u', 'v']))  # Default: u, v enabled
        self.comp_vars[comp_name] = var
        ctk.CTkCheckBox(row, text=comp_name, variable=var, width=100).pack(side="left")
        
        # Min entry
        vmin_var = tk.StringVar(value="")
        self.comp_vmin[comp_name] = vmin_var
        ctk.CTkEntry(row, textvariable=vmin_var, width=100, 
                    placeholder_text="Auto").pack(side="left", padx=(20, 0))
        
        # Max entry
        vmax_var = tk.StringVar(value="")
        self.comp_vmax[comp_name] = vmax_var
        ctk.CTkEntry(row, textvariable=vmax_var, width=100,
                    placeholder_text="Auto").pack(side="left", padx=(10, 0))
    
    def _create_preview_section(self, parent):
        """Create preview section showing output structure."""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(frame, text="Preview", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.preview_text = ctk.CTkTextbox(frame, height=100, state="disabled")
        self.preview_text.pack(fill="x", padx=10, pady=(0, 10))
        
        self._update_preview()
        
        # Bind changes to update preview
        self.frame_mode.trace_add("write", lambda *_: self._update_preview())
        self.frame_start.trace_add("write", lambda *_: self._update_preview())
        self.frame_end.trace_add("write", lambda *_: self._update_preview())
    
    def _create_buttons(self):
        """Create dialog buttons."""
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(btn_frame, text="Cancel", width=100, 
                     fg_color="gray", command=self.destroy).pack(side="right", padx=5)
        ctk.CTkButton(btn_frame, text="Export", width=100, 
                     command=self._on_export_clicked).pack(side="right", padx=5)
    
    def _browse_output(self):
        """Open directory browser."""
        path = filedialog.askdirectory(initialdir=self.output_dir.get())
        if path:
            self.output_dir.set(path)
            self._update_preview()
    
    def _autofill_ranges(self):
        """Fill color ranges from current view settings."""
        for comp_name, (vmin, vmax) in self.current_component_ranges.items():
            if comp_name in self.comp_vmin:
                self.comp_vmin[comp_name].set(f"{vmin:.4f}" if vmin is not None else "")
                self.comp_vmax[comp_name].set(f"{vmax:.4f}" if vmax is not None else "")
        _log("Auto-filled color ranges from current view")
    
    def _clear_ranges(self):
        """Clear all color range entries."""
        for var in self.comp_vmin.values():
            var.set("")
        for var in self.comp_vmax.values():
            var.set("")
        _log("Cleared all color ranges")
    
    def _update_preview(self):
        """Update the preview text."""
        try:
            # Count enabled components
            enabled = [name for name, var in self.comp_vars.items() if var.get()]
            
            # Get frame range
            if self.frame_mode.get() == "all":
                start, end = 1, self.total_frames
            else:
                try:
                    start = int(self.frame_start.get())
                    end = int(self.frame_end.get())
                except ValueError:
                    start, end = 1, self.total_frames
            
            num_frames = max(0, end - start + 1)
            total_images = len(enabled) * num_frames
            
            # Build preview text
            lines = [
                f"Output: {self.output_dir.get()}/export_YYYYMMDD_HHMMSS/",
                f"",
                f"Components: {', '.join(enabled) if enabled else '(none selected)'}",
                f"Frames: {start} to {end} ({num_frames} frames)",
                f"",
                f"Estimated: {total_images} images"
            ]
            
            self.preview_text.configure(state="normal")
            self.preview_text.delete("1.0", "end")
            self.preview_text.insert("1.0", "\n".join(lines))
            self.preview_text.configure(state="disabled")
            
        except Exception as e:
            _log(f"Error updating preview: {e}")
    
    def _validate_inputs(self) -> Optional[str]:
        """Validate all inputs. Returns error message or None if valid."""
        # Check output directory
        if not os.path.isdir(self.output_dir.get()):
            return "Output directory does not exist."
        
        # Check transparency
        try:
            alpha = float(self.alpha.get())
            if alpha < 0 or alpha > 1:
                return "Transparency must be between 0 and 1."
        except ValueError:
            return "Invalid transparency value."
        
        # Check frame range
        try:
            if self.frame_mode.get() == "custom":
                start = int(self.frame_start.get())
                end = int(self.frame_end.get())
                if start < 1 or end > self.total_frames or start > end:
                    return f"Frame range must be between 1 and {self.total_frames}."
        except ValueError:
            return "Invalid frame range values."
        
        # Check at least one component selected
        enabled = [name for name, var in self.comp_vars.items() if var.get()]
        if not enabled:
            return "Please select at least one component to export."
        
        return None
    
    def _get_export_settings(self) -> Dict:
        """Build and return the export settings dictionary."""
        # Frame range
        if self.frame_mode.get() == "all":
            frame_range = (1, self.total_frames)
        else:
            frame_range = (int(self.frame_start.get()), int(self.frame_end.get()))
        
        # Components
        components = {}
        for comp_name in self.available_components:
            if self.comp_vars[comp_name].get():
                vmin_str = self.comp_vmin[comp_name].get().strip()
                vmax_str = self.comp_vmax[comp_name].get().strip()
                
                vmin = float(vmin_str) if vmin_str else None
                vmax = float(vmax_str) if vmax_str else None
                
                components[comp_name] = {
                    'enabled': True,
                    'vmin': vmin,
                    'vmax': vmax
                }
        
        return {
            'output_dir': self.output_dir.get(),
            'display_mode': self.display_mode.get(),
            'colormap': self.colormap.get(),
            'alpha': float(self.alpha.get()),
            'format': self.format.get(),
            'include_colorbar': self.include_colorbar.get(),
            'include_title': self.include_title.get(),
            'frame_range': frame_range,
            'components': components,
            # Pass through physical units settings from default_settings
            'use_physical_units': self.default_settings.get('use_physical_units', False),
            'physical_ratio': self.default_settings.get('physical_ratio', 1.0),
            'physical_unit': self.default_settings.get('physical_unit', 'mm'),
            'fps': self.default_settings.get('fps', 1.0),
            # Pass through arrow settings
            'show_quiver': self.default_settings.get('show_quiver', False),
            'show_streamlines': self.default_settings.get('show_streamlines', False),
            'arrow_spacing': self.default_settings.get('arrow_spacing', 100),
            'arrow_scale': self.default_settings.get('arrow_scale', 100),
            'arrow_color': self.default_settings.get('arrow_color', 'white'),
            'arrow_width': self.default_settings.get('arrow_width', 1.0),
            # Pass through log scale setting
            'use_log_scale': self.default_settings.get('use_log_scale', False),
        }
    
    def _on_export_clicked(self):
        """Handle Export button click."""
        # Validate
        error = self._validate_inputs()
        if error:
            from tkinter import messagebox
            messagebox.showerror("Validation Error", error)
            return
        
        # Get settings
        settings = self._get_export_settings()
        _log(f"Export requested: {settings}")
        
        # Close dialog
        self.destroy()
        
        # Trigger callback
        if self.on_export:
            self.on_export(settings)
