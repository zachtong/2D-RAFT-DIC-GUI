# Report Enhancement Design

**Date:** 2026-03-13
**Status:** Approved

## Overview

Comprehensive overhaul of the RAFT-DIC report generator to transform it from a basic session dump into a professional, customizable analysis report suitable for academic papers, engineering documentation, and presentations.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PDF export | weasyprint (HTML→PDF) | Reuses existing HTML template, consistent styling, supports @media print |
| User custom content | Lightweight: title + author + notes | Covers 90% of needs without editor complexity |
| Line probe display | Summary stats + kymograph | Kymograph captures spatiotemporal evolution in one image |
| Area probe display | Summary stats + sampled 2D distribution | Equal-interval sampling (max 6 frames), predictable |
| Key frames components | Fixed 3 frames × user-selectable components | First/peak/last is meaningful for DIC; peak per-component |
| Statistics trends | Table + trend line charts (C) | Table for reference, chart for temporal insight |
| Theme system | 4 preset themes (Light/Dark/Academic/Minimal) | Simple dropdown, no complex editor |
| Area frame sampling | Equal-interval (max 6) | Predictable, doesn't miss slow changes |

## Architecture

### Data Flow

```
Frontend (ExportDialog)
  ├── custom_title, author, notes
  ├── sections[] (9 sections now)
  ├── key_frame_components[] (U, V, magnitude, strain...)
  ├── theme: "light" | "dark" | "academic" | "minimal"
  ├── format: "html" | "pdf" | "both"
  └── vis_settings: { colormap, vmin, vmax, physical_ratio, physical_unit, physical_enabled }
        │
        ▼
POST /export/report
        │
        ▼
report_generator.py
  ├── _render_header_section()      — custom title/author/notes
  ├── _render_experiment_section()   — unchanged
  ├── _render_parameters_section()   — unchanged
  ├── _render_roi_section()          — unchanged
  ├── _render_key_frames_section()   — multi-component, GUI colormap/vmin/vmax
  ├── _render_statistics_section()   — full stats + trend charts
  ├── _render_probes_section()       — point probes with physical units
  ├── _render_line_probes_section()  — NEW: summary + kymograph
  └── _render_area_probes_section()  — NEW: summary + sampled 2D
        │
        ▼
Jinja2 template + theme CSS injection
        │
        ├──► .html file
        └──► .pdf file (via weasyprint)
```

### Template System

Single `report_template.html` with theme CSS injected via Jinja2 variable:

```html
<style>
  /* Base styles (shared) */
  ...
  /* Theme overrides */
  {{ theme_css }}
</style>
```

Theme CSS stored as Python string constants in `report_generator.py`:

```python
THEMES = {
    "light": "/* white bg, blue-gray headings */",
    "dark": "/* dark bg #1a1a2e, light text */",
    "academic": "/* serif fonts, black/white, numbered sections */",
    "minimal": "/* borderless tables, large whitespace */",
}
```

## Section Details

### Header (enhanced)

- `custom_title`: User-provided title (default: "RAFT-DIC Analysis Report")
- `author`: Author / institution name (optional)
- `notes`: Multi-line notes rendered as gray italic paragraph (optional)
- `date`, `software`, `model`: unchanged

### Key Frames (enhanced)

- User selects which components to display (U, V, magnitude, strain components)
- Each component gets its own row: 3 columns (first / peak / last frame)
- Peak frame calculated per-component (e.g., peak εxx frame ≠ peak magnitude frame)
- Charts use GUI's current colormap + vmin/vmax (passed via `vis_settings`)
- Physical units shown in colorbar labels when enabled

### Statistics (enhanced)

- **Table**: Same as current but:
  - Remove the 5-component strain limit — show ALL computed components
  - Add physical unit column when enabled
  - Displacement precision: 4 decimal places; strain: 6
- **Trend charts**: One chart per component
  - X-axis: frame number
  - Y-axis: statistical value
  - Lines: min (blue), max (red), mean (black)
  - Band: mean ± std (semi-transparent gray fill)
  - Physical units on Y-axis label when enabled

### Point Probes (enhanced)

- Same U/V time series charts
- Y-axis label includes physical unit when enabled (e.g., "U displacement (mm)")
- Probe color preserved from GUI

### Line Probes (NEW)

For each line probe, two visualizations:

1. **Summary time series**: Average value along the line vs. frame number
   - Separate charts for U, V components
   - Same style as point probe charts

2. **Kymograph (spatiotemporal map)**:
   - X-axis: position along line (0 to line length)
   - Y-axis: frame number
   - Color: displacement/strain value
   - Uses GUI colormap settings
   - One kymograph per component (U, V)

Implementation: Sample N points along the line (e.g., 100 points), extract values via `map_coordinates` for each frame, stack into 2D array.

### Area Probes (NEW)

For each area probe, three visualizations:

1. **Summary time series**: Area-average value vs. frame number
   - Separate charts for U, V components
   - Include min/max envelope as semi-transparent band

2. **Statistics table**: Per-component min/max/mean/std within the area (aggregated across all frames)

3. **2D distribution snapshots**: Equal-interval sampled frames (max 6)
   - Frame selection: first, last, + 4 evenly spaced in between (deduplicated)
   - Each snapshot: 2D color map of the component within the area boundary
   - Grid layout similar to Key Frames section
   - Uses GUI colormap settings

Implementation:
- Rect area: direct array slicing
- Circle area: boolean mask from center + radius
- Polygon area: `matplotlib.path.Path.contains_points()` mask

## Frontend UI Layout

```
Report Generator
├── Custom Info
│   ├── Title input (default "RAFT-DIC Analysis Report")
│   ├── Author / Institution input (optional)
│   └── Notes textarea (optional, 2-3 rows)
├── Sections (checkbox grid)
│   ├── ✓ Header
│   ├── ✓ Experiment
│   ├── ✓ Parameters
│   ├── ✓ ROI
│   ├── ✓ Key Frames
│   ├── ✓ Statistics
│   ├── ✓ Point Probes
│   ├── ✓ Line Probes      ← NEW
│   └── ✓ Area Probes      ← NEW
├── Key Frame Components (collapsible, shown when key_frames checked)
│   └── Multi-select: U, V, Magnitude, [strain components...]
├── Output Settings
│   ├── Theme dropdown: Light / Dark / Academic / Minimal
│   ├── Format dropdown: HTML / PDF / Both
│   └── Output path + browse button
└── Generate Report button
```

## Theme Specifications

| Theme | Background | Text | Headings | Table | Font |
|-------|-----------|------|----------|-------|------|
| **Light** (default) | #fff | #1a1a2e | #0f3460, blue underline | Striped #f4f6f9 | System sans-serif |
| **Dark** | #1a1a2e | #e0e0e0 | #7ec8e3, subtle underline | Striped #252545 | System sans-serif |
| **Academic** | #fff | #000 | Black, numbered (1. 2. 3.) | Bordered, no stripe | Georgia / Times New Roman |
| **Minimal** | #fff | #333 | #333, no underline | Borderless, spacing only | Inter / system sans-serif |

## Physical Units Integration

When `physical_enabled=True`:
- All displacement values multiplied by `1 / physical_ratio`
- Axis labels: "U displacement (mm)" instead of "U displacement"
- Statistics table: values in physical units, unit column shown
- Colorbar labels include unit
- Strain values unaffected (dimensionless)

## Dependencies

New dependency: `weasyprint` added to `setup.py`

```python
install_requires=[
    ...,
    "weasyprint>=60.0",
]
```

Note: weasyprint requires system GTK libraries on some platforms. Add installation note to README.

## Implementation Phases

### Phase 1: Backend Core (report_generator.py)
1. Add theme CSS constants (4 themes)
2. Refactor `generate_report()` to accept new parameters
3. Enhance `_render_header_section()` with custom title/author/notes
4. Enhance `_render_key_frames_section()` with multi-component + GUI vis settings
5. Enhance `_render_statistics_section()` — remove limit, add trend charts, physical units
6. Enhance `_render_probes_section()` — physical units on point probes
7. Implement `_render_line_probes_section()` — summary + kymograph
8. Implement `_render_area_probes_section()` — summary + table + 2D snapshots
9. Add PDF output via weasyprint
10. Update Jinja2 template with new sections and theme injection

### Phase 2: API Layer (server/routes/export.py)
11. Extend `POST /export/report` to accept new parameters
12. Pass vis_settings (colormap, vmin/vmax, physical units) from session
13. Handle format parameter (html/pdf/both)
14. Validate new inputs

### Phase 3: Frontend (ExportDialog.tsx)
15. Add custom info inputs (title, author, notes)
16. Split probe sections into Point/Line/Area checkboxes
17. Add key frame component multi-select (collapsible)
18. Add theme dropdown
19. Add format dropdown (HTML / PDF / Both)
20. Wire all new parameters to API call

### Phase 4: Polish & Testing
21. Test all 4 themes with real data
22. Test PDF output quality and page breaks
23. Test with edge cases (no probes, no strain, single frame)
24. Verify physical unit display consistency
