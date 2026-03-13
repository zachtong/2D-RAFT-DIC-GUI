# Unified Colorbar Settings Design

## Goal

Unify colorbar settings so they control both live display and export output.
Main panel stays minimal; advanced options behind a gear icon popup.

## Architecture

```
appStore.colorbarSettings
  ├─→ ColorbarOverlay.tsx (live CSS) — mapped to CSS properties
  ├─→ ExportDialog → exportImages() → settings.colorbar_settings
  └─→ ExportDialog → exportAnimation() → settings.colorbar_settings
```

## Store

```typescript
interface ColorbarSettings {
  labelText: string;        // "" = auto from component
  labelFontSize: number;    // default 12
  tickCount: number;        // 0 = auto
  tickFontSize: number;     // default 10
  shrink: number;           // 0.3-1.0, default 0.8 (export only)
  hideOutline: boolean;     // default false
  discreteLevels: number;   // 0 = continuous, >0 = discrete bins
}
```

## UI Layout

### Main Panel (VisualizationSettings)
Existing controls + new gear button:
```
[⚙ Colorbar Settings...]
```

### Popup Panel (ColorbarSettingsPanel)
Floating popover anchored to gear button, closes on outside click.
~260px wide, single column, no scroll needed.

Sections: Label (text + font size), Ticks (count + font size),
Style (shrink + no outline + discrete levels), Reset Defaults button.

## ColorbarOverlay Mapping

| Setting | CSS Mapping | Export Mapping |
|---|---|---|
| labelText | DOM span at top | cbar.set_label() |
| labelFontSize | CSS font-size | fontsize param |
| tickCount | linearTicks(vmin,vmax,count) | MaxNLocator(nbins=) |
| tickFontSize | CSS font-size (ticks) | tick_params(labelsize=) |
| hideOutline | border: none on gradient | outline.set_visible(False) |
| discreteLevels | CSS stepped gradient | BoundaryNorm + get_cmap(N) |
| shrink | NOT mapped (export only) | colorbar(shrink=) |

## Bug Fix: Animation Color Range

Animation export currently ignores user's Fixed Range settings.
Fix: pass vis.fixedRange vmin/vmax into animation export settings.

## Files

| Action | File |
|---|---|
| MODIFY | `frontend/src/stores/appStore.ts` |
| NEW | `frontend/src/components/shared/ColorbarSettingsPanel.tsx` |
| MODIFY | `frontend/src/components/shared/ColorbarOverlay.tsx` |
| MODIFY | `frontend/src/components/postprocessing/ExportDialog.tsx` |
| MODIFY | `frontend/src/components/postprocessing/PostProcessingView.tsx` |
| MODIFY | `raft_dic_gui/export_images.py` |
| MODIFY | `raft_dic_gui/export_animation.py` |

## Implementation Order

1. Store + Panel + gear button
2. ColorbarOverlay enhancement
3. Export unification + animation vmin/vmax bug fix
