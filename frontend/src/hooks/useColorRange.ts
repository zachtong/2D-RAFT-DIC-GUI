import { useState, useEffect } from "react";
import client from "@/api/client";

const STRAIN_COMPONENTS = new Set([
  "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
]);

export interface ColorRange {
  vmin: number;
  vmax: number;
}

/**
 * Fetch the auto vmin/vmax for a given frame and display component.
 * Hits /displacement/range or /strain/range depending on the component.
 */
export function useColorRange(
  frameIdx: number,
  component: string,
  active: boolean
): ColorRange | null {
  const [range, setRange] = useState<ColorRange | null>(null);

  useEffect(() => {
    if (!active || frameIdx < 0) {
      setRange(null);
      return;
    }

    const isStrain = STRAIN_COMPONENTS.has(component);
    const url = isStrain
      ? `/strain/range/${frameIdx}?component=${component}`
      : `/displacement/range/${frameIdx}?component=${component}`;

    let cancelled = false;
    client
      .get<ColorRange>(url)
      .then(({ data }) => {
        if (!cancelled) setRange(data);
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
  }, [frameIdx, component, active]);

  return range;
}
