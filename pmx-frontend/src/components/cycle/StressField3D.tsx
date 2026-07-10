import { useMemo } from 'react';
import Plot from 'react-plotly.js';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { MetricCard } from '@/components/common/MetricCard';
import { fmtNumber } from '@/lib/format';

interface StressPoint {
  x: number;
  y: number;
  z: number;
  stress_mpa: number;
  is_hotspot?: boolean;
}

interface StressField {
  geometry?: string;
  points?: StressPoint[];
  hotspot_region?: string;
  hotspot_count?: number;
  hotspot_threshold_mpa?: number;
  max_stress_mpa?: number;
  note?: string;
}

const CAMERA_BY_GEOMETRY: Record<string, { eye: { x: number; y: number; z: number } }> = {
  beam:     { eye: { x: 1.6, y: 1.6, z: 0.6 } },
  disc:     { eye: { x: 0.2, y: 0.2, z: 2.2 } },
  cylinder: { eye: { x: 1.8, y: 1.8, z: 1.4 } },
  strut:    { eye: { x: 2.0, y: 2.0, z: 0.9 } },
  shell:    { eye: { x: 1.8, y: 1.8, z: 1.2 } },
  box:      { eye: { x: 1.6, y: 1.6, z: 1.2 } },
};

export function StressField3D({ field }: { field: StressField | null | undefined }) {
  const trace = useMemo(() => {
    if (!field?.points?.length) return null;
    const pts = field.points;
    const maxStress = Math.max(...pts.map((p) => p.stress_mpa), 1e-6);
    return {
      type: 'scatter3d' as const,
      mode: 'markers' as const,
      x: pts.map((p) => p.x),
      y: pts.map((p) => p.y),
      z: pts.map((p) => p.z),
      marker: {
        size: pts.map((p) => 4 + 8 * (p.stress_mpa / maxStress)),
        color: pts.map((p) => p.stress_mpa),
        colorscale: 'Turbo' as const,
        colorbar: { title: { text: 'Stress (MPa)' } },
        opacity: 0.85,
        symbol: pts.map((p) => (p.is_hotspot ? 'diamond' : 'circle')) as any,
        line: { width: 0 },
      },
      text: pts.map(
        (p) => `stress=${p.stress_mpa.toFixed(1)} MPa${p.is_hotspot ? ' ⚠ HOTSPOT' : ''}`,
      ),
      hovertemplate: '%{text}<extra></extra>',
    };
  }, [field]);

  if (!field?.points?.length || !trace) return null;

  const geom = field.geometry ?? '-';
  const camera = CAMERA_BY_GEOMETRY[geom] ?? CAMERA_BY_GEOMETRY.box;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          3D stress intensity — geometry: <code className="rounded bg-muted px-1">{geom}</code> (synthetic)
        </CardTitle>
        <p className="text-xs text-muted-foreground">
          Geometry archetype is chosen from the selected asset type
          (aircraft_engine / traction_motor → cylinder, wing → beam, brake / wheel → disc,
          landing_gear → strut, fuselage → shell, bogie → box).
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <MetricCard label="Geometry" value={geom} />
          <MetricCard label="Hotspot region" value={field.hotspot_region ?? '—'} />
          <MetricCard label="Max stress (MPa)" value={fmtNumber(field.max_stress_mpa, 1)} />
          <MetricCard label="Hotspot nodes" value={field.hotspot_count ?? 0} />
        </div>

        <div className="rounded-md border">
          <Plot
            data={[trace as any]}
            layout={{
              title: { text: `Geometry archetype: ${geom} — ${field.points.length} nodes`, x: 0.02 },
              margin: { l: 0, r: 0, t: 40, b: 0 },
              scene: {
                xaxis: { title: { text: 'X' } },
                yaxis: { title: { text: 'Y' } },
                zaxis: { title: { text: 'Z' } },
                aspectmode: 'data',
                camera,
              },
              height: 520,
              autosize: true,
            }}
            style={{ width: '100%', height: 520 }}
            config={{ displaylogo: false, responsive: true }}
            useResizeHandler
          />
        </div>

        <p className="text-[11px] text-muted-foreground">
          ⚠ Synthetic stress field — {field.note ?? ''} Diamond markers are nodes above the hotspot threshold (
          {fmtNumber(field.hotspot_threshold_mpa, 0)} MPa).
        </p>
      </CardContent>
    </Card>
  );
}
