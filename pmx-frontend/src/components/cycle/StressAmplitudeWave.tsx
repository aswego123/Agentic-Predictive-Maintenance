import { useMemo } from 'react';
import Plot from 'react-plotly.js';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { MetricCard } from '@/components/common/MetricCard';
import { fmtNumber } from '@/lib/format';

interface PhysicsPrediction {
  stress_amplitude_mpa?: number;
  mean_stress_mpa?: number;
  cycles_per_hour?: number;
  endurance_limit_mpa?: number;
  yield_strength_mpa?: number;
  stress_ratio_vs_endurance?: number;
  [k: string]: any;
}

/**
 * 3D stress wave surface. Renders the cyclic loading as a traveling
 * wave through the component:
 *
 *   σ(x, t) = μ + a · sin(2π (f·t − x/λ))
 *
 * X-axis = position along the component (0 → 1, normalised)
 * Y-axis = time (seconds)
 * Z-axis = stress (MPa)
 *
 * This is the natural 3D counterpart to the Basquin / S–N amplitude
 * that the physics engine reports — it shows *both* how stress
 * oscillates over time AND how a stress wave propagates spatially,
 * which is what causes the diamond-shaped nodal patterns you see in
 * FEA. Endurance limit + yield strength are drawn as translucent
 * horizontal planes so an engineer can see at a glance if the wave
 * ever pokes through them.
 */
export function StressAmplitudeWave({
  physics,
}: {
  physics: PhysicsPrediction | null | undefined;
}) {
  const data = useMemo(() => {
    if (!physics) return null;
    const amp = Number(physics.stress_amplitude_mpa ?? NaN);
    if (!Number.isFinite(amp) || amp <= 0) return null;
    const mean = Number(physics.mean_stress_mpa ?? 0) || 0;
    const cyclesPerHour = Number(physics.cycles_per_hour ?? 3600) || 3600;
    const freqHz = cyclesPerHour / 3600;
    const period = freqHz > 0 ? 1 / freqHz : 1;

    // Build a coarse grid — 40×40 keeps Plotly snappy while still
    // giving a smooth-looking wave surface.
    const nX = 40; // spatial samples
    const nT = 40; // time samples
    const nCycles = 3;
    const wavelength = 0.5; // normalised (2 full wavelengths across the component)
    const duration = nCycles * period;

    const x: number[] = [];
    const yTime: number[] = [];
    for (let i = 0; i < nX; i++) x.push(i / (nX - 1));
    for (let j = 0; j < nT; j++) yTime.push((j / (nT - 1)) * duration);

    // z is a 2D matrix [nT][nX] as required by Plotly surface.
    const z: number[][] = [];
    for (let j = 0; j < nT; j++) {
      const row: number[] = [];
      for (let i = 0; i < nX; i++) {
        const val =
          mean + amp * Math.sin(2 * Math.PI * (freqHz * yTime[j] - x[i] / wavelength));
        row.push(val);
      }
      z.push(row);
    }

    return { x, yTime, z, amp, mean, freqHz, period, duration, nX, nT };
  }, [physics]);

  if (!data) return null;

  const endurance = Number(physics?.endurance_limit_mpa ?? NaN);
  const yieldMpa = Number(physics?.yield_strength_mpa ?? NaN);
  const stressRatio = Number(physics?.stress_ratio_vs_endurance ?? NaN);

  // Colour scale keyed to the physical stress range so peaks glow red.
  const zMin = data.mean - data.amp;
  const zMax = data.mean + data.amp;

  const traces: any[] = [
    {
      type: 'surface',
      x: data.x,
      y: data.yTime,
      z: data.z,
      colorscale: 'Turbo',
      cmin: zMin,
      cmax: zMax,
      showscale: true,
      colorbar: { title: { text: 'σ (MPa)' }, thickness: 14 },
      contours: {
        z: {
          show: true,
          usecolormap: true,
          highlightcolor: '#ffffff',
          project: { z: true },
          width: 2,
        },
      },
      lighting: { ambient: 0.65, diffuse: 0.8, specular: 0.15, roughness: 0.9 },
      opacity: 0.95,
      hovertemplate:
        'position=%{x:.2f}<br>t=%{y:.3f} s<br>σ=%{z:.1f} MPa<extra></extra>',
      name: 'σ(x,t)',
    },
  ];

  // Reference planes for endurance limit + yield strength.
  const cornerX = [0, 1, 1, 0];
  const cornerY = [0, 0, data.duration, data.duration];
  if (Number.isFinite(endurance) && endurance > 0 && endurance <= zMax * 1.05) {
    traces.push({
      type: 'mesh3d',
      x: cornerX,
      y: cornerY,
      z: cornerX.map(() => endurance),
      i: [0, 0],
      j: [1, 2],
      k: [2, 3],
      color: 'rgb(245, 158, 11)',
      opacity: 0.18,
      hoverinfo: 'skip',
      showlegend: true,
      name: `endurance ${endurance.toFixed(0)} MPa`,
    });
  }
  if (Number.isFinite(yieldMpa) && yieldMpa > 0 && yieldMpa <= zMax * 1.2) {
    traces.push({
      type: 'mesh3d',
      x: cornerX,
      y: cornerY,
      z: cornerX.map(() => yieldMpa),
      i: [0, 0],
      j: [1, 2],
      k: [2, 3],
      color: 'rgb(220, 38, 38)',
      opacity: 0.15,
      hoverinfo: 'skip',
      showlegend: true,
      name: `yield ${yieldMpa.toFixed(0)} MPa`,
    });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Stress Amplitude</CardTitle>
        <p className="text-xs text-muted-foreground">
          3D cyclic-stress wave σ(x, t) = μ + a·sin(2π(f·t − x/λ)) travelling
          along the component. Amber plane = endurance limit (fatigue threshold);
          red plane = yield strength. Peaks glowing above the amber plane indicate
          finite-life fatigue accumulation.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <MetricCard
            label="Amplitude (a)"
            value={`${fmtNumber(data.amp, 1)} MPa`}
            tone="info"
          />
          <MetricCard label="Mean (μ)" value={`${fmtNumber(data.mean, 1)} MPa`} />
          <MetricCard
            label="Frequency"
            value={`${fmtNumber(data.freqHz, 3)} Hz`}
            hint={`period ${fmtNumber(data.period * 1000, 1)} ms`}
          />
          <MetricCard
            label="σ / endurance"
            value={Number.isFinite(stressRatio) ? stressRatio.toFixed(3) : '—'}
            tone={
              Number.isFinite(stressRatio)
                ? stressRatio < 0.8
                  ? 'success'
                  : stressRatio < 1.0
                    ? 'warning'
                    : 'danger'
                : 'default'
            }
            hint={
              Number.isFinite(endurance) ? `limit ${endurance.toFixed(0)} MPa` : undefined
            }
          />
        </div>

        <div className="rounded-md border bg-background">
          <Plot
            data={traces}
            layout={{
              title: {
                text: `Stress Amplitude — ${data.amp.toFixed(1)} MPa @ ${data.freqHz.toFixed(2)} Hz`,
                x: 0.02,
                font: { size: 13 },
              },
              margin: { l: 0, r: 0, t: 44, b: 0 },
              height: 520,
              scene: {
                xaxis: {
                  title: { text: 'Position (normalised)' },
                  gridcolor: 'rgba(148,163,184,0.25)',
                },
                yaxis: {
                  title: { text: 'Time (s)' },
                  gridcolor: 'rgba(148,163,184,0.25)',
                },
                zaxis: {
                  title: { text: 'Stress σ (MPa)' },
                  gridcolor: 'rgba(148,163,184,0.25)',
                },
                aspectmode: 'cube',
                camera: { eye: { x: 1.6, y: 1.6, z: 0.9 } },
              },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'right',
                x: 1,
                font: { size: 10 },
              },
              autosize: true,
            }}
            style={{ width: '100%', height: 520 }}
            config={{ displaylogo: false, responsive: true }}
            useResizeHandler
          />
        </div>

        {/* 2D time-series slice — a densely-sampled σ(t) view at
            position x=0. This gives an at-a-glance reading of the
            sinusoid shape, its peaks, and how it sits relative to
            the endurance / yield reference lines. */}
        <div className="rounded-md border bg-background">
          <Plot
            data={build2DTrace(data, endurance, yieldMpa)}
            layout={{
              title: {
                text: 'Stress Amplitude vs Time',
                x: 0.02,
                font: { size: 13 },
              },
              margin: { l: 60, r: 20, t: 44, b: 48 },
              height: 320,
              xaxis: {
                title: { text: 'Time (s)' },
                showgrid: true,
                gridcolor: 'rgba(148,163,184,0.15)',
                zeroline: false,
              },
              yaxis: {
                title: { text: 'Stress σ (MPa)' },
                showgrid: true,
                gridcolor: 'rgba(148,163,184,0.15)',
                zeroline: true,
                zerolinecolor: 'rgba(148,163,184,0.4)',
              },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              shapes: build2DShapes(endurance, yieldMpa),
              annotations: build2DAnnotations(endurance, yieldMpa),
              legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'right',
                x: 1,
                font: { size: 10 },
              },
              hovermode: 'x unified',
              autosize: true,
            }}
            style={{ width: '100%', height: 320 }}
            config={{ displaylogo: false, responsive: true, displayModeBar: false }}
            useResizeHandler
          />
        </div>
      </CardContent>
    </Card>
  );
}

// ─── 2D helpers ──────────────────────────────────────────────────
// These render a time-slice σ(t) at position x=0 (equivalent to
// sampling the surface's front edge), plus the endurance / yield
// reference lines. Split out so the 3D block above stays readable.

interface WaveData {
  amp: number;
  mean: number;
  freqHz: number;
  duration: number;
}

function build2DTrace(data: WaveData, endurance: number, yieldMpa: number) {
  const nPoints = 240; // dense enough for a smooth sinusoid
  const t: number[] = [];
  const y: number[] = [];
  const peaks: number[] = [];
  const troughs: number[] = [];
  for (let i = 0; i < nPoints; i++) {
    const ti = (i / (nPoints - 1)) * data.duration;
    t.push(ti);
    y.push(data.mean + data.amp * Math.sin(2 * Math.PI * data.freqHz * ti));
    peaks.push(data.mean + data.amp);
    troughs.push(data.mean - data.amp);
  }
  void endurance;
  void yieldMpa;
  return [
    // Amplitude envelope band (peak↔trough) — visual "corridor".
    {
      x: [...t, ...[...t].reverse()],
      y: [...peaks, ...[...troughs].reverse()],
      type: 'scatter',
      mode: 'lines',
      fill: 'toself',
      fillcolor: 'rgba(59,130,246,0.12)',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
    },
    // Mean-stress centreline.
    {
      x: t,
      y: t.map(() => data.mean),
      type: 'scatter',
      mode: 'lines',
      line: { color: 'rgba(100,116,139,0.7)', width: 1, dash: 'dot' },
      name: `mean σ (${data.mean.toFixed(1)} MPa)`,
      hoverinfo: 'skip',
    },
    // Main sinusoidal stress curve.
    {
      x: t,
      y,
      type: 'scatter',
      mode: 'lines',
      line: { color: 'rgb(37,99,235)', width: 2.5, shape: 'spline', smoothing: 1.1 },
      name: 'σ(t)',
      hovertemplate: 't=%{x:.4f} s<br>σ=%{y:.2f} MPa<extra></extra>',
    },
  ] as any[];
}

function build2DShapes(endurance: number, yieldMpa: number): any[] {
  const shapes: any[] = [];
  if (Number.isFinite(endurance) && endurance > 0) {
    shapes.push({
      type: 'line',
      xref: 'paper',
      x0: 0,
      x1: 1,
      y0: endurance,
      y1: endurance,
      line: { color: 'rgba(245,158,11,0.85)', width: 1.5, dash: 'dash' },
    });
  }
  if (Number.isFinite(yieldMpa) && yieldMpa > 0) {
    shapes.push({
      type: 'line',
      xref: 'paper',
      x0: 0,
      x1: 1,
      y0: yieldMpa,
      y1: yieldMpa,
      line: { color: 'rgba(220,38,38,0.6)', width: 1, dash: 'dot' },
    });
  }
  return shapes;
}

function build2DAnnotations(endurance: number, yieldMpa: number): any[] {
  const annotations: any[] = [];
  if (Number.isFinite(endurance) && endurance > 0) {
    annotations.push({
      xref: 'paper',
      x: 0.995,
      y: endurance,
      xanchor: 'right',
      yanchor: 'bottom',
      text: `endurance • ${endurance.toFixed(0)} MPa`,
      showarrow: false,
      font: { size: 10, color: 'rgb(180,83,9)' },
    });
  }
  if (Number.isFinite(yieldMpa) && yieldMpa > 0) {
    annotations.push({
      xref: 'paper',
      x: 0.995,
      y: yieldMpa,
      xanchor: 'right',
      yanchor: 'bottom',
      text: `yield • ${yieldMpa.toFixed(0)} MPa`,
      showarrow: false,
      font: { size: 10, color: 'rgb(153,27,27)' },
    });
  }
  return annotations;
}
