import { useMemo } from 'react';
import { CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyState } from '@/components/common/EmptyState';
import type { CycleSnapshot } from '@/types/domain';

const SENSOR_CHANNELS = [
  { key: 'vibration', color: 'hsl(221 83% 53%)', label: 'Vibration' },
  { key: 'temperature', color: 'hsl(0 84% 60%)', label: 'Temperature' },
  { key: 'pressure', color: 'hsl(142 71% 45%)', label: 'Pressure' },
  { key: 'acoustic_emission', color: 'hsl(272 71% 55%)', label: 'Acoustic emission' },
];

/**
 * Show up to 4 sensor channels over time (using operational_cycles or
 * timestamp order). Highlights anomaly indices on the top channel.
 * Sensor payload lives on `state.sensor_batch` on the server but is
 * NOT included in the /cycles snapshot — we read from state.trace's
 * `anomaly_result` for anomaly hints, and rely on the raw stress_features
 * `extras.timeseries` when available.
 */
export function AnomalyChartsPanel({ cycle }: { cycle: CycleSnapshot }) {
  const anomaly = cycle.anomaly_result as Record<string, any> | null;
  const extras = (cycle.stress_features?.extras ?? {}) as Record<string, any>;
  const timeseries = (extras.timeseries ?? extras.channels) as Record<string, number[]> | undefined;
  const topChannel = anomaly?.top_channel as string | undefined;
  const anomalyIndices = (anomaly?.anomaly_indices ?? anomaly?.indices ?? []) as number[];

  const chartRows = useMemo(() => {
    if (!timeseries) return null;
    const nRows = Math.max(...Object.values(timeseries).map((v) => v?.length ?? 0), 0);
    if (nRows === 0) return null;
    const rows: Record<string, number>[] = [];
    for (let i = 0; i < nRows; i++) {
      const row: Record<string, number> = { idx: i };
      for (const ch of SENSOR_CHANNELS) {
        const arr = timeseries[ch.key];
        if (arr && typeof arr[i] === 'number') row[ch.key] = arr[i];
      }
      rows.push(row);
    }
    return rows;
  }, [timeseries]);

  if (!anomaly) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Sensor channels</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {!chartRows ? (
          <EmptyState
            title="No time-series available"
            description="This snapshot doesn't include per-sample sensor traces. Anomaly summary is shown above."
          />
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {SENSOR_CHANNELS.filter((ch) => chartRows.some((r) => r[ch.key] !== undefined)).map((ch) => (
              <div key={ch.key} className="rounded-md border bg-card p-3">
                <div className="mb-1 flex items-center justify-between">
                  <div className="text-xs font-semibold uppercase text-muted-foreground">{ch.label}</div>
                  {topChannel === ch.key ? (
                    <span className="rounded-full bg-destructive/10 px-2 py-0.5 text-[10px] font-semibold uppercase text-destructive">
                      top anomaly
                    </span>
                  ) : null}
                </div>
                <div className="h-40 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartRows} margin={{ top: 6, right: 6, left: -18, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
                      <XAxis dataKey="idx" fontSize={10} />
                      <YAxis fontSize={10} />
                      <Tooltip contentStyle={{ fontSize: 12 }} />
                      <Line
                        type="monotone"
                        dataKey={ch.key}
                        stroke={ch.color}
                        strokeWidth={1.4}
                        dot={false}
                        isAnimationActive={false}
                      />
                      {topChannel === ch.key
                        ? anomalyIndices
                            .filter((i) => i < chartRows.length)
                            .map((i) => (
                              <ReferenceDot
                                key={i}
                                x={i}
                                y={chartRows[i]?.[ch.key]}
                                r={3}
                                fill="hsl(0 84% 60%)"
                                stroke="hsl(0 0% 100%)"
                                strokeWidth={1}
                              />
                            ))
                        : null}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
