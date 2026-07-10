import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyState } from '@/components/common/EmptyState';

/**
 * Convergence chart: RUL + stress across negotiation rounds for
 * physics vs ML. Only rendered when we have >=2 rounds of history.
 */
export function NegotiationHistoryPanel({
  history,
}: {
  history: Array<Record<string, any>>;
}) {
  if (!history || history.length < 2) return null;

  // Two chart series: RUL and stress, both by round with physics + ml lines.
  const rulData = history.map((h) => ({
    round: Number(h.round ?? h.negotiation_round ?? 0),
    Physics: Number(h.physics_rul_hours ?? h.physics?.rul_hours ?? NaN),
    ML: Number(h.ml_rul_hours ?? h.ml?.rul_hours ?? NaN),
  }));
  const stressData = history.map((h) => ({
    round: Number(h.round ?? h.negotiation_round ?? 0),
    Physics: Number(h.physics_stress_mpa ?? h.physics?.stress_amplitude_mpa ?? NaN),
    ML: Number(h.ml_stress_mpa ?? h.ml?.predicted_stress_mpa ?? NaN),
  }));

  const hasRul = rulData.some((r) => !Number.isNaN(r.Physics) || !Number.isNaN(r.ML));
  const hasStress = stressData.some((r) => !Number.isNaN(r.Physics) || !Number.isNaN(r.ML));

  if (!hasRul && !hasStress) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Negotiation convergence</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {hasRul ? (
          <MiniChart title="RUL (hours) per round" data={rulData} />
        ) : (
          <EmptyState title="No RUL history" />
        )}
        {hasStress ? (
          <MiniChart title="Stress (MPa) per round" data={stressData} />
        ) : (
          <EmptyState title="No stress history" />
        )}
      </CardContent>
    </Card>
  );
}

function MiniChart({ title, data }: { title: string; data: Array<{ round: number; Physics: number; ML: number }> }) {
  return (
    <div className="rounded-md border bg-card p-3">
      <div className="mb-1 text-xs font-semibold uppercase text-muted-foreground">{title}</div>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
            <XAxis dataKey="round" fontSize={10} />
            <YAxis fontSize={10} />
            <Tooltip contentStyle={{ fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="Physics" stroke="hsl(221 83% 53%)" strokeWidth={2} dot />
            <Line type="monotone" dataKey="ML" stroke="hsl(24 95% 53%)" strokeWidth={2} dot />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
