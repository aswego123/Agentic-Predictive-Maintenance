import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { AlertTriangle, Waves, Sparkles } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { EmptyState } from '@/components/common/EmptyState';
import { fmtHours, fmtNumber, fmtPercent } from '@/lib/format';

interface Props {
  physics: Record<string, any> | null;
  ml: Record<string, any> | null;
}

/**
 * Side-by-side physics + ML cards with:
 *   - Big RUL numbers with colored status accents
 *   - Confidence-interval error bar on the ML side
 *   - Unified bar chart comparing stress + RUL with error bars
 *   - Delta callouts between the two predictions
 */
export function PhysicsMLPanel({ physics, ml }: Props) {
  if (!physics && !ml) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Physics vs ML</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            title="Not run"
            description="Physics + ML did not fire (non-anomalous short-circuit)."
          />
        </CardContent>
      </Card>
    );
  }

  const physRul = Number(physics?.rul_hours ?? NaN);
  const mlRul = Number(ml?.rul_hours ?? NaN);
  const mlLo = Number(ml?.confidence_lower_hours ?? NaN);
  const mlHi = Number(ml?.confidence_upper_hours ?? NaN);
  const mlErrorLower = !Number.isNaN(mlLo) && !Number.isNaN(mlRul) ? mlRul - mlLo : 0;
  const mlErrorUpper = !Number.isNaN(mlHi) && !Number.isNaN(mlRul) ? mlHi - mlRul : 0;

  const physStress = Number(physics?.stress_amplitude_mpa ?? 0);
  const mlStress = Number(ml?.predicted_stress_mpa ?? 0);

  const rulDelta = !Number.isNaN(physRul) && !Number.isNaN(mlRul) ? mlRul - physRul : null;
  const rulDeltaPct = rulDelta !== null && physRul > 0 ? rulDelta / physRul : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          Physics vs ML
          {rulDelta !== null ? (
            <Badge
              variant={Math.abs(rulDeltaPct ?? 0) > 0.15 ? 'warning' : 'success'}
              className="ml-2"
            >
              Δ RUL {rulDelta > 0 ? '+' : ''}{rulDelta.toFixed(1)} h
            </Badge>
          ) : null}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <PredictionCard
            icon={<Waves className="h-4 w-4" />}
            label="Physics"
            rul={physRul}
            stress={physStress}
            method={String(physics?.failure_mode ?? 'Basquin + Paris + NASGRO')}
            healthStatus={physics?.health_status}
            healthScore={physics?.health_score}
            colorClass="border-blue-500/30 bg-blue-500/5"
          />
          <PredictionCard
            icon={<Sparkles className="h-4 w-4" />}
            label="ML correction"
            rul={mlRul}
            stress={mlStress}
            method={String(ml?.method ?? 'ML')}
            ciLow={mlLo}
            ciHigh={mlHi}
            colorClass="border-orange-500/30 bg-orange-500/5"
          />
        </div>

        {physics && ml ? (
          <div className="rounded-md border bg-card p-3">
            <div className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
              Side-by-side comparison
              {mlErrorLower !== 0 || mlErrorUpper !== 0 ? (
                <span className="ml-2 font-normal normal-case text-muted-foreground/80">
                  · ML RUL shown with confidence interval [{fmtHours(mlLo)}, {fmtHours(mlHi)}]
                </span>
              ) : null}
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={[
                    {
                      metric: 'Stress (MPa)',
                      Physics: physStress,
                      ML: mlStress,
                      MLerr: [0, 0],
                    },
                    {
                      metric: 'RUL (h)',
                      Physics: physRul,
                      ML: mlRul,
                      MLerr: [mlErrorLower, mlErrorUpper],
                    },
                  ]}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="metric" fontSize={12} />
                  <YAxis fontSize={12} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Physics" fill="hsl(221 83% 53%)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="ML" fill="hsl(24 95% 53%)" radius={[4, 4, 0, 0]}>
                    <ErrorBar dataKey="MLerr" width={4} strokeWidth={1.5} stroke="hsl(24 95% 30%)" direction="y" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : null}

        {rulDeltaPct !== null && Math.abs(rulDeltaPct) > 0.15 ? (
          <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-xs text-amber-900 dark:text-amber-200">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <strong>Significant divergence:</strong> ML predicts RUL{' '}
              <strong>{fmtPercent(Math.abs(rulDeltaPct), 0)}</strong>{' '}
              {rulDeltaPct > 0 ? 'higher' : 'lower'} than physics. The Judge is
              likely to escalate.
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function PredictionCard({
  icon,
  label,
  rul,
  stress,
  method,
  colorClass,
  healthStatus,
  healthScore,
  ciLow,
  ciHigh,
}: {
  icon: React.ReactNode;
  label: string;
  rul: number;
  stress: number;
  method: string;
  colorClass: string;
  healthStatus?: string | null;
  healthScore?: number | null;
  ciLow?: number;
  ciHigh?: number;
}) {
  const hasCI = ciLow !== undefined && ciHigh !== undefined && !Number.isNaN(ciLow) && !Number.isNaN(ciHigh);
  return (
    <div className={`rounded-lg border p-4 ${colorClass}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold">
          {icon}
          {label}
        </div>
        {healthStatus ? (
          <Badge variant="outline" className="text-[10px] uppercase">{healthStatus}</Badge>
        ) : null}
      </div>
      <div className="mt-3 flex items-baseline gap-2">
        <span className="text-3xl font-semibold tabular-nums tracking-tight">
          {Number.isNaN(rul) ? '—' : rul.toFixed(1)}
        </span>
        <span className="text-xs text-muted-foreground">hours RUL</span>
      </div>
      {hasCI ? (
        <div className="mt-1 text-[11px] text-muted-foreground">
          95% CI: [{ciLow!.toFixed(1)}, {ciHigh!.toFixed(1)}] h
        </div>
      ) : null}
      <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
        <div>
          <div className="text-muted-foreground">Stress</div>
          <div className="font-medium tabular-nums">{fmtNumber(stress, 1)} MPa</div>
        </div>
        <div>
          <div className="text-muted-foreground">{healthScore !== undefined && healthScore !== null ? 'Health score' : 'Method'}</div>
          <div className="truncate font-medium">
            {healthScore !== undefined && healthScore !== null ? fmtNumber(healthScore, 1) : method}
          </div>
        </div>
      </div>
    </div>
  );
}
