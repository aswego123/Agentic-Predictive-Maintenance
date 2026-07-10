import { CalendarClock, CalendarCheck2, AlertTriangle, ShieldAlert, ClipboardList, Gauge } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { MetricCard, toneForNumber } from '@/components/common/MetricCard';
import { EmptyState } from '@/components/common/EmptyState';
import { cn } from '@/lib/utils';
import {
  computeSchedule,
  fmtCalendarDate,
  fmtDurationHours,
  fmtRelativeDays,
  type UrgencyBand,
} from '@/lib/lifecycle';
import { fmtNumber } from '@/lib/format';
import type { CycleSnapshot } from '@/types/domain';

const URGENCY_STYLE: Record<UrgencyBand, { border: string; bg: string; icon: React.ComponentType<{ className?: string }>; accent: string; iconColor: string; badge: 'default' | 'destructive' | 'warning' | 'success' | 'info' }> = {
  immediate: {
    border: 'border-red-500/50',
    bg: 'bg-red-500/5',
    icon: ShieldAlert,
    accent: 'text-red-700 dark:text-red-300',
    iconColor: 'text-red-600 dark:text-red-400',
    badge: 'destructive',
  },
  urgent: {
    border: 'border-orange-500/50',
    bg: 'bg-orange-500/5',
    icon: AlertTriangle,
    accent: 'text-orange-800 dark:text-orange-300',
    iconColor: 'text-orange-600 dark:text-orange-400',
    badge: 'destructive',
  },
  planned: {
    border: 'border-amber-500/50',
    bg: 'bg-amber-500/5',
    icon: CalendarClock,
    accent: 'text-amber-800 dark:text-amber-300',
    iconColor: 'text-amber-600 dark:text-amber-400',
    badge: 'warning',
  },
  routine: {
    border: 'border-emerald-500/40',
    bg: 'bg-emerald-500/5',
    icon: CalendarCheck2,
    accent: 'text-emerald-700 dark:text-emerald-300',
    iconColor: 'text-emerald-600 dark:text-emerald-400',
    badge: 'success',
  },
  none: {
    border: 'border-border',
    bg: 'bg-muted/30',
    icon: Gauge,
    accent: 'text-muted-foreground',
    iconColor: 'text-muted-foreground',
    badge: 'default',
  },
};

export function LifecyclePredictionPanel({ cycle }: { cycle: CycleSnapshot }) {
  const physics = (cycle.physics_prediction ?? {}) as Record<string, any>;

  // Panel is only meaningful once the physics engine has finished for
  // *this* cycle. Two guards:
  //
  //   1. The cycle must not still be running/pending — LangGraph writes
  //      partial state to the checkpoint as each node completes, so a
  //      polling client can briefly see physics_prediction fields
  //      populated from a *previous* run of the same thread. Rendering
  //      those values here would flash stale data before the current
  //      cycle's physics_agent actually runs.
  //
  //   2. The key numeric fields (rul_hours + health_score) must be
  //      present and finite. If either is missing/NaN the panel would
  //      render "—" placeholders which look like real (but wrong) data.
  const stillRunning =
    cycle.status === 'running' ||
    cycle.status === 'pending' ||
    !cycle.status; // undefined = client just navigated, no snapshot yet
  const rulHoursRaw = Number(physics.rul_hours);
  const healthScoreRaw = Number(physics.health_score);
  const physicsReady =
    Number.isFinite(rulHoursRaw) && Number.isFinite(healthScoreRaw);

  if (stillRunning || !physicsReady) return null;

  const schedule = computeSchedule(cycle);
  const style = URGENCY_STYLE[schedule.urgency];
  const Icon = style.icon;

  const totalLifeHours = Number(physics.total_life_hours ?? NaN);
  const totalLifeYears = Number(physics.total_life_years ?? physics.rul_years ?? NaN);
  const rulHours = Number(physics.rul_hours ?? NaN);
  const rulYears = Number(physics.rul_years ?? NaN);
  const cyclesUsedPct = Number(physics.cycles_used_percent ?? NaN);
  const healthScore = Number(physics.health_score ?? NaN);
  const healthStatus = String(physics.health_status ?? '');
  const failureMode = String(physics.failure_mode ?? '');
  const simCorrelation = Number(physics.simulation_correlation ?? NaN);

  const healthTone = toneForNumber(healthScore, { danger: 20, warning: 60 });
  const usedTone = toneForNumber(cyclesUsedPct, { danger: 80, warning: 60 }, true);

  const scheduleByDate = schedule.scheduleByDate;
  const failureDate = schedule.predictedFailureDate;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <ClipboardList className="h-4 w-4 text-primary" />
          Lifecycle prediction &amp; maintenance schedule
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Big scheduling banner */}
        <div className={cn('relative overflow-hidden rounded-lg border p-4', style.border, style.bg)}>
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div className="flex items-start gap-3">
              <div className={cn('rounded-md border bg-background/80 p-2.5', style.border)}>
                <Icon className={cn('h-5 w-5', style.iconColor)} />
              </div>
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className={cn('text-xs font-semibold uppercase tracking-wider', style.accent)}>
                    Maintenance urgency
                  </span>
                  <Badge variant={style.badge}>{schedule.urgencyLabel}</Badge>
                </div>
                <div className="mt-1 text-lg font-semibold tracking-tight">
                  {schedule.windowDescription}
                </div>
                <p className="mt-1 max-w-xl text-xs text-muted-foreground">{schedule.rationale}</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 md:min-w-[300px]">
              <ScheduleStat
                label="Schedule by"
                value={fmtCalendarDate(scheduleByDate)}
                hint={fmtRelativeDays(scheduleByDate)}
              />
              <ScheduleStat
                label="Predicted failure"
                value={fmtCalendarDate(failureDate)}
                hint={fmtRelativeDays(failureDate)}
              />
            </div>
          </div>
        </div>

        {/* Life snapshot metrics */}
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <MetricCard
            label="Health score"
            value={Number.isNaN(healthScore) ? '—' : `${healthScore.toFixed(1)} / 100`}
            hint={healthStatus ? healthStatus.toUpperCase() : undefined}
            tone={healthTone}
          />
          <MetricCard
            label="Cycles used"
            value={Number.isNaN(cyclesUsedPct) ? '—' : `${cyclesUsedPct.toFixed(0)}%`}
            tone={usedTone}
          />
          <MetricCard
            label="Remaining life"
            value={fmtDurationHours(rulHours)}
            hint={!Number.isNaN(rulYears) ? `~${rulYears.toFixed(2)} years` : undefined}
          />
          <MetricCard
            label="Total design life"
            value={fmtDurationHours(totalLifeHours)}
            hint={!Number.isNaN(totalLifeYears) ? `~${totalLifeYears.toFixed(1)} years` : undefined}
          />
        </div>

        {/* Progress bar for cycles used */}
        {!Number.isNaN(cyclesUsedPct) ? (
          <div>
            <div className="mb-1 flex items-center justify-between text-xs">
              <span className="font-medium text-muted-foreground">Life consumed</span>
              <span className="tabular-nums text-muted-foreground">
                {cyclesUsedPct.toFixed(1)}%
              </span>
            </div>
            <Progress value={Math.min(100, Math.max(0, cyclesUsedPct))} className="h-2" />
          </div>
        ) : null}

        {/* Failure mode + simulation correlation */}
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {failureMode ? (
            <div className="rounded-md border p-3">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Predicted failure mode
              </div>
              <div className="mt-1 text-sm font-medium">{failureMode.replace(/_/g, ' ')}</div>
            </div>
          ) : null}
          {!Number.isNaN(simCorrelation) ? (
            <div className="rounded-md border p-3">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Simulation correlation
              </div>
              <div className="mt-1 text-sm font-medium tabular-nums">
                {fmtNumber(simCorrelation * 100, 1)}%{' '}
                <span className="text-xs text-muted-foreground">(model vs sensor fit)</span>
              </div>
            </div>
          ) : null}
        </div>

        {/* Recommendation list — verbatim from prediction-managent.py */}
        {schedule.recommendations.length > 0 ? (
          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Recommendations
            </div>
            <ul className="space-y-1.5 rounded-md border bg-muted/30 p-3">
              {schedule.recommendations.map((r, i) => (
                <li key={i} className="text-sm leading-relaxed">
                  {r}
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <EmptyState title="No recommendations" description="Health status did not trigger any actions." />
        )}
      </CardContent>
    </Card>
  );
}

function ScheduleStat({
  label,
  value,
  hint,
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
}) {
  return (
    <div className="rounded-md border bg-background/60 p-2.5 backdrop-blur">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 text-sm font-semibold">{value}</div>
      {hint ? <div className="mt-0.5 text-[11px] text-muted-foreground">{hint}</div> : null}
    </div>
  );
}
