import { AlertOctagon, CheckCircle2, Clock, Gauge, ShieldAlert, ShieldCheck } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { fmtHours, fmtNumber } from '@/lib/format';
import { computeSchedule } from '@/lib/lifecycle';
import type { CycleSnapshot } from '@/types/domain';

interface Props {
  cycle: CycleSnapshot;
}

/**
 * Big, colored hero banner that sits at the top of the cycle-detail
 * page.
 *
 * Urgency is derived from the SAME lifecycle scheduler the Lifecycle
 * Prediction panel uses (`computeSchedule` → based on physics
 * health_status). This keeps the hero, the maintenance schedule, and
 * the recommendation list all telling the same story. The Judge's
 * `route` used to drive this banner directly, which meant a healthy
 * component could still show "Immediate maintenance" whenever the LLM
 * was cautious.
 */
export function VerdictHero({ cycle }: Props) {
  const verdict = (cycle.judge_verdict ?? {}) as Record<string, any>;
  const physics = (cycle.physics_prediction ?? {}) as Record<string, any>;
  const ml = (cycle.ml_correction ?? {}) as Record<string, any>;

  const confidence = Number(verdict.confidence_score ?? 0);
  const rul = Number(ml.rul_hours ?? physics.rul_hours ?? NaN);
  const stress = Number(physics.stress_amplitude_mpa ?? verdict.stress_delta_mpa ?? NaN);
  const rootCause = verdict.root_cause_llm || verdict.root_cause || '';

  const schedule = computeSchedule(cycle);
  const kind = deriveKind(schedule.urgency, cycle);
  const meta = KIND_META[kind];
  const Icon = meta.icon;

  // If the anomaly gate fired but stress-based health is still normal
  // ("ok" banner), acknowledge the anomaly in the headline so the
  // banner doesn't contradict the "Anomaly DETECTED" KPI card below.
  // We keep the banner green because health remains safe — this is
  // just a wording tweak, not a colour override.
  const headline =
    kind === 'ok' && cycle.is_anomalous
      ? 'Anomalies detected — stress still within safe envelope'
      : meta.headline;

  return (
    <div className={`relative overflow-hidden rounded-xl border ${meta.border} ${meta.bg} p-5 shadow-sm`}>
      {/* soft radial highlight */}
      <div
        className="pointer-events-none absolute inset-0 opacity-40"
        style={{
          background: `radial-gradient(circle at 20% 0%, ${meta.glow}, transparent 55%)`,
        }}
      />
      <div className="relative flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="flex items-start gap-4">
          <div className={`rounded-lg border ${meta.iconBorder} ${meta.iconBg} p-3`}>
            <Icon className={`h-6 w-6 ${meta.iconColor}`} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={`text-xs font-semibold uppercase tracking-wider ${meta.accent}`}>
                {meta.label}
              </span>
              <Badge variant="outline" className="border-current text-[10px] font-mono">
                schedule: {schedule.urgencyLabel.toLowerCase()}
              </Badge>
            </div>
            <h2 className="mt-1 text-xl font-semibold tracking-tight">{headline}</h2>
            {rootCause ? (
              <p className="mt-2 max-w-2xl text-sm text-muted-foreground">{String(rootCause)}</p>
            ) : null}
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3 md:min-w-[380px]">
          <HeroStat icon={<Clock className="h-3.5 w-3.5" />} label="RUL" value={Number.isNaN(rul) ? '—' : fmtHours(rul)} />
          <HeroStat icon={<Gauge className="h-3.5 w-3.5" />} label="Stress" value={Number.isNaN(stress) ? '—' : `${fmtNumber(stress, 0)} MPa`} />
          <HeroStat icon={<ShieldCheck className="h-3.5 w-3.5" />} label="Confidence" value={confidence ? `${(confidence * 100).toFixed(0)}%` : '—'} />
        </div>
      </div>
    </div>
  );
}

function HeroStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-md border bg-background/60 p-2.5 backdrop-blur">
      <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {icon}
        {label}
      </div>
      <div className="mt-1 text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

type Kind = 'critical' | 'action' | 'monitor' | 'ok' | 'running';

/**
 * Map the lifecycle-scheduler's UrgencyBand → the hero's visual kind.
 * Keeps the hero, the schedule card, and the recommendation list all
 * in lockstep with the same physics-derived health_status.
 */
function deriveKind(urgency: string, cycle: CycleSnapshot): Kind {
  if (cycle.status === 'running') return 'running';
  switch (urgency) {
    case 'immediate':
      return 'critical';
    case 'urgent':
      return 'action';
    case 'planned':
      return 'monitor';
    case 'routine':
      return 'ok';
    default:
      return cycle.status === 'unresolved_divergence' ? 'action' : 'ok';
  }
}

const KIND_META: Record<Kind, {
  label: string;
  headline: string;
  border: string;
  bg: string;
  glow: string;
  accent: string;
  icon: React.ComponentType<{ className?: string }>;
  iconBg: string;
  iconBorder: string;
  iconColor: string;
}> = {
  critical: {
    label: 'Critical',
    headline: 'Immediate maintenance required',
    border: 'border-red-500/40',
    bg: 'bg-red-500/5',
    glow: 'rgba(239, 68, 68, 0.35)',
    accent: 'text-red-700 dark:text-red-300',
    icon: AlertOctagon,
    iconBg: 'bg-red-500/10',
    iconBorder: 'border-red-500/30',
    iconColor: 'text-red-600 dark:text-red-400',
  },
  action: {
    label: 'Action recommended',
    headline: 'Schedule maintenance soon',
    border: 'border-amber-500/40',
    bg: 'bg-amber-500/5',
    glow: 'rgba(245, 158, 11, 0.3)',
    accent: 'text-amber-800 dark:text-amber-200',
    icon: ShieldAlert,
    iconBg: 'bg-amber-500/10',
    iconBorder: 'border-amber-500/30',
    iconColor: 'text-amber-600 dark:text-amber-400',
  },
  monitor: {
    label: 'Monitor',
    headline: 'Continue observation — no action needed',
    border: 'border-blue-500/40',
    bg: 'bg-blue-500/5',
    glow: 'rgba(59, 130, 246, 0.3)',
    accent: 'text-blue-700 dark:text-blue-300',
    icon: Gauge,
    iconBg: 'bg-blue-500/10',
    iconBorder: 'border-blue-500/30',
    iconColor: 'text-blue-600 dark:text-blue-400',
  },
  ok: {
    label: 'Healthy',
    headline: 'No anomalies detected',
    border: 'border-emerald-500/40',
    bg: 'bg-emerald-500/5',
    glow: 'rgba(16, 185, 129, 0.3)',
    accent: 'text-emerald-700 dark:text-emerald-300',
    icon: CheckCircle2,
    iconBg: 'bg-emerald-500/10',
    iconBorder: 'border-emerald-500/30',
    iconColor: 'text-emerald-600 dark:text-emerald-400',
  },
  running: {
    label: 'In progress',
    headline: 'Pipeline running — updates streaming in…',
    border: 'border-primary/40',
    bg: 'bg-primary/5',
    glow: 'rgba(59, 130, 246, 0.35)',
    accent: 'text-primary',
    icon: Clock,
    iconBg: 'bg-primary/10',
    iconBorder: 'border-primary/30',
    iconColor: 'text-primary',
  },
};
