import { cn } from '@/lib/utils';

export type MetricTone = 'default' | 'success' | 'warning' | 'danger' | 'info';

interface MetricCardProps {
  label: string;
  value: React.ReactNode;
  delta?: React.ReactNode;
  hint?: React.ReactNode;
  tone?: MetricTone;
  icon?: React.ReactNode;
  className?: string;
}

const TONE_CLASSES: Record<MetricTone, { border: string; bg: string; value: string }> = {
  default: { border: 'border-border', bg: 'bg-card', value: '' },
  success: { border: 'border-emerald-500/30', bg: 'bg-emerald-500/5', value: 'text-emerald-700 dark:text-emerald-300' },
  warning: { border: 'border-amber-500/30', bg: 'bg-amber-500/5', value: 'text-amber-800 dark:text-amber-300' },
  danger:  { border: 'border-red-500/30', bg: 'bg-red-500/5', value: 'text-red-700 dark:text-red-300' },
  info:    { border: 'border-blue-500/30', bg: 'bg-blue-500/5', value: 'text-blue-700 dark:text-blue-300' },
};

/**
 * Small metric tile — replaces Streamlit's `st.metric` widget.
 * Accepts a `tone` for status-coloring and an optional icon.
 */
export function MetricCard({ label, value, delta, hint, tone = 'default', icon, className }: MetricCardProps) {
  const t = TONE_CLASSES[tone];
  return (
    <div className={cn('rounded-lg border p-4 transition-colors', t.border, t.bg, className)}>
      <div className="flex items-center justify-between">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          {label}
        </div>
        {icon ? <span className="text-muted-foreground/70">{icon}</span> : null}
      </div>
      <div className={cn('mt-1 text-xl font-semibold tabular-nums', t.value)}>
        {value ?? '—'}
      </div>
      {delta ? <div className="mt-0.5 text-xs text-muted-foreground">{delta}</div> : null}
      {hint ? <div className="mt-0.5 text-[11px] text-muted-foreground/80">{hint}</div> : null}
    </div>
  );
}

/**
 * Pick a tone from a numeric value using threshold bands. Used by
 * cycle-detail pages to color-code RUL, confidence, etc.
 */
export function toneForNumber(
  value: number | null | undefined,
  { danger, warning }: { danger: number; warning: number },
  inverted = false,
): MetricTone {
  if (value === null || value === undefined || Number.isNaN(value)) return 'default';
  if (inverted) {
    // High = bad (e.g. stress ratio)
    if (value >= danger) return 'danger';
    if (value >= warning) return 'warning';
    return 'success';
  }
  // Low = bad (e.g. remaining hours)
  if (value <= danger) return 'danger';
  if (value <= warning) return 'warning';
  return 'success';
}
