import { AlertTriangle, Radar } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { MetricCard } from '@/components/common/MetricCard';
import { EmptyState } from '@/components/common/EmptyState';
import { fmtBool, fmtNumber } from '@/lib/format';
import { cn } from '@/lib/utils';

type AnomalyPayload = {
  is_anomalous?: boolean;
  anomaly_count?: number;
  max_severity?: string;
  top_channel?: string | null;
  top_score?: number;
  severity_breakdown?: Record<string, number>;
  by_type?: Record<string, number>;
  top_anomalies?: Array<{
    anomaly_type?: string;
    sensor?: string;
    value?: number | null;
    threshold?: number | null;
    score?: number | null;
    severity?: string;
    probable_failure_mode?: string | null;
    trend_direction?: string;
    remediation_hint?: string;
  }>;
} | null;

/**
 * Colored dot per severity band. Mirrors the CLI legend:
 *   🔴 critical / failed
 *   🟠 warning
 *   🟡 monitoring
 *   🟢 normal
 */
const SEVERITY_STYLE: Record<string, { dot: string; emoji: string; label: string; row: string }> = {
  failed:     { dot: 'bg-red-600',     emoji: '🔴', label: 'critical', row: 'bg-red-500/5 border-l-red-500' },
  critical:   { dot: 'bg-red-500',     emoji: '🔴', label: 'critical', row: 'bg-red-500/5 border-l-red-500' },
  warning:    { dot: 'bg-amber-500',   emoji: '🟠', label: 'high',     row: 'bg-amber-500/5 border-l-amber-500' },
  monitoring: { dot: 'bg-yellow-400',  emoji: '🟡', label: 'medium',   row: 'bg-yellow-500/5 border-l-yellow-400' },
  normal:     { dot: 'bg-emerald-500', emoji: '🟢', label: 'normal',   row: 'bg-emerald-500/5 border-l-emerald-500' },
};

// Order used when rendering the severity chips — worst → best.
const SEVERITY_ORDER = ['failed', 'critical', 'warning', 'monitoring', 'normal'];

/**
 * Format one row like the CLI:
 *   `1. 🟠 vibration_spike  | sensor=vibration  value=0.72  expected=0.22  score=2.28`
 * We render it as monospace with padded columns so multiple rows line up
 * exactly like the terminal output.
 */
function padRight(s: string, n: number): string {
  return s.length >= n ? s.slice(0, n) : s + ' '.repeat(n - s.length);
}

export function AnomalyPanel({ anomaly }: { anomaly: AnomalyPayload }) {
  if (!anomaly) return null;

  const breakdown = anomaly.severity_breakdown ?? {};
  const byType = anomaly.by_type ?? {};
  const top = anomaly.top_anomalies ?? [];
  const remaining = Math.max(0, (anomaly.anomaly_count ?? 0) - top.length);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Radar className="h-4 w-4 text-primary" />
          Anomaly gate
          {anomaly.is_anomalous ? (
            <Badge variant="destructive" className="ml-1">Anomalies present</Badge>
          ) : (
            <Badge variant="success" className="ml-1">Clean</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <MetricCard
            label="Is anomalous?"
            value={fmtBool(anomaly.is_anomalous)}
            tone={anomaly.is_anomalous ? 'warning' : 'success'}
          />
          <MetricCard
            label="Count"
            value={anomaly.anomaly_count ?? '—'}
            tone={anomaly.anomaly_count && anomaly.anomaly_count > 0 ? 'info' : 'default'}
          />
        </div>

        {anomaly.top_channel ? (
          <p className="text-xs text-muted-foreground">
            Highest-scoring anomaly was on channel{' '}
            <code className="rounded bg-muted px-1">{String(anomaly.top_channel)}</code> with score{' '}
            <strong className="tabular-nums">{fmtNumber(anomaly.top_score, 2)}</strong>.
          </p>
        ) : null}

        {/* Severity breakdown chips — mirrors CLI: "🟠 high: 117  🟡 medium: 145  🔴 critical: 213" */}
        {Object.keys(breakdown).length > 0 ? (
          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Severity breakdown
            </div>
            <div className="flex flex-wrap gap-2">
              {SEVERITY_ORDER.filter((k) => breakdown[k]).map((k) => {
                const meta = SEVERITY_STYLE[k] ?? SEVERITY_STYLE.normal;
                return (
                  <div key={k} className="flex items-center gap-2 rounded-full border bg-card px-3 py-1.5 text-xs">
                    <span className="text-sm leading-none">{meta.emoji}</span>
                    <span className="font-medium">{meta.label}</span>
                    <span className="tabular-nums text-muted-foreground">{breakdown[k]}</span>
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}

        {/* By-type breakdown — chips for vibration_spike / temp_spike / etc. */}
        {Object.keys(byType).length > 0 ? (
          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              By type
            </div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(byType)
                .sort((a, b) => b[1] - a[1])
                .map(([type, count]) => (
                  <div key={type} className="flex items-center gap-2 rounded-full border bg-muted/30 px-3 py-1 text-xs">
                    <code className="font-mono text-[11px]">{type}</code>
                    <span className="tabular-nums text-muted-foreground">{count}</span>
                  </div>
                ))}
            </div>
          </div>
        ) : null}

        {/* Top-N detected anomalies — CLI-style monospace list */}
        {top.length > 0 ? (
          <div>
            <div className="mb-2 flex items-center justify-between">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Detected anomalies (top {top.length})
              </div>
              {remaining > 0 ? (
                <span className="text-[11px] italic text-muted-foreground">
                  …and {remaining} more
                </span>
              ) : null}
            </div>
            <div className="overflow-x-auto rounded-md border bg-muted/10 p-2">
              <div className="font-mono text-[11px] leading-relaxed">
                {top.map((row, i) => {
                  const meta = SEVERITY_STYLE[row.severity ?? 'normal'] ?? SEVERITY_STYLE.normal;
                  const idx = padRight(`${i + 1}.`, 4);
                  const type = padRight(row.anomaly_type ?? '?', 22);
                  const sensor = padRight(`sensor=${row.sensor ?? '?'}`, 24);
                  const value = padRight(`value=${row.value ?? '—'}`, 14);
                  const expected = padRight(`expected=${row.threshold ?? '—'}`, 16);
                  const score = `score=${row.score ?? '—'}`;
                  return (
                    <div
                      key={i}
                      className={cn(
                        'flex items-center gap-2 rounded border-l-2 px-2 py-1',
                        meta.row,
                      )}
                    >
                      <span className="text-sm leading-none">{meta.emoji}</span>
                      <span className="whitespace-pre tabular-nums text-foreground/90">
                        {idx}
                        {type}
                        <span className="text-muted-foreground">|  </span>
                        {sensor}
                        {value}
                        {expected}
                        {score}
                      </span>
                    </div>
                  );
                })}
                {remaining > 0 ? (
                  <div className="mt-2 pl-6 italic text-muted-foreground">
                    ...and {remaining} more
                  </div>
                ) : null}
              </div>
            </div>

            {/* Show the top row's remediation hint if present */}
            {top[0]?.remediation_hint ? (
              <div className="mt-2 flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-xs">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
                <div>
                  <strong>Top-anomaly hint:</strong> {top[0].remediation_hint}
                </div>
              </div>
            ) : null}
          </div>
        ) : anomaly.is_anomalous ? (
          <EmptyState title="No per-anomaly detail returned" description="The detector didn't ship per-row diagnostics with this batch." />
        ) : null}
      </CardContent>
    </Card>
  );
}
