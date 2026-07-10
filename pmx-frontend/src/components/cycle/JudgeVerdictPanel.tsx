import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { MetricCard } from '@/components/common/MetricCard';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { fmtBool, fmtNumber } from '@/lib/format';

export function JudgeVerdictPanel({ verdict }: { verdict: Record<string, any> | null }) {
  if (!verdict) return null;

  const hist = (verdict.historical_context ?? {}) as Record<string, any>;
  const rootCause = verdict.root_cause_llm || verdict.root_cause || '';

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">⚖️ Judge verdict</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <MetricCard label="Confidence" value={fmtNumber(verdict.confidence_score, 2)} />
          <MetricCard label="Divergence" value={fmtBool(verdict.divergence)} />
          <MetricCard label="Maintenance required" value={fmtBool(verdict.maintenance_required)} />
          <MetricCard label="Route" value={String(verdict.route ?? '—')} />
        </div>

        {rootCause ? (
          <Alert variant="info">
            <AlertDescription>{rootCause}</AlertDescription>
          </Alert>
        ) : null}

        {Object.keys(hist).length > 0 ? (
          <div>
            <div className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
              Historical context (fleet memory)
            </div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <MetricCard label="History size" value={hist.history_size ?? 0} />
              <MetricCard label="Past action cycles" value={hist.past_action_cycles ?? 0} />
              <MetricCard label="Past divergence rate" value={fmtNumber(hist.past_divergence_rate, 2)} />
              <MetricCard label="Eng. approval rate" value={fmtNumber(hist.past_engineer_approval_rate, 2)} />
            </div>
            {hist.last_action_type ? (
              <p className="mt-2 text-xs text-muted-foreground">
                Last action for this asset: <strong>{String(hist.last_action_type)}</strong>
              </p>
            ) : null}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
