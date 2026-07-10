import { ArrowRight, Factory, Package, DollarSign, Calendar, ShieldCheck, TrendingUp, Info } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { JsonViewer } from '@/components/common/JsonViewer';
import { fmtPercent } from '@/lib/format';
import { cn } from '@/lib/utils';

interface MaterialRec {
  current_material?: string;
  recommended_material?: string;
  supplier?: string;
  part_number?: string;
  reason?: string;
  expected_improvement?: Record<string, number>;
  cost_impact?: number;
  implementation_days?: number;
  risk_level?: string;
  recommendation_type?: string;
  confidence_score?: number;
}

interface RecommendationsPayload {
  current_material?: string;
  best_recommendation?: MaterialRec | null;
  alternatives?: MaterialRec[];
  supplier_of_current?: string | null;
}

interface Props {
  recommendations: RecommendationsPayload | null | undefined;
  currentMaterial?: string | null;
}

/**
 * CLI-style material recommendation view. Ports the block emitted by
 * anamoly-detection.py::generate_report:
 *   🔄 Recommended Material
 *   📋 Supplier
 *   💡 Reason
 *   📈 Expected Improvement
 *   ⚠️ Risk Level
 *   📅 Implementation Time
 *   💰 Estimated Savings
 */
export function MaterialRecommendationPanel({ recommendations, currentMaterial }: Props) {
  if (!recommendations || !recommendations.best_recommendation) return null;
  const best = recommendations.best_recommendation;
  const alts = (recommendations.alternatives ?? []).filter(
    (r) => r.recommended_material !== best.recommended_material,
  );

  const displayedCurrent =
    currentMaterial ||
    recommendations.current_material ||
    best.current_material ||
    '—';

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Package className="h-4 w-4 text-primary" />
          Material &amp; supplier recommendation
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Current → recommended transition */}
        <div className="flex flex-wrap items-center gap-2 rounded-md border bg-muted/30 p-3">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Current
            </span>
            <code className="rounded bg-muted px-2 py-0.5 font-mono text-xs">{displayedCurrent}</code>
            {recommendations.supplier_of_current ? (
              <span className="text-xs text-muted-foreground">· {recommendations.supplier_of_current}</span>
            ) : null}
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground" />
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-primary">
              Recommended
            </span>
            <code className="rounded bg-primary/10 px-2 py-0.5 font-mono text-xs text-primary">
              {best.recommended_material ?? '—'}
            </code>
          </div>
        </div>

        {/* Detail grid */}
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <DetailRow icon={<Factory className="h-3.5 w-3.5" />} label="Supplier" value={best.supplier ?? '—'} />
          <DetailRow icon={<Package className="h-3.5 w-3.5" />} label="Part number" value={best.part_number ?? '—'} monoValue />
          <DetailRow
            icon={<ShieldCheck className="h-3.5 w-3.5" />}
            label="Risk level"
            value={
              <Badge variant={riskToVariant(best.risk_level)} className="text-[10px]">
                {best.risk_level ?? '—'} · confidence {best.confidence_score !== undefined ? fmtPercent(best.confidence_score, 0) : '—'}
              </Badge>
            }
          />
          <DetailRow
            icon={<Calendar className="h-3.5 w-3.5" />}
            label="Implementation"
            value={best.implementation_days ? `${best.implementation_days} days` : '—'}
          />
          <DetailRow
            icon={<DollarSign className="h-3.5 w-3.5" />}
            label="Cost impact"
            value={
              best.cost_impact !== undefined
                ? costImpactLabel(best.cost_impact)
                : '—'
            }
          />
          <DetailRow
            icon={<Info className="h-3.5 w-3.5" />}
            label="Type"
            value={best.recommendation_type ?? '—'}
          />
        </div>

        {/* Reason */}
        {best.reason ? (
          <div className="rounded-md border border-primary/20 bg-primary/5 p-3 text-sm">
            <div className="mb-1 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-primary">
              <Info className="h-3 w-3" /> Reason
            </div>
            {best.reason}
          </div>
        ) : null}

        {/* Expected improvement */}
        {best.expected_improvement && Object.keys(best.expected_improvement).length > 0 ? (
          <div>
            <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              <TrendingUp className="h-3 w-3" /> Expected improvement
            </div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(best.expected_improvement).map(([k, v]) => (
                <div key={k} className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/5 px-3 py-1 text-xs">
                  <span className="text-muted-foreground">{k}</span>
                  <span className={cn('font-semibold tabular-nums', v > 0 ? 'text-emerald-700 dark:text-emerald-300' : 'text-red-700 dark:text-red-300')}>
                    {v > 0 ? '+' : ''}{fmtPercent(v, 0)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {/* Alternatives */}
        {alts.length > 0 ? (
          <Accordion type="single" collapsible>
            <AccordionItem value="alts" className="border-0">
              <AccordionTrigger className="text-xs">
                Alternative recommendations ({alts.length})
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-2">
                  {alts.map((a, i) => (
                    <div key={i} className="rounded-md border p-3 text-xs">
                      <div className="mb-1 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <code className="font-mono">{a.recommended_material}</code>
                          <span className="text-muted-foreground">· {a.supplier ?? '—'}</span>
                        </div>
                        <Badge variant={riskToVariant(a.risk_level)} className="text-[10px]">
                          {a.risk_level ?? '—'} · {fmtPercent(a.confidence_score, 0)}
                        </Badge>
                      </div>
                      {a.reason ? (
                        <p className="text-muted-foreground">{a.reason}</p>
                      ) : null}
                      {a.expected_improvement ? (
                        <div className="mt-1 flex flex-wrap gap-1 text-[10px]">
                          {Object.entries(a.expected_improvement).map(([k, v]) => (
                            <span key={k} className="rounded bg-muted px-1.5 py-0.5">
                              {k} <span className="font-semibold">{v > 0 ? '+' : ''}{fmtPercent(v, 0)}</span>
                            </span>
                          ))}
                        </div>
                      ) : null}
                      <div className="mt-1 text-[10px] text-muted-foreground">
                        {a.implementation_days ? `${a.implementation_days} days` : ''}{' · '}
                        {a.cost_impact !== undefined ? costImpactLabel(a.cost_impact) : ''}
                      </div>
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        ) : null}

        <Accordion type="single" collapsible>
          <AccordionItem value="raw" className="border-0">
            <AccordionTrigger className="text-xs">Raw recommendation payload</AccordionTrigger>
            <AccordionContent>
              <JsonViewer data={recommendations} />
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}

function DetailRow({
  icon,
  label,
  value,
  monoValue,
}: {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
  monoValue?: boolean;
}) {
  return (
    <div className="rounded-md border p-3">
      <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {icon}
        {label}
      </div>
      <div className={cn('mt-1 text-sm', monoValue && 'font-mono')}>{value}</div>
    </div>
  );
}

function costImpactLabel(v: number): string {
  // Convention from the recommender: 1.0 = same cost, 1.3 = 30% costlier.
  if (v === 1) return 'Same cost';
  const pct = (v - 1) * 100;
  return `${pct > 0 ? '+' : ''}${pct.toFixed(0)}% vs current`;
}

function riskToVariant(risk?: string) {
  const r = (risk ?? '').toLowerCase();
  if (r === 'low') return 'success' as const;
  if (r === 'medium') return 'warning' as const;
  if (r === 'high') return 'destructive' as const;
  return 'outline' as const;
}
