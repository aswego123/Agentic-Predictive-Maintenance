import { useEffect } from 'react';
import { Brain } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { MetricCard } from '@/components/common/MetricCard';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { JsonViewer } from '@/components/common/JsonViewer';
import { Sparkline } from '@/components/common/Sparkline';
import { useAppStore } from '@/store';
import { fmtNumber } from '@/lib/format';

export function CriticPanel({
  critic,
  assetId,
}: {
  critic: Record<string, any> | null;
  assetId: string | null;
}) {
  const pushWeight = useAppStore((s) => s.pushPhysicsWeight);
  const weightHistory = useAppStore((s) =>
    assetId ? s.history.physicsWeightByAsset[assetId] ?? [] : [],
  );

  const newW = critic?.physics_weight;

  useEffect(() => {
    if (assetId && typeof newW === 'number' && !Number.isNaN(newW)) {
      pushWeight(assetId, Number(newW.toFixed(3)));
    }
  }, [assetId, newW, pushWeight]);

  if (!critic) return null;
  const prevW = critic.previous_physics_weight;
  const nCycles = critic.n_cycles_considered ?? 0;
  const rationale = critic.rationale;
  const delta =
    typeof prevW === 'number' && typeof newW === 'number' ? Number((newW - prevW).toFixed(3)) : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Brain className="h-4 w-4 text-primary" />
          Critic review (post-cycle reflection)
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <MetricCard label="Cycles considered" value={nCycles} />
          <MetricCard
            label="Physics weight"
            value={fmtNumber(newW ?? 0.8, 2)}
            delta={delta !== null && delta !== 0 ? `${delta > 0 ? '+' : ''}${delta.toFixed(2)}` : undefined}
            tone={delta === null ? 'default' : delta > 0 ? 'info' : delta < 0 ? 'warning' : 'default'}
          />
          <MetricCard label="Prev physics weight" value={prevW !== undefined ? fmtNumber(prevW, 2) : '—'} />
        </div>

        {weightHistory.length >= 2 ? (
          <div className="rounded-md border bg-card p-3">
            <div className="mb-1 flex items-center justify-between">
              <div className="text-xs font-semibold uppercase text-muted-foreground">
                Physics weight trend
              </div>
              <div className="text-[10px] text-muted-foreground">
                {weightHistory.length} points · session-local
              </div>
            </div>
            <Sparkline history={weightHistory} height={48} />
          </div>
        ) : null}

        {rationale ? (
          <Alert variant="info">
            <AlertDescription>{String(rationale)}</AlertDescription>
          </Alert>
        ) : null}

        <Accordion type="single" collapsible>
          <AccordionItem value="raw" className="border-0">
            <AccordionTrigger className="text-xs">Raw critic payload</AccordionTrigger>
            <AccordionContent>
              <JsonViewer data={critic} />
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}
