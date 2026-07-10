import { Handshake, Shield, Search, MessageSquare } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { JsonViewer } from '@/components/common/JsonViewer';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { fmtNumber } from '@/lib/format';
import { cn } from '@/lib/utils';

const MOVE_META: Record<string, { icon: React.ComponentType<{ className?: string }>; color: string; label: string }> = {
  concede: { icon: Handshake, color: 'text-emerald-600 dark:text-emerald-400', label: 'Concede' },
  hold: { icon: Shield, color: 'text-blue-600 dark:text-blue-400', label: 'Hold' },
  request_data: { icon: Search, color: 'text-amber-600 dark:text-amber-400', label: 'Request data' },
};

function groupByRound(history: Array<Record<string, any>>): Record<number, Array<Record<string, any>>> {
  const out: Record<number, Array<Record<string, any>>> = {};
  for (const m of history) {
    const r = Number(m.round ?? 0);
    (out[r] ??= []).push(m);
  }
  return out;
}

export function DialoguePanel({
  history,
  fetched,
}: {
  history: Array<Record<string, any>>;
  fetched: Record<string, any>;
}) {
  if (!history || history.length === 0) return null;

  const rounds = groupByRound(history);
  const roundNums = Object.keys(rounds).map(Number).sort((a, b) => a - b);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <MessageSquare className="h-4 w-4 text-primary" />
          Physics ⇄ ML dialogue
          <Badge variant="outline" className="ml-2 text-[10px]">
            {history.length} moves · {roundNums.length} rounds
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {roundNums.map((r) => {
          const moves = rounds[r];
          const physicsMove = moves.find((m) => m.source === 'physics');
          const mlMove = moves.find((m) => m.source === 'ml');
          return (
            <div key={r} className="relative">
              <div className="mb-2 flex items-center gap-2">
                <div className="rounded-full border bg-muted px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Round {r}
                </div>
                <div className="h-px flex-1 bg-border" />
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <MoveCard side="physics" move={physicsMove} />
                <MoveCard side="ml" move={mlMove} />
              </div>
            </div>
          );
        })}

        {fetched && Object.keys(fetched).length > 0 ? (
          <Accordion type="single" collapsible>
            <AccordionItem value="fetched" className="border-0">
              <AccordionTrigger className="text-xs">
                Data-fetch results (from dialogue request)
              </AccordionTrigger>
              <AccordionContent>
                <JsonViewer data={fetched} />
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        ) : null}
      </CardContent>
    </Card>
  );
}

function MoveCard({ side, move }: { side: 'physics' | 'ml'; move?: Record<string, any> }) {
  const sideAccent = side === 'physics' ? 'border-blue-500/30 bg-blue-500/5' : 'border-orange-500/30 bg-orange-500/5';
  const sideBadge = side === 'physics' ? 'bg-blue-500/15 text-blue-700 dark:text-blue-300' : 'bg-orange-500/15 text-orange-700 dark:text-orange-300';

  if (!move) {
    return (
      <div className={cn('rounded-md border border-dashed p-3 text-xs text-muted-foreground', sideAccent)}>
        No move from {side.toUpperCase()} in this round.
      </div>
    );
  }

  const kind = String(move.move ?? '?');
  const meta = MOVE_META[kind] ?? MOVE_META.hold;
  const Icon = meta.icon;
  const revised = move.revised_prediction as Record<string, any> | null;

  return (
    <div className={cn('rounded-md border p-3', sideAccent)}>
      <div className="mb-2 flex items-center justify-between">
        <span className={cn('rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase', sideBadge)}>
          {side}
        </span>
        <div className={cn('flex items-center gap-1.5 text-xs font-semibold', meta.color)}>
          <Icon className="h-3.5 w-3.5" />
          {meta.label}
          {move.data_request ? (
            <code className="ml-1 rounded bg-muted px-1 text-[10px] text-muted-foreground">
              → {String(move.data_request)}
            </code>
          ) : null}
        </div>
      </div>
      {move.rationale ? (
        <p className="text-xs text-muted-foreground">"{String(move.rationale)}"</p>
      ) : null}
      {revised ? (
        <p className="mt-2 text-[11px] text-muted-foreground">
          revised → stress <strong className="tabular-nums">{fmtNumber(revised.stress_mpa, 1)}</strong> MPa,
          RUL <strong className="tabular-nums">{fmtNumber(revised.rul_hours, 1)}</strong> h
        </p>
      ) : null}
    </div>
  );
}
