import { useState } from 'react';
import {
  Wrench,
  Database,
  Search,
  FlaskConical,
  ClipboardCheck,
  BookOpen,
  ChevronDown,
} from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { JsonViewer } from '@/components/common/JsonViewer';
import { cn } from '@/lib/utils';
import type { CycleSnapshot } from '@/types/domain';

const TOOL_META: Record<string, { icon: React.ComponentType<{ className?: string }>; label: string; blurb: string; color: string }> = {
  get_fleet_memory_stats: {
    icon: Database,
    label: 'Fleet memory stats',
    blurb: 'Reads the last N cycle summaries for this asset from fleet memory.',
    color: 'text-blue-600 dark:text-blue-400',
  },
  find_similar_failures: {
    icon: Search,
    label: 'Find similar failures',
    blurb: 'Matches past cycles within ±tolerance MPa of the current physics stress.',
    color: 'text-purple-600 dark:text-purple-400',
  },
  query_material_limits: {
    icon: BookOpen,
    label: 'Query material limits',
    blurb: 'Looks up UTS / endurance / yield / K_IC and derived stress ratios.',
    color: 'text-emerald-600 dark:text-emerald-400',
  },
  simulate_what_if: {
    icon: FlaskConical,
    label: 'What-if simulation',
    blurb: 'Re-runs the physics engine with perturbed stress / crack size to see if the RUL cliffs.',
    color: 'text-amber-600 dark:text-amber-400',
  },
  get_recent_maintenance: {
    icon: ClipboardCheck,
    label: 'Recent maintenance',
    blurb: 'Checks for cycle_action / engineer_decision entries within the look-back window.',
    color: 'text-cyan-600 dark:text-cyan-400',
  },
};

/**
 * Try to parse a JSON snippet — the backend sends observation_snippet
 * as a trimmed string. If it's valid JSON we render structured;
 * otherwise fall back to a preformatted block.
 */
function tryParseJson(v: unknown): unknown | null {
  if (typeof v !== 'string') return v ?? null;
  const trimmed = v.trim();
  if (!trimmed) return null;
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) return trimmed;
  try {
    return JSON.parse(trimmed);
  } catch {
    // Sometimes the snippet is truncated with "..." — try to close it.
    if (trimmed.endsWith('…') || trimmed.endsWith('...')) {
      return trimmed;
    }
    return trimmed;
  }
}

export function JudgeToolCallsPanel({ cycle }: { cycle: CycleSnapshot }) {
  const verdict = cycle.judge_verdict ?? {};
  const madeCount = Number(verdict.tool_calls_made ?? 0);
  const toolCalls = (cycle.trace ?? []).filter((t) =>
    String(t.node ?? '').startsWith('judge_agent.tool_call'),
  );
  if (!madeCount || toolCalls.length === 0) return null;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="flex items-center gap-2 text-base">
          <Wrench className="h-4 w-4 text-primary" />
          Judge tool calls
          <Badge variant="outline" className="ml-1 text-[10px]">{toolCalls.length}</Badge>
        </CardTitle>
        <Badge variant="outline" className="font-mono text-[10px]">
          source: {String(verdict.source ?? '?')}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        {toolCalls.map((entry, i) => (
          <ToolCallRow key={i} index={i + 1} entry={entry} />
        ))}
      </CardContent>
    </Card>
  );
}

function ToolCallRow({ index, entry }: { index: number; entry: CycleSnapshot['trace'][number] }) {
  const [open, setOpen] = useState(false);
  const data = (entry.data ?? {}) as Record<string, any>;
  const toolName = String(data.tool ?? '?');
  const args = data.args ?? {};
  const meta = TOOL_META[toolName] ?? { icon: Wrench, label: toolName, blurb: '', color: 'text-primary' };
  const Icon = meta.icon;
  const observation = tryParseJson(data.observation_snippet ?? entry.note ?? '');

  return (
    <div className="rounded-md border transition-colors hover:border-primary/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-3 p-3 text-left"
      >
        <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-semibold text-muted-foreground">
          {index}
        </div>
        <Icon className={cn('mt-0.5 h-4 w-4 shrink-0', meta.color)} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="text-sm font-semibold">{meta.label}</span>
            <code className="rounded bg-muted px-1 text-[10px] text-muted-foreground">
              {toolName}
            </code>
          </div>
          {meta.blurb ? (
            <p className="mt-0.5 text-xs text-muted-foreground">{meta.blurb}</p>
          ) : null}
          <div className="mt-1 flex flex-wrap gap-1">
            {Object.entries(args).map(([k, v]) => (
              <span key={k} className="inline-flex items-center gap-1 rounded bg-muted/50 px-1.5 py-0.5 text-[10px] font-mono">
                <span className="text-muted-foreground">{k}=</span>
                <span>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
              </span>
            ))}
          </div>
        </div>
        <ChevronDown className={cn('h-4 w-4 shrink-0 text-muted-foreground transition-transform', open && 'rotate-180')} />
      </button>
      {open ? (
        <div className="border-t bg-muted/30 p-3">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Observation
          </div>
          {typeof observation === 'string' ? (
            <pre className="overflow-x-auto whitespace-pre-wrap rounded bg-background p-2 font-mono text-[11px] leading-relaxed">
              {observation}
            </pre>
          ) : observation ? (
            <JsonViewer data={observation} maxHeight={280} />
          ) : (
            <div className="text-xs text-muted-foreground">No observation returned.</div>
          )}
        </div>
      ) : null}
    </div>
  );
}
