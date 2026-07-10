import { Check, Circle, Pause, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { CycleSnapshot } from '@/types/domain';

export type Stage = {
  id: string;
  label: string;
  present: boolean;
  paused?: boolean;
  active?: boolean;
};

/**
 * Derive per-stage completion from a CycleSnapshot. A stage is marked:
 *   - `present` when its output field is populated in state
 *   - `paused` for the approval stage while the graph is interrupted
 *   - `active` for the *first* not-yet-present stage while status=running
 */
export function stagesFromSnapshot(cycle: CycleSnapshot): Stage[] {
  const raw: Stage[] = [
    { id: 'anomaly', label: 'Anomaly', present: !!cycle.anomaly_result },
    { id: 'physics_ml', label: 'Physics + ML', present: !!cycle.physics_prediction || !!cycle.ml_correction },
    { id: 'dialogue', label: 'Dialogue', present: (cycle.dialogue_history?.length ?? 0) > 0 },
    { id: 'judge', label: 'Judge', present: !!cycle.judge_verdict },
    { id: 'calibration', label: 'Calibration', present: !!cycle.calibration_result },
    { id: 'approval', label: 'Approval', present: !!cycle.engineer_decision, paused: cycle.awaiting_engineer_approval },
    { id: 'action', label: 'Action', present: !!cycle.action || !!cycle.work_order },
    { id: 'critic', label: 'Critic', present: !!cycle.critic_review },
  ];

  if (cycle.status === 'running') {
    const nextIdx = raw.findIndex((s) => !s.present && !s.paused);
    if (nextIdx >= 0) raw[nextIdx].active = true;
  }
  return raw;
}

export function StageStepper({
  cycle,
  onSelect,
  activeId,
}: {
  cycle: CycleSnapshot;
  onSelect?: (id: string) => void;
  activeId?: string | null;
}) {
  const stages = stagesFromSnapshot(cycle);

  return (
    <div className="flex flex-wrap items-center gap-2">
      {stages.map((s, i) => {
        const state: 'done' | 'paused' | 'active' | 'pending' = s.paused
          ? 'paused'
          : s.present
            ? 'done'
            : s.active
              ? 'active'
              : 'pending';
        const isSelected = activeId === s.id;
        return (
          <button
            key={s.id}
            type="button"
            onClick={() => onSelect?.(s.id)}
            className={cn(
              'flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium transition-all',
              state === 'done' && 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
              state === 'paused' && 'border-amber-500/40 bg-amber-500/10 text-amber-800 dark:text-amber-300 animate-soft-pulse',
              state === 'active' && 'border-primary/50 bg-primary/10 text-primary shadow-sm shadow-primary/20 animate-soft-pulse',
              state === 'pending' && 'border-muted bg-muted/40 text-muted-foreground',
              isSelected && 'ring-2 ring-primary/40 ring-offset-1 ring-offset-background',
            )}
          >
            <span className="text-[10px] tabular-nums text-muted-foreground">{i + 1}</span>
            {state === 'done' ? (
              <Check className="h-3.5 w-3.5" />
            ) : state === 'paused' ? (
              <Pause className="h-3.5 w-3.5" />
            ) : state === 'active' ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Circle className="h-3.5 w-3.5" />
            )}
            {s.label}
          </button>
        );
      })}
    </div>
  );
}
