import { Badge, type BadgeProps } from '@/components/ui/badge';
import type { CycleStatus } from '@/types/domain';

const STATUS_META: Record<
  string,
  { label: string; variant: BadgeProps['variant']; emoji: string }
> = {
  running: { label: 'Running', variant: 'warning', emoji: '🟡' },
  pending: { label: 'Starting…', variant: 'outline', emoji: '⏳' },
  normal_end: { label: 'Normal end', variant: 'success', emoji: '🟢' },
  action_taken: { label: 'Action taken', variant: 'info', emoji: '🔵' },
  unresolved_divergence: {
    label: 'Unresolved divergence',
    variant: 'destructive',
    emoji: '🔴',
  },
  unknown: { label: 'Unknown', variant: 'outline', emoji: '⚪' },
};

export function StatusBadge({ status }: { status: CycleStatus | string }) {
  const meta = STATUS_META[status] ?? STATUS_META.unknown;
  return (
    <Badge variant={meta.variant} className="gap-1.5">
      <span aria-hidden>{meta.emoji}</span>
      {meta.label}
    </Badge>
  );
}
