import { useQuery } from '@tanstack/react-query';

import { api } from '@/services/api';
import { useAppStore } from '@/store';

/**
 * Fetch a single cycle snapshot. Auto-polls fast (750 ms) while the
 * cycle is `running` OR while the store flags it as actively polling
 * (e.g. right after /analyze fires) — the pipeline graph animates
 * off the streamed trace, so we want new nodes to light up promptly.
 * Slows to 3s idle polling if the cycle is paused at engineer_approval
 * (LangGraph won't move until the user acts).
 */
export function useCycle(cycleId: string | null | undefined) {
  const isPolling = useAppStore((s) =>
    cycleId ? !!s.cycles.polling[cycleId] : false,
  );

  return useQuery({
    queryKey: ['cycle', cycleId],
    queryFn: () => api.getCycle(cycleId!),
    enabled: !!cycleId,
    refetchInterval: (q) => {
      const data = q.state.data;
      if (isPolling) return 750;
      if (!data) return false;
      if (data.status === 'running' || data.status === 'pending') return 750;
      if (data.awaiting_engineer_approval) return 3_000;
      return false;
    },
  });
}
