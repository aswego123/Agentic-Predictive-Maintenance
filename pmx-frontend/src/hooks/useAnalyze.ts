import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';

import { api } from '@/services/api';
import { useAppStore } from '@/store';
import type { AnalyzeRequest, CycleSnapshot } from '@/types/domain';

/**
 * Mutation for POST /analyze. On success, primes the react-query cache
 * for the new cycle and updates the store so /cycles/:id can pre-render
 * from cache while the poll takes over.
 */
export function useAnalyze() {
  const queryClient = useQueryClient();
  const setLastCycleId = useAppStore((s) => s.setLastCycleId);
  const setCycle = useAppStore((s) => s.setCycle);
  const setPolling = useAppStore((s) => s.setPolling);

  return useMutation<CycleSnapshot, Error, AnalyzeRequest>({
    mutationFn: api.analyze,
    onSuccess: (cycle) => {
      queryClient.setQueryData(['cycle', cycle.cycle_id], cycle);
      setCycle(cycle);
      setLastCycleId(cycle.cycle_id);
      // Keep polling briefly in case the graph continues past the initial
      // return (interrupted at engineer_approval, resim rounds, etc.).
      setPolling(cycle.cycle_id, true);
      toast.success(`Cycle ${cycle.cycle_id} started`, {
        description: `status=${cycle.status}`,
      });
    },
  });
}
