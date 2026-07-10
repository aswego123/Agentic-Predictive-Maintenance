import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';

import { api } from '@/services/api';
import { useAppStore } from '@/store';
import type { CycleSnapshot, EngineerApproveRequest } from '@/types/domain';

/**
 * Mutation for POST /engineer/approve.
 *
 * Uses an optimistic update: the cached cycle is immediately flipped to
 * `awaiting_engineer_approval=false, status='running'` so the UI doesn't
 * hang for the entire post-approval graph run. The real server response
 * reconciles the cache when it lands.
 */
export function useApproveCycle() {
  const queryClient = useQueryClient();
  const setCycle = useAppStore((s) => s.setCycle);

  return useMutation<
    CycleSnapshot,
    Error,
    EngineerApproveRequest,
    { previous?: CycleSnapshot }
  >({
    mutationFn: api.approve,
    onMutate: async (req) => {
      const key = ['cycle', req.cycle_id];
      await queryClient.cancelQueries({ queryKey: key });
      const previous = queryClient.getQueryData<CycleSnapshot>(key);
      if (previous) {
        queryClient.setQueryData<CycleSnapshot>(key, {
          ...previous,
          awaiting_engineer_approval: false,
          status: 'running',
        });
      }
      return { previous };
    },
    onError: (_err, req, ctx) => {
      if (ctx?.previous) {
        queryClient.setQueryData(['cycle', req.cycle_id], ctx.previous);
      }
    },
    onSuccess: (cycle) => {
      queryClient.setQueryData(['cycle', cycle.cycle_id], cycle);
      setCycle(cycle);
      toast.success(`Cycle ${cycle.cycle_id} resumed`, {
        description: `status=${cycle.status}`,
      });
    },
  });
}
