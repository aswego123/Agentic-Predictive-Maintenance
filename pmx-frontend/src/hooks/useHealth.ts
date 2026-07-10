import { useQuery } from '@tanstack/react-query';

import { api } from '@/services/api';

/**
 * Lightweight liveness check. Powers the topbar heartbeat dot.
 * Polled every 30s so a backend restart is noticed quickly without
 * flooding the API.
 */
export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30_000,
    retry: 2,
  });
}
