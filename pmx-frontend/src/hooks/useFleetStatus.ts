import { useQuery } from '@tanstack/react-query';

import { api } from '@/services/api';
import { useAppStore } from '@/store';

/**
 * Fleet-wide summary. Polled every 30s while mounted.
 */
export function useFleetStatus() {
  const setFleet = useAppStore((s) => s.setFleet);

  return useQuery({
    queryKey: ['fleet', 'status'],
    queryFn: async () => {
      const data = await api.getFleetStatus();
      setFleet(data);
      return data;
    },
    refetchInterval: 30_000,
  });
}
