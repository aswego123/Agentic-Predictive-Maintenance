import { QueryClient } from '@tanstack/react-query';

/**
 * Global React Query client.
 *
 * Design notes:
 *   - staleTime 5s: server state is fresh enough for a few seconds; keeps
 *     mount-remount thrash from hammering the API.
 *   - retry: 1 for GETs — the Streamlit app just showed the error, so
 *     one retry is friendlier without hiding real backend outages.
 *   - refetchOnWindowFocus: false — long-running cycles shouldn't kick
 *     off surprise refetches when the user tabs back.
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      gcTime: 5 * 60_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 0,
    },
  },
});
