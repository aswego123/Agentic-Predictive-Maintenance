import axios, { type AxiosError } from 'axios';
import { toast } from 'sonner';

import { useAppStore } from '@/store';
import type {
  AnalyzeRequest,
  CycleSnapshot,
  EngineerApproveRequest,
  FleetStatus,
  HealthResponse,
} from '@/types/domain';

/**
 * Shared axios instance. baseURL is pulled fresh from the Zustand store
 * on every request so that a user changing `apiUrl` under Settings
 * takes effect immediately without a full reload.
 *
 * Timeouts intentionally mirror the Streamlit dashboard:
 *   GET  → 60s   (fleet/status can be slow on cold cache)
 *   POST → 300s  (Gemini "thinking" + Judge ReAct loop = 90-150s)
 */
export const http = axios.create({
  baseURL: useAppStore.getState().session.apiUrl,
  timeout: 60_000,
  headers: { 'Content-Type': 'application/json' },
});

// Keep baseURL in sync with the store.
useAppStore.subscribe((state) => {
  http.defaults.baseURL = state.session.apiUrl;
});

// Global error interceptor → toast, then rethrow so React Query still
// sees the failure and can drive its own state.
http.interceptors.response.use(
  (r) => r,
  (error: AxiosError<{ detail?: string }>) => {
    const detail =
      error.response?.data?.detail ||
      error.message ||
      'Unknown API error';
    toast.error(`API error: ${detail}`, {
      description: `${error.config?.method?.toUpperCase() ?? '?'} ${error.config?.url ?? ''}`,
    });
    return Promise.reject(error);
  },
);

// ---------------------------------------------------------------
// Endpoint functions — thin wrappers, one per FastAPI route.
// ---------------------------------------------------------------
export const api = {
  health: () => http.get<HealthResponse>('/health').then((r) => r.data),

  analyze: (req: AnalyzeRequest) =>
    http
      .post<CycleSnapshot>('/analyze', req, { timeout: 300_000 })
      .then((r) => r.data),

  getFleetStatus: () =>
    http.get<FleetStatus>('/fleet/status').then((r) => r.data),

  getCycle: (cycleId: string) =>
    http.get<CycleSnapshot>(`/cycles/${cycleId}`).then((r) => r.data),

  approve: (req: EngineerApproveRequest) =>
    http
      .post<CycleSnapshot>('/engineer/approve', req, { timeout: 300_000 })
      .then((r) => r.data),
};
