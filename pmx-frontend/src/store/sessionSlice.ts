import type { StateCreator } from 'zustand';

import type { AppState } from './index';

export type Theme = 'light' | 'dark';

export interface SessionState {
  session: {
    apiUrl: string;
    engineerId: string;
    theme: Theme;
    lastCycleId: string | null;
  };
  setApiUrl: (url: string) => void;
  setEngineerId: (id: string) => void;
  setTheme: (theme: Theme) => void;
  setLastCycleId: (id: string | null) => void;
}

const DEFAULT_API_URL =
  import.meta.env.VITE_API_URL ?? '/api'; // via Vite proxy in dev

export const createSessionSlice: StateCreator<
  AppState,
  [['zustand/devtools', never], ['zustand/persist', unknown]],
  [],
  SessionState
> = (set) => ({
  session: {
    apiUrl: DEFAULT_API_URL,
    engineerId: 'eng-42',
    theme: 'light',
    lastCycleId: null,
  },
  setApiUrl: (url) =>
    set((s) => ({ session: { ...s.session, apiUrl: url } }), false, 'session/setApiUrl'),
  setEngineerId: (id) =>
    set((s) => ({ session: { ...s.session, engineerId: id } }), false, 'session/setEngineerId'),
  setTheme: (theme) =>
    set((s) => ({ session: { ...s.session, theme } }), false, 'session/setTheme'),
  setLastCycleId: (id) =>
    set((s) => ({ session: { ...s.session, lastCycleId: id } }), false, 'session/setLastCycleId'),
});
