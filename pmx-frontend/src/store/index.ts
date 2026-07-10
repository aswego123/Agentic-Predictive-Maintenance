import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

import { type SessionState, createSessionSlice } from './sessionSlice';
import { type AnalyzeFormState, createAnalyzeFormSlice } from './analyzeFormSlice';
import { type CyclesState, createCyclesSlice } from './cyclesSlice';
import { type FleetState, createFleetSlice } from './fleetSlice';
import { type UiState, createUiSlice } from './uiSlice';
import { type HistoryState, createHistorySlice } from './historySlice';

export type AppState = SessionState &
  AnalyzeFormState &
  CyclesState &
  FleetState &
  UiState &
  HistoryState;

/**
 * Single Zustand store composed from feature slices.
 *
 * Persistence rule: only `session` + `form` + `history` survive reloads.
 * Everything else is intentionally in-memory so a refresh always
 * re-syncs with the backend.
 */
export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (...a) => ({
        ...createSessionSlice(...a),
        ...createAnalyzeFormSlice(...a),
        ...createCyclesSlice(...a),
        ...createFleetSlice(...a),
        ...createUiSlice(...a),
        ...createHistorySlice(...a),
      }),
      {
        name: 'eix-store',
        partialize: (state) => ({
          session: state.session,
          form: state.form,
          history: state.history,
        }),
      },
    ),
    { name: 'EIxStore' },
  ),
);
