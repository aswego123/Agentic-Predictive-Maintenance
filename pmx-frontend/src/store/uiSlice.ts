import type { StateCreator } from 'zustand';

import type { AppState } from './index';

/**
 * Pipeline stages surfaced by the CycleDetailView stepper.
 * Order matters: this is the display order top-to-bottom.
 */
export type PipelineStage =
  | 'anomaly'
  | 'physics_ml'
  | 'dialogue'
  | 'judge'
  | 'calibration'
  | 'approval'
  | 'action'
  | 'critic';

export interface UiState {
  ui: {
    sidebarOpen: boolean;
    activeStage: PipelineStage | null;
  };
  toggleSidebar: () => void;
  setActiveStage: (stage: PipelineStage | null) => void;
}

export const createUiSlice: StateCreator<
  AppState,
  [['zustand/devtools', never], ['zustand/persist', unknown]],
  [],
  UiState
> = (set) => ({
  ui: { sidebarOpen: false, activeStage: null },
  toggleSidebar: () =>
    set(
      (s) => ({ ui: { ...s.ui, sidebarOpen: !s.ui.sidebarOpen } }),
      false,
      'ui/toggleSidebar',
    ),
  setActiveStage: (stage) =>
    set((s) => ({ ui: { ...s.ui, activeStage: stage } }), false, 'ui/setActiveStage'),
});
