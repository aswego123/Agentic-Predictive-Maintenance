import type { StateCreator } from 'zustand';

import type { AppState } from './index';
import type { FleetStatus } from '@/types/domain';

export interface FleetState {
  fleet: {
    data: FleetStatus | null;
    fetchedAt: number | null;
  };
  setFleet: (data: FleetStatus) => void;
  clearFleet: () => void;
}

export const createFleetSlice: StateCreator<
  AppState,
  [['zustand/devtools', never], ['zustand/persist', unknown]],
  [],
  FleetState
> = (set) => ({
  fleet: { data: null, fetchedAt: null },
  setFleet: (data) =>
    set({ fleet: { data, fetchedAt: Date.now() } }, false, 'fleet/set'),
  clearFleet: () =>
    set({ fleet: { data: null, fetchedAt: null } }, false, 'fleet/clear'),
});
