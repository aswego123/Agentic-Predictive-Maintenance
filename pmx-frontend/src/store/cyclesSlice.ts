import type { StateCreator } from 'zustand';

import type { AppState } from './index';
import type { CycleSnapshot } from '@/types/domain';

export interface CyclesState {
  cycles: {
    byId: Record<string, CycleSnapshot>;
    listOrder: string[];
    polling: Record<string, boolean>;
  };
  setCycle: (cycle: CycleSnapshot) => void;
  setPolling: (id: string, on: boolean) => void;
  invalidateCycle: (id: string) => void;
}

export const createCyclesSlice: StateCreator<
  AppState,
  [['zustand/devtools', never], ['zustand/persist', unknown]],
  [],
  CyclesState
> = (set) => ({
  cycles: {
    byId: {},
    listOrder: [],
    polling: {},
  },
  setCycle: (cycle) =>
    set(
      (s) => {
        const id = cycle.cycle_id;
        const byId = { ...s.cycles.byId, [id]: cycle };
        const listOrder = s.cycles.listOrder.includes(id)
          ? s.cycles.listOrder
          : [id, ...s.cycles.listOrder];
        return { cycles: { ...s.cycles, byId, listOrder } };
      },
      false,
      'cycles/setCycle',
    ),
  setPolling: (id, on) =>
    set(
      (s) => ({
        cycles: {
          ...s.cycles,
          polling: { ...s.cycles.polling, [id]: on },
        },
      }),
      false,
      'cycles/setPolling',
    ),
  invalidateCycle: (id) =>
    set(
      (s) => {
        const { [id]: _drop, ...rest } = s.cycles.byId;
        return { cycles: { ...s.cycles, byId: rest } };
      },
      false,
      'cycles/invalidate',
    ),
});
