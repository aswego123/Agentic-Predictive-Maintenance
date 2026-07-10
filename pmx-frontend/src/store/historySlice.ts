import type { StateCreator } from 'zustand';

import type { AppState } from './index';

/**
 * Per-asset history buffers accumulated *client-side* across the
 * session — used for tiny sparklines that the API doesn't yet expose
 * as a first-class endpoint.
 */
export interface HistoryState {
  history: {
    physicsWeightByAsset: Record<string, number[]>;
  };
  pushPhysicsWeight: (assetId: string, weight: number) => void;
}

const MAX_POINTS = 30;

export const createHistorySlice: StateCreator<
  AppState,
  [['zustand/devtools', never], ['zustand/persist', unknown]],
  [],
  HistoryState
> = (set) => ({
  history: { physicsWeightByAsset: {} },
  pushPhysicsWeight: (assetId, weight) =>
    set(
      (s) => {
        const prev = s.history.physicsWeightByAsset[assetId] ?? [];
        const last = prev[prev.length - 1];
        // De-dupe consecutive identical values so re-fetching the same
        // snapshot doesn't inflate the buffer.
        if (last === weight) return s;
        const next = [...prev, weight].slice(-MAX_POINTS);
        return {
          history: {
            ...s.history,
            physicsWeightByAsset: {
              ...s.history.physicsWeightByAsset,
              [assetId]: next,
            },
          },
        };
      },
      false,
      'history/pushPhysicsWeight',
    ),
});
