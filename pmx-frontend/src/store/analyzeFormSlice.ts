import type { StateCreator } from 'zustand';

import type { AppState } from './index';
import type { AssetType, SensorRow } from '@/types/domain';

export type DataSource = 'synthetic' | 'upload' | 'manual';
export type Scenario = 'force_anomaly' | 'force_normal' | 'as_generated';

export interface AnalyzeFormState {
  form: {
    assetId: string;
    assetType: AssetType;
    material: string;
    component: string;
    dataSource: DataSource;
    synthetic: { nSamples: number; scenario: Scenario };
    upload: { records: SensorRow[] | null; error: string | null };
    manual: { records: SensorRow[] };
  };
  updateForm: (patch: Partial<AnalyzeFormState['form']>) => void;
  resetForm: () => void;
}

const INITIAL_FORM: AnalyzeFormState['form'] = {
  assetId: 'ENGINE-001',
  assetType: 'aircraft_engine',
  material: 'Inconel718',
  component: 'turbine_blade',
  dataSource: 'synthetic',
  synthetic: { nSamples: 100, scenario: 'force_anomaly' },
  upload: { records: null, error: null },
  manual: { records: [] },
};

export const createAnalyzeFormSlice: StateCreator<
  AppState,
  [['zustand/devtools', never], ['zustand/persist', unknown]],
  [],
  AnalyzeFormState
> = (set) => ({
  form: INITIAL_FORM,
  updateForm: (patch) =>
    set((s) => ({ form: { ...s.form, ...patch } }), false, 'form/update'),
  resetForm: () => set({ form: INITIAL_FORM }, false, 'form/reset'),
});
