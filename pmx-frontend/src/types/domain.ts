/**
 * Domain types — hand-written aliases for the FastAPI response shapes.
 *
 * These are intentionally loose (Record<string, any> for deeply nested
 * agent payloads) until we run `npm run gen:api-types` to produce
 * a strict OpenAPI-derived counterpart in `api.generated.ts`.
 *
 * Sources: predictive_maintenance_agentic/api/main.py::_snapshot and
 * predictive_maintenance_agentic/agents/*.
 */

export type AssetType =
  | 'aircraft_engine'
  | 'aircraft_landing_gear'
  | 'aircraft_brake'
  | 'aircraft_wing'
  | 'aircraft_fuselage'
  | 'train_bogie'
  | 'train_brake'
  | 'train_wheel'
  | 'train_traction_motor';

export type CycleStatus =
  | 'running'
  | 'normal_end'
  | 'action_taken'
  | 'unresolved_divergence'
  | 'unknown';

export type SensorRow = Record<string, string | number | null>;

export interface SensorBatch {
  records: SensorRow[];
  columns?: string[];
}

export interface AnalyzeRequest {
  asset_id: string;
  asset_type: AssetType;
  material_name?: string;
  component?: string | null;
  cycle_id?: string;
  sensor_batch?: SensorBatch;
  generate_synthetic?: boolean;
  synthetic_n_samples?: number;
  force_anomaly?: boolean;
  force_normal?: boolean;
}

export interface EngineerApproveRequest {
  cycle_id: string;
  approved: boolean;
  engineer_id: string;
  notes?: string;
  extra?: Record<string, unknown> | null;
}

/**
 * Response snapshot from POST /analyze, GET /cycles/:id, and
 * POST /engineer/approve. Fields mirror api/main.py::_snapshot.
 */
export interface CycleSnapshot {
  cycle_id: string;
  status: CycleStatus | string;
  is_anomalous: boolean;
  asset_id: string | null;
  asset_type: string | null;
  component: string | null;
  anomaly_result: Record<string, any> | null;
  stress_features: Record<string, any> | null;
  physics_prediction: Record<string, any> | null;
  ml_correction: Record<string, any> | null;
  judge_verdict: Record<string, any> | null;
  calibration_result: Record<string, any> | null;
  engineer_decision: Record<string, any> | null;
  negotiation_round: number;
  negotiation_history: Array<Record<string, any>>;
  dialogue_history: Array<Record<string, any>>;
  fetched_features: Record<string, any>;
  resimulation_round: number;
  action: Record<string, any> | null;
  critic_review: Record<string, any> | null;
  work_order: Record<string, any> | null;
  trace: Array<{
    ts: number;
    node: string;
    note: string;
    data: Record<string, any>;
  }>;
  fleet_memory_refs: string[];
  next: string[];
  awaiting_engineer_approval: boolean;
  simulation_adapter: string;
  simulation_is_synthetic: boolean;
}

export interface AssetSummary {
  asset_id: string;
  counts?: Record<string, number>;
  last_entry_at?: number | null;
}

export interface FleetStatus {
  known_assets: string[];
  asset_summaries: AssetSummary[];
  known_cycles: Array<{
    cycle_id: string;
    asset_id?: string;
    created_at?: number;
  }>;
  simulation_adapter: string;
  simulation_is_synthetic: boolean;
}

export interface HealthResponse {
  status: string;
  simulation_adapter: string;
}
