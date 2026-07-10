/**
 * Constants ported from predictive_maintenance_agentic/ui/dashboard.py.
 * Keep these in one place so the form dropdowns and sensor-template
 * generators stay in sync.
 */
import type { AssetType } from '@/types/domain';

export const ASSET_TYPES: AssetType[] = [
  'aircraft_engine',
  'aircraft_landing_gear',
  'aircraft_brake',
  'aircraft_wing',
  'aircraft_fuselage',
  'train_bogie',
  'train_brake',
  'train_wheel',
  'train_traction_motor',
];

export const MATERIALS = [
  'Al7075-T6',
  'Al2024-T3',
  'Steel4340',
  'AISI4130',
  'Ti-6Al-4V',
  'Inconel718',
  'CastIron',
] as const;

export const SENSOR_COLUMNS = [
  'asset_id',
  'asset_type',
  'timestamp',
  'operational_cycles',
  'vibration',
  'temperature',
  'pressure',
  'load_factor',
  'speed',
  'acoustic_emission',
  'oil_pressure',
  'oil_temperature',
] as const;

export const AIRCRAFT_DEFAULTS = {
  vibration: 0.4,
  temperature: 950.0,
  pressure: 100.0,
  load_factor: 0.7,
  speed: 10000.0,
  acoustic_emission: 28.0,
  oil_pressure: 50.0,
  oil_temperature: 100.0,
} as const;

export const TRAIN_DEFAULTS = {
  vibration: 1.2,
  temperature: 50.0,
  pressure: 200.0,
  load_factor: 0.6,
  speed: 120.0,
  acoustic_emission: 20.0,
  oil_pressure: 150.0,
  oil_temperature: 75.0,
} as const;

export function defaultsFor(assetType: AssetType) {
  return assetType.startsWith('train_') ? TRAIN_DEFAULTS : AIRCRAFT_DEFAULTS;
}
