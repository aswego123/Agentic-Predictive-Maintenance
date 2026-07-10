import { defaultsFor, SENSOR_COLUMNS } from './constants';
import type { AssetType, SensorRow } from '@/types/domain';

/**
 * Client-side port of `_template_dataframe` from the Streamlit dashboard.
 * Generates a starter batch of sensor rows for the manual editor / CSV
 * template download.
 */
export function generateSensorTemplate(
  assetId: string,
  assetType: AssetType,
  nRows = 5,
): SensorRow[] {
  const defaults = defaultsFor(assetType);
  const start = new Date('2026-01-01T00:00:00Z').getTime();
  const rows: SensorRow[] = [];
  for (let i = 0; i < nRows; i++) {
    const ts = new Date(start + i * 3600_000).toISOString();
    const row: SensorRow = {
      asset_id: assetId,
      asset_type: assetType,
      timestamp: ts,
      operational_cycles: i + 1,
      ...defaults,
    };
    rows.push(row);
  }
  return rows;
}

/**
 * Ensure incoming records have every canonical sensor column populated,
 * mirroring the Streamlit upload path's "fill missing columns with
 * defaults" behaviour.
 */
export function fillMissingSensorColumns(
  records: SensorRow[],
  assetId: string,
  assetType: AssetType,
): SensorRow[] {
  const defaults = defaultsFor(assetType);
  return records.map((r, i) => {
    const out: SensorRow = { ...r };
    for (const [k, v] of Object.entries(defaults)) {
      if (out[k] === undefined || out[k] === null || out[k] === '') {
        out[k] = v;
      }
    }
    if (out.asset_id === undefined || out.asset_id === null || out.asset_id === '') {
      out.asset_id = assetId;
    }
    if (out.asset_type === undefined || out.asset_type === null || out.asset_type === '') {
      out.asset_type = assetType;
    }
    if (out.operational_cycles === undefined || out.operational_cycles === null || out.operational_cycles === '') {
      out.operational_cycles = i + 1;
    }
    if (out.timestamp === undefined || out.timestamp === null || out.timestamp === '') {
      out.timestamp = new Date(Date.UTC(2026, 0, 1) + i * 3600_000).toISOString();
    }
    return out;
  });
}

export function sensorRowsToCsv(rows: SensorRow[]): string {
  const cols = SENSOR_COLUMNS;
  const header = cols.join(',');
  const body = rows
    .map((r) => cols.map((c) => (r[c] === undefined || r[c] === null ? '' : String(r[c]))).join(','))
    .join('\n');
  return `${header}\n${body}\n`;
}

export function downloadCsv(filename: string, csv: string) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
