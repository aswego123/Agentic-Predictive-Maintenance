import Papa from 'papaparse';
import type { SensorRow } from '@/types/domain';

/**
 * Parse an uploaded sensor file. Supports:
 *   - CSV (with header row)
 *   - JSON: either a bare array of records or `{records: [...]}`.
 *
 * Numeric columns are coerced to number where the value parses cleanly;
 * everything else stays as string (matches Streamlit upload semantics).
 */
export async function parseSensorFile(file: File): Promise<SensorRow[]> {
  const name = file.name.toLowerCase();
  const text = await file.text();

  if (name.endsWith('.json')) {
    const parsed = JSON.parse(text);
    const list: unknown[] = Array.isArray(parsed)
      ? parsed
      : parsed && typeof parsed === 'object' && Array.isArray((parsed as any).records)
        ? (parsed as any).records
        : [];
    return list
      .filter((r): r is Record<string, unknown> => !!r && typeof r === 'object')
      .map(coerceRow);
  }

  return new Promise<SensorRow[]>((resolve, reject) => {
    Papa.parse<Record<string, unknown>>(text, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: (result) => resolve(result.data.map(coerceRow)),
      error: (err: Error) => reject(err),
    });
  });
}

function coerceRow(row: Record<string, unknown>): SensorRow {
  const out: SensorRow = {};
  for (const [k, v] of Object.entries(row)) {
    if (v === null || v === undefined || v === '') {
      out[k] = null;
    } else if (typeof v === 'number' || typeof v === 'string') {
      out[k] = v;
    } else {
      out[k] = String(v);
    }
  }
  return out;
}
