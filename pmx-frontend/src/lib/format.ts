/**
 * Number / date / percent formatters used across the dashboard panels.
 * Kept in one place so display conventions stay consistent.
 */

export function fmtNumber(value: unknown, digits = 2): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = typeof value === 'number' ? value : Number(value);
  if (Number.isNaN(n)) return String(value);
  return n.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtInt(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = typeof value === 'number' ? value : Number(value);
  if (Number.isNaN(n)) return String(value);
  return Math.round(n).toLocaleString();
}

export function fmtPercent(value: unknown, digits = 0): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = typeof value === 'number' ? value : Number(value);
  if (Number.isNaN(n)) return String(value);
  return `${(n * 100).toFixed(digits)}%`;
}

export function fmtHours(value: unknown, digits = 1): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = typeof value === 'number' ? value : Number(value);
  if (Number.isNaN(n)) return String(value);
  return `${n.toFixed(digits)} h`;
}

export function fmtBool(value: unknown): string {
  if (value === null || value === undefined) return '—';
  return value ? 'yes' : 'no';
}

/**
 * Accepts unix seconds, unix ms, or an ISO string; returns a locale
 * datetime string. The backend mixes formats (float ts, isoformat
 * strings), so this handles all three.
 */
export function fmtTimestamp(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—';
  let d: Date;
  if (typeof value === 'number') {
    d = new Date(value < 1e12 ? value * 1000 : value);
  } else if (typeof value === 'string') {
    // Try ISO first; if that fails and it's a numeric string, try that.
    d = new Date(value);
    if (Number.isNaN(d.getTime())) {
      const n = Number(value);
      if (!Number.isNaN(n)) d = new Date(n < 1e12 ? n * 1000 : n);
    }
  } else {
    return String(value);
  }
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString();
}
