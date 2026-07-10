import { useEffect, useMemo, useState } from 'react';
import { Plus, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { generateSensorTemplate } from '@/lib/sensorTemplate';
import { SENSOR_COLUMNS } from '@/lib/constants';
import type { AssetType, SensorRow } from '@/types/domain';

interface Props {
  assetId: string;
  assetType: AssetType;
  records: SensorRow[];
  onChange: (rows: SensorRow[]) => void;
}

const NUMERIC_COLUMNS = new Set([
  'operational_cycles',
  'vibration',
  'temperature',
  'pressure',
  'load_factor',
  'speed',
  'acoustic_emission',
  'oil_pressure',
  'oil_temperature',
]);

/**
 * Compact editable table — port of Streamlit's `st.data_editor`.
 * Users can add / remove rows and edit any cell inline.
 */
export function SensorTableEditor({ assetId, assetType, records, onChange }: Props) {
  const [rows, setRows] = useState<SensorRow[]>(
    records.length > 0 ? records : generateSensorTemplate(assetId, assetType, 5),
  );

  // Reset the editor when the asset context changes and we don't yet have
  // user-authored rows for it.
  useEffect(() => {
    if (records.length === 0) {
      const fresh = generateSensorTemplate(assetId, assetType, 5);
      setRows(fresh);
      onChange(fresh);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [assetId, assetType]);

  const columns = useMemo(() => {
    const set = new Set<string>(SENSOR_COLUMNS);
    for (const r of rows) for (const k of Object.keys(r)) set.add(k);
    return Array.from(set);
  }, [rows]);

  const update = (next: SensorRow[]) => {
    setRows(next);
    onChange(next);
  };

  const updateCell = (rowIdx: number, col: string, raw: string) => {
    const next = rows.map((r, i) => (i === rowIdx ? { ...r } : r));
    const isNumeric = NUMERIC_COLUMNS.has(col);
    if (raw === '') {
      next[rowIdx][col] = null;
    } else if (isNumeric) {
      const parsed = Number(raw);
      next[rowIdx][col] = Number.isNaN(parsed) ? raw : parsed;
    } else {
      next[rowIdx][col] = raw;
    }
    update(next);
  };

  const addRow = () => {
    const idx = rows.length;
    const template = generateSensorTemplate(assetId, assetType, 1)[0];
    template.operational_cycles = idx + 1;
    template.timestamp = new Date(Date.UTC(2026, 0, 1) + idx * 3600_000).toISOString();
    update([...rows, template]);
  };

  const removeRow = (rowIdx: number) => {
    update(rows.filter((_, i) => i !== rowIdx));
  };

  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground">
        Edit values inline. Numeric columns coerce automatically. Any column
        name is allowed — the anomaly engine will pick up extra numeric channels.
      </p>

      <div className="max-h-96 overflow-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              {columns.map((c) => (
                <TableHead key={c} className="whitespace-nowrap">{c}</TableHead>
              ))}
              <TableHead className="w-10"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r, i) => (
              <TableRow key={i}>
                {columns.map((c) => (
                  <TableCell key={c} className="p-1">
                    <Input
                      value={r[c] === null || r[c] === undefined ? '' : String(r[c])}
                      onChange={(e) => updateCell(i, c, e.target.value)}
                      className="h-7 min-w-[6rem] px-2 text-xs"
                    />
                  </TableCell>
                ))}
                <TableCell className="p-1">
                  <Button type="button" variant="ghost" size="icon" className="h-7 w-7" onClick={() => removeRow(i)}>
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <Button type="button" variant="outline" size="sm" onClick={addRow}>
        <Plus className="mr-1 h-4 w-4" />
        Add row
      </Button>
    </div>
  );
}
