import { useCallback, useRef, useState } from 'react';
import { toast } from 'sonner';
import { Upload, FileJson } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { EmptyState } from '@/components/common/EmptyState';
import { downloadCsv, generateSensorTemplate, sensorRowsToCsv } from '@/lib/sensorTemplate';
import { parseSensorFile } from '@/lib/csvParser';
import { SENSOR_COLUMNS } from '@/lib/constants';
import type { AssetType, SensorRow } from '@/types/domain';

interface Props {
  assetId: string;
  assetType: AssetType;
  records: SensorRow[] | null;
  error: string | null;
  onRecords: (rows: SensorRow[] | null, error: string | null) => void;
}

export function FileUploadPanel({ assetId, assetType, records, error, onRecords }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);

  const handleFile = useCallback(
    async (file: File) => {
      setBusy(true);
      try {
        const rows = await parseSensorFile(file);
        onRecords(rows, null);
        toast.success(`Parsed ${rows.length} rows from ${file.name}`);
      } catch (exc) {
        const msg = exc instanceof Error ? exc.message : String(exc);
        onRecords(null, msg);
        toast.error(`Parse failed: ${msg}`);
      } finally {
        setBusy(false);
      }
    },
    [onRecords],
  );

  const downloadTemplate = () => {
    const rows = generateSensorTemplate(assetId, assetType, 5);
    downloadCsv('sensor_template.csv', sensorRowsToCsv(rows));
  };

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Upload sensor data with the canonical columns. Extra numeric columns are
        kept; missing ones are auto-filled with asset-family defaults on the
        server.
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <Button type="button" variant="outline" size="sm" onClick={downloadTemplate}>
          <FileJson className="mr-1 h-4 w-4" />
          Download template CSV
        </Button>
        <Button
          type="button"
          size="sm"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
        >
          <Upload className="mr-1 h-4 w-4" />
          {busy ? 'Parsing…' : 'Choose file (CSV / JSON)'}
        </Button>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.json,application/json,text/csv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
            e.target.value = '';
          }}
        />
      </div>

      <div className="text-[11px] text-muted-foreground">
        Canonical columns:{' '}
        <code className="rounded bg-muted px-1">{SENSOR_COLUMNS.join(', ')}</code>
      </div>

      {error ? (
        <Alert variant="destructive">
          <AlertTitle>Upload failed</AlertTitle>
          <AlertDescription className="whitespace-pre-wrap">{error}</AlertDescription>
        </Alert>
      ) : records && records.length > 0 ? (
        <div className="rounded-md border bg-muted/30 p-3 text-xs">
          Parsed <strong>{records.length}</strong> rows across{' '}
          <strong>{Object.keys(records[0]).length}</strong> columns.
        </div>
      ) : (
        <EmptyState title="No file selected" description="Choose a CSV or JSON file to preview." />
      )}
    </div>
  );
}
