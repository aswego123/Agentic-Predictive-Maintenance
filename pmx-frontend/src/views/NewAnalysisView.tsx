import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { PlayCircle, Info } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Separator } from '@/components/ui/separator';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { FileUploadPanel } from '@/components/forms/FileUploadPanel';
import { SensorTableEditor } from '@/components/forms/SensorTableEditor';
import { useAnalyze } from '@/hooks/useAnalyze';
import { useAppStore } from '@/store';
import { ASSET_TYPES } from '@/lib/constants';
import type { AnalyzeRequest, AssetType, SensorRow } from '@/types/domain';

/**
 * Extract asset metadata directly from the uploaded / edited sensor rows.
 * The CSV is expected to carry `asset_id` and `asset_type` on every row;
 * we read the first non-empty occurrence. `material_name` / `material`
 * is optional — if present we forward it, otherwise the backend infers
 * a sensible default from asset_type.
 */
function deriveAssetFromRows(rows: SensorRow[] | null | undefined): {
  assetId: string | null;
  assetType: AssetType | null;
  material: string | null;
} {
  if (!rows || rows.length === 0) return { assetId: null, assetType: null, material: null };
  const firstAssetId = rows.find((r) => r.asset_id !== undefined && r.asset_id !== null && r.asset_id !== '')?.asset_id;
  const firstAssetType = rows.find((r) => r.asset_type !== undefined && r.asset_type !== null && r.asset_type !== '')?.asset_type;
  const firstMaterial =
    rows.find((r) => r.material_name !== undefined && r.material_name !== null && r.material_name !== '')?.material_name ??
    rows.find((r) => r.material !== undefined && r.material !== null && r.material !== '')?.material;
  const assetType = typeof firstAssetType === 'string' && (ASSET_TYPES as readonly string[]).includes(firstAssetType)
    ? (firstAssetType as AssetType)
    : null;
  return {
    assetId: firstAssetId ? String(firstAssetId) : null,
    assetType,
    material: firstMaterial ? String(firstMaterial) : null,
  };
}

export function NewAnalysisView() {
  const navigate = useNavigate();
  const form = useAppStore((s) => s.form);
  const updateForm = useAppStore((s) => s.updateForm);
  const setPolling = useAppStore((s) => s.setPolling);
  const setLastCycleId = useAppStore((s) => s.setLastCycleId);
  const analyze = useAnalyze();
  const [submitting, setSubmitting] = useState(false);

  // For CSV/manual modes we derive asset metadata from the rows themselves;
  // for synthetic we fall back to the form draft (which still holds sane
  // defaults so we can keep synthetic generation working).
  const derived = useMemo(() => {
    if (form.dataSource === 'upload') return deriveAssetFromRows(form.upload.records);
    if (form.dataSource === 'manual') return deriveAssetFromRows(form.manual.records);
    return { assetId: form.assetId || null, assetType: form.assetType || null, material: null };
  }, [form]);

  const uploadReady = form.upload.records && form.upload.records.length > 0 && !form.upload.error;
  const manualReady = form.manual.records.length > 0;
  const canSubmit =
    !!derived.assetId &&
    !!derived.assetType &&
    (form.dataSource === 'synthetic'
      ? true
      : form.dataSource === 'upload'
        ? !!uploadReady
        : !!manualReady);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit || submitting || !derived.assetId || !derived.assetType) return;
    setSubmitting(true);

    // Generate the cycle_id client-side so we can navigate to the
    // detail page *immediately* — the user then watches the pipeline
    // graph light up node-by-node instead of staring at a form spinner
    // for 60-150 seconds while POST /analyze runs.
    const cycleId = `cycle-${crypto.randomUUID().replace(/-/g, '').slice(0, 12)}`;

    const payload: AnalyzeRequest = {
      asset_id: derived.assetId,
      asset_type: derived.assetType,
      cycle_id: cycleId,
      // material_name is forwarded only if we found it in the CSV. When
      // omitted the backend infers the correct default per asset_type
      // (aircraft_wing -> Al7075-T6, etc). See api/main.py::_resolve_material.
      ...(derived.material ? { material_name: derived.material } : {}),
      // component is optional server-side; leave it off so the backend
      // uses its own defaults. If the CSV later carries it we can wire
      // it here.
    };

    if (form.dataSource === 'synthetic') {
      payload.generate_synthetic = true;
      payload.synthetic_n_samples = form.synthetic.nSamples;
      payload.force_anomaly = form.synthetic.scenario === 'force_anomaly';
      payload.force_normal = form.synthetic.scenario === 'force_normal';
    } else {
      const records =
        form.dataSource === 'upload' ? form.upload.records ?? [] : form.manual.records;
      payload.sensor_batch = { records, columns: Object.keys(records[0] ?? {}) };
    }

    // Fire-and-forget: kick off the POST in the background so we can
    // navigate immediately. The response will land eventually and prime
    // the react-query cache for the cycle-detail page, but the detail
    // page's 750ms poll will show live progress in the meantime.
    setLastCycleId(cycleId);
    setPolling(cycleId, true);
    analyze.mutate(payload);
    navigate(`/cycles/${cycleId}`);
    setSubmitting(false);
  };

  return (
    <div className="mx-auto max-w-3xl">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">New analysis</h1>
        <p className="text-sm text-muted-foreground">
          Upload sensor data — asset metadata is read directly from the file.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Sensor data</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <RadioGroup
              value={form.dataSource}
              onValueChange={(v) => updateForm({ dataSource: v as typeof form.dataSource })}
              className="grid grid-cols-1 gap-2 md:grid-cols-3"
            >
              <SourceOption id="upload" label="Upload file (CSV / JSON)" />
              <SourceOption id="manual" label="Manual entry (table editor)" />
              <SourceOption id="synthetic" label="Generate synthetic (auto)" />
            </RadioGroup>

            <Separator />

            {form.dataSource === 'upload' ? (
              <FileUploadPanel
                assetId={form.assetId || 'ENGINE-001'}
                assetType={form.assetType || 'aircraft_engine'}
                records={form.upload.records}
                error={form.upload.error}
                onRecords={(records, error) =>
                  updateForm({ upload: { records, error } })
                }
              />
            ) : form.dataSource === 'manual' ? (
              <Accordion type="single" collapsible defaultValue="editor">
                <AccordionItem value="editor">
                  <AccordionTrigger>Sensor table editor</AccordionTrigger>
                  <AccordionContent>
                    <SensorTableEditor
                      assetId={form.assetId || 'ENGINE-001'}
                      assetType={form.assetType || 'aircraft_engine'}
                      records={form.manual.records}
                      onChange={(records) => updateForm({ manual: { records } })}
                    />
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            ) : (
              <SyntheticControls />
            )}
          </CardContent>
        </Card>

        {form.dataSource !== 'synthetic' ? (
          <DerivedAssetSummary
            assetId={derived.assetId}
            assetType={derived.assetType}
            material={derived.material}
            hasRows={form.dataSource === 'upload' ? !!uploadReady : manualReady}
          />
        ) : null}

        <div className="flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            Long-running Gemini + Judge ReAct loop — typical duration 60–150 s.
          </div>
          <Button type="submit" disabled={!canSubmit || submitting} size="lg">
            {submitting ? (
              <>
                <LoadingSpinner className="text-primary-foreground" /> Running graph…
              </>
            ) : (
              <>
                <PlayCircle className="mr-1 h-4 w-4" /> Analyze
              </>
            )}
          </Button>
        </div>
      </form>
    </div>
  );
}

function SourceOption({ id, label }: { id: string; label: string }) {
  return (
    <label
      htmlFor={`ds-${id}`}
      className="flex cursor-pointer items-center gap-2 rounded-md border p-3 hover:bg-accent has-[[data-state=checked]]:border-primary has-[[data-state=checked]]:bg-primary/5"
    >
      <RadioGroupItem id={`ds-${id}`} value={id} />
      <span className="text-sm">{label}</span>
    </label>
  );
}

function SyntheticControls() {
  const synth = useAppStore((s) => s.form.synthetic);
  const assetId = useAppStore((s) => s.form.assetId);
  const assetType = useAppStore((s) => s.form.assetType);
  const updateForm = useAppStore((s) => s.updateForm);

  return (
    <div className="space-y-4">
      <Alert variant="info">
        <AlertTitle className="flex items-center gap-2"><Info className="h-4 w-4" /> Synthetic mode</AlertTitle>
        <AlertDescription className="text-xs">
          No CSV upload — the backend generates a sensor batch. Pick the asset
          identity and scenario below.
        </AlertDescription>
      </Alert>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="synth-asset-id">Asset ID</Label>
          <Input
            id="synth-asset-id"
            value={assetId}
            onChange={(e) => updateForm({ assetId: e.target.value })}
            placeholder="ENGINE-001"
          />
        </div>
        <div className="space-y-1.5">
          <Label>Asset type</Label>
          <Select
            value={assetType}
            onValueChange={(v) => updateForm({ assetType: v as AssetType })}
          >
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              {ASSET_TYPES.map((t) => (
                <SelectItem key={t} value={t}>{t}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="n-samples">Synthetic samples: {synth.nSamples}</Label>
          <Input
            id="n-samples"
            type="range"
            min={20}
            max={300}
            step={10}
            value={synth.nSamples}
            onChange={(e) =>
              updateForm({ synthetic: { ...synth, nSamples: Number(e.target.value) } })
            }
          />
        </div>
        <div className="space-y-1.5">
          <Label>Scenario</Label>
          <Select
            value={synth.scenario}
            onValueChange={(v) =>
              updateForm({ synthetic: { ...synth, scenario: v as typeof synth.scenario } })
            }
          >
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="force_anomaly">Force anomaly</SelectItem>
              <SelectItem value="force_normal">Force normal (short-circuit)</SelectItem>
              <SelectItem value="as_generated">As generated</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
}

function DerivedAssetSummary({
  assetId,
  assetType,
  material,
  hasRows,
}: {
  assetId: string | null;
  assetType: AssetType | null;
  material: string | null;
  hasRows: boolean;
}) {
  if (!hasRows) return null;

  const missing = !assetId || !assetType;

  return (
    <Alert variant={missing ? 'destructive' : 'info'}>
      <AlertTitle className="flex items-center gap-2">
        <Info className="h-4 w-4" />
        {missing ? 'Missing asset metadata' : 'Asset detected from file'}
      </AlertTitle>
      <AlertDescription className="text-xs">
        {missing ? (
          <>
            The uploaded rows must include <code>asset_id</code> and{' '}
            <code>asset_type</code> columns. Please re-upload a file that
            carries this metadata (see the template CSV in the upload panel).
          </>
        ) : (
          <span className="inline-flex flex-wrap items-center gap-x-4 gap-y-1">
            <span>
              <span className="text-muted-foreground">asset_id</span>{' '}
              <code className="rounded bg-muted px-1">{assetId}</code>
            </span>
            <span>
              <span className="text-muted-foreground">asset_type</span>{' '}
              <code className="rounded bg-muted px-1">{assetType}</code>
            </span>
            <span>
              <span className="text-muted-foreground">material</span>{' '}
              {material ? (
                <code className="rounded bg-muted px-1">{material}</code>
              ) : (
                <span className="italic text-muted-foreground/80">inferred from asset_type on server</span>
              )}
            </span>
          </span>
        )}
      </AlertDescription>
    </Alert>
  );
}
