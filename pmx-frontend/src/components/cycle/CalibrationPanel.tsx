import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { MetricCard } from '@/components/common/MetricCard';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { JsonViewer } from '@/components/common/JsonViewer';
import { fmtNumber } from '@/lib/format';

export function CalibrationPanel({ calib }: { calib: Record<string, any> | null }) {
  if (!calib) return null;
  const suggestion = calib.engineer_suggestion_llm || calib.engineer_suggestion;
  const oldGeom = calib.geometry_factor_old;
  const newGeom = calib.geometry_factor_new;
  const geomDelta =
    typeof oldGeom === 'number' && typeof newGeom === 'number'
      ? `was ${oldGeom.toFixed(2)}`
      : undefined;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Calibration suggestion</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <MetricCard label="Prior stress" value={`${fmtNumber(calib.prior_stress_mpa, 1)} MPa`} />
          <MetricCard label="Posterior stress" value={`${fmtNumber(calib.posterior_stress_mpa, 1)} MPa`} />
          <MetricCard label="Geometry factor" value={fmtNumber(newGeom, 2)} delta={geomDelta} />
        </div>

        {suggestion ? (
          <div className="rounded-md border bg-muted/30 p-3 text-sm">
            {String(suggestion)}
          </div>
        ) : null}

        <Accordion type="single" collapsible>
          <AccordionItem value="raw" className="border-0">
            <AccordionTrigger className="text-xs">Raw calibration payload</AccordionTrigger>
            <AccordionContent>
              <JsonViewer data={calib} />
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}
