import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { JsonViewer } from '@/components/common/JsonViewer';

export function ActionPanel({
  action,
  workOrder,
}: {
  action: Record<string, any> | null;
  workOrder: Record<string, any> | null;
}) {
  if (!action && !workOrder) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Action + work order</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {action ? (
          <div>
            <div className="mb-2 text-xs font-semibold uppercase text-muted-foreground">Action</div>
            <JsonViewer data={action} />
          </div>
        ) : null}
        {workOrder ? (
          <div>
            <div className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
              MRO/ERP work order (stub)
            </div>
            <JsonViewer data={workOrder} />
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
