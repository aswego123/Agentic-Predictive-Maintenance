import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import type { CycleSnapshot } from '@/types/domain';

export function TracePanel({ trace }: { trace: CycleSnapshot['trace'] }) {
  if (!trace || trace.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Node-by-node trace</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="max-h-96 overflow-y-auto rounded-md border">
          <Table>
            <TableHeader className="sticky top-0 bg-card">
              <TableRow>
                <TableHead className="w-10">#</TableHead>
                <TableHead>Node</TableHead>
                <TableHead>Note</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trace.map((t, i) => (
                <TableRow key={i}>
                  <TableCell className="text-xs text-muted-foreground">{i + 1}</TableCell>
                  <TableCell className="font-mono text-xs">{t.node}</TableCell>
                  <TableCell className="text-xs">{t.note}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
