import { RefreshCw } from 'lucide-react';

import { useFleetStatus } from '@/hooks/useFleetStatus';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { MetricCard } from '@/components/common/MetricCard';
import { EmptyState } from '@/components/common/EmptyState';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { fmtTimestamp } from '@/lib/format';

export function FleetView() {
  const fleet = useFleetStatus();
  const summaries = fleet.data?.asset_summaries ?? [];
  const cycles = fleet.data?.known_cycles ?? [];

  // Compute the union of all count-kinds across assets for stable columns.
  const kinds = Array.from(
    new Set(summaries.flatMap((s) => Object.keys(s.counts ?? {}))),
  ).sort();

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Fleet</h1>
          <p className="text-sm text-muted-foreground">
            Per-asset fleet-memory summaries. Simulation adapter:{' '}
            <span className="font-mono">{fleet.data?.simulation_adapter ?? '—'}</span>.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => fleet.refetch()} disabled={fleet.isFetching}>
          <RefreshCw className={`mr-1 h-4 w-4 ${fleet.isFetching ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </header>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricCard label="Known assets" value={fleet.isLoading ? <LoadingSpinner /> : fleet.data?.known_assets.length ?? 0} />
        <MetricCard label="Known cycles" value={fleet.isLoading ? <LoadingSpinner /> : cycles.length} />
        <MetricCard label="Memory kinds tracked" value={kinds.length} />
        <MetricCard label="Simulation" value={fleet.data?.simulation_is_synthetic ? 'synthetic' : 'real'} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Per-asset fleet-memory counts</CardTitle>
        </CardHeader>
        <CardContent>
          {summaries.length === 0 ? (
            <EmptyState title="No asset summaries" description="Run a few cycles to populate fleet memory." />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Asset</TableHead>
                  {kinds.map((k) => (
                    <TableHead key={k} className="text-right">{k}</TableHead>
                  ))}
                  <TableHead className="text-right">Last entry</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {summaries.map((s) => (
                  <TableRow key={s.asset_id}>
                    <TableCell className="font-mono text-xs">{s.asset_id}</TableCell>
                    {kinds.map((k) => (
                      <TableCell key={k} className="text-right tabular-nums">
                        {s.counts?.[k] ?? 0}
                      </TableCell>
                    ))}
                    <TableCell className="text-right text-xs text-muted-foreground">
                      {fmtTimestamp(s.last_entry_at)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Known cycles</CardTitle>
        </CardHeader>
        <CardContent>
          {cycles.length === 0 ? (
            <EmptyState title="No cycles yet" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Cycle ID</TableHead>
                  <TableHead>Asset</TableHead>
                  <TableHead>Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {cycles.map((c) => (
                  <TableRow key={c.cycle_id}>
                    <TableCell className="font-mono text-xs">{c.cycle_id}</TableCell>
                    <TableCell className="font-mono text-xs">{c.asset_id ?? '—'}</TableCell>
                    <TableCell className="text-xs text-muted-foreground">{fmtTimestamp(c.created_at)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
