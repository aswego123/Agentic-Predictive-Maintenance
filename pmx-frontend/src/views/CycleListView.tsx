import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { RefreshCw, PlayCircle, Search, Boxes } from 'lucide-react';

import { useFleetStatus } from '@/hooks/useFleetStatus';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { EmptyState } from '@/components/common/EmptyState';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { CopyableId } from '@/components/common/CopyableId';
import { fmtTimestamp } from '@/lib/format';

export function CycleListView() {
  const fleet = useFleetStatus();
  const cyclesRaw = fleet.data?.known_cycles ?? [];
  const knownAssets = fleet.data?.known_assets ?? [];

  const [query, setQuery] = useState('');
  const [assetFilter, setAssetFilter] = useState<string>('__all__');

  const cycles = useMemo(() => {
    const q = query.trim().toLowerCase();
    return [...cyclesRaw]
      .filter((c) => (assetFilter === '__all__' ? true : c.asset_id === assetFilter))
      .filter((c) => (!q ? true : `${c.cycle_id} ${c.asset_id ?? ''}`.toLowerCase().includes(q)))
      .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0));
  }, [cyclesRaw, query, assetFilter]);

  return (
    <div className="mx-auto max-w-6xl space-y-4 animate-fade-in-up">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Cycles</h1>
          <p className="text-sm text-muted-foreground">
            All cycles known to this process. Click any row to open its full
            trace and verdict.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => fleet.refetch()}
            disabled={fleet.isFetching}
          >
            <RefreshCw className={`mr-1 h-4 w-4 ${fleet.isFetching ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button asChild size="sm">
            <Link to="/analyze/new">
              <PlayCircle className="mr-1 h-4 w-4" />
              New analysis
            </Link>
          </Button>
        </div>
      </header>

      <Card>
        <CardHeader className="flex flex-col gap-3 space-y-0 md:flex-row md:items-center md:justify-between">
          <CardTitle className="text-base">
            {cycles.length} of {cyclesRaw.length} cycle{cyclesRaw.length === 1 ? '' : 's'}
          </CardTitle>
          <div className="flex flex-1 flex-col gap-2 md:max-w-md md:flex-row">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search cycle id or asset…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="pl-8"
              />
            </div>
            <Select value={assetFilter} onValueChange={setAssetFilter}>
              <SelectTrigger className="w-full md:w-48">
                <SelectValue placeholder="All assets" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All assets</SelectItem>
                {knownAssets.map((a) => (
                  <SelectItem key={a} value={a}>{a}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          {fleet.isLoading ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <LoadingSpinner /> Loading cycles…
            </div>
          ) : cyclesRaw.length === 0 ? (
            <EmptyState
              icon={<Boxes className="h-5 w-5" />}
              title="No cycles yet"
              description="Kick off your first analysis to populate this list."
              action={
                <Button asChild size="sm">
                  <Link to="/analyze/new">Start a cycle</Link>
                </Button>
              }
            />
          ) : cycles.length === 0 ? (
            <EmptyState
              icon={<Search className="h-5 w-5" />}
              title="No matches"
              description="Adjust the search or filter to see more cycles."
              action={
                <Button variant="outline" size="sm" onClick={() => { setQuery(''); setAssetFilter('__all__'); }}>
                  Clear filters
                </Button>
              }
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Cycle ID</TableHead>
                  <TableHead>Asset</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="text-right"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {cycles.map((c) => (
                  <TableRow key={c.cycle_id}>
                    <TableCell className="p-2"><CopyableId value={c.cycle_id} /></TableCell>
                    <TableCell className="font-mono text-xs">{c.asset_id ?? '—'}</TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {fmtTimestamp(c.created_at)}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button asChild variant="ghost" size="sm">
                        <Link to={`/cycles/${c.cycle_id}`}>Open →</Link>
                      </Button>
                    </TableCell>
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
