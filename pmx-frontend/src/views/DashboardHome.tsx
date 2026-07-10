import { Link } from 'react-router-dom';
import { PlayCircle, ListOrdered, Factory, ArrowRight, Activity, Boxes, ClipboardCheck } from 'lucide-react';

import { useFleetStatus } from '@/hooks/useFleetStatus';
import { MetricCard } from '@/components/common/MetricCard';
import { EmptyState } from '@/components/common/EmptyState';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { CopyableId } from '@/components/common/CopyableId';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { fmtTimestamp } from '@/lib/format';

export function DashboardHome() {
  const fleet = useFleetStatus();

  const knownAssets = fleet.data?.known_assets ?? [];
  const knownCycles = fleet.data?.known_cycles ?? [];
  const recentCycles = [...knownCycles]
    .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
    .slice(0, 8);

  const totalActionCycles = (fleet.data?.asset_summaries ?? []).reduce(
    (acc, s) => acc + (s.counts?.cycle_action ?? 0),
    0,
  );
  const totalEngineerDecisions = (fleet.data?.asset_summaries ?? []).reduce(
    (acc, s) => acc + (s.counts?.engineer_decision ?? 0),
    0,
  );

  return (
    <div className="mx-auto max-w-6xl space-y-6 animate-fade-in-up">
      {/* Hero */}
      <div className="relative overflow-hidden rounded-xl border bg-gradient-to-br from-primary/10 via-primary/5 to-background p-6 shadow-sm">
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              'radial-gradient(circle at 85% 20%, rgba(59, 130, 246, 0.15), transparent 55%)',
          }}
        />
        <div className="relative flex flex-col justify-between gap-4 md:flex-row md:items-end">
          <div className="max-w-xl">
            <div className="mb-2 inline-flex items-center gap-1.5 rounded-full border bg-background/60 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground backdrop-blur">
              <span className="h-1.5 w-1.5 animate-soft-pulse rounded-full bg-emerald-500" />
              Simulation: {fleet.data?.simulation_adapter ?? '—'}
            </div>
            <h1 className="text-3xl font-semibold tracking-tight">
              <span className="bg-gradient-to-r from-primary to-purple-500 bg-clip-text text-transparent">
                EIx
              </span>{' '}
              — Engineering Intelligence, agentic.
            </h1>
            <p className="mt-2 text-sm text-muted-foreground">
              Cross-domain multi-agent predictive maintenance: physics, ML, a
              Gemini judge with tools, and a critic — every fleet decision
              justified and traceable.
            </p>
          </div>
          <div className="flex gap-2">
            <Button asChild size="lg" className="shadow-md shadow-primary/20">
              <Link to="/analyze/new">
                <PlayCircle className="mr-2 h-4 w-4" />
                New analysis
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link to="/cycles">
                <ListOrdered className="mr-2 h-4 w-4" />
                Browse cycles
              </Link>
            </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard
          label="Known assets"
          value={fleet.isLoading ? <LoadingSpinner /> : knownAssets.length}
          tone="info"
          icon={<Boxes className="h-3.5 w-3.5" />}
        />
        <MetricCard
          label="Known cycles"
          value={fleet.isLoading ? <LoadingSpinner /> : knownCycles.length}
          tone="default"
          icon={<Activity className="h-3.5 w-3.5" />}
        />
        <MetricCard
          label="Action cycles"
          value={fleet.isLoading ? <LoadingSpinner /> : totalActionCycles}
          hint="fleet-memory: cycle_action"
          tone={totalActionCycles > 0 ? 'warning' : 'default'}
        />
        <MetricCard
          label="Engineer decisions"
          value={fleet.isLoading ? <LoadingSpinner /> : totalEngineerDecisions}
          hint="fleet-memory: engineer_decision"
          tone={totalEngineerDecisions > 0 ? 'success' : 'default'}
          icon={<ClipboardCheck className="h-3.5 w-3.5" />}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ActionTile to="/analyze/new" icon={PlayCircle} title="New analysis" desc="Run the LangGraph pipeline on a sensor batch." tone="from-primary/20" />
        <ActionTile to="/cycles" icon={ListOrdered} title="Cycles" desc="Browse and inspect prior cycles." tone="from-emerald-500/20" />
        <ActionTile to="/fleet" icon={Factory} title="Fleet" desc="Aggregate memory across all known assets." tone="from-purple-500/20" />
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <CardTitle className="text-base">Recent cycles</CardTitle>
          <Button asChild variant="ghost" size="sm">
            <Link to="/cycles">
              View all <ArrowRight className="ml-1 h-3.5 w-3.5" />
            </Link>
          </Button>
        </CardHeader>
        <CardContent>
          {fleet.isLoading ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <LoadingSpinner /> Loading fleet status…
            </div>
          ) : recentCycles.length === 0 ? (
            <EmptyState
              title="No cycles yet"
              description="Kick off your first analysis to populate the dashboard."
              action={
                <Button asChild size="sm">
                  <Link to="/analyze/new">Start a cycle</Link>
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
                {recentCycles.map((c) => (
                  <TableRow key={c.cycle_id}>
                    <TableCell className="p-2"><CopyableId value={c.cycle_id} /></TableCell>
                    <TableCell className="font-mono text-xs">{c.asset_id ?? '—'}</TableCell>
                    <TableCell className="text-xs text-muted-foreground">{fmtTimestamp(c.created_at)}</TableCell>
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

function ActionTile({
  to,
  icon: Icon,
  title,
  desc,
  tone,
}: {
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  desc: string;
  tone: string;
}) {
  return (
    <Link
      to={to}
      className={`group relative overflow-hidden rounded-lg border bg-card p-5 transition-all hover:border-primary hover:shadow-md`}
    >
      <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${tone} to-transparent opacity-60 transition-opacity group-hover:opacity-100`} />
      <div className="relative">
        <div className="mb-3 grid h-9 w-9 place-items-center rounded-md bg-background/80 shadow-sm">
          <Icon className="h-5 w-5 text-primary" />
        </div>
        <div className="font-medium">{title}</div>
        <div className="mt-1 text-xs text-muted-foreground">{desc}</div>
      </div>
    </Link>
  );
}
