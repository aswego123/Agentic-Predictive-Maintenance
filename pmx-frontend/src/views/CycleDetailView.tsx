import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  RefreshCw,
  ArrowLeft,
  LayoutDashboard,
  Radar,
  Sigma,
  Brain,
  Wrench,
  Bug,
  Activity,
  Gauge,
  Sparkles,
  ShieldAlert,
  Waves,
} from 'lucide-react';
import { toast } from 'sonner';

import { useCycle } from '@/hooks/useCycle';
import { useAppStore } from '@/store';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { StatusBadge } from '@/components/common/StatusBadge';
import { CopyableId } from '@/components/common/CopyableId';
import { MetricCard, toneForNumber } from '@/components/common/MetricCard';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { EmptyState } from '@/components/common/EmptyState';
import { PanelErrorBoundary } from '@/components/common/PanelErrorBoundary';
import { PipelineGraph } from '@/components/cycle/PipelineGraph';
import { VerdictHero } from '@/components/cycle/VerdictHero';
import { AnomalyPanel } from '@/components/cycle/AnomalyPanel';
import { StressField3D } from '@/components/cycle/StressField3D';
import { PhysicsMLPanel } from '@/components/cycle/PhysicsMLPanel';
import { LifecyclePredictionPanel } from '@/components/cycle/LifecyclePredictionPanel';
import { DialoguePanel } from '@/components/cycle/DialoguePanel';
import { NegotiationHistoryPanel } from '@/components/cycle/NegotiationHistoryPanel';
import { JudgeVerdictPanel } from '@/components/cycle/JudgeVerdictPanel';
import { JudgeToolCallsPanel } from '@/components/cycle/JudgeToolCallsPanel';
import { CalibrationPanel } from '@/components/cycle/CalibrationPanel';
import { MaterialRecommendationPanel } from '@/components/cycle/MaterialRecommendationPanel';
import { CriticPanel } from '@/components/cycle/CriticPanel';
import { TracePanel } from '@/components/cycle/TracePanel';
import { RawStatePanel } from '@/components/cycle/RawStatePanel';
import { ApprovalForm } from '@/components/cycle/ApprovalForm';
import { DownloadReportButton } from '@/components/cycle/DownloadReportButton';

const HEALTH_TONE: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'default'> = {
  normal: 'success',
  monitoring: 'info',
  warning: 'warning',
  critical: 'danger',
  failed: 'danger',
};

export function CycleDetailView() {
  const { cycleId } = useParams();
  const setPolling = useAppStore((s) => s.setPolling);
  const cycleQuery = useCycle(cycleId);
  const cycle = cycleQuery.data;

  const [activeTab, setActiveTab] = useState<string>('overview');
  const prevStatusRef = useRef<string | null>(null);
  const prevAwaitingRef = useRef<boolean>(false);

  // Stop polling once the graph resolves; also toast on status transitions.
  useEffect(() => {
    if (!cycleId || !cycle) return;

    const prevStatus = prevStatusRef.current;
    const prevAwaiting = prevAwaitingRef.current;
    const currStatus = cycle.status;
    const currAwaiting = cycle.awaiting_engineer_approval;

    if (prevStatus === 'running' && currStatus !== 'running') {
      toast.success(`Cycle finished — status: ${currStatus}`);
    }
    if (!prevAwaiting && currAwaiting) {
      toast.warning('Engineer approval required', {
        description: 'The graph is paused at the interrupt.',
      });
    }
    prevStatusRef.current = currStatus;
    prevAwaitingRef.current = currAwaiting;

    if (currStatus !== 'running' && currStatus !== 'pending' && !currAwaiting) {
      setPolling(cycleId, false);
    }
  }, [cycle, cycleId, setPolling]);

  // Derived KPI values used in the sticky header strip.
  const physRul = Number(cycle?.physics_prediction?.rul_hours ?? NaN);
  const rulTone = toneForNumber(physRul, { danger: 24, warning: 100 });
  const healthStatus = String(cycle?.physics_prediction?.health_status ?? '').toLowerCase();
  const healthTone = HEALTH_TONE[healthStatus] ?? 'default';
  const judgeConfidence = Number(cycle?.judge_verdict?.confidence_score ?? NaN);
  const toolCallsMade = Number(
    cycle?.judge_verdict?.tool_calls_made ??
      (Array.isArray((cycle as any)?.judge_verdict?.tool_call_details)
        ? (cycle as any).judge_verdict.tool_call_details.length
        : NaN),
  );

  // Tab badge counts — surface where the action is without a click.
  const badges = useMemo(
    () => ({
      detection: cycle?.is_anomalous ? '!' : null,
      reasoning:
        (Number.isFinite(toolCallsMade) && toolCallsMade > 0
          ? String(toolCallsMade)
          : null) ??
        (cycle?.negotiation_history?.length
          ? String(cycle.negotiation_history.length)
          : null),
      action: cycle?.action ? '●' : null,
      debug: cycle?.trace?.length ? String(cycle.trace.length) : null,
    }),
    [cycle, toolCallsMade],
  );

  if (cycleQuery.isLoading) {
    return (
      <div className="mx-auto flex max-w-4xl items-center gap-2 py-16 text-sm text-muted-foreground">
        <LoadingSpinner /> Loading cycle {cycleId}…
      </div>
    );
  }

  if (cycleQuery.isError || !cycle) {
    return (
      <div className="mx-auto max-w-2xl py-16">
        <EmptyState
          title="Cycle not found"
          description={`No snapshot available for ${cycleId}.`}
          action={
            <Button asChild size="sm">
              <Link to="/cycles">Back to cycles</Link>
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl space-y-4 animate-fade-in-up">
      {/* ─── Compact header ─────────────────────────────────────── */}
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Button asChild variant="ghost" size="sm" className="-ml-2 h-8">
            <Link to="/cycles">
              <ArrowLeft className="mr-1 h-3.5 w-3.5" />
              Cycles
            </Link>
          </Button>
          <span className="text-muted-foreground/40">/</span>
          <CopyableId value={cycle.cycle_id} label="Cycle ID" />
          <StatusBadge status={cycle.status} />
          {cycle.asset_id ? (
            <span className="text-xs text-muted-foreground">
              · Asset <CopyableId value={cycle.asset_id} label="Asset ID" />
            </span>
          ) : null}
          {cycle.asset_type ? (
            <code className="rounded bg-muted px-1.5 py-0.5 text-[11px]">{cycle.asset_type}</code>
          ) : null}
        </div>
        <div className="flex gap-2">
          <DownloadReportButton cycle={cycle} />
          <Button
            variant="outline"
            size="sm"
            onClick={() => cycleQuery.refetch()}
            disabled={cycleQuery.isFetching}
          >
            <RefreshCw className={`mr-1 h-4 w-4 ${cycleQuery.isFetching ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </header>

      {/* ─── Compass zone: verdict + KPIs + urgent banners ───── */}
      {/*
        This block used to be `sticky top-0` but the user found the
        floating overlay distracting (scrolling content passed under
        the KPI cards). It now flows inline and scrolls normally with
        the rest of the page.
      */}
      <div className="space-y-3">
        <PanelErrorBoundary name="Verdict hero">
          <VerdictHero cycle={cycle} />
        </PanelErrorBoundary>

        {/* KPI strip — 4 numbers that summarise the whole cycle */}
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <MetricCard
            label="Health"
            value={healthStatus ? healthStatus.toUpperCase() : '—'}
            tone={healthTone}
            icon={<Gauge className="h-3.5 w-3.5" />}
            hint={cycle.physics_prediction?.failure_mode as string | undefined}
          />
          <MetricCard
            label="Anomaly"
            value={cycle.is_anomalous ? 'DETECTED' : 'clean'}
            tone={cycle.is_anomalous ? 'warning' : 'success'}
            icon={<ShieldAlert className="h-3.5 w-3.5" />}
            hint={
              cycle.anomaly_result?.score !== undefined
                ? `score ${Number(cycle.anomaly_result.score).toFixed(3)}`
                : undefined
            }
          />
          <MetricCard
            label="Physics RUL"
            value={!Number.isNaN(physRul) ? `${physRul.toFixed(1)} h` : '—'}
            tone={rulTone}
            icon={<Activity className="h-3.5 w-3.5" />}
            hint={
              cycle.ml_correction?.rul_hours !== undefined
                ? `ML ${Number(cycle.ml_correction.rul_hours).toFixed(1)} h`
                : undefined
            }
          />
          <MetricCard
            label="Judge confidence"
            value={
              !Number.isNaN(judgeConfidence)
                ? `${(judgeConfidence * 100).toFixed(0)}%`
                : '—'
            }
            tone={
              !Number.isNaN(judgeConfidence)
                ? judgeConfidence >= 0.85
                  ? 'success'
                  : judgeConfidence >= 0.6
                    ? 'info'
                    : 'warning'
                : 'default'
            }
            icon={<Sparkles className="h-3.5 w-3.5" />}
            hint={
              Number.isFinite(toolCallsMade) && toolCallsMade > 0
                ? `${toolCallsMade} tool call${toolCallsMade === 1 ? '' : 's'}`
                : undefined
            }
          />
        </div>

        {/* Urgent banners — pinned so users can't miss them */}
        {cycle.status === 'running' || cycle.status === 'pending' ? (
          <Alert variant="info" className="animate-soft-pulse">
            <AlertTitle className="flex items-center gap-2">
              <LoadingSpinner />
              {cycle.status === 'pending' ? 'Starting analysis…' : 'Graph still running'}
            </AlertTitle>
            <AlertDescription>
              Polling every ~0.75 s. Physics → ML → Judge ReAct loop typically completes in 60–150 s —
              the panels below will fill in progressively.
            </AlertDescription>
          </Alert>
        ) : null}

        {cycle.awaiting_engineer_approval ? (
          <PanelErrorBoundary name="Engineer approval">
            <ApprovalForm cycleId={cycle.cycle_id} onResolved={() => cycleQuery.refetch()} />
          </PanelErrorBoundary>
        ) : null}
      </div>

      {/* ─── Tabbed detail area ─────────────────────────────────── */}
      {/*
        Custom-sized tabs (h-11, text-sm, px-4 py-2) — the default
        shadcn h-9 / text-xs combo was too small against the dense
        content below, so users struggled to spot the tab bar.
      */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="flex h-auto w-full flex-wrap justify-start gap-1.5 rounded-xl border bg-muted/40 p-1.5">
          <TabsTrigger
            value="overview"
            className="gap-2 rounded-lg px-4 py-2 text-sm font-medium data-[state=active]:shadow-md"
          >
            <LayoutDashboard className="h-4 w-4" /> Overview
          </TabsTrigger>
          <TabsTrigger
            value="detection"
            className="gap-2 rounded-lg px-4 py-2 text-sm font-medium data-[state=active]:shadow-md"
          >
            <Radar className="h-4 w-4" /> Detection
            {badges.detection ? (
              <Badge variant="warning" className="ml-1 h-5 px-1.5 text-[11px]">
                {badges.detection}
              </Badge>
            ) : null}
          </TabsTrigger>
          <TabsTrigger
            value="stress"
            className="gap-2 rounded-lg px-4 py-2 text-sm font-medium data-[state=active]:shadow-md"
          >
            <Waves className="h-4 w-4" /> Stress Intensity
          </TabsTrigger>
          <TabsTrigger
            value="prediction"
            className="gap-2 rounded-lg px-4 py-2 text-sm font-medium data-[state=active]:shadow-md"
          >
            <Sigma className="h-4 w-4" /> Physics & ML
          </TabsTrigger>
          <TabsTrigger
            value="reasoning"
            className="gap-2 rounded-lg px-4 py-2 text-sm font-medium data-[state=active]:shadow-md"
          >
            <Brain className="h-4 w-4" /> Reasoning
            {badges.reasoning ? (
              <Badge variant="info" className="ml-1 h-5 px-1.5 text-[11px]">
                {badges.reasoning}
              </Badge>
            ) : null}
          </TabsTrigger>
          <TabsTrigger
            value="action"
            className="gap-2 rounded-lg px-4 py-2 text-sm font-medium data-[state=active]:shadow-md"
          >
            <Wrench className="h-4 w-4" /> Action plan
            {badges.action ? (
              <Badge variant="success" className="ml-1 h-5 px-1.5 text-[11px]">
                {badges.action}
              </Badge>
            ) : null}
          </TabsTrigger>
          <TabsTrigger
            value="debug"
            className="gap-2 rounded-lg px-4 py-2 text-sm font-medium data-[state=active]:shadow-md"
          >
            <Bug className="h-4 w-4" /> Debug
            {badges.debug ? (
              <Badge variant="outline" className="ml-1 h-5 px-1.5 text-[11px]">
                {badges.debug}
              </Badge>
            ) : null}
          </TabsTrigger>
        </TabsList>

        {/* Overview — pipeline compass + lifecycle scheduler */}
        <TabsContent value="overview" className="space-y-4 pt-3">
          <PanelErrorBoundary name="Pipeline graph">
            <PipelineGraph cycle={cycle} />
          </PanelErrorBoundary>
          <PanelErrorBoundary name="Lifecycle prediction">
            <LifecyclePredictionPanel cycle={cycle} />
          </PanelErrorBoundary>
        </TabsContent>

        {/* Detection — sensor-side findings */}
        <TabsContent value="detection" className="space-y-4 pt-3">
          <PanelErrorBoundary name="Anomaly">
            <AnomalyPanel anomaly={cycle.anomaly_result} />
          </PanelErrorBoundary>
        </TabsContent>

        {/* Stress Intensity — 3D stress-density scatter over the component mesh */}
        <TabsContent value="stress" className="space-y-4 pt-3">
          <PanelErrorBoundary name="3D stress field">
            <StressField3D field={(cycle.stress_features?.extras as any)?.stress_field_3d} />
          </PanelErrorBoundary>
        </TabsContent>

        {/* Physics & ML — model predictions and disagreement */}
        <TabsContent value="prediction" className="space-y-4 pt-3">
          <PanelErrorBoundary name="Physics vs ML">
            <PhysicsMLPanel physics={cycle.physics_prediction} ml={cycle.ml_correction} />
          </PanelErrorBoundary>
        </TabsContent>

        {/* Reasoning — the LLM Judge's chain of thought */}
        <TabsContent value="reasoning" className="space-y-4 pt-3">
          <PanelErrorBoundary name="Judge verdict">
            <JudgeVerdictPanel verdict={cycle.judge_verdict} />
          </PanelErrorBoundary>
          <PanelErrorBoundary name="Judge tool calls">
            <JudgeToolCallsPanel cycle={cycle} />
          </PanelErrorBoundary>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <PanelErrorBoundary name="Dialogue">
              <DialoguePanel
                history={cycle.dialogue_history ?? []}
                fetched={cycle.fetched_features ?? {}}
              />
            </PanelErrorBoundary>
            <PanelErrorBoundary name="Negotiation history">
              <NegotiationHistoryPanel history={cycle.negotiation_history ?? []} />
            </PanelErrorBoundary>
          </div>
        </TabsContent>

        {/* Action plan — what to do next */}
        <TabsContent value="action" className="space-y-4 pt-3">
          <PanelErrorBoundary name="Material recommendation">
            <MaterialRecommendationPanel
              recommendations={(cycle.action as any)?.material_recommendations ?? null}
              currentMaterial={(cycle as any).material_name ?? null}
            />
          </PanelErrorBoundary>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <PanelErrorBoundary name="Calibration">
              <CalibrationPanel calib={cycle.calibration_result} />
            </PanelErrorBoundary>
            <PanelErrorBoundary name="Critic">
              <CriticPanel critic={cycle.critic_review} assetId={cycle.asset_id} />
            </PanelErrorBoundary>
          </div>
        </TabsContent>

        {/* Debug — raw internals for power users */}
        <TabsContent value="debug" className="space-y-4 pt-3">
          <PanelErrorBoundary name="Trace">
            <TracePanel trace={cycle.trace} />
          </PanelErrorBoundary>
          <PanelErrorBoundary name="Raw state">
            <RawStatePanel cycle={cycle} />
          </PanelErrorBoundary>
        </TabsContent>
      </Tabs>
    </div>
  );
}
