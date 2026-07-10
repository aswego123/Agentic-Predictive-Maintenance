import { useMemo, useCallback, useEffect } from 'react';
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  type Edge,
  type EdgeTypes,
  type Node,
  type NodeProps,
  type NodeTypes,
} from 'reactflow';
import { Check, Circle, Loader2, Pause, Wrench, MessageSquare } from 'lucide-react';

import 'reactflow/dist/style.css';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { fmtNumber } from '@/lib/format';
import type { CycleSnapshot } from '@/types/domain';

// ---------------------------------------------------------------
// Types
// ---------------------------------------------------------------

type NodeState = 'done' | 'active' | 'paused' | 'pending' | 'terminal';

type NodeCategory = 'agent' | 'judge' | 'terminal';

interface AgentNodeData {
  label: string;
  state: NodeState;
  runs: number;
  secondaryLine?: string;
  tertiaryLine?: string;
  durationMs?: number;
}

interface JudgeToolCall {
  order: number;
  tool: string;
  args: Record<string, any>;
  observationSnippet: string;
}

interface JudgeNodeData extends AgentNodeData {
  toolCalls: JudgeToolCall[];
  source?: string;
  confidence?: number;
  route?: string;
}

// ---------------------------------------------------------------
// Node → graph topology (mirrors graph/build_graph.py)
// ---------------------------------------------------------------

const NODE_ORDER = [
  'init', 'ingest', 'end_normal', 'simulation', 'physics', 'ml',
  'data_fetch', 'judge', 'calibration', 'engineer_approval', 'action',
  'critic', 'end',
] as const;
type NodeId = typeof NODE_ORDER[number];

const NODE_META: Record<NodeId, { label: string; category: NodeCategory }> = {
  init:              { label: 'init',              category: 'agent' },
  ingest:            { label: 'ingest',            category: 'agent' },
  end_normal:        { label: 'END (normal)',      category: 'terminal' },
  simulation:        { label: 'simulation',        category: 'agent' },
  physics:           { label: 'physics',           category: 'agent' },
  ml:                { label: 'ml_correction',     category: 'agent' },
  data_fetch:        { label: 'data_fetch',        category: 'agent' },
  judge:             { label: 'judge',             category: 'judge' },
  calibration:       { label: 'calibration',       category: 'agent' },
  engineer_approval: { label: 'engineer_approval', category: 'agent' },
  action:            { label: 'action',            category: 'agent' },
  critic:            { label: 'critic',            category: 'agent' },
  end:               { label: 'END',               category: 'terminal' },
};

const EDGES_STATIC: Array<{
  id: string;
  source: NodeId;
  target: NodeId;
  sourceHandle: string;
  targetHandle: string;
  label?: string;
  dashed?: boolean;
}> = [
  { id: 'e-init-ingest',   source: 'init',       target: 'ingest',     sourceHandle: 's-b', targetHandle: 't-t' },
  { id: 'e-ingest-sim',    source: 'ingest',     target: 'simulation', sourceHandle: 's-b', targetHandle: 't-t', label: 'anomalous' },
  { id: 'e-ingest-endn',   source: 'ingest',     target: 'end_normal', sourceHandle: 's-r', targetHandle: 't-l', label: 'normal',  dashed: true },
  { id: 'e-sim-physics',   source: 'simulation', target: 'physics',    sourceHandle: 's-l', targetHandle: 't-t' },
  { id: 'e-sim-ml',        source: 'simulation', target: 'ml',         sourceHandle: 's-b', targetHandle: 't-t' },
  { id: 'e-physics-ml',    source: 'physics',    target: 'ml',         sourceHandle: 's-r', targetHandle: 't-l', label: 'round 1' },
  { id: 'e-ml-physics',    source: 'ml',         target: 'physics',    sourceHandle: 's-b', targetHandle: 't-b', label: 'round<2', dashed: true },
  { id: 'e-ml-fetch',      source: 'ml',         target: 'data_fetch', sourceHandle: 's-r', targetHandle: 't-l', label: 'request_data', dashed: true },
  { id: 'e-ml-judge',      source: 'ml',         target: 'judge',      sourceHandle: 's-b', targetHandle: 't-t' },
  { id: 'e-fetch-judge',   source: 'data_fetch', target: 'judge',      sourceHandle: 's-b', targetHandle: 't-r' },
  { id: 'e-judge-calib',   source: 'judge',      target: 'calibration', sourceHandle: 's-l', targetHandle: 't-r', label: 'divergent', dashed: true },
  { id: 'e-judge-action',  source: 'judge',      target: 'action',     sourceHandle: 's-b', targetHandle: 't-t', label: 'action_fast_path' },
  { id: 'e-calib-eng',     source: 'calibration', target: 'engineer_approval', sourceHandle: 's-b', targetHandle: 't-t' },
  { id: 'e-eng-sim',       source: 'engineer_approval', target: 'simulation',  sourceHandle: 's-t', targetHandle: 't-l', label: 'approved & resim<5', dashed: true },
  { id: 'e-eng-action',    source: 'engineer_approval', target: 'action',      sourceHandle: 's-r', targetHandle: 't-l', label: 'rejected / cap' },
  { id: 'e-action-critic', source: 'action',     target: 'critic',     sourceHandle: 's-r', targetHandle: 't-l' },
  { id: 'e-critic-end',    source: 'critic',     target: 'end',        sourceHandle: 's-r', targetHandle: 't-l' },
];

const TRACE_NODE_TO_GRAPH: Record<string, NodeId> = {
  'orchestrator.init':      'init',
  'ingestion.anomaly_gate': 'ingest',
  simulation_layer:         'simulation',
  physics_agent:            'physics',
  ml_correction_agent:      'ml',
  data_fetch_agent:         'data_fetch',
  judge_agent:              'judge',
  'judge_agent.tool_call':  'judge',
  calibration_agent:        'calibration',
  engineer_approval:        'engineer_approval',
  action_agent:             'action',
  critic_agent:             'critic',
};

function graphNodeFor(traceNode?: string): NodeId | null {
  if (!traceNode) return null;
  if (TRACE_NODE_TO_GRAPH[traceNode]) return TRACE_NODE_TO_GRAPH[traceNode];
  for (const prefix of Object.keys(TRACE_NODE_TO_GRAPH)) {
    if (traceNode.startsWith(prefix)) return TRACE_NODE_TO_GRAPH[prefix];
  }
  return null;
}

// ---------------------------------------------------------------
// Layout — dagre auto-layout for a nice top-down flow
// ---------------------------------------------------------------

const AGENT_NODE_WIDTH = 200;
const JUDGE_NODE_WIDTH = 240;
const TERMINAL_NODE_WIDTH = 130;

/**
 * Manual XY positions arranged in a compact grid so the whole graph
 * fits in a roughly square canvas (~840×880). Beats dagre's TB/LR
 * strict rank layout which either goes too tall or too wide.
 *
 * The layout has 3 columns:
 *   Left  (x ~ 40)  — recovery loop (calibration → engineer_approval)
 *   Mid   (x ~ 340) — core pipeline (init → ingest → simulation → judge → action → critic)
 *   Right (x ~ 640) — branches (end_normal, ml, data_fetch, critic-tail)
 */
const MANUAL_POSITIONS: Record<NodeId, { x: number; y: number }> = {
  init:              { x: 340, y: 10  },
  ingest:            { x: 340, y: 105 },
  end_normal:        { x: 640, y: 130 },  // small, aligned vertically with ingest center
  simulation:        { x: 340, y: 210 },
  physics:           { x: 40,  y: 320 },
  ml:                { x: 340, y: 320 },
  data_fetch:        { x: 640, y: 320 },
  judge:             { x: 320, y: 440 },  // 240 wide, sits under ml with slight left offset
  calibration:       { x: 40,  y: 690 },
  action:            { x: 340, y: 690 },
  critic:            { x: 640, y: 690 },
  engineer_approval: { x: 40,  y: 795 },
  end:               { x: 700, y: 715 },  // small, aligned with critic vertically
};

/**
 * Apply MANUAL_POSITIONS. We still expose the function so the caller
 * signature stays stable, but dagre auto-layout is intentionally NOT
 * used — it produced either an over-tall TB tree or an over-wide LR
 * chain and both required aggressive zoom-out to fit.
 */
function layoutNodes(rfNodes: Node[], _rfEdges: Edge[]): Node[] {
  return rfNodes.map((n) => ({
    ...n,
    position: MANUAL_POSITIONS[n.id as NodeId] ?? { x: 0, y: 0 },
  }));
}

// ---------------------------------------------------------------
// Custom nodes
// ---------------------------------------------------------------

function StateChip({ state }: { state: NodeState }) {
  const map = {
    done:     { icon: Check,   text: 'done',    cls: 'text-emerald-600 bg-emerald-500/10 border-emerald-500/30' },
    active:   { icon: Loader2, text: 'active',  cls: 'text-primary bg-primary/10 border-primary/30 animate-soft-pulse' },
    paused:   { icon: Pause,   text: 'paused',  cls: 'text-amber-600 bg-amber-500/10 border-amber-500/30 animate-soft-pulse' },
    pending:  { icon: Circle,  text: 'pending', cls: 'text-muted-foreground bg-muted/40 border-border' },
    terminal: { icon: Check,   text: 'end',     cls: 'text-muted-foreground bg-muted/40 border-border' },
  } as const;
  const { icon: Icon, text, cls } = map[state];
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider', cls)}>
      <Icon className={cn('h-2.5 w-2.5', state === 'active' && 'animate-spin')} />
      {text}
    </span>
  );
}

function AgentNode({ data }: NodeProps<AgentNodeData>) {
  const style = STATE_STYLE[data.state];
  return (
    <div
      className={cn(
        'rounded-md border-2 bg-card px-3 py-2 shadow-sm transition-all',
        data.state === 'active' && 'shadow-lg shadow-primary/30',
      )}
      style={{ borderColor: style.border, width: AGENT_NODE_WIDTH }}
    >
      <Handle type="target" position={Position.Left} className="!bg-muted-foreground/60" />
      <div className="flex items-center justify-between gap-2">
        <div className="truncate text-sm font-semibold" style={{ color: style.textColor }}>
          {data.label}
        </div>
        <StateChip state={data.state} />
      </div>
      {data.secondaryLine ? (
        <div className="mt-1 truncate text-xs tabular-nums text-foreground/80">{data.secondaryLine}</div>
      ) : null}
      {data.tertiaryLine ? (
        <div className="truncate text-[10px] text-muted-foreground">{data.tertiaryLine}</div>
      ) : null}
      <div className="mt-1 flex items-center justify-between text-[10px] text-muted-foreground/70">
        {data.runs > 1 ? <span className="rounded-full bg-muted px-1.5 py-0.5 font-bold">×{data.runs}</span> : <span />}
        {data.durationMs !== undefined && data.state !== 'pending' ? (
          <span className="tabular-nums">{formatMs(data.durationMs)}</span>
        ) : null}
      </div>
      <Handle type="source" position={Position.Right} className="!bg-muted-foreground/60" />
    </div>
  );
}

function TerminalNode({ data }: NodeProps<AgentNodeData>) {
  const style = STATE_STYLE[data.state === 'terminal' ? 'done' : data.state];
  return (
    <div
      className="rounded-full border-2 border-dashed bg-card px-4 py-1.5 text-xs font-semibold shadow-sm"
      style={{ borderColor: style.border, color: style.textColor, width: TERMINAL_NODE_WIDTH, textAlign: 'center' }}
    >
      <Handle type="target" position={Position.Left} className="!bg-muted-foreground/60" />
      {data.label}
    </div>
  );
}

function JudgeNode({ data }: NodeProps<JudgeNodeData>) {
  const style = STATE_STYLE[data.state];
  return (
    <div
      className={cn(
        'rounded-md border-2 bg-card p-3 shadow-sm',
        data.state === 'active' && 'shadow-lg shadow-primary/30',
      )}
      style={{ borderColor: style.border, width: JUDGE_NODE_WIDTH }}
    >
      <Handle type="target" position={Position.Left} className="!bg-muted-foreground/60" />
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: style.textColor }}>
          <Wrench className="h-3.5 w-3.5" />
          {data.label}
        </div>
        <StateChip state={data.state} />
      </div>
      {data.secondaryLine ? (
        <div className="mt-1 text-xs tabular-nums text-foreground/80">{data.secondaryLine}</div>
      ) : null}
      {data.tertiaryLine ? (
        <div className="text-[10px] text-muted-foreground">→ {data.tertiaryLine}</div>
      ) : null}
      {data.toolCalls.length > 0 ? (
        <div className="mt-2 space-y-0.5 rounded-md border bg-muted/30 p-1.5">
          <div className="mb-0.5 text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">
            Tool calls ({data.toolCalls.length})
          </div>
          {data.toolCalls.map((tc) => (
            <div key={tc.order} className="flex items-center gap-1.5 text-[10px]">
              <span className="inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[9px] font-bold text-primary">
                {tc.order}
              </span>
              <code className="truncate font-mono text-[10px]" title={`${tc.tool}(${JSON.stringify(tc.args)})`}>
                {tc.tool}
              </code>
            </div>
          ))}
        </div>
      ) : null}
      <div className="mt-1 flex items-center justify-between text-[10px] text-muted-foreground/70">
        {data.runs > 1 ? <span className="rounded-full bg-muted px-1.5 py-0.5 font-bold">×{data.runs}</span> : <span />}
        {data.durationMs !== undefined && data.state !== 'pending' ? (
          <span className="tabular-nums">{formatMs(data.durationMs)}</span>
        ) : null}
      </div>
      <Handle type="source" position={Position.Right} className="!bg-muted-foreground/60" />
    </div>
  );
}

const nodeTypes: NodeTypes = {
  agent: AgentNode,
  judge: JudgeNode,
  terminal: TerminalNode,
};

const edgeTypes: EdgeTypes = {};

// ---------------------------------------------------------------
// Palette
// ---------------------------------------------------------------

const STATE_STYLE: Record<NodeState, { border: string; textColor: string; edge: string }> = {
  done:     { border: 'hsl(142 71% 45%)', textColor: 'hsl(142 71% 25%)', edge: 'hsl(142 71% 45%)' },
  active:   { border: 'hsl(221 83% 53%)', textColor: 'hsl(221 83% 30%)', edge: 'hsl(221 83% 53%)' },
  paused:   { border: 'hsl(38 92% 50%)',  textColor: 'hsl(28 80% 30%)',  edge: 'hsl(38 92% 50%)' },
  pending:  { border: 'hsl(0 0% 78%)',    textColor: 'hsl(0 0% 40%)',    edge: 'hsl(0 0% 78%)' },
  terminal: { border: 'hsl(0 0% 55%)',    textColor: 'hsl(0 0% 25%)',    edge: 'hsl(0 0% 55%)' },
};

// ---------------------------------------------------------------
// Public component (with provider)
// ---------------------------------------------------------------

export function PipelineGraph({ cycle }: { cycle: CycleSnapshot }) {
  return (
    <ReactFlowProvider>
      <PipelineGraphInner cycle={cycle} />
    </ReactFlowProvider>
  );
}

function PipelineGraphInner({ cycle }: { cycle: CycleSnapshot }) {
  const { rfNodes, rfEdges, activityFeed, activeId, dialogueSummary, fetchedSummary } = useMemo(
    () => build(cycle),
    [cycle],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(rfNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(rfEdges);

  // Sync live data into existing node/edge objects so user-dragged
  // positions survive across snapshot updates.
  useEffect(() => {
    setNodes((prev) => {
      const posById = new Map(prev.map((n) => [n.id, n.position]));
      return rfNodes.map((n) => ({ ...n, position: posById.get(n.id) ?? n.position }));
    });
    setEdges(rfEdges);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rfNodes, rfEdges]);

  const proOptions = useMemo(() => ({ hideAttribution: true }), []);
  const isValidConnection = useCallback(() => false, []);

  const running = cycle.status === 'running';
  const paused = cycle.awaiting_engineer_approval;
  const activeLabel = activeId ? NODE_META[activeId].label : null;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          Pipeline graph — live
          {activeLabel ? (
            <Badge variant="info" className="animate-soft-pulse gap-1.5">
              <Loader2 className="h-3 w-3 animate-spin" />
              {activeLabel}
            </Badge>
          ) : running ? (
            <Badge variant="info" className="animate-soft-pulse">running…</Badge>
          ) : paused ? (
            <Badge variant="warning" className="animate-soft-pulse">paused at approval</Badge>
          ) : (
            <Badge variant="success">complete</Badge>
          )}
        </CardTitle>
        <Legend />
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="h-[620px] w-full rounded-md border bg-muted/10">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            proOptions={proOptions}
            fitView
            fitViewOptions={{ padding: 0.08, minZoom: 0.4, maxZoom: 1.3 }}
            minZoom={0.3}
            maxZoom={2}
            nodesDraggable
            nodesConnectable={false}
            elementsSelectable
            panOnScroll={false}
            zoomOnScroll
            isValidConnection={isValidConnection}
          >
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
            <MiniMap
              zoomable
              pannable
              nodeStrokeWidth={2}
              className="!bg-background !border"
              nodeColor={(n) => STATE_STYLE[(n.data?.state as NodeState) ?? 'pending'].edge}
            />
            <Controls showInteractive={false} className="!bg-background !shadow-md !border" />
          </ReactFlow>
        </div>

        {(dialogueSummary || fetchedSummary) ? (
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            {dialogueSummary ? (
              <div className="flex items-start gap-2 rounded-md border border-primary/20 bg-primary/5 p-2 text-xs">
                <MessageSquare className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
                <div>
                  <div className="font-semibold text-primary">Physics ⇄ ML dialogue</div>
                  <div className="mt-0.5 text-muted-foreground">{dialogueSummary}</div>
                </div>
              </div>
            ) : null}
            {fetchedSummary ? (
              <div className="flex items-start gap-2 rounded-md border p-2 text-xs">
                <Wrench className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <div>
                  <div className="font-semibold">Data fetch payload</div>
                  <div className="mt-0.5 font-mono text-[11px] text-muted-foreground">{fetchedSummary}</div>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        <ActivityFeed feed={activityFeed} running={running} />
      </CardContent>
    </Card>
  );
}

function Legend() {
  return (
    <div className="hidden flex-wrap items-center gap-3 text-[10px] text-muted-foreground md:flex">
      <LegendItem color="bg-emerald-500" label="done" />
      <LegendItem color="bg-primary" label="active" />
      <LegendItem color="bg-amber-500" label="paused" />
      <LegendItem color="bg-muted-foreground/40" label="pending" />
      <span className="text-muted-foreground/60">· drag nodes · scroll to zoom</span>
    </div>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1">
      <span className={cn('inline-block h-2 w-2 rounded-full', color)} />
      {label}
    </div>
  );
}

// ---------------------------------------------------------------
// Activity feed
// ---------------------------------------------------------------

interface ActivityEntry {
  ts: number;
  node: string;
  note: string;
  durationMs: number;
  isToolCall: boolean;
  toolName?: string;
}

function ActivityFeed({ feed, running }: { feed: ActivityEntry[]; running: boolean }) {
  if (feed.length === 0) return null;
  return (
    <div className="rounded-md border">
      <div className="flex items-center justify-between border-b bg-muted/30 px-3 py-1.5">
        <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          {running ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
          Activity feed ({feed.length} events)
        </div>
        <div className="text-[10px] text-muted-foreground">newest first · durations = time to next event</div>
      </div>
      <div className="max-h-72 overflow-y-auto">
        {[...feed].reverse().map((e, i) => (
          <div key={i} className={cn(
            'flex items-start gap-2 border-b px-3 py-1.5 last:border-b-0',
            i === 0 && running && 'animate-soft-pulse bg-primary/5',
          )}>
            <div className="w-14 shrink-0 text-right font-mono text-[10px] tabular-nums text-muted-foreground">
              {formatMs(e.durationMs)}
            </div>
            {e.isToolCall ? <Wrench className="mt-1 h-3 w-3 shrink-0 text-primary" /> : <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary/70" />}
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-baseline gap-2">
                <code className="rounded bg-muted px-1 text-[10px] font-semibold">{e.node}</code>
                {e.toolName ? (
                  <code className="rounded bg-primary/10 px-1 text-[10px] font-semibold text-primary">{e.toolName}</code>
                ) : null}
                <span className="text-[10px] text-muted-foreground">{new Date(e.ts * 1000).toLocaleTimeString()}</span>
              </div>
              <div className="truncate text-xs" title={e.note}>{e.note}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------
// Build (derive nodes/edges + feed from CycleSnapshot)
// ---------------------------------------------------------------

function build(cycle: CycleSnapshot) {
  const nodeStates = new Map<NodeId, NodeState>();
  const runs = new Map<NodeId, number>();
  const firstTs = new Map<NodeId, number>();
  const lastTs = new Map<NodeId, number>();
  NODE_ORDER.forEach((id) => nodeStates.set(id, 'pending'));

  const trace = cycle.trace ?? [];
  let lastGid: NodeId | null = null;
  const toolCalls: JudgeToolCall[] = [];
  for (const t of trace) {
    const gid = graphNodeFor(t.node);
    if (!gid) continue;
    nodeStates.set(gid, 'done');
    runs.set(gid, (runs.get(gid) ?? 0) + 1);
    if (!firstTs.has(gid)) firstTs.set(gid, t.ts);
    lastTs.set(gid, t.ts);
    lastGid = gid;
    if (String(t.node ?? '').startsWith('judge_agent.tool_call')) {
      toolCalls.push({
        order: toolCalls.length + 1,
        tool: String((t.data as any)?.tool ?? '?'),
        args: (t.data as any)?.args ?? {},
        observationSnippet: String((t.data as any)?.observation_snippet ?? ''),
      });
    }
  }

  let activeId: NodeId | null = null;
  if (cycle.status === 'running' && lastGid) {
    nodeStates.set(lastGid, 'active');
    activeId = lastGid;
  }
  if (cycle.awaiting_engineer_approval) {
    nodeStates.set('engineer_approval', 'paused');
    activeId = 'engineer_approval';
  }
  if (cycle.status && !['running', 'unknown'].includes(cycle.status)) {
    if (cycle.status === 'normal_end') nodeStates.set('end_normal', 'terminal');
    else nodeStates.set('end', 'terminal');
  }

  const secondaryLines = computeSecondaryLines(cycle);

  const rfNodes: Node[] = NODE_ORDER.map((id) => {
    const cat = NODE_META[id].category;
    const state = nodeStates.get(id)!;
    const base: AgentNodeData = {
      label: NODE_META[id].label,
      state,
      runs: runs.get(id) ?? 0,
      durationMs:
        firstTs.has(id)
          ? Math.max(0, Math.round((lastTs.get(id)! - firstTs.get(id)!) * 1000))
          : undefined,
      ...(secondaryLines[id] ?? {}),
    };

    if (cat === 'judge') {
      const verdict = (cycle.judge_verdict ?? {}) as any;
      const judgeData: JudgeNodeData = {
        ...base,
        toolCalls: toolCalls.slice(0, 5),
        source: verdict.source,
        confidence: verdict.confidence_score,
        route: verdict.route,
      };
      return { id, type: 'judge', data: judgeData, position: { x: 0, y: 0 } };
    }
    if (cat === 'terminal') {
      return { id, type: 'terminal', data: base, position: { x: 0, y: 0 } };
    }
    return { id, type: 'agent', data: base, position: { x: 0, y: 0 } };
  });

  const rfEdges: Edge[] = EDGES_STATIC.map((e) => {
    const fromState = nodeStates.get(e.source)!;
    const toState = nodeStates.get(e.target)!;
    const traversed =
      fromState === 'done' &&
      (toState === 'done' || toState === 'active' || toState === 'paused' || toState === 'terminal');
    return {
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle,
      targetHandle: e.targetHandle,
      label: e.label,
      labelStyle: { fontSize: 10, fill: traversed ? 'hsl(221 83% 40%)' : 'hsl(var(--muted-foreground))', fontWeight: 500 },
      labelBgPadding: [4, 2] as [number, number],
      labelBgBorderRadius: 3,
      labelBgStyle: { fill: 'hsl(var(--background))', fillOpacity: 0.9 },
      animated: traversed && cycle.status === 'running',
      style: {
        stroke: traversed ? 'hsl(221 83% 53%)' : 'hsl(var(--border))',
        strokeWidth: traversed ? 2 : 1,
        strokeDasharray: e.dashed ? '5 4' : undefined,
      },
      type: 'smoothstep',
      markerEnd: { type: 'arrowclosed' as any, color: traversed ? 'hsl(221 83% 53%)' : 'hsl(var(--muted-foreground))' },
    };
  });

  const laidOut = layoutNodes(rfNodes, rfEdges);

  const activityFeed: ActivityEntry[] = trace.map((t, i) => {
    const nextTs = i + 1 < trace.length ? trace[i + 1].ts : Date.now() / 1000;
    const durationMs = Math.max(0, Math.round((nextTs - t.ts) * 1000));
    const isToolCall = String(t.node ?? '').startsWith('judge_agent.tool_call');
    return {
      ts: t.ts,
      node: t.node,
      note: t.note,
      durationMs,
      isToolCall,
      toolName: isToolCall ? (t.data as any)?.tool : undefined,
    };
  });

  const dialogueHistory = cycle.dialogue_history ?? [];
  const dialogueSummary = dialogueHistory.length > 0 ? summarizeDialogue(dialogueHistory) : null;
  const fetched = cycle.fetched_features ?? {};
  const fetchedSummary = Object.keys(fetched).length > 0 ? summarizeFetched(fetched) : null;

  return { rfNodes: laidOut, rfEdges, activityFeed, activeId, dialogueSummary, fetchedSummary };
}

function computeSecondaryLines(cycle: CycleSnapshot): Partial<Record<NodeId, { secondaryLine?: string; tertiaryLine?: string }>> {
  const out: Partial<Record<NodeId, { secondaryLine?: string; tertiaryLine?: string }>> = {};

  const anomaly = cycle.anomaly_result as any;
  if (anomaly?.is_anomalous) {
    out.ingest = { secondaryLine: `${anomaly.anomaly_count ?? 0} anomalies`, tertiaryLine: `severity: ${anomaly.max_severity ?? '?'}` };
  }

  const stress = cycle.stress_features as any;
  if (stress?.stress_amplitude_mpa !== undefined) {
    out.simulation = { secondaryLine: `stress ${fmtNumber(stress.stress_amplitude_mpa, 0)} MPa` };
  }

  const physics = cycle.physics_prediction as any;
  if (physics?.rul_hours !== undefined) {
    out.physics = {
      secondaryLine: `RUL ${fmtNumber(physics.rul_hours, 1)} h`,
      tertiaryLine: physics.failure_mode ? String(physics.failure_mode).replace(/_/g, ' ') : undefined,
    };
  }

  const ml = cycle.ml_correction as any;
  if (ml?.rul_hours !== undefined) {
    const ci =
      ml.confidence_lower_hours !== undefined && ml.confidence_upper_hours !== undefined
        ? `CI [${fmtNumber(ml.confidence_lower_hours, 1)}, ${fmtNumber(ml.confidence_upper_hours, 1)}]`
        : ml.method;
    out.ml = { secondaryLine: `RUL ${fmtNumber(ml.rul_hours, 1)} h`, tertiaryLine: ci };
  }

  const fetched = cycle.fetched_features as any;
  if (fetched && Object.keys(fetched).length > 0) {
    out.data_fetch = { secondaryLine: Object.keys(fetched)[0] };
  }

  const verdict = cycle.judge_verdict as any;
  if (verdict?.confidence_score !== undefined) {
    out.judge = {
      secondaryLine: `conf ${fmtNumber(verdict.confidence_score, 2)} · ${verdict.tool_calls_made ?? 0} tools`,
      tertiaryLine: verdict.route,
    };
  }

  const calib = cycle.calibration_result as any;
  if (calib?.geometry_factor_new !== undefined) {
    out.calibration = {
      secondaryLine: `k ${fmtNumber(calib.geometry_factor_old, 2)} → ${fmtNumber(calib.geometry_factor_new, 2)}`,
    };
  }

  const action = cycle.action as any;
  if (action?.type) {
    out.action = { secondaryLine: String(action.type).replace(/_/g, ' ') };
  }

  const critic = cycle.critic_review as any;
  if (critic?.physics_weight !== undefined) {
    const prev = critic.previous_physics_weight;
    out.critic = {
      secondaryLine:
        prev !== undefined
          ? `w ${fmtNumber(prev, 2)} → ${fmtNumber(critic.physics_weight, 2)}`
          : `w ${fmtNumber(critic.physics_weight, 2)}`,
    };
  }

  const engDec = cycle.engineer_decision as any;
  if (engDec?.approved !== undefined) {
    out.engineer_approval = { secondaryLine: engDec.approved ? 'approved' : 'rejected', tertiaryLine: engDec.engineer_id };
  }

  return out;
}

function summarizeDialogue(history: Array<Record<string, any>>): string {
  const byRound: Record<string, string[]> = {};
  for (const h of history) {
    const r = `R${h.round ?? '?'}`;
    (byRound[r] ??= []).push(`${h.source ?? '?'}=${h.move ?? '?'}${h.data_request ? `(→${h.data_request})` : ''}`);
  }
  return Object.entries(byRound)
    .map(([r, moves]) => `${r}: ${moves.join(' · ')}`)
    .join(' | ');
}

function summarizeFetched(fetched: Record<string, any>): string {
  return Object.entries(fetched)
    .map(([k, v]) => {
      if (v && typeof v === 'object') {
        const inner = Object.entries(v as Record<string, any>)
          .slice(0, 2)
          .map(([kk, vv]) => `${kk}=${typeof vv === 'number' ? vv.toFixed(2) : String(vv).slice(0, 8)}`)
          .join(', ');
        return `${k}: {${inner}}`;
      }
      return `${k}=${v}`;
    })
    .join(' · ');
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}m${(s % 60).toString().padStart(2, '0')}s`;
}
