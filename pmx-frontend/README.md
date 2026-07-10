# pmx-frontend

React frontend for the EIx Engineering Intelligence. Talks to the FastAPI
backend at `predictive_maintenance_agentic/api/main.py`.

## Stack

- **Vite** + **React 18** + **TypeScript**
- **Tailwind CSS** + **shadcn/ui** primitives
- **React Router v6** for routing
- **Zustand** for client state (sliced per feature)
- **TanStack Query** for server state (cache + polling)
- **Plotly.js** for the 3D stress-field visualization
- **Recharts** for the physics-vs-ML bar charts
- **Sonner** for toasts, **Lucide** for icons

## Setup

```bash
cd pmx-frontend
cp .env.example .env
npm install
npm run dev
```

Then open http://127.0.0.1:5173.

The backend must be running on `http://127.0.0.1:8000`:

```bash
# from the repo root
python -m uvicorn predictive_maintenance_agentic.api.main:app --host 127.0.0.1 --port 8000
```

The dev server proxies `/api/*` to the backend, so no CORS config is
needed in dev.

## Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start Vite dev server on port 5173 |
| `npm run build` | Type-check + production build to `dist/` |
| `npm run preview` | Preview the production build |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run lint` | ESLint over `src/` |
| `npm run gen:api-types` | Regenerate `src/types/api.generated.ts` from the running backend's OpenAPI schema |

## Project structure

```
src/
  main.tsx            Entry — mounts <RouterProvider> + <QueryClientProvider>
  routes/
    router.tsx        createBrowserRouter definition (all routes)
  views/              One file per route (7 total + NotFound)
  components/
    layout/           AppShell, Sidebar, Topbar
    forms/            Sensor input, file upload, table editor
    cycle/            Detail-view sub-panels (Anomaly, Judge, Critic, ...)
    fleet/            Fleet-summary widgets
    common/           StatusBadge, MetricCard, ErrorBoundary, ...
    ui/               shadcn primitives
  hooks/              Feature-scoped hooks wrapping React Query
  services/           axios instance + queryClient
  store/              Zustand slices (session, form, cycles, fleet, ui)
  types/              domain.ts (hand-written) + api.generated.ts
  lib/                utils, constants, formatters, sensor defaults
  styles/             globals.css (Tailwind + theme tokens)
```

## Routes

| Path | View | Purpose |
| --- | --- | --- |
| `/` | DashboardHome | Landing tiles + recent cycles + fleet health |
| `/analyze/new` | NewAnalysisView | Configure & start a cycle |
| `/cycles` | CycleListView | Browse all cycles |
| `/cycles/:cycleId` | CycleDetailView | Full analyst report (with stage stepper) |
| `/cycles/:cycleId/approve` | EngineerApprovalView | Human-in-the-loop approval |
| `/fleet` | FleetView | Per-asset fleet-memory summaries |
| `/settings` | SettingsView | API URL, engineer ID, theme |

## State management

Server state → **TanStack Query** (cache, polling, retries, invalidation).
Client state → **Zustand**, split into 5 slices in `src/store/`:

- `sessionSlice` — user prefs (persisted): API URL, engineer ID, theme
- `analyzeFormSlice` — the "new analysis" draft (persisted)
- `cyclesSlice` — normalized cache of cycle snapshots (in-memory)
- `fleetSlice` — fleet-status cache (in-memory)
- `uiSlice` — sidebar, active stepper stage (in-memory)

Only `session` + `form` survive reloads; everything else re-syncs from the
backend on refresh.

## Migration status

This is the initial scaffold. Views + panels are implemented in phases:

- **Phase 4a** (this commit): skeleton — router, layout, empty view stubs, store slices, API client, hooks
- **Phase 4b**: DashboardHome + FleetView (read-only, easy wins)
- **Phase 4c**: NewAnalysisView (the 3-mode sensor input)
- **Phase 4d**: CycleDetailView shell + StageStepper + polling
- **Phase 4e**: Per-stage panels (Anomaly, Stress3D, Physics/ML, Dialogue, Judge, Calibration, Action, Critic, Trace)
- **Phase 4f**: EngineerApprovalView + optimistic update polish
- **Phase 4g**: Error boundaries, toast polish, responsive tuning
