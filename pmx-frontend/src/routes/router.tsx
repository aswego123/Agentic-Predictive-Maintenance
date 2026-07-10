import { createBrowserRouter, Navigate } from 'react-router-dom';

import { AppShell } from '@/components/layout/AppShell';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';
import { DashboardHome } from '@/views/DashboardHome';
import { NewAnalysisView } from '@/views/NewAnalysisView';
import { CycleListView } from '@/views/CycleListView';
import { CycleDetailView } from '@/views/CycleDetailView';
import { EngineerApprovalView } from '@/views/EngineerApprovalView';
import { FleetView } from '@/views/FleetView';
import { SettingsView } from '@/views/SettingsView';
import { NotFoundView } from '@/views/NotFoundView';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    errorElement: <ErrorBoundary />,
    children: [
      { index: true, element: <DashboardHome /> },
      { path: 'analyze/new', element: <NewAnalysisView /> },
      { path: 'cycles', element: <CycleListView /> },
      { path: 'cycles/:cycleId', element: <CycleDetailView /> },
      { path: 'cycles/:cycleId/approve', element: <EngineerApprovalView /> },
      { path: 'fleet', element: <FleetView /> },
      { path: 'settings', element: <SettingsView /> },
      { path: '404', element: <NotFoundView /> },
      { path: '*', element: <Navigate to="/404" replace /> },
    ],
  },
]);
