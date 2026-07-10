import { Outlet } from 'react-router-dom';

import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';
import { useAppStore } from '@/store';
import { cn } from '@/lib/utils';

/**
 * Root layout: fixed sidebar on the left, topbar on top, routed content
 * in the main region via <Outlet />. Sidebar is a slide-over drawer on
 * mobile (controlled by `ui.sidebarOpen`).
 */
export function AppShell() {
  const sidebarOpen = useAppStore((s) => s.ui.sidebarOpen);
  const toggleSidebar = useAppStore((s) => s.toggleSidebar);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      {/* Desktop sidebar */}
      <div className="hidden md:block">
        <Sidebar />
      </div>

      {/* Mobile drawer */}
      {sidebarOpen ? (
        <div className="fixed inset-0 z-40 md:hidden">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={toggleSidebar}
            aria-hidden
          />
          <div
            className={cn(
              'absolute inset-y-0 left-0 z-50 shadow-lg transition-transform',
              sidebarOpen ? 'translate-x-0' : '-translate-x-full',
            )}
          >
            <Sidebar onNavigate={toggleSidebar} />
          </div>
        </div>
      ) : null}

      <div className="flex flex-1 flex-col overflow-hidden">
        <Topbar />
        <main className="flex-1 overflow-y-auto bg-gradient-to-b from-background to-muted/20 p-4 md:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
