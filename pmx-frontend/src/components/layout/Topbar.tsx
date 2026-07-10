import { useLocation } from 'react-router-dom';
import { Menu } from 'lucide-react';

import { useHealth } from '@/hooks/useHealth';
import { Button } from '@/components/ui/button';
import { useAppStore } from '@/store';

const ROUTE_TITLES: Record<string, string> = {
  '/': 'Dashboard',
  '/analyze/new': 'New analysis',
  '/cycles': 'Cycles',
  '/fleet': 'Fleet',
  '/settings': 'Settings',
};

function titleForPath(path: string) {
  if (ROUTE_TITLES[path]) return ROUTE_TITLES[path];
  if (path.startsWith('/cycles/') && path.endsWith('/approve')) return 'Engineer approval';
  if (path.startsWith('/cycles/')) return 'Cycle detail';
  return '';
}

export function Topbar() {
  const { data, isError, isLoading } = useHealth();
  const location = useLocation();
  const toggleSidebar = useAppStore((s) => s.toggleSidebar);

  const status: 'ok' | 'down' | 'loading' = isLoading
    ? 'loading'
    : isError || !data
      ? 'down'
      : 'ok';

  const dotColor =
    status === 'ok'
      ? 'bg-emerald-500'
      : status === 'down'
        ? 'bg-red-500'
        : 'bg-yellow-500';

  return (
    <header className="flex h-14 items-center justify-between border-b bg-card/80 px-4 backdrop-blur md:px-6">
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 md:hidden"
          onClick={toggleSidebar}
          aria-label="Toggle menu"
        >
          <Menu className="h-4 w-4" />
        </Button>
        <div className="text-sm font-medium text-muted-foreground">
          {titleForPath(location.pathname)}
        </div>
      </div>

      <div className="flex items-center gap-2 rounded-full border bg-background/60 px-2.5 py-1 text-xs">
        <span className={`inline-block h-2 w-2 rounded-full ${dotColor} ${status === 'ok' ? 'animate-soft-pulse' : ''}`} />
        <span className="text-muted-foreground">
          {status === 'ok'
            ? <>API <span className="font-medium text-foreground">up</span> · {data?.simulation_adapter ?? 'synthetic'}</>
            : status === 'down'
              ? 'API unreachable'
              : 'checking…'}
        </span>
      </div>
    </header>
  );
}
