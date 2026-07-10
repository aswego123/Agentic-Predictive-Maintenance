import type { ComponentType } from 'react';
import { NavLink, Link } from 'react-router-dom';
import {
  LayoutDashboard,
  PlayCircle,
  ListOrdered,
  Factory,
  Settings,
  Activity,
  Clock,
  X,
} from 'lucide-react';

import { cn } from '@/lib/utils';
import { useAppStore } from '@/store';

type NavItem = {
  to: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
  end?: boolean;
};

const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/analyze/new', label: 'New analysis', icon: PlayCircle },
  { to: '/cycles', label: 'Cycles', icon: ListOrdered },
  { to: '/fleet', label: 'Fleet', icon: Factory },
  { to: '/settings', label: 'Settings', icon: Settings },
];

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const lastCycleId = useAppStore((s) => s.session.lastCycleId);
  const engineerId = useAppStore((s) => s.session.engineerId);
  const running = useAppStore((s) => Object.values(s.cycles.polling).some(Boolean));

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r bg-card">
      <div className="flex h-14 items-center justify-between border-b px-4">
        <div className="flex items-center gap-2">
          <div className="grid h-7 w-7 place-items-center rounded-md bg-primary text-primary-foreground shadow-sm">
            <Activity className="h-4 w-4" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold">EIx</div>
            <div className="text-[10px] text-muted-foreground">Engineering Intelligence</div>
          </div>
        </div>
        {onNavigate ? (
          <button
            type="button"
            onClick={onNavigate}
            className="rounded-md p-1 text-muted-foreground hover:bg-accent md:hidden"
            aria-label="Close menu"
          >
            <X className="h-4 w-4" />
          </button>
        ) : null}
      </div>

      <nav className="flex-1 space-y-1 p-2">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            onClick={onNavigate}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground',
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
            {to === '/cycles' && running ? (
              <span className="ml-auto flex h-1.5 w-1.5 animate-soft-pulse rounded-full bg-primary" />
            ) : null}
          </NavLink>
        ))}
      </nav>

      {lastCycleId ? (
        <Link
          to={`/cycles/${lastCycleId}`}
          onClick={onNavigate}
          className="mx-2 mb-2 rounded-lg border bg-muted/30 p-3 transition-colors hover:bg-accent"
        >
          <div className="mb-1 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            <Clock className="h-3 w-3" />
            Last cycle
          </div>
          <div className="truncate font-mono text-xs">{lastCycleId}</div>
        </Link>
      ) : null}

      <div className="border-t p-3">
        <div className="flex items-center gap-2">
          <div className="grid h-7 w-7 place-items-center rounded-full bg-secondary text-xs font-semibold uppercase">
            {engineerId?.slice(0, 2) || 'EN'}
          </div>
          <div className="min-w-0 flex-1">
            <div className="truncate text-xs font-medium">{engineerId || 'engineer'}</div>
            <div className="text-[10px] text-muted-foreground">Session-only</div>
          </div>
        </div>
      </div>
    </aside>
  );
}
