import { useEffect } from 'react';
import { toast } from 'sonner';

import { useAppStore } from '@/store';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import type { Theme } from '@/store/sessionSlice';

export function SettingsView() {
  const session = useAppStore((s) => s.session);
  const setApiUrl = useAppStore((s) => s.setApiUrl);
  const setEngineerId = useAppStore((s) => s.setEngineerId);
  const setTheme = useAppStore((s) => s.setTheme);
  const resetForm = useAppStore((s) => s.resetForm);

  // Apply theme class on <html> whenever it changes.
  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('dark', session.theme === 'dark');
  }, [session.theme]);

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground">Local preferences, persisted per browser.</p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Backend</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="api-url">API base URL</Label>
            <Input
              id="api-url"
              value={session.apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="/api"
            />
            <p className="text-[11px] text-muted-foreground">
              In dev, leave as <code>/api</code> — the Vite proxy forwards to
              your FastAPI backend. Set an absolute URL only if you're calling
              a different host.
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Engineer</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="engineer-id">Engineer ID</Label>
            <Input
              id="engineer-id"
              value={session.engineerId}
              onChange={(e) => setEngineerId(e.target.value)}
              placeholder="eng-42"
            />
            <p className="text-[11px] text-muted-foreground">
              Pre-fills the approval form. Also submitted alongside every
              engineer decision.
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Appearance</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label>Theme</Label>
            <Select value={session.theme} onValueChange={(v) => setTheme(v as Theme)}>
              <SelectTrigger className="w-40"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="light">Light</SelectItem>
                <SelectItem value="dark">Dark</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Danger zone</CardTitle>
        </CardHeader>
        <CardContent>
          <Button
            variant="outline"
            onClick={() => {
              resetForm();
              toast.success('Analysis form draft reset');
            }}
          >
            Reset analysis form draft
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
