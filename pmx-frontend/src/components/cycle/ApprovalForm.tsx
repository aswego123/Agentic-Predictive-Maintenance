import { useState } from 'react';
import { CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { useApproveCycle } from '@/hooks/useApproveCycle';
import { useAppStore } from '@/store';

interface Props {
  cycleId: string;
  onResolved?: () => void;
}

export function ApprovalForm({ cycleId, onResolved }: Props) {
  const engineerId = useAppStore((s) => s.session.engineerId);
  const setEngineerId = useAppStore((s) => s.setEngineerId);
  const approve = useApproveCycle();

  const [approved, setApproved] = useState(true);
  const [notes, setNotes] = useState('Calibration looks reasonable.');

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await approve.mutateAsync({
        cycle_id: cycleId,
        approved,
        engineer_id: engineerId,
        notes,
      });
      onResolved?.();
    } catch (exc) {
      toast.error(exc instanceof Error ? exc.message : String(exc));
    }
  };

  return (
    <Card className="border-amber-500/40 bg-amber-500/5">
      <CardHeader>
        <CardTitle className="text-base text-amber-900 dark:text-amber-200">
          ⏸ Awaiting engineer approval
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Alert variant="warning" className="mb-4">
          <AlertTitle>Graph paused</AlertTitle>
          <AlertDescription>
            The graph is stopped at <code>engineer_approval</code>. Submit a
            decision to resume; the pipeline will re-run downstream nodes with
            your input.
          </AlertDescription>
        </Alert>

        <form onSubmit={submit} className="space-y-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="engineer-id">Engineer ID</Label>
              <Input
                id="engineer-id"
                value={engineerId}
                onChange={(e) => setEngineerId(e.target.value)}
                required
              />
            </div>
            <div className="space-y-1.5">
              <Label>Decision</Label>
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant={approved ? 'default' : 'outline'}
                  className="flex-1"
                  onClick={() => setApproved(true)}
                >
                  <CheckCircle2 className="mr-1 h-4 w-4" />
                  Approve
                </Button>
                <Button
                  type="button"
                  variant={!approved ? 'destructive' : 'outline'}
                  className="flex-1"
                  onClick={() => setApproved(false)}
                >
                  <XCircle className="mr-1 h-4 w-4" />
                  Reject
                </Button>
              </div>
            </div>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="notes">Notes</Label>
            <Textarea id="notes" value={notes} onChange={(e) => setNotes(e.target.value)} rows={3} />
          </div>

          <div className="flex justify-end">
            <Button type="submit" disabled={approve.isPending}>
              {approve.isPending ? (
                <>
                  <Loader2 className="mr-1 h-4 w-4 animate-spin" /> Submitting…
                </>
              ) : (
                'Submit decision'
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
