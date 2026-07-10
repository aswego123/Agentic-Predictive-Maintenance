import { useParams, useNavigate, Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

import { useCycle } from '@/hooks/useCycle';
import { ApprovalForm } from '@/components/cycle/ApprovalForm';
import { StatusBadge } from '@/components/common/StatusBadge';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { EmptyState } from '@/components/common/EmptyState';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

/**
 * Focused, standalone approval page. The same ApprovalForm is embedded
 * inline in CycleDetailView; this route exists so an on-call engineer
 * can be linked directly to the decision without wading through the
 * full report.
 */
export function EngineerApprovalView() {
  const { cycleId } = useParams();
  const navigate = useNavigate();
  const cycleQuery = useCycle(cycleId);
  const cycle = cycleQuery.data;

  if (cycleQuery.isLoading) {
    return (
      <div className="mx-auto flex max-w-lg items-center gap-2 py-16 text-sm text-muted-foreground">
        <LoadingSpinner /> Loading cycle {cycleId}…
      </div>
    );
  }

  if (!cycle) {
    return (
      <div className="mx-auto max-w-lg py-16">
        <EmptyState title="Cycle not found" description={`No snapshot for ${cycleId}.`} />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <div>
        <Button asChild variant="ghost" size="sm">
          <Link to={`/cycles/${cycle.cycle_id}`}>
            <ArrowLeft className="mr-1 h-4 w-4" />
            Back to cycle detail
          </Link>
        </Button>
      </div>

      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Engineer approval</h1>
        <p className="font-mono text-xs text-muted-foreground">{cycle.cycle_id}</p>
        <div className="mt-2">
          <StatusBadge status={cycle.status} />
        </div>
      </header>

      {cycle.awaiting_engineer_approval ? (
        <ApprovalForm
          cycleId={cycle.cycle_id}
          onResolved={() => navigate(`/cycles/${cycle.cycle_id}`)}
        />
      ) : (
        <Alert>
          <AlertTitle>Nothing to decide</AlertTitle>
          <AlertDescription>
            This cycle is not currently paused at the engineer-approval
            interrupt. Its status is <code>{cycle.status}</code>.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
