import { Component, type ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';

interface Props {
  name: string;
  children: ReactNode;
}

interface State {
  hasError: boolean;
  message?: string;
}

/**
 * Component-scoped error boundary. Prevents one crashing panel
 * (bad JSON shape, undefined field on a legacy cycle, etc.) from
 * taking down the whole cycle-detail page.
 */
export class PanelErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(err: Error): State {
    return { hasError: true, message: err.message };
  }

  componentDidCatch(err: Error) {
    // Log to console only — sonner toast would spam.
    // eslint-disable-next-line no-console
    console.error(`Panel ${this.props.name} crashed:`, err);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          <div className="mb-1 flex items-center gap-2 font-medium">
            <AlertTriangle className="h-4 w-4" />
            {this.props.name} failed to render
          </div>
          <div className="text-xs text-destructive/80">
            {this.state.message ?? 'Unknown error'}
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
