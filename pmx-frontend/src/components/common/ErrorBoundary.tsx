import { useRouteError, isRouteErrorResponse, Link } from 'react-router-dom';

/**
 * Router-level error boundary. Caught by React Router when a loader,
 * action, or render throws. Nested component errors should still be
 * wrapped in their own boundary inside the view — this is the
 * top-level fallback.
 */
export function ErrorBoundary() {
  const error = useRouteError();

  const title = isRouteErrorResponse(error)
    ? `${error.status} ${error.statusText}`
    : 'Something went wrong';

  const detail =
    error instanceof Error
      ? error.message
      : isRouteErrorResponse(error)
        ? (error.data as string) || 'Unhandled route error'
        : 'Unhandled error';

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-background p-6">
      <div className="max-w-md space-y-4 text-center">
        <h1 className="text-2xl font-semibold text-destructive">{title}</h1>
        <p className="text-sm text-muted-foreground">{detail}</p>
        <Link to="/" className="text-sm text-primary underline">
          Back to dashboard
        </Link>
      </div>
    </div>
  );
}
