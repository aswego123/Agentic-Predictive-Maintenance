import { Link } from 'react-router-dom';

export function NotFoundView() {
  return (
    <div className="mx-auto max-w-md space-y-4 py-24 text-center">
      <h1 className="text-3xl font-semibold">404</h1>
      <p className="text-sm text-muted-foreground">Route not found.</p>
      <Link to="/" className="text-sm text-primary underline">
        Back to dashboard
      </Link>
    </div>
  );
}
