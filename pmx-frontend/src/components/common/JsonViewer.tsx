import { cn } from '@/lib/utils';

/**
 * Pretty-print a JSON blob in a scrollable pre. Replaces Streamlit's
 * `st.json` / `st.code(json, language="json")`.
 */
export function JsonViewer({ data, className, maxHeight = 320 }: { data: unknown; className?: string; maxHeight?: number }) {
  const text =
    typeof data === 'string' ? data : JSON.stringify(data, null, 2) ?? 'null';
  return (
    <pre
      className={cn(
        'overflow-auto rounded-md border bg-muted/40 p-3 font-mono text-[11px] leading-relaxed text-foreground/90',
        className,
      )}
      style={{ maxHeight }}
    >
      {text}
    </pre>
  );
}
