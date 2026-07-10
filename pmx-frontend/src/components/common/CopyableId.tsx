import { useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { toast } from 'sonner';

import { cn } from '@/lib/utils';

interface Props {
  value: string;
  className?: string;
  label?: string;
}

/**
 * Compact `<code>` chip with a copy-on-click button. Great for cycle IDs
 * that users want to bookmark or share.
 */
export function CopyableId({ value, className, label }: Props) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      toast.success(`${label ?? 'ID'} copied`);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error('Clipboard write failed');
    }
  };

  return (
    <button
      type="button"
      onClick={copy}
      className={cn(
        'group inline-flex items-center gap-1.5 rounded-md border bg-muted/40 px-2 py-0.5 font-mono text-xs transition-colors hover:bg-muted',
        className,
      )}
      title="Copy to clipboard"
    >
      <span className="max-w-[16rem] truncate">{value}</span>
      {copied ? (
        <Check className="h-3 w-3 text-emerald-600" />
      ) : (
        <Copy className="h-3 w-3 text-muted-foreground group-hover:text-foreground" />
      )}
    </button>
  );
}
