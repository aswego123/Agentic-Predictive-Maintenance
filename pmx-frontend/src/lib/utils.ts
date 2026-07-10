import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Tailwind class-merging helper used by all shadcn-generated components.
 * See https://ui.shadcn.com/docs/installation/manual for the canonical pattern.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
