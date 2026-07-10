import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis } from 'recharts';

interface Props {
  history: number[];
  color?: string;
  height?: number;
}

/**
 * Tiny inline area chart with no axes / grid — for "trend at a glance".
 * Fed with a plain number array.
 */
export function Sparkline({ history, color = 'hsl(221 83% 53%)', height = 40 }: Props) {
  if (!history || history.length < 2) return null;
  const data = history.map((v, i) => ({ i, v }));
  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="spark" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.35} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="i" hide />
          <Tooltip
            contentStyle={{ fontSize: 11, padding: '4px 8px' }}
            labelFormatter={() => ''}
            formatter={(v) => [Number(v).toFixed(3), 'weight']}
          />
          <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.5} fill="url(#spark)" isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
