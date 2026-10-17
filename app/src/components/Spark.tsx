/** Tiny day sparkline; the dashed line is the previous close. */
export default function Spark({
  values,
  prev,
  className,
}: {
  values: number[];
  prev: number | null;
  className?: string;
}) {
  if (values.length < 2) return <div className={className} />;
  const all = prev ? [...values, prev] : values;
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const W = 100;
  const H = 40;
  const y = (v: number) => H - 2 - ((v - lo) / span) * (H - 4);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * W},${y(v)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className={className} aria-hidden="true">
      {prev && (
        <line
          x1="0"
          x2={W}
          y1={y(prev)}
          y2={y(prev)}
          stroke="var(--muted)"
          strokeOpacity="0.5"
          strokeDasharray="2 2"
          vectorEffect="non-scaling-stroke"
        />
      )}
      <polyline
        points={pts}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
