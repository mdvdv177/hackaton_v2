import { useId } from 'react';
import { number, parseTime, time } from '../format';

export default function TelemetryChart({ samples, unit }: { samples: { at: string; value: number | null }[]; unit: string }) {
  const gradientId = useId().replace(/:/g, '');
  const points = samples.map(s => ({ ...s, x: parseTime(s.at)?.valueOf() })).filter((s): s is { at: string; value: number; x: number } => s.x != null && s.value != null && Number.isFinite(s.value)).sort((a, b) => a.x - b.x);
  if (!points.length) return <div className="chart-empty"><IconChart /><span>История появится по мере<br />поступления данных</span></div>;
  const minX = points[0].x;
  const maxX = points[points.length - 1].x;
  const minY = Math.min(0, ...points.map(p => p.value));
  const maxY = Math.max(10, ...points.map(p => p.value)) * 1.1;
  const projectX = (x: number) => 34 + ((x - minX) / (maxX - minX || 1)) * 228;
  const projectY = (value: number) => 112 - ((value - minY) / (maxY - minY || 1)) * 87;
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${projectX(p.x).toFixed(1)},${projectY(p.value).toFixed(1)}`).join(' ');
  const area = `${path} L${projectX(maxX)},112 L34,112Z`;
  return <svg className="telemetry-chart" viewBox="0 0 275 144" role="img" aria-label={`История: ${points.length} измерений, ${unit}`}>
    <defs><linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2f907e" stopOpacity="0.17"/><stop offset="100%" stopColor="#2f907e" stopOpacity="0.01"/></linearGradient></defs>
    <text x="4" y="10" className="chart-unit">{unit}</text>
    {[0, 0.5, 1].map(ratio => { const value = minY + (maxY - minY) * ratio; const y = projectY(value); return <g key={ratio}><line x1="34" x2="262" y1={y} y2={y} stroke="#e9eeed" strokeDasharray="3 3"/><text x="27" y={y + 3} textAnchor="end" className="chart-label">{number(value)}</text></g>; })}
    <path d={area} fill={`url(#${gradientId})`} /><path d={path} stroke="#2f907e" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    {points.length === 1 && <circle cx={projectX(points[0].x)} cy={projectY(points[0].value)} r="3" fill="#2f907e" />}
    <text x="34" y="133" className="chart-label">{time(points[0].at)}</text><text x="262" y="133" textAnchor="end" className="chart-label">{time(points[points.length - 1].at)}</text>
  </svg>;
}

function IconChart() {
  return <svg width="52" height="25" viewBox="0 0 52 25" fill="none" aria-hidden="true"><path d="m1 20 10-5 9 3L30 5l10 5L51 1" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"/></svg>;
}
