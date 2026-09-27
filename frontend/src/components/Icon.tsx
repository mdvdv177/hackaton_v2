import type { CSSProperties } from 'react';

export type IconName = 'grid' | 'bus' | 'bell' | 'pulse' | 'arrow' | 'search' | 'chevron' | 'clock' | 'play' | 'pause' | 'refresh' | 'pin' | 'check' | 'close' | 'layers' | 'target' | 'signal' | 'info' | 'external' | 'chart' | 'expand';
const paths: Record<IconName, React.ReactNode> = {
  grid: <><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></>,
  bus: <><rect x="5" y="3" width="14" height="16" rx="3"/><path d="M5 11h14M8 19v2m8-2v2M9 6h6"/><path d="M8 15h1m6 0h1"/></>,
  bell: <><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/></>,
  pulse: <><path d="M2 12h5l3-8 4 16 3-8h5"/></>,
  arrow: <><path d="M4 12h16m-6-6 6 6-6 6"/></>,
  search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/></>,
  chevron: <path d="m9 5 7 7-7 7"/>,
  clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
  play: <path d="m8 4 12 8-12 8Z"/>,
  pause: <><path d="M8 4v16M16 4v16"/></>,
  refresh: <><path d="M20 7v5h-5M4 17v-5h5"/><path d="M6 6a8 8 0 0 1 13 1l1 5M4 12l1 5a8 8 0 0 0 13 1"/></>,
  pin: <><path d="M19 10c0 5-7 11-7 11S5 15 5 10a7 7 0 0 1 14 0Z"/><circle cx="12" cy="10" r="2"/></>,
  check: <path d="m5 12 4 4L19 6"/>,
  close: <path d="m6 6 12 12M6 18 18 6"/>,
  layers: <><path d="m12 3 10 5-10 5L2 8Zm-10 9 10 5 10-5M2 16l10 5 10-5"/></>,
  target: <><circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/><path d="M12 1v4m0 14v4M1 12h4m14 0h4"/></>,
  signal: <><path d="M5 19v-3m5 3v-7m5 7V8m5 11V4"/></>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/></>,
  external: <><path d="M14 3h7v7m0-7L11 13M10 4H4v16h16v-6"/></>,
  chart: <><path d="M3 3v18h18M7 15l4-5 4 3 6-7"/></>,
  expand: <><path d="M9 3H3v6m12-6h6v6M3 15v6h6m12-6v6h-6"/></>,
};

export default function Icon({ name, size = 20, style, className = '' }: { name: IconName; size?: number; style?: CSSProperties; className?: string }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={style} className={className}>{paths[name]}</svg>;
}

export function Brand({ small = false }: { small?: boolean }) {
  return <div className={`brand ${small ? 'brand-small' : ''}`}><svg viewBox="0 0 38 38" fill="none" aria-hidden="true"><path d="M8 29V9h8v13h14V8" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"/><circle cx="30" cy="30" r="2.5" fill="currentColor"/></svg>{!small && <span>контур<span className="brand-dot">.</span></span>}</div>;
}
