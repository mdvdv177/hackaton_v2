import type { Risk, Vehicle } from './types';

export const riskLabels: Record<Risk, string> = {
  red: 'Высокий риск', yellow: 'Внимание', green: 'Низкий риск', gray: 'Нет прогноза',
};
export const riskOrder: Record<Risk, number> = { red: 0, yellow: 1, green: 2, gray: 3 };

export function vehicleRisk(vehicle: Vehicle): Risk {
  if (vehicle.stale || !predictionIsCurrent(vehicle.prediction)) return 'gray';
  return vehicle.prediction?.risk || 'gray';
}

export function parseTime(value?: string | null): Date | null {
  if (!value) return null;
  const normalized = value.replace(' ', 'T');
  const date = new Date(/Z$|[+-]\d\d:\d\d$/.test(normalized) ? normalized : `${normalized}Z`);
  return Number.isNaN(date.valueOf()) ? null : date;
}

export function time(value?: string | null, seconds = false): string {
  const date = parseTime(value);
  return date ? new Intl.DateTimeFormat('ru-RU', {
    timeZone: 'Europe/Moscow', hour: '2-digit', minute: '2-digit',
    ...(seconds ? { second: '2-digit' } : {}),
  }).format(date) : '—';
}

export function day(value?: string | null): string {
  const date = parseTime(value);
  return date ? new Intl.DateTimeFormat('ru-RU', {
    timeZone: 'Europe/Moscow', day: 'numeric', month: 'long', year: 'numeric',
  }).format(date) : 'Время не задано';
}

export function delay(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '—';
  const seconds = Math.round(Math.abs(value));
  const sign = value >= 0 ? '+' : '−';
  return seconds < 60 ? `${sign}${seconds} с` : `${sign}${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

export function percent(value?: number | null): string {
  return value == null || !Number.isFinite(value) ? '—' : `${Math.round(value * 100)}%`;
}

export function age(value?: number | null): string {
  if (value == null) return 'нет позиции';
  if (value < 60) return `${Math.max(0, Math.round(value))} с назад`;
  return `${Math.floor(value / 60)} мин назад`;
}

export function number(value?: number | null, digits = 0): string {
  return value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('ru-RU', { maximumFractionDigits: digits });
}

export function predictionIsCurrent(prediction: import('./types').Prediction | null | undefined): boolean {
  return !!prediction && prediction.current_prediction !== false && !['monitoring', 'stale', 'prediction_stale', 'retrospective'].includes(prediction.display_state || '') && prediction.timing_status !== 'retrospective';
}

export function stopName(name?: string, id?: string): string {
  return name && !/^\d+$/.test(name) ? name : `Остановка без адреса · ${id || name || '—'}`;
}

export function segmentName(segment?: import('./types').Segment | null): string {
  return segment ? `${stopName(segment.from_name, segment.from_visit_id)} → ${stopName(segment.to_name, segment.to_visit_id)}` : 'Участок не определён';
}

export function freshnessLabel(prediction?: import('./types').Prediction | null, offline = false): string {
  if (offline) return 'Нет связи · последнее состояние';
  if (!prediction) return 'Нет текущего прогноза';
  if (prediction.timing_status === 'retrospective') return 'Ретроспективный расчёт';
  if (['stale', 'prediction_stale'].includes(prediction.display_state || '')) return 'Прогноз устарел';
  if (prediction.display_state === 'monitoring' || prediction.current_prediction === false) return 'Наблюдение · прежняя цель';
  return 'Текущий прогноз';
}
