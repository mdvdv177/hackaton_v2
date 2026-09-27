import { useEffect, useMemo, useState } from 'react';
import { CircleMarker, MapContainer, Marker, Polyline, TileLayer, Tooltip, useMap } from 'react-leaflet';
import { divIcon, latLngBounds } from 'leaflet';
import type { LatLngTuple } from 'leaflet';
import type { Network, Prediction, Risk, SegmentRisk, Snapshot, Vehicle } from '../types';
import { age, delay, riskLabels, segmentName, vehicleRisk } from '../format';
import Icon from './Icon';

function validPosition(vehicle: Vehicle): boolean {
  return vehicle.lat != null && vehicle.lon != null && Number.isFinite(vehicle.lat) && Number.isFinite(vehicle.lon);
}

function Viewport({ vehicles, selectedId, runId, centerSignal, network }: { vehicles: Vehicle[]; selectedId: string | null; runId: string | undefined; centerSignal: number; network: Network | null }) {
  const map = useMap();
  const [fittedRun, setFittedRun] = useState<string | undefined>();
  const version = `${runId || 'idle'}:${network?.network_version || 'none'}`;
  const networkPositions = useMemo<LatLngTuple[]>(() => (network?.segments || []).flatMap(s => s.geometry?.coordinates.map(([lon, lat]) => [lat, lon] as LatLngTuple) || []), [network]);
  useEffect(() => {
    const positioned = vehicles.filter(validPosition);
    if ((positioned.length || networkPositions.length) && fittedRun !== version) {
      map.fitBounds(latLngBounds([...networkPositions, ...positioned.map(v => [v.lat!, v.lon!] as LatLngTuple)]), { padding: [35, 35], maxZoom: 13 });
      setFittedRun(version);
    }
  }, [vehicles, map, version, fittedRun]);
  useEffect(() => {
    const selected = vehicles.find(v => v.id === selectedId);
    if (selected && validPosition(selected)) {
      map.panTo([selected.lat!, selected.lon!], { animate: true });
    }
    // Moving telemetry must not recenter a map the dispatcher is exploring.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, map]);
  useEffect(() => {
    if (!centerSignal) return;
    const positioned = vehicles.filter(validPosition);
    if (positioned.length || networkPositions.length) map.fitBounds(latLngBounds([...networkPositions, ...positioned.map(v => [v.lat!, v.lon!] as LatLngTuple)]), { padding: [35, 35], maxZoom: 13 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [centerSignal, map]);
  useEffect(() => {
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(map.getContainer());
    return () => observer.disconnect();
  }, [map]);
  return null;
}

function VehicleMarker({ vehicle, selected, onSelect }: { vehicle: Vehicle; selected: boolean; onSelect: () => void }) {
  const risk = vehicleRisk(vehicle);
  const icon = useMemo(() => divIcon({
    className: `vehicle-marker vehicle-${vehicle.id.replace(/[^a-zA-Z0-9_-]/g, '')} marker-${risk} ${selected ? 'marker-selected' : ''}`,
    html: '<span class="marker-inner"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="3" width="14" height="16" rx="3"/><path d="M5 11h14M8 19v2m8-2v2M9 6h6M8 15h1m6 0h1"/></svg></span>',
    iconSize: [34, 34], iconAnchor: [17, 17],
  }), [risk, selected]);
  return <Marker position={[vehicle.lat!, vehicle.lon!]} icon={icon} eventHandlers={{ click: onSelect }} zIndexOffset={selected ? 1000 : risk === 'red' ? 100 : 0}>
    <Tooltip direction="top" offset={[0, -18]}><strong>ТС {vehicle.id}</strong><br />{vehicle.stale ? 'Данные устарели' : `Прогноз: ${delay(vehicle.prediction?.prediction_delay_s)}`}<br /><span>Позиция: {age(vehicle.position_age_s)}</span></Tooltip>
  </Marker>;
}

const riskColors: Record<Risk, string> = { red: '#cc5a53', yellow: '#b58b32', green: '#388b72', gray: '#9ba8a4' };

export default function FleetMap({ vehicles, selectedId, prediction, network, segmentRisks, networkError, offline, onSelect, runId, isIdle }: {
  vehicles: Vehicle[]; selectedId: string | null; prediction?: Prediction | null; network: Network | null;
  segmentRisks?: Snapshot['segment_risks']; networkError: boolean; offline: boolean;
  onSelect: (id: string) => void; runId?: string; isIdle: boolean;
}) {
  const [showVisits, setShowVisits] = useState(true);
  const [centerSignal, setCenterSignal] = useState(0);
  const [tilesUnavailable, setTilesUnavailable] = useState(false);
  const positioned = vehicles.filter(validPosition);
  const risks: Record<string, SegmentRisk> = Array.isArray(segmentRisks)
    ? Object.fromEntries(segmentRisks.map(item => [item.segment_id || item.id || '', item]))
    : Object.fromEntries(Object.entries(segmentRisks || {}).map(([id, value]) => [id, typeof value === 'string' ? { risk: value } : value]));
  const segments = network?.segments || [];
  const visits = useMemo(() => {
    const seen = new Set<string>();
    return (network?.visits || []).filter(visit => {
      if (!Number.isFinite(visit.lat) || !Number.isFinite(visit.lon)) return false;
      const key = `${visit.lat}:${visit.lon}`;
      if (seen.has(key)) return false;
      seen.add(key); return true;
    });
  }, [network]);
  const focusSegment = prediction?.target_segment?.id;
  const supplied = network?.geometry_kind === 'supplied_route';

  return <section className="map-panel panel" aria-label="Карта транспорта">
    <div className="map-toolbar"><div><span className="tiny-label">{supplied ? 'МАРШРУТНАЯ СЕТЬ' : 'СЕТЬ ПЛАНОВЫХ УЧАСТКОВ'}</span><span className="map-count">{positioned.length} ТС · {segments.length} участков</span></div></div>
    <div className="map-canvas">
      <MapContainer center={[55.77, 37.54]} zoom={11} zoomControl={false} attributionControl scrollWheelZoom>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' eventHandlers={{ tileerror: () => setTilesUnavailable(true), tileload: () => setTilesUnavailable(false) }} />
        <Viewport vehicles={vehicles} selectedId={selectedId} runId={runId} centerSignal={centerSignal} network={network} />
        {showVisits && segments.map(segment => {
          const coordinates = segment.geometry?.coordinates;
          if (!coordinates || coordinates.length < 2 || coordinates.some(point => !point.every(Number.isFinite))) return null;
          const state = risks[segment.id];
          const risk: Risk = offline ? 'gray' : state?.risk || 'gray';
          const focused = segment.id === focusSegment;
          return <Polyline key={`${segment.id}:${risk}:${focused}`} className={`network-segment segment-${risk}${focused ? ' segment-focused' : ''}`} positions={coordinates.map(([lon, lat]) => [lat, lon] as LatLngTuple)} pathOptions={{
            className: `network-segment segment-${risk}${focused ? ' segment-focused' : ''}`,
            color: riskColors[risk], weight: focused ? 6 : risk === 'gray' ? 2 : 4,
            opacity: focused ? 1 : risk === 'gray' ? .35 : .85,
            dashArray: segment.geometry_kind === 'supplied_route' ? undefined : '5 5',
          }}><Tooltip><strong>{segmentName(segment)}</strong><br/>{riskLabels[risk]} · риск прибытия к конечной остановке<br/>{segment.geometry_kind === 'supplied_route' ? 'Импортированная геометрия' : 'Восстановленная схема'}</Tooltip></Polyline>;
        })}
        {showVisits && visits.map(visit => <CircleMarker key={visit.id} center={[visit.lat, visit.lon]} radius={2.5} pathOptions={{ color: '#667e76', fillColor: '#fff', fillOpacity: .75, weight: 1 }}><Tooltip>{visit.name}</Tooltip></CircleMarker>)}
        {positioned.map(vehicle => <VehicleMarker key={vehicle.id} vehicle={vehicle} selected={selectedId === vehicle.id} onSelect={() => onSelect(vehicle.id)} />)}
        {prediction?.target_lat != null && prediction?.target_lon != null && <CircleMarker center={[prediction.target_lat, prediction.target_lon]} radius={8} pathOptions={{ color: '#163f44', fillColor: '#b6e6d7', fillOpacity: 1, weight: 2 }}><Tooltip direction="top">Целевая остановка</Tooltip></CircleMarker>}
        <MapZoomControls />
      </MapContainer>
      <div className="map-actions"><button className="map-control" onClick={() => setCenterSignal(s => s + 1)} aria-label="Показать всю сеть" title="Показать всю сеть"><Icon name="target" size={19}/></button><button className={`map-control ${showVisits ? 'is-on' : ''}`} onClick={() => setShowVisits(s => !s)} aria-label="Показать сеть участков" aria-pressed={showVisits}><Icon name="layers" size={19}/></button></div>
      {isIdle && !segments.length && <div className="map-empty"><h3>Город в поле зрения</h3><p>Запустите replay или подключите NDTP.<br/>Сеть загружается независимо от выбора ТС.</p></div>}
      <div className="map-legend"><span><i className="dot red"/>Высокий риск</span><span><i className="dot yellow"/>Внимание</span><span><i className="dot green"/>Низкий риск</span><span><i className="dot gray"/>Нет текущего прогноза</span></div>
      {(tilesUnavailable || networkError) && <span className="tile-warning">{networkError ? 'Сеть временно недоступна' : 'Подложка недоступна · данные сохраняются'}</span>}
    </div>
    <div className="map-caption"><Icon name="info" size={14}/><span>{supplied ? 'Импортированные участки' : 'Пунктир — восстановленная схема'} · цвет — риск прибытия, не причина сбоя</span></div>
  </section>;
}

function MapZoomControls() {
  const map = useMap();
  return <div className="map-zoom"><button onClick={() => map.zoomIn()} aria-label="Приблизить карту">+</button><button onClick={() => map.zoomOut()} aria-label="Отдалить карту">−</button></div>;
}
