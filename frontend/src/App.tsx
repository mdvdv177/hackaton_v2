import { useCallback, useEffect, useMemo, useRef, useState, useLayoutEffect } from 'react';
import { api, post } from './api';
import { age, day, delay, freshnessLabel, number, percent, predictionIsCurrent, riskLabels, riskOrder, segmentName, stopName, time, vehicleRisk } from './format';
import type { Factor, Incident, Network, Prediction, ReplayMode, Risk, Scenario, Snapshot, Vehicle, VehicleDetails } from './types';
import Icon, { Brand } from './components/Icon';
import FleetMap from './components/FleetMap';
import TelemetryChart from './components/TelemetryChart';

const emptySnapshot: Snapshot = { run: null, vehicles: [], incidents: [], system: {} };
const factorLabels: Record<string, string> = {
  cur_dev_s: 'Текущее отклонение', last_speed: 'Последняя скорость',
  speed_mean_60s: 'Средняя скорость за минуту', speed_mean_180s: 'Средняя скорость за 3 минуты',
  speed_mean_300s: 'Средняя скорость за 5 минут', speed_mean_900s: 'Средняя скорость за 15 минут',
  distance_to_target_m: 'Расстояние до цели', horizon_s: 'До планового прибытия',
  stop_duration_s: 'Длительность простоя', segment_speed_kmh: 'Средняя скорость на участке',
};

function useDispatcher() {
  const [snapshot, setSnapshot] = useState<Snapshot>(emptySnapshot);
  const [connection, setConnection] = useState('connecting');
  const [loadError, setLoadError] = useState(false);
  const [receivedAt, setReceivedAt] = useState<number | null>(null);
  const [wallTime, setWallTime] = useState(Date.now());
  const inFlight = useRef(false);
  const mounted = useRef(true);
  const accepted = useRef<Snapshot>(emptySnapshot);
  const accept = useCallback((next: Snapshot) => {
    if (!mounted.current || !Array.isArray(next.vehicles) || !Array.isArray(next.incidents)) return;
    const previous = accepted.current;
    const nextPublication = Date.parse(next.published_at || '');
    const previousPublication = Date.parse(previous.published_at || '');
    const comparablePublications = Number.isFinite(nextPublication) && Number.isFinite(previousPublication);
    if (comparablePublications && nextPublication < previousPublication) return;
    // A recovered run keeps its ID while the process-local SSE sequence restarts.
    // A later publication proves the new sequence is fresh; late old responses
    // are rejected above even when they carry a larger sequence number.
    if (next.run?.id === previous.run?.id && next.event_id != null && previous.event_id != null && next.event_id < previous.event_id
      && !(comparablePublications && nextPublication > previousPublication)) return;
    accepted.current = next;
    setSnapshot(next); setLoadError(false); setReceivedAt(Date.now());
  }, []);
  const refresh = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try { accept(await api<Snapshot>('/snapshot', { signal: AbortSignal.timeout(5000) })); }
    catch { if (mounted.current) setLoadError(true); }
    finally { inFlight.current = false; }
  }, [accept]);
  useEffect(() => {
    mounted.current = true;
    void refresh();
    let stream: EventSource;
    let retry: number | undefined;
    let retryDelay = 1000;
    const connect = () => {
      const source = new EventSource('/api/v1/events');
      stream = source;
      source.onopen = () => {
        if (!mounted.current || source !== stream) return;
        retryDelay = 1000; setConnection('connected'); void refresh();
      };
      source.onerror = () => {
        if (!mounted.current || source !== stream) return;
        setConnection('reconnecting');
        // Native EventSource retries CONNECTING itself, but a non-200 response
        // can close it permanently while the Backend container is recreated.
        if (source.readyState === EventSource.CLOSED && retry === undefined) {
          source.close();
          retry = window.setTimeout(() => { retry = undefined; if (mounted.current) connect(); }, retryDelay);
          retryDelay = Math.min(retryDelay * 2, 10000);
        }
      };
      source.addEventListener('snapshot', event => {
        if (source !== stream) return;
        try { accept(JSON.parse((event as MessageEvent).data)); }
        catch { void refresh(); }
      });
      source.addEventListener('reset', () => { if (source === stream) void refresh(); });
    };
    connect();
    const poll = window.setInterval(() => void refresh(), 5000);
    const clock = window.setInterval(() => setWallTime(Date.now()), 1000);
    return () => { mounted.current = false; stream.close(); clearTimeout(retry); clearInterval(poll); clearInterval(clock); };
  }, [accept, refresh]);
  const offline = loadError || receivedAt != null && wallTime - receivedAt > 12000;
  return { snapshot, refresh, connection, offline, receivedAt };
}

export default function App() {
  const { snapshot, refresh, connection, offline, receivedAt } = useDispatcher();
  const { run, system } = snapshot;
  useLayoutEffect(() => {
    window.dispatchEvent(new CustomEvent('dispatcher:rendered', { detail: { event_id: snapshot.event_id, rendered_at: Date.now(), snapshot } }));
  }, [snapshot]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null);
  const [incidentDetails, setIncidentDetails] = useState<Incident | null>(null);
  const [details, setDetails] = useState<VehicleDetails | null>(null);
  const [network, setNetwork] = useState<Network | null>(null);
  const [networkError, setNetworkError] = useState(false);
  const [detailError, setDetailError] = useState(false);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [speed, setSpeed] = useState(20);
  const [mode, setMode] = useState<ReplayMode>('dispatcher');
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId] = useState('demo');
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const [chart, setChart] = useState<'speed' | 'delay'>('speed');
  const [incidentFilter, setIncidentFilter] = useState('open');
  const [historyIncidents, setHistoryIncidents] = useState<Incident[]>([]);
  const [historyLimit, setHistoryLimit] = useState(100);
  const [historyError, setHistoryError] = useState(false);
  const incidentSection = useRef<HTMLElement>(null);
  const fleetSection = useRef<HTMLElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const priorRun = useRef<string | null>(null);

  const loadScenarios = useCallback(async () => {
    try { setScenarios(await api<Scenario[]>('/scenarios')); }
    catch { /* Starting historical replay remains available without the registry. */ }
  }, []);
  useEffect(() => { void loadScenarios(); }, [loadScenarios]);
  useEffect(() => {
    if (run?.id && priorRun.current !== run.id) {
      setSelectedId(null); setSelectedIncidentId(null); setDetails(null); setIncidentDetails(null); setHistoryIncidents([]); setHistoryLimit(100);
      setMode(['live', 'ndtp'].includes(run.mode) ? 'live' : run.mode === 'evaluation' ? 'evaluation' : 'dispatcher');
      setSpeed(run.speed || 20);
      if (['live', 'ndtp'].includes(run.mode)) setScenarioId(run.schedule_mode === 'demo_rebased' ? 'demo' : run.scenario_id || 'demo');
      priorRun.current = run.id;
    }
  }, [run?.id, run?.mode, run?.speed]);
  useEffect(() => {
    const controller = new AbortController();
    setNetwork(null); setNetworkError(false);
    api<Network>(`/network${run?.id ? `?run_id=${encodeURIComponent(run.id)}` : ''}`, { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted && (!value.run_id || value.run_id === run?.id)) setNetwork(value); })
      .catch(error => { if (error.name !== 'AbortError') setNetworkError(true); });
    return () => controller.abort();
  }, [run?.id, snapshot.network_version]);

  const vehicles = useMemo(() => snapshot.vehicles.map(v => offline ? { ...v, stale: true } : v), [snapshot.vehicles, offline]);
  const incidents = snapshot.incidents;
  const sortedVehicles = useMemo(() => [...vehicles].sort((a, b) => riskOrder[vehicleRisk(a)] - riskOrder[vehicleRisk(b)] || (b.prediction?.p_late || 0) - (a.prediction?.p_late || 0)), [vehicles]);
  useEffect(() => {
    if (!selectedIncidentId && sortedVehicles.length && (!selectedId || !vehicles.some(v => v.id === selectedId))) setSelectedId(sortedVehicles[0].id);
  }, [sortedVehicles, selectedId, selectedIncidentId, vehicles]);
  useEffect(() => {
    if (!selectedId) { setDetails(null); return; }
    const controller = new AbortController();
    setDetailError(false);
    api<VehicleDetails>(`/vehicles/${encodeURIComponent(selectedId)}`, { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) setDetails(value); })
      .catch(error => { if (error.name !== 'AbortError') setDetailError(true); });
    return () => controller.abort();
  }, [selectedId, snapshot.event_id, run?.id]);
  useEffect(() => {
    if (!selectedIncidentId) { setIncidentDetails(null); return; }
    const controller = new AbortController();
    api<Incident>(`/incidents/${encodeURIComponent(selectedIncidentId)}`, { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) setIncidentDetails(value); })
      .catch(error => { if (error.name !== 'AbortError') setDetailError(true); });
    return () => controller.abort();
  }, [selectedIncidentId, snapshot.event_id, run?.id]);

  useEffect(() => {
    if (incidentFilter === 'open') return;
    const controller = new AbortController();
    api<Incident[]>(`/incidents?limit=${historyLimit}&offset=0`, { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) { setHistoryIncidents(value); setHistoryError(false); } })
      .catch(error => { if (error.name !== 'AbortError') setHistoryError(true); });
    return () => controller.abort();
  }, [incidentFilter, historyLimit, snapshot.event_id, run?.id]);

  const selectedVehicle = vehicles.find(v => v.id === selectedId) || null;
  const currentDetails = details?.id === selectedId ? details : null;
  const selectedIncident = selectedIncidentId ? incidents.find(i => i.id === selectedIncidentId) || (incidentDetails?.id === selectedIncidentId ? incidentDetails : undefined) : undefined;
  const currentIncident = selectedIncident || incidents.find(i => i.tr_id === selectedId && i.target_visit_id === selectedVehicle?.prediction?.target_visit_id && (!i.prediction_id || !selectedVehicle?.prediction?.id || i.prediction_id === selectedVehicle.prediction.id));
  const riskCount = vehicles.filter(v => vehicleRisk(v) === 'red').length;
  const attentionCount = vehicles.filter(v => vehicleRisk(v) === 'yellow').length;
  const staleCount = vehicles.filter(v => v.stale).length;
  const activeIncidents = incidents.filter(i => i.status === 'active');
  const displayedIncidents = (incidentFilter === 'open' ? activeIncidents : historyIncidents.filter(i => incidentFilter === 'all' || i.status !== 'active')).slice().sort((a, b) => riskOrder[a.risk] - riskOrder[b.risk]);
  const filteredVehicles = sortedVehicles.filter(v => String(v.id).includes(search.trim()) && (filter === 'all' || filter === 'risk' && ['red', 'yellow'].includes(vehicleRisk(v)) || filter === 'stale' && v.stale));
  const sameMode = (run?.mode === mode || mode === 'live' && ['ndtp', 'live'].includes(run?.mode || '')) && (mode !== 'live' || scenarioId === 'demo' && run?.schedule_mode === 'demo_rebased' || scenarioId === run?.scenario_id);
  const running = run?.status === 'running';
  const isLive = ['live', 'ndtp'].includes(run?.mode || '');
  const finished = !run || ['completed', 'finished', 'idle', 'stopped', 'error'].includes(run.status);
  const action = async (fn: () => Promise<unknown>) => {
    setBusy(true); setActionError(null);
    try { await fn(); await refresh(); }
    catch (error) { setActionError(error instanceof Error ? error.message : 'Не удалось выполнить действие'); }
    finally { setBusy(false); }
  };
  const start = () => action(() => mode === 'live'
    ? post('/live/start', scenarioId === 'demo' ? { schedule_mode: 'demo_rebased' } : { scenario_id: scenarioId, schedule_mode: 'as_is' })
    : post('/replay/start', { speed, mode }));
  const mainAction = () => !sameMode || finished ? start() : action(() => post(running ? '/replay/pause' : '/replay/resume'));
  const mainLabel = busy ? 'Подождите…' : !sameMode || finished ? mode === 'live' ? 'Подключить поток' : 'Запустить replay' : isLive ? 'Поток включён' : running ? 'Пауза' : 'Продолжить';
  const changeSpeed = (value: number) => {
    setSpeed(value);
    if (run && !isLive && ['running', 'paused'].includes(run.status)) void action(() => post('/replay/speed', { speed: value }));
  };
  const selectVehicle = (id: string) => { setSelectedId(id); setSelectedIncidentId(null); setChart('speed'); };
  const selectIncident = (incident: Incident) => {
    setSelectedId(incident.tr_id); setSelectedIncidentId(incident.id); setIncidentDetails(incident);
    fleetSection.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  const importScenario = async (file: File) => action(async () => {
    const body: unknown = JSON.parse(await file.text());
    await post('/scenarios/import?dry_run=true', body);
    const imported = await post<{ scenario: Scenario }>('/scenarios/import', body);
    await loadScenarios();
    if (imported.scenario?.id) setScenarioId(imported.scenario.id);
    setNotice('Сценарий проверен и импортирован. Выберите его для нового потока.');
  });
  const focusPrediction = selectedIncident ? incidentPrediction(selectedIncident) : selectedVehicle?.prediction;
  const provenance = run?.mode === 'evaluation' ? 'Ретроспективная оценка · не живое предупреждение'
    : isLive ? run?.schedule_mode === 'demo_rebased' ? 'Демо NDTP · время расписания перенесено' : `NDTP · ${run?.scenario_name || 'выбранный сценарий'}`
    : 'Исторический replay · виртуальное время';

  return <div className="app-shell" data-event-id={snapshot.event_id}>
    <aside className="sidebar" aria-label="Навигация"><a className="sidebar-brand" href="#top" aria-label="Контур — обзор"><Brand small /></a><nav>
      <button className="nav-button active" title="Обзор" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}><Icon name="grid" /></button>
      <button className="nav-button" title="Транспорт" onClick={() => fleetSection.current?.scrollIntoView({ behavior: 'smooth' })}><Icon name="bus" /></button>
      <button className="nav-button" title="Инциденты" onClick={() => incidentSection.current?.scrollIntoView({ behavior: 'smooth' })}><Icon name="bell" /></button>
    </nav><div className="sidebar-bottom"><button className="nav-button" aria-label="Как работает прогноз" onClick={() => setShowHelp(true)}><Icon name="info" /></button></div></aside>
    <div className="app-content" id="top">
      <header className="topbar"><div className="topbar-brand"><Brand /><span className="brand-divider" /><span className="product-label">Диспетчерская</span></div><span className={`connection-status ${offline ? 'offline' : 'online'}`}><i className="dot" />{offline ? 'Нет связи с сервером' : connection === 'connected' ? 'Система подключена' : 'Проверяем обновления'}</span></header>
      <main>
        <div className="page-heading"><div><h1>Обзор движения</h1><p>Прогноз отклонений за 10–15 минут до планового прибытия</p></div><div className="simulation-clock"><Icon name="clock" size={18}/><div><strong>{time(run?.virtual_time, true)}</strong><span>{run ? `${day(run.virtual_time)} · МСК` : 'Ожидание телеметрии'}</span></div></div></div>
        <div className="control-bar"><div className="mode-control"><label className="sr-only" htmlFor="mode">Источник данных</label><select id="mode" value={mode} onChange={event => setMode(event.target.value as ReplayMode)} disabled={busy}><option value="dispatcher">Диспетчерский replay</option><option value="evaluation">Ретроспективная оценка</option><option value="live">Живой поток NDTP</option></select>
          {mode === 'live' && <><label className="sr-only" htmlFor="scenario">Сценарий NDTP</label><select id="scenario" value={scenarioId} onChange={event => setScenarioId(event.target.value)} disabled={busy}><option value="demo">Демо с согласованным временем</option>{scenarios.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}</select></>}
          <button className="text-button import-button" onClick={() => fileInput.current?.click()} disabled={busy}>Импорт JSON</button><input ref={fileInput} type="file" accept="application/json,.json" aria-label="Импорт сценария JSON" className="sr-only" onChange={event => { const file = event.target.files?.[0]; if (file) void importScenario(file); event.target.value = ''; }} />
        </div><div className="replay-controls">{mode !== 'live' && <div className="speed-control" role="group" aria-label="Скорость воспроизведения">{[1, 5, 20].map(value => <button key={value} onClick={() => changeSpeed(value)} disabled={busy} className={speed === value ? 'selected' : ''}>×{value}</button>)}</div>}
          {run && <button className="button-icon" aria-label="Начать новый прогон" title="Начать новый прогон" onClick={() => void start()} disabled={busy}><Icon name="refresh" size={16}/></button>}
          <button className="button primary replay-button" onClick={() => void mainAction()} disabled={busy || isLive && sameMode && running}><Icon name={sameMode && running && !isLive ? 'pause' : 'play'} size={14}/>{mainLabel}</button><span className="run-status">{!run ? 'Не запущен' : running ? 'В работе' : run.status === 'paused' ? 'На паузе' : 'Завершён'}</span></div></div>
        <div className="provenance-strip" data-testid="provenance">{provenance}<span>{network?.geometry_kind === 'supplied_route' ? 'Импортированная маршрутная сеть' : network?.geometry_kind === 'mixed' ? 'Импортированная геометрия + восстановленная схема' : 'Восстановленная схема по расписанию'}</span></div>
        {actionError && <div className="alert-banner" role="alert">{actionError}<button className="text-button" onClick={() => setActionError(null)}>Закрыть</button></div>}
        {notice && <div className="alert-banner muted" role="status">{notice}<button className="text-button" onClick={() => setNotice(null)}>Закрыть</button></div>}
        {offline && <div className="alert-banner muted" role="status" data-testid="offline-warning"><Icon name="signal" size={18}/><span>Показано последнее состояние{receivedAt ? `, полученное в ${time(new Date(receivedAt).toISOString(), true)}` : ''}. Текущие вероятности скрыты до восстановления связи.</span><button className="text-button" onClick={() => void refresh()}>Повторить</button></div>}

        <div className="metrics-grid compact-metrics">
          <Metric label="Транспорт" value={number(vehicles.length)} description={`${staleCount} с устаревшими данными`} />
          <Metric label="Высокий риск" value={offline ? '—' : number(riskCount)} description="Задержка > 2 мин · вероятность ≥ 70%" color="red" testId="high-risk-count" />
          <Metric label="Требуют внимания" value={offline ? '—' : number(attentionCount)} description="Риск 40–70% или опережение > 1 мин" color="yellow" />
          <Metric label="Модель" value={system.ml_status === 'ready' ? 'Активна' : 'Ожидание'} description={system.model_version || 'CPU inference'} />
        </div>

        <div className="workspace-grid">
          <section className="fleet-panel panel" ref={fleetSection} aria-label="Транспортные средства"><div className="panel-heading"><h2>Транспорт <span className="count-badge">{vehicles.length}</span></h2><span className="tiny-label">ПО РИСКУ</span></div><div className="fleet-search"><Icon name="search" size={17}/><input type="search" placeholder="Поиск по номеру ТС" value={search} onChange={e => setSearch(e.target.value)} aria-label="Поиск по номеру ТС"/></div><div className="fleet-filters"><button className={filter === 'all' ? 'selected' : ''} onClick={() => setFilter('all')}>Все</button><button className={filter === 'risk' ? 'selected' : ''} onClick={() => setFilter('risk')}>С риском</button><button className={filter === 'stale' ? 'selected' : ''} onClick={() => setFilter('stale')}>Без связи</button></div><div className="fleet-list">{filteredVehicles.map(vehicle => <VehicleRow key={vehicle.id} vehicle={vehicle} selected={selectedId === vehicle.id && !selectedIncidentId} onSelect={() => selectVehicle(vehicle.id)} />)}{!filteredVehicles.length && <div className="empty-list"><h3>{vehicles.length ? 'Транспорт не найден' : 'Ожидание транспорта'}</h3><p>{vehicles.length ? 'Измените фильтр.' : 'Выберите источник и запустите поток.'}</p></div>}</div></section>
          <FleetMap vehicles={vehicles} selectedId={selectedId} prediction={focusPrediction} network={network} segmentRisks={snapshot.segment_risks} networkError={networkError} offline={offline} onSelect={selectVehicle} runId={run?.id} isIdle={!run} />
          <section className="detail-panel panel" aria-label="Карточка транспорта"><div className="panel-heading"><h2>{selectedIncidentId ? 'Карточка инцидента' : 'Текущий прогноз'}</h2>{selectedIncidentId && selectedId && <button className="text-button" onClick={() => selectVehicle(selectedId)}>К ТС</button>}</div>
            {selectedVehicle ? <VehicleDetail vehicle={selectedVehicle} details={currentDetails} incident={currentIncident} explicitIncident={!!selectedIncidentId} offline={offline} chart={chart} setChart={setChart} detailError={detailError} onAck={id => void action(() => post(`/incidents/${encodeURIComponent(id)}/ack`))} busy={busy} /> : <div className="empty-detail"><h3>Выберите транспорт</h3><p>Сеть показана полностью. Карточка откроется при выборе ТС или предупреждения.</p></div>}
          </section>
        </div>
        <section className="incidents-panel panel" ref={incidentSection} aria-label="Журнал инцидентов"><div className="section-heading"><div><h2>Центр внимания <span className="count-badge">{activeIncidents.length}</span></h2><p>Прежние цели хранятся отдельно от текущего прогноза</p></div><div className="segmented"><button className={incidentFilter === 'open' ? 'selected' : ''} onClick={() => setIncidentFilter('open')}>Текущие</button><button className={incidentFilter === 'history' ? 'selected' : ''} onClick={() => setIncidentFilter('history')}>Наблюдение / история</button><button className={incidentFilter === 'all' ? 'selected' : ''} onClick={() => setIncidentFilter('all')}>Все</button></div></div>
          {historyError && incidentFilter !== 'open' && <p className="detail-error">История временно недоступна. Повторяем запрос.</p>}
          <div className="incident-table-wrap"><table className="incident-table"><thead><tr><th>Транспорт / Время</th><th>Участок</th><th>Прогноз</th><th>Наблюдение</th><th>Состояние</th><th>Просмотр</th></tr></thead><tbody>{displayedIncidents.map(incident => {
            const stale = offline || ['data_stale', 'expired'].includes(incident.status) || ['stale', 'prediction_stale'].includes(incident.prediction?.display_state || '');
            const risk = stale ? 'gray' : incident.risk;
            return <tr key={incident.id} data-incident-id={incident.id} data-risk={risk}><td><button className="incident-vehicle" onClick={() => selectIncident(incident)}><span className={`incident-icon ${risk}`}><Icon name="bell" size={16}/></span><span><strong>ТС {incident.tr_id}</strong><small>{time(incident.prediction_time, true)}</small></span></button></td><td className="incident-target">{segmentName(incident.target_segment || incident.prediction?.target_segment)}<small>{stopName(incident.target_name, incident.target_visit_id)}</small></td><td><strong className={`delay-text ${risk}`}>{delay(incident.prediction_delay_s)}</strong><small>{stale ? 'Последний прогноз' : incident.status === 'monitoring' ? 'Прежняя цель' : `${percent(incident.p_late)} вероятность`}</small></td><td className="incident-reason">{incident.reason || 'Причина не установлена'}</td><td><span className={`status-pill ${risk}`}>{incidentStatus(incident, offline)}</span></td><td>{incident.acknowledged ? <span className="ack-note">Просмотрено</span> : <button className="button incident-ack" onClick={() => void action(() => post(`/incidents/${encodeURIComponent(incident.id)}/ack`))} disabled={busy || offline}>Просмотрено</button>}</td></tr>;
          })}</tbody></table>{!displayedIncidents.length && <div className="incidents-empty"><Icon name="check" size={23}/><p>В этом разделе предупреждений нет</p></div>}</div>
          {incidentFilter !== 'open' && (snapshot.incidents_total || 0) > historyLimit && historyLimit < 500 && <button className="button history-more" onClick={() => setHistoryLimit(value => Math.min(value + 100, 500))}>Показать ещё 100</button>}
          {(snapshot.incidents_total || 0) > (incidentFilter === 'open' ? 100 : historyLimit) && <p className="journal-limit">Показана часть журнала ({incidentFilter === 'open' ? 'до 100 текущих' : `последние ${historyLimit}`}), всего событий: {snapshot.incidents_total}.</p>}
        </section>
        <section className="system-strip" aria-label="Состояние системы"><span>Backend <strong>{system.backend_status === 'ok' ? 'онлайн' : 'ожидание'}</strong></span><span>Телеметрия <strong>{number(system.telemetry_count)}</strong></span><span>Очередь <strong>{number(system.queue_depth)}</strong></span><span>NDTP ошибки <strong>{number(system.ndtp_errors)}</strong></span><span>Пакет прогнозов p95 <strong>{number(system.processing_latency_p95_ms)} мс</strong></span><span>MAE <strong>{system.evaluation_mae_s != null ? `${number(system.evaluation_mae_s, 1)} с` : 'нет эталона в потоке'}</strong></span></section>
        <footer className="page-footer"><span>КОНТУР · Поддержка решений диспетчера</span><span>{run ? `Прогон ${run.id.slice(0, 8)}` : 'Готов к подключению'} · Все времена — МСК</span></footer>
      </main>
    </div>
    {showHelp && <HelpDialog onClose={() => setShowHelp(false)} />}
  </div>;
}

function Metric({ label, value, description, color = '', testId }: { label: string; value: string; description: string; color?: string; testId?: string }) {
  return <section className="metric-card"><div className="metric-top"><span>{label}</span></div><div className={`metric-value ${color}`} data-testid={testId}>{value}</div><div className="metric-description" title={description}>{description}</div></section>;
}

function RiskBadge({ risk, label }: { risk: Risk; label?: string }) {
  return <span className={`risk-badge ${risk}`}><i className="dot"/>{label || riskLabels[risk]}</span>;
}

function VehicleRow({ vehicle, selected, onSelect }: { vehicle: Vehicle; selected: boolean; onSelect: () => void }) {
  const risk = vehicleRisk(vehicle);
  return <button className={`vehicle-row ${selected ? 'selected' : ''}`} data-vehicle-id={vehicle.id} data-risk={risk} onClick={onSelect} aria-pressed={selected}><div className="vehicle-row-top"><span className={`vehicle-icon ${risk}`}><Icon name="bus" size={17}/></span><strong>ТС {vehicle.id}</strong><span className={`vehicle-delay ${risk}`}>{vehicle.stale ? '—' : delay(vehicle.prediction?.prediction_delay_s)}</span></div><div className="vehicle-row-target"><Icon name="pin" size={12}/><span>{vehicle.prediction?.target_segment || vehicle.target_segment ? segmentName(vehicle.prediction?.target_segment || vehicle.target_segment) : stopName(vehicle.prediction?.target_name, vehicle.prediction?.target_visit_id)}</span></div><div className="vehicle-row-bottom"><RiskBadge risk={risk} label={vehicle.stale ? 'Данные устарели' : !predictionIsCurrent(vehicle.prediction) ? 'Нет текущего прогноза' : undefined}/><span>{vehicle.stale || !predictionIsCurrent(vehicle.prediction) ? age(vehicle.position_age_s) : percent(vehicle.prediction?.p_late)}</span></div></button>;
}

function incidentPrediction(incident: Incident): Prediction {
  return incident.prediction || { ...incident, target_time_begin: incident.target_time_begin || '', source: 'model', display_state: incident.status === 'active' ? 'current' : incident.status === 'data_stale' ? 'stale' : 'monitoring', timing_status: incident.timing_status as Prediction['timing_status'] };
}

function incidentStatus(incident: Incident, offline: boolean): string {
  if (offline) return 'Нет связи · последнее состояние';
  return ({ active: 'Текущий риск', monitoring: 'Наблюдение · прежняя цель', data_stale: 'Данные устарели', expired: 'Горизонт завершён', resolved: 'Риск снизился', closed: 'Закрыт' })[incident.status] || incident.status;
}

function VehicleDetail({ vehicle, details, incident, explicitIncident, offline, chart, setChart, detailError, onAck, busy }: {
  vehicle: Vehicle; details: VehicleDetails | null; incident?: Incident; explicitIncident: boolean; offline: boolean;
  chart: 'speed' | 'delay'; setChart: (value: 'speed' | 'delay') => void; detailError: boolean; onAck: (id: string) => void; busy: boolean;
}) {
  const prediction = explicitIncident && incident ? incidentPrediction(incident) : vehicle.prediction;
  const stale = offline || vehicle.stale || ['stale', 'prediction_stale'].includes(prediction?.display_state || '') || explicitIncident && ['data_stale', 'expired'].includes(incident?.status || '');
  const risk: Risk = stale ? 'gray' : explicitIncident && incident ? incident.risk : vehicleRisk(vehicle);
  const probability = stale || prediction?.timing_status === 'retrospective' || prediction?.display_state === 'monitoring' ? null : prediction?.p_late;
  const segment = prediction?.target_segment || (explicitIncident ? incident?.target_segment : vehicle.target_segment);
  const factors = prediction?.factors || [];
  const chartSamples = chart === 'speed' ? (details?.telemetry || []).map(t => ({ at: t.event_time, value: t.speed })) : (details?.history || []).map(p => ({ at: p.prediction_time, value: p.prediction_delay_s }));
  const currentSegment = vehicle.current_segment || details?.current_segment;
  return <div className="detail-content" data-testid="incident-card" data-target-id={prediction?.target_visit_id} data-risk={risk} data-incident-id={incident?.id || ''}>
    <div className="critical-card">
      <div className="vehicle-title"><div className="vehicle-number">ТС {vehicle.id}</div><RiskBadge risk={risk} label={stale ? 'Данные устарели' : undefined}/></div>
      <div className="freshness-note" data-testid="prediction-freshness">{explicitIncident && incident ? incidentStatus(incident, offline) : freshnessLabel(prediction, offline)} · {time(prediction?.prediction_time, true)}</div>
      <div className={`forecast-card forecast-${risk}`}><div className="forecast-label">{stale ? 'ПОСЛЕДНИЙ ПРОГНОЗ' : 'ОТКЛОНЕНИЕ К ЦЕЛЕВОЙ ОСТАНОВКЕ'}</div><div className="forecast-inline"><div className="forecast-value">{delay(prediction?.prediction_delay_s)}</div><div className="probability-summary"><strong data-testid="card-probability">{percent(probability)}</strong><span>вероятность<br/>задержки &gt; 2 мин</span></div></div></div>
      <div className="target-block"><div className="detail-section-label">УЧАСТОК ПРИБЫТИЯ</div><h3 data-testid="card-segment">{segmentName(segment)}</h3><span className="geometry-note">{segment?.geometry_kind === 'supplied_route' ? 'Импортированная геометрия' : 'Схема по расписанию'}</span><div className="arrival-times"><div><span>По расписанию</span><strong>{time(prediction?.target_time_begin)}</strong></div><Icon name="arrow" size={16}/><div><span>Ожидаемое</span><strong>{time(prediction?.predicted_arrival_at)}</strong></div></div><div className="horizon-note">{prediction ? `Горизонт при расчёте: ${number((prediction.publication_horizon_s ?? prediction.horizon_s ?? 0) / 60, 1)} мин` : 'Нет цели в горизонте 10–15 минут'}</div></div>
      <div className="reason-summary"><div className="detail-section-label">НАБЛЮДЕНИЕ / ВОЗМОЖНАЯ ПРИЧИНА</div><p data-testid="card-reason">{incident?.reason || 'Причина не установлена'}</p><span>Причина не подтверждена внешними данными.</span></div>
      {incident && <div className="card-ack">{incident.acknowledged ? <span className="ack-note"><Icon name="check" size={14}/>Просмотрено диспетчером</span> : <button className="button" onClick={() => onAck(incident.id)} disabled={busy || offline} data-testid="card-ack">Отметить просмотренным</button>}</div>}
    </div>
    <details className="detail-extra"><summary>Движение, факторы и рекомендация</summary><div className="current-values"><div><span>Скорость</span><strong>{number(vehicle.speed)} км/ч</strong></div><div><span>Текущее отклонение</span><strong>{delay(vehicle.cur_dev_s)}</strong></div></div><p className="source-note">Текущий участок: {segmentName(currentSegment)}<br/>Средняя скорость участка: {number(currentSegment?.segment_speed_kmh ?? currentSegment?.speed_mean_kmh ?? currentSegment?.mean_speed_kmh, 1)} км/ч<br/>Отклонение: {vehicle.cur_dev_source === 'provided' ? 'из прогнозной точки' : vehicle.cur_dev_source === 'estimated' ? 'оценка по посещению' : 'нет достоверной оценки'}<br/>Позиция: {age(vehicle.position_age_s)}</p>
      <div className="history-section"><div className="detail-chart-heading"><span className="detail-section-label">ИСТОРИЯ ТС</span><div className="chart-tabs"><button className={chart === 'speed' ? 'selected' : ''} onClick={() => setChart('speed')}>Скорость</button><button className={chart === 'delay' ? 'selected' : ''} onClick={() => setChart('delay')}>Прогноз</button></div></div>{detailError ? <p className="detail-error">История временно недоступна</p> : <TelemetryChart samples={chartSamples} unit={chart === 'speed' ? 'км/ч' : 'секунды'}/>}</div>
      {factors.length > 0 && <div className="factors-section"><div className="detail-section-label">ФАКТОРЫ ПРОГНОЗА</div>{factors.slice(0, 3).map((factor, index) => <div className="factor-row" key={index}><span>{factorName(factor)}</span>{typeof factor !== 'string' && factor.value != null && <strong>{typeof factor.value === 'number' ? number(factor.value, 1) : factor.value}</strong>}</div>)}</div>}
      {incident && <p className="recommendation">{incident.recommendation}</p>}
      <p className="detail-version">{prediction?.source === 'baseline' ? 'Резервный baseline' : 'ML-модель'}: {prediction?.model_version || '—'}<br/>Подтверждение просмотра не закрывает инцидент.</p>
    </details>
  </div>;
}

function factorName(factor: Factor | string): string {
  const key = typeof factor === 'string' ? factor : factor.feature || factor.name || '';
  return typeof factor !== 'string' && factor.label || factorLabels[key] || key.replace(/_/g, ' ');
}

function HelpDialog({ onClose }: { onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { ref.current?.showModal(); }, []);
  return <dialog ref={ref} className="help-dialog" onCancel={onClose}><h2>Предупреждение до планового прибытия</h2><p>Прогноз относится к первой плановой остановке через 10–15 минут. Плюс — опоздание, минус — опережение. Цвет меняется сразу по последнему прогнозу.</p><p>Красный: вероятность задержки более 2 минут ≥70%. Жёлтый: вероятность ≥40% либо опережение более минуты. Серый: нет текущих данных. ACK отмечает просмотр и не изменяет риск.</p><p>Исторический replay использует виртуальные часы. Ретроспективный расчёт не считается живым предупреждением. В демо NDTP расписание переносится во времени; для реальной эксплуатации импортируйте актуальный сценарий.</p><p>Линии без импортированной геометрии — восстановленная схема плановых посещений. Цвет участка относится к риску прибытия на его конечную остановку, а не подтверждает пробку на дороге.</p><button className="button primary" onClick={onClose}>Понятно</button></dialog>;
}
