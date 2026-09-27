"""Isolated browser regressions: local built assets and API fixtures, no live mutations."""
from __future__ import annotations

import json
import copy
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / 'artifacts' / 'ui_regressions'
    output.mkdir(parents=True, exist_ok=True)
    segment_a = {'id': 'seg-a', 'path_id': 'plan-v1', 'from_visit_id': 'origin', 'to_visit_id': 'a',
                 'from_name': 'Северная', 'to_name': 'Площадь', 'geometry_kind': 'schedule_schematic',
                 'geometry': {'type': 'LineString', 'coordinates': [[37.60, 55.70], [37.61, 55.71]]}}
    segment_b = {'id': 'seg-b', 'path_id': 'plan-v1', 'from_visit_id': 'a', 'to_visit_id': 'b',
                 'from_name': 'Площадь', 'to_name': 'Вокзал', 'geometry_kind': 'schedule_schematic',
                 'geometry': {'type': 'LineString', 'coordinates': [[37.61, 55.71], [37.63, 55.72]]}}
    prediction = {'id': 'p-b', 'run_id': 'fixture', 'prediction_time': '2026-01-06T07:00:00Z',
                  'target_visit_id': 'b', 'target_name': 'Вокзал', 'target_lat': 55.72, 'target_lon': 37.63,
                  'target_time_begin': '2026-01-06T07:13:00Z', 'predicted_arrival_at': '2026-01-06T07:16:00Z',
                  'prediction_delay_s': 180, 'p_late': .85, 'risk': 'red', 'source': 'model', 'horizon_s': 780,
                  'timing_status': 'verified', 'display_state': 'current', 'current_prediction': True,
                  'model_version': 'fixture-model', 'target_segment': segment_b}
    previous = {**prediction, 'id': 'p-a', 'target_visit_id': 'a', 'target_name': 'Площадь',
                'target_time_begin': '2026-01-06T07:11:00Z', 'predicted_arrival_at': '2026-01-06T07:14:00Z',
                'target_segment': segment_a, 'display_state': 'monitoring', 'current_prediction': False}
    vehicle = {'id': 'v1', 'lat': 55.705, 'lon': 37.605, 'speed': 20, 'event_time': '2026-01-06T07:00:00Z',
               'position_age_s': 5, 'stale': False, 'cur_dev_s': 10, 'cur_dev_source': 'estimated', 'prediction': prediction}
    old_incident = {'id': 'i-a', 'run_id': 'fixture', 'tr_id': 'v1', 'target_visit_id': 'a', 'risk': 'red',
                    'status': 'monitoring', 'acknowledged': False, 'prediction_id': 'p-a', 'prediction': previous,
                    'prediction_delay_s': 180, 'p_late': .85, 'prediction_time': prediction['prediction_time'],
                    'target_name': 'Площадь', 'reason': 'Простой у Северной', 'recommendation': 'Уточнить обстановку',
                    'target_segment': segment_a}
    current_incident = {**old_incident, 'id': 'i-b', 'target_visit_id': 'b', 'status': 'active',
                        'prediction_id': 'p-b', 'prediction': prediction, 'target_name': 'Вокзал',
                        'reason': 'Снижение скорости к вокзалу', 'target_segment': segment_b}
    state = {'run': {'id': 'fixture', 'mode': 'dispatcher', 'status': 'paused', 'virtual_time': '2026-01-06T07:00:00Z', 'speed': 20},
             'vehicles': [], 'incidents': [], 'system': {'backend_status': 'ok', 'ml_status': 'ready'},
             'event_id': 1, 'network_version': 'n1', 'segment_risks': {'seg-a': {'risk': 'gray'}, 'seg-b': {'risk': 'red'}}}
    network = {'network_version': 'n1', 'geometry_kind': 'schedule_schematic', 'paths': [{'id': 'plan-v1'}],
               'visits': [], 'segments': [segment_a, segment_b]}
    scenarios = [{'id': 'imported-1', 'name': 'Проверенный сценарий'}]
    calls: list[tuple[str, object]] = []
    errors: list[str] = []
    checks: list[str] = []
    fail_snapshot = False
    snapshot_override = None

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(viewport={'width': 1366, 'height': 768}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script("""(() => {
          window.EventSource = class {
            static CLOSED = 2;
            constructor() { this.listeners = {}; this.readyState = 0; window.__source = this; window.__sourceCount = (window.__sourceCount || 0) + 1; setTimeout(() => { if (this.readyState !== 2) { this.readyState = 1; this.onopen?.(); } }, 0); }
            addEventListener(name, fn) { this.listeners[name] = fn; }
            close() { this.readyState = 2; }
          };
          window.__snapshot = value => window.__source.listeners.snapshot({data: JSON.stringify(value)});
        })();""")

        def handle(route):
            nonlocal fail_snapshot, snapshot_override
            request = route.request
            path = urlparse(request.url).path
            if 'tile.openstreetmap.org' in request.url:
                route.abort(); return
            if path.startswith('/api/v1/'):
                body = request.post_data_json if request.method == 'POST' else None
                if request.method == 'POST':
                    calls.append((path + ('?' + urlparse(request.url).query if urlparse(request.url).query else ''), body))
                if path == '/api/v1/snapshot':
                    if fail_snapshot:
                        route.abort(); return
                    payload = snapshot_override if snapshot_override is not None else state
                    snapshot_override = None
                elif path == '/api/v1/network':
                    payload = network
                elif path == '/api/v1/scenarios':
                    payload = scenarios
                elif path == '/api/v1/scenarios/import':
                    payload = {'valid': True, 'dry_run': 'dry_run=true' in request.url, 'scenario': scenarios[0]}
                elif path.startswith('/api/v1/vehicles/'):
                    requested_vehicle = next((item for item in state['vehicles'] if item['id'] == path.rsplit('/', 1)[-1]), vehicle)
                    payload = {**requested_vehicle, 'history': [previous, prediction], 'telemetry': [], 'planned_visits': []}
                elif path == '/api/v1/incidents':
                    payload = state['incidents']
                elif path.startswith('/api/v1/incidents/'):
                    identifier = path.split('/')[4]
                    found = next(item for item in state['incidents'] if item['id'] == identifier)
                    if path.endswith('/ack'):
                        found['acknowledged'] = True
                    payload = found
                elif path in {'/api/v1/live/start', '/api/v1/replay/start'}:
                    payload = state
                else:
                    route.fulfill(status=404, json={'detail': path}); return
                route.fulfill(json=payload); return
            asset = ROOT / 'frontend' / 'dist' / path.lstrip('/')
            if path == '/':
                asset = ROOT / 'frontend' / 'dist' / 'index.html'
            if asset.is_file():
                route.fulfill(body=asset.read_bytes(), content_type=mimetypes.guess_type(asset)[0] or 'application/octet-stream')
            else:
                route.fulfill(status=404, body='Not found')

        page.route('**/*', handle)
        page.goto('http://dispatcher.test/', wait_until='domcontentloaded')
        try:
            expect(page.locator('.network-segment')).to_have_count(2)
        except AssertionError:
            print(json.dumps({'errors': errors, 'body': page.locator('body').inner_text(), 'network_calls': calls}, ensure_ascii=False))
            raise
        expect(page.locator('.vehicle-row')).to_have_count(0)
        checks.append('entire network before vehicle selection, with failed background tiles')

        def publish():
            state['event_id'] += 1
            page.evaluate('value => window.__snapshot(value)', state)

        state['vehicles'] = [vehicle]
        state['incidents'] = [old_incident, current_incident]
        publish()
        card = page.get_by_test_id('incident-card')
        expect(card).to_have_attribute('data-target-id', 'b')
        expect(page.get_by_test_id('card-reason')).to_have_text('Снижение скорости к вокзалу')
        expect(page.get_by_test_id('card-probability')).to_have_text('85%')
        expect(page.locator('.vehicle-v1.marker-red.marker-selected')).to_have_count(1)
        assert page.get_by_test_id('card-ack').bounding_box()['y'] + page.get_by_test_id('card-ack').bounding_box()['height'] < 768
        assert page.locator('.detail-panel').evaluate('(e) => e.scrollHeight <= e.clientHeight + 1')
        page.screenshot(path=str(output / 'desktop-1366.png'), full_page=True)
        checks.append('current target reason, probability, section and ACK visible at 1366×768 without inner scrolling')

        prediction.update(risk='green', p_late=.10, prediction_delay_s=30)
        current_incident.update(risk='green', p_late=.10, prediction_delay_s=30, status='resolved')
        state['segment_risks']['seg-b'] = {'risk': 'green'}
        publish()
        expect(card).to_have_attribute('data-risk', 'green')
        expect(page.locator('.vehicle-row')).to_have_attribute('data-risk', 'green')
        expect(page.get_by_test_id('high-risk-count')).to_have_text('0')
        expect(page.locator('.vehicle-v1.marker-green.marker-selected')).to_have_count(1)
        expect(page.locator('.segment-green')).to_have_count(1)
        expect(page.get_by_test_id('card-probability')).to_have_text('10%')
        checks.append('first lower forecast immediately updates card, list, marker, segment and count')

        page.get_by_role('button', name='Наблюдение / история', exact=True).click()
        page.locator('[data-incident-id="i-a"] .incident-vehicle').click()
        expect(card).to_have_attribute('data-target-id', 'a')
        expect(page.get_by_test_id('card-reason')).to_have_text('Простой у Северной')
        expect(page.get_by_test_id('card-segment')).to_have_text('Северная → Площадь')
        page.get_by_test_id('card-ack').click()
        expect(card.get_by_text('Просмотрено диспетчером')).to_be_visible()
        assert any(path == '/api/v1/incidents/i-a/ack' for path, _ in calls)
        assert not any(path == '/api/v1/incidents/i-b/ack' for path, _ in calls)
        checks.append('historical incident uses its own target, segment, cause and ACK identifier')

        page.get_by_role('button', name='К ТС', exact=True).click()
        prediction.update(display_state='stale', current_prediction=False, risk='gray', p_late=None, last_known_p_late=.10)
        publish()
        expect(card).to_have_attribute('data-risk', 'gray')
        expect(page.get_by_test_id('card-probability')).to_have_text('—')
        expect(page.get_by_test_id('prediction-freshness')).to_contain_text('Прогноз устарел')
        checks.append('old prediction is marked stale even while telemetry is fresh')

        prediction.update(display_state='current', current_prediction=True, risk='red', p_late=.85)
        publish()
        expect(page.get_by_test_id('card-probability')).to_have_text('85%')
        fail_snapshot = True
        page.clock.install()
        page.clock.fast_forward(6000)
        expect(page.get_by_test_id('offline-warning')).to_be_visible()
        expect(page.get_by_test_id('card-probability')).to_have_text('—')
        expect(page.locator('.vehicle-row')).to_have_attribute('data-risk', 'gray')
        fail_snapshot = False
        page.get_by_role('button', name='Повторить', exact=True).click()
        expect(page.get_by_test_id('offline-warning')).to_have_count(0)
        expect(page.get_by_test_id('card-probability')).to_have_text('85%')
        checks.append('backend disconnect preserves last state, hides probabilities, and recovers')

        # A crash preserves run_id but restarts the process-local sequence.
        # The late HTTP response below belongs to the preceding process.
        state.update(event_id=199, published_at='2026-01-06T07:00:20Z')
        publish()
        expect(page.locator('[data-event-id]')).to_have_attribute('data-event-id', '200')
        old_response = copy.deepcopy(state)
        old_response['event_id'] = 205
        state.update(event_id=1, published_at='2026-01-06T07:00:21Z')
        prediction.update(risk='green', p_late=.10)
        with page.expect_response('**/api/v1/snapshot'):
            page.evaluate("window.__source.listeners.reset({data:'{}'})")
        expect(page.locator('[data-event-id]')).to_have_attribute('data-event-id', '1')
        expect(page.get_by_test_id('card-probability')).to_have_text('10%')
        snapshot_override = old_response
        with page.expect_response('**/api/v1/snapshot'):
            page.evaluate("window.__source.listeners.reset({data:'{}'})")
        page.wait_for_timeout(100)
        expect(page.locator('[data-event-id]')).to_have_attribute('data-event-id', '1')
        expect(page.get_by_test_id('card-probability')).to_have_text('10%')
        unproven_reset = copy.deepcopy(state)
        unproven_reset['event_id'] = 0
        unproven_reset['vehicles'][0]['prediction']['p_late'] = .95
        page.evaluate('value => window.__snapshot(value)', unproven_reset)
        expect(page.get_by_test_id('card-probability')).to_have_text('10%')
        checks.append('same-run restart accepts newer publication with reset sequence and rejects late old-process HTTP response')

        source_count = page.evaluate('window.__sourceCount')
        page.evaluate('window.__source.readyState = 2; window.__source.onerror()')
        page.clock.fast_forward(1100)
        assert page.evaluate('window.__sourceCount') == source_count + 1
        state['published_at'] = '2026-01-06T07:00:22Z'
        prediction.update(risk='red', p_late=.85)
        publish()
        expect(page.get_by_test_id('card-probability')).to_have_text('85%')
        expect(page.locator('[data-event-id]')).to_have_attribute('data-event-id', '2')
        checks.append('permanently closed EventSource is recreated and resumes live snapshots')

        page.locator('#mode').select_option('live')
        page.get_by_role('button', name='Подключить поток', exact=True).click()
        assert ('/api/v1/live/start', {'schedule_mode': 'demo_rebased'}) in calls
        page.locator('#scenario').select_option('imported-1')
        page.get_by_role('button', name='Подключить поток', exact=True).click()
        assert ('/api/v1/live/start', {'scenario_id': 'imported-1', 'schedule_mode': 'as_is'}) in calls
        package = {'schema_version': '1.0', 'source_id': 'ui-test', 'name': 'Тест импорта', 'timezone': 'UTC', 'planned_visits': [], 'device_bindings': {}}
        page.get_by_label('Импорт сценария JSON').set_input_files({'name': 'scenario.json', 'mimeType': 'application/json', 'buffer': json.dumps(package).encode()})
        expect(page.get_by_role('status').filter(has_text='Сценарий проверен')).to_be_visible()
        assert ('/api/v1/scenarios/import?dry_run=true', package) in calls
        assert ('/api/v1/scenarios/import', package) in calls
        checks.append('NDTP demo/real scenario selection and JSON validation/import contracts')

        state['run']['mode'] = 'evaluation'
        prediction.update(timing_status='retrospective', current_prediction=False, display_state='monitoring')
        publish()
        expect(page.get_by_test_id('card-probability')).to_have_text('—')
        expect(page.get_by_test_id('provenance')).to_contain_text('Ретроспективная оценка')
        checks.append('retrospective evaluation cannot appear as a current live warning')

        page.set_viewport_size({'width': 390, 'height': 844})
        page.screenshot(path=str(output / 'mobile-390.png'), full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
        assert page.locator('.fleet-panel').bounding_box()['y'] < page.locator('.map-panel').bounding_box()['y']
        checks.append('390px layout has no page overflow and prioritizes vehicles before map')
        # Three frozen, explicitly synthetic screens prepare a human test; no participant results are fabricated.
        usability = ROOT / 'artifacts' / 'usability'
        (usability / 'screens').mkdir(parents=True, exist_ok=True)
        page.set_viewport_size({'width': 1366, 'height': 768})
        if page.get_by_role('status').filter(has_text='Сценарий проверен').count():
            page.get_by_role('status').filter(has_text='Сценарий проверен').get_by_role('button', name='Закрыть').click()
        base_prediction = {**prediction, 'timing_status': 'verified', 'display_state': 'current', 'current_prediction': True,
                           'risk': 'red', 'p_late': .85, 'prediction_delay_s': 180}
        segments = [segment_a, segment_b]
        example_vehicles = [{**vehicle, 'prediction': base_prediction}]
        names = [('v2', 'Парк', 'Больница'), ('v3', 'Музей', 'Рынок')]
        for index, (identifier, origin, destination) in enumerate(names, start=1):
            offset = index * .02
            segment = {'id': f'seg-{identifier}', 'path_id': f'plan-{identifier}', 'from_visit_id': f'{identifier}-start',
                       'to_visit_id': f'{identifier}-end', 'from_name': origin, 'to_name': destination,
                       'geometry_kind': 'schedule_schematic', 'geometry': {'type': 'LineString', 'coordinates': [[37.60 + offset, 55.70], [37.615 + offset, 55.718]]}}
            segments.append(segment)
            predicted = {**base_prediction, 'id': f'p-{identifier}', 'target_visit_id': f'{identifier}-end',
                         'target_name': destination, 'target_segment': segment, 'risk': 'green', 'p_late': .10,
                         'prediction_delay_s': 30, 'predicted_arrival_at': '2026-01-06T07:13:30Z',
                         'target_lat': 55.718, 'target_lon': 37.615 + offset}
            example_vehicles.append({**vehicle, 'id': identifier, 'lon': 37.605 + offset, 'prediction': predicted})
        answer_key = {'status': 'NOT_RUN', 'data_source': 'synthetic isolated UI fixtures', 'scenarios': []}
        for scenario in ['high-risk', 'multiple-risks', 'stale']:
            sample = copy.deepcopy(example_vehicles)
            if scenario == 'multiple-risks':
                sample[1]['prediction'].update(risk='red', p_late=.78, prediction_delay_s=160,
                                              predicted_arrival_at='2026-01-06T07:15:40Z')
                sample[2]['prediction'].update(risk='yellow', p_late=.52, prediction_delay_s=100,
                                              predicted_arrival_at='2026-01-06T07:14:40Z')
            if scenario == 'stale':
                sample[0].update(stale=True, position_age_s=120, telemetry_age_s=120)
                sample[0]['prediction'].update(risk='gray', p_late=None, last_known_p_late=.85,
                                                current_prediction=False, display_state='stale')
            state.update(run={**state['run'], 'id': f'usability-{scenario}', 'mode': 'dispatcher', 'status': 'paused'},
                         vehicles=sample, incidents=[], network_version=f'usability-{scenario}')
            network.update(network_version=state['network_version'], segments=segments)
            state['segment_risks'] = {item['prediction']['target_segment']['id']: item['prediction']['risk'] for item in sample}
            for item in sample:
                if item['prediction']['risk'] in {'red', 'yellow'}:
                    p = item['prediction']
                    state['incidents'].append({**current_incident, 'id': f'i-{item['id']}', 'tr_id': item['id'],
                                              'target_visit_id': p['target_visit_id'], 'prediction_id': p['id'],
                                              'prediction': p, 'risk': p['risk'], 'p_late': p['p_late'], 'status': 'active',
                                              'target_segment': p['target_segment'], 'reason': 'Наблюдается снижение скорости',
                                              'prediction_delay_s': p['prediction_delay_s']})
            publish()
            expect(page.locator('.vehicle-row')).to_have_count(3)
            expect(page.locator('.network-segment')).to_have_count(4)
            expected_count = '2' if scenario == 'multiple-risks' else '0' if scenario == 'stale' else '1'
            expect(page.get_by_test_id('high-risk-count')).to_have_text(expected_count)
            if scenario == 'stale':
                page.locator('[data-vehicle-id="v1"]').click()
                expect(page.get_by_test_id('card-probability')).to_have_text('—')
            page.evaluate('window.scrollTo(0, 0)')
            page.screenshot(path=str(usability / 'screens' / f'{scenario}.png'), full_page=False, animations='disabled')
            answer_key['scenarios'].append({'id': scenario,
                'high_risk_vehicles': [item['id'] for item in sample if item['prediction']['risk'] == 'red'],
                'high_risk_segments': [f"{item['prediction']['target_segment']['from_name']} → {item['prediction']['target_segment']['to_name']}" for item in sample if item['prediction']['risk'] == 'red'],
                'stale_vehicle': 'v1' if scenario == 'stale' else None, 'current_probability_for_stale_vehicle': None})
        (usability / 'answer_key.json').write_text(json.dumps(answer_key, ensure_ascii=False, indent=2), encoding='utf-8')
        results = usability / 'results.csv'
        if not results.exists():
            results.write_text('status,participant_id,scenario_id,response_time_s,vehicles_answer,segments_answer,stale_answer,correct,notes\nNOT_RUN,,,,,,,,Awaiting human participants\n', encoding='utf-8')
        assert not errors, errors
        browser.close()
    result = {'passed': len(checks), 'checks': checks, 'page_errors': errors, 'isolation': 'built frontend + intercepted API fixtures; no live stack requests',
              'usability_5_second_test': 'not performed; requires human participants'}
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
