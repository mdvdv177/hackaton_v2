export type Risk = 'red' | 'yellow' | 'green' | 'gray';
export type ReplayMode = 'dispatcher' | 'evaluation' | 'live';
export type GeometryKind = 'schedule_schematic' | 'supplied_route' | 'mixed' | string;

export interface Factor {
  name?: string; feature?: string; label?: string;
  value?: number | string | null; contribution?: number;
}

export interface Segment {
  id: string;
  path_id?: string;
  from_visit_id?: string;
  to_visit_id?: string;
  from_name?: string;
  to_name?: string;
  geometry_kind?: GeometryKind;
  geometry?: { type: 'LineString'; coordinates: [number, number][] };
  segment_speed_kmh?: number | null;
  speed_coverage?: number;
  speed_mean_kmh?: number | null;
  coverage_ratio?: number | null;
  mean_speed_kmh?: number | null;
}

export interface Prediction {
  id?: string;
  run_id?: string;
  prediction_time: string;
  target_visit_id: string;
  target_name?: string;
  target_lat?: number | null;
  target_lon?: number | null;
  target_time_begin: string;
  prediction_delay_s: number | null;
  predicted_arrival_at?: string | null;
  p_late: number | null;
  risk: Risk;
  last_known_p_late?: number | null;
  last_known_risk?: Risk;
  source: string;
  status?: string;
  freshness?: string;
  display_state?: 'current' | 'monitoring' | 'stale' | 'prediction_stale' | 'retrospective';
  current_prediction?: boolean;
  horizon_s?: number;
  publication_horizon_s?: number | null;
  timing_status?: 'verified' | 'retrospective' | 'unknown_legacy';
  generated_at?: string;
  published_at?: string;
  model_version?: string;
  factors?: (Factor | string)[];
  target_segment?: Segment | null;
  current_segment?: Segment | null;
  incident_id?: string | null;
}

export interface Vehicle {
  id: string;
  lat: number | null;
  lon: number | null;
  speed: number | null;
  event_time: string | null;
  received_at?: string;
  packet_id?: string;
  position_age_s?: number | null;
  telemetry_age_s?: number | null;
  stale: boolean;
  cur_dev_s: number | null;
  cur_dev_source: string;
  prediction: Prediction | null;
  current_prediction_id?: string | null;
  current_segment?: Segment | null;
  target_segment?: Segment | null;
}

export interface Visit {
  id: string;
  name: string;
  lat: number;
  lon: number;
  time_begin?: string;
  tr_id?: string;
}

export interface VehicleDetails extends Vehicle {
  telemetry: { event_time: string; lat: number | null; lon: number | null; speed: number | null; location_valid?: boolean }[];
  history: Prediction[];
  planned_visits: Visit[];
}

export interface Incident {
  id: string;
  run_id?: string;
  tr_id: string;
  target_visit_id: string;
  risk: Risk;
  status: string;
  freshness?: string;
  acknowledged: boolean;
  prediction_id?: string;
  prediction?: Prediction;
  prediction_delay_s: number | null;
  p_late: number | null;
  last_known_p_late?: number | null;
  last_known_risk?: Risk;
  target_name?: string;
  target_time_begin?: string;
  target_segment?: Segment | null;
  predicted_arrival_at?: string | null;
  prediction_time: string;
  timing_status?: string;
  reason: string;
  recommendation: string;
}

export interface Run {
  id: string;
  mode: string;
  source?: string;
  status: string;
  virtual_time: string | null;
  speed: number;
  scenario_id?: string;
  scenario_name?: string;
  schedule_mode?: 'as_is' | 'demo_rebased' | null;
  provenance?: string;
}

export interface SystemStatus {
  backend_status?: string;
  ml_status?: string;
  model_version?: string | null;
  active_vehicles?: number;
  stale_vehicles?: number;
  processing_latency_ms?: number | null;
  processing_latency_p95_ms?: number | null;
  queue_depth?: number;
  telemetry_count?: number;
  ndtp_errors?: number;
  mae_s?: number | null;
  evaluation_mae_s?: number | null;
}

export interface SegmentRisk {
  segment_id?: string;
  id?: string;
  risk: Risk;
  tr_id?: string;
  tr_ids?: string[];
  incident_ids?: string[];
}

export interface Network {
  run_id?: string | null;
  network_version: string;
  geometry_kind: GeometryKind;
  paths: { id: string; name?: string; tr_id?: string }[];
  visits: Visit[];
  segments: Segment[];
}

export interface Scenario {
  id: string;
  name: string;
  description?: string;
  kind?: string;
  provenance?: string;
  geometry_kind?: string;
}

export interface Snapshot {
  run: Run | null;
  vehicles: Vehicle[];
  incidents: Incident[];
  system: SystemStatus;
  event_id?: number;
  incidents_total?: number;
  published_at?: string;
  network_version?: string;
  segment_risks?: SegmentRisk[] | Record<string, Risk | SegmentRisk>;
}
