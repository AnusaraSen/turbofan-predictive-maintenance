import type {
  BatchResultItem,
  CycleRow,
  DatasetName,
  EngineDetail,
  FeatureImportanceItem,
  FleetResponse,
  HealthResponse,
  LSTMExplainResponse,
  MetricRow,
  MetricsBreakdownRow,
  PredictResponse,
  TrajectoryResponse,
} from "../types/turbofan";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export function fetchHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/health");
}

export function fetchFleetEngines(dataset: DatasetName): Promise<FleetResponse> {
  return getJson<FleetResponse>(`/datasets/${dataset}/engines`);
}

export function fetchEngineDetail(
  dataset: DatasetName,
  unitId: number
): Promise<EngineDetail> {
  return getJson<EngineDetail>(`/datasets/${dataset}/engines/${unitId}`);
}

export async function fetchMetrics(): Promise<MetricRow[]> {
  const data = await getJson<{ metrics: MetricRow[] }>("/metrics");
  return data.metrics;
}

export async function fetchTrajectory(
  dataset: DatasetName,
  unitId: number
): Promise<TrajectoryResponse> {
  const res = await fetch(`${API_BASE}/predict/trajectory`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_name: dataset, unit_number: unitId }),
  });
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<TrajectoryResponse>;
}

export async function predictWindow(
  dataset: DatasetName,
  modelType: "gbr" | "lstm",
  unitNumber: number,
  readings: CycleRow[]
): Promise<PredictResponse> {
  const payloadReadings = readings.map((r) => {
    const { unit_number: _u, RUL: _r, regime_id: _reg, ...rest } = r;
    return rest;
  });
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_name: dataset,
      model_type: modelType,
      unit_number: unitNumber,
      readings: payloadReadings,
    }),
  });
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<PredictResponse>;
}

export async function fetchFeatureImportance(
  dataset: DatasetName
): Promise<FeatureImportanceItem[]> {
  const data = await getJson<{ features: FeatureImportanceItem[] }>(
    `/feature-importance/${dataset}`
  );
  return data.features;
}

export async function fetchLSTMExplain(
  dataset: DatasetName,
  unitNumber: number,
  readings: CycleRow[]
): Promise<LSTMExplainResponse> {
  const payloadReadings = readings.map((r) => {
    const { unit_number: _u, RUL: _r, regime_id: _reg, ...rest } = r;
    return rest;
  });
  const res = await fetch(`${API_BASE}/explain/lstm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_name: dataset,
      unit_number: unitNumber,
      readings: payloadReadings,
    }),
  });
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<LSTMExplainResponse>;
}

export async function predictBatch(
  dataset: DatasetName,
  rows: Record<string, number>[]
): Promise<BatchResultItem[]> {
  const res = await fetch(`${API_BASE}/predict/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_name: dataset, rows }),
  });
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as { results: BatchResultItem[] };
  return data.results;
}

export async function fetchMetricsBreakdown(): Promise<MetricsBreakdownRow[]> {
  const data = await getJson<{ breakdown: MetricsBreakdownRow[] }>(
    "/metrics/breakdown"
  );
  return data.breakdown;
}
