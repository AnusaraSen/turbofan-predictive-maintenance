export type DatasetName = "FD001" | "FD002" | "FD003" | "FD004";

export type EngineStatus = "Urgent" | "Monitor" | "Healthy";

export interface HealthResponse {
  status: string;
  datasets: DatasetName[];
}

export interface EngineSummary {
  unit_number: number;
  total_cycles: number;
  gbr_rul: number;
  lstm_rul: number;
  true_rul: number | null;
  status: EngineStatus;
  capped: boolean;
}

export interface FleetResponse {
  dataset_name: DatasetName;
  engines: EngineSummary[];
}

export type CycleRow = {
  unit_number: number;
  time_cycles: number;
  regime_id: number;
  RUL?: number | null;
  [key: string]: number | null | undefined;
};

export interface EngineDetail {
  dataset_name: DatasetName;
  unit_number: number;
  total_cycles: number;
  informative_sensors: string[];
  raw_rows: CycleRow[];
  normalized_rows: CycleRow[];
}

export interface MetricRow {
  dataset: DatasetName;
  model: "GBR" | "LSTM";
  RMSE: number;
  PHM08_score: number;
}

export interface TrajectoryPoint {
  time_cycles: number;
  regime_id: number;
  gbr_rul: number;
  lstm_rul: number;
  true_rul: number | null;
  status: EngineStatus;
  capped: boolean;
}

export interface TrajectoryResponse {
  dataset_name: DatasetName;
  unit_number: number;
  points: TrajectoryPoint[];
}

export interface PredictResponse {
  unit_number: number;
  dataset_name: DatasetName;
  model_type: "gbr" | "lstm";
  predicted_RUL: number;
  capped: boolean;
  note?: string | null;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface LSTMExplainResponse {
  dataset_name: DatasetName;
  unit_number: number;
  base_rul: number;
  sensors: string[];
  cycle_bins: string[];
  matrix: [number, number, number][];
}

export interface BatchResultItem {
  unit_number: number;
  total_cycles: number;
  gbr_rul: number;
  lstm_rul: number;
  status: EngineStatus;
  capped: boolean;
}

export interface MetricsBreakdownRow {
  dataset: DatasetName;
  model: "GBR" | "LSTM";
  n_all: number;
  n_within_cap: number;
  RMSE_all: number;
  PHM08_all: number;
  RMSE_within_cap: number;
  PHM08_within_cap: number;
}
