import { useEffect, useState } from "react";
import { Activity, BarChart3, BookOpen, FileUp, Gauge, Layers } from "lucide-react";
import {
  fetchEngineDetail,
  fetchFleetEngines,
  fetchHealth,
  fetchMetrics,
  fetchMetricsBreakdown,
  fetchTrajectory,
} from "./api/client";
import { BatchUploadView } from "./components/BatchUploadView";
import { BenchmarksView } from "./components/BenchmarksView";
import { EngineDetailView } from "./components/EngineDetailView";
import { FleetTriage } from "./components/FleetTriage";
import { GuideView } from "./components/GuideView";
import type {
  DatasetName,
  EngineDetail,
  EngineSummary,
  MetricRow,
  MetricsBreakdownRow,
  TrajectoryPoint,
} from "./types/turbofan";

type Tab = "triage" | "detail" | "benchmarks" | "upload" | "guide";
const DATASETS: DatasetName[] = ["FD001", "FD002", "FD003", "FD004"];

export function App() {
  const [dataset, setDataset] = useState<DatasetName>("FD001");
  const [tab, setTab] = useState<Tab>("triage");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  const [engines, setEngines] = useState<EngineSummary[]>([]);
  const [fleetLoading, setFleetLoading] = useState(true);
  const [fleetError, setFleetError] = useState<string | null>(null);

  const [selectedUnit, setSelectedUnit] = useState<number | null>(null);
  const [engineDetail, setEngineDetail] = useState<EngineDetail | null>(null);
  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([]);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [breakdown, setBreakdown] = useState<MetricsBreakdownRow[]>([]);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
  }, []);

  useEffect(() => {
    let cancelled = false;
    setFleetLoading(true);
    setFleetError(null);
    setSelectedUnit(null);
    setEngineDetail(null);
    setTrajectory([]);

    fetchFleetEngines(dataset)
      .then((res) => {
        if (!cancelled) {
          setEngines(res.engines);
          setApiOnline(true);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setFleetError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setFleetLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [dataset]);

  useEffect(() => {
    if (tab !== "benchmarks" || metrics.length > 0) return;
    let cancelled = false;
    setMetricsLoading(true);
    setMetricsError(null);

    Promise.all([fetchMetrics(), fetchMetricsBreakdown()])
      .then(([rows, bd]) => {
        if (!cancelled) {
          setMetrics(rows);
          setBreakdown(bd);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setMetricsError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setMetricsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [tab, metrics.length]);

  function handleSelectEngine(unitNumber: number) {
    setSelectedUnit(unitNumber);
    setTab("detail");
    setDetailLoading(true);
    setDetailError(null);

    Promise.all([
      fetchEngineDetail(dataset, unitNumber),
      fetchTrajectory(dataset, unitNumber),
    ])
      .then(([detail, traj]) => {
        setEngineDetail(detail);
        setTrajectory(traj.points);
      })
      .catch((err: unknown) =>
        setDetailError(err instanceof Error ? err.message : String(err))
      )
      .finally(() => setDetailLoading(false));
  }

  const selectedSummary =
    engines.find((e) => e.unit_number === selectedUnit) ?? null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600/20 text-indigo-400">
              <Gauge className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-semibold text-slate-100">
                Turbofan RUL Predictive Maintenance
              </h1>
              <p className="text-xs text-slate-400">
                C-MAPSS Fleet Triage &amp; Model Explorer
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950 p-0.5 text-xs">
              {DATASETS.map((ds) => (
                <button
                  key={ds}
                  type="button"
                  onClick={() => setDataset(ds)}
                  className={`rounded-md px-3 py-1.5 font-medium transition ${
                    dataset === ds
                      ? "bg-indigo-600 text-white"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {ds}
                </button>
              ))}
            </div>

            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs ${
                apiOnline
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                  : apiOnline === false
                    ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
                    : "border-slate-700 bg-slate-800 text-slate-400"
              }`}
            >
              <Activity className="h-3.5 w-3.5" />
              {apiOnline ? "API Connected" : apiOnline === false ? "API Offline" : "Checking..."}
            </span>
          </div>
        </div>

        <nav className="mx-auto flex max-w-7xl gap-1 px-6">
          <button
            type="button"
            onClick={() => setTab("triage")}
            className={`inline-flex items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium transition ${
              tab === "triage"
                ? "border-indigo-500 text-indigo-300"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <Layers className="h-4 w-4" /> Fleet Triage
          </button>
          <button
            type="button"
            onClick={() => setTab("detail")}
            className={`inline-flex items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium transition ${
              tab === "detail"
                ? "border-indigo-500 text-indigo-300"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <Gauge className="h-4 w-4" /> Engine Detail
            {selectedUnit !== null && (
              <span className="rounded bg-slate-800 px-1.5 py-0.5 text-xs text-slate-300">
                #{selectedUnit}
              </span>
            )}
          </button>
          <button
            type="button"
            onClick={() => setTab("benchmarks")}
            className={`inline-flex items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium transition ${
              tab === "benchmarks"
                ? "border-indigo-500 text-indigo-300"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <BarChart3 className="h-4 w-4" /> Benchmarks
          </button>
          <button
            type="button"
            onClick={() => setTab("upload")}
            className={`inline-flex items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium transition ${
              tab === "upload"
                ? "border-indigo-500 text-indigo-300"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <FileUp className="h-4 w-4" /> Batch Upload
          </button>
          <button
            type="button"
            onClick={() => setTab("guide")}
            className={`inline-flex items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium transition ${
              tab === "guide"
                ? "border-indigo-500 text-indigo-300"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <BookOpen className="h-4 w-4" /> Guide
          </button>
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-6">
        {tab === "triage" && (
          <FleetTriage
            engines={engines}
            loading={fleetLoading}
            error={fleetError}
            onSelectEngine={handleSelectEngine}
          />
        )}
        {tab === "detail" && (
          <EngineDetailView
            summary={selectedSummary}
            detail={engineDetail}
            trajectory={trajectory}
            loading={detailLoading}
            error={detailError}
            onBack={() => setTab("triage")}
          />
        )}
        {tab === "benchmarks" && (
          <BenchmarksView
            dataset={dataset}
            engines={engines}
            metrics={metrics}
            breakdown={breakdown}
            loading={metricsLoading}
            error={metricsError}
          />
        )}
        {tab === "upload" && <BatchUploadView dataset={dataset} />}
        {tab === "guide" && <GuideView />}
      </main>
    </div>
  );
}

export default App;
