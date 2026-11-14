import { useMemo, useState } from "react";
import type * as echarts from "echarts";
import type {
  DatasetName,
  EngineSummary,
  MetricRow,
  MetricsBreakdownRow,
} from "../types/turbofan";
import { EChartBox } from "./EChartBox";
import { InfoTip } from "./InfoTip";

interface BenchmarksViewProps {
  dataset: DatasetName;
  engines: EngineSummary[];
  metrics: MetricRow[];
  breakdown: MetricsBreakdownRow[];
  loading: boolean;
  error: string | null;
}

export function BenchmarksView({
  dataset,
  engines,
  metrics,
  breakdown,
  loading,
  error,
}: BenchmarksViewProps) {
  const [scope, setScope] = useState<"all" | "within_cap">("all");
  const [threshold, setThreshold] = useState(25);

  const chartData = useMemo(() => {
    const datasets: DatasetName[] = ["FD001", "FD002", "FD003", "FD004"];
    if (breakdown.length > 0) {
      const getVal = (
        ds: DatasetName,
        model: "GBR" | "LSTM",
        metric: "RMSE" | "PHM08"
      ) => {
        const row = breakdown.find((r) => r.dataset === ds && r.model === model);
        if (!row) return 0;
        if (metric === "RMSE") {
          return scope === "all" ? row.RMSE_all : row.RMSE_within_cap;
        }
        return scope === "all" ? row.PHM08_all : row.PHM08_within_cap;
      };
      return {
        datasets,
        gbrRmse: datasets.map((d) => getVal(d, "GBR", "RMSE")),
        lstmRmse: datasets.map((d) => getVal(d, "LSTM", "RMSE")),
        gbrPhm: datasets.map((d) => getVal(d, "GBR", "PHM08")),
        lstmPhm: datasets.map((d) => getVal(d, "LSTM", "PHM08")),
      };
    }

    const getFallback = (
      ds: DatasetName,
      model: "GBR" | "LSTM",
      metric: "RMSE" | "PHM08_score"
    ) => {
      const row = metrics.find((r) => r.dataset === ds && r.model === model);
      return row ? Number(row[metric].toFixed(2)) : 0;
    };
    return {
      datasets,
      gbrRmse: datasets.map((d) => getFallback(d, "GBR", "RMSE")),
      lstmRmse: datasets.map((d) => getFallback(d, "LSTM", "RMSE")),
      gbrPhm: datasets.map((d) => getFallback(d, "GBR", "PHM08_score")),
      lstmPhm: datasets.map((d) => getFallback(d, "LSTM", "PHM08_score")),
    };
  }, [breakdown, metrics, scope]);

  const rmseOption = useMemo<echarts.EChartsOption>(
    () => ({
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" },
      legend: { textStyle: { color: "#cbd5e1", fontSize: 11 }, top: 0 },
      grid: { left: 45, right: 16, top: 32, bottom: 28 },
      xAxis: {
        type: "category",
        data: chartData.datasets,
        axisLabel: { color: "#94a3b8" },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#94a3b8" },
        splitLine: { lineStyle: { color: "#1e293b" } },
      },
      series: [
        {
          name: "GBR",
          type: "bar",
          data: chartData.gbrRmse,
          itemStyle: { color: "#38bdf8", borderRadius: [4, 4, 0, 0] },
        },
        {
          name: "LSTM",
          type: "bar",
          data: chartData.lstmRmse,
          itemStyle: { color: "#818cf8", borderRadius: [4, 4, 0, 0] },
        },
      ],
    }),
    [chartData]
  );

  const phmOption = useMemo<echarts.EChartsOption>(
    () => ({
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" },
      legend: { textStyle: { color: "#cbd5e1", fontSize: 11 }, top: 0 },
      grid: { left: 56, right: 16, top: 32, bottom: 28 },
      xAxis: {
        type: "category",
        data: chartData.datasets,
        axisLabel: { color: "#94a3b8" },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#94a3b8" },
        splitLine: { lineStyle: { color: "#1e293b" } },
      },
      series: [
        {
          name: "GBR",
          type: "bar",
          data: chartData.gbrPhm,
          itemStyle: { color: "#38bdf8", borderRadius: [4, 4, 0, 0] },
        },
        {
          name: "LSTM",
          type: "bar",
          data: chartData.lstmPhm,
          itemStyle: { color: "#818cf8", borderRadius: [4, 4, 0, 0] },
        },
      ],
    }),
    [chartData]
  );

  // Alert threshold simulator on active dataset's engines
  const thresholdStats = useMemo(() => {
    const withTrue = engines.filter((e) => e.true_rul !== null);
    const computeFor = (key: "gbr_rul" | "lstm_rul") => {
      let lateMissed = 0; // true_rul <= threshold, predicted > threshold (dangerous!)
      let earlyFalse = 0; // true_rul > threshold, predicted <= threshold (conservative)
      let trueAlerts = 0;
      for (const e of withTrue) {
        const t = e.true_rul as number;
        const p = e[key];
        if (t <= threshold && p > threshold) lateMissed += 1;
        else if (t > threshold && p <= threshold) earlyFalse += 1;
        else if (t <= threshold && p <= threshold) trueAlerts += 1;
      }
      return { total: withTrue.length, lateMissed, earlyFalse, trueAlerts };
    };
    return { gbr: computeFor("gbr_rul"), lstm: computeFor("lstm_rul") };
  }, [engines, threshold]);

  if (loading) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-8 text-center text-slate-400">
        Loading benchmark metrics...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-6 text-rose-300">
        Failed to load metrics: {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">
            Model Benchmarks (FD001–FD004)
          </h2>
          <p className="text-sm text-slate-400">
            Compare overall test-set metrics vs. engines within the 125-cycle training cap.
          </p>
        </div>

        <div className="inline-flex rounded-lg border border-slate-800 bg-slate-900 p-0.5 text-xs">
          <button
            type="button"
            onClick={() => setScope("all")}
            className={`rounded-md px-3 py-1.5 font-medium ${
              scope === "all" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
            }`}
          >
            All Engines
          </button>
          <button
            type="button"
            onClick={() => setScope("within_cap")}
            className={`rounded-md px-3 py-1.5 font-medium ${
              scope === "within_cap"
                ? "bg-indigo-600 text-white"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Within Cap (RUL ≤ 125)
          </button>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <h3 className="mb-2 text-xs font-medium uppercase text-slate-400">
            RMSE (cycles — lower is better)
          </h3>
          <EChartBox option={rmseOption} height={250} />
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <h3 className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase text-slate-400">
            PHM08 Asymmetric Score (lower is better)
            <InfoTip text="Safety-weighted score that penalizes overestimating remaining life (late warning) much more heavily than an early conservative warning." />
          </h3>
          <EChartBox option={phmOption} height={250} />
        </div>
      </div>

      {/* Alert Threshold Simulator */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-medium text-slate-100">
              Maintenance Alert Threshold Simulator ({dataset})
            </h3>
            <p className="text-xs text-slate-400">
              Adjust the RUL alert cutoff to compare missed late warnings (high risk) vs.
              early conservative alerts.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="font-mono text-sm text-indigo-300">
              Threshold: {threshold} cycles
            </span>
            <input
              type="range"
              min={10}
              max={60}
              step={5}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-40 cursor-pointer accent-indigo-500"
            />
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          {(
            [
              { name: "LSTM", stats: thresholdStats.lstm, color: "text-indigo-300" },
              { name: "GBR", stats: thresholdStats.gbr, color: "text-sky-300" },
            ] as const
          ).map(({ name, stats, color }) => (
            <div
              key={name}
              className="rounded-lg border border-slate-800 bg-slate-950/70 p-4"
            >
              <div className={`text-sm font-semibold ${color}`}>{name}</div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
                <div className="rounded bg-slate-900 p-2">
                  <div className="text-slate-400">True Alerts</div>
                  <div className="mt-1 font-mono text-lg text-emerald-300">
                    {stats.trueAlerts}
                  </div>
                </div>
                <div className="rounded bg-slate-900 p-2">
                  <div className="text-slate-400">Late / Missed</div>
                  <div className="mt-1 font-mono text-lg text-rose-400">
                    {stats.lateMissed}
                  </div>
                </div>
                <div className="rounded bg-slate-900 p-2">
                  <div className="text-slate-400">Early / False</div>
                  <div className="mt-1 font-mono text-lg text-amber-300">
                    {stats.earlyFalse}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
