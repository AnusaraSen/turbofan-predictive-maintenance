import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Brain, Pause, Play, RotateCcw, Sliders } from "lucide-react";
import type * as echarts from "echarts";
import {
  fetchFeatureImportance,
  fetchLSTMExplain,
  predictWindow,
} from "../api/client";
import type {
  CycleRow,
  EngineDetail,
  EngineSummary,
  FeatureImportanceItem,
  LSTMExplainResponse,
  TrajectoryPoint,
} from "../types/turbofan";
import { EChartBox } from "./EChartBox";
import { InfoTip } from "./InfoTip";
import { CappedTag, StatusBadge } from "./StatusBadge";

interface EngineDetailViewProps {
  summary: EngineSummary | null;
  detail: EngineDetail | null;
  trajectory: TrajectoryPoint[];
  loading: boolean;
  error: string | null;
  onBack: () => void;
}

const REGIME_COLORS = [
  "rgba(99, 102, 241, 0.10)",
  "rgba(16, 185, 129, 0.10)",
  "rgba(245, 158, 11, 0.10)",
  "rgba(236, 72, 153, 0.10)",
  "rgba(14, 165, 233, 0.10)",
  "rgba(168, 85, 247, 0.10)",
];

const LINE_COLORS = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f472b6"];

export function EngineDetailView({
  summary,
  detail,
  trajectory,
  loading,
  error,
  onBack,
}: EngineDetailViewProps) {
  const [normalized, setNormalized] = useState(true);
  const [selectedSensors, setSelectedSensors] = useState<string[]>([]);
  const [cursorIdx, setCursorIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<1 | 5 | 10>(5);

  // What-If sensor percentage offsets (-10% to +10%)
  const [offsets, setOffsets] = useState<Record<string, number>>({});
  const [whatIfResult, setWhatIfResult] = useState<{
    gbr: number;
    lstm: number;
  } | null>(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);

  const [featureImp, setFeatureImp] = useState<FeatureImportanceItem[]>([]);
  const [lstmExplain, setLstmExplain] = useState<LSTMExplainResponse | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);

  // Initialize defaults when engine changes
  useEffect(() => {
    if (!detail) return;
    setSelectedSensors(detail.informative_sensors.slice(0, 3));
    setCursorIdx(Math.max(0, detail.total_cycles - 1));
    setPlaying(false);
    setOffsets({});
    setWhatIfResult(null);
    setLstmExplain(null);
    fetchFeatureImportance(detail.dataset_name)
      .then((items) => setFeatureImp(items.slice(0, 10)))
      .catch(() => setFeatureImp([]));
  }, [detail]);

  // Playback timer
  useEffect(() => {
    if (!playing || !detail) return;
    const maxIdx = detail.total_cycles - 1;
    if (cursorIdx >= maxIdx) {
      setPlaying(false);
      return;
    }
    const intervalMs = Math.max(40, Math.round(300 / speed));
    const timer = window.setInterval(() => {
      setCursorIdx((prev) => {
        if (prev >= maxIdx) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [playing, speed, detail, cursorIdx]);

  // Debounced What-If prediction when sliders move
  const hasActiveOffsets = useMemo(
    () => Object.values(offsets).some((v) => Math.abs(v) > 0.001),
    [offsets]
  );

  useEffect(() => {
    if (!detail || !hasActiveOffsets) {
      setWhatIfResult(null);
      return;
    }
    const startIdx = Math.max(0, cursorIdx - 29);
    const windowRows: CycleRow[] = detail.raw_rows
      .slice(startIdx, cursorIdx + 1)
      .map((row) => {
        const copy: CycleRow = { ...row };
        for (const [sensor, pct] of Object.entries(offsets)) {
          const val = copy[sensor];
          if (typeof val === "number" && pct !== 0) {
            copy[sensor] = val * (1 + pct / 100);
          }
        }
        return copy;
      });

    let cancelled = false;
    setWhatIfLoading(true);
    const timer = window.setTimeout(() => {
      Promise.all([
        predictWindow(detail.dataset_name, "gbr", detail.unit_number, windowRows),
        predictWindow(detail.dataset_name, "lstm", detail.unit_number, windowRows),
      ])
        .then(([gbrRes, lstmRes]) => {
          if (!cancelled) {
            setWhatIfResult({
              gbr: gbrRes.predicted_RUL,
              lstm: lstmRes.predicted_RUL,
            });
          }
        })
        .catch(() => {
          if (!cancelled) setWhatIfResult(null);
        })
        .finally(() => {
          if (!cancelled) setWhatIfLoading(false);
        });
    }, 250);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [detail, cursorIdx, offsets, hasActiveOffsets]);

  const currentPoint = trajectory[cursorIdx] ?? null;

  const sensorOption = useMemo<echarts.EChartsOption>(() => {
    if (!detail) return {};
    const rows = normalized ? detail.normalized_rows : detail.raw_rows;
    const cycles = rows.map((r) => r.time_cycles);
    const activeCycle = cycles[cursorIdx] ?? cycles[cycles.length - 1];
    const windowStartCycle = cycles[Math.max(0, cursorIdx - 29)] ?? cycles[0];

    // Contiguous regime background bands when multiple regimes exist
    const uniqueRegimes = new Set(rows.map((r) => r.regime_id));
    const markAreaData: unknown[] = [];
    if (uniqueRegimes.size > 1 && rows.length > 0) {
      let spanStart = rows[0].time_cycles;
      let prevRegime = rows[0].regime_id;
      for (let i = 1; i < rows.length; i++) {
        if (rows[i].regime_id !== prevRegime) {
          markAreaData.push([
            {
              xAxis: spanStart,
              itemStyle: { color: REGIME_COLORS[prevRegime % REGIME_COLORS.length] },
            },
            { xAxis: rows[i - 1].time_cycles },
          ]);
          spanStart = rows[i].time_cycles;
          prevRegime = rows[i].regime_id;
        }
      }
      markAreaData.push([
        {
          xAxis: spanStart,
          itemStyle: { color: REGIME_COLORS[prevRegime % REGIME_COLORS.length] },
        },
        { xAxis: rows[rows.length - 1].time_cycles },
      ]);
    }

    // Highlight active 30-cycle lookback window
    markAreaData.push([
      {
        name: "30-Cycle Window",
        xAxis: windowStartCycle,
        itemStyle: {
          color: "rgba(99, 102, 241, 0.18)",
          borderColor: "rgba(129, 140, 248, 0.7)",
          borderWidth: 1,
        },
      },
      { xAxis: activeCycle },
    ]);

    const series: echarts.SeriesOption[] = selectedSensors.map((sensor, idx) => {
      const vals = rows.map((r) => {
        const v = r[sensor];
        return typeof v === "number" ? Number(v.toFixed(3)) : null;
      });
      return {
        name: sensor,
        type: "line",
        data: vals,
        showSymbol: false,
        smooth: false,
        lineStyle: { width: 1.8, color: LINE_COLORS[idx % LINE_COLORS.length] },
        itemStyle: { color: LINE_COLORS[idx % LINE_COLORS.length] },
        ...(idx === 0 ? { markArea: { silent: true, data: markAreaData as never } } : {}),
      };
    });

    return {
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" },
      legend: { textStyle: { color: "#cbd5e1", fontSize: 11 }, top: 0 },
      grid: { left: 48, right: 20, top: 32, bottom: 28 },
      xAxis: {
        type: "category",
        data: cycles,
        axisLabel: { color: "#94a3b8", fontSize: 11 },
        axisLine: { lineStyle: { color: "#334155" } },
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLabel: { color: "#94a3b8", fontSize: 11 },
        splitLine: { lineStyle: { color: "#1e293b" } },
      },
      series,
    };
  }, [detail, normalized, selectedSensors, cursorIdx]);

  const rulOption = useMemo<echarts.EChartsOption>(() => {
    if (!trajectory.length) return {};
    const visible = trajectory.slice(0, cursorIdx + 1);
    const cycles = trajectory.map((p) => p.time_cycles);

    const hasTrue = trajectory.some((p) => p.true_rul !== null);
    const series: echarts.SeriesOption[] = [
      {
        name: "LSTM RUL",
        type: "line",
        data: visible.map((p) => p.lstm_rul),
        showSymbol: false,
        lineStyle: { width: 2.5, color: "#818cf8" },
        itemStyle: { color: "#818cf8" },
        markLine: {
          silent: true,
          symbol: "none",
          data: [
            {
              yAxis: 125,
              lineStyle: { type: "dashed", color: "#38bdf8" },
              label: { formatter: "Cap 125", color: "#38bdf8", fontSize: 10 },
            },
            {
              yAxis: 25,
              lineStyle: { type: "dashed", color: "#f43f5e" },
              label: { formatter: "Urgent 25", color: "#f43f5e", fontSize: 10 },
            },
          ],
        },
      },
      {
        name: "GBR RUL",
        type: "line",
        data: visible.map((p) => p.gbr_rul),
        showSymbol: false,
        lineStyle: { width: 2, color: "#38bdf8" },
        itemStyle: { color: "#38bdf8" },
      },
    ];

    if (hasTrue) {
      series.push({
        name: "True RUL",
        type: "line",
        data: visible.map((p) => p.true_rul),
        showSymbol: false,
        lineStyle: { width: 1.8, type: "dotted", color: "#34d399" },
        itemStyle: { color: "#34d399" },
      });
    }

    return {
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" },
      legend: { textStyle: { color: "#cbd5e1", fontSize: 11 }, top: 0 },
      grid: { left: 48, right: 24, top: 32, bottom: 28 },
      xAxis: {
        type: "category",
        data: cycles,
        axisLabel: { color: "#94a3b8", fontSize: 11 },
        axisLine: { lineStyle: { color: "#334155" } },
      },
      yAxis: {
        type: "value",
        min: 0,
        axisLabel: { color: "#94a3b8", fontSize: 11 },
        splitLine: { lineStyle: { color: "#1e293b" } },
      },
      series,
    };
  }, [trajectory, cursorIdx]);

  const gbrFeatureOption = useMemo<echarts.EChartsOption>(() => {
    if (!featureImp.length) return {};
    const reversed = [...featureImp].reverse();
    return {
      backgroundColor: "transparent",
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      grid: { left: 175, right: 16, top: 10, bottom: 24 },
      xAxis: {
        type: "value",
        axisLabel: { color: "#94a3b8", fontSize: 10 },
        splitLine: { lineStyle: { color: "#1e293b" } },
      },
      yAxis: {
        type: "category",
        data: reversed.map((f) => f.feature),
        axisLabel: { color: "#cbd5e1", fontSize: 10 },
        axisLine: { lineStyle: { color: "#334155" } },
      },
      series: [
        {
          type: "bar",
          data: reversed.map((f) => Number(f.importance.toFixed(4))),
          itemStyle: { color: "#38bdf8", borderRadius: [0, 4, 4, 0] },
        },
      ],
    };
  }, [featureImp]);

  const lstmHeatmapOption = useMemo<echarts.EChartsOption>(() => {
    if (!lstmExplain) return {};
    const maxVal = Math.max(0.1, ...lstmExplain.matrix.map((m) => m[2]));
    return {
      backgroundColor: "transparent",
      tooltip: {
        position: "top",
        formatter: (params: unknown) => {
          const p = params as { value: [number, number, number] };
          const [bIdx, sIdx, val] = p.value;
          return `${lstmExplain.sensors[sIdx]} (${lstmExplain.cycle_bins[bIdx]}): Δ ${val.toFixed(2)} cycles`;
        },
      },
      grid: { left: 72, right: 20, top: 12, bottom: 48 },
      xAxis: {
        type: "category",
        data: lstmExplain.cycle_bins,
        axisLabel: { color: "#94a3b8", fontSize: 10 },
      },
      yAxis: {
        type: "category",
        data: lstmExplain.sensors,
        axisLabel: { color: "#cbd5e1", fontSize: 10 },
      },
      visualMap: {
        min: 0,
        max: maxVal,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        itemWidth: 12,
        itemHeight: 110,
        textStyle: { color: "#94a3b8", fontSize: 10 },
        inRange: { color: ["#0f172a", "#4f46e5", "#f43f5e"] },
      },
      series: [
        {
          type: "heatmap",
          data: lstmExplain.matrix,
          label: { show: false },
        },
      ],
    };
  }, [lstmExplain]);

  function handleComputeLSTMExplain() {
    if (!detail) return;
    const startIdx = Math.max(0, cursorIdx - 29);
    const windowRows = detail.raw_rows.slice(startIdx, cursorIdx + 1);
    setExplainLoading(true);
    fetchLSTMExplain(detail.dataset_name, detail.unit_number, windowRows)
      .then((res) => setLstmExplain(res))
      .catch(() => setLstmExplain(null))
      .finally(() => setExplainLoading(false));
  }

  if (loading) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-8 text-center text-slate-400">
        Computing cycle-by-cycle trajectory...
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-1.5 text-sm text-slate-300 hover:text-white"
        >
          <ArrowLeft className="h-4 w-4" /> Back to Fleet Triage
        </button>
        <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-6 text-rose-300">
          {error}
        </div>
      </div>
    );
  }

  if (!detail || !summary) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-8 text-center text-slate-400">
        Select an engine from the Fleet Triage table to launch the Flight Simulator.
      </div>
    );
  }

  const sliderSensors = detail.informative_sensors.slice(0, 3);

  function toggleSensor(sensor: string) {
    setSelectedSensors((prev) =>
      prev.includes(sensor)
        ? prev.length > 1
          ? prev.filter((s) => s !== sensor)
          : prev
        : [...prev, sensor].slice(0, 5)
    );
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-sm text-slate-300 hover:border-slate-700 hover:text-white"
        >
          <ArrowLeft className="h-4 w-4" /> Back to Fleet Triage
        </button>

        <div className="flex items-center gap-2">
          {currentPoint && <StatusBadge status={currentPoint.status} />}
          {currentPoint?.capped && <CappedTag />}
        </div>
      </div>

      {/* Live Cycle KPIs */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="text-xs text-slate-400">Engine / Cycle</div>
          <div className="mt-1 text-lg font-semibold text-slate-100">
            {detail.dataset_name} · #{detail.unit_number}
          </div>
          <div className="mt-1 font-mono text-xs text-slate-400">
            Cycle {currentPoint?.time_cycles ?? 1} / {detail.total_cycles}
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            Operating Regime
            <InfoTip text="Flight condition cluster (altitude, speed, throttle) detected by K-means so normal flight changes aren't confused with wear." />
          </div>
          <div className="mt-1 font-mono text-2xl font-semibold text-amber-300">
            R{currentPoint?.regime_id ?? 0}
          </div>
          <div className="mt-1 text-xs text-slate-500">K-means cluster</div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            LSTM RUL
            <InfoTip text="Deep-learning prediction reading the last 30 cycles in sequence to spot gradual degradation trends." />
          </div>
          <div className="mt-1 font-mono text-2xl font-semibold text-indigo-300">
            {currentPoint ? currentPoint.lstm_rul.toFixed(1) : summary.lstm_rul.toFixed(1)}
          </div>
          <div className="mt-1 text-xs text-slate-500">30-cycle window</div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="text-xs text-slate-400">GBR RUL</div>
          <div className="mt-1 font-mono text-2xl font-semibold text-sky-300">
            {currentPoint ? currentPoint.gbr_rul.toFixed(1) : summary.gbr_rul.toFixed(1)}
          </div>
          <div className="mt-1 text-xs text-slate-500">baseline features</div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="text-xs text-slate-400">True RUL</div>
          <div className="mt-1 font-mono text-2xl font-semibold text-emerald-300">
            {currentPoint?.true_rul !== null && currentPoint?.true_rul !== undefined
              ? currentPoint.true_rul.toFixed(0)
              : "—"}
          </div>
          <div className="mt-1 text-xs text-slate-500">ground truth</div>
        </div>
      </div>

      {/* Flight Simulator Playback Bar */}
      <div className="flex flex-wrap items-center gap-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              if (!playing && cursorIdx >= detail.total_cycles - 1) {
                setCursorIdx(0);
              }
              setPlaying((p) => !p);
            }}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-500"
          >
            {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            {playing ? "Pause" : "Play"}
          </button>

          <button
            type="button"
            onClick={() => {
              setPlaying(false);
              setCursorIdx(0);
            }}
            className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs text-slate-300 hover:text-white"
          >
            <RotateCcw className="h-3.5 w-3.5" /> Reset
          </button>
        </div>

        <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950 p-0.5 text-xs">
          {([1, 5, 10] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSpeed(s)}
              className={`rounded-md px-2.5 py-1 font-medium ${
                speed === s ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>

        <input
          type="range"
          min={0}
          max={Math.max(0, detail.total_cycles - 1)}
          value={cursorIdx}
          onChange={(e) => {
            setPlaying(false);
            setCursorIdx(Number(e.target.value));
          }}
          className="h-2 flex-1 cursor-pointer accent-indigo-500"
        />
      </div>

      {/* Sensor Traces + Regime Bands + Lookback Window */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-xs font-medium text-slate-400">Sensors (max 5):</span>
            {detail.informative_sensors.slice(0, 8).map((s) => {
              const active = selectedSensors.includes(s);
              return (
                <button
                  key={s}
                  type="button"
                  onClick={() => toggleSensor(s)}
                  className={`rounded-md px-2 py-1 font-mono text-xs transition ${
                    active
                      ? "bg-indigo-600 text-white"
                      : "bg-slate-800 text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {s}
                </button>
              );
            })}
          </div>

          <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950 p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setNormalized(false)}
              className={`rounded-md px-3 py-1 font-medium ${
                !normalized ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              Raw
            </button>
            <button
              type="button"
              onClick={() => setNormalized(true)}
              className={`rounded-md px-3 py-1 font-medium ${
                normalized ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              Regime-Normalized
            </button>
          </div>
        </div>

        <EChartBox option={sensorOption} height={260} />
      </div>

      {/* Live RUL Trajectory Chart */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-2">
        <div className="text-xs font-medium text-slate-300">
          Predicted RUL Trajectory (GBR vs. LSTM)
        </div>
        <EChartBox option={rulOption} height={250} />
      </div>

      {/* What-If Sensor Perturbation Panel */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Sliders className="h-4 w-4 text-indigo-400" />
            <h3 className="text-sm font-medium text-slate-200">
              What-If Sensor Perturbation (Active 30-Cycle Window)
            </h3>
          </div>
          {hasActiveOffsets && (
            <button
              type="button"
              onClick={() => setOffsets({})}
              className="text-xs text-indigo-400 hover:text-indigo-300"
            >
              Reset sliders
            </button>
          )}
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          {sliderSensors.map((sensor) => {
            const val = offsets[sensor] ?? 0;
            return (
              <div key={sensor} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="font-mono text-slate-300">{sensor}</span>
                  <span className="font-mono text-indigo-300">
                    {val >= 0 ? `+${val.toFixed(1)}%` : `${val.toFixed(1)}%`}
                  </span>
                </div>
                <input
                  type="range"
                  min={-10}
                  max={10}
                  step={0.5}
                  value={val}
                  onChange={(e) =>
                    setOffsets((prev) => ({
                      ...prev,
                      [sensor]: Number(e.target.value),
                    }))
                  }
                  className="h-1.5 w-full cursor-pointer accent-indigo-500"
                />
              </div>
            );
          })}
        </div>

        {hasActiveOffsets && currentPoint && (
          <div className="grid gap-3 border-t border-slate-800 pt-3 sm:grid-cols-2">
            <div className="rounded-lg bg-slate-950/70 p-3 text-xs">
              <div className="text-slate-400">What-If LSTM RUL</div>
              <div className="mt-1 flex items-baseline gap-2 font-mono">
                <span className="text-lg font-semibold text-indigo-300">
                  {whatIfLoading || !whatIfResult ? "..." : whatIfResult.lstm.toFixed(1)}
                </span>
                {whatIfResult && (
                  <span
                    className={
                      whatIfResult.lstm - currentPoint.lstm_rul >= 0
                        ? "text-emerald-400"
                        : "text-rose-400"
                    }
                  >
                    ({whatIfResult.lstm - currentPoint.lstm_rul >= 0 ? "+" : ""}
                    {(whatIfResult.lstm - currentPoint.lstm_rul).toFixed(1)} cycles)
                  </span>
                )}
              </div>
            </div>

            <div className="rounded-lg bg-slate-950/70 p-3 text-xs">
              <div className="text-slate-400">What-If GBR RUL</div>
              <div className="mt-1 flex items-baseline gap-2 font-mono">
                <span className="text-lg font-semibold text-sky-300">
                  {whatIfLoading || !whatIfResult ? "..." : whatIfResult.gbr.toFixed(1)}
                </span>
                {whatIfResult && (
                  <span
                    className={
                      whatIfResult.gbr - currentPoint.gbr_rul >= 0
                        ? "text-emerald-400"
                        : "text-rose-400"
                    }
                  >
                    ({whatIfResult.gbr - currentPoint.gbr_rul >= 0 ? "+" : ""}
                    {(whatIfResult.gbr - currentPoint.gbr_rul).toFixed(1)} cycles)
                  </span>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Explainability Panel */}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-sky-400" />
              <h3 className="text-sm font-medium text-slate-200">
                GBR Top Feature Importances ({detail.dataset_name})
              </h3>
            </div>
            <div className="flex flex-wrap gap-1">
              {featureImp.slice(0, 3).map((f) => {
                const match = f.feature.match(/^(sensor_\d+)/);
                if (!match) return null;
                const s = match[1];
                return (
                  <button
                    key={f.feature}
                    type="button"
                    onClick={() =>
                      setSelectedSensors((prev) =>
                        prev.includes(s) ? prev : [...prev.slice(0, 4), s]
                      )
                    }
                    className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-sky-300 hover:bg-slate-700"
                  >
                    +{s}
                  </button>
                );
              })}
            </div>
          </div>
          {featureImp.length > 0 ? (
            <EChartBox option={gbrFeatureOption} height={260} />
          ) : (
            <div className="py-12 text-center text-xs text-slate-500">
              No feature importance data available.
            </div>
          )}
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-indigo-400" />
              <h3 className="text-sm font-medium text-slate-200">
                LSTM Window Occlusion Sensitivity (|Δ RUL|)
              </h3>
            </div>
            <button
              type="button"
              onClick={handleComputeLSTMExplain}
              disabled={explainLoading}
              className="rounded-lg bg-indigo-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
            >
              {explainLoading ? "Computing..." : lstmExplain ? "Recompute" : "Compute Heatmap"}
            </button>
          </div>
          {lstmExplain ? (
            <EChartBox option={lstmHeatmapOption} height={260} />
          ) : (
            <div className="flex h-[260px] items-center justify-center text-center text-xs text-slate-500">
              Click &ldquo;Compute Heatmap&rdquo; to see which 5-cycle window slices and sensors
              contribute most to the LSTM prediction at cycle {currentPoint?.time_cycles ?? 1}.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
