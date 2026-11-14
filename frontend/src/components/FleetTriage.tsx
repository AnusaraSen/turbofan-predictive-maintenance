import { useMemo, useState } from "react";
import { ArrowUpDown, Search } from "lucide-react";
import type { EngineStatus, EngineSummary } from "../types/turbofan";
import { CappedTag, StatusBadge } from "./StatusBadge";

type FilterOption = "All" | EngineStatus | "Capped";
type SortKey = "lstm_rul" | "gbr_rul" | "unit_number" | "total_cycles";

interface FleetTriageProps {
  engines: EngineSummary[];
  loading: boolean;
  error: string | null;
  onSelectEngine: (unitNumber: number) => void;
}

export function FleetTriage({
  engines,
  loading,
  error,
  onSelectEngine,
}: FleetTriageProps) {
  const [filter, setFilter] = useState<FilterOption>("All");
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("lstm_rul");
  const [sortAsc, setSortAsc] = useState(true);

  const counts = useMemo(() => {
    let urgent = 0;
    let monitor = 0;
    let healthy = 0;
    let capped = 0;
    for (const e of engines) {
      if (e.status === "Urgent") urgent += 1;
      else if (e.status === "Monitor") monitor += 1;
      else healthy += 1;
      if (e.capped) capped += 1;
    }
    return { total: engines.length, urgent, monitor, healthy, capped };
  }, [engines]);

  const filtered = useMemo(() => {
    const query = search.trim();
    return engines
      .filter((e) => {
        if (filter === "Capped") return e.capped;
        if (filter !== "All" && e.status !== filter) return false;
        if (query && !String(e.unit_number).includes(query)) return false;
        return true;
      })
      .sort((a, b) => {
        const diff = a[sortKey] - b[sortKey];
        return sortAsc ? diff : -diff;
      });
  }, [engines, filter, search, sortKey, sortAsc]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortAsc((prev) => !prev);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  }

  if (loading) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-8 text-center text-slate-400">
        Running batched fleet inference...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-6 text-rose-300">
        Failed to load fleet data: {error}
      </div>
    );
  }

  const kpis: { label: string; value: number; filterKey: FilterOption; accent: string }[] = [
    { label: "Total Engines", value: counts.total, filterKey: "All", accent: "text-slate-100" },
    { label: "Urgent (<25)", value: counts.urgent, filterKey: "Urgent", accent: "text-rose-400" },
    { label: "Monitor (25–75)", value: counts.monitor, filterKey: "Monitor", accent: "text-amber-400" },
    { label: "Healthy (≥75)", value: counts.healthy, filterKey: "Healthy", accent: "text-emerald-400" },
    { label: "At Cap (≥124)", value: counts.capped, filterKey: "Capped", accent: "text-sky-400" },
  ];

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {kpis.map((kpi) => {
          const active = filter === kpi.filterKey;
          return (
            <button
              key={kpi.label}
              type="button"
              onClick={() => setFilter(kpi.filterKey)}
              className={`rounded-xl border p-4 text-left transition ${
                active
                  ? "border-indigo-500 bg-indigo-500/10"
                  : "border-slate-800 bg-slate-900/60 hover:border-slate-700"
              }`}
            >
              <div className="text-xs text-slate-400">{kpi.label}</div>
              <div className={`mt-1 text-2xl font-semibold ${kpi.accent}`}>
                {kpi.value}
              </div>
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2">
          {(["All", "Urgent", "Monitor", "Healthy", "Capped"] as FilterOption[]).map(
            (opt) => (
              <button
                key={opt}
                type="button"
                onClick={() => setFilter(opt)}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                  filter === opt
                    ? "bg-indigo-600 text-white"
                    : "bg-slate-900 text-slate-300 hover:bg-slate-800"
                }`}
              >
                {opt}
              </button>
            )
          )}
        </div>

        <div className="relative w-56">
          <Search className="pointer-events-none absolute top-2.5 left-3 h-4 w-4 text-slate-500" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter by unit #..."
            className="w-full rounded-lg border border-slate-800 bg-slate-900 py-1.5 pr-3 pl-9 text-sm text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
          />
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60">
        <div className="max-h-[540px] overflow-auto">
          <table className="w-full border-collapse text-left text-sm">
            <thead className="sticky top-0 bg-slate-900 text-xs uppercase text-slate-400">
              <tr>
                <th className="px-4 py-3">
                  <button
                    type="button"
                    onClick={() => toggleSort("unit_number")}
                    className="inline-flex items-center gap-1 hover:text-slate-200"
                  >
                    Unit # <ArrowUpDown className="h-3 w-3" />
                  </button>
                </th>
                <th className="px-4 py-3">
                  <button
                    type="button"
                    onClick={() => toggleSort("total_cycles")}
                    className="inline-flex items-center gap-1 hover:text-slate-200"
                  >
                    Cycles <ArrowUpDown className="h-3 w-3" />
                  </button>
                </th>
                <th className="px-4 py-3">
                  <button
                    type="button"
                    onClick={() => toggleSort("gbr_rul")}
                    className="inline-flex items-center gap-1 hover:text-slate-200"
                  >
                    GBR RUL <ArrowUpDown className="h-3 w-3" />
                  </button>
                </th>
                <th className="px-4 py-3">
                  <button
                    type="button"
                    onClick={() => toggleSort("lstm_rul")}
                    className="inline-flex items-center gap-1 hover:text-slate-200"
                  >
                    LSTM RUL <ArrowUpDown className="h-3 w-3" />
                  </button>
                </th>
                <th className="px-4 py-3">True RUL</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {filtered.map((engine) => (
                <tr
                  key={engine.unit_number}
                  onClick={() => onSelectEngine(engine.unit_number)}
                  className="cursor-pointer transition hover:bg-slate-800/60"
                >
                  <td className="px-4 py-2.5 font-medium text-slate-100">
                    Engine {engine.unit_number}
                  </td>
                  <td className="px-4 py-2.5 text-slate-300">
                    {engine.total_cycles}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-sky-300">
                    {engine.gbr_rul.toFixed(1)}
                  </td>
                  <td className="px-4 py-2.5 font-mono font-semibold text-indigo-300">
                    {engine.lstm_rul.toFixed(1)}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-slate-300">
                    {engine.true_rul !== null ? engine.true_rul.toFixed(0) : "—"}
                  </td>
                  <td className="px-4 py-2.5">
                    <StatusBadge status={engine.status} />
                  </td>
                  <td className="px-4 py-2.5">
                    {engine.capped ? <CappedTag /> : <span className="text-slate-600">—</span>}
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-slate-500">
                    No engines match the current filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
