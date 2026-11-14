import { useState } from "react";
import { Download, FileUp, Play } from "lucide-react";
import { predictBatch } from "../api/client";
import type { BatchResultItem, DatasetName } from "../types/turbofan";
import { CappedTag, StatusBadge } from "./StatusBadge";

interface BatchUploadViewProps {
  dataset: DatasetName;
}

const CMAPSS_COLUMNS = [
  "unit_number",
  "time_cycles",
  "op_setting_1",
  "op_setting_2",
  "op_setting_3",
  ...Array.from({ length: 21 }, (_, i) => `sensor_${i + 1}`),
];

function parseInputFile(text: string): Record<string, number>[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    const parsed = JSON.parse(trimmed) as unknown;
    const arr = Array.isArray(parsed) ? parsed : [parsed];
    return arr as Record<string, number>[];
  }

  const lines = trimmed.split(/\r?\n/).filter((l) => l.trim().length > 0);
  const firstTokens = lines[0].trim().split(/[,\s]+/);
  const hasHeader = isNaN(Number(firstTokens[0]));

  const headers = hasHeader ? firstTokens : CMAPSS_COLUMNS;
  const dataLines = hasHeader ? lines.slice(1) : lines;

  return dataLines.map((line) => {
    const vals = line.trim().split(/[,\s]+/);
    const row: Record<string, number> = {};
    headers.forEach((h, i) => {
      if (i < vals.length && h !== "RUL") {
        row[h] = Number(vals[i]);
      }
    });
    return row;
  });
}

export function BatchUploadView({ dataset }: BatchUploadViewProps) {
  const [rows, setRows] = useState<Record<string, number>[]>([]);
  const [fileName, setFileName] = useState<string | null>(null);
  const [results, setResults] = useState<BatchResultItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleFile(file: File) {
    setError(null);
    setResults([]);
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = parseInputFile(String(reader.result ?? ""));
        if (parsed.length === 0) {
          setError("No valid rows found in file.");
        }
        setRows(parsed);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to parse file");
        setRows([]);
      }
    };
    reader.readAsText(file);
  }

  function handleRunBatch() {
    if (!rows.length) return;
    setLoading(true);
    setError(null);
    predictBatch(dataset, rows)
      .then((res) => setResults(res))
      .catch((err: unknown) =>
        setError(err instanceof Error ? err.message : String(err))
      )
      .finally(() => setLoading(false));
  }

  function handleExportCsv() {
    if (!results.length) return;
    const header = "unit_number,total_cycles,gbr_rul,lstm_rul,status,capped";
    const lines = results.map(
      (r) =>
        `${r.unit_number},${r.total_cycles},${r.gbr_rul},${r.lstm_rul},${r.status},${r.capped}`
    );
    const blob = new Blob([[header, ...lines].join("\n")], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${dataset}_batch_predictions.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const unitCount = new Set(rows.map((r) => r.unit_number)).size;

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-semibold text-slate-100">
          Batch Inference Upload ({dataset})
        </h2>
        <p className="text-sm text-slate-400">
          Upload a C-MAPSS <code>.txt</code>, <code>.csv</code>, or <code>.json</code> file
          containing one or more engines to run batched GBR + LSTM predictions.
        </p>
      </div>

      <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 bg-slate-900/50 p-8 text-center transition hover:border-indigo-500">
        <FileUp className="h-8 w-8 text-indigo-400" />
        <span className="mt-2 text-sm font-medium text-slate-200">
          {fileName ? `Loaded: ${fileName}` : "Click or drag-and-drop a sensor file"}
        </span>
        <span className="mt-1 text-xs text-slate-500">
          Supports .csv, whitespace-delimited C-MAPSS .txt, or .json array
        </span>
        <input
          type="file"
          accept=".csv,.txt,.json"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
      </label>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-300">
          {error}
        </div>
      )}

      {rows.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="text-sm text-slate-300">
            Parsed <span className="font-mono text-indigo-300">{rows.length}</span> rows across{" "}
            <span className="font-mono text-indigo-300">{unitCount}</span> engine(s).
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleRunBatch}
              disabled={loading}
              className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3.5 py-2 text-xs font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
            >
              <Play className="h-3.5 w-3.5" />
              {loading ? "Running..." : "Run Batch Prediction"}
            </button>
            {results.length > 0 && (
              <button
                type="button"
                onClick={handleExportCsv}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3.5 py-2 text-xs font-medium text-slate-200 hover:text-white"
              >
                <Download className="h-3.5 w-3.5" /> Export CSV
              </button>
            )}
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60">
          <table className="w-full border-collapse text-left text-sm">
            <thead className="bg-slate-900 text-xs uppercase text-slate-400">
              <tr>
                <th className="px-4 py-3">Unit #</th>
                <th className="px-4 py-3">Cycles</th>
                <th className="px-4 py-3">GBR RUL</th>
                <th className="px-4 py-3">LSTM RUL</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {results.map((r) => (
                <tr key={r.unit_number} className="hover:bg-slate-800/50">
                  <td className="px-4 py-2.5 font-medium text-slate-100">
                    Engine {r.unit_number}
                  </td>
                  <td className="px-4 py-2.5 text-slate-300">{r.total_cycles}</td>
                  <td className="px-4 py-2.5 font-mono text-sky-300">
                    {r.gbr_rul.toFixed(1)}
                  </td>
                  <td className="px-4 py-2.5 font-mono font-semibold text-indigo-300">
                    {r.lstm_rul.toFixed(1)}
                  </td>
                  <td className="px-4 py-2.5">
                    <StatusBadge status={r.status} />
                  </td>
                  <td className="px-4 py-2.5">
                    {r.capped ? <CappedTag /> : <span className="text-slate-600">—</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
