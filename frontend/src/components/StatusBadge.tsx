import type { EngineStatus } from "../types/turbofan";
import { InfoTip } from "./InfoTip";

const STATUS_STYLES: Record<EngineStatus, string> = {
  Urgent: "bg-rose-500/15 text-rose-300 border-rose-500/40",
  Monitor: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  Healthy: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
};

export function StatusBadge({ status }: { status: EngineStatus }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${STATUS_STYLES[status]}`}
    >
      {status}
    </span>
  );
}

export function CappedTag() {
  return (
    <span className="inline-flex items-center gap-1 rounded border border-sky-500/40 bg-sky-500/10 px-1.5 py-0.5 text-[11px] font-medium text-sky-300">
      Cap 125
      <InfoTip text="Model predictions are capped at 125 cycles because early healthy engines show no detectable wear yet." />
    </span>
  );
}
