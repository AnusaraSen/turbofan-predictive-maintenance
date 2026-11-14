import { Info } from "lucide-react";

interface InfoTipProps {
  text: string;
}

export function InfoTip({ text }: InfoTipProps) {
  return (
    <span className="group relative inline-flex items-center align-middle">
      <Info
        aria-label={text}
        className="h-3.5 w-3.5 cursor-help text-slate-500 transition group-hover:text-indigo-400"
      />
      <span className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-1.5 hidden w-60 -translate-x-1/2 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-2 text-[11px] font-normal leading-snug text-slate-200 shadow-lg group-hover:block">
        {text}
      </span>
    </span>
  );
}
