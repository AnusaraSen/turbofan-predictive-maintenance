import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BookOpen,
  Compass,
  Cpu,
  Gauge,
  Layers,
  Search,
  Sliders,
} from "lucide-react";

interface SensorInfo {
  id: string;
  code: string;
  name: string;
  plainMeaning: string;
  role: "Usually Informative" | "Often Constant (FD001/FD003)";
}

const SENSOR_GUIDE: SensorInfo[] = [
  { id: "sensor_1", code: "T2", name: "Total temperature at fan inlet", plainMeaning: "Air temperature entering the front of the engine (depends mostly on outside weather/altitude).", role: "Often Constant (FD001/FD003)" },
  { id: "sensor_2", code: "T24", name: "Total temperature at LPC outlet", plainMeaning: "Air temperature leaving the low-pressure compressor; rises as compressor efficiency drops.", role: "Usually Informative" },
  { id: "sensor_3", code: "T30", name: "Total temperature at HPC outlet", plainMeaning: "High-pressure compressor exit temperature; a key indicator of core wear and heat buildup.", role: "Usually Informative" },
  { id: "sensor_4", code: "T50", name: "Total temperature at LPT outlet", plainMeaning: "Exhaust gas temperature after the turbine; climbs when the engine works harder to maintain thrust.", role: "Usually Informative" },
  { id: "sensor_5", code: "P2", name: "Pressure at fan inlet", plainMeaning: "Incoming air pressure at the engine intake (driven by flight altitude and speed).", role: "Often Constant (FD001/FD003)" },
  { id: "sensor_6", code: "P15", name: "Total pressure in bypass-duct", plainMeaning: "Air pressure in the outer bypass stream around the engine core.", role: "Often Constant (FD001/FD003)" },
  { id: "sensor_7", code: "P30", name: "Total pressure at HPC outlet", plainMeaning: "High-pressure air feeding the combustor; drops when compressor blades wear down.", role: "Usually Informative" },
  { id: "sensor_8", code: "Nf", name: "Physical fan speed", plainMeaning: "How fast the large front fan is spinning (RPM).", role: "Usually Informative" },
  { id: "sensor_9", code: "Nc", name: "Physical core speed", plainMeaning: "How fast the inner high-pressure shaft is spinning (RPM); shifts as the core degrades.", role: "Usually Informative" },
  { id: "sensor_10", code: "epr", name: "Engine pressure ratio (P50/P2)", plainMeaning: "Ratio of exhaust pressure to intake pressure (overall thrust indicator).", role: "Often Constant (FD001/FD003)" },
  { id: "sensor_11", code: "Ps30", name: "Static pressure at HPC outlet", plainMeaning: "Steady pressure at the high-pressure compressor exit; one of the strongest wear signals.", role: "Usually Informative" },
  { id: "sensor_12", code: "phi", name: "Ratio of fuel flow to Ps30", plainMeaning: "How much fuel is burned relative to compressor pressure; drops as efficiency worsens.", role: "Usually Informative" },
  { id: "sensor_13", code: "NRf", name: "Corrected fan speed", plainMeaning: "Fan speed adjusted for outside air temperature and pressure.", role: "Usually Informative" },
  { id: "sensor_14", code: "NRc", name: "Corrected core speed", plainMeaning: "Core shaft speed adjusted for outside flight conditions.", role: "Usually Informative" },
  { id: "sensor_15", code: "BPR", name: "Bypass ratio", plainMeaning: "Ratio of air flowing around the core vs. through the core; rises as core efficiency falls.", role: "Usually Informative" },
  { id: "sensor_16", code: "farB", name: "Burner fuel-air ratio", plainMeaning: "Fuel-to-air mix inside the combustion chamber.", role: "Often Constant (FD001/FD003)" },
  { id: "sensor_17", code: "htBleed", name: "Bleed enthalpy", plainMeaning: "Heat energy of air bled off the compressor for aircraft systems (cabin/anti-ice).", role: "Usually Informative" },
  { id: "sensor_18", code: "Nf_dmd", name: "Demanded fan speed", plainMeaning: "Target fan speed requested by the engine controller.", role: "Often Constant (FD001/FD003)" },
  { id: "sensor_19", code: "PCNfR_dmd", name: "Demanded corrected fan speed", plainMeaning: "Target corrected fan speed requested by the engine controller.", role: "Often Constant (FD001/FD003)" },
  { id: "sensor_20", code: "W31", name: "HPT coolant bleed", plainMeaning: "Cooling airflow sent to the high-pressure turbine blades.", role: "Usually Informative" },
  { id: "sensor_21", code: "W32", name: "LPT coolant bleed", plainMeaning: "Cooling airflow sent to the low-pressure turbine blades.", role: "Usually Informative" },
];

export function GuideView() {
  const [sensorSearch, setSensorSearch] = useState("");

  const filteredSensors = useMemo(() => {
    const q = sensorSearch.trim().toLowerCase();
    if (!q) return SENSOR_GUIDE;
    return SENSOR_GUIDE.filter(
      (s) =>
        s.id.toLowerCase().includes(q) ||
        s.code.toLowerCase().includes(q) ||
        s.name.toLowerCase().includes(q) ||
        s.plainMeaning.toLowerCase().includes(q)
    );
  }, [sensorSearch]);

  return (
    <div className="space-y-8 pb-10">
      {/* Intro Banner */}
      <div className="rounded-2xl border border-indigo-500/30 bg-gradient-to-br from-indigo-950/60 via-slate-900 to-slate-900 p-6">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 rounded-lg bg-indigo-500/20 p-2 text-indigo-300">
            <BookOpen className="h-5 w-5" />
          </div>
          <div className="space-y-2">
            <h2 className="text-xl font-semibold text-slate-100">
              Plain-English Guide to Turbofan Predictive Maintenance
            </h2>
            <p className="max-w-3xl text-sm leading-relaxed text-slate-300">
              This tool acts like an early-warning health monitor for aircraft jet engines.
              Using sensor readings collected on every flight cycle, it predicts{" "}
              <strong className="text-white">Remaining Useful Life (RUL)</strong> — the
              number of flights left before an engine needs maintenance — so airlines can
              schedule repairs <em>before</em> a breakdown without replacing healthy parts
              too early.
            </p>
            <div className="flex flex-wrap gap-2 pt-2 text-xs">
              {[
                ["#about", "1. Datasets"],
                ["#mechanism", "2. How Predictions Work"],
                ["#kpis", "3. KPI Cards & Badges"],
                ["#visuals", "4. Reading the Charts"],
                ["#sensors", "5. Sensor Dictionary (1–21)"],
              ].map(([href, label]) => (
                <a
                  key={href}
                  href={href}
                  className="rounded-lg border border-slate-700 bg-slate-900/80 px-3 py-1 text-slate-300 transition hover:border-indigo-500 hover:text-white"
                >
                  {label}
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Section 1: Datasets */}
      <section id="about" className="space-y-3">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-indigo-400" />
          <h3 className="text-base font-semibold text-slate-100">
            1. What the 4 NASA Datasets (FD001–FD004) Represent
          </h3>
        </div>
        <p className="text-sm text-slate-400">
          The data comes from NASA&apos;s C-MAPSS jet engine simulator. Each dataset tests a
          different level of real-world complexity:
        </p>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              ds: "FD001",
              cond: "1 Flight Condition (Sea Level)",
              fault: "1 Fault Mode (High-Pressure Compressor)",
              note: "Simplest baseline — steady conditions, one way to wear out.",
            },
            {
              ds: "FD002",
              cond: "6 Flight Conditions (Altitudes & Speeds)",
              fault: "1 Fault Mode (High-Pressure Compressor)",
              note: "Harder — normal flight changes shift sensor readings dramatically.",
            },
            {
              ds: "FD003",
              cond: "1 Flight Condition (Sea Level)",
              fault: "2 Fault Modes (Compressor + Fan)",
              note: "Two different parts can fail, creating different wear signatures.",
            },
            {
              ds: "FD004",
              cond: "6 Flight Conditions (Altitudes & Speeds)",
              fault: "2 Fault Modes (Compressor + Fan)",
              note: "Most realistic & hardest — multiple flight regimes and multiple failure types.",
            },
          ].map((item) => (
            <div
              key={item.ds}
              className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-1.5"
            >
              <div className="font-mono text-sm font-semibold text-indigo-300">
                {item.ds}
              </div>
              <div className="text-xs font-medium text-slate-200">{item.cond}</div>
              <div className="text-xs text-sky-300">{item.fault}</div>
              <p className="pt-1 text-xs text-slate-400">{item.note}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Section 2: Mechanism */}
      <section id="mechanism" className="space-y-3">
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-indigo-400" />
          <h3 className="text-base font-semibold text-slate-100">
            2. How Predictions Are Made (Step-by-Step Mechanism)
          </h3>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          {[
            {
              step: "Step 1 · Group by Flight Condition (Regimes)",
              text: "An engine cruising at 35,000 ft naturally has different temperatures and pressures than one taking off at sea level. A clustering algorithm (K-means) groups each flight into one of 6 Operating Regimes (R0–R5) so normal flight changes aren't mistaken for damage.",
            },
            {
              step: "Step 2 · Drop Flat Sensors & Normalize",
              text: "Sensors that never change within a flight regime carry no wear signal and are ignored. The remaining informative sensors are normalized (rescaled) within each regime so the AI sees pure degradation.",
            },
            {
              step: "Step 3 · The 125-Cycle 'Healthy Ceiling'",
              text: "Early in an engine's life, no physical wear is visible yet. During training, any remaining life above 125 cycles is capped at 125 — teaching the models that 'healthy is healthy' and focusing accuracy on the final 125 cycles before failure.",
            },
            {
              step: "Step 4 · Two AI Models Compared",
              text: "GBR (Gradient Boosting) looks at summary snapshots like recent 10-cycle averages and rate of change. LSTM (Deep Learning) watches the last 30 cycles in sequence — like a short video clip — to spot subtle degradation curves over time.",
            },
          ].map((card) => (
            <div
              key={card.step}
              className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-1.5"
            >
              <div className="text-sm font-semibold text-slate-100">{card.step}</div>
              <p className="text-xs leading-relaxed text-slate-300">{card.text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Section 3: KPI Cards & Badges */}
      <section id="kpis" className="space-y-3">
        <div className="flex items-center gap-2">
          <Gauge className="h-4 w-4 text-indigo-400" />
          <h3 className="text-base font-semibold text-slate-100">
            3. What Each KPI Card &amp; Badge Means
          </h3>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {[
            {
              title: "Urgent (< 25 cycles)",
              badge: "bg-rose-500/15 text-rose-300 border-rose-500/40",
              desc: "High risk — the model estimates fewer than 25 flights remain. Schedule maintenance immediately.",
            },
            {
              title: "Monitor (25–75 cycles)",
              badge: "bg-amber-500/15 text-amber-300 border-amber-500/40",
              desc: "Early signs of wear are visible. Keep the engine on watch and plan a maintenance slot.",
            },
            {
              title: "Healthy (≥ 75 cycles)",
              badge: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
              desc: "No immediate maintenance needed; the engine is operating normally.",
            },
            {
              title: "Cap 125",
              badge: "bg-sky-500/10 text-sky-300 border-sky-500/40",
              desc: "Prediction is at the 125-cycle training ceiling. The engine is healthy, but exact life beyond 125 cycles cannot be estimated from sensors.",
            },
            {
              title: "Operating Regime (R0–R5)",
              badge: "bg-amber-500/10 text-amber-300 border-amber-500/30",
              desc: "The flight condition cluster (altitude / speed / throttle setting) detected for that cycle.",
            },
            {
              title: "True RUL",
              badge: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
              desc: "The actual ground-truth remaining cycles from NASA's test benchmark, shown so you can judge how close GBR and LSTM are.",
            },
          ].map((k) => (
            <div
              key={k.title}
              className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-2"
            >
              <span
                className={`inline-block rounded-full border px-2.5 py-0.5 text-xs font-medium ${k.badge}`}
              >
                {k.title}
              </span>
              <p className="text-xs leading-relaxed text-slate-300">{k.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Section 4: Visuals */}
      <section id="visuals" className="space-y-3">
        <div className="flex items-center gap-2">
          <Compass className="h-4 w-4 text-indigo-400" />
          <h3 className="text-base font-semibold text-slate-100">
            4. What Each Visual &amp; Interactive Tool Tells You
          </h3>
        </div>
        <div className="space-y-3">
          {[
            {
              icon: Layers,
              title: "Fleet Triage Board",
              body: "Lists every engine in the selected dataset sorted by urgency (lowest predicted RUL at the top). Click any KPI card or filter pill to focus on Urgent or Capped engines, and click any row to inspect that engine.",
            },
            {
              icon: Activity,
              title: "Sensor Traces + Regime Bands + 30-Cycle Highlight",
              body: "Shows how key sensors behave over the engine's life. The shaded purple box marks the 30-cycle window the LSTM is currently reading. On FD002/FD004, colored background bands show flight regime switches — toggle between Raw and Regime-Normalized to see how normalization removes flight-condition jumps and reveals the true wear trend.",
            },
            {
              icon: Gauge,
              title: "Predicted RUL Trajectory Chart (Play / Scrub)",
              body: "Press Play or drag the slider to watch how GBR and LSTM predictions evolve from flight 1 to the final cycle. The dashed blue line at 125 is the training cap, and the dashed red line at 25 is the Urgent maintenance threshold.",
            },
            {
              icon: Sliders,
              title: "What-If Sensor Perturbation Sliders",
              body: "Nudge key sensors up or down by up to ±10% on the active 30-cycle window to simulate sensor drift or a sudden anomaly, and immediately see how many cycles GBR and LSTM shift.",
            },
            {
              icon: AlertTriangle,
              title: "Explainability, Benchmarks & PHM08 Score",
              body: "GBR Feature Importance shows which sensor features mattered most; the LSTM Heatmap highlights which 5-cycle time slices and sensors had the biggest impact on the prediction. On the Benchmarks tab, RMSE measures average cycle error, while PHM08 is an asymmetric safety score that penalizes overestimating remaining life ('you have more time than you do') much more heavily than an early conservative warning.",
            },
          ].map((v) => {
            const Icon = v.icon;
            return (
              <div
                key={v.title}
                className="flex gap-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4"
              >
                <div className="mt-0.5 text-indigo-400">
                  <Icon className="h-4 w-4" />
                </div>
                <div className="space-y-1">
                  <div className="text-sm font-medium text-slate-100">{v.title}</div>
                  <p className="text-xs leading-relaxed text-slate-300">{v.body}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Section 5: Sensor Dictionary */}
      <section id="sensors" className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-slate-100">
              5. Plain-English Sensor Reference (sensor_1 – sensor_21)
            </h3>
            <p className="text-xs text-slate-400">
              Physical meaning of each NASA C-MAPSS turbofan sensor and whether it typically
              carries degradation signals.
            </p>
          </div>
          <div className="relative w-64">
            <Search className="pointer-events-none absolute top-2.5 left-3 h-4 w-4 text-slate-500" />
            <input
              type="text"
              value={sensorSearch}
              onChange={(e) => setSensorSearch(e.target.value)}
              placeholder="Search sensors (e.g. pressure, T30)..."
              className="w-full rounded-lg border border-slate-800 bg-slate-900 py-1.5 pr-3 pl-9 text-xs text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
            />
          </div>
        </div>

        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="max-h-[460px] overflow-auto">
            <table className="w-full border-collapse text-left text-xs">
              <thead className="sticky top-0 bg-slate-900 uppercase text-slate-400">
                <tr>
                  <th className="px-4 py-3">Sensor</th>
                  <th className="px-4 py-3">Code</th>
                  <th className="px-4 py-3">What It Measures</th>
                  <th className="px-4 py-3">Plain-English Meaning</th>
                  <th className="px-4 py-3">Typical Role</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80">
                {filteredSensors.map((s) => (
                  <tr key={s.id} className="hover:bg-slate-800/40">
                    <td className="px-4 py-2.5 font-mono font-medium text-indigo-300">
                      {s.id}
                    </td>
                    <td className="px-4 py-2.5 font-mono text-slate-300">{s.code}</td>
                    <td className="px-4 py-2.5 font-medium text-slate-200">{s.name}</td>
                    <td className="px-4 py-2.5 text-slate-300">{s.plainMeaning}</td>
                    <td className="px-4 py-2.5">
                      <span
                        className={`inline-block rounded px-2 py-0.5 text-[11px] font-medium ${
                          s.role === "Usually Informative"
                            ? "bg-emerald-500/15 text-emerald-300"
                            : "bg-slate-800 text-slate-400"
                        }`}
                      >
                        {s.role}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
