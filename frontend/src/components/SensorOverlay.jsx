// Minimal sensor legend rendered over the map (Requirement 14).

export default function SensorOverlay({ sensors }) {
  if (!sensors?.length) return null;
  return (
    <div className="pointer-events-none absolute left-4 top-4 z-20 max-w-xs rounded-xl border border-slate-700/60 bg-slate-900/80 p-4 text-xs shadow-lg backdrop-blur">
      <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-slate-400">Sensor network</p>
      <ul className="mt-2 space-y-1.5">
        {sensors.map((sensor) => (
          <li key={sensor.id} className="flex items-center gap-2">
            <span
              className={`inline-flex h-2.5 w-2.5 flex-none rounded-full ${
                sensor.is_toll_plaza ? 'bg-cyan-400' : 'bg-sky-400'
              }`}
              aria-hidden="true"
            />
            <span className="font-mono text-slate-200">{sensor.id}</span>
            <span className="text-slate-400">·</span>
            <span className="text-slate-300">{sensor.name}</span>
            {sensor.is_toll_plaza && (
              <span className="ml-1 rounded-full border border-cyan-500/40 bg-cyan-500/10 px-1.5 py-0.5 text-[10px] uppercase text-cyan-200">
                Toll
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
