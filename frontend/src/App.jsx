import { useEffect, useMemo, useState } from 'react';
import MapAdapter from './components/MapAdapter.jsx';
import AlertPanel from './components/AlertPanel.jsx';
import SensorOverlay from './components/SensorOverlay.jsx';
import AudioMonitor from './components/AudioMonitor.jsx';
import useWebSocket from './hooks/useWebSocket.js';
import { fetchRecentEvents, fetchSensors, postSimulateCrash, WS_URL } from './api.js';

// R26.2/R26.3/R26.4 viewport classification.
//   <768           : mobile  (Map full-width, AlertPanel as bottom drawer @ 60vh)
//   768–1023       : tablet  (Map 70% / AlertPanel 30%)
//   >=1024         : desktop (Map 80% / AlertPanel 20%)
function classifyViewport(width) {
  if (width < 768) return 'mobile';
  if (width < 1024) return 'tablet';
  return 'desktop';
}

function useViewportMode() {
  const initial =
    typeof window === 'undefined' ? 'desktop' : classifyViewport(window.innerWidth);
  const [mode, setMode] = useState(initial);
  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const onResize = () => setMode(classifyViewport(window.innerWidth));
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);
  return mode;
}

export default function App() {
  const viewportMode = useViewportMode();
  const [sensors, setSensors] = useState([]);
  const [sensorError, setSensorError] = useState(null);
  const { events, latestEvent: streamLatest, isConnected, hydrate } = useWebSocket(WS_URL);
  const [latestEvent, setLatestEvent] = useState(null);
  const [keepPanelOpenUntil, setKeepPanelOpenUntil] = useState(0);
  const [, forceTick] = useState(0);

  useEffect(() => {
    if (streamLatest) setLatestEvent(streamLatest);
  }, [streamLatest]);

  // Hydrate the alert panel with persisted events on first load.
  useEffect(() => {
    let cancelled = false;
    fetchRecentEvents(50)
      .then(({ events: recent }) => {
        if (cancelled || !Array.isArray(recent) || recent.length === 0) return;
        // hydrate is provided by useWebSocket; falls back to no-op if absent
        if (typeof hydrate === 'function') hydrate(recent);
        const newest = recent[0];  // /events/recent returns newest-first
        if (newest && !latestEvent) setLatestEvent(newest);
      })
      .catch((err) => console.warn('hydrate failed:', err.message));
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrate]);

  // When a crash arrives, keep the alert panel open for at least
  // 14 seconds (≈ 8 s drone flight + 2 s arrival pulse + 4 s linger)
  // so the operator can read the arrival card before the panel slides back.
  useEffect(() => {
    if (!streamLatest) return;
    if (streamLatest.status === 'DETECTED' || streamLatest.status === 'DRONE_DISPATCHED') {
      setKeepPanelOpenUntil(Date.now() + 14_000);
    } else if (streamLatest.status === 'DRONE_ARRIVED') {
      setKeepPanelOpenUntil(Date.now() + 4_000);
    }
  }, [streamLatest]);

  useEffect(() => {
    if (keepPanelOpenUntil <= Date.now()) return;
    const t = setTimeout(() => forceTick((n) => n + 1), keepPanelOpenUntil - Date.now() + 50);
    return () => clearTimeout(t);
  }, [keepPanelOpenUntil]);

  // Sensor fetch with retry-up-to-3 (Requirement 14 criterion 5).
  useEffect(() => {
    let cancelled = false;
    let attempt = 0;
    const tryFetch = async () => {
      try {
        const data = await fetchSensors();
        if (cancelled) return;
        if (Array.isArray(data) && data.length >= 3) {
          setSensors(data);
          setSensorError(null);
        } else {
          throw new Error(`unexpected sensor payload (${data?.length ?? 0})`);
        }
      } catch (err) {
        attempt += 1;
        if (cancelled) return;
        if (attempt < 4) {
          setTimeout(tryFetch, 5000);
        } else {
          setSensorError(err.message || 'Could not load sensors');
        }
      }
    };
    tryFetch();
    return () => { cancelled = true; };
  }, []);

  const sensorTriangle = useMemo(() => {
    if (sensors.length < 3) return null;
    return { p1: sensors[0], p2: sensors[1], p3: sensors[2] };
  }, [sensors]);

  const isCrashActive =
    Boolean(latestEvent) &&
    (latestEvent.status === 'DETECTED' ||
     latestEvent.status === 'DRONE_DISPATCHED' ||
     keepPanelOpenUntil > Date.now());

  const handleReplay = () => {
    if (!latestEvent) return;
    // Force a re-render by emitting a new object reference.
    setLatestEvent({ ...latestEvent, replayedAt: Date.now() });
  };

  const [simulating, setSimulating] = useState(false);
  const [simulateError, setSimulateError] = useState(null);
  const triggerSimulate = async () => {
    if (simulating || !sensorTriangle) return;
    setSimulating(true);
    setSimulateError(null);
    try {
      const { p1, p2, p3 } = sensorTriangle;
      let r1 = Math.random();
      let r2 = Math.random();
      if (r1 + r2 > 1) { r1 = 1 - r1; r2 = 1 - r2; }
      const r3 = 1 - r1 - r2;
      const lat = r1 * p1.lat + r2 * p2.lat + r3 * p3.lat;
      const lon = r1 * p1.lon + r2 * p2.lon + r3 * p3.lon;
      await postSimulateCrash(lat, lon);
    } catch (err) {
      setSimulateError(err.message || 'Simulation failed');
    } finally {
      setSimulating(false);
    }
  };

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-slate-950 text-slate-100">
      {/* Top status bar */}
      <header className="absolute inset-x-0 top-0 z-20 flex items-center justify-between border-b border-slate-800/80 bg-slate-950/85 px-6 py-3 backdrop-blur">
        <div className="flex items-center gap-3">
          <span className="text-lg font-semibold tracking-tight">CrashSense</span>
          <span className="rounded-full border border-slate-700/60 bg-slate-900/80 px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.18em] text-slate-400">
            Acoustic Triangulation
          </span>
        </div>
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-wider">
            <span
              className={`inline-flex h-2 w-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-rose-400'}`}
              aria-hidden="true"
            />
            <span className="text-slate-400">WS {isConnected ? 'live' : 'offline'}</span>
          </div>
          <div
            role="status"
            aria-live="polite"
            className={`flex items-center gap-2 rounded-full border px-4 py-1.5 text-sm font-semibold transition-colors duration-300 ${
              isCrashActive
                ? 'border-rose-500/50 bg-rose-500/15 text-rose-200'
                : 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200'
            }`}
          >
            <span aria-hidden="true">{isCrashActive ? '⚠' : '●'}</span>
            <span>{isCrashActive ? 'CRASH DETECTED' : 'MONITORING'}</span>
          </div>
        </div>
      </header>

      {/* Main content area below the status bar */}
      <main className="absolute inset-x-0 bottom-0 top-[60px]">
        <div
          className="absolute inset-y-0 left-0"
          style={{
            // R26.2: mobile keeps the map full-width even while the bottom
            //        drawer is open (drawer overlays the lower 60vh).
            // R26.3: tablet 70/30 split when a crash is active.
            // R26.4: desktop 80/20 split (base-spec).
            width: (() => {
              if (viewportMode === 'mobile') return '100%';
              if (!isCrashActive) return '100%';
              return viewportMode === 'tablet' ? '70%' : '80%';
            })(),
            transition: 'width 300ms ease-out',
          }}
        >
          <MapAdapter sensors={sensors} crashEvents={events} latestEvent={latestEvent} />
          <SensorOverlay sensors={sensors} />
          <AudioMonitor />

          {/* Always-visible primary action so the operator can trigger the first crash. */}
          <div className="pointer-events-auto absolute left-1/2 top-3 z-20 flex -translate-x-1/2 flex-col items-center gap-2">
            <button
              type="button"
              data-testid="simulate-crash-floating"
              onClick={triggerSimulate}
              disabled={simulating || !sensorTriangle}
              className="cursor-pointer rounded-lg border border-rose-500/50 bg-rose-500/15 px-5 py-2 text-sm font-semibold text-rose-100 shadow-lg shadow-rose-500/10 transition-colors duration-200 hover:bg-rose-500/25 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {simulating ? 'Simulating…' : 'Simulate Crash'}
            </button>
            {simulateError && (
              <span role="alert" className="rounded-md bg-rose-500/15 px-2 py-1 text-[11px] text-rose-200">
                {simulateError}
              </span>
            )}
          </div>

          {sensorError && (
            <div
              role="alert"
              className="pointer-events-auto absolute left-1/2 top-4 -translate-x-1/2 rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-2 text-sm text-rose-200"
            >
              Sensor data unavailable — {sensorError}
            </div>
          )}
        </div>

        <AlertPanel
          open={Boolean(isCrashActive)}
          events={events}
          sensorTriangle={sensorTriangle}
          onReplay={handleReplay}
          hasEvent={Boolean(latestEvent)}
          viewportMode={viewportMode}
        />
      </main>
    </div>
  );
}
