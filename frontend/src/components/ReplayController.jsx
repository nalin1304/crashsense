import { useEffect, useRef, useState } from 'react';
import { fetchEventById } from '../api.js';

// Replay_Controller — R23.1 / R23.2 / R23.3 / R23.5.
//
// Owns three concerns:
//
//   1. Fetching the persisted CrashEvent from `GET /events/{event_id}`.
//      The backend is local FastAPI and the route reads from the
//      Event_Store WAL, so a 1 s budget (R23.2) is comfortably met under
//      normal conditions; we still wrap the fetch with an AbortController
//      timeout so a hung backend surfaces as an error rather than a
//      silent loading state.
//
//   2. Rendering the non-modal "REPLAY (<original_timestamp>)" banner
//      (R23.3). The banner is fixed near the top of the viewport with
//      pointer-events constrained to its own controls, so the live map,
//      Alert_Panel, and any concurrent live CrashEvent continue to
//      render and update underneath without being intercepted (R23.5).
//
//   3. Lifecycle — once the event is loaded the component reports it to
//      the parent (`onEventLoaded`) which threads it through `MapAdapter`
//      so a second `<DroneTracker replayMode>` instance re-runs the full
//      animation against the same map. After the animation linger window
//      ends, `onClose` is invoked so the parent can clear the replay
//      branch. The user can also dismiss the replay early via the banner
//      close button.
//
// The component does not itself broadcast `DRONE_ARRIVED`; suppression
// happens inside `<DroneTracker replayMode>` (R23.3).

// Animation timing: matches `<DroneTracker>` (Leaflet) and `<MapboxMap>`.
// 8 s drone flight + 2.2 s arrival pulse + ~3.8 s linger so the operator
// has time to read the arrival before the replay clears itself.
const REPLAY_ANIMATION_DURATION_MS = 8000;
const REPLAY_ARRIVAL_LINGER_MS = 2200;
const REPLAY_TAIL_LINGER_MS = 3800;
const REPLAY_TOTAL_MS =
  REPLAY_ANIMATION_DURATION_MS + REPLAY_ARRIVAL_LINGER_MS + REPLAY_TAIL_LINGER_MS;

// Wall-clock fetch budget. The acceptance criterion is 1 s, but a single
// missed deadline should not strand the operator on a perpetual loading
// banner; we abort at 5 s and surface an error so the user can retry.
const FETCH_TIMEOUT_MS = 5000;

function formatTimestamp(value) {
  if (!value) return '';
  try {
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    const pad = (n) => String(n).padStart(2, '0');
    return (
      `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
      `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
    );
  } catch {
    return String(value);
  }
}

export default function ReplayController({ eventId, onEventLoaded, onClose }) {
  const [phase, setPhase] = useState('loading'); // loading | replaying | error
  const [event, setEvent] = useState(null);
  const [error, setError] = useState(null);

  // Stable refs for parent callbacks so re-renders do not re-fire the fetch.
  const onEventLoadedRef = useRef(onEventLoaded);
  const onCloseRef = useRef(onClose);
  useEffect(() => { onEventLoadedRef.current = onEventLoaded; }, [onEventLoaded]);
  useEffect(() => { onCloseRef.current = onClose; }, [onClose]);

  // Fetch + lifecycle wiring: re-runs only when the eventId itself changes.
  useEffect(() => {
    if (!eventId) return undefined;
    let cancelled = false;
    setPhase('loading');
    setEvent(null);
    setError(null);

    (async () => {
      try {
        const evt = await fetchEventById(eventId, { timeoutMs: FETCH_TIMEOUT_MS });
        if (cancelled) return;
        setEvent(evt);
        setPhase('replaying');
        if (typeof onEventLoadedRef.current === 'function') {
          onEventLoadedRef.current(evt);
        }
      } catch (err) {
        if (cancelled) return;
        const isMissing = err?.status === 404;
        setError(
          isMissing
            ? `Event ${eventId} not found.`
            : `Replay fetch failed: ${err?.message || 'unknown error'}`,
        );
        setPhase('error');
      }
    })();

    return () => { cancelled = true; };
  }, [eventId]);

  // Auto-close after the replay animation completes (R23.3 — suppress
  // DRONE_ARRIVED is handled inside `<DroneTracker replayMode>`; here we
  // just clean up the parent state so the next replay starts fresh).
  useEffect(() => {
    if (phase !== 'replaying') return undefined;
    const t = setTimeout(() => {
      if (typeof onCloseRef.current === 'function') onCloseRef.current();
    }, REPLAY_TOTAL_MS);
    return () => clearTimeout(t);
  }, [phase]);

  if (!eventId) return null;

  // Banner positioning: fixed near the top of the map area. Container is
  // `pointer-events-none` so map drag/zoom and live-event clicks pass
  // through; the inner panel re-enables pointer events for its own
  // controls (R23.5).
  const containerCls =
    'pointer-events-none fixed inset-x-0 top-[72px] z-40 flex justify-center px-4';

  if (phase === 'loading') {
    return (
      <div className={containerCls} aria-live="polite">
        <div className="pointer-events-auto flex items-center gap-3 rounded-full border border-cyan-500/40 bg-slate-900/90 px-4 py-2 text-sm text-cyan-100 shadow-lg backdrop-blur">
          <span
            className="inline-block h-2 w-2 animate-pulse rounded-full bg-cyan-300"
            aria-hidden="true"
          />
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-cyan-300">
            Loading replay…
          </span>
          <button
            type="button"
            onClick={() => onCloseRef.current?.()}
            className="cursor-pointer rounded-md border border-slate-700/70 bg-slate-900/70 px-2 py-0.5 text-[11px] font-medium text-slate-300 transition-colors duration-200 hover:bg-slate-800/80"
            aria-label="Cancel replay"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  if (phase === 'error') {
    // R23.4 surface: non-blocking error indicator. Live events continue
    // to flow because the rest of the dashboard is unaffected.
    return (
      <div className={containerCls}>
        <div
          role="alert"
          className="pointer-events-auto flex items-center gap-3 rounded-full border border-rose-500/50 bg-rose-500/15 px-4 py-2 text-sm text-rose-100 shadow-lg backdrop-blur"
        >
          <span aria-hidden="true">⚠</span>
          <span className="font-medium">{error}</span>
          <button
            type="button"
            onClick={() => onCloseRef.current?.()}
            className="cursor-pointer rounded-md border border-rose-500/50 bg-rose-500/20 px-2 py-0.5 text-[11px] font-semibold text-rose-100 transition-colors duration-200 hover:bg-rose-500/30"
            aria-label="Dismiss replay error"
          >
            Dismiss
          </button>
        </div>
      </div>
    );
  }

  // Replaying: non-modal banner with the original timestamp.
  return (
    <div className={containerCls} aria-live="polite">
      <div className="pointer-events-auto flex items-center gap-3 rounded-full border border-amber-400/50 bg-amber-500/15 px-4 py-2 text-sm text-amber-100 shadow-lg backdrop-blur">
        <span
          className="inline-flex h-2 w-2 rounded-full bg-amber-300"
          aria-hidden="true"
        />
        <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.2em] text-amber-200">
          REPLAY ({formatTimestamp(event?.timestamp)})
        </span>
        <button
          type="button"
          onClick={() => onCloseRef.current?.()}
          className="cursor-pointer rounded-md border border-amber-400/50 bg-amber-500/20 px-2 py-0.5 text-[11px] font-semibold text-amber-100 transition-colors duration-200 hover:bg-amber-500/30"
          aria-label="Close replay"
        >
          Close
        </button>
      </div>
    </div>
  );
}
