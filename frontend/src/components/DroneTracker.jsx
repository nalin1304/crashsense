import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { interpolatePosition } from '../utils/geo.js';
import { postDroneArrived, postDroneDispatched } from '../api.js';

const ANIMATION_DURATION_MS = 8000;

const DRONE_SVG = `
<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" class="drone-marker">
  <g>
    <circle cx="6" cy="6" r="3" fill="#22d3ee" />
    <circle cx="26" cy="6" r="3" fill="#22d3ee" />
    <circle cx="6" cy="26" r="3" fill="#22d3ee" />
    <circle cx="26" cy="26" r="3" fill="#22d3ee" />
    <line x1="6" y1="6" x2="26" y2="26" stroke="#0f172a" stroke-width="1.6"/>
    <line x1="26" y1="6" x2="6" y2="26" stroke="#0f172a" stroke-width="1.6"/>
    <rect x="12" y="12" width="8" height="8" rx="2" fill="#0ea5e9" stroke="#0f172a" stroke-width="1.4"/>
  </g>
</svg>`;

function buildDroneIcon() {
  return L.divIcon({
    html: DRONE_SVG,
    className: 'drone-icon-wrapper',
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
}

function buildArrivalIcon() {
  return L.divIcon({
    html: '<div class="arrival-pulse"></div>',
    className: 'arrival-pulse-wrapper',
    iconSize: [36, 36],
    iconAnchor: [18, 18],
  });
}

export default function DroneTracker({ map, latestEvent, replayMode = false }) {
  const droneMarkerRef = useRef(null);
  const arrivalMarkerRef = useRef(null);
  const animFrameRef = useRef(null);
  const dispatchedForRef = useRef(null);
  const [eta, setEta] = useState(null);

  useEffect(() => {
    const lmap = map?.current;
    if (!lmap || !latestEvent) return;
    // In replay mode the persisted event will typically have its terminal
    // status (DRONE_ARRIVED), so the live-mode `DETECTED` gate is bypassed
    // and the animation is re-run unconditionally for the supplied event.
    if (!replayMode && latestEvent.status !== 'DETECTED') return;

    const { event_id, drone_origin_lat, drone_origin_lon, crash_lat, crash_lon } = latestEvent;
    if (
      typeof drone_origin_lat !== 'number' ||
      typeof drone_origin_lon !== 'number' ||
      typeof crash_lat !== 'number' ||
      typeof crash_lon !== 'number'
    ) {
      return;
    }
    // Avoid replaying the same event twice (StrictMode double-mount, etc.)
    if (dispatchedForRef.current === event_id) return;
    dispatchedForRef.current = event_id;

    // Reset any previous run.
    if (droneMarkerRef.current) lmap.removeLayer(droneMarkerRef.current);
    if (arrivalMarkerRef.current) lmap.removeLayer(arrivalMarkerRef.current);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);

    droneMarkerRef.current = L.marker([drone_origin_lat, drone_origin_lon], {
      icon: buildDroneIcon(),
      interactive: false,
    }).addTo(lmap);

    const start = { lat: drone_origin_lat, lon: drone_origin_lon };
    const end = { lat: crash_lat, lon: crash_lon };
    const startedAt = performance.now();

    // Tell the backend dispatch has begun so all dashboards transition the
    // alert badge from DETECTING (yellow) to DRONE DISPATCHED (orange).
    // Replay mode is read-only: it must not rewrite the persisted timeline
    // or fan a stale status out to other clients (R23.3 — extending the
    // suppression beyond DRONE_ARRIVED to keep replays side-effect free).
    if (!replayMode) {
      postDroneDispatched(latestEvent).catch((err) =>
        console.warn('drone-dispatched POST failed', err),
      );
    }

    const tick = (now) => {
      const elapsed = now - startedAt;
      const progress = Math.min(1, elapsed / ANIMATION_DURATION_MS);
      const { lat, lon } = interpolatePosition(start, end, progress);
      droneMarkerRef.current?.setLatLng([lat, lon]);
      const remaining = Math.max(0, Math.ceil((ANIMATION_DURATION_MS - elapsed) / 1000));
      setEta(remaining);
      if (progress < 1) {
        animFrameRef.current = requestAnimationFrame(tick);
      } else {
        setEta(0);
        // Green pulse at arrival
        arrivalMarkerRef.current = L.marker([crash_lat, crash_lon], {
          icon: buildArrivalIcon(),
          interactive: false,
        }).addTo(lmap);
        // Notify backend so dashboards stay in sync.
        // R23.3: replay mode SHALL NOT broadcast a DRONE_ARRIVED update.
        if (!replayMode) {
          postDroneArrived(latestEvent).catch((err) =>
            console.warn('drone-arrived POST failed', err),
          );
        }
        // Fade out after 2 seconds
        setTimeout(() => {
          if (arrivalMarkerRef.current) {
            lmap.removeLayer(arrivalMarkerRef.current);
            arrivalMarkerRef.current = null;
          }
          setEta(null);
        }, 2200);
      }
    };

    animFrameRef.current = requestAnimationFrame(tick);

    // Note: we deliberately do NOT cancel the animation on cleanup, because
    // when DRONE_DISPATCHED is broadcast back to us via WebSocket, latestEvent
    // changes and this effect re-runs. The cleanup from the previous run
    // would otherwise cancel the in-flight animation. The dispatchedForRef
    // guard above prevents duplicate flights.
  }, [map, latestEvent, replayMode]);

  // Cancel the animation only when the component unmounts entirely.
  useEffect(() => {
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  if (eta == null) return null;

  return (
    <div className="pointer-events-none absolute right-4 top-4 rounded-xl border border-cyan-500/30 bg-slate-900/85 px-4 py-2 text-sm shadow-lg backdrop-blur">
      <div className="font-mono text-xs uppercase tracking-wider text-cyan-300">Drone in transit</div>
      <div className="mt-0.5 font-mono text-2xl font-semibold text-cyan-100">
        ETA {eta}s
      </div>
    </div>
  );
}
