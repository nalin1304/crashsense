import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { MAPBOX_API_KEY } from '../utils/mapProvider.js';
import { interpolatePosition } from '../utils/geo.js';
import { postDroneArrived, postDroneDispatched } from '../api.js';

// Mapbox-backed Map_View. Internal component for `<MapAdapter>`; selected
// at build time when `VITE_MAP_PROVIDER === "mapbox"` AND a non-empty
// `VITE_MAPBOX_API_KEY` is present (R22.3). Loaded lazily by `<MapAdapter>`
// so the default Leaflet build never imports `mapbox-gl`.

mapboxgl.accessToken = MAPBOX_API_KEY;

const ANIMATION_DURATION_MS = 8000;
const MAX_PINS = 50;

const SENSOR_DOCK_CSS = 'mb-sensor-marker mb-sensor-marker--toll';
const SENSOR_BASE_CSS = 'mb-sensor-marker';

function buildSensorElement(sensor) {
  const el = document.createElement('div');
  el.className = sensor.is_toll_plaza ? SENSOR_DOCK_CSS : SENSOR_BASE_CSS;
  el.innerHTML = sensor.is_toll_plaza
    ? `<div class="sensor-marker toll"><span class="label">${sensor.id}</span>
         <svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
           <defs>
             <linearGradient id="dockG-mb-${sensor.id}" x1="0" y1="0" x2="0" y2="1">
               <stop offset="0%" stop-color="#22d3ee"/>
               <stop offset="100%" stop-color="#0ea5e9"/>
             </linearGradient>
           </defs>
           <circle cx="16" cy="16" r="13" fill="url(#dockG-mb-${sensor.id})" stroke="#082f49" stroke-width="2"/>
           <path d="M9 12h14M16 9v14M11 16h10" stroke="#0f172a" stroke-width="1.6" stroke-linecap="round"/>
           <circle cx="16" cy="16" r="2.4" fill="#0f172a"/>
         </svg>
       </div>`
    : `<div class="sensor-marker"><span class="label">${sensor.id}</span><div class="pulse"></div></div>`;
  el.title = `${sensor.id} · ${sensor.name}`;
  return el;
}

function buildCrashPinElement(crashLat, crashLon) {
  const el = document.createElement('div');
  el.className = 'mb-crash-pin';
  el.innerHTML = `
    <div class="crash-pin">
      <svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="pinG-mb" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#fb7185"/>
            <stop offset="100%" stop-color="#dc2626"/>
          </linearGradient>
        </defs>
        <path d="M16 2c-6 0-10 4.4-10 10 0 7.5 10 18 10 18s10-10.5 10-18c0-5.6-4-10-10-10z" fill="url(#pinG-mb)" stroke="#7f1d1d" stroke-width="1.4"/>
        <circle cx="16" cy="12" r="3.6" fill="#fff5f5" stroke="#7f1d1d" stroke-width="1"/>
      </svg>
    </div>
    <div class="crash-coord-label">CRASH ${crashLat.toFixed(6)}, ${crashLon.toFixed(6)}</div>
  `;
  return el;
}

function buildWavefrontElement() {
  const el = document.createElement('div');
  el.className = 'wavefront-wrapper';
  el.innerHTML = '<div class="wavefront-ring"></div>';
  return el;
}

function buildDroneElement() {
  const el = document.createElement('div');
  el.className = 'drone-icon-wrapper';
  el.innerHTML = `
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
  return el;
}

function buildArrivalElement() {
  const el = document.createElement('div');
  el.className = 'arrival-pulse-wrapper';
  el.innerHTML = '<div class="arrival-pulse"></div>';
  return el;
}

export default function MapboxMap({ sensors, crashEvents, latestEvent }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const sensorMarkersRef = useRef([]);
  const eventStateRef = useRef(new Map()); // event_id → { pin, wavefront, ts }
  const droneStateRef = useRef({
    dispatchedFor: null,
    marker: null,
    arrival: null,
    rafId: null,
  });
  const [providerNote, setProviderNote] = useState('Mapbox GL JS');
  const [eta, setEta] = useState(null);

  // Bootstrap the map exactly once.
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return;
    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [72.876, 19.115],
      zoom: 13,
      attributionControl: true,
    });
    map.addControl(new mapboxgl.NavigationControl({ showCompass: false }), 'top-right');
    mapRef.current = map;
    setProviderNote('Mapbox GL JS');

    return () => {
      // Tear down sensor markers + active crash pins before removing the map.
      sensorMarkersRef.current.forEach((m) => m.remove());
      sensorMarkersRef.current = [];
      eventStateRef.current.forEach(({ pin, wavefront }) => {
        pin?.remove();
        wavefront?.remove();
      });
      eventStateRef.current.clear();
      const drone = droneStateRef.current;
      drone.marker?.remove();
      drone.arrival?.remove();
      if (drone.rafId) cancelAnimationFrame(drone.rafId);
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, []);

  // Render sensor markers.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !sensors?.length) return;

    const finalize = () => {
      sensorMarkersRef.current.forEach((m) => m.remove());
      sensorMarkersRef.current = [];

      const bounds = new mapboxgl.LngLatBounds();
      sensors.forEach((sensor) => {
        const el = buildSensorElement(sensor);
        const marker = new mapboxgl.Marker({ element: el, anchor: 'center' })
          .setLngLat([sensor.lon, sensor.lat])
          .addTo(map);
        sensorMarkersRef.current.push(marker);
        bounds.extend([sensor.lon, sensor.lat]);
      });
      if (sensors.length >= 2) {
        map.fitBounds(bounds, { padding: 60, animate: false });
      }
    };

    if (map.loaded()) finalize();
    else map.once('load', finalize);
  }, [sensors]);

  // Render incoming crash event + run the drone animation inline.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !latestEvent) return;
    const {
      event_id,
      crash_lat,
      crash_lon,
      drone_origin_lat,
      drone_origin_lon,
      status,
    } = latestEvent;
    if (
      typeof crash_lat !== 'number' ||
      typeof crash_lon !== 'number' ||
      crash_lat < -90 ||
      crash_lat > 90 ||
      crash_lon < -180 ||
      crash_lon > 180
    ) {
      return;
    }

    const placePin = () => {
      if (eventStateRef.current.has(event_id)) return;
      const pinEl = buildCrashPinElement(crash_lat, crash_lon);
      const pin = new mapboxgl.Marker({ element: pinEl, anchor: 'bottom' })
        .setLngLat([crash_lon, crash_lat])
        .addTo(map);

      const waveEl = buildWavefrontElement();
      const wavefront = new mapboxgl.Marker({ element: waveEl, anchor: 'center' })
        .setLngLat([crash_lon, crash_lat])
        .addTo(map);
      setTimeout(() => {
        wavefront.remove();
        const entry = eventStateRef.current.get(event_id);
        if (entry) entry.wavefront = null;
      }, 3200);

      eventStateRef.current.set(event_id, { pin, wavefront, ts: Date.now() });

      // Cap the number of active pins.
      if (eventStateRef.current.size > MAX_PINS) {
        const [oldestId, oldestEntry] = [...eventStateRef.current.entries()][0];
        oldestEntry.pin?.remove();
        oldestEntry.wavefront?.remove();
        eventStateRef.current.delete(oldestId);
      }

      map.flyTo({ center: [crash_lon, crash_lat], zoom: 15, duration: 1200 });
    };

    if (map.loaded()) placePin();
    else map.once('load', placePin);

    // Drone animation parity with `<DroneTracker>` (Leaflet path).
    if (
      status === 'DETECTED' &&
      typeof drone_origin_lat === 'number' &&
      typeof drone_origin_lon === 'number' &&
      droneStateRef.current.dispatchedFor !== event_id
    ) {
      droneStateRef.current.dispatchedFor = event_id;
      // Clean up any previous run.
      droneStateRef.current.marker?.remove();
      droneStateRef.current.arrival?.remove();
      if (droneStateRef.current.rafId) cancelAnimationFrame(droneStateRef.current.rafId);

      const droneEl = buildDroneElement();
      const droneMarker = new mapboxgl.Marker({ element: droneEl, anchor: 'center' })
        .setLngLat([drone_origin_lon, drone_origin_lat])
        .addTo(map);
      droneStateRef.current.marker = droneMarker;

      const start = { lat: drone_origin_lat, lon: drone_origin_lon };
      const end = { lat: crash_lat, lon: crash_lon };
      const startedAt = performance.now();

      postDroneDispatched(latestEvent).catch((err) =>
        console.warn('drone-dispatched POST failed', err),
      );

      const tick = (now) => {
        const elapsed = now - startedAt;
        const progress = Math.min(1, elapsed / ANIMATION_DURATION_MS);
        const { lat, lon } = interpolatePosition(start, end, progress);
        droneMarker.setLngLat([lon, lat]);
        const remaining = Math.max(0, Math.ceil((ANIMATION_DURATION_MS - elapsed) / 1000));
        setEta(remaining);
        if (progress < 1) {
          droneStateRef.current.rafId = requestAnimationFrame(tick);
        } else {
          setEta(0);
          const arrivalEl = buildArrivalElement();
          const arrival = new mapboxgl.Marker({ element: arrivalEl, anchor: 'center' })
            .setLngLat([crash_lon, crash_lat])
            .addTo(map);
          droneStateRef.current.arrival = arrival;
          postDroneArrived(latestEvent).catch((err) =>
            console.warn('drone-arrived POST failed', err),
          );
          setTimeout(() => {
            arrival.remove();
            droneStateRef.current.arrival = null;
            setEta(null);
          }, 2200);
        }
      };
      droneStateRef.current.rafId = requestAnimationFrame(tick);
    }
  }, [latestEvent]);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="absolute inset-0" />
      <div className="pointer-events-none absolute bottom-3 left-3 rounded-full border border-slate-700/60 bg-slate-900/70 px-3 py-1 text-[11px] font-mono text-slate-300">
        Map provider: {providerNote}
      </div>
      {eta != null && (
        <div className="pointer-events-none absolute right-4 top-4 rounded-xl border border-cyan-500/30 bg-slate-900/85 px-4 py-2 text-sm shadow-lg backdrop-blur">
          <div className="font-mono text-xs uppercase tracking-wider text-cyan-300">
            Drone in transit
          </div>
          <div className="mt-0.5 font-mono text-2xl font-semibold text-cyan-100">
            ETA {eta}s
          </div>
        </div>
      )}
    </div>
  );
}
