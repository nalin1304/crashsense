import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import DroneTracker from './DroneTracker.jsx';

// Leaflet-backed Map_View. Internal component for `<MapAdapter>`; selected
// at build time when `VITE_MAP_PROVIDER` is unset, equal to `"leaflet"`, or
// equal to `"mapbox"` without a `VITE_MAPBOX_API_KEY` (R22.1, R22.4).

// Inline drone-dock SVG used for the toll plazas.
const DRONE_DOCK_SVG = `
<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="dockG" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#22d3ee"/>
      <stop offset="100%" stop-color="#0ea5e9"/>
    </linearGradient>
  </defs>
  <circle cx="16" cy="16" r="13" fill="url(#dockG)" stroke="#082f49" stroke-width="2"/>
  <path d="M9 12h14M16 9v14M11 16h10" stroke="#0f172a" stroke-width="1.6" stroke-linecap="round"/>
  <circle cx="16" cy="16" r="2.4" fill="#0f172a"/>
</svg>`;

const CRASH_PIN_SVG = `
<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="pinG" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#fb7185"/>
      <stop offset="100%" stop-color="#dc2626"/>
    </linearGradient>
  </defs>
  <path d="M16 2c-6 0-10 4.4-10 10 0 7.5 10 18 10 18s10-10.5 10-18c0-5.6-4-10-10-10z" fill="url(#pinG)" stroke="#7f1d1d" stroke-width="1.4"/>
  <circle cx="16" cy="12" r="3.6" fill="#fff5f5" stroke="#7f1d1d" stroke-width="1"/>
</svg>`;

function buildSensorMarker(sensor) {
  const isToll = sensor.is_toll_plaza;
  const html = isToll
    ? `<div class="sensor-marker toll"><span class="label">${sensor.id}</span>${DRONE_DOCK_SVG}</div>`
    : `<div class="sensor-marker"><span class="label">${sensor.id}</span><div class="pulse"></div></div>`;
  return L.divIcon({
    html,
    className: 'sensor-marker-wrapper',
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
}

function buildCrashIcon() {
  return L.divIcon({
    html: `<div class="crash-pin">${CRASH_PIN_SVG}</div>`,
    className: 'crash-pin-wrapper',
    iconSize: [32, 32],
    iconAnchor: [16, 30],
  });
}

function buildWavefrontIcon() {
  return L.divIcon({
    html: '<div class="wavefront-ring"></div>',
    className: 'wavefront-wrapper',
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
}

const MAX_PINS = 50;

export default function LeafletMap({ sensors, crashEvents, latestEvent }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const sensorLayerRef = useRef(null);
  const crashLayerRef = useRef(null);
  const eventStateRef = useRef(new Map());
  const [providerNote, setProviderNote] = useState('OpenStreetMap');

  // Bootstrap the map exactly once.
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return;
    const map = L.map(containerRef.current, {
      center: [19.115, 72.876],
      zoom: 14,
      zoomControl: true,
      preferCanvas: true,
    });
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19,
      // Single hostname; OSM deprecated subdomain sharding in 2021.
      subdomains: '',
    }).addTo(map);
    mapRef.current = map;
    sensorLayerRef.current = L.layerGroup().addTo(map);
    crashLayerRef.current = L.layerGroup().addTo(map);
    setProviderNote('OpenStreetMap (Leaflet)');
  }, []);

  // Render sensor markers.
  useEffect(() => {
    const map = mapRef.current;
    const layer = sensorLayerRef.current;
    if (!map || !layer || !sensors?.length) return;
    layer.clearLayers();
    const bounds = [];
    sensors.forEach((sensor) => {
      const marker = L.marker([sensor.lat, sensor.lon], {
        icon: buildSensorMarker(sensor),
        title: `${sensor.id} · ${sensor.name}`,
        keyboard: false,
      });
      marker.bindTooltip(`${sensor.id} · ${sensor.name}`, { direction: 'top' });
      marker.addTo(layer);
      bounds.push([sensor.lat, sensor.lon]);
    });
    if (bounds.length >= 2) {
      map.fitBounds(bounds, { padding: [60, 60] });
    }
  }, [sensors]);

  // Render incoming crash events.
  useEffect(() => {
    const map = mapRef.current;
    const layer = crashLayerRef.current;
    if (!map || !layer || !latestEvent) return;
    const { event_id, crash_lat, crash_lon } = latestEvent;
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
    if (eventStateRef.current.has(event_id)) return;

    const pin = L.marker([crash_lat, crash_lon], {
      icon: buildCrashIcon(),
      keyboard: false,
    }).addTo(layer);

    pin.bindTooltip(
      `<div class="crash-coord-label">CRASH ${crash_lat.toFixed(6)}, ${crash_lon.toFixed(6)}</div>`,
      { permanent: true, direction: 'top', offset: [0, -28], className: 'crash-coord-tooltip' },
    );

    const wavefront = L.marker([crash_lat, crash_lon], {
      icon: buildWavefrontIcon(),
      interactive: false,
    }).addTo(layer);
    setTimeout(() => layer.removeLayer(wavefront), 3200);

    eventStateRef.current.set(event_id, { pin, ts: Date.now() });

    // Cap the number of active pins.
    if (eventStateRef.current.size > MAX_PINS) {
      const [oldestId, oldestEntry] = [...eventStateRef.current.entries()][0];
      layer.removeLayer(oldestEntry.pin);
      eventStateRef.current.delete(oldestId);
    }

    map.flyTo([crash_lat, crash_lon], 15, { duration: 1.2 });
  }, [latestEvent]);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="absolute inset-0" />
      <div className="pointer-events-none absolute bottom-3 left-3 rounded-full border border-slate-700/60 bg-slate-900/70 px-3 py-1 text-[11px] font-mono text-slate-300">
        Map provider: {providerNote}
      </div>
      <DroneTracker map={mapRef} latestEvent={latestEvent} />
    </div>
  );
}
