// Map_View provider selection for the Dashboard.
//
// Build-time env vars (read via Vite's `import.meta.env`):
//   VITE_MAP_PROVIDER     "leaflet" (default) | "mapbox"
//   VITE_MAPBOX_API_KEY   Required only when VITE_MAP_PROVIDER === "mapbox".
//
// Mapbox is selected only when both the provider flag is "mapbox" AND a
// non-empty API key is present at build time. Any other combination
// (unset, "leaflet", "mapbox" without a key) falls back to Leaflet without
// emitting a console error. See R22.2-R22.5.

const rawProvider = import.meta.env.VITE_MAP_PROVIDER;
const rawKey = import.meta.env.VITE_MAPBOX_API_KEY;

export const MAPBOX_API_KEY =
  typeof rawKey === "string" && rawKey.length > 0 ? rawKey : "";

export const MAP_PROVIDER =
  rawProvider === "mapbox" && MAPBOX_API_KEY ? "mapbox" : "leaflet";

export const IS_MAPBOX = MAP_PROVIDER === "mapbox";
export const IS_LEAFLET = MAP_PROVIDER === "leaflet";
