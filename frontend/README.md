# CrashSense Dashboard (frontend)

React + Vite single-page app that renders the Map_View, AlertPanel, and
DroneTracker for CrashSense.

## Quick start

```bash
npm install
npm run dev      # http://127.0.0.1:5173
npm run build    # production bundle in dist/
npm run preview  # serve dist/ locally on :4173
```

A default `npm install` does **not** pull `mapbox-gl` or
`@mapbox/mapbox-gl-geocoder`. They live under `optionalDependencies`, so
the Leaflet code path works on a clean checkout without a Mapbox account.

## Environment variables

All env vars are read at build time via Vite (`import.meta.env`). Set them
in a `.env`, `.env.local`, or directly on the build command.

| Name                  | Required             | Default     | Values                   |
| --------------------- | -------------------- | ----------- | ------------------------ |
| `VITE_MAP_PROVIDER`   | no                   | `leaflet`   | `leaflet` \| `mapbox`    |
| `VITE_MAPBOX_API_KEY` | only when `=mapbox`  | _(empty)_   | Mapbox public token (`pk.…`) |

Selection logic (see `src/utils/mapProvider.js`):

- `VITE_MAP_PROVIDER=mapbox` **and** a non-empty `VITE_MAPBOX_API_KEY` →
  Mapbox GL JS.
- Anything else (unset, `leaflet`, or `mapbox` without a key) → Leaflet
  over OpenStreetMap tiles, without console errors. (R22.4)

### Default (Leaflet)

```bash
npm install
npm run build
```

### Enable Mapbox

```bash
# install the optional Mapbox packages explicitly
npm install --include=optional

# build with the Mapbox provider
VITE_MAP_PROVIDER=mapbox \
VITE_MAPBOX_API_KEY=pk.your_token_here \
  npm run build
```

If `VITE_MAP_PROVIDER=mapbox` is set but the key is missing or empty, the
build still succeeds and the runtime falls back to Leaflet silently.
