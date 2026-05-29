import { Component, Suspense, lazy } from 'react';
import { IS_MAPBOX } from '../utils/mapProvider.js';
import LeafletMap from './LeafletMap.jsx';

// `<MapAdapter>` selects the map renderer at build time based on
// `VITE_MAP_PROVIDER` and the presence of `VITE_MAPBOX_API_KEY`. Both code
// paths share this single component interface (R22.1, R22.2, R22.3, R22.4).
//
// Selection (decided in `utils/mapProvider.js`):
//   - VITE_MAP_PROVIDER === "mapbox" AND non-empty VITE_MAPBOX_API_KEY
//       → IS_MAPBOX = true → render `<MapboxMap>`
//   - any other combination (unset / "leaflet" / "mapbox" without a key)
//       → IS_MAPBOX = false → render `<LeafletMap>` directly
//
// When IS_MAPBOX is false, `MapboxMap.jsx` is never imported (the `lazy()`
// factory below is only evaluated when IS_MAPBOX is true), so the default
// build does not pull `mapbox-gl` even though it sits in the source tree.

// Only construct the lazy loader when Mapbox is selected. Using a top-level
// ternary keeps `import('./MapboxMap.jsx')` out of the bundle in the
// Leaflet path — Vite will tree-shake the unreachable branch.
const MapboxMapLazy = IS_MAPBOX
  ? lazy(() =>
      import('./MapboxMap.jsx').catch((err) => {
        // Mapbox dependency missing at runtime (e.g. `npm install` ran
        // without `--include=optional`). Swap to Leaflet without surfacing
        // a console error tied to a missing Mapbox key (R22.4).
        console.info('MapAdapter: Mapbox unavailable, falling back to Leaflet.', err?.message);
        return { default: LeafletMap };
      }),
    )
  : null;

// Error boundary wraps the Mapbox tree so a runtime failure inside
// `<MapboxMap>` (key rejected, GL context lost, etc.) falls back to
// Leaflet without breaking the dashboard.
class MapboxErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error) {
    console.info('MapAdapter: Mapbox runtime error, falling back to Leaflet.', error?.message);
  }
  render() {
    if (this.state.failed) {
      return <LeafletMap {...this.props.fallbackProps} />;
    }
    return this.props.children;
  }
}

export default function MapAdapter(props) {
  if (!IS_MAPBOX || !MapboxMapLazy) {
    return <LeafletMap {...props} />;
  }
  return (
    <MapboxErrorBoundary fallbackProps={props}>
      <Suspense fallback={<LeafletMap {...props} />}>
        <MapboxMapLazy {...props} />
      </Suspense>
    </MapboxErrorBoundary>
  );
}
