import { useCallback, useMemo, useState } from 'react';
import { postSimulateCrash } from '../api.js';

const STATUS_BADGES = {
  DETECTED: {
    label: 'DETECTING',
    className: 'bg-amber-400/20 text-amber-300 border-amber-400/40',
  },
  DRONE_DISPATCHED: {
    label: 'DRONE DISPATCHED',
    className: 'bg-orange-500/20 text-orange-300 border-orange-500/40',
  },
  DRONE_ARRIVED: {
    label: 'DRONE ARRIVED',
    className: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
  },
  // R16.6 + R24.6: terminal failure path counts as "resolved" for filter purposes.
  DISPATCH_FAILED: {
    label: 'DISPATCH FAILED',
    className: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
  },
};

const FALLBACK_BADGE = {
  label: 'UNKNOWN',
  className: 'bg-slate-500/20 text-slate-300 border-slate-500/40',
};

// R24.6: classify a CrashEvent's status as either "active" or "resolved".
// Returns null for unknown statuses so they never match a status filter.
const ACTIVE_STATUSES = new Set(['DETECTED', 'DRONE_DISPATCHED']);
const RESOLVED_STATUSES = new Set(['DRONE_ARRIVED', 'DISPATCH_FAILED']);

function classifyStatus(status) {
  if (ACTIVE_STATUSES.has(status)) return 'active';
  if (RESOLVED_STATUSES.has(status)) return 'resolved';
  return null;
}

// R24.1 / R24.6: severity is one of {minor, moderate, severe}. The backend
// SeverityInfo schema currently exposes label ∈ {minor, moderate, major};
// we normalise "major" → "severe" so the filter vocabulary matches the spec
// regardless of which end of the contract is updated next.
const SEVERITY_LABELS = ['minor', 'moderate', 'severe'];
const SEVERITY_RANK = { severe: 3, moderate: 2, minor: 1 };

function getSeverityLabel(evt) {
  let raw = evt?.severity_label;
  if (raw == null) raw = evt?.severity?.label;
  if (raw === 'major') return 'severe';
  if (SEVERITY_LABELS.includes(raw)) return raw;
  return 'moderate';
}

const SORT_OPTIONS = [
  { value: 'newest_first', label: 'Newest first' },
  { value: 'oldest_first', label: 'Oldest first' },
  { value: 'severity_high_to_low', label: 'Severity (high → low)' },
  { value: 'severity_low_to_high', label: 'Severity (low → high)' },
];

function makeDefaultFilters() {
  return {
    statuses: new Set(['active', 'resolved']),
    severities: new Set(SEVERITY_LABELS),
    sortMode: 'newest_first',
  };
}

function StatusBadge({ status }) {
  const meta = STATUS_BADGES[status] || FALLBACK_BADGE;
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${meta.className}`}
    >
      {meta.label}
    </span>
  );
}

function formatTimestamp(value) {
  try {
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return value;
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  } catch {
    return String(value);
  }
}

function reduceEvents(events) {
  // Collapse multiple status updates for the same event_id into one card,
  // keeping the most recent status. Newest first.
  const byId = new Map();
  events.forEach((evt) => {
    if (!evt || typeof evt !== 'object' || !evt.event_id) return;
    byId.set(evt.event_id, evt);
  });
  return Array.from(byId.values()).reverse();
}

function timestampMs(evt) {
  const t = new Date(evt?.timestamp).getTime();
  return Number.isFinite(t) ? t : 0;
}

// Per-viewport positioning rules (R26.2–R26.4):
//   mobile  : bottom drawer, 60vh tall, full-width
//   tablet  : right side panel, 30% viewport width
//   desktop : right side panel, 20% viewport width (base-spec)
function panelPositioningClasses(viewportMode, open) {
  if (viewportMode === 'mobile') {
    return {
      base:
        'pointer-events-auto fixed inset-x-0 bottom-0 z-30 flex flex-col border-t border-slate-800/80 bg-slate-950/95 backdrop-blur transition-transform duration-300 ease-out',
      transform: open ? 'translate-y-0' : 'translate-y-full',
      style: { height: '60vh' },
    };
  }
  // Tablet (768–1023) and desktop (≥1024) share the right-panel anatomy;
  // only the width differs.
  const widthStyle =
    viewportMode === 'tablet'
      ? { width: '30vw', minWidth: 280 }
      : { width: 'min(20vw, 460px)', minWidth: 320 };
  return {
    base:
      'pointer-events-auto fixed right-0 top-0 z-30 flex h-full flex-col border-l border-slate-800/80 bg-slate-950/95 backdrop-blur transition-transform duration-300 ease-out',
    transform: open ? 'translate-x-0' : 'translate-x-full',
    style: widthStyle,
  };
}

function FilterChip({ active, onClick, children, label }) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={active}
      aria-label={label}
      onClick={onClick}
      className={`cursor-pointer rounded-full border px-2.5 py-1 text-[11px] font-medium uppercase tracking-wider transition-colors duration-200 ${
        active
          ? 'border-cyan-400/60 bg-cyan-500/15 text-cyan-100 hover:bg-cyan-500/25'
          : 'border-slate-700/70 bg-slate-900/70 text-slate-400 hover:border-slate-600 hover:text-slate-200'
      }`}
    >
      {children}
    </button>
  );
}

export default function AlertPanel({
  open,
  events,
  sensorTriangle,
  onReplay,
  hasEvent,
  viewportMode = 'desktop',
}) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState(makeDefaultFilters);

  const allCards = useMemo(() => reduceEvents(events), [events]);

  // R24.2 + R24.4: filter and sort entirely on the client. Microsecond-scale
  // for the order of magnitude of cards we keep in memory (~100), well under
  // the 100 ms re-render budget.
  const visibleCards = useMemo(() => {
    const { statuses, severities, sortMode } = filters;

    const filtered = allCards.filter((evt) => {
      const cls = classifyStatus(evt.status);
      if (cls === null) return false;
      if (!statuses.has(cls)) return false;
      const sev = getSeverityLabel(evt);
      if (!severities.has(sev)) return false;
      return true;
    });

    const byNewestFirst = (a, b) => timestampMs(b) - timestampMs(a);
    const bySeverity = (dir) => (a, b) => {
      const ra = SEVERITY_RANK[getSeverityLabel(a)] || 0;
      const rb = SEVERITY_RANK[getSeverityLabel(b)] || 0;
      if (ra !== rb) return dir === 'desc' ? rb - ra : ra - rb;
      return byNewestFirst(a, b); // tie-break: newest first
    };

    const sorted = [...filtered];
    if (sortMode === 'oldest_first') {
      sorted.sort((a, b) => timestampMs(a) - timestampMs(b));
    } else if (sortMode === 'severity_high_to_low') {
      sorted.sort(bySeverity('desc'));
    } else if (sortMode === 'severity_low_to_high') {
      sorted.sort(bySeverity('asc'));
    } else {
      sorted.sort(byNewestFirst);
    }
    return sorted;
  }, [allCards, filters]);

  const toggleStatus = useCallback((status) => {
    setFilters((prev) => {
      const next = new Set(prev.statuses);
      if (next.has(status)) next.delete(status);
      else next.add(status);
      return { ...prev, statuses: next };
    });
  }, []);

  const toggleSeverity = useCallback((sev) => {
    setFilters((prev) => {
      const next = new Set(prev.severities);
      if (next.has(sev)) next.delete(sev);
      else next.add(sev);
      return { ...prev, severities: next };
    });
  }, []);

  const setSortMode = useCallback((sortMode) => {
    setFilters((prev) => ({ ...prev, sortMode }));
  }, []);

  const clearFilters = useCallback(() => {
    setFilters(makeDefaultFilters());
  }, []);

  const filtersActive =
    filters.statuses.size !== 2 || filters.severities.size !== SEVERITY_LABELS.length;

  // R24.5: empty-state summary identifying the active filters.
  const activeFilterSummary = useMemo(() => {
    const parts = [];
    if (filters.statuses.size === 0) {
      parts.push('no statuses');
    } else if (filters.statuses.size < 2) {
      parts.push(`status: ${[...filters.statuses].join(', ')}`);
    }
    if (filters.severities.size === 0) {
      parts.push('no severities');
    } else if (filters.severities.size < SEVERITY_LABELS.length) {
      parts.push(`severity: ${[...filters.severities].join(', ')}`);
    }
    return parts.join(' · ');
  }, [filters]);

  const handleSimulate = async () => {
    if (submitting) return;
    if (!sensorTriangle) {
      setError('Sensor data not loaded yet.');
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const { p1, p2, p3 } = sensorTriangle;
      let r1 = Math.random();
      let r2 = Math.random();
      if (r1 + r2 > 1) {
        r1 = 1 - r1;
        r2 = 1 - r2;
      }
      const r3 = 1 - r1 - r2;
      const lat = r1 * p1.lat + r2 * p2.lat + r3 * p3.lat;
      const lon = r1 * p1.lon + r2 * p2.lon + r3 * p3.lon;
      await postSimulateCrash(lat, lon);
    } catch (err) {
      setError(err.message || 'Simulation failed');
    } finally {
      setSubmitting(false);
    }
  };

  const { base, transform, style } = panelPositioningClasses(viewportMode, open);
  const totalCount = allCards.length;
  const visibleCount = visibleCards.length;

  return (
    <aside
      aria-label="Crash alerts"
      data-viewport-mode={viewportMode}
      className={`${base} ${transform}`}
      style={style}
    >
      <header className="flex items-center justify-between border-b border-slate-800/80 px-5 py-4">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-slate-400">Alerts</p>
          <h2 className="text-lg font-semibold text-slate-100">Active crash queue</h2>
        </div>
        <span
          className="rounded-full border border-slate-700/60 bg-slate-900/60 px-2.5 py-1 font-mono text-xs text-slate-300"
          aria-label={
            filtersActive
              ? `${visibleCount} of ${totalCount} alerts visible`
              : `${totalCount} alerts`
          }
        >
          {filtersActive && visibleCount !== totalCount
            ? `${visibleCount} / ${totalCount}`
            : totalCount}
        </span>
      </header>

      <div className="flex flex-col gap-2 px-5 py-3">
        <button
          type="button"
          onClick={handleSimulate}
          disabled={submitting}
          className="cursor-pointer rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-2 text-sm font-semibold text-cyan-200 transition-colors duration-200 hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {submitting ? 'Simulating…' : 'Simulate Crash'}
        </button>
        <button
          type="button"
          onClick={onReplay}
          disabled={!hasEvent}
          className="cursor-pointer rounded-lg border border-slate-700/70 bg-slate-900/70 px-3 py-2 text-sm font-medium text-slate-200 transition-colors duration-200 hover:bg-slate-800/80 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Replay last event
        </button>
        {error && (
          <p role="alert" className="text-xs font-medium text-rose-300">
            {error}
          </p>
        )}
      </div>

      {/* R24.1 / R24.3: filter chips + sort dropdown rendered at the top of
          the panel (above the card list). Status and severity are
          multi-selects implemented as toggle chips. */}
      <section
        aria-label="Alert filters"
        className="flex flex-col gap-3 border-t border-slate-800/60 px-5 py-3"
      >
        <div className="flex flex-col gap-1.5" role="group" aria-label="Status filter">
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500">Status</p>
          <div className="flex flex-wrap gap-1.5">
            <FilterChip
              active={filters.statuses.has('active')}
              onClick={() => toggleStatus('active')}
              label="Toggle active status filter"
            >
              Active
            </FilterChip>
            <FilterChip
              active={filters.statuses.has('resolved')}
              onClick={() => toggleStatus('resolved')}
              label="Toggle resolved status filter"
            >
              Resolved
            </FilterChip>
          </div>
        </div>

        <div className="flex flex-col gap-1.5" role="group" aria-label="Severity filter">
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500">Severity</p>
          <div className="flex flex-wrap gap-1.5">
            {SEVERITY_LABELS.map((sev) => (
              <FilterChip
                key={sev}
                active={filters.severities.has(sev)}
                onClick={() => toggleSeverity(sev)}
                label={`Toggle ${sev} severity filter`}
              >
                {sev}
              </FilterChip>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <label
            htmlFor="alert-sort"
            className="font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500"
          >
            Sort
          </label>
          <select
            id="alert-sort"
            value={filters.sortMode}
            onChange={(e) => setSortMode(e.target.value)}
            className="cursor-pointer rounded-lg border border-slate-700/70 bg-slate-900/70 px-2.5 py-1.5 text-xs text-slate-200 transition-colors duration-200 hover:border-slate-600 focus:border-cyan-400/60 focus:outline-none focus:ring-1 focus:ring-cyan-400/40"
          >
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </section>

      <div className="flex-1 overflow-y-auto px-5 pb-5 pt-3">
        {totalCount === 0 ? (
          <div className="flex h-full items-center justify-center text-center">
            <p className="text-sm text-slate-500">
              No crash events yet. The dashboard is monitoring.
            </p>
          </div>
        ) : visibleCount === 0 ? (
          // R24.5: empty-state when filters hide every card.
          <div
            role="status"
            className="flex h-full flex-col items-center justify-center gap-3 text-center"
          >
            <p className="text-sm text-slate-300">No alerts match the active filters.</p>
            {activeFilterSummary && (
              <p className="font-mono text-[11px] uppercase tracking-wider text-slate-500">
                {activeFilterSummary}
              </p>
            )}
            <button
              type="button"
              onClick={clearFilters}
              className="cursor-pointer rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold text-cyan-200 transition-colors duration-200 hover:bg-cyan-500/20"
            >
              Clear filters
            </button>
          </div>
        ) : (
          <ul className="flex flex-col gap-3">
            {visibleCards.map((evt) => (
              <li
                key={evt.event_id}
                className="rounded-xl border border-slate-800/80 bg-slate-900/70 p-4 shadow-lg shadow-black/30"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-mono text-[11px] uppercase tracking-wider text-slate-400">
                      {formatTimestamp(evt.timestamp)}
                    </p>
                    <p className="mt-0.5 font-mono text-sm text-slate-100">
                      {Number(evt.crash_lat).toFixed(6)}, {Number(evt.crash_lon).toFixed(6)}
                    </p>
                  </div>
                  <StatusBadge status={evt.status} />
                </div>
                <dl className="mt-3 grid grid-cols-2 gap-2 text-[12px]">
                  <div>
                    <dt className="text-slate-500">Confidence</dt>
                    <dd className="font-mono text-slate-200">
                      {((Number(evt.confidence) || 0) * 100).toFixed(1)}%
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-500">Severity</dt>
                    <dd className="font-mono text-slate-200 capitalize">
                      {getSeverityLabel(evt)}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-500">Nearest sensor</dt>
                    <dd className="font-mono text-slate-200">{evt.nearest_sensor}</dd>
                  </div>
                  <div>
                    <dt className="text-slate-500">ETA</dt>
                    <dd className="font-mono text-slate-200">{evt.eta_seconds}s</dd>
                  </div>
                  <div className="col-span-2">
                    <dt className="text-slate-500">Drone origin</dt>
                    <dd className="font-mono text-slate-200">
                      {Number(evt.drone_origin_lat).toFixed(4)},{' '}
                      {Number(evt.drone_origin_lon).toFixed(4)}
                    </dd>
                  </div>
                </dl>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
