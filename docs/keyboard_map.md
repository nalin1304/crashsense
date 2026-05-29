# Keyboard Activation Map

**Status:** Implemented (task 6.8) — pointer-free Dashboard navigation contract
**Owner:** Frontend team
**Last updated:** 2025-01-15
**Related spec:** `.kiro/specs/crashsense-hardening/` — R25.3
**Related design section:** `design.md` §3.4 (Parallel tracks sketch — Accessibility audit) and `tasks.md` 6.8 (this document)

This document is the literal R25.3 deliverable: a flat enumeration of every interactive element on the Dashboard and the keys that focus or activate it. Operators using assistive technology, or any keyboard-only operator, must be able to reach and activate every row below using only **Tab, Shift+Tab, Enter, Space, Escape, and Arrow keys**. Each row also names the source file so a reader can audit the implementation directly.

The Dashboard is a single-page React app rooted at `frontend/src/App.jsx`. Element order below follows DOM order, which is also the default Tab order — there are no `tabindex` overrides above `0`. Shift+Tab walks the same sequence in reverse.

---

## 1. Top status bar

Source: `frontend/src/App.jsx` `<header>`.

| Element | Tab order | Activation key(s) | Notes |
| --- | --- | --- | --- |
| `CrashSense` brand label | not focusable | — | Static text. Not interactive. |
| `Acoustic Triangulation` mode chip | not focusable | — | Static text. Not interactive. |
| WebSocket connection indicator (`WS live` / `WS offline`) | not focusable | — | Informational. Backed by `aria-hidden="true"` dot plus text. R25.4 contrast applies; no keyboard activation. |
| Crash status badge (`MONITORING` / `CRASH DETECTED`) | not focusable | — | `role="status" aria-live="polite"`. Announces on change to assistive tech. Not a button. |

All status-bar elements are intentionally non-interactive — the bar reports state, it does not accept input.

---

## 2. Map container

Source: `frontend/src/components/MapAdapter.jsx` → `LeafletMap.jsx` (default) or `MapboxMap.jsx` (`VITE_MAP_PROVIDER=mapbox`).

| Element | Tab order | Activation key(s) | Notes |
| --- | --- | --- | --- |
| Map viewport | 1 | Tab focuses; **Arrow keys** pan; **`+` / `-`** (or **`=`**) zoom | Leaflet's built-in `keyboard: true` handler is enabled by default; Mapbox GL has the same key set. Focus indicator is the browser default outline plus the `:focus-visible` ring baked into the Tailwind theme (R25.2 ≥3:1). |
| Zoom-in control (`+`) | 2 | **Enter** or **Space** | Leaflet renders zoom controls as `<a role="button">`; both keys activate. |
| Zoom-out control (`-`) | 3 | **Enter** or **Space** | Same as zoom-in. |
| Sensor circle markers | not focusable | — | Decorative SVG. Information is duplicated in the alert cards and the (forthcoming) sensor list. |
| Crash event markers / drone polylines | not focusable | — | Same rationale. |

If `VITE_MAP_PROVIDER=mapbox` is enabled, an additional **Geocoder search box** (`@mapbox/mapbox-gl-geocoder`) inserts itself after the zoom controls: Tab focuses the input, **typing** filters results, **Arrow keys** walk results, **Enter** selects, **Escape** closes. Default Leaflet build (the only currently shipped path) does not include that control.

---

## 3. Floating Simulate Crash button

Source: `frontend/src/App.jsx` (button with `data-testid="simulate-crash-floating"`).

| Element | Tab order | Activation key(s) | Notes |
| --- | --- | --- | --- |
| `Simulate Crash` button | 4 | **Enter** or **Space** | Disabled until at least three sensors have loaded (`sensorTriangle != null`). Disabled state is conveyed by `disabled` attribute, not just opacity, so screen readers announce it correctly. Hit target ≥44 × 44 px on mobile (R26.5). |

When focused while disabled, the browser will not invoke `onClick`; assistive tech announces the disabled state and Tab continues to the next element.

---

## 4. Alert panel — controls

Source: `frontend/src/components/AlertPanel.jsx`.

| Element | Tab order | Activation key(s) | Notes |
| --- | --- | --- | --- |
| Status filter chips (`active`, `resolved`) | 5, 6 | **Space** or **Enter** to toggle | Per R24.1 these are multi-select toggles. Chips render as `<button aria-pressed="true|false">`; pressed state is the toggle truth. Task 6.5 owns the chip implementation. |
| Severity filter chips (`minor`, `moderate`, `severe`) | 7, 8, 9 | **Space** or **Enter** to toggle | Same `aria-pressed` pattern as the status chips (R24.2). |
| Sort dropdown (`newest_first` / `oldest_first` / `severity_high_to_low` / `severity_low_to_high`) | 10 | **Enter** or **Space** opens; **Arrow Up / Arrow Down** walks options; **Enter** selects; **Escape** closes without selecting | Implemented as a native `<select>` so all key bindings are inherited from the user agent and stay consistent with assistive technology expectations (R24.3, R25.3). |
| `Clear filters` action (visible only when no card matches) | 11 | **Enter** or **Space** | Resets the chip and dropdown state so the empty-state message is dismissable without a pointer (R24.6). |
| `Simulate Crash` (in-panel) | 12 | **Enter** or **Space** | Same effect as the floating button. Kept in the panel for parity with mouse layouts. |
| `Replay last event` | 13 | **Enter** or **Space** | Disabled until the first CrashEvent arrives. When enabled, triggers `onReplay` which feeds `<DroneTracker>` in replay mode (R23.1). |

The filter chip and sort dropdown rows are owned by task 6.5; this map documents their final keyboard contract so the implementation has a target to build against.

---

## 5. Alert panel — crash cards

Source: `frontend/src/components/AlertPanel.jsx` (`<ul><li>` per event).

| Element | Tab order | Activation key(s) | Notes |
| --- | --- | --- | --- |
| Crash card container | 14, 15, … (one per active card) | **Tab** to focus next card; **Shift+Tab** to walk back | Each card is a focusable `<li tabindex="0">` so screen readers can land on it as a discrete region. |
| Card timestamp / coordinates / status badge / metadata | not focusable | — | Read aloud as part of the parent card's accessible name and description; no separate focus stop. |
| `Replay` action on a specific card (when present) | 14a, 15a, … | **Enter** or **Space** | Each card may surface an inline replay control to re-run *that* event's animation. Same `<button>` semantics as the panel-level Replay. |

Card focus order follows the rendered order (newest first by default; sort dropdown can flip it). When a sort or filter operation re-orders the list, focus stays on the previously-focused card by `event_id` if it is still in the result set; otherwise focus falls back to the panel header so Tab continues into the new top card.

---

## 6. Mobile bottom drawer

Source: `frontend/src/components/AlertPanel.jsx` (`<aside aria-label="Crash alerts" data-viewport-mode="mobile">`).

| Element | Tab order | Activation key(s) | Notes |
| --- | --- | --- | --- |
| Drawer container | logical entry point on mobile | **Tab** focuses first child (filter chips, then cards) | Drawer is open when `isCrashActive` is true. Focus is **not** trapped — Tab eventually wraps to the map, which matches the desktop side-panel behaviour. |
| Drawer close affordance | last drawer focus stop | **Enter** or **Space** to close; **Escape** also closes | On mobile the close affordance is exposed as a button in the drawer header; **Escape** anywhere inside the drawer dismisses it. |

Closing the drawer returns focus to the element that opened it (the floating Simulate Crash button on user-triggered opens; the alert badge on auto-open from a live CrashEvent), so the operator does not lose their place in the Tab sequence.

---

## 7. Replay banner

Source: `frontend/src/components/ReplayController.jsx` (task 6.4) and `frontend/src/components/DroneTracker.jsx`.

| Element | Tab order | Activation key(s) | Notes |
| --- | --- | --- | --- |
| `REPLAY (<original_timestamp>)` banner | injected at top of focus order while replay active | — | Non-modal, `role="status"`, announced by `aria-live="polite"`. Does not steal focus. |
| Banner close button (`×`) | first focusable element while banner visible | **Enter** or **Space** | Dismisses replay mode (R23.2). After dismissal, focus returns to the card that initiated the replay. |

The banner does not block live CrashEvents — concurrent live events continue to render in the alert panel underneath, and Tab order continues into them after the banner close button.

---

## 8. Global key conventions

| Key | Effect anywhere on the Dashboard |
| --- | --- |
| **Tab** | Move focus forward through the order documented above. |
| **Shift+Tab** | Move focus backward through the same order. |
| **Enter** | Activate the focused button, link, card, or selected dropdown option. |
| **Space** | Activate the focused button or toggle the focused chip; opens a focused `<select>`. |
| **Escape** | Close the sort dropdown, the mobile drawer, or the replay banner — whichever is currently topmost. |
| **Arrow keys** | Pan the map when the map has focus; walk options inside an open sort dropdown. |
| **`+` / `-` / `=`** | Zoom the map (Leaflet/Mapbox built-in) when the map has focus. |

There are no chord shortcuts (Ctrl+, Alt+, Cmd+) reserved by the application. This avoids collisions with screen-reader and browser shortcuts.

---

## 9. References

- `frontend/src/App.jsx` — top-level layout, status bar, floating Simulate Crash button
- `frontend/src/components/MapAdapter.jsx` — Leaflet/Mapbox selection
- `frontend/src/components/AlertPanel.jsx` — filter chips, sort dropdown, card list, drawer behaviour
- `frontend/src/components/ReplayController.jsx` — replay banner and close button
- `frontend/src/styles/reduced_motion.css` — focus-visible ring and reduced-motion rules referenced by R25.2 and R25.5
- `.kiro/specs/crashsense-hardening/requirements.md` — R25.3, R24, R26.5
- `docs/accessibility_audit.md` — manual-testing items that complement this map
