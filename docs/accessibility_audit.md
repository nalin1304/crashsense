# Accessibility Audit

**Status:** Open — automated coverage in CI; manual coverage tracked below
**Owner:** Frontend team (rotates per quarter)
**Last updated:** 2025-01-15
**Related spec:** `.kiro/specs/crashsense-hardening/` — R25.6
**Related design section:** `design.md` §3.4 (Parallel tracks sketch — Accessibility audit) and `tasks.md` 6.8 (this document)

This document is the literal R25.6 deliverable. It tracks the accessibility work that **automated tooling cannot verify**: manual screen-reader walkthroughs, cognitive-load review, expert WCAG audit, manual contrast checks, reduced-motion verification, and a keyboard-only navigation drill. Every item below is **open work** until an owner signs the row with a date and a finding link.

The automated half of R25 lives elsewhere:

- R25.1 (axe-core gate) → `frontend/src/__tests__/a11y.spec.jsx` (task 6.6) and the `npm test` job in CI.
- R25.2 / R25.4 (focus indicator, contrast tokens) → `frontend/src/styles/reduced_motion.css` and Tailwind theme tokens (task 6.7).
- R25.3 (keyboard reachability) → `docs/keyboard_map.md` (task 6.8).
- R25.5 (reduced motion) → `frontend/src/styles/reduced_motion.css` (task 6.7).

The automated checks are necessary but not sufficient. Empirically, axe-core catches roughly 30–50 percent of the WCAG 2.1 AA failures a real user encounters; the rest land in this file.

---

## 1. Manual screen-reader walkthroughs

R25.6 calls these out by name. The walkthroughs below must be repeated on each major UI change (anything touching `App.jsx`, `AlertPanel.jsx`, `MapAdapter.jsx`, or `ReplayController.jsx`) and at least once per release.

### 1.1 Tooling matrix

| Screen reader | Browser | OS | Priority |
| --- | --- | --- | --- |
| NVDA (latest stable) | Firefox ESR | Windows 10 / 11 | **Required** |
| NVDA (latest stable) | Chrome stable | Windows 10 / 11 | Required |
| VoiceOver | Safari stable | macOS 14+ | **Required** |
| VoiceOver | Chrome stable | macOS 14+ | Recommended |
| TalkBack | Chrome stable | Android 13+ | Recommended (mobile drawer specifically) |
| Orca | Firefox ESR | Ubuntu 22.04 LTS | Optional |

Keep VoiceOver verbosity at default and NVDA at "Some" punctuation by default; the goal is to test what an operator with default settings actually hears.

### 1.2 Scenarios to walk

For each scenario record: the exact phrase the screen reader announces at each focus stop, any focus stop the operator could not predict, any element the screen reader skipped, and any element where the announcement contradicts the visible state.

| # | Scenario | Open work |
| --- | --- | --- |
| 1 | **Initial load** — open the Dashboard with no events; Tab from the address bar all the way through to the alert panel | unowned |
| 2 | **Crash arrival** — receive a `DETECTED → DRONE_DISPATCHED → DRONE_ARRIVED` sequence with the alert panel previously closed; verify the polite live region announces the badge change without stealing focus | unowned |
| 3 | **Filter usage (R24)** — toggle status and severity chips with Space, switch sort with the dropdown, hit the empty state, recover via `Clear filters` | unowned |
| 4 | **Replay (R23)** — trigger Replay from a specific crash card, verify the banner is announced once, dismiss with Escape, confirm focus returns to the originating card | unowned |
| 5 | **Sensor errors** — backend returns a 5xx on `/sensors`; verify the inline `role="alert"` reads aloud and that retries do not produce repeated announcements | unowned |
| 6 | **Mobile drawer (R26.2)** — at 375 px on TalkBack, swipe through the drawer, verify Escape (or the close affordance) dismisses without focus loss | unowned |

Each scenario gets its own row in the change log when completed (`[YYYY-MM-DD] [Owner] — findings link`).

---

## 2. Cognitive-load review

Not yet performed. Tracked here because R25.6 names cognitive-load review explicitly and because automated tooling cannot evaluate it.

A cognitive-load review answers questions automated checks cannot:

- Is the meaning of `MONITORING` versus `CRASH DETECTED` clear without prior training?
- Does the simultaneous appearance of the badge change, the alert card, and the drone animation overwhelm the operator on first exposure?
- Are filter chip labels (`active` / `resolved`, `minor` / `moderate` / `severe`) understandable to an operator who has not read the spec?
- Do error messages explain what to do next, or only what went wrong?

### 2.1 Checklist (open)

- [ ] Plain-language pass on every visible label (target: 8th-grade reading level)
- [ ] First-use walkthrough video recorded with an operator who has not seen the Dashboard before
- [ ] Two-minute "what is happening right now" test: a new operator opens the Dashboard mid-event and explains the state aloud
- [ ] Critical-path icon audit: every icon used to convey state has a text equivalent within 3 px
- [ ] Confirm color is never the sole indicator (cross-reference: status badge uses both color and the literal word `MONITORING` or `CRASH DETECTED`)

When closed, each item gets a `[YYYY-MM-DD] [Owner] — link` line below it.

---

## 3. Expert WCAG audit

R25's user story explicitly notes that full WCAG 2.1 AA conformance requires "an accessibility expert review". This row tracks that engagement.

### 3.1 Scope

- **Standard:** WCAG 2.1 Level AA (the spec's stated target). WCAG 2.2 success criteria reviewed informally; not part of the conformance claim.
- **Pages in scope:** Dashboard (only route), including all states: monitoring, dispatched, arrived, dispatch-failed, and replay.
- **Viewports in scope:** 375 px, 768 px, 1024 px, 1440 px (matches R26.1).

### 3.2 Open work

- [ ] Identify and engage an external accessibility specialist (firm or independent CPACC/WAS-credentialed reviewer)
- [ ] Provide them with this document, `docs/keyboard_map.md`, and a staging URL
- [ ] Receive findings report with severity per finding (Blocker / Major / Minor / Best Practice)
- [ ] File issues for every Blocker and Major finding; agree remediation timeline
- [ ] Re-audit after remediation; obtain conformance statement

This row is the gating reason CrashSense **does not currently make a WCAG 2.1 AA conformance claim**. The automated CI gate (R25.1) plus the controls in this document constitute a good-faith effort, not a conformance certification.

---

## 4. Manual contrast verification

The Tailwind tokens in task 6.7 give us 4.5:1 body text and ≥3:1 focus indicators *by construction*. That is the floor, not the ceiling — real displays apply gamma curves, ambient-light sensors, and HDR remappings the design tokens do not see.

### 4.1 Spot-check matrix

| Surface | Target ratio | Tested |
| --- | --- | --- |
| Body text on `bg-slate-950` background | ≥4.5:1 | unowned |
| Card text on `bg-slate-900/70` background | ≥4.5:1 | unowned |
| Status badge text (`MONITORING`, `CRASH DETECTED`) on its own background | ≥4.5:1 | unowned |
| `:focus-visible` ring on every interactive element | ≥3:1 against adjacent colors | unowned |
| Disabled button text and border (e.g. floating Simulate Crash before sensors load) | ≥3:1 (disabled controls are non-text indicators per WCAG 1.4.11) | unowned |
| Connection indicator dot (green/red) against the bar background | ≥3:1 (non-text indicator) | unowned |

### 4.2 Procedure

1. Display the Dashboard at native resolution on each target display.
2. Use a screen-pixel sampler (DigitalColor Meter on macOS, ColorPic on Windows, or Firefox DevTools' Accessibility Inspector) to read the foreground and background colors after rendering.
3. Compute the ratio per WCAG 1.4.3 and 1.4.11.
4. Record below if any ratio comes out below the target on at least one tested display.

Displays to spot-check: built-in MacBook Pro Retina, an external 24" sRGB IPS monitor, a low-end Windows laptop panel, and a recent Android phone (TalkBack target).

---

## 5. Reduced-motion verification

R25.5 says all motion in `prefers-reduced-motion: reduce` collapses to ≤200 ms cross-fades. Task 6.7 implements this in `frontend/src/styles/reduced_motion.css`. This section verifies it in the browser, not just in the stylesheet.

### 5.1 Procedure

1. **Chrome / Edge:** DevTools → Rendering panel → "Emulate CSS media feature `prefers-reduced-motion`" → set to `reduce`.
2. **Firefox:** DevTools → Inspector → Toggle the `prefers-reduced-motion` media query at the top of the rules pane.
3. **Safari:** System Settings → Accessibility → Display → Reduce motion. (Cannot be emulated per-tab.)
4. **VoiceOver / NVDA mobile:** OS-level reduce motion (iOS: Settings → Accessibility → Motion → Reduce Motion; Android: Settings → Accessibility → Remove Animations).

### 5.2 Open work

- [ ] Drone-flight animation collapses to a ≤200 ms fade between origin and destination markers
- [ ] Alert-panel slide is replaced with an opacity cross-fade
- [ ] Wavefront expansion does not animate; static circle at final radius
- [ ] No element exceeds 200 ms of motion when reduced-motion is on, measured by Performance panel
- [ ] Auto-scroll behaviours (cards arriving) use `scroll-behavior: auto`, not `smooth`, when reduced-motion is on

---

## 6. Keyboard-only navigation drill

This is the operational complement to `docs/keyboard_map.md`. The map says what *should* work; this drill says what *did* work, on a real machine, with a real operator, on a real day.

### 6.1 Procedure

Disconnect the pointer (literally — unplug the mouse and trackpad on a laptop, or use a tester who keeps their hands behind their back). Then walk every row of `docs/keyboard_map.md` in order, plus the following flows:

1. **Cold start:** Open the Dashboard, use the floating Simulate Crash button, watch the resulting CrashEvent traverse `DETECTED → DRONE_DISPATCHED → DRONE_ARRIVED`, dismiss the alert panel.
2. **Filter & sort:** From a populated list, narrow to severe-active events and sort by oldest-first. Verify focus stays on a sensible element when results change.
3. **Replay:** From a specific card, trigger Replay and dismiss with Escape. Confirm focus returns to the card.
4. **Mobile drawer:** Reproduce on a 375-px viewport with a hardware keyboard (Bluetooth keyboard on a phone, or DevTools mobile emulation plus a desktop keyboard).
5. **Error recovery:** Simulate `/sensors` returning a 503; confirm the error is reachable by Tab and that the retry path completes without a pointer.

### 6.2 Log

Log entries take the form:

```
[YYYY-MM-DD] [Operator] [Browser/OS] — Flows passed: <list>. Obstacles: <list with element identifier and observed behaviour>.
```

Track obstacles as issues; each obstacle that contradicts `docs/keyboard_map.md` is a R25.3 regression and blocks merge.

---

## 7. Change log

Append-only. Newest first.

```
[2025-01-15] [Frontend team] — Document authored. All sections open work. Automated gates (R25.1, R25.2, R25.4, R25.5) green via tasks 6.6 and 6.7.
```

---

## 8. References

- `.kiro/specs/crashsense-hardening/requirements.md` — R25 (full text), R26.5 (touch target floor), R23 (replay), R24 (filter and sort)
- `.kiro/specs/crashsense-hardening/design.md` §3.4 — Parallel tracks sketch (Accessibility audit)
- `docs/keyboard_map.md` — element-by-element keyboard contract complementing this document
- `frontend/src/__tests__/a11y.spec.jsx` — automated axe-core gate (task 6.6)
- `frontend/src/styles/reduced_motion.css` — reduced-motion implementation (task 6.7)
- WCAG 2.1 AA — https://www.w3.org/TR/WCAG21/
