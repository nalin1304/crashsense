// Backend client for the CrashSense dashboard.

const DEFAULT_BACKEND = 'http://127.0.0.1:8000';

export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || DEFAULT_BACKEND;
export const WS_URL =
  import.meta.env.VITE_WS_URL ||
  BACKEND_URL.replace(/^http/, 'ws') + '/ws/events';

async function jsonOrThrow(res) {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${body ? ` — ${body}` : ''}`);
  }
  return res.json();
}

export async function fetchSensors() {
  const res = await fetch(`${BACKEND_URL}/sensors`);
  return jsonOrThrow(res);
}

export async function fetchRecentEvents(limit = 50) {
  const res = await fetch(`${BACKEND_URL}/events/recent?limit=${limit}`);
  return jsonOrThrow(res);
}

// Fetch a single persisted CrashEvent by id (R23.1, R23.2). Backed by
// `GET /events/{event_id}`, which is sourced from the durable Event_Store.
// Throws an Error whose `.status` is the HTTP status code (404 when the
// id is well-formed but no record exists, per R23.4) so callers can
// surface a non-blocking "event not found" indicator.
export async function fetchEventById(eventId) {
  const res = await fetch(`${BACKEND_URL}/events/${eventId}`);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    const err = new Error(
      `${res.status} ${res.statusText}${body ? ` — ${body}` : ''}`,
    );
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export async function postSimulateCrash(lat, lon) {
  const res = await fetch(`${BACKEND_URL}/simulate-crash`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lat, lon }),
  });
  return jsonOrThrow(res);
}

export async function postDroneArrived(crashEvent) {
  const res = await fetch(`${BACKEND_URL}/drone-arrived`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(crashEvent),
  });
  return jsonOrThrow(res);
}

export async function postDroneDispatched(crashEvent) {
  const res = await fetch(`${BACKEND_URL}/drone-dispatched`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(crashEvent),
  });
  return jsonOrThrow(res);
}
