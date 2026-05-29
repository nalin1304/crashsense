import { useEffect, useRef, useState, useCallback } from 'react';

const MAX_EVENTS = 100;
const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30000;

/**
 * useWebSocket(url)
 *
 * Opens a WebSocket on mount, parses incoming JSON, retains the 100 most
 * recent events, and reconnects with exponential backoff (1s → 30s).
 *
 * Returns: { events, latestEvent, isConnected, sendMessage }.
 */
export default function useWebSocket(url) {
  const [events, setEvents] = useState([]);
  const [latestEvent, setLatestEvent] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  const backoffRef = useRef(INITIAL_BACKOFF_MS);
  const timerRef = useRef(null);
  const closedByCleanupRef = useRef(false);

  const connect = useCallback(() => {
    closedByCleanupRef.current = false;
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      console.warn('WS construct failed', err);
      scheduleReconnect();
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      backoffRef.current = INITIAL_BACKOFF_MS;
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      let parsed;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        // Discard malformed payloads without disturbing state.
        return;
      }
      setEvents((prev) => {
        const next = [...prev, parsed];
        if (next.length > MAX_EVENTS) next.splice(0, next.length - MAX_EVENTS);
        return next;
      });
      setLatestEvent(parsed);
    };

    ws.onerror = () => {
      // onclose will fire next; nothing to do here.
    };

    ws.onclose = () => {
      setIsConnected(false);
      if (!closedByCleanupRef.current) scheduleReconnect();
    };

    function scheduleReconnect() {
      const delay = backoffRef.current;
      backoffRef.current = Math.min(delay * 2, MAX_BACKOFF_MS);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => connect(), delay);
    }
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      closedByCleanupRef.current = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      if (wsRef.current && wsRef.current.readyState <= 1) {
        try { wsRef.current.close(); } catch (_) {}
      }
    };
  }, [connect]);

  const sendMessage = useCallback((data) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    try {
      ws.send(typeof data === 'string' ? data : JSON.stringify(data));
      return true;
    } catch {
      return false;
    }
  }, []);

  const hydrate = useCallback((seedEvents) => {
    if (!Array.isArray(seedEvents) || seedEvents.length === 0) return;
    setEvents((prev) => {
      // Prepend seeds (oldest -> newest) so they appear before any live events.
      // /events/recent returns newest-first, so reverse before inserting.
      const ordered = [...seedEvents].reverse();
      // De-dup against anything already in the live array
      const existingIds = new Set(prev.map((e) => e?.event_id).filter(Boolean));
      const additions = ordered.filter((e) => !existingIds.has(e?.event_id));
      const merged = [...additions, ...prev];
      if (merged.length > MAX_EVENTS) merged.splice(0, merged.length - MAX_EVENTS);
      return merged;
    });
  }, []);

  return { events, latestEvent, isConnected, sendMessage, hydrate };
}
