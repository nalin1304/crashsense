import { useEffect, useRef, useState } from 'react';

const DASHCAM_CLIP_URL = '/dashcam-crash.wav';

// Fallback synth used only if the real clip can't be loaded — keeps the demo
// alive even without an internet connection during the data-fetch step.
function buildSyntheticBuffer(audioCtx, durationS = 6, impactAt = 3.0) {
  const sampleRate = audioCtx.sampleRate;
  const buffer = audioCtx.createBuffer(1, sampleRate * durationS, sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) {
    data[i] = (Math.random() * 2 - 1) * 0.04;
  }
  const impactStart = Math.floor(sampleRate * impactAt);
  const impactLen = Math.floor(sampleRate * 0.25);
  for (let i = 0; i < impactLen && impactStart + i < data.length; i++) {
    const t = i / sampleRate;
    const env = Math.exp(-t * 6);
    const chirp = Math.sin(2 * Math.PI * (200 + 4000 * t) * t);
    data[impactStart + i] += chirp * env * 0.95;
  }
  return buffer;
}

async function loadDashcamBuffer(audioCtx) {
  try {
    const resp = await fetch(DASHCAM_CLIP_URL, { cache: 'force-cache' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const arr = await resp.arrayBuffer();
    return await audioCtx.decodeAudioData(arr);
  } catch (err) {
    console.warn('falling back to synthetic dashcam clip:', err.message);
    return buildSyntheticBuffer(audioCtx);
  }
}

export default function AudioMonitor() {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const sourceRef = useRef(null);
  const analyserRef = useRef(null);
  const rafRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState(null);
  const [usingReal, setUsingReal] = useState(false);

  // Render a gentle idle waveform so the canvas isn't blank before play.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const y = h / 2 + Math.sin(x * 0.05) * 1.2;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, []);

  const draw = () => {
    const analyser = analyserRef.current;
    const canvas = canvasRef.current;
    if (!analyser || !canvas) return;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.fftSize;
    const arr = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(arr);
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1.5;
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, '#22d3ee');
    grad.addColorStop(1, '#a855f7');
    ctx.strokeStyle = grad;
    ctx.beginPath();
    const step = w / bufferLength;
    for (let i = 0; i < bufferLength; i++) {
      const v = arr[i] / 128 - 1;
      const x = i * step;
      const y = h / 2 + v * h * 0.45;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    rafRef.current = requestAnimationFrame(draw);
  };

  const start = async () => {
    try {
      if (!ctxRef.current) {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        ctxRef.current = new Ctx();
      }
      const audioCtx = ctxRef.current;
      if (audioCtx.state === 'suspended') await audioCtx.resume();
      const buffer = await loadDashcamBuffer(audioCtx);
      // Heuristic: real clips are usually ≥ 4 seconds at 44.1 kHz; the synth
      // is 6 s at the AudioContext rate. We just check duration > 0.
      setUsingReal(buffer.duration > 0 && buffer.numberOfChannels > 0 && buffer.length > 0
                  // synth rejects via specific length
                  && (buffer.sampleRate !== audioCtx.sampleRate || buffer.duration !== 6.0));
      const source = audioCtx.createBufferSource();
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      source.buffer = buffer;
      source.connect(analyser);
      analyser.connect(audioCtx.destination);
      source.onended = () => {
        setPlaying(false);
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
      };
      source.start();
      sourceRef.current = source;
      analyserRef.current = analyser;
      setPlaying(true);
      setError(null);
      requestAnimationFrame(draw);
    } catch (err) {
      setError(err.message || 'audio init failed');
    }
  };

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      try { sourceRef.current?.stop(); } catch {}
      try { ctxRef.current?.close(); } catch {}
    };
  }, []);

  return (
    <div className="pointer-events-auto absolute bottom-4 left-1/2 z-20 -translate-x-1/2 rounded-xl border border-slate-700/60 bg-slate-900/85 p-3 shadow-xl backdrop-blur">
      <div className="mb-1 flex items-center gap-3">
        <button
          type="button"
          onClick={start}
          disabled={playing}
          className="cursor-pointer rounded-md border border-cyan-500/40 bg-cyan-500/10 px-3 py-1 text-xs font-semibold text-cyan-200 transition-colors duration-200 hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {playing ? 'Playing dashcam…' : 'Play dashcam clip'}
        </button>
        <span className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
          {usingReal ? 'real Freesound CC-BY clip' : 'waveform · live'}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={420}
        height={64}
        className="block rounded-md bg-slate-950"
      />
      {error && <p className="mt-1 text-[11px] text-rose-300">{error}</p>}
    </div>
  );
}
