"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Song = {
  title: string;
  x: number;
  y: number;
  cluster: number;
  label: string;
  preview_url: string;
};

type Transform = { scale: number; tx: number; ty: number };
type Vec2 = { x: number; y: number };

// Validated categorical slots (light surface) from the design system's palette —
// blue / yellow / violet, fixed order, never cycled. Chosen over the default
// blue/aqua/yellow ordering for max protanopia separation: worst adjacent-pair
// ΔE 123.7 (protan) vs 47.2 for the default trio — well clear of the 12 floor.
const CLUSTER_COLORS = ["#2a78d6", "#eda100", "#4a3aa7"];
const DOT_STROKE = "#0b0b0b";

const PADDING_FRACTION = 0.12;
const MIN_ZOOM = 0.6;
const MAX_ZOOM = 16;
const HIT_RADIUS = 16;
const DOT_RADIUS = 6.5;
const HOVER_RADIUS = 10;

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v));
}

function centerOf(size: { width: number; height: number }): Vec2 {
  return { x: size.width / 2, y: size.height / 2 };
}

function computeBaseTransform(songs: Song[], width: number, height: number): Transform {
  if (!songs.length || width <= 0 || height <= 0) {
    return { scale: 1, tx: width / 2, ty: height / 2 };
  }
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const s of songs) {
    if (s.x < minX) minX = s.x;
    if (s.x > maxX) maxX = s.x;
    if (s.y < minY) minY = s.y;
    if (s.y > maxY) maxY = s.y;
  }
  const dataWidth = Math.max(maxX - minX, 1e-6);
  const dataHeight = Math.max(maxY - minY, 1e-6);
  const availW = width * (1 - PADDING_FRACTION * 2);
  const availH = height * (1 - PADDING_FRACTION * 2);
  const scale = Math.min(availW / dataWidth, availH / dataHeight);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return { scale, tx: width / 2 - cx * scale, ty: height / 2 - cy * scale };
}

function getEffectiveTransform(base: Transform, zoom: number, pan: Vec2, center: Vec2): Transform {
  const scale = base.scale * zoom;
  const tx = (base.tx - center.x) * zoom + center.x + pan.x;
  const ty = (base.ty - center.y) * zoom + center.y + pan.y;
  return { scale, tx, ty };
}

function hitTest(mx: number, my: number, songs: Song[], eff: Transform): number {
  let best = -1;
  let bestDist = HIT_RADIUS;
  for (let i = 0; i < songs.length; i++) {
    const sx = songs[i].x * eff.scale + eff.tx;
    const sy = songs[i].y * eff.scale + eff.ty;
    const d = Math.hypot(sx - mx, sy - my);
    if (d <= bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

function drawDot(
  ctx: CanvasRenderingContext2D,
  song: Song,
  eff: Transform,
  color: string,
  hovered: boolean,
  playing: boolean
) {
  const sx = song.x * eff.scale + eff.tx;
  const sy = song.y * eff.scale + eff.ty;
  const r = playing ? HOVER_RADIUS + 1 : hovered ? HOVER_RADIUS : DOT_RADIUS;

  if (playing) {
    ctx.beginPath();
    ctx.arc(sx, sy, r + 6, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  ctx.beginPath();
  ctx.arc(sx, sy, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.globalAlpha = hovered || playing ? 1 : 0.92;
  ctx.fill();
  ctx.globalAlpha = 1;

  if (hovered || playing) {
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = DOT_STROKE;
    ctx.globalAlpha = 0.85;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
}

export default function TasteMap() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const dataRef = useRef<Song[]>([]);
  const clusterColorRef = useRef<Map<number, string>>(new Map());
  const baseTransformRef = useRef<Transform>({ scale: 1, tx: 0, ty: 0 });
  const zoomRef = useRef(1);
  const panRef = useRef<Vec2>({ x: 0, y: 0 });
  const sizeRef = useRef({ width: 0, height: 0, dpr: 1 });

  const draggingRef = useRef(false);
  const dragMovedRef = useRef(false);
  const lastPosRef = useRef<Vec2>({ x: 0, y: 0 });
  const hoverIndexRef = useRef<number | null>(null);
  const playingIndexRef = useRef<number | null>(null);

  const [loaded, setLoaded] = useState(false);
  const [songCount, setSongCount] = useState(0);
  const [legend, setLegend] = useState<{ cluster: number; label: string; color: string }[]>([]);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; title: string } | null>(null);
  const [nowPlaying, setNowPlaying] = useState<string | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const { width, height, dpr } = sizeRef.current;
    if (width <= 0 || height <= 0) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const bg = ctx.createRadialGradient(
      width / 2,
      height / 2,
      0,
      width / 2,
      height / 2,
      Math.max(width, height) * 0.7
    );
    bg.addColorStop(0, "#ffffff");
    bg.addColorStop(1, "#f2f1ec");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    const songs = dataRef.current;
    const eff = getEffectiveTransform(baseTransformRef.current, zoomRef.current, panRef.current, centerOf(sizeRef.current));
    const hoverIndex = hoverIndexRef.current;
    const playingIndex = playingIndexRef.current;

    for (let i = 0; i < songs.length; i++) {
      if (i === hoverIndex || i === playingIndex) continue;
      const color = clusterColorRef.current.get(songs[i].cluster) ?? "#898781";
      drawDot(ctx, songs[i], eff, color, false, false);
    }
    if (hoverIndex !== null && hoverIndex !== playingIndex) {
      const song = songs[hoverIndex];
      drawDot(ctx, song, eff, clusterColorRef.current.get(song.cluster) ?? "#898781", true, false);
    }
    if (playingIndex !== null) {
      const song = songs[playingIndex];
      drawDot(
        ctx,
        song,
        eff,
        clusterColorRef.current.get(song.cluster) ?? "#898781",
        playingIndex === hoverIndex,
        true
      );
    }
  }, []);

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    playingIndexRef.current = null;
    setNowPlaying(null);
    draw();
  }, [draw]);

  const playSong = useCallback(
    (index: number) => {
      const song = dataRef.current[index];
      if (!song) return;
      if (audioRef.current) {
        audioRef.current.pause();
      }
      const audio = new Audio(song.preview_url);
      audioRef.current = audio;
      playingIndexRef.current = index;
      setNowPlaying(song.title);
      audio.addEventListener("ended", () => {
        if (playingIndexRef.current === index) {
          playingIndexRef.current = null;
          setNowPlaying(null);
          draw();
        }
      });
      audio.play().catch(() => {
        if (playingIndexRef.current === index) {
          playingIndexRef.current = null;
          setNowPlaying(null);
        }
        draw();
      });
      draw();
    },
    [draw]
  );

  // load data
  useEffect(() => {
    let cancelled = false;
    fetch("/taste_map.json")
      .then((r) => r.json())
      .then((songs: Song[]) => {
        if (cancelled) return;
        dataRef.current = songs;

        const byCluster = new Map<number, string>();
        for (const s of songs) {
          if (!byCluster.has(s.cluster)) byCluster.set(s.cluster, s.label);
        }
        const sortedClusters = [...byCluster.entries()].sort((a, b) => a[0] - b[0]);
        const colorMap = new Map<number, string>();
        sortedClusters.forEach(([cluster], i) => colorMap.set(cluster, CLUSTER_COLORS[i % CLUSTER_COLORS.length]));
        clusterColorRef.current = colorMap;

        setLegend(sortedClusters.map(([cluster, label]) => ({ cluster, label, color: colorMap.get(cluster)! })));
        setSongCount(songs.length);

        const { width, height } = sizeRef.current;
        baseTransformRef.current = computeBaseTransform(songs, width, height);
        zoomRef.current = 1;
        panRef.current = { x: 0, y: 0 };
        setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    draw();
  }, [loaded, draw]);

  // resize handling
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      const { width, height } = entry.contentRect;
      const dpr = window.devicePixelRatio || 1;
      sizeRef.current = { width, height, dpr };
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      if (dataRef.current.length) {
        baseTransformRef.current = computeBaseTransform(dataRef.current, width, height);
      }
      draw();
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, [draw]);

  // wheel zoom (native listener so preventDefault works)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const center = centerOf(sizeRef.current);
      const base = baseTransformRef.current;
      const zoom = zoomRef.current;
      const pan = panRef.current;
      const eff = getEffectiveTransform(base, zoom, pan, center);

      const wx = (mx - eff.tx) / eff.scale;
      const wy = (my - eff.ty) / eff.scale;
      const factor = Math.exp(-e.deltaY * 0.0015);
      const newZoom = clamp(zoom * factor, MIN_ZOOM, MAX_ZOOM);
      const newScale = base.scale * newZoom;
      const newTx = mx - wx * newScale;
      const newTy = my - wy * newScale;

      zoomRef.current = newZoom;
      panRef.current = {
        x: newTx - (base.tx - center.x) * newZoom - center.x,
        y: newTy - (base.ty - center.y) * newZoom - center.y,
      };
      draw();
    };

    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [draw]);

  // safety net: release drag state if mouseup happens outside the canvas
  useEffect(() => {
    const onWindowMouseUp = () => {
      draggingRef.current = false;
    };
    window.addEventListener("mouseup", onWindowMouseUp);
    return () => window.removeEventListener("mouseup", onWindowMouseUp);
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    draggingRef.current = true;
    dragMovedRef.current = false;
    lastPosRef.current = { x: e.clientX, y: e.clientY };
  }, []);

  const onMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      if (draggingRef.current) {
        const dx = e.clientX - lastPosRef.current.x;
        const dy = e.clientY - lastPosRef.current.y;
        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) dragMovedRef.current = true;
        panRef.current = { x: panRef.current.x + dx, y: panRef.current.y + dy };
        lastPosRef.current = { x: e.clientX, y: e.clientY };
        canvas.style.cursor = "grabbing";
        setTooltip(null);
        draw();
        return;
      }

      const eff = getEffectiveTransform(baseTransformRef.current, zoomRef.current, panRef.current, centerOf(sizeRef.current));
      const idx = hitTest(mx, my, dataRef.current, eff);
      if (idx !== hoverIndexRef.current) {
        hoverIndexRef.current = idx >= 0 ? idx : null;
        draw();
      }
      if (idx >= 0) {
        setTooltip({ x: mx, y: my, title: dataRef.current[idx].title });
        canvas.style.cursor = "pointer";
      } else {
        setTooltip(null);
        canvas.style.cursor = "grab";
      }
    },
    [draw]
  );

  const onMouseUp = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const wasDragging = draggingRef.current;
      draggingRef.current = false;
      const moved = dragMovedRef.current;
      dragMovedRef.current = false;
      if (!wasDragging || moved) return;

      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const eff = getEffectiveTransform(baseTransformRef.current, zoomRef.current, panRef.current, centerOf(sizeRef.current));
      const idx = hitTest(mx, my, dataRef.current, eff);
      if (idx >= 0) {
        playSong(idx);
      }
    },
    [playSong]
  );

  const onMouseLeave = useCallback(() => {
    if (hoverIndexRef.current !== null) {
      hoverIndexRef.current = null;
      draw();
    }
    setTooltip(null);
  }, [draw]);

  const resetView = useCallback(() => {
    zoomRef.current = 1;
    panRef.current = { x: 0, y: 0 };
    draw();
  }, [draw]);

  return (
    <div className="relative flex h-dvh w-full flex-col overflow-hidden bg-[#f9f9f7] text-[#0b0b0b]">
      <header className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-start justify-between p-5 sm:p-7">
        <div>
          <h1 className="text-sm font-medium tracking-wide text-[#0b0b0b]">Taste Map</h1>
          <p className="mt-1 text-xs text-[#898781]">
            {songCount || "…"} songs · placed by sound similarity
          </p>
        </div>
        <div className="pointer-events-auto flex flex-col items-end gap-2">
          <button
            type="button"
            onClick={resetView}
            className="rounded-full border border-black/10 bg-black/5 px-3 py-1.5 text-xs text-[#52514e] backdrop-blur transition hover:bg-black/10 hover:text-[#0b0b0b]"
          >
            Reset view
          </button>
          <p className="hidden text-right text-[11px] text-[#898781] sm:block">
            scroll to zoom · drag to pan · click a dot to play
          </p>
        </div>
      </header>

      <div ref={containerRef} className="relative h-full w-full">
        <canvas
          ref={canvasRef}
          className="block h-full w-full cursor-grab"
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseLeave}
        />
        {!loaded && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-[#898781]">
            Loading songs…
          </div>
        )}
        {tooltip && (
          <div
            className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-[calc(100%+10px)] whitespace-nowrap rounded-md border border-black/10 bg-[#fcfcfb]/95 px-2.5 py-1.5 text-xs text-[#0b0b0b] shadow-lg backdrop-blur"
            style={{ left: tooltip.x, top: tooltip.y }}
          >
            {tooltip.title}
          </div>
        )}
      </div>

      <footer className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex items-end justify-between gap-4 p-5 sm:p-7">
        <div className="pointer-events-auto flex flex-col gap-1.5 rounded-xl border border-black/10 bg-black/5 p-3 backdrop-blur">
          {legend.map((l) => (
            <div key={l.cluster} className="flex items-center gap-2">
              <span className="h-3 w-3 shrink-0 rounded-full" style={{ backgroundColor: l.color }} />
              <span className="text-xs text-[#52514e]">{l.label}</span>
            </div>
          ))}
        </div>

        {nowPlaying && (
          <div className="pointer-events-auto flex items-center gap-3 rounded-full border border-black/10 bg-black/5 px-4 py-2 backdrop-blur">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-black/40" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-[#0b0b0b]" />
            </span>
            <span className="max-w-[40vw] truncate text-xs text-[#0b0b0b]/90">{nowPlaying}</span>
            <button
              type="button"
              onClick={stopAudio}
              aria-label="Stop playback"
              className="rounded-full p-1 text-[#52514e] transition hover:text-[#0b0b0b]"
            >
              <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor">
                <rect width="10" height="10" rx="1.5" />
              </svg>
            </button>
          </div>
        )}
      </footer>
    </div>
  );
}
