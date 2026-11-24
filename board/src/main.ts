import '@fontsource/newsreader/400.css';
import '@fontsource/newsreader/400-italic.css';
import '@fontsource/newsreader/500.css';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/500.css';
import './ui/styles.css';

import { Camera } from './camera';
import type { Msg, Source } from './contract';
import { Dots, Renderer } from './gl/renderer';
import { Scene } from './scene';
import { MockSource } from './sources/mock';
import { ReplaySource } from './sources/replay';
import { WsSource } from './sources/ws';
import { ownerInCell, Store } from './state';
import { BatchStrip } from './ui/batchstrip';
import { Fig2 } from './ui/fig2';
import { Fig4 } from './ui/fig4';
import { Fig5 } from './ui/fig5';
import { Follow } from './ui/follow';
import { Inspector } from './ui/inspector';
import { Overlay } from './ui/overlay';
import { TimeCtl } from './ui/timectl';

const params = new URLSearchParams(location.search);
const store = new Store();
const canvas = document.getElementById('plate') as HTMLCanvasElement;
const root = document.getElementById('overlay')!;
const renderer = new Renderer(canvas);
const scene = new Scene(store);
const camera = new Camera();
const overlay = new Overlay(root, store, scene);
overlay.setCamera(camera);
const dots = new Dots();
const follow = new Follow(root, store, scene);
const fig2 = new Fig2(root, store, (rid) => follow.followReq(rid, performance.now()));
const inspector = new Inspector(root, store, scene, follow);
const time = new TimeCtl(root, store, scene);
const strip = new BatchStrip(root, store, scene);
const fig4 = new Fig4(root, store, () => String(store.hello?.model ?? ''));
// the middle slot: expert routing for MoE engines, the prefix cache otherwise
const fig5 = new Fig5(root, store, (rid) => follow.followReq(rid, performance.now()));
let moeSlot = false;
// expert routing is shown when the engine says it reports it, or when routing data has actually arrived
const hasMoe = () => !!store.hello?.coverage?.moe || store.moeAt > 0;
time.resetBtn.addEventListener('click', () => camera.reset());

// Events are buffered by the time controller and, while live, applied once per frame:
// high-rate streams (engine.step ~200/s) cost one pass per frame, not one render per event.
let inbox: Msg[] = [];
const emit = (m: Msg) => {
  if (time.push(m, performance.now())) inbox.push(m);
};
let wasDown = false;
const status = (s: string) => {
  if (s.startsWith('disconnected')) wasDown = true;
  if (s === 'connected' && wasDown) {
    wasDown = false;
    store.markLost();
  }
  store.status = s;
};

// A public build (VITE_PUBLIC_WS set at build time) shows only its own feed and recordings from its own site:
// links cannot point it at another server with ?ws= or ?replay=.
const PUBLIC_WS: string | undefined = import.meta.env.VITE_PUBLIC_WS || undefined;
const PUBLIC_FALLBACK = 'samples/sglang-copilot-agent.jsonl';
const sameOrigin = (u: string) => {
  try {
    return new URL(u, location.href).origin === location.origin;
  } catch {
    return false;
  }
};

function pickSource(): Source {
  const mock = params.get('mock');
  if (mock) return new MockSource((['steady', 'burst', 'pd', 'dp', 'rep'].includes(mock) ? mock : 'steady') as 'steady');
  const replay = params.get('replay');
  if (replay && (!PUBLIC_WS || sameOrigin(replay))) return new ReplaySource(replay, Number(params.get('speed') ?? 1) || 1, params.get('loop') === '1');
  return new WsSource(PUBLIC_WS ?? params.get('ws') ?? 'ws://127.0.0.1:18200/ws');
}

let source = pickSource();
store.sourceLabel = source.label;
if (!(source instanceof WsSource)) time.headLabel = 'latest';
source.start(emit, status);

// Live is the default; if the socket never connects, fall back so the plate isn't empty: a public build plays its
// bundled recording, a local one the labelled mock.
if (source instanceof WsSource && (PUBLIC_WS || !params.has('ws')) && params.get('fallback') !== '0') {
  window.setTimeout(() => {
    if (store.events === 0 && inbox.length === 0 && store.status !== 'connected') {
      source.stop();
      source = PUBLIC_WS ? new ReplaySource(PUBLIC_FALLBACK, 1, true) : new MockSource('steady');
      store.sourceLabel = PUBLIC_WS ? source.label : `${source.label} (live socket unavailable)`;
      if (PUBLIC_WS) store.sourceNote = 'live feed busy or offline · reload to retry';
      time.headLabel = 'latest';
      source.start(emit, status);
    }
  }, 4000);
}

function relayout(): void {
  renderer.resize();
  const W = renderer.cssW;
  const H = renderer.cssH;
  // Three zones: header (title, readout), the plate, and a figure band at the bottom (Fig. 3 · 4 · 2).
  // The band is reserved space: figures never sit on top of the specimen.
  const notesW = W >= 1100 ? Math.min(460, Math.max(330, Math.round(W * 0.25))) : 0;
  const showBand = W - notesW > 900 && H > 600;
  const band = showBand ? Math.round(Math.min(280, Math.max(190, H * 0.24))) : 0;
  scene.layout(W, H, band);
  const L = scene.L;
  overlay.layout(L);
  const pad = 20;
  const gap = 14;
  const top = H - band + 6;
  const h = band - 16;
  const avail = L.PW - 2 * pad - 2 * gap;
  const w3 = Math.round(avail * 0.32);
  const w4 = Math.round(avail * 0.38);
  const w2 = avail - w3 - w4;
  strip.setRect(pad, top, w3, h, showBand);
  moeSlot = hasMoe();
  fig4.setRect(pad + w3 + gap, top, w4, h, showBand && moeSlot);
  fig5.setRect(pad + w3 + gap, top, w4, h, showBand && !moeSlot);
  fig2.setRect(pad + w3 + w4 + 2 * gap, top, w2, h, showBand);
  // one line above the band: "?" then the follow narration
  const lineY = (showBand ? top : H) - 30;
  overlay.placeKeyBtn(pad + 4, lineY);
  follow.placeLine(pad + 36, lineY, L.PW - 2 * pad - 36);
  time.place(L.notesW + 36, 100);
  // inspector: docked over the notes column when there is one, so it never covers the plate
  if (L.notesW) inspector.place(12, 14, L.notesW - 24, H - 28);
  else inspector.place(16, 150, 340, H - 170);
  camera.version++;
}
relayout();
window.addEventListener('resize', relayout);

// ---- input: zoom, pan, pinch, click to inspect, double-click to fly --------------------------------
canvas.addEventListener(
  'wheel',
  (ev) => {
    ev.preventDefault();
    const f = Math.exp(-ev.deltaY * (ev.ctrlKey ? 0.012 : 0.0018)); // ctrl+wheel = trackpad pinch
    camera.zoomAt(ev.clientX, ev.clientY, f);
  },
  { passive: false },
);

const pointers = new Map<number, { x: number; y: number }>();
let drag: { x0: number; y0: number; moved: boolean } | null = null;
let pinch: { d: number; mx: number; my: number } | null = null;
const pinchState = () => {
  const [a, b] = [...pointers.values()];
  return { d: Math.hypot(a.x - b.x, a.y - b.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
};

canvas.addEventListener('pointerdown', (ev) => {
  canvas.setPointerCapture(ev.pointerId);
  pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
  if (pointers.size === 1) drag = { x0: ev.clientX, y0: ev.clientY, moved: false };
  if (pointers.size === 2) {
    pinch = pinchState();
    if (drag) drag.moved = true;
  }
});

canvas.addEventListener('pointermove', (ev) => {
  const p = pointers.get(ev.pointerId);
  if (!p) {
    const [wx, wy] = camera.toWorld(ev.clientX, ev.clientY);
    const hit = scene.componentAt(wx, wy);
    let req = hit?.kind === 'request' ? (store.reqs.get(hit.id) ?? null) : scene.pick(wx, wy);
    if (hit?.kind === 'cell') {
      // memory map: the tooltip shows the cell's owner request
      const [bi, bin] = hit.id.split(':').map(Number);
      const m = scene.mapFor(bi);
      if (m?.levels) {
        const o = m.owners.find((ow) => ownerInCell(ow, bin));
        req = o ? (store.reqs.get(o.rid) ?? null) : null;
      } else {
        const st = m?.state[bin] ?? -1;
        const k = st >= 0 ? st : st === -3 || st === -4 ? (m?.shared.get(bin)?.[0] ?? -1) : -1;
        req = k >= 0 && m ? (store.reqs.get(m.owners[k].rid) ?? null) : null;
      }
    }
    overlay.setHover(req, ev.clientX, ev.clientY);
    canvas.classList.toggle('over', !!hit);
    return;
  }
  const dx = ev.clientX - p.x;
  const dy = ev.clientY - p.y;
  p.x = ev.clientX;
  p.y = ev.clientY;
  if (pointers.size === 2 && pinch) {
    const n = pinchState();
    camera.zoomAt(n.mx, n.my, n.d / Math.max(1, pinch.d));
    camera.panBy(n.mx - pinch.mx, n.my - pinch.my);
    pinch = n;
  } else if (drag) {
    if (!drag.moved && Math.hypot(ev.clientX - drag.x0, ev.clientY - drag.y0) > 4) {
      drag.moved = true;
      canvas.classList.add('dragging');
      overlay.setHover(null, 0, 0);
    }
    if (drag.moved) camera.panBy(dx, dy);
  }
});

const endPointer = (ev: PointerEvent) => {
  const wasClick = drag && !drag.moved && pointers.size === 1;
  pointers.delete(ev.pointerId);
  if (wasClick) {
    const [wx, wy] = camera.toWorld(ev.clientX, ev.clientY);
    const hit = scene.componentAt(wx, wy);
    if (hit) inspector.show(hit);
  }
  if (pointers.size < 2) pinch = null;
  if (pointers.size === 0) {
    drag = null;
    canvas.classList.remove('dragging');
  }
};
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('mouseleave', () => overlay.setHover(null, 0, 0));
canvas.addEventListener('dblclick', (ev) => {
  const [wx, wy] = camera.toWorld(ev.clientX, ev.clientY);
  const hit = scene.componentAt(wx, wy);
  if (hit) camera.flyTo(hit.box, scene.L.PW, scene.L.H);
});

window.addEventListener('keydown', (ev) => {
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const tag = (ev.target as HTMLElement)?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  if (ev.key === 'f' || ev.key === 'F') follow.toggle(performance.now());
  else if (ev.key === ' ') {
    ev.preventDefault();
    time.toggle(performance.now());
  } else if (ev.key === 'Escape') {
    if (inspector.open) inspector.close();
    else camera.reset();
  } else if (ev.key === 'm' || ev.key === 'M') scene.memMap = !scene.memMap;
  else if (ev.key === '0') camera.reset();
  else if (ev.key === '+' || ev.key === '=') camera.zoomAt(scene.L.PW / 2, scene.L.H / 2, 1.4);
  else if (ev.key === '-') camera.zoomAt(scene.L.PW / 2, scene.L.H / 2, 1 / 1.4);
});

// ---- frame loop --------------------------------------------------------------------------------------
const perf = { frames: 0, frameMs: 0, dots: 0, worst: 0 };
(window as unknown as { __board: unknown }).__board = { store, scene, perf, camera, time, inspector, follow };

let last = performance.now();
let lastUi = 0;
let lastError = '';
function frame(now: number): void {
  requestAnimationFrame(frame); // schedule first: an exception below must never stop the plate
  try {
    step(now);
  } catch (e) {
    const msg = String((e as Error)?.stack ?? e);
    if (msg !== lastError) console.error('[board] frame error', e);
    lastError = msg;
  }
}

function step(now: number): void {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  const t0 = performance.now();
  camera.tick(now);
  if (time.mode === 'live' && inbox.length) {
    const batch = inbox;
    inbox = [];
    for (const m of batch) store.ingest(m, now);
  } else if (time.mode !== 'live') inbox = [];
  time.tick(now, dt);
  const frozen = time.frozen;
  if (!frozen) store.tick(now);
  scene.setView(camera.k, camera.viewBox(renderer.cssW, renderer.cssH));
  scene.update(frozen ? 0 : dt, now);
  dots.view(camera.k, camera.tx, camera.ty, renderer.cssW, renderer.cssH);
  scene.draw(dots, now);
  renderer.cam = [camera.k, camera.tx, camera.ty];
  renderer.render(dots);
  overlay.reposition();
  root.classList.toggle('moved', !camera.isHome);
  if (now - lastUi > 200) {
    lastUi = now;
    overlay.update(now);
    follow.tick(now);
    fig2.update(now, scene.follow);
    if (hasMoe() !== moeSlot) relayout();
    fig4.update(now);
    fig5.update(now);
  }
  inspector.tick(now);
  strip.update(now);
  const ms = performance.now() - t0;
  perf.frames++;
  perf.frameMs = perf.frameMs * 0.95 + ms * 0.05;
  perf.worst = Math.max(perf.worst * 0.999, ms);
  perf.dots = dots.n;
}
requestAnimationFrame(frame);
