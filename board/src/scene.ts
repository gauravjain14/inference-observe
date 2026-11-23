// The specimen: requests as cells, KV cache as ultramarine tissue, decode as crimson
// slime-mold veins, GPUs as sap-green neurons. Every moving mark is driven by an event
// or a measured value from the Store; ambient drift is only the medium.
import { Dots } from './gl/renderer';
import { classLabel, ownerInCell, trafficClass, type Instance, type KvCapacity, type KvMap, type KvOwner, type Req, type Store } from './state';
import { INK, clamp, classInk, gauss, h1, hashStr, lerp, mixInk, mulberry32, smooth, type Ink } from './util';

export interface Hit {
  kind: 'request' | 'core' | 'body' | 'expert' | 'layer' | 'moe' | 'branch' | 'trunk' | 'gpu' | 'cell';
  id: string;
  box: { x0: number; y0: number; x1: number; y1: number };
}

export interface Layout {
  W: number;
  H: number;
  PW: number; // plate width (left of the notes column)
  notesW: number;
  memX: number;
  memTop: number;
  memBot: number;
  kv: { cx: number; cy: number; rx: number; ry: number };
  exit: { x: number; y: number };
  net: { x0: number; x1: number; y0: number; y1: number };
  moe: { cx: number; cy: number; r: number };
  organY: number;
  organX0: number;
  organX1: number;
}

/** Plate geometry. `band` is the height reserved at the bottom for the figure band (Fig. 3/4/2):
 *  everything drawn on the plate stays above it, so figures never cover the specimen. */
export function computeLayout(W: number, H: number, band = 0): Layout {
  const notesW = W >= 1100 ? clamp(Math.round(W * 0.25), 330, 460) : 0;
  const PW = W - notesW;
  const Hp = H - band; // plate height
  const memTop = Math.max(212, Hp * 0.28);
  const memBot = Hp - Math.max(128, Hp * 0.19); // room below for the GPU neurons and their labels
  const kvCy = (memTop + memBot) / 2 - Hp * 0.01;
  return {
    W, H, PW, notesW,
    memX: PW * 0.215,
    memTop, memBot,
    kv: { cx: PW * 0.47, cy: kvCy, rx: PW * 0.135, ry: (memBot - memTop) * 0.4 },
    exit: { x: PW * 0.915, y: kvCy },
    net: { x0: PW * 0.645, x1: PW * 0.865, y0: memTop + 10, y1: memBot - 30 },
    moe: { cx: -1e4, cy: -1e4, r: 0 }, // expert routing moved to Fig. 4 in the band
    organY: memBot + 36, // neuron + its two-line label end ~40 px above the plate's bottom edge
    organX0: PW * 0.33,
    organX1: PW * 0.52,
  };
}

interface Poly {
  x: Float32Array;
  y: Float32Array;
  cum: Float32Array;
  len: number;
}

function polyFrom(pts: [number, number][]): Poly {
  const n = pts.length;
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const cum = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = pts[i][0];
    y[i] = pts[i][1];
    if (i) cum[i] = cum[i - 1] + Math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]);
  }
  return { x, y, cum, len: cum[n - 1] };
}

function bezierPts(ax: number, ay: number, bx: number, by: number, bend: number, n: number): [number, number][] {
  const mx = (ax + bx) / 2;
  const my = (ay + by) / 2;
  const dx = bx - ax;
  const dy = by - ay;
  const cx = mx - dy * bend;
  const cy = my + dx * bend;
  const out: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const u = 1 - t;
    out.push([u * u * ax + 2 * u * t * cx + t * t * bx, u * u * ay + 2 * u * t * cy + t * t * by]);
  }
  return out;
}

/** Point and unit normal at arc length s along a polyline. */
function at(p: Poly, s: number, out: { x: number; y: number; nx: number; ny: number }): void {
  if (s <= 0) s = 0;
  if (s >= p.len) s = p.len - 1e-3;
  let lo = 0;
  let hi = p.cum.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (p.cum[mid] <= s) lo = mid;
    else hi = mid;
  }
  const seg = p.cum[hi] - p.cum[lo] || 1;
  const t = (s - p.cum[lo]) / seg;
  const dx = p.x[hi] - p.x[lo];
  const dy = p.y[hi] - p.y[lo];
  const l = Math.hypot(dx, dy) || 1;
  out.x = p.x[lo] + dx * t;
  out.y = p.y[lo] + dy * t;
  out.nx = -dy / l;
  out.ny = dx / l;
}

/** A request's way out: twig -> traffic-class branch -> instance trunk -> outlet duct. */
interface Route {
  segs: (Poly | null)[];
}

function routeLen(r: Route): number {
  let n = 0;
  for (const p of r.segs) if (p) n += p.len;
  return n;
}

function atRoute(r: Route, s: number, out: { x: number; y: number; nx: number; ny: number }): void {
  let last: Poly | null = null;
  for (const p of r.segs) {
    if (!p) continue;
    last = p;
    if (s <= p.len) {
      at(p, s, out);
      return;
    }
    s -= p.len;
  }
  if (last) at(last, last.len, out);
}

/** A junction of the decode network. Branch = traffic class within an instance; trunk = instance. */
interface Junction {
  key: string;
  kind: 'branch' | 'trunk';
  name: string; // class or instance name
  parent: string | null; // trunk key for branches
  x: number;
  y: number;
  tx: number;
  ty: number;
  flux: number; // smoothed tokens/s through its edge
  target: number;
  edge: Poly | null; // to its parent (trunk) or to the aperture
  ex: number; // endpoints the edge was built for
  ey: number;
  px: number;
  py: number;
  seed: number;
  idle: number;
}

const CLASS_ORDER = ['chat', 'agentic', 'rag', 'burst', 'spec', 'short', 'medium', 'long'];

interface Spark {
  rid: string;
  route: Route;
  s: number;
  v: number;
  ink: Ink;
  size: number;
}

interface Spore {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  born: number;
  life: number;
  ink: Ink;
  seed: number;
}

interface Body {
  name: string;
  role: string;
  url: string | null;
  cap: KvCapacity;
  dpb0: number; // dots per KV block at zoom 1
  dpb: number; // dots per KV block at the current zoom (more detail when zoomed in)
  sp: number; // sunflower spacing: colony radius = sp * sqrt(dots)
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  shown: number; // smoothed fill fraction
}

/** Blocks that outlive their request: reusable by the prefix cache until evicted. */
interface Colony {
  x: number;
  y: number;
  blocks: number;
  seed: number;
  born: number;
  life: number;
  body: number;
}

interface FamCore {
  id: string;
  x: number;
  y: number;
  body: number;
  blocks: number;
  tokens: number; // shared tokens, each once
  members: number; // running requests that hold them now
  measured: boolean; // from the engine's KV map (slot-exact); otherwise estimated from cached-token counts
  seed: number;
  spin: number; // next attachment angle
  lastActive: number;
}

/** Memory-map cell layout for one body: bins fill the ellipse row by row (contiguous ranges read as runs). */
interface CellGeo {
  key: string;
  c: number; // cell pitch (world px)
  x: Float32Array;
  y: Float32Array;
  rows: { y: number; x0: number; start: number; n: number }[];
}

interface Transfer {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  bend: number;
  t: number;
  dur: number;
}

interface Organ {
  x: number;
  y: number;
  seed: number;
  dendrites: Poly[];
  tips: [number, number][];
  util: number; // smoothed 0..1
  emit: number; // fractional signals to emit
}

class Entity {
  req: Req;
  seed: number;
  x: number;
  y: number;
  vx = 0;
  vy = 0;
  r: number;
  qs = 1; // shrink factor while crowded in a queue
  inside = false; // crossed the membrane
  dissolve = 0; // 0 = cell, 1 = fully KV tissue
  px = 0; // patch centre
  py = 0;
  patch = false;
  body = -1; // index into Scene.bodies
  arrive = 1; // 0..1 while KV arrives from a prefill worker
  oldPx = 0; // prefill-side patch, fading after handover
  oldPy = 0;
  oldPr = 0;
  oldFade = 0;
  pr = 0; // colony radius
  tx = 0; // colony target centre (colonies glide when they attach to a family core)
  ty = 0;
  blocks = 0;
  dots = 0;
  innerDots = 0; // cached prefix (or a missed duplicate core)
  promptDots = 0;
  shared = 0; // blocks held by its shared core instead
  coreId: string | null = null; // the shared core it reads, when two or more running requests hold the same slots
  oldBlocks = 0;
  oldBody = 0;
  cachedOut = false;
  flash: { kind: 'hit' | 'miss'; t: number } | null = null;
  route: Route = { segs: [] };
  capillary: Poly | null = null; // twig: patch -> branch root
  branch: string | null = null;
  twigKey = '';
  flux = 0; // tokens/s, smoothed
  sparkQ = 0; // tokens received but not yet drawn as sparks
  fade = 0; // 0..1 after done
  spored = false;
  rnd: () => number;

  constructor(req: Req, x: number, y: number) {
    this.req = req;
    this.seed = hashStr(req.rid);
    this.rnd = mulberry32(this.seed);
    this.x = x;
    this.y = y;
    const toks = promptTokens(req);
    this.r = clamp(6 + Math.sqrt(toks) * 0.3, 7, 22);
  }
}

function promptTokens(r: Req): number {
  if (r.promptTokens) return r.promptTokens;
  if (r.promptChars) return r.promptChars / 3.8;
  return 200;
}

const TMP = { x: 0, y: 0, nx: 0, ny: 0 };
const MAX_SPARKS = 9000;

export class Scene {
  L!: Layout;
  ents = new Map<string, Entity>();
  trunks = new Map<string, Junction>();
  branches = new Map<string, Junction>();
  duct: Poly | null = null;
  aperture = 0; // smoothed total tokens/s leaving
  follow: string | null = null; // request being narrated
  highlight: { rids: Set<string>; until: number } | null = null; // requests of a clicked scheduler step
  famCores = new Map<string, FamCore>();
  private ownerIdx = new WeakMap<KvMap, Map<string, KvOwner>>();
  private discs: [number, number, number, number][] = [];
  private cent: { m: KvMap; g: CellGeo; map: Map<string, [number, number]> } | null = null; // x, y, r, seed of the colony (-1 for a core)
  cachedCol: Colony[] = [];
  flecks: { x: number; y: number; vx: number; vy: number; t: number }[] = [];
  kvRead = 0; // KV tokens read per second by decode steps (Σ context × steps/s)
  moeShown: number[] = [];
  zoom = 1;
  view = { x0: -1e9, y0: -1e9, x1: 1e9, y1: 1e9 };
  crossings: { y: number; t: number }[] = []; // admissions, where they crossed the membrane
  memMap = false; // KV body drawn as the engine's real block layout (kv.map) instead of the colony layout
  private geo = new Map<number, CellGeo>();
  mapCentroid = new Map<string, [number, number]>(); // rid -> centre of its cells (memory-map mode)
  pressing: { y: number; r: number }[] = []; // requests held at the proxy, pressing on it
  sparks: Spark[] = [];
  spores: Spore[] = [];
  organs: Organ[] = [];
  signals: { path: Poly; s: number; v: number }[] = [];
  wash: { x: Float32Array; y: Float32Array; sz: Float32Array } | null = null; // unit-disk template
  bodies: Body[] = [];
  pd = false;
  transfers: Transfer[] = [];
  transferRate = 0;
  private bodyKey = '';
  private workers = new Map<string, number>();
  t = 0;
  private store: Store;

  constructor(store: Store) {
    this.store = store;
  }

  layout(W: number, H: number, band = 0): void {
    this.L = computeLayout(W, H, band);
    this.duct = polyFrom([[this.L.exit.x + 6, this.L.exit.y], [this.L.PW - 6, this.L.exit.y]]);
    for (const j of [...this.trunks.values(), ...this.branches.values()]) j.edge = null;
    for (const e of this.ents.values()) e.twigKey = '';
    this.buildWash();
    this.organs = [];
    this.bodyKey = '';
    this.syncBodies(performance.now());
  }

  /** Engine instances -> KV bodies. One instance keeps the single tissue; PD gets two columns. */
  private syncBodies(now: number): void {
    const inst: Instance[] = this.store.instances(now);
    const key = inst.map((i) => `${i.name}:${i.role}:${i.url ?? ''}`).join('|');
    if (key === this.bodyKey && this.bodies.length) {
      for (const b of this.bodies) {
        const cap = this.store.kvCapacity(this.bodies.length === 1 && !inst.length ? undefined : b.name);
        if (cap.blocks !== b.cap.blocks || cap.known !== b.cap.known) {
          b.cap = cap;
          this.scaleBody(b);
        }
      }
      return;
    }
    this.bodyKey = key;
    const { kv } = this.L;
    const old = new Map(this.bodies.map((b) => [b.name, b.shown]));
    const pre = inst.filter((i) => i.role === 'prefill');
    const rest = inst.filter((i) => i.role !== 'prefill');
    this.pd = pre.length > 0 && rest.some((i) => i.role === 'decode');
    const mk = (i: Instance | undefined, cx: number, cy: number, rx: number, ry: number): Body => {
      const name = i?.name ?? 'engine';
      const b: Body = { name, role: String(i?.role ?? 'engine'), url: normUrl(i?.url), cap: this.store.kvCapacity(i?.name), dpb0: 1, dpb: 1, sp: 1, cx, cy, rx, ry, shown: old.get(name) ?? 0 };
      this.scaleBody(b);
      return b;
    };
    const column = (arr: Instance[], cx: number, rx: number) => {
      const h = (2 * kv.ry) / arr.length;
      return arr.map((it, k) => mk(it, cx, kv.cy - kv.ry + h * (k + 0.5), rx, Math.min(h * 0.44, rx * 1.7)));
    };
    if (inst.length <= 1) this.bodies = [mk(inst[0], kv.cx, kv.cy, kv.rx, kv.ry)];
    else if (this.pd) this.bodies = [...column(pre, kv.cx - kv.rx * 0.7, kv.rx * 0.5), ...column(rest, kv.cx + kv.rx * 0.62, kv.rx * 0.64)];
    else this.bodies = column(inst, kv.cx, kv.rx * 0.95);
    // re-home every patch into the new anatomy
    for (const e of this.ents.values()) {
      if (!e.patch) continue;
      e.patch = false;
      e.oldFade = 0;
      this.allocPatch(e, this.bodyFor(e.req));
    }
  }

  /** Linear capacity: the whole ellipse is the GPU's KV cache; one dot per block (thinned on huge caches). */
  private scaleBody(b: Body): void {
    b.dpb0 = Math.min(1, 36000 / Math.max(1, b.cap.blocks));
    this.applyDetail(b);
  }

  /** Zoomed in, colonies get more dots per block (up to one per block): detail, not just magnification. */
  private applyDetail(b: Body): void {
    b.dpb = Math.min(1, b.dpb0 * clamp(this.zoom * this.zoom, 1, 400));
    b.sp = Math.sqrt((b.rx * b.ry) / (b.cap.blocks * b.dpb));
  }

  /** Called every frame with the camera's zoom and the visible world rectangle. */
  setView(zoom: number, view: { x0: number; y0: number; x1: number; y1: number }): void {
    this.view = view;
    if (Math.abs(zoom - this.zoom) > 1e-3) {
      this.zoom = zoom;
      for (const b of this.bodies) this.applyDetail(b);
    }
  }

  /** On-plate radius of `blocks` KV blocks in body `bi` (true to scale, independent of zoom detail). */
  blockR(bi: number, blocks: number): number {
    const b = this.bodies[bi];
    if (!b) return 0;
    return Math.sqrt((b.rx * b.ry * Math.max(0, blocks)) / Math.max(1, b.cap.blocks));
  }

  /** Estimated reusable (cached) blocks per body: finished requests' tissue that has not eroded yet. */
  cachedBlocks(bi: number): number {
    let n = 0;
    for (const c of this.cachedCol) {
      if (c.body !== bi) continue;
      const age = (this.t - c.born) / c.life;
      n += c.blocks * Math.max(0, 1 - Math.pow(age, 1.6));
    }
    return n;
  }

  private workerIdx(url: string | undefined, kind: string): number {
    if (!url) return 0;
    const k = `${kind}|${url}`;
    let i = this.workers.get(k);
    if (i === undefined) {
      i = [...this.workers.keys()].filter((x) => x.startsWith(`${kind}|`)).length;
      this.workers.set(k, i);
    }
    return i;
  }

  private byUrl(url: string | undefined): number {
    const u = normUrl(url);
    return u ? this.bodies.findIndex((b) => b.url === u) : -1;
  }

  /** Which body holds this request's KV right now: by worker URL, else first-seen order. */
  bodyFor(r: Req): number {
    const n = this.bodies.length;
    if (n <= 1) return 0;
    if (!this.pd) {
      const i = this.byUrl(r.worker);
      return i >= 0 ? i : hashStr(r.rid) % n; // DP rank per request is not reported
    }
    const pre = this.bodies.map((b, i) => (b.role === 'prefill' ? i : -1)).filter((i) => i >= 0);
    const dec = this.bodies.map((b, i) => (b.role !== 'prefill' ? i : -1)).filter((i) => i >= 0);
    if (r.pd === 'prefill') {
      const i = this.byUrl(r.prefillWorker);
      return i >= 0 ? i : pre[this.workerIdx(r.prefillWorker, 'p') % pre.length];
    }
    const i = this.byUrl(r.decodeWorker);
    return i >= 0 ? i : dec[this.workerIdx(r.decodeWorker, 'd') % dec.length];
  }

  /** Instance name serving a worker URL (single instance: that one). */
  instanceOfWorker(url?: string): string | null {
    if (this.bodies.length === 1) return this.bodies[0].name;
    const i = this.byUrl(url);
    return i >= 0 ? this.bodies[i].name : null;
  }

  /** Vertical home for an admitted cell whose replica is known, so it waits opposite its body. */
  homeY(r: Req): number | null {
    if (this.pd || this.bodies.length <= 1) return null;
    const i = this.byUrl(r.worker);
    return i >= 0 ? this.bodies[i].cy : null;
  }

  /** KV moves from a prefill body to a decode body: the patch re-forms there, grains stream across. */
  private handoff(e: Entity, to: number): void {
    const r = e.req;
    e.oldPx = e.px;
    e.oldPy = e.py;
    e.oldPr = e.pr;
    e.oldBlocks = Math.max(1, e.blocks - e.shared);
    e.oldBody = e.body;
    e.oldFade = 1;
    const fx = e.px;
    const fy = e.py;
    const fr = e.pr;
    e.patch = false;
    this.allocPatch(e, to);
    e.arrive = 0;
    const n = Math.round(clamp(r.kvBlocks ?? promptTokens(r) / 16, 8, 90));
    for (let i = 0; i < n; i++) {
      this.transfers.push({
        x0: fx + gauss(e.rnd) * fr * 0.35, y0: fy + gauss(e.rnd) * fr * 0.35,
        x1: e.px + gauss(e.rnd) * e.pr * 0.35, y1: e.py + gauss(e.rnd) * e.pr * 0.35,
        bend: (e.rnd() - 0.5) * 0.6, t: -i * 0.012 - e.rnd() * 0.08, dur: 0.55 + e.rnd() * 0.3,
      });
    }
    this.transferRate += n;
  }

  // ---- static structures ------------------------------------------------------------------

  private buildWash(): void {
    const kv = { cx: 0, cy: 0, rx: 1, ry: 1 }; // unit template, transformed per body
    const rnd = mulberry32(11);
    const N = 9000;
    const items: { x: number; y: number; k: number; s: number }[] = [];
    while (items.length < N) {
      const a = rnd() * Math.PI * 2;
      const rr = Math.sqrt(rnd());
      const wob = 1 + 0.08 * Math.sin(a * 3 + 1.7) + 0.05 * Math.sin(a * 7);
      const x = kv.cx + Math.cos(a) * rr * kv.rx * wob;
      const y = kv.cy + Math.sin(a) * rr * kv.ry * wob;
      // fill order: from the centre outward, with organic noise
      const k = rr + 0.22 * (rnd() - 0.5) + 0.1 * Math.sin(a * 4);
      items.push({ x, y, k, s: 1.3 + rnd() * 1.1 });
    }
    items.sort((a, b) => a.k - b.k);
    this.wash = {
      x: Float32Array.from(items, (i) => i.x),
      y: Float32Array.from(items, (i) => i.y),
      sz: Float32Array.from(items, (i) => i.s),
    };
  }

  private buildOrgans(n: number): void {
    const { organX0, organX1, organY, kv } = this.L;
    this.organs = [];
    for (let i = 0; i < n; i++) {
      const x = n === 1 ? (organX0 + organX1) / 2 : lerp(organX0, organX1, i / (n - 1));
      const y = organY + (i % 2 ? 6 : -4);
      const rnd = mulberry32(1000 + i * 31);
      const dendrites: Poly[] = [];
      const tips: [number, number][] = [];
      const branches = 5 + Math.floor(rnd() * 3);
      for (let b = 0; b < branches; b++) {
        let ang = -Math.PI / 2 + (b / (branches - 1) - 0.5) * 2.3 + (rnd() - 0.5) * 0.3;
        let px = x;
        let py = y;
        const pts: [number, number][] = [[px, py]];
        const reachUp = b > 0 && b < branches - 1;
        const steps = reachUp ? 16 + Math.floor(rnd() * 12) : 6 + Math.floor(rnd() * 6);
        for (let s = 0; s < steps; s++) {
          ang += (rnd() - 0.5) * 0.5;
          if (reachUp) ang = lerp(ang, -Math.PI / 2, 0.08);
          const step = 5 + rnd() * 3;
          px += Math.cos(ang) * step;
          py += Math.sin(ang) * step;
          if (reachUp && py < kv.cy + kv.ry * 0.35) break;
          pts.push([px, py]);
        }
        dendrites.push(polyFrom(pts));
        tips.push([px, py]);
      }
      this.organs.push({ x, y, seed: i * 7919 + 3, dendrites, tips, util: 0, emit: 0 });
    }
  }

  membraneX(y: number): number {
    let press = 0;
    for (const p of this.pressing) {
      const u = (y - p.y) / (p.r * 1.6);
      press += Math.exp(-u * u) * Math.min(6, p.r * 0.45);
    }
    return this.L.memX + 7 * Math.sin(y * 0.013 + this.t * 0.22) + 4 * Math.sin(y * 0.041 + 1.3) + Math.min(9, press);
  }

  // ---- simulation ----------------------------------------------------------------------------

  update(dt: number, now: number): void {
    this.t += dt;
    const L = this.L;
    const store = this.store;
    this.syncBodies(now);

    for (const r of store.reqs.values()) {
      if (this.ents.has(r.rid)) continue;
      if (r.phase === 'done' || r.phase === 'failed') continue;
      const rnd = mulberry32(hashStr(r.rid) ^ 0x5bd1e995);
      const y = lerp(L.memTop + 20, L.memBot - 20, clamp(0.5 + gauss(rnd) * 0.2, 0, 1));
      const e = new Entity(r, -30 - rnd() * 20, y);
      if (r.phase === 'decode') {
        // joined mid-flight: place straight into the tissue
        e.inside = true;
        e.dissolve = 1;
        this.allocPatch(e, this.bodyFor(r));
        e.x = e.px;
        e.y = e.py;
      } else {
        e.vx = 220 + rnd() * 80;
      }
      this.ents.set(r.rid, e);
    }

    // decode-network flux targets reset
    for (const j of this.branches.values()) j.target = 0;
    for (const j of this.trunks.values()) j.target = 0;

    const queued: Entity[] = [];
    const ante: Entity[] = [];
    for (const [rid, e] of this.ents) {
      const r = e.req;
      const ph = r.phase;
      if (ph === 'queued') {
        queued.push(e);
      } else if (ph === 'prefill' && r.pd && this.pd) {
        // disaggregated: prefill happens in a prefill body, then KV is handed to a decode body
        e.inside = true;
        const want = this.bodyFor(r);
        if (!e.patch) this.allocPatch(e, want);
        else if (e.body !== want) this.handoff(e, want);
        const k = Math.min(1, dt * 5);
        e.x = lerp(e.x, e.px, k);
        e.y = lerp(e.y, e.py, k);
        e.dissolve = Math.min(r.pd === 'prefill' ? 0.85 : 1, e.dissolve + dt * 1.6);
      } else if (ph === 'prefill') {
        if (!e.inside) {
          e.inside = true;
          e.vx = 260;
          this.crossings.push({ y: e.y, t: this.t });
        }
        ante.push(e);
      } else if (ph === 'decode' || ph === 'done') {
        e.qs = lerp(e.qs, 1, Math.min(1, dt * 2));
        if (!e.inside) {
          e.inside = true;
          if (e.x < this.L.memX) this.crossings.push({ y: e.y, t: this.t });
        }
        if (!e.patch) this.allocPatch(e, this.bodyFor(r));
        else if (this.pd && ph === 'decode' && this.bodies[e.body]?.role === 'prefill') this.handoff(e, this.bodyFor(r));
        // travel into the patch and dissolve
        const k = Math.min(1, dt * 5);
        e.x = lerp(e.x, e.px, k);
        e.y = lerp(e.y, e.py, k);
        e.dissolve = Math.min(1, e.dissolve + dt * 2.2);
      }

      if (e.oldFade > 0) e.oldFade = Math.max(0, e.oldFade - dt / 0.9);
      if (e.arrive < 1) e.arrive = Math.min(1, e.arrive + dt / 0.8);

      // flux from real chunks
      if (e.patch) {
        const fresh = r.pendingSparks;
        r.pendingSparks = 0;
        const inst = fresh / Math.max(dt, 1e-3);
        e.flux = e.flux * Math.exp(-dt * 2.5) + inst * (1 - Math.exp(-dt * 2.5));
        this.sizeColony(e);
        // token batches (~10 Hz per request) are released as sparks over ~100 ms
        e.sparkQ = Math.min(e.sparkQ + fresh, 80);
        const release = e.sparkQ > 0 ? Math.min(Math.ceil(e.sparkQ * Math.min(1, dt / 0.1)), 8) : 0;
        e.sparkQ -= release;
        if (e.branch && release > 0 && this.sparks.length < MAX_SPARKS) {
          const n = release;
          for (let i = 0; i < n; i++) {
            this.sparks.push({
              rid: r.rid,
              route: e.route,
              s: -i * 4 - e.rnd() * 3,
              v: 210 + e.rnd() * 70,
              ink: e.rnd() < 0.15 ? INK.crimsonDeep : INK.crimson,
              size: 1.9 + e.rnd() * 0.7,
            });
          }
        }
        // twig -> branch membership: one running request inside its traffic class and instance
        const b = this.bodies[e.body];
        if (b && b.role !== 'prefill' && (ph === 'decode' || ph === 'done')) {
          const br = this.ensureBranch(b.name, trafficClass(r));
          br.target += e.flux;
          e.branch = br.key;
        } else e.branch = null;
      } else {
        r.pendingSparks = 0;
      }

      if (e.patch) {
        // a colony grows with its request: keep it whole inside its body (it drifts in, no jump)
        const bb = this.bodies[e.body];
        if (bb && e.pr > 0) {
          const lim = Math.max(0, 1 - (e.pr + 6) / Math.min(bb.rx, bb.ry));
          const nx = (e.tx - bb.cx) / bb.rx;
          const ny = (e.ty - bb.cy) / bb.ry;
          const d = Math.hypot(nx, ny);
          if (d > lim + 1e-3) {
            e.tx = bb.cx + (nx * lim / d) * bb.rx;
            e.ty = bb.cy + (ny * lim / d) * bb.ry;
          }
        }
        const k = Math.min(1, dt * 3);
        e.px = lerp(e.px, e.tx, k);
        e.py = lerp(e.py, e.ty, k);
      }
      if (e.flash) {
        e.flash.t += dt;
        if (e.flash.t > 2.2) e.flash = null;
      }

      if (ph === 'done') {
        if (!e.cachedOut && e.patch) {
          // its blocks stay behind as reusable prefix-cache tissue until evicted
          e.cachedOut = true;
          const b = this.bodies[e.body];
          const kv = this.store.engine?.kv_usage ?? 0;
          if (b && e.blocks > 0) this.cachedCol.push({ x: e.px, y: e.py, blocks: Math.max(1, e.blocks - e.shared), seed: e.seed, born: this.t, life: kv > 0.85 ? 5 : 25, body: e.body });
          if (this.cachedCol.length > 3000) this.cachedCol.splice(0, this.cachedCol.length - 3000);
        }
        e.fade = Math.min(1, e.fade + dt / 2.6);
        if (!e.spored) {
          e.spored = true;
          const out = r.completionTokens ?? r.chunks;
          this.spores.push({
            x: L.exit.x, y: L.exit.y + (e.rnd() - 0.5) * 16, vx: 40 + e.rnd() * 30, vy: (e.rnd() - 0.5) * 24,
            r: clamp(3 + Math.sqrt(out) * 0.35, 3.5, 12), born: this.t, life: 2.4, ink: INK.crimson, seed: e.seed,
          });
        }
        if (e.fade >= 1 && e.flux < 0.2) this.ents.delete(rid);
      } else if (ph === 'failed') {
        e.fade = Math.min(1, e.fade + dt / 2.2);
        e.vx = -60;
        e.x += e.vx * dt;
        e.dissolve = 0;
        if (e.fade >= 1) this.ents.delete(rid);
      }
    }

    this.packQueue(queued, dt, -1);
    this.packQueue(ante, dt, +1);
    this.pressing = queued.filter((e) => e.x > this.membraneX(e.y) - e.r * e.qs * 2.6).map((e) => ({ y: e.y, r: e.r * e.qs }));
    this.crossings = this.crossings.filter((c) => this.t - c.t < 2.5);

    this.updateNetwork(dt);
    this.updateKv(dt, now);

    // sparks: with per-step batches, a request's sparks hop once per scheduler step it is in,
    // so a request left out of the batch visibly stalls; otherwise they glide
    const paced = now - store.batchAt < 2000;
    const sps = store.engine?.steps_per_s ?? 100;
    const hop = 220 / Math.max(5, sps);
    const hits = store.stepHits;
    const sp = this.sparks;
    let w = 0;
    for (let i = 0; i < sp.length; i++) {
      const s = sp[i];
      if (paced && this.ents.get(s.rid)?.req.phase === 'decode') s.s += (hits.get(s.rid) ?? 0) * hop;
      else s.s += s.v * dt;
      if (s.s < routeLen(s.route) + 20) sp[w++] = s;
    }
    sp.length = w;
    hits.clear();
    if (this.highlight && this.t > this.highlight.until) this.highlight = null;

    // KV handovers in flight
    this.transfers = this.transfers.filter((tr) => (tr.t += dt) < tr.dur);
    this.transferRate *= Math.exp(-dt * 1.5);

    // spores
    this.spores = this.spores.filter((s) => {
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      return this.t - s.born < s.life;
    });

    // anonymous preemptions: a grey cell pushed back out across the membrane
    while (store.preemptPending > 0) {
      store.preemptPending--;
      const a = Math.random() * Math.PI * 2;
      this.spores.push({
        x: L.kv.cx + Math.cos(a) * L.kv.rx * 0.6, y: L.kv.cy + Math.sin(a) * L.kv.ry * 0.6,
        vx: -(L.kv.cx - L.memX) / 2.2, vy: (Math.random() - 0.5) * 20, r: 7, born: this.t, life: 2.8, ink: INK.greyDeep,
        seed: (Math.random() * 1e9) | 0,
      });
    }

    this.updateOrgans(dt, now);
  }

  private packQueue(cells: Entity[], dt: number, side: -1 | 1): void {
    const L = this.L;
    const scale = clamp(Math.sqrt(45 / Math.max(45, cells.length)), 0.42, 1);
    for (const e of cells) {
      e.qs = lerp(e.qs, scale, Math.min(1, dt * 3));
      const er = e.r * e.qs;
      const mx = this.membraneX(e.y);
      if (side < 0) {
        const tx = mx - er - 2;
        e.vx += (tx - e.x) * 9 * dt;
        e.vx *= Math.exp(-dt * 3.2);
        e.x += e.vx * dt;
        if (e.x > tx) {
          e.x = tx;
          e.vx = Math.min(0, e.vx);
        }
      } else {
        const tx = mx + er + 18; // leave the membrane visible between the held and the admitted crowds
        if (e.x < mx) {
          e.x += Math.max(e.vx, 200) * dt; // crossing the membrane
        } else {
          e.vx += (tx - e.x) * 6 * dt;
          e.vx *= Math.exp(-dt * 3.2);
          e.x += e.vx * dt;
          if (e.x < tx) e.x = Math.max(e.x, mx + er + 10);
        }
        e.dissolve = Math.min(0.18, e.dissolve + dt * 0.1);
        const hy = this.homeY(e.req);
        if (hy !== null) e.vy += (hy - e.y) * 3 * dt;
      }
      e.vy *= Math.exp(-dt * 4);
      e.y += e.vy * dt;
    }
    // overlap resolution
    for (let it = 0; it < 2; it++) {
      for (let i = 0; i < cells.length; i++) {
        const a = cells[i];
        for (let j = i + 1; j < cells.length; j++) {
          const b = cells[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const min = a.r * a.qs + b.r * b.qs + 0.8;
          const d2 = dx * dx + dy * dy;
          if (d2 >= min * min || d2 === 0) continue;
          const d = Math.sqrt(d2);
          const push = (min - d) / 2;
          const ux = dx / d;
          const uy = dy / d;
          // stack away from the membrane, slide along it
          a.x -= ux * push * (side < 0 ? 1.4 : 0.6);
          b.x += ux * push * (side < 0 ? 0.6 : 1.4);
          a.y -= uy * push;
          b.y += uy * push;
        }
      }
      for (const e of cells) {
        const er = e.r * e.qs;
        e.y = clamp(e.y, L.memTop + er, L.memBot - er);
        const mx = this.membraneX(e.y);
        if (side < 0 && e.x > mx - er - 2) e.x = mx - er - 2;
        if (side > 0 && e.inside && e.x < mx + er + 10 && e.x > mx - 4) e.x = mx + er + 10;
      }
    }
  }

  private allocPatch(e: Entity, bi = 0): void {
    const b = this.bodies[bi] ?? this.bodies[0];
    if (!b) return;
    e.body = Math.max(0, Math.min(bi, this.bodies.length - 1));
    e.patch = true;
    this.sizeColony(e);
    const [x, y] = this.freeSpot(e, b); // shared-core members move next to their core once the group is known
    e.tx = x;
    e.ty = y;
    e.px = x;
    e.py = y;
    e.capillary = null;
    e.twigKey = '';
    if (b.role === 'prefill') e.branch = null;
  }

  /** A free spot in the body, away from live colonies and family cores. */
  private freeSpot(e: Entity, b: Body): [number, number] {
    let best: [number, number] = [b.cx, b.cy];
    let bestScore = -Infinity;
    for (let c = 0; c < 16; c++) {
      const a = e.rnd() * Math.PI * 2;
      const rr = Math.sqrt(e.rnd()) * 0.86;
      const x = b.cx + Math.cos(a) * rr * b.rx;
      const y = b.cy + Math.sin(a) * rr * b.ry;
      let score = Infinity;
      for (const o of this.ents.values()) {
        if (!o.patch || o === e || o.fade > 0.6 || o.body !== e.body) continue;
        score = Math.min(score, Math.hypot(o.tx - x, o.ty - y) - o.pr - e.pr);
      }
      for (const f of this.famCores.values()) {
        if (f.body !== e.body) continue;
        score = Math.min(score, Math.hypot(f.x - x, f.y - y) - this.blockR(f.body, f.blocks) - e.pr - 6);
      }
      score -= rr * 6;
      if (score > bestScore) {
        bestScore = score;
        best = [x, y];
      }
    }
    return best;
  }

  /** Members of a shared core attach around it. */
  private attachPoint(e: Entity, f: FamCore): [number, number] {
    f.spin += 2.399963;
    const d = this.blockR(f.body, f.blocks) + e.pr + 2;
    return [f.x + Math.cos(f.spin) * d, f.y + Math.sin(f.spin) * d];
  }

  /** The engine's KV-map entry for a running request (kv.map format 2), if its body has one. */
  ownerOf(e: Entity): KvOwner | null {
    const m = this.mapFor(e.body);
    if (!m || m.format !== 2) return null;
    let idx = this.ownerIdx.get(m);
    if (!idx) {
      idx = new Map(m.owners.map((o) => [o.rid, o]));
      this.ownerIdx.set(m, idx);
    }
    return idx.get(e.req.rid) ?? null;
  }

  /**
   * Colony size = the request's KV, to scale: ceil(tokens / block size), growing during decode.
   * From the centre out: the prefix it found in the cache (dark), the prompt it computed, then generated tokens.
   * Tokens it shares with other running requests are drawn once, in their shared core, and left out here.
   */
  private sizeColony(e: Entity): void {
    const b = this.bodies[e.body];
    if (!b) return;
    const r = e.req;
    const bs = b.cap.blockSize;
    const prompt = promptTokens(r);
    const o = this.ownerOf(e);
    const core = e.coreId ? this.famCores.get(e.coreId) : undefined;
    const tokens = o ? o.tokens : prompt + r.chunks;
    const cached = Math.min(tokens, o?.cachedTokens ?? r.cachedTokens ?? 0);
    const shared = core ? Math.min(cached || tokens, o ? (o.sharedTokens ?? 0) : core.tokens) : 0;
    e.blocks = Math.ceil(tokens / bs);
    e.shared = Math.floor(shared / bs);
    const own = Math.max(1, e.blocks - e.shared);
    e.dots = Math.max(1, Math.round(own * b.dpb));
    e.innerDots = Math.min(e.dots, Math.round((Math.max(0, cached - shared) / bs) * b.dpb));
    e.promptDots = clamp(Math.round((Math.max(0, prompt - shared) / bs) * b.dpb), e.innerDots, e.dots);
    e.pr = b.sp * Math.sqrt(e.dots) + 1;
  }

  /** Family cores, cached tissue, flashes, evictions and the KV read rate. */
  private updateKv(dt: number, now: number): void {
    const s = this.store;
    // shared cores: two or more running requests that hold the same prefix right now. With the engine's
    // KV map (format 2) the groups and sizes are slot-exact; without it, requests of one prefix family that
    // each reused cached tokens are grouped, sized by the smallest reuse (an estimate).
    const want = new Map<string, { members: Entity[]; tokens: number; body: number; measured: boolean }>();
    for (const e of this.ents.values()) {
      e.coreId = null;
      const r = e.req;
      if (!e.patch || e.body < 0 || e.cachedOut || r.phase === 'done' || r.phase === 'failed') continue;
      if (this.bodies[e.body]?.role === 'prefill') continue;
      const o = this.ownerOf(e);
      let key: string | null = null;
      let tokens = 0;
      let measured = false;
      if (o) {
        if (o.group && (o.sharedTokens ?? 0) > 0) [key, tokens, measured] = [`g:${o.group}`, o.groupTokens, true];
      } else if (!this.mapFor(e.body) || this.mapFor(e.body)!.format !== 2) {
        if (r.family && (r.cachedTokens ?? 0) > 0 && (r.prefixOutcome === 'hit' || r.prefixOutcome === 'partial')) {
          [key, tokens] = [r.family, r.cachedTokens!];
        }
      }
      if (!key) continue;
      const w = want.get(key) ?? { members: [], tokens, body: e.body, measured };
      w.members.push(e);
      w.tokens = measured ? Math.max(w.tokens, tokens) : Math.min(w.tokens, tokens);
      want.set(key, w);
    }
    for (const [id, w] of want) {
      if (w.members.length < 2) continue;
      const b = this.bodies[w.body];
      if (!b) continue;
      let core = this.famCores.get(id);
      const blocks = Math.max(1, Math.ceil(w.tokens / b.cap.blockSize));
      if (!core) {
        const first = w.members[0];
        core = { id, x: first.tx, y: first.ty, body: w.body, blocks, tokens: w.tokens, members: 0, measured: w.measured,
          seed: hashStr(id), spin: 0, lastActive: this.t };
        this.famCores.set(id, core);
      }
      Object.assign(core, { blocks, tokens: w.tokens, members: w.members.length, measured: w.measured, lastActive: this.t });
      for (const e of w.members) {
        e.coreId = id;
        // members sit around their one core
        if (Math.hypot(e.tx - core.x, e.ty - core.y) > this.blockR(core.body, core.blocks) + e.pr + 8) {
          const p = this.attachPoint(e, core);
          [e.tx, e.ty] = p;
        }
      }
    }
    for (const [id, core] of this.famCores) {
      // a core grows as more tokens are shared: keep it whole inside its body (drift, no jump)
      const bb = this.bodies[core.body];
      if (bb) {
        const R = this.blockR(core.body, core.blocks);
        const lim = Math.max(0, 1 - (R + 8) / Math.min(bb.rx, bb.ry));
        const nx = (core.x - bb.cx) / bb.rx;
        const ny = (core.y - bb.cy) / bb.ry;
        const d = Math.hypot(nx, ny);
        if (d > lim + 1e-3) {
          const f = lim / d;
          const k = Math.min(1, dt * 2.5);
          core.x += (bb.cx + nx * f * bb.rx - core.x) * k;
          core.y += (bb.cy + ny * f * bb.ry - core.y) * k;
        }
      }
      if (core.lastActive < this.t) {
        // fewer than two running holders: the prefix is cached, not shared. With the engine's map the
        // cached share is drawn from the map; without it, it lingers here as estimated reusable tissue.
        if (!core.measured) this.cachedCol.push({ x: core.x, y: core.y, blocks: core.blocks, seed: core.seed, born: this.t, life: 30, body: core.body });
        this.famCores.delete(id);
      }
    }
    // hit / miss flashes from the engine's cached-token counts
    for (const f of s.prefixFlashes.splice(0)) {
      const e = this.ents.get(f.rid);
      if (e) e.flash = { kind: f.kind, t: 0 };
    }
    // evictions erode the oldest reusable tissue and throw flecks off the rim
    while (s.evictPending > 0) {
      s.evictPending--;
      const old = this.cachedCol.find((c) => c.life > 1);
      if (old) old.life = Math.max(0.6, this.t - old.born + 0.6);
      const b = this.bodies[old?.body ?? 0];
      if (b) {
        const a = Math.random() * Math.PI * 2;
        this.flecks.push({ x: b.cx + Math.cos(a) * b.rx, y: b.cy + Math.sin(a) * b.ry, vx: Math.cos(a) * 30, vy: Math.sin(a) * 30, t: 0 });
      }
    }
    this.flecks = this.flecks.filter((f) => {
      f.t += dt;
      f.x += f.vx * dt;
      f.y += f.vy * dt;
      return f.t < 1.4;
    });
    this.cachedCol = this.cachedCol.filter((c) => this.t - c.born < c.life);
    // KV read rate: every decode step reads each running request's whole context
    const inst = new Map(s.instances(now).map((i) => [i.name, i]));
    let read = 0;
    for (const e of this.ents.values()) {
      if (e.req.phase !== 'decode' || !e.patch) continue;
      const b = this.bodies[e.body];
      const sps = inst.get(b?.name ?? '')?.steps_per_s ?? s.engine?.steps_per_s ?? 0;
      read += (promptTokens(e.req) + e.req.chunks) * (sps ?? 0);
    }
    this.kvRead = lerp(this.kvRead, read, 1 - Math.exp(-dt * 2));
  }

  private ensureTrunk(name: string): Junction {
    let t = this.trunks.get(name);
    if (!t) {
      const { exit } = this.L;
      t = {
        key: name, kind: 'trunk', name, parent: null, x: exit.x - 60, y: exit.y, tx: exit.x - 60, ty: exit.y,
        flux: 0, target: 0, edge: null, ex: 0, ey: 0, px: 0, py: 0, seed: hashStr(name), idle: 0,
      };
      this.trunks.set(name, t);
    }
    return t;
  }

  private ensureBranch(trunk: string, cls: string): Junction {
    const key = `${trunk}|${cls}`;
    let b = this.branches.get(key);
    if (!b) {
      const t = this.ensureTrunk(trunk);
      b = {
        key, kind: 'branch', name: cls, parent: trunk, x: t.x - 40, y: t.y, tx: t.x - 40, ty: t.y,
        flux: 0, target: 0, edge: null, ex: 0, ey: 0, px: 0, py: 0, seed: hashStr(key), idle: 0,
      };
      this.branches.set(key, b);
    }
    return b;
  }

  /** Lay out the measured hierarchy (request -> class -> instance -> exit) and move it smoothly. */
  private updateNetwork(dt: number): void {
    const { net, exit } = this.L;
    const k = 1 - Math.exp(-dt * 2.2);
    const kf = 1 - Math.exp(-dt * 3);
    // trunks: one per decode-capable instance, in body order
    const live = this.bodies.filter((b) => b.role !== 'prefill');
    const names = new Set(live.map((b) => b.name));
    for (const name of [...this.trunks.keys()]) if (!names.has(name)) this.trunks.delete(name);
    for (const [key, b] of this.branches) if (!names.has(b.parent!)) this.branches.delete(key);
    const trunkX = lerp(net.x0, net.x1, 0.62);
    live.forEach((b) => {
      const t = this.ensureTrunk(b.name);
      t.tx = trunkX;
      t.ty = live.length === 1 ? exit.y : lerp(exit.y, b.cy, 0.62);
    });
    // branches: grouped opposite their own body (where the twigs come from), ordered by class
    const span = ((net.y1 - net.y0) * 0.86) / Math.max(1, live.length);
    const homeY = new Map(live.map((b) => [b.name, live.length === 1 ? exit.y : b.cy]));
    for (const t of this.trunks.values()) {
      const mine = [...this.branches.values()]
        .filter((b) => b.parent === t.key)
        .sort((a, b) => (CLASS_ORDER.indexOf(a.name) + 1 || 99) - (CLASS_ORDER.indexOf(b.name) + 1 || 99) || a.name.localeCompare(b.name));
      const gap = clamp((span * 0.8) / Math.max(1, mine.length), 22, 66);
      const cy = homeY.get(t.key) ?? t.ty;
      mine.forEach((b, i) => {
        b.tx = lerp(net.x0, net.x1, 0.06) + (h1(b.seed) - 0.5) * 16;
        b.ty = cy + (i - (mine.length - 1) / 2) * gap;
        t.target += b.target;
      });
    }
    for (const j of [...this.trunks.values(), ...this.branches.values()]) {
      j.x = lerp(j.x, j.tx, k);
      j.y = lerp(j.y, j.ty, k);
      j.flux = lerp(j.flux, j.target, kf);
    }
    for (const [key, b] of this.branches) {
      b.idle = b.target === 0 && b.flux < 0.3 ? b.idle + dt : 0;
      if (b.idle > 4) this.branches.delete(key);
    }
    this.aperture = lerp(this.aperture, [...this.trunks.values()].reduce((a, t) => a + t.target, 0), kf);
    // edges, rebuilt only when their endpoints move
    const edge = (j: Junction, x2: number, y2: number, bend: number) => {
      if (j.edge && Math.abs(j.ex - x2) + Math.abs(j.ey - y2) + Math.abs(j.px - j.x) + Math.abs(j.py - j.y) < 0.5) return;
      j.edge = polyFrom(bezierPts(j.x, j.y, x2, y2, bend, 16));
      j.ex = x2;
      j.ey = y2;
      j.px = j.x;
      j.py = j.y;
    };
    for (const t of this.trunks.values()) edge(t, exit.x - 4, exit.y, (h1(t.seed) - 0.5) * 0.22);
    for (const b of this.branches.values()) {
      const t = this.trunks.get(b.parent!)!;
      edge(b, t.x, t.y, (h1(b.seed) - 0.5) * 0.3);
    }
    // twigs and routes
    for (const e of this.ents.values()) {
      const b = e.branch ? this.branches.get(e.branch) : undefined;
      if (!b) {
        e.route.segs = [];
        continue;
      }
      let [ox, oy] = (this.memMap && this.mapCentroid.get(e.req.rid)) || [e.px, e.py];
      if (!this.memMap) {
        // a vein leaves its own colony at the rim facing its branch
        const dx = b.x - ox;
        const dy = b.y - oy;
        const dd = Math.hypot(dx, dy) || 1;
        ox += (dx / dd) * (e.pr + 1.5);
        oy += (dy / dd) * (e.pr + 1.5);
      }
      const key = `${b.x.toFixed(0)},${b.y.toFixed(0)},${ox.toFixed(0)},${oy.toFixed(0)}`;
      if (key !== e.twigKey || !e.capillary) {
        e.capillary = polyFrom(bezierPts(ox, oy, b.x, b.y, (h1(e.seed) - 0.5) * 0.45, 14));
        e.twigKey = key;
      }
      e.route.segs = [e.capillary, b.edge, this.trunks.get(b.parent!)?.edge ?? null, this.duct];
    }
  }

  private updateOrgans(dt: number, now: number): void {
    const s = this.store;
    const cov = s.coverage(now);
    const n = cov.gpu ? s.gpu!.gpus!.length : Math.max(1, s.hello?.gpus ?? 1);
    if (this.organs.length !== n) this.buildOrgans(n);
    const stats = s.stats(now);
    const stepsPerS = stats.stepsPerS;
    for (let i = 0; i < this.organs.length; i++) {
      const o = this.organs[i];
      const g = cov.gpu ? s.gpu!.gpus![i] : undefined;
      const u = g?.util != null ? g.util / 100 : 0;
      o.util = lerp(o.util, u, 1 - Math.exp(-dt * 4));
      if (!cov.gpu) continue;
      // signal rate follows the engine's step rate when known, else GPU utilisation
      const rate = stepsPerS != null && cov.steps ? clamp(stepsPerS * 0.18, 0, 12) * (0.3 + 0.7 * o.util) : o.util * 7;
      o.emit += rate * dt;
      while (o.emit >= 1) {
        o.emit -= 1;
        const d = o.dendrites[Math.floor(Math.random() * o.dendrites.length)];
        this.signals.push({ path: d, s: 0, v: 90 + Math.random() * 40 });
      }
    }
    this.signals = this.signals.filter((g) => (g.s += g.v * dt) < g.path.len);
  }

  // ---- drawing ------------------------------------------------------------------------------

  draw(d: Dots, now: number): void {
    const s = this.store;
    const cov = s.coverage(now);
    d.reset();
    // colonies and cores that veins and sparks pass beneath (colony view)
    this.discs.length = 0;
    if (!this.memMap) {
      for (const e of this.ents.values()) if (e.patch && e.dissolve > 0.5 && e.fade < 0.6) this.discs.push([e.px, e.py, e.pr + 1, e.seed]);
      for (const f of this.famCores.values()) this.discs.push([f.x, f.y, this.blockR(f.body, f.blocks) + 3, -1]);
    }
    this.drawBodies(d, cov.kv, now);
    this.drawTransfers(d);
    this.drawMembrane(d);
    this.drawNetwork(d);
    for (const e of this.ents.values()) this.drawPatch(d, e);
    for (const e of this.ents.values()) this.drawCapillary(d, e);
    this.drawOrgans(d, cov.gpu);
    for (const e of this.ents.values()) this.drawCell(d, e, now);
    this.drawSparks(d);
    this.drawSpores(d);
    this.drawFollow(d);
    this.drawHighlight(d);
  }

  /** Expert routing (MoE): one dot per expert on a ring, size ∝ routed-token share; inner ticks = per-layer imbalance. */
  moeLive(now: number): boolean {
    const m = this.store.moe;
    return !!m && now - this.store.moeAt < 6000 && !!m.load?.length;
  }

  moeExperts(): number {
    const m = this.store.moe;
    if (m?.experts) return m.experts;
    if (m?.load?.length) return m.load.length;
    const c = this.store.hello?.config ?? {};
    const n = Number((c as Record<string, unknown>).num_experts ?? 0);
    if (n) return n;
    return /A3B/i.test(String(this.store.hello?.model ?? '')) ? 128 : 0;
  }

  private drawMoe(d: Dots, now: number): void {
    const E = this.moeExperts();
    if (!E) return;
    const { cx, cy, r } = this.L.moe;
    const live = this.moeLive(now);
    const m = this.store.moe;
    if (this.moeShown.length !== E) this.moeShown = new Array(E).fill(1 / E);
    const dim = this.follow ? 0.7 : 1;
    for (let i = 0; i < E; i++) {
      const a = (i / E) * Math.PI * 2 - Math.PI / 2;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      if (!live) {
        if (i % 2 === 0) d.add(x, y, 1.4, INK.grey, 0.6);
        continue;
      }
      const share = m!.load![i] ?? 0;
      this.moeShown[i] = lerp(this.moeShown[i], share, 0.1);
      const rel = this.moeShown[i] * E; // 1 = perfectly balanced
      const hot = clamp((rel - 1) / 2, 0, 1);
      const pulse = hot > 0.15 ? 0.75 + 0.25 * Math.sin(this.t * 6 + i) : 1;
      const n = 3 + Math.round(clamp(rel, 0, 4) * 3);
      for (let k = 0; k < n; k++) {
        const ang = h1(i * 31 + k) * Math.PI * 2;
        const rr = Math.sqrt(h1(i * 17 + k)) * (1.4 + rel * 1.9);
        d.add(x + Math.cos(ang) * rr, y + Math.sin(ang) * rr, 1.7 + hot * 1.5, INK.sap, (0.45 + 0.5 * Math.min(1.5, rel) / 1.5) * pulse * dim);
      }
      // a fine spoke towards the centre for hot experts
      if (hot > 0.3) for (let q = 0.45; q < 0.95; q += 0.05) d.add(cx + Math.cos(a) * r * q, cy + Math.sin(a) * r * q, 1.1, INK.sap, 0.35 * hot * pulse * dim);
    }
    // per-layer imbalance strip above the ring: one tick per layer, height ∝ max/mean
    const st = this.moeStrip();
    const lay = live ? m!.layer_imbalance ?? [] : [];
    const top = Math.max(2, ...lay);
    for (let l = 0; l < lay.length; l++) {
      const x = st.x0 + ((l + 0.5) / lay.length) * (st.x1 - st.x0);
      const h = clamp((lay[l] - 1) / (top - 1 || 1), 0, 1) * 22 + 1.5;
      const worst = lay[l] === top;
      for (let q = 0; q <= h; q += 1.5) d.add(x, st.y - q, worst ? 1.6 : 1.3, worst ? INK.sap : INK.sapLight, (worst ? 1 : 0.8) * dim);
    }
    if (!live) for (let x = st.x0; x <= st.x1; x += 4) d.add(x, st.y, 1.2, INK.grey, 0.55);
    // centre: a faint hub, the router
    for (let q = 0; q < 36; q++) d.add(cx + Math.cos((q / 36) * Math.PI * 2) * r * 0.22, cy + Math.sin((q / 36) * Math.PI * 2) * r * 0.22, 1, live ? INK.sapLight : INK.grey, 0.4 * dim);
  }

  /** Geometry of the per-layer strip and its worst layer, for the overlay labels. */
  moeStrip(): { x0: number; x1: number; y: number; worst: { l: number; v: number; x: number; y: number } | null; hot: { i: number; x: number; y: number }[] } {
    const { cx, cy, r } = this.L.moe;
    const x0 = cx - r * 1.25;
    const x1 = cx + r * 1.25;
    const y = cy - r - 12;
    const m = this.store.moe;
    const lay = m?.layer_imbalance ?? [];
    let worst: { l: number; v: number; x: number; y: number } | null = null;
    if (lay.length) {
      const v = Math.max(...lay);
      const l = lay.indexOf(v);
      worst = { l, v, x: x0 + ((l + 0.5) / lay.length) * (x1 - x0), y };
    }
    const E = this.moeExperts();
    const hot = ((m as { hot?: number[] } | null)?.hot ?? []).slice(0, 3).map((i) => {
      const a = (i / E) * Math.PI * 2 - Math.PI / 2;
      return { i, x: cx + Math.cos(a) * (r + 13), y: cy + Math.sin(a) * (r + 13) };
    });
    return { x0, x1, y, worst, hot };
  }

  /** Mark the requests of one scheduler step (clicked in the batch strip). */
  highlightStep(rids: string[]): void {
    this.highlight = { rids: new Set(rids), until: this.t + 5 };
  }

  private drawHighlight(d: Dots): void {
    if (!this.highlight) return;
    const fade = clamp(this.highlight.until - this.t, 0, 1);
    for (const rid of this.highlight.rids) {
      const e = this.ents.get(rid);
      if (!e) continue;
      const cx = e.patch ? e.px : e.x;
      const cy = e.patch ? e.py : e.y;
      const R = (e.patch ? e.pr : e.r) + 5;
      const n = Math.max(12, Math.floor(R * 2));
      for (let k = 0; k < n; k++) {
        const a = (k / n) * Math.PI * 2 - this.t;
        d.add(cx + Math.cos(a) * R, cy + Math.sin(a) * R, 1.6, INK.ultraDeep, 0.9 * fade);
      }
    }
  }

  /** Ring the narrated request and ink its whole way out. */
  private drawFollow(d: Dots): void {
    const e = this.follow ? this.ents.get(this.follow) : undefined;
    if (!e) return;
    const cellVisible = e.dissolve < 0.6 || !e.patch;
    if (this.memMap && !cellVisible && this.mapFor(e.body)?.levels) return; // the map marks its cells instead
    const cx = cellVisible ? e.x : e.px;
    const cy = cellVisible ? e.y : e.py;
    const R = (cellVisible ? e.r * e.qs : e.pr) + 8;
    // a bold dashed ring, and a fainter one outside it: easy to find at any zoom
    for (const [rr, size, alpha] of [[R, 2.6, 1], [R + 6, 1.8, 0.55]] as const) {
      const n = Math.floor(rr * 2.2);
      for (let k = 0; k < n; k++) {
        if (k % 6 === 5) continue;
        const a = (k / n) * Math.PI * 2 + this.t * 0.6;
        d.add(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, size, INK.crimsonDeep, alpha * (1 - e.fade * 0.7));
      }
    }
    const len = routeLen(e.route);
    for (let s = 0; s < len; s += 3.2) {
      atRoute(e.route, s, TMP);
      d.add(TMP.x, TMP.y, 1.5, INK.crimsonDeep, 0.55 * (1 - e.fade * 0.8));
    }
  }

  private drawBodies(d: Dots, kvLive: boolean, now: number): void {
    const inst = new Map(this.store.instances(now).map((i) => [i.name, i]));
    const global = this.store.engine?.kv_usage ?? null;
    if (this.pd) this.drawBridge(d);
    this.mapCentroid.clear();
    this.bodies.forEach((b, bi) => {
      const own = inst.get(b.name)?.kv_usage;
      const usage = kvLive ? (own ?? (this.bodies.length === 1 ? global : null)) : null;
      if (this.memMap) {
        this.drawCapacity(d, b, bi, null === usage ? null : 0); // rim only
        this.drawMemMap(d, bi, now);
      } else {
        this.drawCapacity(d, b, bi, usage);
        this.drawMapCached(d, b, bi);
      }
    });
    if (!this.memMap) {
      this.drawCached(d);
      for (const f of this.famCores.values()) this.drawCore(d, f);
    }
    for (const f of this.flecks) d.add(f.x, f.y, 1.4, INK.ultraPale, 0.7 * (1 - f.t / 1.4));
  }

  /** The KV cache at true scale: a faint rim is 100%; used blocks not owned by a visible request fill from the centre. */
  private drawCapacity(d: Dots, b: Body, bi: number, usage: number | null): void {
    const w = this.wash!;
    const rim = Math.floor(300 * Math.sqrt((b.rx * b.ry) / (this.L.kv.rx * this.L.kv.ry)));
    for (let i = 0; i < rim; i++) {
      const a = (i / rim) * Math.PI * 2;
      if (usage == null && i % 7 > 4) continue;
      const wob = 1 + 0.05 * Math.sin(a * 3 + 1.7) + 0.03 * Math.sin(a * 7);
      d.add(b.cx + Math.cos(a) * b.rx * wob, b.cy + Math.sin(a) * b.ry * wob, 1.9, usage == null ? INK.grey : b.role === 'prefill' ? INK.violet : INK.ultra, usage == null ? 0.6 : 0.55);
    }
    if (usage == null) return;
    // the medium: free capacity, barely there
    for (let i = 0; i < w.x.length; i += 9) d.add(b.cx + w.x[i] * b.rx, b.cy + w.y[i] * b.ry, 1.15, INK.ultraPale, 0.12);
    // used blocks the engine reports beyond what visible colonies account for
    let attributed = 0;
    for (const e of this.ents.values()) if (e.patch && e.body === bi && !e.cachedOut) attributed += e.blocks - e.shared;
    for (const f of this.famCores.values()) if (f.body === bi) attributed += f.blocks;
    const used = usage * b.cap.blocks;
    const extra = Math.max(0, used - attributed);
    b.shown = lerp(b.shown, extra * b.dpb, 0.08);
    const n = Math.min(w.x.length, Math.floor(b.shown));
    const scale = Math.sqrt(n / w.x.length);
    for (let i = 0; i < n; i++) d.add(b.cx + w.x[i] * b.rx * scale * 1.0, b.cy + w.y[i] * b.ry * scale, w.sz[i] * 0.9, INK.ultra, 0.75);
  }

  private sunflower(d: Dots, x: number, y: number, n: number, sp: number, seed: number, from: number, to: number, ink: Ink, alpha: number, size = 2.3): void {
    const R = sp * Math.sqrt(to + 1) + 4;
    const v = this.view;
    if (x + R < v.x0 || x - R > v.x1 || y + R < v.y0 || y - R > v.y1) return;
    const rot = (seed % 628) / 100;
    for (let k = from; k < to; k++) {
      const rho = sp * Math.sqrt(k + 0.5) * (1 + 0.16 * (h1(seed + k) - 0.5));
      const th = k * 2.399963 + rot;
      d.add(x + Math.cos(th) * rho, y + Math.sin(th) * rho, size + h1(seed + k * 3) * 0.7, ink, alpha);
    }
    void n;
  }

  /** With kv.map, cached prefixes are measured: hatch exactly that share of the body (to scale). */
  private drawMapCached(d: Dots, b: Body, bi: number): void {
    const sh = this.kvShares(bi);
    if (!sh) return;
    const frac = clamp(sh.cached, 0, 1);
    const w = this.wash!;
    const n = Math.floor(frac * w.x.length);
    const dim = this.follow ? 0.6 : 1;
    for (let i = 0; i < n; i++) {
      const x = b.cx + w.x[i] * b.rx;
      const y = b.cy + w.y[i] * b.ry;
      if (((x + y) / 6) % 2 > 1.05) continue; // diagonal bands: hatched, not solid
      d.add(x, y, 2.0, [58, 80, 172], 0.72 * dim);
    }
  }

  /** Reusable cache, not in use: a hollow outline with sparse hatching, lighter than live colonies, fading as it ages. */
  private drawCached(d: Dots): void {
    const ink: Ink = [78, 98, 184];
    const dim = this.follow ? 0.6 : 1;
    for (const c of this.cachedCol) {
      if (this.mapFor(c.body)?.usedTokens != null) continue; // measured cache is drawn from kv.map instead
      const b = this.bodies[c.body];
      if (!b) continue;
      const age = (this.t - c.born) / c.life;
      const appear = clamp((this.t - c.born) / 0.8, 0, 1);
      const fade = (1 - smooth(clamp(age, 0, 1))) * appear * dim;
      const keep = Math.max(0, 1 - Math.pow(age, 1.6)); // erodes from the outer edge
      const R = this.blockR(c.body, c.blocks * keep);
      if (R < 0.6 || fade <= 0.02) continue;
      const v = this.view;
      if (c.x + R < v.x0 || c.x - R > v.x1 || c.y + R < v.y0 || c.y - R > v.y1) continue;
      // outline
      const n = Math.max(10, Math.floor(R * 2.4 * Math.min(4, this.zoom)));
      for (let k = 0; k < n; k++) {
        const a = (k / n) * Math.PI * 2;
        d.add(c.x + Math.cos(a) * R, c.y + Math.sin(a) * R, 1.8, ink, 0.85 * fade);
      }
      // hatching: every fourth block-dot on diagonal bands only
      const dots = Math.round(c.blocks * keep * b.dpb);
      const rot = (c.seed % 628) / 100;
      for (let k = 0; k < dots; k += 4) {
        const rho = b.sp * Math.sqrt(k + 0.5);
        const th = k * 2.399963 + rot;
        const x = Math.cos(th) * rho;
        const y = Math.sin(th) * rho;
        if (((x + y) / (b.sp * 3.2)) % 2 > 0.9 || rho > R - 1) continue;
        d.add(c.x + x, c.y + y, 1.7, ink, 0.5 * fade);
      }
    }
  }

  private drawCore(d: Dots, f: FamCore): void {
    const dim = this.follow ? 0.6 : 1;
    const b = this.bodies[f.body];
    if (!b) return;
    const dots = Math.max(1, Math.round(f.blocks * b.dpb));
    this.sunflower(d, f.x, f.y, dots, b.sp, f.seed, 0, dots, INK.ultraDeep, 0.97 * dim, 2.4);
    // a fine rim marks the shared prefix
    const R = this.blockR(f.body, f.blocks) + 2.5;
    const n = Math.max(12, Math.floor(R * 2.4));
    for (let k = 0; k < n; k++) {
      const a = (k / n) * Math.PI * 2;
      d.add(f.x + Math.cos(a) * R, f.y + Math.sin(a) * R, 1.2, INK.ultraDeep, 0.75 * dim);
    }
  }

  private drawBridge(d: Dots): void {
    // the KV-transfer channel between prefill and decode bodies
    const pre = this.bodies.filter((b) => b.role === 'prefill');
    const dec = this.bodies.filter((b) => b.role !== 'prefill');
    const busy = clamp(this.transferRate / 120, 0, 1);
    for (const p of pre) {
      for (const q of dec) {
        const pts = polyFrom(bezierPts(p.cx + p.rx * 0.9, p.cy, q.cx - q.rx * 0.9, q.cy, 0.08, 12));
        const cnt = Math.floor(pts.len / 2.2);
        for (let k = 0; k < cnt; k++) {
          const u = (k / cnt) * pts.len;
          at(pts, u, TMP);
          const lat = (h1(k * 7 + p.cx) - 0.5) * (5 + 6 * busy);
          d.add(TMP.x + TMP.nx * lat, TMP.y + TMP.ny * lat, 1.4, INK.ultra, 0.35 + 0.45 * busy);
        }
      }
    }
  }

  private drawTransfers(d: Dots): void {
    for (const tr of this.transfers) {
      if (tr.t < 0) continue;
      const p = smooth(clamp(tr.t / tr.dur, 0, 1));
      const mx = (tr.x0 + tr.x1) / 2 - (tr.y1 - tr.y0) * tr.bend;
      const my = (tr.y0 + tr.y1) / 2 + (tr.x1 - tr.x0) * tr.bend;
      for (let k = 0; k < 3; k++) {
        const q = Math.max(0, p - k * 0.03);
        const u = 1 - q;
        const x = u * u * tr.x0 + 2 * u * q * mx + q * q * tr.x1;
        const y = u * u * tr.y0 + 2 * u * q * my + q * q * tr.y1;
        d.add(x, y, 2.1 - k * 0.5, k ? INK.ultra : INK.ultraDeep, (0.9 - k * 0.3) * (1 - p * 0.3));
      }
    }
  }

  /** The admission membrane: thicker where requests were just admitted, a ripple at each crossing,
   *  bowed and darker where the proxy holds requests against it. */
  private drawMembrane(d: Dots): void {
    const { memTop, memBot } = this.L;
    const n = Math.floor((memBot - memTop) / 2.2);
    for (let i = 0; i <= n; i++) {
      const y = lerp(memTop, memBot, i / n);
      const x = this.membraneX(y);
      const edge = Math.min(1, Math.min(i, n - i) / 18);
      let local = 0; // admissions density near this height (recent crossings)
      for (const c of this.crossings) {
        const u = (y - c.y) / 26;
        local += Math.exp(-u * u) * Math.exp(-(this.t - c.t) / 1.4);
      }
      let pressed = 0;
      for (const p of this.pressing) {
        const u = (y - p.y) / (p.r * 1.6);
        pressed += Math.exp(-u * u);
      }
      const rows = Math.min(5, Math.floor(local * 1.6));
      const ink = pressed > 0.3 ? INK.crimsonDeep : INK.membrane;
      d.add(x + (h1(i * 3) - 0.5) * 2.4, y, 2.5, ink, Math.min(1, 0.9 + 0.1 * pressed) * edge);
      if (i % 2 === 0) d.add(x - 3.8 + (h1(i * 7) - 0.5) * 3, y + 1, 2.0, INK.membrane, 0.62 * edge);
      for (let r = 1; r <= rows; r++) d.add(x + r * 2.6 + (h1(i * 11 + r) - 0.5) * 1.6, y + (h1(i + r) - 0.5) * 2, 2.1, INK.membrane, (0.8 - r * 0.1) * edge);
    }
    // ripple: a short arc spreading into the engine where each request crossed
    for (const c of this.crossings) {
      const p = (this.t - c.t) / 0.7;
      if (p >= 1) continue;
      const R = 5 + 34 * p;
      const x0 = this.membraneX(c.y);
      for (let k = 0; k < 14; k++) {
        const a = -Math.PI / 2.6 + (k / 13) * (Math.PI / 1.3);
        d.add(x0 + Math.cos(a) * R, c.y + Math.sin(a) * R, 2.0, INK.membrane, 0.8 * (1 - p));
      }
    }
  }

  private drawVein(d: Dots, ed: Poly, f: number, seed: number, dim: number): void {
    const busy = clamp(f / 150, 0, 1);
    const width = clamp(0.7 + 1.5 * Math.sqrt(f / 8), 0.7, 16);
    const flow = this.t * (5 + 36 * busy);
    const ink = mixInk(INK.rose, INK.crimson, Math.sqrt(busy));
    // core strand: granular, gaussian across the vein, swelling mid-way
    const core = Math.floor((ed.len / 1.7) * (0.45 + 0.2 * width));
    const alpha = Math.min(0.95, 0.24 + 0.7 * Math.sqrt(busy)) * dim;
    for (let k = 0; k < core; k++) {
      const u = (h1(seed * 131 + k) * ed.len + flow) % ed.len;
      at(ed, u, TMP);
      const swell = 0.75 + 0.35 * Math.sin((u / ed.len) * Math.PI) + 0.15 * Math.sin(u * 0.08 + seed);
      const lat = gaussLike(seed * 7 + k * 3) * width * 0.34 * swell;
      d.add(TMP.x + TMP.nx * lat, TMP.y + TMP.ny * lat, 1.15 + h1(k + seed) * 1.1, ink, alpha);
    }
    // halo: sparse spray around busy veins, like ink feathering into paper
    if (busy > 0.02) {
      const halo = Math.floor((ed.len / 3.5) * (0.3 + 0.12 * width));
      for (let k = 0; k < halo; k++) {
        const u = (h1(seed * 313 + k) * ed.len + flow * 0.6) % ed.len;
        at(ed, u, TMP);
        const lat = (h1(seed * 509 + k * 11) - 0.5) * 2 * (width * 1.5 + 4);
        d.add(TMP.x + TMP.nx * lat, TMP.y + TMP.ny * lat, 1.15, INK.rose, (0.1 + 0.26 * busy) * dim);
      }
    }
  }

  private drawKnot(d: Dots, x: number, y: number, f: number, seed: number, dim: number): void {
    const busy = clamp(f / 150, 0, 1);
    const width = clamp(0.7 + 1.5 * Math.sqrt(f / 8), 0.7, 16);
    const kn = Math.floor(8 + 34 * busy);
    for (let k = 0; k < kn; k++) {
      const a = h1(seed + k * 17) * Math.PI * 2;
      const rr = Math.sqrt(h1(seed + k * 19)) * (width * 0.7 + 2.5);
      d.add(x + Math.cos(a) * rr, y + Math.sin(a) * rr, 1.6, INK.crimsonDeep, (0.3 + 0.55 * busy) * dim);
    }
  }

  private drawNetwork(d: Dots): void {
    const dim = this.follow ? 0.55 : 1;
    for (const b of this.branches.values()) if (b.edge) this.drawVein(d, b.edge, b.flux, b.seed, dim);
    for (const t of this.trunks.values()) if (t.edge) this.drawVein(d, t.edge, t.flux, t.seed, dim);
    for (const b of this.branches.values()) this.drawKnot(d, b.x, b.y, b.flux, b.seed, dim);
    for (const t of this.trunks.values()) this.drawKnot(d, t.x, t.y, t.flux, t.seed, dim);
    this.drawAperture(d);
  }

  /** The mouth where every trunk meets: responses leave here for the clients. */
  private drawAperture(d: Dots): void {
    const { exit, PW } = this.L;
    const busy = clamp(Math.log10(1 + this.aperture) / Math.log10(20001), 0, 1); // log scale up to ~20k tok/s
    const rx = 8 + 4 * busy;
    const ry = 18 + 8 * busy;
    // lips: open towards the duct on the right
    const n = 90;
    for (let i = 0; i < n; i++) {
      const a = Math.PI * 0.32 + (i / n) * Math.PI * 1.36;
      const thick = 0.55 + 0.45 * Math.abs(Math.cos(a));
      for (let l = 0; l < 2; l++) {
        const rr = 1 + (h1(i * 5 + l) - 0.5) * 0.18 + l * 0.12;
        d.add(exit.x + Math.cos(a) * rx * rr, exit.y + Math.sin(a) * ry * rr, 1.5 + thick * 0.8, INK.crimsonDeep, 0.85 * thick);
      }
    }
    // throat: density follows the measured outflow
    const core = Math.floor(10 + 60 * busy);
    for (let k = 0; k < core; k++) {
      const a = h1(k * 7 + 3) * Math.PI * 2;
      const rr = Math.sqrt(h1(k * 11 + 5));
      d.add(exit.x + Math.cos(a) * rx * 0.7 * rr, exit.y + Math.sin(a) * ry * 0.7 * rr, 1.4, INK.crimson, 0.25 + 0.5 * busy);
    }
    // outlet duct to the clients
    const x0 = exit.x + rx * 0.6;
    const x1 = PW - 6;
    const cnt = Math.floor((x1 - x0) / 2);
    const flow = this.t * (10 + 60 * busy);
    for (let k = 0; k < cnt; k++) {
      const x = x0 + (k / cnt) * (x1 - x0);
      const fadeOut = clamp((x1 - x) / 30, 0, 1);
      d.add(x, exit.y - 5.5 + (h1(k) - 0.5), 1.4, INK.crimsonDeep, 0.7 * fadeOut);
      d.add(x, exit.y + 5.5 + (h1(k + 99) - 0.5), 1.4, INK.crimsonDeep, 0.7 * fadeOut);
      const fx = x0 + (((k * 7.3 + flow) % (x1 - x0)) + (x1 - x0)) % (x1 - x0);
      d.add(fx, exit.y + (h1(k * 3) - 0.5) * 7, 1.2, INK.crimson, (0.1 + 0.4 * busy) * clamp((x1 - fx) / 30, 0, 1));
    }
  }

  private drawCapillary(d: Dots, e: Entity): void {
    if (!e.capillary || !e.branch || e.flux < 0.05) return;
    const dim = this.follow && this.follow !== e.req.rid ? 0.35 : 1;
    const p = e.capillary;
    const width = clamp(0.3 + 0.55 * Math.sqrt(e.flux / 6), 0.3, 4);
    const busy = clamp(e.flux / 60, 0, 1);
    const count = Math.floor((p.len / 2.6) * (0.4 + 0.3 * width));
    const flow = this.t * (8 + 30 * busy);
    for (let k = 0; k < count; k++) {
      const u = (h1(e.seed + k * 13) * p.len + flow) % p.len;
      at(p, u, TMP);
      const lat = (h1(e.seed + k * 29) - 0.5) * 2 * width * 0.7;
      const x = TMP.x + TMP.nx * lat;
      const y = TMP.y + TMP.ny * lat;
      if (this.underOther(x, y, e.seed)) continue; // the vein passes beneath other colonies
      d.add(x, y, 1.2 + h1(e.seed + k) * 0.6, INK.crimson, (0.28 + 0.6 * busy) * (1 - e.fade) * dim);
    }
    if (!this.memMap) {
      // root: where the vein leaves its own colony
      at(p, 0, TMP);
      for (let k = 0; k < 5; k++) {
        const a = (k / 5) * Math.PI * 2 + e.seed;
        d.add(TMP.x + Math.cos(a) * 1.6, TMP.y + Math.sin(a) * 1.6, 1.6, INK.crimsonDeep, 0.85 * (1 - e.fade) * dim);
      }
    }
  }

  private drawPatch(d: Dots, e: Entity): void {
    if (!e.patch) return;
    if (this.memMap && this.mapFor(e.body)) return; // its blocks are drawn in the memory map
    const ob = this.bodies[e.oldBody];
    if (e.oldFade > 0 && ob) {
      const od = Math.max(1, Math.round(e.oldBlocks * ob.dpb));
      this.sunflower(d, e.oldPx, e.oldPy, od, ob.sp, e.seed, 0, od, INK.ultra, 0.85 * e.oldFade);
    }
    const dim = this.follow && this.follow !== e.req.rid ? 0.45 : 1;
    const a = smooth(clamp(e.dissolve, 0, 1)) * (1 - e.fade) * smooth(e.arrive) * dim;
    if (a <= 0.01) return;
    const b = this.bodies[e.body];
    if (!b) return;
    const missing = e.flash?.kind === 'miss' ? clamp(1 - e.flash.t / 2.2, 0, 1) : 0;
    // inner: cached prefix (deep) or a missed duplicate core (flashes crimson); then prompt; then decode growth
    this.sunflower(d, e.px, e.py, e.dots, b.sp, e.seed, 0, e.innerDots, missing > 0 ? mixInk(INK.ultraDeep, INK.crimson, missing) : INK.ultraDeep, 0.98 * a, 2.3);
    this.sunflower(d, e.px, e.py, e.dots, b.sp, e.seed, e.innerDots, Math.max(e.innerDots, e.promptDots), INK.ultra, 0.97 * a, 2.4);
    this.sunflower(d, e.px, e.py, e.dots, b.sp, e.seed, Math.max(e.innerDots, e.promptDots), e.dots, INK.violet, 0.95 * a, 2.4);
    // read shimmer: decode steps read the whole context; brighter for long contexts at high step rates
    if (e.req.phase === 'decode') {
      const sps = this.store.engine?.steps_per_s ?? 0;
      const read = (promptTokens(e.req) + e.req.chunks) * sps;
      const g = clamp((Math.log10(Math.max(1, read)) - 3) / 4, 0, 1);
      if (g > 0.02) {
        const rot = (e.seed % 628) / 100;
        for (let k = 0; k < e.dots; k += 3) {
          const w = 0.5 + 0.5 * Math.sin(this.t * 9 + k * 0.9 + e.seed);
          const rho = b.sp * Math.sqrt(k + 0.5);
          const th = k * 2.399963 + rot;
          d.add(e.px + Math.cos(th) * rho, e.py + Math.sin(th) * rho, 1.9, [214, 224, 250], 0.55 * g * w * a);
        }
      }
    }
    // hit / miss ring
    if (e.flash) {
      const p = e.flash.t / 2.2;
      const R = e.pr + 3 + 10 * p;
      const n = Math.max(14, Math.floor(R * 2.6));
      const ink = e.flash.kind === 'hit' ? INK.ultraDeep : INK.crimson;
      for (let k = 0; k < n; k++) {
        const ang = (k / n) * Math.PI * 2;
        d.add(e.px + Math.cos(ang) * R, e.py + Math.sin(ang) * R, 1.6, ink, 0.85 * (1 - p));
      }
    }
  }

  private drawCell(d: Dots, e: Entity, now: number): void {
    const r = e.req;
    const dz = smooth(clamp(e.dissolve, 0, 1));
    const fade = r.phase === 'failed' ? 1 - e.fade : 1;
    const a = (1 - dz) * fade * (this.follow && this.follow !== r.rid ? 0.45 : 1);
    if (a <= 0.02) return;
    const failed = r.phase === 'failed';
    // real queue time since arrival darkens and thickens the rim
    const waited = r.phase === 'queued' ? (now - r.lArrive) / 1000 : 0;
    const strain = clamp(waited / 4, 0, 1);
    const rimInk = failed ? INK.greyDeep : mixInk(INK.crimson, INK.crimsonDeep, strain);
    const bodyInk = failed ? INK.grey : mixInk(classInk(trafficClass(r)), INK.crimson, 0.15 + 0.35 * strain);
    const rad = e.r * e.qs * (1 - 0.5 * dz);
    const sd = e.seed;
    const wob = this.t * 0.8;
    const shape = (ang: number) => 1 + 0.07 * Math.sin(ang * 3 + sd) + 0.045 * Math.sin(ang * 5 + sd * 0.7) + 0.03 * Math.sin(wob + ang * 2 + sd);
    // rim: dense, irregular membrane
    const rimN = Math.floor(rad * (7.5 + 3 * strain));
    for (let k = 0; k < rimN; k++) {
      const ang = (k / rimN) * Math.PI * 2 + h1(sd + k) * 0.18;
      const rr = rad * (0.88 + 0.14 * h1(sd + k * 5)) * shape(ang);
      d.add(e.x + Math.cos(ang) * rr, e.y + Math.sin(ang) * rr, 1.6 + h1(sd + k * 9) * 1.0, rimInk, 0.96 * a);
    }
    // body: stippled, denser towards the rim like the reference colonies
    const bodyN = Math.floor(rad * rad * 0.62);
    for (let k = 0; k < bodyN; k++) {
      const ang = h1(sd + k * 17) * Math.PI * 2;
      const rr = Math.pow(h1(sd + k * 23), 0.38) * rad * 0.84 * shape(ang);
      d.add(e.x + Math.cos(ang) * rr, e.y + Math.sin(ang) * rr, 1.2 + h1(sd + k * 29) * 0.8, bodyInk, (0.5 + 0.25 * strain) * a);
    }
    // nucleus
    const nx = e.x + (h1(sd + 3) - 0.5) * rad * 0.25;
    const ny = e.y + (h1(sd + 4) - 0.5) * rad * 0.25;
    // nucleus ∝ prompt length: a big core is a prefill-heavy request
    const nucR = clamp(Math.sqrt(promptTokens(r)) * 0.14, 1.6, rad * 0.78);
    const nucN = Math.floor(nucR * nucR * 1.3 + 4);
    for (let k = 0; k < nucN; k++) {
      const ang = h1(sd + k * 31) * Math.PI * 2;
      const rr = Math.sqrt(h1(sd + k * 37)) * nucR;
      d.add(nx + Math.cos(ang) * rr, ny + Math.sin(ang) * rr, 1.6, failed ? INK.greyDeep : INK.crimsonDeep, 0.88 * a);
    }
    if (e.dissolve > 0.02 && e.patch) {
      // condensing into tissue: ultramarine grains appear inside the cell
      const g = Math.floor(rad * 3 * e.dissolve);
      for (let k = 0; k < g; k++) {
        const ang = h1(sd + k * 41) * Math.PI * 2;
        const rr = Math.sqrt(h1(sd + k * 43)) * rad * 0.85;
        d.add(e.x + Math.cos(ang) * rr, e.y + Math.sin(ang) * rr, 1.7, INK.ultra, 0.92 * a);
      }
    }
  }

  /** Whether a point lies under a colony or core other than the one with this seed. */
  private underOther(x: number, y: number, seed: number): boolean {
    for (const [cx, cy, r, sd] of this.discs) if (sd !== seed && (x - cx) ** 2 + (y - cy) ** 2 < r * r) return true;
    return false;
  }

  private drawSparks(d: Dots): void {
    const fr = this.follow ? this.ents.get(this.follow)?.route : undefined;
    for (const s of this.sparks) {
      if (s.s < 0) continue;
      const len = routeLen(s.route);
      if (!len) continue;
      const dim = this.follow && s.route !== fr ? 0.4 : 1;
      const end = clamp((len + 20 - s.s) / 40, 0, 1) * dim;
      atRoute(s.route, s.s, TMP);
      if (this.underOther(TMP.x, TMP.y, this.ents.get(s.rid)?.seed ?? -2)) continue; // passing beneath another colony
      d.add(TMP.x, TMP.y, s.size, s.ink, 0.92 * end);
      atRoute(s.route, s.s - 4, TMP);
      d.add(TMP.x, TMP.y, s.size * 0.75, s.ink, 0.62 * end);
      atRoute(s.route, s.s - 8, TMP);
      d.add(TMP.x, TMP.y, s.size * 0.55, s.ink, 0.2 * end);
    }
  }

  private drawSpores(d: Dots): void {
    const edge = this.L.PW - 8;
    for (const s of this.spores) {
      const age = (this.t - s.born) / s.life;
      const edgeFade = clamp((edge - s.r - s.x) / 36, 0, 1); // never drift over the notes column
      const a = (age < 0.15 ? age / 0.15 : 1 - smooth((age - 0.15) / 0.85)) * edgeFade;
      if (a <= 0.01) continue;
      const n = Math.floor(s.r * 4);
      for (let k = 0; k < n; k++) {
        const ang = (k / n) * Math.PI * 2 + h1(s.seed + k) * 0.3;
        const rr = s.r * (0.85 + 0.2 * h1(s.seed + k * 3));
        d.add(s.x + Math.cos(ang) * rr, s.y + Math.sin(ang) * rr, 1.5, s.ink, 0.75 * a);
      }
    }
  }

  private drawOrgans(d: Dots, live: boolean): void {
    for (const o of this.organs) {
      const u = o.util;
      const ink = live ? INK.sap : INK.grey;
      const inkL = live ? INK.sapLight : INK.grey;
      const act = live ? 0.25 + 0.75 * u : 0.35;
      // soma: dense stippled core with a soft halo
      const somaR = 13 + 7 * (live ? u : 0);
      const n = 130 + Math.floor(170 * (live ? u : 0));
      for (let k = 0; k < n; k++) {
        const ang = h1(o.seed + k * 3) * Math.PI * 2;
        const rr = Math.abs(gaussLike(o.seed + k * 5)) * somaR * 0.55;
        d.add(o.x + Math.cos(ang) * rr, o.y + Math.sin(ang) * rr, 1.4 + h1(o.seed + k) * 0.9, ink, 0.75 * act);
      }
      // dendrites
      for (let b = 0; b < o.dendrites.length; b++) {
        const p = o.dendrites[b];
        const cnt = Math.floor(p.len / 1.6);
        for (let k = 0; k < cnt; k++) {
          const u2 = (k / cnt) * p.len;
          at(p, u2, TMP);
          const taper = 1 - (u2 / p.len) * 0.65;
          const lat = (h1(o.seed + b * 97 + k) - 0.5) * 2.4 * taper;
          d.add(TMP.x + TMP.nx * lat, TMP.y + TMP.ny * lat, 1.5 * taper + 0.4, k % 3 ? inkL : ink, (0.42 + 0.5 * act) * taper);
        }
        const [tx, ty] = o.tips[b];
        for (let k = 0; k < 6; k++) {
          const ang = (k / 6) * Math.PI * 2;
          d.add(tx + Math.cos(ang) * 2.2, ty + Math.sin(ang) * 2.2, 1.1, inkL, 0.35 * act);
        }
      }
    }
    if (!live) return;
    for (const g of this.signals) {
      at(g.path, g.s, TMP);
      d.add(TMP.x, TMP.y, 2.8, INK.sap, 0.95);
      at(g.path, g.s - 3, TMP);
      d.add(TMP.x, TMP.y, 2.0, INK.sapLight, 0.65);
      at(g.path, g.s - 6, TMP);
      d.add(TMP.x, TMP.y, 1.5, INK.sapLight, 0.25);
    }
  }

  // ---- queries for the overlay ---------------------------------------------------------------

  pick(x: number, y: number): Req | null {
    let best: Entity | null = null;
    let bestD = Infinity;
    for (const e of this.ents.values()) {
      const cellVisible = e.dissolve < 0.6;
      const cx = cellVisible ? e.x : e.px;
      const cy = cellVisible ? e.y : e.py;
      const rr = cellVisible ? e.r * e.qs : Math.max(e.pr, 6);
      const dd = Math.hypot(cx - x, cy - y);
      if (dd < rr + 5 && dd < bestD) {
        bestD = dd;
        best = e;
      }
    }
    return best?.req ?? null;
  }

  bodyAnchors(): { x: number; y: number; name: string; role: string; side: boolean }[] {
    // stacked ranks are labelled at their left flank; PD columns above each body
    const side = !this.pd && this.bodies.length > 1;
    return this.bodies.map((b) => ({
      x: side ? b.cx - b.rx * 1.08 - 10 : b.cx,
      y: side ? b.cy - 14 : b.cy - b.ry - 30,
      name: b.name, role: b.role, side,
    }));
  }

  /** Labels for the measured hierarchy: one per branch root (class) and per trunk (instance). */
  netLabels(): { id: string; kind: 'branch' | 'trunk'; x: number; y: number; name: string; flux: number; multi: boolean }[] {
    const multi = this.trunks.size > 1;
    const out: { id: string; kind: 'branch' | 'trunk'; x: number; y: number; name: string; flux: number; multi: boolean }[] = [];
    for (const b of this.branches.values()) {
      if (b.flux < 0.5 && b.target === 0) continue;
      out.push({ id: b.key, kind: 'branch', x: b.x, y: b.y, name: classLabel(b.name), flux: b.flux, multi });
    }
    for (const t of this.trunks.values()) out.push({ id: `trunk:${t.key}`, kind: 'trunk', x: t.x, y: t.y, name: t.name, flux: t.flux, multi });
    return out;
  }

  /** Points the stage captions lead to. */
  stageAnchors() {
    const L = this.L;
    const b0 = this.bodies[0];
    let vein: { x: number; y: number } | null = null;
    let best = -1;
    for (const b of this.branches.values()) {
      if (b.edge && b.flux > best) {
        best = b.flux;
        at(b.edge, b.edge.len * 0.5, TMP);
        vein = { x: TMP.x, y: TMP.y };
      }
    }
    if (!vein) vein = { x: lerp(L.net.x0, L.exit.x, 0.5), y: L.exit.y };
    return {
      arrive: { x: L.memX - 34, y: L.memTop + 30 },
      membrane: { x: this.membraneX(L.memTop + 14) + 1, y: L.memTop + 14 },
      body: b0 ? { x: b0.cx, y: b0.cy - b0.ry + 4 } : { x: L.kv.cx, y: L.kv.cy - L.kv.ry },
      vein,
      exit: { x: L.exit.x, y: L.exit.y - 20 },
    };
  }

  /**
   * Live / cached / free shares of a body's KV, in tokens, all measured:
   *   used   = the allocator's used tokens (kv.map)
   *   live   = the engine's own KV usage, which counts tokens held by running requests once each
   *   cached = used − live (prefixes kept for reuse, evictable)
   * Without the engine's usage figure, falls back to the map's cells (coarser: a cell counts as live
   * if any of its tokens are).
   */
  kvShares(bi: number): { live: number; cached: number; free: number } | null {
    const m = this.mapFor(bi);
    if (!m) return null;
    if (m.tokens && m.capacityTokens) {
      // format 2: every slot classified by the engine's own tables, each slot once
      const t = m.tokens;
      const cap = m.capacityTokens;
      return { live: (t.own + t.reused + t.shared) / cap, cached: t.cached / cap, free: t.free / cap };
    }
    const usage = this.bodies.length === 1 ? this.store.engine?.kv_usage : undefined;
    if (m.capacityTokens && m.usedTokens != null && usage != null) {
      const used = clamp(m.usedTokens / m.capacityTokens, 0, 1);
      const live = Math.min(usage, used);
      return { live, cached: used - live, free: 1 - used };
    }
    if (m.state && m.bins) {
      let live = 0;
      let cached = 0;
      let free = 0;
      for (let i = 0; i < m.bins; i++) {
        const v = m.state[i];
        if (v === -1) free++;
        else if (v === -2) cached++;
        else live++;
      }
      return { live: live / m.bins, cached: cached / m.bins, free: free / m.bins };
    }
    return null;
  }

  /** Large shared prefixes on the plate (colony view): where they are, how big, how many requests reuse them. */
  familyLabels(): { id: string; x: number; y: number; tokens: number; members: number; measured: boolean }[] {
    // one label (the largest shared prefix), single-engine plates only: several would crowd the body labels
    if (this.memMap || this.bodies.length > 1) return [];
    const out: { id: string; x: number; y: number; tokens: number; members: number; measured: boolean }[] = [];
    for (const f of this.famCores.values()) {
      if (!this.bodies[f.body] || f.members < 2 || f.tokens < 1000) continue;
      out.push({ id: f.id, x: f.x, y: f.y + this.blockR(f.body, f.blocks) + 10, tokens: f.tokens, members: f.members, measured: f.measured });
    }
    return out.sort((a, b) => b.tokens - a.tokens).slice(0, 1);
  }

  /** Memory-map view: where the shared-prefix cells sit (left of the body, at their mean height) and how many tokens. */
  mapSharedLabel(): { x: number; y: number; share: number; tokens: boolean } | null {
    if (!this.memMap) return null;
    const m = this.mapFor(0);
    const b = this.bodies[0];
    if (!m || !b) return null;
    const g = this.cellGeo(0, m.bins);
    if (!g) return null;
    if (m.levels && m.tokens && m.capacityTokens) {
      // format 2: the share of tokens held by two or more running requests, placed at those cells' mean height
      const share = m.tokens.shared / m.capacityTokens;
      if (share < 0.01) return null;
      let w = 0;
      let sy = 0;
      for (let i = 0; i < m.bins && i < g.y.length; i++) {
        w += m.levels.shared[i];
        sy += m.levels.shared[i] * g.y[i];
      }
      return w ? { x: b.cx - b.rx - 14, y: sy / w, share, tokens: true } : null;
    }
    let n = 0;
    let sy = 0;
    for (let i = 0; i < m.bins && i < g.y.length; i++) {
      if (m.state[i] === -3) {
        n++;
        sy += g.y[i];
      }
    }
    if (n < m.bins * 0.03) return null;
    return { x: b.cx - b.rx - 14, y: sy / n, share: n / m.bins, tokens: false };
  }

  /** kv.map for a body (single-body plates accept the only map there is). */
  mapFor(bi: number): KvMap | undefined {
    const b = this.bodies[bi];
    if (!b) return undefined;
    return this.store.kvMaps.get(b.name) ?? (this.bodies.length === 1 ? [...this.store.kvMaps.values()][0] : undefined);
  }

  cellGeo(bi: number, bins: number): CellGeo | null {
    const b = this.bodies[bi];
    if (!b || bins <= 0) return null;
    const key = `${bins}:${b.cx.toFixed(1)}:${b.cy.toFixed(1)}:${b.rx.toFixed(1)}:${b.ry.toFixed(1)}`;
    const g0 = this.geo.get(bi);
    if (g0 && g0.key === key) return g0;
    let c = Math.sqrt((Math.PI * b.rx * b.ry) / bins);
    for (let tries = 0; tries < 60; tries++) {
      const rows: CellGeo['rows'] = [];
      let n = 0;
      for (let y = b.cy - b.ry + c / 2; y < b.cy + b.ry; y += c) {
        const h = Math.sqrt(Math.max(0, 1 - ((y - b.cy) / b.ry) ** 2)) * b.rx;
        const cnt = Math.floor((2 * h) / c);
        if (cnt <= 0) continue;
        rows.push({ y, x0: b.cx - (cnt * c) / 2 + c / 2, start: n, n: cnt });
        n += cnt;
      }
      if (n >= bins) {
        const x = new Float32Array(bins);
        const yy = new Float32Array(bins);
        for (const r of rows) for (let i = 0; i < r.n && r.start + i < bins; i++) {
          x[r.start + i] = r.x0 + i * c;
          yy[r.start + i] = r.y;
        }
        const g = { key, c, x, y: yy, rows };
        this.geo.set(bi, g);
        return g;
      }
      c *= 0.985;
    }
    return null;
  }

  private ownerInk(rid: string): Ink {
    const h = h1(hashStr(rid));
    return mixInk(INK.ultra, h < 0.5 ? INK.violet : [40, 70, 160], Math.abs(h - 0.5) * 1.4);
  }

  /** The KV pool as the engine lays it out: one cell per bin. */
  private drawMemMap(d: Dots, bi: number, now: number): void {
    const m0 = this.mapFor(bi);
    if (m0?.levels) {
      this.drawMemMap2(d, bi, m0, now);
      return;
    }
    const m = m0;
    const g = this.cellGeo(bi, m?.bins ?? 4096);
    if (!g) return;
    const v = this.view;
    const zc = Math.pow(this.zoom, 0.45); // cells scale exactly with zoom (the shader applies k^0.55)
    const full = g.c * 0.86 * zc;
    const cachedInk: Ink = [104, 124, 196];
    const grid = this.zoom > 2.5; // free cells are bare paper; the grid itself only shows up close
    const sums = new Map<number, [number, number, number]>();
    for (let i = 0; i < g.x.length; i++) {
      const x = g.x[i];
      const y = g.y[i];
      const st = m ? m.state[i] : -1;
      if (m && st >= 0 && !m.prefix[i]) {
        const acc = sums.get(st) ?? [0, 0, 0];
        acc[0] += x;
        acc[1] += y;
        acc[2]++;
        sums.set(st, acc);
      }
      if (x < v.x0 - 4 || x > v.x1 + 4 || y < v.y0 - 4 || y > v.y1 + 4) continue;
      const appear = m ? clamp((now - m.changedAt[i]) / 600, 0, 1) : 1;
      if (st === -1) {
        if (grid) d.add(x, y, full * 0.25, INK.ultraPale, 0.12);
        // just freed: let the old owner fade out
        const ps = m?.prevState?.[i] ?? -1;
        if (appear < 1 && ps >= 0) d.addSquare(x, y, full, this.ownerInk(m!.prevOwners[ps].rid), 0.9 * (1 - appear));
      } else if (st === -2) {
        // cached radix prefix: used, owned by no running request — diagonal hatching
        const a = 0.8 * appear + 0.12;
        d.addSquare(x, y, full, [212, 218, 238], 0.85 * a); // pale square, then the hatch on top
        const nh = 3 + Math.floor(Math.min(6, this.zoom / 2)); // finer stroke when zoomed in
        const sz = full * (0.3 / Math.sqrt(nh / 3));
        for (let q = 0; q < nh; q++) {
          const o = (q / (nh - 1) - 0.5) * 0.56 * g.c;
          d.add(x + o, y + o, sz, cachedInk, a);
        }
      } else if (st === -3) {
        // a prefix that two or more running requests share right now (engine-reported)
        d.addSquare(x, y, full, INK.ultraDeep, 0.97 * appear + 0.03);
      } else if (st === -4 && m) {
        // co-located: several requests' own tokens fall in this one cell — drawn split, not as a core
        const ks = m.shared.get(i) ?? [];
        const a = 0.9 * appear + 0.06;
        d.addSquare(x - g.c * 0.22, y, full * 0.46, ks[0] != null ? this.ownerInk(m.owners[ks[0]].rid) : INK.ultra, a);
        d.addSquare(x + g.c * 0.22, y, full * 0.46, ks[1] != null ? this.ownerInk(m.owners[ks[1]].rid) : INK.ultra, a);
      } else if (m) {
        const ink = this.ownerInk(m.owners[st].rid);
        // a single request's reused prefix (from the radix cache, not shared right now): mid-deep tone
        d.addSquare(x, y, full, m.prefix[i] ? mixInk(ink, INK.ultraDeep, 0.55) : ink, 0.94 * appear + 0.04);
      }
    }
    if (m) for (const [k, [sx, sy, n]] of sums) this.mapCentroid.set(m.owners[k].rid, [sx / n, sy / n]);
  }

  /**
   * kv.map format 2: each cell is split into sub-squares in proportion to its tokens by kind: shared (deepest),
   * reused from the cache (deep), own (ultramarine), cached for reuse (pale, hatched), free (bare paper).
   * Cells the followed request has tokens in carry a crimson dot.
   */
  private drawMemMap2(d: Dots, bi: number, m: KvMap, now: number): void {
    const g = this.cellGeo(bi, m.bins);
    const L = m.levels;
    if (!g || !L) return;
    const v = this.view;
    const zc = Math.pow(this.zoom, 0.45);
    const n = this.zoom > 2.5 ? 3 : 2; // n × n sub-squares per cell
    const sub = (g.c / n) * 0.9 * zc;
    const kinds: [Uint8Array, Ink, number][] = [
      [L.shared, INK.ultraDeep, 0.97], [L.reused, [30, 44, 120], 0.95], [L.own, INK.ultra, 0.9], [L.cached, [176, 188, 226], 0.85],
    ];
    const cachedInk: Ink = [104, 124, 196];
    const fol = this.follow ? m.owners.find((o) => o.rid === this.follow) : undefined;
    const slots = n * n;
    const take = new Int32Array(5);
    if (this.cent?.m !== m || this.cent.g !== g) {
      // where each request's tokens sit on average (veins leave from there in this view); once per map
      const map = new Map<string, [number, number]>();
      for (const o of m.owners) {
        let sx = 0;
        let sy = 0;
        let k = 0;
        for (let i = 0; i < g.x.length && i < m.bins; i++) {
          if (!ownerInCell(o, i)) continue;
          sx += g.x[i];
          sy += g.y[i];
          k++;
        }
        if (k) map.set(o.rid, [sx / k, sy / k]);
      }
      this.cent = { m, g, map };
    }
    for (const [rid, c] of this.cent.map) this.mapCentroid.set(rid, c);
    for (let i = 0; i < g.x.length && i < m.bins; i++) {
      const x = g.x[i];
      const y = g.y[i];
      if (x < v.x0 - 4 || x > v.x1 + 4 || y < v.y0 - 4 || y > v.y1 + 4) continue;
      const appear = clamp((now - m.changedAt[i]) / 600, 0, 1) * 0.85 + 0.15;
      // largest-remainder split of the n² sub-squares over the five kinds (free stays bare paper)
      const lv = [kinds[0][0][i], kinds[1][0][i], kinds[2][0][i], kinds[3][0][i], L.free[i]];
      const tot = lv[0] + lv[1] + lv[2] + lv[3] + lv[4] || 1;
      let used = 0;
      const rem: [number, number][] = [];
      for (let k = 0; k < 5; k++) {
        const exact = (lv[k] / tot) * slots;
        take[k] = Math.floor(exact);
        used += take[k];
        rem.push([exact - take[k], k]);
      }
      rem.sort((a2, b2) => b2[0] - a2[0]);
      for (let r = 0; used < slots && r < 5; r++, used++) take[rem[r][1]]++;
      let q = 0;
      for (let k = 0; k < 4; k++) {
        for (let t = 0; t < take[k] && q < slots; t++, q++) {
          const sx = x + ((q % n) - (n - 1) / 2) * (g.c / n);
          const sy = y + (Math.floor(q / n) - (n - 1) / 2) * (g.c / n);
          d.addSquare(sx, sy, sub, kinds[k][1], kinds[k][2] * appear);
          if (k === 3) d.add(sx, sy, sub * 0.35, cachedInk, 0.8 * appear); // hatch mark: kept for reuse
        }
      }
      if (fol && ownerInCell(fol, i)) d.add(x, y, Math.max(2.4, g.c * 0.5 * zc), INK.crimson, 0.95);
    }
  }

  /** Which memory-map cell is under a world point. */
  cellAt(bi: number, wx: number, wy: number): number | null {
    const m = this.mapFor(bi);
    const g = this.cellGeo(bi, m?.bins ?? 4096);
    if (!g) return null;
    for (const r of g.rows) {
      if (Math.abs(wy - r.y) > g.c / 2) continue;
      const i = Math.round((wx - r.x0) / g.c);
      if (i < 0 || i >= r.n) return null;
      const b = r.start + i;
      return b < g.x.length ? b : null;
    }
    return null;
  }

  /** Drop all animated state (the store is being rebuilt from the event buffer). */
  reset(): void {
    this.ents.clear();
    this.sparks = [];
    this.spores = [];
    this.signals = [];
    this.transfers = [];
    this.cachedCol = [];
    this.famCores.clear();
    this.flecks = [];
    this.branches.clear();
    this.follow = null;
  }

  /** What is under a world point: the most specific component first. */
  componentAt(wx: number, wy: number): Hit | null {
    if (this.memMap) {
      for (let i = 0; i < this.bodies.length; i++) {
        const b = this.bodies[i];
        if (((wx - b.cx) / b.rx) ** 2 + ((wy - b.cy) / b.ry) ** 2 > 1.02) continue;
        const cell = this.cellAt(i, wx, wy);
        if (cell != null) {
          const g = this.geo.get(i)!;
          const x = g.x[cell];
          const y = g.y[cell];
          return { kind: 'cell', id: `${i}:${cell}`, box: { x0: x - 40, y0: y - 30, x1: x + 40, y1: y + 30 } };
        }
      }
    }
    const req = this.pick(wx, wy);
    if (req) {
      const e = this.ents.get(req.rid)!;
      const cx = e.dissolve < 0.6 ? e.x : e.px;
      const cy = e.dissolve < 0.6 ? e.y : e.py;
      const r = Math.max(e.dissolve < 0.6 ? e.r : e.pr, 10);
      return { kind: 'request', id: req.rid, box: { x0: cx - r, y0: cy - r, x1: cx + r, y1: cy + r } };
    }
    for (const f of this.famCores.values()) {
      const r = this.blockR(f.body, f.blocks) + 4;
      if (Math.hypot(wx - f.x, wy - f.y) < r) return { kind: 'core', id: f.id, box: { x0: f.x - r * 3, y0: f.y - r * 3, x1: f.x + r * 3, y1: f.y + r * 3 } };
    }
    const E = 0; // expert routing lives in Fig. 4, not on the plate
    if (E) {
      const { cx, cy, r } = this.L.moe;
      const st = this.moeStrip();
      const box = { x0: cx - r * 1.6, y0: st.y - 70, x1: cx + r * 1.6, y1: cy + r + 20 };
      if (wx >= st.x0 && wx <= st.x1 && wy >= st.y - 26 && wy <= st.y + 3) {
        const n = this.store.moe?.layer_imbalance?.length ?? 0;
        const l = n ? Math.floor(((wx - st.x0) / (st.x1 - st.x0)) * n) : 0;
        return { kind: 'layer', id: String(clamp(l, 0, Math.max(0, n - 1))), box };
      }
      const d = Math.hypot(wx - cx, wy - cy);
      if (Math.abs(d - r) < 9) {
        let a = Math.atan2(wy - cy, wx - cx) + Math.PI / 2;
        if (a < 0) a += Math.PI * 2;
        return { kind: 'expert', id: String(Math.round((a / (Math.PI * 2)) * E) % E), box };
      }
      if (d < r) return { kind: 'moe', id: 'moe', box };
    }
    for (const b of this.branches.values()) {
      if (Math.hypot(wx - b.x, wy - b.y) < 16) return { kind: 'branch', id: b.key, box: { x0: b.x - 60, y0: b.y - 50, x1: b.x + 160, y1: b.y + 50 } };
    }
    for (const t of this.trunks.values()) {
      if (Math.hypot(wx - t.x, wy - t.y) < 18) return { kind: 'trunk', id: t.key, box: { x0: t.x - 80, y0: t.y - 80, x1: this.L.exit.x + 30, y1: t.y + 80 } };
    }
    for (let i = 0; i < this.organs.length; i++) {
      const o = this.organs[i];
      if (Math.hypot(wx - o.x, wy - o.y) < 34) return { kind: 'gpu', id: String(i), box: { x0: o.x - 70, y0: o.y - 150, x1: o.x + 70, y1: o.y + 40 } };
    }
    for (let i = 0; i < this.bodies.length; i++) {
      const b = this.bodies[i];
      if (((wx - b.cx) / b.rx) ** 2 + ((wy - b.cy) / b.ry) ** 2 < 1.05) {
        return { kind: 'body', id: b.name, box: { x0: b.cx - b.rx, y0: b.cy - b.ry, x1: b.cx + b.rx, y1: b.cy + b.ry } };
      }
    }
    return null;
  }

  moeAnchor(): { x: number; y: number; r: number } {
    return { x: this.L.moe.cx, y: this.L.moe.cy, r: this.L.moe.r };
  }

  bridgeAnchor(): { x: number; y: number } | null {
    const p = this.bodies.find((b) => b.role === 'prefill');
    const q = this.bodies.find((b) => b.role !== 'prefill');
    if (!this.pd || !p || !q) return null;
    return { x: (p.cx + p.rx * 0.9 + q.cx - q.rx * 0.9) / 2, y: Math.max(p.cy + p.ry, q.cy + q.ry) + 14 };
  }

  organAnchors(): { x: number; y: number }[] {
    return this.organs.map((o) => ({ x: o.x, y: o.y }));
  }
}

/** Area fraction for a KV occupancy: log scale, so 0.1% and 10% are both visible. */
export function kvArea(kv: number): number {
  return clamp(Math.log10(1 + 999 * clamp(kv, 0, 1)) / 3, 0, 1);
}

function normUrl(u?: string | null): string | null {
  if (!u) return null;
  return String(u).trim().replace(/\/+$/, '').replace(/^https?:\/\//, '').replace(/^localhost(?=:)/, '127.0.0.1');
}

function gaussLike(seed: number): number {
  // sum of uniforms ≈ gaussian, stable per seed
  return (h1(seed) + h1(seed + 1) + h1(seed + 2) + h1(seed + 3) - 2) * 1.2;
}
