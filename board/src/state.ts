// The board's model of the world, built only from received events.
import type { AdvisorNote, Coverage, EngineStep, EngineTick, FinishedReq, GpuTick, Hello, InstanceInfo, Msg } from './contract';

export type Phase = 'queued' | 'prefill' | 'decode' | 'done' | 'failed';

export interface Req {
  rid: string;
  workload?: string;
  partial: boolean; // joined mid-flight (no arrive event seen)
  // source times (ms since epoch, from t_ns)
  tArrive: number;
  tAdmit?: number;
  tFirst?: number;
  tDone?: number;
  // local receive times (performance.now ms) for animation
  lArrive: number;
  lAdmit?: number;
  lFirst?: number;
  lDone?: number;
  promptChars?: number;
  maxTokens?: number;
  promptTokens?: number;
  cachedTokens?: number;
  completionTokens?: number;
  chunks: number;
  engineTtftMs?: number;
  finish?: string;
  phase: Phase;
  pendingSparks: number; // chunks the scene has not drawn yet
  lastAt: number; // local time of the last event for this request
  engine?: FinishedReq; // engine-side breakdown when the tap reports it
  error?: string;
  // prefill/decode disaggregation (proxy-reported handover)
  pd?: 'prefill' | 'transfer' | 'decode';
  worker?: string; // replica chosen by the proxy (req.route)
  prefillWorker?: string;
  decodeWorker?: string;
  prefillMs?: number;
  kvBlocks?: number;
  // prefix cache
  family?: string; // hash of the prompt's first ~1k chars (req.arrive.prefix_family)
  prefixChars?: number;
  prefixOutcome?: 'founder' | 'hit' | 'partial' | 'miss';
  lStats?: number; // local time prefill_stats arrived
}

export interface Family {
  id: string;
  members: number; // requests seen with prefill stats
  coreTokens: number; // best estimate of the shared prefix length
  hits: number;
  misses: number;
  lastAt: number;
}

export interface KvCapacity {
  blocks: number;
  blockSize: number;
  tokens: number;
  known: boolean;
}

export interface MoeTick {
  type: 'moe.tick';
  layers?: number | null;
  experts?: number | null;
  top_k?: number | null;
  load?: number[] | null; // routed-token share per expert over the window
  layer_imbalance?: number[] | null; // max/mean per layer
  window_s?: number | null;
}

/** One kv.map snapshot, decoded per cell.
 *  state: -1 free, -2 cached (radix prefix, no running owner), -3 shared prefix (≥2 running requests),
 *  -4 co-located (several requests' own tokens in one cell), k >= 0 single owner (prefix[k] = 1 if via its reused prefix). */
export interface KvMap {
  at: number;
  instance: string;
  bins: number;
  binTokens: number;
  capacityTokens: number;
  usedTokens: number | null; // measured: allocator free list
  ownedTokens: number | null; // derived: sum over running requests
  bytesPerToken: number | null;
  state: Int32Array;
  prefix: Uint8Array; // single-owner cells reached through the owner's reused prefix
  owners: KvOwner[];
  shared: Map<number, number[]>; // bin -> owner indexes (shared and co-located cells)
  prevState: Int32Array | null;
  prevOwners: KvOwner[];
  changedAt: Float64Array; // local ms when each cell last changed
}

export interface KvOwner {
  rid: string;
  tokens: number;
  prefixTokens: number | null; // tokens owned by the radix tree (reused / published prefix)
  prefixRanges: [number, number][];
  ownRanges: [number, number][];
}

export interface Batch {
  t: number; // local ms
  seq: number;
  dur: number | null; // step time, ms
  reqs: [string, string, number][]; // [rid, 'p' | 'd', tokens]
  truncated: boolean;
  pTok: number;
  pReq: number;
  dReq: number;
}

export interface Recent {
  rid: string;
  cls: string;
  prompt: number;
  output: number;
  at: number;
}

const CLASS_ORDER_ = ['chat', 'agentic', 'rag', 'burst', 'spec'];

/** Traffic class: the workload tag when present, else prompt-size bucket. */
export function trafficClass(r: Req): string {
  if (r.workload) return r.workload.toLowerCase();
  // joined mid-flight with no size known: say so rather than guessing a bucket
  if (r.partial || (r.promptTokens == null && !r.promptChars)) return 'unclassified';
  const t = r.promptTokens ?? r.promptChars! / 3.8;
  return t < 1000 ? 'short' : t <= 4000 ? 'medium' : 'long';
}
void CLASS_ORDER_;

export interface Note extends AdvisorNote {
  firstSeen: number; // source ms when first seen
  updatedAt: number; // local ms of last update
  clearedAt?: number;
}

export interface Instance extends InstanceInfo {
  name: string;
}

/** Strip the per-replay suffix ("burst-0012~9f3a1c" -> "burst-0012"). */
export const displayRid = (rid: string) => rid.replace(/~[0-9a-z]+$/i, '');

export function isExperiment(n: AdvisorNote): boolean {
  return n.kind === 'experiment' || (!!n.config && /^(exp|auto)-\d+/.test(n.id));
}

const KEEP_FINISHED_MS = 15_000;
const FRESH_MS = 3_000;
const WINDOW_MS = 60_000;
const STALE_MS = 45_000;

interface Sample { at: number; v: number }

function pctl(xs: number[], p: number): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))))];
}

export class Store {
  reqs = new Map<string, Req>();
  hello: Hello | null = null;
  engine: EngineTick | null = null;
  engineAt = -1e12;
  gpu: GpuTick | null = null;
  gpuAt = -1e12;
  lastStep: EngineStep | null = null;
  stepAt = -1e12;
  stepTimes: number[] = []; // local ms of recent steps
  stepPulses = 0; // steps not yet turned into heartbeat pulses
  notes = new Map<string, Note>(); // every advisor/experiment note, upserted by id
  cap: number | null | undefined = undefined;
  notesVersion = 0;
  families = new Map<string, Family>();
  prefixFlashes: { rid: string; kind: 'hit' | 'miss' }[] = []; // for the scene to draw once
  evictPending = 0;
  moe: MoeTick | null = null;
  moeAt = -1e12;
  recent: Recent[] = [];
  // history for the inspector (5 min)
  hist = {
    kv: [] as { t: number; v: number }[],
    gpu: new Map<number, { t: number; util: number | null; mem: number | null; power: number | null; clock: number | null }[]>(),
    moe: [] as { t: number; load: number[]; layers: number[] }[],
    inst: new Map<string, { t: number; kv: number | null }[]>(),
  };
  ttftLog: { at: number; cls: string; worker?: string; ttft: number; rid: string }[] = [];
  evictTotal = 0;
  evictSeen = false;
  wallOffsetMs = 0; // >0 while rewound: how far the plate is behind the wall clock
  // per-step batches (engine.step.batch): who was in each scheduler step
  batches: Batch[] = [];
  batchAt = -1e12;
  stepHits = new Map<string, number>(); // rid -> steps it was in since the scene last looked
  kvMaps = new Map<string, KvMap>(); // by instance
  private lastSample = 0;
  private prefixSamples: { at: number; prompt: number; cached: number; outcome?: string }[] = [];
  lastRequestEventAt = -1e12;
  preemptPending = 0; // anonymous preemptions not yet drawn
  private lastPreemptTotal: number | null = null;
  private ttft: Sample[] = [];
  private e2e: Sample[] = [];
  private queue: Sample[] = [];
  private chunkBucket = 0;
  private chunkBucketAt = 0;
  chunkRate = 0; // tokens/s derived from proxy chunks (fallback when no engine tick)
  admitRate = 0; // requests admitted per second (from req.admit)
  private admitBucket = 0;
  status = 'connecting';
  sourceLabel = '';
  events = 0;

  ingest(m: Msg, now = performance.now()): void {
    if (!m || typeof m !== 'object') return;
    this.events++;
    const tms = typeof m.t_ns === 'number' ? m.t_ns / 1e6 : Date.now();
    switch (m.type) {
      case 'hello':
        this.hello = { ...(this.hello ?? {}), ...(m as Hello), coverage: { ...(this.hello?.coverage ?? {}), ...((m as Hello).coverage ?? {}) } };
        return;
      case 'engine.tick': {
        const t = m as EngineTick;
        this.engine = t;
        this.engineAt = now;
        if (typeof t.preemptions_total === 'number') {
          if (this.lastPreemptTotal !== null && t.preemptions_total > this.lastPreemptTotal) {
            this.preemptPending += Math.min(12, t.preemptions_total - this.lastPreemptTotal);
          }
          this.lastPreemptTotal = t.preemptions_total;
        }
        return;
      }
      case 'gpu.tick':
        this.gpu = m as GpuTick;
        this.gpuAt = now;
        return;
      case 'kv.map':
        this.onKvMap(m, now);
        return;
      case 'moe.tick': {
        this.moe = m as MoeTick;
        this.moeAt = now;
        const mt = m as MoeTick;
        if (mt.load?.length) {
          this.hist.moe.push({ t: now, load: mt.load, layers: mt.layer_imbalance ?? [] });
          if (this.hist.moe.length > 180) this.hist.moe.shift();
        }
        return;
      }
      case 'engine.step':
        this.onStep(m as EngineStep, now);
        return;
      case 'advisor.note': {
        const n = m as AdvisorNote;
        if (!n.id) return;
        const prev = this.notes.get(n.id);
        const merged: Note = { ...(prev ?? {}), ...n, firstSeen: prev?.firstSeen ?? tms, updatedAt: now };
        if (n.status === 'cleared' && !prev?.clearedAt) merged.clearedAt = now;
        this.notes.set(n.id, merged);
        this.notesVersion++;
        return;
      }
      case 'ctl.set':
        if (m.key === 'max_inflight') this.cap = m.new ?? null;
        return;
      case 'proxy.start':
        if (this.cap === undefined) this.cap = m.max_inflight ?? null;
        return;
    }
    if (!m.type?.startsWith('req.') || !m.rid) return;
    this.lastRequestEventAt = now;
    const r = this.reqs.get(m.rid) ?? this.create(m.rid, tms, now, m.type !== 'req.arrive');
    r.lastAt = now;
    switch (m.type) {
      case 'req.arrive':
        r.family = m.prefix_family ?? r.family;
        r.prefixChars = m.prefix_chars ?? r.prefixChars;
        r.workload = m.workload ?? r.workload;
        r.promptChars = m.prompt_chars ?? r.promptChars;
        r.maxTokens = m.max_tokens ?? r.maxTokens;
        r.tArrive = tms;
        r.lArrive = now;
        break;
      case 'req.admit':
        this.admitBucket++;
        r.tAdmit = tms;
        r.lAdmit = now;
        if (r.phase === 'queued') r.phase = 'prefill';
        if (!r.partial) this.push(this.queue, now, tms - r.tArrive);
        break;
      case 'req.first_token':
        r.tFirst = tms;
        r.lFirst = now;
        r.engineTtftMs = m.ttft_ms;
        if (r.phase === 'queued' || r.phase === 'prefill') r.phase = 'decode';
        if (r.pd) r.pd = 'decode';
        if (!r.partial) {
          this.push(this.ttft, now, tms - r.tArrive);
          this.ttftLog.push({ at: now, cls: trafficClass(r), worker: r.worker ?? r.decodeWorker, ttft: tms - r.tArrive, rid: r.rid });
          if (this.ttftLog.length > 6000) this.ttftLog.splice(0, this.ttftLog.length - 6000);
        }
        break;
      case 'req.chunk': {
        const n = typeof m.n === 'number' && m.n > 0 ? m.n : 1;
        r.chunks += n;
        r.pendingSparks += n;
        this.chunkBucket += n;
        if (r.phase === 'queued' || r.phase === 'prefill') {
          r.phase = 'decode';
          r.tFirst ??= tms;
          r.lFirst ??= now;
        }
        if (r.pd) r.pd = 'decode';
        break;
      }
      case 'req.prefill_stats':
        this.onPrefillStats(r, m, now);
        break;
      case 'req.route':
        r.worker = m.worker ?? r.worker;
        break;
      case 'req.prefill_start':
        r.pd = 'prefill';
        r.prefillWorker = m.worker ?? r.prefillWorker;
        if (r.phase === 'queued') r.phase = 'prefill';
        break;
      case 'req.prefill_done':
        r.pd = 'transfer';
        r.prefillMs = m.prefill_ms ?? r.prefillMs;
        r.promptTokens = m.prompt_tokens ?? r.promptTokens;
        r.cachedTokens = m.cached_tokens ?? r.cachedTokens;
        r.kvBlocks = m.kv_blocks ?? r.kvBlocks;
        break;
      case 'req.decode_start':
        r.pd = r.pd === 'decode' ? 'decode' : 'transfer';
        r.decodeWorker = m.worker ?? r.decodeWorker;
        break;
      case 'req.done': {
        r.tDone = tms;
        r.lDone = now;
        r.phase = 'done';
        r.finish = m.finish_reason ?? undefined;
        const u = m.usage ?? {};
        r.promptTokens = u.prompt_tokens ?? r.promptTokens;
        r.completionTokens = u.completion_tokens ?? r.completionTokens ?? r.chunks;
        r.cachedTokens = u.prompt_tokens_details?.cached_tokens ?? r.cachedTokens;
        if (!r.partial) this.push(this.e2e, now, tms - r.tArrive);
        if (r.promptTokens != null) {
          this.recent.push({ rid: r.rid, cls: trafficClass(r), prompt: r.promptTokens, output: r.completionTokens ?? r.chunks, at: now });
          if (this.recent.length > 320) this.recent.splice(0, this.recent.length - 320);
        }
        break;
      }
      case 'req.cancel':
      case 'req.error':
        r.tDone = tms;
        r.lDone = now;
        r.phase = 'failed';
        r.error = m.type === 'req.error' ? String(m.detail ?? m.status ?? 'error') : `cancelled (${m.stage ?? '?'})`;
        break;
    }
  }

  private onKvMap(m: Msg, now: number): void {
    const bins = Number(m.bins) || 0;
    if (bins <= 0 || bins > 1 << 20) return;
    const inst = String(m.instance ?? 'engine');
    const prev = this.kvMaps.get(inst);
    const state = new Int32Array(bins).fill(-1);
    const prefix = new Uint8Array(bins);
    const shared = new Map<number, number[]>();
    const each = (ranges: unknown, f: (b: number) => void) => {
      for (const r of (ranges as [number, number][]) ?? []) for (let b = Math.max(0, r[0]); b < Math.min(bins, r[1]); b++) f(b);
    };
    each(m.cached, (b) => (state[b] = -2));
    const owners: KvOwner[] = [];
    type RawOwner = { rid: string; tokens?: number; prefix_tokens?: number; ranges?: [number, number][]; prefix_ranges?: [number, number][]; own_ranges?: [number, number][] };
    for (const o of (m.owners as RawOwner[]) ?? []) {
      const k = owners.length;
      const split = !!(o.prefix_ranges || o.own_ranges);
      const ow: KvOwner = {
        rid: o.rid, tokens: o.tokens ?? 0, prefixTokens: o.prefix_tokens ?? null,
        prefixRanges: o.prefix_ranges ?? [], ownRanges: split ? (o.own_ranges ?? []) : (o.ranges ?? []),
      };
      owners.push(ow);
      const claim = (b: number, viaPrefix: boolean) => {
        const cur = state[b];
        if (cur >= 0 && cur !== k) {
          state[b] = -4; // two requests in one cell: co-located unless the engine says it is a shared prefix
          shared.set(b, [cur, k]);
        } else if (cur === -4 || cur === -3) {
          const l = shared.get(b)!;
          if (!l.includes(k)) l.push(k);
        } else if (cur !== k) {
          state[b] = k;
          prefix[b] = viaPrefix ? 1 : 0;
        }
      };
      each(ow.prefixRanges, (b) => claim(b, true));
      each(ow.ownRanges, (b) => claim(b, false));
    }
    // true prefix sharing, as reported by the engine
    each(m.shared, (b) => {
      if (state[b] >= 0) shared.set(b, [state[b]]);
      else if (!shared.has(b)) shared.set(b, []);
      state[b] = -3;
    });
    const changedAt = prev && prev.bins === bins ? prev.changedAt : new Float64Array(bins).fill(-1e12);
    if (prev && prev.bins === bins) {
      const key = (st: Int32Array, ow: KvOwner[], b: number) => (st[b] >= 0 ? ow[st[b]].rid : String(st[b]));
      for (let b = 0; b < bins; b++) if (key(prev.state, prev.owners, b) !== key(state, owners, b)) changedAt[b] = now;
    }
    this.kvMaps.set(inst, {
      at: now, instance: inst, bins, binTokens: Number(m.bin_tokens) || 0, capacityTokens: Number(m.capacity_tokens) || 0,
      usedTokens: typeof m.used_tokens === 'number' ? m.used_tokens : null,
      ownedTokens: typeof m.owned_tokens === 'number' ? m.owned_tokens : owners.reduce((a2, o) => a2 + (o.tokens || 0), 0),
      bytesPerToken: typeof m.bytes_per_token === 'number' ? m.bytes_per_token : null,
      state, prefix, owners, shared, prevState: prev?.bins === bins ? prev.state : null, prevOwners: prev?.owners ?? [], changedAt,
    });
  }

  /** Prefix-cache outcome for a family member, from the engine's own cached-token count. */
  private onPrefillStats(r: Req, m: Msg, now: number): void {
    r.promptTokens = m.prompt_tokens ?? r.promptTokens;
    r.cachedTokens = m.cached_tokens ?? r.cachedTokens;
    r.lStats = now;
    const prompt = r.promptTokens ?? 0;
    const cached = r.cachedTokens ?? 0;
    this.prefixSamples.push({ at: now, prompt, cached });
    if (!r.family) return;
    let f = this.families.get(r.family);
    if (!f) {
      f = { id: r.family, members: 0, coreTokens: 0, hits: 0, misses: 0, lastAt: now };
      this.families.set(r.family, f);
    }
    if (f.members === 0) {
      r.prefixOutcome = cached > 0 ? 'hit' : 'founder';
      if (cached > 0) f.hits++; // the prefix was already cached before this page saw the family
      f.coreTokens = Math.max(cached, Math.min(prompt, (r.prefixChars ?? 1024) / 3.8));
    } else if (cached > 0 && cached >= 0.5 * f.coreTokens) {
      r.prefixOutcome = 'hit';
      f.hits++;
      f.coreTokens = Math.max(f.coreTokens, cached);
    } else if (cached < 0.1 * f.coreTokens) {
      r.prefixOutcome = 'miss';
      f.misses++;
    } else r.prefixOutcome = 'partial';
    if (r.prefixOutcome === 'hit' || r.prefixOutcome === 'miss') this.prefixFlashes.push({ rid: r.rid, kind: r.prefixOutcome });
    this.prefixSamples[this.prefixSamples.length - 1].outcome = r.prefixOutcome;
    f.members++;
    f.lastAt = now;
  }

  /** Live prefix-cache view over the last minute, from the engine's cached-token counts. */
  prefixLive(now = performance.now()): { tokenRate: number | null; hits: number; misses: number; n: number } {
    while (this.prefixSamples.length && now - this.prefixSamples[0].at > WINDOW_MS) this.prefixSamples.shift();
    let p = 0;
    let c = 0;
    let hits = 0;
    let misses = 0;
    for (const x of this.prefixSamples) {
      p += x.prompt;
      c += x.cached;
      if (x.outcome === 'hit') hits++;
      if (x.outcome === 'miss') misses++;
    }
    return { tokenRate: p ? c / p : null, hits, misses, n: this.prefixSamples.length };
  }

  /** KV capacity of an instance (or the single engine), from the engine's reported config. */
  kvCapacity(name?: string): KvCapacity {
    const inst = name ? this.hello?.instances?.[name]?.config : undefined;
    const c = { ...(this.hello?.config ?? {}), ...(inst ?? {}) } as Record<string, unknown>;
    const n = (k: string) => (typeof c[k] === 'number' && Number.isFinite(c[k] as number) ? (c[k] as number) : null);
    const blocks = n('num_gpu_blocks');
    const bs = n('block_size');
    if (blocks && bs) return { blocks, blockSize: bs, tokens: blocks * bs, known: true };
    const tot = n('max_total_tokens');
    if (tot) {
      const ps = Math.max(1, n('page_size') ?? 1);
      return { blocks: Math.ceil(tot / ps), blockSize: ps, tokens: tot, known: true };
    }
    return { blocks: 65536, blockSize: 16, tokens: 65536 * 16, known: false };
  }

  private onStep(s: EngineStep, now: number): void {
    this.lastStep = s;
    const bt = (s as unknown as { batch?: { reqs?: [string, string, number][]; truncated?: boolean } }).batch;
    if (bt?.reqs) {
      let pTok = 0;
      let pReq = 0;
      let dReq = 0;
      for (const [rid, ph, tok] of bt.reqs) {
        if (ph === 'p') {
          pTok += tok;
          pReq++;
        } else dReq++;
        this.stepHits.set(rid, (this.stepHits.get(rid) ?? 0) + 1);
      }
      this.batches.push({ t: now, seq: s.step ?? (s as unknown as { seq?: number }).seq ?? 0, dur: s.iter?.elapsed_ms ?? s.dur_ms ?? null, reqs: bt.reqs, truncated: !!bt.truncated, pTok, pReq, dReq });
      this.batchAt = now;
      while (this.batches.length && now - this.batches[0].t > 12_000) this.batches.shift();
      if (this.batches.length > 5000) this.batches.splice(0, this.batches.length - 5000);
    }
    const ev = (s as unknown as { evictions?: number }).evictions;
    if (typeof ev === 'number') this.evictSeen = true;
    if (typeof ev === 'number' && ev > 0) {
      this.evictPending += Math.min(ev, 64);
      this.evictTotal += ev;
    }
    this.stepAt = now;
    this.stepTimes.push(now);
    this.stepPulses++;
    if (s.preempted) this.preemptPending += Math.min(12, s.preempted);
    for (const f of s.finished ?? []) {
      const r = f.rid ? this.reqs.get(f.rid) : undefined;
      if (r) {
        r.engine = f;
        r.promptTokens ??= f.prompt_tokens ?? undefined;
        r.cachedTokens ??= f.cached_tokens ?? undefined;
      }
    }
  }

  private create(rid: string, tms: number, now: number, partial: boolean): Req {
    const r: Req = { rid, partial, tArrive: tms, lArrive: now, chunks: 0, phase: 'queued', pendingSparks: 0, lastAt: now };
    this.reqs.set(rid, r);
    return r;
  }

  private push(arr: Sample[], at: number, v: number): void {
    if (Number.isFinite(v) && v >= 0) arr.push({ at, v });
  }

  /** Wall-clock "now" for the plate (behind real time while rewound). */
  wallNow(): number {
    return Date.now() - this.wallOffsetMs;
  }

  /** Forget everything (used by the time scrubber before replaying the buffer). */
  reset(): void {
    const keep = { status: this.status, sourceLabel: this.sourceLabel };
    Object.assign(this, new Store(), keep);
  }

  private sample(now: number): void {
    const cut = now - 5 * 60_000;
    const fresh = this.engineFresh(now);
    this.hist.kv.push({ t: now, v: fresh ? (this.engine?.kv_usage ?? 0) : 0 });
    while (this.hist.kv.length && this.hist.kv[0].t < cut) this.hist.kv.shift();
    if (fresh) {
      for (const [name, it] of Object.entries(this.engine?.instances ?? {})) {
        const arr = this.hist.inst.get(name) ?? [];
        arr.push({ t: now, kv: it.kv_usage ?? null });
        while (arr.length && arr[0].t < cut) arr.shift();
        this.hist.inst.set(name, arr);
      }
    }
    if (now - this.gpuAt < 3000) {
      for (const g of this.gpu?.gpus ?? []) {
        const arr = this.hist.gpu.get(g.i) ?? [];
        arr.push({ t: now, util: g.util ?? null, mem: g.mem_used_gb ?? null, power: g.power_w ?? null, clock: g.sm_clock_mhz ?? null });
        while (arr.length && arr[0].t < cut) arr.shift();
        this.hist.gpu.set(g.i, arr);
      }
    }
    while (this.ttftLog.length && this.ttftLog[0].at < cut) this.ttftLog.shift();
  }

  /** Housekeeping once per frame. */
  tick(now: number): void {
    if (now - this.lastSample >= 1000) {
      this.lastSample = now;
      this.sample(now);
    }
    for (const arr of [this.ttft, this.e2e, this.queue]) {
      while (arr.length && now - arr[0].at > WINDOW_MS) arr.shift();
    }
    while (this.stepTimes.length && now - this.stepTimes[0] > 2000) this.stepTimes.shift();
    if (now - this.chunkBucketAt >= 500) {
      const dt = (now - this.chunkBucketAt) / 1000;
      const rate = this.chunkBucketAt ? this.chunkBucket / dt : 0;
      this.chunkRate = this.chunkRate * 0.5 + rate * 0.5;
      this.chunkBucket = 0;
      const ar = this.chunkBucketAt ? this.admitBucket / dt : 0;
      this.admitRate = this.admitRate * 0.6 + ar * 0.4;
      this.admitBucket = 0;
      this.chunkBucketAt = now;
    }
    for (const r of this.reqs.values()) {
      if ((r.phase === 'queued' || r.phase === 'prefill' || r.phase === 'decode') && now - r.lastAt > STALE_MS) {
        this.fail(r, now, 'no events for 45 s');
      }
    }
    for (const [id, n] of this.notes) {
      if (n.clearedAt !== undefined && now - n.clearedAt > 4000) {
        this.notes.delete(id);
        this.notesVersion++;
      }
    }
    for (const [rid, r] of this.reqs) {
      if ((r.phase === 'done' || r.phase === 'failed') && r.lDone !== undefined && now - r.lDone > KEEP_FINISHED_MS) {
        this.reqs.delete(rid);
      }
    }
  }

  /** Engine instances (prefill/decode workers, DP ranks), freshest data first from ticks. */
  instances(now = performance.now()): Instance[] {
    const fromHello = this.hello?.instances ?? {};
    const fromTick = this.engineFresh(now) ? (this.engine?.instances ?? {}) : {};
    const names = new Set([...Object.keys(fromHello), ...Object.keys(fromTick)]);
    const order = (r?: string | null) => (r === 'prefill' ? 0 : r === 'decode' ? 2 : 1);
    return [...names]
      .map((name) => ({ name, ...(fromHello[name] ?? {}), ...(fromTick[name] ?? {}), role: fromTick[name]?.role ?? fromHello[name]?.role ?? name.split('-')[0] }))
      .sort((a, b) => order(a.role) - order(b.role) || a.name.localeCompare(b.name, 'en', { numeric: true }));
  }

  private fail(r: Req, now: number, why: string): void {
    r.phase = 'failed';
    r.error = why;
    r.lDone = now;
    r.tDone ??= Date.now();
  }

  /** The stream dropped (engine/collector restart): in-flight requests will never report back. */
  markLost(now = performance.now()): void {
    for (const r of this.reqs.values()) {
      if (r.phase === 'queued' || r.phase === 'prefill' || r.phase === 'decode') this.fail(r, now, 'stream lost — engine restarted');
    }
    this.engineAt = -1e12;
    this.gpuAt = -1e12;
    this.stepAt = -1e12;
  }

  coverage(now = performance.now()): Coverage & { declared: Partial<Coverage> } {
    const d = this.hello?.coverage ?? {};
    const tickFresh = now - this.engineAt < FRESH_MS;
    return {
      declared: d,
      requests: !!d.requests || now - this.lastRequestEventAt < 30_000,
      steps: now - this.stepAt < FRESH_MS,
      kv: tickFresh && this.engine?.kv_usage != null,
      spec: tickFresh && this.engine?.spec != null,
      gpu: now - this.gpuAt < FRESH_MS && !!this.gpu?.gpus?.length,
    };
  }

  engineFresh(now = performance.now()): boolean {
    return now - this.engineAt < FRESH_MS;
  }

  stats(now = performance.now()) {
    let queued = 0;
    let active = 0;
    for (const r of this.reqs.values()) {
      if (r.phase === 'queued') queued++;
      else if (r.phase === 'prefill' || r.phase === 'decode') active++;
    }
    const fresh = this.engineFresh(now);
    const e = fresh ? this.engine : null;
    const steps = this.stepTimes.length / 2;
    return {
      proxyQueued: queued,
      inFlight: active,
      running: e?.running ?? (e?.active_requests != null ? e.active_requests : null),
      waiting: e?.waiting ?? null,
      kv: e?.kv_usage ?? null,
      prefixHit: e?.prefix_hit_rate ?? null,
      tokS: e?.gen_tok_s ?? (this.lastRequestEventAt > now - 5000 ? this.chunkRate : null),
      tokSource: e?.gen_tok_s != null ? 'engine' : 'proxy',
      promptTokS: e?.prompt_tok_s ?? null,
      stepsPerS: e?.steps_per_s ?? (now - this.stepAt < FRESH_MS ? steps : null),
      spec: e?.spec ?? null,
      ttftP50: pctl(this.ttft.map((s) => s.v), 50),
      ttftP90: pctl(this.ttft.map((s) => s.v), 90),
      e2eP50: pctl(this.e2e.map((s) => s.v), 50),
      queueP50: pctl(this.queue.map((s) => s.v), 50),
      cap: this.cap,
    };
  }
}
