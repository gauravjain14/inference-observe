// Synthetic source for developing the board without an engine. Clearly labelled as mock.
// A small queueing model: proxy cap -> engine queue -> chunked prefill -> shared decode.
import type { Msg, Source } from '../contract';
import { clamp, mulberry32, poisson } from '../util';

type Mode = 'steady' | 'burst' | 'pd' | 'dp' | 'rep';

interface MReq {
  rid: string;
  family: string;
  hitCached: number; // what the engine will report as cached at first token
  prompt: number;
  cached: number;
  out: number;
  gen: number;
  acc: number;
  tArrive: number;
  tAdmit: number;
  tSched: number;
  tFirst: number;
  prefillLeft: number;
  preempts: number;
}

const CONF: Record<Mode, { gpus: number; engine: string; model: string; maxRunning: number; kvCap: number; tokCap: number }> = {
  steady: { gpus: 1, engine: 'vllm', model: 'Qwen3-30B-A3B-Instruct-2507', maxRunning: 64, kvCap: 260_000, tokCap: 4200 },
  burst: { gpus: 8, engine: 'sglang', model: 'Qwen3-235B-A22B-FP8 (TP8)', maxRunning: 96, kvCap: 300_000, tokCap: 6400 },
  pd: { gpus: 2, engine: 'vllm', model: 'Qwen3-30B-A3B · 1P1D (NIXL)', maxRunning: 64, kvCap: 260_000, tokCap: 4800 },
  dp: { gpus: 2, engine: 'vllm', model: 'Qwen3-30B-A3B · DP2', maxRunning: 128, kvCap: 520_000, tokCap: 8800 },
  rep: { gpus: 3, engine: 'sglang', model: 'Qwen3-30B-A3B · 3 replicas', maxRunning: 192, kvCap: 780_000, tokCap: 12000 },
};

export class MockSource implements Source {
  readonly label: string;
  private timer: number | undefined;
  private rnd = mulberry32(42);
  private t = 0;
  private seq = 0;
  private step = 0;
  private stepAcc = 0;
  private tickAcc = 0;
  private proxyQ: MReq[] = [];
  private engQ: MReq[] = [];
  private running: MReq[] = [];
  private cap: number | null = null;
  private preemptions = 0;
  private genWindow: number[] = [];
  private promptWindow: number[] = [];
  private stepsWindow: number[] = [];
  private pendingFinished: Record<string, unknown>[] = [];
  private stepPreempted = 0;
  private notesSent = new Set<string>();
  private emit: (m: Msg) => void = () => {};

  constructor(private mode: Mode) {
    this.label = `mock · ${mode} · synthetic data`;
  }

  start(emit: (m: Msg) => void, status: (s: string) => void): void {
    this.emit = emit;
    const c = CONF[this.mode];
    status('mock');
    this.send({
      type: 'hello', engine: c.engine, engine_version: c.engine === 'vllm' ? '0.34.0' : '0.5.23', model: c.model,
      gpus: c.gpus, gpu_ids: [...Array(c.gpus).keys()],
      coverage: { requests: true, steps: true, kv: true, spec: false, gpu: true },
      config: c.engine === 'sglang'
        ? { max_total_tokens: c.kvCap, page_size: 16, tp_size: c.gpus, max_running_requests: c.maxRunning, chunked_prefill_size: 8192, schedule_policy: 'lpm', overlap_schedule: true }
        : { num_gpu_blocks: Math.round(c.kvCap / 16), block_size: 16, data_parallel_size: this.mode === 'dp' ? 2 : 1, max_num_seqs: c.maxRunning, max_num_batched_tokens: 8192, enable_prefix_caching: true, ...(this.mode === 'pd' ? { kv_connector: 'NixlConnector' } : {}) },
      label: 'Mock specimen — synthetic data',
      instances: this.instanceNames().reduce((o, [n, role, url]) => ({ ...o, [n]: { role, url } }), {}),
    });
    this.send({ type: 'proxy.start', max_inflight: null });
    let last = performance.now();
    this.timer = window.setInterval(() => {
      const now = performance.now();
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      this.advance(dt);
    }, 16);
  }

  stop(): void {
    window.clearInterval(this.timer);
  }

  private instanceNames(): [string, string, string][] {
    if (this.mode === 'pd') return [['prefill-0', 'prefill', 'http://127.0.0.1:18110'], ['decode-0', 'decode', 'http://127.0.0.1:18120']];
    if (this.mode === 'dp') return [['engine-0.dp0', 'engine', ''], ['engine-0.dp1', 'engine', '']];
    if (this.mode === 'rep') return [0, 1, 2].map((k) => [`replica-${k}`, 'replica', `http://127.0.0.1:${18130 + k}`] as [string, string, string]);
    return [['engine', 'engine', '']];
  }

  private replicaOf = new Map<string, number>();
  // kv.map allocator: 4096 bins, first-fit so allocations read as runs
  private kv = { bins: 4096, owned: new Map<string, number[]>(), core: new Map<string, number[]>(), cached: [] as { bins: number[]; until: number }[], free: [] as number[], acc: 0 };
  private famSeen = new Map<string, number>();
  private moeHot = [3, 17, 42, 77, 101];
  private repLoad = [0, 0, 0];

  private send(m: Msg): void {
    this.emit({ v: 0, src: 'mock', t_ns: Date.now() * 1e6, ...m });
  }

  private arrivalRate(): number {
    const t = this.t;
    if (this.mode === 'burst') {
      const ph = t % 50;
      return 6 + (ph > 10 && ph < 22 ? 30 : 0);
    }
    if (this.mode === 'pd') return 7;
    return 6 + 2 * Math.sin(t / 9);
  }

  private setCap(v: number | null): void {
    if (v === this.cap) return;
    this.send({ type: 'ctl.set', key: 'max_inflight', old: this.cap, new: v });
    this.cap = v;
  }

  private advance(dt: number): void {
    const c = CONF[this.mode];
    this.t += dt;
    const rnd = this.rnd;

    if (this.mode === 'burst') {
      const ph = this.t % 50;
      this.setCap(ph > 12 && ph < 27 ? 24 : null);
    }

    // arrivals
    const n = poisson(rnd, this.arrivalRate() * dt);
    for (let i = 0; i < n; i++) {
      const shared = rnd() < 0.45;
      const fam = shared ? `fam${Math.floor(rnd() * 4)}` : `u${this.seq}`;
      const core = shared ? 1500 + 500 * Number(fam.slice(3)) : 0;
      const prompt = shared ? core + Math.round(80 + rnd() * 600) : Math.round(clamp(Math.exp(6.6 + rnd() * 1.9), 120, 9000));
      const seen = this.famSeen.get(fam) ?? 0;
      this.famSeen.set(fam, seen + 1);
      // the first member builds the prefix; later ones mostly hit, ~12% miss (evicted / different replica)
      const hitCached = shared && seen > 0 && rnd() > 0.12 ? core : 0;
      const r: MReq = {
        rid: `mock-${(this.seq++).toString(36).padStart(4, '0')}`,
        family: fam,
        hitCached,
        prompt,
        cached: hitCached,
        out: Math.round(80 + rnd() * 460),
        gen: 0, acc: 0, tArrive: this.t, tAdmit: 0, tSched: 0, tFirst: 0, prefillLeft: 0, preempts: 0,
      };
      r.cached = Math.max(0, r.cached);
      this.proxyQ.push(r);
      this.send({ type: 'req.arrive', rid: r.rid, workload: shared ? 'agentic' : prompt > 3000 ? 'rag' : 'chat', prompt_chars: Math.round(prompt * 3.8), max_tokens: r.out, n_messages: shared ? 5 : 1, prefix_family: r.family, prefix_chars: 1024 });
    }

    // proxy admission
    let inflight = this.engQ.length + this.running.length;
    while (this.proxyQ.length && (this.cap === null || inflight < this.cap)) {
      const r = this.proxyQ.shift()!;
      r.tAdmit = this.t;
      inflight++;
      this.engQ.push(r);
      this.send({ type: 'req.admit', rid: r.rid, inflight, waiting: this.proxyQ.length });
      if (this.mode === 'rep') {
        const k = this.repLoad.indexOf(Math.min(...this.repLoad));
        this.replicaOf.set(r.rid, k);
        this.repLoad[k]++;
        this.send({ type: 'req.route', rid: r.rid, worker: `http://127.0.0.1:${18130 + k}`, inflight: this.repLoad[k] });
      }
    }

    // engine scheduling
    let kvUsed = this.running.reduce((s, r) => s + r.prompt + r.gen, 0);
    while (this.engQ.length && this.running.length < c.maxRunning && kvUsed + this.engQ[0].prompt < c.kvCap * 0.95) {
      const r = this.engQ.shift()!;
      r.tSched = this.t;
      r.prefillLeft = r.prompt - r.cached;
      kvUsed += r.prompt;
      this.running.push(r);
      if (this.mode === 'pd') this.send({ type: 'req.prefill_start', rid: r.rid, worker: 'http://127.0.0.1:18101' });
    }

    // a scheduler step happens every stepMs; model steps as discrete ticks
    const prefilling = this.running.filter((r) => r.prefillLeft > 0);
    const decoding = this.running.filter((r) => r.prefillLeft <= 0);
    const stepMs = 9 + 0.07 * decoding.length + (prefilling.length ? 7 : 0);
    this.stepAcc += dt * 1000;
    while (this.stepAcc >= stepMs) {
      this.stepAcc -= stepMs;
      this.doStep(prefilling, decoding, stepMs, c);
    }

    // preemption under KV pressure
    kvUsed = this.running.reduce((s, r) => s + r.prompt + r.gen, 0);
    if (kvUsed > c.kvCap * 0.985 && this.running.length > 1) {
      const victim = this.running.pop()!;
      victim.preempts++;
      victim.prefillLeft = victim.prompt + victim.gen - victim.cached;
      this.engQ.unshift(victim);
      this.preemptions++;
      this.stepPreempted++;
    }

    // 10 Hz ticks
    this.tickAcc += dt;
    if (this.tickAcc >= 0.1) {
      this.tickAcc = 0;
      this.ticks(c, kvUsed);
    }
    this.notes();
  }

  private doStep(prefilling: MReq[], decoding: MReq[], stepMs: number, c: (typeof CONF)[Mode]): void {
    this.step++;
    const budget = 8192;
    let ctxTokens = 0;
    let ctxReqs = 0;
    const members: [string, string, number][] = [];
    for (const r of prefilling) {
      if (ctxTokens >= budget) break;
      const take = Math.min(r.prefillLeft, budget - ctxTokens);
      members.push([r.rid, 'p', take]);
      r.prefillLeft -= take;
      ctxTokens += take;
      ctxReqs++;
      if (r.prefillLeft <= 0 && this.mode === 'pd') {
        this.send({ type: 'req.prefill_done', rid: r.rid, worker: 'http://127.0.0.1:18110', prefill_ms: (this.t - r.tSched) * 1000, prompt_tokens: r.prompt, cached_tokens: r.cached, kv_blocks: Math.ceil(r.prompt / 16) });
        this.send({ type: 'req.decode_start', rid: r.rid, worker: 'http://127.0.0.1:18120' });
      }
    }
    const perReq = Math.min(1, (c.tokCap * stepMs) / 1000 / Math.max(1, decoding.length));
    let gen = 0;
    const done: MReq[] = [];
    for (const r of decoding) {
      members.push([r.rid, 'd', 1]);
      r.acc += perReq;
      while (r.acc >= 1 && r.gen < r.out) {
        r.acc -= 1;
        r.gen++;
        gen++;
        if (r.gen === 1) {
          r.tFirst = this.t;
          this.send({ type: 'req.first_token', rid: r.rid, ttft_ms: (this.t - r.tAdmit) * 1000 });
          this.send({ type: 'req.prefill_stats', rid: r.rid, prompt_tokens: r.prompt, cached_tokens: r.hitCached, t_ms: (this.t - r.tAdmit) * 1000 });
        }
        this.send({ type: 'req.chunk', rid: r.rid, i: r.gen - 1, n_chars: 4 });
      }
      if (r.gen >= r.out) done.push(r);
    }
    for (const r of done) {
      this.running.splice(this.running.indexOf(r), 1);
      const k = this.replicaOf.get(r.rid);
      if (k !== undefined) {
        this.repLoad[k]--;
        this.replicaOf.delete(r.rid);
      }
      this.send({
        type: 'req.done', rid: r.rid, e2e_ms: (this.t - r.tAdmit) * 1000, n_chunks: r.out, finish_reason: 'length',
        usage: { prompt_tokens: r.prompt, completion_tokens: r.out, prompt_tokens_details: { cached_tokens: r.cached } },
      });
      this.pendingFinished.push({
        rid: r.rid, queued_s: r.tSched - r.tAdmit, prefill_s: r.tFirst - r.tSched, decode_s: this.t - r.tFirst,
        prompt_tokens: r.prompt, cached_tokens: r.cached, gen_tokens: r.out, preemptions: r.preempts,
      });
    }
    this.genWindow.push(gen);
    this.promptWindow.push(ctxTokens);
    this.stepsWindow.push(this.t);
    const kv = this.running.reduce((s, r) => s + r.prompt + r.gen, 0) / c.kvCap;
    this.send({
      type: 'engine.step', step: this.step, running: this.running.length, waiting: this.engQ.length, kv_usage: kv,
      iter: { ctx_reqs: ctxReqs, ctx_tokens: ctxTokens, gen_reqs: decoding.length, gen_tokens: gen, elapsed_ms: stepMs },
      gen_tokens: gen, preempted: this.stepPreempted, finished: this.pendingFinished.splice(0),
      evictions: kv > 0.8 && this.rnd() < 0.2 ? 1 + Math.floor(this.rnd() * 3) : 0,
      batch: { reqs: members.slice(0, 256), truncated: members.length > 256 },
    });
    this.stepPreempted = 0;
  }

  private ranges(bins: number[]): [number, number][] {
    const s = [...bins].sort((a, b) => a - b);
    const out: [number, number][] = [];
    for (const b of s) {
      const last = out[out.length - 1];
      if (last && last[1] === b) last[1]++;
      else out.push([b, b + 1]);
    }
    return out;
  }

  private takeBin(): number | null {
    const kv = this.kv;
    if (!kv.free.length && kv.cached.length) {
      const old = kv.cached.shift()!; // evict the oldest cached run
      kv.free.push(...old.bins);
      kv.free.sort((a, b) => a - b);
    }
    return kv.free.length ? kv.free.shift()! : null;
  }

  private kvMap(c: (typeof CONF)[Mode]): void {
    const kv = this.kv;
    if (!kv.free.length && !kv.owned.size && !kv.cached.length) kv.free = Array.from({ length: kv.bins }, (_, i) => i);
    const binTok = Math.ceil(c.kvCap / kv.bins);
    const live = new Map(this.running.map((r) => [r.rid, r]));
    for (const [rid, bins] of kv.owned) {
      if (!live.has(rid)) {
        kv.owned.delete(rid);
        kv.cached.push({ bins, until: this.t + 20 });
      }
    }
    for (const [fam, bins] of kv.core) {
      if (![...live.values()].some((r) => r.family === fam && r.hitCached > 0)) {
        kv.core.delete(fam);
        kv.cached.push({ bins, until: this.t + 30 });
      }
    }
    for (const x of kv.cached.filter((x) => x.until < this.t)) kv.free.push(...x.bins);
    kv.cached = kv.cached.filter((x) => x.until >= this.t);
    kv.free.sort((a, b) => a - b);
    for (const r of this.running) {
      if (r.hitCached > 0 && !kv.core.has(r.family)) {
        const need = Math.ceil(r.hitCached / binTok);
        const core: number[] = [];
        for (let i = 0; i < need; i++) {
          const b = this.takeBin();
          if (b == null) break;
          core.push(b);
        }
        kv.core.set(r.family, core);
      }
      const own = kv.owned.get(r.rid) ?? [];
      const need = Math.ceil((r.prompt - r.hitCached + r.gen) / binTok);
      while (own.length < need) {
        const b = this.takeBin();
        if (b == null) break;
        own.push(b);
      }
      kv.owned.set(r.rid, own);
    }
    const owners = this.running.map((r) => {
      const bins = [...(kv.owned.get(r.rid) ?? []), ...(r.hitCached > 0 ? (kv.core.get(r.family) ?? []) : [])];
      return { rid: r.rid, tokens: r.prompt + r.gen, ranges: this.ranges(bins) };
    });
    this.send({
      type: 'kv.map', instance: 'engine', capacity_tokens: c.kvCap, bin_tokens: binTok, bins: kv.bins,
      owners, free: this.ranges(kv.free), cached: this.ranges(kv.cached.flatMap((x) => x.bins)),
    });
  }

  private ticks(c: (typeof CONF)[Mode], kvUsed: number): void {
    this.kv.acc++;
    if (this.mode === 'steady' && this.kv.acc % 10 === 0) this.kvMap(c);
    while (this.stepsWindow.length && this.t - this.stepsWindow[0] > 1) {
      this.stepsWindow.shift();
      this.genWindow.shift();
      this.promptWindow.shift();
    }
    const gen = this.genWindow.reduce((a, b) => a + b, 0);
    const prompt = this.promptWindow.reduce((a, b) => a + b, 0);
    this.send({
      type: 'engine.tick', engine: c.engine, source: 'tap', running: this.running.length, waiting: this.engQ.length,
      kv_usage: kvUsed / c.kvCap, prefix_hit_rate: 0.31, prompt_tok_s: prompt, gen_tok_s: gen,
      steps_per_s: this.stepsWindow.length, preemptions_total: this.preemptions, spec: null,
      active_requests: this.proxyQ.length + this.engQ.length + this.running.length,
      instances: this.instanceTicks(c, gen, prompt),
    });
    const load = clamp(this.running.length / (c.maxRunning * 0.6), 0, 1);
    const gpus = [...Array(c.gpus).keys()].map((i) => {
      let util = 8 + 88 * load;
      if (this.mode === 'pd') util = i === 0 ? clamp(prompt / 30_000, 0, 1) * 95 : 10 + 85 * load;
      util = clamp(util + (this.rnd() - 0.5) * 6, 0, 100);
      return { i, util: Math.round(util), mem_used_gb: 118 + 18 * (kvUsed / c.kvCap), mem_total_gb: 140.4, power_w: Math.round(180 + 480 * util / 100), sm_clock_mhz: 1980 };
    });
    this.send({ type: 'gpu.tick', gpus });
    // expert routing: a few hot experts drifting over time
    if (Math.floor(this.t * 2) % 5 === 0) {
      if (this.rnd() < 0.05) this.moeHot[Math.floor(this.rnd() * this.moeHot.length)] = Math.floor(this.rnd() * 128);
      const load = Array.from({ length: 128 }, (_, i) => 1 + (this.moeHot.includes(i) ? 2.2 + this.rnd() : 0) + this.rnd() * 0.6);
      const sum = load.reduce((a, b) => a + b, 0);
      this.send({
        type: 'moe.tick', layers: 48, experts: 128, top_k: 8, window_s: 2,
        load: load.map((x) => x / sum),
        layer_imbalance: Array.from({ length: 48 }, (_, l) => 1.3 + 1.6 * Math.abs(Math.sin(l * 0.7 + this.t * 0.1)) + this.rnd() * 0.3),
      });
    }
  }

  private instanceTicks(c: (typeof CONF)[Mode], gen: number, prompt: number): Record<string, unknown> {
    const kvOf = (rs: MReq[]) => rs.reduce((a, r) => a + r.prompt + r.gen, 0) / c.kvCap;
    const steps = this.stepsWindow.length;
    if (this.mode === 'pd') {
      const pre = this.running.filter((r) => r.prefillLeft > 0);
      const dec = this.running.filter((r) => r.prefillLeft <= 0);
      return {
        'prefill-0': { role: 'prefill', running: pre.length, waiting: this.engQ.length, kv_usage: kvOf(pre) * 4, steps_per_s: steps, prompt_tok_s: prompt, gen_tok_s: 0 },
        'decode-0': { role: 'decode', running: dec.length, waiting: 0, kv_usage: kvOf(dec), steps_per_s: steps, prompt_tok_s: 0, gen_tok_s: gen },
      };
    }
    if (this.mode === 'dp') {
      const a = this.running.filter((_, i) => i % 2 === 0);
      const b = this.running.filter((_, i) => i % 2 === 1);
      return {
        'engine-0.dp0': { role: 'engine', running: a.length, waiting: 0, kv_usage: kvOf(a) * 2, steps_per_s: steps, gen_tok_s: gen / 2, prompt_tok_s: prompt / 2 },
        'engine-0.dp1': { role: 'engine', running: b.length, waiting: 0, kv_usage: kvOf(b) * 2, steps_per_s: steps, gen_tok_s: gen / 2, prompt_tok_s: prompt / 2 },
      };
    }
    if (this.mode === 'rep') {
      const out: Record<string, unknown> = {};
      [0, 1, 2].forEach((k) => {
        const mine = this.running.filter((r) => this.replicaOf.get(r.rid) === k);
        out[`replica-${k}`] = { role: 'replica', running: mine.length, waiting: 0, kv_usage: kvOf(mine) * 3, steps_per_s: steps, gen_tok_s: gen / 3, prompt_tok_s: prompt / 3 };
      });
      return out;
    }
    return { engine: { role: 'engine', running: this.running.length, waiting: this.engQ.length, kv_usage: kvOf(this.running), steps_per_s: steps, gen_tok_s: gen, prompt_tok_s: prompt } };
  }

  private notes(): void {
    const t = this.t;
    const note = (id: string, at: number, m: Record<string, unknown>) => {
      if (t >= at && !this.notesSent.has(id)) {
        this.notesSent.add(id);
        this.send({ type: 'advisor.note', id, sample: true, ...m });
      }
    };
    if (this.mode === 'burst') {
      note('cap', 16, {
        agent: 'Admission', region: 'queue', title: 'The cap, not the engine, is holding requests at the membrane',
        body: 'The proxy cap is 24 while the engine runs 24 of 96 slots and KV sits under 60%. Requests wait outside with nothing to do.',
        predicted: 'lift cap → queue time p50 −85%, TTFT p50 −70%', measured: null,
      });
      note('cap-m', 30, { id: 'cap', measured: 'queue time p50 −88% (replayed 60 s)' });
      note('kv', 36, {
        agent: 'Memory', region: 'kv', title: 'KV tissue near full during the spike; preemptions follow',
        body: 'Occupancy crossed 95% twice. Each preemption recomputes the prompt. FP8 KV would roughly double the room.',
        predicted: 'kv-cache-dtype fp8 → preemptions −90%, ITL p90 −12%', measured: null,
      });
    } else if (this.mode === 'pd') {
      note('pd', 10, {
        agent: 'Disaggregation', region: 'compute', title: 'Prefill GPU idles between long prompts',
        body: 'GPU 0 (prefill) averages 34% while GPU 1 (decode) is saturated. A 1P1D split is over-provisioned on prefill for this mix.',
        predicted: 'aggregate both GPUs (DP2) → tok/s +38%', measured: null,
      });
    } else {
      note('chunk', 12, {
        agent: 'Scheduler', region: 'decode', title: 'Long prompts stall decode steps',
        body: 'Steps that carry more than 4k prefill tokens run 2.1× slower; every decoding request pauses with them.',
        predicted: 'max-num-batched-tokens 8192 → 2048: ITL p90 −30%, TTFT p50 +8%', measured: null,
      });
    }
  }
}
