// Right column: the experiment ledger ("specimen log") and the advisors' field notes.
import type { BenchResult, ExpConfig } from '../contract';
import { isExperiment, type Note } from '../state';
import { fmtInt, fmtMs } from '../util';

const esc = (s: unknown) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

const VERDICT: Record<string, [string, string]> = {
  confirmed: ['✓', 'confirmed'],
  rejected: ['✕', 'rejected'],
  neutral: ['≈', 'neutral'],
  measured: ['•', 'measured'],
  testing: ['◌', 'testing'],
  failed: ['!', 'failed'],
  tested: ['↺', 'already tested'],
  proposed: ['○', 'proposed'],
  superseded: ['–', 'superseded'],
};

type Deltas = NonNullable<Note['deltas']>;

const toMs = (v: string, unit: string) => parseFloat(v) * (unit === 's' ? 1000 : 1);
const toNum = (v: string) => parseFloat(v.replace(/,/g, ''));

/** Older notes carry numbers only in `measured` text; recover them so the ledger can compare. */
function parsed(n: Note): { result: Record<string, BenchResult>; deltas: Deltas } {
  if (n.result && Object.keys(n.result).length) return { result: n.result, deltas: n.deltas ?? {} };
  const result: Record<string, BenchResult> = {};
  const deltas: Deltas = {};
  for (const line of String(n.measured ?? '').split('\n')) {
    const m = /^\s*([a-z_]+):\s*(.*)$/i.exec(line);
    if (!m) continue;
    const [, name, rest] = m;
    const r: BenchResult = {};
    const d: Deltas[string] = {};
    let x = /TTFT p50\/p90 ([\d.]+)(ms|s)\/([\d.]+)(ms|s)/.exec(rest);
    if (x) {
      r.ttft_p50_ms = toMs(x[1], x[2]);
      r.ttft_p90_ms = toMs(x[3], x[4]);
    }
    x = /TTFT p90 ([\d.]+)(ms|s) → ([\d.]+)(ms|s)(?: \(([+\-−]?\d+)%\))?/.exec(rest);
    if (x) {
      r.ttft_p90_ms = toMs(x[3], x[4]);
      if (x[5]) d.ttft_p90 = parseFloat(x[5].replace('−', '-')) / 100;
    }
    x = /e2e p90 ([\d.]+)(ms|s)(?: → ([\d.]+)(ms|s))?/.exec(rest);
    if (x) r.e2e_p90_ms = x[3] ? toMs(x[3], x[4]) : toMs(x[1], x[2]);
    x = /([\d,.]+) → ([\d,.]+) tok\/s(?: \(([+\-−]?\d+)%\))?/.exec(rest);
    if (x) {
      r.tok_s = toNum(x[2]);
      if (x[3]) d.tok_s = parseFloat(x[3].replace('−', '-')) / 100;
    } else if ((x = /([\d,.]+) tok\/s/.exec(rest))) r.tok_s = toNum(x[1]);
    x = /(\d+) errors?/.exec(rest);
    if (x) r.errors = Number(x[1]);
    if (Object.keys(r).length) {
      result[name] = r;
      deltas[name] = d;
    }
  }
  return { result, deltas };
}

function primaryBench(n: Note): string | null {
  const r = parsed(n).result;
  if (r.burst) return 'burst';
  const keys = Object.keys(r);
  return keys.length ? keys[0] : null;
}

export function engineName(e?: string | null): string {
  if (!e) return 'engine';
  const k = e.toLowerCase();
  return k === 'vllm' ? 'vLLM' : k === 'sglang' ? 'SGLang' : e;
}

function gpuCount(g: ExpConfig['gpus']): number | null {
  if (g == null || g === '') return null;
  if (typeof g === 'number') return g;
  return String(g).split(/[,\s]+/).filter(Boolean).length;
}

export function configChips(c?: ExpConfig | null, max = 4): string {
  if (!c) return '';
  const chips: string[] = [`<i class="eng">${esc(engineName(c.engine))}</i>`];
  if (c.pd_prefill || c.pd_decode) {
    const np = String(c.pd_prefill ?? '').split(/\s+/).filter(Boolean).length;
    const nd = String(c.pd_decode ?? '').split(/\s+/).filter(Boolean).length;
    chips.push(`<i class="pd">${np}P${nd}D</i>`);
  } else {
    const g = gpuCount(c.gpus);
    if (c.tp && c.tp > 1) chips.push(`<i>TP${c.tp}</i>`);
    else if (g) chips.push(`<i>${g}× GPU</i>`);
  }
  const args = argChips(c.args ?? []);
  const room = Math.max(0, max - chips.length);
  chips.push(...args.slice(0, room).map((a) => `<i class="arg">${esc(a)}</i>`));
  if (args.length > room) chips.push(`<i class="more">+${args.length - room}</i>`);
  if (!args.length) chips.push(`<i class="arg dim">defaults</i>`);
  return chips.join('');
}

export function argChips(args: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (!a.startsWith('--')) continue;
    const name = a.replace(/^--/, '');
    const next = args[i + 1];
    if (next !== undefined && !next.startsWith('--')) {
      out.push(`${name} ${next.length > 18 ? next.slice(0, 17) + '…' : next}`);
      i++;
    } else out.push(name);
  }
  return out;
}

function pct(frac: number | null | undefined, lowerIsBetter: boolean): string {
  if (frac == null || !Number.isFinite(frac) || Math.abs(frac) < 0.005) return '';
  const good = lowerIsBetter ? frac < 0 : frac > 0;
  const sign = frac > 0 ? '+' : '−';
  return `<em class="${good ? 'good' : 'bad'}">${sign}${Math.round(Math.abs(frac) * 100)}%</em>`;
}

function nums(n: Note): string {
  const b = primaryBench(n);
  if (!b) {
    if (n.status === 'testing' || n.status === 'failed') {
      const t = String(n.measured ?? '');
      return `<div class="nums run">${esc(t.length > 150 ? t.slice(0, 149) + '…' : t)}</div>`;
    }
    return '';
  }
  const p = parsed(n);
  const r: BenchResult = p.result[b] ?? {};
  const d = p.deltas[b] ?? {};
  const ttft = r.ttft_p90_ms != null ? `<span>TTFT p90 <b>${fmtMs(r.ttft_p90_ms)}</b> ${pct(d.ttft_p90, true)}</span>` : '';
  const tok = r.tok_s != null ? `<span><b>${fmtInt(r.tok_s)}</b> tok/s ${pct(d.tok_s, false)}</span>` : '';
  const err = r.errors ? `<span class="err">${r.errors} errors</span>` : '';
  const vs = n.baseline ? `<span class="vs">${esc(b)} · vs ${esc(n.baseline)}</span>` : `<span class="vs">${esc(b)}</span>`;
  return `<div class="nums">${ttft}${tok}${err}${vs}</div>`;
}

function verdict(st: string): string {
  const [mark, word] = VERDICT[st] ?? ['·', st || 'info'];
  return `<span class="verdict v-${esc(st)}"><span class="mk">${mark}</span>${esc(word)}</span>`;
}

/** Tiny dot plot: burst TTFT p90 per experiment, with the best-so-far step line. */
function chart(exps: Note[]): string {
  const pts = exps
    .map((n, i) => ({ n, i, v: parsed(n).result.burst?.ttft_p90_ms ?? null }))
    .filter((p): p is { n: Note; i: number; v: number } => p.v != null && p.v > 0);
  if (pts.length < 2) return '';
  const W = 300;
  const H = 40;
  const lo = Math.min(...pts.map((p) => p.v));
  const hi = Math.max(...pts.map((p) => p.v));
  const ly = (v: number) => {
    const a = Math.log(lo * 0.9);
    const b = Math.log(hi * 1.1);
    return 6 + (H - 14) * (1 - (Math.log(v) - a) / (b - a || 1));
  };
  const xs = (k: number) => 8 + ((W - 16) * k) / Math.max(1, pts.length - 1);
  let best = Infinity;
  const step: string[] = [];
  pts.forEach((p, k) => {
    const counts = p.n.status === 'confirmed' || k === 0;
    if (counts && p.v < best) best = p.v;
    if (Number.isFinite(best)) step.push(`${step.length ? 'L' : 'M'}${xs(k).toFixed(1)},${ly(best).toFixed(1)}`);
    if (k < pts.length - 1 && Number.isFinite(best)) step.push(`L${xs(k + 1).toFixed(1)},${ly(best).toFixed(1)}`);
  });
  const color: Record<string, string> = { confirmed: 'var(--sap)', rejected: 'var(--crimson)', neutral: 'var(--grey)', measured: 'var(--ink)' };
  const dots = pts
    .map((p, k) => {
      const c = color[p.n.status ?? ''] ?? 'var(--ultra)';
      const hollow = p.n.status === 'testing' || p.n.status === 'failed';
      return `<circle cx="${xs(k).toFixed(1)}" cy="${ly(p.v).toFixed(1)}" r="3.1" fill="${hollow ? 'none' : c}" stroke="${c}" stroke-width="1.2"><title>${esc(p.n.id)} · ${fmtMs(p.v)}</title></circle>`;
    })
    .join('');
  return `<figure class="chart">
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-label="burst TTFT p90 per experiment">
      <path d="${step.join(' ')}" fill="none" stroke="var(--sap)" stroke-width="1" stroke-dasharray="2 2"/>
      ${dots}
    </svg>
    <figcaption><span>burst TTFT p90 per run (log)</span><span>${fmtMs(lo)} – ${fmtMs(hi)}</span></figcaption>
  </figure>`;
}

function bestLine(active: Note[]): string {
  const burst = (n: Note) => parsed(n).result.burst!;
  const withBurst = active.filter((n) => parsed(n).result.burst);
  if (!withBurst.length) return `<div class="best dim">No measured experiments yet.</div>`;
  const base = withBurst[0];
  const confirmed = withBurst.filter((n) => n.status === 'confirmed');
  if (!confirmed.length) {
    const r = burst(base);
    return `<div class="best dim">No confirmed improvement yet. Reference ${esc(base.id)}: burst TTFT p90 <b>${fmtMs(r.ttft_p90_ms)}</b> · <b>${r.tok_s != null ? fmtInt(r.tok_s) : '—'}</b> tok/s</div>`;
  }
  const pool = [base, ...confirmed];
  const bt = pool.reduce((a, n) => ((burst(n).ttft_p90_ms ?? Infinity) < (burst(a).ttft_p90_ms ?? Infinity) ? n : a));
  const bk = pool.reduce((a, n) => ((burst(n).tok_s ?? -1) > (burst(a).tok_s ?? -1) ? n : a));
  const t0 = burst(base).ttft_p90_ms;
  const k0 = burst(base).tok_s;
  const t1 = burst(bt).ttft_p90_ms;
  const k1 = burst(bk).tok_s;
  return `<div class="best">
    <span class="lab">Best so far</span>
    <span>burst TTFT p90 <b>${fmtMs(t1)}</b> ${t0 && t1 ? pct(t1 / t0 - 1, true) : ''} <i>${esc(bt.id)}</i></span>
    <span><b>${k1 != null ? fmtInt(k1) : '—'}</b> tok/s ${k0 && k1 ? pct(k1 / k0 - 1, false) : ''} <i>${esc(bk.id)}</i></span>
    <span class="vs">vs first reference ${esc(base.id)}</span>
  </div>`;
}

// ---- scenarios: only compare experiments run on the same GPUs at the same load ----

function gpuSet(c?: ExpConfig | null): number | null {
  if (!c) return null;
  const extra = c as ExpConfig & { replicas?: string | null };
  const ids = new Set<string>();
  for (const v of [c.gpus, extra.replicas, c.pd_prefill, c.pd_decode]) {
    if (v == null || v === '') continue;
    for (const t of String(v).split(/[\s,]+/)) if (t !== '') ids.add(t);
  }
  return ids.size || null;
}

function topology(c?: ExpConfig | null): string {
  if (!c) return '';
  const extra = c as ExpConfig & { replicas?: string | null };
  if (c.pd_prefill || c.pd_decode) {
    const np = String(c.pd_prefill ?? '').split(/\s+/).filter(Boolean).length;
    const nd = String(c.pd_decode ?? '').split(/\s+/).filter(Boolean).length;
    return `${np}P${nd}D`;
  }
  const reps = String(extra.replicas ?? '').split(/\s+/).filter(Boolean).length;
  if (reps > 1) return `${reps} replicas`;
  const a = c.args ?? [];
  const dp = a.indexOf('--data-parallel-size');
  if (dp >= 0 && a[dp + 1]) return `DP=${a[dp + 1]}`;
  if (c.tp && Number(c.tp) > 1) return `TP${c.tp}`;
  return '';
}

function describe(n: Note): string {
  return topology(n.config) || engineName(n.config?.engine);
}

/** Load of an experiment: its own speed, else that of experiments that used it as their baseline. */
function speedOf(n: Note, all: Note[]): number | null {
  if (typeof n.speed === 'number') return n.speed;
  const child = all.find((x) => x.baseline === n.id && typeof x.speed === 'number');
  if (child) return child.speed as number;
  const m = /at (\d+(?:\.\d+)?)× speed/.exec(String(n.measured ?? ''));
  return m ? Number(m[1]) : null;
}

function gpuName(all: Note[]): string {
  for (const n of all) {
    const m = /\b([A-Z]{1,2}\d{2,4}[A-Z]?)\b/.exec(String(n.title ?? ''));
    if (m) return m[1];
  }
  return 'GPU';
}

interface Scenario {
  key: string;
  gpus: number | null;
  speed: number | null;
  exps: Note[];
}

function scenarios(active: Note[], all: Note[]): Scenario[] {
  const map = new Map<string, Scenario>();
  for (const n of active) {
    const g = gpuSet(n.config);
    const sp = speedOf(n, all);
    const key = `${g ?? '?'}|${sp ?? '?'}`;
    let sc = map.get(key);
    if (!sc) map.set(key, (sc = { key, gpus: g, speed: sp, exps: [] }));
    sc.exps.push(n);
  }
  return [...map.values()]
    .filter((sc) => sc.exps.some((n) => parsed(n).result.burst || n.status === 'testing'))
    .sort((a, b) => (a.gpus ?? 99) - (b.gpus ?? 99) || (a.speed ?? 99) - (b.speed ?? 99));
}

function scenarioTitle(sc: Scenario, gpu: string): string {
  return `${sc.gpus ?? '?'}× ${gpu} · ${sc.speed != null ? `${sc.speed}× load` : 'load not recorded'}`;
}

function scenarioBest(sc: Scenario): string {
  const burst = (n: Note) => parsed(n).result.burst;
  const withB = sc.exps.filter((n) => burst(n)?.ttft_p90_ms != null);
  if (!withB.length) return `<div class="best dim">No measured runs yet.</div>`;
  // reference: the run the others compared against, else the first one measured
  const refId = sc.exps.map((n) => n.baseline).find((b) => b && withB.some((x) => x.id === b));
  const ref = withB.find((n) => n.id === refId) ?? withB[0];
  const r0 = burst(ref)!;
  const confirmed = withB.filter((n) => n.status === 'confirmed' && n !== ref);
  const best = confirmed.reduce<Note | null>((a, n) => (!a || (burst(n)!.ttft_p90_ms ?? Infinity) < (burst(a)!.ttft_p90_ms ?? Infinity) ? n : a), null);
  const refTxt = `<span class="vs">vs ${fmtMs(r0.ttft_p90_ms)} reference (${esc(describe(ref))}, ${esc(ref.id)})</span>`;
  if (!best || (burst(best)!.ttft_p90_ms ?? Infinity) >= (r0.ttft_p90_ms ?? Infinity)) {
    return `<div class="best dim">No confirmed improvement yet — reference burst TTFT p90 <b>${fmtMs(r0.ttft_p90_ms)}</b> (${esc(describe(ref))}, ${esc(ref.id)})</div>`;
  }
  const b = burst(best)!;
  return `<div class="best">
    <span>best burst TTFT p90 <b>${fmtMs(b.ttft_p90_ms)}</b> ${pct((b.ttft_p90_ms ?? 0) / (r0.ttft_p90_ms ?? 1) - 1, true)} <i>(${esc(describe(best))}, ${esc(best.id)})</i></span>
    <span><b>${b.tok_s != null ? fmtInt(b.tok_s) : '—'}</b> tok/s ${r0.tok_s && b.tok_s ? pct(b.tok_s / r0.tok_s - 1, false) : ''}</span>
    ${refTxt}
  </div>`;
}

export function ledgerHtml(notes: Note[]): string {
  const exps = notes.filter(isExperiment).sort((a, b) => a.firstSeen - b.firstSeen || a.id.localeCompare(b.id, 'en', { numeric: true }));
  const active = exps;
  if (!exps.length) return '';
  const gpu = gpuName(exps);
  const scs = scenarios(active, exps);
  const scOf = new Map<string, string>();
  for (const sc of scs) for (const n of sc.exps) scOf.set(n.id, `${sc.gpus ?? '?'}× · ${sc.speed != null ? `${sc.speed}×` : '?'}`);
  const scBlocks = scs
    .map((sc) => `<div class="scenario"><div class="sct">${esc(scenarioTitle(sc, gpu))}</div>${scenarioBest(sc)}${chart(sc.exps)}</div>`)
    .join('');
  const rows = [...active]
    .reverse()
    .map(
      (n) => `<li class="exp s-${esc(n.status ?? 'info')}">
        <div class="l1"><span class="id">${esc(n.id)}<span class="sc">${esc(scOf.get(n.id) ?? '')}</span></span>${verdict(n.status ?? 'info')}</div>
        <div class="t">${esc(n.title ?? '')}</div>
        <div class="chips">${configChips(n.config)}</div>
        ${nums(n)}
      </li>`,
    )
    .join('');
  return `<section class="ledger">
    <div class="kick">Specimen log</div>
    <h2>Experiments <span class="count">${active.length}</span></h2>
    ${scBlocks}
    <ol>${rows}</ol>
  </section>`;
}

function lifecycle(n: Note): string {
  const st = n.status ?? 'info';
  if (!['proposed', 'testing', 'confirmed', 'rejected', 'neutral', 'failed', 'tested', 'measured'].includes(st)) return '';
  const stage = st === 'proposed' ? 0 : st === 'testing' ? 1 : 2;
  const final = stage === 2 ? (VERDICT[st]?.[1] ?? st) : 'verdict';
  const step = (i: number, label: string, cls = '') => `<span class="stp ${i < stage ? 'done' : i === stage ? 'cur' : ''} ${cls}">${label}</span>`;
  return `<div class="life">${step(0, 'proposed')}<span class="ln ${stage > 0 ? 'done' : ''}"></span>${step(1, 'testing')}<span class="ln ${stage > 1 ? 'done' : ''}"></span>${step(2, esc(final), stage === 2 ? `v-${esc(st)}` : '')}</div>`;
}

function fieldNote(n: Note, exps: Map<string, Note>): string {
  const st = n.status ?? (n.measured ? 'measured' : n.predicted ? 'proposed' : 'info');
  const change = n.experiment?.args?.length
    ? `<div class="change"><span class="lab">change</span>${argChips(n.experiment.args).map((a) => `<i class="arg">${esc(a)}</i>`).join('')}${n.experiment.bench?.length ? `<span class="on">on ${esc(n.experiment.bench.join(', '))}</span>` : ''}${n.experiment.goal ? `<span class="on">goal ${esc(n.experiment.goal.replace('_', ' '))}</span>` : ''}</div>`
    : '';
  const linked = n.experiment_id ? exps.get(n.experiment_id) : undefined;
  const link = n.experiment_id
    ? `<div class="linked"><span class="lab">test</span><b>${esc(n.experiment_id)}</b>${linked ? nums(linked).replace('class="nums"', 'class="nums inline"') : ''}</div>`
    : '';
  return `<div class="note ${n.sample ? 'sample' : ''} ${st === 'cleared' ? 'cleared' : ''}">
    <div class="who"><span>${esc(n.agent ?? 'advisor')}${n.region ? ` · ${esc(n.region)}` : ''}</span><span class="status ${esc(st)}">${esc(st)}</span></div>
    <h3>${esc(n.title ?? '')}</h3>
    ${n.body ? `<p>${esc(n.body)}</p>` : ''}
    ${change}
    ${lifecycle(n)}
    ${n.predicted || n.measured ? `<div class="pm">${n.predicted ? `<div><span class="lab">predicted</span>${esc(n.predicted)}</div>` : ''}${n.measured ? `<div><span class="lab">measured</span><span class="meas">${esc(n.measured)}</span></div>` : ''}</div>` : ''}
    ${link}
  </div>`;
}

export function fieldNotesHtml(notes: Note[]): string {
  const exps = new Map(notes.filter(isExperiment).map((n) => [n.id, n]));
  const field = notes
    .filter((n) => !isExperiment(n))
    .sort((a, b) => {
      // live lifecycles first, then newest
      const live = (n: Note) => (n.status === 'testing' ? 2 : n.status === 'proposed' ? 1 : 0);
      return live(b) - live(a) || b.firstSeen - a.firstSeen;
    })
    .slice(0, 10);
  const body = field.length
    ? field.map((n) => fieldNote(n, exps)).join('')
    : `<div class="empty">No notes yet. Advisors write here when something in the specimen needs attention, with a predicted gain and a measured one after they test it.</div>`;
  return `<section class="field"><div class="kick">Advisors</div><h2>Field notes</h2>${body}</section>`;
}
