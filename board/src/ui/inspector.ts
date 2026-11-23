// Click-to-inspect: a pinned, draggable panel describing whatever was clicked, from live data.
import type { Hit, Scene } from '../scene';
import { displayRid, ownerInCell, trafficClass, type KvTokens, type Req, type Store } from '../state';
import { fmtInt, fmtMs } from '../util';
import { draggable } from './draggable';
import { tag, TAG_LEGEND, type Src } from './tags';
import type { Follow } from './follow';

const esc = (s: unknown) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
const pct = (x: number | null | undefined, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(x < 0.1 ? d : 0)}%`);
const fmtK = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${Math.round(n)}`);

function q(xs: number[], p: number): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.round(p * (s.length - 1)))];
}

/** Inline sparkline: values over time, with min/max/now labels. */
function spark(vals: (number | null)[], label: string, fmt: (v: number) => string, ink = '#1c2c8e', src: Src = 'm'): string {
  label = `${label} ${tag(src)}`;
  const v = vals.filter((x): x is number => x != null && Number.isFinite(x));
  if (v.length < 2) return `<div class="spk"><span class="sl">${label}</span><span class="sv">${v.length ? fmt(v[0]) : '—'}</span><em>collecting…</em></div>`;
  const W = 230;
  const H = 30;
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  const span = hi - lo || 1;
  const pts = vals
    .map((x, i) => (x == null ? null : `${((i / (vals.length - 1)) * W).toFixed(1)},${(H - 2 - ((x - lo) / span) * (H - 4)).toFixed(1)}`))
    .filter(Boolean)
    .join(' ');
  return `<div class="spk"><span class="sl">${label}</span><span class="sv">${fmt(v[v.length - 1])}</span>
    <svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><polyline points="${pts}" fill="none" stroke="${ink}" stroke-width="1.3"/></svg>
    <span class="rng">${fmt(lo)} – ${fmt(hi)} · last ${Math.round((vals.length * 1) / 60) || 1} min</span></div>`;
}

// where each inspector value comes from (see tags.ts); a value shown with ≈ is an estimate
const SRC: Record<string, Src> = {
  'arrive (proxy)': 'm', 'admitted to engine': 'm', 'prefill done (prefill GPU)': 'm', 'first token': 'm', done: 'm',
  phase: 'd', 'prompt tokens': 'm', 'from prefix cache': 'm', generated: 'm', 'decode rate': 'd', 'KV blocks held': 'd',
  'traffic class': 'd', instance: 'm', 'prefix family': 'd',
  capacity: 'm', 'live (in use)': 'm', 'cached, reusable (est.)': 'e', free: 'e', 'prefix cache (live)': 'd', evictions: 'm', 'KV read by decode': 'd',
  'members seen': 'd', 'running now': 'd', 'hits / misses': 'd', 'shared prefix': 'd', 'core blocks': 'd',
  'routed-token share': 'm', rank: 'd', window: 'm', 'max / mean expert load': 'm', 'rank among layers': 'd',
  'tokens/s': 'd', 'requests decoding': 'd', 'TTFT p50 / p90 · 1 min': 'd', 'by class': 'd',
};
const row = (k: string, v: string, src?: Src) => {
  const s = /≈|assumed/.test(v) ? 'e' : (src ?? SRC[k]);
  return `<tr><td>${k}</td><td>${v}${s && v !== '—' ? ` ${tag(s)}` : ''}</td></tr>`;
};

export class Inspector {
  private el: HTMLElement;
  private head: HTMLElement;
  private title: HTMLElement;
  private body: HTMLElement;
  hit: Hit | null = null;
  private req: Req | null = null;
  private last = 0;

  constructor(root: HTMLElement, private store: Store, private scene: Scene, private follow: Follow) {
    this.el = document.createElement('section');
    this.el.className = 'inspector';
    this.el.style.display = 'none';
    this.head = document.createElement('header');
    this.title = document.createElement('span');
    const close = document.createElement('button');
    close.type = 'button';
    close.textContent = '×';
    close.title = 'Close (Esc)';
    close.addEventListener('click', () => this.close());
    this.head.append(this.title, close);
    this.body = document.createElement('div');
    this.body.className = 'ib';
    this.el.append(this.head, this.body);
    root.appendChild(this.el);
    draggable(this.el, this.head);
    this.body.addEventListener('click', (e) => {
      const b = (e.target as HTMLElement).closest<HTMLElement>('[data-act]');
      if (!b) return;
      if (b.dataset.act === 'follow' && this.req) {
        if (this.follow.rid === this.req.rid && this.follow.enabled) this.follow.toggle(performance.now());
        else this.follow.followReq(this.req.rid, performance.now());
        this.render(performance.now());
      }
    });
  }

  get open(): boolean {
    return !!this.hit;
  }

  /** Dock the inspector: right/top offsets, its width, and the room below it (CSS px). */
  place(right: number, top: number, width = 340, maxH = 0): void {
    this.el.style.width = `${width}px`;
    const ib = this.el.querySelector<HTMLElement>('.ib');
    if (ib && maxH) ib.style.maxHeight = `${Math.max(200, maxH - 48)}px`;
    if (this.el.dataset.moved) return;
    this.el.style.right = `${right}px`;
    this.el.style.top = `${top}px`;
  }

  show(hit: Hit): void {
    this.hit = hit;
    this.req = hit.kind === 'request' ? (this.store.reqs.get(hit.id) ?? null) : null;
    this.el.style.display = 'block';
    this.el.classList.remove('collapsed');
    this.render(performance.now());
  }

  close(): void {
    this.hit = null;
    this.req = null;
    this.el.style.display = 'none';
  }

  tick(now: number): void {
    if (!this.hit || now - this.last < 400) return;
    this.render(now);
  }

  private render(now: number): void {
    this.last = now;
    const h = this.hit!;
    const [t, html] =
      h.kind === 'request' ? this.request() :
      h.kind === 'body' ? this.kvBody(h.id, now) :
      h.kind === 'core' ? this.core(h.id) :
      h.kind === 'expert' ? this.expert(Number(h.id)) :
      h.kind === 'layer' ? this.layer(Number(h.id)) :
      h.kind === 'moe' ? this.moe() :
      h.kind === 'cell' ? this.cell(h.id) :
      h.kind === 'branch' ? this.branch(h.id, now) :
      h.kind === 'trunk' ? this.trunk(h.id, now) :
      this.gpu(Number(h.id));
    this.title.innerHTML = t;
    this.body.innerHTML = `${html}<p class="srcleg">${TAG_LEGEND}</p>`;
  }

  // ---- request -------------------------------------------------------------------------------
  private request(): [string, string] {
    const r = this.req;
    if (!r) return ['Request', '<p class="dim">This request has left the plate.</p>'];
    const e = this.scene.ents.get(r.rid);
    const at = (t?: number) => (t == null ? '—' : `+${fmtMs(t - r.tArrive)}`);
    const tl = [
      ['arrive (proxy)', '+0 ms'],
      ['admitted to engine', at(r.tAdmit)],
      ...(r.prefillMs != null ? [['prefill done (prefill GPU)', `${fmtMs(r.prefillMs)} long`]] : []),
      ['first token', at(r.tFirst)],
      ['done', r.phase === 'failed' ? esc(r.error ?? 'failed') : at(r.tDone)],
    ];
    const end = r.tDone ?? this.store.wallNow();
    const rate = r.tFirst != null && end > r.tFirst ? (r.chunks / (end - r.tFirst)) * 1000 : null;
    const b = e ? this.scene.bodies[e.body] : undefined;
    const o = e ? this.scene.ownerOf(e) : null;
    const fam = r.family ? this.store.families.get(r.family) : undefined;
    const following = this.follow.enabled && this.follow.rid === r.rid;
    const html = `
      <table class="tl">${tl.map(([k, v]) => row(k, v)).join('')}</table>
      <table>
        ${row('phase', esc(r.phase))}
        ${row('prompt tokens', r.promptTokens != null ? fmtInt(r.promptTokens) : r.promptChars ? `≈${fmtInt(r.promptChars / 3.8)}` : '—')}
        ${row('from prefix cache', r.cachedTokens != null ? fmtInt(r.cachedTokens) : '—')}
        ${row('generated', `${fmtInt(r.completionTokens ?? r.chunks)}${r.maxTokens ? ` / ${fmtInt(r.maxTokens)}` : ''}`)}
        ${row('decode rate', rate != null ? `${rate.toFixed(1)} tok/s` : '—')}
        ${o ? row('KV held (engine)', `${fmtInt(o.tokens)} tok · ${fmtInt(o.cachedTokens ?? 0)} from cache${o.sharedTokens ? ` · ${fmtInt(o.sharedTokens)} shared with other running requests` : ''}`, 'm')
          : row('KV blocks held', e && e.blocks ? `${fmtInt(e.blocks)}${e.shared ? ` (${fmtInt(e.shared)} in the shared core)` : ''}${b ? ` × ${b.cap.blockSize} tok` : ''}` : '—')}
        ${row('traffic class', esc(trafficClass(r)))}
        ${row('instance', esc(b?.name ?? r.worker ?? r.decodeWorker ?? '—'))}
        ${row('prefix family', r.family ? `${esc(r.family)} · ${esc(r.prefixOutcome ?? 'no stats yet')}${fam ? ` · ${fam.hits} hits / ${fam.misses} misses` : ''}` : '—')}
      </table>
      <button type="button" class="act ${following ? 'on' : ''}" data-act="follow">${following ? '◉ following — stop' : '◎ follow this request'}</button>`;
    return [`<b>${esc(displayRid(r.rid))}</b> <i>${esc(r.workload ?? '')}</i>`, html];
  }

  // ---- KV body -------------------------------------------------------------------------------
  private kvBody(name: string, now: number): [string, string] {
    const bi = Math.max(0, this.scene.bodies.findIndex((b) => b.name === name));
    const b = this.scene.bodies[bi];
    const cap = b?.cap ?? this.store.kvCapacity();
    const inst = this.store.instances(now).find((i) => i.name === name);
    const usage = inst?.kv_usage ?? this.store.engine?.kv_usage ?? null;
    const km = this.scene.mapFor(bi);
    if (km?.tokens && km.capacityTokens) return this.kvBody2(name, km.tokens, km.capacityTokens, bi, now);
    const fromMap = !!(km && km.usedTokens != null && km.capacityTokens);
    const tokPerBlock = cap.blockSize || 1;
    const used = fromMap ? Math.min(km!.usedTokens!, km!.ownedTokens ?? 0) / tokPerBlock : usage != null ? usage * cap.blocks : null;
    const cached = fromMap ? (km!.usedTokens! - Math.min(km!.usedTokens!, km!.ownedTokens ?? 0)) / tokPerBlock : this.scene.cachedBlocks(bi);
    const free = fromMap ? (km!.capacityTokens - km!.usedTokens!) / tokPerBlock : used != null ? Math.max(0, cap.blocks - used - cached) : null;
    const fams = [...this.scene.famCores.values()].filter((f) => f.body === bi).sort((a, c) => c.blocks - a.blocks).slice(0, 5);
    const pl = this.store.prefixLive(now);
    const hist = (this.store.hist.inst.get(name) ?? []).length > 1 ? this.store.hist.inst.get(name)!.map((x) => x.kv) : this.store.hist.kv.map((x) => x.v);
    const html = `
      <table>
        ${row('capacity', `${cap.blockSize === 1 ? `${fmtInt(cap.tokens)} tok · page 1` : `${fmtInt(cap.blocks)} × ${cap.blockSize} tok = ${fmtK(cap.tokens)} tok`}${cap.known ? '' : ' (assumed)'}`)}
        ${row('live (in use)', used != null ? `${fmtInt(used)} blocks · ${pct(used / cap.blocks)}` : '—', fromMap ? 'd' : 'm')}
        ${fromMap ? row('cached (used − live)', `${fmtInt(cached)} blocks · ${pct(cached / cap.blocks)}`, 'd') : row('cached, reusable (est.)', `≈${fmtInt(cached)} blocks · ${pct(cached / cap.blocks)}`)}
        ${row('free', free != null ? `${fmtInt(free)} blocks · ${pct(free / cap.blocks)}` : '—', fromMap ? 'm' : 'e')}
        ${fromMap ? row('source', 'engine allocator (kv.map)', 'm') : ''}
        ${row('prefix cache (live)', pl.tokenRate != null ? `${pct(pl.tokenRate, 0)} of prompt tok · ${pl.hits} hit / ${pl.misses} miss` : this.store.engine?.prefix_hit_rate != null ? `${pct(this.store.engine.prefix_hit_rate, 0)} (engine)` : '—')}
        ${row('evictions', this.store.evictSeen ? fmtInt(this.store.evictTotal) : 'not reported by this engine')}
        ${row('KV read by decode', `${fmtK(this.scene.kvRead)} tok/s`)}
      </table>
      ${spark(hist, 'KV in use', (v) => pct(v))}
      <h4>Largest prefix families</h4>
      ${fams.length ? `<table>${fams.map((f) => {
        const st = this.store.families.get(f.id);
        const hr = st && st.hits + st.misses ? st.hits / (st.hits + st.misses) : null;
        const n = st?.members ?? 0;
        return row(esc(f.id), `${fmtInt(f.blocks)} blocks · ${n} ${n === 1 ? 'member' : 'members'} · hit ${pct(hr, 0)}`, 'd');
      }).join('')}</table>` : '<p class="dim">No shared prefix cores right now.</p>'}`;
    return [`KV cache · <b>${esc(name)}</b>`, html];
  }

  /** KV body from kv.map format 2: every slot classified, each once. */
  private kvBody2(name: string, t: KvTokens, cap: number, bi: number, now: number): [string, string] {
    const tok = (n: number) => `${fmtInt(n)} tok · ${pct(n / cap)}`;
    const cores = [...this.scene.famCores.values()].filter((f) => f.body === bi).sort((a, c) => c.tokens - a.tokens).slice(0, 5);
    const pl = this.store.prefixLive(now);
    const hist = (this.store.hist.inst.get(name) ?? []).length > 1 ? this.store.hist.inst.get(name)!.map((x) => x.kv) : this.store.hist.kv.map((x) => x.v);
    const html = `
      <table>
        ${row('capacity', `${fmtInt(cap)} tok`, 'm')}
        ${row('own (running requests)', tok(t.own), 'm')}
        ${row('reused from the cache', tok(t.reused), 'm')}
        ${row('shared by 2+ running', tok(t.shared), 'm')}
        ${row('cached, no running owner', tok(t.cached), 'm')}
        ${row('free', tok(t.free), 'm')}
        ${row('source', 'engine KV tables (kv.map), every slot once', 'm')}
        ${row('prefix cache (live)', pl.tokenRate != null ? `${pct(pl.tokenRate, 0)} of prompt tok · ${pl.hits} hit / ${pl.misses} miss` : this.store.engine?.prefix_hit_rate != null ? `${pct(this.store.engine.prefix_hit_rate, 0)} (engine)` : '—')}
        ${row('KV read by decode', `${fmtK(this.scene.kvRead)} tok/s`)}
      </table>
      ${spark(hist, 'KV in use', (v) => pct(v))}
      <h4>Shared now</h4>
      ${cores.length ? `<table>${cores.map((f) => row(esc(f.id.replace(/^g:/, '').replace(/~[0-9a-z]+$/i, '')), `${fmtInt(f.tokens)} tok · held by ${f.members} running requests`, f.measured ? 'm' : 'e')).join('')}</table>` : '<p class="dim">No running requests share KV slots right now.</p>'}`;
    return [`KV cache · <b>${esc(name)}</b>`, html];
  }

  private cell(id: string): [string, string] {
    const [bi, bin] = id.split(':').map(Number);
    const m = this.scene.mapFor(bi);
    if (!m) return ['KV cell', '<p class="dim">The memory map is not reported yet.</p>'];
    if (m.levels) {
      const L = m.levels;
      const n = (lv: Uint8Array) => Math.round((lv[bin] / 255) * m.binTokens);
      const holders = m.owners.filter((o) => ownerInCell(o, bin));
      const html = `<table>
        ${row('own', `≈${fmtInt(n(L.own))} tok`, 'd')}
        ${row('reused from the cache', `≈${fmtInt(n(L.reused))} tok`, 'd')}
        ${row('shared by 2+ running', `≈${fmtInt(n(L.shared))} tok`, 'd')}
        ${row('cached, no running owner', `≈${fmtInt(n(L.cached))} tok`, 'd')}
        ${row('free', `≈${fmtInt(n(L.free))} tok`, 'd')}
        ${row('cell', `${fmtInt(bin)} of ${fmtInt(m.bins)} · ${fmtInt(m.binTokens)} tokens`, 'm')}
      </table>
      ${holders.length ? `<h4>Running requests with tokens here</h4><table>${holders.map((o) => {
        const r = this.store.reqs.get(o.rid);
        return row(esc(displayRid(o.rid)), `${fmtInt(o.tokens)} tok · ${fmtInt(o.cachedTokens ?? 0)} from cache${o.sharedTokens ? ` · ${fmtInt(o.sharedTokens)} shared` : ''}${r ? ` · ${esc(trafficClass(r))}` : ''}`, 'm');
      }).join('')}</table>` : ''}
      <p class="dim">Token counts per cell are rounded to 1/255 of the cell. Pages are scattered: a request's tokens can sit in many cells.</p>`;
      return [`KV cell <b>#${bin}</b>`, html];
    }
    const st = m.state[bin];
    const ownersIdx = st >= 0 ? [st] : st === -3 || st === -4 ? (m.shared.get(bin) ?? []) : [];
    const state =
      st === -1 ? 'free' :
      st === -2 ? 'cached prefix — used, no running request owns it' :
      st === -3 ? `shared prefix — ${ownersIdx.length} running requests reuse it now` :
      st === -4 ? `co-located — ${ownersIdx.length} requests' own tokens share this ${fmtInt(m.binTokens)}-token cell` :
      m.prefix[bin] ? 'reused prefix of one request (from the radix cache, not shared right now)' : 'own tokens of one request';
    const inR = (rs: [number, number][], b: number) => rs.some(([s0, e]) => b >= s0 && b < e);
    const owners = ownersIdx
      .map((k) => {
        const o = m.owners[k];
        const r = this.store.reqs.get(o.rid);
        const via = inR(o.prefixRanges, bin) ? 'prefix' : inR(o.ownRanges, bin) ? 'own' : '—';
        const split = o.prefixTokens != null ? ` · prefix ${fmtInt(o.prefixTokens)} / own ${fmtInt(Math.max(0, o.tokens - o.prefixTokens))}` : '';
        return row(esc(displayRid(o.rid)), `here via <b>${via}</b> · ${fmtInt(o.tokens)} tok${split}${r ? ` · ${esc(trafficClass(r))}${r.family ? ` · ${esc(r.family)}` : ''}` : ''}`, 'm');
      })
      .join('');
    const html = `<table>
      ${row('state', esc(state), 'm')}
      ${row('cell', `${fmtInt(bin)} of ${fmtInt(m.bins)} · tokens ${fmtInt(bin * m.binTokens)}–${fmtInt((bin + 1) * m.binTokens - 1)}`, 'm')}
      ${row('1 cell', `${fmtInt(m.binTokens)} tokens`, 'm')}
    </table>
    ${owners ? `<h4>Owner${ownersIdx.length > 1 ? 's' : ''} · prefix / own split</h4><table>${owners}</table>` : ''}`;
    return [`KV cell <b>#${bin}</b>`, html];
  }

  private core(id: string): [string, string] {
    const f = this.scene.famCores.get(id);
    if (!f) return ['Shared prefix', '<p class="dim">No longer shared: fewer than two running requests hold it.</p>'];
    const members = [...this.scene.ents.values()].filter((e) => e.coreId === id);
    const html = `<table>
      ${row('shared tokens', `${fmtInt(f.tokens)} (held once in memory)`, f.measured ? 'm' : 'e')}
      ${row('held by', `${members.length} running requests`, f.measured ? 'm' : 'e')}
      ${row('source', f.measured ? 'engine KV tables: the same slots in each request' : 'estimate: same prompt start, each reused cached tokens', f.measured ? 'm' : 'e')}
    </table>
    <table>${members.map((e) => row(esc(displayRid(e.req.rid)), esc(trafficClass(e.req)))).join('')}</table>
    <p class="dim">The core is drawn once; each member's colony holds only its own tokens.</p>`;
    return [`Shared prefix · <b>${esc(id.replace(/^g:/, '').replace(/~[0-9a-z]+$/i, ''))}</b>`, html];
  }

  // ---- experts -------------------------------------------------------------------------------
  private expert(i: number): [string, string] {
    const m = this.store.moe;
    const load = m?.load ?? [];
    if (!load.length) return [`Expert ${i}`, '<p class="dim">Expert routing is not reported yet.</p>'];
    const sum = load.reduce((a, b) => a + b, 0) || 1;
    const share = load[i] / sum;
    const rank = [...load].sort((a, b) => b - a).indexOf(load[i]) + 1;
    const hist = this.store.hist.moe.map((h) => (h.load[i] ?? 0) / (h.load.reduce((a, b) => a + b, 0) || 1));
    const html = `<table>
      ${row('routed-token share', `${(share * 100).toFixed(2)}% · ${(share * load.length).toFixed(2)}× the mean`)}
      ${row('rank', `${rank} of ${load.length}`)}
      ${row('window', m?.window_s ? `${m.window_s.toFixed(1)} s` : '—')}
    </table>
    ${spark(hist, 'share over time', (v) => `${(v * 100).toFixed(2)}%`, '#3a5a18')}
    <p class="dim">Per-layer routing for a single expert is not reported; the strip above the ring shows imbalance per layer.</p>`;
    return [`Expert <b>#${i}</b>`, html];
  }

  private layer(l: number): [string, string] {
    const lay = this.store.moe?.layer_imbalance ?? [];
    if (!lay.length) return [`Layer ${l}`, '<p class="dim">Per-layer data not reported yet.</p>'];
    const v = lay[l];
    const rank = [...lay].sort((a, b) => b - a).indexOf(v) + 1;
    const hist = this.store.hist.moe.map((h) => h.layers[l] ?? null);
    const html = `<table>
      ${row('max / mean expert load', `${v.toFixed(2)}×`)}
      ${row('rank among layers', `${rank} of ${lay.length}`)}
    </table>
    ${spark(hist, 'imbalance over time', (x) => `${x.toFixed(1)}×`, '#3a5a18')}`;
    return [`MoE layer <b>${l}</b>`, html];
  }

  private moe(): [string, string] {
    const load = this.store.moe?.load ?? [];
    if (!load.length) return ['Expert routing', '<p class="dim">Not reported yet.</p>'];
    const sum = load.reduce((a, b) => a + b, 0) || 1;
    const top = load.map((v, i) => [i, v / sum] as const).sort((a, b) => b[1] - a[1]).slice(0, 8);
    return ['Expert routing', `<h4>Hottest experts</h4><table>${top.map(([i, s]) => row(`#${i}`, `${(s * 100).toFixed(2)}% · ${(s * load.length).toFixed(1)}×`, 'm')).join('')}</table><p class="dim">Click an expert on the ring or a tick in the layer strip for its history.</p>`];
  }

  // ---- decode network -------------------------------------------------------------------------
  private ttft(now: number, pred: (x: { cls: string; worker?: string }) => boolean): string {
    const xs = this.store.ttftLog.filter((x) => now - x.at < 60_000 && pred(x)).map((x) => x.ttft);
    return xs.length ? `${fmtMs(q(xs, 0.5))} / ${fmtMs(q(xs, 0.9))} <i>(${xs.length} req)</i>` : '—';
  }

  private instanceOf(worker?: string): string | null {
    return this.scene.instanceOfWorker(worker);
  }

  private branch(key: string, now: number): [string, string] {
    const b = this.scene.branches.get(key);
    const [trunk, cls] = key.split('|');
    const flight = [...this.scene.ents.values()].filter((e) => e.branch === key && e.req.phase === 'decode').length;
    const multi = this.scene.trunks.size > 1;
    const html = `<table>
      ${row('tokens/s', b ? fmtK(b.flux) : '—')}
      ${row('requests decoding', fmtInt(flight))}
      ${row('TTFT p50 / p90 · 1 min', this.ttft(now, (x) => x.cls === cls && (!multi || this.instanceOf(x.worker) === trunk)))}
    </table>`;
    return [`Traffic class · <b>${esc(cls)}</b>${multi ? ` <i>on ${esc(trunk)}</i>` : ''}`, html];
  }

  private trunk(name: string, now: number): [string, string] {
    const t = this.scene.trunks.get(name);
    const flight = [...this.scene.ents.values()].filter((e) => e.branch?.startsWith(`${name}|`) && e.req.phase === 'decode').length;
    const multi = this.scene.trunks.size > 1;
    const classes = [...this.scene.branches.values()].filter((b) => b.parent === name).map((b) => `${esc(b.name)} ${fmtK(b.flux)}`).join(' · ');
    const html = `<table>
      ${row('tokens/s', t ? fmtK(t.flux) : '—')}
      ${row('requests decoding', fmtInt(flight))}
      ${row('TTFT p50 / p90 · 1 min', this.ttft(now, (x) => !multi || this.instanceOf(x.worker) === name))}
      ${row('by class', classes || '—')}
    </table>`;
    return [`Instance · <b>${esc(name)}</b>`, html];
  }

  private gpu(i: number): [string, string] {
    const g = this.store.gpu?.gpus?.[i];
    const id = g?.i ?? i;
    const h = this.store.hist.gpu.get(id) ?? [];
    const html = `
      ${spark(h.map((x) => x.util), 'utilisation', (v) => `${Math.round(v)}%`, '#3a5a18')}
      ${spark(h.map((x) => x.mem), 'memory used', (v) => `${v.toFixed(1)} GB`, '#3a5a18')}
      ${spark(h.map((x) => x.power), 'power', (v) => `${Math.round(v)} W`, '#3a5a18')}
      ${spark(h.map((x) => x.clock), 'SM clock', (v) => `${Math.round(v)} MHz`, '#3a5a18')}`;
    return [`GPU <b>${id}</b>${g?.mem_total_gb ? ` <i>${g.mem_total_gb.toFixed(0)} GB</i>` : ''}`, html];
  }
}
