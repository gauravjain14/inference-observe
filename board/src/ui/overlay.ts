// HTML layer: plate caption, readout, legend, region labels, field notes, hover card.
import type { Camera } from '../camera';
import type { Layout, Scene } from '../scene';

let CAM: Camera | null = null;

const pctTxt = (x: number) => (x < 0.1 ? (x * 100).toFixed(1) : String(Math.round(x * 100)));
import { displayRid, type Req, type Store } from '../state';
import { draggable } from './draggable';
import { engineName, fieldNotesHtml, ledgerHtml } from './ledger';
import { tag, TAG_LEGEND } from './tags';
import { fmtInt, fmtMs } from '../util';

const esc = (s: unknown) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

function el(tag: string, cls: string, parent: HTMLElement): HTMLElement {
  const e = document.createElement(tag);
  e.className = cls;
  parent.appendChild(e);
  return e;
}

export class Overlay {
  private root: HTMLElement;
  private caption: HTMLElement;
  private readout: HTMLElement;
  private legend: HTMLElement;
  private notes: HTMLElement;
  private card: HTMLElement;
  private labels: HTMLElement[] = []; // stage captions 1-5
  private leaders: SVGSVGElement;
  private netLabels = new Map<string, HTMLElement>();
  private gpuLabels: HTMLElement[] = [];
  private bodyLabels: HTMLElement[] = [];
  private bridge: HTMLElement;
  private moeLbl: HTMLElement;
  private mapBtn: HTMLButtonElement;
  private moeWorst: HTMLElement;
  private moeHot: HTMLElement[] = [];
  private key: HTMLElement;
  private keyBtn: HTMLButtonElement;
  private keyOpen = keyNotSeen();
  private keyBack: HTMLElement;
  private leaderSegs: [number, number, number, number][] = [];
  private leaderEnd: number[] = []; // fraction of each leader drawn; < 1 when a label is in the way
  private camVersion = -1;
  private notesKey = '\u0000';
  private hover: { req: Req; x: number; y: number } | null = null;
  private L: Layout | null = null;

  constructor(root: HTMLElement, private store: Store, private scene: Scene) {
    this.root = root;
    this.caption = el('div', 'caption', root);
    this.readout = el('div', 'readout', root);
    this.legend = el('div', 'legend', root);
    this.notes = el('aside', 'notes', root);
    this.card = el('div', 'card', root);
    this.card.style.display = 'none';
    this.leaders = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.leaders.classList.add('leaders');
    root.appendChild(this.leaders);
    for (let i = 0; i < 5; i++) this.labels.push(el('div', 'lbl stage', root));
    this.bridge = el('div', 'lbl bridge', root);
    this.moeLbl = el('div', 'lbl moe', root);
    this.mapBtn = document.createElement('button');
    this.mapBtn.type = 'button';
    this.mapBtn.className = 'mapbtn';
    this.mapBtn.title = 'Toggle the KV body between the colony layout and the engine’s real block layout (M)';
    this.mapBtn.addEventListener('click', () => {
      this.scene.memMap = !this.scene.memMap;
      this.update(performance.now());
    });
    root.appendChild(this.mapBtn);
    this.moeWorst = el('div', 'lbl moe-worst', root);
    for (let i = 0; i < 3; i++) this.moeHot.push(el('div', 'lbl moe-hot', root));
    this.keyBack = el('div', 'keyback', root);
    this.keyBack.addEventListener('click', () => this.setKey(false));
    this.key = el('div', 'key', root);
    this.key.setAttribute('role', 'dialog');
    this.key.setAttribute('aria-label', 'How to read this plate');
    this.key.innerHTML = KEY_HTML;
    this.keyBtn = document.createElement('button');
    this.keyBtn.type = 'button';
    this.keyBtn.className = 'keybtn';
    this.keyBtn.textContent = '?';
    this.keyBtn.title = 'How to read this plate';
    root.appendChild(this.keyBtn);
    this.keyBtn.addEventListener('click', () => this.setKey(!this.keyOpen));
    this.key.querySelector('.close')?.addEventListener('click', () => this.setKey(false));
    window.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && this.keyOpen) this.setKey(false);
    });
    this.legend.style.display = 'none'; // the legend lives inside the key dialog
    this.syncKey();
  }

  /** Open or close the reading key; closing it once is remembered (it opens by itself only on a first visit). */
  setKey(open: boolean): void {
    this.keyOpen = open;
    if (!open) markKeySeen();
    this.syncKey();
  }

  get keyIsOpen(): boolean {
    return this.keyOpen;
  }

  private syncKey(): void {
    this.key.style.display = this.keyOpen ? 'block' : 'none';
    this.keyBack.style.display = this.keyOpen ? 'block' : 'none';
    this.keyBtn.classList.toggle('open', this.keyOpen);
  }

  /** The "?" button sits at the left end of the line above the figure band. */
  placeKeyBtn(x: number, y: number): void {
    this.keyBtn.style.left = `${x}px`;
    this.keyBtn.style.top = `${y}px`;
    this.keyBtn.style.bottom = '';
  }

  setCamera(c: Camera): void {
    CAM = c;
  }

  /** Re-anchor every on-plate label to the camera (cheap; called every frame the view moves). */
  reposition(): void {
    if (!CAM || CAM.version === this.camVersion) return;
    this.camVersion = CAM.version;
    this.root.querySelectorAll<HTMLElement>('[data-wx]').forEach((e) => apply(e));
    this.drawLeaders();
    const k = CAM.k;
    this.root.classList.toggle('zoom1', k > 1.6);
    this.root.classList.toggle('zoom2', k > 2.8);
  }

  private drawLeaders(): void {
    const line = (x1: number, y1: number, x2: number, y2: number) => {
      const [a, b] = CAM ? CAM.toScreen(x1, y1) : [x1, y1];
      const [c, d] = CAM ? CAM.toScreen(x2, y2) : [x2, y2];
      return [a, b, c, d];
    };
    this.leaders.innerHTML = this.leaderSegs
      .map((g, i) => {
        const [a, b, c, d] = line(...g);
        const t = this.leaderEnd[i] ?? 1;
        const ex = a + (c - a) * t;
        const ey = b + (d - b) * t;
        // a full leader ends in a dot on its subject; a cut one stops at the label it points to
        return `<path d="M${a.toFixed(1)},${b.toFixed(1)} L${ex.toFixed(1)},${ey.toFixed(1)}"/>${t >= 1 ? `<circle cx="${c.toFixed(1)}" cy="${d.toFixed(1)}" r="1.8"/>` : ''}`;
      })
      .join('');
  }

  layout(L: Layout): void {
    this.L = L;
    this.readout.style.right = `${L.notesW + 36}px`;
    this.notes.style.width = `${L.notesW}px`;
    this.notes.style.display = L.notesW ? 'block' : 'none';
    this.leaders.setAttribute('width', String(L.W));
    this.leaders.setAttribute('height', String(L.H));
    this.notesKey = '\u0000';
  }

  setHover(req: Req | null, x: number, y: number): void {
    this.hover = req ? { req, x, y } : null;
    this.renderCard();
  }

  update(now: number): void {
    const s = this.store;
    const st = s.stats(now);
    const cov = s.coverage(now);
    const h = s.hello;
    const engine = h?.engine ? `${engineName(h.engine)}${h.engine_version ? ` ${h.engine_version}` : ''}` : 'engine';
    const cfg = configLine(h?.config ?? {});
    const model = h?.model ? String(h.model).split('/').pop() : 'model unknown';
    const gpus = cov.gpu ? s.gpu!.gpus!.length : h?.gpus;
    const live = s.status === 'connected' || s.status === 'playing' || s.status === 'mock';
    const down = s.status.startsWith('disconnected') || (s.status === 'connected' && !s.engineFresh(now) && s.events > 0);
    const statusTxt = s.status.startsWith('disconnected') ? 'engine restarting — reconnecting quietly' : down ? 'connected · waiting for engine' : s.status;
    this.caption.innerHTML = `
      <div class="plate">Plate I · ${esc(h?.label ?? 'Live specimen')}${gpus ? ` · ${gpus} ${gpus > 1 ? 'GPUs' : 'GPU'}` : ''}</div>
      <h1 title="${esc(`${engine} · ${model}`)}">${esc(engine)} <em>·</em> ${esc(model)}</h1>
      <p>Requests in crimson, KV cache in ultramarine, compute in sap green.</p>
      ${cfg ? `<div class="cfg">${cfg}</div>` : ''}
      <div class="src"><span class="dot ${live ? 'on' : ''}"></span><b>${esc(h?.replay ? 'replay' : s.sourceLabel)}</b> · ${esc(h?.replay ? `recorded ${h.replay.recorded_utc ?? ''} UTC${h.replay.speed && h.replay.speed !== 1 ? ` · ${h.replay.speed}× speed` : ''}` : statusTxt)}</div>`;

    const m = (k: string, v: string | null, sub = '', unit = '') =>
      `<div class="m"><span class="k">${k}${v != null ? ` ${tag(k === 'ttft p50' || (k === 'tokens/s' && st.tokSource !== 'engine') || (k === 'running' && st.running == null) ? 'd' : 'm')}` : ''}</span><span class="v ${v == null ? 'un' : ''}">${v ?? '—'}${v != null && unit ? `<small>${unit}</small>` : ''}</span>${sub ? `<span class="sub">${sub}</span>` : ''}</div>`;
    const capTxt = st.cap == null ? 'no cap' : `cap ${st.cap}`;
    if (down) {
      this.readout.innerHTML = ['outside', 'running', 'waiting', 'ttft p50', 'tokens/s', 'kv cache']
        .map((k) => m(k, null, k === 'running' ? 'engine restarting' : ''))
        .join('');
    } else this.readout.innerHTML = [
      m('outside', fmtInt(st.proxyQueued), capTxt),
      m('running', st.running != null ? fmtInt(st.running) : fmtInt(st.inFlight), st.running != null ? 'in engine' : 'in flight (proxy)'),
      m('waiting', st.waiting != null ? fmtInt(st.waiting) : null, 'engine queue'),
      m('ttft p50', st.ttftP50 != null ? fmtMs(st.ttftP50) : null, st.ttftP90 != null ? `p90 ${fmtMs(st.ttftP90)}` : ''),
      m('tokens/s', st.tokS != null ? fmtInt(st.tokS) : null, st.tokS != null ? `from ${st.tokSource}` : ''),
      m('kv cache', st.kv != null ? pctTxt(st.kv) : null, st.prefixHit != null ? `prefix hit ${Math.round(st.prefixHit * 100)}%` : '', '%'),
    ].join('');

    if (this.L) {
      const room = this.L.PW - 36 - (this.readout.offsetWidth + 36) - 28;
      this.caption.style.maxWidth = `${Math.max(320, room)}px`;
    }

    const row = (on: boolean, color: string, name: string, what: string) =>
      `<div class="row ${on ? '' : 'off'}"><span class="sw" style="background:${color}"></span><span><b>${name}</b> — ${what}<span class="st ${on ? 'on' : ''}">${on ? 'live' : 'unstained'}</span></span></div>`;
    const legendHtml = [
      row(cov.requests, 'var(--crimson)', 'Crimson cells', 'requests (one per HTTP request)'),
      row(cov.kv, 'var(--ultra)', 'Ultramarine', 'KV blocks in GPU memory, to scale · solid = live · deep = shared prefix · hollow = cached, reusable'),
      row(cov.requests, 'var(--crimson)', 'Crimson veins', 'tokens streaming back; width = tokens/s'),
      row(cov.gpu, 'var(--sap)', 'Green neuron', cov.steps ? 'one GPU; fires each scheduler step' : 'one GPU; activity = utilisation'),
      `<div class="foot">${TAG_LEGEND}. Unstained parts are signals this engine does not report.</div>`,
    ].join('');
    const kl = this.key.querySelector<HTMLElement>('.klegend');
    if (kl && kl.dataset.v !== legendHtml) {
      kl.innerHTML = legendHtml;
      kl.dataset.v = legendHtml;
    }

    this.stages(st, now);

    // one label per KV body (prefill / decode workers, DP ranks)
    const bodies = this.scene.bodyAnchors();
    const multi = bodies.length > 1;
    this.labels[2].style.display = multi ? 'none' : 'block';
    const inst = new Map(s.instances(now).map((i) => [i.name, i]));
    while (this.bodyLabels.length < bodies.length) this.bodyLabels.push(el('div', 'lbl body', this.root));
    this.bodyLabels.forEach((lb, i) => {
      const b = bodies[i];
      if (!b || !multi) {
        lb.style.display = 'none';
        return;
      }
      lb.style.display = 'block';
      lb.classList.toggle('side', b.side);
      place(lb, b.x, b.y);
      const it = inst.get(b.name);
      const dp = /\.dp(\d+)$/.exec(b.name);
      const rep = /^replica-(\d+)$/.exec(b.name);
      const title = b.role === 'prefill' ? '③ prefill GPU' : b.role === 'decode' ? 'decode GPU' : rep ? `replica ${rep[1]}` : dp ? `DP rank ${dp[1]}` : b.name;
      const first = i === 0 && b.role !== 'prefill' ? '③ ' : '';
      const run = it?.running != null ? `${it.running} running` : '';
      const kvShort = it?.kv_usage != null ? `KV ${pctTxt(it.kv_usage)}% ${tag('m')}` : 'KV —';
      const stats = b.side ? [it?.running != null ? `${it.running} run` : '', kvShort] : [run, kvShort];
      lb.innerHTML = `${first}${esc(title)}${b.role === 'prefill' || b.role === 'decode' ? ` <i>${esc(b.name)}</i>` : ''}<br><span class="n">${stats.filter(Boolean).join(' · ')}</span>`;
    });
    if (this.scene.pd && bodies.length > 1) {
      const pre = bodies.filter((b) => b.role === 'prefill');
      const dec = bodies.filter((b) => b.role !== 'prefill');
      const at = this.scene.bridgeAnchor();
      this.bridge.style.display = at && pre.length && dec.length ? 'block' : 'none';
      this.bridge.innerHTML = `KV hand-off${transport(s) ? ` (${esc(transport(s))})` : ''}`;
      if (at) place(this.bridge, at.x, at.y);
    } else this.bridge.style.display = 'none';

    // expert routing is drawn by Fig. 4 in the figure band, not on the plate
    this.moeLbl.style.display = 'none';
    this.moeWorst.style.display = 'none';
    this.moeHot.forEach((h) => (h.style.display = 'none'));

    // GPU labels under each neuron
    const anchors = this.scene.organAnchors();
    while (this.gpuLabels.length < anchors.length) this.gpuLabels.push(el('div', 'gpu', this.root));
    this.gpuLabels.forEach((g, i) => {
      const p = anchors[i];
      if (!p) {
        g.style.display = 'none';
        return;
      }
      g.style.display = 'block';
      const many = anchors.length > 4;
      place(g, p.x, p.y + 22 + (many && i % 2 ? 14 : 0));
      const smp = cov.gpu ? s.gpu!.gpus![i] : undefined;
      g.className = `gpu ${smp ? '' : 'off'}`;
      g.innerHTML = smp ? (many ? `${smp.i} · ${smp.util ?? '—'}%` : `GPU ${smp.i}<br>${smp.util ?? '—'}% ${tag('m')}`) : `GPU<br>unstained`;
    });

    this.renderNotes();
    this.renderNetLabels();
    if (this.hover) this.renderCard();
  }

  /** Numbered stage captions with thin leader lines to the part of the plate they describe. */
  private stages(st: ReturnType<Store['stats']>, now: number): void {
    const L = this.L;
    if (!L) return;
    const A = this.scene.stageAnchors();
    const [s1, s2, s3, s4, s5] = this.labels;
    const top = L.memTop - 52;
    const cap = (no: string, main: string, sub: string) => `<span class="no">${no}</span>${main}<span class="sub">${sub}</span>`;
    const pre = countPhase(this.store, 'prefill');
    s1.innerHTML = cap('①', 'requests arrive', `proxy${st.proxyQueued ? ` · ${st.proxyQueued} held ${tag('m')}` : ''}`);
    s2.innerHTML = cap('②', 'admitted', `${this.store.admitRate.toFixed(1)}/s ${tag('d')}${pre ? `<br>${pre} before first token` : ''}`);
    s3.innerHTML = this.scene.memMap ? cap('③', 'KV cache · memory map', mapLine(this.store, this.scene)) : cap('③', 'KV cache', kvLine(this.store, this.scene, st.kv, undefined));
    const kb = this.scene.bodies[0];
    if (kb) {
      place(this.mapBtn, kb.cx - kb.rx * 0.76 - 8, kb.cy - kb.ry * 0.74); // right edge just outside the body's shoulder
      const has = !!this.scene.mapFor(0);
      this.mapBtn.style.display = has || this.scene.memMap ? 'block' : 'none';
      this.mapBtn.innerHTML = this.scene.memMap ? '◎ show layout' : '◫ memory map';
      this.mapBtn.classList.toggle('on', this.scene.memMap);
    }
    s4.innerHTML = cap('④', 'decode → tokens out', `width = tok/s ${tag('d')}`);
    s5.innerHTML = cap('⑤', 'responses', st.tokS != null ? `${fmtInt(st.tokS)} tok/s ${tag(st.tokSource === 'engine' ? 'm' : 'd')}` : '—');
    const b0 = this.scene.bodyAnchors()[0];
    const p1 = { x: L.memX - 92, y: top };
    const p2 = { x: L.memX + 96, y: top };
    // ③ sits fully above the body it names (its height changes with the map caption and font size)
    const p3 = this.scene.bodies.length <= 1
      ? { x: A.body.x, y: A.body.y - (s3.offsetHeight || 50) / (CAM?.k ?? 1) - 14 }
      : { x: A.body.x, y: (b0 ? b0.y : A.body.y - 40) - 18 - (this.scene.memMap ? 14 : 0) };
    const p4 = { x: lerp(L.net.x0, L.exit.x, 0.45), y: top };
    const p5 = { x: L.PW - 16, y: L.exit.y - 104 };
    // ①–④ are centred captions on one band: if one would touch the next, push the next one right
    {
      const k0 = CAM?.k ?? 1;
      const half = (e: HTMLElement) => (e.offsetWidth || 120) / k0 / 2;
      const row: [HTMLElement, { x: number; y: number }][] = [[s1, p1], [s2, p2], [s3, p3], [s4, p4]];
      for (let i = 1; i < row.length; i++) {
        const [ea, pa] = row[i - 1];
        const [eb, pb] = row[i];
        const vertical = Math.abs(pa.y - pb.y) < (ea.offsetHeight + eb.offsetHeight) / k0;
        const need = pa.x + half(ea) + 14 - (pb.x - half(eb));
        if (vertical && need > 0) pb.x += need;
      }
    }
    place(s1, p1.x, p1.y);
    place(s2, p2.x, p2.y);
    place(s3, p3.x, p3.y);
    place(s4, p4.x, p4.y);
    place(s5, p5.x, p5.y);
    s5.classList.add('right');
    const k = CAM?.k ?? 1;
    const h = (e: HTMLElement) => (e.offsetHeight || 30) / k; // label height in world units
    const single = this.scene.bodies.length <= 1;
    this.leaderSegs = [
      [p1.x, p1.y + h(s1) + 2, A.arrive.x, A.arrive.y],
      [p2.x, p2.y + h(s2) + 2, A.membrane.x, A.membrane.y],
      ...(single ? [[p3.x, p3.y + h(s3) + 2, A.body.x, A.body.y] as [number, number, number, number]] : []),
      [p4.x, p4.y + h(s4) + 2, A.vein.x, A.vein.y],
      [p5.x - 30, p5.y + h(s5) + 2, A.exit.x, A.exit.y],
    ];
    this.cutLeaders();
    this.drawLeaders();
    void now;
  }

  /** Stop each leader short of the first label it would run through (so it points at it, not across it). */
  private cutLeaders(): void {
    const own = new Set<Element>(this.labels);
    const o = this.root.getBoundingClientRect();
    const boxes: DOMRect[] = [];
    for (const e of this.root.querySelectorAll<HTMLElement>('.lbl, .mapbtn')) {
      if (own.has(e) || e.offsetParent === null) continue;
      const r = e.getBoundingClientRect();
      if (r.width > 2) boxes.push(new DOMRect(r.left - o.left - 6, r.top - o.top - 6, r.width + 12, r.height + 12));
    }
    this.leaderEnd = this.leaderSegs.map(([x1, y1, x2, y2]) => {
      const [a, b] = CAM ? CAM.toScreen(x1, y1) : [x1, y1];
      const [c, d] = CAM ? CAM.toScreen(x2, y2) : [x2, y2];
      const len = Math.hypot(c - a, d - b) || 1;
      for (let t = 0.15; t <= 1; t += 2 / len) {
        const x = a + (c - a) * t;
        const y = b + (d - b) * t;
        if (boxes.some((r) => x >= r.left && x <= r.right && y >= r.top && y <= r.bottom)) return Math.max(0.15, t - 3 / len);
      }
      return 1;
    });
  }

  private renderNetLabels(): void {
    const seen = new Set<string>();
    for (const n of this.scene.netLabels()) {
      seen.add(n.id);
      let e = this.netLabels.get(n.id);
      if (!e) {
        e = el('div', `lbl net ${n.kind}`, this.root);
        this.netLabels.set(n.id, e);
      }
      const rate = n.flux >= 0.5 ? `${fmtK(n.flux)} tok/s ${tag('d')}` : 'idle';
      const rep = /^replica-(\d+)$/.exec(n.name);
      const dp = /\.dp(\d+)$/.exec(n.name);
      const name = n.kind === 'trunk' ? (rep ? `replica ${rep[1]}` : dp ? `DP rank ${dp[1]}` : n.name) : n.name;
      e.innerHTML = `${esc(name)}<span class="n">${rate}</span>`;
      // several instances: per-class labels only appear once zoomed in (the instance labels carry the overview)
      e.classList.toggle('multi', n.multi && n.kind === 'branch');
      place(e, n.x, n.y - (n.kind === 'trunk' ? 22 : 18));
    }
    for (const [id, e] of this.netLabels) {
      if (!seen.has(id)) {
        e.remove();
        this.netLabels.delete(id);
      }
    }
    this.renderFamilyLabels();
    this.declutterNetLabels();
  }

  /** "shared prefix · 10.8k tok · reused by 9 running requests" under each large shared core. */
  private famLabels = new Map<string, HTMLElement>();
  private renderFamilyLabels(): void {
    const seen = new Set<string>();
    const ms = this.scene.mapSharedLabel();
    if (ms) {
      seen.add('map:shared');
      let e = this.famLabels.get('map:shared');
      if (!e) {
        e = el('div', 'lbl fam region', this.root);
        this.famLabels.set('map:shared', e);
      }
      // a cell counts if any of its tokens are shared, so this is a share of cells, not of tokens
      e.innerHTML = `shared prefix ▸<span class="n">${Math.round(ms.share * 100)}% of cells ${tag('m')}</span>`;
      place(e, ms.x, ms.y);
    }
    for (const f of this.scene.familyLabels()) {
      seen.add(f.id);
      let e = this.famLabels.get(f.id);
      if (!e) {
        e = el('div', 'lbl fam', this.root);
        this.famLabels.set(f.id, e);
      }
      e.innerHTML = `shared prefix<span class="n">${fmtK(f.tokens)} tok · reused by ${f.members} running requests ${tag('m')}</span>`;
      place(e, f.x, f.y);
    }
    for (const [id, e] of this.famLabels) {
      if (!seen.has(id)) {
        e.remove();
        this.famLabels.delete(id);
      }
    }
  }

  /**
   * Vein labels never touch: instance (trunk) labels keep their spot, class labels step down (then up)
   * to the nearest free slot. Stable order (kind, then id), so labels do not swap places between frames.
   */
  private declutterNetLabels(): void {
    const k = CAM?.k ?? 1;
    const o = this.root.getBoundingClientRect();
    type Box = { x0: number; y0: number; x1: number; y1: number };
    const hit = (a: Box, b: Box) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
    // measured, rendered boxes (labels anchor differently: centred, bottom-left, right-aligned)
    const rect = (e: HTMLElement): Box => {
      const r = e.getBoundingClientRect();
      return { x0: r.left - o.left - 6, y0: r.top - o.top - 3, x1: r.right - o.left + 6, y1: r.bottom - o.top + 3 };
    };
    const placed: Box[] = [];
    for (const e of [...this.labels, ...this.famLabels.values()]) if (e.offsetParent) placed.push(rect(e));
    const items = [...this.netLabels.entries()]
      .filter(([, e]) => e.offsetParent && e.offsetWidth)
      .sort(([a, ea], [b, eb]) => Number(eb.classList.contains('trunk')) - Number(ea.classList.contains('trunk')) || (a < b ? -1 : 1));
    for (const [, e] of items) {
      const base = rect(e);
      let dy = 0; // screen px
      if (!e.classList.contains('trunk')) {
        const step = base.y1 - base.y0 + 2;
        for (const d of [0, step, -step, 2 * step, -2 * step, 3 * step, -3 * step]) {
          dy = d;
          if (!placed.some((p) => hit(p, { ...base, y0: base.y0 + d, y1: base.y1 + d }))) break;
        }
        if (dy) place(e, Number(e.dataset.wx), Number(e.dataset.wy) + dy / k);
      }
      placed.push({ ...base, y0: base.y0 + dy, y1: base.y1 + dy });
    }
  }


  private renderNotes(): void {
    const key = String(this.store.notesVersion);
    if (key === this.notesKey) return;
    this.notesKey = key;
    const notes = [...this.store.notes.values()];
    this.notes.innerHTML = ledgerHtml(notes) + fieldNotesHtml(notes);
  }

  private renderCard(): void {
    const h = this.hover;
    if (!h) {
      this.card.style.display = 'none';
      return;
    }
    const r = h.req;
    const nowMs = Date.now();
    const end = r.tDone ?? nowMs;
    const queue = (r.tAdmit ?? (r.phase === 'queued' ? nowMs : r.tArrive)) - r.tArrive;
    const pre = r.tFirst != null ? r.tFirst - (r.tAdmit ?? r.tArrive) : r.tAdmit != null ? end - r.tAdmit : 0;
    const dec = r.tFirst != null ? end - r.tFirst : 0;
    const total = Math.max(1, queue + pre + dec);
    const pct = (x: number) => `${(100 * Math.max(0, x)) / total}%`;
    const eng = r.engine;
    const out = r.completionTokens ?? r.chunks;
    const rate = r.tFirst != null && dec > 0 ? (r.chunks / dec) * 1000 : null;
    const rows: [string, string][] = [
      ['outside (proxy)', r.partial ? '—' : fmtMs(queue)],
      ['to first token', r.tFirst != null ? fmtMs(r.tFirst - r.tArrive) : r.phase === 'failed' ? '—' : 'pending'],
      ['end to end', r.tDone != null ? fmtMs(r.tDone - r.tArrive) : 'in flight'],
      ['prompt tokens', r.promptTokens != null ? fmtInt(r.promptTokens) + (r.cachedTokens ? ` (${fmtInt(r.cachedTokens)} cached)` : '') : r.promptChars ? `≈${fmtInt(r.promptChars / 3.8)}` : '—'],
      ['output tokens', `${fmtInt(out)}${r.maxTokens ? ` / ${fmtInt(r.maxTokens)}` : ''}`],
      ['decode rate', rate != null ? `${rate.toFixed(1)} tok/s` : '—'],
    ];
    if (eng) {
      rows.push(['engine queue', eng.queued_s != null ? fmtMs(eng.queued_s * 1000) : '—']);
      rows.push(['engine prefill', eng.prefill_s != null ? fmtMs(eng.prefill_s * 1000) : '—']);
      if (eng.preemptions) rows.push(['preemptions', String(eng.preemptions)]);
    }
    const phase: Record<string, string> = {
      queued: 'Waiting outside the membrane (proxy admission).',
      prefill: 'Admitted; in the engine queue or prefilling.',
      decode: 'Decoding — tokens streaming out.',
      done: `Finished${r.finish ? ` (${r.finish})` : ''}.`,
      failed: r.error ?? 'Failed.',
    };
    this.card.innerHTML = `
      <span class="rid">${esc(displayRid(r.rid))}</span><span class="wl">${esc(r.workload ?? '')}</span>
      <div class="bar"><i style="width:${pct(queue)};background:#c9a7ae"></i><i style="width:${pct(pre)};background:var(--ultra)"></i><i style="width:${pct(dec)};background:var(--crimson)"></i></div>
      <table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${esc(v)}${v !== '—' && v !== 'pending' && v !== 'in flight' ? ` ${tag(/≈/.test(v) ? 'e' : CARD_SRC[k] ?? 'm')}` : ''}</td></tr>`).join('')}</table>
      <div class="ph">${esc(phase[r.phase])}</div>`;
    this.card.style.display = 'block';
    const W = window.innerWidth;
    const x = h.x + 18 + 268 > W - (this.L?.notesW ?? 0) ? h.x - 286 : h.x + 18;
    const y = Math.min(h.y + 12, window.innerHeight - this.card.offsetHeight - 12);
    this.card.style.left = `${x}px`;
    this.card.style.top = `${y}px`;
  }
}

// Engine settings worth reading at a glance (vLLM and SGLang names).
const CFG: [string, string, ((v: unknown) => string)?][] = [
  ['tensor_parallel_size', 'TP'], ['tp_size', 'TP'], ['data_parallel_size', 'DP'], ['dp_size', 'DP'],
  ['max_num_seqs', 'max seqs'], ['max_running_requests', 'max running'],
  ['max_num_batched_tokens', 'batch tokens'], ['chunked_prefill_size', 'chunked prefill'],
  ['block_size', 'block'], ['page_size', 'page'],
  ['enable_prefix_caching', 'prefix cache', (v) => (v ? 'on' : 'off')],
  ['schedule_policy', 'policy'],
  ['overlap_schedule', 'overlap', (v) => (v ? 'on' : 'off')],
  ['disable_overlap_schedule', 'overlap', (v) => (v ? 'off' : 'on')],
  ['kv_cache_dtype', 'kv dtype'], ['gpu_memory_utilization', 'mem'], ['mem_fraction_static', 'mem'],
];

function configLine(c: Record<string, unknown>): string {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const [k, label, fmt] of CFG) {
    const v = c[k];
    if (v === undefined || v === null || v === '' || seen.has(label)) continue;
    if ((label === 'TP' || label === 'DP') && Number(v) <= 1) continue;
    seen.add(label);
    out.push(`${label} <b>${esc(fmt ? fmt(v) : String(v))}</b>`);
    if (out.length >= 7) break;
  }
  return out.join('<span class="sep">·</span>');
}

const KEY_DOTS = (() => {
  // a small stippled KV patch for the schematic
  let out = '';
  for (let i = 0; i < 46; i++) {
    const a = (i * 2.399963) % (Math.PI * 2);
    const r = Math.sqrt((i + 0.5) / 46) * 13;
    out += `<circle cx="${(104 + Math.cos(a) * r * 1.2).toFixed(1)}" cy="${(30 + Math.sin(a) * r).toFixed(1)}" r="${i % 3 ? 0.9 : 1.2}" fill="#2a3ea8" opacity="${(0.55 + 0.4 * ((i * 37) % 10) / 10).toFixed(2)}"/>`;
  }
  return out;
})();

const KEY_HTML = `
  <div class="kh"><span>How to read this plate</span><button type="button" class="close" title="Hide">×</button></div>
  <svg viewBox="0 0 380 82" aria-hidden="true">
    <circle cx="18" cy="30" r="9" fill="rgba(204,88,116,.28)" stroke="#9e1432" stroke-width="1.7" stroke-dasharray="1.1 1.5"/>
    <path d="M30 30 h12" stroke="#57504a" stroke-width=".8" fill="none"/><path d="M40 27 l3 3 l-3 3" stroke="#57504a" stroke-width=".8" fill="none"/>
    <path d="M54 6 q5 12 0 24 q-5 12 0 24 q4 8 0 14" stroke="#803e4c" stroke-width="1.3" stroke-dasharray="1 2" fill="none"/>
    ${KEY_DOTS}
    <path d="M120 22 C150 14, 170 22, 200 30" stroke="#c95874" stroke-width="1" fill="none"/>
    <path d="M120 38 C150 46, 170 38, 200 30" stroke="#c95874" stroke-width="1" fill="none"/>
    <path d="M200 30 C235 30, 262 30, 296 30" stroke="#9e1432" stroke-width="3.2" fill="none" stroke-linecap="round"/>
    <circle cx="160" cy="19" r="1.4" fill="#9e1432"/><circle cx="232" cy="30" r="1.6" fill="#760a24"/><circle cx="268" cy="30" r="1.6" fill="#760a24"/>
    <path d="M306 16 q-12 14 0 28" stroke="#760a24" stroke-width="2.2" fill="none"/>
    <path d="M309 26 h52 M309 34 h52" stroke="#760a24" stroke-width=".9" stroke-dasharray="1.5 1.5"/>
    <g><text x="20" y="66">request</text><text x="88" y="66">KV cache</text><text x="160" y="66">class</text><text x="248" y="66">instance</text><text x="336" y="66">clients</text><text x="50" y="79" class="sm">↑ admission: the engine’s edge</text></g>
  </svg>
  <p>A request (crimson cell) waits at the proxy, crosses the membrane into the engine, and its prompt becomes KV cache (ultramarine). Generated tokens leave as crimson veins: <b>request → traffic class → engine instance → clients</b>. Green neurons below are the GPUs.</p>
  <p>Positions and curves are layout; sizes, counts and rates are measured or derived — every value carries a tag: ● measured, ◐ derived, ○ estimated.</p>
  <div class="legend klegend"></div>
  <p class="hint">Scroll to zoom · drag to pan · click anything to inspect · double-click to fly to it · space pauses · F follows a request · M toggles the KV memory map.</p>`;

const KEY_SEEN = 'io.board.keySeen';
function keyNotSeen(): boolean {
  try {
    return localStorage.getItem(KEY_SEEN) !== '1';
  } catch {
    return true;
  }
}
function markKeySeen(): void {
  try {
    localStorage.setItem(KEY_SEEN, '1');
  } catch {
    /* private mode: the key will simply open again next visit */
  }
}

/** "live 0.3% ● · cached ≈17% ○ · free 83% ○", then prefix hit and KV read when known. */
function kvLine(s: Store, scene: Scene, usage: number | null, name: string | undefined): string {
  const km = scene.mapFor(0);
  let l1: string;
  const sh = scene.kvShares(0);
  if (km && sh) {
    // live from the engine's KV usage; used and free from its allocator; cached = used − live
    l1 = `live ${pctTxt(sh.live)}% ${tag('m')} · cached ${pctTxt(sh.cached)}% ${tag('m')} · free ${pctTxt(sh.free)}% ${tag('m')}`;
  } else {
    if (usage == null) return 'occupancy not reported';
    const cap = s.kvCapacity(name);
    const cached = cap.known ? Math.min(1 - usage, scene.cachedBlocks(0) / cap.blocks) : 0;
    l1 = `live ${pctTxt(usage)}% ${tag('m')} · cached ≈${pctTxt(cached)}% ${tag('e')} · free ${pctTxt(Math.max(0, 1 - usage - cached))}% ${tag('e')}`;
  }
  const p = s.prefixLive();
  const pre = p.tokenRate != null ? `prefix hit ${Math.round(p.tokenRate * 100)}% ${tag('d')}` : s.engine?.prefix_hit_rate != null ? `prefix hit ${Math.round(s.engine.prefix_hit_rate * 100)}% ${tag('m')}` : '';
  const read = scene.kvRead > 1 ? `read ${fmtK(scene.kvRead)} tok/s ${tag('d')}` : '';
  const l2 = [pre, read].filter(Boolean).join(' · ');
  return l2 ? `${l1}<br>${l2}` : l1;
}

const CARD_SRC: Record<string, 'm' | 'd' | 'e'> = { 'decode rate': 'd' };

/** KV bytes per token for models we know (2 × layers × KV heads × head dim × 2 bytes, bf16). */
export function kvBytesPerToken(model: string | null | undefined): number | null {
  const m = String(model ?? '');
  if (/Qwen3-30B-A3B/i.test(m)) return 2 * 48 * 4 * 128 * 2;
  if (/Qwen3-235B-A22B/i.test(m)) return 2 * 94 * 4 * 128 * 2;
  if (/Qwen3-8B/i.test(m)) return 2 * 36 * 8 * 128 * 2;
  return null;
}

/** Memory-map caption: grid, what one cell holds, and measured shares from the map itself. */
function mapLine(s: Store, scene: Scene): string {
  const m = scene.mapFor(0);
  if (!m) return `not reported yet — waiting for kv.map`;
  let own = 0;
  let reused = 0;
  let shared = 0;
  let coloc = 0;
  let cached = 0;
  for (let i = 0; i < m.bins; i++) {
    const v = m.state[i];
    if (v >= 0) m.prefix[i] ? reused++ : own++;
    else if (v === -3) shared++;
    else if (v === -4) coloc++;
    else if (v === -2) cached++;
  }
  const side = Math.sqrt(m.bins);
  const grid = Number.isInteger(side) ? `${side}×${side}` : `${fmtInt(m.bins)} cells`;
  const bpt = m.bytesPerToken ?? kvBytesPerToken(s.hello?.model);
  const mib = bpt ? (m.binTokens * bpt) / 2 ** 20 : 0;
  const mb = bpt ? ` ≈ ${mib >= 10 ? mib.toFixed(0) : mib.toFixed(1)} MiB` : '';
  const p = (n: number) => `${Math.round((100 * n) / m.bins)}%`;
  const free = m.bins - own - reused - shared - coloc - cached;
  // three short lines, so the caption stays clear of ② and ④
  return `${grid} · cell = ${fmtInt(m.binTokens)} tok${mb} ${tag('d')}<br>cells: own ${p(own)} · reused ${p(reused)} · shared ${p(shared)} ${tag('m')}<br>co-located ${p(coloc)} · cached ${p(cached)} · free ${p(free)} ${tag('m')}`;
}

function fmtK(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}k` : String(Math.round(n));
}

function transport(s: Store): string {
  const c = String(s.hello?.config?.kv_connector ?? s.hello?.config?.kv_transfer ?? '').toLowerCase();
  if (c.includes('nixl')) return 'NIXL';
  if (c.includes('mooncake')) return 'Mooncake';
  if (c.includes('lmcache')) return 'LMCache';
  if (c) return String(s.hello?.config?.kv_connector);
  return (s.hello?.engine ?? '').toLowerCase() === 'vllm' ? 'NIXL' : '';
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function countPhase(s: Store, p: Req['phase']): number {
  let n = 0;
  for (const r of s.reqs.values()) if (r.phase === p) n++;
  return n;
}

/** Anchor an on-plate label at a world point; it follows the camera. */
function place(e: HTMLElement, x: number, y: number): void {
  e.dataset.wx = String(x);
  e.dataset.wy = String(y);
  apply(e);
}

function apply(e: HTMLElement): void {
  const x = Number(e.dataset.wx);
  const y = Number(e.dataset.wy);
  const [sx, sy] = CAM ? CAM.toScreen(x, y) : [x, y];
  e.style.left = `${Math.round(sx)}px`;
  e.style.top = `${Math.round(sy)}px`;
}
