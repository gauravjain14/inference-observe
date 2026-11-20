// Fig. 4 (dense models): the prefix cache, request by request.
// One column per request that has produced its first token, oldest left, newest right.
//   up   : prompt tokens, split into tokens reused from the prefix cache (ultramarine: memory) and
//          tokens computed by prefill (sap green: compute), to one linear scale
//   down : time to first token after admission (crimson, log scale) — the price of what was computed
// The footer compares requests that were mostly served from cache with ones computed from scratch.
// Same definitions as the follow narration: cached = engine-reported cached tokens, TTFT = first token − admission.
import { displayRid, type Req, type Store } from '../state';
import { fmtInt } from '../util';
import { draggable } from './draggable';
import { tag } from './tags';

const esc = (s: unknown) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
const ULTRA = 'rgba(16,28,128,0.95)';
const SAP = 'rgba(59,85,32,0.92)';
const CRIMSON = 'rgba(146,14,44,0.88)';
const MUTED = '#4a443c';
const T_MIN = 10; // ms, bottom of the log axis
const T_MAX = 10_000;

const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10_000 ? 0 : 1)} s` : `${Math.round(ms)} ms`);
const fmtTok = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}k` : String(Math.round(n)));
const median = (xs: number[]) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

interface Col {
  x0: number;
  x1: number;
  r: Req;
  prompt: number;
  cached: number;
  ttft: number;
}

export class Fig5 {
  private el: HTMLElement;
  private cv: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private foot: HTMLElement;
  private tip: HTMLElement;
  private W = 400;
  private H = 120;
  private last = 0;
  private cols: Col[] = [];
  // requests that reached their first token; kept here after the store lets finished requests go
  private hist = new Map<string, Col>();

  constructor(root: HTMLElement, private store: Store, onPick: (rid: string) => void) {
    this.el = document.createElement('figure');
    this.el.className = 'bandfig fig5';
    const cap = document.createElement('figcaption');
    cap.innerHTML = `<span class="no">Fig. 4</span> prefix cache <span class="sub">· per request, newest right ${tag('m')}</span>`;
    this.cv = document.createElement('canvas');
    this.ctx = this.cv.getContext('2d')!;
    this.foot = document.createElement('div');
    this.foot.className = 'f5foot';
    this.tip = document.createElement('div');
    this.tip.className = 'f4tip';
    this.el.append(cap, this.cv, this.foot, this.tip);
    root.appendChild(this.el);
    draggable(this.el, cap);
    this.resize(this.W, this.H);
    this.cv.addEventListener('mousemove', (e) => this.hover(e));
    this.cv.addEventListener('mouseleave', () => (this.tip.style.display = 'none'));
    this.cv.addEventListener('click', (e) => {
      const c = this.colAt(e);
      if (c) onPick(c.r.rid);
    });
  }

  setRect(x: number, y: number, w: number, h: number, visible = true): void {
    this.el.style.display = visible ? 'block' : 'none';
    if (!visible) return;
    if (!this.el.dataset.moved) {
      this.el.style.left = `${x}px`;
      this.el.style.top = `${y}px`;
    }
    this.el.style.width = `${w}px`;
    this.el.style.height = `${h}px`;
    this.resize(Math.max(220, Math.floor(w - 20)), Math.max(80, Math.floor(h - 76)));
  }

  private resize(w: number, h: number): void {
    if (w === this.W && h === this.H && this.cv.width) return;
    this.W = w;
    this.H = h;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.cv.width = Math.round(w * dpr);
    this.cv.height = Math.round(h * dpr);
    this.cv.style.width = `${w}px`;
    this.cv.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.last = 0;
  }

  private colAt(e: MouseEvent): Col | null {
    const r = this.cv.getBoundingClientRect();
    const x = e.clientX - r.left;
    return this.cols.find((c) => x >= c.x0 - 1 && x <= c.x1 + 1) ?? null;
  }

  private hover(e: MouseEvent): void {
    const c = this.colAt(e);
    if (!c) {
      this.tip.style.display = 'none';
      return;
    }
    const computed = c.prompt - c.cached;
    this.tip.innerHTML = `<b>${esc(displayRid(c.r.rid))}</b> <i>${esc(c.r.workload ?? '')}</i><br>${fmtInt(c.prompt)} prompt tokens: <b>${fmtInt(c.cached)}</b> reused from cache, <b>${fmtInt(computed)}</b> computed<br>first token after <b>${fmtMs(c.ttft)}</b> · click to follow`;
    this.tip.style.display = 'block';
    const box = this.el.getBoundingClientRect();
    const cr = this.cv.getBoundingClientRect();
    this.tip.style.left = `${Math.min(box.width - 260, Math.max(4, e.clientX - box.left - 120))}px`;
    this.tip.style.top = `${cr.top - box.top + 6}px`;
  }

  update(now: number): void {
    if (this.el.style.display === 'none' || now - this.last < 250) return;
    this.last = now;
    const ctx = this.ctx;
    const { W, H } = this;
    ctx.clearRect(0, 0, W, H);
    for (const r of this.store.reqs.values()) {
      if (r.tFirst == null || r.promptTokens == null || r.partial) continue;
      const ttft = r.tFirst - (r.tAdmit ?? r.tArrive);
      if (!(ttft >= 0)) continue;
      this.hist.set(r.rid, { x0: 0, x1: 0, r, prompt: r.promptTokens, cached: Math.min(r.promptTokens, r.cachedTokens ?? 0), ttft });
    }
    while (this.hist.size > 300) this.hist.delete(this.hist.keys().next().value!);
    const all = [...this.hist.values()].sort((a, b) => a.r.tArrive - b.r.tArrive);
    if (!all.length) {
      ctx.fillStyle = MUTED;
      ctx.font = 'italic 13px Newsreader, Georgia, serif';
      ctx.fillText(['connected', 'playing', 'mock'].includes(this.store.status) ? 'waiting for requests to reach their first token…' : 'waiting for the engine…', 8, H / 2);
      this.foot.innerHTML = '';
      this.cols = [];
      return;
    }
    const G = 13; // label gutters
    const left = 2;
    const right = 44; // gutter for the TTFT tick labels
    const pitch = 7;
    const n = Math.max(8, Math.floor((W - left - right) / pitch));
    const shown = all.slice(-n);
    const mid = Math.round(G + (H - 2 * G) * 0.6);
    const up = mid - G - 2;
    const down = H - G - mid - 2;
    const maxP = Math.max(1000, ...shown.map((c) => c.prompt));
    const yT = (ms: number) => (Math.log10(Math.min(T_MAX, Math.max(T_MIN, ms)) / T_MIN) / Math.log10(T_MAX / T_MIN)) * down;
    // midline and TTFT gridlines (100 ms, 1 s)
    ctx.fillStyle = 'rgba(42,38,34,0.45)';
    ctx.fillRect(0, mid, W - right + 4, 1);
    ctx.font = '500 10px "IBM Plex Mono", monospace';
    for (const t of [100, 1000]) {
      const y = mid + 2 + yT(t);
      ctx.fillStyle = 'rgba(42,38,34,0.14)';
      ctx.fillRect(0, Math.round(y), W - right + 4, 1);
      ctx.fillStyle = MUTED;
      ctx.textAlign = 'left';
      ctx.fillText(fmtMs(t), W - right + 8, y + 4);
    }
    ctx.textAlign = 'left';
    this.cols = [];
    const x0 = W - right - shown.length * pitch;
    shown.forEach((c, i) => {
      const x = x0 + i * pitch;
      const w = pitch - 2;
      c.x0 = x;
      c.x1 = x + w;
      const hc = (c.cached / maxP) * up;
      const hp = ((c.prompt - c.cached) / maxP) * up;
      ctx.fillStyle = ULTRA;
      ctx.fillRect(x, mid - hc, w, hc);
      ctx.fillStyle = SAP;
      ctx.fillRect(x, mid - hc - hp, w, Math.max(hp, c.prompt > c.cached ? 1 : 0));
      ctx.fillStyle = CRIMSON;
      ctx.fillRect(x, mid + 2, w, Math.max(1.5, yT(c.ttft)));
      this.cols.push(c);
    });
    // axis notes, in the gutters
    ctx.fillStyle = MUTED;
    ctx.fillText(`prompt ≤ ${fmtTok(maxP)} tok`, 4, 10);
    ctx.fillText('time to first token (log)', 4, H - 3);
    // legend, top right
    const items: [string, string][] = [['reused from cache', ULTRA], ['computed', SAP]];
    ctx.textAlign = 'right';
    let lx = W - 2;
    for (const [label, ink] of items.reverse()) {
      ctx.fillStyle = MUTED;
      ctx.fillText(label, lx, 10);
      lx -= ctx.measureText(label).width + 5;
      ctx.fillStyle = ink;
      ctx.fillRect(lx - 9, 2, 9, 9);
      lx -= 18;
    }
    ctx.textAlign = 'left';
    // footer: mostly cached vs computed from scratch, over the requests in the figure
    const warm = all.filter((c) => c.prompt >= 500 && c.cached / c.prompt >= 0.8).map((c) => c.ttft);
    const cold = all.filter((c) => c.prompt >= 500 && c.cached / c.prompt < 0.2).map((c) => c.ttft);
    const mw = median(warm);
    const mc = median(cold);
    const reuse = all.reduce((a, c) => a + c.cached, 0) / Math.max(1, all.reduce((a, c) => a + c.prompt, 0));
    this.foot.innerHTML =
      (mw != null && mc != null
        ? `<div>First token <b>${fmtMs(mw)}</b> cached vs <b>${fmtMs(mc)}</b> from scratch${mc > mw * 1.5 ? ` · <b>${(mc / mw).toFixed(0)}×</b> faster` : ''} ${tag('d')}`
        : `<div>${Math.round(reuse * 100)}% of prompt tokens reused from cache ${tag('d')}`) +
      `</div><div class="scale">medians · cached ≥80% reused (n=${warm.length}) · scratch &lt;20% (n=${cold.length}) · ${Math.round(reuse * 100)}% of tokens reused</div>`;
  }
}
