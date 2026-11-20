// Fig. 3 — the scheduler's heartbeat: one column per engine step.
// Above the midline: prefill (or extend) tokens in that step, ultramarine, one segment per request.
// Below the midline: requests decoding in that step, crimson, one unit per request.
// Continuous batching, chunked prefill and batch size are visible directly.
import type { Scene } from '../scene';
import { displayRid, type Batch, type Store } from '../state';
import { fmtInt } from '../util';
import { draggable } from './draggable';
import { tag } from './tags';

const esc = (s: unknown) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);


export class BatchStrip {
  private el: HTMLElement;
  private cv: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private info: HTMLElement;
  private tip: HTMLElement;
  private modeBtn: HTMLButtonElement;
  private windowMs = 10_000;
  private last = 0;
  private cols: { x0: number; x1: number; b: Batch }[] = [];
  private W = 380;
  private H = 104;
  private static G = 13; // label gutter above and below the bars
  private get MID(): number {
    return Math.round(BatchStrip.G + (this.H - 2 * BatchStrip.G) * 0.48);
  }
  /** Room for bars above (prefill) and below (decode) the midline. */
  private get UP(): number {
    return this.MID - BatchStrip.G - 2;
  }
  private get DOWN(): number {
    return this.H - BatchStrip.G - this.MID - 2;
  }

  constructor(root: HTMLElement, private store: Store, private scene: Scene) {
    this.el = document.createElement('figure');
    this.el.className = 'bandfig fig3';
    const cap = document.createElement('figcaption');
    cap.innerHTML = `<span class="no">Fig. 3</span> scheduler steps ${tag('m')}`;
    this.modeBtn = document.createElement('button');
    this.modeBtn.type = 'button';
    this.modeBtn.addEventListener('click', () => {
      this.windowMs = this.windowMs === 10_000 ? 1_000 : 10_000;
      this.last = 0;
    });
    cap.appendChild(this.modeBtn);
    this.cv = document.createElement('canvas');
    this.ctx = this.cv.getContext('2d')!;
    this.resize(this.W, this.H);
    this.info = document.createElement('div');
    this.info.className = 'f3info';
    this.tip = document.createElement('div');
    this.tip.className = 'f3tip';
    this.tip.style.display = 'none';
    this.el.append(cap, this.cv, this.info, this.tip);
    root.appendChild(this.el);
    draggable(this.el, cap);
    this.cv.addEventListener('mousemove', (e) => this.hover(e));
    this.cv.addEventListener('mouseleave', () => (this.tip.style.display = 'none'));
    this.cv.addEventListener('click', (e) => {
      const c = this.colAt(e);
      if (c) this.scene.highlightStep(c.b.reqs.map((r) => r[0]));
    });
  }

  /** Position and size the figure (CSS px, relative to the overlay). */
  setRect(x: number, y: number, w: number, h: number, visible = true): void {
    this.el.style.display = visible ? 'block' : 'none';
    if (!visible) return;
    if (!this.el.dataset.moved) {
      this.el.style.left = `${x}px`;
      this.el.style.top = `${y}px`;
    }
    this.el.style.width = `${w}px`;
    this.el.style.height = `${h}px`;
    this.resize(Math.max(200, Math.floor(w - 20)), Math.max(70, Math.floor(h - 58)));
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

  private colAt(e: MouseEvent): { x0: number; x1: number; b: Batch } | null {
    const r = this.cv.getBoundingClientRect();
    const x = e.clientX - r.left;
    let best: { x0: number; x1: number; b: Batch } | null = null;
    let bd = Infinity;
    for (const c of this.cols) {
      const d = x < c.x0 ? c.x0 - x : x > c.x1 ? x - c.x1 : 0;
      if (d < bd) {
        bd = d;
        best = c;
      }
    }
    return bd <= 3 ? best : null;
  }

  private hover(e: MouseEvent): void {
    const c = this.colAt(e);
    if (!c) {
      this.tip.style.display = 'none';
      return;
    }
    const b = c.b;
    const ago = (performance.now() - b.t) / 1000;
    const list = b.reqs
      .slice(0, 12)
      .map(([rid, ph, tok]) => `<tr><td>${esc(displayRid(rid))}</td><td class="${ph === 'p' ? 'p' : 'd'}">${ph === 'p' ? 'prefill' : 'decode'}</td><td>${fmtInt(tok)}</td></tr>`)
      .join('');
    this.tip.innerHTML = `<b>step ${fmtInt(b.seq)}</b> · ${ago.toFixed(2)} s ago${b.dur != null ? ` · ${b.dur.toFixed(1)} ms ${tag('m')}` : ''}<br>
      ${b.pReq} prefill (${fmtInt(b.pTok)} tok) · ${b.dReq} decoding${b.truncated ? ' · list truncated' : ''}
      <table>${list}</table>${b.reqs.length > 12 ? `<i>+${b.reqs.length - 12} more · click to mark them all on the plate</i>` : '<i>click to mark them on the plate</i>'}`;
    this.tip.style.display = 'block';
    const r = this.cv.getBoundingClientRect();
    this.tip.style.left = `${Math.min(this.W - 230, Math.max(0, e.clientX - r.left - 110))}px`;
  }

  update(now: number): void {
    const detail = this.windowMs <= 1000;
    if (now - this.last < (detail ? 100 : 66)) return;
    this.last = now;
    this.modeBtn.textContent = detail ? '1 s · each step' : '10 s · overview';
    const all = this.store.batches;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.W, this.H);
    this.cols = [];
    const live = now - this.store.batchAt < 3000;
    if (!all.length || !live) {
      ctx.fillStyle = 'rgba(42,38,34,0.55)';
      ctx.font = 'italic 12px Newsreader, Georgia, serif';
      const msg = !['connected', 'playing', 'mock'].includes(this.store.status)
        ? 'waiting for the engine…'
        : all.length
          ? 'no steps in the last 3 s'
          : 'unstained: this engine does not report per-step batches';
      ctx.fillText(msg, 8, this.MID + 4);
      this.info.textContent = '';
      return;
    }
    const t0 = now - this.windowMs;
    const shown = all.filter((b) => b.t >= t0);
    const maxP = Math.max(512, ...shown.map((b) => b.pTok));
    const maxD = Math.max(8, ...shown.map((b) => b.dReq));
    const xOf = (t: number) => ((t - t0) / this.windowMs) * this.W;
    // midline
    ctx.fillStyle = 'rgba(42,38,34,0.45)';
    ctx.fillRect(0, this.MID, this.W, 1);
    for (let i = 0; i < shown.length; i++) {
      const b = shown[i];
      const x0 = xOf(b.t);
      const x1 = detail ? Math.max(x0 + 1, i + 1 < shown.length ? xOf(shown[i + 1].t) - 0.5 : x0 + 2) : x0 + 1;
      const w = Math.max(1, x1 - x0);
      this.cols.push({ x0, x1: x0 + w, b });
      const hp = (b.pTok / maxP) * this.UP;
      const hd = (b.dReq / maxD) * this.DOWN;
      if (detail) {
        // stacked by request
        let y = this.MID;
        for (const [, ph, tok] of b.reqs) {
          if (ph !== 'p') continue;
          const h = (tok / maxP) * this.UP;
          ctx.fillStyle = 'rgba(28,44,150,0.92)';
          ctx.fillRect(x0, y - h, w, Math.max(0.6, h - 0.4));
          y -= h;
        }
        const u = this.DOWN / maxD;
        let yd = this.MID + 1;
        for (const [, ph] of b.reqs) {
          if (ph === 'p') continue;
          ctx.fillStyle = 'rgba(146,14,44,0.9)';
          ctx.fillRect(x0, yd, w, Math.max(0.5, u - (u > 2.5 ? 0.5 : 0)));
          yd += u;
        }
      } else {
        ctx.fillStyle = 'rgba(28,44,150,0.85)';
        ctx.fillRect(x0, this.MID - hp, w, hp);
        ctx.fillStyle = 'rgba(146,14,44,0.8)';
        ctx.fillRect(x0, this.MID + 1, w, hd);
      }
    }
    // axis notes
    ctx.fillStyle = '#4a443c';
    ctx.font = '500 10px "IBM Plex Mono", monospace';
    ctx.fillText(`prefill ≤ ${fmtInt(maxP)} tok`, 4, 10);
    ctx.fillText(`decoding ≤ ${maxD} req`, 4, this.H - 3);
    ctx.textAlign = 'right';
    ctx.fillText(detail ? '−1 s' : '−10 s', this.W - 2, 10);
    ctx.textAlign = 'left';
    const sps = shown.length / (this.windowMs / 1000);
    const avgD = shown.reduce((a, b) => a + b.dReq, 0) / shown.length;
    const durs = shown.map((b) => b.dur).filter((x): x is number => x != null);
    const avgDur = durs.length ? durs.reduce((a, b) => a + b, 0) / durs.length : null;
    const withP = Math.round((100 * shown.filter((b) => b.pReq).length) / shown.length);
    this.info.innerHTML = `${fmtInt(sps)} steps/s${avgDur != null ? ` · ${avgDur.toFixed(1)} ms/step` : ''} ${tag('d')} · batch ≈${avgD.toFixed(0)} · prefill in ${withP}%`;
  }
}
