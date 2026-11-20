// Fig. 4: expert routing for MoE models.
//
// Top: per layer, how much busier the busiest expert is than the average expert (stippled columns,
// log scale, 1× = perfectly balanced). This is where routing skew actually lives: summed over all
// layers the load looks even, inside a single layer one expert can take 10× its share.
// Below, only when the live layer×expert grid is present (moe.tick.grid_b64): a heatmap whose
// columns line up with the layer columns above (rows = experts), so an overloaded expert is a dark
// cell under its layer's tall column. Without the grid (recordings, older engines) the stipple
// takes the full height and the footer names the busiest experts overall.
import type { Store } from '../state';
import { draggable } from './draggable';

const esc = (s: unknown) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

const INK = '#2a2622';
const MUTED = '#57504a';
const SAP = '#3b5520';
const CRIMSON = '#9e1432';
const LIGHT: [number, number, number] = [236, 231, 216]; // ~fair share or less
const DARK: [number, number, number] = [34, 52, 16]; // hottest
const Y_MAX = 16; // top of the per-layer scale (busiest ÷ average)

interface MoeLike {
  layers?: number;
  experts?: number;
  load?: number[];
  layer_imbalance?: (number | null)[];
  hot?: number[];
  top_k?: number | null;
  grid_b64?: string;
  grid_scale?: number;
  window_s?: number;
}

type Hit = { kind: 'layer'; l: number; v: number } | { kind: 'cell'; l: number; e: number; rel: number };

function ramp(t: number): string {
  const k = Math.max(0, Math.min(1, t));
  const c = LIGHT.map((a, i) => Math.round(a + (DARK[i] - a) * k));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export class Fig4 {
  private el: HTMLElement;
  private cap: HTMLElement;
  private cv: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private tip: HTMLElement;
  private foot: HTMLElement;
  private w = 400;
  private h = 160;
  private last = 0;
  private sig = '';
  private hit: Hit | null = null;
  private mouse: { x: number; y: number } | null = null;
  private grid: { L: number; E: number; rel: Float32Array } | null = null;
  // geometry of the last draw, for hit-testing
  private g = { x0: 0, x1: 0, colW: 0, aTop: 0, aBot: 0, bTop: 0, bBot: 0, rows: 1, perRow: 128, cellW: 0, cellH: 0 };

  constructor(root: HTMLElement, private store: Store, private modelName: () => string) {
    this.el = document.createElement('figure');
    this.el.className = 'bandfig fig4';
    this.cap = document.createElement('figcaption');
    this.cv = document.createElement('canvas');
    this.ctx = this.cv.getContext('2d')!;
    this.tip = document.createElement('div');
    this.tip.className = 'f4tip';
    this.foot = document.createElement('div');
    this.foot.className = 'f4foot';
    this.el.append(this.cap, this.cv, this.foot, this.tip);
    root.appendChild(this.el);
    draggable(this.el, this.cap);
    this.cv.addEventListener('mousemove', (e) => {
      const r = this.cv.getBoundingClientRect();
      this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
      this.hit = this.hitTest(this.mouse.x, this.mouse.y);
      this.renderTip();
      this.draw();
    });
    this.cv.addEventListener('mouseleave', () => {
      this.mouse = null;
      this.hit = null;
      this.tip.style.display = 'none';
      this.draw();
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
    const cw = Math.max(200, Math.floor(w - 20));
    const ch = Math.max(90, Math.floor(h - 76)); // caption above, two footer lines below
    if (cw !== this.w || ch !== this.h) {
      this.w = cw;
      this.h = ch;
      const dpr = window.devicePixelRatio || 1;
      this.cv.width = Math.round(cw * dpr);
      this.cv.height = Math.round(ch * dpr);
      this.cv.style.width = `${cw}px`;
      this.cv.style.height = `${ch}px`;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.sig = '';
    }
  }

  private moe(now: number): MoeLike | null {
    const m = this.store.moe as unknown as MoeLike | null;
    if (!m || !m.load?.length) return null;
    return now - this.store.moeAt < 15_000 ? m : null;
  }

  update(now: number): void {
    if (now - this.last < 400) return;
    this.last = now;
    const m = this.moe(now);
    const sig = m ? `${this.store.moeAt}|${this.w}x${this.h}` : `none|${this.w}x${this.h}`;
    if (sig === this.sig) return;
    this.sig = sig;
    this.grid = null;
    if (m?.grid_b64 && m.layers && m.experts) {
      try {
        const bin = atob(m.grid_b64);
        const scale = m.grid_scale || 32;
        const rel = new Float32Array(bin.length);
        for (let i = 0; i < bin.length; i++) rel[i] = bin.charCodeAt(i) / scale;
        if (rel.length === m.layers * m.experts) this.grid = { L: m.layers, E: m.experts, rel };
      } catch {
        this.grid = null;
      }
    }
    this.renderCaption(m);
    this.draw();
  }

  private renderCaption(m: MoeLike | null): void {
    const E = m?.experts ?? m?.load?.length ?? 0;
    const k = m?.top_k ?? (/A3B|A22B|A35B/i.test(this.modelName()) ? 8 : null);
    this.cap.innerHTML = `<span class="no">Fig. 4</span> expert routing${E ? ` <span class="sub">· ${E} experts${k ? ` · top-${k}` : ''}</span>` : ''}`;
    if (!m) {
      this.foot.innerHTML = '';
      return;
    }
    const lay = (m.layer_imbalance ?? []).map((v) => v ?? 0);
    const worst = lay.length ? lay.reduce((b, v, i) => (v > lay[b] ? i : b), 0) : -1;
    const sum = m.load!.reduce((a, b) => a + b, 0) || 1;
    const hot = (m.hot ?? []).slice(0, 3).map((e) => `#${e} ${((m.load![e] / sum) * E).toFixed(1)}×`).join(' · ');
    this.foot.innerHTML =
      (worst >= 0 ? `<div>Most skewed: <b>layer ${worst}</b>, busiest expert at <b>${lay[worst].toFixed(1)}×</b> the average.</div>` : '') +
      (this.grid
        ? `<div class="scale">cells: <i style="background:${ramp(0)}"></i>≤ fair share <i style="background:${ramp(0.5)}"></i><i style="background:${ramp(1)}"></i>≥ 8× its share of the layer</div>`
        : hot
          ? `<div class="scale">hottest overall ${esc(hot)}</div>`
          : '');
  }

  private draw(): void {
    const c = this.ctx;
    const { w, h } = this;
    c.clearRect(0, 0, w, h);
    const m = this.moe(performance.now());
    if (!m) {
      c.fillStyle = MUTED;
      c.font = 'italic 13px Newsreader, Georgia, serif';
      // one message, in the chart; the footer stays empty
      c.fillText(['connected', 'playing', 'mock'].includes(this.store.status) ? 'No expert routing data: a dense model, or the engine does not report it.' : 'waiting for the engine…', 8, h / 2);
      return;
    }
    const lay = (m.layer_imbalance ?? []).map((v) => v ?? 1);
    const L = lay.length || m.layers || 48;
    const E = m.experts ?? m.load!.length;
    const x0 = 34; // room for the y labels
    const x1 = w - 78; // room for the "1× balanced" label
    const colW = (x1 - x0) / L;
    const aTop = 14;
    const aBot = this.grid ? Math.round(h * 0.42) : h - 18; // full height unless the heatmap is shown
    const bTop = aBot + 34; // below the layer numbers and the section label
    const bBot = h - 2;
    const yOf = (v: number) => aBot - (Math.log2(Math.max(1, Math.min(Y_MAX, v))) / Math.log2(Y_MAX)) * (aBot - aTop);
    const worst = lay.reduce((b, v, i) => (v > lay[b] ? i : b), 0);
    this.g = { ...this.g, x0, x1, colW, aTop, aBot, bTop, bBot };

    // --- section A: per-layer stippled columns -------------------------------------------------------
    c.font = '500 10px "IBM Plex Mono", ui-monospace, monospace';
    c.fillStyle = MUTED;
    c.textAlign = 'right';
    c.textBaseline = 'middle';
    for (const v of [1, 2, 4, 8, 16]) {
      const y = yOf(v);
      c.fillText(`${v}×`, x0 - 6, y);
      c.strokeStyle = v === 1 ? 'rgba(42,38,34,0.75)' : 'rgba(42,38,34,0.12)';
      c.setLineDash(v === 1 ? [4, 3] : []);
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(x0, y + 0.5);
      c.lineTo(x1, y + 0.5);
      c.stroke();
    }
    c.setLineDash([]);
    c.textAlign = 'left';
    c.fillStyle = INK;
    c.fillText('1× balanced', x1 + 6, yOf(1));
    c.fillStyle = MUTED;
    c.textBaseline = 'bottom';
    c.fillText('busiest', x1 + 6, yOf(16) + 4);
    c.textBaseline = 'top';
    c.fillText('÷ average', x1 + 6, yOf(16) + 5);
    c.textBaseline = 'middle';
    const pitch = 3.6; // vertical spacing of the stipple
    const r = Math.max(1.05, Math.min(1.9, colW * 0.22));
    for (let l = 0; l < L; l++) {
      const cx = x0 + (l + 0.5) * colW;
      const top = yOf(lay[l]);
      const isWorst = l === worst;
      const isHover = this.hit && ((this.hit.kind === 'layer' && this.hit.l === l) || (this.hit.kind === 'cell' && this.hit.l === l));
      c.fillStyle = isWorst ? CRIMSON : SAP;
      c.globalAlpha = isHover ? 1 : 0.9;
      for (let y = aBot; y >= top - 0.5; y -= pitch) {
        c.beginPath();
        c.arc(cx, y, r, 0, Math.PI * 2);
        c.fill();
      }
      c.globalAlpha = 1;
    }
    // worst-layer label above its column
    {
      const cx = x0 + (worst + 0.5) * colW;
      c.font = '500 10.5px "IBM Plex Mono", ui-monospace, monospace';
      c.fillStyle = CRIMSON;
      c.textAlign = cx > (x0 + x1) / 2 ? 'right' : 'left';
      c.textBaseline = 'bottom';
      c.fillText(`layer ${worst} · ${lay[worst].toFixed(1)}×`, cx + (c.textAlign === 'right' ? -4 : 4), Math.max(aTop + 2, yOf(lay[worst]) - 3));
    }
    // layer axis
    c.font = '500 9.5px "IBM Plex Mono", ui-monospace, monospace';
    c.fillStyle = MUTED;
    c.textBaseline = 'top';
    c.textAlign = 'center';
    for (const l of [0, Math.round(L / 4), Math.round(L / 2), Math.round((3 * L) / 4), L - 1]) c.fillText(`L${l}`, x0 + (l + 0.5) * colW, aBot + 4);

    // --- section B ---------------------------------------------------------------------------------------
    if (this.grid) {
      // heatmap: columns = layers (aligned with the stipple above), rows = experts
      const { L: gL, E: gE, rel } = this.grid;
      const cellW = (x1 - x0) / gL;
      const cellH = (bBot - bTop) / gE;
      for (let l = 0; l < gL; l++) {
        for (let e = 0; e < gE; e++) {
          const v = rel[l * gE + e];
          c.fillStyle = ramp(Math.log2(Math.max(1, v)) / 3); // 1× → light, ≥8× → darkest
          c.fillRect(x0 + l * cellW, bTop + e * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
      this.g = { ...this.g, cellW, cellH, rows: gE, perRow: gL };
      c.fillStyle = MUTED;
      c.textAlign = 'right';
      c.textBaseline = 'top';
      c.fillText('expert 0', x0 - 4, bTop);
      c.textBaseline = 'bottom';
      c.fillText(String(gE - 1), x0 - 4, bBot);
      c.textAlign = 'left';
      c.textBaseline = 'bottom';
      c.font = '500 9.5px "IBM Plex Mono", ui-monospace, monospace';
      c.fillText('each layer’s 128 experts ↓ (darker = more than its fair share)', x0, bTop - 3);
    }
  }

  private hitTest(x: number, y: number): Hit | null {
    const m = this.moe(performance.now());
    if (!m) return null;
    const g = this.g;
    if (x < g.x0 || x > g.x1) return null;
    const l = Math.floor((x - g.x0) / g.colW);
    const lay = (m.layer_imbalance ?? []).map((v) => v ?? 1);
    if (y >= g.aTop - 4 && y <= g.aBot + 14 && l >= 0 && l < lay.length) return { kind: 'layer', l, v: lay[l] };
    if (y >= g.bTop && y <= g.bBot) {
      if (this.grid) {
        const e = Math.floor((y - g.bTop) / g.cellH);
        if (e >= 0 && e < this.grid.E && l >= 0 && l < this.grid.L) return { kind: 'cell', l, e, rel: this.grid.rel[l * this.grid.E + e] };
      }
    }
    return null;
  }

  private renderTip(): void {
    const h = this.hit;
    if (!h || !this.mouse) {
      this.tip.style.display = 'none';
      return;
    }
    this.tip.style.display = 'block';
    this.tip.innerHTML =
      h.kind === 'layer'
        ? `<b>Layer ${h.l}</b>: the busiest of its experts takes <b>${h.v.toFixed(1)}×</b> the average load ${h.v < 1.5 ? '(well balanced)' : h.v < 4 ? '(somewhat skewed)' : '(heavily skewed)'}.`
        : `<b>Layer ${h.l}, expert #${h.e}</b>: <b>${h.rel.toFixed(1)}×</b> its fair share of that layer’s tokens.`;
    const left = Math.min(this.mouse.x + 14, this.w - 230);
    this.tip.style.left = `${Math.max(4, left)}px`;
    this.tip.style.top = `${Math.max(4, this.mouse.y - 8)}px`;
  }
}
