// Fig. 2: prompt vs output tokens per request (log–log), coloured by traffic class.
// Below the diagonal a request is prefill-heavy, above it decode-heavy.
import { classLabel, displayRid, trafficClass, type Store } from '../state';
import { classInk, fmtInt, inkCss } from '../util';
import { draggable } from './draggable';

const esc = (s: unknown) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

// chart size in CSS px; set by setRect so text renders at true size instead of being scaled
let W = 300;
let H = 176;
const PAD = { l: 50, r: 12, t: 22, b: 32 }; // left: tick labels + the rotated axis title
const X0 = 10;
const X1 = 40_000; // prompt tokens
const Y0 = 1;
const Y1 = 8_000; // output tokens
const WINDOW_MS = 5 * 60_000;

const lx = (v: number) => PAD.l + ((Math.log10(Math.max(X0, v)) - Math.log10(X0)) / (Math.log10(X1) - Math.log10(X0))) * (W - PAD.l - PAD.r);
const ly = (v: number) => H - PAD.b - ((Math.log10(Math.max(Y0, v)) - Math.log10(Y0)) / (Math.log10(Y1) - Math.log10(Y0))) * (H - PAD.t - PAD.b);

export class Fig2 {
  private el: HTMLElement;
  private body: HTMLElement;
  private last = 0;

  constructor(root: HTMLElement, private store: Store, private onPick: (rid: string) => void) {
    this.el = document.createElement('figure');
    this.el.className = 'bandfig fig2';
    const cap = document.createElement('figcaption');
    cap.innerHTML = '<span class="no">Fig. 2</span> prompt vs output tokens · last 5 min';
    this.body = document.createElement('div');
    this.el.append(cap, this.body);
    root.appendChild(this.el);
    draggable(this.el, cap);
    this.el.addEventListener('mouseover', (ev) => {
      const rid = (ev.target as Element).getAttribute?.('data-rid');
      if (rid && this.store.reqs.get(rid) && ['queued', 'prefill', 'decode'].includes(this.store.reqs.get(rid)!.phase)) this.onPick(rid);
    });
  }

  /** Position and size the figure (CSS px, relative to the overlay); the chart scales to fit. */
  setRect(x: number, y: number, w: number, h: number, visible = true): void {
    this.el.style.display = visible ? 'block' : 'none';
    if (!visible) return;
    if (!this.el.dataset.moved) {
      this.el.style.left = `${x}px`;
      this.el.style.top = `${y}px`;
    }
    this.el.style.width = `${w}px`;
    this.el.style.height = `${h}px`;
    const nw = Math.max(220, Math.floor(w - 20));
    const nh = Math.max(110, Math.floor(h - 60));
    if (nw !== W || nh !== H) {
      W = nw;
      H = nh;
      this.last = 0; // redraw at the new size
    }
  }

  update(now: number, follow: string | null): void {
    if (now - this.last < 500) return;
    this.last = now;
    const pts: { rid: string; cls: string; x: number; y: number; live: boolean }[] = [];
    for (const r of this.store.recent) if (now - r.at < WINDOW_MS) pts.push({ rid: r.rid, cls: r.cls, x: r.prompt, y: Math.max(1, r.output), live: false });
    for (const r of this.store.reqs.values()) {
      if (r.phase !== 'decode' || r.chunks < 1) continue;
      const p = r.promptTokens ?? (r.promptChars ? r.promptChars / 3.8 : null);
      if (p) pts.push({ rid: r.rid, cls: trafficClass(r), x: p, y: r.chunks, live: true });
    }
    const classes = [...new Set(pts.map((p) => p.cls))].sort();
    const ticks = (vals: number[], f: (v: number) => number, axis: 'x' | 'y') =>
      vals
        .map((v) =>
          axis === 'x'
            ? `<line x1="${f(v)}" y1="${H - PAD.b}" x2="${f(v)}" y2="${H - PAD.b + 3}"/><text x="${f(v)}" y="${H - PAD.b + 12}" class="tk">${v >= 1000 ? `${v / 1000}k` : v}</text>`
            : `<line x1="${PAD.l - 3}" y1="${f(v)}" x2="${PAD.l}" y2="${f(v)}"/><text x="${PAD.l - 5}" y="${f(v) + 3}" class="tk end">${v >= 1000 ? `${v / 1000}k` : v}</text>`,
        )
        .join('');
    const dots = pts
      .map((p) => {
        const c = inkCss(classInk(p.cls), p.live ? 1 : 0.9);
        const on = p.rid === follow;
        const edge = inkCss(classInk(p.cls).map((v) => v * 0.6) as unknown as [number, number, number], 1);
        return `<circle data-rid="${esc(p.rid)}" cx="${lx(p.x).toFixed(1)}" cy="${ly(p.y).toFixed(1)}" r="${on ? 4.8 : p.live ? 3.4 : 2.9}" fill="${p.live ? 'none' : c}" stroke="${on ? '#5e0418' : edge}" stroke-width="${on ? 1.8 : p.live ? 1.4 : 0.9}"><title>${esc(displayRid(p.rid))} · ${esc(classLabel(p.cls))} · ${fmtInt(p.x)} prompt → ${fmtInt(p.y)} output${p.live ? ' (so far)' : ''}</title></circle>`;
      })
      .join('');
    // diagonal: output = prompt
    const a = Math.max(X0, Y0);
    const b = Math.min(X1, Y1);
    this.body.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
        <rect x="${PAD.l}" y="${PAD.t}" width="${W - PAD.l - PAD.r}" height="${H - PAD.t - PAD.b}" class="frame"/>
        <line x1="${lx(a)}" y1="${ly(a)}" x2="${lx(b)}" y2="${ly(b)}" class="diag"/>
        <text x="${lx(14)}" y="${ly(4500)}" class="zone">decode-heavy ↖</text>
        <text x="${lx(36000)}" y="${ly(1.5)}" class="zone end">↘ prefill-heavy</text>
        <g class="axis">${ticks([100, 1000, 10000], lx, 'x')}${ticks([10, 100, 1000], ly, 'y')}</g>
        <text x="${(PAD.l + W - PAD.r) / 2}" y="${H - 4}" class="ax">prompt tokens →</text>
        <text x="11" y="${(PAD.t + H - PAD.b) / 2}" class="ax" transform="rotate(-90 11 ${(PAD.t + H - PAD.b) / 2})">output tokens →</text>
        ${dots}
      </svg>
      <div class="cls">${classes.map((c) => `<span><i style="background:${inkCss(classInk(c))}"></i>${esc(classLabel(c))}</span>`).join('')}${pts.length ? '' : '<em>no finished requests yet</em>'}<span class="hint">○ live · ● done · hover to follow</span></div>`;
  }
}
