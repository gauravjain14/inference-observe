// Pan / zoom for the plate. World coordinates are the layout's CSS pixels at zoom 1.
import { clamp } from './util';

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export class Camera {
  k = 1;
  tx = 0;
  ty = 0;
  readonly min = 0.5;
  readonly max = 12;
  /** bumped whenever the transform changes, so the overlay can reposition labels */
  version = 0;
  private anim: { k0: number; x0: number; y0: number; k1: number; x1: number; y1: number; t0: number; dur: number } | null = null;

  toScreen(wx: number, wy: number): [number, number] {
    return [wx * this.k + this.tx, wy * this.k + this.ty];
  }

  toWorld(sx: number, sy: number): [number, number] {
    return [(sx - this.tx) / this.k, (sy - this.ty) / this.k];
  }

  /** World rectangle currently visible in a viewport of w × h CSS px. */
  viewBox(w: number, h: number): Box {
    const [x0, y0] = this.toWorld(0, 0);
    const [x1, y1] = this.toWorld(w, h);
    return { x0, y0, x1, y1 };
  }

  zoomAt(sx: number, sy: number, factor: number): void {
    this.anim = null;
    const k2 = clamp(this.k * factor, this.min, this.max);
    const f = k2 / this.k;
    this.tx = sx - (sx - this.tx) * f;
    this.ty = sy - (sy - this.ty) * f;
    this.k = k2;
    this.version++;
  }

  panBy(dx: number, dy: number): void {
    this.anim = null;
    this.tx += dx;
    this.ty += dy;
    this.version++;
  }

  /** Animate so that `box` fills the viewport (w × h) with some margin. */
  flyTo(box: Box, w: number, h: number, pad = 1.35, dur = 650): void {
    const bw = Math.max(20, box.x1 - box.x0);
    const bh = Math.max(20, box.y1 - box.y0);
    const k1 = clamp(Math.min(w / (bw * pad), h / (bh * pad)), this.min, this.max);
    const cx = (box.x0 + box.x1) / 2;
    const cy = (box.y0 + box.y1) / 2;
    this.animateTo(k1, w / 2 - cx * k1, h / 2 - cy * k1, dur);
  }

  reset(dur = 650): void {
    this.animateTo(1, 0, 0, dur);
  }

  get isHome(): boolean {
    return Math.abs(this.k - 1) < 1e-3 && Math.abs(this.tx) < 0.5 && Math.abs(this.ty) < 0.5 && !this.anim;
  }

  private animateTo(k1: number, x1: number, y1: number, dur = 650): void {
    this.anim = { k0: this.k, x0: this.tx, y0: this.ty, k1, x1, y1, t0: performance.now(), dur };
  }

  tick(now: number): void {
    const a = this.anim;
    if (!a) return;
    const t = clamp((now - a.t0) / a.dur, 0, 1);
    const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    // zoom interpolates geometrically, translation linearly in screen space
    this.k = a.k0 * Math.pow(a.k1 / a.k0, e);
    this.tx = a.x0 + (a.x1 - a.x0) * e;
    this.ty = a.y0 + (a.y1 - a.y0) * e;
    this.version++;
    if (t >= 1) this.anim = null;
  }
}
