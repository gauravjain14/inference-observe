// Small deterministic helpers shared by the scene and the mock.

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Cheap stable pseudo-random in [0,1) from an integer. */
export function h1(n: number): number {
  let x = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

export function gauss(rnd: () => number): number {
  const u = Math.max(1e-9, rnd());
  const v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const clamp = (x: number, a: number, b: number) => (x < a ? a : x > b ? b : x);
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const smooth = (t: number) => t * t * (3 - 2 * t);

export function poisson(rnd: () => number, lambda: number): number {
  if (lambda <= 0) return 0;
  if (lambda > 30) return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * gauss(rnd)));
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= rnd();
  } while (p > L);
  return k - 1;
}

export function fmtInt(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 10_000) return `${(ms / 1000).toFixed(2)} s`;
  return `${(ms / 1000).toFixed(1)} s`;
}

// Inks: [r, g, b] in 0..255
export type Ink = readonly [number, number, number];
export const INK = {
  crimson: [146, 14, 44] as Ink,
  crimsonDeep: [102, 6, 30] as Ink,
  rose: [184, 62, 94] as Ink,
  ultra: [16, 28, 128] as Ink,
  ultraDeep: [10, 16, 78] as Ink,
  ultraPale: [52, 70, 160] as Ink, // cached, reusable: lighter than live colonies but clearly ink
  violet: [58, 32, 138] as Ink,
  sap: [58, 90, 24] as Ink,
  sapLight: [92, 124, 44] as Ink,
  grey: [112, 106, 94] as Ink,
  greyDeep: [80, 74, 66] as Ink,
  membrane: [96, 28, 46] as Ink,
} as const;

export function mixInk(a: Ink, b: Ink, t: number): Ink {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

/** One ink per traffic class (cells' body tint, Fig. 2 dots). Rims stay crimson: they are requests. */
export function classInk(cls: string): Ink {
  switch (cls) {
    case 'chat':
      return [190, 70, 96];
    case 'agentic':
      return [118, 84, 176];
    case 'rag':
      return [70, 110, 178];
    case 'burst':
      return [176, 116, 52];
    case 'spec':
      return [96, 132, 62];
    case 'short':
      return [170, 150, 130];
    case 'medium':
      return [140, 118, 100];
    case 'long':
      return [100, 82, 70];
    default:
      return [150, 140, 128];
  }
}

export const inkCss = (k: Ink, a = 1) => `rgba(${k[0] | 0},${k[1] | 0},${k[2] | 0},${a})`;
