// Source tags: every number says where it comes from.
//   ● measured  — reported by the engine, proxy or NVML
//   ◐ derived   — computed by the board from measured data (sizes from tokens ÷ page size, rates, grouping)
//   ○ estimated — a heuristic (cached-tissue lifetime, cached share)
// Positions, curves and drift are layout and carry no tag.
export type Src = 'm' | 'd' | 'e';

const INFO: Record<Src, [string, string]> = {
  m: ['●', 'measured — reported by the engine, proxy or NVML'],
  d: ['◐', 'derived — computed by the board from measured data'],
  e: ['○', 'estimated — a heuristic, not reported by the engine'],
};

export function tag(s: Src): string {
  const [g, t] = INFO[s];
  return `<i class="tg tg-${s}" title="${t}">${g}</i>`;
}

export const TAG_LEGEND = `${tag('m')} measured ${tag('d')} derived ${tag('e')} estimated · positions and curves are layout`;
