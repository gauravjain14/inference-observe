import type { Msg, Source } from '../contract';

/** Replays a recorded JSONL event log (proxy.jsonl or collector.jsonl) with its original timing. */
export class ReplaySource implements Source {
  readonly label: string;
  private timer: number | undefined;
  private stopped = false;

  constructor(private url: string, private speed = 1, private loop = false) {
    const name = url.split('/').pop()?.replace(/\.jsonl$/, '') ?? url;
    this.label = `replay · ${name}${speed !== 1 ? ` · ${speed}×` : ''}`;
  }

  start(emit: (m: Msg) => void, status: (s: string) => void): void {
    status('loading');
    fetch(this.url)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.text();
      })
      .then((text) => {
        const evs: Msg[] = [];
        for (const line of text.split('\n')) {
          if (!line.trim()) continue;
          try {
            const m = JSON.parse(line) as Msg;
            if (m && typeof m.type === 'string') evs.push(m);
          } catch {
            /* skip */
          }
        }
        evs.sort((a, b) => (a.t_ns ?? 0) - (b.t_ns ?? 0));
        if (!evs.length) throw new Error('no events in file');
        this.play(evs, emit, status);
      })
      .catch((e) => status(`replay failed: ${e.message ?? e}`));
  }

  private play(evs: Msg[], emit: (m: Msg) => void, status: (s: string) => void): void {
    const hasHello = evs.some((e) => e.type === 'hello');
    const sources = new Set(evs.map((e) => e.src).filter(Boolean));
    let lap = 0;
    const run = () => {
      if (this.stopped) return;
      const tag = lap++ ? `·${lap - 1}` : '';
      if (!hasHello) {
        emit({
          type: 'hello',
          engine: 'recorded run',
          model: evs.find((e) => e.type === 'req.arrive' && e.model)?.model ?? null,
          gpus: null,
          coverage: { requests: true, steps: false, kv: false, spec: false, gpu: false },
          label: `Recorded ${[...sources].join(' + ') || 'events'}`,
        });
      }
      status('playing');
      const t0 = evs[0].t_ns ?? 0;
      const wall0 = performance.now();
      const shift = Date.now() * 1e6 - t0; // re-stamp to the present so ages read naturally
      let i = 0;
      const pump = () => {
        if (this.stopped) return;
        const elapsed = (performance.now() - wall0) * this.speed * 1e6;
        while (i < evs.length && (evs[i].t_ns ?? 0) - t0 <= elapsed) {
          const e = evs[i++];
          // a recording says so in the header, with the time it was recorded
          const rec = e.type === 'hello'
            ? { label: 'Recorded specimen', replay: { file: this.url, speed: this.speed, recorded_utc: new Date((e.t_ns ?? t0) / 1e6).toISOString().slice(0, 16).replace('T', ' ') } }
            : {};
          emit({ ...e, ...rec, t_ns: (e.t_ns ?? t0) + shift, ...(e.rid && tag ? { rid: `${e.rid}${tag}` } : {}) } as Msg);
        }
        if (i < evs.length) {
          this.timer = window.setTimeout(pump, 8);
        } else if (this.loop) {
          status('looping');
          this.timer = window.setTimeout(run, 2500);
        } else {
          status('replay finished');
        }
      };
      pump();
    };
    run();
  }

  stop(): void {
    this.stopped = true;
    window.clearTimeout(this.timer);
  }
}
