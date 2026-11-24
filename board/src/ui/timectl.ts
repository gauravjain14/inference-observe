// Time: pause the plate (events keep buffering), rewind up to 5 minutes from a client-side
// ring buffer of events, and jump back to live.
import type { Msg } from '../contract';
import type { Scene } from '../scene';
import type { Store } from '../state';

const KEEP_MS = 5 * 60_000;
const MAX_EVENTS = 250_000;

export type Mode = 'live' | 'paused' | 'replay';

export class TimeCtl {
  mode: Mode = 'live';
  headLabel = 'live'; // what following the newest event is called: 'live' on a feed, 'latest' in a recording
  private buf: { lt: number; m: Msg }[] = [];
  private head = 0; // index of the oldest kept event (buffer is compacted lazily)
  private lastHello: Msg | null = null;
  private pausedAt = 0; // buffer length when paused
  private cursor = 0; // local time being shown while replaying
  private shift = 0; // replayed events are stamped lt + shift so ages read naturally
  private idx = 0; // next buffered event to apply while replaying
  private bar: HTMLElement;
  private btn: HTMLButtonElement;
  private range: HTMLInputElement;
  private label: HTMLElement;
  private liveBtn: HTMLButtonElement;
  readonly resetBtn: HTMLButtonElement;

  constructor(root: HTMLElement, private store: Store, private scene: Scene) {
    this.bar = document.createElement('div');
    this.bar.className = 'timebar';
    this.btn = document.createElement('button');
    this.btn.type = 'button';
    this.btn.title = 'Pause / resume (space)';
    this.btn.addEventListener('click', () => this.toggle(performance.now()));
    this.range = document.createElement('input');
    this.range.type = 'range';
    this.range.min = String(-KEEP_MS / 1000);
    this.range.max = '0';
    this.range.step = '1';
    this.range.value = '0';
    this.range.title = 'Rewind the last 5 minutes';
    this.range.addEventListener('change', () => this.seekBack(-Number(this.range.value) * 1000, performance.now()));
    this.label = document.createElement('span');
    this.label.className = 'tl';
    this.liveBtn = document.createElement('button');
    this.liveBtn.type = 'button';
    this.liveBtn.textContent = '● live';
    this.liveBtn.addEventListener('click', () => this.goLive(performance.now()));
    this.resetBtn = document.createElement('button');
    this.resetBtn.type = 'button';
    this.resetBtn.className = 'reset';
    this.resetBtn.textContent = 'reset view';
    this.resetBtn.title = 'Reset zoom and pan (Esc)';
    this.bar.append(this.btn, this.range, this.label, this.liveBtn, this.resetBtn);
    root.appendChild(this.bar);
    this.paint(performance.now());
  }

  place(right: number, top: number): void {
    this.bar.style.right = `${right}px`;
    this.bar.style.top = `${top}px`;
  }

  /** Every incoming event lands here first. Returns true if it should be applied now (live). */
  push(m: Msg, lt: number): boolean {
    if (m.type === 'hello') this.lastHello = m;
    this.buf.push({ lt, m });
    // forget events older than the window (compact occasionally)
    while (this.head < this.buf.length && (lt - this.buf[this.head].lt > KEEP_MS || this.buf.length - this.head > MAX_EVENTS)) this.head++;
    if (this.head > 50_000) {
      this.buf.splice(0, this.head);
      this.idx = Math.max(0, this.idx - this.head);
      this.pausedAt = Math.max(0, this.pausedAt - this.head);
      this.head = 0;
    }
    return this.mode === 'live';
  }

  toggle(now: number): void {
    if (this.mode === 'live') {
      this.mode = 'paused';
      this.pausedAt = this.buf.length;
    } else if (this.mode === 'paused' && this.idx === 0 && this.cursor === 0) {
      // resume from a live pause: fast-forward everything that arrived meanwhile
      for (let i = this.pausedAt; i < this.buf.length; i++) this.store.ingest(this.buf[i].m, this.buf[i].lt);
      this.mode = 'live';
    } else if (this.mode === 'paused') {
      this.mode = 'replay';
      this.shift = now - this.cursor;
    } else {
      this.mode = 'paused';
    }
    this.paint(now);
  }

  get frozen(): boolean {
    return this.mode === 'paused';
  }

  /** Show the plate as it was `backMs` ago, then play forward in real time. */
  seekBack(backMs: number, now: number): void {
    if (backMs <= 500) return this.goLive(now);
    const oldest = this.buf[this.head]?.lt ?? now;
    const T = Math.max(now - backMs, oldest); // can't rewind past what this page has seen
    this.rebuild(T, now);
    this.mode = 'replay';
    this.paint(now);
  }

  goLive(now: number): void {
    this.rebuild(now, now);
    this.mode = 'live';
    this.cursor = 0;
    this.idx = 0;
    this.store.wallOffsetMs = 0;
    this.range.value = '0';
    this.paint(now);
  }

  private rebuild(T: number, now: number): void {
    const notes = new Map(this.store.notes);
    this.store.reset();
    this.scene.reset();
    this.shift = now - T;
    if (this.lastHello) this.store.ingest(this.lastHello, now);
    let i = this.head;
    for (; i < this.buf.length && this.buf[i].lt <= T; i++) this.store.ingest(this.buf[i].m, this.buf[i].lt + this.shift);
    for (const [id, n] of notes) if (!this.store.notes.has(id)) this.store.notes.set(id, n);
    this.store.notesVersion++;
    this.idx = i;
    this.cursor = T;
    this.store.wallOffsetMs = now - T;
  }

  /** While replaying, feed buffered events whose time has come. */
  tick(now: number, dt: number): void {
    if (this.mode === 'replay') {
      this.cursor += dt * 1000;
      while (this.idx < this.buf.length && this.buf[this.idx].lt <= this.cursor) {
        const e = this.buf[this.idx++];
        this.store.ingest(e.m, e.lt + this.shift);
      }
      this.store.wallOffsetMs = now - this.cursor;
      if (this.cursor >= now - 30) this.goLive(now);
    }
    this.paint(now);
  }

  private lastPaint = 0;
  private paint(now: number): void {
    if (now - this.lastPaint < 200) return;
    this.lastPaint = now;
    this.btn.textContent = this.mode === 'live' || this.mode === 'replay' ? '❚❚' : '▶';
    this.bar.dataset.mode = this.mode;
    const behind = this.mode === 'live' ? 0 : this.mode === 'paused' && this.cursor === 0 ? (now - (this.buf[this.pausedAt - 1]?.lt ?? now)) : now - this.cursor;
    if (this.mode !== 'live' && document.activeElement !== this.range) this.range.value = String(-Math.round(behind / 1000));
    const mm = Math.floor(behind / 60_000);
    const ss = Math.floor((behind % 60_000) / 1000);
    this.label.textContent =
      this.mode === 'live' ? this.headLabel : this.mode === 'paused' ? `paused · −${mm}:${String(ss).padStart(2, '0')}` : `replay · −${mm}:${String(ss).padStart(2, '0')}`;
    this.liveBtn.textContent = `● ${this.headLabel}`;
    this.liveBtn.style.display = this.mode === 'live' ? 'none' : 'inline-block';
    const held = Math.min(KEEP_MS, now - (this.buf[this.head]?.lt ?? now));
    this.range.min = String(-Math.max(1, Math.round(held / 1000)));
    this.range.title = `Rewind up to ${Math.round(held / 1000)} s (this page keeps the last 5 minutes)`;
  }
}
