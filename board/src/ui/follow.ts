// "Follow a request": ring one live request, dim the rest, and narrate it from its real events.
import type { Scene } from '../scene';
import { displayRid, type Req, type Store } from '../state';
import { fmtInt, fmtMs } from '../util';

const esc = (s: unknown) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

const REPICK_MS = 12_000; // a fresh request at most this often
const LINGER_MS = 3_500; // keep a finished story on screen this long
const MAX_FOLLOW_MS = 30_000;

export class Follow {
  enabled = true;
  rid: string | null = null;
  private pickedAt = -1e12;
  private endedAt = 0;
  private el: HTMLElement;
  private btn: HTMLButtonElement;
  private text: HTMLElement;

  constructor(root: HTMLElement, private store: Store, private scene: Scene) {
    this.el = document.createElement('div');
    this.el.className = 'follow';
    this.btn = document.createElement('button');
    this.btn.type = 'button';
    this.btn.addEventListener('click', () => this.toggle(performance.now()));
    this.text = document.createElement('div');
    this.text.className = 'story';
    this.el.append(this.btn, this.text);
    root.appendChild(this.el);
  }

  /** One line, just above the figure band; long narrations are cut with an ellipsis, never wrapped. */
  placeLine(x: number, y: number, w: number): void {
    this.el.style.left = `${x}px`;
    this.el.style.top = `${y}px`;
    this.el.style.width = `${w}px`;
    this.el.style.right = '';
    this.el.style.bottom = '';
  }

  toggle(now: number): void {
    this.enabled = !this.enabled;
    if (!this.enabled) this.set(null, now);
    else this.pick(now);
  }

  /** Follow a specific request (from a click). */
  followReq(rid: string, now: number): void {
    this.enabled = true;
    this.set(rid, now);
  }

  private set(rid: string | null, now: number): void {
    this.rid = rid;
    this.pickedAt = now;
    this.endedAt = 0;
    this.scene.follow = rid;
  }

  private pick(now: number): void {
    const live = [...this.store.reqs.values()].filter(
      (r) => !r.partial && this.scene.ents.has(r.rid) && (r.phase === 'queued' || r.phase === 'prefill' || r.phase === 'decode'),
    );
    // prefer one that just arrived, so the story starts at the door
    const fresh = live.filter((r) => now - r.lArrive < 2500 && r.phase !== 'decode').sort((a, b) => b.lArrive - a.lArrive);
    const decoding = live
      .filter((r) => r.phase === 'decode')
      .sort((a, b) => (b.maxTokens ?? 0) - b.chunks - ((a.maxTokens ?? 0) - a.chunks));
    const r = fresh[0] ?? decoding[0];
    this.set(r ? r.rid : null, now);
  }

  tick(now: number): void {
    this.btn.textContent = this.enabled ? '◉ following · F' : '◎ follow a request · F';
    this.btn.classList.toggle('on', this.enabled);
    if (!this.enabled) {
      this.text.innerHTML = '';
      return;
    }
    const r = this.rid ? this.store.reqs.get(this.rid) : undefined;
    const ended = !r || r.phase === 'done' || r.phase === 'failed';
    if (r && ended && !this.endedAt) this.endedAt = now;
    const due =
      !r ||
      !this.scene.ents.has(r.rid) ||
      (ended && now - this.endedAt > LINGER_MS && now - this.pickedAt > REPICK_MS * 0.5) ||
      now - this.pickedAt > MAX_FOLLOW_MS;
    if (due && now - this.pickedAt > 1500) this.pick(now);
    const cur = this.rid ? this.store.reqs.get(this.rid) : undefined;
    this.text.innerHTML = cur ? narrate(cur, this.scene.pd) : '<span class="wait">waiting for a request to follow…</span>';
  }
}

/** One line, built only from events this request has produced so far. */
export function narrate(r: Req, pd: boolean): string {
  const nowMs = Date.now();
  const b = (s: string) => `<b>${s}</b>`;
  const parts: string[] = [];
  parts.push(`${b(esc(displayRid(r.rid)))}${r.workload ? ` <i>${esc(r.workload)}</i>` : ''}`);
  if (r.phase === 'queued') parts.push(`waiting at the door ${b(fmtMs(nowMs - r.tArrive))}`);
  else if (r.tAdmit != null) parts.push(`waited ${b(fmtMs(r.tAdmit - r.tArrive))} at the door`);
  const prompt = r.promptTokens != null ? fmtInt(r.promptTokens) : r.promptChars ? `≈${fmtInt(r.promptChars / 3.8)}` : null;
  const cached = r.cachedTokens ? ` (${fmtInt(r.cachedTokens)} from cache)` : '';
  if (pd && r.pd) {
    if (r.prefillMs != null) parts.push(`prefilled ${prompt ? `${b(prompt)} tokens${cached} ` : ''}on the prefill GPU in ${b(fmtMs(r.prefillMs))}`);
    else if (r.pd === 'prefill') parts.push(`prefilling ${prompt ? `${b(prompt)} tokens ` : ''}on the prefill GPU…`);
    if (r.pd === 'transfer' || r.pd === 'decode') parts.push(`KV handed to the decode GPU${r.kvBlocks ? ` (${fmtInt(r.kvBlocks)} blocks)` : ''}`);
  } else if (r.tFirst != null) {
    const ttft = r.tFirst - (r.tAdmit ?? r.tArrive);
    const eng = r.engine?.prefill_s != null ? ` (engine queue ${fmtMs((r.engine.queued_s ?? 0) * 1000)}, prefill ${fmtMs(r.engine.prefill_s * 1000)})` : '';
    parts.push(`prefilled ${prompt ? `${b(prompt)} tokens${cached}` : 'the prompt'} → first token after ${b(fmtMs(ttft))}${eng}`);
  } else if (r.phase === 'prefill') {
    parts.push(`prefilling ${prompt ? `${b(prompt)} tokens` : 'the prompt'}…`);
  }
  if (r.tFirst != null) {
    const end = r.tDone ?? nowMs;
    const secs = Math.max(0.001, (end - r.tFirst) / 1000);
    const rate = r.chunks / secs;
    parts.push(`decoding ${b(`${rate.toFixed(0)} tok/s`)} · ${fmtInt(r.chunks)} tokens${r.phase === 'decode' ? ' so far' : ''}`);
  }
  if (r.phase === 'done' && r.tDone != null) parts.push(`done in ${b(fmtMs(r.tDone - r.tArrive))}${r.finish ? ` <i>(${esc(r.finish)})</i>` : ''}`);
  if (r.phase === 'failed') parts.push(`<span class="err">${esc(r.error ?? 'failed')}</span>`);
  return parts.join(' <span class="arr">→</span> ');
}
