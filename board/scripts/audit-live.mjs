// Accuracy audit for a live board: every headline number next to an independent measurement.
// Engine: SGLang's own Prometheus counters and gauges. Requests: the collector's recorded events.
//   BOARD_URL=http://127.0.0.1:18310/ node scripts/audit-live.mjs <engine-url> <collector.jsonl> [seconds]
import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';

const [engine, events, secs = '60'] = process.argv.slice(2);
const base = process.env.BOARD_URL ?? 'http://127.0.0.1:18300/';
const ws = process.env.BOARD_WS ?? 'ws://127.0.0.1:18220/ws';

async function metrics() {
  const txt = await (await fetch(`${engine}/metrics`)).text();
  const out = {};
  for (const line of txt.split('\n')) {
    if (!line.startsWith('sglang:')) continue;
    const m = line.match(/^(sglang:[a-z_]+)(\{[^}]*\})?\s+([-0-9.eE+]+)$/);
    if (!m) continue;
    const key = m[1] + (m[1].endsWith('_bucket') ? `|${(m[2].match(/le="([^"]+)"/) || [])[1]}` : '');
    out[key] = (out[key] ?? 0) + Number(m[3]);
  }
  out.t = Date.now() / 1000;
  return out;
}
const pct = (a, p) => (a.length ? [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.round((p / 100) * (a.length - 1)))] : null);

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
await page.addInitScript(() => { try { localStorage.setItem('io.board.keySeen', '1'); } catch {} });
await page.goto(`${base}?ws=${ws}`, { waitUntil: 'load' });
await page.waitForTimeout(15000);
const m0 = await metrics();
await page.waitForTimeout(Number(secs) * 1000);
const [m1, board] = await Promise.all([metrics(), page.evaluate(() => {
  const b = window.__board;
  const s = b.store.stats();
  const map = b.scene.mapFor(0);
  const txt = (sel) => [...document.querySelectorAll(sel)].map((e) => e.textContent.replace(/\s+/g, ' ').trim());
  return { stats: s, shares: b.scene.kvShares(0), tokens: map?.tokens ?? null, cap: map?.capacityTokens ?? null,
    readout: txt('.readout .m'), captions: txt('.cap').slice(0, 5), fig5: txt('.fig5 .foot, .fig5 .footer, .fig5 .note').slice(0, 3) };
})]);
await browser.close();

const dt = m1.t - m0.t;
const d = (k) => (m1[k] ?? 0) - (m0[k] ?? 0);
// proxy-measured TTFT in the board's own 60 s window, from the recorded events
const now = Date.now() * 1e6;
const ttft = [];
for (const line of readFileSync(events, 'utf8').split('\n')) {
  if (!line.includes('"req.first_token"')) continue;
  const e = JSON.parse(line);
  if (now - e.t_ns < 60e9) ttft.push(e.ttft_ms);
}
// Cumulative, per finished request: the tap's counts against the usage the engine returned through the API
// (recorded by the proxy). The tap counts a request's prefix at its first prefill step; the step's batch names
// which requests those were. Generated tokens: all steps against all streamed tokens.
const firstSeen = new Set();
const tapPrompt = new Map(), tapCached = new Map();
const apiPrompt = new Map(), apiCached = new Map(), done = new Map(), streamed = new Map();
let gen = 0, t0ns = null;
for (const line of readFileSync(events, 'utf8').split('\n')) {
  if (!line) continue;
  const e = JSON.parse(line);
  if (e.type === 'engine.step') {
    t0ns ??= e.t_ns;
    gen += e.gen_tokens ?? 0;
    const fresh = (e.batch?.reqs ?? []).filter(([rid, ph]) => ph === 'p' && !firstSeen.has(rid)).map(([rid]) => rid);
    fresh.forEach((rid) => firstSeen.add(rid));
    if (e.prefix && fresh.length) {
      const key = fresh.sort().join('+');
      tapPrompt.set(key, e.prefix.queries);
      tapCached.set(key, e.prefix.hits);
    }
  } else if (e.type === 'req.prefill_stats' && t0ns && e.t_ns >= t0ns) {
    if (e.prompt_tokens != null) apiPrompt.set(e.rid, e.prompt_tokens);
    if (e.cached_tokens != null) apiCached.set(e.rid, e.cached_tokens);
  } else if (e.type === 'req.done' && e.usage) done.set(e.rid, e.usage.completion_tokens ?? 0);
  else if (e.type === 'req.chunk') streamed.set(e.rid, (streamed.get(e.rid) ?? 0) + (e.n ?? 1));
}
let pT = 0, pA = 0, cT = 0, cA = 0, groups = 0, mism = 0;
for (const [key, q] of tapPrompt) {
  const rids = key.split('+');
  if (!rids.every((r) => done.has(r) && apiCached.has(r) && apiPrompt.has(r))) continue; // finished requests only
  groups++;
  const ap = rids.reduce((a, r) => a + apiPrompt.get(r), 0);
  const ac = rids.reduce((a, r) => a + apiCached.get(r), 0);
  pT += q; pA += ap; cT += tapCached.get(key); cA += ac;
  if (q !== ap || tapCached.get(key) !== ac) mism++;
}
let out = 0;
for (const [rid, n] of streamed) out += done.has(rid) ? done.get(rid) : n;
// each KV map carries the radix tree's own counts from the same instant: cached must equal evictable
let maps = 0, exact = 0;
for (const line of readFileSync(events, 'utf8').split('\n')) {
  if (!line.includes('"kv.map"')) continue;
  const m = JSON.parse(line);
  if (m.tree?.evictable == null || !m.tokens) continue;
  maps++;
  if (m.tokens.cached === m.tree.evictable) exact++;
}
const rows = [
  [`prompt tokens (${groups} finished)`, pT, pA, `tap first-prefill vs API usage; ${mism} differ`],
  [`cached tokens (${groups} finished)`, cT, cA, 'tap first-prefill vs API usage'],
  ['generated tokens (total)', gen, out, 'tap steps vs streamed tokens'],
  ['KV maps: cached = tree evictable', exact, maps, 'maps that match / maps, same instant'],
  // SGLang updates these gauges on its own logging interval: under load they can trail the board by a second
  ['running (engine)', board.stats.running, m1['sglang:num_running_reqs'], 'gauge'],
  ['waiting (engine queue)', board.stats.waiting, m1['sglang:num_queue_reqs'], 'gauge'],
  ['KV in use (header)', board.stats.kv, m1['sglang:token_usage'], 'gauge'],
  ['KV live share (map)', board.shares?.live, m1['sglang:kv_used_tokens'] / m1['sglang:max_total_num_tokens'], 'gauge kv_used / max'],
  ['KV cached share (map)', board.shares?.cached, m1['sglang:kv_evictable_tokens'] / m1['sglang:max_total_num_tokens'], 'gauge kv_evictable / max'],
  ['KV free share (map)', board.shares?.free, m1['sglang:kv_available_tokens'] / m1['sglang:max_total_num_tokens'], 'gauge kv_available / max'],
  ['tokens/s (2 s)', board.stats.tokS, d('sglang:generation_tokens_total') / dt, `engine counter over ${dt.toFixed(0)} s (rough)`],
  ['TTFT p50 (60 s)', board.stats.ttftP50, pct(ttft, 50), `proxy events, n=${ttft.length}`],
  ['TTFT p90 (60 s)', board.stats.ttftP90, pct(ttft, 90), 'proxy events'],
];
for (const [k, a, b2, how] of rows) {
  const fa = typeof a === 'number' ? a.toFixed(3) : String(a);
  const fb = typeof b2 === 'number' ? b2.toFixed(3) : String(b2);
  console.log(`${k.padEnd(24)} board ${fa.padStart(10)}   independent ${fb.padStart(10)}   (${how})`);
}
console.log('header:', board.readout.join(' | '));
console.log('captions:', board.captions.join(' | '));
