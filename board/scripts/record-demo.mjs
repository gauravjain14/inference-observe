// Record the demo video: drive the board on a script and capture its frames (Chrome DevTools screencast).
//   io-replay-ws runs/local/<run> --start <seconds> --port 18230 &
//   node scripts/record-demo.mjs --ws ws://127.0.0.1:18230/ws --uncached <new session rid> \
//     --cached <its next call's rid> --gl gpu --warm 30 --subtitle "..." --source "..."
// Writes JPEG frames and frames.txt (an ffmpeg concat list with real durations):
//   ffmpeg -f concat -safe 0 -i frames.txt -vf fps=30,format=yuv420p -c:v libx264 -crf 18 demo.mp4
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync } from 'node:fs';

const arg = (k, d) => {
  const i = process.argv.indexOf(`--${k}`);
  return i > 0 ? process.argv[i + 1] : d;
};
const BOARD = arg('board', 'http://127.0.0.1:18300/');
const WS = arg('ws', 'ws://127.0.0.1:18230/ws');
const U = arg('uncached');
const C = arg('cached');
const OUT = arg('out', '../runs/frames');
const GL = arg('gl', 'swiftshader');
const SUB = arg('subtitle', '');
const WARM = Number(arg('warm', '5')) * 1000; // seconds the board runs before capture starts
mkdirSync(OUT, { recursive: true });

const args = GL === 'gpu'
  ? ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan']
  : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
const browser = await chromium.launch({ channel: 'chrome', headless: true, args });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.addInitScript(() => { try { localStorage.setItem('io.board.keySeen', '1'); } catch {} });
await page.goto(`${BOARD}?ws=${encodeURIComponent(WS)}`, { waitUntil: 'load' });
console.log('renderer:', await page.evaluate(() => {
  const gl = document.createElement('canvas').getContext('webgl2');
  const x = gl && gl.getExtension('WEBGL_debug_renderer_info');
  return x ? gl.getParameter(x.UNMASKED_RENDERER_WEBGL) : 'unknown';
}));

// overlay: title card, narration card, end card — in the board's own type and colours
await page.addStyleTag({ content: `
  #dm-title, #dm-end { position: fixed; inset: 0; z-index: 100; display: flex; flex-direction: column; align-items: center;
    justify-content: center; background: rgba(243, 238, 227, 0.94); transition: opacity 0.6s ease; }
  #dm-title h1, #dm-end h1 { margin: 0; font: 400 76px/1.05 Newsreader, Georgia, serif; color: #2a2622; letter-spacing: -0.01em; }
  #dm-title p, #dm-end p { margin: 16px 0 0; font: italic 400 30px/1.3 Newsreader, Georgia, serif; color: #3f3a33; }
  #dm-title .k, #dm-end .k { margin-top: 28px; font: 500 15px/1.4 "IBM Plex Mono", monospace; letter-spacing: 0.14em; text-transform: uppercase; color: #9e1432; }
  #dm-card { position: fixed; right: 0; top: 0; width: 460px; height: 1080px; z-index: 90; box-sizing: border-box;
    padding: 34px 34px; background: #f5f1e7; border-left: 1px solid rgba(42,38,34,0.18); transition: opacity 0.4s ease; }
  #dm-card .k { font: 500 13px/1.4 "IBM Plex Mono", monospace; letter-spacing: 0.16em; text-transform: uppercase; color: #9e1432; }
  #dm-card h2 { margin: 14px 0 0; font: 400 36px/1.15 Newsreader, Georgia, serif; color: #2a2622; }
  #dm-card p { margin: 16px 0 0; font: 400 21px/1.45 Newsreader, Georgia, serif; color: #3f3a33; }
  #dm-card .m { margin-top: 30px; padding-top: 18px; border-top: 1px solid rgba(42,38,34,0.2); }
  #dm-card .m b { display: block; font: 500 54px/1 "IBM Plex Mono", monospace; color: #9e1432; }
  #dm-card .m span { display: block; margin-top: 8px; font: 400 15px/1.3 "IBM Plex Mono", monospace; color: #57504a; }
  #dm-card .m.blue b { color: #1c2c96; }
  #dm-card .src { position: absolute; left: 34px; right: 34px; bottom: 30px; font: 400 13px/1.5 "IBM Plex Mono", monospace; color: #6b6258; }
` });
await page.evaluate((sub) => {
  const t = document.createElement('div');
  t.id = 'dm-title';
  t.innerHTML = `<h1>inference-observe</h1><p>A live view inside an LLM inference engine</p><div class="k">${sub}</div>`;
  document.body.appendChild(t);
  const c = document.createElement('div');
  c.id = 'dm-card';
  c.style.opacity = '0';
  document.body.appendChild(c);
}, SUB);

// --- capture -------------------------------------------------------------------------------------
const frames = [];
const cdp = await page.context().newCDPSession(page);
cdp.on('Page.screencastFrame', async (f) => {
  frames.push({ t: f.metadata.timestamp, data: f.data });
  await cdp.send('Page.screencastFrameAck', { sessionId: f.sessionId }).catch(() => {});
});
const wait = (ms) => page.waitForTimeout(ms);
const now = () => page.evaluate(() => performance.now());

// helpers that run in the page
const rid = (prefix) => page.evaluate((p) => [...window.__board.store.reqs.keys()].find((k) => k.startsWith(p)) ?? null, prefix);
const reqInfo = (full) => page.evaluate((k) => {
  const r = window.__board.store.reqs.get(k);
  if (!r) return null;
  return { prompt: r.promptTokens ?? null, cached: r.cachedTokens ?? null, ttft: r.tFirst != null ? r.tFirst - (r.tAdmit ?? r.tArrive) : null };
}, full);
const card = (html, opacity = 1) => page.evaluate(([h, o]) => {
  const c = document.getElementById('dm-card');
  c.innerHTML = h;
  c.style.opacity = String(o);
}, [html, opacity]);
const fmt = (n) => Number(n).toLocaleString('en-US');
const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
const src = `<div class="src">${arg('source', '')}</div>`;

async function until(fn, ms) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    const v = await fn();
    if (v) return v;
    await wait(80);
  }
  return null;
}

// let the board fill before the first frame
await wait(WARM);
await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
const t0 = Date.now();

// 1. title, until the uncached request arrives (3 s minimum)
await wait(3000);
const u = await until(() => rid(U), 9000);
await page.evaluate(() => { document.getElementById('dm-title').style.opacity = '0'; });
const kvBox = await page.evaluate(() => {
  const s = window.__board.scene;
  const b = s.bodies[0];
  return { x0: b.cx - b.rx - 30, x1: b.cx + b.rx + 30, y0: b.cy - b.ry - 70, y1: b.cy + b.ry + 20 };
});
// fit the KV body (and its caption) between the header and the follow line above the figures
const frameKv = () => page.evaluate((b) => {
  const B = window.__board;
  const top = 165;
  const bottom = document.querySelector('.fig3').getBoundingClientRect().top - 44;
  const f = (top + bottom) / (bottom - top); // flyTo centres on h/2: stretch the box so it lands in [top, bottom]
  const cy = (b.y0 + b.y1) / 2;
  const half = ((b.y1 - b.y0) / 2) * f;
  B.camera.flyTo({ x0: b.x0, x1: b.x1, y0: cy - half, y1: cy + half }, B.scene.L.PW, top + bottom, 1.08, 1800);
}, kvBox);

// 2. a new agent session: nothing in the cache
if (u) {
  await page.evaluate((k) => window.__board.follow.followReq(k, performance.now()), u);
  await frameKv();
  await card(`<div class="k">1 · A new session</div><h2>A coding agent starts a session</h2>
    <p>Nothing of it is in the KV cache yet: the engine computes the prompt, 2,048 tokens per step.</p>${src}`);
  const info = await until(async () => { const i = await reqInfo(u); return i && i.prompt && i.ttft != null && i.cached != null ? i : null; }, 15000);
  if (info) await card(`<div class="k">1 · A new session</div><h2>${fmt(info.prompt)} tokens, ${fmt(info.cached)} from the cache</h2>
    <p>Every token of the prompt is computed before the first token comes back.</p>
    <div class="m"><b>${fmtMs(info.ttft)}</b><span>to first token, after admission</span></div>${src}`);
  await wait(3000);
}

// 3. the tool runs: the KV pool, cell by cell, while the session's conversation waits in the cache
await page.evaluate(() => { const f = window.__board.follow; f.enabled = false; f.set(null, performance.now()); });
await page.keyboard.press('m');
await card(`<div class="k">2 · The agent runs a tool</div><h2>Its conversation stays in the KV cache</h2>
  <p>The KV pool as SGLang holds it, each cell split by its tokens: <b>deep blue</b> held by two or more running requests,
  <b>blue</b> reused from the cache, <b>ultramarine</b> computed by its request, <b>pale</b> kept for reuse, bare paper free.</p>${src}`);
await wait(6000);
await page.keyboard.press('m');

// 4. meanwhile, every request so far: Fig. 4 in the spotlight
await page.evaluate(() => window.__board.camera.reset(1400));
await card(`<div class="k">3 · Every request so far</div><h2>Reused vs computed, request by request</h2>
  <p>Blue: prompt tokens served from the cache. Green: tokens computed. Below: time to first token, on a log scale.</p>${src}`);
await page.evaluate(() => {
  const f = document.querySelector('.fig5');
  const dim = document.createElement('div');
  dim.id = 'dm-dim';
  dim.style.cssText = 'position:fixed;inset:0;z-index:84;background:rgba(243,238,227,0.86);opacity:0;transition:opacity .6s ease';
  f.parentElement.insertBefore(dim, f);
  const r = f.getBoundingClientRect();
  const s = 1.75;
  const cx = (1920 - 460) / 2;
  const cy = 1080 / 2 + 40;
  f.style.transition = 'transform .8s cubic-bezier(.2,.7,.2,1)';
  f.style.transformOrigin = 'top left';
  f.style.zIndex = '85';
  requestAnimationFrame(() => {
    dim.style.opacity = '1';
    f.style.transform = `translate(${cx - (r.left + r.width * s / 2)}px, ${cy - (r.top + r.height * s / 2)}px) scale(${s})`;
  });
});
await wait(5200);
await page.evaluate(() => {
  const f = document.querySelector('.fig5');
  const dim = document.getElementById('dm-dim');
  f.style.transform = '';
  if (dim) dim.style.opacity = '0';
  setTimeout(() => { dim?.remove(); f.style.zIndex = ''; }, 800);
});
const c = await until(() => rid(C), 20000);

// 5. the same session's next call
if (c) {
  await page.evaluate((k) => window.__board.follow.followReq(k, performance.now()), c);
  await frameKv();
  await card(`<div class="k">4 · Same session, next call</div><h2>The agent sends the conversation back</h2>
    <p>With the tool's output added. The earlier part is still in the KV cache.</p>${src}`);
  const info = await until(async () => { const i = await reqInfo(c); return i && i.prompt && i.ttft != null && i.cached != null ? i : null; }, 12000);
  if (info) await card(`<div class="k">4 · Same session, next call</div><h2>${fmt(info.prompt)} tokens, ${fmt(info.cached)} from the cache</h2>
    <p>Only ${fmt(info.prompt - info.cached)} tokens are computed.</p>
    <div class="m blue"><b>${fmtMs(info.ttft)}</b><span>to first token, after admission</span></div>${src}`);
  await wait(3200);
}

// 6. end card
await page.evaluate(() => {
  const e = document.createElement('div');
  e.id = 'dm-end';
  e.style.opacity = '0';
  e.innerHTML = '<h1>inference-observe</h1><p>Every request, scheduler step and KV block, as it happens.</p><div class="k">SGLang · vLLM · Apache-2.0</div>';
  document.body.appendChild(e);
  requestAnimationFrame(() => (e.style.opacity = '1'));
});
await wait(3200);
await cdp.send('Page.stopScreencast');
console.log(`captured ${frames.length} frames in ${((Date.now() - t0) / 1000).toFixed(1)} s`);

// frames + concat list (each frame shown until the next one arrived)
const lines = [];
frames.forEach((f, i) => {
  const name = `f${String(i).padStart(5, '0')}.jpg`;
  writeFileSync(`${OUT}/${name}`, Buffer.from(f.data, 'base64'));
  const d = i + 1 < frames.length ? frames[i + 1].t - f.t : 1 / 30;
  lines.push(`file '${name}'`, `duration ${Math.max(0.001, d).toFixed(4)}`);
});
lines.push(`file 'f${String(frames.length - 1).padStart(5, '0')}.jpg'`);
writeFileSync(`${OUT}/frames.txt`, lines.join('\n') + '\n');
await browser.close();
