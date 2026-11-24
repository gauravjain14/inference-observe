// Evaluation suite: screenshot the board across viewports, modes and interactions, and record
// console errors and frame times. Output: shots/eval/<case>.png + shots/eval/report.json
//   BOARD_URL=http://127.0.0.1:18300/ node scripts/evaluate.mjs [filter]
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync } from 'node:fs';

const base = process.env.BOARD_URL ?? 'http://127.0.0.1:18300/';
const only = process.argv[2] ?? '';
mkdirSync('shots/eval', { recursive: true });

const cases = [
  { name: 'replay-2560', q: '', w: 2560, h: 1440, wait: 14 },
  { name: 'replay-1920', q: '', w: 1920, h: 1080, wait: 14 },
  { name: 'firstvisit-1600', q: '', w: 1600, h: 900, wait: 8, firstVisit: true },
  { name: 'replay-1600', q: '', w: 1600, h: 900, wait: 14 },
  { name: 'replay-1366', q: '', w: 1366, h: 768, wait: 14 },
  { name: 'replay-1600-later', q: '', w: 1600, h: 900, wait: 40 },
  { name: 'memmap-1600', q: '', w: 1600, h: 900, wait: 12, keys: ['m'], after: 3 },
  { name: 'zoom-kv-1600', q: '', w: 1600, h: 900, wait: 12, zoom: { at: 'kv', steps: 6 } },
  { name: 'hover-fig4-1600', q: '', w: 1600, h: 900, wait: 12, hover: { sel: '.fig5 canvas, .fig4 canvas', fx: 0.42, fy: 0.5 } },
  { name: 'inspect-kv-1600', q: '', w: 1600, h: 900, wait: 12, click: 'kv' },
  { name: 'mock-burst-1600', q: 'mock=burst', w: 1600, h: 900, wait: 20 },
  { name: 'mock-pd-1600', q: 'mock=pd', w: 1600, h: 900, wait: 14 },
  { name: 'mock-rep-1600', q: 'mock=rep', w: 1600, h: 900, wait: 14 },
  { name: 'nodata-1600', q: 'ws=ws://127.0.0.1:1/ws', w: 1600, h: 900, wait: 2 },
].filter((c) => c.name.includes(only));

const browser = await chromium.launch({
  channel: 'chrome', headless: true,
  args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist'],
});
const report = [];
for (const c of cases) {
  const page = await browser.newPage({ viewport: { width: c.w, height: c.h }, deviceScaleFactor: 1 });
  // returning visitor by default: the reading key only opens by itself on a first visit
  if (!c.firstVisit) await page.addInitScript(() => { try { localStorage.setItem('io.board.keySeen', '1'); } catch {} });
  const errors = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(base + (c.q ? `?${c.q}` : ''), { waitUntil: 'load' });
  await page.waitForTimeout(c.wait * 1000);
  const anchor = async (what) => page.evaluate((what) => {
    const s = window.__board?.scene;
    if (!s) return null;
    if (what === 'kv') return { x: s.L.kv.cx, y: s.L.kv.cy };
    if (what === 'moe') return s.L.moe ? { x: s.L.moe.cx ?? s.L.moe.x, y: s.L.moe.cy ?? s.L.moe.y } : null;
    return null;
  }, what);
  if (c.keys) {
    for (const k of c.keys) await page.keyboard.press(k);
    await page.waitForTimeout((c.after ?? 2) * 1000);
  }
  if (c.zoom) {
    const p = await anchor(c.zoom.at);
    if (p) {
      await page.mouse.move(p.x, p.y);
      for (let i = 0; i < c.zoom.steps; i++) { await page.mouse.wheel(0, -240); await page.waitForTimeout(120); }
      await page.waitForTimeout(1500);
    } else errors.push(`no anchor for ${c.zoom.at}`);
  }
  if (c.hover) {
    const target = page.locator(c.hover.sel).filter({ visible: true }).first(); // dense models show Fig. 4 as .fig5, MoE models as .fig4
    await target.waitFor({ state: 'visible', timeout: 5000 }).catch(() => {});
    const box = await target.boundingBox();
    if (box) { await page.mouse.move(box.x + box.width * c.hover.fx, box.y + box.height * c.hover.fy); await page.waitForTimeout(900); }
    else errors.push(`no element ${c.hover.sel}`);
  }
  if (c.click) {
    const p = await anchor(c.click);
    if (p) { await page.mouse.click(p.x, p.y); await page.waitForTimeout(1200); } else errors.push(`no anchor for ${c.click}`);
  }
  const file = `shots/eval/${c.name}.png`;
  await page.screenshot({ path: file });
  const perf = await page.evaluate(() => {
    const b = window.__board;
    return b ? { ...b.perf, status: b.store.status, events: b.store.events } : null;
  });
  // text overlap check: count pairs of visible text boxes in the overlay that intersect
  const overlaps = await page.evaluate(() => {
    const els = [...document.querySelectorAll('#overlay *')].filter((e) => {
      if (!e.childNodes.length || ![...e.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())) return false;
      const st = getComputedStyle(e);
      if (st.visibility === 'hidden' || st.display === 'none' || Number(st.opacity) < 0.2) return false;
      const r = e.getBoundingClientRect();
      return r.width > 2 && r.height > 2 && r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
    });
    // the visible part of a box: clipped by every ancestor that hides overflow
    const clip = (e) => {
      const b = e.getBoundingClientRect();
      let r = { left: b.left, top: b.top, right: b.right, bottom: b.bottom };
      for (let p = e.parentElement; p && p !== document.body; p = p.parentElement) {
        const st = getComputedStyle(p);
        if (st.overflowX === 'visible' && st.overflowY === 'visible') continue;
        const q = p.getBoundingClientRect();
        r = { left: Math.max(r.left, q.left), top: Math.max(r.top, q.top), right: Math.min(r.right, q.right), bottom: Math.min(r.bottom, q.bottom) };
      }
      return r;
    };
    const boxes = els.map((e) => ({ e, r: clip(e) })).filter((b) => b.r.right - b.r.left > 2 && b.r.bottom - b.r.top > 2);
    const hits = [];
    for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      if (a.e.contains(b.e) || b.e.contains(a.e)) continue;
      // opaque panels (the reading key, the docked inspector) cover what is under them
      const covered = [...document.querySelectorAll('.key, .inspector')].some((m) => getComputedStyle(m).display !== 'none' && m.contains(a.e) !== m.contains(b.e));
      if (covered) continue;
      const ix = Math.min(a.r.right, b.r.right) - Math.max(a.r.left, b.r.left);
      const iy = Math.min(a.r.bottom, b.r.bottom) - Math.max(a.r.top, b.r.top);
      if (ix > 3 && iy > 3) {
        const where = (e) => { const r = e.getBoundingClientRect(); let p = e, path = []; for (let k = 0; k < 3 && p; k++, p = p.parentElement) path.push(typeof p.className === 'string' && p.className ? p.className.split(' ')[0] : p.tagName.toLowerCase()); return `${path.join('<')}@${Math.round(r.left)},${Math.round(r.top)}`; };
        hits.push([a.e.textContent.trim().slice(0, 40), b.e.textContent.trim().slice(0, 40), where(a.e), where(b.e)]);
      }
    }
    return hits.slice(0, 12);
  });
  report.push({ case: c.name, file, perf, errors: errors.slice(0, 5), overlaps });
  console.log(c.name, JSON.stringify(perf), errors.length ? `ERRORS ${errors.length}` : '', overlaps.length ? `OVERLAPS ${overlaps.length}` : '');
  await page.close();
}
writeFileSync('shots/eval/report.json', JSON.stringify(report, null, 2));
await browser.close();
