// Drag a floating panel by its title bar; double-click the title to collapse it.
import { clamp } from '../util';

export function draggable(panel: HTMLElement, handle: HTMLElement): void {
  handle.classList.add('handle');
  let st: { x: number; y: number; l: number; t: number; id: number } | null = null;
  handle.addEventListener('pointerdown', (e) => {
    if ((e.target as HTMLElement).closest('button, a, input')) return;
    const r = panel.getBoundingClientRect();
    panel.style.left = `${r.left}px`;
    panel.style.top = `${r.top}px`;
    panel.style.right = 'auto';
    panel.style.bottom = 'auto';
    st = { x: e.clientX, y: e.clientY, l: r.left, t: r.top, id: e.pointerId };
    handle.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  handle.addEventListener('pointermove', (e) => {
    if (!st || e.pointerId !== st.id) return;
    const l = clamp(st.l + e.clientX - st.x, -panel.offsetWidth + 80, window.innerWidth - 60);
    const t = clamp(st.t + e.clientY - st.y, 0, window.innerHeight - 30);
    panel.style.left = `${l}px`;
    panel.style.top = `${t}px`;
    panel.dataset.moved = '1';
  });
  const end = () => (st = null);
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);
  handle.addEventListener('dblclick', (e) => {
    if ((e.target as HTMLElement).closest('button')) return;
    panel.classList.toggle('collapsed');
  });
}
