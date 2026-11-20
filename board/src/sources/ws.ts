import type { Msg, Source } from '../contract';

/** Live source: the collector's WebSocket. Frames are one event or a JSON array of events. */
export class WsSource implements Source {
  readonly label: string;
  private ws: WebSocket | null = null;
  private stopped = false;
  private retry = 500;
  private timer: number | undefined;

  constructor(private url: string) {
    this.label = `live · ${url.replace(/^wss?:\/\//, '')}`;
  }

  start(emit: (m: Msg) => void, status: (s: string) => void): void {
    const connect = () => {
      if (this.stopped) return;
      status('connecting');
      const ws = new WebSocket(this.url);
      this.ws = ws;
      ws.onopen = () => {
        this.retry = 500;
        status('connected');
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') return;
        const text = ev.data.trim();
        try {
          if (text.startsWith('[')) {
            for (const m of JSON.parse(text) as Msg[]) emit(m);
          } else if (text.includes('\n')) {
            for (const line of text.split('\n')) if (line.trim()) emit(JSON.parse(line));
          } else {
            emit(JSON.parse(text));
          }
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onclose = () => {
        if (this.stopped) return;
        status(`disconnected — retrying in ${(this.retry / 1000).toFixed(1)} s`);
        this.timer = window.setTimeout(connect, this.retry);
        this.retry = Math.min(this.retry * 1.6, 8000);
      };
      ws.onerror = () => ws.close();
    };
    connect();
  }

  stop(): void {
    this.stopped = true;
    window.clearTimeout(this.timer);
    this.ws?.close();
  }
}
