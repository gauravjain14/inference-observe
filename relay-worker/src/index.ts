/**
 * Public relay for the board stream, on Cloudflare (a Worker and one Durable Object).
 *
 * Your machine pushes the stream out to /ingest (io-push, with a secret token). Viewers read it from /ws.
 * - /ws: GET WebSocket only, from the allowed page origins; viewers only watch, and one that sends data is
 *   disconnected. At most MAX_VIEWERS viewers, MAX_PER_IP per address (IPv6 by /64), each for at most an hour
 *   (the board reconnects by itself).
 * - /ingest: one connection, with the token; a new one replaces the old. Unknown or unauthorised requests get 404.
 * - Ingest messages are "s" + a snapshot (sent to each viewer that joins) or "f" + frames (sent to every viewer).
 *   The relay forwards text and does not parse it.
 */
import { DurableObject } from 'cloudflare:workers';

export interface Env {
  RELAY: DurableObjectNamespace<Relay>;
  INGEST_TOKEN: string; // secret: wrangler secret put INGEST_TOKEN
  ALLOWED_ORIGINS: string; // comma-separated page origins
  MAX_VIEWERS?: string;
  MAX_PER_IP?: string;
}

const MAX_SESSION_MS = 60 * 60 * 1000;
const notFound = () => new Response('not found', { status: 404 });

async function tokenOk(req: Request, env: Env): Promise<boolean> {
  if (!env.INGEST_TOKEN || env.INGEST_TOKEN.length < 32) return false;
  const got = new TextEncoder().encode(req.headers.get('Authorization') ?? '');
  const want = new TextEncoder().encode(`Bearer ${env.INGEST_TOKEN}`);
  return got.byteLength === want.byteLength && crypto.subtle.timingSafeEqual(got, want);
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const { pathname } = new URL(req.url);
    const upgrade = req.headers.get('Upgrade')?.toLowerCase() === 'websocket';
    const relay = env.RELAY.get(env.RELAY.idFromName('board'));
    if (pathname === '/ws') {
      if (req.method !== 'GET') return new Response('method not allowed', { status: 405 });
      if (!upgrade) return new Response('expected a WebSocket', { status: 426 });
      const allowed = env.ALLOWED_ORIGINS.split(',').map((s) => s.trim()).filter(Boolean);
      if (!allowed.includes(req.headers.get('Origin') ?? '')) return new Response('origin not allowed', { status: 403 });
      return relay.fetch(req);
    }
    if (pathname === '/ingest' && req.method === 'GET' && upgrade && (await tokenOk(req, env))) return relay.fetch(req);
    return notFound();
  },
} satisfies ExportedHandler<Env>;

/** IPv6 addresses count by /64, the block one household or host holds. */
function addressKey(ip: string): string {
  if (!ip.includes(':')) return ip;
  const head = ip.toLowerCase().split('::')[0].split(':');
  const groups = ip.includes('::') ? [...head, '0', '0', '0', '0'].slice(0, 4) : ip.toLowerCase().split(':').slice(0, 4);
  return `${groups.map((h) => (parseInt(h || '0', 16) || 0).toString(16)).join(':')}::/64`;
}

export class Relay extends DurableObject<Env> {
  private snapshot: string | null = null;
  private sweptAt = 0;

  async fetch(req: Request): Promise<Response> {
    const { pathname } = new URL(req.url);
    const [client, server] = Object.values(new WebSocketPair());
    if (pathname === '/ingest') {
      for (const old of this.ctx.getWebSockets('ingest')) old.close(1012, 'replaced');
      this.ctx.acceptWebSocket(server, ['ingest']);
      return new Response(null, { status: 101, webSocket: client });
    }
    const viewers = this.ctx.getWebSockets('viewer');
    const key = addressKey(req.headers.get('CF-Connecting-IP') ?? 'unknown');
    const maxViewers = Number(this.env.MAX_VIEWERS ?? 200);
    const maxPerIp = Number(this.env.MAX_PER_IP ?? 4);
    const fromKey = viewers.filter((ws) => (ws.deserializeAttachment() as { key?: string } | null)?.key === key).length;
    if (viewers.length >= maxViewers || fromKey >= maxPerIp) return new Response('too many viewers', { status: 429 });
    this.ctx.acceptWebSocket(server, ['viewer']);
    server.serializeAttachment({ key, since: Date.now() });
    if (this.snapshot) server.send(this.snapshot);
    return new Response(null, { status: 101, webSocket: client });
  }

  async webSocketMessage(ws: WebSocket, msg: string | ArrayBuffer): Promise<void> {
    if (!this.ctx.getTags(ws).includes('ingest')) {
      ws.close(1008, 'read-only'); // viewers only watch
      return;
    }
    if (typeof msg !== 'string' || msg.length < 2) return;
    const body = msg.slice(1);
    if (msg[0] === 's') {
      this.snapshot = body;
      return;
    }
    if (msg[0] !== 'f') return;
    const now = Date.now();
    const sweep = now - this.sweptAt > 60_000;
    if (sweep) this.sweptAt = now;
    for (const v of this.ctx.getWebSockets('viewer')) {
      if (sweep && now - ((v.deserializeAttachment() as { since?: number } | null)?.since ?? now) > MAX_SESSION_MS) {
        v.close(1000, 'session time limit; reconnect');
        continue;
      }
      try {
        v.send(body);
      } catch {
        // closed between listing and sending
      }
    }
  }

  async webSocketClose(ws: WebSocket, code: number): Promise<void> {
    try {
      ws.close([1005, 1006, 1015].includes(code) ? 1000 : code, 'closing'); // reserved codes cannot be sent
    } catch {
      // already closed
    }
  }

  async webSocketError(ws: WebSocket): Promise<void> {
    try {
      ws.close(1011, 'error');
    } catch {
      // already closed
    }
  }
}
