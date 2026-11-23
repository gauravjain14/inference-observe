import { defineConfig, type Plugin } from 'vite';

/**
 * Public build (VITE_PUBLIC_WS=wss://<host>/ws): the page gets a Content-Security-Policy that allows only its own
 * files and that one feed, and sends no referrer.
 */
function publicPage(): Plugin {
  return {
    name: 'public-page',
    apply: 'build',
    transformIndexHtml(html) {
      const ws = process.env.VITE_PUBLIC_WS;
      if (!ws) return html;
      const feed = new URL(ws);
      const local = feed.hostname === '127.0.0.1' || feed.hostname === 'localhost';
      if (feed.protocol !== 'wss:' && !(local && feed.protocol === 'ws:')) throw new Error('VITE_PUBLIC_WS must be a wss:// URL');
      const csp = [
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data:",
        "font-src 'self' data:",
        `connect-src 'self' ${feed.origin}`,
        "base-uri 'none'",
        "form-action 'none'",
        "object-src 'none'",
      ].join('; ');
      return html.replace(
        '<head>',
        `<head>\n    <meta http-equiv="Content-Security-Policy" content="${csp}" />\n    <meta name="referrer" content="no-referrer" />`,
      );
    },
  };
}

export default defineConfig({
  plugins: [publicPage()],
  server: { host: '127.0.0.1', port: 18300, strictPort: true },
  preview: { host: '127.0.0.1', port: 18301, strictPort: true },
  build: { target: 'es2022' },
});
