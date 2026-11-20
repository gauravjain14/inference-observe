import { defineConfig } from 'vite';

export default defineConfig({
  server: { host: '127.0.0.1', port: 18300, strictPort: true },
  preview: { host: '127.0.0.1', port: 18301, strictPort: true },
  build: { target: 'es2022' },
});
