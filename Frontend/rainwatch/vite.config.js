import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ml': {
        target: 'file://' + path.resolve(__dirname, '../../ml'),
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ml/, '')
      },
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  }
});