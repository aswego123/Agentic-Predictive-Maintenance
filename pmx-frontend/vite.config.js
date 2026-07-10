import path from 'node:path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig(function (_a) {
    var mode = _a.mode;
    var env = loadEnv(mode, process.cwd(), '');
    var apiTarget = env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000';
    return {
        plugins: [react()],
        resolve: {
            alias: {
                '@': path.resolve(__dirname, './src'),
            },
        },
        server: {
            port: 5173,
            host: '127.0.0.1',
            // Dev proxy: frontend calls `/api/*` -> FastAPI at 127.0.0.1:8000.
            // Avoids CORS entirely in dev (CORS not needed in production either
            // until we deploy, per project decision).
            proxy: {
                '/api': {
                    target: apiTarget,
                    changeOrigin: true,
                    rewrite: function (p) { return p.replace(/^\/api/, ''); },
                },
            },
        },
        build: {
            outDir: 'dist',
            sourcemap: true,
        },
    };
});
