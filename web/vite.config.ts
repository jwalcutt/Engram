import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The local API endpoints the UI consumes (proxied to `mem serve` during `npm run dev`).
const API_ROUTES = [
  "/health", "/ask", "/chat", "/concepts", "/timeline", "/memory-stats",
  "/forgetting", "/recall", "/drift", "/distill", "/usage", "/conversations",
];

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs so the bundle works when served at the API root by FastAPI.
  base: "./",
  build: {
    // Built bundle is committed and shipped inside the [ui] extra; `mem serve` serves it.
    outDir: "../src/crossmodalrag/api/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: Object.fromEntries(API_ROUTES.map((p) => [p, "http://127.0.0.1:8765"])),
  },
});
