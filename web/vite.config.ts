/// <reference types="vitest/config" />
import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: path.resolve(root, "../src/plugins/static"),
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    proxy: {
      "/chat": "http://127.0.0.1:3000",
      "/api": "http://127.0.0.1:3000",
      "/sessions": "http://127.0.0.1:3000",
      "/approve": "http://127.0.0.1:3000",
      "/resume": "http://127.0.0.1:3000",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
});
