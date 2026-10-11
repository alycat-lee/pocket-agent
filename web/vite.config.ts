/// <reference types="vitest/config" />
import { execSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const root = path.dirname(fileURLToPath(import.meta.url));

function gitShortHash(): string {
  try {
    return execSync("git rev-parse --short HEAD").toString().trim();
  } catch {
    return `t${Date.now()}`;
  }
}

const outDir = path.resolve(root, "../src/plugins/static");
const version = gitShortHash();

export default defineConfig({
  define: {
    __BUILD_VERSION__: JSON.stringify(version),
  },
  plugins: [
    react(),
    {
      name: "build-info",
      closeBundle() {
        const info = {
          version,
          built_at: new Date().toISOString(),
        };
        writeFileSync(
          path.join(outDir, "build-info.json"),
          JSON.stringify(info, null, 2),
        );
      },
    },
  ],
  base: "/",
  build: {
    outDir,
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    proxy: {
      "/chat": "http://127.0.0.1:3000",
      "/sessions": "http://127.0.0.1:3000",
      "/api": "http://127.0.0.1:3000",
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
