import { defineConfig } from "evalite/config";

export default defineConfig({
  // Answers go through a real model and a tool loop: allow time, keep concurrency modest
  // (the API rate-limits per user and the provider has its own limits).
  testTimeout: 180_000,
  maxConcurrency: 2,
  server: { port: 3006 },
  hideTable: true, // the UI (npm run eval:dev) is the place to read cases; keep CI logs short
});
