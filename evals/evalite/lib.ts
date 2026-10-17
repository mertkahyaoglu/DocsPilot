/**
 * Shared helpers for the Evalite suites. The suites are black-box: they call a *running*
 * DocsPilot API (default http://127.0.0.1:8000) and reuse the same golden dataset as the
 * Python harness (evals/golden.jsonl), so both harnesses score the same cases.
 *
 * Start the API for evals with limits/caching off (otherwise the rate limiter and the
 * response cache distort results):  make evalite-server
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { reportTrace } from "evalite/traces";

export const BASE = (process.env.DOCSPILOT_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const API_KEY = process.env.DOCSPILOT_API_KEY;

export type Category =
  | "answerable"
  | "multi_hop"
  | "contradiction"
  | "unanswerable"
  | "injection"
  | "adversarial"
  | "ticket_intent"
  | "smalltalk";

export interface Case {
  id: string;
  question: string;
  category: Category;
  expected_sources: string[];
  expected_facts: string[];
  must_not: string[];
  expected_behavior: "answer" | "abstain" | "create_ticket" | "chitchat";
  ticket_must_contain?: string[];
  notes: string;
}

export function loadGolden(): Case[] {
  const file = fileURLToPath(new URL("../golden.jsonl", import.meta.url));
  let cases: Case[] = readFileSync(file, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as Case);
  const ids = process.env.EVAL_IDS?.split(",").map((s) => s.trim());
  if (ids?.length) cases = cases.filter((c) => ids.includes(c.id));
  const only = process.env.EVAL_CATEGORIES?.split(",").map((s) => s.trim());
  if (only?.length) cases = cases.filter((c) => only.includes(c.category));
  const limit = Number(process.env.EVAL_LIMIT ?? "0");
  return limit > 0 ? cases.slice(0, limit) : cases;
}

function headers(): Record<string, string> {
  const h: Record<string, string> = { "content-type": "application/json" };
  if (API_KEY) h.authorization = `Bearer ${API_KEY}`;
  return h;
}

export interface ChatResult {
  text: string;
  tools: string[];
  citations: string[];
  confirmation: { tool: string; summary: string; arguments: Record<string, unknown> }[];
  model: string | null;
  inputTokens: number;
  outputTokens: number;
  costUsd: number | null;
  stopReason: string;
  error: string | null;
  latencyMs: number;
}

export async function chat(question: string): Promise<ChatResult> {
  const started = Date.now();
  const res = await fetch(`${BASE}/v1/chat`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ messages: [{ role: "user", content: question }] }),
  });
  if (!res.ok) throw new Error(`POST /v1/chat -> ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const raw = (await res.text()).replaceAll("\r\n", "\n");
  const out: ChatResult = {
    text: "",
    tools: [],
    citations: [],
    confirmation: [],
    model: null,
    inputTokens: 0,
    outputTokens: 0,
    costUsd: null,
    stopReason: "",
    error: null,
    latencyMs: 0,
  };
  for (const block of raw.trim().split("\n\n")) {
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (!data) continue;
    const d = JSON.parse(data);
    if (event === "delta") out.text += d.text;
    else if (event === "reset") out.text = "";
    else if (event === "replace") out.text = d.text;
    else if (event === "tool_start") out.tools.push(d.name);
    else if (event === "citations") out.citations = d.sources.map((s: { path: string }) => s.path);
    else if (event === "confirmation_required") out.confirmation = d.items;
    else if (event === "error") out.error = `${d.code}: ${d.message}`;
    else if (event === "done") {
      out.stopReason = d.stop_reason;
      out.model = d.model ?? null;
      out.inputTokens = d.usage?.input_tokens ?? 0;
      out.outputTokens = d.usage?.output_tokens ?? 0;
      out.costUsd = d.cost_usd ?? null;
    }
  }
  out.latencyMs = Date.now() - started;
  // Feeds Evalite's trace view (token usage and timing per call).
  reportTrace({
    input: question,
    output: out.text,
    usage: {
      inputTokens: out.inputTokens,
      outputTokens: out.outputTokens,
      totalTokens: out.inputTokens + out.outputTokens,
    },
    start: started,
    end: Date.now(),
  });
  return out;
}

export interface SearchHit {
  path: string;
  section: string;
  rerank_score: number | null;
}

/** Retrieval alone (no model): the dev-only debug endpoint. */
export async function search(query: string): Promise<SearchHit[]> {
  const res = await fetch(`${BASE}/v1/debug/search?q=${encodeURIComponent(query)}`, {
    headers: headers(),
  });
  if (!res.ok) throw new Error(`GET /v1/debug/search -> ${res.status} (is the API running in dev?)`);
  return ((await res.json()) as { hits: SearchHit[] }).hits;
}

export const uniqueInOrder = (xs: string[]): string[] => [...new Set(xs)];

export function money(usd: number | null): string {
  if (usd === null) return "n/a";
  return usd === 0 ? "$0" : `$${usd.toFixed(usd < 0.01 ? 5 : 4)}`;
}
