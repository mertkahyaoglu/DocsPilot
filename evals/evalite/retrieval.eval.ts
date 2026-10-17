import { evalite } from "evalite";
import { createScorer } from "evalite";
import { type Case, loadGolden, search, uniqueInOrder } from "./lib.js";

const K = 5;

type Input = { id: string; question: string };

// Same definitions as docspilot/evals/metrics.py: ranked *documents* (chunks deduped),
// cut at k, so the two harnesses agree.
const recall = createScorer<Input, string[], string[]>({
  name: `recall@${K}`,
  description: "Share of expected documents present in the top-k documents",
  scorer: ({ output, expected }) => {
    const want = new Set(expected);
    return [...want].filter((p) => output.slice(0, K).includes(p)).length / want.size;
  },
});

const hit = createScorer<Input, string[], string[]>({
  name: `hit@${K}`,
  description: "At least one expected document in the top-k",
  scorer: ({ output, expected }) => (expected.some((p) => output.slice(0, K).includes(p)) ? 1 : 0),
});

const mrr = createScorer<Input, string[], string[]>({
  name: "mrr",
  description: "Reciprocal rank of the first expected document",
  scorer: ({ output, expected }) => {
    const i = output.slice(0, K).findIndex((p) => expected.includes(p));
    return i < 0 ? 0 : 1 / (i + 1);
  },
});

evalite<Input, string[], string[]>("Retrieval (hybrid, production settings)", {
  data: () =>
    loadGolden()
      .filter((c: Case) => c.expected_sources.length > 0)
      .map((c) => ({
        input: { id: c.id, question: c.question },
        expected: c.expected_sources,
      })),
  task: async (input) => uniqueInOrder((await search(input.question)).map((h) => h.path)),
  scorers: [recall, hit, mrr],
  columns: ({ input, output, expected }) => [
    { label: "case", value: input.id },
    { label: "question", value: input.question },
    { label: "top docs", value: output.slice(0, K).join("\n") },
    { label: "expected", value: (expected ?? []).join("\n") },
  ],
});
