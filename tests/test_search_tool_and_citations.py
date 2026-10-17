from __future__ import annotations

import json
from pathlib import Path

import httpx

from docspilot.agent.loop import Agent, AgentConfig, AgentFinished, extract_citations
from docspilot.agent.pending import InMemoryPendingStore
from docspilot.agent.tools import (
    SearchDocsTool,
    Source,
    ToolContext,
    ToolRegistry,
    default_tools,
)
from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.types import ToolCall
from docspilot.rag.embeddings import HashEmbedder
from docspilot.rag.ingest import ingest_corpus
from docspilot.rag.rerank import NoopReranker
from docspilot.rag.retrieve import RetrievalConfig, Retriever
from docspilot.rag.store import InMemoryChunkStore
from docspilot.tickets import InMemoryTicketStore
from tests.test_chat_api import parse_sse

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


async def build_retriever(k: int = 5) -> Retriever:
    store, emb = InMemoryChunkStore(), HashEmbedder()
    await ingest_corpus(CORPUS, store, emb)
    return Retriever(store, emb, NoopReranker(), RetrievalConfig(k_final=k))


async def test_search_tool_returns_sources_within_budget() -> None:
    tool = SearchDocsTool(await build_retriever(k=10))
    tool.char_budget = 1500
    registry = ToolRegistry([tool])
    parsed = registry.parse("search_docs", {"query": "refund policy within days"})
    assert not hasattr(parsed, "content")
    t, args = parsed  # type: ignore[misc]
    ctx = ToolContext(user_id="u", tickets=InMemoryTicketStore())
    outcome = await registry.execute(t, args, ctx, timeout_s=5)
    payload = json.loads(outcome.content)
    assert 1 <= len(payload["results"]) < 10  # budget cut the tail
    assert {s.path for s in outcome.sources} == {r["source"] for r in payload["results"]}
    first = payload["results"][0]
    assert set(first) == {"source", "title", "section", "updated", "text"}


async def test_search_tool_no_results_note() -> None:
    store, emb = InMemoryChunkStore(), HashEmbedder()
    tool = SearchDocsTool(Retriever(store, emb, NoopReranker()))
    out = await tool.run(
        tool.input_model(query="anything"), ToolContext("u", InMemoryTicketStore())
    )
    assert out["results"] == [] and "No relevant" in out["note"]


def test_citations_are_validated_against_what_was_retrieved() -> None:
    retrieved = {
        "policies/refunds.md": Source("policies/refunds.md", "Refund Policy", "Eligibility")
    }
    text = (
        "Refunds are allowed within 30 days [policies/refunds.md]. "
        "See also [policies/made-up.md] and [policies/refunds.md] again."
    )
    cites = extract_citations(text, retrieved)
    assert [c.path for c in cites] == ["policies/refunds.md"]  # deduped, hallucination dropped
    assert extract_citations("no cites", retrieved) == ()


async def test_agent_reports_citations_end_to_end() -> None:
    retriever = await build_retriever()
    llm = FakeLLM(
        [
            Script(
                tool_calls=[
                    ToolCall("s1", "search_docs", {"query": "how many days to request a refund"})
                ]
            ),
            Script(text="30 days [policies/refunds.md] (not [policies/ghost.md])."),
        ]
    )
    agent = Agent(
        llm,
        ToolRegistry(default_tools(retriever)),
        InMemoryPendingStore(),
        AgentConfig(model="m", system="s"),
    )
    events = [
        e
        async for e in agent.run(
            [{"role": "user", "content": "refund window?"}],
            ToolContext("u", InMemoryTicketStore()),
        )
    ]
    done = events[-1]
    assert isinstance(done, AgentFinished)
    assert [c.path for c in done.citations] == ["policies/refunds.md"]
    assert "policies/refunds.md" in {s.path for s in done.retrieved}
    tool_result = llm.requests[1].messages[-1]["content"][0]
    assert not tool_result.get("is_error")


async def test_chat_api_emits_citations_event(app, use_llm, client: httpx.AsyncClient) -> None:  # type: ignore[no-untyped-def]
    use_llm(
        FakeLLM(
            [
                Script(tool_calls=[ToolCall("s1", "search_docs", {"query": "refund window days"})]),
                Script(text="30 days [policies/refunds.md]"),
            ]
        )
    )
    res = await client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "refund window?"}]}
    )
    events = parse_sse(res.text)
    names = [e for e, _ in events]
    assert names.index("citations") < names.index("done")
    cites = next(d for e, d in events if e == "citations")["sources"]
    assert cites[0]["path"] == "policies/refunds.md"
    assert cites[0]["title"] == "Refund Policy"


async def test_readyz_and_debug_search(client: httpx.AsyncClient) -> None:
    ready = await client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["chunks"] > 50

    res = await client.get("/v1/debug/search", params={"q": "webhook endpoint disabled"})
    body = res.json()
    assert res.status_code == 200 and body["embedder"] == "hash-1024"
    assert body["hits"] and {"vector_rank", "keyword_rank", "fused_score"} <= set(body["hits"][0])


async def test_tool_output_is_always_valid_json_and_sources_match() -> None:
    retriever = await build_retriever(k=20)
    tool = SearchDocsTool(retriever)
    tool.char_budget = 2500
    registry = ToolRegistry([tool])
    parsed = registry.parse("search_docs", {"query": "refund policy invoices payments webhooks"})
    t, args = parsed  # type: ignore[misc]
    out = await registry.execute(t, args, ToolContext("u", InMemoryTicketStore()), timeout_s=5)
    payload = json.loads(out.content)  # would raise on a chopped-off document
    assert len(out.content) <= 2500 + 600  # one oversized first result may be shortened
    assert {s.path for s in out.sources} == {r["source"] for r in payload["results"]}


async def test_generic_backstop_truncation_stays_valid_json() -> None:
    from docspilot.agent.tools import MAX_TOOL_OUTPUT_CHARS, Tool, ToolInput

    class Big(Tool):
        name = "big"
        description = "huge output"
        input_model = ToolInput

        async def run(self, args, ctx):  # type: ignore[no-untyped-def]
            return {"blob": "x" * (MAX_TOOL_OUTPUT_CHARS * 2)}

    registry = ToolRegistry([Big()])
    t, args = registry.parse("big", {})  # type: ignore[misc]
    out = await registry.execute(t, args, ToolContext("u", InMemoryTicketStore()), timeout_s=5)
    data = json.loads(out.content)
    assert data["truncated"] is True and len(out.content) <= MAX_TOOL_OUTPUT_CHARS


def test_citation_variants_are_parsed() -> None:
    r = {
        "a.md": Source("a.md", "A", "A > S"),
        "dir/b.md": Source("dir/b.md", "B", "B > S"),
    }
    text = "x [a.md, dir/b.md] y [a.md#sec] z [source: dir/b.md] w [nope.md]"
    assert [c.path for c in extract_citations(text, r)] == ["a.md", "dir/b.md"]


async def test_readyz_flags_embedding_model_mismatch(app, client: httpx.AsyncClient) -> None:  # type: ignore[no-untyped-def]
    class Other(HashEmbedder):
        name = "voyage:other"

    app.state.rag.embedder = Other()
    res = await client.get("/readyz")
    assert res.status_code == 503 and res.json()["status"] == "embedding_model_mismatch"


def test_fullwidth_bracket_citations_from_open_models_are_parsed() -> None:
    r = {"policies/refunds.md": Source("policies/refunds.md", "Refund Policy", "Eligibility")}
    assert [c.path for c in extract_citations("30 days\u3010policies/refunds.md\u3011", r)] == [
        "policies/refunds.md"
    ]
