from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from docspilot.agent.factory import build_orchestrator
from docspilot.agent.loop import NeedsConfirmation
from docspilot.agent.orchestrator import Orchestrator, is_ticket_followup
from docspilot.agent.pending import InMemoryPendingStore
from docspilot.agent.tools import ToolContext
from docspilot.auth import ApiKeyAuthenticator, ChainAuthenticator, OIDCAuthenticator
from docspilot.config import Settings
from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.types import LLMRequest, ToolCall
from docspilot.prompts import PromptRegistry
from docspilot.rag.embeddings import HashEmbedder
from docspilot.rag.ingest import ingest_corpus
from docspilot.rag.rerank import NoopReranker
from docspilot.rag.retrieve import Retriever
from docspilot.rag.store import InMemoryChunkStore
from docspilot.tickets import InMemoryTicketStore
from tests.conftest import ROOT
from tests.test_chat_api import parse_sse
from tests.test_observability import base_script, close, open_client

ISSUER = "https://idp.example.com/"
AUD = "docspilot-api"


# ---- OIDC ----------------------------------------------------------------------------


class IdP:
    """A fake identity provider: key material + a JWKS endpoint we can mutate."""

    def __init__(self) -> None:
        self.keys: dict[str, Any] = {}
        self.jwks_hits = 0
        self.discovery_hits = 0
        self.down = False
        self.add_key("k1")

    def add_key(self, kid: str) -> Any:
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.keys[kid] = private
        return private

    def jwks(self) -> dict[str, Any]:
        out = []
        for kid, private in self.keys.items():
            jwk = json.loads(RSAAlgorithm.to_jwk(private.public_key()))
            jwk.update(kid=kid, use="sig", alg="RS256")
            out.append(jwk)
        return {"keys": out}

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            return httpx.Response(503)
        if request.url.path.endswith("openid-configuration"):
            self.discovery_hits += 1
            return httpx.Response(200, json={"jwks_uri": "https://idp.example.com/jwks.json"})
        self.jwks_hits += 1
        return httpx.Response(200, json=self.jwks())

    def token(self, kid: str = "k1", **claims: Any) -> str:
        now = int(time.time())
        payload = {
            "iss": ISSUER,
            "aud": AUD,
            "sub": "user_123",
            "iat": now,
            "exp": now + 300,
            **claims,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        return jwt.encode(payload, self.keys[kid], algorithm="RS256", headers={"kid": kid})


def make_auth(idp: IdP, now: list[float] | None = None, **kw: Any) -> OIDCAuthenticator:
    clock = (lambda: now[0]) if now else time.monotonic
    return OIDCAuthenticator(
        issuer=ISSUER,
        audience=AUD,
        client=httpx.AsyncClient(transport=httpx.MockTransport(idp.handler)),
        clock=clock,
        **kw,
    )


async def test_valid_token_yields_the_subject_and_discovery_is_used() -> None:
    idp = IdP()
    auth = make_auth(idp)
    principal = await auth.authenticate(idp.token())
    assert principal is not None and principal.user_id == "oidc:user_123"
    assert idp.discovery_hits == 1 and idp.jwks_hits == 1
    await auth.authenticate(idp.token())  # cached: no further IdP traffic
    assert idp.jwks_hits == 1


@pytest.mark.parametrize(
    "claims",
    [
        {"exp": int(time.time()) - 3600},  # expired
        {"aud": "someone-else"},
        {"iss": "https://evil.example.com/"},
        {"exp": None},  # no expiry at all
        {"sub": ""},
        {"sub": "anonymous"},  # reserved id
        {"nbf": int(time.time()) + 3600},  # not yet valid
    ],
)
async def test_invalid_tokens_are_rejected(claims: dict[str, Any]) -> None:
    idp = IdP()
    auth = make_auth(idp)
    assert await auth.authenticate(idp.token(**claims)) is None


async def test_algorithm_confusion_and_none_are_rejected() -> None:
    idp = IdP()
    auth = make_auth(idp)
    public_pem = (
        idp.keys["k1"]
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": AUD, "sub": "attacker", "exp": now + 300}

    # classic: sign with HMAC using the *public* key as the secret. PyJWT refuses to even
    # *create* this token, so the attacker's token is assembled by hand.
    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    head = b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": "k1"}).encode())
    body = b64(json.dumps(claims).encode())
    sig = hmac.new(public_pem, f"{head}.{body}".encode(), hashlib.sha256).digest()
    hs = f"{head}.{body}.{b64(sig)}"
    assert await auth.authenticate(hs) is None
    none_token = jwt.encode(claims, None, algorithm="none", headers={"kid": "k1"})
    assert await auth.authenticate(none_token) is None
    # a token signed by a *different* key that claims the known kid
    forged = jwt.encode(
        claims,
        rsa.generate_private_key(public_exponent=65537, key_size=2048),
        algorithm="RS256",
        headers={"kid": "k1"},
    )
    assert await auth.authenticate(forged) is None
    with pytest.raises(ValueError):
        OIDCAuthenticator(issuer=ISSUER, audience=AUD, algorithms=("HS256",))


async def test_key_rotation_refreshes_once_and_garbage_cannot_hammer_the_idp() -> None:
    idp = IdP()
    now = [1000.0]
    auth = make_auth(idp, now, min_refresh_s=30)
    assert await auth.authenticate(idp.token()) is not None
    assert idp.jwks_hits == 1

    idp.add_key("k2")  # the IdP rotates
    token2 = idp.token("k2")
    now[0] += 31  # past the refresh rate limit
    assert (await auth.authenticate(token2)) is not None
    assert idp.jwks_hits == 2

    # a flood of tokens with unknown kids triggers at most one refresh per window
    idp.add_key("k3")
    junk = idp.token("k3")
    idp.keys.pop("k3")
    hits_before = idp.jwks_hits
    for _ in range(20):
        assert await auth.authenticate(junk) is None
    assert idp.jwks_hits - hits_before <= 1


async def test_idp_outage_keeps_serving_with_cached_keys() -> None:
    idp = IdP()
    now = [0.0]
    auth = make_auth(idp, now, jwks_ttl_s=60, min_refresh_s=5)
    assert await auth.authenticate(idp.token()) is not None
    idp.down = True
    now[0] += 120  # cache is stale and the refresh fails
    assert await auth.authenticate(idp.token()) is not None


async def test_chain_routes_jwts_to_oidc_and_other_tokens_to_api_keys() -> None:
    idp = IdP()
    chain = ChainAuthenticator(
        make_auth(idp), ApiKeyAuthenticator({"svc": "service-key-0123456789"})
    )
    assert (await chain.authenticate(idp.token())).user_id == "oidc:user_123"  # type: ignore[union-attr]
    assert (await chain.authenticate("service-key-0123456789")).user_id == "svc"  # type: ignore[union-attr]
    assert await chain.authenticate("garbage") is None
    assert await chain.authenticate(None) is None
    assert chain.required
    open_chain = ChainAuthenticator(None, ApiKeyAuthenticator({}))
    assert (await open_chain.authenticate(None)).user_id == "anonymous"  # type: ignore[union-attr]


async def test_api_accepts_oidc_tokens_and_scopes_data_to_the_subject(
    settings: Settings,
) -> None:
    idp = IdP()
    llm = FakeLLM(
        [
            Script(
                tool_calls=[
                    ToolCall(
                        "w",
                        "create_ticket",
                        {
                            "title": "Stuck payout",
                            "description": "In transit 6 days.",
                            "severity": "normal",
                        },
                    )
                ]
            ),
            Script(text="Opened."),
        ]
    )
    app, client, gen = await open_client(settings, llm)
    app.state.authenticator = ChainAuthenticator(make_auth(idp), None)
    body = {"messages": [{"role": "user", "content": "open a ticket for my payout"}]}
    assert (await client.post("/v1/chat", json=body)).status_code == 401
    bearer = {"authorization": f"Bearer {idp.token(sub='user_abc')}"}
    res = await client.post("/v1/chat", json=body, headers=bearer)
    pid = parse_sse(res.text)[0][1]["pending_id"]
    other = {"authorization": f"Bearer {idp.token(sub='user_zzz')}"}
    assert (
        await client.post(
            "/v1/chat/confirm", json={"pending_id": pid, "approve": True}, headers=other
        )
    ).status_code == 404
    ok = await client.post(
        "/v1/chat/confirm", json={"pending_id": pid, "approve": True}, headers=bearer
    )
    assert ok.status_code == 200
    (ticket,) = app.state.tickets._by_id.values()
    assert ticket.user_id == "oidc:user_abc"
    await close(gen)


def test_oidc_settings_must_be_complete_and_https_when_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import ValidationError

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    with pytest.raises(ValidationError, match="go together"):
        Settings(oidc_issuer="https://idp.example.com", _env_file=None)  # type: ignore[call-arg]
    base: dict[str, Any] = {
        "env": "prod",
        "database_url": "postgresql://x/y",
        "_env_file": None,
    }
    with pytest.raises(ValidationError, match="authentication"):
        Settings(**base)
    with pytest.raises(ValidationError, match="https"):
        Settings(**base, oidc_issuer="http://idp.local", oidc_audience="a")
    ok = Settings(**base, oidc_issuer="https://idp.example.com/", oidc_audience="api")
    assert ok.oidc_issuer


# ---- multi-agent -----------------------------------------------------------------------


class StubClassifier:
    name = "stub"

    def __init__(self, intent: str = "question", fail: bool = False) -> None:
        self.intent, self.fail = intent, fail

    async def classify(self, message: str) -> Any:
        if self.fail:
            raise RuntimeError("classifier down")
        return self.intent


async def build_multi(llm: FakeLLM, classifier: StubClassifier) -> Orchestrator:
    store, emb = InMemoryChunkStore(), HashEmbedder()
    await ingest_corpus(ROOT / "corpus", store, emb)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    reg = PromptRegistry(ROOT / "prompts")
    return build_orchestrator(
        settings,
        llm=llm,
        retriever=Retriever(store, emb, NoopReranker()),
        pending=InMemoryPendingStore(),
        docs_prompt=reg.get("agent_docs"),
        tickets_prompt=reg.get("agent_tickets"),
        classifier=classifier,
    )


def spec_names(agent: Any) -> set[str]:
    return {s["name"] for s in agent._registry.specs()}


async def test_specialists_have_disjoint_privileges() -> None:
    orch = await build_multi(FakeLLM(), StubClassifier())
    docs, tickets = orch._agents["docs"], orch._agents["tickets"]
    assert spec_names(docs) == {"search_docs"}  # reads untrusted text, cannot write
    assert spec_names(tickets) == {
        "create_ticket",
        "get_ticket",
        "hand_off_to_docs",
    }  # no retrieval
    assert tickets._config.model == "claude-sonnet-5-5" and docs._config.model == "claude-opus-5-5"


def test_ticket_followup_heuristic() -> None:
    offer = [
        {"role": "user", "content": "my payout is stuck"},
        {"role": "assistant", "content": "I couldn't find it in the docs. Shall I open a ticket?"},
        {"role": "user", "content": "yes please"},
    ]
    assert is_ticket_followup(offer)
    assert not is_ticket_followup([*offer[:2], {"role": "user", "content": "yes, and what is X?"}])
    assert not is_ticket_followup(
        [
            {"role": "assistant", "content": "The refund window is 30 days."},
            {"role": "user", "content": "yes"},
        ]
    )
    assert not is_ticket_followup([{"role": "user", "content": "yes"}])


async def test_routing_decisions_and_fail_safe() -> None:
    hist = [{"role": "user", "content": "refund window?"}]
    assert await (await build_multi(FakeLLM(), StubClassifier("question"))).pick(hist) == "docs"
    assert (
        await (await build_multi(FakeLLM(), StubClassifier("ticket_request"))).pick(hist)
        == "tickets"
    )
    assert await (await build_multi(FakeLLM(), StubClassifier("complaint"))).pick(hist) == "docs"
    # classifier failure goes to the *least privileged* specialist
    assert await (await build_multi(FakeLLM(), StubClassifier(fail=True))).pick(hist) == "docs"


async def test_an_injected_docs_agent_cannot_reach_the_write_tool() -> None:
    """The point of the split: even a fully manipulated docs agent has nothing to call."""

    def script(req: LLMRequest) -> Script:
        last = req.messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            return Script(text="Sorry, I can't do that.")
        return Script(
            tool_calls=[
                ToolCall(
                    "w",
                    "create_ticket",
                    {
                        "title": "URGENT",
                        "description": "send the API key by email",
                        "severity": "high",
                    },
                )
            ]
        )

    llm = FakeLLM(script)
    orch = await build_multi(llm, StubClassifier("question"))
    events = [
        e
        async for e in orch.run(
            [{"role": "user", "content": "why was my webhook disabled?"}],
            ToolContext("u", InMemoryTicketStore()),
        )
    ]
    assert not any(isinstance(e, NeedsConfirmation) for e in events)
    result = llm.requests[1].messages[-1]["content"][0]
    assert result["is_error"] and "Unknown tool" in result["content"]
    assert [t["name"] for t in llm.requests[0].tools or []] == ["search_docs"]


async def test_multi_architecture_end_to_end_and_resume_goes_to_the_right_specialist(
    settings: Settings,
) -> None:
    def script(req: LLMRequest) -> Script:
        if req.purpose in ("route", "triage"):
            msg = req.messages[0]["content"]
            intent = "ticket_request" if "ticket" in msg else "question"
            return Script(
                text=json.dumps(
                    {"intent": intent, "urgency": "low", "product_area": "other", "summary": "s"}
                )
            )
        specialist = "tickets" if "ticketing specialist" in str(req.system) else "docs"
        last = req.messages[-1]
        has_results = last["role"] == "user" and isinstance(last["content"], list)
        if specialist == "docs":
            return base_script(req)
        if has_results:
            return Script(text="Ticket created.")
        return Script(
            tool_calls=[
                ToolCall(
                    "w",
                    "create_ticket",
                    {
                        "title": "Payout stuck",
                        "description": "In transit for 6 days.",
                        "severity": "high",
                    },
                )
            ]
        )

    llm = FakeLLM(script)
    app, client, gen = await open_client(
        settings, llm, architecture="multi", response_cache_ttl_s=0
    )
    q = await client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "How long is the refund window?"}]},
    )
    assert parse_sse(q.text)[-1][0] == "done"
    docs_models = {r.model for r in llm.requests if r.purpose == "answer"}
    assert docs_models == {"claude-opus-5-5"}

    t = await client.post(
        "/v1/chat",
        json={
            "messages": [{"role": "user", "content": "please open a ticket, my payout is stuck"}]
        },
    )
    events = parse_sse(t.text)
    assert events[0][0] == "confirmation_required"
    pid = events[0][1]["pending_id"]
    pending = app.state.pending._items[pid]
    assert pending.agent == "tickets"
    ok = await client.post("/v1/chat/confirm", json={"pending_id": pid, "approve": True})
    assert parse_sse(ok.text)[-1][1]["model"] == "claude-sonnet-5-5"
    assert len(app.state.tickets._by_id) == 1
    await close(gen)


# ---- review round -----------------------------------------------------------------------


async def test_non_rsa_keys_and_forged_headers_cannot_cause_a_500() -> None:
    from cryptography.hazmat.primitives.asymmetric import ec

    idp = IdP()
    ec_key = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(ec_key.public_key()))
    jwk.update(kid="ec1", use="sig", alg="ES256")
    base = idp.jwks

    def jwks_with_ec() -> dict[str, Any]:
        data = base()
        data["keys"].append(jwk)
        return data

    idp.jwks = jwks_with_ec  # type: ignore[method-assign]
    auth = make_auth(idp)
    assert await auth.authenticate(idp.token()) is not None
    # attacker claims RS256 but points at the EC key's kid
    forged = jwt.encode(
        {"iss": ISSUER, "aud": AUD, "sub": "x", "exp": int(time.time()) + 60},
        idp.keys["k1"],
        algorithm="RS256",
        headers={"kid": "ec1"},
    )
    assert await auth.authenticate(forged) is None  # filtered by key type: not a 500


async def test_subject_validation_and_namespacing() -> None:
    idp = IdP()
    auth = make_auth(idp)
    for bad in ("anonymous ", " alice", "a\nb", "x" * 201, "*"):
        assert await auth.authenticate(idp.token(sub=bad)) is None, repr(bad)
    ok = await auth.authenticate(idp.token(sub="alice"))
    assert ok is not None and ok.user_id == "oidc:alice"  # cannot collide with API-key "alice"


async def test_idp_outage_is_unavailable_not_unauthenticated_and_does_not_pile_up() -> None:
    import asyncio

    from docspilot.auth import AuthUnavailableError

    idp = IdP()
    idp.down = True
    hits = {"n": 0}

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        await asyncio.sleep(0.15)
        return httpx.Response(503)

    auth = OIDCAuthenticator(
        issuer=ISSUER,
        audience=AUD,
        jwks_url="https://idp.example.com/jwks.json",
        client=httpx.AsyncClient(transport=httpx.MockTransport(slow_handler)),
    )
    token = IdP().token()  # signature irrelevant: we never get keys
    results = await asyncio.gather(
        *(auth.authenticate(token) for _ in range(6)), return_exceptions=True
    )
    assert all(isinstance(r, AuthUnavailableError) for r in results)
    assert hits["n"] <= 2  # waiters reuse the failed attempt instead of repeating it


async def test_api_returns_503_when_keys_are_unavailable(settings: Settings) -> None:
    app, client, gen = await open_client(settings, FakeLLM())
    dead = OIDCAuthenticator(
        issuer=ISSUER,
        audience=AUD,
        jwks_url="https://idp.example.com/jwks.json",
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))),
    )
    app.state.authenticator = ChainAuthenticator(dead, None)
    token = IdP().token()
    res = await client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"authorization": f"Bearer {token}"},
    )
    assert res.status_code == 503 and res.headers["retry-after"] == "5"
    await close(gen)


async def test_strict_environment_refuses_to_start_without_working_auth(
    settings: Settings,
) -> None:
    from pydantic import SecretStr

    from docspilot.api.app import create_app

    cfg = settings.model_copy(update={"env": "staging", "api_keys": SecretStr(",")})
    app = create_app(cfg, FakeLLM())
    with pytest.raises(RuntimeError, match="authentication is required"):
        async with app.router.lifespan_context(app):
            pass


# ---- orchestrator: sticky routing, handoff, no double classification ---------------------


def test_followup_rules_are_conservative() -> None:
    def hist(*pairs: tuple[str, str]) -> list[dict[str, str]]:
        return [{"role": r, "content": c} for r, c in pairs]

    clarify = hist(
        ("user", "please open a ticket"),
        ("assistant", "Sure. Which invoice is affected and what was the error?"),
    )
    # the details answering a ticketing clarification stay with the ticketing specialist
    assert is_ticket_followup(clarify + hist(("user", "INV-0192, charged twice on Oct 3")))
    # ... but a new question is classified normally
    assert not is_ticket_followup(
        hist(
            ("assistant", "Shall I open a ticket for this?"),
            ("user", "what is the refund window?"),
        )
    )
    # a bare "ok" after an answer that merely mentions tickets is not a ticket flow
    assert not is_ticket_followup(
        hist(("assistant", "See the ticket policy page for details."), ("user", "ok"))
    )
    assert is_ticket_followup(hist(("assistant", "Want me to open a ticket?"), ("user", "ok")))


async def test_misrouted_product_question_is_handed_back_to_docs() -> None:
    def script(req: LLMRequest) -> Script:
        if "ticketing specialist" in str(req.system):
            return Script(text="", tool_calls=[ToolCall("h", "hand_off_to_docs", {})])
        last = req.messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            return Script(text="30 days [policies/refunds.md]")
        return Script(tool_calls=[ToolCall("s", "search_docs", {"query": "refund window days"})])

    llm = FakeLLM(script)
    orch = await build_multi(llm, StubClassifier("ticket_request"))  # classifier got it wrong
    events = [
        e
        async for e in orch.run(
            [{"role": "user", "content": "how long is the refund window"}],
            ToolContext("u", InMemoryTicketStore()),
        )
    ]
    final = events[-1]
    assert type(final).__name__ == "AgentFinished" and final.text.startswith("30 days")
    assert "hand_off_to_docs" in spec_names(orch._agents["tickets"])
    assert "hand_off_to_docs" not in spec_names(orch._agents["docs"])  # no ping-pong


async def test_multi_architecture_classifies_once_even_with_routing_enabled(
    settings: Settings,
) -> None:
    calls = {"n": 0}

    def script(req: LLMRequest) -> Script:
        if req.purpose in ("route", "triage"):
            calls["n"] += 1
            return Script(
                text=json.dumps(
                    {
                        "intent": "question",
                        "urgency": "low",
                        "product_area": "other",
                        "summary": "s",
                    }
                )
            )
        return base_script(req)

    _, client, gen = await open_client(
        settings, FakeLLM(script), architecture="multi", routing=True, response_cache_ttl_s=0
    )
    await client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "refund window?"}]}
    )
    await close(gen)
    assert calls["n"] == 1
