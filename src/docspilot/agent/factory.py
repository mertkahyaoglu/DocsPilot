"""Build the agent the same way for the API and for offline evals."""

from __future__ import annotations

from docspilot.agent.loop import Agent, AgentConfig
from docspilot.agent.orchestrator import Orchestrator
from docspilot.agent.pending import PendingStore
from docspilot.agent.tools import Tool, ToolRegistry, default_tools
from docspilot.config import Settings
from docspilot.intents import IntentClassifier
from docspilot.llm.types import LLM
from docspilot.prompts import Prompt
from docspilot.rag.retrieve import Retriever
from docspilot.telemetry import Telemetry


def build_agent(
    settings: Settings,
    *,
    llm: LLM,
    retriever: Retriever,
    pending: PendingStore,
    answer_prompt: Prompt,
    telemetry: Telemetry | None = None,
    extra_tools: list[Tool] | None = None,
) -> Agent:
    return Agent(
        llm,
        ToolRegistry(
            [
                *default_tools(
                    retriever, injection_policy=settings.injection_policy, telemetry=telemetry
                ),
                *(extra_tools or []),
            ],
            telemetry=telemetry,
        ),
        pending,
        AgentConfig(
            model=settings.model_answer,
            system=answer_prompt.text,
            max_tokens=settings.max_tokens,
            effort=settings.effort_answer,
            max_steps=settings.max_agent_steps,
            tool_timeout_s=settings.tool_timeout_s,
            prompt_caching=settings.prompt_caching,
            prompt_ref=answer_prompt.ref,
            prompt_sha=answer_prompt.sha,
        ),
        confirmation_ttl_s=settings.confirmation_ttl_s,
    )


def build_orchestrator(
    settings: Settings,
    *,
    llm: LLM,
    retriever: Retriever,
    pending: PendingStore,
    docs_prompt: Prompt,
    tickets_prompt: Prompt,
    classifier: IntentClassifier,
    telemetry: Telemetry | None = None,
) -> Orchestrator:
    """Docs specialist: reads (untrusted) documents, cannot write. Tickets specialist:
    can act (human-gated), never sees documents."""
    all_tools = {
        t.name: t
        for t in default_tools(
            retriever, injection_policy=settings.injection_policy, telemetry=telemetry
        )
    }

    def make(name: str, prompt: Prompt, tool_names: list[str], model: str) -> Agent:
        return Agent(
            llm,
            ToolRegistry([all_tools[n] for n in tool_names], telemetry=telemetry),
            pending,
            AgentConfig(
                model=model,
                system=prompt.text,
                max_tokens=settings.max_tokens,
                effort=settings.effort_answer,
                max_steps=settings.max_agent_steps,
                tool_timeout_s=settings.tool_timeout_s,
                prompt_caching=settings.prompt_caching,
                prompt_ref=prompt.ref,
                prompt_sha=prompt.sha,
                name=name,
            ),
            confirmation_ttl_s=settings.confirmation_ttl_s,
        )

    return Orchestrator(
        classifier=classifier,
        specialists={
            "docs": make("docs", docs_prompt, ["search_docs"], settings.model_answer),
            "tickets": make(
                "tickets", tickets_prompt, ["create_ticket", "get_ticket"], settings.model_tickets
            ),
        },
    )
