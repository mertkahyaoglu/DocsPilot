"""Build the agent the same way for the API and for offline evals."""

from __future__ import annotations

from docspilot.agent.loop import Agent, AgentConfig
from docspilot.agent.pending import PendingStore
from docspilot.agent.tools import ToolRegistry, default_tools
from docspilot.config import Settings
from docspilot.llm.types import LLM
from docspilot.prompts import Prompt
from docspilot.rag.retrieve import Retriever


def build_agent(
    settings: Settings,
    *,
    llm: LLM,
    retriever: Retriever,
    pending: PendingStore,
    answer_prompt: Prompt,
) -> Agent:
    return Agent(
        llm,
        ToolRegistry(default_tools(retriever)),
        pending,
        AgentConfig(
            model=settings.model_answer,
            system=answer_prompt.text,
            max_tokens=settings.max_tokens,
            effort=settings.effort_answer,
            max_steps=settings.max_agent_steps,
            tool_timeout_s=settings.tool_timeout_s,
            prompt_ref=answer_prompt.ref,
            prompt_sha=answer_prompt.sha,
        ),
        confirmation_ttl_s=settings.confirmation_ttl_s,
    )
