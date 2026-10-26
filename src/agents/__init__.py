"""Hermes Multi-Agent package (Bull, Bear, Judge agents)."""

from src.agents.orchestrator import (
    run_multi_agent_analysis,
    run_multi_agent_analysis_async,
)
from src.agents.prompts import (
    BEAR_AGENT_PROMPT,
    BULL_AGENT_PROMPT,
    FUNDAMENTAL_AGENT_PROMPT,
    JUDGE_AGENT_PROMPT,
)

__all__ = [
    "FUNDAMENTAL_AGENT_PROMPT",
    "BULL_AGENT_PROMPT",
    "BEAR_AGENT_PROMPT",
    "JUDGE_AGENT_PROMPT",
    "run_multi_agent_analysis",
    "run_multi_agent_analysis_async",
]
