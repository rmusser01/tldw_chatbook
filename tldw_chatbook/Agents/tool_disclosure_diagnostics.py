"""Bounded ordinary-log representation of agent terminal steps."""

import json
from collections.abc import Iterable

from loguru import logger

from .agent_models import AGENT_KIND_PRIMARY, AgentStep


def log_agent_steps(steps: Iterable[AgentStep], run_id: str) -> None:
    """Log step metadata without copying arguments or result bodies."""
    for step in steps:
        logger.info(
            "agent run step kind={} tool={:.128} outcome={} run={}",
            step.kind,
            json.dumps(step.tool_name),
            step.tool_outcome,
            run_id,
            agent_kind=AGENT_KIND_PRIMARY,
            step_kind=step.kind,
            tool_name=step.tool_name,
            summary=step.summary,
            step_index=step.index,
        )
