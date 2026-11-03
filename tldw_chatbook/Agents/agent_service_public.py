"""Public names for the ``agent_service`` helpers that Console (Chat) calls.

Chat used to import these under their underscore names (TASK-36944.2).
``agent_service.py`` is size-pinned, so the public spellings live here instead
of as renames in place. The private names stay the implementation (and the
test patch targets): the plain functions are the same objects, and
``setting`` looks ``agent_service._setting`` up on every call so a pin on it
still reaches Console.
"""

from __future__ import annotations

from tldw_chatbook.Agents.agent_service import (
    _budget_weighted_tokens as budget_weighted_tokens,
    _coerce_autowake_enabled as coerce_autowake_enabled,
    _coerce_max_live_subagents as coerce_max_live_subagents,
    _coerce_retained_transcript_max_chars as coerce_retained_transcript_max_chars,
    _coerce_retained_transcripts as coerce_retained_transcripts,
    _count_model_messages as count_model_messages,
    _spawn_override_targets as spawn_override_targets,
)


def setting(key: str, default):
    """Read one ``[agents]`` key through ``agent_service._setting``."""
    from tldw_chatbook.Agents import agent_service

    return agent_service._setting(key, default)
