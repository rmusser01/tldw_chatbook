"""Finite session facts for the original parent workspace display builders."""

from __future__ import annotations

from dataclasses import dataclass

from ..Workspaces.conversation_attention import ConversationAttentionFact
from .console_chat_models import ConsoleRunMarker
from .console_execution_projection_values import (
    _boolean,
    _count,
    _enum,
    _enum_record,
    _integer,
    _optional,
    _record,
    _text,
    _tuple,
)


@dataclass(frozen=True, slots=True)
class ConsoleWorkspaceSessionFacts:
    """Actual per-session display facts; no session/controller authority."""

    session_id: str
    updated_at: str
    character_id: int | None
    character_name: str | None
    handoff_label: str | None
    queued_count: int
    run_marker: ConsoleRunMarker
    attention: tuple[ConversationAttentionFact, ...] = ()


@dataclass(frozen=True, slots=True)
class ConsoleWorkspaceCount:
    owner_id: str
    count: int


@dataclass(frozen=True, slots=True)
class ConsoleWorkspaceReceipt:
    conversation_id: str | None
    session_id: str | None
    status: str


@dataclass(frozen=True, slots=True)
class ConsoleWorkspaceSurface:
    """Captured store workspace scope and facts for the published session rows."""

    active_workspace_id: str | None
    controller_present: bool
    sessions: tuple[ConsoleWorkspaceSessionFacts, ...]
    progress_counts: tuple[ConsoleWorkspaceCount, ...] = ()
    subagent_counts: tuple[ConsoleWorkspaceCount, ...] = ()
    subagent_count_ids: tuple[str, ...] = ()
    unseen_receipts: tuple[ConsoleWorkspaceReceipt, ...] = ()


def _marker(value: object, decode: bool):
    return _enum_record(value, ConsoleRunMarker, decode)


def _attention(value: object, decode: bool):
    return _record(value, ConversationAttentionFact, {
        "kind": _enum("approval", "blocked", "failed", "running", "paused",
                      "stopped", "unread", "ready", "outcome_unknown"),
        "label": _text,
    }, decode)


def _workspace_count(value: object, decode: bool):
    return _record(value, ConsoleWorkspaceCount, {
        "owner_id": _text, "count": _count,
    }, decode)


def _receipt(value: object, decode: bool):
    return _record(value, ConsoleWorkspaceReceipt, {
        "conversation_id": _optional(_text), "session_id": _optional(_text),
        "status": _text,
    }, decode)


_SESSION_FACTS = {
    "session_id": _text,
    "updated_at": _text,
    "character_id": _optional(_integer),
    "character_name": _optional(_text),
    "handoff_label": _optional(_text),
    "queued_count": _integer,
    "run_marker": _marker,
    "attention": _tuple(_attention),
}


def _session_facts(value: object, decode: bool):
    return _record(value, ConsoleWorkspaceSessionFacts, _SESSION_FACTS, decode)


_SURFACE = {
    "active_workspace_id": _optional(_text),
    "controller_present": _boolean,
    "sessions": _tuple(_session_facts),
    "progress_counts": _tuple(_workspace_count),
    "subagent_counts": _tuple(_workspace_count),
    "subagent_count_ids": _tuple(_text),
    "unseen_receipts": _tuple(_receipt),
}


def encode_workspace_surface(surface: ConsoleWorkspaceSurface) -> dict:
    """Detach only declared display facts; session-row alignment is owner-checked."""
    return _record(surface, ConsoleWorkspaceSurface, _SURFACE, False)


def decode_workspace_surface(payload: dict) -> ConsoleWorkspaceSurface:
    """Restore original marker enum and frozen facts from an exact finite shape."""
    return _record(payload, ConsoleWorkspaceSurface, _SURFACE, True)
