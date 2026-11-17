"""Finite session facts for the original parent workspace display builders."""

from __future__ import annotations

from dataclasses import dataclass

from .console_chat_models import ConsoleRunMarker
from .console_execution_projection_values import (
    _boolean,
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


@dataclass(frozen=True, slots=True)
class ConsoleWorkspaceSurface:
    """Captured store workspace scope and facts for the published session rows."""

    active_workspace_id: str | None
    controller_present: bool
    sessions: tuple[ConsoleWorkspaceSessionFacts, ...]


def _marker(value: object, decode: bool):
    return _enum_record(value, ConsoleRunMarker, decode)


_SESSION_FACTS = {
    "session_id": _text,
    "updated_at": _text,
    "character_id": _optional(_integer),
    "character_name": _optional(_text),
    "handoff_label": _optional(_text),
    "queued_count": _integer,
    "run_marker": _marker,
}


def _session_facts(value: object, decode: bool):
    return _record(value, ConsoleWorkspaceSessionFacts, _SESSION_FACTS, decode)


_SURFACE = {
    "active_workspace_id": _optional(_text),
    "controller_present": _boolean,
    "sessions": _tuple(_session_facts),
}


def encode_workspace_surface(surface: ConsoleWorkspaceSurface) -> dict:
    """Detach only declared display facts; session-row alignment is owner-checked."""
    return _record(surface, ConsoleWorkspaceSurface, _SURFACE, False)


def decode_workspace_surface(payload: dict) -> ConsoleWorkspaceSurface:
    """Restore original marker enum and frozen facts from an exact finite shape."""
    return _record(payload, ConsoleWorkspaceSurface, _SURFACE, True)
