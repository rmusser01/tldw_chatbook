"""Immutable, normalized v2 hook protocol values.

Construction is owned by validation.py; callers should use its parsing entry points.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, model_serializer


def _plain(value: Any) -> Any:
    """Return stable JSON-ready copies of immutable protocol values."""
    if isinstance(value, _FrozenModel):
        return {key: _plain(getattr(value, key)) for key in type(value).model_fields}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, frozenset):
        return sorted(_plain(item) for item in value)
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return _plain(self)


class HookHandler(_FrozenModel):
    id: str
    event: str
    type: str
    effects: frozenset[str]
    required: bool = False
    require_context: bool = False
    match: Mapping[str, tuple[str, ...]] | None = None
    timeout_seconds: float = 10.0
    argv: tuple[str, ...] | None = None
    env: Mapping[str, str | Mapping[str, str]] | None = None
    cwd: str | None = None
    server: str | None = None
    tool: str | None = None
    input: Mapping[str, Any] | None = None


class HookEvent(_FrozenModel):
    protocol_version: int
    event_id: str
    event: str
    timestamp: str
    runtime_session_id: str
    initiator: str
    origin: str
    causal_chain_id: str
    causal_depth: int
    data: Mapping[str, Any]
    run_id: str | None = None
    parent_run_id: str | None = None
    turn_id: str | None = None
    workspace_id: str | None = None
    owner_installation_id: str | None = None
    owner_component_id: str | None = None


class ContextBlock(_FrozenModel):
    text: str
    lifetime: str


class HookResult(_FrozenModel):
    version: int = 2
    decision: str = "pass"
    context: tuple[ContextBlock, ...] = ()
    updated_input: Mapping[str, Any] | None = None
    child_limits: Mapping[str, Any] | None = None
    continuation: Mapping[str, str] | None = None
    stop_continuations: bool | None = None
    reason: str | None = None
