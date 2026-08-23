"""Detached configuration captured for one owning-session Console turn.

The prompt queue must be able to validate and dispatch a background turn while a
different session is viewed.  This module intentionally contains configuration
only: credentials, permission grants, trust decisions, cancellation signals,
streams, and other live authority stay behind their existing runtime seams.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleProviderSelection,
    ConsoleStagedSource,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSnapshot


def _freeze(value: Any) -> Any:
    """Return a recursively detached, immutable configuration value."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {deepcopy(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return deepcopy(value)


def _detached_selection(
    selection: ConsoleProviderSelection,
) -> ConsoleProviderSelection:
    """Copy a provider selection without retaining live workspace objects."""
    source = selection.workspace_context
    workspace_context = ConsoleWorkspaceContext(
        active_workspace_id=str(source.active_workspace_id),
        staged_sources=tuple(
            ConsoleStagedSource(
                source_id=str(item.source_id),
                label=str(item.label),
                source_type=str(item.source_type),
                workspace_id=(
                    str(item.workspace_id) if item.workspace_id is not None else None
                ),
            )
            for item in source.staged_sources
        ),
        active_run_id=(
            str(source.active_run_id) if source.active_run_id is not None else None
        ),
        handoff_id=str(source.handoff_id) if source.handoff_id is not None else None,
    )
    return ConsoleProviderSelection(
        provider=str(selection.provider),
        base_url=selection.base_url,
        explicit_model=selection.explicit_model,
        configured_model=selection.configured_model,
        temperature=selection.temperature,
        top_p=selection.top_p,
        min_p=selection.min_p,
        top_k=selection.top_k,
        max_tokens=selection.max_tokens,
        seed=selection.seed,
        presence_penalty=selection.presence_penalty,
        frequency_penalty=selection.frequency_penalty,
        reasoning_effort=selection.reasoning_effort,
        reasoning_summary=selection.reasoning_summary,
        verbosity=selection.verbosity,
        thinking_effort=selection.thinking_effort,
        thinking_budget_tokens=selection.thinking_budget_tokens,
        streaming=selection.streaming,
        system_prompt=selection.system_prompt,
        workspace_context=workspace_context,
    )


@dataclass(frozen=True, slots=True)
class ConsoleTurnExecutionContext:
    """Immutable provider-input configuration for one Console turn."""

    session_id: str
    provider_selection: ConsoleProviderSelection
    scratch_space: ConsoleScratchSnapshot | None = None
    session_settings: ConsoleSessionSettings | None = None
    workspace_roots: tuple[str, ...] = ()
    capabilities: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    rag_defaults: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    tool_configuration: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    provider_payload_settings: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Detach constructor inputs even when callers bypass ``capture``."""
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(
            self,
            "provider_selection",
            _detached_selection(self.provider_selection),
        )
        object.__setattr__(
            self,
            "session_settings",
            deepcopy(self.session_settings),
        )
        object.__setattr__(
            self,
            "workspace_roots",
            tuple(str(root) for root in deepcopy(self.workspace_roots)),
        )
        for field_name in (
            "capabilities",
            "rag_defaults",
            "tool_configuration",
            "provider_payload_settings",
        ):
            object.__setattr__(self, field_name, _freeze(getattr(self, field_name)))

    @classmethod
    def capture(
        cls,
        *,
        session_id: str,
        provider_selection: ConsoleProviderSelection,
        scratch_space: ConsoleScratchSnapshot | None = None,
        session_settings: ConsoleSessionSettings | None = None,
        workspace_roots: Sequence[object] = (),
        capabilities: Mapping[str, Any] | None = None,
        rag_defaults: Mapping[str, Any] | None = None,
        tool_configuration: Mapping[str, Any] | None = None,
        provider_payload_settings: Mapping[str, Any] | None = None,
    ) -> "ConsoleTurnExecutionContext":
        """Capture detached values from mutable application-owned sources."""
        return cls(
            session_id=str(session_id),
            provider_selection=provider_selection,
            scratch_space=scratch_space,
            session_settings=session_settings,
            workspace_roots=tuple(workspace_roots),
            capabilities=capabilities or {},
            rag_defaults=rag_defaults or {},
            tool_configuration=tool_configuration or {},
            provider_payload_settings=provider_payload_settings or {},
        )

    @property
    def effective_model(self) -> str | None:
        """Return the explicit model or its captured configured fallback."""
        return (
            self.provider_selection.explicit_model
            or self.provider_selection.configured_model
        )
