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
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicySnapshot
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSnapshot
from tldw_chatbook.Workspaces.change_review_consent import SkippedReviewRoot


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
class ConsoleTurnConfigurationSnapshot:
    """Immutable provider-input configuration captured before gateway resolution."""

    session_id: str
    provider_selection: ConsoleProviderSelection
    scratch_space: ConsoleScratchSnapshot | None = None
    session_settings: ConsoleSessionSettings | None = None
    workspace_roots: tuple[str, ...] = ()
    change_review_root_aliases: tuple[str, ...] = ()
    change_review_skipped_roots: tuple[SkippedReviewRoot, ...] = ()
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
        object.__setattr__(
            self,
            "change_review_root_aliases",
            tuple(str(alias) for alias in deepcopy(self.change_review_root_aliases)),
        )
        object.__setattr__(
            self,
            "change_review_skipped_roots",
            tuple(deepcopy(self.change_review_skipped_roots)),
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
        change_review_root_aliases: Sequence[str] = (),
        change_review_skipped_roots: Sequence[SkippedReviewRoot] = (),
        capabilities: Mapping[str, Any] | None = None,
        rag_defaults: Mapping[str, Any] | None = None,
        tool_configuration: Mapping[str, Any] | None = None,
        provider_payload_settings: Mapping[str, Any] | None = None,
    ) -> "ConsoleTurnConfigurationSnapshot":
        """Capture detached values from mutable application-owned sources."""
        return cls(
            session_id=str(session_id),
            provider_selection=provider_selection,
            scratch_space=scratch_space,
            session_settings=session_settings,
            workspace_roots=tuple(workspace_roots),
            change_review_root_aliases=tuple(change_review_root_aliases),
            change_review_skipped_roots=tuple(change_review_skipped_roots),
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


def _detached_configuration(
    configuration: ConsoleTurnConfigurationSnapshot,
) -> ConsoleTurnConfigurationSnapshot:
    """Copy an already-frozen configuration at the final-context boundary."""
    return ConsoleTurnConfigurationSnapshot(
        session_id=configuration.session_id,
        provider_selection=configuration.provider_selection,
        scratch_space=configuration.scratch_space,
        session_settings=configuration.session_settings,
        workspace_roots=configuration.workspace_roots,
        change_review_root_aliases=configuration.change_review_root_aliases,
        change_review_skipped_roots=configuration.change_review_skipped_roots,
        capabilities=configuration.capabilities,
        rag_defaults=configuration.rag_defaults,
        tool_configuration=configuration.tool_configuration,
        provider_payload_settings=configuration.provider_payload_settings,
    )


def _detached_authority(
    authority: ConsoleTurnLibraryAuthority,
) -> ConsoleTurnLibraryAuthority:
    """Copy the complete Library authority without retaining caller containers."""
    policy = authority.policy
    return ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=policy.auto_retrieve,
            assistant_access=policy.assistant_access,
            policy_revision=policy.policy_revision,
            source=policy.source,
            error_code=policy.error_code,
        ),
        direct_library_tools=bool(authority.direct_library_tools),
        source_types=tuple(str(value) for value in authority.source_types),
        scope_snapshot=ConsoleLibraryItemScopeSnapshot(
            note_ids=tuple(str(value) for value in authority.scope_snapshot.note_ids),
            media_ids=tuple(str(value) for value in authority.scope_snapshot.media_ids),
            conversations_allowed=bool(authority.scope_snapshot.conversations_allowed),
        ),
        provider_intent=ConsoleProviderIntent(
            provider=str(authority.provider_intent.provider),
            model=(
                str(authority.provider_intent.model)
                if authority.provider_intent.model is not None
                else None
            ),
            endpoint=(
                str(authority.provider_intent.endpoint)
                if authority.provider_intent.endpoint is not None
                else None
            ),
        ),
        attempt_id=str(authority.attempt_id),
    )


def _detached_destination(
    destination: ConsoleResolvedDestination,
) -> ConsoleResolvedDestination:
    """Copy the credential-free gateway result at the final-context boundary."""
    return ConsoleResolvedDestination(
        provider=str(destination.provider),
        model=str(destination.model) if destination.model is not None else None,
        endpoint_identity=str(destination.endpoint_identity),
        egress_class=destination.egress_class,
    )


@dataclass(frozen=True, slots=True)
class ConsoleTurnExecutionContext:
    """Complete immutable execution authority constructed after the gateway."""

    configuration: ConsoleTurnConfigurationSnapshot
    library_authority: ConsoleTurnLibraryAuthority
    resolved_destination: ConsoleResolvedDestination

    def __post_init__(self) -> None:
        """Reject incomplete contexts and detach every constructor input."""
        if not isinstance(self.configuration, ConsoleTurnConfigurationSnapshot):
            raise TypeError("configuration must be a ConsoleTurnConfigurationSnapshot")
        if not isinstance(self.library_authority, ConsoleTurnLibraryAuthority):
            raise TypeError("library_authority must be a ConsoleTurnLibraryAuthority")
        if not isinstance(self.resolved_destination, ConsoleResolvedDestination):
            raise TypeError("resolved_destination must be a ConsoleResolvedDestination")
        object.__setattr__(
            self,
            "configuration",
            _detached_configuration(self.configuration),
        )
        object.__setattr__(
            self,
            "library_authority",
            _detached_authority(self.library_authority),
        )
        object.__setattr__(
            self,
            "resolved_destination",
            _detached_destination(self.resolved_destination),
        )

    @property
    def session_id(self) -> str:
        """Return the captured owning-session identifier."""
        return self.configuration.session_id

    @property
    def effective_model(self) -> str | None:
        """Return the explicit model or its captured configured fallback."""
        return self.configuration.effective_model

    @property
    def provider_selection(self) -> ConsoleProviderSelection:
        """Return the detached pre-gateway provider selection."""
        return self.configuration.provider_selection

    @property
    def session_settings(self) -> ConsoleSessionSettings | None:
        """Return the detached owning-session settings."""
        return self.configuration.session_settings

    @property
    def scratch_space(self) -> ConsoleScratchSnapshot | None:
        """Return the frozen scratch-space authority for this turn."""
        return self.configuration.scratch_space

    @property
    def workspace_roots(self) -> tuple[str, ...]:
        """Return the detached workspace roots."""
        return self.configuration.workspace_roots

    @property
    def change_review_root_aliases(self) -> tuple[str, ...]:
        """Return roots admitted to Change Review for this turn."""
        return self.configuration.change_review_root_aliases

    @property
    def change_review_skipped_roots(self) -> tuple[SkippedReviewRoot, ...]:
        """Return roots skipped by Change Review admission for this turn."""
        return self.configuration.change_review_skipped_roots

    @property
    def capabilities(self) -> Mapping[str, object]:
        """Return the detached provider-capability mapping."""
        return self.configuration.capabilities

    @property
    def rag_defaults(self) -> Mapping[str, object]:
        """Return the detached retrieval defaults."""
        return self.configuration.rag_defaults

    @property
    def tool_configuration(self) -> Mapping[str, object]:
        """Return the detached tool configuration."""
        return self.configuration.tool_configuration

    @property
    def provider_payload_settings(self) -> Mapping[str, object]:
        """Return the detached provider-payload settings."""
        return self.configuration.provider_payload_settings
