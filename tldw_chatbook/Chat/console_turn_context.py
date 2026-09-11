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

from loguru import logger

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
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.console_roleplay_identity import ConsolePresentationContext
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSnapshot
from tldw_chatbook.Workspaces.change_review_consent import SkippedReviewRoot
from tldw_chatbook.Character_Chat.emote_directives import CharacterEmoteRunSnapshot


def capture_change_review_admission(
    app: Any, workspace_id: str | None
) -> tuple[tuple[Any, ...], tuple[str, ...], tuple[SkippedReviewRoot, ...]]:
    """Read only consent-admitted roots for the owning workspace."""
    service = getattr(app, "change_review_consent_service", None)
    if service is not None:
        try:
            admission = service.admit_turn(workspace_id)
            return (
                tuple(admission.ready_roots),
                tuple(getattr(admission, "ready_aliases", ())),
                tuple(admission.skipped_roots),
            )
        except Exception:  # noqa: BLE001 -- review never blocks a send
            pass
    return (), (), ()


def resolve_turn_tool_policy_profile_id(app: Any, workspace_id: str | None) -> str:
    """Read the owning workspace's named profile, preserving legacy defaults."""
    try:
        registry = getattr(app, "workspace_registry_service", None)
        if not workspace_id or registry is None:
            return "default"
        record = registry.get_workspace(workspace_id)
        defaults = getattr(record, "assistant_defaults", None) if record else None
        profile_id = getattr(defaults, "tool_policy_profile_id", None)
        if isinstance(profile_id, str) and profile_id.strip():
            return profile_id.strip()
    except Exception as exc:  # noqa: BLE001 -- preserve existing fallback
        logger.warning(
            "Console turn context: tool policy profile resolution failed; "
            "using the default profile; error_type={}",
            type(exc).__name__,
        )
    return "default"


def resolve_turn_persona_policy_rules(
    app: Any, session: Any
) -> tuple[Mapping[str, Any], ...]:
    """Read rules from the owning session's durable persona identity."""
    try:
        if session is None or session.assistant_kind != "persona":
            return ()
        assistant_id = str(session.assistant_id or "").strip()
        service = getattr(app, "local_character_persona_service", None)
        if not assistant_id or service is None:
            return ()
        profile = service.get_persona_profile(assistant_id)
        rules = profile.get("policy_rules") if isinstance(profile, Mapping) else None
        if isinstance(rules, (list, tuple)):
            return tuple(rule for rule in rules if isinstance(rule, Mapping))
    except Exception as exc:  # noqa: BLE001 -- preserve existing fallback
        logger.warning(
            "Console turn context: persona policy rules resolution failed; "
            "running with no persona rules; error_type={}",
            type(exc).__name__,
        )
    return ()


@dataclass(frozen=True, slots=True)
class ConsoleProjectBindingSnapshot:
    """Detached local-folder authority for one accepted turn."""

    binding_id: str
    workspace_id: str
    display_name: str
    root: str = field(repr=False)
    locator_fingerprint: str = field(repr=False)
    allow_write: bool = False
    root_identity: tuple[tuple[str, int, int, int], ...] = field(
        default=(),
        repr=False,
    )


@dataclass(frozen=True, slots=True)
class ConsoleProjectAuthoritySnapshot:
    """Maximum project authority and setup choices frozen at handoff."""

    workspace_id: str
    enabled: bool
    working_folder_binding_id: str | None = None
    working_folder_locator_fingerprint: str | None = field(
        default=None,
        repr=False,
    )
    project_instruction_notice_key: str | None = field(default=None, repr=False)
    selected: ConsoleProjectBindingSnapshot | None = field(default=None, repr=False)
    options: tuple[ConsoleProjectBindingSnapshot, ...] = field(
        default=(),
        repr=False,
    )


@dataclass(frozen=True, slots=True)
class ConsoleCharacterAuthoritySnapshot:
    """Maximum character/emote identity accepted for one turn."""

    identity_revision: int
    runtime_backend: str
    assistant_id: str | None
    assistant_authority_id: str | None
    local_character_id: int | None
    emote_snapshot: CharacterEmoteRunSnapshot | None = field(
        default=None, repr=False
    )


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
        configured_endpoint_fallback_allowed=(
            selection.configured_endpoint_fallback_allowed
        ),
        endpoint_provenance=selection.endpoint_provenance,
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
    provider_selection: ConsoleProviderSelection = field(repr=False)
    scratch_space: ConsoleScratchSnapshot | None = field(default=None, repr=False)
    session_settings: ConsoleSessionSettings | None = field(default=None, repr=False)
    workspace_roots: tuple[str, ...] = field(default=(), repr=False)
    presentation_context: ConsolePresentationContext | None = field(
        default=None,
        repr=False,
    )
    library_policy_maximum: ConsoleLibraryPolicySnapshot | None = field(
        default=None,
        repr=False,
    )
    library_scope_maximum: ConsoleLibraryItemScopeSnapshot | None = field(
        default=None,
        repr=False,
    )
    project_authority: ConsoleProjectAuthoritySnapshot | None = field(
        default=None,
        repr=False,
    )
    character_authority: ConsoleCharacterAuthoritySnapshot | None = field(
        default=None,
        repr=False,
    )
    prompt_transform_inputs: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
    )
    skill_context_maximum: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
    )
    mcp_tool_maximum: frozenset[str] | None = field(default=None, repr=False)
    mcp_definition_maximum: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )
    change_review_root_aliases: tuple[str, ...] = field(default=(), repr=False)
    change_review_skipped_roots: tuple[SkippedReviewRoot, ...] = field(default=(), repr=False)
    capabilities: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
    )
    rag_defaults: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
    )
    tool_configuration: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
    )
    provider_payload_settings: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
    )
    #: Workspace assistant defaults (Task 7): the owning session's persona
    #: policy rules (already normalized by the persona service); ``()`` is
    #: the identity posture -- no rule can widen, absence changes nothing.
    persona_policy_rules: tuple[Mapping[str, Any], ...] = ()
    #: The workspace's named permission profile this turn resolves tool
    #: gates under; ``"default"`` keeps the single-profile behavior.
    tool_policy_profile_id: str = "default"

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
        object.__setattr__(
            self,
            "presentation_context",
            deepcopy(self.presentation_context),
        )
        object.__setattr__(
            self,
            "library_policy_maximum",
            deepcopy(self.library_policy_maximum),
        )
        object.__setattr__(
            self,
            "library_scope_maximum",
            deepcopy(self.library_scope_maximum),
        )
        object.__setattr__(self, "project_authority", deepcopy(self.project_authority))
        object.__setattr__(
            self,
            "character_authority",
            deepcopy(self.character_authority),
        )
        object.__setattr__(
            self,
            "mcp_tool_maximum",
            (
                frozenset(str(value) for value in self.mcp_tool_maximum)
                if self.mcp_tool_maximum is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "mcp_definition_maximum",
            _freeze(self.mcp_definition_maximum),
        )
        for field_name in (
            "prompt_transform_inputs",
            "skill_context_maximum",
            "capabilities",
            "rag_defaults",
            "tool_configuration",
            "provider_payload_settings",
        ):
            object.__setattr__(self, field_name, _freeze(getattr(self, field_name)))
        # Posture (Task 7): frozen like the mappings above -- a tuple of
        # recursively immutable rule mappings, and a plain coerced string.
        object.__setattr__(
            self,
            "persona_policy_rules",
            _freeze(tuple(self.persona_policy_rules)),
        )
        object.__setattr__(
            self,
            "tool_policy_profile_id",
            str(self.tool_policy_profile_id or "default"),
        )

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
        presentation_context: ConsolePresentationContext | None = None,
        library_policy_maximum: ConsoleLibraryPolicySnapshot | None = None,
        library_scope_maximum: ConsoleLibraryItemScopeSnapshot | None = None,
        project_authority: ConsoleProjectAuthoritySnapshot | None = None,
        character_authority: ConsoleCharacterAuthoritySnapshot | None = None,
        prompt_transform_inputs: Mapping[str, Any] | None = None,
        skill_context_maximum: Mapping[str, Any] | None = None,
        mcp_tool_maximum: Sequence[object] | None = None,
        mcp_definition_maximum: Mapping[str, str] | None = None,
        capabilities: Mapping[str, Any] | None = None,
        rag_defaults: Mapping[str, Any] | None = None,
        tool_configuration: Mapping[str, Any] | None = None,
        provider_payload_settings: Mapping[str, Any] | None = None,
        persona_policy_rules: Sequence[Mapping[str, Any]] | None = None,
        tool_policy_profile_id: str = "default",
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
            presentation_context=presentation_context,
            library_policy_maximum=library_policy_maximum,
            library_scope_maximum=library_scope_maximum,
            project_authority=project_authority,
            character_authority=character_authority,
            prompt_transform_inputs=prompt_transform_inputs or {},
            skill_context_maximum=skill_context_maximum or {},
            mcp_tool_maximum=(
                frozenset(str(value) for value in mcp_tool_maximum)
                if mcp_tool_maximum is not None
                else None
            ),
            mcp_definition_maximum=mcp_definition_maximum or {},
            capabilities=capabilities or {},
            rag_defaults=rag_defaults or {},
            tool_configuration=tool_configuration or {},
            provider_payload_settings=provider_payload_settings or {},
            persona_policy_rules=tuple(persona_policy_rules or ()),
            tool_policy_profile_id=tool_policy_profile_id,
        )

    @property
    def effective_model(self) -> str | None:
        """Return the explicit model or its captured configured fallback."""
        return (
            self.provider_selection.explicit_model
            or self.provider_selection.configured_model
        )


@dataclass(frozen=True, slots=True)
class ConsoleTurnCustodyRequest:
    """Sensitive, detached inputs retained by the app-owned turn runtime."""

    turn_id: str
    session_id: str
    draft: str = field(repr=False)
    configuration: ConsoleTurnConfigurationSnapshot = field(repr=False)
    attachment_ids: tuple[str, ...] = ()
    one_shot_prefill: str | None = field(default=None, repr=False)
    one_shot_prefill_revision: int | None = field(default=None, repr=False)
    staged_evidence_launch: ConsoleLiveWorkLaunch | None = field(
        default=None,
        repr=False,
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
        presentation_context=configuration.presentation_context,
        library_policy_maximum=configuration.library_policy_maximum,
        library_scope_maximum=configuration.library_scope_maximum,
        project_authority=configuration.project_authority,
        character_authority=configuration.character_authority,
        prompt_transform_inputs=configuration.prompt_transform_inputs,
        skill_context_maximum=configuration.skill_context_maximum,
        mcp_tool_maximum=configuration.mcp_tool_maximum,
        mcp_definition_maximum=configuration.mcp_definition_maximum,
        capabilities=configuration.capabilities,
        rag_defaults=configuration.rag_defaults,
        tool_configuration=configuration.tool_configuration,
        provider_payload_settings=configuration.provider_payload_settings,
        persona_policy_rules=configuration.persona_policy_rules,
        tool_policy_profile_id=configuration.tool_policy_profile_id,
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
            endpoint_provenance=authority.provider_intent.endpoint_provenance,
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
        endpoint_provenance=destination.endpoint_provenance,
    )


@dataclass(frozen=True, slots=True)
class ConsoleTurnExecutionContext:
    """Complete immutable execution authority constructed after the gateway."""

    configuration: ConsoleTurnConfigurationSnapshot = field(repr=False)
    library_authority: ConsoleTurnLibraryAuthority = field(repr=False)
    resolved_destination: ConsoleResolvedDestination = field(repr=False)

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
    def presentation_context(self) -> ConsolePresentationContext | None:
        """Return the identity frozen before runtime acceptance."""
        return self.configuration.presentation_context

    @property
    def library_policy_maximum(self) -> ConsoleLibraryPolicySnapshot | None:
        """Return the Library policy maximum frozen before acceptance."""
        return self.configuration.library_policy_maximum

    @property
    def library_scope_maximum(self) -> ConsoleLibraryItemScopeSnapshot | None:
        """Return the Library item scope maximum frozen before acceptance."""
        return self.configuration.library_scope_maximum

    @property
    def project_authority(self) -> ConsoleProjectAuthoritySnapshot | None:
        """Return the project authority maximum frozen before acceptance."""
        return self.configuration.project_authority

    @property
    def character_authority(self) -> ConsoleCharacterAuthoritySnapshot | None:
        """Return the character/emote identity maximum frozen at handoff."""
        return self.configuration.character_authority

    @property
    def prompt_transform_inputs(self) -> Mapping[str, object]:
        """Return frozen dictionary/world-info inputs."""
        return self.configuration.prompt_transform_inputs

    @property
    def skill_context_maximum(self) -> Mapping[str, object]:
        """Return the maximum local skill context frozen at handoff."""
        return self.configuration.skill_context_maximum

    @property
    def mcp_tool_maximum(self) -> frozenset[str] | None:
        """Return exact MCP tool identities eligible at handoff."""
        return self.configuration.mcp_tool_maximum

    @property
    def mcp_definition_maximum(self) -> Mapping[str, str]:
        """Exact admitted MCP definition hashes keyed by raw tool ID."""
        return self.configuration.mcp_definition_maximum

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

    @property
    def persona_policy_rules(self) -> tuple[Mapping[str, Any], ...]:
        """Return the frozen persona policy rules for this turn."""
        return self.configuration.persona_policy_rules

    @property
    def tool_policy_profile_id(self) -> str:
        """Return the workspace's named permission profile for this turn."""
        return self.configuration.tool_policy_profile_id
