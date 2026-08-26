"""Native Console chat controller for send, stream, stop, and retry flows."""

from __future__ import annotations

import asyncio
import copy
from contextvars import ContextVar
import functools
import hashlib
import inspect
import os
import re
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    TypedDict,
)
from uuid import uuid4

from loguru import logger
from rich.markup import escape as escape_markup

from tldw_chatbook.Character_Chat.emote_directives import (
    CharacterEmoteAssetReference,
    CharacterEmoteRunSnapshot,
    append_character_emote_prompt_instruction,
    project_character_emote_assets,
)
from tldw_chatbook.Chat.attachment_core import (
    PendingAttachment,
    image_url_part,
    max_history_images,
    vision_block_reason,
)
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_CAP_REFUSAL_TITLE_LIMIT,
    CONSOLE_DEFAULT_MAX_PARALLEL_RUNS,
    CONSOLE_DISPATCH_DISCARDED_COPY,
    ConsoleChatMessage,
    ConsoleControllerActivity,
    ConsoleLifecycleImpact,
    ConsoleNextSendHistoryProjection,
    ConsoleCitationNoticeCode,
    ConsoleCitationPhase,
    ConsoleCitationPresentation,
    ConsoleContextSnapshot,
    ProjectInstructionActivationEvent,
    ProjectInstructionPreview,
    ConsoleMessageRole,
    ConsoleDispatchRecoveryActionId,
    ConsoleDispatchRecoveryKind,
    ConsoleProviderSelection,
    ConsoleRunMarker,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleQueuedAcceptanceEvent,
    ConsoleSubmissionOrigin,
    ConsoleStagedSource,
    ConsoleWorkspaceContext,
    MessageAttachment,
    derive_console_session_title,
    fold_greeting_into_system_prompt,
    is_default_console_session_title,
)
from tldw_chatbook.Chat.citation_repair import (
    REPAIR_ANSWER_BODY_UTF8_BYTES_MAX,
    CitationRepairContract,
    CitationRepairDecision,
    build_citation_repair_messages,
    decide_citation_repair,
    repair_request_fits_model_window,
    select_repaired_body,
)
from tldw_chatbook.Chat.citation_trace_builder import (
    CitationTraceBuilder,
    CitationTraceBuildUnavailable,
)
from tldw_chatbook.Chat.citation_trace_models import SealedCitationWrite
from tldw_chatbook.Chat.citation_evidence_models import EvidenceBundle
from tldw_chatbook.Chat.answer_citations import format_evidence_for_cited_answer
from tldw_chatbook.Chat.console_chat_store import (
    CapturePurgeStaleError,
    CapturePolicyStaleError,
    ConsoleChatSession,
    ConsoleChatStore,
    ConsoleDispatchSettlementError,
    ConsoleDurableAcceptanceRetired,
    ConsoleDurableAcceptanceFingerprint,
    ConsoleDurableTurnCommit,
    ConsoleThinkingCompatibilityError,
    TerminalCitationFinalizer,
    require_thinking_persistence_support,
)
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    CapturePolicyResolution,
    resolve_capture_policy,
)
from tldw_chatbook.Chat.console_capture_policy_repository import (
    CapturePolicyReadStatus,
    CapturePolicyWriteStatus,
    ConsoleCapturePolicyRepository,
)
from tldw_chatbook.Chat.console_command_grammar import COMMAND_PREFIX
from tldw_chatbook.Chat.console_history_budget import (
    DEFAULT_RESPONSE_RESERVATION,
    ProviderContinuationSidecar,
    bound_messages_to_window,
    count_console_messages_tokens,
    provider_continuation_owner_groups,
)
from tldw_chatbook.Chat.console_context_compaction import (
    CompactionAdmission,
    CompactionDecision,
    CompactionPromptSnapshot,
    CompactionTerminal,
    ConsoleCompactionService,
    DurableMessageSnapshot,
    compactable_units_after,
    decide_compaction,
    plan_compaction,
    prefix_digest,
    select_valid_memory,
)
from tldw_chatbook.Chat.console_context_policy import (
    CompactionFailureBehavior,
    ConsoleContextCapacity,
    ConsoleContextPolicyOverrides,
    ContextCarryForwardMode,
    ContextCompactionRepresentation,
    context_policy_overrides_from_console_config,
    merge_context_policy,
    resolve_context_policy,
)
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleContextRepository,
    ConsoleMemoryRecord,
)
from tldw_chatbook.Chat.assistant_generation_state import (
    assistant_state_allows_provider_history,
)
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpointState,
    ConsoleDispatchReconstructability,
    ConsoleDispatchResultStatus,
    ConsoleDispatchTransition,
    ConsoleDurableTurnAcceptance,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.Chat.console_prepared_request import (
    CONTINUATION_OWNER_KEY,
    PreparedConsoleRequest,
    tagged_memory_message,
    tagged_visual_memory_message,
)
from tldw_chatbook.Chat.console_visual_transcript import (
    count_semantic_images,
    plan_visual_compaction,
    render_visual_transcript,
    resolve_effective_compaction_representation,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_default_console_session_settings,
    normalize_llamacpp_base_url,
)
from tldw_chatbook.Chat.console_provider_endpoints import first_configured_endpoint
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    fingerprint_canonical_locator,
    project_instruction_notice_key,
    sanitized_destination_label,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationConflictError,
    ContinuationRestoreTarget,
    ProviderContinuationCheckpoint,
    validate_continuation_restore,
)
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Chat.console_roleplay_identity import (
    ConsoleMessagePresentation,
    ConsolePresentationContext,
    expand_character_template,
    resolve_console_message_presentation,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsolePreparationTransition,
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
    initial_preparation_state,
)
from tldw_chatbook.Chat.library_preparation import (
    LibraryPreparationContribution,
    library_preparation_event_for_outcome,
)
from tldw_chatbook.Chat.rag_scope import EffectiveScope
from tldw_chatbook.Chat.console_prompt_queue import (
    ConsolePromptQueueRegistry,
    PromptQueueMutationResult,
    QueueMutationStatus,
)
from tldw_chatbook.Chat.console_prompt_queue_coordinator import (
    ConsolePromptQueueCoordinator,
    QueueGenerationAuthorization,
)
from tldw_chatbook.Chat.console_fleet_wake import (
    AgentWakeAuthorization,
    ConsoleFleetWakeCoordinator,
)
from tldw_chatbook.Chat.message_metadata import (
    MESSAGE_ORIGIN_AGENT_WAKE,
    MessageMetadata,
)
from tldw_chatbook.Chat.console_skill_resolver import (
    MENTION_SIGIL,
    SKILL_MENTION_SKIPPED_NOTE,
    SKILL_UNTRUSTED_REFUSE,
    SkillCommandCandidate,
    cap_skill_args,
    find_embedded_mentions,
    resolve_skill_command,
)
from tldw_chatbook.Chat.prompt_history import PromptHistory

if TYPE_CHECKING:
    from tldw_chatbook.Persona_Buddy.console_adapter import PersonaBuddyConsoleAdapter

from tldw_chatbook.Agents.builtin_tool_gate import (
    LOCAL_TOOLS_DEFAULT_ENABLED,
    build_builtin_gate,
)
from tldw_chatbook.Agents.human_input_wait import use_human_input_wait
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionSnapshot,
    ProjectInstructionResolver,
    StartupInstructionCandidate,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall, MCPToolProvider
from tldw_chatbook.Agents.run_context import current_run_id
from tldw_chatbook.Agents.session_todo_store import (
    SessionTodoStore,
    TodoChangeCallback,
)
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
from tldw_chatbook.config import (
    ConfigMutationResult,
    DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
    MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
    MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
    coerce_bool_setting,
    coerce_int_setting,
    get_cli_setting,
    apply_console_capture_settings,
    runtime_capture_policy,
)
from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS
from tldw_chatbook.Library.library_rag_service import (
    LibraryRagSearchRequest,
    _outcome_from_service_result,
)
from tldw_chatbook.UI.Views.RAGSearch.search_handoff import (
    build_library_rag_evidence_bundle,
)
from tldw_chatbook.MCP.permission_store import BUILTIN_TOOL_SERVER_KEY
from tldw_chatbook.runtime_policy.bootstrap import (
    load_default_runtime_source_state,
)
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Tools.file_operation_tools import path_precheck_failed
from tldw_chatbook.Tools.watchlists_tool_service import WatchlistsToolService
from tldw_chatbook.Utils.input_validation import validate_console_draft
from tldw_chatbook.Chat.provider_failures import (  # noqa: F401  (re-export: tests and callers import describe_stream_failure from here)
    describe_stream_failure,
)
from tldw_chatbook.Chat.console_cost_tracker import (
    PayloadFingerprint,
    fingerprint_payload,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
    ProviderProprietaryThinkingEvidence,
    ProviderThinkingDelta,
)
from tldw_chatbook.Chat.console_thinking_capture import ThinkingCapture
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.model_capabilities import (
    is_vision_capable,
    moonshot_model_returns_reasoning_content,
)

if TYPE_CHECKING:
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool


def get_internal_prompt(prompt_id: str) -> str:
    """Resolve an internal prompt without putting ``Internal_Prompts`` on boot.

    TASK-22213: this module is on the Chat first-paint import leg (via
    ``UI/Screens/chat_screen.py``), and the former module-scope
    ``from tldw_chatbook.Internal_Prompts import get_internal_prompt`` put
    all 10 prompt-catalog modules in front of first paint -- the exact leg
    TASK-21731's guard could not see. The import now happens on first use
    (the ``/rewind`` summarize path), which is always long after mount.
    Same name and signature as the real resolver, so call sites and any
    module-namespace patches are unchanged. Guarded by
    ``Tests/Packaging/test_rag_boot_import_closure.py``.

    Args:
        prompt_id: Prompt identifier (e.g., ``"console.rewind_summarize"``).

    Returns:
        Resolved prompt text with placeholders intact.

    Raises:
        KeyError: If ``prompt_id`` is not registered in the catalog.
    """
    from tldw_chatbook.Internal_Prompts import get_internal_prompt as _resolve

    return _resolve(prompt_id)


class _TodoWiring(TypedDict, total=False):
    """Typed optional task kwargs for ``LocalToolProvider`` composition."""

    todo_store: SessionTodoStore
    on_todo_change: TodoChangeCallback


#: task-1337 (plan Task 8): raw built-in tool names the Console-composed
#: ``MCPToolProvider`` must exclude -- the 18 ``library_*`` descriptor tools
#: (served to Console agents by the run's own direct/RAG Library provider, in
#: either retrieval mode) plus the five legacy RAG/chat readers whose Console
#: coverage those providers replace. The legacy names live HERE, not in the
#: shared descriptor table: they are not part of the 18-tool contract. The
#: filter is source-scoped inside the provider (``builtin:tldw_chatbook``
#: only), so external/local MCP profiles fronting the same raw names stay
#: eligible and permission-governed.
CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS: frozenset = frozenset(
    tuple(LIBRARY_TOOL_DESCRIPTORS)
    + (
        "search_rag",
        "search_notes",
        "search_conversations",
        "get_conversation_history",
        "export_conversation",
    )
)


#: ADR-067: 0 -- the default for all three human-prompt timeouts below --
#: means "no deadline": the round stays armed until the user answers or the
#: run is stopped/cancelled. A positive value still fails undecided calls
#: closed to ``"timeout"``. A human prompt is not a wedged tool; making the
#: user race a clock to keep their run was the defect, auto-deny is opt-in.
_WINDOWS = os.name == "nt"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", None)


def build_console_provider_selection_from_settings(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, Any],
    workspace_context: Any,
    legacy_model: Any = None,
) -> ConsoleProviderSelection:
    """Reconstruct the effective selection from one session snapshot."""

    def section(value: Any, key: str) -> Mapping[str, Any]:
        child = value.get(key, {}) if isinstance(value, Mapping) else {}
        return child if isinstance(child, Mapping) else {}

    def selected(value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text if text and text.lower() not in {"none", "null"} else None

    provider = provider_config_key(settings.provider) or "llama_cpp"
    explicit_model = selected(settings.model)
    provider_config = section(section(app_config, "api_settings"), provider)
    configured_model = selected(
        provider_config.get("model")
        or provider_config.get("api_model")
        or provider_config.get("default_model")
    )
    if selected(legacy_model) is None and explicit_model == configured_model:
        explicit_model = None

    base_url: str | None = None
    if provider in {"llama_cpp", "local_llamacpp"}:
        console_config = section(app_config, "console")
        fallback_url = (
            os.environ.get("TLDW_CONSOLE_LLAMA_CPP_BASE_URL")
            or console_config.get("llama_cpp_base_url_override")
            or first_configured_endpoint(provider_config)
        )
        base_url = normalize_llamacpp_base_url(
            selected(settings.base_url) or selected(fallback_url)
        )
    elif selected(settings.base_url) is not None:
        base_url = selected(settings.base_url)

    defaults = build_default_console_session_settings(app_config, provider, None)
    return ConsoleProviderSelection(
        provider=provider,
        base_url=base_url,
        explicit_model=explicit_model,
        configured_model=configured_model,
        temperature=settings.temperature,
        top_p=settings.top_p,
        min_p=settings.min_p,
        top_k=settings.top_k,
        max_tokens=(
            settings.max_tokens
            if settings.max_tokens is not None
            else defaults.max_tokens
        ),
        seed=settings.seed,
        presence_penalty=settings.presence_penalty,
        frequency_penalty=settings.frequency_penalty,
        reasoning_effort=settings.reasoning_effort,
        reasoning_summary=settings.reasoning_summary,
        verbosity=settings.verbosity,
        thinking_effort=settings.thinking_effort,
        thinking_budget_tokens=settings.thinking_budget_tokens,
        streaming=settings.streaming,
        system_prompt=settings.system_prompt,
        workspace_context=workspace_context,
    )


#: Fallback used when no `mcp_approval_timeout_seconds` seam is injected --
#: mirrors `UnifiedMCPControlPlaneService.approval_timeout_seconds`'s own
#: default (task-201/T2), read directly here since the controller has no
#: dependency on that service (T6 wires the service into `MCPToolProvider`,
#: not into this controller).
#:
#: The PRE-ADR default was 120.0, kept strictly below
#: ``RunBudget.max_tool_call_seconds`` (300s at defaults,
#: ``Agents/agent_models.py``) because the invoke-path approval wait runs
#: INSIDE ``agent_service._call_with_timeout``'s per-call wrapper: an
#: approval timeout at/above the wrapper ceiling would let the wrapper fire
#: first, report the call failed, and a late approval would then execute
#: the tool for real on the abandoned thread. That invariant is SUPERSEDED:
#: the wrapper's deadline now PAUSES while a human decision is pending for
#: the run (``Agents/human_input_wait`` marks the wait;
#: ``_call_with_timeout``'s ``pauses_deadline`` re-arms the ceiling each
#: poll slice), so wall-clock counts tool execution, not human deliberation
#: -- an indefinite wait can no longer lose that race. Cancellation was
#: already closed by ``revoke_approval_rounds_for_run`` and is unaffected.
_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS = 0.0
#: Poll granularity for `request_mcp_approvals`'s wait loop (binding, from
#: the Phase-5 plan) -- also the worst-case slack added on top of a
#: configured timeout/cancellation before this method observes it.
_MCP_APPROVAL_POLL_SECONDS = 1.0
#: Same ADR-067 contract as `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS`, for
#: `request_skill_install_confirm`'s own wait loop (fallback used when no
#: `skill_install_confirm_timeout_seconds` seam is injected).
_DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS = 0.0
#: Same ADR-067 contract as `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS`, for
#: `request_skill_script_confirm`'s own wait loop (fallback used when no
#: `skill_script_confirm_timeout_seconds` seam is injected).
_DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS = 0.0
_DISPATCH_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z", re.ASCII)
#: TASK-1050: synthetic round id `set_run_pending_approval`'s deprecated
#: boolean shim registers under, internally, so its add/discard composes
#: safely with the round-keyed `_pending_approvals` accounting (see that
#: method's docstring) without ever colliding with a real bridge round's
#: `uuid4()` id -- every genuine round id is a UUID string; this is not.
_LEGACY_PENDING_APPROVAL_ROUND_ID = "__legacy_pending_approval__"


def project_recovery_should_skip_send_interception(
    recovery_code: str, state: "ProjectInstructionControlState"
) -> bool:
    """Whether a binding-recovery on the send path has nothing to recover.

    TASK-21145 (UAT H-2): a session that never had a folder bound and has
    no eligible folders must not intercept the user's message with a setup
    modal — project instructions simply don't apply to that send. Sessions
    whose EXISTING binding broke (unavailable/retargeted) or where several
    folders need an explicit choice still get the recovery dialog.

    Args:
        recovery_code: The ProjectInstructionBindingRecovery code raised
            by the resolver.
        state: The session's project-instruction control state.

    Returns:
        True when the send should proceed without project instructions
        instead of showing the recovery dialog.
    """
    return (
        recovery_code == "no_eligible_binding"
        and state.working_folder_binding_id is None
    )


class ProjectInstructionBindingRecovery(RuntimeError):
    """Raised when an enabled session cannot prove one selected folder root."""


@dataclass(frozen=True, slots=True)
class ProjectInstructionBindingSelection:
    """Validated folder binding used as one agent dispatch's authority root."""

    binding: Any
    root: Path
    locator_fingerprint: str
    allow_write: bool
    root_identity: tuple[tuple[str, int, int, int], ...]


@dataclass(frozen=True, slots=True)
class _ProjectInstructionAuthoritySnapshot:
    """Minimal immutable session state used for off-loop authority checks."""

    workspace_id: str
    project_instruction_state: ProjectInstructionControlState


@dataclass(frozen=True, slots=True)
class ProjectInstructionDispatchNotice:
    """Content-free consent data for one owning session and destination."""

    session_id: str
    destination_label: str
    relative_source: str | None
    scope: str
    byte_count: int
    outcomes: tuple[str, ...]
    warning_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectInstructionDisplayMetadata:
    """Immutable content-free result of one final primary delivery decision."""

    binding_id: str
    locator_fingerprint: str
    relative_source: str | None
    scope: str
    byte_count: int
    outcome: str
    warning_codes: tuple[str, ...]


def build_project_instruction_dispatch_notice(
    snapshot: InstructionSnapshot, *, session_id: str, resolution: Any
) -> ProjectInstructionDispatchNotice:
    """Build sanitized consent data from the exact resolved destination."""
    source = getattr(snapshot, "startup_source_metadata", None)
    if source is None:
        source = snapshot.startup_source
    provider_label = str(
        getattr(resolution, "provider", "")
        or getattr(resolution, "execution_key", "")
        or "Provider"
    )
    return ProjectInstructionDispatchNotice(
        session_id=session_id,
        destination_label=sanitized_destination_label(
            provider_label, getattr(resolution, "base_url", None)
        ),
        relative_source=source.relative_path if source else None,
        scope=source.scope if source else ".",
        byte_count=source.byte_count if source else 0,
        outcomes=tuple(outcome.code for outcome in snapshot.primary_delivery.outcomes),
        warning_codes=snapshot.warning_codes,
    )


def _capture_project_root_identity(
    root: Path,
) -> tuple[tuple[str, int, int, int], ...] | None:
    """Capture root/ancestor identities while rejecting every symlink component."""
    identities: list[tuple[str, int, int, int]] = []
    try:
        for component in (*reversed(root.parents), root):
            value = os.lstat(component)
            if (
                stat.S_ISLNK(value.st_mode)
                or _is_project_root_reparse(value)
                or not stat.S_ISDIR(value.st_mode)
            ):
                return None
            identities.append(
                (str(component), value.st_dev, value.st_ino, value.st_mode)
            )
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return tuple(identities)


def _is_project_root_reparse(value: object) -> bool:
    if not _WINDOWS:
        return False
    attributes = getattr(value, "st_file_attributes")
    if attributes is None or _REPARSE_POINT is None:
        raise ValueError("unsafe reparse metadata")
    return bool(int(attributes) & int(_REPARSE_POINT))


def _project_root_identity_matches(
    root: Path, expected: tuple[tuple[str, int, int, int], ...]
) -> bool:
    """Fail closed when a selected root or any ancestor changed identity."""
    return _capture_project_root_identity(root) == expected


def resolve_project_instruction_binding(
    session: ConsoleChatSession, registry: Any
) -> ProjectInstructionBindingSelection | None:
    """Resolve one enabled session's binding without silently retargeting it."""
    state = session.project_instruction_state
    if not state.project_instructions_enabled:
        return None
    if registry is None:
        raise ProjectInstructionBindingRecovery("binding_unavailable")

    selected_id = state.working_folder_binding_id
    if selected_id:
        try:
            binding = registry.get_runtime_binding(selected_id)
        except (KeyError, OSError, RuntimeError, ValueError, AttributeError):
            raise ProjectInstructionBindingRecovery("binding_unavailable") from None
        selection = _validate_project_instruction_binding(session, binding)
        if selection is None:
            raise ProjectInstructionBindingRecovery("binding_unavailable")
        if selection.locator_fingerprint != state.working_folder_locator_fingerprint:
            raise ProjectInstructionBindingRecovery("binding_retargeted")
        return selection
    eligible = list_project_instruction_bindings(session, registry)
    if not eligible:
        raise ProjectInstructionBindingRecovery("no_eligible_binding")
    if len(eligible) != 1:
        raise ProjectInstructionBindingRecovery("choose_binding")
    return eligible[0]


def list_project_instruction_bindings(
    session: ConsoleChatSession, registry: Any
) -> tuple[ProjectInstructionBindingSelection, ...]:
    """Return currently eligible bindings for an explicit setup choice."""
    if registry is None:
        return ()

    try:
        bindings = registry.list_runtime_bindings(session.workspace_id)
    except (KeyError, OSError, RuntimeError, ValueError, AttributeError):
        raise ProjectInstructionBindingRecovery("binding_unavailable") from None
    return tuple(
        selection
        for binding in bindings
        if (selection := _validate_project_instruction_binding(session, binding))
        is not None
    )


def _validate_project_instruction_binding(
    session: ConsoleChatSession, binding: Any
) -> ProjectInstructionBindingSelection | None:
    if binding is None or str(getattr(binding, "workspace_id", "")) != str(
        session.workspace_id
    ):
        return None
    kind = getattr(getattr(binding, "binding_kind", None), "value", None) or str(
        getattr(binding, "binding_kind", "")
    )
    status = getattr(getattr(binding, "status", None), "value", None) or str(
        getattr(binding, "status", "")
    )
    if kind != "local-filesystem" or status != "ready":
        return None
    try:
        lexical = Path(str(binding.locator)).expanduser().absolute()
        if _capture_project_root_identity(lexical) is None:
            return None
        root = lexical.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if root != lexical:
        return None
    root_identity = _capture_project_root_identity(root)
    if root_identity is None:
        return None
    return ProjectInstructionBindingSelection(
        binding=binding,
        root=root,
        locator_fingerprint=fingerprint_canonical_locator(str(root)),
        allow_write=str(getattr(binding, "metadata", {}).get("access", "ro")) == "rw",
        root_identity=root_identity,
    )


def _same_project_instruction_authority(
    current: ProjectInstructionBindingSelection,
    expected: ProjectInstructionBindingSelection,
) -> bool:
    return (
        str(current.binding.binding_id) == str(expected.binding.binding_id)
        and current.root == expected.root
        and current.locator_fingerprint == expected.locator_fingerprint
        and current.allow_write == expected.allow_write
        and current.root_identity == expected.root_identity
        and _project_root_identity_matches(expected.root, expected.root_identity)
    )


def project_instruction_authority_is_current(
    *,
    store: Any,
    session_id: str,
    registry: Any,
    expected_selection: ProjectInstructionBindingSelection,
) -> bool:
    """Re-resolve binding state and selected-root identity without retargeting."""
    session = next((item for item in store.sessions() if item.id == session_id), None)
    if session is None:
        return False
    return project_instruction_authority_snapshot_is_current(
        session_snapshot=session,
        registry=registry,
        expected_selection=expected_selection,
    )


def project_instruction_authority_snapshot_is_current(
    *,
    session_snapshot: Any,
    registry: Any,
    expected_selection: ProjectInstructionBindingSelection,
) -> bool:
    """Re-resolve filesystem authority from an immutable session snapshot."""
    session = session_snapshot
    state = session.project_instruction_state
    if (
        not state.project_instructions_enabled
        or state.working_folder_binding_id != expected_selection.binding.binding_id
        or state.working_folder_locator_fingerprint
        != expected_selection.locator_fingerprint
    ):
        return False
    try:
        current_selection = resolve_project_instruction_binding(session, registry)
    except ProjectInstructionBindingRecovery:
        return False
    return current_selection is not None and _same_project_instruction_authority(
        current_selection, expected_selection
    )


def commit_project_instruction_setup_decision(
    *,
    store: Any,
    session_id: str,
    registry: Any,
    expected_state: ProjectInstructionControlState,
    expected_options: tuple[ProjectInstructionBindingSelection, ...],
    action: str,
    binding_id: str | None,
) -> tuple[
    Literal["select", "disable", "cancel"],
    ProjectInstructionBindingSelection | None,
]:
    """Commit one chooser result only while state and authority stay exact."""
    session = next((item for item in store.sessions() if item.id == session_id), None)
    if session is None or session.project_instruction_state != expected_state:
        return "cancel", None
    if action == "cancel":
        return "cancel", None
    if action == "disable":
        store.set_session_project_instruction_state(
            session_id, ProjectInstructionControlState.legacy_disabled()
        )
        return "disable", None
    if action != "select" or binding_id is None:
        return "cancel", None
    expected = next(
        (
            option
            for option in expected_options
            if str(option.binding.binding_id) == str(binding_id)
        ),
        None,
    )
    if expected is None:
        return "cancel", None
    try:
        fresh_options = list_project_instruction_bindings(session, registry)
    except ProjectInstructionBindingRecovery:
        return "cancel", None
    current = next(
        (
            option
            for option in fresh_options
            if str(option.binding.binding_id) == str(binding_id)
        ),
        None,
    )
    if current is None or not _same_project_instruction_authority(current, expected):
        return "cancel", None
    store.set_session_project_instruction_state(
        session_id,
        ProjectInstructionControlState(
            project_instructions_enabled=True,
            working_folder_binding_id=current.binding.binding_id,
            working_folder_locator_fingerprint=current.locator_fingerprint,
            project_instruction_notice_key=None,
        ),
    )
    return "select", current


def commit_project_instruction_dispatch_decision(
    *,
    store: Any,
    session_id: str,
    registry: Any,
    expected_state: ProjectInstructionControlState,
    expected_selection: ProjectInstructionBindingSelection,
    notice_key: str,
    decision: Literal["proceed", "cancel", "disable"] | None,
) -> Literal["proceed", "cancel", "disable", "prompt"]:
    """Atomically validate authority and apply consent on the owning loop."""
    session = next((item for item in store.sessions() if item.id == session_id), None)
    if session is None or session.project_instruction_state != expected_state:
        return "cancel"
    if not project_instruction_authority_is_current(
        store=store,
        session_id=session_id,
        registry=registry,
        expected_selection=expected_selection,
    ):
        return "cancel"
    if expected_state.project_instruction_notice_key == notice_key:
        return "proceed"
    if decision is None:
        return "prompt"
    if decision == "cancel":
        return "cancel"
    enabled = decision == "proceed"
    store.set_session_project_instruction_state(
        session_id,
        ProjectInstructionControlState(
            project_instructions_enabled=enabled,
            working_folder_binding_id=expected_selection.binding.binding_id,
            working_folder_locator_fingerprint=expected_selection.locator_fingerprint,
            project_instruction_notice_key=notice_key if enabled else None,
        ),
    )
    return decision


#: TASK-1861: the tool result a call gets when the user refused it at the
#: approval card. The runtime turns any non-"proceed" verdict string into the
#: call's result without dispatching it, so this text is what the MODEL reads
#: and must say who refused and which tool -- matching the wording
#: `BuiltinToolGate.check` already uses for a denied tool, so a refusal reads
#: the same whether it was stopped here or at the gate.
USER_DENIED_REFUSAL = "tool call denied by the user: {name}"

#: TASK-631: the result every tool call gets while the kill switch is on.
#: Enforced at the review hook -- the one place EVERY parsed call passes,
#: including the families neither provider claims (skills, spawn_subagent,
#: find_tools, load_tools) that previously ran normally with the switch on.
#: Deliberately names the switch so the model (and a user reading the
#: transcript) can tell this from a per-call denial.
KILL_SWITCH_REFUSAL = "tool call blocked: chat tool calls are disabled (kill switch)"

#: TASK-1861: how broad each approval scope is. A session/always grant is
#: recorded against a tool NAME, so when per-call rows of one tool are
#: approved at different scopes only one can be stamped -- and it must be the
#: broadest the user chose, or the grant they asked for is silently dropped
#: and re-prompted. Unknown values rank 0 (narrower than anything known).
_APPROVAL_SCOPE_RANK: dict[str, int] = {
    "approve_once": 1,
    "approve_session": 2,
    "always_allow": 3,
}


CONSOLE_CONTINUE_INSTRUCTION = "Continue and extend the selected message."
#: The generic copy for a turn ended by the session closing. One name for
#: it so the six sites that reported a close cannot drift apart (TASK-22690).
SESSION_CLOSED_COPY = "Session closed."
PROVIDER_CONTINUATION_RECOVERY_REQUIRED = (
    "Recover the interrupted tool run before sending a new message: "
    "Resume or Discard it first."
)

# Private payload-row key threading a transcript message's native id from the
# payload builder to the dispatch choke point, where `/rewind`
# "summarize up to here" compaction anchors the boundary by IDENTITY rather
# than by content (see `_apply_context_summary_compaction`). It is opt-in
# (send paths only, `annotate_ids=True`) and ALWAYS stripped from every row
# before the payload leaves the controller for a provider/agent, so no
# provider ever sees it.
NATIVE_MESSAGE_ID_KEY = "_native_message_id"


def _flatten_preflight_messages(
    semantic: PreparedConsoleRequest,
) -> list[dict[str, Any]]:
    """Return visible rows while preserving private owner association."""
    flattened: list[dict[str, Any]] = []
    for message in semantic.flattened_messages():
        row = dict(message)
        owner_id = row.pop(CONTINUATION_OWNER_KEY, None)
        if type(owner_id) is str:
            row[NATIVE_MESSAGE_ID_KEY] = owner_id
        flattened.append(row)
    return flattened


def _continuation_restore_target_for_resolution(
    resolution: Any,
) -> ContinuationRestoreTarget | None:
    """Build an exact target only from explicitly pinned resolution fields.

    ``api_base_url`` is carried through VERBATIM. It used to be passed
    through ``normalize_generic_endpoint_for_compare`` first, which expands
    a base URL to its full chat-completions endpoint
    (``https://api.moonshot.ai/v1`` -> ``.../v1/chat/completions``). That
    silently broke every Resume: the checkpoint pins whatever
    ``ConsoleAgentBridge`` recorded, and the bridge records
    ``resolution.base_url`` RAW (see its ``ContinuationRestoreTarget(...)``
    construction), so recovery compared an expanded URL against a
    non-expanded one and ``validate_continuation_restore`` -- which is
    byte-exact by design, down to a trailing slash -- rejected it with
    "Pinned provider settings no longer match".

    Normalizing BOTH sides instead would have been the wrong repair: the
    exactness is deliberate (``test_provider_continuation`` pins that
    ``https://api.deepseek.com/v1/`` and ``https://api.deepseek.com/v1``
    are a MISMATCH), and this comparison is what stops a private
    continuation from being replayed against a different endpoint than the
    one that produced it. The two writers simply have to agree, and the
    checkpoint's writer is the one that defines the format.
    """
    protocol = getattr(resolution, "continuation_protocol", None) or getattr(
        resolution, "api_mode", None
    )
    base_url = getattr(resolution, "base_url", None)
    model = getattr(resolution, "model", None)
    if (
        not protocol
        or not isinstance(base_url, str)
        or not base_url.startswith(("http://", "https://"))
        or not model
    ):
        return None
    return ContinuationRestoreTarget(
        provider=provider_config_key(str(getattr(resolution, "provider", ""))),
        model=str(model),
        protocol=str(protocol),
        api_base_url=base_url,
    )


def _normalize_world_info_history(
    messages: "list[dict[str, Any]]",
) -> "list[dict[str, Any]]":
    """Flatten messages to ``{"role","content": str}`` for world-info scanning.

    ``WorldInfoProcessor.process_messages`` types content as ``str``; native
    provider messages may carry multimodal list content, so extract the text
    parts (joined) and drop images before scanning. System messages are
    skipped entirely -- world-info should scan only the user/assistant
    conversation, matching the legacy path; keywords in the system prompt
    must not spuriously activate entries.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == ConsoleMessageRole.SYSTEM.value:
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )
        else:
            text = ""
        out.append({"role": message.get("role", ""), "content": text})
    return out


def _collect_mcp_pending(
    provider: MCPToolProvider, calls: list["ToolCall"]
) -> list["MCPPendingCall"]:
    """Resolve each call's MCP gate; return the subset that needs asking.

    Extracted so `build_mcp_review_hook` (MCP-only, still used directly by
    its own long-standing tests) and `build_tool_review_hook` (T6: the
    run-level hook that folds built-ins in too) share this ONE walk over
    `provider.pending_gate_for` rather than one copying the other's body.
    `None` per call means either "not an MCP call this provider owns" or
    "an MCP call whose current state doesn't need asking" -- see
    `pending_gate_for`'s own docstring for why callers do not need to
    distinguish those two cases.
    """
    pending: list["MCPPendingCall"] = []
    for call in calls:
        gate = provider.pending_gate_for(
            call.name, call.args, str(getattr(call, "call_id", "") or "")
        )
        if gate is not None:
            pending.append(gate)
    return pending


def build_mcp_review_hook(
    provider: MCPToolProvider,
    request_mcp_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"]], dict[str, str]]:
    """Build this run's T4 `review_tool_calls` hook for one composed MCP provider.

    Handed to `ConsoleAgentBridge.run_reply` (P5-T6), which forwards it
    straight through to `AgentService`/`LoopDeps.review_tool_calls` (T4):
    called ONCE per turn with the full batch of tool calls about to be
    dispatched, before any of them is invoked.

    For every call in the batch, `provider.pending_gate_for(name, args)`
    resolves whether it needs human gating (`None` for both "not an MCP
    call this provider owns" and "an MCP call whose current state doesn't
    need asking" -- `invoke()` re-resolves either case for itself, so
    this hook does not need to distinguish them). When at least one call
    needs asking, this makes exactly ONE `request_mcp_approvals` round
    trip for the whole batch (never one per call) and hands the resulting
    decisions to `provider.apply_batch_decisions` -- a per-turn stamp
    every same-named call `invoke()` makes THIS turn peeks (Finding F1:
    never popped, so two calls to the same tool in one batch both see the
    approval, not just the first).

    Finding F1 also requires this hook to call
    `provider.apply_batch_decisions` on EVERY invocation, even when
    `pending` ends up empty (a turn whose calls are all non-MCP, or all
    already resolved without asking) -- passing `{}` in that case.
    `apply_batch_decisions` REPLACES the stamp set rather than merging, so
    this is what guarantees a stamp from an earlier turn can never survive
    into a later one and be misread as this turn's verdict for a
    repeated tool name.

    I3 (probe-verified): that clear happens at hook ENTRY, before
    `pending_gate_for` is even resolved and before the
    `request_mcp_approvals` round trip -- not only after a successful one.
    `request_mcp_approvals` can raise (e.g. the unguarded
    `_marshal_pending_approval` call mid-shutdown); `run_agent_loop`'s own
    hook-exception handling fails the WHOLE batch open (treats every call
    in it as `"proceed"`) when that happens. If the clear only ran after a
    successful round trip, a raise would leave THIS turn's stamp set
    exactly as the PREVIOUS turn left it -- so the fail-open runtime would
    hand `invoke()` a stale prior-turn stamp (e.g. a real `"approve_once"`)
    for a call the user never decided on this turn. Clearing first means a
    raised round trip always leaves `invoke()` with no stamp to peek,
    falling through to its own fresh gate -- which fails closed for an
    `"ask"` tool with no approval_callback wired.

    Design choice (binding, per the Phase-5 plan): this hook never
    returns a refusal string itself. Every MCP call it stamped is left to
    resolve through `invoke()`'s own gate on dispatch -- `invoke()`
    already handles every decision string uniformly (`approve_once`/
    `approve_session`/`always_allow` execute; `deny`/`timeout` refuse with
    the exact model-facing copy AND record the audit decision), so
    routing every decision through that ONE place keeps the refusal copy
    and the audit trail single-sourced instead of duplicating that logic
    here. The verdict map this hook returns therefore only ever contains
    `"proceed"` entries (for calls it gated this turn) -- purely
    documentary, since `run_agent_loop` already treats any name this hook
    doesn't mention as `"proceed"` by default; returning `{}` when nothing
    needed gating is exactly as correct as omitting entries would be.
    Non-MCP calls are untouched either way: `pending_gate_for` returns
    `None` for any name the provider doesn't own, so they never enter
    `pending` and are never mentioned in the returned map.

    Args:
        provider: This run's already-composed `MCPToolProvider` (P5-T6:
            built and `compose_catalog()`-ed by the caller on the main
            loop before the run's worker thread starts).
        request_mcp_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run -- runs on the
            agent bridge's worker thread and blocks until the batch is
            decided, cancelled, or times out (T5).

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """

    def review_tool_calls(calls: list["ToolCall"], run_id: str) -> dict[str, str]:
        # I3: clear THIS turn's stamps FIRST, before pending_gate_for/the
        # approval round trip even run -- subsumes the `if not pending`
        # branch's own clear below (every invocation of this hook clears,
        # unconditionally). See this function's own docstring for why the
        # clear must happen at entry, not only after a successful round
        # trip: a raising `request_mcp_approvals` must never leave a stale
        # prior-turn stamp live for the fail-open runtime to hand straight
        # to `invoke()`. PR2a Task 5: that clear is scoped to `run_id` --
        # it still wipes THIS run's prior turn, and no longer wipes a
        # concurrent sibling's live verdicts.
        provider.apply_batch_decisions(run_id, {})
        pending = _collect_mcp_pending(provider, calls)
        if not pending:
            return {}
        decisions = request_mcp_approvals(pending)
        provider.apply_batch_decisions(run_id, decisions)
        return {call.llm_name: "proceed" for call in pending}

    return review_tool_calls


def build_tool_review_hook(
    builtin_gate: "BuiltinToolGate",
    builtin_provider: "BuiltinToolProvider",
    mcp_provider: MCPToolProvider | None,
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    workspace_id: str | None = None,
    kill_switch: Callable[[], bool] | None = None,
) -> Callable[[list["ToolCall"]], dict[str, str]]:
    """Build THIS run's run-level `review_tool_calls` hook (P5-T6/task-545).

    TASK-631: when ``kill_switch`` reports on, EVERY call in the batch is
    refused here, without prompting -- the runtime turns any non-"proceed"
    verdict into the call's result and skips dispatch. MCP composition is
    already skipped and ``BuiltinToolGate.check`` already refuses with the
    switch on, but names neither provider claims (skills,
    ``spawn_subagent``, ``find_tools``, ``load_tools``) used to pass
    through unreviewed and RUN NORMALLY -- the switch's label promises
    "block tool calls in chat", and this hook is the one place every
    parsed call passes, so this is where the promise is kept. Read fresh
    per turn (a callable, not a bool) so flipping the switch mid-run takes
    effect on the next batch.

    Unlike `build_mcp_review_hook`, this is wired UNCONDITIONALLY -- every
    run gets one, even a user with no MCP servers configured at all --
    because built-in tools (calculator/datetime today, more later) must be
    gated regardless of whether MCP happens to be composed this turn.
    `BuiltinToolProvider.invoke` already enforces the gate as defense in
    depth, but without this hook the ONLY review a built-in call would ever
    get is that per-call fallback -- never the batched, one-card-per-turn
    review MCP calls already get, and never a chance to ask before
    dispatch for calls this hook doesn't stamp.

    Routing per call, MCP first: `mcp_provider.pending_gate_for` (when a
    provider was composed this run) is asked before the built-in provider,
    so a name that provider actually owns is never mistakenly re-resolved
    against the built-in side too. Note this hook's own precedent is the
    OPPOSITE of `console_agent_bridge._non_colliding_mcp_names`, which
    resolves a name collision the other way -- it drops the colliding MCP
    name from the run's registry so the built-in wins composition. That
    inconsistency is moot in practice: `MCP/tool_naming.py:106` always
    mints MCP tool names as `mcp__<server>__<tool>`, which can never equal
    a bare built-in name like `calculator`/`get_current_datetime`, so no
    call is ever ambiguous between the two orders. A name neither provider
    claims (a skill, `spawn_subagent`, `find_tools`, ...) passes through
    unreviewed, exactly as it does for `build_mcp_review_hook` today.

    Built-in rows use `server_key=BUILTIN_TOOL_SERVER_KEY`
    (`"agent:builtin"`), `server_label="Built-in"`, and `reason=
    "risk_floored"` when `EffectiveToolState.risk_floored` else `"ask"`
    (built-ins never set `config_changed` -- see `resolve_builtin_state`'s
    own docstring for why). Every built-in row's `path_precheck_failed`
    (TASK-1231/F3 AC2) is set via `Tools.file_operation_tools.
    path_precheck_failed`: for `read_file`/`list_directory`/`write_file`
    this pre-flights the SAME `allowed_file_roots`/`validate_path_multi`
    check `invoke()` runs at dispatch, so the approval card can warn the
    user this exact call will fail even if approved -- it never gates or
    auto-denies; `False` for every other builtin tool and every MCP row.
    Only a resolved `"ask"` state ever produces a row: `"allow"` never
    prompts, and `"deny"` is refused outright by
    `invoke()`'s own gate WITHOUT ever reaching the user -- a tool the
    operator switched Off must not appear on the approval card at all.
    Nor does an `"ask"` tool that already has a live session approval
    (`builtin_gate.is_session_approved(name)`) -- review finding 1
    (T6 review): `resolve()`/`resolve_builtin_state` read the permission
    store ONLY, never session approvals, so without this check a user who
    picked "Approve for session" on turn 1 would be re-prompted on turn 2
    even though `invoke()`'s own `check()` already honors that same
    session approval and would execute it anyway. Mirrors MCP's own
    `pending_gate_for`, which applies the identical
    `_is_session_approved_safe` skip for exactly this reason.

    `options=("approve_once", "approve_session", "deny")` -- deliberately
    excluding ONLY `"always_allow"` (verified at
    `Agents/mcp_tool_provider.py:556-564`: `always_allow` is the sole
    PERSISTENT write via `set_tool_state`; `approve_session` is an
    in-memory session cache and `deny`/`timeout` are turn-scoped refusals
    that persist nothing). `"deny"` MUST stay offered -- an earlier draft
    of this design mistakenly dropped it too, which would have made a
    built-in row impossible to refuse from the card at all (the bulk "Deny
    all" button would silently leave it on whatever the row's default
    was).

    Mirrors `build_mcp_review_hook`'s I3 clear-at-entry discipline, extended
    to the built-in side: `builtin_gate.begin_turn(run_id)` runs FIRST,
    unconditionally -- before the MCP stamp clear, before any
    `pending_gate_for`/`resolve` call, before the `request_approvals` round
    trip -- so a raising round trip can never leave a stale built-in stamp
    (or a stale cached permission payload) live for the next turn to
    consume. `mcp_provider.apply_batch_decisions(run_id, {})` follows the
    same reasoning for the MCP side, only when a provider was actually
    composed this run.

    PR2a Task 5: every one of those mutations is scoped to `run_id`, the
    second argument this hook now receives (`AgentService` binds its own
    run id into the callable it hands `LoopDeps`). The gate and the MCP
    provider are shared by a parent run and every sub-agent it spawns, so
    an unscoped clear here wipes -- and an unscoped stamp overwrites --
    verdicts another run in the tree has already been granted and has not
    yet consumed. It still clears THIS run's own previous turn, which is
    what the I3 discipline above requires.

    Exactly ONE `request_approvals` round trip is made per turn, carrying
    BOTH the MCP and built-in pending rows together -- never one call per
    owner. Decisions are then applied back to each owner separately:
    `mcp_provider.apply_batch_decisions(run_id, ...)` for MCP rows,
    `builtin_gate.stamp(run_id, name, decision)` for built-in rows. The returned
    verdict map carries "proceed" for approved calls and REFUSAL STRINGS
    for per-call denials (TASK-1861) and kill-switch blocks (TASK-631) --
    the runtime enforces those directly, skipping dispatch. Approvals are
    still left to `invoke()`'s gate on dispatch, which records the audit
    decision.

    Args:
        builtin_gate: THIS run's `BuiltinToolGate` -- the SAME instance
            the run's `BuiltinToolProvider.invoke` checks, so a stamp
            written here is visible there. Two separate instances would
            mean a decision made here is invisible to `invoke()`, silently
            re-prompting (a stamp `invoke()` never sees) or failing closed
            (an approval that never reaches the gate that checks it).
        builtin_provider: THIS run's `BuiltinToolProvider` (only
            `.tool_for(name)` is used here, to resolve a `ToolCall.name`
            to the `Tool` object `builtin_gate.resolve` needs).
        mcp_provider: THIS run's already-composed `MCPToolProvider`, or
            `None` when no MCP tools should be offered this run (no
            service, kill switch on, or composition yielded nothing) --
            the entire point of this hook existing separately from
            `build_mcp_review_hook` is that built-in gating must not
            depend on this being non-`None`.
        request_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run (the name predates
            built-in gating; the method itself is owner-agnostic -- it
            only reads `MCPPendingCall` fields, never assumes MCP
            ownership).
        workspace_id: THIS run's OWN workspace id (round 1 review CRITICAL
            1) -- e.g. `self.store.session_workspace_id(session_id)` --
            threaded into every builtin file-tool row's `path_precheck_
            failed` computation via `Tools.file_operation_tools.
            path_precheck_failed`'s own `workspace_id=` parameter. Must be
            the SAME workspace id `ConsoleAgentBridge.run_reply` resolves
            for this run's real dispatch (`BuiltinToolProvider(workspace_
            id=...)`) -- otherwise the pre-flight can resolve a DIFFERENT
            workspace than the one the call will actually run against
            (e.g. whatever happens to be active in the UI for a parked
            background session), making the warning wrong in either
            direction. `None` (the default) reproduces the pre-existing
            active-workspace fallback for a caller with no session
            context at all; every caller that has a real session id MUST
            resolve and pass its workspace id.

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """

    def review_tool_calls(calls: list["ToolCall"], run_id: str) -> dict[str, str]:
        # PR2a Task 5: every gate mutation below is scoped to `run_id` --
        # the run whose batch this is, supplied by `AgentService` (which
        # binds its own run id into the hook it puts on `LoopDeps`). The
        # gate and provider instances are shared by a parent and every
        # sub-agent it spawns, so an unscoped clear/stamp here reaches
        # verdicts a concurrent sibling has not yet consumed.
        builtin_gate.begin_turn(run_id)
        # TASK-631: the kill switch outranks everything -- no prompting, no
        # stamps, every call refused. Per-call keys where the runtime can
        # address them; an id-less (fence-path) call is refused by NAME,
        # which stops every same-name call -- fail-closed, same reasoning
        # as TASK-1861's refusal fallback.
        if kill_switch is not None:
            try:
                switch_on = bool(kill_switch())
            except Exception:  # noqa: BLE001 -- an unreadable switch fails CLOSED
                logger.opt(exception=True).warning(
                    "build_tool_review_hook: kill-switch read failed; "
                    "refusing this turn's tool calls"
                )
                switch_on = True
            if switch_on:
                if mcp_provider is not None:
                    mcp_provider.apply_batch_decisions(run_id, {})
                return {
                    (str(getattr(call, "call_id", "") or "") or call.name): (
                        KILL_SWITCH_REFUSAL
                    )
                    for call in calls
                }
        if mcp_provider is not None:
            mcp_provider.apply_batch_decisions(run_id, {})

        mcp_pending = (
            _collect_mcp_pending(mcp_provider, calls)
            if mcp_provider is not None
            else []
        )
        mcp_claimed_names = {row.llm_name for row in mcp_pending}

        # Minor (round 1 review): memoize `allowed_file_roots` across every
        # builtin file-tool row THIS batch checks -- `workspace_id` is fixed
        # for the whole call, so a turn with several read_file/write_file
        # rows would otherwise re-hit the workspace registry (a repeat query
        # against `WorkspaceDB`'s held, per-thread connection -- task-3011)
        # once per row.
        # Fresh dict per `review_tool_calls` call -- never reused across
        # turns, so a folder binding added/removed between turns is still
        # picked up on the very next call.
        path_roots_cache: dict[bool, tuple] = {}

        builtin_pending: list["MCPPendingCall"] = []
        for call in calls:
            if call.name in mcp_claimed_names:
                continue
            tool = builtin_provider.tool_for(call.name)
            if tool is None:
                continue  # not ours either -- a skill/native tool, unreviewed
            state = builtin_gate.resolve(tool)
            if state.state != "ask":
                # "allow" never prompts; "deny" is refused outright by
                # invoke()'s own gate -- neither is offered a card.
                continue
            if builtin_gate.is_session_approved(call.name):
                # Review finding 1 (T6 review): already approved for this
                # session -- `invoke()`'s own `check()` will honor it via
                # the identical `is_session_approved` read, so re-asking
                # here would just re-prompt for a decision the user
                # already made. Not added to `builtin_pending` and so
                # never mentioned in the returned verdict map either --
                # exactly as undecided-but-not-needed-this-turn MCP calls
                # already work (see this function's own docstring).
                continue
            builtin_pending.append(
                MCPPendingCall(
                    llm_name=call.name,
                    server_key=BUILTIN_TOOL_SERVER_KEY,
                    tool_name=call.name,
                    server_label="Built-in",
                    arguments=dict(call.args or {}),
                    # Per-call verdict key: lets the user allow one target and
                    # refuse another in the same batch. Empty on the fence
                    # path, where the runtime falls back to the name.
                    call_id=str(getattr(call, "call_id", "") or ""),
                    reason="risk_floored" if state.risk_floored else "ask",
                    options=("approve_once", "approve_session", "deny"),
                    # TASK-1231/F3 AC2: pre-flight the roots check for the
                    # three file tools -- never gates or auto-denies, just
                    # tells the card this specific path is doomed even if
                    # approved (see path_precheck_failed's own docstring).
                    # `workspace_id=workspace_id` (round 1 review CRITICAL
                    # 1): the pre-flight MUST resolve THIS run's own
                    # workspace, never whatever happens to be active in the
                    # UI -- see this function's own docstring.
                    path_precheck_failed=path_precheck_failed(
                        call.name,
                        call.args,
                        workspace_id=workspace_id,
                        roots_cache=path_roots_cache,
                        sandbox_root=getattr(builtin_provider, "sandbox_root", None),
                        sandbox_lease=getattr(
                            builtin_provider,
                            "sandbox_lease",
                            None,
                        ),
                    ),
                )
            )

        all_pending = mcp_pending + builtin_pending
        if not all_pending:
            return {}
        decisions = request_approvals(all_pending)

        def _decision_for(row: "MCPPendingCall") -> str | None:
            """Resolve one row's verdict, per-call id first then name.

            The card now keys verdicts by `call_id` where the runtime can
            address them (so two reads of two files are two decisions), but
            BOTH consumers below are name-keyed by contract:
            `MCPToolProvider.apply_batch_decisions` takes llm_names, and
            `builtin_gate.stamp` records a grant against a tool NAME because
            a session/always grant is per tool, not per call. Resolving here
            keeps the finer-grained card from silently starving them --
            without this, MCP received {} and no gate grant was ever stamped.
            """
            key = str(getattr(row, "call_id", "") or "")
            if key and key in decisions:
                return decisions[key]
            return decisions.get(row.llm_name)

        def _stamps_for(rows: "list[MCPPendingCall]") -> dict[str, str]:
            """Name-keyed stamps for `rows`: approvals win, all-denied denies.

            TASK-1861. A refusal must NOT be stamped against the name when a
            sibling call of the same tool was approved -- the stamp is what
            `invoke()` peeks at, and it cannot express "allow this one,
            refuse that one", so stamping the refusal would also stop the
            call the user allowed. Refusals are enforced per call by the
            verdict map below instead.

            Stamping the approval is safe even with a refused sibling,
            because that sibling is stopped before dispatch and never
            reaches `invoke()`. When EVERY call of a name was refused there
            is no approval to preserve, so "deny" is stamped as defense in
            depth for any path that bypasses the verdict map.
            """
            approvals: dict[str, str] = {}
            denied: set[str] = set()
            for row in rows:
                decision = _decision_for(row)
                if decision is None:
                    continue
                if decision == "deny":
                    denied.add(row.llm_name)
                    continue
                # Per-call rows can disagree on SCOPE, not just allow/refuse,
                # and only one scope per name can be stamped. Taking the last
                # silently downgraded "Approve for session" to "approve once"
                # whenever a later row of the same tool was approved once --
                # dropping the grant the user asked for and re-prompting on
                # the next call. Choosing "for session" on ANY call of a tool
                # is choosing to grant that tool for the session (that is what
                # the control means, and its label says so), so the broadest
                # chosen scope wins.
                current = approvals.get(row.llm_name)
                if current is None or _APPROVAL_SCOPE_RANK.get(
                    decision, 0
                ) > _APPROVAL_SCOPE_RANK.get(current, 0):
                    approvals[row.llm_name] = decision
            stamps = dict(approvals)
            for name in denied:
                stamps.setdefault(name, "deny")
            return stamps

        if mcp_provider is not None:
            mcp_provider.apply_batch_decisions(
                run_id,
                _stamps_for(
                    [r for r in mcp_pending if r.llm_name in mcp_claimed_names]
                ),
            )
        for name, decision in _stamps_for(builtin_pending).items():
            builtin_gate.stamp(run_id, name, decision)

        # The refusal half, enforced HERE rather than through the stamps.
        # The runtime resolves `call_id` before name and turns any
        # non-"proceed" verdict string into that call's result without
        # dispatching it, so this is the only layer that can refuse one
        # target while running another.
        verdicts: dict[str, str] = {row.llm_name: "proceed" for row in all_pending}
        for row in all_pending:
            if _decision_for(row) != "deny":
                continue
            # Prefer the per-call key. A row with no `call_id` -- the fence
            # path, or an MCP row whose provider omitted an id -- can only be
            # addressed by name, which stops every same-name call in the
            # batch. That is fail-closed, and the only honest option when the
            # runtime cannot tell those calls apart.
            key = str(getattr(row, "call_id", "") or "") or row.llm_name
            verdicts[key] = USER_DENIED_REFUSAL.format(name=row.llm_name)
        return verdicts

    return review_tool_calls


def build_local_review_hook(
    provider: "LocalToolProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"]], dict[str, str]]:
    """Build this run's review_tool_calls hook for the local provider.

    Identical discipline to build_mcp_review_hook (see its docstring for
    the full rationale -- every binding point applies unchanged here):
    clear-first stamps at entry (I3: a raising approval round trip must
    never leave a stale prior-turn stamp live for the fail-open runtime
    to hand to `invoke()`), exactly ONE approval round trip per batch,
    and verdicts only ever "proceed" -- `LocalToolProvider.invoke()`
    single-sources refusals (pinned LOCAL_* refusal strings) and the
    persistence side effects of approve_session/always_allow stamps, so
    this hook never returns a refusal string itself. Calls the provider
    doesn't own resolve `None` from `pending_gate_for` and never enter
    the batch.

    Args:
        provider: This run's already-composed `LocalToolProvider` (built
            by `_compose_local_provider` on the main loop before the
            run's worker thread starts).
        request_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run -- the same
            approval-card bridge the MCP hook uses; it consumes
            `MCPPendingCall` payloads regardless of origin.

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """

    def review_tool_calls(calls: list["ToolCall"], run_id: str) -> dict[str, str]:
        # I3: clear THIS turn's stamps FIRST -- see build_mcp_review_hook.
        # PR2a Task 5: scoped to `run_id`, so the clear cannot reach a
        # concurrent sibling run's live verdicts.
        provider.apply_batch_decisions(run_id, {})
        pending: list["MCPPendingCall"] = []
        for call in calls:
            gate = provider.pending_gate_for(call.name, call.args)
            if gate is not None:
                pending.append(gate)
        if not pending:
            return {}
        decisions = request_approvals(pending)
        provider.apply_batch_decisions(run_id, decisions)
        return {call.llm_name: "proceed" for call in pending}

    return review_tool_calls


def build_combined_review_hook(
    hooks: list[Callable[[list["ToolCall"]], dict[str, str]]],
) -> Callable[[list["ToolCall"]], dict[str, str]]:
    """Fan one batch through every provider's hook; merge verdict maps.

    Each hook gates only the calls its provider owns (pending_gate_for
    returns None for foreign tools), so merging is collision-free --
    except when two providers own the SAME name (not possible today:
    local names carry fs_/web_/todo_ prefixes, MCP names mcp__*), where
    the later hook's "proceed" would simply win; both stamps are still
    applied by each provider's own hook regardless.

    I3 across providers: every hook runs even when an earlier one RAISES.
    `run_agent_loop` fails the batch OPEN on hook exception
    (agent_runtime.py:367-376), and each hook's clear-first stamp wipe is
    the only thing standing between a stale prior-turn stamp and the
    fail-open runtime handing it to `invoke()`. A naive sequential loop
    would let one hook's raising approval round trip (the documented I3
    mid-shutdown case) skip every LATER hook -- including its entry clear
    -- stranding that provider's stale stamp. So each hook is invoked
    under its own try/except and the FIRST exception is re-raised after
    all hooks have run: every provider gets its clear (and, when its own
    round trip succeeds, its fresh this-turn decisions), and the runtime
    still sees the raise and applies its fail-open policy against stamps
    that are guaranteed non-stale.

    Args:
        hooks: The per-provider review hooks to fan each batch through,
            in application order.

    Returns:
        A `review_tool_calls`-shaped callable that merges every hook's
        verdict map into one.
    """

    def review_tool_calls(calls: list["ToolCall"], run_id: str) -> dict[str, str]:
        verdicts: dict[str, str] = {}
        first_exc: Exception | None = None
        for hook in hooks:
            try:
                verdicts.update(hook(calls, run_id))
            except Exception as exc:  # noqa: BLE001 -- re-raised after ALL hooks ran
                logger.opt(exception=True).warning(
                    "combined review_tool_calls: a provider hook raised; "
                    "running remaining hooks so their entry clears still fire"
                )
                if first_exc is None:
                    first_exc = exc
        if first_exc is not None:
            raise first_exc
        return verdicts

    return review_tool_calls


def _split_skill_command_word(text: str) -> tuple[str, str]:
    """Split a ``$word rest`` string into its leading token and the remainder.

    Mirrors ``console_command_grammar._split_leading_token``'s single-
    whitespace-character split rule. That helper is module-private (by
    design -- callers own their own tokenization per its module docstring),
    so this is a deliberate small duplicate rather than an import, the same
    precedent ``ConsoleSkillController._split_console_skill_name_args``
    already follows. ``text`` is assumed to already start with
    `MENTION_SIGIL` (the `$`-mention leading form, not `COMMAND_PREFIX`'s
    `/` -- its sole caller is `_apply_skill_substitution`'s leading-form
    branch).
    """
    for index, character in enumerate(text):
        if character.isspace():
            return text[:index], text[index + 1 :]
    return text, ""


def _render_skill_bundle_block(results: Iterable[Mapping[str, Any]]) -> str:
    """Render one combined "Bundled files" block for a turn's bound skills.

    Task 5 (skills-fork-reachability): `_apply_skill_substitution` builds
    this as pure string work from `execute_skill` results it already holds
    -- no re-execution, no extra service calls -- for every skill actually
    bound this turn (leading-resolved, or embedded mentions that spliced).
    `run_reply` (never here) is the only place that ever appends the
    returned string to a message, so plain sends and the stored transcript
    never see it.

    Row format matches `_BridgeSkillRunner.run`'s own bundle-pointer block
    byte-for-byte (Task 4) -- ``{path} ({size} bytes)`` / ``{path} ({size}
    bytes, binary)``, comma-joined under one combined header -- so a bound
    skill's `skill_file` reads look identical whether granted turn-side
    (this function) or fork-side (a spawned skill reading its own bundle).

    Args:
        results: `execute_skill` result mappings for the bound skills, in
            any order; a result missing `reference_files` (absent when a
            skill has no bundle beyond SKILL.md) or with it empty
            contributes no rows.

    Returns:
        The combined block, or ``""`` when no result carries any rows.
    """
    rows: list[str] = []
    for result in results:
        refs = result.get("reference_files") if isinstance(result, Mapping) else None
        if not refs:
            continue
        rows.extend(
            f"{ref['path']} ({ref['size']} bytes"
            f"{'' if ref.get('is_text', True) else ', binary'})"
            for ref in refs
        )
    if not rows:
        return ""
    return "Bundled files (readable via skill_file): " + ", ".join(rows)


def _is_empty_transcript_row(message: ConsoleChatMessage) -> bool:
    """Whether ``message`` is a committed voice turn whose transcript was empty.

    task-2391: the realtime loop persists such a row with a real, non-blank
    placeholder as its CONTENT ("(no speech detected)") -- durable rows need
    real content, since the DB layer refuses one with neither text nor an
    image. That placeholder is UI chrome the app wrote so the row could
    exist at all, not something the user said, so every builder that walks
    the transcript to construct a request TO A MODEL must treat it as
    absent rather than as a real turn: ``_provider_message_payloads`` (the
    ordinary send/retry/edit/fork/regenerate path), ``summarize_up_to``'s
    span (feeds the summarizer), and ``impersonate_user_reply``'s
    transcript (asks a model to draft text "in the user's voice" from this
    exact history -- arguably the most dangerous place to leave it in) --
    the same exclusion the realtime reseed builder already applies at
    reconnect (``ChatScreen._console_realtime_seed_items``). A HUMAN-facing
    read of the row (the transcript itself, a single-message "Save as
    Note/Media/Prompt" export the user explicitly selected) is NOT a use of
    this helper -- the placeholder is honest, readable text there, and
    hiding it would defeat this task's own AC#1.

    Args:
        message: A transcript row to test. Both ``.metadata`` and
            ``.metadata.transcript_status`` are read via ``getattr`` (never
            a plain attribute access), so a narrow test double that duck-
            types only some fields -- several already exist in this
            codebase, and this helper runs on every row of three model-
            facing send paths -- returns False rather than raising
            ``AttributeError`` on an attribute it never declared.

    Returns:
        True when the row's ``metadata.transcript_status`` is ``"empty"``.
        Every non-realtime row (the overwhelming majority) has no metadata
        at all and returns False without inspecting content.
    """
    metadata = getattr(message, "metadata", None)
    return getattr(metadata, "transcript_status", None) == "empty"


class ConsoleProviderGatewayProtocol(Protocol):
    """Provider gateway surface required by the Console controller."""

    async def resolve_for_send(self, selection: ConsoleProviderSelection) -> Any:
        """Resolve provider readiness for a send."""

    async def stream_chat(
        self,
        resolution: Any,
        messages: list[dict[str, Any]],
        signals: ConsoleProviderStreamSignals | None = None,
    ) -> Any:
        """Stream response chunks for provider messages."""


@dataclass(slots=True)
class ConsoleCitationRepairSession:
    contract: CitationRepairContract | None
    resolution: ConsoleProviderResolution | None
    attempt_started: bool = False
    selection_committed: bool = False
    phase: str = "initial_streaming"
    cancel_reason: Literal["user", "session_close", "shutdown"] | None = None

    def clear_governed_state(self) -> None:
        """Release request content and provider configuration after cleanup."""
        self.contract = None
        self.resolution = None


@dataclass(frozen=True, slots=True)
class ConsoleCitationSelectionOutcome:
    selected_body: str
    state: Literal["bypassed", "valid", "repaired", "unavailable", "canceled"]


@dataclass(frozen=True, slots=True)
class ConsolePreparationOutcome:
    """Immutable result of one automatic Library preparation attempt."""

    preparation_id: str
    attempt_id: str
    state: ConsoleTurnPreparationState
    evidence_bundle: EvidenceBundle | None
    contribution: LibraryPreparationContribution | None
    error_code: str | None


@dataclass(slots=True)
class _PreparedEvidenceLease:
    """Live-only exact staged launch held until the turn is accepted."""

    launch: Any
    release: Callable[[Any, Any], None] | None = None
    capture_result: Any | None = None
    released: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedSendContinuation:
    """Bounded volatile inputs needed to continue one admitted send."""

    preparation_id: str
    attachments: tuple[PendingAttachment, ...]
    prefill: str | None
    prefill_from_one_shot: bool
    one_shot_prefill_revision: int | None
    staged_evidence_frozen: bool
    staged_evidence: _PreparedEvidenceLease | None


@dataclass(frozen=True, slots=True)
class _DurablePostcommitContinuation:
    """App-lifetime inputs for idempotent postcommit re-entry."""

    preparation_id: str
    fingerprint: ConsoleDurableAcceptanceFingerprint
    session_id: str
    origin: ConsoleSubmissionOrigin
    queue_entry_id: str | None
    clean_draft: str
    commit: ConsoleDurableTurnCommit
    echoed_user_id: str
    resolution: ConsoleProviderResolution
    provider_messages: list[dict[str, Any]]
    prefill: str | None
    prefill_from_one_shot: bool
    one_shot_prefill_revision: int | None
    skill_bindings: tuple[Any, ...]
    skill_bundle_block: str | None
    citation_repair_session: Any | None
    turn_context: ConsoleTurnExecutionContext
    prepared: _PreparedSendContinuation | None
    committed_context_epoch: int
    stream_signals: ConsoleProviderStreamSignals
    #: TASK-22302: the durable turn commits in `_accept_durable_turn` and
    #: publishes its live owners in `resume_durable_postcommit`; the terminal
    #: citation finalizer has to survive that hand-off. It did not -- the
    #: publish site passed a hard-coded None -- so no durable Console turn
    #: persisted any citation provenance from `a26cdafd8` onward.
    terminal_citation_finalizer: TerminalCitationFinalizer | None = None


@dataclass(frozen=True, slots=True)
class CapturePolicySnapshot:
    """Future and active capture policy for one immutable Console session."""

    session_id: str
    conversation_id: str | None
    conversation_title: str
    enabled: bool
    next_detail: CaptureDetail | None
    conversation_detail: CaptureDetail | None
    global_detail: CaptureDetail
    effective: CapturePolicyResolution
    policy_revision: int
    config_generation: int
    capture_revision: int
    active_run_detail: CaptureDetail | None
    queued_consumer: bool
    save_pending: bool
    error_code: str | None


class CapturePolicyMutationStatus(str, Enum):
    APPLIED = "applied"
    SAFE_SESSION_ONLY = "safe_session_only"
    STALE = "stale"
    TARGET_MISSING = "target_missing"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CapturePurgeAvailability:
    can_purge: bool
    reason_code: str | None = None


class CapturePurgeStatus(str, Enum):
    DELETED = "deleted"
    BLOCKED = "blocked"
    STALE = "stale"
    FAILED = "failed"


_MISSING_CAPTURE_REVISION = -1


@dataclass(frozen=True, slots=True)
class CapturePurgeResult:
    status: CapturePurgeStatus
    removed_count: int
    capture_revision: int
    reason_code: str | None = None

    @classmethod
    def blocked(cls, revision: int, reason_code: str) -> "CapturePurgeResult":
        return cls(CapturePurgeStatus.BLOCKED, 0, revision, reason_code)

    @classmethod
    def deleted(cls, removed_count: int, revision: int) -> "CapturePurgeResult":
        return cls(CapturePurgeStatus.DELETED, removed_count, revision)


@dataclass(frozen=True, slots=True)
class CapturePolicyMutationResult:
    status: CapturePolicyMutationStatus
    snapshot: CapturePolicySnapshot
    retryable: bool
    reason_code: str | None
    config_result: ConfigMutationResult | None = None


@dataclass(frozen=True, slots=True)
class _DispatchRetryContext:
    """Freshly revalidated inputs for one explicit recovery retry."""

    resolution: Any
    authority: ConsoleTurnLibraryAuthority
    destination: ConsoleResolvedDestination
    provider_messages: list[dict[str, Any]]
    turn_context: ConsoleTurnExecutionContext


class _DispatchRecoveryRefusal(RuntimeError):
    """Bounded user-visible refusal raised before recovery provider entry."""


@dataclass(frozen=True)
class ConsoleSubmitResult:
    """Result returned to the composer after a Console submit attempt."""

    accepted: bool
    should_clear_draft: bool
    visible_copy: str = ""
    #: Task 4 (D2 fix wave): set only by ``_session_closed_result``. The
    #: owning session no longer exists by the time this result is produced
    #: (``ConsoleChatStore.close_session`` already purged it), so there is no
    #: live transcript left to append a SYSTEM row to -- unlike ``_block``/
    #: ``_active_run_rejection``, whose target session is still live. The
    #: screen-side caller (``ChatScreen._submit_console_native_draft``) uses
    #: this flag to show a toast instead, so the outcome is never silent.
    session_closed: bool = False
    session_id: str | None = None
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    terminal_status: ConsoleRunStatus | None = None
    origin: ConsoleSubmissionOrigin | None = None
    queue_entry_id: str | None = None
    committed_context_epoch: int | None = None
    preparation_id: str | None = None
    provider_started: bool = False


@dataclass(frozen=True)
class ImpersonateResult:
    """Outcome of an Impersonate draft request (task-1683 / Qodo #1160).

    Attributes:
        text: The drafted user reply, or "" when nothing was produced.
        reason: "" on success, else one of ``provider-not-ready``,
            ``empty-transcript``, ``provider-error``, ``empty-completion``.
        detail: Optional provider-supplied copy for the blocked case.
    """

    text: str
    reason: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _CharacterEmoteAuthority:
    """Captured character ownership fence for one provider dispatch."""

    identity_revision: int
    runtime_backend: str
    assistant_id: str | None
    assistant_authority_id: str | None
    local_character_id: int | None


class _CharacterEmoteAuthorityChanged(RuntimeError):
    """The owning character identity changed during the off-thread read."""


@dataclass(frozen=True, slots=True)
class _LightweightProviderHistoryRow:
    """Pre-serialization provider row retaining only admitted media references."""

    source_message_id: str
    role: str
    text: str
    attachments: tuple[MessageAttachment, ...] = ()


class ConsoleChatController:
    """Coordinate native Console chat state between store and provider gateway."""

    #: TASK-21145 (UAT H-3): "Validating provider." must always reach a
    #: terminal state — the UAT run sat on it 30s+ with no error, no retry,
    #: and no way forward. Generous enough for a slow first TLS handshake;
    #: finite so the composer never wedges.
    PROVIDER_VALIDATION_TIMEOUT_SECONDS = 30.0

    async def _resolve_for_send_bounded(self, selection: Any) -> Any:
        """resolve_for_send with a hard deadline (UAT H-3).

        Returns:
            The gateway resolution, or a not-ready stand-in carrying
            actionable timeout copy. Cancellation propagates untouched.
        """
        from types import SimpleNamespace

        try:
            return await asyncio.wait_for(
                self.provider_gateway.resolve_for_send(selection),
                timeout=self.PROVIDER_VALIDATION_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return SimpleNamespace(
                ready=False,
                visible_copy=(
                    "Provider validation timed out. Check the server or "
                    "your connection, then try again."
                ),
            )

    def __init__(
        self,
        *,
        store: ConsoleChatStore,
        provider_gateway: ConsoleProviderGatewayProtocol,
        provider: str = "llama_cpp",
        model: str | None = None,
        configured_model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        verbosity: str | None = None,
        thinking_effort: str | None = None,
        thinking_budget_tokens: int | None = None,
        streaming: bool = True,
        system_prompt: str | None = None,
        agent_bridge: "ConsoleAgentBridge | None" = None,
        agent_runtime_enabled: bool = True,
        skills_service: Any | None = None,
        skill_substitution_enabled: bool = True,
        chat_dictionary_applier: "Callable[[str | None, str], str] | None" = None,
        world_info_applier: "Callable[[str | None, str, list], str] | None" = None,
        rag_capture_provider: "Callable[..., Awaitable[Any]] | None" = None,
        default_session_settings: "Callable[[], ConsoleSessionSettings] | None" = None,
        library_provider_factory: "Callable[..., Any | None] | None" = None,
        global_user_display_name: Callable[[], str] | None = None,
        context_repository: ConsoleContextRepository | None = None,
        turn_context_provider: "Callable[[str], ConsoleTurnConfigurationSnapshot] | None" = None,
        queued_staged_rider_provider: "Callable[[str], bool] | None" = None,
        provider_config: "Callable[[], Mapping[str, Any]] | None" = None,
        confirm_project_instruction_dispatch: Callable[
            [ProjectInstructionDispatchNotice], Literal["proceed", "cancel", "disable"]
        ]
        | None = None,
        select_project_instruction_binding: Callable[
            [str, tuple[ProjectInstructionBindingSelection, ...], str],
            Awaitable[tuple[Literal["select", "disable", "cancel"], str | None]],
        ]
        | None = None,
        buddy_sink: "PersonaBuddyConsoleAdapter | None" = None,
        scratch_spaces: ConsoleScratchSpaceManager | None = None,
        staged_evidence_provider: Callable[[str], bool] | None = None,
        library_preparation_timeout: float = 5.0,
    ) -> None:
        self.store = store
        self.provider_gateway = provider_gateway
        self.provider = provider
        self.model = model
        self.configured_model = configured_model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.seed = seed
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.verbosity = verbosity
        self.thinking_effort = thinking_effort
        self.thinking_budget_tokens = thinking_budget_tokens
        self.streaming = streaming
        self.system_prompt = system_prompt
        self._agent_bridge = agent_bridge
        self._buddy_sink = buddy_sink
        self._owns_scratch_spaces = scratch_spaces is None
        self._scratch_spaces = scratch_spaces or ConsoleScratchSpaceManager()
        self._agent_runtime_enabled = agent_runtime_enabled
        self._skills_service = skills_service
        self._skill_substitution_enabled = skill_substitution_enabled
        self._chat_dictionary_applier = chat_dictionary_applier
        self._world_info_applier = world_info_applier
        self._rag_capture_provider = rag_capture_provider
        self._staged_evidence_provider = staged_evidence_provider
        self._library_preparation_timeout = max(
            0.001, float(library_preparation_timeout)
        )
        self._preparation_outcomes: dict[str, ConsolePreparationOutcome] = {}
        self._prepared_send_continuations: dict[str, _PreparedSendContinuation] = {}
        self._durable_postcommit_continuations: dict[
            str, _DurablePostcommitContinuation
        ] = {}
        self._provider_config = provider_config
        #: Task 4 (D2 fix wave, "bonus race"): screen-owned callable that
        #: builds a fresh default `ConsoleSessionSettings` snapshot (mirrors
        #: `ChatScreen._default_console_session_settings`, wired by
        #: `_ensure_console_chat_controller`). Used ONLY by `submit_draft`'s
        #: no-`session_id` bootstrap branch, so a session created THAT way
        #: gets real settings from the start instead of `None` -- the mount-
        #: time creator (`_ensure_active_console_session_settings`) already
        #: passes settings when IT creates the first session; this closes
        #: the same gap for the other creator. `None` in most controller-
        #: only tests (that bootstrap path just keeps its pre-fix `None`
        #: settings then, matching every other UI-bridge hook's no-op
        #: default here).
        self._default_session_settings = default_session_settings
        #: task-1337: screen-owned seam that builds THIS run's Library
        #: retrieval provider (the descriptor-backed ``LibraryToolProvider``
        #: when direct Library tools are on, the bounded
        #: ``LibraryRagToolProvider`` when off, or ``None`` when neither can
        #: be constructed). Called exactly once per agent run, on the main
        #: loop, inside ``_run_agent_reply`` -- the bridge worker thread
        #: never reads Textual config itself, and flipping the setting
        #: between runs takes effect on the very next run without
        #: rebuilding this controller or the cached bridge. ``None`` (the
        #: default) means no Library tools are offered at all.
        self._library_provider_factory = library_provider_factory
        self._global_user_display_name = global_user_display_name or (lambda: "User")
        persistence_db = getattr(getattr(store, "persistence", None), "db", None)
        self._capture_policy_repository = getattr(
            store, "capture_policy_repository", None
        ) or (
            ConsoleCapturePolicyRepository(persistence_db)
            if persistence_db is not None
            else None
        )
        self._capture_policy_hydrated: set[str] = set()
        self._capture_policy_hydration_errors: dict[str, str] = {}
        self._visual_identity_repository = (
            VisualIdentityRepository(persistence_db)
            if persistence_db is not None
            else None
        )
        self._context_repository = context_repository
        if self._context_repository is None and persistence_db is not None:
            try:
                self._context_repository = ConsoleContextRepository(persistence_db)
            except Exception as exc:
                self._context_repository = None
                logger.bind(
                    error_type=type(exc).__name__,
                    persistence_db_present=True,
                ).warning("console_context_repository_init_failed")
        self._compaction_service = (
            ConsoleCompactionService(self._context_repository, provider_gateway)
            if self._context_repository is not None
            and callable(getattr(provider_gateway, "complete_auxiliary", None))
            else None
        )
        self._turn_context_provider = turn_context_provider
        self._queued_staged_rider_provider = queued_staged_rider_provider
        self._confirm_project_instruction_dispatch = (
            confirm_project_instruction_dispatch
        )
        self._select_project_instruction_binding = select_project_instruction_binding
        self._project_instruction_display: dict[
            str, ProjectInstructionDisplayMetadata
        ] = {}
        self._project_instruction_activation_events: dict[
            str, list[ProjectInstructionActivationEvent]
        ] = {}
        # Parallel-agents spec §2: run state is a PER-SESSION map, not a
        # single global slot -- two sessions can each have their own
        # in-flight/terminal run without stamping each other. `run_state`/
        # `run_state_history` below become read-only facades over these maps
        # (see the property block right after `__init__`); every WRITE goes
        # through `_set_run_state`/`_clear_terminal_run_state`, which take an
        # explicit `session_id` so a background completion can target its
        # OWNING session instead of whatever the user currently has open.
        self._run_states: dict[str, ConsoleRunState] = {}
        self._run_state_histories: dict[str, list[ConsoleRunStatus]] = {}
        self._buddy_run_owner_context: ContextVar[dict[str, str] | None] = ContextVar(
            "console_buddy_run_owners", default=None
        )
        # Parallel-agents spec §6: run-marker state (Task 7). Both maps are
        # keyed by session id like the run-state maps above, but track
        # marker-only bookkeeping that ``_run_states`` doesn't capture on
        # its own:
        #   - `_pending_approvals`: TASK-1050 (round-keyed accounting):
        #     session id -> the set of outstanding approval-like round ids
        #     (real bridge round/request ids, or the deprecated shim's
        #     `_LEGACY_PENDING_APPROVAL_ROUND_ID` sentinel) currently
        #     blocking that session. A session is "pending" iff it is a KEY
        #     here with a non-empty value set -- `add_pending_round`/
        #     `discard_pending_round` are the ONLY writers and keep that
        #     invariant (an emptied set is popped, never left as `{}`), so
        #     every reader (`run_marker_for`, `fleet_summary_counts`,
        #     plain `in`/`not in` membership tests throughout the test
        #     suite) can keep treating this exactly like the plain
        #     `set[str]` of session ids it used to be. Was a single global
        #     boolean per session (`set_run_pending_approval`) shared by
        #     THREE independent bridges (MCP tool approvals, skill-install
        #     confirms, skill-script confirms) -- whichever bridge's round
        #     finished first cleared the badge even if a SIBLING round from
        #     a different bridge (or a second round from the SAME bridge)
        #     was still outstanding for that session. `set_run_pending_
        #     approval` is now a deprecated boolean shim kept ONLY for
        #     callers that genuinely have no round id of their own (see its
        #     docstring) -- Task 9 originally wired the approval paths that
        #     called it; TASK-1050 migrated all three bridges themselves to
        #     `add_pending_round`/`discard_pending_round` with their real
        #     round/request ids. Named to avoid colliding with the
        #     PRE-EXISTING `self.set_pending_approval` INSTANCE ATTRIBUTE
        #     below (the MCP batch-approval UI callback slot, task-5) -- a
        #     same-named method here would be silently clobbered by that
        #     assignment.
        #   - `_unvisited_outcomes`: sessions whose run reached a terminal
        #     COMPLETED/FAILED status while NOT the active (viewed)
        #     session, stamped by `_set_run_state` and cleared by
        #     `mark_session_visited` (called from `switch_session`). The
        #     viewed session's own terminal transition is seen live and is
        #     deliberately never stamped here.
        self._pending_approvals: dict[str, set[str]] = {}
        self._unvisited_outcomes: dict[str, ConsoleRunMarker] = {}
        #: F2b fix (Qodo wave): guards every mutation of `_pending_
        #: approvals`, `_parked_approval_payloads`, and `_pending_
        #: approval_rounds` -- the three approval-marker collections a
        #: worker thread (`request_mcp_approvals`'s own body/`finally`) can
        #: mutate WHILE the UI thread iterates them every ~0.2s sync tick
        #: (`fleet_summary_counts`). An unguarded set/dict mutation racing
        #: an unguarded iteration over the SAME object can raise
        #: `RuntimeError: Set/dictionary changed size during iteration`.
        #: `_unvisited_outcomes`/`_run_states` are NOT covered here: both
        #: are written only from the main thread (`_set_run_state`,
        #: `mark_session_visited`), never from a worker thread, so they
        #: carry no cross-thread hazard this lock needs to close.
        self._approval_state_lock = threading.Lock()
        # Revision tokens fence destructive lifecycle confirmations without
        # mirroring any activity values. Counts remain derived on demand from
        # ``ConsoleControllerActivity``; these integers only reveal that one
        # of its existing authorities changed while a dialog was open.
        self._lifecycle_revision_lock = threading.Lock()
        self._lifecycle_revision = 0
        self._session_lifecycle_revisions: dict[str, int] = {}
        #: Optional owner hook invoked once a submit is accepted (user message
        #: persisted, run about to start) so the composer can clear immediately
        #: instead of holding the sent text for the whole run.
        self.on_submission_accepted: Callable[[], None] | None = None
        #: Content-free queued counterpart to ``on_submission_accepted``.
        #: It may refresh transcript/queue UI, but cannot clear a composer.
        self.on_queued_submission_accepted: (
            Callable[[ConsoleQueuedAcceptanceEvent], None] | None
        ) = None
        #: TASK-1364: optional shared JSONL prompt-history store, assigned by
        #: the owning screen (mirroring ``on_submission_accepted``). An
        #: ACCEPTED send's cleaned draft is appended here -- never a blocked,
        #: refused, or empty (attachment-only) one -- so the composer's ghost
        #: text and Up/Down recall can offer it later. ``None`` (e.g. in
        #: controller-only tests) disables recording.
        self.prompt_history: PromptHistory | None = None
        self.prompt_queue_registry = ConsolePromptQueueRegistry()
        context_epoch = getattr(self.store, "conversation_context_epoch", None)
        if not callable(context_epoch):
            # Narrow compatibility seam for minimal controller test doubles
            # and embedders that do not exercise queue progression. The real
            # ConsoleChatStore always supplies the authoritative epoch.
            def context_epoch(_session_id: str) -> int:
                return 0

        self.prompt_queue_coordinator = ConsolePromptQueueCoordinator(
            registry=self.prompt_queue_registry,
            context_epoch=context_epoch,
            run_status=lambda session_id: self.run_state_for(session_id).status,
            submit_queued=self._submit_queued_entry,
            has_staged_rider=self._queued_staged_rider_present,
            needs_approval=self.has_pending_approval_round,
            can_reacquire_slot=self._queue_can_reacquire_slot,
            on_queued_accepted=self._notify_queued_submission_accepted,
            on_chain_terminal=self._publish_queue_chain_terminal,
            on_activity_changed=self._note_controller_activity_changed,
        )
        for restored_session in self.store.sessions():
            self._hydrate_dispatch_recovery_queue(restored_session.id, force=True)
        # Task 3b: PER-SESSION maps, mirroring `_run_states`' own keying --
        # two sessions can each have their own in-flight stream/cancel state
        # without clobbering each other. Written/cleared at the SAME
        # lifecycle points the old singulars were (`_stream_assistant_
        # response`/`_run_agent_reply`'s start and `finally`), keyed by the
        # run's OWNING session id (the same `owner_id`/`session_id` locals
        # Task 1 threaded), never by whatever session the user currently has
        # open. `stop_active_run` is the one place that DELIBERATELY reads
        # by the ACTIVE (viewed) session -- see its own docstring.
        self._active_assistant_message_ids: dict[str, str] = {}
        self._active_stream_tasks: dict[str, asyncio.Task] = {}
        # Exact workspace authority captured for each live provider dispatch.
        # Undo/commit probes run on worker threads, so snapshot reads and the
        # stream-lifecycle writes share this lock.
        self._active_workspace_roots_lock = threading.Lock()
        self._active_workspace_roots_by_session: dict[str, tuple[str, ...]] = {}
        # Volatile-only Task-13 owner fence. It starts before submit's first
        # await and ends only after the submit finalizer; Task 14 will add
        # durable recovery/checkpoint semantics.
        self._active_submit_tasks: dict[asyncio.Task, str] = {}
        self._active_submit_preparations: dict[asyncio.Task, str] = {}
        self._active_submit_tasks_lock = threading.RLock()
        self._capture_quiescence_lock = threading.RLock()
        self._capture_exchange_flush_sessions: set[str] = set()
        try:
            self._owner_loop: asyncio.AbstractEventLoop | None = (
                asyncio.get_running_loop()
            )
        except RuntimeError:
            self._owner_loop = None
        self._provider_continuation_recovery_sessions: set[str] = set()
        self._stop_requested = False
        #: F5 fix (Qodo wave): set ONLY by ``shutdown()`` and NEVER reset
        #: (unlike ``_stop_requested``, which every run's own lifecycle
        #: resets to ``False`` -- see ``shutdown``/``_run_agent_reply``/
        #: ``_stream_assistant_response``'s own resets -- making it
        #: race-dependent whether a still-polling bridge thread observes a
        #: Stop that raced a reset). The three worker-thread approval/
        #: confirm bridges (``request_mcp_approvals``, ``request_skill_
        #: install_confirm``, ``request_skill_script_confirm``) OR this
        #: with ``_is_active_session_cancelled()`` at their poll sites
        #: instead of the old, session-agnostic ``_stop_requested`` --
        #: a single session's Stop must never deny another session's
        #: unrelated approval round; only real process teardown (the one
        #: case where every session's run legitimately ends at once) does.
        #:
        #: task-15860 (the lifetime landing): this Event is now
        #: **PER-VISIT**, not per-instance. Its old "never reset" contract
        #: rested on one premise, stated verbatim in ``shutdown``'s
        #: docstring -- "``ChatScreen`` never reuses an instance after
        #: unmounting it". An app-owned runtime falsifies that premise: the
        #: SAME controller now serves visit after visit. So ``leave_console
        #: ()`` sets THIS Event (denying every round armed during the visit
        #: that is ending), and the NEXT ``attach_view`` calls
        #: ``begin_visit()``, which REPLACES the attribute with a fresh,
        #: unset Event. ``shutdown()``/``begin_shutdown()`` keep the old
        #: permanent meaning and additionally set ``_disposed``, after
        #: which ``begin_visit`` refuses to install anything.
        #:
        #: **Because the attribute is replaced, every poll site must have
        #: captured the Event it answers to at ARM time.** A site that
        #: re-read ``self._shutdown_requested`` on each poll would see the
        #: NEXT visit's fresh, unset Event and resurrect a round the
        #: previous visit's teardown already denied. See
        #: ``_bind_visit_cancel_signal``.
        self._shutdown_requested = threading.Event()
        #: The cancellation Event for rounds armed with NO Console visit
        #: open (task-15860, plan Task 5). ``None`` until the first such
        #: round arms.
        #:
        #: ``_shutdown_requested`` answers the question "did the visit that
        #: armed this round end?". A round armed while the runtime is
        #: DETACHED was not armed during any visit, so reading that
        #: (already-set) Event for it answered a different question and
        #: denied the round at the first 1.0s poll -- measured, 1.01s to
        #: ``deny``, with nothing ever surfaced. That is the same category
        #: error the wake-fires landing fixed one layer up, where
        #: ``_attempt`` read "a visit ended" as "the app is exiting".
        #:
        #: This Event stands in for the visit that has not happened yet:
        #: unset while detached (so the round waits for the user to open
        #: Console and answer it) and set by the next ``leave_console()``
        #: -- by then the user HAS seen it and navigated away, which is
        #: exactly the case AC#2's "leaving denies parked approvals" is
        #: about -- or by ``begin_shutdown()`` at app exit. It is dropped
        #: once set, so the next detached round binds a fresh one.
        #:
        #: Nothing here weakens a fail-closed gate: the round still cannot
        #: resolve to anything but a human's own decision, a CONFIGURED
        #: deadline (unchanged, never paused or extended -- ADR-067's
        #: ``<= 0`` default is "no deadline" for the mounted case too),
        #: this run's own cancel event, or these two teardown signals.
        self._headless_visit_cancel: threading.Event | None = None
        #: Whether a Console visit is OPEN on this controller right now
        #: (``begin_visit()`` .. ``leave_console()``). Qodo audit S2
        #: (PR 1752): ``_bind_visit_cancel_signal`` used to infer "no
        #: visit open" from ``_shutdown_requested.is_set()`` -- but a
        #: controller that has NEVER had a visit holds the constructor's
        #: unset Event, so a round armed viewless-from-birth (wake-at-
        #: launch) bound THAT Event, and the first ``begin_visit()``
        #: REPLACED the attribute, orphaning it: neither the next leave
        #: nor ``begin_shutdown`` could ever reach the round. This flag
        #: states the visit lifecycle instead of inferring it. False at
        #: birth on purpose: a controller built lazily DURING a visit
        #: (``attach_view`` ran before the controller existed, so
        #: ``begin_visit`` never fired on it) binds the headless Event --
        #: which both teardown paths set, so its rounds still deny at
        #: leave/exit exactly as before.
        self._visit_open = False
        #: True once this controller has been torn down for good
        #: (``begin_shutdown``). Blocks ``begin_visit`` from ever handing
        #: a disposed controller a fresh, unset cancellation Event.
        self._disposed = False
        #: Sessions running an ``AGENT_WAKE`` turn right now. ``leave_
        #: console()`` skips them: the owner ruled that cancelling an
        #: in-flight wake turn re-creates the exact "only completes if you
        #: stay" gap this arc exists to close, and a wake turn is
        #: structurally the same class of work as the fleet survivor AC#2
        #: already keeps running. ``shutdown()`` (app exit) still takes
        #: everything.
        self._agent_wake_turn_sessions: set[str] = set()
        # Rebase note (dev citation-repair vs. Task 3b): dev added this as a
        # singular slot (no per-session awareness); rescoped here the same
        # way as the two maps above -- keyed by the run's OWNING session id,
        # so a background session's in-flight repair can never be read/
        # cleared by another session's close/stop/teardown path.
        self._active_citation_repair_sessions: dict[
            str, ConsoleCitationRepairSession
        ] = {}
        self._original_attempts: OrderedDict[str, str] = OrderedDict()
        #: Per-run cancellation flag for the agent bridge's background
        #: thread (see ``_run_agent_reply``), keyed by owning session id
        #: like the two maps above. ``threading.Event`` rather than a
        #: shared bool: ``asyncio.to_thread`` survives Task cancellation
        #: (the coroutine detaches from the still-running OS thread), so
        #: the closure handed to that thread must observe a signal that,
        #: once set, is never reset for THIS run -- unlike
        #: ``_stop_requested``, which the run's own ``finally`` block
        #: resets as soon as the coroutine side is done (task-227).
        #: ``_stop_requested`` itself stays a single shared flag (Task 3b
        #: did not rescope it) -- see ``_is_active_session_cancelled``'s
        #: docstring for the resulting, deliberately-scoped-down, limit on
        #: the three worker-thread approval/confirm bridges below.
        self._active_cancel_events: dict[str, threading.Event] = {}
        self._active_capture_details: dict[str, CaptureDetail] = {}
        # Cost-ticker PR3: per-session cache-break/TTL ground truth for the
        # cost chip. All three are process-local and best-effort -- a missed
        # write means a stale chip, never a broken send.
        #   - `_payload_fingerprint_baselines`: the fingerprint of the
        #     payload actually dispatched on this session's most recent
        #     send, recorded at the SAME pre-compaction stage
        #     `compute_current_fingerprint` recomputes from, so the two are
        #     always comparable (see `_stream_assistant_response_inner`).
        #   - `_cache_warm_until`: monotonic deadline the Anthropic prompt
        #     cache is expected to stay warm until, stamped only after a
        #     send that actually showed cache activity (see
        #     `_attach_stream_usage`).
        #   - `_cache_last_activity`: whether the session's most recent
        #     Anthropic send reported any cache read/write at all.
        self._payload_fingerprint_baselines: dict[str, PayloadFingerprint] = {}
        self._cache_warm_until: dict[str, float] = {}
        self._cache_last_activity: dict[str, bool] = {}
        #: The composed MCP provider for the current agent run, captured
        #: on the main loop in ``_run_agent_reply`` so ``build_context_snapshot``
        #: can read tool metadata later without recomposing.
        self._mcp_provider: Any | None = None

        # -- MCP batch-approval bridge (task-5) ------------------------------
        #: Textual App-like object exposing ``call_from_thread`` -- assigned
        #: by the owning screen (``ChatScreen._ensure_console_chat_
        #: controller``), mirroring how ``on_submission_accepted`` is wired.
        #: ``None`` (e.g. in most existing controller-only tests) makes
        #: ``request_mcp_approvals`` a safe no-op UI bridge that still
        #: resolves via cancellation/timeout.
        self.app: Any | None = None
        #: UI-thread callback that pushes/clears the pending-approval batch
        #: into the owning screen's task-resume state (``ChatScreen.
        #: _set_console_pending_approval``). Always invoked through
        #: ``self.app.call_from_thread`` from ``request_mcp_approvals``.
        self.set_pending_approval: Callable[[dict[str, Any] | None], None] | None = None
        #: Task 9 (parked background approvals): UI-thread callback invoked
        #: (via ``self.app.call_from_thread``) when ``request_mcp_approvals``
        #: raises a round for a NON-active session -- sets the fleet
        #: pending-approval badge and fires the one-per-card toast, WITHOUT
        #: touching ``set_pending_approval``'s mounted-card slot (that stays
        #: reserved for whichever session is actually being viewed). Wired
        #: to ``ChatScreen._park_console_approval`` by ``_ensure_console_
        #: chat_controller``, mirroring ``set_pending_approval``'s own
        #: wiring. ``None`` in most controller-only tests, matching every
        #: other UI bridge slot here.
        self.park_pending_approval: Callable[[str], None] | None = None
        #: Task 10 (background completion toasts): UI-thread callback
        #: invoked DIRECTLY (never via ``self.app.call_from_thread`` --
        #: unlike the two bridges above, every terminal ``_set_run_state``
        #: call already runs on the main event-loop thread: worker-thread
        #: agent runs resume here only after ``await asyncio.to_thread(...)``
        #: returns in ``_run_agent_reply``) from ``_set_run_state``'s
        #: non-active COMPLETED/FAILED branch, once per transition INTO a
        #: terminal state. Wired to ``ChatScreen._notify_console_run_
        #: outcome`` by ``_ensure_console_chat_controller``, mirroring
        #: ``park_pending_approval``'s wiring and reusing its exact
        #: session-title/workspace-name resolution
        #: (``ConsoleWorkspaceController._console_workspace_display_name``). ``None`` in
        #: most controller-only tests, matching every other UI bridge slot
        #: here.
        self.notify_run_outcome: Callable[[str, ConsoleRunStatus], None] | None = None
        #: PR3a-1 Task 6b (audit F3): per session, the turn's provider
        #: signals object plus HOW MANY of its usage payloads were attached
        #: to the assistant message. A fleet child keeps streaming into the
        #: SAME signals object after `run_reply` returns, and the agent path
        #: attaches usage exactly ONCE (`_finalize_agent_reply`), so every
        #: payload closed out after that instant is real money billed to the
        #: user and read by nobody. This slot is what makes the difference
        #: readable (`unattributed_fleet_tokens`) instead of silent.
        self._post_turn_usage_watch: dict[str, tuple[Any, Any, int]] = {}
        #: PR3a-2 Task 3 (tasks 15660/15667): per ORIGINATING ASSISTANT
        #: MESSAGE, the turn's signals object + resolution + the ``partial``
        #: flag its own attach used -- what the last-child-settled fold
        #: recomputes from. Keyed by message (not session) because a later
        #: turn in the same session REPLACES the session watch above while
        #: an earlier turn's survivor is still billing into ITS OWN signals
        #: object. Recorded by ``_watch_post_turn_usage`` only when the
        #: bridge says the conversation is still owed a drain
        #: (``has_unsettled_children``); popped by the fold, so entries
        #: never outlive the drain that consumes them. Money over memory on
        #: every ambiguity: a broken/absent check records anyway.
        self._fleet_usage_reattach_sources: dict[str, tuple[Any, Any, bool]] = {}
        #: The loop ``_attach_stream_usage`` normally runs on (the app's
        #: asyncio loop -- every turn-end attach happens there), captured
        #: at watch time. The drain consumer fires on the CHILD's thread,
        #: possibly after the Console screen is gone; it hops here so the
        #: store mutation always runs on the thread that owns the store.
        #: The app loop outlives the screen, so the hop works after
        #: teardown too (proven by the teardown test in
        #: ``test_fleet_usage_reattach.py``).
        self._usage_reattach_loop: asyncio.AbstractEventLoop | None = None
        # Register the fold as a fan-out consumer NOW, next to bridge
        # attachment -- never from `run_reply` (bridge-lifetime registry;
        # see `FleetDrainFanout.register` for why).
        self._register_fleet_usage_reattach(agent_bridge)
        #: PR3a-2 Task 5: the auto-wake coordinator (pending completions,
        #: gating, delivery). Constructed unconditionally so `_set_run_
        #: state`'s terminal retry hook always has it; registered on the
        #: bridge fan-out next to the usage fold above.
        self._fleet_wake = ConsoleFleetWakeCoordinator(self)
        #: PR3a-2 Task 5, user-wins-ties: screen-wired probe returning
        #: True while the USER holds a claim on sending (a non-empty
        #: composer draft -- which also covers the dispatch gap, since the
        #: composer clears only on ACCEPTED manual sends). A due wake
        #: defers while it reports (or raises -- user wins on uncertainty)
        #: and is retried on the next trigger. ``None`` = no probe wired
        #: (headless/tests): no user claim to lose to.
        self.wake_user_priority_probe: Callable[[str], bool] | None = None
        #: task-15971: screen-wired ``(conversation_id, session_id) ->
        #: bool`` probe the delivery COMMIT consults -- True only while
        #: this conversation is actually being viewed (the screen is the
        #: DISPLAYED one and the session is the active one). A wake that
        #: completes off-view leaves the FLEET_UNSEEN mark set so the ◈
        #: badge points at the delivered result (the coordinator's design
        #: ruling: off-screen delivery is intended for a mounted-but-
        #: hidden Console; the user must still LEARN of it). ``None`` =
        #: unwired (controller doubles): the historical clear-on-delivery
        #: stands.
        self.wake_conversation_in_view: Callable[[str, str], bool] | None = None
        self._register_fleet_wake(agent_bridge)
        #: task-2154.16 (FB-05): UI-thread callback invoked DIRECTLY (same
        #: main-loop guarantee as ``notify_run_outcome`` above) from
        #: ``_set_run_state``'s once-guarded transition INTO ``FAILED`` for
        #: the ACTIVE (viewed) session -- the case ``notify_run_outcome``
        #: deliberately skips. The viewed session's failure was previously
        #: confined to a transcript system row plus run-state copy on a
        #: hidden surface, so a user composing their next message got no
        #: ambient failure signal. Carries the run's ``visible_copy`` (the
        #: same text as the transcript system row). Wired to
        #: ``ChatScreen._notify_console_run_failure`` by
        #: ``_ensure_console_chat_controller``. ``None`` in most
        #: controller-only tests, matching every other UI bridge slot here.
        self.notify_run_failure: Callable[[str], None] | None = None
        #: Optional override for how long ``request_mcp_approvals`` waits
        #: for a human decision before failing every undecided call to
        #: ``"timeout"``. Defaults to reading ``[mcp] approval_timeout_
        #: seconds`` (T2's ``approval_timeout_seconds``) when unset --
        #: ADR-067: that default is 0 = no deadline (wait indefinitely);
        #: a positive value re-arms the auto-deny clock.
        self.mcp_approval_timeout_seconds: Callable[[], float] | None = None
        #: Task 9 (Fix round 1): each batch-approval round's release signal
        #: + shared decisions holder + owning session id, keyed by a
        #: freshly minted ROUND id (``uuid4()``, stamped into the payload
        #: as ``"round_id"`` and round-tripped through ``ChatApprovalCard``
        #: -> ``ApprovalDecided`` -> ``resolve_pending_approval``) --
        #: mirrors ``_pending_skill_script_rounds``'s identical
        #: ``request_id``-keyed design. Superseded TWO earlier, both-wrong
        #: shapes: the pre-Task-9 single ``_pending_approval_event``/
        #: ``_pending_approval_decisions`` pair (only ever tracked ONE
        #: round controller-wide -- fatal once two sessions can each have
        #: their own concurrent pending approval), and this task's own
        #: first draft keyed by session id alone (still wrong: `Approval
        #: Decided` travels as an async Textual message, so a
        #: `switch_session` landing in the gap between the user's click and
        #: the handler running could resolve session A's decision against
        #: session B's completely different batch -- review CRITICAL
        #: finding, fix round 1). Read/written from the UI thread by
        #: ``resolve_pending_approval``, which resolves ONLY the round
        #: whose id was stamped onto the card the user actually decided --
        #: never "whichever session happens to be active right now".
        self._pending_approval_rounds: dict[str, dict[str, Any]] = {}
        #: PR0: retained payload per ROUND (was per session), keyed by
        #: `round_id`. `switch_session` and every teardown re-derive the
        #: mounted card from this map's FIFO head for the session, so a
        #: second same-session round no longer evicts an older sibling's
        #: card. Every payload carries its own `round_id` and `session_id`.
        self._parked_approval_payloads: dict[str, dict[str, Any]] = {}
        #: UI-thread callback that pushes/clears the pending skill-install
        #: confirm payload into the owning screen's task-resume state
        #: (`ConsoleSkillController._set_console_pending_skill_install`). Invoked through
        #: self.app.call_from_thread from request_skill_install_confirm.
        self.set_pending_skill_install: Callable[[dict | None], None] | None = None
        #: Optional test override for the confirm timeout.
        self.skill_install_confirm_timeout_seconds: Callable[[], float] | None = None
        #: TASK-910: per-round release Event + shared decision box + owning
        #: session id, keyed by a freshly minted request id -- mirrors
        #: `_pending_skill_script_rounds`' identical shape (itself task-581's
        #: fix for the same "single shared slot clobbers a second concurrent
        #: round" hazard `request_mcp_approvals` solved with `round_id`).
        #: Pre-TASK-910 this was a single `_pending_skill_install_event`/
        #: `_pending_skill_install_decision` pair -- fine while only one
        #: session could ever have a live install confirm, but parking makes
        #: two DIFFERENT background sessions' install confirms genuinely
        #: concurrent.
        self._pending_skill_install_rounds: dict[str, dict[str, Any]] = {}
        self._pending_skill_install_lock = threading.Lock()
        #: PR0: retained payload per ROUND (was per session), keyed by
        #: `request_id`. The mounted card is the session's FIFO head, so a
        #: second same-session confirm no longer evicts an older sibling.
        self._parked_skill_install_payloads: dict[str, dict[str, Any]] = {}
        #: UI-thread callback that pushes/clears the pending skill-SCRIPT
        #: confirm payload into the owning screen's task-resume state.
        #: Invoked through self.app.call_from_thread from
        #: request_skill_script_confirm. Mirrors set_pending_skill_install,
        #: but the round-trip decision carries a "remember" flag too.
        self.set_pending_skill_script: Callable[[dict | None], None] | None = None
        #: Optional test override for the confirm timeout, mirroring
        #: `skill_install_confirm_timeout_seconds`.
        self.skill_script_confirm_timeout_seconds: Callable[[], float] | None = None
        #: The active script-confirm round's release Event + shared
        #: decision box ({"allow": bool, "remember": bool}), now also
        #: carrying the round's owning session id (TASK-910) so teardown can
        #: tell whether ANOTHER still-armed round belongs to the SAME
        #: session (must not clear the mounted card out from under it --
        #: see `request_skill_script_confirm`) independently of whether some
        #: OTHER session also has a round outstanding.
        #: task-581: rounds keyed by request_id, not a single slot. Two rounds
        #: armed at once previously clobbered each other's event/decision and
        #: both worker threads then blocked to their full deadline.
        self._pending_skill_script_rounds: dict[str, dict[str, Any]] = {}
        self._pending_skill_script_lock = threading.Lock()
        #: PR0: retained payload per ROUND (was per session), keyed by
        #: `request_id`. The mounted card is the session's FIFO head, so a
        #: second same-session confirm no longer evicts an older sibling.
        self._parked_skill_script_payloads: dict[str, dict[str, Any]] = {}

    def _hydrate_capture_policy(self, session: ConsoleChatSession) -> None:
        if session.id in self._capture_policy_hydrated:
            return
        if session.persisted_conversation_id is None:
            self._capture_policy_hydrated.add(session.id)
            self._capture_policy_hydration_errors.pop(session.id, None)
            return
        outcome = self.store.hydrate_session_capture_policy(session.id)
        if outcome.status in {
            CapturePolicyReadStatus.ABSENT,
            CapturePolicyReadStatus.FOUND,
        }:
            self._capture_policy_hydrated.add(session.id)
            self._capture_policy_hydration_errors.pop(session.id, None)
        else:
            self._capture_policy_hydration_errors[session.id] = (
                "conversation_policy_unavailable"
            )

    def capture_policy_snapshot(self, session_id: str) -> CapturePolicySnapshot:
        """Resolve the future policy for one immutable session identity."""
        session = next((item for item in self.store.sessions() if item.id == session_id), None)
        if session is None:
            raise KeyError(session_id)
        self._hydrate_capture_policy(session)
        state = self.store.capture_policy_state(session_id)
        runtime = runtime_capture_policy()
        effective = resolve_capture_policy(
            enabled=runtime.enabled,
            next_send=state.next_detail,
            conversation=state.conversation_detail,
            global_default=runtime.detail,
        )
        queue = self.prompt_queue_registry.snapshot(session_id)
        return CapturePolicySnapshot(
            session_id=session_id,
            conversation_id=session.persisted_conversation_id,
            conversation_title=session.title,
            enabled=runtime.enabled,
            next_detail=state.next_detail,
            conversation_detail=state.conversation_detail,
            global_detail=runtime.detail,
            effective=effective,
            policy_revision=state.policy_revision,
            config_generation=runtime.generation,
            capture_revision=state.capture_revision,
            active_run_detail=self._active_capture_details.get(session_id),
            queued_consumer=bool(getattr(queue, "entries", ())),
            save_pending=state.save_pending,
            error_code=self._capture_policy_hydration_errors.get(session_id) or (
                "invalid_" + "_".join(effective.invalid_sources)
                if effective.invalid_sources
                else None
            ),
        )

    def capture_revision(self, session_id: str) -> int:
        """Return the authoritative process-local capture revision."""
        return self.store.capture_revision(session_id)

    def capture_purge_availability(
        self, session_id: str
    ) -> CapturePurgeAvailability:
        """Report the first bounded writer reason preventing quiescence."""
        try:
            self.store.capture_revision(session_id)
        except KeyError:
            return CapturePurgeAvailability(False, "target_missing")
        with self._capture_quiescence_lock:
            reason = self._capture_purge_blocker(session_id, include_lease=True)
            return CapturePurgeAvailability(reason is None, reason)

    def _capture_purge_blocker(
        self, session_id: str, *, include_lease: bool
    ) -> str | None:
        """Return the first bounded writer code for one session."""
        if include_lease and self.store.capture_quiescent(session_id):
            return "purge_in_progress"
        task = self._active_stream_tasks.get(session_id)
        if task is not None and not task.done():
            return "primary_writer_active"
        with self._active_submit_tasks_lock:
            if session_id in self._active_submit_tasks.values():
                return "preparation_active"
        if self.store.preparation_for_session(session_id) is not None:
            return "preparation_active"
        checker = getattr(self._agent_bridge, "has_unsettled_children", None)
        if callable(checker):
            try:
                if checker(self._agent_conversation_id(session_id)):
                    return "fleet_writer_active"
            except Exception:
                return "fleet_state_unavailable"
        if session_id in self._capture_exchange_flush_sessions:
            return "exchange_flush_active"
        for message_id in self._fleet_usage_reattach_sources:
            try:
                if self.store.session_id_for_message(message_id) == session_id:
                    return "retained_signals_active"
            except KeyError:
                continue
        return None

    async def purge_full_captures(
        self, session_id: str, expected_capture_revision: int
    ) -> CapturePurgeResult:
        """Logically erase Full captures while every session writer is fenced."""
        with self._capture_quiescence_lock:
            try:
                revision = self.store.capture_revision(session_id)
            except KeyError:
                return CapturePurgeResult.blocked(
                    _MISSING_CAPTURE_REVISION,
                    "target_missing",
                )
            reason = self._capture_purge_blocker(session_id, include_lease=True)
            if reason is not None:
                return CapturePurgeResult.blocked(
                    revision, reason
                )
            if revision != expected_capture_revision:
                return CapturePurgeResult(
                    CapturePurgeStatus.STALE,
                    0,
                    revision,
                    "stale_capture_revision",
                )
            if not self.store.begin_capture_quiescence(session_id):
                return CapturePurgeResult.blocked(revision, "purge_in_progress")
            reason = self._capture_purge_blocker(session_id, include_lease=False)
            if reason is not None:
                self.store.end_capture_quiescence(session_id)
                return CapturePurgeResult.blocked(revision, reason)
        try:
            stage = self.store.stage_full_capture_purge(session_id)
            removed = self.store.commit_full_capture_purge(stage)
            return CapturePurgeResult.deleted(
                removed, self.store.capture_revision(session_id)
            )
        except CapturePurgeStaleError:
            return CapturePurgeResult(
                CapturePurgeStatus.STALE,
                0,
                self.store.capture_revision(session_id),
                "stale_capture_revision",
            )
        except Exception as exc:
            logger.warning(
                "capture_purge_failed (exception_type={})", type(exc).__name__
            )
            return CapturePurgeResult(
                CapturePurgeStatus.FAILED,
                0,
                self.store.capture_revision(session_id),
                "persistence_unavailable",
            )
        finally:
            self.store.end_capture_quiescence(session_id)

    def set_next_capture_detail(
        self,
        session_id: str,
        detail: CaptureDetail | None,
        *,
        expected_policy_revision: int,
    ) -> CapturePolicyMutationResult:
        try:
            self.store.set_session_next_capture_detail(
                session_id,
                detail,
                expected_policy_revision=expected_policy_revision,
            )
        except CapturePolicyStaleError:
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.STALE,
                self.capture_policy_snapshot(session_id),
                True,
                "stale_policy_revision",
            )
        except KeyError:
            raise
        return CapturePolicyMutationResult(
            CapturePolicyMutationStatus.APPLIED,
            self.capture_policy_snapshot(session_id),
            False,
            None,
        )

    async def replace_conversation_capture_detail(
        self,
        session_id: str,
        detail: CaptureDetail | None,
        *,
        expected_policy_revision: int,
    ) -> CapturePolicyMutationResult:
        before = self.capture_policy_snapshot(session_id)
        if before.policy_revision != expected_policy_revision:
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.STALE,
                before,
                True,
                "stale_policy_revision",
            )
        try:
            reservation = self.store.reserve_capture_policy_mutation(
                expected_policy_revision=expected_policy_revision
            )
        except CapturePolicyStaleError:
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.STALE,
                self.capture_policy_snapshot(session_id),
                True,
                "stale_policy_revision",
            )
        inherited = resolve_capture_policy(
            enabled=before.enabled,
            conversation=detail,
            global_default=before.global_detail,
            allow_next_send=False,
        ).detail
        has_durable_identity = before.conversation_id is not None
        privacy_safe_result = inherited is CaptureDetail.SAFE
        if privacy_safe_result:
            try:
                self.store.publish_reserved_capture_safe(
                    reservation,
                    session_id=session_id,
                    save_pending=has_durable_identity,
                )
            except KeyError:
                self.store.abandon_capture_policy_mutation(reservation)
                return CapturePolicyMutationResult(
                    CapturePolicyMutationStatus.TARGET_MISSING,
                    before,
                    False,
                    "session_closed",
                )

        async def reconcile() -> CapturePolicyMutationResult:
            reservation_owned = True
            reconciliation_cancelled = False
            try:
                session_only = False
                if (
                    has_durable_identity
                    and self._capture_policy_repository is not None
                ):
                    repository = self._capture_policy_repository
                    repository_settled = asyncio.Event()
                    repository_result: list[Any] = []
                    repository_error: list[BaseException] = []
                    loop = asyncio.get_running_loop()

                    def run_repository_call() -> None:
                        try:
                            repository_result.append(
                                repository.replace(
                                    before.conversation_id,
                                    detail,
                                )
                            )
                        except BaseException as exc:
                            repository_error.append(exc)
                        finally:
                            loop.call_soon_threadsafe(repository_settled.set)

                    if self._durable_db_call_offloadable():
                        threading.Thread(
                            target=run_repository_call,
                            name="console-capture-policy-write",
                        ).start()
                    else:
                        run_repository_call()
                    while not repository_settled.is_set():
                        try:
                            await repository_settled.wait()
                        except asyncio.CancelledError:
                            reconciliation_cancelled = True
                    if repository_error:
                        if not privacy_safe_result:
                            if reconciliation_cancelled:
                                raise asyncio.CancelledError from None
                            raise repository_error[0]
                        session_only = True
                        write_status = None
                    else:
                        write_status = repository_result[0]
                    if (
                        write_status is not None
                        and write_status.status
                        is CapturePolicyWriteStatus.MISSING_CONVERSATION
                    ):
                        self.store.abandon_capture_policy_mutation(reservation)
                        reservation_owned = False
                        if reconciliation_cancelled:
                            raise asyncio.CancelledError
                        return CapturePolicyMutationResult(
                            CapturePolicyMutationStatus.TARGET_MISSING,
                            before,
                            False,
                            "conversation_missing",
                        )
                    if write_status is not None:
                        session_only = (
                            write_status.status is CapturePolicyWriteStatus.UNAVAILABLE
                        )
                elif has_durable_identity:
                    session_only = True
                if session_only and inherited is CaptureDetail.FULL:
                    self.store.abandon_capture_policy_mutation(reservation)
                    reservation_owned = False
                    if reconciliation_cancelled:
                        raise asyncio.CancelledError
                    return CapturePolicyMutationResult(
                        CapturePolicyMutationStatus.FAILED,
                        before,
                        True,
                        "save_failed",
                    )
                try:
                    self.store.finish_capture_policy_mutation(
                        reservation,
                        session_id=session_id,
                        detail=(
                            CaptureDetail.SAFE
                            if session_only and privacy_safe_result
                            else detail
                        ),
                        save_pending=session_only and has_durable_identity,
                    )
                except KeyError:
                    reservation_owned = False
                    if reconciliation_cancelled:
                        raise asyncio.CancelledError
                    return CapturePolicyMutationResult(
                        CapturePolicyMutationStatus.TARGET_MISSING,
                        before,
                        False,
                        "session_closed",
                    )
                reservation_owned = False
                if has_durable_identity and not session_only:
                    self._capture_policy_hydrated.add(session_id)
                    self._capture_policy_hydration_errors.pop(session_id, None)
                result = CapturePolicyMutationResult(
                    CapturePolicyMutationStatus.SAFE_SESSION_ONLY
                    if session_only and has_durable_identity
                    else CapturePolicyMutationStatus.APPLIED,
                    self.capture_policy_snapshot(session_id),
                    session_only and has_durable_identity,
                    "save_failed"
                    if session_only and has_durable_identity
                    else None,
                )
                if reconciliation_cancelled:
                    raise asyncio.CancelledError
                return result
            finally:
                if reservation_owned:
                    try:
                        self.store.abandon_capture_policy_mutation(reservation)
                    except CapturePolicyStaleError:
                        pass

        reconciliation = asyncio.create_task(reconcile())
        caller_cancelled = False
        while not reconciliation.done():
            try:
                await asyncio.shield(reconciliation)
            except asyncio.CancelledError:
                if reconciliation.cancelled():
                    break
                caller_cancelled = True
            except BaseException:
                if caller_cancelled:
                    raise asyncio.CancelledError from None
                raise
        if reconciliation.cancelled():
            try:
                self.store.abandon_capture_policy_mutation(reservation)
            except CapturePolicyStaleError:
                pass
            raise asyncio.CancelledError
        try:
            result = reconciliation.result()
        except BaseException:
            if caller_cancelled:
                raise asyncio.CancelledError from None
            raise
        if caller_cancelled:
            raise asyncio.CancelledError
        return result

    def apply_global_capture_settings(
        self,
        *,
        enabled: bool,
        detail: CaptureDetail,
        expected_config_generation: int,
        expected_policy_revision: int,
    ) -> CapturePolicyMutationResult:
        session_id = self.store.active_session_id
        if session_id is None:
            raise KeyError("No active Console session")
        before = self.capture_policy_snapshot(session_id)
        if before.policy_revision != expected_policy_revision:
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.STALE,
                before,
                True,
                "stale_policy_revision",
            )
        try:
            reservation = self.store.reserve_capture_policy_mutation(
                expected_policy_revision=expected_policy_revision
            )
        except CapturePolicyStaleError:
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.STALE,
                self.capture_policy_snapshot(session_id),
                True,
                "stale_policy_revision",
            )
        try:
            config_result = apply_console_capture_settings(
                enabled=enabled,
                detail=detail,
                expected_generation=expected_config_generation,
            )
        except BaseException:
            self.store.abandon_capture_policy_mutation(reservation)
            raise
        if config_result.conflict:
            self.store.abandon_capture_policy_mutation(reservation)
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.STALE,
                before,
                True,
                "stale_config_generation",
                config_result,
            )
        active = not enabled or detail is CaptureDetail.SAFE or config_result.file_replaced
        if not active:
            self.store.abandon_capture_policy_mutation(reservation)
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.FAILED,
                before,
                True,
                "save_failed",
                config_result,
            )
        self.store.finish_capture_policy_mutation(
            reservation,
            disarm_next=not enabled,
        )
        return CapturePolicyMutationResult(
            CapturePolicyMutationStatus.SAFE_SESSION_ONLY
            if config_result.failure_phase == "before_replace"
            else CapturePolicyMutationStatus.APPLIED,
            self.capture_policy_snapshot(session_id),
            config_result.failure_phase == "before_replace",
            "save_failed"
            if config_result.failure_phase == "before_replace"
            else "cache_refresh_degraded"
            if config_result.failure_phase == "cache_reload"
            else None,
            config_result,
        )

    def _admit_capture_policy(
        self,
        session_id: str,
        origin: ConsoleSubmissionOrigin,
    ) -> ConsoleProviderStreamSignals:
        """Freeze capture detail once, after an accepted owner exists."""
        eligible = origin in {
            ConsoleSubmissionOrigin.MANUAL,
            ConsoleSubmissionOrigin.QUEUED,
        }
        try:
            session = next(
                (item for item in self.store.sessions() if item.id == session_id),
                None,
            )
            if session is None:
                raise KeyError(session_id)
            self._hydrate_capture_policy(session)
            state = self.store.capture_policy_state(session_id)
            runtime = runtime_capture_policy()
            resolution = resolve_capture_policy(
                enabled=runtime.enabled,
                next_send=state.next_detail,
                conversation=state.conversation_detail,
                global_default=runtime.detail,
                allow_next_send=eligible,
            )
        except Exception as exc:
            logger.bind(
                phase="resolution",
                error_type=type(exc).__name__,
            ).warning("capture_policy_resolution_failed")
            return ConsoleProviderStreamSignals(
                exchange_capture_enabled=False,
                capture_detail=CaptureDetail.SAFE,
            )
        signals = ConsoleProviderStreamSignals(
            exchange_capture_enabled=resolution.enabled,
            capture_detail=resolution.detail,
        )
        if eligible and state.next_detail is not None:
            try:
                self.store.consume_session_next_capture_detail(
                    session_id,
                    expected_next_revision=state.next_revision,
                )
            except Exception as exc:
                logger.bind(
                    phase="one_shot_consumption",
                    error_type=type(exc).__name__,
                ).warning("capture_policy_resolution_failed")
                return ConsoleProviderStreamSignals(
                    exchange_capture_enabled=False,
                    capture_detail=CaptureDetail.SAFE,
                )
        return signals

    @property
    def run_state(self) -> ConsoleRunState:
        """The ACTIVE session's run state (parallel-agents spec §2).

        Read-only facade: the ~16 pre-existing read sites in chat_screen
        keep their semantics ("the viewed session's run"), while writes go
        through ``_set_run_state``/``_clear_terminal_run_state`` with an
        explicit owning session id. There is deliberately no setter --
        assigning ``controller.run_state = ...`` now raises ``AttributeError``
        so a stray direct-assignment writer (bypassing the per-session map)
        fails loudly instead of silently reintroducing the single-slot bug.

        Returns:
            The active session's recorded ``ConsoleRunState`` (a fresh idle
            state when the active session has no recorded run).
        """
        return self.run_state_for(self.store.active_session_id or "")

    def run_state_for(self, session_id: str) -> ConsoleRunState:
        """Return ``session_id``'s own run state (a fresh idle one when unset).

        Args:
            session_id: The session id to look up.

        Returns:
            The session's recorded ``ConsoleRunState``, or a fresh idle
            ``ConsoleRunState`` when the session has no recorded run.
        """
        return self._run_states.get(session_id) or ConsoleRunState()

    def run_states(self) -> dict[str, ConsoleRunState]:
        """Raw map snapshot incl. entries for closed sessions.

        This is the UNFILTERED ``self._run_states`` copy -- it can contain
        orphaned entries for sessions ``ConsoleChatStore.close_session`` has
        already removed (closing never touches the controller's map). Use
        ``in_flight_run_count`` (or ``_live_busy_session_ids``) for cap/fleet
        math; those exclude orphans. This raw snapshot is for callers that
        want the full recorded history regardless of session lifetime.

        Returns:
            A shallow copy of the internal session-id -> ``ConsoleRunState``
            map, including entries for sessions the store has since closed.
        """
        return dict(self._run_states)

    def run_active_for_workspace(self, root: str) -> bool:
        """Whether any live session is executing against the given root.

        The roots come from each dispatch's immutable execution context, not
        from the currently viewed session or mutable workspace selection.

        Args:
            root: Workspace root about to be mutated.

        Returns:
            True when a non-terminal run captured the same canonical root.
        """

        def _canonical(value: str) -> str:
            try:
                return os.path.normcase(str(Path(value).expanduser().resolve()))
            except OSError:
                return os.path.normcase(
                    os.path.abspath(os.path.expanduser(str(value)))
                )

        target = _canonical(root)
        with self._active_workspace_roots_lock:
            captured = tuple(self._active_workspace_roots_by_session.items())
        return any(
            not self.run_state_for(session_id).is_send_allowed
            and any(_canonical(candidate) == target for candidate in roots)
            for session_id, roots in captured
        )

    def activity_for(self, session_id: str) -> ConsoleControllerActivity:
        """Return the single queue-aware activity projection for ``session_id``."""

        if self.store.dispatch_recovery_needs_queue_hydration(session_id):
            self._hydrate_dispatch_recovery_queue(session_id)
        return self.prompt_queue_coordinator.activity(session_id)

    def _hydrate_dispatch_recovery_queue(
        self,
        session_id: str,
        *,
        force: bool = False,
    ) -> bool:
        """Project queued recovery before any activity consumer can advance it."""

        if not force and not self.store.dispatch_recovery_needs_queue_hydration(
            session_id
        ):
            return False
        recovery = self.store.dispatch_recovery_for_session(session_id)
        checkpoint = recovery.checkpoint if recovery is not None else None
        if (
            checkpoint is None
            or checkpoint.origin != "queued"
            or checkpoint.queue_entry_id is None
        ):
            self.store.mark_dispatch_recovery_queue_hydrated(session_id)
            return False
        hydrated = self.prompt_queue_coordinator.hydrate_dispatch_recovery(
            session_id,
            queue_entry_id=checkpoint.queue_entry_id,
            preparation_id=checkpoint.preparation_id,
            checkpoint_state=checkpoint.state,
        )
        if hydrated:
            self.store.mark_dispatch_recovery_queue_hydrated(session_id)
        return hydrated

    def _restore_dispatch_recovery_after_settlement_failure(
        self,
        session_id: str,
        assistant_message_id: str,
    ) -> None:
        """Publish one rollback-preserved owner before any queue can advance."""

        self.store.mark_dispatch_recovery_needed(
            session_id,
            assistant_message_id,
        )
        self._set_run_state(
            ConsoleRunState(
                ConsoleRunStatus.BLOCKED,
                "Response recovery failed. Try again or discard.",
            ),
            session_id=session_id,
        )
        self._hydrate_dispatch_recovery_queue(session_id, force=True)

    def _advance_lifecycle_revision(self, session_id: str) -> None:
        """Advance content-free fleet and owning-session confirmation fences."""

        with self._lifecycle_revision_lock:
            self._lifecycle_revision += 1
            if session_id:
                self._session_lifecycle_revisions[session_id] = (
                    self._session_lifecycle_revisions.get(session_id, 0) + 1
                )

    def _note_controller_activity_changed(self, session_id: str) -> None:
        """Receive queue-owner changes without copying its state."""

        self._advance_lifecycle_revision(session_id)

    def _lifecycle_revision_for(self, session_id: str | None) -> int:
        with self._lifecycle_revision_lock:
            if session_id is None:
                return self._lifecycle_revision
            return self._session_lifecycle_revisions.get(session_id, 0)

    def lifecycle_impact(
        self, *, session_id: str | None = None
    ) -> ConsoleLifecycleImpact:
        """Derive exact loss counts for one session or the whole live fleet.

        The optimistic revision read prevents a worker-thread approval change
        from producing counts paired with an older token. Queue state itself
        remains owner-thread confined and is read through the coordinator's
        immutable activity projection.
        """

        while True:
            revision = self._lifecycle_revision_for(session_id)
            live_ids = {session.id for session in self.store.sessions()}
            if session_id is not None:
                target_ids = (session_id,) if session_id in live_ids else ()
            else:
                target_ids = tuple(sorted(live_ids))
            activities = tuple(self.activity_for(target_id) for target_id in target_ids)
            if revision == self._lifecycle_revision_for(session_id):
                break
        return ConsoleLifecycleImpact(
            revision=revision,
            live_run_count=sum(
                bool(activity.occupies_slot or activity.needs_approval)
                for activity in activities
            ),
            queued_session_count=sum(
                activity.has_queued_work for activity in activities
            ),
            unsent_prompt_count=sum(activity.queued_count for activity in activities),
        )

    def queue_prompt(
        self,
        session_id: str,
        *,
        text: str,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        """Attempt atomic text-only admission behind queue-owned work."""

        clean_text, validation_error = self._validated_draft(text)
        if validation_error is not None:
            return PromptQueueMutationResult(
                QueueMutationStatus.INVALID,
                self.prompt_queue_registry.snapshot(session_id),
                detail=validation_error,
            )
        return self.prompt_queue_coordinator.admit(
            session_id,
            text=clean_text,
            expected_revision=expected_revision,
        )

    def edit_queued_prompt(
        self,
        session_id: str,
        *,
        entry_id: str,
        text: str,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        """Edit one waiting prompt and publish the queue activity revision."""

        clean_text, validation_error = self._validated_draft(text)
        if validation_error is not None:
            return PromptQueueMutationResult(
                QueueMutationStatus.INVALID,
                self.prompt_queue_registry.snapshot(session_id),
                detail=validation_error,
            )
        result = self.prompt_queue_registry.edit(
            session_id,
            entry_id=entry_id,
            text=clean_text,
            expected_revision=expected_revision,
        )
        if result.applied:
            self.prompt_queue_coordinator.publish_registry_change(session_id)
        return result

    def move_queued_prompt(
        self,
        session_id: str,
        *,
        entry_id: str,
        new_index: int,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        """Reorder one waiting prompt and publish the queue activity revision."""

        result = self.prompt_queue_registry.move(
            session_id,
            entry_id=entry_id,
            new_index=new_index,
            expected_revision=expected_revision,
        )
        if result.applied:
            self.prompt_queue_coordinator.publish_registry_change(session_id)
        return result

    def remove_queued_prompt(
        self, session_id: str, *, entry_id: str, expected_revision: int
    ) -> PromptQueueMutationResult:
        """Remove one waiting prompt and publish the queue activity revision."""

        result = self.prompt_queue_registry.remove(
            session_id,
            entry_id=entry_id,
            expected_revision=expected_revision,
        )
        if result.applied:
            self.prompt_queue_coordinator.publish_registry_change(session_id)
        return result

    def clear_queued_prompts(
        self, session_id: str, *, expected_revision: int
    ) -> PromptQueueMutationResult:
        """Clear waiting prompts and publish the queue activity revision."""

        result = self.prompt_queue_registry.clear_waiting(
            session_id, expected_revision=expected_revision
        )
        if result.applied:
            self.prompt_queue_coordinator.publish_registry_change(session_id)
        return result

    async def run_prompt_chain(
        self,
        draft: str,
        *,
        session_id: str | None = None,
    ) -> ConsoleSubmitResult:
        """Submit one manual draft and drain accepted follow-ups sequentially."""

        target_id = session_id or self.store.active_session_id
        if not target_id:
            session = self.store.ensure_session(
                workspace_id=self.store.workspace_context.active_workspace_id,
                settings=(
                    self._default_session_settings()
                    if self._default_session_settings is not None
                    else None
                ),
            )
            target_id = session.id
        return await self.prompt_queue_coordinator.run_prompt_chain(
            target_id,
            lambda: self.submit_draft(
                draft,
                session_id=target_id,
            ),
        )

    def _queued_staged_rider_present(self, session_id: str) -> bool:
        """Return whether a queued text turn would consume a manual attachment rider."""

        try:
            if self.store.pending_attachments(session_id):
                return True
        except KeyError:
            return True
        provider = self._queued_staged_rider_provider
        if provider is None:
            return False
        try:
            return bool(provider(session_id))
        except Exception:
            # Fail closed: a rider seam that cannot be revalidated must never
            # let an automatic queued turn consume screen-owned state.
            return True

    def _queue_can_reacquire_slot(self, session_id: str) -> bool:
        """Check the cap without registering a hidden waiter."""

        live_ids = {session.id for session in self.store.sessions()}
        occupied = sum(
            1
            for candidate in live_ids
            if candidate != session_id and self.activity_for(candidate).occupies_slot
        )
        return occupied < self.max_parallel_runs

    def payload_fingerprint_baseline(
        self, session_id: str
    ) -> PayloadFingerprint | None:
        """Return the fingerprint of the last payload actually sent for a session.

        Recorded at dispatch time by ``_stream_assistant_response_inner``,
        from the same pre-compaction stage ``compute_current_fingerprint``
        recomputes from. A failed or blocked send never reaches that
        recording site, so this always reflects what was genuinely
        transmitted -- there is nothing to roll back on failure.

        Args:
            session_id: The session id to look up.

        Returns:
            The recorded :class:`PayloadFingerprint`, or ``None`` when the
            session has never sent (or the process-local record was lost,
            e.g. across a restart).
        """
        return self._payload_fingerprint_baselines.get(session_id)

    def compute_current_fingerprint(self, session_id: str) -> PayloadFingerprint:
        """Fingerprint what a send RIGHT NOW would dispatch for a session.

        Uses ``_provider_messages_for_session`` -- the same pre-compaction,
        pre-window stage the baseline above was recorded from -- so the two
        are directly comparable via ``fingerprint_break_reason`` without
        either side needing to account for compaction/windowing drift.

        Args:
            session_id: The session id to fingerprint.

        Returns:
            The current :class:`PayloadFingerprint`.
        """
        messages = self._provider_messages_for_session(session_id)
        return fingerprint_payload(
            self.provider, self.model or self.configured_model, messages
        )

    def cache_ttl_snapshot(self, session_id: str) -> tuple[float | None, bool]:
        """Return the session's recorded Anthropic prompt-cache TTL ground truth.

        Both values are stamped by ``_attach_stream_usage`` only for
        Anthropic sends with prompt caching enabled -- every other provider/
        session reads back ``(None, False)``.

        Args:
            session_id: The session id to look up.

        Returns:
            A ``(warm_until, had_cache_activity)`` pair: ``warm_until`` is
            the ``time.monotonic()`` deadline the cache is expected to stay
            warm until (``None`` when never stamped, e.g. no cache activity
            has been observed yet), and ``had_cache_activity`` is whether
            the session's most recent Anthropic send reported any cache
            read/write at all.
        """
        return (
            self._cache_warm_until.get(session_id),
            self._cache_last_activity.get(session_id, False),
        )

    def _live_busy_session_ids(self) -> list[str]:
        """Busy session ids that still exist in the store, insertion-ordered.

        Intersects ``self._run_states`` with ``store.sessions()``: a session
        closed mid-VALIDATING leaves its entry in the map behind (Task 1
        review finding), and neither cap/fleet math nor the refusal copy's
        session list may count or name a session that no longer exists.
        Shared by ``in_flight_run_count`` and ``send_refusal_copy`` so both
        apply the same live-session filter.
        """
        live_ids = {session.id for session in self.store.sessions()}
        ordered_ids = list(self._run_states)
        ordered_ids.extend(sid for sid in live_ids if sid not in self._run_states)
        return [
            sid
            for sid in ordered_ids
            if sid in live_ids and self.activity_for(sid).occupies_slot
        ]

    def in_flight_run_count(self) -> int:
        """Count of LIVE sessions whose recorded run currently disallows a new send.

        Excludes orphaned entries for sessions the store no longer has (see
        ``_live_busy_session_ids``) -- consumers (cap math, fleet UX) must
        never see a closed session's stale run inflate this count.

        Returns:
            The number of live sessions whose recorded run currently
            disallows a new send.
        """
        return len(self._live_busy_session_ids())

    def add_pending_round(self, session_id: str, round_id: str) -> None:
        """Register ``round_id`` as an outstanding approval-like round for ``session_id``.

        TASK-1050 (Defect A): the fleet-visible pending-approval badge used
        to be a single boolean per session (``_pending_approvals`` as a
        plain ``set[str]``, flipped by the now-deprecated ``set_run_
        pending_approval``) shared by THREE independent bridges -- MCP tool
        approvals, skill-install confirms, and skill-script confirms. Any
        one bridge's teardown cleared the badge for its own session_id
        regardless of whether a SIBLING round (same bridge or a different
        one) was still outstanding for that same session, so the badge
        could go dark while a live confirm was still waiting on the user.

        ``_pending_approvals`` is now keyed by session id to the SET of
        round ids currently outstanding for it -- a session reads as
        "pending" (``run_marker_for``/``fleet_summary_counts``) iff that
        set is non-empty. Idempotent: adding an already-registered
        ``round_id`` again is a no-op (set semantics), so a caller never
        needs to check first.

        Every genuine bridge round already mints a fresh ``uuid4()`` round/
        request id before arming (``request_mcp_approvals``'s ``round_id``,
        ``request_skill_install_confirm``'s/``request_skill_script_
        confirm``'s ``request_id``) -- this is the id each bridge now
        passes here instead of the old boolean.

        Args:
            session_id: The session the round belongs to.
            round_id: The round's own unique id (a real bridge round id, or
                the reserved ``_LEGACY_PENDING_APPROVAL_ROUND_ID`` sentinel
                -- see ``set_run_pending_approval``).
        """
        # F2b fix (Qodo wave), preserved: reachable from a worker thread
        # while the UI thread concurrently iterates `_pending_approvals`
        # via `fleet_summary_counts` -- guard the mutation with the shared
        # lock so iteration never observes a torn add/discard.
        with self._approval_state_lock:
            rounds = self._pending_approvals.setdefault(session_id, set())
            changed = round_id not in rounds
            rounds.add(round_id)
        if changed:
            if self._buddy_sink is not None:
                self._buddy_sink.approval_round(session_id, round_id, pending=True)
            self._advance_lifecycle_revision(session_id)

    def discard_pending_round(self, session_id: str, round_id: str) -> None:
        """Clear ``round_id`` from ``session_id``'s outstanding approval-like rounds.

        TASK-1050 (Defect A) counterpart to ``add_pending_round``: discards
        only THIS round's id from the session's round-id set. The fleet
        badge (``run_marker_for``) clears only once that set is empty --
        i.e. once every bridge round for the session has resolved, not just
        this one. Idempotent: discarding an id that was never added (or was
        already discarded) is a safe no-op, and discarding the SAME id
        twice never double-decrements anything (set semantics -- there is
        nothing to corrupt).

        Args:
            session_id: The session the round belongs to.
            round_id: The round's own unique id, as passed to the matching
                ``add_pending_round`` call.
        """
        changed = False
        with self._approval_state_lock:
            rounds = self._pending_approvals.get(session_id)
            if rounds is None:
                return
            changed = round_id in rounds
            rounds.discard(round_id)
            if not rounds:
                self._pending_approvals.pop(session_id, None)
        if changed:
            if self._buddy_sink is not None:
                self._buddy_sink.approval_round(session_id, round_id, pending=False)
            self._advance_lifecycle_revision(session_id)

    def has_pending_approval_round(self, session_id: str) -> bool:
        """Return whether ``session_id`` currently has ANY outstanding approval-like round.

        TASK-1050: exposed so a caller that lacks a round id of its own
        (see ``set_run_pending_approval``'s docstring) can check whether a
        REAL round is already registered before redundantly stamping the
        deprecated boolean shim -- ``ChatScreen._park_console_approval`` is
        the one production caller that needs this (its owning bridge always
        registers the real round id via ``add_pending_round`` moments
        before invoking the park callback, so by the time this runs, the
        real round is normally already present).

        Args:
            session_id: The session to check.

        Returns:
            ``True`` iff at least one round id is currently registered for
            ``session_id``.
        """
        with self._approval_state_lock:
            return session_id in self._pending_approvals

    def set_run_pending_approval(self, session_id: str, pending: bool) -> None:
        """DEPRECATED boolean shim -- prefer ``add_pending_round``/``discard_pending_round``.

        Parallel-agents spec §6 (Task 7 stores/exposes the flag; Task 9
        wired the approval paths -- MCP batch approvals, skill-install/
        script confirms -- that originally called this). TASK-1050 (Defect
        A) migrated all three bridges to the round-keyed ``add_pending_
        round``/``discard_pending_round`` instead, since a plain boolean
        cannot represent "N independent rounds outstanding for one
        session" without one clobbering another's clear.

        This shim survives for the ONE remaining caller genuinely without a
        round id of its own: ``ChatScreen._park_console_approval`` (wired
        as ``park_pending_approval``), whose own public contract is a
        single-arg ``Callable[[str], None]`` with no room for a round id --
        changing that would ripple into every test that wires ``park_
        pending_approval = some_list.append`` -- and it is ALSO used
        directly, standalone, by tests exercising the marker/badge
        lifecycle without a live round (mirrors how those tests already
        drive other controller seams directly).

        Internally represented as the reserved
        ``_LEGACY_PENDING_APPROVAL_ROUND_ID`` sentinel round id, so it
        composes safely alongside real round ids in the same per-session
        set -- ``pending=True`` adds the sentinel, ``pending=False``
        discards ONLY the sentinel (a real round registered separately via
        ``add_pending_round`` is untouched either way). Because of this, a
        caller that calls this with ``pending=True`` while a REAL round is
        ALREADY registered for the session adds a harmless, redundant
        no-op-visible entry -- but that same caller must not rely on this
        call's own ``pending=False`` (or a real round's ``discard_pending_
        round``) to fully clear the badge on its own; whichever one runs
        last is the one that actually clears it. ``ChatScreen._park_
        console_approval`` avoids this ambiguity by checking ``has_
        pending_approval_round`` first and only falling back to this shim
        when no real round is registered yet.

        Args:
            session_id: The session whose pending-approval flag to update.
            pending: ``True`` to mark the session as awaiting a decision,
                ``False`` to clear it.
        """
        if pending:
            self.add_pending_round(session_id, _LEGACY_PENDING_APPROVAL_ROUND_ID)
        else:
            self.discard_pending_round(session_id, _LEGACY_PENDING_APPROVAL_ROUND_ID)

    def mark_session_visited(self, session_id: str) -> None:
        """Clear ``session_id``'s unvisited terminal outcome.

        Parallel-agents spec §6. Called from ``switch_session`` once the
        store has swapped to ``session_id`` -- visiting a session is what
        "sees" its terminal outcome, so that marker resets to steady state
        (``run_marker_for`` then falls through to ``ConsoleRunMarker.NONE``
        unless a fresh run starts).

        Task 9 correction: this used to ALSO discard ``session_id``'s
        pending-approval flag, which directly contradicted the parked-
        approval design once background sessions could carry a live
        approval round -- a plain visit (e.g. just checking on a background
        session, or the auto-mount ``switch_session`` now performs to show
        its parked card) would silently deny-in-spirit the outstanding
        round's badge before the human ever made a decision. The flag now
        clears ONLY on the round's own resolution (``request_mcp_
        approvals``' ``finally``) or a terminal run-state transition
        (``_set_run_state``) -- never merely from being looked at. See
        ``switch_session`` for the (separate) mount-the-parked-card step
        that visiting now performs instead.

        Args:
            session_id: The session just switched to (or otherwise
                visited).
        """
        self._unvisited_outcomes.pop(session_id, None)

    def run_marker_for(self, session_id: str) -> ConsoleRunMarker:
        """Fleet-visible marker for ``session_id`` (parallel-agents spec §6).

        Precedence, checked in order:

        1. ``NEEDS_APPROVAL`` -- outranks ``RUNNING`` even though a parked
           run is technically still in-flight: the marker must announce
           the thing that needs a human, not just "something is
           happening".
        2. ``RUNNING`` -- derived from the same live/busy definition as
           ``in_flight_run_count`` (``_live_busy_session_ids``), so this
           never invents a second notion of "in-flight".
        3. ``FINISHED_OK``/``FINISHED_FAILED`` -- from
           ``_unvisited_outcomes``, stamped only for non-active sessions
           by ``_set_run_state``'s terminal transitions and cleared by
           ``mark_session_visited``.
        4. ``NONE`` otherwise.

        Args:
            session_id: The session to compute the marker for.

        Returns:
            The single ``ConsoleRunMarker`` that best describes
            ``session_id``'s current fleet-visible state, per the
            precedence above.
        """
        # TASK-1050: `_pending_approvals` is keyed by session id to a SET of
        # outstanding round ids (see `add_pending_round`) -- a plain `in`
        # dict-key check is exactly "does this session have ANY pending
        # round", which is what NEEDS_APPROVAL means; `add_pending_round`/
        # `discard_pending_round` guarantee an emptied round set is popped
        # rather than left behind as a stale `{}`, so this can never read
        # "pending" for a session with zero live rounds.
        if session_id in self._pending_approvals:
            return ConsoleRunMarker.NEEDS_APPROVAL
        if session_id in self._live_busy_session_ids():
            return ConsoleRunMarker.RUNNING
        return self._unvisited_outcomes.get(session_id, ConsoleRunMarker.NONE)

    def fleet_summary_counts(self) -> tuple[int, int]:
        """Counts of OTHER live sessions running / needing approval.

        Parallel-agents spec §6. Returns ``(other running, other pending-
        approval)`` relative to the active (viewed) session -- its own
        status is visible directly in the transcript, not through the
        fleet summary, so it is excluded from both counts. Sessions the
        store no longer has (orphaned ``_pending_approvals``/`
        `_run_states`` entries) are excluded via the same live-session
        filter ``_live_busy_session_ids`` applies. A session that is both
        busy and pending-approval is counted only as pending, mirroring
        ``run_marker_for``'s NEEDS_APPROVAL-outranks-RUNNING precedence --
        neither count double-books it.

        Returns:
            A ``(other_running, other_pending_approval)`` tuple of counts,
            both excluding the active (viewed) session.
        """
        active = self.store.active_session_id or ""
        live_ids = {session.id for session in self.store.sessions()}
        # F2b fix (Qodo wave): snapshot under the lock rather than
        # iterating `_pending_approvals` live -- this runs on the UI
        # thread's ~0.2s sync tick while a worker thread can concurrently
        # add/discard entries (`request_mcp_approvals`'s own body/
        # `finally`), so an unguarded comprehension here risked
        # `RuntimeError: Set changed size during iteration`. The
        # comprehension itself runs OUTSIDE the lock, over the snapshot.
        with self._approval_state_lock:
            pending_snapshot = set(self._pending_approvals)
        other_pending = {
            sid for sid in pending_snapshot if sid in live_ids and sid != active
        }
        other_busy = {sid for sid in self._live_busy_session_ids() if sid != active}
        return len(other_busy - other_pending), len(other_pending)

    def busy_fleet_session_count(self) -> int:
        """Count of LIVE sessions ``shutdown()`` would tear down right now.

        TASK-1143 (F5): union of ``_live_busy_session_ids()`` (a session
        with an active stream/citation-repair task -- the same set
        ``in_flight_run_count`` reports) and every LIVE session with at
        least one outstanding approval-like round, mounted or parked
        (``_pending_approvals``, the same registry ``run_marker_for``'s
        NEEDS_APPROVAL branch and ``has_pending_approval_round`` read --
        MCP tool approvals, skill-install, and skill-script confirms all
        register through the same ``add_pending_round``). A session that
        is both busy and mid-approval is counted once: this answers "how
        many agent runs" for fleet-teardown UX (the Console
        confirm-on-navigate guard and its post-navigate record), not "how
        many independent events" -- no new definition of "busy" beyond
        the union of the two predicates those existing callers already
        use.

        PR3a-1 Task 6b (audit F5, second half): a THIRD leg -- sessions
        holding a surviving fleet child. Since PR3a-1 a sub-agent keeps
        running after the turn that spawned it returns, and such a session
        has no active stream task and no pending approval round, so the two
        legs above both read it as idle. The confirm dialog therefore told
        the user "0 runs will be killed" and `shutdown()` then killed one:
        a dialog that lies to the user is worse than no dialog. Reproduced
        by execution in `Tests/Chat/test_console_agent_bridge.py::test_busy_
        fleet_session_count_sees_a_session_whose_only_work_is_a_survivor`.

        Returns:
            The number of live sessions with in-flight work, an outstanding
            approval-like round, and/or a still-running sub-agent -- 0 when
            the fleet is idle.
        """
        killed, surviving = self.fleet_teardown_split()
        return killed + surviving

    def fleet_teardown_split(self) -> tuple[int, int]:
        """Partition ``busy_fleet_session_count``'s union by teardown fate.

        PR3a-2 Task 4 (Task 1 A4, executed): post-3a-1 the two halves of
        that union meet OPPOSITE fates at ``shutdown()``, and the
        teardown notice was reporting both as "cancelled":

        - **killed**: sessions with an active stream task or an
          outstanding approval-like round. Shutdown's ``_signal_stop``
          fanout sets the in-flight turn's own cancel event, and the
          TURN genuinely dies. PR3b Task 5 (spec Sec 8) narrowed what
          dies with it: under the shipped ``subagents_outlive_turn``
          default a cancelled turn's still-running children now SURVIVE
          the stop and continue as background survivors
          (``AgentService._surviving_handles``; pinned by
          ``Tests/Agents/test_fleet_stop_semantics.py``), so "killed"
          describes the session's run, and its children only under the
          kill switch (``test_fleet_runtime.py::test_stopping_the_turn_
          still_stops_its_children``, now pinned turn-scoped). The
          teardown notice therefore under-reports in the same direction
          the "surviving" bullet already documents: a killed run's
          surviving children go unmentioned there and are reported by
          their own settle toasts instead.
        - **surviving**: sessions whose ONLY busy-ness is a cross-turn
          survivor. Task 1 A1 executed the fate: no cancel signal ever
          reaches such a child (its turn's cancel event was popped when
          the turn settled), it runs to completion and its terminal row
          lands durably after the screen is gone.

        Disjoint by construction -- a session holding BOTH an active
        stream and earlier-turn survivors lands in **killed** (its
        in-flight turn really is killed; the stated under-report is that
        its earlier survivors' continuing goes unmentioned in the
        next-mount notice -- their own settle toast still reports them).
        ``killed + surviving`` therefore equals the union the navigation
        confirm has always shown.

        Returns:
            ``(killed, surviving)`` live-session counts.
        """
        live_ids = {session.id for session in self.store.sessions()}
        with self._approval_state_lock:
            pending_ids = set(self._pending_approvals)
        killed_ids = set(self._live_busy_session_ids()) | (pending_ids & live_ids)
        surviving_ids = self._fleet_survivor_session_ids() - killed_ids
        return len(killed_ids), len(surviving_ids)

    def fleet_has_unsettled_children(self) -> bool:
        """Whether ANY live session's conversation is still owed a drain.

        PR3a-2 Task 4: the drive/stop condition for the screen's survivor
        tick (task-15664). Reads the bridge's drain-paired unsettled
        counter (``has_unsettled_children``, Task 3) per live session --
        cheap dict reads under one lock, safe on a UI timer, unlike
        ``_fleet_survivor_session_ids``'s coordinator sweep (which
        ``busy_fleet_session_count`` documents as navigation-only and
        which task-15666 records as prune-on-read). True exactly while at
        least one fleet child of a live session has entered its run scope
        and not yet reached its settle hook -- so the tick keeps painting
        through the scope-exit->settle window and stops on the same edge
        the drain (and the badge it stamps) fires on.

        Returns:
            True while any live session's fleet still owes a drain.
        """
        bridge = self._agent_bridge
        checker = (
            getattr(bridge, "has_unsettled_children", None)
            if bridge is not None
            else None
        )
        if not callable(checker):
            return False
        for session in self.store.sessions():
            try:
                if checker(self._agent_conversation_id(session.id)):
                    return True
            except Exception as exc:  # noqa: BLE001 -- a UI timer must never crash on a read
                logger.debug(
                    "fleet unsettled check failed for a session; treated as idle (exception_type={})",
                    type(exc).__name__,
                )
        return False

    def _fleet_survivor_session_ids(self) -> set[str]:
        """Live sessions with at least one still-running sub-agent.

        Reads the agent bridge's own live coordinator view
        (`fleet_snapshot`, per conversation) rather than any DB row: a
        child's real status lives in `FleetCoordinator` while it runs, and
        `agent_runs` only catches up when it ends. Terminal handles are
        filtered out -- `fleet_snapshot` includes them while a run is in
        flight, and a finished child is nothing teardown would kill.

        Called only from `busy_fleet_session_count` (navigation confirm and
        the teardown record), never on the rail's 0.2s tick. Degrades to an
        empty set with no bridge, no `fleet_snapshot`, or a raising one:
        under-counting is the pre-PR3a-1 behaviour, and this must never be
        the thing that breaks a navigation.
        """
        from tldw_chatbook.Agents.agent_models import TERMINAL_RUN_STATUSES

        bridge = self._agent_bridge
        # `is not None`, not truthiness: a bridge double defining `__len__`
        # would otherwise read as absent.
        snapshot = (
            getattr(bridge, "fleet_snapshot", None) if bridge is not None else None
        )
        if snapshot is None:
            return set()
        busy: set[str] = set()
        for session in self.store.sessions():
            try:
                handles = snapshot(self._agent_conversation_id(session.id))
            except Exception:  # noqa: BLE001 -- never block a navigation
                logger.debug(
                    "fleet survivor count failed for a session; treated as idle"
                )
                continue
            if any(
                getattr(handle, "status", "") not in TERMINAL_RUN_STATUSES
                for handle in handles or ()
            ):
                busy.add(session.id)
        return busy

    @property
    def max_parallel_runs(self) -> int:
        """User-adjustable global cap on simultaneous runs (parallel-agents spec §4).

        Reads ``[console] max_parallel_runs`` through the same
        ``get_cli_setting`` seam used elsewhere in this module (see
        ``_resolve_mcp_approval_timeout_seconds``). Floored at 1 and
        defaulted to ``CONSOLE_DEFAULT_MAX_PARALLEL_RUNS`` so a bad/blank
        config value can never lock every session out of sending.

        Returns:
            The configured cap on simultaneous runs, floored at 1.
        """
        raw = get_cli_setting(
            "console", "max_parallel_runs", CONSOLE_DEFAULT_MAX_PARALLEL_RUNS
        )
        if raw is None:
            return CONSOLE_DEFAULT_MAX_PARALLEL_RUNS
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = CONSOLE_DEFAULT_MAX_PARALLEL_RUNS
        return max(1, value)

    def send_refusal_copy(self, session_id: str) -> str | None:
        """Why a send to ``session_id`` must be refused right now, or ``None``.

        Parallel-agents spec §4. Two gates, checked in order:

        1. Per-session -- ``session_id``'s own run is still in flight.
        2. Global cap -- ``max_parallel_runs`` busy sessions already exist,
           so a NEW send (from any session, including an idle one) must
           wait.

        The cap's busy list comes from ``_live_busy_session_ids`` (shared
        with ``in_flight_run_count``): a session closed mid-VALIDATING
        leaves its entry in ``self._run_states`` behind
        (``ConsoleChatStore.close_session`` never touches the controller's
        map -- Task 1 review finding), and a session that no longer exists
        must not consume a cap slot or be named in the refusal copy.

        Args:
            session_id: The session id attempting to send.

        Returns:
            A human-readable refusal message if the send must be blocked
            right now, otherwise ``None`` when the send is allowed.
        """
        interrupted = self.store.interrupted_provider_continuation_message(session_id)
        if interrupted is not None and not self.provider_continuation_owner_is_live(
            interrupted.id
        ):
            return PROVIDER_CONTINUATION_RECOVERY_REQUIRED
        if self.prompt_queue_coordinator.controls_generation(session_id):
            return (
                "Queued messages control the next turn. "
                "Resume or manage the queue first."
            )
        if not self.run_state_for(session_id).is_send_allowed:
            return "A run is already running in this tab."
        if self.store.dispatch_recovery_blocks_submission(session_id):
            return (
                "Finish or discard the pending response before sending another "
                "message."
            )
        busy_ids = self._live_busy_session_ids()
        if len(busy_ids) < self.max_parallel_runs:
            return None
        live_sessions = {session.id: session for session in self.store.sessions()}
        limit = CONSOLE_CAP_REFUSAL_TITLE_LIMIT
        titles = [live_sessions[sid].title for sid in busy_ids[:limit]]
        suffix = f" and {len(busy_ids) - limit} more" if len(busy_ids) > limit else ""
        busy_count = len(busy_ids)
        # Fleet-UX expert review F7 (task-1234): number agreement -- "1
        # agents already running" read as a grammar bug on the very first
        # cap refusal a solo user could ever see (max_parallel_runs=1).
        agent_noun = "agent" if busy_count == 1 else "agents"
        return (
            f"{busy_count} {agent_noun} already running "
            f"({', '.join(titles)}{suffix}). "
            "Wait for one to finish or interrupt it."
        )

    async def recover_provider_continuation(
        self,
        action: str,
        message_id: str,
        expected_message_version: int,
    ) -> bool:
        """Perform one explicit, optimistic recovery action."""
        if action not in {"resume", "take_over", "discard"}:
            return False
        try:
            message = self.store.get_message(message_id)
            checkpoint = message.provider_continuation
            session_id = self.store.session_id_for_message(message_id)
        except KeyError:
            return False
        if self.provider_continuation_owner_is_live(message_id):
            self.store.set_provider_continuation_warning(
                message.id,
                "This tool run is still active. Wait for it to finish or Stop it before recovery.",
            )
            return False
        if session_id in self._provider_continuation_recovery_sessions:
            self.store.set_provider_continuation_warning(
                message.id,
                "A recovery action is already running. Wait for it to finish.",
            )
            return False
        self._provider_continuation_recovery_sessions.add(session_id)
        try:
            if not self._provider_continuation_recovery_target_is_current(
                session_id=session_id,
                message_id=message.id,
                expected_message_version=expected_message_version,
            ):
                return False
            if action == "discard":
                try:
                    return self.store.discard_provider_continuation(
                        message_id,
                        expected_message_version=expected_message_version,
                    )
                except Exception:
                    return False
            return await self._resume_provider_continuation(
                action=action,
                message=message,
                checkpoint=checkpoint,
                session_id=session_id,
                expected_message_version=expected_message_version,
            )
        finally:
            self._provider_continuation_recovery_sessions.discard(session_id)

    async def _resume_provider_continuation(
        self,
        *,
        action: str,
        message: ConsoleChatMessage,
        checkpoint: ProviderContinuationCheckpoint | None,
        session_id: str,
        expected_message_version: int,
    ) -> bool:
        """Validate and execute one serialized Resume or Take over action."""
        if (
            checkpoint is None
            or checkpoint.state != "active"
            or message.provider_continuation_message_version != expected_message_version
            or (message.provider_continuation_remote and action != "take_over")
            or (not message.provider_continuation_remote and action != "resume")
            or any(
                call.state == "executing"
                for round_ in checkpoint.rounds
                for call in round_.calls
            )
        ):
            return False
        translator = getattr(
            self.provider_gateway, "expand_provider_continuation", None
        )
        if not callable(translator) or self._agent_bridge is None:
            self.store.set_provider_continuation_warning(
                message.id,
                "Continuation replay support is not enabled for this provider integration. "
                "Enable or configure it, or Discard the interrupted run.",
            )
            return False
        (
            resolution,
            turn_context,
        ) = await self._capture_and_resolve_turn_execution_context(session_id)
        if not self._provider_continuation_recovery_target_is_current(
            session_id=session_id,
            message_id=message.id,
            expected_message_version=expected_message_version,
        ):
            return False
        if not getattr(resolution, "ready", False):
            self.store.set_provider_continuation_warning(
                message.id,
                "Provider credentials are not ready. Fix Settings, then retry.",
            )
            return False
        assert turn_context is not None
        target = _continuation_restore_target_for_resolution(resolution)
        if target is None:
            self.store.set_provider_continuation_warning(
                message.id,
                "Pinned provider settings are incomplete. Restore those settings or Discard.",
            )
            return False
        try:
            validate_continuation_restore(checkpoint, target)
        except Exception:
            self.store.set_provider_continuation_warning(
                message.id,
                "Pinned provider settings no longer match. Restore those settings or Discard.",
            )
            return False
        prior_sidecar, prior_target = (
            self._provider_continuation_resume_history_for_resolution(
                session_id,
                resolution,
                before_message_id=message.id,
            )
        )
        recovery_task = asyncio.current_task()
        try:
            await self._run_agent_reply(
                resolution=resolution,
                provider_messages=self._provider_messages_for_session(
                    session_id,
                    before_message_id=message.id,
                    annotate_ids=bool(prior_sidecar),
                    turn_context=turn_context,
                ),
                assistant_message_id=message.id,
                prepare_retry=False,
                variant_mode=False,
                restore_provider_continuation=checkpoint,
                restore_provider_target=target,
                expand_provider_continuation=translator,
                resume_provider_continuation=True,
                continuation_sidecar=prior_sidecar,
                continuation_history_target=prior_target,
                turn_context=turn_context,
            )
        finally:
            if (
                self._active_stream_tasks.get(session_id) is recovery_task
                and self._active_assistant_message_ids.get(session_id) == message.id
            ):
                self._active_stream_tasks.pop(session_id, None)
                self._active_assistant_message_ids.pop(session_id, None)
                self._active_cancel_events.pop(session_id, None)
                self._stop_requested = False
        try:
            current = self.store.get_message(message.id)
        except KeyError:
            return False
        current_checkpoint = current.provider_continuation
        if current_checkpoint is None or current_checkpoint.state != "active":
            return True
        status = self.run_state_for(session_id).status
        if status is ConsoleRunStatus.STOPPED:
            warning = (
                "Recovery stopped before the interrupted run completed. "
                "Resume again or Discard it."
            )
        elif status is ConsoleRunStatus.FAILED:
            warning = (
                "Recovery failed before the interrupted run completed. "
                "Check provider settings and retry, or Discard it."
            )
        else:
            warning = (
                "Recovery did not complete the interrupted run. "
                "Retry when the provider is ready, or Discard it."
            )
        self.store.set_provider_continuation_warning(message.id, warning)
        return False

    def provider_continuation_owner_is_live(self, message_id: str) -> bool:
        """Return whether the exact continuation owner still belongs to a live run."""
        try:
            session_id = self.store.session_id_for_message(message_id)
        except KeyError:
            return False
        return (
            self._active_assistant_message_ids.get(session_id) == message_id
            and not self.run_state_for(session_id).is_send_allowed
        )

    def _provider_continuation_recovery_target_is_current(
        self,
        *,
        session_id: str,
        message_id: str,
        expected_message_version: int,
    ) -> bool:
        """Fail closed unless the requested owner is active and durably current."""
        current = self.store.interrupted_provider_continuation_message(session_id)
        if (
            current is None
            or current.id != message_id
            or not current.provider_continuation_actions_enabled
            or current.provider_continuation_message_version != expected_message_version
            or current.persisted_message_id is None
        ):
            return False
        version_reader = getattr(self.store.persistence, "get_message_version", None)
        if not callable(version_reader):
            return False
        try:
            return (
                version_reader(current.persisted_message_id) == expected_message_version
            )
        except Exception:
            return False

    def provider_continuation_recovery_message(
        self,
    ) -> ConsoleChatMessage | None:
        """Return the recoverable owner, excluding this controller's live run."""
        message = self.store.provider_continuation_recovery_message()
        if message is None or self.provider_continuation_owner_is_live(message.id):
            return None
        return message

    def provider_continuation_replay_available(self) -> bool:
        """Return whether the selected gateway exposes a replay translator."""
        return self._agent_bridge is not None and callable(
            getattr(self.provider_gateway, "expand_provider_continuation", None)
        )

    @property
    def run_state_history(self) -> list[ConsoleRunStatus]:
        """The ACTIVE session's run-status history (read-only facade, mirrors ``run_state``).

        Returns:
            The active session's list of recorded ``ConsoleRunStatus`` values.
        """
        return self.run_state_history_for(self.store.active_session_id or "")

    def run_state_history_for(self, session_id: str) -> list[ConsoleRunStatus]:
        """Return (creating if absent) ``session_id``'s run-status history.

        Args:
            session_id: The session id to look up.

        Returns:
            The session's list of recorded ``ConsoleRunStatus`` values,
            initialized to ``[ConsoleRunStatus.IDLE]`` when absent.
        """
        return self._run_state_histories.setdefault(session_id, [ConsoleRunStatus.IDLE])

    def _preparation_by_id(self, preparation_id: str) -> ConsoleTurnPreparation | None:
        """Resolve one store-owned preparation without mirroring its state."""

        if not isinstance(preparation_id, str) or not preparation_id:
            return None
        return self.store.preparation_by_id(preparation_id)

    @staticmethod
    def _automatic_scope_for_authority(
        authority: ConsoleTurnLibraryAuthority,
    ) -> EffectiveScope | None:
        """Translate the frozen Task-8 scope snapshot without live re-reads."""

        snapshot = authority.scope_snapshot
        if snapshot.conversations_allowed:
            return None
        allowlist: dict[str, frozenset[str]] = {}
        if snapshot.note_ids:
            allowlist["notes"] = frozenset(snapshot.note_ids)
        if snapshot.media_ids:
            allowlist["media"] = frozenset(snapshot.media_ids)
        if not allowlist:
            return EffectiveScope(
                state="empty",
                allowlist={},
                cause="no-workspace-overlap",
            )
        return EffectiveScope(state="scoped", allowlist=allowlist, cause=None)

    @staticmethod
    def _preparation_contribution(
        outcome: Literal["zero_matches", "bypassed"],
        preparation: ConsoleTurnPreparation,
    ) -> LibraryPreparationContribution:
        event = library_preparation_event_for_outcome(
            outcome,
            attempt_id=preparation.attempt_id,
            result_count=0,
            source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
        )
        assert event is not None
        return LibraryPreparationContribution(event=event)

    def preparation_outcome(
        self, preparation_id: str
    ) -> ConsolePreparationOutcome | None:
        """Return the last immutable payload attached to this preparation."""

        return self._preparation_outcomes.get(preparation_id)

    async def prepare_library_for_turn(
        self, preparation_id: str
    ) -> ConsolePreparationOutcome:
        """Run one frozen automatic retrieval attempt and CAS its result."""

        preparation = self._preparation_by_id(preparation_id)
        if preparation is None:
            raise KeyError(preparation_id)
        if preparation.state is not ConsoleTurnPreparationState.PREPARING:
            existing = self._preparation_outcomes.get(preparation_id)
            if existing is not None and existing.attempt_id == preparation.attempt_id:
                return existing
            return ConsolePreparationOutcome(
                preparation_id=preparation.preparation_id,
                attempt_id=preparation.attempt_id,
                state=preparation.state,
                evidence_bundle=None,
                contribution=None,
                error_code=None,
            )

        authority = preparation.execution_context.library_authority
        request = LibraryRagSearchRequest(
            query=preparation.executed_draft,
            source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
            mode="rag",
            top_k=5,
            include_citations=True,
            scope=self._automatic_scope_for_authority(authority),
        )
        error_code: str | None = None
        try:
            async with asyncio.timeout(self._library_preparation_timeout):
                service = getattr(self.app, "library_rag_search_service", None)
                search = getattr(service, "search", None)
                if not callable(search):
                    raise RuntimeError("library service unavailable")
                kwargs: dict[str, object] = {
                    "top_k": request.top_k,
                    "include_citations": request.include_citations,
                }
                if request.scope is not None:
                    kwargs["scope"] = request.scope
                raw_result = search(
                    request.query,
                    request.source_types,
                    request.mode,
                    **kwargs,
                )
                if inspect.isawaitable(raw_result):
                    raw_result = await raw_result
                result = _outcome_from_service_result(raw_result)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            result = None
            error_code = "library_retrieval_timeout"
        except Exception:
            result = None
            error_code = "library_retrieval_failed"

        results = tuple(getattr(result, "results", ()) or ()) if result else ()
        status = str(getattr(result, "status", "") or "") if result else ""
        if error_code is None and status not in {"ready", "empty"}:
            error_code = "library_retrieval_failed"
        if error_code is not None:
            paused = self.store.compare_and_set_preparation(
                preparation.session_id,
                ConsolePreparationTransition(
                    preparation_id=preparation.preparation_id,
                    expected_state=ConsoleTurnPreparationState.PREPARING,
                    new_state=ConsoleTurnPreparationState.PAUSED,
                    pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
                    new_attempt_id=None,
                ),
            )
            current = paused or self._preparation_by_id(preparation_id)
            if current is None:
                outcome = ConsolePreparationOutcome(
                    preparation_id=preparation.preparation_id,
                    attempt_id=preparation.attempt_id,
                    state=ConsoleTurnPreparationState.CANCELLED,
                    evidence_bundle=None,
                    contribution=None,
                    error_code=None,
                )
                return outcome
            outcome = ConsolePreparationOutcome(
                preparation_id=current.preparation_id,
                attempt_id=current.attempt_id,
                state=current.state,
                evidence_bundle=None,
                contribution=None,
                error_code=error_code if paused is not None else None,
            )
            self._preparation_outcomes[preparation_id] = outcome
            return outcome

        bundle = (
            build_library_rag_evidence_bundle(
                results,
                query=preparation.executed_draft,
            )
            if results
            else None
        )
        ready = self.store.compare_and_set_preparation(
            preparation.session_id,
            ConsolePreparationTransition(
                preparation_id=preparation.preparation_id,
                expected_state=ConsoleTurnPreparationState.PREPARING,
                new_state=ConsoleTurnPreparationState.READY,
                pause_kind=None,
                new_attempt_id=None,
            ),
        )
        current = ready or self._preparation_by_id(preparation_id)
        if current is None:
            outcome = ConsolePreparationOutcome(
                preparation_id=preparation.preparation_id,
                attempt_id=preparation.attempt_id,
                state=ConsoleTurnPreparationState.CANCELLED,
                evidence_bundle=None,
                contribution=None,
                error_code=None,
            )
            return outcome
        contribution = (
            self._preparation_contribution("zero_matches", current)
            if ready is not None and bundle is None
            else None
        )
        outcome = ConsolePreparationOutcome(
            preparation_id=current.preparation_id,
            attempt_id=current.attempt_id,
            state=current.state,
            evidence_bundle=bundle if ready is not None else None,
            contribution=contribution,
            error_code=None,
        )
        self._preparation_outcomes[preparation_id] = outcome
        return outcome

    async def retry_library_preparation(
        self, preparation_id: str
    ) -> ConsoleSubmitResult:
        """Retry a retrieval pause with one fresh attempt and frozen authority."""

        preparation = self._preparation_by_id(preparation_id)
        if preparation is None:
            return self._prepared_action_refusal()
        if preparation.state is not ConsoleTurnPreparationState.PAUSED:
            return self._prepared_action_refusal(preparation)
        if preparation.pause_kind is not ConsolePreparationPauseKind.RETRIEVAL:
            return await self._continue_prepared_submission(preparation_id)
        retried = self.store.compare_and_set_preparation(
            preparation.session_id,
            ConsolePreparationTransition(
                preparation_id=preparation.preparation_id,
                expected_state=ConsoleTurnPreparationState.PAUSED,
                new_state=ConsoleTurnPreparationState.PREPARING,
                pause_kind=None,
                new_attempt_id=str(uuid4()),
            ),
        )
        if retried is None:
            return self._prepared_action_refusal(
                self._preparation_by_id(preparation_id)
            )
        self._preparation_outcomes.pop(preparation_id, None)
        outcome = await self.prepare_library_for_turn(preparation_id)
        if outcome.state is not ConsoleTurnPreparationState.READY:
            return ConsoleSubmitResult(
                False, False, "Library preparation remains paused."
            )
        return await self._continue_prepared_submission(preparation_id)

    async def bypass_library_preparation(
        self, preparation_id: str
    ) -> ConsoleSubmitResult:
        """Advance one retrieval pause without changing its standing policy."""

        preparation = self._preparation_by_id(preparation_id)
        if preparation is None:
            return self._prepared_action_refusal()
        if (
            preparation.state is not ConsoleTurnPreparationState.PAUSED
            or preparation.pause_kind is not ConsolePreparationPauseKind.RETRIEVAL
        ):
            return self._prepared_action_refusal(preparation)
        ready = self.store.compare_and_set_preparation(
            preparation.session_id,
            ConsolePreparationTransition(
                preparation_id=preparation.preparation_id,
                expected_state=ConsoleTurnPreparationState.PAUSED,
                new_state=ConsoleTurnPreparationState.READY,
                pause_kind=None,
                new_attempt_id=None,
            ),
        )
        if ready is None:
            return self._prepared_action_refusal(
                self._preparation_by_id(preparation_id)
            )
        outcome = ConsolePreparationOutcome(
            preparation_id=ready.preparation_id,
            attempt_id=ready.attempt_id,
            state=ready.state,
            evidence_bundle=None,
            contribution=self._preparation_contribution("bypassed", ready),
            error_code=None,
        )
        self._preparation_outcomes[preparation_id] = outcome
        return await self._continue_prepared_submission(preparation_id)

    def _prepared_continuation_block_copy(
        self, resolution: object, expected_destination: object
    ) -> str:
        """Why a prepared continuation must not proceed, or "" to proceed.

        Qodo review of PR #2131: bounding this path made the resolver
        RETURN a not-ready stand-in on timeout rather than raise, and the
        caller's single combined condition reported every such case as
        "Prepared destination changed." — the wrong reason, and it dropped
        the recovery guidance the bound exists to deliver. Not-ready and
        destination-changed are now distinct answers; both remain
        retryable (the pause kind's action set is unchanged).

        Args:
            resolution: The gateway resolution (or the bounded stand-in).
            expected_destination: The destination frozen at preparation.

        Returns:
            User-facing refusal copy, or "" when the continuation may run.
        """
        if not getattr(resolution, "ready", False):
            return self._blocked_visible_copy(
                str(getattr(resolution, "visible_copy", "") or "").strip()
            )
        destination = getattr(resolution, "resolved_destination", None)
        if (
            not isinstance(destination, ConsoleResolvedDestination)
            or destination != expected_destination
        ):
            return "Prepared destination changed."
        return ""

    async def _continue_prepared_submission(
        self, preparation_id: str
    ) -> ConsoleSubmitResult:
        """Resume the exact frozen send after Retry or one-shot Bypass."""

        preparation = self._preparation_by_id(preparation_id)
        continuation = self._prepared_send_continuations.get(preparation_id)
        if preparation is None or continuation is None:
            return self._prepared_action_refusal(preparation)
        selection = preparation.execution_context.configuration.provider_selection
        try:
            resolution = await self._resolve_for_send_bounded(selection)
        except BaseException:
            self._pause_prepared_commit(
                preparation_id, ConsolePreparationPauseKind.DESTINATION_CHANGED
            )
            return self._prepared_action_refusal(
                self._preparation_by_id(preparation_id),
                "Prepared destination could not be verified.",
            )
        block_copy = self._prepared_continuation_block_copy(
            resolution, preparation.execution_context.resolved_destination
        )
        if block_copy:
            self._pause_prepared_commit(
                preparation_id, ConsolePreparationPauseKind.DESTINATION_CHANGED
            )
            return self._prepared_action_refusal(
                self._preparation_by_id(preparation_id), block_copy
            )
        current = self._preparation_by_id(preparation_id)
        if current is not None and current.state is ConsoleTurnPreparationState.PAUSED:
            current = self.store.compare_and_set_preparation(
                current.session_id,
                ConsolePreparationTransition(
                    preparation_id=preparation_id,
                    expected_state=ConsoleTurnPreparationState.PAUSED,
                    new_state=ConsoleTurnPreparationState.COMMITTING,
                    pause_kind=None,
                    new_attempt_id=None,
                ),
            )
            if current is None:
                return self._prepared_action_refusal(
                    self._preparation_by_id(preparation_id),
                    "Prepared turn changed before continuation.",
                )
        queue_authorization = None
        if preparation.origin == ConsoleSubmissionOrigin.QUEUED.value:
            assert preparation.queue_entry_id is not None
            try:
                queue_authorization = (
                    self.prompt_queue_coordinator.reclaim_prepared_entry(
                        preparation.session_id,
                        preparation.queue_entry_id,
                        preparation.preparation_id,
                    )
                )
            except BaseException:
                await self.prompt_queue_coordinator.finish_recovered_entry(
                    preparation.session_id, preparation.queue_entry_id, None
                )
                self._pause_prepared_commit(
                    preparation_id, ConsolePreparationPauseKind.PERSISTENCE
                )
                return self._prepared_action_refusal(
                    self._preparation_by_id(preparation_id),
                    "Queued preparation could not reclaim its entry.",
                )
            if queue_authorization is None:
                await self.prompt_queue_coordinator.finish_recovered_entry(
                    preparation.session_id, preparation.queue_entry_id, None
                )
                self._pause_prepared_commit(
                    preparation_id, ConsolePreparationPauseKind.PERSISTENCE
                )
                return self._prepared_action_refusal(
                    self._preparation_by_id(preparation_id),
                    "Queued preparation could not reclaim its entry.",
                )
        result: ConsoleSubmitResult | None = None
        try:
            result = await self.submit_draft(
                preparation.executed_draft,
                session_id=preparation.session_id,
                origin=ConsoleSubmissionOrigin(preparation.origin),
                queue_entry_id=preparation.queue_entry_id,
                queue_authorization=queue_authorization,
                _resume_preparation_id=preparation_id,
                _resume_resolution=resolution,
            )
            return result
        except BaseException:
            if (
                preparation.queue_entry_id is not None
                and self.prompt_queue_coordinator.recovered_entry_is_accepted(
                    preparation.session_id, preparation.queue_entry_id
                )
            ):
                assistant_message_id = None
                rows = self.store.messages_for_session(preparation.session_id)
                for index, row in enumerate(rows):
                    if row.id == preparation.transient_user_message_id:
                        assistant_message_id = next(
                            (
                                candidate.id
                                for candidate in rows[index + 1 :]
                                if candidate.role is ConsoleMessageRole.ASSISTANT
                            ),
                            None,
                        )
                        break
                result = ConsoleSubmitResult(
                    True,
                    True,
                    "Accepted turn failed before provider dispatch.",
                    session_id=preparation.session_id,
                    user_message_id=preparation.transient_user_message_id,
                    assistant_message_id=assistant_message_id,
                    terminal_status=ConsoleRunStatus.FAILED,
                    origin=ConsoleSubmissionOrigin.QUEUED,
                    queue_entry_id=preparation.queue_entry_id,
                )
                return result
            return self._prepared_action_refusal(
                self._preparation_by_id(preparation_id),
                "Prepared send could not continue.",
            )
        finally:
            if queue_authorization is not None:
                await self.prompt_queue_coordinator.finish_recovered_entry(
                    preparation.session_id,
                    preparation.queue_entry_id,
                    result,
                )

    def cancel_library_preparation(self, preparation_id: str) -> ConsoleSubmitResult:
        """Cancel one exact precommit preparation without provider dispatch."""

        preparation = self._preparation_by_id(preparation_id)
        if preparation is None:
            return self._prepared_action_refusal()
        cancelled = self.store.cancel_preparation(
            preparation.session_id,
            preparation.preparation_id,
            expected_state=preparation.state,
        )
        if cancelled is None:
            return self._prepared_action_refusal(
                self._preparation_by_id(preparation_id)
            )
        self._drop_preparation(
            preparation_id,
            expected_states=frozenset({ConsoleTurnPreparationState.CANCELLED}),
        )
        return self._prepared_action_refusal(None, "Library preparation canceled.")

    @staticmethod
    def _prepared_action_refusal(
        preparation: ConsoleTurnPreparation | None = None,
        visible_copy: str = "Prepared turn is no longer available.",
    ) -> ConsoleSubmitResult:
        """Return the stable public result shape for action races/refusals."""

        return ConsoleSubmitResult(
            False,
            False,
            visible_copy,
            session_id=preparation.session_id if preparation is not None else None,
            origin=(
                ConsoleSubmissionOrigin(preparation.origin)
                if preparation is not None
                else ConsoleSubmissionOrigin.MANUAL
            ),
            queue_entry_id=(
                preparation.queue_entry_id if preparation is not None else None
            ),
        )

    def _pause_prepared_commit(
        self,
        preparation_id: str,
        pause_kind: ConsolePreparationPauseKind,
    ) -> ConsoleTurnPreparation | None:
        """Move READY/PAUSED recovery work back to one explicit pause."""

        preparation = self._preparation_by_id(preparation_id)
        if preparation is None:
            return None
        if preparation.state is ConsoleTurnPreparationState.PAUSED:
            committing = self.store.compare_and_set_preparation(
                preparation.session_id,
                ConsolePreparationTransition(
                    preparation_id=preparation_id,
                    expected_state=ConsoleTurnPreparationState.PAUSED,
                    new_state=ConsoleTurnPreparationState.COMMITTING,
                    pause_kind=None,
                    new_attempt_id=None,
                ),
            )
        elif preparation.state is ConsoleTurnPreparationState.READY:
            committing = self.store.compare_and_set_preparation(
                preparation.session_id,
                ConsolePreparationTransition(
                    preparation_id=preparation_id,
                    expected_state=ConsoleTurnPreparationState.READY,
                    new_state=ConsoleTurnPreparationState.COMMITTING,
                    pause_kind=None,
                    new_attempt_id=None,
                ),
            )
        elif preparation.state is ConsoleTurnPreparationState.COMMITTING:
            committing = preparation
        else:
            return preparation
        if committing is None:
            return self._preparation_by_id(preparation_id)
        paused = self.store.compare_and_set_preparation(
            committing.session_id,
            ConsolePreparationTransition(
                preparation_id=preparation_id,
                expected_state=ConsoleTurnPreparationState.COMMITTING,
                new_state=ConsoleTurnPreparationState.PAUSED,
                pause_kind=pause_kind,
                new_attempt_id=None,
            ),
        )
        return paused or self._preparation_by_id(preparation_id)

    def _has_explicit_staged_evidence(self, session_id: str) -> bool | None:
        provider = self._staged_evidence_provider
        if provider is None:
            owner = getattr(self._rag_capture_provider, "__self__", None)
            provider = getattr(owner, "_has_staged_evidence", None)
        if not callable(provider):
            return False
        try:
            return bool(provider(session_id))
        except TypeError:
            try:
                return bool(provider())
            except Exception:
                return None
        except Exception:
            return None

    def _snapshot_staged_evidence(
        self,
    ) -> tuple[bool, Any | None, Callable[[Any, Any], None] | None]:
        """Freeze the production retrieval owner's current live launch, if any."""

        owner = getattr(self._rag_capture_provider, "__self__", None)
        snapshot = getattr(owner, "_snapshot_console_staged_evidence", None)
        release = getattr(owner, "_release_frozen_console_staged_rag", None)
        release_callback = release if callable(release) else None
        if not callable(snapshot):
            return False, None, None
        try:
            return True, snapshot(), release_callback
        except Exception:
            return True, None, release_callback

    @staticmethod
    def _ordinary_library_text(
        draft: str,
        origin: ConsoleSubmissionOrigin,
        *,
        has_pending_attachment: bool,
    ) -> bool:
        """Closed Task-13 admission map; unknown future kinds skip spending."""

        admitted_origins = {
            ConsoleSubmissionOrigin.MANUAL: True,
            ConsoleSubmissionOrigin.QUEUED: True,
            ConsoleSubmissionOrigin.AGENT_WAKE: False,
        }
        if not admitted_origins.get(origin, False):
            return False
        stripped = draft.lstrip()
        return bool(stripped) and not stripped.startswith(
            (COMMAND_PREFIX, MENTION_SIGIL)
        )

    def _drop_preparation(
        self,
        preparation_id: str,
        *,
        expected_states: frozenset[ConsoleTurnPreparationState],
    ) -> None:
        """Remove one exact volatile owner and all controller sidecars."""

        preparation = self._preparation_by_id(preparation_id)
        if preparation is None:
            return
        removed = self.store.remove_preparation(
            preparation.session_id,
            preparation.preparation_id,
            expected_states=expected_states,
        )
        if removed is not None:
            self._preparation_outcomes.pop(preparation_id, None)
            self._prepared_send_continuations.pop(preparation_id, None)
            fingerprint = self.store.durable_acceptance_fingerprint_for(preparation_id)
            if (
                fingerprint is None
                or self.store.durable_turn_commit_for(
                    preparation_id, fingerprint=fingerprint
                )
                is None
            ):
                self.store.discard_uncommitted_durable_preparation(preparation_id)

    def _abandon_preparation(self, preparation_id: str) -> None:
        """Cancel and remove one exact preaccept preparation without a wedge."""

        preparation = self._preparation_by_id(preparation_id)
        if preparation is None:
            return
        cancelled = self.store.cancel_preparation(
            preparation.session_id,
            preparation.preparation_id,
            expected_state=preparation.state,
        )
        if cancelled is not None:
            self._drop_preparation(
                preparation_id,
                expected_states=frozenset({ConsoleTurnPreparationState.CANCELLED}),
            )

    def _rollback_committing_preparation(self, preparation_id: str) -> None:
        """Restore one failed preaccept commit through the legal store CAS path."""

        preparation = self._preparation_by_id(preparation_id)
        if preparation is None:
            return
        paused = self.store.compare_and_set_preparation(
            preparation.session_id,
            ConsolePreparationTransition(
                preparation_id=preparation.preparation_id,
                expected_state=ConsoleTurnPreparationState.COMMITTING,
                new_state=ConsoleTurnPreparationState.PAUSED,
                pause_kind=ConsolePreparationPauseKind.PERSISTENCE,
                new_attempt_id=None,
            ),
        )
        if paused is not None:
            self._abandon_preparation(preparation_id)

    def _settle_accepted_preparation(self, preparation_id: str) -> None:
        """Settle a volatile accepted owner after its live path exits."""

        preparation = self._preparation_by_id(preparation_id)
        if preparation is None:
            return
        if preparation.state is ConsoleTurnPreparationState.DISPATCH_STARTED:
            self._transition_preparation(
                preparation_id,
                ConsoleTurnPreparationState.DISPATCH_STARTED,
                ConsoleTurnPreparationState.DISPATCHED,
            )
        current = self._preparation_by_id(preparation_id)
        if current is None:
            return
        if current.state in {
            ConsoleTurnPreparationState.ACCEPTED,
            ConsoleTurnPreparationState.DISPATCHED,
        }:
            self._transition_preparation(
                preparation_id,
                current.state,
                ConsoleTurnPreparationState.SETTLED,
            )
        self._drop_preparation(
            preparation_id,
            expected_states=frozenset({ConsoleTurnPreparationState.SETTLED}),
        )

    def _transition_preparation(
        self,
        preparation_id: str,
        expected: ConsoleTurnPreparationState,
        new: ConsoleTurnPreparationState,
    ) -> bool:
        """Cross one exact volatile boundary at the matching real operation."""

        preparation = self._preparation_by_id(preparation_id)
        if preparation is None or preparation.state is not expected:
            return False
        return (
            self.store.compare_and_set_preparation(
                preparation.session_id,
                ConsolePreparationTransition(
                    preparation_id=preparation.preparation_id,
                    expected_state=expected,
                    new_state=new,
                    pause_kind=None,
                    new_attempt_id=None,
                ),
            )
            is not None
        )

    def _register_submit_task(self, task: asyncio.Task, session_id: str | None) -> None:
        """Register one exact submit owner without replacing a peer task."""

        with self._active_submit_tasks_lock:
            self._active_submit_tasks[task] = session_id or ""
            if self._owner_loop is None or self._owner_loop.is_closed():
                self._owner_loop = task.get_loop()

    def _rebind_submit_task(self, task: asyncio.Task, session_id: str) -> None:
        """Bind a provisional submit owner to its resolved session."""

        with self._active_submit_tasks_lock:
            if task in self._active_submit_tasks:
                self._active_submit_tasks[task] = session_id

    def _bind_submit_preparation(self, task: asyncio.Task, preparation_id: str) -> None:
        """Bind one exact volatile preparation to its submit owner."""

        with self._active_submit_tasks_lock:
            if task in self._active_submit_tasks:
                self._active_submit_preparations[task] = preparation_id

    def _begin_submit_preparation(
        self,
        task: asyncio.Task | None,
        preparation: ConsoleTurnPreparation,
    ) -> ConsoleTurnPreparation | None:
        """Begin and bind one preparation without a shutdown ownership gap."""

        if task is None:
            return self.store.begin_preparation(preparation)
        with self._active_submit_tasks_lock:
            begun = self.store.begin_preparation(preparation)
            if begun is not None and task in self._active_submit_tasks:
                self._active_submit_preparations[task] = begun.preparation_id
            return begun

    def _unregister_submit_task(self, task: asyncio.Task) -> None:
        """Remove only the completing submit task's own registry entry."""

        with self._active_submit_tasks_lock:
            self._active_submit_tasks.pop(task, None)
            self._active_submit_preparations.pop(task, None)

    def _submit_task_session(self, task: asyncio.Task) -> str:
        """Return the task's latest resolved session binding, if any."""

        with self._active_submit_tasks_lock:
            return self._active_submit_tasks.get(task, "")

    def _submit_tasks_snapshot(self) -> dict[asyncio.Task, str]:
        """Return a stable task-to-session snapshot for teardown."""

        with self._active_submit_tasks_lock:
            return dict(self._active_submit_tasks)

    def _submit_tasks_for_session(self, session_id: str) -> tuple[asyncio.Task, ...]:
        """Return every exact live submit owned by one session."""

        with self._active_submit_tasks_lock:
            return tuple(
                task
                for task, owner_session_id in self._active_submit_tasks.items()
                if owner_session_id == session_id
            )

    def _detach_closed_submit_tasks(self) -> tuple[str, ...]:
        """Detach closed-loop owners for emergency fail-closed cleanup.

        A closed event loop cannot terminally cancel or await its pending tasks
        through public asyncio APIs. This helper therefore removes only the
        controller's volatile ownership and returns exclusively owned
        preparations for synchronous cleanup. It does not promise a terminal
        Task state or suppress Python's destroyed-pending-task diagnostic.
        """

        with self._active_submit_tasks_lock:
            closed_preparations: list[str] = []
            for task in tuple(self._active_submit_tasks):
                if not task.get_loop().is_closed():
                    continue
                self._active_submit_tasks.pop(task, None)
                preparation_id = self._active_submit_preparations.pop(task, None)
                if preparation_id is not None:
                    closed_preparations.append(preparation_id)
            live_preparations = frozenset(self._active_submit_preparations.values())
        return tuple(
            preparation_id
            for preparation_id in dict.fromkeys(closed_preparations)
            if preparation_id not in live_preparations
        )

    def _cleanup_unreachable_preparation(self, preparation_id: str) -> None:
        """Synchronously remove one exact unreachable volatile Task-13 owner."""

        try:
            preparation = self._preparation_by_id(preparation_id)
            if preparation is None:
                return
            if preparation.state in {
                ConsoleTurnPreparationState.PREPARING,
                ConsoleTurnPreparationState.READY,
                ConsoleTurnPreparationState.PAUSED,
            }:
                self._abandon_preparation(preparation_id)
            elif preparation.state is ConsoleTurnPreparationState.COMMITTING:
                self._rollback_committing_preparation(preparation_id)
            elif preparation.state in {
                ConsoleTurnPreparationState.ACCEPTED,
                ConsoleTurnPreparationState.DISPATCH_STARTED,
                ConsoleTurnPreparationState.DISPATCHED,
            }:
                self._settle_accepted_preparation(preparation_id)
            elif preparation.state in {
                ConsoleTurnPreparationState.CANCELLED,
                ConsoleTurnPreparationState.SETTLED,
            }:
                self._drop_preparation(
                    preparation_id,
                    expected_states=frozenset({preparation.state}),
                )
        finally:
            self._preparation_outcomes.pop(preparation_id, None)
            self._prepared_send_continuations.pop(preparation_id, None)

    @staticmethod
    def _cancel_task_on_owner_loop(task: asyncio.Task) -> None:
        """Cancel an asyncio task only from its owning event-loop thread."""

        if task.done():
            return
        loop = task.get_loop()
        if loop.is_closed():
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            task.cancel()
            return
        try:
            loop.call_soon_threadsafe(lambda: None if task.done() else task.cancel())
        except RuntimeError:
            # The loop closed after the bounded check above. Its task cannot
            # be awaited or safely mutated from this foreign thread.
            return

    def _mark_transient_echo_blocked(self, message_id: str) -> None:
        """Fail a live echo, tolerating only an exact close-time removal."""

        try:
            self.store.mark_message_send_blocked(message_id)
        except KeyError:
            try:
                self.store.get_message(message_id)
            except KeyError:
                return
            raise

    async def submit_draft(
        self,
        draft: str,
        *,
        session_id: str | None = None,
        origin: ConsoleSubmissionOrigin = ConsoleSubmissionOrigin.MANUAL,
        queue_entry_id: str | None = None,
        queue_authorization: QueueGenerationAuthorization | None = None,
        wake_authorization: AgentWakeAuthorization | None = None,
        _resume_preparation_id: str | None = None,
        _resume_resolution: Any | None = None,
    ) -> ConsoleSubmitResult:
        """Fence one complete submit lifecycle for close and shutdown."""

        owner_key = session_id or self.store.active_session_id
        if self._disposed or (
            self._shutdown_requested.is_set()
            and origin is not ConsoleSubmissionOrigin.AGENT_WAKE
        ):
            return ConsoleSubmitResult(False, False, "Console is shutting down.")
        active_task = asyncio.current_task()
        with self._capture_quiescence_lock:
            if owner_key is not None and self.store.capture_quiescent(owner_key):
                return ConsoleSubmitResult(
                    False,
                    False,
                    "Stored captures are being updated; retry shortly.",
                    session_id=owner_key,
                    origin=origin,
                    queue_entry_id=queue_entry_id,
                )
            if active_task is not None:
                self._register_submit_task(active_task, owner_key)
        if active_task is None:
            return await self._submit_draft_inner(
                draft,
                session_id=session_id,
                origin=origin,
                queue_entry_id=queue_entry_id,
                queue_authorization=queue_authorization,
                wake_authorization=wake_authorization,
                _resume_preparation_id=_resume_preparation_id,
                _resume_resolution=_resume_resolution,
            )
        try:
            return await self._submit_draft_inner(
                draft,
                session_id=session_id,
                origin=origin,
                queue_entry_id=queue_entry_id,
                queue_authorization=queue_authorization,
                wake_authorization=wake_authorization,
                _resume_preparation_id=_resume_preparation_id,
                _resume_resolution=_resume_resolution,
            )
        except asyncio.CancelledError:
            bound_owner_key = self._submit_task_session(active_task) or owner_key
            if not self._shutdown_requested.is_set():
                if bound_owner_key and all(
                    session.id != bound_owner_key for session in self.store.sessions()
                ):
                    return ConsoleSubmitResult(
                        False,
                        False,
                        "Session closed before turn acceptance.",
                        session_closed=True,
                        session_id=bound_owner_key,
                        origin=origin,
                        queue_entry_id=queue_entry_id,
                    )
                raise
            return ConsoleSubmitResult(
                False,
                False,
                "Console shut down before turn acceptance.",
                session_id=bound_owner_key or None,
                origin=origin,
                queue_entry_id=queue_entry_id,
            )
        finally:
            self._unregister_submit_task(active_task)

    async def _submit_draft_inner(
        self,
        draft: str,
        *,
        session_id: str | None = None,
        origin: ConsoleSubmissionOrigin = ConsoleSubmissionOrigin.MANUAL,
        queue_entry_id: str | None = None,
        queue_authorization: QueueGenerationAuthorization | None = None,
        wake_authorization: AgentWakeAuthorization | None = None,
        _resume_preparation_id: str | None = None,
        _resume_resolution: Any | None = None,
    ) -> ConsoleSubmitResult:
        """Submit a composer draft through native Console validation and provider resolution.

        PR3a-2 Task 5: ``origin=AGENT_WAKE`` (requires a coordinator-issued
        ``wake_authorization``, the queue-token precedent) submits a
        machine-injected auto-wake notice instead of a user draft. The
        wake branch: no USER transcript row is echoed (a SYSTEM-class row
        carrying ``MessageMetadata(origin="agent_wake")`` is appended at
        the acceptance point instead); the composer hook is never invoked
        (non-MANUAL); pending attachments and the one-shot prefill are
        left untouched (they are the USER's staged state); auto-titling,
        RAG capture and prompt history are skipped; and the notice
        reaches the model as a payload-only trailing user-role entry
        appended after every per-send transform (see
        ``console_fleet_wake``'s module docstring for the delivery-path
        decision record). Skill substitution and dictionary/world-info
        transforms still run -- they are HISTORY transforms every send
        re-applies, and the wake's own notice is appended after them, so
        it is never itself substituted.

        F4 fix (Qodo wave, parallel-agents spec §2): sends are dispatched
        per-session -- ``chat_screen._dispatch_console_draft_send`` captures
        the target session at DISPATCH time and threads it through
        ``run_worker``'s coroutine args (see ``_submit_console_native_
        draft``). Before this fix, this method always re-resolved "the
        session to submit into" via ``store.ensure_session()``/
        ``store.active_session_id`` at EXECUTION time instead -- a session
        switch during the scheduling gap between ``run_worker(...)`` and
        this coroutine's body actually running could silently submit the
        draft into whichever session the user switched TO, not the one
        that was showing when Send was pressed.

        Args:
            draft: The raw composer text to submit.
            session_id: The session this draft was dispatched for, captured
                by the caller at dispatch time. ``None`` (the default)
                preserves the pre-fix behavior -- resolve/create the
                CURRENTLY active session -- for direct-call test idioms and
                other callers that have no per-session dispatch to capture.
                An empty string is treated the same as ``None`` (the
                dispatch-time sentinel for "no session existed yet").

        Returns:
            The submission outcome: ``accepted`` False (with an explanatory
            ``visible_copy``) when blocked before any provider call, or
            when ``session_id`` names a session that no longer exists by
            the time this runs (see ``_session_closed_result``); ``True``
            once the turn actually proceeds.
        """
        if not isinstance(origin, ConsoleSubmissionOrigin):
            raise ValueError("origin must be an explicit ConsoleSubmissionOrigin")
        target_id = session_id or self.store.active_session_id or ""
        resumed_preparation = (
            self._preparation_by_id(_resume_preparation_id)
            if _resume_preparation_id is not None
            else None
        )
        prepared_continuation = (
            self._prepared_send_continuations.get(_resume_preparation_id)
            if _resume_preparation_id is not None
            else None
        )
        if _resume_preparation_id is not None and resumed_preparation is None:
            return ConsoleSubmitResult(
                False, False, "Prepared turn is no longer available."
            )
        if _resume_preparation_id is not None and prepared_continuation is None:
            return ConsoleSubmitResult(
                False, False, "Prepared turn is no longer available."
            )
        if origin is ConsoleSubmissionOrigin.QUEUED:
            if not queue_entry_id or (
                _resume_preparation_id is None
                and not self.prompt_queue_coordinator.authorizes(
                    queue_authorization, target_id
                )
            ):
                raise PermissionError(
                    "queued sends require coordinator-issued generation authority"
                )
        elif origin is ConsoleSubmissionOrigin.AGENT_WAKE:
            # PR3a-2 Task 5: only the wake coordinator can mint the token
            # (queue-token precedent) -- no other code path can fabricate
            # a machine-origin send.
            if not self._fleet_wake.authorizes(wake_authorization, target_id):
                raise PermissionError(
                    "agent-wake sends require coordinator-issued wake authority"
                )
            if target_id and self.prompt_queue_coordinator.controls_generation(
                target_id
            ):
                # Defense-in-depth twin of the coordinator's own gate: a
                # queue-owned session's next turn belongs to the queue.
                # Refused WITHOUT a transcript row -- a machine deferral
                # is not user-visible news; the wake retries later.
                return ConsoleSubmitResult(
                    False,
                    False,
                    "Queued messages control the next turn.",
                )
        elif target_id and self.prompt_queue_coordinator.controls_generation(target_id):
            visible_copy = "Queued messages control the next turn. Resume or manage the queue first."
            if target_id and any(
                session.id == target_id for session in self.store.sessions()
            ):
                self.store.append_message(
                    target_id,
                    role=ConsoleMessageRole.SYSTEM,
                    content=visible_copy,
                )
            return ConsoleSubmitResult(False, False, visible_copy)

        active_rejection = self._active_run_rejection(
            session_id=session_id,
            # A raced wake refusal is machine-internal (retried later);
            # only user-facing origins get the explanatory SYSTEM row.
            append_row=origin is not ConsoleSubmissionOrigin.AGENT_WAKE,
            queue_authorization=queue_authorization,
        )
        if active_rejection is not None and resumed_preparation is None:
            return active_rejection

        if (
            target_id
            and resumed_preparation is None
            and self.store.dispatch_recovery_blocks_submission(target_id)
        ):
            return ConsoleSubmitResult(
                False,
                False,
                "Finish or discard the pending response before sending another "
                "message.",
                session_id=target_id,
                origin=origin,
                queue_entry_id=queue_entry_id,
            )

        if session_id:
            session = next(
                (s for s in self.store.sessions() if s.id == session_id), None
            )
            if session is None:
                # The dispatching session was closed during the gap between
                # dispatch and this coroutine actually running -- there is
                # nothing left to submit into. Stamp the (now-orphaned)
                # session id, never whatever is active now (see
                # `_session_closed_result`'s own docstring). `dispatch_gap`
                # is what makes THIS call site (uniquely among ~19) toast --
                # every other one fires mid-run, after the user already
                # confirmed closing that session themselves.
                return self._session_closed_result(
                    session_id=session_id, dispatch_gap=True
                )
        else:
            # Task 4 (D2 fix wave, "bonus race"): mirror the mount-time
            # creator (`ConsoleSessionController._ensure_active_console_session_settings`),
            # which always passes `settings=` -- without this, a session
            # bootstrapped from THIS branch (no dispatch-captured session id
            # at all) got `settings=None` while every other creator gave the
            # first session a real snapshot, and whichever creator ran first
            # decided the outcome.
            session = self.store.ensure_session(
                workspace_id=self.store.workspace_context.active_workspace_id,
                settings=(
                    self._default_session_settings()
                    if self._default_session_settings is not None
                    else None
                ),
            )
        active_task = asyncio.current_task()
        if active_task is not None:
            self._rebind_submit_task(active_task, session.id)
            if resumed_preparation is not None:
                self._bind_submit_preparation(
                    active_task, resumed_preparation.preparation_id
                )
        # PR3a-2 Task 5: a wake never touches the user's staged state --
        # pending attachments belong to the USER's next send and must be
        # neither embedded nor cleared by a machine turn.
        pendings = (
            list(prepared_continuation.attachments)
            if prepared_continuation is not None
            else self.store.pending_attachments(session.id)
            if origin is not ConsoleSubmissionOrigin.AGENT_WAKE
            else []
        )
        attachment_mode_pendings = [
            pending
            for pending in pendings
            if pending.insert_mode == "attachment" and pending.data is not None
        ]
        has_pending_attachment = bool(attachment_mode_pendings)
        if origin is ConsoleSubmissionOrigin.AGENT_WAKE:
            # The notice is machine-composed from DB text and bounded by
            # `compose_wake_notice`'s own result budget; `_validated_draft`
            # exists to validate USER drafts (its length cap and markup
            # rules are composer policy, not payload policy).
            clean_draft = str(draft or "").strip()
            validation_error = None if clean_draft else "Empty wake notice."
        else:
            clean_draft, validation_error = self._validated_draft(
                draft, allow_empty=has_pending_attachment
            )
        if validation_error is not None:
            return self._block(session.id, validation_error)
        configuration = (
            resumed_preparation.execution_context.configuration
            if resumed_preparation is not None
            else self.resolve_turn_configuration_snapshot(session.id)
        )
        turn_selection = configuration.provider_selection
        if has_pending_attachment:
            vision_model = configuration.effective_model
            # ONE capability check decides the gate AND the copy: this
            # module's is_vision_capable (the documented monkeypatch seam) is
            # injected into vision_block_reason instead of being re-checked
            # around it — the two seams could otherwise disagree under test.
            block_reason = vision_block_reason(
                turn_selection.provider,
                vision_model,
                is_capable=lambda _provider, _model: bool(
                    configuration.capabilities.get("vision", False)
                ),
            )
            if block_reason is not None:
                return self._block(session.id, block_reason)
        if turn_selection.workspace_context.has_policy_blocks:
            return self._block(
                session.id, turn_selection.workspace_context.recovery_copy
            )
        library_authority = (
            resumed_preparation.execution_context.library_authority
            if resumed_preparation is not None
            else await self._capture_turn_library_authority(session.id, configuration)
        )
        existing_preparation = self.store.preparation_for_session(session.id)
        if (
            existing_preparation is not None
            and existing_preparation is not resumed_preparation
            and existing_preparation.state
            not in {
                ConsoleTurnPreparationState.CANCELLED,
                ConsoleTurnPreparationState.SETTLED,
            }
        ):
            return ConsoleSubmitResult(
                False,
                False,
                "Another send is still preparing for this conversation.",
            )
        pre_send_title = (
            resumed_preparation.pre_send_title
            if resumed_preparation is not None
            else session.title
        )
        pre_send_conversation_id = (
            resumed_preparation.pre_send_conversation_id
            if resumed_preparation is not None
            else session.persisted_conversation_id
        )
        explicit_evidence_staged = self._has_explicit_staged_evidence(session.id)

        # TASK-457(a): echo the USER message BEFORE resolving the provider, so a
        # slow/cold readiness probe no longer leaves the transcript blank while
        # the composer clears — the message reads as "sent", not lost. On a
        # not-ready provider the row persists next to the honest block-row below
        # (the message is no longer silently dropped) and the draft is kept (the
        # composer clears only on the accepted path via
        # `_notify_submission_accepted`), so the user can re-attempt. Staged
        # attachments are embedded on the row here but only CLEARED on the
        # success path below, so a blocked attempt leaves them staged for retry.
        #
        # Auto-title BEFORE the append: a persisting append creates the durable
        # conversation from `session.title` (persist_session_if_needed) and sets
        # `persisted_conversation_id`, after which `_maybe_auto_title_session`
        # early-returns. Titling first means the conversation is created as the
        # derived title (e.g. "hello") instead of the default "Chat 1", so the
        # workspace rail shows it immediately after persistence.
        durable_commit = getattr(self.store.persistence, "commit_durable_turn", None)
        durable_turn = bool(
            not session.ephemeral
            and origin
            in {ConsoleSubmissionOrigin.MANUAL, ConsoleSubmissionOrigin.QUEUED}
        )
        if durable_turn and not callable(durable_commit):
            # TASK-22030: a refusal the user cannot see is indistinguishable
            # from a broken app. `_block_undurable_turn` writes the run state,
            # the transcript row, and the toast that `56db75386` dropped.
            return self._block_undurable_turn(
                session.id,
                origin=origin,
                queue_entry_id=queue_entry_id,
            )
        staged_title = session.title
        if (
            origin is not ConsoleSubmissionOrigin.AGENT_WAKE
            and resumed_preparation is None
        ):
            derived_title = (
                derive_console_session_title(clean_draft)
                if session.persisted_conversation_id is None
                and is_default_console_session_title(session.title)
                else ""
            )
            if durable_turn:
                staged_title = derived_title or session.title
            else:
                self._maybe_auto_title_session(session, clean_draft)
        staged_attachments = tuple(
            MessageAttachment(
                data=pending.data,
                mime_type=pending.mime_type or "image/png",
                display_name=pending.display_name,
                position=index,
            )
            for index, pending in enumerate(attachment_mode_pendings)
        )
        # TASK-485: the optimistic echo is appended WITHOUT persistence. A send
        # that is blocked/fails before it reaches the provider must leave no
        # durable record — otherwise the resume path (which reconstructs every
        # row as "complete") would silently drop the row's failed state and let a
        # never-sent message re-enter the next send's context, and the orphan
        # would render as a lonely user prompt. The row is flushed to storage
        # only once the turn is confirmed to proceed (below).
        #
        # PR3a-2 Task 5: a wake echoes NOTHING here -- invariant 5 forbids
        # a USER row for machine input, and the SYSTEM notice row is
        # appended only at the acceptance point below (TASK-457(a)'s
        # "reads as sent, not lost" concern protects a HUMAN's typed
        # message during a slow readiness probe; a machine notice has no
        # one watching for it, and appending late means a blocked wake
        # leaves no orphaned notice row to clean up).
        echoed_user = (
            self.store.get_message(resumed_preparation.transient_user_message_id)
            if resumed_preparation is not None
            and resumed_preparation.transient_user_message_id is not None
            else self.store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content=clean_draft,
                attachments=staged_attachments,
                persist=False,
            )
            if origin is not ConsoleSubmissionOrigin.AGENT_WAKE
            else None
        )

        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."),
            session_id=session.id,
        )
        try:
            resolution = (
                _resume_resolution
                if resumed_preparation is not None
                else await self._resolve_for_send_bounded(turn_selection)
            )
        except BaseException:
            # A readiness probe that raises or is cancelled AFTER the optimistic
            # USER echo must still fail that row — otherwise a never-sent USER
            # message leaks into the NEXT send's provider context (`skip_failed`
            # only drops "failed" rows). Fail it, then re-raise so the caller
            # still sees the probe failure. (A wake echoed nothing: None guard.)
            if echoed_user is not None:
                if self._shutdown_requested.is_set():
                    self.store.delete_message(echoed_user.id)
                    session.title = pre_send_title
                    session.persisted_conversation_id = pre_send_conversation_id
                else:
                    self._mark_transient_echo_blocked(echoed_user.id)
            raise
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
            # The echoed row stays visible but never reached a provider — fail it
            # so it is excluded from the NEXT send's provider context
            # (`skip_failed`) and reads honestly as unsent rather than polluting
            # the history. (A wake echoed nothing: None guard.)
            if echoed_user is not None:
                self._mark_transient_echo_blocked(echoed_user.id)
            return self._block(session.id, visible_copy)

        if resumed_preparation is not None:
            turn_context = resumed_preparation.execution_context
        else:
            try:
                turn_context = self._finalize_turn_execution_context(
                    configuration,
                    library_authority,
                    resolution,
                )
            except (TypeError, ValueError):
                if echoed_user is not None:
                    self._mark_transient_echo_blocked(echoed_user.id)
                    self.store.delete_message(echoed_user.id)
                return self._block(session.id, "Provider destination is incomplete.")

        preparation: ConsoleTurnPreparation | None = resumed_preparation
        preparation_outcome: ConsolePreparationOutcome | None = (
            self._preparation_outcomes.get(resumed_preparation.preparation_id)
            if resumed_preparation is not None
            else None
        )
        ordinary_library_text = self._ordinary_library_text(
            clean_draft,
            origin,
            has_pending_attachment=has_pending_attachment,
        )
        if ordinary_library_text and resumed_preparation is None:
            automatic_eligible = (
                library_authority.policy.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
                and explicit_evidence_staged is False
            )
            initial_state = (
                initial_preparation_state(library_authority.policy.auto_retrieve)
                if automatic_eligible
                or library_authority.policy.auto_retrieve is ConsoleAutoRetrieve.NEVER
                else ConsoleTurnPreparationState.READY
            )
            queue_generation = None
            if origin is ConsoleSubmissionOrigin.QUEUED:
                queue_generation = self.prompt_queue_registry.snapshot(
                    session.id
                ).revision
            one_shot_prefill, captured_prefill_revision = (
                self.store.session_one_shot_prefill_snapshot(session.id)
            )
            frozen_prefill, frozen_prefill_from_one_shot = self._resolve_submit_prefill(
                session.id
            )
            (
                staged_evidence_frozen,
                staged_evidence,
                staged_evidence_release,
            ) = self._snapshot_staged_evidence()
            preparation = ConsoleTurnPreparation(
                preparation_id=str(uuid4()),
                attempt_id=library_authority.attempt_id,
                session_id=session.id,
                origin=origin.value,
                queue_entry_id=queue_entry_id,
                executed_draft=clean_draft,
                execution_context=turn_context,
                transient_user_message_id=(
                    echoed_user.id if echoed_user is not None else None
                ),
                attachment_ids=tuple(pending.attachment_id for pending in pendings),
                evidence_ids=(
                    ("explicit-staged-evidence",) if explicit_evidence_staged else ()
                ),
                prefill_id=(
                    "prefill-"
                    + hashlib.sha256(one_shot_prefill.encode("utf-8")).hexdigest()[:24]
                    if one_shot_prefill is not None
                    else None
                ),
                queue_generation=queue_generation,
                pre_send_title=pre_send_title,
                pre_send_conversation_id=pre_send_conversation_id,
                state=initial_state,
                pause_kind=None,
                one_shot_bypass=False,
                ephemeral=session.ephemeral,
            )
            if self._begin_submit_preparation(active_task, preparation) is None:
                if echoed_user is not None:
                    self._mark_transient_echo_blocked(echoed_user.id)
                return ConsoleSubmitResult(
                    False,
                    False,
                    "Another send is still preparing for this conversation.",
                )
            if origin is ConsoleSubmissionOrigin.QUEUED and (
                queue_entry_id is None
                or not self.prompt_queue_coordinator.bind_claimed_preparation(
                    session.id,
                    entry_id=queue_entry_id,
                    preparation_id=preparation.preparation_id,
                )
            ):
                self._abandon_preparation(preparation.preparation_id)
                if echoed_user is not None:
                    self._mark_transient_echo_blocked(echoed_user.id)
                return ConsoleSubmitResult(
                    False,
                    False,
                    "Queued preparation could not bind its exact entry.",
                    session_id=session.id,
                    origin=origin,
                    queue_entry_id=queue_entry_id,
                )
            self._prepared_send_continuations[preparation.preparation_id] = (
                _PreparedSendContinuation(
                    preparation_id=preparation.preparation_id,
                    attachments=tuple(pendings),
                    prefill=frozen_prefill,
                    prefill_from_one_shot=frozen_prefill_from_one_shot,
                    one_shot_prefill_revision=(
                        captured_prefill_revision
                        if frozen_prefill_from_one_shot
                        else None
                    ),
                    staged_evidence_frozen=staged_evidence_frozen,
                    staged_evidence=(
                        _PreparedEvidenceLease(
                            staged_evidence,
                            release=staged_evidence_release,
                        )
                        if staged_evidence is not None
                        else None
                    ),
                )
            )
            prepared_continuation = self._prepared_send_continuations[
                preparation.preparation_id
            ]
            if preparation.state is ConsoleTurnPreparationState.PREPARING:
                preparation_outcome = await self.prepare_library_for_turn(
                    preparation.preparation_id
                )
                if preparation_outcome.state is not ConsoleTurnPreparationState.READY:
                    self._set_run_state(
                        ConsoleRunState.blocked(
                            "Library preparation paused before provider dispatch."
                        ),
                        session_id=session.id,
                    )
                    return ConsoleSubmitResult(
                        False,
                        False,
                        "Library preparation paused before provider dispatch.",
                        session_id=session.id,
                        origin=origin,
                        queue_entry_id=queue_entry_id,
                    )

        citation_context: str | None = None
        citation_trace_builder: CitationTraceBuilder | None = None
        prompt_evidence_set_id: str | None = None
        citation_repair_contract: CitationRepairContract | None = None
        terminal_citation_finalizer: TerminalCitationFinalizer | None = None
        try:
            provider_messages = self._provider_messages_for_session(
                session.id, annotate_ids=True, turn_context=turn_context
            )
            (
                provider_messages,
                refuse,
                skill_notes,
                skill_bindings,
                skill_bundle_block,
            ) = await self._apply_skill_substitution(provider_messages)
            if refuse is not None:
                # A substitution refusal is a block outcome like any other
                # (provider not ready, probe raise): fail the echoed row so the
                # refused command never enters the next send's provider context.
                # (A wake echoed nothing: None guard.)
                if echoed_user is not None:
                    self._mark_transient_echo_blocked(echoed_user.id)
                if preparation is not None:
                    self._abandon_preparation(preparation.preparation_id)
                return self._block(session.id, refuse)
            for note in skill_notes:
                # An embedded skipped-skill note is never an abort: append the
                # same system-row copy `_block` would, then let the turn proceed.
                self.store.append_message(
                    session.id, role=ConsoleMessageRole.SYSTEM, content=note
                )
            if (
                preparation_outcome is not None
                and preparation_outcome.evidence_bundle is not None
            ):
                citation_context = format_evidence_for_cited_answer(
                    preparation_outcome.evidence_bundle
                )
            elif (
                origin is not ConsoleSubmissionOrigin.AGENT_WAKE
                and prepared_continuation is not None
                and prepared_continuation.staged_evidence_frozen
            ):
                (
                    citation_context,
                    citation_trace_builder,
                    prompt_evidence_set_id,
                    citation_repair_contract,
                ) = await self._capture_frozen_rag_context(
                    clean_draft,
                    turn_context,
                    prepared_continuation,
                )
            elif origin is not ConsoleSubmissionOrigin.AGENT_WAKE:
                # PR3a-2 Task 5: a wake notice is a delivery, not a query
                # -- retrieving evidence "about" a machine notice would
                # inject RAG context the user never asked for. The
                # pre-initialized Nones above stand.
                (
                    citation_context,
                    citation_trace_builder,
                    prompt_evidence_set_id,
                    citation_repair_contract,
                ) = await self._capture_rag_context(
                    clean_draft,
                    turn_context=turn_context,
                    origin=origin,
                )
            has_exact_citation_context = (
                citation_trace_builder is not None
                or citation_repair_contract is not None
            )
            if citation_context and not has_exact_citation_context:
                provider_messages = self._prepend_evidence_context(
                    provider_messages,
                    citation_context,
                )
            provider_messages = await self._apply_chat_dictionaries(
                provider_messages, session.id
            )
            provider_messages = await self._apply_world_info(
                provider_messages, session.id
            )
            if citation_context and has_exact_citation_context:
                provider_messages = self._prepend_evidence_context(
                    provider_messages,
                    citation_context,
                )
            if citation_context and echoed_user is not None:
                trace_prefix = f"console-trace:{echoed_user.id}:retrieval"
                retrieval_event_id = f"{trace_prefix}:retrieval_completed"
                attached_event_id = f"{trace_prefix}:context_attached"
                self.store.record_trace_event(
                    session.id,
                    anchor_message_id=echoed_user.id,
                    event_kind="context_attached",
                    summary="Retrieved context attached",
                    status="completed",
                    event_id=attached_event_id,
                    parent_event_id=retrieval_event_id,
                    source_event_id=retrieval_event_id,
                    sensitivity="system_context",
                )
                self.store.record_trace_event(
                    session.id,
                    anchor_message_id=echoed_user.id,
                    event_kind="context_injected",
                    summary="Retrieved context injected into provider request",
                    status="completed",
                    event_id=f"{trace_prefix}:context_injected",
                    parent_event_id=attached_event_id,
                    source_event_id=attached_event_id,
                    sensitivity="system_context",
                )
            if origin is ConsoleSubmissionOrigin.AGENT_WAKE:
                # The one-shot prefill is USER-staged state; a wake must
                # not consume (and thereby destroy) it.
                prefill, prefill_from_one_shot, one_shot_prefill_revision = (
                    None,
                    False,
                    None,
                )
            elif prepared_continuation is not None:
                prefill = prepared_continuation.prefill
                prefill_from_one_shot = prepared_continuation.prefill_from_one_shot
                one_shot_prefill_revision = (
                    prepared_continuation.one_shot_prefill_revision
                )
            else:
                prefill, prefill_from_one_shot = self._resolve_submit_prefill(
                    session.id
                )
                one_shot_prefill_revision = (
                    self.store.session_one_shot_prefill_snapshot(session.id)[1]
                    if prefill_from_one_shot
                    else None
                )
            terminal_citation_finalizer = self._build_terminal_citation_finalizer(
                context=citation_context,
                builder=citation_trace_builder,
                prompt_evidence_set_id=prompt_evidence_set_id,
            )
        except BaseException:
            # Any failure between the optimistic echo and the confirmed turn
            # (dictionary/world-info application, prefill resolution) must also
            # fail the echoed row, or a never-sent message leaks into the next
            # send's provider context (`skip_failed` only drops "failed" rows).
            # (A wake echoed nothing: None guard.)
            if echoed_user is not None:
                self._mark_transient_echo_blocked(echoed_user.id)
            if preparation is not None:
                self._abandon_preparation(preparation.preparation_id)
            raise
        # The accepted-hook fires only once the turn is confirmed to
        # actually proceed (Qodo finding 3, PR #636 bot review): it used to
        # fire right after the USER row was appended, BEFORE this skill
        # substitution/trust check ran. In the real ChatScreen this hook
        # clears the composer, so firing it before a substitution refusal
        # ate the refused draft the user needs to correct. A substitution
        # refusal is a `_block()` outcome exactly like any other (provider
        # not ready, policy block, validation failure) and those already
        # never reach this hook -- this ordering just extends that same
        # rule to cover it too.
        if origin is ConsoleSubmissionOrigin.QUEUED and not (
            self.prompt_queue_coordinator.authorizes(queue_authorization, session.id)
        ):
            # Close/shutdown can tombstone the chain while this claimed turn
            # awaits readiness/substitution/RAG. Revalidate immediately before
            # acceptance so cancellation cannot turn that stale claim into a
            # durable user message or provider dispatch. (A wake echoed
            # nothing: None guard.)
            if echoed_user is not None:
                self._mark_transient_echo_blocked(echoed_user.id)
            if preparation is not None:
                self._abandon_preparation(preparation.preparation_id)
            return ConsoleSubmitResult(
                False,
                False,
                "Queued turn canceled before it could start.",
            )
        # PR3a-2 Task 5: the wake notice enters the MODEL PAYLOAD here, as
        # a payload-only trailing user-role entry -- appended AFTER every
        # per-send transform (substitution/dictionaries/world-info ran on
        # the history above and must never rewrite the notice) and never
        # written to the store (the transcript's record is the SYSTEM
        # machine-origin row at the acceptance point below). Trailing
        # user-role is deliberate: SYSTEM transcript rows are dropped from
        # payloads by design, and a payload ending on an assistant row is
        # a prefill to strict providers -- see console_fleet_wake's
        # delivery-path decision record for why neither turn_bundle_block
        # nor the system fold can carry this.
        if origin is ConsoleSubmissionOrigin.AGENT_WAKE:
            provider_messages = [
                *provider_messages,
                {
                    "role": ConsoleMessageRole.USER.value,
                    "content": clean_draft,
                },
            ]
        if preparation is not None:
            current_preparation = self._preparation_by_id(preparation.preparation_id)
            if current_preparation is None or (
                current_preparation.state is not ConsoleTurnPreparationState.COMMITTING
                and not self._transition_preparation(
                    preparation.preparation_id,
                    ConsoleTurnPreparationState.READY,
                    ConsoleTurnPreparationState.COMMITTING,
                )
            ):
                return ConsoleSubmitResult(
                    False,
                    False,
                    "Prepared turn changed before provider dispatch.",
                    session_id=session.id,
                    origin=origin,
                    queue_entry_id=queue_entry_id,
                )
        committed_context_epoch = self.store.conversation_context_epoch(session.id)
        if durable_turn and preparation is not None and echoed_user is not None:
            return await self._accept_durable_turn(
                session=session,
                preparation=preparation,
                preparation_outcome=preparation_outcome,
                prepared_continuation=prepared_continuation,
                echoed_user=echoed_user,
                staged_title=staged_title,
                staged_attachments=staged_attachments,
                resolution=resolution,
                provider_messages=provider_messages,
                prefill=prefill,
                prefill_from_one_shot=prefill_from_one_shot,
                one_shot_prefill_revision=one_shot_prefill_revision,
                skill_bindings=tuple(skill_bindings),
                skill_bundle_block=skill_bundle_block,
                citation_repair_contract=citation_repair_contract,
                terminal_citation_finalizer=terminal_citation_finalizer,
                turn_context=turn_context,
                origin=origin,
                queue_entry_id=queue_entry_id,
                committed_context_epoch=committed_context_epoch,
            )
        # TASK-1364: record the accepted send to the shared prompt history.
        # Same placement rule as the accepted-hook above: only a send that is
        # confirmed to proceed is recorded -- every `_block`/refusal path
        # returns before this point, and `_record_prompt_history` itself
        # skips empty (attachment-only) drafts. A wake notice is not a
        # prompt the user typed and never enters their prompt history.
        if origin is not ConsoleSubmissionOrigin.AGENT_WAKE:
            try:
                await self._record_prompt_history(clean_draft)
            except BaseException:
                if preparation is not None:
                    self._rollback_committing_preparation(preparation.preparation_id)
                raise
        if self._disposed or (
            self._shutdown_requested.is_set()
            and origin is not ConsoleSubmissionOrigin.AGENT_WAKE
        ):
            if echoed_user is not None:
                try:
                    self._mark_transient_echo_blocked(echoed_user.id)
                except KeyError:
                    pass
            if preparation is not None:
                self._rollback_committing_preparation(preparation.preparation_id)
            return ConsoleSubmitResult(
                False,
                False,
                "Console shut down before turn acceptance.",
                session_id=session.id,
                origin=origin,
                queue_entry_id=queue_entry_id,
            )
        # TASK-485: the turn is confirmed to proceed — flush the deferred USER
        # echo to durable storage now (creating the conversation), BEFORE the
        # assistant row, so a reload shows the user's prompt ahead of its reply.
        #
        # PR3a-2 Task 5, the wake half: the SYSTEM-class notice row is
        # appended HERE, only once the turn is confirmed -- so a blocked
        # wake leaves no orphaned notice -- ahead of the assistant row,
        # persisted, and carrying the machine-origin metadata that marks
        # it as not-user-input for every machine consumer.
        if origin is ConsoleSubmissionOrigin.AGENT_WAKE:
            echoed_user = self.store.append_message(
                session.id,
                role=ConsoleMessageRole.SYSTEM,
                content=clean_draft,
                persist=self.store.persistence is not None,
                metadata=MessageMetadata(origin=MESSAGE_ORIGIN_AGENT_WAKE),
            )
        else:
            try:
                self.store.persist_message_if_needed(echoed_user.id)
            except BaseException:
                if preparation is not None:
                    self._rollback_committing_preparation(preparation.preparation_id)
                raise
        assistant: ConsoleChatMessage | None = None
        citation_repair_session = (
            ConsoleCitationRepairSession(
                contract=citation_repair_contract,
                resolution=resolution,
            )
            if citation_repair_contract is not None
            else None
        )
        # task-15860: a wake turn in flight is exempt from `leave_console()`
        # (owner ruling -- see that method). Registered here, released in
        # the `finally` below, so the exemption cannot outlive the turn.
        if origin is ConsoleSubmissionOrigin.AGENT_WAKE:
            self._agent_wake_turn_sessions.add(session.id)
        try:
            assistant = self.store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="",
                persist=self.store.persistence is not None,
                terminal_citation_finalizer=terminal_citation_finalizer,
                defer_terminal_persistence=citation_repair_session is not None,
            )
            if (
                session.ephemeral
                and origin
                in {ConsoleSubmissionOrigin.MANUAL, ConsoleSubmissionOrigin.QUEUED}
                and preparation is not None
            ):
                self.store.register_ephemeral_dispatch_recovery(
                    session.id,
                    user_message_id=echoed_user.id,
                    assistant_message_id=assistant.id,
                    preparation_id=preparation.preparation_id,
                    attempt_id=turn_context.library_authority.attempt_id,
                    checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
                    origin=origin.value,
                    queue_entry_id=queue_entry_id,
                    frozen_authority=turn_context.library_authority,
                    resolved_destination=turn_context.resolved_destination,
                    reconstructability=ConsoleDispatchReconstructability(
                        attachments_reconstructable=True,
                        evidence_reconstructable=not bool(
                            prepared_continuation is not None
                            and (
                                prepared_continuation.staged_evidence_frozen
                                or prepared_continuation.staged_evidence is not None
                            )
                        ),
                        prefill_reconstructable=(
                            prefill is None and not prefill_from_one_shot
                        ),
                        opaque_reference=(f"opaque:{preparation.preparation_id}"),
                    ),
                    runtime_active=True,
                )
            if preparation is not None and not self._transition_preparation(
                preparation.preparation_id,
                ConsoleTurnPreparationState.COMMITTING,
                ConsoleTurnPreparationState.ACCEPTED,
            ):
                raise RuntimeError("Prepared turn changed before acceptance.")
            stream_signals = self._admit_capture_policy(session.id, origin)
            self._release_prepared_evidence(prepared_continuation)
            for pending in pendings:
                self.store.consume_pending_attachment(session.id, pending.attachment_id)
            self._notify_submission_accepted(
                session_id=session.id,
                origin=origin,
                entry_id=queue_entry_id,
                context_epoch=committed_context_epoch,
                defer_queued_settlement=(
                    resumed_preparation is not None
                    and origin is ConsoleSubmissionOrigin.QUEUED
                ),
            )
            if (
                session.ephemeral
                and self.store.dispatch_recovery_for_session(session.id) is not None
            ):
                if (
                    self.store.begin_ephemeral_dispatch(
                        session.id,
                        assistant_message_id=assistant.id,
                        new_attempt_id=turn_context.library_authority.attempt_id,
                    )
                    is None
                ):
                    raise RuntimeError(
                        "Ephemeral dispatch checkpoint changed before provider entry."
                    )
            stream_result = await self._stream_assistant_response(
                resolution=resolution,
                provider_messages=provider_messages,
                assistant_message_id=assistant.id,
                prefill=prefill,
                prefill_from_one_shot=prefill_from_one_shot,
                one_shot_prefill_revision=one_shot_prefill_revision,
                skill_bindings=skill_bindings,
                skill_bundle_block=skill_bundle_block,
                citation_repair_session=citation_repair_session,
                turn_context=turn_context,
                preparation_id=(
                    preparation.preparation_id if preparation is not None else None
                ),
                stream_signals=stream_signals,
            )
            result = replace(
                stream_result,
                session_id=session.id,
                user_message_id=echoed_user.id,
                assistant_message_id=assistant.id,
                terminal_status=self.run_state_for(session.id).status,
                origin=origin,
                queue_entry_id=queue_entry_id,
                committed_context_epoch=committed_context_epoch,
            )
            if preparation is not None:
                self._settle_accepted_preparation(preparation.preparation_id)
            return result
        except BaseException as exc:
            if isinstance(exc, ConsoleDispatchSettlementError):
                if assistant is not None:
                    self.store.release_dispatch_recovery_action(
                        session.id,
                        assistant.id,
                    )
                raise
            accepted_cancellation = isinstance(exc, asyncio.CancelledError) and (
                assistant is not None and echoed_user is not None
            )
            if assistant is not None:
                try:
                    self.store.mark_message_failed(assistant.id)
                    self._set_run_state(
                        ConsoleRunState(
                            ConsoleRunStatus.FAILED,
                            "Accepted turn failed before provider dispatch.",
                        ),
                        session_id=session.id,
                    )
                except KeyError:
                    pass
            if preparation is not None:
                current = self._preparation_by_id(preparation.preparation_id)
                if (
                    current is not None
                    and current.state is ConsoleTurnPreparationState.COMMITTING
                ):
                    self._rollback_committing_preparation(preparation.preparation_id)
                elif current is not None and current.state in {
                    ConsoleTurnPreparationState.ACCEPTED,
                    ConsoleTurnPreparationState.DISPATCH_STARTED,
                    ConsoleTurnPreparationState.DISPATCHED,
                }:
                    self._settle_accepted_preparation(preparation.preparation_id)
            if accepted_cancellation:
                terminal_state = self.run_state_for(session.id)
                return ConsoleSubmitResult(
                    True,
                    True,
                    terminal_state.visible_copy
                    or "Accepted turn failed before provider dispatch.",
                    session_id=session.id,
                    user_message_id=echoed_user.id,
                    assistant_message_id=assistant.id,
                    terminal_status=terminal_state.status,
                    origin=origin,
                    queue_entry_id=queue_entry_id,
                    committed_context_epoch=committed_context_epoch,
                )
            raise
        finally:
            if origin is ConsoleSubmissionOrigin.AGENT_WAKE:
                self._agent_wake_turn_sessions.discard(session.id)
            if assistant is not None:
                self.store.clear_terminal_citation_state(assistant.id)
            del terminal_citation_finalizer
            del citation_trace_builder

    async def _run_durable_postcommit_effect(
        self,
        preparation_id: str,
        effect_name: str,
        callback: Callable[[], Any],
        *,
        fingerprint: ConsoleDurableAcceptanceFingerprint,
    ) -> Any:
        """Run one preparation-keyed effect and mark it only after success."""

        effects = self.store.durable_postcommit_effects_for(
            preparation_id, fingerprint=fingerprint
        )
        if effects is None:
            raise RuntimeError("Durable postcommit effects are unavailable.")
        if effect_name in effects.completed:
            return None
        if not self.store.claim_durable_postcommit_effect(
            preparation_id, effect_name, fingerprint=fingerprint
        ):
            raise RuntimeError("Durable postcommit effect is already in flight.")
        try:
            result = callback()
            if inspect.isawaitable(result):
                result = await result
        except BaseException:
            # TASK-22587: releasing the claim must never REPLACE the failure
            # that sent us here. Bookkeeping is strictly less informative than
            # the original exception, and this arm also runs for CancelledError.
            try:
                self.store.abandon_durable_postcommit_effect(
                    preparation_id, effect_name, fingerprint=fingerprint
                )
            except Exception as release_exc:
                logger.warning(
                    "Durable postcommit effect release failed; keeping the "
                    "original failure (effect={}, release_exception_type={})",
                    effect_name,
                    type(release_exc).__name__,
                )
            raise
        try:
            self.store.complete_durable_postcommit_effect(
                preparation_id, effect_name, fingerprint=fingerprint
            )
        except ConsoleDurableAcceptanceRetired:
            # TASK-22587: the session was closed while this effect ran. The
            # work itself succeeded; there is simply no ledger left to record
            # it in. Closing a chat is not an error.
            logger.debug(
                "Durable postcommit effect completed after retirement "
                "(effect={})",
                effect_name,
            )
        return result

    def _durable_db_call_offloadable(self) -> bool:
        """True when a durable persistence call may run on a worker thread.

        TASK-22205: the per-send ``BEGIN IMMEDIATE`` durable-turn commit and
        the pre-dispatch checkpoint CAS used to run synchronously on the
        event loop — tens of ms steady-state, up to the 15 s busy timeout
        under write-lock contention (e.g. the messages_fts backfill window).
        Follows the ``_is_memory_backed`` precedent in
        ``chat_conversation_scope_service``: thread-local file-backed sqlite
        connections are safe on ``asyncio.to_thread``, but a ``:memory:``
        CharactersRAGDB is per-connection — a worker thread would see an
        empty, unmigrated database — so memory-backed persistence stays
        inline. A persistence fake with no ``.db`` threads harmlessly.
        """

        db = getattr(self.store.persistence, "db", None)
        return not bool(getattr(db, "is_memory_db", False))

    async def _run_durable_db_call(
        self, call: Callable[..., Any], /, *args: Any
    ) -> Any:
        """Run one durable DB transaction off the event loop when safe.

        The ``await`` is the ordering barrier: the coroutine resumes only
        after the transaction durably exists (or raised), so provider
        dispatch can never precede the commit, and every state transition
        after the call still sees its result. Per-session serialization is
        upstream: ``begin_preparation`` holds a single live slot per
        session, and the prompt-queue dispatcher refuses/queues while a
        turn is preparing or accepted. Exceptions cross ``to_thread``
        unchanged, so the off-loop failure path is byte-identical to the
        inline one. Task cancellation during the await leaves the thread
        running to completion (``to_thread`` survives cancellation): the
        transaction still commits or rolls back atomically on the worker
        thread, which is exactly the crash-window state the restore
        reconcile already recovers.
        """

        if self._durable_db_call_offloadable():
            return await asyncio.to_thread(call, *args)
        return call(*args)

    async def _accept_durable_turn(
        self,
        *,
        session: ConsoleChatSession,
        preparation: ConsoleTurnPreparation,
        preparation_outcome: ConsolePreparationOutcome | None,
        prepared_continuation: _PreparedSendContinuation | None,
        echoed_user: ConsoleChatMessage,
        staged_title: str,
        staged_attachments: tuple[MessageAttachment, ...],
        resolution: ConsoleProviderResolution,
        provider_messages: list[dict[str, Any]],
        prefill: str | None,
        prefill_from_one_shot: bool,
        one_shot_prefill_revision: int | None,
        skill_bindings: tuple[Any, ...],
        skill_bundle_block: str,
        citation_repair_contract: CitationRepairContract | None,
        terminal_citation_finalizer: TerminalCitationFinalizer | None,
        turn_context: ConsoleTurnExecutionContext,
        origin: ConsoleSubmissionOrigin,
        queue_entry_id: str | None,
        committed_context_epoch: int,
    ) -> ConsoleSubmitResult:
        """Commit one durable owner, then enter its idempotent effect chain."""

        staged_identity = self.store.staged_durable_turn_identity_for(
            preparation.preparation_id
        )
        identity = self.store.stage_durable_turn_identity(
            session.id,
            preparation.preparation_id,
            title=staged_identity.title
            if staged_identity is not None
            else staged_title,
        )
        owner_ids = self.store.stage_durable_turn_owner_ids(
            session.id,
            preparation.preparation_id,
            user_message_id=echoed_user.id,
        )
        # This used to read `echoed_user.parent_message_id`, which is the
        # PERSISTED parent id that `_persist_new_message` assigns -- and this
        # echo was appended with `persist=False`, so it was ALWAYS None. Every
        # checkpointed turn was therefore written as a fresh DB root, forking
        # the conversation away from its own history on the second send
        # (TASK-22060).
        #
        # Resolve the nearest PERSISTED ancestor from the store's own tree
        # bookkeeping instead -- the identical walk `_persist_new_message`
        # performs, so the two persistence paths agree by construction rather
        # than by coincidence. `None` is not an error here: it is the
        # documented "true persisted root" answer, which is exactly what the
        # store does with it, and `insert_with_messages` validates a parent
        # only when one is given.
        parent_message_id = self.store.durable_parent_for_message(echoed_user.id)
        contributions = (
            (preparation_outcome.contribution,)
            if preparation_outcome is not None
            and preparation_outcome.contribution is not None
            else ()
        )
        acceptance = ConsoleDurableTurnAcceptance(
            conversation_id=identity.conversation_id,
            user_message_id=owner_ids.user_message_id,
            assistant_message_id=owner_ids.assistant_message_id,
            parent_message_id=parent_message_id,
            user_content=preparation.executed_draft,
            attachments=tuple(
                {
                    "position": attachment.position,
                    "data": attachment.data,
                    "mime_type": attachment.mime_type,
                    "display_name": attachment.display_name,
                }
                for attachment in staged_attachments
            ),
            preparation_id=preparation.preparation_id,
            attempt_id=preparation.attempt_id,
            origin=origin.value,
            queue_entry_id=queue_entry_id,
            frozen_authority=turn_context.library_authority,
            resolved_destination=turn_context.resolved_destination,
            reconstructability=ConsoleDispatchReconstructability(
                attachments_reconstructable=True,
                evidence_reconstructable=not bool(
                    prepared_continuation is not None
                    and (
                        prepared_continuation.staged_evidence_frozen
                        or prepared_continuation.staged_evidence is not None
                    )
                ),
                prefill_reconstructable=(prefill is None and not prefill_from_one_shot),
                opaque_reference=f"opaque:{preparation.preparation_id}",
            ),
            contributions=contributions,
        )
        try:
            # TASK-22205: the ~10-statement BEGIN IMMEDIATE turn commit runs
            # off the event loop; the await is the dispatch-ordering barrier.
            commit = await self._run_durable_db_call(
                self.store.commit_durable_turn, acceptance
            )
        except Exception as exc:  # noqa: BLE001 -- a failed commit is a retry, not a crash
            # TASK-22251: the user-facing copy stays deliberately generic, but
            # something must record WHICH failure occurred. `commit_durable_turn`
            # is a multi-step transaction -- conversation create, Library-policy
            # write, workspace validation, checkpoint insert -- and swallowing
            # the exception collapsed every one of them into a single sentence.
            # Two distinct causes ("Workspace registry is required for workspace
            # conversations" and "Unknown workspace: <id>") were previously
            # indistinguishable, and each needed a temporary print inside this
            # method to identify. Type only, never the message: an exception
            # string here can carry conversation or workspace identifiers.
            logger.warning(
                "Durable turn commit failed; turn refused (exception_type={})",
                type(exc).__name__,
            )
            return ConsoleSubmitResult(
                False,
                False,
                "Couldn't save the prepared turn. Retry or cancel.",
                session_id=session.id,
                user_message_id=echoed_user.id,
                origin=origin,
                queue_entry_id=queue_entry_id,
                preparation_id=preparation.preparation_id,
            )
        fingerprint = self.store.durable_acceptance_fingerprint_for(
            preparation.preparation_id
        )
        if fingerprint is None:
            raise RuntimeError("Durable acceptance fingerprint is unavailable.")
        citation_repair_session = (
            ConsoleCitationRepairSession(
                contract=citation_repair_contract,
                resolution=resolution,
            )
            if citation_repair_contract is not None
            else None
        )
        continuation = _DurablePostcommitContinuation(
            preparation_id=preparation.preparation_id,
            fingerprint=fingerprint,
            session_id=session.id,
            origin=origin,
            queue_entry_id=queue_entry_id,
            clean_draft=preparation.executed_draft,
            commit=commit,
            echoed_user_id=echoed_user.id,
            resolution=resolution,
            provider_messages=provider_messages,
            prefill=prefill,
            prefill_from_one_shot=prefill_from_one_shot,
            one_shot_prefill_revision=one_shot_prefill_revision,
            skill_bindings=skill_bindings,
            skill_bundle_block=skill_bundle_block,
            citation_repair_session=citation_repair_session,
            turn_context=turn_context,
            prepared=prepared_continuation,
            committed_context_epoch=committed_context_epoch,
            stream_signals=self._admit_capture_policy(session.id, origin),
            terminal_citation_finalizer=terminal_citation_finalizer,
        )
        with self.store.durable_preparation_lock:
            self.store.validate_durable_acceptance_fingerprint(fingerprint)
            existing = self._durable_postcommit_continuations.get(
                preparation.preparation_id
            )
            if existing is not None and existing.fingerprint != fingerprint:
                raise RuntimeError("Durable continuation owner changed.")
            self._durable_postcommit_continuations[preparation.preparation_id] = (
                continuation
            )
        return await self.resume_durable_postcommit(preparation.preparation_id)

    def _postcommit_stopped_by_close(
        self,
        *,
        preparation_id: str,
        session_id: str,
        commit: Any,
        continuation: Any,
    ) -> ConsoleSubmitResult:
        """Terminal benign outcome when the chat closed mid-postcommit.

        TASK-22587. Runs the same continuation cleanup the normal tail runs --
        settle, drop the continuation, release prepared evidence -- but not the
        owner-changed check (the owner is legitimately gone) and not `retire`
        (closing already did it, and it is idempotent besides).
        """

        logger.debug("Durable postcommit sequence stopped: session closed")
        self.store.release_durable_postcommit_activity(preparation_id)
        self._settle_accepted_preparation(preparation_id)
        with self.store.durable_preparation_lock:
            current = self._durable_postcommit_continuations.pop(preparation_id, None)
            if current is not None:
                self._release_retired_prepared_evidence(current)
        return ConsoleSubmitResult(
            True,
            True,
            # TASK-22690: the generic copy every other close site uses. This
            # returned "" when TASK-22587 added it, which reads to the caller
            # as "no message" rather than "the session closed".
            SESSION_CLOSED_COPY,
            session_id=session_id,
            user_message_id=commit.user_message_id,
            assistant_message_id=commit.assistant_message_id,
            terminal_status=self.run_state_for(session_id).status,
            origin=continuation.origin,
            queue_entry_id=continuation.queue_entry_id,
            committed_context_epoch=continuation.committed_context_epoch,
            preparation_id=preparation_id,
            provider_started=True,
        )

    async def resume_durable_postcommit(
        self,
        preparation_id: str,
        *,
        continue_to_provider: bool = True,
    ) -> ConsoleSubmitResult:
        """Resume missing postcommit effects without allocating another turn.

        ``continue_to_provider=False`` is the Discard prerequisite path. It
        completes the accepted turn's required local publication effects, but
        deliberately stops before checkpoint CAS/provider entry so Discard can
        settle the still-accepted durable owner atomically.
        """

        with self.store.durable_preparation_lock:
            continuation = self._durable_postcommit_continuations.get(preparation_id)
            if continuation is not None:
                self.store.validate_durable_acceptance_fingerprint(
                    continuation.fingerprint
                )
        if continuation is None:
            return ConsoleSubmitResult(
                False,
                False,
                "Committed turn continuation is unavailable.",
                preparation_id=preparation_id,
            )
        commit = continuation.commit
        fingerprint = continuation.fingerprint
        session_id = continuation.session_id
        try:
            existing_effects = self.store.durable_postcommit_effects_for(
                preparation_id,
                fingerprint=fingerprint,
            )
        except ConsoleDurableAcceptanceRetired:
            # TASK-22587: the chat was closed before this resume began, so the
            # whole sequence is moot. This lookup sits BEFORE the try block
            # below, which is why guarding only the sequence was not enough.
            return self._postcommit_stopped_by_close(
                preparation_id=preparation_id,
                session_id=session_id,
                commit=commit,
                continuation=continuation,
            )
        if (
            existing_effects is not None
            and "checkpoint_transition" in existing_effects.completed
            and "provider_entry" not in existing_effects.completed
        ):
            self.store.mark_dispatch_recovery_needed(
                session_id,
                commit.assistant_message_id,
            )
            self._hydrate_dispatch_recovery_queue(session_id, force=True)
            return ConsoleSubmitResult(
                True,
                True,
                "Delivery status is unknown. Use Retry anyway or Discard.",
                session_id=session_id,
                user_message_id=commit.user_message_id,
                assistant_message_id=commit.assistant_message_id,
                terminal_status=self.run_state_for(session_id).status,
                origin=continuation.origin,
                queue_entry_id=continuation.queue_entry_id,
                committed_context_epoch=continuation.committed_context_epoch,
                preparation_id=preparation_id,
                provider_started=True,
            )
        assistant_holder: dict[str, ConsoleChatMessage] = {}

        def publish_owners() -> None:
            _user, assistant = self.store.publish_durable_turn_owners(
                session_id,
                commit,
                terminal_citation_finalizer=continuation.terminal_citation_finalizer,
                defer_terminal_persistence=(
                    continuation.citation_repair_session is not None
                ),
            )
            assistant_holder["assistant"] = assistant

        def clear_staged_input() -> None:
            self._release_prepared_evidence(continuation.prepared)
            if continuation.prepared is not None:
                for pending in continuation.prepared.attachments:
                    self.store.consume_pending_attachment(
                        session_id, pending.attachment_id
                    )
            revision = continuation.one_shot_prefill_revision
            if continuation.prefill_from_one_shot and revision is not None:
                self.store.consume_session_one_shot_prefill(session_id, revision)
            live_session = next(
                (row for row in self.store.sessions() if row.id == session_id), None
            )
            if (
                live_session is not None
                and live_session.draft == continuation.clean_draft
            ):
                live_session.draft = ""

        def queue_acknowledgement() -> None:
            if continuation.origin is ConsoleSubmissionOrigin.QUEUED:
                entry_id = continuation.queue_entry_id
                if entry_id is None or not (
                    self.prompt_queue_coordinator.acknowledge_durable_acceptance(
                        session_id,
                        entry_id=entry_id,
                        preparation_id=preparation_id,
                        context_epoch=continuation.committed_context_epoch,
                    )
                ):
                    raise RuntimeError(
                        "Durable queued acceptance could not settle its exact claim."
                    )
                return
            self.prompt_queue_coordinator.turn_accepted(
                session_id,
                origin=continuation.origin,
                context_epoch=continuation.committed_context_epoch,
                entry_id=None,
            )

        def project_workspace() -> None:
            live_session = next(
                row for row in self.store.sessions() if row.id == session_id
            )
            self.store._project_workspace_membership_after_commit(live_session)
            if self.store.has_pending_workspace_projection(session_id):
                raise RuntimeError("Workspace projection remains pending.")

        def accepted_hook() -> None:
            if continuation.origin is ConsoleSubmissionOrigin.MANUAL:
                callback = self.on_submission_accepted
                if callback is not None:
                    callback()

        async def prompt_history() -> None:
            history = self.prompt_history
            if history is not None and continuation.clean_draft.strip():
                await history.append(continuation.clean_draft)

        def publish_preparation() -> None:
            current = self._preparation_by_id(preparation_id)
            if (
                current is not None
                and current.state is ConsoleTurnPreparationState.ACCEPTED
            ):
                return
            if not self._transition_preparation(
                preparation_id,
                ConsoleTurnPreparationState.COMMITTING,
                ConsoleTurnPreparationState.ACCEPTED,
            ):
                raise RuntimeError(
                    "Prepared turn changed before acceptance publication."
                )

        async def transition_checkpoint() -> None:
            current_commit = self.store.durable_turn_commit_for(
                preparation_id, fingerprint=fingerprint
            )
            if current_commit is None:
                raise RuntimeError("Durable acceptance is unavailable.")
            repository = getattr(
                self.store.persistence, "console_dispatch_repository", None
            )
            if repository is None:
                raise RuntimeError("Durable dispatch repository is unavailable.")
            # TASK-22205: the pre-dispatch checkpoint CAS is the second
            # per-send BEGIN IMMEDIATE transaction; it runs off the event
            # loop behind the same await barrier as the turn commit. The
            # publication and preparation CAS below stay on the loop and
            # re-validate their owners, so an interleaved discard/close
            # during the await fails this effect exactly like a crash
            # between CAS and provider entry (a state resume already
            # recovers).
            result = await self._run_durable_db_call(
                repository.cas_state,
                ConsoleDispatchTransition(
                    assistant_message_id=current_commit.assistant_message_id,
                    expected_state=ConsoleDispatchCheckpointState.ACCEPTED,
                    expected_checkpoint_revision=(
                        current_commit.checkpoint.checkpoint_revision
                    ),
                    expected_user_message_version=(current_commit.user_message_version),
                    expected_assistant_message_version=(
                        current_commit.assistant_message_version
                    ),
                    new_state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
                    new_attempt_id=current_commit.checkpoint.attempt_id,
                ),
            )
            if result.status is not ConsoleDispatchResultStatus.COMMITTED:
                raise RuntimeError("Durable dispatch checkpoint transition failed.")
            if result.checkpoint is None:
                raise RuntimeError("Durable dispatch checkpoint is unavailable.")
            self.store.publish_durable_dispatch_checkpoint(
                session_id,
                result.checkpoint,
                in_flight=True,
            )
            if not self._transition_preparation(
                preparation_id,
                ConsoleTurnPreparationState.ACCEPTED,
                ConsoleTurnPreparationState.DISPATCH_STARTED,
            ):
                raise RuntimeError("Prepared turn changed before provider dispatch.")

        try:
            await self._run_durable_postcommit_effect(
                preparation_id,
                "identity_publication",
                lambda: self.store.publish_durable_turn_identity(session_id, commit),
                fingerprint=fingerprint,
            )
            await self._run_durable_postcommit_effect(
                preparation_id,
                "durable_owner_publication",
                publish_owners,
                fingerprint=fingerprint,
            )
            await self._run_durable_postcommit_effect(
                preparation_id,
                "staged_input_clearing",
                clear_staged_input,
                fingerprint=fingerprint,
            )
            await self._run_durable_postcommit_effect(
                preparation_id,
                "workspace_projection",
                project_workspace,
                fingerprint=fingerprint,
            )
            await self._run_durable_postcommit_effect(
                preparation_id,
                "queue_acknowledgement",
                queue_acknowledgement,
                fingerprint=fingerprint,
            )
            await self._run_durable_postcommit_effect(
                preparation_id,
                "accepted_hook",
                accepted_hook,
                fingerprint=fingerprint,
            )
            await self._run_durable_postcommit_effect(
                preparation_id,
                "prompt_history",
                prompt_history,
                fingerprint=fingerprint,
            )
            await self._run_durable_postcommit_effect(
                preparation_id,
                "preparation_publication",
                publish_preparation,
                fingerprint=fingerprint,
            )
            if not continue_to_provider:
                self._restore_dispatch_recovery_after_settlement_failure(
                    session_id,
                    commit.assistant_message_id,
                )
                if continuation.origin is ConsoleSubmissionOrigin.QUEUED:
                    self.prompt_queue_coordinator.retain_durable_acceptance(session_id)
                return ConsoleSubmitResult(
                    True,
                    False,
                    "Accepted turn prerequisites completed.",
                    session_id=session_id,
                    user_message_id=commit.user_message_id,
                    assistant_message_id=commit.assistant_message_id,
                    terminal_status=ConsoleRunStatus.BLOCKED,
                    origin=continuation.origin,
                    queue_entry_id=continuation.queue_entry_id,
                    committed_context_epoch=continuation.committed_context_epoch,
                    preparation_id=preparation_id,
                    provider_started=False,
                )
            await self._run_durable_postcommit_effect(
                preparation_id,
                "checkpoint_transition",
                transition_checkpoint,
                fingerprint=fingerprint,
            )
            assistant = assistant_holder.get("assistant")
            if assistant is None:
                assistant = self.store.get_message(commit.assistant_message_id)
            stream_result = await self._run_durable_postcommit_effect(
                preparation_id,
                "provider_entry",
                lambda: self._stream_assistant_response(
                    resolution=continuation.resolution,
                    provider_messages=continuation.provider_messages,
                    assistant_message_id=assistant.id,
                    prefill=continuation.prefill,
                    prefill_from_one_shot=continuation.prefill_from_one_shot,
                    one_shot_prefill_revision=(continuation.one_shot_prefill_revision),
                    skill_bindings=continuation.skill_bindings,
                    skill_bundle_block=continuation.skill_bundle_block,
                    citation_repair_session=continuation.citation_repair_session,
                    turn_context=continuation.turn_context,
                    preparation_id=None,
                    stream_signals=continuation.stream_signals,
                ),
                fingerprint=fingerprint,
            )
        except ConsoleDurableAcceptanceRetired:
            # TASK-22587: the user closed the chat mid-sequence. Every REMAINING
            # effect validates against a preparation that no longer exists, so
            # retirement is terminal-benign for the whole orchestration, not
            # just the effect that noticed it first.
            return self._postcommit_stopped_by_close(
                preparation_id=preparation_id,
                session_id=session_id,
                commit=commit,
                continuation=continuation,
            )
        except ConsoleDispatchSettlementError:
            self._restore_dispatch_recovery_after_settlement_failure(
                session_id,
                commit.assistant_message_id,
            )
            if continuation.origin is ConsoleSubmissionOrigin.QUEUED:
                self.prompt_queue_coordinator.retain_durable_acceptance(session_id)
            return ConsoleSubmitResult(
                True,
                True,
                "Accepted turn is retained for recovery.",
                session_id=session_id,
                user_message_id=commit.user_message_id,
                assistant_message_id=commit.assistant_message_id,
                terminal_status=ConsoleRunStatus.BLOCKED,
                origin=continuation.origin,
                queue_entry_id=continuation.queue_entry_id,
                committed_context_epoch=continuation.committed_context_epoch,
                preparation_id=preparation_id,
                provider_started=True,
            )
        except BaseException:
            # TASK-22587: this lookup runs inside the failure handler, so it
            # must not raise -- a close mid-turn retires the ledger and the
            # raise would REPLACE the failure being handled. The tombstone
            # keeps `completed`, so the answer survives the close.
            completed = self.store.durable_completed_effects_for(
                preparation_id, fingerprint=fingerprint
            )
            provider_started = "checkpoint_transition" in completed
            if self.store.dispatch_recovery_for_session(session_id) is None:
                self.store.publish_durable_recovery_owner(
                    session_id,
                    commit,
                    terminal_citation_finalizer=None,
                    defer_terminal_persistence=(
                        continuation.citation_repair_session is not None
                    ),
                )
            self._restore_dispatch_recovery_after_settlement_failure(
                session_id,
                commit.assistant_message_id,
            )
            if continuation.origin is ConsoleSubmissionOrigin.QUEUED:
                self.prompt_queue_coordinator.retain_durable_acceptance(session_id)
            return ConsoleSubmitResult(
                True,
                True,
                "Accepted turn is retained for recovery.",
                session_id=session_id,
                user_message_id=commit.user_message_id,
                assistant_message_id=commit.assistant_message_id,
                terminal_status=self.run_state_for(session_id).status,
                origin=continuation.origin,
                queue_entry_id=continuation.queue_entry_id,
                committed_context_epoch=continuation.committed_context_epoch,
                preparation_id=preparation_id,
                provider_started=provider_started,
            )
        # Terminal persistence and checkpoint deletion completed atomically
        # inside the stream finalizer. The volatile preparation can now leave.
        self._settle_accepted_preparation(preparation_id)
        with self.store.durable_preparation_lock:
            current = self._durable_postcommit_continuations.get(preparation_id)
            if current is None or current.fingerprint != fingerprint:
                if current is None and self.store.durable_acceptance_retired(
                    preparation_id, fingerprint
                ):
                    # TASK-22690: the user closed the chat while this sequence
                    # ran, so the close already dropped the continuation and
                    # retired the preparation. The tombstone proves it is THIS
                    # acceptance, which is what separates an ordinary close from
                    # an owner that genuinely changed underneath us. A fourth
                    # site of the conflation TASK-22587 removed.
                    return self._postcommit_stopped_by_close(
                        preparation_id=preparation_id,
                        session_id=session_id,
                        commit=commit,
                        continuation=continuation,
                    )
                raise RuntimeError("Durable continuation owner changed.")
            self._durable_postcommit_continuations.pop(preparation_id, None)
            self._release_retired_prepared_evidence(current)
            self.store.retire_durable_acceptance(preparation_id, fingerprint)
        self.store.release_durable_postcommit_activity(preparation_id)
        if not isinstance(stream_result, ConsoleSubmitResult):
            stream_result = ConsoleSubmitResult(True, True)
        return replace(
            stream_result,
            session_id=session_id,
            user_message_id=commit.user_message_id,
            assistant_message_id=commit.assistant_message_id,
            terminal_status=self.run_state_for(session_id).status,
            origin=continuation.origin,
            queue_entry_id=continuation.queue_entry_id,
            committed_context_epoch=continuation.committed_context_epoch,
            preparation_id=preparation_id,
            provider_started=True,
        )

    async def _resolve_dispatch_retry_context(
        self,
        session_id: str,
        recovery: Any,
    ) -> _DispatchRetryContext:
        """Freshly revalidate frozen authority/destination and rebuild history."""

        checkpoint = recovery.checkpoint
        if checkpoint is None:
            raise _DispatchRecoveryRefusal(
                "Dispatch recovery checkpoint is unavailable."
            )
        configuration = self.resolve_turn_configuration_snapshot(session_id)
        authority = await self._capture_turn_library_authority(
            session_id,
            configuration,
        )
        resolution = await self._resolve_for_send_bounded(
            configuration.provider_selection
        )
        if not getattr(resolution, "ready", False):
            raise _DispatchRecoveryRefusal(
                self._blocked_visible_copy(getattr(resolution, "visible_copy", ""))
            )
        destination = self._resolved_destination_for_context(resolution)
        frozen_authority = checkpoint.frozen_authority
        if (
            not self._dispatch_authority_matches(authority, frozen_authority)
            or destination.identity_key != checkpoint.resolved_destination.identity_key
        ):
            raise _DispatchRecoveryRefusal(
                "The provider destination or Library authority changed. Review it "
                "before retrying."
            )
        turn_context = self._finalize_turn_execution_context(
            configuration,
            authority,
            resolution,
        )
        provider_messages = self._provider_messages_for_session(
            session_id,
            before_message_id=recovery.assistant_message_id,
            annotate_ids=True,
            turn_context=turn_context,
        )
        if authority.policy.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC:
            user = self.store.get_message(checkpoint.user_message_id)
            request = LibraryRagSearchRequest(
                query=user.content,
                source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
                mode="rag",
                top_k=5,
                include_citations=True,
                scope=self._automatic_scope_for_authority(authority),
            )
            service = getattr(self.app, "library_rag_search_service", None)
            search = getattr(service, "search", None)
            if not callable(search):
                raise _DispatchRecoveryRefusal(
                    "Library retrieval is unavailable for retry."
                )
            kwargs: dict[str, object] = {
                "top_k": request.top_k,
                "include_citations": request.include_citations,
            }
            if request.scope is not None:
                kwargs["scope"] = request.scope
            try:
                async with asyncio.timeout(self._library_preparation_timeout):
                    raw = search(
                        request.query,
                        request.source_types,
                        request.mode,
                        **kwargs,
                    )
                    if inspect.isawaitable(raw):
                        raw = await raw
            except TimeoutError as exc:
                raise _DispatchRecoveryRefusal(
                    "Library retrieval timed out during retry."
                ) from exc
            result = _outcome_from_service_result(raw)
            if result.status not in {"ready", "empty"}:
                raise _DispatchRecoveryRefusal("Library retrieval failed during retry.")
            rows = tuple(result.results or ())
            if rows:
                bundle = build_library_rag_evidence_bundle(
                    rows,
                    query=request.query,
                )
                provider_messages = self._prepend_evidence_context(
                    provider_messages,
                    format_evidence_for_cited_answer(bundle),
                )
        return _DispatchRetryContext(
            resolution=resolution,
            authority=authority,
            destination=destination,
            provider_messages=provider_messages,
            turn_context=turn_context,
        )

    @staticmethod
    def _dispatch_authority_matches(
        current: ConsoleTurnLibraryAuthority,
        frozen: ConsoleTurnLibraryAuthority,
    ) -> bool:
        """Compare authority with only the exact first-save bookkeeping change."""

        if current.policy.error_code != frozen.policy.error_code:
            return False
        frozen_source = frozen.policy.source
        frozen_revision = frozen.policy.policy_revision
        current_source = current.policy.source
        current_revision = current.policy.policy_revision
        first_save = (
            frozen_source == "new_session"
            and frozen_revision is None
            and current_source == "durable"
            and current_revision == 1
        )
        if not first_save and (
            current_source != frozen_source or current_revision != frozen_revision
        ):
            return False
        normalized_policy = replace(
            current.policy,
            policy_revision=frozen_revision,
            source=frozen_source,
        )
        return (
            replace(
                current,
                policy=normalized_policy,
                attempt_id=frozen.attempt_id,
            )
            == frozen
        )

    async def retry_dispatch_recovery(
        self,
        session_id: str,
    ) -> ConsoleSubmitResult:
        """Explicitly retry one accepted/indeterminate owner without new rows."""

        recovery = self.store.dispatch_recovery_for_session(session_id)
        if recovery is None:
            return ConsoleSubmitResult(
                False, False, "No response recovery is available."
            )
        action_id = (
            ConsoleDispatchRecoveryActionId.RETRY_ANYWAY
            if recovery.kind
            in {
                ConsoleDispatchRecoveryKind.DISPATCH_STARTED,
                ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
            }
            else ConsoleDispatchRecoveryActionId.RETRY_RESPONSE
        )
        claimed = self.store.claim_dispatch_recovery_action(session_id, action_id)
        if claimed is None:
            action = next(
                (item for item in recovery.actions if item.action_id is action_id),
                None,
            )
            return ConsoleSubmitResult(
                False,
                False,
                action.disabled_reason
                if action is not None and action.disabled_reason
                else "That response recovery action is unavailable.",
            )
        retry_attempt_id: str | None = None
        try:
            if self._recovery_has_live_postcommit_continuation(session_id, claimed):
                preparation_id = claimed.preparation_id
                if preparation_id is None:  # pragma: no cover - authenticated above
                    raise _DispatchRecoveryRefusal(
                        "Committed turn continuation is unavailable."
                    )
                result = await self.resume_durable_postcommit(preparation_id)
                if self.store.dispatch_recovery_for_session(session_id) is not None:
                    return result
                await self._settle_recovered_queue_owner(session_id, claimed, result)
                return result
            context = await self._resolve_dispatch_retry_context(session_id, claimed)
            thinking_block = self._thinking_persistence_preflight(
                session_id=session_id,
                resolution=context.resolution,
            )
            if thinking_block is not None:
                self.store.release_dispatch_recovery_action(
                    session_id,
                    claimed.assistant_message_id,
                )
                return thinking_block
            checkpoint = claimed.checkpoint
            if checkpoint is None:
                raise _DispatchRecoveryRefusal(
                    "Dispatch recovery checkpoint is unavailable."
                )
            authority = context.authority
            retry_attempt_id = authority.attempt_id
            started = self.store.transition_dispatch_recovery_for_retry(
                session_id,
                assistant_message_id=claimed.assistant_message_id,
                new_attempt_id=authority.attempt_id,
            )
            if started is None:
                raise RuntimeError("Dispatch recovery changed before provider entry.")
            generation_token = self.store.begin_generation_attempt(
                claimed.assistant_message_id
            )
            self.store.prepare_dispatch_recovery_message(
                session_id,
                claimed.assistant_message_id,
                generation_token=generation_token,
            )
            turn_context = getattr(context, "turn_context", None)
            if not isinstance(turn_context, ConsoleTurnExecutionContext):
                turn_context = self._finalize_turn_execution_context(
                    self.resolve_turn_configuration_snapshot(session_id),
                    authority,
                    context.resolution,
                )
            result = await self._stream_assistant_response(
                resolution=context.resolution,
                provider_messages=context.provider_messages,
                assistant_message_id=claimed.assistant_message_id,
                turn_context=turn_context,
                generation_token=generation_token,
            )
        except asyncio.CancelledError:
            if self._retry_checkpoint_cas_completed(
                session_id,
                claimed,
                expected_attempt_id=retry_attempt_id,
            ):
                self._restore_dispatch_recovery_after_settlement_failure(
                    session_id,
                    claimed.assistant_message_id,
                )
            else:
                self.store.release_dispatch_recovery_action(
                    session_id,
                    claimed.assistant_message_id,
                )
            raise
        except _DispatchRecoveryRefusal as exc:
            self.store.release_dispatch_recovery_action(
                session_id,
                claimed.assistant_message_id,
            )
            return ConsoleSubmitResult(False, False, str(exc))
        except Exception:
            visible_copy = "Response recovery failed. Try again or discard."
            if self._retry_checkpoint_cas_completed(
                session_id,
                claimed,
                expected_attempt_id=retry_attempt_id,
            ):
                self._restore_dispatch_recovery_after_settlement_failure(
                    session_id,
                    claimed.assistant_message_id,
                )
            else:
                self.store.release_dispatch_recovery_action(
                    session_id,
                    claimed.assistant_message_id,
                )
                self._set_run_state(
                    ConsoleRunState(ConsoleRunStatus.BLOCKED, visible_copy),
                    session_id=session_id,
                )
            return ConsoleSubmitResult(
                False,
                False,
                visible_copy,
            )
        self._retire_live_recovery_continuation(claimed)
        await self._settle_recovered_queue_owner(session_id, claimed, result)
        return replace(
            result,
            session_id=session_id,
            user_message_id=(
                claimed.checkpoint.user_message_id if claimed.checkpoint else None
            ),
            assistant_message_id=claimed.assistant_message_id,
            queue_entry_id=claimed.queue_entry_id,
            preparation_id=claimed.preparation_id,
            provider_started=True,
        )

    def _retry_checkpoint_cas_completed(
        self,
        session_id: str,
        claimed: Any,
        *,
        expected_attempt_id: str | None,
    ) -> bool:
        """Detect an exact Retry CAS even if a local wrapper raises afterward."""

        before = claimed.checkpoint
        current = self.store.dispatch_recovery_for_session(session_id)
        after = current.checkpoint if current is not None else None
        if (
            before is None
            or after is None
            or current.assistant_message_id != claimed.assistant_message_id
            or before.state
            not in {
                ConsoleDispatchCheckpointState.ACCEPTED,
                ConsoleDispatchCheckpointState.DISPATCH_STARTED,
            }
            or type(expected_attempt_id) is not str
            or _DISPATCH_IDENTIFIER_RE.fullmatch(expected_attempt_id) is None
            or expected_attempt_id == before.attempt_id
        ):
            return False
        return after == replace(
            before,
            attempt_id=expected_attempt_id,
            state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
            checkpoint_revision=before.checkpoint_revision + 1,
            assistant_message_version=before.assistant_message_version + 1,
        )

    def _recovery_has_live_postcommit_continuation(
        self,
        session_id: str,
        recovery: Any,
    ) -> bool:
        """Authenticate an accepted owner against its app-lifetime effect ledger."""

        if recovery.kind is not ConsoleDispatchRecoveryKind.ACCEPTED:
            return False
        preparation_id = recovery.preparation_id
        if preparation_id is None:
            return False
        fingerprint = self.store.durable_acceptance_fingerprint_for(preparation_id)
        if fingerprint is None:
            # A process restart reconstructs from SQLite and has no live effect
            # ledger. Its accepted owner follows the ordinary durable Retry path.
            return False
        effects = self.store.durable_postcommit_effects_for(
            preparation_id,
            fingerprint=fingerprint,
        )
        if effects is None or "checkpoint_transition" in effects.completed:
            return False
        with self.store.durable_preparation_lock:
            continuation = self._durable_postcommit_continuations.get(preparation_id)
            if continuation is None:
                raise _DispatchRecoveryRefusal(
                    "Committed turn continuation is unavailable."
                )
            self.store.validate_durable_acceptance_fingerprint(continuation.fingerprint)
            if (
                continuation.fingerprint != fingerprint
                or continuation.session_id != session_id
                or continuation.commit.assistant_message_id
                != recovery.assistant_message_id
                or continuation.commit.checkpoint != recovery.checkpoint
                or continuation.queue_entry_id != recovery.queue_entry_id
                or continuation.origin.value
                != getattr(recovery.checkpoint, "origin", "")
            ):
                raise _DispatchRecoveryRefusal("Committed turn continuation changed.")
        return True

    async def discard_dispatch_recovery(
        self,
        session_id: str,
    ) -> ConsoleSubmitResult:
        """Atomically discard one exact owner while retaining its USER."""

        claimed = self.store.claim_dispatch_recovery_action(
            session_id,
            ConsoleDispatchRecoveryActionId.DISCARD,
        )
        if claimed is None:
            return ConsoleSubmitResult(
                False,
                False,
                "That response recovery action is unavailable.",
            )
        try:
            if self._recovery_has_live_postcommit_continuation(session_id, claimed):
                preparation_id = claimed.preparation_id
                if preparation_id is None:  # pragma: no cover - authenticated above
                    raise _DispatchRecoveryRefusal(
                        "Committed turn continuation is unavailable."
                    )
                prerequisite_result = await self.resume_durable_postcommit(
                    preparation_id,
                    continue_to_provider=False,
                )
                fingerprint = self.store.durable_acceptance_fingerprint_for(
                    preparation_id
                )
                effects = (
                    self.store.durable_postcommit_effects_for(
                        preparation_id,
                        fingerprint=fingerprint,
                    )
                    if fingerprint is not None
                    else None
                )
                if (
                    effects is None
                    or "preparation_publication" not in effects.completed
                ):
                    return ConsoleSubmitResult(
                        False,
                        False,
                        prerequisite_result.visible_copy
                        or "Accepted turn is retained for recovery.",
                    )
                reclaimed = self.store.claim_dispatch_recovery_action(
                    session_id,
                    ConsoleDispatchRecoveryActionId.DISCARD,
                )
                if (
                    reclaimed is None
                    or reclaimed.assistant_message_id != claimed.assistant_message_id
                    or reclaimed.preparation_id != claimed.preparation_id
                ):
                    raise _DispatchRecoveryRefusal(
                        "Committed turn continuation changed."
                    )
                claimed = reclaimed
        except _DispatchRecoveryRefusal as exc:
            self.store.release_dispatch_recovery_action(
                session_id,
                claimed.assistant_message_id,
            )
            return ConsoleSubmitResult(False, False, str(exc))
        if not self.store.settle_dispatch_recovery(
            session_id,
            assistant_message_id=claimed.assistant_message_id,
            terminal_state="discarded",
            content=CONSOLE_DISPATCH_DISCARDED_COPY,
        ):
            self.store.release_dispatch_recovery_action(
                session_id,
                claimed.assistant_message_id,
            )
            visible_copy = "Response recovery changed. Reload and try again."
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.BLOCKED, visible_copy),
                session_id=session_id,
            )
            return ConsoleSubmitResult(
                False,
                False,
                visible_copy,
            )
        result = ConsoleSubmitResult(
            True,
            False,
            CONSOLE_DISPATCH_DISCARDED_COPY,
            session_id=session_id,
            user_message_id=(
                claimed.checkpoint.user_message_id if claimed.checkpoint else None
            ),
            assistant_message_id=claimed.assistant_message_id,
            terminal_status=ConsoleRunStatus.STOPPED,
            queue_entry_id=claimed.queue_entry_id,
            preparation_id=claimed.preparation_id,
        )
        self._retire_live_recovery_continuation(claimed)
        await self._settle_recovered_queue_owner(session_id, claimed, result)
        return result

    def _retire_live_recovery_continuation(self, recovery: Any) -> None:
        """Drop app-lifetime acceptance state after explicit settlement."""

        preparation_id = recovery.preparation_id
        if preparation_id is None:
            return
        self._settle_accepted_preparation(preparation_id)
        with self.store.durable_preparation_lock:
            continuation = self._durable_postcommit_continuations.pop(
                preparation_id,
                None,
            )
            if continuation is not None:
                self._release_retired_prepared_evidence(continuation)
                self.store.retire_durable_acceptance(
                    preparation_id,
                    continuation.fingerprint,
                )

    def _release_retired_prepared_evidence(
        self,
        continuation: _DurablePostcommitContinuation,
    ) -> None:
        """Release an exact frozen evidence lease once at owner retirement."""

        prepared = continuation.prepared
        lease = prepared.staged_evidence if prepared is not None else None
        if lease is not None and not lease.released:
            try:
                self._release_prepared_evidence(prepared)
            except Exception:
                # Settlement/app disposal has already made this lease
                # non-retryable. A view cleanup failure must not retain the
                # accepted request body or its bound UI owner indefinitely.
                lease.released = True
                lease.launch = None
                lease.capture_result = None
                lease.release = None

    def _retire_all_live_recovery_continuations(self) -> None:
        """Bound app-lifetime accepted-turn content at permanent teardown."""

        with self.store.durable_preparation_lock:
            continuations = tuple(self._durable_postcommit_continuations.values())
            for continuation in continuations:
                current = self._durable_postcommit_continuations.get(
                    continuation.preparation_id
                )
                if current is not continuation:
                    continue
                preparation = self._preparation_by_id(continuation.preparation_id)
                if preparation is not None:
                    self._drop_preparation(
                        continuation.preparation_id,
                        expected_states=frozenset({preparation.state}),
                    )
                self._preparation_outcomes.pop(continuation.preparation_id, None)
                self._prepared_send_continuations.pop(continuation.preparation_id, None)
                self._durable_postcommit_continuations.pop(
                    continuation.preparation_id, None
                )
                self._release_retired_prepared_evidence(continuation)
                self.store.retire_durable_acceptance(
                    continuation.preparation_id,
                    continuation.fingerprint,
                )

    async def _settle_recovered_queue_owner(
        self,
        session_id: str,
        recovery: Any,
        result: ConsoleSubmitResult,
    ) -> None:
        if recovery.queue_entry_id is None or recovery.preparation_id is None:
            return
        await self.prompt_queue_coordinator.settle_dispatch_recovery_and_drain(
            session_id,
            queue_entry_id=recovery.queue_entry_id,
            preparation_id=recovery.preparation_id,
            terminal_status=result.terminal_status or ConsoleRunStatus.COMPLETED,
        )

    def new_session(
        self,
        *,
        title: str | None = None,
        settings: ConsoleSessionSettings | None = None,
        ephemeral: bool = False,
    ) -> ConsoleChatSession:
        """Create and activate a new native Console session.

        Args:
            ephemeral: Create the session temporary -- never written to local
                storage until explicitly saved.
        """
        next_number = len(self.store.sessions()) + 1
        session = self.store.create_session(
            title=title or f"Chat {next_number}",
            settings=settings,
            ephemeral=ephemeral,
        )
        # `create_session` above already activated the new session, so the
        # default (no explicit session_id -> active session) targets the
        # session JUST created here -- which is fresh/never-recorded and
        # therefore already idle, making this call a no-op today. Left
        # unchanged (rather than reaching for the session being replaced)
        # for the same reason `switch_session` below is: per-session run
        # state is meant to persist on the session you're leaving, not be
        # wiped just because a sibling session appeared.
        self._clear_terminal_run_state()
        # Fix wave (IMPORTANT 2, final review): re-derive the mounted
        # approval card for the brand-new (now active) session, exactly
        # like `switch_session`/`close_session`'s neighbor-activation
        # branch already do -- without this, a round mounted on the
        # session being left behind stayed rendered over the new tab
        # (`create_session` above activates `session`, but nothing else
        # ever told the card to re-derive for it). A fresh session can
        # never itself have a parked payload, so this always resolves to
        # `None` here -- i.e. it always clears -- but going through the
        # same `_parked_approval_payloads` lookup (rather than a bespoke
        # unconditional clear) keeps this call site honest with the same
        # "card state derives from the run's pending review state" rule
        # every other activation path follows.
        if self.set_pending_approval is not None:
            self.set_pending_approval(
                self._head_round_payload(self._parked_approval_payloads, session.id)
            )
        # TASK-910: same re-derive for the skill-install/script cards -- a
        # brand-new session can never itself have a parked confirm, so this
        # always resolves to clearing whatever the session being left behind
        # had shown (mirrors the approval re-derive immediately above).
        self._remount_parked_skill_install(session.id)
        self._remount_parked_skill_script(session.id)
        return session

    def _maybe_auto_title_session(
        self, session: ConsoleChatSession, draft: str
    ) -> None:
        """Title a default-named session from its first accepted message."""
        if session.persisted_conversation_id is not None:
            return
        if not is_default_console_session_title(session.title):
            return
        derived = derive_console_session_title(draft)
        if derived:
            self.store.rename_session(
                session.id, derived
            )  # (session, persisted) — auto-title best-effort

    def update_provider_selection(self, selection: ConsoleProviderSelection) -> None:
        """Sync controller provider settings from a Console selection."""
        # task-15511: the clear-below compares EFFECTIVE selections, so the
        # model term is what would actually run -- `explicit or configured`
        # (the exact resolution `_build_console_turn_execution_context` and
        # the send path use). `configured_model` alone is DERIVED state: it
        # resolves late (e.g. once a provider key exists) and can flip on a
        # routine resync with nothing user-visible changing. Comparing it
        # separately made that churn look like a settings change and wiped a
        # COMPLETED run state seconds after the run finished.
        previous_selection = (
            self.provider,
            self.model or self.configured_model,
            self.base_url,
            self.temperature,
            self.top_p,
            self.min_p,
            self.top_k,
            self.max_tokens,
            self.seed,
            self.presence_penalty,
            self.frequency_penalty,
            self.reasoning_effort,
            self.reasoning_summary,
            self.verbosity,
            self.thinking_effort,
            self.thinking_budget_tokens,
            self.streaming,
            self.system_prompt,
        )
        self.provider = selection.provider
        self.model = selection.explicit_model
        self.configured_model = selection.configured_model
        self.base_url = selection.base_url
        self.temperature = selection.temperature
        self.top_p = selection.top_p
        self.min_p = selection.min_p
        self.top_k = selection.top_k
        self.max_tokens = selection.max_tokens
        self.seed = selection.seed
        self.presence_penalty = selection.presence_penalty
        self.frequency_penalty = selection.frequency_penalty
        self.reasoning_effort = selection.reasoning_effort
        self.reasoning_summary = selection.reasoning_summary
        self.verbosity = selection.verbosity
        self.thinking_effort = selection.thinking_effort
        self.thinking_budget_tokens = selection.thinking_budget_tokens
        self.streaming = selection.streaming
        self.system_prompt = selection.system_prompt
        current_selection = (
            self.provider,
            self.model or self.configured_model,
            self.base_url,
            self.temperature,
            self.top_p,
            self.min_p,
            self.top_k,
            self.max_tokens,
            self.seed,
            self.presence_penalty,
            self.frequency_penalty,
            self.reasoning_effort,
            self.reasoning_summary,
            self.verbosity,
            self.thinking_effort,
            self.thinking_budget_tokens,
            self.streaming,
            self.system_prompt,
        )
        if current_selection != previous_selection:
            # No session in scope here -- this is a global provider/model
            # settings change, not tied to any particular session's run.
            # Active-session UI path: clears whatever the user is currently
            # looking at (parallel-agents spec §2).
            self._clear_terminal_run_state()

    def update_agent_runtime(
        self, *, enabled: bool, bridge: "ConsoleAgentBridge | None"
    ) -> None:
        """Refresh the agent-runtime gate and bridge from a fresh config read.

        Both were previously read only once, at controller construction
        (Plan-B Task 6 Important 3): the ``[console] agent_runtime``
        kill-switch is meant to take effect on the next send, but a
        controller built before a config change stayed on its original
        path until the owning screen tore it down. The owner must call
        this every time it refreshes provider selection (see
        ``update_provider_selection``) so the gate and bridge presence
        never go stale.
        """
        self._agent_runtime_enabled = enabled
        self._agent_bridge = bridge
        # PR3a-2 Task 3: a refreshed bridge is a fresh fan-out registry --
        # re-register the usage fold on it (replace-by-name makes calling
        # this with the SAME bridge a safe no-op).
        self._register_fleet_usage_reattach(bridge)
        # PR3a-2 Task 5: same rule for the auto-wake consumer.
        self._register_fleet_wake(bridge)

    def switch_session(self, session_id: str) -> ConsoleChatSession:
        """Activate an existing native Console session."""
        # Resolve the OUTGOING session BEFORE `store.switch_session` below
        # moves `active_session_id` -- the no-arg default on
        # `_clear_terminal_run_state` would otherwise target the session
        # being ARRIVED AT (active_session_id already points there by the
        # time it runs). Per the spec's "clear the session you are leaving
        # if terminal" semantic, every session-scoped write in this refactor
        # is explicit, so this one is too: pass the outgoing session's id
        # directly. A session you're ARRIVING AT keeps whatever terminal/
        # in-flight state it already had (parallel-agents spec §2).
        previous_session_id = self.store.active_session_id
        session = self.store.switch_session(session_id)
        # Parallel-agents spec §6: visiting the session you just switched TO
        # clears its unvisited outcome marker -- must run AFTER the store
        # swap above so `session_id` really is the new active session by
        # the time downstream reads (e.g. `run_marker_for`) observe it.
        self.mark_session_visited(session_id)
        if previous_session_id is not None:
            self._clear_terminal_run_state(session_id=previous_session_id)
        # Task 9 (parked background approvals): mount `session_id`'s
        # parked round, if any, through the SAME UI bridge
        # `request_mcp_approvals` uses for an active session's round --
        # `self.set_pending_approval` is always safe to call with `None`
        # too (clears whatever the session being LEFT had shown), so this
        # single call both mounts a newly-visited parked card AND hides a
        # departing session's card in one step. No `call_from_thread`
        # marshal needed: `switch_session` always runs on the UI/main
        # thread already (same convention as `mark_session_visited`/
        # `_clear_terminal_run_state` above). Card state is entirely
        # derived from `_parked_approval_payloads` (the round's own
        # retained pending-review payload) every time this runs, never
        # from whatever the card happened to be showing before -- so
        # switching away and back re-mounts it unchanged (spec).
        #
        # Supersedes the pre-Task-9 `_deny_pending_approval_on_context_
        # change()` call that used to run here: that assumed only one
        # approval round could ever be in flight controller-wide (true
        # before Task 3's concurrent runs), so ANY switch force-denied it.
        # Once a background session can carry its own live round, denying
        # it just for being switched away from directly contradicts
        # parking -- the round now stays alive until its own resolution
        # (decision, cancel, or timeout).
        if self.set_pending_approval is not None:
            self.set_pending_approval(
                self._head_round_payload(self._parked_approval_payloads, session_id)
            )
        # TASK-910: skill-install/script confirms now get the SAME park/
        # re-derive treatment as MCP batch approvals above -- a context
        # change (switch away) no longer force-denies either bridge's
        # pending confirm; the round stays alive (parked, badge + one
        # toast via `park_pending_approval`) until its own resolution,
        # cancellation, or shutdown. Superseded the pre-TASK-910
        # `_deny_pending_skill_install_on_context_change()`/`_deny_pending_
        # skill_script_on_context_change()` calls that used to run here
        # unconditionally on every switch.
        self._remount_parked_skill_install(session_id)
        self._remount_parked_skill_script(session_id)
        return session

    def close_session(self, session_id: str) -> ConsoleChatSession | None:
        """Close an existing native Console session.

        Args:
            session_id: Native Console session ID to close.

        Returns:
            The session activated after closing, or ``None`` when no sessions remain.
        """
        recovery = self.store.dispatch_recovery_for_session(session_id)
        if (
            recovery is not None
            and recovery.recovery_needed
            and recovery.kind
            in {
                ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
                ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
            }
        ):
            raise RuntimeError(
                "Finish or discard the pending turn before closing this chat."
            )
        # Revoke file authority before any close action can wake a worker or
        # remove the owning session from the store.
        self._scratch_spaces.close(session_id)
        forget_file_authority = getattr(
            self._agent_bridge,
            "forget_session_file_authority",
            None,
        )
        if callable(forget_file_authority):
            try:
                forget_file_authority(session_id)
            except Exception:  # noqa: BLE001 -- teardown remains best-effort
                logger.warning("close_session could not forget run-log authority")
        # Queue tombstone MUST precede stop/cancel: cancellation can wake a
        # terminal callback, which must observe that no next claim is legal.
        self.prompt_queue_coordinator.mark_closing(session_id)
        preparation = self.store.preparation_for_session(session_id)
        if preparation is not None and preparation.state in {
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.READY,
            ConsoleTurnPreparationState.PAUSED,
        }:
            self._abandon_preparation(preparation.preparation_id)
        # PR3b Task 5 (Qodo #1808 finding 3): closing a session is
        # DESTRUCTIVE -- `ConsoleChatStore.close_session` purges every
        # message and drops the session -- so its fleet must die with it.
        # Navigation-away teardowns (`leave_console`/`shutdown`) preserve
        # the conversation and its survivors rightly continue; here a
        # surviving child would outlive its own conversation with no
        # panel row left to cancel it from, a wake targeting a dead
        # conversation, and a leaked unseen-mark. The conversation id is
        # derived NOW, while the session still exists (persisted id when
        # set -- the key the bridge's fleet state actually lives under),
        # and every live child goes through the explicit whole-fleet
        # path: `cancel_all_subagents` reuses the per-handle cancel, so
        # approval-card revocation and cancelled-is-never-retained ride
        # along. getattr-guarded and wrapped: a bare bridge double, no
        # bridge, or a raising cancel must never break a close.
        fleet_conversation_id = self._agent_conversation_id(session_id)
        cancel_all = (
            getattr(self._agent_bridge, "cancel_all_subagents", None)
            if self._agent_bridge is not None
            else None
        )
        if callable(cancel_all):
            try:
                cancelled_children = int(cancel_all(fleet_conversation_id))
                if cancelled_children:
                    logger.info(
                        "close_session cancelled {} sub-agent(s) of the closed conversation",
                        cancelled_children,
                    )
            except Exception:  # noqa: BLE001 -- teardown never fails on a fleet read
                logger.warning(
                    "close_session could not cancel the conversation's sub-agents"
                )
        repair_session = self._active_citation_repair_sessions.get(session_id)
        self.clear_original_attempts_for_session(session_id)
        owns_active_stream = self._active_stream_belongs_to_session(session_id)
        submit_tasks = self._submit_tasks_for_session(session_id)
        if repair_session is not None and owns_active_stream:
            repair_session.cancel_reason = "session_close"
        if owns_active_stream:
            self._signal_stop(session_id=session_id)
            task = self._active_stream_tasks.get(session_id)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STOPPED, SESSION_CLOSED_COPY),
                # `session_id` here is the session being CLOSED, which the
                # `_active_stream_belongs_to_session` guard above confirms
                # owns the active stream -- not necessarily the currently
                # ACTIVE session (you can close a background tab while
                # viewing another one), so this must be explicit rather
                # than falling back to the active-session default.
                session_id=session_id,
            )
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        for submit_task in submit_tasks:
            if submit_task is current_task:
                continue
            self._signal_stop(session_id=session_id)
            self._cancel_task_on_owner_loop(submit_task)
        if preparation is not None:
            self._preparation_outcomes.pop(preparation.preparation_id, None)
            self._prepared_send_continuations.pop(preparation.preparation_id, None)
        previous_active_id = self.store.active_session_id
        with self.store.durable_preparation_lock:
            durable_continuations = tuple(
                continuation
                for continuation in self._durable_postcommit_continuations.values()
                if continuation.session_id == session_id
            )
            for continuation in durable_continuations:
                self._durable_postcommit_continuations.pop(
                    continuation.preparation_id, None
                )
                self._release_retired_prepared_evidence(continuation)
                self.store.retire_durable_acceptance(
                    continuation.preparation_id, continuation.fingerprint
                )
        closed = self.store.close_session(session_id)
        self.prompt_queue_coordinator.remove_session(session_id)
        self._clear_project_instruction_delivery(session_id)
        new_active_id = self.store.active_session_id
        if (
            owns_active_stream
            and repair_session is not None
            and self._active_citation_repair_sessions.get(session_id) is repair_session
        ):
            self._active_citation_repair_sessions.pop(session_id, None)
        # Parallel-agents spec §6: closing the ACTIVE session auto-activates
        # a neighbor (`ConsoleChatStore.close_session`, console_chat_store.py
        # ~594-604) -- that neighbor is now the VIEWED session exactly as if
        # `switch_session` had navigated to it, so its unvisited outcome
        # must clear the same way, AND (Task 9) its parked approval card
        # (if any) must mount the same way too -- closing a background tab
        # must never leave the newly-viewed session's own pending approval
        # invisible just because it arrived here via auto-activation rather
        # than an explicit switch. Closing a BACKGROUND (non-active) session
        # leaves `active_session_id` unchanged, so this is a no-op in that
        # case.
        if new_active_id is not None and new_active_id != previous_active_id:
            self.mark_session_visited(new_active_id)
            if self.set_pending_approval is not None:
                self.set_pending_approval(
                    self._head_round_payload(
                        self._parked_approval_payloads, new_active_id
                    )
                )
            # TASK-910: same re-derive for the skill-install/script cards --
            # closing the ACTIVE session auto-activates a neighbor, which is
            # now the VIEWED session exactly as if `switch_session` had
            # navigated to it.
            self._remount_parked_skill_install(new_active_id)
            self._remount_parked_skill_script(new_active_id)
        return closed

    def original_attempt_for_message(self, message_id: str) -> str | None:
        """Return and refresh one current-session original attempt."""
        body = self._original_attempts.get(message_id)
        if body is None:
            return None
        try:
            message = self.store.get_message(message_id)
        except KeyError:
            self._original_attempts.pop(message_id, None)
            return None
        presentation = message.citation_presentation
        if presentation is None or not presentation.original_attempt_available:
            self._original_attempts.pop(message_id, None)
            return None
        self._original_attempts.move_to_end(message_id)
        return body

    def clear_original_attempt(self, message_id: str) -> None:
        """Forget one preview and clear its content-free availability flag."""
        self._original_attempts.pop(message_id, None)
        self._set_original_attempt_availability(message_id, False)

    def clear_original_attempts_for_session(self, session_id: str) -> None:
        """Forget every original attempt owned by one Console session."""
        for message_id in tuple(self._original_attempts):
            try:
                owner_session_id = self.store.session_id_for_message(message_id)
            except KeyError:
                self._original_attempts.pop(message_id, None)
                continue
            if owner_session_id == session_id:
                self.clear_original_attempt(message_id)

    def _remember_original_attempt(
        self,
        message_id: str,
        body: str,
        *,
        update_presentation: bool = True,
    ) -> None:
        """Insert one successful repair preview into the eight-entry LRU."""
        self._original_attempts.pop(message_id, None)
        self._original_attempts[message_id] = body
        if update_presentation:
            self._set_original_attempt_availability(message_id, True)
        while len(self._original_attempts) > 8:
            evicted_id, _evicted_body = self._original_attempts.popitem(last=False)
            self._set_original_attempt_availability(evicted_id, False)

    def _set_original_attempt_availability(
        self,
        message_id: str,
        available: bool,
    ) -> None:
        """Update only the bounded presentation flag for a live message."""
        try:
            message = self.store.get_message(message_id)
        except KeyError:
            return
        presentation = message.citation_presentation
        if presentation is None:
            return
        self.store.set_citation_presentation(
            message_id,
            ConsoleCitationPresentation(
                phase=presentation.phase,
                notice_code=presentation.notice_code,
                original_attempt_available=available,
            ),
        )

    def _signal_stop(self, *, session_id: str) -> None:
        """Set the shared UI-facing stop flag AND ``session_id``'s own
        permanent per-run cancel signal.

        ``_stop_requested`` stays a single shared flag, but as of Fix
        round 1 (Critical 1) NO run's own cancellation-check loop reads it
        any more -- ``should_cancel`` (``_run_agent_reply``) and the
        direct/legacy stream path's own checks (``_stream_assistant_
        response``) read ONLY their run's own ``_active_cancel_events[
        owner_id]``, captured by closure. Reading the shared flag from
        inside a specific run's loop let ANY session's Stop/Close silently
        truncate an unrelated, untouched session's still-streaming reply
        (Fix round 1 finding).

        F5 fix (Qodo wave): the three worker-thread approval/confirm
        bridges no longer read ``_stop_requested`` either (see
        ``_is_active_session_cancelled``) -- a single session's Stop/Close
        must not deny an unrelated session's in-flight approval round any
        more than it may truncate an unrelated session's stream. Real
        process teardown (``shutdown()``) is the one case where denying
        every session's round at once is correct; that now goes through
        the dedicated, never-reset ``_shutdown_requested`` instead.
        ``_stop_requested`` itself is left set here for any other/legacy
        reader (kept for back-compat; this method's own contract has
        always been "set it," not "this is its only reader").

        ``_active_cancel_events[session_id]``, once set here, is never
        reset for that run, so a still-running bridge thread always
        observes the Stop correctly regardless of what the coroutine side
        has already reset (task-227). Every caller (``close_session``,
        ``stop_active_run``, ``shutdown``) already knows the exact session
        it is signalling -- there is deliberately no active-session
        fallback here, unlike ``_set_run_state``.
        """
        self._stop_requested = True
        cancel_event = self._active_cancel_events.get(session_id)
        if cancel_event is not None:
            cancel_event.set()

    def _is_active_session_cancelled(self) -> bool:
        """Best-effort cancel-signal check that falls back to the VIEWED
        session -- the pre-Task-9 behavior of the three worker-thread
        approval/confirm bridges below (``request_mcp_approvals``,
        ``request_skill_install_confirm``, ``request_skill_script_
        confirm``), preserved here as the fallback ``_is_session_
        cancelled`` uses when a caller has no ``session_id`` of its own to
        pass (e.g. a legacy direct call in an older test). See
        ``_is_session_cancelled``'s own docstring for the Task 9 fix this
        was carved out of, and for the F5 fix (Qodo wave) that replaced
        the shared, lifecycle-reset ``_stop_requested`` flag with the
        never-reset ``_shutdown_requested`` in that same fallback branch.
        """
        cancel_event = self._active_cancel_events.get(
            self.store.active_session_id or ""
        )
        return cancel_event is not None and cancel_event.is_set()

    def _bind_round_cancel_signal(
        self, session_id: str | None
    ) -> threading.Event | None:
        """Resolve the cancel Event an approval round must listen to, ONCE,
        at ARM time -- PR3a-1 Task 6b, audit F4.

        Callers pass the resolved Event to ``_is_session_cancelled`` for
        the whole life of the round instead of letting that method re-read
        ``_active_cancel_events[session_id]`` on every poll. The
        difference only shows up once a run can outlive the turn that
        started it (``[agents] subagents_outlive_turn``, PR3a-1), and it
        shows up in both directions, silently:

        * **Between turns** the per-session entry has been popped, so a
          poll-time ``.get()`` returns ``None`` and the round listens to
          nothing at all -- until the user sends again.
        * **During the NEXT turn** the entry is the NEXT turn's Event, so
          pressing Stop on turn 2 denied turn 1's surviving child's
          still-open card and failed a legitimate tool call closed, with
          nothing anywhere saying the denial came from another turn.
          Reproduced by execution in
          ``Tests/UI/test_console_mcp_approval.py::test_the_next_turns_
          stop_does_not_deny_an_earlier_turns_survivors_card`` (the audit
          had this direction as inference from structure only).

        Binding by value is the same discipline ``_run_agent_reply``'s own
        ``should_cancel`` closure already applies to this exact Event, and
        the pull-side counterpart to ``revoke_approval_rounds_for_run``'s
        run-keyed push: a round is answerable to the run that armed it,
        never to whatever run happens to own the session later.

        ``None`` -- a round armed with no turn in flight, i.e. a survivor's
        -- means no SESSION Stop can reach it. That is deliberate and it is
        not a lost signal: such a round is still released by its own run's
        revoke (``revoke_approval_rounds_for_run``, what the fleet panel's
        Cancel presses), by its own approval timeout, and by
        ``_shutdown_requested`` on teardown. What it is no longer reachable
        by is an UNRELATED turn's Stop button, which never had any business
        answering for it. (The Stop button itself no-ops between turns
        anyway -- ``stop_active_run`` returns ``False`` with no active
        assistant message -- so nothing that used to work stops working.)

        Args:
            session_id: The round's OWNING session, or ``None`` for a
                legacy caller with no session context (which keeps the
                viewed-session fallback; see ``_is_session_cancelled``).

        Returns:
            The owning run's cancel Event, or ``None`` when the session has
            no run in flight right now.
        """
        if session_id is None:
            return None
        return self._active_cancel_events.get(session_id)

    def _bind_visit_cancel_signal(self) -> threading.Event:
        """Capture THIS visit's teardown Event, ONCE, at ARM time.

        task-15860 (the lifetime landing). ``_shutdown_requested`` used to
        be per-instance and never reset, so reading it live on every poll
        was safe. It is now per-VISIT: ``leave_console()`` sets it and the
        next ``attach_view`` REPLACES the attribute with a fresh, unset
        Event.

        A poll site that re-read ``self._shutdown_requested`` would
        therefore answer with the NEXT visit's Event and **resurrect a
        round the previous visit's teardown already denied** -- a round
        armed on visit 1, still polling while the user is on visit 2,
        would silently un-deny itself and go on to approve or execute a
        tool call for a UI that no longer exists. Same discipline, same
        reason, as ``_bind_round_cancel_signal``'s arm-time binding of the
        per-run cancel event.

        Every site fails CLOSED today: an armed round observes its
        captured Event set and denies. Nothing here can make a round
        fail open.

        **task-15860, plan Task 5 -- the DETACHED arm.** With the runtime
        outliving the screen, a round can be armed when no visit is open
        at all (a headless wake turn reaching a risk-tagged tool). That
        round was not armed *during* the visit whose Event is currently
        set, so answering to it is a category error -- measured, it denied
        the round at the first 1.0s poll, silently, making a headless wake
        unable to use any risk-tagged tool and giving the user no chance
        to answer. Such a round binds ``_headless_visit_cancel`` instead:
        unset now, set by the NEXT ``leave_console()`` (the user has seen
        it by then) or by ``begin_shutdown()``. A DISPOSED controller is
        excluded -- it keeps the permanently-set Event, so an app-exit
        round still denies immediately.

        Qodo audit S2 (PR 1752): "no visit open" is now the explicit
        ``_visit_open`` lifecycle flag, not an ``is_set()`` inference. The
        inference read a NEVER-VISITED controller (constructor Event
        unset) as "visit in progress", so a wake-at-launch round bound the
        constructor Event -- which the first ``begin_visit()`` then
        replaced, orphaning the round beyond the reach of every teardown
        signal until its own deadline. Only a round armed while a visit
        is genuinely open (and not already ending) binds the visit Event;
        every other live round binds the headless Event, which BOTH
        teardown paths set (``leave_console``/``begin_shutdown``, via
        ``_cancel_headless_rounds``).

        Returns:
            The Event whose set() this round must treat as cancellation.
        """
        if self._disposed:
            return self._shutdown_requested
        if self._visit_open and not self._shutdown_requested.is_set():
            return self._shutdown_requested
        event = self._headless_visit_cancel
        if event is None or event.is_set():
            event = threading.Event()
            self._headless_visit_cancel = event
        return event

    def _is_session_cancelled(
        self,
        session_id: str | None,
        *,
        cancel_event: threading.Event | None,
        visit_event: threading.Event,
    ) -> bool:
        """Cancellation check for the three worker-thread approval/confirm
        bridges below, scoped to the round's OWN cancel event
        (PA-T9 finding #1 fix; bound at arm time since PR3a-1 Task 6b).

        ``cancel_event`` is keyword-only and has NO default on purpose:
        every call site must state which Event this round answers to, so a
        future bridge cannot silently inherit "no signal" (fail-open) by
        forgetting the argument. Get it from
        ``_bind_round_cancel_signal(session_id)`` at arm time -- see that
        method for why binding by value, rather than re-reading
        ``_active_cancel_events`` per poll, is what keeps one turn's Stop
        out of another turn's surviving child's approval round.

        Pre-Task-9, all three bridges checked ``self._stop_requested or
        self._is_active_session_cancelled()`` -- the shared global flag
        OR'd with the VIEWED session's cancel event, regardless of which
        session's round was actually waiting. Once background sessions can
        each carry their own in-flight approval round (parked or not),
        that was a real cross-session bug: Session A's Stop
        (``stop_active_run``/``close_session``, via ``_signal_stop``)
        always sets the shared ``_stop_requested`` flag alongside A's own
        cancel event, so the OR-check let A's Stop spuriously deny B's
        completely unrelated, still-waiting approval batch.

        Fix: when ``session_id`` is known, check ONLY that session's own
        ``_active_cancel_events`` entry -- never the shared flag. This
        still correctly resolves every INTENTIONAL global-reach case:
        ``shutdown()`` (the one caller that must stop every session at
        once) signals every live session's cancel event individually
        (``_signal_stop(session_id=...)`` in its own per-session loop), so
        a round scoped to ANY session still observes shutdown via its own
        event -- ``_stop_requested`` was never the mechanism that made
        shutdown reach a specific round, just a side effect of
        ``_signal_stop`` also setting it.

        ``session_id=None`` (a caller with no session context of its own --
        e.g. an existing test calling ``request_mcp_approvals`` directly
        with no ``session_id=`` kwarg) falls back to the exact pre-Task-9
        behavior via ``_is_active_session_cancelled``, so those callers'
        existing global-flag expectations are unchanged.

        F5 fix (Qodo wave, folded in during the PR2 restack): that
        ``session_id=None`` fallback used to OR in the shared
        ``_stop_requested`` flag, which (a) any session's Stop set
        regardless of which round was waiting, and (b) various run
        lifecycles reset to ``False`` mid-flight, making whether a
        still-polling bridge observed an earlier Stop a race. It now ORs
        in ``_shutdown_requested`` instead -- set exactly once, only by
        ``shutdown()``, and never reset -- so a legacy no-session caller's
        "global stop denies" expectation is preserved for the one case
        this controller INSTANCE is ever torn down for (see ``shutdown()``
        's own docstring for exactly what that covers -- NOT only real
        process exit) where that is actually correct, without
        reintroducing cross-session poisoning for everyday per-session
        Stop/Close.

        TASK-1052 fix: the ``session_id is not None`` branch now ALSO ORs
        in ``_shutdown_requested``. Previously it checked ONLY that
        session's own ``_active_cancel_events`` entry, relying entirely on
        ``shutdown()``'s per-session ``_signal_stop`` fanout (see the
        docstring paragraph above) to ever reach a real-session round --
        but that fanout walks a SNAPSHOT of ``_active_stream_tasks`` taken
        when ``shutdown()``/``close_session`` runs. A round armed for a
        session BEFORE that session is registered there is invisible to
        the snapshot and was previously left to fail closed only via its
        own (up to ~120s) confirm/approval timeout -- promptness, not
        correctness, but still a real gap for a signal this controller
        instance's teardown is supposed to reach every live round with,
        unconditionally.

        Correction (review, TASK-1052): an earlier revision of this
        docstring justified ORing in ``_shutdown_requested`` here by
        calling it "real process teardown" and treating that as
        inherently global/safe. That premise was FALSE: ``shutdown()`` is
        also called from ordinary Console-screen unmount
        (``ChatScreen.on_unmount``), which fires on every navigation AWAY
        from the Console tab, not only on app exit -- so
        ``_shutdown_requested`` can be set on a controller instance the
        user is still actively using the app around. The actual safety
        argument does not rest on "global by definition"; it rests on
        this controller's OWN lifecycle: ``ChatScreen`` only ever
        constructs a fresh ``ConsoleChatController`` lazily
        (``_ensure_console_chat_controller``) after ``on_unmount`` has
        both run this instance's ``shutdown()`` and dropped the screen's
        reference to it, so a torn-down instance -- flag permanently set
        or not -- is never reused for a later Console visit, and no round
        still parked on it could ever be resolved through a UI that no
        longer exists anyway. ``_shutdown_requested`` is set exactly once,
        only by ``shutdown()``, and never reset for THIS instance's
        lifetime (see ``shutdown()``'s own docstring and its ``self.
        _shutdown_requested.set()`` call), so ORing it in here for a real
        ``session_id`` can never wrongly deny a live round while this
        controller instance is still the one actually in use -- it can
        only ever fire once this instance itself is being (or has been)
        torn down. This does NOT widen scoping for everyday per-session
        Stop/Close: ``_signal_stop`` still only touches the ONE session's
        own cancel event; an unrelated session's Stop still leaves both
        this branch's checks unset.
        """
        if session_id is not None:
            # task-15860: `visit_event`, NOT a fresh read of
            # `self._shutdown_requested` -- that attribute is replaced by
            # the next visit's `begin_visit()`, and re-reading it here
            # would resurrect a round this visit's teardown already denied.
            # See `_bind_visit_cancel_signal`.
            if visit_event.is_set():
                return True
            # PR3a-1 Task 6b (audit F4): the ARM-TIME binding, not a fresh
            # `self._active_cancel_events.get(session_id)` per poll. See
            # `_bind_round_cancel_signal` for the two silent cross-turn
            # failures that re-read produced.
            return cancel_event is not None and cancel_event.is_set()
        return visit_event.is_set() or self._is_active_session_cancelled()

    # -- MCP batch-approval bridge (task-5) ----------------------------------

    def request_mcp_approvals(
        self, pending: list[MCPPendingCall], *, session_id: str | None = None
    ) -> dict[str, str]:
        """Bridge one batch of pending tool-approval rows to the Console UI and back.

        TASK-630: OWNER-AGNOSTIC, despite the legacy ``mcp`` in the name.
        Since TASK-545/P1's run-level ``build_tool_review_hook``, the rows
        handed here may come from MCP tools OR from built-in agent-runtime
        tools (``server_key="agent:builtin"``); every row is marshalled to
        the same ``ChatApprovalCard`` and resolved through the same
        Event-polling loop below. There is no separate approval path for
        built-ins -- a reader assuming "MCP-only" here would go looking for
        one that does not exist. (The name is kept: it is the wire between
        this method, ``resolve_pending_approval``, and the round-id
        plumbing, and renaming it is churn without a defect.)

        WORKER THREAD. Bound (via a ``functools.partial`` binding this
        run's ``session_id``, Task 9) as ``MCPToolProvider``'s
        ``approval_callback`` and ``build_tool_review_hook``'s
        ``request_approvals``, so this runs on the agent bridge's
        background OS thread (the ``asyncio.to_thread`` call inside
        ``_run_agent_reply``) -- it must never touch a widget directly,
        only through ``self.app.call_from_thread``.

        Builds a fresh ``threading.Event`` + shared decisions dict (stored
        under this round's own entry in ``_pending_approval_rounds``, keyed
        by a freshly minted ``round_id`` -- see that map's own docstring
        for why a single shared slot, or a slot keyed by session id alone,
        could not survive concurrent sessions or same-session round
        replacement). Either MOUNTS the card immediately (``session_id`` is
        the currently ACTIVE/viewed session, or unknown -- legacy
        no-session callers keep the pre-Task-9 always-mount behavior) or
        PARKS it (``session_id`` is a DIFFERENT, background session --
        Task 9: the retained ``payload`` goes into
        ``_parked_approval_payloads`` for ``switch_session`` to mount
        later, and ``park_pending_approval`` fires the fleet badge +
        one-shot toast instead of touching the mounted-card slot). PR0
        adds a third case: an ACTIVE-session round that is not its
        session's FIFO head neither mounts nor parks -- an older sibling
        still owns the card, and this round's payload is retained under
        its own ``round_id`` until that sibling's teardown promotes it.
        Either way it then polls ``event.wait(1.0)`` re-checking this run's OWN
        cancel signal (``_is_session_cancelled``) and -- only when a
        POSITIVE timeout is configured (ADR-067: the default is 0 = none)
        -- a deadline, every second until one of three things happens: the
        user submits a decision (``resolve_pending_approval``, called from
        the UI thread once the card's own stamped ``round_id`` is delivered
        back, sets the Event -- Fix round 1: NOT "whichever round belongs
        to the active session", see ``resolve_pending_approval``'s own
        docstring for why that was a real cross-session misattribution
        hazard), the run is cancelled/torn down (``_is_session_cancelled``
        -- F5 fix, Qodo wave: this round's OWN cancel event, or real
        process teardown via ``_shutdown_requested``, never any OTHER
        session's bare Stop -- see that method's own docstring), or the
        configured approval timeout elapses. With no deadline armed the
        round simply waits for one of the first two, however long the
        human takes -- the wait is marked in ``Agents.human_input_wait``
        so a per-call wrapper hosting it pauses its ceiling. Whichever
        unique ``llm_name`` never received an explicit decision by then
        fails closed to ``"deny"``
        (cancellation) or ``"timeout"`` (deadline) -- see
        ``MCPToolProvider._apply_verdict`` for how each decision string is
        consumed. The mounted card (if any) is always cleared afterwards
        (``finally``), regardless of outcome -- but ONLY if this round's
        session is STILL the active one at that moment, so a background
        round resolving (timeout/cancel) while some OTHER session's card is
        showing never clobbers it.

        Args:
            pending: One turn's pending tool calls awaiting approval,
                possibly containing repeated ``llm_name``s (T3: calls
                sharing a name share one verdict).
            session_id: The run's OWNING session (Task 3 threads it through
                ``_run_agent_reply``). ``None`` preserves every pre-Task-9
                call site's behavior (always mounts against whatever
                session is active at ROUND-key time; no parking).

        Returns:
            A decision string (``approve_once``/``approve_session``/
            ``always_allow``/``deny``/``timeout``) for every unique
            ``llm_name`` in ``pending``.
        """
        unique_names: list[str] = []
        seen: set[str] = set()
        call_by_name: dict[str, "MCPPendingCall"] = {}
        for call in pending:
            if call.llm_name not in seen:
                seen.add(call.llm_name)
                unique_names.append(call.llm_name)
                call_by_name[call.llm_name] = call
        if not unique_names:
            return {}
        if self.app is None:
            # ADR-067: with no app bridge nothing can ever surface or
            # resolve this round, and the no-deadline default means the
            # loop below would never end -- fail closed on the spot,
            # mirroring both skill confirms' own no-app guards. (A wired
            # app with a missing card seam is NOT this case: the round
            # stays resolvable via `resolve_pending_approval`/cancel, and
            # `_marshal_pending_approval` already no-ops its missing seam.)
            return {name: "deny" for name in unique_names}

        event = threading.Event()
        decisions: dict[str, str] = {}
        # Fix round 1 (review CRITICAL finding): keyed by a freshly minted
        # ROUND id, not by session id (or the active session) -- a session-
        # keyed slot is ambiguous the moment either (a) the ACTIVE session
        # changes between this round starting and the user's decision
        # arriving (`ApprovalDecided` travels as an async Textual message,
        # so a `switch_session` can land in that gap), or (b) a second
        # round starts for the SAME session before a first round's stale
        # decision message is delivered -- either way a session-keyed slot
        # would let a stale/misdirected decision resolve the WRONG round.
        # `round_id` is stamped into `payload` below, round-trips through
        # `ChatApprovalCard.set_batch` -> `ApprovalDecided` ->
        # `resolve_pending_approval`, and is the ONLY thing that round is
        # ever resolved by -- mirrors `_pending_skill_script_rounds`'
        # identical `request_id`-keyed defense.
        round_id = str(uuid4())
        owning_session_id = (
            session_id
            if session_id is not None
            else (self.store.active_session_id or "")
        )
        # PR3a-1 Task 6b (audit F4): the cancel signal THIS round answers
        # to, resolved once, HERE, and passed to every poll below. See
        # `_bind_round_cancel_signal`.
        round_cancel_event = self._bind_round_cancel_signal(session_id)
        # task-15860: the visit's teardown Event, captured at ARM time for
        # the same reason the run's cancel event is -- see
        # `_bind_visit_cancel_signal`.
        visit_cancel_event = self._bind_visit_cancel_signal()
        # PR2a Task 7: which RUN armed this round. Read from the
        # `run_context` ContextVar, which `AgentService` binds around both
        # arming paths -- the per-turn review hook (`build_tool_review_
        # hook`/`build_mcp_review_hook`/`build_local_review_hook`, which
        # call straight through to this method on their own run's thread)
        # and each tool invocation (the single-call fallback approval
        # `MCPToolProvider.invoke` raises through `approval_callback`,
        # which has no run_id parameter to thread at all). A session id
        # cannot substitute: every child of a fleet turn shares the
        # parent's session, so only the run id can tell a cancelled
        # child's card apart from its live sibling's. `""` (no run bound
        # -- e.g. the MCP workbench's Test Tool) is never revocable, which
        # is correct: no run owns it.
        owning_run_id = current_run_id()
        round_state: dict[str, Any] = {
            "event": event,
            "decisions": decisions,
            "session_id": owning_session_id,
            "run_id": owning_run_id,
            # The names this round must answer for -- what `revoke_
            # approval_rounds_for_run` fills with "deny".
            "names": tuple(unique_names),
            # Flipped by revocation. Re-read after the wait below so a
            # decision that lands in `decisions` AFTER the revoke (the
            # `ApprovalDecided` message is async, and `resolve_pending_
            # approval` snapshots the box before writing to it) can never
            # turn a revoked round back into an approval.
            "revoked": False,
        }
        # F2b fix (Qodo wave): guard the round registration -- the UI
        # thread's `resolve_pending_approval` (TASK-913: fails closed by
        # round_id now, no more active-session scan) and the
        # `fleet_summary_counts` sync tick can read/iterate this map
        # concurrently with this worker thread's own writes.
        with self._approval_state_lock:
            self._pending_approval_rounds[round_id] = round_state

        timeout_seconds = self._resolve_mcp_approval_timeout_seconds()
        # ADR-067: <= 0 arms NO deadline -- the round waits for a decision
        # or this run's cancellation, however long the human takes (the
        # card renders no countdown copy for 0; see
        # `format_approval_deadline`).
        deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        payload = {
            "round_id": round_id,
            "session_id": owning_session_id,
            "calls": [
                {
                    "llm_name": call.llm_name,
                    "server_key": call.server_key,
                    "tool_name": call.tool_name,
                    "server_label": call.server_label,
                    "arguments": dict(call.arguments or {}),
                    "reason": call.reason,
                    "options": list(call.options),
                    "path_precheck_failed": call.path_precheck_failed,
                }
                for call in pending
            ],
            "timeout_seconds": timeout_seconds,
            # Qodo PR #1836 finding 1: the absolute deadline, so a mount
            # that happens AFTER arm (promotion, switch-back, attach) can
            # show the remaining window instead of the arm-time total --
            # see `_head_round_payload`'s snapshot.
            "deadline_monotonic": deadline,
        }
        # Task 9: park rather than mount when this round's session is a
        # DIFFERENT, background session -- `session_id is None` (a legacy
        # caller with no session context) always mounts, matching every
        # pre-Task-9 call site.
        is_parked = session_id is not None and session_id != (
            self.store.active_session_id or ""
        )
        # PR0: legacy `session_id is None` callers never park and never
        # queue -- they keep the unconditional mount below.
        is_head = True
        if session_id is not None:
            # Register THIS round's own id directly here (worker thread,
            # plain-dict/set mutation -- same no-marshal convention as
            # `_active_cancel_events` elsewhere in this class) so it is
            # authoritative regardless of whether a UI bridge happens to be
            # wired. TASK-1050 (Defect A): round-keyed, not a plain
            # boolean -- a sibling round from this bridge or either of the
            # other two (skill-install/skill-script confirms) for the SAME
            # session stays independently tracked, so THIS round's own
            # teardown can never clear a badge a sibling still needs.
            # `park_pending_approval`/`ChatScreen._park_console_approval`
            # only falls back to the deprecated boolean shim when NO round
            # is registered yet (`has_pending_approval_round`), so it never
            # double-counts against this call.
            self.add_pending_round(session_id, round_id)
            # Fix wave (CRITICAL 1, final review): retain THIS round's
            # payload for EVERY session-attributed round -- mounted or
            # parked -- not just a parked one. `switch_session` re-derives
            # the card EXCLUSIVELY from `_parked_approval_payloads` (never
            # from whatever the card happened to already be showing), so a
            # round that mounted immediately (session_id was the active
            # session at round-start) was previously unrecoverable the
            # moment the user switched away and back: the lookup found
            # nothing, mounted `None`, and the round silently hung with a
            # stale NEEDS_APPROVAL badge and no card until its 120s
            # timeout. The `finally` below already pops this key
            # unconditionally (whenever `session_id is not None`,
            # regardless of `is_parked`) -- storing it unconditionally
            # here too makes retention symmetric with that cleanup, per
            # spec §5 ("card state survives tab switches") for every round,
            # not only parked ones.
            # PR0: keyed by ROUND, and the return says whether THIS round
            # is its session's FIFO head. A non-head round must not mount:
            # an older sibling is still holding the card, and evicting it
            # is exactly the task-15661 defect this replaced.
            is_head = self._park_round_payload(
                self._parked_approval_payloads, round_id, payload
            )

        try:
            if self._approval_view_is_detached():
                # task-15860 Task 5: no Console view exists, so BOTH the
                # mount seam and the park seam are `None` and neither
                # branch below would surface anything at all. Announce
                # app-wide instead -- the toast renders on whatever screen
                # the user is actually on, which is the only seam that can
                # reach them here.
                self._announce_detached_approval(owning_session_id)
            elif is_parked:
                if self.app is not None and self.park_pending_approval is not None:
                    self.app.call_from_thread(self.park_pending_approval, session_id)
            elif is_head:
                self._marshal_pending_approval(payload)
            # ADR-067: mark this run as waiting on a human decision for the
            # duration of the wait, so a per-call wrapper hosting this round
            # (the invoke-path fallback approval) pauses its deadline -- an
            # indefinite wait must not trip `max_tool_call_seconds`.
            with use_human_input_wait(owning_run_id):
                while not event.wait(_MCP_APPROVAL_POLL_SECONDS):
                    if self._is_session_cancelled(
                        session_id,
                        cancel_event=round_cancel_event,
                        visit_event=visit_cancel_event,
                    ):
                        # Finding I3: a stop/unmount that resolves THIS round
                        # denies every still-undecided call, but
                        # `run_agent_loop`'s own `should_cancel()` check fires
                        # for every call in this turn's batch BEFORE any of
                        # them reaches `invoke()` -- so the "deny" verdict
                        # stamped below is never consumed there and would
                        # otherwise leave no audit record at all (contrast
                        # with the timeout branch, whose calls DO still reach
                        # `invoke()`'s own gate and get logged there, since a
                        # timeout is not itself a cancellation). Log directly
                        # here, best-effort, for exactly the names this branch
                        # is about to fail closed.
                        cancelled_names = [
                            name for name in unique_names if name not in decisions
                        ]
                        for name in unique_names:
                            decisions.setdefault(name, "deny")
                        self._record_cancelled_approval_decisions(
                            cancelled_names,
                            call_by_name,
                        )
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        for name in unique_names:
                            decisions.setdefault(name, "timeout")
                        break
            # PR2a Task 7: a revoked round answers "deny" for every name,
            # unconditionally -- it does not consult `decisions` at all.
            # The run this round belongs to has been cancelled or
            # abandoned, and `resolve_pending_approval` can still write
            # into the shared `decisions` box after revocation (it
            # snapshots the box under the lock, then updates it outside),
            # so honouring that box here would let a click delivered
            # microseconds after the cancellation execute the tool for
            # real. `revoke_approval_rounds_for_run` already filled every
            # name with "deny"; this is the guard that makes it stick.
            with self._approval_state_lock:
                was_revoked = bool(round_state.get("revoked"))
            if was_revoked:
                # Same audit gap Finding I3 documents for the cancellation
                # branch above: the child's loop is being torn down, so
                # these calls never reach `invoke()`'s own gate and would
                # otherwise leave no record of having been denied.
                self._record_cancelled_approval_decisions(
                    list(unique_names), call_by_name
                )
                return {name: "deny" for name in unique_names}
            # Any name the resolution path above didn't already cover (e.g.
            # a partial/empty decisions dict handed to `resolve_pending_
            # approval`) fails closed to "deny" rather than silently
            # dropping the call from the returned mapping.
            for name in unique_names:
                decisions.setdefault(name, "deny")
            # Finding F4: build the snapshot by keyed lookup over the
            # (locally-owned, never-mutated) `unique_names` list rather
            # than `dict(decisions)` -- the latter iterates `decisions`
            # itself, which `resolve_pending_approval` can concurrently
            # `.update()` from the UI thread; a same-size update can't
            # change dict length, so this is unreachable today, but a
            # keyed `.get()` per name can never raise "dictionary changed
            # size during iteration" regardless. The `setdefault` pass
            # above already guarantees every name resolves, so `.get`'s
            # own "deny" fallback here is a belt-and-suspenders no-op, not
            # a second source of truth.
            return {name: decisions.get(name, "deny") for name in unique_names}
        finally:
            # F2b fix (Qodo wave): guard the pop -- `resolve_pending_
            # approval`'s round_id lookup and the `fleet_summary_counts`
            # sync tick can each observe this map from the UI thread while
            # this worker thread tears the round down.
            with self._approval_state_lock:
                self._pending_approval_rounds.pop(round_id, None)
            # PR0: drop exactly THIS round's retained payload. Pre-PR0 the
            # slot was shared per session, so the pop had to be guarded by
            # an order-dependent "is this the last armed round for the
            # session" test to avoid discarding a still-armed sibling's
            # only copy. Per-round storage makes that guard meaningless --
            # each round owns its own key -- and takes the accepted
            # last-armed-wins limitation (task-15661) with it.
            self._unpark_round_payload(self._parked_approval_payloads, round_id)
            if session_id is not None:
                # TASK-1050 (Defect A): discard ONLY this round's own id --
                # the badge clears only once every bridge round for this
                # session (this one included) has resolved.
                self.discard_pending_round(session_id, round_id)
            # PR0: re-derive the card from the session's remaining FIFO
            # head rather than deciding whether to CLEAR it. This
            # subsumes `_clear_pending_approval_if_round_is_current`'s
            # two-part TOCTOU guard: clearing was order-dependent, so the
            # decision could go stale between a worker-thread snapshot and
            # the UI thread running it; a head re-derive is a pure
            # function of current state. The race-proofing principle is
            # unchanged -- `_remount_head` still computes the answer
            # INSIDE the callable that runs on the UI thread, never from a
            # snapshot taken here.
            # `owning_session_id`, not `session_id`: a legacy no-session
            # caller retains no payload, so its head resolves to `None` and
            # the card clears exactly as the pre-PR0 unconditional clear
            # did -- unless a real session-attributed sibling is armed for
            # the session it mounted over, in which case that sibling's
            # card is (correctly) what stays up.
            try:
                # Qodo PR #1836 finding 2: a legacy no-session round's card
                # mounted unconditionally and may sit over ANY session by
                # now -- pass None so `_remount_head` re-derives for the
                # session active when the callback runs, not the arm-time
                # snapshot (whose mismatch would strand the card).
                self._remount_head(
                    self._parked_approval_payloads,
                    self.set_pending_approval,
                    owning_session_id if session_id is not None else None,
                )
            except Exception:  # noqa: BLE001 -- suppress teardown-time errors
                logger.opt(exception=True).debug(
                    "Failed to marshal approval remount during teardown"
                )

    def _record_cancelled_approval_decisions(
        self,
        names: list[str],
        call_by_name: dict[str, "MCPPendingCall"],
    ) -> None:
        """Best-effort audit log for calls denied by a stop/unmount mid-approval.

        Finding I3: see the cancellation branch's own comment in
        ``request_mcp_approvals`` for why this direct call is necessary --
        `MCPToolProvider._record_decision_safe` (the normal recording
        path) is never reached for these calls, since `run_agent_loop`
        cancels the whole turn before dispatching any of them. Reached via
        `self.app.unified_mcp_service` (the same object
        `_compose_mcp_provider` built this run's `MCPToolProvider` from --
        see that method), never raises: a missing app/service, or the
        service lacking `record_tool_decision`, is a silent no-op, and any
        exception the real call raises is logged and swallowed, mirroring
        `MCPToolProvider._record_decision_safe`'s own never-raise
        contract.
        """
        service = getattr(self.app, "unified_mcp_service", None)
        if service is None:
            return
        record = getattr(service, "record_tool_decision", None)
        if not callable(record):
            return
        for name in names:
            call = call_by_name.get(name)
            if call is None:
                continue
            try:
                record(
                    call.server_key,
                    call.tool_name,
                    decision="denied",
                    initiator="agent",
                    error="run stopped while approval pending",
                )
            except Exception:  # noqa: BLE001 -- best-effort audit trail only
                logger.opt(exception=True).debug(
                    "Failed to record cancelled MCP approval decision"
                )

    def _marshal_pending_approval(self, payload: dict[str, Any] | None) -> None:
        """Push ``payload`` (or clear it) onto the UI thread, if wired."""
        if self.app is not None and self.set_pending_approval is not None:
            self.app.call_from_thread(self.set_pending_approval, payload)

    # -- PR0: per-round retained payloads ------------------------------
    #
    # All three bridges' retained-payload maps are keyed by ROUND id and
    # guarded by `_approval_state_lock`. The mounted card is always the
    # session's FIFO HEAD -- its oldest-armed round. Dict insertion order
    # is arm order, which is why every write goes through
    # `_park_round_payload` and nothing assigns into these maps directly.
    #
    # This replaces the pre-PR0 single-slot-per-session maps, whose
    # last-armed-wins semantics let a second same-session round overwrite
    # the first's payload and strand it until timeout (task-15661).

    @staticmethod
    def _head_round_payload_locked(
        store: dict[str, dict[str, Any]], session_id: str | None
    ) -> dict[str, Any] | None:
        """The session's oldest-armed payload. Caller holds the lock."""
        for payload in store.values():
            if payload.get("session_id") == session_id:
                return payload
        return None

    def _park_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str, payload: dict[str, Any]
    ) -> bool:
        """Retain ``payload``; return whether it is now its session's head.

        A round that is NOT the head must not mount -- an older sibling is
        still holding the card.
        """
        session_id = payload.get("session_id")
        with self._approval_state_lock:
            store[round_id] = payload
            head = self._head_round_payload_locked(store, session_id)
        return head is payload

    def _head_round_payload(
        self, store: dict[str, dict[str, Any]], session_id: str
    ) -> dict[str, Any] | None:
        """The payload whose card ``session_id`` should currently show.

        Qodo PR #1836 finding 1: a round's auto-deny deadline starts at ARM
        time, but a queued round can mount much later (promotion at the
        head's resolve, a switch back to a parked session, a headless
        attach). Handing the card the arm-time ``timeout_seconds`` then
        overstates the decision window -- "Auto-denies in 2:00" on a card
        whose worker denies in seconds. When the payload carries its
        ``deadline_monotonic``, return a shallow SNAPSHOT whose
        ``timeout_seconds`` is the remaining time at this call; the
        retained payload is never mutated, so every later re-derive
        computes its own fresh snapshot. A payload without a deadline
        (ADR-067 arms none for ``timeout <= 0`` script confirms) passes
        through untouched. ``_park_round_payload``'s ``head is payload``
        identity check goes through ``_head_round_payload_locked`` and is
        unaffected.
        """
        with self._approval_state_lock:
            payload = self._head_round_payload_locked(store, session_id)
        if payload is None:
            return None
        deadline = payload.get("deadline_monotonic")
        if not deadline:
            return payload
        snapshot = dict(payload)
        snapshot["timeout_seconds"] = max(0.0, deadline - time.monotonic())
        return snapshot

    def _session_round_payloads(
        self, store: dict[str, dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Every payload ``store`` retains for ``session_id``, arm order first.

        Key-shape agnostic on purpose: it matches on the payload's own
        ``session_id``, so it reads a PR0 round-keyed map and a not-yet-
        migrated session-keyed one identically (`ChatScreen._current_park_
        round_ids` scans all three bridges' maps through it).
        """
        with self._approval_state_lock:
            return [
                payload
                for payload in store.values()
                if payload.get("session_id") == session_id
            ]

    def _unpark_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str
    ) -> None:
        """Drop one round's retained payload. Idempotent."""
        with self._approval_state_lock:
            store.pop(round_id, None)

    def _remount_head(
        self,
        store: dict[str, dict[str, Any]],
        setter: Callable[[dict[str, Any] | None], None] | None,
        session_id: str | None,
    ) -> None:
        """WORKER THREAD: enqueue a head re-derive onto the UI thread.

        Replaces the pre-PR0 two-part TOCTOU guard. That guard existed
        because CLEARING the card was order-dependent -- whether to clear
        depended on which sibling resolved first, and a worker-thread
        snapshot of that answer could be stale by the time the UI thread
        ran it. Re-deriving the head is order-INDEPENDENT: it is a pure
        function of current state. The race-proofing principle is
        unchanged -- the decision still runs inside the callable on the UI
        thread, never from a snapshot -- but the decision itself is now one
        lookup instead of an identity check plus a sibling check.

        ``session_id=None`` means "the session being VIEWED when the
        callback runs" (Qodo PR #1836 finding 2): a legacy no-session round
        mounts unconditionally, so its card can be sitting over ANY session
        by teardown time, and re-deriving for the arm-time snapshot id
        no-ops on a mismatch -- stranding the card where the deleted
        pre-PR0 guard cleared it unconditionally. Resolving the active
        session inside ``_apply`` clears the stale legacy card AND restores
        that session's own real head if it has one -- strictly better than
        the old unconditional clear, which could wipe a real sibling's
        card. Session-attributed rounds keep the exact-match guard so a
        background teardown can never touch the viewed session's card.
        """
        if self.app is None or setter is None:
            return

        def _apply() -> None:
            if session_id is None:
                active = self.store.active_session_id or ""
                setter(self._head_round_payload(store, active))
                return
            if session_id != (self.store.active_session_id or ""):
                return
            setter(self._head_round_payload(store, session_id))

        self.app.call_from_thread(_apply)

    def remount_pending_approval_for_active_session(self) -> bool:
        """Mount the ACTIVE session's still-armed approval round, if any.

        task-15860 Task 5. UI THREAD (called from
        ``ConsoleRuntime.attach_view``, which runs on it). Re-derives the
        card from ``_parked_approval_payloads`` exactly as
        ``switch_session`` does -- same single source of truth, no second
        copy of "what is this session's card showing".

        Deliberately mounts NOTHING when no round is armed: pushing
        ``None`` here would clear a card on every new claim, and an attach
        is not a reason to hide anything.

        PR0 (task-15661, fixed): ``_parked_approval_payloads`` is keyed by
        ROUND now, so two rounds armed for one session each keep their own
        payload. This mounts the session's FIFO HEAD -- the oldest-armed
        round -- and each later sibling mounts in turn as the head ahead of
        it resolves. Covered by ``Tests/UI/test_console_headless_approval.
        py::test_two_headless_rounds_each_mount_in_turn``.

        Returns:
            True when a card was mounted.
        """
        if self.set_pending_approval is None:
            return False
        session_id = self.store.active_session_id
        if not session_id:
            return False
        # The pre-PR0 `still_armed` pre-test is redundant: a round unparks
        # its own payload in its own teardown, so a payload present here
        # necessarily belongs to a live round.
        payload = self._head_round_payload(self._parked_approval_payloads, session_id)
        if payload is None:
            return False
        self.set_pending_approval(payload)
        return True

    def _approval_view_is_detached(self) -> bool:
        """True when NO Console view can surface an approval round.

        task-15860 Task 5. Deliberately a property of the SEAMS, not of
        ``ConsoleRuntime.view``: these two slots are what an announcement
        would travel through, and ``detach_view`` clears them together
        (``CONSOLE_VIEW_HOOK_SLOTS``). Asking the runtime instead would
        make this method wrong in exactly the case it exists for -- a
        controller whose seams are unwired for any other reason would
        still surface nothing while claiming a view.
        """
        return self.set_pending_approval is None and self.park_pending_approval is None

    def _announce_detached_approval(self, session_id: str) -> None:
        """Raise the app-wide toast for a round armed with no Console view.

        WORKER THREAD. ``App.notify`` is documented thread-safe (it posts
        a message), so this needs no ``call_from_thread`` marshal -- and
        the toast renders on whatever screen the user is currently
        looking at, which is the whole point: the screen-owned seam
        (``ChatScreen._park_console_approval``) is unreachable here.

        Best-effort in both directions. An app double with no ``notify``
        (several controller-level tests) is silently skipped, and a
        raising/incompatible ``notify`` is logged rather than allowed to
        break the round -- a missing toast must never turn into a missing
        approval.

        Args:
            session_id: The round's owning session, used only to name the
                conversation in the notice.
        """
        app = self.app
        notify = getattr(app, "notify", None) if app is not None else None
        if not callable(notify):
            return
        title = ""
        try:
            for session in self.store.sessions():
                if session.id == session_id:
                    title = str(getattr(session, "title", "") or "")
                    break
        except Exception:  # noqa: BLE001 -- a missing title never blocks the notice
            title = ""
        where = f" in {escape_markup(title)}" if title else ""
        message = (
            f"Agent{where} needs approval to use a tool. "
            "Open Console to review -- nothing runs until you answer."
        )
        try:
            notify(message, severity="warning")
        except TypeError:
            # An app double whose `notify` takes the message alone.
            try:
                notify(message)
            except Exception:  # noqa: BLE001
                logger.debug("Detached approval notice could not be delivered")
        except Exception as exc:  # noqa: BLE001 -- surfacing is best-effort
            logger.debug(
                "Detached approval notice raised (exception_type={})",
                type(exc).__name__,
            )

    def _resolve_mcp_approval_timeout_seconds(self) -> float:
        if self.mcp_approval_timeout_seconds is not None:
            try:
                return float(self.mcp_approval_timeout_seconds())
            except Exception:  # noqa: BLE001 -- fail open to the documented default
                pass
        try:
            return float(
                get_cli_setting(
                    "mcp",
                    "approval_timeout_seconds",
                    _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS,
                )
            )
        except (TypeError, ValueError):
            return _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS

    # -- MCP provider registration (task-6) ----------------------------------

    def _publish_mcp_inspector_counts(
        self,
        tool_count: int | None,
        not_connected_count: int | None,
    ) -> None:
        """Publish this run's MCP catalog counts for the inspector's "MCP" row.

        ``setattr`` onto ``self.app`` -- the exact same object
        ``ChatScreen._console_mcp_tool_count``/``_console_mcp_not_connected_
        count`` ``getattr`` from (wired onto this controller as ``self.app``
        by ``ChatScreen._ensure_console_chat_controller``). Every
        ``_compose_mcp_provider`` return path calls this: ``(None, None)``
        is the row's documented "absent" contract (see
        ``console_display_state._mcp_inspector_row``) for the no-service /
        kill-switch-on / compose-failed / empty-catalog paths; the eligible
        path publishes the real counts.

        No separate UI refresh is triggered here by design -- piggybacking
        on machinery the screen already runs, not a new mechanism:
        ``_compose_mcp_provider`` always executes on the main loop while
        this run's state is already STREAMING (set moments earlier by
        ``_run_agent_reply``), so the screen's own active-run poll timer
        (``ChatScreen._start_console_transcript_sync_timer``, already
        ticking every 0.2s by the time this runs -- started before
        ``submit_draft`` is even awaited) and the guaranteed post-
        ``submit_draft`` sync (``ChatScreen._submit_console_native_draft``)
        both already re-derive inspector state from these attributes on
        their own next pass.
        """
        if self.app is None:
            return
        self.app.console_mcp_tool_count = tool_count
        self.app.console_mcp_not_connected_count = not_connected_count

    def _console_tool_kill_switch_reader(self) -> Callable[[], bool] | None:
        """Return a fresh-per-call kill-switch reader, or ``None`` without a service.

        TASK-631. A callable rather than a bool so `build_tool_review_hook`
        observes a mid-run flip on the next batch; reading raises -> the
        hook fails CLOSED (refuses the turn), which is the only safe answer
        for a security control that cannot be read.

        Returns:
            A zero-arg callable returning the switch state, or ``None``
            when the app has no ``unified_mcp_service`` (nothing to honor).
        """
        service = getattr(self.app, "unified_mcp_service", None)
        if service is None:
            return None
        getter = getattr(service, "get_kill_switch", None)
        if not callable(getter):
            return None
        return lambda: bool(getter())

    async def _compose_mcp_provider(
        self,
        session_id: str | None = None,
        *,
        publish_counts: bool = True,
    ) -> MCPToolProvider | None:
        """Build + compose THIS run's MCPToolProvider on the running main loop.

        MUST be awaited from an async caller with the real Textual main
        loop running (``_run_agent_reply``, BEFORE its own
        ``asyncio.to_thread`` call) -- never from the agent bridge's
        worker thread. See ``MCPToolProvider``'s own module docstring:
        ``compose_catalog()`` performs async I/O
        (``local_external_catalog()``) that is documented to run on the
        main loop at registration time.

        TASK-632: returns the provider ALONE. It used to also build and
        return a per-run `build_mcp_review_hook` closure, but TASK-545's
        run-level `build_tool_review_hook` made that second element dead --
        the production call site unpacked it into `_unused_...` and threw it
        away, while the closure was still constructed on every run.

        Returns ``None`` whenever MCP tools should not be offered
        this run: no ``unified_mcp_service`` on the app, the kill switch
        is on, ``get_kill_switch``/``compose_catalog`` raised, or the
        composed catalog is empty (nothing to register, and -- since
        ``not_connected_count`` is only ever non-zero for servers that
        already contributed at least one eligible tool -- nothing an
        empty catalog could usefully report either). Live composition
        publishes this run's inspector counts on every return path via
        ``_publish_mcp_inspector_counts``. Disposable Context composition
        passes ``publish_counts=False`` and never mutates those app-level
        inspector values.

        Args:
            session_id: The run's OWNING session (Task 3/9) -- threaded
                into the composed provider's ``approval_callback`` (via a
                ``functools.partial`` binding, since ``MCPToolProvider``
                calls it with a fixed ``[pending]`` single-list arg) so a
                single-call fallback approval raised through
                ``invoke()``'s own gate parks/mounts and scopes its cancel
                check exactly like the batch review-hook path does.
                ``None`` (the default -- every pre-Task-9 call site) keeps
                ``request_mcp_approvals``' legacy no-session behavior.
            publish_counts: Whether to publish app-level MCP inspector counts.
                Live dispatch keeps the default; disposable preview disables it.

        Returns:
            A composed ``MCPToolProvider`` ready to hand to
            ``ConsoleAgentBridge.run_reply`` when eligible; ``None``
            otherwise.
        """

        def publish(tool_count: int | None, not_connected: int | None) -> None:
            if publish_counts:
                self._publish_mcp_inspector_counts(tool_count, not_connected)

        service = getattr(self.app, "unified_mcp_service", None)
        if service is None:
            publish(None, None)
            return None
        try:
            kill_switch = service.get_kill_switch()
        except Exception:  # noqa: BLE001 -- fail closed to "no MCP this run"
            logger.opt(exception=True).warning(
                "ConsoleChatController: get_kill_switch failed; skipping MCP this run"
            )
            publish(None, None)
            return None
        if kill_switch:
            publish(None, None)
            return None
        bound_request_approvals = functools.partial(
            self.request_mcp_approvals, session_id=session_id
        )
        provider = MCPToolProvider(
            service=service,
            main_loop=asyncio.get_running_loop(),
            approval_callback=bound_request_approvals,
            # task-1337 (plan Task 8): the Console always excludes the
            # shadowed built-in raw names, in BOTH retrieval modes -- the
            # run's own Library provider (direct or bounded-RAG) is the only
            # Library retrieval path Console agents get.
            builtin_raw_name_exclusions=CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
        )
        try:
            await provider.compose_catalog()
        except Exception:  # noqa: BLE001 -- a composition failure must not abort the send
            logger.opt(exception=True).warning(
                "ConsoleChatController: MCP compose_catalog failed; skipping MCP this run"
            )
            publish(None, None)
            return None
        catalog = provider.list_catalog()
        if not catalog:
            publish(None, None)
            return None
        publish(len(catalog), provider.not_connected_count)
        return provider

    async def _compose_agent_request_providers(
        self,
        *,
        session_id: str,
        project_selection: ProjectInstructionBindingSelection | None,
        project_authority_guard: Callable[[], bool] | None,
        turn_context: ConsoleTurnExecutionContext | None = None,
        publish_mcp_counts: bool = True,
    ) -> tuple[
        MCPToolProvider | None,
        "BuiltinToolGate",
        LocalToolProvider | None,
        Callable[[list["ToolCall"]], dict[str, str]] | None,
    ]:
        """Compose providers shared by live and preview.

        ``publish_mcp_counts`` is true for live dispatch and false for the
        disposable Context preflight, whose composition must not mutate the
        app-level inspector counters.
        """
        mcp_provider = await self._compose_mcp_provider(
            session_id, publish_counts=publish_mcp_counts
        )
        builtin_gate = build_builtin_gate(
            getattr(self.app, "unified_mcp_service", None)
        )
        local_provider, local_review_hook = self._compose_local_provider(
            session_id=session_id,
            turn_context=turn_context,
            project_root=(project_selection.root if project_selection else None),
            allow_write=(project_selection.allow_write if project_selection else True),
            project_root_identity=(
                project_selection.root_identity if project_selection else None
            ),
            project_root_guard=project_authority_guard,
        )
        return mcp_provider, builtin_gate, local_provider, local_review_hook

    def _compose_local_provider(
        self,
        session_id: str | None = None,
        turn_context: ConsoleTurnExecutionContext | None = None,
        *,
        project_root: Path | None = None,
        allow_write: bool = True,
        project_root_identity: tuple[tuple[str, int, int, int], ...] | None = None,
        project_root_guard: Callable[[], bool] | None = None,
    ) -> tuple[
        LocalToolProvider | None, Callable[[list["ToolCall"]], dict[str, str]] | None
    ]:
        """Build THIS run's LocalToolProvider + review hook, or ``(None, None)``.

        Sync, unlike ``_compose_mcp_provider``: ``LocalToolProvider``'s
        specs are static (no async catalog composition), so there is no
        main-loop I/O constraint -- but it is still called from
        ``_run_agent_reply`` alongside the MCP composition, before the
        bridge is dispatched onto ``asyncio.to_thread``.

        Returns ``(None, None)`` whenever local tools should not be
        offered this run:

        - ``[console] local_tools_enabled`` is false (the master flag);
        - no ``unified_mcp_service`` on the app -- ADR-032 reuses the MCP
          permission store under the synthetic ``local:__local__`` server
          key, so without the service there is no state source, no
          session-approval cache, and no persistence path;
        - the kill switch is on, or reading it raised (fail closed --
          mirrors ``_compose_mcp_provider``).

        Wiring (all straight MCP parity): ``resolve_state`` is the
        service's own ``gate_tool_test`` -- the exact payload source the
        MCP gate uses (fresh ``store.load()`` + ``resolve_effective_
        state`` per call, fail-closed to "ask" with no store);
        ``persist_approval`` routes ``approve_session`` to the in-memory
        session cache and ``always_allow`` to ``set_tool_state``, which
        fingerprints ``definition_hash(description, input_schema)``
        itself (the rug-pull guard, spec §3.2); ``record_decision`` is
        the same ``record_tool_decision`` audit path the MCP provider
        uses (``initiator="agent"``), recording local refusals as
        "denied"/"denied-timeout" under the ``local:__local__`` server
        key.

        Todo wiring (TASK-13216): when ``session_id`` resolves to a live
        session, the provider is handed that session's stable task store
        plus an ``on_todo_change`` callback that renders defensive task
        snapshots via ``ConsoleAgentBridge.append_todo_marker``. Without a
        session (or without a bridge), the provider stays context-free and
        no task specs are registered.

        Args:
            session_id: THIS run's owning session id, bound into the
                approval bridge exactly as ``_compose_mcp_provider`` and
                the built-in review hook bind it, so cancellation checks
                and card parking scope to this run's own session.

        Returns:
            ``(provider, review_tool_calls)`` when eligible -- a
            ``LocalToolProvider`` confined to the resolved workspace root
            and this run's ``build_local_review_hook``-built batch-review
            closure; ``(None, None)`` otherwise.
        """
        # task-3240 fix round 1 (Critical 2): get_cli_setting returns the RAW
        # TOML value -- a hand-typed quoted "false" is a non-empty string
        # and therefore truthy under bare `not` truthiness, which would
        # COMPOSE the entire local tool group while the MCP-hub gate
        # checkbox (Agents/builtin_tool_gate.py's all_tool_gates()) and
        # mcp_workbench.py's own [console] read both show it OFF. Note
        # this is NOT normalized by load_settings() the way some [console]
        # keys are -- that normalization lives in load_settings(), never
        # in get_cli_setting() itself, which reads the raw bootstrap
        # config. coerce_bool_setting is the arc's sixth such site.
        local_tools_enabled = (
            turn_context.tool_configuration.get(
                "local_tools_enabled", LOCAL_TOOLS_DEFAULT_ENABLED
            )
            if turn_context is not None
            else get_cli_setting(
                "console", "local_tools_enabled", LOCAL_TOOLS_DEFAULT_ENABLED
            )
        )
        if not coerce_bool_setting(local_tools_enabled, LOCAL_TOOLS_DEFAULT_ENABLED):
            return None, None
        service = getattr(self.app, "unified_mcp_service", None)
        if service is None:
            return None, None
        try:
            kill_switch = service.get_kill_switch()
        except Exception:  # noqa: BLE001 -- fail closed to "no local tools this run"
            logger.opt(exception=True).warning(
                "ConsoleChatController: get_kill_switch failed; skipping local tools this run"
            )
            return None, None
        if kill_switch:
            return None, None

        bound_request_approvals = functools.partial(
            self.request_mcp_approvals, session_id=session_id
        )

        def _kill_switch() -> bool:
            # invoke()-time read, mirroring MCPToolProvider._kill_switch_
            # engaged: never raises, and a read failure must not block
            # execution (compose-time read above already failed closed).
            try:
                return bool(service.get_kill_switch())
            except Exception as exc:  # noqa: BLE001 -- invoke must never raise
                logger.warning(
                    f"ConsoleChatController: get_kill_switch failed during local invoke: {exc}"
                )
                return False

        def _persist_approval(hub: "HubTool", decision: str) -> None:
            if decision == "approve_session":
                service.approve_for_session(hub.server_key, hub.name)
            elif decision == "always_allow":
                # set_tool_state computes definition_hash(hub.description,
                # hub.input_schema) itself -- required for the rug-pull guard.
                service.set_tool_state(hub.server_key, hub.name, "allow", tool=hub)

        def _record_decision(hub: "HubTool", decision: str) -> None:
            # Same audit path MCPToolProvider._record_decision_safe uses;
            # the provider guards the call (never-raise seam).
            service.record_tool_decision(
                hub.server_key,
                hub.name,
                decision=decision,
                initiator="agent",
            )

        if project_root is None:
            snapshot = turn_context.scratch_space if turn_context is not None else None
            if snapshot is None:
                return None, None
            root = snapshot.root
            authority_scope = functools.partial(self._scratch_spaces.lease, snapshot)
        else:
            root = project_root
            authority_scope = None
        subscriptions_db = getattr(self.app, "subscriptions_db", None)
        watchlists_service = WatchlistsToolService(
            db_resolver=lambda: subscriptions_db,
            # Owner-module loader (TASK-18609): constructing the store here
            # violated the runtime-policy ownership boundary.
            runtime_source_loader=load_default_runtime_source_state,
        )
        provider = LocalToolProvider(
            workspace_root=root,
            allow_write=allow_write,
            authority_scope=authority_scope,
            result_redaction_root=(root if project_root is None else None),
            root_guard=(
                project_root_guard
                if project_root_guard is not None
                else (
                    lambda: (
                        _project_root_identity_matches(root, project_root_identity)
                        if project_root_identity is not None
                        else True
                    )
                )
            ),
            resolve_state=service.gate_tool_test,
            kill_switch=_kill_switch,
            approval_callback=bound_request_approvals,
            is_session_approved=lambda hub: service.is_session_approved(
                hub.server_key, hub.name
            ),
            persist_approval=_persist_approval,
            record_decision=_record_decision,
            watchlists_service=watchlists_service,
            **self._todo_wiring(session_id),
        )
        return provider, build_local_review_hook(provider, bound_request_approvals)

    def _todo_wiring(self, session_id: str | None) -> _TodoWiring:
        """The todo_store/on_todo_change kwargs for ``_compose_local_provider``.

        Empty dict (no todo capability) when there is no session context,
        the session is unknown, or there is no bridge to render through.
        """
        if session_id is None:
            return {}
        bridge = self._agent_bridge
        if bridge is None:
            return {}
        session = next((s for s in self.store.sessions() if s.id == session_id), None)
        if session is None:
            return {}

        def _on_todo_change(tasks: list[dict[str, object]]) -> None:
            bridge.append_todo_marker(session_id, tasks)

        return {
            "todo_store": session.todo_store,
            "on_todo_change": _on_todo_change,
        }

    def _library_provider_for_context(
        self, turn_context: ConsoleTurnExecutionContext
    ) -> tuple[Any, Any] | None:
        """Build the provider and its run-owned authority from final context."""
        self._require_complete_turn_execution_context(turn_context)
        library_authority = turn_context.library_authority
        if (
            library_authority.policy.assistant_access
            is not ConsoleAssistantLibraryAccess.ALLOWED
        ):
            return None
        factory = self._library_provider_factory
        if factory is None:
            return None
        provider = factory(turn_context)
        if provider is None:
            return None

        from tldw_chatbook.Agents.library_rag_tool_provider import (
            LibraryRagToolProvider,
        )
        from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
        from tldw_chatbook.Agents.tool_catalog import LIBRARY_RESERVED_TOOL_NAMES

        expected_type = (
            LibraryToolProvider
            if library_authority.direct_library_tools
            else LibraryRagToolProvider
        )
        if type(provider) is not expected_type:
            return None
        authority = provider.issue_builtin_authority(
            reserved_names=LIBRARY_RESERVED_TOOL_NAMES,
            assistant_access=library_authority.policy.assistant_access,
        )
        return provider, authority

    def resolve_pending_approval(
        self, decisions: dict[str, str], *, round_id: str | None = None
    ) -> None:
        """UI THREAD: apply the user's batch decision, releasing the waiting worker thread.

        Called by ``ChatScreen``'s ``ChatApprovalCard.ApprovalDecided``
        handler, which forwards ``event.round_id`` -- the SAME id
        ``request_mcp_approvals`` stamped into the payload the card was
        built from (``ChatApprovalCard.set_batch`` stashes it;
        ``_submit_batch_decisions`` echoes it back on submit, mirroring
        ``resolve_pending_skill_script``'s identical ``request_id``
        round-trip).

        Fix round 1 (review CRITICAL finding): resolves ONLY the round
        whose id matches ``round_id`` -- never "whichever round belongs to
        the currently active session". ``ApprovalDecided`` travels as an
        async Textual message: a ``switch_session`` landing in the gap
        between the user's click and this handler running would otherwise
        let session A's decision resolve session B's completely different,
        unreviewed batch (or, for the same session, let a STALE decision
        from an already-ended round 1 resolve a newer round 2 that
        happened to arm before the stale message was delivered). A
        mismatched or stale ``round_id`` -- including one belonging to a
        round that already resolved and was popped -- is a safe no-op: the
        real round (if any) stays pending and its card re-derives
        unchanged on the next visit; nothing is ever auto-approved or
        denied-by-accident here.

        TASK-913 (AC#2): ``round_id=None`` no longer falls back to
        "whichever round belongs to the currently active session" -- it
        fails closed immediately, mirroring
        ``resolve_pending_skill_script``'s/``resolve_pending_skill_install``'s
        identical ``if request_id is None: return`` contract. Production
        (``ChatApprovalCard``/``ChatScreen``) has only ever had a single
        emitter (``ChatApprovalCard._submit_batch_decisions``) and it
        always threads the real ``round_id`` through; the active-session
        fallback existed only for legacy direct-call tests, which have
        been migrated to pass the real round id captured from the
        mounted/parked payload instead.

        A no-op both when ``round_id`` is ``None`` and when it doesn't
        match any currently-armed round (e.g. a stale message arriving
        after a timeout/cancellation already resolved and cleared it) --
        the real round (if any) stays pending and undecided; nothing is
        ever auto-approved or denied-by-accident here.

        NOTE: Snapshots the round's ``decisions``/``event`` into locals to
        avoid TOCTOU race: the worker thread's ``finally`` block pops the
        round entry out of ``_pending_approval_rounds`` concurrently. Guard
        and act only on the snapshots.

        Args:
            decisions: The user's per-``llm_name`` decision strings
                (``approve_once``/``approve_session``/``always_allow``/
                ``deny``) to merge into the round's shared decisions dict.
            round_id: The specific round to resolve (the id stamped onto
                the card the user actually decided). ``None`` (the
                default) never matches an armed round, so an un-migrated
                or malformed caller fails closed by omission.
        """
        # TASK-913 (AC#2): fail closed on a missing round_id rather than
        # scanning `_pending_approval_rounds.values()` for "whichever round
        # belongs to the active session" -- that active-session fallback
        # was production-unreachable (see docstring) and is now removed
        # entirely, taking its AC#1 lock-guarded-snapshot protection with
        # it (moot once the scan itself is gone). The remaining branch's
        # `.get()` read stays guarded: the worker thread's own registration
        # (`request_mcp_approvals`) and teardown (its `finally`) can mutate
        # this dict concurrently.
        if round_id is None:
            return
        with self._approval_state_lock:
            round_state = self._pending_approval_rounds.get(round_id)
        if round_state is None:
            return
        # Snapshot both at once to prevent TOCTOU race with worker thread's finally block
        decisions_dict = round_state["decisions"]
        approval_event = round_state["event"]
        decisions_dict.update(decisions or {})
        approval_event.set()

    def revoke_approval_rounds_for_run(self, run_id: str) -> int:
        """Fail every approval round owned by ``run_id`` closed, right now.

        PR2a Task 7 (safety). The approval wait blocks inside
        ``_call_with_timeout``'s per-call daemon thread, which keeps
        running after the fleet cooperatively cancels -- or outright
        ABANDONS -- the child that owns it. Until this existed, that
        child's card stayed on screen and stayed live: pressing Approve
        resolved the round, the waiting thread returned the approval, and
        the tool EXECUTED FOR REAL (a file written, a message sent) for a
        run whose handle and run row already read ``cancelled``. The
        documented ``approval_timeout < max_tool_call_seconds`` invariant
        (see ``_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS``) bounds the same
        class of hazard for the timeout path; this closes the
        cancellation path.

        Called by ``AgentService`` (through its injected
        ``revoke_approvals`` seam) at both moments a child stops being
        allowed to act: the cooperative cancel and the end-of-turn
        abandon. Safe to call for a run that never armed a card -- the
        common case -- and never touches another run's rounds, which
        matters because every child of a fleet turn shares ONE console
        session: session-keyed teardown could not tell a cancelled child's
        card from its live sibling's.

        Covers BOTH card registries a cancelled child can be holding:
        ``_pending_approval_rounds`` (tool-call approvals) and
        ``_pending_skill_script_rounds`` (run_skill_script confirms). The
        skill-script leg is the wider hazard of the two -- that tool is
        all-agents scope, its schema is not filtered by
        ``config.allowed_tools``, and ``console_agent_bridge``'s closure
        runs the script on the very next line after the confirm returns
        Allow, with no cancellation checkpoint in between. (Skill-INSTALL
        confirms are deliberately not swept: ``install_skill`` is wired
        for the primary agent only, so no sub-agent can arm one.)

        Each revoked round is (a) marked ``revoked`` so the waiting thread
        fails closed even if a click lands in its shared decision box
        afterwards, (b) pre-filled with the closed verdict, (c) removed
        from its registry, so a late ``resolve_pending_approval``/
        ``resolve_pending_skill_script`` finds nothing to resolve, (d)
        released via its Event, so the waiting thread returns immediately
        rather than at its auto-deny deadline, (e) discarded from
        ``_pending_approvals`` so the session's NEEDS_APPROVAL badge
        clears once its last round is gone, and (f) taken off screen
        through the SAME FIFO-head re-derive (``_remount_head``) that the
        round's own teardown uses, so a sibling round's card is never
        clobbered.

        Thread-safe. Each registry is swept under its own lock, and the
        two locks are taken SEQUENTIALLY, never nested. ``discard_pending_
        round`` and both UI clears take those (non-reentrant) locks
        themselves, so they are deliberately called after every critical
        section is released.

        Args:
            run_id: The cancelled/abandoned run whose cards must die. A
                falsy id is a no-op -- ``""`` is the "no run bound" key
                that rounds armed outside any agent run carry, and
                sweeping those would deny cards no run owns.

        Returns:
            How many rounds were revoked across both registries (``0``
            when the run had none).
        """
        if not run_id:
            return 0
        revoked = self._revoke_tool_approval_rounds(run_id)
        script_revoked = self._revoke_skill_script_rounds(run_id)
        for round_id, session_id in revoked:
            if session_id is not None:
                self.discard_pending_round(session_id, round_id)
            try:
                # PR0: re-derive from the session's remaining FIFO head --
                # revoking one round of several must promote the next, not
                # blank the session's card.
                self._remount_head(
                    self._parked_approval_payloads,
                    self.set_pending_approval,
                    session_id,
                )
            except Exception:  # noqa: BLE001 -- a UI remount must never
                # break the cancellation path that called us.
                logger.debug("Failed to marshal approval remount during revocation")
        for request_id, session_id in script_revoked:
            if session_id is not None:
                self.discard_pending_round(session_id, request_id)
            try:
                # PR0: re-derive from the session's remaining FIFO head --
                # revoking one round of several must promote the next, not
                # blank the session's card.
                self._remount_head(
                    self._parked_skill_script_payloads,
                    self.set_pending_skill_script,
                    session_id,
                )
            except Exception:  # noqa: BLE001 -- as above.
                logger.debug("Failed to clear skill-script confirm during revocation")
        total = len(revoked) + len(script_revoked)
        if total:
            logger.info("Revoked pending approval rounds for cancelled run")
        return total

    def _revoke_tool_approval_rounds(self, run_id: str) -> list[tuple[str, str | None]]:
        """Fail this run's tool-approval rounds closed. Registry work only.

        Args:
            run_id: The cancelled/abandoned run.

        Returns:
            ``(round_id, session_id)`` for each revoked round, for the
            caller's badge/card teardown (which must run outside the lock
            held here).
        """
        revoked: list[tuple[str, str | None]] = []
        with self._approval_state_lock:
            for round_id, state in list(self._pending_approval_rounds.items()):
                if state.get("run_id") != run_id:
                    continue
                state["revoked"] = True
                # Defense in depth only: the post-wait `revoked` guard in
                # `request_mcp_approvals` is the actual mechanism (it
                # ignores this box entirely). Filling it keeps the box
                # honest for anything that reads it directly.
                decisions = state.get("decisions")
                if isinstance(decisions, dict):
                    for name in state.get("names") or ():
                        decisions[name] = "deny"
                self._pending_approval_rounds.pop(round_id, None)
                session_id = state.get("session_id") or None
                revoked.append((round_id, session_id))
                event = state.get("event")
                if event is not None:
                    # Last, and only once the round is unreachable: the
                    # thread this releases returns the moment it wakes.
                    event.set()
        # PR0: mirrors `request_mcp_approvals`' `finally` exactly -- each
        # round drops exactly its OWN retained payload, so a still-armed
        # sibling keeps its own copy. The pre-PR0 "is this the last armed
        # round for the session" test existed only because the slot was
        # shared. Runs outside the critical section above because
        # `_unpark_round_payload` takes the (non-reentrant) same lock.
        for round_id_to_drop, _session_id in revoked:
            self._unpark_round_payload(self._parked_approval_payloads, round_id_to_drop)
        return revoked

    def _revoke_skill_script_rounds(self, run_id: str) -> list[tuple[str, str | None]]:
        """Fail this run's ``run_skill_script`` confirms closed.

        Registry work only, under ``_pending_skill_script_lock`` and then
        (sequentially, never nested) ``_approval_state_lock`` for the
        retained-payload slot.

        Args:
            run_id: The cancelled/abandoned run.

        Returns:
            ``(request_id, session_id)`` for each revoked confirm.
        """
        revoked: list[tuple[str, str | None]] = []
        with self._pending_skill_script_lock:
            for request_id, state in list(self._pending_skill_script_rounds.items()):
                if state.get("run_id") != run_id:
                    continue
                state["revoked"] = True
                # Defense in depth -- the post-wait `revoked` guard in
                # `request_skill_script_confirm` is what actually denies.
                decision = state.get("decision")
                if isinstance(decision, dict):
                    decision["allow"] = False
                    decision["remember"] = False
                self._pending_skill_script_rounds.pop(request_id, None)
                session_id = state.get("session_id") or None
                revoked.append((request_id, session_id))
                event = state.get("event")
                if event is not None:
                    event.set()
        # PR0: mirrors `request_skill_script_confirm`'s `finally` exactly --
        # each round drops exactly its OWN retained payload, so a
        # still-armed sibling keeps its own copy. The pre-PR0 "is this the
        # last armed round for the session" test existed only because the
        # slot was shared. Runs outside the critical section above because
        # `_unpark_round_payload` takes the (non-reentrant) same lock.
        for round_id_to_drop, _session_id in revoked:
            self._unpark_round_payload(
                self._parked_skill_script_payloads, round_id_to_drop
            )
        return revoked

    # -- Skill-install confirm bridge (task-5, parked TASK-910) --------------

    def request_skill_install_confirm(
        self, url: str, *, session_id: str | None = None
    ) -> bool:
        """WORKER THREAD: ask the user to confirm a skill install before any fetch.

        TASK-910: mirrors ``request_mcp_approvals``' park/mount/retain
        contract. Registers a fresh round (event + decision box + owning
        session id) under a freshly minted request id in
        ``_pending_skill_install_rounds`` (mirrors ``_pending_skill_script_
        rounds``' identical per-round design -- the pre-TASK-910 single
        ``_pending_skill_install_event``/``_pending_skill_install_decision``
        pair could not survive two DIFFERENT sessions each raising their own
        install confirm concurrently, exactly the hazard task-581 already
        fixed for skill-script). Either MOUNTS the card immediately
        (``session_id`` is the active/viewed session, or unknown -- legacy
        no-session callers keep the pre-TASK-910 always-mount behavior) or
        PARKS it (a different, background session -- the retained payload
        goes into ``_parked_skill_install_payloads`` for ``switch_session``/
        ``new_session``/``close_session`` to remount later, and
        ``park_pending_approval`` fires the SAME fleet badge + one-shot
        toast machinery ``request_mcp_approvals`` uses, per the train's
        toast-copy convention).

        Then polls re-checking this round's OWN cancel signal
        (``_is_session_cancelled``, scoped to ``session_id`` when known) and
        a deadline. Cancel/stop (of the OWNING session, or real process
        teardown via ``_shutdown_requested``), timeout, or no wired UI all
        resolve to DENY (fail-closed). A plain switch away no longer denies
        -- the round parks and stays alive until its own resolution,
        cancellation, or shutdown. Returns True only on an explicit Allow.

        Args:
            url: The skill source URL the model wants to install, surfaced
                verbatim on the confirm card for the user to inspect.
            session_id: The run's OWNING session (Task 3/9/TASK-910).
                ``None`` preserves the pre-Task-9 VIEWED-session/global-flag
                fallback (see ``_is_session_cancelled``) and never parks.

        Returns:
            True only on an explicit Allow; every other path (deny, cancel,
            stop, timeout, or no wired UI) returns False.
        """
        # No UI bridge wired means the marshal below is a no-op and nothing
        # can ever set the Event -- fail closed immediately instead of
        # blocking for the full timeout with no way to be resolved.
        if self.app is None or self.set_pending_skill_install is None:
            return False

        event = threading.Event()
        decision: dict[str, bool] = {}
        request_id = str(uuid4())
        owning_session_id = (
            session_id
            if session_id is not None
            else (self.store.active_session_id or "")
        )
        # PR3a-1 Task 6b (audit F4): arm-time cancel binding, identical to
        # `request_mcp_approvals`' -- see `_bind_round_cancel_signal`.
        round_cancel_event = self._bind_round_cancel_signal(session_id)
        # task-15860: the visit's teardown Event, captured at ARM time for
        # the same reason the run's cancel event is -- see
        # `_bind_visit_cancel_signal`.
        visit_cancel_event = self._bind_visit_cancel_signal()
        # ADR-067: same run stamp as `request_mcp_approvals` -- keys the
        # human-input wait mark below (the install confirm is primary-agent
        # only and in-loop, so no wrapper hosts it today, but the mark
        # keeps every human wait on one contract).
        owning_run_id = current_run_id()
        with self._pending_skill_install_lock:
            self._pending_skill_install_rounds[request_id] = {
                "event": event,
                "decision": decision,
                "session_id": owning_session_id,
            }

        timeout_seconds = (
            self.skill_install_confirm_timeout_seconds()
            if self.skill_install_confirm_timeout_seconds is not None
            else _DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS
        )
        # ADR-067: <= 0 arms NO deadline (the default) -- the round waits
        # for a decision or the owning run's cancellation.
        deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        payload = {
            "url": url,
            "timeout_seconds": timeout_seconds,
            "request_id": request_id,
            "session_id": owning_session_id,
            # Qodo PR #1836 finding 1 -- see `_head_round_payload`'s
            # remaining-time snapshot. None when ADR-067 armed no deadline.
            "deadline_monotonic": deadline,
        }
        # TASK-910: park rather than mount when this round's session is a
        # DIFFERENT, background session -- mirrors `request_mcp_approvals`'
        # identical `is_parked` gate. `session_id is None` (a legacy caller
        # with no session context) always mounts.
        is_parked = session_id is not None and session_id != (
            self.store.active_session_id or ""
        )
        # PR0: legacy `session_id is None` callers never park and never
        # queue -- they keep the unconditional mount below.
        is_head = True
        if session_id is not None:
            # TASK-1050 (Defect A): round-keyed, not a plain boolean -- see
            # `request_mcp_approvals`' identical `add_pending_round` call
            # for the full rationale (a sibling round from this bridge or
            # either of the other two must not have its badge stolen by
            # THIS round's own teardown).
            self.add_pending_round(session_id, request_id)
            # Retain THIS round's payload for EVERY session-attributed
            # round -- mounted or parked -- not just a parked one, mirroring
            # `request_mcp_approvals`' identical retention (Fix wave,
            # CRITICAL 1): a round that mounted immediately must still be
            # recoverable after a switch-away-and-back.
            # PR0: keyed by ROUND, and the return says whether THIS round
            # is its session's FIFO head. A non-head round must not mount:
            # an older sibling is still holding the card.
            is_head = self._park_round_payload(
                self._parked_skill_install_payloads, request_id, payload
            )
        try:
            if is_parked:
                if self.app is not None and self.park_pending_approval is not None:
                    self.app.call_from_thread(self.park_pending_approval, session_id)
            elif is_head:
                self._marshal_pending_skill_install(payload)
            # ADR-067: mark the owning run as waiting on a human decision
            # (see `request_mcp_approvals`' identical wrap for the why).
            with use_human_input_wait(owning_run_id):
                while not event.wait(_MCP_APPROVAL_POLL_SECONDS):
                    if self._is_session_cancelled(
                        session_id,
                        cancel_event=round_cancel_event,
                        visit_event=visit_cancel_event,
                    ):
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        break
            return bool(decision.get("allow", False))
        finally:
            with self._pending_skill_install_lock:
                self._pending_skill_install_rounds.pop(request_id, None)
            # PR0: drop exactly THIS round's retained payload. Pre-PR0 the
            # slot was shared per session, so the pop had to be guarded by
            # an order-dependent "is this the last armed round for the
            # session" test to avoid discarding a still-armed sibling's
            # only copy. Per-round storage makes that guard meaningless --
            # each round owns its own key.
            self._unpark_round_payload(self._parked_skill_install_payloads, request_id)
            if session_id is not None:
                # TASK-1050 (Defect A): discard ONLY this round's own id --
                # the badge clears only once every bridge round for this
                # session (this one included) has resolved.
                self.discard_pending_round(session_id, request_id)
            # PR0: re-derive the card from the session's remaining FIFO
            # head rather than deciding whether to CLEAR it. This
            # subsumes `_clear_pending_skill_install_if_round_is_current`'s
            # two-part TOCTOU guard -- see `request_mcp_approvals`'
            # identical `_remount_head` teardown call for the full race
            # analysis.
            # `owning_session_id`, not `session_id`: a legacy no-session
            # caller retains no payload, so its head resolves to `None` and
            # the card clears exactly as the pre-PR0 unconditional clear
            # did -- `_remount_head` no-ops on a `None` session_id, which
            # would otherwise leave a legacy caller's card stuck on screen
            # (caught live by `test_console_skill_install_confirm.py::
            # test_confirm_round_trip_allow`).
            try:
                # Qodo PR #1836 finding 2 -- see the approvals teardown's
                # identical legacy-round note.
                self._remount_head(
                    self._parked_skill_install_payloads,
                    self.set_pending_skill_install,
                    owning_session_id if session_id is not None else None,
                )
            except Exception:  # noqa: BLE001 -- suppress teardown-time errors
                logger.opt(exception=True).debug(
                    "Failed to marshal skill-install remount during teardown"
                )

    def _remount_parked_skill_install(self, session_id: str) -> None:
        """Re-derive the mounted skill-install confirm card for ``session_id``.

        TASK-910: called from `switch_session`/`new_session`/`close_session`
        exactly like the MCP approval card's own re-derive -- mounts
        ``session_id``'s retained payload (if any) and clears whatever the
        departing session had shown, all in one call. A no-op when no UI
        bridge is wired.

        Args:
            session_id: The session now being activated/viewed.
        """
        if self.set_pending_skill_install is None:
            return
        self.set_pending_skill_install(
            self._head_round_payload(self._parked_skill_install_payloads, session_id)
        )

    def _marshal_pending_skill_install(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a skill-install confirm payload to the UI thread.

        No-op when no UI bridge is wired (``self.app`` or
        ``set_pending_skill_install`` is None).

        Args:
            payload: The pending confirm's ``{"url", "timeout_seconds"}``
                dict to show, or None to clear/hide the card.
        """
        if self.app is not None and self.set_pending_skill_install is not None:
            self.app.call_from_thread(self.set_pending_skill_install, payload)

    def resolve_pending_skill_install(
        self, allow: bool, *, request_id: str | None = None
    ) -> None:
        """UI THREAD: apply the user's Allow/Deny, releasing the worker thread.

        TASK-910: strict match against ``request_id``, mirroring
        ``resolve_pending_skill_script``'s identical contract -- a resolve
        carrying no id, or an id belonging to any round other than the one
        it names, is silently dropped rather than resolved. This closes the
        same stale-late-click hazard ``resolve_pending_skill_script``'s own
        docstring documents: once two sessions can each have their own
        concurrent install-confirm round (TASK-910 parking), "whichever
        round happens to be active" is no longer a safe fallback the way it
        was pre-TASK-910 (a single global slot could only ever have one
        candidate).

        Args:
            allow: True to allow the pending install, False to deny it.
            request_id: The armed round's id, as echoed back by the UI
                (``SkillInstallConfirmCard.InstallDecided.request_id``).
                ``None`` (the default) never matches an armed round, so an
                un-migrated or malformed caller fails closed by omission.
        """
        if request_id is None:
            return
        with self._pending_skill_install_lock:
            round_state = self._pending_skill_install_rounds.get(request_id)
        if round_state is None:
            return
        round_state["decision"]["allow"] = bool(allow)
        round_state["event"].set()

    def pending_skill_install_ids(self) -> list[str]:
        """Return the request ids of every currently-armed install-confirm round.

        Mirrors ``pending_skill_script_ids`` -- exposed for tests and for
        any surface that needs to know whether a decision is outstanding.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending.
        """
        with self._pending_skill_install_lock:
            return list(self._pending_skill_install_rounds)

    # -- Skill-script confirm bridge -----------------------------------------

    def request_skill_script_confirm(
        self, payload: dict[str, Any], *, session_id: str | None = None
    ) -> dict[str, bool]:
        """WORKER THREAD: ask the user to confirm running a skill's script.

        Mirrors request_skill_install_confirm, but carries a two-part decision:
        allow this run, and whether to remember the choice for this skill.

        Each call arms a fresh round under a newly-generated request id
        (embedded in the payload handed to the UI as ``"request_id"``) so
        that ``resolve_pending_skill_script`` can reject a decision left
        over from a prior, already-torn-down round -- see that method's
        docstring for why this matters.

        TASK-910: also carries the SAME park/mount/retain contract as
        ``request_mcp_approvals``/``request_skill_install_confirm`` -- see
        ``request_skill_install_confirm``'s docstring for the full
        mount-vs-park/retain rationale, identical here. The per-round
        registry (keyed by ``request_id``, task-581) now also stores this
        round's owning session id, so teardown can distinguish "another
        round for a DIFFERENT session is still armed" (must not suppress
        clearing THIS session's card) from "another round for the SAME
        session is still armed" (must not clear it out from under that
        sibling round, preserving task-581's original guarantee).

        Args:
            payload: Confirm details to render ({"skill_name", "script_path",
                "mechanism", "args", ...}); "timeout_seconds" and
                "request_id" keys are added before marshaling to the UI.
            session_id: The run's OWNING session (Task 3/9/TASK-910), scoping
                the cancel check (``_is_session_cancelled`` -- PA-T9 finding
                #1) and the park/mount decision. ``None`` preserves the
                pre-Task-9 VIEWED-session/global-flag fallback and never
                parks.

        Returns:
            ``{"allow": bool, "remember": bool}``. Every non-Allow path (deny,
            cancel, stop, timeout, no wired UI) returns ``allow=False``.
        """
        if self.app is None or self.set_pending_skill_script is None:
            return {"allow": False, "remember": False}

        event = threading.Event()
        decision: dict[str, bool] = {}
        request_id = str(uuid4())
        owning_session_id = (
            session_id
            if session_id is not None
            else (self.store.active_session_id or "")
        )
        # PR3a-1 Task 6b (audit F4): arm-time cancel binding, identical to
        # `request_mcp_approvals`' -- see `_bind_round_cancel_signal`.
        round_cancel_event = self._bind_round_cancel_signal(session_id)
        # task-15860: the visit's teardown Event, captured at ARM time for
        # the same reason the run's cancel event is -- see
        # `_bind_visit_cancel_signal`.
        visit_cancel_event = self._bind_visit_cancel_signal()
        # PR2a Task 7 (review M1): same run-ownership stamp
        # `request_mcp_approvals` carries, and for a WIDER hazard --
        # `run_skill_script` is all-agents scope (no agent_kind gate in
        # `AgentService._run_one`) and runtime schemas are not filtered by
        # `config.allowed_tools`, so a fleet child is offered it
        # unconditionally; the bridge closure then runs the script as the
        # very next statement after this confirm returns Allow, with no
        # cancellation checkpoint in between. `current_run_id()` is bound
        # for the whole loop by `AgentService` (this tool is dispatched
        # IN-LOOP, not through `invoke_tool`'s per-call thread).
        script_round_state: dict[str, Any] = {
            "event": event,
            "decision": decision,
            "session_id": owning_session_id,
            "run_id": current_run_id(),
            # Re-read after the wait: a late Allow must not stick. See
            # `revoke_approval_rounds_for_run`.
            "revoked": False,
        }
        with self._pending_skill_script_lock:
            self._pending_skill_script_rounds[request_id] = script_round_state

        timeout_seconds = (
            self.skill_script_confirm_timeout_seconds()
            if self.skill_script_confirm_timeout_seconds is not None
            else _DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS
        )
        # ADR-067: <= 0 arms NO deadline (the default) -- the round waits
        # for a decision or the owning run's cancellation.
        deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        card_payload = dict(payload)
        card_payload["timeout_seconds"] = timeout_seconds
        card_payload["request_id"] = request_id
        card_payload["session_id"] = owning_session_id
        # Qodo PR #1836 finding 1 -- see `_head_round_payload`'s
        # remaining-time snapshot. None when ADR-067 armed no deadline.
        card_payload["deadline_monotonic"] = deadline
        is_parked = session_id is not None and session_id != (
            self.store.active_session_id or ""
        )
        # PR0: legacy `session_id is None` callers never park and never
        # queue -- they keep the unconditional mount below.
        is_head = True
        if session_id is not None:
            # TASK-1050 (Defect A): round-keyed, not a plain boolean -- see
            # `request_mcp_approvals`' identical `add_pending_round` call
            # for the full rationale.
            self.add_pending_round(session_id, request_id)
            # PR0: keyed by ROUND, and the return says whether THIS round
            # is its session's FIFO head. A non-head round must not mount:
            # an older sibling is still holding the card.
            is_head = self._park_round_payload(
                self._parked_skill_script_payloads, request_id, card_payload
            )
        try:
            if is_parked:
                if self.app is not None and self.park_pending_approval is not None:
                    self.app.call_from_thread(self.park_pending_approval, session_id)
            elif is_head:
                self._marshal_pending_skill_script(card_payload)
            # ADR-067: mark the owning run as waiting on a human decision
            # (see `request_mcp_approvals`' identical wrap for the why).
            with use_human_input_wait(str(script_round_state.get("run_id") or "")):
                while not event.wait(_MCP_APPROVAL_POLL_SECONDS):
                    if self._is_session_cancelled(
                        session_id,
                        cancel_event=round_cancel_event,
                        visit_event=visit_cancel_event,
                    ):
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        break
            # PR2a Task 7 (review M1): a revoked round denies
            # unconditionally, without consulting `decision` at all --
            # `resolve_pending_skill_script` writes into that shared box
            # after snapshotting the round, so an Allow delivered just
            # after the child was cancelled could otherwise reach the
            # `asyncio.run(scope.run_skill_script(...))` call the bridge
            # makes on the very next line. Mirrors `request_mcp_
            # approvals`' identical post-wait guard.
            with self._pending_skill_script_lock:
                was_revoked = bool(script_round_state.get("revoked"))
            if was_revoked:
                return {"allow": False, "remember": False}
            return {
                "allow": bool(decision.get("allow", False)),
                "remember": bool(decision.get("remember", False)),
            }
        finally:
            with self._pending_skill_script_lock:
                self._pending_skill_script_rounds.pop(request_id, None)
            # PR0: drop exactly THIS round's retained payload. Pre-PR0 the
            # slot was shared per session, so the pop had to be guarded by
            # an order-dependent "is this the last armed round for the
            # session" test to avoid discarding a still-armed sibling's
            # only copy. Per-round storage makes that guard meaningless --
            # each round owns its own key.
            self._unpark_round_payload(self._parked_skill_script_payloads, request_id)
            if session_id is not None:
                # TASK-1050 (Defect A): discard ONLY this round's own id --
                # the badge clears only once every bridge round for this
                # session (this one included) has resolved.
                self.discard_pending_round(session_id, request_id)
            # PR0: re-derive the card from the session's remaining FIFO
            # head rather than deciding whether to CLEAR it. This subsumes
            # the pre-PR0 `_clear_pending_skill_script_if_round_is_current`
            # guard's two-part TOCTOU check -- see `request_mcp_approvals`'
            # identical `_remount_head` teardown call for the full race
            # analysis.
            # Qodo PR #1836 finding 2: a legacy no-session round passes
            # None, so `_remount_head` re-derives for the session active
            # WHEN THE CALLBACK RUNS -- the arm-time `owning_session_id`
            # snapshot could mismatch after a switch and strand the
            # unconditionally-mounted legacy card. Session-attributed
            # rounds keep the exact-match owning id.
            try:
                self._remount_head(
                    self._parked_skill_script_payloads,
                    self.set_pending_skill_script,
                    owning_session_id if session_id is not None else None,
                )
            except Exception:  # noqa: BLE001 -- suppress teardown-time errors
                logger.opt(exception=True).debug(
                    "Failed to marshal skill-script remount during teardown"
                )

    def _remount_parked_skill_script(self, session_id: str) -> None:
        """Re-derive the mounted skill-script confirm card for ``session_id``.

        TASK-910: called from `switch_session`/`new_session`/`close_session`
        exactly like the MCP approval card's own re-derive -- mounts
        ``session_id``'s retained payload (if any) and clears whatever the
        departing session had shown, all in one call. A no-op when no UI
        bridge is wired.

        PR0: re-keyed by round, so this now re-derives the session's FIFO
        head instead of a single per-session slot. It already runs on the
        UI thread, so it calls `_head_round_payload` directly rather than
        `_remount_head`.

        Args:
            session_id: The session now being activated/viewed.
        """
        if self.set_pending_skill_script is None:
            return
        self.set_pending_skill_script(
            self._head_round_payload(self._parked_skill_script_payloads, session_id)
        )

    def _marshal_pending_skill_script(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a skill-script confirm payload to the UI thread.

        Args:
            payload: The pending confirm dict to show, or None to hide it.
        """
        if self.app is not None and self.set_pending_skill_script is not None:
            self.app.call_from_thread(self.set_pending_skill_script, payload)

    def resolve_pending_skill_script(
        self, allow: bool, remember: bool, request_id: str | None = None
    ) -> None:
        """UI THREAD: apply the user's decision, releasing the worker thread.

        ``request_id`` must be the exact ``"request_id"`` value the pending
        confirm's payload carried (``request_skill_script_confirm`` embeds
        a fresh one per round, and the confirm card built in a later task
        MUST echo it back here unchanged). This is a strict match: a
        resolve carrying no id, or an id from any round other than the one
        currently armed, is silently dropped rather than resolved.

        This guards against a real arbitrary-code-execution hazard: if
        round 1 ends (deadline, cancel, stop, conversation switch) and the
        agent immediately issues a second ``run_skill_script`` call
        arming round 2, a ``Button.Pressed`` queued for round 1 just
        before its teardown could otherwise be handled after round 2 is
        armed -- resolving round 2 (a script the user never saw) with
        round 1's stale click. Widget messages and ``call_from_thread``
        calls are separate queues, so ordering across a round boundary is
        not guaranteed.

        Args:
            allow: True to run the script this once.
            remember: True to also grant this skill standing permission.
            request_id: The armed round's id, as echoed back by the UI.
                ``None`` (the default) never matches an armed round, so an
                un-migrated or malformed caller fails closed by omission.
        """
        if request_id is None:
            return
        with self._pending_skill_script_lock:
            round_state = self._pending_skill_script_rounds.get(request_id)
        if round_state is None:
            return
        round_state["decision"]["allow"] = bool(allow)
        round_state["decision"]["remember"] = bool(remember)
        round_state["event"].set()

    def pending_skill_script_ids(self) -> list[str]:
        """Return the request ids of every currently-armed confirm round.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending. Exposed for tests and for any surface that needs to
            know whether a decision is outstanding.
        """
        with self._pending_skill_script_lock:
            return list(self._pending_skill_script_rounds)

    def stop_active_run(self, *, record_user_stop: bool = True) -> bool:
        """Request the ACTIVE (viewed) session's stream to stop at the next
        safe boundary.

        Task 3b requirement 2: name and public semantics are unchanged --
        this is the Stop button's contract, and it only ever targets
        whatever session ``self.store.active_session_id`` currently is,
        never a background run in another tab. A background run is
        completely unaffected by this call (its own entries in the
        per-session maps below are untouched); see ``shutdown`` for the
        teardown path that stops every session at once.

        Args:
            record_user_stop: Append the explicit "stopped by user"
                transcript record (TASK-337 AC3). ``shutdown`` passes
                ``False`` — a teardown stop is not a user action.

        Returns:
            True when the viewed session had an active run and it was
            stopped; False (a no-op) when it did not.
        """
        session_id = self.store.active_session_id or ""
        repair_session = self._active_citation_repair_sessions.get(session_id)
        if repair_session is not None and repair_session.selection_committed:
            return False
        if repair_session is not None and repair_session.phase in {
            "checking",
            "repair_streaming",
        }:
            if self._active_assistant_message_ids.get(session_id) is None:
                return False
            repair_session.cancel_reason = "user" if record_user_stop else "shutdown"
            self.prompt_queue_coordinator.pause_for_stop(session_id)
            self._signal_stop(session_id=session_id)
            task = self._active_stream_tasks.get(session_id)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
            return True

        if self.run_state.status is not ConsoleRunStatus.STREAMING:
            assistant_message_id = self._active_streaming_assistant_message_id()
            if assistant_message_id is None:
                return False
        else:
            assistant_message_id = (
                self._active_assistant_message_ids.get(session_id)
                or self._active_streaming_assistant_message_id()
            )
        if assistant_message_id is None:
            return False
        self.prompt_queue_coordinator.pause_for_stop(session_id)
        self._signal_stop(session_id=session_id)
        settlement_failed = False
        try:
            self._mark_stream_stopped(
                assistant_message_id,
                visible_copy="Response stopped.",
            )
        except ConsoleDispatchSettlementError:
            settlement_failed = True
            self._restore_dispatch_recovery_after_settlement_failure(
                session_id,
                assistant_message_id,
            )
        if record_user_stop and not settlement_failed:
            # TASK-337 AC3: a durable, explicit record — the run-state chip
            # copy is transient and the review found nothing else marked
            # the interruption.
            try:
                owner_id = self.store.session_id_for_message(assistant_message_id)
                self.store.append_message(
                    owner_id,
                    role=ConsoleMessageRole.SYSTEM,
                    content="Response stopped by user.",
                )
            except KeyError:
                pass
        task = self._active_stream_tasks.get(session_id)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        return True

    async def shutdown(self) -> None:
        """Stop and await EVERY session's active stream task before owner
        teardown.

        This is the supported clean lifecycle boundary. It must run on, or be
        coordinated with, every task's live owner loop before that loop closes.
        Under that ordering, shutdown signals cancellation, awaits all
        non-current submit and stream tasks to terminal state, and leaves no
        pending/destroyed-task or never-awaited-coroutine diagnostic.

        Task 3b requirement 3: unlike ``stop_active_run`` (deliberately
        scoped to the VIEWED session only), teardown is global across THIS
        controller instance's OWN sessions -- a background run must never
        survive this instance's shutdown just because the user was looking
        at a different tab. Mirrors ``stop_active_run``'s manual
        signal-then-cancel fallback for every session with a live entry,
        rather than reusing ``stop_active_run`` itself, which by contract
        only ever resolves the active session.

        Production caller: app-owned :class:`ConsoleRuntime` disposal at
        application exit. Ordinary Console navigation calls
        ``ConsoleRuntime.leave_console`` instead; that runtime and this
        controller survive unmount/remount and are reused by the next view.

        F5 fix (Qodo wave): sets ``_shutdown_requested`` unconditionally
        and FIRST -- before the no-tasks early return below -- so a
        worker-thread approval/confirm bridge polling on behalf of a run
        this method doesn't (yet) see in ``_active_stream_tasks`` still
        observes this instance's teardown. TASK-1052: this was true
        immediately only for a legacy ``session_id=None`` caller (whose
        fallback branch in ``_is_session_cancelled`` already OR'd in this
        flag); a round armed with a REAL ``session_id`` before its session
        reached ``_active_stream_tasks`` -- exactly the case this
        docstring describes -- previously still had to fall through to
        its own confirm/approval timeout, since the per-session
        ``_signal_stop`` fanout below only reaches sessions present in
        this method's ``tasks`` snapshot. ``_is_session_cancelled``'s
        real-``session_id`` branch now also ORs in ``_shutdown_requested``
        directly, closing that gap so this paragraph is accurate for
        every caller.

        Setting ``_shutdown_requested`` unconditionally is safe because this
        is the permanent app-disposal boundary. Navigation never calls it;
        ``leave_console`` supplies the reversible per-visit cancellation
        boundary for the same app-owned controller.
        """
        self.begin_shutdown()
        for message_id in tuple(self._original_attempts):
            self.clear_original_attempt(message_id)
        submit_tasks = self._submit_tasks_snapshot()
        stream_tasks = dict(self._active_stream_tasks)
        tasks = set(submit_tasks) | set(stream_tasks.values())
        if not tasks:
            self._retire_all_live_recovery_continuations()
            if self._owns_scratch_spaces:
                await asyncio.to_thread(self._scratch_spaces.dispose)
            return
        current = asyncio.current_task()
        session_ids = set(submit_tasks.values()) | set(stream_tasks)
        session_ids.discard("")
        for session_id in session_ids:
            # Dev's citation-repair feature threads a `cancel_reason`
            # ("user" vs "shutdown") through `ConsoleCitationRepairSession`
            # so `commit_canceled()` knows whether to append a "canceled by
            # user" system row (`stop_active_run` sets this for the VIEWED
            # session it targets) -- global teardown must set the same
            # field for EVERY session's own in-flight repair, or a
            # still-checking/repair-streaming session falls back to
            # whatever `cancel_reason` (if any) was already there.
            repair_session = self._active_citation_repair_sessions.get(session_id)
            if (
                repair_session is not None
                and not repair_session.selection_committed
                and repair_session.phase in {"checking", "repair_streaming"}
            ):
                repair_session.cancel_reason = "shutdown"
            self._signal_stop(session_id=session_id)
        for task in tasks:
            if task is not current:
                self._cancel_task_on_owner_loop(task)
        for task in tasks:
            if task is current:
                # Shutdown was invoked from within its own run's task --
                # cannot cancel/await itself; that run's own finally will
                # still fire once this coroutine naturally unwinds.
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Shutdown is a teardown path; stale task failures should not crash owner cleanup.
                pass
        self._stop_requested = False
        # Safety net: each task's own `finally` (in `_stream_assistant_
        # response`/`_run_agent_reply`) already pops ITS OWN session's
        # entries on the happy path -- this only catches a task that
        # somehow never reached that finally (e.g. a test double, or a
        # task that failed before it), so teardown never leaves a stale
        # entry behind for any session.
        for session_id, task in stream_tasks.items():
            if self._active_stream_tasks.get(session_id) is task:
                self._active_stream_tasks.pop(session_id, None)
                self._active_assistant_message_ids.pop(session_id, None)
                self._active_cancel_events.pop(session_id, None)
        for task in submit_tasks:
            self._unregister_submit_task(task)
        if current not in tasks:
            self._retire_all_live_recovery_continuations()
        if self._owns_scratch_spaces:
            await asyncio.to_thread(self._scratch_spaces.dispose)

    def begin_shutdown(self) -> None:
        """Synchronously fence work and marshal teardown to the owner loop.

        The PERMANENT form (app exit / `prepare_for_quit`). `_disposed`
        is what stops a later `begin_visit()` from handing this instance a
        fresh, unset cancellation Event.

        Callers should follow this fence with awaited :meth:`shutdown` while
        task owner loops are alive. If an owner loop is already closed, this
        method enters emergency fail-closed detachment: it synchronously drops
        controller/store volatile ownership, removes exclusively owned
        preparation sidecars, cannot dispatch, and never calls closed-loop
        scheduling/cancellation APIs. Public asyncio provides no way to make
        that abandoned pending Task terminal, so this emergency path cannot
        promise clean shutdown, recovery, durability, or suppression of
        Python's ``Task was destroyed but it is pending!`` diagnostic.
        """

        self._disposed = True
        self._visit_open = False
        self._fleet_wake.dispose()
        self._shutdown_requested.set()
        unreachable_preparations = self._detach_closed_submit_tasks()
        for preparation_id in unreachable_preparations:
            self._cleanup_unreachable_preparation(preparation_id)
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        with self._active_submit_tasks_lock:
            owner_loop = self._owner_loop
        if (
            owner_loop is not None
            and running_loop is not owner_loop
            and not owner_loop.is_closed()
        ):
            try:
                owner_loop.call_soon_threadsafe(
                    self._finish_begin_shutdown_from_scheduled_callback
                )
            except RuntimeError:
                self._cancel_headless_rounds()
            return
        if owner_loop is not None and owner_loop.is_closed():
            self._cancel_headless_rounds()
            return
        self._finish_begin_shutdown()

    def _finish_begin_shutdown_from_scheduled_callback(self) -> None:
        """Finish off-thread teardown without exposing callback exception text."""

        try:
            self._finish_begin_shutdown()
        except Exception:
            return

    def _finish_begin_shutdown(self) -> None:
        """Run queue/preparation/task teardown on the asyncio owner loop."""

        submit_tasks = self._submit_tasks_snapshot()
        stream_tasks = dict(self._active_stream_tasks)
        queue_failure: BaseException | None = None
        cleanup_failure: BaseException | None = None
        try:
            self.prompt_queue_coordinator.shutdown()
        except BaseException as exc:
            queue_failure = exc
        finally:
            for session in tuple(self.store.sessions()):
                preparation = self.store.preparation_for_session(session.id)
                if preparation is None or preparation.state not in {
                    ConsoleTurnPreparationState.PREPARING,
                    ConsoleTurnPreparationState.READY,
                    ConsoleTurnPreparationState.PAUSED,
                }:
                    continue
                try:
                    self._abandon_preparation(preparation.preparation_id)
                except BaseException as exc:
                    if cleanup_failure is None:
                        cleanup_failure = exc
            try:
                current = asyncio.current_task()
            except RuntimeError:
                current = None
            for task, session_id in submit_tasks.items():
                if session_id:
                    self._signal_stop(session_id=session_id)
                if task is not current:
                    self._cancel_task_on_owner_loop(task)
            for session_id, task in stream_tasks.items():
                self._signal_stop(session_id=session_id)
                if task is not current:
                    self._cancel_task_on_owner_loop(task)
            self._cancel_headless_rounds()
        if queue_failure is not None:
            raise queue_failure
        if cleanup_failure is not None:
            raise cleanup_failure

    def _cancel_headless_rounds(self) -> None:
        """Deny every round armed while no Console visit was open.

        task-15860 Task 5. Called by both teardown paths
        (``leave_console``, ``begin_shutdown``); a no-op when no detached
        round ever armed. The Event is DROPPED once set so the next
        detached round binds a fresh, unset one rather than inheriting a
        pre-set Event and denying instantly -- which would silently
        restore the exact 1.01s self-deny this task removed.
        """
        event = self._headless_visit_cancel
        if event is None:
            return
        self._headless_visit_cancel = None
        event.set()

    def begin_visit(self) -> None:
        """Open a new Console visit on a controller that survived the last.

        task-15860. Called by `ConsoleRuntime.attach_view` when a NEW view
        claims the runtime. Two things reset, and only two:

        1. A **fresh** `_shutdown_requested` Event. The previous visit's
           Event stays set forever, so every round that captured it at arm
           time stays denied (`_bind_visit_cancel_signal`), while this
           visit's sends, approvals and wake attempts start clean.
        2. Prompt-queue **admission re-opens**. `leave_console()`
           tombstones the visit's chains through the coordinator's
           `shutdown()`, which is a permanent `_shutting_down` latch --
           without this the queue would be dead for the rest of the app's
           life after the first navigation away.

        A disposed controller (app exit) is never re-opened.
        """

        if self._disposed:
            return
        self._shutdown_requested = threading.Event()
        # Qodo audit S2: state the visit lifecycle explicitly -- see
        # `_visit_open`'s own comment and `_bind_visit_cancel_signal`.
        self._visit_open = True
        self._stop_requested = False
        reopen = getattr(self.prompt_queue_coordinator, "reopen", None)
        if callable(reopen):
            reopen()

    async def leave_console(self) -> None:
        """End ONE Console visit. This controller SURVIVES it.

        The nav-away half of the teardown split (task-15860). Everything
        AC#2 names as screen-scoped still happens:

        - this visit's queue chains are tombstoned, before any
          cancellation, exactly as `begin_shutdown` did it;
        - this visit's cancellation Event is set, which denies every parked
          approval/confirm round armed during the visit (each captured the
          Event at arm time);
        - this visit's in-flight USER turns are signalled, cancelled and
          awaited, with `cancel_reason="shutdown"` stamped on each one's
          in-flight citation repair -- the same stamp `shutdown()` makes,
          for the same reason (`commit_canceled()` needs to know it was not
          the user who stopped it);
        - cross-turn fleet SURVIVORS keep running, untouched, as they
          already did.

        What does NOT happen: a wake is not blocked. task-15860's
        wake-fires-headless slice moved `ConsoleFleetWakeCoordinator.
        _attempt`'s gate onto `_disposed`, so a survivor settling after
        this call delivers a full wake turn with no Console mounted;
        only `begin_shutdown()` (app exit) refuses one.

        What does NOT happen, by owner ruling: an in-flight `AGENT_WAKE`
        turn is not cancelled. Cancelling it would re-create the exact
        "only completes if you stay" gap this arc exists to close, and a
        wake turn is structurally the same class of work as the survivor
        AC#2 keeps running. AC#2 names USER turns only.

        The provider gateway is NOT closed here -- it is app-owned now and
        a surviving turn still needs it. `ConsoleRuntime.dispose` closes it
        at exit.
        """

        # Tombstone first: `begin_shutdown`'s ordering contract ("before any
        # teardown cancellation"), unchanged.
        self.prompt_queue_coordinator.shutdown()
        self._visit_open = False
        self._shutdown_requested.set()
        # task-15860 Task 5: a round armed while DETACHED deferred to this
        # moment. The user has now had a Console visit in which to answer
        # it and has navigated away instead, so AC#2's rule applies to it
        # exactly as it does to a round armed during the visit.
        self._cancel_headless_rounds()
        for message_id in tuple(self._original_attempts):
            self.clear_original_attempt(message_id)
        wake_sessions = set(self._agent_wake_turn_sessions)
        tasks = {
            session_id: task
            for session_id, task in self._active_stream_tasks.items()
            if session_id not in wake_sessions
        }
        if not tasks:
            return
        current = asyncio.current_task()
        for session_id in tasks:
            repair_session = self._active_citation_repair_sessions.get(session_id)
            if (
                repair_session is not None
                and not repair_session.selection_committed
                and repair_session.phase in {"checking", "repair_streaming"}
            ):
                repair_session.cancel_reason = "shutdown"
            self._signal_stop(session_id=session_id)
        for task in tasks.values():
            if task is not current:
                task.cancel()
        for task in tasks.values():
            if task is current:
                # Left running from inside its own task, exactly as
                # `shutdown()` does: its own `finally` still fires.
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - teardown never crashes on a stale task
                pass
        self._stop_requested = False
        for session_id, task in tasks.items():
            if self._active_stream_tasks.get(session_id) is task:
                self._active_stream_tasks.pop(session_id, None)
                self._active_assistant_message_ids.pop(session_id, None)
                self._active_cancel_events.pop(session_id, None)

    def _active_streaming_assistant_message_id(self) -> str | None:
        """Return the visible streaming assistant message for the active session."""
        session_id = self.store.active_session_id
        if session_id is None:
            return None
        try:
            messages = self.store.messages_for_session(session_id)
        except KeyError:
            return None
        for message in reversed(messages):
            if (
                message.role is ConsoleMessageRole.ASSISTANT
                and message.status == "streaming"
            ):
                return message.id
        return None

    async def retry_message(
        self,
        message_id: str,
        *,
        queue_authorization: QueueGenerationAuthorization | None = None,
    ) -> ConsoleSubmitResult:
        """Retry a failed assistant message using the original turn context."""
        active_session_id = self.store.active_session_id
        if active_session_id is None and queue_authorization is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")
        message_session_id = self.store.session_id_for_message(message_id)
        queue_authorized = self.prompt_queue_coordinator.authorizes(
            queue_authorization, message_session_id
        )
        session_id = message_session_id if queue_authorized else active_session_id
        if session_id is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")
        active_rejection = self._active_run_rejection(
            session_id=session_id,
            queue_authorization=queue_authorization,
        )
        if active_rejection is not None:
            return active_rejection

        message = self.store.get_message(message_id)
        if message_session_id != session_id:
            visible_copy = "Open the original session before retrying this message."
            self._set_run_state(
                ConsoleRunState.blocked(visible_copy), session_id=session_id
            )
            return ConsoleSubmitResult(False, False, visible_copy)
        if message.status != "failed":
            return self._block(session_id, "Only failed messages can be retried.")

        self._set_run_state(
            ConsoleRunState.retrying("Retrying failed response."),
            session_id=session_id,
        )
        (
            resolution,
            turn_context,
        ) = await self._capture_and_resolve_turn_execution_context(session_id)
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
            return self._block(session_id, visible_copy)
        assert turn_context is not None
        thinking_block = self._thinking_persistence_preflight(
            session_id=session_id,
            resolution=resolution,
        )
        if thinking_block is not None:
            return thinking_block

        provider_messages = self._provider_messages_for_session(
            session_id,
            before_message_id=message_id,
            annotate_ids=True,
            turn_context=turn_context,
        )
        self._ensure_user_continuation_instruction(provider_messages)
        (
            provider_messages,
            refuse,
            skill_notes,
            skill_bindings,
            skill_bundle_block,
        ) = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session_id, refuse)
        for note in skill_notes:
            # An embedded skipped-skill note is never an abort: append the
            # same system-row copy `_block` would, then let the turn proceed.
            self.store.append_message(
                session_id, role=ConsoleMessageRole.SYSTEM, content=note
            )
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session_id
        )
        provider_messages = await self._apply_world_info(provider_messages, session_id)
        prefill = self._pinned_prefill_for_session(session_id)
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=message_id,
            prepare_retry=True,
            prefill=prefill,
            skill_bindings=skill_bindings,
            skill_bundle_block=skill_bundle_block,
            turn_context=turn_context,
        )

    async def resume_prompt_queue(self, session_id: str) -> PromptQueueMutationResult:
        """Resume next queued prompt after visibly reacquiring one agent slot."""

        return await self.prompt_queue_coordinator.resume_and_drain(session_id)

    def pause_prompt_queue_after_turn(
        self, session_id: str, *, expected_revision: int
    ) -> PromptQueueMutationResult:
        """Request a pause after the currently accepted turn settles."""

        return self.prompt_queue_coordinator.request_pause_after_turn(
            session_id, expected_revision=expected_revision
        )

    def keep_prompt_queue_draining(
        self, session_id: str, *, expected_revision: int
    ) -> PromptQueueMutationResult:
        """Cancel a pending pause-after-turn request."""

        return self.prompt_queue_coordinator.keep_draining(
            session_id, expected_revision=expected_revision
        )

    async def skip_and_resume_prompt_queue(
        self, session_id: str
    ) -> PromptQueueMutationResult:
        """Leave the failed/stopped turn visible and dispatch the next prompt."""

        return await self.prompt_queue_coordinator.resume_and_drain(session_id)

    async def retry_failed_queue_turn(
        self, message_id: str
    ) -> PromptQueueMutationResult:
        """Retry the failed turn under narrow queue authority, then drain."""

        session_id = self.store.session_id_for_message(message_id)
        return await self.prompt_queue_coordinator.recover_and_drain(
            session_id,
            lambda authorization: self.retry_message(
                message_id, queue_authorization=authorization
            ),
        )

    async def retry_stopped_queue_turn(
        self, message_id: str
    ) -> PromptQueueMutationResult:
        """Regenerate a stopped turn as a sibling, then drain on success."""

        session_id = self.store.session_id_for_message(message_id)
        return await self.prompt_queue_coordinator.recover_and_drain(
            session_id,
            lambda authorization: self.regenerate_message(
                message_id, queue_authorization=authorization
            ),
        )

    async def use_current_context_and_resume_prompt_queue(
        self,
        session_id: str,
        *,
        expected_revision: int,
        reviewed_context_epoch: int,
    ) -> PromptQueueMutationResult:
        """Adopt one explicitly reviewed context epoch and resume the queue."""

        return await self.prompt_queue_coordinator.use_current_context_and_resume(
            session_id,
            expected_revision=expected_revision,
            reviewed_context_epoch=reviewed_context_epoch,
        )

    async def continue_from_message(self, message_id: str) -> ConsoleSubmitResult:
        """Continue from a selected message by streaming a new assistant turn."""
        active_rejection = self._active_run_rejection()
        if active_rejection is not None:
            return active_rejection

        session_id = self.store.active_session_id
        if session_id is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")
        message_session_id = self.store.session_id_for_message(message_id)
        if message_session_id != session_id:
            visible_copy = (
                "Open the original session before continuing from this message."
            )
            self._set_run_state(
                ConsoleRunState.blocked(visible_copy), session_id=session_id
            )
            return ConsoleSubmitResult(False, False, visible_copy)

        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."),
            session_id=session_id,
        )
        (
            resolution,
            turn_context,
        ) = await self._capture_and_resolve_turn_execution_context(session_id)
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
            return self._block(session_id, visible_copy)
        assert turn_context is not None

        provider_messages = self._provider_messages_through_message(
            session_id,
            message_id,
            annotate_ids=True,
            turn_context=turn_context,
        )
        self._ensure_user_continuation_instruction(provider_messages)
        if not self._has_user_turn(provider_messages):
            return self._block(
                session_id,
                "Nothing to continue before the first message.",
            )
        (
            provider_messages,
            refuse,
            skill_notes,
            skill_bindings,
            skill_bundle_block,
        ) = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session_id, refuse)
        for note in skill_notes:
            # An embedded skipped-skill note is never an abort: append the
            # same system-row copy `_block` would, then let the turn proceed.
            self.store.append_message(
                session_id, role=ConsoleMessageRole.SYSTEM, content=note
            )
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session_id
        )
        provider_messages = await self._apply_world_info(provider_messages, session_id)
        assistant = self.store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=self.store.persistence is not None,
        )
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=assistant.id,
            skill_bindings=skill_bindings,
            skill_bundle_block=skill_bundle_block,
            turn_context=turn_context,
        )

    async def regenerate_message(
        self,
        message_id: str,
        *,
        queue_authorization: QueueGenerationAuthorization | None = None,
    ) -> ConsoleSubmitResult:
        """Regenerate a selected assistant message by forking a sibling branch.

        Unlike the pre-Task-6 behavior (streaming a replacement *variant*
        into the SAME message via ``variant_mode=True`` /
        ``begin_variant_stream``/``finalize_variant_stream``), this forks a
        new assistant node alongside ``message_id`` under its own parent
        (``store.create_sibling``) and streams into that NEW node normally
        (``variant_mode=False``). The anchor (and any old tail beneath it,
        for a mid-conversation regenerate) is left untouched and simply
        drops off the active path -- still reachable via
        ``store.set_active_leaf``, never deleted.

        All validation/blocking checks (provider readiness, "nothing to
        regenerate before the first message", a refusing skill) run BEFORE
        the sibling is created, mirroring the old mutate-only-once-committed
        discipline: a blocked regenerate must not leave a stray empty node
        forked into the tree. Because the fork shares the anchor's own
        parent, ``provider_messages`` computed with
        ``before_message_id=message_id`` (while ``message_id`` is still on
        the active path) is identical to computing it against the new
        sibling's id afterward -- both yield the anchor's ancestor chain --
        so it is safe to build once, up front.

        On stream FAILURE, the new sibling node itself becomes a ``failed``
        node on the active path (retryable via ``retry_message``), rather
        than restoring the anchor's prior reply in place -- this is the
        intended node-model behavior, not a regression: the anchor is a
        completely separate node and was never touched.
        """
        active_session_id = self.store.active_session_id
        if active_session_id is None and queue_authorization is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")
        message_session_id = self.store.session_id_for_message(message_id)
        queue_authorized = self.prompt_queue_coordinator.authorizes(
            queue_authorization, message_session_id
        )
        session_id = message_session_id if queue_authorized else active_session_id
        if session_id is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")
        active_rejection = self._active_run_rejection(
            session_id=session_id,
            queue_authorization=queue_authorization,
        )
        if active_rejection is not None:
            return active_rejection

        message = self.store.get_message(message_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            return self._block(
                session_id, "Only assistant messages can be regenerated."
            )
        if message_session_id != session_id:
            visible_copy = "Open the original session before regenerating this message."
            self._set_run_state(
                ConsoleRunState.blocked(visible_copy), session_id=session_id
            )
            return ConsoleSubmitResult(False, False, visible_copy)

        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."),
            session_id=session_id,
        )
        (
            resolution,
            turn_context,
        ) = await self._capture_and_resolve_turn_execution_context(session_id)
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
            return self._block(session_id, visible_copy)
        assert turn_context is not None
        thinking_block = self._thinking_persistence_preflight(
            session_id=session_id,
            resolution=resolution,
        )
        if thinking_block is not None:
            return thinking_block

        provider_messages = self._provider_messages_for_session(
            session_id,
            before_message_id=message_id,
            annotate_ids=True,
            turn_context=turn_context,
        )
        self._ensure_user_continuation_instruction(provider_messages)
        if not self._has_user_turn(provider_messages):
            return self._block(
                session_id,
                "Nothing to regenerate before the first message.",
            )
        (
            provider_messages,
            refuse,
            skill_notes,
            skill_bindings,
            skill_bundle_block,
        ) = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session_id, refuse)
        for note in skill_notes:
            # An embedded skipped-skill note is never an abort: append the
            # same system-row copy `_block` would, then let the turn proceed.
            self.store.append_message(
                session_id, role=ConsoleMessageRole.SYSTEM, content=note
            )
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session_id
        )
        provider_messages = await self._apply_world_info(provider_messages, session_id)
        prefill = self._pinned_prefill_for_session(session_id)
        self.clear_original_attempt(message_id)
        new_message = self.store.create_sibling(
            message_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=self.store.persistence is not None,
        )
        result = await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=new_message.id,
            variant_mode=False,
            prefill=prefill,
            skill_bindings=skill_bindings,
            skill_bundle_block=skill_bundle_block,
            turn_context=turn_context,
        )
        try:
            persisted_sibling = self.store.get_message(new_message.id)
        except KeyError:
            persisted_sibling = None
        replacement_event_id = (
            f"message:{persisted_sibling.persisted_message_id}"
            if persisted_sibling is not None
            and persisted_sibling.persisted_message_id is not None
            else None
        )
        replacement_status = (
            persisted_sibling.status if persisted_sibling is not None else "incomplete"
        )
        trace_status = (
            "completed"
            if replacement_event_id is not None and replacement_status == "complete"
            else replacement_status
            if replacement_event_id is not None
            and replacement_status in {"failed", "stopped"}
            else "incomplete"
        )
        self.store.record_trace_event(
            session_id,
            anchor_message_id=message_id,
            event_kind="message_regenerated",
            summary="Assistant response regenerated",
            status=trace_status,
            source_event_id=(
                f"message:{message.persisted_message_id}"
                if message.persisted_message_id is not None
                else None
            ),
            replacement_event_id=replacement_event_id,
            field_states={
                "replacement_event_id": (
                    "observed" if replacement_event_id is not None else "not_available"
                )
            },
        )
        return result

    #: Guidance cap for the transcript span fed to the summarizer (Task 3).
    #: Well above any realistic single-summary span so it never trims in tests
    #: or normal use; a runaway history drops its OLDEST turns before the call.
    _SUMMARY_SPAN_TOKEN_BUDGET = 12000

    async def summarize_up_to(self, message_id: str) -> ConsoleSubmitResult:
        """Summarize the active path up to (excluding) a USER message.

        Console `/rewind` "Summarize up to here" (SP2, Task 3). Runs the
        session's resolved provider (non-streaming) over the active-path turns
        before ``message_id`` and stores the result as the session's boundary
        summary (``store.set_session_context_summary``). The visible transcript
        is never mutated -- only the provider CONTEXT is later compacted at the
        dispatch choke point (see ``_apply_context_summary_compaction``).

        Gates run FIRST and NONE of them mutates transcript state (the Phase B
        discipline): an active run, a missing session, an off-path or non-USER
        target, a target with nothing before it, and provider-not-ready each
        return a blocked ``ConsoleSubmitResult`` via ``_summarize_block`` --
        which only sets the run state, never appends a system row. Rolling
        re-summarize (a prior boundary already on the path before ``message_id``)
        prepends the prior summary and only re-sends the turns SINCE that
        boundary. On an empty reply or a provider error the stored summary is
        left untouched.

        Args:
            message_id: Native id of the USER turn to summarize UP TO.

        Returns:
            ``ConsoleSubmitResult`` -- ``accepted`` True only when a non-empty
            summary was generated and stored.
        """
        active_rejection = self._active_run_rejection()
        if active_rejection is not None:
            return active_rejection

        session_id = self.store.active_session_id
        if session_id is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")

        if message_id not in self.store.active_path_message_ids(session_id):
            return self._summarize_block(
                session_id, "Switch to that branch before summarizing."
            )
        try:
            target = self.store.get_message(message_id)
        except KeyError:
            return self._summarize_block(
                session_id, "Switch to that branch before summarizing."
            )
        if target.role is not ConsoleMessageRole.USER:
            return self._summarize_block(
                session_id, "Only your own messages can be summarized up to here."
            )

        messages = self.store.messages_for_session(session_id)
        target_index = next(
            (i for i, m in enumerate(messages) if m.id == message_id), None
        )
        if target_index is None:
            return self._summarize_block(
                session_id, "Switch to that branch before summarizing."
            )
        before = [
            m
            for m in messages[:target_index]
            if m.role in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}
            and not _is_empty_transcript_row(m)
        ]
        if not before:
            return self._summarize_block(
                session_id, "Nothing to summarize before that message."
            )

        # Rolling compaction: when a prior boundary sits on this path BEFORE the
        # target, the prior summary already covers everything strictly before
        # it, so re-summarize only from that boundary (inclusive) forward and
        # fold the prior summary in.
        prev_summary, prev_boundary_id = self.store.session_context_summary(session_id)
        start_index = 0
        rolling_summary: str | None = None
        if prev_boundary_id is not None and prev_summary:
            prev_index = next(
                (i for i, m in enumerate(messages) if m.id == prev_boundary_id), None
            )
            if prev_index is not None and prev_index < target_index:
                start_index = prev_index
                rolling_summary = prev_summary
        span = [
            m
            for m in messages[start_index:target_index]
            if m.role in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}
            and not _is_empty_transcript_row(m)
        ]

        # "Summarizing..." run state, set the way regenerate sets VALIDATING.
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Summarizing conversation…"),
            session_id=session_id,
        )
        turn_context = self.resolve_turn_execution_context(session_id)
        resolution = await self._resolve_for_send_bounded(
            turn_context.provider_selection
        )
        if not getattr(resolution, "ready", False):
            return self._summarize_block(
                session_id,
                self._blocked_visible_copy(getattr(resolution, "visible_copy", "")),
            )

        span_text = self._build_summary_span_text(
            span, rolling_summary, model=getattr(resolution, "model", None) or ""
        )
        summarize_messages = [
            {
                "role": ConsoleMessageRole.SYSTEM.value,
                "content": get_internal_prompt("console.rewind_summarize"),
            },
            {"role": ConsoleMessageRole.USER.value, "content": span_text},
        ]
        try:
            summary_text = await self._collect_summary_completion(
                resolution, summarize_messages
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 -- failure = no-op + honest copy
            logger.opt(exception=True).warning(
                "Console summarize-up-to failed", error=str(error)
            )
            visible_copy = "Couldn't summarize the conversation. Try again."
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
                session_id=session_id,
            )
            return ConsoleSubmitResult(False, False, visible_copy)

        if not summary_text.strip():
            return self._summarize_block(
                session_id, "The model returned an empty summary."
            )

        self.store.set_session_context_summary(session_id, summary_text, message_id)
        turns = sum(1 for m in before if m.role is ConsoleMessageRole.USER)
        visible_copy = f"Summarized {turns} earlier turn{'s' if turns != 1 else ''}."
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, visible_copy),
            session_id=session_id,
        )
        return ConsoleSubmitResult(True, False, visible_copy)

    def _summarize_block(
        self, session_id: str, visible_copy: str
    ) -> ConsoleSubmitResult:
        """Blocked-summarize result that mutates NO transcript state.

        Unlike ``_block`` (which appends a SYSTEM row), a blocked summarize
        must leave the transcript untouched -- the run-state copy alone carries
        the reason to the control surfaces (Phase B discipline).
        """
        self._set_run_state(
            ConsoleRunState.blocked(visible_copy), session_id=session_id
        )
        return ConsoleSubmitResult(False, False, visible_copy)

    def _build_summary_span_text(
        self,
        span: list[ConsoleChatMessage],
        prior_summary: str | None,
        *,
        model: str,
    ) -> str:
        """Build the plain-text transcript span fed to the summarizer.

        Emits ``User: ...`` / ``Assistant: ...`` lines, prepending a
        ``[Previous summary]`` block when rolling. If the assembled span blows
        past ``_SUMMARY_SPAN_TOKEN_BUDGET`` (counted with
        ``count_console_messages_tokens``), the OLDEST turns are dropped until
        it fits -- the newest detail and the prior summary are always kept.
        """

        def assemble(rows: list[ConsoleChatMessage]) -> str:
            lines = [
                f"{'User' if m.role is ConsoleMessageRole.USER else 'Assistant'}: {m.content}"
                for m in rows
            ]
            transcript_text = "\n".join(lines)
            if prior_summary:
                return (
                    f"[Previous summary]\n{prior_summary}\n\n{transcript_text}".rstrip()
                )
            return transcript_text

        rows = list(span)
        body = assemble(rows)
        while (
            len(rows) > 1
            and count_console_messages_tokens(
                [{"role": "user", "content": body}], model
            )
            > self._SUMMARY_SPAN_TOKEN_BUDGET
        ):
            rows = rows[1:]
            body = assemble(rows)
        return body

    async def impersonate_user_reply(self, session_id: str) -> "ImpersonateResult":
        """Draft the USER's next message with the session's current model.

        task-1683: "Impersonate" writes a candidate reply *as the user*,
        for review in the composer -- it never sends and never appends to
        the transcript. Reuses the same resolve + collect path as
        ``summarize_up_to``.

        Qodo PR #1160: returns a reason alongside the text so the caller
        can say WHY nothing came back; a bare "" made "provider not
        ready" and "empty transcript" indistinguishable.

        Args:
            session_id: The session whose transcript and provider to use.

        Returns:
            An ``ImpersonateResult`` carrying the drafted text, or an empty
            text plus a machine-readable ``reason``.

        Raises:
            asyncio.CancelledError: Propagated unchanged when the caller's
                task is cancelled, so cancellation is never swallowed.
        """
        turn_context = self.resolve_turn_execution_context(session_id)
        resolution = await self._resolve_for_send_bounded(
            turn_context.provider_selection
        )
        if not getattr(resolution, "ready", False):
            return ImpersonateResult(
                "",
                "provider-not-ready",
                self._blocked_visible_copy(getattr(resolution, "visible_copy", "")),
            )
        session_messages = self.store.messages_for_session(session_id)
        # Mirror _provider_message_payloads' rules exactly (cubic PR #1160):
        # drop failed rows, and drop every ASSISTANT turn before the first
        # USER turn -- strict providers reject an assistant-first array
        # (task-427). A seeded character greeting therefore travels in the
        # system row instead, via _seeded_greeting_text (task-1531). Also
        # drop an empty-transcript row (task-2391) -- its content is a
        # placeholder written so the row could persist, not real user
        # words, and this prompt asks the model to write "in the user's
        # voice" from exactly this transcript, so a fabricated turn here is
        # if anything MORE dangerous than in the ordinary send path.
        transcript: list[dict[str, Any]] = []
        seen_user = False
        for message in session_messages:
            role = getattr(message, "role", None)
            if role not in (ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT):
                continue
            if getattr(message, "status", None) == "failed":
                continue
            if not seen_user and role is ConsoleMessageRole.ASSISTANT:
                continue
            if _is_empty_transcript_row(message):
                continue
            content = self._context_content_for(
                session_id,
                message,
                fallback=str(getattr(message, "content", "") or ""),
            ).strip()
            if not content:
                continue
            if role is ConsoleMessageRole.USER:
                seen_user = True
            transcript.append({"role": role.value, "content": content})
        if not transcript:
            return ImpersonateResult("", "empty-transcript", "")
        # Keep the newest turns within the same budget summarize uses, so a
        # long thread degrades by dropping OLD context rather than by
        # blowing the provider's window (cubic PR #1160).
        transcript = self._trim_transcript_to_budget(transcript)
        instruction = (
            "You are helping the USER write their next message in the "
            "conversation below. Reply with that message only -- their "
            "words, in their voice, no quotation marks, no narration, no "
            "preamble, and never as the assistant."
        )
        greeting = self._seeded_greeting_text(session_id, session_messages)
        if greeting:
            instruction = (
                f"{instruction}\n\nThe conversation opened with this "
                f"assistant message, which the user has seen:\n{greeting}"
            )
        messages = [
            {"role": ConsoleMessageRole.SYSTEM.value, "content": instruction},
            *transcript,
        ]
        # Providers that require a user-final array reject a request ending
        # on an assistant turn, which is the normal state after a completed
        # reply (cubic PR #1160).
        if messages[-1]["role"] != ConsoleMessageRole.USER.value:
            messages.append(
                {
                    "role": ConsoleMessageRole.USER.value,
                    "content": (
                        "Write my next message in this conversation. "
                        "Reply with that message only."
                    ),
                }
            )
        try:
            text = (
                await self._collect_summary_completion(resolution, messages)
            ).strip()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).warning("Impersonate completion failed.")
            return ImpersonateResult("", "provider-error", "")
        if not text:
            return ImpersonateResult("", "empty-completion", "")
        return ImpersonateResult(text, "", "")

    def _trim_transcript_to_budget(
        self, transcript: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Keep the newest turns within ``_SUMMARY_SPAN_TOKEN_BUDGET``.

        Args:
            transcript: Ordered provider rows, oldest first.

        Returns:
            The newest rows that fit the budget (at least the final row, so
            a single huge turn still produces a request).
        """
        kept: list[dict[str, Any]] = []
        total = 0
        for row in reversed(transcript):
            cost = max(1, len(str(row.get("content", ""))) // 4)
            if kept and total + cost > self._SUMMARY_SPAN_TOKEN_BUDGET:
                break
            kept.append(row)
            total += cost
        kept.reverse()
        # Never lead with an assistant row after trimming (task-427).
        while kept and kept[0]["role"] != ConsoleMessageRole.USER.value:
            kept.pop(0)
        return kept or transcript[-1:]

    async def _collect_summary_completion(
        self, resolution: Any, messages: list[dict[str, Any]]
    ) -> str:
        """Collect a NON-streaming completion via the gateway's streaming seam.

        The provider gateway protocol exposes only ``stream_chat``; there is no
        separate non-streaming completion method on the Console surface, so the
        summary is accumulated from its chunks WITHOUT appending to any
        transcript message (summarize never mutates the tree). Non-``str``
        yields (e.g. tool-call payloads, never requested here) are ignored.
        """
        chunks: list[str] = []
        async for chunk in self.provider_gateway.stream_chat(resolution, messages):
            if isinstance(chunk, str) and chunk:
                chunks.append(chunk)
        return "".join(chunks)

    async def edit_and_resend_message(
        self, message_id: str, new_content: str
    ) -> ConsoleSubmitResult:
        """Edit a USER message and resend it, forking a NEW sibling branch.

        Sibling counterpart to ``regenerate_message``, but the anchor is a
        USER message rather than an assistant one, and this creates TWO new
        nodes instead of one: a USER sibling of ``message_id`` (``store.
        create_sibling``, parented at the anchor's own parent, carrying the
        edited text) followed by an empty ASSISTANT node appended under it
        (``store.append_message``, which always parents at the current
        active leaf -- the freshly created sibling). The anchor
        (``message_id``) and any old tail beneath it (its prior assistant
        reply, and anything after it for a mid-conversation edit) are left
        untouched and simply drop off the active path -- still reachable via
        ``store.set_active_leaf``, never deleted.

        All validation/blocking checks (active run, message role/session
        ownership, non-blank content, provider readiness) AND every payload
        transform (skill substitution, chat dictionaries, world info) run
        BEFORE either new node is created, mirroring ``regenerate_message``'s
        "mutate last" discipline: a blocked or refused edit-and-resend must
        not leave a stray orphan sibling -- or an un-streamed, un-retryable
        ``"pending"`` assistant node -- forked into the tree. Unlike
        ``regenerate_message`` (whose anchor is still on the active path, so
        its payload can be read straight off the store), the edited text
        does not exist as a stored node yet, so ``provider_messages`` is
        built from the anchor's ancestors (``_provider_messages_for_session``
        with ``before_message_id=message_id``, which excludes the anchor and
        its subtree) plus a synthesized ``{"role": "user", "content":
        clean_content}`` dict standing in for the not-yet-created sibling.
        The transform pipeline operates purely on that ``list[dict]``
        payload and never needs the real nodes to exist, so a
        skill-substitution refusal aborts the turn via ``_block`` with
        nothing to clean up. Only once every transform has succeeded are
        ``new_user`` (``store.create_sibling``) and the empty ``assistant``
        node (``store.append_message``) actually created, and the stream is
        started against them.

        On stream FAILURE, the new assistant node becomes a ``failed`` node
        on the active path (retryable via ``retry_message``), rather than
        restoring the anchor's prior reply in place -- this is the intended
        node-model behavior, not a regression: the anchor is a completely
        separate node and was never touched.

        Args:
            message_id: Native id of the USER message being edited (the
                anchor whose ancestor chain -- read with
                ``before_message_id=message_id``, which excludes the anchor
                and its own subtree -- becomes the base for the new branch).
            new_content: The edited text to resend as the new sibling USER
                message.

        Returns:
            A ``ConsoleSubmitResult``. ``accepted`` is ``True`` once the new
            USER/ASSISTANT sibling pair has been created and streaming has
            started (whether the stream itself later completes or fails);
            ``False`` if any pre-mutation block gate (active run, message
            role, session ownership, off-active-path anchor, blank content,
            provider readiness, skill refusal) rejected the resend before
            either new node was created. ``visible_copy`` carries the
            block/refusal copy shown to the user when ``accepted`` is
            ``False`` (and the streamed/failure copy otherwise).
        """
        active_rejection = self._active_run_rejection()
        if active_rejection is not None:
            return active_rejection

        session_id = self.store.active_session_id
        if session_id is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")
        message = self.store.get_message(message_id)
        if message.role is not ConsoleMessageRole.USER:
            return self._block(
                session_id, "Only your messages can be edited and re-sent."
            )
        if self.store.session_id_for_message(message_id) != session_id:
            visible_copy = "Open the original session before editing this message."
            self._set_run_state(
                ConsoleRunState.blocked(visible_copy), session_id=session_id
            )
            return ConsoleSubmitResult(False, False, visible_copy)
        if message_id not in self.store.active_path_message_ids(session_id):
            # Task 2 review fix (Qodo finding 2): `_provider_messages_for_session`
            # builds the resend payload by scanning the ACTIVE-PATH transcript
            # until `message_id` is seen. If the anchor is not on the active
            # path, that scan never breaks and the payload would be built from
            # the wrong branch entirely. Edit is only exposed on active-path
            # rows today, so this is currently unreachable from the UI -- but
            # guard it here too so the method is safe to call directly.
            return self._block(
                session_id,
                "Switch to that branch before editing and re-sending this message.",
            )

        clean_content, validation_error = self._validated_draft(new_content)
        if validation_error is not None:
            return self._block(session_id, validation_error)
        configuration = self.resolve_turn_configuration_snapshot(session_id)
        turn_selection = configuration.provider_selection

        # task-573: the resend carries the anchor's attachments, so the same
        # vision gate a fresh send applies (see ``submit_draft``) must fire
        # here too -- BEFORE any node is created (mutate-last discipline).
        anchor_attachments = tuple(message.attachments)
        if any(a.data is not None for a in anchor_attachments):
            vision_model = configuration.effective_model
            block_reason = vision_block_reason(
                turn_selection.provider,
                vision_model,
                is_capable=lambda _provider, _model: bool(
                    configuration.capabilities.get("vision", False)
                ),
            )
            if block_reason is not None:
                return self._block(session_id, block_reason)

        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."),
            session_id=session_id,
        )
        (
            resolution,
            turn_context,
        ) = await self._capture_and_resolve_turn_execution_context(
            session_id,
            configuration,
        )
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
            return self._block(session_id, visible_copy)
        assert turn_context is not None

        # Build + transform the payload BEFORE creating either new node
        # (task-2 review fix): the edited turn is synthesized as a
        # not-yet-stored ``ConsoleChatMessage`` standing in for the sibling,
        # so a skill-substitution refusal (or any other transform failure)
        # has nothing to clean up -- no orphan sibling, no stuck "pending"
        # assistant node. task-573: running ancestors + the synthesized turn
        # through ONE ``_provider_message_payloads`` pass gives the carried
        # attachments the same image-budget/vision/mime treatment as a fresh
        # send (newest-first reservation included), instead of a hand-rolled
        # text-only dict.
        ancestors: list[ConsoleChatMessage] = []
        for candidate in self.store.messages_for_session(session_id):
            if candidate.id == message_id:
                break
            ancestors.append(candidate)
        ancestors.append(
            ConsoleChatMessage(
                role=ConsoleMessageRole.USER,
                content=clean_content,
                attachments=anchor_attachments,
            )
        )
        provider_messages = self._leading_system_message(
            session_id=session_id,
            turn_context=turn_context,
        ) + (
            self._provider_message_payloads(
                ancestors,
                skip_failed=True,
                annotate_ids=True,
                session_id=session_id,
                turn_context=turn_context,
            )
        )
        self._ensure_user_continuation_instruction(provider_messages)
        (
            provider_messages,
            refuse,
            skill_notes,
            skill_bindings,
            skill_bundle_block,
        ) = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session_id, refuse)
        for note in skill_notes:
            # An embedded skipped-skill note is never an abort: append the
            # same system-row copy `_block` would, then let the turn proceed.
            self.store.append_message(
                session_id, role=ConsoleMessageRole.SYSTEM, content=note
            )
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session_id
        )
        provider_messages = await self._apply_world_info(provider_messages, session_id)
        prefill = self._pinned_prefill_for_session(session_id)

        # Every transform succeeded: now (and only now) fork the edited USER
        # sibling and append the empty ASSISTANT node to stream into.
        active_path = self.store.active_path_message_ids(session_id)
        anchor_index = active_path.index(message_id)
        for replaced_message_id in active_path[anchor_index:]:
            self.clear_original_attempt(replaced_message_id)
        edited_message = self.store.create_sibling(
            message_id,
            role=ConsoleMessageRole.USER,
            content=clean_content,
            persist=self.store.persistence is not None,
            attachments=anchor_attachments,
        )
        self.store.record_trace_event(
            session_id,
            anchor_message_id=message_id,
            event_kind="message_edited",
            summary="Message edited and resent",
            status="completed",
            source_event_id=(
                f"message:{message.persisted_message_id}"
                if message.persisted_message_id is not None
                else None
            ),
            replacement_event_id=(
                f"message:{edited_message.persisted_message_id}"
                if edited_message.persisted_message_id is not None
                else None
            ),
        )
        assistant = self.store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=self.store.persistence is not None,
        )
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=assistant.id,
            variant_mode=False,
            prefill=prefill,
            skill_bindings=skill_bindings,
            skill_bundle_block=skill_bundle_block,
            turn_context=turn_context,
        )

    async def build_context_snapshot(
        self,
        draft: str,
        attachments: Iterable[MessageAttachment] | None = None,
        staged_sources: Iterable[ConsoleStagedSource] | None = None,
        *,
        session_id: str | None = None,
    ) -> ConsoleContextSnapshot:
        """Return a read-only snapshot of the current transcript and the assembled next-send payload.

        Skill rendering runs on disposable messages; live session state is untouched.

        Args:
            draft: The current composer draft text to include as a synthetic user turn.
            attachments: Pending attachments to include with the synthetic user turn.
            staged_sources: Staged workspace sources to include in the payload.

        Returns:
            A ``ConsoleContextSnapshot`` containing a deep-copied transcript and the
            redacted next-send provider payload. If assembly fails, the payload may
            contain an ``"error"`` key with a human-readable message.
        """
        session_id = session_id or self.store.active_session_id
        if not session_id:
            return ConsoleContextSnapshot(current_messages=[], next_send_payload={})

        session = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        turn_context = self.resolve_turn_execution_context(session_id)
        provider_selection = turn_context.provider_selection
        current_messages = list(self.store.messages_for_session(session_id))
        staged_sources_list = [
            {"source_id": s.source_id, "label": s.label, "type": s.source_type}
            for s in (staged_sources or ())
        ]

        provider_messages: list[dict[str, Any]] = []

        try:
            # Build the next-send payload as submit_draft would, but do not persist.
            # task-548: annotate native ids so the boundary-summary compaction
            # below can anchor by identity, exactly like the real dispatch path
            # (the keys are stripped again before the snapshot is returned).
            provider_messages = self._provider_messages_for_session(
                session_id,
                annotate_ids=True,
                turn_context=turn_context,
            )

            # Append a synthetic user turn for the draft so the preview matches what would be sent.
            attachment_tuple = tuple(attachments or ())
            synthetic_turn_added = bool(draft.strip() or attachment_tuple)
            if synthetic_turn_added:
                synthetic_user = self._provider_message_payloads(
                    [
                        ConsoleChatMessage(
                            role=ConsoleMessageRole.USER,
                            content=draft,
                            attachments=attachment_tuple,
                        )
                    ],
                    skip_failed=True,
                    session_id=session_id,
                    turn_context=turn_context,
                )
                provider_messages.extend(synthetic_user)

            prefill, prefill_from_one_shot = self._resolve_submit_prefill(session_id)
            dispatch_eligible = self._agent_dispatch_is_eligible(
                session, prefill=prefill
            )
            skill_bindings: tuple[str, ...] = ()
            skill_bundle_block = ""
            if dispatch_eligible:
                (
                    provider_messages,
                    skill_refusal,
                    _skill_notes,
                    skill_bindings,
                    skill_bundle_block,
                ) = await self._apply_skill_substitution(
                    copy.deepcopy(provider_messages)
                )
                if skill_refusal is not None:
                    dispatch_eligible = False
            else:
                # Historical turns have already been resolved at send time;
                # annotate only a newly synthesized bypassed turn.
                provider_messages = self._annotate_skill_commands(
                    provider_messages, synthetic_turn_added=synthetic_turn_added
                )

            # Chat dictionaries are safe to apply (string replacements only).
            provider_messages = await self._apply_chat_dictionaries(
                provider_messages, session_id
            )

            # task-548: mirror the dispatch choke point's boundary-summary
            # compaction so the preview matches what is actually sent when a
            # `/rewind` summary is active (pre-boundary turns replaced by the
            # summary folded into the leading system row). Applied after the
            # transforms, exactly like the send path; a payload without the
            # boundary row (or no stored summary) is untouched. The private
            # id-threading key is stripped immediately after, so it can never
            # appear in the preview rows.
            provider_messages = self._apply_context_summary_compaction(
                session_id, provider_messages
            )
            provider_messages = [
                {k: v for k, v in row.items() if k != NATIVE_MESSAGE_ID_KEY}
                for row in provider_messages
            ]

            # task-401: mirror the send path's response prefill exactly --
            # same resolution (one-shot wins over pinned) and same trailing
            # assistant turn -- WITHOUT consuming the one-shot (this is a
            # read-only preview). Placed after dictionaries to match
            # `_stream_assistant_response`'s ordering (dictionaries never
            # rewrite prefill text).
            if prefill:
                provider_messages = [
                    *provider_messages,
                    {
                        "role": ConsoleMessageRole.ASSISTANT.value,
                        "content": prefill,
                    },
                ]

            # Preserve the exact provider projection for disposable agent
            # admission.  Redaction and image placeholders are display-only:
            # applying either before AgentService's token admission can change
            # whether the project source fits compared with the live request.
            exact_provider_messages = copy.deepcopy(provider_messages)

            # Replace image data with placeholders for the preview, including historical images.
            provider_messages = self._replace_image_data_with_placeholders(
                provider_messages
            )

            # Gather native tool schemas and MCP note.
            tools_info = self._build_tools_info_for_snapshot()

            # Redact secrets before returning.
            redacted_messages = self._redact_secrets(provider_messages)
            # task-548: derive the duplicated `system` field from the payload's
            # own leading system row when present, so a folded boundary summary
            # shows there too (falling back to the bare session prompt when the
            # payload carries no system row).
            leading_system: list[dict[str, Any]] = (
                [provider_messages[0]]
                if provider_messages
                and provider_messages[0].get("role") == ConsoleMessageRole.SYSTEM.value
                else self._leading_system_message(
                    session_id=session_id, turn_context=turn_context
                )
            )
            redacted_system = self._redact_secrets(leading_system)

            # Deep-copy messages so the snapshot is independent of the store.
            copied_messages = self._presented_message_snapshots(
                session_id, current_messages
            )

            next_send_payload: dict[str, Any] = {
                "model": (
                    provider_selection.explicit_model
                    or provider_selection.configured_model
                ),
                "messages": redacted_messages,
                # `system` is intentionally duplicated from the leading system
                # message in `messages` so the preview viewer can show the
                # effective system prompt at a glance without scanning the
                # message list.  It is the same redacted value.
                "system": redacted_system,
                "staged_sources": staged_sources_list,
                "tools": tools_info,
            }
            if prefill:
                # Text mirrors the redacted trailing assistant turn so the
                # indicator can never leak what the messages list redacted.
                next_send_payload["response_prefill"] = {
                    "source": "one-shot" if prefill_from_one_shot else "pinned",
                    "text": redacted_messages[-1]["content"]
                    if redacted_messages
                    else prefill,
                    "agent_loop_bypassed": True,
                }
            preview = None
            if dispatch_eligible:
                preview = await self._build_project_instruction_preview_for_session(
                    session_id,
                    next_send_payload,
                    exact_provider_messages,
                    provider_selection=provider_selection,
                    turn_skill_bindings=skill_bindings,
                    turn_bundle_block=skill_bundle_block,
                )
            if preview is not None:
                next_send_payload = preview.next_send_payload
            return ConsoleContextSnapshot(
                current_messages=copied_messages,
                next_send_payload=next_send_payload,
                project_instruction_preview=preview,
            )
        except Exception as exc:
            logger.exception(
                "Failed to build context snapshot: session_id={session_id} "
                "draft_length={draft_length} attachments={attachments_count} "
                "staged_sources={staged_sources_count}",
                session_id=session_id,
                draft_length=len(draft),
                attachments_count=len(tuple(attachments or ())),
                staged_sources_count=len(tuple(staged_sources or ())),
            )
            # Preserve whatever was assembled before the failure so the viewer
            # still sees the transcript-derived payload and effective system
            # prompt rather than an empty placeholder. A failure inside the
            # annotate->strip window leaves the private id-threading key on the
            # assembled rows, so strip it here too (Qodo, PR #860).
            degraded_messages = self._replace_image_data_with_placeholders(
                self._redact_secrets(
                    [
                        {k: v for k, v in row.items() if k != NATIVE_MESSAGE_ID_KEY}
                        for row in provider_messages
                    ]
                )
            )
            degraded_system = self._redact_secrets(
                self._leading_system_message(
                    session_id=session_id, turn_context=turn_context
                )
            )
            return ConsoleContextSnapshot(
                current_messages=self._presented_message_snapshots(
                    session_id, current_messages
                ),
                next_send_payload={
                    "model": (
                        provider_selection.explicit_model
                        or provider_selection.configured_model
                    ),
                    "messages": degraded_messages,
                    "system": degraded_system,
                    "staged_sources": staged_sources_list,
                    "tools": {
                        "native_schemas": [],
                        "mcp_note": None,
                        "preview_note": "Preview unavailable due to an internal error.",
                    },
                    "error": f"Failed to build context snapshot: {exc}",
                },
            )

    async def _build_project_instruction_preview_for_session(
        self,
        session_id: str,
        base_payload: dict[str, Any],
        provider_messages: list[dict[str, Any]],
        *,
        provider_selection: ConsoleProviderSelection | None = None,
        turn_skill_bindings: tuple[str, ...] = (),
        turn_bundle_block: str = "",
    ) -> ProjectInstructionPreview | None:
        """Securely reread root guidance into a disposable preview only."""
        if not self._agent_runtime_enabled or self._agent_bridge is None:
            return None
        session = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        if (
            session is None
            or not session.project_instruction_state.project_instructions_enabled
        ):
            return None
        expected_control = session.project_instruction_state
        expected_session_snapshot = _ProjectInstructionAuthoritySnapshot(
            workspace_id=session.workspace_id,
            project_instruction_state=session.project_instruction_state,
        )
        owning_provider_selection = (
            provider_selection or self._provider_selection_for_session(session_id)
        )
        try:
            turn_context = self.resolve_turn_execution_context(session_id)
        except Exception:  # noqa: BLE001 - preview failure stays content-free
            return None
        scratch_snapshot = turn_context.scratch_space
        if scratch_snapshot is None:
            return None
        scratch_lease = functools.partial(
            self._scratch_spaces.lease,
            scratch_snapshot,
        )
        try:
            resolution = await self._resolve_for_send_bounded(
                owning_provider_selection
            )
        except Exception:  # noqa: BLE001 - preview failure stays content-free
            return None
        if not getattr(resolution, "ready", True):
            return None
        registry = getattr(self.app, "workspace_registry_service", None)
        try:
            selection = resolve_project_instruction_binding(session, registry)
            if selection is None:
                return None
            candidate = await asyncio.to_thread(
                ProjectInstructionResolver().resolve_startup,
                binding_id=selection.binding.binding_id,
                binding_root=selection.root,
                locator_fingerprint=selection.locator_fingerprint,
                max_bytes=coerce_int_setting(
                    get_cli_setting(
                        "console",
                        "project_instructions_startup_max_bytes",
                        DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                    ),
                    DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                    minimum=MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                    maximum=MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                ),
                dispatch_started_wall_ns=time.time_ns(),
            )
        except Exception:  # noqa: BLE001 - preview failure stays content-free
            return None
        bridge = self._agent_bridge
        if bridge is None:
            return None
        bound = bound_messages_to_window(
            copy.deepcopy(provider_messages),
            model=getattr(resolution, "model", None) or "",
            provider=getattr(resolution, "provider", "") or "",
            response_reservation=(
                getattr(resolution, "max_tokens", None) or DEFAULT_RESPONSE_RESERVATION
            ),
        )
        agent_messages = list(bound.messages)
        session_system_prompt = ""
        if (
            agent_messages
            and agent_messages[0].get("role") == ConsoleMessageRole.SYSTEM.value
        ):
            session_system_prompt = str(agent_messages[0].get("content", ""))
            agent_messages = agent_messages[1:]
        (
            mcp_provider,
            builtin_gate,
            local_provider,
            _local_review_hook,
        ) = await self._compose_agent_request_providers(
            session_id=session_id,
            project_selection=selection,
            project_authority_guard=None,
            turn_context=turn_context,
            publish_mcp_counts=False,
        )
        try:
            preview_result = await asyncio.to_thread(
                bridge.build_project_instruction_preview_request,
                candidate=candidate,
                session_id=session_id,
                resolution=resolution,
                fallback_model=(
                    owning_provider_selection.explicit_model
                    or owning_provider_selection.configured_model
                    or ""
                ),
                session_system_prompt=session_system_prompt,
                agent_messages=agent_messages,
                mcp_provider=mcp_provider,
                builtin_gate=builtin_gate,
                local_provider=local_provider,
                scratch_root=scratch_snapshot.root,
                scratch_lease=scratch_lease,
                turn_skill_bindings=turn_skill_bindings,
                turn_bundle_block=turn_bundle_block,
                request_skill_install_enabled=True,
                request_skill_script_enabled=(
                    self.set_pending_skill_script is not None
                ),
            )
        except Exception:  # noqa: BLE001 - preview failure stays content-free
            return None
        if preview_result is None:
            source = candidate.source
            code = "preview_uncertain_run_log_binding"
            return ProjectInstructionPreview(
                relative_source=source.relative_path if source else None,
                scope=source.scope if source else ".",
                byte_count=source.byte_count if source else 0,
                outcomes=(code,),
                warning_codes=(code,),
                next_send_payload=copy.deepcopy(base_payload),
            )
        exact_payload, snapshot = preview_result
        current_session = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        if (
            current_session is None
            or current_session.project_instruction_state != expected_control
        ):
            return None
        try:
            authority_current = await asyncio.to_thread(
                project_instruction_authority_snapshot_is_current,
                session_snapshot=expected_session_snapshot,
                registry=registry,
                expected_selection=selection,
            )
        except Exception:  # noqa: BLE001 - authority doubt fails closed
            return None
        if not authority_current:
            return None
        current_session = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        if (
            current_session is None
            or current_session.project_instruction_state != expected_control
        ):
            return None
        try:
            current_binding = registry.get_runtime_binding(
                str(selection.binding.binding_id)
            )
        except Exception:  # noqa: BLE001 - in-memory registry doubt fails closed
            return None
        if current_binding != selection.binding:
            return None
        if not _project_root_identity_matches(selection.root, selection.root_identity):
            return None
        payload = copy.deepcopy(base_payload)
        payload.pop("tools", None)
        display_payload = self._replace_image_data_with_placeholders(
            copy.deepcopy(exact_payload.get("messages") or ())
        )
        projected_exact = copy.deepcopy(exact_payload)
        projected_exact["messages"] = self._redact_secrets(display_payload)
        payload.update(projected_exact)
        messages = list(payload.get("messages") or ())
        payload["system"] = (
            [copy.deepcopy(messages[0])]
            if messages and messages[0].get("role") == ConsoleMessageRole.SYSTEM.value
            else []
        )
        source = snapshot.startup_source_metadata or snapshot.startup_source
        outcomes = tuple(outcome.code for outcome in snapshot.primary_delivery.outcomes)
        return ProjectInstructionPreview(
            relative_source=source.relative_path if source else None,
            scope=source.scope if source else ".",
            byte_count=source.byte_count if source else 0,
            outcomes=outcomes,
            warning_codes=tuple(snapshot.warning_codes),
            next_send_payload=payload,
        )

    def _agent_dispatch_is_eligible(self, session, *, prefill: str | None) -> bool:
        """Return the single live/preview agent-dispatch eligibility decision."""
        return bool(
            self._agent_runtime_enabled
            and self._agent_bridge is not None
            and not prefill
            and session is not None
            and session.assistant_kind != "character"
        )

    def _remember_project_instruction_delivery(
        self, session_id: str, snapshot: InstructionSnapshot
    ) -> None:
        """Retain only content-free metadata from a final primary delivery."""
        self._project_instruction_activation_events.pop(session_id, None)
        source = snapshot.startup_source_metadata
        outcomes = tuple(outcome.code for outcome in snapshot.primary_delivery.outcomes)
        delivered = bool(
            snapshot.startup_source is not None
            and snapshot.startup_source.digest
            in snapshot.primary_delivery.source_digests
        )
        self._project_instruction_display[session_id] = (
            ProjectInstructionDisplayMetadata(
                binding_id=snapshot.binding_id,
                locator_fingerprint=snapshot.locator_fingerprint,
                relative_source=source.relative_path if source else None,
                scope=source.scope if source else ".",
                byte_count=source.byte_count if source else 0,
                outcome="active"
                if delivered
                else (outcomes[0] if outcomes else "none"),
                warning_codes=tuple(snapshot.warning_codes),
            )
        )

    def _clear_project_instruction_delivery(self, session_id: str) -> None:
        self._project_instruction_display.pop(session_id, None)
        self._project_instruction_activation_events.pop(session_id, None)

    def _record_project_instruction_activation(
        self, session_id: str, event: ProjectInstructionActivationEvent
    ) -> None:
        """Keep one run's content-free activation notices in session memory."""
        self._project_instruction_activation_events.setdefault(session_id, []).append(
            event
        )

    def project_instruction_activation_events(
        self, session_id: str
    ) -> tuple[ProjectInstructionActivationEvent, ...]:
        """Return content-free activation notices for one live session."""
        return tuple(self._project_instruction_activation_events.get(session_id, ()))

    def project_instruction_display_metadata(
        self, session_id: str
    ) -> ProjectInstructionDisplayMetadata | None:
        """Return current content-free final-delivery metadata for one session."""
        metadata = self._project_instruction_display.get(session_id)
        if metadata is None:
            return None
        session = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        state = session.project_instruction_state if session is not None else None
        if (
            state is None
            or not state.project_instructions_enabled
            or state.working_folder_binding_id != metadata.binding_id
            or state.working_folder_locator_fingerprint != metadata.locator_fingerprint
        ):
            self._clear_project_instruction_delivery(session_id)
            return None
        return metadata

    @staticmethod
    def _replace_image_data_with_placeholders(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result = copy.deepcopy(messages)

        def _is_data_url(value: Any) -> bool:
            return isinstance(value, str) and value.startswith("data:")

        def _redact_image_url_value(value: Any) -> Any:
            """Redact an image URL value while preserving its original shape."""
            if isinstance(value, dict) and _is_data_url(value.get("url")):
                return {**value, "url": "[image: data redacted for preview]"}
            if isinstance(value, str) and _is_data_url(value):
                return "[image: data redacted for preview]"
            return value

        def _redact_image_source(source: dict[str, Any]) -> dict[str, Any]:
            """Redact base64 or data-URL content inside an image source dict."""
            if not isinstance(source, dict):
                return source
            redacted = {**source}
            if _is_data_url(redacted.get("data")) or redacted.get("type") == "base64":
                redacted["data"] = "[image: data redacted for preview]"
            if _is_data_url(redacted.get("url")):
                redacted["url"] = "[image: data redacted for preview]"
            return redacted

        for message in result:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "image_url":
                        part["image_url"] = _redact_image_url_value(
                            part.get("image_url")
                        )
                    if part.get("type") == "image":
                        # Anthropic-style image parts use a ``source`` dict with
                        # base64 data; preserve the surrounding structure.
                        if isinstance(part.get("source"), dict):
                            part["source"] = _redact_image_source(part["source"])
                        if "image" in part:
                            part["image"] = _redact_image_url_value(part["image"])
            elif isinstance(content, str):
                # Some providers may inline image data URLs directly in a string
                # content body; redact them so they never leak into the preview.
                message["content"] = re.sub(
                    r"data:[^\s\"'<>]+",
                    "[image: data redacted for preview]",
                    content,
                )
        return result

    @staticmethod
    def _annotate_skill_commands(
        messages: list[dict[str, Any]],
        *,
        synthetic_turn_added: bool = True,
    ) -> list[dict[str, Any]]:
        """Flag a draft that LOOKS like an unresolved leading `$name` skill mention.

        Cheap textual heuristic only (a leading `MENTION_SIGIL`) -- this
        preview path deliberately never calls `_apply_skill_substitution`
        (see the caller's comment), so it has no candidate snapshot to
        actually resolve the word against. Re-sigiled for the `$`-mention
        migration (Task 5): a leading ``/`` is now a registered slash
        command (``/skills``, ``/prompt``, ...), not a skill invocation, so
        it must NOT be annotated here. Embedded ``$name`` mentions
        elsewhere in the draft are intentionally not flagged -- this only
        covers the leading form, mirroring `_apply_skill_substitution`'s
        own "leading form tried first" precedence.

        Only STRING content is ever annotated. A multimodal (list-content)
        draft -- e.g. a text part plus an image attachment -- is left
        completely unchanged, even when its text part starts with a
        `$name` mention: `_apply_skill_substitution` early-returns on
        non-str content at send time (replacing list content outright would
        drop the attachments), so this preview never actually substitutes a
        multimodal draft's skill mention. Annotating it here would promise
        a substitution the send never performs -- a dishonest preview
        (Qodo fix 4, PR #801 review).
        """
        result = copy.deepcopy(messages)
        if not synthetic_turn_added or not result or result[-1].get("role") != "user":
            return result

        content = result[-1].get("content", "")
        annotation = (
            "[Skill command not resolved in preview; "
            "actual substitution happens at send time.]"
        )

        if isinstance(content, str) and content.lstrip().startswith(MENTION_SIGIL):
            result[-1]["content"] = f"{content}\n\n{annotation}"

        return result

    def _build_tools_info_for_snapshot(self) -> dict[str, Any]:
        """Return native tool schemas and preview notes for the snapshot."""
        tools: list[dict[str, Any]] = []
        if self._agent_bridge is not None:
            # Native tools only; live MCP catalog composition is out of scope.
            tools = self._agent_bridge.native_tool_schemas()
        mcp_note: str | None = None
        if self._mcp_provider:
            mcp_note = "MCP tools are configured but live catalog composition is not shown in this preview."
        if tools:
            preview_note = (
                "This preview shows only builtin native tools. "
                "The live run may add skills/MCP tools."
            )
        else:
            preview_note = "No native tools are configured for preview."
        return {
            "native_schemas": tools,
            "mcp_note": mcp_note,
            "preview_note": preview_note,
        }

    _SECRET_REDACTION_KEYS = {
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
        "bearer",
    }
    _SECRET_REDACTION_KEYS_NORMALIZED = {
        k.replace("-", "").replace("_", "") for k in _SECRET_REDACTION_KEYS
    }
    _SECRET_REDACTION_PATTERN = re.compile(
        r"(?P<open_quote>[\"']?)"
        r"(?P<key>" + "|".join(re.escape(k) for k in _SECRET_REDACTION_KEYS) + r")"
        r"(?P=open_quote)"
        r"(?P<sep>\s*[:=]\s*)"
        r"(?P<value>"
        + r'"(?:\\.|[^"\\])*"'
        + r"|'(?:\\.|[^'\\])*'"
        + r"|[^\s,;}\]\)\"']+"
        + r")",
        re.IGNORECASE,
    )

    @staticmethod
    def _redact_secrets(payload: Any) -> Any:
        """Return a deep-copied payload with likely secret values replaced.

        Redaction is best-effort and intended for preview/export convenience
        only. Do not rely on it for security-sensitive export or disclosure
        scenarios.
        """
        redacted = copy.deepcopy(payload)

        def _redact_string(value: str) -> str:
            def _replace_value(match: re.Match[str]) -> str:
                matched_value = match.group("value")
                if matched_value.startswith('"'):
                    redacted_value = '"[redacted]"'
                elif matched_value.startswith("'"):
                    redacted_value = "'[redacted]'"
                else:
                    redacted_value = "[redacted]"
                open_quote = match.group("open_quote")
                key = match.group("key")
                sep = match.group("sep")
                return f"{open_quote}{key}{open_quote}{sep}{redacted_value}"

            return ConsoleChatController._SECRET_REDACTION_PATTERN.sub(
                _replace_value, value
            )

        def _matches_secret_key(key: str) -> bool:
            """Return True when ``key`` matches or ends with a secret word.

            Matches exact keys such as ``api_key``, suffixed keys such as
            ``my_api_key``, and hyphenated/camelCase variants such as
            ``x-api-key`` or ``apiKey``.
            """
            lowered = key.lower()
            normalized = lowered.replace("-", "").replace("_", "")
            if normalized in ConsoleChatController._SECRET_REDACTION_KEYS_NORMALIZED:
                return True
            for secret in ConsoleChatController._SECRET_REDACTION_KEYS:
                if lowered.endswith(f"_{secret}"):
                    return True
                normalized_secret = secret.replace("-", "").replace("_", "")
                if normalized.endswith(normalized_secret):
                    return True
            return False

        def _redact_obj(obj: Any, under_secret: bool = False) -> Any:
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    key_is_secret = _matches_secret_key(key)
                    if key_is_secret and isinstance(value, str):
                        result[key] = "[redacted]"
                    elif key_is_secret:
                        # Structured values under a secret key are recursively
                        # redacted so nested strings do not leak.
                        result[key] = _redact_obj(value, under_secret=True)
                    elif under_secret and isinstance(value, str):
                        result[key] = "[redacted]"
                    else:
                        result[key] = _redact_obj(value, under_secret=under_secret)
                return result
            if isinstance(obj, list):
                return [_redact_obj(item, under_secret=under_secret) for item in obj]
            if isinstance(obj, str):
                if under_secret:
                    return "[redacted]"
                return _redact_string(obj)
            return obj

        return _redact_obj(redacted)

    def _provider_selection(self) -> ConsoleProviderSelection:
        return ConsoleProviderSelection(
            provider=self.provider,
            base_url=self.base_url,
            explicit_model=self.model,
            configured_model=self.configured_model,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            seed=self.seed,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            verbosity=self.verbosity,
            thinking_effort=self.thinking_effort,
            thinking_budget_tokens=self.thinking_budget_tokens,
            streaming=self.streaming,
            system_prompt=self.system_prompt,
            workspace_context=self.store.workspace_context,
        )

    def _provider_selection_for_session(
        self, session_id: str
    ) -> ConsoleProviderSelection:
        """Resolve provider inputs from the owning session, never the viewed tab."""
        settings = self.store.session_settings(session_id)
        workspace_id = self.store.session_workspace_id(session_id)
        current_workspace = self.store.workspace_context
        workspace_context = (
            current_workspace
            if current_workspace.active_workspace_id == workspace_id
            else ConsoleWorkspaceContext(active_workspace_id=workspace_id)
        )
        if settings is None:
            return replace(
                self._provider_selection(),
                system_prompt=self._resolved_system_prompt(session_id),
                workspace_context=workspace_context,
            )

        app_config = self._provider_config() if self._provider_config else {}
        selection = build_console_provider_selection_from_settings(
            settings,
            app_config=app_config,
            workspace_context=workspace_context,
        )
        session = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        if session is not None and session.assistant_kind == "character":
            selection = replace(
                selection, system_prompt=self._resolved_system_prompt(session_id)
            )
        return selection

    def resolve_turn_configuration_snapshot(
        self, session_id: str
    ) -> ConsoleTurnConfigurationSnapshot:
        """Capture detached pre-gateway configuration for an owning session."""
        if self._turn_context_provider is not None:
            context = self._turn_context_provider(session_id)
            if not isinstance(context, ConsoleTurnConfigurationSnapshot):
                raise TypeError(
                    "Console turn-context provider must return "
                    "ConsoleTurnConfigurationSnapshot."
                )
            if context.session_id != session_id:
                raise ValueError(
                    "Console turn-context provider returned a different session."
                )
            return context

        selection = self._provider_selection_for_session(session_id)
        model = selection.explicit_model or selection.configured_model
        return ConsoleTurnConfigurationSnapshot.capture(
            session_id=session_id,
            provider_selection=selection,
            scratch_space=self._scratch_spaces.snapshot(session_id),
            session_settings=self.store.session_settings(session_id),
            workspace_roots=(),
            capabilities={
                "vision": bool(model)
                and is_vision_capable(selection.provider, model or ""),
                "max_history_images": max_history_images(selection.provider, model),
            },
            rag_defaults={},
            tool_configuration={
                "agent_runtime_enabled": self._agent_runtime_enabled,
                "native_tool_calls_enabled": True,
                "local_tools_enabled": coerce_bool_setting(
                    get_cli_setting("console", "local_tools_enabled", False),
                    False,
                ),
                "direct_library_tools": coerce_bool_setting(
                    get_cli_setting("console", "direct_library_tools", True),
                    True,
                ),
            },
            provider_payload_settings={
                "streaming": selection.streaming,
                "temperature": selection.temperature,
                "top_p": selection.top_p,
                "min_p": selection.min_p,
                "top_k": selection.top_k,
                "max_tokens": selection.max_tokens,
                "seed": selection.seed,
                "presence_penalty": selection.presence_penalty,
                "frequency_penalty": selection.frequency_penalty,
                "reasoning_effort": selection.reasoning_effort,
                "reasoning_summary": selection.reasoning_summary,
                "verbosity": selection.verbosity,
                "thinking_effort": selection.thinking_effort,
                "thinking_budget_tokens": selection.thinking_budget_tokens,
            },
        )

    def resolve_turn_execution_context(
        self, session_id: str
    ) -> ConsoleTurnConfigurationSnapshot:
        """Return pre-gateway configuration for legacy read-only consumers.

        Runtime send paths must construct :class:`ConsoleTurnExecutionContext`
        only after fresh Library-policy capture and gateway resolution.  This
        compatibility method therefore returns the explicitly named pre-gateway
        type and cannot manufacture an incomplete final context.
        """
        return self.resolve_turn_configuration_snapshot(session_id)

    async def _capture_turn_library_authority(
        self,
        session_id: str,
        configuration: ConsoleTurnConfigurationSnapshot,
    ) -> ConsoleTurnLibraryAuthority:
        """Freshly read and freeze maximum Library authority for one turn."""
        session = next(
            (item for item in self.store.sessions() if item.id == session_id),
            None,
        )
        if session is None:
            raise KeyError(f"Unknown Console session: {session_id}")

        coordinator = self.store.library_policy_coordinator
        if coordinator is None:
            policy = ConsoleLibraryPolicySnapshot(
                auto_retrieve=ConsoleAutoRetrieve.NEVER,
                assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
                policy_revision=None,
                source="unavailable",
                error_code="policy_read_error",
            )
        else:
            try:
                policy = await coordinator.capture_for_execution(session_id)
            except Exception:  # noqa: BLE001 - authority always fails closed
                policy = ConsoleLibraryPolicySnapshot(
                    auto_retrieve=ConsoleAutoRetrieve.NEVER,
                    assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
                    policy_revision=None,
                    source="unavailable",
                    error_code="policy_read_error",
                )

        held_scope = session.rag_scope_holder.scope
        note_ids: tuple[str, ...] = ()
        media_ids: tuple[str, ...] = ()
        conversations_allowed = True
        if held_scope is not None:
            note_ids = tuple(
                str(item.source_id)
                for item in held_scope.items
                if item.source_type == "note"
            )
            media_ids = tuple(
                str(item.source_id)
                for item in held_scope.items
                if item.source_type == "media"
            )
            conversations_allowed = False

        selection = configuration.provider_selection
        return ConsoleTurnLibraryAuthority(
            policy=policy,
            direct_library_tools=bool(
                configuration.tool_configuration.get(
                    "direct_library_tools",
                    True,
                )
            ),
            source_types=tuple(AUTOMATIC_LIBRARY_SOURCE_TYPES),
            scope_snapshot=ConsoleLibraryItemScopeSnapshot(
                note_ids=tuple(note_ids),
                media_ids=tuple(media_ids),
                conversations_allowed=conversations_allowed,
            ),
            provider_intent=ConsoleProviderIntent(
                provider=str(selection.provider),
                model=configuration.effective_model,
                endpoint=(
                    str(selection.base_url) if selection.base_url is not None else None
                ),
            ),
            attempt_id=str(uuid4()),
        )

    @staticmethod
    def _resolved_destination_for_context(
        resolution: Any,
    ) -> ConsoleResolvedDestination:
        """Require the exact typed Task-9 destination from a ready gateway."""
        destination = getattr(resolution, "resolved_destination", None)
        if isinstance(destination, ConsoleResolvedDestination):
            return destination
        raise ValueError("Ready provider resolution omitted its typed destination.")

    def _finalize_turn_execution_context(
        self,
        configuration: ConsoleTurnConfigurationSnapshot,
        library_authority: ConsoleTurnLibraryAuthority,
        resolution: Any,
    ) -> ConsoleTurnExecutionContext:
        """Combine frozen inputs only after the provider gateway resolves."""
        return ConsoleTurnExecutionContext(
            configuration=configuration,
            library_authority=library_authority,
            resolved_destination=self._resolved_destination_for_context(resolution),
        )

    async def _capture_and_resolve_turn_execution_context(
        self,
        session_id: str,
        configuration: ConsoleTurnConfigurationSnapshot | None = None,
    ) -> tuple[Any, ConsoleTurnExecutionContext | None]:
        """Capture one attempt's authority, then resolve and finalize its context.

        Callers invoke this only after their action-specific admission checks (or,
        for queued recovery, after the coordinator has reacquired the claim).  A
        non-ready gateway result has no execution context because no provider will
        execute; ready results always carry the complete immutable runtime input.
        """
        captured_configuration = (
            configuration
            if configuration is not None
            else self.resolve_turn_configuration_snapshot(session_id)
        )
        library_authority = await self._capture_turn_library_authority(
            session_id,
            captured_configuration,
        )
        # TASK-21145 (UAT H-3): the single resolve choke point carries the
        # hard deadline, so no send-path validation can hang unbounded.
        resolution = await self._resolve_for_send_bounded(
            captured_configuration.provider_selection
        )
        if not getattr(resolution, "ready", False):
            return resolution, None
        turn_context = self._finalize_turn_execution_context(
            captured_configuration,
            library_authority,
            resolution,
        )
        return resolution, turn_context

    @staticmethod
    def _require_complete_turn_execution_context(
        turn_context: object,
    ) -> ConsoleTurnExecutionContext:
        """Reject pre-gateway snapshots at every provider execution boundary."""
        if not isinstance(turn_context, ConsoleTurnExecutionContext):
            raise TypeError(
                "turn_context must be a complete ConsoleTurnExecutionContext"
            )
        return turn_context

    @staticmethod
    def _ensure_user_continuation_instruction(
        provider_messages: list[dict[str, Any]],
    ) -> None:
        if (
            provider_messages
            and provider_messages[-1].get("role") == ConsoleMessageRole.ASSISTANT.value
        ):
            provider_messages.append(
                {
                    "role": ConsoleMessageRole.USER.value,
                    "content": CONSOLE_CONTINUE_INSTRUCTION,
                }
            )

    @staticmethod
    def _has_user_turn(provider_messages: list[dict[str, Any]]) -> bool:
        return any(
            m.get("role") == ConsoleMessageRole.USER.value for m in provider_messages
        )

    def _pinned_prefill_for_session(self, session_id: str) -> str | None:
        """Return the session's pinned response prefill, if any."""
        settings = self.store.session_settings(session_id)
        pinned = getattr(settings, "pinned_prefill", None) if settings else None
        return pinned or None

    def _resolve_submit_prefill(self, session_id: str) -> tuple[str | None, bool]:
        """Return ``(prefill, from_one_shot)`` for a normal send.

        One-shot wins over pinned for the send it is armed for; pinned
        resumes afterward (the one-shot is only cleared on a complete or
        stopped outcome — see ``_consume_one_shot_prefill``).
        """
        one_shot = self.store.session_one_shot_prefill(session_id)
        if one_shot:
            return one_shot, True
        return self._pinned_prefill_for_session(session_id), False

    def _consume_one_shot_prefill(
        self, assistant_message_id: str, used_revision: int | None
    ) -> None:
        """Clear the armed one-shot after a send that used it terminated
        ``complete`` or ``stopped``. Blocked and failed sends never call
        this, so retry reproduces the original intent (spec §2).

        ``used_revision`` is the opaque slot identity captured at admission.
        Re-arming even identical text creates a newer identity and survives.
        """
        if used_revision is None:
            return
        try:
            session_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            return
        self.store.consume_session_one_shot_prefill(session_id, used_revision)

    async def _apply_skill_substitution(
        self, provider_messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None, tuple[str, ...], tuple[str, ...], str]:
        """Render-fresh the triggering turn's skill mention(s) at payload build time.

        Spec: "Invocation semantics" §5 (the substitution rule) -- one rule
        for fresh sends AND retry/regenerate/continue. Only the FINAL
        ``role == "user"`` message in ``provider_messages`` (the turn
        actually driving this send) is ever a substitution candidate; every
        earlier message -- including an earlier raw skill mention sitting
        in history -- is left untouched, so the persisted transcript always
        keeps the literal text the user typed (the raw mention is what gets
        submitted and stored; only the ephemeral provider payload for this
        turn is ever rendered). Re-resolves against a FRESH candidate
        snapshot and re-verifies trust through ``execute_skill`` on every
        call (never a cached snapshot), so a retry issued after a skill was
        edited (now untrusted) refuses/skips instead of silently re-running
        a stale render.

        Both forms are DETECTED against trusted candidates UNION
        user-invocable blocked (needs-review) skills -- a blocked skill must
        still be found (leading refuses, embedded degrades to literal +
        note) rather than silently staying plain, sigil-prefixed text with
        no signal at all. `execute_skill` remains the sole trust authority;
        detection here never grants execution.

        Two independent forms, tried in order:

        Leading form -- the message, with leading whitespace stripped
        (mirroring `_annotate_skill_commands`'s own preview `lstrip()`
        assumption -- a resolved leading mention replaces the whole message
        either way, so the leading whitespace simply disappears), starts
        with `MENTION_SIGIL` and the leading word resolves to a known
        skill: the REST of the (stripped) message is passed as that skill's
        args (`cap_skill_args`). A resolved leading mention is never also
        scanned for embedded mentions -- its args are opaque payload, not
        further mentions to expand.

        Embedded form -- tried whenever the leading form doesn't apply (no
        leading `MENTION_SIGIL`, or the leading word doesn't resolve):
        scans the ORIGINAL (unstripped) message. Every `$skill-name`
        mention anywhere in the message (case-sensitive, code-span-masked,
        document order -- `find_embedded_mentions`) is looked up ARGLESS
        (`execute_skill(name, mode="local", args="")`, once per unique
        name, right-to-left splice so earlier spans stay valid) and spliced
        in place at the mention's exact span, preserving all surrounding
        prose. Only an ``execution_mode == "inline"`` result splices;
        anything else (e.g. ``fork``, which has no "in place" meaning for
        an embedded mention) silently leaves that mention's literal `$name`
        text untouched. A trust-blocked mention (`SkillTrustBlockedError`)
        also leaves the literal text untouched but records a
        `SKILL_MENTION_SKIPPED_NOTE` for the caller to surface as a
        non-aborting system row.

        Args:
            provider_messages: The fully-built payload about to be sent to
                the provider (already includes any leading session-system
                message and any synthesized continuation instruction).

        Returns:
            A 5-tuple ``(provider_messages, refuse, notes, skill_bindings,
            skill_bundle_block)`` (Task 5, skills-fork-reachability).
            ``skill_bindings`` is the leading-RESOLVED skill's name (both
            ``inline`` and ``fork`` outcomes -- never on refuse) plus every
            embedded mention name that actually SPLICED (never a
            trust-blocked-literal or fork-literal mention).
            ``skill_bundle_block`` is the fully-rendered "Bundled files"
            block (`_render_skill_bundle_block`) for every bound skill
            whose `execute_skill` result carried non-empty
            `reference_files`, built as pure string work from the results
            already in hand this call (no re-execution, no extra service
            calls), or ``""`` when nothing bound has any. It is NEVER
            inserted into ``provider_messages`` here -- only ``run_reply``
            (bridge-side) ever appends it, so plain sends and the stored
            transcript never see it.

            ``(provider_messages, None, (), (), "")`` unchanged when there
            is no skills service configured, substitution is disabled,
            there is no final user message, that message's content isn't a
            string, or neither form applies. ``(new_messages, None, notes,
            skill_bindings, skill_bundle_block)`` when the leading form
            resolves and renders (``notes`` always empty for the leading
            form) or when the embedded pass splices one or more mentions
            (``notes`` carries one `SKILL_MENTION_SKIPPED_NOTE` per unique
            trust-blocked mention name, in document order); ``inline``
            replaces just the final message in place (history preserved);
            leading-form ``fork`` drops every message before it except a
            leading ``role == "system"`` message (clean context = session
            system prompt + rendered turn only).
            ``(provider_messages, refuse_copy, (), (), "")`` -- the
            ORIGINAL, unmodified messages, paired with
            `SKILL_UNTRUSTED_REFUSE` copy -- when a LEADING resolved skill
            is no longer trusted (`SkillTrustBlockedError` at
            execute-time); the caller must append `refuse_copy` as a
            system row and abort the turn without sending. An embedded
            mention never refuses/aborts -- it degrades to a
            literal-plus-note instead, and the send proceeds.
        """
        if self._skills_service is None or not self._skill_substitution_enabled:
            return provider_messages, None, (), (), ""

        final_index: int | None = None
        for index in range(len(provider_messages) - 1, -1, -1):
            if provider_messages[index].get("role") == ConsoleMessageRole.USER.value:
                final_index = index
                break
        if final_index is None:
            return provider_messages, None, (), (), ""

        content = provider_messages[final_index].get("content")
        if not isinstance(content, str):
            return provider_messages, None, (), (), ""
        if MENTION_SIGIL not in content:
            # Fast path: no sigil anywhere means neither form can possibly
            # apply -- plain-text sends never touch the skills service.
            return provider_messages, None, (), (), ""

        context = await self._skills_service.get_context(mode="local")
        candidates = self._skill_candidates_from_context(context)
        # DETECTION population = trusted candidates UNION user-invocable
        # blocked (needs-review) skills. A blocked skill must still be
        # DETECTED -- leading refuses, embedded degrades to literal + note
        # -- rather than silently staying plain, sigil-prefixed text with no
        # signal at all. `execute_skill` (not this resolution step) remains
        # the sole authority on whether a resolved name may actually run:
        # a name that resolves here to a blocked skill hits
        # `SkillTrustBlockedError` at the `execute_skill` call below/in the
        # embedded loop, which already drives the refuse/skip-with-note
        # paths.
        detection_candidates = candidates + self._skill_blocked_candidates_from_context(
            context
        )

        # --- Leading form: message starts with a resolvable $skill-name.
        # Leading whitespace is tolerated (stripped before the sigil check
        # and the word/rest split) to match `_annotate_skill_commands`'s own
        # `lstrip()` assumption in the preview -- a resolved leading mention
        # replaces the ENTIRE message on both the inline-replace and fork
        # paths, so the leading whitespace simply disappears either way.
        stripped_content = content.lstrip()
        if stripped_content.startswith(MENTION_SIGIL):
            word, rest = _split_skill_command_word(stripped_content)
            name = word[len(MENTION_SIGIL) :]
            if name:
                resolution = resolve_skill_command(name, rest, detection_candidates)
                if resolution.kind == "resolved":
                    args = cap_skill_args(rest)
                    try:
                        result = await self._skills_service.execute_skill(
                            resolution.name, mode="local", args=args
                        )
                    except SkillTrustBlockedError as exc:
                        refuse = SKILL_UNTRUSTED_REFUSE.format(
                            name=resolution.name, reason=exc.reason_code
                        )
                        return provider_messages, refuse, (), (), ""

                    rendered = (
                        result.get("rendered_prompt", "")
                        if isinstance(result, Mapping)
                        else ""
                    )
                    rendered_message = {
                        "role": ConsoleMessageRole.USER.value,
                        "content": rendered,
                    }
                    execution_mode = (
                        result.get("execution_mode")
                        if isinstance(result, Mapping)
                        else None
                    )
                    # Task 5: a resolved leading mention always binds its
                    # name (fork AND inline outcomes -- never on refuse,
                    # which already returned above) and its block is
                    # rendered from this single execute_skill result.
                    bindings = (resolution.name,)
                    block = (
                        _render_skill_bundle_block([result])
                        if isinstance(result, Mapping)
                        else ""
                    )
                    if execution_mode == "fork":
                        leading = (
                            [provider_messages[0]]
                            if provider_messages
                            and provider_messages[0].get("role")
                            == ConsoleMessageRole.SYSTEM.value
                            else []
                        )
                        return leading + [rendered_message], None, (), bindings, block

                    new_messages = list(provider_messages)
                    new_messages[final_index] = {
                        **provider_messages[final_index],
                        "content": rendered,
                    }
                    return new_messages, None, (), bindings, block

        # --- Embedded pass: no leading mention, or the leading word didn't
        # resolve to a known skill. Scans the ORIGINAL (unstripped) content
        # -- the leading-whitespace tolerance above only applies to the
        # leading form. `names` is the same detection population (trusted
        # UNION user-invocable blocked) so a blocked mention is found and
        # routed through the trust-blocked-note path below instead of
        # staying invisible.
        names = frozenset(candidate.name for candidate in detection_candidates)
        mentions = find_embedded_mentions(content, names)
        if not mentions:
            return provider_messages, None, (), (), ""

        rendered_by_name: dict[str, str | None] = {}
        # Task 5: results_by_name only keeps a name's execute_skill result
        # when that mention actually SPLICED (execution_mode == "inline")
        # -- a blocked-literal or fork-literal mention's result is
        # discarded here, so it can never leak into skill_bindings or the
        # rendered bundle block below.
        results_by_name: dict[str, Mapping[str, Any]] = {}
        notes: list[str] = []
        for mention in mentions:
            if mention.name in rendered_by_name:
                continue
            try:
                result = await self._skills_service.execute_skill(
                    mention.name, mode="local", args=""
                )
            except SkillTrustBlockedError:
                rendered_by_name[mention.name] = None
                notes.append(SKILL_MENTION_SKIPPED_NOTE.format(name=mention.name))
                continue
            execution_mode = (
                result.get("execution_mode") if isinstance(result, Mapping) else None
            )
            rendered = (
                result.get("rendered_prompt", "") if isinstance(result, Mapping) else ""
            )
            # Fork (or anything non-inline) cannot splice in place: leave
            # the mention literal, no note (this is not a trust failure).
            rendered_by_name[mention.name] = (
                rendered if execution_mode == "inline" else None
            )
            if execution_mode == "inline" and isinstance(result, Mapping):
                results_by_name[mention.name] = result

        new_content = content
        for mention in reversed(mentions):
            body = rendered_by_name.get(mention.name)
            if body is None:
                continue
            new_content = (
                new_content[: mention.start] + body + new_content[mention.end :]
            )
        if new_content == content:
            return provider_messages, None, tuple(notes), (), ""

        # Task 5: bound names are every unique mention that actually
        # spliced, in first-occurrence document order (`dict.fromkeys` on
        # `mentions` dedups while preserving order) -- never a
        # blocked-literal or fork-literal mention, which never reached
        # `results_by_name`.
        spliced_names = tuple(
            name
            for name in dict.fromkeys(mention.name for mention in mentions)
            if rendered_by_name.get(name) is not None
        )
        block = _render_skill_bundle_block(
            results_by_name[name] for name in spliced_names if name in results_by_name
        )
        new_messages = list(provider_messages)
        new_messages[final_index] = {
            **provider_messages[final_index],
            "content": new_content,
        }
        return new_messages, None, tuple(notes), spliced_names, block

    async def _apply_world_info(
        self, provider_messages: list[dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Inject conversation world-info into the final user message of the
        ephemeral provider payload (never the stored transcript).

        Runs AFTER `_apply_chat_dictionaries` so world-info matches the
        dict-substituted text the model will see. Conversation-only (the bound
        applier passes `char_data=None`). Offloaded via `asyncio.to_thread`;
        any failure returns the payload unchanged; `CancelledError` re-raised.
        """
        applier = self._world_info_applier
        if applier is None:
            return provider_messages

        session = next((s for s in self.store.sessions() if s.id == session_id), None)
        conversation_id = (
            session.persisted_conversation_id if session is not None else None
        )
        if not conversation_id:
            return provider_messages

        final_index: int | None = None
        for index in range(len(provider_messages) - 1, -1, -1):
            if provider_messages[index].get("role") == ConsoleMessageRole.USER.value:
                final_index = index
                break
        if final_index is None:
            return provider_messages

        message = provider_messages[final_index]
        content = message.get("content")
        if isinstance(content, str) and content.startswith(COMMAND_PREFIX):
            return provider_messages

        history = _normalize_world_info_history(provider_messages[:final_index])

        try:
            if isinstance(content, str):
                injected: Any = await asyncio.to_thread(
                    applier, conversation_id, content, history
                )
                if injected == content:
                    return provider_messages
                new_content = injected
            elif isinstance(content, list):
                combined = "\n".join(
                    part["text"]
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                )
                if not combined:
                    return provider_messages
                injected = await asyncio.to_thread(
                    applier, conversation_id, combined, history
                )
                if injected == combined:
                    return provider_messages
                prefix, _, suffix = injected.partition(combined)
                text_indices = [
                    i
                    for i, part in enumerate(content)
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ]
                first_idx = text_indices[0]
                last_idx = text_indices[-1]
                new_parts: list[Any] = []
                for i, part in enumerate(content):
                    if i == first_idx or i == last_idx:
                        new_text = part["text"]
                        if i == first_idx:
                            new_text = prefix + new_text
                        if i == last_idx:
                            new_text = new_text + suffix
                        new_parts.append({**part, "text": new_text})
                    else:
                        new_parts.append(part)
                new_content = new_parts
            else:
                return provider_messages
        except asyncio.CancelledError:
            raise
        except Exception:
            return provider_messages

        new_messages = list(provider_messages)
        new_messages[final_index] = {**message, "content": new_content}
        return new_messages

    async def _apply_chat_dictionaries(
        self, provider_messages: list[dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Apply the active conversation chat dictionaries to the final user
        message of the ephemeral provider payload (never the stored transcript).

        Mirrors `_apply_skill_substitution` (final `role == "user"` message
        only, one rule for fresh sends AND retry/continue/regenerate). The
        synchronous DB read + regex substitution are offloaded via
        `asyncio.to_thread` because native sends run as async workers on the UI
        event loop. Skill commands are left untouched. Any failure returns the
        payload unchanged so a dictionary problem can never break a send;
        `asyncio.CancelledError` is re-raised so a mid-send Stop still cancels.
        """
        applier = self._chat_dictionary_applier
        if applier is None:
            return provider_messages

        session = next((s for s in self.store.sessions() if s.id == session_id), None)
        conversation_id = (
            session.persisted_conversation_id if session is not None else None
        )
        if not conversation_id:
            return provider_messages

        final_index: int | None = None
        for index in range(len(provider_messages) - 1, -1, -1):
            if provider_messages[index].get("role") == ConsoleMessageRole.USER.value:
                final_index = index
                break
        if final_index is None:
            return provider_messages

        message = provider_messages[final_index]
        content = message.get("content")
        if isinstance(content, str) and content.startswith(COMMAND_PREFIX):
            return provider_messages

        try:
            if isinstance(content, str):
                new_content: Any = await asyncio.to_thread(
                    applier, conversation_id, content
                )
                if new_content == content:
                    return provider_messages
            elif isinstance(content, list):
                new_parts: list[Any] = []
                changed = False
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    ):
                        new_text = await asyncio.to_thread(
                            applier, conversation_id, part["text"]
                        )
                        if new_text != part["text"]:
                            changed = True
                            new_parts.append({**part, "text": new_text})
                            continue
                    new_parts.append(part)
                if not changed:
                    return provider_messages
                new_content = new_parts
            else:
                return provider_messages
        except asyncio.CancelledError:
            raise
        except Exception:
            return provider_messages

        new_messages = list(provider_messages)
        new_messages[final_index] = {**message, "content": new_content}
        return new_messages

    @staticmethod
    def _skill_candidates_from_context(
        context: Any,
    ) -> tuple[SkillCommandCandidate, ...]:
        """Build the user-invocable, trusted skill candidate population.

        Mirrors ``ConsoleSkillController.
        _console_skill_trusted_candidates_from_context``'s filter -- kept as
        a small duplicate rather than a shared import because `Chat/`
        business logic must not depend on `UI/Screens/` (project layering),
        and `console_skill_resolver` deliberately stays unaware of trust/
        context shape (see its own module docstring).
        """
        available = (
            context.get("available_skills") if isinstance(context, Mapping) else None
        )
        return tuple(
            SkillCommandCandidate(
                name=str(item.get("name")),
                description=str(item.get("description") or ""),
            )
            for item in (available or [])
            if isinstance(item, Mapping)
            and item.get("name")
            and item.get("user_invocable", True)
            and not item.get("trust_blocked", False)
        )

    @staticmethod
    def _skill_blocked_candidates_from_context(
        context: Any,
    ) -> tuple[SkillCommandCandidate, ...]:
        """Build the user-invocable, trust-BLOCKED (needs-review) skill
        candidate population.

        Companion to `_skill_candidates_from_context`: unioned with it in
        `_apply_skill_substitution` to widen the DETECTION population (never
        the executable one) so a `$blocked-name` mention resolves a name
        instead of silently staying literal, sigil-prefixed text with no
        refusal or note at all. `execute_skill` remains the sole authority
        on whether a resolved name may actually run -- candidates built here
        are never executed directly by this method's caller. A blocked
        skill flagged ``user_invocable: False`` is excluded, mirroring
        `_skill_candidates_from_context`'s own filter discipline.
        """
        blocked = (
            context.get("blocked_skills") if isinstance(context, Mapping) else None
        )
        return tuple(
            SkillCommandCandidate(
                name=str(item.get("name")),
                description=str(item.get("description") or ""),
            )
            for item in (blocked or [])
            if isinstance(item, Mapping)
            and item.get("name")
            and item.get("user_invocable", True)
        )

    @staticmethod
    def _validated_draft(
        draft: str, *, allow_empty: bool = False
    ) -> tuple[str, str | None]:
        return validate_console_draft(draft, allow_empty=allow_empty)

    @staticmethod
    def _blocked_visible_copy(copy: str) -> str:
        if "Provider blocked" in copy:
            return copy
        if copy.startswith("WIP:"):
            return f"Provider blocked: {copy}"
        return copy or "Provider blocked."

    def _block(self, session_id: str, visible_copy: str) -> ConsoleSubmitResult:
        self._set_run_state(
            ConsoleRunState.blocked(visible_copy), session_id=session_id
        )
        self.store.append_message(
            session_id,
            role=ConsoleMessageRole.SYSTEM,
            content=visible_copy,
        )
        return ConsoleSubmitResult(
            accepted=False,
            should_clear_draft=False,
            visible_copy=visible_copy,
        )

    def _notify_app(self, message: str, *, severity: str = "warning") -> None:
        """Raise one best-effort toast through the app, never raising.

        Mirrors ``_notify_detached_approval``'s defensive shape: an app double
        whose ``notify`` takes the message alone, or no app at all, must not
        turn a refusal into an exception on the send path.
        """

        app = self.app
        notify = getattr(app, "notify", None) if app is not None else None
        if not callable(notify):
            return
        try:
            notify(message, severity=severity)
        except TypeError:
            try:
                notify(message)
            except Exception:  # noqa: BLE001 -- surfacing is best-effort
                logger.debug("Console send refusal notice could not be delivered")
        except Exception as exc:  # noqa: BLE001 -- surfacing is best-effort
            logger.debug(
                "Console send refusal notice raised (exception_type={})",
                type(exc).__name__,
            )

    def _block_undurable_turn(
        self,
        session_id: str,
        *,
        origin: ConsoleSubmissionOrigin,
        queue_entry_id: str | None,
    ) -> ConsoleSubmitResult:
        """Refuse a durable turn nothing can commit -- visibly (TASK-22030).

        `56db75386` turned this refusal into a bare ``ConsoleSubmitResult``:
        no run state, no transcript row, no toast. With an unopenable
        ChaChaNotes database that made Send do *nothing at all* -- the draft
        stayed put and the app looked like it had ignored the keypress, which
        reads as "the app is broken" rather than "your database is broken".

        The refusal itself is correct and stays (a turn that cannot be
        committed must not reach the provider), but it now names its real
        cause, keeps the draft, and points at the one thing that still works.
        """

        persistence = self.store.persistence
        if persistence is None or getattr(persistence, "db", None) is None:
            visible_copy = (
                "Not sent: your conversation database could not be opened, so "
                "this message could not be saved. Restart Chatbook, and check "
                "the app log for the database error if it keeps happening. "
                "Your draft was kept; a temporary chat still sends."
            )
        else:
            visible_copy = (
                "Not sent: this conversation cannot be saved right now, so the "
                "message was not sent to the provider. Your draft was kept; a "
                "temporary chat still sends."
            )
        blocked = self._block(session_id, visible_copy)
        self._notify_app(visible_copy, severity="error")
        return replace(
            blocked,
            session_id=session_id,
            origin=origin,
            queue_entry_id=queue_entry_id,
        )

    async def _capture_rag_context(
        self,
        draft: str,
        turn_context: ConsoleTurnExecutionContext | None = None,
        origin: ConsoleSubmissionOrigin = ConsoleSubmissionOrigin.MANUAL,
    ) -> tuple[
        str | None,
        CitationTraceBuilder | None,
        str | None,
        CitationRepairContract | None,
    ]:
        """Resolve optional staged RAG context without exposing request state."""

        provider = self._rag_capture_provider
        if provider is None:
            return None, None, None, None
        session_id = turn_context.session_id if turn_context is not None else None
        anchor_message_id = (
            self.store.active_leaf(session_id) if session_id is not None else None
        )
        trace_prefix = (
            f"console-trace:{anchor_message_id}:retrieval"
            if anchor_message_id is not None
            else None
        )
        previous_event_id: str | None = None

        def record(event_kind: str, summary: str, status: str) -> None:
            nonlocal previous_event_id
            if session_id is None or anchor_message_id is None:
                return
            event_id = f"{trace_prefix}:{event_kind}"
            self.store.record_trace_event(
                session_id,
                anchor_message_id=anchor_message_id,
                event_kind=event_kind,
                summary=summary,
                status=status,
                event_id=event_id,
                parent_event_id=previous_event_id,
                source_event_id=previous_event_id,
                sensitivity="retrieval_metadata",
            )
            previous_event_id = event_id

        record("retrieval_started", "Retrieval started", "started")
        try:
            parameters = inspect.signature(provider).parameters
            accepts_context = any(
                parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
                for index, parameter in enumerate(parameters.values())
                if index > 0
            )
            origin_parameter = parameters.get("origin")
        except (TypeError, ValueError):
            accepts_context = False
            origin_parameter = None
        try:
            if origin_parameter is not None:
                captured = await provider(draft, turn_context, origin=origin)
            elif accepts_context:
                captured = await provider(draft, turn_context)
            else:
                captured = await provider(draft)
        except asyncio.CancelledError:
            record("retrieval_failed", "Retrieval cancelled", "cancelled")
            raise
        except Exception:
            logger.error(
                "Console RAG capture unavailable; "
                f"reason=capture_provider_failure; draft_length={len(draft)}"
            )
            record("retrieval_failed", "Retrieval failed", "failed")
            return None, None, None, None
        normalized = self._normalize_rag_capture(captured)
        if normalized[2] is not None:
            record(
                "retrieval_candidates_selected",
                "Retrieval candidates selected",
                "completed",
            )
        record("retrieval_completed", "Retrieval completed", "completed")
        return normalized

    async def _capture_frozen_rag_context(
        self,
        draft: str,
        turn_context: ConsoleTurnExecutionContext,
        continuation: _PreparedSendContinuation,
    ) -> tuple[
        str | None,
        CitationTraceBuilder | None,
        str | None,
        CitationRepairContract | None,
    ]:
        """Capture only evidence frozen at original send admission."""

        lease = continuation.staged_evidence
        if lease is None:
            return None, None, None, None
        owner = getattr(self._rag_capture_provider, "__self__", None)
        provider = getattr(owner, "_capture_frozen_console_staged_rag", None)
        if not callable(provider):
            return None, None, None, None
        try:
            captured = await provider(
                draft,
                turn_context,
                lease.launch,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                "Frozen Console RAG capture unavailable; "
                f"reason=capture_provider_failure; draft_length={len(draft)}"
            )
            return None, None, None, None
        lease.capture_result = captured
        return self._normalize_rag_capture(captured)

    def _release_prepared_evidence(
        self, continuation: _PreparedSendContinuation | None
    ) -> None:
        """Release a captured launch at the exact accepted-turn boundary."""

        lease = continuation.staged_evidence if continuation is not None else None
        if lease is None or lease.released or lease.capture_result is None:
            return
        release = lease.release
        if not callable(release):
            return
        release(lease.launch, lease.capture_result)
        lease.released = True
        lease.launch = None
        lease.capture_result = None
        lease.release = None

    @staticmethod
    def _normalize_rag_capture(
        captured: Any,
    ) -> tuple[
        str | None,
        CitationTraceBuilder | None,
        str | None,
        CitationRepairContract | None,
    ]:
        """Normalize one retrieval capture into the controller contract."""

        captured_context = getattr(captured, "context", None)
        context = (
            captured_context
            if isinstance(captured_context, str) and captured_context.strip()
            else None
        )
        captured_builder = getattr(captured, "citation_builder", None)
        builder = (
            captured_builder
            if isinstance(captured_builder, CitationTraceBuilder)
            else None
        )
        captured_prompt_id = getattr(captured, "prompt_evidence_set_id", None)
        prompt_evidence_set_id = (
            captured_prompt_id
            if isinstance(captured_prompt_id, str) and captured_prompt_id.strip()
            else None
        )
        captured_repair_contract = getattr(
            captured,
            "citation_repair_contract",
            None,
        )
        repair_contract = (
            captured_repair_contract
            if isinstance(captured_repair_contract, CitationRepairContract)
            and context is not None
            and captured_repair_contract.evidence_context == context
            else None
        )
        return context, builder, prompt_evidence_set_id, repair_contract

    @staticmethod
    def _build_terminal_citation_finalizer(
        *,
        context: str | None,
        builder: CitationTraceBuilder | None,
        prompt_evidence_set_id: str | None,
    ) -> TerminalCitationFinalizer | None:
        """Build exact-body citation finalization for one eligible initial send."""

        if (
            not isinstance(context, str)
            or not context.strip()
            or not isinstance(builder, CitationTraceBuilder)
            or not isinstance(prompt_evidence_set_id, str)
            or not prompt_evidence_set_id.strip()
        ):
            return None

        def finalize(answer_body: str) -> SealedCitationWrite | None:
            terminal_at = datetime.now(UTC)
            try:
                attempt_id = builder.record_initial_answer_attempt(
                    prompt_evidence_set_id=prompt_evidence_set_id,
                    answer_body=answer_body,
                    completed_at=terminal_at,
                )
                return builder.seal(
                    selected_attempt_id=attempt_id,
                    sealed_at=terminal_at,
                )
            except CitationTraceBuildUnavailable:
                logger.warning(
                    "Console citation finalization unavailable; "
                    "reason=occurrence_mapping_unavailable"
                )
            except Exception:
                logger.warning(
                    "Console citation finalization unavailable; "
                    "reason=attempt_or_seal_failure"
                )
            return None

        return finalize

    @staticmethod
    def _prepend_evidence_context(
        provider_messages: list[dict[str, Any]],
        context: str,
    ) -> list[dict[str, Any]]:
        """Prefix exact evidence to the final provider-only user message."""

        final_index = next(
            (
                index
                for index in range(len(provider_messages) - 1, -1, -1)
                if provider_messages[index].get("role") == ConsoleMessageRole.USER.value
            ),
            None,
        )
        if final_index is None:
            return provider_messages
        prefix = f"Evidence: {context}\n\n---\n\n"
        message = provider_messages[final_index]
        content = message.get("content")
        if isinstance(content, str):
            new_content: Any = prefix + content
        elif isinstance(content, list):
            new_content = list(content)
            text_index = next(
                (
                    index
                    for index, part in enumerate(new_content)
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ),
                None,
            )
            if text_index is None:
                new_content.insert(0, {"type": "text", "text": prefix})
            else:
                text_part = new_content[text_index]
                new_content[text_index] = {
                    **text_part,
                    "text": prefix + text_part["text"],
                }
        else:
            return provider_messages
        updated = list(provider_messages)
        updated[final_index] = {**message, "content": new_content}
        return updated

    def _notify_submission_accepted(
        self,
        *,
        session_id: str,
        origin: ConsoleSubmissionOrigin,
        entry_id: str | None,
        context_epoch: int,
        defer_queued_settlement: bool = False,
    ) -> None:
        """Commit queue ownership, then invoke the origin-appropriate UI hook."""

        self.prompt_queue_coordinator.turn_accepted(
            session_id,
            origin=origin,
            context_epoch=context_epoch,
            entry_id=entry_id,
            defer_queued_settlement=defer_queued_settlement,
        )
        if origin is not ConsoleSubmissionOrigin.MANUAL:
            return
        callback = self.on_submission_accepted
        if callback is None:
            return
        try:
            callback()
        except Exception:
            # The hook is a UI convenience (composer clearing); a failure there
            # must never abort an already-accepted provider run.
            pass

    def _notify_queued_submission_accepted(
        self, event: ConsoleQueuedAcceptanceEvent
    ) -> None:
        """Forward a content-free queued acceptance without touching the composer."""

        callback = self.on_queued_submission_accepted
        if callback is None:
            return
        try:
            callback(event)
        except Exception:
            pass

    async def _submit_queued_entry(
        self,
        text: str,
        *,
        session_id: str,
        entry_id: str,
        authorization: QueueGenerationAuthorization,
    ) -> ConsoleSubmitResult:
        """Submit one coordinator-claimed entry through the normal turn pipeline."""

        return await self.submit_draft(
            text,
            session_id=session_id,
            origin=ConsoleSubmissionOrigin.QUEUED,
            queue_entry_id=entry_id,
            queue_authorization=authorization,
        )

    async def _record_prompt_history(self, text: str) -> None:
        """Append an accepted send's draft to the shared prompt history.

        Best-effort (TASK-1364): ``PromptHistory.append`` already logs and
        swallows its own IO failures, and the guard here keeps even an
        unexpected raise from breaking an already-accepted run. Empty or
        whitespace-only drafts (attachment-only sends) record nothing.

        Args:
            text: The cleaned draft text that was just accepted for sending.
        """
        history = self.prompt_history
        if history is None or not text.strip():
            return
        try:
            await history.append(text)
        except Exception:
            logger.opt(exception=True).warning(
                "Prompt-history recording failed for an accepted send."
            )

    _IMAGE_REJECTION_RECOVERY_HINT = (
        " This conversation includes an image attachment; if the model can't "
        "accept images, remove that message (select it and use Delete) or "
        "switch to a vision-capable model."
    )

    def _session_history_carries_images(self, session_id: str) -> bool:
        """Return whether any message in the session carries an image.

        TASK-335: history re-sends attachments on every turn, so a provider
        that rejects images fails ALL later sends in the conversation with
        the same opaque status — the failure copy names the likely cause.
        """
        try:
            messages = self.store.messages_for_session(session_id)
        except KeyError:
            return False
        for message in messages:
            if getattr(message, "attachments", None):
                return True
            if getattr(message, "image_data", None) is not None:
                return True
        return False

    def _append_failure_system_row(self, session_id: str, visible_copy: str) -> None:
        """Append a transcript-only system row describing a provider failure."""
        try:
            self.store.append_message(
                session_id,
                role=ConsoleMessageRole.SYSTEM,
                content=visible_copy,
            )
        except KeyError:
            # Session vanished mid-failure (e.g. closed); the run-state copy
            # still carries the failure for the control surfaces.
            pass

    def _append_history_trimmed_note(self, session_id: str, dropped: int) -> None:
        """Append a transcript-only system row noting history was trimmed."""
        try:
            self.store.append_message(
                session_id,
                role=ConsoleMessageRole.SYSTEM,
                content=(
                    "Earlier messages were trimmed to fit the model's context "
                    f"window ({dropped} dropped)."
                ),
            )
        except KeyError:
            # Session vanished mid-send; the dispatched payload was still bounded.
            pass

    def _durable_context_snapshots(
        self, session_id: str
    ) -> tuple[DurableMessageSnapshot, ...] | None:
        """Capture the active durable lineage without leaking content to logs."""
        persistence = getattr(self.store, "persistence", None)
        version_reader = getattr(persistence, "get_message_version", None)
        if not callable(version_reader):
            return None
        try:
            active_ids = self.store.active_path_message_ids(session_id)
            messages = {
                message.id: message
                for message in self.store.messages_for_session(session_id)
            }
        except KeyError:
            return None
        snapshots: list[DurableMessageSnapshot] = []
        for native_id in active_ids:
            message = messages.get(native_id)
            if message is None:
                return None
            persisted_id = message.persisted_message_id
            if not persisted_id:
                # The just-created assistant placeholder is not part of the
                # request and has no durable content to summarize.
                if (
                    message.role is ConsoleMessageRole.ASSISTANT
                    and message.status != "complete"
                ):
                    continue
                if message.role is ConsoleMessageRole.SYSTEM:
                    continue
                return None
            try:
                version = version_reader(persisted_id)
            except Exception:
                return None
            if type(version) is not int or version < 1:
                return None
            variant_id: str | None = None
            variant_index: int | None = None
            if message.variants is not None:
                try:
                    variant_id = message.variants.current.id
                    variant_index = message.variants.selected_index
                except (AttributeError, IndexError):
                    return None
            attachment_digests: list[str] = []
            for attachment in message.attachments:
                data_digest = (
                    hashlib.sha256(attachment.data).hexdigest()
                    if attachment.data is not None
                    else "unavailable"
                )
                attachment_digests.append(
                    hashlib.sha256(
                        (
                            f"{attachment.position}\0{attachment.mime_type}\0"
                            f"{attachment.display_name}\0{data_digest}"
                        ).encode("utf-8")
                    ).hexdigest()
                )
            if not message.attachments and message.image_data is not None:
                attachment_digests.append(
                    hashlib.sha256(
                        (message.image_mime_type or "image/unknown").encode("utf-8")
                        + b"\0"
                        + message.image_data
                    ).hexdigest()
                )
            snapshots.append(
                DurableMessageSnapshot(
                    message_id=persisted_id,
                    version=version,
                    role=message.role.value,
                    content=message.content,
                    selected_variant_id=variant_id,
                    selected_variant_index=variant_index,
                    attachment_digests=tuple(attachment_digests),
                )
            )
        return tuple(snapshots)

    @staticmethod
    def _messages_after_memory_boundary(
        provider_messages: list[dict[str, Any]],
        boundary_native_id: str,
    ) -> list[dict[str, Any]] | None:
        """Retain the system prefix and only transcript rows after a memory boundary."""
        leading_end = 0
        while (
            leading_end < len(provider_messages)
            and provider_messages[leading_end].get("role") == "system"
        ):
            leading_end += 1
        boundary_index = next(
            (
                index
                for index, row in enumerate(provider_messages)
                if row.get(NATIVE_MESSAGE_ID_KEY) == boundary_native_id
            ),
            None,
        )
        if boundary_index is None or boundary_index < leading_end:
            return None
        return [
            *provider_messages[:leading_end],
            *provider_messages[boundary_index + 1 :],
        ]

    def _global_context_policy_overrides(self):
        keys = (
            "conversation_budget_mode",
            "conversation_budget_tokens",
            "compaction_mode",
            "compaction_representation",
            "compaction_trigger_ratio",
            "compaction_target_ratio",
            "compaction_summary_max_tokens",
            "compaction_failure_behavior",
            "compaction_carry_forward_mode",
        )
        values = {key: get_cli_setting("console", key, None) for key in keys}
        return context_policy_overrides_from_console_config(values)

    def context_control_inputs(
        self, session_id: str
    ) -> tuple[
        ConsoleContextPolicyOverrides,
        ConsoleContextPolicyOverrides | None,
        ConsoleMemoryRecord | None,
    ]:
        """Return policy and branch-valid memory inputs for settings UI.

        This read-only seam keeps widgets away from the repository and from
        the active-lineage validation rules used by provider dispatch.
        """
        owner = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        if owner is None:
            raise KeyError(session_id)
        try:
            global_overrides = self._global_context_policy_overrides()
        except Exception:
            global_overrides = None
        memory = None
        if (
            self._context_repository is not None
            and owner.persisted_conversation_id is not None
        ):
            snapshots = self._durable_context_snapshots(session_id)
            # Truthiness on purpose: None (unvalidatable lineage) and ()
            # (no durable rows yet) both select no memory — an empty prefix
            # can't validate any candidate — matching the send-path guards.
            if snapshots:
                memory = select_valid_memory(
                    self._context_repository.list_active_memories(
                        owner.persisted_conversation_id
                    ),
                    snapshots,
                )
        return owner.context_policy_overrides, global_overrides, memory

    def reset_active_context_memory(self, session_id: str) -> tuple[str, int] | None:
        """Deactivate only the branch-valid memory and return its undo token."""
        owner = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        repository = self._context_repository
        if (
            owner is None
            or repository is None
            or owner.persisted_conversation_id is None
        ):
            return None
        snapshots = self._durable_context_snapshots(session_id)
        # Truthiness on purpose: None and () both mean nothing can be reset.
        if not snapshots:
            return None
        memory = select_valid_memory(
            repository.list_active_memories(owner.persisted_conversation_id),
            snapshots,
        )
        if memory is None:
            return None
        reset_at = datetime.now(UTC).isoformat()
        if not repository.deactivate_memory(
            memory.memory_id,
            expected_revision=memory.revision,
            reset_at=reset_at,
        ):
            return None
        return memory.memory_id, memory.revision + 1

    def undo_context_memory_reset(
        self,
        memory_id: str,
        expected_revision: int,
    ) -> bool:
        """Undo a current-branch reset if its exact revision is still inactive."""
        repository = self._context_repository
        if repository is None:
            return False
        return repository.reactivate_memory(
            memory_id,
            expected_revision=expected_revision,
        )

    def reset_all_context_memories(self, session_id: str) -> int:
        """Deactivate every branch memory for one durable conversation."""
        owner = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        repository = self._context_repository
        if (
            owner is None
            or repository is None
            or owner.persisted_conversation_id is None
        ):
            return 0
        return repository.deactivate_all_memories(
            owner.persisted_conversation_id,
            reset_at=datetime.now(UTC).isoformat(),
        )

    async def compact_context_now(self, session_id: str) -> tuple[bool, str]:
        """Run one user-initiated bounded compaction without sending a turn."""
        if not self.run_state_for(session_id).is_send_allowed:
            return False, "Wait for the active run to finish before compacting."
        owner = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        if owner is None or owner.persisted_conversation_id is None:
            return False, "Send or save this conversation before compacting it."
        try:
            resolution = await self._resolve_for_send_bounded(
                self._provider_selection()
            )
        except Exception:
            return False, "The active provider could not be prepared for compaction."
        if not getattr(resolution, "ready", False):
            return False, self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
        overrides, global_overrides, before_memory = self.context_control_inputs(
            session_id
        )
        requested_representation = merge_context_policy(
            global_overrides=global_overrides,
            conversation_overrides=overrides,
        ).compaction_representation
        try:
            continuation_sidecar, continuation_target = (
                self._provider_continuation_history_for_resolution(
                    session_id, resolution
                )
            )
        except ContinuationConflictError:
            return False, PROVIDER_CONTINUATION_RECOVERY_REQUIRED
        _messages, blocked_result = await self._apply_conversation_memory_preflight(
            session_id=session_id,
            resolution=resolution,
            provider_messages=self._provider_messages_for_session(
                session_id, annotate_ids=True
            ),
            assistant_message_id="",
            agent_tools_enabled=False,
            force_compaction=True,
            manual_action=True,
            continuation_sidecar=continuation_sidecar,
            continuation_target=continuation_target,
        )
        if blocked_result is not None:
            return False, blocked_result.visible_copy
        _overrides, _global, after_memory = self.context_control_inputs(session_id)
        if after_memory is None or (
            before_memory is not None
            and after_memory.memory_id == before_memory.memory_id
        ):
            if (
                requested_representation
                is ContextCompactionRepresentation.VISUAL_TRANSCRIPT
                and is_vision_capable(resolution.provider, resolution.model or "")
            ):
                return (
                    True,
                    "Visual transcript fits and will be regenerated locally for each request; transcript unchanged.",
                )
            return False, "There are not enough older complete turns to compact yet."
        return True, "Conversation memory updated; transcript messages were unchanged."

    def _compaction_admission(
        self,
        *,
        session_id: str,
        resolution: ConsoleProviderResolution,
        prompt: CompactionPromptSnapshot,
    ) -> CompactionAdmission | None:
        repository = self._context_repository
        if repository is None:
            return None
        owner = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        if owner is None or owner.persisted_conversation_id is None:
            return None
        snapshots = self._durable_context_snapshots(session_id)
        if not snapshots:
            return None
        policy_read = repository.load_policy(owner.persisted_conversation_id)
        memory = select_valid_memory(
            repository.list_active_memories(owner.persisted_conversation_id),
            snapshots,
        )
        return CompactionAdmission(
            conversation_id=owner.persisted_conversation_id,
            captured_leaf_message_id=snapshots[-1].message_id,
            lineage=tuple(message.message_id for message in snapshots),
            payload_revision=self.store.payload_revision(session_id),
            identity_revision=owner.identity_revision,
            policy_revision=policy_read.revision,
            active_memory_id=memory.memory_id if memory is not None else None,
            active_memory_revision=memory.revision if memory is not None else None,
            provider=resolution.provider,
            model=resolution.model or "",
            prompt_digest=prompt.digest,
            prefix_digest=prefix_digest(snapshots),
        )

    def _block_context_preflight(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        visible_copy: str,
    ) -> ConsoleSubmitResult:
        try:
            self.store.mark_message_failed(assistant_message_id)
        except (KeyError, ValueError):
            pass
        self._append_failure_system_row(session_id, visible_copy)
        self._set_run_state(
            ConsoleRunState.blocked(visible_copy), session_id=session_id
        )
        return ConsoleSubmitResult(True, True, visible_copy)

    async def _apply_conversation_memory_preflight(
        self,
        *,
        session_id: str,
        resolution: ConsoleProviderResolution,
        provider_messages: list[dict[str, Any]],
        assistant_message_id: str,
        agent_tools_enabled: bool,
        force_compaction: bool = False,
        manual_action: bool = False,
        continuation_sidecar: tuple[ProviderContinuationSidecar, ...] = (),
        continuation_target: ContinuationRestoreTarget | None = None,
    ) -> tuple[list[dict[str, Any]], ConsoleSubmitResult | None]:
        """Revalidate memory and optionally run one automatic summary call."""

        def blocked(visible_copy: str) -> ConsoleSubmitResult:
            if manual_action:
                return ConsoleSubmitResult(False, True, visible_copy)
            return self._block_context_preflight(
                session_id=session_id,
                assistant_message_id=assistant_message_id,
                visible_copy=visible_copy,
            )

        repository = self._context_repository
        service = self._compaction_service
        prepare = getattr(self.provider_gateway, "prepare_chat_request", None)
        owner = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        if (
            repository is None
            or service is None
            or not callable(prepare)
            or owner is None
            or owner.persisted_conversation_id is None
        ):
            return provider_messages, None
        snapshots = self._durable_context_snapshots(session_id)
        if not snapshots:
            return provider_messages, None
        conversation_id = owner.persisted_conversation_id
        memory = select_valid_memory(
            repository.list_active_memories(conversation_id), snapshots
        )
        retained_messages = provider_messages
        memory_rows: tuple[Mapping[str, Any], ...] = ()
        if memory is not None:
            boundary_native_id = next(
                (
                    message.id
                    for message in self.store.messages_for_session(session_id)
                    if message.persisted_message_id == memory.boundary_message_id
                ),
                None,
            )
            retained = self._messages_after_memory_boundary(
                provider_messages, boundary_native_id or ""
            )
            if retained is None:
                memory = None
            else:
                retained_messages = retained
                memory_rows = (tagged_memory_message(memory.summary_text),)

        tools: list[Mapping[str, Any]] = []
        if agent_tools_enabled and self._agent_bridge is not None:
            preview = getattr(self._agent_bridge, "preview_tool_schemas", None)
            if callable(preview):
                try:
                    tools = list(preview())
                except Exception:
                    tools = []
        prepared_before = prepare(
            resolution,
            retained_messages,
            tools=tools,
            apply_safety_window=False,
            continuation_target=continuation_target,
            continuation_sidecar=continuation_sidecar,
            continuation_owner_key=(
                NATIVE_MESSAGE_ID_KEY if continuation_sidecar else None
            ),
        )
        semantic = prepared_before.semantic
        if memory_rows:
            semantic = replace(semantic, memory=memory_rows)
            prepared_before = prepare(
                resolution,
                semantic,
                apply_safety_window=False,
                continuation_target=continuation_target,
            )
        capacity = prepared_before.capacity
        mandatory_tokens = (
            prepared_before.accounting.non_compactable_tokens
            - prepared_before.accounting.memory_tokens
        )
        try:
            global_overrides = self._global_context_policy_overrides()
        except Exception:
            global_overrides = None
        resolved = resolve_context_policy(
            capacity=ConsoleContextCapacity(
                model_context_window_tokens=capacity.context_window_tokens,
                provider_input_cap_tokens=capacity.provider_input_cap_tokens,
                response_reservation_tokens=capacity.effective_response_tokens,
                safety_margin_tokens=capacity.safety_margin_tokens,
                mandatory_input_tokens=mandatory_tokens,
            ),
            global_overrides=global_overrides,
            conversation_overrides=owner.context_policy_overrides,
        )

        def prepare_main(request: PreparedConsoleRequest):
            return prepare(
                resolution,
                request,
                apply_safety_window=False,
                continuation_target=continuation_target,
            )

        units = compactable_units_after(
            snapshots,
            boundary_message_id=(
                memory.boundary_message_id if memory is not None else None
            ),
        )
        decision = decide_compaction(
            resolved,
            conversation_tokens=(
                prepared_before.accounting.memory_tokens
                + prepared_before.accounting.compactable_tokens
            ),
            compactable_units=len(units),
        )
        if force_compaction and units:
            decision = CompactionDecision.AUTOMATIC
        logger.info("console_context_policy_decision")
        if decision in {CompactionDecision.OFF, CompactionDecision.BELOW_TRIGGER}:
            return _flatten_preflight_messages(semantic), None
        if decision is CompactionDecision.ASK:
            result = blocked(
                (
                    "Conversation context reached its compaction threshold. "
                    "Review and approve compaction before sending again."
                )
            )
            return provider_messages, result
        if decision in {
            CompactionDecision.UNKNOWN_WINDOW,
            CompactionDecision.NON_COMPACTABLE,
        }:
            # A missing compaction threshold or an empty set of replaceable
            # units is not itself a provider overflow.  Unknown/new models
            # historically remained sendable with an explicit unverified
            # label, and reaching a policy high-water mark while the exact
            # request still fits must not turn that advisory threshold into
            # an admission failure.  Block only when the immutable prepared
            # request proves that the effective input ceiling is exceeded.
            if not prepared_before.known_overflow:
                return _flatten_preflight_messages(semantic), None
            if (
                resolved.policy.failure_behavior
                is CompactionFailureBehavior.OMIT_OLDER_CONTEXT
            ):
                return _flatten_preflight_messages(semantic), None
            if decision is CompactionDecision.NON_COMPACTABLE:
                limiting_reason = (
                    "No older complete conversation turns are available to compact."
                )
                recovery = (
                    "Reduce the active request, system/tool/source context, or "
                    "response maximum."
                )
            else:
                limiting_reason = (
                    resolved.validation_errors[0]
                    if resolved.validation_errors
                    else "The effective model input ceiling is unavailable."
                )
                recovery = (
                    "Repair the model limit, reduce mandatory context or the "
                    "response maximum, or allow older turns to be omitted."
                )
            result = blocked(
                (
                    "This request cannot fit the selected model. "
                    f"{limiting_reason} Summarizing older turns cannot make "
                    f"enough room. {recovery}"
                )
            )
            return provider_messages, result

        requested_representation = resolved.policy.compaction_representation
        vision_available = False
        if requested_representation is not ContextCompactionRepresentation.TEXT_SUMMARY:
            try:
                vision_available = is_vision_capable(
                    resolution.provider, resolution.model or ""
                )
            except Exception:
                vision_available = False
        effective_representation, visual_fallback_reason = (
            resolve_effective_compaction_representation(
                requested_representation,
                vision_available=vision_available,
            )
        )

        if (
            effective_representation
            is ContextCompactionRepresentation.VISUAL_TRANSCRIPT
        ):
            budget = resolved.effective_conversation_budget_tokens
            visual_plan = None
            if budget is not None:
                try:
                    visual_plan = await asyncio.to_thread(
                        plan_visual_compaction,
                        semantic=semantic,
                        prepared_before=prepared_before,
                        durable_units=units,
                        budget_tokens=budget,
                        target_ratio=resolved.policy.target_ratio,
                        max_images=max_history_images(
                            resolution.provider, resolution.model or ""
                        ),
                        keep_latest_exchange=(
                            resolved.policy.carry_forward_mode
                            is ContextCarryForwardMode.MEMORY_WITH_LATEST_EXCHANGE
                        ),
                        prepare_main=prepare_main,
                    )
                except Exception:
                    visual_fallback_reason = "local_visual_render_failed"
            if visual_plan is not None and visual_plan.plan is not None:
                logger.info("console_visual_compaction_prepared")
                return _flatten_preflight_messages(visual_plan.plan.semantic), None
            effective_representation = ContextCompactionRepresentation.TEXT_SUMMARY
            if visual_fallback_reason is None:
                visual_fallback_reason = (
                    visual_plan.reason
                    if visual_plan is not None
                    else "visual_compaction_unavailable"
                )

        if visual_fallback_reason is not None:
            logger.info("console_visual_compaction_fell_back_to_text")

        prompt = CompactionPromptSnapshot(
            get_internal_prompt("console.rewind_summarize")
        )

        def prepare_auxiliary(messages, output_cap):
            return prepare(
                replace(
                    resolution,
                    streaming=False,
                    max_tokens=output_cap,
                ),
                list(messages),
                apply_safety_window=False,
            )

        planned = plan_compaction(
            semantic=semantic,
            prepared_before=prepared_before,
            durable_units=units,
            resolved_policy=resolved,
            prompt=prompt,
            prior_memory=memory,
            prepare_main=prepare_main,
            prepare_auxiliary=prepare_auxiliary,
        )
        if planned.plan is None:
            if (
                resolved.policy.failure_behavior
                is CompactionFailureBehavior.OMIT_OLDER_CONTEXT
            ):
                return _flatten_preflight_messages(semantic), None
            result = blocked(
                (
                    "Conversation compaction could not reach the configured "
                    "target in one bounded summary call."
                )
            )
            return provider_messages, result

        admission = self._compaction_admission(
            session_id=session_id,
            resolution=resolution,
            prompt=prompt,
        )
        if admission is None:
            return provider_messages, blocked(
                "Conversation changed before compaction could start."
            )
        boundary_index = next(
            index
            for index, snapshot in enumerate(snapshots)
            if snapshot.message_id == planned.plan.boundary_message_id
        )
        transaction = await service.compact(
            admission=admission,
            plan=planned.plan,
            resolution=resolution,
            prompt=prompt,
            current_admission=lambda: self._compaction_admission(
                session_id=session_id,
                resolution=resolution,
                prompt=prompt,
            ),
            prepare_main=prepare_main,
            prefix_messages=snapshots[: boundary_index + 1],
        )
        if transaction.terminal is CompactionTerminal.SUCCEEDED:
            memory_rows_after: tuple[Mapping[str, Any], ...] = (
                tagged_memory_message(transaction.memory.summary_text),
            )
            if effective_representation is ContextCompactionRepresentation.HYBRID:
                hybrid_visual_added = False
                try:
                    image_limit = max_history_images(
                        resolution.provider, resolution.model or ""
                    )
                    remaining_image_capacity = image_limit - count_semantic_images(
                        planned.plan.remaining_semantic
                    )
                    if remaining_image_capacity > 0:
                        artifact = await asyncio.to_thread(
                            render_visual_transcript,
                            planned.plan.selected_units,
                            summarized_prefix_digest=(
                                transaction.memory.summarized_prefix_digest
                            ),
                            max_pages=remaining_image_capacity,
                        )
                        visual_row = tagged_visual_memory_message(
                            [page.png_bytes for page in artifact.pages],
                            # Wire integrity (exact PNG bytes), not renderer identity.
                            page_hashes=[page.png_sha256 for page in artifact.pages],
                        )
                        hybrid_semantic = PreparedConsoleRequest(
                            system=planned.plan.remaining_semantic.system,
                            memory=memory_rows_after + (visual_row,),
                            mandatory=planned.plan.remaining_semantic.mandatory,
                            compactable=planned.plan.remaining_semantic.compactable,
                            active_request=planned.plan.remaining_semantic.active_request,
                            active_continuation_groups=(
                                planned.plan.remaining_semantic.active_continuation_groups
                            ),
                            tools=planned.plan.remaining_semantic.tools,
                        )
                        hybrid_prepared = prepare_main(hybrid_semantic)
                        hybrid_conversation_tokens = (
                            hybrid_prepared.accounting.memory_tokens
                            + hybrid_prepared.accounting.compactable_tokens
                        )
                        if (
                            not hybrid_prepared.known_overflow
                            and hybrid_conversation_tokens
                            <= planned.plan.target_conversation_tokens
                        ):
                            memory_rows_after += (visual_row,)
                            hybrid_visual_added = True
                except Exception:
                    pass
                if not hybrid_visual_added:
                    logger.info("console_visual_compaction_fell_back_to_text")
            after = PreparedConsoleRequest(
                system=planned.plan.remaining_semantic.system,
                memory=memory_rows_after,
                mandatory=planned.plan.remaining_semantic.mandatory,
                compactable=planned.plan.remaining_semantic.compactable,
                active_request=planned.plan.remaining_semantic.active_request,
                active_continuation_groups=(
                    planned.plan.remaining_semantic.active_continuation_groups
                ),
                tools=planned.plan.remaining_semantic.tools,
            )
            return _flatten_preflight_messages(after), None
        if (
            resolved.policy.failure_behavior
            is CompactionFailureBehavior.OMIT_OLDER_CONTEXT
        ):
            return _flatten_preflight_messages(semantic), None
        result = blocked(
            (
                "Conversation compaction did not complete; the provider request "
                "was not sent."
            )
        )
        return provider_messages, result

    async def _stream_assistant_response(
        self,
        *,
        resolution: Any,
        provider_messages: list[dict[str, str]],
        assistant_message_id: str,
        prepare_retry: bool = False,
        variant_mode: bool = False,
        prefill: str | None = None,
        prefill_from_one_shot: bool = False,
        one_shot_prefill_revision: int | None = None,
        skill_bindings: tuple[str, ...] = (),
        skill_bundle_block: str = "",
        citation_repair_session: ConsoleCitationRepairSession | None = None,
        turn_context: ConsoleTurnExecutionContext | None = None,
        preparation_id: str | None = None,
        stream_signals: ConsoleProviderStreamSignals | None = None,
        generation_token: int | None = None,
    ) -> ConsoleSubmitResult:
        try:
            return await self._stream_assistant_response_inner(
                resolution=resolution,
                provider_messages=provider_messages,
                assistant_message_id=assistant_message_id,
                prepare_retry=prepare_retry,
                variant_mode=variant_mode,
                prefill=prefill,
                prefill_from_one_shot=prefill_from_one_shot,
                one_shot_prefill_revision=one_shot_prefill_revision,
                skill_bindings=skill_bindings,
                skill_bundle_block=skill_bundle_block,
                citation_repair_session=citation_repair_session,
                turn_context=turn_context,
                preparation_id=preparation_id,
                stream_signals=stream_signals,
                generation_token=generation_token,
            )
        finally:
            if isinstance(turn_context, ConsoleTurnExecutionContext):
                try:
                    self.store.settle_session_library_destination(
                        turn_context.session_id,
                        expected_attempt_id=turn_context.library_authority.attempt_id,
                        expected_message_id=assistant_message_id,
                    )
                except KeyError:
                    pass
            if citation_repair_session is not None:
                citation_repair_session.clear_governed_state()

    async def _stream_assistant_response_inner(
        self,
        *,
        resolution: Any,
        provider_messages: list[dict[str, str]],
        assistant_message_id: str,
        prepare_retry: bool = False,
        variant_mode: bool = False,
        prefill: str | None = None,
        prefill_from_one_shot: bool = False,
        one_shot_prefill_revision: int | None = None,
        skill_bindings: tuple[str, ...] = (),
        skill_bundle_block: str = "",
        citation_repair_session: ConsoleCitationRepairSession | None = None,
        turn_context: ConsoleTurnExecutionContext | None = None,
        preparation_id: str | None = None,
        stream_signals: ConsoleProviderStreamSignals | None = None,
        generation_token: int | None = None,
    ) -> ConsoleSubmitResult:
        try:
            owner_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            # The message itself is already gone -- no owning session to
            # attribute this to; default (active session) is a harmless
            # no-op since nothing will ever read a closed session's state.
            return self._session_closed_result()
        owner = next((s for s in self.store.sessions() if s.id == owner_id), None)
        turn_context = self._require_complete_turn_execution_context(turn_context)
        if turn_context.session_id != owner_id:
            raise ValueError("Console turn context does not own the assistant row.")
        try:
            continuation_sidecar, continuation_target = (
                self._provider_continuation_history_for_resolution(owner_id, resolution)
            )
        except ContinuationConflictError:
            return self._block_context_preflight(
                session_id=owner_id,
                assistant_message_id=assistant_message_id,
                visible_copy=PROVIDER_CONTINUATION_RECOVERY_REQUIRED,
            )
        # A character session always takes the plain-provider
        # path, even with the global agent runtime enabled and a bridge
        # present. Keyed on the message's OWNING session (looked up here,
        # not the controller's active session) so a session switch racing
        # this send can't flip which branch a still-in-flight message uses.
        force_plain = owner is not None and owner.assistant_kind == "character"
        # Cost-ticker PR3: record this send's payload-fingerprint baseline at
        # the single dispatch choke point covering BOTH the direct-provider
        # and agent paths (they branch further down). Two boundaries are
        # deliberate here, not oversights:
        #
        # (a) Fingerprint from a FRESH `_provider_messages_for_session(
        #     owner_id)` call -- the same raw, pre-compaction, pre-window
        #     stage `compute_current_fingerprint` recomputes from -- rather
        #     than the `provider_messages` parameter in scope here. Every
        #     caller of `_stream_assistant_response` has already run its own
        #     per-send transforms (skill substitution, chat-dictionary/
        #     world-info folding, RAG injection, ...) on `provider_messages`
        #     before passing it in, so fingerprinting the parameter directly
        #     would compare a TRANSFORMED payload against
        #     `compute_current_fingerprint`'s untransformed one and falsely
        #     report "earlier history changed" for any session using those
        #     features. Re-deriving from the store keeps both sides on the
        #     same raw view. A compaction fold or a token-window trim
        #     (below) can still shrink what actually gets sent relative to
        #     this snapshot -- neither is a payload EDIT, so neither counts
        #     as a cache break either.
        # (b) Per-send final-turn substitutions themselves (dictionaries/
        #     world-info/skills) are ephemeral and stay invisible to the
        #     fingerprint even though they can cause real provider-side
        #     prefix instability for the sessions that use them. The chip's
        #     GROUND-TRUTH cache fields (`cache_ttl_snapshot`, sourced from
        #     actual usage) still report honestly regardless; only the
        #     break-alert's named REASON won't attribute to this cause. V1
        #     boundary -- a named "content substitution changed" reason is a
        #     reasonable follow-up, not required here.
        #
        # Provider/model come from the RESOLUTION actually being dispatched,
        # not `self.provider`/`self.model` -- those are controller-wide
        # mutable fields shared across every fleet session, and a
        # provider/model switch racing the awaits between `resolve_for_send`
        # and here (or a background session's send racing a foreground
        # switch) can leave them holding a DIFFERENT pair than what this
        # call actually sends (see `bound_messages_to_window`'s own comment
        # a few lines down, which avoids the same hazard for the same
        # reason). Falling back to `self.provider`/`self.model` only covers
        # narrow stand-in resolutions that don't carry the attributes at
        # all (e.g. some test doubles) use the same captured context.
        try:
            baseline_messages = self._provider_messages_for_session(
                owner_id, turn_context=turn_context
            )
            baseline_provider = (
                getattr(resolution, "provider", None)
                or turn_context.provider_selection.provider
            )
            baseline_model = (
                getattr(resolution, "model", None) or turn_context.effective_model
            )
            self._payload_fingerprint_baselines[owner_id] = fingerprint_payload(
                baseline_provider, baseline_model, baseline_messages
            )
        except Exception as exc:
            logger.bind(session_id=owner_id, error=repr(exc)).warning(
                "cost_fingerprint_record_failed"
            )
        character_emote_snapshot: CharacterEmoteRunSnapshot | None = None
        if force_plain:
            try:
                character_emote_snapshot = (
                    await self._character_emote_snapshot_for_run(owner_id)
                )
            except _CharacterEmoteAuthorityChanged:
                return self._block_context_preflight(
                    session_id=owner_id,
                    assistant_message_id=assistant_message_id,
                    visible_copy=(
                        "Character context changed before dispatch; try again."
                    ),
                )
            if character_emote_snapshot is not None:
                provider_messages = self._apply_character_emote_prompt(
                    provider_messages,
                    character_emote_snapshot,
                )
        # SP2 /rewind "summarize up to here": at the SINGLE dispatch choke point
        # (agent + direct both flow through here), fold the session's boundary
        # summary into the payload -- but ONLY when the boundary message is
        # actually present in it (the leak rule; see
        # _apply_context_summary_compaction). Runs BEFORE bound_messages_to_
        # window so the summary lands in the leading system prefix the trimmer
        # preserves.
        provider_messages = self._apply_context_summary_compaction(
            owner_id, provider_messages
        )
        if isinstance(resolution, ConsoleProviderResolution):
            (
                provider_messages,
                context_block,
            ) = await self._apply_conversation_memory_preflight(
                session_id=owner_id,
                resolution=resolution,
                provider_messages=provider_messages,
                assistant_message_id=assistant_message_id,
                agent_tools_enabled=(
                    self._agent_runtime_enabled
                    and self._agent_bridge is not None
                    and not prefill
                    and not force_plain
                ),
                continuation_sidecar=continuation_sidecar,
                continuation_target=continuation_target,
            )
            if context_block is not None:
                return context_block
        # TASK-14811.2: the real gateway now owns exact capacity resolution,
        # whole-unit windowing, provider serialization, accounting, and
        # dispatch as one immutable artifact. Do not pre-trim production
        # payloads here: doing so used a parallel estimate and the historical
        # hidden half-window response clamp. Older gateway fakes retain the
        # legacy path so their established two-argument contract remains a
        # useful isolated controller seam.
        exact_preparation = callable(
            getattr(self.provider_gateway, "prepare_chat_request", None)
        )
        bound = None
        if not exact_preparation:
            bound = bound_messages_to_window(
                provider_messages,
                model=(
                    getattr(resolution, "model", None)
                    or turn_context.effective_model
                    or ""
                ),
                provider=(
                    getattr(resolution, "provider", None)
                    or turn_context.provider_selection.provider
                ),
                response_reservation=(
                    getattr(resolution, "max_tokens", None)
                    or turn_context.provider_selection.max_tokens
                    or DEFAULT_RESPONSE_RESERVATION
                ),
            )
            provider_messages = bound.messages
        # Strip the private id-threading key from every row before dispatch:
        # it existed solely so the compaction above could anchor the boundary
        # by identity (see NATIVE_MESSAGE_ID_KEY). This is the single latest
        # point covering BOTH the direct stream path (`stream_chat` below) and
        # the agent path (`agent_messages = list(provider_messages)` in
        # `_run_agent_reply`), so no provider/gateway/agent ever sees the key.
        # Rebuild fresh row dicts rather than mutating in place, since transforms
        # can leave earlier rows aliased to freshly-built builder dicts.
        selected_owner_ids = {
            row.get(NATIVE_MESSAGE_ID_KEY)
            for row in provider_messages
            if type(row.get(NATIVE_MESSAGE_ID_KEY)) is str
        }
        continuation_sidecar = tuple(
            item
            for item in continuation_sidecar
            if item.owner_message_id in selected_owner_ids
        )
        if not continuation_sidecar:
            provider_messages = [
                {k: v for k, v in row.items() if k != NATIVE_MESSAGE_ID_KEY}
                for row in provider_messages
            ]
        if bound is not None and bound.dropped_count:
            # Reuse the guarded owner_id resolved above; the note helper
            # swallows a store-close race that happens during the append.
            self._append_history_trimmed_note(owner_id, bound.dropped_count)
        self.store.begin_session_library_destination_attempt(
            owner_id,
            turn_context.library_authority,
            turn_context.resolved_destination,
            assistant_message_id,
        )
        active_task = asyncio.current_task()
        with self._active_workspace_roots_lock:
            self._active_workspace_roots_by_session[owner_id] = (
                turn_context.workspace_roots
            )
        self._active_assistant_message_ids[owner_id] = assistant_message_id
        self._active_stream_tasks[owner_id] = active_task
        self._stop_requested = False
        self._active_citation_repair_sessions[owner_id] = citation_repair_session
        # Unconditional (final-review F1): these signals used to be built
        # only for citation repair, which left the DEFAULT send path -- the
        # agent runtime, on by config default with the bridge always wired --
        # forwarding `provider_stream_signals=None`, so the bridge never
        # passed `signals=` to the gateway and NOTHING was ever captured for
        # the path virtually every real send takes. Cost is not an opt-in
        # feature of one repair mode; every run needs its own signals object.
        stream_signals = stream_signals or self._new_run_stream_signals()
        self._active_capture_details[owner_id] = stream_signals.capture_detail
        # Trajectory sidecar (schema v38): arm this turn's timing capture at
        # the single dispatch choke point covering BOTH the direct-provider
        # and agent paths, BEFORE the provider call. First-token is stamped
        # at the store's chunk seam; completion at usage-attach. Best-effort
        # -- a sidecar failure must never fail the send.
        try:
            self.store.record_trajectory_timing(
                assistant_message_id, step_started_at=time.time()
            )
        except Exception as exc:
            logger.bind(message_id=assistant_message_id, error=repr(exc)).warning(
                "trajectory_step_start_failed"
            )
        try:
            if (
                bool(
                    turn_context.tool_configuration.get(
                        "agent_runtime_enabled", self._agent_runtime_enabled
                    )
                )
                and self._agent_bridge is not None
                and not prefill
                and not force_plain
            ):
                return await self._run_agent_reply(
                    resolution=resolution,
                    provider_messages=provider_messages,
                    assistant_message_id=assistant_message_id,
                    prepare_retry=prepare_retry,
                    variant_mode=variant_mode,
                    skill_bindings=skill_bindings,
                    skill_bundle_block=skill_bundle_block,
                    citation_repair_session=citation_repair_session,
                    stream_signals=stream_signals,
                    turn_context=turn_context,
                    continuation_sidecar=continuation_sidecar,
                    continuation_history_target=continuation_target,
                    preparation_id=preparation_id,
                    generation_token=generation_token,
                )
            return await self._run_direct_provider_reply(
                resolution=resolution,
                provider_messages=provider_messages,
                assistant_message_id=assistant_message_id,
                prepare_retry=prepare_retry,
                variant_mode=variant_mode,
                prefill=prefill,
                prefill_from_one_shot=prefill_from_one_shot,
                one_shot_prefill_revision=one_shot_prefill_revision,
                citation_repair_session=citation_repair_session,
                stream_signals=stream_signals,
                continuation_sidecar=continuation_sidecar,
                continuation_target=continuation_target,
                character_emote_snapshot=character_emote_snapshot,
                preparation_id=preparation_id,
                generation_token=generation_token,
            )
        finally:
            if (
                self._active_stream_tasks.get(owner_id) is active_task
                and self._active_assistant_message_ids.get(owner_id)
                == assistant_message_id
            ):
                self._active_stream_tasks.pop(owner_id, None)
                self._active_assistant_message_ids.pop(owner_id, None)
                with self._active_workspace_roots_lock:
                    self._active_workspace_roots_by_session.pop(owner_id, None)
                self._stop_requested = False
                if (
                    self._active_citation_repair_sessions.get(owner_id)
                    is citation_repair_session
                ):
                    self._active_citation_repair_sessions.pop(owner_id, None)
                # Task 3b (agent path): `_run_agent_reply`'s own finally
                # deliberately leaves its cancel_event live past its own
                # return (see that finally's docstring) so the citation-
                # repair post-generation check -- which runs afterward, on
                # this same task, via `_finalize_agent_reply` -- still
                # observes it. This is the one place left to retire it, now
                # that the whole run (agent OR direct) has fully finished.
                # A no-op for the direct path, whose own finally already
                # popped its own cancel_event before returning.
                self._active_cancel_events.pop(owner_id, None)
                self._active_capture_details.pop(owner_id, None)

    @staticmethod
    def _usage_payloads(stream_signals: Any) -> list[Any]:
        """Every provider-call usage payload this turn has closed out so far.

        Extracted so ``_attach_stream_usage`` and ``unattributed_fleet_
        tokens`` read the accumulator through ONE seam -- the second is
        defined entirely as "what the first has not billed yet", and a
        second copy of this tolerance would let the two drift.
        """
        payloads_getter = getattr(stream_signals, "usage_payloads", None)
        if callable(payloads_getter):
            return list(payloads_getter())
        # Tolerate narrow stand-ins that only expose the single-call
        # attribute (the pre-accumulation shape), same defensive posture as
        # the getattr-based resolution reads in `_attach_stream_usage`.
        single = getattr(stream_signals, "usage_payload", None)
        return [single] if single else []

    def _new_run_stream_signals(self) -> ConsoleProviderStreamSignals:
        """One run's signals object, with exchange capture gated by config.

        ``get_cli_setting`` reads the RESOLVED settings layer -- never raw
        TOML top-level, which nests under COMPREHENSIVE_CONFIG_RAW and
        silently never fires (cost-ticker PR2 Qodo F4 was exactly that
        bug). Both signals-creation call sites (the dispatch site and the
        defensive belt inside the direct-provider stream method) route
        through this one helper so the kill-switch reaches every run.

        ``get_cli_setting`` also returns the RAW TOML value, never coerced
        -- a hand-typed ``exchange_capture = "false"`` is a non-empty
        string and therefore truthy under bare ``bool()``, which would
        silently defeat the only escape hatch for this privacy-sensitive
        feature. ``coerce_bool_setting`` (already imported at module scope)
        is the arc's sixth site with this exact trap -- see the
        ``local_tools_enabled`` read a few hundred lines up for the first.
        """
        runtime = runtime_capture_policy()
        return ConsoleProviderStreamSignals(
            exchange_capture_enabled=runtime.enabled,
            capture_detail=CaptureDetail.SAFE,
        )

    def _watch_post_turn_usage(
        self,
        session_id: str,
        stream_signals: ConsoleProviderStreamSignals | None,
        resolution: Any,
        *,
        assistant_message_id: str | None = None,
        partial: bool = False,
    ) -> None:
        """Remember where this turn's billing stopped (PR3a-1 Task 6b, F3).

        Called immediately after the turn's ONE usage attach. Everything
        appended to ``stream_signals`` from here on is a surviving fleet
        child's spend: real money, on a message nobody attaches to again --
        until the conversation's fleet DRAINS.

        Two consumers read what this records (PR3a-2 Task 3 closed the
        3a-1 "observable, not fixed" gap):

        - ``unattributed_fleet_tokens`` (the chip's "Sub-agents: N tok
          (not priced)" line) reads the per-SESSION watch -- spend billed
          since the attach, zeroed again when the fold below runs;
        - ``_reattach_fleet_usage`` (the ``FleetDrained`` fan-out consumer)
          reads the per-MESSAGE source recorded here and re-attaches the
          whole turn (recompute-all + REPLACE -- idempotent, pinned by
          ``test_re_attaching_the_same_signals_is_idempotent``), carrying
          the same ``partial`` flag this turn's own attach used.

        The per-message source is recorded only when the bridge reports the
        conversation still OWES a drain (``has_unsettled_children``): a
        turn whose children all settled within the turn -- or that never
        had any -- would otherwise retain its signals object until
        teardown with no drain ever coming to pop it. On any ambiguity
        (no such seam on the bridge, or a raising check) the source is
        recorded anyway: money over memory.

        Args:
            session_id: The session whose turn just attached usage.
            stream_signals: The turn's signals object; ``None`` clears the
                session watch.
            resolution: The turn's provider resolution (provider/model for
                payload normalization).
            assistant_message_id: The turn's originating assistant message
                -- the row the drain fold re-attaches to. ``None`` (the
                legacy shape) records no re-attach source.
            partial: The flag the turn's own attach used; the fold reuses
                it so a stopped turn's record stays partial.
        """
        if stream_signals is None:
            self._post_turn_usage_watch.pop(session_id, None)
            return
        try:
            # The thread running this IS the one that owns the store (every
            # caller is an async controller method); remember its loop so
            # the drain consumer -- a child's thread, maybe post-teardown --
            # can hop back onto it.
            self._usage_reattach_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass  # sync test harness: the consumer then runs inline
        self._post_turn_usage_watch[session_id] = (
            stream_signals,
            resolution,
            len(self._usage_payloads(stream_signals)),
        )
        if assistant_message_id is None:
            return
        bridge = self._agent_bridge
        if bridge is None:
            # No bridge -> no fleet -> nothing can ever drain: recording a
            # source would be a straight leak.
            return
        checker = getattr(bridge, "has_unsettled_children", None)
        if callable(checker):
            try:
                if not checker(self._agent_conversation_id(session_id)):
                    return
            except Exception as exc:  # noqa: BLE001 -- money over memory: keep the source
                logger.debug(
                    "has_unsettled_children raised; recording re-attach source anyway (exception_type={})",
                    type(exc).__name__,
                )
        self._fleet_usage_reattach_sources[assistant_message_id] = (
            stream_signals,
            resolution,
            partial,
        )

    def _register_fleet_usage_reattach(self, bridge: Any) -> None:
        """Register the last-child-settled usage fold on this bridge.

        PR3a-2 Task 3 (tasks 15660/15667): one bridge-lifetime fan-out
        consumer, registered next to bridge attachment (constructor and
        ``update_agent_runtime``) -- never from ``run_reply``, per
        ``FleetDrainFanout.register``'s contract. Tolerates a bridge
        without the seam (older fakes) and no bridge at all.

        Args:
            bridge: The Console agent bridge to register on, or ``None``.
        """
        if bridge is None:
            return
        register = getattr(bridge, "on_fleet_drained", None)
        if callable(register):
            register("usage-reattach", self._on_fleet_drained_reattach_usage)

    @property
    def fleet_wake(self) -> ConsoleFleetWakeCoordinator:
        """The auto-wake coordinator (PR3a-2 Task 5): the screen wires its
        app object (``fleet_wake.wire``), calls ``seed_from_marks`` at
        mount BEFORE the first tab sync, and pokes ``retry_soon`` when the
        composer empties."""
        return self._fleet_wake

    def _register_fleet_wake(self, bridge: Any) -> None:
        """Register the auto-wake drain consumer on this bridge.

        PR3a-2 Task 5: same contract and same call sites as
        ``_register_fleet_usage_reattach`` directly above -- constructor
        and ``update_agent_runtime``, never ``run_reply``; replace-by-name
        makes re-registration on the same bridge a safe no-op. Also
        captures the running loop (the app loop in production) as the
        thread the delivery half hops onto.

        Args:
            bridge: The Console agent bridge to register on, or ``None``.
        """
        self._fleet_wake.capture_loop_if_running()
        if bridge is None:
            return
        register = getattr(bridge, "on_fleet_drained", None)
        if callable(register):
            register(
                ConsoleFleetWakeCoordinator.NAME,
                self._fleet_wake.on_fleet_drained,
            )

    def _on_fleet_drained_reattach_usage(self, event: Any) -> None:
        """``FleetDrained`` consumer: hop off the child's thread and fold.

        Runs under the fan-out's consumer contract: the CHILD's own daemon
        thread, possibly after the Console screen (and this controller's
        owner) are gone. The store is single-threaded, so the actual fold
        is scheduled onto the loop captured at watch time -- the app loop,
        which outlives the screen -- via ``call_soon_threadsafe``. With no
        loop ever captured (sync harnesses), or the loop already closed
        (app exit while a child settles: the last chance to record real
        money), the fold runs inline instead; every path is wrapped so
        nothing propagates back into the fan-out.

        Args:
            event: The ``FleetDrained`` event.
        """
        loop = self._usage_reattach_loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._reattach_fleet_usage_guarded, event)
                return
            except RuntimeError:
                pass  # closed between the check and the call: fall through
        self._reattach_fleet_usage_guarded(event)

    def _reattach_fleet_usage_guarded(self, event: Any) -> None:
        """Never-raise wrapper: a ``call_soon_threadsafe`` callback that
        raised would land in the loop's exception handler, and an inline
        call would propagate into the fan-out's per-consumer catch --
        neither may happen for a best-effort cost figure."""
        try:
            self._reattach_fleet_usage(event)
        except Exception as exc:  # noqa: BLE001 -- a dropped fold is a missing figure, not a broken run
            logger.warning(
                "fleet usage re-attach failed (exception_type={})",
                type(exc).__name__,
            )

    def _reattach_fleet_usage(self, event: Any) -> None:
        """Fold every drained turn's full spend back onto its own message.

        PR3a-2 Task 3 (tasks 15660/15667). For each distinct originating
        assistant message in the drain: recompute the turn's TOTAL from
        ALL of its signals' payloads -- the turn's own calls plus
        everything its children (survivors included; ``error``/
        ``cancelled`` children's partial spend is still real spend) billed
        since -- and REPLACE the stored usage, with the same ``partial``
        flag the turn's own attach used. Then sync the session watch's
        attached-count so ``unattributed_fleet_tokens`` falls back to
        zero, and pop the source: after a drain no child of that turn
        exists to bill into its signals again, so a second delivery of the
        same event finds nothing and is a no-op (15660 AC#2's end-to-end
        idempotence; the attach itself is idempotent besides).

        A child whose turn recorded no source (a within-turn drain firing
        before the finalize, or ``run_id is None`` for a child that died
        pre-``create_run``) is skipped: the turn's own attach covers
        everything billed up to it.

        Args:
            event: The ``FleetDrained`` event to fold.
        """
        processed: set[str] = set()
        for child in getattr(event, "children", ()) or ():
            message_id = getattr(child, "assistant_message_id", None)
            if not message_id or message_id in processed:
                continue
            processed.add(message_id)
            source = self._fleet_usage_reattach_sources.pop(message_id, None)
            if source is None:
                continue
            stream_signals, resolution, partial = source
            self._attach_stream_usage(
                message_id, stream_signals, resolution, partial=partial
            )
            session_id = getattr(child, "session_id", None)
            watch = self._post_turn_usage_watch.get(session_id) if session_id else None
            if watch is not None and watch[0] is stream_signals:
                self._post_turn_usage_watch[session_id] = (
                    stream_signals,
                    watch[1],
                    len(self._usage_payloads(stream_signals)),
                )

    def unattributed_fleet_tokens(self, session_id: str) -> int:
        """Tokens this session billed AFTER its turn's usage was attached.

        A fleet child outlives the turn that spawned it and keeps streaming
        into the same ``ConsoleProviderStreamSignals``; the agent path
        attaches usage once, the instant ``run_reply`` returns. This is the
        difference -- spend the user was charged for that the message row
        and its cost figure do not include YET: when the conversation's
        fleet drains, ``_reattach_fleet_usage`` folds it onto the message
        row and re-baselines the watch, so this reads non-zero only in the
        window between a survivor's billing and its last sibling settling
        (PR3a-2 Task 3, tasks 15660/15667).

        Read by ``ChatScreen._build_console_cost_state`` and folded into the
        chip's unpriced sub-agent token line, so the money is named rather
        than silently gone. Cheap: a list slice and a normalize per payload,
        over the handful of calls one turn's survivors make.

        Args:
            session_id: The session to report.

        Returns:
            The unattributed token total, or 0 when the session has no
            watched turn, nothing new since the attach, or nothing that
            normalizes to usage.
        """
        watch = self._post_turn_usage_watch.get(session_id)
        if watch is None:
            return 0
        stream_signals, resolution, attached_count = watch
        payloads = self._usage_payloads(stream_signals)[attached_count:]
        if not payloads:
            return 0
        provider = str(getattr(resolution, "provider", "") or "")
        model = str(getattr(resolution, "model", "") or "")
        total: ProviderUsage | None = None
        for payload in payloads:
            usage = ProviderUsage.from_provider_payload(
                payload, provider=provider, model=model, partial=True
            )
            if usage is None:
                continue
            total = usage if total is None else total.plus(usage)
        return total.total_tokens if total is not None else 0

    def _attach_stream_usage(
        self,
        assistant_message_id: str,
        stream_signals: ConsoleProviderStreamSignals | None,
        resolution: Any,
        *,
        partial: bool,
    ) -> None:
        """Sum every provider call this turn made; never fail a send (spec PR1).

        One turn can make N provider calls (an agent loop runs one per step),
        each of which the gateway closes out separately. Each payload is
        normalized into disjoint buckets ON ITS OWN and the buckets are then
        summed -- raw payloads are never merged across calls, or a later
        call's ``prompt_tokens`` would be priced against an earlier call's
        stale ``prompt_tokens_details.cached_tokens``.
        """
        # Trajectory sidecar (schema v38): completion stamp + finalize flush.
        # Runs BEFORE the usage early-returns so a turn with no usage still
        # gets its completed_at stamped and its assistant row flushed. Never
        # fails the send (same posture as the usage attach below).
        try:
            self.store.record_trajectory_timing(
                assistant_message_id,
                completed_at=time.time(),
                model=str(getattr(resolution, "model", "") or "") or None,
                provider=str(getattr(resolution, "provider", "") or "") or None,
                flush=True,
            )
        except Exception as exc:
            logger.bind(message_id=assistant_message_id, error=repr(exc)).warning(
                "trajectory_completion_failed"
            )
        if stream_signals is None:
            return
        # Conversation Inspector (task-7): attach captured exchanges at the
        # same call sites usage attaches, on the SAME never-fail posture --
        # deliberately unconditional on the usage total below (a turn can
        # capture an exchange with no billable usage payload at all), so
        # this sits ahead of the usage-total early returns rather than
        # nested inside them.
        try:
            captures = list(stream_signals.exchange_captures())
            if captures:
                session_id = self.store.session_id_for_message(assistant_message_id)
                with self._capture_quiescence_lock:
                    if self.store.capture_quiescent(session_id):
                        captures = []
                    else:
                        self._capture_exchange_flush_sessions.add(session_id)
                try:
                    if captures:
                        self.store.attach_message_exchanges(
                            assistant_message_id, captures
                        )
                finally:
                    with self._capture_quiescence_lock:
                        self._capture_exchange_flush_sessions.discard(session_id)
        except Exception as exc:
            logger.bind(
                message_id=assistant_message_id,
                error_type=type(exc).__name__,
            ).warning(
                "exchange_attach_failed"
            )
        payloads = self._usage_payloads(stream_signals)
        provider = str(getattr(resolution, "provider", "") or "")
        model = str(getattr(resolution, "model", "") or "")
        total: ProviderUsage | None = None
        for payload in payloads:
            usage = ProviderUsage.from_provider_payload(
                payload,
                provider=provider,
                model=model,
                partial=partial,
            )
            if usage is None:
                continue
            total = usage if total is None else total.plus(usage)
        if total is None:
            return
        attached = False
        try:
            self.store.set_message_usage(assistant_message_id, total)
            attached = True
        except KeyError:
            pass
        except Exception as exc:
            # Broadened past KeyError (Qodo round): `set_message_usage` now
            # persists immediately for an already-terminal message (the
            # stop-path flush, see its own docstring), so any
            # SQLite/persistence exception raised from that flush would
            # otherwise escape into stop/cancel control flow -- exactly the
            # "never fail a send" contract this method promises. Swallow and
            # log instead; a dropped usage attach is a missing cost figure,
            # not a broken turn.
            logger.bind(message_id=assistant_message_id, error=repr(exc)).warning(
                "usage_attach_failed"
            )
        if not attached:
            return
        # Cost-ticker PR3: cache-TTL ground truth, Anthropic prompt-caching
        # only -- other providers/gateways never set `prompt_caching`, and a
        # non-Anthropic `provider` string has no cache-warm concept to
        # stamp. Same never-fail posture as the attach above: this is
        # read by the chip, never by send control flow.
        try:
            if provider_config_key(provider) == "anthropic" and getattr(
                resolution, "prompt_caching", None
            ):
                sid = self.store.session_id_for_message(assistant_message_id)
                had_cache_activity = (total.cache_read + total.cache_write) > 0
                self._cache_last_activity[sid] = had_cache_activity
                if had_cache_activity:
                    self._cache_warm_until[sid] = time.monotonic() + 300.0
        except Exception as exc:
            logger.bind(message_id=assistant_message_id, error=repr(exc)).warning(
                "cost_cache_ttl_record_failed"
            )

    def _teardown_refuses_turn(self, session_id: str | None) -> bool:
        """Whether teardown must refuse this session's turn before dispatch.

        App exit (``_disposed``) refuses everything. The PER-VISIT
        ``_shutdown_requested`` fence must not: ``leave_console``'s owner
        ruling keeps an in-flight ``AGENT_WAKE`` turn running headless
        ("cancelling it would re-create the exact 'only completes if you
        stay' gap this arc exists to close"), and
        ``_agent_wake_turn_sessions`` is the registry that ruling already
        uses to spare those sessions from the same method's cancel fan-out.
        Reading the flag alone refused the wake anyway, one layer down.

        Args:
            session_id: Session owning the turn, or None when unresolved.

        Returns:
            True when the turn must be refused.
        """

        if self._disposed:
            return True
        if not self._shutdown_requested.is_set():
            return False
        return session_id not in self._agent_wake_turn_sessions

    def _accepted_shutdown_before_dispatch(
        self, assistant_message_id: str, session_id: str
    ) -> ConsoleSubmitResult:
        """Settle an accepted owner without claiming an external dispatch."""

        try:
            self.store.mark_message_failed(assistant_message_id)
        except KeyError:
            return self._session_closed_result(session_id=session_id)
        self._set_run_state(
            ConsoleRunState(
                ConsoleRunStatus.STOPPED,
                "Console shut down before provider dispatch.",
            ),
            session_id=session_id,
        )
        return ConsoleSubmitResult(
            True,
            True,
            "Console shut down before provider dispatch.",
        )

    def _thinking_persistence_preflight(
        self,
        *,
        session_id: str,
        resolution: Any,
    ) -> ConsoleSubmitResult | None:
        """Reject an unsupported durable thinking stream without transcript writes."""
        try:
            require_thinking_persistence_support(
                self.store.persistence,
                persistent=(
                    self.store.persistence is not None
                    and not self.store.session_is_ephemeral(session_id)
                ),
                may_emit_thinking=bool(getattr(resolution, "may_emit_thinking", False)),
            )
        except ConsoleThinkingCompatibilityError as exc:
            visible_copy = str(exc)
            self._set_run_state(
                ConsoleRunState.blocked(visible_copy), session_id=session_id
            )
            return ConsoleSubmitResult(False, False, visible_copy)
        return None

    async def _run_direct_provider_reply(
        self,
        *,
        resolution: Any,
        provider_messages: list[dict[str, Any]],
        assistant_message_id: str,
        prepare_retry: bool,
        variant_mode: bool,
        prefill: str | None,
        prefill_from_one_shot: bool,
        one_shot_prefill_revision: int | None,
        citation_repair_session: ConsoleCitationRepairSession | None,
        stream_signals: ConsoleProviderStreamSignals | None,
        continuation_sidecar: tuple[ProviderContinuationSidecar, ...] = (),
        continuation_target: ContinuationRestoreTarget | None = None,
        character_emote_snapshot: CharacterEmoteRunSnapshot | None = None,
        preparation_id: str | None = None,
        generation_token: int | None = None,
    ) -> ConsoleSubmitResult:
        # Dev's citation-repair refactor extracted this streaming body out of
        # the wrapper (`_stream_assistant_response_inner`) into its own
        # method, which left it without the `owner_id` the wrapper already
        # resolved for ITSELF -- re-resolve independently here, same as
        # `_run_agent_reply` does for its own `session_id`, rather than
        # threading it through as a parameter (`None` on KeyError mirrors
        # every other guarded call site below: no owning session to
        # attribute a closed-session result to).
        try:
            owner_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            return self._session_closed_result()
        one_shot_used = one_shot_prefill_revision if prefill_from_one_shot else None
        if prefill:
            provider_messages = [
                *provider_messages,
                {
                    "role": ConsoleMessageRole.ASSISTANT.value,
                    "content": prefill,
                },
            ]
        dispatch_request: Any = provider_messages
        prepare_request = getattr(self.provider_gateway, "prepare_chat_request", None)
        if callable(prepare_request):
            dispatch_request = prepare_request(
                resolution,
                provider_messages,
                continuation_target=continuation_target,
                continuation_sidecar=continuation_sidecar,
                continuation_owner_key=(
                    NATIVE_MESSAGE_ID_KEY if continuation_sidecar else None
                ),
            )
            dropped_messages = int(
                getattr(dispatch_request, "dropped_messages", 0) or 0
            )
            if dropped_messages:
                self._append_history_trimmed_note(owner_id, dropped_messages)
        # Fix round 1 (Critical 1): a per-session cancel signal for this
        # direct/legacy stream path too, mirroring `_run_agent_reply`'s own
        # `cancel_event` -- the shared `_stop_requested` flag below is
        # GLOBAL (set by ANY session's Stop/Close via `_signal_stop`), so
        # reading it inside a specific run's own loop let stopping session
        # B silently truncate an untouched session A's still-streaming
        # reply. Captured by closure/local (not re-read off `self.
        # _active_cancel_events` each poll) for the same reason
        # `should_cancel` isn't: a concurrent NEXT run for this same
        # session (after this one's own finally already popped its entry)
        # must never be torn down by a stale reference to THIS run's event.
        cancel_event = threading.Event()
        self._active_cancel_events[owner_id] = cancel_event
        # Narrowed once, here, rather than inside the try below: every use
        # in this method (the gateway dispatch, the usage attachments, the
        # post-generation citation selection) then sees a real signals
        # object. `_stream_assistant_response_inner` always supplies one;
        # this belt keeps direct callers (tests) working.
        if stream_signals is None:
            stream_signals = self._new_run_stream_signals()
        stream_signals.model_retry_callback = lambda: self.store.record_trace_event(
            owner_id,
            anchor_message_id=assistant_message_id,
            event_kind="model_retry",
            summary="Provider stream retried as a non-streaming request",
            status="retrying",
        )
        require_thinking_persistence_support(
            self.store.persistence,
            persistent=(
                self.store.persistence is not None
                and not self.store.session_is_ephemeral(owner_id)
            ),
            may_emit_thinking=bool(getattr(resolution, "may_emit_thinking", False)),
        )
        if generation_token is None:
            generation_token = self.store.begin_generation_attempt(
                assistant_message_id
            )
        if variant_mode:
            self.store.begin_variant_stream(
                assistant_message_id,
                generation_token=generation_token,
            )
        if character_emote_snapshot is not None and not prepare_retry:
            self.store.begin_character_emote_capture(
                assistant_message_id,
                character_emote_snapshot,
            )
        if prefill and not prepare_retry:
            try:
                self.store.append_stream_chunk(assistant_message_id, prefill)
            except KeyError:
                return self._session_closed_result(session_id=owner_id)
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response."),
            session_id=owner_id,
        )
        retry_prepared = False
        emitted_content = False
        thinking_capture = ThinkingCapture(assistant_owner_id=assistant_message_id)

        def project_thinking(update: Any) -> None:
            if update.envelope is not None:
                self.store.replace_message_thinking(
                    assistant_message_id,
                    update.envelope,
                    generation_token=generation_token,
                )

        def settle_thinking(outcome: Literal["complete", "stopped", "failed"]) -> None:
            project_thinking(thinking_capture.settle(outcome))

        if prepare_retry:
            self.store.record_trace_event(
                owner_id,
                anchor_message_id=assistant_message_id,
                event_kind="message_retry_requested",
                summary="User requested another response attempt",
                status="started",
            )
        try:
            if self._teardown_refuses_turn(owner_id):
                return self._accepted_shutdown_before_dispatch(
                    assistant_message_id, owner_id
                )
            if preparation_id is not None and not self._transition_preparation(
                preparation_id,
                ConsoleTurnPreparationState.ACCEPTED,
                ConsoleTurnPreparationState.DISPATCH_STARTED,
            ):
                raise RuntimeError("Prepared turn changed before provider dispatch.")
            provider_stream = self.provider_gateway.stream_chat(
                resolution,
                dispatch_request,
                signals=stream_signals,
            )
            async for chunk in provider_stream:
                if not chunk:
                    continue
                thinking_event = isinstance(
                    chunk,
                    (ProviderThinkingDelta, ProviderProprietaryThinkingEvidence),
                )
                if thinking_event:
                    if prepare_retry and not retry_prepared:
                        self.store.prepare_message_retry(
                            assistant_message_id,
                            generation_token=generation_token,
                        )
                        retry_prepared = True
                        if character_emote_snapshot is not None:
                            self.store.begin_character_emote_capture(
                                assistant_message_id,
                                character_emote_snapshot,
                            )
                        if prefill:
                            try:
                                self.store.append_stream_chunk(
                                    assistant_message_id, prefill
                                )
                            except KeyError:
                                return self._session_closed_result(session_id=owner_id)
                    project_thinking(thinking_capture.observe(chunk))
                if cancel_event.is_set():
                    self.store.record_trajectory_timing(
                        assistant_message_id, model_status="cancelled"
                    )
                    self.store.record_trace_event(
                        owner_id,
                        anchor_message_id=assistant_message_id,
                        event_kind="model_cancelled",
                        summary="Provider request cancelled",
                        status="cancelled",
                    )
                    self._attach_stream_usage(
                        assistant_message_id, stream_signals, resolution, partial=True
                    )
                    settle_thinking("stopped")
                    try:
                        stopped = self._mark_stream_stopped(
                            assistant_message_id,
                            visible_copy="Response stopped.",
                            prepare_retry=prepare_retry,
                            retry_prepared=retry_prepared,
                        )
                    except KeyError:
                        return self._session_closed_result(session_id=owner_id)
                    self._consume_one_shot_prefill(assistant_message_id, one_shot_used)
                    return ConsoleSubmitResult(True, True, stopped.content)
                if thinking_event:
                    continue
                if type(chunk) is not str:
                    if prepare_retry and not retry_prepared:
                        self.store.prepare_message_retry(
                            assistant_message_id,
                            generation_token=generation_token,
                        )
                        retry_prepared = True
                        if character_emote_snapshot is not None:
                            self.store.begin_character_emote_capture(
                                assistant_message_id,
                                character_emote_snapshot,
                            )
                        if prefill:
                            try:
                                self.store.append_stream_chunk(
                                    assistant_message_id, prefill
                                )
                            except KeyError:
                                return self._session_closed_result(session_id=owner_id)
                    project_thinking(thinking_capture.observe(chunk))
                    continue
                if prepare_retry and not retry_prepared:
                    self.store.prepare_message_retry(
                        assistant_message_id,
                        generation_token=generation_token,
                    )
                    retry_prepared = True
                    if character_emote_snapshot is not None:
                        self.store.begin_character_emote_capture(
                            assistant_message_id,
                            character_emote_snapshot,
                        )
                    if prefill:
                        try:
                            self.store.append_stream_chunk(
                                assistant_message_id, prefill
                            )
                        except KeyError:
                            return self._session_closed_result(session_id=owner_id)
                thinking_capture.observe(chunk)
                try:
                    self.store.append_stream_chunk(assistant_message_id, chunk)
                except KeyError:
                    return self._session_closed_result(session_id=owner_id)
                if chunk:
                    emitted_content = True
            if cancel_event.is_set():
                self.store.record_trajectory_timing(
                    assistant_message_id, model_status="cancelled"
                )
                self.store.record_trace_event(
                    owner_id,
                    anchor_message_id=assistant_message_id,
                    event_kind="model_cancelled",
                    summary="Provider request cancelled",
                    status="cancelled",
                )
                self._attach_stream_usage(
                    assistant_message_id, stream_signals, resolution, partial=True
                )
                settle_thinking("stopped")
                try:
                    stopped = self._mark_stream_stopped(
                        assistant_message_id,
                        visible_copy="Response stopped.",
                        prepare_retry=prepare_retry,
                        retry_prepared=retry_prepared,
                    )
                except KeyError:
                    return self._session_closed_result(session_id=owner_id)
                self._consume_one_shot_prefill(assistant_message_id, one_shot_used)
                return ConsoleSubmitResult(True, True, stopped.content)
            if not emitted_content:
                self.store.record_trajectory_timing(
                    assistant_message_id, model_status="failed"
                )
                self.store.record_trace_event(
                    owner_id,
                    anchor_message_id=assistant_message_id,
                    event_kind="model_error",
                    summary="Provider stream ended without content",
                    status="failed",
                )
                try:
                    failed = self.store.get_message(assistant_message_id)
                except KeyError:
                    return self._session_closed_result(session_id=owner_id)
                self._set_run_state(
                    ConsoleRunState(
                        ConsoleRunStatus.FAILED,
                        "Provider stream ended without content.",
                    ),
                    session_id=owner_id,
                )
                # Billed-but-contentless turn (refusal/empty stream after
                # usage arrived): spec's "total = money actually spent" wins
                # over "failed sends produce no usage row", which covers
                # transport failures where nothing was billed. Attached
                # BEFORE the terminal mark so the mark flushes it.
                self._attach_stream_usage(
                    assistant_message_id, stream_signals, resolution, partial=True
                )
                settle_thinking("failed")
                if not prepare_retry or retry_prepared:
                    try:
                        failed = self.store.mark_message_failed(assistant_message_id)
                    except KeyError:
                        return self._session_closed_result(session_id=owner_id)
                return ConsoleSubmitResult(True, True, failed.content)
            if citation_repair_session is not None:
                try:
                    selection = await self._select_post_generation_body(
                        assistant_message_id=assistant_message_id,
                        repair_session=citation_repair_session,
                        stream_signals=stream_signals,
                    )
                except KeyError:
                    # F4 fix (Qodo wave): `owner_id` was already resolved
                    # above (line ~4290) and is in scope here, same as
                    # every other guarded call site in this method -- the
                    # bare no-arg call defaulted to whatever session is
                    # ACTIVE right now, wrongly stamping a STOPPED run
                    # state on an unrelated live session instead of this
                    # run's own (now-orphaned) one.
                    return self._session_closed_result(session_id=owner_id)
                if selection.state == "canceled":
                    self._consume_one_shot_prefill(
                        assistant_message_id,
                        one_shot_used,
                    )
                    return ConsoleSubmitResult(True, True, selection.selected_body)
            self.store.record_trajectory_timing(
                assistant_message_id, model_status="completed"
            )
            self._attach_stream_usage(
                assistant_message_id, stream_signals, resolution, partial=False
            )
            settle_thinking("complete")
            try:
                if variant_mode:
                    completed = self.store.finalize_variant_stream(assistant_message_id)
                else:
                    completed = self.store.mark_message_complete(assistant_message_id)
            except KeyError:
                return self._session_closed_result(session_id=owner_id)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."),
                session_id=owner_id,
            )
            self._consume_one_shot_prefill(assistant_message_id, one_shot_used)
            return ConsoleSubmitResult(True, True, completed.content)
        except asyncio.CancelledError:
            if cancel_event.is_set():
                self.store.record_trajectory_timing(
                    assistant_message_id, model_status="cancelled"
                )
                self.store.record_trace_event(
                    owner_id,
                    anchor_message_id=assistant_message_id,
                    event_kind="model_cancelled",
                    summary="Provider request cancelled",
                    status="cancelled",
                )
                self._attach_stream_usage(
                    assistant_message_id, stream_signals, resolution, partial=True
                )
                settle_thinking("stopped")
                try:
                    stopped = self._mark_stream_stopped(
                        assistant_message_id,
                        visible_copy="Response stopped.",
                        prepare_retry=prepare_retry,
                        retry_prepared=retry_prepared,
                    )
                except KeyError:
                    return self._session_closed_result(session_id=owner_id)
                self._consume_one_shot_prefill(assistant_message_id, one_shot_used)
                return ConsoleSubmitResult(True, True, stopped.content)
            raise
        except ConsoleDispatchSettlementError:
            raise
        except Exception as exc:
            # Provider failures are surfaced as run status plus a transcript
            # system row; they must never be written into assistant message
            # content, which is persisted and replayed as model context.
            visible_copy = f"Provider stream failed: {describe_stream_failure(exc)}"
            self.store.record_trajectory_timing(
                assistant_message_id, model_status="failed"
            )
            self.store.record_trace_event(
                owner_id,
                anchor_message_id=assistant_message_id,
                event_kind="model_error",
                summary="Provider request failed",
                status="failed",
            )
            try:
                settle_thinking("failed")
                if not prepare_retry or retry_prepared:
                    self.store.mark_message_failed(assistant_message_id)
                else:
                    self.store.get_message(assistant_message_id)
            except KeyError:
                return self._session_closed_result(session_id=owner_id)
            # Reuse the guarded owner_id resolved at the top of this method
            # (rather than re-deriving it) -- this is the run's OWNING
            # session regardless of whatever the user currently has open.
            self._append_failure_system_row(owner_id, visible_copy)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
                session_id=owner_id,
            )
            return ConsoleSubmitResult(True, True, visible_copy)
        finally:
            # Fix round 1 (Critical 1): this run's own per-session cancel
            # signal (created above, mirroring `_run_agent_reply`'s own)
            # must not survive the run -- a stale entry would let a LATER,
            # unrelated run on this same session inherit an already-set
            # Event. Not `cancel_event.clear()`: `_select_post_generation_
            # body` (already returned by the time this fires) captured this
            # by session-id lookup rather than closure, so identity-gated
            # pop (not reset-in-place) is what matches `_run_agent_reply`'s
            # own matching pop.
            if self._active_cancel_events.get(owner_id) is cancel_event:
                self._active_cancel_events.pop(owner_id, None)

    async def _select_post_generation_body(
        self,
        *,
        assistant_message_id: str,
        repair_session: ConsoleCitationRepairSession,
        stream_signals: ConsoleProviderStreamSignals,
    ) -> ConsoleCitationSelectionOutcome:
        """Select one bounded reply before terminal persistence."""

        try:
            initial_message = self.store.get_message(assistant_message_id)
            owner_session_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            return ConsoleCitationSelectionOutcome("", "unavailable")
        initial_body = initial_message.content

        def owns_request() -> bool:
            # Task 3b: check the OWNING session's own map entries, not a
            # global singular slot -- a concurrent session's own in-flight
            # stream/repair must never be mistaken for this one.
            if (
                self._active_assistant_message_ids.get(owner_session_id)
                != assistant_message_id
                or self._active_stream_tasks.get(owner_session_id)
                is not asyncio.current_task()
                or self._active_citation_repair_sessions.get(owner_session_id)
                is not repair_session
            ):
                return False
            try:
                return (
                    self.store.session_id_for_message(assistant_message_id)
                    == owner_session_id
                )
            except KeyError:
                return False

        def cancellation_requested() -> bool:
            # Fix round 1 (Critical 1): this run's own per-session cancel
            # signal, not the shared `_stop_requested` flag -- reading the
            # global flag here let an UNRELATED session's Stop/Close
            # silently cancel this session's still-running citation repair,
            # the exact hazard this fix closes for the sibling stream
            # loops. `_run_direct_provider_reply` registers this session's
            # `cancel_event` in `_active_cancel_events[owner_session_id]`
            # before ever calling this method.
            cancel_event = self._active_cancel_events.get(owner_session_id)
            return (
                repair_session.cancel_reason is not None
                and cancel_event is not None
                and cancel_event.is_set()
                and not repair_session.selection_committed
                and repair_session.phase in {"checking", "repair_streaming"}
            )

        def commit_canceled() -> ConsoleCitationSelectionOutcome:
            if not owns_request():
                visible_copy = (
                    SESSION_CLOSED_COPY
                    if repair_session.cancel_reason == "session_close"
                    else initial_body
                )
                return ConsoleCitationSelectionOutcome(
                    visible_copy,
                    "canceled",
                )
            try:
                current = self.store.get_message(assistant_message_id)
            except KeyError:
                return ConsoleCitationSelectionOutcome(
                    SESSION_CLOSED_COPY,
                    "canceled",
                )
            if current.content != initial_body:
                return ConsoleCitationSelectionOutcome(
                    current.content,
                    "canceled",
                )

            repair_session.phase = "selected"
            repair_session.selection_committed = True
            self.store.set_citation_presentation(
                assistant_message_id,
                ConsoleCitationPresentation(
                    phase=ConsoleCitationPhase.SELECTED,
                    notice_code=ConsoleCitationNoticeCode.CANCELED,
                    original_attempt_available=False,
                ),
            )
            completed = self.store.mark_message_complete(assistant_message_id)
            self._set_run_state(
                ConsoleRunState(
                    ConsoleRunStatus.STOPPED,
                    "Citation repair canceled.",
                ),
                session_id=owner_session_id,
            )
            if repair_session.cancel_reason == "user":
                try:
                    self.store.append_message(
                        owner_session_id,
                        role=ConsoleMessageRole.SYSTEM,
                        content="Citation repair canceled by user.",
                        persist=self.store.persistence is not None,
                    )
                except Exception:
                    logger.warning(
                        "Console citation repair cancellation record unavailable; "
                        "reason=citation_repair_cancel_record_persistence_failed"
                    )
            return ConsoleCitationSelectionOutcome(
                completed.content,
                "canceled",
            )

        def commit(
            state: Literal["valid", "repaired", "unavailable"],
            *,
            notice_code: ConsoleCitationNoticeCode | None = None,
            selected_body: str | None = None,
        ) -> ConsoleCitationSelectionOutcome:
            if cancellation_requested():
                return commit_canceled()
            if not owns_request():
                return ConsoleCitationSelectionOutcome(
                    initial_body,
                    "unavailable",
                )
            if selected_body is not None:
                try:
                    self.store.replace_deferred_terminal_body(
                        assistant_message_id,
                        selected_body,
                    )
                except ValueError:
                    state = "unavailable"
                    notice_code = ConsoleCitationNoticeCode.UNAVAILABLE
            repair_session.phase = "selected"
            repair_session.selection_committed = True
            self.store.set_citation_presentation(
                assistant_message_id,
                ConsoleCitationPresentation(
                    phase=ConsoleCitationPhase.SELECTED,
                    notice_code=notice_code,
                    original_attempt_available=state == "repaired",
                ),
            )
            if state == "repaired":
                self._remember_original_attempt(
                    assistant_message_id,
                    initial_body,
                    update_presentation=False,
                )
            selected = self.store.get_message(assistant_message_id)
            return ConsoleCitationSelectionOutcome(selected.content, state)

        if (
            not initial_body
            or stream_signals.synthetic_fallback_emitted
            or repair_session.selection_committed
        ):
            repair_session.phase = "selected"
            repair_session.selection_committed = True
            return ConsoleCitationSelectionOutcome(initial_body, "bypassed")

        contract = repair_session.contract
        resolution = repair_session.resolution
        if contract is None or resolution is None:
            return ConsoleCitationSelectionOutcome(initial_body, "unavailable")

        decision = decide_citation_repair(initial_body, contract)
        if decision is CitationRepairDecision.VALID:
            return commit("valid")
        if decision is CitationRepairDecision.UNAVAILABLE:
            return commit(
                "unavailable",
                notice_code=ConsoleCitationNoticeCode.UNAVAILABLE,
            )

        repair_session.phase = "checking"
        self.store.set_citation_presentation(
            assistant_message_id,
            ConsoleCitationPresentation(phase=ConsoleCitationPhase.CHECKING),
        )
        self._set_run_state(
            ConsoleRunState(
                ConsoleRunStatus.CHECKING_CITATIONS,
                "Checking citations…",
            ),
            session_id=owner_session_id,
        )
        repaired_chunks: list[str] = []
        repair_output_available = False
        try:
            await asyncio.sleep(0)
            if cancellation_requested():
                return commit_canceled()
            try:
                current_message = self.store.get_message(assistant_message_id)
            except KeyError:
                return ConsoleCitationSelectionOutcome(
                    SESSION_CLOSED_COPY,
                    "canceled",
                )
            if (
                not owns_request()
                or current_message.content != initial_body
                or repair_session.attempt_started
                or stream_signals.synthetic_fallback_emitted
            ):
                return commit(
                    "unavailable",
                    notice_code=ConsoleCitationNoticeCode.UNAVAILABLE,
                )

            repair_messages = build_citation_repair_messages(
                contract,
                initial_body,
            )
            if repair_messages is None or not repair_request_fits_model_window(
                repair_messages,
                initial_answer=initial_body,
                model=resolution.model or "",
                provider=resolution.provider,
                max_tokens=resolution.max_tokens,
            ):
                return commit(
                    "unavailable",
                    notice_code=ConsoleCitationNoticeCode.UNAVAILABLE,
                )
            if cancellation_requested():
                return commit_canceled()

            repair_session.attempt_started = True
            repair_session.phase = "repair_streaming"
            self.store.set_citation_presentation(
                assistant_message_id,
                ConsoleCitationPresentation(phase=ConsoleCitationPhase.REPAIRING),
            )

            repaired_size = 0
            repair_output_available = True
            async for chunk in self.provider_gateway.stream_chat(
                resolution,
                repair_messages,
                signals=stream_signals,
            ):
                if cancellation_requested():
                    repaired_chunks.clear()
                    return commit_canceled()
                if type(chunk) is not str:
                    repair_output_available = False
                    break
                if not chunk:
                    continue
                try:
                    repaired_size += len(chunk.encode("utf-8"))
                except UnicodeEncodeError:
                    repair_output_available = False
                    break
                if repaired_size > REPAIR_ANSWER_BODY_UTF8_BYTES_MAX:
                    repair_output_available = False
                    break
                repaired_chunks.append(chunk)
        except asyncio.CancelledError:
            if cancellation_requested():
                return commit_canceled()
            raise
        except Exception:
            repair_output_available = False

        if cancellation_requested():
            repaired_chunks.clear()
            return commit_canceled()
        if not repair_output_available or stream_signals.synthetic_fallback_emitted:
            return commit(
                "unavailable",
                notice_code=ConsoleCitationNoticeCode.UNAVAILABLE,
            )

        repaired_body = "".join(repaired_chunks)
        selected = select_repaired_body(
            initial_body,
            repaired_body,
            contract,
        )
        if not selected.repaired:
            return commit(
                "unavailable",
                notice_code=ConsoleCitationNoticeCode.UNAVAILABLE,
            )
        return commit(
            "repaired",
            notice_code=ConsoleCitationNoticeCode.REPAIRED,
            selected_body=selected.selected_body,
        )

    async def _run_agent_reply(
        self,
        *,
        resolution: Any,
        provider_messages: list[dict[str, Any]],
        assistant_message_id: str,
        prepare_retry: bool,
        variant_mode: bool,
        skill_bindings: tuple[str, ...] = (),
        skill_bundle_block: str = "",
        citation_repair_session: ConsoleCitationRepairSession | None = None,
        stream_signals: ConsoleProviderStreamSignals | None = None,
        turn_context: ConsoleTurnExecutionContext | None = None,
        restore_provider_continuation: ProviderContinuationCheckpoint | None = None,
        restore_provider_target: ContinuationRestoreTarget | None = None,
        expand_provider_continuation: (
            Callable[[ProviderContinuationCheckpoint], list[dict]] | None
        ) = None,
        resume_provider_continuation: bool = False,
        continuation_sidecar: tuple[ProviderContinuationSidecar, ...] = (),
        continuation_history_target: ContinuationRestoreTarget | None = None,
        preparation_id: str | None = None,
        generation_token: int | None = None,
    ) -> ConsoleSubmitResult:
        """Run the agent loop as the reply engine, streaming into the target row."""
        logger.info(
            "console agent reply start",
            assistant_message_id=assistant_message_id,
            variant_mode=variant_mode,
            prepare_retry=prepare_retry,
        )
        # Resolve the run's OWNING session FIRST (Task 3b): every write
        # below -- the per-session stream/cancel maps AND run state -- must
        # target it explicitly rather than whatever the user currently has
        # open (parallel-agents spec §2). Moved ahead of those writes
        # (previously ran after them, back when they were single shared
        # slots with no session to key by).
        try:
            session_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            return self._session_closed_result()
        turn_context = self._require_complete_turn_execution_context(turn_context)
        if turn_context.session_id != session_id:
            raise ValueError("Console turn context does not own the assistant row.")
        scratch_snapshot = turn_context.scratch_space
        if scratch_snapshot is None:
            return self._block(session_id, "Private scratch space is unavailable.")
        thinking_block = self._thinking_persistence_preflight(
            session_id=session_id,
            resolution=resolution,
        )
        if thinking_block is not None:
            return thinking_block
        scratch_lease = functools.partial(
            self._scratch_spaces.lease,
            scratch_snapshot,
        )
        session = next((s for s in self.store.sessions() if s.id == session_id), None)
        startup_candidate: StartupInstructionCandidate | None = None
        project_selection: ProjectInstructionBindingSelection | None = None
        confirm_project_dispatch = None
        project_authority_guard = None
        project_activation_callback = None
        if (
            session is not None
            and session.project_instruction_state.project_instructions_enabled
        ):
            try:
                registry = getattr(self.app, "workspace_registry_service", None)
            except Exception:
                registry = None
            try:
                project_selection = resolve_project_instruction_binding(
                    session, registry
                )
            except ProjectInstructionBindingRecovery as exc:
                expected_setup_state = session.project_instruction_state
                try:
                    options = list_project_instruction_bindings(session, registry)
                except ProjectInstructionBindingRecovery:
                    options = ()
                # An unselected session with no usable folder is a valid
                # scratch-only Chat/Workspace, not a project-instruction
                # setup failure. Keep the optional feature armed so a folder
                # added later can be selected, but do not block this send or
                # ask for one now. A previously selected folder still fails
                # closed and reaches recovery when it disappears or changes.
                if (
                    expected_setup_state.working_folder_binding_id is None
                    and not options
                ):
                    project_selection = None
                else:
                    callback = self._select_project_instruction_binding
                    if callback is None:
                        return ConsoleSubmitResult(False, False, str(exc))
                    action, binding_id = await callback(session_id, options, str(exc))
                    action, project_selection = (
                        commit_project_instruction_setup_decision(
                            store=self.store,
                            session_id=session_id,
                            registry=registry,
                            expected_state=expected_setup_state,
                            expected_options=options,
                            action=action,
                            binding_id=binding_id,
                        )
                    )
                    if action == "disable":
                        self._clear_project_instruction_delivery(session_id)
                        return self._block(session_id, "project_instructions_disabled")
                    if action != "select" or project_selection is None:
                        self._clear_project_instruction_delivery(session_id)
                        return ConsoleSubmitResult(False, False, str(exc))
            if project_selection is not None:
                state = session.project_instruction_state
                if state.working_folder_binding_id is None:
                    state = ProjectInstructionControlState(
                        project_instructions_enabled=True,
                        working_folder_binding_id=(
                            project_selection.binding.binding_id
                        ),
                        working_folder_locator_fingerprint=(
                            project_selection.locator_fingerprint
                        ),
                        project_instruction_notice_key=None,
                    )
                    self.store.set_session_project_instruction_state(session_id, state)
                startup_candidate = ProjectInstructionResolver().resolve_startup(
                    binding_id=project_selection.binding.binding_id,
                    binding_root=project_selection.root,
                    locator_fingerprint=project_selection.locator_fingerprint,
                    max_bytes=coerce_int_setting(
                        get_cli_setting(
                            "console",
                            "project_instructions_startup_max_bytes",
                            DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                        ),
                        DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                        minimum=MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                        maximum=MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                    ),
                    dispatch_started_wall_ns=time.time_ns(),
                )
                destination_provider = str(
                    getattr(resolution, "execution_key", "")
                    or getattr(resolution, "provider", "")
                    or "agent"
                )
                destination_endpoint = getattr(resolution, "base_url", None)
                notice_key = project_instruction_notice_key(
                    project_selection.locator_fingerprint,
                    destination_provider,
                    destination_endpoint,
                )
                expected_project_state = state

                def on_owning_loop(callback):
                    call_from_thread = getattr(self.app, "call_from_thread", None)
                    if callable(call_from_thread):
                        return call_from_thread(callback)
                    return callback()

                def project_activation_callback(event):
                    return on_owning_loop(
                        lambda: self._record_project_instruction_activation(
                            session_id, event
                        )
                    )

                def project_authority_guard():
                    return bool(
                        on_owning_loop(
                            lambda: project_instruction_authority_is_current(
                                store=self.store,
                                session_id=session_id,
                                registry=registry,
                                expected_selection=project_selection,
                            )
                        )
                    )

                def confirm_project_dispatch(snapshot):
                    def commit_and_record(decision):
                        committed = commit_project_instruction_dispatch_decision(
                            store=self.store,
                            session_id=session_id,
                            registry=registry,
                            expected_state=expected_project_state,
                            expected_selection=project_selection,
                            notice_key=notice_key,
                            decision=decision,
                        )
                        if committed != "prompt":
                            if committed == "proceed":
                                self._remember_project_instruction_delivery(
                                    session_id, snapshot
                                )
                            else:
                                self._clear_project_instruction_delivery(session_id)
                        return committed

                    initial = on_owning_loop(lambda: commit_and_record(None))
                    if initial != "prompt":
                        return initial
                    callback = self._confirm_project_instruction_dispatch
                    notice = build_project_instruction_dispatch_notice(
                        snapshot,
                        session_id=session_id,
                        resolution=resolution,
                    )
                    decision = callback(notice) if callback is not None else "cancel"
                    if decision not in {"proceed", "cancel", "disable"}:
                        decision = "cancel"
                    return on_owning_loop(lambda: commit_and_record(decision))

        self._active_assistant_message_ids[session_id] = assistant_message_id
        self._active_stream_tasks[session_id] = asyncio.current_task()
        self._stop_requested = False
        self._mcp_provider = None
        # A fresh per-run Event, captured by `should_cancel` below by
        # closure (not read off `self` each time) -- see
        # `_active_cancel_events`'s docstring for why this, rather than
        # `_stop_requested` alone, is what makes a still-running bridge
        # thread observe a Stop correctly (task-227).
        cancel_event = threading.Event()
        self._active_cancel_events[session_id] = cancel_event
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Agent running."),
            session_id=session_id,
        )
        if generation_token is None:
            generation_token = self.store.begin_generation_attempt(
                assistant_message_id
            )
        if variant_mode:
            self.store.begin_variant_stream(
                assistant_message_id,
                generation_token=generation_token,
            )
        elif prepare_retry:
            self.store.prepare_message_retry(
                assistant_message_id,
                generation_token=generation_token,
            )

        # Split the leading session system message off the payload; the
        # agent config carries it (composed with the operating prompt).
        session_system_prompt = ""
        agent_messages = list(provider_messages)
        if (
            agent_messages
            and agent_messages[0].get("role") == ConsoleMessageRole.SYSTEM.value
        ):
            session_system_prompt = str(agent_messages[0].get("content", ""))
            agent_messages = agent_messages[1:]

        conversation_id = self._agent_conversation_id(session_id)
        # noqa: E731 — tiny closure. Fix round 1 (Critical 1): reads ONLY
        # `cancel_event` -- captured by value, not via `self.
        # _active_cancel_events[session_id]` -- never the shared
        # `_stop_requested` flag. `_stop_requested` is GLOBAL (set by ANY
        # session's Stop/Close via `_signal_stop`), so OR'ing it in here
        # let stopping an unrelated session silently cancel THIS run too.
        # `cancel_event` alone is still correct once this run's own
        # `finally` below has already reset `_stop_requested` while the
        # bridge's background thread is still running (task-227: an
        # `asyncio.to_thread` call survives Task cancellation, so the
        # coroutine can finish handling a Stop and reset its own shared
        # bookkeeping well before the OS thread it detached from actually
        # returns) -- `stop_active_run`/`close_session`/`shutdown` all set
        # THIS session's `cancel_event` via `_signal_stop(session_id=...)`
        # the moment Stop is requested, and nothing ever clears it again
        # for this run, so a late poll from the surviving thread still
        # sees the cancellation regardless of `_stop_requested`'s state.
        should_cancel = lambda: cancel_event.is_set()  # noqa: E731

        # P5-T6: compose this run's MCP tool provider (if eligible) HERE,
        # on the running main loop, BEFORE the bridge is dispatched onto
        # asyncio.to_thread below -- see `_compose_mcp_provider`'s own
        # docstring for why `compose_catalog()`'s async I/O can never run
        # from the worker thread. `(None, None)` (no service, kill switch
        # on, or nothing composed) leaves the bridge's MCP-free path
        # byte-identical to before this task.
        #
        # task-545/T6: `_compose_mcp_provider`'s own `mcp_review_hook`
        # (built from `build_mcp_review_hook`) is deliberately discarded
        # here rather than wired -- it is `None` whenever MCP is not
        # eligible for this run, and built-in tools (calculator/datetime
        # today) must be gated regardless of MCP eligibility. Changing
        # `_compose_mcp_provider`'s own return contract to drop that
        # second element was considered and rejected: several existing
        # test suites (`Tests/Chat/test_console_agent_swap.py`,
        # `Tests/UI/test_console_internals_decomposition.py`) assert its
        # exact `(provider, hook)` / `(None, None)` shape directly and sit
        # outside this task's file scope, so keeping that function
        # byte-identical and building the run-level hook separately here
        # is the lower-blast-radius choice.
        (
            mcp_provider,
            builtin_gate,
            local_provider,
            local_review_hook,
        ) = await self._compose_agent_request_providers(
            session_id=session_id,
            project_selection=project_selection,
            project_authority_guard=project_authority_guard,
            turn_context=turn_context,
        )
        self._mcp_provider = mcp_provider

        # task-545/T6: build THIS run's built-in permission gate and hand
        # the SAME instance to both the review hook (below) and
        # `ConsoleAgentBridge.run_reply` (which threads it into the
        # `BuiltinToolProvider` that actually invokes tools) -- a second,
        # independently-built gate would silently desynchronize stamps:
        # a decision made here would never be visible to `invoke()`'s own
        # gate, and vice versa. `build_builtin_gate(None)` (no
        # `unified_mcp_service` on the app) is fail-closed-correct, not
        # "ungated" -- see that function's own docstring.
        # Only `.tool_for(name)` is used by the review hook below, to
        # resolve a `ToolCall.name` to the `Tool` object `builtin_gate.
        # resolve` needs -- this instance is never used to invoke a tool,
        # so it does not need to be the SAME `BuiltinToolProvider` object
        # the bridge's registry actually dispatches through (its `_tools`
        # dict is stateless data rebuilt identically by any instance).
        # Round 1 review CRITICAL 1: resolve THIS run's OWN workspace id --
        # the SAME lookup `ConsoleAgentBridge.run_reply` makes
        # (`self._store.session_workspace_id(session_id)`) for the real
        # `BuiltinToolProvider(workspace_id=...)` dispatch below -- and
        # thread it into the review hook so its `path_precheck_failed`
        # pre-flight resolves the IDENTICAL workspace dispatch will, never
        # whatever happens to be active in the UI for a parked/background
        # session. `KeyError` (an already-closed session) degrades to
        # `None`, matching `allowed_file_roots`'s own fail-safe posture.
        try:
            review_workspace_id = self.store.session_workspace_id(session_id)
        except KeyError:
            review_workspace_id = None
        builtin_review_provider = BuiltinToolProvider(
            gate=builtin_gate,
            workspace_id=review_workspace_id,
            sandbox_root=scratch_snapshot.root,
            sandbox_lease=scratch_lease,
        )
        # Task 9: bind THIS run's owning session id into the approval
        # bridge so `request_mcp_approvals` can (a) scope its cancellation
        # check to this run's own cancel event rather than falling back to
        # whichever session is currently VIEWED (finding #1), and (b) park
        # rather than mount when `session_id` is not the active session.
        review_hook = build_tool_review_hook(
            builtin_gate,
            builtin_review_provider,
            mcp_provider,
            functools.partial(self.request_mcp_approvals, session_id=session_id),
            workspace_id=review_workspace_id,
            # TASK-631: the switch must cover the tool families NEITHER
            # provider claims (skills/spawn/find/load), and this hook is the
            # only choke point they all pass. Read fresh per turn so a
            # mid-run flip takes effect on the next batch. Absent service ->
            # no switch to honor (None), matching `_compose_mcp_provider`.
            kill_switch=self._console_tool_kill_switch_reader(),
        )

        # Local tools (ADR-032): same per-run composition point. Both
        # hooks see every batch; each gates only what its provider owns,
        # so the combined hook is a collision-free merge.
        if local_review_hook is not None:
            review_hook = build_combined_review_hook([review_hook, local_review_hook])

        # task-1337: THIS run's Library retrieval provider (direct tools or
        # the bounded RAG fallback), resolved ONCE here on the main loop via
        # the injected factory -- a raising factory degrades to None (no
        # Library tools this run) rather than breaking the send.
        library_provider: Any | None = None
        library_provider_authority: Any | None = None
        if self._library_provider_factory is not None:
            try:
                library_selection = self._library_provider_for_context(turn_context)
                if library_selection is not None:
                    library_provider, library_provider_authority = library_selection
            except Exception:  # noqa: BLE001 -- never block a send
                logger.opt(exception=True).warning(
                    "library_provider_factory failed; running without Library tools"
                )

        # TASK-1971 (Agent Change Review): THIS run's tracked roots -- the
        # same workspace folder bindings the file tools resolve against.
        # Best-effort: an unavailable registry yields no roots and an
        # untracked (but otherwise normal) run.
        change_roots: list = [Path(root) for root in turn_context.workspace_roots]

        # Swap site: the agent loop runs synchronously on a worker thread via
        # asyncio.to_thread, so Stop is cooperative-only -- `should_cancel` is
        # polled between chunks/steps inside the bridge, never preempts the
        # thread itself. A provider that hangs mid-request without emitting a
        # single chunk cannot be interrupted here; RunBudget.max_wall_seconds
        # (agent_models.py) is what bounds a run overall, but only once
        # control returns to a checkpoint the loop actually polls -- it is
        # not a hard timeout on an in-flight, zero-chunk provider call.
        if self._teardown_refuses_turn(session_id):
            return self._accepted_shutdown_before_dispatch(
                assistant_message_id, session_id
            )
        if preparation_id is not None and not self._transition_preparation(
            preparation_id,
            ConsoleTurnPreparationState.ACCEPTED,
            ConsoleTurnPreparationState.DISPATCH_STARTED,
        ):
            raise RuntimeError("Prepared turn changed before provider dispatch.")
        try:
            # run_reply returns (run_id, outcome): run_id lets us write the
            # produced reply's PERSISTED id back onto the run after
            # completion (the load-bearing write for resume marker anchoring).
            run_id, outcome = await asyncio.to_thread(
                self._agent_bridge.run_reply,
                conversation_id=conversation_id,
                session_id=session_id,
                resolution=resolution,
                assistant_message_id=assistant_message_id,
                model=(
                    getattr(resolution, "model", None)
                    or turn_context.effective_model
                    or ""
                ),
                session_system_prompt=session_system_prompt,
                agent_messages=agent_messages,
                should_cancel=should_cancel,
                provider_stream_signals=stream_signals,
                supersede_previous=bool(prepare_retry or variant_mode),
                mcp_provider=mcp_provider,
                builtin_gate=builtin_gate,
                scratch_root=scratch_snapshot.root,
                scratch_lease=scratch_lease,
                review_tool_calls=review_hook,
                local_provider=local_provider,
                library_provider=library_provider,
                library_authority=library_provider_authority,
                native_tools_enabled=bool(
                    turn_context.tool_configuration.get(
                        "native_tool_calls_enabled", True
                    )
                ),
                change_roots=change_roots,
                turn_skill_bindings=skill_bindings,
                turn_bundle_block=skill_bundle_block,
                request_skill_install_confirm=functools.partial(
                    self.request_skill_install_confirm, session_id=session_id
                ),
                startup_instruction_candidate=startup_candidate,
                confirm_project_instruction_dispatch=confirm_project_dispatch,
                on_project_instruction_activation=project_activation_callback,
                # Advertised must equal usable (the #847 lesson, restated in
                # the run_skill_script docstring below): only pass the
                # confirm callback -- and therefore only let the bridge
                # build/advertise the run_skill_script tool at all -- once a
                # UI sink is actually wired. Until then
                # `request_skill_script_confirm`'s own no-UI guard would
                # auto-deny every call, offering the model a tool it can
                # never successfully use.
                request_skill_script_confirm=(
                    functools.partial(
                        self.request_skill_script_confirm, session_id=session_id
                    )
                    if self.set_pending_skill_script is not None
                    else None
                ),
                # PR2a Task 7: the fleet cancels/abandons children on the
                # bridge's worker thread and hands each stopped run's id
                # here, so a card still on screen for a child that is
                # already `cancelled` is denied and cleared rather than
                # left pressable (an approval that would still EXECUTE the
                # tool for real). Run-keyed, so a live sibling child --
                # which shares this same session -- keeps its own card.
                revoke_approvals=self.revoke_approval_rounds_for_run,
                restore_provider_continuation=restore_provider_continuation,
                restore_provider_target=restore_provider_target,
                expand_provider_continuation=expand_provider_continuation,
                resume_provider_continuation=resume_provider_continuation,
                continuation_sidecar=continuation_sidecar,
                continuation_target=continuation_history_target,
                continuation_owner_key=(
                    NATIVE_MESSAGE_ID_KEY if continuation_sidecar else None
                ),
                generation_token=generation_token,
            )
        except asyncio.CancelledError:
            if cancel_event.is_set():
                # Whatever the provider already billed for this turn's
                # completed steps is real money -- record it (partial)
                # before the terminal mark, exactly as the direct path's
                # own CancelledError branch does.
                self._attach_stream_usage(
                    assistant_message_id, stream_signals, resolution, partial=True
                )
                # PR3a-1 Task 6b (audit F3): a Stop ends the TURN, not its
                # surviving children -- so this branch needs the same
                # post-turn watch the normal finalizer sets. PR3a-2 Task 3:
                # message + partial flag ride along so the drain fold can
                # re-attach to this row with the same (partial) semantics.
                self._watch_post_turn_usage(
                    session_id,
                    stream_signals,
                    resolution,
                    assistant_message_id=assistant_message_id,
                    partial=True,
                )
                try:
                    stopped = self._mark_stream_stopped(
                        assistant_message_id, visible_copy="Response stopped."
                    )
                except KeyError:
                    return self._session_closed_result(session_id=session_id)
                # task-543: this is the dominant user-Stop path --
                # ``task.cancel()`` raised before ``(run_id, outcome)`` ever
                # bound, so recover the active run's id via the bridge's
                # latest-unanchored-primary lookup and record the stopped
                # reply's persisted id, same as every finalizer terminal
                # path. A never-persisted stop (or an anchored/missing row)
                # no-ops and leaves the row NULL -> ordinal fallback.
                self._record_run_assistant_message(
                    self._latest_unanchored_primary_run_id(conversation_id),
                    stopped,
                )
                return ConsoleSubmitResult(True, True, stopped.content)
            raise
        except ConsoleDispatchSettlementError:
            raise
        except Exception as exc:
            # Bridge failures can originate OUTSIDE AgentService's own
            # narrow loop guard (agent_service.py wraps only
            # `run_agent_loop`; `db.create_run`, `_persist`
            # (append_steps/set_status), and `supersede_run_tree` are not
            # covered). Left uncaught here, run_state would stay STREAMING
            # forever and every future send on every session would be
            # rejected ("A run is already running in this tab.") until app
            # restart (Plan-B Task 6 Critical 1). Mirror the legacy stream
            # path's catch-all above, including the Task-1 variant-restore
            # semantics: `begin_variant_stream`/`prepare_message_retry`
            # already ran before the bridge call, so `mark_message_failed`
            # resolves the correct terminal content on its own (restores
            # the pre-regenerate base + status for a failed regenerate;
            # preserves whatever partial content already streamed
            # otherwise).
            visible_copy = f"Agent run failed: {describe_stream_failure(exc)}"
            if getattr(
                getattr(exc, "response", None), "status_code", None
            ) is not None and self._session_history_carries_images(session_id):
                visible_copy += self._IMAGE_REJECTION_RECOVERY_HINT
            try:
                self.store.mark_message_failed(assistant_message_id)
            except KeyError:
                return self._session_closed_result(session_id=session_id)
            self._append_failure_system_row(session_id, visible_copy)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
                session_id=session_id,
            )
            return ConsoleSubmitResult(True, True, visible_copy)
        finally:
            # Task 3b: this finally intentionally pops NONE of the three
            # per-session entries (stream task, assistant message id,
            # cancel event) -- `_finalize_agent_reply` below (and, through
            # it, `_finalize_agent_success`'s citation-repair post-
            # generation check) still runs AFTER this try/finally, on this
            # SAME task, and both `owns_request()` (stream task/assistant
            # message id) and `cancellation_requested()` (cancel_event,
            # NOT clear()'d for the same reason noted where it was created
            # above -- task-227) need to see this run as still live and
            # still cancellable. The wrapper (`_stream_assistant_response_
            # inner`), which awaits this entire call including
            # `_finalize_agent_reply`, is what clears every per-session
            # entry once everything has actually finished.
            run_state = self.run_state_for(session_id)
            logger.info(
                "console agent reply end",
                assistant_message_id=assistant_message_id,
                run_status=run_state.status.value,
                run_copy=run_state.visible_copy,
            )

        # Captured here, before `_finalize_agent_reply` runs: this run's own
        # cancel_event is the authority on whether IT was stopped,
        # independent of what status `mark_message_stopped` may have left
        # the message at (task-227 AC3 follow-up -- see the guard below).
        return await self._finalize_agent_reply(
            assistant_message_id,
            session_id,
            outcome,
            variant_mode=variant_mode,
            cancel_event=cancel_event,
            run_id=run_id,
            citation_repair_session=citation_repair_session,
            stream_signals=stream_signals,
            resolution=resolution,
        )

    def _agent_conversation_id(self, session_id: str) -> str:
        """Return the durable id the run store is keyed by (persisted id when set)."""
        for session in self.store.sessions():
            if session.id == session_id:
                return session.persisted_conversation_id or session_id
        return session_id

    async def _finalize_agent_reply(
        self,
        assistant_message_id: str,
        session_id: str,
        outcome: Any,
        *,
        variant_mode: bool,
        cancel_event: threading.Event | None = None,
        run_id: str | None = None,
        citation_repair_session: ConsoleCitationRepairSession | None = None,
        stream_signals: ConsoleProviderStreamSignals | None = None,
        resolution: Any = None,
    ) -> ConsoleSubmitResult:
        from tldw_chatbook.Agents.agent_models import RUN_CANCELLED, RUN_DONE

        current = self._ensure_assistant_placeholder(assistant_message_id, session_id)
        # task-227 LOW-2 (+ AC3 follow-up): a Stop can land in the
        # ultra-narrow window after asyncio.to_thread returns an outcome
        # but before this method runs. `current.status == "stopped"` alone
        # only catches a plain send/retry -- `mark_message_stopped`
        # (console_chat_store.py) RESTORES a mid-regenerate message to its
        # *prior* status (e.g. "complete"), not "stopped", so that check
        # never fires for a stopped regenerate. Trust the run's own
        # per-run `cancel_event` instead: it is set by `_signal_stop` the
        # instant Stop is requested and never cleared for this run, so
        # `.is_set()` is true here if and only if THIS run was stopped --
        # regardless of which status `mark_message_stopped` left the
        # message at. Every branch below would otherwise either raise via
        # _validate_can_mark_terminal (mark_message_complete /
        # mark_message_failed) or silently resurrect the message back to
        # "complete" with a phantom variant (finalize_variant_stream,
        # which has no such guard at all). The `current.status`
        # comparison stays as a belt for any future caller that reaches
        # this method without a `cancel_event` in scope. Stop already won
        # and settled the message (mark_message_stopped's own restore --
        # prior status for a regenerate, "stopped" for a plain send) and
        # the variant base (already popped), so this is a benign no-op
        # read-back, never an error, in either case.
        stopped_now = (current is not None and current.status == "stopped") or (
            cancel_event is not None and cancel_event.is_set()
        )
        # F1: the agent path's single usage attachment point. EVERY branch
        # below settles the placeholder terminal (complete / stopped /
        # failed), and usage must be on the message BEFORE that mark so the
        # mark flushes it to persistence -- so attach once, here, ahead of
        # all of them. `partial` is true for anything that is not a clean
        # RUN_DONE: a stopped or errored turn still cost real money for the
        # provider calls it did make, but its output side is incomplete.
        # A no-op when the run captured no usage at all (best-effort: absent
        # usage must never fail a send).
        attach_partial = stopped_now or getattr(outcome, "status", None) != RUN_DONE
        self._attach_stream_usage(
            assistant_message_id,
            stream_signals,
            resolution,
            partial=attach_partial,
        )
        # PR3a-1 Task 6b (audit F3): mark where THIS turn's billing stopped,
        # so a surviving child's later provider calls are readable rather
        # than silently dropped -- see `_watch_post_turn_usage`. PR3a-2
        # Task 3: message + partial flag ride along so the drain fold can
        # re-attach to this row with the same semantics.
        self._watch_post_turn_usage(
            session_id,
            stream_signals,
            resolution,
            assistant_message_id=assistant_message_id,
            partial=attach_partial,
        )
        if stopped_now:
            # The stopped message was already persisted by
            # `mark_message_stopped` (`_persist_existing_message`), so its
            # durable persisted id is available NOW -- record it onto the run
            # so resume can anchor markers by it. Without this the run keeps
            # whatever `create_run` stored (a stale native id pre-fix, NULL
            # post-fix); a never-persisted stop leaves `current` without a
            # persisted id and the helper no-ops (row stays NULL -> ordinal
            # fallback -- correct).
            self._record_run_assistant_message(run_id, current)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STOPPED, "Response stopped."),
                session_id=session_id,
            )
            return ConsoleSubmitResult(
                True, True, current.content if current is not None else ""
            )

        if outcome.status == RUN_CANCELLED:
            return self._finalize_agent_cancelled(
                assistant_message_id,
                session_id,
                variant_mode=variant_mode,
                run_id=run_id,
            )

        if outcome.status != RUN_DONE:
            return self._finalize_agent_failure(
                assistant_message_id,
                session_id,
                outcome,
                variant_mode=variant_mode,
                run_id=run_id,
            )

        return await self._finalize_agent_success(
            assistant_message_id,
            session_id,
            outcome,
            variant_mode=variant_mode,
            run_id=run_id,
            citation_repair_session=citation_repair_session,
            stream_signals=stream_signals,
        )

    def _ensure_assistant_placeholder(
        self,
        assistant_message_id: str,
        session_id: str,
    ) -> ConsoleChatMessage | None:
        """Return the assistant placeholder message if it still exists.

        ``KeyError`` means the session/placeholder was closed/removed mid-run;
        ``None`` is returned so callers can recover by appending a fresh
        assistant message instead of aborting the whole turn.
        """
        try:
            return self.store.get_message(assistant_message_id)
        except KeyError:
            return None

    def _find_runtime_written_assistant(
        self,
        session_id: str,
    ) -> ConsoleChatMessage | None:
        """Return the most recent assistant message in ``session_id``, if any."""
        try:
            messages = self.store.messages_for_session(session_id)
        except KeyError:
            return None
        for message in reversed(messages):
            if message.role is ConsoleMessageRole.ASSISTANT:
                return message
        return None

    def _complete_agent_message(
        self,
        assistant_message_id: str,
        variant_mode: bool,
        outcome: Any,
    ) -> ConsoleChatMessage:
        """Finalize a placeholder, applying the empty-final-text fallback.

        The fallback text is streamed into the placeholder so the store's
        existing persistence/validation paths stay unchanged.
        """
        final_text = getattr(outcome, "final_text", "")
        settled_reader = getattr(
            self.store,
            "provider_continuation_terminal_message",
            None,
        )
        if callable(settled_reader):
            settled = settled_reader(
                assistant_message_id,
                expected_content=final_text,
            )
            if settled is not None:
                return settled
        if not final_text:
            self.store.clear_terminal_citation_state(assistant_message_id)
            self.store.append_stream_chunk(
                assistant_message_id,
                "No response was generated.",
            )
        if variant_mode:
            return self.store.finalize_variant_stream(assistant_message_id)
        return self.store.mark_message_complete(assistant_message_id)

    def _finalize_agent_cancelled(
        self,
        assistant_message_id: str,
        session_id: str,
        *,
        variant_mode: bool,
        run_id: str | None = None,
    ) -> ConsoleSubmitResult:
        """Handle a ``RUN_CANCELLED`` outcome: the placeholder becomes ``failed``.

        Per the agent turn-control spec, a runtime-reported cancellation is a
        terminal failure, not a user-initiated stop. If the placeholder has
        vanished, append a failed assistant message carrying the visible copy.
        The terminal message (``mark_message_failed``/``_append_failed_assistant``,
        both persisted) has its durable id recorded onto the run so resume can
        anchor markers by it; a never-persisted reply no-ops (row stays NULL ->
        ordinal fallback -- see ``_record_run_assistant_message``).
        """
        visible_copy = "Response stopped/cancelled."
        placeholder = self._ensure_assistant_placeholder(
            assistant_message_id, session_id
        )
        if placeholder is not None:
            failed = self.store.mark_message_failed(assistant_message_id)
        else:
            failed = self._append_failed_assistant(session_id, visible_copy)
        self._record_run_assistant_message(run_id, failed)
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
            session_id=session_id,
        )
        return ConsoleSubmitResult(True, True, failed.content)

    def _finalize_agent_failure(
        self,
        assistant_message_id: str,
        session_id: str,
        outcome: Any,
        *,
        variant_mode: bool,
        run_id: str | None = None,
    ) -> ConsoleSubmitResult:
        """Handle ``RUN_ERROR``, ``RUN_STUCK``, or any unknown non-done outcome.

        A present placeholder is marked ``failed`` and a system row explains
        the failure (preserving the existing failure UX). If the placeholder
        is missing, the runtime may have already written an assistant message
        (e.g. streamed partial content before the error); use it when
        possible, otherwise append a new failed assistant message.

        Whichever terminal message resolves (all persisted via
        ``mark_message_failed``/``_append_failed_assistant``) has its durable id
        recorded onto the run so resume can anchor markers by it; a
        never-persisted reply no-ops (row stays NULL -> ordinal fallback -- see
        ``_record_run_assistant_message``).
        """
        visible_copy = self._agent_failure_visible_copy(outcome)
        if "provider returned HTTP" in visible_copy and (
            self._session_history_carries_images(session_id)
        ):
            visible_copy += self._IMAGE_REJECTION_RECOVERY_HINT
        placeholder = self._ensure_assistant_placeholder(
            assistant_message_id, session_id
        )
        if placeholder is not None:
            failed = self.store.mark_message_failed(assistant_message_id)
            self._record_run_assistant_message(run_id, failed)
            self._append_failure_system_row(session_id, visible_copy)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
                session_id=session_id,
            )
            return ConsoleSubmitResult(True, True, failed.content)

        runtime_written = self._find_runtime_written_assistant(session_id)
        if runtime_written is not None and runtime_written.status in {
            "pending",
            "streaming",
        }:
            self.store.append_stream_chunk(runtime_written.id, f"\n\n{visible_copy}")
            failed = self.store.mark_message_failed(runtime_written.id)
        else:
            failed = self._append_failed_assistant(session_id, visible_copy)
        self._record_run_assistant_message(run_id, failed)
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
            session_id=session_id,
        )
        return ConsoleSubmitResult(True, True, failed.content)

    async def _finalize_agent_success(
        self,
        assistant_message_id: str,
        session_id: str,
        outcome: Any,
        *,
        variant_mode: bool,
        run_id: str | None = None,
        citation_repair_session: ConsoleCitationRepairSession | None = None,
        stream_signals: ConsoleProviderStreamSignals | None = None,
    ) -> ConsoleSubmitResult:
        """Handle ``RUN_DONE``: complete the placeholder (or a runtime-written one).

        An empty ``final_text`` is replaced with the fallback copy ``No
        response was generated.``. If the placeholder is missing, the runtime
        may have streamed content into an assistant row already; complete it
        when possible, otherwise append a new assistant message.

        Once the reply is completed (and its durable ``persisted_message_id``
        assigned), that persisted id is written back onto the agent run via
        ``_record_run_assistant_message`` -- the load-bearing correction of
        the native id ``create_run`` recorded, which resume anchors markers by.
        """
        placeholder = self._ensure_assistant_placeholder(
            assistant_message_id, session_id
        )
        if placeholder is not None:
            if (
                citation_repair_session is not None
                and stream_signals is not None
                and placeholder.content
                and bool(getattr(outcome, "final_text", ""))
                and not stream_signals.synthetic_fallback_emitted
            ):
                try:
                    selection = await self._select_post_generation_body(
                        assistant_message_id=assistant_message_id,
                        repair_session=citation_repair_session,
                        stream_signals=stream_signals,
                    )
                except KeyError:
                    # F4 fix (Qodo wave): `session_id` is a REQUIRED
                    # parameter of this method (always known, never
                    # re-derived) -- the bare no-arg call defaulted to
                    # whatever session is ACTIVE right now, wrongly
                    # stamping a STOPPED run state on an unrelated live
                    # session instead of this run's own owning session.
                    return self._session_closed_result(session_id=session_id)
                if selection.state == "canceled":
                    completed = self._ensure_assistant_placeholder(
                        assistant_message_id,
                        session_id,
                    )
                    if completed is None:
                        return self._session_closed_result(session_id=session_id)
                    self._record_run_assistant_message(run_id, completed)
                    return ConsoleSubmitResult(
                        True,
                        True,
                        selection.selected_body,
                    )
            completed = self._complete_agent_message(
                assistant_message_id, variant_mode, outcome
            )
            self._record_run_assistant_message(run_id, completed)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."),
                session_id=session_id,
            )
            return ConsoleSubmitResult(True, True, completed.content)

        runtime_written = self._find_runtime_written_assistant(session_id)
        if runtime_written is not None and runtime_written.status in {
            "pending",
            "streaming",
        }:
            completed = self._complete_agent_message(
                runtime_written.id, variant_mode=False, outcome=outcome
            )
            self._record_run_assistant_message(run_id, completed)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."),
                session_id=session_id,
            )
            return ConsoleSubmitResult(True, True, completed.content)

        final_text = getattr(outcome, "final_text", "") or "No response was generated."
        completed = self.store.append_message(
            session_id, role=ConsoleMessageRole.ASSISTANT, content=final_text
        )
        self._record_run_assistant_message(run_id, completed)
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."),
            session_id=session_id,
        )
        return ConsoleSubmitResult(True, True, completed.content)

    def _record_run_assistant_message(
        self,
        run_id: str | None,
        completed: ConsoleChatMessage,
    ) -> None:
        """Write the completed reply's PERSISTED id onto the agent run.

        On resume, markers anchor by matching a transcript message's durable
        ``persisted_message_id``; the id recorded at ``create_run`` time is
        the native in-memory id (the reply was not persisted yet), so it must
        be corrected here, once the reply has its persisted id. A no-op when
        there is no run id, no bridge, or no persistence (the native id would
        be useless to resume). Never fails the turn -- a marker-anchoring
        bookkeeping write, wrapped defensively like the file's other seams.
        """
        persisted = getattr(completed, "persisted_message_id", None)
        if not run_id or persisted is None or self._agent_bridge is None:
            return
        try:
            self._agent_bridge.record_run_assistant_message(run_id, persisted)
        except Exception:  # noqa: BLE001 -- bookkeeping must never fail the turn
            logger.opt(exception=True).warning(
                "failed to record persisted assistant id on agent run",
                run_id=run_id,
                persisted_message_id=persisted,
            )

    def _latest_unanchored_primary_run_id(self, conversation_id: str) -> str | None:
        """Return the active run's id for the stopped-via-cancel path.

        task-543: thin defensive wrapper over the bridge's
        ``latest_unanchored_primary_run_id`` (see its docstring for the
        NULL-anchor guard) -- a bookkeeping lookup on the Stop path must
        never fail the stop itself.

        Args:
            conversation_id: Durable conversation id whose runs to inspect.

        Returns:
            The recoverable run id, or ``None`` when there is no bridge, no
            matching unanchored primary run, or the lookup fails.
        """
        if self._agent_bridge is None:
            return None
        try:
            return self._agent_bridge.latest_unanchored_primary_run_id(conversation_id)
        except Exception:  # noqa: BLE001 -- bookkeeping must never fail the stop
            logger.opt(exception=True).warning(
                "failed to look up unanchored primary run for stop recording",
                conversation_id=conversation_id,
            )
            return None

    def _append_failed_assistant(
        self,
        session_id: str,
        visible_copy: str,
    ) -> ConsoleChatMessage:
        """Append a failed assistant message carrying ``visible_copy``.

        The store's terminal-status validation only accepts pending/streaming
        assistant messages, so the message is created empty, the copy is
        streamed in, and then it is marked failed.
        """
        message = self.store.append_message(
            session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        self.store.append_stream_chunk(message.id, visible_copy)
        return self.store.mark_message_failed(message.id)

    @staticmethod
    def _agent_failure_visible_copy(outcome: Any) -> str:
        """Return user-facing copy for a non-done agent outcome, naming the reason.

        ``RUN_STUCK`` in particular must read as visibly distinct from a
        generic failure -- it means the run hit a budget or loop-detection
        limit (agent_runtime.py), not a raw exception -- so the concrete
        reason recorded on the last ``STEP_ERROR`` step (e.g. "step budget
        exhausted", "model-turn budget exhausted", "wall-clock budget
        exhausted", or the loop-guard's own user-facing "Agent stopped:
        ..." copy -- TASK-1231/F3 AC4) is surfaced when available.
        """
        from tldw_chatbook.Agents.agent_models import RUN_STUCK, STEP_ERROR

        reason = ""
        for step in reversed(getattr(outcome, "steps", None) or []):
            if getattr(step, "kind", None) == STEP_ERROR and getattr(
                step, "summary", ""
            ):
                reason = step.summary
                break
        if outcome.status == RUN_STUCK:
            if reason.startswith("Agent stopped:"):
                # Round 1 review (Minor): the loop-guard's own copy
                # (agent_runtime.py) already reads as a complete,
                # user-facing sentence -- prefixing "Agent run stuck: "
                # here would double the lead-in ("Agent run stuck: Agent
                # stopped: ...").
                return reason
            return f"Agent run stuck: {reason or 'budget or loop limit reached'}."
        return f"Agent run failed: {reason or outcome.status}."

    def _presentation_context_for(self, session_id: str) -> ConsolePresentationContext:
        """Return one session's presentation context with a safe global fallback."""
        try:
            global_default = self._global_user_display_name()
        except Exception as exc:
            logger.warning(
                "Console global user display-name accessor failed (error_type={}).",
                type(exc).__name__,
            )
            global_default = "User"
        return self.store.presentation_context(session_id, global_default)

    def _presentation_for(
        self, session_id: str, message: ConsoleChatMessage
    ) -> ConsoleMessagePresentation:
        """Resolve one provider-facing message from its live session identity."""
        return resolve_console_message_presentation(
            message, self._presentation_context_for(session_id)
        )

    def _presented_message_snapshots(
        self,
        session_id: str,
        messages: Iterable[ConsoleChatMessage],
    ) -> list[ConsoleChatMessage]:
        """Deep-copy transcript rows and apply their current visible content."""
        copied_messages = copy.deepcopy(list(messages))
        for copied_message in copied_messages:
            copied_message.content = self._presentation_for(
                session_id, copied_message
            ).content
        return copied_messages

    def _context_content_for(
        self,
        session_id: str,
        message: ConsoleChatMessage,
        *,
        fallback: str,
    ) -> str:
        """Project only explicitly trusted template content for model context."""
        metadata = message.metadata
        if (
            metadata is not None
            and metadata.template_kind == "character_greeting"
            and isinstance(metadata.template_source, str)
            and metadata.template_source.strip()
        ):
            return self._presentation_for(session_id, message).content
        return fallback

    def _resolved_system_prompt(self, session_id: str | None) -> str | None:
        """Resolve a trusted character system template for the current identity."""
        if session_id is None:
            return self.system_prompt
        session = next(
            (
                candidate
                for candidate in self.store.sessions()
                if candidate.id == session_id
            ),
            None,
        )
        if (
            session is None
            or session.assistant_kind != "character"
            or not isinstance(session.character_name, str)
            or not session.character_name.strip()
            or not isinstance(session.character_system_template, str)
            or not session.character_system_template.strip()
        ):
            return self.system_prompt
        context = self._presentation_context_for(session_id)
        return expand_character_template(
            session.character_system_template,
            user_name=context.user_name,
            character_name=session.character_name.strip(),
        )

    def _character_emote_authority(
        self, session_id: str
    ) -> _CharacterEmoteAuthority | None:
        """Return the character identity fence currently owning ``session_id``."""

        session = next(
            (candidate for candidate in self.store.sessions() if candidate.id == session_id),
            None,
        )
        if session is None or session.assistant_kind != "character":
            return None
        return _CharacterEmoteAuthority(
            identity_revision=session.identity_revision,
            runtime_backend=session.runtime_backend,
            assistant_id=session.assistant_id,
            assistant_authority_id=session.assistant_authority_id,
            local_character_id=session.local_character_id(),
        )

    @staticmethod
    def _build_character_emote_snapshot(
        authority: _CharacterEmoteAuthority,
        graph: Mapping[str, Any] | None,
        *,
        fallback_reason: str,
    ) -> CharacterEmoteRunSnapshot:
        """Project one validated active graph into bounded run-local identities."""

        if graph is None:
            return CharacterEmoteRunSnapshot(
                actor_id=authority.local_character_id,
                fallback_reason=fallback_reason,
            )
        try:
            pack = graph["pack"]
            version = graph["version"]
            raw_assets = tuple(graph["assets"])
            pack_id = int(pack["id"])
            pack_version_id = int(version["id"])
            if pack_id < 1 or pack_version_id < 1:
                raise ValueError
            # TASK-22227: one O(assets) pass replaces the per-state singleton
            # re-projection (which was O(assets^2) regex evaluations per send).
            sources = project_character_emote_assets(raw_assets)
            assets: list[CharacterEmoteAssetReference] = []
            for state, source in sources.items():
                if not isinstance(source, Mapping):
                    continue
                asset_id = source.get("id")
                expression_key = source.get("expression_key")
                if (
                    isinstance(asset_id, bool)
                    or not isinstance(asset_id, int)
                    or asset_id < 1
                    or not isinstance(expression_key, str)
                ):
                    continue
                assets.append(
                    CharacterEmoteAssetReference(
                        state=state,
                        expression_key=expression_key,
                        asset_id=asset_id,
                    )
                )
            return CharacterEmoteRunSnapshot(
                actor_id=authority.local_character_id,
                pack_id=pack_id,
                pack_version_id=pack_version_id,
                states=tuple(asset.state for asset in assets),
                assets=tuple(assets),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return CharacterEmoteRunSnapshot(
                actor_id=authority.local_character_id,
                fallback_reason="resolver_error",
            )

    async def _character_emote_snapshot_for_run(
        self, session_id: str
    ) -> CharacterEmoteRunSnapshot | None:
        """Read and revalidate one immutable character-emote run authority."""

        initial = self._character_emote_authority(session_id)
        if initial is None:
            return None
        for _attempt in range(2):
            authority = self._character_emote_authority(session_id)
            if authority is None:
                raise _CharacterEmoteAuthorityChanged
            graph: Mapping[str, Any] | None = None
            fallback_reason = "no_active_pack"
            repository = self._visual_identity_repository
            if authority.local_character_id is not None and repository is not None:
                try:
                    graph = await asyncio.to_thread(
                        repository.get_active_actor_pack,
                        "character",
                        authority.local_character_id,
                    )
                except Exception:
                    logger.warning("character_emote_snapshot_read_failed")
                    fallback_reason = "resolver_error"
            if self._character_emote_authority(session_id) != authority:
                continue
            return self._build_character_emote_snapshot(
                authority,
                graph,
                fallback_reason=fallback_reason,
            )
        raise _CharacterEmoteAuthorityChanged

    @staticmethod
    def _apply_character_emote_prompt(
        provider_messages: list[dict[str, Any]],
        snapshot: CharacterEmoteRunSnapshot,
    ) -> list[dict[str, Any]]:
        """Compose the pinned instruction without mutating stored settings."""

        messages = [dict(row) for row in provider_messages]
        if messages and messages[0].get("role") == ConsoleMessageRole.SYSTEM.value:
            messages[0]["content"] = append_character_emote_prompt_instruction(
                str(messages[0].get("content", "")),
                snapshot.states,
            )
            return messages
        instruction = append_character_emote_prompt_instruction("", snapshot.states)
        return [
            {"role": ConsoleMessageRole.SYSTEM.value, "content": instruction},
            *messages,
        ]

    def _leading_system_message(
        self,
        *,
        greeting: str = "",
        session_id: str | None = None,
        turn_context: (
            ConsoleTurnConfigurationSnapshot | ConsoleTurnExecutionContext | None
        ) = None,
    ) -> list[dict[str, str]]:
        """Return a single-item system message list when a system prompt is set.

        Applies to every native Console send path (submit, retry, regenerate,
        continue) since they all build their provider payload by prepending
        this to the transcript-derived messages. Blank/whitespace-only prompts
        are treated as "no system prompt" (native Console default stays silent
        unless a user has explicitly set one for this session) -- ``strip()``
        is used ONLY for that emptiness check. The message content itself is
        ``self.system_prompt`` verbatim: leading/trailing whitespace and
        internal formatting (blank lines, indentation) are never altered, so
        a formatting-sensitive prompt reaches the provider unchanged.

        Args:
            greeting: Seeded assistant greeting to fold after the prompt
                (task-1531); a non-blank greeting produces a system row even
                when no system prompt is set, since the message array itself
                must stay user-first for strict providers (task-427).
        """
        raw_system_prompt = (
            turn_context.provider_selection.system_prompt
            if turn_context is not None
            else self._resolved_system_prompt(session_id)
        )
        if not isinstance(raw_system_prompt, str) or not raw_system_prompt.strip():
            raw_system_prompt = ""
        content = fold_greeting_into_system_prompt(raw_system_prompt, greeting)
        if not content.strip():
            return []
        return [{"role": ConsoleMessageRole.SYSTEM.value, "content": content}]

    def _seeded_greeting_text(
        self,
        session_id: str,
        session_messages: list[ConsoleChatMessage],
    ) -> str:
        """Return the text of leading assistant turns (the seeded greeting).

        Mirrors ``_provider_message_payloads``'s leading-assistant drop rule:
        every ASSISTANT message before the first USER turn is excluded from
        the message array, so its text must travel in the system row instead.
        Failed messages are skipped to match ``skip_failed`` send payloads.
        """
        collected: list[str] = []
        for message in session_messages:
            if message.role is ConsoleMessageRole.USER:
                break
            if message.role is not ConsoleMessageRole.ASSISTANT:
                continue
            if message.status == "failed":
                continue
            text = self._context_content_for(
                session_id,
                message,
                fallback=message.content or "",
            ).strip()
            if text:
                collected.append(text)
        return "\n\n".join(collected)

    def _apply_context_summary_compaction(
        self, session_id: str, provider_messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fold the session's boundary summary into ``provider_messages``.

        THE LEAK RULE (spec-review fix): compaction applies ONLY when the
        boundary USER message is actually PRESENT in this payload. When present,
        the payload rows BEFORE it are dropped and the summary is appended to
        the leading system prefix (which ``bound_messages_to_window`` preserves).
        When ABSENT -- e.g. regenerating a message that sits BEFORE the boundary,
        whose ancestors-only payload ends pre-boundary -- the payload is returned
        untouched: a summary covering LATER turns must never be substituted into
        an earlier point's context.

        Payload-row -> boundary matching mechanism: match by native message
        IDENTITY, not by content. Send-path payload builds thread each row's
        source transcript id onto it (``annotate_ids=True`` ->
        ``NATIVE_MESSAGE_ID_KEY``); the boundary is the row whose id equals the
        stored ``boundary_native_id``. The transform pipeline between build and
        this choke point only ever rewrites/drops the FINAL user turn (skill
        fork drops leading rows; chat-dictionary/world-info AND skill-
        substitution's own inline rewrites -- leading-mention replace and
        embedded-mention splice -- rewrite the last user row via ``{**row}``
        spreads that PRESERVE the key) and appends a synthesized continuation
        turn (no key) -- so every earlier row, and thus any strictly-earlier
        boundary, keeps its id intact.

        This is the genuine fail-safe: if the boundary id is not present on any
        row -- because the boundary sits after the payload's end
        (pre-boundary regenerate/retry/continue/edit-resend), or a branch
        switch/deletion made it dangling, or the payload was built WITHOUT id
        annotation -- NOTHING matches and the FULL history is sent unchanged.
        A byte-identical earlier duplicate of the boundary's text (e.g. a repeat
        "continue"/"yes") can no longer false-fire the way first-occurrence
        content matching did, so the summary of LATER turns is never injected
        into an EARLIER point's context.

        Args:
            session_id: Session owning the payload being dispatched.
            provider_messages: The fully-built, post-transform payload
                (id-annotated on the send path).

        Returns:
            The compacted payload, or ``provider_messages`` unchanged.
        """
        summary, boundary_native_id = self.store.session_context_summary(session_id)
        if not summary or boundary_native_id is None:
            return provider_messages

        boundary_index: int | None = None
        for index, row in enumerate(provider_messages):
            if row.get(NATIVE_MESSAGE_ID_KEY) == boundary_native_id:
                boundary_index = index
                break
        if boundary_index is None:
            return provider_messages

        sys_end = 0
        while (
            sys_end < len(provider_messages)
            and provider_messages[sys_end].get("role")
            == ConsoleMessageRole.SYSTEM.value
        ):
            sys_end += 1
        system_prefix = provider_messages[:sys_end]
        tail = provider_messages[boundary_index:]

        summary_suffix = "\n\n[Summary of earlier conversation]\n" + summary
        if system_prefix:
            first = system_prefix[0]
            merged_first = {
                **first,
                "content": (first.get("content") or "") + summary_suffix,
            }
            new_system = [merged_first, *system_prefix[1:]]
        else:
            new_system = [
                {
                    "role": ConsoleMessageRole.SYSTEM.value,
                    "content": summary_suffix.lstrip(),
                }
            ]
        return new_system + tail

    def _provider_messages_for_session(
        self,
        session_id: str,
        *,
        before_message_id: str | None = None,
        annotate_ids: bool = False,
        turn_context: ConsoleTurnExecutionContext | None = None,
    ) -> list[dict[str, Any]]:
        collected: list[ConsoleChatMessage] = []
        for message in self.store.messages_for_session(session_id):
            if message.id == before_message_id:
                break
            collected.append(message)
        return self._leading_system_message(
            greeting=self._seeded_greeting_text(session_id, collected),
            session_id=session_id,
            turn_context=turn_context,
        ) + self._provider_message_payloads(
            collected,
            skip_failed=True,
            annotate_ids=annotate_ids,
            session_id=session_id,
            turn_context=turn_context,
        )

    def provider_messages_for_next_send_estimate(
        self, session_id: str
    ) -> ConsoleNextSendHistoryProjection:
        """Project canonical next-send history without mutation or serialization.

        Args:
            session_id: Session whose next-send provider history to project.

        Returns:
            Detached role/text rows and the admitted historical-media count.

        Raises:
            KeyError: If ``session_id`` does not identify a stored session.
        """
        configuration = self.resolve_turn_configuration_snapshot(session_id)
        collected = self.store.read_only_messages_for_session(session_id)
        system_rows = self._leading_system_message(
            greeting=self._seeded_greeting_text(session_id, collected),
            session_id=session_id,
            turn_context=configuration,
        )
        history_rows = self._lightweight_provider_message_rows(
            collected,
            skip_failed=True,
            session_id=session_id,
            turn_context=configuration,
        )
        return ConsoleNextSendHistoryProjection(
            rows=tuple(
                [(row["role"], row["content"]) for row in system_rows]
                + [(row.role, row.text) for row in history_rows]
            ),
            historical_media_count=sum(len(row.attachments) for row in history_rows),
        )

    def _provider_continuation_sidecar_for_session(
        self, session_id: str
    ) -> tuple[ProviderContinuationSidecar, ...]:
        """Capture private checkpoints for assistant owners on the active path."""
        active_ids = set(self.store.active_path_message_ids(session_id))
        return tuple(
            ProviderContinuationSidecar(message.id, message.provider_continuation)
            for message in self.store.messages_for_session(session_id)
            if message.id in active_ids
            and message.role is ConsoleMessageRole.ASSISTANT
            and isinstance(
                message.provider_continuation, ProviderContinuationCheckpoint
            )
        )

    def _provider_continuation_history_for_resolution(
        self, session_id: str, resolution: Any
    ) -> tuple[
        tuple[ProviderContinuationSidecar, ...], ContinuationRestoreTarget | None
    ]:
        """Select only complete private groups matching this frozen send."""
        sidecar = self._provider_continuation_sidecar_for_session(session_id)
        if not sidecar:
            return (), None
        if any(item.checkpoint.state != "complete" for item in sidecar):
            raise ContinuationConflictError(
                "Active continuation requires explicit recovery."
            )
        target = _continuation_restore_target_for_resolution(resolution)
        if target is None:
            return (), None
        owner_ids = {
            group.owner_message_id
            for group in provider_continuation_owner_groups(sidecar, target=target)
        }
        return (
            tuple(item for item in sidecar if item.owner_message_id in owner_ids),
            target,
        )

    def _provider_continuation_resume_history_for_resolution(
        self,
        session_id: str,
        resolution: Any,
        *,
        before_message_id: str,
    ) -> tuple[
        tuple[ProviderContinuationSidecar, ...], ContinuationRestoreTarget | None
    ]:
        """Select policy-retained completed history before an active owner."""
        target = _continuation_restore_target_for_resolution(resolution)
        if target is None:
            return (), None
        # TASK-19170: preserved-thinking retention follows the versioned kimi
        # reasoning family (probe-verified reasoning_content across the
        # family), not the kimi-k3 literal.
        keep_all = target.provider == "moonshot" and (
            moonshot_model_returns_reasoning_content(target.model)
        )
        keep_tool_history = target.provider == "deepseek"
        if not keep_all and not keep_tool_history:
            return (), None

        retained: list[ProviderContinuationSidecar] = []
        active_ids = set(self.store.active_path_message_ids(session_id))
        for message in self.store.messages_for_session(session_id):
            if message.id == before_message_id:
                break
            checkpoint = message.provider_continuation
            if (
                message.id not in active_ids
                or message.role is not ConsoleMessageRole.ASSISTANT
                or not isinstance(checkpoint, ProviderContinuationCheckpoint)
                or checkpoint.state != "complete"
            ):
                continue
            sidecar = ProviderContinuationSidecar(message.id, checkpoint)
            if not provider_continuation_owner_groups((sidecar,), target=target):
                continue
            if keep_all or any(round_.calls for round_ in checkpoint.rounds):
                retained.append(sidecar)
        return (tuple(retained), target) if retained else ((), None)

    def _provider_messages_through_message(
        self,
        session_id: str,
        message_id: str,
        *,
        annotate_ids: bool = False,
        turn_context: ConsoleTurnExecutionContext | None = None,
    ) -> list[dict[str, Any]]:
        collected: list[ConsoleChatMessage] = []
        for message in self.store.messages_for_session(session_id):
            collected.append(message)
            if message.id == message_id:
                break
        return self._leading_system_message(
            greeting=self._seeded_greeting_text(session_id, collected),
            session_id=session_id,
            turn_context=turn_context,
        ) + self._provider_message_payloads(
            collected,
            skip_failed=False,
            use_variant_content=True,
            annotate_ids=annotate_ids,
            session_id=session_id,
            turn_context=turn_context,
        )

    def _lightweight_provider_message_rows(
        self,
        session_messages: list[ConsoleChatMessage],
        *,
        skip_failed: bool,
        use_variant_content: bool = False,
        session_id: str | None = None,
        turn_context: (
            ConsoleTurnConfigurationSnapshot | ConsoleTurnExecutionContext | None
        ) = None,
    ) -> list[_LightweightProviderHistoryRow]:
        """Apply provider-history admission without serializing media bytes."""
        selection = (
            turn_context.provider_selection
            if turn_context is not None
            else self._provider_selection()
        )
        model = selection.explicit_model or selection.configured_model
        vision = (
            bool(turn_context.capabilities.get("vision", False))
            if turn_context is not None
            else bool(model) and is_vision_capable(selection.provider, model or "")
        )

        # Reserve the image budget newest-message-first, counting IMAGES (not
        # messages): a message with several attachments can consume more than
        # one unit of budget, and the walk stops as soon as the budget is
        # exhausted regardless of how many messages remain.
        budget = (
            int(turn_context.capabilities.get("max_history_images", 0) or 0)
            if turn_context is not None and vision
            else max_history_images(selection.provider, model)
            if vision
            else 0
        )
        allowed_counts: dict[str, int] = {}
        for message in reversed(session_messages):
            if budget <= 0:
                break
            if message.role is not ConsoleMessageRole.USER:
                continue
            if skip_failed and message.status == "failed":
                # A send-blocked echo keeps its attachment data but is dropped
                # from the emitted payload below (skip_failed); it must not
                # reserve image budget a real message would then lose (TASK-457
                # code-review finding 2).
                continue
            if _is_empty_transcript_row(message):
                # task-2391: an empty-transcript row never carries real
                # attachments (it is a bare text placeholder), but excluding
                # it here too keeps this loop's skip set identical to the
                # emit loop's below -- same reasoning as skip_failed just
                # above.
                continue
            usable = [
                attachment
                for attachment in message.attachments
                if attachment.data is not None
            ]
            if not usable:
                continue
            take = min(len(usable), budget)
            allowed_counts[message.id] = take
            budget -= take

        rows: list[_LightweightProviderHistoryRow] = []
        seen_user = False
        for message in session_messages:
            if message.role not in {
                ConsoleMessageRole.USER,
                ConsoleMessageRole.ASSISTANT,
            }:
                continue
            if skip_failed and message.status == "failed":
                continue
            if _is_empty_transcript_row(message):
                # task-2391: this row's content is a placeholder
                # ("(no speech detected)") written so a committed voice
                # turn with no words could still be durably created -- it
                # is UI chrome, not something the user said, and must never
                # be narrated to the model as a real turn (mirrors the
                # exclusion the realtime reseed builder already applies at
                # reconnect, `_console_realtime_seed_items`). Skipped
                # entirely rather than emitted as empty text, same as a
                # failed row above.
                continue
            # A seeded character greeting must not ride in the message array:
            # strict providers (Anthropic, Gemini) reject an assistant-first
            # array (task-427). Its text still reaches the provider -- the
            # send seams fold it into the system row via
            # ``_seeded_greeting_text`` (task-1531).
            if not seen_user and message.role is ConsoleMessageRole.ASSISTANT:
                continue
            if message.role is ConsoleMessageRole.USER:
                seen_user = True
            if (
                message.role is ConsoleMessageRole.ASSISTANT
                and not assistant_state_allows_provider_history(
                    state=message.assistant_generation_state,
                    has_valid_continuation=(
                        isinstance(
                            message.provider_continuation,
                            ProviderContinuationCheckpoint,
                        )
                        and message.provider_continuation.state == "active"
                    ),
                    content=message.content,
                )
            ):
                continue
            base_text = (
                message.variants.current.content
                if use_variant_content and message.variants is not None
                else message.content
            )
            text = (
                self._context_content_for(
                    session_id,
                    message,
                    fallback=base_text,
                )
                if session_id is not None
                else base_text
            )
            take = allowed_counts.get(message.id, 0)
            if take > 0:
                # Partially-budgeted messages retain their images in POSITION
                # order up to the reserved count (oldest-attached first),
                # not in reservation order.
                usable = [
                    attachment
                    for attachment in message.attachments
                    if attachment.data is not None
                ]
                rows.append(
                    _LightweightProviderHistoryRow(
                        source_message_id=message.id,
                        role=message.role.value,
                        text=text,
                        attachments=tuple(usable[:take]),
                    )
                )
                continue
            if not text:
                # An image-only user turn whose images all fell outside the
                # budget (over-cap, or a non-vision model) must not vanish —
                # a silently dropped turn distorts the conversation shape the
                # model sees. Emit a text placeholder instead.
                omitted = [
                    attachment
                    for attachment in message.attachments
                    if attachment.data is not None
                ]
                if message.role is ConsoleMessageRole.USER and omitted:
                    placeholder = (
                        "[image omitted]"
                        if len(omitted) == 1
                        else f"[{len(omitted)} images omitted]"
                    )
                    rows.append(
                        _LightweightProviderHistoryRow(
                            source_message_id=message.id,
                            role=message.role.value,
                            text=placeholder,
                        )
                    )
                continue
            rows.append(
                _LightweightProviderHistoryRow(
                    source_message_id=message.id,
                    role=message.role.value,
                    text=text,
                )
            )
        return rows

    def _provider_message_payloads(
        self,
        session_messages: list[ConsoleChatMessage],
        *,
        skip_failed: bool,
        use_variant_content: bool = False,
        annotate_ids: bool = False,
        session_id: str | None = None,
        turn_context: ConsoleTurnExecutionContext | None = None,
    ) -> list[dict[str, Any]]:
        lightweight_rows = self._lightweight_provider_message_rows(
            session_messages,
            skip_failed=skip_failed,
            use_variant_content=use_variant_content,
            session_id=session_id,
            turn_context=turn_context,
        )
        payloads: list[dict[str, Any]] = []
        for lightweight in lightweight_rows:
            content: Any = lightweight.text
            if lightweight.attachments:
                parts: list[dict[str, Any]] = []
                if lightweight.text:
                    parts.append({"type": "text", "text": lightweight.text})
                for attachment in lightweight.attachments:
                    # Resumed rows can have an empty persisted mime type. Keep
                    # dispatch's existing valid data-URI fallback at the
                    # serialization boundary, never in the estimate path.
                    parts.append(
                        image_url_part(
                            attachment.data, attachment.mime_type or "image/png"
                        )
                    )
                content = parts
            row: dict[str, Any] = {"role": lightweight.role, "content": content}
            if annotate_ids:
                row[NATIVE_MESSAGE_ID_KEY] = lightweight.source_message_id
            payloads.append(row)
        return payloads

    def _mark_stream_stopped(
        self,
        assistant_message_id: str,
        *,
        visible_copy: str,
        prepare_retry: bool = False,
        retry_prepared: bool = True,
    ) -> ConsoleChatMessage:
        """Mark a streaming assistant message stopped, tolerating an earlier stop request.

        ``stop_active_run`` finalizes the message synchronously and then
        cancels the active stream task; that task's own ``CancelledError``
        handler in ``_stream_assistant_response`` calls this a second,
        redundant time. ``store.mark_message_stopped`` raises ``ValueError``
        for that redundant call because the message is no longer pending/
        streaming -- i.e. some earlier call already finalized it -- so any
        such error here is tolerated by simply reading back the
        already-finalized message rather than re-raising. Before Plan-B
        final-review Medium-2, the only reachable terminal status from this
        path was "stopped" itself; a mid-regenerate stop now legitimately
        settles the message at its pre-regenerate status instead (e.g.
        "complete"), so this must tolerate any terminal status, not just
        "stopped".
        """
        try:
            owner_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            owner_id = None
        recovery = self.store.dispatch_recovery_for_session(owner_id)
        if (
            recovery is not None
            and recovery.assistant_message_id == assistant_message_id
            and not recovery.in_flight
        ):
            raise ConsoleDispatchSettlementError(
                "Dispatch terminal settlement previously failed."
            )
        if prepare_retry and not retry_prepared:
            stopped = self.store.get_message(assistant_message_id)
        else:
            try:
                stopped = self.store.mark_message_stopped(assistant_message_id)
            except ValueError:
                stopped = self.store.get_message(assistant_message_id)
        # Derive the owning session the same way `_active_stream_belongs_to_
        # session`/`streaming_session_id` do, rather than requiring every
        # caller to thread it through -- `assistant_message_id` is stable
        # even once the run finishes, so this is always resolvable unless
        # the session was closed out from under the run (in which case
        # there is nothing left to attribute the STOPPED stamp to).
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STOPPED, visible_copy),
            session_id=owner_id,
        )
        return stopped

    def _set_run_state(
        self, run_state: ConsoleRunState, *, session_id: str | None = None
    ) -> None:
        """Write ``run_state`` for ``session_id`` (default: the active session).

        Parallel-agents spec §2: this is the ONLY path that mutates the
        per-session run-state map -- ``run_state``/``run_state_history``
        stay read-only facades (see their property definitions near
        ``__init__``). ``session_id=None`` preserves every pre-existing
        call site's behavior (targets whatever session is currently active);
        callers that know the run's OWNING session (which may not be the
        active one once a background run outlives a session switch) pass it
        explicitly.
        """
        target = (
            session_id
            if session_id is not None
            else (self.store.active_session_id or "")
        )
        # Task 10 (background completion toasts): captured BEFORE the
        # overwrite below so the once-guard downstream can tell a genuine
        # transition INTO a terminal outcome (toast) apart from a
        # defensive re-stamp of the SAME terminal status onto an already-
        # terminal session (no toast -- the brief's own re-set test pins
        # this).
        previous_status = self.run_state_for(target).status
        self._run_states[target] = run_state
        self.run_state_history_for(target).append(run_state.status)
        if self._buddy_sink is not None:
            context_owners = self._buddy_run_owner_context.get() or {}
            if run_state.status is ConsoleRunStatus.VALIDATING:
                run_owner = self._buddy_sink.run_state(target, run_state.status)
                if run_owner is not None:
                    updated_owners = dict(context_owners)
                    updated_owners[target] = run_owner
                    self._buddy_run_owner_context.set(updated_owners)
            else:
                run_owner = context_owners.get(target)
                self._buddy_sink.run_state(
                    target, run_state.status, run_owner=run_owner
                )
                if (
                    run_state.status
                    in {
                        ConsoleRunStatus.BLOCKED,
                        ConsoleRunStatus.COMPLETED,
                        ConsoleRunStatus.STOPPED,
                        ConsoleRunStatus.IDLE,
                    }
                    and run_owner is not None
                ):
                    updated_owners = dict(context_owners)
                    updated_owners.pop(target, None)
                    self._buddy_run_owner_context.set(updated_owners or None)
        self._advance_lifecycle_revision(target)
        terminal_notification_eligible = self.activity_for(
            target
        ).terminal_notification_eligible
        # Task 9 finding #2 (deferred from Task 7 review): a terminal run
        # has no live approval left to decide, so the pending-approval flag
        # must be discarded for ANY terminal transition -- including the
        # currently ACTIVE session's own. Pre-Task-9 this discard lived
        # ONLY inside the non-active branch below (alongside the unvisited-
        # outcome stamp), so a pending flag on the session you were actually
        # LOOKING AT survived its own run ending, leaving a misleading
        # NEEDS_APPROVAL badge with no round left behind it. Kept as its own
        # unconditional block, separate from the unvisited-outcome stamp,
        # which deliberately STAYS non-active-only (the viewed session's own
        # COMPLETED/FAILED transition is visible live in its transcript and
        # must never grow a stale "unvisited" fleet marker on itself).
        # TASK-1050: a terminal run state means NO approval-like round can
        # legitimately remain live for this session from ANY bridge -- pop
        # the session's ENTIRE round-id set (not just the deprecated shim's
        # sentinel), unlike a single bridge's own teardown which only ever
        # discards ITS OWN round id.
        if run_state.status in {
            ConsoleRunStatus.BLOCKED,
            ConsoleRunStatus.COMPLETED,
            ConsoleRunStatus.FAILED,
            ConsoleRunStatus.STOPPED,
        }:
            # F2b fix (Qodo wave): this call always runs on the main
            # thread today, but guard it with the same lock as every other
            # `_pending_approvals` mutation for consistency (and so it
            # stays correct if a future caller ever moves this off-thread).
            with self._approval_state_lock:
                self._pending_approvals.pop(target, None)
            if self._buddy_sink is not None:
                self._buddy_sink.release_session(target, sources={"approval"})
            # PR3a-2 Task 5: a terminal transition frees send capacity
            # (this session's own slot, possibly the global cap) -- retry
            # any deferred wake. Scheduled via the coordinator's loop hop,
            # never inline: a wake attempt must not reenter whatever send
            # flow is stamping this terminal state right now. Guarded for
            # exotic construction orders where the attribute is not up yet.
            wake = getattr(self, "_fleet_wake", None)
            if wake is not None:
                wake.retry_soon()
        # Parallel-agents spec §6: stamp an unvisited terminal outcome, but
        # ONLY for a session other than the currently active (viewed) one --
        # the viewed session's own COMPLETED/FAILED transition is visible
        # live in its transcript and must never grow a stale "unvisited"
        # fleet marker on itself. `mark_session_visited` is the sole path
        # that clears an entry stamped here.
        if (
            target != (self.store.active_session_id or "")
            and terminal_notification_eligible
        ):
            if run_state.status is ConsoleRunStatus.COMPLETED:
                self._unvisited_outcomes[target] = ConsoleRunMarker.FINISHED_OK
            elif run_state.status is ConsoleRunStatus.FAILED:
                self._unvisited_outcomes[target] = ConsoleRunMarker.FINISHED_FAILED
            # Task 10 (background completion toasts, parallel-agents spec):
            # ONE toast on a non-active session's run finishing/failing --
            # the viewed session's own terminal transition gets none FROM
            # THIS branch (same "user is watching" rule as the
            # unvisited-outcome stamp just above; task-2154.16/FB-05 added
            # the viewed session's FAILED toast as its own branch below).
            # Once-guarded on the transition INTO a terminal state:
            # `previous_status` was NOT already one of the four terminal
            # statuses, so re-setting the same COMPLETED/FAILED status again
            # (e.g. a defensive re-stamp) does not re-toast.
            if (
                run_state.status
                in (ConsoleRunStatus.COMPLETED, ConsoleRunStatus.FAILED)
                and previous_status
                not in {
                    ConsoleRunStatus.BLOCKED,
                    ConsoleRunStatus.COMPLETED,
                    ConsoleRunStatus.FAILED,
                    ConsoleRunStatus.STOPPED,
                }
                and self.notify_run_outcome is not None
            ):
                self.notify_run_outcome(target, run_state.status)
        # task-2154.16 (FB-05): the ACTIVE session's own transition INTO
        # FAILED gets an ambient error toast carrying the run's visible copy
        # (the same text as the transcript system row) -- the system row
        # alone left a user composing their next message with no failure
        # signal (the run-state surface is hidden; the header badge stays
        # Ready). COMPLETED stays silent here (FB-07 is out of scope), and
        # the same once-guard as the non-active branch above keeps a
        # defensive re-stamp of an already-terminal status from re-toasting.
        if (
            target == (self.store.active_session_id or "")
            and run_state.status is ConsoleRunStatus.FAILED
            and terminal_notification_eligible
            and previous_status
            not in {
                ConsoleRunStatus.BLOCKED,
                ConsoleRunStatus.COMPLETED,
                ConsoleRunStatus.FAILED,
                ConsoleRunStatus.STOPPED,
            }
            and self.notify_run_failure is not None
        ):
            self.notify_run_failure(run_state.visible_copy)

    def _publish_queue_chain_terminal(
        self, session_id: str, status: ConsoleRunStatus
    ) -> None:
        """Publish the one terminal marker/toast deferred across a queue chain."""

        # PR3a-2 Task 5: chain end is the moment queue ownership actually
        # releases (`finalize_empty_chain`/pause ran before this publish),
        # and no further terminal run-state transition follows it --
        # without this retry a wake deferred behind a queue chain would
        # starve until an unrelated trigger. Before the status filters
        # below: the release happens for EVERY chain-terminal status.
        self._fleet_wake.retry_soon()
        if status not in {ConsoleRunStatus.COMPLETED, ConsoleRunStatus.FAILED}:
            return
        if not self.activity_for(session_id).terminal_notification_eligible:
            return
        active_id = self.store.active_session_id or ""
        if session_id != active_id:
            marker = (
                ConsoleRunMarker.FINISHED_OK
                if status is ConsoleRunStatus.COMPLETED
                else ConsoleRunMarker.FINISHED_FAILED
            )
            self._unvisited_outcomes[session_id] = marker
            if self.notify_run_outcome is not None:
                self.notify_run_outcome(session_id, status)
        elif status is ConsoleRunStatus.FAILED and self.notify_run_failure is not None:
            self.notify_run_failure(self.run_state_for(session_id).visible_copy)

    def _clear_terminal_run_state(self, session_id: str | None = None) -> None:
        """Clear stale terminal status copy for ``session_id`` (default: active).

        Parallel-agents spec §2: terminal-only guard preserved verbatim --
        a NON-terminal (e.g. STREAMING) run is never reset by this, so a
        background run in progress on another session is untouched when the
        viewed session changes.
        """
        target = (
            session_id
            if session_id is not None
            else (self.store.active_session_id or "")
        )
        if self.run_state_for(target).status in {
            ConsoleRunStatus.BLOCKED,
            ConsoleRunStatus.COMPLETED,
            ConsoleRunStatus.FAILED,
            ConsoleRunStatus.STOPPED,
        }:
            self._set_run_state(ConsoleRunState(), session_id=target)

    def _active_stream_belongs_to_session(self, session_id: str) -> bool:
        """Whether ``session_id`` has its own registered in-flight stream.

        Task 3b: a direct membership check now that the underlying map is
        keyed by session id -- no lookup (or ``KeyError`` guard) needed.
        """
        return session_id in self._active_assistant_message_ids

    def streaming_session_id(self) -> str | None:
        """Return A session with an in-flight stream, for tab status glyphs.

        Task 3b: this single-value contract predates true concurrency
        (Task 3) -- under concurrent runs there can be MULTIPLE streaming
        sessions at once, and this still returns only one. Prefers the
        ACTIVE (viewed) session when it has a live entry (keeps today's
        "the tab you're looking at shows the spinner" behavior), else an
        arbitrary (insertion-order) live entry. Full multi-session tab/
        fleet markers are PA-T8's job; this just keeps the existing
        single-glyph caller (``console_session_surface``'s tab strip) from
        going stale now that the underlying map is per-session.
        """
        active = self.store.active_session_id
        if active is not None and active in self._active_assistant_message_ids:
            return active
        for session_id in self._active_assistant_message_ids:
            return session_id
        return None

    def _session_closed_result(
        self, *, session_id: str | None = None, dispatch_gap: bool = False
    ) -> ConsoleSubmitResult:
        """Result for a KeyError caused by the message's session vanishing mid-run.

        ``session_id`` is the run's owning session when the caller still has
        it in scope (most call sites do -- ``owner_id``/``session_id``
        resolved earlier in the same method); ``None`` only where the very
        first lookup of that owning session is what failed, i.e. there is
        genuinely nothing to attribute the STOPPED stamp to. Either way the
        owning session no longer exists in the store (``close_session``
        purges it), so this write is at worst an orphaned map entry -- never
        a stamp on a live, currently-viewed session.

        Args:
            dispatch_gap: Task 4 fix-round-2 (I2/M2): ``True`` ONLY at
                ``submit_draft``'s own call site, where the DISPATCHED
                session was closed in the gap between dispatch and this
                coroutine actually running -- there the draft simply vanishes
                with no other signal at all (the composer was already
                cleared at the keypress). Every other call site of this
                method (~19, across the stream/direct/agent finalization
                paths) fires MID-RUN, after the user has already closed a
                session they were actively viewing -- that session's run
                state already reflects STOPPED and the close itself was a
                deliberate, already-acknowledged user action, so toasting
                "Session closed" there too would be a redundant, confusing
                second signal for something the user just did on purpose.
        """
        visible_copy = SESSION_CLOSED_COPY
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STOPPED, visible_copy),
            session_id=session_id,
        )
        if not dispatch_gap:
            return ConsoleSubmitResult(True, True, visible_copy)
        # Task 4 (D2 fix wave): the owning session is gone by definition here
        # (every call site already failed a session lookup) -- appending a
        # SYSTEM row the way `_block` does would raise `KeyError` against a
        # purged session, and writing it into whatever session happens to be
        # ACTIVE now would leak this outcome into an unrelated tab (exactly
        # the cross-session leak this codebase's per-session maps exist to
        # prevent). `session_closed=True` lets the screen-side caller show a
        # toast instead, so the swallowed send is still visible, just not as
        # a transcript row. The copy is deliberately more specific than the
        # generic "Session closed." used everywhere else: this is the ONE
        # case where a keypress-captured draft is simply gone, so the user
        # needs to be told THAT, not just that some session closed.
        return ConsoleSubmitResult(
            True,
            True,
            "Console session closed before your message could send.",
            session_closed=True,
        )

    def _active_run_rejection(
        self,
        *,
        session_id: str | None = None,
        append_row: bool = False,
        queue_authorization: QueueGenerationAuthorization | None = None,
    ) -> ConsoleSubmitResult | None:
        """Defense-in-depth double-send guard for ``submit_draft``.

        F4 fix (Qodo wave): accepts an optional ``session_id`` so
        ``submit_draft`` can check the DISPATCHED session's own run state
        rather than whichever session happens to be active right now (the
        two can differ once a session switch races a background
        dispatch -- see ``submit_draft``'s own docstring). Every
        pre-existing caller (``retry_message``/``continue_from_message``/
        etc., which operate only on the active session by construction --
        each already blocks with "Open the original session..." if a
        target message belongs elsewhere) omits ``session_id`` and keeps
        checking the active session exactly as before.

        Args:
            session_id: The session to check, or ``None``/empty to check
                the currently active session (the pre-fix behavior).
            append_row: Task 4 fix-round-2 (I1): ``False`` by default so the
                five pre-existing screen-level wrappers around this guard
                (``retry_message``/``continue_from_message``/
                ``regenerate_message``/``summarize_up_to``/
                ``edit_and_resend_message``) stay byte-identical -- each of
                those already surfaces this exact refusal copy as a toast
                via its OWN screen wrapper (mirroring TASK-232's mid-run
                gate), so appending a SYSTEM row here too would double-report
                the same rejection. Only ``submit_draft`` -- whose caller
                (``_dispatch_console_draft_send``) has no equivalent
                defense-in-depth toast of its own -- passes ``True``.

        Returns:
            ``None`` when a new send may proceed; otherwise a blocked
            ``ConsoleSubmitResult`` carrying the refusal copy.
        """
        target_id = session_id if session_id else (self.store.active_session_id or "")
        if self.store.interrupted_provider_continuation_message(target_id) is not None:
            visible_copy = PROVIDER_CONTINUATION_RECOVERY_REQUIRED
            if append_row and any(
                session.id == target_id for session in self.store.sessions()
            ):
                self.store.append_message(
                    target_id,
                    role=ConsoleMessageRole.SYSTEM,
                    content=visible_copy,
                )
            return ConsoleSubmitResult(False, False, visible_copy)
        if self.prompt_queue_coordinator.authorizes(
            queue_authorization, target_id
        ) and self.run_state_for(target_id).status in {
            ConsoleRunStatus.BLOCKED,
            ConsoleRunStatus.COMPLETED,
            ConsoleRunStatus.FAILED,
            ConsoleRunStatus.STOPPED,
        }:
            return None
        if target_id and self.prompt_queue_coordinator.controls_generation(target_id):
            visible_copy = "Queued messages control the next turn. Resume or manage the queue first."
            return ConsoleSubmitResult(False, False, visible_copy)
        if self.run_state_for(target_id).is_send_allowed:
            return None
        visible_copy = "A run is already running in this tab."
        if append_row:
            # Task 4 (D2 fix wave): this is defense-in-depth -- the screen's
            # own `send_refusal_copy` gate (checked BEFORE any worker is even
            # spawned) already notifies for the common case, so reaching
            # this branch at all means that first check raced a run that
            # started in the gap. Silently returning a blocked result here
            # (the pre-fix behavior) left the transcript looking exactly
            # like the send never happened -- no row, no toast, nothing to
            # explain it. Append a SYSTEM row the same way `_block` does for
            # every other pre-echo gate in `submit_draft`, but only when the
            # target session is still live: `target_id` can be an orphaned/
            # empty id when there is no active session at all, and
            # `store.append_message` raises `KeyError` for an unknown id.
            if any(session.id == target_id for session in self.store.sessions()):
                self.store.append_message(
                    target_id, role=ConsoleMessageRole.SYSTEM, content=visible_copy
                )
        return ConsoleSubmitResult(
            accepted=False,
            should_clear_draft=False,
            # Must match the screen gate's `send_refusal_copy` own-session
            # copy (parallel-agents spec §4) -- a rapid double-send can hit
            # this internal defense-in-depth check instead of the screen's
            # gate (the loser of the exclusive-worker creation race), and a
            # mismatched copy there would read as two different bugs instead
            # of one lost race.
            visible_copy=visible_copy,
        )
