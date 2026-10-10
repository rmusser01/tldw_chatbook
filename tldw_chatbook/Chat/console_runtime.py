"""App-owned Console runtime holder (task-15860, headless wake — Task 1).

**This module changes WHO constructs the Console runtime objects, and
nothing else.** It is the "pure ownership move" the owner made a staging
condition for design A (`.superpowers/sdd/2026-08-14-headless-wake/
DECISIONS.md`, owner answer (3)): separately reviewable and separately
revertable from the semantics work that follows it.

## What it owns

One `ConsoleRuntime` per app owns the four objects `ChatScreen` used to
build for itself:

- the `ConsoleChatStore` (with its `ChatPersistenceService`),
- the `ConsoleProviderGateway`,
- the `ConsoleAgentBridge` and the sibling `AgentRunsDB` file it is keyed
  off, together with the `register_fleet_attention` fan-out registration
  that `FleetDrainFanout.register`'s contract requires to sit next to
  bridge construction (the wake coordinator's own registration travels
  inside `ConsoleChatController.__init__`, so it moves with the
  controller),
- the `ConsoleChatController`.

It also owns the process-memory raw CLI refusal stash bank so an exact draft
survives replacement of the screen/controller that received the refusal.

Each is built lazily, on first `ensure_*` call, from parameters the
calling view supplies — the same parameters, in the same order, the
screen's own `_ensure_*` methods passed before this module existed.

## The lifetime landing (this file's second half)

The runtime now **survives the screen's unmount**, and teardown is split
in two:

| Call | When | What it does |
|---|---|---|
| `leave_console_runtime` | every navigation AWAY from Console | detaches only that exact view claim and clears its disposable projections |
| `dispose_console_runtime` | app exit (`_shutdown_app_owned_lifecycles`) | the permanent form — `controller.shutdown()` then `gateway.aclose()`, exactly the order `on_unmount` used to run |

No turn is cancelled by `leave_console`: runtime custody and pending
decisions outlive navigation. App disposal and explicit user cancellation
remain the domain shutdown boundaries.

TASK-31520 reuses and suspends Console during ordinary navigation. Actual
visibility, including covering modals, controls decision answerability.

## The view seam

`attach_view` / `detach_view` are the ONE place a screen's callables meet
the runtime, over the single enumerated `CONSOLE_VIEW_HOOK_SLOTS` list —
set on attach, restored to viewless defaults on detach, same list both
ways. They **replace** Task 1's `ConsoleRuntime.view` stand-in, which
"protected" the overlapping-screens window by building a second runtime
(i.e. by reproducing dispose-at-unmount). The real ordering is now
explicit: `_complete_screen_navigation` constructs and `restore_state`s
the INCOMING screen before `switch_screen` unmounts the outgoing one, so
the incoming screen's `attach_view` runs FIRST and claims the runtime, and
the outgoing screen's later `detach_view`/`leave_console` finds a
different claimant and does nothing at all.

Screen-owned TIMERS (transcript sync, fleet survivor tick, cost TTL) are
not runtime state and are not in that list: they stay screen-owned and
stay stopped at unmount.

## What "viewless" MEANS (Task 4)

The lifetime landing made every slot CLEARABLE; this one makes each
cleared value semantically right, and says why next to it (`why` on each
`ConsoleViewHookSlot`). Three slots are not `None`:
`wake_conversation_in_view` (whose read site reads unwired as IN VIEW and
would clear the ◈ mark for a delivery nobody could have seen),
`wake_user_priority_probe` (no composer, so no user claim) and
`_global_user_display_name` (called with no guard). Everywhere else
`None` is kept only because the production read site's own guard makes it
inert — or, for the two skill confirms, fail-closed-at-once.

A runtime is viewless in two ways, and both are covered: after a detach,
and from BIRTH (`ensure_chat_controller` with nothing attached — the
wake-at-launch shape). `delivery_ui_hook` gets a third guarantee: it is
re-armed by the next attach if a wake is still delivering, because a
Console opened mid-delivery would otherwise never repaint the turn.

## Why an app attribute rather than a global

`app.console_runtime` follows `app.console_image_edit_operations`
(`app.py`, Phase 1 of `TldwCli.__init__`): constructed on the app, and
re-created lazily by the screen when a test's app object never had one
(`ChatScreen._h3_image_edit_registry` is the shape copied here). Screens
are never cached (`app.py` `_create_navigation_screen`), so anything that
must outlive a navigation cannot live on one.

## The wake gate (was: "still deliberately unchanged")

- `_attempt` gates on `_disposed`: app exit refuses a wake; navigation does
  not mutate the controller's cancellation event or pending decisions.

## Continuity (task-15860 Task 3, landed)

The store this holds is now the SINGLE source of truth for Console message
history. `ChatScreen`'s `ScreenStateStore` snapshot no longer carries
`sessions`, `messages_by_session` or `active_session_id`: it carries view
state only (image view modes, task-resume projection, RAG source scope,
staged live-work launch). Task 0's P3b executed why — with the runtime
app-owned but the snapshot still carrying history, a wake turn that ran
while Console was unmounted persisted four rows to ChaChaNotes and the
returning user saw the two that predated the snapshot.

Concretely: `_restore_native_console_state` no longer calls
`ConsoleChatStore.restore_state`, so a returning view reads the live store
it left behind — tree, active leaf, drafts, pending attachments and all.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import os
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock, get_ident
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_SESSION_CLOSE_RECOVERY_REFUSAL,
    ConsoleLifecycleRevisionChanged,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_display_state import console_prompted_source_count
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyDefaults,
)
from tldw_chatbook.Chat.console_onboarding_state import (
    coerce_console_first_send_completed,
)
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.DB.base_db import run_owned_db_call

if TYPE_CHECKING:
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
    from tldw_chatbook.Chat.console_voice_promotion import (
        VoicePromotionOwner,
        VoicePromotionQuitPermit,
        VoicePromotionSessionCloseToken,
    )
    from tldw_chatbook.Chat.console_voice_supervisor import VoiceDispatchSupervisor
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
from tldw_chatbook.Chat.thinking_blocks import normalize_thinking_history_policy
from tldw_chatbook.config import coerce_bool_setting, runtime_capture_policy
from tldw_chatbook.Persona_Buddy.console_adapter import PersonaBuddyConsoleAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Agents.hook_permissions import (
        HookPermissions,
        HookReviewSnapshot,
    )
    from tldw_chatbook.Chat.console_hook_review import HookReviewResult
    from tldw_chatbook.Agents.run_hooks import RunHooksEngine
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_received_turn import ConsoleReceivedTurnClaim
    from tldw_chatbook.Chat.console_received_intent import ConsoleReceivedTurnIntent
    from tldw_chatbook.Chat.console_worktree_recovery import ConsoleWorktreeRecovery

#: The app attribute this module's helpers read and write. Named once so a
#: test can assert on the protocol rather than on a string literal.
CONSOLE_RUNTIME_ATTR = "console_runtime"

#: Lazy app-owned engine slot; construction performs no authority I/O.
_UNSET = object()

#: Where a runtime hides when the app object cannot hold one (a `None` app,
#: or a read-only double). Never the production path — `TldwCli.__init__`
#: always takes `CONSOLE_RUNTIME_ATTR`.
_VIEW_RUNTIME_FALLBACK_ATTR = "_console_runtime_fallback"

# Keep migration imports outside the first-interactive-frame window even on
# slower runners where Textual can still be settling the mount after setting
# ``_ui_ready``.  Match the app's deliberately post-startup media-cleanup
# delay: legacy normalization is idle maintenance, never readiness work.
LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS = 5.0
LEGACY_TRACE_MAINTENANCE_RETRY_DELAY_SECONDS = 1.0
TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS = 60.0
#: PERF-10 (TASK-33269): how often a parked maintenance loop checks its
#: in-memory wake conditions. Parked checks touch no database.
LEGACY_TRACE_MAINTENANCE_PARK_POLL_SECONDS = 1.0
TRACE_PHYSICAL_MAINTENANCE_RETRYABLE_REASONS = frozenset(
    {
        "provider_active",
        "activity_threshold",
        "maintenance_busy",
        "retry_backoff",
        "connections_busy",
        "active_transaction",
        "wal_checkpoint_failed",
        "lease_lost",
        "insufficient_disk",
        "integrity_check_failed",
        "interrupted",
        "cancelled",
        "vacuum_failed",
        "sqlite_failure",
        "compaction_failure",
        "database_threshold",
        "freelist_threshold",
        "freelist_ratio_threshold",
    }
)


class _LazyTraceCompatibilityMetrics:
    """Load the rollout counter implementation on its first actual use."""

    def __init__(self) -> None:
        self._delegate: Any | None = None
        self._lock = Lock()

    def _get_delegate(self) -> Any:
        delegate = self._delegate
        if delegate is not None:
            return delegate
        with self._lock:
            delegate = self._delegate
            if delegate is None:
                from tldw_chatbook.Chat.console_trace_metrics import (
                    TraceCompatibilityMetrics,
                )

                delegate = TraceCompatibilityMetrics()
                self._delegate = delegate
        return delegate

    def record(self, path: str, count: int = 1) -> None:
        """Record a content-free compatibility path."""

        self._get_delegate().record(path, count)

    def snapshot(self) -> Mapping[str, int]:
        """Return the current immutable compatibility counts."""

        return self._get_delegate().snapshot()


class _LazyConsoleActivityReceiptService:
    """Load receipt coordination on first switcher or settlement use."""

    def __init__(self, runs_db: Any, marks: Any | None) -> None:
        self._runs_db = runs_db
        self._marks = marks
        self._delegate: Any | None = None
        self._lock = Lock()

    def _get_delegate(self) -> Any:
        delegate = self._delegate
        if delegate is not None:
            return delegate
        with self._lock:
            delegate = self._delegate
            if delegate is None:
                from tldw_chatbook.Chat.console_activity_receipts import (
                    ConsoleActivityReceiptService,
                )

                delegate = ConsoleActivityReceiptService(
                    self._runs_db,
                    self._marks,
                )
                self._delegate = delegate
        return delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_delegate(), name)

    def unseen_snapshot(self) -> tuple[Any, ...]:
        """Read already-loaded receipts without initializing their coordinator.

        Returns:
            The current in-memory snapshot, or an empty tuple before hydration
            or settlement has initialized the authoritative receipt service.
        """
        delegate = self._delegate
        return () if delegate is None else delegate.unseen_snapshot()


class _LazyTraceBoundaryFactory:
    """Load normalized write planning only when a provider call reserves."""

    def __init__(self, database: Any, repository: Any | None) -> None:
        self._database = database
        self._repository = repository
        self._delegate: Any | None = None
        self._lock = Lock()

    def _get_delegate(self) -> Any:
        delegate = self._delegate
        if delegate is not None:
            return delegate
        with self._lock:
            delegate = self._delegate
            if delegate is None:
                from tldw_chatbook.Chat.console_trace_runtime import (
                    ConsoleTraceBoundaryFactory,
                )

                delegate = ConsoleTraceBoundaryFactory(
                    self._database,
                    repository=self._repository,
                )
                self._delegate = delegate
        return delegate

    def __call__(self, request: Any, resolution: Any, route: Any) -> object:
        """Create one provider-call boundary through the shared delegate."""

        return self._get_delegate()(request, resolution, route)


def recover_console_trace_calls(
    database: object,
    *,
    occurred_at: str | None = None,
    repository: object | None = None,
    recovery_grace_seconds: int = 300,
) -> tuple[object, ...]:
    """Close normalized provider calls stale at startup.

    Args:
        database: Trace database to recover.
        occurred_at: Optional recovery timestamp override.
        repository: Optional repository override for tests.
        recovery_grace_seconds: Minimum inactivity before a call is stale.

    Returns:
        Calls transitioned by the recovery pass.
    """

    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.Chat.console_trace_settlement import (
        ConsoleTraceSettlementCoordinator,
    )

    trace_repository = (
        repository
        if isinstance(repository, ConsoleTraceRepository)
        else ConsoleTraceRepository()
    )
    timestamp = occurred_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return ConsoleTraceSettlementCoordinator(trace_repository).recover_open_calls(
        database,
        occurred_at=timestamp,
        recovery_grace_seconds=recovery_grace_seconds,
    )


_ATTACH_WITHOUT_PRIOR_CLAIM = object()
_PROJECT_INSTRUCTION_NOTICE_TIMEOUT_SECONDS = 120.0
_PROJECT_INSTRUCTION_NOTICE_POLL_SECONDS = 0.1
_PROJECT_INSTRUCTION_PROJECTION_RETRY_DELAYS = (0.05, 0.1, 0.2)
CONSOLE_SESSION_CLOSE_GRACE_SECONDS = 2.0
CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS = 3.0
#: How long app exit waits for a message Delete or Undo still saving before
#: it ends the store (TASK-33628.5). A 3,000-message save takes ~0.2-0.3 s.
CONSOLE_DURABLE_WRITE_TEARDOWN_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class _HookPreparationSource:
    """Original sources for one finite runtime hook preparation invocation."""

    app: Any = field(repr=False)
    store: Any = field(repr=False)
    controller: Any = field(repr=False)
    session_id: str
    session: Any = field(repr=False)
    session_identity: tuple[Any, ...] = field(repr=False)
    permissions: Any = field(repr=False)
    persistence: Any = field(repr=False)
    chat_database: Any = field(repr=False)
    registry: Any = field(repr=False)
    workspace_database: Any = field(repr=False)
    consent: Any = field(repr=False)
    authority_reader: Any = field(repr=False)
    controller_app: Any = field(repr=False)
    context_provider: Any = field(repr=False)


@dataclass(slots=True)
class _ConsoleTurnCustodyInputs:
    """Sensitive turn-only values kept out of the public request repr."""

    attachments: tuple[Any, ...] = field(default=(), repr=False)
    staged_evidence_revision: int | None = None
    durable_accepted: bool = False


@dataclass(slots=True)
class _ConsoleTurnCustodyRecord:
    """Process-local ownership of one accepted runtime task."""

    turn_id: str
    session_id: str
    request: ConsoleTurnCustodyRequest | None = field(repr=False)
    inputs: _ConsoleTurnCustodyInputs = field(default_factory=_ConsoleTurnCustodyInputs, repr=False)
    task: asyncio.Task[Any] | None = field(default=None, repr=False)
    archive_conversation_id: str | None = None
    store: ConsoleChatStore | None = field(default=None, repr=False)
    received_claim: ConsoleReceivedTurnClaim | None = field(default=None, repr=False)
    received_intent: ConsoleReceivedTurnIntent | None = field(default=None, repr=False)


class _ConsoleTurnRefusedError(RuntimeError):
    """A custodied turn refused before durable acceptance, with its reason.

    TASK-33621.2: the refusal copy is the turn's OUTCOME, so it travels on
    the task's terminal exception to `_finish_custodied_turn`, which hands
    it to the recovery entry. It is not stored on the custody record: that
    record holds lifetime handles only (pinned by
    `test_runtime_owned_custody_tracks_only_lifetime_handles`). The message
    stays the fixed diagnostic text; the copy is kept off `args` so a
    logged or re-raised exception does not repeat it.
    """

    def __init__(self, message: str, *, reason: str = "") -> None:
        super().__init__(message)
        #: The code-owned copy the unsent-turn shelf states; may be empty.
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ConsoleTurnRecoveryEntry:
    """One exact pre-durable draft retained across Console navigation."""

    turn_id: str
    session_id: str
    draft: str = field(repr=False)
    attachments: tuple[Any, ...] = field(repr=False)
    insertion_order: int
    #: TASK-33621.2: why the controller refused this turn, so the unsent-turn
    #: strip can say so; empty when the turn ended for another reason.
    reason: str = field(default="", repr=False)
    source_claim: ConsoleReceivedTurnClaim | None = field(default=None, repr=False)


@dataclass(slots=True)
class _ConsoleStagedEvidenceState:
    """One app-owned staged-evidence value with a monotonic ownership fence."""

    launch: Any | None = field(default=None, repr=False)
    revision: int = 0
    sent_source_count: int | None = None


@dataclass(frozen=True, slots=True)
class ConsoleProjectBindingChoice:
    """Content-free projection of one app-owned workspace choice."""

    binding_id: str
    label: str
    eligible: bool
    recovery: str = ""


@dataclass(slots=True)
class _ProjectBindingDecision:
    """Runtime-owned setup decision that survives Console projection loss."""

    decision_id: str
    session_id: str
    order: int
    choices: tuple[ConsoleProjectBindingChoice, ...] = field(repr=False)
    completed: asyncio.Event = field(repr=False)
    result: tuple[str, str | None] = ("cancel", None)
    projected_generation: int | None = None
    retry_generation: int | None = None
    retry_budget_generation: int | None = None
    retry_attempts: int = 0


@dataclass(slots=True)
class _ProjectDispatchDecision:
    """Runtime-owned dispatch decision that survives Console projection loss."""

    decision_id: str
    session_id: str
    order: int
    notice: Any = field(repr=False)
    completed: threading.Event = field(repr=False)
    owning_cancel_event: Any = field(default=None, repr=False)
    result: str = "cancel"
    projected_generation: int | None = None
    retry_generation: int | None = None
    retry_budget_generation: int | None = None
    retry_attempts: int = 0
    answerable_seconds_remaining: float = 0.0
    answerable_since: float | None = None


def _current_library_policy_defaults(app: Any) -> ConsoleLibraryPolicyDefaults:
    """Read fresh future-session defaults from the app's current config."""
    config = getattr(app, "app_config", None)
    if not isinstance(config, Mapping):
        config = {}
    console = config.get("console", {})
    if not isinstance(console, Mapping):
        console = {}
    chat_defaults = config.get("chat_defaults", {})
    if not isinstance(chat_defaults, Mapping):
        chat_defaults = {}
    return ConsoleLibraryPolicyDefaults(
        auto_retrieve=(
            ConsoleAutoRetrieve.AUTOMATIC
            if coerce_bool_setting(
                chat_defaults.get("rag_auto_retrieve_on_send", False), False
            )
            else ConsoleAutoRetrieve.NEVER
        ),
        assistant_access=(
            ConsoleAssistantLibraryAccess.ALLOWED
            if coerce_bool_setting(
                console.get("assistant_library_access_default", False), False
            )
            else ConsoleAssistantLibraryAccess.BLOCKED
        ),
    )


def _current_thinking_history_policy_default(app: Any) -> str:
    """Read the optional replay default copied into the next new session."""

    config = getattr(app, "app_config", None)
    console = config.get("console", {}) if isinstance(config, Mapping) else {}
    if not isinstance(console, Mapping):
        console = {}
    return normalize_thinking_history_policy(
        console.get("thinking_history_policy_default")
    )


def _provider_config_for_app(app: Any) -> Mapping[str, Any]:
    """Return fresh provider configuration without retaining a Console view."""
    config = getattr(app, "app_config", None)
    snapshot = config if isinstance(config, Mapping) else {}
    if not all(section in snapshot for section in ("general", "logging")):
        return snapshot
    try:
        from tldw_chatbook.config import load_settings

        fresh = load_settings()
    except Exception:
        return snapshot
    return fresh if isinstance(fresh, Mapping) and fresh else snapshot


_PROVIDER_CONFIG_FOR_APP_ORIGINAL = (
    globals(), _provider_config_for_app, _provider_config_for_app.__code__
)


def _native_tools_enabled_for_app(app: Any) -> bool:
    """Read the native-tools gate from app-owned configuration."""
    console = _provider_config_for_app(app).get("console", {})
    value = (
        console.get("native_tool_calls", True)
        if isinstance(console, Mapping)
        else True
    )
    return coerce_bool_setting(value, True)


def _global_user_display_name_for_app(app: Any) -> str:
    """Return the global chat identity without retaining a Console view."""
    from tldw_chatbook.Chat.console_roleplay_identity import (
        ChatDisplayNameError,
        normalize_chat_display_name,
    )

    defaults = _provider_config_for_app(app).get("chat_defaults", {})
    raw = (
        defaults.get("user_display_name", "User")
        if isinstance(defaults, Mapping)
        else "User"
    )
    try:
        return normalize_chat_display_name(raw, blank_means_none=False) or "User"
    except ChatDisplayNameError:
        return "User"


def _apply_chat_dictionaries_for_app(
    app: Any,
    conversation_id: str | None,
    text: str,
    frozen_inputs: Mapping[str, Any] | None = None,
) -> str:
    """Apply conversation dictionaries through app-owned database state."""
    if frozen_inputs is not None:
        if not isinstance(text, str):
            return text
        try:
            from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as cdl

            entries = tuple(frozen_inputs.get("dictionary_entries") or ())
            if not entries:
                return text
            return cdl.process_user_input(
                text,
                list(entries),
                max_tokens=500,
                strategy="sorted_evenly",
            )
        except Exception:  # noqa: BLE001 -- optional context never blocks send
            return text
    db = getattr(app, "chachanotes_db", None)
    if db is None or not conversation_id or not isinstance(text, str):
        return text
    from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as cdl

    return cdl.apply_active_chatdicts_to_text(
        db,
        conversation_id,
        None,
        text,
        max_tokens=500,
        strategy="sorted_evenly",
    )


def _apply_world_info_for_app(
    app: Any,
    conversation_id: str | None,
    text: str,
    history: list[Any],
    frozen_inputs: Mapping[str, Any] | None = None,
) -> str:
    """Apply conversation world-info through app-owned database state."""
    if frozen_inputs is not None:
        if not isinstance(text, str) or not frozen_inputs.get("world_enabled"):
            return text
        try:
            from tldw_chatbook.Character_Chat.world_info_processor import (
                WorldInfoProcessor,
            )

            def thaw(value: Any) -> Any:
                if isinstance(value, Mapping):
                    return {key: thaw(item) for key, item in value.items()}
                if isinstance(value, tuple):
                    return [thaw(item) for item in value]
                return value

            world_books = thaw(tuple(frozen_inputs.get("world_books") or ()))
            if not world_books:
                return text
            processor = WorldInfoProcessor(world_books=world_books)
            result = processor.process_messages(text, history or [])
            matched = result.get("matched_entries") or []
            if not matched:
                return text
            formatted = processor.format_injections(result.get("injections", {}))
            parts = [formatted[key] for key in ("at_start", "before_char") if formatted.get(key)]
            parts.append(text)
            parts.extend(
                formatted[key]
                for key in ("after_char", "at_end")
                if formatted.get(key)
            )
            return "\n\n".join(parts)
        except Exception:  # noqa: BLE001 -- optional context never blocks send
            return text
    db = getattr(app, "chachanotes_db", None)
    if db is None or not conversation_id or not isinstance(text, str):
        return text
    from tldw_chatbook.config import get_cli_setting

    if not get_cli_setting("character_chat", "enable_world_info", True):
        return text
    from tldw_chatbook.Character_Chat.world_info_resolver import (
        apply_world_info_to_message,
    )

    return apply_world_info_to_message(db, conversation_id, None, text, history or [])


def _library_provider_for_app(
    app: Any,
    turn_context: Any | None = None,
    **activity_kwargs: Any,
) -> Any | None:
    """Build one Library provider from frozen authority and app services.

    The single builder for both Console entry points.
    `ConsoleLibraryActivityController.build_provider` used to hold a
    byte-for-byte copy of this body and delegates here instead
    (TASK-32892 P0-1): the copies drifted when 5dd1077df6 retired
    `LocalLibraryToolService`'s `collections_service` parameter and updated only
    one of them, and because `ensure_chat_controller` binds THIS function over
    whatever the screen passed, the stale copy was the one every shipped run
    called -- raising `TypeError` on the default configuration and costing the
    agent all 24 Library tools behind one swallowed warning.

    Args:
        app: Application instance whose local service attributes are read live.
        turn_context: Immutable production turn context, if available.
        **activity_kwargs: Provider activity-capture bindings, when a view owns
            one (`activity_attempt_id` / `activity_sink`).

    Returns:
        Configured provider, or ``None`` without a turn context.
    """
    if turn_context is None:
        return None
    if not turn_context.library_authority.direct_library_tools:
        from tldw_chatbook.Agents.library_rag_tool_provider import (
            LibraryRagToolProvider,
        )

        return LibraryRagToolProvider(
            getattr(app, "library_rag_search_service", None),
            **activity_kwargs,
        )

    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
    from tldw_chatbook.Library.local_library_tool_service import (
        LocalLibraryToolService,
    )

    media_chunk_service = None
    media_reading_service = getattr(app, "local_media_reading_service", None)
    media_db = getattr(app, "media_db", None) or getattr(
        media_reading_service, "media_db", None
    )
    if media_db is not None or media_reading_service is not None:
        from tldw_chatbook.Chunking.chunking_interop_library import (
            get_chunking_service,
        )
        from tldw_chatbook.Library.local_media_chunk_tool_service import (
            LocalMediaChunkToolService,
        )

        media_chunk_service = LocalMediaChunkToolService(
            media_db,
            media_reading_service,
            template_interop=(
                get_chunking_service(media_db) if media_db is not None else None
            ),
            policy_enforcer=getattr(app, "service_policy_enforcer", None),
        )
    service = LocalLibraryToolService(
        media_service=media_reading_service,
        notes_service=getattr(app, "notes_service", None),
        prompt_service=getattr(app, "local_prompt_service", None),
        skills_service=getattr(app, "local_skills_service", None),
        conversation_service=getattr(app, "local_chat_conversation_service", None),
        media_chunk_service=media_chunk_service,
        notes_scope_service=getattr(app, "notes_scope_service", None),
        policy_enforcer=getattr(app, "service_policy_enforcer", None),
    )
    return LibraryToolProvider(service, **activity_kwargs)


def _default_session_settings_for_app(app: Any) -> Any:
    """Build new-session defaults from app-owned inputs."""
    from tldw_chatbook.Chat.console_session_settings import (
        default_console_session_settings,
    )

    return default_console_session_settings(_provider_config_for_app(app))


__all__ = [
    "CONSOLE_RUNTIME_ATTR",
    "CONSOLE_VIEW_HOOK_SLOTS",
    "ConsoleRuntime",
    "ConsoleTurnRecoveryEntry",
    "ConsoleViewHookSlot",
    "dispose_console_runtime",
    "ensure_console_runtime",
    "leave_console_runtime",
    "viewless_conversation_in_view",
    "viewless_user_display_name",
    "viewless_user_priority_probe",
]


def viewless_user_display_name() -> str:
    """The display name a runtime with no view uses.

    `ConsoleChatController.__init__` does `global_user_display_name or
    (lambda: "User")`, so `None` is NOT this slot's viewless default —
    clearing it to `None` would turn every read into a `TypeError`.
    (`_presentation_context_for`'s broad `except` catches that TypeError
    and falls back to "User" anyway — which is *worse* than a raise: the
    slot would be silently degraded, logging a warning per read, which is
    exactly the class of failure Task 4 exists to remove.)
    """
    return "User"


def viewless_conversation_in_view(conversation_id: str, session_id: str) -> bool:
    """A runtime with no view is watching nothing. Always `False`.

    task-15860 Task 4, and the reason this function exists rather than a
    `None`: `ConsoleFleetWakeCoordinator._conversation_in_view` reads an
    UNWIRED probe as **in view** (the pre-screen rig's documented
    clear-on-delivery), so a viewless default of `None` makes a wake that
    nobody could possibly have watched commit as "seen" and CLEARS the
    `FLEET_UNSEEN` ◈ mark. task-15971's whole point is the opposite: the
    user must be able to learn that a supervisor turn ran and landed
    while they were elsewhere.

    Args:
        conversation_id: The delivered conversation (unused — no view
            means no conversation is displayed).
        session_id: The session the wake turn ran in (unused, same
            reason).

    Returns:
        False, always.
    """
    return False


def viewless_user_priority_probe(session_id: str) -> bool:
    """A runtime with no view has no user claim. Always `False`.

    `_attempt`'s user-wins-ties gate asks the view whether the user is
    mid-thought (a non-empty composer draft). With no view there is no
    composer, so no user can hold a claim and a wake must not defer.

    `None` happens to produce the same outcome today, because `_attempt`
    guards with `callable(probe)` — but only by accident of that guard's
    direction. The sibling probe above uses the OPPOSITE convention for
    an unwired slot (uncertainty defers toward the badge), so leaving
    this one's correctness resting on a `callable()` check one line of
    someone else's refactor away is not a default, it is a coincidence.

    Args:
        session_id: The session a wake would fire into (unused).

    Returns:
        False, always.
    """
    return False


@dataclass(frozen=True)
class ConsoleViewHookSlot:
    """One runtime-object attribute a mounted Console view owns.

    Args:
        name: The attribute name on the target object.
        target: Which runtime object holds it — ``"controller"``,
            ``"store"`` or ``"wake"`` (the controller's
            ``fleet_wake`` coordinator).
        viewless_default: What `detach_view` restores. Almost always
            `None`, which is every one of these slots' documented
            "no UI wired" value.
    """

    name: str
    target: str
    viewless_default: Any = None
    #: WHY this slot's viewless default is correct — i.e. what the
    #: production read site does with it. Task 4's rule: a `None` default
    #: is only allowed where the read site's own guard makes `None` mean
    #: the semantically right thing (inert, or fail-closed); anywhere the
    #: guard's fallback is a WRONG answer, the default must be an explicit
    #: callable. Kept next to the value so the two cannot drift.
    why: str = ""


@dataclass(frozen=True)
class _CanvasNativeViewBinding:
    """Latest mounted Console callbacks for the lazy native authority."""

    scope_resolver: Callable[[str], Any]
    bridge_sink: Callable[[Any, str], None] | None
    bridge_prepare: Callable[[Any], Callable[[str], None]] | None
    auto_open: Callable[[str, Any], None] | None
    publication_guard: Callable[[Any], bool] | None
    source_scope_resolver: Callable[[str], Any]
    controller: Any
    view: Any
    attachment_generation: int | None


#: **The one enumerated list of screen-owned hook slots.** `attach_view`
#: sets every entry from the view's `console_view_hooks()` map and
#: `detach_view` restores every entry's `viewless_default` — the same list
#: in both directions, so a slot cannot be bound without being cleared.
#:
#: Task 0's P3 measured that a VIEWLESS wake turn touches five of these
#: (`delivery_ui_hook`, `wake_conversation_in_view`,
#: `wake_user_priority_probe`, `_chat_dictionary_applier`,
#: `_world_info_applier`) — and that all five were still bound to a DEAD
#: `ChatScreen` and none raised. A silent wrong answer is worse than a
#: raise: `wake_conversation_in_view` decides whether the unseen ◈ mark
#: survives (task-15971) and `wake_user_priority_probe` decides whether the
#: user wins a tie.
#:
#: Four entries here were NOT in P3's list of fifteen, because P3 only
#: wrapped callables that were `ChatScreen` methods:
#: `_default_session_settings` and `_turn_context_provider` are bound to
#: the screen's `ConsoleSessionController`, `prompt_history` is a value
#: built by the screen's prompts controller, and `on_scope_flushed` lives
#: on the STORE, not the controller. Each still holds the dead screen
#: transitively.
#:
#: `controller.app` is deliberately absent: it is the APP, which outlives
#: every view, and clearing it would break the `call_from_thread` bridge a
#: surviving turn still needs.
#:
#: **Task 4's rule for `viewless_default`.** Every entry now carries a
#: `why` naming the production read site that makes its value correct.
#: `None` is allowed only where that read site's own guard turns `None`
#: into the semantically right behaviour — inert, or fail-closed. The two
#: slots where it did NOT (`wake_conversation_in_view`, whose read site
#: treats "unwired" as IN-VIEW; and `_global_user_display_name`, whose read
#: site calls the slot with no guard at all) carry explicit callables. A
#: third, `wake_user_priority_probe`, is explicit for a different reason:
#: `None` is right there only by accident of one `callable()` check, and
#: its sibling probe uses the opposite unwired convention.
CONSOLE_VIEW_HOOK_SLOTS: tuple[ConsoleViewHookSlot, ...] = (
    # Only disposable screen projections belong here. Domain dependencies
    # are frozen into custody or resolved through app-owned services before
    # a turn task starts.
    ConsoleViewHookSlot(
        "follow_watchlists_operations",
        "controller",
        why="Guarded by `remount_watchlists_operation_receipts`; canonical "
        "receipt identities remain retained by the app-owned controller "
        "while no Console view is mounted.",
    ),
    ConsoleViewHookSlot(
        "set_pending_approval",
        "controller",
        why="Inert but NOT lossy: `request_mcp_approvals` calls "
        "`add_pending_round` and retains the round's payload in "
        "`_parked_approval_payloads` BEFORE consulting this hook, and "
        "does so unconditionally. So a round armed with no view is still "
        "registered and still claimable at the next mount. (Surfacing it "
        "app-wide, and the 120s clock it runs against, are plan Task 5.)",
    ),
    ConsoleViewHookSlot(
        "update_pending_approval_summary",
        "controller",
        why="ADR-090: `_deliver_permission_summary` guards with `is not "
        "None` and the summary it would patch is ALREADY durably stored "
        "on the round state and the retained payload before this fires, "
        "so a viewless delivery loses nothing -- the next mount re-renders "
        "the line from the payload's own `summary` slot.",
    ),
    ConsoleViewHookSlot(
        "park_pending_approval",
        "controller",
        why="Same as above — the badge/toast half of the same round. "
        "Guarded by `is not None` at both call sites; the registry write "
        "that makes the round recoverable does not depend on it.",
    ),
    ConsoleViewHookSlot(
        "notify_run_outcome",
        "controller",
        why="Guarded at every call site. A toast with no screen to toast "
        "into is the exact dead-screen call Task 0's P3 found; inert is "
        "the whole point.",
    ),
    ConsoleViewHookSlot(
        "notify_run_failure",
        "controller",
        why="Guarded at every call site; same reasoning. The run's own "
        "terminal state and its DB row are unaffected by the missing "
        "toast, so nothing durable is lost.",
    ),
    ConsoleViewHookSlot(
        "set_task_panel",
        "controller",
        why="Every read site is an `is not None` guard: the pinned task "
        "panel is a mirror of the session's todo store, and with no view "
        "there is nothing to mirror into. The store itself keeps the tasks, "
        "so the next attach re-derives the panel from it.",
    ),
    ConsoleViewHookSlot(
        "set_pending_worktree_merge",
        "controller",
        why="Disposable real confirmation surface; exact rounds remain controller-owned.",
    ),
    ConsoleViewHookSlot(
        "set_pending_question",
        "controller",
        why="`request_user_questions` returns `{answered: False, reason: "
        "'cancelled'}` when it is None, and `_ask_user_wiring` registers no "
        "tool at all without it -- a viewless run cannot be asked, which is "
        "PRD A10's headless posture.",
    ),
    ConsoleViewHookSlot(
        "set_pending_chat_create",
        "controller",
        why="Same fail-closed-at-once contract for "
        "`request_chat_create_confirm` (`allow=False, remember=False`): "
        "with no view nothing could ever set the Event, so denying at "
        "once beats blocking for the full timeout.",
    ),
    ConsoleViewHookSlot(
        "complete_agent_chat_create",
        "controller",
        why="Guarded (`if self.app is not None and self.complete_agent_"
        "chat_create is not None`) in `execute_agent_chat_create`, and it "
        "is the EXECUTE half's advertised-equals-usable gate: with it None "
        "the controller passes `execute_agent_chat_create=None` to the "
        "bridge, which never builds the fork_chat/new_chat closures at "
        "all. The durable conversation row is complete before the callback "
        "is consulted, so a viewless create loses only the (Task 8) "
        "session placement, never data.",
    ),
    ConsoleViewHookSlot(
        "wake_user_priority_probe",
        "controller",
        viewless_user_priority_probe,
        why="**Not None.** No composer exists, so no user can be "
        "mid-thought and a wake must not defer. See the function.",
    ),
    ConsoleViewHookSlot(
        "wake_conversation_in_view",
        "controller",
        viewless_conversation_in_view,
        why="**Not None.** `_conversation_in_view` reads an unwired probe "
        "as IN VIEW and CLEARS the ◈ FLEET_UNSEEN mark. See the function.",
    ),
    # -- the store's one screen-owned callback ----------------------------
    ConsoleViewHookSlot(
        "on_scope_flushed",
        "store",
        why="Guarded (`if flushed_scope is not None and self."
        "on_scope_flushed is not None`). It repaints the scope chip; the "
        "flush itself already happened in the store.",
    ),
    # -- the wake coordinator's repaint hook ------------------------------
    ConsoleViewHookSlot(
        "delivery_ui_hook",
        "wake",
        why="Guarded (`if callable(hook)`), so None is exactly 'no repaint "
        "target' — correct while detached. The hazard is not the inert "
        "detached value but the MISSING RE-ARM: see "
        "`ConsoleRuntime._rearm_delivery_ui_hook`.",
    ),
)


class ConsoleRuntime:
    """The app-owned holder for one Console runtime.

    Every `ensure_*` method is idempotent: it builds its object on the
    first call and returns the cached instance afterwards, ignoring the
    parameters on subsequent calls. That is the same laziness
    `ChatScreen._ensure_*` had — the parameters were only ever read at
    construction there too.

    Attributes are exposed through non-constructing read-only properties
    so a caller can ask "has this been built yet?" without building it,
    which several `ChatScreen` call sites do (`store = self.
    _console_chat_store` followed by an `is None` early return).
    """

    def __init__(
        self,
        app: Any,
        *,
        canvas_enabled_reader: Callable[[], bool] | None = None,
    ) -> None:
        """Bind the holder to one app object.

        Args:
            app: The `TldwCli` app (or, in tests, whatever object plays
                that role for the screen under test). Read for
                `chachanotes_db`, `citation_trace_repository`,
                `workspace_registry_service` and the
                `console_provider_gateway_factory` test seam — never
                mutated.
        """
        self._execution_capacity = None
        self._execution_capacity_lock = threading.RLock()
        self._app = app
        self._canvas_profile_snapshot = getattr(app, "_canvas_profile_snapshot", None)
        # -- setters, for the screen handles that now READ THROUGH here ----
        # `ChatScreen._console_chat_store`/`_console_provider_gateway`/
        # `_console_chat_controller` (and `ConsoleAgentController.
        # _console_agent_bridge`) are properties over these slots since the
        # runtime started outliving the screen: a fresh screen's own `None`
        # would otherwise SHADOW a live runtime object until `_ensure_*`
        # ran. See `set_chat_store` and friends.
        self._chat_store: Any | None = None
        self._provider_gateway: Any | None = None
        self._agent_bridge: Any | None = None
        self._worktree_recovery = None
        self._agent_runs_db: Any | None = None
        self._activity_receipts: Any | None = None
        self._activity_receipts_lock = Lock()
        self._receipt_owner_thread_id = get_ident()
        self._activity_hydration_task: asyncio.Task[int] | None = None
        self._change_review_coordinator: Any | None = None
        self._chat_controller: Any | None = None
        self._canvas_controller: Any | None = None
        self._canvas_gateway: Any | None = None
        self._canvas_gateway_authority: Any | None = None
        self._canvas_native_authority: Any | None = None
        self._canvas_native_view_binding: _CanvasNativeViewBinding | None = None
        self._canvas_native_lock = Lock()
        self._canvas_maintenance_closed = False
        self._canvas_maintenance_generation = 0
        self._canvas_policy_cleanups: set[asyncio.Task[None]] = set()
        self._progress_cleanup_tasks: set[asyncio.Task[Any]] = set()
        self._canvas_settlement_listener = self._forward_canvas_settlement
        if canvas_enabled_reader is None:
            from tldw_chatbook.config import get_canvas_execution_enabled

            canvas_enabled_reader = get_canvas_execution_enabled
        self._canvas_enabled_reader = canvas_enabled_reader
        self._canvas_disabled_latched = self._read_canvas_enabled() is False
        self._canvas_policy_watch_task: asyncio.Task[None] | None = None
        self._canvas_policy_read_task: asyncio.Task[bool] | None = None
        self._legacy_trace_maintenance_task: asyncio.Task[None] | None = None
        # One app-wide mutation lane for exact persisted-conversation opens.
        # Individual ChatScreen workspaces are disposable views over this
        # runtime, so a per-screen lock would allow their hydration/rollback
        # sequences to interleave.
        self.character_conversation_activation_lock = asyncio.Lock()
        self.trace_compatibility_metrics = _LazyTraceCompatibilityMetrics()
        self._scratch_spaces = ConsoleScratchSpaceManager()
        self._raw_cli_refusal_stash_bank: dict[str, list[Any]] = {}
        self._voice_dispatch_supervisor: VoiceDispatchSupervisor | None = None
        self._voice_worker = None
        self._voice_process_supervisor = None
        self._voice_promotion_owner: VoicePromotionOwner | None = None
        self._voice_promotion_pending_closes: dict[
            str, VoicePromotionSessionCloseToken | None
        ] = {}
        self._persona_buddy_sink = PersonaBuddyConsoleAdapter(
            getattr(app, "persona_buddy_controller", None)
        )
        #: The app-owned run-hooks engine, `_UNSET` until one is BUILT --
        #: and then latched for the app lifetime (spec section 4
        #: singleton). "No [hooks] configured" is never stored: it is a
        #: per-call `None` answer that `ensure_run_hooks` re-decides
        #: while unconfigured (Ruling R17), hence the sentinel.
        self._run_hooks_engine: Any = _UNSET
        self._run_hooks_lock = RLock()
        self._hook_permissions: HookPermissions | None = None
        self._preparation_reads: set[Any] = set()
        # V2 sessions share the app loop and budgets, including viewless work.
        self._hooks_v2_budget_owner: Any = None
        self._hooks_v2_engines: dict[str, Any] = {}
        self._hooks_v2_lifecycles: dict[str, Any] = {}
        self._hooks_v2_configured: dict[str, Any] = {}
        self._hooks_v2_cleanup_task: asyncio.Task[Any] | None = None
        #: The view (a `ChatScreen`) currently attached, or `None` while the
        #: runtime is VIEWLESS -- which is now a real, supported state, not
        #: a transient. Written only by `attach_view`/`detach_view`.
        #:
        #: Two `ChatScreen`s are briefly alive at once whenever a navigation
        #: lands back on Console: `_complete_screen_navigation` constructs
        #: and `restore_state`s the incoming screen BEFORE `switch_screen`
        #: unmounts the outgoing one (`app.py`), and `restore_state` reaches
        #: `ensure_chat_store`. The incoming screen therefore attaches
        #: first and this attribute names it; the outgoing screen's later
        #: detach sees a different claimant and does nothing.
        self.view: Any | None = None
        self._attachment_generation = 0
        self._attached_generation: int | None = None
        self._reconciled_view: Any | None = None
        #: Latched by `dispose()` (app exit). Every `ensure_*` returns what
        #: it already holds afterwards and builds nothing new -- see
        #: `dispose` for why a rebuild during quit is the hazard.
        self._disposed: bool = False
        self._start_canvas_policy_watcher()
        self.authority_token = str(uuid4())
        database = getattr(app, "chachanotes_db", None)
        database_path = getattr(database, "db_path", None)
        self.profile_authority = (
            str(Path(database_path).expanduser().resolve(strict=False))
            if database_path and str(database_path) != ":memory:"
            else ""
        )
        #: Task-lifetime records only. Controller/store stay authoritative for
        #: run state, queueing, terminalization, and transcript state.
        self._turn_custody: dict[str, _ConsoleTurnCustodyRecord] = {}
        self._turn_recoveries: dict[str, ConsoleTurnRecoveryEntry] = {}
        self._recovery_turns_by_session: dict[str, list[str]] = {}
        self._recovery_order = 0
        self._staged_evidence = _ConsoleStagedEvidenceState()
        self._prompt_history: Any | None = None
        self._admission_fenced_sessions: set[str] = set()
        self._project_decision_lock = threading.RLock()
        self._project_decision_order = 0
        self._project_binding_decisions: dict[str, _ProjectBindingDecision] = {}
        self._project_dispatch_decisions: dict[str, _ProjectDispatchDecision] = {}
        # Content-free shell projection. Durable receipt IDs remain owned by
        # ConversationLocalMarksService; decision IDs remain owned by the
        # controller's Task 5 announcement registry. These sets only suppress
        # duplicate terminal toasts inside this app process.
        self._attention_lock = threading.RLock()
        self._attention_operation_lock = threading.RLock()
        self._attention_revision = 0
        self._console_needs_attention: bool | None = None
        self._last_known_terminal_marks: tuple[tuple[str, str], ...] | None = None
        self._rendered_receipt_ack_owner: tuple[Any, Any, int | None] | None = None
        self._rendered_receipt_acks: set[tuple[str, str]] = set()
        self._notified_terminal_receipts: set[str] = set()
        self._notifying_terminal_receipts: set[str] = set()
        #: Bumped by every `dispose()` -- i.e. once per app run, not once
        #: per navigation. `Tests/UI/test_console_runtime_ownership.py`
        #: reads it to prove the runtime survived a visit.
        self.generation: int = 0

    # -- non-constructing accessors ---------------------------------------

    @property
    def app(self) -> Any:
        """The app this runtime is bound to."""
        return self._app

    @property
    def chat_store(self) -> "ConsoleChatStore | None":
        """The built store, or `None` if nothing has asked for one yet."""
        return self._chat_store

    @property
    def provider_gateway(self) -> Any | None:
        """The built provider gateway, or `None`."""
        return self._provider_gateway

    @property
    def agent_bridge(self) -> Any | None:
        """The built agent bridge, or `None` (also `None` when resolved to
        "no agent runtime" — see `ensure_agent_bridge`)."""
        return self._agent_bridge

    @property
    def chat_controller(self) -> "ConsoleChatController | None":
        """The built chat controller, or `None`."""
        return self._chat_controller

    @property
    def run_hooks_engine(self) -> "RunHooksEngine | None":
        """The built run-hooks engine, or `None`.

        `None` covers "nothing built yet" and "unconfigured" alike -- the
        property is a peek, not the latch; `ensure_run_hooks` is the one
        that decides (and, while unconfigured, keeps re-deciding, R17).
        """
        engine = self._run_hooks_engine
        return None if engine is _UNSET else engine

    @property
    def activity_receipts(self) -> Any | None:
        """The built app-lifetime receipt coordinator, if available."""
        return self._activity_receipts

    @property
    def scratch_spaces(self) -> ConsoleScratchSpaceManager:
        """The process-lifetime scratch authority shared by Console visits."""
        return self._scratch_spaces

    @property
    def raw_cli_refusal_stash_bank(self) -> dict[str, list[Any]]:
        """The process-memory refusal bank shared by Console visits."""
        return self._raw_cli_refusal_stash_bank

    @property
    def accepts_raw_cli_refusal_callbacks(self) -> bool:
        """Whether raw CLI completion callbacks may still mutate UI state."""
        return not self._disposed

    @property
    def voice_dispatch_supervisor(self) -> VoiceDispatchSupervisor:
        """The app-lifetime hands-free cleanup quarantine."""

        if self._voice_dispatch_supervisor is None:
            if self._disposed:
                raise RuntimeError("voice_runtime_disposed")
            from tldw_chatbook.Chat.console_voice_supervisor import (
                VoiceDispatchSupervisor,
            )

            self._voice_dispatch_supervisor = VoiceDispatchSupervisor()
        return self._voice_dispatch_supervisor

    @property
    def voice_worker(self):
        """One lazy app-owned loop shared by view-owned voice sessions."""
        if self._disposed:
            raise RuntimeError("voice_runtime_disposed")
        if self._voice_worker is None:
            from tldw_chatbook.Chat.console_voice_worker import ConsoleVoiceWorker

            self._voice_worker = ConsoleVoiceWorker()
        return self._voice_worker

    @property
    def voice_process_supervisor(self):
        """One lazy app-owned device lease and parent cleanup custodian."""
        if self._disposed:
            raise RuntimeError("voice_runtime_disposed")
        if self._voice_process_supervisor is None:
            from tldw_chatbook.Chat.console_voice_process import VoiceProcessSupervisor

            self._voice_process_supervisor = VoiceProcessSupervisor()
        return self._voice_process_supervisor

    @property
    def voice_promotion_owner(self) -> VoicePromotionOwner:
        """The single app-lifetime owner for claimed voice publications."""

        if self._voice_promotion_owner is None:
            if self._disposed:
                raise RuntimeError("voice_runtime_disposed")
            from tldw_chatbook.Chat.console_voice_promotion import VoicePromotionOwner

            self._voice_promotion_owner = VoicePromotionOwner(lambda: self._chat_store)
            # A first voice entry may arrive while an ordinary close awaits
            # its drain. Transfer those exact session fences synchronously.
            for session_id in self._voice_promotion_pending_closes:
                self._voice_promotion_pending_closes[session_id] = (
                    self._voice_promotion_owner.begin_session_close(session_id)
                )
        return self._voice_promotion_owner

    def trace_compatibility_snapshot(self) -> Mapping[str, int]:
        """Return the runtime's content-free semantic-trace rollout totals.

        Returns:
            A fixed-key snapshot containing only compatibility event counts.
        """

        return self.trace_compatibility_metrics.snapshot()

    @property
    def persona_buddy_sink(self) -> PersonaBuddyConsoleAdapter:
        """The app-owned, screen-free sink for trusted Console state."""
        self._persona_buddy_sink.bind_controller(
            getattr(self._app, "persona_buddy_controller", None)
        )
        return self._persona_buddy_sink

    @property
    def change_review_coordinator(self) -> Any | None:
        """The built app-owned Change Review coordinator, if available."""
        return self._change_review_coordinator

    @property
    def canvas_controller(self) -> Any | None:
        """The single process-runtime Canvas lifecycle owner."""

        return self._canvas_controller

    @property
    def canvas_gateway(self) -> Any | None:
        """The one lazy native Canvas browser gateway for this app runtime."""

        return self._canvas_gateway

    @property
    def console_needs_attention(self) -> bool:
        """Return the current content-free Console attention projection."""
        return bool(self._console_needs_attention)

    def _console_local_marks_service(self) -> Any | None:
        """Return the app/store-owned local mark service without constructing it."""
        service = getattr(self._app, "conversation_local_marks_service", None)
        if service is not None:
            return service
        persistence = getattr(self._chat_store, "persistence", None)
        return getattr(persistence, "local_marks", None)

    def _hidden_decision_ids(self) -> frozenset[str]:
        """Snapshot unresolved decisions that have no mounted answerable card."""
        controller = self._chat_controller
        if controller is None:
            return frozenset()
        approval_lock = getattr(controller, "_approval_state_lock", None)
        approval_registry = getattr(controller, "_pending_approval_rounds", None)
        if approval_lock is None or approval_registry is None:
            return frozenset(
                getattr(controller, "_announced_pending_decision_ids", ())
            )
        with approval_lock:
            answerable = set(
                getattr(controller, "_answerable_decision_by_session", {}).values()
            )
            pending = {
                str(decision_id)
                for decision_id, state in approval_registry.items()
                if not state.get("settled")
            }
        # Preserve Task 5's type-lock -> approval-lock order by never nesting:
        # each sibling registry is snapshotted under its own lock, then the
        # shared answerable snapshot above is applied after every lock is free.
        for lock_name, registry_name in (
            ("_pending_skill_install_lock", "_pending_skill_install_rounds"),
            ("_pending_skill_script_lock", "_pending_skill_script_rounds"),
        ):
            registry_lock = getattr(controller, lock_name, None)
            registry = getattr(controller, registry_name, None)
            if registry_lock is None or registry is None:
                continue
            with registry_lock:
                pending.update(
                    str(decision_id)
                    for decision_id, state in registry.items()
                    if not state.get("settled")
                )
        return frozenset(pending - answerable)

    def _reserve_attention_revision(self) -> int:
        """Invalidate older attention reads before waiting to serialize work."""
        with self._attention_lock:
            self._attention_revision += 1
            return self._attention_revision

    def _attention_revision_is_current(self, revision: int) -> bool:
        with self._attention_lock:
            return revision == self._attention_revision

    def _terminal_receipt_outcome(
        self, conversation_id: str, receipt_id: str
    ) -> bool | None:
        """Return failed/completed, or ``None`` until the exact row is known."""
        marks = self._console_local_marks_service()
        outcome_reader = getattr(marks, "console_terminal_outcome", None)
        if callable(outcome_reader):
            try:
                companion_outcome = outcome_reader(conversation_id, receipt_id)
            except Exception as exc:  # noqa: BLE001 -- retry on next recompute
                logger.debug(
                    "Console terminal-outcome companion read failed "
                    "(exception_type={})",
                    type(exc).__name__,
                )
                return None
            if companion_outcome == "failed":
                return True
            if companion_outcome == "complete":
                return False

        # Backward-compatible fallback for receipts written before outcome
        # companions existed. It is intentionally exact: a newer receipt in
        # the same row cannot classify an older still-unseen receipt.
        database = getattr(self._app, "chachanotes_db", None)
        reader = getattr(database, "get_messages_for_conversation", None)
        if not callable(reader):
            return None
        from tldw_chatbook.Chat.message_metadata import MessageMetadata
        from tldw_chatbook.Video_Generation.video_metadata import (
            VideoGenerationMetadata,
        )

        page_size = 256
        offset = 0
        while True:
            try:
                rows = tuple(
                    reader(
                        conversation_id,
                        limit=page_size,
                        offset=offset,
                        order_by_timestamp="DESC",
                        include_image_data=False,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- retry on the next recompute
                logger.debug(
                    "Console terminal-attention classification failed "
                    "(exception_type={})",
                    type(exc).__name__,
                )
                return None
            for row in rows:
                metadata_json = row.get("metadata_json")
                ordinary = MessageMetadata.from_json(metadata_json)
                video = VideoGenerationMetadata.from_json(metadata_json)
                row_receipt = (
                    video.terminal_receipt_id
                    if video is not None
                    else ordinary.terminal_receipt_id
                    if ordinary is not None
                    else ""
                )
                if row_receipt != receipt_id:
                    continue
                terminal_state = row.get("assistant_generation_state")
                if terminal_state == "failed":
                    return True
                if terminal_state == "complete":
                    return False
                return None
            if len(rows) < page_size:
                return None
            offset += len(rows)

    def _receipt_is_on_screen(self, conversation_id: str) -> bool:
        """Whether the receipt's conversation is the tab the user is looking at.

        TASK-34100.5 AC#8 / task-33620.6: "hidden" is decided when the turn
        completes -- Console is the current screen AND the receipt's
        conversation is its active tab. The attached view acknowledges the
        receipt once it renders the row; until then it is not "hidden".
        Review round 2 (V2-F1): a modal over Console (Alt+M's picker, a
        rename dialog, the command palette) leaves Console on screen.
        """
        if not conversation_id or not self._console_is_on_screen():
            return False
        store = self._chat_store
        active_id = getattr(store, "active_session_id", None)
        if not active_id:
            return False
        try:
            sessions = tuple(store.sessions())
        except Exception:  # noqa: BLE001 -- unknown visibility stays "hidden"
            return False
        return any(
            session.id == active_id
            and getattr(session, "persisted_conversation_id", None) == conversation_id
            for session in sessions
        )

    def _console_is_on_screen(self) -> bool:
        """The attached view is current, or only modal screens cover it.

        A modal suspends the view (``_reconciled_view`` is cleared), but the
        screen still shows behind it, so only the stack above it counts.
        """
        if self.has_answerable_view():
            return True
        view = self.view
        if view is None:
            return False
        try:
            from textual.screen import ModalScreen

            stack = list(view.app.screen_stack)
            index = next(i for i, screen in enumerate(stack) if screen is view)
        except Exception:  # noqa: BLE001 -- not in the stack: not on screen
            return False
        return all(isinstance(screen, ModalScreen) for screen in stack[index + 1 :])

    def _request_visible_receipt_render(self) -> None:
        """Re-arm the on-screen view's transcript sync (thread-safe post)."""
        start = getattr(self.view, "_start_console_transcript_sync_timer", None)
        call_later = getattr(self._app, "call_later", None)
        if not callable(start) or not callable(call_later):
            return
        try:
            call_later(start)
        except RuntimeError:  # a closing scheduler: the mark stays durable
            pass

    def _notify_terminal_receipt(
        self, conversation_id: str, receipt_id: str, *, revision: int
    ) -> bool:
        """Emit one sanitized terminal notice, recording only on success."""
        notify = getattr(self._app, "notify", None)
        if not callable(notify) or self._receipt_is_on_screen(conversation_id):
            return False
        with self._attention_lock:
            if (
                revision != self._attention_revision
                or receipt_id in self._notified_terminal_receipts
                or receipt_id in self._notifying_terminal_receipts
            ):
                return False
            self._notifying_terminal_receipts.add(receipt_id)
        failed = self._terminal_receipt_outcome(conversation_id, receipt_id)
        if failed is None:
            with self._attention_lock:
                self._notifying_terminal_receipts.discard(receipt_id)
            return False
        with self._attention_lock:
            if revision != self._attention_revision:
                self._notifying_terminal_receipts.discard(receipt_id)
                return False
        message = (
            "A Console turn failed while hidden. Return to Console to review."
            if failed
            else "A Console turn completed while hidden. Return to Console to review."
        )
        severity = "error" if failed else "information"
        delivered = False
        recorded = False
        try:
            try:
                notify(message, severity=severity)
            except TypeError:
                notify(message)
            delivered = True
        except Exception as exc:  # noqa: BLE001 -- finalization never depends on toast
            logger.debug(
                "Console terminal-attention notice failed "
                "(exception_type={})",
                type(exc).__name__,
            )
        finally:
            with self._attention_lock:
                self._notifying_terminal_receipts.discard(receipt_id)
                receipt_is_still_active = any(
                    active_receipt_id == receipt_id
                    for _conversation_id, active_receipt_id in (
                        self._last_known_terminal_marks or ()
                    )
                )
                if delivered and receipt_is_still_active:
                    self._notified_terminal_receipts.add(receipt_id)
                    recorded = True
        return recorded

    def recompute_console_attention(self, *, force_projection: bool = False) -> bool:
        """Re-derive and publish the one boolean Console attention state.

        Durable receipt marks and Task 5's hidden-decision registry stay the
        authorities. Only fixed copy and the boolean projection leave this
        runtime; receipt/decision IDs are never handed to shell widgets.
        """
        from contextlib import ExitStack

        from tldw_chatbook.Chat.conversation_local_marks_service import (
            ConversationLocalMarksService,
        )
        from tldw_chatbook.DB.base_db import operation_owned_connection
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        revision = self._reserve_attention_revision()
        with self._attention_operation_lock, ExitStack() as connections:
            if not self._attention_revision_is_current(revision):
                return self.console_needs_attention

            service = self._console_local_marks_service()
            if (
                type(service) is ConversationLocalMarksService
                and type(service.db) is CharactersRAGDB
                and not service.db.is_memory_db
            ):
                # Pending-decision callbacks may run on a short-lived worker.
                # Its new marks/outcome handle belongs to this complete read;
                # an existing caller connection or transaction stays borrowed.
                connections.enter_context(operation_owned_connection(service.db))
            list_marks = getattr(service, "list_console_unseen_marks", None)
            marks_known = False
            marks: tuple[tuple[str, str], ...]
            if not callable(list_marks):
                # Apps without the durable local-mark service have no durable
                # receipt source. This is a known-empty capability, unlike a
                # configured reader that transiently fails.
                marks = ()
                marks_known = True
            else:
                try:
                    marks = tuple(list_marks())
                    marks_known = True
                except Exception as exc:  # noqa: BLE001 -- preserve last known state
                    logger.debug(
                        "Console terminal-attention query failed (exception_type={})",
                        type(exc).__name__,
                    )
                    with self._attention_lock:
                        marks = self._last_known_terminal_marks or ()

            if not self._attention_revision_is_current(revision):
                return self.console_needs_attention
            if marks_known:
                with self._attention_lock:
                    self._last_known_terminal_marks = marks
                    # A durable receipt observed again owns another acknowledgement,
                    # even if an earlier repaint already cleared its exact identity.
                    self._rendered_receipt_acks.difference_update(marks)
                for conversation_id, receipt_id in marks:
                    self._notify_terminal_receipt(
                        conversation_id,
                        receipt_id,
                        revision=revision,
                    )
                    if not self._attention_revision_is_current(revision):
                        return self.console_needs_attention

            hidden_decisions = self._hidden_decision_ids()
            if not self._attention_revision_is_current(revision):
                return self.console_needs_attention

            active_receipts = {receipt_id for _conversation_id, receipt_id in marks}
            # A receipt in the visible, active tab is not "away" work: no '!'.
            # The view's transcript poll can stop a tick before the receipt
            # lands, so ask it to render (and thereby acknowledge) the row;
            # otherwise the mark lingers and turns "hidden" on navigation.
            hidden_marks = [
                pair for pair in marks if not self._receipt_is_on_screen(pair[0])
            ]
            if len(hidden_marks) != len(marks) and self.has_answerable_view():
                self._request_visible_receipt_render()
            with self._attention_lock:
                if marks_known:
                    self._notified_terminal_receipts.intersection_update(
                        active_receipts
                    )
                previous = self._console_needs_attention
                if marks_known or self._last_known_terminal_marks is not None:
                    needs_attention = bool(hidden_marks or hidden_decisions)
                elif hidden_decisions:
                    needs_attention = True
                else:
                    # An unknown first read cannot establish an empty durable
                    # set. Preserve the last projection (the shell defaults
                    # false) until a successful query supplies evidence.
                    return bool(previous)
                changed = needs_attention != previous
                self._console_needs_attention = needs_attention

            publish = getattr(self._app, "set_console_attention_projection", None)
            if callable(publish) and (changed or force_projection):

                def _publish() -> None:
                    with self._attention_lock:
                        # Read revisions fence authority queries above. A
                        # same-value query must not invalidate this sole queued
                        # projection: only a newer opposite value makes the
                        # captured boolean stale at actual UI delivery.
                        if needs_attention != self._console_needs_attention:
                            return
                        try:
                            publish(needs_attention)
                        except Exception as exc:  # noqa: BLE001 -- projection is advisory
                            logger.debug(
                                "Console attention projection failed "
                                "(exception_type={})",
                                type(exc).__name__,
                            )

                call_later = getattr(self._app, "call_later", None)
                if callable(call_later):
                    try:
                        # Textual posts this callback thread-safely without
                        # waiting for the UI. Recompute may be nested inside
                        # attach/detach/ack's operation lock; a synchronous
                        # call_from_thread here would deadlock that UI owner.
                        call_later(_publish)
                    except RuntimeError:
                        # A closing scheduler cannot accept projection work.
                        # Durable state remains intact; never block as fallback.
                        pass
                else:
                    # Headless consumers have no UI message pump to marshal to.
                    _publish()
            return needs_attention

    def acknowledge_rendered_terminal_receipts(
        self,
        rendered: tuple[tuple[str, str], ...],
        *,
        view: Any | None = None,
        attachment_generation: int | None = None,
    ) -> tuple[str, ...]:
        """Acknowledge exact durable receipts, serialized with attention reads."""
        if not rendered:
            return ()
        attached_view = self.view
        attached_generation = self._attached_generation
        if (
            attached_view is not None
            or view is not None
            or attachment_generation is not None
        ) and (
            view is not attached_view
            or attachment_generation != attached_generation
        ):
            return ()
        service = self._console_local_marks_service()
        acknowledge = getattr(service, "acknowledge_console_unseen", None)
        if not callable(acknowledge):
            return ()
        self._reserve_attention_revision()
        with self._attention_operation_lock:
            attached_view = self.view
            attached_generation = self._attached_generation
            if (
                attached_view is not None
                or view is not None
                or attachment_generation is not None
            ) and (
                view is not attached_view
                or attachment_generation != attached_generation
            ):
                return ()
            acknowledged: list[str] = []
            removed_pairs: set[tuple[str, str]] = set()
            owner = (service, view, attachment_generation)
            prior_owner = self._rendered_receipt_ack_owner
            if (
                prior_owner is None
                or prior_owner[0] is not service
                or prior_owner[1] is not view
                or prior_owner[2] != attachment_generation
            ):
                self._rendered_receipt_acks.clear()
                self._rendered_receipt_ack_owner = owner
            # Bound this cache to the current mounted receipt set. Exceptions
            # remain retryable; only an exact completed durable delete is reused.
            self._rendered_receipt_acks.intersection_update(rendered)
            for conversation_id, receipt_id in rendered:
                if not conversation_id or not receipt_id:
                    continue
                pair = (conversation_id, receipt_id)
                if pair in self._rendered_receipt_acks:
                    continue
                try:
                    result = acknowledge(conversation_id, receipt_id)
                    removed = result is True
                except Exception as exc:  # noqa: BLE001 -- mark remains retryable
                    logger.debug(
                        "Console terminal-attention acknowledgement failed "
                        "(exception_type={})",
                        type(exc).__name__,
                    )
                    continue
                if result is True or result is False:
                    self._rendered_receipt_acks.add(pair)
                if removed:
                    acknowledged.append(receipt_id)
                    removed_pairs.add((conversation_id, receipt_id))
            if acknowledged:
                with self._attention_lock:
                    self._notified_terminal_receipts.difference_update(acknowledged)
                    if self._last_known_terminal_marks is not None:
                        self._last_known_terminal_marks = tuple(
                            pair
                            for pair in self._last_known_terminal_marks
                            if pair not in removed_pairs
                        )
                self.recompute_console_attention()
            return tuple(acknowledged)

    # -- handle writes (the screen's properties, and 59 test sites) --------

    def set_chat_store(self, value: Any) -> None:
        """Replace the store handle (a test double, or `None` to rebuild)."""
        self._chat_store = value
        if value is not None and hasattr(value, "on_active_session_changed"):
            value.on_active_session_changed = self._on_active_session_changed

    def set_provider_gateway(self, value: Any) -> None:
        """Replace the provider-gateway handle."""
        self._provider_gateway = value

    @property
    def execution_capacity(self) -> RuntimeCapacity:
        """Return shared admission, allocating once unless runtime disposal won.

        Returns:
            The runtime's capacity, including its closed snapshot after disposal.

        Raises:
            RuntimeError: A disposed runtime has no existing capacity.
        """
        return self._get_execution_capacity()

    def _get_execution_capacity(self) -> RuntimeCapacity:
        with self._execution_capacity_lock:
            if self._execution_capacity is None:
                if self._disposed:
                    raise RuntimeError("runtime capacity is disposed")
                from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity

                self._execution_capacity = RuntimeCapacity.from_settings()
            return self._execution_capacity

    def set_agent_bridge(self, value: Any) -> None:
        """Replace the bridge and lazily bind native execution admission.

        Args:
            value: Native bridge, compatible test double, or None to rebuild.

        Raises:
            RuntimeError: Native admission would bind to disposed ownership.
            ValueError: Rebinding would detach an active bridge's capacity.
        """
        from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

        if isinstance(value, ConsoleAgentBridge):
            with self._execution_capacity_lock:
                if self._disposed:
                    raise RuntimeError("runtime capacity is disposed")
                existing_capacity = self._execution_capacity
            value.bind_runtime_capacity(
                self._get_execution_capacity, existing_capacity=existing_capacity
            )
        if self._agent_bridge is not value:
            self._begin_progress_cleanup(self._agent_bridge)
        self._agent_bridge = value

    def _begin_progress_cleanup(self, bridge: Any) -> asyncio.Task[Any] | None:
        """Revoke immediately and retain only the old bridge's physical close leaf."""
        begin_close = getattr(bridge, "begin_close_all_progress", None)
        wait_close = getattr(bridge, "await_all_progress_closed", None)
        if callable(begin_close) and callable(wait_close):
            begin_close()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Constructor/CLI compatibility has no GUI loop to keep responsive.
                bridge.close_all_progress()
                return None
            task = loop.create_task(wait_close(), name="console-retired-progress-close")
            self._progress_cleanup_tasks.add(task)

            def settled(done: asyncio.Task[Any]) -> None:
                self._progress_cleanup_tasks.discard(done)
                self._consume_task_outcome(done)

            task.add_done_callback(settled)
            return task
        close_progress = getattr(bridge, "close_all_progress", None)
        if callable(close_progress):
            close_progress()
        return None

    def set_chat_controller(self, value: Any) -> None:
        """Replace the chat-controller handle."""
        self._chat_controller = value
        if value is not None:
            from .console_preparation_reads import observe_preparation_reads

            reads = getattr(value, "_preparation_reads", None)
            if reads is not None:
                observe_preparation_reads(reads, self._preparation_reads)
            value._hooks_v2_runtime = self
        if value is not None and self._app is not None:
            value.app = self._app
        if value is not None:
            # These are domain-admission seams, not detachable view hooks:
            # ``None`` makes a newly armed skill round deny immediately and
            # can remove ``run_skill_script`` during detached assembly.
            # Stable runtime routers keep the round/tool alive while their
            # projection is absent and retain no screen.
            value.set_pending_skill_install = functools.partial(
                self._project_to_attached_view, "set_pending_skill_install"
            )
            value.set_pending_skill_script = functools.partial(
                self._project_to_attached_view, "set_pending_skill_script"
            )
            value.set_pending_decision = self._project_pending_decision_to_attached_view
            value.on_console_attention_changed = self.recompute_console_attention
        coordinator = getattr(value, "prompt_queue_coordinator", None)
        bind_submitter = getattr(coordinator, "bind_runtime_submitter", None)
        if callable(bind_submitter):
            bind_submitter(self._submit_queued_turn)
            coordinator.bind_continuation_admission(
                self._continuation_admission_current
            )
        fleet_wake = getattr(value, "fleet_wake", None)
        bind_wake_submitter = getattr(fleet_wake, "bind_runtime_submitter", None)
        if callable(bind_wake_submitter):
            bind_wake_submitter(self._submit_fleet_wake)

    def _continuation_admission_current(
        self, request: ConsoleTurnCustodyRequest
    ) -> bool:
        """Refuse old-revision follow-ups while the reviewed plugin drain waits."""
        rows = request.configuration.skill_context_maximum.get("available_skills", ())
        plugin_rows = [row for row in rows if row.get("plugin_installation_id")]
        if not plugin_rows:
            return True
        skills = getattr(self._chat_controller, "_skills_service", None)
        local = getattr(skills, "local_service", None)
        service = getattr(local, "plugin_service", None)
        if service is None:
            return False
        from tldw_chatbook.Plugins.admission import PluginUnavailable

        try:
            for row in plugin_rows:
                service.fences.require_admission(
                    row["plugin_installation_id"], row["plugin_revision"]
                )
                service.fences.check(
                    row["plugin_installation_id"], row.get("plugin_workspace_id")
                )
        except (PluginUnavailable, KeyError):
            return False
        return True

    def _on_active_session_changed(self) -> None:
        """Re-derive app-owned decisions after the store's authoritative swap."""
        controller = self._chat_controller
        callback = getattr(controller, "active_session_changed", None)
        if callable(callback):
            callback()
        self._schedule_project_instruction_projection()

    def ensure_prompt_history(self) -> Any:
        """Return the app-owned prompt-history sink shared by every view."""
        if self._prompt_history is None:
            factory = getattr(self._app, "console_prompt_history_factory", None)
            if callable(factory):
                self._prompt_history = factory()
            else:
                from tldw_chatbook.Chat.prompt_history import PromptHistory

                self._prompt_history = PromptHistory()
        return self._prompt_history

    async def _select_project_instruction_binding(
        self, session_id: str, selections: tuple[Any, ...], recovery_code: str
    ) -> tuple[str, str | None]:
        """Retain project authority until an attached Console resolves it."""
        if self._disposed or not self._session_exists(session_id):
            return "cancel", None
        choices = tuple(
            ConsoleProjectBindingChoice(
                binding_id=str(selection.binding.binding_id),
                label=str(
                    getattr(selection.binding, "display_name", "")
                    or getattr(selection.binding, "label", "")
                    or f"Folder {index + 1}"
                ),
                eligible=True,
            )
            for index, selection in enumerate(selections)
        )
        if not choices:
            choices = (
                ConsoleProjectBindingChoice(
                    binding_id="",
                    label="No eligible folders",
                    eligible=False,
                    recovery=recovery_code,
                ),
            )
        with self._project_decision_lock:
            self._project_decision_order += 1
            decision = _ProjectBindingDecision(
                decision_id=str(uuid4()),
                session_id=session_id,
                order=self._project_decision_order,
                choices=choices,
                completed=asyncio.Event(),
            )
            self._project_binding_decisions[decision.decision_id] = decision
        self._project_pending_project_instruction_decisions()
        try:
            while not decision.completed.is_set():
                if self._disposed or not self._session_exists(session_id):
                    self.resolve_project_instruction_binding(
                        decision.decision_id, "cancel", None
                    )
                    break
                try:
                    await asyncio.wait_for(
                        decision.completed.wait(),
                        timeout=_PROJECT_INSTRUCTION_NOTICE_POLL_SECONDS,
                    )
                except TimeoutError:
                    continue
            return decision.result
        except asyncio.CancelledError:
            raise
        finally:
            with self._project_decision_lock:
                self._project_binding_decisions.pop(decision.decision_id, None)
            self._project_pending_project_instruction_decisions()

    def _confirm_project_instruction_dispatch(self, notice: Any) -> str:
        """Retain a dispatch decision independently of its Console modal."""
        if self._disposed or not self._session_exists(notice.session_id):
            return "cancel"
        active_cancel_events = getattr(
            self._chat_controller, "_active_cancel_events", {}
        )
        owning_cancel_event = active_cancel_events.get(notice.session_id)
        timeout = max(
            0.0,
            float(
                getattr(
                    self,
                    "_project_instruction_notice_timeout_seconds",
                    _PROJECT_INSTRUCTION_NOTICE_TIMEOUT_SECONDS,
                )
            ),
        )
        with self._project_decision_lock:
            self._project_decision_order += 1
            decision = _ProjectDispatchDecision(
                decision_id=str(uuid4()),
                session_id=notice.session_id,
                order=self._project_decision_order,
                notice=notice,
                completed=threading.Event(),
                owning_cancel_event=owning_cancel_event,
                answerable_seconds_remaining=timeout,
            )
            self._project_dispatch_decisions[decision.decision_id] = decision
        self._schedule_project_instruction_projection()
        try:
            while not decision.completed.is_set():
                if (
                    self._disposed
                    or not self._session_exists(decision.session_id)
                    or (
                        decision.owning_cancel_event is not None
                        and decision.owning_cancel_event.is_set()
                    )
                ):
                    self.resolve_project_instruction_dispatch(
                        decision.decision_id, "cancel"
                    )
                    break
                with self._project_decision_lock:
                    remaining = decision.answerable_seconds_remaining
                    if decision.answerable_since is not None:
                        remaining -= time.monotonic() - decision.answerable_since
                if decision.answerable_since is not None and remaining <= 0:
                    self.resolve_project_instruction_dispatch(
                        decision.decision_id, "cancel"
                    )
                    break
                decision.completed.wait(
                    min(
                        _PROJECT_INSTRUCTION_NOTICE_POLL_SECONDS,
                        max(0.001, remaining),
                    )
                    if decision.answerable_since is not None
                    else _PROJECT_INSTRUCTION_NOTICE_POLL_SECONDS
                )
            return decision.result
        finally:
            with self._project_decision_lock:
                self._project_dispatch_decisions.pop(decision.decision_id, None)
            self._schedule_project_instruction_projection()

    def _session_exists(self, session_id: str) -> bool:
        store = self._chat_store
        sessions = store.sessions() if store is not None else ()
        return any(session.id == session_id for session in sessions)

    def resolve_project_instruction_binding(
        self, decision_id: str, action: str, binding_id: str | None
    ) -> bool:
        """Resolve one exact app-owned binding decision."""
        with self._project_decision_lock:
            decision = self._project_binding_decisions.get(decision_id)
            if decision is None or decision.completed.is_set():
                return False
            if action not in {"select", "disable", "cancel"}:
                return False
            if action == "select" and binding_id not in {
                choice.binding_id for choice in decision.choices if choice.eligible
            }:
                return False
            decision.result = (
                action,
                binding_id if action == "select" else None,
            )
            decision.completed.set()
            return True

    def resolve_project_instruction_dispatch(
        self, decision_id: str, action: str
    ) -> bool:
        """Resolve one exact app-owned dispatch decision."""
        with self._project_decision_lock:
            decision = self._project_dispatch_decisions.get(decision_id)
            if decision is None or decision.completed.is_set():
                return False
            if action not in {"proceed", "cancel", "disable"}:
                return False
            decision.result = action
            decision.completed.set()
            return True

    def _schedule_project_instruction_projection(self) -> None:
        """Marshal retained decision projection onto the app loop."""
        active_session_id = getattr(self._chat_store, "active_session_id", None)
        with self._project_decision_lock:
            for decision in (
                *self._project_binding_decisions.values(),
                *self._project_dispatch_decisions.values(),
            ):
                if decision.session_id != active_session_id:
                    continue
                decision.retry_budget_generation = None
                decision.retry_attempts = 0
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            call_from_thread = getattr(self._app, "call_from_thread", None)
            if callable(call_from_thread):
                try:
                    call_from_thread(self._project_pending_project_instruction_decisions)
                    return
                except RuntimeError:
                    return
        self._project_pending_project_instruction_decisions()

    def _pause_project_dispatch_locked(self, decision: Any) -> None:
        """Stop one dispatch decision's answerable-time clock."""
        if not isinstance(decision, _ProjectDispatchDecision):
            return
        if decision.answerable_since is not None:
            decision.answerable_seconds_remaining = max(
                0.0,
                decision.answerable_seconds_remaining
                - (time.monotonic() - decision.answerable_since),
            )
            decision.answerable_since = None

    def _pause_project_instruction_generation(self, generation: int | None) -> None:
        """Pause and unclaim decisions projected by one retiring view."""
        if generation is None:
            return
        with self._project_decision_lock:
            for decision in (
                *self._project_binding_decisions.values(),
                *self._project_dispatch_decisions.values(),
            ):
                if decision.projected_generation != generation:
                    continue
                self._pause_project_dispatch_locked(decision)
                decision.projected_generation = None

    def _project_decision_for_id_locked(self, decision_id: str) -> Any | None:
        return self._project_binding_decisions.get(
            decision_id
        ) or self._project_dispatch_decisions.get(decision_id)

    def _schedule_project_instruction_projection_retry(
        self, decision: Any, generation: int
    ) -> None:
        """Schedule one paced retry within this attachment's bounded budget."""
        with self._project_decision_lock:
            if decision.retry_generation == generation:
                return
            if decision.retry_budget_generation != generation:
                decision.retry_budget_generation = generation
                decision.retry_attempts = 0
            if decision.retry_attempts >= len(
                _PROJECT_INSTRUCTION_PROJECTION_RETRY_DELAYS
            ):
                return
            delay = _PROJECT_INSTRUCTION_PROJECTION_RETRY_DELAYS[
                decision.retry_attempts
            ]
            decision.retry_attempts += 1
            decision.retry_generation = generation
            decision_id = decision.decision_id

        def retry() -> None:
            if (
                self._attached_generation != generation
                or self.view is None
                or self._reconciled_view is not self.view
            ):
                return
            with self._project_decision_lock:
                if self._project_decision_for_id_locked(decision_id) is not decision:
                    return
                if decision.retry_generation != generation:
                    return
                # This scheduled attempt is now being consumed. Clearing its
                # claim before projection lets another failure enqueue one
                # successor while still preventing concurrent duplicates.
                decision.retry_generation = None
            self._project_pending_project_instruction_decisions()

        set_timer = getattr(self._app, "set_timer", None)
        call_later = getattr(self._app, "call_later", None)
        try:
            if callable(set_timer):
                set_timer(delay, retry)
            elif callable(call_later):
                call_later(retry)
            else:
                raise RuntimeError("app has no projection retry scheduler")
        except Exception:  # noqa: BLE001 -- later reconciliation may retry
            with self._project_decision_lock:
                if decision.retry_generation == generation:
                    decision.retry_generation = None
                    decision.retry_attempts = max(0, decision.retry_attempts - 1)

    def _project_pending_project_instruction_decisions(self) -> None:
        """Project one ordered decision for the active Console session."""
        view = self.view
        generation = self._attached_generation
        if not self.has_answerable_view() or generation is None:
            return
        provider = getattr(view, "console_view_hooks", None)
        hooks = provider() if callable(provider) else {}
        with self._project_decision_lock:
            active_session_id = getattr(self._chat_store, "active_session_id", None)
            decisions = (
                *self._project_binding_decisions.values(),
                *self._project_dispatch_decisions.values(),
            )
            candidates = tuple(
                decision
                for decision in decisions
                if decision.session_id == active_session_id
                and not decision.completed.is_set()
            )
            head = min(candidates, key=lambda decision: decision.order, default=None)
            stale = tuple(
                decision
                for decision in decisions
                if decision is not head
                and decision.projected_generation == generation
            )
            for decision in stale:
                self._pause_project_dispatch_locked(decision)
                decision.projected_generation = None

        dismiss = hooks.get("dismiss_project_instruction_decision")
        if callable(dismiss):
            for decision in stale:
                try:
                    dismiss(decision.decision_id)
                except Exception as exc:  # noqa: BLE001 -- projection is retryable
                    logger.debug(
                        "Project decision dismissal raised (exception_type={})",
                        type(exc).__name__,
                    )
        if head is None or head.projected_generation == generation:
            return
        if isinstance(head, _ProjectBindingDecision):
            callback = hooks.get("project_project_instruction_binding")
            args = (head.decision_id, head.choices)
        else:
            callback = hooks.get("project_project_instruction_dispatch")
            args = (head.decision_id, head.notice)
        if not callable(callback):
            return
        try:
            mounted = callback(*args) is True
        except Exception as exc:  # noqa: BLE001 -- projection is retryable
            mounted = False
            logger.debug(
                "Project decision projection raised (exception_type={})",
                type(exc).__name__,
            )
        if not mounted:
            self._schedule_project_instruction_projection_retry(head, generation)
            return
        with self._project_decision_lock:
            if (
                self._project_decision_for_id_locked(head.decision_id) is not head
                or head.completed.is_set()
                or self.view is not view
                or self._attached_generation != generation
                or getattr(self._chat_store, "active_session_id", None)
                != head.session_id
            ):
                return
            head.projected_generation = generation
            if isinstance(head, _ProjectDispatchDecision):
                head.answerable_since = time.monotonic()

    # -- app-owned staged evidence --------------------------------------

    def snapshot_console_staged_evidence(self) -> tuple[Any | None, int, int | None]:
        """Return the exact staged launch, revision, and projection notice."""
        state = self._staged_evidence
        return state.launch, state.revision, state.sent_source_count

    def _has_staged_evidence(self, _session_id: str | None = None) -> bool:
        """Return whether runtime-owned evidence is staged for the next turn."""
        return self._staged_evidence.launch is not None

    def stage_console_staged_evidence(self, launch: Any | None) -> int:
        """Replace the staged launch and return its new ownership revision."""
        state = self._staged_evidence
        state.revision += 1
        state.launch = launch
        state.sent_source_count = None
        return state.revision

    def restore_console_staged_evidence(
        self,
        launch: Any | None,
        *,
        revision: int,
        sent_source_count: int | None,
    ) -> bool:
        """Restore a navigation projection only when it is not stale."""
        state = self._staged_evidence
        if revision < state.revision:
            return False
        if revision == state.revision:
            return state.launch is launch
        state.launch = launch
        state.revision = revision
        state.sent_source_count = sent_source_count
        return True

    def set_console_staged_evidence_notice(self, count: int | None) -> None:
        """Set the source-count receipt without changing launch ownership."""
        self._staged_evidence.sent_source_count = count

    def _staged_evidence_lease_revision(self, launch: Any | None) -> int | None:
        """Return the opaque revision that owns this exact admitted launch."""
        state = self._staged_evidence
        return state.revision if launch is not None and state.launch is launch else None

    def release_console_staged_evidence(
        self,
        launch: Any | None,
        result: Any,
        *,
        revision: int | None,
    ) -> bool:
        """Release a captured launch only after context reached acceptance."""
        context = getattr(result, "context", None)
        state = self._staged_evidence
        if (
            revision is None
            or state.revision != revision
            or state.launch is not launch
            or not isinstance(context, str)
            or not context.strip()
        ):
            return False
        ordinals = getattr(
            getattr(result, "citation_repair_contract", None),
            "allowed_ordinals",
            None,
        )
        state.launch = None
        state.revision += 1
        state.sent_source_count = (
            len(ordinals)
            if isinstance(ordinals, tuple) and ordinals
            else console_prompted_source_count(launch)
        )
        return True

    # -- accepted-turn custody -------------------------------------------

    def _raise_if_disposed_or_session_fenced(self, session_id: str) -> None:
        """Reject admissions outside this runtime's live session boundary."""
        if self._disposed:
            raise RuntimeError("Console runtime is disposed.")
        if session_id in self._admission_fenced_sessions:
            raise RuntimeError("Console session is closed.")

    def _register_custody(
        self,
        request: ConsoleTurnCustodyRequest | None,
        attachments: tuple[Any, ...] = (),
        staged_evidence_revision: int | None = None,
        *,
        store: ConsoleChatStore | None = None,
        received_claim: ConsoleReceivedTurnClaim | None = None,
        received_intent: Any = None,
    ) -> _ConsoleTurnCustodyRecord:
        """Retain one request before its task may begin running."""
        identity = request if request is not None else received_intent
        if identity.turn_id in self._turn_custody:
            raise RuntimeError("Console turn is already in runtime custody.")
        record = _ConsoleTurnCustodyRecord(
            turn_id=identity.turn_id,
            session_id=identity.session_id,
            request=request,
            store=store,
            received_claim=received_claim,
            received_intent=received_intent,
            inputs=_ConsoleTurnCustodyInputs(
                attachments=attachments,
                staged_evidence_revision=staged_evidence_revision,
            ),
        )
        self._turn_custody[record.turn_id] = record
        if self._app is not None and self._chat_store is not None:
            session = next(
                (
                    item
                    for item in self._chat_store.sessions()
                    if item.id == identity.session_id
                ),
                None,
            )
            record.archive_conversation_id = (
                session.persisted_conversation_id if session is not None else None
            )
            if record.archive_conversation_id:
                reservations = getattr(self._app, "_conversation_send_inflight", None)
                if reservations is None:
                    reservations = self._app._conversation_send_inflight = {}
                reservations[record.archive_conversation_id] = (
                    reservations.get(record.archive_conversation_id, 0) + 1
                )
        return record

    def has_custodied_turns(self, session_id: str | None = None) -> bool:
        """Whether accepted work is in custody, even before its run starts.

        Args:
            session_id: Limit the check to one chat, or check all chats when unset.

        Returns:
            True until the selected accepted turn tasks have finished.
        """
        if session_id is None:
            return bool(self._turn_custody)
        return any(
            record.session_id == session_id for record in self._turn_custody.values()
        )

    def _release_custody(self, turn_id: str) -> None:
        """Drop the runtime's final references to an accepted turn."""
        record = self._turn_custody.pop(turn_id, None)
        if record is not None:
            if record.store is not None and record.received_claim is not None:
                if record.store.release_received_turn(record.received_claim):
                    self._note_received_admission_changed(
                        record.store, record.session_id
                    )
            record.store = None
            record.received_claim = None
            record.received_intent = None
            if record.archive_conversation_id:
                reservations = self._app._conversation_send_inflight
                remaining = reservations.get(record.archive_conversation_id, 1) - 1
                if remaining:
                    reservations[record.archive_conversation_id] = remaining
                else:
                    reservations.pop(record.archive_conversation_id, None)
                record.archive_conversation_id = None
            record.request = None
            record.inputs.attachments = ()
            record.inputs.staged_evidence_revision = None
            record.task = None

    def _note_received_admission_changed(self, store, session_id: str) -> None:
        """Publish activity revision without copying the store's admission state."""
        controller = self._chat_controller
        changed = getattr(controller, "_note_controller_activity_changed", None)
        if getattr(controller, "store", None) is store and callable(changed):
            changed(session_id)

    def _create_custody_task(self, coroutine) -> asyncio.Task[Any]:
        """Construct one lazy owned driver without consulting a loop task factory."""
        return asyncio.Task(
            coroutine,
            loop=asyncio.get_running_loop(),
            name="console-turn-custody",
            eager_start=False,
        )

    def _require_received_custody_current(self, record, controller) -> None:
        """Refuse source or admission displacement before initial submit effects."""
        from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

        self._raise_if_disposed_or_session_fenced(record.session_id)
        store, claim = record.store, record.received_claim
        if (
            store is None
            or claim is None
            or self._chat_store is not store
            or self._chat_controller is not controller
            or not store.received_turn_is_current(claim)
            or (
                isinstance(controller, ConsoleChatController)
                and controller.store is not store
            )
        ):
            raise RuntimeError("Received turn owner changed.")

    def accept_turn(
        self,
        request: ConsoleTurnCustodyRequest,
        *,
        origin: ConsoleSubmissionOrigin = ConsoleSubmissionOrigin.MANUAL,
        queue_entry_id: str | None = None,
        queue_authorization: Any | None = None,
        wake_authorization: Any | None = None,
        terminal_callback: Callable[[bool], None] | None = None,
        recover_before_acceptance: bool = True,
    ) -> str:
        """Synchronously place an accepted turn under runtime task custody."""
        self._raise_if_disposed_or_session_fenced(request.session_id)
        if request.configuration.session_id != request.session_id:
            raise ValueError("Turn configuration belongs to another session.")
        store = self._chat_store
        if store is None:
            raise RuntimeError("Console chat store is unavailable.")
        claim = store.claim_received_turn(
            request.session_id,
            request.turn_id,
            origin=origin,
        )
        if claim is None:
            raise RuntimeError(
                "Console session already has a received or prepared turn."
            )
        attachments = ()
        record = None
        coroutine = None
        try:
            self._note_received_admission_changed(store, request.session_id)
            attachments = store.transfer_pending_attachments_to_turn(
                request.session_id,
                request.turn_id,
                request.attachment_ids,
            )
            record = self._register_custody(
                request,
                attachments,
                self._staged_evidence_lease_revision(request.staged_evidence_launch),
                store=store,
                received_claim=claim,
            )
            coroutine = self._run_custodied_turn(
                record,
                origin=origin,
                queue_entry_id=queue_entry_id,
                queue_authorization=queue_authorization,
                wake_authorization=wake_authorization,
                raise_on_refusal=recover_before_acceptance,
            )
            record.task = self._create_custody_task(coroutine)
        except BaseException:
            if coroutine is not None:
                coroutine.close()
            if attachments:
                store.restore_transferred_pending_attachments(
                    request.session_id, attachments
                )
            if record is not None:
                self._release_custody(record.turn_id)
            elif store.release_received_turn(claim):
                self._note_received_admission_changed(store, request.session_id)
            raise
        record.task.add_done_callback(
            functools.partial(
                self._finish_custodied_turn,
                turn_id=record.turn_id,
                recover_before_acceptance=recover_before_acceptance,
                terminal_callback=terminal_callback,
            )
        )
        return record.turn_id

    def accept_received_intent(
        self, intent: ConsoleReceivedTurnIntent, *, _configuration_preparation=None,
        terminal_callback: Callable[[bool], None] | None = None,
    ) -> str:
        """Reserve bounded authored input before any hook or configuration read."""
        from .console_received_intent import ConsoleReceivedTurnIntent
        from .console_received_dispatch import (
            received_preparation_source,
            run_received_intent,
        )

        if type(intent) is not ConsoleReceivedTurnIntent:
            raise TypeError("intent must be ConsoleReceivedTurnIntent")
        self._raise_if_disposed_or_session_fenced(intent.session_id)
        store = self._chat_store
        if store is None or self._chat_controller is None:
            raise RuntimeError("Console chat owner is unavailable.")
        if not store.session_inputs_are_current(
            intent.inputs, include_draft=intent._pressed_inputs is None
        ):
            raise RuntimeError("Console input changed; Send again.")
        if _configuration_preparation is not None:
            from .console_configuration_preparation import (
                require_received_configuration_preparation,
            )

            require_received_configuration_preparation(
                _configuration_preparation,
                self._app,
                store,
                self._chat_controller,
                session_id=intent.session_id,
            )
        source = received_preparation_source(
            self, configuration_preparation=_configuration_preparation
        )
        claim = None
        if intent.queue_revision is None:
            claim = store.claim_received_turn(
                intent.session_id,
                intent.turn_id,
                draft_revision=intent.inputs.draft_revision,
                _allow_draft_change=intent._pressed_inputs is not None,
            )
            if claim is None:
                raise RuntimeError(
                    "Console session already has a received or prepared turn."
                )
        record = coroutine = None
        try:
            record = self._register_custody(
                None,
                store=store,
                received_claim=claim,
                received_intent=intent,
            )
            coroutine = run_received_intent(self, record, source)
            record.task = self._create_custody_task(coroutine)
            record.task.add_done_callback(
                functools.partial(
                    self._finish_custodied_turn,
                    turn_id=record.turn_id,
                    recover_before_acceptance=True,
                    terminal_callback=terminal_callback,
                )
            )
        except BaseException:
            if coroutine is not None:
                coroutine.close()
            if record is not None:
                self._release_custody(record.turn_id)
            elif claim is not None:
                store.release_received_turn(claim)
            raise
        try:
            self._note_received_admission_changed(store, intent.session_id)
            view = self.view
            project = getattr(view, "_project_console_received_preparing", None)
            if callable(project):
                project(intent.session_id)
        except Exception as error:
            logger.debug(
                "Received turn projection failed (exception_type={})",
                type(error).__name__,
            )
        return record.turn_id

    def has_received_intents(
        self, session_id: str | None, *, unpromoted_only: bool = False
    ) -> bool:
        """Project existing intake lifetime without another admission index."""
        return any(
            record.session_id == session_id
            and record.received_intent is not None
            and record.task is not None
            and not record.task.done()
            and (not unpromoted_only or record.request is None)
            for record in self._turn_custody.values()
        )

    def cancel_received_intents(self, session_id: str | None = None) -> bool:
        """Cancel existing intake custody, including a queue/review with no claim."""
        cancelled = False
        for record in tuple(self._turn_custody.values()):
            if record.received_intent is None or (
                session_id is not None and record.session_id != session_id
            ):
                continue
            if record.received_claim is not None and record.store is not None:
                record.store.seal_received_turn(record.received_claim)
            if record.task is not None and not record.task.done():
                record.task.cancel()
                cancelled = True
        controller = self._chat_controller
        host = getattr(controller, "_interrupt_host", None)
        cancel_reviews = getattr(host, "cancel_hook_reviews", None)
        if callable(cancel_reviews):
            cancel_reviews(session_id)
        return cancelled

    def _project_received_input(self, record) -> None:
        """Project a domain CAS into only the original attached composer."""
        intent = record.received_intent
        view = self.view
        if (
            intent is None
            or self._attached_generation != intent.view_attachment_generation
        ):
            return
        project = getattr(view, "_project_console_received_input", None)
        if callable(project):
            if intent._pressed_inputs is not None:
                project(
                    record.session_id,
                    _captured_stash=intent._pressed_stash,
                    _captured_inputs=intent._pressed_inputs,
                )
            else:
                project(record.session_id)

    async def _submit_queued_turn(
        self,
        prompt: Any,
        *,
        session_id: str,
        entry_id: str,
        authorization: Any,
    ) -> Any:
        """Admit one claimed queue entry, then await its runtime-owned task."""
        request = prompt.custody_request
        if request is None or request.session_id != session_id:
            raise RuntimeError("Queued prompt has no matching custody request.")
        turn_id = self.accept_turn(
            request,
            origin=ConsoleSubmissionOrigin.QUEUED,
            queue_entry_id=entry_id,
            queue_authorization=authorization,
            recover_before_acceptance=False,
        )
        return await self.wait_for_turn(turn_id)

    def _submit_fleet_wake(
        self,
        notice: str,
        *,
        session_id: str,
        wake_authorization: Any,
        on_terminal: Callable[[bool], None],
    ) -> str:
        """Freeze and synchronously admit one coordinator-authorized wake."""
        controller = self._chat_controller
        if controller is None:
            raise RuntimeError("Console controller is unavailable for wake custody.")
        configuration = controller.resolve_runtime_turn_configuration_snapshot(
            session_id
        )
        request = ConsoleTurnCustodyRequest(
            turn_id=str(uuid4()),
            session_id=session_id,
            draft=notice,
            configuration=configuration,
        )
        return self.accept_turn(
            request,
            origin=ConsoleSubmissionOrigin.AGENT_WAKE,
            wake_authorization=wake_authorization,
            terminal_callback=on_terminal,
            recover_before_acceptance=False,
        )

    async def _run_custodied_turn(
        self,
        record: _ConsoleTurnCustodyRecord,
        *,
        origin: ConsoleSubmissionOrigin,
        queue_entry_id: str | None,
        queue_authorization: Any | None,
        wake_authorization: Any | None,
        raise_on_refusal: bool,
        hook_read: Any = None,
    ) -> Any:
        """Run one screen-free turn using only its frozen custody record.

        ``hook_read`` is the received intent's one full hook authority read
        (``ConsoleHookAttemptRead``); it goes, as an argument, only to this
        record's own initial submission (ADR-225 decision 3).
        """
        request = record.request
        controller = self._chat_controller
        if request is None or controller is None:
            raise RuntimeError("Console controller is unavailable for runtime custody.")
        self._require_received_custody_current(record, controller)

        if record.archive_conversation_id:
            from tldw_chatbook.Chat.conversation_archive_actions import (
                conversation_send_refusal,
            )

            refusal = await conversation_send_refusal(
                self._app, record.archive_conversation_id
            )
            if refusal:
                notify = getattr(self._app, "notify", None)
                if callable(notify):
                    notify(refusal, severity="warning")
                # TASK-33621.2: the parked turn's shelf entry states this
                # refusal too, like a controller refusal's below.
                raise _ConsoleTurnRefusedError(
                    "Console conversation is unavailable for submission.",
                    reason=refusal,
                )

        def mark_durable_acceptance() -> None:
            record.inputs.durable_accepted = True
            intent = record.received_intent
            if intent is not None:
                try:
                    store, claim = record.store, record.received_claim
                    # Saved acceptance is a fact even when its original owner
                    # changed or Stop won. Those outcomes retain the composer.
                    if (
                        self._disposed
                        or record.session_id in self._admission_fenced_sessions
                        or self._chat_store is not store
                        or self._chat_controller is not controller
                        or controller.store is not store
                        or claim is None
                        or not store.received_turn_matches_session(claim)
                    ):
                        return
                    preparation_id = controller._active_submit_preparations.get(
                        asyncio.current_task()
                    )
                    native_owner = controller._ordinary_native_commit_owner(preparation_id)
                    if native_owner is not None and (
                        native_owner.caller_cancelled
                        or native_owner.explicit_stop
                        or native_owner.commit_error is not None
                        or not controller._ordinary_native_commit_current(native_owner)
                    ):
                        return
                    committed = store.commit_session_input_draft(intent.inputs)
                    if committed or intent._pressed_inputs is not None:
                        self._project_received_input(record)
                except Exception as error:
                    logger.debug(
                        "Received input projection failed (exception_type={})",
                        type(error).__name__,
                    )

        async def submit() -> Any:
            from tldw_chatbook.Chat.console_chat_controller import (
                ConsoleChatController,
            )
            from .console_received_turn import bind_received_turn_claim

            self._require_received_custody_current(record, controller)
            store, claim = record.store, record.received_claim
            shared = {}
            if (
                hook_read is not None
                and getattr(controller.submit_draft, "__func__", None)
                is ConsoleChatController.submit_draft
            ):
                # Only the stock submission knows this private argument; a
                # replaced one keeps its original call and its own reads.
                shared["_hook_read"] = hook_read
            with bind_received_turn_claim(store, claim):
                try:
                    controller.prompt_queue_coordinator.bind_turn_request(
                        request, origin=origin
                    )
                    return await controller.submit_draft(
                        request.draft,
                        session_id=request.session_id,
                        origin=origin,
                        queue_entry_id=queue_entry_id,
                        queue_authorization=queue_authorization,
                        wake_authorization=wake_authorization,
                        configuration=replace(
                            request.configuration,
                            skill_context_maximum={
                                **request.configuration.skill_context_maximum,
                                "plugin_turn_id": request.turn_id,
                            },
                        ),
                        accepted_attachments=record.inputs.attachments,
                        captured_one_shot_prefill=request.one_shot_prefill,
                        captured_one_shot_prefill_revision=(
                            request.one_shot_prefill_revision
                        ),
                        staged_evidence_launch=request.staged_evidence_launch,
                        staged_evidence_capture=self._capture_frozen_console_staged_rag,
                        staged_evidence_release=functools.partial(
                            self.release_console_staged_evidence,
                            revision=record.inputs.staged_evidence_revision,
                        ),
                        custody_acceptance_hook=mark_durable_acceptance,
                        **shared,
                    )

                finally:
                    # Some Capture-Off/machine inputs never construct a full
                    # preparation. Retire initial admission before chain drain.
                    if store.release_received_turn(claim):
                        self._note_received_admission_changed(store, request.session_id)

        result = (
            await submit()
            if origin is ConsoleSubmissionOrigin.AGENT_WAKE
            else await controller.run_prompt_chain(
                session_id=request.session_id,
                initial_turn=submit,
            )
        )
        held_preparation_id = getattr(result, "preparation_id", None)
        if (
            not bool(getattr(result, "accepted", False))
            and not record.inputs.durable_accepted
            and raise_on_refusal
            # TASK-34350: a send held at the compaction threshold is waiting
            # on the user's answer in its own card, not refused; recording it
            # as a turn recovery too showed one send on two surfaces.
            and not (
                held_preparation_id is not None
                and controller.context_compaction_hold(held_preparation_id)
                is not None
            )
        ):
            raise _ConsoleTurnRefusedError(
                "Console turn was refused before durable acceptance.",
                reason=str(getattr(result, "visible_copy", "") or ""),
            )
        run_state_for = getattr(controller, "run_state_for", None)
        run_state = (
            run_state_for(request.session_id) if callable(run_state_for) else None
        )
        if (
            origin is ConsoleSubmissionOrigin.MANUAL
            and bool(getattr(result, "accepted", False))
            and getattr(run_state, "status", None) is ConsoleRunStatus.COMPLETED
        ):
            await self._record_successful_first_send()
        return result

    async def _record_successful_first_send(self) -> None:
        """Record terminal first-send success without retaining a view."""
        app_config = getattr(self._app, "app_config", None)
        if not isinstance(app_config, dict):
            return
        console = app_config.get("console")
        if not isinstance(console, dict):
            console = {}
            app_config["console"] = console
        onboarding = console.get("onboarding")
        if not isinstance(onboarding, dict):
            onboarding = {}
            console["onboarding"] = onboarding
        if coerce_console_first_send_completed(
            onboarding.get("first_send_completed")
        ):
            return
        onboarding["first_send_completed"] = True
        try:
            from tldw_chatbook.config import save_setting_to_cli_config

            await asyncio.to_thread(
                save_setting_to_cli_config,
                "console.onboarding",
                "first_send_completed",
                True,
            )
        except Exception as exc:  # noqa: BLE001 -- the completed turn stays valid
            logger.warning(
                "Failed to persist Console first-send completion "
                "(exception_type={})",
                type(exc).__name__,
            )

    async def _capture_console_staged_rag(
        self, draft: str, turn_context: Any = None
    ) -> Any:
        """Capture the evidence staged at dispatch: the controller's live seam.

        TASK-33940.4 (Qodo #1 on PR #2975): the live seam is called as
        ``provider(draft, turn_context)`` for turns that did not freeze an
        evidence decision at admission -- queued prompts and non-custodied
        callers. This runtime used to wire its three-argument frozen capture
        there, so every such call raised TypeError. Mirrors the retrieval
        owner's live capture: consume the launch staged now, and release it
        once it produced context.
        """
        launch, revision, _notice = self.snapshot_console_staged_evidence()
        result = await self._capture_frozen_console_staged_rag(
            draft, turn_context, launch
        )
        if launch is not None:
            self.release_console_staged_evidence(launch, result, revision=revision)
        return result

    async def _capture_frozen_console_staged_rag(
        self, draft: str, turn_context: Any, launch: Any
    ) -> Any:
        """Capture one admitted evidence launch without consulting a view."""
        del turn_context
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
            capture_console_staged_evidence_for_chat,
        )

        return await capture_console_staged_evidence_for_chat(
            self._app,
            launch,
            user_message=draft,
        )

    async def wait_for_turn(self, turn_id: str) -> Any:
        """Await one specific live custody task (narrow integration-test seam)."""
        record = self._turn_custody.get(turn_id)
        task = record.task if record is not None else None
        if task is None:
            raise KeyError(turn_id)
        return await task

    def recoveries_for_session(
        self, session_id: str
    ) -> tuple[ConsoleTurnRecoveryEntry, ...]:
        """Return exact recoveries in admission-failure order."""
        return tuple(
            self._turn_recoveries[turn_id]
            for turn_id in self._recovery_turns_by_session.get(session_id, ())
            if turn_id in self._turn_recoveries
            and (
                self._turn_recoveries[turn_id].source_claim is None
                or (
                    self._chat_store is not None
                    and self._chat_store.received_turn_matches_session(
                        self._turn_recoveries[turn_id].source_claim
                    )
                )
            )
        )

    def restore_turn_recovery(self, turn_id: str) -> ConsoleTurnRecoveryEntry:
        """Re-stage one exact recovery into its still-live owning session."""
        entry = self._turn_recoveries[turn_id]
        store = self._chat_store
        if store is None or entry.session_id not in {
            session.id for session in store.sessions()
        }:
            raise RuntimeError("Recovery session is no longer available.")
        if entry.source_claim is not None and not store.received_turn_matches_session(
            entry.source_claim
        ):
            raise RuntimeError("Recovery session owner changed.")
        if store.session_draft(entry.session_id):
            raise RuntimeError("Recovery live draft changed; refusing ambiguous merge.")
        store.restore_transferred_pending_attachments(
            entry.session_id, entry.attachments
        )
        store.set_session_draft(entry.session_id, entry.draft)
        self.discard_turn_recovery(turn_id)
        return entry

    def discard_turn_recovery(self, turn_id: str) -> bool:
        """Release one recovery's sensitive references."""
        entry = self._turn_recoveries.pop(turn_id, None)
        if entry is None:
            return False
        turns = self._recovery_turns_by_session.get(entry.session_id, [])
        if turn_id in turns:
            turns.remove(turn_id)
        if not turns:
            self._recovery_turns_by_session.pop(entry.session_id, None)
        return True

    def _record_turn_recovery(
        self, record: _ConsoleTurnCustodyRecord, *, reason: str = ""
    ) -> None:
        request = record.request
        if (
            request is None
            or request.turn_id in self._turn_recoveries
            or self._disposed
            or request.session_id in self._admission_fenced_sessions
        ):
            return
        self._recovery_order += 1
        entry = ConsoleTurnRecoveryEntry(
            turn_id=request.turn_id,
            session_id=request.session_id,
            draft=request.draft,
            attachments=record.inputs.attachments,
            insertion_order=self._recovery_order,
            reason=reason,
            source_claim=record.received_claim,
        )
        self._turn_recoveries[entry.turn_id] = entry
        self._recovery_turns_by_session.setdefault(entry.session_id, []).append(
            entry.turn_id
        )

    def _finish_custodied_turn(
        self,
        task: asyncio.Task[Any],
        *,
        turn_id: str,
        recover_before_acceptance: bool,
        terminal_callback: Callable[[bool], None] | None,
    ) -> None:
        """Consume a task result and release its retained sensitive inputs."""
        record = self._turn_custody.get(turn_id)
        if record is not None and record.task is not task:
            record = None
        accepted = False
        received_reason = ""
        try:
            result = task.result()
            accepted = bool(getattr(result, "accepted", False)) or bool(
                record is not None
                and record.received_intent is not None
                and record.received_intent.queue_revision is not None
                and getattr(result, "applied", False)
            )
        except asyncio.CancelledError:
            if (
                record is not None
                and recover_before_acceptance
                and not record.inputs.durable_accepted
                and (record.store or self._chat_store) is not None
                and any(
                    item.id == record.session_id
                    for item in (record.store or self._chat_store).sessions()
                )
            ):
                self._record_turn_recovery(record)
        except BaseException as exc:
            if (
                record is not None
                and record.received_intent is not None
                and not record.inputs.durable_accepted
            ):
                from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

                received_reason = (
                    "Draft, chat or settings changed; Send again."
                    if isinstance(exc, RecoveryRequired)
                    else "Send could not be prepared; draft kept."
                )
            if (
                record is not None
                and recover_before_acceptance
                and not record.inputs.durable_accepted
            ):
                self._record_turn_recovery(
                    record,
                    reason=(
                        exc.reason
                        if isinstance(exc, _ConsoleTurnRefusedError)
                        else ""
                    ),
                )
            logger.warning(
                "Console runtime turn ended with exception_type={}",
                type(exc).__name__,
            )
        finally:
            accepted = accepted or bool(
                record is not None and record.inputs.durable_accepted
            )
            if record is not None:
                received = record.received_intent is not None
                source_store = record.store
                session_id = record.session_id
                source_current = False
                if received and source_store is self._chat_store:
                    claim = record.received_claim
                    if claim is not None:
                        source_current = source_store.received_turn_matches_session(
                            claim
                        )
                    else:
                        try:
                            actual = source_store.session_input_snapshot(session_id)
                            expected = record.received_intent.inputs
                            source_current = (
                                actual.incarnation_id == expected.incarnation_id
                                and actual.conversation_binding_revision
                                == expected.conversation_binding_revision
                                and actual.ephemeral == expected.ephemeral
                            )
                        except KeyError:
                            pass
                self._release_custody(record.turn_id)
                if received and source_current:
                    project = getattr(
                        self.view, "_project_console_received_finished", None
                    )
                    if callable(project):
                        try:
                            project(session_id, received_reason)
                        except Exception:
                            pass
            if terminal_callback is not None:
                try:
                    terminal_callback(accepted)
                except Exception as exc:  # noqa: BLE001 -- terminal cleanup is final
                    logger.warning(
                        "Console runtime terminal callback failed "
                        "(exception_type={})",
                        type(exc).__name__,
                    )
            # Terminal rows and their exact unseen marks are committed by the
            # store before the controller returns. Recompute once per turn,
            # never from streaming/token callbacks.
            self.recompute_console_attention()

    def _ensure_canvas_profile_snapshot(self) -> Any:
        """Share one lazy process owner with an early served-child handshake."""
        snapshot = self._canvas_profile_snapshot
        if snapshot is None:
            from tldw_chatbook.Canvas.profiles import (
                load_application_profile_snapshot,
            )

            snapshot = load_application_profile_snapshot()
            self._canvas_profile_snapshot = snapshot
        return snapshot

    def ensure_canvas_gateway(self, *, authority: Any) -> Any:
        """Return this app runtime's native Canvas gateway, creating it lazily."""

        if getattr(self._app, "_served_canvas_mode", False):
            return None
        if not self._canvas_enabled():
            return None
        if self._canvas_gateway is not None:
            if self._canvas_gateway_authority is not authority:
                raise ValueError("Canvas gateway is bound to a different authority")
            binder = getattr(authority, "bind_gateway_invalidator", None)
            if callable(binder):
                binder(self._canvas_gateway.mark_browser_session_unavailable)
            return self._canvas_gateway
        if self._disposed:
            return None
        from tldw_chatbook.Canvas.gateway import CanvasGateway

        self._canvas_gateway = CanvasGateway(
            authority=authority, profile_snapshot=self._ensure_canvas_profile_snapshot()
        )
        self._canvas_gateway_authority = authority
        binder = getattr(authority, "bind_gateway_invalidator", None)
        if callable(binder):
            binder(self._canvas_gateway.mark_browser_session_unavailable)
        self._start_canvas_policy_watcher()
        return self._canvas_gateway

    def bind_canvas_native_view(
        self,
        *,
        scope_resolver: Callable[[str], Any],
        bridge_sink: Callable[[Any, str], None] | None = None,
        bridge_prepare: Callable[[Any], Callable[[str], None]] | None = None,
        auto_open: Callable[[str, Any], None] | None = None,
        publication_guard: Callable[[Any], bool] | None = None,
    ) -> Any:
        """Bind the latest view without importing the native authority."""

        def resolve_live_scope(session_id: str) -> Any:
            self._raise_if_disposed_or_session_fenced(session_id)
            return scope_resolver(session_id)

        if not self._canvas_enabled():
            return None
        with self._canvas_native_lock:
            if self._disposed or self._canvas_disabled_latched or self._canvas_maintenance_closed:
                return None
            controller = self._canvas_controller
            binding = _CanvasNativeViewBinding(
                scope_resolver=resolve_live_scope,
                bridge_sink=bridge_sink,
                bridge_prepare=bridge_prepare,
                auto_open=auto_open,
                publication_guard=publication_guard,
                source_scope_resolver=scope_resolver,
                controller=controller,
                view=self.view,
                attachment_generation=self._attached_generation,
            )
            self._canvas_native_view_binding = binding
            if controller is not None:
                controller.add_settlement_listener(self._canvas_settlement_listener)
            authority = self._canvas_native_authority
            if authority is not None:
                authority.rebind_view(
                    scope_resolver=binding.scope_resolver,
                    bridge_sink=binding.bridge_sink,
                    bridge_prepare=binding.bridge_prepare,
                    auto_open=binding.auto_open,
                    publication_guard=binding.publication_guard,
                )
            return authority

    def canvas_native_view_is_bound(
        self,
        view: Any,
        *,
        scope_resolver: Callable[[str], Any],
        bridge_sink: Callable[[Any, str], None] | None = None,
        bridge_prepare: Callable[[Any], Callable[[str], None]] | None = None,
        auto_open: Callable[[str, Any], None] | None = None,
        publication_guard: Callable[[Any], bool] | None = None,
    ) -> bool:
        """Check installed callbacks for UI idempotence, never execution permission."""
        with self._canvas_native_lock:
            binding = self._canvas_native_view_binding
            if (
                self._disposed
                or self._canvas_disabled_latched
                or self._canvas_maintenance_closed
                or view is None
                or self.view is not view
                or self._attached_generation is None
                or binding is None
                or binding.view is not view
                or binding.attachment_generation != self._attached_generation
                or self._canvas_controller is None
                or binding.controller is not self._canvas_controller
            ):
                return False
            return all(
                current is proposed
                or (
                    type(current) is MethodType
                    and type(proposed) is MethodType
                    and current.__self__ is proposed.__self__
                    and current.__func__ is proposed.__func__
                )
                for current, proposed in (
                    (binding.source_scope_resolver, scope_resolver),
                    (binding.bridge_sink, bridge_sink),
                    (binding.bridge_prepare, bridge_prepare),
                    (binding.auto_open, auto_open),
                    (binding.publication_guard, publication_guard),
                )
            )

    def _canvas_scope_for_run(self, session_id: str) -> Any:
        """Resolve a live run's exact owner independently of the selected view.

        Browser operations retain the view's active-session resolver. Queued
        turns may start while another session or screen is selected, but must
        still match their captured conversation and transcript branch.
        """
        from tldw_chatbook.Canvas.models import CanvasScope

        store = self._chat_store
        if self._disposed or store is None:
            raise RuntimeError("Canvas session is unavailable")
        session = next(
            (item for item in store.sessions() if item.id == session_id), None
        )
        if session is None:
            raise RuntimeError("Canvas session is unavailable")
        active_ids = store.canvas_active_path_message_ids(session_id)
        if not active_ids:
            raise RuntimeError("Canvas requires an active transcript message")
        return CanvasScope(
            session_id=session_id,
            conversation_id=session.persisted_conversation_id or session_id,
            active_message_ids=active_ids,
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id=str(uuid4()),
        )

    def _materialize_canvas_native_authority(self) -> Any:
        """Construct the single authority for an actual publication/open."""

        if not self._canvas_enabled():
            return None
        with self._canvas_native_lock:
            if self._disposed or self._canvas_disabled_latched or self._canvas_maintenance_closed:
                return None
            binding = self._canvas_native_view_binding
            controller = self._canvas_controller
            if binding is None or controller is None:
                return None
            if self._canvas_native_authority is not None:
                return self._canvas_native_authority
            from tldw_chatbook.Canvas.native_authority import (
                NativeConsoleCanvasAuthority,
            )

            self._canvas_native_authority = NativeConsoleCanvasAuthority(
                scope_resolver=binding.scope_resolver,
                canvas_controller=controller,
                run_scope_resolver=self._canvas_scope_for_run,
                bridge_sink=binding.bridge_sink,
                bridge_prepare=binding.bridge_prepare,
                auto_open=binding.auto_open,
                publication_guard=binding.publication_guard,
                enabled_reader=self._canvas_enabled,
            )
            return self._canvas_native_authority

    def _forward_canvas_settlement(self, publication: Any) -> None:
        """Materialize and synchronously forward the first settled mutation."""

        authority = self._materialize_canvas_native_authority()
        if authority is not None:
            authority.on_settlement_publication(publication)

    def ensure_canvas_native_authority(
        self,
        *,
        scope_resolver: Callable[[str], Any],
        bridge_sink: Callable[[Any, str], None] | None = None,
        bridge_prepare: Callable[[Any], Callable[[str], None]] | None = None,
        auto_open: Callable[[str, Any], None] | None = None,
        publication_guard: Callable[[Any], bool] | None = None,
    ) -> Any:
        """Return the single Console-bound Canvas browser authority."""

        self.bind_canvas_native_view(
            scope_resolver=scope_resolver,
            bridge_sink=bridge_sink,
            bridge_prepare=bridge_prepare,
            auto_open=auto_open,
            publication_guard=publication_guard,
        )
        return self._materialize_canvas_native_authority()

    def _read_canvas_enabled(self) -> bool | None:
        """Read policy, distinguishing temporary backup refusal from disable."""
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        try:
            return self._canvas_enabled_reader() is True
        except RecoveryRequired as error:
            if type(error) is RecoveryRequired and error.args == ("storage_locally_paused",):
                return None
            return False
        except Exception:  # noqa: BLE001 - execution policy fails closed
            return False

    def _canvas_enabled(self) -> bool:
        """Read the global kill switch and the restart-required runtime latch."""

        generation = self._canvas_maintenance_generation
        if self._disposed or self._canvas_disabled_latched or self._canvas_maintenance_closed:
            return False
        enabled = self._read_canvas_enabled()
        # A read may span a whole pause/resume on another thread. Keep config
        # reads independent of the Canvas publication lock, and reject that
        # observation without converting temporary refusal to permanent disable.
        if (
            generation != self._canvas_maintenance_generation
            or self._disposed
            or self._canvas_disabled_latched
            or self._canvas_maintenance_closed
        ):
            return False
        if enabled is False:
            self._canvas_disabled_latched = True
        return enabled is True

    def canvas_enabled(self) -> bool:
        """Expose this app runtime's restart-latched Canvas availability."""

        return self._canvas_enabled()

    def canvas_disabled(self) -> bool:
        """Distinguish permanent policy disable from temporary backup unavailability."""
        return self._disposed or self._canvas_disabled_latched

    def canvas_authority_is_current(self, authority: Any) -> bool:
        """Return whether *authority* still owns enabled Canvas effects."""

        return self._canvas_enabled() and self._canvas_native_authority is authority

    def latch_canvas_disabled(self) -> None:
        """Synchronously accept a disable before asynchronous cleanup begins."""

        with self._canvas_native_lock:
            self._canvas_disabled_latched = True

    def ensure_hooks_v2(
        self,
        session_id: str,
        definitions: tuple,
        authority_check: Callable,
        **owner_options: Any,
    ) -> Any:
        """Bind an immutable hook session on the application's running loop.

        The admitted H4 session owner supplies definitions and captured current
        authority. Re-entry reuses that snapshot; replacement is explicit close
        and a new session identity. This does not publish lifecycle events.
        """
        from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
        from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

        self._raise_if_disposed_or_session_fenced(session_id)
        loop = asyncio.get_running_loop()
        with self._run_hooks_lock:
            if self._disposed:
                raise RuntimeError("Console runtime is disposed.")
            if self._hooks_v2_budget_owner is None:
                self._hooks_v2_budget_owner = HookBudgetOwner()
            elif self._hooks_v2_budget_owner.loop is not loop:
                raise RuntimeError("Console hooks belong to the application loop.")
            existing = self._hooks_v2_engines.get(session_id)
            if existing is not None:
                if existing.definitions != tuple(definitions):
                    raise RuntimeError("Hook session definitions are immutable.")
                return existing
            engine = HookEngine(
                tuple(definitions),
                authority_check,
                self._hooks_v2_budget_owner,
                **owner_options,
            )
            self._hooks_v2_engines[session_id] = engine
            return engine

    def _capture_hook_preparation_source(self, session_id, permissions):
        store, controller, app = self._chat_store, self._chat_controller, self._app
        session = (
            next((row for row in store.sessions() if row.id == session_id), None)
            if store is not None
            else None
        )
        identity = (
            getattr(session, "incarnation_id", None),
            getattr(session, "conversation_binding_revision", None),
            getattr(session, "ephemeral", None),
            getattr(session, "workspace_id", None),
        )
        persistence = getattr(store, "persistence", None)
        registry = getattr(app, "workspace_registry_service", None)
        return _HookPreparationSource(
            app,
            store,
            controller,
            session_id,
            session,
            identity,
            permissions,
            persistence,
            getattr(persistence, "db", None),
            registry,
            getattr(registry, "db", None),
            getattr(app, "change_review_consent_service", None),
            getattr(controller, "_hook_authority_values", None),
            getattr(controller, "app", None),
            getattr(controller, "_turn_context_provider", None),
        )

    def _require_hook_preparation_source(self, source):
        self._raise_if_disposed_or_session_fenced(source.session_id)
        session = (
            next(
                (row for row in source.store.sessions() if row.id == source.session_id),
                None,
            )
            if source.store is not None
            else None
        )
        reader = getattr(source.controller, "_hook_authority_values", None)
        same_reader = reader is source.authority_reader or (
            inspect.ismethod(reader)
            and inspect.ismethod(source.authority_reader)
            and reader.__self__ is source.authority_reader.__self__
            and reader.__func__ is source.authority_reader.__func__
        )
        if (
            self._app is not source.app
            or self._chat_store is not source.store
            or self._chat_controller is not source.controller
            or self._hook_permissions is not source.permissions
            or not same_reader
            or session is not source.session
            or (
                getattr(session, "incarnation_id", None),
                getattr(session, "conversation_binding_revision", None),
                getattr(session, "ephemeral", None),
                getattr(session, "workspace_id", None),
            )
            != source.session_identity
            or getattr(source.store, "persistence", None) is not source.persistence
            or getattr(source.persistence, "db", None) is not source.chat_database
            or getattr(source.app, "workspace_registry_service", None)
            is not source.registry
            or getattr(source.registry, "db", None) is not source.workspace_database
            or getattr(source.app, "change_review_consent_service", None)
            is not source.consent
            or (
                source.controller is not None
                and (
                    getattr(source.controller, "store", source.store)
                    is not source.store
                    or getattr(source.controller, "app", None)
                    is not source.controller_app
                    or getattr(source.controller, "_turn_context_provider", None)
                    is not source.context_provider
                    or getattr(source.controller, "_disposed", False)
                    or (
                        getattr(source.controller, "_shutdown_requested", None)
                        is not None
                        and source.controller._shutdown_requested.is_set()
                    )
                )
            )
        ):
            raise RuntimeError("Console hook preparation owner changed.")

    async def _read_hook_preparation(self, callback, source):
        from .console_hook_preparation import run_hook_preparation_read

        current_callback = callback
        observers = ()
        controller_reads = getattr(source.controller, "_preparation_reads", None)
        if controller_reads is not None:
            observers = (controller_reads,)
        return await run_hook_preparation_read(
            current_callback,
            creator=self,
            session_id=source.session_id,
            reads=self._preparation_reads,
            observers=observers,
            require_current=lambda: self._require_hook_preparation_source(source),
            source=source,
        )

    async def _drain_hook_preparation_reads(self, session_id=None):
        from .console_hook_preparation import (
            drain_hook_preparation_reads,
            hook_preparation_reads_for,
        )

        # Availability's outer owner has only its cache-lane finally after the
        # finite native body. App disposal must retain that exact finally too.
        owners = {
            read.task
            for read in hook_preparation_reads_for(self._preparation_reads, session_id)
            if session_id is None
            and getattr(read.creator, "_workspace_files_availability_task", None)
            is read.task
        }
        cancelled = await drain_hook_preparation_reads(self._preparation_reads, session_id)
        for owner in owners:
            while not owner.done():
                try:
                    await asyncio.shield(owner)
                except asyncio.CancelledError:
                    if not owner.done():
                        cancelled = True
                except Exception:
                    break
            self._consume_task_outcome(owner)
        return cancelled

    def _hooks_v2_context_key(self, session_id: str):
        """Capture host workspace/binding authority, without prompt bodies."""
        from .console_hook_preparation import hook_preparation_source_for

        source = hook_preparation_source_for(self)
        if isinstance(source, _HookPreparationSource):
            if source.session_id != session_id:
                raise RuntimeError("Console hook preparation session changed.")
            self._require_hook_preparation_source(source)
            store, controller, session = source.store, source.controller, source.session
        else:
            store, controller = self._chat_store, self._chat_controller
            session = (
                next((row for row in store.sessions() if row.id == session_id), None)
                if store is not None
                else None
            )
        if store is None or controller is None:
            return None
        if session is None:
            raise RuntimeError("Console hook preparation session changed.")
        from tldw_chatbook.DB.base_db import operation_owned_connection

        # Only the stock finite callback gains Workspace connection ownership.
        # Custom/injected callbacks retain the original call and cleanup shape.
        from tldw_chatbook.Chat import console_chat_controller as controller_module
        from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
        from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
        from tldw_chatbook.Workspaces import registry_service as registry_module
        from tldw_chatbook.Workspaces.change_review_consent import (
            ChangeReviewConsentService,
        )

        app = source.app if isinstance(source, _HookPreparationSource) else self._app
        registry = (
            source.registry
            if isinstance(source, _HookPreparationSource)
            else getattr(app, "workspace_registry_service", None)
        )
        database = (
            source.workspace_database
            if isinstance(source, _HookPreparationSource)
            else getattr(registry, "db", None)
        )
        consent = (
            source.consent
            if isinstance(source, _HookPreparationSource)
            else getattr(app, "change_review_consent_service", None)
        )
        hook_anchor = controller_module._HOOK_AUTHORITY_VALUES_ORIGINAL
        controller_class, name, function, code, defaults, kwdefaults, closure = (
            hook_anchor
        )
        reader_anchors = registry_module._HOOK_WORKSPACE_READERS
        registry_class = reader_anchors[0][0]

        def bindings_current():
            return (
                controller_module._HOOK_AUTHORITY_VALUES_ORIGINAL is hook_anchor
                and controller_module.ConsoleChatController is controller_class
                and registry_module.LocalWorkspaceRegistryService is registry_class
                and registry_module._HOOK_WORKSPACE_READERS is reader_anchors
                and inspect.getattr_static(type(controller), name) is function
                and name not in vars(controller)
                and function.__code__ is code
                and function.__defaults__ is defaults
                and function.__kwdefaults__ is kwdefaults
                and function.__closure__ is closure
                and function.__globals__ is controller_module.__dict__
                and all(
                    owner is registry_class
                    and inspect.getattr_static(type(registry), label) is reader
                    and label not in vars(registry)
                    and reader.__code__ is reader_code
                    and reader.__defaults__ is reader_defaults
                    and reader.__kwdefaults__ is reader_kwdefaults
                    and reader.__closure__ is reader_closure
                    and reader.__globals__ is registry_module.__dict__
                    for owner, label, reader, reader_code, reader_defaults, reader_kwdefaults, reader_closure in reader_anchors
                )
            )

        stock = (
            type(self) is _HOOK_CONTEXT_KEY_ORIGINAL_OWNER
            and ConsoleRuntime is _HOOK_CONTEXT_KEY_ORIGINAL_OWNER
            and type(store) is ConsoleChatStore
            and type(controller) is controller_class
            and controller.store is store
            and controller.app is app
            and controller._turn_context_provider is None
            and type(registry) is registry_class
            and type(database) is WorkspaceDB
            and not database.is_memory_db
            and type(consent) is ChangeReviewConsentService
            and consent._registry is registry
            and bindings_current()
        )
        if not stock:
            with operation_owned_connection(getattr(store.persistence, "db", None)):
                if isinstance(source, _HookPreparationSource):
                    self._require_hook_preparation_source(source)
                    values = source.authority_reader(session_id)
                    self._require_hook_preparation_source(source)
                else:
                    values = controller._hook_authority_values(session_id)
                return (
                    session.workspace_id,
                    values["workspace_roots"],
                    values["project_authority"],
                )

        chat_database = getattr(store.persistence, "db", None)
        workspace_id = session.workspace_id
        actor = (os.getpid(), threading.current_thread())
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None

        def require_current():
            try:
                current_task = asyncio.current_task()
            except RuntimeError:
                current_task = None
            if (
                self._disposed
                or ConsoleRuntime is not _HOOK_CONTEXT_KEY_ORIGINAL_OWNER
                or self._app is not app
                or self._chat_store is not store
                or self._chat_controller is not controller
                or controller.store is not store
                or controller.app is not app
                or controller._turn_context_provider is not None
                or not bindings_current()
                or getattr(store.persistence, "db", None) is not chat_database
                or getattr(app, "workspace_registry_service", None) is not registry
                or registry.db is not database
                or getattr(app, "change_review_consent_service", None) is not consent
                or consent._registry is not registry
                or not any(row is session for row in store.sessions())
                or session.workspace_id != workspace_id
                or os.getpid() != actor[0]
                or threading.current_thread() is not actor[1]
                or current_task is not task
            ):
                raise RuntimeError("Console hook workspace ownership changed.")

        require_current()
        with (
            operation_owned_connection(chat_database),
            operation_owned_connection(database),
            database.connection(),
        ):
            require_current()
            # Invoke the qualified original instead of a later mutable lookup.
            values = function(controller, session_id)
            require_current()
            result = (
                workspace_id,
                values["workspace_roots"],
                values["project_authority"],
            )
        # Retirement can run callbacks; refuse a redirected result afterwards too.
        require_current()
        return result

    def _hook_read_prepares_nothing(
        self, session_id: str, review: Any, configuration: Any
    ) -> bool:
        """Whether an attempt's shared read answers ``prepare_hooks_v2`` "nothing".

        ADR-225 decision 3 lets an attempt's earlier consent read stand in for
        the ``v2_configuration()`` re-read only when the answer is "no v2 hook
        to prepare": a ready review whose section configures no v2 handler
        (not even an invalid batch), no plugin-owned skill that could add
        native definitions, and a session holding no hook engine, lifecycle
        or configured signature. Each clause is one of the fresh path's own
        early-return conditions, checked at least as strictly, so the answer
        is the one a fresh read of the same state gives.

        Anything else -- building, keeping or replacing an engine -- reads
        fresh. A handler captured from an earlier read may have been disabled
        or removed by another process since (invisible in memory); an engine
        built or kept around it would refuse that handler at its fresh
        authority check, blocking the Send when the handler is required, where
        a fresh read would configure nothing.

        Args:
            session_id: The preparing session.
            review: The shared read's ``HookReviewSnapshot``.
            configuration: The turn configuration ``prepare_hooks_v2`` got.

        Returns:
            ``True`` only for the "nothing to prepare" answer. Never raises:
            an unexpected shape answers ``False`` so the fresh path reproduces
            its existing behaviour and errors at their existing point.
        """
        try:
            if not review.ready:
                return False
            section = review.config.section if review.config.section_present else {}
            handlers = (
                section.get("handler", []) if isinstance(section, Mapping) else []
            )
            # ``load_hooks_config`` maps exactly an empty list (or no key) to
            # no v2 handlers and no invalid admissions; other shapes go fresh.
            if not isinstance(handlers, list) or handlers:
                return False
            if configuration is not None and any(
                row.get("plugin_owned")
                for row in configuration.skill_context_maximum.get(
                    "available_skills", ()
                )
            ):
                # Plugin-owned skills may contribute native definitions.
                return False
            # No engine also means no engine-carried native plugins.
            return (
                self.get_hooks_v2(session_id) is None
                and session_id not in self._hooks_v2_engines
                and session_id not in self._hooks_v2_lifecycles
                and session_id not in self._hooks_v2_configured
            )
        except Exception:  # noqa: BLE001 -- unknown shape: the fresh path decides
            return False

    async def prepare_hooks_v2(
        self,
        session_id: str,
        *,
        reason="startup",
        initiator="manual",
        configuration=None,
        _hook_read=None,
    ):
        """Initialize only at validated execution admission, never at view access.

        ``_hook_read`` is the submitting attempt's own earlier full consent
        read (``ConsoleHookAttemptRead``). While it stands for this session
        and owner, and only when it answers "no v2 hook to prepare"
        (``_hook_read_prepares_nothing``), it replaces the
        ``v2_configuration()`` re-read (ADR-225 decision 3). Everything that
        builds, keeps or replaces an engine still reads fresh, as before.
        """
        from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
        from tldw_chatbook.Agents.run_hooks import load_hooks_config

        self._raise_if_disposed_or_session_fenced(session_id)
        permissions = self.ensure_hook_permissions()
        source = self._capture_hook_preparation_source(session_id, permissions)
        shared = None
        authority = (
            _hook_read.authority_for(permissions, session_id)
            if _hook_read is not None
            else None
        )
        if authority is not None:
            from tldw_chatbook.Agents.hook_permissions import HookPermissions

            # A replaced reader keeps its original call (ADR-225 decision 8).
            if (
                "v2_configuration" not in vars(permissions)
                and type(permissions).v2_configuration
                is HookPermissions.v2_configuration
            ):
                shared = permissions.attempt_v2_configuration(authority)
        if shared is not None and self._hook_read_prepares_nothing(
            session_id, shared[0], configuration
        ):
            # The early return below, reached without the re-read: nothing in
            # the attempt's read or this session's state needs an engine.
            self._require_hook_preparation_source(source)
            self._raise_if_disposed_or_session_fenced(session_id)
            return None
        review, targets = await self._read_hook_preparation(
            permissions.v2_configuration, source
        )
        self._require_hook_preparation_source(source)
        configured = load_hooks_config(
            {"hooks": review.config.section} if review.config.section_present else {}
        )
        engine = self.get_hooks_v2(session_id)
        native = (
            getattr(engine, "native_plugins", None) if configuration is None else None
        )
        if configuration is not None:
            maximum = configuration.skill_context_maximum
            if any(
                row.get("plugin_owned") for row in maximum.get("available_skills", ())
            ):
                local = getattr(
                    self._chat_controller._skills_service, "local_service", None
                )
                service = getattr(local, "plugin_service", None)
                if service is None:
                    raise PermissionError("plugin_hook_authority_unavailable")
                native = await service.hook_configuration(maximum)
                self._require_hook_preparation_source(source)
        signature = (
            configured,
            targets,
            native.signature if native is not None else None,
        )
        engine = self.get_hooks_v2(session_id)
        previous = self._hooks_v2_configured.get(session_id)
        if (
            review.ready
            and not configured.v2_handlers
            and not configured.v2_invalid_admissions
            and (native is None or not native.definitions)
            and engine is None
            and session_id not in self._hooks_v2_engines
            and session_id not in self._hooks_v2_lifecycles
            and session_id not in self._hooks_v2_configured
        ):
            self._raise_if_disposed_or_session_fenced(session_id)
            return None
        context_key = await self._read_hook_preparation(
            functools.partial(self._hooks_v2_context_key, session_id), source
        )
        self._require_hook_preparation_source(source)
        self._raise_if_disposed_or_session_fenced(session_id)
        owner = self._hooks_v2_lifecycles.get(session_id)
        context_changed = owner is not None and owner.context_key != context_key
        if context_changed and native is not None and configuration is None:
            raise PermissionError("plugin_hook_workspace_admission_required")
        if (previous is not None and previous != signature) or context_changed:
            # The controller owns a reversible validation slot at this point.
            owner = self._hooks_v2_lifecycles.get(session_id)
            if owner is not None and getattr(owner, "turn_scope", None) is not None:
                raise RuntimeError("hook replacement requires idle session")
            await self.close_hooks_v2(session_id)
            self._require_hook_preparation_source(source)
            if self._hooks_v2_lifecycles.get(session_id) is owner:
                self._hooks_v2_lifecycles.pop(session_id, None)
            if context_changed and previous is None and engine is not None:
                # Host-injected definitions retain their authority resolver.
                engine = self.ensure_hooks_v2(
                    session_id,
                    engine.definitions,
                    engine.authority_check,
                    enabled=engine.enabled,
                    invalid_admissions=engine.invalid_admissions,
                )
            else:
                engine = None
            reason = "configuration_changed"
        if engine is None:
            if not review.ready:
                raise RuntimeError("Review enabled hooks before execution.")
            if (
                not configured.v2_handlers
                and not configured.v2_invalid_admissions
                and (native is None or not native.definitions)
            ):
                return None
            captured = {target.spec.id: target for target in targets}

            def session_end_current(handler, event):
                owner = getattr(engine, "lifecycle_owner", None)
                return bool(
                    self._disposed
                    and handler.type == "command"
                    and handler.event == "SessionEnd"
                    and not handler.effects
                    and not handler.required
                    and owner is not None
                    and self.get_hooks_v2(session_id) is engine
                    and owner._session_end_current(event)
                )

            def authority(handler, _event, _stage):
                if native is not None and handler.id in native.owners:
                    return (
                        not self._disposed
                        and permissions.configuration_current(review)
                        and native.authority(handler, _event, _stage)
                        and permissions.configuration_current(review)
                    )
                target = captured.get(handler.id)
                return bool(
                    target is not None
                    and (
                        not self._disposed
                        and permissions.target_current(target)
                        or session_end_current(handler, _event)
                        and permissions._session_end_current(
                            review, target, lambda: session_end_current(handler, _event)
                        )
                    )
                )

            def effects_current(handler, _event, _stage):
                if native is not None and handler.id in native.owners:
                    return (
                        not self._disposed
                        and permissions.configuration_current(review)
                        and native.effects_current(handler, _event, _stage)
                    )
                target = captured.get(handler.id)
                return bool(
                    target is not None and not self._disposed
                    and permissions.configuration_current(review)
                    and permissions.target_current(target, refresh=False)
                )

            @contextlib.contextmanager
            def launch_guard(handler, event):
                if native is not None and handler.id in native.owners:
                    from tldw_chatbook import config

                    with config.locked_hooks_config_snapshot() as current:
                        if (
                            current.section_stamp != review.config.section_stamp
                            or not effects_current(handler, event, "launch")
                        ):
                            raise PermissionError("plugin_hook_authority_changed")
                        yield
                else:
                    target = captured[handler.id]
                    guard = (
                        permissions._session_end_launch_guard(
                            review, target, lambda: session_end_current(handler, event)
                        )
                        if session_end_current(handler, event)
                        else permissions.launch_guard(target, tool_name=None)
                    )
                    with guard:
                        yield

            engine = self.ensure_hooks_v2(
                session_id,
                configured.v2_handlers
                + (native.definitions if native is not None else ()),
                authority,
                process_owner=native,
                host_environment=native.host_environment
                if native is not None
                else None,
                event_projector=native.project_event if native is not None else None,
                dependency_required=native.dependency_required
                if native is not None
                else None,
                launch_guard=launch_guard,
                effect_authority_check=effects_current,
                enabled=configured.enabled,
                invalid_admissions=configured.v2_invalid_admissions,
            )
            if native is not None:
                native.engine = engine
                engine.native_plugins = native
            self._hooks_v2_configured[session_id] = signature
        lifecycle = self._hooks_v2_lifecycles.get(session_id)
        if lifecycle is None:

            def current():
                try:
                    return (
                        not self._disposed
                        and self.get_hooks_v2(session_id) is engine
                        and self._hooks_v2_context_key(session_id) == context_key
                        and (
                            session_id not in self._hooks_v2_configured
                            or permissions.configuration_current(review)
                            and all(
                                permissions.target_current(target, refresh=False)
                                for target in targets
                            )
                        )
                    )
                except Exception:  # noqa: BLE001 -- hook boundary
                    return False

            lifecycle = HookSessionLifecycle(engine, session_id, current=current)
            lifecycle.context_key = context_key
            self._hooks_v2_lifecycles[session_id] = lifecycle
            engine.lifecycle_owner = lifecycle
        pending_scope = None
        if configuration is not None:
            pending_scope = lifecycle.open_scope()
            lifecycle.turn_scope = pending_scope
            self._chat_controller._hooks_v2_submissions[asyncio.current_task()] = (
                lifecycle,
                pending_scope,
                session_id,
            )
            try:
                if engine.mcp_executor is not None:
                    context = (
                        await self._chat_controller.compose_prospective_hook_context(
                            configuration, lifecycle, pending_scope
                        )
                    )
                    self._require_hook_preparation_source(source)
                    engine.mcp_executor.bind_context(context)
            except BaseException:
                lifecycle.close_scope(pending_scope)
                lifecycle.turn_scope = None
                raise
        if not lifecycle.live:
            token = lifecycle.reserve(
                lifecycle.event(
                    "SessionStart",
                    data={"reason": reason},
                    initiator=initiator,
                )
            )
            try:
                await lifecycle.initialize(token)
                self._require_hook_preparation_source(source)
                lifecycle.publish(token)
            except BaseException:
                lifecycle.cancel(token)
                if pending_scope is not None:
                    lifecycle.close_scope(pending_scope)
                    lifecycle.turn_scope = None
                # A failed provisional initialization cannot remove a successor.
                if self._hooks_v2_lifecycles.get(session_id) is lifecycle:
                    self._hooks_v2_lifecycles.pop(session_id, None)
                raise
        if configuration is not None and engine.mcp_executor is not None:
            engine.mcp_executor.retain_runtime(session_id)
        return lifecycle

    def get_hooks_v2(self, session_id: str) -> Any:
        """Return the exact pinned snapshot, including disabled/closed requirements."""
        with self._run_hooks_lock:
            return self._hooks_v2_engines.get(session_id)

    def _seal_hooks_v2(self, session_id: str | None = None) -> None:
        with self._run_hooks_lock:
            engines = (
                tuple(self._hooks_v2_engines.values())
                if session_id is None
                else (self._hooks_v2_engines.get(session_id),)
            )
        # Currentness reads take the map lock from the checkpoint condition.
        # Never enter checkpoints while holding the map lock. Fence admission
        # and cancel immediately, before any checkpoint condition can block.
        for engine in engines:
            if engine is not None:
                engine.begin_close()
        for engine in engines:
            if engine is not None:
                owner = getattr(engine, "lifecycle_owner", None)
                if owner is not None:
                    owner.seal()

    @property
    def hooks_v2_cleanup_pending(self) -> bool:
        """Unresolved process/launch owners remain attached after disposal."""
        return any(engine.cleanup_pending for engine in self._hooks_v2_engines.values())

    async def close_hooks_v2(self, session_id: str | None = None) -> None:
        """Join retained cleanup; caller cancellation cannot cancel the owner.

        H4/H5 publish authorized teardown via the existing session engine before
        this final close. Each engine's seal fixes the 3 + 5 second allowance;
        waiting here does not reset it or admit ordinary work.
        """
        self._seal_hooks_v2(session_id)
        if session_id is not None:
            engine = self._hooks_v2_engines.get(session_id)
            if engine is not None:
                await engine.close()
                if not engine.cleanup_pending:
                    self._hooks_v2_engines.pop(session_id, None)
            return
        if self._hooks_v2_cleanup_task is None:

            async def drain() -> None:
                await asyncio.gather(
                    *(
                        engine.close()
                        for engine in tuple(self._hooks_v2_engines.values())
                    )
                )

            self._hooks_v2_cleanup_task = asyncio.create_task(
                drain(), name="console-v2-hook-cleanup"
            )
        await asyncio.shield(self._hooks_v2_cleanup_task)

    def start_async_lifecycles(self) -> None:
        """Start loop-bound runtime work after the Textual loop is running.

        ``TldwCli`` is constructed synchronously before ``App.run`` creates
        its event loop, so the constructor's best-effort watcher start cannot
        cover the shipping CLI lifecycle by itself.  App mount calls this
        idempotent handoff independently of Canvas preview creation.
        """

        self._start_canvas_policy_watcher()

    def _canvas_maintenance_close_admission(self) -> None:
        """Stop policy polling without turning temporary storage pause into disable."""
        with self._canvas_native_lock:
            self._canvas_maintenance_closed = True
            self._canvas_maintenance_generation += 1

    async def _canvas_maintenance_drain(self, deadline: float) -> bool:
        """Retain the watcher and accepted revocation until they actually finish."""
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        if not self._canvas_maintenance_closed:
            raise RecoveryRequired("runtime_producer_not_closed")
        tasks = set(self._canvas_policy_cleanups)
        if self._canvas_policy_read_task is not None:
            tasks.add(self._canvas_policy_read_task)
        if self._canvas_policy_watch_task is not None:
            tasks.add(self._canvas_policy_watch_task)
        if tasks:
            done, pending = await asyncio.wait(
                tasks, timeout=max(0.0, deadline - time.monotonic())
            )
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    raise RecoveryRequired("runtime_work_not_settled")
            if pending or self._canvas_policy_cleanups:
                return False
        return True

    def _canvas_maintenance_resume(self) -> None:
        """Reopen policy checks after native storage readmission, preserving latches."""
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        watcher = self._canvas_policy_watch_task
        reader = self._canvas_policy_read_task
        if (
            self._canvas_policy_cleanups
            or (watcher is not None and not watcher.done())
            or (reader is not None and not reader.done())
        ):
            raise RecoveryRequired("runtime_work_not_settled")
        with self._canvas_native_lock:
            self._canvas_maintenance_closed = False
        self._start_canvas_policy_watcher()

    async def apply_canvas_policy(self) -> None:
        """Idempotently revoke all browser delivery after Canvas is disabled.

        Re-enabling requires a process restart. Stored and staged artifact data
        remain owned by their existing repositories and lifecycle controllers.
        """

        if self._canvas_enabled():
            return
        with self._canvas_native_lock:
            if self._canvas_maintenance_closed or not self.canvas_disabled():
                return
            self._canvas_disabled_latched = True
            gateway, self._canvas_gateway = self._canvas_gateway, None
            authority, self._canvas_native_authority = (
                self._canvas_native_authority,
                None,
            )
            self._canvas_gateway_authority = None
            self._canvas_native_view_binding = None
        async def cleanup() -> None:
            close_gateway = getattr(gateway, "aclose", None)
            if callable(close_gateway):
                result = close_gateway()
                if inspect.isawaitable(result):
                    await result
            dispose_authority = getattr(authority, "dispose", None)
            if callable(dispose_authority):
                result = dispose_authority()
                if inspect.isawaitable(result):
                    await result

        completion = asyncio.create_task(cleanup(), name="console-canvas-policy-cleanup")
        self._canvas_policy_cleanups.add(completion)

        def finished(task):
            # Unproved cleanup keeps maintenance closed; observe detached errors.
            if not task.cancelled() and task.exception() is None:
                self._canvas_policy_cleanups.discard(task)

        completion.add_done_callback(finished)
        await asyncio.shield(completion)

    def _start_canvas_policy_watcher(self) -> None:
        """Watch shared config while a native Canvas preview can be open."""

        if self._disposed or self._canvas_maintenance_closed:
            return
        task = self._canvas_policy_watch_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._canvas_policy_watch_task = loop.create_task(
            self._watch_canvas_policy(),
            name="console-canvas-policy-watch",
        )

    async def _watch_canvas_policy(self) -> None:
        """Revoke native delivery promptly after an external config disable."""

        while not self._canvas_maintenance_closed:
            # Native policy reads may wait for a config writer. Keep the UI
            # runnable, retaining the actual read if the watcher is cancelled.
            reader = self._canvas_policy_read_task
            if reader is None:
                reader = asyncio.create_task(
                    asyncio.to_thread(self._canvas_enabled),
                    name="console-canvas-policy-read",
                )
                self._canvas_policy_read_task = reader
            enabled = await asyncio.shield(reader)
            self._canvas_policy_read_task = None
            if not enabled and self.canvas_disabled():
                await self.apply_canvas_policy()
                return
            await asyncio.sleep(0.25)

    def _sync_canvas_native_context(self, session_id: str | None) -> None:
        """Forward live store context changes to an already-built authority."""

        authority = self._canvas_native_authority
        if authority is not None:
            authority.sync_live_context(session_id)

    # -- construction ------------------------------------------------------

    def ensure_chat_store(
        self,
        *,
        workspace_context: Any | None = None,
        on_scope_flushed: Callable[..., Any] | None = None,
    ) -> "ConsoleChatStore":
        """Return the Console chat store, creating it lazily.

        Moved verbatim from `ChatScreen._ensure_console_chat_store`: the
        durable `ChatPersistenceService` is attached only when the app has
        a ChaChaNotes DB, and the citation repository is dropped when it
        belongs to a different DB than the one being persisted to.

        Args:
            workspace_context: The view's current
                `ConsoleWorkspaceContext`. Read at construction only.
            on_scope_flushed: The view's session-scope-flushed callback.
                Read at construction only.

        Returns:
            ConsoleChatStore: The runtime's store.
        """
        if self._chat_store is not None or self._disposed:
            return self._chat_store
        from tldw_chatbook.Chat.chat_persistence_service import (
            ChatPersistenceService,
        )
        from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
        from tldw_chatbook.Chat.console_library_policy_coordinator import (
            ConsoleLibraryPolicyCoordinator,
        )
        from tldw_chatbook.Chat.console_trace_projection import ConsoleTraceProjection

        persistence = None
        db = getattr(self._app, "chachanotes_db", None)
        if db is not None:
            try:
                recover_console_trace_calls(db)
            except Exception as exc:
                logger.warning("console_trace_recovery_failed: {}", type(exc).__name__)
            citation_repository = getattr(
                self._app,
                "citation_trace_repository",
                None,
            )
            if (
                citation_repository is not None
                and getattr(citation_repository, "db", None) is not db
            ):
                citation_repository = None
            persistence = ChatPersistenceService(
                db,
                workspace_registry=getattr(
                    self._app,
                    "workspace_registry_service",
                    None,
                ),
                citation_repository=citation_repository,
            )
            persistence.retry_recovered_media_references()
            legacy_normalization_enabled = callable(getattr(db, "transaction", None))
            legacy_normalizer: Any | None = None
            native_reader: Any | None = None

            def get_legacy_normalizer() -> Any:
                """Build the legacy adapter only after first paint or first use."""

                nonlocal legacy_normalizer
                if legacy_normalizer is None:
                    from tldw_chatbook.Chat.console_trace_legacy import (
                        LegacyTraceNormalizer,
                    )

                    legacy_normalizer = LegacyTraceNormalizer(db)
                return legacy_normalizer

            def get_native_reader() -> Any:
                """Build the native ledger reader only on first trace inspection."""

                nonlocal native_reader
                if native_reader is None:
                    from tldw_chatbook.Chat.console_trace_native_reader import (
                        ConsoleTraceNativeReader,
                    )

                    native_reader = ConsoleTraceNativeReader(
                        db,
                        repository=persistence.console_trace_repository,
                    )
                return native_reader

            def read_normalized_calls(message_id: str) -> Any:
                """Read native calls first, followed by migrated legacy snapshots."""

                return (
                    *get_native_reader().read_calls(message_id),
                    *get_legacy_normalizer().read_calls(message_id),
                )
        else:
            legacy_normalization_enabled = False
        from tldw_chatbook.Chat.console_canvas_controller import (
            ConsoleCanvasController,
        )

        snapshot = self._ensure_canvas_profile_snapshot()
        durable_canvas_service = None
        if db is not None:
            from tldw_chatbook.Canvas.service import CanvasService
            from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

            if isinstance(db, CharactersRAGDB):
                durable_canvas_service = CanvasService(db, profile_snapshot=snapshot)
        self._canvas_controller = ConsoleCanvasController(
            durable_service=durable_canvas_service, profile_snapshot=snapshot
        )
        self.set_chat_store(ConsoleChatStore(
            persistence=persistence,
            assistant_defaults_provider=self._resolve_new_console_assistant,
            on_assistant_default_notice=lambda notice: self._app.notify(
                notice, severity="warning"
            ),
            settle_provider_traces_off_thread=True,
            trace_projection=(
                ConsoleTraceProjection(
                    legacy_reader=db.get_message_exchanges,
                    normalized_reader=(
                        read_normalized_calls if legacy_normalization_enabled else None
                    ),
                    normalized_reads_enabled=lambda: (
                        runtime_capture_policy().normalized_reads_enabled
                    ),
                    normalized_writes_enabled=lambda: (
                        runtime_capture_policy().normalized_writes_enabled
                    ),
                    compatibility_metrics=self.trace_compatibility_metrics,
                )
                if db is not None
                else None
            ),
            workspace_context=workspace_context,
            on_scope_flushed=on_scope_flushed,
            library_policy_coordinator=(
                ConsoleLibraryPolicyCoordinator(
                    persistence.console_library_policy_repository
                )
                if persistence is not None
                else None
            ),
            library_policy_defaults_provider=lambda: _current_library_policy_defaults(
                self._app
            ),
            thinking_history_policy_default_provider=lambda: (
                _current_thinking_history_policy_default(self._app)
            ),
            canvas_promotion_participant=self._canvas_controller,
            canvas_turn_controller=self._canvas_controller,
            on_canvas_context_changed=self._sync_canvas_native_context,
        )
        )
        if db is not None and legacy_normalization_enabled:
            self._schedule_legacy_trace_maintenance(db, get_legacy_normalizer)
        self._bind_view_hooks()
        return self._chat_store

    def _resolve_new_console_assistant(self, workspace_id: str, settings: Any) -> Any:
        """Resolve new-chat identity without requiring a mounted Console view."""
        from collections.abc import Mapping

        from ..Workspaces.models import DEFAULT_WORKSPACE_ID
        from .console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID
        from .console_session_settings import (
            ConsoleAssistantStartup,
            blank_console_session_settings,
        )

        # Default/global scopes never inherit a Persona (ADR-079). Keep their
        # ordinary boot path free of workspace Persona resolution.
        if workspace_id in (CONSOLE_GLOBAL_WORKSPACE_ID, DEFAULT_WORKSPACE_ID, ""):
            config = getattr(self._app, "app_config", {})
            return ConsoleAssistantStartup(
                settings
                or blank_console_session_settings(
                    config if isinstance(config, Mapping) else {}
                )
            )
        from .console_assistant_defaults import resolve_new_console_assistant

        return resolve_new_console_assistant(self._app, workspace_id, settings)

    def _schedule_legacy_trace_maintenance(
        self,
        database: Any,
        normalizer_factory: Callable[[], Any],
    ) -> None:
        """Start one yielding post-readiness legacy-normalization worker."""

        if self._legacy_trace_maintenance_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def provider_active() -> bool:
            controller = self._chat_controller
            tasks = getattr(controller, "_active_stream_tasks", None)
            return bool(tasks)

        async def run_maintenance_call(operation, *args, **kwargs):
            # Only this finite database callback survives cancellation. The
            # outer scheduler still stops before admitting another batch.
            owned = asyncio.Task(
                run_owned_db_call(database, operation, *args, **kwargs), loop=loop
            )
            try:
                return await asyncio.shield(owned)
            except asyncio.CancelledError:
                while not owned.done():
                    try:
                        await asyncio.shield(owned)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not owned.cancelled():
                    owned.exception()
                raise

        async def run() -> None:
            while not self._disposed and not getattr(self._app, "_ui_ready", True):
                await asyncio.sleep(0.05)
            if self._disposed:
                return
            await asyncio.sleep(LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS)
            if self._disposed:
                return
            from tldw_chatbook.Chat.console_trace_maintenance import (
                LegacyTraceMaintenance,
            )

            maintenance = LegacyTraceMaintenance(
                database,
                normalizer=normalizer_factory(),
                provider_active=provider_active,
            )
            from tldw_chatbook.Chat.chat_persistence_service import (
                trace_maintenance_work_generation,
            )

            last_provider_activity = time.monotonic()
            last_physical_attempt = 0.0
            last_collected_epoch: int | None = None
            pending_gc_result: Any | None = None
            # PERF-10 (TASK-33269): once a pass finds nothing to normalize and
            # the GC interval has not elapsed, park. Parked, the loop does no
            # database work (each run_batch was a write transaction, an
            # admission and a helper-spawning connection, once a second
            # forever). It wakes when an exchange row is written, and once per
            # GC interval regardless: trace-call state, retention roots,
            # semantic revisions and other processes advance the graph epoch
            # without signalling, and a failed GC attempt must be retried.
            parked = False
            seen_work = trace_maintenance_work_generation()
            while not self._disposed:
                if parked:
                    await asyncio.sleep(LEGACY_TRACE_MAINTENANCE_PARK_POLL_SECONDS)
                    if trace_maintenance_work_generation() != seen_work or (
                        time.monotonic() - last_physical_attempt
                        >= TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS
                    ):
                        parked = False
                    continue
                # TASK-33801: an exchange written since the last pass -- while
                # parked, during that pass or its cleanup -- means this pass has
                # work, so it skips the idle check's extra admission.
                current_work = trace_maintenance_work_generation()
                if current_work != seen_work:
                    maintenance.expect_work = True
                seen_work = current_work
                try:
                    result = await run_maintenance_call(maintenance.run_batch)
                except Exception as exc:  # noqa: BLE001 - retry remains restart-safe
                    logger.warning(
                        "legacy trace maintenance paused after {}",
                        type(exc).__name__,
                    )
                    await asyncio.sleep(LEGACY_TRACE_MAINTENANCE_RETRY_DELAY_SECONDS)
                    continue
                if result.logical_complete:
                    now = time.monotonic()
                    if provider_active():
                        last_provider_activity = now
                        await asyncio.sleep(1.0)
                        continue
                    if (
                        now - last_physical_attempt
                        < TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS
                    ):
                        parked = True
                        continue
                    last_physical_attempt = now
                    try:
                        from tldw_chatbook.Chat.console_trace_maintenance import (
                            PhysicalTraceCompactor,
                            TraceGarbageCollector,
                        )
                        from tldw_chatbook.Chat.console_trace_models import (
                            new_opaque_id,
                        )
                        from tldw_chatbook.config import (
                            resolve_trace_compaction_policy,
                        )

                        controller = self._chat_controller
                        pause = getattr(
                            controller,
                            "pause_trace_maintenance_dispatch",
                            lambda: None,
                        )
                        resume = getattr(
                            controller,
                            "resume_trace_maintenance_dispatch",
                            lambda: None,
                        )
                        app_config = getattr(self._app, "app_config", {}) or {}
                        console_config = (
                            app_config.get("console", {})
                            if isinstance(app_config, Mapping)
                            else {}
                        )
                        controller_idle = getattr(
                            controller,
                            "trace_maintenance_idle_seconds",
                            None,
                        )
                        collector = TraceGarbageCollector(database)
                        current_epoch = await run_maintenance_call(
                            collector.current_graph_epoch
                        )
                        if pending_gc_result is None:
                            if current_epoch == last_collected_epoch:
                                await asyncio.sleep(1.0)
                                continue
                            pending_gc_result = await run_maintenance_call(
                                collector.collect,
                                request_id=f"auto-{new_opaque_id()}",
                            )
                            last_collected_epoch = int(
                                getattr(
                                    pending_gc_result,
                                    "marked_epoch",
                                    current_epoch,
                                )
                            )
                        compactor = PhysicalTraceCompactor(
                            database,
                            policy=resolve_trace_compaction_policy(console_config),
                            provider_active=provider_active,
                            idle_seconds=(
                                controller_idle
                                if callable(controller_idle)
                                else lambda: max(
                                    0.0, time.monotonic() - last_provider_activity
                                )
                            ),
                            pause_dispatch=pause,
                            resume_dispatch=resume,
                            cancel_requested=lambda: self._disposed,
                        )
                        outcome = await run_maintenance_call(
                            compactor.run_after_gc,
                            pending_gc_result,
                        )
                        if outcome.reason_code == "logical_gc_unavailable":
                            pending_gc_result = None
                            last_collected_epoch = None
                        elif outcome.completed or (
                            outcome.reason_code
                            not in TRACE_PHYSICAL_MAINTENANCE_RETRYABLE_REASONS
                        ):
                            pending_gc_result = None
                    except ImportError:
                        # Narrow test doubles may provide only the legacy worker.
                        pass
                    except Exception as exc:  # noqa: BLE001 - durable retry state
                        logger.warning(
                            "trace physical maintenance paused after {}",
                            type(exc).__name__,
                        )
                    await asyncio.sleep(1.0)
                    continue
                if not result.admitted:
                    await asyncio.sleep(1.0)
                    continue
                await asyncio.sleep(0)

        self._legacy_trace_maintenance_task = loop.create_task(run())

    def ensure_provider_gateway(
        self,
        *,
        config_provider: Callable[[], Any] | None = None,
        trace_call_boundary_factory: Callable[[Any, Any, Any], object] | None = None,
    ) -> Any:
        """Return the Console provider gateway, creating it lazily.

        Moved verbatim from `ChatScreen._ensure_console_provider_gateway`,
        including the `console_provider_gateway_factory` test-injection
        seam read off the app.

        Args:
            config_provider: Fresh-config source handed to a
                real gateway; the gateway re-resolves readiness at send
                time and must see Settings saves made after boot. Ignored
                when the app supplies a factory.
            trace_call_boundary_factory: Optional hard-off seam that owns
                durable reservation through pre-adapter dispatch-start.

        Returns:
            Any: The runtime's provider gateway.
        """
        if self._provider_gateway is not None or self._disposed:
            return self._provider_gateway
        factory = getattr(self._app, "console_provider_gateway_factory", None)
        if callable(factory):
            self._provider_gateway = factory()
        else:
            from tldw_chatbook.Chat.console_provider_gateway import (
                ConsoleProviderGateway,
            )

            if trace_call_boundary_factory is None:
                database = getattr(self._app, "chachanotes_db", None)
                if database is not None and callable(
                    getattr(database, "transaction", None)
                ):
                    chat_store = self.ensure_chat_store()
                    persistence = getattr(chat_store, "persistence", None)
                    repository = getattr(
                        persistence,
                        "console_trace_repository",
                        None,
                    )
                    trace_call_boundary_factory = _LazyTraceBoundaryFactory(
                        database,
                        repository=repository,
                    )

            self._provider_gateway = ConsoleProviderGateway(
                config_provider=functools.partial(
                    _provider_config_for_app, self._app
                ),
                trace_call_boundary_factory=trace_call_boundary_factory,
                normalized_writes_enabled=lambda: (
                    runtime_capture_policy().normalized_writes_enabled
                ),
                trace_compatibility_metrics=self.trace_compatibility_metrics,
            )
        return self._provider_gateway

    def ensure_activity_receipt_service(self) -> Any | None:
        """Create local result storage without constructing Console or a provider.

        Call via ``asyncio.to_thread`` from async presentation code. Construction
        is serialized with other readers and disposal, and background callers
        close their initialization connection before handing ownership back.

        Returns:
            The app-owned lazy receipt service, or None without a durable local
            profile database or after shutdown admission has closed.
        """
        with self._activity_receipts_lock:
            if self._disposed:
                return None
            if self._activity_receipts is not None:
                return self._activity_receipts
            app = self._app
            db = getattr(app, "chachanotes_db", None)
            db_path = getattr(db, "db_path", None) if db is not None else None
            marks = getattr(app, "conversation_local_marks_service", None)
            receipt_source = getattr(_INITIAL_ACTIVITY_RECEIPT_SCOPE, "proof", None)
            require_source = (receipt_source[1] if receipt_source is not None
                              and receipt_source[0] is self else None)
            if not db_path or str(db_path) == ":memory:":
                return None
            from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

            runs_db = AgentRunsDB(Path(db_path).parent / "agent_runs.db")
            published = False
            try:
                service = _LazyConsoleActivityReceiptService(
                    runs_db,
                    marks,
                )
                # Dispose writes its lifetime latch under this same lock. Never
                # publish an owner between that latch and its resource snapshot.
                with self._canvas_native_lock:
                    if (
                        (require_source is not None and not require_source())
                        or self._disposed
                        or self._app is not app
                        or getattr(app, "chachanotes_db", None) is not db
                        or getattr(db, "db_path", None) != db_path
                        or getattr(app, "conversation_local_marks_service", None) is not marks
                    ):
                        return None
                    self._agent_runs_db = runs_db
                    self._activity_receipts = service
                    published = True
                    return service
            finally:
                # AgentRunsDB.close affects only the calling thread. A worker's
                # held initialization connection cannot be closed by app exit.
                if (not published or self._disposed
                        or get_ident() != self._receipt_owner_thread_id):
                    runs_db.close()

    async def _prepare_initial_activity_receipts(
        self, app: Any, *, require_current: Callable[[], None]
    ) -> bool:
        """Prepare only stock receipt storage under the initial startup task.

        The original synchronous reader and UI-bound bridge APIs stay unchanged.
        A selected callback retains custody through physical return, including
        repeated cancellation of this awaiting startup task.
        """
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        anchor = _INITIAL_ACTIVITY_RECEIPT_READERS
        scope = anchor[2]
        if _initial_receipt_preparer(self) is None:
            raise RuntimeError("initial_receipt_source_changed")
        database = getattr(app, "chachanotes_db", None)
        path = getattr(database, "db_path", None)
        if (
            type(database) is not CharactersRAGDB
            or database.is_memory_db
            or not path
        ):
            return False
        require_current()
        if (
            self._app is not app
            or getattr(app, "console_runtime", None) is not self
            or self._disposed
            or threading.get_ident() != self._receipt_owner_thread_id
            or (self._canvas_policy_watch_task is not None
                and self._canvas_policy_watch_task.get_loop() is not asyncio.get_running_loop())
        ):
            raise RuntimeError("initial_receipt_owner_changed")
        if self._activity_receipts is not None:
            return True
        marks = getattr(app, "conversation_local_marks_service", None)
        generation = self.generation
        owner_thread = threading.current_thread()
        owner_loop, owner_task = asyncio.get_running_loop(), asyncio.current_task()
        if owner_thread.ident != self._receipt_owner_thread_id:
            raise RuntimeError("initial_receipt_owner_changed")
        reader = MethodType(anchor[1][0][1], self)

        def source_current() -> bool:
            return (
                _INITIAL_ACTIVITY_RECEIPT_READERS is anchor
                and _INITIAL_ACTIVITY_RECEIPT_SCOPE is scope
                and _initial_receipt_preparer(self) is not None
                and self._app is app
                and getattr(app, "console_runtime", None) is self
                and not self._disposed
                and self.generation == generation
                and getattr(app, "chachanotes_db", None) is database
                and getattr(database, "db_path", None) == path
                and getattr(app, "conversation_local_marks_service", None) is marks
            )

        def initialize() -> Any:
            previous = getattr(scope, "proof", None)
            scope.proof = (self, source_current)
            try:
                if not source_current():
                    raise RuntimeError("initial_receipt_source_changed")
                result = reader()
                if not source_current():
                    raise RuntimeError("initial_receipt_source_changed")
                return result
            finally:
                if previous is None:
                    del scope.proof
                else:
                    scope.proof = previous

        coroutine = asyncio.to_thread(initialize)
        try:
            pending = asyncio.create_task(coroutine, name="initial_console_receipts")
        except BaseException:
            coroutine.close()
            raise
        try:
            result = await asyncio.shield(pending)
        except asyncio.CancelledError:
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not pending.cancelled():
                try:
                    pending.result()
                except Exception:
                    pass
            raise
        if (
            threading.current_thread() is not owner_thread
            or asyncio.get_running_loop() is not owner_loop
            or asyncio.current_task() is not owner_task
            or not source_current()
        ):
            raise RuntimeError("initial_receipt_owner_changed")
        require_current()
        return result is not None

    def ensure_agent_bridge(
        self,
        *,
        store_factory: Callable[[], Any],
        provider_gateway_factory: Callable[[], Any],
        skills_service: Any | None = None,
        native_tools_enabled_factory: Callable[[], Any] | None = None,
    ) -> Any:
        """Return the Console agent bridge, creating it lazily.

        Moved verbatim from
        `ConsoleAgentController._ensure_console_agent_bridge`, ordering
        included: the durable-DB probe runs FIRST and returns `None` (no
        agent runtime) before the store or the gateway is touched, so an
        in-memory harness still builds neither.

        Factories remain in this compatibility signature to preserve the
        durable-DB probe ordering. The bridge's stored native-tools gate is
        always app-owned and never retains a view callback.

        Args:
            store_factory: Returns the chat store the bridge should use.
                Called only past the durable-DB probe.
            provider_gateway_factory: Returns the provider gateway. Same.
            skills_service: The app's skills scope service, or `None`. A
                plain value: it is a `getattr` on the APP, which the probe
                has already touched.
            native_tools_enabled_factory: Compatibility-only legacy seam;
                the bridge stores the app-owned native-tools gate.

        Returns:
            Any: The `ConsoleAgentBridge`, or `None` when there is no
            durable ChaChaNotes DB to key the sibling `AgentRunsDB` off.
        """
        if self._agent_bridge is not None or self._disposed:
            return self._agent_bridge
        if self.ensure_activity_receipt_service() is None or self._disposed:
            return None
        from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

        runs_db = self._agent_runs_db
        # TASK-1971 (Agent Change Review): the tracker is None when git is
        # absent -- the bridge then skips tracking entirely, and runs behave
        # exactly as before the feature existed (spec gating decision).
        from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker

        change_tracker = ChangeTurnTracker()
        change_coordinator = None
        if change_tracker.available:
            from tldw_chatbook.Workspaces.change_review_finalization import (
                ChangeReviewFinalizationCoordinator,
            )

            def _publish_change_review(item: Any) -> None:
                runs_db.record_change_snapshots_batch(
                    run_id=item.run_id,
                    records=[record.__dict__ for record in item.records],
                    kind=item.kind,
                )

            change_coordinator = ChangeReviewFinalizationCoordinator(
                tracker=change_tracker,
                publish=_publish_change_review,
                close_publisher=runs_db.close,
            )
            self._change_review_coordinator = change_coordinator
        self._agent_bridge = ConsoleAgentBridge(
            agent_runs_db=runs_db,
            runtime_capacity_factory=self._get_execution_capacity,
            store=store_factory(),
            provider_gateway=provider_gateway_factory(),
            skills_service=skills_service,
            native_tools_enabled=(
                functools.partial(_native_tools_enabled_for_app, self._app)
            ),
            change_tracker=change_tracker if change_tracker.available else None,
            buddy_sink=self.persona_buddy_sink,
            change_finalization_coordinator=change_coordinator,
            # run-hooks (Task 5): hand the bridge this runtime's engine
            # accessor so per-turn fire sites (PostToolUse today; PreToolUse
            # and the settle events in later tasks) resolve the app-owned
            # singleton through the same runtime a mounted Console uses --
            # never a view, so headless wake runs reach it identically.
            ensure_run_hooks=self.ensure_run_hooks,
            get_hooks_v2=self.get_hooks_v2,
        )
        # PR3a-2 Task 4: the survivor-completion attention consumer (durable
        # unseen mark + app-wide toast + deep link), registered NEXT TO
        # bridge construction per `FleetDrainFanout.register`'s contract.
        # Captures the APP object only -- never a screen -- because the
        # bridge (and its registered consumers) outlives the screen whenever
        # a survivor is still running at teardown.
        from tldw_chatbook.Chat.console_fleet_attention import (
            register_fleet_attention,
        )

        register_fleet_attention(
            self._agent_bridge,
            self._app,
            receipt_service=self._activity_receipts,
        )
        return self._agent_bridge

    def ensure_activity_hydration(self) -> asyncio.Task[int] | None:
        """Start or reuse the one off-loop receipt hydration for this runtime."""
        service = self._activity_receipts
        if service is None or self._disposed:
            return None
        task = self._activity_hydration_task
        if task is not None and not task.done():
            return task
        if service.hydration_state() == "ready":
            return self._activity_hydration_task
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        token = self.authority_token
        runs_db = self._agent_runs_db

        def read_receipts() -> int:
            from .console_activity_receipts import ConsoleActivityReceiptService
            from .conversation_local_marks_service import ConversationLocalMarksService
            from ..DB.AgentRuns_DB import AgentRunsDB
            from ..DB.ChaChaNotes_DB import CharactersRAGDB
            from ..DB.base_db import operation_owned_connection

            finite_runs = type(runs_db) is AgentRunsDB and not runs_db.is_memory_db
            marks = (
                service._marks
                if type(service)
                in {_LazyConsoleActivityReceiptService, ConsoleActivityReceiptService}
                else None
            )
            notes_db = (
                marks.db if type(marks) is ConversationLocalMarksService else None
            )
            with contextlib.ExitStack() as owned:
                if finite_runs:
                    owned.enter_context(operation_owned_connection(runs_db))
                if type(notes_db) is CharactersRAGDB and not notes_db.is_memory_db:
                    owned.enter_context(operation_owned_connection(notes_db))
                try:
                    return service.hydrate_from_storage()
                finally:
                    if not finite_runs and runs_db is not None:
                        runs_db.close()

        async def hydrate() -> int:
            from .console_preparation_reads import run_preparation_read

            result = await run_preparation_read(
                read_receipts,
                creator=self,
                session_id=None,
                reads=self._preparation_reads,
                require_current=lambda: None,
            )
            if self._disposed or self.authority_token != token:
                return 0
            return result

        # This private finite owner must not depend on a configurable task factory.
        task = asyncio.Task(hydrate(), loop=loop)
        self._activity_hydration_task = task
        return task

    async def _drain_activity_hydration(self) -> bool:
        """Join the exact hydration owner, retaining cancellation until it settles."""
        task = self._activity_hydration_task
        cancelled = False
        if task is None:
            return cancelled
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled |= not task.done()
            except Exception:  # noqa: BLE001 - consume the completed read failure.
                break
        self._consume_task_outcome(task)
        return cancelled

    def ensure_chat_controller(self, **kwargs: Any) -> "ConsoleChatController":
        """Return the Console chat controller, creating it lazily.

        Selection values are passed through from the custody-producing view.
        Every callable needed after custody is replaced here by an app-owned
        service or runtime-owned frozen-input accessor; only disposable UI
        projections are rebound on attach.

        Returns:
            ConsoleChatController: The runtime's controller.
        """
        if self._chat_controller is not None or self._disposed:
            return self._chat_controller
        from tldw_chatbook.Chat.console_chat_controller import (
            ConsoleChatController,
        )

        # Run hooks (spec 2026-09-11, Task 7): hand the controller this
        # runtime's engine accessor exactly as `ensure_agent_bridge` hands
        # it to the bridge (Task 5) -- the production path always wires it,
        # while the optional param's `None` default keeps every direct
        # (controller-only) construction, tests included, unchanged.
        kwargs.setdefault("ensure_run_hooks", self.ensure_run_hooks)
        kwargs.setdefault("hook_permissions_accessor", self.ensure_hook_permissions)
        kwargs.update(
            chat_dictionary_applier=functools.partial(
                _apply_chat_dictionaries_for_app, self._app
            ),
            world_info_applier=functools.partial(
                _apply_world_info_for_app, self._app
            ),
            # The LIVE seam (two arguments). Every turn this runtime admits
            # carries its frozen capture and release explicitly
            # (``staged_evidence_capture`` / ``staged_evidence_release``);
            # the controller no longer looks for them by name (TASK-34352).
            rag_capture_provider=self._capture_console_staged_rag,
            staged_evidence_provider=self._has_staged_evidence,
            default_session_settings=functools.partial(
                _default_session_settings_for_app,
                self._app,
            ),
            library_provider_factory=functools.partial(
                _library_provider_for_app, self._app
            ),
            global_user_display_name=functools.partial(
                _global_user_display_name_for_app, self._app
            ),
            turn_context_provider=None,
            provider_config=functools.partial(_provider_config_for_app, self._app),
            confirm_project_instruction_dispatch=(
                self._confirm_project_instruction_dispatch
            ),
            select_project_instruction_binding=(
                self._select_project_instruction_binding
            ),
        )
        kwargs.setdefault("buddy_sink", self.persona_buddy_sink)
        kwargs.setdefault("scratch_spaces", self._scratch_spaces)
        kwargs.setdefault("activity_receipts", self._activity_receipts)
        if "canvas_enabled_reader" not in kwargs:
            kwargs["canvas_enabled_reader"] = self._canvas_enabled
            kwargs.setdefault("canvas_disabled_reader", self.canvas_disabled)
        raw_cli_runtime = getattr(self._app, "raw_cli_runtime", None)
        kwargs.setdefault(
            "cancel_raw_cli_session",
            getattr(raw_cli_runtime, "cancel_session", None),
        )
        self.set_chat_controller(ConsoleChatController(**kwargs))
        self._chat_controller.prompt_history = self.ensure_prompt_history()
        if self.view is None:
            # task-15860 Task 4: a runtime can be VIEWLESS FROM BIRTH, not
            # only after a detach — nothing about the caller that supplies
            # these constructor parameters makes it an attached view, and
            # the wake-at-launch case (Console never opened) has no view at
            # all. Without this the fresh controller would keep the
            # constructor's own `None`s, and `wake_conversation_in_view`'s
            # read site reads that as IN VIEW: the ◈ mark cleared for a
            # delivery nobody could have seen. `only="controller"` because
            # the STORE's one slot (`on_scope_flushed`) is a constructor
            # parameter its own caller just supplied, and nulling it in the
            # restore-before-attach window would drop scope flushes.
            # Production attaches one line later
            # (`ChatScreen._ensure_console_chat_controller`), so this costs
            # a mounted Console nothing.
            self._clear_view_hooks(only="controller")
            self._clear_view_hooks(only="wake")
        else:
            self._bind_view_hooks()
        # ADR-135: the native controller's birth owns recovery. View remounts
        # and repeated ensure/read calls return above without auditing owners.
        wake = self._chat_controller.fleet_wake
        wake.wire(
            app=self._app,
            startup_ready=lambda: bool(getattr(self._app, "_ui_ready", True)),
        )
        wake.start_recovery()
        return self._chat_controller

    def ensure_hook_permissions(self) -> HookPermissions:
        """Return the app-owned consent owner, independent of any view."""
        with self._run_hooks_lock:
            if self._disposed:
                raise RuntimeError("Console runtime is disposed.")
            if self._hook_permissions is None:
                from tldw_chatbook.Agents.hook_permissions import HookPermissions

                self._hook_permissions = HookPermissions()
            return self._hook_permissions

    async def request_initial_hook_review(
        self,
        session_id: str,
        request_id: str,
        generation: int,
        snapshot: HookReviewSnapshot,
        *,
        waiting_for_send: bool = True,
    ) -> HookReviewResult:
        """Await one resident review without transferring cancellation to its answer."""
        self._raise_if_disposed_or_session_fenced(session_id)
        controller = self.ensure_chat_controller()
        answer = controller._interrupt_host.begin_hook_review(
            session_id,
            request_id,
            generation,
            snapshot,
            waiting_for_send=waiting_for_send,
            owner=self.ensure_hook_permissions(),
            loop=asyncio.get_running_loop(),
        )
        return await asyncio.shield(answer)

    def _hook_review_presentation_current(
        self,
        review_id: str,
        generation: int,
        token: object,
    ) -> bool:
        view = self.view
        if self._disposed or view is None:
            return False
        # The owning modal suspends the Console and clears reconciliation.
        # Its current token and attachment still identify an answerable review.
        try:
            projection = view.app.screen._console_hook_review_projection
        except (AttributeError, RuntimeError):
            return False
        return bool(
            projection is not None
            and projection.review_id == review_id
            and projection.generation == generation
            and projection.presentation_token is token
            and projection.attachment_generation == self._attached_generation
        )

    @staticmethod
    async def _await_hook_review_work(task):
        """Retain the original finite producer through repeated waiter cancellation."""
        cancelled = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException:
                break
        if cancelled is not None:
            if not task.cancelled():
                task.exception()
            raise cancelled
        return task.result()

    async def _execute_hook_review_operation(self, host, operation, expected, keys):
        snapshot = None
        error = None
        producer = None
        try:
            self._raise_if_disposed_or_session_fenced(operation.session_id)
            if self.ensure_hook_permissions() is not operation.owner:
                raise RuntimeError("Hook review owner changed.")
            if expected is not operation.expected:
                raise RuntimeError("Hook review snapshot changed.")
            owner = operation.owner
            if operation.purpose == "approve":
                call = functools.partial(owner.approve, expected, keys)
            elif operation.purpose == "revoke":
                call = functools.partial(owner.revoke, expected, keys[0])
            elif operation.purpose == "disable":
                call = functools.partial(owner.disable, expected, keys[0])
            elif operation.purpose == "recover":
                call = owner.recover
            elif operation.purpose == "reset":
                call = functools.partial(owner.reset_invalid_state, expected)
            elif operation.purpose == "verify":
                call = owner.snapshot
            else:
                raise ValueError("Unknown hook review action.")
            submit_resolved = threading.Event()
            submitted = False

            def invoke():
                # A queued item may survive executor thread-start failure.
                submit_resolved.wait()
                return call() if submitted else None

            try:
                context = contextvars.copy_context()
                producer = asyncio.get_running_loop().run_in_executor(
                    None, context.run, invoke
                )
                submitted = True
            finally:
                submit_resolved.set()
            # Keep the native Future private: cancelling all Tasks must not
            # turn cancellation of a to_thread wrapper into physical retirement.
            snapshot = await self._await_hook_review_work(producer)
            return snapshot
        except BaseException as failure:
            error = failure
            raise
        finally:
            # Waiter cancellation cannot change the actual native write outcome.
            if producer is not None and producer.done() and not producer.cancelled():
                error = producer.exception()
                if error is None:
                    snapshot = producer.result()
            host.finish_hook_review_operation(operation, snapshot, error=error)

    def _start_hook_review_operation(self, host, operation, expected, keys):
        coroutine = self._execute_hook_review_operation(host, operation, expected, keys)
        try:
            # A configurable factory can eagerly issue native work, then raise
            # without returning its handle. This private driver must start lazily.
            task = asyncio.Task(coroutine, loop=asyncio.get_running_loop())
        except BaseException as error:
            coroutine.close()
            host.finish_hook_review_operation(operation, None, error=error)
            raise
        operation.task = task

        def finished(driver):
            if driver.cancelled():
                # Cancellation before the first step never enters its finally.
                # An entered driver drains native work before becoming done.
                host.finish_hook_review_operation(
                    operation, None, error=asyncio.CancelledError()
                )
            self._consume_task_outcome(driver)

        task.add_done_callback(finished)
        return task

    async def apply_hook_review_action(
        self,
        review_id: str,
        generation: int,
        action: Literal["approve", "revoke", "disable", "recover", "reset"],
        expected: HookReviewSnapshot,
        keys: tuple[str, ...] = (),
        *,
        presentation_token: object,
    ) -> HookReviewSnapshot:
        """Issue one checked consent action under the resident operation owner."""
        if action not in {"approve", "revoke", "disable", "recover", "reset"}:
            raise ValueError("Unknown hook review action.")
        if (action in {"revoke", "disable"} and len(keys) != 1) or (
            action in {"recover", "reset"} and keys
        ):
            raise ValueError("Invalid hook review selection.")
        if not self._hook_review_presentation_current(
            review_id, generation, presentation_token
        ):
            raise RuntimeError("Hook review presentation changed.")
        host = self._chat_controller._interrupt_host
        operation = host.begin_hook_review_operation(
            review_id, generation, presentation_token, action
        )
        if operation is None:
            raise RuntimeError("Hook review changed or is busy.")
        task = self._start_hook_review_operation(host, operation, expected, keys)
        return await self._await_hook_review_work(task)

    def resolve_initial_hook_review(
        self,
        review_id: str,
        generation: int,
        result: HookReviewResult,
        *,
        presentation_token: object,
    ) -> bool:
        """Treat Ready as a fresh verification intent; other answers settle exactly."""
        if not self._hook_review_presentation_current(
            review_id, generation, presentation_token
        ):
            return False
        host = self._chat_controller._interrupt_host
        if result.kind == "ready":
            operation = host.begin_hook_review_operation(
                review_id, generation, presentation_token, "verify"
            )
            if operation is None:
                return False
            self._start_hook_review_operation(host, operation, operation.expected, ())
            return True
        return host.resolve_hook_review(
            review_id, generation, result, presentation_token=presentation_token
        )

    async def _drain_hook_review_operations(self, session_id=None) -> bool:
        controller = self._chat_controller
        host = getattr(controller, "_interrupt_host", None)
        retirements = getattr(host, "hook_review_retirements", None)
        if not callable(retirements):
            return False
        cancelled = False
        while completions := retirements(session_id):
            for completion in completions:
                try:
                    await self._await_hook_review_work(completion)
                except asyncio.CancelledError:
                    if completion.cancelled():
                        raise RuntimeError(
                            "Hook review retirement signal was cancelled."
                        ) from None
                    cancelled = True
        return cancelled

    def ensure_run_hooks(self) -> RunHooksEngine | None:
        """Build one engine whose launch authority reads saved config."""
        with self._run_hooks_lock:
            if self._disposed:
                return None
            if self._run_hooks_engine is not _UNSET:
                return self._run_hooks_engine
            from tldw_chatbook.Agents.run_hooks import RunHooksEngine

            owner = self.ensure_hook_permissions()

            def cwd_provider() -> str:
                # Bound runs supply their selected workspace per fire.
                console = (getattr(self._app, "app_config", None) or {}).get("console")
                root = (
                    str(console.get("workspace_root", "") or "").strip()
                    if isinstance(console, dict)
                    else ""
                )
                return root or os.getcwd()

            self._run_hooks_engine = RunHooksEngine(
                owner.targets,
                cwd_provider,
                notification_targets=owner.notification_targets,
                launch_guard=owner.launch_guard,
            )
            return self._run_hooks_engine

    # -- the view seam -----------------------------------------------------

    def has_answerable_view(self) -> bool:
        """A reconciled attachment must also be the currently displayed screen."""
        view = self.view
        if view is None or self._reconciled_view is not view:
            return False
        try:
            return view.app.screen is view
        except Exception:  # screen-free projection doubles have no Textual app
            return True

    def _project_to_attached_view(
        self, hook_name: str, *args: Any, **kwargs: Any
    ) -> Any | None:
        """Call one disposable projection only while a view is attached."""
        view = self.view if self.has_answerable_view() else None
        provider = getattr(view, "console_view_hooks", None)
        hooks = provider() if callable(provider) else {}
        callback = hooks.get(hook_name)
        return callback(*args, **kwargs) if callable(callback) else None

    def _project_pending_decision_to_attached_view(self, projection: Any) -> bool:
        """Render exactly one mixed-type head through disposable card hooks."""
        controller = self._chat_controller
        session_id = getattr(projection, "session_id", None) or getattr(
            getattr(controller, "store", None), "active_session_id", None
        )
        set_answerable = getattr(controller, "set_answerable_decision", None)
        if session_id and callable(set_answerable):
            set_answerable(session_id, None)
            # Pausing can exhaust the exact head's active-time allowance and
            # settle it.  Never hand the pre-pause snapshot to a view (or to
            # the hidden-decision announcer); derive again after settlement.
            if projection is not None:
                derive = getattr(controller, "pending_decision_projection", None)
                if callable(derive):
                    projection = derive(session_id)
        try:
            app = self.view.app
            visible_hook_review = any(
                getattr(screen, "_console_hook_review_projection", None) is not None
                for screen in getattr(app, "screen_stack", (app.screen,))
            )
        except (AttributeError, RuntimeError):
            visible_hook_review = None
        if (
            getattr(projection, "decision_type", None) == "hook_review"
            or visible_hook_review
        ):
            from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
                project_runtime_hook_review,
            )

            hook_mounted = project_runtime_hook_review(self, projection)
            if hook_mounted is not None:
                self.recompute_console_attention()
                return hook_mounted
        view = self.view if self.has_answerable_view() else None
        provider = getattr(view, "console_view_hooks", None)
        hooks = provider() if callable(provider) else {}
        selected = getattr(projection, "decision_type", None)
        payload = getattr(projection, "payload", None)
        hook_by_type = {
            "approval": "set_pending_approval",
            "skill_install": "set_pending_skill_install",
            "skill_script": "set_pending_skill_script",
        }
        unified = hooks.get("set_pending_decision")
        if callable(unified):
            try:
                mounted = unified(projection) is True
            except Exception as exc:  # noqa: BLE001 -- projection is retryable
                mounted = False
                logger.debug(
                    "Pending-decision projection raised (exception_type={})",
                    type(exc).__name__,
                )
            finishing = isinstance(payload, Mapping) and payload.get("phase") == "finishing"
            if mounted and not finishing and session_id and callable(set_answerable):
                mounted = set_answerable(
                    session_id, getattr(projection, "decision_id", None)
                )
            if not mounted and projection is not None:
                announce = getattr(controller, "_announce_hidden_decision", None)
                if callable(announce):
                    announce(
                        getattr(projection, "decision_type", "approval"),
                        session_id or "",
                        getattr(projection, "decision_id", ""),
                    )
            self.recompute_console_attention()
            return mounted
        mounted = False
        for decision_type, hook_name in hook_by_type.items():
            callback = hooks.get(hook_name)
            if not callable(callback):
                continue
            try:
                result = callback(payload if decision_type == selected else None)
            except Exception as exc:  # noqa: BLE001 -- projection is retryable
                result = False
                logger.debug(
                    "Pending-decision card hook raised (exception_type={})",
                    type(exc).__name__,
                )
            if decision_type == selected:
                mounted = result is True
        if mounted and session_id and callable(set_answerable):
            mounted = set_answerable(
                session_id, getattr(projection, "decision_id", None)
            )
        if not mounted and projection is not None:
            announce = getattr(controller, "_announce_hidden_decision", None)
            if callable(announce):
                announce(
                    getattr(projection, "decision_type", "approval"),
                    session_id or "",
                    getattr(projection, "decision_id", ""),
                )
        self.recompute_console_attention()
        return mounted

    def _hook_target(self, kind: str) -> Any | None:
        """Resolve one slot's owning object, or `None` if unbuilt."""
        if kind == "controller":
            return self._chat_controller
        if kind == "store":
            return self._chat_store
        if kind == "wake":
            controller = self._chat_controller
            return (
                getattr(controller, "fleet_wake", None)
                if controller is not None
                else None
            )
        return None

    def _bind_view_hooks(self) -> None:
        """Point every slot in `CONSOLE_VIEW_HOOK_SLOTS` at the current view.

        Called on attach AND at the end of each `ensure_*`, because a view
        can claim the runtime before the object owning a slot exists —
        `_restore_native_console_state` reaches `ensure_chat_store` long
        before anything asks for a controller.
        """
        view = self.view
        if view is None:
            return
        workspace = getattr(view, "_workspace", None)
        reads = getattr(workspace, "_preparation_reads", None)
        if reads is not None:
            from .console_preparation_reads import observe_preparation_reads

            observe_preparation_reads(reads, self._preparation_reads)
        provider = getattr(view, "console_view_hooks", None)
        hooks = provider() if callable(provider) else {}
        for slot in CONSOLE_VIEW_HOOK_SLOTS:
            target = self._hook_target(slot.target)
            if target is None:
                continue
            setattr(target, slot.name, hooks.get(slot.name, slot.viewless_default))

    def _clear_view_hooks(self, *, only: str | None = None) -> None:
        """Restore every slot's viewless default. The mirror of the above.

        Args:
            only: Restrict to one `target` kind (`"controller"`, `"store"`,
                `"wake"`). Used by `ensure_chat_controller` to give a
                controller built with NO view claimed its viewless values
                without touching the store, whose `on_scope_flushed` is a
                CONSTRUCTOR parameter the caller just supplied.
        """
        for slot in CONSOLE_VIEW_HOOK_SLOTS:
            if only is not None and slot.target != only:
                continue
            target = self._hook_target(slot.target)
            if target is None:
                continue
            setattr(target, slot.name, slot.viewless_default)

    def _rearm_delivery_ui_hook(self) -> None:
        """Fire the just-bound `delivery_ui_hook` if a wake is mid-delivery.

        task-15860 Task 4. A wake turn entering through the coordinator is
        the ONLY turn that arms the screen's 0.2s transcript poll from
        outside the user-send worker, and it arms it exactly once, in
        `_attempt`, at delivery start. With a runtime that survives the
        screen, delivery start and view attach are now independent events:
        a wake can begin with nothing attached (no repaint target — inert
        and correct) and the user can open Console *during* it. Without
        this re-arm that Console shows a frozen transcript for the rest of
        the turn — the live 4+ minute freeze PR 3a-2 Task 7 measured, which
        is what makes a missing re-arm the expensive half of this slot.

        Best-effort in both directions: no delivery in flight arms nothing
        (a poll with nothing to repaint is the recurring-idle-repaint
        regression 15664 AC#2 forbids), and a raising hook is logged, never
        propagated into the attach.

        Deliberately NOT gated on "the view actually changed": `attach_view`
        runs on every `_ensure_console_chat_controller()` call, and the
        production hook is idempotent (`_start_console_transcript_sync_
        timer` early-returns when a timer already exists), so an extra
        re-arm costs one pump hop while a MISSED one costs the freeze.
        """
        wake = self._hook_target("wake")
        if wake is None:
            return
        reader = getattr(wake, "delivering_session_ids", None)
        session_ids = reader() if callable(reader) else ()
        if not session_ids:
            return
        hook = getattr(wake, "delivery_ui_hook", None)
        if not callable(hook):
            return
        for session_id in session_ids:
            try:
                hook(session_id)
            except Exception as exc:  # noqa: BLE001 -- UI freshness is best-effort
                logger.debug(
                    "wake delivery UI hook re-arm raised (exception_type={})",
                    type(exc).__name__,
                )

    @property
    def worktree_recovery(self) -> ConsoleWorktreeRecovery:
        """Retain manual recovery independently of disposable Console views."""
        if self._worktree_recovery is None:
            if (
                self._disposed
                or self._chat_controller is None
                or self._agent_bridge is None
            ):
                raise RuntimeError("Console agent recovery is unavailable.")
            from .console_worktree_recovery import ConsoleWorktreeRecovery

            self._worktree_recovery = ConsoleWorktreeRecovery(
                self._chat_controller, self._agent_bridge
            )
        return self._worktree_recovery

    def remount_pending_approval(self) -> None:
        """Re-derive decision cards for rounds armed while viewless.

        task-15860 Task 5. The screen's approval card is derived entirely
        from its own `_task_resume_state`, which a FRESH screen starts
        empty — and screens are never cached. So a round armed headlessly
        (a risk-tagged tool in a wake turn) would sit registered,
        announced app-wide, and still invisible the moment the user acted
        on that announcement and opened Console. `switch_session`'s
        identical re-derive would eventually mount it, but only if the
        user switched sessions — which they have no reason to do, never
        having seen a card.

        Gated on successful full reconciliation, not run on every
        `_ensure_console_chat_controller()`: re-pushing the payload rebuilds
        the card's rows, so an unconditional re-derive would discard a
        half-made decision (a chosen Select, not yet submitted) on any tick
        that happens to touch the controller.

        MCP, skill-install, and skill-script payloads remain controller-owned;
        this method only asks their existing registries for the active
        session's head. Best-effort: a raising seam is logged, never
        propagated into the attach.
        """
        controller = self._chat_controller
        if controller is None:
            return
        store = getattr(controller, "store", None) or self._chat_store
        active_session_id = getattr(store, "active_session_id", None)
        remount_worktree = getattr(controller, "_remount_parked_worktree_merge", None)
        if active_session_id and callable(remount_worktree):
            try:
                remount_worktree(active_session_id)
            except Exception as exc:  # noqa: BLE001 - a disposable view cannot abort attach
                logger.debug(
                    "Worktree remount failed (exception_type={})", type(exc).__name__
                )
        projection_for = getattr(controller, "pending_decision_projection", None)
        if callable(projection_for) and (
            not active_session_id or projection_for(active_session_id) is None
        ):
            return
        project = getattr(
            controller, "project_pending_decision_for_active_session", None
        )
        if callable(project) and getattr(controller, "set_pending_decision", None):
            try:
                project()
            except Exception as exc:  # noqa: BLE001 -- attach never dies on this
                logger.debug(
                    "Pending-decision projection raised at attach "
                    "(exception_type={})",
                    type(exc).__name__,
                )
            return
        for method_name in (
            "remount_pending_approval_for_active_session",
            "_remount_parked_skill_install",
            "_remount_parked_skill_script",
        ):
            remount = getattr(controller, method_name, None)
            if not callable(remount):
                continue
            try:
                if method_name == "remount_pending_approval_for_active_session":
                    remount()
                else:
                    if active_session_id:
                        remount(active_session_id)
            except Exception as exc:  # noqa: BLE001 -- attach never dies on this
                logger.debug(
                    "Pending-decision remount raised at attach "
                    "(exception_type={})",
                    type(exc).__name__,
                )

    def finish_view_reconciliation(self, view: Any, generation: int | None) -> bool:
        """Project decisions only after this exact view completed a full sync."""
        if self.view is not view or generation != self._attached_generation:
            return False
        self._reconciled_view = view
        self.remount_pending_approval()
        self._remount_task_panel()
        controller = self._chat_controller
        store = self._chat_store
        if controller is not None and store is not None and store.active_session_id:
            remount = getattr(controller, "_remount_session_kinds", None)
            if callable(remount):
                remount(store.active_session_id)
        self._rearm_delivery_ui_hook()
        self._project_pending_project_instruction_decisions()
        return True

    def attach_view(
        self,
        view: Any,
        *,
        prior_generation: int | None | object = _ATTACH_WITHOUT_PRIOR_CLAIM,
    ) -> int | None:
        """Claim this runtime for `view` and return its monotonic generation.

        **This replaces Task 1's `ConsoleRuntime.view` stand-in.** That
        device kept a runtime claimed by a different view from being shared
        — it simply built a second runtime, which was only ever a way of
        reproducing dispose-at-unmount semantics. Now there is one runtime
        and the claim is real: a fresh view can claim it, while a view that
        presents a superseded prior token cannot reclaim it. A superseded
        view's `detach_view` is also a no-op (see there).

        Every fresh claim receives a new token; refreshing the exact current
        claim keeps its token. Detach must present both this exact view and
        token, so a late unmount can never clear a successor's projections.
        """
        with self._attention_operation_lock:
            previous = self.view
            if prior_generation is not _ATTACH_WITHOUT_PRIOR_CLAIM:
                stale_claim = prior_generation is not None and (
                    previous is not view
                    or prior_generation != self._attached_generation
                )
                if stale_claim:
                    return None
            if previous is view and prior_generation == self._attached_generation:
                self._bind_view_hooks()
                return self._attached_generation
            if previous is not view:
                if (
                    self._chat_controller is not None
                    and self._attached_generation is not None
                ):
                    self._chat_controller._interrupt_host.release_hook_review_attachment(
                        self._attached_generation
                    )
                self._pause_project_instruction_generation(self._attached_generation)
            self._attachment_generation += 1
            generation = self._attachment_generation
            self.view = view
            self._rendered_receipt_ack_owner = None
            self._rendered_receipt_acks.clear()
            self._attached_generation = generation
            if previous is not view:
                self._reconciled_view = None
            try:
                setattr(view, "_console_runtime_attachment_generation", generation)
            except Exception:  # noqa: BLE001 -- read-only view doubles are supported
                pass
            self._bind_view_hooks()
            return generation

    def _remount_task_panel(self) -> None:
        """Push the active session's tasks into a newly claimed view's panel.

        PRD Feature B: the runtime and its todo stores outlive the screen,
        but the panel widget is screen-owned and mounts empty. Without this
        a return visit to Console shows no tasks until the next ``todo_*``
        change or session switch.
        """
        controller = self._chat_controller
        store = self._chat_store
        remount = getattr(controller, "_remount_task_panel", None)
        if store is not None and callable(remount):
            remount(store.active_session_id)

    def detach_view(
        self,
        view: Any | None = None,
        generation: int | None = None,
    ) -> bool:
        """Clear every screen-owned slot; the runtime itself survives.

        Args:
            view: The view detaching. When another view has already
                claimed this runtime — the overlapping window where
                `_complete_screen_navigation` has constructed and
                `restore_state`d the INCOMING screen before `switch_screen`
                unmounts the outgoing one — this is a **no-op**: a
                superseded screen may not clear a hook its successor just
                bound. `None` detaches unconditionally (app exit).

        Returns:
            True when the detach actually ran.
        """
        with self._attention_operation_lock:
            if view is not None:
                if self.view is not view or generation != self._attached_generation:
                    return False
            controller = self._chat_controller
            session_id = getattr(
                getattr(controller, "store", None), "active_session_id", None
            )
            pause = getattr(controller, "set_answerable_decision", None)
            if session_id and callable(pause):
                pause(session_id, None)
            self._pause_project_instruction_generation(self._attached_generation)
            if controller is not None and self._attached_generation is not None:
                controller._interrupt_host.release_hook_review_attachment(
                    self._attached_generation
                )
            self._clear_view_hooks()
            self.view = None
            self._attached_generation = None
            self._rendered_receipt_ack_owner = None
            self._rendered_receipt_acks.clear()
            self._reconciled_view = None
            self.recompute_console_attention()
            return True

    # -- teardown ----------------------------------------------------------

    async def leave_console(
        self,
        view: Any | None = None,
        generation: int | None = None,
    ) -> bool:
        """Detach one Console projection without changing domain work.

        Args:
            view: The unmounting view. A superseded view leaves nothing
                (`detach_view`'s no-op), because the successor is still
                using this runtime's turns.

        Returns:
            True when this visit was actually ended.
        """
        return self.detach_view(view, generation)

    @staticmethod
    def _consume_task_outcome(task: asyncio.Future[Any]) -> None:
        """Retrieve one terminal task outcome without exposing content."""

        if task.cancelled() or not task.done():
            return
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            return

    async def _bounded_wait(
        self,
        tasks: set[asyncio.Future[Any]],
        *,
        timeout_seconds: float,
    ) -> set[asyncio.Future[Any]]:
        """Wait once under a shared deadline and retrieve completed failures."""

        if not tasks:
            return set()
        done, pending = await asyncio.wait(
            tasks,
            timeout=max(0.0, float(timeout_seconds)),
        )
        for task in done:
            self._consume_task_outcome(task)
        return set(pending)

    async def close_session(
        self,
        session_id: str,
        *,
        expected_revision: int,
        timeout_seconds: float = CONSOLE_SESSION_CLOSE_GRACE_SECONDS,
    ) -> Any | None:
        """Drain already-claimed voice publication before closing its session.

        Args:
            session_id: Exact Console session to close.
            expected_revision: Revision from the caller's lifecycle impact snapshot.
            timeout_seconds: Grace period for bounded publication and turn drains.

        Returns:
            The removed session, or None if claimed voice publication does not
            drain within its grace period.

        Raises:
            RuntimeError: Recovery retains a session admission fence, a voice
                close is already active, or the runtime/controller cannot
                accept closure.
        """

        # A recreated store row cannot retire this app-lifetime close fence.
        # Refuse before taking any new voice-close ownership.
        if session_id in self._admission_fenced_sessions:
            raise RuntimeError(CONSOLE_SESSION_CLOSE_RECOVERY_REFUSAL)
        owner = self._voice_promotion_owner
        if owner is None:
            if session_id in self._voice_promotion_pending_closes:
                raise RuntimeError("A voice-promotion session close is already active.")
            self._voice_promotion_pending_closes[session_id] = None
            try:
                return await self._close_session_after_voice_drain(
                    session_id,
                    expected_revision=expected_revision,
                    timeout_seconds=timeout_seconds,
                )
            finally:
                token = self._voice_promotion_pending_closes.pop(session_id)
                if token is not None:
                    self._voice_promotion_owner.abort_session_close(token)
        token = owner.begin_session_close(session_id)
        completed = False
        try:
            self._raise_if_disposed_or_session_fenced(session_id)
            if not await owner.wait_for_session(session_id, timeout_seconds):
                return None
            result = await self._close_session_after_voice_drain(
                session_id,
                expected_revision=expected_revision,
                timeout_seconds=timeout_seconds,
            )
            owner.complete_session_close(token)
            completed = True
            return result
        finally:
            if not completed:
                owner.abort_session_close(token)

    async def _drain_ordinary_native_commits(self, controller, session_id=None) -> bool:
        """Keep only exact native save lifetimes past the surrounding grace."""
        read = getattr(controller, "_ordinary_native_commit_retirements", None)
        if not callable(read):
            return False
        tasks = getattr(controller, "_ordinary_native_commit_tasks", None)
        if callable(tasks) and asyncio.current_task() in tasks(session_id):
            raise RuntimeError("An ordinary save cannot finalize its own runtime.")
        cancelled = False
        while True:
            completions = read(session_id)
            if not completions:
                return cancelled
            for completion in completions:
                while not completion.done():
                    try:
                        await asyncio.shield(completion)
                    except asyncio.CancelledError:
                        task = asyncio.current_task()
                        if task is not None and task.cancelling():
                            cancelled = True
                        elif completion.cancelled():
                            raise RuntimeError(
                                "Native save retirement signal was cancelled."
                            ) from None
                completion.result()

    async def _close_session_after_voice_drain(
        self,
        session_id: str,
        *,
        expected_revision: int,
        timeout_seconds: float = CONSOLE_SESSION_CLOSE_GRACE_SECONDS,
    ) -> Any | None:
        """Fence, cancel, bounded-drain, then delete one Console session."""

        self._raise_if_disposed_or_session_fenced(session_id)
        controller = self._chat_controller
        if controller is None:
            raise RuntimeError("Console controller is unavailable.")
        drain_handoff = getattr(controller.store, "drain_agent_handoff", None)
        if callable(drain_handoff) and not await drain_handoff(session_id):
            raise RuntimeError(
                "Draft custody must be confirmed before closing this chat."
            )
        self._admission_fenced_sessions.add(session_id)
        try:
            ticket = controller.begin_session_close(
                session_id,
                expected_revision=expected_revision,
            )
        except BaseException:
            self._admission_fenced_sessions.discard(session_id)
            raise

        self._seal_hooks_v2(session_id)
        if self._worktree_recovery is not None:
            self._worktree_recovery.cancel_session(session_id)

        # The exact close ticket makes this owner's fence irreversible. Keep
        # recoveries on a refused/provisional close, but not through its drain.
        for turn_id in tuple(self._recovery_turns_by_session.get(session_id, ())):
            self.discard_turn_recovery(turn_id)

        tasks: set[asyncio.Future[Any]] = {
            record.task
            for record in self._turn_custody.values()
            if record.session_id == session_id and record.task is not None
        }
        snapshot_tasks = getattr(controller, "session_shutdown_tasks", None)
        if callable(snapshot_tasks):
            tasks.update(snapshot_tasks(session_id))

        bridge = self._agent_bridge
        await_progress = getattr(bridge, "await_progress_closed", None)
        if callable(await_progress):
            tasks.add(
                asyncio.create_task(
                    await_progress(session_id),
                    name=f"console-close-progress-{ticket.close_id[:8]}",
                )
            )
        await_fleet = getattr(bridge, "await_fleet_terminal", None)
        fleet_waiters: list[asyncio.Task[Any]] = []
        if callable(await_fleet):
            for conversation_id in dict.fromkeys((session_id, ticket.conversation_id)):
                waiter = asyncio.create_task(
                    await_fleet(conversation_id),
                    name=f"console-close-fleet-{ticket.close_id[:8]}",
                )
                fleet_waiters.append(waiter)
                tasks.add(waiter)

        # Once the controller issues a close ticket the operation is
        # irreversible: queue/wake/fleet admission is already terminally
        # fenced. Keep the bounded drain alive if the UI worker that invoked
        # close is cancelled (for example by navigation), finalize deletion,
        # and only then re-deliver cancellation to that caller.
        drain_task = asyncio.create_task(
            self._bounded_wait(tasks, timeout_seconds=timeout_seconds),
            name=f"console-close-drain-{ticket.close_id[:8]}",
        )
        cancel_requested = False
        while True:
            try:
                pending = await asyncio.shield(drain_task)
                break
            except asyncio.CancelledError:
                cancel_requested = True
        for task in pending:
            task.cancel()
            task.add_done_callback(self._consume_task_outcome)
        cancel_requested |= await self._drain_ordinary_native_commits(
            controller, session_id
        )
        cancel_requested |= await self._drain_hook_review_operations(session_id)
        cancel_requested |= await self._drain_hook_preparation_reads(session_id)
        pending = {task for task in pending if not task.done()}
        fleet_fenced = callable(getattr(bridge, "fence_fleet", None))
        fleet_drain_succeeded = not fleet_fenced and not fleet_waiters
        if fleet_waiters:
            try:
                fleet_drain_succeeded = all(
                    waiter.done() and not waiter.cancelled() and bool(waiter.result())
                    for waiter in fleet_waiters
                )
            except Exception:  # noqa: BLE001 -- unknown drain stays fenced
                fleet_drain_succeeded = False
        for turn_id, record in tuple(self._turn_custody.items()):
            if record.session_id == session_id:
                self._release_custody(turn_id)
        hook_engine = self._hooks_v2_engines.get(session_id)
        hook_drain = asyncio.create_task(self.close_hooks_v2(session_id))
        while True:
            try:
                await asyncio.shield(hook_drain)
                break
            except asyncio.CancelledError:
                cancel_requested = True
        closed = controller.finalize_session_close(ticket)
        hook_drain_succeeded = hook_engine is None or not hook_engine.cleanup_pending
        if not pending and fleet_drain_succeeded and hook_drain_succeeded:
            # The fence was provisional while this exact session scope
            # drained. With every task and delegated child terminal, no stale
            # producer remains, so a later resume of the saved conversation
            # may create a fresh fleet. Timeout keeps ``pending`` non-empty and
            # intentionally leaves this generation latched for the process.
            release_fences = getattr(controller, "release_session_close_fences", None)
            if callable(release_fences):
                release_fences(ticket)
        if cancel_requested:
            raise asyncio.CancelledError
        return closed

    def begin_dispose(
        self,
        *,
        expected_revision: int | None = None,
        voice_promotion_permit: VoicePromotionQuitPermit | None = None,
    ) -> None:
        """Synchronously revision-check and fence all new Console work."""

        controller = self._chat_controller
        sessions = (
            tuple(getattr(self._chat_store, "sessions", lambda: ())())
            if self._chat_store is not None
            else ()
        )
        session_ids = {str(session.id) for session in sessions}
        conversation_id_for_session = getattr(
            controller,
            "conversation_id_for_session",
            None,
        )
        fence_fleet = getattr(self._agent_bridge, "fence_fleet", None)
        abort_fleet_fence = getattr(
            self._agent_bridge,
            "abort_fleet_fence",
            None,
        )
        provisional_generation = self.generation + 1
        acquired_fences: list[str] = []

        def abort_provisional_fences() -> None:
            if not callable(abort_fleet_fence):
                return
            for conversation_id in reversed(acquired_fences):
                try:
                    abort_fleet_fence(
                        conversation_id,
                        generation=provisional_generation,
                    )
                except Exception:
                    logger.warning(
                        "Console runtime: provisional fleet fence could not abort."
                    )

        try:
            if callable(fence_fleet):
                for session_id in session_ids:
                    conversation_id = session_id
                    if callable(conversation_id_for_session):
                        try:
                            conversation_id = conversation_id_for_session(session_id)
                        except Exception:
                            pass
                    for causal_id in dict.fromkeys((session_id, conversation_id)):
                        if fence_fleet(causal_id, generation=provisional_generation):
                            acquired_fences.append(causal_id)
            # Reservation observers publish before the fleet fence can take
            # the coordinator lock. Rechecking here closes the dialog-to-
            # shutdown race while the provisional fence prevents another
            # child admission from appearing behind this snapshot.
            if expected_revision is not None and controller is not None:
                impact = controller.lifecycle_impact()
                if impact.revision != expected_revision:
                    raise ConsoleLifecycleRevisionChanged(
                        "Console activity changed during shutdown."
                    )
            if self._voice_promotion_owner is not None:
                if voice_promotion_permit is None:
                    raise RuntimeError("Voice-promotion quit permit is required.")
                self._voice_promotion_owner.consume_quit_permit(voice_promotion_permit)
        except BaseException:
            abort_provisional_fences()
            raise
        with self._execution_capacity_lock:
            with self._canvas_native_lock:
                self._disposed = True
        self._seal_hooks_v2()
        with self._run_hooks_lock:
            engine = self.run_hooks_engine
            if self._hook_permissions is not None:
                self._hook_permissions.close()
            if engine is not None:
                engine.close()
        hook_host = getattr(self._chat_controller, "_interrupt_host", None)
        cancel_hook_reviews = getattr(hook_host, "cancel_hook_reviews", None)
        if callable(cancel_hook_reviews):
            cancel_hook_reviews()
        if self._worktree_recovery is not None:
            self._worktree_recovery.begin_close()
        if self._voice_process_supervisor is not None:
            self._voice_process_supervisor.begin_close()
        self._admission_fenced_sessions.update(session_ids)
        begin_progress_close = getattr(
            self._agent_bridge, "begin_close_all_progress", None
        )
        if callable(begin_progress_close):
            begin_progress_close()
        for turn_id in tuple(self._turn_recoveries):
            self.discard_turn_recovery(turn_id)
        begin_shutdown = getattr(controller, "begin_shutdown", None)
        if callable(begin_shutdown):
            try:
                begin_shutdown()
            except Exception:  # noqa: BLE001 -- terminal fence cannot roll back
                logger.warning(
                    "Console runtime: controller teardown failed after the "
                    "terminal shutdown fence."
                )

    async def dispose(
        self,
        *,
        timeout_seconds: float = CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS,
    ) -> None:
        """Destroy the runtime. The permanent, app-exit form.

        Keeps the pre-15860 `ChatScreen.on_unmount` order exactly:
        `await controller.shutdown()` (which tombstones the queue, sets the
        cancellation Event permanently, and cancels/awaits EVERY session's
        stream task) and then `await gateway.aclose()`. Reached from
        `TldwCli._shutdown_app_owned_lifecycles`.

        **The built objects are NOT dropped, and `_disposed` latches.**
        `_shutdown_app_owned_lifecycles` runs BEFORE Textual closes screen
        state, so a Console screen -- and its timers -- can still be live
        while this runs, and there are ~75 `_ensure_console_chat_*` call
        sites reachable from those. Dropping the references would let one
        of them BUILD A FRESH CONTROLLER during quit, which nothing would
        ever shut down; returning `None` instead would crash a tick that
        has never had to handle it. Keeping the torn-down objects is the
        only option that does neither: a shut-down controller already
        refuses work through its permanently-set cancellation Event, which
        is exactly the right answer at exit.
        """
        from .console_fleet_wake import ConsoleFleetWakeCoordinator

        controller = self._chat_controller
        recovery_owner = getattr(controller, "fleet_wake", None)
        if not isinstance(recovery_owner, ConsoleFleetWakeCoordinator):
            recovery_owner = None
        if recovery_owner is not None:
            recovery_owner.dispose()
        try:
            await self._dispose_owned(
                timeout_seconds=timeout_seconds, recovery_owner=recovery_owner
            )
        finally:
            cancelled = False
            if recovery_owner is not None:
                cancelled |= await recovery_owner.drain_recovery()
            if controller is not None:
                await self._drain_ordinary_native_commits(controller)
                await self._drain_hook_review_operations()
            cancelled |= await self._drain_hook_preparation_reads()
            cancelled |= await self._drain_activity_hydration()
            if cancelled:
                raise asyncio.CancelledError

    async def _dispose_owned(
        self,
        *,
        timeout_seconds: float,
        recovery_owner=None,
    ) -> None:
        """Run the existing teardown once under its original shared deadline."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(timeout_seconds))

        # The hook owner survives a cancelled dispose caller. Teardown producers
        # have the fixed notification window before final queue closure.
        self._seal_hooks_v2()
        if self._hooks_v2_cleanup_task is None and self._hooks_v2_engines:

            async def hook_shutdown() -> None:
                engines = tuple(self._hooks_v2_engines.values())
                # Controller teardown runs concurrently. Admission stays sealed;
                # only Interrupt/SessionEnd can enter this remaining window.
                await asyncio.sleep(
                    max(
                        0.0,
                        max(e.teardown_deadline for e in engines) - time.monotonic(),
                    )
                )
                await asyncio.gather(*(e.close() for e in engines))

            self._hooks_v2_cleanup_task = asyncio.create_task(
                hook_shutdown(), name="console-v2-hook-cleanup"
            )

        def remaining_seconds() -> float:
            return max(0.0, deadline - loop.time())

        # Capacity allocation precedes the existing Canvas/receipt publication
        # lock; no Canvas critical section acquires capacity admission.
        with self._execution_capacity_lock:
            with self._canvas_native_lock:
                self._disposed = True
                self._canvas_native_view_binding = None
        self._seal_hooks_v2()
        begin_progress_close = getattr(
            self._agent_bridge, "begin_close_all_progress", None
        )
        if callable(begin_progress_close):
            begin_progress_close()
        with self._run_hooks_lock:
            engine = self.run_hooks_engine
            if self._hook_permissions is not None:
                self._hook_permissions.close()
            if engine is not None:
                engine.close()
        hook_host = getattr(self._chat_controller, "_interrupt_host", None)
        cancel_hook_reviews = getattr(hook_host, "cancel_hook_reviews", None)
        if callable(cancel_hook_reviews):
            cancel_hook_reviews()
        if self._voice_process_supervisor is not None:
            self._voice_process_supervisor.begin_close()
        if self._worktree_recovery is not None:
            await self._worktree_recovery.close()
        for turn_id in tuple(self._turn_recoveries):
            self.discard_turn_recovery(turn_id)
        canvas_policy_watch_task = self._canvas_policy_watch_task
        self._canvas_policy_watch_task = None
        if canvas_policy_watch_task is not None and not canvas_policy_watch_task.done():
            canvas_policy_watch_task.cancel()
            try:
                await canvas_policy_watch_task
            except asyncio.CancelledError:
                pass
        if self._canvas_policy_cleanups:
            await asyncio.shield(asyncio.gather(*self._canvas_policy_cleanups))
        if self._canvas_policy_read_task is not None:
            await asyncio.shield(self._canvas_policy_read_task)
            self._canvas_policy_read_task = None
        maintenance_task = self._legacy_trace_maintenance_task
        if maintenance_task is not None and not maintenance_task.done():
            maintenance_task.cancel()
            try:
                await maintenance_task
            except asyncio.CancelledError:
                pass
        self._raw_cli_refusal_stash_bank.clear()
        with self._project_decision_lock:
            binding_decisions = tuple(self._project_binding_decisions.values())
            dispatch_decisions = tuple(self._project_dispatch_decisions.values())
        for decision in binding_decisions:
            self.resolve_project_instruction_binding(
                decision.decision_id, "cancel", None
            )
        for decision in dispatch_decisions:
            self.resolve_project_instruction_dispatch(decision.decision_id, "cancel")
        self._scratch_spaces.tombstone_all()
        self._begin_progress_cleanup(self._agent_bridge)
        with self._execution_capacity_lock:
            if self._execution_capacity is not None:
                self._execution_capacity.close()
        self.detach_view(None)
        controller, gateway = self._chat_controller, self._provider_gateway
        canvas_gateway = self._canvas_gateway
        canvas_authority = self._canvas_native_authority
        coordinator = self._change_review_coordinator

        def receipt_database_after_creation() -> Any:
            # An inbox may still be initializing storage on a worker. Wait away
            # from the UI loop; the disposed latch prevents late publication.
            with self._activity_receipts_lock:
                return self._agent_runs_db

        runs_db = await asyncio.to_thread(receipt_database_after_creation)
        self.generation += 1
        hydration_task = self._activity_hydration_task
        if hydration_task is not None and not hydration_task.done():
            hydration_task.cancel()
        # Revoke browser admission before tearing down any controller/store
        # authority that gateway callbacks could otherwise reach.
        close_canvas_gateway = getattr(canvas_gateway, "aclose", None)
        if callable(close_canvas_gateway):
            try:
                result = close_canvas_gateway()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - app exit must keep progressing
                logger.warning(
                    "Console runtime: Canvas gateway close failed at dispose."
                )
        dispose_canvas_authority = getattr(canvas_authority, "dispose", None)
        if callable(dispose_canvas_authority):
            try:
                result = dispose_canvas_authority()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - app exit must keep progressing
                logger.warning(
                    "Console runtime: Canvas authority dispose failed at exit."
                )
        sessions = tuple(
            getattr(self._chat_store, "sessions", lambda: ())()
            if self._chat_store is not None
            else ()
        )
        session_ids = {str(session.id) for session in sessions}
        self._admission_fenced_sessions.update(session_ids)
        conversation_ids: set[str] = set(session_ids)
        conversation_id_for_session = getattr(
            controller,
            "conversation_id_for_session",
            None,
        )
        for session_id in session_ids:
            if callable(conversation_id_for_session):
                try:
                    conversation_ids.add(conversation_id_for_session(session_id))
                except Exception:
                    conversation_ids.add(session_id)
            else:
                conversation_ids.add(session_id)

        begin_shutdown = getattr(controller, "begin_shutdown", None)
        if callable(begin_shutdown):
            try:
                begin_shutdown()
            except Exception:
                logger.warning("Console runtime: shutdown fence failed at dispose.")

        progress_cleanups = set(self._progress_cleanup_tasks)
        drain_tasks: set[asyncio.Future[Any]] = set(progress_cleanups)
        voice_cleanup = None
        if self._voice_worker is not None or self._voice_process_supervisor is not None:

            async def close_voice_worker():
                try:
                    if self._voice_process_supervisor is not None:
                        await self._voice_process_supervisor.aclose()
                    if self._voice_dispatch_supervisor is not None:
                        await self._voice_dispatch_supervisor.wait_for_cleanup()
                    if self._voice_worker is not None:
                        await self._voice_worker.aclose()
                except Exception as exc:
                    logger.warning(
                        "Console runtime: voice cleanup failed category={}",
                        type(exc).__name__,
                    )

            voice_cleanup = asyncio.create_task(
                close_voice_worker(), name="console-dispose-voice"
            )
            drain_tasks.add(voice_cleanup)
        for record in tuple(self._turn_custody.values()):
            task = record.task
            if task is None:
                continue
            task.cancel()
            drain_tasks.add(task)

        bridge = self._agent_bridge
        fence_fleet = getattr(bridge, "fence_fleet", None)
        cancel_all = getattr(bridge, "cancel_all_subagents", None)
        await_fleet = getattr(bridge, "await_fleet_terminal", None)
        for conversation_id in conversation_ids:
            if callable(fence_fleet):
                try:
                    fence_fleet(conversation_id, generation=self.generation)
                except Exception:
                    logger.warning("Console runtime: fleet fence failed at dispose.")
            if callable(cancel_all):
                try:
                    cancel_all(conversation_id)
                except Exception:
                    logger.warning("Console runtime: fleet cancel failed at dispose.")
            if callable(await_fleet):
                drain_tasks.add(
                    asyncio.create_task(
                        await_fleet(conversation_id),
                        name="console-dispose-fleet",
                    )
                )
        if controller is not None:
            shutdown = getattr(controller, "shutdown", None)
            if callable(shutdown):
                drain_tasks.add(
                    asyncio.create_task(
                        shutdown(),
                        name="console-dispose-controller",
                    )
                )
        drain = asyncio.create_task(
            self._bounded_wait(drain_tasks, timeout_seconds=remaining_seconds()),
            name="console-dispose-drain",
        )
        cancel_requested = False
        while True:
            try:
                pending = await asyncio.shield(drain)
                break
            except asyncio.CancelledError:
                cancel_requested = True
        if voice_cleanup in pending:
            logger.warning("Console runtime: voice cleanup remains pending at dispose.")
        for task in pending:
            if task is voice_cleanup or task in progress_cleanups:
                continue
            task.cancel()
            task.add_done_callback(self._consume_task_outcome)
        if voice_cleanup is not None:
            # The child budget cannot end UI-owned provider/TTS/claimed work.
            # Keep its original owner loop and store until actual custody settles.
            await asyncio.shield(voice_cleanup)
        # An admitted SQL leaf outlives the grace budget; keep its exact owner
        # until it physically releases the lock before downstream disposal.
        for task in progress_cleanups:
            while True:
                try:
                    await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    cancel_requested = True
        if controller is not None:
            await self._drain_ordinary_native_commits(controller)
            await self._drain_hook_review_operations()
        cancel_requested |= await self._drain_hook_preparation_reads()
        cancel_requested |= await self._drain_activity_hydration()
        if recovery_owner is not None:
            cancel_requested |= await recovery_owner.drain_recovery()
        await self.close_hooks_v2()
        for turn_id in tuple(self._turn_custody):
            self._release_custody(turn_id)
        if self._chat_store is not None:
            end_app_runtime = getattr(self._chat_store, "end_app_runtime", None)
            if callable(end_app_runtime):
                try:
                    from .console_durable_writes import end_store_after_writes

                    # A Delete/Undo saving off the loop holds the store's
                    # voice admission until applied: settle it first.
                    await end_store_after_writes(
                        self._chat_store,
                        end_app_runtime,
                        CONSOLE_DURABLE_WRITE_TEARDOWN_SECONDS,
                    )
                except Exception:  # noqa: BLE001 - quit must continue cleanup
                    logger.opt(exception=True).warning(
                        "Console runtime: trace settlement shutdown failed at dispose."
                    )
        # Controller shutdown begins by terminally fencing the fleet-wake
        # coordinator. Only after every trusted producer is tombstoned may
        # the shared Buddy sink release its remaining owner tokens.
        self._persona_buddy_sink.dispose()
        if coordinator is not None:
            try:
                await asyncio.to_thread(coordinator.shutdown, 2.0)
            except Exception:  # noqa: BLE001 - quit must not die on teardown
                logger.opt(exception=True).warning(
                    "Console runtime: Change Review shutdown failed at dispose."
                )
        # AgentRunsDB connections are thread-local. The coordinator closes
        # the publisher thread's connection after its last callback; this
        # closes the runtime/UI thread's separate held connection. Even when
        # bounded coordinator shutdown times out, neither close invalidates
        # the other thread's connection.
        if runs_db is not None:
            try:
                runs_db.close()
            except Exception:  # noqa: BLE001 - same
                logger.opt(exception=True).warning(
                    "Console runtime: AgentRunsDB close failed at dispose."
                )

        async def dispose_scratch() -> None:
            try:
                await asyncio.to_thread(self._scratch_spaces.dispose)
            except Exception as exc:  # noqa: BLE001 - quit must continue
                logger.warning(
                    "Console runtime: scratch cleanup failed at dispose category={}",
                    type(exc).__name__,
                )

        async def close_gateway() -> None:
            close = getattr(gateway, "aclose", None)
            if not callable(close):
                return
            try:
                if inspect.iscoroutinefunction(close):
                    await close()
                    return
                result = await asyncio.to_thread(close)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - quit must continue
                logger.opt(exception=True).warning(
                    "Console runtime: provider gateway close failed at dispose."
                )
        try:
            totals = self.trace_compatibility_snapshot()
            logger.info(
                "Console trace compatibility totals: normalized_write={} "
                "normalized_read={} legacy_read={} fallback_read={} incomplete={}",
                totals.get("normalized_write", 0),
                totals.get("normalized_read", 0),
                totals.get("legacy_read", 0),
                totals.get("fallback_read", 0),
                totals.get("incomplete", 0),
            )
        except Exception as exc:  # noqa: BLE001 - shutdown metrics are best effort
            logger.warning(
                "Console trace compatibility totals unavailable: error_type={}",
                type(exc).__name__,
            )

        cleanup_tasks: set[asyncio.Future[Any]] = {
            asyncio.create_task(
                dispose_scratch(),
                name="console-dispose-scratch",
            )
        }
        if callable(getattr(gateway, "aclose", None)):
            cleanup_tasks.add(
                asyncio.create_task(
                    close_gateway(),
                    name="console-dispose-gateway",
                )
            )
        cleanup_pending = await self._bounded_wait(
            cleanup_tasks,
            timeout_seconds=remaining_seconds(),
        )
        for task in cleanup_pending:
            task.cancel()
            task.add_done_callback(self._consume_task_outcome)
        if cancel_requested or asyncio.current_task().cancelling():
            raise asyncio.CancelledError


def _attach(app: Any, runtime: ConsoleRuntime | None) -> None:
    """Write `runtime` (or `None`) onto the app's runtime attribute.

    A failed or rewritten attach never raises -- read-only app doubles in
    tests are an expected, tolerated case -- but it is recorded loudly:
    a silently unattached runtime breaks the ownership invariant
    ``ensure_console_runtime`` exists to guarantee (every later call would
    miss the attribute and construct a duplicate runtime).
    """
    if app is None:
        return
    try:
        setattr(app, CONSOLE_RUNTIME_ATTR, runtime)
    except Exception:  # noqa: BLE001 - a read-only app double is not an error
        # Metadata-only by design (TASK-15743 audit table): no
        # opt(exception=True) here -- the except is an expected read-only
        # app double, and the attribute name plus consequence is the whole
        # diagnosis. `test_task_15743_final_rebase_diagnostics_are_metadata_
        # only` pins this shape.
        logger.warning(
            "Console runtime: could not write the app's %s attribute; the "
            "runtime stays detached and future Console visits will each "
            "build their own.",
            CONSOLE_RUNTIME_ATTR,
        )
        return
    # Read back inside its own guard: a custom __getattribute__ that raises
    # must not break _attach's never-raise contract either.
    try:
        attached = getattr(app, CONSOLE_RUNTIME_ATTR, None)
    except Exception:  # noqa: BLE001 - never raise from the post-check
        logger.opt(exception=True).warning(
            "Console runtime: could not read back the app's %s attribute "
            "to confirm the attach.",
            CONSOLE_RUNTIME_ATTR,
        )
        return
    if attached is not runtime:
        logger.warning(
            "Console runtime: the app's %s attribute is not the runtime "
            "just attached (a property or validator rewrote it); future "
            "Console visits will each build their own.",
            CONSOLE_RUNTIME_ATTR,
        )


def ensure_console_runtime(app: Any, *, view: Any | None = None) -> ConsoleRuntime:
    """Return `app`'s Console runtime, creating and attaching one if needed.

    Mirrors `ChatScreen._h3_image_edit_registry`: the app normally builds
    this in `__init__`, but a test app object (or one whose runtime was
    disposed at the last unmount) has none, and the screen must still be
    able to run.

    Args:
        app: The app object to read/attach the runtime on. `None` is
            tolerated — a detached runtime is returned rather than raising,
            because the screen's own accessors are reached from bare
            `ChatScreen.__new__` fixtures.
        view: The `ChatScreen` asking. The SAME runtime is shared across
            views now (it outlives every one of them); a new view simply
            claims it through `attach_view`, which opens a fresh visit.

    Returns:
        ConsoleRuntime: The runtime `view` should use.
    """
    runtime = getattr(app, CONSOLE_RUNTIME_ATTR, None)
    if not isinstance(runtime, ConsoleRuntime) and view is not None:
        # An app object that cannot hold the attribute (`None`, or a
        # read-only double) would otherwise hand every caller a BRAND-NEW
        # runtime, so a write through `ChatScreen._console_chat_store`'s
        # setter would be invisible to the very next read. Bare
        # `ChatScreen.__new__` fixtures reach the handles exactly that way.
        runtime = getattr(view, _VIEW_RUNTIME_FALLBACK_ATTR, None)
    if not isinstance(runtime, ConsoleRuntime):
        runtime = ConsoleRuntime(app)
        _attach(app, runtime)
        if view is not None and getattr(app, CONSOLE_RUNTIME_ATTR, None) is not runtime:
            try:
                setattr(view, _VIEW_RUNTIME_FALLBACK_ATTR, runtime)
            except Exception:  # noqa: BLE001 - a read-only view double is fine
                logger.debug("Console runtime: could not hold a fallback on the view.")
    if view is not None and runtime.view is not view:
        generation = runtime.attach_view(view)
        try:
            setattr(view, "_console_runtime_attachment_generation", generation)
        except Exception:  # noqa: BLE001 -- read-only view doubles are supported
            pass
    return runtime


async def leave_console_runtime(
    app: Any,
    *,
    view: Any | None = None,
    generation: int | None = None,
) -> bool:
    """Detach one Console projection while its app-owned work survives.

    Args:
        app: The app object holding the runtime.
        view: The `ChatScreen` unmounting. A superseded view leaves
            nothing — see `ConsoleRuntime.detach_view`.
        generation: The exact monotonic claim returned by ``attach_view``.

    Returns:
        True when this visit was actually ended.
    """
    runtime = getattr(app, CONSOLE_RUNTIME_ATTR, None)
    if not isinstance(runtime, ConsoleRuntime) and view is not None:
        runtime = getattr(view, _VIEW_RUNTIME_FALLBACK_ATTR, None)
    if not isinstance(runtime, ConsoleRuntime):
        return False
    return await runtime.leave_console(view, generation)


async def dispose_console_runtime(app: Any, *, view: Any | None = None) -> None:
    """Destroy `app`'s Console runtime and detach it. **App exit only.**

    Registered in `TldwCli._shutdown_app_owned_lifecycles`. Ordinary
    navigation away from Console goes through `leave_console_runtime`
    instead — that is the whole teardown split.

    Args:
        app: The app object holding the runtime. A missing or foreign
            attribute is a no-op.
        view: Present for symmetry; a runtime claimed by a different,
            still-live view is not disposed by a dying screen.
    """
    runtime = getattr(app, CONSOLE_RUNTIME_ATTR, None)
    if not isinstance(runtime, ConsoleRuntime):
        return
    if view is not None and runtime.view is not None and runtime.view is not view:
        return
    await runtime.dispose()
    if not runtime.hooks_v2_cleanup_pending:
        _attach(app, None)


# Definition-time owner for the optional stock hook connection scope.
_HOOK_CONTEXT_KEY_ORIGINAL_OWNER = ConsoleRuntime
_HOOK_PERMISSION_ACCESSOR_ORIGINAL = (
    ConsoleRuntime.ensure_hook_permissions,
    ConsoleRuntime.ensure_hook_permissions.__code__,
)


# Only the selected finite worker carries a publication source proof. Direct
# customized readers keep their original ABI and may delegate to the original.
_INITIAL_ACTIVITY_RECEIPT_SCOPE = threading.local()
_INITIAL_ACTIVITY_RECEIPT_ABSENT = object()

# Definition-time bodies for the optional finite initial Console preparation.
_INITIAL_ACTIVITY_RECEIPT_READERS = (
    ConsoleRuntime,
    tuple(
        (name, function, function.__code__, function.__globals__,
         function.__defaults__, function.__kwdefaults__, function.__closure__)
        for name in ("ensure_activity_receipt_service", "_prepare_initial_activity_receipts")
        for function in (getattr(ConsoleRuntime, name),)
    ),
    _INITIAL_ACTIVITY_RECEIPT_SCOPE,
    (inspect.getattr_static(ConsoleRuntime, "__getattribute__"),
     inspect.getattr_static(ConsoleRuntime, "__getattr__", _INITIAL_ACTIVITY_RECEIPT_ABSENT),
     inspect.getattr_static(ConsoleRuntime, "_app", _INITIAL_ACTIVITY_RECEIPT_ABSENT),
     _INITIAL_ACTIVITY_RECEIPT_ABSENT),
)


def _initial_receipt_preparer(runtime: Any) -> Any | None:
    owner, records, scope, lookup = _INITIAL_ACTIVITY_RECEIPT_READERS
    if (type(runtime) is not owner or ConsoleRuntime is not owner
            or _INITIAL_ACTIVITY_RECEIPT_SCOPE is not scope
            or lookup[0] is not object.__getattribute__
            or lookup[1] is not lookup[3] or lookup[2] is not lookup[3]
            or inspect.getattr_static(owner, "__getattribute__") is not lookup[0]
            or inspect.getattr_static(owner, "__getattr__", lookup[3]) is not lookup[1]
            or inspect.getattr_static(owner, "_app", lookup[3]) is not lookup[2]):
        return None
    for name, function, code, namespace, defaults, kwdefaults, closure in records:
        if (
            inspect.getattr_static(owner, name, None) is not function
            or name in vars(runtime)
            or function.__code__ is not code
            or function.__globals__ is not namespace
            or namespace is not globals()
            or function.__defaults__ is not defaults
            or function.__kwdefaults__ is not kwdefaults
            or function.__closure__ is not closure
        ):
            return None
    return MethodType(records[1][1], runtime)


# Definition-time original peeks used by the optional pending-only display.
_CONSOLE_PENDING_RUNTIME_PROPERTIES = (
    ConsoleRuntime,
    tuple(
        (
            name,
            descriptor,
            descriptor.fget,
            descriptor.fget.__code__,
            descriptor.fget.__globals__,
            descriptor.fget.__defaults__,
            descriptor.fget.__kwdefaults__,
            tuple((descriptor.fget.__kwdefaults__ or {}).items()),
            descriptor.fget.__closure__,
            tuple(
                (cell, cell.cell_contents) for cell in descriptor.fget.__closure__ or ()
            ),
        )
        for name in ("chat_controller", "chat_store")
        for descriptor in (inspect.getattr_static(ConsoleRuntime, name),)
    ),
)
