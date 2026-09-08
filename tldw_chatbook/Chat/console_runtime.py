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
import inspect
import functools
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, get_ident
from typing import TYPE_CHECKING, Any, Callable, Mapping
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleLifecycleRevisionChanged,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyDefaults,
)
from tldw_chatbook.Chat.console_display_state import console_prompted_source_count
from tldw_chatbook.Chat.console_onboarding_state import (
    coerce_console_first_send_completed,
)
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.Chat.console_voice_supervisor import VoiceDispatchSupervisor
from tldw_chatbook.Chat.console_voice_promotion import (
    VoicePromotionOwner,
    VoicePromotionQuitPermit,
)
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
from tldw_chatbook.Chat.thinking_blocks import normalize_thinking_history_policy
from tldw_chatbook.config import coerce_bool_setting, runtime_capture_policy
from tldw_chatbook.Persona_Buddy.console_adapter import PersonaBuddyConsoleAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

#: The app attribute this module's helpers read and write. Named once so a
#: test can assert on the protocol rather than on a string literal.
CONSOLE_RUNTIME_ATTR = "console_runtime"

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


@dataclass(frozen=True, slots=True)
class ConsoleTurnRecoveryEntry:
    """One exact pre-durable draft retained across Console navigation."""

    turn_id: str
    session_id: str
    draft: str = field(repr=False)
    attachments: tuple[Any, ...] = field(repr=False)
    insertion_order: int


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


def _library_provider_for_app(app: Any, turn_context: Any | None = None) -> Any | None:
    """Build one Library provider from frozen authority and app services."""
    if turn_context is None:
        return None
    if not turn_context.library_authority.direct_library_tools:
        from tldw_chatbook.Agents.library_rag_tool_provider import (
            LibraryRagToolProvider,
        )

        return LibraryRagToolProvider(
            getattr(app, "library_rag_search_service", None)
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
        collections_service=getattr(app, "local_library_collections_service", None),
        media_chunk_service=media_chunk_service,
        notes_scope_service=getattr(app, "notes_scope_service", None),
        policy_enforcer=getattr(app, "service_policy_enforcer", None),
    )
    return LibraryToolProvider(service)


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
        "set_pending_question",
        "controller",
        why="`request_user_questions` returns `{answered: False, reason: "
        "'cancelled'}` when it is None, and `_ask_user_wiring` registers no "
        "tool at all without it -- a viewless run cannot be asked, which is "
        "PRD A10's headless posture.",
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
        self._canvas_settlement_listener = self._forward_canvas_settlement
        if canvas_enabled_reader is None:
            from tldw_chatbook.config import get_canvas_execution_enabled

            canvas_enabled_reader = get_canvas_execution_enabled
        self._canvas_enabled_reader = canvas_enabled_reader
        self._canvas_disabled_latched = not self._read_canvas_enabled()
        self._canvas_policy_watch_task: asyncio.Task[None] | None = None
        self._legacy_trace_maintenance_task: asyncio.Task[None] | None = None
        # One app-wide mutation lane for exact persisted-conversation opens.
        # Individual ChatScreen workspaces are disposable views over this
        # runtime, so a per-screen lock would allow their hydration/rollback
        # sequences to interleave.
        self.character_conversation_activation_lock = asyncio.Lock()
        self.trace_compatibility_metrics = _LazyTraceCompatibilityMetrics()
        self._scratch_spaces = ConsoleScratchSpaceManager()
        self._raw_cli_refusal_stash_bank: dict[str, list[Any]] = {}
        self._voice_dispatch_supervisor = VoiceDispatchSupervisor()
        self._voice_worker = None
        self._voice_process_supervisor = None
        self._voice_promotion_owner = VoicePromotionOwner(lambda: self._chat_store)
        self._persona_buddy_sink = PersonaBuddyConsoleAdapter(
            getattr(app, "persona_buddy_controller", None)
        )
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

    def _notify_terminal_receipt(
        self, conversation_id: str, receipt_id: str, *, revision: int
    ) -> bool:
        """Emit one sanitized terminal notice, recording only on success."""
        notify = getattr(self._app, "notify", None)
        if not callable(notify):
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
        revision = self._reserve_attention_revision()
        with self._attention_operation_lock:
            if not self._attention_revision_is_current(revision):
                return self.console_needs_attention

            service = self._console_local_marks_service()
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
            with self._attention_lock:
                if marks_known:
                    self._notified_terminal_receipts.intersection_update(
                        active_receipts
                    )
                previous = self._console_needs_attention
                if marks_known or self._last_known_terminal_marks is not None:
                    needs_attention = bool(marks or hidden_decisions)
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
            for conversation_id, receipt_id in rendered:
                if not conversation_id or not receipt_id:
                    continue
                try:
                    removed = acknowledge(conversation_id, receipt_id) is True
                except Exception as exc:  # noqa: BLE001 -- mark remains retryable
                    logger.debug(
                        "Console terminal-attention acknowledgement failed "
                        "(exception_type={})",
                        type(exc).__name__,
                    )
                    continue
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

    def set_agent_bridge(self, value: Any) -> None:
        """Replace the agent-bridge handle."""
        self._agent_bridge = value

    def set_chat_controller(self, value: Any) -> None:
        """Replace the chat-controller handle."""
        self._chat_controller = value
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
        fleet_wake = getattr(value, "fleet_wake", None)
        bind_wake_submitter = getattr(fleet_wake, "bind_runtime_submitter", None)
        if callable(bind_wake_submitter):
            bind_wake_submitter(self._submit_fleet_wake)

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
                from tldw_chatbook.Chat.prompt_history import (
                    PromptHistory,
                    default_prompt_history_path,
                )

                self._prompt_history = PromptHistory(default_prompt_history_path())
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
        request: ConsoleTurnCustodyRequest,
        attachments: tuple[Any, ...] = (),
        staged_evidence_revision: int | None = None,
    ) -> _ConsoleTurnCustodyRecord:
        """Retain one request before its task may begin running."""
        if request.turn_id in self._turn_custody:
            raise RuntimeError("Console turn is already in runtime custody.")
        record = _ConsoleTurnCustodyRecord(
            turn_id=request.turn_id,
            session_id=request.session_id,
            request=request,
            inputs=_ConsoleTurnCustodyInputs(
                attachments=attachments,
                staged_evidence_revision=staged_evidence_revision,
            ),
        )
        self._turn_custody[record.turn_id] = record
        return record

    def _release_custody(self, turn_id: str) -> None:
        """Drop the runtime's final references to an accepted turn."""
        record = self._turn_custody.pop(turn_id, None)
        if record is not None:
            record.request = None
            record.inputs.attachments = ()
            record.inputs.staged_evidence_revision = None
            record.task = None

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
        attachments = store.transfer_pending_attachments_to_turn(
            request.session_id,
            request.turn_id,
            request.attachment_ids,
        )
        try:
            record = self._register_custody(
                request,
                attachments,
                self._staged_evidence_lease_revision(request.staged_evidence_launch),
            )
        except BaseException:
            store.restore_transferred_pending_attachments(
                request.session_id, attachments
            )
            raise
        coroutine = self._run_custodied_turn(
            record,
            origin=origin,
            queue_entry_id=queue_entry_id,
            queue_authorization=queue_authorization,
            wake_authorization=wake_authorization,
            raise_on_refusal=recover_before_acceptance,
        )
        try:
            record.task = asyncio.create_task(coroutine)
        except BaseException:
            coroutine.close()
            store.restore_transferred_pending_attachments(
                request.session_id, attachments
            )
            self._release_custody(record.turn_id)
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
    ) -> Any:
        """Run one screen-free turn using only its frozen custody record."""
        request = record.request
        controller = self._chat_controller
        if request is None or controller is None:
            raise RuntimeError("Console controller is unavailable for runtime custody.")

        def mark_durable_acceptance() -> None:
            record.inputs.durable_accepted = True

        async def submit() -> Any:
            return await controller.submit_draft(
                request.draft,
                session_id=request.session_id,
                origin=origin,
                queue_entry_id=queue_entry_id,
                queue_authorization=queue_authorization,
                wake_authorization=wake_authorization,
                configuration=request.configuration,
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
            )

        result = (
            await submit()
            if origin is ConsoleSubmissionOrigin.AGENT_WAKE
            else await controller.run_prompt_chain(
                session_id=request.session_id,
                initial_turn=submit,
            )
        )
        if (
            not bool(getattr(result, "accepted", False))
            and not record.inputs.durable_accepted
            and raise_on_refusal
        ):
            raise RuntimeError("Console turn was refused before durable acceptance.")
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
        )

    def restore_turn_recovery(self, turn_id: str) -> ConsoleTurnRecoveryEntry:
        """Re-stage one exact recovery into its still-live owning session."""
        entry = self._turn_recoveries[turn_id]
        store = self._chat_store
        if store is None or entry.session_id not in {
            session.id for session in store.sessions()
        }:
            raise RuntimeError("Recovery session is no longer available.")
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

    def _record_turn_recovery(self, record: _ConsoleTurnCustodyRecord) -> None:
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
        accepted = False
        try:
            result = task.result()
            accepted = bool(getattr(result, "accepted", False))
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            if (
                record is not None
                and recover_before_acceptance
                and not record.inputs.durable_accepted
            ):
                self._record_turn_recovery(record)
            logger.warning(
                "Console runtime turn ended with exception_type={}",
                type(exc).__name__,
            )
        finally:
            accepted = accepted or bool(
                record is not None and record.inputs.durable_accepted
            )
            if record is not None:
                self._release_custody(record.turn_id)
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

        binding = _CanvasNativeViewBinding(
            scope_resolver=resolve_live_scope,
            bridge_sink=bridge_sink,
            bridge_prepare=bridge_prepare,
            auto_open=auto_open,
            publication_guard=publication_guard,
        )
        with self._canvas_native_lock:
            if not self._canvas_enabled():
                return None
            self._canvas_native_view_binding = binding
            controller = self._canvas_controller
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

        with self._canvas_native_lock:
            if not self._canvas_enabled():
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

    def _read_canvas_enabled(self) -> bool:
        """Read the shared policy without changing the process-lifetime latch."""

        try:
            return self._canvas_enabled_reader() is True
        except Exception:  # noqa: BLE001 - execution policy fails closed
            return False

    def _canvas_enabled(self) -> bool:
        """Read the global kill switch and the restart-required runtime latch."""

        if self._disposed or self._canvas_disabled_latched:
            return False
        if not self._read_canvas_enabled():
            self._canvas_disabled_latched = True
            return False
        return True

    def canvas_enabled(self) -> bool:
        """Expose this app runtime's restart-latched Canvas availability."""

        return self._canvas_enabled()

    def canvas_authority_is_current(self, authority: Any) -> bool:
        """Return whether *authority* still owns enabled Canvas effects."""

        return self._canvas_enabled() and self._canvas_native_authority is authority

    def latch_canvas_disabled(self) -> None:
        """Synchronously accept a disable before asynchronous cleanup begins."""

        with self._canvas_native_lock:
            self._canvas_disabled_latched = True

    def start_async_lifecycles(self) -> None:
        """Start loop-bound runtime work after the Textual loop is running.

        ``TldwCli`` is constructed synchronously before ``App.run`` creates
        its event loop, so the constructor's best-effort watcher start cannot
        cover the shipping CLI lifecycle by itself.  App mount calls this
        idempotent handoff independently of Canvas preview creation.
        """

        self._start_canvas_policy_watcher()

    async def apply_canvas_policy(self) -> None:
        """Idempotently revoke all browser delivery after Canvas is disabled.

        Re-enabling requires a process restart. Stored and staged artifact data
        remain owned by their existing repositories and lifecycle controllers.
        """

        if self._canvas_enabled():
            return
        self.latch_canvas_disabled()
        with self._canvas_native_lock:
            gateway, self._canvas_gateway = self._canvas_gateway, None
            authority, self._canvas_native_authority = (
                self._canvas_native_authority,
                None,
            )
            self._canvas_gateway_authority = None
            self._canvas_native_view_binding = None
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

    def _start_canvas_policy_watcher(self) -> None:
        """Watch shared config while a native Canvas preview can be open."""

        if self._disposed:
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

        while self._canvas_enabled():
            await asyncio.sleep(0.25)
        await self.apply_canvas_policy()

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
            last_provider_activity = time.monotonic()
            last_physical_attempt = 0.0
            last_collected_epoch: int | None = None
            pending_gc_result: Any | None = None
            while not self._disposed:
                try:
                    result = await asyncio.to_thread(maintenance.run_batch)
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
                        await asyncio.sleep(1.0)
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
                        current_epoch = await asyncio.to_thread(
                            collector.current_graph_epoch
                        )
                        if pending_gc_result is None:
                            if current_epoch == last_collected_epoch:
                                await asyncio.sleep(1.0)
                                continue
                            pending_gc_result = await asyncio.to_thread(
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
                        outcome = await asyncio.to_thread(
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
            db = getattr(self._app, "chachanotes_db", None)
            db_path = getattr(db, "db_path", None) if db is not None else None
            if not db_path or str(db_path) == ":memory:":
                return None
            from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

            runs_db = AgentRunsDB(Path(db_path).parent / "agent_runs.db")
            try:
                service = _LazyConsoleActivityReceiptService(
                    runs_db,
                    getattr(self._app, "conversation_local_marks_service", None),
                )
                # Dispose writes its lifetime latch under this same lock. Never
                # publish an owner between that latch and its resource snapshot.
                with self._canvas_native_lock:
                    if self._disposed:
                        return None
                    self._agent_runs_db = runs_db
                    self._activity_receipts = service
                    return service
            finally:
                # AgentRunsDB.close affects only the calling thread. A worker's
                # held initialization connection cannot be closed by app exit.
                if self._disposed or get_ident() != self._receipt_owner_thread_id:
                    runs_db.close()

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
            store=store_factory(),
            provider_gateway=provider_gateway_factory(),
            skills_service=skills_service,
            native_tools_enabled=(
                functools.partial(_native_tools_enabled_for_app, self._app)
            ),
            change_tracker=change_tracker if change_tracker.available else None,
            buddy_sink=self.persona_buddy_sink,
            change_finalization_coordinator=change_coordinator,
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
            try:
                return service.hydrate_from_storage()
            finally:
                if runs_db is not None:
                    runs_db.close()

        async def hydrate() -> int:
            result = await asyncio.to_thread(read_receipts)
            if self._disposed or self.authority_token != token:
                return 0
            return result

        task = loop.create_task(hydrate())
        self._activity_hydration_task = task
        return task

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

        kwargs.update(
            chat_dictionary_applier=functools.partial(
                _apply_chat_dictionaries_for_app, self._app
            ),
            world_info_applier=functools.partial(
                _apply_world_info_for_app, self._app
            ),
            rag_capture_provider=self._capture_frozen_console_staged_rag,
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
        kwargs.setdefault("canvas_enabled_reader", self._canvas_enabled)
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
        return self._chat_controller

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
        reader = getattr(wake, "delivering_session_id", None)
        session_id = reader() if callable(reader) else None
        if not session_id:
            return
        hook = getattr(wake, "delivery_ui_hook", None)
        if not callable(hook):
            return
        try:
            hook(session_id)
        except Exception as exc:  # noqa: BLE001 -- UI freshness is best-effort
            logger.debug(
                "wake delivery UI hook re-arm raised (exception_type={})",
                type(exc).__name__,
            )

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
                self._pause_project_instruction_generation(self._attached_generation)
            self._attachment_generation += 1
            generation = self._attachment_generation
            self.view = view
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
            self._clear_view_hooks()
            self.view = None
            self._attached_generation = None
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
        """Drain already-claimed voice publication before closing its session."""

        owner = self._voice_promotion_owner
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
        self._admission_fenced_sessions.add(session_id)
        try:
            ticket = controller.begin_session_close(
                session_id,
                expected_revision=expected_revision,
            )
        except BaseException:
            self._admission_fenced_sessions.discard(session_id)
            raise

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
        await_fleet = getattr(bridge, "await_fleet_terminal", None)
        fleet_waiter: asyncio.Task[Any] | None = None
        if callable(await_fleet):
            fleet_waiter = asyncio.create_task(
                await_fleet(ticket.conversation_id),
                name=f"console-close-fleet-{ticket.close_id[:8]}",
            )
            tasks.add(fleet_waiter)

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
        fleet_fenced = callable(getattr(bridge, "fence_fleet", None))
        fleet_drain_succeeded = not fleet_fenced and fleet_waiter is None
        if (
            fleet_waiter is not None
            and fleet_waiter.done()
            and not fleet_waiter.cancelled()
        ):
            try:
                fleet_drain_succeeded = bool(fleet_waiter.result())
            except Exception:  # noqa: BLE001 -- unknown drain stays fenced
                fleet_drain_succeeded = False
        for turn_id, record in tuple(self._turn_custody.items()):
            if record.session_id == session_id:
                self._release_custody(turn_id)
        closed = controller.finalize_session_close(ticket)
        if not pending and fleet_drain_succeeded:
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
                    if fence_fleet(
                        conversation_id,
                        generation=provisional_generation,
                    ):
                        acquired_fences.append(conversation_id)
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
            if voice_promotion_permit is None:
                raise RuntimeError("Voice-promotion quit permit is required.")
            self._voice_promotion_owner.consume_quit_permit(voice_promotion_permit)
        except BaseException:
            abort_provisional_fences()
            raise
        self._disposed = True
        if self._voice_process_supervisor is not None:
            self._voice_process_supervisor.begin_close()
        self._admission_fenced_sessions.update(session_ids)
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
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(timeout_seconds))

        def remaining_seconds() -> float:
            return max(0.0, deadline - loop.time())

        with self._canvas_native_lock:
            self._disposed = True
            self._canvas_native_view_binding = None
        if self._voice_process_supervisor is not None:
            self._voice_process_supervisor.begin_close()
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
        self._activity_hydration_task = None
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
        conversation_ids: set[str] = set()
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

        drain_tasks: set[asyncio.Future[Any]] = set()
        voice_cleanup = None
        if self._voice_worker is not None or self._voice_process_supervisor is not None:

            async def close_voice_worker():
                try:
                    if self._voice_process_supervisor is not None:
                        await self._voice_process_supervisor.aclose()
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
        pending = await self._bounded_wait(
            drain_tasks,
            timeout_seconds=remaining_seconds(),
        )
        if voice_cleanup in pending:
            logger.warning("Console runtime: voice cleanup remains pending at dispose.")
        for task in pending:
            if task is voice_cleanup:
                continue
            task.cancel()
            task.add_done_callback(self._consume_task_outcome)
        if voice_cleanup is not None:
            # The child budget cannot end UI-owned provider/TTS/claimed work.
            # Keep its original owner loop and store until actual custody settles.
            await asyncio.shield(voice_cleanup)
        for turn_id in tuple(self._turn_custody):
            self._release_custody(turn_id)
        if self._chat_store is not None:
            end_app_runtime = getattr(self._chat_store, "end_app_runtime", None)
            if callable(end_app_runtime):
                try:
                    await asyncio.to_thread(end_app_runtime)
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
    _attach(app, None)
