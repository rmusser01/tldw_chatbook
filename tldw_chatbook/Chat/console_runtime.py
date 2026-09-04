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
| `leave_console_runtime` | every navigation AWAY from Console | ends ONE visit: clears the view's hook slots, cancels+awaits this visit's USER stream tasks, denies its parked approval rounds, tombstones its queue chains |
| `dispose_console_runtime` | app exit (`_shutdown_app_owned_lifecycles`) | the permanent form — `controller.shutdown()` then `gateway.aclose()`, exactly the order `on_unmount` used to run |

An `AGENT_WAKE` turn is deliberately NOT cancelled by `leave_console`
(owner ruling): cancelling it would re-create the "only completes if you
stay" gap this whole arc exists to close, and a wake turn is structurally
the same class of work as the fleet survivor AC#2 keeps running. AC#2
names USER turns only.

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

- ~~**`_attempt`'s `_shutdown_requested` gate**~~ — **relaxed by
  task-15860's wake-fires-headless slice.** The lifetime landing left it
  alone deliberately: `leave_console` SETS that visit's Event and only
  `begin_visit` (the next `attach_view`) replaces it, so between visits
  the flag stayed set and no wake fired headless. That slice has now
  landed, and the gate reads `_disposed` instead — `dispose()` (app exit)
  refuses a wake; a visit that merely ended does not. `leave_console`'s
  Event is unchanged and still denies this visit's parked approval
  rounds.

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
from datetime import datetime, timezone
import inspect
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyDefaults,
)
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.Chat.thinking_blocks import normalize_thinking_history_policy
from tldw_chatbook.Persona_Buddy.console_adapter import PersonaBuddyConsoleAdapter
from tldw_chatbook.config import coerce_bool_setting, runtime_capture_policy

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


__all__ = [
    "CONSOLE_RUNTIME_ATTR",
    "CONSOLE_VIEW_HOOK_SLOTS",
    "ConsoleRuntime",
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
    # -- constructor-supplied callables (read at construction only, so with
    #    a surviving runtime they would otherwise stay bound to visit 1's
    #    screen for the whole app's life) -----------------------------------
    ConsoleViewHookSlot(
        "_chat_dictionary_applier",
        "controller",
        why="`_apply_chat_dictionaries` early-returns the payload unchanged "
        "when it is None: a viewless turn substitutes nothing, which is "
        "correct — dictionaries are a per-conversation view concern and a "
        "wake notice is machine text.",
    ),
    ConsoleViewHookSlot(
        "_world_info_applier",
        "controller",
        why="`_apply_world_info` early-returns the payload unchanged when "
        "it is None. Same reasoning as the dictionary applier.",
    ),
    ConsoleViewHookSlot(
        "_rag_capture_provider",
        "controller",
        why="`_resolve_staged_rag_context` returns the empty 4-tuple when "
        "it is None. Staged RAG is composed IN the composer; with no "
        "composer there is nothing staged to consume.",
    ),
    ConsoleViewHookSlot(
        "_default_session_settings",
        "controller",
        why="Both read sites are `if self._default_session_settings is not "
        "None` guards; a viewless session falls back to the controller's "
        "own defaults rather than a dead screen's widget state.",
    ),
    ConsoleViewHookSlot(
        "_library_provider_factory",
        "controller",
        why="`_library_provider_for_context` returns None when it is None, "
        "and the send path already degrades a missing factory to 'no "
        "Library tools this run'. The factory reads Settings through the "
        "SCREEN, so a viewless turn must not use a stale one.",
    ),
    ConsoleViewHookSlot(
        "_global_user_display_name",
        "controller",
        viewless_user_display_name,
        why="**Not None.** `_presentation_context_for` CALLS this slot with "
        'no `is None` guard (the constructor\'s `... or (lambda: "User")` '
        "runs once, at construction). None makes every read raise "
        "TypeError into a broad `except` that logs and falls back — a "
        "silently degraded slot. The explicit accessor returns the same "
        '"User" that fallback would, without the warning per read.',
    ),
    ConsoleViewHookSlot(
        "_turn_context_provider",
        "controller",
        why="`_compose_turn_context` guards with `is not None` and uses the "
        "controller's own selection state instead. The provider reads live "
        "WIDGET state (mode toggles, tool switches), so a viewless turn "
        "must not consult it at all.",
    ),
    # -- post-construction UI bridges -------------------------------------
    ConsoleViewHookSlot(
        "on_submission_accepted",
        "controller",
        why="Guarded (`if callback is None: return`) and MANUAL-origin "
        "only. Its whole job is clearing a composer that does not exist.",
    ),
    ConsoleViewHookSlot(
        "follow_watchlists_operations",
        "controller",
        why="Guarded by `remount_watchlists_operation_receipts`; canonical "
        "receipt identities remain retained by the app-owned controller "
        "while no Console view is mounted.",
    ),
    ConsoleViewHookSlot(
        "prompt_history",
        "controller",
        why="Guarded (`if history is None ... return`). Prompt history "
        "records what the USER typed; a wake notice is not user input "
        "(the wake invariant), so a viewless turn must record nothing.",
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
        "set_pending_skill_install",
        "controller",
        why="None makes `request_skill_install_confirm` **fail closed "
        "immediately** — its read site says so in as many words: nothing "
        "could ever set the Event, so denying at once beats blocking for "
        "the full timeout with no way to be resolved.",
    ),
    ConsoleViewHookSlot(
        "set_pending_skill_script",
        "controller",
        why="Same fail-closed-at-once contract for "
        "`request_skill_script_confirm` (`allow=False, remember=False`).",
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

    def __init__(self, app: Any) -> None:
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
        self._activity_hydration_task: asyncio.Task[int] | None = None
        self._change_review_coordinator: Any | None = None
        self._chat_controller: Any | None = None
        self._legacy_trace_maintenance_task: asyncio.Task[None] | None = None
        self.trace_compatibility_metrics = _LazyTraceCompatibilityMetrics()
        self._scratch_spaces = ConsoleScratchSpaceManager()
        self._raw_cli_refusal_stash_bank: dict[str, list[Any]] = {}
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
        #: Latched by `dispose()` (app exit). Every `ensure_*` returns what
        #: it already holds afterwards and builds nothing new -- see
        #: `dispose` for why a rebuild during quit is the hazard.
        self._disposed: bool = False
        self.authority_token = str(uuid4())
        database = getattr(app, "chachanotes_db", None)
        database_path = getattr(database, "db_path", None)
        self.profile_authority = (
            str(Path(database_path).expanduser().resolve(strict=False))
            if database_path and str(database_path) != ":memory:"
            else ""
        )
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

    # -- handle writes (the screen's properties, and 59 test sites) --------

    def set_chat_store(self, value: Any) -> None:
        """Replace the store handle (a test double, or `None` to rebuild)."""
        self._chat_store = value

    def set_provider_gateway(self, value: Any) -> None:
        """Replace the provider-gateway handle."""
        self._provider_gateway = value

    def set_agent_bridge(self, value: Any) -> None:
        """Replace the agent-bridge handle."""
        self._agent_bridge = value

    def set_chat_controller(self, value: Any) -> None:
        """Replace the chat-controller handle."""
        self._chat_controller = value

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
            legacy_normalization_enabled = callable(
                getattr(db, "transaction", None)
            )
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
        self._chat_store = ConsoleChatStore(
            persistence=persistence,
            settle_provider_traces_off_thread=True,
            trace_projection=(
                ConsoleTraceProjection(
                    legacy_reader=db.get_message_exchanges,
                    normalized_reader=(
                        read_normalized_calls
                        if legacy_normalization_enabled
                        else None
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
        )
        if db is not None and legacy_normalization_enabled:
            self._schedule_legacy_trace_maintenance(db, get_legacy_normalizer)
        self._bind_view_hooks()
        return self._chat_store

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
                    await asyncio.sleep(
                        LEGACY_TRACE_MAINTENANCE_RETRY_DELAY_SECONDS
                    )
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
                config_provider=config_provider,
                trace_call_boundary_factory=trace_call_boundary_factory,
                normalized_writes_enabled=lambda: (
                    runtime_capture_policy().normalized_writes_enabled
                ),
                trace_compatibility_metrics=self.trace_compatibility_metrics,
            )
        return self._provider_gateway

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

        Every screen-supplied dependency arrives as a callable, precisely
        to preserve that ordering — nothing on the view is read until the
        probe has passed.

        Args:
            store_factory: Returns the chat store the bridge should use.
                Called only past the durable-DB probe.
            provider_gateway_factory: Returns the provider gateway. Same.
            skills_service: The app's skills scope service, or `None`. A
                plain value: it is a `getattr` on the APP, which the probe
                has already touched.
            native_tools_enabled_factory: Returns the callable the bridge
                stores and calls later to read the `[console]
                native_tool_calls` gate. A factory, not that callable, so
                the view is not read on the no-agent-runtime path.

        Returns:
            Any: The `ConsoleAgentBridge`, or `None` when there is no
            durable ChaChaNotes DB to key the sibling `AgentRunsDB` off.
        """
        if self._agent_bridge is not None or self._disposed:
            return self._agent_bridge
        db = getattr(self._app, "chachanotes_db", None)
        db_path = getattr(db, "db_path", None) if db is not None else None
        if not db_path or str(db_path) == ":memory:":
            self._agent_bridge = None
            return None
        from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

        runs_db = AgentRunsDB(Path(db_path).parent / "agent_runs.db")
        self._agent_runs_db = runs_db
        self._activity_receipts = _LazyConsoleActivityReceiptService(
            runs_db,
            getattr(self._app, "conversation_local_marks_service", None),
        )
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
                native_tools_enabled_factory()
                if native_tools_enabled_factory is not None
                else None
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
        if service.hydration_state() == "ready":
            return self._activity_hydration_task
        task = self._activity_hydration_task
        if task is not None and not task.done():
            return task
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        token = self.authority_token

        async def hydrate() -> int:
            result = await asyncio.to_thread(service.hydrate_from_storage)
            if self._disposed or self.authority_token != token:
                return 0
            return result

        task = loop.create_task(hydrate())
        self._activity_hydration_task = task
        return task

    def ensure_chat_controller(self, **kwargs: Any) -> "ConsoleChatController":
        """Return the Console chat controller, creating it lazily.

        The keyword arguments are `ConsoleChatController.__init__`'s, passed
        straight through from the view exactly as
        `ChatScreen._ensure_console_chat_controller` passed them. They are
        read at construction only; the screen keeps re-applying its
        selection state and its UI hooks after this returns, as before.

        Returns:
            ConsoleChatController: The runtime's controller.
        """
        if self._chat_controller is not None or self._disposed:
            return self._chat_controller
        from tldw_chatbook.Chat.console_chat_controller import (
            ConsoleChatController,
        )

        kwargs.setdefault("buddy_sink", self.persona_buddy_sink)
        kwargs.setdefault("scratch_spaces", self._scratch_spaces)
        kwargs.setdefault("activity_receipts", self._activity_receipts)
        raw_cli_runtime = getattr(self._app, "raw_cli_runtime", None)
        kwargs.setdefault(
            "cancel_raw_cli_session",
            getattr(raw_cli_runtime, "cancel_session", None),
        )
        self._chat_controller = ConsoleChatController(**kwargs)
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
        """Mount the card for a round armed while nothing was attached.

        task-15860 Task 5. The screen's approval card is derived entirely
        from its own `_task_resume_state`, which a FRESH screen starts
        empty — and screens are never cached. So a round armed headlessly
        (a risk-tagged tool in a wake turn) would sit registered,
        announced app-wide, and still invisible the moment the user acted
        on that announcement and opened Console. `switch_session`'s
        identical re-derive would eventually mount it, but only if the
        user switched sessions — which they have no reason to do, never
        having seen a card.

        Gated on a NEW claim (`attach_view`'s `previous is not view`), not
        run on every `_ensure_console_chat_controller()`: re-pushing the
        payload rebuilds the card's rows, so an unconditional re-derive
        would discard a half-made decision (a chosen Select, not yet
        submitted) on any tick that happens to touch the controller.

        Best-effort: a raising seam is logged, never propagated into the
        attach.
        """
        controller = self._chat_controller
        remount = (
            getattr(controller, "remount_pending_approval_for_active_session", None)
            if controller is not None
            else None
        )
        if not callable(remount):
            return
        try:
            remount()
        except Exception as exc:  # noqa: BLE001 -- an attach never dies on this
            logger.debug(
                "Pending-approval remount raised at attach (exception_type={})",
                type(exc).__name__,
            )

    def attach_view(self, view: Any) -> None:
        """Claim this runtime for `view` and open a new Console visit.

        **This replaces Task 1's `ConsoleRuntime.view` stand-in.** That
        device kept a runtime claimed by a different view from being shared
        — it simply built a second runtime, which was only ever a way of
        reproducing dispose-at-unmount semantics. Now there is one runtime
        and the claim is real: whoever attached LAST owns it, and a
        superseded view's `detach_view` is a no-op (see there).

        A NEW claim (the view changed) also opens a visit on the surviving
        controller: a fresh per-visit cancellation Event and re-opened
        prompt-queue admission. Re-attaching the SAME view only re-binds
        hooks, so `_ensure_console_chat_controller` staying idempotent
        costs nothing.
        """
        previous = self.view
        self.view = view
        claimed = previous is not view
        if claimed:
            controller = self._chat_controller
            begin_visit = (
                getattr(controller, "begin_visit", None)
                if controller is not None
                else None
            )
            if callable(begin_visit):
                begin_visit()
        self._bind_view_hooks()
        self._rearm_delivery_ui_hook()
        if claimed:
            self.remount_pending_approval()

    def detach_view(self, view: Any | None = None) -> bool:
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
        if view is not None and self.view is not view:
            return False
        self._clear_view_hooks()
        self.view = None
        return True

    # -- teardown ----------------------------------------------------------

    async def leave_console(self, view: Any | None = None) -> bool:
        """End ONE Console visit. The runtime survives.

        The order matters and is asserted: hooks are cleared FIRST, then
        the controller's per-visit teardown runs. Cancelling a turn drives
        it to a terminal run state, and a terminal state fires
        `notify_run_outcome`/`notify_run_failure` — into the screen that is
        being torn down if the slots were still bound.

        Args:
            view: The unmounting view. A superseded view leaves nothing
                (`detach_view`'s no-op), because the successor is still
                using this runtime's turns.

        Returns:
            True when this visit was actually ended.
        """
        if not self.detach_view(view):
            return False
        controller = self._chat_controller
        leave = getattr(controller, "leave_console", None) if controller else None
        if callable(leave):
            result = leave()
            if inspect.isawaitable(result):
                await result
        return True

    async def dispose(self) -> None:
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
        self._disposed = True
        maintenance_task = self._legacy_trace_maintenance_task
        if maintenance_task is not None and not maintenance_task.done():
            maintenance_task.cancel()
            try:
                await maintenance_task
            except asyncio.CancelledError:
                pass
        self._raw_cli_refusal_stash_bank.clear()
        self._scratch_spaces.tombstone_all()
        self.detach_view(None)
        controller, gateway = self._chat_controller, self._provider_gateway
        coordinator, runs_db = (
            self._change_review_coordinator,
            self._agent_runs_db,
        )
        self.generation += 1
        hydration_task = self._activity_hydration_task
        self._activity_hydration_task = None
        if hydration_task is not None and not hydration_task.done():
            hydration_task.cancel()
        if controller is not None:
            try:
                await controller.shutdown()
            except Exception:  # noqa: BLE001 - quit must not die on teardown
                logger.opt(exception=True).warning(
                    "Console runtime: controller shutdown failed at dispose."
                )
        if self._chat_store is not None:
            end_app_runtime = getattr(self._chat_store, "end_app_runtime", None)
            if callable(end_app_runtime):
                try:
                    await asyncio.to_thread(end_app_runtime)
                except Exception:  # noqa: BLE001 - quit must continue cleanup
                    logger.opt(exception=True).warning(
                        "Console runtime: trace settlement shutdown failed at dispose."
                    )
        try:
            await asyncio.to_thread(self._scratch_spaces.dispose)
        except Exception as exc:  # noqa: BLE001 - quit must continue after cleanup failure
            logger.warning(
                "Console runtime: scratch cleanup failed at dispose category={}",
                type(exc).__name__,
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
        close = getattr(gateway, "aclose", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - same
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
        runtime.attach_view(view)
    return runtime


async def leave_console_runtime(app: Any, *, view: Any | None = None) -> bool:
    """End one Console visit on `app`'s runtime, which SURVIVES.

    The nav-away counterpart to `dispose_console_runtime`. Cancels this
    visit's user turns, denies its parked approval rounds and tombstones
    its queue chains — and leaves the store, gateway, bridge and controller
    alive for the next visit (and for a survivor that outlives the screen).

    Args:
        app: The app object holding the runtime.
        view: The `ChatScreen` unmounting. A superseded view leaves
            nothing — see `ConsoleRuntime.detach_view`.

    Returns:
        True when this visit was actually ended.
    """
    runtime = getattr(app, CONSOLE_RUNTIME_ATTR, None)
    if not isinstance(runtime, ConsoleRuntime) and view is not None:
        runtime = getattr(view, _VIEW_RUNTIME_FALLBACK_ATTR, None)
    if not isinstance(runtime, ConsoleRuntime):
        return False
    return await runtime.leave_console(view)


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
