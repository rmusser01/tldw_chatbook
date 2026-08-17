"""Wake at launch: deliver what was already owed when the app starts.

task-15860 Task 6, and the last fire point the arc was missing. The
wake-fires-headless landing made a survivor settling with **no Console
mounted** deliver a full supervisor turn — but only inside a process that
had opened Console at least once, because `ensure_chat_controller` /
`ensure_agent_bridge` are lazy and their only callers were `ChatScreen`
and its Console modules. A child that finished while the app was closed —
or one whose delivery the user quit out from under — waited for the next
Console visit, however many launches later that was.

**The owner's ruling, which this module implements literally:** wake at
launch is YES, and it is **mark-gated**. At startup do ONE cheap indexed
read; deliver only for a conversation that already carries a
`FLEET_UNSEEN` mark AND an owed `agent_runs` ledger row — i.e. only work
the user already started that already finished. It stays behind the
existing `[agents] autowake_enabled`; there is **no** separate
`autowake_on_launch` switch.

## When there are no marks, this constructs NOTHING

That is the overwhelmingly common case and it is pinned as such
(`Tests/UI/test_console_launch_wake.py::
test_a_launch_with_no_marks_constructs_nothing_and_reads_once`): the kill
switch is read, one indexed `conversation_local_marks` listing runs, and
the function returns. No `ConsoleChatStore`, no provider gateway, no
`ConsoleAgentBridge` — so no `agent_runs.db` file is even opened — and no
`ConsoleChatController`. Startup for a user who never uses the fleet is
byte-identical to before this module existed.

## Why the claim stays MARKS-indexed

Seeding from the ledger alone manufactures phantom wakes. A crash-killed
child is swept to `error` by `AgentRunsDB`'s reconcile pass, which makes
it a terminal, undelivered row — genuinely "owed" by the ledger's own
definition — but it carries **no mark**, because nothing ever settled it
through the fan-out. `undelivered_wake_runs` defines *what* is owed; the
mark defines *which conversations*. Both are needed, and this module asks
them in that order.

## Hydration, and the stale-mark case it exposes

A launch wake needs a session for a conversation nobody has opened, so
this module hydrates one through the shared
`Chat/console_conversation_hydration.py` policy — the same tree load, the
same whole-tree node build, the same active-leaf pointer and the same
roleplay overlay `ChatScreen`'s saved-conversation resume uses. Only the
BASE settings differ, and only because they must: the screen inherits the
active session's settings and a launch has none, so it starts from the
config defaults the screen itself falls back to.

Some marked conversations **cannot** be hydrated, ever. A fleet turn in an
UNSAVED (temporary) session is keyed by the ephemeral `session.id`
(`ConsoleChatController._agent_conversation_id` returns
`session.persisted_conversation_id or session_id`), and that id names no
ChaChaNotes conversation once the process that held the session is gone.
Executed, not assumed: the mark survives the restart and resolves to
nothing. Leaving it costs a permanent, undismissable row and a futile
hydration attempt on every launch for the rest of the install's life, so
this module CLEARS such a mark — but only when the local DB itself says
the conversation does not exist, and only for a mark that is otherwise
owed, i.e. one this module was about to act on.
"""

from __future__ import annotations

from typing import Any, Sequence

from loguru import logger

from tldw_chatbook.Chat.console_conversation_hydration import (
    ConversationLoadFailed,
    ConversationServiceUnavailable,
    apply_resume_settings_overrides,
    hydrate_console_session,
    load_console_conversation_tree,
)
from tldw_chatbook.Chat.console_fleet_attention import clear_fleet_unseen_completion
from tldw_chatbook.Chat.console_fleet_wake import autowake_enabled
from tldw_chatbook.Chat.console_runtime import ensure_console_runtime
from tldw_chatbook.Chat.console_session_settings import (
    default_console_session_settings,
)

__all__ = [
    "LAUNCH_WAKE_TASK_NAME",
    "deliver_launch_wakes",
    "marked_conversations_at_launch",
]

#: The name the app gives the deferred startup task, so a test (and a log
#: reader) can name it rather than matching on a coroutine repr.
LAUNCH_WAKE_TASK_NAME = "deferred_launch_wake"


def marked_conversations_at_launch(app: Any) -> tuple[str, ...]:
    """The one cheap indexed read a launch is allowed to pay for.

    Reads the `[agents] autowake_enabled` kill switch FIRST, so an install
    with auto-wake off pays nothing at all — not even the listing. An
    empty result must lead the caller to construct nothing; see this
    module's docstring.

    Args:
        app: The app object; read for `conversation_local_marks_service`.

    Returns:
        The conversation ids carrying a durable `FLEET_UNSEEN` mark, or an
        empty tuple (switch off / no marks service / a failed read — a
        launch must never die on this).
    """
    if not autowake_enabled():
        return ()
    service = getattr(app, "conversation_local_marks_service", None)
    if service is None:
        return ()
    try:
        return tuple(service.list_marked_conversation_ids(service.FLEET_UNSEEN))
    except Exception as exc:  # noqa: BLE001 -- a launch never dies on a claim
        logger.warning(
            "launch wake mark listing failed (exception_type={})",
            type(exc).__name__,
        )
        return ()


def _clear_unresolvable_mark(app: Any, conversation_id: str) -> None:
    """Drop a `FLEET_UNSEEN` mark that can never be resolved again.

    Only reached for a conversation the ledger still owes AND that the
    local ChaChaNotes DB has no row for — the ephemeral-session shape in
    this module's docstring. The owed `agent_runs` rows are deliberately
    left alone: they are stamped by delivery, never by give-up, and with
    the mark gone nothing indexes them, so nothing re-announces them.
    """
    logger.info(
        "launch wake: clearing an unresolvable ◈ mark (the conversation no "
        "longer exists locally; an unsaved chat's fleet work is keyed by a "
        "session id that dies with its process)"
    )
    clear_fleet_unseen_completion(app, conversation_id)


def _conversation_exists_locally(app: Any, conversation_id: str) -> bool:
    """Whether the local ChaChaNotes DB still has this conversation.

    Asked of the DB directly rather than inferred from a failed tree load:
    a tree load can fail for reasons that have nothing to do with the row
    existing (a scope service in server mode, a transient error), and
    clearing a live user's badge on one of those would be a real loss. A
    DB that cannot answer is treated as "exists", so uncertainty keeps the
    mark.
    """
    db = getattr(app, "chachanotes_db", None)
    getter = getattr(db, "get_conversation_by_id", None)
    if not callable(getter):
        return True
    try:
        return getter(conversation_id) is not None
    except Exception as exc:  # noqa: BLE001 -- uncertainty keeps the mark
        logger.debug(
            "launch wake conversation existence check raised; keeping the mark "
            "(exception_type={})",
            type(exc).__name__,
        )
        return True


def _ensure_launch_runtime(app: Any) -> Any:
    """Return the Console controller a launch delivery needs, building it.

    Mirrors `ChatScreen`'s own `_ensure_console_*` chain with the screen
    removed. The constructor arguments that a mounted Console derives from
    widget state are deliberately NOT reconstructed here: every screen-owned
    callable is a `CONSOLE_VIEW_HOOK_SLOTS` entry that
    `ensure_chat_controller` gives its viewless default (the runtime is
    "viewless from birth" — `console_runtime.py`), and the send-time
    provider selection comes from the SESSION's settings
    (`_provider_selection_for_session`), not from these fields. The first
    real Console mount re-applies the whole selection through
    `_sync_console_chat_core_state`, so nothing here is sticky.

    Every `ensure_*` is idempotent, so when Console IS the startup tab this
    returns the screen's own, fully-wired controller and every argument
    below is ignored — which is why the function is not named "viewless".

    Returns:
        The `ConsoleChatController`, or `None` when there is no durable
        ChaChaNotes DB to key an `AgentRunsDB` off (an in-memory harness) —
        in which case nothing could be owed in the first place.
    """
    runtime = ensure_console_runtime(app)
    app_config = getattr(app, "app_config", {}) or {}
    console_config = app_config.get("console", {})
    if not isinstance(console_config, dict):
        console_config = {}

    def _gate(name: str) -> bool:
        value = console_config.get(name, True)
        return bool(value) if isinstance(value, (bool, int)) else True

    store = runtime.ensure_chat_store()
    gateway = runtime.ensure_provider_gateway(
        config_provider=lambda: getattr(app, "app_config", {}) or {}
    )
    bridge = runtime.ensure_agent_bridge(
        store_factory=lambda: store,
        provider_gateway_factory=lambda: gateway,
        skills_service=getattr(app, "skills_scope_service", None),
        native_tools_enabled_factory=lambda: (lambda: _gate("native_tool_calls")),
    )
    if bridge is None:
        return None
    defaults = default_console_session_settings(app_config)
    controller = runtime.ensure_chat_controller(
        store=store,
        provider_gateway=gateway,
        provider=defaults.provider,
        model=defaults.model,
        base_url=defaults.base_url,
        agent_bridge=bridge,
        agent_runtime_enabled=_gate("agent_runtime"),
        skills_service=getattr(app, "skills_scope_service", None),
    )
    # Deliberately NOT a view-hook slot: this is the APP, which outlives
    # every view, and a headless approval round's `call_from_thread` bridge
    # (and its app-wide toast) needs it. `ChatScreen` sets the same handle
    # one line after its own construction call.
    controller.app = app
    return controller


def _restore_active_session(store: Any, session_id: str | None) -> None:
    """Put the active tab back where hydration found it.

    A launch with Console as the startup tab already has an active session
    -- the one the user is looking at. Hydrating a marked conversation
    activates the session it creates, so without this the wake would move
    the user off their tab while they watched. With nothing open (the
    headless launch) there is no prior session and the hydrated one stays
    active, which is the only sensible answer there.
    """
    if not session_id or getattr(store, "active_session_id", None) == session_id:
        return
    switch = getattr(store, "switch_session", None)
    if not callable(switch):
        return
    try:
        switch(session_id)
    except Exception as exc:  # noqa: BLE001 -- a gone session is not an error
        logger.debug(
            "launch wake could not restore the prior active session "
            "(exception_type={})",
            type(exc).__name__,
        )


async def deliver_launch_wakes(app: Any, marked: Sequence[str]) -> int:
    """Hydrate and fire every wake this launch already owed.

    Runs as a deferred startup task, after the first interactive frame.
    Every gate the mounted case passes still applies unchanged — this
    function seeds and hydrates, and `ConsoleFleetWakeCoordinator` does the
    rest: the kill switch again at the fire point, the send gate
    (`send_refusal_copy`: run state, queue ownership, `max_parallel_runs`),
    user-wins-ties, one delivery at a time app-wide, the `wake_delivered_at`
    ledger for exactly-once, and the viewless `wake_conversation_in_view`
    that keeps the ◈ mark on a delivery nobody watched.

    Args:
        app: The app object.
        marked: The conversation ids from `marked_conversations_at_launch`.
            An empty sequence returns immediately, having built nothing.

    Returns:
        How many conversations were hydrated for delivery.
    """
    if not marked:
        return 0
    controller = _ensure_launch_runtime(app)
    if controller is None:
        return 0
    wake = getattr(controller, "fleet_wake", None)
    if wake is None:
        return 0
    wake.wire(app=app)
    # The ledger read that turns "which conversations" into "what is owed".
    # Honours the kill switch itself, and drops anything an earlier process
    # already delivered (`_rows_for`'s stale drop / `undelivered_wake_runs`'
    # NULL filter), so a second launch re-announces nothing.
    if not wake.seed_from_marks():
        return 0
    store = controller.store
    app_config = getattr(app, "app_config", {}) or {}
    # `restore_persisted_session` ACTIVATES what it creates, which is right
    # for a launch with nothing open and wrong for one where Console is the
    # startup tab: a wake must never move the user off the tab they landed
    # on. Captured here and restored below.
    prior_active_session_id = getattr(store, "active_session_id", None)
    hydrated = 0
    for conversation_id in marked:
        if not wake.has_pending(conversation_id):
            continue  # marked, but nothing owed: a delivered-but-unseen badge
        if any(
            conversation_id
            in (session.persisted_conversation_id, session.id)
            for session in store.sessions()
        ):
            continue  # already open (a re-entrant call); the coordinator has it
        if not _conversation_exists_locally(app, conversation_id):
            _clear_unresolvable_mark(app, conversation_id)
            continue
        try:
            tree = await load_console_conversation_tree(app, conversation_id)
        except (ConversationServiceUnavailable, ConversationLoadFailed) as exc:
            logger.warning(
                "launch wake could not load a marked conversation; it stays "
                "staged for the next Console visit (exception_type={})",
                type(exc).__name__,
            )
            continue
        if tree is None:
            # The row vanished between the existence check and the load.
            _clear_unresolvable_mark(app, conversation_id)
            continue
        conversation = tree.get("conversation")
        if not isinstance(conversation, dict):
            conversation = {}
        try:
            hydrate_console_session(
                app=app,
                store=store,
                conversation_id=conversation_id,
                tree=tree,
                settings=apply_resume_settings_overrides(
                    default_console_session_settings(app_config), conversation
                ),
            )
        except Exception as exc:  # noqa: BLE001 -- one bad row never stops the rest
            logger.warning(
                "launch wake hydration failed for a marked conversation "
                "(exception_type={})",
                type(exc).__name__,
            )
            continue
        hydrated += 1
    if hydrated:
        _restore_active_session(store, prior_active_session_id)
        wake.retry_soon()
    return hydrated
