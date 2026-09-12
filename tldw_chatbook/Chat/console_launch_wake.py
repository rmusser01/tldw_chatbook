"""Recover and hydrate saved native fleet results at application startup.

ADR-135 makes run history the discovery index; attention badges are a view
projection. Read only an existing sibling runs database before constructing any
Console components. Native runtime creation owns the single recovery audit,
which must finish before launch hydration or automatic admission proceeds.
Claims with uncertain outcomes remain saved for review, never replayed.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_chatbook.Agents.agent_models import TERMINAL_RUN_STATUSES
from tldw_chatbook.Chat.console_conversation_hydration import (
    ConversationLoadFailed,
    ConversationServiceUnavailable,
    hydrate_console_generation_settings,
    hydrate_console_session,
    load_console_conversation_tree,
)
from tldw_chatbook.Chat.console_fleet_attention import clear_fleet_unseen_completion
from tldw_chatbook.Chat.console_fleet_wake import autowake_enabled
from tldw_chatbook.Chat.console_runtime import ensure_console_runtime
from tldw_chatbook.Chat.console_session_settings import (
    default_console_session_settings,
)
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

__all__ = [
    "LAUNCH_WAKE_TASK_NAME",
    "deliver_launch_wakes",
    "marked_conversations_at_launch",
    "pending_conversations_at_launch",
]

#: The name the app gives the deferred startup task, so a test (and a log
#: reader) can name it rather than matching on a coroutine repr.
LAUNCH_WAKE_TASK_NAME = "deferred_launch_wake"


def pending_conversations_at_launch(app: Any) -> tuple[str, ...]:
    """Discover pending native survivor identities without creating a database.

    Includes claimed results awaiting recovery or review, regardless of badges.
    The private read-only connection performs no migration or run reconciliation.
    Historical schemas without delivery stamps expose survivor candidates so the
    native database owner can upgrade them before recovery. Discovery itself
    grants no automatic authority; chainless results require manual review.
    Disabled automatic work, memory databases, absent files, and failed reads
    return no launch candidates. Explicit Console startup still owns recovery.
    """
    if not autowake_enabled():
        return ()
    db = getattr(app, "chachanotes_db", None)
    db_path = getattr(db, "db_path", None)
    if not db_path or str(db_path) == ":memory:" or str(db_path).startswith("file:"):
        return ()
    try:
        path = Path(db_path).parent / "agent_runs.db"
        if not path.is_file():
            return ()
        with closing(
            connect_private_sqlite(
                "chat.launch_wake", path, read_only=True, must_exist=True
            )
        ) as connection:
            # Inspect actual columns, as AgentRunsDB's guarded migrations do.
            # Historical survivors must reach that owner before a new delivery
            # column can exist. This read only discovers candidates: subsequent
            # runtime recovery still refuses automatic work without lineage.
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(agent_runs)")
            }
            if not columns:
                return ()
            delivery_filter = (
                "child.wake_delivered_at IS NULL AND "
                if "wake_delivered_at" in columns
                else ""
            )
            # Same survivor predicate as AgentRunsDB.pending_wake_conversation_ids.
            # Do not construct AgentRunsDB here: its startup sweep mutates runs.
            rows = connection.execute(
                "SELECT DISTINCT child.conversation_id FROM agent_runs AS child "
                "JOIN agent_runs AS parent ON parent.id=child.parent_run_id "
                f"WHERE {delivery_filter}child.agent_kind!='primary' "
                "AND child.status IN ('done','error','cancelled') "
                f"AND parent.status IN ({', '.join('?' for _ in TERMINAL_RUN_STATUSES)}) "
                "AND child.updated_at>=parent.updated_at ORDER BY child.conversation_id",
                tuple(sorted(TERMINAL_RUN_STATUSES)),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)
    except Exception as exc:  # noqa: BLE001 -- a launch never dies on discovery
        logger.warning(
            "launch wake result discovery failed (exception_type={})",
            type(exc).__name__,
        )
        return ()


def marked_conversations_at_launch(app: Any) -> tuple[str, ...]:
    """Compatibility name for durable result discovery; marks are not consulted."""
    return pending_conversations_at_launch(app)


def _clear_unresolvable_mark(app: Any, conversation_id: str) -> None:
    """Drop a `FLEET_UNSEEN` mark that can never be resolved again.

    Only reached for a conversation the ledger still owes AND that the
    local ChaChaNotes DB has no row for, such as an unsaved session from a
    previous process. Saved run rows and execution claims remain
    available for review; clearing attention cannot authorize their replay.
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

    def _gate(name: str) -> bool:
        """One `[console]` boolean gate, read FRESH.

        Re-reads `app.app_config` per call rather than closing over a
        snapshot: `native_tools_enabled` is stored by the bridge and called
        much later, and `ChatScreen`'s own factory
        (`_console_native_tool_calls_enabled`) re-reads too. A launch-built
        bridge that never sees a Console mount would otherwise hold the
        boot-time answer for the whole run.
        """
        config = getattr(app, "app_config", {}) or {}
        section = config.get("console", {})
        if not isinstance(section, dict):
            section = {}
        value = section.get(name, True)
        return bool(value) if isinstance(value, (bool, int)) else True

    store = runtime.ensure_chat_store()
    gateway = runtime.ensure_provider_gateway(
        config_provider=lambda: getattr(app, "app_config", {}) or {}
    )
    bridge = runtime.ensure_agent_bridge(
        store_factory=lambda: store,
        provider_gateway_factory=lambda: gateway,
        skills_service=getattr(app, "skills_scope_service", None),
        native_tools_enabled_factory=lambda: lambda: _gate("native_tool_calls"),
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


async def deliver_launch_wakes(app: Any, marked: Sequence[str] = ()) -> int:
    """Recover, then hydrate saved results for bounded automatic follow-up.

    Runs after the first interactive frame. Discovery is rechecked before
    creating Console components. The coordinator owns durable claims, finite
    allowances, capacity, and user-priority gates; interrupted claims remain
    paused for review. A missing badge never hides a pending result.

    Args:
        app: The app object.
        marked: Legacy discovery snapshot, retained for call compatibility.
            Current durable results determine which conversations are loaded.

    Returns:
        How many conversations were hydrated for delivery.
    """
    pending = pending_conversations_at_launch(app)
    if not pending:
        return 0
    controller = _ensure_launch_runtime(app)
    if controller is None:
        return 0
    wake = getattr(controller, "fleet_wake", None)
    if wake is None:
        return 0
    wake.wire(app=app)
    if not await wake.wait_for_recovery():
        return 0
    # Runtime startup has audited old owners before any launch seed or retry.
    if not wake.seed_from_marks():
        return 0
    # Claim the pre-hydration coarse marks first: a genuine pre-v15 owed wake
    # has no receipt to reconstruct and startup reconciliation may correctly
    # remove that legacy-only mark.  Once the exact owed ledger rows are held
    # in the wake registry, hydrate before delivery so every later clear/set
    # request makes its decision against a ready (or explicitly degraded)
    # receipt snapshot.
    runtime = getattr(app, "console_runtime", None)
    ensure_hydration = getattr(runtime, "ensure_activity_hydration", None)
    if callable(ensure_hydration):
        hydration = ensure_hydration()
        if hydration is not None:
            try:
                await hydration
            except Exception as exc:  # noqa: BLE001 - degraded receipts keep marks
                logger.warning(
                    "launch activity hydration failed (exception_type={})",
                    type(exc).__name__,
                )
    store = controller.store
    # `restore_persisted_session` ACTIVATES what it creates, which is right
    # for a launch with nothing open and wrong for one where Console is the
    # startup tab: a wake must never move the user off the tab they landed
    # on. Captured here and restored below.
    prior_active_session_id = getattr(store, "active_session_id", None)
    hydrated = 0
    for conversation_id in pending:
        if not wake.has_pending(conversation_id):
            continue  # Another caller completed the result after discovery.
        if any(
            conversation_id in (session.persisted_conversation_id, session.id)
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
                "launch wake could not load a pending conversation; it stays "
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
            hydration = hydrate_console_generation_settings(
                getattr(app, "app_config", {}) or {},
                conversation,
            )
            await hydrate_console_session(
                app=app,
                store=store,
                conversation_id=conversation_id,
                tree=tree,
                settings=hydration.settings,
                generation_durable_snapshot=hydration.durable_snapshot,
                generation_metadata_status=hydration.metadata_status,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad row never stops the rest
            logger.warning(
                "launch wake hydration failed for a pending conversation "
                "(exception_type={})",
                type(exc).__name__,
            )
            continue
        hydrated += 1
    if hydrated:
        _restore_active_session(store, prior_active_session_id)
        wake.retry_soon()
    return hydrated
