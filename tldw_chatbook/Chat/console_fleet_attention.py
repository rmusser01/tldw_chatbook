"""App-level attention surface for background sub-agent completions.

PR 3a-2 Task 4. A fleet child that outlives its turn (a SURVIVOR, PR 3a-1)
settles on its own daemon thread, possibly long after the Console screen --
the only surface that used to know about it -- was torn down. This module
owns everything that must still happen at that moment:

- a **durable unseen-completion mark** (``ConversationLocalMarksService.
  FLEET_UNSEEN``) written on the child's own thread, straight to the
  ChaChaNotes DB -- the one path Task 1 proved survives screen teardown and
  even a closing app loop;
- an **app-wide toast** naming the conversation and the honest outcome
  (``error``/``cancelled`` say so), fired via the app object after hopping
  to the loop captured at registration time (Task 3's captured-loop
  precedent) -- it renders on whatever screen the user is on;
- a **deep-link staging** (``HandoffChannel.CONSOLE_FLEET_COMPLETION``)
  when Console is not the active screen, so the next Console mount can
  claim it and switch to the settled conversation's session.

The consumer registers on ``ConsoleAgentBridge.on_fleet_drained`` (next to
bridge construction, per ``FleetDrainFanout.register``'s contract) and
filters each drain to the children whose ``settled_after_turn`` flag is
set: a child that finished INSIDE its turn is the turn's news, already
covered by the per-turn notify, and gets no mark and no toast. One drain
event produces at most ONE toast, however many children it carries.

The mark's clear seam is :func:`clear_fleet_unseen_completion` -- called
when the user views the conversation in Console, and by Task 5's delivery
when auto-wake hands the result to the supervisor. Ordering note for Task
5: the view-clear runs on the first tab sync after the conversation becomes
active, so a mount-claim that needs the mark as its undelivered bit must
read it BEFORE switching the session to active.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from loguru import logger

#: App attribute bumped on every durable mark write/clear, so screen-side
#: caches of the unseen set can re-read only when something changed rather
#: than hitting the DB on every 0.2s sync tick.
FLEET_UNSEEN_REVISION_ATTR = "_console_fleet_unseen_revision"


def _activity_receipt_service(app: Any) -> Any | None:
    runtime = getattr(app, "console_runtime", None)
    return getattr(runtime, "activity_receipts", None)


def bump_fleet_unseen_revision(app: Any) -> None:
    """Invalidate screen-side caches of the unseen-completion set.

    Args:
        app: The application object carrying the revision attribute.
    """
    try:
        setattr(
            app,
            FLEET_UNSEEN_REVISION_ATTR,
            int(getattr(app, FLEET_UNSEEN_REVISION_ATTR, 0)) + 1,
        )
    except Exception as exc:  # noqa: BLE001 -- a broken app double must not break a settle
        logger.debug(
            "fleet unseen revision bump failed (exception_type={})",
            type(exc).__name__,
        )


def fleet_unseen_conversation_ids(app: Any) -> frozenset[str]:
    """Conversation ids currently carrying the durable unseen mark.

    Args:
        app: The application object holding ``conversation_local_marks_service``.

    Returns:
        The marked ids, empty when the service is missing or the read fails
        (an indicator must never break a paint).
    """
    service = getattr(app, "conversation_local_marks_service", None)
    if service is None:
        return frozenset()
    try:
        return frozenset(service.list_marked_conversation_ids(service.FLEET_UNSEEN))
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "fleet unseen mark listing failed (exception_type={})",
            type(exc).__name__,
        )
        return frozenset()


def clear_fleet_unseen_completion(app: Any, conversation_id: str) -> bool:
    """Clear one conversation's unseen-completion mark (the named clear seam).

    The two legitimate callers, by design: the Console screen when the user
    VIEWS the conversation (it is no longer unseen), and Task 5's wake
    delivery (the result reached the supervisor).

    Args:
        app: The application object holding the marks service.
        conversation_id: The conversation to clear.

    Returns:
        True when a mark existed and was cleared; False when there was
        nothing to clear or the service is unavailable.
    """
    receipt_service = _activity_receipt_service(app)
    service = getattr(app, "conversation_local_marks_service", None)
    if service is None or not str(conversation_id or "").strip():
        return False
    if receipt_service is not None:
        try:
            cleared = receipt_service.clear_fleet_mark_if_seen(conversation_id)
        except Exception as exc:  # noqa: BLE001 - preserve a stale mark on uncertainty
            logger.warning(
                "fleet receipt reconciliation failed (exception_type={})",
                type(exc).__name__,
            )
            return False
        if cleared:
            bump_fleet_unseen_revision(app)
        return cleared
    try:
        if not service.has_mark(conversation_id, service.FLEET_UNSEEN):
            return False
        service.clear_mark(conversation_id, service.FLEET_UNSEEN)
    except Exception as exc:  # noqa: BLE001 -- a failed clear leaves a stale badge, not a crash
        logger.warning(
            "fleet unseen mark clear failed (exception_type={})",
            type(exc).__name__,
        )
        return False
    bump_fleet_unseen_revision(app)
    return True


def set_fleet_unseen_completion(app: Any, conversation_id: str) -> bool:
    """Set one conversation's unseen-completion mark (the named set seam).

    task-15971: the set-side sibling of :func:`clear_fleet_unseen_completion`.
    Its one production caller beyond the attention consumer's settle-time
    write: Task 5's delivery commit, when a wake turn completes while its
    conversation is NOT in view (the coordinator's design ruling -- a
    mounted-but-hidden Console delivers immediately, and the ◈ badge is
    how the user learns the result landed). Idempotent (the marks upsert),
    and bumps the badge revision so screen caches repaint.

    Args:
        app: The application object holding the marks service.
        conversation_id: The conversation to mark.

    Returns:
        True when the mark was written; False when the service is
        unavailable or the write failed (a lost mark is a missing badge,
        never a broken delivery).
    """
    receipt_service = _activity_receipt_service(app)
    service = getattr(app, "conversation_local_marks_service", None)
    if service is None or not str(conversation_id or "").strip():
        return False
    if receipt_service is not None:
        try:
            written = receipt_service.ensure_fleet_mark(conversation_id)
        except Exception as exc:  # noqa: BLE001 - never break delivery
            logger.warning(
                "fleet receipt reconciliation failed (exception_type={})",
                type(exc).__name__,
            )
            return False
        if written:
            bump_fleet_unseen_revision(app)
        return written
    try:
        service.set_mark(conversation_id, service.FLEET_UNSEEN)
    except Exception as exc:  # noqa: BLE001 -- a lost mark is a missing badge, not a crash
        logger.warning(
            "fleet unseen mark set failed (exception_type={})",
            type(exc).__name__,
        )
        return False
    bump_fleet_unseen_revision(app)
    return True


def fleet_completion_toast_copy(title: str, statuses: list[str]) -> tuple[str, str]:
    """Compose the one coalesced toast for a drain's survivor settles.

    Honest by construction: ``error``/``cancelled`` outcomes are named,
    never folded into "finished" (the ``notify_run_failure`` sibling's
    rule). N children of one conversation settling together produce one
    message, not N.

    Args:
        title: The conversation's display title.
        statuses: Terminal statuses of the drain's after-turn children.

    Returns:
        ``(message, severity)`` for ``app.notify``.
    """
    counts = Counter(statuses)
    failed = counts.get("error", 0)
    cancelled = counts.get("cancelled", 0)
    done = len(statuses) - failed - cancelled
    if failed:
        severity = "error"
    elif cancelled:
        severity = "warning"
    else:
        severity = "information"
    if len(statuses) == 1:
        if failed:
            verb = "failed"
        elif cancelled:
            verb = "was cancelled"
        else:
            verb = "finished"
        return (f"Background sub-agent {verb} in “{title}”.", severity)
    parts = []
    if done:
        parts.append(f"{done} finished")
    if failed:
        parts.append(f"{failed} failed")
    if cancelled:
        parts.append(f"{cancelled} cancelled")
    summary = ", ".join(parts)
    return (
        f"{len(statuses)} background sub-agents in “{title}”: {summary}.",
        severity,
    )


class ConsoleFleetAttentionConsumer:
    """The ``FleetDrained`` consumer behind the mark, the toast, and the
    deep link (PR 3a-2 Task 4).

    Runs under ``FleetDrainFanout``'s contract: the child's own daemon
    thread, possibly after the Console screen is gone -- DBs and
    thread-safe callables only. The durable mark is written INLINE on that
    thread (the ChaChaNotes DB is thread-local-connection safe and, unlike
    the captured loop, cannot be closed out from under an exiting app); the
    UI half -- revision bump, toast, deep-link staging -- hops to the loop
    captured at construction time, which in production is the app loop and
    outlives every screen (Task 3's precedent). With no loop (sync
    harnesses) or a closed one (app exit) the UI half runs inline as a
    best-effort last chance; ``app.notify`` is documented thread-safe, and
    the handoff store's own owner-thread assertion rejects a cross-thread
    stage harmlessly inside this consumer's catch.

    Idempotent per event for its durable effect: the mark upsert is an
    ``ON CONFLICT DO UPDATE``, so a re-delivered drain refreshes a
    timestamp rather than duplicating anything.
    """

    #: The fan-out registration name (also the replace key).
    NAME = "fleet-attention"

    def __init__(
        self,
        app: Any,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        receipt_service: Any | None = None,
    ):
        """Capture the app object and the loop to hop the UI half onto.

        Args:
            app: The application object (marks service, ``notify``,
                ``pending_handoffs``). App-lifetime; never a screen.
            loop: Explicit loop override for tests. When ``None``, the
                running loop at construction time is captured -- in
                production that is the app loop, because registration
                happens during bridge construction on the UI thread.
        """
        self._app = app
        self._receipt_service = receipt_service
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        self._loop = loop

    def __call__(self, event: Any) -> None:
        """Handle one drain: mark durably, then hop for the toast.

        Args:
            event: The ``FleetDrained`` event.
        """
        survivors = [
            child
            for child in (getattr(event, "children", ()) or ())
            if getattr(child, "settled_after_turn", False)
        ]
        if not survivors:
            return
        conversation_id = str(getattr(event, "conversation_id", "") or "")
        if not conversation_id:
            return
        supported_statuses = {"done", "error", "stuck", "cancelled"}
        statuses = [
            str(getattr(child, "status", "") or "")
            for child in survivors
            if str(getattr(child, "status", "") or "") in supported_statuses
        ]
        if not statuses:
            return
        if self._receipt_service is not None:
            publication = self._receipt_service.publish_fleet_drain(event)
            if not getattr(publication, "complete", False):
                try:
                    written = self._receipt_service.ensure_fleet_mark(conversation_id)
                except Exception as exc:  # noqa: BLE001 - retain the fallback path
                    logger.warning(
                        "fleet receipt fallback failed (exception_type={})",
                        type(exc).__name__,
                    )
                    written = False
                if not written:
                    self._write_mark(conversation_id)
        else:
            self._write_mark(conversation_id)
        session_id = str(getattr(survivors[-1], "session_id", "") or "")
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(
                    self._announce, conversation_id, session_id, statuses
                )
                return
            except RuntimeError:
                pass  # closed between the check and the call: fall through
        self._announce(conversation_id, session_id, statuses)

    # -- child-thread half ---------------------------------------------------

    def _write_mark(self, conversation_id: str) -> None:
        """Write the durable unseen mark; never raise into the fan-out."""
        service = getattr(self._app, "conversation_local_marks_service", None)
        if service is None:
            return
        try:
            service.set_mark(conversation_id, service.FLEET_UNSEEN)
        except Exception as exc:  # noqa: BLE001 -- a lost mark is a missing badge, not a broken settle
            logger.warning(
                "fleet unseen mark write failed (exception_type={})",
                type(exc).__name__,
            )

    # -- app-loop half -------------------------------------------------------

    def _announce(
        self, conversation_id: str, session_id: str, statuses: list[str]
    ) -> None:
        """Bump the badge revision, toast once, stage the deep link.

        Wrapped never-raise: as a ``call_soon_threadsafe`` callback an
        exception would land in the loop's exception handler; inline it
        would propagate into the fan-out's per-consumer catch.
        """
        try:
            bump_fleet_unseen_revision(self._app)
            message, severity = fleet_completion_toast_copy(
                self._conversation_title(conversation_id), statuses
            )
            notify = getattr(self._app, "notify", None)
            if callable(notify):
                notify(message, severity=severity)
            if not self._console_screen_active():
                self._stage_deep_link(conversation_id, session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "fleet completion announce failed (exception_type={})",
                type(exc).__name__,
            )

    def _conversation_title(self, conversation_id: str) -> str:
        """Best-effort display title: DB row, else an open session, else generic."""
        db = getattr(self._app, "chachanotes_db", None)
        get_conversation = getattr(db, "get_conversation_by_id", None)
        if callable(get_conversation):
            try:
                row = get_conversation(conversation_id)
                title = str((row or {}).get("title") or "").strip()
                if title:
                    return title
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "fleet toast title lookup failed (exception_type={})",
                    type(exc).__name__,
                )
        # An unpersisted session has no DB row; if a Console screen is up,
        # its store still knows the tab title.
        store = getattr(getattr(self._app, "screen", None), "_console_chat_store", None)
        sessions = getattr(store, "sessions", None)
        if callable(sessions):
            try:
                for session in sessions():
                    if conversation_id in (
                        session.id,
                        session.persisted_conversation_id,
                    ):
                        title = str(session.title or "").strip()
                        if title:
                            return title
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "fleet toast session-title fallback failed (exception_type={})",
                    type(exc).__name__,
                )
        return "an unsaved Console chat"

    def _console_screen_active(self) -> bool:
        """Whether the active screen is the Console (Chat) screen.

        Duck-typed on a Console-only seam rather than an import of
        ``ChatScreen`` (UI layer) from this Chat-layer module.
        """
        screen = getattr(self._app, "screen", None)
        return hasattr(screen, "_ensure_console_chat_controller")

    def _stage_deep_link(self, conversation_id: str, session_id: str) -> None:
        """Stage the mount-claimable deep link; owner-thread only.

        Import is local: ``pending_handoff_store`` (UI.Navigation) imports
        Chat models, so a module-level import here would be a cycle risk
        and would also drag UI into every Chat-layer test that touches
        this module.
        """
        store = getattr(self._app, "pending_handoffs", None)
        if store is None:
            return
        from tldw_chatbook.Chat.console_chat_models import (
            ConsoleFleetCompletionTarget,
        )
        from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

        store.stage(
            HandoffChannel.CONSOLE_FLEET_COMPLETION,
            ConsoleFleetCompletionTarget(
                conversation_id=conversation_id, session_id=session_id
            ),
        )


def register_fleet_attention(
    bridge: Any, app: Any, *, receipt_service: Any | None = None
) -> None:
    """Register the attention consumer on a bridge, next to its construction.

    Tolerates a bridge without the fan-out seam (older fakes) and no bridge
    at all -- registration must never be the thing that breaks bridge
    setup.

    Args:
        bridge: The ``ConsoleAgentBridge`` (or a double) to register on.
        app: The application object the consumer captures.
    """
    register = getattr(bridge, "on_fleet_drained", None) if bridge is not None else None
    if callable(register):
        register(
            ConsoleFleetAttentionConsumer.NAME,
            ConsoleFleetAttentionConsumer(app, receipt_service=receipt_service),
        )
