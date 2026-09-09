"""Console fleet lifecycle with explicit late-bound dependencies.

The app identity is stable for a screen visit. Framework scheduling services
are resolved from the screen on every access; DOM identity, user draft text,
and leaving the original view are named callable projections. No widget is
stored or queried by this controller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from loguru import logger

from ...Chat.console_chat_models import ConsoleFleetCompletionTarget, ConsoleRunMarker
from ...Chat.console_chat_store import ConsoleChatSession
from ...Chat.console_fleet_attention import (
    FLEET_UNSEEN_REVISION_ATTR,
    fleet_unseen_conversation_ids,
)
from ..Navigation.pending_handoff_store import HandoffChannel


class ConsoleFleetLifecycleController:
    """Own wake policy, durable unseen projections, and survivor repaint timing."""

    def __init__(
        self,
        screen: Any,
        *,
        app_instance: Any,
        ensure_console_chat_store: Callable[..., Any],
        ensure_console_agent_bridge: Callable[..., Any],
        ensure_console_chat_controller: Callable[..., Any],
        set_active_workspace_for_console_session: Callable[..., Any],
        sync_native_console_chat_ui: Callable[..., Any],
        record_ui_timer_created: Callable[..., Any],
        record_ui_timer_stopped: Callable[..., Any],
        start_console_transcript_sync_timer: Callable[..., Any],
        console_screen_displayed: Callable[..., Any],
        console_user_draft_text: Callable[..., Any],
        leave_console_runtime: Callable[..., Any],
        console_chat_controller_accessor: Callable[[], Any],
        console_chat_store_accessor: Callable[[], Any],
        console_transcript_sync_timer_accessor: Callable[[], Any],
    ) -> None:
        self._screen = screen
        self.app_instance = app_instance
        self._ensure_console_chat_store = ensure_console_chat_store
        self._ensure_console_agent_bridge = ensure_console_agent_bridge
        self._ensure_console_chat_controller = ensure_console_chat_controller
        self._set_active_workspace_for_console_session = (
            set_active_workspace_for_console_session
        )
        self._sync_native_console_chat_ui = sync_native_console_chat_ui
        self._record_ui_timer_created = record_ui_timer_created
        self._record_ui_timer_stopped = record_ui_timer_stopped
        self._start_console_transcript_sync_timer = start_console_transcript_sync_timer
        self._console_screen_displayed = console_screen_displayed
        self._console_user_draft_text = console_user_draft_text
        self._leave_console_runtime = leave_console_runtime
        self._console_chat_controller_accessor = console_chat_controller_accessor
        self._console_chat_store_accessor = console_chat_store_accessor
        self._console_transcript_sync_timer_accessor = (
            console_transcript_sync_timer_accessor
        )
        self._console_fleet_survivor_timer: Any | None = None
        self._console_fleet_unseen_cache: Any | None = None

    @property
    def run_worker(self) -> Any:
        return self._screen.run_worker

    @property
    def set_interval(self) -> Any:
        return self._screen.set_interval

    @property
    def call_later(self) -> Any:
        return self._screen.call_later

    @property
    def is_mounted(self) -> Any:
        return self._screen.is_mounted

    @property
    def _console_chat_controller(self) -> Any:
        return self._console_chat_controller_accessor()

    @property
    def _console_chat_store(self) -> Any:
        return self._console_chat_store_accessor()

    @property
    def _console_transcript_sync_timer(self) -> Any:
        return self._console_transcript_sync_timer_accessor()

    def consume_pending_console_fleet_completion(self) -> bool:
        """Claim a staged background sub-agent completion and switch to it.

        PR3a-2 Task 4: the fleet-attention consumer stages a
        ``ConsoleFleetCompletionTarget`` while Console is NOT the active
        screen; this claim (mount + resume, 0.15s settle hedge like its
        sibling handoff claims) switches the store to the settled
        conversation's still-open session so the user lands on the news
        the toast announced. A target whose session is no longer open is
        acknowledged and dropped -- the durable ``fleet_unseen`` mark (and
        the sidebar badge it drives) still points at the conversation, and
        wake delivery discovers the durable result ledger independently of
        this channel, so nothing is lost by not force-resuming here.

        Returns:
            True when a target was claimed and its session activated.
        """
        claim = self.app_instance.pending_handoffs.claim(
            HandoffChannel.CONSOLE_FLEET_COMPLETION
        )
        if claim is None:
            return False
        try:
            target = claim.value
            if not isinstance(target, ConsoleFleetCompletionTarget):
                raise TypeError("Console fleet completion handoff was not typed")
            store = self._ensure_console_chat_store()
            match = None
            for session in store.sessions():
                if target.session_id and session.id == target.session_id:
                    match = session
                    break
                if target.conversation_id in (
                    session.id,
                    session.persisted_conversation_id,
                ):
                    match = session
            if match is None:
                # Session closed since the toast: the badge/mark remains
                # the durable pointer; nothing to switch to.
                self.app_instance.pending_handoffs.acknowledge(claim)
                return False
            if store.active_session_id != match.id:
                controller = self._ensure_console_chat_controller()
                self._set_active_workspace_for_console_session(match.id)
                controller.switch_session(match.id)
                self.run_worker(
                    self._sync_native_console_chat_ui(),
                    exclusive=True,
                    group="console-sync",
                )
        except Exception as exc:  # noqa: BLE001 -- release for retry, never crash a mount
            self.app_instance.pending_handoffs.release(claim)
            logger.warning(
                "Console fleet completion handoff will retry "
                "(revision={}, exception_category={})",
                claim.revision,
                type(exc).__name__,
            )
            return False
        self.app_instance.pending_handoffs.acknowledge(claim)
        return True

    def _claim_console_fleet_wake_marks(self) -> None:
        """Discover saved results independently of attention badges (ADR-135).

        Mount/view attachment seeds history only. Startup recovery belongs to
        the runtime and must never invalidate a live owner during a remount.
        History reads run on a worker; viewing may clear a badge at any time.
        """
        try:
            if self._ensure_console_agent_bridge() is None:
                return
            controller = self._ensure_console_chat_controller()
            wake = getattr(controller, "fleet_wake", None)
            if wake is None:
                return
            wake.wire(app=self.app_instance)
            self.run_worker(
                self._seed_console_fleet_wake_history(wake),
                exclusive=True,
                group="console-fleet-seed",
            )
        except Exception as exc:  # noqa: BLE001 -- a failed claim must never break a mount
            logger.warning(
                "console fleet wake mount-claim failed (exception_type={})",
                type(exc).__name__,
            )

    async def _seed_console_fleet_wake_history(self, wake: Any) -> None:
        """Seed the owning coordinator without blocking the view's event loop."""
        try:
            if await asyncio.to_thread(wake.seed_from_marks):
                wake.retry_soon()
        except Exception as exc:  # noqa: BLE001 -- saved history can be retried
            logger.warning(
                "console fleet wake history seed failed (exception_type={})",
                type(exc).__name__,
            )

    def _console_wake_user_priority(self, session_id: str) -> bool:
        """User-wins-ties probe for the auto-wake coordinator.

        True while the Console composer holds a non-empty draft -- for ANY
        session, deliberately: the composer is the user's live claim on
        sending, it clears only once a manual send is ACCEPTED (so this
        also covers the dispatch gap between pressing Send and the run
        state turning busy), and under cap contention a wake for one
        session can cost another session's user their slot. A raising
        probe defers too (coordinator-side: user wins on uncertainty).
        The wake is retried when the composer empties
        (``_on_console_composer_draft_changed``'s poke) and on every
        terminal run-state transition.

        task-15970: "the Console composer" means the one the user can
        actually TYPE into. This screen can outlive its display (a
        navigation issued while a pushed screen sat above it pops the
        MODAL off the stack and leaves this screen resident-but-hidden --
        the residue arc's live ``mounted=True`` state), and the live bug
        was exactly this probe reading the hidden screen's own empty
        composer while the user held a typed draft in the DISPLAYED
        screen's: the wake fired straight through it, twice. When the
        displayed screen is a different Console screen, ITS composer is
        the user's hands; this screen's own composer is the fallback
        (nobody can type into any composer while e.g. Library is
        displayed, and a stale non-empty draft deferring is the probe's
        conservative direction).

        Args:
            session_id: The session the wake would fire into (unused by
                the any-session rule; part of the probe contract).

        Returns:
            Whether the user currently holds a sending claim.
        """
        return bool(self._console_user_draft_text().strip())

    def _console_wake_conversation_in_view(
        self, conversation_id: str, session_id: str
    ) -> bool:
        """Delivery-commit visibility probe (task-15971).

        True only when this screen is the DISPLAYED one and the wake's
        session is the ACTIVE session -- i.e. the user actually watched
        the result land. Anything else (Library displayed, a resident
        hidden Console, a non-active session tab) is off-view: the
        coordinator leaves the FLEET_UNSEEN mark set so the ◈ badge
        points at the delivered result until the user views it.

        Args:
            conversation_id: The delivered conversation (unused: the
                active-session comparison already scopes the view).
            session_id: The session the wake turn ran in.

        Returns:
            Whether the delivery landed in the user's view.
        """
        if not self._console_screen_displayed():
            return False
        store = getattr(self, "_console_chat_store", None)
        active = getattr(store, "active_session_id", None)
        return active is not None and active == session_id

    def _poke_console_wake_retry(self) -> None:
        """Retry a staged auto-wake (task-15864 AC#2: session-open trigger).

        Called after a persisted conversation is resumed into a native
        session -- the moment a mount-claimed pending wake finally has an
        open session to deliver into. Session tabs do not restore across
        restart, so before this trigger a restart-staged wake sat pending
        until an unrelated composer keystroke. getattr-guarded like every
        wake seam here (UI tests swap controller doubles).
        """
        controller = getattr(self, "_console_chat_controller", None)
        wake = getattr(controller, "fleet_wake", None)
        retry = getattr(wake, "retry_soon", None)
        if callable(retry):
            retry()

    def _on_console_wake_delivery_started(self, session_id: str) -> None:
        """Arm the transcript poll for a machine-injected wake turn.

        task-15862: manual sends arm the 0.2s poll in
        ``_submit_console_native_draft``; a wake turn bypasses that worker
        entirely (``ConsoleFleetWakeCoordinator._deliver`` calls
        ``controller.submit_draft`` directly), so before this hook nothing
        repainted the wake turn's stream, its terminal tab glyph, or the
        composer state until the user interacted. Runs on the app loop
        (the coordinator's ``_attempt`` thread), with the coordinator's
        ``_delivering`` already set -- so the poll's wake-delivery stop
        guard holds it alive through the scheduling gap. The poll still
        self-stops at the wake turn's terminal edge (15664 AC#2: no
        recurring idle repaint).

        Args:
            session_id: The session the wake turn fires into (unused; the
                poll repaints every session's surfaces).
        """
        if not self.is_mounted:
            return
        # Hop through the message pump before creating the timer. Textual's
        # ``Timer._tick`` reads the ``active_app`` ContextVar, and an
        # asyncio task inherits the context it was CREATED in -- this hook
        # can run in a bare ``call_soon_threadsafe`` callback context (the
        # coordinator's drain intake hops from the child's thread, whose
        # copied context has no active_app), where a directly-created
        # timer's task dies on its first tick without ever beating.
        # Observed live (task-15862 diagnosis): "arm-poll" logged, zero
        # beats, transcript frozen through the whole wake turn. A
        # ``Callback`` message runs inside the pump's own task, which
        # carries the app context, so the timer it creates ticks.
        self.call_later(self._start_console_transcript_sync_timer)

    def _console_wake_turn_active(self, session_id: str | None) -> bool:
        """Whether the auto-wake coordinator is delivering into ``session_id``.

        task-15862 AC#3: the composer's blocked-state copy must name the
        actual blocker during a wake turn. getattr-guarded throughout --
        several UI tests swap in hand-built controller doubles.

        Args:
            session_id: The session whose composer state is being synced.

        Returns:
            True while a wake delivery targets this session's conversation.
        """
        if not session_id:
            return False
        controller = getattr(self, "_console_chat_controller", None)
        wake = getattr(controller, "fleet_wake", None)
        delivering_read = getattr(wake, "delivering_session_ids", None)
        delivering = delivering_read() if callable(delivering_read) else ()
        return session_id in delivering

    async def _record_console_fleet_teardown(self, controller: Any) -> None:
        """Snapshot this teardown's true fates, LEAVE, stage the notice.

        TASK-1143 (F5) + PR3a-2 Task 4: snapshot BEFORE the teardown,
        using ``fleet_teardown_split()`` -- the same union
        ``busy_fleet_session_count`` (and the pre-navigate confirm) has
        always counted, partitioned by what actually happens next.
        Sessions with an in-flight turn or pending approval are killed by
        the teardown below; sessions whose only work is a cross-turn
        survivor KEEP RUNNING through it (Task 1 A1, executed) and their
        results/spend land after the screen is gone. The app (not this
        doomed screen) holds both counts so the NEXT Console mount -- a
        fresh instance; screens are never cached -- reports each
        truthfully via ``_notify_console_fleet_teardown_if_any``.

        task-15860: the teardown is now ``leave_console_runtime`` -- this
        VISIT ends, the runtime does not. The provider gateway is no longer
        closed here either; it is app-owned and closes at
        ``ConsoleRuntime.dispose``. An in-flight ``AGENT_WAKE`` turn is
        exempt from the cancellation (owner ruling), so the ``killed``
        count can over-report by one in the rare case a wake turn is
        mid-flight at nav-away; ``fleet_teardown_split``'s own contract is
        deliberately left untouched.

        Qodo audit S1 (PR 1680): the staging is gated on
        ``leave_console_runtime``'s return. On an overlapping
        ChatScreen→ChatScreen navigation (a fleet-completion deep link
        clicked while already on Console), the incoming screen claims the
        runtime in ``restore_state`` BEFORE this screen unmounts, so this
        superseded screen's leave is a designed no-op (``ConsoleRuntime.
        detach_view``) and the sessions keep running under the successor.
        Staging unconditionally toasted "N session(s) cancelled when you
        left Console" for work that was never cancelled -- and never left
        Console. A leave that did not end the visit reports nothing.
        """
        killed, surviving = controller.fleet_teardown_split()
        ended = await self._leave_console_runtime()
        if not ended:
            return
        if killed:
            self.app_instance._console_fleet_teardown_notice = killed
        if surviving:
            self.app_instance._console_fleet_survivor_notice = surviving

    def _console_fleet_unseen_ids(self) -> frozenset[str]:
        """Conversation ids carrying the durable unseen-completion mark.

        PR3a-2 Task 4. Cached against the app-level revision counter the
        fleet-attention consumer bumps on every mark write/clear, so the
        0.2s sync tick pays a DB read only when something actually changed
        (the TASK-251 discipline) -- and the badge still survives restart,
        because a fresh screen's first read comes from the DB.
        """
        app = self.app_instance
        revision = getattr(app, FLEET_UNSEEN_REVISION_ATTR, 0)
        cache = getattr(self, "_console_fleet_unseen_cache", None)
        if cache is not None and cache[0] == revision:
            return cache[1]
        ids = fleet_unseen_conversation_ids(app)
        self._console_fleet_unseen_cache = (revision, ids)
        return ids

    def _console_run_marker_with_unseen(
        self,
        controller: Any,
        session: ConsoleChatSession,
        unseen_ids: frozenset[str],
    ) -> ConsoleRunMarker:
        """A session's fleet marker, backed by the durable unseen mark.

        PR3a-2 Task 4: ``run_marker_for``'s derivation is untouched -- any
        live or unvisited TURN state it reports outranks this -- but a
        session whose conversation carries the ``fleet_unseen`` mark and
        would otherwise show nothing gets ``SUBAGENT_UNSEEN``. Derived in
        the screen layer because the mark lives in an app-level service the
        controller deliberately has no handle on.
        """
        marker = controller.run_marker_for(session.id)
        conversation_id = session.persisted_conversation_id or session.id
        if marker is ConsoleRunMarker.NONE and conversation_id in unseen_ids:
            return ConsoleRunMarker.SUBAGENT_UNSEEN
        return marker

    def _console_fleet_survivors_live(self) -> bool:
        """Whether any live session's fleet still owes a drain."""
        controller = self._console_chat_controller
        checker = (
            getattr(controller, "fleet_has_unsettled_children", None)
            if controller is not None
            else None
        )
        try:
            return bool(checker()) if callable(checker) else False
        except Exception as exc:  # noqa: BLE001 -- a timer predicate must never raise
            logger.debug(
                "fleet survivor check failed (exception_type={})",
                type(exc).__name__,
            )
            return False

    def _maybe_start_console_fleet_survivor_tick(self) -> None:
        """Arm the 1s survivor tick when survivors are live and it is not.

        Called from the transcript poll's self-stop edge (the state
        task-15664 describes: only survivors run, nothing else repaints)
        and, as a mount hedge, shortly after ``on_mount``. Idempotent; a
        no-op with no live survivors, so an idle Console never gains a
        timer (15664 AC#2).
        """
        if self._console_fleet_survivor_timer is not None:
            return
        if not self._console_fleet_survivors_live():
            return
        self._console_fleet_survivor_timer = self.set_interval(
            1.0, self._console_fleet_survivor_tick
        )
        self._record_ui_timer_created("console-fleet-survivor-tick")

    def _stop_console_fleet_survivor_tick(self) -> None:
        if self._console_fleet_survivor_timer is None:
            return
        try:
            self._console_fleet_survivor_timer.stop()
        finally:
            self._record_ui_timer_stopped("console-fleet-survivor-tick")
            self._console_fleet_survivor_timer = None

    async def _console_fleet_survivor_tick(self) -> None:
        """One survivor-tick beat: repaint, or stop when nothing is live.

        While the 0.2s transcript poll is running it already repaints
        everything this would (at 5x the cadence), so the beat is skipped
        rather than doubled. When the last child has settled, the tick
        stops itself FIRST and then paints once more -- that final pass is
        what flips the rail rows to their terminal glyphs and surfaces the
        unseen badge without any user interaction; it is a settle paint,
        not a recurring repaint of an idle rail (15664 AC#2).
        """
        if self._console_transcript_sync_timer is not None:
            return
        controller = self._console_chat_controller
        if controller is None:
            self._stop_console_fleet_survivor_tick()
            return
        if not self._console_fleet_survivors_live():
            self._stop_console_fleet_survivor_tick()
            await self._sync_native_console_chat_ui()
            return
        await self._sync_native_console_chat_ui()
