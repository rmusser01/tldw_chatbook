"""Console fleet completion, wake, marker, teardown, and timer policy."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleFleetCompletionTarget,
    ConsoleRunMarker,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from textual.timer import Timer
    from textual.worker import Worker

    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import (
        ConsoleChatSession,
        ConsoleChatStore,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import PendingHandoffStore


class ConsoleFleetLifecycleController:
    """Own Console fleet lifecycle policy without owning screen or DOM state."""

    def __init__(
        self,
        *,
        pending_handoffs_accessor: Callable[[], PendingHandoffStore],
        ensure_chat_store: Callable[[], ConsoleChatStore],
        ensure_chat_controller: Callable[[], ConsoleChatController],
        activate_workspace_for_session: Callable[[str], None],
        switch_chat_session: Callable[[str], ConsoleChatSession],
        schedule_native_console_sync: Callable[[], Worker[None]],
        ensure_agent_bridge: Callable[[], ConsoleAgentBridge | None],
        wire_wake_coordinator: Callable[[], bool],
        seed_wake_from_marks: Callable[[], bool],
        retry_wake_soon: Callable[[], None],
        wake_has_pending: Callable[[str], bool],
        wake_delivering_conversation_id: Callable[[], str | None],
        displayed_composer_draft_accessor: Callable[[], str | None],
        screen_displayed_accessor: Callable[[], bool],
        screen_mounted_accessor: Callable[[], bool],
        active_session_id_accessor: Callable[[], str | None],
        chat_sessions_accessor: Callable[[], tuple[ConsoleChatSession, ...]],
        defer_on_message_pump: Callable[[Callable[[], None]], bool],
        start_transcript_sync_timer: Callable[[], None],
        transcript_sync_timer_active: Callable[[], bool],
        sync_native_console_ui: Callable[[], Awaitable[None]],
        create_interval: Callable[[float, Callable[[], Awaitable[None]]], Timer],
        record_timer_created: Callable[[str], None],
        record_timer_stopped: Callable[[str], None],
        chat_controller_available: Callable[[], bool],
        fleet_has_unsettled_children: Callable[[], bool],
        run_marker_for_session: Callable[[str], ConsoleRunMarker],
        fleet_teardown_split: Callable[[], tuple[int, int]],
        leave_runtime: Callable[[], Awaitable[bool]],
        stage_teardown_notices: Callable[[int, int], tuple[None, None]],
        fleet_unseen_revision_accessor: Callable[[], int],
        read_fleet_unseen_ids: Callable[[], frozenset[str]],
        clear_fleet_unseen: Callable[[str], bool],
    ) -> None:
        self._pending_handoffs_accessor = pending_handoffs_accessor
        self._ensure_chat_store = ensure_chat_store
        self._ensure_chat_controller = ensure_chat_controller
        self._activate_workspace_for_session = activate_workspace_for_session
        self._switch_chat_session = switch_chat_session
        self._schedule_native_console_sync = schedule_native_console_sync
        self._ensure_agent_bridge = ensure_agent_bridge
        self._wire_wake_coordinator = wire_wake_coordinator
        self._seed_wake_from_marks = seed_wake_from_marks
        self._retry_wake_soon = retry_wake_soon
        self._wake_has_pending = wake_has_pending
        self._wake_delivering_conversation_id = wake_delivering_conversation_id
        self._displayed_composer_draft_accessor = displayed_composer_draft_accessor
        self._screen_displayed_accessor = screen_displayed_accessor
        self._screen_mounted_accessor = screen_mounted_accessor
        self._active_session_id_accessor = active_session_id_accessor
        self._chat_sessions_accessor = chat_sessions_accessor
        self._defer_on_message_pump = defer_on_message_pump
        self._start_transcript_sync_timer = start_transcript_sync_timer
        self._transcript_sync_timer_active = transcript_sync_timer_active
        self._sync_native_console_ui = sync_native_console_ui
        self._create_interval = create_interval
        self._record_timer_created = record_timer_created
        self._record_timer_stopped = record_timer_stopped
        self._chat_controller_available = chat_controller_available
        self._fleet_has_unsettled_children = fleet_has_unsettled_children
        self._run_marker_for_session = run_marker_for_session
        self._fleet_teardown_split = fleet_teardown_split
        self._leave_runtime = leave_runtime
        self._stage_teardown_notices = stage_teardown_notices
        self._fleet_unseen_revision_accessor = fleet_unseen_revision_accessor
        self._read_fleet_unseen_ids = read_fleet_unseen_ids
        self._clear_fleet_unseen = clear_fleet_unseen

        self._console_fleet_survivor_timer: Any | None = None
        self._console_fleet_unseen_cache: tuple[int, frozenset[str]] | None = None

    def consume_pending_console_fleet_completion(self) -> bool:
        """Claim a staged completion and activate its still-open session.

        Returns:
            ``True`` when a matching completion was acknowledged; otherwise ``False``.
        """
        pending_handoffs = self._pending_handoffs_accessor()
        claim = pending_handoffs.claim(HandoffChannel.CONSOLE_FLEET_COMPLETION)
        if claim is None:
            return False
        try:
            target = claim.value
            if not isinstance(target, ConsoleFleetCompletionTarget):
                raise TypeError("Console fleet completion handoff was not typed")
            store = self._ensure_chat_store()
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
                pending_handoffs.acknowledge(claim)
                return False
            if store.active_session_id != match.id:
                self._ensure_chat_controller()
                self._activate_workspace_for_session(match.id)
                self._switch_chat_session(match.id)
                self._schedule_native_console_sync()
        except Exception as exc:  # noqa: BLE001 -- release for retry
            pending_handoffs.release(claim)
            logger.warning(
                "Console fleet completion handoff will retry "
                "(revision={}, exception_category={})",
                claim.revision,
                type(exc).__name__,
            )
            return False
        pending_handoffs.acknowledge(claim)
        return True

    def _claim_console_fleet_wake_marks(self) -> None:
        """Synchronously seed staged wakes from an uncached durable read."""
        try:
            marked = self._read_fleet_unseen_ids()
            if not marked:
                return
            if self._ensure_agent_bridge() is None:
                return
            if not self._wire_wake_coordinator():
                return
            if self._seed_wake_from_marks():
                self._retry_wake_soon()
        except Exception as exc:  # noqa: BLE001 -- never break mount
            logger.warning(
                "console fleet wake mount-claim failed (exception_type={})",
                type(exc).__name__,
            )

    def _console_wake_user_priority(self, session_id: str) -> bool:
        """Return whether the displayed Console composer holds a draft."""
        del session_id
        draft = self._console_wake_probe_composer()
        return bool(draft and draft.strip())

    def _console_wake_probe_composer(self) -> str | None:
        """Read the plain draft value selected by the wiring adapter."""
        return self._displayed_composer_draft_accessor()

    def _console_screen_displayed(self) -> bool:
        """Return whether this Console screen is currently displayed."""
        return bool(self._screen_displayed_accessor())

    def _console_wake_conversation_in_view(
        self,
        conversation_id: str,
        session_id: str,
    ) -> bool:
        """Return whether a wake delivery targets the displayed active tab."""
        del conversation_id
        if not self._console_screen_displayed():
            return False
        active_session_id = self._active_session_id_accessor()
        return active_session_id is not None and active_session_id == session_id

    def _poke_console_wake_retry(self) -> None:
        """Ask the wake coordinator to retry a staged delivery."""
        self._retry_wake_soon()

    def _on_console_wake_delivery_started(self, session_id: str) -> None:
        """Arm transcript syncing from inside Textual's message pump."""
        del session_id
        if not self._screen_mounted_accessor():
            return
        self._defer_on_message_pump(self._start_transcript_sync_timer)

    def _console_wake_turn_active(self, session_id: str | None) -> bool:
        """Return whether a wake is delivering into ``session_id``."""
        if not session_id:
            return False
        delivering = self._wake_delivering_conversation_id()
        if delivering is None:
            return False
        session = next(
            (item for item in self._chat_sessions_accessor() if item.id == session_id),
            None,
        )
        if session is None:
            return False
        return delivering in (session.persisted_conversation_id, session.id)

    async def _record_console_fleet_teardown(self) -> None:
        """Snapshot fleet fates, leave the runtime, then stage notices."""
        killed, surviving = self._fleet_teardown_split()
        ended = await self._leave_runtime()
        if not ended:
            return
        self._stage_teardown_notices(killed, surviving)

    def _console_fleet_unseen_ids(self) -> frozenset[str]:
        """Return durable unseen IDs cached against their service revision."""
        revision = self._fleet_unseen_revision_accessor()
        cache = self._console_fleet_unseen_cache
        if cache is not None and cache[0] == revision:
            return cache[1]
        ids = self._read_fleet_unseen_ids()
        self._console_fleet_unseen_cache = (revision, ids)
        return ids

    def _console_run_marker_with_unseen(
        self,
        session: ConsoleChatSession,
        unseen_ids: frozenset[str],
    ) -> ConsoleRunMarker:
        """Derive a live run marker with unseen as the lowest precedence."""
        marker = self._run_marker_for_session(session.id)
        if marker is ConsoleRunMarker.NONE and (
            (session.persisted_conversation_id or session.id) in unseen_ids
        ):
            return ConsoleRunMarker.SUBAGENT_UNSEEN
        return marker

    def prepare_session_run_markers(
        self,
        sessions: tuple[ConsoleChatSession, ...],
        active_session_id: str | None,
    ) -> dict[str, ConsoleRunMarker] | None:
        """Clear a viewed unseen mark when safe and derive session markers.

        Args:
            sessions: Current Console sessions to derive markers for.
            active_session_id: Identifier of the session currently in view, if any.

        Returns:
            Markers keyed by session ID, or ``None`` when no chat controller exists.
        """
        chat_controller_available = self._chat_controller_available()
        unseen_ids = self._console_fleet_unseen_ids()
        if unseen_ids and active_session_id:
            active = next(
                (session for session in sessions if session.id == active_session_id),
                None,
            )
            if active is not None:
                conversation_id = active.persisted_conversation_id or active.id
                wake_owed = bool(self._wake_has_pending(conversation_id))
                if (
                    conversation_id in unseen_ids
                    and not wake_owed
                    and self._console_screen_displayed()
                    and self._clear_fleet_unseen(conversation_id)
                ):
                    unseen_ids = self._console_fleet_unseen_ids()
        if not chat_controller_available:
            return None
        return {
            session.id: self._console_run_marker_with_unseen(session, unseen_ids)
            for session in sessions
        }

    def _console_fleet_survivors_live(self) -> bool:
        """Return whether the fleet still owes a surviving child drain."""
        if not self._chat_controller_available():
            return False
        try:
            return bool(self._fleet_has_unsettled_children())
        except Exception as exc:  # noqa: BLE001 -- timer predicate cannot raise
            logger.debug(
                "fleet survivor check failed (exception_type={})",
                type(exc).__name__,
            )
            return False

    def _maybe_start_console_fleet_survivor_tick(self) -> None:
        """Arm one one-second survivor timer only while work is unsettled."""
        if self._console_fleet_survivor_timer is not None:
            return
        if not self._console_fleet_survivors_live():
            return
        self._console_fleet_survivor_timer = self._create_interval(
            1.0,
            self._console_fleet_survivor_tick,
        )
        self._record_timer_created("console-fleet-survivor-tick")

    def _stop_console_fleet_survivor_tick(self) -> None:
        """Stop and clear the survivor timer if one is armed."""
        if self._console_fleet_survivor_timer is None:
            return
        try:
            self._console_fleet_survivor_timer.stop()
        finally:
            self._record_timer_stopped("console-fleet-survivor-tick")
            self._console_fleet_survivor_timer = None

    async def _console_fleet_survivor_tick(self) -> None:
        """Repaint unsettled survivors or stop before the final settle paint."""
        if self._transcript_sync_timer_active():
            return
        if not self._chat_controller_available():
            self._stop_console_fleet_survivor_tick()
            return
        if not self._console_fleet_survivors_live():
            self._stop_console_fleet_survivor_tick()
            await self._sync_native_console_ui()
            return
        await self._sync_native_console_ui()
