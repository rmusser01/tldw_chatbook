"""Deliberately RED shell for Console fleet and wake lifecycle ownership."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ConsoleFleetLifecycleController:
    """Hold the reviewed fleet callback boundary without behavior yet."""

    def __init__(
        self,
        *,
        pending_handoffs_accessor: Callable[..., Any],
        ensure_chat_store: Callable[..., Any],
        chat_store_accessor: Callable[..., Any],
        activate_workspace_for_session: Callable[..., Any],
        switch_chat_session: Callable[..., Any],
        schedule_native_console_sync: Callable[..., Any],
        ensure_agent_bridge: Callable[..., Any],
        wire_wake_coordinator: Callable[..., Any],
        seed_wake_from_marks: Callable[..., Any],
        retry_wake_soon: Callable[..., Any],
        wake_has_pending: Callable[..., Any],
        wake_delivering_conversation_id: Callable[..., Any],
        displayed_composer_draft_accessor: Callable[..., Any],
        screen_displayed_accessor: Callable[..., Any],
        screen_mounted_accessor: Callable[..., Any],
        active_session_id_accessor: Callable[..., Any],
        chat_sessions_accessor: Callable[..., Any],
        defer_on_message_pump: Callable[..., Any],
        start_transcript_sync_timer: Callable[..., Any],
        transcript_sync_timer_active: Callable[..., Any],
        sync_native_console_ui: Callable[..., Any],
        create_interval: Callable[..., Any],
        record_timer_created: Callable[..., Any],
        record_timer_stopped: Callable[..., Any],
        chat_controller_available: Callable[..., Any],
        fleet_has_unsettled_children: Callable[..., Any],
        run_marker_for_session: Callable[..., Any],
        fleet_teardown_split: Callable[..., Any],
        leave_runtime: Callable[..., Any],
        stage_teardown_notices: Callable[..., Any],
        fleet_unseen_revision_accessor: Callable[..., Any],
        read_fleet_unseen_ids: Callable[..., Any],
        clear_fleet_unseen: Callable[..., Any],
    ) -> None:
        self._pending_handoffs_accessor = pending_handoffs_accessor
        self._ensure_chat_store = ensure_chat_store
        self._chat_store_accessor = chat_store_accessor
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

        self._console_fleet_survivor_timer = None
        self._console_fleet_unseen_cache = None

    def consume_pending_console_fleet_completion(self) -> bool:
        return False

    def _claim_console_fleet_wake_marks(self) -> None:
        return None

    def _console_wake_user_priority(self, session_id: str) -> bool:
        return False

    def _console_wake_probe_composer(self) -> None:
        return None

    def _console_screen_displayed(self) -> bool:
        return False

    def _console_wake_conversation_in_view(
        self,
        conversation_id: str,
        session_id: str,
    ) -> bool:
        return False

    def _poke_console_wake_retry(self) -> None:
        return None

    def _on_console_wake_delivery_started(self, session_id: str) -> None:
        return None

    def _console_wake_turn_active(self, session_id: str | None) -> bool:
        return False

    async def _record_console_fleet_teardown(self) -> None:
        return None

    def _console_fleet_unseen_ids(self) -> dict[Any, Any]:
        return {}

    def _console_run_marker_with_unseen(
        self,
        session: Any,
        unseen_ids: frozenset[str],
    ) -> None:
        return None

    def _console_fleet_survivors_live(self) -> bool:
        return False

    def _maybe_start_console_fleet_survivor_tick(self) -> None:
        return None

    def _stop_console_fleet_survivor_tick(self) -> None:
        return None

    async def _console_fleet_survivor_tick(self) -> None:
        return None

    def prepare_session_run_markers(
        self,
        sessions: tuple[Any, ...],
        active_session_id: str | None,
    ) -> dict[str, Any] | None:
        return {}
