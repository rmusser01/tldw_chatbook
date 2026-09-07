"""Console staging and follow for the Watchlists screen.

Extracted from the screen shell so the shell stays thin. This is a pure
move: the logic, messages, and stable selectors are unchanged from when it
lived on the screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from ...Chat.chat_handoff_models import ChatHandoffPayload


if TYPE_CHECKING:
    from ...Home.dashboard_state import HomeActiveWorkItem


logger = logger.bind(module="WatchlistsConsoleHandoff")

WC_EMPTY_COPY = "No local Watchlists are available yet."


class WatchlistsConsoleHandoff:
    """Owns the Watchlists screen's Console staging and follow state."""

    def __init__(self, app_instance: Any) -> None:
        self.app_instance = app_instance
        self._latest_console_follow_item_id = None
        self._latest_console_follow_item_cache = None
        self._latest_console_follow_loaded = False
        self._latest_console_follow_error_logged = False

    # Each method below is moved verbatim from the screen. Keep the bodies
    # byte-identical apart from `self.app_instance` resolution.

    def _latest_console_follow_item(self):
        if self._latest_console_follow_loaded:
            return self._latest_console_follow_item_cache
        adapter = getattr(self.app_instance, "home_active_work_adapter", None)
        build_dashboard_input = getattr(adapter, "build_dashboard_input", None)
        if not callable(build_dashboard_input):
            self._latest_console_follow_item_cache = None
            self._latest_console_follow_loaded = True
            self._latest_console_follow_error_logged = False
            return None
        try:
            dashboard_input = build_dashboard_input(
                providers_models={},
                has_recent_work=False,
            )
        except Exception:
            if not self._latest_console_follow_error_logged:
                logger.opt(exception=True).warning(
                    "Failed to load Watchlists Console follow item from Home active-work adapter.",
                )
                self._latest_console_follow_error_logged = True
            self._latest_console_follow_item_cache = None
            return None
        selected_item = None
        for item in tuple(getattr(dashboard_input, "active_work_items", ()) or ()):
            if (
                str(getattr(item, "source", None) or "").strip().lower()
                in {"watchlists", "w+c", "watchlists+collections"}
                and bool(getattr(item, "console_available", False))
                and getattr(item, "item_id", None)
            ):
                selected_item = item
                break
        self._latest_console_follow_item_cache = selected_item
        self._latest_console_follow_loaded = True
        self._latest_console_follow_error_logged = False
        return selected_item

    def resolve_latest_follow_item(self) -> HomeActiveWorkItem | None:
        """Refresh the cached follow item/id ahead of a render pass.

        This is the same population logic that used to sit inline in the
        screen's ``compose_content`` (compute the latest follow item, then
        stash its id for the follow-in-console handler to read later).

        Returns:
            The latest Watchlists-eligible active-work item available to
            follow in Console, or `None` when none is available.
        """
        latest_console_item = self._latest_console_follow_item()
        self._latest_console_follow_item_id = (
            getattr(latest_console_item, "item_id", None)
            if latest_console_item is not None
            else None
        )
        return latest_console_item

    def attach_to_console(self, screen: Any) -> None:
        if not screen._has_local_wc_context():
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    screen._wc_lookup_error or WC_EMPTY_COPY,
                    severity="warning",
                )
            return
        open_chat_with_handoff = getattr(
            self.app_instance, "open_chat_with_handoff", None
        )
        if not callable(open_chat_with_handoff):
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "Console handoff is unavailable for Watchlists in this runtime.",
                    severity="warning",
                )
            return
        open_chat_with_handoff(
            ChatHandoffPayload(
                source="watchlists_collections",
                item_type="wc-context",
                title="Local Watchlists snapshot",
                body=screen._snapshot_body(),
                display_summary="Local Watchlists snapshot staged.",
                suggested_prompt="Use these monitored sources as context.",
                runtime_backend="local",
                source_owner="local",
                source_selector_state="local",
                metadata=screen._snapshot_metadata(),
            )
        )

    def follow_in_console(self) -> None:
        target_id = self._latest_console_follow_item_id
        if not target_id:
            self.app_instance.notify(
                "No active Watchlists run is available for Console follow.",
                severity="warning",
            )
            return
        open_in_console = getattr(
            self.app_instance, "open_active_home_item_in_console", None
        )
        if not callable(open_in_console):
            self.app_instance.notify(
                "Console follow is unavailable for Watchlists in this runtime.",
                severity="warning",
            )
            return
        open_in_console(
            target_id=target_id,
            target_route="chat",
        )

    def handle_stage_in_console_requested(self) -> None:
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify("Stage in Console is not implemented yet.", severity="information")
