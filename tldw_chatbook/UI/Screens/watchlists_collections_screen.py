"""Watchlists destination shell.

The route, class name, and stable widget selectors retain the historical
``watchlists_collections``/``wc`` identifiers so older tests, shortcuts, and
handoffs keep working while Collections moves under Library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...runtime_policy.types import PolicyDeniedError
from ...Utils.input_validation import sanitize_string, validate_text_input
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from .destination_recovery import DestinationRecoveryState, policy_denied_recovery_state


logger = logger.bind(module="WatchlistsCollectionsScreen")
WC_LOCAL_PAGE_SIZE = 5
WC_SERVICE_ERROR_COPY = "Watchlists services unavailable; retry Watchlists later."
WC_SERVICE_UNAVAILABLE_COPY = "Watchlists services are unavailable in this runtime."
WC_EMPTY_COPY = "No local Watchlists are available yet."


class WatchlistsCollectionsScreen(BaseAppScreen):
    """Monitored sources, runs, alerts, and recovery."""

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "watchlists_collections", **kwargs)
        self._latest_console_follow_item_id = None
        self._latest_console_follow_item_cache = None
        self._latest_console_follow_loaded = False
        self._latest_console_follow_error_logged = False
        self._local_watchlist_records: tuple[Mapping[str, Any], ...] = ()
        self._local_collection_records: tuple[Mapping[str, Any], ...] = ()
        self._local_watchlist_count = 0
        self._local_collection_count = 0
        self._watchlist_total_known = True
        self._collection_total_known = True
        self._wc_lookup_error: str | None = None
        self._wc_lookup_recovery_state: DestinationRecoveryState | None = None
        self._wc_loaded = False

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_local_wc_snapshot()

    @work(exclusive=True)
    async def _refresh_local_wc_snapshot(self) -> None:
        (
            watchlists,
            collections,
            watchlist_count,
            collection_count,
            watchlist_total_known,
            collection_total_known,
            lookup_error,
            recovery_state,
        ) = await self._list_local_wc_snapshot()
        self._apply_local_wc_snapshot(
            watchlists,
            collections,
            watchlist_count,
            collection_count,
            watchlist_total_known,
            collection_total_known,
            lookup_error,
            recovery_state,
        )

    def _apply_local_wc_snapshot(
        self,
        watchlists: tuple[Mapping[str, Any], ...],
        collections: tuple[Mapping[str, Any], ...],
        watchlist_count: int,
        collection_count: int,
        watchlist_total_known: bool,
        collection_total_known: bool,
        lookup_error: str | None = None,
        recovery_state: DestinationRecoveryState | None = None,
    ) -> None:
        self._local_watchlist_records = watchlists
        self._local_collection_records = collections
        self._local_watchlist_count = watchlist_count
        self._local_collection_count = collection_count
        self._watchlist_total_known = watchlist_total_known
        self._collection_total_known = collection_total_known
        self._wc_lookup_error = lookup_error
        self._wc_lookup_recovery_state = recovery_state
        self._wc_loaded = True
        if self.is_mounted:
            self.refresh(recompose=True)

    @staticmethod
    def _safe_text(value: Any, fallback: str = "", *, max_length: int = 500) -> str:
        text = sanitize_string(str(value or ""), max_length=max_length).strip()
        if not text:
            return fallback
        if validate_text_input(text, max_length=max_length, allow_html=False):
            return text
        return fallback

    @classmethod
    def _record_title(cls, record: Mapping[str, Any]) -> str:
        for key in ("title", "name", "label", "url", "source"):
            title = cls._safe_text(record.get(key))
            if title:
                return title
        return "Untitled item"

    @staticmethod
    def _response_records_and_count(result: Any) -> tuple[tuple[Mapping[str, Any], ...], int, bool]:
        total = None
        if isinstance(result, Mapping):
            raw_items = result.get("items")
            pagination = result.get("pagination")
            total = result.get("total")
            if isinstance(pagination, Mapping):
                total = pagination.get("total", pagination.get("total_items", total))
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            raw_items = result
        else:
            raw_items = ()

        records = tuple(record for record in tuple(raw_items or ()) if isinstance(record, Mapping))
        total_known = total is not None
        try:
            count = int(total) if total is not None else len(records)
        except (TypeError, ValueError):
            count = len(records)
            total_known = False
        return records, max(count, 0), total_known

    async def _list_local_wc_snapshot(
        self,
    ) -> tuple[
        tuple[Mapping[str, Any], ...],
        tuple[Mapping[str, Any], ...],
        int,
        int,
        bool,
        bool,
        str | None,
        DestinationRecoveryState | None,
    ]:
        watchlist_service = getattr(self.app_instance, "watchlist_scope_service", None)
        list_watch_items = getattr(watchlist_service, "list_watch_items", None)
        if not callable(list_watch_items):
            return (), (), 0, 0, True, True, WC_SERVICE_UNAVAILABLE_COPY, None

        try:
            watchlist_result = await list_watch_items(
                runtime_backend="local",
                limit=WC_LOCAL_PAGE_SIZE,
                offset=0,
            )
        except PolicyDeniedError as exc:
            policy_message = self._safe_text(exc.user_message, WC_SERVICE_ERROR_COPY)
            recovery_state = policy_denied_recovery_state(
                exc,
                unavailable_what="Stage Watchlists context in Console",
                stable_selector="wc-service-error",
                policy_message=policy_message,
            )
            return (), (), 0, 0, True, True, recovery_state.visible_copy, recovery_state
        except Exception:
            logger.debug(
                "Failed to load local Watchlists snapshot.",
                exc_info=True,
            )
            return (), (), 0, 0, True, True, WC_SERVICE_ERROR_COPY, None

        watchlists, watchlist_count, watchlist_total_known = self._response_records_and_count(watchlist_result)
        return (
            watchlists,
            (),
            watchlist_count,
            0,
            watchlist_total_known,
            True,
            None,
            None,
        )

    def _has_local_wc_context(self) -> bool:
        return self._local_watchlist_count > 0

    def _count_label(self, label: str, count: int, total_known: bool) -> str:
        if total_known:
            return f"{label}: {count}"
        return f"{label} (showing up to {WC_LOCAL_PAGE_SIZE}): {count}"

    def _snapshot_body(self) -> str:
        lines = ["Local Watchlists snapshot staged for Console:", ""]
        lines.append(self._count_label("Watchlists", self._local_watchlist_count, self._watchlist_total_known))
        for index, record in enumerate(self._local_watchlist_records, start=1):
            lines.append(f"  {index}. {self._record_title(record)}")
        return "\n".join(lines).strip()

    def _snapshot_metadata(self) -> dict[str, Any]:
        return {
            "watchlist_count": self._local_watchlist_count,
            "watchlist_sample_count": len(self._local_watchlist_records),
            "watchlist_titles": [self._record_title(record) for record in self._local_watchlist_records],
            "backend": "local",
        }

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
                logger.warning(
                    "Failed to load Watchlists Console follow item from Home active-work adapter.",
                    exc_info=True,
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

    def compose_content(self) -> ComposeResult:
        latest_console_item = self._latest_console_follow_item()
        self._latest_console_follow_item_id = (
            getattr(latest_console_item, "item_id", None)
            if latest_console_item is not None
            else None
        )
        with Vertical(id="watchlists-collections-shell"):
            yield Static(
                "Watchlists",
                id="watchlists-collections-title",
                classes="ds-destination-header",
            )
            yield Static(
                "Monitored sources, runs, alerts, and recovery.",
                id="watchlists-collections-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="watchlists-collections-sections", classes="ds-panel"):
                yield Static("Watchlists", classes="destination-section")
                yield Static(
                    "Monitored sources, filters, jobs, runs, outputs, templates, alerts, telemetry, retry/backoff."
                )
                yield Static(
                    "Library now owns curated source groups, imports, saved searches, and reading workflows."
                )
                if not self._wc_loaded:
                    yield Static(
                        "Loading local Watchlists snapshot...",
                        id="wc-loading-state",
                    )
                    attach_disabled = True
                    attach_tooltip = "Stage local Watchlists context after the local snapshot loads."
                elif self._wc_lookup_error:
                    recovery_state = self._wc_lookup_recovery_state
                    yield Static(
                        self._wc_lookup_error,
                        id=(
                            recovery_state.stable_selector
                            if recovery_state is not None
                            else "wc-service-error"
                        ),
                    )
                    attach_disabled = True
                    attach_tooltip = (
                        recovery_state.disabled_tooltip
                        if recovery_state is not None
                        else "Watchlists services are unavailable; retry Watchlists before staging Console context."
                    )
                elif not self._has_local_wc_context():
                    yield Static(
                        WC_EMPTY_COPY,
                        id="wc-empty-state",
                    )
                    attach_disabled = True
                    attach_tooltip = "Stage local Watchlists context once local watchlists exist."
                else:
                    yield Static(
                        "Local Watchlists snapshot",
                        id="wc-snapshot-title",
                        classes="destination-section",
                    )
                    yield Static(
                        self._count_label(
                            "Watchlists",
                            self._local_watchlist_count,
                            self._watchlist_total_known,
                        ),
                        id="wc-watchlists-summary",
                    )
                    for index, record in enumerate(self._local_watchlist_records):
                        yield Static(
                            Text.from_markup(escape_markup(self._record_title(record))),
                            id=f"wc-watchlist-item-{index}",
                        )
                    attach_disabled = False
                    attach_tooltip = "Stage local Watchlists context in Console."
                yield Button(
                    "Stage Watchlists Context in Console",
                    id="wc-attach-to-console",
                    disabled=attach_disabled,
                    tooltip=attach_tooltip,
                )
                yield Button(
                    "Open current Watchlists",
                    id="wc-open-watchlists",
                    tooltip="Open the current watchlist/subscription surface.",
                )
                if latest_console_item is not None:
                    title = str(getattr(latest_console_item, "title", None) or "Untitled")
                    status = str(getattr(latest_console_item, "status", None) or "unknown")
                    yield Static(
                        Text.from_markup(
                            "Console can follow latest Watchlists run: "
                            f"{escape_markup(title)} ({escape_markup(status)})."
                        ),
                        id="watchlists-console-available",
                    )
                    yield Button(
                        Text.from_markup(f"Follow {escape_markup(title)} in Console"),
                        id="watchlists-follow-in-console",
                        tooltip="Open the latest active Watchlists run in Console.",
                    )
                else:
                    yield Static(
                        "No active Watchlists run is available for Console follow.",
                        id="watchlists-console-unavailable",
                    )
                    yield Button(
                        "Console follow unavailable",
                        id="watchlists-follow-in-console",
                        disabled=True,
                        tooltip="Unavailable until Watchlists has an active run with Console context.",
                    )

    @on(Button.Pressed, "#wc-open-watchlists")
    def open_watchlists(self) -> None:
        self.post_message(NavigateToScreen("subscriptions"))

    @on(Button.Pressed, "#wc-attach-to-console")
    def attach_to_console(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._has_local_wc_context():
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    self._wc_lookup_error or WC_EMPTY_COPY,
                    severity="warning",
                )
            return
        open_chat_with_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
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
                body=self._snapshot_body(),
                display_summary="Local Watchlists snapshot staged.",
                suggested_prompt="Use these monitored sources as context.",
                runtime_backend="local",
                source_owner="local",
                source_selector_state="local",
                metadata=self._snapshot_metadata(),
            )
        )

    @on(Button.Pressed, "#watchlists-follow-in-console")
    def follow_latest_watchlist_run_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        target_id = self._latest_console_follow_item_id
        if not target_id:
            self.app_instance.notify(
                "No active Watchlists run is available for Console follow.",
                severity="warning",
            )
            return
        open_in_console = getattr(self.app_instance, "open_active_home_item_in_console", None)
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
