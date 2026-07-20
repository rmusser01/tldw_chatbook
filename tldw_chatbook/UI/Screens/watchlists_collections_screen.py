"""Watchlists destination shell.

The route, class name, and stable widget selectors retain the historical
``watchlists_collections``/``wc`` identifiers so older tests, shortcuts, and
handoffs keep working while Collections moves under Library.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Rule, Select, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...runtime_policy.types import PolicyDeniedError
from ...Utils.input_validation import sanitize_string, validate_text_input
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from ..Watchlists_Modules.inspector_pane import (
    CheckNowRequested,
    DeleteRequested,
    EditRuleRequested,
    IgnoreRequested,
    IngestRequested,
    InspectorPane,
    MarkReviewedRequested,
    PreviewRequested,
    StageInConsoleRequested,
)
from ..Watchlists_Modules.items_pane import ItemSelected, ItemsPane, RefreshItemsRequested
from ..Watchlists_Modules.opml_dialogs import (
    ConfirmDeleteDialog,
    OpmlExportDialog,
    OpmlImportDialog,
)
from ..Watchlists_Modules.overview_pane import OverviewPane
from ..Watchlists_Modules.rules_pane import (
    RefreshRulesRequested,
    RuleSelected,
    RulesPane,
    SaveRuleRequested,
)
from ..Watchlists_Modules.runs_pane import CancelRunRequested, RerunRunRequested, RunsPane, RunSelected
from ..Watchlists_Modules.sources_pane import (
    CreateSourceRequested,
    ExportOpmlRequested,
    ImportOpmlRequested,
    SourceSelected,
    SourcesPane,
)
from ..Watchlists_Modules.watchlists_backend_controller import WatchlistsBackendController
from ..Watchlists_Modules.watchlists_navigator import SectionSelected, WatchlistsNavigator
from .destination_recovery import DestinationRecoveryState, policy_denied_recovery_state


logger = logger.bind(module="WatchlistsCollectionsScreen")
WC_LOCAL_PAGE_SIZE = 5
WC_SERVICE_ERROR_COPY = "Watchlists services unavailable; retry Watchlists later."
WC_SERVICE_UNAVAILABLE_COPY = "Watchlists services are unavailable in this runtime."
WC_EMPTY_COPY = "No local Watchlists are available yet."
WC_SNAPSHOT_TIMEOUT_SECONDS = 1.5


class WatchlistsCollectionsScreen(BaseAppScreen):
    """Monitored sources, runs, alerts, and recovery."""

    BINDINGS = [
        ("1", "switch_section('overview')", "Overview"),
        ("2", "switch_section('sources')", "Sources"),
        ("3", "switch_section('items')", "Items"),
        ("4", "switch_section('runs')", "Runs"),
        ("5", "switch_section('rules')", "Rules"),
        ("question", "show_help", "Help"),
        ("n", "new_source", "New source"),
        ("d", "delete_selected", "Delete"),
        ("c", "check_now_selected", "Check now"),
        ("p", "preview_selected", "Preview"),
    ]

    active_section = reactive("overview")
    runtime_backend = reactive("local")
    selected_source = reactive(None)
    selected_run = reactive(None)
    selected_entity = reactive(None)
    recovery_state = reactive(None)
    overview_data = reactive({}, recompose=True)

    _SECTION_DETAIL_TITLE = {
        "overview": "Overview",
        "sources": "Sources",
        "items": "Items",
        "runs": "Runs",
        "rules": "Rules",
    }

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "watchlists_collections", **kwargs)
        self._latest_console_follow_item_id = None
        self._latest_console_follow_item_cache = None
        self._latest_console_follow_loaded = False
        self._latest_console_follow_error_logged = False
        self._local_watchlist_records: tuple[Mapping[str, Any], ...] = ()
        self._local_watchlist_count = 0
        self._watchlist_total_known = True
        self._wc_lookup_error: str | None = None
        self._wc_lookup_recovery_state: DestinationRecoveryState | None = None
        self._wc_loaded = False
        self._pending_open_create_form = False
        self._pending_open_import_opml = False
        self._pending_delete_entity: dict[str, Any] | None = None
        self._controller = WatchlistsBackendController(
            app_instance=app_instance,
            scope_service=getattr(app_instance, "watchlist_scope_service", None),
            server_service=getattr(app_instance, "server_watchlists_service", None),
        )

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()
        self.set_timer(
            WC_SNAPSHOT_TIMEOUT_SECONDS, self._apply_snapshot_timeout_if_still_loading
        )

    def _apply_snapshot_timeout_if_still_loading(self) -> None:
        if self._wc_loaded:
            return
        self._apply_local_wc_snapshot(
            (),
            0,
            True,
            WC_SERVICE_ERROR_COPY,
            None,
        )

    @work(exclusive=True, group="wc_snapshot")
    async def _refresh_local_wc_snapshot(self) -> None:
        (
            watchlists,
            watchlist_count,
            watchlist_total_known,
            lookup_error,
            recovery_state,
        ) = await self._list_local_wc_snapshot()
        self._apply_local_wc_snapshot(
            watchlists,
            watchlist_count,
            watchlist_total_known,
            lookup_error,
            recovery_state,
        )

    @work(exclusive=True, group="wc_overview")
    async def _refresh_overview_data(self) -> None:
        try:
            data = await self._controller.get_overview_data(
                runtime_backend=self.runtime_backend,
            )
            self.overview_data = data
        except Exception:
            logger.opt(exception=True).debug("Failed to refresh watchlists overview data.")
            self.overview_data = {
                "total_sources": 0,
                "active_sources": 0,
                "sources_in_error": 0,
                "total_items": 0,
                "new_items": 0,
                "latest_run_status": "unavailable",
                "failed_runs": [],
                "active_alert_rules": 0,
            }

    def _apply_local_wc_snapshot(
        self,
        watchlists: tuple[Mapping[str, Any], ...],
        watchlist_count: int,
        watchlist_total_known: bool,
        lookup_error: str | None = None,
        recovery_state: DestinationRecoveryState | None = None,
    ) -> None:
        self._local_watchlist_records = watchlists
        self._local_watchlist_count = watchlist_count
        self._watchlist_total_known = watchlist_total_known
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
    def _response_records_and_count(
        result: Any,
    ) -> tuple[tuple[Mapping[str, Any], ...], int, bool]:
        total = None
        if isinstance(result, Mapping):
            raw_items = result.get("items")
            pagination = result.get("pagination")
            total = result.get("total")
            if isinstance(pagination, Mapping):
                total = pagination.get("total", pagination.get("total_items", total))
        elif isinstance(result, Sequence) and not isinstance(
            result, (str, bytes, bytearray)
        ):
            raw_items = result
        else:
            raw_items = ()

        records = tuple(
            record for record in tuple(raw_items or ()) if isinstance(record, Mapping)
        )
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
        int,
        bool,
        str | None,
        DestinationRecoveryState | None,
    ]:
        watchlist_service = getattr(self.app_instance, "watchlist_scope_service", None)
        list_watch_items = getattr(watchlist_service, "list_watch_items", None)
        if not callable(list_watch_items):
            return (), 0, True, WC_SERVICE_UNAVAILABLE_COPY, None

        try:
            watchlist_result = await asyncio.wait_for(
                list_watch_items(
                    runtime_backend="local",
                    limit=WC_LOCAL_PAGE_SIZE,
                    offset=0,
                ),
                timeout=WC_SNAPSHOT_TIMEOUT_SECONDS,
            )
        except PolicyDeniedError as exc:
            policy_message = self._safe_text(exc.user_message, WC_SERVICE_ERROR_COPY)
            recovery_state = policy_denied_recovery_state(
                exc,
                unavailable_what="Stage Watchlists context in Console",
                stable_selector="wc-service-error",
                policy_message=policy_message,
            )
            return (), 0, True, recovery_state.visible_copy, recovery_state
        except TimeoutError:
            logger.debug("Timed out loading local Watchlists snapshot.")
            return (), 0, True, WC_SERVICE_ERROR_COPY, None
        except Exception:
            logger.opt(exception=True).debug(
                "Failed to load local Watchlists snapshot."
            )
            return (), 0, True, WC_SERVICE_ERROR_COPY, None

        watchlists, watchlist_count, watchlist_total_known = (
            self._response_records_and_count(watchlist_result)
        )
        return (
            watchlists,
            watchlist_count,
            watchlist_total_known,
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
        lines.append(
            self._count_label(
                "Watchlists", self._local_watchlist_count, self._watchlist_total_known
            )
        )
        for index, record in enumerate(self._local_watchlist_records, start=1):
            lines.append(f"  {index}. {self._record_title(record)}")
        return "\n".join(lines).strip()

    def _snapshot_metadata(self) -> dict[str, Any]:
        return {
            "watchlist_count": self._local_watchlist_count,
            "watchlist_sample_count": len(self._local_watchlist_records),
            "watchlist_titles": [
                self._record_title(record) for record in self._local_watchlist_records
            ],
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

    @staticmethod
    def _column_divider(divider_id: str) -> Rule:
        return Rule(
            line_style="heavy",
            orientation="vertical",
            id=divider_id,
            classes="destination-pane-divider",
        )

    def compose_content(self) -> ComposeResult:
        latest_console_item = self._latest_console_follow_item()
        self._latest_console_follow_item_id = (
            getattr(latest_console_item, "item_id", None)
            if latest_console_item is not None
            else None
        )
        with Vertical(id="watchlists-collections-shell"):
            yield Static(
                "Watchlists | Monitored sources, runs, alerts, recovery | Mixed | Local/Server",
                id="watchlists-collections-title",
                classes="ds-destination-header",
            )
            with Horizontal(id="watchlists-header-bar", classes="destination-filter-strip"):
                yield Select(
                    [("Local", "local"), ("Server", "server")],
                    value="local",
                    id="watchlists-backend-select",
                    allow_blank=False,
                )
                yield Static(
                    f"Backend: {self.runtime_backend}",
                    id="watchlists-backend-label",
                )
            with Horizontal(id="watchlists-workbench", classes="ds-panel destination-workbench"):
                yield WatchlistsNavigator(id="watchlists-navigator")
                yield self._column_divider("watchlists-nav-list-divider")
                with Vertical(id="watchlists-list-pane", classes="destination-workbench-pane"):
                    yield Static("Sources", classes="destination-section watchlists-column-title")
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
                            "No sources yet.",
                            id="wc-empty-state",
                        )
                        with Horizontal(id="wc-empty-actions", classes="destination-filter-strip"):
                            yield Button(
                                "Create source",
                                id="wc-empty-create-source",
                                variant="primary",
                                tooltip="Add a new Watchlists source.",
                            )
                            yield Button(
                                "Import OPML",
                                id="wc-empty-import-opml",
                                tooltip="Import sources from an OPML file.",
                            )
                        attach_disabled = True
                        attach_tooltip = "Stage local Watchlists context once local sources exist."
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
                                Text.from_markup(
                                    escape_markup(self._record_title(record))
                                ),
                                id=f"wc-watchlist-item-{index}",
                            )
                        attach_disabled = False
                        attach_tooltip = "Stage local Watchlists context in Console."
                yield self._column_divider("watchlists-list-detail-divider")
                with Vertical(id="watchlists-detail-pane", classes="destination-workbench-pane"):
                    detail_title = self._SECTION_DETAIL_TITLE.get(
                        self.active_section, "Detail"
                    )
                    yield Static(
                        detail_title,
                        classes="destination-section watchlists-column-title",
                        id="watchlists-detail-title",
                    )
                    if self.active_section == "overview":
                        overview = OverviewPane(id="watchlists-overview-pane")
                        overview.data = self.overview_data
                        yield overview
                    elif self.active_section == "sources":
                        yield SourcesPane(id="watchlists-sources-pane")
                    elif self.active_section == "runs":
                        yield RunsPane(id="watchlists-runs-pane")
                    elif self.active_section == "items":
                        yield ItemsPane(id="watchlists-items-pane")
                    elif self.active_section == "rules":
                        yield RulesPane(id="watchlists-rules-pane")
                yield self._column_divider("watchlists-detail-inspector-divider")
                with Vertical(
                    id="watchlists-inspector-pane",
                    classes="destination-workbench-pane ds-inspector",
                ):
                    yield Static(
                        "Inspector",
                        classes="destination-section watchlists-column-title",
                    )
                    yield Static(
                        "State: ready"
                        if self._wc_loaded and not self._wc_lookup_error
                        else "State: unavailable",
                        id="watchlists-state-summary",
                    )
                    yield Static(
                        f"Alert rules active: {self.overview_data.get('active_alert_rules', 0)}",
                        id="watchlists-alerts-summary",
                    )
                    yield Static(
                        f"Latest run status: {self.overview_data.get('latest_run_status', 'unavailable')}",
                        id="watchlists-latest-run-summary",
                    )
                    yield Static("Console actions", classes="destination-section")
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
                        title = str(
                            getattr(latest_console_item, "title", None) or "Untitled"
                        )
                        status = str(
                            getattr(latest_console_item, "status", None) or "unknown"
                        )
                        yield Static(
                            Text.from_markup(
                                "Console can follow latest Watchlists run: "
                                f"{escape_markup(title)} ({escape_markup(status)})."
                            ),
                            id="watchlists-console-available",
                        )
                        yield Button(
                            Text.from_markup(
                                f"Follow {escape_markup(title)} in Console"
                            ),
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
                    yield InspectorPane(id="watchlists-entity-inspector")

    def watch_active_section(self) -> None:
        if self.active_section == "overview":
            self.selected_entity = None
        if self.is_mounted:
            self.refresh(recompose=True)
        if self.active_section == "items":
            self.run_worker(self._load_items(), exclusive=True)
        elif self.active_section == "rules":
            self.run_worker(self._load_rules(), exclusive=True)
        elif self.active_section == "runs":
            self.run_worker(self._load_runs(), exclusive=True)
        elif self.active_section == "sources":
            self.run_worker(self._load_sources(), exclusive=True)

        if self._pending_open_create_form:
            self._pending_open_create_form = False
            self.set_timer(0.05, self._open_sources_create_form)
        if self._pending_open_import_opml:
            self._pending_open_import_opml = False
            self.set_timer(0.05, self._open_sources_import_opml)

    def _open_sources_create_form(self) -> None:
        if not self.is_mounted:
            return
        try:
            pane = self.query_one("#watchlists-sources-pane", SourcesPane)
            pane.show_create_form = True
        except Exception:
            pass

    def _open_sources_import_opml(self) -> None:
        if not self.is_mounted:
            return
        self.app.push_screen(OpmlImportDialog(), callback=self._on_opml_import_complete)

    def watch_runtime_backend(self) -> None:
        if not self.is_mounted:
            return
        try:
            label = self.query_one("#watchlists-backend-label", Static)
            label.update(f"Backend: {self.runtime_backend}")
        except Exception:
            pass
        self.selected_source = None
        self.selected_run = None
        self.selected_entity = None
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()

    def watch_selected_entity(self) -> None:
        if not self.is_mounted:
            return
        try:
            inspector = self.query_one("#watchlists-entity-inspector", InspectorPane)
            inspector.selected_entity = self.selected_entity
        except Exception:
            pass

    @on(SectionSelected)
    def handle_section_selected(self, event: SectionSelected) -> None:
        event.stop()
        self.active_section = event.section_id

    @on(Select.Changed, "#watchlists-backend-select")
    def handle_backend_changed(self, event: Select.Changed) -> None:
        event.stop()
        self.runtime_backend = str(event.value or "local")

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

    @on(Button.Pressed, "#wc-empty-create-source")
    def handle_empty_create_source(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_new_source()

    @on(Button.Pressed, "#wc-empty-import-opml")
    def handle_empty_import_opml(self, event: Button.Pressed) -> None:
        event.stop()
        self.active_section = "sources"
        self._pending_open_import_opml = True

    @on(SourceSelected)
    def handle_source_selected(self, event: SourceSelected) -> None:
        event.stop()
        self.selected_source = event.source
        self.selected_entity = event.source

    @on(RunSelected)
    def handle_run_selected(self, event: RunSelected) -> None:
        event.stop()
        self.selected_run = event.run
        self.selected_entity = event.run

    @on(CreateSourceRequested)
    def handle_create_source_requested(self, event: CreateSourceRequested) -> None:
        event.stop()
        self.run_worker(self._create_source(event.payload), exclusive=True)

    async def _create_source(self, payload: dict[str, Any]) -> None:
        try:
            await self._controller.create_source(
                runtime_backend=self.runtime_backend,
                payload=payload,
            )
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Source created.", severity="information")
        except Exception:
            logger.opt(exception=True).debug("Failed to create source.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to create source.", severity="error")
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()

    @on(CancelRunRequested)
    def handle_cancel_run_requested(self, event: CancelRunRequested) -> None:
        event.stop()
        self.run_worker(self._cancel_run(event.run_id), exclusive=True)

    async def _cancel_run(self, run_id: Any) -> None:
        try:
            await self._controller.cancel_run(
                runtime_backend=self.runtime_backend,
                run_id=run_id,
            )
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Run cancellation requested.", severity="information")
        except Exception:
            logger.opt(exception=True).debug("Failed to cancel run.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to cancel run.", severity="error")
        self._refresh_overview_data()

    @on(RerunRunRequested)
    def handle_rerun_run_requested(self, event: RerunRunRequested) -> None:
        event.stop()
        self.run_worker(self._rerun_run(event.source_id), exclusive=True)

    async def _rerun_run(self, source_id: Any) -> None:
        try:
            await self._controller.launch_run(
                runtime_backend=self.runtime_backend,
                source_id=source_id,
            )
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Run launched.", severity="information")
        except Exception:
            logger.opt(exception=True).debug("Failed to launch run.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to launch run.", severity="error")
        self._refresh_overview_data()

    @on(PreviewRequested)
    def handle_preview_requested(self, event: PreviewRequested) -> None:
        event.stop()
        entity = event.entity
        if entity is None:
            return
        self.run_worker(self._preview_source(entity), exclusive=True)

    async def _preview_source(self, source: dict[str, Any]) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            result = await self._controller.preview_source(
                runtime_backend=self.runtime_backend,
                source_config=source,
            )
            items = result.get("items") or []
            log_text = result.get("log_text", "Preview complete.")
            if callable(notify):
                notify(
                    f"Preview: {log_text} ({len(items)} item(s))",
                    severity="information",
                    timeout=10,
                )
        except Exception:
            logger.opt(exception=True).debug("Failed to preview source.")
            if callable(notify):
                notify("Failed to preview source.", severity="error")

    @on(CheckNowRequested)
    def handle_check_now_requested(self, event: CheckNowRequested) -> None:
        event.stop()
        entity = event.entity
        if entity is None:
            return
        self.run_worker(self._check_now_source(entity), exclusive=True)

    async def _check_now_source(self, source: dict[str, Any]) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            await self._controller.check_now(
                runtime_backend=self.runtime_backend,
                source_id=source.get("id"),
            )
            if callable(notify):
                notify("Check now started.", severity="information")
        except Exception:
            logger.opt(exception=True).debug("Failed to check source.")
            if callable(notify):
                notify("Failed to check source.", severity="error")
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()

    @on(ImportOpmlRequested)
    def handle_import_opml_requested(self, event: ImportOpmlRequested) -> None:
        event.stop()
        self.app.push_screen(OpmlImportDialog(), callback=self._on_opml_import_complete)

    async def _on_opml_import_complete(self, xml_text: str | None) -> None:
        if not xml_text:
            return
        notify = getattr(self.app_instance, "notify", None)
        try:
            result = await self._controller.import_opml(
                runtime_backend=self.runtime_backend,
                xml_text=xml_text,
            )
            created = result.get("created", 0)
            if callable(notify):
                notify(f"Imported {created} source(s) from OPML.", severity="information")
        except Exception:
            logger.opt(exception=True).debug("Failed to import OPML.")
            if callable(notify):
                notify("Failed to import OPML.", severity="error")
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()

    @on(ExportOpmlRequested)
    def handle_export_opml_requested(self, event: ExportOpmlRequested) -> None:
        event.stop()
        self.run_worker(self._export_opml(), exclusive=True)

    async def _export_opml(self) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            xml_text = await self._controller.export_opml(
                runtime_backend=self.runtime_backend,
            )
            self.app.push_screen(OpmlExportDialog(xml_text))
        except Exception:
            logger.opt(exception=True).debug("Failed to export OPML.")
            if callable(notify):
                notify("Failed to export OPML.", severity="error")

    @on(StageInConsoleRequested)
    def handle_stage_in_console_requested(self, event: StageInConsoleRequested) -> None:
        event.stop()
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify("Stage in Console is not implemented yet.", severity="information")

    async def _load_sources(self) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            sources = await self._controller.list_sources(
                runtime_backend=self.runtime_backend,
                limit=100,
            )
            if self.is_mounted:
                try:
                    sources_pane = self.query_one("#watchlists-sources-pane", SourcesPane)
                    sources_pane.sources = sources
                    if self.selected_source is not None:
                        source_id = self.selected_source.get("id")
                        if source_id is not None:
                            sources_pane.select_source_by_id(str(source_id))
                except Exception:
                    pass
        except Exception:
            logger.opt(exception=True).debug("Failed to load watchlist sources.")
            if callable(notify):
                notify("Failed to load watchlist sources.", severity="error")

    async def _load_runs(self) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            runs = await self._controller.list_runs(
                runtime_backend=self.runtime_backend,
                limit=100,
            )
            if self.is_mounted:
                try:
                    runs_pane = self.query_one("#watchlists-runs-pane", RunsPane)
                    runs_pane.runs = runs
                except Exception:
                    pass
        except Exception:
            logger.opt(exception=True).debug("Failed to load watchlist runs.")
            if callable(notify):
                notify("Failed to load watchlist runs.", severity="error")

    async def _load_items(self) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            items = await self._controller.list_items(
                runtime_backend=self.runtime_backend,
                status=None,
                limit=100,
                offset=0,
            )
            if self.is_mounted:
                try:
                    items_pane = self.query_one("#watchlists-items-pane", ItemsPane)
                    items_pane.items = items
                except Exception:
                    pass
        except Exception:
            logger.opt(exception=True).debug("Failed to load watchlist items.")
            if callable(notify):
                notify("Failed to load watchlist items.", severity="error")

    @on(ItemSelected)
    def handle_item_selected(self, event: ItemSelected) -> None:
        event.stop()
        self.selected_entity = event.item

    @on(RefreshItemsRequested)
    def handle_refresh_items_requested(self, event: RefreshItemsRequested) -> None:
        event.stop()
        self.run_worker(self._load_items(), exclusive=True)

    async def _load_rules(self) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            rules = await self._controller.list_alert_rules(
                runtime_backend=self.runtime_backend,
            )
            if self.is_mounted:
                try:
                    rules_pane = self.query_one("#watchlists-rules-pane", RulesPane)
                    rules_pane.rules = rules
                except Exception:
                    pass
        except Exception:
            logger.opt(exception=True).debug("Failed to load alert rules.")
            if callable(notify):
                notify("Failed to load alert rules.", severity="error")

    @on(RuleSelected)
    def handle_rule_selected(self, event: RuleSelected) -> None:
        event.stop()
        self.selected_entity = event.rule

    @on(RefreshRulesRequested)
    def handle_refresh_rules_requested(self, event: RefreshRulesRequested) -> None:
        event.stop()
        self.run_worker(self._load_rules(), exclusive=True)

    @on(SaveRuleRequested)
    def handle_save_rule_requested(self, event: SaveRuleRequested) -> None:
        event.stop()
        self.run_worker(self._save_rule(event.payload), exclusive=True)

    @on(EditRuleRequested)
    def handle_edit_rule_requested(self, event: EditRuleRequested) -> None:
        event.stop()
        rule = event.entity
        if rule is None:
            return
        self.active_section = "rules"

        def open_edit_form() -> None:
            if not self.is_mounted:
                return
            try:
                rules_pane = self.query_one("#watchlists-rules-pane", RulesPane)
                rules_pane.edit_rule(rule)
            except Exception:
                pass

        self.set_timer(0.05, open_edit_form)

    @on(MarkReviewedRequested)
    def handle_mark_reviewed_requested(self, event: MarkReviewedRequested) -> None:
        event.stop()
        entity = event.entity
        if entity is None:
            return
        self.run_worker(self._update_item_status(entity.get("id"), "reviewed"), exclusive=True)

    @on(IngestRequested)
    def handle_ingest_requested(self, event: IngestRequested) -> None:
        event.stop()
        entity = event.entity
        if entity is None:
            return
        self.run_worker(self._update_item_status(entity.get("id"), "ingested"), exclusive=True)

    @on(IgnoreRequested)
    def handle_ignore_requested(self, event: IgnoreRequested) -> None:
        event.stop()
        entity = event.entity
        if entity is None:
            return
        self.run_worker(self._update_item_status(entity.get("id"), "ignored"), exclusive=True)

    async def _update_item_status(self, item_id: Any, status: str) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            await self._controller.update_item_status(
                runtime_backend=self.runtime_backend,
                item_id=item_id,
                status=status,
            )
            if callable(notify):
                notify(f"Item marked {status}.", severity="information")
        except Exception:
            logger.opt(exception=True).debug(f"Failed to mark item {status}.")
            if callable(notify):
                notify(f"Failed to mark item {status}.", severity="error")
        self.run_worker(self._load_items(), exclusive=True)
        self._refresh_overview_data()

    async def _save_rule(self, payload: dict[str, Any]) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            await self._controller.save_alert_rule(
                runtime_backend=self.runtime_backend,
                payload=payload,
            )
            if callable(notify):
                notify("Alert rule saved.", severity="information")
        except Exception:
            logger.opt(exception=True).debug("Failed to save alert rule.")
            if callable(notify):
                notify("Failed to save alert rule.", severity="error")
        self.run_worker(self._load_rules(), exclusive=True)
        self._refresh_overview_data()

    @on(DeleteRequested)
    def handle_delete_requested(self, event: DeleteRequested) -> None:
        event.stop()
        entity = event.entity
        if entity is None:
            return
        self._pending_delete_entity = dict(entity)
        title = entity.get("name") or entity.get("source_title") or entity.get("title") or "this item"
        self.app.push_screen(
            ConfirmDeleteDialog(title),
            callback=self._on_delete_confirmed,
        )

    async def _on_delete_confirmed(self, confirmed: bool) -> None:
        entity = self._pending_delete_entity
        self._pending_delete_entity = None
        if not confirmed or entity is None:
            return
        entity_type = InspectorPane._entity_type(entity)
        if entity_type == "source":
            self.run_worker(self._delete_source(entity.get("id")), exclusive=True)
        elif entity_type == "run":
            self.run_worker(self._delete_run(entity.get("id")), exclusive=True)
        elif entity_type == "rule":
            self.run_worker(self._delete_rule(entity.get("id")), exclusive=True)
        elif entity_type == "item":
            self.run_worker(self._delete_item(entity.get("id")), exclusive=True)

    async def _delete_source(self, source_id: Any) -> None:
        try:
            await self._controller.delete_source(
                runtime_backend=self.runtime_backend,
                item_id=source_id,
            )
            self.selected_entity = None
            self.selected_source = None
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Source deleted.", severity="information")
        except Exception:
            logger.opt(exception=True).debug("Failed to delete source.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to delete source.", severity="error")
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()

    async def _delete_run(self, run_id: Any) -> None:
        try:
            await self._controller.delete_run(
                runtime_backend=self.runtime_backend,
                run_id=run_id,
            )
            self.selected_entity = None
            self.selected_run = None
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Run deleted.", severity="information")
        except Exception:
            logger.opt(exception=True).debug("Failed to delete run.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to delete run.", severity="error")
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()

    async def _delete_rule(self, rule_id: Any) -> None:
        try:
            await self._controller.delete_alert_rule(
                runtime_backend=self.runtime_backend,
                rule_id=rule_id,
            )
            self.selected_entity = None
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Alert rule deleted.", severity="information")
        except Exception:
            logger.opt(exception=True).debug("Failed to delete alert rule.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to delete alert rule.", severity="error")
        self.run_worker(self._load_rules(), exclusive=True)
        self._refresh_overview_data()

    async def _delete_item(self, item_id: Any) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            await self._controller.update_item_status(
                runtime_backend=self.runtime_backend,
                item_id=item_id,
                status="ignored",
            )
            if callable(notify):
                notify("Item ignored.", severity="information")
        except Exception:
            logger.opt(exception=True).debug("Failed to ignore item.")
            if callable(notify):
                notify("Failed to ignore item.", severity="error")
        self.run_worker(self._load_items(), exclusive=True)
        self._refresh_overview_data()

    def action_switch_section(self, section_id: str) -> None:
        """Switch to the named section via keyboard shortcut."""
        if section_id in self._SECTION_DETAIL_TITLE:
            self.active_section = section_id
        else:
            self.app_instance.notify(
                f"Unknown section: {section_id}",
                severity="warning",
            )

    def action_show_help(self) -> None:
        """Show a notification with available keyboard shortcuts."""
        self.app_instance.notify(
            "1=Overview 2=Sources 3=Items 4=Runs 5=Rules | n=new d=delete c=check p=preview ?=help",
            severity="information",
            timeout=8,
        )

    def action_new_source(self) -> None:
        """Open the create-source form when in the Sources section."""
        if self.active_section != "sources":
            self.active_section = "sources"
            self._pending_open_create_form = True
            return
        if self.is_mounted:
            try:
                pane = self.query_one("#watchlists-sources-pane", SourcesPane)
                pane.show_create_form = True
            except Exception:
                pass

    def action_delete_selected(self) -> None:
        """Delete the currently selected entity after confirmation."""
        entity = self.selected_entity
        if entity is None:
            self.app_instance.notify(
                "Nothing selected to delete.",
                severity="warning",
            )
            return
        self.handle_delete_requested(DeleteRequested(entity))

    def action_check_now_selected(self) -> None:
        """Trigger a check now on the selected source."""
        entity = self.selected_entity
        if entity is None or InspectorPane._entity_type(entity) != "source":
            self.app_instance.notify(
                "Select a source to check.",
                severity="warning",
            )
            return
        self.handle_check_now_requested(CheckNowRequested(entity))

    def action_preview_selected(self) -> None:
        """Preview the selected source."""
        entity = self.selected_entity
        if entity is None or InspectorPane._entity_type(entity) != "source":
            self.app_instance.notify(
                "Select a source to preview.",
                severity="warning",
            )
            return
        self.handle_preview_requested(PreviewRequested(entity))
