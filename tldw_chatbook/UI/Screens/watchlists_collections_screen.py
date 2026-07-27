"""Watchlists destination shell.

The route, class name, and stable widget selectors retain the historical
``watchlists_collections``/``wc`` identifiers so older tests, shortcuts, and
handoffs keep working while Collections moves under Library.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import events, on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Select, Static

from ...Constants import (
    WATCHLISTS_NAV_CONTEXT_BACKEND,
    WATCHLISTS_NAV_CONTEXT_RUN_ID,
    WATCHLISTS_NAV_CONTEXT_SECTION,
    WATCHLISTS_SECTION_RUNS,
)
from ...runtime_policy.types import PolicyDeniedError
from ...Subscriptions.watchlist_bundle_service import WatchlistBundleService
from ...Utils.input_validation import sanitize_string, validate_text_input
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from ..Subscription_Modules.notifications_inbox_controller import (
    NotificationsInboxController,
)
from ..Watchlists_Modules.inspector_pane import (
    BreadcrumbScopeSelected,
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
from ..Watchlists_Modules.notifications_pane import (
    DismissNotificationRequested,
    MarkNotificationReadRequested,
    NotificationSelected,
    NotificationsPane,
    RefreshNotificationsRequested,
)
from ..Watchlists_Modules.opml_dialogs import (
    ConfirmDeleteDialog,
    OpmlExportDialog,
    OpmlImportDialog,
)
from ..Watchlists_Modules.overview_pane import OverviewPane
from ..Watchlists_Modules.region_layout import CENTRE_REGIONS, Region, RegionLayout
from ..Watchlists_Modules.region_layout_store import load_region_layout, save_region_layout
from ..Watchlists_Modules.rules_pane import (
    RefreshRulesRequested,
    RuleFormVisibilityChanged,
    RuleSelected,
    RulesPane,
    SaveRuleRequested,
)
from ..Watchlists_Modules.runs_pane import CancelRunRequested, RerunRunRequested, RunsPane, RunSelected
from ..Watchlists_Modules.sources_pane import (
    CreateFormDraftChanged,
    CreateFormVisibilityChanged,
    CreateSourceRequested,
    ExportOpmlRequested,
    ImportOpmlRequested,
    SourceSelected,
    SourcesPane,
)
from ..Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged, WatchlistTree
from ..Watchlists_Modules.watchlists_backend_controller import WatchlistsBackendController
from ..Watchlists_Modules.watchlists_console_handoff import WatchlistsConsoleHandoff
from ..Watchlists_Modules.watchlists_tab_strip import SectionSelected, WatchlistsTabStrip
from ..Watchlists_Modules.watchlists_workbench import RegionToggled, WatchlistsWorkbench
from .destination_recovery import DestinationRecoveryState, policy_denied_recovery_state


logger = logger.bind(module="WatchlistsCollectionsScreen")
WC_LOCAL_PAGE_SIZE = 5
WC_SERVICE_ERROR_COPY = "Watchlists services unavailable; retry Watchlists later."
WC_SERVICE_UNAVAILABLE_COPY = "Watchlists services are unavailable in this runtime."
WC_SNAPSHOT_TIMEOUT_SECONDS = 1.5


class WatchlistsCollectionsScreen(BaseAppScreen):
    """Monitored sources, runs, alerts, and recovery."""

    BINDINGS = [
        ("1", "switch_section('overview')", "Overview"),
        ("2", "switch_section('sources')", "Sources"),
        ("3", "switch_section('items')", "Items"),
        ("4", "switch_section('runs')", "Runs"),
        ("5", "switch_section('rules')", "Rules"),
        ("6", "switch_section('notifications')", "Notifications"),
        ("question", "show_help", "Help"),
        ("n", "new_source", "New source"),
        ("d", "delete_selected", "Delete"),
        ("c", "check_now_selected", "Check now"),
        ("p", "preview_selected", "Preview"),
        ("z", "toggle_region", "Collapse"),
        ("Z", "solo_region", "Solo"),
        ("left_square_bracket", "toggle_left_rail", "Left rail"),
        ("right_square_bracket", "toggle_right_rail", "Right rail"),
    ]

    active_section = reactive("overview")
    runtime_backend = reactive("local")
    selected_source = reactive(None)
    selected_run = reactive(None)
    selected_notification = reactive(None)
    selected_entity = reactive(None)
    recovery_state = reactive(None)
    overview_data = reactive({}, recompose=True)
    # CONTENT hosts the Phase D reader stub, so it starts collapsed to avoid
    # spending screen space on a placeholder. `on_mount` overlays whatever is
    # actually persisted (see `region_layout_store`) on top of this default.
    region_layout = reactive(RegionLayout(collapsed=frozenset({Region.CONTENT})))
    focused_region = reactive(Region.FEEDS)
    # Two scopes, deliberately: they answer different questions and they
    # diverge (fix round 1, Finding 2).
    #
    # `tree_scope` is where the user has NAVIGATED -- the tree node in view.
    # It drives the Feeds region (`scoped_source_rows`), and only a tree
    # click or a breadcrumb promotion moves it.
    #
    # `selected_scope` is the ancestry the Inspector is entitled to CLAIM
    # for whatever is currently selected. It follows `tree_scope` on a tree
    # move, but resets to "all" when a pane row is selected, because a
    # Sources/Runs/Items/Rules row carries no watchlist/source ancestry --
    # asserting one would put a breadcrumb over an entity that may not
    # belong to it (Task 5 fix round 2, Finding 1).
    #
    # Task 7 made `selected_scope` drive Feeds as well, which silently
    # merged the two: clicking a pane row to inspect it then reset the Feeds
    # region back to "All sources", discarding tree navigation the user had
    # done in another region. Splitting them keeps both properties -- the
    # Feeds region follows the tree, and the Inspector still claims no
    # ancestry it does not know. Clearing `_breadcrumb_labels` alone would
    # NOT have been enough: `InspectorPane._scope_levels` derives an
    # ancestor level from `scope` alone and falls back to a `Watchlist {id}`
    # label, so the crumb would still render, just anonymously.
    #
    # Both live on the screen, not on the tree widget, precisely because
    # `region_layout` is `recompose=True`: any collapse/solo/rail toggle
    # rebuilds the whole workbench, constructing a brand new `WatchlistTree`
    # instance. Pane-local state does not survive that (see `selected_run`
    # and the create-form draft above for the same reasoning already applied
    # elsewhere on this screen).
    selected_scope = reactive(TreeScope(kind="all"))
    tree_scope = reactive(TreeScope(kind="all"))

    _SECTION_DETAIL_TITLE = {
        "overview": "Overview",
        "sources": "Sources",
        "items": "Items",
        "runs": "Runs",
        "rules": "Rules",
        "notifications": "Notifications",
    }

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "watchlists_collections", **kwargs)
        self._console_handoff = WatchlistsConsoleHandoff(app_instance)
        self._local_watchlist_records: tuple[Mapping[str, Any], ...] = ()
        self._local_watchlist_count = 0
        self._watchlist_total_known = True
        self._wc_lookup_error: str | None = None
        self._wc_lookup_recovery_state: DestinationRecoveryState | None = None
        self._wc_loaded = False
        self._pending_open_create_form = False
        self._pending_open_import_opml = False
        self._pending_delete_entity: dict[str, Any] | None = None
        self._pending_navigation_run_id: str | None = None
        self._pending_navigation_run_backend: str | None = None
        self._loaded_runs: list[dict[str, Any]] = []
        self._loaded_notifications: list[dict[str, Any]] = []
        # Mirrors what's currently loaded for Sources/Items/Rules the same way
        # `_loaded_runs`/`_loaded_notifications` already do (Finding 2, fix
        # round 2): `_build_detail_pane` constructs a brand new
        # SourcesPane/ItemsPane/RulesPane on every workbench rebuild (any
        # region collapse/solo/rail toggle, not just switching sections), and
        # a fresh pane's `sources`/`items`/`rules` reactive starts at its
        # class default (`[]`). Without holding the last-loaded rows here and
        # re-seeding them below, the table would render empty until the next
        # unrelated navigation happened to trigger a reload.
        self._loaded_sources: list[dict[str, Any]] = []
        self._loaded_items: list[dict[str, Any]] = []
        self._loaded_rules: list[dict[str, Any]] = []
        # Left-rail tree inputs (Task 4): loaded together by `_load_tree_data`
        # in exactly two queries (`list_watchlists` + `get_watchlist_item_counts`),
        # never one per node -- see that method's docstring.
        self._tree_watchlists: list[dict[str, Any]] = []
        self._tree_counts: dict[int, dict[str, int]] = {}
        # Breadcrumb display names for `selected_scope` (Task 5 fix round 1):
        # resolved once in `_on_tree_scope_changed`, not on every Inspector
        # render, and held here for the same reason `selected_scope` itself
        # is screen-held -- `_build_inspector_pane` re-seeds a brand new
        # `InspectorPane` on every workbench rebuild, and a fresh pane's own
        # reactive would otherwise start back at its class default.
        self._breadcrumb_labels: list[str] = []
        self._applying_navigation_context = False
        # Mirrors SourcesPane's create-form state (Finding 1, fix round 1):
        # `region_layout` is `recompose=True`, so any collapse/solo/rail
        # toggle rebuilds the whole workbench, constructing a brand new
        # SourcesPane. Without holding the draft here — the same way
        # selected_source/selected_run/active_section already survive pane
        # rebuilds — a half-typed create form would be silently destroyed by
        # a keybinding that has nothing to do with Sources.
        self._source_create_form_open = False
        self._source_create_draft: dict[str, str] = {"name": "", "url": "", "tags": ""}
        # Mirrors RulesPane's edit-form state (Finding 4, fix round 2): the
        # same rebuild-destroys-pane-local-state failure mode as the Sources
        # create form above, but for an in-progress rule EDIT rather than a
        # create. `_rule_form_editing` holds the rule being edited, or `None`
        # when the open form is for a brand new rule.
        self._rule_form_open = False
        self._rule_form_editing: dict[str, Any] | None = None
        # Layout-persistence bookkeeping (PR #926 review, Bug 2): avoids
        # writing to config on every `_apply_layout` call — `on_mount`'s
        # initial push-back of the just-loaded layout, and every no-op
        # toggle, would otherwise trigger `save_setting_to_cli_config`'s
        # synchronous whole-file read-modify-write on the UI thread. See
        # `_schedule_layout_persist`/`_persist_layout_worker` below.
        self._last_persisted_collapsed: frozenset[Region] | None = None
        self._pending_persist_layout: RegionLayout | None = None
        self._layout_persist_lock = threading.Lock()
        self._controller = WatchlistsBackendController(
            app_instance=app_instance,
            scope_service=getattr(app_instance, "watchlist_scope_service", None),
            server_service=getattr(app_instance, "server_watchlists_service", None),
        )
        self._notifications_controller = NotificationsInboxController(
            app_instance=app_instance,
            store=getattr(app_instance, "client_notifications_db", None),
        )

    def _watchlist_bundle_service(self) -> WatchlistBundleService | None:
        """The live watchlist bundle service, or ``None`` if unavailable.

        Mirrors how the screen reaches ``watchlist_scope_service``: via
        ``getattr(..., None)`` on the app instance, so the tree and other
        callers degrade rather than crash when the service has not been
        wired (e.g. a bare app stub in tests).
        """
        return getattr(self.app_instance, "watchlist_bundle_service", None)

    def on_mount(self) -> None:
        super().on_mount()
        # Push the persisted layout into the already-mounted workbench, not just
        # this screen's own reactive: `compose_content` already ran by the time
        # `on_mount` fires (compose always precedes the Mount event), so the
        # WatchlistsWorkbench child was built with whatever `region_layout` held
        # at THAT moment. Without also reaching into the mounted workbench via
        # `_apply_layout`, a persisted collapse would silently not render until
        # some unrelated later recompose happened to pick it up.
        loaded_layout = load_region_layout()
        # Prime the "last persisted" marker with what we just read (PR #926
        # review, Bug 2) BEFORE calling `_apply_layout`: this value is by
        # definition already on disk, so pushing it back into the workbench
        # here must not itself trigger a redundant write.
        self._last_persisted_collapsed = loaded_layout.collapsed_for_persistence()
        self._apply_layout(loaded_layout)
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()
        self._load_active_section_data()
        self._load_tree_data()
        self.set_timer(
            WC_SNAPSHOT_TIMEOUT_SECONDS, self._apply_snapshot_timeout_if_still_loading
        )

    def apply_navigation_context(self, context: Mapping[str, Any]) -> None:
        """Apply a validated section/run deep link from shell navigation."""
        section = str(context.get(WATCHLISTS_NAV_CONTEXT_SECTION) or "").strip()
        if section not in self._SECTION_DETAIL_TITLE:
            return

        requested_backend = str(
            context.get(WATCHLISTS_NAV_CONTEXT_BACKEND) or self.runtime_backend
        ).strip()
        if requested_backend not in {"local", "server"}:
            requested_backend = self.runtime_backend

        self._applying_navigation_context = True
        try:
            self.runtime_backend = requested_backend
            self.active_section = section
        finally:
            self._applying_navigation_context = False

        run_id = context.get(WATCHLISTS_NAV_CONTEXT_RUN_ID)
        self._pending_navigation_run_id = (
            str(run_id).strip()
            if section == WATCHLISTS_SECTION_RUNS and run_id not in (None, "")
            else None
        )
        self._pending_navigation_run_backend = (
            requested_backend if self._pending_navigation_run_id else None
        )
        if self.is_mounted:
            self._load_active_section_data()

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

    @work(exclusive=True, group="wc_tree")
    async def _load_tree_data(self) -> None:
        """Load the left-rail tree's two inputs: watchlists and counts.

        Exactly two queries total, never one per node: `list_watchlists()`
        for the watchlist rows themselves, and `get_watchlist_item_counts()`
        for every bucket's total/unread counts in a single statement. Both
        are reached through `WatchlistBundleService` (Task 1) rather than a
        second accessor onto `SubscriptionsDB` directly.
        """
        try:
            service = self._watchlist_bundle_service()
            self._tree_watchlists = service.list_watchlists()
            self._tree_counts = service.get_watchlist_item_counts()
        except Exception:
            logger.opt(exception=True).debug("Failed to load watchlists tree data.")
            self._tree_watchlists, self._tree_counts = [], {}
        if self.is_mounted:
            self.refresh(recompose=True)

    def _resolve_breadcrumb_labels(self, scope: TreeScope) -> list[str]:
        """Display names for `scope`'s ancestor chain, for the Inspector.

        Called once from `_apply_tree_scope` -- itself invoked from a real
        tree click (`_on_tree_scope_changed`) and a breadcrumb promotion
        (`handle_breadcrumb_scope_selected`), both discrete, user-driven
        events -- never from a render path, so this is not a query-per-render:
        the watchlist name costs nothing (`_tree_watchlists` is already
        loaded by `_load_tree_data`), and a source name costs exactly the one
        `list_source_rows` JOIN the tree itself already uses to expand a
        watchlist, only when the scope actually names a source.
        """
        if scope.kind not in ("watchlist", "source") or scope.watchlist_id is None:
            return []

        labels = [
            next(
                (
                    str(watchlist.get("name"))
                    for watchlist in self._tree_watchlists
                    if int(watchlist.get("id", -1)) == int(scope.watchlist_id)
                ),
                f"Watchlist {scope.watchlist_id}",
            )
        ]

        if scope.kind == "source" and scope.source_id is not None:
            source_label = f"Source {scope.source_id}"
            service = self._watchlist_bundle_service()
            if service is not None:
                try:
                    rows = service.list_source_rows(scope.watchlist_id)
                    source_label = next(
                        (
                            str(row.get("name"))
                            for row in rows
                            if int(row.get("id", -1)) == int(scope.source_id)
                        ),
                        source_label,
                    )
                except Exception:
                    logger.opt(exception=True).debug(
                        "Failed to resolve breadcrumb source name."
                    )
            labels.append(source_label)

        return labels

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
        # Recorded but no longer rendered (fix round 1, Finding 1): the
        # "(showing up to N)" qualifier belonged to the per-record listing
        # the staging block used to draw. Kept on the screen -- and in this
        # signature, which the parity guards call positionally -- because it
        # is still the honest answer to "was that count a total or a page?"
        # for anything that needs it next.
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
        """Whether there is anything for Console staging to send.

        Asks the tree scope as well as the async snapshot (fix round 1,
        Finding 1). Staging now sends `scoped_source_rows()`, so a gate that
        only consulted the snapshot could disable the Stage button -- and
        render "No sources yet." -- directly underneath a Feeds list that
        was showing rows. That split is not hypothetical: the two paths
        resolve their `SubscriptionsDB` independently, and in the UI
        harnesses they land on different temp files entirely.

        The snapshot stays the *health* probe (`_wc_loaded` /
        `_wc_lookup_error` in `_wc_attach_state` and `_build_list_pane` are
        untouched): it is the only caller that can distinguish "the service
        is unavailable" or "policy denied" from "there are no rows", which a
        synchronous local query cannot report.
        """
        if self._local_watchlist_count > 0:
            return True
        return bool(self.scoped_source_rows())

    def _staging_summary_line(self, rows: Sequence[Mapping[str, Any]]) -> Text:
        """The one line the Console-staging block collapses to.

        Fix round 1, Finding 1: the block used to enumerate
        `_local_watchlist_records`, which reaches `get_all_subscriptions`
        through `WatchlistScopeService.list_watch_items` -- the *same* table
        `scoped_source_rows()` reads. FEEDS therefore printed every source
        twice in one box (once scoped, once not), in identical typography.
        Staging now follows the tree scope instead, so the block only has to
        say what pressing the button would send.

        Args:
            rows: The current scope's rows, as returned by
                `scoped_source_rows()`.

        Returns:
            A single-line ``Text``, escaped -- the scope label can be a
            remote feed's own title.
        """
        label = escape_markup(self._tree_scope_label(rows))
        noun = "source" if len(rows) == 1 else "sources"
        return Text.from_markup(
            f"Local Watchlists snapshot: {label} ({len(rows)} {noun})"
        )

    def _snapshot_body(self) -> str:
        """The Console handoff body: the sources the tree scope covers.

        Reads the same `scoped_source_rows()` the Feeds region renders (fix
        round 1, Finding 1), so selecting "Morning AI Brief" and then
        staging stages Morning AI Brief -- not, as before, every local
        source regardless of where the user had navigated.

        Names still go through `_record_title` (and therefore `_safe_text`'s
        sanitise + validate pass), unchanged: a source name can come
        straight from a remote feed, and this string is handed to a chat
        prompt.
        """
        rows = self.scoped_source_rows()
        label = self._safe_text(self._tree_scope_label(rows), "the current scope")
        lines = [f"Local Watchlists snapshot staged for Console: {label}", ""]
        lines.append(f"Sources: {len(rows)}")
        for index, row in enumerate(rows[:WC_LOCAL_PAGE_SIZE], start=1):
            lines.append(f"  {index}. {self._record_title(row)}")
        remainder = len(rows) - WC_LOCAL_PAGE_SIZE
        if remainder > 0:
            lines.append(f"  ... and {remainder} more")
        return "\n".join(lines).strip()

    def _snapshot_metadata(self) -> dict[str, Any]:
        """Structured companion to `_snapshot_body`.

        `watchlist_count`/`watchlist_sample_count`/`watchlist_titles` keep
        describing the local snapshot exactly as before -- they are the
        service-reported inventory, and Console consumers already read them.
        The `scope_*`/`source_*` keys are what the press actually staged.
        """
        rows = self.scoped_source_rows()
        scope = self.tree_scope
        return {
            "watchlist_count": self._local_watchlist_count,
            "watchlist_sample_count": len(self._local_watchlist_records),
            "watchlist_titles": [
                self._record_title(record) for record in self._local_watchlist_records
            ],
            "scope_kind": scope.kind,
            "scope_label": self._safe_text(
                self._tree_scope_label(rows), "the current scope"
            ),
            "scope_watchlist_id": scope.watchlist_id,
            "scope_source_id": scope.source_id,
            "source_count": len(rows),
            "source_titles": [self._record_title(row) for row in rows],
            "backend": "local",
        }

    def _wc_attach_state(self) -> tuple[bool, str]:
        """Whether "Stage Watchlists Context in Console" should be enabled,
        and its tooltip — the same loading/error/empty/populated branching
        `_build_list_pane` uses, split out so `compose_content` can get it
        without constructing (and discarding) a pane widget just to read it.
        """
        if not self._wc_loaded:
            return True, "Stage local Watchlists context after the local snapshot loads."
        if self._wc_lookup_error:
            recovery_state = self._wc_lookup_recovery_state
            tooltip = (
                recovery_state.disabled_tooltip
                if recovery_state is not None
                else "Watchlists services are unavailable; retry Watchlists before staging Console context."
            )
            return True, tooltip
        if not self._has_local_wc_context():
            return True, "Stage local Watchlists context once local sources exist."
        return False, "Stage local Watchlists context in Console."

    def _build_tree_pane(self) -> WatchlistTree:
        """Build the LEFT_RAIL-region content: the watchlist tree.

        A factory, not an instance: `region_layout` is `recompose=True`, so
        any collapse/solo/rail toggle rebuilds every region, and a widget
        instance can only be mounted once (see the factory note on
        `WatchlistsWorkbench.__init__`).
        """
        return WatchlistTree(
            watchlists=self._tree_watchlists,
            counts=self._tree_counts,
            source_rows_loader=self._load_source_rows_for_tree,
            id="wl-tree",
        )

    def _load_source_rows_for_tree(self, watchlist_id: int) -> list[dict[str, Any]]:
        """Fetch one watchlist's source rows for the tree, synchronously.

        Safe on the UI thread: the tree calls this during `compose()` when a
        watchlist is expanded, and `list_source_rows` is one JOIN (Task 1),
        not a fan-out of per-source queries.
        """
        try:
            return self._watchlist_bundle_service().list_source_rows(watchlist_id)
        except Exception:
            logger.opt(exception=True).debug("Failed to load tree source rows.")
            return []

    def scoped_source_rows(self) -> list[dict[str, Any]]:
        """Source rows the current tree scope covers.

        The Feeds region renders these, so selecting a node in the tree
        actually narrows what the centre shows rather than only recording a
        selection (Task 7). Kept on the screen (not the pane) because the
        workbench recomposes and pane-local state does not survive it -- the
        same reasoning already applied to `tree_scope` itself.

        Reads `tree_scope`, not `selected_scope`: only tree navigation
        changes what Feeds covers. See the note on those two reactives for
        why they are not the same value.

        Each branch costs exactly one query (`list_source_rows`,
        `list_all_source_rows`, or `list_unassigned_source_rows`); the
        `source` scope reuses whichever of those already names the right
        table rather than adding a second query just to filter down to one
        row.

        Returns:
            One dict per source with ``id``, ``name`` and ``type``, or an
            empty list if the bundle service is unavailable or lookup fails.
        """
        service = self._watchlist_bundle_service()
        if service is None:
            return []
        scope = self.tree_scope
        try:
            if scope.kind == "watchlist" and scope.watchlist_id is not None:
                return service.list_source_rows(scope.watchlist_id)
            if scope.kind == "source" and scope.source_id is not None:
                rows = (
                    service.list_source_rows(scope.watchlist_id)
                    if scope.watchlist_id is not None
                    else service.list_all_source_rows()
                )
                return [r for r in rows if int(r["id"]) == int(scope.source_id)]
            if scope.kind == "unassigned":
                return service.list_unassigned_source_rows()
            return service.list_all_source_rows()
        except Exception:
            logger.opt(exception=True).debug("Failed to resolve scoped source rows.")
            return []

    def _tree_scope_label(self, rows: Sequence[Mapping[str, Any]]) -> str:
        """A human name for `tree_scope` -- "All sources", "Unassigned", a
        watchlist's name, or a single source's name.

        Resolved from data already in memory (`_tree_watchlists`, and `rows`
        itself for a source's own name) rather than by issuing another query
        -- `rows` is the same list `scoped_source_rows()` just resolved, so
        this is display formatting, not a second lookup.

        The returned string is RAW (a watchlist name is user-authored and a
        source name can come straight from a remote feed's own title), so
        every caller must escape it before it reaches a rendered label.

        Args:
            rows: The current scope's rows, as returned by
                `scoped_source_rows()`.

        Returns:
            The scope's display name, unescaped.
        """
        scope = self.tree_scope
        if scope.kind == "unassigned":
            return "Unassigned"
        if scope.kind == "watchlist" and scope.watchlist_id is not None:
            return next(
                (
                    str(watchlist.get("name"))
                    for watchlist in self._tree_watchlists
                    if int(watchlist.get("id", -1)) == int(scope.watchlist_id)
                ),
                f"Watchlist {scope.watchlist_id}",
            )
        if scope.kind == "source":
            if rows:
                return str(rows[0].get("name"))
            if scope.source_id is not None:
                return f"Source {scope.source_id}"
        return "All sources"

    def _scoped_feeds_heading(self, rows: Sequence[Mapping[str, Any]]) -> Text:
        """The scope-named Feeds heading, e.g. ``Feeds in Morning AI Brief (3)``.

        Replaces the previous hardcoded "Sources" title (Task 7): with the
        Sources tab active, that hardcoded word duplicated ITEMS's own
        `_SECTION_DETAIL_TITLE["sources"]` heading in the adjacent box. This
        also makes the heading say what Feeds is actually showing, since a
        tree click now changes that.

        Args:
            rows: The current scope's rows, as returned by
                `scoped_source_rows()` -- passed in rather than re-resolved
                so the caller (which needs the rows anyway, to render them)
                does not pay for the query twice.

        Returns:
            A single-line ``Text``, pre-parsed via `Text.from_markup` over
            an escaped label -- the same "escape untrusted content, then
            build a `Text` rather than hand a raw f-string to `Static`"
            convention `_build_inspector_pane`'s follow-in-Console line
            already uses, since the label may come straight from a remote
            feed's own title.
        """
        label = escape_markup(self._tree_scope_label(rows))
        return Text.from_markup(f"Feeds in {label} ({len(rows)})")

    def _build_list_pane(self) -> Vertical:
        """Build the FEEDS-region content: the section tab strip, a heading
        naming the current tree scope, that scope's source rows, the
        snapshot's own loading/error/empty markers, and a one-line summary
        of what Console staging would send.

        The source list appears ONCE (fix round 1, Finding 1). Task 7 added
        the scoped rows above a block that already enumerated
        `_local_watchlist_records` -- which resolve, via
        `WatchlistScopeService.list_watch_items` ->
        `local_watchlists_service.list_sources` -> `get_all_subscriptions`,
        to the same `subscriptions` table the scope resolvers read. Every
        source therefore printed twice in one box, in identical typography.
        Staging now follows the tree scope (see `_snapshot_body`), so that
        block collapses to a single line.

        The loading/error/empty markers stay keyed on the async snapshot,
        NOT on `scoped_source_rows()`: the snapshot is the only
        service-health probe on this screen -- it is what distinguishes
        "the Watchlists service is unavailable" and "policy denied" (whose
        recovery state supplies `#wc-service-error`'s copy) from "there are
        no rows". `scoped_source_rows()` is a synchronous local query that
        returns `[]` for every one of those cases, and `#wc-loading-state`
        has no meaning for it at all. In production the two agree anyway:
        both read `subscriptions`.

        The tab strip is unchanged -- `Tests/UI/test_destination_shells.py`
        and `Tests/UI/test_destination_visual_parity_correction.py` both
        drive its stable selectors.

        Byte-identical logic to the pre-rehost inline composition for the
        snapshot itself; only the `yield` calls became list appends and a
        `Vertical(...)` return so the result can be handed to
        `WatchlistsWorkbench` as a content factory instead of being mounted
        directly by `compose_content`. The tab strip is prepended here
        (rather than left unwired) so section-switching by click is not lost
        now that the navigator is retired — `Region.LEFT_RAIL` will take over
        the tree once Task 4 wires it in, at which point this stays the
        strip's permanent home per the design (a one-row strip at the top of
        the centre).

        This is called fresh on every region rebuild (see
        `WatchlistsWorkbench.__init__`'s docstring on why `content` holds
        factories, not instances), so it must stay side-effect-free.
        """
        scoped_rows = self.scoped_source_rows()
        children: list[Widget] = [
            WatchlistsTabStrip(active_section=self.active_section, id="wl-tabs"),
            Static(
                self._scoped_feeds_heading(scoped_rows),
                classes="destination-section watchlists-column-title",
                id="wl-feeds-scope-heading",
            ),
        ]
        for row in scoped_rows:
            # Source names are untrusted (imported OPML, a remote feed's own
            # title, ...), so they must be escaped before reaching a
            # rendered label -- this repo has shipped that bug before.
            name = escape_markup(str(row.get("name") or ""))
            source_type = escape_markup(str(row.get("type") or ""))
            children.append(
                Static(
                    Text.from_markup(f"{name}  ({source_type})"),
                    id=f"wl-feeds-source-{row.get('id')}",
                    classes="watchlist-feed-source-row",
                )
            )
        if not self._wc_loaded:
            children.append(
                Static(
                    "Loading local Watchlists snapshot...",
                    id="wc-loading-state",
                )
            )
        elif self._wc_lookup_error:
            recovery_state = self._wc_lookup_recovery_state
            children.append(
                Static(
                    self._wc_lookup_error,
                    id=(
                        recovery_state.stable_selector
                        if recovery_state is not None
                        else "wc-service-error"
                    ),
                )
            )
        elif not self._has_local_wc_context():
            children.append(
                Static(
                    "No sources yet.",
                    id="wc-empty-state",
                )
            )
            children.append(
                Horizontal(
                    Button(
                        "Create source",
                        id="wc-empty-create-source",
                        variant="primary",
                        tooltip="Add a new Watchlists source.",
                    ),
                    Button(
                        "Import OPML",
                        id="wc-empty-import-opml",
                        tooltip="Import sources from an OPML file.",
                    ),
                    id="wc-empty-actions",
                    classes="destination-filter-strip",
                )
            )
        else:
            # One line, not a second source list (fix round 1, Finding 1).
            # `#wc-watchlists-summary` keeps its id -- it is the "snapshot
            # finished loading" terminal selector the guard suites wait on
            # -- and now says what pressing Stage would send, which is the
            # scope above it. `#wc-snapshot-title` is folded into this same
            # line rather than kept as a separate heading; no test
            # referenced it, and a one-line block does not need a title row.
            children.append(
                Static(
                    self._staging_summary_line(scoped_rows),
                    id="wc-watchlists-summary",
                    classes="destination-section",
                )
            )
        return Vertical(
            *children,
            id="watchlists-list-pane",
            classes="destination-workbench-pane",
        )

    def _build_detail_pane(self) -> Vertical:
        """Build the ITEMS-region content: the active-section-routed pane.

        Called fresh on every region rebuild — see the factory note on
        `WatchlistsWorkbench.__init__`.
        """
        detail_title = self._SECTION_DETAIL_TITLE.get(self.active_section, "Detail")
        children: list[Widget] = [
            Static(
                detail_title,
                classes="destination-section watchlists-column-title",
                id="watchlists-detail-title",
            )
        ]
        if self.active_section == "overview":
            overview = OverviewPane(id="watchlists-overview-pane")
            overview.data = self.overview_data
            children.append(overview)
        elif self.active_section == "sources":
            sources_pane = SourcesPane(id="watchlists-sources-pane")
            # Seed the last-loaded rows and selection (Finding 2, fix round
            # 2) the same way RunsPane/NotificationsPane already do below —
            # without this the table renders empty until the next unrelated
            # navigation happens to trigger `_load_sources` again.
            sources_pane.sources = self._loaded_sources
            sources_pane.selected_source = self.selected_source
            # Seed the create-form draft so it survives this pane being
            # reconstructed (see the note on `_source_create_draft` in
            # __init__ and CreateFormDraftChanged/CreateFormVisibilityChanged
            # in sources_pane.py).
            sources_pane.show_create_form = self._source_create_form_open
            sources_pane.create_draft_name = self._source_create_draft["name"]
            sources_pane.create_draft_url = self._source_create_draft["url"]
            sources_pane.create_draft_tags = self._source_create_draft["tags"]
            children.append(sources_pane)
        elif self.active_section == "runs":
            runs_pane = RunsPane(id="watchlists-runs-pane")
            runs_pane.runs = self._loaded_runs
            runs_pane.selected_run = self.selected_run
            children.append(runs_pane)
        elif self.active_section == "items":
            # Seed the last-loaded rows (Finding 2, fix round 2) — see the
            # note on `sources_pane.sources` above; same rebuild, same gap.
            items_pane = ItemsPane(id="watchlists-items-pane")
            items_pane.items = self._loaded_items
            children.append(items_pane)
        elif self.active_section == "rules":
            # Seed the last-loaded rows (Finding 2, fix round 2) — see the
            # note on `sources_pane.sources` above; same rebuild, same gap.
            rules_pane = RulesPane(id="watchlists-rules-pane")
            rules_pane.rules = self._loaded_rules
            # Seed the edit-form state so an in-progress rule edit survives
            # this pane being reconstructed (Finding 4, fix round 2) — the
            # same treatment the Sources create-form draft already gets
            # above; see `_rule_form_open`/`_rule_form_editing` in __init__
            # and RuleFormVisibilityChanged in rules_pane.py.
            if self._rule_form_open:
                if self._rule_form_editing is not None:
                    rules_pane.edit_rule(self._rule_form_editing)
                else:
                    rules_pane.show_rule_form = True
            children.append(rules_pane)
        elif self.active_section == "notifications":
            notifications_pane = NotificationsPane(id="watchlists-notifications-pane")
            notifications_pane.notifications = self._loaded_notifications
            notifications_pane.selected_notification = self.selected_notification
            children.append(notifications_pane)
        return Vertical(
            *children,
            id="watchlists-detail-pane",
            classes="destination-workbench-pane",
        )

    def _build_inspector_pane(
        self,
        latest_console_item: Any,
        attach_disabled: bool,
        attach_tooltip: str,
    ) -> Vertical:
        """Build the RIGHT_RAIL-region content: state summaries, Console
        actions, and the entity Inspector.

        `latest_console_item`/`attach_disabled`/`attach_tooltip` are captured
        once per `compose_content` call and passed in rather than
        recomputed, since a factory wrapping this method (see
        `compose_content`) is called on every region rebuild.
        """
        children: list[Widget] = [
            Static(
                "Inspector",
                classes="destination-section watchlists-column-title",
            ),
            Static(
                "State: ready"
                if self._wc_loaded and not self._wc_lookup_error
                else "State: unavailable",
                id="watchlists-state-summary",
            ),
            Static(
                f"Alert rules active: {self.overview_data.get('active_alert_rules', 0)}",
                id="watchlists-alerts-summary",
            ),
            Static(
                f"Latest run status: {self.overview_data.get('latest_run_status', 'unavailable')}",
                id="watchlists-latest-run-summary",
            ),
            Static("Console actions", classes="destination-section"),
            Button(
                "Stage Watchlists Context in Console",
                id="wc-attach-to-console",
                disabled=attach_disabled,
                tooltip=attach_tooltip,
            ),
            Button(
                "Open current Watchlists",
                id="wc-open-watchlists",
                tooltip="Open the current watchlist/subscription surface.",
            ),
        ]
        if latest_console_item is not None:
            title = str(getattr(latest_console_item, "title", None) or "Untitled")
            status = str(getattr(latest_console_item, "status", None) or "unknown")
            children.append(
                Static(
                    Text.from_markup(
                        "Console can follow latest Watchlists run: "
                        f"{escape_markup(title)} ({escape_markup(status)})."
                    ),
                    id="watchlists-console-available",
                )
            )
            children.append(
                Button(
                    Text.from_markup(f"Follow {escape_markup(title)} in Console"),
                    id="watchlists-follow-in-console",
                    tooltip="Open the latest active Watchlists run in Console.",
                )
            )
        else:
            children.append(
                Static(
                    "No active Watchlists run is available for Console follow.",
                    id="watchlists-console-unavailable",
                )
            )
            children.append(
                Button(
                    "Console follow unavailable",
                    id="watchlists-follow-in-console",
                    disabled=True,
                    tooltip="Unavailable until Watchlists has an active run with Console context.",
                )
            )
        # Seed from screen state (Finding 3, fix round 2): `region_layout` is
        # `recompose=True`, so any collapse/solo/rail toggle constructs a
        # brand new InspectorPane. Without this, the screen keeps
        # `selected_entity` but the freshly-built Inspector starts at its
        # class default (`None`) until the NEXT explicit selection change —
        # `watch_selected_entity` only pushes on change, so a rebuild alone
        # never re-syncs it. That left `d`/`c`/`p` silently operating on a
        # selection the user could no longer see.
        #
        # `scope`/`breadcrumb_labels` get the identical treatment (Task 5 fix
        # round 1) and for the identical reason: a `[`/`]`/`z`/`Z` toggle
        # rebuilds this factory from scratch, and `watch_selected_scope`
        # alone would leave a freshly-built Inspector's breadcrumb blank
        # until the next tree click.
        inspector = InspectorPane(id="watchlists-entity-inspector")
        inspector.selected_entity = self.selected_entity
        inspector.scope = self.selected_scope
        inspector.breadcrumb_labels = self._breadcrumb_labels
        children.append(inspector)
        return Vertical(
            *children,
            id="watchlists-inspector-pane",
            classes="destination-workbench-pane ds-inspector",
        )

    def compose_content(self) -> ComposeResult:
        latest_console_item = self._console_handoff.resolve_latest_follow_item()
        with Vertical(id="watchlists-collections-shell"):
            yield Static(
                "Watchlists | Monitored sources, runs, alerts, recovery | Mixed | Local/Server",
                id="watchlists-collections-title",
                classes="ds-destination-header",
            )
            with Horizontal(id="watchlists-header-bar", classes="destination-filter-strip"):
                yield Select(
                    [("Local", "local"), ("Server", "server")],
                    value=self.runtime_backend,
                    id="watchlists-backend-select",
                    allow_blank=False,
                    disabled=self.active_section == "notifications",
                    tooltip=(
                        "The notifications inbox is local to this device."
                        if self.active_section == "notifications"
                        else "Choose the Watchlists data backend."
                    ),
                )
                yield Static(
                    (
                        "Inbox: local"
                        if self.active_section == "notifications"
                        else f"Backend: {self.runtime_backend}"
                    ),
                    id="watchlists-backend-label",
                )
            attach_disabled, attach_tooltip = self._wc_attach_state()
            yield WatchlistsWorkbench(
                self.region_layout,
                content={
                    # Factories, not instances: `region_layout` is
                    # `recompose=True`, so any collapse/solo/rail toggle
                    # rebuilds every region, not just the one that changed.
                    # A pre-built container's constructor-supplied children
                    # only mount on its FIRST mount; the same instance
                    # remounted a second time comes back childless (verified
                    # empirically — see `WatchlistsWorkbench.__init__`).
                    Region.LEFT_RAIL: self._build_tree_pane,
                    Region.FEEDS: self._build_list_pane,
                    Region.ITEMS: self._build_detail_pane,
                    Region.RIGHT_RAIL: lambda: self._build_inspector_pane(
                        latest_console_item, attach_disabled, attach_tooltip
                    ),
                },
                id="wl-workbench",
            )

    def _apply_layout(self, layout: RegionLayout) -> None:
        """Set the layout, push it to the workbench, and persist any change.

        Args:
            layout: The layout to apply. Persistence (see
                `_schedule_layout_persist`) is skipped entirely when it
                does not actually change what is on disk — e.g. `on_mount`
                re-applying the layout it just loaded, or a keypress that
                happens to leave the persisted collapsed set unchanged.
        """
        self.region_layout = layout
        try:
            # The workbench reactive is `region_layout`, NOT `layout` —
            # `Widget.layout` is an existing read-only Textual property the
            # compositor calls `.arrange()` on every render, so shadowing it
            # breaks rendering outright. Verified empirically in Task 3.
            self.query_one(WatchlistsWorkbench).region_layout = layout
        except Exception:
            logger.debug("Workbench not mounted yet; layout applies on compose.")
        self._schedule_layout_persist(layout)

    def _schedule_layout_persist(self, layout: RegionLayout) -> None:
        """Persist ``layout`` off the UI thread, skipping genuine no-ops.

        `save_setting_to_cli_config` is a synchronous whole-file
        read-modify-write plus a full config-cache reload; calling it
        directly from `_apply_layout` would block the UI event loop on
        every single `z`/`Z`/`[`/`]` keypress, including calls (like
        `on_mount`'s initial push) where nothing has actually changed. This
        repo has already paid for that class of bug once — see the
        Console's 0.2s tick doing unconditional sqlite work on the event
        loop (task-280).

        Args:
            layout: The layout whose solo-resolved collapsed set (see
                `RegionLayout.collapsed_for_persistence`) should be written
                to config if it differs from what is already persisted.
        """
        collapsed = layout.collapsed_for_persistence()
        if collapsed == self._last_persisted_collapsed:
            return
        self._last_persisted_collapsed = collapsed
        self._pending_persist_layout = layout
        self.run_worker(
            self._persist_layout_worker,
            exclusive=True,
            group="wl-layout-persist",
            thread=True,
        )

    def _persist_layout_worker(self) -> None:
        """Write the most recently requested layout to config.

        Runs off the UI thread via `run_worker(thread=True)`. Reads
        `_pending_persist_layout` fresh, under `_layout_persist_lock`, at
        the moment this worker actually executes rather than capturing it
        as an argument at schedule time: Textual's `exclusive=True` cancels
        a worker still queued in the `"wl-layout-persist"` group, but
        cannot force an already-running thread-pool call to stop before a
        newer one starts, so a rapid burst of toggles could otherwise still
        interleave writes out of order. Reading the shared "latest
        requested" value inside the lock guarantees that whichever
        invocation is the last to actually acquire it — necessarily after
        every `_schedule_layout_persist` call the burst has made so far —
        writes the true final layout, so the last write always wins.
        """
        with self._layout_persist_lock:
            layout = self._pending_persist_layout
            if layout is None:
                return
            save_region_layout(layout)

    def action_toggle_region(self) -> None:
        """Collapse or expand whichever region currently has focus."""
        self._apply_layout(self.region_layout.toggle(self.focused_region))

    def action_solo_region(self) -> None:
        """Isolate the focused centre pane; press again to restore."""
        if self.focused_region not in CENTRE_REGIONS:
            self.notify("Solo applies to the Feeds, Items, or Content panes.")
            return
        self._apply_layout(self.region_layout.solo(self.focused_region))

    def action_toggle_left_rail(self) -> None:
        self._apply_layout(self.region_layout.toggle(Region.LEFT_RAIL))

    def action_toggle_right_rail(self) -> None:
        self._apply_layout(self.region_layout.toggle(Region.RIGHT_RAIL))

    @on(RegionToggled)
    def _on_region_toggled(self, event: RegionToggled) -> None:
        event.stop()
        self._apply_layout(self.region_layout.toggle(event.region))

    def _apply_tree_scope(self, scope: TreeScope) -> None:
        """The single reconciliation point for "the tree scope is now `scope`".

        Used by both a real tree click (`_on_tree_scope_changed`) and a
        breadcrumb promotion (`handle_breadcrumb_scope_selected`) -- Task 5
        fix round 2, Finding 3 -- since promoting a breadcrumb means exactly
        the same thing a tree click at that node would.

        Clears `selected_entity` (Finding 1): the entity, if any, was
        selected from a pane row under whatever scope was previously in
        view. Leaving it in place here is exactly the reproduced bug --
        select an item under Watchlist 1, switch the tree to Watchlist 2,
        and the breadcrumb names Watchlist 2 while the actions still act on
        the Watchlist-1 item, with no indication of the mismatch. Navigating
        the tree means navigating away from that entity, full stop; there is
        no "keep both in sync" reading of `_resolve_levels` appending
        `selected_entity` as deepest that survives this.

        Also clears `selected_source`/`selected_run`/`selected_notification`
        (Task 5 fix round 3, Finding 1's remaining gap): these three are not
        independent state -- they are persisted shadows of the same
        selection `selected_entity` represents, one per pane, kept around so
        a highlighted row survives that pane's own reactive-recompose. If a
        tree click clears `selected_entity` but leaves its shadow standing,
        a later re-derive (`_load_notifications` re-deriving `selected_entity`
        from `self.selected_notification` is the one measured; a pane
        re-selecting its own surviving `selected_source`/`selected_run` on
        rebuild is the visual half of the same gap) resurrects the entity, or
        the pane's highlighted row, under a scope the tree has since moved
        away from. Clearing all three here, alongside the entity itself,
        keeps "navigating the tree means navigating away from the
        selection" true for its persisted form as well as its live one.

        Sets BOTH scopes (fix round 1, Finding 2): a tree click is the one
        event where "where the user is" and "what ancestry the Inspector may
        claim" genuinely agree. They part company again in `_select_entity`.
        """
        self._breadcrumb_labels = self._resolve_breadcrumb_labels(scope)
        self.selected_entity = None
        self.selected_source = None
        self.selected_run = None
        self.selected_notification = None
        self.selected_scope = scope
        self.tree_scope = scope

    @on(TreeScopeChanged)
    def _on_tree_scope_changed(self, event: TreeScopeChanged) -> None:
        """Store the tree's selection on the screen, not the tree.

        `selected_scope` lives here for the same reason `selected_run` and
        the create-form draft do: the workbench's `region_layout` is
        `recompose=True`, so a bare rail toggle rebuilds a brand new
        `WatchlistTree` that would otherwise lose the selection.
        """
        event.stop()
        self._apply_tree_scope(event.scope)

    @on(BreadcrumbScopeSelected)
    def handle_breadcrumb_scope_selected(self, event: BreadcrumbScopeSelected) -> None:
        """Promote a collapsed breadcrumb level (Task 5 fix round 2, Finding 3).

        `InspectorPane` posts this when a shallower breadcrumb is clicked;
        until this handler existed, the click -- the literal interaction the
        spec describes -- did nothing. Delegates to the same reconciliation
        a real tree click uses, since promoting a breadcrumb IS navigating
        the tree to that node.
        """
        event.stop()
        self._apply_tree_scope(event.scope)

    def watch_selected_scope(self) -> None:
        """Push scope + resolved labels into the live Inspector.

        Mirrors `watch_selected_entity` immediately below it: this only
        covers the "selection changed without a workbench rebuild" case --
        `_build_inspector_pane` covers the rebuild case by seeding a
        freshly-constructed `InspectorPane` from this same screen state.

        Does NOT refresh FEEDS; `watch_tree_scope` owns that (fix round 1,
        Finding 2). `selected_scope` also moves when a pane row is selected,
        which is not navigation and must leave the Feeds region alone.
        """
        if not self.is_mounted:
            return
        try:
            inspector = self.query_one("#watchlists-entity-inspector", InspectorPane)
            inspector.scope = self.selected_scope
            inspector.breadcrumb_labels = self._breadcrumb_labels
        except Exception:
            pass

    def watch_tree_scope(self) -> None:
        """Rebuild FEEDS in place so it follows the tree selection (Task 7).

        Deliberately does NOT do what `watch_active_section` does
        (`self.refresh(recompose=True)`): that rebuilds every region,
        including the Inspector, and a fresh `InspectorPane` instance is
        exactly what `watch_selected_scope`'s in-place push exists to avoid
        -- `test_changing_scope_clears_a_stale_entity_selection` (Task 5)
        holds a reference to the Inspector from *before* a scope change and
        asserts against it *after*, which a full recompose would silently
        break by handing that reference a defunct, unmounted widget.
        `WatchlistsWorkbench.refresh_region_content` instead rebuilds only
        FEEDS's own supplied content -- the one region whose display this
        task makes scope-dependent -- leaving the Tree and Inspector
        instances untouched.
        """
        if not self.is_mounted:
            return
        self._refresh_feeds_region_for_scope()

    @work(exclusive=True, group="wc_feeds_scope_refresh")
    async def _refresh_feeds_region_for_scope(self) -> None:
        """Worker wrapper so `watch_selected_scope` (a sync reactive watcher)
        can await `WatchlistsWorkbench.refresh_region_content`'s remove/mount
        pair. `exclusive=True` collapses a fast burst of tree clicks to the
        last one requested, the same reasoning `_schedule_layout_persist`
        documents for its own worker.
        """
        if not self.is_mounted:
            return
        try:
            workbench = self.query_one(WatchlistsWorkbench)
        except NoMatches:
            return
        try:
            await workbench.refresh_region_content(Region.FEEDS)
        except Exception:
            logger.opt(exception=True).debug(
                "Failed to refresh the Feeds region for the new scope."
            )

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Keep `focused_region` in step with whatever actually holds focus.

        Without this, `z` always collapses whichever region `focused_region`
        happened to default to, regardless of where the user actually is.
        Both id prefixes are checked so that focusing a *collapsed* region's
        header targets that region rather than expanding some other one.
        """
        node = event.widget
        while node is not None:
            node_id = getattr(node, "id", None) or ""
            for prefix in ("wl-region-", "wl-header-"):
                if node_id.startswith(prefix):
                    try:
                        self.focused_region = Region(node_id[len(prefix):])
                    except ValueError:
                        pass
                    return
            node = node.parent

    def watch_active_section(self) -> None:
        if self.active_section == "overview":
            self.selected_entity = None
        if self.active_section != WATCHLISTS_SECTION_RUNS:
            self._pending_navigation_run_id = None
            self._pending_navigation_run_backend = None
        if self.is_mounted:
            self.refresh(recompose=True)
            if not self._applying_navigation_context:
                self._load_active_section_data()

        if self._pending_open_create_form:
            self._pending_open_create_form = False
            self.set_timer(0.05, self._open_sources_create_form)
        if self._pending_open_import_opml:
            self._pending_open_import_opml = False
            self.set_timer(0.05, self._open_sources_import_opml)

    def _load_active_section_data(self) -> None:
        """Start the loader owned by the currently visible section."""
        if self.active_section == "items":
            self.run_worker(self._load_items(), exclusive=True)
        elif self.active_section == "rules":
            self.run_worker(self._load_rules(), exclusive=True)
        elif self.active_section == "runs":
            self.run_worker(self._load_runs(), exclusive=True)
        elif self.active_section == "sources":
            self.run_worker(self._load_sources(), exclusive=True)
        elif self.active_section == "notifications":
            self.run_worker(self._load_notifications(), exclusive=True)

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
        if (
            self._pending_navigation_run_backend is not None
            and self._pending_navigation_run_backend != self.runtime_backend
        ):
            self._pending_navigation_run_id = None
            self._pending_navigation_run_backend = None
        if not self.is_mounted:
            return
        try:
            label = self.query_one("#watchlists-backend-label", Static)
            label.update(
                "Inbox: local"
                if self.active_section == "notifications"
                else f"Backend: {self.runtime_backend}"
            )
        except Exception:
            pass
        self.selected_source = None
        self.selected_run = None
        self.selected_notification = None
        self.selected_entity = None
        self._loaded_runs = []
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

    def _select_entity(self, entity: dict[str, Any] | None) -> None:
        """The single reconciliation point for "the deepest selection is now
        `entity`" (Task 5 fix round 2, Finding 1) -- the other half of
        `_apply_tree_scope`.

        Resets `selected_scope` back to the tree's root, rather than leaving
        whatever a PRIOR tree click set in place: Sources/Runs/Items/Rules
        list rows independent of the tree's scope in this slice (Task 7
        gives Feeds/Items real scoping; these tabs still don't carry
        watchlist/source ancestry), so "all" is the only ancestry actually
        known here. Asserting a specific watchlist/source the entity may not
        even belong to would be the identical lie in the other direction --
        exactly what let a Watchlist-2 breadcrumb sit above Watchlist-1's
        item actions before this fix.

        Deliberately leaves `tree_scope` ALONE (fix round 1, Finding 2).
        Inspecting a row is not navigation: the tree has not moved, so the
        Feeds region must keep showing the watchlist the user opened. Before
        the two scopes were split, this reset silently rebuilt Feeds back to
        "All sources" -- an interaction in one region discarding the user's
        navigation in another, with no tree selection highlight to fall back
        on. Clearing `_breadcrumb_labels` alone would not have been a
        substitute: `InspectorPane._scope_levels` derives an ancestor level
        from `scope` alone and falls back to a `Watchlist {id}` label, so an
        anonymous crumb would still have rendered above the entity.

        Only reconciles when selecting a real entity; clearing back to
        `None` (deletion completing, section switching to Overview, etc.)
        leaves whatever scope is already in view alone; there is nothing
        to reconcile against.
        """
        self.selected_entity = entity
        if entity is not None:
            self._breadcrumb_labels = []
            self.selected_scope = TreeScope(kind="all")

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
        self._console_handoff.attach_to_console(self)

    @on(Button.Pressed, "#watchlists-follow-in-console")
    def follow_latest_watchlist_run_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        self._console_handoff.follow_in_console()

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
        self._select_entity(event.source)

    @on(CreateFormDraftChanged)
    def handle_source_create_draft_changed(self, event: CreateFormDraftChanged) -> None:
        event.stop()
        self._source_create_draft = {
            "name": event.name,
            "url": event.url,
            "tags": event.tags,
        }

    @on(CreateFormVisibilityChanged)
    def handle_source_create_visibility_changed(
        self, event: CreateFormVisibilityChanged
    ) -> None:
        event.stop()
        self._source_create_form_open = event.is_open

    @on(RunSelected)
    def handle_run_selected(self, event: RunSelected) -> None:
        event.stop()
        self.selected_run = event.run
        self._select_entity(event.run)

    @on(CreateSourceRequested)
    def handle_create_source_requested(self, event: CreateSourceRequested) -> None:
        event.stop()
        # Clear synchronously here, in the same handler, rather than relying
        # solely on the pane's own CreateFormVisibilityChanged/
        # CreateFormDraftChanged messages: `_create_source` below can finish
        # its own snapshot refresh and trigger a full-screen recompose fast
        # enough to win the race against those two separately-posted
        # messages still being processed, which would seed the freshly
        # rebuilt SourcesPane with the stale (pre-submit) draft. Clearing
        # here guarantees the screen's mirrored state is already correct
        # before `run_worker` even starts the async chain that can recompose.
        self._source_create_form_open = False
        self._source_create_draft = {"name": "", "url": "", "tags": ""}
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
        self._console_handoff.handle_stage_in_console_requested()

    async def _load_sources(self) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            sources = await self._controller.list_sources(
                runtime_backend=self.runtime_backend,
                limit=100,
            )
            # Mirror to screen state (Finding 2, fix round 2) so a later
            # workbench rebuild — any region collapse/solo/rail toggle, not
            # just a fresh section switch — can re-seed a brand new
            # SourcesPane instead of leaving its table empty; see
            # `_build_detail_pane` and `_loaded_sources` in __init__.
            self._loaded_sources = [dict(source) for source in sources]
            if self.is_mounted:
                try:
                    sources_pane = self.query_one("#watchlists-sources-pane", SourcesPane)
                    sources_pane.sources = self._loaded_sources
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
            self._loaded_runs = [dict(run) for run in runs]
            requested_run = self._matching_requested_run(self._loaded_runs)
            had_pending_target = self._pending_navigation_run_id is not None
            self._pending_navigation_run_id = None
            self._pending_navigation_run_backend = None
            if had_pending_target:
                self.selected_run = requested_run
                # A deep-linked run is a new selection exactly like a user
                # picking a row (Task 5 fix round 2, Finding 1) -- route it
                # through the same reconciliation rather than setting
                # `selected_entity` directly.
                self._select_entity(requested_run)
            if self.is_mounted:
                try:
                    runs_pane = self.query_one("#watchlists-runs-pane", RunsPane)
                    runs_pane.runs = self._loaded_runs
                    if had_pending_target:
                        runs_pane.selected_run = requested_run
                except Exception:
                    pass
        except Exception:
            logger.opt(exception=True).debug("Failed to load watchlist runs.")
            if callable(notify):
                notify("Failed to load watchlist runs.", severity="error")

    def _matching_requested_run(
        self, runs: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        """Return the loaded record matching the one-shot run deep link."""
        requested = self._pending_navigation_run_id
        if not requested:
            return None
        marker = ":watchlist_run:"
        requested_raw = requested.rsplit(marker, 1)[1] if marker in requested else requested
        requested_backend = self._pending_navigation_run_backend

        for run in runs:
            record_backend = str(run.get("backend") or "").strip()
            if record_backend not in {"local", "server"}:
                record_backend = ""
            candidate_id = run.get("id")
            if candidate_id not in (None, ""):
                candidate_text = str(candidate_id)
                if marker in candidate_text:
                    candidate_backend, candidate_raw = candidate_text.split(marker, 1)
                    if (
                        requested_backend == self.runtime_backend
                        and candidate_backend == requested_backend
                        and candidate_raw == requested_raw
                    ):
                        return run if isinstance(run, dict) else dict(run)
                    continue
                if (
                    requested_backend == self.runtime_backend
                    and record_backend in {"", requested_backend}
                    and candidate_text == requested_raw
                ):
                    return run if isinstance(run, dict) else dict(run)

            candidate_raw = run.get("run_id")
            if (
                candidate_raw not in (None, "")
                and requested_backend == self.runtime_backend
                and record_backend in {"", requested_backend}
                and str(candidate_raw) == requested_raw
            ):
                return run if isinstance(run, dict) else dict(run)
        return None

    async def _load_notifications(self) -> None:
        """Load the client-owned local notification inbox."""
        notify = getattr(self.app_instance, "notify", None)
        try:
            rows = await self._notifications_controller.load_rows()
        except Exception:
            logger.opt(exception=True).debug("Failed to load local notifications.")
            if callable(notify):
                notify("Failed to load local notifications.", severity="error")
            return

        self._loaded_notifications = [dict(row) for row in rows]
        selected_id = (
            self.selected_notification.get("id")
            if self.selected_notification
            else None
        )
        self.selected_notification = next(
            (
                notification
                for notification in self._loaded_notifications
                if notification.get("id") == selected_id
            ),
            None,
        )
        # Route through `_select_entity` (Task 5 fix round 3) rather than
        # assigning `self.selected_entity` directly: this re-derive is a new
        # selection exactly like a pane row click whenever a notification
        # actually survives the reload, so it must reconcile `selected_scope`
        # the same way. `_select_entity(None)` is a no-op for scope by
        # design (fix round 2), so this only matters on that surviving
        # branch -- but routing it through the one reconciliation point
        # keeps the invariant true even if a mirror is repopulated by a
        # future code path that does not go through `_apply_tree_scope`.
        self._select_entity(
            {
                **self.selected_notification,
                "entity_kind": "client_notification",
            }
            if self.selected_notification
            else None
        )
        if not self.is_mounted:
            return
        try:
            pane = self.query_one("#watchlists-notifications-pane", NotificationsPane)
        except NoMatches:
            return

        try:
            pane.notifications = self._loaded_notifications
            pane.selected_notification = self.selected_notification
        except Exception:
            logger.opt(exception=True).debug(
                "Failed to update the local notifications pane."
            )

    @on(NotificationSelected)
    def handle_notification_selected(self, event: NotificationSelected) -> None:
        event.stop()
        self.selected_notification = event.notification
        # Not one of the four handlers the reviewer named, but the identical
        # bug shape: a notification is an entity like any other, and was
        # setting `selected_entity` directly, leaving a stale scope in place.
        self._select_entity(
            {**event.notification, "entity_kind": "client_notification"}
            if event.notification
            else None
        )

    @on(RefreshNotificationsRequested)
    def handle_refresh_notifications_requested(
        self, event: RefreshNotificationsRequested
    ) -> None:
        event.stop()
        self.run_worker(self._load_notifications(), exclusive=True)

    @on(MarkNotificationReadRequested)
    def handle_mark_notification_read_requested(
        self, event: MarkNotificationReadRequested
    ) -> None:
        event.stop()
        self.run_worker(
            self._mark_notification_read(event.notification_id), exclusive=True
        )

    async def _mark_notification_read(self, notification_id: int) -> None:
        updated = await self._notifications_controller.mark_read(
            notification_id, is_read=True
        )
        if updated:
            await self._load_notifications()

    @on(DismissNotificationRequested)
    def handle_dismiss_notification_requested(
        self, event: DismissNotificationRequested
    ) -> None:
        event.stop()
        self.run_worker(
            self._dismiss_notification(event.notification_id), exclusive=True
        )

    async def _dismiss_notification(self, notification_id: int) -> None:
        dismissed = await self._notifications_controller.dismiss(
            notification_id, is_dismissed=True
        )
        if dismissed:
            await self._load_notifications()

    async def _load_items(self) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            items = await self._controller.list_items(
                runtime_backend=self.runtime_backend,
                status=None,
                limit=100,
                offset=0,
            )
            # Mirror to screen state (Finding 2, fix round 2) — see the note
            # on `_loaded_sources` in `_load_sources` above; same rebuild,
            # same gap, same fix.
            self._loaded_items = [dict(item) for item in items]
            if self.is_mounted:
                try:
                    items_pane = self.query_one("#watchlists-items-pane", ItemsPane)
                    items_pane.items = self._loaded_items
                except Exception:
                    pass
        except Exception:
            logger.opt(exception=True).debug("Failed to load watchlist items.")
            if callable(notify):
                notify("Failed to load watchlist items.", severity="error")

    @on(ItemSelected)
    def handle_item_selected(self, event: ItemSelected) -> None:
        event.stop()
        self._select_entity(event.item)

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
            # Mirror to screen state (Finding 2, fix round 2) — see the note
            # on `_loaded_sources` in `_load_sources` above; same rebuild,
            # same gap, same fix.
            self._loaded_rules = [dict(rule) for rule in rules]
            if self.is_mounted:
                try:
                    rules_pane = self.query_one("#watchlists-rules-pane", RulesPane)
                    rules_pane.rules = self._loaded_rules
                except Exception:
                    pass
        except Exception:
            logger.opt(exception=True).debug("Failed to load alert rules.")
            if callable(notify):
                notify("Failed to load alert rules.", severity="error")

    @on(RuleSelected)
    def handle_rule_selected(self, event: RuleSelected) -> None:
        event.stop()
        self._select_entity(event.rule)

    @on(RuleFormVisibilityChanged)
    def handle_rule_form_visibility_changed(
        self, event: RuleFormVisibilityChanged
    ) -> None:
        event.stop()
        self._rule_form_open = event.is_open
        # Always clear on close, regardless of what `event.editing_rule`
        # carries: RulesPane's Cancel/Submit handlers clear `show_rule_form`
        # before clearing `_editing_rule_id`, so the message posted at that
        # instant can still report the rule that WAS being edited. Ignoring
        # it here (rather than trusting it) keeps a closed form from being
        # re-seeded as still-editing on the next rebuild.
        self._rule_form_editing = event.editing_rule if event.is_open else None

    @on(RefreshRulesRequested)
    def handle_refresh_rules_requested(self, event: RefreshRulesRequested) -> None:
        event.stop()
        self.run_worker(self._load_rules(), exclusive=True)

    @on(SaveRuleRequested)
    def handle_save_rule_requested(self, event: SaveRuleRequested) -> None:
        event.stop()
        # Clear synchronously here, in the same handler, rather than relying
        # solely on the pane's own RuleFormVisibilityChanged message: `_save_rule`
        # below can finish its own snapshot refresh and trigger a full-screen
        # recompose fast enough to win the race against that separately-posted
        # message still being processed, which would seed the freshly rebuilt
        # RulesPane with the just-submitted rule still open for edit (see
        # `handle_create_source_requested` above for the same fix on the
        # Sources create form). Clearing here guarantees the screen's mirrored
        # state is already correct before `run_worker` even starts the async
        # chain that can recompose.
        self._rule_form_open = False
        self._rule_form_editing = None
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
        if InspectorPane._entity_type(entity) == "notification":
            self.app_instance.notify(
                "Use Dismiss to remove a notification from the inbox.",
                severity="information",
            )
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
            "1=Overview 2=Sources 3=Items 4=Runs 5=Rules 6=Notifications | "
            "n=new d=delete c=check p=preview ?=help",
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
