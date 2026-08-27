"""Watchlists destination shell.

The route, class name, and stable widget selectors retain the historical
``watchlists_collections``/``wc`` identifiers so older tests, shortcuts, and
handoffs keep working while Collections moves under Library.
"""

from __future__ import annotations

import asyncio
import re
import threading
import webbrowser
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import events, on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static, TextArea

from ...Constants import (
    WATCHLISTS_NAV_CONTEXT_BACKEND,
    WATCHLISTS_NAV_CONTEXT_RUN_ID,
    WATCHLISTS_NAV_CONTEXT_SECTION,
    WATCHLISTS_SECTION_RUNS,
)
from ...config import get_cli_setting
from ...runtime_policy.types import PolicyDeniedError
from ...Subscriptions.briefing_audio import (
    AudioGenerationError,
    active_audio_claim_row_ids,
    fail_interrupted_audio,
    generate_script_audio,
    pending_audio_claim_script_ids,
)
from ...Subscriptions.briefing_cast import (
    ScriptCastError,
    active_cast_claim_row_ids,
    fail_interrupted_scripts,
    generate_script,
    pending_cast_claim_briefing_ids,
)
from ...Subscriptions.briefing_export import (
    BriefingExportError,
    briefing_markdown_document,
    default_briefing_filename,
    export_feed_directory,
)
from ...Subscriptions.briefing_keep import KeepRefused, keep_briefing
from ...Subscriptions.briefing_selection import MODE_AUTO_FEATURED, VALID_MODES
from ...Subscriptions.briefing_service import (
    GenerationInFlightError,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_GENERATING,
    active_briefing_claim_row_ids,
    default_briefing_provider,
    extract_citation_ids,
    fail_interrupted_briefings,
    generate_briefing,
    pending_briefing_claim_watchlist_ids,
)
from ...Subscriptions.feed_server import (
    FeedDirectoryServer,
    FeedServerError,
    configured_bind_and_port,
    is_loopback_bind,
)
from ...Subscriptions.html_text import strip_control_characters
from ...Subscriptions.item_dates import effective_date
from ...Subscriptions.watchlist_item_page import WatchlistItemPage
from ...Subscriptions.watchlist_bundle_service import WatchlistBundleService
from ...Subscriptions.watchlist_normalizers import normalize_watchlist_item
from ...Third_Party.textual_fspicker import FileSave, SelectDirectory
from ...TTS.audio_player import play_audio_file
from ...Utils.input_validation import sanitize_string, validate_text_input, validate_url
from ...Utils.path_validation import validate_path_simple
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.prune_safe_select import PruneSafeSelect
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from ..Subscription_Modules.notifications_inbox_controller import (
    NotificationsInboxController,
)
from ..Watchlists_Modules.inspector_pane import (
    AssignSourceToWatchlistRequested,
    BreadcrumbScopeSelected,
    CheckNowRequested,
    DeleteRequested,
    EditRuleRequested,
    IgnoreRequested,
    IngestRequested,
    InspectorPane,
    ResumeSourceRequested,
    SaveNoiseSelectorsRequested,
    PreviewRequested,
    StageInConsoleRequested,
    ToggleBriefingQueueRequested,
    ViewSnapshotRequested,
)
from ..Watchlists_Modules.artifacts_pane import (
    ArtifactsPane,
    BriefingCadenceChanged,
    BriefingDefaultPresetChanged,
    BriefingModeChanged,
    BriefingSelected,
    CastScriptRequested,
    CitationActivated,
    ExportBriefingRequested,
    ExportFeedRequested,
    GenerateBriefingRequested,
    KeepBriefingRequested,
    KeptBriefingsRequested,
    ManagePresetsRequested,
    PlayAudioRequested,
    RefreshBriefingsRequested,
    ScriptSelected,
    ServeFeedRequested,
    StopAudioRequested,
    StopFeedServerRequested,
    SynthesizeAudioRequested,
    audio_file_path_is_safe,
    cadence_scope_phrase,
)
from ..Watchlists_Modules.briefing_preset_modal import BriefingPresetModal
from ..Watchlists_Modules.content_pane import (
    ContentPane,
    OpenInBrowserRequested,
    StarToggleRequested,
    UnreadToggleRequested,
)
from ..Watchlists_Modules.article_list import (
    ArticleListPane,
    NextItemsPageRequested,
    PreviousItemsPageRequested,
)
from ..Watchlists_Modules.items_pane import (
    ItemSelected,
    ItemsFilterChanged,
    NextUnreadRequested,
    RefreshItemsRequested,
)
from ..Watchlists_Modules.kept_briefings_modal import KeptBriefingsModal
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
    WatchlistNameDialog,
    WatchlistPickerDialog,
    WatchlistSourcePickerDialog,
)
from ..Watchlists_Modules.overview_pane import OverviewPane
from ..Watchlists_Modules.region_layout import (
    COLLAPSIBLE_REGIONS,
    MANAGEMENT_SIDE_PANE_ORDER,
    READ_SIDE_PANE_ORDER,
    Region,
    RegionLayout,
    resolve_effective_layout,
)
from ..Watchlists_Modules.region_layout_store import load_region_layout, save_region_layout
from ..Watchlists_Modules.reader_item_snapshot import (
    ReaderItemQuery,
    ReaderItemSnapshot,
)
from ..Watchlists_Modules.rules_pane import (
    RefreshRulesRequested,
    RuleFormVisibilityChanged,
    RuleSelected,
    RulesPane,
    SaveRuleRequested,
)
from ..Watchlists_Modules.runs_pane import (
    CancelRunRequested,
    RerunRunRequested,
    RunProgressTick,
    RunSelected,
    RunsPane,
)
from ..Watchlists_Modules.snapshot_view_modal import SnapshotViewModal
from ..Watchlists_Modules.sources_pane import (
    CreateFormDraftChanged,
    CreateFormVisibilityChanged,
    CreateSourceRequested,
    DEFAULT_SOURCE_FREQUENCY_SECONDS,
    ExportOpmlRequested,
    ImportOpmlRequested,
    SourceSelected,
    SourcesPane,
)
from ..Watchlists_Modules.watchlist_tree import (
    ALL_SOURCES_BUCKET,
    STARRED_BUCKET,
    TODAY_BUCKET,
    AggregateRootKind,
    AddSourceToWatchlistRequested,
    CreateWatchlistRequested,
    DeleteWatchlistRequested,
    RemoveSourceFromWatchlistRequested,
    RenameWatchlistRequested,
    TreeExpansionChanged,
    TreeScope,
    TreeScopeChanged,
    TreeTagFilterChanged,
    WatchlistTree,
)
from ..Watchlists_Modules.watchlists_backend_controller import WatchlistsBackendController
from ..Watchlists_Modules.watchlists_console_handoff import WatchlistsConsoleHandoff
from ..Watchlists_Modules.watchlists_tab_strip import SectionSelected, WatchlistsTabStrip
from ..Watchlists_Modules.watchlists_workbench import (
    RegionLayoutApplied,
    RegionLayoutApplyFailed,
    RegionToggled,
    WatchlistsWorkbench,
)
from .destination_recovery import DestinationRecoveryState, policy_denied_recovery_state


LayoutRecomputeCause = Literal["initial", "resize", "explicit", "article_focus"]

logger = logger.bind(module="WatchlistsCollectionsScreen")
WC_LOCAL_PAGE_SIZE = 5
WC_SERVICE_ERROR_COPY = "Watchlists services unavailable; retry Watchlists later."
WC_SERVICE_UNAVAILABLE_COPY = "Watchlists services are unavailable in this runtime."
WC_SNAPSHOT_TIMEOUT_SECONDS = 1.5

#: Success copy for the Inspector's ignore-rule Save (TASK-1362).
#:
#: The third sentence is the whole-branch review's Critical 1. Spec §3 accepts
#: that a settings edit costs one diff window -- a change the page makes before
#: the next check is compared against nothing and is never reported -- and it
#: accepts that cost only if the user is told. The Runs pane now says it after
#: the fact ("N re-baselined (settings changed)"); this says it at the moment
#: the user causes it, which is the only point at which they can decide to wait
#: for a check before saving.
NOISE_SELECTORS_SAVED_TOAST = (
    "Ignore rules saved. The next check re-baselines this source. "
    "A change the page makes before that check will not be reported."
)

#: Per-item worker-group prefix for the item-status write drainer (TASK-1541,
#: Qodo redesign). Ingest, Ignore, the unread toggle, and mark-read-on-open
#: ALL funnel through `_dispatch_item_status`/`_drain_item_status` now, one
#: drainer per item id -- see those methods' docstrings.
#:
#: This replaces two separate `exclusive=True` "supersede" worker groups an
#: earlier version of this fix used (one shared cross-item group for the
#: read/unread pair, one per-item group for Ingest/Ignore): a whole-branch
#: re-review found that "supersede by cancellation" model unsound for a
#: durable write, two independent ways, once the write got a genuine
#: `asyncio.to_thread` suspension point (see `_update_item_status_off_loop`):
#:
#: 1. `asyncio.to_thread`'s underlying OS thread is not itself cancellable
#:    once it has started running -- cancelling the AWAITING coroutine does
#:    not stop the write. So a superseded write's thread and its
#:    replacement's thread became two independent, un-ordered writes to the
#:    SAME row, either able to commit last: rapid Ingest-then-Ignore on one
#:    item could leave the DATABASE on the FIRST action while the UI (and any
#:    cache patch) showed the second.
#: 2. The opposite failure mode: `asyncio.to_thread` CAN be cancelled before
#:    the executor picks the work item up at all (reachable under a
#:    saturated default executor, no exotic timing needed) -- in which case
#:    the write never runs. The old `except asyncio.CancelledError` handler
#:    assumed "cancelled implies the write is durable" and patched the cache
#:    to the target status regardless, which is simply false in this case:
#:    the cache would then claim a status the database never reached.
#:
#: Desired-status coalescing (`_ItemStatusIntent`) replaces cancellation
#: entirely rather than patching either hole: each item gets at most one
#: QUEUED write (the latest dispatch overwrites the dict entry) plus at most
#: one IN-FLIGHT write, and the drainer always `await`s a write to genuine
#: completion -- success or a real exception -- before popping the next
#: entry for that SAME item, so two writes for one item can never race each
#: other, and nothing is ever cancelled mid-write. Two DIFFERENT items'
#: drainers still never interact at all (own group each, `exclusive=False`),
#: which is the one part of the old design worth keeping: a fast `j`/`k` run
#: or a rapid Ingest/Ignore burst still costs at most one queued write per
#: item, not one write per keystroke.
_ITEM_STATUS_DRAIN_GROUP_PREFIX = "wl-item-status-drain:"

#: Item statuses the reader's "Mark unread" button must never overwrite: they
#: are not read/unread states at all, and `new` would destroy the record.
#: A frozenset, since `_blocking_status_for` now asks the backend for the
#: item's one status and only has to decide whether it is in this set.
_NON_READ_STATE_STATUSES: frozenset[str] = frozenset({"ingested", "ignored", "error"})

#: Statuses the reader's "All" filter actually queries (TASK-3072). "All" in
#: a reader means "everything I might still want to read", not "every row in
#: the table": `ignored` items were explicitly triaged away and `error` rows
#: are a Runs-tab concern, so neither belongs in the article list.
_READER_ALL_STATUSES: tuple[str, ...] = ("new", "reviewed", "ingested")
_ITEMS_PAGE_SIZE = 50
_UNREAD_CONTEXT_FILTER_REASON = "All Unread always shows unread items."
_INDIVIDUAL_FEED_SELECTION_DISABLED = (
    "Individual feed selection is available in Read or the Local backend."
)


def _normalize_items_status_filter(value: Any) -> str:
    """Map any stored items-filter value onto the reader's Unread/All pair.

    TASK-3072. `ArticleListPane`'s Select only offers `unread`/`all`, but the
    screen mirrors the filter across workbench rebuilds and the pre-reader
    `ItemsPane` used per-status values (`new`, `reviewed`, `ignored`, ...).
    Seeding one of those into the two-option Select would raise, so every
    read of the mirrored value goes through here: legacy `new` is exactly
    the reader's `unread`; everything else falls back to `all`.
    """
    text = str(value or "").strip().lower()
    return "unread" if text in {"unread", "new"} else "all"


def _opml_import_summary_text(result: Mapping[str, Any]) -> str:
    """The post-import toast, saying the WHOLE of what happened (TASK-3604).

    The pre-round-trip toast counted only created sources, so importing a
    structured document read identically whether it filed twelve feeds into
    three watchlists or did nothing at all on a re-import. The summary now
    names new vs already-present sources, the watchlists created or
    reused, and the Unassigned remainder.

    Args:
        result: The scope service's import summary dict.

    Returns:
        One sentence for the toast.
    """
    created = int(result.get("created", 0) or 0)
    existing = int(result.get("existing", 0) or 0)
    created_wl = list(result.get("watchlists_created") or [])
    reused_wl = list(result.get("watchlists_reused") or [])
    assignments = int(result.get("assignments", 0) or 0)
    explicit_unassigned = result.get("unassigned")
    unassigned = (
        max(created + existing - assignments, 0)
        if explicit_unassigned is None
        else int(explicit_unassigned or 0)
    )

    sources_bit = f"{created} new" if existing == 0 else f"{created} new + {existing} already present"
    text = f"Imported {sources_bit} source(s) from OPML"
    if assignments:
        total_wl = len(created_wl) + len(reused_wl)
        wl_word = "watchlist" if total_wl == 1 else "watchlists"
        new_bit = f", {len(created_wl)} new" if created_wl else ""
        text += f": {assignments} into {total_wl} {wl_word}{new_bit}"
        if unassigned:
            text += f", {unassigned} unassigned"
    text += "."
    return text


@dataclass(frozen=True)
class _ItemStatusIntent:
    """One desired item-status write, captured at the moment it is dispatched.

    TASK-1541 (Qodo redesign). `_dispatch_item_status` stores exactly one of
    these per item id in `WatchlistsCollectionsScreen._item_status_desired`
    -- a second dispatch for the same item before the first has been popped
    OVERWRITES this dict entry rather than queuing a second one, which is
    the whole of the "coalescing" scheme: at most one write is ever queued
    per item, and the drainer (`_drain_item_status`) always acts on
    whichever intent is current when it next looks.

    Fields mirror exactly what `_update_item_status`'s four callers used to
    pass directly (Ingest, Ignore, the unread toggle, mark-read-on-open) --
    nothing new was invented, the dispatch-time context just now has to
    travel through a dict entry instead of a function call.

    Attributes:
        status: The target `subscription_items.status` value.
        notify_toast: Whether a successful (or refused/failed) write should
            surface a toast. `False` only for the silent mark-read-on-open
            path, matching `_update_item_status`'s previous default.
        refresh: Whether a successful write should reload `ItemsPane.items`
            and refresh the overview counts via `_refresh_overview_data()`.
            `False` only for mark-read-on-open, which patches `patch_item`
            in place instead -- see `_mark_item_read_on_open`'s docstring
            for why a recompose on every item SELECTION was a CRITICAL
            regression. (TASK-2200 removed the screen-level recompose that
            made it one; `refresh=False` is kept because reloading every
            item and re-querying the overview on every arrow key is still
            work nobody asked for.)
        patch_item: The live dict object (already held by `ItemsPane.items`/
            `_selected_content_item`/`ContentPane.item`) to mutate in place
            on a successful write, or `None` when the caller relies on
            `refresh` instead. Passed by exactly two callers in the whole
            app, `_mark_item_read_on_open` and `action_toggle_read_selected`
            -- see `handle_unread_toggle_requested`'s docstring for why
            that matters (guards must not read the dispatched item dict,
            since the non-patching callers leave it stale).
        gate: Whether the drainer must re-ask the backend
            (`_blocking_status_for`) immediately before writing and refuse
            if the item already holds a terminal status
            (`_NON_READ_STATE_STATUSES`). `True` for the unread toggle and
            mark-read-on-open, which can otherwise clobber an Ingest/Ignore
            that neither of those two dispatch paths would ever see coming
            (Ingest/Ignore pass no `patch_item=`, so nothing patches the
            cache those two read from). `False` for Ingest/Ignore
            themselves -- there is nothing to protect an ingest/ignore FROM.
    """

    status: str
    notify_toast: bool = True
    refresh: bool = True
    patch_item: dict[str, Any] | None = None
    gate: bool = False


@dataclass(frozen=True)
class ResponsivePriorityLease:
    """A manually prioritized pane and the mode where it originated."""

    target: Region
    read_mode: bool


@dataclass(frozen=True)
class ManualLayoutRollback:
    """One manual preference intent owned by its latest request token."""

    token: int
    attempted_layout: RegionLayout
    attempted_preferred: RegionLayout
    preferred_before: RegionLayout
    effective_before: RegionLayout
    responsive_before: RegionLayout | None
    article_focus_before: bool
    priority_lease_before: ResponsivePriorityLease | None


@dataclass(frozen=True)
class SectionViewIntent:
    """A section reconciliation snapshot, independent of later tab clicks."""

    token: int
    section: str
    read_mode: bool
    layout: RegionLayout
    items_factory: Callable[[], Widget]
    header_factory: Callable[[], Widget]


@dataclass(frozen=True)
class TreeDataSnapshot:
    """One complete, generation-checked left-rail data publication."""

    watchlists: tuple[dict[str, Any], ...]
    all_source_rows: tuple[dict[str, Any], ...]
    unassigned_source_rows: tuple[dict[str, Any], ...]
    counts: dict[int, dict[str, int]]
    source_counts: dict[int, dict[str, int]]
    failures: frozenset[str] = frozenset()


def watchlist_delete_consequence(source_count: int) -> str:
    """Explain what happens to a watchlist's sources when it is deleted.

    Split out so the wording is testable without driving a modal. The noun was
    already pluralised; the verb and pronoun were not, so a single-source
    watchlist read "Its 1 source are not deleted. They stay in..." (TASK-1091).
    """
    if source_count == 1:
        return (
            "Its 1 source is not deleted. It stays in Watchlists and appears "
            "under Unassigned unless it also belongs to another watchlist."
        )
    return (
        f"Its {source_count} sources are not deleted. They stay in Watchlists "
        "and appear under Unassigned unless they also belong to another "
        "watchlist."
    )

# task-895. Watchlist bundles and their membership are a LOCAL concept: the
# server API has no wire path for them at all -- `SourceUpdateRequest`
# carries no `group_ids`, neither group request carries members, and all of
# them are `extra="forbid"`, so a request naming one would be rejected
# rather than silently ignored. So the five write verbs are disabled, not
# hidden, and they say why.
#
# Built through `DestinationRecoveryState` rather than as a bare string so
# this blocker is described in the same taxonomy every other unavailable
# action on this screen uses (`policy_denied_recovery_state` supplies
# `#wc-service-error`'s copy and `#wc-attach-to-console`'s tooltip the same
# way). `disabled_tooltip` is what the tree renders -- as both the tooltip
# AND the visible note, so the hover copy and the on-screen copy cannot
# drift apart; `visible_copy`'s full six-line form does not fit a 26-column
# rail.
WC_SERVER_WRITE_RECOVERY = DestinationRecoveryState(
    status_label="Server backend",
    unavailable_what=(
        "Creating, renaming and deleting watchlists, and editing their membership"
    ),
    why=(
        "The server Watchlists API carries no watchlist membership fields, so "
        "there is no wire path for these edits"
    ),
    next_action="Switch the backend to Local to organise watchlists",
    recovery_action="Backend selector",
    authority_owner="server Watchlists API",
    stable_selector="wl-tree-actions-unavailable",
    disabled_tooltip=(
        "Server backend: the server Watchlists API carries no watchlist "
        "membership fields, so there is no wire path for these edits. "
        "Switch the backend to Local to organise watchlists."
    ),
)


class WatchlistsCollectionsScreen(BaseAppScreen):
    """Monitored sources, runs, alerts, and recovery."""

    BINDINGS = [
        ("1", "switch_section('items')", "Read"),
        ("2", "switch_section('sources')", "Sources"),
        ("3", "switch_section('runs')", "Runs"),
        ("4", "switch_section('rules')", "Rules"),
        ("5", "switch_section('notifications')", "Notifications"),
        ("6", "switch_section('artifacts')", "Artifacts"),
        ("7", "switch_section('overview')", "Overview"),
        ("question", "show_help", "Help"),
        ("n", "new_source", "New source"),
        # Round 2, O3: the label names BOTH verbs because the key performs
        # both. On a source/run/rule it deletes, after a confirmation dialog;
        # on an ITEM it ignores, unconfirmed, exactly as the Inspector's own
        # Ignore button does (review wave, Minor 2 -- it used to say "Delete"
        # in a dialog and then write `ignored`, which was the lie). A Textual
        # binding description is static, so it states the pair rather than
        # promising whichever verb the current selection is not.
        ("d", "delete_selected", "Delete / Ignore"),
        ("c", "check_now_selected", "Check now"),
        ("p", "preview_selected", "Preview"),
        ("j", "next_item", "Next item"),
        ("k", "previous_item", "Previous item"),
        # task-2513 Task 10, the reading-loop verbs. `space` (next unread) is
        # deliberately NOT here — it is bound on ItemsPane so it cannot fire
        # from the rail (see `NextUnreadRequested`).
        ("m", "toggle_read_selected", "Read/Unread"),
        # TASK-3072 plan task 7: NNW's one-key star. Same resolution and
        # gating as `m` (`_reader_verb_blocked`, the open item), one shared
        # handler with the reader's Star button.
        ("s", "toggle_star_selected", "Star"),
        # TASK-3072 plan task 8: open the item in the system browser,
        # http/https only (the URL is a remote-derived string reaching an OS
        # primitive, so the scheme check lives in the handler, not here).
        ("o", "open_in_browser", "Open in browser"),
        # TASK-3791 plan task 3: jump to the search box; typing then drives
        # the corpus-wide FTS path through the debounced reload below.
        ("/", "focus_items_search", "Search"),
        # TASK-3791 plan task 5: refresh every active source, one aggregated
        # toast + the new-items pill at the end -- never N toasts.
        ("r", "refresh_all", "Refresh all"),
        ("a", "mark_all_read", "Mark all read"),
        ("u", "undo_mark_all_read", "Undo mark-all-read"),
        ("z", "toggle_region", "Toggle focused side pane"),
        ("Z", "article_focus", "Article Focus (Read only)"),
        ("left_square_bracket", "toggle_left_rail", "Navigation"),
        ("right_square_bracket", "toggle_right_rail", "Inspector"),
    ]

    active_section = reactive("items")
    runtime_backend = reactive("local")
    selected_source = reactive(None)
    selected_run = reactive(None)
    selected_notification = reactive(None)
    selected_entity = reactive(None)
    recovery_state = reactive(None)
    # NOT `recompose=True` (TASK-2200). Its only writer is
    # `_refresh_overview_data`, a background worker fired by mount, every
    # backend switch and every write verb on this screen -- so a screen-level
    # recompose here was a third background destroyer alongside
    # `_load_tree_data`/`_apply_local_wc_snapshot`, and a documented one: it
    # detached the mounted `ItemsPane`, reset the `DataTable` cursor and
    # dropped focus on any item-status write whose counts actually changed
    # (see `_update_item_status`'s `refresh=False` path, which exists solely
    # to dodge it). `watch_overview_data` pushes the payload into the three
    # live surfaces that read it instead.
    overview_data = reactive(dict)
    # Through Phase C, CONTENT held only a placeholder stub and started
    # collapsed to avoid spending screen space on it. Phase D wires a real
    # reader (`ContentPane`) into CONTENT, so it now starts expanded like
    # every other region.
    #
    # This class-level value is a Textual-required placeholder, not what a
    # real screen actually starts with (TASK-15775): `__init__` immediately
    # overwrites it via `set_reactive` with whatever
    # `region_layout_store.load_region_layout()` returns — the persisted
    # layout, or `_FIRST_RUN_DEFAULT` (RIGHT_RAIL collapsed) on a genuinely
    # fresh install, including that function's one-time migration that drops
    # any CONTENT collapse a user saved before Phase D, since that could
    # only be a leftover of the old stub-era default, never a deliberate
    # choice about the real reader. Seeding happens before `compose_content`
    # ever runs, so the workbench is built with the SAME layout `on_mount`
    # used to apply a moment later — see `__init__`'s own comment for why
    # this class-level default used to disagree with that call's result,
    # and the ordering guarantee against `_last_persisted_collapsed`.
    #
    # task-16843: this is a shared *instance* default (`reactive(RegionLayout())`
    # installs the SAME `RegionLayout` object on every screen instance until
    # `set_reactive` overwrites it above) -- but it is harmless: `RegionLayout`
    # is `frozen=True` and every field is itself immutable (`frozenset`,
    # `Region | None`), so there is no mutable container underneath to mutate
    # in place. Allowlisted in
    # `Tests/Architecture/test_reactive_mutable_default_inventory.py`'s
    # `IMMUTABLE_INSTANCE_ALLOWLIST` rather than rewritten into a factory.
    region_layout = reactive(RegionLayout())
    focused_region = reactive(Region.ITEMS)
    # Two scopes, deliberately: they answer different questions and they
    # diverge (fix round 1, Finding 2).
    #
    # `tree_scope` is where the user has NAVIGATED -- the tree node in view.
    # It drives the scoped readouts (`scoped_source_rows`: the centre
    # header's summary line and the Sources table), and only a tree click
    # or a breadcrumb promotion moves it.
    #
    # `selected_scope` is the ancestry the Inspector is entitled to CLAIM
    # for whatever is currently selected. It follows `tree_scope` on a tree
    # move, but resets to "all" when a pane row is selected, because a
    # Sources/Runs/Items/Rules row carries no watchlist/source ancestry --
    # asserting one would put a breadcrumb over an entity that may not
    # belong to it (Task 5 fix round 2, Finding 1).
    #
    # Task 7 made `selected_scope` drive the scoped readout as well, which
    # silently merged the two: clicking a pane row to inspect it then reset
    # the scope back to "All sources", discarding tree navigation the user
    # had done in another region. Splitting them keeps both properties --
    # the scoped readout follows the tree, and the Inspector still claims
    # no ancestry it does not know. Clearing `_breadcrumb_labels` alone
    # would NOT have been enough: `InspectorPane._scope_levels` derives an
    # ancestor level from `scope` alone and falls back to a `Watchlist {id}`
    # label, so the crumb would still render, just anonymously.
    #
    # Both live on the screen, not on the tree widget, precisely because
    # collapsing or expanding the left rail constructs a brand new
    # `WatchlistTree` instance -- the region's rendered form changes, so its
    # widget is swapped for a freshly built one (task-15461 narrowed the
    # blast radius of a layout change from "every region" to "the region that
    # changed", but a toggled region is still rebuilt). Pane-local state does
    # not survive that (see `selected_run` and the create-form draft above
    # for the same reasoning already applied elsewhere on this screen).
    #
    # task-16843: both are shared *instance* defaults, same shape as
    # `region_layout` above (each reactive's own `TreeScope("all")` object is
    # shared across every screen instance that has not reassigned it -- the
    # two reactives get their own separate default instances, not each
    # other's) -- and equally harmless: `TreeScope` is `frozen=True` with
    # only immutable field types (`Literal` str, `int | None`). Allowlisted
    # in `Tests/Architecture/test_reactive_mutable_default_inventory.py`'s
    # `IMMUTABLE_INSTANCE_ALLOWLIST`.
    selected_scope = reactive(TreeScope(kind="all"))
    tree_scope = reactive(TreeScope(kind="all"))

    _SECTION_DETAIL_TITLE = {
        "overview": "Overview",
        "sources": "Sources",
        "items": "Read",
        "runs": "Runs",
        "rules": "Rules",
        "notifications": "Notifications",
        "artifacts": "Artifacts",
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
        # Whether focus currently sits in the centre header/tab strip
        # (`#wl-centre-status`), outside every region/grip wrapper. See
        # `on_descendant_focus` and `action_toggle_region`.
        self._focus_in_centre_header = False
        self._pending_open_create_form = False
        self._pending_open_import_opml = False
        self._pending_delete_entity: dict[str, Any] | None = None
        self._pending_navigation_run_id: str | None = None
        self._pending_navigation_run_backend: str | None = None
        self._loaded_runs: list[dict[str, Any]] = []
        # TASK-2306: the selected run's Items and Logs, mirrored here for the
        # same reason `_loaded_runs` is -- `_build_detail_pane` constructs a
        # brand new `RunsPane` on every workbench rebuild, and a pane seeded
        # with a `selected_run` but no detail renders the exact blank the
        # user was told is a bug.
        self._run_detail_items: list[dict[str, Any]] = []
        self._run_detail_logs: str = ""
        self._run_detail_items_note: str = ""
        self._loaded_notifications: list[dict[str, Any]] = []
        # Mirrors what's currently loaded for Sources/Items/Rules the same way
        # `_loaded_runs`/`_loaded_notifications` already do (Finding 2, fix
        # round 2): `_build_detail_pane` constructs a brand new
        # SourcesPane/ItemsPane/RulesPane on every workbench rebuild (any
        # region collapse/expand or tab switch), and
        # a fresh pane's `sources`/`items`/`rules` reactive starts at its
        # class default (`[]`). Without holding the last-loaded rows here and
        # re-seeding them below, the table would render empty until the next
        # unrelated navigation happened to trigger a reload.
        self._loaded_sources: list[dict[str, Any]] = []
        self._loaded_items: list[dict[str, Any]] = []
        self._items_snapshot: ReaderItemSnapshot | None = None
        self._items_page_index = 0
        self._items_has_next = False
        self._items_page_loading = False
        self._items_snapshot_count = 0
        self._items_pending_arrivals = 0
        self._items_arrival_generation = 0
        self._items_pending_query_key: tuple[Any, ...] | None = None
        self._pending_tree_scope: TreeScope | None = None
        self._items_retry_message: str | None = None
        self._items_retry_inflight = False
        self._items_snapshot_generation = 0
        self._items_page_presentation_lock = asyncio.Lock()
        self._items_inflight_replacement: tuple[
            tuple[Any, ...], asyncio.Future[bool]
        ] | None = None
        self._items_inflight_continuation: asyncio.Future[bool] | None = None
        self._selected_content_page_key: tuple[Any, ...] | None = None
        self._items_search_results_authoritative = False
        # The undo batch for `action_mark_all_read` (task-2513 Task 10): the
        # raw DB ids the last catch-up touched, cleared on undo. Raw ids —
        # `mark_all_read` returns database ids, which the loaded item dicts
        # carry as `item_id` (their `id` is the normalized table row key).
        self._last_mark_all_read_batch: list[int] = []
        # The single debounce timer behind `_request_tree_counts_refresh`
        # (review wave, Minor 6). Declared here so the attribute always
        # exists, rather than springing into being on the first item opened.
        self._tree_counts_refresh_timer: Any = None
        self._loaded_rules: list[dict[str, Any]] = []
        # Artifacts (spec #2 phase 1, task 4): the same rebuild-survival
        # mirror as the four lists above, plus the selection the pane's
        # detail area renders.
        self._loaded_briefings: list[dict[str, Any]] = []
        self._selected_briefing: dict[str, Any] | None = None
        # Task 4: the current watchlist's stored briefing selection mode and
        # default preset id, mirrored here for the same rebuild-survival
        # reason as `_loaded_briefings` above -- `_build_detail_pane` seeds
        # a freshly built `ArtifactsPane` from this state on every region
        # rebuild. Defaults match `_selection_mode`'s own NULL/no-scope
        # fallback (`briefing_service.py`) so a pane that has not yet heard
        # from `_load_briefings` shows the same mode generation would
        # actually use.
        self._briefing_selection_mode: str = MODE_AUTO_FEATURED
        self._briefing_default_preset_id: int | None = None
        # Spec #2 phase 4, Task 4: the current watchlist's stored
        # `briefing_cadence_seconds`, mirrored here for the identical
        # rebuild-survival reason as the two fields above -- `_build_
        # detail_pane` seeds a freshly built `ArtifactsPane` from this on
        # every region rebuild. `None` (never scheduled) matches the
        # column's own default and `ArtifactsPane.briefing_cadence_
        # seconds`'s own fallback.
        self._briefing_cadence_seconds: int | None = None
        # True only while THIS screen's `wl-briefing` worker is running.
        # `fail_interrupted_briefings` cannot tell a crashed worker's row
        # from a live one -- both read `generating` -- so the live case is
        # answered from memory here and the sweep is only ever asked about
        # rows this session did not create. See
        # `handle_generate_briefing_requested`.
        self._briefing_in_flight = False
        # Which watchlist `_briefing_in_flight` refers to, or `None`.
        # `_briefing_in_flight` is deliberately screen-global (one
        # `wl-briefing` worker at a time, `exclusive=True`), so a refusal
        # while it is set may belong to a DIFFERENT watchlist than the one
        # on screen. Set alongside the flag so the refusal toast can name
        # the watchlist actually generating instead of falsely claiming
        # "this watchlist" (whole-branch review fix 4).
        self._briefing_in_flight_watchlist_id: int | None = None
        # Briefing presets (spec #2 phase 2a, Task 3): reloaded whenever
        # `BriefingPresetModal` dismisses `True` (see
        # `_open_briefing_preset_manager`). Task 4 wires this list into the
        # Artifacts toolbar's default-preset picker; held here now so that
        # contract -- "the modal dismisses True, the screen reloads its
        # preset list" -- is real before that picker exists to consume it.
        self._loaded_briefing_presets: list[dict[str, Any]] = []
        # Task 5: this same rebuild-survival mirroring, for the SELECTED
        # briefing's cast scripts. Scoped to one briefing (not the whole
        # watchlist) because a script belongs to exactly one briefing and
        # this pane only ever shows one briefing's detail at a time --
        # loaded and re-resolved alongside `_selected_briefing` inside
        # `_load_briefings` (see that method).
        self._loaded_scripts: list[dict[str, Any]] = []
        self._selected_script: dict[str, Any] | None = None
        # Review round 1, Minor #4: `{script_id: status}` for every one of
        # `_loaded_scripts`' ids that has at least one `briefing_audio`
        # render -- the rebuild-survival mirror of `pane.scripts_with_
        # audio`, resolved alongside `_loaded_scripts` inside `_load_
        # briefings` (see that method). Owner decision, task-7 phase 2b
        # follow-up: upgraded from a bare `frozenset[int]` of "has an
        # attempt" to carry each script's newest audio status, so the
        # scripts table can distinguish a failed render from a
        # successful one instead of painting both identically.
        self._scripts_with_audio: dict[int, str] = {}
        # True only while THIS screen's `wl-cast` worker is running -- the
        # exact sibling of `_briefing_in_flight` above, for the same reason:
        # `fail_interrupted_scripts` cannot tell a crashed worker's row from
        # a live one, so the live case is answered from memory here. See
        # `handle_cast_script_requested`.
        self._cast_in_flight = False
        # Which briefing `_cast_in_flight` refers to, or `None` -- the
        # `_briefing_in_flight_watchlist_id` sibling, so a refusal can name
        # the briefing actually being cast.
        self._cast_in_flight_briefing_id: int | None = None
        # TASK-2309. The source ids ("id" field -- the namespaced form
        # `selected_source`/`selected_entity` carry) with a "Check now"
        # currently running, from EITHER activation site (the Sources pane's
        # own button and the Inspector's copy of it both post the same
        # `CheckNowRequested`). This is the debounce state
        # `handle_check_now_requested` consults to refuse a second press
        # rather than silently starting a second run, and it is mirrored
        # onto `SourcesPane.busy_source_ids`/`InspectorPane.busy_source_ids`
        # (`_set_check_now_busy`) so both buttons show it -- including a
        # freshly-rebuilt pane (`_build_detail_pane`/`_build_inspector_pane`
        # re-seed it, the same rebuild-survival reason every other mirror in
        # this method exists).
        self._checks_in_flight: set[str] = set()
        # Task 6: the SELECTED briefing's citations -- the rebuild-survival
        # mirror of `pane.citations`, resolved alongside `_selected_briefing`
        # inside `_load_briefings` (see that method). `_citation_item_lookup`
        # is the OTHER half of that same resolution: normalized item dicts
        # (shaped exactly like `ItemsPane.items`' own entries -- see
        # `normalize_watchlist_item`) for every citation that still resolves
        # to a live row, keyed by the raw id `[item N]` names. A citation
        # NOT in this dict is the pruned signal `handle_citation_activated`
        # acts on -- there is no separate "available" flag to fall out of
        # sync with it.
        self._loaded_citations: list[dict[str, Any]] = []
        self._citation_item_lookup: dict[int, dict[str, Any]] = {}
        # Task 7: the SELECTED script's newest `briefing_audio` render, or
        # `None` when it has never been synthesized -- the rebuild-survival
        # mirror of `pane.script_audio`, resolved alongside `_selected_
        # script` inside `_load_briefings` (see that method).
        self._loaded_script_audio: dict[str, Any] | None = None
        # True only while THIS screen's `wl-audio` worker is running -- the
        # exact sibling of `_cast_in_flight` above, for the same reason:
        # `fail_interrupted_audio` cannot tell a crashed worker's row from a
        # live one, so the live case is answered from memory here. See
        # `handle_synthesize_audio_requested`.
        self._audio_in_flight = False
        # Which script `_audio_in_flight` refers to, or `None` -- the
        # `_cast_in_flight_briefing_id` sibling, so a refusal can name the
        # script actually being synthesized.
        self._audio_in_flight_script_id: int | None = None
        # Task 1 (phase 3): True from the moment Export is pressed until
        # the `FileSave` dialog it pushed actually resolves -- via a real
        # path, a cancel, or a rejected path -- not merely until the dialog
        # is MOUNTED. Review round 1 (Important #1): a live repro (two
        # rapid presses) produced `['FileSave', 'FileSave']` on the screen
        # stack -- Textual stacks a second `push_screen`, it does not
        # refuse one -- so without this flag a second press before the
        # first dialog resolves opens a phantom second dialog, and
        # completing both silently has the second write clobber the
        # first's file. Claimed in `handle_export_briefing_requested`
        # BEFORE `run_worker` (the same reason `_briefing_in_flight` is
        # claimed before its own `run_worker`: a check made after
        # scheduling leaves a window two presses can both pass). Cleared
        # in `_write_briefing_export_file`'s `finally` -- not in `_push_
        # export_briefing_dialog`, whose own `await self.app.push_screen`
        # returns once the dialog is MOUNTED, long before the user has
        # picked a path or cancelled -- so every resolution (success,
        # cancel, rejected path, write failure) re-arms Export, and a
        # failure to even OPEN the dialog clears it too (see that
        # method's own `except`).
        self._briefing_export_in_flight = False
        # Task 5 (phase 3): the identical guard, for exporting a
        # watchlist's audio as a podcast feed directory. Claimed in
        # `handle_export_feed_requested` BEFORE `run_worker`, cleared in
        # `_export_feed_directory`'s `finally` -- not in `_push_export_
        # feed_dialog`, whose own `await self.app.push_screen` returns
        # once the `SelectDirectory` dialog is MOUNTED, long before the
        # user has picked a directory or cancelled. See `handle_export_
        # feed_requested`'s own docstring for why this mirrors `_briefing_
        # export_in_flight` rather than reusing it: a briefing export and
        # a feed export are two independent actions a user could plausibly
        # run at the same time (different destinations, different files).
        self._feed_export_in_flight = False
        # task-1780, Task 5: True from the moment Keep is pressed until
        # `keep_briefing` (Task 2) returns or raises. Same claimed-before-
        # `run_worker` discipline as every other in-flight guard on this
        # screen (`_briefing_export_in_flight` immediately above is the
        # closest sibling: keeping, like exporting, is a one-shot action on
        # the selected briefing with no target-naming refusal needed --
        # this screen only ever runs one Keep at a time, full stop).
        self._keep_in_flight = False
        # Whether THIS watchlist has at least one export-ready audio
        # episode -- mirrored onto `ArtifactsPane.has_audio_episodes` by
        # `_load_briefings`. Read (never written) by `handle_export_feed_
        # requested`'s own re-check, for the identical reason `handle_
        # export_briefing_requested`'s docstring gives for re-checking the
        # selected briefing's status: the button's disabled state and the
        # message it posts are two different frames.
        self._watchlist_has_audio_episodes = False
        # task-1760: this screen's own feed server, and the directory it
        # would serve. `FeedDirectoryServer` is not a module singleton --
        # each screen instance owns exactly one (this instance's `stop()`
        # is called from `on_unmount`, so a second screen instance never
        # needs to know about the first's server at all). `_last_feed_
        # export_directory` is set by `_export_feed_directory` on every
        # SUCCESSFUL export (full or partial -- both leave a valid,
        # servable `feed.xml` on disk) and read by `handle_serve_feed_
        # requested`'s own re-check, mirroring `_watchlist_has_audio_
        # episodes`'s own "button disabled state and the message it posts
        # are two different frames" reasoning immediately above.
        self._feed_server = FeedDirectoryServer()
        self._last_feed_export_directory: Path | None = None
        # The item currently open in the CONTENT reader (Task 4). Held here
        # for the identical reason as `_loaded_items` above: `_build_content_pane`
        # is a factory the workbench calls on every region rebuild, and a
        # freshly built `ContentPane`'s `item` reactive would otherwise start
        # back at `None` on every collapse/expand, clearing the
        # reader out from under a user who hadn't touched Items at all.
        self._selected_content_item: dict[str, Any] | None = None
        self._read_recovery_active = False
        # Left-rail tree inputs. Aggregate source rows are deliberately
        # separate from `_loaded_sources`, whose management-table page may be
        # capped; expanding All Sources must always reveal the complete
        # navigation snapshot.
        self._tree_watchlists: list[dict[str, Any]] = []
        self._tree_all_source_rows: list[dict[str, Any]] = []
        self._tree_unassigned_source_rows: list[dict[str, Any]] = []
        self._tree_counts: dict[int, dict[str, int]] = {}
        # Per-source totals/unread for the tree's source badges (Task 8 of
        # the reader-first plan); loaded with the rest, rendered there.
        self._tree_source_counts: dict[int, dict[str, int]] = {}
        # Which watchlists are expanded in the rail, and the rail's tag
        # filter (whole-branch review, Finding 2). Held here, not on
        # `WatchlistTree`, for exactly the reason the create-form draft and
        # `tree_scope` already are: `_build_tree_pane` is a factory the
        # workbench calls on every full recompose, and `watch_active_section`
        # -- the screen's PRIMARY interaction -- does a full
        # `refresh(recompose=True)`. A brand new `WatchlistTree` starts both
        # of its own reactives at their class defaults, so clicking a section
        # tab used to collapse the rail and drop the tag filter, leaving the
        # node the centre is scoped to no longer in the DOM. Plain attributes
        # rather than screen reactives: nothing on the screen needs to watch
        # them, and `_breadcrumb_labels`/`_source_create_draft` already
        # establish that shape for screen-mirrored pane state.
        self._tree_expanded_root_kinds: frozenset[AggregateRootKind] = frozenset()
        self._tree_expanded_watchlist_ids: frozenset[int] = frozenset()
        self._tree_active_tag: str | None = None
        self._tree_load_generation = 0
        self._tree_snapshot = TreeDataSnapshot((), (), (), {}, {})
        self._tree_snapshot_failures: frozenset[str] = frozenset()
        self._tree_failure_episode_active = False
        # task-895: one tree write (each of which owns a modal dialog) at a
        # time -- see `_start_tree_write` for why this is a guard rather
        # than `run_worker(exclusive=True)`.
        self._tree_write_active = False
        # Breadcrumb display names for `selected_scope` (Task 5 fix round 1):
        # resolved once in `_on_tree_scope_changed`, not on every Inspector
        # render, and held here for the same reason `selected_scope` itself
        # is screen-held -- `_build_inspector_pane` re-seeds a brand new
        # `InspectorPane` on every workbench rebuild, and a fresh pane's own
        # reactive would otherwise start back at its class default.
        self._breadcrumb_labels: list[str] = []
        self._applying_navigation_context = False
        # Mirrors SourcesPane's create-form state (Finding 1, fix round 1):
        # collapsing or expanding the region the pane lives in constructs a
        # brand new SourcesPane, and so does a section switch. Without
        # holding the draft here — the same way selected_source/selected_run/
        # active_section already survive pane rebuilds — a half-typed create
        # form would be silently destroyed by a keybinding that has nothing
        # to do with Sources. (Since task-15461 a rebuild is scoped to the
        # region that actually changed, which narrows how OFTEN this fires,
        # not whether the draft has to survive it.)
        self._source_create_form_open = False
        self._source_create_draft: dict[str, str] = {"name": "", "url": "", "tags": ""}
        self._source_create_draft_active = True
        self._source_create_draft_frequency = DEFAULT_SOURCE_FREQUENCY_SECONDS
        # The create form's noise-selector text, mirrored for the same reason
        # as the three fields above (TASK-1362). Held separately, and `None`
        # rather than `""` when untouched, because its empty state is not its
        # default: `SourcesPane` prefills it with the shipped selector set, and
        # `""` is a user deliberately clearing it. Seeding `""` back over a
        # fresh pane would silently turn "watch everything" into the default,
        # and seeding the default over a cleared field would be the reverse.
        self._source_create_draft_selectors: str | None = None
        # TASK-2302: the create form's chosen type and destination, mirrored
        # for the same reason as the fields above. `None` means "the pane has
        # not reported one", which is why the destination mirror is `None`
        # here and not `SourcesPane.UNASSIGNED_DESTINATION` -- an untouched
        # form takes its destination from the live scope
        # (`_scope_default_destination`), not from a stale mirror.
        self._source_create_draft_type: str | None = None
        self._source_create_draft_destination: Any = None
        # Mirrors RulesPane's edit-form state (Finding 4, fix round 2): the
        # same rebuild-destroys-pane-local-state failure mode as the Sources
        # create form above, but for an in-progress rule EDIT rather than a
        # create. `_rule_form_editing` holds the rule being edited, or `None`
        # when the open form is for a brand new rule.
        self._rule_form_open = False
        self._rule_form_editing: dict[str, Any] | None = None
        # Mirrors ItemsPane's filter/search state, for exactly the reason
        # above (whole-branch review, Important). Any workbench rebuild
        # constructs a brand new `ItemsPane` via `_build_detail_pane`, and
        # without these the user's status filter reset to "all", their search
        # box emptied and their selection cleared -- from a `z`/`[`/`]`
        # keypress or a chevron click that had nothing to do with Items, or
        # from the `overview_data` recompose an item-status refresh
        # ("Mark unread", Ingest, Ignore) triggers whenever the overview
        # counts actually change value.
        self._items_status_filter = "all"
        self._items_search_query = ""
        # Layout-persistence bookkeeping (PR #926 review, Bug 2): avoids
        # writing to config on every `_apply_layout` call — `on_mount`'s
        # initial push-back of the just-loaded layout, and every no-op
        # toggle, would otherwise trigger `save_setting_to_cli_config`'s
        # synchronous whole-file read-modify-write on the UI thread. See
        # `_schedule_layout_persist`/`_persist_layout_worker` below.
        #
        # TASK-15775: `region_layout`'s class-level reactive default
        # (`RegionLayout()`, nothing collapsed) disagreed with what a cold
        # open actually renders (`_FIRST_RUN_DEFAULT`, RIGHT_RAIL collapsed
        # — `region_layout_store.py`). `compose_content` reads
        # `self.region_layout` to build the initial `WatchlistsWorkbench`,
        # and compose always runs before `on_mount`, so every cold open
        # used to compose the full 13-widget Inspector pane and then, the
        # instant `on_mount` fired `_apply_layout(load_region_layout())`,
        # tear it straight back down for the one-line collapsed header a
        # fresh config actually wants (task-15462's profiling: ~5-10ms,
        # 1-2% of a ~450ms push, plus a `_swap_region_widget` call that
        # regression tests now pin at zero for a normal visit).
        #
        # Seeding `region_layout` here, at construction, instead closes
        # that gap: `_create_navigation_screen` builds a screen only to
        # push/switch to it immediately after (never cached, never left
        # un-mounted — see that method's own docstring), so there is no
        # meaningful ordering difference from doing this at the top of
        # `on_mount` as before. `set_reactive` (not assignment) matches
        # `WatchlistsWorkbench.__init__`'s own seeding: it stores the value
        # without running a watcher that has nothing mounted yet to act on.
        #
        # `_last_persisted_collapsed` is primed from this SAME call's
        # result, on the very next line — closing the ordering risk
        # task-15462 flagged when it chose not to make this exact move
        # inside a profiling task: moving `load_region_layout()` earlier is
        # only safe if this priming moves in lockstep with it. A
        # construction-time load whose priming still happened later (e.g.
        # still in `on_mount`, reading a SECOND call's result) would leave
        # a window where `_schedule_layout_persist` could read
        # `_last_persisted_collapsed` as stale against an already-applied
        # layout, misfiring its `!=` no-op guard and scheduling a redundant
        # persist worker for a value already on disk — or, worse, running
        # `load_region_layout()`'s one-time migration branch a second time.
        # There is exactly one call to `load_region_layout()` per screen
        # lifecycle now; `on_mount` reuses `self.region_layout` rather than
        # calling it again (see `on_mount`).
        loaded_layout = load_region_layout()
        self.set_reactive(WatchlistsCollectionsScreen.region_layout, loaded_layout)
        self._effective_region_layout = loaded_layout
        self._responsive_region_layout: RegionLayout | None = None
        self._article_focus_active = False
        self._responsive_priority_lease: ResponsivePriorityLease | None = None
        self._layout_request_generation = 0
        self._current_layout_request_token = 0
        # Avoid initializing the reactive (and its watcher) before Textual
        # attaches this screen to an app; on_mount replaces this seed.
        self._rendered_section = "items"
        self._manual_layout_rollback: ManualLayoutRollback | None = None
        self._items_view_anchor_id: str | None = None
        self._items_view_scroll_y = 0.0
        self._items_view_had_focus = False
        self._items_view_focus_id: str | None = None
        self._items_view_context_key: tuple[Any, ...] | None = None
        self._last_persisted_collapsed: frozenset[Region] | None = (
            loaded_layout.collapsed
        )
        self._pending_persist_layout: RegionLayout | None = None
        self._pending_persist_generation: int | None = None
        self._layout_persist_generation = 0
        self._layout_persist_draining = False
        self._layout_persist_lock = threading.Lock()
        # Desired-status coalescing for the four item-status write paths
        # (Ingest, Ignore, the unread toggle, mark-read-on-open) -- TASK-1541,
        # Qodo redesign. See `_ItemStatusIntent`'s docstring for the full
        # unsoundness `_ITEM_STATUS_DRAIN_GROUP_PREFIX`'s comment names, and
        # `_dispatch_item_status`/`_drain_item_status` for the mechanism these
        # two dicts drive. Event-loop-only state, exactly like every other
        # in-flight flag on this screen (`_briefing_in_flight` et al. above):
        # read and written only from the loop thread, never from inside the
        # `asyncio.to_thread` write itself.
        #
        # `_item_status_desired` holds at most one queued write per item id --
        # a second dispatch for the same item overwrites the entry rather
        # than adding a second one, which is the whole of the coalescing
        # scheme. `_item_status_draining` names which items currently have a
        # drainer worker running, so a dispatch that lands while one is
        # already draining does not start a second one (and NEVER cancels the
        # running one) -- it just relies on the running drainer to notice the
        # new entry on its own next loop.
        self._item_status_desired: dict[Any, "_ItemStatusIntent"] = {}
        self._item_status_draining: set[Any] = set()
        # TASK-2200. Which workbench surfaces still need rebuilding in place,
        # and whether the drainer that rebuilds them is already running. See
        # `_request_surface_refresh` for why this is a record-intent/drain
        # queue rather than `run_worker(exclusive=True)` per surface.
        self._pending_surface_refresh: set[str] = set()
        self._surface_refresh_draining = False
        self._pending_section_intent: SectionViewIntent | None = None
        # The Console-follow adapter's latest answer, mirrored here so the
        # RIGHT_RAIL factory reads an attribute instead of polling from
        # `compose()` (TASK-2200 review wave, M4). Refreshed by
        # `compose_content` and by `_resolve_console_follow_drift`.
        self._console_follow_item: Any = None
        self._controller = WatchlistsBackendController(
            app_instance=app_instance,
            scope_service=getattr(app_instance, "watchlist_scope_service", None),
            server_service=getattr(app_instance, "server_watchlists_service", None),
        )
        self._notifications_controller = NotificationsInboxController(
            app_instance=app_instance,
            store=getattr(app_instance, "client_notifications_db", None),
        )

    @property
    def _dom_is_live(self) -> bool:
        """Whether this screen's widgets are in the DOM and can be patched.

        **Not `self.is_mounted`** (TASK-2200 live-verification wave), and the
        difference is a real, shipped defect rather than a nicety.
        `Widget.is_mounted` returns `_is_mounted`, which
        `MessagePump._pre_process` sets in its `finally` -- *after* it has
        dispatched `Compose` **and** `Mount`. So for the whole of `on_mount`,
        and for anything `on_mount` starts that completes before that
        `finally` runs, `is_mounted` is `False` while the entire subtree is
        already registered and queryable.

        This screen's loaders complete inside exactly that window on a cold
        local database. Instrumented on a real terminal:

        ```
        OVERVIEW watcher is_mounted=False keys=[]  pane=0 inspector=0
        ON_MOUNT         is_mounted=False wb=1 centre=1 status=1
        SNAPSHOT applied is_mounted=False loaded=True wb=1 centre=1 status=1
        OVERVIEW watcher is_mounted=False keys=[...] pane=1 inspector=1
        ```

        -- the workbench, `#wl-centre`, the status header, the Overview pane
        and the Inspector are all present, and `is_mounted` is `False` for
        every one of those lines. An `is_mounted` guard therefore dropped
        every in-place update on the floor, and nothing re-requested them:
        the screen sat on "Loading local Watchlists snapshot..." /
        "Loading watchlist activity..." / "State: unavailable" indefinitely
        until an unrelated tab switch recomposed it.

        The full-screen `refresh(recompose=True)` this task replaced did not
        have that problem, because the `overview_data` reactive calls
        `Widget.refresh` itself, with no `is_mounted` guard anywhere in the
        path. Reproducing that reach is what this property is for.

        `is_attached` asks the question that actually matters -- "is there a
        path from me up to the DOM root" (`MessagePump.is_attached`) -- which
        is `True` from `App._register` onwards and `False` once unmounted or
        once the app is exiting. Every caller still degrades per-widget via
        `except NoMatches`, so this is a cheap outer gate, not the only
        protection.
        """
        return self.is_attached

    def _watchlist_bundle_service(self) -> WatchlistBundleService | None:
        """The live watchlist bundle service, or ``None`` if unavailable.

        Mirrors how the screen reaches ``watchlist_scope_service``: via
        ``getattr(..., None)`` on the app instance, so the tree and other
        callers degrade rather than crash when the service has not been
        wired (e.g. a bare app stub in tests).
        """
        return getattr(self.app_instance, "watchlist_bundle_service", None)

    def on_mount(self) -> None:
        # No super().on_mount(): the dispatcher already invokes
        # BaseAppScreen.on_mount separately for this Mount event.
        # Push the layout into the already-mounted workbench, not just this
        # screen's own reactive: `compose_content` already ran by the time
        # `on_mount` fires (compose always precedes the Mount event), so the
        # WatchlistsWorkbench child was built with whatever `region_layout`
        # held at THAT moment. Without also reaching into the mounted
        # workbench via `_apply_layout`, a persisted collapse would silently
        # not render until some unrelated later recompose happened to pick
        # it up.
        #
        # TASK-15775: `region_layout` (and `_last_persisted_collapsed`) were
        # already seeded from `load_region_layout()` in `__init__` — reused
        # here rather than calling it a second time, both to keep the
        # one-time-migration read/write genuinely one-time per screen and to
        # keep this call a true no-op: `self.region_layout` already equals
        # what `_apply_layout` sets it to, and Textual's reactive skips
        # `watch_region_layout` on an unchanged value, so this produces zero
        # `_swap_region_widget` calls on a normal visit. Kept, rather than
        # dropped outright, so a screen mounted a second way (a test that
        # changes `region_layout` between construction and mount, or a
        # future caller) still gets the persistence reconciliation pass
        # `_apply_layout` performs for anything that genuinely did change.
        self._rendered_section = self.active_section
        self._recompute_effective_layout(cause="initial")
        server_read = (
            self.active_section == "items" and self.runtime_backend != "local"
        )
        if server_read:
            self._enter_server_read_recovery()
        else:
            self._refresh_local_wc_snapshot()
            self._load_active_section_data()
            self._load_tree_data()
            self.set_timer(
                WC_SNAPSHOT_TIMEOUT_SECONDS,
                self._apply_snapshot_timeout_if_still_loading,
            )
        self._refresh_overview_data()

    def on_resize(self, _event: events.Resize) -> None:
        """Re-derive responsive state without changing the preference."""
        self._recompute_effective_layout(cause="resize")

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

        if not self.is_mounted and section == "items" and requested_backend != "local":
            # Compose precedes on_mount. Arm recovery now so the cold
            # workbench factories use their query-free empty models.
            self._enter_server_read_recovery()

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
                # Review finding I1: was the bare literal "unavailable" --
                # inconsistent with the "states, not faults" vocabulary
                # TASK-2313 AC#2 established everywhere else on this
                # screen, and (worse) indistinguishable from a plain
                # missing value if some future caller ever compared it
                # loosely. This IS a real fault (an exception was just
                # raised fetching the profile), so it stays distinct from
                # both `WatchlistsBackendController.NOT_CONFIGURED_STATUS`
                # (feature not wired up) and `None`/"no runs yet" (healthy,
                # simply-unrun) -- see `_latest_run_status_text`.
                "latest_run_status": WatchlistsBackendController.LOOKUP_FAILED_STATUS,
                "failed_runs": [],
                "active_alert_rules": 0,
            }

    def watch_overview_data(self) -> None:
        """Push new overview counts into the live panes, never recompose.

        TASK-2200 — see the note on the reactive itself. Three surfaces read
        `overview_data`, and each is updated at the narrowest granularity
        that actually re-renders it:

        * `OverviewPane.data` — the pane's own `recompose=True` reactive, so
          this is a PANE-scoped rebuild. It has to be: the pane swaps between
          three whole layouts (loading line / first-run copy / seven cards
          plus a failed-runs table) depending on `profile_state`, so there is
          no set of cells to patch.
        * `InspectorPane.profile_state` — likewise pane-scoped, and likewise
          layout-changing.
        * The Inspector's two count `Static`s — one `update()` each, since
          only their text changes.

        Every lookup degrades: the Overview pane exists only on its own tab,
        and the Inspector is unmounted whenever the right rail is collapsed.
        """
        if not self._dom_is_live:
            return
        try:
            overview = self.query_one("#watchlists-overview-pane", OverviewPane)
        except NoMatches:
            pass
        else:
            overview.data = self.overview_data
        try:
            inspector = self.query_one("#watchlists-entity-inspector", InspectorPane)
        except NoMatches:
            pass
        else:
            inspector.profile_state = self._watchlists_profile_state()
        # Built exactly as `_build_inspector_pane` builds them, so a rebuild
        # and a patch cannot render differently.
        for widget_id, text in (
            (
                "#watchlists-alerts-summary",
                f"Alert rules active: {self.overview_data.get('active_alert_rules', 0)}",
            ),
            (
                "#watchlists-latest-run-summary",
                self._latest_run_status_text(),
            ),
        ):
            try:
                self.query_one(widget_id, Static).update(text)
            except NoMatches:
                continue
        # An overview refresh follows every write verb on this screen, so this
        # is one of the points where the recompose this reactive no longer
        # triggers used to re-enter the Console-follow adapter. What that
        # actually recovers is an adapter that FAILED on the first compose --
        # not a run that has since started, which the handoff caches away
        # permanently after one success. See `_resolve_console_follow_drift`,
        # which states both halves of that cache's behaviour.
        self._request_surface_refresh(self._SURFACE_INSPECTOR)

    @staticmethod
    def _read_tree_data_snapshot(
        service: WatchlistBundleService | None,
        previous: TreeDataSnapshot,
        today_floor_iso: str,
    ) -> TreeDataSnapshot:
        """Read one tree snapshot on a worker thread, retaining failed branches."""
        failures: set[str] = set()

        def read_branch(name: str, reader: Callable[[], Any], fallback: Any) -> Any:
            try:
                return reader()
            except Exception:
                failures.add(name)
                logger.opt(exception=True).debug(
                    "Failed to load watchlists tree branch: {}.", name
                )
                return fallback

        if service is None:
            failures.update(
                {"watchlists", "all_sources", "unassigned_sources", "counts", "source_counts"}
            )
            return TreeDataSnapshot(
                previous.watchlists,
                previous.all_source_rows,
                previous.unassigned_source_rows,
                previous.counts,
                previous.source_counts,
                frozenset(failures),
            )

        watchlists = tuple(
            dict(row)
            for row in read_branch(
                "watchlists", service.list_watchlists, previous.watchlists
            )
        )
        all_source_rows = tuple(
            dict(row)
            for row in read_branch(
                "all_sources", service.list_all_source_rows, previous.all_source_rows
            )
        )
        unassigned_source_rows = tuple(
            dict(row)
            for row in read_branch(
                "unassigned_sources",
                service.list_unassigned_source_rows,
                previous.unassigned_source_rows,
            )
        )

        def read_counts() -> dict[int, dict[str, int]]:
            counts = {
                int(bucket): dict(values)
                for bucket, values in service.get_watchlist_item_counts().items()
            }
            starred = service.get_flagged_items_count()
            counts[STARRED_BUCKET] = {"total": starred, "unread": starred}
            today = service.get_unread_items_count_since(today_floor_iso)
            counts[TODAY_BUCKET] = {"total": today, "unread": today}
            return counts

        counts = read_branch("counts", read_counts, previous.counts)
        source_counts = read_branch(
            "source_counts", lambda: service.get_source_item_counts(), previous.source_counts
        )
        return TreeDataSnapshot(
            watchlists,
            all_source_rows,
            unassigned_source_rows,
            {int(bucket): dict(values) for bucket, values in counts.items()},
            {int(source_id): dict(values) for source_id, values in source_counts.items()},
            frozenset(failures),
        )

    @work(group="wc_tree")
    async def _load_tree_data(self) -> None:
        """Acquire and publish the complete left-rail snapshot off-loop."""
        if self.active_section == "items" and self.runtime_backend != "local":
            self._items_page_loading = False
            self._push_items_pager_state()
            return

        self._tree_load_generation += 1
        generation = self._tree_load_generation
        service = self._watchlist_bundle_service()
        previous = self._tree_snapshot
        snapshot = await asyncio.to_thread(
            self._read_tree_data_snapshot,
            service,
            previous,
            self._today_floor_iso(),
        )
        if generation != self._tree_load_generation:
            return

        self._tree_snapshot = snapshot
        self._tree_watchlists = [dict(row) for row in snapshot.watchlists]
        self._tree_all_source_rows = [dict(row) for row in snapshot.all_source_rows]
        self._tree_unassigned_source_rows = [
            dict(row) for row in snapshot.unassigned_source_rows
        ]
        self._tree_counts = {
            bucket: dict(values) for bucket, values in snapshot.counts.items()
        }
        self._tree_source_counts = {
            source_id: dict(values)
            for source_id, values in snapshot.source_counts.items()
        }
        self._tree_snapshot_failures = snapshot.failures
        notify = getattr(self.app_instance, "notify", None)
        if snapshot.failures and not self._tree_failure_episode_active:
            self._tree_failure_episode_active = True
            if callable(notify):
                notify("Failed to load watchlists.", severity="error")
        elif not snapshot.failures:
            self._tree_failure_episode_active = False
        # Re-resolve the Inspector's breadcrumb against what was just loaded
        # (task-895). `_resolve_breadcrumb_labels` reads `_tree_watchlists`,
        # and until this task nothing could change that list while a scope
        # was in view, so resolving once in `_apply_tree_scope` was enough.
        # The write verbs break that: creating a watchlist scopes to an id
        # that is not in the list yet (the crumb would read "Watchlist 3"),
        # and renaming one leaves the crumb on the old name until the user
        # navigates away and back. `_apply_tree_data_to_live_surfaces` below
        # pushes the resolved labels into the mounted Inspector.
        self._breadcrumb_labels = self._resolve_breadcrumb_labels(self.selected_scope)
        self._apply_tree_data_to_live_surfaces()
        await self._refresh_items_pending_arrivals()

    def _rail_unread_suffix(self) -> str:
        """The collapsed left rail's "N unread" suffix (task-2513 Task 9).

        Empty when there is nothing unread — a permanent "0 unread" is
        chrome, not information, and an expanded rail already shows the
        per-node numbers.
        """
        unread = self._tree_counts.get(ALL_SOURCES_BUCKET, {}).get("unread", 0)
        return f"{unread} unread" if unread else ""

    def _apply_tree_data_to_live_surfaces(self) -> None:
        """Publish freshly loaded tree data without recomposing the screen.

        TASK-2200. `_load_tree_data` used to end in `refresh(recompose=True)`,
        which tore down and rebuilt every region — including ITEMS, whose
        pane may be mid-recompose of its own (a `SourcesPane` closing its
        create form, a `RulesPane` closing an edit form). That is the
        confirmed destroyer behind TASK-1960's crash class: a widget caught
        by the resulting prune mounts nothing, silently, while still
        reporting `is_mounted=True`.

        Everything on this screen that reads what this loader writes, and
        nothing else:

        * The rail itself (`_tree_watchlists`/`_tree_counts`) — rebuilt from
          `_build_tree_pane`, which re-seeds the expansion, tag filter and
          scope the user had, exactly as the full recompose did.
        * The centre header, whose scoped summary resolves a watchlist's
          display name out of `_tree_watchlists` (a rename would otherwise
          sit stale) and whose source count is re-queried.
        * The Inspector's breadcrumb labels, and — in the ITEMS region — the
          Overview's watchlist count and the Artifacts pane's scope label,
          all pushed into the live panes the way `watch_selected_scope` and
          `_load_sources` already push theirs.

        The two ITEMS-region pushes are not an exception to "background loads
        leave ITEMS alone"; they are the reason that rule is stated as "no
        pane is REBUILT" rather than "no pane is touched". Both are single
        reactive assignments onto whichever pane happens to be mounted, so a
        `SourcesPane` create form or a `RulesPane` edit form -- neither of
        which reads tree data -- is not even queried, let alone torn down.

        `ArtifactsPane.scope_label` was missed on the first pass and found by
        the whole-branch review (I1): it resolves a watchlist DISPLAY NAME via
        `_briefing_scope_label` -> `_watchlist_display_name` -> the same
        `_tree_watchlists` list, so renaming the scoped watchlist while the
        Artifacts tab was open repainted the rail, the header summary and the
        breadcrumb but left `#artifacts-scope-note` naming the old one — two
        surfaces on one screen disagreeing about the same watchlist.

        CONTENT is untouched: `ContentPane` reads nothing this loader writes.
        """
        if not self._dom_is_live:
            return
        self._request_surface_refresh(
            self._SURFACE_RAIL,
            self._SURFACE_HEADER,
            self._SURFACE_INSPECTOR,
        )
        try:
            inspector = self.query_one("#watchlists-entity-inspector", InspectorPane)
        except NoMatches:
            pass
        else:
            inspector.breadcrumb_labels = self._breadcrumb_labels
        try:
            overview = self.query_one("#watchlists-overview-pane", OverviewPane)
        except NoMatches:
            pass
        else:
            # TASK-998: the first-run panel distinguishes "no watchlists at
            # all" from "a watchlist with no sources", and only this loader
            # knows the answer. `_build_detail_pane` seeds the same value on
            # a rebuild.
            overview.watchlist_count = len(self._tree_watchlists)
        try:
            artifacts = self.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        except NoMatches:
            pass
        else:
            # Review wave, I1. `_briefing_scope_label` resolves the scoped
            # watchlist's display NAME out of `_tree_watchlists`, so a rename
            # left `#artifacts-scope-note` on the old one. Same seed
            # `_build_detail_pane` applies on a rebuild.
            artifacts.scope_label = self._briefing_scope_label()
        # TASK-2304 AC#2, found in live verification, not by the suite. Which
        # sources the current scope covers is WATCHLIST MEMBERSHIP, and this
        # loader runs after every write that changes it (`Add source`,
        # `Remove`, watchlist delete). The scope itself does not move on those
        # writes, so `watch_tree_scope` never fires, and nothing re-queries
        # the source list either -- so scoping the table (this task's own
        # change) left `Add source` assigning a source into a watchlist whose
        # table stayed empty while the header one line above it had already
        # updated to "(1 source)". The same disagreement this task exists to
        # remove, in the opposite direction.
        #
        # A third ITEMS-region push, and the same kind as the other two: a
        # single reactive assignment onto whichever pane happens to be
        # mounted, never a rebuild -- an open create form is not even queried.
        self._push_scoped_sources_to_pane()

    def _resolve_breadcrumb_labels(self, scope: TreeScope) -> list[str]:
        """Display names for `scope`'s ancestor chain, for the Inspector.

        Both levels resolve from the screen-owned tree snapshot, so the same
        feed remains visibly distinct under All Sources, Unassigned, All
        Unread, and a created watchlist without issuing a per-occurrence DB
        query.
        """
        if scope.kind == "watchlist" and scope.watchlist_id is not None:
            return [self._watchlist_display_name(scope.watchlist_id)]
        if scope.kind != "source" or scope.source_id is None:
            return []

        parent_labels = {
            "all": "All Sources",
            "unassigned": "Unassigned",
            "unread": "All Unread",
        }
        if scope.parent_context == "watchlist" or (
            scope.parent_context is None and scope.watchlist_id is not None
        ):
            parent_label = (
                self._watchlist_display_name(scope.watchlist_id)
                if scope.watchlist_id is not None
                else "Watchlist"
            )
        else:
            parent_label = parent_labels.get(scope.parent_context, "All Sources")
        source_label = next(
            (
                str(row.get("name"))
                for row in self._tree_all_source_rows
                if int(row.get("id", -1)) == int(scope.source_id)
            ),
            f"Source {scope.source_id}",
        )
        return [parent_label, source_label]

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
        self._apply_snapshot_to_live_surfaces()

    def _apply_snapshot_to_live_surfaces(self) -> None:
        """Publish a freshly applied local snapshot without recomposing.

        TASK-2200, the twin of `_apply_tree_data_to_live_surfaces` — see that
        method for why the full-screen `refresh(recompose=True)` this
        replaces was the destroyer behind TASK-1960.

        The snapshot's loading/error/empty/summary marker
        (`_watchlists_status_marker_widgets`) is rendered in exactly one
        place -- the centre header, mounted on every tab since task-2513 --
        and this rebuilds it. (The Read tab used to have a second, inline
        copy inside the FEEDS region; that region is gone.)

        The Inspector's two snapshot-derived widgets are patched rather than
        rebuilt, following `_repaint_item_status_cell`'s discipline: the
        `State:` line is one `Static.update`, and the Console attach control
        is one `disabled`/`tooltip` pair straight off `_wc_attach_state()` —
        the same tuple `compose_content` hands `_build_inspector_pane`.
        """
        if not self._dom_is_live:
            return
        self._request_surface_refresh(
            self._SURFACE_HEADER, self._SURFACE_INSPECTOR
        )
        try:
            summary = self.query_one("#watchlists-state-summary", Static)
        except NoMatches:
            pass
        else:
            summary.update(self._watchlists_state_summary_text())
        attach_disabled, attach_tooltip = self._wc_attach_state()
        try:
            attach = self.query_one("#wc-attach-to-console", Button)
        except NoMatches:
            pass
        else:
            attach.disabled = attach_disabled
            attach.tooltip = attach_tooltip

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
        render "No sources yet." -- directly underneath a scoped readout
        that was showing rows. That split is not hypothetical: the two paths
        resolve their `SubscriptionsDB` independently, and in the UI
        harnesses they land on different temp files entirely.

        The snapshot stays the *health* probe (`_wc_loaded` /
        `_wc_lookup_error` in `_wc_attach_state` and
        `_watchlists_status_marker_widgets` are untouched): it is the only
        caller that can distinguish "the service is unavailable" or "policy
        denied" from "there are no rows", which a synchronous local query
        cannot report.
        """
        if self._local_watchlist_count > 0:
            return True
        return bool(self.scoped_source_rows())

    def _staging_summary_line(self, rows: Sequence[Mapping[str, Any]]) -> Text:
        """The one line the Console-staging block collapses to.

        Fix round 1, Finding 1: the block used to enumerate
        `_local_watchlist_records`, which reaches `get_all_subscriptions`
        through `WatchlistScopeService.list_watch_items` -- the *same* table
        `scoped_source_rows()` reads. The screen's scoped readout therefore
        printed every source twice in one box (once scoped, once not), in
        identical typography. Staging now follows the tree scope instead, so
        the block only has to say what pressing the button would send.

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

        Reads the same `scoped_source_rows()` the centre header's summary
        line renders (fix round 1, Finding 1), so selecting "Morning AI
        Brief" and then staging stages Morning AI Brief -- not, as before,
        every local source regardless of where the user had navigated.

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
        `_watchlists_status_marker_widgets` uses, split out so
        `compose_content` can get it without constructing (and discarding) a
        pane widget just to read it.
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

        A factory, not an instance: collapsing or expanding the left rail
        swaps its widget for a freshly built one, and a widget instance can
        only be mounted once (see the factory note on
        `WatchlistsWorkbench.__init__`).

        Seeds `expanded`/`active_tag` from screen state (whole-branch review,
        Finding 2) the same way `_build_detail_pane` seeds the panes' rows
        and `_build_inspector_pane` seeds scope and breadcrumbs -- otherwise
        the rail collapses on every section switch. Seeds `active_scope`
        from `tree_scope` for the same reason (task-876): a section switch
        or rail toggle rebuilds a brand new `WatchlistTree`, and without
        this the selection highlight would reset to nothing every time,
        even though the scope itself survived on the screen.
        """
        recovering = self.active_section == "items" and self._read_recovery_active
        return WatchlistTree(
            watchlists=[] if recovering else self._tree_watchlists,
            counts={} if recovering else self._tree_counts,
            source_rows_loader=(
                (lambda _watchlist_id: [])
                if recovering
                else self._load_source_rows_for_tree
            ),
            expanded=(
                frozenset() if recovering else self._tree_expanded_watchlist_ids
            ),
            expanded_root_kinds=(
                frozenset() if recovering else self._tree_expanded_root_kinds
            ),
            active_tag=None if recovering else self._tree_active_tag,
            active_scope=self.tree_scope,
            write_disabled_reason=self._tree_write_disabled_reason(),
            selection_disabled_reason=self._tree_selection_disabled_reason(),
            source_counts={} if recovering else self._tree_source_counts,
            all_source_rows=[] if recovering else self._tree_all_source_rows,
            unassigned_source_rows=(
                [] if recovering else self._tree_unassigned_source_rows
            ),
            unread_pin_source_id=(
                self.tree_scope.source_id
                if self.tree_scope.kind == "source"
                and self.tree_scope.parent_context == "unread"
                else None
            ),
            id="wl-tree",
        )

    def _tree_write_disabled_reason(
        self, *, runtime_backend: str | None = None
    ) -> str | None:
        """Why the tree's five write verbs cannot run, or `None` (task-895).

        Two blockers, in the order the user can act on them:

        * The **server** backend. Not a cosmetic hide -- there is no request
          shape that can carry a watchlist membership edit (see
          `WC_SERVER_WRITE_RECOVERY`), so the actions are disabled and say
          so, with the backend selector named as the way out.
        * **No bundle service.** The same degrade-don't-crash contract every
          other caller of `_watchlist_bundle_service()` follows; the copy is
          the screen's existing `WC_SERVICE_UNAVAILABLE_COPY` rather than a
          second phrasing of the same condition.

        Args:
            runtime_backend: Backend whose write availability to evaluate.
                Defaults to the backend currently visible on the screen.

        Returns:
            The reason string, used verbatim as both the disabled buttons'
            tooltip and the visible note beneath them, or `None` when writes
            are available.
        """
        backend = self.runtime_backend if runtime_backend is None else runtime_backend
        if backend == "server":
            return WC_SERVER_WRITE_RECOVERY.disabled_tooltip
        if self._watchlist_bundle_service() is None:
            return WC_SERVICE_UNAVAILABLE_COPY
        return None

    def _tree_selection_disabled_reason(self) -> str | None:
        """Why contextual feed children cannot commit on this surface."""
        if self.runtime_backend == "server" and self.active_section != "items":
            return _INDIVIDUAL_FEED_SELECTION_DISABLED
        return None

    def _load_source_rows_for_tree(self, watchlist_id: int) -> list[dict[str, Any]]:
        """Fetch one watchlist's source rows for the tree, synchronously.

        Safe on the UI thread: the tree calls this during `compose()` when a
        watchlist is expanded, and `list_source_rows` is one JOIN (Task 1),
        not a fan-out of per-source queries.
        """
        if self.active_section == "items" and self._read_recovery_active:
            return []
        try:
            return self._watchlist_bundle_service().list_source_rows(watchlist_id)
        except Exception:
            logger.opt(exception=True).debug("Failed to load tree source rows.")
            return []

    def scoped_source_rows(self) -> list[dict[str, Any]]:
        """Source rows the current tree scope covers.

        The centre header's scoped summary line and the Sources table both
        render these (and, after Task 7, the items list scope does too), so
        selecting a node in the tree actually narrows what the centre shows
        rather than only recording a selection (Task 7). Kept on the screen
        (not the pane) because the workbench recomposes and pane-local state
        does not survive it -- the same reasoning already applied to
        `tree_scope` itself.

        Reads `tree_scope`, not `selected_scope`: only tree navigation
        changes what that summary covers. See the note on those two
        reactives for why they are not the same value.

        Each branch costs exactly one query (`list_source_rows`,
        `list_all_source_rows`, or `list_unassigned_source_rows`); the
        `source` scope reuses whichever of those already names the right
        table rather than adding a second query just to filter down to one
        row.

        Returns:
            One dict per source with ``id``, ``name`` and ``type``, or an
            empty list if the bundle service is unavailable or lookup fails.
        """
        if self.active_section == "items" and self._read_recovery_active:
            return []
        scope = self.tree_scope
        try:
            if scope.kind == "watchlist" and scope.watchlist_id is not None:
                service = self._watchlist_bundle_service()
                return [] if service is None else service.list_source_rows(scope.watchlist_id)
            if scope.kind == "source" and scope.source_id is not None:
                if scope.parent_context == "unassigned":
                    rows = self._tree_unassigned_source_rows
                elif scope.watchlist_id is not None:
                    service = self._watchlist_bundle_service()
                    rows = [] if service is None else service.list_source_rows(scope.watchlist_id)
                else:
                    rows = self._tree_all_source_rows
                return [r for r in rows if int(r["id"]) == int(scope.source_id)]
            if scope.kind == "unassigned":
                return [dict(row) for row in self._tree_unassigned_source_rows]
            return [dict(row) for row in self._tree_all_source_rows]
        except Exception:
            logger.opt(exception=True).debug("Failed to resolve scoped source rows.")
            return []

    def _create_form_watchlist_choices(self) -> list[dict[str, Any]]:
        """Watchlists the create form may file a new source into (TASK-2302).

        Empty whenever membership cannot be written at all -- the server
        backend has no wire path for it and a missing bundle service has no
        store, exactly the two conditions `_tree_write_disabled_reason`
        already names for the rail's own write verbs. Offering a destination
        the create would then silently ignore is the defect this task exists
        to remove, restated one control over; with no choices the Select
        still renders, showing `Unassigned`, which is what actually happens.
        """
        if self._tree_write_disabled_reason() is not None:
            return []
        return [dict(watchlist) for watchlist in self._tree_watchlists]

    def _create_form_source_types(
        self, runtime_backend: str
    ) -> tuple[str, ...]:
        """Return the create-form contract for one runtime backend."""
        return tuple(
            self._controller.create_form_source_types(
                runtime_backend=runtime_backend
            )
        )

    def _sync_live_source_create_backend(self) -> None:
        """Push the visible backend contract into the mounted Sources pane."""
        if not self._dom_is_live:
            return
        try:
            pane = self.query_one("#watchlists-sources-pane", SourcesPane)
        except NoMatches:
            return
        pane.configure_create_backend(
            self.runtime_backend,
            self._create_form_source_types(self.runtime_backend),
        )

    def _scope_default_destination(self) -> Any:
        """The watchlist a new source joins by default: the one in scope.

        TASK-2302 AC#1. The `all` and `unassigned` roots are not watchlists,
        so they resolve to Unassigned -- which is the truthful answer for
        both, and for a `source` scope the answer is that source's own
        watchlist (a source node is always reached THROUGH one).

        Returns:
            A watchlist id, or `SourcesPane.UNASSIGNED_DESTINATION`.
        """
        scope = self.tree_scope
        if scope.kind in ("watchlist", "source") and scope.watchlist_id is not None:
            if any(
                int(watchlist.get("id", -1)) == int(scope.watchlist_id)
                for watchlist in self._create_form_watchlist_choices()
            ):
                return int(scope.watchlist_id)
        return SourcesPane.UNASSIGNED_DESTINATION

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
        if scope.kind == "starred":
            return "Starred"
        if scope.kind == "unread":
            return "All Unread"
        if scope.kind == "today":
            return "Today"
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
            labels = self._resolve_breadcrumb_labels(scope)
            if rows:
                source_label = str(rows[0].get("name"))
                return (
                    f"{labels[0]} / {source_label}"
                    if labels
                    else source_label
                )
            if len(labels) == 2:
                return " / ".join(labels)
            if scope.source_id is not None:
                return f"Source {scope.source_id}"
        return "All sources"

    def _watchlists_status_marker_widgets(
        self,
        scoped_rows: Sequence[Mapping[str, Any]],
        *,
        section: str | None = None,
    ) -> list[Widget]:
        """The snapshot's own loading/error/empty/summary marker.

        Rendered in exactly one place: `_build_centre_status_header`, the
        centre header mounted on every tab. (TASK-1344 extracted this so the
        readout could also live inline in the FEEDS region's Read-tab body --
        avoiding a real regression, Sources/Runs/... silently losing all
        visibility into "snapshot still loading" / "service unavailable" /
        "no sources yet"; task-2513 removed that region, leaving the header
        as the only home.)

        Keyed on the async snapshot, NOT on `scoped_source_rows()`: the
        snapshot is the only service-health probe on this screen -- it is
        what distinguishes "the Watchlists service is unavailable" and
        "policy denied" (whose recovery state supplies `#wc-service-error`'s
        copy) from "there are no rows". `scoped_source_rows()` is a
        synchronous local query that returns `[]` for every one of those
        cases, and `#wc-loading-state` has no meaning for it at all. In
        production the two agree anyway: both read `subscriptions`.

        Called fresh on every header rebuild, so it must stay
        side-effect-free (same discipline as every other content factory
        here -- see `WatchlistsWorkbench.__init__`'s docstring on why
        `content` holds factories, not instances).

        Args:
            scoped_rows: The current tree scope's source rows
                (`scoped_source_rows()`), used only by the summary-line
                branch. Passed in rather than re-resolved so the caller
                does not query twice.

        Returns:
            One widget (the loading/error/empty-state text, or the
            one-line staging summary) in every case except the empty-state,
            which also appends the Create-source/Import-OPML action row.
        """
        if not self._wc_loaded:
            return [
                Static(
                    "Loading local Watchlists snapshot...",
                    id="wc-loading-state",
                )
            ]
        if self._wc_lookup_error:
            recovery_state = self._wc_lookup_recovery_state
            return [
                Static(
                    self._wc_lookup_error,
                    id=(
                        recovery_state.stable_selector
                        if recovery_state is not None
                        else "wc-service-error"
                    ),
                )
            ]
        if not self._has_local_wc_context():
            # TASK-2312, AC#3: this marker renders in the shared header on
            # EVERY section (it is genuinely global -- "is there any local
            # Watchlists data at all" -- not a Sources-tab fact), so its
            # copy must read that way rather than as borrowed Sources-tab
            # content. "No Watchlists sources yet." names the app-level
            # noun instead of a bare "No sources", which on, say, the
            # Overview or Runs tab previously read like Sources-tab copy
            # had leaked in (UAT finding).
            widgets: list[Widget] = [
                Static(
                    "No Watchlists sources yet.",
                    id="wc-empty-state",
                ),
            ]
            # TASK-2313, AC#3 (duplicate affordances): the New source/
            # Import OPML pair here is the only bootstrap path from a
            # section that has no create form of its own -- genuinely
            # needed everywhere else. On Sources itself, its own toolbar
            # (`sources_pane.py`) already offers the identical pair one
            # row below this header, so repeating them here was the "Import
            # OPML twice on one screen" UAT finding. Omitted on Sources
            # only; every other section still gets the one bootstrap path.
            if (self.active_section if section is None else section) != "sources":
                widgets.append(
                    Horizontal(
                        # TASK-2303 AC#1: the same create verb the Sources
                        # pane uses. This button and that one open the same
                        # form, so they carry the same label; "Create
                        # source" beside the rail's old "Add source" was
                        # two verbs for two operations that read as one.
                        Button(
                            "New source",
                            id="wc-empty-create-source",
                            variant="primary",
                            tooltip="Create a Watchlists source that does not exist yet.",
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
            return widgets
        # One line, not a second source list (fix round 1, Finding 1).
        # `#wc-watchlists-summary` keeps its id -- it is the "snapshot
        # finished loading" terminal selector the guard suites wait on --
        # and says what pressing Stage would send, which is the scope
        # `_build_centre_status_header` names just above it.
        # `#wc-snapshot-title` is folded into this same line rather than
        # kept as a separate heading; no test referenced it, and a one-line
        # block does not need a title row.
        return [
            Static(
                self._staging_summary_line(scoped_rows),
                id="wc-watchlists-summary",
                classes="destination-section",
            )
        ]

    def _build_centre_status_header(self, section: str | None = None) -> Vertical:
        """Build the ALWAYS-rendered centre header: the section tab strip
        plus the snapshot's own loading/error/empty/summary marker.

        The tab strip and the snapshot markers are cross-cutting chrome,
        not region content, so they live here rather than inside any
        region: `WatchlistsWorkbench`'s `header=` factory, wired
        unconditionally (`compose_content`) since task-2513 removed the
        FEEDS region, whose Read-tab body used to carry an identical-
        looking inline copy of both — mounting both would duplicate
        `#wl-tabs`.

        Called fresh on every workbench rebuild, so it must stay
        side-effect-free, like every other factory here.

        Returns:
            A `Vertical` holding the tab strip and whichever status marker
            `_watchlists_status_marker_widgets` returns for the current
            snapshot state.
        """
        section = self.active_section if section is None else section
        children: list[Widget] = [
            WatchlistsTabStrip(active_section=section, id="wl-tabs"),
        ]
        if section == "items" and self._read_recovery_active:
            children.append(
                Static(
                    "Server-backed Read is unavailable. Switch to Local in Reader.",
                    id="watchlists-read-recovery-status",
                )
            )
        else:
            scoped_rows = self.scoped_source_rows()
            children.extend(
                self._watchlists_status_marker_widgets(scoped_rows, section=section)
            )
        return Vertical(
            *children,
            id="wl-centre-status",
            classes="watchlists-centre-status",
        )

    def _build_detail_pane(self, section: str | None = None) -> Vertical:
        """Build the ITEMS-region content: the active-section-routed pane.

        Called fresh on every region rebuild — see the factory note on
        `WatchlistsWorkbench.__init__`.

        Seeding discipline (task-15778): every `recompose=True` reactive is
        seeded with `set_reactive`, never plain assignment. A plain
        assignment on the freshly constructed, not-yet-mounted pane runs
        `refresh(recompose=True)` the instant the seeded value differs from
        the class default (`[] != [row]`), which queues a `_check_recompose`
        that fires just after the pane mounts — one full extra pane rebuild
        per region build, measured on every data-carrying section switch
        (task-15461's residual 4). `compose()` reads the same reactives, so
        `set_reactive` renders identically without the queued rebuild. The
        panes' pre-mount watchers were audited per-branch before this
        change: every watcher on a converted reactive is an
        `is_mounted`-gated no-op pre-mount, so nothing is lost by skipping
        it. NON-recompose reactives deliberately stay plain assignments —
        `RunsPane.selected_run`'s watcher side effects are load-bearing
        (see that branch).
        """
        section = self.active_section if section is None else section
        detail_title = self._SECTION_DETAIL_TITLE.get(section, "Detail")
        children: list[Widget] = [
            Static(
                detail_title,
                classes="destination-section watchlists-column-title",
                id="watchlists-detail-title",
            )
        ]
        if section == "overview":
            overview = OverviewPane(id="watchlists-overview-pane")
            overview.set_reactive(OverviewPane.data, self.overview_data)
            # TASK-998: lets the first-run panel distinguish "no watchlists at
            # all" from "a watchlist with no sources in it" -- `overview_data`
            # counts sources, items and runs, never watchlists.
            overview.set_reactive(
                OverviewPane.watchlist_count, len(self._tree_watchlists)
            )
            children.append(overview)
        elif section == "sources":
            sources_pane = SourcesPane(id="watchlists-sources-pane")
            source_types = self._create_form_source_types(self.runtime_backend)
            sources_pane.configure_create_backend(
                self.runtime_backend,
                source_types,
            )
            # Seed the last-loaded rows and selection (Finding 2, fix round
            # 2) the same way RunsPane/NotificationsPane already do below —
            # without this the table renders empty until the next unrelated
            # navigation happens to trigger `_load_sources` again.
            # Scoped (TASK-2304 AC#2): a rebuild must not quietly re-widen
            # the table back to every source while the header still names one
            # watchlist.
            sources_pane.set_reactive(
                SourcesPane.sources, self.scoped_loaded_sources()
            )
            sources_pane.selected_source = self.selected_source
            # TASK-2309: re-seed from screen state for the identical
            # rebuild-survival reason as `selected_source` on the line
            # above -- a region rebuild (collapse/expand, a tab
            # switch) constructs a brand new `SourcesPane`, and without this
            # a check still running would render its Check-now button back
            # to enabled/"Check now" until the run's own completion repaint
            # reached this new instance.
            sources_pane.busy_source_ids = frozenset(self._checks_in_flight)
            # TASK-2302: the destination Select's options and its default,
            # seeded BEFORE `show_create_form` so a form that opens as part
            # of this very rebuild has them. The pane holds no service of its
            # own, the same as every other pane here.
            sources_pane.watchlist_choices = self._create_form_watchlist_choices()
            sources_pane.default_destination = self._scope_default_destination()
            sources_pane.create_draft_destination = (
                self._source_create_draft_destination
                if self._source_create_draft_destination is not None
                else sources_pane.default_destination
            )
            if self._source_create_draft_type is not None:
                sources_pane.set_reactive(
                    SourcesPane.create_draft_source_type,
                    (
                        self._source_create_draft_type
                        if self._source_create_draft_type in source_types
                        else "rss"
                    ),
                )
            # Seed the create-form draft so it survives this pane being
            # reconstructed (see the note on `_source_create_draft` in
            # __init__ and CreateFormDraftChanged/CreateFormVisibilityChanged
            # in sources_pane.py). `set_reactive` (task-15778):
            # `watch_show_create_form` is entirely `is_mounted`-gated, so the
            # plain assignment bought nothing pre-mount but the extra
            # recompose.
            sources_pane.set_reactive(
                SourcesPane.show_create_form, self._source_create_form_open
            )
            sources_pane.create_draft_name = self._source_create_draft["name"]
            sources_pane.create_draft_url = self._source_create_draft["url"]
            sources_pane.create_draft_tags = self._source_create_draft["tags"]
            sources_pane.create_draft_active = self._source_create_draft_active
            sources_pane.create_draft_frequency = (
                self._source_create_draft_frequency
            )
            if self._source_create_draft_selectors is not None:
                sources_pane.create_draft_ignore_selectors = (
                    self._source_create_draft_selectors
                )
            children.append(sources_pane)
        elif section == "runs":
            runs_pane = RunsPane(id="watchlists-runs-pane")
            # `runs` is the pane's only `recompose=True` reactive, so it is
            # the only one converted to `set_reactive` (task-15778). The
            # four below stay PLAIN assignments on purpose:
            # `watch_selected_run` is load-bearing even pre-mount -- it
            # starts the status poll when the seeded run is still running
            # (without it, a region rebuild mid-run would freeze the run's
            # status until a manual refresh) -- and none of the four is
            # `recompose=True`, so none of them costs the extra rebuild this
            # task removes.
            runs_pane.set_reactive(RunsPane.runs, self._loaded_runs)
            runs_pane.selected_run = self.selected_run
            # After `selected_run`, never before: setting the selection clears
            # the pane's detail (a run's items must never outlive the run they
            # belong to -- see `RunsPane.watch_selected_run`).
            runs_pane.run_items = self._run_detail_items
            runs_pane.run_logs = self._run_detail_logs
            runs_pane.run_items_note = self._run_detail_items_note
            children.append(runs_pane)
        elif section == "items":
            # Seed the last-loaded rows (Finding 2, fix round 2) — see the
            # note on `sources_pane.sources` above; same rebuild, same gap.
            # `items` is seeded without its async watcher: on a freshly
            # constructed pane there is no mounted list to patch, and
            # invoking that watcher here only creates an un-awaited
            # coroutine. Compose reads the seeded value normally.
            items_pane = ArticleListPane(id="watchlists-items-pane")
            # The surrounding detail pane also owns its one-line title.
            # Consume only the remaining height so the fixed legend/pager
            # stay inside the permanent Feed Items column at every terminal
            # height; `100%` here would place that chrome below the viewport.
            items_pane.styles.height = "1fr"
            items_pane.styles.min_height = 0
            items_pane.set_reactive(ArticleListPane.items, self._loaded_items)
            # Seed the filter, the search box and the selection too
            # (whole-branch review, Important) -- the sibling Sources/Runs/
            # Notifications panes above and below already re-seed their
            # selection, and this one seeded only `.items`, so every rebuild
            # silently reset the user's filtered view to "all items, nothing
            # selected". See `_items_status_filter` in `__init__`. The value
            # goes through `_normalize_items_status_filter` (TASK-3072): the
            # mirror can still hold a pre-reader per-status value, which the
            # two-option Select would reject.
            items_pane.status_filter = self._effective_items_status_filter()
            items_pane.status_filter_disabled_reason = (
                self._items_filter_disabled_reason()
            )
            items_pane.search_query = self._items_search_query
            items_pane.selected_item = self._selected_content_item
            items_pane.page_number = self._items_page_index + 1
            items_pane.has_previous = self._items_page_index > 0
            items_pane.has_next = self._items_has_next
            items_pane.page_loading = self._items_page_loading
            items_pane.snapshot_count = self._items_snapshot_count
            items_pane.new_items_note = self._items_arrival_note()
            items_pane.search_results_authoritative = (
                self._items_search_results_authoritative
            )
            if self._items_retry_message is not None:
                items_pane.display = False
                children.extend(
                    (
                        Static(
                            Text(self._items_retry_message),
                            id="watchlists-items-retry-state",
                        ),
                        Button(
                            "Retry",
                            id="watchlists-items-retry-button",
                            variant="primary",
                            disabled=self._items_retry_inflight,
                        ),
                    )
                )
            children.append(items_pane)
        elif section == "rules":
            # Seed the last-loaded rows (Finding 2, fix round 2) — see the
            # note on `sources_pane.sources` above; same rebuild, same gap.
            rules_pane = RulesPane(id="watchlists-rules-pane")
            rules_pane.set_reactive(RulesPane.rules, self._loaded_rules)
            # Seed the edit-form state so an in-progress rule edit survives
            # this pane being reconstructed (Finding 4, fix round 2) — the
            # same treatment the Sources create-form draft already gets
            # above; see `_rule_form_open`/`_rule_form_editing` in __init__
            # and RuleFormVisibilityChanged in rules_pane.py. `edit_rule`
            # itself takes the `set_reactive` route on an unmounted pane
            # (task-15778) — see its own comment.
            if self._rule_form_open:
                if self._rule_form_editing is not None:
                    rules_pane.edit_rule(self._rule_form_editing)
                else:
                    rules_pane.set_reactive(RulesPane.show_rule_form, True)
            children.append(rules_pane)
        elif section == "notifications":
            notifications_pane = NotificationsPane(id="watchlists-notifications-pane")
            notifications_pane.set_reactive(
                NotificationsPane.notifications, self._loaded_notifications
            )
            notifications_pane.set_reactive(
                NotificationsPane.selected_notification,
                self.selected_notification,
            )
            children.append(notifications_pane)
        elif section == "artifacts":
            # Seeded from screen state for the same reason every sibling
            # above is -- this is a factory the workbench calls on every
            # region rebuild, so a fresh pane's reactives start at their
            # class defaults. Every reactive goes through `set_reactive`
            # (task-15778): most are `recompose=True`, and the six
            # selection-derived ones task-15779 flipped to plain reactives
            # carry watchers with mounted-only side effects (a
            # `BriefingSelected`/`ScriptSelected` post, an in-place table
            # patch, a detail-region refresh) that a seeding assignment
            # must not fire -- `watch_selected_briefing` even guards
            # specifically against wiping this very seeding.
            artifacts_pane = ArtifactsPane(id="watchlists-artifacts-pane")
            seed = artifacts_pane.set_reactive
            seed(ArtifactsPane.briefings, self._loaded_briefings)
            seed(ArtifactsPane.selected_briefing, self._selected_briefing)
            seed(ArtifactsPane.scope_label, self._briefing_scope_label())
            seed(ArtifactsPane.can_generate, self._can_generate_briefing())
            seed(
                ArtifactsPane.default_provider_display,
                self._briefing_provider_display(),
            )
            seed(ArtifactsPane.selection_mode, self._briefing_selection_mode)
            seed(ArtifactsPane.presets, self._loaded_briefing_presets)
            seed(ArtifactsPane.default_preset_id, self._briefing_default_preset_id)
            seed(
                ArtifactsPane.briefing_cadence_seconds,
                self._briefing_cadence_seconds,
            )
            seed(
                ArtifactsPane.briefing_schedules_enabled,
                self._briefing_schedules_enabled(),
            )
            seed(ArtifactsPane.scripts, self._loaded_scripts)
            seed(ArtifactsPane.selected_script, self._selected_script)
            seed(ArtifactsPane.script_audio, self._loaded_script_audio)
            seed(ArtifactsPane.scripts_with_audio, self._scripts_with_audio)
            seed(ArtifactsPane.citations, self._loaded_citations)
            seed(
                ArtifactsPane.has_audio_episodes,
                self._watchlist_has_audio_episodes,
            )
            seed(
                ArtifactsPane.chachanotes_available,
                self._chachanotes_db() is not None,
            )
            seed(
                ArtifactsPane.can_serve_feed,
                self._last_feed_export_directory is not None,
            )
            seed(ArtifactsPane.feed_server_running, self._feed_server.is_running)
            seed(ArtifactsPane.feed_server_url, self._feed_server.url)
            children.append(artifacts_pane)
        return Vertical(
            *children,
            id="watchlists-detail-pane",
            classes="destination-workbench-pane",
        )

    def _push_items_pager_state(self) -> None:
        """Push screen-owned snapshot presentation into the mounted Read pane."""
        try:
            pane = self.query_one("#watchlists-items-pane", ArticleListPane)
        except NoMatches:
            return
        pane.page_number = self._items_page_index + 1
        pane.has_previous = self._items_page_index > 0
        pane.has_next = self._items_has_next
        pane.page_loading = self._items_page_loading
        pane.snapshot_count = self._items_snapshot_count
        pane.new_items_note = self._items_arrival_note()
        pane.search_results_authoritative = (
            self._items_search_results_authoritative
        )

    def _items_arrival_note(self) -> str:
        """Return the Reader pill copy for the committed arrival count."""
        count = self._items_pending_arrivals
        if count <= 0:
            return ""
        noun = "item" if count == 1 else "items"
        return f"{count} new {noun}"

    async def _refresh_items_pending_arrivals(self) -> bool:
        """Publish arrivals for the exact committed Reader snapshot only."""
        snapshot = self._items_snapshot
        backend = self.runtime_backend
        section = self.active_section
        if snapshot is None or backend != "local" or section != "items":
            return False
        self._items_arrival_generation += 1
        generation = self._items_arrival_generation
        try:
            count = await self._controller.count_reader_item_arrivals(
                runtime_backend=backend,
                snapshot_max_item_id=snapshot.watermark,
                **snapshot.query.as_kwargs(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "Failed to count watchlist Reader arrivals (exception_type={}).",
                type(exc).__name__,
            )
            return False
        if (
            generation != self._items_arrival_generation
            or self._items_snapshot is not snapshot
            or self.runtime_backend != backend
            or self.active_section != section
        ):
            return False
        self._items_pending_arrivals = max(0, int(count))
        self._push_items_pager_state()
        return True

    def _reset_items_paging_for_context(self, *, loading: bool) -> None:
        """Invalidate parked Reader paging without issuing an item query."""
        self._discard_items_view_state()
        timer = getattr(self, "_items_search_reload_timer", None)
        if timer is not None:
            timer.stop()
            self._items_search_reload_timer = None
        self._items_page_index = 0
        self._items_has_next = False
        self._items_page_loading = loading
        self._items_search_results_authoritative = False
        self._items_snapshot_generation += 1
        self._items_pending_query_key = None
        self._items_inflight_replacement = None
        self._items_inflight_continuation = None
        self._push_items_pager_state()

    def _enter_server_read_recovery(self) -> None:
        """Clear item-specific state before presenting Server Read recovery."""
        self._read_recovery_active = True
        self._items_retry_message = None
        self._items_retry_inflight = False
        self._reset_items_paging_for_context(loading=False)
        self._items_status_filter = "all"
        self._items_search_query = ""
        self._items_snapshot = None
        self._items_snapshot_count = 0
        self._items_pending_arrivals = 0
        self._selected_content_page_key = None
        self._loaded_items = []
        self._selected_content_item = None
        try:
            pane = self.query_one("#watchlists-items-pane", ArticleListPane)
        except NoMatches:
            pass
        else:
            pane.items = []
            pane.selected_item = None
            pane.status_filter = "all"
            pane.search_query = ""
            self._push_items_pager_state()
        self._request_surface_refresh(
            self._SURFACE_RAIL,
            self._SURFACE_HEADER,
            self._SURFACE_READER,
            self._SURFACE_INSPECTOR,
        )

    async def _recover_local_read(self) -> None:
        """Commit local navigation only after the normal item load succeeds."""
        if not await self._replace_items_snapshot(reason="return_to_read"):
            return
        if self.runtime_backend != "local" or self.active_section != "items":
            return
        self._read_recovery_active = False
        self._load_tree_data()
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()
        self._request_surface_refresh(
            self._SURFACE_RAIL,
            self._SURFACE_HEADER,
            self._SURFACE_READER,
            self._SURFACE_INSPECTOR,
        )

    def _build_content_pane(self) -> Widget:
        """Build the CONTENT-region content: the reader for the last
        selected item (Task 4).

        Called fresh on every region rebuild, like every other region
        builder here -- see the factory note on `WatchlistsWorkbench.__init__`.
        Seeded from `_selected_content_item` (Finding pattern established by
        `_build_inspector_pane`'s `selected_entity` seeding above): without
        this, a collapse/expand would construct a brand new
        `ContentPane` whose `item` reactive starts back at its class default
        of `None`, silently clearing the reader.

        Deliberately not gated on `active_section`: unlike `_build_detail_pane`,
        which swaps in a different pane per tab, the reader is a persistent
        cross-cutting surface for whatever item was last opened, regardless
        of which section the user is currently viewing.

        `ContentPane` does not draw its own heading (see `SELF_HEADED_REGIONS`
        in `watchlists_workbench.py`), so `WatchlistsWorkbench` prepends the
        generic "Content" title above whatever this returns.

        """
        if self.active_section == "items" and self._read_recovery_active:
            return Vertical(
                Static(
                    "Read and its permanent Reader are local-only. "
                    "Switch to Local to browse items stored on this device.",
                    id="watchlists-read-local-only-copy",
                ),
                Button(
                    "Switch to Local",
                    id="watchlists-switch-local",
                    tooltip="Switch to the Local backend and load feed items.",
                ),
                id="watchlists-read-local-only",
                classes="destination-workbench-pane",
            )

        pane = ContentPane(id="watchlists-content-pane")
        # `set_reactive`: `item` is the pane's one `recompose=True` reactive
        # and has no watcher, so a plain assignment here bought nothing but
        # the queued extra recompose whenever an item was selected — a full
        # second render of the article, inside the very swap task-15778
        # batches. `position` below is a non-recompose reactive whose watcher
        # patches in place and stays plain, same audit as `_build_detail_pane`.
        pane.set_reactive(ContentPane.item, self._selected_content_item)
        # TASK-3072 plan task 9: re-seed the footer the same way `item` is
        # re-seeded just above, so a region rebuild re-renders the same
        # position. Guarded inside `_reader_position_text` for the build
        # order where the items pane is not mounted yet ("" then; the first
        # selection pushes the real value).
        pane.position = self._reader_position_text()
        return pane

    def _watchlists_are_empty(self) -> bool:
        """Whether this profile has nothing in Watchlists yet (TASK-998).

        Delegates to `OverviewPane.profile_is_empty`, which is the one
        definition of this question (Qodo #3 on PR #1017). It used to be
        copied here, and two copies deciding what the Overview region and the
        Inspector each say is a drift waiting to happen -- the two disagreeing
        is precisely the confusing first-run state TASK-998 removed.

        Returns:
            True only once `overview_data` has loaded and reports nothing.
        """
        return OverviewPane.profile_is_empty(self.overview_data)

    def _watchlists_profile_state(self) -> str:
        """Loading, empty or populated (TASK-1020).

        The same call the Overview region makes, so the Inspector's own
        first-run text can never contradict it.

        Returns:
            One of `OverviewPane.LOADING`/`EMPTY`/`POPULATED`.
        """
        return OverviewPane.profile_state(self.overview_data)

    @staticmethod
    def _console_follow_copy(
        latest_console_item: Any,
    ) -> tuple[str, Any, Any, bool, str]:
        """Everything the Inspector's Console-follow row renders, as data.

        Extracted (TASK-2200) so the same derivation serves two callers: the
        builder below, and `_resolve_console_follow_drift`, which compares it
        against what is actually on screen. The Console-follow state comes
        from an app-level adapter that is only ever *polled* at render time --
        until this task, "render time" meant every full-screen recompose, so
        an adapter that failed once was picked up by whichever background
        loader recomposed next. With those recomposes gone, that recovery has
        to be detected deliberately.

        Args:
            latest_console_item: The adapter's answer, or `None`.

        Returns:
            `(status_widget_id, status_copy, button_label, button_disabled,
            button_tooltip)` -- byte-identical to what this section rendered
            before the extraction, including the two DIFFERENT status ids
            (they are the section's own available/unavailable signal).
        """
        if latest_console_item is None:
            # TASK-2313, AC#6: "Console follow" appeared twice in two
            # adjacent lines (the status text, then the button label) --
            # UAT read it as jargon duplicated. The button's label now
            # matches the ENABLED state's own verb phrasing ("Follow ... in
            # Console") instead of restating the noun phrase a second
            # time; `disabled=True` (below) already conveys unavailability
            # visually, and the tooltip still spells out why.
            return (
                "watchlists-console-unavailable",
                "No active Watchlists run is available for Console follow.",
                "Follow in Console",
                True,
                "Unavailable until Watchlists has an active run with Console context.",
            )
        title = str(getattr(latest_console_item, "title", None) or "Untitled")
        status = str(getattr(latest_console_item, "status", None) or "unknown")
        return (
            "watchlists-console-available",
            Text.from_markup(
                "Console can follow latest Watchlists run: "
                f"{escape_markup(title)} ({escape_markup(status)})."
            ),
            Text.from_markup(f"Follow {escape_markup(title)} in Console"),
            False,
            "Open the latest active Watchlists run in Console.",
        )

    def _resolve_console_follow_drift(self) -> bool:
        """Re-poll the Console-follow adapter and say whether the rail is stale.

        **What this can and cannot detect** (review wave, M3 -- verified
        against `WatchlistsConsoleHandoff._latest_console_follow_item`,
        `watchlists_console_handoff.py:39-75`, not assumed):

        * A **successful** poll sets `_latest_console_follow_loaded = True`,
          and nothing ever resets it -- so after one success the adapter's
          answer is frozen for the life of this screen, and this method can
          only return `False`. That is not a regression: the full-screen
          recompose this replaces hit the identical cache. A live re-poll on
          every background load would be NEW behaviour -- and expensive, since
          the adapter fans out over watchlist-run, chatbook-artifact,
          ingest-job and notification queries plus a server-event fetch -- so
          it is deliberately not built here.
        * A **failed** poll does NOT set that flag, so failure is retried. That
          is the one real case this exists for, and it is the case the old
          recompose covered: an adapter that fails during the first compose
          renders `#watchlists-console-unavailable`, and the next background
          loader picks up the recovered answer. Pinned by
          `test_watchlists_destination_retries_console_follow_after_initial_adapter_failure`.

        Not a pure predicate despite the boolean return, which is why it is
        named for the action: it refreshes the handoff's cached item id (a
        deliberate side effect, "ahead of a render pass" -- see
        `resolve_latest_follow_item`) and mirrors the answer onto
        `_console_follow_item` for `_build_inspector_region` to render.

        Compares against the DOM rather than a remembered key, so there is no
        second copy of "what is currently rendered" to drift: the rendered
        answer IS the state. Because the only reachable transition is
        failure -> success, which changes the status widget's *id*, the
        `if not status_widgets` branch below is what actually fires; the two
        text comparisons after it are belt-and-braces for a future handoff
        whose cache does expire.

        Returns:
            True when the Inspector is mounted and its Console-follow row no
            longer matches the adapter, so the right rail should be rebuilt.
        """
        self._console_follow_item = self._console_handoff.resolve_latest_follow_item()
        status_id, status_copy, button_label, _disabled, _tooltip = (
            self._console_follow_copy(self._console_follow_item)
        )
        try:
            button = self.query_one("#watchlists-follow-in-console", Button)
        except NoMatches:
            # Nothing rendered (right rail collapsed, or mid-rebuild): the
            # next build resolves it from scratch anyway.
            return False
        status_widgets = list(self.query(f"#{status_id}"))
        if not status_widgets:
            return True
        if str(status_widgets[0].renderable) != str(status_copy):
            return True
        return str(button.label) != str(button_label)

    def _build_inspector_region(self) -> Vertical:
        """The RIGHT_RAIL content factory.

        Reads `_console_follow_item` -- a plain attribute -- rather than
        polling the adapter (review wave, M4). The workbench calls this
        factory again on every rebuild of the RIGHT_RAIL region, so a `]`
        (or a chevron on the Inspector) would otherwise re-run the adapter's
        multi-query fan-out from inside `compose()`. The
        handoff's cache makes that free once a poll has SUCCEEDED, but on the
        failure path it is retried every time -- exactly the shape this file
        insists its content factories must not have.

        The attribute is refreshed in the two places a fresh answer is
        actually wanted: once per compose pass (`compose_content`, restoring
        the pre-TASK-2200 call site) and by `_resolve_console_follow_drift`
        immediately before it asks for this rebuild -- so the targeted rail
        rebuild still repaints the staleness it was asked to fix.
        """
        attach_disabled, attach_tooltip = self._wc_attach_state()
        return self._build_inspector_pane(
            self._console_follow_item,
            attach_disabled,
            attach_tooltip,
        )

    def _build_inspector_pane(
        self,
        latest_console_item: Any,
        attach_disabled: bool,
        attach_tooltip: str,
    ) -> Vertical:
        """Build the RIGHT_RAIL-region content: state summaries, the
        entity Inspector, and Console actions -- in that order.

        `latest_console_item`/`attach_disabled`/`attach_tooltip` are captured
        once per `compose_content` call and passed in rather than
        recomputed, since a factory wrapping this method (see
        `compose_content`) is called on every region rebuild.

        TASK-2313, AC#5: Console actions used to sit BETWEEN the state
        summaries and the entity Inspector, so the thing the user actually
        selected -- and the actions that act on it -- were permanently
        below a block of global chrome unrelated to that selection (UAT
        finding). The entity Inspector (built first, below, so its own
        selection-derived state exists before it is placed) now comes
        right after the brief state summaries; Console actions -- a
        cross-cutting action unrelated to whatever is currently selected --
        moves to the bottom.
        """
        # Seed from screen state (Finding 3, fix round 2): collapsing or
        # expanding the right rail constructs a brand new InspectorPane, and
        # so does the drain's conditional rail rebuild
        # (`_rebuild_inspector_if_console_follow_drifted`). Without this,
        # the screen keeps
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
        # TASK-998, widened by TASK-1020: same seeding rationale as the three
        # lines above -- and the Inspector cannot work this out for itself,
        # since it is handed a selection rather than the data behind it. The
        # value is the same one the Overview region keys off, so the rail's
        # first-run text and the region's can never disagree.
        inspector.profile_state = self._watchlists_profile_state()
        # Review wave, I1: the same seeding rationale again, for the same
        # value the rail is handed in `_build_tree_pane`. The Inspector's
        # `Add existing` is the watchlist-side twin of the rail's, so the two
        # must be enabled and disabled by one condition, not two.
        inspector.write_disabled_reason = self._tree_write_disabled_reason()
        # TASK-2309: same rebuild-survival seeding as `SourcesPane.busy_
        # source_ids` in `_build_detail_pane`, and for the identical reason
        # -- this factory builds a brand new `InspectorPane` on every region
        # rebuild.
        inspector.busy_source_ids = frozenset(self._checks_in_flight)

        children: list[Widget] = [
            Static(
                "Inspector",
                classes="destination-section watchlists-column-title",
            ),
            Static(
                self._watchlists_state_summary_text(),
                id="watchlists-state-summary",
            ),
            Static(
                f"Alert rules active: {self.overview_data.get('active_alert_rules', 0)}",
                id="watchlists-alerts-summary",
            ),
            Static(
                self._latest_run_status_text(),
                id="watchlists-latest-run-summary",
            ),
            inspector,
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
        status_id, status_copy, button_label, button_disabled, button_tooltip = (
            self._console_follow_copy(latest_console_item)
        )
        children.append(Static(status_copy, id=status_id))
        children.append(
            Button(
                button_label,
                id="watchlists-follow-in-console",
                disabled=button_disabled,
                tooltip=button_tooltip,
            )
        )
        return Vertical(
            *children,
            id="watchlists-inspector-pane",
            classes="destination-workbench-pane ds-inspector",
        )

    #: Sections whose data has no server half at all, and the two pieces of
    #: header copy that have to say so. The Backend selector is disabled on
    #: these, because offering a choice that changes nothing is a lie about
    #: where the rows come from. Notifications established the pattern;
    #: Artifacts joins it -- briefings are written to, and read from, this
    #: device's `SubscriptionsDB` whatever the selector says.
    _LOCAL_ONLY_SECTIONS: dict[str, dict[str, str]] = {
        "items": {
            "label": "Read: local",
            "tooltip": (
                "Read and its permanent Reader use items stored on this device. "
                "Switch to Local to load them."
            ),
        },
        "notifications": {
            "label": "Inbox: local",
            "tooltip": "The notifications inbox is local to this device.",
        },
        "artifacts": {
            "label": "Artifacts: local",
            "tooltip": (
                "Briefings are written to and read from this device's "
                "watchlist store."
            ),
        },
    }

    def _watchlists_state_summary_text(self) -> str:
        """TASK-2313, AC#2: state vocabulary, not fault vocabulary.

        UAT: "State: ready" read as "ready for what?" -- an unexplained
        state name with no referent. "State: unavailable" was worse: it
        conflated two genuinely different conditions (the snapshot is
        still LOADING, versus a real lookup ERROR) under one word that
        reads as a fault either way, so a normal, brief loading flicker on
        startup looked identical to a real problem. Three real states now.
        """
        if not self._wc_loaded:
            return "Watchlists: loading…"
        if self._wc_lookup_error:
            return "Watchlists: error"
        return "Watchlists: loaded"

    def _latest_run_status_text(self) -> str:
        """TASK-2313, AC#2: "no runs yet" is a state; a bare "unavailable"
        reads as a fault for the same reason `_watchlists_state_summary_
        text` was corrected -- there is nothing unavailable about a
        watchlist that has simply never been checked yet.

        Review finding I1: that fix originally collapsed a THIRD, distinct
        condition -- `scope_service` itself not being wired up -- into the
        same `None`/"no runs yet" text, so "the feature isn't connected"
        silently read as "this watchlist is healthy and hasn't run." Both
        of the honest-but-degraded sentinels
        (`WatchlistsBackendController.NOT_CONFIGURED_STATUS` and
        `.LOOKUP_FAILED_STATUS`) get their own text here, checked before
        the generic `not status` fallback so neither is ever mistaken for
        "no runs yet" or folded into the free-text branch below (which
        exists for real DB-sourced run statuses like "completed"/"failed").
        """
        status = self.overview_data.get("latest_run_status")
        if status == WatchlistsBackendController.NOT_CONFIGURED_STATUS:
            return "Latest run status: not connected"
        if status == WatchlistsBackendController.LOOKUP_FAILED_STATUS:
            return "Latest run status: couldn't check"
        if not status:
            return "Latest run status: no runs yet"
        return f"Latest run status: {status}"

    def _local_only_section(self) -> dict[str, str] | None:
        """Header copy for the active section if it has no server half."""
        return self._LOCAL_ONLY_SECTIONS.get(self.active_section)

    def _backend_label_text(self) -> str | None:
        """What the header bar says about where this section's rows live.

        TASK-2313, AC#3 (duplicate affordances): the Select's own current
        value already reads "Local"/"Server" -- a trailing "Backend:
        local" Static right beside it restated the identical fact a
        SECOND time for the one case where the Select is actually a live
        choice. `None` here (compose_content renders the inline "Backend"
        label -- the 2310 idiom -- ahead of the Select instead, naming the
        axis without repeating its value) drops that redundant copy. The
        LOCAL-ONLY sections keep their own reason text: "Artifacts: local"
        is not a restatement of the Select's value, it explains why the
        Select is DISABLED regardless of what it shows -- genuinely new
        information the Select cannot carry on its own.
        """
        local_only = self._local_only_section()
        return local_only["label"] if local_only is not None else None

    def compose_content(self) -> ComposeResult:
        # Resolved once per compose pass, as it was before TASK-2200 -- but
        # mirrored onto the screen instead of captured in the RIGHT_RAIL
        # factory's closure, so a region rebuild reads an attribute rather
        # than re-running the adapter (review wave, M4). See
        # `_build_inspector_region`.
        self._console_follow_item = self._console_handoff.resolve_latest_follow_item()
        with Vertical(id="watchlists-collections-shell"):
            # TASK-2313, AC#6: "Mixed | Local/Server" was a hardcoded
            # constant -- it never reflected `self.runtime_backend` and
            # read as cryptic, undiscoverable jargon in the UAT (what is
            # "Mixed"? mixed with what?). The row directly below already
            # names the backend precisely, live, via the labeled Select
            # (task-2310/2313's own "Backend" label fix) -- dropped rather
            # than duplicated.
            yield Static(
                "Watchlists | Monitored sources, runs, alerts, recovery",
                id="watchlists-collections-title",
                classes="ds-destination-header",
            )
            with Horizontal(id="watchlists-header-bar", classes="destination-filter-strip"):
                # TASK-995: `compact=True` for the same reason as the
                # Sources/Items toolbars -- `.destination-filter-strip` is
                # `height: 1` and a bordered Select is three rows, so this
                # backend picker was painting its top border and nothing
                # else. See `sources_pane.compose()`.
                #
                # TASK-2313, AC#3: a "Backend" label (the 2310 idiom) names
                # the axis ahead of the Select instead of a trailing
                # "Backend: local" Static restating the Select's own
                # current value -- see `_backend_label_text`'s docstring.
                yield Static("Backend", classes="watchlists-inline-select-label")
                yield PruneSafeSelect(
                    [("Local", "local"), ("Server", "server")],
                    value=self.runtime_backend,
                    id="watchlists-backend-select",
                    allow_blank=False,
                    compact=True,
                    disabled=self._local_only_section() is not None,
                    tooltip=(
                        self._local_only_section() or {}
                    ).get("tooltip")
                    or "Choose the Watchlists data backend.",
                )
                backend_label_text = self._backend_label_text()
                if backend_label_text is not None:
                    yield Static(
                        backend_label_text,
                        id="watchlists-backend-label",
                    )
            yield WatchlistsWorkbench(
                self._effective_region_layout,
                content={
                    # Factories, not instances: a region whose rendered form
                    # changes (collapse/expand, and for ITEMS a section
                    # switch) is swapped for a freshly built widget, so each
                    # of these is called more than once.
                    # A pre-built container's constructor-supplied children
                    # only mount on its FIRST mount; the same instance
                    # remounted a second time comes back childless (verified
                    # empirically — see `WatchlistsWorkbench.__init__`).
                    Region.LEFT_RAIL: self._build_tree_pane,
                    Region.ITEMS: self._build_detail_pane,
                    Region.CONTENT: self._build_content_pane,
                    Region.RIGHT_RAIL: self._build_inspector_region,
                },
                # Unconditional since task-2513: the tab strip and the
                # snapshot markers are cross-cutting chrome carried by the
                # centre header on every tab, Read included -- see
                # `_build_centre_status_header`. (They used to ride inside
                # the FEEDS region's own body on Read; that region is gone.)
                header=self._build_centre_status_header,
                read_mode=self.active_section == "items",
                id="wl-workbench",
                classes=(
                    "watchlists-read-mode"
                    if self.active_section == "items"
                    else ""
                ),
            )

    def _available_layout_width(self) -> int | None:
        """Return positive screen allocation, never descendant content width."""
        width = self.size.width
        return width if width > 0 else None

    def _next_layout_request_token(self) -> int:
        """Allocate the one current controller/workbench request token."""
        self._layout_request_generation += 1
        self._current_layout_request_token = self._layout_request_generation
        rollback = self._manual_layout_rollback
        if (
            rollback is not None
            and self.region_layout == rollback.attempted_preferred
        ):
            self._manual_layout_rollback = ManualLayoutRollback(
                token=self._current_layout_request_token,
                attempted_layout=self._effective_region_layout,
                attempted_preferred=rollback.attempted_preferred,
                preferred_before=rollback.preferred_before,
                effective_before=rollback.effective_before,
                responsive_before=rollback.responsive_before,
                article_focus_before=rollback.article_focus_before,
                priority_lease_before=rollback.priority_lease_before,
            )
        return self._current_layout_request_token

    def _recompute_effective_layout(
        self,
        *,
        cause: LayoutRecomputeCause,
        section: str | None = None,
        request_workbench: bool = True,
        previous: RegionLayout | None = None,
    ) -> int | None:
        """Resolve and push transient responsive/Article Focus state."""
        width = self._available_layout_width()
        if width is None:
            return None

        section = self.active_section if section is None else section
        read_mode = section == "items"
        mounted = READ_SIDE_PANE_ORDER if read_mode else MANAGEMENT_SIDE_PANE_ORDER
        responsive = self._responsive_region_layout
        if cause != "article_focus" or responsive is None:
            previous = responsive if cause == "resize" else None
            lease = self._responsive_priority_lease
            priority_target = (
                lease.target
                if lease is not None and lease.read_mode == read_mode
                else None
            )
            if (
                cause == "resize"
                and priority_target is not None
                and not self._article_focus_active
            ):
                unprioritized_previous = previous
                if previous is not None:
                    # The explicit open placed the leased target in responsive
                    # history. Re-collapse only that target for the expiry
                    # probe so the same dead-band width cannot immediately
                    # clear the lease it just created.
                    unprioritized_previous = RegionLayout(
                        collapsed=previous.collapsed.union({priority_target})
                    )
                unprioritized = resolve_effective_layout(
                    self.region_layout,
                    width=width,
                    read_mode=read_mode,
                    article_focus=False,
                    priority_target=None,
                    previous=unprioritized_previous,
                )
                preferred_mounted = frozenset(
                    self.region_layout.collapsed.intersection(mounted)
                )
                if unprioritized.collapsed == preferred_mounted:
                    self._responsive_priority_lease = None
                    priority_target = None

            responsive = resolve_effective_layout(
                self.region_layout,
                width=width,
                read_mode=read_mode,
                article_focus=False,
                priority_target=priority_target,
                previous=previous,
            )
            self._responsive_region_layout = responsive

        effective = responsive
        if self._article_focus_active:
            effective = RegionLayout(
                collapsed=responsive.collapsed.union(mounted)
            )

        previous = self._effective_region_layout
        if effective == previous:
            return None
        if (
            read_mode
            and not previous.is_collapsed(Region.ITEMS)
            and effective.is_collapsed(Region.ITEMS)
        ):
            self._capture_items_view_state()
        self._effective_region_layout = effective
        if not request_workbench:
            return None
        try:
            workbench = self.query_one(WatchlistsWorkbench)
            if workbench.read_mode == read_mode:
                token = self._next_layout_request_token()
                workbench.request_region_layout(effective, token=token)
                return token
        except Exception:
            logger.debug("Workbench not mounted yet; layout applies on compose.")
        return None

    def _capture_items_view_state(self) -> None:
        try:
            pane = self.query_one("#watchlists-items-pane", ArticleListPane)
            table = pane.query_one("#items-table")
        except NoMatches:
            return
        highlighted = getattr(table, "highlighted_child", None)
        self._items_view_anchor_id = getattr(highlighted, "item_id_key", None)
        self._items_view_scroll_y = float(getattr(table, "scroll_y", 0.0))
        self._items_view_had_focus = bool(table.has_focus)
        self._items_view_focus_id = None
        focused = self.focused
        while focused is not None and focused is not pane:
            focused_id = getattr(focused, "id", None)
            if focused_id:
                self._items_view_focus_id = focused_id
                break
            focused = focused.parent
        snapshot = self._items_snapshot
        self._items_view_context_key = (
            snapshot.query.context_key if snapshot is not None else None
        )

    def _discard_items_view_state(self) -> None:
        """Discard one consumed or invalidated Items restoration snapshot."""
        self._items_view_anchor_id = None
        self._items_view_focus_id = None
        self._items_view_context_key = None
        self._items_view_had_focus = False

    def _restore_items_view_state(self) -> None:
        if self._items_view_context_key is None:
            return
        if self._items_page_loading:
            return
        snapshot = self._items_snapshot
        if (
            snapshot is None
            or self._items_view_context_key != snapshot.query.context_key
        ):
            self._discard_items_view_state()
            return
        try:
            pane = self.query_one("#watchlists-items-pane", ArticleListPane)
            table = pane.query_one("#items-table")
        except NoMatches:
            return
        if self._items_view_anchor_id is not None:
            for index, row in enumerate(table.children):
                if getattr(row, "item_id_key", None) == self._items_view_anchor_id:
                    pane._suppressed_highlight_item_id = self._items_view_anchor_id
                    table.index = index
                    break
        table.scroll_to(y=self._items_view_scroll_y, animate=False)
        focus_id = self._items_view_focus_id
        restored_focus = False
        if focus_id is not None:
            try:
                pane.query_one(f"#{focus_id}").focus()
                restored_focus = True
            except NoMatches:
                pass
        if not restored_focus and (
            self._items_view_had_focus or self._items_view_anchor_id is not None
        ):
            table.focus()
        elif not restored_focus:
            pane.focus()
        self._discard_items_view_state()

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
        self._recompute_effective_layout(cause="explicit")
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
            layout: The preferred side-pane layout to write when it differs
                from what is already persisted.
        """
        collapsed = layout.collapsed
        if (
            collapsed == self._last_persisted_collapsed
            and self._pending_persist_layout is None
        ):
            return
        with self._layout_persist_lock:
            self._layout_persist_generation += 1
            self._pending_persist_generation = self._layout_persist_generation
            self._pending_persist_layout = layout
            if self._layout_persist_draining:
                return
            self._layout_persist_draining = True
        try:
            self.run_worker(
                self._persist_layout_worker,
                group="wl-layout-persist",
                thread=True,
            )
        except Exception:
            with self._layout_persist_lock:
                self._layout_persist_draining = False
            logger.opt(exception=True).debug(
                "Could not schedule preferred Watchlists layout persistence."
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
        while True:
            with self._layout_persist_lock:
                generation = self._pending_persist_generation
                layout = self._pending_persist_layout
                if generation is None or layout is None:
                    self._layout_persist_draining = False
                    return
            try:
                success = save_region_layout(layout)
            except Exception:
                logger.opt(exception=True).debug(
                    "Failed to persist preferred Watchlists pane layout."
                )
                success = False
            try:
                self.app.call_from_thread(
                    self._acknowledge_layout_persist,
                    generation,
                    layout,
                    success,
                )
            except Exception:
                logger.opt(exception=True).debug(
                    "Could not acknowledge preferred Watchlists layout write."
                )
                with self._layout_persist_lock:
                    if self._pending_persist_generation != generation:
                        continue
                    self._layout_persist_draining = False
                    return
            with self._layout_persist_lock:
                pending_generation = self._pending_persist_generation
                if pending_generation is None or pending_generation == generation:
                    self._layout_persist_draining = False
                    return

    def _acknowledge_layout_persist(
        self,
        generation: int,
        layout: RegionLayout,
        success: bool,
    ) -> None:
        """Commit only the current generation's successful write."""
        with self._layout_persist_lock:
            if generation != self._pending_persist_generation:
                return
            if success:
                self._last_persisted_collapsed = layout.collapsed
                self._pending_persist_layout = None
                self._pending_persist_generation = None

    def _refuse_region_gesture_off_read_tab(self, region: Region) -> bool:
        """Refuse a layout change aimed at a centre region off the Read tab.

        Generalized from `_refuse_content_toggle_off_read_tab` (TASK-1344
        AC#2): that name special-cased `Region.CONTENT`, which was the only
        gated region when it was written. A stray `focused_region` left over
        from the Read tab (e.g. the user last touched CONTENT there, then
        switched to Sources without moving focus) must be refused the same
        way -- this is the ONE place both `action_toggle_region` and
        `_on_region_toggled` consult, per the
        prompt's "one source of truth for is region R visible on section S".

        Named as an ACTION, not a predicate (it was `_content_toggle_is_blocked`
        before TASK-1349), because it is NOT pure: when it refuses it calls
        `self.notify(...)`. A side-effecting predicate is safe only until
        someone wires it into a render path -- this codebase already shipped
        `provider_is_configured()` writing an `eval_models` row from
        `compose()`, so opening the Evals screen mutated the DB on every fresh
        install. The verb in the name is the warning that innocent name never
        gave; keep it a verb if the notify stays here.

        Whole-branch review (Important, Task 4): off the Read (Items) tab, a
        hidden region used to still render a real, focusable `▸ <Region>`
        header button (the AC#4-era design). Clicking it -- or pressing `z`
        with it focused -- ran the toggle against the REAL `region_layout`,
        not the derived view the user was actually looking at. So the click
        did nothing visible, silently flipped the user's genuine preference
        to collapsed, and `_schedule_layout_persist` wrote it to disk, honored
        forever. TASK-1344 AC#4 now unmounts hidden regions outright rather
        than rendering that header, which
        removes the click/chevron route entirely -- but `focused_region` is a
        screen-level reactive that outlives the widget that last set it
        (`on_descendant_focus`), so `z`/`Z` can still be invoked with it
        pointed at a region that is not currently mounted at all. This
        refusal is what stops that stale reference from mutating the real
        layout blind.

        Also gates SOLO (PR #1091 review, F2 / TASK-1344 AC#2). `Z` on a
        stale `focused_region` would otherwise collapse the OTHER centre
        regions around one the user cannot see on this tab, the same class
        of harm as the chevron, through the one route that was still open.

        Whole-branch review round 2 (task-1344 review, B1): ITEMS is never
        hidden, always the section's own full-width pane off Read, and was
        once never refused. But the derived layout only forces ITEMS open for
        the RENDER; the gesture handlers above still call `_apply_layout`
        against the real, persisted layout. So an off-Read `z`/`Z` with
        `focused_region == ITEMS` (reachable any time focus lands inside the
        section pane -- `on_descendant_focus` sets exactly that while using
        Sources/Runs/...) toggled and PERSISTED a real ITEMS collapse with
        zero visible feedback on the current tab, and -- combined with
        CONTENT already collapsed on Read -- returning to Read rendered
        headers over an empty centre: the exact dead-end AC#3 exists to
        rule out, written to disk, surviving a restart.
        Region-layout gestures for the Read centre
        simply do not apply to ANY centre region off Read, not just the
        ones that happen to be unmounted there, so the gate now refuses
        ITEMS off Read unconditionally. The
        notify copy forks on whether the region is actually hidden here:
        CONTENT keeps the "only shown on the Read tab" copy (true for
        it), while ITEMS -- visible, just not collapsible from this tab
        -- gets copy that says so honestly instead of claiming it is not
        shown at all.

        Args:
            region: The region the user's gesture targets.

        Returns:
            `True` when the gesture must be refused (and the user has been
            told why), `False` when it may proceed.
        """
        if region is not Region.ITEMS or self.active_section == "items":
            return False
        self.notify(
            "Feed Items can only be collapsed on the Read tab.",
            markup=False,
        )
        return True

    def action_toggle_region(self) -> None:
        """Collapse or expand whichever region currently has focus.

        Silently refused while focus sits in the centre header/tab strip
        (`_focus_in_centre_header`, task-1344 fix wave, Qodo correctness):
        `focused_region` there names wherever the user last actually
        visited, not where they are now, and a rail (LEFT_RAIL/RIGHT_RAIL)
        is never gated by `_refuse_region_gesture_off_read_tab` below, so
        without this check a stale
        `focused_region` pointing at a rail would still collapse -- and
        persist -- it from a keypress that has no visible relationship to
        either the rail or the tab strip the user is actually looking at.
        """
        node = self.focused
        region: Region | None = None
        while node is not None:
            node_id = getattr(node, "id", None) or ""
            for prefix in ("wl-region-", "wl-grip-"):
                if node_id.startswith(prefix):
                    try:
                        region = Region(node_id.removeprefix(prefix))
                    except ValueError:
                        region = None
                    break
            if region is not None:
                break
            node = node.parent
        if region not in COLLAPSIBLE_REGIONS:
            return
        if self._refuse_region_gesture_off_read_tab(region):
            return
        self._toggle_preferred_region(region)

    def action_article_focus(self) -> None:
        """Toggle transient Article Focus on Read."""
        if self.active_section != "items":
            self.notify("Article Focus is available on Read.", markup=False)
            return
        self._article_focus_active = not self._article_focus_active
        self._recompute_effective_layout(cause="article_focus")

    def action_toggle_left_rail(self) -> None:
        self._toggle_preferred_region(Region.LEFT_RAIL)

    def action_toggle_right_rail(self) -> None:
        self._toggle_preferred_region(Region.RIGHT_RAIL)

    def _toggle_preferred_region(self, region: Region) -> None:
        """Apply one manual gesture, inferred from effective state."""
        requested_open = self._effective_region_layout.is_collapsed(region)
        self._manual_layout_rollback = None
        before = (
            self.region_layout,
            self._effective_region_layout,
            self._responsive_region_layout,
            self._article_focus_active,
            self._responsive_priority_lease,
        )
        self._article_focus_active = False
        read_mode = self.active_section == "items"
        preferred = self.region_layout
        if requested_open:
            if preferred.is_collapsed(region):
                preferred = preferred.toggle_preferred(region)
            self._responsive_priority_lease = ResponsivePriorityLease(
                target=region,
                read_mode=read_mode,
            )
        else:
            if not preferred.is_collapsed(region):
                preferred = preferred.toggle_preferred(region)
            lease = self._responsive_priority_lease
            if (
                lease is not None
                and lease.target is region
                and lease.read_mode == read_mode
            ):
                self._responsive_priority_lease = None
        self.region_layout = preferred
        token = self._recompute_effective_layout(cause="explicit")
        if token is not None:
            self._manual_layout_rollback = ManualLayoutRollback(
                token=token,
                attempted_layout=self._effective_region_layout,
                attempted_preferred=preferred,
                preferred_before=before[0],
                effective_before=before[1],
                responsive_before=before[2],
                article_focus_before=before[3],
                priority_lease_before=before[4],
            )
        self._schedule_layout_persist(preferred)

    @on(RegionToggled)
    def _on_region_toggled(self, event: RegionToggled) -> None:
        event.stop()
        if event.region not in COLLAPSIBLE_REGIONS:
            return
        if self._refuse_region_gesture_off_read_tab(event.region):
            return
        self._toggle_preferred_region(event.region)

    @on(RegionLayoutApplyFailed)
    def _on_region_layout_apply_failed(
        self, event: RegionLayoutApplyFailed
    ) -> None:
        """Correct screen preference only while the failed intent is current."""
        event.stop()
        if event.token != self._current_layout_request_token:
            return
        rollback = self._manual_layout_rollback
        if rollback is not None and event.token == rollback.token:
            current_preferred = self.region_layout
            self.region_layout = rollback.preferred_before
            self._article_focus_active = rollback.article_focus_before
            self._responsive_priority_lease = rollback.priority_lease_before
            self._responsive_region_layout = rollback.responsive_before
            self._effective_region_layout = event.fallback
            self._manual_layout_rollback = None
            if current_preferred != rollback.preferred_before:
                self._schedule_layout_persist(rollback.preferred_before)
            return
        if rollback is None:
            self._effective_region_layout = event.fallback
            return

    @on(RegionLayoutApplied)
    def _on_region_layout_applied(self, event: RegionLayoutApplied) -> None:
        """Restore pane-local view state after a successful remount."""
        event.stop()
        if event.token != self._current_layout_request_token:
            return
        if (
            self._manual_layout_rollback is not None
            and event.token == self._manual_layout_rollback.token
        ):
            self._manual_layout_rollback = None
        if (
            event.previous.is_collapsed(Region.ITEMS)
            and not event.layout.is_collapsed(Region.ITEMS)
        ):
            self._restore_items_view_state()

    def _apply_tree_scope(self, scope: TreeScope) -> None:
        """The single reconciliation point for "the tree scope is now `scope`".

        Read navigation reaches this only after its first page mounts;
        management navigation may commit immediately. Both tree clicks and
        breadcrumb promotion enter through `_request_tree_scope` so an
        attempted Read scope cannot relabel the committed Reader early.

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
        self._clear_pane_selections()
        self.selected_scope = scope
        self.tree_scope = scope

    def _scope_display_label(self, scope: TreeScope) -> str:
        """Return the unescaped user-facing label for an explicit scope."""
        if scope.kind == "starred":
            return "Starred"
        if scope.kind == "unread":
            return "All Unread"
        if scope.kind == "today":
            return "Today"
        if scope.kind == "unassigned":
            return "Unassigned"
        if scope.kind == "watchlist" and scope.watchlist_id is not None:
            return self._watchlist_display_name(scope.watchlist_id)
        if scope.kind == "source" and scope.source_id is not None:
            labels = self._resolve_breadcrumb_labels(scope)
            if len(labels) == 2:
                return f"{labels[1]} under {labels[0]}"
            return labels[-1] if labels else f"Source {scope.source_id}"
        return "All Sources"

    def _notify_pending_scope_failure(self, attempted: TreeScope) -> None:
        """Explain a failed navigation without relabelling committed rows."""
        self._notify_watchlists(
            f"Couldn't open {self._scope_display_label(attempted)}; still showing "
            f"{self._scope_display_label(self.tree_scope)}.",
            severity="error",
            markup=False,
        )

    def _show_items_retry_state(self) -> None:
        """Replace an empty returned Reader with an honest retry surface."""
        self._items_retry_message = (
            f"Couldn't load {self._scope_display_label(self.tree_scope)}. "
            "Retry to load Feed Items."
        )
        self._request_surface_refresh(self._SURFACE_SECTION)

    def _invalidate_parked_reader(self, *, loading: bool) -> None:
        """Drop every Reader authority after an immediate management move."""
        self._items_retry_message = None
        self._items_retry_inflight = False
        self._reset_items_paging_for_context(loading=loading)
        self._items_snapshot = None
        self._loaded_items = []
        self._items_snapshot_count = 0
        self._items_pending_arrivals = 0
        self._selected_content_item = None
        self._selected_content_page_key = None
        try:
            pane = self.query_one("#watchlists-items-pane", ArticleListPane)
        except NoMatches:
            pass
        else:
            pane.items = []
            pane.selected_item = None
            pane.show_new_items_pill(0)
            self._push_items_pager_state()
        try:
            content = self.query_one("#watchlists-content-pane", ContentPane)
        except NoMatches:
            pass
        else:
            content.item = None
            content.position = ""

    def _commit_management_tree_scope(self, scope: TreeScope) -> None:
        """Commit non-Read navigation and invalidate parked Reader state."""
        with self.app.batch_update():
            self._pending_tree_scope = None
            self._apply_tree_scope(scope)
            self._invalidate_parked_reader(loading=False)

    def _request_tree_scope(self, scope: TreeScope) -> None:
        """Request Read navigation, or commit management navigation now."""
        if self.active_section == "items" and self.runtime_backend == "local":
            self._pending_tree_scope = scope
            self._supersede_items_query_intent(scope=scope)
            try:
                self.query_one("#wl-tree", WatchlistTree).active_scope = (
                    self.tree_scope
                )
            except NoMatches:
                pass
            self.run_worker(
                self._replace_items_snapshot(
                    scope=scope,
                    reason="scope",
                    clear_reader_on_commit=True,
                ),
                exclusive=True,
                group="wc_items",
            )
            return
        self._commit_management_tree_scope(scope)

    def _clear_pane_selections(self) -> None:
        """Clear the mounted panes' OWN selection copies, not just the
        screen's mirrors of them (whole-branch review, Finding 1).

        The three mirrors `_apply_tree_scope` clears above live on the
        screen; each pane keeps a second copy of the same selection so a
        highlighted row survives that pane's own reactive-recompose. Clearing
        only the screen half leaves the pane half live, and a pane's live
        selection is not inert:

        * `RunsPane.run_poll` re-posts `RunSelected(self.selected_run)` once a
          second for sixty ticks while that run's status is `running`. A tick
          landing after a tree move re-selects the pre-move run, which routes
          through `_select_entity` and snaps `selected_scope` back to "all"
          with an empty breadcrumb -- with no user action, and again every
          second, so the user cannot hold the new scope at all.
        * `SourcesPane`'s Preview/Check-now and `RunsPane`'s Cancel/Re-run
          stay armed against the pre-move row and post messages naming a
          selection the screen believes is gone.

        Setting each pane's reactive to `None` fixes both: the poll's own
        `current is None` guard returns on its next tick, and each pane's
        watcher disarms its own buttons.

        Degrades quietly when a pane is absent -- only the active section's
        pane is mounted at all, the workbench recomposes, and regions
        collapse -- matching how the rest of this screen reaches its panes.
        """
        for selector, pane_type, attribute in (
            ("#watchlists-sources-pane", SourcesPane, "selected_source"),
            ("#watchlists-runs-pane", RunsPane, "selected_run"),
            (
                "#watchlists-notifications-pane",
                NotificationsPane,
                "selected_notification",
            ),
        ):
            try:
                setattr(self.query_one(selector, pane_type), attribute, None)
            except Exception:
                continue

    @on(TreeExpansionChanged)
    def handle_tree_expansion_changed(self, event: TreeExpansionChanged) -> None:
        """Mirror the rail's expansion onto the screen (Finding 2).

        See the two `_tree_expanded_*` fields in `__init__` for why this
        cannot live on the tree widget, and `_build_tree_pane` for where both
        independent sets are seeded back.
        """
        event.stop()
        self._tree_expanded_root_kinds = event.expanded_root_kinds
        self._tree_expanded_watchlist_ids = event.expanded_watchlist_ids

    @on(TreeTagFilterChanged)
    def handle_tree_tag_filter_changed(self, event: TreeTagFilterChanged) -> None:
        """Mirror the rail's tag filter onto the screen (Finding 2)."""
        event.stop()
        self._tree_active_tag = event.tag

    @on(TreeScopeChanged)
    def _on_tree_scope_changed(self, event: TreeScopeChanged) -> None:
        """Store the tree's selection on the screen, not the tree.

        `selected_scope` lives here for the same reason `selected_run` and
        the create-form draft do: a bare rail toggle swaps in a brand new
        `WatchlistTree` that would otherwise lose the selection.
        """
        event.stop()
        self._request_tree_scope(event.scope)

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
        self._request_tree_scope(event.scope)

    # --- task-895: the tree's write verbs -------------------------------
    #
    # Five `WatchlistBundleService` methods (`create`, `rename`, `delete`,
    # `add_source`, `remove_source`) had no production caller: Phase C
    # shipped the tree's read half only, so watchlists could be browsed but
    # not made. Each verb follows the same three-step shape the rest of this
    # screen already uses for a user-initiated write -- a handler that only
    # starts a worker, a worker that owns the dialog + service call, and a
    # `_load_tree_data()` reload so the rail shows the result without the
    # user refreshing anything.
    #
    # The dialogs are awaited (`push_screen_wait`) rather than driven by
    # `push_screen(..., callback=...)`: an add-source flow needs the picked
    # id *before* it can call the service, and the sequential form keeps
    # "prompt, then write, then reload" readable as one function. That is
    # only legal inside a worker, which is why every handler defers.

    def _notify_watchlists(
        self, message: str, severity: str = "information", *, markup: bool = True
    ) -> None:
        """Notify through the app instance, degrading when it has none.

        Matches the `getattr(self.app_instance, "notify", None)` idiom every
        other action on this screen uses -- the app instance is a stub in
        several harnesses.

        Args:
            message: The toast body.
            severity: Textual severity level.
            markup: Textual renders toast bodies as Rich markup by default.
                Callers whose message can contain content this app did not
                author -- a watchlist name, a provider's error text -- pass
                `False` so a bracket-shaped fragment paints instead of being
                interpreted (or swallowed as an unclosed tag).
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity=severity, markup=markup)

    def _watchlist_display_name(self, watchlist_id: int) -> str:
        """A watchlist's name from data already loaded, RAW (unescaped).

        Resolved from `_tree_watchlists` for the same reason
        `_tree_scope_label` does: `_load_tree_data` has already paid for that
        list, and a display name is not worth a second query. The result is
        user-authored free text, so every caller must escape it before it
        reaches a rendered label or a notification (Textual renders toast
        messages as markup).
        """
        return next(
            (
                str(watchlist.get("name"))
                for watchlist in self._tree_watchlists
                if int(watchlist.get("id", -1)) == int(watchlist_id)
            ),
            f"Watchlist {watchlist_id}",
        )

    def _start_tree_write(self, flow_factory: Any) -> None:
        """Run one tree write at a time, in a worker.

        Args:
            flow_factory: Zero-argument callable returning the flow
                coroutine. A callable rather than a coroutine so the
                already-in-flight branch does not have to discard an
                un-awaited coroutine.

        A plain guard rather than `run_worker(exclusive=True)`: exclusive
        cancels the *previous* worker, and these workers own a modal dialog
        -- cancelling one mid-prompt would leave its dialog on the screen
        stack with nothing left to dismiss it.
        """
        if self._tree_write_active:
            return
        # Build and schedule BEFORE arming the guard. `_run_tree_write`'s
        # `finally` is the only thing that lowers this flag, and it never runs
        # if `flow_factory()` or `run_worker` raises synchronously -- which
        # would leave the flag stuck True and silently swallow every later
        # create/rename/delete for the life of the screen.
        try:
            worker_coro = self._run_tree_write(flow_factory())
        except Exception:
            logger.opt(exception=True).warning("Watchlist tree write could not start.")
            self._notify_watchlists(
                "That watchlist action could not be started.", severity="error"
            )
            return
        # Arm before scheduling, and disarm if scheduling fails. Arming
        # afterwards would be its own race: the worker's `finally` could
        # already have lowered the flag by the time we raised it, leaving it
        # stuck True with nothing running.
        self._tree_write_active = True
        try:
            self.run_worker(worker_coro, group="wl-tree-write")
        except Exception:
            self._tree_write_active = False
            worker_coro.close()
            logger.opt(exception=True).warning("Watchlist tree write could not start.")
            self._notify_watchlists(
                "That watchlist action could not be started.", severity="error"
            )

    async def _run_tree_write(self, flow: Any) -> None:
        """Await one write flow, reporting rather than raising.

        The flows call the service directly, so a `sqlite3` error, a
        `KeyError` from a watchlist deleted underneath the user, or a
        `ValueError` from a name that slipped past the dialog all surface
        here. Every other worker on this screen reports failures the same
        way; a raising worker would be swallowed into a log line the user
        never sees.
        """
        try:
            await flow
        except Exception:
            logger.opt(exception=True).warning("Watchlist tree write failed.")
            self._notify_watchlists(
                "That watchlist action could not be completed.", severity="error"
            )
        finally:
            self._tree_write_active = False

    async def _prompt_watchlist_name(
        self,
        *,
        dialog_title: str,
        submit_label: str,
        initial_name: str = "",
        exclude_id: int | None = None,
    ) -> str | None:
        """Ask for a watchlist name, or `None` when the user cancels.

        The dialog itself refuses an empty or duplicate name with a visible
        reason and stays open, so a non-`None` return is always a name the
        service will store as typed.

        Args:
            dialog_title: Heading for the dialog.
            submit_label: Label for the confirming button.
            initial_name: Value the input starts with.
            exclude_id: Watchlist to leave out of the duplicate check --
                the one being renamed, so re-submitting its own current
                name is a no-op rather than a reported collision, matching
                `WatchlistBundleService.rename`'s own `exclude_id`.
        """
        taken = [
            str(watchlist.get("name") or "")
            for watchlist in self._tree_watchlists
            if exclude_id is None or int(watchlist.get("id", -1)) != int(exclude_id)
        ]
        return await self.app.push_screen_wait(
            WatchlistNameDialog(
                dialog_title=dialog_title,
                submit_label=submit_label,
                initial_name=initial_name,
                taken_names=taken,
            )
        )

    @on(CreateWatchlistRequested)
    def handle_create_watchlist_requested(
        self, event: CreateWatchlistRequested
    ) -> None:
        event.stop()
        self._start_tree_write(self._create_watchlist_flow)

    async def _create_watchlist_flow(self) -> None:
        service = self._watchlist_bundle_service()
        if service is None:
            self._notify_watchlists(WC_SERVICE_UNAVAILABLE_COPY, severity="error")
            return
        name = await self._prompt_watchlist_name(
            dialog_title="New watchlist", submit_label="Create"
        )
        if name is None:
            return
        created = service.create(name)
        # Scope the tree to what was just made, so the rail's Rename/Delete/
        # Add-source verbs are armed on it immediately rather than requiring
        # a second click to select the thing the user just created.
        self._request_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=int(created["id"]))
        )
        self._notify_watchlists(
            f"Watchlist \"{escape_markup(str(created['name']))}\" created."
        )
        self._load_tree_data()

    @on(RenameWatchlistRequested)
    def handle_rename_watchlist_requested(
        self, event: RenameWatchlistRequested
    ) -> None:
        event.stop()
        watchlist_id = event.watchlist_id
        self._start_tree_write(lambda: self._rename_watchlist_flow(watchlist_id))

    async def _rename_watchlist_flow(self, watchlist_id: int) -> None:
        service = self._watchlist_bundle_service()
        if service is None:
            self._notify_watchlists(WC_SERVICE_UNAVAILABLE_COPY, severity="error")
            return
        current = self._watchlist_display_name(watchlist_id)
        name = await self._prompt_watchlist_name(
            dialog_title="Rename watchlist",
            submit_label="Rename",
            initial_name=current,
            exclude_id=watchlist_id,
        )
        if name is None:
            return
        updated = service.rename(watchlist_id, name)
        self._notify_watchlists(
            f"Watchlist renamed to \"{escape_markup(str(updated['name']))}\"."
        )
        self._load_tree_data()

    @on(DeleteWatchlistRequested)
    def handle_delete_watchlist_requested(
        self, event: DeleteWatchlistRequested
    ) -> None:
        event.stop()
        watchlist_id = event.watchlist_id
        self._start_tree_write(lambda: self._delete_watchlist_flow(watchlist_id))

    async def _delete_watchlist_flow(self, watchlist_id: int) -> None:
        """Delete a watchlist after saying, up front, what happens to its
        sources -- and then show the user where they went.

        Deleting a watchlist cascades only the membership rows; the sources
        themselves survive and become unassigned. That is invisible unless
        someone says so, which is exactly the "orphaned into invisibility"
        failure the tree's permanent Unassigned root exists to prevent, so
        the confirmation states the count and the destination before the
        user commits, and the scope moves to Unassigned afterwards rather
        than sitting on an id that no longer resolves.
        """
        service = self._watchlist_bundle_service()
        if service is None:
            self._notify_watchlists(WC_SERVICE_UNAVAILABLE_COPY, severity="error")
            return
        name = self._watchlist_display_name(watchlist_id)
        source_count = len(service.list_source_rows(watchlist_id))
        # Still needed by the post-delete notification below, which reads
        # correctly either way ("Its 1 source moved", "Its 2 sources moved").
        noun = "source" if source_count == 1 else "sources"
        confirmed = await self.app.push_screen_wait(
            ConfirmationDialog(
                title="Delete watchlist",
                message=(
                    f'Delete the watchlist "{escape_markup(name)}"?\n\n'
                    + watchlist_delete_consequence(source_count)
                ),
                confirm_label="Delete watchlist",
                cancel_label="Keep it",
            )
        )
        if not confirmed:
            return
        service.delete(watchlist_id)
        self._request_tree_scope(TreeScope(kind="unassigned"))
        self._notify_watchlists(
            f'Watchlist "{escape_markup(name)}" deleted. Its {source_count} '
            f"{noun} moved to Unassigned."
        )
        self._load_tree_data()

    @on(AddSourceToWatchlistRequested)
    def handle_add_source_to_watchlist_requested(
        self, event: AddSourceToWatchlistRequested
    ) -> None:
        event.stop()
        # Review wave, I1. Two widgets post this now -- the rail, which has
        # been rendered disabled on a blocked backend since task-895, and the
        # Inspector's `Add existing`, which shipped ungated and wrote local
        # membership rows on the server backend. Both are gated at their own
        # render, but the refusal belongs HERE as well: this handler is the
        # single point every present and future poster of this message
        # reaches, and it is one call away from a durable write. Notified,
        # not silent, because either poster is a real button on screen.
        blocked = self._tree_write_disabled_reason()
        if blocked is not None:
            self._notify_watchlists(blocked, severity="warning", markup=False)
            return
        watchlist_id = event.watchlist_id
        self._start_tree_write(lambda: self._add_source_to_watchlist_flow(watchlist_id))

    async def _add_source_to_watchlist_flow(self, watchlist_id: int) -> None:
        service = self._watchlist_bundle_service()
        if service is None:
            self._notify_watchlists(WC_SERVICE_UNAVAILABLE_COPY, severity="error")
            return
        # `list_sources` (ids only) is the right query here rather than
        # `list_source_rows`: the members are needed as a membership test,
        # not for display, and the candidate rows already come from
        # `list_all_source_rows`.
        members = {int(source_id) for source_id in service.list_sources(watchlist_id)}
        candidates = [
            row for row in service.list_all_source_rows() if int(row["id"]) not in members
        ]
        chosen = await self.app.push_screen_wait(
            WatchlistSourcePickerDialog(
                self._watchlist_display_name(watchlist_id), candidates
            )
        )
        if chosen is None:
            return
        service.add_source(watchlist_id, int(chosen))
        source_name = next(
            (
                str(row.get("name"))
                for row in candidates
                if int(row["id"]) == int(chosen)
            ),
            f"Source {chosen}",
        )
        self._notify_watchlists(
            f'Added "{escape_markup(source_name)}" to '
            f'"{escape_markup(self._watchlist_display_name(watchlist_id))}".'
        )
        self._load_tree_data()

    @on(AssignSourceToWatchlistRequested)
    def handle_assign_source_to_watchlist_requested(
        self, event: AssignSourceToWatchlistRequested
    ) -> None:
        """Dispatch the Inspector's `Add to watchlist` press (TASK-2303).

        Refuses anything that is not a LOCAL `subscription`, for the reason
        `handle_resume_source_requested` spells out: membership rows key on
        a raw local subscription id, so a server entity carrying a numeric
        `source_id` would file a completely unrelated local source. Unlike
        that handler this one DOES notify -- the press is a real user gesture
        on a button that is on screen, and a silent refusal is the
        dead-affordance shape this task exists to remove.
        """
        event.stop()
        entity = event.entity
        if entity is None:
            return
        # Review wave, I1: the same backend gate the watchlist-side twin now
        # carries, so both directions of one write are refused by one
        # condition rather than by two that can drift.
        blocked = self._tree_write_disabled_reason()
        if blocked is not None:
            self._notify_watchlists(blocked, severity="warning", markup=False)
            return
        if (
            str(entity.get("backend") or "") != "local"
            or str(entity.get("entity_kind") or "") != "subscription"
        ):
            self._notify_watchlists(
                "Only local sources can be added to a watchlist.",
                severity="warning",
            )
            return
        source_id = entity.get("source_id")
        if source_id is None:
            self._notify_watchlists(
                "That source has no local id to file.", severity="error"
            )
            return
        source_name = str(
            entity.get("name")
            or entity.get("source_title")
            or entity.get("title")
            or f"Source {source_id}"
        )
        self._start_tree_write(
            lambda: self._assign_source_to_watchlist_flow(
                int(source_id), source_name
            )
        )

    async def _assign_source_to_watchlist_flow(
        self, source_id: int, source_name: str
    ) -> None:
        """The source-first half of membership editing (TASK-2303 AC#2).

        Mirrors `_add_source_to_watchlist_flow` exactly, in the other
        direction: candidates are the watchlists this source is NOT already
        in, the write is the same idempotent `add_source`, and the toast
        names both ends so it is clear a membership row was added and
        nothing was created.
        """
        service = self._watchlist_bundle_service()
        if service is None:
            self._notify_watchlists(WC_SERVICE_UNAVAILABLE_COPY, severity="error")
            return
        # One membership query, not one per watchlist (review wave, M5) --
        # matching `_add_source_to_watchlist_flow`'s single `list_sources`
        # call in the other direction.
        already_in = {
            int(watchlist_id)
            for watchlist_id in service.list_watchlists_for_source(source_id)
        }
        all_watchlists = service.list_watchlists()
        candidates = [
            watchlist
            for watchlist in all_watchlists
            if int(watchlist["id"]) not in already_in
        ]
        chosen = await self.app.push_screen_wait(
            # `total_watchlists` distinguishes "this source is in all of
            # them" from "there are none" (review wave, M2).
            WatchlistPickerDialog(
                source_name, candidates, total_watchlists=len(all_watchlists)
            )
        )
        if chosen is None:
            return
        service.add_source(int(chosen), source_id)
        # Named off `candidates`, not `_tree_watchlists`: that mirror is
        # refreshed by `_load_tree_data` below, so reading it here could name
        # a watchlist by whatever it was called at the last reload.
        watchlist_name = next(
            (
                str(watchlist.get("name"))
                for watchlist in candidates
                if int(watchlist.get("id", -1)) == int(chosen)
            ),
            f"Watchlist {chosen}",
        )
        # markup=False: both halves are user-authored free text (a watchlist
        # name typed here, a source name that can come straight out of a
        # remote feed's own <title>), so neither may be interpreted as Rich
        # markup on its way to a toast.
        self._notify_watchlists(
            f'Added "{source_name}" to "{watchlist_name}".',
            markup=False,
        )
        self._load_tree_data()

    @on(RemoveSourceFromWatchlistRequested)
    def handle_remove_source_from_watchlist_requested(
        self, event: RemoveSourceFromWatchlistRequested
    ) -> None:
        event.stop()
        watchlist_id = event.watchlist_id
        source_id = event.source_id
        self._start_tree_write(
            lambda: self._remove_source_from_watchlist_flow(watchlist_id, source_id)
        )

    async def _remove_source_from_watchlist_flow(
        self, watchlist_id: int, source_id: int
    ) -> None:
        """Drop one membership row. No confirmation: the source itself
        survives and the action is one Add-source press away from being
        undone, unlike deleting a watchlist. The notification names both
        ends so it is clear nothing was destroyed.
        """
        service = self._watchlist_bundle_service()
        if service is None:
            self._notify_watchlists(WC_SERVICE_UNAVAILABLE_COPY, severity="error")
            return
        source_name = next(
            (
                str(row.get("name"))
                for row in service.list_source_rows(watchlist_id)
                if int(row.get("id", -1)) == int(source_id)
            ),
            f"Source {source_id}",
        )
        watchlist_name = self._watchlist_display_name(watchlist_id)
        service.remove_source(watchlist_id, source_id)
        # The scope named a node that no longer exists; fall back to its
        # parent watchlist, which does.
        self._request_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        )
        self._notify_watchlists(
            f'Removed "{escape_markup(source_name)}" from '
            f'"{escape_markup(watchlist_name)}". The source itself is kept.'
        )
        self._load_tree_data()

    def watch_selected_scope(self) -> None:
        """Push scope + resolved labels into the live Inspector.

        Mirrors `watch_selected_entity` immediately below it: this only
        covers the "selection changed without a workbench rebuild" case --
        `_build_inspector_pane` covers the rebuild case by seeding a
        freshly-constructed `InspectorPane` from this same screen state.

        Does NOT refresh the scoped readouts; `watch_tree_scope` owns that
        (fix round 1, Finding 2). `selected_scope` also moves when a pane
        row is selected, which is not navigation and must leave the tree
        scope's surfaces alone.
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
        """Keep the scope-driven surfaces in step with the tree selection
        (Task 7; header half added task-1344 fix wave; items reload added
        task-2513).

        Deliberately does NOT do what `watch_active_section` does
        (`self.refresh(recompose=True)`): that rebuilds every region,
        including the Inspector, and a fresh `InspectorPane` instance is
        exactly what `watch_selected_scope`'s in-place push exists to avoid
        -- `test_changing_scope_clears_a_stale_entity_selection` (Task 5)
        holds a reference to the Inspector from *before* a scope change and
        asserts against it *after*, which a full recompose would silently
        break by handing that reference a defunct, unmounted widget.

        What moves instead, in place:

        * The centre header (`_refresh_centre_header_for_scope`), which
          carries the scoped summary on every tab since task-2513. (The
          FEEDS region, whose inline copy this used to refresh, is gone.)
        * The items list on the Read tab: a scope move changes which items
          are in view, so `_load_items` is re-dispatched — and the reload
          itself is scope-plumbed through `_items_scope_query` (task-2513).
        * The Sources table (`_push_scoped_sources_to_pane`), an in-place
          push on the pane's own reactive, not a region rebuild.
        * The still-mounted `WatchlistTree` itself (task-876): since this
          watcher is the single reconciliation point for BOTH a real tree
          click and a breadcrumb promotion (the latter never touches the
          tree widget at all -- see `handle_breadcrumb_scope_selected`),
          and neither one rebuilds the Tree instance, the tree's own
          `active_scope` would otherwise go stale the moment the scope
          changes by any path other than a fresh `_build_tree_pane`
          construction.
        """
        if not self.is_mounted:
            return
        self._refresh_centre_header_for_scope()
        # TASK-2304 AC#2. The Sources table follows the same scope the
        # centre header just took, so the two counts of "how many sources
        # are in view" cannot disagree. An in-place push on the pane's own
        # reactive, not a region rebuild -- see `_push_scoped_sources_to_pane`.
        self._push_scoped_sources_to_pane()
        self._sync_tree_navigation_authority()
        self._sync_items_filter_authority()
        if self.active_section == "artifacts":
            # Artifacts is the one section whose entire subject is the tree
            # scope: a briefing belongs to exactly one watchlist. Moving the
            # tree therefore changes what this pane is about, and without
            # this it would keep showing the previous watchlist's briefings
            # (and offer Generate against the new one) -- the split-brain
            # shape, on a surface that spends the user's provider quota.
            self._selected_briefing = None
            self.run_worker(
                self._load_briefings(), exclusive=True, group="wl-briefings-load"
            )

    def _sync_tree_navigation_authority(self) -> None:
        """Push contextual selection availability into the mounted rail."""
        try:
            tree = self.query_one("#wl-tree", WatchlistTree)
        except NoMatches:
            return
        reason = self._tree_selection_disabled_reason()
        tree.selection_disabled_reason = reason
        tree.active_scope = None if reason is not None else self.tree_scope

    def _sync_items_filter_authority(self) -> None:
        """Show the committed effective filter while preserving preference."""
        try:
            pane = self.query_one("#watchlists-items-pane", ArticleListPane)
        except NoMatches:
            return
        pane.status_filter = self._effective_items_status_filter()
        pane.status_filter_disabled_reason = self._items_filter_disabled_reason()

    def _refresh_centre_header_for_scope(self) -> None:
        """Queue a centre-header rebuild so the scoped summary follows the
        tree selection (task-1344 fix wave, Qodo correctness; made the only
        scoped readout by task-2513).

        Records intent on the shared surface queue rather than swapping the
        DOM itself: `_apply_local_wc_snapshot` and `_load_tree_data` also
        rebuild the header, and independent `exclusive=True` workers
        swapping the SAME surface would either interleave two remove/mount
        pairs over one `#wl-centre-status` id or -- with a shared group --
        cancel one of them between its `remove()` and its `mount()`,
        leaving nothing mounted. See `_request_surface_refresh`.
        """
        try:
            summary = self.query_one("#wc-watchlists-summary", Static)
        except NoMatches:
            pass
        else:
            summary.update(self._staging_summary_line(self.scoped_source_rows()))
        self._request_surface_refresh(self._SURFACE_HEADER)

    #: The workbench surfaces this screen rebuilds in place, rather than by
    #: recomposing itself (TASK-2200). Each maps to one call on
    #: `WatchlistsWorkbench`; ITEMS and CONTENT are deliberately absent --
    #: their panes are patched through their own reactives (see
    #: `_load_sources`, `watch_overview_data`) precisely so an in-flight
    #: create/edit form is never torn down by a background load.
    #:
    #: `_SURFACE_INSPECTOR` is the one CONDITIONAL surface: it rebuilds the
    #: right rail only when `_resolve_console_follow_drift` finds the
    #: Console-follow row no longer matches the adapter. The rail is where
    #: the noise-selector editor lives, and rebuilding it unconditionally on
    #: every background load would destroy a half-typed selector set -- the
    #: same class of harm this task removes from the Sources create form.
    _SURFACE_RAIL = "rail"
    _SURFACE_HEADER = "header"
    _SURFACE_INSPECTOR = "inspector"
    _SURFACE_READER = "reader"
    #: task-15461. The section swap: the ITEMS region's pane (routed by
    #: `active_section`), the centre header (which carries the tab strip) and
    #: whichever centre regions the new tab hides or shows. Queued here rather
    #: than run straight from `watch_active_section` because it swaps the SAME
    #: `#wl-centre-status` widget `_SURFACE_HEADER` does, and two independent
    #: remove/mount pairs over one id is exactly the interleaving this queue
    #: exists to prevent.
    _SURFACE_SECTION = "section"

    def _request_surface_refresh(self, *surfaces: str) -> None:
        """Record that one or more workbench surfaces need rebuilding.

        Record intent, drain serially, never cancel (TASK-1541's lesson,
        applied here to DOM swaps rather than durable writes). Both
        `WatchlistsWorkbench.refresh_region_content` and
        `refresh_header_content` are remove-then-mount pairs with an `await`
        between the two halves -- Textual's `NodeList._ensure_unique_id`
        refuses to mount the replacement while the old widget still holds the
        same id, so there is no atomic single-await swap available. A worker
        cancelled in that window (which is exactly what `exclusive=True`
        does to its predecessor) leaves the region with its content removed
        and nothing put back: a bordered empty box that survives until some
        unrelated rebuild happens along.

        So callers queue a surface name here and at most one drainer ever
        runs. Requests that arrive while it is running are picked up by its
        next loop rather than starting -- or cancelling -- anything.

        **Scheduled on the message pump, not as a worker** (task-15461). This
        used to be `run_worker(..., group="wc_surface_refresh")`, whose own
        group existed so that the several `run_worker(..., exclusive=True)`
        call sites on this screen without a group (e.g.
        `_load_active_section_data`) could not cancel the drainer mid-swap.
        `call_next` is immune to that by construction -- it is not a worker at
        all -- and it restores a property the section swap depends on:
        `refresh(recompose=True)`, which `watch_active_section` used to call,
        is itself a `call_next` callback, so anything that waits for the
        message pump to go quiet (`Pilot.pause`'s `_wait_for_screen`, and the
        app's own idle handling) waits for the DOM swap. A worker is invisible
        to that wait. Moving the section swap into a worker made every
        harness that opens a section and immediately queries its pane a
        coin-flip on how long the swap happened to take -- measured at ~250 ms
        for Artifacts, against a 200 ms pause.

        Args:
            surfaces: Any of `_SURFACE_RAIL`, `_SURFACE_HEADER`,
                `_SURFACE_READER`, `_SURFACE_SECTION` (each an unconditional rebuild of that
                surface) or `_SURFACE_INSPECTOR` (conditional -- the right
                rail is rebuilt only when the Console-follow row no longer
                matches the adapter; see `_resolve_console_follow_drift`).
                Unknown names are ignored by the drainer.
        """
        if not self._dom_is_live:
            return
        self._pending_surface_refresh.update(surfaces)
        if self._surface_refresh_draining:
            return
        # Arm, then disarm if scheduling fails -- `_start_tree_write`'s
        # discipline, for the identical failure mode (review wave, M1). Only
        # `_drain_surface_refresh`'s `finally` ever lowers this flag, and it
        # never runs if scheduling raises synchronously; the flag would then
        # be stuck True for the life of the screen, every later request would
        # queue and return, and the rail/header would silently stop
        # following every background loader. Arming *after* scheduling is not
        # the fix either: the drainer's `finally` could already have lowered
        # the flag by the time we raised it.
        self._surface_refresh_draining = True
        try:
            self.call_next(self._drain_surface_refresh)
        except Exception:
            self._surface_refresh_draining = False
            logger.opt(exception=True).warning(
                "Watchlists surface refresh could not be scheduled."
            )

    async def _drain_surface_refresh(self) -> None:
        """Rebuild every queued surface in place, one at a time.

        Loops until the queue is empty so a request that lands mid-drain is
        served by this same worker. There is no `await` between the loop's
        emptiness check and the `finally` that clears the flag, so a request
        can never slip into the gap and be dropped by a drainer that has
        already decided to stop.
        """
        try:
            while self._dom_is_live and self._pending_surface_refresh:
                surfaces = self._pending_surface_refresh
                self._pending_surface_refresh = set()
                try:
                    workbench = self.query_one(WatchlistsWorkbench)
                except NoMatches:
                    # Nothing to patch. Whatever removed the workbench (a tab
                    # switch, a layout push) rebuilds every region from this
                    # screen's current state on the way back, so the pending
                    # update is not lost -- it arrives with that rebuild.
                    break
                if self._SURFACE_SECTION in surfaces:
                    await self._rebuild_surface(
                        self._swap_active_section(workbench),
                        "the Watchlists section",
                    )
                    # The swap rebuilds the centre header itself, so a HEADER
                    # request that arrived in the same batch is already
                    # served; running it again would be the second of the two
                    # rebuilds task-15461 removes.
                    surfaces = surfaces - {self._SURFACE_HEADER}
                if self._SURFACE_RAIL in surfaces:
                    await self._rebuild_surface(
                        workbench.refresh_region_content(Region.LEFT_RAIL),
                        "the Watchlists rail",
                    )
                if self._SURFACE_HEADER in surfaces:
                    await self._rebuild_surface(
                        workbench.refresh_header_content(),
                        "the centre header",
                    )
                if self._SURFACE_READER in surfaces:
                    await self._rebuild_surface(
                        workbench.refresh_region_content(Region.CONTENT),
                        "the Reader",
                    )
                if self._SURFACE_INSPECTOR in surfaces:
                    await self._rebuild_surface(
                        self._rebuild_inspector_if_console_follow_drifted(workbench),
                        "the Inspector rail",
                    )
        finally:
            self._pending_surface_refresh = set()
            self._surface_refresh_draining = False

    async def _swap_active_section(self, workbench: WatchlistsWorkbench) -> None:
        """Move the workbench onto `active_section`, region by region.

        task-15461. This replaces `watch_active_section`'s whole-screen
        `refresh(recompose=True)`, which rebuilt the navigation bar, the
        footer, the header bar, both rails and both centre regions to change
        which pane the centre shows -- 75-176 mounted widgets per tab click,
        measured, including a fresh `WatchlistTree` whose `compose` runs one
        synchronous source-row query per expanded watchlist.

        Only three things on this screen actually read `active_section`:

        * the ITEMS region, whose pane `_build_detail_pane` routes by section;
        * the centre header, which carries the tab strip and the scoped
          snapshot markers;
        * the derived effective layout, which parks CONTENT off Read and
          forces the management canvas open.

        The header bar's backend Select is the fourth, and is patched in
        place by `_sync_backend_header_bar` (which also runs synchronously
        from the watcher, so the disabled state never lags the tab).

        Known and accepted: this path removes widgets WITHOUT going through
        `BaseAppScreen.refresh`, so the task-627 defensive `capture_mouse
        (None)` no longer runs on a tab switch. Reachability is low -- it
        needs a mouse still captured by something inside the header or the
        ITEMS pane at the instant a tab changes, i.e. a drag whose `MouseUp`
        has not landed while the section changes by some other route -- and
        the panes that recompose themselves already carry
        `RecomposeCaptureGuard`. Documented rather than fixed; a guard here
        would have to release only captures inside the two surfaces this
        swap actually tears down.

        Args:
            workbench: The mounted workbench, resolved once by the drainer.
        """
        intent = self._pending_section_intent
        self._pending_section_intent = None
        if intent is None:
            section = self.active_section
            self._recompute_effective_layout(
                cause="explicit", section=section, request_workbench=False
            )
            token = self._next_layout_request_token()
            detail_builder = self._build_detail_pane
            header_builder = self._build_centre_status_header
            intent = SectionViewIntent(
                token=token,
                section=section,
                read_mode=section == "items",
                layout=self._effective_region_layout,
                items_factory=lambda: detail_builder(section),
                header_factory=lambda: header_builder(section),
            )
        # Asked BEFORE the swap: afterwards the widget is already gone and
        # Textual has already re-homed focus (see `_restore_focus_after_swap`).
        rehome_focus = self._swap_will_destroy_focus()
        applied = await workbench.apply_section_view(
            read_mode=intent.read_mode,
            layout=intent.layout,
            token=intent.token,
            rebuild_regions=(Region.ITEMS,),
            rebuild_header=True,
            content={**workbench._content, Region.ITEMS: intent.items_factory},
            header=intent.header_factory,
        )
        if not applied:
            if intent.token != self._current_layout_request_token:
                return
            previous_section = self._rendered_section
            self.set_reactive(
                WatchlistsCollectionsScreen.active_section, previous_section
            )
            self._article_focus_active = False
            self._effective_region_layout = workbench.region_layout
            self._responsive_region_layout = workbench.region_layout
            self._sync_backend_header_bar()
            return
        self._rendered_section = intent.section
        if intent.token != self._current_layout_request_token:
            return
        if rehome_focus:
            self._restore_focus_after_swap()
        self._reseed_active_section_pane()

    #: The two surfaces `_swap_active_section` tears down: the centre header
    #: (which carries the tab strip) and the ITEMS region (the section's own
    #: pane). Focus living inside either one does not survive the swap.
    _SWAP_OWNED_CONTAINER_IDS = frozenset({"wl-centre-status", "wl-region-items"})

    def _swap_will_destroy_focus(self) -> bool:
        """Whether the section swap is about to unmount the focused widget.

        Asked before the swap, because Textual answers it for us afterwards --
        badly. See `_restore_focus_after_swap`.

        Returns:
            `True` when something is focused and it sits inside the centre
            header or the ITEMS region.
        """
        node = self.focused
        while node is not None:
            if (getattr(node, "id", None) or "") in self._SWAP_OWNED_CONTAINER_IDS:
                return True
            node = node.parent
        return False

    def _restore_focus_after_swap(self) -> None:
        """Put focus back on the tab the user just switched to.

        A tab click focuses the tab `Button`, which lives in the centre header
        -- and the swap rebuilds that header, so the focused widget is removed
        from under the user. Textual's `Screen._reset_focus` then walks for a
        replacement and lands on the first focusable thing it finds, which on
        this screen is a node in the LEFT RAIL. That is not cosmetic:
        `on_descendant_focus` reads the new focus and sets
        `focused_region = LEFT_RAIL`, so the very next `z` collapses the rail
        AND persists that collapse to config -- from a keypress the user aimed
        at the section they were looking at. Measured A/B against the
        pre-task code, where the whole-screen recompose left focus at `None`
        and `z` was refused by `_refuse_region_gesture_off_read_tab`.

        Re-focusing the freshly built tab restores that refusal by the honest
        route rather than by accident: focus lands in `#wl-centre-status`, so
        `on_descendant_focus` sets `_focus_in_centre_header`, which is exactly
        what `action_toggle_region` already consults.

        Only called when the swap really did unmount the focused widget
        (`_swap_will_destroy_focus`): a section change driven from elsewhere
        -- a deep link, `EditRuleRequested` -- must not steal focus from
        wherever the user actually is.
        """
        try:
            self.query_one(f"#wl-tab-{self.active_section}", Button).focus()
        except NoMatches:
            # No tab for this section (nothing in `SECTIONS` matches), or the
            # header could not be rebuilt. Leaving focus where Textual put it
            # is still wrong, but there is nothing better to reach for here.
            logger.debug("No section tab to restore focus to after the swap.")

    def _reseed_active_section_pane(self) -> None:
        """Re-apply the section's loaded rows to the pane the swap mounted.

        `watch_active_section` dispatches this section's loader and the swap
        in the same breath, and `WatchlistsWorkbench.refresh_region_content`
        deliberately calls the region factory BEFORE detaching the old pane
        (so a factory that raises leaves what is on screen standing). Those
        two facts open a window: the factory reads screen state, then the
        remove/mount pair awaits, and a loader that lands in that gap writes
        its rows to `self._loaded_*` and then fails to find its pane, because
        the replacement is built but not yet mounted. The pane then renders
        the state as it was one instant before the data arrived, with nothing
        left to correct it -- measured as an Alert-rules table that stayed
        empty over a `self._loaded_rules` holding the row.

        (The whole-screen `refresh(recompose=True)` this replaced did not have
        the gap: Textual's `Widget.recompose` removes its children first and
        calls `compose()` only afterwards, so it always read screen state on
        the late side of the yield.)

        Re-applying after the mount closes it from the other end, and costs
        nothing when no loader landed: these are reactive assignments, and an
        unchanged value fires no watcher and no recompose. It is also what
        keeps a section switch to ONE pane build -- the loader's own push,
        arriving later, finds the values already in place.
        """
        if not self._dom_is_live:
            return
        section = self.active_section
        try:
            if section == "items":
                self.query_one(
                    "#watchlists-items-pane", ArticleListPane
                ).items = self._loaded_items
            elif section == "sources":
                self.query_one(
                    "#watchlists-sources-pane", SourcesPane
                ).sources = self.scoped_loaded_sources()
            elif section == "runs":
                self._reseed_live_detail_pane()
            elif section == "rules":
                self.query_one("#watchlists-rules-pane", RulesPane).rules = (
                    self._loaded_rules
                )
            elif section == "notifications":
                pane = self.query_one(
                    "#watchlists-notifications-pane", NotificationsPane
                )
                pane.notifications = self._loaded_notifications
                pane.selected_notification = self.selected_notification
            elif section == "artifacts":
                self._apply_briefing_state_to_pane()
        except NoMatches:
            # The region is collapsed, or the swap could not build its pane.
            # Either way the next build seeds from this same state.
            pass

    def _sync_backend_header_bar(self) -> None:
        """Repaint the backend picker row for the active section, in place.

        `_LOCAL_ONLY_SECTIONS` disables the Select and adds an explanatory
        `#watchlists-backend-label`; both used to arrive via the whole-screen
        recompose `watch_active_section` no longer does (task-15461). Kept
        synchronous, unlike the region swap: this is three attribute writes
        and at most one one-widget mount, and a Select that stays enabled for
        even one frame on a local-only tab is a control that lies about what
        it can do.

        The Select instance itself is deliberately NOT rebuilt -- its `value`
        already tracks `runtime_backend`, and rebuilding it would close an
        open dropdown mid-choice.
        """
        local_only = self._local_only_section()
        try:
            backend_select = self.query_one("#watchlists-backend-select", PruneSafeSelect)
        except NoMatches:
            return
        backend_select.disabled = local_only is not None
        backend_select.tooltip = (
            (local_only or {}).get("tooltip")
            or "Choose the Watchlists data backend."
        )
        label_text = self._backend_label_text()
        try:
            label: Static | None = self.query_one("#watchlists-backend-label", Static)
        except NoMatches:
            label = None
        if label_text is None:
            if label is not None:
                label.remove()
            return
        if label is None:
            try:
                self.query_one("#watchlists-header-bar").mount(
                    Static(label_text, id="watchlists-backend-label")
                )
            except NoMatches:
                pass
            return
        label.update(label_text)

    async def _rebuild_inspector_if_console_follow_drifted(
        self, workbench: WatchlistsWorkbench
    ) -> None:
        """Re-poll the Console-follow adapter; rebuild the rail if it moved.

        Wrapped in a coroutine rather than inlined into the drain loop's `if`
        (review wave, M2) so `_rebuild_surface`'s `except Exception` covers
        the poll as well as the rebuild. `run_worker` defaults to
        `exit_on_error=True`, so an exception escaping the drainer reaches
        `App._handle_exception` and takes the app down -- and every other step
        in that loop was deliberately made non-fatal.

        Args:
            workbench: The mounted workbench, resolved once by the drainer.
        """
        if not self._resolve_console_follow_drift():
            return
        await workbench.refresh_region_content(Region.RIGHT_RAIL)

    @staticmethod
    async def _rebuild_surface(rebuild: Any, what: str) -> None:
        """Await one surface rebuild, logging rather than killing the drain.

        A single failing surface must not abandon the ones queued behind it,
        and it must not leave `_surface_refresh_draining` stuck either (the
        drainer's own `finally` covers that, but only if this coroutine does
        not propagate).

        Args:
            rebuild: The workbench coroutine to await.
            what: Human-readable surface name, for the debug log only.
        """
        try:
            await rebuild
        except Exception:
            logger.opt(exception=True).debug(f"Failed to rebuild {what} in place.")

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Keep `focused_region` in step with whatever actually holds focus.

        Without this, `z` always collapses whichever region `focused_region`
        happened to default to, regardless of where the user actually is.
        Both id prefixes are checked so that focusing a *collapsed* region's
        header targets that region rather than expanding some other one.

        Also tracks `_focus_in_centre_header` (task-1344 fix wave, Qodo
        correctness): the centre header/tab strip (`#wl-centre-status`,
        mounted directly under `#wl-centre` by `_build_centre_status_header`
        -- see `compose_content`) sits outside every region/grip wrapper on
        every section including Read (TASK-2312;
        Read used to mount its own copy of the tab strip INSIDE FEEDS's own
        `wl-region-feeds` wrapper, so this branch never fired there and
        focusing the tab strip on Read instead matched the `wl-region-`
        prefix and set `focused_region = FEEDS`), so neither prefix above
        ever matches while focus is in it. Without this branch `focused_region` would keep
        naming whatever region the user last actually visited, silently
        indistinguishable here from a live selection: a user who tabs into
        the tab strip and presses `z`/`Z` would act on that stale reference
        with no visible relationship to where they are.
        `action_toggle_region` consults `_focus_in_centre_header` to refuse
        exactly that.
        """
        node = event.widget
        while node is not None:
            node_id = getattr(node, "id", None) or ""
            for prefix in ("wl-region-", "wl-grip-"):
                if node_id.startswith(prefix):
                    try:
                        self.focused_region = Region(node_id[len(prefix):])
                    except ValueError:
                        pass
                    self._focus_in_centre_header = False
                    return
            if node_id == "wl-centre-status":
                self._focus_in_centre_header = True
                return
            node = node.parent
        # Focus landed in neither zone -- a widget outside both the centre
        # regions and the status header (e.g. `#watchlists-backend-select`, a
        # sibling of the workbench). The flag tracks ONLY "focus is in the
        # status header", so anything else must clear it; leaving a stale
        # `True` from a prior header focus would wrongly refuse a later
        # `z`/`Z` (task-1344 fix wave re-review, Qodo-follow-up). `focused_
        # region` keeps its last real value, as it already did for this case.
        self._focus_in_centre_header = False

    def watch_active_section(self) -> None:
        # A tab switch always rebuilds the centre header and the section's own
        # pane (below): whatever `_focus_in_centre_header` was tracking about
        # the OLD DOM is moot the instant that DOM is torn down. Reset
        # to "not in the header" rather than leave a stale `True` standing
        # -- `on_descendant_focus` will set it again the moment a fresh
        # focus event actually lands there, but nothing guarantees one
        # fires if the new tab happens to auto-focus nothing trackable, and
        # a stale `True` would wrongly refuse a legitimate `z`/`Z` on the
        # new tab (task-1344 fix wave, Qodo correctness).
        self._focus_in_centre_header = False
        leaving_read_recovery = (
            self.active_section != "items" and self._read_recovery_active
        )
        if self.active_section != "items":
            self._article_focus_active = False
            if self._read_recovery_active:
                self._read_recovery_active = False
                if self.is_mounted and not self._wc_loaded:
                    # A cold Server Read deliberately skipped these local
                    # management models. Load them only if the user leaves
                    # recovery for a management tab.
                    self._refresh_local_wc_snapshot()
                    self._load_tree_data()
                    self.set_timer(
                        WC_SNAPSHOT_TIMEOUT_SECONDS,
                        self._apply_snapshot_timeout_if_still_loading,
                    )
        self._recompute_effective_layout(
            cause="explicit", request_workbench=False
        )
        if self.active_section == "overview":
            self.selected_entity = None
        if self.active_section != WATCHLISTS_SECTION_RUNS:
            self._pending_navigation_run_id = None
            self._pending_navigation_run_backend = None
        if self.is_mounted:
            self._sync_tree_navigation_authority()
            if self.active_section == "items" and self.runtime_backend != "local":
                self._enter_server_read_recovery()
            token = self._next_layout_request_token()
            section = self.active_section
            detail_builder = self._build_detail_pane
            header_builder = self._build_centre_status_header
            self._pending_section_intent = SectionViewIntent(
                token=token,
                section=section,
                read_mode=section == "items",
                layout=self._effective_region_layout,
                items_factory=lambda: detail_builder(section),
                header_factory=lambda: header_builder(section),
            )
            # task-15461: one region-scoped swap, not a whole-screen
            # `refresh(recompose=True)`. Queued on the surface drain rather
            # than run here because it swaps `#wl-centre-status`, the same
            # widget `_SURFACE_HEADER` swaps -- see `_SURFACE_SECTION`.
            self._sync_backend_header_bar()
            surfaces = [self._SURFACE_SECTION]
            if leaving_read_recovery:
                # Recovery replaced the live rail with an empty model. The
                # ordinary section swap deliberately rebuilds only centre
                # surfaces, so restore the parked navigation explicitly.
                surfaces.append(self._SURFACE_RAIL)
            self._request_surface_refresh(*surfaces)
            if not self._applying_navigation_context:
                self._load_active_section_data()

        if self._pending_open_create_form:
            self._pending_open_create_form = False
            self.set_timer(0.05, self._open_sources_create_form)
        if self._pending_open_import_opml:
            self._pending_open_import_opml = False
            self.set_timer(0.05, self._open_sources_import_opml)

    def _load_active_section_data(self) -> None:
        """Start the loader owned by the currently visible section.

        Every branch names its own group. TASK-19559: for four releases only
        the `items` and `artifacts` branches did -- the hazard comment below
        sat directly above four siblings it did not actually protect, so
        switching to Rules/Runs/Sources/Notifications cancelled whatever
        default-group worker was in flight (a source create, a rule save, a
        run cancellation, ...).
        """
        if self.active_section == "items":
            if self.runtime_backend != "local":
                return
            # Own group (task-2513), as in `watch_tree_scope`:
            # `exclusive=True` in the default group would cancel every
            # in-flight default-group worker (`_create_source`, ...).
            self.run_worker(
                self._replace_items_snapshot(
                    reason=(
                        "return_to_read"
                        if self._items_snapshot is None
                        else "initial"
                    )
                ),
                exclusive=True,
                group="wc_items",
            )
        elif self.active_section == "rules":
            self.run_worker(self._load_rules(), exclusive=True, group="wc_rules")
        elif self.active_section == "runs":
            self.run_worker(self._load_runs(), exclusive=True, group="wc_runs")
        elif self.active_section == "sources":
            self.run_worker(self._load_sources(), exclusive=True, group="wc_sources")
        elif self.active_section == "notifications":
            self.run_worker(
                self._load_notifications(),
                exclusive=True,
                group="wc_notifications",
            )
        elif self.active_section == "artifacts":
            # Own group (TASK-1362): `exclusive=True` without one cancels
            # every other worker in the default group, which here would
            # include an in-flight briefing generation.
            self.run_worker(
                self._load_briefings(), exclusive=True, group="wl-briefings-load"
            )

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
        source_types = self._create_form_source_types(self.runtime_backend)
        if (
            self._source_create_draft_type is not None
            and self._source_create_draft_type not in source_types
        ):
            self._source_create_draft_type = "rss"
        if not self.is_mounted:
            return
        self._sync_live_source_create_backend()
        read_is_active = self.active_section == "items"
        local_read_is_active = read_is_active and self.runtime_backend == "local"
        if read_is_active:
            if local_read_is_active:
                self._reset_items_paging_for_context(loading=True)
                if self._read_recovery_active:
                    self.run_worker(
                        self._recover_local_read(),
                        exclusive=True,
                        group="wc_items",
                    )
                else:
                    self._load_tree_data()
                    self.run_worker(
                        self._replace_items_snapshot(reason="return_to_read"),
                        exclusive=True,
                        group="wc_items",
                    )
            else:
                self._enter_server_read_recovery()
        else:
            # Paging belongs to a backend-specific Read query context even
            # while another management tab is visible. Invalidate it without
            # loading the hidden Read surface.
            self._reset_items_paging_for_context(loading=False)
        try:
            label = self.query_one("#watchlists-backend-label", Static)
            label_text = self._backend_label_text()
            if label_text is not None:
                label.update(label_text)
        except Exception:
            pass
        # task-895: push the new write-availability into the still-mounted
        # tree, the same way `watch_tree_scope` pushes `active_scope`. This
        # is now the ONLY thing that updates it on a backend switch
        # (TASK-2200): the snapshot refresh below used to recompose the whole
        # screen, and `_build_tree_pane` re-seeded this on the way through.
        # It no longer does, so without this the five action buttons would
        # sit enabled over a backend that cannot service them -- the exact
        # "disabled button that looks enabled" shape in reverse.
        try:
            self.query_one("#wl-tree", WatchlistTree).write_disabled_reason = (
                self._tree_write_disabled_reason()
            )
        except NoMatches:
            pass
        self._sync_tree_navigation_authority()
        # Review wave, I1: and into the Inspector, which carries the same
        # verb. Pushed from here rather than left to the next rebuild for
        # exactly the reason the tree push above documents -- nothing
        # recomposes the screen on a backend switch any more (TASK-2200), so
        # without this the Inspector's `Add existing` sits enabled over a
        # backend that cannot service it.
        try:
            self.query_one(
                "#watchlists-entity-inspector", InspectorPane
            ).write_disabled_reason = self._tree_write_disabled_reason()
        except NoMatches:
            pass
        self.selected_source = None
        self.selected_run = None
        self.selected_notification = None
        self.selected_entity = None
        self._loaded_runs = []
        # Same reason as the tree push above (TASK-2200): the four lines
        # above clear the screen's mirrored state, and until this task the
        # snapshot refresh's full-screen recompose is what carried that into
        # whichever detail pane was mounted (`_build_detail_pane` re-seeds
        # every pane from exactly these attributes). `selected_entity` is
        # the one that already had its own watcher.
        self._reseed_live_detail_pane()
        if not (read_is_active and self._read_recovery_active):
            self._refresh_local_wc_snapshot()
            self._refresh_overview_data()

    def _reseed_live_detail_pane(self) -> None:
        """Push the screen's mirrored rows/selection into the mounted pane.

        The in-place half of `_build_detail_pane`'s seeding, for callers that
        change that mirrored state without rebuilding the ITEMS region
        (TASK-2200). Only the pane for the active section can be mounted, so
        at most one branch does anything; each is a no-op when its pane is
        absent (the region collapsed, or a different tab).
        """
        if not self._dom_is_live:
            return
        for selector, pane_type, values in (
            (
                "#watchlists-sources-pane",
                SourcesPane,
                {"sources": self._loaded_sources, "selected_source": self.selected_source},
            ),
            (
                "#watchlists-runs-pane",
                RunsPane,
                # Insertion order is load-bearing: `selected_run` clears the
                # pane's detail, so the detail must be re-pushed after it (see
                # `_build_detail_pane`'s identical ordering note).
                {
                    "runs": self._loaded_runs,
                    "selected_run": self.selected_run,
                    "run_items": self._run_detail_items,
                    "run_logs": self._run_detail_logs,
                    "run_items_note": self._run_detail_items_note,
                },
            ),
            (
                "#watchlists-notifications-pane",
                NotificationsPane,
                {
                    "notifications": self._loaded_notifications,
                    "selected_notification": self.selected_notification,
                },
            ),
        ):
            try:
                pane = self.query_one(selector, pane_type)
            except NoMatches:
                continue
            for attribute, value in values.items():
                setattr(pane, attribute, value)

    def watch_selected_entity(self) -> None:
        """Push the current selection into the live Inspector.

        `_dom_is_live`, not `is_mounted` (Qodo, PR #1331): this watcher IS
        reachable inside the mount window, unlike its `watch_selected_scope`
        sibling. The Watchlists run deep link arms `_pending_navigation_run_id`
        on an unmounted screen (`apply_navigation_context`), `on_mount` starts
        `_load_runs`, and on a cold database that loader resolves the target
        and calls `_select_entity(requested_run)` before Textual has flipped
        `_is_mounted`. An `is_mounted` guard dropped that push, and nothing
        re-seeds it: `_build_inspector_pane` re-seeds only on a REBUILD, and
        the one right-rail rebuild this screen still schedules is gated on
        `_resolve_console_follow_drift()`, which is `False` on a normal cold
        start. The user followed a run link and the Inspector said "Nothing to
        inspect yet." over a run the screen believed was selected.
        """
        if not self._dom_is_live:
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
        centre header's scoped summary must keep naming the watchlist the
        user opened. Before the two scopes were split, this reset silently
        rebuilt that readout back to "All sources" -- an interaction in one
        region discarding the user's navigation in another, with no tree
        selection highlight to fall back
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

    @on(Button.Pressed, "#watchlists-switch-local")
    def handle_switch_to_local(self, event: Button.Pressed) -> None:
        """Recover Read through the same selector path as a manual change."""
        event.stop()
        selector = self.query_one("#watchlists-backend-select", PruneSafeSelect)
        if self.runtime_backend == "local":
            self.run_worker(
                self._recover_local_read(), exclusive=True, group="wc_items"
            )
        else:
            selector.value = "local"

    @on(Button.Pressed, "#watchlists-items-retry-button")
    def handle_items_retry(self, event: Button.Pressed) -> None:
        """Retry the committed Reader scope without exposing stale rows."""
        event.stop()
        if self._items_retry_message is None or self._items_retry_inflight:
            return
        self._items_retry_inflight = True
        self._items_page_loading = True
        self._request_surface_refresh(self._SURFACE_SECTION)
        retry = self._retry_items_snapshot()
        try:
            self.run_worker(retry, exclusive=True, group="wc_items")
        except Exception:
            retry.close()
            self._items_retry_inflight = False
            self._items_page_loading = False
            self._request_surface_refresh(self._SURFACE_SECTION)

    async def _retry_items_snapshot(self) -> None:
        """Keep retry authority mounted until one publication succeeds."""
        try:
            await self._replace_items_snapshot(reason="return_to_read")
        finally:
            self._items_retry_inflight = False
            if self._items_retry_message is not None:
                self._request_surface_refresh(self._SURFACE_SECTION)

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
        if event.active is not None:
            self._source_create_draft_active = event.active
        if event.frequency is not None:
            self._source_create_draft_frequency = event.frequency
        if event.ignore_selectors is not None:
            self._source_create_draft_selectors = event.ignore_selectors
        if event.source_type is not None:
            self._source_create_draft_type = event.source_type
        if event.destination is not None:
            self._source_create_draft_destination = event.destination

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
        # TASK-2306. Nothing in the product had ever written
        # `RunsPane.run_items` / `run_logs` -- only the pane's own unit test
        # did -- so the Items and Logs sub-regions of the Runs tab were
        # structurally empty in the running app no matter what was selected.
        self.run_worker(
            self._load_run_detail(event.run),
            exclusive=True,
            group="wc_run_detail",
        )

    @on(RunProgressTick)
    def handle_run_progress_tick(self, event: RunProgressTick) -> None:
        """A running run may have moved on -- check, cheaply (Qodo #1348)."""
        event.stop()
        self.run_worker(
            self._refresh_running_run(event.run_id),
            exclusive=True,
            group="wc_run_tick",
        )

    #: The fields of a run that a tick can find changed. Everything else on a
    #: run record is fixed at launch, so a fingerprint over these is what
    #: decides whether a tick does any work at all.
    _RUN_PROGRESS_FIELDS = (
        "status",
        "finished_at",
        "found_count",
        "processed_count",
        "filtered_count",
        "error_count",
        "log_text",
        "error_msg",
    )

    @classmethod
    def _run_progress_fingerprint(cls, run: Mapping[str, Any]) -> tuple[str, ...]:
        """The volatile part of a run record, as a comparable tuple."""
        return tuple(str(run.get(field) or "") for field in cls._RUN_PROGRESS_FIELDS)

    async def _refresh_running_run(self, run_id: Any) -> None:
        """Re-read one running run and repaint only if it actually changed.

        Qodo, PR #1348. `run_poll` used to re-post `RunSelected` every second,
        and `handle_run_selected` cannot tell a tick from a click -- so a
        selected running run scheduled a full `_load_run_detail` (worker plus
        item query) once a second, with no user action, for up to a minute.

        The shape chosen here is (a): a distinct tick message whose handler
        refreshes what a run can actually change. That matters because the
        naive alternative -- skipping on an unchanged id -- would freeze the
        detail at its first paint, and during a LOCAL run the first paint is
        exactly the useless one: `execute_run` writes `stats_json`,
        `finished_at` and `log_text` in `record_run_result` and upserts the
        items in one go at the END, so a run polled while running has nothing
        to show until it finishes. The tick's real job is to notice that
        moment. Until it arrives the fingerprint is unchanged and this costs
        one cheap read and nothing else -- no item query, no repaint.

        Args:
            run_id: The namespaced id the poll is watching.
        """
        selected = self.selected_run
        if selected is None or str(selected.get("id") or "") != str(run_id):
            # The user moved on between the tick being posted and this worker
            # starting. Nothing to refresh, and nothing to resurrect.
            return
        try:
            record = await self._controller.get_run(
                runtime_backend=self.runtime_backend,
                run_id=run_id,
            )
        except Exception as exc:
            # Deliberately silent: this fires once a second on a timer the
            # user did not press, so a toast per tick would be its own defect.
            # The run row keeps its last known state, which is honest.
            logger.opt(exception=True).debug(
                f"Failed to re-read running watchlist run {run_id!r}: "
                f"{type(exc).__name__}"
            )
            return
        if not isinstance(record, Mapping) or not record:
            return
        if self._run_progress_fingerprint(record) == self._run_progress_fingerprint(
            selected
        ):
            return

        record = dict(record)
        for index, candidate in enumerate(self._loaded_runs):
            if str(candidate.get("id") or "") == str(run_id):
                self._loaded_runs[index] = record
                break
        self.selected_run = record
        if self._dom_is_live:
            try:
                self.query_one(
                    "#watchlists-runs-pane", RunsPane
                ).apply_run_progress(record)
            except Exception:
                pass
        # Only now is a full detail load worth its cost: the run reached a new
        # state, which for a local run is when its items land.
        await self._load_run_detail(record)

    async def _load_run_detail(self, run: dict[str, Any] | None) -> None:
        """Fill the selected run's Items and Logs sub-regions.

        The log text is already on the run record (`normalize_watchlist_run`
        carries `log_text`), so only the items need a query.

        Every road out of here that yields no rows also names ITSELF (review
        wave, Important 1). An empty items table renders identically whether a
        later check re-claimed this run's rows, the run genuinely found
        nothing, the backend cannot list items at all, or the query failed --
        and it sits directly beneath a stats block that may well say
        `Found: 3`. The note is what tells those four apart. Storage semantics
        (`persist_subscription_item`'s `run_id = excluded.run_id`) stay out of
        scope; the label does not.

        Args:
            run: The newly selected run, or `None` when the selection was
                cleared.
        """
        if run is None:
            self._run_detail_items = []
            self._run_detail_logs = ""
            self._run_detail_items_note = ""
            self._push_run_detail_to_live_pane(None)
            return

        items: list[dict[str, Any]] = []
        note = ""
        # `normalize_watchlist_run` reads `payload["id"]` unsubscripted, so
        # every run that reaches this screen HAS a `run_id` -- there is no
        # user-facing "unidentified run" state, and the label this branch used
        # to carry was dead (re-review, m6). The guard itself stays and is not
        # a label: `list_items(run_id=None)` drops the predicate and returns
        # EVERY item, which would attribute the whole database to one run. It
        # is an invariant backstop, so it falls through to the ordinary
        # count-derived note rather than inventing a state of its own.
        run_id = run.get("run_id")
        backend = str(run.get("backend") or self.runtime_backend)
        if backend != "local":
            # `WatchlistScopeService.list_items` refuses the server backend
            # outright, so there is no query here to fail -- say so, rather
            # than drawing the same blank a local run with no items draws.
            note = self._RUN_ITEMS_SERVER_NOTE
        elif run_id is None:
            note = self._run_items_note(run, [])
        else:
            try:
                rows = await self._controller.list_items(
                    runtime_backend="local",
                    run_id=run_id,
                    status=None,
                    limit=self._RUN_ITEMS_LIMIT,
                )
            except Exception as exc:
                # Review wave, Important 2. The "loaders may log at debug"
                # exemption (`test_watchlists_check_now_failure.py`) is paid
                # for by a visible toast, and every sibling loader on this
                # screen pays it. Without one, a denied `items.list` policy or
                # a database locked by a concurrent write rendered
                # byte-identically to "this run produced no items".
                #
                # Type only in the message: an exception's text can carry a
                # remote URL or a local path, and `opt(exception=True)`
                # already delivers the full traceback to the sink.
                logger.opt(exception=True).debug(
                    "Failed to load the items of watchlist run "
                    f"{run.get('id')!r}: {type(exc).__name__}"
                )
                notify = getattr(self.app_instance, "notify", None)
                if callable(notify):
                    notify(
                        "Failed to load this run's items.",
                        severity="error",
                        markup=False,
                    )
                note = self._RUN_ITEMS_FAILED_NOTE
            else:
                items = [dict(item) for item in rows]
                note = self._run_items_note(run, items)

        self._run_detail_items = items
        self._run_detail_logs = self._run_log_text(run)
        self._run_detail_items_note = note
        self._push_run_detail_to_live_pane(run)

    #: How many of a run's items the detail region lists. A run can produce
    #: more (the `sitemap`/`url_list` arms fan out per URL), so the page size
    #: gets said out loud rather than silently truncating a table sitting
    #: under a `Found:` count that disagrees with it -- review wave, Minor 2.
    _RUN_ITEMS_LIMIT = 200
    _RUN_ITEMS_SERVER_NOTE = "Items are not listed for server-backend runs."
    _RUN_ITEMS_FAILED_NOTE = "Could not load this run's items."
    _RUN_ITEMS_REATTRIBUTED_NOTE = (
        "No item rows are still attributed to this run — a later check "
        "re-claimed the items that had not changed."
    )
    _RUN_ITEMS_ALL_FILTERED_NOTE = (
        "Every item this run found was excluded by a filter, so it stored "
        "none."
    )
    _RUN_ITEMS_EMPTY_NOTE = "This run produced no items."

    @classmethod
    def _run_items_note(
        cls, run: Mapping[str, Any], items: Sequence[Mapping[str, Any]]
    ) -> str:
        """What to say about a successful item query's result.

        **`processed_count`, never `found_count`** (re-review, I1-b). `Found`
        is the tally the FETCH reported; `Processed` is how many rows the run
        actually persisted, and rows are the only thing this table can ever
        show. Discriminating on `Found` mistook filtering for
        re-attribution: a source with an exclude filter, checked ONCE
        (`found 5 · processed 0 · filtered 5`), was told "a later check
        re-claimed the items that had not changed" when no later check
        existed. The truncation line had the same bug in reverse — a run of
        `found 500 · processed 200` returning exactly 200 rows claimed 300
        were hidden when every row it ever stored was on screen.

        Args:
            run: The run whose items were queried.
            items: The rows that came back.

        Returns:
            The note, or `""` when the table speaks for itself.
        """
        found = cls._run_count(run, "found_count")
        processed = cls._run_count(run, "processed_count")
        if not items:
            if processed > 0:
                # It stored rows and none are left: something took them, and
                # `persist_subscription_item`'s `run_id = excluded.run_id` is
                # the only thing that does.
                return cls._RUN_ITEMS_REATTRIBUTED_NOTE
            if found > 0:
                # It fetched, kept nothing, and therefore stored nothing.
                # Empty is the CORRECT render here -- the note exists to stop
                # the user reading it as breakage, and to point at the filter
                # that caused it.
                return cls._RUN_ITEMS_ALL_FILTERED_NOTE
            return cls._RUN_ITEMS_EMPTY_NOTE
        if len(items) >= cls._RUN_ITEMS_LIMIT and processed > len(items):
            return f"Showing the first {len(items)} of {processed} items."
        return ""

    @staticmethod
    def _run_count(run: Mapping[str, Any], key: str) -> int:
        """One of a run's accounting counters as an int, 0 if unreadable."""
        value = run.get(key) or 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _push_run_detail_to_live_pane(self, run: Mapping[str, Any] | None) -> None:
        """Push the mirrored run detail into the mounted `RunsPane`.

        `_dom_is_live`, not `is_mounted` (TASK-2200's mount-window lesson):
        the run deep link arms a selection before mount and `_load_runs`
        answers it inside `on_mount`, so this can genuinely be reached while
        `is_mounted` is still False and the whole subtree is queryable.

        Args:
            run: The run the mirrored detail belongs to, or `None` when the
                selection was cleared. A selection that moved on while the
                query was in flight discards the result rather than
                attributing one run's items to another.
        """
        if not self._dom_is_live:
            return
        try:
            runs_pane = self.query_one("#watchlists-runs-pane", RunsPane)
        except Exception:
            return
        current = runs_pane.selected_run
        if run is None:
            if current is not None:
                return
        elif current is None or str(current.get("id")) != str(run.get("id")):
            return
        runs_pane.run_items = self._run_detail_items
        runs_pane.run_logs = self._run_detail_logs
        runs_pane.run_items_note = self._run_detail_items_note

    def watch_selected_run(self, run: dict[str, Any] | None) -> None:
        """Drop the mirrored run detail the moment the selection moves.

        TASK-2306. `_run_detail_items`/`_run_detail_logs` describe ONE run, and
        three paths clear `selected_run` without going near the loader
        (`_apply_tree_scope`, the backend switch, `_delete_run` -- the last two
        then call `_reseed_live_detail_pane`, which would otherwise re-push the
        departed run's items into the pane that had just correctly cleared
        them). One watcher on the field the mirror is keyed to owns the
        invariant, rather than three call sites remembering it.

        Args:
            run: The newly selected run, or `None`.
        """
        self._run_detail_items = []
        self._run_detail_logs = ""
        self._run_detail_items_note = ""

    @staticmethod
    def _run_log_text(run: Mapping[str, Any]) -> str:
        """What the Logs sub-region shows for `run`.

        A run that recorded no log at all is not the same as one whose log is
        empty, and "" renders identically to "never ran" -- so the absence is
        said out loud instead of being drawn as a blank box.
        """
        log_text = run.get("log_text")
        if log_text:
            return str(log_text)
        error_msg = run.get("error_msg")
        if error_msg:
            return str(error_msg)
        return "No log was recorded for this run."

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
        self._source_create_draft_active = True
        self._source_create_draft_frequency = DEFAULT_SOURCE_FREQUENCY_SECONDS
        # Back to "untouched", so the next create form is prefilled again
        # rather than inheriting the selectors of the source just submitted.
        self._source_create_draft_selectors = None
        # Same, for the type and the destination (TASK-2302): the next form
        # opens at the default feed type and at whatever scope is current
        # THEN, not at the one this submission happened to use.
        self._source_create_draft_type = None
        self._source_create_draft_destination = None
        self.run_worker(
            self._create_source(
                event.payload,
                runtime_backend=event.runtime_backend,
            ),
            exclusive=True,
            group="wc_create_source",
        )

    async def _create_source(
        self, payload: dict[str, Any], *, runtime_backend: str
    ) -> None:
        # TASK-2302: the destination is not part of the source record -- it
        # is a membership row -- so it is lifted out before the payload
        # reaches a backend that has no column for it.
        payload = dict(payload)
        watchlist_id = payload.pop("watchlist_id", None)
        try:
            created = await self._controller.create_source(
                runtime_backend=runtime_backend,
                payload=payload,
            )
            destination = self._file_created_source(
                created,
                watchlist_id,
                runtime_backend=runtime_backend,
            )
            # Review wave, M3. The statement is true either way -- that IS
            # where the source is -- but a destination the user chose and did
            # not get is news, not routine. `warning` on the degraded branch
            # keeps the "the toast cannot lie" property and adds the one bit
            # it was missing: that something did not go to plan.
            degraded = watchlist_id is not None and destination == "Unassigned"
            # markup=False: the destination is a user-typed watchlist name.
            self._notify_watchlists(
                f"Source created in {destination}."
                + (
                    " The watchlist you chose could not be used."
                    if degraded
                    else ""
                ),
                severity="warning" if degraded else "information",
                markup=False,
            )
        except Exception:
            logger.opt(exception=True).warning("Failed to create source.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to create source.", severity="error")
        # Reload every view derived from the source list, not just the two
        # that happened to be here (TASK-1040). `_refresh_local_wc_snapshot`
        # feeds the staging line and `_refresh_overview_data` the cards, but
        # `#sources-table` and the rail's counts read their own queries — so
        # without these the table kept the previous list and the rail said
        # `All sources  0` while the centre said `Feeds in All sources (1)`
        # (then Feeds, now the header summary),
        # describing the same thing on one screen.
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()
        self.run_worker(self._load_sources(), exclusive=True, group="wc_sources")
        self._load_tree_data()

    def _file_created_source(
        self,
        created: Mapping[str, Any] | None,
        watchlist_id: Any,
        *,
        runtime_backend: str,
    ) -> str:
        """Write the new source's membership row and name where it landed.

        TASK-2302 AC#2. The confirmation is derived from what this method
        actually did, not from what the form asked for: a destination that
        could not be honoured (no bundle service, a source the backend gave
        no local id) reports Unassigned, which is where the source really is.
        A toast claiming a watchlist the source is not in would be the exact
        defect this task exists to remove, restated as a lie instead of a
        silence.

        Args:
            created: The normalized row `create_source` returned.
            watchlist_id: The chosen watchlist id, or None for Unassigned.
            runtime_backend: Backend captured when Create was pressed.

        Returns:
            The destination as it should be named to the user -- a quoted
            watchlist name, or `Unassigned`.
        """
        unassigned = "Unassigned"
        if watchlist_id is None:
            return unassigned
        service = self._watchlist_bundle_service()
        if (
            service is None
            or self._tree_write_disabled_reason(
                runtime_backend=runtime_backend
            )
            is not None
        ):
            return unassigned
        # The raw local subscription id, not the namespaced `id`
        # (`local:subscription:5`) -- membership rows key on the former, the
        # same distinction `_resume_source` documents.
        source_id = (created or {}).get("source_id")
        if source_id is None:
            logger.warning(
                "Created source carries no local id; leaving it unassigned."
            )
            return unassigned
        try:
            service.add_source(int(watchlist_id), int(source_id))
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to add the new source to its watchlist."
            )
            return unassigned
        return f'"{self._watchlist_display_name(int(watchlist_id))}"'

    @on(CancelRunRequested)
    def handle_cancel_run_requested(self, event: CancelRunRequested) -> None:
        event.stop()
        self.run_worker(
            self._cancel_run(event.run_id),
            exclusive=True,
            group="wc_cancel_run",
        )

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
            logger.opt(exception=True).warning("Failed to cancel run.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to cancel run.", severity="error")
        self._refresh_overview_data()

    @on(RerunRunRequested)
    def handle_rerun_run_requested(self, event: RerunRunRequested) -> None:
        event.stop()
        # A coroutine worker, never thread=True — this launches a check, so
        # the in-flight guard's single-loop invariant applies (see
        # `handle_check_now_requested`'s launch site).
        self.run_worker(
            self._rerun_run(event.source_id),
            exclusive=True,
            group="wc_rerun_run",
        )

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
            logger.opt(exception=True).warning("Failed to launch run.")
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
        self.run_worker(
            self._preview_source(entity),
            exclusive=True,
            group="wc_preview_source",
        )

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
            logger.opt(exception=True).warning("Failed to preview source.")
            if callable(notify):
                notify("Failed to preview source.", severity="error")

    @staticmethod
    def _check_now_entity_name(entity: dict[str, Any]) -> str:
        """The name a Check now toast should use for `entity`.

        Matches the title fallback chain `_build_inspector_pane`'s
        `Selected: {title}` line and `_resume_source`'s `name` already use,
        so a source is never referred to by three different fallbacks across
        three toasts.
        """
        return str(
            entity.get("name")
            or entity.get("source_title")
            or entity.get("title")
            or "that source"
        )

    def _set_check_now_busy(self) -> None:
        """Paint `_checks_in_flight` onto whichever Check-now buttons exist.

        TASK-2309. `_checks_in_flight` is the source of truth; this only
        pushes it onto panes that happen to be mounted right now -- Sources
        and the Inspector each host their own copy of this button, and
        either, both, or neither may be on screen for a given source at a
        given moment (the active section may not be Sources, or the
        Inspector's deepest selection may be a different source or none at
        all). A pane that is not currently mounted needs nothing done to it
        here: `_build_detail_pane`/`_build_inspector_pane` re-seed
        `busy_source_ids` from this same set on every rebuild, so a freshly
        constructed pane never has to be told separately.

        `_dom_is_live`, not `is_mounted` (TASK-2200's mount-window lesson,
        applied throughout this screen): a check can complete inside the
        same `on_mount`-adjacent window that lesson documents.
        """
        if not self._dom_is_live:
            return
        busy_ids = frozenset(self._checks_in_flight)
        try:
            sources_pane = self.query_one("#watchlists-sources-pane", SourcesPane)
        except Exception:
            pass
        else:
            sources_pane.busy_source_ids = busy_ids
        try:
            inspector = self.query_one("#watchlists-entity-inspector", InspectorPane)
        except Exception:
            pass
        else:
            inspector.busy_source_ids = busy_ids

    @on(CheckNowRequested)
    def handle_check_now_requested(self, event: CheckNowRequested) -> None:
        """Start a check, unless this exact source already has one running.

        TASK-2309 (UAT F19). Three things a bare `run_worker(..., exclusive=
        True)` did not give: an immediate acknowledgment (the toast below,
        posted before the worker even starts, so the ~5s of dead air the UAT
        measured has SOMETHING on screen from the first frame), a busy state
        that outlives the toast (`_set_check_now_busy`, cleared only in
        `_check_now_source`'s `finally`), and a stated refusal of a second
        press instead of silently queuing (or, under the old
        `exclusive=True`, silently CANCELLING the first run mid-write --
        exactly the unsound cancellation-supersede shape TASK-1541
        documents: a cancelled `execute_run` leaves its row at `running`
        forever). `run_worker` below uses a named group instead, so a second,
        DIFFERENT source's check is unaffected by this one either way.

        Args:
            event: Carries the source entity to check (`event.entity`), or
                `None` if the pane posting it had nothing selected -- see
                `SourcesPane`/`InspectorPane`'s own `CheckNowRequested`
                call sites.
        """
        event.stop()
        entity = event.entity
        if entity is None:
            return
        source_key = str(entity.get("id") or "")
        name = self._check_now_entity_name(entity)
        if source_key and source_key in self._checks_in_flight:
            # Stated, not silent (AC#2): a second press while this exact
            # source is mid-check does not queue a duplicate run.
            self._notify_watchlists(
                f"Already checking {name}.", severity="warning", markup=False
            )
            return
        if source_key:
            self._checks_in_flight.add(source_key)
            self._set_check_now_busy()
        # Immediate acknowledgment (AC#1), posted before the worker starts.
        # `markup=False`: `name` is user-entered (the source's Name field).
        self._notify_watchlists(
            f"Checking {name}...", severity="information", markup=False
        )
        # A COROUTINE worker, never thread=True: the watchlists in-flight
        # guard (`local_watchlists_service._IN_FLIGHT_URL_CHECKS`) is a
        # lock-free set whose safety rests on every check entrant running on
        # the app's one event loop. Moving this off-loop needs a lock there.
        self.run_worker(
            self._check_now_source(entity, source_key, name), group="wc_check_now"
        )

    #: Run statuses that mean the check did not succeed. `execute_run` catches
    #: the fetch error itself and RETURNS a run in one of these states rather
    #: than raising, so a screen that only watched for exceptions reported
    #: success over a feed that had just 404'd (TASK-1090).
    _FAILED_RUN_STATUSES = frozenset({"failed", "error", "errored"})
    #: Statuses meaning the run is over. `check_now` on the server backend
    #: delegates to `launch_run`, which returns `queued`/`running` while the
    #: fetch is still in flight — so "complete" may only be claimed for these.
    _TERMINAL_RUN_STATUSES = frozenset({"completed", "complete", "succeeded", "success"})

    @classmethod
    def _check_failure_message(cls, result: Any) -> str | None:
        """The reason a completed check failed, or None if it succeeded.

        Args:
            result: Whatever `check_now` returned.

        Returns:
            A human-readable reason when the run reports a failed status,
            otherwise None.
        """
        if not isinstance(result, Mapping):
            return None
        if str(result.get("status") or "").lower() not in cls._FAILED_RUN_STATUSES:
            return None
        stats = result.get("stats")
        error_msg = result.get("error_msg")
        if not error_msg and isinstance(stats, Mapping):
            error_msg = stats.get("error_msg")
        return str(error_msg or "the source reported a failed run")

    @staticmethod
    def _check_was_entirely_skipped(result: Any) -> bool:
        """Whether a completed check checked NOTHING because another was running.

        task-16838. The per-(subscription, url) in-flight guard
        (`LocalWatchlistsService._check_url_guarded`) makes a Check Now that
        lands while a scheduled check of the same source is mid-flight skip
        rather than double-check -- the run completes with a `skipped`
        disposition count and nothing else. Without this, that run's toast
        read "Check complete: X — 0 found, 0 new.", which tells the user
        their page was checked and unchanged when it was not checked at all.

        Only an ENTIRELY skipped run qualifies: a `url_list` run that checked
        most URLs and skipped one did real work, and its ordinary completion
        toast stays honest for it (the Runs pane detail carries the per-URL
        skip count).

        Args:
            result: Whatever `check_now` returned.

        Returns:
            True when the run's dispositions show at least one skip and
            zero of everything else.
        """
        if not isinstance(result, Mapping):
            return False
        stats = result.get("stats")
        if not isinstance(stats, Mapping):
            return False
        dispositions = stats.get("dispositions")
        if not isinstance(dispositions, Mapping):
            return False
        try:
            skipped = int(dispositions.get("skipped", 0) or 0)
            others = sum(
                int(value or 0)
                for counter, value in dispositions.items()
                if counter != "skipped"
            )
        except (TypeError, ValueError):
            return False
        return skipped > 0 and others == 0

    async def _check_now_source(
        self,
        source: dict[str, Any],
        source_key: str | None = None,
        name: str | None = None,
    ) -> None:
        """Run a check for one source and report what actually happened.

        `source_key`/`name` default to `None` and are derived from `source`
        when omitted (the same derivation `handle_check_now_requested` uses)
        so a caller that drives this worker directly -- bypassing the
        message handler entirely, an established pattern in this test suite
        (`Tests/UI/test_watchlists_rail_counts_and_scope.py`) -- keeps
        working without knowing about TASK-2309's debounce bookkeeping.
        `handle_check_now_requested` itself always passes both explicitly,
        since it already computed them for the immediate-ack toast and the
        pre-registration in `_checks_in_flight`.

        TASK-1090. This wrapped the whole call in `except Exception`, logged at
        **debug** and showed a transient toast, which is the swallow that hid
        TASK-1100: `Check now` raised `ValueError` on every press, the feature
        was dead, and the only evidence was a debug line and a toast that had
        gone before anyone looked. Three UAT runs called the screen working.

        Two things changed. An unexpected exception is logged at `warning`
        with the source it was checking, and its message is put in front of
        the user instead of a generic "Failed to check source." And a run that
        *completed* failed is now detected: `execute_run` records the failure
        and returns normally, so the old code's `try` succeeded and it said
        "Check now started." over a feed that had just failed to fetch.

        The durable trace lives where it belongs -- `subscriptions.last_error`
        and a `failed` row in `local_watchlist_runs`, both written by the
        service (see `LocalWatchlistsService.record_run_failure`) -- and the
        source list is reloaded here so the Sources table's Status column
        shows it once the toast is gone.

        TASK-2309: every toast below now names `name` (so a running check is
        identifiable when more than one source exists) and is `markup=False`
        (a source's Name field is user-entered free text, and a bracket in it
        must reach the user verbatim rather than being interpreted -- or
        swallowed -- as Rich markup, same reasoning as `_resume_source`'s
        toast). The whole body is now wrapped in `try`/`finally`:
        `source_key` is cleared from `_checks_in_flight` and the busy state
        is repainted off unconditionally, on EVERY exit path including an
        exception this method itself does not expect -- a raise that skipped
        that cleanup would strand both Check-now buttons permanently
        disabled for a source no worker is actually still checking.
        """
        if source_key is None:
            source_key = str(source.get("id") or "")
        if name is None:
            name = self._check_now_entity_name(source)
        notify = getattr(self.app_instance, "notify", None)
        source_id = source.get("id")
        #: Whether this check actually finished and therefore actually produced
        #: (or failed to produce) items. Review wave, Minor 4 -- see the rail
        #: refresh at the end of this method.
        reached_terminal = False
        try:
            try:
                result = await self._controller.check_now(
                    runtime_backend=self.runtime_backend,
                    source_id=source_id,
                )
            except Exception as exc:
                logger.opt(exception=True).warning(
                    f"Check now failed for watchlist source {source_id!r}: {exc}"
                )
                if callable(notify):
                    notify(
                        f"Check failed: {name} — {exc}",
                        severity="error",
                        timeout=10,
                        markup=False,
                    )
            else:
                failure = self._check_failure_message(result)
                if failure is not None:
                    logger.warning(
                        f"Check now for watchlist source {source_id!r} finished "
                        f"failed: {failure}"
                    )
                    if callable(notify):
                        notify(
                            f"Check failed: {name} — {failure}",
                            severity="error",
                            timeout=10,
                            markup=False,
                        )
                else:
                    # Only claim completion for a terminal status. `check_now`
                    # on the server backend delegates to `launch_run`, which
                    # triggers execution asynchronously and returns
                    # `queued`/`running` — so a fixed "Check complete." would
                    # tell the user the fetch had finished while it was still
                    # in flight (Qodo #4 on PR #1047).
                    status = str((result or {}).get("status") or "").lower()
                    reached_terminal = status in self._TERMINAL_RUN_STATUSES
                    if callable(notify):
                        if reached_terminal and self._check_was_entirely_skipped(
                            result
                        ):
                            # task-16838: the in-flight guard skipped every
                            # URL -- say so rather than "0 found, 0 new",
                            # which would claim the page was checked and
                            # unchanged when it was not checked at all.
                            notify(
                                f"Check skipped: {name} — a check of this "
                                f"source is already running.",
                                severity="warning",
                                markup=False,
                            )
                        elif reached_terminal:
                            # The run's own counters (TASK-2309), when the
                            # result actually carries them -- the local
                            # backend's `execute_run` always returns a
                            # normalized run row with both; a completed
                            # result missing them degrades to the bare
                            # completion line rather than printing "None".
                            found = (result or {}).get("found_count")
                            processed = (result or {}).get("processed_count")
                            if found is not None or processed is not None:
                                notify(
                                    f"Check complete: {name} — "
                                    f"{found or 0} found, {processed or 0} new.",
                                    severity="information",
                                    markup=False,
                                )
                            else:
                                notify(
                                    f"Check complete: {name}.",
                                    severity="information",
                                    markup=False,
                                )
                        else:
                            notify(
                                f"Check started: {name}.",
                                severity="information",
                                markup=False,
                            )
            self._refresh_local_wc_snapshot()
            self._refresh_overview_data()
            # Reload the source list so the Status and Last scraped columns carry
            # the outcome after the toast has gone (AC#2). Same reload
            # `_delete_source` performs for the same reason.
            # Preserve the user's selection across the reload. Rebuilding the
            # table emits a row-0 highlight, which `SourcesPane` treats as a real
            # selection — so without this the reload silently retargets Preview /
            # Check now / Delete at the first source (Qodo #3 on PR #1047, and
            # the defect filed as task-1161).
            self.run_worker(
                self._load_sources_preserving_selection(), exclusive=True, group="wc_sources"
            )
            # TASK-2304 AC#1. A check is the ONE gesture that manufactures items,
            # and the rail's numbers are unread item counts -- so this was the
            # single most visible place they went stale. Measured in the
            # 2026-08-04 UAT: create a watchlist, assign a source, press Check
            # now, watch a feed's worth of items arrive in the centre while every
            # rail count stayed on 0 until the screen was left and re-entered.
            # `_load_tree_data` publishes through TASK-2200's surface-refresh
            # drain, so this is a rail rebuild, not a screen recompose.
            #
            # Review wave, Minor 4: only once the run has actually FINISHED. The
            # local backend runs `check_now` to completion and returns
            # `completed`, so this fires exactly as before there. The server
            # backend delegates to `launch_run` and returns `queued`/`running`
            # (the toast three lines up is careful about the same distinction),
            # so re-reading the counts here would read them before the items it
            # is meant to be reporting exist -- an authoritative-looking query
            # against a state the user's own action has not reached yet. A run
            # that finishes later is picked up by the next refresh, which is the
            # same guarantee every other server-backend surface on this screen
            # gives.
            if reached_terminal:
                self._load_tree_data()
        finally:
            # TASK-2309 AC#1 (the failure case): reached whether the `try`
            # above returned normally, notified a failure, or raised
            # something neither branch anticipated.
            if source_key:
                self._checks_in_flight.discard(source_key)
            self._set_check_now_busy()

    async def _load_sources_preserving_selection(self) -> None:
        """Reload the source list without discarding the current selection."""
        keep = self.selected_source
        await self._load_sources()
        if keep is None or not self.is_mounted:
            return
        keep_id = keep.get("id")
        if keep_id is None:
            return
        try:
            pane = self.query_one("#watchlists-sources-pane", SourcesPane)
        except Exception:
            return
        if any(str(s.get("id")) == str(keep_id) for s in (pane.sources or [])):
            pane.select_source_by_id(str(keep_id))

    @on(ResumeSourceRequested)
    def handle_resume_source_requested(self, event: ResumeSourceRequested) -> None:
        """Dispatch a Resume press to `_resume_source` on a worker (task-2050).

        Refuses any entity that is not a LOCAL `subscription` (task-2050
        Qodo): `resume_source` takes a raw local db id, so a message carrying
        some other entity kind that happens to hold a numeric `source_id`
        (e.g. a server `watchlist_source`) would reset the counters of
        whatever unrelated LOCAL subscription shares that number. The
        Inspector's render gate already makes that unreachable today; this
        guard keeps it unreachable from any future caller too. Type-only log,
        no toast -- a refused programmatic message is not a user action.
        """
        event.stop()
        entity = event.entity
        if entity is None:
            return
        if (
            str(entity.get("backend") or "") != "local"
            or str(entity.get("entity_kind") or "") != "subscription"
        ):
            logger.warning(
                "ResumeSourceRequested for a non-local-subscription entity "
                f"(backend={entity.get('backend')!r}, "
                f"kind={entity.get('entity_kind')!r}); refusing."
            )
            return
        self.run_worker(
            self._resume_source(entity),
            exclusive=True,
            group="wc_resume_source",
        )

    async def _resume_source(self, source: dict[str, Any]) -> None:
        """Clear an auto-paused source's pause via the real service (AC#2/#3).

        Local-only, the same reason `_open_snapshot_view`'s `url_snapshots`
        lookup reaches `_local_watchlists_service()` directly rather than
        through `self._controller` (`WatchlistsBackendController`): only a
        local subscription currently carries a pause concept at all (see
        `normalize_server_watchlist_source`, which always stamps `paused:
        False`), so the controller -- which exists to route a call to
        whichever of local/server is active -- has no reason to gain a
        method for a concept the server backend does not have.

        `source["source_id"]` (the subscription's raw db id) is read rather
        than `source["id"]` (the namespaced `local:subscription:5` form
        `self._controller` calls take): there is no routing layer here to
        parse that namespacing back off, so the raw id
        `normalize_local_subscription_row` already carries under
        `source_id` is used directly, matching how `LocalWatchlistsService`
        itself takes ids everywhere.
        """
        service = self._local_watchlists_service()
        source_id = source.get("source_id")
        name = (
            source.get("name")
            or source.get("source_title")
            or source.get("title")
            or "the source"
        )
        if service is None or source_id is None:
            self._notify_watchlists(WC_SERVICE_UNAVAILABLE_COPY, severity="error")
            return
        try:
            await service.resume_source(source_id)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to resume watchlist source {source_id!r}."
            )
            self._notify_watchlists(
                "Could not resume that source. Check the logs and try again.",
                severity="error",
                markup=False,
            )
            return
        # markup=False: the name is user-entered (Create Source's Name
        # field), same reasoning as every other toast on this screen that
        # embeds one -- a bracket in a source name must reach the user
        # verbatim rather than being interpreted (or swallowed) as Rich
        # markup.
        self._notify_watchlists(
            f"Resumed {name}. It will be checked on its normal schedule.",
            severity="information",
            markup=False,
        )
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()
        # Reload preserving selection, same as `_check_now_source`: the row
        # stays selected, but the Sources table's Status column -- and the
        # Inspector's own `paused` flag, which decides whether this very
        # Resume button renders -- both pick up the cleared pause once the
        # reload lands.
        self.run_worker(
            self._load_sources_preserving_selection(), exclusive=True, group="wc_sources"
        )

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
            if callable(notify):
                notify(_opml_import_summary_text(result), severity="information")
        except Exception:
            logger.opt(exception=True).warning("Failed to import OPML.")
            if callable(notify):
                notify("Failed to import OPML.", severity="error")
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()

    @on(ExportOpmlRequested)
    def handle_export_opml_requested(self, event: ExportOpmlRequested) -> None:
        event.stop()
        self.run_worker(self._export_opml(), exclusive=True, group="wc_export_opml")

    async def _export_opml(self) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            xml_text = await self._controller.export_opml(
                runtime_backend=self.runtime_backend,
            )
            self.app.push_screen(OpmlExportDialog(xml_text))
        except Exception:
            logger.opt(exception=True).warning("Failed to export OPML.")
            if callable(notify):
                notify("Failed to export OPML.", severity="error")

    @on(StageInConsoleRequested)
    def handle_stage_in_console_requested(self, event: StageInConsoleRequested) -> None:
        event.stop()
        self._console_handoff.handle_stage_in_console_requested()

    def scoped_loaded_sources(self) -> list[dict[str, Any]]:
        """`_loaded_sources`, narrowed to the sources the tree scope covers.

        TASK-2304 AC#2. `_load_sources` lists every source the backend holds,
        with no scope predicate at all, so the Sources table ignored the tree
        entirely: the 2026-08-04 UAT selected a watchlist whose own header
        read "AI Research News (0 sources)" and the table underneath it still
        listed an Unassigned source. Two counts of the same fact, on one
        screen, disagreeing -- because only one of them was scoped.

        Resolved through `scoped_source_rows()`, which is the SAME resolver
        `_staging_summary_line` counts, so the header and the table cannot
        drift by construction; making the table agree by re-deriving the
        scope some other way would just create a third answer. That call costs one query, and the `all` scope short
        -circuits before paying it -- it is the default, and its answer is
        "everything" regardless.

        `_loaded_sources` itself stays UNSCOPED. It is the screen's mirror of
        the backend listing, re-seeded into a freshly built `SourcesPane` on
        every workbench rebuild, and the Console handoff reads it too;
        narrowing the mirror would make the scope sticky in places that never
        asked about it. The scope is applied at each push instead, which is
        also what lets `watch_tree_scope` re-push without re-querying the
        backend.

        Review wave, Minor 3: LOCAL backend only, and it says so rather than
        guessing. `scoped_source_rows()` resolves ids through the local
        `WatchlistBundleService` (the watchlists/watchlist_sources tables live
        only in the local database), while a server row's `source_id` is a
        SERVER id from a different namespace entirely
        (`normalize_server_watchlist_source`). Intersecting the two yields the
        empty set for every non-`all` scope, so scoping under the server
        backend would have emptied the table beneath a header claiming N
        sources -- the exact defect this method exists to remove, produced by
        the fix for it. There is no server-side membership query to scope
        with, so the honest answer is to leave the listing unscoped there; the
        rail's write verbs are already disabled on that backend for the same
        underlying reason (`_tree_write_disabled_reason`).

        Returns:
            The subset of `_loaded_sources` in the current tree scope, in the
            backend listing's own order. Every loaded source when the scope is
            `all`, or when the runtime backend is not `local`.
        """
        if self.tree_scope.kind in ("all", "starred", "unread", "today") or self.runtime_backend != "local":
            # The smart feeds scope the ITEMS list (a flag/status/date
            # predicate), not the Sources table -- every source can hold an
            # unread/starred/today item, so the truthful listing here is the
            # same unscoped one `all` gets.
            return list(self._loaded_sources)
        allowed = {
            str(row.get("id")) for row in self.scoped_source_rows() if row.get("id") is not None
        }
        return [
            source
            for source in self._loaded_sources
            # `source_id` is the bare local row id `normalize_local_
            # subscription_row` carries alongside the namespaced `id`
            # ("local:subscription:7"), and it is what the bundle service's
            # row dicts hold -- comparing against `id` would match nothing.
            if str(source.get("source_id")) in allowed
        ]

    def _push_scoped_sources_to_pane(self) -> None:
        """Push the scoped source rows into the mounted `SourcesPane`.

        In place, on the pane's own reactive -- never a region rebuild. The
        Sources pane lives in ITEMS, which TASK-2200 deliberately excludes
        from the surface-refresh drain precisely so an in-flight create form
        is not torn down by something happening elsewhere on the screen; a
        scope change must not become the exception to that.
        """
        if not self._dom_is_live:
            return
        try:
            sources_pane = self.query_one("#watchlists-sources-pane", SourcesPane)
        except NoMatches:
            return
        sources_pane.sources = self.scoped_loaded_sources()
        # TASK-2302, and pushed from exactly here for the same reason the
        # rows are: this method is called from `_apply_tree_data_to_live_
        # surfaces` (a watchlist was created, renamed or deleted) and from
        # `watch_tree_scope` (the user moved), which are the only two events
        # that can change either the destination CHOICES or the default. Both
        # are plain reactive assignments -- an open create form is not
        # rebuilt, so a half-typed draft is untouched; its Select re-reads
        # these on the next compose, which the next open supplies.
        sources_pane.watchlist_choices = self._create_form_watchlist_choices()
        sources_pane.default_destination = self._scope_default_destination()

    async def _load_sources(self) -> None:
        notify = getattr(self.app_instance, "notify", None)
        try:
            sources = await self._controller.list_sources(
                runtime_backend=self.runtime_backend,
                limit=100,
            )
            # Mirror to screen state (Finding 2, fix round 2) so a later
            # workbench rebuild — any region collapse/expand, not
            # just a fresh section switch — can re-seed a brand new
            # SourcesPane instead of leaving its table empty; see
            # `_build_detail_pane` and `_loaded_sources` in __init__.
            self._loaded_sources = [dict(source) for source in sources]
            if self._dom_is_live:
                try:
                    sources_pane = self.query_one("#watchlists-sources-pane", SourcesPane)
                    sources_pane.sources = self.scoped_loaded_sources()
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
            if self._dom_is_live:
                try:
                    runs_pane = self.query_one("#watchlists-runs-pane", RunsPane)
                    runs_pane.runs = self._loaded_runs
                    if had_pending_target:
                        runs_pane.selected_run = requested_run
                except Exception:
                    pass
            if had_pending_target and requested_run is not None:
                # TASK-2306. The deep link cannot rely on `RunSelected` to
                # trigger the detail load the way a click does: the pane only
                # posts that message `if self.is_mounted`, and this loader is
                # started by `on_mount` -- inside the window where
                # `is_mounted` is still False (TASK-2200). Awaited in this
                # worker rather than started as another so the ordering is
                # the same one the assertions can observe.
                await self._load_run_detail(requested_run)
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
        if not self._dom_is_live:
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
        self.run_worker(
            self._load_notifications(),
            exclusive=True,
            group="wc_notifications",
        )

    @on(MarkNotificationReadRequested)
    def handle_mark_notification_read_requested(
        self, event: MarkNotificationReadRequested
    ) -> None:
        event.stop()
        self.run_worker(
            self._mark_notification_read(event.notification_id),
            exclusive=True,
            group="wc_mark_notification_read",
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
            self._dismiss_notification(event.notification_id),
            exclusive=True,
            group="wc_dismiss_notification",
        )

    async def _dismiss_notification(self, notification_id: int) -> None:
        dismissed = await self._notifications_controller.dismiss(
            notification_id, is_dismissed=True
        )
        if dismissed:
            await self._load_notifications()

    # --- Artifacts: the briefings a watchlist has produced -----------------
    #
    # Spec #2 phase 1, task 4. Briefings are per-watchlist by schema
    # (`briefings.watchlist_id` is NOT NULL), local by construction (they are
    # written into this device's `SubscriptionsDB`, whatever the Backend
    # selector says), and generated only on request.

    def _briefings_db(self) -> Any:
        """The local `SubscriptionsDB` briefings live in, or `None`.

        Reached through `WatchlistBundleService` rather than a second
        accessor onto the database, which is the rule `_load_tree_data`
        already states for this screen; degrades to `None` in harnesses
        where the service is not wired.
        """
        service = self._watchlist_bundle_service()
        return getattr(service, "db", None) if service is not None else None

    def _briefing_watchlist_id(self) -> int | None:
        """The watchlist Artifacts is scoped to, or `None`.

        `tree_scope` rather than `selected_scope`: this is a question about
        what the user is looking at in the rail, which is exactly the split
        those two reactives exist to keep (see their declarations). A
        "source" scope deliberately does not answer it -- one source can sit
        in several watchlists, and briefings belong to exactly one.
        """
        scope = self.tree_scope
        if scope is not None and scope.kind == "watchlist":
            return scope.watchlist_id
        return None

    def _can_generate_briefing(self) -> bool:
        """Whether Generate has both a store and a watchlist to act on."""
        return self._briefings_db() is not None and (
            self._briefing_watchlist_id() is not None
        )

    def _briefing_provider_display(self) -> str:
        """TASK-2311, AC#3: the provider Generate will actually use, for
        display BEFORE the user presses it.

        UAT: Generate silently used whatever `default_briefing_provider()`
        resolved to (openai, in that run) with no indication anywhere in
        the UI of what provider was about to be charged/attempted. Mirrors
        `generate_briefing`'s own resolution order exactly (explicit
        `provider` arg is never used by this screen's call site, so it is
        skipped here): the watchlist's stored default preset's provider,
        else the app's configured default endpoint.
        """
        from ...Chat.provider_catalog import provider_display_name

        preset_id = self._briefing_default_preset_id
        provider_key = ""
        if preset_id is not None:
            for preset in self._loaded_briefing_presets:
                if preset.get("id") == preset_id:
                    provider_key = str(preset.get("provider") or "")
                    break
        if not provider_key:
            provider_key = default_briefing_provider()
        return provider_display_name(provider_key)

    #: UAT batch-5 review, m3. `briefing_service._error_text` already caps
    #: at 1000 chars server-side and never a traceback, but this task made
    #: that text fire in an UNCONDITIONAL toast at failure time rather than
    #: requiring a click into the row's detail region -- a wider default
    #: exposure than before, even though nothing here is a NEW leak. A
    #: toast is a one-line surface; 1000 chars (or any embedded newline,
    #: the shape a raw HTTP response body or header dump takes, unlike a
    #: curated one-line provider message such as "OpenAI API Key is
    #: required but not found.") is far more than it should ever carry.
    _MAX_BRIEFING_FAILURE_REASON_CHARS = 160

    @classmethod
    def _bounded_briefing_failure_reason(cls, raw_reason: str) -> str:
        """Bound/shape a provider's own failure text before an unconditional
        toast (review finding m3): collapse embedded newlines/whitespace
        first -- a multi-line dump must not survive as a single
        deceptively-long "clean" line -- then cap the result to a toast-
        appropriate length. Every sampled curated provider message (e.g.
        "OpenAI API Key is required but not found.") is short and single-
        line and passes through unchanged; anything shaped like a raw
        payload or header dump is truncated or, for pathological input,
        replaced entirely, so a bug elsewhere in the ~9 bridged provider
        handlers cannot turn this into a payload sink.
        """
        collapsed = " ".join(raw_reason.split())
        if not collapsed:
            return "no reason was recorded"
        if len(collapsed) > cls._MAX_BRIEFING_FAILURE_REASON_CHARS:
            collapsed = collapsed[: cls._MAX_BRIEFING_FAILURE_REASON_CHARS].rstrip() + "…"
        return collapsed

    def _notify_briefing_failure(self, row: Mapping[str, Any]) -> None:
        """TASK-2311: surface a failed generation's reason at failure time,
        without requiring the user to click the row.

        `row`'s `model_used`/`error` are `briefing_service._finish_failure`'s
        own record of what actually happened -- read directly rather than
        recomputed, so this always names the provider that was ACTUALLY
        attempted (which can differ from `_briefing_provider_display()`'s
        prospective answer if the default preset changed mid-flight), and
        the provider's own failure text. `markup=False`: a provider's error
        message is untrusted, remote-influenced text. The reason itself is
        bound/shaped by `_bounded_briefing_failure_reason` (review finding
        m3) before it ever reaches the toast -- the full, uncapped text
        (still capped at 1000 chars server-side, never a traceback) remains
        available by clicking into the row's own detail region, unchanged
        from before this task.
        """
        from ...Chat.provider_catalog import provider_display_name

        model_used = str(row.get("model_used") or "")
        provider_key = model_used.split("/", 1)[0] if model_used else ""
        provider = (
            provider_display_name(provider_key)
            if provider_key
            else self._briefing_provider_display()
        )
        reason = self._bounded_briefing_failure_reason(str(row.get("error") or ""))
        # Live-verified trap: the provider's own message is very often
        # already a full sentence with its own trailing period (e.g.
        # "OpenAI API Key is required but not found.") -- appending another
        # unconditionally produced a visible ".." in the toast. One strip.
        reason = reason.rstrip(".")
        self._notify_watchlists(
            f"Briefing generation failed using {provider}: {reason}. Check "
            "Settings ▸ Providers & Models, then press Generate again.",
            severity="error",
            markup=False,
        )

    def _chachanotes_db(self) -> Any:
        """The live ChaChaNotes handle, or `None` (task-1780, Task 5).

        `getattr(self.app_instance, "chachanotes_db", None)` -- the exact
        idiom `_load_character_options`/`_cast_load_character` already use
        on this screen -- pulled out into its own accessor now that a
        THIRD caller (Keep/`KeptBriefingsModal`'s opener) needs the
        identical read. Degrades to `None` in harnesses where the app
        instance carries no such attribute at all.
        """
        return getattr(self.app_instance, "chachanotes_db", None)

    def _local_watchlists_service(self) -> Any:
        """The live `LocalWatchlistsService`, or `None` (TASK-1494).

        `url_snapshots` is local-only storage -- only `URLMonitor.check_url`
        (the local monitoring engine) ever writes it, and there is no
        server-backend equivalent to route to -- so the reader's snapshot
        viewer reaches this service directly, the same `getattr(self.
        app_instance, ..., None)` idiom `_watchlist_bundle_service`/`_
        chachanotes_db` above use, rather than through `self._controller`
        (`WatchlistsBackendController`), which exists to route a call to
        whichever of local/server is active and has no reason to gain a
        local-only method just for this one read.
        """
        return getattr(self.app_instance, "local_watchlists_service", None)

    def _briefing_schedules_enabled(self) -> bool:
        """Whether `[scheduling] briefing_schedules_enabled` is on for this
        run (task-1812, AC #1).

        The identical config read `app.py`'s `_wire_watchlists_and_
        notifications_services` uses to decide whether to build a
        `BriefingProjection`/`BriefingJobHandler` pair at all -- read
        directly via `get_cli_setting` (the "queries config helper"
        convention several other screens/windows already use, e.g.
        `Tools_Settings_Window`/`STTS_Window`) rather than through a live
        handle on `self.app_instance`, even though one now exists:
        `self.app_instance.scheduling_service.briefing_projection is not
        None` is `app.py`'s own live reflection of this identical decision
        (non-`None` iff the flag is truthy), added by task-1810 -- the
        first commit on this same branch, two commits before this function
        -- and asserted by `Tests/Scheduling/test_scheduling_service.py::
        test_app_wiring_briefing_projection_is_live_not_a_frozen_none`. Kept
        as a direct config read rather than switched to that mirror: today
        both resolve identically, but a config reload mid-run (or a future
        UI control this docstring anticipated) could make them diverge, so
        a caller that cares about liveness rather than configuration should
        read the mirror instead. Defaults to `True`, matching `app.py`'s own
        default, so a watchlist with a stored cadence still reads as
        scheduled unless an operator has explicitly turned the flag off.
        """
        return bool(
            get_cli_setting("scheduling", "briefing_schedules_enabled", True)
        )

    def _briefing_scope_label(self) -> str:
        """The pane's one-line statement of what it is showing, and from where."""
        watchlist_id = self._briefing_watchlist_id()
        if watchlist_id is None:
            return (
                "Select a watchlist in the rail to see or write its briefings — "
                "a briefing covers one watchlist."
            )
        name = self._watchlist_display_name(watchlist_id)
        # Spec #2 phase 4, Task 4: this used to always say "written on this
        # device, on request" -- true in phase 1, when nothing could write
        # `briefing_cadence_seconds`, but a lie the moment Task 2 gave that
        # column a writer. `cadence_scope_phrase` answers `None` (never
        # scheduled) with `None`, so "on request" stays the honest default;
        # anything else names the actual cadence, "while the app is open"
        # and all -- see that function's own docstring for why the phrase
        # is worded that way. `schedules_enabled` (task-1812, AC #1) closes
        # a second honesty gap the same reasoning missed: with the app-level
        # kill switch off, nothing in this process would ever read a stored
        # cadence back, so a phrase implying an active schedule would be a
        # lie of the identical shape.
        cadence_phrase = cadence_scope_phrase(
            self._briefing_cadence_seconds,
            schedules_enabled=self._briefing_schedules_enabled(),
        )
        provenance = (
            f"written on this device — {cadence_phrase}"
            if cadence_phrase is not None
            else "written on this device, on request"
        )
        # RAW, deliberately: the pane wraps this in a `rich.text.Text`, which
        # is never markup-parsed, so escaping here would put visible
        # backslashes in front of every bracket a real name contains. See
        # `ArtifactsPane.compose` for why that wrapper is load-bearing --
        # a bare `str` in a `Static` IS parsed as markup.
        return f"Briefings for {name} · {provenance}"

    async def _load_briefings(
        self, *, select_briefing_id: int | None = None
    ) -> None:
        """Re-read this watchlist's briefings and repaint the pane.

        Repaints the PANE, never the screen: `self.refresh(recompose=True)`
        would rebuild every region through its factory and hand any live
        reference a defunct widget (the Phase D lesson `watch_tree_scope`
        records). Pushing the rows into the mounted `ArtifactsPane` lets the
        pane's own `recompose=True` reactives rebuild just its children,
        with the pane instance itself surviving.

        Args:
            select_briefing_id: Select this row after loading -- used by the
                generation worker so a finished briefing is the one on
                screen. Otherwise the current selection is re-resolved
                against the reloaded rows and dropped if it is gone.
        """
        db = self._briefings_db()
        watchlist_id = self._briefing_watchlist_id()
        if db is None or watchlist_id is None:
            self._loaded_briefings = []
            self._selected_briefing = None
            self._briefing_selection_mode = MODE_AUTO_FEATURED
            self._briefing_default_preset_id = None
            self._briefing_cadence_seconds = None
            self._loaded_scripts = []
            self._selected_script = None
            self._loaded_script_audio = None
            self._scripts_with_audio = {}
            self._loaded_citations = []
            self._citation_item_lookup = {}
            self._watchlist_has_audio_episodes = False
        else:
            try:
                # Zombie recovery, before the list query, so a row this
                # sweep just failed shows up as failed/interrupted in THIS
                # same load rather than requiring a second one (whole-branch
                # review fix 3). Best-effort: a failure here must not stop
                # the list query below, and must not exit the app -- this
                # runs inside the `wl-briefings-load` worker, whose default
                # `exit_on_error=True` would take the app down with it.
                await self._fail_interrupted_briefings_if_safe(db, watchlist_id)
            except Exception as exc:  # noqa: BLE001 - best-effort, not fatal
                logger.warning(
                    f"Zombie-briefing sweep failed for watchlist {watchlist_id}: "
                    f"{type(exc).__name__}"
                )
            try:
                # `asyncio.to_thread`, not a direct call: this coroutine runs
                # inside the `wl-briefings-load` worker, which `run_worker`
                # only schedules back onto the SAME event loop -- a
                # synchronous `list_briefings` call here would still block
                # the UI thread for the length of the SELECT, same shape as
                # the write `_toggle_briefing_queue` documents.
                rows = await asyncio.to_thread(db.list_briefings, watchlist_id)
                self._loaded_briefings = [dict(row) for row in rows]
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                # Type only, never `logger.opt(exception=True)`: this app's
                # file sink runs with `diagnose=True`, so a traceback here
                # would dump the failing frame's locals -- which on this
                # path are briefing rows, i.e. item-derived content. Same
                # rule `briefing_service` states for its own failure log.
                logger.warning(
                    f"Failed to list briefings for watchlist {watchlist_id}: "
                    f"{type(exc).__name__}"
                )
                self._notify_watchlists(
                    "Failed to read this watchlist's briefings.",
                    severity="error",
                    markup=False,
                )
                self._loaded_briefings = []
            wanted = (
                select_briefing_id
                if select_briefing_id is not None
                else (self._selected_briefing or {}).get("id")
            )
            self._selected_briefing = next(
                (
                    row
                    for row in self._loaded_briefings
                    if wanted is not None and row.get("id") == wanted
                ),
                None,
            )
            # Task 5: the selected briefing's cast scripts. Scoped to ONE
            # briefing (the resolved selection above) rather than every
            # briefing in `self._loaded_briefings` -- a script belongs to
            # exactly one briefing and this pane only ever renders one
            # briefing's detail, so there is nothing to gain from fetching
            # scripts for briefings that are not on screen. This runs
            # inside the SAME `wl-briefings-load` worker as everything else
            # in this method (see `handle_briefing_selected`, which
            # re-dispatches this whole method -- rather than a
            # scripts-only reload -- exactly so a row click's scripts
            # arrive through this one worker/to_thread pattern).
            selected_briefing_id = (self._selected_briefing or {}).get("id")
            if selected_briefing_id is None:
                self._loaded_scripts = []
                self._selected_script = None
                self._loaded_script_audio = None
                self._scripts_with_audio = {}
            else:
                try:
                    # Zombie recovery for scripts, mirroring the briefing
                    # sweep above: a cast worker that crashed mid-cast
                    # leaves a `generating` row that would otherwise wedge
                    # a one-cast-at-a-time guard shut forever. Gated on
                    # `_cast_sweep_is_safe` for the identical reason
                    # `_fail_interrupted_briefings_if_safe` is gated on
                    # `_zombie_sweep_is_safe`: a load racing a cast THIS
                    # screen started must not fail that cast's own row out
                    # from under it.
                    await self._fail_interrupted_scripts_if_safe(
                        db, selected_briefing_id
                    )
                except Exception as exc:  # noqa: BLE001 - best-effort, not fatal
                    logger.warning(
                        "Zombie-script sweep failed for briefing "
                        f"{selected_briefing_id}: {type(exc).__name__}"
                    )
                try:
                    rows = await asyncio.to_thread(
                        db.list_briefing_scripts, selected_briefing_id
                    )
                    self._loaded_scripts = [dict(row) for row in rows]
                except Exception as exc:  # noqa: BLE001 - reported, not raised
                    logger.warning(
                        "Failed to list scripts for briefing "
                        f"{selected_briefing_id}: {type(exc).__name__}"
                    )
                    self._notify_watchlists(
                        "Failed to read this briefing's scripts.",
                        severity="error",
                        markup=False,
                    )
                    self._loaded_scripts = []
                # Review round 1, Minor #4: which of THESE scripts have any
                # audio at all, so the scripts table can show an indicator
                # for every row -- not just the one currently selected
                # (`_loaded_script_audio` below only ever answers that one
                # question for ONE script). One `asyncio.to_thread` hop for
                # the whole set, same "bundle several small reads into one
                # thread-pool round trip" idiom `_read_watchlist_briefing_
                # state` already uses -- `SubscriptionsDB` has no query
                # batched by many script ids at once, and a briefing's own
                # script count is small in practice, so N per-script
                # existence checks inside ONE hop beats N separate hops.
                try:
                    script_ids = [
                        row["id"]
                        for row in self._loaded_scripts
                        if row.get("id") is not None
                    ]
                    self._scripts_with_audio = await asyncio.to_thread(
                        self._read_scripts_with_audio, db, script_ids
                    )
                except Exception as exc:  # noqa: BLE001 - reported, not raised
                    logger.warning(
                        "Failed to read audio presence for briefing "
                        f"{selected_briefing_id}'s scripts: {type(exc).__name__}"
                    )
                    self._scripts_with_audio = {}
                wanted_script = (
                    (self._selected_script or {}).get("id")
                    if self._selected_script
                    else None
                )
                self._selected_script = next(
                    (
                        row
                        for row in self._loaded_scripts
                        if wanted_script is not None and row.get("id") == wanted_script
                    ),
                    None,
                )
            # Task 7: the SELECTED script's newest audio render. Scoped to
            # ONE script (the resolved selection above), the identical
            # shape `_loaded_scripts` uses for ONE briefing above it --
            # audio belongs to exactly one script, and this pane only ever
            # shows one script's detail at a time. Runs inside the SAME
            # `wl-briefings-load` worker/hop as everything else here (see
            # `handle_script_selected`, which re-dispatches this whole
            # method -- rather than an audio-only reload -- exactly so a
            # script row click's audio arrives through this one pattern).
            selected_script_id = (
                (self._selected_script or {}).get("id")
                if self._selected_script
                else None
            )
            if selected_script_id is None:
                self._loaded_script_audio = None
            else:
                try:
                    # Zombie recovery for audio, mirroring the script sweep
                    # above: a synthesis worker that crashed mid-render
                    # leaves a `generating` row that would otherwise wedge
                    # a one-synthesis-at-a-time guard shut forever. Gated
                    # on `_audio_sweep_is_safe` for the identical reason
                    # `_fail_interrupted_scripts_if_safe` is gated on
                    # `_cast_sweep_is_safe`: a load racing a synthesis THIS
                    # screen started must not fail that attempt's own row
                    # out from under it.
                    await self._fail_interrupted_audio_if_safe(
                        db, selected_script_id
                    )
                except Exception as exc:  # noqa: BLE001 - best-effort, not fatal
                    logger.warning(
                        "Zombie-audio sweep failed for script "
                        f"{selected_script_id}: {type(exc).__name__}"
                    )
                try:
                    audio_rows = await asyncio.to_thread(
                        db.list_briefing_audio, selected_script_id, limit=1
                    )
                    self._loaded_script_audio = (
                        dict(audio_rows[0]) if audio_rows else None
                    )
                except Exception as exc:  # noqa: BLE001 - reported, not raised
                    logger.warning(
                        "Failed to read audio for script "
                        f"{selected_script_id}: {type(exc).__name__}"
                    )
                    self._notify_watchlists(
                        "Failed to read this script's audio.",
                        severity="error",
                        markup=False,
                    )
                    self._loaded_script_audio = None
            # Task 6: which items the SELECTED briefing's body actually
            # cites. `extract_citation_ids` is pure and cheap (a regex over
            # a body already held in memory), so it runs inline; only the
            # DB lookup goes through `asyncio.to_thread`, ONE call per
            # selection, inside this SAME worker/hop -- never a second
            # worker group, per the brief. A missing key in the returned
            # dict IS the pruned signal (`SubscriptionsDB.
            # get_subscription_items_by_ids`'s own contract): there is no
            # separate "does this still exist" query.
            citation_ids = extract_citation_ids(
                (self._selected_briefing or {}).get("body_markdown") or ""
            )
            if not citation_ids:
                self._loaded_citations = []
                self._citation_item_lookup = {}
            else:
                try:
                    rows_by_id = await asyncio.to_thread(
                        db.get_subscription_items_by_ids, citation_ids
                    )
                except Exception as exc:  # noqa: BLE001 - reported, not raised
                    logger.warning(
                        "Failed to resolve citations for briefing "
                        f"{selected_briefing_id}: {type(exc).__name__}"
                    )
                    rows_by_id = {}
                citations: list[dict[str, Any]] = []
                lookup: dict[int, dict[str, Any]] = {}
                for item_id in citation_ids:
                    row = rows_by_id.get(item_id)
                    if row is None:
                        # The named invariant: an id that does not resolve
                        # degrades honestly rather than quietly passing as
                        # available. `available=False` and a label that
                        # already says so are BOTH set here -- there is no
                        # follow-up query for `handle_citation_activated`
                        # to get wrong later; the pruned state is decided
                        # once, at resolution time.
                        citations.append(
                            {
                                "item_id": item_id,
                                "label": Text(
                                    f"item {item_id} — no longer available"
                                ),
                                "available": False,
                            }
                        )
                        continue
                    normalized = normalize_watchlist_item("local", row)
                    lookup[item_id] = normalized
                    title = str(normalized.get("title") or "Untitled item")
                    citations.append(
                        {
                            "item_id": item_id,
                            # A remote-authored title, appended into a
                            # `Text` rather than an f-string handed to a
                            # markup-parsing sink -- `Text(...)` never
                            # re-parses its argument, so this is safe for
                            # the identical reason `_script_turns_
                            # renderable` states for a script's own turns.
                            "label": Text(f"[item {item_id}] {title}"),
                            "available": True,
                        }
                    )
                self._loaded_citations = citations
                self._citation_item_lookup = lookup
            # Task 4: the toolbar's pickers. One combined `to_thread` hop
            # for both the watchlist's stored settings (the SAME columns
            # `briefing_service._selection_mode` reads) and the full preset
            # list, rather than two sequential hops -- `_load_briefings`
            # already pays for a zombie sweep and a `list_briefings` read
            # above, and every extra round trip through the thread pool is
            # latency this section's own toolbar adds on top of that,
            # measured to matter under a busy full-suite run. A read
            # failure degrades to the fallback mode, no default preset, and
            # whatever presets were already loaded, rather than aborting
            # the whole load -- the briefing list above is real data this
            # failure must not hide.
            #
            # Task 5 (phase 3): `has_audio_episodes` rides in the SAME hop
            # -- one more cheap read bundled alongside the other two,
            # rather than a fourth separate `to_thread` round trip just for
            # a boolean the Export Feed button needs. A read failure
            # degrades to `False` (button stays disabled) rather than
            # guessing: an export button that might reach a broken query
            # is worse than one that stays honestly disabled until the
            # next successful load.
            try:
                settings_row, preset_rows, has_audio_episodes = await asyncio.to_thread(
                    self._read_watchlist_briefing_state, db, watchlist_id
                )
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                logger.warning(
                    "Failed to read briefing settings/presets for watchlist "
                    f"{watchlist_id}: {type(exc).__name__}"
                )
                settings_row, preset_rows, has_audio_episodes = (
                    {},
                    self._loaded_briefing_presets,
                    False,
                )
            mode = settings_row.get("briefing_selection_mode")
            self._briefing_selection_mode = (
                str(mode) if mode in VALID_MODES else MODE_AUTO_FEATURED
            )
            self._briefing_default_preset_id = settings_row.get(
                "default_briefing_preset_id"
            )
            self._briefing_cadence_seconds = settings_row.get(
                "briefing_cadence_seconds"
            )
            self._loaded_briefing_presets = preset_rows
            self._watchlist_has_audio_episodes = has_audio_episodes
        self._apply_briefing_state_to_pane()

    def _apply_briefing_state_to_pane(self) -> None:
        """Push every screen-held briefing value into the mounted pane.

        Extracted from `_load_briefings`' tail (task-15461) so the section
        swap can re-apply the same state after mounting a fresh
        `ArtifactsPane` -- see `_reseed_active_section_pane` for why that is
        not redundant. Every write is a reactive assignment, so a value that
        has not moved costs nothing: Textual only fires watchers (and the
        recompose that follows) when `!=` says the value actually changed.
        """
        if not self._dom_is_live:
            return
        try:
            pane = self.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        except NoMatches:
            return
        pane.briefings = self._loaded_briefings
        pane.selected_briefing = self._selected_briefing
        pane.scope_label = self._briefing_scope_label()
        pane.can_generate = self._can_generate_briefing()
        pane.default_provider_display = self._briefing_provider_display()
        pane.selection_mode = self._briefing_selection_mode
        pane.presets = self._loaded_briefing_presets
        pane.default_preset_id = self._briefing_default_preset_id
        pane.briefing_cadence_seconds = self._briefing_cadence_seconds
        pane.briefing_schedules_enabled = self._briefing_schedules_enabled()
        pane.scripts = self._loaded_scripts
        pane.selected_script = self._selected_script
        pane.script_audio = self._loaded_script_audio
        pane.scripts_with_audio = self._scripts_with_audio
        pane.citations = self._loaded_citations
        pane.has_audio_episodes = self._watchlist_has_audio_episodes
        pane.chachanotes_available = self._chachanotes_db() is not None

    @staticmethod
    def _read_watchlist_briefing_settings(db: Any, watchlist_id: int) -> dict[str, Any]:
        """The watchlist's stored `briefing_selection_mode`/
        `default_briefing_preset_id`/`briefing_cadence_seconds`, as a plain
        dict.

        Matches `briefing_service._selection_mode`'s own read of the same
        column -- `WatchlistBundleService.list_watchlists`/`_get`
        deliberately select a narrower column list that predates these two
        (Task 1), so there is no existing service-layer getter for them to
        reuse. `briefing_cadence_seconds` (spec #2 phase 4, Task 4) rides
        in the same read: one more column on an already-narrow `WHERE id =
        ?` lookup, not a second query. Reads run inside `with
        db.transaction() as conn:`, not a bare `db.conn.execute` (Qodo
        rule 1011851: every accessor this stream has shipped goes through
        `transaction()`, reads included, so rollback-on-exception is
        consistently wired even for read paths). Always called through
        `asyncio.to_thread`; never call this directly from the UI thread.
        """
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT briefing_selection_mode, default_briefing_preset_id, "
                "briefing_cadence_seconds FROM watchlists WHERE id = ?",
                (watchlist_id,),
            ).fetchone()
        return dict(row) if row is not None else {}

    @staticmethod
    def _read_watchlist_briefing_state(
        db: Any, watchlist_id: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
        """`_read_watchlist_briefing_settings` plus `list_briefing_presets`
        plus whether the watchlist has any export-ready audio, as one
        synchronous unit for `_load_briefings` to dispatch through a
        SINGLE `asyncio.to_thread` call.

        All three are cheap reads on the same thread-local connection;
        bundling them here is purely about round trips through the thread
        pool, not about the SQL itself -- `_load_briefing_presets` still
        does its own separate `list_briefing_presets` call for its OTHER
        caller (`_open_briefing_preset_manager`), which has no settings
        row to read alongside it. Always called through `asyncio.
        to_thread`; never call this directly from the UI thread.

        Returns:
            `(settings_row, preset_rows, has_audio_episodes)`.
        """
        settings_row = WatchlistsCollectionsScreen._read_watchlist_briefing_settings(
            db, watchlist_id
        )
        preset_rows = [dict(row) for row in db.list_briefing_presets()]
        # Task 5 (phase 3): a `limit=1` probe -- the Export Feed button
        # only needs a boolean, not the full episode page `export_feed_
        # directory` itself re-queries at export time, so this avoids
        # fetching a page nothing here reads (CLAUDE.md Performance Rules:
        # paginate DB results).
        has_audio_episodes = bool(
            db.list_watchlist_audio_episodes(watchlist_id, limit=1)
        )
        return settings_row, preset_rows, has_audio_episodes

    @staticmethod
    def _read_scripts_with_audio(db: Any, script_ids: list[int]) -> dict[int, str]:
        """Each of `script_ids` that has at least one `briefing_audio`
        render, mapped to that render's NEWEST status.

        Review round 1, Minor #4: a single read per script
        (`list_briefing_audio(script_id, limit=1)`, newest-first --
        Subscriptions_DB's own `ORDER BY created_at DESC, id DESC`),
        bundled into one synchronous unit for `_load_briefings` to
        dispatch through a SINGLE `asyncio.to_thread` call -- the
        `_read_watchlist_briefing_state` idiom immediately above.
        `SubscriptionsDB` has no query batched by many script ids at once
        (`list_briefing_audio` is scoped to exactly one), and a
        briefing's own cast-script count is small in real use, so N of
        these small reads inside one thread hop beats N separate hops
        through the thread pool. Always called through `asyncio.
        to_thread`; never call this directly from the UI thread.

        Owner decision, task-7 phase 2b follow-up ("if synthesis fails,
        show the audio glyph with a red x"): this used to return a bare
        `frozenset[int]` of "has at least one attempt" (review round 1's
        original ask), which let a failed synthesis paint identically to
        a successful one in the scripts table. `limit=1` already fetches
        the newest render -- reusing its `status` costs nothing extra,
        so `ArtifactsPane._audio_cell` can tell a `STATUS_FAILED` render
        apart from a `STATUS_COMPLETE`/`STATUS_GENERATING` one without a
        second query or a second `to_thread` hop.

        Args:
            db: An open `SubscriptionsDB`.
            script_ids: `briefing_scripts.id` values to check.

        Returns:
            `{script_id: status}` for every id in `script_ids` that has
            at least one `briefing_audio` row, using that row's newest
            status. A `script_id` with no `briefing_audio` row at all is
            simply absent from the mapping.
        """
        result: dict[int, str] = {}
        for script_id in script_ids:
            rows = db.list_briefing_audio(script_id, limit=1)
            if rows:
                result[script_id] = str(rows[0].get("status") or "")
        return result

    @on(BriefingSelected)
    def handle_briefing_selected(self, event: BriefingSelected) -> None:
        """Mirror the pane's selection so a region rebuild can re-seed it.

        Deliberately NOT routed through `_select_entity`, unlike every other
        pane's selection: the Inspector's verbs (Preview, Check now, Ingest,
        Ignore, Delete) are all things you do to a monitored source or the
        items it produced. A briefing is an artifact those verbs cannot act
        on, and handing one to the Inspector would render its fields under
        actions that do not apply to it.
        """
        event.stop()
        self._selected_briefing = event.briefing
        # Task 5: a different briefing means different scripts. Clear the
        # stale selection immediately (synchronously, so nothing renders a
        # PREVIOUS briefing's script against the NEW one even for the one
        # frame before the reload below lands), then reload through
        # `_load_briefings` -- the SAME worker/to_thread batch that method
        # already uses, rather than standing up a second worker group just
        # to fetch one briefing's scripts.
        #
        # Fix round 1, Minor: clearing `self._selected_script` alone (the
        # SCREEN's own rebuild-survival mirror) was not enough -- the
        # mounted pane's OWN `scripts`/`selected_script` reactives are what
        # actually render the scripts table/detail, and those keep
        # whatever the PREVIOUS briefing left in them until `_load_
        # briefings`'s asynchronous reload lands. Without patching the pane
        # directly here too, the old briefing's scripts stay on screen,
        # under the NEW briefing's own detail, for every frame between this
        # click and that reload's completion.
        self._selected_script = None
        self._loaded_scripts = []
        # Task 7: a different briefing means different scripts, which means
        # different audio -- the identical stale-window hazard fixed for
        # scripts above, one level down.
        self._loaded_script_audio = None
        self._scripts_with_audio = {}
        # Task 6: a different briefing also means different citations --
        # the identical stale-window hazard fix round 1 fixed for scripts
        # above, for the identical reason. Without clearing
        # `_citation_item_lookup` too, `handle_citation_activated` could
        # briefly resolve an id against the PREVIOUS briefing's citations
        # in the one frame before `_load_briefings`'s reload lands.
        self._loaded_citations = []
        self._citation_item_lookup = {}
        # task-15461: the PANE's own copies are cleared by
        # `ArtifactsPane._clear_selection_derived_state`, inside the same
        # synchronous instant the selection moves, so the clearing rides the
        # one rebuild that selection already scheduled instead of adding a
        # second (since task-15779 that is the pane's `BriefingDetailRegion`
        # refresh -- the briefings table itself is no longer rebuilt by a
        # selection at all). The screen-side mirrors above still have to be
        # cleared here -- they are what `handle_citation_activated` and a
        # later rebuild read.
        self.run_worker(
            self._load_briefings(), exclusive=True, group="wl-briefings-load"
        )

    @on(ScriptSelected)
    def handle_script_selected(self, event: ScriptSelected) -> None:
        """Mirror the pane's script selection, for rebuild survival --
        the `_selected_briefing`/`handle_briefing_selected` sibling.

        Task 7: a REAL selection (`event.script is not None`) also means
        different audio to fetch -- `_loaded_script_audio` is scoped to
        ONE script, the same way `_loaded_scripts` is scoped to one
        briefing, and (unlike a script's own row, already fully loaded as
        part of `_loaded_scripts`) its audio is not preloaded for every
        script up front, so a newly selected script's audio still needs a
        fresh read. Re-dispatches the whole `_load_briefings` (mirroring
        `handle_briefing_selected`) rather than a narrower audio-only
        worker, for the same one-worker-pattern reason that method's own
        docstring gives.

        Deliberately does NOT re-dispatch when `event.script is None`:
        that case fires either from a genuine deselection or -- far more
        commonly -- as the reactive echo of `handle_briefing_selected`
        clearing this pane's `selected_script` to `None` itself, right
        before dispatching its OWN `_load_briefings()` reload. Re-
        dispatching here too would only be a second, redundant worker
        racing the one already started.
        """
        event.stop()
        self._selected_script = event.script
        if event.script is None:
            self._loaded_script_audio = None
            return
        self._loaded_script_audio = None
        try:
            pane = self.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        except NoMatches:
            pane = None
        if pane is not None:
            pane.script_audio = None
        self.run_worker(
            self._load_briefings(), exclusive=True, group="wl-briefings-load"
        )

    @on(CitationActivated)
    def handle_citation_activated(self, event: CitationActivated) -> None:
        """A citation click is an OPEN (spec #2 phase 2a, Task 6 design
        ruling -- do not relitigate): resolved exactly like clicking the
        item's own row in the Items table, mark-read side effect included,
        so a future "why did my item get marked read" question has THIS
        path, not just a mouse click on the Items table, to find.

        `event.item_id` was already resolved against the database when this
        briefing was selected (`_load_briefings`, via `get_subscription_
        items_by_ids`) -- `_citation_item_lookup` holds the result, keyed by
        the same id. A missing key IS the pruned signal (the plan's named
        invariant): the item existed when the briefing was written but does
        not resolve now, and this refuses to switch sections over it --
        there would be nothing in the reader to show, and moving the user
        off whatever section they are on to reveal that would be worse than
        staying put. `markup=False`: nothing here is app-authored prose the
        toast needs escaping protection FROM, but the id came from a body an
        LLM wrote, so the same caution `_load_briefings`'s own failure
        toasts already take applies.

        For a resolving id, switches to the Items ("Read") section, the only
        section where `ContentPane` is mounted, and once that section's
        `ItemsPane` exists, hands it the
        resolved item via `ItemsPane.select_and_reveal` (NOT `handle_item_
        selected` directly: that method's own docstring warns the pane's
        `selected_item` reactive would go stale against the table's actual
        cursor/scroll position). This reuses the exact `selected_item` ->
        `watch_selected_item` -> `ItemSelected` -> `handle_item_selected`
        path a real click already uses, so the reader update and the
        mark-read side effect both come along for free. A cited item hidden
        by the active items filter still opens (design ruling): `select_
        and_reveal` sets the reactive regardless, and the cursor simply
        stays put rather than pointing at a row that is not on screen.

        The section switch is not visible to a query until the NEXT
        recompose (`watch_active_section`'s own `refresh(recompose=True)`
        is asynchronous, not immediate), so opening the item is deferred by
        one short timer -- the identical idiom `handle_edit_rule_requested`
        already uses to act on a freshly-switched section's pane.
        """
        event.stop()
        item = self._citation_item_lookup.get(event.item_id)
        if item is None:
            self._notify_watchlists(
                f"Item {event.item_id} is no longer available.",
                severity="warning",
                markup=False,
            )
            return
        self.active_section = "items"

        def _open_citation() -> None:
            if not self.is_mounted:
                return
            try:
                items_pane = self.query_one("#watchlists-items-pane", ArticleListPane)
            except NoMatches:
                return
            items_pane.select_and_reveal(item)

        self.set_timer(0.05, _open_citation)

    @on(RefreshBriefingsRequested)
    def handle_refresh_briefings_requested(
        self, event: RefreshBriefingsRequested
    ) -> None:
        event.stop()
        self.run_worker(
            self._load_briefings(), exclusive=True, group="wl-briefings-load"
        )

    # --- Exporting a briefing as markdown (spec #2 phase 3, Task 1) --------

    @on(ExportBriefingRequested)
    def handle_export_briefing_requested(
        self, event: ExportBriefingRequested
    ) -> None:
        """Claim the one-export-at-a-time guard, then dispatch.

        `ArtifactsPane.compose` already disables Export for no-selection
        and any non-`complete` status, but this handler re-checks both
        anyway: the button's disabled state and the message it posts are
        two different frames, and a press already in flight when the
        selection changes underneath it must not be trusted just because
        it once passed a disabled check.

        Review round 1 (Important #1): an earlier draft shipped with no
        guard at all, reasoning that Textual "refuses to stack" a second
        `FileSave`. A live repro of two rapid presses disproved that --
        the screen stack ended up `['FileSave', 'FileSave']`, two live
        dialogs, not one refused. `_briefing_export_in_flight` is claimed
        HERE, before `run_worker` -- the same reason `_briefing_in_flight`
        is claimed before ITS `run_worker` in `handle_generate_briefing_
        requested`: claiming inside the worker body leaves a window where
        two presses both pass the check before either sets the flag.
        """
        event.stop()
        briefing = self._selected_briefing
        if briefing is None or str(briefing.get("status") or "").strip().lower() != (
            STATUS_COMPLETE
        ):
            self._notify_watchlists(
                "Select a completed briefing to export.",
                severity="warning",
                markup=False,
            )
            return
        if self._briefing_export_in_flight:
            self._notify_watchlists(
                "A briefing export is already in progress. Nothing else "
                "was started.",
                severity="warning",
                markup=False,
            )
            return
        watchlist_id = briefing.get("watchlist_id")
        watchlist_name = (
            self._watchlist_display_name(watchlist_id)
            if watchlist_id is not None
            else "this watchlist"
        )
        self._briefing_export_in_flight = True
        self.run_worker(
            self._push_export_briefing_dialog(dict(briefing), watchlist_name),
            group="wl-briefing-export",
        )

    async def _push_export_briefing_dialog(
        self, briefing: dict[str, Any], watchlist_name: str
    ) -> None:
        """Push `FileSave`, seeded with a sanitized default filename.

        Mirrors `_export_library_note`'s flow (`library_screen.py:6428`):
        a `FileSave` prompt pre-filled with a sanitized default filename,
        whose callback writes the export once a path is chosen. Imports
        the VENDORED picker (`Third_Party.textual_fspicker.FileSave`,
        keyword `default_file`) -- not the enhanced picker in `Widgets/
        enhanced_file_picker.py`, a different class whose keyword is
        `default_filename` and which also takes `context=`; mixing the two
        raises `TypeError`.

        `briefing["watchlist_name"]` is merged in here, once, rather than
        threaded through as a separate parameter to every downstream
        function: `briefing_markdown_document`/`default_briefing_filename`
        both read it directly off the mapping they are given.

        Does NOT clear `_briefing_export_in_flight` on the ordinary
        success path: `await self.app.push_screen(...)` returns once the
        dialog is MOUNTED, not once the user has chosen a path or
        cancelled -- clearing the guard here would re-open the window a
        second press could race through while the first dialog is still
        on screen. `_write_briefing_export_file` (the callback) is what
        clears it, in its own `finally`, once the dialog actually
        resolves. The guard IS cleared here on any path that never reaches
        that callback at all -- a failure to even open the dialog, or the
        worker being cancelled while awaiting the push.

        Review round 1 named a real (if not currently exploitable) gap
        here: the original shape cleared the guard inside `except
        Exception`, which does NOT catch `asyncio.CancelledError` (a
        `BaseException` since Python 3.8) -- a worker cancelled mid-`await
        self.app.push_screen(...)` would skip straight past that `except`
        with the guard still `True`, stranding Export shut for the rest of
        this screen instance's life. Not reachable today (nothing cancels
        this worker's `wl-briefing-export` group, and the flag dies with
        the screen instance regardless), but `_generate_briefing`'s own
        try/finally shape fixes it for free: `finally` runs for *every*
        exit path, including a `BaseException`, so the `pushed` sentinel
        below -- set only once the dialog has actually mounted -- is
        enough to tell "never reached the callback" apart from "did" --
        without needing a second `except` clause for cancellation.
        """
        pushed = False
        try:
            enriched = {**briefing, "watchlist_name": watchlist_name}
            default_filename = default_briefing_filename(
                enriched, watchlist_name=watchlist_name
            )
            await self.app.push_screen(
                FileSave(
                    location=str(Path.home()),
                    title="Export Briefing as Markdown",
                    default_file=default_filename,
                ),
                callback=lambda path: self._write_briefing_export_file(
                    path, enriched
                ),
            )
            pushed = True
        except Exception as exc:  # noqa: BLE001 - a worker crash exits the app
            logger.warning(f"Failed to open the export dialog: {type(exc).__name__}")
            self._notify_watchlists(
                "Could not open the export dialog.",
                severity="error",
                markup=False,
            )
        finally:
            if not pushed:
                self._briefing_export_in_flight = False

    async def _write_briefing_export_file(
        self, selected_path: Path | None, briefing: Mapping[str, Any]
    ) -> None:
        """Validate the chosen path, build the document, and write it.

        Mirrors `_write_library_note_export_file`'s validate -> write ->
        honest-toasts shape (`library_screen.py:6445`), with three
        deliberate differences this feature's own rules (and review round
        1) require: the write itself runs in `asyncio.to_thread` (a
        `FileSave`-chosen destination can be anywhere, including a slow or
        network-mounted path, so the write must never block the event
        loop); exception logging is type-only -- never `logger.opt(
        exception=True)`, and never the briefing body; and the write's
        `except` is broad (`Exception`, not just `OSError`) -- review
        round 1 (Important #2) confirmed a `UnicodeEncodeError` (entirely
        plausible from model/feed-derived body text) escaped uncaught
        under the narrower catch, a silent failure with no toast at all.
        `asyncio.CancelledError` is deliberately re-raised rather than
        caught by that broad except: a cancelled worker must not be
        reported as a failed export.

        `_briefing_export_in_flight` is cleared in `finally`, on every
        exit path -- cancelled, rejected path, export-error, write
        failure, or success alike -- so a cancel or a refusal never wedges
        Export shut for the rest of the session (review round 1's own
        re-arm requirement).

        Args:
            selected_path: The chosen destination, or `None` if the
                dialog was cancelled.
            briefing: The briefing row, with `watchlist_name` merged in by
                `_push_export_briefing_dialog`.
        """
        try:
            if not selected_path:
                self._notify_watchlists(
                    "Briefing export cancelled.", severity="information"
                )
                return
            try:
                validated_path = validate_path_simple(
                    selected_path, require_exists=False
                )
            except ValueError as exc:
                logger.warning(
                    f"Rejected briefing export path: {type(exc).__name__}"
                )
                self._notify_watchlists(
                    f"Rejected export path: {exc}",
                    severity="warning",
                    markup=False,
                )
                return
            try:
                document = briefing_markdown_document(briefing)
            except BriefingExportError as exc:
                self._notify_watchlists(str(exc), severity="warning", markup=False)
                return
            try:
                await asyncio.to_thread(
                    validated_path.write_text, document, encoding="utf-8"
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    f"Briefing export write failed: {type(exc).__name__}"
                )
                self._notify_watchlists(
                    f"Error exporting briefing: {type(exc).__name__}",
                    severity="error",
                    markup=False,
                )
                return
            self._notify_watchlists(
                f"Briefing exported successfully to {validated_path.name}",
                severity="information",
                markup=False,
            )
        finally:
            self._briefing_export_in_flight = False

    # --- Keeping a briefing into ChaChaNotes (task-1780, Task 5) ------------
    #
    # `briefing_keep.keep_briefing` (Task 2) is the one writer for
    # `kept_briefings`/`kept_scripts`; this handler is only "mount/dismiss
    # wiring" around it, the same division of labour every service on this
    # screen already gets (`generate_briefing`/`generate_script` and their
    # own handlers). Additive-idempotent and safe to press again on an
    # already-kept briefing -- the honest re-keep toast below is what makes
    # that safe to do BY DESIGN rather than by accident.

    @on(KeepBriefingRequested)
    def handle_keep_briefing_requested(self, event: KeepBriefingRequested) -> None:
        """Re-check both requirements, claim the guard, then dispatch.

        `ArtifactsPane.compose` already disables Keep without a complete
        selection or without a ChaChaNotes handle, but this handler
        re-checks both anyway -- the button's disabled state and the
        message it posts are two different frames, exactly the same
        reasoning `handle_export_briefing_requested`'s own docstring gives
        for its identical re-check.

        `_keep_in_flight` is claimed HERE, before `run_worker`, for the
        same reason every other in-flight guard on this screen is: a check
        made inside the worker body leaves a window where two presses both
        pass before either sets the flag.
        """
        event.stop()
        briefing = self._selected_briefing
        subs_db = self._briefings_db()
        chacha_db = self._chachanotes_db()
        if (
            briefing is None
            or str(briefing.get("status") or "").strip().lower() != STATUS_COMPLETE
            or subs_db is None
            or chacha_db is None
        ):
            self._notify_watchlists(
                "Select a completed briefing to keep it.",
                severity="warning",
                markup=False,
            )
            return
        if self._keep_in_flight:
            self._notify_watchlists(
                "A keep is already in progress. Nothing else was started.",
                severity="warning",
                markup=False,
            )
            return
        self._keep_in_flight = True
        self.run_worker(
            self._keep_briefing(subs_db, chacha_db, briefing["id"]),
            group="wl-keep",
        )

    async def _keep_briefing(
        self, subs_db: Any, chacha_db: Any, briefing_id: int
    ) -> None:
        """Worker body: keep, then toast honestly. Sibling of `_generate_
        briefing`/`_cast_script`: one bare `except` around the DB call
        turns a database error into a toast instead of taking the whole
        app down (an exception escaping a Textual worker with the default
        `exit_on_error=True` does exactly that).

        `KeepRefused` is caught separately, first: `keep_briefing`'s own
        honest, safe-to-show-verbatim pre-flight refusal (a missing
        briefing, a non-`complete` status, or an empty body) -- none of
        these are reachable through this handler's own re-check above in
        practice (the selection was already re-verified `complete`), but
        the selected briefing's status could still change between that
        check and this worker actually running (a concurrent Refresh, or a
        second window against the same database file), so this stays a
        real, not merely defensive, branch.

        The success toast reports the two branches `keep_briefing` itself
        distinguishes (spec, and this task's own AC): `created=True` says
        how many scripts came along; `created=False` (a re-keep) says the
        briefing was already kept and reports only what was newly added --
        the additive-idempotent re-keep, surfaced honestly rather than
        claiming a fresh keep happened.
        """
        try:
            try:
                result = await asyncio.to_thread(
                    keep_briefing, subs_db, chacha_db, briefing_id, origin="manual"
                )
            except KeepRefused as exc:
                self._notify_watchlists(str(exc), severity="warning", markup=False)
                return
            except Exception as exc:  # noqa: BLE001 - a worker crash exits the app
                logger.warning(
                    f"Keep failed for briefing {briefing_id}: {type(exc).__name__}"
                )
                self._notify_watchlists(
                    "Could not keep this briefing: the database could not "
                    "be reached. Nothing was recorded.",
                    severity="error",
                    markup=False,
                )
                return
            scripts_added = result["scripts_added"]
            if result["created"]:
                message = f"Kept with {scripts_added} scripts"
            else:
                message = f"Already kept — added {scripts_added} new scripts"
            self._notify_watchlists(message, severity="information", markup=False)
        finally:
            self._keep_in_flight = False

    # --- Kept briefings modal (task-1780, Task 5) ---------------------------
    #
    # Deliberately scope-independent -- see `KeptBriefingsRequested`'s own
    # docstring. Gated on nothing but a live ChaChaNotes handle: unlike
    # `_open_briefing_preset_manager` (which refuses without `_briefings_db
    # ()` too, since a preset IS a `SubscriptionsDB` row), this modal's own
    # content lives entirely in ChaChaNotes, and `subs_db` here is merely an
    # optional convenience the modal degrades gracefully without (see
    # `KeptBriefingsModal`'s own module docstring).

    @on(KeptBriefingsRequested)
    def handle_kept_briefings_requested(self, event: KeptBriefingsRequested) -> None:
        """Wire the toolbar's "Kept Briefings…" button to the modal opener.

        No `exclusive=True` -- `_open_kept_briefings_modal` owns a modal via
        `push_screen_wait`, and `_open_briefing_preset_manager`'s own
        sibling handler states exactly why an exclusive worker is the wrong
        tool for that: cancelling one mid-view would leave its modal on the
        screen stack with nothing left to dismiss it.
        """
        event.stop()
        chacha_db = self._chachanotes_db()
        if chacha_db is None:
            self._notify_watchlists(
                "Connect a ChaChaNotes database to browse kept briefings.",
                severity="error",
                markup=False,
            )
            return
        self.run_worker(
            self._open_kept_briefings_modal(chacha_db),
            group="wl-kept-briefings",
        )

    async def _open_kept_briefings_modal(self, chacha_db: Any) -> None:
        """Push `KeptBriefingsModal`, then forget it -- it owns its own
        reads and writes, and this screen holds no kept-briefing state of
        its own to refresh afterward (see the modal's own dismiss-protocol
        docstring).

        `subs_db` may be `None` (the watchlist bundle service itself is
        unavailable) -- the modal is built to degrade around that, offering
        only the app-default cast (see its own module docstring). `load_
        character` reuses `_cast_load_character`, the SAME resolver the
        screen's own live-cast path (`_cast_script`) already builds for an
        identical reason: a roster speaker bound to a character card must
        resolve against the SAME database this screen would use anywhere
        else, not a second, differently-scoped lookup.
        """
        subs_db = self._briefings_db()
        await self.app.push_screen_wait(
            KeptBriefingsModal(
                chacha_db,
                subs_db=subs_db,
                load_character=self._cast_load_character,
            )
        )

    # --- Exporting a watchlist's podcast feed directory (spec #2 phase -----
    # 3, Task 5). Sibling of the markdown-export flow immediately above in
    # every respect that matters: an independent in-flight guard
    # (`_feed_export_in_flight` -- deliberately its OWN flag, not reused
    # from `_briefing_export_in_flight`, since a briefing export and a
    # feed export are two independent actions with two different
    # destinations a user could plausibly run at the same time), claimed
    # before `run_worker` for the identical reason
    # `handle_export_briefing_requested`'s docstring gives, and a picker
    # pushed via `push_screen(..., callback=...)` rather than `push_
    # screen_wait` -- NOT `exclusive=True` on the worker either.
    #
    # Review round 1: the durable reason, confirmed by the reviewer, is
    # structural rather than just "the guard already prevents a second
    # dispatch." `Screen._push_result_callback` wraps the callback in a
    # `ResultCallback`, and `ResultCallback.__call__` -- what `Screen.
    # dismiss` invokes -- schedules it via `requester.call_next(...)`,
    # i.e. onto the REQUESTER's (this screen's) own message pump, never
    # back inside the `wl-feed-export` worker. `_push_export_feed_dialog`'s
    # entire life is therefore the single `await self.app.push_screen(...)`
    # that returns once `SelectDirectory` MOUNTS -- by the time a user
    # could ever pick a path or cancel, that worker has already finished
    # and exited; `_export_feed_directory` (where the guard actually
    # clears) never runs inside it at all. `exclusive=True` cancelling a
    # "previous" worker in this group is therefore a no-op on every
    # reachable path: there is nothing long-lived left to cancel, and the
    # `_start_tree_write`-style zombie-modal failure mode (a live worker
    # still awaiting a picker's dismissal, cancelled out from under it) is
    # structurally impossible here -- not merely guarded against. Left
    # here anyway as belt-and-suspenders should a future refactor route
    # the guard differently: the boolean check below still makes a second
    # dispatch impossible while one is in flight, so `exclusive` would
    # only ever fire on a bug in THAT guard.
    #
    # Two differences from the markdown flow, both Task 4's own contract:
    # the destination must already EXIST (`export_feed_directory`'s own
    # `validate_path_simple(..., require_exists=True)`, which is exactly
    # what `SelectDirectory` -- a directory BROWSER, not a name prompt --
    # ever hands back), and a partial export (some episodes skipped) is a
    # SUCCESSFUL result, not an error: `FeedExportResult.skipped` exists
    # precisely so the toast can say "N of M exported" rather than
    # silently claiming everything landed.

    @on(ExportFeedRequested)
    def handle_export_feed_requested(self, event: ExportFeedRequested) -> None:
        """Claim the one-feed-export-at-a-time guard, then dispatch.

        Re-checks `_watchlist_has_audio_episodes` even though `Artifacts
        Pane.compose` already disables the button for the same reason
        `handle_export_briefing_requested`'s own docstring gives: the
        button's disabled state and the message this handler acts on are
        two different frames, and a press already in flight when the
        underlying audio changes underneath it must not be trusted just
        because it once passed a disabled check.
        """
        event.stop()
        db = self._briefings_db()
        watchlist_id = self._briefing_watchlist_id()
        if db is None or watchlist_id is None:
            self._notify_watchlists(
                "Select a watchlist in the rail to export its feed.",
                severity="warning",
                markup=False,
            )
            return
        if not self._watchlist_has_audio_episodes:
            self._notify_watchlists(
                "This watchlist has no complete audio episodes to export.",
                severity="warning",
                markup=False,
            )
            return
        if self._feed_export_in_flight:
            self._notify_watchlists(
                "A feed export is already in progress. Nothing else was "
                "started.",
                severity="warning",
                markup=False,
            )
            return
        watchlist_name = self._watchlist_display_name(watchlist_id)
        self._feed_export_in_flight = True
        # `db`/`watchlist_id`/`watchlist_name` are snapshotted here, on the
        # UI thread, alongside the rest of this synchronous dispatch --
        # not re-read from `self` later, so a concurrent scope change on
        # the rail cannot redirect an export already in flight to a
        # different watchlist's audio.
        self.run_worker(
            self._push_export_feed_dialog(db, watchlist_id, watchlist_name),
            group="wl-feed-export",
        )

    async def _push_export_feed_dialog(
        self, db: Any, watchlist_id: int, watchlist_name: str
    ) -> None:
        """Push `SelectDirectory`, starting from the user's home directory.

        Mirrors `_push_export_briefing_dialog` exactly, including its own
        review-round-1 fix: `pushed` is set only once the dialog has
        actually mounted, so `finally` clears the guard on any path that
        never reaches `_export_feed_directory` (the callback) at all -- a
        failure to even open the dialog, or this worker being cancelled
        while awaiting the push.
        """
        pushed = False
        try:
            await self.app.push_screen(
                SelectDirectory(str(Path.home()), title="Export Podcast Feed"),
                callback=lambda path: self._export_feed_directory(
                    db, watchlist_id, watchlist_name, path
                ),
            )
            pushed = True
        except Exception as exc:  # noqa: BLE001 - a worker crash exits the app
            logger.warning(
                f"Failed to open the feed export dialog: {type(exc).__name__}"
            )
            self._notify_watchlists(
                "Could not open the feed export dialog.",
                severity="error",
                markup=False,
            )
        finally:
            if not pushed:
                self._feed_export_in_flight = False

    #: Task 5 (phase 3), review round 1, Minor #2: the ceiling on how many
    #: of `FeedExportResult.skipped`'s reasons are inlined into the
    #: partial-export toast. `export_feed_directory` can skip arbitrarily
    #: many episodes, and a toast quoting every one of them is unreadable
    #: long before it gets there -- the headline "N of M" count already
    #: states the honest outcome; these are just the first few reasons,
    #: for a user who wants to know why.
    _MAX_INLINE_SKIP_REASONS = 3

    #: Every reason `export_feed_directory` writes is `f"audio {audio_id}:
    #: ..."` (that module's own docstring, decision 3) -- the id is
    #: load-bearing for support/debugging (kept in the log line in
    #: `_export_feed_directory` below) but means nothing to a user reading
    #: a toast, so it is stripped before any reason reaches one.
    _SKIP_REASON_ID_PREFIX = re.compile(r"^audio \d+: ")

    @classmethod
    def _user_facing_skip_reasons(cls, reasons: list[str]) -> str:
        """The first `_MAX_INLINE_SKIP_REASONS` of `reasons`, id-stripped
        and joined for a toast.

        Args:
            reasons: `FeedExportResult.skipped`, verbatim.

        Returns:
            `"reason one; reason two; reason three; …and 4 more"` -- or
            just the id-stripped reasons, joined, with no trailer, when
            `len(reasons) <= _MAX_INLINE_SKIP_REASONS`.
        """
        stripped = [cls._SKIP_REASON_ID_PREFIX.sub("", reason, count=1) for reason in reasons]
        shown = stripped[: cls._MAX_INLINE_SKIP_REASONS]
        remaining = len(stripped) - len(shown)
        text = "; ".join(shown)
        return f"{text}; …and {remaining} more" if remaining > 0 else text

    async def _export_feed_directory(
        self,
        db: Any,
        watchlist_id: int,
        watchlist_name: str,
        selected_path: Path | None,
    ) -> None:
        """Validate the chosen directory, export the feed, and toast honestly.

        All DB and filesystem work -- `export_feed_directory` itself --
        runs in ONE `asyncio.to_thread` hop; that function does its own
        destination validation (`require_exists=True`, matching what
        `SelectDirectory` ever hands back), the per-episode safety checks
        and copies, and the atomic `feed.xml` write.

        A partial export (`result.skipped` non-empty) is reported as
        exactly that -- "N of M episodes exported", plus up to `_MAX_
        INLINE_SKIP_REASONS` of the reasons `export_feed_directory` wrote
        (id-stripped -- see `_user_facing_skip_reasons`), with an honest
        "…and N more" trailer past that cap -- never collapsed into a
        plain success toast. `_feed_export_in_flight` is cleared in
        `finally`, on every exit path (cancelled, rejected destination,
        export-error, or success alike), mirroring `_write_briefing_
        export_file`'s own re-arm guarantee.

        Args:
            db: Snapshotted by the handler at dispatch time.
            watchlist_id: Snapshotted by the handler at dispatch time.
            watchlist_name: Snapshotted by the handler at dispatch time.
            selected_path: The chosen destination directory, or `None` if
                the dialog was cancelled.
        """
        try:
            if not selected_path:
                self._notify_watchlists(
                    "Feed export cancelled.", severity="information"
                )
                return
            try:
                result = await asyncio.to_thread(
                    export_feed_directory,
                    db,
                    watchlist_id,
                    destination=selected_path,
                    watchlist_name=watchlist_name,
                    now=datetime.now(timezone.utc),
                )
            except asyncio.CancelledError:
                raise
            except ValueError as exc:
                logger.warning(
                    f"Rejected feed export destination: {type(exc).__name__}"
                )
                self._notify_watchlists(
                    f"Rejected export destination: {exc}",
                    severity="warning",
                    markup=False,
                )
                return
            except Exception as exc:
                logger.warning(f"Feed export failed: {type(exc).__name__}")
                self._notify_watchlists(
                    f"Error exporting the feed: {type(exc).__name__}",
                    severity="error",
                    markup=False,
                )
                return
            # task-1760: a successful export (partial or full -- either way
            # `result.directory` holds a real, just-written `feed.xml`) is
            # now something the Serve button can act on. Recorded here,
            # not only inside the two toast branches below, so it applies
            # to both outcomes -- and patched onto the mounted pane
            # directly (the same "patch it in place" idiom `_sync_feed_
            # server_pane_state` uses elsewhere), since a fresh export can
            # arrive while Artifacts is already on screen.
            self._last_feed_export_directory = result.directory
            self._sync_feed_server_pane_state()
            # task-1760 review, L1: a running server keeps serving whatever
            # directory it was STARTED with -- `FeedDirectoryServer` never
            # picks up a later export on its own (refuses a second `start()`
            # instead, `start`'s own docstring). If this export landed
            # somewhere other than that directory, the running server is
            # now silently stale: the only prior explanation was the Serve
            # button's own disabled state, easy to miss. Said only when it
            # actually differs -- a re-export into the SAME directory is
            # already reflected live, since the server reads from disk on
            # every request rather than caching anything.
            still_serving_stale_export = (
                self._feed_server.is_running
                and self._feed_server.directory != result.directory
            )
            stale_export_note = (
                " Still serving the previously-exported folder — Stop "
                "Serving and Serve again to publish this export."
                if still_serving_stale_export
                else ""
            )
            total = result.episode_count + len(result.skipped)
            if result.skipped:
                # Honest, not a success toast: this is Task 4's own named
                # invariant (`FeedExportResult.skipped`'s docstring) applied
                # at the UI boundary -- a user who exported ten episodes and
                # got eight must be told, in the reasons `export_feed_
                # directory` already wrote in plain language.
                #
                # Review round 1, Minor #2: the FULL list (with each
                # episode's `audio_id`, useful for support) is logged
                # here -- these are already-benign, app-generated reason
                # strings, never model/user content, so this is not the
                # "type only" rule `logger.warning` calls elsewhere on this
                # screen apply to an exception. The TOAST gets a separate,
                # user-facing rendering: capped to `_MAX_INLINE_SKIP_
                # REASONS` (with an honest "…and N more" trailer) and with
                # the internal `audio {id}:` prefix stripped, since a raw
                # database id means nothing to a user.
                logger.info(
                    f"Feed export for watchlist {watchlist_id} skipped "
                    f"{len(result.skipped)} of {total} episode(s): "
                    f"{'; '.join(result.skipped)}"
                )
                reasons = self._user_facing_skip_reasons(result.skipped)
                self._notify_watchlists(
                    f"Exported {result.episode_count} of {total} episodes "
                    f"to {result.directory.name} ({reasons})."
                    f"{stale_export_note}",
                    severity="warning",
                    markup=False,
                )
            else:
                plural = "" if result.episode_count == 1 else "s"
                self._notify_watchlists(
                    f"Exported {result.episode_count} episode{plural} to "
                    f"{result.directory.name}.{stale_export_note}",
                    severity="information" if not stale_export_note else "warning",
                    markup=False,
                )
        finally:
            self._feed_export_in_flight = False

    # --- Serving the exported feed directory over localhost (task-1760) -----
    #
    # `Subscriptions.feed_server.FeedDirectoryServer` does the actual work
    # (a `ThreadingHTTPServer` on a daemon thread); this screen only owns
    # ONE instance of it (`self._feed_server`, constructed in `__init__`)
    # and decides when to start/stop it. No `run_worker` here, unlike every
    # other action on this screen: `start()` binds a socket and spawns a
    # thread with no `await` boundary, and `stop()` -- after the task-1760
    # review's M2 fix, which starts `serve_forever` at a 50ms poll interval
    # instead of the stdlib's 0.5s default -- now blocks the UI thread for
    # roughly a tenth of what it used to (measured ~50ms here vs. a
    # measured ~501ms before the fix), a bound this screen accepts as
    # "fast enough not to need a worker" rather than eliminating entirely;
    # see `FeedDirectoryServer.stop`'s own docstring for the mechanics.
    # Both handlers re-check state before acting, the same "the button's
    # disabled state and the message it posts are two different frames"
    # reasoning `handle_export_feed_requested` already states for its own
    # re-check.

    def _sync_feed_server_pane_state(self) -> None:
        """Patch the mounted `ArtifactsPane`'s feed-server reactives from
        this screen's own state, if the pane is currently mounted.

        Called after every state change (a fresh export, Serve, Stop) --
        the same "patch it in place, never rebuild via `self.refresh
        (recompose=True)`" idiom the picker writers use (see
        `handle_briefing_mode_changed`'s own comment), for the identical
        reason: a full workbench rebuild is a much bigger hammer than one
        widget's reactive assignment, and `_build_detail_pane` already
        seeds a FRESH pane from this same state on every rebuild anyway,
        so nothing is lost if this screen is not showing Artifacts (or not
        attached) when this is called.
        """
        if not self.is_attached:
            return
        try:
            pane = self.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        except NoMatches:
            return
        pane.can_serve_feed = self._last_feed_export_directory is not None
        pane.feed_server_running = self._feed_server.is_running
        pane.feed_server_url = self._feed_server.url

    @on(ServeFeedRequested)
    def handle_serve_feed_requested(self, event: ServeFeedRequested) -> None:
        """Re-check both requirements, then start the server.

        Refuses (names the running URL) rather than restarting when a
        server is already running -- `FeedDirectoryServer.start`'s own
        docstring states why refusing was chosen as the simpler of the two
        options task-1760's plan allowed: a caller that wants to serve a
        DIFFERENT directory presses Stop first. `ArtifactsPane.compose`
        already disables the button in both refusal cases, but this
        re-checks anyway, for the identical reason every other handler on
        this screen does.
        """
        event.stop()
        if self._last_feed_export_directory is None:
            self._notify_watchlists(
                "Export a feed directory first, then serve it.",
                severity="warning",
                markup=False,
            )
            return
        if self._feed_server.is_running:
            self._notify_watchlists(
                f"A feed is already being served at {self._feed_server.url}. "
                "Stop it before serving a different directory.",
                severity="warning",
                markup=False,
            )
            return

        bind, port = configured_bind_and_port()
        try:
            url = self._feed_server.start(
                self._last_feed_export_directory, bind=bind, port=port
            )
        except (FeedServerError, OSError) as exc:
            logger.warning(f"Could not start the feed server: {type(exc).__name__}")
            self._notify_watchlists(
                "Could not start the feed server. Nothing is being served.",
                severity="error",
                markup=False,
            )
            return
        self._sync_feed_server_pane_state()
        # AC #4's posture, restated at the moment it matters most: every
        # time serving actually starts, not merely in a docstring or the
        # user guide. `markup=False` -- this interpolates a URL and (in
        # the widened-bind branch) a bind address this process itself
        # built/resolved (never model or remote content), but every toast
        # on this screen that is not a hand-written literal already takes
        # this same posture.
        #
        # task-1760 review, M4: says "this folder AND its subfolders" --
        # not just "the feed" -- since serving is recursive and the export
        # picker can point at any folder the user chooses, up to and
        # including their home directory.
        message = (
            f"Serving the exported feed at {url}. No authentication — "
            "anyone who can reach this address can read every file in "
            "this folder and its subfolders while it is serving."
        )
        # task-1760 review, M3: the posture above assumes loopback-only.
        # When the actually-bound address is NOT loopback (a deliberate
        # widening, or a config value that survived `_normalize_bind`
        # because it was a real address rather than blank/typo'd), say so
        # here too -- not just in the one-time `logger.warning` `start()`
        # already emits -- since this toast is what a user actually sees.
        served_bind = self._feed_server.bind
        if served_bind is not None and not is_loopback_bind(served_bind):
            message += (
                f" This is bound to {served_bind}, which is reachable "
                "from beyond this machine, not just localhost."
            )
        self._notify_watchlists(message, severity="warning", markup=False)

    @on(StopFeedServerRequested)
    def handle_stop_feed_server_requested(
        self, event: StopFeedServerRequested
    ) -> None:
        event.stop()
        if not self._feed_server.is_running:
            self._notify_watchlists(
                "Nothing is being served.", severity="warning", markup=False
            )
            return
        self._feed_server.stop()
        self._sync_feed_server_pane_state()
        self._notify_watchlists(
            "Stopped serving the feed.", severity="information", markup=False
        )

    def on_unmount(self) -> None:
        """Stop the feed server so a running listening socket never
        outlives this screen.

        The server's own thread is a daemon (`FeedDirectoryServer.start`),
        so it would not by itself block the app from exiting -- but
        leaving it running is still a wedged, forgotten listening socket
        for as long as the app process stays up otherwise (switching away
        from Watchlists, or closing this screen, must not silently keep
        serving). `is_running` guards a redundant `stop()` on a screen that
        never served anything, exactly like `ArtifactsScreen.on_unmount`'s
        own guard around its worker cancellation.
        """
        if self._feed_server.is_running:
            self._feed_server.stop()
        super().on_unmount()

    # --- Briefing selection-mode, default-preset, and cadence pickers -------
    # (Task 4, phase 2a; cadence added by Task 4, phase 4)
    #
    # Same write-first-patch-after shape as `handle_toggle_briefing_queue_
    # requested` -> `_toggle_briefing_queue`: the handler answers the
    # no-database case from memory and dispatches a worker; the worker does
    # the write off the UI thread (`asyncio.to_thread`), then on success
    # patches `_briefing_selection_mode`/`_briefing_default_preset_id`/
    # `_briefing_cadence_seconds` and the mounted pane's matching reactive
    # DIRECTLY -- never `_load_briefings()`, which would re-query the
    # database for a value this write already knows. No `exclusive=True`:
    # each picker's own writes target a single row with `UPDATE ... WHERE
    # id = ?`, so two overlapping presses are safe to interleave (last write
    # wins), and cancelling one mid-write would leave `_briefing_selection_
    # mode`/`_briefing_default_preset_id`/`_briefing_cadence_seconds`
    # disagreeing with what actually landed in the database.

    @on(BriefingModeChanged)
    def handle_briefing_mode_changed(self, event: BriefingModeChanged) -> None:
        event.stop()
        db = self._briefings_db()
        watchlist_id = self._briefing_watchlist_id()
        if db is None or watchlist_id is None:
            self._notify_watchlists(
                "Could not reach the local database, so nothing was saved.",
                severity="error",
            )
            return
        self.run_worker(
            self._write_briefing_selection_mode(db, watchlist_id, event.mode),
            group="wl-briefing-settings-write",
        )

    async def _write_briefing_selection_mode(
        self, db: Any, watchlist_id: int, mode: str
    ) -> None:
        try:
            await asyncio.to_thread(
                db.set_watchlist_briefing_settings,
                watchlist_id,
                selection_mode=mode,
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            logger.warning(
                f"Failed to save the selection mode for watchlist "
                f"{watchlist_id}: {type(exc).__name__}"
            )
            if self.is_attached:
                self._notify_watchlists(
                    "Could not save the selection mode. Nothing changed.",
                    severity="error",
                )
            return
        # Whole-branch review fix wave, Important #3: the write above is
        # correctly keyed to `watchlist_id` captured at dispatch and needs
        # no change, but this in-memory patch runs on the SCREEN, which is
        # global, singular state -- if the user switched Artifacts to a
        # DIFFERENT watchlist while this write was still in flight, `self.
        # _briefing_selection_mode`/the pane's reactive must not be
        # clobbered with the watchlist THIS write was about. Only patch
        # when the screen is still scoped to the same watchlist; the
        # scope-change path (`watch_tree_scope`) already re-dispatched its
        # own `_load_briefings()` reload the moment the scope moved, so the
        # new watchlist's own settings are not lost -- just not overwritten
        # by this stale completion. This guard has no mutation test of its
        # own: the claim is carried by
        # `test_switching_watchlists_mid_write_does_not_let_the_stale_write_clobber_the_new_one`,
        # which pins the identical guard on the preset writer below.
        if self._briefing_watchlist_id() != watchlist_id:
            return
        self._briefing_selection_mode = mode
        if not self.is_attached:
            return
        try:
            pane = self.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        except NoMatches:
            return
        pane.selection_mode = mode

    @on(BriefingDefaultPresetChanged)
    def handle_briefing_default_preset_changed(
        self, event: BriefingDefaultPresetChanged
    ) -> None:
        event.stop()
        db = self._briefings_db()
        watchlist_id = self._briefing_watchlist_id()
        if db is None or watchlist_id is None:
            self._notify_watchlists(
                "Could not reach the local database, so nothing was saved.",
                severity="error",
            )
            return
        self.run_worker(
            self._write_briefing_default_preset(db, watchlist_id, event.preset_id),
            group="wl-briefing-settings-write",
        )

    async def _write_briefing_default_preset(
        self, db: Any, watchlist_id: int, preset_id: int | None
    ) -> None:
        try:
            await asyncio.to_thread(
                db.set_watchlist_briefing_settings,
                watchlist_id,
                default_preset_id=preset_id,
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            logger.warning(
                f"Failed to save the default preset for watchlist "
                f"{watchlist_id}: {type(exc).__name__}"
            )
            if self.is_attached:
                self._notify_watchlists(
                    "Could not save the default preset. Nothing changed.",
                    severity="error",
                )
            return
        # Whole-branch review fix wave, Important #3: see the identical
        # note in `_write_briefing_selection_mode` -- the DB write above is
        # correctly keyed to `watchlist_id` and needs no change, but this
        # patch must not land if Artifacts has since moved to a different
        # watchlist. `handle_generate_briefing_requested:3880` reads `self.
        # _briefing_default_preset_id` at ITS OWN dispatch time, so an
        # unguarded patch here is not merely cosmetic: a Generate press for
        # a DIFFERENT, newly-scoped watchlist could otherwise pick up a
        # preset id that belongs to the watchlist this write was about.
        if self._briefing_watchlist_id() != watchlist_id:
            return
        self._briefing_default_preset_id = preset_id
        if not self.is_attached:
            return
        try:
            pane = self.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        except NoMatches:
            return
        pane.default_preset_id = preset_id
        # TASK-2311, AC#3: a preset's own provider can differ from the app
        # default (or from the PREVIOUS preset's provider) -- without this,
        # picking a new default preset would leave the scope note naming
        # whatever provider was displayed before this write.
        pane.default_provider_display = self._briefing_provider_display()

    @on(BriefingCadenceChanged)
    def handle_briefing_cadence_changed(self, event: BriefingCadenceChanged) -> None:
        """Spec #2 phase 4, Task 4: same shape as `handle_briefing_mode_
        changed`/`handle_briefing_default_preset_changed` above -- the
        no-database case is answered from memory, the real write dispatches
        a worker in the same `wl-briefing-settings-write` group (so an
        overlapping mode/preset/cadence write for the same watchlist is
        safe to interleave, last write wins, exactly like its two siblings).
        """
        event.stop()
        db = self._briefings_db()
        watchlist_id = self._briefing_watchlist_id()
        if db is None or watchlist_id is None:
            self._notify_watchlists(
                "Could not reach the local database, so nothing was saved.",
                severity="error",
            )
            return
        self.run_worker(
            self._write_briefing_cadence(db, watchlist_id, event.seconds),
            group="wl-briefing-settings-write",
        )

    async def _write_briefing_cadence(
        self, db: Any, watchlist_id: int, seconds: int | None
    ) -> None:
        try:
            await asyncio.to_thread(
                db.set_watchlist_briefing_settings,
                watchlist_id,
                briefing_cadence_seconds=seconds,
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            logger.warning(
                f"Failed to save the briefing schedule for watchlist "
                f"{watchlist_id}: {type(exc).__name__}"
            )
            if self.is_attached:
                self._notify_watchlists(
                    "Could not save the schedule. Nothing changed.",
                    severity="error",
                )
            return
        # Whole-branch review fix wave, Important #3: see the identical
        # note in `_write_briefing_selection_mode`/`_write_briefing_default_
        # preset` above -- the DB write is correctly keyed to `watchlist_id`
        # and needs no change, but this patch must not land if Artifacts
        # has since moved to a different watchlist.
        if self._briefing_watchlist_id() != watchlist_id:
            return
        self._briefing_cadence_seconds = seconds
        if not self.is_attached:
            return
        try:
            pane = self.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        except NoMatches:
            return
        pane.briefing_cadence_seconds = seconds
        # Unlike mode/preset, the scope label's TEXT depends on cadence
        # (`_briefing_scope_label` -> `cadence_scope_phrase`) -- without
        # this, the honesty fix this task exists to ship would only take
        # effect on the NEXT full `_load_briefings()` reload, not the
        # instant the user picks a cadence, leaving the toolbar Select and
        # the scope note disagreeing until then.
        pane.scope_label = self._briefing_scope_label()

    # --- Briefing presets (spec #2 phase 2a, Task 3): manager modal --------
    #
    # `BriefingPresetModal` owns its own reads and writes; this screen's job
    # is only what the brief calls "mount/dismiss wiring" -- build the two
    # option lists the modal is not entitled to query for itself, push it,
    # and reload the preset list iff the modal reports a real change. The
    # toolbar's "Presets..." button (Task 4) calls `_open_briefing_preset_
    # manager` unchanged, through `handle_manage_presets_requested` below.

    async def _load_character_options(self) -> list[tuple[str, int]]:
        """Character cards for the preset modal's per-speaker Select.

        Built here rather than inside the modal (brief: "the modal never
        queries other DBs itself"). Degrades to `[]` -- disabling the field,
        never the modal -- when `chachanotes_db` is unbound or the lookup
        fails.
        """
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            return []
        try:
            cards = await asyncio.to_thread(db.list_character_cards)
        except Exception as exc:  # noqa: BLE001 - degrade the field, not the modal
            logger.warning(
                "Failed to load character cards for briefing presets: "
                f"{type(exc).__name__}"
            )
            return []
        return [
            (str(card.get("name") or ""), int(card["id"]))
            for card in cards
            if card.get("id") is not None
        ]

    async def _load_voice_options(self) -> list[tuple[str, str]]:
        """Voice profiles for the preset modal's per-speaker Select.

        Same degrade-the-field rule as `_load_character_options`.
        `TTSProfileService.list_profiles` is already async and already
        offloads its own repository I/O (see `STTSProfileLibrary`'s
        identical direct-`await` usage) -- no `asyncio.to_thread` wrapper
        needed around it here.
        """
        service = getattr(self.app_instance, "_tts_profile_service", None)
        if service is None:
            return []
        try:
            page = await service.list_profiles()
        except Exception as exc:  # noqa: BLE001 - degrade the field, not the modal
            logger.warning(
                "Failed to load voice profiles for briefing presets: "
                f"{type(exc).__name__}"
            )
            return []
        return [
            (profile.display_name, str(profile.profile_id))
            for profile in page.profiles
        ]

    async def _load_briefing_presets(self) -> None:
        """Re-read every stored `briefing_presets` row, name ASC, and patch
        the Artifacts toolbar's default-preset picker in place.

        Two callers: `_load_briefings` (an Artifacts-section/scope load,
        which patches the pane's OTHER reactives itself right after this
        returns -- setting `pane.presets` here too is a harmless repeat of
        the same value) and `_open_briefing_preset_manager` (a preset
        modal's `True` dismiss, which has no other reason to touch the
        pane). Patched here rather than left to each caller so both stay
        honest without duplicating the query-and-patch shape.
        """
        db = self._briefings_db()
        if db is None:
            self._loaded_briefing_presets = []
        else:
            try:
                rows = await asyncio.to_thread(db.list_briefing_presets)
                self._loaded_briefing_presets = [dict(row) for row in rows]
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                logger.warning(
                    f"Failed to list briefing presets: {type(exc).__name__}"
                )
                self._loaded_briefing_presets = []
        if not self.is_mounted:
            return
        try:
            pane = self.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        except NoMatches:
            return
        pane.presets = self._loaded_briefing_presets

    async def _open_briefing_preset_manager(self) -> None:
        """Open `BriefingPresetModal`, then reload presets iff it changed.

        A `False`/cancelled dismiss leaves `_loaded_briefing_presets`
        untouched, matching every other reload-on-change flow already on
        this screen (`_create_watchlist_flow`, `_rename_watchlist_flow`,
        `_delete_watchlist_flow`).
        """
        db = self._briefings_db()
        if db is None:
            self._notify_watchlists(WC_SERVICE_UNAVAILABLE_COPY, severity="error")
            return
        character_options = await self._load_character_options()
        voice_options = await self._load_voice_options()
        changed = await self.app.push_screen_wait(
            BriefingPresetModal(
                db,
                character_options=character_options,
                voice_options=voice_options,
            )
        )
        if changed:
            await self._load_briefing_presets()

    @on(ManagePresetsRequested)
    def handle_manage_presets_requested(self, event: ManagePresetsRequested) -> None:
        """Wire the toolbar's "Presets…" button to Task 3's opener.

        No `exclusive=True`: `_open_briefing_preset_manager` owns a modal
        via `push_screen_wait`, and `_start_tree_write`'s own docstring
        names exactly why an exclusive worker is the wrong tool for that --
        cancelling one mid-prompt would leave its dialog on the screen
        stack with nothing left to dismiss it.
        """
        event.stop()
        self.run_worker(
            self._open_briefing_preset_manager(), group="wl-briefing-presets"
        )

    @on(GenerateBriefingRequested)
    def handle_generate_briefing_requested(
        self, event: GenerateBriefingRequested
    ) -> None:
        """Claim the one-generation-per-watchlist guard, then dispatch.

        This handler runs on the UI thread, so it does exactly two things
        that thread is entitled to do: answer from memory, and dispatch.
        Everything that touches the database -- the zombie sweep, the
        generating-check, the generation itself -- happens in the worker
        (fix round 1, Finding 1). `fail_interrupted_briefings` is a
        transactional UPDATE, no busy timeout beyond SQLite's default is
        configured, and this feature's own design admits a second app
        instance against the same database file: a contended write here
        would freeze the interface.

        `_briefing_in_flight` is claimed HERE, before `run_worker`, and not
        inside the worker body (fix round 1, Finding 2). `run_worker` only
        schedules; a check made inside the worker leaves a window in which
        two presses both pass, and `exclusive=True` then cancels the first
        one *mid-generation* -- leaving behind exactly the `generating` row
        this guard exists to prevent. The guard would be manufacturing the
        state it guards against.
        """
        event.stop()
        db = self._briefings_db()
        watchlist_id = self._briefing_watchlist_id()
        if db is None or watchlist_id is None:
            self._notify_watchlists(
                "Select a watchlist in the rail to brief it.",
                severity="warning",
                markup=False,
            )
            return
        if self._briefing_in_flight:
            # `_briefing_in_flight` is screen-global on purpose (one
            # `wl-briefing` worker at a time -- see the field's own
            # comment), so the running generation may belong to a
            # DIFFERENT watchlist than the one on screen right now. Naming
            # it when that name is cheaply available (whole-branch review
            # fix 4) keeps the toast truthful instead of always claiming
            # "this watchlist".
            running_id = self._briefing_in_flight_watchlist_id
            if running_id is not None:
                running_name = self._watchlist_display_name(running_id)
                message = (
                    f"A briefing is already being written for {running_name}. "
                    "Nothing else was started."
                )
            else:
                message = (
                    "A briefing is already being written. Nothing else was "
                    "started."
                )
            self._notify_watchlists(message, severity="warning", markup=False)
            return
        self._briefing_in_flight = True
        self._briefing_in_flight_watchlist_id = watchlist_id
        # Task 4: cast the die now, on the UI thread, alongside the rest of
        # this synchronous snapshot -- not read again later inside the
        # worker, where a concurrent picker write (a different worker
        # group, so not excluded by `exclusive=True` above) could otherwise
        # change `_briefing_default_preset_id` out from under a generation
        # already in flight for THIS watchlist.
        preset_id = self._briefing_default_preset_id
        self.run_worker(
            self._generate_briefing(db, watchlist_id, preset_id),
            exclusive=True,
            group="wl-briefing",
        )

    @staticmethod
    def _briefing_row_label(row: Mapping[str, Any]) -> str:
        """Name one briefing the way a toast has to: which row, and when."""
        return (
            f"briefing {row.get('id')} "
            f"(started {row.get('created_at') or 'at an unknown time'})"
        )

    def _zombie_sweep_is_safe(self) -> bool:
        """Whether `fail_interrupted_briefings` may run right now.

        `fail_interrupted_briefings`'s own `exclude` (phase 4) now spares the
        specific row a LIVE in-process claim is writing -- this screen's
        own, or a future scheduled run's (task-1812, AC #3: row-scoped, not
        merely watchlist-scoped, so a genuine crash zombie for the SAME
        watchlist is not incidentally shielded too) -- so it no longer fails
        EVERY `generating` row unconditionally the way it did before claims
        existed. This flag is a narrower, purely local check on top of that:
        it answers "is THIS screen instance mid-generation", which the
        Generate path (`_sweep_and_guard`) never needs -- it always runs at
        the very front of `_generate_briefing`, before that worker's own row
        is inserted, so there is nothing of "its own" yet to protect. The
        Artifacts-load path (`_load_briefings`) has no such ordering
        guarantee -- it can run at any time, including while a generation
        THIS screen started is still mid-flight -- so it consults this flag
        too, on top of the claim-aware `exclude`, rather than relying on the
        claim alone (whole-branch review fix 3).
        """
        return not self._briefing_in_flight

    async def _fail_interrupted_briefings_if_safe(
        self, db: Any, watchlist_id: int
    ) -> int:
        """Zombie recovery for the Artifacts-load path, off the UI thread.

        Spec: a `generating` row not backed by a live worker is failed "on
        the next Generate attempt or Artifacts load" -- only the Generate
        path was wired (`_sweep_and_guard`). Gated by
        `_zombie_sweep_is_safe` so a load racing a live generation this
        screen started cannot clobber that generation's own row.

        `active_briefing_claim_row_ids()` is snapshotted HERE, on the UI
        thread, before the sweep is dispatched to a worker thread (Locked
        decision 2): the claim registry is mutated only on the event loop,
        so a live read of it from the executor thread `asyncio.to_thread`
        uses would be racy in a way this snapshot never is. Passed through
        as `exclude` so a genuinely live claim's OWN row -- e.g. a scheduled
        run once phase 4's scheduler exists -- survives an Artifacts open
        instead of being falsified as interrupted (survey finding (a)).
        Row-scoped, not watchlist-scoped (task-1812, AC #3): a crash-zombie
        row from a prior process for this SAME watchlist must still be
        swept even while a fresh claim is live, and only naming the live
        row itself (not its whole watchlist) lets that happen.

        `pending_briefing_claim_watchlist_ids()` is snapshotted here too,
        same thread, same instant (whole-branch review, `chore/briefings-
        residuals-1810-1812`, Important 1): it closes the window `exclude`
        alone cannot -- a claim taken but whose row id has not yet been
        recorded, still inside `_start_generation`'s own `to_thread` hop.
        Passed as `exclude_watchlists`, never in place of `exclude`.
        """
        if not self._zombie_sweep_is_safe():
            return 0
        claims = active_briefing_claim_row_ids()
        pending = pending_briefing_claim_watchlist_ids()
        return await asyncio.to_thread(
            fail_interrupted_briefings,
            db,
            watchlist_id,
            exclude=claims,
            exclude_watchlists=pending,
        )

    def _sweep_and_guard(
        self,
        db: Any,
        watchlist_id: int,
        exclude: Collection[int],
        exclude_watchlists: Collection[int] = (),
    ) -> tuple[list[str], list[str]]:
        """Zombie sweep, then the generating-check. Runs off the UI thread.

        The order is the contract `briefing_service` states: the service
        neither checks nor recovers -- folding either in would make it both
        the thing guarded and the guard -- so the caller sweeps FIRST, and
        only then asks whether anything is still generating. A row orphaned
        by a crashed worker can therefore never wedge the guard shut.

        `exclude` -- the caller's `active_briefing_claim_row_ids()` snapshot,
        taken before this whole method was dispatched to a worker thread --
        is passed straight to `fail_interrupted_briefings`. This screen's
        own claim for THIS watchlist has not been taken yet at this point
        (`generate_briefing` takes it, later, inside the same worker), so
        the only thing `exclude` can protect here is ANOTHER in-process
        caller's live row on the same watchlist. A row that survives the
        sweep for that reason is not a crash zombie -- it is a live
        generation this screen must not duplicate -- and it correctly ends
        up in `blocking`, triggering the existing refusal toast rather than
        letting Generate proceed over the top of it (survey finding (b)).
        Row-scoped (task-1812, AC #3): a crash-zombie row for THIS watchlist
        left by a prior process is swept here even while that other live
        row survives, rather than the whole watchlist being spared.

        `exclude_watchlists` -- the caller's `pending_briefing_claim_
        watchlist_ids()` snapshot, taken at the same instant as `exclude`
        (whole-branch review, `chore/briefings-residuals-1810-1812`,
        Important 1) -- closes the same window `exclude` alone cannot for
        THAT other in-process caller too: if it is still inside `_start_
        generation`'s own `to_thread` hop, its row exists and reads
        `generating` but has no id recorded yet, so only naming its
        watchlist (not yet its row) spares it here.

        Returns:
            `(recovered, blocking)` -- labels for the rows this sweep failed
            as interrupted, and labels for any row still `generating`
            afterwards. Labels rather than counts so the toast can name what
            it is talking about (fix round 1, Minor a).
        """
        stuck = [
            self._briefing_row_label(row)
            for row in db.list_briefings(watchlist_id)
            if str(row.get("status") or "").strip().lower() == STATUS_GENERATING
        ]
        fail_interrupted_briefings(
            db, watchlist_id, exclude=exclude, exclude_watchlists=exclude_watchlists
        )
        blocking = [
            self._briefing_row_label(row)
            for row in db.list_briefings(watchlist_id)
            if str(row.get("status") or "").strip().lower() == STATUS_GENERATING
        ]
        return stuck, blocking

    async def _generate_briefing(
        self, db: Any, watchlist_id: int, preset_id: int | None
    ) -> None:
        """Worker body: recover, guard, generate, repaint.

        The whole sequence is one worker so the guard cannot come apart from
        the generation it guards, and every database call inside it is
        awaited off the UI thread (`asyncio.to_thread`) -- see the handler.

        `preset_id` (Task 4) is the watchlist's stored default preset,
        snapshotted by the handler at dispatch time -- see its own comment.
        `None` means "no default preset stored", and `generate_briefing`
        treats that identically to "no preset given": the app default
        provider/model, no style notes.

        `generate_briefing` is wrapped in a bare `except` on purpose. It
        turns *provider* failures into `failed` rows rather than exceptions,
        but deliberately lets database errors propagate -- a database error
        is not a briefing outcome. An exception escaping a Textual worker
        with the default `exit_on_error=True` takes the whole application
        down, so the escape hatch has to be here.

        The log lines name the exception TYPE only. `logger.opt(exception=True)`
        would dump the failing frame's locals into a file sink running with
        `diagnose=True`, and the frames under this call hold the prompt --
        item titles and excerpts the user never chose to write to disk. Task
        3's review found exactly that leak in the service; this is the same
        rule, one layer up.
        """
        generated_id: int | None = None
        try:
            try:
                recovered, blocking = await asyncio.to_thread(
                    self._sweep_and_guard,
                    db,
                    watchlist_id,
                    active_briefing_claim_row_ids(),
                    pending_briefing_claim_watchlist_ids(),
                )
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                logger.warning(
                    f"Briefing guard failed for watchlist {watchlist_id}: "
                    f"{type(exc).__name__}"
                )
                self._notify_watchlists(
                    "Failed to read this watchlist's briefings. Nothing was "
                    "started.",
                    severity="error",
                    markup=False,
                )
                return
            if blocking:
                # Survived the sweep, so it was inserted after it: another
                # live writer against this same database file. Not ours to
                # cancel, and not ours to duplicate.
                self._notify_watchlists(
                    f"{', '.join(blocking)} is already in progress for this "
                    "watchlist. Nothing was started.",
                    severity="warning",
                    markup=False,
                )
                return
            if recovered:
                # Recovered, and reported rather than silently regenerated:
                # that row may have belonged to another live instance of
                # this app, and starting a second generation over the top of
                # one still running would spend the user's provider quota
                # twice on the same window.
                self._notify_watchlists(
                    f"{', '.join(recovered)} was still marked in progress and "
                    "has been marked interrupted. Press Generate again to "
                    "write a new one.",
                    severity="warning",
                    markup=False,
                )
                return
            try:
                row = await generate_briefing(db, watchlist_id, preset_id=preset_id)
                generated_id = (row or {}).get("id")
                # TASK-2311: `generate_briefing` never raises for a PROVIDER
                # failure -- it turns it into a `failed` row instead (see
                # this method's own docstring) -- so this is the one place
                # that outcome is actually observable. UAT: Generate with no
                # provider configured produced a bare "failed" row with no
                # toast; the reason ("OpenAI API Key is required but not
                # found") only appeared after clicking the row, and the
                # provider had silently defaulted to openai. `markup=False`:
                # a provider's own error text is untrusted.
                if str((row or {}).get("status") or "").strip().lower() == STATUS_FAILED:
                    self._notify_briefing_failure(row or {})
            except GenerationInFlightError as exc:
                # The race `_sweep_and_guard` cannot close: another
                # in-process caller claimed this watchlist AFTER the sweep
                # read the database (finding no `generating` row, so
                # `blocking` stayed empty) but BEFORE its own row landed --
                # this attempt then reached `generate_briefing`'s own claim
                # check instead. `str(exc)` already names the watchlist and
                # is user-safe (the class's own contract, mirroring
                # `ScriptCastError`'s) -- the bare `except Exception` below
                # must not swallow it as a generic database failure.
                self._notify_watchlists(str(exc), severity="warning", markup=False)
            except Exception as exc:  # noqa: BLE001 - a worker crash exits the app
                logger.warning(
                    f"Briefing generation failed for watchlist {watchlist_id}: "
                    f"{type(exc).__name__}"
                )
                self._notify_watchlists(
                    "Could not write a briefing: the watchlist database could "
                    "not be reached. Nothing was recorded.",
                    severity="error",
                    markup=False,
                )
        finally:
            self._briefing_in_flight = False
            self._briefing_in_flight_watchlist_id = None
            # Repaint on every path: a refusal has just changed a row's
            # status, and the failure path may leave a `generating` row this
            # attempt inserted before it broke.
            await self._load_briefings(select_briefing_id=generated_id)

    # --- Cast a script from the selected briefing (spec #2 phase 2a, ------
    # Task 5). Sibling of the Generate chain immediately above: own
    # in-flight flag (`_cast_in_flight`), own worker group (`wl-cast`,
    # `exclusive=True`), claimed at DISPATCH time for the identical reason
    # `handle_generate_briefing_requested`'s docstring gives -- a check made
    # inside the worker body leaves a window where two presses both pass.
    #
    # One real difference from Generate remains: `briefing_scripts` has no
    # one-COMPLETE-row-per-briefing invariant the way `briefings` has one
    # per watchlist (a briefing can be cast many times, with different
    # rosters, and `briefing_cast.py`'s own module docstring says so
    # explicitly) -- recovering a genuine zombie script does not itself
    # refuse a FRESH cast attempt the way recovering a zombie briefing
    # refuses a fresh generation. What phase 4 Task 1 adds is narrower: a
    # `_sweep_and_guard`-style `blocking` check for the one case that IS a
    # real problem -- a cast for THIS SAME briefing that is already
    # genuinely in flight (this screen's own, or another in-process
    # caller's, once phase 4's scheduler exists) must refuse rather than run
    # a second, concurrent cast over the top of it (survey finding (c):
    # before this, there was no such refusal at all). The zombie sweep still
    # runs in the same TWO seams it always has: `_load_briefings` whenever
    # Artifacts loads (gated on `_cast_sweep_is_safe`), and `_cast_script`
    # below at the front of every cast -- both now claim-aware via
    # `active_cast_claim_row_ids()`, exactly like Generate's own sweeps
    # (task-1890: row-scoped, not merely briefing-scoped, mirroring
    # task-1812's briefings-side fix).

    def _cast_sweep_is_safe(self) -> bool:
        """Whether `fail_interrupted_scripts` may run right now.

        Sibling of `_zombie_sweep_is_safe`: a load racing a cast THIS
        screen started must not fail that cast's own `generating` row out
        from under it, so the load path only sweeps when nothing this
        screen started is still in flight.
        """
        return not self._cast_in_flight

    async def _fail_interrupted_scripts_if_safe(
        self, db: Any, briefing_id: int
    ) -> int:
        """Zombie recovery for the Artifacts-load path's scripts, off the
        UI thread. Sibling of `_fail_interrupted_briefings_if_safe`, scoped
        to one briefing's scripts rather than one watchlist's briefings.

        `active_cast_claim_row_ids()` is snapshotted HERE, on the UI
        thread, before the sweep is dispatched -- see `_fail_interrupted_
        briefings_if_safe`'s own docstring for why a snapshot, not a live
        read, is required, and row-scoped rather than briefing-scoped
        (task-1890, mirroring task-1812's `active_briefing_claim_row_ids`).

        `pending_cast_claim_briefing_ids()` is snapshotted here too, same
        thread, same instant, closing the window `exclude` alone cannot: a
        claim taken but whose row id has not yet been recorded, still
        inside `_start_script`'s own `to_thread` hop. Passed as `exclude_
        briefings`, never in place of `exclude`.
        """
        if not self._cast_sweep_is_safe():
            return 0
        claims = active_cast_claim_row_ids()
        pending = pending_cast_claim_briefing_ids()
        return await asyncio.to_thread(
            fail_interrupted_scripts,
            db,
            briefing_id,
            exclude=claims,
            exclude_briefings=pending,
        )

    @staticmethod
    def _script_row_label(row: Mapping[str, Any]) -> str:
        """Name one script the way a toast has to: which row, and when.

        Sibling of `_briefing_row_label`, for the identical reason.
        """
        return (
            f"script {row.get('id')} "
            f"(started {row.get('created_at') or 'at an unknown time'})"
        )

    def _sweep_and_guard_cast(
        self,
        db: Any,
        briefing_id: int,
        exclude: Collection[int],
        exclude_briefings: Collection[int] = (),
    ) -> tuple[list[str], list[str]]:
        """Zombie sweep, then the generating-check, for a cast. Runs off the
        UI thread. Sibling of `_sweep_and_guard` -- see that method's own
        docstring for the full reasoning; this is the identical shape,
        scoped to one briefing's scripts instead of one watchlist's
        briefings (phase 4 Task 1, survey finding (c); row-scoped since
        task-1890).

        `exclude_briefings` -- the caller's `pending_cast_claim_briefing_
        ids()` snapshot, taken at the same instant as `exclude` -- closes
        the same window `exclude` alone cannot for another in-process
        caller: if it is still inside `_start_script`'s own `to_thread`
        hop, its row (once inserted) reads `generating` but has no id
        recorded yet, so only naming its briefing (not yet its row) spares
        it here.

        Returns:
            `(recovered, blocking)`, exactly like `_sweep_and_guard`.
        """
        stuck = [
            self._script_row_label(row)
            for row in db.list_briefing_scripts(briefing_id)
            if str(row.get("status") or "").strip().lower() == STATUS_GENERATING
        ]
        fail_interrupted_scripts(
            db, briefing_id, exclude=exclude, exclude_briefings=exclude_briefings
        )
        blocking = [
            self._script_row_label(row)
            for row in db.list_briefing_scripts(briefing_id)
            if str(row.get("status") or "").strip().lower() == STATUS_GENERATING
        ]
        return stuck, blocking

    def _cast_load_character(self, character_id: int) -> dict[str, Any] | None:
        """`generate_script`'s `load_character` seam: a plain, idempotent
        character-card lookup, with no caching layer and no session state.

        Task 2 review's carried finding: `_snapshot_roster` tolerates a
        transient failure here by degrading to `character_name: None`, and
        `_resolve_character_texts` calls `load_character` AGAIN, later, to
        resolve the same card strictly. If this held a cache -- or any
        other state that could answer the two calls differently -- the
        snapshot and the strict resolution could disagree about the SAME
        card within one cast. A bare `get_character_card_by_id` call never
        can: it is one SELECT by id, nothing memoized, so both calls always
        see the same, current answer.

        Only ever called from `_cast_script`, which already checked
        `chachanotes_db` is bound before passing this method as
        `load_character` -- see that worker.
        """
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            return None
        return db.get_character_card_by_id(character_id)

    def _briefing_default_preset_is_dangling(self) -> bool:
        """Whether the stored default preset id no longer resolves.

        Whole-branch review fix wave, Important #1: `BriefingPresetModal`
        hard-deletes a preset (Task 3; no FK enforces the pointer), and
        `_load_briefings`'s combined read (`_read_watchlist_briefing_
        state`) reloads the preset LIST but re-reads the watchlist's own
        `default_briefing_preset_id` column verbatim -- so a preset deleted
        while it was a watchlist's default leaves `_briefing_default_
        preset_id` pointing at a row that no longer exists among `_loaded_
        briefing_presets`. `ArtifactsPane._preset_select_options` already
        assumes exactly this shape (its own synthetic "Preset N (deleted)"
        option) -- this is that same check, on the screen side, so Cast can
        refuse before ever reaching `generate_script`'s own raw
        `ScriptCastError` text for it.
        """
        preset_id = self._briefing_default_preset_id
        if preset_id is None:
            return False
        return not any(
            preset.get("id") == preset_id for preset in self._loaded_briefing_presets
        )

    @on(CastScriptRequested)
    def handle_cast_script_requested(self, event: CastScriptRequested) -> None:
        """Claim the one-cast-at-a-time guard, then dispatch.

        Answers from memory and dispatches, exactly like
        `handle_generate_briefing_requested`: `_cast_in_flight` is claimed
        HERE, before `run_worker`, and not inside the worker body, for the
        identical reason that handler's own docstring gives.
        """
        event.stop()
        db = self._briefings_db()
        briefing = self._selected_briefing
        if db is None or briefing is None:
            self._notify_watchlists(
                "Select a briefing to cast.", severity="warning", markup=False,
            )
            return
        if self._briefing_default_preset_is_dangling():
            # Whole-branch review fix wave, Important #1: the stored
            # default resolved to a real preset once, but that preset was
            # since hard-deleted (from the toolbar's own picker, which is
            # already showing "(deleted)" for this same id). Refuse HERE,
            # before dispatch, with copy that tells the user what to do --
            # not `generate_script`'s own raw `ScriptCastError` text for
            # this case ("briefing preset 1 does not exist"), which names
            # an id but no action.
            self._notify_watchlists(
                "The stored default preset no longer exists. Pick another "
                "in the toolbar, or create one via Presets…, before "
                "casting.",
                severity="warning",
                markup=False,
            )
            return
        if self._briefing_default_preset_id is None and self._loaded_briefing_presets:
            # Task 5 review round 1, ruling 2: the Cast BUTTON stays enabled
            # here on purpose (`ArtifactsPane.compose`'s own disabled
            # condition is "no default AND no presets exist at all" --
            # presets exist here, just none chosen as the default), so a
            # press in this state must still be refused, but with copy that
            # tells the user what to do about it -- not the raw
            # `ScriptCastError` `generate_script` would otherwise produce
            # (`"briefing preset None does not exist"`, honest but useless
            # as an instruction).
            self._notify_watchlists(
                "Choose a default preset in the toolbar, or create one via "
                "Presets…, before casting.",
                severity="warning",
                markup=False,
            )
            return
        if self._cast_in_flight:
            running_id = self._cast_in_flight_briefing_id
            message = (
                f"A script is already being cast for briefing {running_id}. "
                "Nothing else was started."
                if running_id is not None
                else "A script is already being cast. Nothing else was started."
            )
            self._notify_watchlists(message, severity="warning", markup=False)
            return
        briefing_id = briefing.get("id")
        self._cast_in_flight = True
        self._cast_in_flight_briefing_id = briefing_id
        # Cast the die on the UI thread, alongside the rest of this
        # synchronous snapshot -- the `_briefing_default_preset_id` read
        # `handle_generate_briefing_requested` already does the same way,
        # for the same reason: not read again later inside the worker,
        # where a concurrent picker write could otherwise change it out
        # from under a cast already in flight for THIS briefing.
        preset_id = self._briefing_default_preset_id
        self.run_worker(
            self._cast_script(db, briefing_id, preset_id),
            exclusive=True,
            group="wl-cast",
        )

    async def _cast_script(
        self, db: Any, briefing_id: int, preset_id: int | None
    ) -> None:
        """Worker body: sweep, cast, repaint. Sibling of `_generate_briefing`.

        `generate_script`'s own DB calls (`_start_script`, `_finish_script_
        success`/`_finish_script_failure`) already run through `asyncio.
        to_thread` internally -- see that function's own docstring -- but a
        DATABASE error inside any of them still propagates OUT of
        `generate_script` uncaught: it only wraps the chat-call/parse block
        in its own try/except, not the whole function. An exception
        escaping a Textual worker with the default `exit_on_error=True`
        takes the whole application down, so -- exactly like
        `_generate_briefing` -- the call is wrapped in a bare `except` that
        turns any surviving exception into a toast instead of a crash.

        `ScriptCastError` is caught FIRST and separately: it is `generate_
        script`'s own honest, pre-flight refusal (the briefing is not
        `complete`, or the preset does not exist) -- a message safe to show
        verbatim (see that exception's own docstring), not a database
        failure to hide behind a generic toast.

        Phase 4 Task 1 (survey finding (c)): the sweep is now followed by a
        `blocking` check, mirroring `_generate_briefing`'s own -- a row that
        SURVIVES `_sweep_and_guard_cast`'s sweep because it is claimed by a
        live in-process cast refuses THIS attempt instead of starting a
        second, concurrent one over the top of it. Deliberately NOT
        mirroring `_generate_briefing`'s `recovered` branch too: unlike a
        briefing, `briefing_scripts` has no one-COMPLETE-row-per-briefing
        invariant (a briefing may be cast many times), so a zombie this
        sweep actually recovers (i.e. NOT `blocking` -- nothing claims it)
        must not itself refuse a fresh cast the way a recovered zombie
        briefing refuses a fresh generation; the same press both recovers
        the zombie AND casts a real script, exactly as it always has
        (`test_casting_recovers_a_zombie_script_via_its_own_sweep`).
        """
        chachanotes_db = getattr(self.app_instance, "chachanotes_db", None)
        load_character = (
            self._cast_load_character if chachanotes_db is not None else None
        )
        try:
            try:
                _recovered, blocking = await asyncio.to_thread(
                    self._sweep_and_guard_cast,
                    db,
                    briefing_id,
                    active_cast_claim_row_ids(),
                    pending_cast_claim_briefing_ids(),
                )
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                logger.warning(
                    f"Script guard failed for briefing {briefing_id}: "
                    f"{type(exc).__name__}"
                )
                self._notify_watchlists(
                    "Failed to check this briefing's scripts. Nothing was "
                    "started.",
                    severity="error",
                    markup=False,
                )
                return
            if blocking:
                # Survived the sweep because a live in-process claim holds
                # it -- not ours to duplicate. Mirrors `_generate_briefing`'s
                # own `blocking` refusal; see this method's own docstring for
                # why there is no `recovered`-branch sibling here.
                self._notify_watchlists(
                    f"{', '.join(blocking)} is already being cast for this "
                    "briefing. Nothing else was started.",
                    severity="warning",
                    markup=False,
                )
                return
            try:
                row = await generate_script(
                    db,
                    briefing_id,
                    preset_id=preset_id,
                    load_character=load_character,
                )
                # The freshly cast script (whatever its outcome) is the one
                # on screen, exactly like `_generate_briefing`'s own
                # `select_briefing_id=generated_id` -- re-resolved against
                # the reload below.
                self._selected_script = row
            except ScriptCastError as exc:
                self._notify_watchlists(str(exc), severity="warning", markup=False)
            except Exception as exc:  # noqa: BLE001 - a worker crash exits the app
                logger.warning(
                    f"Script cast failed for briefing {briefing_id}: "
                    f"{type(exc).__name__}"
                )
                self._notify_watchlists(
                    "Could not cast a script: the watchlist database could "
                    "not be reached. Nothing was recorded.",
                    severity="error",
                    markup=False,
                )
        finally:
            self._cast_in_flight = False
            self._cast_in_flight_briefing_id = None
            # Repaint on every path: a refusal may have just failed a
            # zombie row, and a completed cast has a new script to show.
            if self.is_attached:
                await self._load_briefings()

    # --- Artifacts: synthesizing and playing a script's audio (spec #2 --
    # phase 2b, Task 7) --------------------------------------------------
    #
    # Sibling of the Cast machinery immediately above in every respect: an
    # in-flight flag (`_audio_in_flight`), own worker group (`wl-audio`,
    # `exclusive=True`), claimed at DISPATCH time for the identical reason
    # `handle_cast_script_requested`'s docstring gives -- a check made
    # inside the worker body leaves a window where two presses both pass.
    # Unlike Cast (which has no one-generating-row-per-briefing invariant),
    # audio's own zombie sweep runs in the SAME two separate seams `_cast_
    # script`'s own comment names: `_load_briefings` sweeps whenever
    # Artifacts loads (gated on `_audio_sweep_is_safe`), and `_synthesize_
    # audio` below sweeps again at its own front, exactly where `_cast_
    # script` sweeps for Cast. Both are now claim-aware via
    # `active_audio_claim_row_ids()` (phase 4 Task 1; row-scoped, not
    # merely script-scoped, since task-1890 -- mirroring task-1812's
    # briefings-side fix), so a live in-process render -- this screen's
    # own, or another in-process caller's -- survives either sweep
    # unconditionally.
    #
    # Phase 4 Task 1 investigated whether Synthesize needs the SAME
    # `blocking` refusal Cast just gained (survey finding (c)'s sibling
    # question): structurally, yes -- `_synthesize_audio` had no `blocking`
    # check either, so two presses could in principle start two concurrent
    # renders for the same script. Phase 4 left it AS-IS, deliberately, as a
    # natural small follow-up; task-1811 is that follow-up: `_synthesize_
    # audio` below now runs `_sweep_and_guard_audio`, the identical shape as
    # `_sweep_and_guard_cast`, and refuses on `blocking` exactly like
    # `_cast_script` does.

    def _audio_sweep_is_safe(self) -> bool:
        """Whether `fail_interrupted_audio` may run right now.

        Sibling of `_cast_sweep_is_safe`: a load racing a synthesis THIS
        screen started must not fail that attempt's own `generating` row
        out from under it, so the load path only sweeps when nothing this
        screen started is still in flight.
        """
        return not self._audio_in_flight

    async def _fail_interrupted_audio_if_safe(self, db: Any, script_id: int) -> int:
        """Zombie recovery for the Artifacts-load path's audio, off the UI
        thread. Sibling of `_fail_interrupted_scripts_if_safe`, scoped to
        one script's audio renders rather than one briefing's scripts.

        `active_audio_claim_row_ids()` is snapshotted HERE, on the UI
        thread, before the sweep is dispatched -- see `_fail_interrupted_
        briefings_if_safe`'s own docstring for why a snapshot, not a live
        read, is required, and row-scoped rather than script-scoped
        (task-1890, mirroring task-1812's `active_briefing_claim_row_ids`).

        `pending_audio_claim_script_ids()` is snapshotted here too, same
        thread, same instant, closing the window `exclude` alone cannot: a
        claim taken but whose row id has not yet been recorded, still
        inside `generate_script_audio`'s own `db.create_briefing_audio`
        `to_thread` call. Passed as `exclude_scripts`, never in place of
        `exclude`.
        """
        if not self._audio_sweep_is_safe():
            return 0
        claims = active_audio_claim_row_ids()
        pending = pending_audio_claim_script_ids()
        return await asyncio.to_thread(
            fail_interrupted_audio,
            db,
            script_id,
            exclude=claims,
            exclude_scripts=pending,
        )

    @staticmethod
    def _audio_row_label(row: Mapping[str, Any]) -> str:
        """Name one audio render the way a toast has to: which row, and
        when. Sibling of `_script_row_label`, for the identical reason
        (task-1811).
        """
        return (
            f"audio {row.get('id')} "
            f"(started {row.get('created_at') or 'at an unknown time'})"
        )

    def _sweep_and_guard_audio(
        self,
        db: Any,
        script_id: int,
        exclude: Collection[int],
        exclude_scripts: Collection[int] = (),
    ) -> tuple[list[str], list[str]]:
        """Zombie sweep, then the generating-check, for a synthesis. Runs
        off the UI thread. Sibling of `_sweep_and_guard_cast` -- see that
        method's own docstring for the full reasoning; this is the
        identical shape, scoped to one script's audio renders instead of
        one briefing's scripts (task-1811, mirroring Cast's own `blocking`
        refusal from phase 4 Task 1 onto Synthesize; row-scoped `exclude`
        plus `exclude_scripts` since task-1890 -- see `_sweep_and_guard_
        cast`'s own docstring for the identical `exclude_briefings`
        reasoning).

        Returns:
            `(recovered, blocking)`, exactly like `_sweep_and_guard_cast`.
        """
        stuck = [
            self._audio_row_label(row)
            for row in db.list_briefing_audio(script_id)
            if str(row.get("status") or "").strip().lower() == STATUS_GENERATING
        ]
        fail_interrupted_audio(
            db, script_id, exclude=exclude, exclude_scripts=exclude_scripts
        )
        blocking = [
            self._audio_row_label(row)
            for row in db.list_briefing_audio(script_id)
            if str(row.get("status") or "").strip().lower() == STATUS_GENERATING
        ]
        return stuck, blocking

    @on(SynthesizeAudioRequested)
    def handle_synthesize_audio_requested(
        self, event: SynthesizeAudioRequested
    ) -> None:
        """Claim the one-synthesis-at-a-time guard, then dispatch.

        Mirrors `handle_cast_script_requested` exactly: `_audio_in_flight`
        is claimed HERE, before `run_worker`, and not inside the worker
        body, for the identical reason that handler's own docstring gives.
        """
        event.stop()
        db = self._briefings_db()
        script = self._selected_script
        if db is None or script is None:
            self._notify_watchlists(
                "Select a script to synthesize its audio.",
                severity="warning",
                markup=False,
            )
            return
        if self._audio_in_flight:
            running_id = self._audio_in_flight_script_id
            message = (
                f"Audio is already being synthesized for script {running_id}. "
                "Nothing else was started."
                if running_id is not None
                else "Audio is already being synthesized. Nothing else was "
                "started."
            )
            self._notify_watchlists(message, severity="warning", markup=False)
            return
        script_id = script.get("id")
        self._audio_in_flight = True
        self._audio_in_flight_script_id = script_id
        # Snapshotted on the UI thread, alongside the rest of this
        # synchronous dispatch -- the `preset_id` read
        # `handle_cast_script_requested` already does the same way, for
        # the same reason: read once, here, not again later inside the
        # worker where a concurrent app-level rebind could change it out
        # from under a synthesis already in flight for THIS script.
        tts_service = getattr(self.app_instance, "tts_service", None)
        profile_service = getattr(self.app_instance, "_tts_profile_service", None)
        self.run_worker(
            self._synthesize_audio(db, script_id, tts_service, profile_service),
            exclusive=True,
            group="wl-audio",
        )

    async def _synthesize_audio(
        self, db: Any, script_id: int, tts_service: Any, profile_service: Any
    ) -> None:
        """Worker body: sweep, synthesize, repaint. Sibling of `_cast_script`.

        `generate_script_audio`'s own DB calls already run through
        `asyncio.to_thread` internally (Task 6's own docstring), but a
        DATABASE error inside any of them still propagates OUT of
        `generate_script_audio` uncaught -- that is Task 6's deliberate
        "DB errors propagate" contract, the caller's own worker is where
        they must be wrapped. An exception escaping a Textual worker with
        the default `exit_on_error=True` takes the whole application down,
        so -- exactly like `_cast_script` -- the call is wrapped in a bare
        `except` that turns any surviving exception into a toast instead
        of a crash.

        `AudioGenerationError` is caught FIRST and separately: it is
        `generate_script_audio`'s own honest, pre-flight refusal (the
        script is not `complete`, or its turns/roster snapshot cannot be
        parsed) -- a message safe to show verbatim (see that exception's
        own docstring), not a database failure to hide behind a generic
        toast.

        The sweep is claim-aware (`active_audio_claim_row_ids()`, row-scoped
        since task-1890), like every other sweep call site (phase 4 Task
        1), and is now followed by a `blocking` check, mirroring `_cast_
        script`'s own (task-1811): a row that SURVIVES `_sweep_and_guard_
        audio`'s sweep because it is claimed by a live in-process synthesis
        refuses THIS attempt instead of starting a second, concurrent one
        over the top of it. Unlike `_cast_script`'s own `recovered` branch,
        there is no one-COMPLETE-row-per-script invariant here either (a
        script may be synthesized many times), so a zombie this sweep
        actually recovers (i.e. NOT `blocking`) does not itself refuse a
        fresh synthesis -- the same press both recovers the zombie AND
        synthesizes real audio (`test_synthesizing_recovers_a_zombie_audio_
        row_via_its_own_sweep`). Row-scoping (task-1890) also means this
        `blocking` toast no longer names a crash-zombie row shielded by an
        unrelated live claim on the same script -- the zombie is swept by
        the same call, before `blocking` is computed -- once that claim's
        row id is recorded. In the brief window before recording, the
        pending-claim guard deliberately spares the whole script (zombie
        included), so the toast can still name one there.
        """
        try:
            try:
                _recovered, blocking = await asyncio.to_thread(
                    self._sweep_and_guard_audio,
                    db,
                    script_id,
                    active_audio_claim_row_ids(),
                    pending_audio_claim_script_ids(),
                )
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                logger.warning(
                    f"Audio guard failed for script {script_id}: "
                    f"{type(exc).__name__}"
                )
                self._notify_watchlists(
                    "Failed to check this script's audio. Nothing was "
                    "started.",
                    severity="error",
                    markup=False,
                )
                return
            if blocking:
                # Survived the sweep because a live in-process claim holds
                # it -- not ours to duplicate. Mirrors `_cast_script`'s own
                # `blocking` refusal; see this method's own docstring for
                # why there is no `recovered`-branch sibling here.
                self._notify_watchlists(
                    f"{', '.join(blocking)} is already being synthesized "
                    "for this script. Nothing else was started.",
                    severity="warning",
                    markup=False,
                )
                return
            try:
                row = await generate_script_audio(
                    db,
                    script_id,
                    tts_service=tts_service,
                    profile_service=profile_service,
                )
                # The freshly synthesized audio (whatever its outcome) is
                # the one on screen, re-resolved against the reload below.
                self._loaded_script_audio = row
            except AudioGenerationError as exc:
                self._notify_watchlists(str(exc), severity="warning", markup=False)
            except Exception as exc:  # noqa: BLE001 - a worker crash exits the app
                logger.warning(
                    f"Audio synthesis failed for script {script_id}: "
                    f"{type(exc).__name__}"
                )
                self._notify_watchlists(
                    "Could not synthesize audio: the watchlist database "
                    "could not be reached. Nothing was recorded.",
                    severity="error",
                    markup=False,
                )
        finally:
            self._audio_in_flight = False
            self._audio_in_flight_script_id = None
            # Repaint on every path: a refusal may have just failed a
            # zombie row, and a completed (or failed) attempt has a new
            # audio row to show.
            if self.is_attached:
                await self._load_briefings()

    @on(PlayAudioRequested)
    def handle_play_audio_requested(self, event: PlayAudioRequested) -> None:
        """Play the selected script's audio file, if it still exists.

        Never routed through a worker/`asyncio.to_thread`: `play_audio_
        file` only spawns a detached subprocess and returns
        (`SimpleAudioPlayer.play`) -- `TTSEventHandler.handle_tts_
        playback`'s own "play" branch calls it exactly this way, direct,
        with no thread hop. Playback state itself is never held here or
        on the pane: the shared `SimpleAudioPlayer` singleton (`TTS.
        audio_player.get_audio_player`) is the only source of truth for
        "what's currently loaded" -- see `ArtifactsPane`'s own module
        docstring on why (a script selection recomposes every widget this
        pane renders).

        A missing file is refused with a toast rather than handed to the
        player: `ArtifactsPane.compose` already disables Play for exactly
        this case (`_audio_file_is_playable`), so reaching here with no
        file, or a file since deleted, means the disk state changed
        between that render and this press -- an honest race, not a bug
        to silently swallow.

        Qodo review round 1, FIX B: `audio_file_path_is_safe` is checked
        BEFORE any filesystem access -- a path that resolves outside
        `briefing_audio_dir()` (a tampered or corrupted row) is treated
        exactly like the "no file at all" case: a silent return, no
        `.exists()` probe, no exception. `ArtifactsPane.compose` already
        disables Play for this case too (`_audio_file_is_playable` uses the
        same helper), so reaching here with an unsafe path is the same kind
        of race as the missing-file case above, not a new failure mode.
        """
        event.stop()
        row = self._loaded_script_audio
        file_path = row.get("file_path") if row else None
        if not file_path:
            return
        if not audio_file_path_is_safe(file_path):
            return
        path = Path(str(file_path))
        if not path.exists():
            self._notify_watchlists(
                "This audio file no longer exists on disk.",
                severity="warning",
                markup=False,
            )
            return
        play_audio_file(path)

    @on(StopAudioRequested)
    def handle_stop_audio_requested(self, event: StopAudioRequested) -> None:
        """Stop playback, but only if THIS script's audio is what the
        shared player actually has loaded.

        Delegates to `tts_events.stop_audio_playback_if_current` rather
        than reimplementing its comparison here: that function's own
        docstring records the task-559 fix rounds (the guard must key off
        the player's LIVE `get_current_file()`, never a local cache with a
        TTL -- `SimpleAudioPlayer` is a single-slot, APP-WIDE singleton, so
        a bare `.stop()` could silence a completely unrelated clip playing
        elsewhere, e.g. Console TTS). A private copy of that same
        comparison here would only ever be one edit away from drifting
        from it -- exactly the failure mode that cost this branch a whole
        task to reconcile for the two legacy TTS id-builders (Task 2's own
        carried finding, `TTS/legacy_request_builder.py`'s module
        docstring). Imported locally rather than at module scope, matching
        `chat_screen.py`'s own lazy-import convention for this exact
        module: `Event_Handlers.TTS_Events.tts_events` pulls in the TTS
        adapter package (`tldw_chatbook.TTS`'s own imports), a cost most
        Watchlists sessions never need to pay just to stop a clip.
        """
        event.stop()
        row = self._loaded_script_audio
        file_path = row.get("file_path") if row else None
        if not file_path:
            return
        from ...Event_Handlers.TTS_Events.tts_events import (
            stop_audio_playback_if_current,
        )

        stop_audio_playback_if_current(Path(str(file_path)))

    def _items_status_kwargs(
        self, status_filter: str | None = None
    ) -> dict[str, Any]:
        """The status predicate the item PAGE should be fetched with.

        Review wave, I2. TASK-2301 made the Reader query ask for every status,
        which fixed "triaged items are unreachable" and quietly broke a
        different guarantee: the query pages at 50 rows and the pane's filter
        is applied in memory afterwards (`ItemsPane._filtered_items` never
        re-queries), so the page went from "the newest 50 UNREAD items" to
        "the newest 50 items of any status". On a source with 300 items whose
        newest 50 have all been triaged, picking "New" showed ZERO rows while
        the rail -- which this same branch made accurate -- honestly reported
        200 unread. Two numbers on one screen disagreeing about the same fact,
        which is the defect class this batch exists to remove.

        Pushing the active filter into the query makes a page 50 rows OF THE
        FILTERED STATUS, so "Unread" can reach unread items however deep they
        sit. TASK-3072 renames the filter vocabulary to the reader's
        Unread/All pair (`_normalize_items_status_filter`): "unread" queries
        `status="new"`; "all" queries `_READER_ALL_STATUSES` -- every status
        a reader can still act on, excluding `ignored` (triaged away on
        purpose) and `error` (a Runs-tab concern), which the pre-reader
        pane's literal "all statuses" mixed in.

        The pane keeps its own in-memory filter as well. That is not redundant
        -- it is what pins the currently-open item into a view its status no
        longer matches (see `_filtered_items`), and it is what makes the list
        correct in the window between a filter change and its reload landing.
        """
        effective_filter = (
            self._items_status_filter
            if status_filter is None
            else status_filter
        )
        if _normalize_items_status_filter(effective_filter) == "unread":
            return {"status": "new"}
        return {"statuses": list(_READER_ALL_STATUSES)}

    @staticmethod
    def _scope_forces_unread(scope: TreeScope) -> bool:
        """Whether a committed or candidate scope owns the Unread filter."""
        return scope.kind == "unread" or (
            scope.kind == "source" and scope.parent_context == "unread"
        )

    def _effective_items_status_filter(
        self, scope: TreeScope | None = None
    ) -> str:
        """Return visible/query filter without mutating the manual choice."""
        candidate = self.tree_scope if scope is None else scope
        if self._scope_forces_unread(candidate):
            return "unread"
        return _normalize_items_status_filter(self._items_status_filter)

    def _items_filter_disabled_reason(
        self, scope: TreeScope | None = None
    ) -> str | None:
        """Explain the temporary filter override for contextual Unread."""
        candidate = self.tree_scope if scope is None else scope
        return (
            _UNREAD_CONTEXT_FILTER_REASON
            if self._scope_forces_unread(candidate)
            else None
        )

    def _items_page_key(
        self,
        *,
        scope: TreeScope,
        status: str,
        search: str,
    ) -> tuple[Any, ...]:
        """Return the page-independent identity of one Reader query."""
        return (
            self.runtime_backend,
            scope.kind,
            scope.parent_context,
            scope.watchlist_id,
            scope.source_id,
            _normalize_items_status_filter(status),
            search.strip().casefold(),
        )

    def _items_scope_query(
        self, scope: TreeScope | None = None
    ) -> dict[str, Any]:
        """The tree scope as `list_items` kwargs.

        `all` passes nothing (every source). A `source` scope collapses to its
        single `source_id`; watchlist membership (many-to-many) is resolved by
        the query, not here. This is the wiring the whole phase exists for:
        before it, the Reader fetched the newest 50 items of ANY source
        regardless of the rail selection.
        """
        scope = self.tree_scope if scope is None else scope
        if scope.kind == "starred":
            # TASK-3072 plan task 6: the Starred smart feed. The flag is
            # global (same ADR-018 semantics as the briefing queue), so no
            # membership predicate -- just the flag itself.
            return {"is_flagged": True}
        if scope.kind == "unread":
            # TASK-3791 plan task 4: All Unread. The node forces the unread
            # bucket regardless of the pane's filter -- `_load_items` drops
            # any `statuses` kwarg when this scope is active, because the
            # DB raises on status+statuses together.
            return {"status": "new"}
        if scope.kind == "today":
            return {"since": self._today_floor_iso()}
        if scope.kind == "source" and scope.source_id is not None:
            query: dict[str, Any] = {"source_id": scope.source_id}
            if scope.parent_context == "unassigned":
                query["unassigned_only"] = True
            elif scope.parent_context == "unread":
                query["status"] = "new"
            elif (
                scope.parent_context == "watchlist"
                and scope.watchlist_id is not None
            ):
                query["watchlist_id"] = scope.watchlist_id
            return query
        if scope.kind == "watchlist" and scope.watchlist_id is not None:
            return {"watchlist_id": scope.watchlist_id}
        if scope.kind == "unassigned":
            return {"unassigned_only": True}
        return {}

    @staticmethod
    def _today_floor_iso() -> str:
        """Local midnight tonight's floor, as a UTC ISO string (TASK-3791).

        The Today feed is a LOCAL-day concept (the user's "today"), while
        `subscription_items` dates are UTC ISO -- so the floor is computed
        in local time and converted back to UTC before it reaches the
        `COALESCE(published_date, created_at) >= ?` string comparison, which
        is exact only between same-shape ISO strings.
        """
        local_midnight = datetime.now().astimezone().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return local_midnight.astimezone(timezone.utc).isoformat()

    def _with_open_item(
        self,
        page: list[dict[str, Any]],
        *,
        max_items: int | None = None,
    ) -> list[dict[str, Any]]:
        """`page`, guaranteed to contain the item the reader currently has open.

        Round 2, O2. Pushing the status filter into the query (I2) reopened
        the CRITICAL that `ItemsPane._filtered_items`'s pin exists to prevent,
        and its docstring names the scenario exactly: opening an item marks it
        read, so under a "New" filter it drops out of its own list the instant
        it is opened, and everything keyed off "where is the open item in the
        displayed list" fails at once -- `j` walks backwards from a not-found
        index and `k` is dead for the rest of the session.

        The pin can only retain what the query RETURNED. Pre-I2 the query
        returned every status, so it always had the open item to keep;
        afterwards a reload under `status="new"` came back without it and the
        item the user was reading vanished. Measured: filter New, open the
        only unread item, any Reader replacement -> `items == []`.

        Two fixes were on the table. Dropping the status predicate while an
        item is open was rejected: it un-fixes I2 for the whole time the
        reader is in use -- which is precisely when a user is triaging, and so
        precisely when "unread items past the newest 50 are unreachable"
        bites hardest. Carrying the open item alongside the page keeps both
        guarantees at once, and costs no query: the dict is the same object
        the reader, the pane and `_mark_item_read_on_open`'s in-place patch
        all already share, so its status is current by construction.

        Inserted in `created_at DESC` order rather than at either end, so the
        page keeps the ordering every other consumer assumes -- `j`/`k` walk
        this sequence, and an item teleporting to the top of the list when its
        status changed would be its own small lie.

        TASK-3072 removed the old "All statuses covers everything" skip: the
        reader's "all" is itself a restricted query (`_READER_ALL_STATUSES`),
        so BOTH filters can now return a page without the open item, and the
        pin applies unconditionally. Under "all" the common case still
        short-circuits on the open item being present; the pin only fires
        when the item genuinely fell out of the query -- e.g. it was ignored
        while open, which is exactly the "the article I'm reading vanished"
        moment this method exists to prevent.

        Args:
            page: The rows the backend returned for the current filter.
            max_items: Optional visible-row cap. A carried item replaces the
                final slot when it sorts beyond a full page.

        Returns:
            `page` unchanged when no item is open or when the open item is
            in it already; otherwise a sorted page containing that item,
            without exceeding `max_items` when supplied.
        """
        if max_items is not None:
            max_items = max(0, max_items)
            page = page[:max_items]
            if max_items == 0:
                return []

        open_item = self._selected_content_item
        if open_item is None:
            return page
        open_id = str(open_item.get("id") or "")
        if not open_id or any(str(row.get("id")) == open_id for row in page):
            return page
        carried = dict(open_item)
        created = effective_date(carried) or datetime.min.replace(tzinfo=timezone.utc)
        for index, row in enumerate(page):
            row_date = effective_date(row) or datetime.min.replace(
                tzinfo=timezone.utc
            )
            if row_date < created:
                inserted = [*page[:index], carried, *page[index:]]
                return inserted if max_items is None else inserted[:max_items]
        if max_items is not None and len(page) >= max_items:
            return [*page[:-1], carried]
        return [*page, carried]

    def _reader_item_query(
        self,
        *,
        scope: TreeScope | None = None,
        status: str | None = None,
        search: str | None = None,
    ) -> ReaderItemQuery:
        """Freeze one candidate Reader query from explicit screen intent."""
        candidate_scope = self.tree_scope if scope is None else scope
        candidate_status = (
            self._effective_items_status_filter(candidate_scope)
            if status is None
            else status
        )
        candidate_search = self._items_search_query if search is None else search
        kwargs = {
            **self._items_status_kwargs(candidate_status),
            **self._items_scope_query(candidate_scope),
        }
        if "status" in kwargs:
            kwargs.pop("statuses", None)
        normalized_search = candidate_search.strip()
        if normalized_search:
            kwargs["search"] = normalized_search
        return ReaderItemQuery.freeze(
            self._items_page_key(
                scope=candidate_scope,
                status=candidate_status,
                search=candidate_search,
            ),
            kwargs,
        )

    def _items_request_is_current(
        self, generation: int, query_key: tuple[Any, ...]
    ) -> bool:
        return (
            generation == self._items_snapshot_generation
            and query_key == self._items_pending_query_key
        )

    def _supersede_items_query_intent(
        self, *, scope: TreeScope | None = None
    ) -> None:
        """Park old rows while immediately invalidating older query work."""
        if scope is None:
            self._pending_tree_scope = None
        query = self._reader_item_query(scope=scope)
        self._items_snapshot_generation += 1
        self._items_pending_query_key = query.context_key
        self._items_inflight_replacement = None
        self._items_inflight_continuation = None
        self._items_page_loading = True
        self._push_items_pager_state()

    async def _publish_items_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        focus_first: bool,
        is_current: Callable[[], bool],
        commit: Callable[[], None],
        atomic_batch: bool = False,
    ) -> bool:
        """Mount rows and commit their authority together, with rollback."""
        notify = getattr(self.app_instance, "notify", None)
        async with self._items_page_presentation_lock:
            if not is_current():
                return False
            batch = self.app.batch_update() if atomic_batch else nullcontext()
            with batch:
                prior_rows = self._loaded_items
                pane: ArticleListPane | None = None
                prior_authority = self._items_search_results_authoritative
                if self._dom_is_live:
                    try:
                        pane = self.query_one("#watchlists-items-pane", ArticleListPane)
                        prior_authority = pane.search_results_authoritative
                        pane.search_results_authoritative = True
                        await pane.apply_page_items(rows, focus_first=focus_first)
                    except NoMatches:
                        pane = None
                    except asyncio.CancelledError:
                        if pane is not None:
                            pane.search_results_authoritative = prior_authority
                            await pane.apply_page_items(prior_rows, focus_first=False)
                        raise
                    except Exception as exc:
                        if pane is not None:
                            pane.search_results_authoritative = prior_authority
                            await pane.apply_page_items(prior_rows, focus_first=False)
                        logger.debug(
                            "Failed to present watchlist items (exception_type={}).",
                            type(exc).__name__,
                        )
                        if callable(notify) and not atomic_batch:
                            notify("Failed to load watchlist items.", severity="error")
                        return False
                if not is_current():
                    if pane is not None:
                        pane.search_results_authoritative = prior_authority
                        await pane.apply_page_items(prior_rows, focus_first=False)
                    return False
                commit()
                return True

    async def _replace_items_snapshot(
        self,
        *,
        scope: TreeScope | None = None,
        reason: Literal[
            "initial",
            "refresh",
            "filter",
            "search",
            "scope",
            "return_to_read",
        ],
        clear_reader_on_commit: bool = False,
        focus_first: bool = False,
    ) -> bool:
        """Load page one off-screen and publish only after rows mount."""
        if self.runtime_backend != "local":
            self._items_page_loading = False
            self._push_items_pager_state()
            return False
        query = self._reader_item_query(scope=scope)
        query_key = query.context_key
        inflight = self._items_inflight_replacement
        if inflight is not None and inflight[0] == query_key:
            return await asyncio.shield(inflight[1])

        completion = asyncio.get_running_loop().create_future()
        self._items_inflight_replacement = (query_key, completion)
        self._items_snapshot_generation += 1
        generation = self._items_snapshot_generation
        self._items_pending_query_key = query_key
        self._items_page_loading = True
        self._push_items_pager_state()
        result = False
        try:
            page = await self._controller.list_reader_items_page(
                runtime_backend=self.runtime_backend,
                limit=_ITEMS_PAGE_SIZE,
                **query.as_kwargs(),
            )
            if not isinstance(page, WatchlistItemPage):
                raise TypeError("Reader item service returned an invalid page")
            if not self._items_request_is_current(generation, query_key):
                return False
            backend_rows = list(page.items)
            first_page_rows = backend_rows
            displaced_rows: list[dict[str, Any]] = []
            if reason in {"filter", "search"}:
                first_page_rows = self._with_open_item(
                    first_page_rows, max_items=_ITEMS_PAGE_SIZE
                )
                visible_ids = {str(row.get("id") or "") for row in first_page_rows}
                displaced_rows = [
                    row
                    for row in backend_rows
                    if str(row.get("id") or "") not in visible_ids
                ]
                page = WatchlistItemPage(
                    items=tuple(first_page_rows),
                    has_more=page.has_more,
                    snapshot_max_item_id=page.snapshot_max_item_id,
                    snapshot_count=page.snapshot_count,
                    next_cursor=page.next_cursor,
                )
            candidate = ReaderItemSnapshot.start(query, page)
            if displaced_rows:
                candidate = candidate.with_pending_items(tuple(displaced_rows))
            rows = list(candidate.page(0))

            def commit() -> None:
                had_retry_state = self._items_retry_message is not None
                self._items_retry_message = None
                self._items_snapshot = candidate
                self._loaded_items = rows
                self._items_page_index = 0
                self._items_has_next = candidate.has_next(0)
                self._items_snapshot_count = candidate.snapshot_count
                self._items_pending_arrivals = candidate.pending_arrivals
                self._items_search_results_authoritative = True
                self._items_page_loading = False
                self._items_pending_query_key = None
                if clear_reader_on_commit:
                    if scope is not None:
                        self._apply_tree_scope(scope)
                    self._selected_content_item = None
                    self._selected_content_page_key = None
                    try:
                        pane = self.query_one(
                            "#watchlists-items-pane", ArticleListPane
                        )
                        pane.selected_item = None
                    except NoMatches:
                        pass
                    try:
                        content = self.query_one(
                            "#watchlists-content-pane", ContentPane
                        )
                        content.item = None
                        content.position = ""
                    except NoMatches:
                        pass
                    if self._pending_tree_scope == scope:
                        self._pending_tree_scope = None
                self._push_items_pager_state()
                self._restore_items_view_state()
                if had_retry_state:
                    self._request_surface_refresh(self._SURFACE_SECTION)

            result = await self._publish_items_rows(
                rows,
                focus_first=focus_first,
                is_current=lambda: self._items_request_is_current(
                    generation, query_key
                ),
                commit=commit,
                atomic_batch=clear_reader_on_commit,
            )
            if not result and self._items_request_is_current(generation, query_key):
                self._items_page_loading = False
                self._push_items_pager_state()
                if clear_reader_on_commit and scope is not None:
                    self._notify_pending_scope_failure(scope)
                    if self._pending_tree_scope == scope:
                        self._pending_tree_scope = None
                elif reason == "return_to_read":
                    self._show_items_retry_state()
            return result
        except asyncio.CancelledError:
            if self._items_request_is_current(generation, query_key):
                self._items_page_loading = False
                self._push_items_pager_state()
                if self._pending_tree_scope == scope:
                    self._pending_tree_scope = None
            raise
        except Exception as exc:
            if self._items_request_is_current(generation, query_key):
                logger.debug(
                    "Failed to load watchlist items (exception_type={}).",
                    type(exc).__name__,
                )
                self._items_page_loading = False
                self._push_items_pager_state()
                notify = getattr(self.app_instance, "notify", None)
                if clear_reader_on_commit and scope is not None:
                    self._notify_pending_scope_failure(scope)
                elif reason == "return_to_read":
                    self._show_items_retry_state()
                elif callable(notify):
                    notify("Failed to load watchlist items.", severity="error")
                if self._pending_tree_scope == scope:
                    self._pending_tree_scope = None
            return False
        finally:
            if not completion.done():
                completion.set_result(result)
            if self._items_inflight_replacement == (query_key, completion):
                self._items_inflight_replacement = None

    async def _present_cached_items_page(
        self, index: int, *, focus_first: bool = True
    ) -> bool:
        """Present an already-cached page without a backend request."""
        snapshot = self._items_snapshot
        if snapshot is None or index < 0 or index >= snapshot.page_count:
            return False
        generation = self._items_snapshot_generation
        rows = list(snapshot.page(index))

        def current() -> bool:
            return (
                generation == self._items_snapshot_generation
                and self._items_snapshot is snapshot
            )

        def commit() -> None:
            self._loaded_items = rows
            self._items_page_index = index
            self._items_has_next = snapshot.has_next(index)
            self._items_search_results_authoritative = True
            self._items_page_loading = False
            self._push_items_pager_state()
            self._restore_items_view_state()

        return await self._publish_items_rows(
            rows, focus_first=focus_first, is_current=current, commit=commit
        )

    _MAX_DUPLICATE_CONTINUATIONS = 100

    async def _load_next_items_page(self) -> bool:
        """Present cached forward rows or append one bounded continuation."""
        snapshot = self._items_snapshot
        if snapshot is None:
            return False
        next_index = self._items_page_index + 1
        if next_index < snapshot.page_count:
            return await self._present_cached_items_page(next_index)
        if not snapshot.has_next(self._items_page_index):
            return False
        inflight = self._items_inflight_continuation
        if inflight is not None:
            return await asyncio.shield(inflight)

        completion = asyncio.get_running_loop().create_future()
        self._items_inflight_continuation = completion
        generation = self._items_snapshot_generation
        query = snapshot.query
        candidate = snapshot
        self._items_page_loading = True
        self._push_items_pager_state()
        result = False

        def current() -> bool:
            return (
                generation == self._items_snapshot_generation
                and self._items_snapshot is snapshot
            )

        try:
            for _ in range(self._MAX_DUPLICATE_CONTINUATIONS):
                if candidate.has_more and candidate.cursor is not None:
                    page = await self._controller.list_reader_items_page(
                        runtime_backend=self.runtime_backend,
                        limit=_ITEMS_PAGE_SIZE,
                        **query.as_kwargs(),
                        snapshot_max_item_id=candidate.watermark,
                        after=candidate.cursor,
                    )
                    if not isinstance(page, WatchlistItemPage):
                        raise TypeError("Reader item service returned an invalid page")
                    if not current():
                        return False
                    candidate, appended = candidate.with_continuation(
                        page, page_size=_ITEMS_PAGE_SIZE
                    )
                elif candidate.pending_items:
                    candidate, appended = candidate.with_pending_page(
                        _ITEMS_PAGE_SIZE
                    )
                else:
                    break
                if not appended:
                    if candidate.has_more or candidate.pending_items:
                        continue
                    async with self._items_page_presentation_lock:
                        if not current():
                            return False
                        self._items_snapshot = candidate
                        self._items_has_next = False
                        self._items_page_loading = False
                        self._push_items_pager_state()
                    result = True
                    return True
                rows = list(candidate.page(candidate.page_count - 1))

                def commit() -> None:
                    self._items_snapshot = candidate
                    self._loaded_items = rows
                    self._items_page_index = candidate.page_count - 1
                    self._items_has_next = candidate.has_next(
                        self._items_page_index
                    )
                    self._items_search_results_authoritative = True
                    self._items_page_loading = False
                    self._push_items_pager_state()
                    self._restore_items_view_state()

                result = await self._publish_items_rows(
                    rows,
                    focus_first=True,
                    is_current=current,
                    commit=commit,
                )
                return result
            if current():
                self._items_page_loading = False
                self._push_items_pager_state()
            return False
        except asyncio.CancelledError:
            if current():
                self._items_page_loading = False
                self._push_items_pager_state()
            raise
        except Exception as exc:
            if current():
                logger.debug(
                    "Failed to load watchlist item page (exception_type={}).",
                    type(exc).__name__,
                )
                self._items_page_loading = False
                self._push_items_pager_state()
                notify = getattr(self.app_instance, "notify", None)
                if callable(notify):
                    notify("Failed to load watchlist items.", severity="error")
            return False
        finally:
            if not completion.done():
                completion.set_result(result)
            if self._items_inflight_continuation is completion:
                self._items_inflight_continuation = None

    @on(ItemSelected)
    async def handle_item_selected(self, event: ItemSelected) -> None:
        event.stop()
        if self.runtime_backend != "local":
            return
        async with self._items_page_presentation_lock:
            snapshot = self._items_snapshot
            selection_generation = self._items_snapshot_generation
            selection_page_key = (
                snapshot.query.context_key if snapshot is not None else None
            )
        # TASK-15464: fetch the DETAIL body BEFORE any of the selection
        # writes below, not after. `ContentPane.item` is a `recompose=True`
        # reactive, so merging `content` into `event.item` first means one
        # recompose per selection with the body already in it, instead of
        # an empty-bodied recompose immediately followed by a second one
        # once a background fetch lands -- exactly the recompose-storm
        # shape this whole audit exists to remove from this screen.
        await self._load_item_content(event.item)
        async with self._items_page_presentation_lock:
            current_snapshot = self._items_snapshot
            current_page_key = (
                current_snapshot.query.context_key
                if current_snapshot is not None
                else None
            )
            if (
                selection_generation != self._items_snapshot_generation
                or selection_page_key != current_page_key
            ):
                return
            self._select_entity(event.item)
            # Route to the reader (Task 4), independent of `_select_entity`'s
            # generic Inspector reconciliation above: Sources/Runs/Rules also
            # flow through `_select_entity`, and none of those dicts carry
            # `content_kind`/`content` -- pushing them into `ContentPane` would
            # render `render_for`'s article-fallback over the WRONG entity's
            # fields instead of leaving the reader showing the last real item.
            # Held on the screen (`_selected_content_item`), not just pushed to
            # the mounted pane, so `_build_content_pane` can re-seed a rebuilt
            # `ContentPane` the same way `_build_inspector_pane` re-seeds
            # `selected_entity` — see that seeding note above.
            self._selected_content_item = event.item
            self._selected_content_page_key = selection_page_key
            try:
                content = self.query_one("#watchlists-content-pane", ContentPane)
                content.item = event.item
                content.position = self._reader_position_text()
            except NoMatches:
                pass
            self._mark_item_read_on_open(event.item)

    async def _load_item_content(self, item: dict[str, Any] | None) -> None:
        """Backfill `item["content"]` from the DETAIL fetch (TASK-15464).

        `get_new_items`'s list-page projection no longer selects `content`
        (up to 50 rows' worth of full scraped article/diff text, on every
        Items-pane refresh, for a column no list row ever rendered -- the
        audit's named cost), so a freshly loaded list row carries no
        `content` key at all. This fetches it for exactly the item about to
        be opened, mutating `item` IN PLACE so the one dict object already
        shared by `ItemsPane.items`, `_selected_content_item`, and (about to
        be) `ContentPane.item` all see the same body once this returns --
        never a second, separate copy for the reader to drift from the list.

        A miss (`get_item_content` returning `None`) leaves `item` untouched
        rather than clearing an existing `content` key -- a caller that
        built its own item dict directly, bypassing this screen's own query
        entirely (every test that seeds `ItemsPane.items` with a synthetic
        dict already carrying `content`, e.g.
        `test_selecting_an_item_renders_it_in_the_content_region`), keeps
        working unchanged.

        `item is None` (nothing selected) is a silent no-op -- there is
        nothing to report. An actual FETCH failure (the active backend does
        not support single-item reads, the row no longer exists, a transient
        DB error) never raises into the caller -- content is the reader's
        body, not a status `handle_item_selected` is relying on this to
        report, matching `content_pane.render_for`'s own "never take the app
        down over a reader nicety" rule -- but it is not silent either: this
        is a background `_load*` read (`test_watchlists_check_now_failure.
        py`'s structural contract), and that exemption from the
        user-initiated-action "log at warning" rule is paid for with a
        toast, exactly like the sibling `_load_items`/`_load_run_detail`
        immediately around it. Without one, a denied `items.detail` policy
        or a database locked by a concurrent write would render
        byte-identically to "this item just has no body" -- an empty reader
        with nothing said.

        Args:
            item: The item about to be opened, or `None`. Mutated in place
                when a fetch returns a real value.
        """
        if item is None:
            return
        item_id = item.get("id")
        if item_id is None:
            return
        notify = getattr(self.app_instance, "notify", None)
        try:
            fetched = await self._controller.get_item_content(
                runtime_backend=self.runtime_backend,
                item_id=item_id,
            )
        except Exception:
            logger.debug("Failed to load watchlist item content for the reader.")
            if callable(notify):
                notify(
                    "Failed to load this item's full content.",
                    severity="error",
                )
            return
        if fetched is not None:
            item["content"] = fetched

    def _reader_position_text(self) -> str:
        """The reader footer's "N of M" (TASK-3072 plan task 9).

        M is the article list's `displayed_items()` -- the list the user is
        actually looking at, filter applied -- and N the open item's 1-based
        place in it, so `j` and the footer walk the same sequence. The pane
        holds no list state, which is why this is computed here and pushed
        into its `position` reactive (the `_selected_content_item` re-seed
        pattern). Empty when nothing is open (the footer is absent then,
        never "0 of 0") or when the open item is not in the displayed list.
        """
        item = self._selected_content_item
        if item is None:
            return ""
        try:
            pane = self.query_one("#watchlists-items-pane", ArticleListPane)
        except NoMatches:
            return ""
        items = pane.displayed_items()
        open_id = str(item.get("id") or "")
        for index, candidate in enumerate(items):
            if str(candidate.get("id")) == open_id:
                return f"{index + 1} of {len(items)}"
        return ""

    def _mark_item_read_on_open(self, item: dict[str, Any] | None) -> None:
        """Opening an item in the reader marks it read (Task 5).

        Only fires the "new" -> "reviewed" transition: an item already at
        "reviewed"/"ingested"/"ignored"/"error" left the unread bucket
        through some other deliberate action already, and re-opening it here
        must not clobber that back down to a bare "reviewed". Silent
        (`notify_toast=False`) because this fires on every selection, not on
        a deliberate user request for a status change -- a toast per click
        would be noise, unlike the explicit unread toggle.

        `refresh=False` + `patch_item=item` (Task 5 fix round 1, CRITICAL):
        this fires on every single item SELECTION, not just a deliberate
        button click. A default `refresh` reloads `ItemsPane.items` and
        calls `_refresh_overview_data()`, which used to set `overview_data`,
        then a `reactive({}, recompose=True)` -- a SCREEN-level recompose,
        which rebuilt every region via its factory
        (`_build_detail_pane`/`_build_content_pane`/etc.), replacing the live
        `ItemsPane`/`DataTable` instances wholesale. Proven live: with the
        default refresh, one item selection detached the old `ItemsPane`,
        reset the table cursor to 0, cleared screen focus, and a SECOND
        arrow-key press did nothing at all. TASK-2200 took the recompose off
        that reactive (`watch_overview_data` patches the three surfaces that
        read it), so the crash-shaped half of this is gone; the reload of
        every item on every arrow key is not, which is why this path still
        passes `refresh=False`. `patch_item` mutates the same dict object
        already held by `ItemsPane.items`/`_selected_content_item`/
        `ContentPane.item` in place instead, so a later status check sees
        "reviewed" without forcing a rebuild.

        This reuses the exact status column Ingest/Ignore/the unread toggle
        already write -- `SubscriptionsDB.mark_item_status`, keyed by the
        item's own row id, not by any (watchlist, item) pair -- so it is
        global by construction: the same article read from "All sources" is
        read in every watchlist whose sources include it.

        The "new" check above is a cheap pre-filter against the CACHED dict
        only, so a plain non-"new" selection dispatches nothing at all -- but
        the write itself is gated again, against the BACKEND, immediately
        before it happens inside `_drain_item_status` (`gate=True` below;
        fix wave, F2b, Important, and TASK-1541's Qodo redesign afterwards).
        Ingest/Ignore never patch this dict (they pass no `patch_item=`), so
        if either ran behind this cache's back -- or the dict went stale for
        any other reason -- the cached "new" check above cannot see it;
        asking the backend right before the write, mirroring the unread
        toggle's own `_blocking_status_for` guard, closes that regardless of
        the cause.
        """
        if item is None:
            return
        if str(item.get("status") or "").strip().lower() != "new":
            return
        item_id = item.get("id")
        if item_id is None:
            return
        self._dispatch_item_status(
            item_id,
            _ItemStatusIntent(
                status="reviewed",
                notify_toast=False,
                refresh=False,
                patch_item=item,
                gate=True,
            ),
        )
        self._request_tree_counts_refresh()

    #: How long the rail's unread counts may lag behind a run of silent
    #: mark-read-on-open writes. Long enough that a fast `j`/`j`/`j` walk pays
    #: for ONE reload rather than one per keystroke; short enough that the
    #: number is right by the time a reader looks up at it.
    _TREE_COUNTS_REFRESH_DEBOUNCE_SECONDS = 0.6

    def _request_tree_counts_refresh(self) -> None:
        """Reload the rail's counts once the user stops opening items.

        Review wave, Minor 6. The rail legend says "Counts: unread items"
        unconditionally, and opening an item moves it out of the unread bucket
        -- but `_mark_item_read_on_open` deliberately passes `refresh=False`
        (it fires on every arrow key, and a reload per keystroke was proven
        live to detach the mounted `ItemsPane` and drop focus). So the number
        lagged by however many items had been opened since the last deliberate
        action, and the honest choices were to weaken the label or to remove
        the lag.

        This removes the lag, at one query pair per PAUSE rather than per
        keystroke: each call re-arms a single timer, so a burst of `j`
        presses collapses into one `_load_tree_data()` after the burst ends.
        `_load_tree_data` is itself `@work(exclusive=True, group="wc_tree")`
        and publishes through TASK-2200's surface-refresh drain, so nothing
        here can stack up rail rebuilds either.

        The timer is stopped before it is replaced -- Textual keeps a live
        `Timer` running until it is stopped or its node unmounts, and this
        method can be reached many times a second.
        """
        timer = getattr(self, "_tree_counts_refresh_timer", None)
        if timer is not None:
            timer.stop()
        self._tree_counts_refresh_timer = self.set_timer(
            self._TREE_COUNTS_REFRESH_DEBOUNCE_SECONDS,
            self._load_tree_data,
        )

    @on(UnreadToggleRequested)
    def handle_unread_toggle_requested(self, event: UnreadToggleRequested) -> None:
        """The explicit way back (Task 5): marking read is otherwise
        irreversible from the reader, since it drops the item out of the
        unread list. Reuses the same global status column, just the other
        direction.

        Refuses to downgrade a status that is not a read/unread state at all
        (whole-branch review, Minor -- data loss). `_mark_item_read_on_open`
        already declines to touch anything but `new`, for the same reason:
        `ingested`, `ignored` and `error` are terminal records of something
        that happened to the item, and this button would overwrite them with
        `new`, losing the fact of an ingest and dropping the item out of the
        Ingested filter that was the only way to find it again.

        The refusal is decided by `_drain_item_status`'s gate
        (`_item_status_write_allowed`), by asking the backend, NOT from
        `event.item` (re-review, Important). `event.item` is
        `ContentPane.item` -- the dict the screen has held since the item was
        selected -- and `handle_ingest_requested`/`handle_ignore_requested`
        dispatch with no `patch_item=`, so that dict is never updated when
        they run. `patch_item=` is passed by exactly one dispatch path in the
        whole app (`_mark_item_read_on_open`) and by neither of those two.
        Ingest an open item and the reader's dict still says `reviewed`, so a
        guard reading `event.item` never fires and the button destroys the
        ingest anyway -- reproduced end to end.
        """
        event.stop()
        item = event.item
        if item is None:
            return
        item_id = item.get("id")
        if item_id is None:
            return
        self._dispatch_item_status(item_id, _ItemStatusIntent(status="new", gate=True))

    @on(ViewSnapshotRequested)
    def handle_view_snapshot_requested(self, event: ViewSnapshotRequested) -> None:
        """The Inspector's stored-page affordances (TASK-1494).

        Deferred to a worker for the same reason every other DB-touching
        handler on this screen is: `_open_snapshot_view` awaits a service
        call and, on success, `push_screen_wait`, neither legal directly
        inside a synchronous `@on` handler. No `exclusive=True`: like
        `handle_kept_briefings_requested`'s sibling note explains, cancelling
        a modal-owning worker mid-view would leave the modal on the screen
        stack with nothing left to dismiss it.
        """
        event.stop()
        item = event.item
        if item is None:
            return
        self.run_worker(
            self._open_snapshot_view(item, event.which),
            group="wl-view-snapshot",
        )

    async def _open_snapshot_view(self, item: dict[str, Any], which: str) -> None:
        """Resolve `which` against `url_snapshots` and show it, or say why not.

        AC#2: an absent snapshot (a `full_page` request against an item
        whose page was somehow never stored, or a `previous` request when
        only one snapshot exists yet) degrades to an honest toast, never an
        empty modal and never a silent no-op -- the two failure modes the
        acceptance criterion explicitly rules out.

        Args:
            item: The normalized watchlist item `ViewSnapshotRequested`
                carried -- `source_id`/`url` key the `url_snapshots` lookup.
            which: `"full_page"` (the newest snapshot) or `"previous"` (the
                second-newest).
        """
        service = self._local_watchlists_service()
        source_id = item.get("source_id")
        url = item.get("url")
        if service is None or source_id is None or not url:
            self._notify_watchlists(
                "Could not look up this page's stored snapshots.",
                severity="error",
                markup=False,
            )
            return
        try:
            snapshots = await service.get_url_snapshots(source_id, url, limit=2)
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to load url_snapshots for the reader's snapshot viewer."
            )
            self._notify_watchlists(
                "Could not load this page's stored snapshots.",
                severity="error",
                markup=False,
            )
            return
        # Closed vocabulary, refused rather than defaulted (task-1494 Qodo):
        # an unrecognized `which` silently treated as "previous" would open
        # the WRONG snapshot after a typo'd/future caller, with a misleading
        # toast on the absent case. Type-only log; nothing user-derived.
        _SNAPSHOT_INDEX = {"full_page": 0, "previous": 1}
        index = _SNAPSHOT_INDEX.get(which)
        if index is None:
            logger.warning(
                f"ViewSnapshotRequested with unknown which={which!r}; refusing."
            )
            return
        if index >= len(snapshots):
            self._notify_watchlists(
                "No page snapshot saved yet for this item."
                if which == "full_page"
                else "No previous snapshot yet for this page.",
                severity="warning",
                markup=False,
            )
            return
        snapshot = snapshots[index]
        await self.app.push_screen_wait(
            SnapshotViewModal(
                url=url,
                created_at=snapshot.get("created_at"),
                content=snapshot.get("extracted_content"),
            )
        )

    async def _item_status_write_allowed(
        self, item_id: Any, intent: "_ItemStatusIntent"
    ) -> bool:
        """The backend terminal-status gate, re-asked right before the write.

        TASK-1541 (Qodo redesign). Shared by both gated intents (the unread
        toggle and mark-read-on-open, `_ItemStatusIntent.gate=True`) --
        previously this check was duplicated between `_mark_item_unread` and
        `_confirm_new_then_mark_item_read_on_open`, one per caller; both are
        gone now, folded into this one method plus `_drain_item_status`'s
        loop.

        Deciding from a live backend query, rather than keeping the screen's
        cached dicts patched at every status writer (re-review, Important).
        Both were on the table; this leaves strictly fewer places able to
        drift:

        * Patching would have to keep the reader's dict in step for every
          present and future writer of an item status, and it structurally
          cannot cover a status this screen did not write at all -- a
          scheduled run marking an item `error`, the server backend, or a
          second screen.
        * Asking has exactly one decision point, and it asks the system of
          record, so it is right no matter who moved the item or when.

        Called from INSIDE `_drain_item_status`, immediately before the
        write, not once at dispatch time: a desired entry can sit queued for
        a moment (another item's write draining, or simply the worker not
        yet scheduled), and only re-asking right before the write catches
        anything that moved the item to a terminal status during that wait
        -- including another intent for the SAME item, drained just before
        this one, that happened to be an Ingest/Ignore.

        `_loaded_items` is NOT the system of record and cannot be used here.
        Until TASK-2301 it could not even have been read as a hint:
        `local_watchlists_service.list_items` collapsed `status=None` to
        `status="new"`, so an ingested item was not merely stale in that
        cache -- it was absent from it entirely. That collapse is gone and the
        cache now carries every status, which changes nothing here: it is
        still a page snapshot taken at load time, of at most `limit` rows,
        and this gate is deciding whether a write may destroy an ingest. Ask
        the row.

        Fails CLOSED. If the backend cannot be asked, the write is refused:
        marking unread/read is a convenience the user can repeat, whereas
        overwriting an ingest is not recoverable, so an unanswered question
        must not resolve in favour of the destructive branch.

        Args:
            item_id: Normalized id of the item to check.
            intent: The queued write this gate is deciding whether to allow.

        Returns:
            `True` if the write may proceed, `False` if it was refused (a
            toast already fired when `intent.notify_toast`).
        """
        label = "unread" if intent.status == "new" else intent.status
        try:
            blocking = await self._blocking_status_for(item_id)
        except Exception:
            logger.opt(exception=True).warning(
                f"Could not confirm an item's status before marking it {label}; "
                "leaving it unchanged."
            )
            if intent.notify_toast:
                self.notify(
                    "Could not confirm this item's current status, so it was "
                    "left unchanged. Try again.",
                    severity="warning",
                )
            return False
        if blocking is not None:
            if intent.notify_toast:
                self.notify(
                    f"This item is marked {blocking}; leaving it as it is "
                    f"rather than overwriting that with {label}.",
                    severity="warning",
                )
            return False
        return True

    def _dispatch_item_status(self, item_id: Any, intent: "_ItemStatusIntent") -> None:
        """Queue `intent` as `item_id`'s desired write and ensure a drainer runs.

        TASK-1541 (Qodo redesign). Every one of the four item-status dispatch
        paths (Ingest, Ignore, the unread toggle, mark-read-on-open) calls
        this instead of directly starting its own worker -- see
        `_ItemStatusIntent`'s and `_ITEM_STATUS_DRAIN_GROUP_PREFIX`'s
        docstrings for why (cancellation-based "supersede" was unsound for a
        durable write, two independent ways).

        Overwriting `self._item_status_desired[item_id]` unconditionally is
        the coalescing: a second dispatch for the same item before the first
        has been drained simply replaces what the drainer will act on next --
        there is never more than one write queued per item. If a drainer is
        already running for this item (`item_id in self._item_status_
        draining`), nothing further happens here: the running drainer is
        NEVER cancelled and NEVER told to stop early, it just picks the new
        entry up itself the next time its loop checks the dict (see
        `_drain_item_status`). Only when no drainer is currently running is
        one started, in this item's OWN group -- so a burst of dispatches
        across MANY different items still gets one independent drainer each,
        never sharing a group and never able to interact.
        """
        self._item_status_desired[item_id] = intent
        if item_id in self._item_status_draining:
            return
        self._item_status_draining.add(item_id)
        self.run_worker(
            self._drain_item_status(item_id),
            group=f"{_ITEM_STATUS_DRAIN_GROUP_PREFIX}{item_id}",
            exclusive=False,
        )

    async def _drain_item_status(self, item_id: Any) -> None:
        """Per-item worker: pop the desired write, perform it, repeat.

        TASK-1541 (Qodo redesign). This is the ONLY worker body that ever
        writes an item's status now -- Ingest, Ignore, the unread toggle, and
        mark-read-on-open all reach it through `_dispatch_item_status`, one
        drainer per item id, never shared across items and never cancelled.

        The invariant that replaces cancellation-based "supersede": pop this
        item's desired entry, `await` `_update_item_status` (which itself
        `await`s the actual `asyncio.to_thread` write) to GENUINE completion
        -- success or a real exception, never a cancellation, since nothing
        here is ever cancelled -- THEN check the dict again before exiting.
        If a newer desired entry appeared while that write was in flight
        (another dispatch for this SAME item, which only ever overwrites the
        dict rather than starting a second drainer), the loop goes around
        again and writes THAT instead. Only when the dict holds nothing more
        for this item does the drainer exit and clear itself from
        `_item_status_draining`.

        Two consequences fall out of this shape directly:

        * At most one write is ever queued per item (the dict holds a single
          entry) plus at most one in flight -- the same bound the old
          cross-item/per-item `exclusive=True` groups gave, without the
          unsoundness: a fast `j`/`k` run or an Ingest-then-Ignore burst on
          one item still costs at most two writes, not one per keystroke,
          but BOTH writes always run to completion in dispatch order, so the
          LAST action dispatched is always the one the database (and the
          cache, if `patch_item` is set) settles on -- deterministically,
          not "whichever OS thread happened to finish last" (the old
          model's actual behaviour once the write got a genuine suspension
          point; see `_ITEM_STATUS_DRAIN_GROUP_PREFIX`'s docstring).
        * A gated intent (`intent.gate`, the unread toggle and
          mark-read-on-open) is re-checked against the backend
          (`_item_status_write_allowed`) immediately before ITS OWN write,
          not once at dispatch time -- so an Ingest/Ignore that lands on this
          same item while a gated entry is still queued is seen, even though
          neither of those two ever patches the cache this dict's staleness
          check would otherwise rely on.

        `try`/`finally` around the loop, not just around the body: whatever
        exits the loop (the normal empty-dict return, or -- not expected in
        practice, since `_update_item_status` itself catches `Exception` --
        anything else) must still clear this item from `_item_status_
        draining`, or the item would be permanently unable to dispatch a new
        write again for the rest of the screen's life.
        """
        try:
            while True:
                intent = self._item_status_desired.pop(item_id, None)
                if intent is None:
                    return
                if intent.gate:
                    allowed = await self._item_status_write_allowed(item_id, intent)
                    if not allowed:
                        continue
                await self._update_item_status(
                    item_id,
                    intent.status,
                    notify_toast=intent.notify_toast,
                    refresh=intent.refresh,
                    patch_item=intent.patch_item,
                )
        finally:
            self._item_status_draining.discard(item_id)

    async def _blocking_status_for(self, item_id: Any) -> str | None:
        """Which `_NON_READ_STATE_STATUSES` value the backend holds for this item.

        One authoritative single-item read
        (`WatchlistsBackendController.get_item_status`, down to
        `SubscriptionsDB.get_item_status`), not an inference from a listing.

        An earlier version asked `list_items` once per candidate status with
        `limit=500` and looked for the item in each result (PR #1091 review,
        F1). `LocalWatchlistsService.list_items` slices to the requested
        window, so an `ingested`/`ignored`/`error` item outside the first page
        was simply absent from the answer -- and absence from a truncated page
        is not proof of absence. The guard returned `None`, and `Mark unread`
        overwrote the ingest: exactly the data loss the guard exists to
        prevent, for any source with more than 500 items in a blocking
        status. Adding pagination would only have moved the boundary; reading
        the item's own row removes the boundary.

        Args:
            item_id: Normalized id of the item to check.

        Returns:
            The blocking status the backend holds for this item, or `None`
            when its status is a read/unread state (`new`/`reviewed`) that
            `Mark unread` may legitimately overwrite.

        Raises:
            Exception: Whatever the controller raises -- including `KeyError`
                for an item that is no longer there, and
                `NotImplementedError` for a backend with no single-item read.
                The caller treats an unanswerable question as a refusal, not
                as a green light.
        """
        status = await self._controller.get_item_status(
            runtime_backend=self.runtime_backend,
            item_id=item_id,
        )
        normalized = str(status or "").strip().lower()
        return normalized if normalized in _NON_READ_STATE_STATUSES else None

    @on(ItemsFilterChanged)
    def handle_items_filter_changed(self, event: ItemsFilterChanged) -> None:
        """Mirror the Items filter/search, and re-page when EITHER moves.

        Review wave, I2. The status is now part of the query
        (`_items_status_kwargs`), so changing it has to re-fetch or the pane is
        left filtering the previous status's page in memory -- which is exactly
        the "the filter can only narrow what was already fetched" defect this
        fix removes. TASK-3791 extends the same rule to the search box: the
        term is part of the query too (`_load_items` weaves it in), so a
        search reaches the whole corpus rather than the newest-50 page --
        debounced, since this message fires on every keystroke.

        The status branch re-fetches immediately; the search branch re-arms a
        0.3s timer. Both sides are compared through
        `_normalize_items_status_filter` (TASK-3072): a pre-reader `new`
        still sitting in the mirror IS the reader's `unread`, and must not
        trigger a spurious reload when the pane first posts it.
        """
        event.stop()
        if self.runtime_backend != "local":
            self._items_page_loading = False
            self._push_items_pager_state()
            return
        incoming = _normalize_items_status_filter(event.status_filter)
        forced = self._scope_forces_unread(self.tree_scope)
        status_changed = not forced and incoming != _normalize_items_status_filter(
            self._items_status_filter
        )
        query_changed = event.search_query != self._items_search_query
        if not forced:
            self._items_status_filter = incoming
        self._items_search_query = event.search_query
        if status_changed:
            self._supersede_items_query_intent()
            # Own group, as in `watch_tree_scope`: an exclusive reload in
            # the default group cancels unrelated in-flight workers.
            self.run_worker(
                self._replace_items_snapshot(reason="filter"),
                exclusive=True,
                group="wc_items",
            )
        elif query_changed:
            self._supersede_items_query_intent()
            # TASK-3791 plan task 3: a search edit re-fetches too, now that
            # the term is part of the query (`_load_items` weaves it in) --
            # debounced, because this message fires on every keystroke and a
            # query per character is exactly what the debounce timer shape
            # (`_request_tree_counts_refresh`) already exists to avoid.
            self._request_items_search_reload()

    #: Debounce for the search-driven items reload (TASK-3791): one corpus
    #: query per typing pause, not one per keystroke.
    _ITEMS_SEARCH_DEBOUNCE_SECONDS = 0.3

    def _request_items_search_reload(self) -> None:
        """Re-arm the single search-reload timer (the counts-refresh shape)."""
        timer = getattr(self, "_items_search_reload_timer", None)
        if timer is not None:
            timer.stop()
        self._items_search_reload_timer = self.set_timer(
            self._ITEMS_SEARCH_DEBOUNCE_SECONDS,
            lambda: self.run_worker(
                self._replace_items_snapshot(reason="search"),
                exclusive=True,
                group="wc_items",
            ),
        )

    @on(RefreshItemsRequested)
    def handle_refresh_items_requested(self, event: RefreshItemsRequested) -> None:
        event.stop()
        self._supersede_items_query_intent()
        # Own group, as in `watch_tree_scope`: an exclusive reload in the
        # default group cancels unrelated in-flight workers.
        self.run_worker(
            self._replace_items_snapshot(reason="refresh"),
            exclusive=True,
            group="wc_items",
        )

    @on(PreviousItemsPageRequested)
    def handle_previous_items_page_requested(
        self, event: PreviousItemsPageRequested
    ) -> None:
        event.stop()
        if self._items_page_loading or self._items_page_index == 0:
            return
        self.run_worker(
            self._present_cached_items_page(self._items_page_index - 1),
            exclusive=True,
            group="wc_items",
        )

    @on(NextItemsPageRequested)
    def handle_next_items_page_requested(
        self, event: NextItemsPageRequested
    ) -> None:
        event.stop()
        if self._items_page_loading or not self._items_has_next:
            return
        self.run_worker(
            self._load_next_items_page(),
            exclusive=True,
            group="wc_items",
        )

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
            if self._dom_is_live:
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
        self.run_worker(self._load_rules(), exclusive=True, group="wc_rules")

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
        self.run_worker(
            self._save_rule(event.payload),
            exclusive=True,
            group="wc_save_rule",
        )

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

    @on(SaveNoiseSelectorsRequested)
    def handle_save_noise_selectors_requested(
        self, event: SaveNoiseSelectorsRequested
    ) -> None:
        event.stop()
        if event.source_id is None:
            # Fix round 1 (Minor 4). A bare `return` here made Save a dead
            # button: the press produced no write, no error and no toast, the
            # exact pattern this stream keeps paying for (see the watchlist
            # -level actions in Task 5 fix round 2, disabled rather than left
            # silently inert). Reachable only from an entity with no `id`,
            # which is a state defect rather than anything the user did --
            # so it is logged as well as toasted.
            logger.warning(
                "Ignore-rule save requested for an entity carrying no id; "
                "nothing was written."
            )
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "Nothing to save: no source is selected.", severity="warning"
                )
            return
        self.run_worker(
            self._save_noise_selectors(event.source_id, event.text),
            exclusive=True,
            group="wc_noise_selectors",
        )

    async def _save_noise_selectors(self, source_id: Any, text: str) -> None:
        """Persist one source's `ignore_selectors` and patch, never recompose.

        TASK-1362 (spec §2). Deliberately does NOT call `_load_sources()`,
        `_refresh_overview_data()` or `_refresh_local_wc_snapshot()` the way
        `_create_source` does. `overview_data` was `reactive({}, recompose=
        True)` on this screen when that decision was taken, so touching it
        rebuilt every region through its factory and replaced the mounted
        panes wholesale -- proven live in Phase D Task 5 to detach the
        `ItemsPane`, reset the `DataTable` cursor and drop keyboard focus.
        TASK-2200 removed that recompose, but the conclusion is unchanged
        for a simpler reason: nothing the user can see is derived from a
        source's selectors -- not the Sources table's five columns, not the
        overview counts, not the staging line. So the only stale surface is
        the entity dict itself, which is patched in place -- the SAME object
        held by `selected_entity`, `selected_source` and `SourcesPane.sources`
        -- so a later read (including the Inspector rebuilt by an unrelated
        region toggle) already sees the new value with no rebuild forced here.

        Args:
            source_id: The source's watchlist item id, namespaced or bare.
            text: The newline-separated selector text to store.
        """
        notify = getattr(self.app_instance, "notify", None)
        try:
            await self._controller.update_source(
                runtime_backend=self.runtime_backend,
                item_id=source_id,
                payload={"ignore_selectors": text},
            )
        except Exception:
            logger.opt(exception=True).warning("Failed to save noise selectors.")
            if callable(notify):
                notify("Failed to save ignore rules.", severity="error")
            return
        self._patch_entity_ignore_selectors(source_id, text)
        if callable(notify):
            notify(
                NOISE_SELECTORS_SAVED_TOAST,
                severity="information",
            )

    def _patch_entity_ignore_selectors(self, source_id: Any, text: str) -> None:
        """Mirror a saved selector text into the in-memory source dicts.

        Matches `normalize_local_subscription_row`'s published shape (a list
        under `settings`, with the key absent when there is nothing stored),
        so a subsequent `InspectorPane._ignore_selectors_text` read reproduces
        exactly what the backend would return.
        """
        selectors = [line.strip() for line in text.split("\n") if line.strip()]
        seen: list[dict[str, Any]] = []
        for entity in (self.selected_entity, self.selected_source):
            if not isinstance(entity, dict) or any(entity is other for other in seen):
                continue
            seen.append(entity)
            if str(entity.get("id")) != str(source_id):
                continue
            settings = entity.get("settings")
            if not isinstance(settings, dict):
                settings = {}
                entity["settings"] = settings
            if selectors:
                settings["ignore_selectors"] = selectors
            else:
                settings.pop("ignore_selectors", None)

    @on(IngestRequested)
    def handle_ingest_requested(self, event: IngestRequested) -> None:
        event.stop()
        entity = event.entity
        if entity is None:
            return
        item_id = entity.get("id")
        if item_id is None:
            # Fix wave (F3, Minor): an id-less entity would otherwise derive
            # `_dispatch_item_status`'s per-item drain group as
            # f"{_ITEM_STATUS_DRAIN_GROUP_PREFIX}None", collapsing EVERY
            # id-less Ingest/Ignore into one shared drainer -- exactly the
            # cross-item interaction the per-item grouping exists to avoid.
            # Believed unreachable in practice: `normalize_watchlist_item`
            # unconditionally sets "id" via `build_watchlist_item_id(source,
            # "watchlist_item", row["id"])` for every item this screen's own
            # Items pane produces, and `row["id"]` is the table's `INTEGER
            # PRIMARY KEY`, never NULL -- so `event.entity` reaching here
            # without an "id" would mean something other than this screen's
            # own item pipeline constructed it. Refusing the dispatch (rather
            # than falling back to a unique per-dispatch suffix) also matches
            # `_mark_item_read_on_open`'s and `handle_unread_toggle_
            # requested`'s existing `item_id is None: return` guards.
            return
        self._dispatch_item_status(item_id, _ItemStatusIntent(status="ingested"))

    @on(IgnoreRequested)
    def handle_ignore_requested(self, event: IgnoreRequested) -> None:
        event.stop()
        entity = event.entity
        if entity is None:
            return
        item_id = entity.get("id")
        if item_id is None:
            # Fix wave (F3, Minor): see the identical guard and comment in
            # `handle_ingest_requested` just above -- same hazard, same
            # reachability analysis, same refusal.
            return
        self._dispatch_item_status(item_id, _ItemStatusIntent(status="ignored"))

    @on(ToggleBriefingQueueRequested)
    def handle_toggle_briefing_queue_requested(
        self, event: ToggleBriefingQueueRequested
    ) -> None:
        """Answer from memory, then dispatch the write to a worker.

        Fix round 1 (Important): `SubscriptionsDB.set_item_briefing_queued`
        is a transactional `UPDATE ... WHERE id = ?` with no busy timeout
        beyond SQLite's default, and this branch's own docstrings (Task 4's
        `_sweep_and_guard`) admit a second app instance against the same
        database file. Running that write on the UI thread meant a
        contended write could block the event loop for up to five seconds
        before raising. This handler now does only what the UI thread is
        entitled to do -- answer the no-selection/no-database cases from
        memory and dispatch -- and the write itself moves into
        `_toggle_briefing_queue`, off the UI thread, following Task 4's
        `handle_generate_briefing_requested` -> `_generate_briefing` shape.

        No `exclusive=True` and no per-item dedup: `set_item_briefing_queued`
        is a single-row idempotent `UPDATE`, so two overlapping writes to the
        SAME item are safe to interleave (last write wins, which is exactly
        what two rapid presses mean), and two presses on DIFFERENT items
        must not cancel each other -- Task 4's own lesson about
        `exclusive=True` manufacturing zombie state applies here too, one
        write at a time being the wrong shape for a control every row in the
        table can trigger independently. The shared `group="wl-queue-toggle"`
        exists only so these workers are nameable together (e.g. at
        shutdown), not to serialize them.
        """
        event.stop()
        if event.item_id is None:
            logger.warning(
                "Queue-for-briefing toggle requested for an entity carrying "
                "no item id; nothing was written."
            )
            self._notify_watchlists(
                "Nothing to queue: no item is selected.", severity="warning"
            )
            return
        db = self._briefings_db()
        if db is None:
            self._notify_watchlists(
                "Could not reach the local database, so nothing was queued.",
                severity="error",
            )
            return
        self.run_worker(
            self._toggle_briefing_queue(db, event.item_id, event.queued),
            group="wl-queue-toggle",
        )

    async def _toggle_briefing_queue(
        self, db: Any, item_id: Any, queued: bool
    ) -> None:
        """Worker body: write the flag off the UI thread, then patch+repaint.

        `asyncio.to_thread` is the load-bearing part -- `run_worker`
        alone only *schedules* a coroutine onto this same event loop; the
        controller's own `_maybe_await` (`watchlists_backend_controller.py`)
        has no `to_thread` either, so dispatch through a worker without it
        would still block the loop for the length of the `UPDATE`.
        `SubscriptionsDB` holds thread-local connections (see its
        `__init__`), so the worker thread gets its own, matching the idiom
        the rest of the UI already uses for off-thread DB writes.

        Honest failure, write-first-patch-after: on a DB error the flag is
        never patched and the indicator never repainted, so a failed write
        leaves the item exactly as it was, with an error toast reporting
        it. `self.is_attached` is checked before every UI mutation after the
        `await`, matching `_generate_briefing`'s own guard, since the screen
        may have been popped while the write was in flight. The log line
        names the exception TYPE only -- `logger.opt(exception=True)` would
        dump the failing frame's locals (including this item's
        title/excerpt) into a file sink running with `diagnose=True` (Task
        3's leak, one layer up).
        """
        try:
            await asyncio.to_thread(db.set_item_briefing_queued, item_id, queued)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            logger.warning(
                "Failed to set the briefing queue flag for item "
                f"{item_id}: {type(exc).__name__}"
            )
            if self.is_attached:
                self._notify_watchlists(
                    "Could not update the briefing queue flag. Nothing changed.",
                    severity="error",
                )
            return
        if not self.is_attached:
            return
        self._patch_item_queued_flag(item_id, queued)
        label = "queued for" if queued else "removed from"
        self._notify_watchlists(
            f"Item {label} the next briefing.", severity="information"
        )

    def _patch_item_queued_flag(self, raw_item_id: Any, queued: bool) -> None:
        """Mirror a saved queue flag into every in-memory dict, then repaint.

        Same shape as `_patch_entity_ignore_selectors`/
        `_repaint_item_status_cell`: patches every dict this screen holds
        that describes the same item, in place, so a later read (including
        the mounted Inspector, rebuilt for an unrelated reason) already sees
        the new value with no rebuild forced here -- then repaints the ONE
        Items-table cell that displays it and, if the Inspector is currently
        showing this same item, its queue button's label. Neither touches a
        `recompose=True` reactive (Phase D pattern): the dicts are mutated
        in place, never reassigned.

        `raw_item_id` is the DB row id (`entity["item_id"]`), not the
        namespaced `entity["id"]` the table row key and
        `update_item_queued_cell` use -- resolved below from whichever
        matching dict is found first, exactly as `_repaint_item_status_cell`
        already has to for the status column.
        """
        row_key: Any = None
        for item in self._loaded_items:
            if item.get("item_id") == raw_item_id:
                item["queued_for_briefing"] = queued
                if row_key is None:
                    row_key = item.get("id")
        for entity in (self.selected_entity, self._selected_content_item):
            if isinstance(entity, dict) and entity.get("item_id") == raw_item_id:
                entity["queued_for_briefing"] = queued
                if row_key is None:
                    row_key = entity.get("id")
        if row_key is not None:
            try:
                pane = self.query_one("#watchlists-items-pane", ArticleListPane)
                pane.update_item_queued_cell(row_key, queued)
            except NoMatches:
                pass
        entity = self.selected_entity
        if isinstance(entity, dict) and entity.get("item_id") == raw_item_id:
            try:
                inspector = self.query_one(
                    "#watchlists-entity-inspector", InspectorPane
                )
                button = inspector.query_one(
                    "#inspector-queue-briefing-button", Button
                )
            except NoMatches:
                return
            button.label = (
                InspectorPane._UNQUEUE_BRIEFING_LABEL
                if queued
                else InspectorPane._QUEUE_BRIEFING_LABEL
            )

    def _patch_committed_items_after_mutation(
        self, item_id: Any, **changes: Any
    ) -> None:
        """Patch one item across every committed Reader projection in place."""
        visited: set[int] = set()
        snapshot = self._items_snapshot
        candidates: list[dict[str, Any]] = []
        if snapshot is not None:
            candidates.extend(row for page in snapshot.pages for row in page)
        candidates.extend(self._loaded_items)
        for entity in (self._selected_content_item, self.selected_entity):
            if isinstance(entity, dict):
                candidates.append(entity)
        row_key: Any = None
        for item in candidates:
            identity = id(item)
            if identity in visited or not self._item_identity_matches(
                item, item_id
            ):
                continue
            visited.add(identity)
            item.update(changes)
            if row_key is None:
                row_key = item.get("id")
        if "status" in changes and row_key is not None:
            self._repaint_item_status_cell(row_key, str(changes["status"]))
        if "is_flagged" in changes and row_key is not None:
            try:
                pane = self.query_one("#watchlists-items-pane", ArticleListPane)
                pane.update_item_starred_cell(row_key, bool(changes["is_flagged"]))
            except NoMatches:
                pass
            try:
                star = self.query_one("#content-star-button", Button)
                star.label = "★ Starred" if changes["is_flagged"] else "☆ Star"
            except NoMatches:
                pass

    @staticmethod
    def _item_identity_matches(item: dict[str, Any], item_id: Any) -> bool:
        """Return whether a normalized or raw item identity matches a row."""
        target = str(item_id)
        return target in {str(item.get("id")), str(item.get("item_id"))}

    async def _update_item_status(
        self,
        item_id: Any,
        status: str,
        *,
        notify_toast: bool = True,
        refresh: bool = True,
        patch_item: dict[str, Any] | None = None,
    ) -> None:
        """Move one item to `status` through the shared item-status API.

        Called exactly once per popped `_ItemStatusIntent`, from inside
        `_drain_item_status`'s loop (TASK-1541, Qodo redesign) -- never
        directly by a dispatch handler any more. That loop always `await`s
        this to completion before looking at this item's desired-status dict
        entry again, so at most one call to this method is ever in flight for
        a given item at a time.

        `notify_toast` is False only for the Task 5 auto-mark-read-on-open
        path -- every other caller (Ingest/Ignore, the unread toggle) is a
        deliberate user action and keeps the toast. The failure toast is
        gated on `notify_toast` too (fix round 1, Minor): "Failed to mark
        item reviewed" for a write the user never asked for reads as an
        alarming report about nothing they did; the failure is still logged
        unconditionally via `logger.opt(exception=True).warning` just below,
        so it is not silent, just not toasted on the automatic path.

        `refresh=False` (fix round 1, CRITICAL) skips the reload of
        `ItemsPane.items` and `_refresh_overview_data()`. The latter sets
        `overview_data`, which was `reactive({}, recompose=True)` on the
        screen until TASK-2200, so calling it after EVERY item selection
        forced a full screen recompose -- proven live to detach the mounted
        `ItemsPane`, reset the `DataTable` cursor, and drop keyboard focus,
        so a second arrow key did nothing. Used only by the silent
        auto-mark-read-on-open path; every deliberate action
        (Ingest/Ignore, the unread toggle)
        keeps refreshing as before. When `refresh` is False and the write
        succeeds, `patch_item` -- the same dict object already held by
        `ItemsPane.items`/`_selected_content_item`/`ContentPane.item` -- is
        mutated in place instead, so a later status check already sees the
        new value without forcing a rebuild.

        TASK-1541: the write itself goes through `_update_item_status_
        off_loop`, not a direct `await self._controller.update_item_status(
        ...)`. See that method's docstring for why a plain `await` here still
        ran the transactional `SubscriptionsDB.mark_item_status` UPDATE on
        the event-loop thread despite this coroutine already being dispatched
        through `run_worker`.

        No `CancelledError` handling (Qodo redesign; an earlier fix wave had
        one here -- deleted). That handler assumed a cancellation always
        meant "the write is durable regardless, just patch the cache to
        match it", which a later re-review found false in a narrow but real
        window (the underlying thread can itself raise, or -- worse -- never
        run at all if cancelled before the executor picked it up; see
        `_ITEM_STATUS_DRAIN_GROUP_PREFIX`'s docstring). The redesign removes
        the premise instead of patching the handler: nothing that calls this
        method is ever cancelled any more (`_drain_item_status`'s worker is
        `exclusive=False` and is never explicitly cancelled), so there is no
        cancellation path here to handle -- if this coroutine is interrupted
        at all, it is an external teardown (e.g. the screen unmounting),
        exactly like every other worker on this screen, not a same-item or
        cross-item "supersede".
        """
        notify = getattr(self.app_instance, "notify", None)
        try:
            await self._update_item_status_off_loop(item_id=item_id, status=status)
            patch_id = patch_item.get("id") if patch_item is not None else item_id
            self._patch_committed_items_after_mutation(patch_id, status=status)
            if notify_toast:
                label = "unread" if status == "new" else status
                # `markup=False`: the body is app-authored today, but toasts
                # on this screen carry item- and feed-derived text elsewhere
                # and the convention here is to escape at the terminal step
                # rather than to audit which messages happen to be safe.
                self._notify_watchlists(
                    f"Item marked {label}.", severity="information", markup=False
                )
        except Exception:
            logger.opt(exception=True).warning(f"Failed to mark item {status}.")
            if notify_toast and callable(notify):
                notify(f"Failed to mark item {status}.", severity="error")
            return
        if refresh:
            self._refresh_overview_data()
            # TASK-2304 AC#1. Every status this path writes moves the item
            # into or out of the `new` bucket the rail counts, so the rail is
            # stale the instant this returns.
            #
            # Deliberately inside `if refresh`, which is a real trade-off and
            # not an oversight. The `refresh=False` callers are the silent
            # mark-read-on-open (`_mark_item_read_on_open`), which fires on
            # EVERY item selection including each `j`/`k` keystroke, and
            # `action_toggle_read_selected` (task-2513's `m` verb) -- neither
            # carries a reload of any kind, which is the whole reason the
            # flag exists (a full refresh per selection was proven live to
            # detach the mounted `ItemsPane` and drop keyboard focus). So the
            # rail's unread count does lag by however many items were opened
            # since the last deliberate action, and is corrected by the next
            # one, by a tab switch, or by any other `_load_tree_data` caller.
            # (`m` compensates directly: `action_toggle_read_selected` calls
            # `_request_tree_counts_refresh()` right after dispatching, so
            # only the open-on-selection path leaves the rail one out.)
            # Two SQLite queries and a rail rebuild per arrow key is the
            # wrong price for a number that is one out.
            self._load_tree_data()

    async def _update_item_status_off_loop(
        self, *, item_id: Any, status: str
    ) -> dict[str, Any]:
        """Drive the item-status write to completion off the UI thread.

        TASK-1541. `_update_item_status` used to `await
        self._controller.update_item_status(...)` directly, and every layer
        of that call chain -- `WatchlistsBackendController.update_item_status`
        -> `WatchlistScopeService.update_item` -> `LocalWatchlistsService.
        update_item` -> `SubscriptionsDB.mark_item_status` -- is an `async
        def` with no genuine `await` of its own. `_maybe_await` (the
        controller's own helper) only awaits a value that is ALREADY
        awaitable; it never puts a plain synchronous call on a thread. So
        awaiting that chain runs the whole thing, including the transactional
        `UPDATE`, synchronously to completion on whichever thread awaits the
        outermost coroutine -- and `run_worker` only *schedules* a coroutine
        back onto this SAME event loop, it does not move it to a thread
        (identical shape to `_toggle_briefing_queue`'s fix, whose docstring
        names this exact trap). A second app instance (or a background
        check) contending for the same row blocked the UI thread for the
        length of the lock wait, not just the write -- up to
        `Subscriptions_DB.BUSY_TIMEOUT_MS` (5 s; task-19562 pinned that
        value explicitly, having previously inherited it, and measured the
        wait: a 1.0 s lock held cost the second writer 1.07 s).

        Mirrors `library_screen.py`'s `_run_library_service_call(...,
        isolate_in_worker=True)`: `asyncio.to_thread` gives the worker thread
        no event loop of its own, so `asyncio.run` builds a throwaway one
        there to drive the controller coroutine to completion, rather than
        resuming it back on this loop. `runtime_backend` is read here, on
        the calling (event-loop) thread, and passed into the thread body as a
        plain string -- the worker body itself must never read a screen
        reactive directly.

        Drain invariant (TASK-1541, Qodo redesign; replaces an earlier,
        incorrect "last press wins is not guaranteed at the database" caveat
        that used to live here). This OS thread is genuinely not cancellable
        once started -- `asyncio.to_thread`'s underlying executor future can
        only be cancelled before it begins running -- which is exactly why an
        earlier design (`exclusive=True` "supersede" worker groups) could not
        safely guarantee write ORDER: a superseded write's thread and its
        replacement's thread were two independent, un-ordered writes to the
        same row, either able to commit last. `_drain_item_status` sidesteps
        that instead of tolerating it: it `await`s this method to genuine
        completion before ever popping this SAME item's next desired entry,
        so at most one call to this method is alive for a given item at any
        time -- there is no second thread for the SAME row to race against.
        Repeat Ingest/Ignore (or any other item-status action) on one item
        now has a real "last dispatched action wins" guarantee, at the
        database, not just at whatever the UI happens to repaint last.

        Returns:
            The controller's result dict, unused by the only current caller
            but kept so a future caller does not have to re-add it.
        """
        runtime_backend = self.runtime_backend

        def _invoke() -> dict[str, Any]:
            return asyncio.run(  # policy-exception: worker-thread loop
                self._controller.update_item_status(
                    runtime_backend=runtime_backend,
                    item_id=item_id,
                    status=status,
                )
            )

        return await asyncio.to_thread(_invoke)

    def _repaint_item_status_cell(self, item_id: Any, status: str) -> None:
        """Push a patched status into the mounted Items table's Status cell."""
        try:
            pane = self.query_one("#watchlists-items-pane", ArticleListPane)
        except NoMatches:
            return
        pane.update_item_status_cell(item_id, status)

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
            logger.opt(exception=True).warning("Failed to save alert rule.")
            if callable(notify):
                notify("Failed to save alert rule.", severity="error")
        self.run_worker(self._load_rules(), exclusive=True, group="wc_rules")
        self._refresh_overview_data()

    @on(DeleteRequested)
    def handle_delete_requested(self, event: DeleteRequested) -> None:
        event.stop()
        entity = event.entity
        if entity is None:
            return
        entity_type = InspectorPane._entity_type(entity)
        if entity_type == "notification":
            self.app_instance.notify(
                "Use Dismiss to remove a notification from the inbox.",
                severity="information",
            )
            return
        if entity_type == "item":
            # Review wave, Minor 2. `d` over an item used to open a dialog
            # saying "Delete <title>?" and then write `status="ignored"` --
            # never a delete. Before TASK-2301 the row vanished on the next
            # reload, so it read as one; now the row stays, so the gesture
            # looked like it had simply failed (and on an ALREADY-ignored row
            # it genuinely did nothing observable at all).
            #
            # Routed to the vocabulary that matches the write, through the
            # same `_dispatch_item_status` the Inspector's own Ignore button
            # uses. That is not only naming: the old `_delete_item` called
            # `self._controller.update_item_status(...)` DIRECTLY, bypassing
            # both TASK-1541's per-item drain and the terminal-status gate --
            # a second, unguarded writer of the one field this screen is
            # careful about. It is gone; this is now the only path.
            item_id = entity.get("id")
            if item_id is None:
                return
            self._dispatch_item_status(item_id, _ItemStatusIntent(status="ignored"))
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
            self.run_worker(
                self._delete_source(entity.get("id")),
                exclusive=True,
                group="wc_delete_source",
            )
        elif entity_type == "run":
            self.run_worker(
                self._delete_run(entity.get("id")),
                exclusive=True,
                group="wc_delete_run",
            )
        elif entity_type == "rule":
            self.run_worker(
                self._delete_rule(entity.get("id")),
                exclusive=True,
                group="wc_delete_rule",
            )
        # No `item` branch: items never reach this dialog any more -- see the
        # Minor 2 note in `handle_delete_requested`.

    async def _delete_source(self, source_id: Any) -> None:
        try:
            await self._controller.delete_source(
                runtime_backend=self.runtime_backend,
                item_id=source_id,
            )
            self.selected_entity = None
            self.selected_source = None
            # The mounted pane keeps its own copy of the selection, and
            # nothing rebuilds it from screen state any more (TASK-2200) --
            # `_load_sources` below refreshes the ROWS but only re-selects
            # when `selected_source` is set. Without this the pane's Preview
            # / Check now buttons stay armed on a source that is gone.
            self._reseed_live_detail_pane()
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Source deleted.", severity="information")
        except Exception:
            logger.opt(exception=True).warning("Failed to delete source.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to delete source.", severity="error")
        # Reload every view derived from the source list, not just the two
        # that happened to be here (TASK-1040). `_refresh_local_wc_snapshot`
        # feeds the staging line and `_refresh_overview_data` the cards, but
        # `#sources-table` and the rail's counts read their own queries — so
        # without these the table kept the previous list and the rail said
        # `All sources  0` while the centre said `Feeds in All sources (1)`
        # (then Feeds, now the header summary),
        # describing the same thing on one screen.
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()
        self.run_worker(self._load_sources(), exclusive=True, group="wc_sources")
        self._load_tree_data()

    async def _delete_run(self, run_id: Any) -> None:
        try:
            await self._controller.delete_run(
                runtime_backend=self.runtime_backend,
                run_id=run_id,
            )
            self.selected_entity = None
            self.selected_run = None
            # See `_delete_source`: the deleted run's own pane holds the
            # selection, and nothing rebuilds it from screen state any more
            # (TASK-2200).
            self._reseed_live_detail_pane()
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Run deleted.", severity="information")
        except Exception:
            logger.opt(exception=True).warning("Failed to delete run.")
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
            logger.opt(exception=True).warning("Failed to delete alert rule.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to delete alert rule.", severity="error")
        self.run_worker(self._load_rules(), exclusive=True, group="wc_rules")
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
        # TASK-3072 plan task 10: the reading-loop verbs join the help line.
        # Decision 031: advertise only implemented actions -- every verb
        # named here is bound above and covered by tests. TASK-3791 adds
        # the search and refresh-all verbs.
        self.app_instance.notify(
            "1=Read 2=Sources 3=Runs 4=Rules 5=Notifications 6=Artifacts "
            "7=Overview | n=new d=delete/ignore c=check p=preview ?=help | "
            "j/k=move space=next-unread m=read/unread s=star o=open "
            "a=mark-all-read u=undo /=search r=refresh-all | "
            "z=toggle focused side pane Z=Article Focus (Read only) "
            "[=Navigation ]=Inspector | Reader is permanent",
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
        """Delete the selected entity, or IGNORE it when it is an item.

        Round 2, O3: this docstring used to say "after confirmation" flatly,
        which stopped being true for one of the four kinds. The split lives in
        `handle_delete_requested`, which is the single place that knows what
        each kind's destructive verb actually is:

        * source / run / rule -- deleted, behind `ConfirmDeleteDialog`.
        * item -- ignored, unconfirmed, through the same dispatch the
          Inspector's `Ignore` button uses. Unconfirmed on purpose: it is the
          same write as that button, which has never had a dialog, and adding
          one only here would make the keyboard path stricter than the mouse
          path for an identical action.

        The method keeps its name because the binding, the help line and this
        action are one triple and the rename would touch every caller for no
        behavioural gain; the copy is what had to become honest.
        """
        entity = self.selected_entity
        if entity is None:
            self.app_instance.notify(
                "Nothing selected.",
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

    def action_next_item(self) -> None:
        """`j`: move the reader to the next item in the list (Task 6)."""
        self._navigate_item(1)

    def action_previous_item(self) -> None:
        """`k`: move the reader to the previous item in the list (Task 6)."""
        self._navigate_item(-1)

    def _navigate_item(self, delta: int) -> None:
        """Shared `j`/`k` implementation.

        Guarded against a focused `Input`, or a focused *editable*
        `TextArea` -- both already consume and stop a printable key
        themselves (`Input._on_key`, `TextArea._on_key` when
        `read_only` is False), so this guard is not what protects typing
        today (confirmed by mutation test:
        `test_typing_j_in_the_search_input_does_not_navigate` does not
        redden when this check is deleted, because that test drives a real
        keypress and `Input` already stopped it before this method could
        ever run). It is kept anyway because this repo already binds a bare
        letter with `priority=True` elsewhere
        (`SearchRAGWindow.BINDINGS`'s `Binding("f", "focus_search", ...,
        priority=True)`) -- `App.on_event` resolves priority bindings
        BEFORE forwarding the key to the focused widget at all, bypassing
        `Input`'s own consumption entirely. If `j`/`k` on this screen were
        ever changed to `priority=True` (a one-line, easy-to-miss edit,
        given the precedent), this guard would become the ONLY thing
        stopping a keystroke from hijacking a user's typing -- load-bearing
        overnight rather than merely defensive. Directly exercised by
        `test_navigate_item_is_a_noop_when_a_text_input_has_focus`, which
        calls `action_next_item()` directly, bypassing the key-event
        pipeline (and therefore `Input`'s own protection) entirely, and
        does go red without this check.

        A *read-only* `TextArea` is deliberately NOT guarded: read-only
        `TextArea._on_key` returns before calling `event.stop()`, so it
        does not consume a printable key at all -- and a read-only
        `TextArea` is exactly the shape a future reader body could take
        (today `ContentPane` renders into a `Static`, not a `TextArea`).
        Guarding it out unconditionally would block `j`/`k` precisely where
        navigating away from the currently-open item is the whole point.

        Scoped to the Items ("Read") tab, where `ContentPane` is actually
        mounted (Task 4 gates CONTENT to that tab) -- firing elsewhere would
        silently write a read-status change to the database, through
        `_mark_item_read_on_open` below, for an item the user cannot even
        see.

        Walks `ItemsPane.displayed_items()` -- the SAME filtered/searched
        sequence the table renders (Task 6 fix round 1, Important #1) --
        not the screen's unfiltered `_loaded_items`. Otherwise, with a
        search query or status filter active, `j`/`k` could open, and
        silently mark read, an item that is not on screen at all.

        Hands the chosen item to `ItemsPane.select_and_reveal` (Task 6 fix
        round 1, Important #2) rather than calling `handle_item_selected`
        directly: that keeps `ItemsPane.selected_item`, the table's cursor
        row, and its scroll position all pointing at the same item as the
        reader, through the exact same `selected_item` ->
        `watch_selected_item` -> `ItemSelected` -> `handle_item_selected`
        path a mouse click or an arrow-key highlight already uses -- so
        this still inherits the Task 5 fix for free
        (`_mark_item_read_on_open` dispatches an `_ItemStatusIntent(refresh=
        False, patch_item=item)`, patching the item dict in place
        instead of forcing the `overview_data` `reactive(recompose=True)`
        full-screen rebuild that once dropped focus and broke a second
        keypress).

        Boundaries do not raise: an out-of-range index is simply a no-op.
        """
        focused = self.focused
        if isinstance(focused, Input) or (
            isinstance(focused, TextArea) and not focused.read_only
        ):
            return
        if self.active_section != "items":
            return
        try:
            pane = self.query_one("#watchlists-items-pane", ArticleListPane)
        except NoMatches:
            return
        items = pane.displayed_items()
        if not items:
            return
        current = self._selected_content_item
        current_id = current.get("id") if current else None
        index: int | None = None
        if current_id is not None:
            for position, candidate in enumerate(items):
                if candidate.get("id") == current_id:
                    index = position
                    break
        if index is None:
            # "Nothing open, or the open item is not in the displayed list"
            # is its own case, not index -1 (whole-branch review, CRITICAL).
            # Falling through on -1 computed `new_index = -1 + delta`, so
            # `j` opened `items[0]` -- visibly BACKWARDS from wherever the
            # reader was -- and `k` computed -2 and silently no-opped, for
            # the rest of the session. Start from the near end of the list
            # instead, which is what "next"/"previous" mean when there is no
            # current position.
            pane.select_and_reveal(items[0] if delta > 0 else items[-1])
            return
        new_index = index + delta
        if new_index < 0 or new_index >= len(items):
            return
        pane.select_and_reveal(items[new_index])

    def _reader_verb_blocked(self) -> bool:
        """Whether a Read-tab verb key (`m`/`a`/`u`) must do nothing.

        The same two guards `_navigate_item` applies — see its docstring for
        the full rationale: typing in an `Input` or editable `TextArea` is
        typing, not a verb (defensive against a future `priority=True` edit);
        and read-state writes are scoped to the Read tab, where the affected
        items are actually visible (off-tab writes would silently rewrite
        rows the user cannot see).
        """
        focused = self.focused
        if isinstance(focused, Input) or (
            isinstance(focused, TextArea) and not focused.read_only
        ):
            return True
        return self.active_section != "items" or self.runtime_backend != "local"

    def action_toggle_read_selected(self) -> None:
        """`m`: flip the open item between new and reviewed (task-2513 Task 10).

        Only the read/unread pair is togglable: `ingested`/`ignored`/`error`
        record deliberate user actions and are never rewritten by a verb key
        (the same rule `_mark_item_read_on_open` and the unread toggle
        follow). Dispatches through `_dispatch_item_status`, so the gating,
        in-place patch and cell repaint come along unchanged.
        `refresh=False` + `patch_item=item` is mark-read-on-open's contract:
        the live dict is patched in place instead of a full reload, so
        `_selected_content_item` stays current and a second `m` flips the
        item back to unread rather than re-deriving from a stale status.
        """
        if self._reader_verb_blocked():
            return
        item = self._selected_content_item
        if item is None:
            return
        item_id = item.get("id")
        if item_id is None:
            return
        current = str(item.get("status") or "").strip().lower()
        if current == "new":
            target = "reviewed"
        elif current == "reviewed":
            target = "new"
        else:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Only read/unread items can be toggled.", severity="warning")
            return
        self._dispatch_item_status(
            item_id,
            _ItemStatusIntent(status=target, gate=True, refresh=False, patch_item=item),
        )
        self._request_tree_counts_refresh()

    def action_toggle_star_selected(self) -> None:
        """`s`: star or unstar the open item (TASK-3072 plan task 7).

        Resolves the item and gates exactly like `action_toggle_read_selected`
        (typing in an Input is typing, and the verb is scoped to the Read
        tab), then shares `_toggle_item_star` with the reader's Star button.
        """
        if self._reader_verb_blocked():
            return
        item = self._selected_content_item
        if item is None:
            return
        self._toggle_item_star(item)

    @on(StarToggleRequested)
    def handle_star_toggle_requested(self, event: StarToggleRequested) -> None:
        """The reader's Star button takes the same path as `s`."""
        event.stop()
        item = event.item or self._selected_content_item
        if item is None:
            return
        self._toggle_item_star(item)

    def _toggle_item_star(self, item: dict[str, Any]) -> None:
        """Flip one item's star: write, patch, repaint, badge refresh.

        The target reads from the live dict, which the worker patches after
        a successful write -- so a second toggle unstars instead of
        re-deriving from a stale flag (`patch_item`'s contract on the
        read/unread side, restated for a flag there is no gate on: starring
        is orthogonal to status, so an ingested item can be starred).
        """
        item_id = item.get("id")
        if item_id is None:
            return
        target = not bool(item.get("is_flagged"))
        self.run_worker(
            self._toggle_star_worker(item_id, item, target),
            exclusive=True,
            group="wl-star-toggle",
        )

    async def _toggle_star_worker(
        self, item_id: Any, item: dict[str, Any], target: bool
    ) -> None:
        """The write half of a star toggle, mirroring every other write
        handler on this screen (a worker, never a render-path query)."""
        notify = getattr(self.app_instance, "notify", None)
        try:
            await self._controller.set_item_flagged(
                runtime_backend=self.runtime_backend, item_id=item_id, flagged=target
            )
        except Exception:
            logger.opt(exception=True).debug("Failed to toggle an item's star.")
            if callable(notify):
                notify("Could not update the star.", severity="error")
            return
        self._patch_committed_items_after_mutation(item_id, is_flagged=target)
        self._request_tree_counts_refresh()

    def action_open_in_browser(self) -> None:
        """`o`: open the open item's URL in the system browser (TASK-3072 plan
        task 8). Gated and resolved exactly like `m`/`s`."""
        if self._reader_verb_blocked():
            return
        item = self._selected_content_item
        if item is None:
            return
        self._open_item_in_browser(item)

    def action_focus_items_search(self) -> None:
        """`/`: put the caret in the items search box (TASK-3791 plan task 3).

        Gated by `_reader_verb_blocked` like every other Read-tab verb:
        once any Input has focus, `/` is text, not a verb (and off the Read
        tab there is no search box to focus).
        """
        if self._reader_verb_blocked():
            return
        try:
            self.query_one("#items-search-input", Input).focus()
        except NoMatches:
            return

    #: One refresh-all batch at a time (TASK-3791): a second `r` while a
    #: batch is in flight is a no-op, not a double launch.
    _refresh_all_in_flight = False

    def action_refresh_all(self) -> None:
        """`r`: check every active source, then say so ONCE (TASK-3791 plan
        task 5). Same gating as the other Read-tab verbs."""
        if self._reader_verb_blocked():
            return
        if self._refresh_all_in_flight:
            return
        # Set SYNCHRONOUSLY, before the worker is scheduled (PR #1443
        # review): setting it inside the worker leaves a window where two
        # rapid `r` presses both pass the check and the second
        # `exclusive=True` scheduling would cancel the first batch.
        self._refresh_all_in_flight = True
        self.run_worker(
            self._refresh_all_worker(), exclusive=True, group="wl-refresh-all"
        )

    async def _refresh_all_worker(self) -> None:
        """The batch half of `r`: launch, aggregate, notify once, reconcile.

        Eligibility reads the normalized source dicts' `active` (already
        `is_active AND NOT paused` -- `normalize_local_subscription_row`),
        so a source auto-paused by repeated failures is skipped, not poked.
        The aggregate toast retains its historical all-sources unread delta.
        The Reader pill does not: the terminal tree reload reconciles it from
        the committed query and creation watermark, so reading an old row
        during this batch cannot hide a genuinely new id. The in-flight flag
        is set by the action before this worker is scheduled; the `finally`
        here is the one reset.
        """
        notify = getattr(self.app_instance, "notify", None)
        try:
            eligible = [
                source
                for source in self._loaded_sources
                if source.get("active") and source.get("source_id") is not None
            ]
            if not eligible:
                if callable(notify):
                    notify("Nothing to check: no active sources.")
                return
            before = self._tree_counts.get(ALL_SOURCES_BUCKET, {}).get("unread", 0)
            result = await self._controller.check_all(
                runtime_backend=self.runtime_backend,
                source_ids=[source["source_id"] for source in eligible],
            )
            try:
                await self._load_tree_data().wait()
            except Exception:
                logger.opt(exception=True).debug(
                    "Refresh-all: the terminal tree reload failed."
                )
            after = self._tree_counts.get(ALL_SOURCES_BUCKET, {}).get("unread", 0)
            delta = max(0, after - before)
            checked = int(result.get("checked", 0))
            failed = list(result.get("failed") or [])
            message = f"Checked {checked} sources — {delta} new items"
            if failed:
                message += f" ({len(failed)} failed)"
            if callable(notify):
                notify(message)
        finally:
            self._refresh_all_in_flight = False

    @on(OpenInBrowserRequested)
    def handle_open_in_browser_requested(self, event: OpenInBrowserRequested) -> None:
        """The reader's Open button takes the same path as `o`."""
        event.stop()
        item = event.item or self._selected_content_item
        if item is None:
            return
        self._open_item_in_browser(item)

    def _open_item_in_browser(self, item: dict[str, Any]) -> None:
        """Validate on the UI thread, then dispatch the OS call to a worker.

        A feed item's `url` is a REMOTE-derived string and `webbrowser.open`
        hands it to the OS, so validation runs at this boundary through
        `input_validation.validate_url` -- the centralized validator: only
        well-formed http/https URLs with a valid host pass, and
        whitespace/backslash/credential/malformed-host shapes are refused
        with a notification, never passed on.
        """
        notify = getattr(self.app_instance, "notify", None)
        # Control characters first (a feed URL is remote-derived text, and
        # the OS open is a shell-shaped sink on some platforms), then the
        # centralized validator decides what is openable.
        url = strip_control_characters(str(item.get("url") or "")).strip()
        if not url or not validate_url(url):
            if callable(notify):
                notify(
                    "This item has no web URL to open (http/https only).",
                    severity="warning",
                )
            return
        self._open_item_in_browser_worker(url)

    @work(thread=True, group="wl-open-browser")
    def _open_item_in_browser_worker(self, url: str) -> None:
        """Invoke the blocking OS browser integration off the UI thread."""
        try:
            opened = webbrowser.open(url)
        except Exception:
            logger.opt(exception=True).warning(
                "The system browser failed while opening a Watchlists item."
            )
            opened = False
        if not opened:
            self.app.call_from_thread(
                self._notify_watchlists,
                "Could not open this item in the system browser.",
                "error",
                markup=False,
            )

    @on(NextUnreadRequested)
    def handle_next_unread_requested(self, event: NextUnreadRequested) -> None:
        """`space` (the ItemsPane binding): open the next unread item.

        No Input/rail focus guards are needed here, unlike `_navigate_item`:
        `Input` consumes printable keys before the pane binding can fire, and
        rail widgets are not ItemsPane descendants, so this message can only
        originate from the items region. Walks the same displayed sequence
        `j`/`k` use, and hands the choice to `select_and_reveal` for the
        same reason (selection, cursor, scroll and reader stay in step).

        Args:
            event: The pane-posted request; stopped here so it cannot
                bubble further.
        """
        event.stop()
        if self.active_section != "items":
            return
        try:
            pane = self.query_one("#watchlists-items-pane", ArticleListPane)
        except NoMatches:
            return
        items = pane.displayed_items()
        if not items:
            return
        current = self._selected_content_item
        current_id = current.get("id") if current else None
        start = -1
        if current_id is not None:
            for position, candidate in enumerate(items):
                if candidate.get("id") == current_id:
                    start = position
                    break
        for candidate in items[start + 1:]:
            if str(candidate.get("status") or "").strip().lower() == "new":
                pane.select_and_reveal(candidate)
                return
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify("All caught up.", severity="information")

    def action_mark_all_read(self) -> None:
        """`a`: catch the current scope up. Undoable with `u` (task-2513 Task 10)."""
        if self._reader_verb_blocked():
            return
        self.run_worker(
            self._mark_all_read_worker(), exclusive=True, group="wl-mark-all-read"
        )

    async def _mark_all_read_worker(self) -> None:
        """The write half of `a`: one scoped bulk UPDATE, then a repaint.

        Runs in a worker (DB write plus a badge refresh), mirroring every
        other write handler on this screen. The in-place patch follows the
        same contract as `_mark_item_read_on_open`'s `patch_item`: mutate
        the cached dicts, repaint the visible cells, never recompose the
        live table.
        """
        notify = getattr(self.app_instance, "notify", None)
        ids = await self._controller.mark_all_read(
            runtime_backend=self.runtime_backend, **self._items_scope_query()
        )
        if not ids:
            if callable(notify):
                notify("Nothing unread in this scope.")
            return
        self._last_mark_all_read_batch = [int(i) for i in ids]
        for item_id in self._last_mark_all_read_batch:
            self._patch_committed_items_after_mutation(
                item_id, status="reviewed"
            )
        self._repaint_visible_status_cells()
        committed = self._items_snapshot
        refreshed = await self._replace_items_snapshot(reason="refresh")
        if not refreshed and committed is not None and self._items_snapshot is committed:
            closed = committed.close_to_cached_pages()
            self._items_snapshot = closed
            self._items_snapshot_count = closed.snapshot_count
            self._items_has_next = closed.has_next(self._items_page_index)
            self._push_items_pager_state()
        self._request_tree_counts_refresh()
        if callable(notify):
            notify(f"Marked {len(ids)} read — press u to undo.")

    def action_undo_mark_all_read(self) -> None:
        """`u`: restore the most recent mark-all-read batch (task-2513 Task 10)."""
        if self._reader_verb_blocked():
            return
        if not self._last_mark_all_read_batch:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Nothing to undo.")
            return
        self.run_worker(
            self._undo_mark_all_read_worker(), exclusive=True, group="wl-mark-all-read"
        )

    async def _undo_mark_all_read_worker(self) -> None:
        """The write half of `u`: put the batch back to new — except rows
        the user has since moved on (ingested/ignored), which
        `restore_items_new`'s `status = 'reviewed'` guard leaves alone."""
        notify = getattr(self.app_instance, "notify", None)
        batch = list(self._last_mark_all_read_batch)
        try:
            restored = await self._controller.restore_items_new(
                runtime_backend=self.runtime_backend, item_ids=batch
            )
        except Exception:
            # The batch is the user's ONLY undo handle: a transient DB
            # failure must not consume it. Keep it so `u` can be retried
            # (Qodo review, PR #1383).
            logger.opt(exception=True).warning(
                "Undo mark-all-read failed; batch kept for retry."
            )
            if callable(notify):
                notify("Undo failed — press u to retry.", severity="error")
            return
        self._last_mark_all_read_batch = []
        for item_id in batch:
            snapshot = self._items_snapshot
            if snapshot is None or not any(
                self._item_identity_matches(item, item_id)
                and item.get("status") == "reviewed"
                for page in snapshot.pages
                for item in page
            ):
                continue
            self._patch_committed_items_after_mutation(item_id, status="new")
        self._repaint_visible_status_cells()
        self._request_tree_counts_refresh()
        if callable(notify):
            notify(f"Restored {restored} to unread.")

    def _repaint_visible_status_cells(self) -> None:
        """Repaint the Status column from the cached item dicts, in place.

        Bulk verbs (`a`/`u`) patch `_loaded_items` without a recompose for
        the same reason mark-read-on-open does (a recompose destroys the
        live table); the visible cells then need the same single-cell
        repaint path `_dispatch_item_status` uses per item. Rows not
        currently rendered are skipped inside `update_item_status_cell`.
        """
        try:
            pane = self.query_one("#watchlists-items-pane", ArticleListPane)
        except NoMatches:
            return
        for item in pane.displayed_items():
            if item.get("id") is None or item.get("status") is None:
                continue
            pane.update_item_status_cell(item["id"], str(item["status"]))
