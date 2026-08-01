"""Watchlists destination shell.

The route, class name, and stable widget selectors retain the historical
``watchlists_collections``/``wc`` identifiers so older tests, shortcuts, and
handoffs keep working while Collections moves under Library.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
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
from textual.widgets import Button, Input, Select, Static, TextArea

from ...Constants import (
    WATCHLISTS_NAV_CONTEXT_BACKEND,
    WATCHLISTS_NAV_CONTEXT_RUN_ID,
    WATCHLISTS_NAV_CONTEXT_SECTION,
    WATCHLISTS_SECTION_RUNS,
)
from ...runtime_policy.types import PolicyDeniedError
from ...Subscriptions.briefing_audio import (
    AudioGenerationError,
    fail_interrupted_audio,
    generate_script_audio,
)
from ...Subscriptions.briefing_cast import (
    ScriptCastError,
    fail_interrupted_scripts,
    generate_script,
)
from ...Subscriptions.briefing_selection import MODE_AUTO_FEATURED, VALID_MODES
from ...Subscriptions.briefing_service import (
    STATUS_GENERATING,
    extract_citation_ids,
    fail_interrupted_briefings,
    generate_briefing,
)
from ...Subscriptions.watchlist_bundle_service import WatchlistBundleService
from ...Subscriptions.watchlist_normalizers import normalize_watchlist_item
from ...TTS.audio_player import play_audio_file
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...Widgets.confirmation_dialog import ConfirmationDialog
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
    SaveNoiseSelectorsRequested,
    PreviewRequested,
    StageInConsoleRequested,
    ToggleBriefingQueueRequested,
)
from ..Watchlists_Modules.artifacts_pane import (
    ArtifactsPane,
    BriefingDefaultPresetChanged,
    BriefingModeChanged,
    BriefingSelected,
    CastScriptRequested,
    CitationActivated,
    GenerateBriefingRequested,
    ManagePresetsRequested,
    PlayAudioRequested,
    RefreshBriefingsRequested,
    ScriptSelected,
    StopAudioRequested,
    SynthesizeAudioRequested,
    audio_file_path_is_safe,
)
from ..Watchlists_Modules.briefing_preset_modal import BriefingPresetModal
from ..Watchlists_Modules.content_pane import ContentPane, UnreadToggleRequested
from ..Watchlists_Modules.items_pane import (
    ItemSelected,
    ItemsFilterChanged,
    ItemsPane,
    RefreshItemsRequested,
)
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
    WatchlistSourcePickerDialog,
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
from ..Watchlists_Modules.watchlist_tree import (
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
from ..Watchlists_Modules.watchlists_workbench import RegionToggled, WatchlistsWorkbench
from .destination_recovery import DestinationRecoveryState, policy_denied_recovery_state


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

#: Worker group for the two item read/unread status writes. They must
#: supersede each other (a fast `j` run should not queue up one write per key)
#: but must NOT supersede unrelated work -- see the note at
#: `_mark_item_read_on_open`'s `run_worker` call.
_ITEM_STATUS_WORKER_GROUP = "wl-item-status"

#: Item statuses the reader's "Mark unread" button must never overwrite: they
#: are not read/unread states at all, and `new` would destroy the record.
#: A frozenset, since `_blocking_status_for` now asks the backend for the
#: item's one status and only has to decide whether it is in this set.
_NON_READ_STATE_STATUSES: frozenset[str] = frozenset({"ingested", "ignored", "error"})


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
        ("1", "switch_section('overview')", "Overview"),
        ("2", "switch_section('sources')", "Sources"),
        ("3", "switch_section('items')", "Items"),
        ("4", "switch_section('runs')", "Runs"),
        ("5", "switch_section('rules')", "Rules"),
        ("6", "switch_section('notifications')", "Notifications"),
        ("7", "switch_section('artifacts')", "Artifacts"),
        ("question", "show_help", "Help"),
        ("n", "new_source", "New source"),
        ("d", "delete_selected", "Delete"),
        ("c", "check_now_selected", "Check now"),
        ("p", "preview_selected", "Preview"),
        ("j", "next_item", "Next item"),
        ("k", "previous_item", "Previous item"),
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
    # Through Phase C, CONTENT held only a placeholder stub and started
    # collapsed to avoid spending screen space on it. Phase D wires a real
    # reader (`ContentPane`) into CONTENT, so it now starts expanded like
    # every other region. `on_mount` overlays whatever is actually persisted
    # (see `region_layout_store.load_region_layout`) on top of this default —
    # including a one-time migration that drops any CONTENT collapse a user
    # saved before this change, since that could only be a leftover of the
    # old stub-era default, never a deliberate choice about the real reader.
    region_layout = reactive(RegionLayout())
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
        # The item currently open in the CONTENT reader (Task 4). Held here
        # for the identical reason as `_loaded_items` above: `_build_content_pane`
        # is a factory the workbench calls on every region rebuild, and a
        # freshly built `ContentPane`'s `item` reactive would otherwise start
        # back at `None` on every collapse/solo/rail toggle, clearing the
        # reader out from under a user who hadn't touched Items at all.
        self._selected_content_item: dict[str, Any] | None = None
        # Left-rail tree inputs (Task 4): loaded together by `_load_tree_data`
        # in exactly two queries (`list_watchlists` + `get_watchlist_item_counts`),
        # never one per node -- see that method's docstring.
        self._tree_watchlists: list[dict[str, Any]] = []
        self._tree_counts: dict[int, dict[str, int]] = {}
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
        self._tree_expanded: frozenset[int] = frozenset()
        self._tree_active_tag: str | None = None
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
        # `region_layout` is `recompose=True`, so any collapse/solo/rail
        # toggle rebuilds the whole workbench, constructing a brand new
        # SourcesPane. Without holding the draft here — the same way
        # selected_source/selected_run/active_section already survive pane
        # rebuilds — a half-typed create form would be silently destroyed by
        # a keybinding that has nothing to do with Sources.
        self._source_create_form_open = False
        self._source_create_draft: dict[str, str] = {"name": "", "url": "", "tags": ""}
        # The create form's noise-selector text, mirrored for the same reason
        # as the three fields above (TASK-1362). Held separately, and `None`
        # rather than `""` when untouched, because its empty state is not its
        # default: `SourcesPane` prefills it with the shipped selector set, and
        # `""` is a user deliberately clearing it. Seeding `""` back over a
        # fresh pane would silently turn "watch everything" into the default,
        # and seeding the default over a cleared field would be the reverse.
        self._source_create_draft_selectors: str | None = None
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

        Notifies on failure (task-876), matching every sibling loader on
        this screen (`_load_sources`, `_load_runs`, `_load_notifications`,
        ...): without this, a real database failure rendered identically to
        "you have zero watchlists" -- two empty roots and no message, since
        the tree is its own only error surface.
        """
        notify = getattr(self.app_instance, "notify", None)
        try:
            service = self._watchlist_bundle_service()
            self._tree_watchlists = service.list_watchlists()
            self._tree_counts = service.get_watchlist_item_counts()
        except Exception:
            logger.opt(exception=True).debug("Failed to load watchlists tree data.")
            self._tree_watchlists, self._tree_counts = [], {}
            if callable(notify):
                notify("Failed to load watchlists.", severity="error")
        # Re-resolve the Inspector's breadcrumb against what was just loaded
        # (task-895). `_resolve_breadcrumb_labels` reads `_tree_watchlists`,
        # and until this task nothing could change that list while a scope
        # was in view, so resolving once in `_apply_tree_scope` was enough.
        # The write verbs break that: creating a watchlist scopes to an id
        # that is not in the list yet (the crumb would read "Watchlist 3"),
        # and renaming one leaves the crumb on the old name until the user
        # navigates away and back. The `refresh(recompose=True)` below
        # rebuilds the Inspector, which seeds itself from this value.
        self._breadcrumb_labels = self._resolve_breadcrumb_labels(self.selected_scope)
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

        Seeds `expanded`/`active_tag` from screen state (whole-branch review,
        Finding 2) the same way `_build_detail_pane` seeds the panes' rows
        and `_build_inspector_pane` seeds scope and breadcrumbs -- otherwise
        the rail collapses on every section switch. Seeds `active_scope`
        from `tree_scope` for the same reason (task-876): a section switch
        or rail toggle rebuilds a brand new `WatchlistTree`, and without
        this the selection highlight would reset to nothing every time,
        even though the scope itself survived on the screen.
        """
        return WatchlistTree(
            watchlists=self._tree_watchlists,
            counts=self._tree_counts,
            source_rows_loader=self._load_source_rows_for_tree,
            expanded=self._tree_expanded,
            active_tag=self._tree_active_tag,
            active_scope=self.tree_scope,
            write_disabled_reason=self._tree_write_disabled_reason(),
            id="wl-tree",
        )

    def _tree_write_disabled_reason(self) -> str | None:
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

        Returns:
            The reason string, used verbatim as both the disabled buttons'
            tooltip and the visible note beneath them, or `None` when writes
            are available.
        """
        if self.runtime_backend == "server":
            return WC_SERVER_WRITE_RECOVERY.disabled_tooltip
        if self._watchlist_bundle_service() is None:
            return WC_SERVICE_UNAVAILABLE_COPY
        return None

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
        now that the navigator is retired — `Region.LEFT_RAIL` hosts the
        watchlist tree (`_build_tree_pane`), and this is the strip's
        permanent home per the design (a one-row strip at the top of the
        centre).

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
            # TASK-998: lets the first-run panel distinguish "no watchlists at
            # all" from "a watchlist with no sources in it" -- `overview_data`
            # counts sources, items and runs, never watchlists.
            overview.watchlist_count = len(self._tree_watchlists)
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
            if self._source_create_draft_selectors is not None:
                sources_pane.create_draft_ignore_selectors = (
                    self._source_create_draft_selectors
                )
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
            # Seed the filter, the search box and the selection too
            # (whole-branch review, Important) -- the sibling Sources/Runs/
            # Notifications panes above and below already re-seed their
            # selection, and this one seeded only `.items`, so every rebuild
            # silently reset the user's filtered view to "all items, nothing
            # selected". See `_items_status_filter` in `__init__`.
            items_pane.status_filter = self._items_status_filter
            items_pane.search_query = self._items_search_query
            items_pane.selected_item = self._selected_content_item
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
        elif self.active_section == "artifacts":
            # Seeded from screen state for the same reason every sibling
            # above is -- this is a factory the workbench calls on every
            # region rebuild, so a fresh pane's reactives start at their
            # class defaults.
            artifacts_pane = ArtifactsPane(id="watchlists-artifacts-pane")
            artifacts_pane.briefings = self._loaded_briefings
            artifacts_pane.selected_briefing = self._selected_briefing
            artifacts_pane.scope_label = self._briefing_scope_label()
            artifacts_pane.can_generate = self._can_generate_briefing()
            artifacts_pane.selection_mode = self._briefing_selection_mode
            artifacts_pane.presets = self._loaded_briefing_presets
            artifacts_pane.default_preset_id = self._briefing_default_preset_id
            artifacts_pane.scripts = self._loaded_scripts
            artifacts_pane.selected_script = self._selected_script
            artifacts_pane.script_audio = self._loaded_script_audio
            artifacts_pane.scripts_with_audio = self._scripts_with_audio
            artifacts_pane.citations = self._loaded_citations
            children.append(artifacts_pane)
        return Vertical(
            *children,
            id="watchlists-detail-pane",
            classes="destination-workbench-pane",
        )

    def _build_content_pane(self) -> ContentPane:
        """Build the CONTENT-region content: the reader for the last
        selected item (Task 4).

        Called fresh on every region rebuild, like every other region
        builder here -- see the factory note on `WatchlistsWorkbench.__init__`.
        Seeded from `_selected_content_item` (Finding pattern established by
        `_build_inspector_pane`'s `selected_entity` seeding above): without
        this, a collapse/solo/rail toggle would construct a brand new
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
        pane = ContentPane(id="watchlists-content-pane")
        pane.item = self._selected_content_item
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
        # TASK-998, widened by TASK-1020: same seeding rationale as the three
        # lines above -- and the Inspector cannot work this out for itself,
        # since it is handed a selection rather than the data behind it. The
        # value is the same one the Overview region keys off, so the rail's
        # first-run text and the region's can never disagree.
        inspector.profile_state = self._watchlists_profile_state()
        children.append(inspector)
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

    def _local_only_section(self) -> dict[str, str] | None:
        """Header copy for the active section if it has no server half."""
        return self._LOCAL_ONLY_SECTIONS.get(self.active_section)

    def _backend_label_text(self) -> str:
        """What the header bar says about where this section's rows live."""
        local_only = self._local_only_section()
        if local_only is not None:
            return local_only["label"]
        return f"Backend: {self.runtime_backend}"

    def compose_content(self) -> ComposeResult:
        latest_console_item = self._console_handoff.resolve_latest_follow_item()
        with Vertical(id="watchlists-collections-shell"):
            yield Static(
                "Watchlists | Monitored sources, runs, alerts, recovery | Mixed | Local/Server",
                id="watchlists-collections-title",
                classes="ds-destination-header",
            )
            with Horizontal(id="watchlists-header-bar", classes="destination-filter-strip"):
                # TASK-995: `compact=True` for the same reason as the
                # Sources/Items toolbars -- `.destination-filter-strip` is
                # `height: 1` and a bordered Select is three rows, so this
                # backend picker was painting its top border and nothing
                # else. See `sources_pane.compose()`.
                yield Select(
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
                yield Static(
                    self._backend_label_text(),
                    id="watchlists-backend-label",
                )
            attach_disabled, attach_tooltip = self._wc_attach_state()
            yield WatchlistsWorkbench(
                self._visible_region_layout(),
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
                    Region.CONTENT: self._build_content_pane,
                    Region.RIGHT_RAIL: lambda: self._build_inspector_pane(
                        latest_console_item, attach_disabled, attach_tooltip
                    ),
                },
                id="wl-workbench",
            )

    def _visible_region_layout(self) -> RegionLayout:
        """The layout actually rendered — `region_layout` with CONTENT
        gated to the Read tab (fix round 1, Task 4).

        Per the approved design spec (`### Tabs`): "Only Read uses the
        three-pane split. Sources, Runs, Rules, and Artifacts take the full
        centre width — they have no collection→feed→item relationship."
        `active_section == "items"` is this implementation's Read tab (the
        spec's five sections don't literally match today's six — Overview
        and Notifications aren't in the spec's list either — but Items is
        unambiguously the one with an items-to-read relationship, and the
        only section `ContentPane` is ever fed from; see
        `handle_item_selected`). On every OTHER section, CONTENT force-
        collapses regardless of what the user has expanded/collapsed it to,
        because the reader has nothing to show there and, before this fix,
        its mere presence (even idle) taxed the Sources create-form's
        already-zero-slack layout at 160x42.

        This is a DERIVED view layered on top of `self.region_layout`, the
        same shape `RegionLayout.solo` already establishes for its own
        collapsed-view-vs-pre-solo-baseline split: the override must never
        reach `_schedule_layout_persist` (only `_apply_layout`'s own
        `layout` argument — the real, un-derived preference — does), or a
        user's real "CONTENT expanded" choice would be silently overwritten
        on disk just because they happened to be looking at Sources when
        some unrelated toggle fired a save.

        A CONTENT solo is derived from the PRE-SOLO baseline, not from the
        solo view (PR #1091 review, F2). Soloing CONTENT sets `collapsed` to
        `{FEEDS, ITEMS}`; adding CONTENT to that off the Read tab collapsed
        all three centre regions at once, and the workbench mounted three
        header buttons with no expanded centre at all -- recoverable only by
        clicking a chevron the user has no reason to suspect. Deriving from
        `collapsed_for_persistence()` (which is exactly "the collapsed set
        before the solo") with solo cleared for the rendered view leaves
        whatever the user had expanded before soloing, and leaves
        `self.region_layout` itself untouched -- so returning to Read
        restores the solo they set.

        `FEEDS` has the identical spec violation -- it is unconditionally
        built by `_build_list_pane` regardless of `active_section`, so it
        also occupies space on every tab the spec says should be full-width.
        That predates this change (Phase C) and is NOT fixed here (TASK-1344
        AC#1); scoping this fix to CONTENT only, since that is what Task 4
        introduced.

        Returns:
            The layout to render: `region_layout` verbatim on the Read tab,
            otherwise a derived copy with CONTENT collapsed -- rebased onto
            the pre-solo baseline when CONTENT is the soloed region.
        """
        if self.active_section == "items":
            return self.region_layout
        if self.region_layout.solo_region is Region.CONTENT:
            # Solo is dropped from the DERIVED view only: `self.region_layout`
            # keeps `solo_region`, so the user's solo comes back on Read.
            return RegionLayout(
                collapsed=frozenset(
                    self.region_layout.collapsed_for_persistence() | {Region.CONTENT}
                )
            )
        return replace(
            self.region_layout,
            collapsed=frozenset(self.region_layout.collapsed | {Region.CONTENT}),
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
            #
            # Pushes the VISIBLE (tab-gated) layout, not the raw `layout`
            # argument -- see `_visible_region_layout`. Persistence just
            # below still persists the real, un-derived `layout`.
            self.query_one(WatchlistsWorkbench).region_layout = self._visible_region_layout()
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

    def _content_toggle_is_blocked(self, region: Region) -> bool:
        """Whether a CONTENT layout change must be refused right now.

        Whole-branch review (Important): off the Read (Items) tab,
        `_visible_region_layout` force-collapses CONTENT, which renders a
        real, focusable `▸ Content` header button. Clicking it -- or pressing
        `z` with it focused -- ran the toggle against the REAL
        `region_layout`, not the derived view the user was actually looking
        at. So the click did nothing visible, silently flipped the user's
        genuine preference to collapsed, and `_schedule_layout_persist` wrote
        `"content"` into `[watchlists].collapsed_regions` on disk. With the
        Phase D migration marker already set, that is then honoured forever
        -- permanently recreating the exact broken state the migration exists
        to repair, from a control that appeared to be inert.

        The header stays focusable (a collapsed region must be), so refusing
        the toggle here, with an explanation, is the fix rather than removing
        the affordance.

        Also gates SOLO (PR #1091 review, F2 / TASK-1344 AC#2). `Z` on the
        same focused header collapsed FEEDS and ITEMS around a region the
        user cannot see on this tab, so the centre went empty -- the same
        class of harm as the chevron, through the one route that was still
        open.

        Args:
            region: The region the user's gesture targets.

        Returns:
            `True` when the gesture must be refused (and the user has been
            told why), `False` when it may proceed.
        """
        if region is not Region.CONTENT or self.active_section == "items":
            return False
        self.notify(
            "The reader is only shown on the Read tab. Switch to Read to "
            "change its layout."
        )
        return True

    def action_toggle_region(self) -> None:
        """Collapse or expand whichever region currently has focus."""
        region = self.focused_region
        if self._content_toggle_is_blocked(region):
            return
        self._apply_layout(self.region_layout.toggle(region))

    def action_solo_region(self) -> None:
        """Isolate the focused centre pane; press again to restore.

        Refused for CONTENT off the Read tab, exactly as the chevron and `z`
        already are -- see `_content_toggle_is_blocked`.
        """
        if self.focused_region not in CENTRE_REGIONS:
            self.notify("Solo applies to the Feeds, Items, or Content panes.")
            return
        if self._content_toggle_is_blocked(self.focused_region):
            return
        self._apply_layout(self.region_layout.solo(self.focused_region))

    def action_toggle_left_rail(self) -> None:
        self._apply_layout(self.region_layout.toggle(Region.LEFT_RAIL))

    def action_toggle_right_rail(self) -> None:
        self._apply_layout(self.region_layout.toggle(Region.RIGHT_RAIL))

    @on(RegionToggled)
    def _on_region_toggled(self, event: RegionToggled) -> None:
        event.stop()
        if self._content_toggle_is_blocked(event.region):
            return
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
        self._clear_pane_selections()
        self.selected_scope = scope
        self.tree_scope = scope

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

        See `_tree_expanded` in `__init__` for why this cannot live on the
        tree widget, and `_build_tree_pane` for where it is seeded back.
        """
        event.stop()
        self._tree_expanded = event.expanded

    @on(TreeTagFilterChanged)
    def handle_tree_tag_filter_changed(self, event: TreeTagFilterChanged) -> None:
        """Mirror the rail's tag filter onto the screen (Finding 2)."""
        event.stop()
        self._tree_active_tag = event.tag

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
        self._apply_tree_scope(
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
        self._apply_tree_scope(TreeScope(kind="unassigned"))
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
        self._apply_tree_scope(TreeScope(kind="watchlist", watchlist_id=watchlist_id))
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

        Also pushes the new scope into the still-mounted `WatchlistTree`
        (task-876): since this watcher is the single reconciliation point
        for BOTH a real tree click and a breadcrumb promotion (the latter
        never touches the tree widget at all -- see
        `handle_breadcrumb_scope_selected`), and neither one rebuilds the
        Tree instance (only FEEDS refreshes above), the tree's own
        `active_scope` would otherwise go stale the moment the scope changes
        by any path other than a fresh `_build_tree_pane` construction.
        """
        if not self.is_mounted:
            return
        self._refresh_feeds_region_for_scope()
        try:
            self.query_one("#wl-tree", WatchlistTree).active_scope = self.tree_scope
        except NoMatches:
            pass
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
        if not self.is_mounted:
            return
        try:
            label = self.query_one("#watchlists-backend-label", Static)
            label.update(self._backend_label_text())
        except Exception:
            pass
        # task-895: push the new write-availability into the still-mounted
        # tree, the same way `watch_tree_scope` pushes `active_scope`. The
        # snapshot refresh below does eventually recompose the whole screen
        # (and `_build_tree_pane` would then re-seed this), but that is an
        # async round trip -- until it lands, the five action buttons would
        # sit enabled over a backend that cannot service them, which is the
        # exact "disabled button that looks enabled" shape in reverse.
        try:
            self.query_one("#wl-tree", WatchlistTree).write_disabled_reason = (
                self._tree_write_disabled_reason()
            )
        except NoMatches:
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
        if event.ignore_selectors is not None:
            self._source_create_draft_selectors = event.ignore_selectors

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
        # Back to "untouched", so the next create form is prefilled again
        # rather than inheriting the selectors of the source just submitted.
        self._source_create_draft_selectors = None
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
            logger.opt(exception=True).warning("Failed to create source.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to create source.", severity="error")
        # Reload every view derived from the source list, not just the two
        # that happened to be here (TASK-1040). `_refresh_local_wc_snapshot`
        # feeds the staging line and `_refresh_overview_data` the cards, but
        # `#sources-table` and the rail's counts read their own queries — so
        # without these the table kept the previous list and the rail said
        # `All sources  0` while the centre said `Feeds in All sources (1)`,
        # describing the same thing on one screen.
        self._refresh_local_wc_snapshot()
        self._refresh_overview_data()
        self.run_worker(self._load_sources(), exclusive=True, group="wc_sources")
        self._load_tree_data()

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
            logger.opt(exception=True).warning("Failed to cancel run.")
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
            logger.opt(exception=True).warning("Failed to preview source.")
            if callable(notify):
                notify("Failed to preview source.", severity="error")

    @on(CheckNowRequested)
    def handle_check_now_requested(self, event: CheckNowRequested) -> None:
        event.stop()
        entity = event.entity
        if entity is None:
            return
        self.run_worker(self._check_now_source(entity), exclusive=True)

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

    async def _check_now_source(self, source: dict[str, Any]) -> None:
        """Run a check for one source and report what actually happened.

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
        """
        notify = getattr(self.app_instance, "notify", None)
        source_id = source.get("id")
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
                notify(f"Check failed: {exc}", severity="error", timeout=10)
        else:
            failure = self._check_failure_message(result)
            if failure is not None:
                logger.warning(
                    f"Check now for watchlist source {source_id!r} finished "
                    f"failed: {failure}"
                )
                if callable(notify):
                    notify(f"Check failed: {failure}", severity="error", timeout=10)
            elif callable(notify):
                # Only claim completion for a terminal status. `check_now` on
                # the server backend delegates to `launch_run`, which triggers
                # execution asynchronously and returns `queued`/`running` — so
                # a fixed "Check complete." would tell the user the fetch had
                # finished while it was still in flight (Qodo #4 on PR #1047).
                status = str((result or {}).get("status") or "").lower()
                if status in self._TERMINAL_RUN_STATUSES:
                    notify("Check complete.", severity="information")
                else:
                    notify("Check started.", severity="information")
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
            logger.opt(exception=True).warning("Failed to import OPML.")
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
            logger.opt(exception=True).warning("Failed to export OPML.")
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

    def _briefing_scope_label(self) -> str:
        """The pane's one-line statement of what it is showing, and from where."""
        watchlist_id = self._briefing_watchlist_id()
        if watchlist_id is None:
            return (
                "Select a watchlist in the rail to see or write its briefings — "
                "a briefing covers one watchlist."
            )
        name = self._watchlist_display_name(watchlist_id)
        # RAW, deliberately: the pane wraps this in a `rich.text.Text`, which
        # is never markup-parsed, so escaping here would put visible
        # backslashes in front of every bracket a real name contains. See
        # `ArtifactsPane.compose` for why that wrapper is load-bearing --
        # a bare `str` in a `Static` IS parsed as markup.
        return f"Briefings for {name} · written on this device, on request"

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
            self._loaded_scripts = []
            self._selected_script = None
            self._loaded_script_audio = None
            self._scripts_with_audio = {}
            self._loaded_citations = []
            self._citation_item_lookup = {}
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
            try:
                settings_row, preset_rows = await asyncio.to_thread(
                    self._read_watchlist_briefing_state, db, watchlist_id
                )
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                logger.warning(
                    "Failed to read briefing settings/presets for watchlist "
                    f"{watchlist_id}: {type(exc).__name__}"
                )
                settings_row, preset_rows = {}, self._loaded_briefing_presets
            mode = settings_row.get("briefing_selection_mode")
            self._briefing_selection_mode = (
                str(mode) if mode in VALID_MODES else MODE_AUTO_FEATURED
            )
            self._briefing_default_preset_id = settings_row.get(
                "default_briefing_preset_id"
            )
            self._loaded_briefing_presets = preset_rows
        if not self.is_mounted:
            return
        try:
            pane = self.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        except NoMatches:
            return
        pane.briefings = self._loaded_briefings
        pane.selected_briefing = self._selected_briefing
        pane.scope_label = self._briefing_scope_label()
        pane.can_generate = self._can_generate_briefing()
        pane.selection_mode = self._briefing_selection_mode
        pane.presets = self._loaded_briefing_presets
        pane.default_preset_id = self._briefing_default_preset_id
        pane.scripts = self._loaded_scripts
        pane.selected_script = self._selected_script
        pane.script_audio = self._loaded_script_audio
        pane.scripts_with_audio = self._scripts_with_audio
        pane.citations = self._loaded_citations

    @staticmethod
    def _read_watchlist_briefing_settings(db: Any, watchlist_id: int) -> dict[str, Any]:
        """The watchlist's stored `briefing_selection_mode`/
        `default_briefing_preset_id`, as a plain dict.

        Raw SQL against `db.conn`, matching `briefing_service._selection_
        mode`'s own read of the same column -- `WatchlistBundleService.
        list_watchlists`/`_get` deliberately select a narrower column list
        that predates these two (Task 1), so there is no existing
        service-layer getter for them to reuse. Always called through
        `asyncio.to_thread`; never call this directly from the UI thread.
        """
        row = db.conn.execute(
            "SELECT briefing_selection_mode, default_briefing_preset_id "
            "FROM watchlists WHERE id = ?",
            (watchlist_id,),
        ).fetchone()
        return dict(row) if row is not None else {}

    @staticmethod
    def _read_watchlist_briefing_state(
        db: Any, watchlist_id: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """`_read_watchlist_briefing_settings` plus `list_briefing_presets`,
        as one synchronous unit for `_load_briefings` to dispatch through a
        SINGLE `asyncio.to_thread` call.

        Both are cheap reads on the same thread-local connection; bundling
        them here is purely about round trips through the thread pool, not
        about the SQL itself -- `_load_briefing_presets` still does its own
        separate `list_briefing_presets` call for its OTHER caller
        (`_open_briefing_preset_manager`), which has no settings row to
        read alongside it. Always called through `asyncio.to_thread`; never
        call this directly from the UI thread.
        """
        settings_row = WatchlistsCollectionsScreen._read_watchlist_briefing_settings(
            db, watchlist_id
        )
        preset_rows = [dict(row) for row in db.list_briefing_presets()]
        return settings_row, preset_rows

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
        try:
            pane = self.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        except NoMatches:
            pane = None
        if pane is not None:
            pane.scripts = []
            pane.selected_script = None
            pane.script_audio = None
            pane.scripts_with_audio = {}
            pane.citations = []
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

        For a resolving id, switches to the Items ("Read") section --
        `ContentPane` is only ever mounted there (`_visible_region_layout`)
        -- and, once that section's `ItemsPane` exists, hands it the
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
                items_pane = self.query_one("#watchlists-items-pane", ItemsPane)
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

    # --- Briefing selection-mode and default-preset pickers (Task 4) -------
    #
    # Same write-first-patch-after shape as `handle_toggle_briefing_queue_
    # requested` -> `_toggle_briefing_queue`: the handler answers the
    # no-database case from memory and dispatches a worker; the worker does
    # the write off the UI thread (`asyncio.to_thread`), then on success
    # patches `_briefing_selection_mode`/`_briefing_default_preset_id` and
    # the mounted pane's matching reactive DIRECTLY -- never `_load_
    # briefings()`, which would re-query the database for a value this
    # write already knows. No `exclusive=True`: each picker's own writes
    # target a single row with `UPDATE ... WHERE id = ?`, so two overlapping
    # presses are safe to interleave (last write wins), and cancelling one
    # mid-write would leave `_briefing_selection_mode`/`_briefing_default_
    # preset_id` disagreeing with what actually landed in the database.

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

        `fail_interrupted_briefings` fails EVERY `generating` row for a
        watchlist unconditionally -- it cannot itself tell a crashed
        worker's row from a live one, both read `generating`. The Generate
        path (`_sweep_and_guard`) never needs this guard: it always runs at
        the very front of `_generate_briefing`, before that worker's own
        row is inserted, so there is nothing of "its own" yet to protect.
        The Artifacts-load path (`_load_briefings`) has no such ordering
        guarantee -- it can run at any time, including while a generation
        THIS screen started is still mid-flight and genuinely owns a
        `generating` row -- so it consults the same signal
        `handle_generate_briefing_requested` already trusts before ever
        dispatching a worker, rather than inventing a second, possibly
        weaker way to tell a zombie from a live row (whole-branch review
        fix 3).
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
        """
        if not self._zombie_sweep_is_safe():
            return 0
        return await asyncio.to_thread(fail_interrupted_briefings, db, watchlist_id)

    def _sweep_and_guard(
        self, db: Any, watchlist_id: int
    ) -> tuple[list[str], list[str]]:
        """Zombie sweep, then the generating-check. Runs off the UI thread.

        The order is the contract `briefing_service` states: the service
        neither checks nor recovers -- folding either in would make it both
        the thing guarded and the guard -- so the caller sweeps FIRST, and
        only then asks whether anything is still generating. A row orphaned
        by a crashed worker can therefore never wedge the guard shut.

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
        fail_interrupted_briefings(db, watchlist_id)
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
                    self._sweep_and_guard, db, watchlist_id
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
    # One real difference from Generate: `briefing_scripts` has no
    # one-generating-row-per-briefing invariant the way `briefings` has one
    # per watchlist (a briefing can be cast many times, with different
    # rosters, and `briefing_cast.py`'s own module docstring says so
    # explicitly). So there is no `_sweep_and_guard`-style `blocking` check
    # here -- recovering a zombie script does not itself refuse THIS cast
    # attempt the way recovering a zombie briefing refuses THIS generation
    # attempt. The zombie sweep still runs, in TWO separate seams, pinned
    # separately by this task's own tests: `_load_briefings` sweeps whenever
    # Artifacts loads (gated on `_cast_sweep_is_safe`, the `_zombie_sweep_
    # is_safe` sibling), and `_cast_script` below sweeps again at the front
    # of every cast, exactly where `_sweep_and_guard` runs for Generate.

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
        """
        if not self._cast_sweep_is_safe():
            return 0
        return await asyncio.to_thread(fail_interrupted_scripts, db, briefing_id)

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
        """
        chachanotes_db = getattr(self.app_instance, "chachanotes_db", None)
        load_character = (
            self._cast_load_character if chachanotes_db is not None else None
        )
        try:
            try:
                await asyncio.to_thread(fail_interrupted_scripts, db, briefing_id)
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
    # script` sweeps for Cast.

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
        """
        if not self._audio_sweep_is_safe():
            return 0
        return await asyncio.to_thread(fail_interrupted_audio, db, script_id)

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
        """
        try:
            try:
                await asyncio.to_thread(fail_interrupted_audio, db, script_id)
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
        try:
            self.query_one("#watchlists-content-pane", ContentPane).item = event.item
        except NoMatches:
            pass
        self._mark_item_read_on_open(event.item)

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
        button click. `_update_item_status`'s default refresh reloads
        `ItemsPane.items` and calls `_refresh_overview_data()`, and
        `overview_data` is `reactive({}, recompose=True)` -- a SCREEN-level
        recompose, which rebuilds every region via its factory
        (`_build_list_pane`/`_build_content_pane`/etc.), replacing the live
        `ItemsPane`/`DataTable` instances wholesale. Proven live: with the
        default refresh, one item selection detached the old `ItemsPane`,
        reset the table cursor to 0, cleared screen focus, and a SECOND
        arrow-key press did nothing at all. `patch_item` mutates the same
        dict object already held by `ItemsPane.items`/
        `_selected_content_item`/`ContentPane.item` in place instead, so a
        later status check sees "reviewed" without forcing a rebuild.

        This reuses the exact status column `_update_item_status` already
        writes for the deliberate item-status actions (Ingest/Ignore, the
        unread toggle) -- `SubscriptionsDB.mark_item_status`, keyed by the
        item's own row id, not by any (watchlist, item) pair -- so it is
        global by construction: the same article read from "All sources" is
        read in every watchlist whose sources include it.
        """
        if item is None:
            return
        if str(item.get("status") or "").strip().lower() != "new":
            return
        item_id = item.get("id")
        if item_id is None:
            return
        self.run_worker(
            self._update_item_status(
                item_id,
                "reviewed",
                notify_toast=False,
                refresh=False,
                patch_item=item,
            ),
            exclusive=True,
            # Whole-branch review (Important): `exclusive=True` with no
            # `group=` lands in the DEFAULT group, which ~25 call sites on
            # this screen share -- including `_check_now_source`. Since Phase
            # D this worker fires on every item selection and every `j`/`k`,
            # so opening an item cancelled an in-flight "Check now" network
            # fetch the user had just been toasted about. Give both
            # item-status writes their own group so they only ever supersede
            # each other.
            group=_ITEM_STATUS_WORKER_GROUP,
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

        The refusal is decided in `_mark_item_unread`, by asking the backend,
        NOT from `event.item` (re-review, Important). `event.item` is
        `ContentPane.item` -- the dict the screen has held since the item was
        selected -- and `handle_ingest_requested`/`handle_ignore_requested`
        call `_update_item_status` with no `patch_item=`, so that dict is
        never updated when they run. `patch_item=` is passed by exactly one
        caller in the whole app (`_mark_item_read_on_open`) and by neither of
        those two. Ingest an open item and the reader's dict still says
        `reviewed`, so a guard reading `event.item` never fires and the button
        destroys the ingest anyway -- reproduced end to end.
        """
        event.stop()
        item = event.item
        if item is None:
            return
        item_id = item.get("id")
        if item_id is None:
            return
        self.run_worker(
            self._mark_item_unread(item_id),
            exclusive=True,
            group=_ITEM_STATUS_WORKER_GROUP,
        )

    async def _mark_item_unread(self, item_id: Any) -> None:
        """Ask the backend for the item's real status, then refuse or write.

        Deciding here, from a live query, rather than keeping the screen's
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

        `_loaded_items` is NOT the system of record and cannot be used here:
        `local_watchlists_service.list_items` collapses `status=None` to
        `status="new"` (verified), so an ingested item is not merely stale in
        that cache -- it is absent from it entirely, along with every other
        non-`new` item. An earlier version of this fix read `_loaded_items`
        after an awaited `_load_items()` and still wrote `new` over a live
        ingest, for exactly that reason.

        Fails CLOSED. If the backend cannot be asked, the write is refused
        and the user is told to retry: marking unread is a convenience the
        user can repeat, whereas overwriting an ingest is not recoverable, so
        an unanswered question must not resolve in favour of the destructive
        branch.

        Args:
            item_id: Normalized id of the item to mark unread.
        """
        try:
            blocking = await self._blocking_status_for(item_id)
        except Exception:
            logger.opt(exception=True).warning(
                "Could not confirm an item's status before marking it unread."
            )
            self.notify(
                "Could not confirm this item's current status, so it was left "
                "unchanged. Try again.",
                severity="warning",
            )
            return
        if blocking is not None:
            self.notify(
                f"This item is marked {blocking}; leaving it as it is rather "
                "than overwriting that with unread.",
                severity="warning",
            )
            return
        await self._update_item_status(item_id, "new")

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
        """Mirror the Items filter/search so a workbench rebuild can restore it."""
        event.stop()
        self._items_status_filter = event.status_filter
        self._items_search_query = event.search_query

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
        `_create_source` does. `overview_data` is `reactive({}, recompose=True)`
        on this screen, so touching it rebuilds every region through its
        factory and replaces the mounted panes wholesale -- proven live in
        Phase D Task 5 to detach the `ItemsPane`, reset the `DataTable` cursor
        and drop keyboard focus. Nothing the user can see is derived from a
        source's selectors: not the Sources table's five columns, not the
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
        self.run_worker(self._update_item_status(entity.get("id"), "ingested"), exclusive=True)

    @on(IgnoreRequested)
    def handle_ignore_requested(self, event: IgnoreRequested) -> None:
        event.stop()
        entity = event.entity
        if entity is None:
            return
        self.run_worker(self._update_item_status(entity.get("id"), "ignored"), exclusive=True)

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
                pane = self.query_one("#watchlists-items-pane", ItemsPane)
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
        `overview_data`, `reactive({}, recompose=True)` on the screen, so
        calling it after EVERY item selection forced a full screen
        recompose -- proven live to detach the mounted `ItemsPane`, reset
        the `DataTable` cursor, and drop keyboard focus, so a second arrow
        key did nothing. Used only by the silent auto-mark-read-on-open
        path; every deliberate action (Ingest/Ignore, the unread toggle)
        keeps refreshing as before. When `refresh` is False and the write
        succeeds, `patch_item` -- the same dict object already held by
        `ItemsPane.items`/`_selected_content_item`/`ContentPane.item` -- is
        mutated in place instead, so a later status check already sees the
        new value without forcing a rebuild.
        """
        notify = getattr(self.app_instance, "notify", None)
        try:
            await self._controller.update_item_status(
                runtime_backend=self.runtime_backend,
                item_id=item_id,
                status=status,
            )
            if patch_item is not None:
                patch_item["status"] = status
                # Whole-branch review (Important): the in-place patch is
                # invisible -- rows are built once in `ItemsPane.compose()`
                # and this path deliberately never recomposes, so the Status
                # column read "new" for every item the user had opened until
                # they left the tab. Repaint the one cell instead.
                self._repaint_item_status_cell(patch_item.get("id"), status)
            if notify_toast and callable(notify):
                label = "unread" if status == "new" else status
                notify(f"Item marked {label}.", severity="information")
        except Exception:
            logger.opt(exception=True).warning(f"Failed to mark item {status}.")
            if notify_toast and callable(notify):
                notify(f"Failed to mark item {status}.", severity="error")
        if refresh:
            self.run_worker(self._load_items(), exclusive=True)
            self._refresh_overview_data()

    def _repaint_item_status_cell(self, item_id: Any, status: str) -> None:
        """Push a patched status into the mounted Items table's Status cell."""
        try:
            pane = self.query_one("#watchlists-items-pane", ItemsPane)
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
            logger.opt(exception=True).warning("Failed to delete source.")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Failed to delete source.", severity="error")
        # Reload every view derived from the source list, not just the two
        # that happened to be here (TASK-1040). `_refresh_local_wc_snapshot`
        # feeds the staging line and `_refresh_overview_data` the cards, but
        # `#sources-table` and the rail's counts read their own queries — so
        # without these the table kept the previous list and the rail said
        # `All sources  0` while the centre said `Feeds in All sources (1)`,
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
            logger.opt(exception=True).warning("Failed to ignore item.")
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
            "1=Overview 2=Sources 3=Items 4=Runs 5=Rules 6=Notifications "
            "7=Artifacts | n=new d=delete c=check p=preview ?=help",
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
        (`_mark_item_read_on_open` calls `_update_item_status(...,
        refresh=False, patch_item=item)`, patching the item dict in place
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
            pane = self.query_one("#watchlists-items-pane", ItemsPane)
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
