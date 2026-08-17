"""Sources pane for the watchlists screen."""

from __future__ import annotations

from typing import Any

from loguru import logger
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.css.query import NoMatches
from textual.widgets import Button, DataTable, Input, Select, Static, Switch, TextArea

from ...Subscriptions.html_text import strip_control_characters
from ...Subscriptions.noise_defaults import (
    default_ignore_selectors_text,
    first_invalid_selector,
    invalid_selector_message,
)
from ...Utils.input_validation import sanitize_string, validate_text_input, validate_url
from ...Widgets.prune_safe_select import PruneSafeSelect
from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .humane_time import humane_timestamp
from .inspector_pane import CheckNowRequested, PreviewRequested


class SourceSelected(Message):
    """Posted when the user selects a source in the sources table."""

    def __init__(self, source: dict[str, Any] | None) -> None:
        self.source = source
        super().__init__()


class CreateSourceRequested(Message):
    """Posted when the user submits the new-source form."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__()


class ImportOpmlRequested(Message):
    """Posted when the user requests an OPML import."""


class ExportOpmlRequested(Message):
    """Posted when the user requests an OPML export."""


class CreateFormDraftChanged(Message):
    """Posted whenever a create-form free-text field changes.

    `SourcesPane` lives inside a `WatchlistsWorkbench` region, and that
    region is swapped for a freshly built one whenever it collapses or
    expands, or whenever the section switches — each of which constructs a
    brand new `SourcesPane`. (Until task-15461 the trigger was wider still:
    `region_layout` was `recompose=True`, so `[` on the left rail — a region
    unrelated to Sources — rebuilt this pane too.) Without this message the
    screen has no way to know what was typed, so the draft would be silently
    lost on the next such rebuild.
    The owning screen mirrors this into its own state and seeds it back into
    the freshly-constructed pane.
    """

    def __init__(
        self,
        name: str,
        url: str,
        tags: str,
        ignore_selectors: str | None = None,
        source_type: str | None = None,
        destination: Any = None,
    ) -> None:
        self.name = name
        self.url = url
        self.tags = tags
        #: The noise-selector text, or None when the pane has nothing to
        #: report (it only ever posts a string, but the screen stores None
        #: for "untouched", so the field is optional for any other caller).
        self.ignore_selectors = ignore_selectors
        #: TASK-2302. The chosen type and destination, carried for the same
        #: reason the three free-text fields are: both are pane-local state
        #: that a workbench rebuild would otherwise reset to a class default,
        #: silently moving a source the user had already aimed somewhere.
        #: Both are None when the poster has nothing to report -- the pane's
        #: own `_post_create_draft_changed` always fills them in, and a
        #: destination of "no watchlist" is the `UNASSIGNED_DESTINATION`
        #: string, never None, precisely so the two cases stay distinct.
        self.source_type = source_type
        self.destination = destination
        super().__init__()


class CreateFormVisibilityChanged(Message):
    """Posted whenever the create form opens or closes, for the same reason
    `CreateFormDraftChanged` exists: `show_create_form` is pane-local state
    that would otherwise reset to its class default on every rebuild."""

    def __init__(self, is_open: bool) -> None:
        self.is_open = is_open
        super().__init__()


class SourcesPane(RecomposeCaptureGuard, Vertical):
    """Source list, search/filter, and create form for watchlists."""

    #: task-876: Rich's own terminal-agnostic "current item" idiom (see
    #: `snippet_editor.py`'s `_WHITESPACE_MARKER_STYLE`), used because a
    #: `DataTable` cell's `Text` cannot reference Textual CSS variables
    #: ($ds-focus-bg etc.) the way a widget's own styles can. Unlike
    #: NotificationsPane, `selected_source` below is deliberately NOT
    #: `recompose=True` (a selection must not rebuild the table under the
    #: user), so this style is also applied via a targeted `update_cell`
    #: pass in `_update_selection_highlight` rather than solely in
    #: `compose()`.
    _SELECTED_ROW_STYLE = "reverse bold"

    sources = reactive[list[dict[str, Any]]](list, recompose=True)
    selected_source = reactive[dict[str, Any] | None](None)
    #: TASK-2309. The source ids ("id" field, the same namespaced form
    #: `selected_source` carries) with a check currently in flight anywhere
    #: on the screen -- not just the one selected here, since the Inspector
    #: can trigger a check for a source that is not this pane's current
    #: selection. `WatchlistsCollectionsScreen._checks_in_flight` is the
    #: source of truth; this mirrors it in, the same way `sources` mirrors
    #: `_loaded_sources`. Deliberately NOT `recompose=True`: a check starting
    #: or finishing must not rebuild the table under the user (the same
    #: reason `selected_source` above is not), so `watch_busy_source_ids`
    #: repaints the one button that can show it, surgically.
    busy_source_ids = reactive[frozenset[str]](frozenset())
    #: task-15460: the five filters are PLAIN reactives. All five were
    #: `recompose=True`, so a single character typed into the search box (or
    #: the tags box) tore down and rebuilt this entire pane -- toolbar, the
    #: eight-control create form if it happened to be open, and the table --
    #: and `recompose()` then had to put focus back into the input it had
    #: just destroyed. A `DataTable`'s rows are data, not widgets, so
    #: re-populating it (`_refresh_table_rows`) mounts nothing and leaves
    #: the focused `Input`, its caret and any open form exactly where they
    #: were. The create-form reactives below stay `recompose=True`: those
    #: genuinely change WHICH CONTROLS EXIST.
    search_query = reactive("")
    source_type_filter = reactive("all")
    status_filter = reactive("all")
    active_filter = reactive("all")
    tags_filter = reactive("")
    show_create_form = reactive(False, recompose=True)
    show_filter_editor = reactive(False, recompose=True)
    # Seed values for the create form's free-text inputs. No `recompose=True`:
    # these only need to be read once per `compose()` call (to seed the
    # Input's `value=`), which already happens whenever `show_create_form` (or
    # anything else) triggers a rebuild — see `CreateFormDraftChanged`.
    create_draft_name = reactive("")
    create_draft_url = reactive("")
    create_draft_tags = reactive("")
    # TASK-1362 (spec §2). Seeds `#sources-create-ignore-selectors`. Its
    # default is the shipped noise set, so a form that has never been touched
    # opens *prefilled and visible* -- that prefill lives here, in the form,
    # rather than being applied silently at save time, because the user has to
    # be able to see and edit what is being suppressed. It is a draft like the
    # three above for the same reason: a workbench rebuild constructs a new
    # pane, and re-seeding the default over a user's edit (or over a field
    # they deliberately emptied) would be re-filling it behind their back.
    create_draft_ignore_selectors = reactive(default_ignore_selectors_text())
    # TASK-2302. Which watchlist the new source will JOIN, and the choices
    # offered for it. Both are screen-seeded (the pane has no service of its
    # own; see `_build_detail_pane`), and both are plain reactives -- read
    # once per `compose()` to build the Select, exactly like the three drafts
    # above. `create_draft_destination` is a draft in the same sense too: a
    # workbench rebuild constructs a brand new pane, and re-deriving the
    # destination from the scope there would silently overwrite a choice the
    # user had already made in a form that is still open.
    watchlist_choices = reactive[list[dict[str, Any]]](list)
    default_destination = reactive[Any]("unassigned")
    create_draft_destination = reactive[Any]("unassigned")
    # TASK-2302. The chosen source type, which decides whether the
    # ignore-selectors field is rendered at all -- hence `recompose=True`,
    # unlike its sibling drafts. Owning the conditional in `compose()` (the
    # single place that already decides what this form contains) rather than
    # mounting and unmounting the field from a watcher is deliberate: a
    # conditionally-composed control with a second, in-place owner is a bug
    # class this codebase has paid for more than once.
    create_draft_source_type = reactive("rss", recompose=True)

    #: The value the destination Select carries for "no watchlist". A string
    #: rather than `None`: `Select` reserves a `NoSelection` sentinel of its
    #: own and this control is `allow_blank=False`, so "unassigned" has to be
    #: a real, selectable option value like any other.
    UNASSIGNED_DESTINATION = "unassigned"

    #: Source types `ignore_selectors` can actually affect. Mirrors
    #: `InspectorPane._URL_FAMILY_SOURCE_TYPES` -- the same question, asked of
    #: a type the user is choosing rather than of a source that exists.
    _URL_FAMILY_TYPES = frozenset({"url", "url_list", "sitemap", "site"})

    # Plain attribute, not a reactive: mirrors which row `compose()` last
    # painted as selected, so `_update_selection_highlight` knows which row
    # to revert without re-deriving it from `selected_source`'s OLD value
    # (a one-argument `watch_selected_source` only ever sees the new one).
    _highlighted_source_key: str | None = None

    #: The create form's focusable controls, in visual order (TASK-1035).
    #: `compose()` yields them in this order and the form is a plain
    #: `Vertical`, so DOM order, paint order and Tab order are the same list.
    #:
    #: TASK-2302: `sources-create-ignore-selectors` is now rendered only for
    #: url-family types, so this tuple is a superset of what is mounted at
    #: any one moment. That is fine for both readers -- it is a membership
    #: test in `_focused_create_field_id` and a focus target in `recompose`,
    #: which already tolerates a missing id (`NoMatches`).
    _CREATE_FORM_FIELD_IDS = (
        "sources-create-name",
        "sources-create-url",
        "sources-create-type",
        "sources-create-active",
        "sources-create-watchlist",
        "sources-create-tags",
        "sources-create-frequency",
        "sources-create-ignore-selectors",
        "sources-create-submit",
        "sources-create-cancel",
    )

    #: The noise field's visible label and its help copy, carried as the
    #: TextArea's border title/subtitle (TASK-1362). Border rows are rows the
    #: field already spends on its border, and the Sources pane is 16 rows
    #: tall at 160x42 with the create form open -- two extra `Static` rows for
    #: this text is two rows the form does not have. Both strings are painted
    #: whenever the field is, which a tooltip would not be.
    #:
    #: TASK-1362 close-out (spec AC#2): the help copy also states
    #: `change_threshold`'s role, since it is the other half of "why did/did
    #: not a change fire" and has no live UI of its own to explain it in.
    #:
    #: TASK-2302 (F11/F12) SHORTENED both strings, and the first version of
    #: this comment justified that with a re-measurement that was itself
    #: wrong -- it read 53/78 columns off a test harness that loads no
    #: stylesheet at all. Corrected by the whole-branch review, which
    #: measured through the production stylesheet: TASK-1362's figure above
    #: is RIGHT. The field is **93** columns at 160x42 and **168** at
    #: 235x52, so the previous 65-character label and 83-character help copy
    #: both fit that layout with room to spare, and the truncation the UAT
    #: filed ("…changes always report; change_threshold") is NOT explained by
    #: this layout -- see the task-2302 notes; it is unresolved, and may be a
    #: narrower terminal or an older build.
    #:
    #: The shorter strings are kept anyway, on their own merits rather than
    #: on a width argument: they say the same thing in half the columns, they
    #: cannot truncate at any size this app supports, and the syntax detail
    #: they displaced moves to the tooltip, which has no width budget at all
    #: -- the same trade the Inspector's copy of this field documents for its
    #: own, even shorter label. `test_the_noise_help_text_fits_the_field_it_
    #: is_painted_on` now measures through the production stylesheet, so the
    #: numbers here are checkable rather than asserted.
    _IGNORE_SELECTORS_LABEL = "Ignore elements (CSS selectors, one per line)"
    _IGNORE_SELECTORS_HELP = "Silence noise; change_threshold limits volume."
    _IGNORE_SELECTORS_TOOLTIP = (
        "One rule per line; a comma inside a line groups selectors, as in "
        ":is(.a, .b). Matching elements are dropped before the page is "
        "compared, so their churn stops reporting a change. Changes to what "
        "is left always report; change_threshold limits how much has to move "
        "before they do."
    )

    #: Cap on the stored selector text. Generous next to the shipped default
    #: (~180 characters) and the Tags field's 1000, because a long-watched
    #: page legitimately accumulates rules.
    _IGNORE_SELECTORS_MAX_LENGTH = 4000

    #: How often a new source is checked, in seconds. Mirrors the
    #: `check_frequency INTEGER DEFAULT 3600` column in Subscriptions_DB, so
    #: leaving the control alone reproduces the database default (TASK-1210).
    _FREQUENCY_OPTIONS = [
        ("Every 15m", 900),
        ("Every 1h", 3600),
        ("Every 6h", 21_600),
        ("Every 24h", 86_400),
    ]
    _DEFAULT_FREQUENCY_SECONDS = 3600

    #: Which create-form control `recompose()` should focus once it has
    #: remounted this pane's children. See `recompose` for why focus has to
    #: be re-homed by hand.
    _pending_create_focus: str | None = None

    #: Ceiling on `_confirm_create_focus`'s self-reschedule (Qodo, TASK-1345
    #: follow-up). See that method's docstring for why a bound is safe here.
    _CREATE_FOCUS_CONFIRM_MAX_ATTEMPTS = 20

    #: TASK-2302 adds `Web page`. Every other entry here is a FEED type, and
    #: `ignore_selectors` (an element-level rule) can only ever affect a
    #: scraped page -- so before this the create form could not produce a
    #: single source the noise field applied to, which is why that field read
    #: as decorative prefill. `url` is the value
    #: `LocalWatchlistsService._local_type_for_source_type` accepts verbatim
    #: and the value `normalize_local_subscription_row` publishes back, so
    #: this one entry serves the create Select and the filter Select alike.
    _TYPE_OPTIONS = [
        ("All", "all"),
        ("RSS", "rss"),
        ("Atom", "atom"),
        ("Feed", "feed"),
        ("Playlist", "playlist"),
        ("Channel", "channel"),
        ("Web page", "url"),
    ]

    _STATUS_OPTIONS = [
        ("All statuses", "all"),
        ("OK", "ok"),
        ("Error", "error"),
        ("Paused", "paused"),
        ("Pending", "pending"),
    ]

    _ACTIVE_OPTIONS = [
        ("All", "all"),
        ("Active", "active"),
        ("Inactive", "inactive"),
    ]

    def compose(self):
        with Vertical(id="sources-toolbar"):
            # TASK-995: `compact=True` on every Input/Select in a
            # `.destination-filter-strip` row. That class pins `height: 1`
            # (layout/_panes.tcss) while a bordered Input/Select is three
            # rows, so without this the whole strip painted as its own top
            # border and nothing else -- no search box, no filters, no
            # `New source` -- and a new user had no way to add a source at
            # all. Widths are pinned alongside it in features/_watchlists.tcss
            # (they have to be in the bundle to beat the global
            # `Select { width: 100% }` in features/_conversations.tcss).
            with Horizontal(classes="destination-filter-strip"):
                yield Input(
                    placeholder="Search sources...",
                    id="sources-search-input",
                    value=self.search_query,
                    # `select_on_focus=False` is load-bearing (same as
                    # `#items-search-input`, task-2513): Textual's default
                    # `select_on_focus=True` makes the programmatic refocus
                    # after a recompose select ALL text, so the user's next
                    # keystroke REPLACES the half-typed query instead of
                    # appending to it. Typing no longer causes that
                    # recompose (task-15460), but opening the create form
                    # still does, and a click back into the box must land
                    # the caret rather than arm the term for deletion.
                    select_on_focus=False,
                    compact=True,
                )
                # TASK-2310: UAT read this row as "All / All statuses /
                # All" -- two of the three unlabeled. A persistent sibling
                # `Static` (this screen's established idiom, TASK-2302) was
                # tried first and measured against the production
                # stylesheet: at this toolbar's *tested floor*, 160x42, the
                # row already spends every column it has -- the search
                # box's placeholder ("Search sources...") only reaches full
                # width today because the three Selects claim exactly zero
                # spare columns, and adding even one label pushes `Filters`
                # off the pane's right edge (measured: `#sources-filter-
                # toggle` at x=118..134 against a 93-column pane -- see
                # `test_watchlists_sources_toolbar_controls_are_actually_
                # visible`). A `tooltip` costs no column at all, so it is
                # what fits: every Select below states what it filters on
                # hover. A compact Select has no border for a border-title
                # to sit on either way (TASK-2300).
                yield PruneSafeSelect(
                    self._TYPE_OPTIONS,
                    value=self.source_type_filter,
                    id="sources-type-select",
                    allow_blank=False,
                    compact=True,
                    tooltip="Filter by source type.",
                )
                yield PruneSafeSelect(
                    self._STATUS_OPTIONS,
                    value=self.status_filter,
                    id="sources-status-filter",
                    allow_blank=False,
                    compact=True,
                    tooltip="Filter by source status.",
                )
                yield PruneSafeSelect(
                    self._ACTIVE_OPTIONS,
                    value=self.active_filter,
                    id="sources-active-filter",
                    allow_blank=False,
                    compact=True,
                    tooltip="Filter by whether a source is active.",
                )
                # TASK-2303 AC#1: `New source`, not `New Source`, and never
                # `Add`. NEW is the create verb across this screen; ADD is
                # membership (the rail's `Add existing…`, the Inspector's
                # `Add to watchlist…`). The two must not be near-synonyms.
                yield Button(
                    "New source",
                    id="sources-new-button",
                    variant="primary",
                    tooltip=(
                        "Create a source that does not exist yet. To put a "
                        "source you already have into a watchlist, use "
                        "Add existing in the rail."
                    ),
                )
                yield Button("Filters", id="sources-filter-toggle", variant="default")
            if self.show_filter_editor:
                with Horizontal(id="sources-filter-editor", classes="destination-filter-strip"):
                    yield Input(
                        placeholder="Tags (comma separated)...",
                        id="sources-tags-filter",
                        value=self.tags_filter,
                        compact=True,
                    )
            with Horizontal(classes="destination-filter-strip"):
                yield Button(
                    "Preview",
                    id="sources-preview-button",
                    disabled=self.selected_source is None,
                )
                # TASK-2309: busy state read at compose time too, not only via
                # the surgical `_update_action_buttons` repaint below -- this
                # pane is reconstructed from scratch on every workbench
                # rebuild (`_build_detail_pane`), and the screen seeds
                # `busy_source_ids` onto the fresh instance BEFORE it mounts,
                # so the very first paint has to already reflect an
                # in-flight check rather than waiting for a watcher that
                # cannot repaint a widget that does not exist yet.
                check_now_busy = self._is_check_now_busy(self.selected_source)
                yield Button(
                    "Checking..." if check_now_busy else "Check now",
                    id="sources-check-now-button",
                    disabled=self.selected_source is None or check_now_busy,
                )
                yield Button("Import OPML", id="sources-import-opml-button")
                yield Button("Export OPML", id="sources-export-opml-button")

        if self.show_create_form:
            # TASK-1035: a `Vertical`, not a `Grid`. Nothing styled
            # `#sources-create-form`, so the `Grid` fell back to Textual's
            # defaults -- one column, and rows that share the container's
            # height as `1fr` each. Measured at 235x52 that spread seven
            # controls over 23 rows with blank gaps between them, put
            # `Create` on row 40 and `Cancel` three rows below it (the
            # "stacked with blank rows" the UAT reported), and starved
            # `#sources-table` down to a single row. A `Vertical` of
            # auto-height children stacks them tight, in visual = DOM =
            # Tab order, which is what the New-watchlist dialog does too.
            with Vertical(id="sources-create-form"):
                # TASK-1362: Name and URL share a row, and the tags row below
                # is compact. Both are here to pay for the noise field: the
                # form was 13 rows in a pane that is exactly 16 at 160x42
                # (toolbar 2 + form 13 + table 1, measured), so a new field of
                # any height pushed `Create`/`Cancel` and then the table off
                # the bottom -- the same unreachable-control defect TASK-1035
                # fixed. Pairing is this form's own established answer to that
                # (see the Type/Active note below), and it costs no control
                # its borders or its size.
                yield Horizontal(
                    Input(
                        placeholder="Name",
                        id="sources-create-name",
                        value=self.create_draft_name,
                    ),
                    Input(
                        placeholder="URL",
                        id="sources-create-url",
                        value=self.create_draft_url,
                    ),
                    classes="sources-create-identity-row",
                )
                # Type and Active share a row. The pane is only 16 rows tall
                # at 160x42 and its toolbar takes two of them, so a form of
                # five full-height rows would push `Create`/`Cancel` off the
                # bottom -- unreachable, which is the same class of defect
                # this task is fixing.
                yield Horizontal(
                    # TASK-2302 AC#3: the Type Select painted as a bare
                    # "RSS ▼" with nothing saying what the word was the type
                    # OF. Labelled with the form's own existing idiom -- the
                    # `Active` Static one control to the right -- rather than
                    # a border title, because a compact/bordered Select draws
                    # its border on its child `SelectCurrent` (TASK-2300), so
                    # a title set on the Select itself has no border to sit
                    # on, and a `Static` row of its own is a row this form
                    # does not have.
                    Static("Type", classes="sources-create-field-label"),
                    PruneSafeSelect(
                        [
                            (label, value)
                            for label, value in self._TYPE_OPTIONS
                            if value != "all"
                        ],
                        value=self.create_draft_source_type,
                        id="sources-create-type",
                        allow_blank=False,
                    ),
                    Static("Active", classes="sources-create-active-label"),
                    Switch(value=True, id="sources-create-active"),
                    classes="sources-create-type-row",
                )
                # TASK-2302 AC#1: where this source will LAND, stated before
                # it is submitted and changeable. The 2026-08-04 UAT created
                # a source with a watchlist in scope, the first-run guidance
                # having just promised "press New source to add a feed to
                # it", and the source silently went to Unassigned. One
                # compact row: this is a single-value choice, and the form
                # has no full-height row to spare (see the pairing notes
                # above).
                yield Horizontal(
                    Static("Watchlist", classes="sources-create-field-label"),
                    PruneSafeSelect(
                        self._destination_options(),
                        value=self._resolved_destination(),
                        id="sources-create-watchlist",
                        allow_blank=False,
                        compact=True,
                    ),
                    classes="sources-create-destination-row",
                )
                # Tags and the check cadence share a row for the same reason
                # Type and Active do: the pane has no spare rows, and a sixth
                # full-height row would push `Create`/`Cancel` off the bottom.
                # `compact=True` on both, matching the toolbar strips two rows
                # above: it takes the row from three rows to one, which is the
                # rest of what the noise field below costs (TASK-1362).
                yield Horizontal(
                    Input(
                        placeholder="Tags (comma separated)",
                        id="sources-create-tags",
                        value=self.create_draft_tags,
                        compact=True,
                    ),
                    PruneSafeSelect(
                        self._FREQUENCY_OPTIONS,
                        value=self._DEFAULT_FREQUENCY_SECONDS,
                        id="sources-create-frequency",
                        allow_blank=False,
                        compact=True,
                    ),
                    classes="sources-create-tags-row",
                )
                # The noise control, spec §2: prefilled, visible, and editable
                # before the source is ever checked. A source's *volume* is
                # not the problem -- a page whose ad slot or view counter
                # rewrites itself reports a change every single check, and
                # nothing on this screen previously let a user say so.
                #
                # TASK-2302 AC#4: rendered only for the types it can affect.
                # CSS selectors describe elements on a scraped PAGE; for an
                # RSS/Atom feed there are no elements for a rule to match, so
                # prefilling four rows of them was prominent, prefilled and
                # inert -- and it cost the rows this form needed for the
                # destination above. The Inspector's own copy of this editor
                # has always been gated the same way
                # (`_is_url_family_source`).
                if self._type_takes_ignore_selectors(self.create_draft_source_type):
                    yield self._ignore_selectors_field()
                # `.dialog-buttons` is the same one-row, side-by-side pairing
                # `WatchlistNameDialog` uses for its own Create/Cancel, so the
                # two creation flows read the same (TASK-1035 AC#6). Only the
                # alignment is overridden for this inline form -- see
                # `features/_watchlists.tcss`.
                with Horizontal(classes="dialog-buttons sources-create-buttons"):
                    yield Button("Create", id="sources-create-submit", variant="success")
                    yield Button("Cancel", id="sources-create-cancel", variant="default")

        table = DataTable(id="sources-table")
        # TASK-2313, AC#2: "checked"/"Check now" is the vocabulary this
        # screen uses everywhere else for the same fetch action (the
        # button here and on the Inspector, toasts like "It will be
        # checked on its normal schedule."); this column was the one
        # holdout still saying "scraped".
        table.add_columns("Name", "Type", "Status", "Last checked", "Active")
        self._populate_table(table)
        yield table

    def _populate_table(self, table: DataTable) -> None:
        """Add one row per filtered source, painting the selected one.

        Shared by `compose()` and `_refresh_table_rows` so the initial paint
        and a filter change can never draw the same row differently.

        Args:
            table: The sources table, already carrying its columns and empty
                of rows.
        """
        selected_key = (
            str(self.selected_source.get("id")) if self.selected_source else None
        )
        for source in self._filtered_sources():
            row_key = str(source.get("id") or id(source))
            table.add_row(
                *self._source_row_cells(source, row_key == selected_key),
                key=row_key,
            )
        # The rows were just painted fresh from `selected_source` itself, so
        # this is authoritative going forward -- see
        # `_update_selection_highlight`'s docstring for why a later,
        # non-rebuilding selection change needs to know what to revert.
        self._highlighted_source_key = selected_key

    def _refresh_table_rows(self) -> None:
        """Re-populate the table for a filter change, without a recompose.

        task-15460. `DataTable` rows are data rather than widgets, so
        clearing and re-adding them destroys no widget: the toolbar, an open
        create form, the focused `Input` and its caret all survive a
        keystroke that changes what the table shows. `clear()` keeps the
        columns, which `_update_selection_highlight` reads back.
        """
        try:
            table = self.query_one("#sources-table", DataTable)
        except NoMatches:
            # Seeded before mount by `_build_detail_pane`; `compose()` will
            # apply the filter when it builds the table.
            return
        table.clear()
        self._populate_table(table)

    @classmethod
    def _type_takes_ignore_selectors(cls, source_type: Any) -> bool:
        """Whether `ignore_selectors` can affect a source of this type."""
        return str(source_type or "").strip().lower() in cls._URL_FAMILY_TYPES

    def _destination_options(self) -> list[tuple[Text, Any]]:
        """Options for the destination Select: Unassigned, then watchlists.

        Unassigned is FIRST and always present -- it is the honest name for
        what happens when a source belongs to no watchlist, and it is the
        only possible answer on a profile that has none. Labels are `Text`,
        not `str`: a watchlist name is user-typed, and Textual renders a
        plain-string option through its markup parser.
        """
        options: list[tuple[Text, Any]] = [
            (Text("Unassigned (no watchlist)"), self.UNASSIGNED_DESTINATION)
        ]
        for watchlist in self.watchlist_choices:
            try:
                watchlist_id = int(watchlist["id"])
            except (KeyError, TypeError, ValueError):
                continue
            options.append(
                (Text(str(watchlist.get("name") or f"Watchlist {watchlist_id}")),
                 watchlist_id)
            )
        return options

    def _resolved_destination(self) -> Any:
        """The destination value `compose()` should seed the Select with.

        Falls back to Unassigned when the draft names a watchlist that is not
        in the current choices -- it was deleted, or the backend changed
        under an open form. `Select` with `allow_blank=False` raises
        `InvalidSelectValueError` on a value it has no option for, so this is
        the difference between a stale draft and a form that will not mount.
        """
        draft = self.create_draft_destination
        if any(draft == value for _label, value in self._destination_options()):
            return draft
        return self.UNASSIGNED_DESTINATION

    def _ignore_selectors_field(self) -> TextArea:
        """The prefilled noise-selector field (TASK-1362, spec §2).

        A `TextArea` rather than an `Input` because the stored format is one
        rule per line: newlines separate independent rules and a comma inside
        a line is a CSS selector group, so a single-line control could not
        express the shipped default at all.

        Returns:
            The field, labelled and seeded from `create_draft_ignore_selectors`.
        """
        field = TextArea(
            self.create_draft_ignore_selectors,
            id="sources-create-ignore-selectors",
            # Wrapped, despite one-rule-per-line being the stored format. The
            # field's inner width at 160x42 is 89 columns and the first
            # shipped rule is exactly 89, so `soft_wrap=False` spends one of
            # the field's two content rows on a horizontal scrollbar and
            # leaves a single rule visible. A wrapped continuation row reads
            # as part of the rule above it; a scrollbar eating a quarter of
            # the field does not read as anything.
            soft_wrap=True,
        )
        field.border_title = self._IGNORE_SELECTORS_LABEL
        field.border_subtitle = self._IGNORE_SELECTORS_HELP
        # The long form, where nothing can truncate it (TASK-2302).
        field.tooltip = self._IGNORE_SELECTORS_TOOLTIP
        return field

    @staticmethod
    def source_status_text(source: dict[str, Any]) -> str:
        """What the Status column says about this source.

        TASK-1090. This read `source.get("status")` alone -- a key **no**
        watchlists normalizer emits. `normalize_local_subscription_row`
        publishes `status_summary` (`active`, `inactive`, `error`,
        `error (3)`), and the server normalizer the same. So the Status column
        rendered `-` for every source in every state, including one whose last
        check had just failed: `subscriptions.last_error` was written and the
        screen showed nothing.

        The bare `status` key is kept as a fallback because it is the shape
        this pane's own tests and any hand-built row use.

        Args:
            source: A source row as published to `sources`.

        Returns:
            The status text, or `-` when the backend reported none.
        """
        return str(source.get("status_summary") or source.get("status") or "-")

    @staticmethod
    def source_last_scraped_text(source: dict[str, Any]) -> str:
        """What the Last scraped column says. Same defect as `status` above:
        the normalizers publish `last_checked_or_scraped_at`, so this column
        read `-` even immediately after a successful check.

        TASK-2308: rendered through `humane_timestamp`, in the viewer's local
        zone. The stored value is a full UTC ISO-8601 string with
        microseconds -- 32 characters, in a table whose Name column is the
        one people actually read.

        Args:
            source: A normalized source dict; `last_checked_or_scraped_at`
                (the current normalizer field) is preferred, falling back to
                the older `last_scraped` for any hand-built row that still
                uses it.

        Returns:
            `humane_timestamp` of whichever of the two fields is present, or
            `"-"` when neither is.
        """
        return humane_timestamp(
            source.get("last_checked_or_scraped_at") or source.get("last_scraped")
        )

    @staticmethod
    def _source_row_cells(source: dict[str, Any], highlighted: bool) -> tuple[Text, ...]:
        """One row's cell values, styled if `highlighted` (task-876).

        Shared between `compose()` (the initial/any-other-reason render) and
        `_update_selection_highlight` (a same-instance selection change,
        which must not rebuild the table) so both draw an identical row.
        """
        style = SourcesPane._SELECTED_ROW_STYLE if highlighted else ""
        # Batch-4 review, I1. `name` is remote-derived: OPML import
        # (`WatchlistOpmlService.parse`) hands an `<outline text=...>`
        # attribute straight to this field with zero sanitization, and a C1
        # control byte (0x80-0x9F, e.g. a single-byte CSI introducer) is
        # valid in XML 1.0's character range -- it survives the OPML parse
        # untouched and would otherwise reach this `Text` cell (and Rich's
        # real render) verbatim. `Text.append`/`Text(...)` only protects
        # against Rich MARKUP, not raw control bytes -- see
        # `html_text.strip_control_characters`'s own docstring, which closed
        # this exact hole for the reader; it did not extend to this pane.
        return (
            Text(
                strip_control_characters(
                    source.get("name") or source.get("title") or "Untitled"
                ),
                style=style,
            ),
            Text(str(source.get("source_type") or "-"), style=style),
            Text(SourcesPane.source_status_text(source), style=style),
            Text(SourcesPane.source_last_scraped_text(source), style=style),
            Text("Yes" if source.get("active") else "No", style=style),
        )

    @staticmethod
    def _matches_status_filter(source: dict[str, Any], status_filter: str) -> bool:
        """Whether a source matches the Status filter's chosen value.

        TASK-1090. Compared `source.get("status")` to the option value with
        `==`, which could never be true for a real source: the normalizers
        publish `status_summary`, and its error form carries a count
        (`error (3)`). Filtering to `Error` -- the whole point of the filter
        for someone whose feeds have started failing -- returned nothing.

        Args:
            source: A source row as published to `sources`.
            status_filter: One of the `_STATUS_OPTIONS` values.

        Returns:
            True when the source's status belongs to that filter bucket.
        """
        status = SourcesPane.source_status_text(source).lower()
        if status_filter == "error":
            # `paused` belongs in the Error bucket too (task-2050 review):
            # an auto-paused source is one that failed PAST the threshold --
            # the most broken feed there is -- and its `status_summary` now
            # reads "paused" rather than "error (N)" (paused wins the
            # precedence). A user filtering to Error to triage broken feeds
            # must not silently miss exactly those. The dedicated Paused
            # bucket below narrows further when wanted.
            return status.startswith("error") or status == "paused"
        if status_filter == "ok":
            # `active` is what a healthy local source reports; `ok` is the
            # hand-built shape used by this pane's own tests.
            return status in ("ok", "active")
        return status == status_filter

    def _filtered_sources(self) -> list[dict[str, Any]]:
        query = self.search_query.strip().lower()
        type_filter = self.source_type_filter
        status_filter = self.status_filter
        active_filter = self.active_filter
        tags_filter = self.tags_filter
        required_tags = [tag.strip().lower() for tag in tags_filter.split(",") if tag.strip()] if tags_filter else []
        results: list[dict[str, Any]] = []
        for source in self.sources:
            if type_filter != "all" and str(source.get("source_type") or "").lower() != type_filter:
                continue
            if status_filter != "all" and not self._matches_status_filter(
                source, status_filter
            ):
                continue
            if active_filter == "active" and not source.get("active"):
                continue
            if active_filter == "inactive" and source.get("active"):
                continue
            if required_tags:
                source_tags = {str(tag).lower() for tag in (source.get("tags") or [])}
                if not any(tag in source_tags for tag in required_tags):
                    continue
            if query:
                text = " ".join(
                    str(source.get(key) or "") for key in ("name", "title", "url", "source_type", "status")
                ).lower()
                if query not in text:
                    continue
            results.append(source)
        return results

    # task-15460: one watcher per filter, all doing the same in-place
    # re-populate. Written out rather than shared through a `watch` alias so
    # each reactive's name still appears where a reader looks for it.
    def watch_search_query(self, search_query: str) -> None:
        self._refresh_table_rows()

    def watch_source_type_filter(self, source_type_filter: str) -> None:
        self._refresh_table_rows()

    def watch_status_filter(self, status_filter: str) -> None:
        self._refresh_table_rows()

    def watch_active_filter(self, active_filter: str) -> None:
        self._refresh_table_rows()

    def watch_tags_filter(self, tags_filter: str) -> None:
        self._refresh_table_rows()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "sources-search-input":
            self.search_query = event.value
        elif event.input.id == "sources-tags-filter":
            self.tags_filter = event.value
        elif event.input.id == "sources-create-name":
            self.create_draft_name = event.value
            self._post_create_draft_changed()
        elif event.input.id == "sources-create-url":
            self.create_draft_url = event.value
            self._post_create_draft_changed()
        elif event.input.id == "sources-create-tags":
            self.create_draft_tags = event.value
            self._post_create_draft_changed()
        event.stop()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Mirror the noise field's text into the draft (TASK-1362).

        Same contract as the three `Input`s above: without this the text is
        pane-local, and the next workbench rebuild re-seeds the field from
        `create_draft_ignore_selectors`'s default -- silently restoring rules
        the user had just deleted.

        Only this field's own event is consumed. A blanket `event.stop()`
        would swallow the `Changed` of any `TextArea` a future descendant of
        this pane adds, leaving its owner with no signal and no clue why.
        """
        if event.text_area.id == "sources-create-ignore-selectors":
            self.create_draft_ignore_selectors = event.text_area.text
            self._post_create_draft_changed()
            event.stop()

    def _post_create_draft_changed(self) -> None:
        self.post_message(
            CreateFormDraftChanged(
                name=self.create_draft_name,
                url=self.create_draft_url,
                tags=self.create_draft_tags,
                ignore_selectors=self.create_draft_ignore_selectors,
                source_type=self.create_draft_source_type,
                destination=self.create_draft_destination,
            )
        )

    def _clear_create_draft(self) -> None:
        """Reset the draft, e.g. after a successful submit or Cancel."""
        self.create_draft_name = ""
        self.create_draft_url = ""
        self.create_draft_tags = ""
        # Back to the pane's defaults, not to whatever was just submitted:
        # the next form is a NEW source, which starts at the scope the user
        # is looking at (TASK-2302) and at the default feed type.
        self.create_draft_source_type = "rss"
        self.create_draft_destination = self.default_destination
        # Back to the shipped default, not to empty: the next time the form
        # opens it is a *new* source, which gets the prefill again. Only a
        # user emptying the field for a source they are creating right now
        # means "this one watches everything" (spec §2).
        self.create_draft_ignore_selectors = default_ignore_selectors_text()
        self._post_create_draft_changed()

    def watch_show_create_form(self, is_open: bool) -> None:
        """Tell the owning screen the create form opened or closed.

        Mirrors `show_create_form` into a `CreateFormVisibilityChanged`
        message so the screen can persist it across a workbench rebuild —
        see that message's docstring for why this pane cannot just rely on
        its own reactive surviving a recompose.

        Also arms the first field to take focus once the recompose this
        assignment triggers has finished mounting it (TASK-1035) — the
        `is_mounted` guard keeps that to a genuinely interactive open, so a
        pane constructed pre-mount by `_build_detail_pane` (re-seeding an
        already-open form after a workbench rebuild) does not later yank
        focus away from wherever the user actually left it.

        Args:
            is_open: The form's new visibility.
        """
        if self.is_mounted:
            if is_open:
                self._pending_create_focus = self._CREATE_FORM_FIELD_IDS[0]
                # TASK-2302 AC#1: opening the form aims it at the scope the
                # user is looking at right now. Gated on `is_mounted` for the
                # same reason the focus arm above is: `_build_detail_pane`
                # re-opens an already-open form on every workbench rebuild,
                # and resetting the destination there would throw away a
                # choice the user had already made.
                self.create_draft_destination = self.default_destination
            self.post_message(CreateFormVisibilityChanged(is_open))

    def _focused_create_field_id(self) -> str | None:
        """Id of this pane's create-form control that currently has focus."""
        if not self.show_create_form or not self.is_mounted:
            return None
        try:
            focused = self.screen.focused
        except Exception:
            return None
        if focused is None or focused.id not in self._CREATE_FORM_FIELD_IDS:
            return None
        try:
            return focused.id if self in focused.ancestors_with_self else None
        except Exception:
            return None

    async def recompose(self) -> None:
        """Re-home focus into the create form after this pane rebuilds.

        TASK-1035. `show_create_form` is `reactive(..., recompose=True)`, and
        `Widget.recompose` removes and remounts *every* child of this pane.
        Textual does not move focus when the focused widget is removed that
        way, so pressing `New source` — a `Button` inside this very pane —
        destroyed the widget holding focus and left `Screen.focused` at
        `None`. The form then opened with nothing focused anywhere on the
        screen: typing went to the void, and `Tab` restarted at the head of
        the screen's focus chain (the top navigation bar), 37 presses away
        from the first form field. That is what the 2026-07-28 UAT reported
        as "the create-source form cannot be filled in".

        TASK-1345: `.focus()` below only *schedules* the actual focus change
        (`Widget.focus` posts to `app.call_later`) -- it has not landed by
        the time this method returns. The previous version read and cleared
        `_pending_create_focus` *before* that schedule fired, so a SECOND
        `recompose=True` assignment landing in that gap (`sources` reloading
        from `_load_sources`, a filter changing) remounted the form's fields
        out from under the still-pending callback: it then fired on a
        **detached** widget and was silently dropped, and because the intent
        was already `None`, the interleaving recompose had nothing to
        re-apply either (`_focused_create_field_id()` also reports `None` --
        focus never actually landed). The armed intent was lost to the
        interleave.

        The fix keeps `_pending_create_focus` STICKY -- armed across this
        method -- and only clears it once `_confirm_create_focus` (scheduled
        below) observes that focus has actually landed on the target.
        Whichever recompose in a burst runs LAST therefore wins: each one
        re-applies `.focus()` against whatever is currently mounted, and the
        intent survives until one of them is actually confirmed.

        task-3071 adds a third case ahead of these, checked first: the
        search box (`#sources-search-input`) held focus when the teardown
        started. It used to be that `search_query` was `reactive(...,
        recompose=True)`, so every keystroke rebuilt this pane and destroyed
        the focused input mid-word -- only the first character of a search
        ever landed (the exact bug `ItemsPane.recompose` fixed in
        task-2513). task-15460 removed that trigger (the filters are plain
        reactives now), but the case remains live for every OTHER rebuild
        that can land while the search box is focused -- a `sources` reload,
        a region collapse, the create form opening. Search focus is restored
        to the fresh input and the create-form path below is skipped for
        that rebuild, so a still-armed `_pending_create_focus` cannot yank
        the caret out of the box.

        Two create-form cases are handled, and only these two — focus is
        never taken from anywhere outside this pane's own create form or
        the search box:

        1. The form just opened: `watch_show_create_form` armed
           `_pending_create_focus` with the first field, matching
           `WatchlistNameDialog.on_mount`, which focuses its input.
        2. The form was already open with one of its fields focused, and
           something *else* rebuilt this pane underneath it — a `sources`
           reload, a region collapse. The draft text already survives that
           (`CreateFormDraftChanged`); this puts the caret back with it.
           `_focused_create_field_id()` reports the user's real, CURRENT
           in-form focus and is checked *first* (Qodo, TASK-1345 follow-up)
           precisely so this case wins even while a stale
           `_pending_create_focus` from case 1 is still armed — a user who
           Tabs away during the confirm window is not yanked back to field 0
           by a later, unrelated rebuild.
        """
        # Order matters (Qodo, TASK-1345 follow-up): the user's CURRENT
        # in-form focus wins over the sticky intent, not the other way
        # round. Three cases, all covered by `_focused_create_field_id`'s
        # own guard (it returns `None` unless `screen.focused` is one of
        # this form's fields):
        #   * Mid-burst (case 1, between `.focus()` scheduling and it
        #     landing): `screen.focused` is still `None` (or the widget
        #     that held focus before the form opened), so
        #     `_focused_create_field_id()` reports `None` and the sticky
        #     `_pending_create_focus` correctly wins, carrying the intent
        #     through the drop.
        #   * The user has since Tabbed to another in-form field while the
        #     confirm callback is still armed for a different target: that
        #     field now wins over the stale intent, so an unrelated
        #     recompose landing in that window cannot yank them back to it.
        #   * First open, before the burst's own `.focus()` has run:
        #     `screen.focused` is still the `New source` button, which is
        #     not one of `_CREATE_FORM_FIELD_IDS`, so
        #     `_focused_create_field_id()` is `None` and
        #     `_pending_create_focus` (armed to field 0 by
        #     `watch_show_create_form`) still supplies the target.
        # task-3071: the search box is this pane's OTHER focusable a
        # recompose destroys -- no longer once per keystroke (task-15460),
        # but still on every rebuild that can land while the user is typing
        # -- so capture it alongside the create-form cases.
        # `self.screen.focused`,
        # NOT `self.app.focused`, for the same ScreenStackError reason
        # `ItemsPane.recompose` documents.
        try:
            focused = self.screen.focused if self.is_mounted else None
        except Exception:
            focused = None
        search_had_focus = (
            focused is not None and focused.id == "sources-search-input"
        )
        # Capture whether the create form is mounted NOW, while the old DOM
        # still exists: after the rebuild, `show_create_form` alone cannot
        # distinguish "the form is OPENING with this recompose" (its
        # focus-on-open must win over the search box) from "the form was
        # already open" (an in-flight keystroke's focus must win).
        # Qodo, PR #1418.
        form_was_open = bool(self.query("#sources-create-form"))
        restore = self._focused_create_field_id() or self._pending_create_focus
        await super().recompose()
        # Guard explicitly after the await: the pane can be torn down while
        # `super().recompose()` is in flight (a section switch, a region
        # collapse), matching how `lab_frame.py` bails out of its own
        # post-recompose work.
        if not self.is_mounted or not self.is_running:
            return
        if not self.show_create_form:
            # The form closed (Cancel/Create) while this recompose was in
            # flight. Any focus intent still armed is for a form that no
            # longer exists -- drop it so it cannot resurface later and
            # steal focus for an unrelated rebuild (TASK-1345). This runs
            # BEFORE the search branch so a closed form's stale intent is
            # cleared even when the search box keeps focus (Qodo, PR #1418).
            self._pending_create_focus = None
            if search_had_focus:
                self.call_after_refresh(self._restore_search_focus)
            return
        if search_had_focus and form_was_open:
            # Typing mid-search must keep the box: refocus the fresh input
            # and never let a still-armed create-form intent yank the caret
            # away mid-keystroke -- the same "the user's CURRENT focus wins
            # over a stale intent" ordering the TASK-1345 follow-up pinned
            # for in-form fields. Gated on `form_was_open`: when THIS
            # recompose is the one mounting the form, focus-on-open wins
            # instead (falls through to the restore path below), so the
            # form's auto-focus-first-field behavior survives the screen's
            # deferred open firing while the search box happens to be
            # focused (Qodo, PR #1418).
            self.call_after_refresh(self._restore_search_focus)
            return
        if not restore:
            return
        try:
            self.query_one(f"#{restore}").focus()
        except NoMatches:
            # The only *expected* miss: the form closed, or its fields were
            # rebuilt under a different id, between arming `restore` and
            # getting here. Debug is right for that -- it is not a fault.
            logger.opt(exception=True).debug(
                f"SourcesPane: #{restore} was gone after recompose; "
                "nothing to focus."
            )
            return
        except Exception:
            # Anything else is a real fault in the focus path, and this pane
            # exists in its current form *because* focus silently going
            # missing after a recompose shipped to users once already
            # (TASK-1035). Logging that at debug would hide the next
            # occurrence behind "focus is randomly missing" with no signal,
            # so it is warned. Still swallowed: a broken focus restore must
            # not take the screen down with it.
            logger.opt(exception=True).warning(
                f"SourcesPane: unexpected failure focusing #{restore} after "
                "recompose; the create form may open with nothing focused."
            )
            return
        # Keep the intent armed (TASK-1345) and confirm once it has actually
        # landed -- see `_confirm_create_focus` and the docstring above.
        self._pending_create_focus = restore
        self.call_after_refresh(self._confirm_create_focus, restore)

    def _restore_search_focus(self) -> None:
        """Focus the freshly recomposed search input, caret at end of query.

        The search half of `recompose()`'s focus preservation (task-3071),
        mirroring `ItemsPane._restore_search_focus`: the input this runs
        against is always the LIVE replacement mounted by
        `super().recompose()`, so a missing match just means the pane was
        torn down in between -- bail quietly.
        """
        try:
            search = self.query_one("#sources-search-input", Input)
        except NoMatches:
            return
        search.focus()
        search.cursor_position = len(search.value)

    def _confirm_create_focus(self, target: str, attempts: int = 0) -> None:
        """Clear the sticky create-focus intent once landed -- or once the
        form is usable some other way.

        TASK-1345. See `recompose`: `.focus()` only *schedules* the focus
        change, so this is the seam that turns "we asked for focus" into
        "focus actually arrived". Scheduled via `call_after_refresh`, which
        runs after the next screen refresh -- if `.focus()`'s own scheduled
        callback has not fired yet by then, this defers to the refresh after
        THAT one, and so on, until it either confirms, is superseded, or the
        form is usable some other way (below).

        Two outcomes once focus is observed (Qodo, TASK-1345 follow-up -- the
        previous version cleared the intent ONLY when focus was the exact
        `target`, so it either never cleared or rescheduled forever whenever
        focus landed anywhere else):

        1. `screen.focused` is any real widget -- an in-form field (the
           original `target` OR a sibling the user Tabbed to), or something
           outside the form (they clicked away). Either way the mid-burst
           drop is over and the intent's job is done, so it clears without
           re-pulling focus. Clearing on ANY landed focus, not only `target`,
           is what lets `recompose`'s focused-first ordering keep the user
           where they are instead of a stale intent yanking them back.
        2. `screen.focused` is still `None` -- the genuine mid-burst drop
           this whole mechanism exists for. Reschedule, up to
           `_CREATE_FOCUS_CONFIRM_MAX_ATTEMPTS` (20) refreshes. That ceiling
           is safe: every real interleave this pane has to tolerate resolves
           within a handful of refreshes (Textual's own internal
           `call_later`/`call_after_refresh` scheduling, not user-paced
           input), so 20 sits nowhere near a burst any production interleave
           produces. It only ever fires when focus is genuinely never going
           to land -- at which point the form remains fully usable via click
           or Tab, and an unbounded reschedule would just be silent,
           permanent per-refresh work with the intent stuck armed forever.

        Guarded on identity (`self._pending_create_focus != target`): if a
        later recompose has since armed a *different* target, that
        recompose's own confirmation owns clearing the intent, and this
        stale one must not clear a newer arm out from under it.
        """
        if self._pending_create_focus != target:
            return
        if not self.is_mounted or not self.is_running or not self.show_create_form:
            # Torn down or closed while this confirmation was pending --
            # the intent is stale either way.
            self._pending_create_focus = None
            return
        try:
            focused = self.screen.focused
        except Exception:
            focused = None
        if focused is not None:
            # Focus reached something real -- an in-form field (the original
            # `target`, OR a sibling the user Tabbed to while this was still
            # pending) OR a widget outside the form (they clicked away).
            # Either way the mid-burst drop this mechanism exists for is over
            # and the intent's job is done: clear it so `recompose`'s
            # focused-first ordering keeps the user wherever they actually
            # are, and never re-pull focus back. (Qodo, TASK-1345 follow-up:
            # the previous version cleared ONLY on the exact `target`, so a
            # stale intent could either reschedule forever or, via the old
            # `pending`-first `recompose` ordering, yank the user off a field
            # they had moved to.)
            self._pending_create_focus = None
            return
        if attempts >= self._CREATE_FOCUS_CONFIRM_MAX_ATTEMPTS:
            # Outcome 3, bound reached: give up rather than reschedule
            # forever with the intent stuck armed. Type-only: no field
            # values, just the target control id and the attempt count.
            logger.debug(
                f"SourcesPane: focus for #{target} never confirmed after "
                f"{attempts} refreshes; giving up -- the form is still "
                "usable via click or Tab."
            )
            self._pending_create_focus = None
            return
        self.call_after_refresh(self._confirm_create_focus, target, attempts + 1)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "sources-type-select":
            self.source_type_filter = str(event.value or "all")
        elif event.select.id == "sources-status-filter":
            self.status_filter = str(event.value or "all")
        elif event.select.id == "sources-active-filter":
            self.active_filter = str(event.value or "all")
        elif event.select.id == "sources-create-type":
            # TASK-2302. `recompose=True`, so this rebuilds the form to add
            # or drop the ignore-selectors field. Every other draft survives
            # that (they are seeded from these same reactives) and
            # `recompose` re-homes focus, which for this control lands back
            # on the Select the user just used.
            self.create_draft_source_type = str(event.value or "rss")
            self._post_create_draft_changed()
        elif event.select.id == "sources-create-watchlist":
            self.create_draft_destination = event.value
            self._post_create_draft_changed()
        event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "sources-new-button":
            self.show_create_form = True
        elif button_id == "sources-filter-toggle":
            self.show_filter_editor = not self.show_filter_editor
        elif button_id == "sources-create-cancel":
            self.show_create_form = False
            self._clear_create_draft()
        elif button_id == "sources-create-submit":
            self._submit_create_form()
        elif button_id == "sources-preview-button" and self.selected_source is not None:
            self.post_message(PreviewRequested(self.selected_source))
        elif button_id == "sources-check-now-button" and self.selected_source is not None:
            self.post_message(CheckNowRequested(self.selected_source))
        elif button_id == "sources-import-opml-button":
            self.post_message(ImportOpmlRequested())
        elif button_id == "sources-export-opml-button":
            self.post_message(ExportOpmlRequested())
        event.stop()

    def _submit_create_form(self) -> None:
        name = sanitize_string(self.query_one("#sources-create-name", Input).value.strip(), max_length=255)
        url = sanitize_string(self.query_one("#sources-create-url", Input).value.strip(), max_length=2000)
        if not name:
            self.app.notify("Source name is required.", severity="error")
            return
        if not validate_text_input(name, max_length=255):
            self.app.notify("Source name contains invalid characters or is too long.", severity="error")
            return
        if not url:
            self.app.notify("Source URL is required.", severity="error")
            return
        if not validate_url(url):
            self.app.notify("Source URL must be a valid http(s) URL.", severity="error")
            return
        source_type = str(self.query_one("#sources-create-type", Select).value or "rss")
        active = self.query_one("#sources-create-active", Switch).value
        tags_text = sanitize_string(self.query_one("#sources-create-tags", Input).value.strip(), max_length=1000)
        raw_tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()] if tags_text else []
        tags: list[str] = []
        for tag in raw_tags:
            clean = sanitize_string(tag, max_length=100)
            if clean and validate_text_input(clean, max_length=100):
                tags.append(clean)
            else:
                self.app.notify(f"Tag '{tag}' was skipped due to invalid content.", severity="warning")
        try:
            check_frequency = int(
                self.query_one("#sources-create-frequency", Select).value
            )
        except (TypeError, ValueError):
            check_frequency = self._DEFAULT_FREQUENCY_SECONDS
        # Whatever the field holds, verbatim apart from outer whitespace and
        # control characters (TASK-1362). Not re-split, not reformatted: a
        # comma inside a line is a CSS selector group, so splitting on commas
        # would break `:is(.a, .b)`. Empty means empty -- a user who cleared
        # the field is saying "watch everything on this page", and re-filling
        # the default here would overrule them silently.
        # Read through the DOM, not through the draft, and only when the
        # field is actually on screen: TASK-2302 renders it for url-family
        # types alone, and a feed source carries no selectors by definition.
        # `_clear_create_draft` keeps the draft prefilled for the next form,
        # so reading the draft here would file the shipped default against
        # every RSS source ever created from this form.
        if self.query("#sources-create-ignore-selectors"):
            ignore_selectors = sanitize_string(
                self.query_one("#sources-create-ignore-selectors", TextArea).text,
                max_length=self._IGNORE_SELECTORS_MAX_LENGTH,
            ).strip()
        else:
            ignore_selectors = ""
        # Refuse a selector CSS cannot parse, here, while the text is still on
        # screen and the user can see which line. `ContentExtractor` now skips
        # a bad line rather than aborting the check, but a silently-skipped
        # rule is still a rule the user believes is suppressing noise and that
        # is doing nothing -- and nothing else in the product would ever tell
        # them. Only NON-EMPTY lines are checked, so the cleared field above
        # stays a valid instruction.
        bad_selector = first_invalid_selector(ignore_selectors)
        if bad_selector is not None:
            # markup=False: selectors are full of `[`, which Textual's toast
            # markup would otherwise eat or choke on -- `[class*="ad"]` must
            # reach the user verbatim, since naming the line IS the message.
            self.app.notify(
                invalid_selector_message(bad_selector),
                severity="error",
                markup=False,
            )
            return
        # TASK-2302 AC#1/#2: whatever the destination Select is SHOWING is
        # what travels with the request. Read off the mounted control rather
        # than off `create_draft_destination` so the payload cannot disagree
        # with the row the user is looking at.
        try:
            destination = self.query_one("#sources-create-watchlist", Select).value
        except Exception:
            destination = self.UNASSIGNED_DESTINATION
        watchlist_id = (
            None if destination == self.UNASSIGNED_DESTINATION else int(destination)
        )
        self.post_message(
            CreateSourceRequested(
                {
                    "name": name,
                    "url": url,
                    "source_type": source_type,
                    "active": active,
                    "tags": tags,
                    "check_frequency": check_frequency,
                    "ignore_selectors": ignore_selectors,
                    # `None` means Unassigned, which is a real destination
                    # and not a missing one -- the pane always supplies this
                    # key.
                    "watchlist_id": watchlist_id,
                }
            )
        )
        # ``show_create_form`` is ``recompose=True``. Closing it immediately
        # can tear this pane down before the queued CreateSourceRequested
        # reaches the owning screen, silently dropping a valid submission.
        # Queue the destructive UI reset behind the request instead.
        if not self.call_later(self._finish_create_submit):
            self._finish_create_submit()

    def _finish_create_submit(self) -> None:
        """Close and clear the form after its submit request has propagated."""
        self.show_create_form = False
        self._clear_create_draft()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.select_source_by_id(str(event.row_key.value))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        event.stop()
        self.select_source_by_id(str(event.cell_key.row_key.value))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Select on cursor movement, which is what a mouse click produces.

        TASK-1100. `RowSelected`/`CellSelected` fire on *activation* — Enter,
        or a second click — not when a click merely moves the cursor onto a
        row. So clicking a source did not select it: `selected_source` stayed
        `None`, `Preview` and `Check now` stayed disabled, and pressing
        `Check now` returned silently because `handle_check_now_requested`
        early-returns on `entity is None`.

        That is why "Check now" appeared to do nothing at all in the
        2026-07-28 live UAT — verified against real feeds, zero runs and zero
        items, with the scrape backend itself working perfectly when driven
        directly.
        """
        event.stop()
        if event.row_key is not None and event.row_key.value is not None:
            self.select_source_by_id(str(event.row_key.value))

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Same, for a table whose cursor is cell-shaped rather than row-shaped."""
        event.stop()
        row_key = getattr(event.cell_key, "row_key", None)
        if row_key is not None and row_key.value is not None:
            self.select_source_by_id(str(row_key.value))

    def select_source_by_id(self, source_id: str) -> None:
        """Select the source with the given id and notify listeners."""
        source = None
        for candidate in self.sources:
            if str(candidate.get("id") or "") == source_id:
                source = candidate
                break
        self.selected_source = source

    def watch_selected_source(self, source: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(SourceSelected(source))
        self._update_action_buttons()
        self._update_selection_highlight(source)

    def watch_busy_source_ids(self, _busy_source_ids: frozenset[str]) -> None:
        """Repaint Check now, without a recompose, when a check starts or ends.

        TASK-2309. Mirrors `watch_selected_source`'s repaint-not-rebuild
        choice above, for the identical reason.

        Args:
            _busy_source_ids: The reactive's new value (Textual's watcher
                convention passes it), unused directly -- `_update_action_
                buttons`/`_is_check_now_busy` re-read the current value off
                `self.busy_source_ids` instead, the same indirection
                `watch_selected_source` already uses for its own reactive.
        """
        self._update_action_buttons()

    def _is_check_now_busy(self, source: dict[str, Any] | None) -> bool:
        """Whether `source` (typically `self.selected_source`) has a check
        in flight right now, per the screen's `busy_source_ids` mirror."""
        if source is None:
            return False
        source_key = str(source.get("id") or "")
        return bool(source_key) and source_key in self.busy_source_ids

    def _update_selection_highlight(self, source: dict[str, Any] | None) -> None:
        """Move the table's selected-row highlight without rebuilding it.

        Mirrors `_update_action_buttons` immediately below: `selected_source`
        is deliberately not `recompose=True` (a selection must not rebuild
        the table under the user -- it would discard scroll position and
        the DataTable's own cursor), so a bare reactive assignment leaves
        `compose()`'s row styling stale. This targets exactly the two rows
        that changed -- the previous highlight (if any, reverted) and the
        new one (if present in the currently-filtered table) -- via
        `DataTable.update_cell`, the same "surgical update, not a rebuild"
        approach `_update_action_buttons` already takes for the two action
        buttons.
        """
        new_key = str(source.get("id")) if source else None
        old_key = self._highlighted_source_key
        if new_key == old_key:
            return
        try:
            table = self.query_one("#sources-table", DataTable)
        except Exception:
            self._highlighted_source_key = new_key
            return
        try:
            column_keys = list(table.columns.keys())
        except Exception:
            column_keys = []
        for row_key, highlighted in ((old_key, False), (new_key, True)):
            if row_key is None:
                continue
            candidate = next(
                (s for s in self.sources if str(s.get("id") or "") == row_key), None
            )
            if candidate is None:
                continue
            cells = self._source_row_cells(candidate, highlighted)
            for column_key, value in zip(column_keys, cells):
                try:
                    table.update_cell(row_key, column_key, value, update_width=False)
                except Exception:
                    pass
        self._highlighted_source_key = new_key

    def _update_action_buttons(self) -> None:
        """Keep Preview/Check-now in step with this pane's own selection.

        `selected_source` is deliberately not `recompose=True` (a selection
        change must not rebuild the table under the user), so the `disabled=`
        values `compose()` baked in never move on their own -- exactly the
        reason `RunsPane` already carries a method of this name for Cancel
        and Re-run. Without it the two buttons keep whatever state the last
        *recompose* happened to leave them in: armed against a source the
        screen has since deselected (whole-branch review, Finding 4), or
        greyed out over a source the user just clicked.
        """
        try:
            preview_button = self.query_one("#sources-preview-button", Button)
            check_now_button = self.query_one("#sources-check-now-button", Button)
        except Exception:
            return
        disabled = self.selected_source is None
        preview_button.disabled = disabled
        # TASK-2309: Check now additionally disables, and relabels, while the
        # selected source has a check in flight -- the busy state a second,
        # confused click must see rather than silently queuing a duplicate
        # run. `Preview` is unaffected: it makes no write and a check running
        # is no reason to block reading the same feed.
        check_now_busy = self._is_check_now_busy(self.selected_source)
        check_now_button.disabled = disabled or check_now_busy
        check_now_button.label = "Checking..." if check_now_busy else "Check now"
