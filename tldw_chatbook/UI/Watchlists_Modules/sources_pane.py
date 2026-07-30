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

from ...Subscriptions.noise_defaults import (
    default_ignore_selectors_text,
    first_invalid_selector,
    invalid_selector_message,
)
from ...Utils.input_validation import sanitize_string, validate_text_input, validate_url
from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
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
    workbench's `region_layout` reactive is `recompose=True` — collapsing or
    expanding *any* region (including one unrelated to Sources, e.g. `[` on
    the left rail) rebuilds the whole workbench and constructs a brand new
    `SourcesPane`. Without this message the screen has no way to know what
    was typed, so the draft would be silently lost on the next such rebuild.
    The owning screen mirrors this into its own state and seeds it back into
    the freshly-constructed pane.
    """

    def __init__(
        self, name: str, url: str, tags: str, ignore_selectors: str | None = None
    ) -> None:
        self.name = name
        self.url = url
        self.tags = tags
        #: The noise-selector text, or None when the pane has nothing to
        #: report (it only ever posts a string, but the screen stores None
        #: for "untouched", so the field is optional for any other caller).
        self.ignore_selectors = ignore_selectors
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

    sources = reactive[list[dict[str, Any]]]([], recompose=True)
    selected_source = reactive[dict[str, Any] | None](None)
    search_query = reactive("", recompose=True)
    source_type_filter = reactive("all", recompose=True)
    status_filter = reactive("all", recompose=True)
    active_filter = reactive("all", recompose=True)
    tags_filter = reactive("", recompose=True)
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

    # Plain attribute, not a reactive: mirrors which row `compose()` last
    # painted as selected, so `_update_selection_highlight` knows which row
    # to revert without re-deriving it from `selected_source`'s OLD value
    # (a one-argument `watch_selected_source` only ever sees the new one).
    _highlighted_source_key: str | None = None

    #: The create form's focusable controls, in visual order (TASK-1035).
    #: `compose()` yields them in this order and the form is a plain
    #: `Vertical`, so DOM order, paint order and Tab order are the same list.
    _CREATE_FORM_FIELD_IDS = (
        "sources-create-name",
        "sources-create-url",
        "sources-create-type",
        "sources-create-active",
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
    #: not a change fire" and has no live UI of its own to explain it in. Kept
    #: to a single added clause -- the field's bottom border is 91 columns
    #: wide at 160x42 (verified: 87 chars is the hard cutoff before Textual's
    #: border-label renderer silently truncates with an ellipsis), so this is
    #: not decorative belt-tightening, it is the actual budget.
    _IGNORE_SELECTORS_LABEL = (
        "Ignore elements (CSS selectors — one rule per line; commas group)"
    )
    _IGNORE_SELECTORS_HELP = (
        "Add a rule to silence noise; changes always report; "
        "change_threshold limits volume."
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

    _TYPE_OPTIONS = [
        ("All", "all"),
        ("RSS", "rss"),
        ("Atom", "atom"),
        ("Feed", "feed"),
        ("Playlist", "playlist"),
        ("Channel", "channel"),
    ]

    _STATUS_OPTIONS = [
        ("All statuses", "all"),
        ("OK", "ok"),
        ("Error", "error"),
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
            # `New Source` -- and a new user had no way to add a source at
            # all. Widths are pinned alongside it in features/_watchlists.tcss
            # (they have to be in the bundle to beat the global
            # `Select { width: 100% }` in features/_conversations.tcss).
            with Horizontal(classes="destination-filter-strip"):
                yield Input(
                    placeholder="Search sources...",
                    id="sources-search-input",
                    value=self.search_query,
                    compact=True,
                )
                yield Select(
                    self._TYPE_OPTIONS,
                    value=self.source_type_filter,
                    id="sources-type-select",
                    allow_blank=False,
                    compact=True,
                )
                yield Select(
                    self._STATUS_OPTIONS,
                    value=self.status_filter,
                    id="sources-status-filter",
                    allow_blank=False,
                    compact=True,
                )
                yield Select(
                    self._ACTIVE_OPTIONS,
                    value=self.active_filter,
                    id="sources-active-filter",
                    allow_blank=False,
                    compact=True,
                )
                yield Button("New Source", id="sources-new-button", variant="primary")
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
                yield Button(
                    "Check now",
                    id="sources-check-now-button",
                    disabled=self.selected_source is None,
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
                    Select(
                        [
                            (label, value)
                            for label, value in self._TYPE_OPTIONS
                            if value != "all"
                        ],
                        value="rss",
                        id="sources-create-type",
                        allow_blank=False,
                    ),
                    Static("Active", classes="sources-create-active-label"),
                    Switch(value=True, id="sources-create-active"),
                    classes="sources-create-type-row",
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
                    Select(
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
                yield self._ignore_selectors_field()
                # `.dialog-buttons` is the same one-row, side-by-side pairing
                # `WatchlistNameDialog` uses for its own Create/Cancel, so the
                # two creation flows read the same (TASK-1035 AC#6). Only the
                # alignment is overridden for this inline form -- see
                # `features/_watchlists.tcss`.
                with Horizontal(classes="dialog-buttons sources-create-buttons"):
                    yield Button("Create", id="sources-create-submit", variant="success")
                    yield Button("Cancel", id="sources-create-cancel", variant="default")

        selected_key = (
            str(self.selected_source.get("id")) if self.selected_source else None
        )
        table = DataTable(id="sources-table")
        table.add_columns("Name", "Type", "Status", "Last scraped", "Active")
        filtered = self._filtered_sources()
        for source in filtered:
            row_key = str(source.get("id") or id(source))
            table.add_row(
                *self._source_row_cells(source, row_key == selected_key),
                key=row_key,
            )
        # `compose()` just painted the highlight fresh from `selected_source`
        # itself, so this is authoritative going forward -- see
        # `_update_selection_highlight`'s docstring for why a later,
        # non-recomposing selection change needs to know what to revert.
        self._highlighted_source_key = selected_key
        yield table

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
        read `-` even immediately after a successful check."""
        return str(
            source.get("last_checked_or_scraped_at")
            or source.get("last_scraped")
            or "-"
        )

    @staticmethod
    def _source_row_cells(source: dict[str, Any], highlighted: bool) -> tuple[Text, ...]:
        """One row's cell values, styled if `highlighted` (task-876).

        Shared between `compose()` (the initial/any-other-reason render) and
        `_update_selection_highlight` (a same-instance selection change,
        which must not rebuild the table) so both draw an identical row.
        """
        style = SourcesPane._SELECTED_ROW_STYLE if highlighted else ""
        return (
            Text(str(source.get("name") or source.get("title") or "Untitled"), style=style),
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
            return status.startswith("error")
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
            )
        )

    def _clear_create_draft(self) -> None:
        """Reset the draft, e.g. after a successful submit or Cancel."""
        self.create_draft_name = ""
        self.create_draft_url = ""
        self.create_draft_tags = ""
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
        way, so pressing `New Source` — a `Button` inside this very pane —
        destroyed the widget holding focus and left `Screen.focused` at
        `None`. The form then opened with nothing focused anywhere on the
        screen: typing went to the void, and `Tab` restarted at the head of
        the screen's focus chain (the top navigation bar), 37 presses away
        from the first form field. That is what the 2026-07-28 UAT reported
        as "the create-source form cannot be filled in".

        Two cases are handled, and only these two — focus is never taken
        from anywhere outside this pane's own create form:

        1. The form just opened: `watch_show_create_form` armed
           `_pending_create_focus` with the first field, matching
           `WatchlistNameDialog.on_mount`, which focuses its input.
        2. The form was already open with one of its fields focused, and
           something *else* rebuilt this pane underneath it — a `sources`
           reload, a region collapse. The draft text already survives that
           (`CreateFormDraftChanged`); this puts the caret back with it.
        """
        restore = self._pending_create_focus
        self._pending_create_focus = None
        if restore is None:
            restore = self._focused_create_field_id()
        await super().recompose()
        # Guard explicitly after the await: the pane can be torn down while
        # `super().recompose()` is in flight (a section switch, a region
        # collapse), matching how `lab_frame.py` bails out of its own
        # post-recompose work.
        if not self.is_mounted or not self.is_running:
            return
        if not restore or not self.show_create_form:
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

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "sources-type-select":
            self.source_type_filter = str(event.value or "all")
        elif event.select.id == "sources-status-filter":
            self.status_filter = str(event.value or "all")
        elif event.select.id == "sources-active-filter":
            self.active_filter = str(event.value or "all")
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
        ignore_selectors = sanitize_string(
            self.query_one("#sources-create-ignore-selectors", TextArea).text,
            max_length=self._IGNORE_SELECTORS_MAX_LENGTH,
        ).strip()
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
                }
            )
        )
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
        check_now_button.disabled = disabled
