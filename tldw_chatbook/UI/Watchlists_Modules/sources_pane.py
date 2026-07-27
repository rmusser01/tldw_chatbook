"""Sources pane for the watchlists screen."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.containers import Grid, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Select, Static, Switch

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

    def __init__(self, name: str, url: str, tags: str) -> None:
        self.name = name
        self.url = url
        self.tags = tags
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

    # Plain attribute, not a reactive: mirrors which row `compose()` last
    # painted as selected, so `_update_selection_highlight` knows which row
    # to revert without re-deriving it from `selected_source`'s OLD value
    # (a one-argument `watch_selected_source` only ever sees the new one).
    _highlighted_source_key: str | None = None

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
            with Horizontal(classes="destination-filter-strip"):
                yield Input(
                    placeholder="Search sources...",
                    id="sources-search-input",
                    value=self.search_query,
                )
                yield Select(
                    self._TYPE_OPTIONS,
                    value=self.source_type_filter,
                    id="sources-type-select",
                    allow_blank=False,
                )
                yield Select(
                    self._STATUS_OPTIONS,
                    value=self.status_filter,
                    id="sources-status-filter",
                    allow_blank=False,
                )
                yield Select(
                    self._ACTIVE_OPTIONS,
                    value=self.active_filter,
                    id="sources-active-filter",
                    allow_blank=False,
                )
                yield Button("New Source", id="sources-new-button", variant="primary")
                yield Button("Filters", id="sources-filter-toggle", variant="default")
            if self.show_filter_editor:
                with Horizontal(id="sources-filter-editor", classes="destination-filter-strip"):
                    yield Input(
                        placeholder="Tags (comma separated)...",
                        id="sources-tags-filter",
                        value=self.tags_filter,
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
            with Grid(id="sources-create-form"):
                yield Input(
                    placeholder="Name", id="sources-create-name", value=self.create_draft_name
                )
                yield Input(
                    placeholder="URL", id="sources-create-url", value=self.create_draft_url
                )
                yield Select(
                    [(label, value) for label, value in self._TYPE_OPTIONS if value != "all"],
                    value="rss",
                    id="sources-create-type",
                    allow_blank=False,
                )
                yield Horizontal(
                    Static("Active"),
                    Switch(value=True, id="sources-create-active"),
                    classes="sources-create-active-row",
                )
                yield Input(
                    placeholder="Tags (comma separated)",
                    id="sources-create-tags",
                    value=self.create_draft_tags,
                )
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
            Text(str(source.get("status") or "-"), style=style),
            Text(str(source.get("last_scraped") or "-"), style=style),
            Text("Yes" if source.get("active") else "No", style=style),
        )

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
            if status_filter != "all" and str(source.get("status") or "").lower() != status_filter:
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

    def _post_create_draft_changed(self) -> None:
        self.post_message(
            CreateFormDraftChanged(
                name=self.create_draft_name,
                url=self.create_draft_url,
                tags=self.create_draft_tags,
            )
        )

    def _clear_create_draft(self) -> None:
        """Reset the draft, e.g. after a successful submit or Cancel."""
        self.create_draft_name = ""
        self.create_draft_url = ""
        self.create_draft_tags = ""
        self._post_create_draft_changed()

    def watch_show_create_form(self, is_open: bool) -> None:
        """Tell the owning screen the create form opened or closed.

        Mirrors `show_create_form` into a `CreateFormVisibilityChanged`
        message so the screen can persist it across a workbench rebuild —
        see that message's docstring for why this pane cannot just rely on
        its own reactive surviving a recompose.

        Args:
            is_open: The form's new visibility.
        """
        if self.is_mounted:
            self.post_message(CreateFormVisibilityChanged(is_open))

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
        self.post_message(
            CreateSourceRequested(
                {
                    "name": name,
                    "url": url,
                    "source_type": source_type,
                    "active": active,
                    "tags": tags,
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
