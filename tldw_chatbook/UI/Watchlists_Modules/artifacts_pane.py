"""Artifacts pane: the briefings a watchlist has produced, and their bodies.

Spec #2 phase 1 (`Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md`,
§UI). Tasks 1-3 built the tables, the selection and the generation service;
this is the surface that makes any of it visible. It shows one row per
briefing -- including the ones that failed and the ones whose window was
empty, because a status IS the observability here (spec §Error-handling
ethos: silence is never a state) -- and renders the selected briefing's body.

Two conventions are load-bearing and copied deliberately from the sibling
panes rather than reinvented:

* `RecomposeCaptureGuard` ahead of the concrete container, because
  `briefings`/`selected_briefing` are `reactive(..., recompose=True)` on a
  non-screen widget (task-627/637 -- see `recompose_capture_guard.py`).
* `highlight_is_user_driven` (i.e. `table.has_focus`) on every
  `RowHighlighted`/`CellHighlighted`, because a freshly recomposed
  `DataTable` announces a row-0 highlight of its own accord, and forwarding
  that to selection turns this pane into a feedback loop against its own
  rebuild (TASK-1105, and the 157-selections-from-one-tab-open lesson).

The body is markdown an LLM wrote from remote feed content, so it is
rendered with `hyperlinks=False` -- see `_MARKDOWN_HYPERLINKS` below.
"""

from __future__ import annotations

from typing import Any

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.coordinate import Coordinate
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Select, Static

from ...Subscriptions.briefing_selection import (
    MODE_AUTO,
    MODE_AUTO_FEATURED,
    MODE_CURATED,
)
from ...Subscriptions.briefing_service import (
    STATUS_COMPLETE,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_GENERATING,
)
from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .table_selection import highlight_is_user_driven

# A briefing body is model output written from remote feed/site content, so
# every link in it is doubly untrusted: the source chose the URL and the
# model chose the label. `rich.markdown.Markdown` defaults to
# `hyperlinks=True`, which emits OSC-8 escapes -- an attacker-chosen label
# over a destination the reader cannot see.
#
# This is the same decision, for the same reason, that `content_pane.py`
# records for the reader (`_MARKDOWN_HYPERLINKS` there; PR #1091 review F3 /
# TASK-1348 AC#2). Sanitizing or allow-listing URLs was rejected there and is
# rejected here for the same reasons: it means owning a URL policy inside a
# renderer, it fails open on whatever the list did not anticipate, and even a
# perfectly-filtered `https://` link still hides its destination behind a
# label someone else wrote. With `hyperlinks=False` the label renders and the
# URL renders beside it as ordinary visible text, so the user judges the
# destination they can actually read.
_MARKDOWN_HYPERLINKS = False

#: What an `empty` row means, in the user's terms rather than the schema's.
_EMPTY_COPY = (
    "The window was empty: nothing new arrived for this watchlist since the "
    "last briefing, so there was nothing to write about."
)

#: What a `generating` row means when it is still on screen.
_GENERATING_COPY = "This briefing is being written now."

#: Shown when a briefing carries no error text but failed anyway -- the
#: service always writes one, so this only covers a hand-edited row.
_UNEXPLAINED_FAILURE = "This briefing failed, but recorded no reason."

_NO_BRIEFINGS = "No briefings yet. Press Generate to write one."
_NO_SELECTION = "Select a briefing to read it."


class BriefingSelected(Message):
    """Posted when the user selects a briefing row."""

    def __init__(self, briefing: dict[str, Any] | None) -> None:
        self.briefing = briefing
        super().__init__()


class GenerateBriefingRequested(Message):
    """Posted when the user asks for a new briefing.

    Carries nothing: the watchlist to brief is the screen's scope, and the
    guard the screen runs before generating (zombie recovery, then the
    one-at-a-time check) is the screen's to hold -- see
    `briefing_service`'s module docstring on why generation does not guard
    itself.
    """


class RefreshBriefingsRequested(Message):
    """Posted when the user asks to re-read the briefing list."""


class BriefingModeChanged(Message):
    """Posted when the user picks a different selection mode.

    Spec #2 phase 2a, Task 4: retires the phase-1 deferral -- until this
    task, `briefing_selection_mode` had a reader (`briefing_service.
    _selection_mode`) but no writer anywhere in the UI, so `auto` and
    `curated` were unreachable. The screen owns the write (`asyncio.
    to_thread(db.set_watchlist_briefing_settings, ...)`); this pane only
    reports the user's pick.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        super().__init__()


class BriefingDefaultPresetChanged(Message):
    """Posted when the user picks a different default preset (or "App
    default", carried as `None`).
    """

    def __init__(self, preset_id: int | None) -> None:
        self.preset_id = preset_id
        super().__init__()


class ManagePresetsRequested(Message):
    """Posted when the user asks to open the preset manager (Task 3's
    `BriefingPresetModal`, via the screen's own `_open_briefing_preset_
    manager`).
    """


#: The selection-mode picker's options, in the order defined by
#: `briefing_selection.VALID_MODES` (the DB's own three-string pact,
#: verbatim -- see `Subscriptions_DB.set_watchlist_briefing_settings`).
_MODE_OPTIONS: list[tuple[str, str]] = [
    ("Auto (window)", MODE_AUTO),
    ("Curated (queue only)", MODE_CURATED),
    ("Auto + featured", MODE_AUTO_FEATURED),
]

#: Label for the "no override" preset choice. Carries the value `None`,
#: which is a REAL option value here (not `Select.NULL`): with
#: `allow_blank=False` and `None` present among the option values passed to
#: `Select`, `None` is a legal, distinct selection, never confused with the
#: widget's own "nothing chosen" sentinel (`Select.NULL`/`NoSelection`),
#: which this picker never uses -- there is always something selected, even
#: when that something means "use the app default".
_APP_DEFAULT_PRESET_LABEL = "App default"


def _status_text(row: dict[str, Any]) -> str:
    """One briefing's status, as a bare lowercase string."""
    return str(row.get("status") or "").strip().lower()


def _window_text(row: dict[str, Any]) -> str:
    """What the briefing says it covers, in one cell.

    Both halves of the coverage line are shown when both exist: the
    timestamp floor a first briefing falls back to (`covers_from_ts`) and
    the item-id watermark every briefing records (`covers_through_item_id`).
    A `failed` row deliberately carries neither -- that is the spec's named
    invariant, not missing data -- so it reads as an em dash rather than as
    a zero.
    """
    parts: list[str] = []
    covers_from = row.get("covers_from_ts")
    if covers_from:
        parts.append(f"since {covers_from}")
    covers_through = row.get("covers_through_item_id")
    if covers_through not in (None, ""):
        parts.append(f"through item {covers_through}")
    return " · ".join(parts) if parts else "—"


class ArtifactsPane(RecomposeCaptureGuard, Vertical):
    """List a watchlist's briefings and render the selected one."""

    #: Same Rich terminal-agnostic "current row" idiom as
    #: `NotificationsPane._SELECTED_ROW_STYLE` -- a `DataTable` cell's `Text`
    #: cannot reference Textual CSS variables the way a widget's styles can.
    _SELECTED_ROW_STYLE = "reverse bold"

    briefings = reactive[list[dict[str, Any]]]([], recompose=True)
    selected_briefing = reactive[dict[str, Any] | None](None, recompose=True)
    #: The scope line, supplied by the screen: which watchlist these
    #: briefings belong to, or the reason there are none to show.
    scope_label = reactive("", recompose=True)
    #: False when no single watchlist is in scope -- briefings are per
    #: watchlist by schema, so there is nothing for Generate to act on.
    can_generate = reactive(False, recompose=True)
    #: The watchlist's stored `briefing_selection_mode` (spec #2 phase 2a,
    #: Task 4). Defaults to the same fallback `briefing_service.
    #: _selection_mode` uses for a NULL/unrecognized column, so a pane that
    #: has not yet heard from the screen shows the same mode generation
    #: would actually use.
    selection_mode = reactive[str](MODE_AUTO_FEATURED, recompose=True)
    #: Every stored `briefing_presets` row, name-ASC (screen-supplied,
    #: watchlist-independent).
    presets = reactive[list[dict[str, Any]]]([], recompose=True)
    #: The watchlist's stored `default_briefing_preset_id`, or `None` for
    #: "use the app default" -- the value `_generate_briefing` passes to
    #: `generate_briefing(..., preset_id=...)`.
    default_preset_id = reactive[int | None](None, recompose=True)

    def _preset_select_options(self) -> list[tuple[str, int | None]]:
        """Options for the default-preset picker: "App default" then every
        loaded preset, name-ASC (already the order `presets` arrives in).

        A `default_preset_id` that names a preset NOT in `presets` (a
        preset deleted after being set as the default, before this pane's
        next reload) gets a synthetic trailing option instead of being
        silently dropped -- the same defensive shape `BriefingPresetModal.
        _select_options_for` uses for a stale `character_card_id`/
        `voice_profile_id` (Task 3). Without it, constructing `Select` with
        `value=self.default_preset_id` would raise `InvalidSelectValueError`
        the moment a stale id was not among the legal option values.
        """
        options: list[tuple[str, int | None]] = [
            (_APP_DEFAULT_PRESET_LABEL, None)
        ]
        known_ids: set[int] = set()
        for preset in self.presets:
            preset_id = preset.get("id")
            if preset_id is None:
                continue
            known_ids.add(preset_id)
            options.append((str(preset.get("name") or f"Preset {preset_id}"), preset_id))
        if self.default_preset_id is not None and self.default_preset_id not in known_ids:
            options.append((f"Preset {self.default_preset_id} (deleted)", self.default_preset_id))
        return options

    def compose(self):
        # `Text`, not a bare `str`: `Static` parses Rich markup by default
        # (`Static(..., markup=True)`), and this line carries a user-authored
        # watchlist name.
        #
        # Measured, not assumed (fix round 1, Minor c): with a bare `str`, a
        # watchlist named `[bold red]Morning [brief` paints as
        # `Morning [brief` -- the tag is SWALLOWED, so the name silently
        # loses characters and the user cannot tell which watchlist the pane
        # is talking about. Textual tolerated the unclosed `[brief` rather
        # than raising, so this is a corruption bug, not a crash bug; the
        # test that pins it asserts the painted characters. Wrapping in
        # `Text` -- which is never re-parsed -- is the whole fix, and it also
        # avoids `escape_markup`, whose backslashes would corrupt every
        # ordinary bracket a real name contains.
        yield Static(
            Text(
                self.scope_label
                or "Briefings are written on this device from the local "
                "watchlist store."
            ),
            id="artifacts-scope-note",
        )
        with Horizontal(id="artifacts-toolbar", classes="destination-filter-strip"):
            # `compact=True` for the reason TASK-995 records for the Sources
            # toolbar: `.destination-filter-strip` is `height: 1`, and a
            # default bordered Button is three rows, so only its top border
            # would paint.
            yield Button(
                "Generate",
                id="artifacts-generate-button",
                variant="primary",
                compact=True,
                disabled=not self.can_generate,
                tooltip=(
                    "Write a new briefing for this watchlist."
                    if self.can_generate
                    else "Select a watchlist in the rail to brief it."
                ),
            )
            yield Button(
                "Refresh",
                id="artifacts-refresh-button",
                compact=True,
                tooltip="Re-read this watchlist's briefings.",
            )

        if self.can_generate:
            # Task 4: the selection-mode and default-preset pickers, plus
            # the entry into Task 3's preset manager. Rendered only when a
            # single watchlist is in scope -- like Generate itself, there is
            # nothing for either picker to act on without one, and unlike
            # Generate (which stays visible-but-disabled to explain itself)
            # a picker with nothing to pick from has no useful disabled
            # state to show.
            with Horizontal(
                id="artifacts-picker-toolbar", classes="destination-filter-strip"
            ):
                yield Select(
                    _MODE_OPTIONS,
                    value=self.selection_mode,
                    id="artifacts-mode-select",
                    allow_blank=False,
                    compact=True,
                    tooltip="Which items go into this watchlist's next briefing.",
                )
                yield Select(
                    self._preset_select_options(),
                    value=self.default_preset_id,
                    id="artifacts-preset-select",
                    allow_blank=False,
                    compact=True,
                    tooltip=(
                        "The preset Generate uses for this watchlist "
                        "(LLM, model, and style notes)."
                    ),
                )
                yield Button(
                    "Presets…",
                    id="artifacts-presets-button",
                    compact=True,
                    tooltip="Create, edit, or delete briefing presets.",
                )

        selected_key = (
            str(self.selected_briefing.get("id")) if self.selected_briefing else None
        )
        table = DataTable(id="artifacts-table")
        table.add_columns(
            "Status", "Window", "Items", "Featured", "Overflow", "Created"
        )
        selected_index: int | None = None
        for index, row in enumerate(self.briefings):
            row_key = str(row.get("id"))
            if row_key == selected_key:
                selected_index = index
            style = self._SELECTED_ROW_STYLE if row_key == selected_key else ""
            table.add_row(
                Text(_status_text(row) or "—", style=style),
                Text(_window_text(row), style=style),
                Text(str(row.get("item_count") or 0), style=style),
                Text(str(row.get("featured_count") or 0), style=style),
                Text(str(row.get("overflow_count") or 0), style=style),
                Text(str(row.get("created_at") or "—"), style=style),
                key=row_key,
            )
        if selected_index is not None:
            # TASK-1105, exactly as `NotificationsPane` documents it: a
            # selection recomposes this pane, so the new table's cursor
            # starts at row 0 and says so. Seeding it from the surviving
            # selection stops that announcement from dragging the selection
            # back to the first row.
            table.cursor_coordinate = Coordinate(selected_index, 0)
        yield table

        yield Static("Briefing detail", classes="pane-title")
        yield Static(self._detail_renderable(), id="artifacts-detail")

    def _detail_renderable(self) -> RenderableType:
        """What the detail area shows for the current selection.

        Every status gets a body of its own: a `failed` briefing shows the
        provider's own message, an `empty` one says the window was empty,
        and a `complete` one renders its markdown. None of them render as a
        blank pane, which is the state this design refuses to have.
        """
        row = self.selected_briefing
        if row is None:
            return Text(_NO_SELECTION if self.briefings else _NO_BRIEFINGS)

        status = _status_text(row)
        header = Text()
        header.append(str(row.get("created_at") or "unknown time"), style="bold")
        header.append(" · ")
        header.append(status or "unknown status")
        model_used = row.get("model_used")
        if model_used:
            header.append(" · ")
            header.append(str(model_used))
        header.append("\n")
        header.append(_window_text(row), style="dim")
        header.append("\n")

        if status == STATUS_COMPLETE:
            body = str(row.get("body_markdown") or "").strip()
            if not body:
                # `complete` with no body cannot be produced by the service
                # (an empty provider response is recorded as `failed`), so
                # say what is true rather than painting nothing.
                return Group(header, Text("This briefing recorded no body."))
            # `Markdown` is a block renderable, so it is grouped with the
            # header rather than appended into it. It does not evaluate Rich
            # markup either -- it parses CommonMark, and `[bold red]x[/]` is
            # merely link-shaped text to it.
            return Group(header, Markdown(body, hyperlinks=_MARKDOWN_HYPERLINKS))
        if status == STATUS_FAILED:
            # The provider's own message, appended to a `Text` -- which never
            # parses Rich markup, so no escaping is needed or wanted here
            # (see `content_pane.render_article` on why "defensive" escaping
            # corrupts ordinary content while protecting nothing).
            return Group(
                header, Text(str(row.get("error") or _UNEXPLAINED_FAILURE))
            )
        if status == STATUS_EMPTY:
            return Group(header, Text(_EMPTY_COPY))
        if status == STATUS_GENERATING:
            return Group(header, Text(_GENERATING_COPY))
        return Group(header, Text(f"Unrecognised briefing status: {status or '—'}"))

    def select_briefing_by_id(self, briefing_id: str) -> None:
        """Select one visible briefing by its row id."""
        self.selected_briefing = next(
            (
                row
                for row in self.briefings
                if str(row.get("id")) == str(briefing_id)
            ),
            None,
        )

    def watch_selected_briefing(self, briefing: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(BriefingSelected(briefing))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        if event.row_key is not None and event.row_key.value is not None:
            self.select_briefing_by_id(str(event.row_key.value))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Select on cursor movement, which is what a single click produces.

        Gated on `highlight_is_user_driven` -- see this module's docstring.
        """
        event.stop()
        if not highlight_is_user_driven(event):
            return
        if event.row_key is not None and event.row_key.value is not None:
            self.select_briefing_by_id(str(event.row_key.value))

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Same, for a table whose cursor is cell-shaped rather than row-shaped."""
        event.stop()
        if not highlight_is_user_driven(event):
            return
        row_key = getattr(event.cell_key, "row_key", None)
        if row_key is not None and row_key.value is not None:
            self.select_briefing_by_id(str(row_key.value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "artifacts-generate-button":
            self.post_message(GenerateBriefingRequested())
        elif button_id == "artifacts-refresh-button":
            self.post_message(RefreshBriefingsRequested())
        elif button_id == "artifacts-presets-button":
            self.post_message(ManagePresetsRequested())
        event.stop()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Report a picker change, guarded against Textual's own mount-time
        noise.

        `Select._on_mount` always assigns its `value` reactive from the
        value it was constructed with, and that assignment always posts a
        `Changed` -- including on a completely ordinary, user-uninitiated
        mount (the Library lesson this pane's sibling modal names too, see
        `briefing_preset_modal.py`'s module docstring).

        Comparing the event's value against THIS pane's *current* state is
        not enough to tell the two apart, and shipped as a real bug before
        this fix: the FIRST render of Artifacts builds this Select from
        whatever `selection_mode`/`default_preset_id` happen to be at that
        instant -- the screen's `__init__` defaults, since `_load_briefings`
        has not loaded the real value yet -- and that Select's own
        mount-time `Changed` (carrying the stale default) is posted but not
        necessarily PROCESSED before `_load_briefings` finishes and pushes
        the real value, recomposing this pane with a fresh Select. By the
        time the stale message is finally processed, `self.selection_mode`
        already equals the NEW (correct) value, not the stale one the
        message carries -- so a same-value guard sees `stale != current`
        and wrongly treats mount noise as a real pick, writing the stale
        default back over the value that was just loaded.

        The fix is to key off the WIDGET INSTANCE instead of a value
        comparison: a freshly composed `Select` posts EXACTLY one `Changed`
        from its own mount (see `Select._on_mount` -> `_init_selected_
        option` -> the `value` assignment), so the first one this pane
        sees from a given `Select` object is always that noise, absorbed
        here unconditionally; every one after it is a real user pick,
        since nothing else in this pane ever re-assigns a mounted `Select`'s
        `value` programmatically. A recompose always builds a brand-new
        `Select` object, so this naturally resets per recompose with no
        bookkeeping to clear.
        """
        event.stop()
        select = event.select
        if not getattr(select, "_briefing_picker_mount_absorbed", False):
            select._briefing_picker_mount_absorbed = True
            return
        if select.id == "artifacts-mode-select":
            self.post_message(BriefingModeChanged(str(event.value)))
        elif select.id == "artifacts-preset-select":
            self.post_message(BriefingDefaultPresetChanged(event.value))
