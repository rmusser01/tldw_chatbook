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

import json
from collections.abc import Mapping
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


class CastScriptRequested(Message):
    """Posted when the user asks to cast a script from the selected briefing.

    Carries nothing, same shape as `GenerateBriefingRequested` and for the
    same reason: the briefing to cast is the screen's own selection state
    (its `_selected_briefing`), and the preset to cast with is this pane's
    `default_preset_id` -- both already live on the screen/pane, so the
    message is only a nudge, not a payload. The screen owns the guard
    (one cast in flight at a time, zombie recovery) exactly as it owns
    `_briefing_in_flight`'s guard for Generate -- see
    `handle_cast_script_requested`.
    """


class ScriptSelected(Message):
    """Posted when the user selects a cast-script row."""

    def __init__(self, script: dict[str, Any] | None) -> None:
        self.script = script
        super().__init__()


class CitationActivated(Message):
    """Posted when the user activates a citation under the briefing body.

    Spec #2 phase 2a, Task 6: retires the phase-1 "citations" deferral. A
    briefing body's `[item N]` markers (`briefing_service.build_briefing_
    prompt`'s own convention) are parsed once, when this briefing is
    selected (`WatchlistsCollectionsScreen._load_briefings`, via
    `briefing_service.extract_citation_ids`), and each resolved against
    `SubscriptionsDB.get_subscription_items_by_ids` -- so by the time this
    message posts, the screen already knows whether `item_id` is still a
    live row. Carries only the id, not the row itself: the screen already
    holds the resolution (`_citation_item_lookup`), and this message would
    just be handing back a payload the screen would look up again.
    """

    def __init__(self, item_id: int) -> None:
        self.item_id = item_id
        super().__init__()


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


#: `briefing_cast.py` defines its OWN `STATUS_GENERATING`/`STATUS_COMPLETE`/
#: `STATUS_FAILED`, but as the exact same three strings as `briefing_
#: service`'s (a script has no `empty` status, so it uses only three of the
#: four already imported above). Reusing the briefing constants here rather
#: than importing a second, string-identical set from `briefing_cast` keeps
#: this module's imports to one status vocabulary rather than two names for
#: the same values.
_SCRIPT_NO_SELECTION = "Select a script to read it."
_SCRIPT_NO_SCRIPTS = "No scripts yet. Press Cast to write one."
_SCRIPT_GENERATING_COPY = "This script is being written now."
_SCRIPT_UNEXPLAINED_FAILURE = "This script failed, but recorded no reason."
_SCRIPT_UNREADABLE_TURNS = "This script recorded turns that could not be read."
_SCRIPT_NO_TURNS = "This script recorded no turns."

#: Turn rendering caps here with an honest "…N more turns" line rather than
#: silently truncating -- the same ethos `briefing_service`'s own item/
#: overflow counts already state for a briefing's source material.
_TURN_RENDER_CAP = 200


def _script_status_text(row: dict[str, Any]) -> str:
    """One script's status, as a bare lowercase string."""
    return str(row.get("status") or "").strip().lower()


def _script_turns_renderable(turns_json: str | None) -> Text:
    """A script's turns as speaker-labelled `Text` lines.

    Never a markup parser: the model wrote this text, from watchlist
    content it did not choose either, so it is appended into a `Text`
    exactly like every other model/source-derived field on this pane
    (`_detail_renderable`'s `error`/body handling) -- a turn containing
    literal Rich markup syntax (`[bold red]x[/]`) must paint as those
    characters, not be interpreted or escaped.

    Args:
        turns_json: A script row's `turns_json` column -- a JSON array of
            `{"speaker", "text"}` objects when the script is `complete`,
            per `briefing_cast.parse_script_turns`'s output contract.

    Returns:
        One `Text`, speaker-labelled per line, capped at
        `_TURN_RENDER_CAP` turns with an honest "…N more turns" trailer
        when the script wrote more than that -- never a silent truncation.
    """
    try:
        turns = json.loads(turns_json or "[]")
    except (TypeError, ValueError):
        return Text(_SCRIPT_UNREADABLE_TURNS)
    if not isinstance(turns, list) or not turns:
        return Text(_SCRIPT_NO_TURNS)

    shown = turns[:_TURN_RENDER_CAP]
    text = Text()
    for turn in shown:
        if not isinstance(turn, Mapping):
            continue
        speaker = str(turn.get("speaker") or "?")
        turn_text = str(turn.get("text") or "")
        text.append(speaker, style="bold")
        text.append(": ")
        text.append(turn_text)
        text.append("\n")
    remaining = len(turns) - len(shown)
    if remaining > 0:
        text.append(f"…{remaining} more turns", style="dim")
    return text


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
    #: Task 5: every `briefing_scripts` row cast from the SELECTED briefing
    #: (newest first, per `list_briefing_scripts`) -- never every script
    #: across the whole watchlist, since a script belongs to exactly one
    #: briefing and this pane only ever shows one briefing's detail at a
    #: time.
    scripts = reactive[list[dict[str, Any]]]([], recompose=True)
    #: The script whose detail is rendered below the scripts table, or
    #: `None` when nothing is selected.
    selected_script = reactive[dict[str, Any] | None](None, recompose=True)
    #: Task 6: every `[item N]` id the SELECTED briefing's body cites,
    #: resolved once per selection by the screen (`_load_briefings`, via
    #: `get_subscription_items_by_ids`) -- `{"item_id": int, "label": Text,
    #: "available": bool}` per citation, in the body's own first-cited
    #: order. `available=False` is the honest-degradation case (the plan's
    #: named invariant): the id no longer resolves to a live row -- pruned
    #: or deleted since the briefing was written -- and `label` already
    #: says so ("item N -- no longer available") rather than the pane
    #: having to re-derive that from an absent dict. `label` is always a
    #: `rich.text.Text`, never a bare `str`: an item title is remote text
    #: (the same reasoning `_script_turns_renderable` states for a turn),
    #: so it must never reach a markup parser.
    citations = reactive[list[dict[str, Any]]]([], recompose=True)

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

        if self.citations:
            # Task 6: a small citations table under the body -- one row per
            # `[item N]` the body actually cites, activatable (a click, or
            # arrow-navigating to it -- the same idiom every table on this
            # pane already uses) to jump straight to that item in the
            # reader, or, for an item pruned since this briefing was
            # written, a toast saying so
            # (`WatchlistsCollectionsScreen.handle_citation_activated`).
            # Never a link inside the Markdown body itself -- see
            # `_MARKDOWN_HYPERLINKS` above; this is a separate widget
            # affordance instead, exactly as the plan requires.
            #
            # Rendered only when there is at least one citation to show:
            # every EXISTING test in this file uses a canned body with none
            # (`CANNED_BODY` carries no `[item N]` marker), and an
            # always-present-but-empty table would spend this pane's
            # already-tight row budget (see the `_watchlists.tcss` comment
            # on `#artifacts-table`'s `min-height`) on a case with nothing
            # to offer.
            citations_table = DataTable(id="artifacts-citations-table")
            citations_table.add_columns("Citation", "Status")
            for citation in self.citations:
                available = bool(citation.get("available"))
                citations_table.add_row(
                    citation.get("label") or Text(""),
                    Text("Available" if available else "Not available"),
                    key=str(citation.get("item_id")),
                )
            yield citations_table

        if self.selected_briefing is not None:
            # Task 5: casting a script is an action on THE SELECTED
            # briefing, so -- unlike Generate, which has a watchlist-wide
            # target and stays visible-but-disabled to explain itself --
            # there is nothing for Cast to act on at all without a
            # selection, and this whole section (button, list, detail)
            # renders only once one exists.
            cast_disabled = self.default_preset_id is None and not self.presets
            with Horizontal(
                id="artifacts-scripts-toolbar", classes="destination-filter-strip"
            ):
                yield Button(
                    "Cast",
                    id="artifacts-cast-button",
                    compact=True,
                    disabled=cast_disabled,
                    tooltip=(
                        "Create a briefing preset (Presets…) before casting "
                        "a script."
                        if cast_disabled
                        else "Cast this briefing into a spoken-style script "
                        "using the current default preset."
                    ),
                )

            selected_script_key = (
                str(self.selected_script.get("id"))
                if self.selected_script
                else None
            )
            scripts_table = DataTable(id="artifacts-scripts-table")
            scripts_table.add_columns("Preset", "Status", "Created")
            selected_script_index: int | None = None
            for index, row in enumerate(self.scripts):
                row_key = str(row.get("id"))
                if row_key == selected_script_key:
                    selected_script_index = index
                style = (
                    self._SELECTED_ROW_STYLE
                    if row_key == selected_script_key
                    else ""
                )
                scripts_table.add_row(
                    Text(str(row.get("preset_name") or "—"), style=style),
                    Text(_script_status_text(row) or "—", style=style),
                    Text(str(row.get("created_at") or "—"), style=style),
                    key=row_key,
                )
            if selected_script_index is not None:
                # Same TASK-1105 seeding as the briefings table above.
                scripts_table.cursor_coordinate = Coordinate(
                    selected_script_index, 0
                )
            yield scripts_table

            # No separate `.pane-title` here (unlike "Briefing detail"
            # above): a `.pane-title` costs 4 rows (`height: 3` +
            # `margin-bottom: 1`) inside a region whose total budget is
            # already fixed and now shared with a second list-over-body
            # pair -- measured to matter, not assumed (a first draft with
            # the title pushed `#artifacts-detail` below the height its own
            # test fixture needs). `_script_detail_renderable`'s own header
            # names it as "Script:" instead, for one row instead of four.
            yield Static(self._script_detail_renderable(), id="artifacts-script-detail")

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

    def _script_detail_renderable(self) -> RenderableType:
        """What the detail area shows for the current script selection.

        Mirrors `_detail_renderable`'s "every status gets a body of its
        own" rule: `generating`/`complete`/`failed` (a script has no
        `empty` -- `validate_roster` refuses an empty roster before any row
        exists) each read as an outcome, never a blank pane.
        """
        row = self.selected_script
        if row is None:
            return Text(_SCRIPT_NO_SELECTION if self.scripts else _SCRIPT_NO_SCRIPTS)

        status = _script_status_text(row)
        header = Text()
        # "Script: " labels this block the way the dropped `.pane-title`
        # would have -- see the compose()-site comment on why there is no
        # separate title `Static` here.
        header.append("Script: ", style="dim")
        header.append(str(row.get("preset_name") or "Untitled preset"), style="bold")
        header.append(" · ")
        header.append(status or "unknown status")
        model_used = row.get("model_used")
        if model_used:
            header.append(" · ")
            header.append(str(model_used))
        header.append("\n")
        header.append(str(row.get("created_at") or "unknown time"), style="dim")
        header.append("\n")

        if status == STATUS_COMPLETE:
            return Group(header, _script_turns_renderable(row.get("turns_json")))
        if status == STATUS_FAILED:
            return Group(
                header,
                Text(str(row.get("error") or _SCRIPT_UNEXPLAINED_FAILURE)),
            )
        if status == STATUS_GENERATING:
            return Group(header, Text(_SCRIPT_GENERATING_COPY))
        return Group(header, Text(f"Unrecognised script status: {status or '—'}"))

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

    def select_script_by_id(self, script_id: str) -> None:
        """Select one visible script by its row id."""
        self.selected_script = next(
            (row for row in self.scripts if str(row.get("id")) == str(script_id)),
            None,
        )

    def activate_citation_by_id(self, citation_id: str) -> None:
        """Post `CitationActivated` for the citation whose row key is
        `citation_id`.

        Unlike `select_briefing_by_id`/`select_script_by_id` above, there is
        no reactive to set here: a citation carries no persistent "selected"
        state of its own to render differently once it is current --
        activating one either switches sections or toasts, and either way
        this pane's own state does not change. This is the same directness
        those two methods give a test (a caller does not have to fabricate
        a `DataTable` row-selection event), used identically by the real
        `DataTable` routing below.
        """
        try:
            item_id = int(citation_id)
        except (TypeError, ValueError):
            return
        self.post_message(CitationActivated(item_id))

    def watch_selected_briefing(self, briefing: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(BriefingSelected(briefing))

    def watch_selected_script(self, script: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(ScriptSelected(script))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        if event.row_key is None or event.row_key.value is None:
            return
        if event.data_table.id == "artifacts-citations-table":
            self.activate_citation_by_id(str(event.row_key.value))
        elif event.data_table.id == "artifacts-scripts-table":
            self.select_script_by_id(str(event.row_key.value))
        else:
            self.select_briefing_by_id(str(event.row_key.value))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Select on cursor movement, which is what a single click produces.

        Gated on `highlight_is_user_driven` -- see this module's docstring.
        Routes by which of this pane's THREE tables (briefings, scripts,
        citations) posted the event -- all three are `recompose=True`-
        backed, so all three announce a row-0 highlight on every rebuild,
        and all three need the same gate for the same reason.
        """
        event.stop()
        if not highlight_is_user_driven(event):
            return
        if event.row_key is None or event.row_key.value is None:
            return
        if event.data_table.id == "artifacts-citations-table":
            self.activate_citation_by_id(str(event.row_key.value))
        elif event.data_table.id == "artifacts-scripts-table":
            self.select_script_by_id(str(event.row_key.value))
        else:
            self.select_briefing_by_id(str(event.row_key.value))

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Same, for a table whose cursor is cell-shaped rather than row-shaped."""
        event.stop()
        if not highlight_is_user_driven(event):
            return
        row_key = getattr(event.cell_key, "row_key", None)
        if row_key is None or row_key.value is None:
            return
        if event.data_table.id == "artifacts-citations-table":
            self.activate_citation_by_id(str(row_key.value))
        elif event.data_table.id == "artifacts-scripts-table":
            self.select_script_by_id(str(row_key.value))
        else:
            self.select_briefing_by_id(str(row_key.value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "artifacts-generate-button":
            self.post_message(GenerateBriefingRequested())
        elif button_id == "artifacts-refresh-button":
            self.post_message(RefreshBriefingsRequested())
        elif button_id == "artifacts-presets-button":
            self.post_message(ManagePresetsRequested())
        elif button_id == "artifacts-cast-button":
            self.post_message(CastScriptRequested())
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
