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

Spec #2 phase 2b, Task 7 (audio): `selected_script` above is `reactive(...,
recompose=True)`, so EVERY script selection rebuilds every widget this
`compose()` yields -- including Play/Stop. That rules out ever holding
"is this row currently playing" as a reactive/attribute on this widget: it
would be silently reset to its default on the very next selection, wrong by
construction. The shared `SimpleAudioPlayer` singleton (`TTS/audio_player.
get_audio_player`) is the only thing that survives a recompose, so it is
also the only thing consulted for playback state -- see
`WatchlistsCollectionsScreen.handle_stop_audio_requested`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.coordinate import Coordinate
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Select, Static

from ...Subscriptions.briefing_audio import audio_file_path_is_safe, briefing_audio_dir
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


class ExportBriefingRequested(Message):
    """Posted when the user asks to export the selected briefing as markdown.

    Carries nothing, same shape as `GenerateBriefingRequested`/`CastScript
    Requested` and for the same reason: the briefing to export is the
    screen's own `_selected_briefing`, already mirrored there by `handle_
    briefing_selected` -- there is nothing this message needs to carry that
    the screen does not already hold.
    """


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


class SynthesizeAudioRequested(Message):
    """Posted when the user asks to synthesize audio for the selected script.

    Carries nothing, same shape as `CastScriptRequested` and for the same
    reason: the script to synthesize is the screen's own selection state
    (`_selected_script`), already mirrored there by `handle_script_
    selected`. The screen owns the guard (one synthesis in flight at a
    time, zombie recovery) exactly as it owns `_cast_in_flight`'s guard for
    Cast -- see `handle_synthesize_audio_requested`.
    """


class PlayAudioRequested(Message):
    """Posted when the user asks to play the selected script's audio.

    Carries nothing: the file to play is the screen's own `_loaded_script_
    audio` state, resolved alongside `_selected_script` inside `_load_
    briefings`. Playback state itself is never held on THIS widget -- see
    the module docstring's own note on `selected_script`'s `recompose=True`
    rebuilding every control on every selection.
    """


class StopAudioRequested(Message):
    """Posted when the user asks to stop the selected script's audio.

    Carries nothing, for the identical reason `PlayAudioRequested` does.
    """


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


#: Spec #2 phase 2b, Task 7. `briefing_audio.py` defines its OWN
#: `STATUS_GENERATING`/`STATUS_COMPLETE`/`STATUS_FAILED`, but -- exactly
#: like `briefing_cast.py`'s own copy above -- as the identical three
#: strings already imported from `briefing_service`. Reused here for the
#: same reason: one status vocabulary, not a third name for the same
#: values.
_AUDIO_NO_AUDIO = "No audio yet. Press Synthesize to generate it."
_AUDIO_GENERATING_COPY = "This audio is being synthesized now."
_AUDIO_UNEXPLAINED_FAILURE = "Audio synthesis failed, but recorded no reason."


def _audio_status_text(row: dict[str, Any]) -> str:
    """One audio render's status, as a bare lowercase string."""
    return str(row.get("status") or "").strip().lower()


def _audio_file_is_playable(row: dict[str, Any] | None) -> bool:
    """Whether Play has a real file to hand the player.

    `False` for no row at all, a row with no `file_path` (never
    synthesized, or the one dedicated voice-resolution-failure row that
    never gets one -- see `briefing_audio._record_voice_resolution_
    failure`), a `file_path` that fails `audio_file_path_is_safe` (Qodo
    review round 1, FIX B -- checked BEFORE any filesystem access, so an
    unsafe path is never even probed), and a `file_path` that no longer
    exists on disk (deleted out from under the row -- the honest-
    degradation case: an artifact whose file is gone must not offer a
    control that can never do anything, spec §Error-handling ethos).

    Args:
        row: `ArtifactsPane.script_audio`, or `None`.

    Returns:
        `True` only when `row["file_path"]` is a non-empty string naming a
        file, inside `briefing_audio_dir()`, that exists right now.
    """
    if row is None:
        return False
    file_path = row.get("file_path")
    if not file_path:
        return False
    if not audio_file_path_is_safe(file_path):
        return False
    return Path(str(file_path)).exists()


def _audio_detail_renderable(row: dict[str, Any] | None) -> RenderableType:
    """What `_script_detail_renderable` appends for the script's audio.

    Mirrors `_script_detail_renderable`'s own "every status gets a body of
    its own" rule: `generating`/`complete`/`failed`, or "nothing
    synthesized yet", each read as an outcome, never a blank pane. A plain
    function (not a method) so it is directly testable against a bare
    `dict | None` with no widget instance required, the same shape
    `_script_turns_renderable` already uses for the turns half of the same
    detail block.

    Args:
        row: `ArtifactsPane.script_audio` -- the selected script's newest
            `briefing_audio` row, or `None`.

    Returns:
        A Rich renderable, grouped into `#artifacts-script-detail`'s own
        render by `_script_detail_renderable`.
    """
    if row is None:
        return Text(_AUDIO_NO_AUDIO)

    status = _audio_status_text(row)
    header = Text()
    header.append("Audio: ", style="dim")
    header.append(status or "unknown status", style="bold")
    duration = row.get("duration_seconds")
    if duration is not None:
        header.append(" · ")
        header.append(f"{float(duration):.1f}s")
    header.append("\n")

    if status == STATUS_FAILED:
        # The provider/service's own message, appended to a `Text` -- which
        # never parses Rich markup -- exactly like `_detail_renderable`'s
        # own `error` handling; this is model/provider data, never a
        # trusted string.
        return Group(
            header, Text(str(row.get("error") or _AUDIO_UNEXPLAINED_FAILURE))
        )
    if status == STATUS_GENERATING:
        return Group(header, Text(_AUDIO_GENERATING_COPY))
    if status == STATUS_COMPLETE:
        return header
    return Group(header, Text(f"Unrecognised audio status: {status or '—'}"))


class ArtifactsPane(RecomposeCaptureGuard, Vertical):
    """List a watchlist's briefings and render the selected one."""

    #: Same Rich terminal-agnostic "current row" idiom as
    #: `NotificationsPane._SELECTED_ROW_STYLE` -- a `DataTable` cell's `Text`
    #: cannot reference Textual CSS variables the way a widget's styles can.
    _SELECTED_ROW_STYLE = "reverse bold"

    #: Review round 1, Minor #4. A plain, app-controlled glyph -- never
    #: provider/model-derived text -- exactly like `ItemsPane._QUEUED_
    #: GLYPH`'s own phase-1 precedent: a single, plain-width character
    #: rather than an emoji, so it cannot skew a `DataTable` column's
    #: alignment the way a double-width glyph could.
    _AUDIO_GLYPH = "♪"

    #: Owner decision, task-7 phase 2b follow-up ("if synthesis fails, show
    #: the audio glyph with a red x", verbatim): the mark appended after
    #: `_AUDIO_GLYPH` when a script's NEWEST `briefing_audio` render is
    #: `STATUS_FAILED` -- the same `✗` a failed status already uses
    #: elsewhere in this app (`chat_screen.py`/`library_rail.py`'s own
    #: ✓/✗ vocabulary). Plain and app-controlled, same reasoning as
    #: `_AUDIO_GLYPH` above. The red comes from an explicit `rich.text.Text`
    #: style, never a markup string -- this pane never markup-parses cell
    #: content (see the module docstring's `hyperlinks=False` note); a
    #: literal `[bold red]` in provider/model text must keep rendering as
    #: those literal characters, not be interpreted as styling.
    _AUDIO_FAILED_MARK = "✗"
    _AUDIO_FAILED_STYLE = "bold red"

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
    #: Task 7: the SELECTED script's newest `briefing_audio` render, or
    #: `None` when it has never been synthesized. Screen-supplied, resolved
    #: alongside `selected_script` inside `_load_briefings` -- never set by
    #: this widget itself, so (unlike `selected_briefing`/`selected_script`)
    #: it carries no `watch_`/message pair: there is nothing on this pane
    #: that "selects" a particular audio render, only ever the newest one.
    script_audio = reactive[dict[str, Any] | None](None, recompose=True)
    #: Review round 1, Minor #4: `{script_id: status}` for every one of
    #: `scripts`' ids that has at least one `briefing_audio` render, keyed
    #: to that render's NEWEST status (`list_briefing_audio` is
    #: newest-first) -- so the scripts table can show an "Audio" indicator
    #: for every row, not just the currently selected one (before this, a
    #: user had to select each script in turn to discover whether it had
    #: ever been synthesized at all). A script id absent from this mapping
    #: has no audio row at all. Owner decision, task-7 phase 2b follow-up:
    #: this used to be a bare `frozenset[int]` of "has at least one
    #: attempt" -- upgraded to carry status so the scripts table can also
    #: distinguish a `STATUS_FAILED` render from a `STATUS_COMPLETE` one
    #: (see `_AUDIO_FAILED_MARK` above), which the old presence-only
    #: shape could not do: a failed synthesis rendered visually identical
    #: to a successful one. Screen-supplied, resolved alongside `scripts`
    #: inside `_load_briefings`.
    scripts_with_audio = reactive[dict[int, str]]({}, recompose=True)
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

    @staticmethod
    def _audio_cell(status: str | None, style: str) -> Text:
        """The scripts table's "Audio" cell for one row.

        Owner decision, task-7 phase 2b follow-up ("if synthesis fails,
        show the audio glyph with a red x", verbatim) -- before this, the
        cell only ever answered "has an attempt of ANY status", so a
        failed synthesis painted identically to a successful one (a
        reviewer independently flagged the same gap). Three states, one
        per value `scripts_with_audio` can carry for a script id:

        * `status is None` (the id is absent from `scripts_with_audio` --
          no `briefing_audio` row at all): an empty cell. A blank cell
          reads as "never attempted", which is exactly the honest state
          here -- unlike the other two branches below, there is no
          attempt to under- or over-state.
        * `status == STATUS_FAILED` (the newest render failed -- this
          also covers a recovered interrupted render: `fail_interrupted_
          audio` writes `STATUS_FAILED`, never a separate "interrupted"
          status, mirroring `fail_interrupted_briefings` exactly): the
          note glyph PLUS a red `_AUDIO_FAILED_MARK`, so a failed
          synthesis finally looks different from a successful one
          without opening the row.
        * Anything else (`STATUS_COMPLETE`, or `STATUS_GENERATING`): the
          note glyph alone, same as a successful render. A `generating`
          row is deliberately read as "an attempt exists" rather than
          "an attempt failed" -- it has not failed yet, and marking it
          with `_AUDIO_FAILED_MARK` pre-emptively would be dishonest in
          the other direction; a THIRD glyph was considered and rejected
          for this one row shape (see the phase-2b report) since nothing
          in the spec calls for one and the column is already
          tight (`test_the_briefings_table_keeps_at_least_three_usable_
          rows`'s own sibling pins this table's width).

        A single `Text`, assembled with `.append(..., style=...)` per
        span -- exactly like `_audio_detail_renderable`'s own header
        above -- never a markup string: this pane never markup-parses
        cell content (module docstring, `hyperlinks=False`), and the red
        mark must come from an explicit style object for the identical
        reason a briefing/watchlist name must never reach a markup
        parser.

        Args:
            status: The newest `briefing_audio` status for this script
                (`scripts_with_audio.get(script_id)`), or `None`.
            style: The row's own selection style (`_SELECTED_ROW_STYLE`
                or `""`) -- applied to the whole cell so a selected row's
                Audio cell reverses/bolds exactly like its siblings.

        Returns:
            A `Text` -- empty, glyph-only, or glyph-plus-red-mark.
        """
        cell = Text("", style=style)
        if status is None:
            return cell
        cell.append(ArtifactsPane._AUDIO_GLYPH, style=style)
        if status == STATUS_FAILED:
            cell.append(" ")
            mark_style = f"{style} {ArtifactsPane._AUDIO_FAILED_STYLE}".strip()
            cell.append(ArtifactsPane._AUDIO_FAILED_MARK, style=mark_style)
        return cell

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
            # Task 1 (phase 3): exporting is an action on THE SELECTED
            # briefing, so -- unlike Generate/Refresh, which are
            # watchlist-wide -- it is disabled with nothing selected, and
            # ALSO disabled for any non-`complete` row: a `failed`/`empty`/
            # `generating` briefing has no body worth exporting (`empty`
            # writes no body by design, `failed` recorded none, and
            # `generating` has not finished). Placed in this SAME toolbar
            # rather than a new `Horizontal` -- adding a row here would cost
            # height this pane's budget cannot spare (see the module
            # docstring's own note on the pane's fixed `fr` split).
            export_disabled = (
                self.selected_briefing is None
                or _status_text(self.selected_briefing) != STATUS_COMPLETE
            )
            yield Button(
                "Export…",
                id="artifacts-export-button",
                compact=True,
                disabled=export_disabled,
                tooltip=(
                    "Select a completed briefing to export it."
                    if export_disabled
                    else "Export this briefing as a markdown file."
                ),
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
            # `[item N]` the body actually cites, activatable via `Enter`
            # (or a second click on an already-current row) to jump
            # straight to that item in the reader, or, for an item pruned
            # since this briefing was written, a toast saying so
            # (`WatchlistsCollectionsScreen.handle_citation_activated`).
            # Deliberately NOT on mere highlight/cursor-arrival, unlike the
            # briefings/scripts tables below -- see `on_data_table_cell_
            # highlighted`'s docstring (review fix round 1): activating a
            # citation switches sections and marks an item read, so arrow-
            # key BROWSING of this list must not perform that action on
            # every single step the way it harmlessly does for the other
            # two, in-place, tables.
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
            scripts_table.add_columns("Preset", "Status", "Created", "Audio")
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
                audio_status = self.scripts_with_audio.get(row.get("id"))
                scripts_table.add_row(
                    Text(str(row.get("preset_name") or "—"), style=style),
                    Text(_script_status_text(row) or "—", style=style),
                    Text(str(row.get("created_at") or "—"), style=style),
                    self._audio_cell(audio_status, style),
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

            if self.selected_script is not None:
                # Task 7: synthesizing/playing audio is an action on THE
                # SELECTED SCRIPT, so -- exactly like Cast's own gating on
                # `selected_briefing` above -- there is nothing for it to
                # act on without one, and this whole section renders only
                # once a script is selected.
                play_disabled = not _audio_file_is_playable(self.script_audio)
                with Horizontal(
                    id="artifacts-audio-toolbar", classes="destination-filter-strip"
                ):
                    yield Button(
                        "Synthesize",
                        id="artifacts-synthesize-button",
                        compact=True,
                        tooltip=(
                            "Synthesize spoken audio for this script, using "
                            "its roster's voices."
                        ),
                    )
                    yield Button(
                        "Play",
                        id="artifacts-play-button",
                        compact=True,
                        disabled=play_disabled,
                        tooltip=(
                            "No playable audio file for this script."
                            if play_disabled
                            else "Play this script's synthesized audio."
                        ),
                    )
                    yield Button(
                        "Stop",
                        id="artifacts-stop-button",
                        compact=True,
                        tooltip="Stop this script's audio playback.",
                    )
                # No separate audio-detail `Static`: its status/duration/
                # error is folded into `_script_detail_renderable`'s own
                # render, right above -- see that method's own comment on
                # why (this section's `fr` budget is already measured and
                # pinned; a second scrollable region would steal from it).

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
            body: RenderableType = _script_turns_renderable(row.get("turns_json"))
        elif status == STATUS_FAILED:
            body = Text(str(row.get("error") or _SCRIPT_UNEXPLAINED_FAILURE))
        elif status == STATUS_GENERATING:
            body = Text(_SCRIPT_GENERATING_COPY)
        else:
            body = Text(f"Unrecognised script status: {status or '—'}")

        # Task 7: the script's audio render, appended below its own body --
        # folded into this SAME renderable rather than a second scrollable
        # `Static`, to stay inside the pane's already fully-measured `fr`
        # budget instead of stealing new fixed rows from it (see
        # `_watchlists.tcss`'s own comment on why that split is pinned).
        return Group(
            header, body, Text("\n"), _audio_detail_renderable(self.script_audio)
        )

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
        """Fires on activation (`Enter`, or a second click on an
        already-current row) -- but only while a table's `cursor_type` is
        `"row"`. None of this pane's tables set that (all three default to
        `"cell"`, unset here same as everywhere else on this pane), so in
        practice `on_data_table_cell_selected` below is the event a real
        `Enter`/re-click activation actually reaches for the citations
        table today; this handler is kept (and still routes the same way)
        in case a future change ever does set `cursor_type = "row"`.

        See `on_data_table_cell_highlighted`'s docstring for why the
        citations table is only ever activated here/`on_data_table_cell_
        selected`, never on mere highlight (review fix round 1): the other
        two tables reach their selection through a highlight (a single
        click) instead, but a citation's activation switches sections and
        marks an item read, so it requires the same deliberate
        confirmation `Enter`/re-click already is for the OTHER kind of
        action a table row can trigger.
        """
        event.stop()
        if event.row_key is None or event.row_key.value is None:
            return
        if event.data_table.id == "artifacts-citations-table":
            self.activate_citation_by_id(str(event.row_key.value))
        elif event.data_table.id == "artifacts-scripts-table":
            self.select_script_by_id(str(event.row_key.value))
        else:
            self.select_briefing_by_id(str(event.row_key.value))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        """The activation event a real `Enter`/re-click actually produces
        here (review fix round 1): every table's `cursor_type` on this pane
        defaults to `"cell"`, so `DataTable._post_selected_message` posts
        `CellSelected`, never `RowSelected` -- see `on_data_table_row_
        selected`'s own docstring. Handled for the CITATIONS table only:
        briefings/scripts already select on `RowHighlighted` (a single
        click reaching the pane's `selected_briefing`/`selected_script`
        reactive), and adding a second, redundant activation path for
        those two is out of this fix's scope -- their `CellSelected` is
        deliberately left unhandled, unchanged from before this review
        round.
        """
        event.stop()
        if event.data_table.id != "artifacts-citations-table":
            return
        row_key = getattr(event.cell_key, "row_key", None)
        if row_key is None or row_key.value is None:
            return
        self.activate_citation_by_id(str(row_key.value))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Select on cursor movement, which is what a single click produces
        -- while a table's `cursor_type` is `"row"`. None of this pane's
        tables set that (see `on_data_table_cell_highlighted`'s docstring:
        the default is `"cell"`, so THAT handler, not this one, is what a
        real click/arrow key actually reaches today); this stays wired, and
        routes the same way, in case a future change ever does set
        `cursor_type = "row"` for one of them.

        Review fix round 1 (Important): the citations table is deliberately
        LEFT OUT of this routing, unlike briefings/scripts -- see
        `on_data_table_cell_highlighted`'s docstring for the full reasoning
        (this method's own citations branch would suffer the identical
        defect, were `cursor_type` ever changed to `"row"`, so it is left
        out here too, for consistency, even though it is not the path the
        reviewer's live repro actually went through).
        """
        event.stop()
        if event.data_table.id == "artifacts-citations-table":
            return
        if not highlight_is_user_driven(event):
            return
        if event.row_key is None or event.row_key.value is None:
            return
        if event.data_table.id == "artifacts-scripts-table":
            self.select_script_by_id(str(event.row_key.value))
        else:
            self.select_briefing_by_id(str(event.row_key.value))

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Same, for a table whose cursor is cell-shaped rather than row-shaped
        -- which, since none of this pane's tables set `cursor_type`, is all
        of them (`DataTable`'s own default is `"cell"`; `DataTable.
        watch_cursor_coordinate` only posts `RowHighlighted` when `cursor_
        type == "row"`). This is therefore the event a real click or arrow
        key actually produces here, unlike `on_data_table_row_highlighted`
        above.

        Review fix round 1: the citations table is inert here too, for the
        identical reason `on_data_table_row_highlighted`'s docstring gives
        -- confirmed live by the reviewer with a real `down` press, which
        reaches THIS handler, not that one.
        """
        event.stop()
        if event.data_table.id == "artifacts-citations-table":
            return
        if not highlight_is_user_driven(event):
            return
        row_key = getattr(event.cell_key, "row_key", None)
        if row_key is None or row_key.value is None:
            return
        if event.data_table.id == "artifacts-scripts-table":
            self.select_script_by_id(str(row_key.value))
        else:
            self.select_briefing_by_id(str(row_key.value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "artifacts-generate-button":
            self.post_message(GenerateBriefingRequested())
        elif button_id == "artifacts-refresh-button":
            self.post_message(RefreshBriefingsRequested())
        elif button_id == "artifacts-export-button":
            self.post_message(ExportBriefingRequested())
        elif button_id == "artifacts-presets-button":
            self.post_message(ManagePresetsRequested())
        elif button_id == "artifacts-cast-button":
            self.post_message(CastScriptRequested())
        elif button_id == "artifacts-synthesize-button":
            self.post_message(SynthesizeAudioRequested())
        elif button_id == "artifacts-play-button":
            self.post_message(PlayAudioRequested())
        elif button_id == "artifacts-stop-button":
            self.post_message(StopAudioRequested())
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
