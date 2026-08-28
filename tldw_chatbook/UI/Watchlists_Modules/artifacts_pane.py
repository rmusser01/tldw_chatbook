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
  `briefings`/`scope_label`/... are `reactive(..., recompose=True)` on a
  non-screen widget (task-627/637 -- see `recompose_capture_guard.py`).
* `highlight_is_user_driven` (i.e. `table.has_focus`) on every
  `RowHighlighted`/`CellHighlighted`, because a freshly recomposed
  `DataTable` announces a row-0 highlight of its own accord, and forwarding
  that to selection turns this pane into a feedback loop against its own
  rebuild (TASK-1105, and the 157-selections-from-one-tab-open lesson).

Task-15779 split the pane's recompose surface in two. The SELECTION-derived
reactives (`selected_briefing`, `scripts`, `selected_script`,
`script_audio`, `scripts_with_audio`, `citations`) no longer recompose the
pane at all: selecting a briefing used to destroy and rebuild the very
briefings `DataTable` the user was arrow-keying through, taking its focus,
cursor and scroll with it (a second arrow-key press then did nothing --
found and measured in task-15461). Their watchers now rebuild ONLY
`BriefingDetailRegion` -- the stateless boundary widget holding everything
below the table -- and patch the table's highlight and the Export/Keep
buttons in place. The remaining `recompose=True` reactives (`briefings`,
the scope/toolbar/picker state) still rebuild the whole pane, table
included, since they change what the table itself must show.

Task-16852 applies the identical recipe one level down. Task-15779
disclosed, as deliberately unexpanded scope, that selecting a SCRIPT still
rebuilt the whole `BriefingDetailRegion` -- scripts `DataTable` included --
because `selected_script`/`script_audio` funnelled into the same
`_refresh_detail_region()` every other selection-derived reactive did.
`selected_script` and `script_audio` now rebuild only `ScriptDetailRegion`,
a second stateless boundary nested inside `BriefingDetailRegion` that holds
just the script detail `Static` and the Synthesize/Play/Stop toolbar;
`watch_selected_script` additionally patches the scripts table's highlight
in place, mirroring `watch_selected_briefing`'s treatment of the briefings
table. `scripts`/`citations` are unchanged: they still rebuild the WHOLE
detail region, because they change what the scripts (or citations) table
itself must show -- a genuine row-SET change. `scripts_with_audio` was
INITIALLY left on that same wide path too, on the same "changes what the
table must show" reasoning -- but review caught that the reasoning does not
actually hold for it: it never adds or removes a script row, only a row's
existing Audio-column cell, and a first synthesis for the selected script
(the pane's own primary action on a script, right after the very selection
this task made survive) was reopening the destroy-rebuild defect on its own
happy path. `watch_scripts_with_audio` now patches every row's cells in
place instead (`_restyle_script_row`, the same helper `watch_selected_
script` already uses), and never touches the table's mounted identity.

The body is markdown an LLM wrote from remote feed content, so it is
rendered with `hyperlinks=False` -- see `_MARKDOWN_HYPERLINKS` below.

Spec #2 phase 2b, Task 7 (audio): `selected_script`/`script_audio` rebuild
`ScriptDetailRegion` (task-16852) on every script selection or audio
arrival, so EVERY one of those rebuilds every widget `compose_script_
detail()` yields -- including Play/Stop. That rules out ever holding "is
this row currently playing" as a reactive/attribute on this widget: it
would be silently reset to its default on the very next selection, wrong
by construction. The shared `SimpleAudioPlayer` singleton (`TTS/audio_
player.get_audio_player`) is the only thing that survives a recompose, so
it is also the only thing consulted for playback state -- see
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
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Select, Static
from textual.widgets.data_table import CellDoesNotExist, ColumnKey, RowDoesNotExist

from ...Subscriptions.briefing_audio import audio_file_path_is_safe
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
from ...Subscriptions.html_text import strip_control_characters
from ...Widgets.prune_safe_select import PruneSafeSelect
from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .humane_time import humane_timestamp
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


class KeepBriefingRequested(Message):
    """Posted when the user asks to keep the selected briefing into
    ChaChaNotes (task-1780, Task 5).

    Carries nothing, same shape as `ExportBriefingRequested` and for the
    same reason: the briefing to keep is the screen's own `_selected_
    briefing`, and the origin is always `"manual"` for a button press --
    there is nothing this message needs to carry that the screen does not
    already hold or already know.
    """


class KeptBriefingsRequested(Message):
    """Posted when the user asks to open the kept-briefings modal
    (task-1780, Task 5: `KeptBriefingsModal`, list + detail + cast).

    Carries nothing: unlike every other message on this pane, the surface
    it opens is deliberately SCOPE-INDEPENDENT -- it lists ChaChaNotes
    content, reachable whether or not a watchlist is currently in scope
    (including after the watchlist that produced a kept briefing is gone)
    -- so there is no selection or scope state for this message to carry.
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


class BriefingCadenceChanged(Message):
    """Posted when the user picks a different scheduled-briefing cadence.

    Spec #2 phase 4, Task 4: `briefing_cadence_seconds` had a writer (Task
    2's `set_watchlist_briefing_settings`) and a reader (Task 3's scheduler,
    through `list_briefing_schedules`) but no way for a user to set it --
    this retires that deferral, mirroring `BriefingModeChanged` exactly: the
    screen owns the write (`asyncio.to_thread(db.set_watchlist_briefing_
    settings, ..., briefing_cadence_seconds=...)`); this pane only reports
    the user's pick. `None` means "Off" (never scheduled) -- a REAL option
    value on the picker's `Select`, not `Select.NULL`, the same real,
    distinct-from-blank value `BriefingDefaultPresetChanged.preset_id`
    already carries for "App default" (see `_APP_DEFAULT_PRESET_LABEL`
    above).
    """

    def __init__(self, seconds: int | None) -> None:
        self.seconds = seconds
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


class ExportFeedRequested(Message):
    """Posted when the user asks to export this watchlist's audio as a
    podcast feed directory (spec #2 phase 3, Task 5).

    Carries nothing, same shape as `GenerateBriefingRequested`/
    `SynthesizeAudioRequested` and for the same reason: the watchlist to
    export is the screen's own scope (`_briefing_watchlist_id`), and
    whether there is anything worth exporting is already mirrored onto
    this pane's own `has_audio_episodes` reactive -- there is nothing this
    message needs to carry that the screen does not already hold.

    Posted from the button living in `#artifacts-toolbar` -- the SAME
    watchlist-scoped toolbar Generate/Refresh/Task 1's markdown Export
    already live in, and (review round 1, Important #1) NOT `#artifacts-
    audio-toolbar`, where an earlier draft placed it: that toolbar only
    renders once a SCRIPT is selected, but this export is WATCHLIST-
    scoped (every complete episode across the whole watchlist), so a
    button hidden behind an unrelated script selection is a button a
    user cannot find at all -- see `compose`'s own comment at the
    button's new site.
    """


class ServeFeedRequested(Message):
    """Posted when the user asks to serve the most recently exported feed
    directory over localhost (task-1760).

    Carries nothing, same shape as `ExportFeedRequested` and for the same
    reason: which directory to serve is the screen's own state (the last
    directory `export_feed_directory` wrote to), and whether one is even
    available yet is already mirrored onto this pane's own `can_serve_
    feed` reactive. Whether a server is ALREADY running is mirrored onto
    `feed_server_running` -- `compose` disables this button while that is
    `True` (see its own comment), but the screen's handler re-checks both
    conditions anyway, the same "the button's disabled state and the
    message it posts are two different frames" reasoning `handle_export_
    feed_requested` already states for its own re-check.
    """


class StopFeedServerRequested(Message):
    """Posted when the user asks to stop serving the feed directory.

    Carries nothing, for the identical reason `StopAudioRequested` does:
    there is exactly one feed server per screen (task-1760's own "one
    directory at a time" decision), so there is nothing to name.
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

#: The cadence picker's options (spec #2 phase 4, Task 4): `Off` (`None`,
#: never scheduled -- Locked Decision 4 of the phase 4 plan, opt-in per
#: watchlist) plus three preset cadences in seconds, mirroring `sources_
#: pane.py`'s own labelled-seconds idiom (`_FREQUENCY_OPTIONS`, for a
#: source's check frequency) rather than inventing a new one. `Off` carries
#: `None` as a REAL option value, for the identical reason
#: `_APP_DEFAULT_PRESET_LABEL` above states for the default-preset picker's
#: own `None` option -- verified directly against this Textual version
#: (`Select.value` compares by `==`, and `None == Select.NULL` is `False`,
#: so the two are never confused): with `allow_blank=False` and `None`
#: among the option values, `None` is a legal, distinct selection.
_CADENCE_OPTIONS: list[tuple[str, int | None]] = [
    ("Off", None),
    ("Every 12 hours", 43_200),
    ("Every 24 hours", 86_400),
    ("Every 7 days", 604_800),
]

#: What each non-Off cadence means in the scope label's own words, keyed by
#: the identical seconds values `_CADENCE_OPTIONS` offers -- so a new
#: cadence option cannot silently drift out of sync with what the scope
#: label says about it. `cadence_scope_phrase` below is the only reader.
_CADENCE_SCOPE_PHRASES: dict[int, str] = {
    43_200: "every 12 hours",
    86_400: "every 24 hours",
    604_800: "every 7 days",
}


def cadence_scope_phrase(
    seconds: int | None, *, schedules_enabled: bool = True
) -> str | None:
    """What the Artifacts scope label should say a stored cadence means.

    Spec #2 phase 4, Task 4: `WatchlistsCollectionsScreen._briefing_scope_
    label` used to always say "written on this device, on request" -- true
    in phase 1, when nothing could write `briefing_cadence_seconds`, but a
    lie the moment Task 2 gave it a writer and this task gave the writer a
    picker. "While the app is open" is the phase 4 plan's own promised
    copy: there is no background service, so a schedule only fires for as
    long as this process keeps running (`Scheduling/scheduler/loop.py`'s
    own worker lifecycle) -- see `Docs/User_Guide/watchlists.md`'s
    "Scheduled briefings" note, which states the identical phrase.

    Task-1812, AC #1: `schedules_enabled` retires a SECOND honesty gap the
    same reasoning above missed -- `[scheduling] briefing_schedules_
    enabled` (`app.py`'s `_wire_watchlists_and_notifications_services`)
    gates whether anything in this process ever reads `briefing_cadence_
    seconds` back at all. With the flag off, a stored cadence is data with
    no reader: "scheduled ... while the app is open" would claim an active
    schedule that cannot exist this run, exactly the same shape of lie
    "on request" was before Task 2 gave the column a writer.

    Args:
        seconds: A watchlist's stored `briefing_cadence_seconds` -- `None`
            for "never scheduled".
        schedules_enabled: Whether `[scheduling] briefing_schedules_
            enabled` is on for this run. Defaults to `True` so every
            pre-task-1812 caller reads unchanged. Irrelevant when `seconds`
            is `None`: there is nothing stored to describe either way.

    Returns:
        `None` for `None` seconds regardless of the flag (the caller falls
        back to "on request"). For a stored cadence: a phrase stating it is
        inert when `schedules_enabled` is `False`; otherwise the existing
        "scheduled <cadence> while the app is open" phrase for one of
        `_CADENCE_OPTIONS`' three cadences, or a generic every-N-seconds
        phrase for any other positive value -- `set_watchlist_briefing_
        settings` (Task 2) accepts any positive int, so a value this
        picker never offered (hand-written, or a future option this pane
        has not shipped yet) must still read as honest, not silently
        fall back to "on request".
    """
    if seconds is None:
        return None
    phrase = _CADENCE_SCOPE_PHRASES.get(seconds, f"every {seconds}s")
    if not schedules_enabled:
        return (
            f"stored to run {phrase}, but scheduled briefings are turned "
            "off for this app -- this schedule will not fire"
        )
    return f"scheduled {phrase} while the app is open"


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


class BriefingDetailRegion(RecomposeCaptureGuard, Vertical):
    """The selection-dependent lower half of the Artifacts pane (task-15779).

    Everything below the briefings table -- the "Briefing detail" title, the
    body, the citations table, and the scripts/audio section -- renders from
    whichever briefing is selected. Before this widget existed it all lived
    directly in `ArtifactsPane.compose`, with the selection-derived
    reactives `recompose=True` ON THE PANE: every selection rebuilt the
    whole pane, briefings `DataTable` included, destroying the very table
    the user was navigating (focus, cursor and scroll all lost with it --
    the "second arrow-key press does nothing" defect task-15461 measured
    and recorded).

    This widget is a recompose BOUNDARY, not a state owner: it holds no
    reactives of its own and renders straight from the parent pane's state
    (`ArtifactsPane.compose_briefing_detail`), so the pane's watchers can
    rebuild JUST this region -- `refresh(recompose=True)`, naturally
    coalesced by Textual's own `_recompose_required` flag when several
    values land in one message-pump drain -- while the table above it
    stands untouched. Keeping the state on the pane (rather than mirroring
    it onto reactives here) means one source of truth: the screen's
    `_apply_briefing_state_to_pane` and `_build_detail_pane` seeding keep
    writing exactly the reactives they always wrote.

    `RecomposeCaptureGuard` for the same task-627/637 reason the pane
    itself carries it: this widget recomposes independently of any screen
    recompose, so a mouse capture held by one of its own children at
    rebuild time would otherwise leak.
    """

    def __init__(self, pane: "ArtifactsPane", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pane = pane

    def compose(self):
        yield from self._pane.compose_briefing_detail()


class ScriptDetailRegion(RecomposeCaptureGuard, Vertical):
    """The script-selection-dependent tail of the detail region (task-16852).

    Nested one level inside `BriefingDetailRegion`: the script detail
    `Static` and the Synthesize/Play/Stop toolbar render from whichever
    script is selected. Before this widget existed, `selected_script`'s
    (and `script_audio`'s) watcher rebuilt the WHOLE `BriefingDetailRegion`
    -- scripts `DataTable` included -- destroying the very table the user
    was navigating (focus, cursor and scroll all lost with it): the exact
    task-15779 defect, one level down, disclosed there as deliberately
    unexpanded scope and found again independently as task-16852.

    Stateless, exactly like its parent: it holds no reactives of its own
    and renders straight from the pane's state
    (`ArtifactsPane.compose_script_detail`), so `watch_selected_script`/
    `watch_script_audio` can rebuild JUST this region while the scripts
    table above it -- and everything else in `BriefingDetailRegion` -- is
    left standing.

    `RecomposeCaptureGuard` for the same task-627/637 reason both of its
    ancestors (`ArtifactsPane`, `BriefingDetailRegion`) carry it: this
    widget recomposes independently of either of them, so a mouse capture
    held by one of its own children at rebuild time would otherwise leak.
    """

    def __init__(self, pane: "ArtifactsPane", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pane = pane

    def compose(self):
        yield from self._pane.compose_script_detail()


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

    briefings = reactive[list[dict[str, Any]]](list, recompose=True)
    #: Task-15779: deliberately NOT `recompose=True`, unlike (almost) every
    #: other reactive on this pane. A selection recompose destroys the very
    #: `DataTable` the user is arrow-keying through -- focus, cursor and
    #: scroll die with the old widget, so the second key press lands on
    #: nothing (measured in task-15461, fixed here). `watch_selected_
    #: briefing` instead patches the table's highlight and the Export/Keep
    #: buttons in place and rebuilds only `BriefingDetailRegion`. The five
    #: selection-DERIVED reactives below (`scripts` through `citations`)
    #: follow the same rule for the same reason: they only ever change as a
    #: consequence of a selection (the synchronous clearing, then the
    #: reload landing), and either arrival used to tear the table down.
    selected_briefing = reactive[dict[str, Any] | None](None)
    #: The scope line, supplied by the screen: which watchlist these
    #: briefings belong to, or the reason there are none to show.
    scope_label = reactive("", recompose=True)
    #: Compact operational state for the selected collection's automation:
    #: interval, app-open boundary, eligibility/history, and reload attention.
    automation_receipt = reactive("", recompose=True)
    #: False when no single watchlist is in scope -- briefings are per
    #: watchlist by schema, so there is nothing for Generate to act on.
    can_generate = reactive(False, recompose=True)
    #: TASK-2311, AC#3: the display name of the provider Generate will
    #: actually use (the watchlist's default preset's provider, else the
    #: app default) -- screen-supplied via `WatchlistsCollectionsScreen.
    #: _briefing_provider_display`, so it stays visible BEFORE the user
    #: presses Generate, not just after in the finished row's `model_used`.
    default_provider_display = reactive("", recompose=True)
    #: The watchlist's stored `briefing_selection_mode` (spec #2 phase 2a,
    #: Task 4). Defaults to the same fallback `briefing_service.
    #: _selection_mode` uses for a NULL/unrecognized column, so a pane that
    #: has not yet heard from the screen shows the same mode generation
    #: would actually use.
    selection_mode = reactive[str](MODE_AUTO_FEATURED, recompose=True)
    #: Every stored `briefing_presets` row, name-ASC (screen-supplied,
    #: watchlist-independent).
    presets = reactive[list[dict[str, Any]]](list, recompose=True)
    #: The watchlist's stored `default_briefing_preset_id`, or `None` for
    #: "use the app default" -- the value `_generate_briefing` passes to
    #: `generate_briefing(..., preset_id=...)`.
    default_preset_id = reactive[int | None](None, recompose=True)
    #: Spec #2 phase 4, Task 4: the watchlist's stored `briefing_cadence_
    #: seconds`. `None` (the default, and the fallback for a pane that has
    #: not yet heard from `_load_briefings`) means "never scheduled" --
    #: Locked Decision 4 of the phase 4 plan: scheduling is opt-in, per
    #: watchlist, off by default, since a schedule spends the user's LLM
    #: tokens unattended.
    briefing_cadence_seconds = reactive[int | None](None, recompose=True)
    #: Task-1812, AC #1: whether `[scheduling] briefing_schedules_enabled`
    #: is on for this run -- the flag that gates whether `app.py` ever
    #: builds the `BriefingProjection`/`BriefingJobHandler` pair that makes
    #: a stored cadence actually fire (there is no UI control for it, so
    #: this pane cannot infer it from anything else it is given). Screen-
    #: supplied, exactly like `chachanotes_available`/`has_audio_episodes`
    #: above -- this widget has no config handle of its own and must not
    #: read `get_cli_setting` itself (see `WatchlistsCollectionsScreen.
    #: _briefing_schedules_enabled`, the one reader). Defaults to `True`,
    #: the flag's own default, so a pane that has not yet heard from the
    #: screen renders exactly as it did before this flag existed.
    briefing_schedules_enabled = reactive(True, recompose=True)
    #: Task 5: every `briefing_scripts` row cast from the SELECTED briefing
    #: (newest first, per `list_briefing_scripts`) -- never every script
    #: across the whole watchlist, since a script belongs to exactly one
    #: briefing and this pane only ever shows one briefing's detail at a
    #: time. Selection-derived: `recompose=False`, watcher refreshes the
    #: WHOLE detail region (task-15779 -- see `selected_briefing` above):
    #: unlike `selected_script`/`script_audio` below, a new `scripts` list
    #: changes what the scripts table itself must show, so it cannot be
    #: scoped to `ScriptDetailRegion` alone (task-16852).
    scripts = reactive[list[dict[str, Any]]](list)
    #: The script whose detail is rendered below the scripts table, or
    #: `None` when nothing is selected. Selection-derived: `recompose=
    #: False`. Task-16852: its watcher patches the scripts table's
    #: highlight in place (mirroring `watch_selected_briefing`) and
    #: refreshes only `ScriptDetailRegion`, the nested boundary holding the
    #: script detail and the audio toolbar -- NOT the whole detail region
    #: (task-15779's original scope), so the scripts table itself is never
    #: rebuilt by a script selection.
    selected_script = reactive[dict[str, Any] | None](None)
    #: Task 7: the SELECTED script's newest `briefing_audio` render, or
    #: `None` when it has never been synthesized. Screen-supplied, resolved
    #: alongside `selected_script` inside `_load_briefings` -- never set by
    #: this widget itself, so (unlike `selected_briefing`/`selected_script`)
    #: it carries no `watch_`/message pair: there is nothing on this pane
    #: that "selects" a particular audio render, only ever the newest one.
    #: Selection-derived: `recompose=False`, watcher refreshes only
    #: `ScriptDetailRegion` (task-16852; task-15779 originally rebuilt the
    #: whole detail region) -- this value never appears on the scripts
    #: table, only in the script detail/Play button below it.
    script_audio = reactive[dict[str, Any] | None](None)
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
    #: inside `_load_briefings`. Selection-derived: `recompose=False`.
    #: Task-16852 review fix: `watch_scripts_with_audio` patches the
    #: scripts table's Audio cells in place (`_restyle_script_row`) --
    #: NOT a whole-region rebuild. Unlike `scripts` above (a real row-SET
    #: change), a synthesis never adds or removes a script row, only an
    #: existing row's Audio cell -- and a first synthesis for the SELECTED
    #: script is the pane's own primary action on a script, so tearing the
    #: table down for it would reopen this task's own defect on the
    #: feature's happy path (caught in review, not in the original pass).
    scripts_with_audio = reactive[dict[int, str]](dict)
    #: Task 5 (phase 3): whether the WHOLE watchlist -- not merely the
    #: selected script -- has at least one export-ready audio episode
    #: (`SubscriptionsDB.list_watchlist_audio_episodes`'s own `complete`
    #: + `file_path IS NOT NULL` predicate). Screen-supplied, resolved
    #: alongside the rest of `_load_briefings`'s watchlist-scoped reads --
    #: never computed on this widget, which has no database handle of its
    #: own. Gates the Export Feed button's disabled state: a dead control
    #: offering to export nothing is a spec violation (phase 2b shipped a
    #: disabled Play for exactly this reason).
    has_audio_episodes = reactive(False, recompose=True)
    #: task-1780, Task 5: whether the screen has a live ChaChaNotes handle
    #: (`getattr(app_instance, "chachanotes_db", None) is not None`).
    #: Screen-supplied, exactly like `has_audio_episodes` above -- this
    #: widget has no database handle of its own, and ChaChaNotes is a
    #: SEPARATE database from the `SubscriptionsDB` briefings live in.
    #: Gates BOTH the Keep button (which also needs a complete selection)
    #: and the Kept Briefings… button (which needs nothing else at all --
    #: see `KeptBriefingsRequested`'s own docstring on why that surface is
    #: scope-independent).
    chachanotes_available = reactive(False, recompose=True)
    #: task-1760: whether the screen holds a directory `export_feed_
    #: directory` has already written to this session -- the Serve button
    #: needs SOMETHING to serve, and (unlike `has_audio_episodes`, which
    #: asks whether an export is *possible*) this asks whether one has
    #: actually happened yet. Screen-supplied, exactly like `has_audio_
    #: episodes` above: this widget owns no filesystem state of its own,
    #: and the directory is `Subscriptions.feed_server.FeedDirectoryServer`'s
    #: to hold, not this pane's.
    can_serve_feed = reactive(False, recompose=True)
    #: task-1760: whether the screen's `FeedDirectoryServer` is currently
    #: serving. Screen-supplied -- this pane never starts or stops the
    #: server itself, it only posts `ServeFeedRequested`/`StopFeedServer
    #: Requested` and renders whatever the screen reports back, the same
    #: division of labour `selected_script`'s own module-docstring note
    #: describes for audio PLAYBACK state (a recompose must never silently
    #: reset "is something running" back to a default that disagrees with
    #: reality).
    feed_server_running = reactive(False, recompose=True)
    #: task-1760: the running server's URL, or `None` when nothing is
    #: being served. Screen-supplied alongside `feed_server_running` --
    #: used only for the Stop button's tooltip (the toast that announces a
    #: fresh URL is the screen's own responsibility, not this pane's).
    feed_server_url = reactive[str | None](None, recompose=True)
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
    #: so it must never reach a markup parser. Selection-derived:
    #: `recompose=False`, watcher refreshes the detail region only
    #: (task-15779).
    citations = reactive[list[dict[str, Any]]](list)

    #: The briefings table's `ColumnKey`s, captured by `compose` when the
    #: table is built so `_restyle_briefing_row` can address cells by key
    #: (task-15779). A tuple default, never a mutable class attribute.
    _briefings_column_keys: tuple[ColumnKey, ...] = ()
    #: The scripts table's own `ColumnKey`s, captured by `compose_briefing_
    #: detail` when the table is built -- `_restyle_script_row`'s own
    #: addressing, mirroring `_briefings_column_keys` one level down
    #: (task-16852).
    _scripts_column_keys: tuple[ColumnKey, ...] = ()

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

    def _cadence_select_options(self) -> list[tuple[str, int | None]]:
        """Options for the cadence picker: `_CADENCE_OPTIONS` verbatim,
        plus a synthetic trailing option when the stored cadence is not
        one of them.

        Same defensive shape as `_preset_select_options`'s stale-id
        fallback immediately above: `set_watchlist_briefing_settings`
        (Task 2) accepts any positive `briefing_cadence_seconds`, so a
        value this picker never offered must not raise `InvalidSelect
        ValueError` the moment `compose` builds `value=self.briefing_
        cadence_seconds`.
        """
        options = list(_CADENCE_OPTIONS)
        known_seconds = {seconds for _, seconds in options}
        current = self.briefing_cadence_seconds
        if current is not None and current not in known_seconds:
            options.append((f"Every {current}s", current))
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

    def _briefing_row_cells(self, row: dict[str, Any], style: str) -> tuple[Text, ...]:
        """One briefings-table row's six cells, styled as a unit.

        Task-15779: shared by `compose` (the build) and `_restyle_briefing_
        row` (the in-place highlight move a selection now performs instead
        of a rebuild), so the selected-row presentation cannot drift
        between the two paths.
        """
        return (
            Text(_status_text(row) or "—", style=style),
            Text(_window_text(row), style=style),
            Text(str(row.get("item_count") or 0), style=style),
            Text(str(row.get("featured_count") or 0), style=style),
            Text(str(row.get("overflow_count") or 0), style=style),
            # TASK-2308. This column is where the UAT found the house
            # style -- but "2026-08-04 18:22:44" is simply SQLite's
            # `CURRENT_TIMESTAMP`, i.e. UTC that happens to look humane.
            # It goes through the same formatter as every other Watchlists
            # table so the whole screen agrees on one zone.
            Text(humane_timestamp(row.get("created_at")), style=style),
        )

    def _script_row_cells(self, row: dict[str, Any], style: str) -> tuple[Text, ...]:
        """One scripts-table row's four cells, styled as a unit.

        Task-16852: shared by `compose_briefing_detail` (the build) and
        `_restyle_script_row` (the in-place highlight move a script
        selection now performs instead of rebuilding the whole detail
        region), so the selected-row presentation cannot drift between the
        two paths -- the same discipline `_briefing_row_cells` established
        for the briefings table in task-15779.
        """
        audio_status = self.scripts_with_audio.get(row.get("id"))
        return (
            Text(strip_control_characters(row.get("preset_name") or "—"), style=style),
            Text(_script_status_text(row) or "—", style=style),
            Text(humane_timestamp(row.get("created_at")), style=style),
            self._audio_cell(audio_status, style),
        )

    def _export_button_state(self) -> tuple[bool, str]:
        """The Export button's (disabled, tooltip) for the current selection.

        Task-15779: shared by `compose` and `_update_selection_dependent_
        buttons` -- a selection patches the mounted button in place rather
        than recomposing the pane, and the two paths must agree.
        """
        disabled = (
            self.selected_briefing is None
            or _status_text(self.selected_briefing) != STATUS_COMPLETE
        )
        tooltip = (
            "Select a completed briefing to export it."
            if disabled
            else "Export this briefing as a markdown file."
        )
        return disabled, tooltip

    def _keep_button_state(self, export_disabled: bool) -> tuple[bool, str]:
        """The Keep button's (disabled, tooltip) -- Export's condition plus
        a live ChaChaNotes handle. Shared for the same task-15779 reason as
        `_export_button_state` above.
        """
        disabled = export_disabled or not self.chachanotes_available
        if export_disabled:
            tooltip = "Select a completed briefing to keep it."
        elif not self.chachanotes_available:
            tooltip = "Connect a ChaChaNotes database to keep briefings."
        else:
            tooltip = (
                "Keep this briefing (and its scripts) so it survives "
                "this watchlist's deletion."
            )
        return disabled, tooltip

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
        #
        # TASK-2311, AC#3: the provider Generate will use is appended to
        # this SAME always-visible line rather than a new row (this pane's
        # toolbars are already at their `.destination-filter-strip` height
        # budget) -- `default_provider_display` is screen-computed, app-
        # controlled text, safe to concatenate before the one `Text(...)`
        # wrap that protects `scope_label` (user-authored) above.
        scope_text = (
            self.scope_label
            or "Briefings are written on this device from the local "
            "watchlist store."
        )
        if self.can_generate and self.default_provider_display:
            scope_text = (
                f"{scope_text} Generate will use "
                f"{self.default_provider_display}."
            )
        yield Static(
            Text(
                self.automation_receipt
                or "Automation: select a collection to inspect its schedule."
            ),
            id="artifacts-automation-receipt",
        )
        yield Static(Text(scope_text), id="artifacts-scope-note")
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
            #
            # Task-15779: the disabled/tooltip pair comes from `_export_
            # button_state`, shared with the in-place update `watch_
            # selected_briefing` performs -- a selection no longer
            # recomposes this pane, so these two buttons are the one piece
            # of selection-dependent chrome OUTSIDE `BriefingDetailRegion`
            # and must be patchable without a rebuild.
            export_disabled, export_tooltip = self._export_button_state()
            yield Button(
                "Export…",
                id="artifacts-export-button",
                compact=True,
                disabled=export_disabled,
                tooltip=export_tooltip,
            )
            # task-1780, Task 5: keeping a briefing into ChaChaNotes so it
            # survives this watchlist's deletion. Same disabled shape as
            # Export above (no selection, or a non-`complete` row) PLUS a
            # second, independent requirement: a live ChaChaNotes handle --
            # `keep_briefing` writes into a database this pane has no
            # access to at all, so a missing handle degrades this ONE
            # button (never the whole pane, never the whole toolbar), with
            # a tooltip naming which of the two conditions is unmet.
            keep_disabled, keep_tooltip = self._keep_button_state(export_disabled)
            yield Button(
                "Keep",
                id="artifacts-keep-button",
                compact=True,
                disabled=keep_disabled,
                tooltip=keep_tooltip,
            )
            # Opens Task 5's `KeptBriefingsModal` -- deliberately gated on
            # `chachanotes_available` ALONE, unlike every sibling button in
            # this toolbar: it lists ChaChaNotes content directly, so it
            # needs no selected briefing and no watchlist in scope at all
            # (see `KeptBriefingsRequested`'s own docstring).
            yield Button(
                "Kept Briefings…",
                id="artifacts-kept-briefings-button",
                compact=True,
                disabled=not self.chachanotes_available,
                tooltip=(
                    "Connect a ChaChaNotes database to browse kept "
                    "briefings."
                    if not self.chachanotes_available
                    else "Browse, cast from, or delete briefings you have "
                    "kept."
                ),
            )
            # Task 5 (phase 3): review round 1, Important #1. The brief
            # originally placed this button in `#artifacts-audio-toolbar`,
            # which only renders once a SCRIPT is selected -- but the feed
            # export itself is WATCHLIST-scoped (every complete episode
            # across the whole watchlist), not script-scoped, so a user
            # could not find it without first selecting some unrelated
            # script. Moved to THIS toolbar instead: it is the one Task 1's
            # own watchlist-scoped markdown Export already lives in, and it
            # renders unconditionally (see `compose`'s own top-level
            # structure -- unlike the picker/scripts/audio sections below,
            # nothing gates this `Horizontal` at all), so the button is
            # discoverable the moment a watchlist is in scope, exactly like
            # Generate/Refresh/Export are. Still costs zero rows: both are
            # EXISTING `.destination-filter-strip` toolbars, `height: 1`
            # either way -- see the pinned geometry tests re-run for this
            # move (`test_the_list_the_button_and_the_body_are_all_on_
            # screen`, `test_the_briefings_table_keeps_at_least_three_
            # usable_rows`).
            yield Button(
                "Export Feed…",
                id="artifacts-export-feed-button",
                compact=True,
                disabled=not self.has_audio_episodes,
                tooltip=(
                    "This watchlist has no complete audio episodes to "
                    "export."
                    if not self.has_audio_episodes
                    else "Export this watchlist's audio episodes as a "
                    "podcast feed directory."
                ),
            )
            # task-1760: Serve/Stop, adjacent to Export Feed for the same
            # reason Export Feed itself lives here (review round 1,
            # Important #1, above) -- serving a feed is the natural next
            # step after exporting one, and this toolbar is the one place
            # that is guaranteed reachable regardless of any briefing/
            # script selection. Two buttons, not one toggling label,
            # mirroring `#artifacts-audio-toolbar`'s own Play/Stop pair
            # below: Serve disabled while already running OR with nothing
            # exported yet; Stop (TASK-2310) rendered only while running --
            # see its own comment below for why it has no useful
            # disabled-but-visible state. This also means a SECOND
            # `ServeFeedRequested` while one is
            # already running is unreachable through the button itself --
            # the screen's handler still re-checks (see that message's own
            # docstring), since the button's disabled state and the
            # message it posts are two different frames, same as every
            # other guard on this pane.
            if self.feed_server_running:
                serve_tooltip = (
                    "A feed is already being served. Stop it before "
                    "serving a different export."
                )
            elif not self.can_serve_feed:
                serve_tooltip = (
                    "Export a feed directory first, then serve it over "
                    "localhost."
                )
            else:
                serve_tooltip = (
                    "Serve the exported feed directory over localhost -- "
                    "no authentication; anyone who can reach the address "
                    "can read it while it is serving."
                )
            yield Button(
                "Serve Feed",
                id="artifacts-serve-feed-button",
                compact=True,
                disabled=self.feed_server_running or not self.can_serve_feed,
                tooltip=serve_tooltip,
            )
            # TASK-2310: unlike every sibling button in this toolbar, "Stop
            # Serving" has no useful disabled-but-visible state -- it can
            # only ever act on a server THIS pane just started, so a user
            # who has never pressed Serve has nothing it could explain by
            # staying visible (contrast Export/Keep/Serve, which stay
            # visible-but-disabled specifically so a first-time user can
            # discover them before they apply). UAT: the toolbar showed
            # "Stop Serving" before any briefing existed, one of 12 controls
            # crowding a brand-new watchlist's empty state. Rendered only
            # once there is something it could act on.
            if self.feed_server_running:
                yield Button(
                    "Stop Serving",
                    id="artifacts-stop-feed-button",
                    compact=True,
                    tooltip=(
                        f"Stop serving {self.feed_server_url}."
                        if self.feed_server_url
                        else "Stop serving."
                    ),
                )

        if self.can_generate:
            # Task 4 (phase 2a): the selection-mode and default-preset
            # pickers, plus the entry into Task 3's preset manager. Task 4
            # (phase 4) adds a third picker -- scheduled-briefing cadence --
            # to this SAME strip rather than a new one, for the identical
            # zero-extra-height reason the phase 3 Export Feed button was
            # folded into an existing toolbar (see that button's own
            # comment above). Rendered only when a single watchlist is in
            # scope -- like Generate itself, there is nothing for any
            # picker to act on without one, and unlike Generate (which
            # stays visible-but-disabled to explain itself) a picker with
            # nothing to pick from has no useful disabled state to show.
            with Horizontal(
                id="artifacts-picker-toolbar", classes="destination-filter-strip"
            ):
                # TASK-2310: UAT read this strip as "Auto + featured ▾ / App
                # default ▾ / Off ▾" -- the third value in particular ("Off")
                # names nothing about what it is off FOR. A sibling `Static`
                # before each Select, same idiom as the Sources/Items filter
                # strips above.
                yield Static("Mode", classes="watchlists-inline-select-label")
                yield PruneSafeSelect(
                    _MODE_OPTIONS,
                    value=self.selection_mode,
                    id="artifacts-mode-select",
                    allow_blank=False,
                    compact=True,
                    tooltip="Which items go into this watchlist's next briefing.",
                )
                yield Static("Preset", classes="watchlists-inline-select-label")
                yield PruneSafeSelect(
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
                # Task-1812, AC #1: disabled -- not merely mis-worded --
                # when the app-level kill switch is off. A stored cadence
                # showing as pickable/active while nothing in this process
                # would ever dispatch it is the exact gap this fix closes;
                # see `cadence_scope_phrase`'s own AC #1 note for the scope
                # label's half of the same fix.
                #
                # Whole-branch review (`chore/briefings-residuals-1810-
                # 1812`), Minor 4: the Select stays disabled ON PURPOSE --
                # its "Off" option is unreachable too, so a stored cadence
                # cannot be cleared from the UI while the flag is off. That
                # is deliberate (a stored value is never silently cleared),
                # but the disabled tooltip must say so plainly: a stored
                # cadence is not merely inert here, it SURVIVES and will
                # resume firing the moment the flag is turned back on.
                schedules_disabled = not self.briefing_schedules_enabled
                # TASK-2310: this is the Select the UAT flagged specifically
                # -- "Off ▾" with no clue what is off. "Cadence" names the
                # axis; the tooltip above already explains what "Off" means.
                yield Static("Cadence", classes="watchlists-inline-select-label")
                yield PruneSafeSelect(
                    self._cadence_select_options(),
                    value=self.briefing_cadence_seconds,
                    id="artifacts-cadence-select",
                    allow_blank=False,
                    compact=True,
                    disabled=schedules_disabled,
                    tooltip=(
                        "Scheduled briefings are turned off for this app "
                        "([scheduling] briefing_schedules_enabled is "
                        "false); a stored cadence stays saved here and "
                        "cannot be changed while this control is disabled "
                        "-- it will resume firing as soon as scheduling is "
                        "turned back on."
                        if schedules_disabled
                        else "How often this watchlist writes a new briefing "
                        "on its own, while the app is open. Off by default. "
                        "After a durable save, a scheduler reload is requested "
                        "immediately and the receipt reports whether it was "
                        "acknowledged."
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
        # The keys are kept so `_restyle_briefing_row` can update cells on
        # the MOUNTED table when the selection moves (task-15779) --
        # `add_columns` auto-generates them, and `update_cell` addresses
        # cells by nothing else.
        self._briefings_column_keys = tuple(
            table.add_columns(
                "Status", "Window", "Items", "Featured", "Overflow", "Created"
            )
        )
        selected_index: int | None = None
        for index, row in enumerate(self.briefings):
            row_key = str(row.get("id"))
            if row_key == selected_key:
                selected_index = index
            style = self._SELECTED_ROW_STYLE if row_key == selected_key else ""
            table.add_row(*self._briefing_row_cells(row, style), key=row_key)
        if selected_index is not None:
            # TASK-1105, exactly as `NotificationsPane` documents it: a
            # recompose of this pane (a reload landing new rows, a section
            # revisit -- since task-15779 no longer a mere selection) builds
            # a new table whose cursor starts at row 0 and says so. Seeding
            # it from the surviving selection stops that announcement from
            # dragging the selection back to the first row.
            table.cursor_coordinate = Coordinate(selected_index, 0)
        yield table

        # Task-15779: everything below the table is selection-dependent, so
        # it lives behind its own recompose boundary -- selecting a briefing
        # rebuilds `BriefingDetailRegion` alone, and the table above it
        # (focus, cursor, scroll) survives. See the class's own docstring.
        yield BriefingDetailRegion(self, id="artifacts-detail-region")

    def compose_briefing_detail(self):
        """The detail region's content, yielded by `BriefingDetailRegion`.

        Everything here reads the PANE's reactives -- the region owns no
        state (see its docstring). Extracted verbatim from this class's own
        `compose` tail by task-15779; the structure, ids and gating are
        unchanged, only the recompose scope moved.
        """
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
            # The keys are kept so `_restyle_script_row` can update cells on
            # the MOUNTED table when the selection moves (task-16852) --
            # mirrors `_briefings_column_keys` above.
            self._scripts_column_keys = tuple(
                scripts_table.add_columns("Preset", "Status", "Created", "Audio")
            )
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
                    *self._script_row_cells(row, style), key=row_key
                )
            if selected_script_index is not None:
                # Same TASK-1105 seeding as the briefings table above.
                scripts_table.cursor_coordinate = Coordinate(
                    selected_script_index, 0
                )
            yield scripts_table

            # Task-16852: everything below the scripts table is SCRIPT-
            # selection-dependent, so it lives behind its own nested
            # recompose boundary -- selecting a script rebuilds
            # `ScriptDetailRegion` alone, and the scripts table above it
            # (focus, cursor, scroll) survives. See the class's own
            # docstring; mirrors `BriefingDetailRegion` one level down.
            yield ScriptDetailRegion(self, id="artifacts-script-detail-region")

    def compose_script_detail(self):
        """The nested script-detail region's content, yielded by
        `ScriptDetailRegion` (task-16852).

        Everything here reads the PANE's reactives -- the region owns no
        state (see its docstring), exactly like `compose_briefing_detail`.
        Extracted verbatim from that method's own tail; the structure, ids
        and gating are unchanged, only the recompose scope moved one level
        deeper.
        """
        # No separate `.pane-title` here (unlike "Briefing detail" above):
        # a `.pane-title` costs 4 rows (`height: 3` + `margin-bottom: 1`)
        # inside a region whose total budget is already fixed and now
        # shared with a second list-over-body pair -- measured to matter,
        # not assumed (a first draft with the title pushed `#artifacts-
        # detail` below the height its own test fixture needs).
        # `_script_detail_renderable`'s own header names it as "Script:"
        # instead, for one row instead of four.
        yield Static(self._script_detail_renderable(), id="artifacts-script-detail")

        if self.selected_script is not None:
            # Task 7: synthesizing/playing audio is an action on THE
            # SELECTED SCRIPT, so -- exactly like Cast's own gating on
            # `selected_briefing` above -- there is nothing for it to act
            # on without one, and this whole section renders only once a
            # script is selected.
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
            # No separate audio-detail `Static`: its status/duration/error
            # is folded into `_script_detail_renderable`'s own render,
            # right above -- see that method's own comment on why (this
            # section's `fr` budget is already measured and pinned; a
            # second scrollable region would steal from it).

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
        header.append(
            humane_timestamp(row.get("created_at")) if row.get("created_at")
            else "unknown time",
            style="bold",
        )
        header.append(" · ")
        header.append(status or "unknown status")
        # Batch-4 review, I1: stripped for the same reason every other
        # identity cell touched by this batch is -- `Text.append` protects
        # only against Rich markup, not a raw control byte.
        model_used = row.get("model_used")
        if model_used:
            header.append(" · ")
            header.append(strip_control_characters(model_used))
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
        header.append(strip_control_characters(row.get("preset_name") or "Untitled preset"), style="bold")
        header.append(" · ")
        header.append(status or "unknown status")
        # Batch-4 review, I1: stripped for the same reason every other
        # identity cell touched by this batch is -- `Text.append` protects
        # only against Rich markup, not a raw control byte.
        model_used = row.get("model_used")
        if model_used:
            header.append(" · ")
            header.append(strip_control_characters(model_used))
        header.append("\n")
        header.append(
            humane_timestamp(row.get("created_at")) if row.get("created_at")
            else "unknown time",
            style="dim",
        )
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

    def watch_selected_briefing(
        self, old: dict[str, Any] | None, briefing: dict[str, Any] | None
    ) -> None:
        if not self.is_mounted:
            # Defense in depth for pre-mount seeding. Since task-15778
            # `WatchlistsCollectionsScreen._build_detail_pane` seeds every
            # reactive with `set_reactive`, so this watcher no longer fires
            # there at all — but a future plain assignment on an unmounted
            # pane must still not wipe the script/citation seeding that
            # follows it, and there is no screen to tell either way.
            return
        self._clear_selection_derived_state()
        self._apply_briefing_selection_in_place(old, briefing)
        self.post_message(BriefingSelected(briefing))

    def _apply_briefing_selection_in_place(
        self, old: dict[str, Any] | None, new: dict[str, Any] | None
    ) -> None:
        """Repaint what a selection changes WITHOUT rebuilding the pane.

        Task-15779. `selected_briefing` used to be `recompose=True`, so
        this was `compose`'s job -- at the cost of destroying the briefings
        `DataTable` the user was navigating (module docstring). A selection
        changes exactly three things this pane shows:

        * the table's selected-row highlight (moved cell-by-cell here, on
          the SURVIVING table);
        * the Export/Keep buttons' disabled/tooltip state;
        * everything below the table -- which is `BriefingDetailRegion`'s
          whole content, rebuilt as one unit.

        Every lookup is guarded: this watcher also fires while a QUEUED
        pane recompose has not run yet (e.g. `_apply_briefing_state_to_
        pane` assigning `briefings` first, then the selection), in which
        case the widgets patched here are the outgoing ones and the
        recompose repaints everything from current state anyway --
        patching them is harmless, failing on them must be too.
        """
        try:
            table = self.query_one("#artifacts-table", DataTable)
        except NoMatches:
            table = None
        if table is not None:
            self._move_briefing_row_highlight(table, old, new)
        self._update_selection_dependent_buttons()
        self._refresh_detail_region()

    def _move_briefing_row_highlight(
        self,
        table: DataTable,
        old: dict[str, Any] | None,
        new: dict[str, Any] | None,
    ) -> None:
        """Restyle the outgoing and incoming rows, and keep the cursor on
        the selection when it was moved programmatically.

        The cursor move is a no-op for the user-driven path (the cursor
        arriving on the row IS what selected it), and deliberately keyed on
        row index alone -- a programmatic selection must never yank the
        COLUMN out from under a cell cursor.
        """
        old_key = str(old.get("id")) if old else None
        new_key = str(new.get("id")) if new else None
        if old_key is not None and old_key != new_key:
            self._restyle_briefing_row(table, old_key, "")
        if new_key is None:
            return
        self._restyle_briefing_row(table, new_key, self._SELECTED_ROW_STYLE)
        try:
            row_index = table.get_row_index(new_key)
        except RowDoesNotExist:
            return
        if table.cursor_coordinate.row != row_index:
            table.move_cursor(row=row_index)

    def _restyle_briefing_row(
        self, table: DataTable, row_key: str, style: str
    ) -> None:
        """Rewrite one row's cells with `style`, by key, on the mounted
        table. Characters are identical either way -- `_briefing_row_cells`
        is the single source for both the build and this patch."""
        row = next(
            (r for r in self.briefings if str(r.get("id")) == row_key), None
        )
        if row is None:
            return
        cells = self._briefing_row_cells(row, style)
        for column_key, cell in zip(self._briefings_column_keys, cells):
            try:
                table.update_cell(row_key, column_key, cell)
            except CellDoesNotExist:
                # The mounted table predates the row (or a queued recompose
                # will replace it) -- nothing to patch, nothing lost.
                return

    def _apply_script_selection_in_place(
        self, old: dict[str, Any] | None, new: dict[str, Any] | None
    ) -> None:
        """Repaint what a script selection changes WITHOUT rebuilding the
        scripts table (task-16852) -- `_apply_briefing_selection_in_place`'s
        own sibling, one level down.

        A script selection changes exactly two things this pane shows:

        * the scripts table's selected-row highlight (moved cell-by-cell
          here, on the SURVIVING table);
        * everything below the scripts table -- `ScriptDetailRegion`'s
          whole content, rebuilt as one unit.

        Guarded exactly like `_apply_briefing_selection_in_place`: this
        watcher can fire before `compose_briefing_detail` has ever built
        the scripts table (e.g. no briefing selected yet, or seeding
        before mount), in which case there is nothing here yet to patch
        and the eventual recompose repaints from current state anyway.
        """
        try:
            table = self.query_one("#artifacts-scripts-table", DataTable)
        except NoMatches:
            table = None
        if table is not None:
            self._move_script_row_highlight(table, old, new)
        self._refresh_script_detail_region()

    def _move_script_row_highlight(
        self,
        table: DataTable,
        old: dict[str, Any] | None,
        new: dict[str, Any] | None,
    ) -> None:
        """Restyle the outgoing and incoming script rows, and keep the
        cursor on the selection when it was moved programmatically.

        Mirrors `_move_briefing_row_highlight` exactly, one level down
        (task-16852): the cursor move is a no-op for the user-driven path
        (the cursor arriving on the row IS what selected it), and keyed on
        row index alone so a programmatic selection never yanks the column
        out from under a cell cursor.
        """
        old_key = str(old.get("id")) if old else None
        new_key = str(new.get("id")) if new else None
        if old_key is not None and old_key != new_key:
            self._restyle_script_row(table, old_key, "")
        if new_key is None:
            return
        self._restyle_script_row(table, new_key, self._SELECTED_ROW_STYLE)
        try:
            row_index = table.get_row_index(new_key)
        except RowDoesNotExist:
            return
        if table.cursor_coordinate.row != row_index:
            table.move_cursor(row=row_index)

    def _restyle_script_row(
        self, table: DataTable, row_key: str, style: str
    ) -> None:
        """Rewrite one scripts-table row's cells with `style`, by key, on
        the mounted table. Characters are identical either way --
        `_script_row_cells` is the single source for both the build and
        this patch (mirrors `_restyle_briefing_row`)."""
        row = next(
            (r for r in self.scripts if str(r.get("id")) == row_key), None
        )
        if row is None:
            return
        cells = self._script_row_cells(row, style)
        for column_key, cell in zip(self._scripts_column_keys, cells):
            try:
                table.update_cell(row_key, column_key, cell)
            except CellDoesNotExist:
                # The mounted table predates the row (or a queued recompose
                # will replace it) -- nothing to patch, nothing lost.
                return

    def _update_selection_dependent_buttons(self) -> None:
        """Patch Export/Keep in place -- the one piece of selection-
        dependent chrome outside `BriefingDetailRegion` (task-15779)."""
        export_disabled, export_tooltip = self._export_button_state()
        keep_disabled, keep_tooltip = self._keep_button_state(export_disabled)
        for button_id, disabled, tooltip in (
            ("#artifacts-export-button", export_disabled, export_tooltip),
            ("#artifacts-keep-button", keep_disabled, keep_tooltip),
        ):
            try:
                button = self.query_one(button_id, Button)
            except NoMatches:
                continue
            button.disabled = disabled
            button.tooltip = tooltip

    def _refresh_detail_region(self) -> None:
        """Schedule ONE rebuild of everything below the briefings table.

        Task-15779: the selection-derived watchers all funnel here rather
        than each recomposing the pane. Multiple calls inside one
        message-pump drain coalesce -- `refresh(recompose=True)` only sets
        `_recompose_required`, and the first `_check_recompose` to run
        clears it -- so a selection plus its reload landing costs at most
        one region rebuild per drain, and the table above is never touched.
        """
        try:
            region = self.query_one(
                "#artifacts-detail-region", BriefingDetailRegion
            )
        except NoMatches:
            return
        region.refresh(recompose=True)

    def _refresh_script_detail_region(self) -> None:
        """Schedule ONE rebuild of the script-detail sub-region only
        (task-16852) -- the Cast toolbar, the scripts table and the
        citations table above it are never touched.

        `_refresh_detail_region`'s own sibling, scoped one level deeper:
        multiple calls inside one message-pump drain coalesce the same way
        (`refresh(recompose=True)` only sets `_recompose_required`), so a
        script selection plus its audio reload landing costs at most one
        `ScriptDetailRegion` rebuild per drain.
        """
        try:
            region = self.query_one(
                "#artifacts-script-detail-region", ScriptDetailRegion
            )
        except NoMatches:
            return
        region.refresh(recompose=True)

    def _clear_selection_derived_state(self) -> None:
        """Drop the previous briefing's scripts, audio and citations.

        A briefing owns its scripts and its citations, so the instant the
        selection moves, whatever is rendered below the detail belongs to a
        briefing that is no longer on screen. `handle_briefing_selected` used
        to clear these from the SCREEN, one reactive assignment at a time, in
        the message handler that runs after this watcher -- which meant the
        select->clear->reload pipeline cost two pane recomposes before the
        reload's own: one for the selection, then a second for the clearing.

        Doing it here, with `set_reactive`, folds the clearing into the ONE
        rebuild the selection change already schedules (task-15461; since
        task-15779 that is `BriefingDetailRegion`'s refresh, no longer a
        whole-pane recompose): `set_reactive` writes the value without
        firing watchers or scheduling a rebuild of its own, and
        `compose_briefing_detail` reads the cleared values when the
        selection watcher's single pending region refresh runs. The
        stale-frame guarantee `handle_briefing_selected` documents is
        unchanged -- if anything it is stronger, since the clearing happens
        in the same synchronous instant as the selection rather than one
        message later.

        `selected_script` is deliberately included: its watcher would post
        `ScriptSelected(None)`, which the screen documents as the redundant
        reactive echo of exactly this clearing.
        """
        self.set_reactive(ArtifactsPane.scripts, [])
        self.set_reactive(ArtifactsPane.selected_script, None)
        self.set_reactive(ArtifactsPane.script_audio, None)
        self.set_reactive(ArtifactsPane.scripts_with_audio, {})
        self.set_reactive(ArtifactsPane.citations, [])

    def watch_selected_script(
        self, old: dict[str, Any] | None, script: dict[str, Any] | None
    ) -> None:
        if not self.is_mounted:
            # Same defense in depth as `watch_selected_briefing`, for the
            # same pre-mount-seeding reason (task-15778/15779).
            return
        self._apply_script_selection_in_place(old, script)
        self.post_message(ScriptSelected(script))

    def watch_scripts(self) -> None:
        """Selection-derived (task-15779): rebuild the detail region only.

        No `is_mounted` guard needed beyond `_refresh_detail_region`'s own:
        an unmounted pane has no region to find, so the query simply
        misses. The same applies to `watch_citations` below. Unlike
        `selected_script`/`script_audio`/`scripts_with_audio` (task-16852),
        a new `scripts` list is a real row-SET change -- it changes what
        the scripts table itself must show, so it stays scoped to the
        WHOLE detail region.
        """
        self._refresh_detail_region()

    def watch_script_audio(self) -> None:
        """Task-16852: scoped to `ScriptDetailRegion` alone -- `script_
        audio` never appears on the scripts table (only `scripts_with_
        audio`, below, does), so refreshing the whole detail region for it
        would rebuild that table for no reason."""
        self._refresh_script_detail_region()

    def watch_scripts_with_audio(self) -> None:
        """Review fix, task-16852: patch the scripts table's Audio column
        in place -- do NOT tear down the whole detail region.

        Unlike `scripts`/`citations` (real row-SET changes: rows are added
        or removed, so the table genuinely needs rebuilding), `scripts_
        with_audio` never adds or removes a script row -- it is only ever
        consulted inside `_script_row_cells` for the Audio column's own
        cell (`self.scripts_with_audio.get(row.get("id"))`), a per-CELL
        change on rows the table already has. A first synthesis for the
        selected script is exactly the pane's own primary action on a
        script, landing right after a selection this task already made
        survive -- so tearing the table down here would silently reopen
        the destroy-rebuild defect this task exists to close, on the
        feature's own happy path (found in review).

        Repaints EVERY row (not just the ones whose status literally
        changed): `_restyle_script_row` already recomputes each row's
        cells from current state, so repainting every row costs one cheap
        `update_cell` call per cell and cannot drift from a rebuild --
        cheaper and simpler than diffing which ids `scripts_with_audio`
        actually touched, and provably identical output either way, since
        both paths share `_script_row_cells`.

        The Synthesize/Play/Stop toolbar is NOT touched here: it renders
        from `script_audio` (the SELECTED script's own newest render),
        never from `scripts_with_audio` (every script's newest STATUS,
        table-only) -- `compose_script_detail`/`_script_detail_renderable`
        never read `scripts_with_audio` at all. `watch_script_audio`
        already refreshes `ScriptDetailRegion` on its own, independently,
        whenever a synthesis lands new playback state.
        """
        try:
            table = self.query_one("#artifacts-scripts-table", DataTable)
        except NoMatches:
            return
        selected_key = (
            str(self.selected_script.get("id")) if self.selected_script else None
        )
        for row in self.scripts:
            row_key = str(row.get("id"))
            style = self._SELECTED_ROW_STYLE if row_key == selected_key else ""
            self._restyle_script_row(table, row_key, style)

    def watch_citations(self) -> None:
        self._refresh_detail_region()

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
        elif button_id == "artifacts-keep-button":
            self.post_message(KeepBriefingRequested())
        elif button_id == "artifacts-kept-briefings-button":
            self.post_message(KeptBriefingsRequested())
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
        elif button_id == "artifacts-export-feed-button":
            self.post_message(ExportFeedRequested())
        elif button_id == "artifacts-serve-feed-button":
            self.post_message(ServeFeedRequested())
        elif button_id == "artifacts-stop-feed-button":
            self.post_message(StopFeedServerRequested())
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
        elif select.id == "artifacts-cadence-select":
            self.post_message(BriefingCadenceChanged(event.value))
