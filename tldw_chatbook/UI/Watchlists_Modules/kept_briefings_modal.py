"""Kept briefings modal (task-1780, Task 5).

`briefing_keep.keep_briefing` (Task 2) copies a briefing -- and its complete
scripts -- into ChaChaNotes' `kept_briefings`/`kept_scripts` tables, so it
survives the source watchlist's deletion. Nothing on this screen let a
person actually LOOK at what was kept until this task: this modal is that
surface -- list every kept briefing (newest-kept first), render the
selected one's body, list its kept scripts underneath, hard-delete with
confirmation, and cast a brand-new script directly from the kept text via
`briefing_cast.generate_script_from_text` (Task 4).

Modelled on this stream's own modal-editor idiom
(`briefing_preset_modal.BriefingPresetModal`): `ModalScreen`, a compact
button-per-row list down one side (task-895's own "one button per
candidate, not a `Select`" convention -- a `Select` posts `Changed` on
mount, and this modal's kept-list selection has no reason to fight that),
a detail region down the other, inline errors (`Text`, never markup-parsed),
and a confirmation dialog before any destructive write.

**Deliberately scope-independent.** Unlike every other Artifacts surface,
this modal is reachable whenever a live ChaChaNotes handle exists --
`WatchlistsCollectionsScreen.handle_kept_briefings_requested` gates it on
nothing else. Kept content is ChaChaNotes content: it has already survived
whatever happened to the watchlist that produced it, so requiring a
watchlist still be in scope to even LOOK at it would defeat half the
feature's own point (spec: "reachable regardless of watchlist scope,
including after the watchlist is gone").

**Two databases, two different roles.** `chacha_db` (a `CharactersRAGDB`)
owns every kept row this modal reads or writes -- listing, the selected
detail, its kept scripts, and the hard delete. `subs_db` (a
`SubscriptionsDB`, may be `None` if the watchlist bundle service itself is
unavailable) is consulted ONLY to list `briefing_presets` for the cast
picker and to resolve whichever preset a cast names --
`generate_script_from_text` never touches `subs_db` at all for an
app-default cast (`preset_id=None`), which is what lets this modal still
function with `subs_db=None`: the preset `Select` then offers only "App
default (single narrator)", and casting still works.

**Cast's own in-flight guard is THIS modal's, not a copy of the screen's.**
`_cast_in_flight` blocks a second Cast press from this modal instance while
one is running -- mirroring how `WatchlistsCollectionsScreen._cast_in_
flight`/`_briefing_in_flight` are screen-global rather than per-target
(the same "one at a time, from here" convention, not a stricter one).
`generate_script_from_text` ALSO holds its own claim, keyed by
`kept_briefing_id` (`briefing_cast._claim_kept_cast`) -- a completely
separate id space from `Subscriptions_DB` `briefings.id`, so this modal's
cast can never collide with a live SCREEN cast of an unrelated live
briefing that happens to share the same small integer.

**`GenerationInFlightError` gets its own, specific toast** (the phase-4
lesson this stream keeps re-learning): it is `generate_script_from_text`'s
own honest, safe-to-show-verbatim refusal ("a script is already being cast
for kept briefing N"), caught and shown BEFORE the generic `ScriptCastError`
branch, which is itself shown before a bare `Exception` falls back to a
generic message -- never the other way around, and never folded into the
generic branch.

Every remote-derived string (a watchlist's own name, denormalized onto the
kept row; the briefing's body) is rendered through `rich.text.Text`
(never a bare `str`, never `Text.from_markup`) or `rich.markdown.Markdown(
..., hyperlinks=False)` -- the same two rules `ArtifactsPane`'s own module
docstring states for exactly the same reason: this content was written by
an LLM from feed/site material neither this app nor the user chose, so a
markup-shaped fragment in it must paint literally, and a link in it must
never carry an invisible destination.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from loguru import logger
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.markup import escape as escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Select, Static

from ...Subscriptions.briefing_cast import ScriptCastError, generate_script_from_text
from ...Subscriptions.briefing_service import GenerationInFlightError
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.prune_safe_select import PruneSafeSelect

# A kept briefing's body is the identical model output `ArtifactsPane`
# already refuses to hyperlink -- see that module's own `_MARKDOWN_
# HYPERLINKS` comment for the full reasoning (an attacker-chosen label over
# a destination the reader cannot see). The same refusal applies here,
# unchanged, since a kept body is a verbatim copy of that same text.
_MARKDOWN_HYPERLINKS = False

#: The cast preset picker's "no preset" choice. `None` is a REAL option
#: value here (never `Select.NULL`) -- same convention as `ArtifactsPane.
#: _APP_DEFAULT_PRESET_LABEL` for the identical reason: with
#: `allow_blank=False` and `None` present among the option values, `None`
#: is a legal, distinct selection, never confused with "nothing chosen".
#: The copy names the roster a `preset_id=None` cast actually uses
#: (`briefing_cast._APP_DEFAULT_ROSTER`, a single "Narrator" speaker) --
#: a carried decision from the design spec/plan, not new wording invented
#: here.
_APP_DEFAULT_CAST_LABEL = "App default (single narrator)"

_NO_KEPT_BRIEFINGS = "No kept briefings yet. Press Keep on a completed briefing to add one."
_NO_SELECTION = "Select a kept briefing to read it."

#: Mirrors `ArtifactsPane._SCRIPT_UNREADABLE_TURNS`/`_SCRIPT_NO_TURNS` --
#: this module defines its OWN copy of the same idea rather than importing
#: a UI-internal helper from a sibling pane module (`briefing_cast.py`/
#: `briefing_service.py` already establish this repo's own precedent of
#: each module owning its own small vocabulary rather than cross-importing
#: another module's private helpers).
_SCRIPT_UNREADABLE_TURNS = "This script recorded turns that could not be read."
_SCRIPT_NO_TURNS = "This script recorded no turns."
_NO_KEPT_SCRIPTS = "No scripts cast from this kept briefing yet."

#: Same overflow-honesty cap as `ArtifactsPane._TURN_RENDER_CAP`.
_TURN_RENDER_CAP = 200

#: Task-1780 whole-branch review, FIX 4: this modal's kept-briefings list
#: had no paging and no overflow signal at all -- `list_kept_briefings`'s
#: own default `limit=200` would silently drop the 201st-oldest-kept row
#: with nothing on screen to say so. Full pagination is not required here
#: (there is no "load more" affordance, and the spec never asked for one);
#: the cheapest honest fix is to fetch one row past this cap -- enough to
#: know whether more exist -- and say so plainly when they do. A bare
#: module global (not a class attribute) so a test can shrink it via
#: `monkeypatch.setattr(kbm_module, "_KEPT_LIST_DISPLAY_CAP", ...)` without
#: seeding hundreds of rows.
_KEPT_LIST_DISPLAY_CAP = 200
_KEPT_LIST_OVERFLOW_TEXT = "…and more kept briefings not shown."

#: Same bound, same reasoning, for one kept briefing's own kept-scripts list.
_KEPT_SCRIPTS_DISPLAY_CAP = 200
_KEPT_SCRIPTS_OVERFLOW_TEXT = "…and more kept scripts not shown."


def _kept_list_label(row: Mapping[str, Any]) -> Text:
    """One kept-briefing list button's label.

    `watchlist_name` is denormalized, user-authored text (the watchlist may
    already be gone) -- built with `Text.append` per span, never a bare
    `str` or an f-string handed to `Static`/`Button`, so a markup-shaped
    name paints literally instead of being parsed or silently swallowing an
    unclosed tag (`ArtifactsPane.compose`'s own precedent for exactly this
    class of bug).
    """
    label = Text()
    label.append(str(row.get("watchlist_name") or "Unknown watchlist"))
    label.append(" · ", style="dim")
    label.append(str(row.get("kept_at") or "—"), style="dim")
    return label


def _kept_detail_renderable(kept: Mapping[str, Any] | None, has_any: bool) -> RenderableType:
    """What the detail region shows for the selected kept briefing.

    Args:
        kept: The selected `kept_briefings` row, or `None`.
        has_any: Whether any kept briefing exists at all -- distinguishes
            "nothing selected yet" from "there is nothing to select"
            (mirrors `ArtifactsPane._detail_renderable`'s identical split).
    """
    if kept is None:
        return Text(_NO_SELECTION if has_any else _NO_KEPT_BRIEFINGS)

    header = Text()
    header.append(str(kept.get("watchlist_name") or "Unknown watchlist"), style="bold")
    header.append(" · ")
    header.append(str(kept.get("origin") or "unknown origin"))
    header.append("\n")
    header.append(f"Kept {kept.get('kept_at') or 'at an unknown time'}", style="dim")
    model_used = kept.get("model_used")
    if model_used:
        header.append(" · ")
        header.append(str(model_used), style="dim")
    header.append("\n")

    body = str(kept.get("body_markdown") or "").strip()
    if not body:
        # `keep_briefing` refuses an empty body before ever writing a row
        # (see `briefing_keep.KeepRefused`), so this is unreachable through
        # any shipped write path -- kept anyway as an honest fallback
        # rather than a silent blank pane, matching `ArtifactsPane._detail_
        # renderable`'s identical "cannot be produced, say so anyway" rule.
        return Group(header, Text("This kept briefing recorded no body."))
    # `Markdown` parses CommonMark, not Rich markup -- `[bold red]x[/]` is
    # merely link-shaped text to it, exactly like `ArtifactsPane._detail_
    # renderable`'s own use of the same renderable.
    return Group(header, Markdown(body, hyperlinks=_MARKDOWN_HYPERLINKS))


def _kept_script_turns_renderable(turns_json: str | None) -> Text:
    """A kept script's turns as speaker-labelled `Text` lines.

    Mirrors `ArtifactsPane._script_turns_renderable` exactly (see that
    function's own docstring for the full reasoning) -- duplicated rather
    than imported, since it is a private helper of a sibling UI module.
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


def _kept_script_renderable(script: Mapping[str, Any]) -> RenderableType:
    """One kept script's full render: header, then its turns."""
    header = Text()
    header.append(str(script.get("preset_name") or "Untitled preset"), style="bold")
    model_used = script.get("model_used")
    if model_used:
        header.append(" · ")
        header.append(str(model_used))
    header.append("\n")
    header.append(f"Kept {script.get('kept_at') or 'at an unknown time'}", style="dim")
    header.append("\n")
    return Group(header, _kept_script_turns_renderable(script.get("turns_json")))


class KeptBriefingsModal(ModalScreen[None]):
    """List, inspect, cast from, and hard-delete kept briefings.

    Always dismisses `None`: unlike `BriefingPresetModal` (whose dismiss
    value tells the screen whether to reload a cached preset list), this
    screen holds no kept-briefing state of its own to refresh -- every read
    here goes straight through `chacha_db`/`subs_db`, so there is nothing
    for a caller to reload afterward.

    Args:
        chacha_db: An open `CharactersRAGDB` -- the one writer/reader of
            `kept_briefings`/`kept_scripts`. All calls go through
            `asyncio.to_thread`.
        subs_db: An open `SubscriptionsDB`, or `None` if the watchlist
            bundle service is unavailable. Consulted only for `list_
            briefing_presets` (the cast picker's options) and passed
            through to `generate_script_from_text` to resolve whichever
            preset a cast names. `None` still lets casting work for the
            app-default (no-preset) roster -- see the module docstring.
        load_character: Character-card lookup by id, or `None` if
            unavailable -- passed straight through to `generate_script_
            from_text`'s own parameter of the same name (see that
            function's docstring). Matches `WatchlistsCollectionsScreen.
            _cast_load_character`'s shape exactly, so the screen can hand
            over the identical callable it already built for its own live
            cast path.
    """

    BINDINGS = [("escape", "close", "Close")]

    def __init__(
        self,
        chacha_db: Any,
        *,
        subs_db: Any | None = None,
        load_character: Any = None,
    ) -> None:
        super().__init__()
        self.chacha_db = chacha_db
        self.subs_db = subs_db
        self._load_character = load_character
        self._kept: list[dict[str, Any]] = []
        self._kept_overflow = False
        self._selected_kept_id: int | None = None
        self._scripts: list[dict[str, Any]] = []
        self._scripts_overflow = False
        self._presets: list[dict[str, Any]] = []
        self._cast_preset_id: int | None = None
        self._cast_in_flight = False
        self._delete_in_flight = False
        # Held here, not read only from the live mounted widget, because
        # `_run_cast`'s `finally` ALWAYS recomposes (to repaint the Cast
        # button's own `disabled` state and, on success, the refreshed
        # scripts list) -- a compose() that hard-coded `""` for `#kbm-error`
        # would silently wipe out whatever `_show_error` had just written
        # to the OLD, about-to-be-discarded widget. `compose()` reads this
        # attribute, so an error painted right before a recompose survives
        # it.
        self._error_text = ""

    # --- Compose ---------------------------------------------------------

    def _selected_kept(self) -> dict[str, Any] | None:
        return next(
            (row for row in self._kept if int(row["id"]) == self._selected_kept_id),
            None,
        )

    def _preset_select_options(self) -> list[tuple[str, int | None]]:
        """Options for the cast preset picker: app-default, then every
        loaded preset, name-ASC (already `list_briefing_presets`'s own
        order). Mirrors `ArtifactsPane._preset_select_options`'s stale-id
        fallback: a preset chosen earlier this session but hard-deleted
        since (from `BriefingPresetModal`, elsewhere) gets a synthetic
        trailing option instead of raising `InvalidSelectValueError` the
        moment `compose` builds `value=self._cast_preset_id`.
        """
        options: list[tuple[str, int | None]] = [(_APP_DEFAULT_CAST_LABEL, None)]
        known_ids: set[int] = set()
        for preset in self._presets:
            preset_id = preset.get("id")
            if preset_id is None:
                continue
            known_ids.add(preset_id)
            options.append(
                (str(preset.get("name") or f"Preset {preset_id}"), preset_id)
            )
        if self._cast_preset_id is not None and self._cast_preset_id not in known_ids:
            options.append(
                (f"Preset {self._cast_preset_id} (deleted)", self._cast_preset_id)
            )
        return options

    def compose(self) -> ComposeResult:
        with Vertical(id="kbm-dialog"):
            yield Static("Kept briefings", id="kbm-title")
            with Horizontal(id="kbm-body"):
                with Vertical(id="kbm-list-column"):
                    yield Static("Kept", classes="kbm-column-heading")
                    with VerticalScroll(id="kbm-kept-list"):
                        if self._kept:
                            for row in self._kept:
                                kept_id = int(row["id"])
                                selected = kept_id == self._selected_kept_id
                                yield Button(
                                    _kept_list_label(row),
                                    id=f"kbm-kept-btn-{kept_id}",
                                    compact=True,
                                    variant="primary" if selected else "default",
                                )
                            if self._kept_overflow:
                                yield Static(
                                    Text(_KEPT_LIST_OVERFLOW_TEXT, style="dim"),
                                    id="kbm-kept-list-overflow",
                                )
                        else:
                            yield Static(
                                "No kept briefings yet.", id="kbm-kept-list-empty"
                            )
                with VerticalScroll(id="kbm-detail-column"):
                    kept = self._selected_kept()
                    yield Static(
                        _kept_detail_renderable(kept, bool(self._kept)),
                        id="kbm-detail",
                    )
                    yield Static(
                        Text(self._error_text) if self._error_text else "",
                        id="kbm-error",
                    )
                    with Horizontal(id="kbm-detail-actions"):
                        yield Button(
                            "Delete",
                            id="kbm-delete-button",
                            variant="error",
                            disabled=(
                                kept is None
                                or self._cast_in_flight
                                or self._delete_in_flight
                            ),
                        )

                    if kept is not None:
                        # Casting is an action on the SELECTED kept
                        # briefing, so -- exactly like `ArtifactsPane`'s
                        # own Cast section gating on `selected_briefing` --
                        # there is nothing for it to act on without one.
                        with Horizontal(
                            id="kbm-cast-toolbar",
                            classes="destination-filter-strip",
                        ):
                            # UAT batch-5 review, m1: was tooltip-only --
                            # the same "bare value, hover-only meaning"
                            # pattern task-2310 removed from the Artifacts
                            # pane's own, structurally identical
                            # `#artifacts-preset-select`. Same label idiom.
                            yield Static("Preset", classes="watchlists-inline-select-label")
                            yield PruneSafeSelect(
                                self._preset_select_options(),
                                value=self._cast_preset_id,
                                id="kbm-preset-select",
                                allow_blank=False,
                                compact=True,
                                tooltip=(
                                    "The preset to cast this kept briefing "
                                    "with."
                                ),
                            )
                            yield Button(
                                "Cast",
                                id="kbm-cast-button",
                                compact=True,
                                disabled=self._cast_in_flight or self._delete_in_flight,
                                tooltip=(
                                    "A cast is already running."
                                    if self._cast_in_flight
                                    else "A delete is in progress."
                                    if self._delete_in_flight
                                    else "Cast a new script from this kept "
                                    "briefing's text."
                                ),
                            )
                        yield Static("Kept scripts", classes="kbm-column-heading")
                        if self._scripts:
                            for script in self._scripts:
                                yield Static(
                                    _kept_script_renderable(script),
                                    classes="kbm-script",
                                )
                            if self._scripts_overflow:
                                yield Static(
                                    Text(_KEPT_SCRIPTS_OVERFLOW_TEXT, style="dim"),
                                    id="kbm-scripts-overflow",
                                )
                        else:
                            yield Static(
                                _NO_KEPT_SCRIPTS, id="kbm-scripts-empty"
                            )
            with Horizontal(id="kbm-actions"):
                yield Button("Close", id="kbm-close")

    async def on_mount(self) -> None:
        await self._load_kept()
        await self._load_presets()
        if self.is_attached:
            self.refresh(recompose=True)

    # --- Loading -----------------------------------------------------------

    async def _load_kept(self) -> None:
        """Re-read every kept briefing, newest-kept first.

        Guarded on `is_attached`, not `is_mounted` -- see `BriefingPreset
        Modal._load_presets`'s docstring for why: awaited directly from
        `on_mount`, before `is_mounted` itself has flipped `True`.

        Fetches one row past `_KEPT_LIST_DISPLAY_CAP` -- not the whole
        table -- purely to know honestly whether more rows exist than are
        shown; see that constant's own comment (task-1780 whole-branch
        review, FIX 4).
        """
        try:
            rows = await asyncio.to_thread(
                self.chacha_db.list_kept_briefings,
                limit=_KEPT_LIST_DISPLAY_CAP + 1,
            )
        except Exception as exc:  # noqa: BLE001 - degrade the list, not the modal
            logger.warning(f"Failed to list kept briefings: {type(exc).__name__}")
            rows = []
        self._kept_overflow = len(rows) > _KEPT_LIST_DISPLAY_CAP
        self._kept = [dict(row) for row in rows[:_KEPT_LIST_DISPLAY_CAP]]

    async def _load_presets(self) -> None:
        """Re-read every stored `briefing_presets` row for the cast picker.

        Degrades to an empty list (offering only the app-default option)
        when `subs_db` is unbound or the read fails -- never the whole
        modal; see the module docstring's own note on why `subs_db=None`
        must still leave the modal usable.
        """
        if self.subs_db is None:
            self._presets = []
            return
        try:
            rows = await asyncio.to_thread(self.subs_db.list_briefing_presets)
        except Exception as exc:  # noqa: BLE001 - degrade the field, not the modal
            logger.warning(f"Failed to list briefing presets: {type(exc).__name__}")
            rows = []
        self._presets = [dict(row) for row in rows]

    async def _load_scripts_and_refresh(self, kept_id: int) -> None:
        """Fetch `kept_id`'s scripts, then repaint -- unless the selection
        moved on before this landed.

        The ownership check mirrors `BriefingPresetModal`'s own
        post-`await` verification: `_select_kept` already cleared `_scripts`
        synchronously (so a stale list never shows under the NEW
        selection, the same fix `ArtifactsPane`'s own briefing-switch test
        pins), but a second, faster selection made while this fetch is
        still in flight must not have its own fresh, correct list clobbered
        by this older fetch landing after it.
        """
        try:
            rows = await asyncio.to_thread(
                self.chacha_db.list_kept_scripts,
                kept_id,
                limit=_KEPT_SCRIPTS_DISPLAY_CAP + 1,
            )
        except Exception as exc:  # noqa: BLE001 - degrade the list, not the modal
            logger.warning(
                f"Failed to list kept scripts for {kept_id}: {type(exc).__name__}"
            )
            rows = []
        if self._selected_kept_id != kept_id:
            return
        self._scripts_overflow = len(rows) > _KEPT_SCRIPTS_DISPLAY_CAP
        self._scripts = [dict(row) for row in rows[:_KEPT_SCRIPTS_DISPLAY_CAP]]
        if self.is_attached:
            self.refresh(recompose=True)

    # --- Selection -----------------------------------------------------

    def _select_kept(self, kept_id: int) -> None:
        if kept_id == self._selected_kept_id:
            return
        self._selected_kept_id = kept_id
        # Cleared synchronously, before the reload lands -- otherwise the
        # PREVIOUS kept briefing's scripts would still show, attached to
        # the wrong selection, for however long the fetch below takes
        # (mirrors `ArtifactsPane.select_briefing_by_id`'s identical
        # clear-before-reload contract for its own scripts table).
        self._scripts = []
        self._clear_error()
        self.refresh(recompose=True)
        self.run_worker(
            self._load_scripts_and_refresh(kept_id), group="kbm-load-scripts"
        )

    # --- Error surface -----------------------------------------------------

    def _show_error(self, message: str) -> None:
        """Record `message` as plain text -- never markup-parsed -- and
        paint it immediately if mounted.

        Every message this modal shows here originates from a provider, a
        cast/keep exception's own safe `str(exc)`, or a bare exception
        type name -- never trusted, so this must stay `Text`-safe exactly
        like `BriefingPresetModal._show_error`. Unlike that sibling,
        THIS modal's callers (`_run_cast`) always recompose in a `finally`
        right after calling this -- so `self._error_text` is the value of
        record `compose()` itself reads; the live `.update()` below is a
        courtesy for the (rare) case nothing recomposes afterward.
        """
        self._error_text = message
        if self.is_mounted:
            self.query_one("#kbm-error", Static).update(Text(message))

    def _clear_error(self) -> None:
        self._error_text = ""
        if self.is_mounted:
            self.query_one("#kbm-error", Static).update("")

    # --- Delete --------------------------------------------------------

    def _dispatch_delete(self) -> None:
        """Claim the delete guard, then dispatch -- before `run_worker`,
        for the identical reason every in-flight guard on this stream is
        claimed at dispatch time rather than inside the worker body (see
        `WatchlistsCollectionsScreen.handle_generate_briefing_requested`'s
        docstring for the canonical statement of why).

        A second press while either guard already holds is now refused
        WITH a toast -- task-1780 whole-branch review (FIX 2) found this
        branch silently did nothing at all, unlike the screen's own Keep
        (`_keep_briefing`'s "A keep is already in progress." toast). No
        selection (`_selected_kept_id is None`) stays silent on purpose:
        the Delete button is `disabled` whenever nothing is selected (see
        `compose`), so that branch is unreachable through the UI -- there
        is nothing a toast could usefully explain there.
        """
        if self._selected_kept_id is None:
            return
        if self._delete_in_flight or self._cast_in_flight:
            self.notify(
                "A delete is already in progress. Nothing else was started."
                if self._delete_in_flight
                else "A cast is in progress. Nothing else was started.",
                severity="warning",
                markup=False,
            )
            return
        self._delete_in_flight = True
        self.run_worker(self._run_delete(), exclusive=True, group="kbm-delete")

    async def _run_delete(self) -> None:
        """Dispatch worker body: delete, then always clear the guard and repaint.

        Task-1780 whole-branch review (FIX 1, Important): this coroutine
        used to be a bare `try/finally` with no `except` at all, but
        `_handle_delete` has two `await`s that can each raise -- the
        confirmation dialog itself (`push_screen_wait`) and the hard
        delete (`asyncio.to_thread(self.chacha_db.delete_kept_briefing,
        ...)`, a real `SQLITE_BUSY`/`CharactersRAGDBError` mid-delete is no
        longer a theoretical concern once auto-keep (Task 3) is writing
        ChaChaNotes concurrently from the scheduler). A Textual worker's
        default `exit_on_error=True` means ANY exception escaping this
        coroutine took the WHOLE APPLICATION down, not just this modal.

        `asyncio.CancelledError` is re-raised rather than caught -- mirrors
        `_write_briefing_export_file`/`_export_feed_to_directory`
        (`watchlists_collections_screen.py`) exactly: a cancelled worker
        must never be reported as a failed delete. Every other exception
        gets a toast (`markup=False` -- the kept briefing's own name is
        not this app's text to interpret as markup; the message itself is
        only a bare exception type name, never exception content) instead
        of exiting the app. The guard clears in `finally` on every path,
        so a failed delete never wedges Delete shut for the rest of this
        modal's life; the kept list itself is left unrefreshed (not
        reloaded past the failed `to_thread` call) but stays consistent --
        it still shows exactly what ChaChaNotes still has, since the
        delete never actually completed.
        """
        try:
            await self._handle_delete()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a worker crash exits the app
            logger.warning(f"Kept-briefing delete failed: {type(exc).__name__}")
            self.notify(
                f"Could not delete this kept briefing: {type(exc).__name__}",
                severity="error",
                markup=False,
            )
        finally:
            self._delete_in_flight = False
            if self.is_attached:
                self.refresh(recompose=True)

    async def _handle_delete(self) -> None:
        if self._selected_kept_id is None:
            return
        # Snapshotted before either `await` below (the confirmation
        # dialog, then the delete itself) -- mirrors `BriefingPresetModal.
        # _handle_delete`'s identical snapshot-before-await discipline.
        target_kept_id = self._selected_kept_id
        kept = self._selected_kept()
        name = str((kept or {}).get("watchlist_name") or "this kept briefing")
        confirmed = await self.app.push_screen_wait(
            ConfirmationDialog(
                title="Delete kept briefing",
                message=(
                    f'Delete the kept briefing from "{escape_markup(name)}"? '
                    "This also deletes its kept scripts, and cannot be "
                    "undone."
                ),
                confirm_label="Delete",
                cancel_label="Cancel",
            )
        )
        if not confirmed:
            return
        await asyncio.to_thread(self.chacha_db.delete_kept_briefing, target_kept_id)
        if self._selected_kept_id == target_kept_id:
            self._selected_kept_id = None
            self._scripts = []
        self._clear_error()
        await self._load_kept()

    # --- Cast ------------------------------------------------------------

    def _dispatch_cast(self) -> None:
        if self._selected_kept_id is None:
            # The Cast toolbar only composes when a kept briefing is
            # selected (see `compose`) -- unreachable through the UI, so
            # this stays silent rather than toasting about nothing.
            return
        if self._cast_in_flight or self._delete_in_flight:
            # `_delete_in_flight` closes a narrow race: a delete's own
            # confirmation dialog blocks other clicks while it is up, but
            # the delete ITSELF (`delete_kept_briefing`) has one `await`
            # after the dialog resolves, during which this modal is again
            # the top screen and its own buttons are clickable. A cast that
            # started in that window and outlived the delete would try to
            # `create_kept_script` against a `kept_briefing_id` the
            # foreign key no longer has -- `_dispatch_delete`'s own mirror
            # check (`_cast_in_flight`) closes the identical window in the
            # other direction.
            #
            # Task-1780 whole-branch review (FIX 2): both refusals now
            # toast -- this branch used to be a silent no-op, unlike the
            # screen's own Keep ("A keep is already in progress.").
            self.notify(
                "A cast is already in progress. Nothing else was started."
                if self._cast_in_flight
                else "A delete is in progress. Nothing else was started.",
                severity="warning",
                markup=False,
            )
            return
        self._cast_in_flight = True
        target_kept_id = self._selected_kept_id
        preset_id = self._cast_preset_id
        self.run_worker(
            self._run_cast(target_kept_id, preset_id),
            exclusive=True,
            group="kbm-cast",
        )

    async def _run_cast(self, kept_id: int, preset_id: int | None) -> None:
        try:
            try:
                await generate_script_from_text(
                    self.chacha_db,
                    kept_id,
                    preset_id=preset_id,
                    subs_db=self.subs_db,
                    load_character=self._load_character,
                )
            except GenerationInFlightError as exc:
                # This function's OWN specific, safe-to-show refusal --
                # shown verbatim, and caught BEFORE the generic
                # `ScriptCastError`/bare-`Exception` branches below (the
                # phase-4 lesson this stream keeps re-learning: a caller
                # that lets this fall into a generic "database
                # unreachable" toast has hidden real, useful information
                # behind a false one).
                self._show_error(str(exc))
                return
            except ScriptCastError as exc:
                # `generate_script_from_text`'s own pre-flight refusal
                # (missing kept briefing, empty body, missing preset, an
                # invalid roster) or in-band failure (unknown speaker,
                # malformed reply, missing character card) -- every raise
                # site names the specific defect, safe to show verbatim.
                self._show_error(str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - a worker crash exits the app
                logger.warning(
                    f"Kept-briefing cast failed for {kept_id}: {type(exc).__name__}"
                )
                self._show_error(
                    f"Could not cast a script: {type(exc).__name__}"
                )
                return
            # Success: only repaint if the selection is still THIS kept
            # briefing -- switching away mid-cast must not have an older
            # cast's landing silently repaint whatever is now on screen
            # (mirrors `BriefingPresetModal`'s own post-await ownership
            # check for `_editing_id`).
            if self._selected_kept_id == kept_id:
                self._clear_error()
                await self._load_scripts_and_refresh(kept_id)
        finally:
            self._cast_in_flight = False
            if self.is_attached:
                self.refresh(recompose=True)

    # --- Dismiss protocol ----------------------------------------------

    def action_close(self) -> None:
        self.dismiss(None)

    # --- Event routing ---------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = str(event.button.id or "")
        if button_id == "kbm-close":
            self.dismiss(None)
        elif button_id == "kbm-delete-button":
            self._dispatch_delete()
        elif button_id == "kbm-cast-button":
            self._dispatch_cast()
        elif button_id.startswith("kbm-kept-btn-"):
            kept_id = int(button_id[len("kbm-kept-btn-") :])
            self._select_kept(kept_id)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Absorb the cast preset picker's own mount-time `Changed`, then
        report every one after it.

        Instance-keyed, NOT value-equality -- the exact idiom `ArtifactsPane
        .on_select_changed` states in full (a freshly composed `Select`
        posts exactly one `Changed` from its own mount, unconditionally;
        comparing against the picker's CURRENT value cannot tell that
        mount-time noise apart from a real pick once a recompose has
        already moved the current value on). Push-, not pull-, style is
        deliberate here (unlike `BriefingPresetModal`'s own text `Input`s):
        this modal recomposes on every kept-briefing switch AND after every
        successful cast, and the chosen preset should SURVIVE both -- a
        pull-only read (only at Cast-press time) would lose that choice the
        moment either recompose rebuilt a brand-new `Select` from whatever
        `self._cast_preset_id` last was.
        """
        event.stop()
        select = event.select
        if select.id != "kbm-preset-select":
            return
        if not getattr(select, "_kbm_preset_mount_absorbed", False):
            select._kbm_preset_mount_absorbed = True
            return
        self._cast_preset_id = event.value
