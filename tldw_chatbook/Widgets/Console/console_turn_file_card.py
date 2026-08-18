"""Stacked per-file change card for one agent turn (turn file card spec).

Pure presentation over TASK-1972's snapshots: the card receives the
marker text (counts precomputed at emit), a run id, and a ZERO-ARG
provider factory (late-binding, the transcript-region builder
convention). File rows load asynchronously on mount; each row's diff
loads asynchronously on FIRST expand and is cached. Every shadow-repo
read runs off the UI thread. A provider failure of any kind degrades to
the marker header alone -- the card must never break the transcript.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Click, Key, Resize
from textual.message import Message
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_display_state import (
    DiffHunk,
    TurnFileEntry,
    hunk_excerpt,
    middle_elide_path,
    split_unified_diff,
    turn_file_entries,
)
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph

_CHEVRON_CLOSED = "▸"
_CHEVRON_OPEN = "▾"

#: Note text cap (TASK-16800 spec §1) -- matches the `Input`'s own
#: `max_length` so the widget-level typing limit and this boundary check
#: can never drift apart.
NOTE_MAX_LENGTH = 2000


def _validate_note_text(raw: str) -> "str | None":
    """Strip and bound-check a note before it reaches the DB.

    The strip-then-empty check runs first because
    ``input_validation.validate_text_input`` treats an empty string as
    valid (it only rejects *oversized*/dangerous text) -- this widget
    boundary additionally rejects a note that is empty after stripping.

    Args:
        raw: The raw ``Input.value`` text.

    Returns:
        The stripped note text, or ``None`` when it is empty or fails
        validation (over ``NOTE_MAX_LENGTH`` chars, or a dangerous
        HTML/script pattern -- defense in depth, since the excerpt is
        later embedded literally in the delivery block).
    """
    text = (raw or "").strip()
    if not text:
        return None
    if not validate_text_input(text, max_length=NOTE_MAX_LENGTH):
        return None
    return text


class ConsoleTurnFileCard(Vertical):
    """Stacked, expandable per-file diff card rendered under a turn marker."""

    DEFAULT_CSS = """
    ConsoleTurnFileCard {
        height: auto;
        min-height: 1;
    }
    ConsoleTurnFileCard .console-turn-file-header-row {
        height: 1;
        width: 100%;
    }
    ConsoleTurnFileCard .console-turn-file-header {
        width: 1fr;
        height: 1;
        text-style: bold;
    }
    ConsoleTurnFileCard .console-turn-file-toggle-all-btn,
    ConsoleTurnFileCard .console-turn-file-review-btn {
        min-width: 3;
    }
    ConsoleTurnFileCard .console-turn-file-row {
        height: 1;
        min-height: 1;
        width: 100%;
        text-align: left;
    }
    ConsoleTurnFileCard .console-turn-file-diff {
        height: auto;
        max-height: 20;
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-size: 1 1;
    }
    ConsoleTurnFileCard .console-turn-file-hunk {
        height: auto;
        margin-bottom: 1;
    }
    ConsoleTurnFileCard .console-turn-file-hunk-actions {
        height: auto;
    }
    ConsoleTurnFileCard .console-turn-file-note-btn {
        min-width: 10;
    }
    ConsoleTurnFileCard .console-turn-file-hunk-notes {
        height: auto;
    }
    ConsoleTurnFileCard .console-turn-file-note {
        height: auto;
        min-height: 1;
    }
    ConsoleTurnFileCard .console-turn-file-note-text {
        width: 1fr;
    }
    ConsoleTurnFileCard .console-turn-file-note-input {
        width: 100%;
    }
    """

    # Selection styling (parity with `.console-transcript-message-selected`)
    # deliberately does NOT live in DEFAULT_CSS: it needs the app bundle's
    # `$ds-focus-bg`/`$ds-focus-fg` tokens, and Textual resolves `$vars`
    # per CSS source, so a local "fallback" here would unconditionally
    # shadow the bundle's values (TASK-16811, caught again by Qodo on
    # PR #1728). The rules live beside the message-selected block in
    # `css/components/_agentic_terminal.tcss`.

    class ReviewRequested(Message):
        """The header ``Review`` button asking to open the Change Review
        screen scoped to this card's run.

        Carries ``run_id`` directly (the card always knows it, unlike the
        plain marker's own "review with `v`" action, which must resolve it
        from the transcript's display model by message id) -- the handler
        can open the screen with no lookup at all.
        """

        def __init__(self, run_id: str) -> None:
            self.run_id = run_id
            super().__init__()

    def __init__(
        self,
        marker_text: str,
        run_id: str,
        provider_factory: Callable[[], Any],
        *,
        message_id: str | None = None,
        selected: bool = False,
        id: str | None = None,
    ) -> None:
        classes = "console-turn-file-card"
        if selected:
            classes += " console-turn-file-card-selected"
        super().__init__(id=id, classes=classes)
        self._marker_text = marker_text
        self._run_id = run_id
        self._provider_factory = provider_factory
        #: The transcript message id this card renders (final-review fix
        #: wave): needed so a click can select the row and
        #: `_update_row_widget` can confirm identity before syncing rather
        #: than rebuilding. ``None`` only for bare unit construction (a
        #: click is then a no-op rather than raising).
        self._message_id = message_id
        self._selected = selected
        self._entries: list[TurnFileEntry] = []
        self._row_for_entry: dict[int, dict] = {}
        #: Segmented hunks per row index, over the FULL diff text (never a
        #: display-truncated slice -- see ``split_unified_diff``). Replaces
        #: the old joined-string cache: styling and the per-hunk display cap
        #: are applied at MOUNT time, from these raw hunks, so re-expanding
        #: a collapsed row never re-derives anything from a lossy cache.
        self._hunk_cache: dict[int, list[DiffHunk]] = {}
        #: Whether the provider exposes the full note read/write/delete
        #: trio -- duck-typed-optional, mirroring how ``turn_for_run`` is
        #: treated: a provider missing any of the three renders NO note UI
        #: at all rather than a partially-working one. Set once, from the
        #: same off-thread read as ``notes_for_run`` itself, in
        #: ``_load_rows``.
        self._notes_capable: bool = False
        #: Existing notes keyed by ``(root, path)`` -- the anchor's file
        #: identity -- populated from ``notes_for_run`` on load and kept in
        #: sync as notes are added/deleted through this card instance.
        self._notes_by_key: dict[tuple[str, str], list[dict]] = {}

    @property
    def marker_text(self) -> str:
        """The marker text this card was built from (identity check for reuse)."""
        return self._marker_text

    @property
    def run_id(self) -> str:
        """The run id this card renders (identity check for reuse)."""
        return self._run_id

    def update_selected(self, selected: bool) -> None:
        """Sync selection styling onto this mounted card without a rebuild.

        Called from ``ConsoleTranscript._update_row_widget`` when only the
        row's selection flipped -- expanded rows and the cached diff text
        must survive a selection change (final-review fix wave).

        Args:
            selected: Whether this card's transcript row is now selected.
        """
        self._selected = selected
        self.set_class(selected, "console-turn-file-card-selected")

    def on_click(self, event: Click) -> None:
        """Clicking the card selects its transcript row (parity with the
        plain marker's ``ConsoleTranscriptMessage.on_click``) -- except a
        file-row button, whose own click already toggles that row's
        expand/collapse and must not also flip transcript selection.

        Args:
            event: The click; its ``control`` distinguishes a file-row
                button press from a click on the card chrome.
        """
        if event.control is not None and event.control.has_class(
            "console-turn-file-row"
        ):
            return
        event.stop()
        if self._message_id is None:
            return
        # Duck-typed walk to the owning ConsoleTranscript rather than an
        # isinstance check: that class lives in console_transcript.py,
        # which already imports THIS module at load time, so a module-level
        # import back here would be circular.
        node = self.parent
        while node is not None:
            toggle = getattr(node, "toggle_message_selection", None)
            if callable(toggle):
                toggle(self._message_id)
                return
            node = node.parent

    def compose(self) -> ComposeResult:
        # Header keeps the marker's counts but drops its "review with v"
        # trailer -- the rows ARE the review affordance now; `v` still
        # works and stays documented in the F1 help. The header also gains
        # two compact, non-destructive affordances (spec §5, AC#3/AC#4): a
        # `Review` button (a mouse-clickable equivalent of `v`, scoped to
        # this card's own run -- no message-id lookup needed, unlike the
        # marker's action) and an expand/collapse-all toggle.
        head = self._marker_text.split(" — ")[0]
        with Horizontal(classes="console-turn-file-header-row"):
            yield Static(
                head,
                classes="console-turn-file-header",
                markup=False,
            )
            toggle_btn = Button(
                f"{resolve_glyph(_CHEVRON_CLOSED)} All",
                classes="console-turn-file-toggle-all-btn",
                compact=True,
                tooltip="Expand every file's diff",
            )
            # Same guard as every other row/note button in this card: a
            # quick second press must never be silently swallowed by the
            # default "-active" flash.
            toggle_btn.active_effect_duration = 0
            yield toggle_btn
            review_btn = Button(
                "Review",
                classes="console-turn-file-review-btn",
                compact=True,
                tooltip="Open the Change Review screen for this turn",
            )
            review_btn.active_effect_duration = 0
            yield review_btn
        yield Vertical(classes="console-turn-file-rows")

    def on_mount(self) -> None:
        """Start the async row load; the header renders immediately."""
        self.run_worker(
            self._load_rows(),
            group="console-turn-file-card-load",
            exit_on_error=False,
        )

    async def _load_rows(self) -> None:
        # The whole prepare-then-mount body lives in one try/except (mirrors
        # ConsoleToolDiffRow._prepare_and_mount in console_transcript.py) so
        # that a failure anywhere -- provider construction, the off-thread
        # read, or the mount calls -- degrades to the marker-only header
        # instead of raising out of this worker.
        try:
            provider = self._provider_factory()
            if provider is None:
                return

            def _read() -> tuple[
                list[TurnFileEntry], dict[int, dict], list[dict], bool
            ]:
                # Run-scoped read when the provider offers it (Qodo,
                # PR #1728): a transcript can hold many cards, and each
                # `turns()` call scans and groups the WHOLE conversation's
                # snapshot history just to keep one run. Fall back to the
                # scan for duck-typed providers without the method.
                turn_for_run = getattr(provider, "turn_for_run", None)
                if callable(turn_for_run):
                    turn = turn_for_run(self._run_id)
                else:
                    turn = next(
                        (t for t in provider.turns() if t.run_id == self._run_id),
                        None,
                    )
                if turn is None:
                    return [], {}, [], False
                # Per-row pairing, never root-keyed: `turn.rows` can hold
                # rows from TWO windows on the SAME root (a turn's own
                # window and its surviving sub-agents' post-turn window,
                # PR3a-1 Task 6c -- both markers carry this run's id). A
                # root-keyed map would collide those rows; pairing each
                # entry to the exact row it came from (by position, via
                # `turn_file_entries`) keeps both windows' files distinct
                # and each one's diff readable against its own row. This
                # deliberately renders the UNION of the run's clean rows --
                # see `turn_file_entries`'s docstring for the ruling.
                row_files = [
                    (row, provider.changed_files(row))
                    for row in turn.rows
                    if not row.get("tracking_error")
                ]
                paired = turn_file_entries(row_files)
                entries = [entry for entry, _row in paired]
                mapping = {idx: row for idx, (_entry, row) in enumerate(paired)}
                # Task 4: note capability is duck-typed-optional, the same
                # posture as `turn_for_run` above -- a provider missing any
                # of the trio renders no note UI at all (checked once here,
                # off-thread, and cached on the instance for every later
                # note-UI decision).
                notes_capable = all(
                    callable(getattr(provider, name, None))
                    for name in (
                        "add_change_note",
                        "delete_change_note",
                        "notes_for_run",
                    )
                )
                notes = provider.notes_for_run(self._run_id) if notes_capable else []
                return entries, mapping, notes, notes_capable

            entries, mapping, notes, notes_capable = await asyncio.to_thread(_read)
            if not self.is_mounted or not entries:
                return
            self._entries = entries
            self._row_for_entry = mapping
            self._notes_capable = notes_capable
            notes_by_key: dict[tuple[str, str], list[dict]] = {}
            for note in notes:
                key = (str(note["root"]), str(note["path"]))
                notes_by_key.setdefault(key, []).append(note)
            self._notes_by_key = notes_by_key
            rows_box = self.query_one(".console-turn-file-rows", Vertical)
            for idx, entry in enumerate(entries):
                row = Button(
                    self._row_label_text(entry, expanded=False),
                    classes="console-turn-file-row",
                    compact=True,
                )
                row.entry_index = idx
                # Button's default 0.2s "-active" flash guards `action_press`
                # against a second Enter/click until the flash clears -- fine
                # for a submit button, but this row toggles open/closed and a
                # quick second press should never be silently swallowed.
                row.active_effect_duration = 0
                # The row's own label is middle-elided to the card's width
                # (spec §5) -- the tooltip always carries the FULL,
                # un-elided path so nothing is lost to the elision.
                row.tooltip = entry.label
                diff_body = VerticalScroll(classes="console-turn-file-diff")
                diff_body.display = False
                await rows_box.mount(row)
                await rows_box.mount(diff_body)
        except Exception:
            logger.opt(exception=True).warning(
                "Turn file card row load failed; keeping marker-only header."
            )
            return

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button
        if button.has_class("console-turn-file-review-btn"):
            event.stop()
            self.post_message(self.ReviewRequested(self._run_id))
            return
        if button.has_class("console-turn-file-toggle-all-btn"):
            event.stop()
            await self._toggle_all(button)
            return
        if button.has_class("console-turn-file-note-btn"):
            event.stop()
            await self._open_note_input(button)
            return
        if button.has_class("console-turn-file-note-delete"):
            event.stop()
            await self._delete_note(button)
            return
        idx = getattr(button, "entry_index", None)
        if idx is None:
            return
        event.stop()
        bodies = list(self.query(".console-turn-file-diff"))
        rows = list(self.query(".console-turn-file-row"))
        # A partial-mount desync (the row-load worker still mid-mount, or a
        # stale button event arriving after a rebuild) could otherwise index
        # past a shorter list here -- an IndexError escaping this `on_*`
        # handler would propagate to `app._handle_exception()` and exit the
        # whole app, exactly the failure class every other seam in this file
        # is guarded against. Degrade to a no-op instead.
        if idx >= len(bodies) or idx >= len(rows) or idx >= len(self._entries):
            return
        body = bodies[idx]
        row = rows[idx]
        entry = self._entries[idx]
        if body.display:
            body.display = False
            row.label = self._row_label_text(entry, expanded=False)
            return
        if idx not in self._hunk_cache:
            snapshot_row = self._row_for_entry.get(idx)
            if snapshot_row is None:
                return
            # Provider construction, the off-thread diff read + segmentation,
            # and the mounts all live in one try/except -- a transient
            # provider-construction failure on first expand must degrade the
            # row (stay collapsed) rather than raise out of this `on_*`
            # handler, which Textual would otherwise propagate to
            # `app._handle_exception()` and exit the whole app.
            try:
                provider = self._provider_factory()
                if provider is None:
                    return
                hunks = await self._read_hunks(provider, snapshot_row, entry)
                self._hunk_cache[idx] = hunks
                if not body.is_mounted:
                    return
                await self._mount_hunk_blocks(body, idx, entry, provider, hunks)
            except Exception:
                logger.opt(exception=True).warning(
                    "Turn file card diff load failed for {}", entry.label
                )
                return
        body.display = True
        row.label = self._row_label_text(entry, expanded=True)

    @staticmethod
    async def _read_hunks(
        provider: Any, snapshot_row: dict, entry: TurnFileEntry
    ) -> list[DiffHunk]:
        """Off-thread read + segmentation of one entry's full diff text.

        Segmentation always runs on the FULL diff text (spec §2) -- never a
        display-truncated slice -- so hunk indices stay stable regardless
        of the per-hunk display cap applied at mount time.

        Args:
            provider: The turn's data source (``diff_text(row, path)``).
            snapshot_row: The ``change_snapshots`` row this entry came from.
            entry: The file entry being expanded.

        Returns:
            The entry's diff, segmented into hunks.
        """

        def _read() -> list[DiffHunk]:
            text = provider.diff_text(snapshot_row, entry.path)
            return split_unified_diff(text)

        return await asyncio.to_thread(_read)

    async def _mount_hunk_blocks(
        self,
        body: VerticalScroll,
        idx: int,
        entry: TurnFileEntry,
        provider: Any,
        hunks: list[DiffHunk],
    ) -> None:
        """Mount one colored ``Static`` + action row + notes box per hunk.

        Shared by a single row's first expand (``on_button_pressed``) and
        expand-all (``_expand_all``) so the two paths can never render a
        row's hunks differently.

        Args:
            body: The row's (already-mounted) diff body container.
            idx: The row's entry index (carried onto each note button).
            entry: The file entry being expanded.
            provider: The turn's data source (only ``diff_display_max_lines``
                is read here).
            hunks: The entry's segmented hunks (already cached by the
                caller).
        """
        cap = int(getattr(provider, "diff_display_max_lines", 2000))
        # Per-hunk display cap (ruling): every hunk gets its own block even
        # when its body is elided, so hunks past the old single-Static's
        # global cap are still present (and, later, annotatable) --
        # floor-guarded so a diff with more hunks than cap lines still shows
        # at least 1 body line each.
        per_hunk_cap = max(1, cap // max(1, len(hunks)))
        for hunk_idx, hunk in enumerate(hunks):
            await body.mount(
                Static(
                    self._styled_diff(
                        self._hunk_display_text(
                            hunk,
                            per_hunk_cap,
                            include_prelude=hunk_idx == 0,
                        )
                    ),
                    classes="console-turn-file-hunk",
                    markup=False,
                )
            )
            actions_row = Horizontal(classes="console-turn-file-hunk-actions")
            await body.mount(actions_row)
            notes_box = Vertical(classes="console-turn-file-hunk-notes")
            await body.mount(notes_box)
            if self._notes_capable:
                note_btn = Button(
                    "✎ note",
                    classes="console-turn-file-note-btn",
                    compact=True,
                )
                note_btn.entry_index = idx
                note_btn.hunk_index = hunk_idx
                # Same guard as the row button above -- a quick second
                # press must never be silently swallowed by the default
                # "-active" flash.
                note_btn.active_effect_duration = 0
                await actions_row.mount(note_btn)
                # (root, path) alone is not a unique key: a run can hold
                # TWO rows on the same root+path (a turn's own window and
                # its post-turn window, `turn_file_entries`'s docstring) --
                # each producing its OWN entry with its OWN diff. Matching
                # by hunk_index alone let a note saved on one window's
                # hunk N bleed into the OTHER window's same-index hunk N,
                # rendering under the wrong diff there (final-review fix
                # wave). The hunk's header text is what actually
                # disambiguates which window's hunk a note anchors to.
                existing_notes = [
                    note
                    for note in self._notes_by_key.get(
                        (entry.root, entry.path), []
                    )
                    if int(note.get("hunk_index", -1)) == hunk_idx
                    and note.get("hunk_header") == hunk.header
                ]
                for note in existing_notes:
                    await notes_box.mount(self._build_note_row(note))

    async def _toggle_all(self, button: Button) -> None:
        """Header expand/collapse-all: a plain state-derived toggle.

        Reads the CURRENT display state of every row's body rather than
        tracking a separate "all expanded" flag -- a user can always
        collapse or expand one row individually via its own button, and
        deriving from the live DOM means this toggle can never drift out of
        sync with what is actually on screen. "Everything already
        expanded" collapses all; anything else (including a mix, or
        nothing yet expanded) expands all.

        Args:
            button: The pressed toggle button (its label/tooltip are
                updated in place to reflect the new state).
        """
        try:
            bodies = list(self.query(".console-turn-file-diff"))
            if bodies and all(body.display for body in bodies):
                self._collapse_all(button)
            else:
                await self._expand_all(button)
        except Exception:
            logger.opt(exception=True).warning(
                "Turn file card expand/collapse-all failed."
            )

    async def _expand_all(self, button: Button) -> None:
        """Expand every row, loading any uncached diff SERIALIZED.

        Deliberately sequential -- one ``await`` per uncached row inside
        this single coroutine, never ``asyncio.gather`` or N separate
        workers -- so a turn with many changed files never launches N
        concurrent git subprocesses at once (spec §5). A single row's
        provider-construction or diff-read failure degrades that ONE row
        (logged, left collapsed) without aborting the rest.

        Args:
            button: The header toggle button, updated to the "expanded"
                state once every row has been processed.
        """
        rows = list(self.query(".console-turn-file-row"))
        bodies = list(self.query(".console-turn-file-diff"))
        provider: Any = None
        provider_attempted = False
        for idx, entry in enumerate(self._entries):
            if idx >= len(rows) or idx >= len(bodies):
                continue
            row = rows[idx]
            body = bodies[idx]
            if idx not in self._hunk_cache:
                if not provider_attempted:
                    provider_attempted = True
                    try:
                        provider = self._provider_factory()
                    except Exception:
                        logger.opt(exception=True).warning(
                            "Turn file card expand-all provider "
                            "construction failed."
                        )
                        provider = None
                if provider is None:
                    continue
                snapshot_row = self._row_for_entry.get(idx)
                if snapshot_row is None:
                    continue
                try:
                    hunks = await self._read_hunks(provider, snapshot_row, entry)
                    self._hunk_cache[idx] = hunks
                    if not body.is_mounted:
                        continue
                    await self._mount_hunk_blocks(body, idx, entry, provider, hunks)
                except Exception:
                    logger.opt(exception=True).warning(
                        "Turn file card expand-all diff load failed for {}",
                        entry.label,
                    )
                    continue
            if not body.display:
                body.display = True
                row.label = self._row_label_text(entry, expanded=True)
        self._update_toggle_all_button(button, expanded=True)

    def _collapse_all(self, button: Button) -> None:
        """Hide every row's diff body (display-managed, never unmounted).

        Args:
            button: The header toggle button, updated to the "collapsed"
                state.
        """
        rows = list(self.query(".console-turn-file-row"))
        bodies = list(self.query(".console-turn-file-diff"))
        for idx, entry in enumerate(self._entries):
            if idx >= len(rows) or idx >= len(bodies):
                continue
            body = bodies[idx]
            if body.display:
                body.display = False
                rows[idx].label = self._row_label_text(entry, expanded=False)
        self._update_toggle_all_button(button, expanded=False)

    @staticmethod
    def _update_toggle_all_button(button: Button, *, expanded: bool) -> None:
        """Sync the toggle button's chevron/tooltip to the new state."""
        if expanded:
            button.label = f"{resolve_glyph(_CHEVRON_OPEN)} All"
            button.tooltip = "Collapse every file's diff"
        else:
            button.label = f"{resolve_glyph(_CHEVRON_CLOSED)} All"
            button.tooltip = "Expand every file's diff"

    def _row_label_text(self, entry: TurnFileEntry, *, expanded: bool) -> str:
        """Build one row's label, middle-eliding the path to the card's width.

        The full, un-elided path always stays available in the row
        Button's ``tooltip`` (set once, at row-creation time) -- this only
        controls what the label itself prints.

        Args:
            entry: The file entry the row renders.
            expanded: Whether the row's diff body is (or is about to be)
                shown -- selects the chevron glyph.

        Returns:
            The row's display label.
        """
        chevron = resolve_glyph(_CHEVRON_OPEN if expanded else _CHEVRON_CLOSED)
        prefix = f"{chevron} {entry.status}  "
        suffix = f"  +{entry.adds} −{entry.dels}"
        width = int(self.size.width)
        if width > 0:
            budget = max(1, width - len(prefix) - len(suffix))
            path_text = middle_elide_path(entry.label, budget)
        else:
            # Not yet laid out (or a bare unit-construction host with no
            # real geometry) -- show the full path rather than guessing a
            # budget; `on_resize` recomputes once real geometry lands.
            path_text = entry.label
        return f"{prefix}{path_text}{suffix}"

    def on_resize(self, event: Resize) -> None:
        """Recompute every mounted row's elided label for the new width.

        ``Resize`` does not bubble (``textual.events.Resize(bubble=False)``)
        so this fires only when THIS card's own width changes -- exactly
        the right trigger, since every row Button spans the card's full
        width (``DEFAULT_CSS``'s ``.console-turn-file-row { width: 100%;
        }``).

        Args:
            event: The card's resize event (unused; the new size is read
                straight off ``self.size``).
        """
        del event
        try:
            if not self._entries:
                return
            rows = list(self.query(".console-turn-file-row"))
            bodies = list(self.query(".console-turn-file-diff"))
            for idx, entry in enumerate(self._entries):
                if idx >= len(rows):
                    continue
                expanded = idx < len(bodies) and bodies[idx].display
                rows[idx].label = self._row_label_text(entry, expanded=expanded)
        except Exception:
            logger.opt(exception=True).warning(
                "Turn file card resize label refresh failed."
            )

    async def on_key(self, event: Key) -> None:
        """Reclaim Enter/Escape/Up/Down from a focused note input's ancestors.

        A BINDINGS-only approach here (the first cut of this feature) is
        provably wrong once this card is mounted inside a real
        ``ConsoleTranscript`` -- traced (and pinned by a regression test)
        after review flagged the interaction as unverified. The root
        cause is Textual's actual key-dispatch order, not a focus race:
        ``App.on_event`` checks *priority* bindings app-wide, then
        forwards the raw ``Key`` MESSAGE to the focused widget, which
        bubbles up the DOM like any other message; **non-priority**
        bindings -- including an `Input`'s own built-in ``enter ->
        submit`` and this card's escape binding -- are resolved only in
        ``App._on_key``, i.e. only once that message reaches the App
        completely UNSTOPPED (see ``textual/app.py``'s
        ``_check_bindings``/``_on_key``). ``ConsoleTranscript`` defines
        its OWN raw ``on_key`` for row navigation
        (``"enter" -> confirm_selection``, ``"escape" ->
        clear_selection``) that unconditionally calls ``event.stop()`` --
        so when this card is nested inside it, that ancestor's handler
        wins the bubble race every time, and the focused note `Input`'s
        own binding never even gets checked. A live-transcript regression
        test proved this: Enter inside the note input selected the
        transcript row instead of saving, with nothing raised or logged.

        Reclaiming both keys HERE -- on this card, a closer ancestor of
        the note input than ``ConsoleTranscript`` -- wins the bubble race
        instead, mirroring ``ConsoleTranscriptActionButton.on_key``'s
        existing reclaiming of Enter/Tab/Escape in this same codebase for
        the identical reason. This is now the single source of truth for
        both keys, in every host (the standalone-card tests keep passing
        unchanged): ``on_input_submitted`` stays as a harmless fallback
        for a programmatically-posted ``Input.Submitted``, but a real
        Enter keypress no longer reaches it -- this handler saves
        directly.

        Up/Down get the same bubble-race treatment for the same root
        cause (final-review fix wave): ``ConsoleTranscript.on_key`` also
        binds ``"up"``/``"down"`` to row navigation
        (``action_select_previous``/``action_select_next``), so a user
        typing into a focused note input who happens to press an arrow
        key -- cursor movement is not even meaningful inside a
        single-line ``Input`` here, there is nothing above/below to move
        to -- would otherwise silently move the transcript's row
        selection out from under them mid-edit. Swallowed here with no
        action of its own: an ``Input`` has no built-in use for either
        key, so this is a pure no-op reclaim, not a redirect.

        Args:
            event: The bubbling key event.
        """
        try:
            focused = self.app.focused
            if focused is None or not focused.has_class(
                "console-turn-file-note-input"
            ):
                return
            if event.key == "enter":
                event.stop()
                event.prevent_default()
                await self._save_note(focused)
            elif event.key == "escape":
                event.stop()
                event.prevent_default()
                await self.action_cancel_note_input()
            elif event.key in ("up", "down"):
                event.stop()
                event.prevent_default()
        except Exception:
            logger.opt(exception=True).warning(
                "Turn file card note-input key handling failed."
            )

    async def action_cancel_note_input(self) -> None:
        """Unmount the focused note input without saving.

        Called from ``on_key`` above once it has confirmed a note input
        is focused and the pressed key was escape.
        """
        try:
            focused = self.app.focused
            if focused is not None and focused.has_class(
                "console-turn-file-note-input"
            ):
                await focused.remove()
        except Exception:
            logger.opt(exception=True).warning(
                "Turn file card note-input cancel failed."
            )

    async def _open_note_input(self, button: Button) -> None:
        """Mount an inline note ``Input`` under the pressed hunk's block.

        Args:
            button: The pressed ``✎ note`` button, carrying ``entry_index``/
                ``hunk_index`` attributes set at mount time.
        """
        try:
            idx = getattr(button, "entry_index", None)
            hunk_idx = getattr(button, "hunk_index", None)
            if idx is None or hunk_idx is None:
                return
            bodies = list(self.query(".console-turn-file-diff"))
            if idx >= len(bodies):
                return
            notes_boxes = list(bodies[idx].query(".console-turn-file-hunk-notes"))
            if hunk_idx >= len(notes_boxes):
                return
            notes_box = notes_boxes[hunk_idx]
            existing_inputs = list(
                notes_box.query(".console-turn-file-note-input")
            )
            if existing_inputs:
                # Already open -- focus it rather than mounting a second
                # input for the same hunk.
                existing_inputs[0].focus()
                return
            note_input = Input(
                classes="console-turn-file-note-input",
                placeholder="Add a note…",
                max_length=NOTE_MAX_LENGTH,
            )
            note_input.entry_index = idx
            note_input.hunk_index = hunk_idx
            await notes_box.mount(note_input)
            note_input.focus()
        except Exception:
            logger.opt(exception=True).warning(
                "Turn file card note-input open failed."
            )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in a note input: save the note off-thread."""
        if not event.input.has_class("console-turn-file-note-input"):
            return
        event.stop()
        await self._save_note(event.input)

    async def _save_note(self, note_input: Input) -> None:
        """Validate, persist off-thread, and render one hunk note.

        On success the input is replaced in place by a rendered note row.
        A raising ``provider.add_change_note`` (or any other failure) is
        swallowed here -- the input stays mounted with the user's text
        intact so nothing is lost, and a warning is logged -- this is the
        card's absolute "no exception escapes an `on_*` handler" rule.

        Args:
            note_input: The submitted note ``Input``.
        """
        try:
            idx = getattr(note_input, "entry_index", None)
            hunk_idx = getattr(note_input, "hunk_index", None)
            if idx is None or hunk_idx is None:
                return
            if idx >= len(self._entries) or idx not in self._hunk_cache:
                return
            hunks = self._hunk_cache[idx]
            if hunk_idx >= len(hunks):
                return
            text = _validate_note_text(note_input.value)
            if text is None:
                return
            entry = self._entries[idx]
            hunk = hunks[hunk_idx]
            # Captured NOW, at save time -- not at input-open time -- per
            # spec §1/§3: the excerpt is the retention safety net, and the
            # card's own cached hunk is the full-diff-derived source for it.
            excerpt = hunk_excerpt(hunk)
            provider = self._provider_factory()
            if provider is None:
                return
            add_change_note = getattr(provider, "add_change_note", None)
            if not callable(add_change_note):
                return

            def _write() -> int:
                return add_change_note(
                    run_id=self._run_id,
                    root=entry.root,
                    path=entry.path,
                    hunk_index=hunk_idx,
                    hunk_header=hunk.header,
                    hunk_excerpt=excerpt,
                    note=text,
                )

            note_id = await asyncio.to_thread(_write)
            if not note_input.is_mounted:
                return
            notes_box = note_input.parent
            note_record = {
                "id": note_id,
                "note": text,
                "delivered_at": None,
                "root": entry.root,
                "path": entry.path,
                "hunk_index": hunk_idx,
                # Mirrors the DB row's own column (see `notes_for_run`) --
                # needed so an in-session note cached here matches
                # `_mount_hunk_blocks`'s hunk_header-qualified filter the
                # same way a reloaded-from-DB note record does.
                "hunk_header": hunk.header,
            }
            self._notes_by_key.setdefault(
                (entry.root, entry.path), []
            ).append(note_record)
            await note_input.remove()
            if notes_box is not None and notes_box.is_mounted:
                await notes_box.mount(self._build_note_row(note_record))
        except Exception:
            logger.opt(exception=True).warning(
                "Turn file card note save failed."
            )

    async def _delete_note(self, button: Button) -> None:
        """Delete a pending note off-thread and remove its rendered row.

        Args:
            button: The pressed ``✕`` button, carrying a ``note_id``
                attribute set at mount time.
        """
        try:
            note_id = getattr(button, "note_id", None)
            if note_id is None:
                return
            provider = self._provider_factory()
            if provider is None:
                return
            delete_change_note = getattr(provider, "delete_change_note", None)
            if not callable(delete_change_note):
                return

            def _delete() -> bool:
                return delete_change_note(note_id)

            deleted = await asyncio.to_thread(_delete)
            if not deleted:
                # A live card is reused in place across transcript syncs and
                # never reloads its own notes (final-review fix wave) -- so
                # a note delivered while this card stayed open still shows
                # its stale ✕ button. `delete_change_note` correctly refuses
                # (``delivered_at IS NOT NULL``), but a silent no-op here
                # would look like a bug to the user. Surface it instead of
                # leaving the press unexplained.
                self.notify(
                    "Note already sent — no longer deletable",
                    severity="warning",
                )
                return
            for notes in self._notes_by_key.values():
                notes[:] = [
                    note for note in notes if int(note.get("id", -1)) != int(note_id)
                ]
            note_row = button.parent
            if note_row is not None and note_row.is_mounted:
                await note_row.remove()
        except Exception:
            logger.opt(exception=True).warning(
                "Turn file card note delete failed."
            )

    @staticmethod
    def _build_note_row(note: dict) -> Horizontal:
        """Render one note as a ``.console-turn-file-note`` row.

        Args:
            note: A ``change_notes`` row dict (at minimum ``id``, ``note``,
                ``delivered_at``).

        Returns:
            A row with the note text, plus a ``✕`` delete button while
            ``delivered_at`` is null -- delivered notes render a ``sent``
            marker instead and carry no delete affordance (they are
            record).
        """
        delivered = note.get("delivered_at") is not None
        text = str(note.get("note", ""))
        label_text = f"{text}  · sent" if delivered else text
        children: list[Any] = [
            Static(
                label_text,
                classes="console-turn-file-note-text",
                markup=False,
            )
        ]
        if not delivered:
            delete_btn = Button(
                "✕", classes="console-turn-file-note-delete", compact=True
            )
            delete_btn.note_id = int(note["id"])
            delete_btn.active_effect_duration = 0
            children.append(delete_btn)
        return Horizontal(*children, classes="console-turn-file-note")

    @staticmethod
    def _hunk_display_text(hunk: DiffHunk, cap: int, *, include_prelude: bool) -> str:
        """Render one hunk's display text: prelude (first hunk only) plus
        its header and a per-hunk-capped, honestly-elided body.

        Reuses ``hunk_excerpt``'s own elision convention (the "... N more
        lines" tail) so the card and the note-delivery block (Task 5) never
        drift on how a capped hunk reads.

        Args:
            hunk: The hunk to render.
            cap: Maximum number of body lines to show before eliding.
            include_prelude: Whether to prepend ``hunk.file_prelude`` (only
                the first hunk of a file carries it in the UI -- every
                hunk's own copy is identical, see ``DiffHunk``'s docstring).

        Returns:
            The combined prelude/header/body text, ready for
            ``_styled_diff``.
        """
        parts: list[str] = []
        if include_prelude and hunk.file_prelude:
            parts.append(hunk.file_prelude)
        parts.append(hunk_excerpt(hunk, cap=cap))
        return "\n".join(parts)

    @staticmethod
    def _styled_diff(text: str):
        from rich.text import Text

        out = Text()
        for line in text.splitlines(keepends=False):
            if line.startswith("+") and not line.startswith("+++"):
                out.append(line + "\n", style="green")
            elif line.startswith("-") and not line.startswith("---"):
                out.append(line + "\n", style="red")
            elif line.startswith("@@"):
                out.append(line + "\n", style="dim")
            else:
                out.append(line + "\n")
        return out
