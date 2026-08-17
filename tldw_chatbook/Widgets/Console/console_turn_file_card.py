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
from textual.events import Click, Key
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_display_state import (
    DiffHunk,
    TurnFileEntry,
    hunk_excerpt,
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
    ConsoleTurnFileCard .console-turn-file-header {
        height: 1;
        text-style: bold;
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
        # works and stays documented in the F1 help.
        head = self._marker_text.split(" — ")[0]
        yield Static(
            head,
            classes="console-turn-file-header",
            markup=False,
        )
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
                chevron = resolve_glyph(_CHEVRON_CLOSED)
                row = Button(
                    f"{chevron} {entry.status}  {entry.label}  "
                    f"+{entry.adds} −{entry.dels}",
                    classes="console-turn-file-row",
                    compact=True,
                )
                row.entry_index = idx
                # Button's default 0.2s "-active" flash guards `action_press`
                # against a second Enter/click until the flash clears -- fine
                # for a submit button, but this row toggles open/closed and a
                # quick second press should never be silently swallowed.
                row.active_effect_duration = 0
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
            row.label = (
                f"{resolve_glyph(_CHEVRON_CLOSED)} {entry.status}  "
                f"{entry.label}  +{entry.adds} −{entry.dels}"
            )
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

                def _read() -> list[DiffHunk]:
                    # Segmentation always runs on the FULL diff text (spec
                    # §2) -- never a display-truncated slice -- so hunk
                    # indices stay stable regardless of the per-hunk display
                    # cap applied below at mount time.
                    text = provider.diff_text(snapshot_row, entry.path)
                    return split_unified_diff(text)

                hunks = await asyncio.to_thread(_read)
                self._hunk_cache[idx] = hunks
                if not body.is_mounted:
                    return
                cap = int(getattr(provider, "diff_display_max_lines", 2000))
                # Per-hunk display cap (ruling): every hunk gets its own
                # block even when its body is elided, so hunks past the old
                # single-Static's global cap are still present (and,
                # later, annotatable) -- floor-guarded so a diff with more
                # hunks than cap lines still shows at least 1 body line
                # each.
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
                        # Same guard as the row button above -- a quick
                        # second press must never be silently swallowed by
                        # the default "-active" flash.
                        note_btn.active_effect_duration = 0
                        await actions_row.mount(note_btn)
                        existing_notes = [
                            note
                            for note in self._notes_by_key.get(
                                (entry.root, entry.path), []
                            )
                            if int(note.get("hunk_index", -1)) == hunk_idx
                        ]
                        for note in existing_notes:
                            await notes_box.mount(self._build_note_row(note))
            except Exception:
                logger.opt(exception=True).warning(
                    "Turn file card diff load failed for {}", entry.label
                )
                return
        body.display = True
        row.label = (
            f"{resolve_glyph(_CHEVRON_OPEN)} {entry.status}  "
            f"{entry.label}  +{entry.adds} −{entry.dels}"
        )

    async def on_key(self, event: Key) -> None:
        """Reclaim Enter/Escape from a focused note input's ancestors.

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
