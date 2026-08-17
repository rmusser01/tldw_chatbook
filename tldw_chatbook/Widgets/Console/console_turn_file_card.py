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
from textual.events import Click
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    DiffHunk,
    TurnFileEntry,
    hunk_excerpt,
    split_unified_diff,
    turn_file_entries,
)
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph

_CHEVRON_CLOSED = "▸"
_CHEVRON_OPEN = "▾"


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

            def _read() -> tuple[list[TurnFileEntry], dict[int, dict]]:
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
                    return [], {}
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
                return entries, mapping

            entries, mapping = await asyncio.to_thread(_read)
            if not self.is_mounted or not entries:
                return
            self._entries = entries
            self._row_for_entry = mapping
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
        idx = getattr(event.button, "entry_index", None)
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
                    # Task 4 populates this row with the `✎ note` affordance
                    # and any existing notes; mounted empty here so the
                    # per-hunk block/action-row pairing is in place ahead of
                    # that task.
                    await body.mount(
                        Horizontal(classes="console-turn-file-hunk-actions")
                    )
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
