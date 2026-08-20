"""Cross-turn "Changed files" rail section (Inspector rail, TASK-18060).

Pure presentation over ``ConversationFileEntry`` (Chat/console_display_
state.py, task 3): the section receives precomputed display state and
renders it -- no DB reads, no git, no provider access of any kind. The
screen-side aggregation, caching, and guard machinery (spec §2) is the
NEXT task's concern; this widget only knows how to draw whatever
``ConsoleChangedFilesState`` it is handed and to resync onto a new one via
``update_state`` without being torn down and rebuilt by its owner.

Row labels are built as ``rich.text.Text``, never a plain ``str``:
``Button.label`` markup-parses a plain string, and the ASCII glyph
fallback for the note badge ("[N]", see ``Widgets/glyph_fallback.py``) is
bracket-wrapped -- a plain f-string would have that bracket read as an
(unknown, unclosed) markup tag and silently vanish from the rendered
label, exactly the failure class ``ConsoleTurnFileCard``'s own note button
already guards against (see that module's docstring on ``_GLYPH_NOTE``).
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Resize
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    ConversationFileEntry,
    middle_elide_path,
)
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard

#: TASK-16800's note badge glyph, shared verbatim with the turn file card
#: (``ConsoleTurnFileCard._GLYPH_NOTE``) -- one vocabulary, one ASCII
#: fallback ("[N]", ``ASCII_GLYPH_FALLBACKS``).
_GLYPH_NOTE = "✎"

#: Rail-section row cap (spec §2, "rail-section cap conventions, task-15110
#: family"): past this many files, an honest "+N more -- open Review" tail
#: Static reports the remainder instead of mounting a row per file -- a
#: conversation with hundreds of changed files must not grow the rail
#: without bound.
MAX_VISIBLE_ROWS = 12


@dataclass(frozen=True)
class ConsoleChangedFilesState:
    """Precomputed display state for one ``ConsoleChangedFilesSection``.

    ``entries`` is the FULL cross-turn list (``conversation_file_summary``'s
    return value, newest-first) -- the widget itself applies the
    ``MAX_VISIBLE_ROWS`` cap at render time, so the header's totals and the
    "+N more" tail can both be derived honestly from the same source of
    truth. ``pruned_rows`` mirrors ``conversation_file_summary``'s own
    ``pruned_rows`` count (retention-pruned snapshot rows the aggregation
    had to skip) -- rendered as a dim tail line rather than silently
    dropped.
    """

    entries: tuple[ConversationFileEntry, ...]
    pruned_rows: int = 0


class ConsoleChangedFilesSection(RecomposeCaptureGuard, Vertical):
    """Rail section listing a conversation's cross-turn changed files.

    One compact ``Button`` row per file (status letter, cell-elided path,
    ``+A −D``, a ``✎ N`` badge when the file carries notes); pressing a row
    posts ``FileSelected`` with the exact identity (``run_id``,
    ``snapshot_id``, ``path``, ``root``) the NEXT task's screen handler
    needs to open the Review screen scoped to that file. Renders NOTHING
    (``display = False``, zero children) when the conversation has no
    changed files and no pruned history -- the rail must not grow a
    permanent empty box for a conversation that has not touched a file.

    ``RecomposeCaptureGuard`` (task-627/637) is mixed in because
    ``update_state`` self-recomposes a widget with interactive ``Button``
    children: without it, a recompose landing mid-click on one of those
    rows would leak ``App.mouse_captured`` onto a now-detached widget and
    silently swallow every mouse event app-wide from then on -- the exact
    shape every other Console rail-section widget with this same
    recompose-over-Buttons pattern already guards against (e.g.
    ``ConsoleRunInspector``, ``ConsoleStagedContextTray``).
    """

    DEFAULT_CSS = """
    ConsoleChangedFilesSection {
        height: auto;
        min-height: 0;
    }
    ConsoleChangedFilesSection .console-changed-files-header {
        height: 1;
        width: 100%;
        text-style: bold;
    }
    ConsoleChangedFilesSection .console-changed-files-row {
        height: 1;
        min-height: 1;
        width: 100%;
        text-align: left;
    }
    ConsoleChangedFilesSection .console-changed-files-tail {
        height: 1;
        width: 100%;
        text-style: italic;
    }
    ConsoleChangedFilesSection .console-changed-files-pruned {
        height: 1;
        width: 100%;
        text-style: dim;
    }
    """

    class FileSelected(Message):
        """A row was pressed; carries the file's exact identity.

        ``snapshot_id`` names the NEWEST clean row that still covers this
        ``(root, path)`` -- the same row ``ConversationFileEntry.status``/
        ``adds``/``dels`` were read from (see that class's docstring) -- so
        the handler can open the Review screen pinned to that exact
        snapshot rather than the ambiguous "first row matching this path"
        default.
        """

        def __init__(
            self, run_id: str, snapshot_id: int, path: str, root: str
        ) -> None:
            self.run_id = run_id
            self.snapshot_id = snapshot_id
            self.path = path
            self.root = root
            super().__init__()

    def __init__(
        self, state: ConsoleChangedFilesState, *, id: str | None = None
    ) -> None:
        """Create the section from precomputed display state.

        Args:
            state: The section's display-state snapshot.
            id: Optional widget id (the rail mounts this with a fixed id;
                bare unit construction may omit it).
        """
        super().__init__(id=id, classes="console-changed-files-section")
        self.state = state
        self.display = not self._is_empty(state)

    @staticmethod
    def _is_empty(state: ConsoleChangedFilesState) -> bool:
        """Whether ``state`` has nothing at all worth rendering."""
        return not state.entries and state.pruned_rows <= 0

    def compose(self) -> ComposeResult:
        state = self.state
        if self._is_empty(state):
            return
        yield Static(
            self._header_text(state),
            id="console-changed-files-header",
            classes="console-changed-files-header",
            markup=False,
        )
        visible = state.entries[:MAX_VISIBLE_ROWS]
        for idx, entry in enumerate(visible):
            row = Button(
                self._row_label(entry),
                id=f"console-changed-files-row-{idx}",
                classes="console-changed-files-row",
                compact=True,
            )
            # Same guard as every other row/note button in the sibling turn
            # file card: a quick second press must never be silently
            # swallowed by the default "-active" flash.
            row.active_effect_duration = 0
            # The entry itself is stored on the button (the frozen
            # dataclass is safe to hold), not just its index: a stale
            # press delivered after `update_state()` has recomposed with a
            # REORDERED or different `entries` tuple would otherwise still
            # be in-bounds for `idx` but resolve to a DIFFERENT file --
            # mis-navigating the Review screen. Reading the entry straight
            # off the pressed button sidesteps that positional trap
            # entirely.
            row.entry = entry
            # The row's own label is middle-elided to the section's width
            # (mirrors ConsoleTurnFileCard) -- the tooltip always carries
            # the FULL, un-elided (multi-root-prefixed) label so nothing is
            # lost to the elision.
            row.tooltip = entry.label
            yield row
        remaining = len(state.entries) - len(visible)
        if remaining > 0:
            yield Static(
                f"+{remaining} more — open Review",
                id="console-changed-files-tail",
                classes="console-changed-files-tail",
                markup=False,
            )
        if state.pruned_rows > 0:
            noun = "turn" if state.pruned_rows == 1 else "turns"
            yield Static(
                f"history pruned for {state.pruned_rows} {noun}",
                id="console-changed-files-pruned",
                classes="console-changed-files-pruned",
                markup=False,
            )

    def update_state(self, state: ConsoleChangedFilesState) -> None:
        """Resync this mounted section onto a new state, in place.

        Called by the owning screen's sync loop (rail convention, spec §2)
        with the SAME widget instance every time -- never replaced. An
        equal state is a no-op; otherwise the display flag is re-derived
        and the section's own children are recomposed (never the owning
        rail/screen).

        Args:
            state: The section's new display-state snapshot.
        """
        if state == self.state:
            return
        self.state = state
        self.display = not self._is_empty(state)
        self.refresh(recompose=True)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """A row press: post ``FileSelected`` with that entry's identity.

        Reads the entry straight off the pressed button (``row.entry``,
        stamped at compose time) rather than re-indexing
        ``self.state.entries`` by a stored position -- a stale press
        delivered after ``update_state()`` recomposed with a reordered or
        different entries tuple could otherwise land in-bounds but resolve
        to the WRONG file. Degrading to a no-op when the button carries no
        entry (a partial-mount desync) keeps this handler from raising out
        into ``app._handle_exception()`` and exiting the whole app.
        """
        try:
            button = event.button
            entry = getattr(button, "entry", None)
            if entry is None:
                return
            event.stop()
            self.post_message(
                self.FileSelected(
                    entry.run_id, entry.snapshot_id, entry.path, entry.root
                )
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Changed-files section row press handling failed."
            )

    def on_resize(self, event: Resize) -> None:
        """Recompute every mounted row's elided label for the new width.

        ``Resize`` does not bubble (``textual.events.Resize(bubble=False)``)
        so this fires only when THIS section's own width changes -- mirrors
        ``ConsoleTurnFileCard.on_resize``. Each row re-elides from its OWN
        stored entry (``row.entry``, stamped at compose time), never from a
        positional re-slice of ``self.state.entries`` -- the same
        stale-index trap ``on_button_pressed`` guards against would
        otherwise let a resize mid-recompose re-label a row from the WRONG
        entry.

        Args:
            event: The section's resize event (unused; the new size is
                read straight off ``self.size``).
        """
        del event
        try:
            for row in self.query(".console-changed-files-row"):
                entry = getattr(row, "entry", None)
                if entry is None:
                    continue
                row.label = self._row_label(entry)
        except Exception:
            logger.opt(exception=True).warning(
                "Changed-files section resize label refresh failed."
            )

    @staticmethod
    def _header_text(state: ConsoleChangedFilesState) -> str:
        """Build the header line: file count + honest latest-turn totals.

        Totals are summed across the FULL ``entries`` tuple (not just the
        capped, rendered rows) -- the same honesty rule
        ``ConversationFileEntry`` itself documents: these are the newest
        covering turn's deltas per file, never a cumulative total, hence
        "latest turn deltas" in the label rather than a bare "+A -D".
        """
        total_adds = sum(entry.adds for entry in state.entries)
        total_dels = sum(entry.dels for entry in state.entries)
        return (
            f"Changed files ({len(state.entries)}) · latest turn deltas "
            f"+{total_adds} −{total_dels}"
        )

    def _row_label(self, entry: ConversationFileEntry) -> Text:
        """Build one row's label: status, cell-elided path, deltas, badge.

        Returns a pre-built ``Text`` -- never a plain ``str`` -- so
        ``Button.label`` never markup-parses it (see this module's
        docstring).

        Args:
            entry: The file entry the row renders.

        Returns:
            The row's display label.
        """
        status = resolve_glyph(entry.status)
        prefix = f"{status}  "
        suffix = f"  +{entry.adds} −{entry.dels}"
        if entry.note_count > 0:
            suffix += f"  {resolve_glyph(_GLYPH_NOTE)} {entry.note_count}"
        width = int(self.size.width)
        if width > 0:
            budget = max(1, width - len(prefix) - len(suffix))
            path_text = middle_elide_path(entry.label, budget)
        else:
            # Not yet laid out (or a bare unit-construction host with no
            # real geometry) -- show the full label rather than guessing a
            # budget; `on_resize` recomputes once real geometry lands.
            path_text = entry.label
        return Text(f"{prefix}{path_text}{suffix}")
