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
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    TurnFileEntry,
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
    """

    def __init__(
        self,
        marker_text: str,
        run_id: str,
        provider_factory: Callable[[], Any],
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, classes="console-turn-file-card")
        self._marker_text = marker_text
        self._run_id = run_id
        self._provider_factory = provider_factory
        self._entries: list[TurnFileEntry] = []
        self._row_for_entry: dict[int, dict] = {}
        self._diff_cache: dict[int, str] = {}

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
                turn = next(
                    (t for t in provider.turns() if t.run_id == self._run_id),
                    None,
                )
                if turn is None:
                    return [], {}
                changed_by_root = {
                    str(row["root"]): provider.changed_files(row)
                    for row in turn.rows
                    if not row.get("tracking_error")
                }
                entries = turn_file_entries(turn.rows, changed_by_root)
                row_by_root = {
                    str(row["root"]): row for row in turn.rows
                }
                mapping = {
                    idx: row_by_root[entry.root]
                    for idx, entry in enumerate(entries)
                }
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
        if idx not in self._diff_cache:
            snapshot_row = self._row_for_entry.get(idx)
            if snapshot_row is None:
                return
            # Provider construction, the cap read, the off-thread diff read,
            # and the mount all live in one try/except -- a transient
            # provider-construction failure on first expand must degrade the
            # row (stay collapsed) rather than raise out of this `on_*`
            # handler, which Textual would otherwise propagate to
            # `app._handle_exception()` and exit the whole app.
            try:
                provider = self._provider_factory()
                if provider is None:
                    return
                text = await asyncio.to_thread(
                    provider.diff_text, snapshot_row, entry.path
                )
                cap = int(getattr(provider, "diff_display_max_lines", 2000))
                lines = text.splitlines()
                if len(lines) > cap:
                    hidden = len(lines) - cap
                    lines = lines[:cap] + [f"… {hidden} more lines (diff capped)"]
                self._diff_cache[idx] = "\n".join(lines)
                if not body.is_mounted:
                    return
                await body.mount(
                    Static(
                        self._styled_diff(self._diff_cache[idx]),
                        classes="console-turn-file-diff-text",
                        markup=False,
                    )
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
