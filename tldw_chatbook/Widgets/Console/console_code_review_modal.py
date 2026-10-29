"""Native session report reader; retained bodies never enter an agent tool call."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Select, Static, TextArea

from ..modal_dismissal import SafeModalDismissMixin
from .console_citation_sources_modal import _render_literal_text


class ConsoleCodeReviewModal(SafeModalDismissMixin, ModalScreen[None]):
    """Read accepted originals, raw output, and host-expanded decisions."""

    BINDINGS = [("escape", "request_safe_cancel", "Close")]
    SAFE_MODAL_CONTENT = "#code-review-dialog"
    AUTO_FOCUS = "#code-review-body"
    BUNDLED_CSS = """
    ConsoleCodeReviewModal { align: center middle; }
    #code-review-dialog {
        width: 95%; height: 95%; background: $ds-surface-panel;
        border: round $ds-control-edge; padding: $ds-space-inset;
    }
    #code-review-body { height: 1fr; }
    #code-review-actions { height: auto; }
    Button.code-review-action { min-width: $ds-control-action-min-width; width: 1fr; }
    """

    def __init__(self, *, service, review_id: str) -> None:
        super().__init__()
        self._service = service
        self._review_id = review_id
        self._body = ""
        self._offset = 0
        self._generation = 0
        self._preferred_slot = None
        self._selection = None
        self._next_offset = None
        self._history = []
        self._options_loaded = False

    def compose(self) -> ComposeResult:
        with Vertical(id="code-review-dialog"):
            yield Static(
                "Code review · session retention", classes="dialog-title", markup=False
            )
            yield Static(
                "Loading retained reports…", id="code-review-status", markup=False
            )
            yield Select(
                [], allow_blank=True, id="code-review-view", classes="form-select"
            )
            yield TextArea(read_only=True, id="code-review-body", soft_wrap=True)
            with Horizontal(id="code-review-actions", classes="dialog-buttons"):
                yield Button("Previous", id="code-review-previous", classes="code-review-action", compact=True)
                yield Button("Next", id="code-review-next", classes="code-review-action", compact=True)
                yield Button("Export", id="code-review-export", classes="code-review-action", compact=True)
                yield Button("Release", id="code-review-release", classes="code-review-action", compact=True)
                yield Button("Close", id="code-review-close", classes="code-review-action", compact=True)

    def on_mount(self) -> None:
        self.run_worker(self._load_options(), exclusive=True, group="review-reader")

    async def _load_options(self) -> None:
        try:
            slots = await asyncio.to_thread(
                self._service.native_report_slots, self._review_id
            )
            sources = await asyncio.to_thread(self._source_options)
            if self.is_attached and not self._pruning:
                self.query_one(Select).set_options(
                    [(f"{slot} · originals", slot) for slot in slots]
                    + [(f"{slot} · raw output", "raw:" + slot) for slot in slots]
                    + [("Unified report", "unified")]
                    + list(sources)
                )
                self._options_loaded = True
                self.query_one(Select).value = self._preferred_slot or (
                    slots[0] if slots else "unified"
                )
        except Exception as exc:
            self._error(exc)

    def _source_options(self):
        return [
            (Text(_render_literal_text(label)), value)
            for label, value in self._service.native_source_options(self._review_id)
        ]

    @on(Select.Changed, "#code-review-view")
    def _select(self, event: Select.Changed) -> None:
        if (
            isinstance(event.value, str)
            and event.value == self.query_one(Select).value
            and event.value != self._selection
        ):
            self._selection = event.value
            self._history = []
            self._generation += 1
            self.query_one("#code-review-next", Button).disabled = True
            self.query_one("#code-review-previous", Button).disabled = True
            self.run_worker(
                self._read(event.value, self._generation),
                exclusive=True,
                group="review-reader",
            )

    @staticmethod
    def literal_page(page: dict) -> dict:
        """Sanitize only a bounded page, on the caller's background worker."""
        return {**page, "body": _render_literal_text(page["body"])}

    def _read_page(self, selection: str, offset: int) -> dict:
        return self.literal_page(
            self._service.native_report_page(self._review_id, selection, offset)
        )

    async def _read(self, selection: str, generation: int, offset: int = 0) -> None:
        try:
            page = await asyncio.to_thread(self._read_page, selection, offset)
        except Exception as exc:
            if generation == self._generation:
                self._error(exc)
            return
        if self.is_attached and not self._pruning and generation == self._generation:
            try:
                body = self.query_one(TextArea)
                self._body, self._offset = page["body"], offset
                self._next_offset = page["next_offset"]
                self._show_page()
                body.focus()
            except NoMatches:
                return

    def present_original(self, slot_id: str, page: dict) -> bool:
        """Receive an off-loop sanitized bounded page after an actual mount."""
        if not self.is_mounted:
            return False
        self._preferred_slot = self._selection = slot_id
        self._generation += 1
        self._history = []
        self._body, self._offset = page["body"], 0
        self._next_offset = page["next_offset"]
        selector = self.query_one(Select)
        # Initial options may still be loading; _load_options uses preferred slot.
        if self._options_loaded:
            selector.value = slot_id
        self._show_page()
        self.query_one(TextArea).focus()
        return True

    def _error(self, exc: Exception) -> None:
        if self.is_attached and not self._pruning:
            try:
                self._body = ""
                self.query_one(TextArea).load_text("")
                self.query_one("#code-review-status", Static).update(
                    "Report unavailable: " + getattr(exc, "reason", type(exc).__name__)
                )
            except NoMatches:
                return

    def _show_page(self) -> None:
        self.query_one(TextArea).load_text(self._body)
        self.query_one("#code-review-status", Static).update(
            f"Page {len(self._history) + 1} · Export to keep a durable copy"
        )
        self.query_one("#code-review-previous", Button).disabled = not self._history
        self.query_one("#code-review-next", Button).disabled = self._next_offset is None

    @on(Button.Pressed)
    async def _action(self, event: Button.Pressed) -> None:
        key = event.button.id
        event.stop()
        if key in {"code-review-previous", "code-review-next"}:
            if key.endswith("next") and self._next_offset is not None:
                self._history.append(self._offset)
                offset = self._next_offset
            elif self._history:
                offset = self._history.pop()
            else:
                return
            self._generation += 1
            self.query_one("#code-review-next", Button).disabled = True
            self.query_one("#code-review-previous", Button).disabled = True
            self.run_worker(
                self._read(self._selection, self._generation, offset),
                exclusive=True,
                group="review-reader",
            )
        elif key == "code-review-export":
            from .console_save_markdown_modal import ConsoleSaveMarkdownModal

            self.app.push_screen(
                ConsoleSaveMarkdownModal(
                    default_path=str(Path.home() / "Downloads" / "code-review.md")
                ),
                callback=self._export,
            )
        elif key == "code-review-release":
            if event.button.label.plain != "Discard?":
                event.button.label = "Discard?"
                self.query_one("#code-review-status", Static).update(
                    "Release permanently discards this session's report bundle. Press Discard? to confirm."
                )
                return
            self.run_worker(self._release(), exclusive=True, group="review-release")
        elif key == "code-review-close":
            await self.request_safe_cancel(source="button")

    def _export(self, path: str | None) -> None:
        if path:
            self.run_worker(self._save(path), exclusive=False, group="review-export")

    async def _save(self, path: str) -> None:
        from ...UI.Console_Modules.markdown_export import write_markdown_file

        try:
            body = await asyncio.to_thread(
                self._service.render_all_reports, self._review_id
            )
            await write_markdown_file(path, body)
            self.app.notify("Review reports exported.", markup=False)
        except Exception as exc:
            self.app.notify(
                "Export failed: " + getattr(exc, "reason", type(exc).__name__),
                severity="error",
                markup=False,
            )

    async def _release(self) -> None:
        try:
            receipt = await asyncio.to_thread(
                self._service.release_native, self._review_id
            )
            self._body = ""
            self.query_one(TextArea).load_text("")
            self.query_one("#code-review-status", Static).update(
                "Reports released."
                if receipt.cleaned
                else "Reports released; cleanup pending."
            )
        except Exception as exc:
            self._error(exc)
