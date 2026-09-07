"""Modal editor for one internal prompt: contract callout, placeholder chips,
TextArea, live render preview (templated prompts only), and Save/Reset/Cancel.

Pure UI — performs no config IO. Dismisses with an action dict the panel acts
on: {"action":"save","text":str} | {"action":"reset"} | None (cancel)."""

from __future__ import annotations

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Static, TextArea

from tldw_chatbook.Internal_Prompts import safe_substitute
from tldw_chatbook.Internal_Prompts.catalog import PromptSpec

# Realistic sample values for the live preview; visible ‹token› fallback for
# any declared token not mapped here.
_SAMPLE_VALUES = {
    "query": "What is quantum computing?",
    "original_query": "What is quantum computing?",
    "original_question": "What is quantum computing?",
    "question": "What is quantum computing?",
    "content": "‹document text›",
    "content_summary": "‹summary of collected sources›",
    "concatenated_texts": "1. ‹source one›\n2. ‹source two›",
    "current_date": "2026-07-22",
    "title": "Example Result Title",
    "title1": "Result One",
    "title2": "Result Two",
    "content1": "‹content one›",
    "content2": "‹content two›",
    "url": "https://example.com/article",
    "published": "2026-07-01",
    "results_list": "0. Title: A\n   Content: ‹...›",
    "tool_list": '{\n  "name": "demo",\n  "description": "…",\n  "parameters": {}\n}',
    "fence_open": "```tool_call",
    "fence_close": "```",
    "change_percentage": "12.5",
    "type": "rss",
    "name": "Example Subscription",
    "reasoning": "",
    "results_list_placeholder": "",
}


def _sample_for(token: str) -> str:
    return _SAMPLE_VALUES.get(token, f"‹{token}›")


class InternalPromptEditorModal(ModalScreen[Optional[dict]]):
    """Edit / reset a single internal prompt."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, *, spec: PromptSpec, active_text: str) -> None:
        super().__init__()
        self._spec = spec
        self._active_text = active_text

    def compose(self) -> ComposeResult:
        spec = self._spec
        with Vertical(id="internal-prompt-editor-modal"):
            yield Static(spec.title, classes="console-modal-header")
            yield Static(spec.description, classes="internal-prompt-editor-desc", markup=False)
            if spec.contract_note:
                yield Static(
                    "⚠ " + spec.contract_note,
                    id="internal-prompt-editor-contract",
                    classes="internal-prompt-editor-contract",
                    markup=False,
                )
            if spec.required_placeholders:
                chips = "  ".join("{" + p + "}" for p in spec.required_placeholders)
                yield Static(
                    "Required placeholders: " + chips,
                    classes="internal-prompt-editor-chips",
                    markup=False,
                )
            if spec.applies and spec.applies != "live":
                yield Static(
                    "Applies: " + spec.applies,
                    classes="internal-prompt-editor-applies",
                    markup=False,
                )
            yield TextArea(self._active_text, id="internal-prompt-editor-text")
            if spec.required_placeholders:
                yield Static("Preview", classes="internal-prompt-editor-section")
                yield Static(
                    self._render_preview(self._active_text),
                    id="internal-prompt-editor-preview",
                    classes="internal-prompt-editor-preview",
                    markup=False,
                )
            with Collapsible(title="Shipped default", collapsed=True):
                yield Static(spec.default, markup=False)
            yield Static("", id="internal-prompt-editor-error",
                         classes="internal-prompt-editor-error", markup=False)
            with Horizontal(classes="internal-prompt-editor-actions"):
                yield Button("Reset to default", id="internal-prompt-editor-reset")
                yield Button("Cancel", id="internal-prompt-editor-cancel")
                yield Button("Save", id="internal-prompt-editor-save", variant="primary")

    def on_mount(self) -> None:
        try:
            self.query_one("#internal-prompt-editor-text", TextArea).focus()
        except (NoMatches, QueryError):
            pass

    def _render_preview(self, text: str) -> str:
        values = {p: _sample_for(p) for p in self._spec.required_placeholders}
        return safe_substitute(text, **values)

    @on(TextArea.Changed, "#internal-prompt-editor-text")
    def _on_text_changed(self, event: TextArea.Changed) -> None:
        if not self._spec.required_placeholders:
            return
        try:
            self.query_one("#internal-prompt-editor-preview", Static).update(
                self._render_preview(event.text_area.text)
            )
        except (NoMatches, QueryError):
            pass

    def _validate(self, text: str) -> Optional[str]:
        if not text.strip():
            return "Prompt text cannot be empty."
        missing = [
            p for p in self._spec.required_placeholders if ("{" + p + "}") not in text
        ]
        if missing:
            return "Missing required placeholder(s): " + ", ".join(
                "{" + p + "}" for p in missing
            )
        return None

    def _do_save(self) -> None:
        text = self.query_one("#internal-prompt-editor-text", TextArea).text
        err = self._validate(text)
        if err:
            self.query_one("#internal-prompt-editor-error", Static).update(err)
            return
        self.dismiss({"action": "save", "text": text})

    async def _save_from_test(self) -> None:  # test seam; same path as the button
        self._do_save()

    @on(Button.Pressed, "#internal-prompt-editor-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        self._do_save()

    @on(Button.Pressed, "#internal-prompt-editor-reset")
    def _reset(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss({"action": "reset"})

    @on(Button.Pressed, "#internal-prompt-editor-cancel")
    def _cancel_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
