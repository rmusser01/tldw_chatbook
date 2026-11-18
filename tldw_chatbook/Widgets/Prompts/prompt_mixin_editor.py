"""Small literal editor; persistence and source authority belong to its host."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input, Label, Select, TextArea

from ...Prompt_Management.prompt_artifact_models import LiteralMixinDefinition


class PromptMixinEditor(Vertical):
    """Edit only a Name, literal Instructions and Before/After placement."""

    class Changed(Message):
        """Ask the host to retain the current raw draft and dirty state."""

    class SaveRequested(Message):
        """Request save through the host's version/source authority."""

    class CancelRequested(Message):
        """Request the host's guarded cancellation."""

    def __init__(
        self, name: str, definition: LiteralMixinDefinition | None, **widget_kwargs: Any
    ) -> None:
        super().__init__(**widget_kwargs)
        self.add_class("h-auto")
        self.draft = (
            name,
            definition.body if definition is not None else "",
            definition.placement if definition is not None else "before",
        )

    def compose(self) -> ComposeResult:
        name, body, placement = self.draft
        yield Label("Name", classes="form-label")
        yield Input(name, id="mixin-name", classes="form-input")
        yield Label("Instructions", classes="form-label")
        yield TextArea(body, id="mixin-body", classes="form-textarea h-8")
        yield Label("Placement", classes="form-label")
        yield Select(
            (("Before", "before"), ("After", "after")),
            value=placement,
            allow_blank=False,
            id="mixin-placement",
            classes="form-select",
        )

    def read_name(self) -> str:
        """Return the name without normalizing user input."""
        return self.query_one("#mixin-name", Input).value

    def read_definition(self) -> LiteralMixinDefinition:
        """Validate one literal definition without expansion or trimming."""
        return LiteralMixinDefinition(
            self.query_one("#mixin-body", TextArea).text,
            self.query_one("#mixin-placement", Select).value,
        )

    def read_draft(self) -> tuple[str, str, str]:
        """Return raw private fields, including temporarily invalid instructions."""
        return (
            self.read_name(),
            self.query_one("#mixin-body", TextArea).text,
            self.query_one("#mixin-placement", Select).value,
        )

    @on(Input.Changed)
    @on(TextArea.Changed)
    @on(Select.Changed)
    def _changed(self, event: Message) -> None:
        event.stop()
        self.post_message(self.Changed())

    def action_save(self) -> None:
        """Ask the parent to save; no source is touched here."""
        self.post_message(self.SaveRequested())

    def action_cancel(self) -> None:
        """Ask the parent to cancel using its existing dirty guard."""
        self.post_message(self.CancelRequested())
