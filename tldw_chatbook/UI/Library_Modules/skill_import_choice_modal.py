"""Presentation-only choice for one skill in a repository package."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, OptionList, Static

from ...Widgets.modal_dismissal import SafeModalDismissMixin


class SkillImportChoiceModal(SafeModalDismissMixin, ModalScreen[str | None]):
    """Require an explicit, single candidate before any import begins."""

    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#skill-import-choice"
    DEFAULT_CSS = """
    SkillImportChoiceModal {
        align: center middle;
        background: $background 75%;
    }
    #skill-import-choice {
        width: 88;
        max-width: 100%;
        height: 28;
        max-height: 100%;
        min-height: 14;
        background: $panel;
        border: round $accent;
        padding: 1 2;
    }
    #skill-import-choice-title,
    #skill-import-choice-help {
        height: auto;
    }
    #skill-import-choice-list {
        height: 1fr;
        min-height: 4;
        background: $surface-darken-1;
    }
    #skill-import-choice-actions {
        height: 3;
    }
    #skill-import-choice-actions Button {
        width: auto;
        min-width: 10;
        height: 3;
        margin-right: 1;
    }
    """

    def __init__(self, candidates: tuple[str, ...]) -> None:
        if not candidates or len(candidates) > 20:
            raise ValueError("Skill candidate list must contain 1–20 paths.")
        super().__init__()
        self._candidates = tuple(candidates)

    def compose(self) -> ComposeResult:
        with Vertical(id="skill-import-choice"):
            yield Static(
                "Choose one skill to import",
                id="skill-import-choice-title",
                markup=False,
            )
            yield Static(
                "This repository contains multiple installable skills. "
                "Only the selected subdirectory will be copied for trust review.",
                id="skill-import-choice-help",
                markup=False,
            )
            yield OptionList(*self._candidates, id="skill-import-choice-list")
            with Horizontal(id="skill-import-choice-actions"):
                yield Button(
                    "Import skill",
                    id="skill-import-choice-import",
                    variant="primary",
                )
                yield Button("Cancel", id="skill-import-choice-cancel")

    def on_mount(self) -> None:
        choices = self.query_one("#skill-import-choice-list", OptionList)
        choices.highlighted = 0
        choices.focus()

    @on(Button.Pressed, "#skill-import-choice-import")
    def import_selected(self, event: Button.Pressed) -> None:
        event.stop()
        highlighted = self.query_one(
            "#skill-import-choice-list", OptionList
        ).highlighted
        if highlighted is not None and 0 <= highlighted < len(self._candidates):
            self.dismiss_safe_once(self._candidates[highlighted])

    @on(Button.Pressed, "#skill-import-choice-cancel")
    async def cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")
