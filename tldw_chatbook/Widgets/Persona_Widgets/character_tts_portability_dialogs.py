"""Focused user decisions for local character-card TTS portability."""

from __future__ import annotations

from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.TTS.profile_service import PortableProfileImportPlan

PortableCollisionChoice = Literal["reuse", "copy"]


class CharacterTTSProfileCollisionDialog(
    ModalScreen[PortableCollisionChoice | None]
):
    """Require an explicit reuse/copy decision for an imported collision."""

    BINDINGS = (("escape", "dismiss", "Cancel"),)

    DEFAULT_CSS = """
    CharacterTTSProfileCollisionDialog {
        align: center middle;
        background: $background 75%;
    }

    #character-tts-collision-dialog {
        width: 70;
        height: auto;
        background: $panel;
        border: round $accent;
        padding: 1 2;
    }

    #character-tts-collision-title {
        text-style: bold;
    }

    #character-tts-collision-copy {
        height: auto;
        margin: 1 0;
    }

    #character-tts-collision-actions {
        height: 3;
        align-horizontal: right;
    }

    #character-tts-collision-actions Button {
        width: auto;
        min-width: 12;
        height: 3;
        margin-left: 1;
        border: none;
    }
    """

    def __init__(self, plan: PortableProfileImportPlan) -> None:
        super().__init__()
        if type(plan) is not PortableProfileImportPlan or "create" in (
            plan.allowed_choices
        ):
            raise ValueError("collision_plan")
        self.plan = plan

    def compose(self) -> ComposeResult:
        can_reuse = "reuse" in self.plan.allowed_choices
        copy = (
            "A local voice profile already uses this imported identity or "
            "name. Reuse the identical local profile or create a separate copy."
            if can_reuse
            else (
                "A local voice profile already uses this imported identity or "
                "name with different generation values. Create a separate copy "
                "to preserve the existing profile."
            )
        )
        with Vertical(id="character-tts-collision-dialog"):
            yield Static(
                "Imported voice profile conflict",
                id="character-tts-collision-title",
            )
            yield Static(copy, id="character-tts-collision-copy")
            with Horizontal(id="character-tts-collision-actions"):
                yield Button("Cancel", id="character-tts-collision-cancel")
                if can_reuse:
                    yield Button("Reuse", id="character-tts-collision-reuse")
                yield Button(
                    "Create copy",
                    id="character-tts-collision-copy-profile",
                    variant="primary",
                )

    @on(Button.Pressed, "#character-tts-collision-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#character-tts-collision-reuse")
    def _reuse(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("reuse")

    @on(Button.Pressed, "#character-tts-collision-copy-profile")
    def _copy(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("copy")


class CharacterTTSExistingAssignmentDialog(ModalScreen[bool]):
    """Confirm that a reused character may receive the imported voice."""

    BINDINGS = (("escape", "dismiss", "Cancel"),)

    DEFAULT_CSS = """
    CharacterTTSExistingAssignmentDialog {
        align: center middle;
        background: $background 75%;
    }

    #character-tts-existing-dialog {
        width: 70;
        height: auto;
        background: $panel;
        border: round $warning;
        padding: 1 2;
    }

    #character-tts-existing-title {
        text-style: bold;
    }

    #character-tts-existing-copy {
        height: auto;
        margin: 1 0;
    }

    #character-tts-existing-actions {
        height: 3;
        align-horizontal: right;
    }

    #character-tts-existing-actions Button {
        width: auto;
        min-width: 12;
        height: 3;
        margin-left: 1;
        border: none;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="character-tts-existing-dialog"):
            yield Static(
                "Apply imported voice?",
                id="character-tts-existing-title",
            )
            yield Static(
                "This card matched an existing character. The character card "
                "was not replaced. Apply the imported voice profile to that "
                "existing character?",
                id="character-tts-existing-copy",
            )
            with Horizontal(id="character-tts-existing-actions"):
                yield Button("Keep current", id="character-tts-existing-cancel")
                yield Button(
                    "Apply voice",
                    id="character-tts-existing-confirm",
                    variant="warning",
                )

    @on(Button.Pressed, "#character-tts-existing-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    @on(Button.Pressed, "#character-tts-existing-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(True)
