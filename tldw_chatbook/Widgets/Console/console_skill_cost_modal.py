"""Read-only view of the current Console skill report."""

from typing import ClassVar, Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

SkillManagementDestination = Literal["skills", "tools", "plugins"]


class ConsoleSkillCostModal(
    SafeModalDismissMixin, ModalScreen[SkillManagementDestination | None]
):
    """Show literal diagnostic text and links to existing management screens."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "request_safe_cancel", "Close")
    ]
    SAFE_MODAL_CONTENT = "#console-skill-cost-modal"

    def __init__(self, report: str) -> None:
        super().__init__()
        self._report = report

    def compose(self) -> ComposeResult:
        with Vertical(id="console-skill-cost-modal"):
            yield Static("Skill costs and usage", classes="console-modal-header")
            yield TextArea(self._report, id="console-skill-cost-text", read_only=True)
            with Horizontal(id="console-skill-cost-actions"):
                yield Button("Skills", id="console-skill-cost-skills")
                yield Button("MCP", id="console-skill-cost-tools")
                yield Button("Plugins", id="console-skill-cost-plugins")
                yield Button("Close", id="console-skill-cost-close", variant="primary")

    def on_mount(self) -> None:
        self.query_one(TextArea).focus()

    @on(Button.Pressed, "#console-skill-cost-close")
    async def _close(self) -> None:
        await self.request_safe_cancel(source="visible")

    @on(Button.Pressed, "#console-skill-cost-skills")
    def _open_skills(self) -> None:
        self.dismiss_safe_once("skills")

    @on(Button.Pressed, "#console-skill-cost-tools")
    def _open_tools(self) -> None:
        self.dismiss_safe_once("tools")

    @on(Button.Pressed, "#console-skill-cost-plugins")
    def _open_plugins(self) -> None:
        self.dismiss_safe_once("plugins")
