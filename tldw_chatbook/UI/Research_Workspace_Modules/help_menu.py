"""Contextual help, operating status, and links to existing owners."""

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ...Research_Workspace.owner_links import RESEARCH_OWNER_LINKS, can_open_workspace_in_console
from ...Widgets.modal_dismissal import SafeModalDismissMixin
from .operation_status import workspace_operation_status
from .owner_navigation import navigate_to_research_owner


class ResearchHelpMenu(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    SAFE_MODAL_CONTENT = "#research-help-dialog"

    def __init__(self, ref=None) -> None:
        super().__init__()
        self.ref = ref

    def compose(self) -> ComposeResult:
        with Vertical(
            id="research-help-dialog",
            classes="research-transfer-dialog research-transfer-scroll-dialog",
        ):
            yield Static("Research Workspace Help", classes="research-pane-title")
            with VerticalScroll(name="Research Help commands", classes="research-transfer-scroll"):
                yield Button(
                    "First-source tour",
                    id="research-help-tour",
                    classes="research-help-action",
                    compact=True,
                )
                yield Button(
                    "Keyboard shortcuts",
                    id="research-help-keyboard",
                    classes="research-help-action",
                    compact=True,
                )
                yield Button(
                    "Storage and operation status",
                    id="research-help-status",
                    classes="research-help-action",
                    compact=True,
                )
                for key in (
                    "console",
                    "agent-task",
                    "acp-history",
                    "sandbox",
                    "mcp",
                    "provider",
                    "library",
                    "artifacts",
                    "study",
                    "settings-workspaces",
                    "speech",
                    "runs",
                ):
                    yield Button(
                        RESEARCH_OWNER_LINKS[key].label,
                        id=f"research-help-{key}",
                        classes="research-help-action",
                        compact=True,
                        disabled=key in {"console", "agent-task"} and not can_open_workspace_in_console(self.ref),
                    )
                if not can_open_workspace_in_console(self.ref):
                    yield Static("Choose a Local workspace to open its context in Console.")
            yield Button("Close", id="research-help-close", compact=True)

    @on(Button.Pressed, ".research-help-action")
    def choose(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id.removeprefix("research-help-"))

    @on(Button.Pressed, "#research-help-close")
    def close(self) -> None:
        self.dismiss(None)


class ResearchHelpDetail(SafeModalDismissMixin, ModalScreen[bool | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    SAFE_MODAL_CONTENT = "#research-help-detail"

    def __init__(self, title: str, body: str, *, can_add_source=False) -> None:
        super().__init__()
        self.title = title
        self.body = body
        self.can_add_source = can_add_source

    def compose(self) -> ComposeResult:
        with Vertical(
            id="research-help-detail",
            classes="research-transfer-dialog research-transfer-scroll-dialog research-wide-dialog",
        ):
            yield Static(self.title, classes="research-pane-title")
            with VerticalScroll(classes="research-transfer-scroll"):
                yield Static(self.body, id="research-help-detail-body", markup=False)
            if self.can_add_source:
                yield Button("Add source", id="research-help-add-source", compact=True)
            yield Button("Close", id="research-help-detail-close", compact=True)

    @on(Button.Pressed, "#research-help-add-source")
    def add_source(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#research-help-detail-close")
    def close(self) -> None:
        self.dismiss(False)


async def show_research_help(screen, bindings) -> None:
    shortcuts = []
    for key, label in (
        ("slash", "/"),
        ("f1", "F1"),
        ("ctrl+p", "Ctrl+P"),
        ("ctrl+q", "Ctrl+Q"),
    ):
        active = bindings.get(key)
        if active is not None and active.enabled and active.binding.description:
            shortcuts.append(f"{label} · {active.binding.description}")
    ref = screen.controller.selected_ref
    action = await screen.app.push_screen_wait(ResearchHelpMenu(ref))
    if not screen.is_attached:
        return
    if action in RESEARCH_OWNER_LINKS and screen.controller.selected_ref == ref:
        navigate_to_research_owner(screen, action, ref)
    elif action == "keyboard":
        await screen.app.push_screen_wait(
            ResearchHelpDetail("Keyboard shortcuts", "\n".join(shortcuts))
        )
    elif action == "status":
        status = await workspace_operation_status(screen)
        if screen.is_attached:
            await screen.app.push_screen_wait(
                ResearchHelpDetail("Storage and operation status", status)
            )
    elif action == "tour":
        selected = screen.controller.selected_ref
        tour = "Choose Local or Server workspace data. Open Workspace actions to create a workspace if needed.\n\nAdd a source from the Library or ingest a new one. Select it after indexing, then ask a question in Grounded Chat. Save an output from Studio when you want to keep the result."
        go = await screen.app.push_screen_wait(
            ResearchHelpDetail(
                "First-source tour", tour, can_add_source=selected is not None
            )
        )
        if go and screen.is_attached and screen.controller.selected_ref == selected:
            screen.open_add_sources()
