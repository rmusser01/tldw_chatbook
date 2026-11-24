"""Working owner links and responsive pane modes for Research Workspace."""

from __future__ import annotations

from typing import Any, Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Select, Static

from ...Research_Workspace.contracts import QualifiedWorkspaceRef
from ..Navigation.main_navigation import NavigateToScreen


class ResearchWorkspaceMenu(Horizontal):
    """Select within the visible authority's catalog or open its manager."""

    class WorkspaceChosen(Message):
        def __init__(self, ref: QualifiedWorkspaceRef) -> None:
            super().__init__()
            self.ref = ref

    def compose(self) -> ComposeResult:
        yield Static(
            "No research workspace selected.", id="research-workspace-selection"
        )
        yield Select(
            [], id="research-workspace-picker", prompt="Workspace", compact=True
        )
        yield Button(
            "Manage Workspaces...",
            id="research-manage-workspaces",
            tooltip="Open Settings workspace management.",
            compact=True,
        )

    def on_mount(self) -> None:
        self.sync_choices((), None)

    def sync_choices(self, workspaces, ref: QualifiedWorkspaceRef | None) -> None:
        picker = self.query_one("#research-workspace-picker", Select)
        choices = [(row.name, row.ref) for row in workspaces]
        with picker.prevent(Select.Changed):
            picker.set_options(choices)
            picker.value = (
                ref if any(row.ref == ref for row in workspaces) else Select.NULL
            )
        picker.display = len(choices) > 1
        self.query_one(
            "#research-workspace-selection", Static
        ).display = not picker.display

    @on(Select.Changed, "#research-workspace-picker")
    def choose(self, event: Select.Changed) -> None:
        if isinstance(event.value, QualifiedWorkspaceRef):
            self.post_message(self.WorkspaceChosen(event.value))

    @on(Button.Pressed, "#research-manage-workspaces")
    def open_workspace_owner(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(NavigateToScreen("settings", {"category": "workspaces"}))


ResearchPaneName = Literal["sources", "chat", "studio"]


class ResearchPaneModeStrip(Horizontal):
    """Sources/Chat/Studio selector shown at medium and narrow widths."""

    class Selected(Message):
        """Request that one pane become visible."""

        def __init__(self, pane: ResearchPaneName) -> None:
            super().__init__()
            self.pane = pane

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visible_panes: tuple[ResearchPaneName, ...] = ("chat",)

    def compose(self) -> ComposeResult:
        for pane, label in (
            ("sources", "Sources (0)"),
            ("chat", "Chat"),
            ("studio", "Studio (0)"),
        ):
            yield Button(
                label,
                id=f"research-pane-mode-{pane}",
                classes="research-pane-mode-button",
                tooltip=f"Show {pane.title()} pane",
                compact=True,
            )

    def sync_visible_panes(self, visible_panes: tuple[ResearchPaneName, ...]) -> None:
        """Paint active pane modes without remounting their buttons."""
        self.visible_panes = visible_panes
        for pane in ("sources", "chat", "studio"):
            button = self.query_one(f"#research-pane-mode-{pane}", Button)
            button.set_class(pane in visible_panes, "is-active")

    @on(Button.Pressed, ".research-pane-mode-button")
    def select_pane(self, event: Button.Pressed) -> None:
        event.stop()
        pane = str(event.button.id or "").removeprefix("research-pane-mode-")
        if pane in {"sources", "chat", "studio"}:
            self.post_message(self.Selected(pane))  # type: ignore[arg-type]
