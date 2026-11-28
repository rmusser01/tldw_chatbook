"""A primary Studio command and its current owner availability."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Research_Workspace.action_registry import RESEARCH_ACTIONS
from ...Research_Workspace.studio_mapping import StudioAvailability
from ...Research_Workspace.studio_models import ResearchOutputKind, StudioOutputKind


def output_label(kind: StudioOutputKind) -> str:
    return RESEARCH_ACTIONS[kind.value].label


class ResearchStudioOutputCard(Vertical):
    def __init__(self, kind: ResearchOutputKind) -> None:
        super().__init__(classes="research-studio-card")
        self.kind = kind
        self.key = kind.value.replace("_", "-")

    def compose(self) -> ComposeResult:
        yield Button(
            output_label(self.kind),
            id=f"research-studio-create-{self.key}",
            classes="research-studio-create",
            disabled=True,
            compact=True,
        )
        yield Static(
            "",
            id=f"research-studio-availability-{self.key}",
            classes="research-studio-caption",
            markup=False,
        )

    def sync_availability(
        self, value: StudioAvailability | None, *, busy=False
    ) -> None:
        button = self.query_one(Button)
        button.disabled = busy or value is None or not value.available
        summary = "Select a workspace."
        if value is not None:
            summary = (
                f"{value.owner_label} · {value.user_message} {value.recovery}".strip()
            )
        self.query_one(Static).update(summary)
        button.tooltip = summary
