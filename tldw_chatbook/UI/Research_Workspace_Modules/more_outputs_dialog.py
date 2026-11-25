"""Browse Studio's extended actions with explicit owner readiness."""

from collections.abc import Mapping

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.events import DescendantFocus, Resize
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ...Research_Workspace.action_registry import (
    PLANNED_OUTPUTS,
    RESEARCH_ACTIONS,
    ResearchStudioAction,
)
from ...Research_Workspace.contracts import QualifiedWorkspaceRef, ResearchCapability
from ...Research_Workspace.studio_models import PRIMARY_RESEARCH_OUTPUT_KINDS
from ...Widgets.modal_dismissal import SafeModalDismissMixin


class ResearchMoreOutputs(
    SafeModalDismissMixin, ModalScreen[ResearchStudioAction | None]
):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    SAFE_MODAL_CONTENT = "#research-more-outputs-dialog"

    def __init__(
        self,
        workspace: QualifiedWorkspaceRef,
        capabilities: Mapping[str, ResearchCapability],
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.capabilities = dict(capabilities)

    def compose(self) -> ComposeResult:
        with Vertical(id="research-more-outputs-dialog"):
            yield Static("More outputs", classes="research-pane-title")
            yield Static(f"{self.workspace.data_source.value.title()} workspace")
            with VerticalScroll(id="research-more-output-list", name="Studio output choices"):
                yield from self._groups()
            yield Button("Close", id="research-more-close", compact=True)

    def _groups(self) -> ComposeResult:
        for group in ("Learn", "Analyze", "Present", "Work products"):
            yield Static(group, classes="research-pane-title research-more-heading")
            for action in RESEARCH_ACTIONS.values():
                if action.group != group:
                    continue
                if action.kind in PRIMARY_RESEARCH_OUTPUT_KINDS:
                    yield Static(action.label, classes="research-more-orientation")
                    continue
                key = action.action_id.replace("_", "-")
                capability = self.capabilities[action.action_id]
                summary = (
                    f"{capability.owner} · {capability.user_message} "
                    f"{capability.recovery_action}"
                ).strip()
                yield Button(
                    action.label,
                    id=f"research-more-{key}",
                    name=action.action_id,
                    classes="research-more-action",
                    compact=True,
                    disabled=not capability.available,
                    tooltip=summary,
                )
                yield Static(summary, id=f"research-more-detail-{key}", markup=False)
        yield Static("Planned", classes="research-pane-title research-more-heading")
        for item in PLANNED_OUTPUTS:
            yield Static(item.label, classes="research-more-planned")

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        event.widget.scroll_visible(animate=False, immediate=True, force=True)

    def on_resize(self, _event: Resize) -> None:
        if self.focused is not None:
            self.focused.scroll_visible(animate=False, force=True)

    @on(Button.Pressed, ".research-more-action")
    def choose(self, event: Button.Pressed) -> None:
        action = RESEARCH_ACTIONS[event.button.name]
        if self.capabilities[action.action_id].available:
            self.dismiss(action.kind)

    @on(Button.Pressed, "#research-more-close")
    def close(self) -> None:
        self.dismiss(None)
