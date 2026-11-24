"""Configure an output and expose actions for its exact owner."""

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Select,
    Static,
    TextArea,
)

from ...Research_Workspace.chat_models import ResearchRetrievalSettings
from ...Research_Workspace.studio_models import (
    ResearchGenerationConfig,
    ResearchOutputKind,
    StudioOutputKind,
)
from ...Research_Workspace.work_product_templates import TEXT_WORK_PRODUCT_TEMPLATES
from ...Widgets.modal_dismissal import SafeModalDismissMixin
from .studio_history import StudioHistoryRow
from .studio_output_card import output_label


class ResearchStudioOptions(
    SafeModalDismissMixin, ModalScreen[ResearchGenerationConfig | None]
):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-studio-dialog"

    def __init__(
        self,
        kind: StudioOutputKind,
        owner: str,
        route: str,
        source_count: int,
        *,
        action: str = "Generate",
    ) -> None:
        super().__init__()
        self.kind = kind
        self.owner = owner
        self.route = route
        self.source_count = source_count
        self.action_label = action

    @property
    def is_study(self) -> bool:
        return self.kind in {ResearchOutputKind.FLASHCARDS, ResearchOutputKind.QUIZ}

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="research-studio-dialog"):
            yield Static(output_label(self.kind), classes="research-pane-title")
            yield Static(
                f"{self.source_count} selected sources · Save to {self.owner}\n{self.route}",
                markup=False,
            )
            yield Static("Title")
            yield Input(placeholder=output_label(self.kind), id="research-studio-title")
            yield Static("Retrieval focus (optional for FTS)")
            yield Input(id="research-studio-focus")
            if self.is_study:
                yield Static("Study items")
                yield Input("8", type="integer", id="research-studio-items")
            yield Static("Detail")
            yield Select(
                [
                    ("Brief", "brief"),
                    ("Standard", "standard"),
                    ("Detailed", "detailed"),
                ],
                value="standard",
                allow_blank=False,
                id="research-studio-detail",
            )
            with Collapsible(title="Advanced options"):
                yield Static("Retrieval method")
                yield Select(
                    [("FTS", "fts"), ("Vector", "semantic"), ("Hybrid", "hybrid")],
                    value="fts",
                    allow_blank=False,
                    id="research-studio-retrieval",
                )
                yield Static("Top K")
                yield Input("8", type="integer", id="research-studio-top-k")
                yield Static("Score threshold (optional)")
                yield Input(type="number", id="research-studio-threshold")
                yield Checkbox("Rerank evidence", id="research-studio-rerank")
                yield Checkbox(
                    "Source citations required"
                    if self.kind in TEXT_WORK_PRODUCT_TEMPLATES
                    else "Include source citations",
                    value=True,
                    id="research-studio-citations",
                    disabled=self.kind in TEXT_WORK_PRODUCT_TEMPLATES,
                )
                yield Static("Temperature")
                yield Input("0.3", type="number", id="research-studio-temperature")
                yield Static("Top P")
                yield Input("1.0", type="number", id="research-studio-top-p")
                yield Static("Max output tokens")
                yield Input("4096", type="integer", id="research-studio-tokens")
            yield Static("", id="research-studio-options-status", markup=False)
            yield Button(
                self.action_label,
                id="research-studio-generate",
                variant="primary",
                compact=True,
            )
            yield Button("Cancel", id="research-studio-close", compact=True)

    def _input(self, name: str) -> str:
        return self.query_one(f"#research-studio-{name}", Input).value

    @on(Button.Pressed, "#research-studio-generate")
    def submit(self) -> None:
        try:
            threshold = self._input("threshold")
            retrieval = ResearchRetrievalSettings(
                mode=self.query_one("#research-studio-retrieval", Select).value,
                top_k=int(self._input("top-k")),
                threshold=float(threshold) if threshold else None,
                rerank=self.query_one("#research-studio-rerank", Checkbox).value,
                citations=self.query_one("#research-studio-citations", Checkbox).value,
                history_limit=0,
            )
            options = ResearchGenerationConfig(
                title=self._input("title"),
                focus=self._input("focus"),
                detail=self.query_one("#research-studio-detail", Select).value,
                item_count=int(self._input("items")) if self.is_study else 8,
                temperature=float(self._input("temperature")),
                top_p=float(self._input("top-p")),
                max_tokens=int(self._input("tokens")),
                retrieval=retrieval,
            )
        except ValueError as exc:
            self.query_one("#research-studio-options-status", Static).update(str(exc))
            return
        self.dismiss(options)

    @on(Button.Pressed, "#research-studio-close")
    def close(self) -> None:
        self.dismiss(None)


class ResearchStudioActions(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    SAFE_MODAL_CONTENT = "#research-studio-actions-dialog"

    def __init__(
        self, row: StudioHistoryRow, actions: dict[str, str], limits: str, owner: str
    ) -> None:
        super().__init__()
        self.row = row
        self.actions = actions
        self.limits = limits
        self.owner = owner

    def compose(self) -> ComposeResult:
        with Vertical(id="research-studio-actions-dialog"):
            with VerticalScroll(classes="research-studio-action-scroll"):
                yield Static(
                    self.row.title, classes="research-pane-title", markup=False
                )
                yield Static(self.owner, markup=False)
                if self.row.sources_changed:
                    yield Static("Sources changed since generation")
                if self.row.content:
                    yield Static(self.row.content, markup=False)
                if self.limits:
                    yield Static(self.limits, markup=False)
                for action, label in self.actions.items():
                    yield Button(
                        label, id=f"research-studio-action-{action}", compact=True
                    )
            yield Button("Close", id="research-studio-actions-close", compact=True)

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "research-studio-actions-close":
            self.dismiss(None)
        elif button_id.startswith("research-studio-action-"):
            self.dismiss(button_id.removeprefix("research-studio-action-"))


class ResearchStudioEdit(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-studio-edit-dialog"

    def __init__(self, row: StudioHistoryRow) -> None:
        super().__init__()
        self.row = row

    def compose(self) -> ComposeResult:
        with Vertical(id="research-studio-edit-dialog"):
            yield Static(self.row.title, classes="research-pane-title", markup=False)
            yield TextArea(self.row.content, id="research-studio-edit-body")
            yield Button(
                "Save edit",
                id="research-studio-edit-save",
                compact=True,
                variant="primary",
            )
            yield Button("Cancel", id="research-studio-edit-cancel", compact=True)

    @on(Button.Pressed, "#research-studio-edit-save")
    def save(self) -> None:
        self.dismiss(self.query_one(TextArea).text)

    @on(Button.Pressed, "#research-studio-edit-cancel")
    def close(self) -> None:
        self.dismiss(None)
