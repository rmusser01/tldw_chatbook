"""A page of references resolved against the primary Studio owners."""

from collections.abc import Mapping
from dataclasses import dataclass

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Select, Static

from ...Research_Workspace.studio_models import WorkspaceOutputRef
from .studio_output_card import output_label


def history_key(output: WorkspaceOutputRef) -> str:
    return f"{output.owner_kind}:{output.owner_id}"


@dataclass(frozen=True, slots=True)
class StudioHistoryRow:
    output: WorkspaceOutputRef
    record: Mapping | None
    sources_changed: bool

    @property
    def title(self) -> str:
        record = self.record or {}
        return str(
            record.get("title")
            or record.get("name")
            or output_label(self.output.output_kind)
        )

    @property
    def content(self) -> str:
        record = self.record or {}
        metadata = record.get("metadata")
        content = (
            metadata.get("content")
            if isinstance(metadata, Mapping)
            else record.get("content")
        )
        return content if isinstance(content, str) else ""


class ResearchStudioHistory(Vertical):
    def __init__(self) -> None:
        super().__init__(classes="research-studio-history")
        self.rows: dict[str, StudioHistoryRow] = {}

    def compose(self) -> ComposeResult:
        yield Static("Output history", classes="research-pane-title")
        yield Select([], prompt="Saved outputs", id="research-studio-history")
        yield Static("", id="research-studio-history-detail", markup=False)
        yield Button(
            "Output actions",
            id="research-studio-output-actions",
            compact=True,
            disabled=True,
        )
        yield Button(
            "Previous outputs",
            id="research-studio-previous",
            compact=True,
            disabled=True,
        )
        yield Button(
            "More saved outputs", id="research-studio-next", compact=True, disabled=True
        )

    @property
    def current(self) -> StudioHistoryRow | None:
        value = self.query_one(Select).value
        return self.rows.get(value) if isinstance(value, str) else None

    def sync_page(self, rows, *, preferred="", offset=0, has_more=False) -> None:
        select = self.query_one(Select)
        previous = preferred or select.value
        self.rows = {history_key(row.output): row for row in rows}
        select.set_options(
            (Text(row.title), owner_id) for owner_id, row in self.rows.items()
        )
        select.value = (
            previous if previous in self.rows else next(iter(self.rows), Select.NULL)
        )
        self.query_one("#research-studio-previous", Button).disabled = not offset
        self.query_one("#research-studio-next", Button).disabled = not has_more
        self.sync_selection()

    def sync_selection(self) -> None:
        row = self.current
        detail = "No saved outputs."
        if row:
            detail = (
                f"Version {row.output.owner_version}"
                if row.record
                else "Missing canonical item"
            )
            if row.sources_changed:
                detail += " · Sources changed since generation"
        self.query_one("#research-studio-history-detail", Static).update(detail)
        self.query_one("#research-studio-output-actions", Button).disabled = row is None
