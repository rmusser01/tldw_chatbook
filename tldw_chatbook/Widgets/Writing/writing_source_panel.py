"""Source and project browser panel for Writing Suite."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, ListView, Select, Static


class WritingSourcePanel(Vertical):
    """Source switch and project list state for Writing Suite."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.current_source = "local"
        self.projects: list[Any] = []
        self.project_titles: list[str] = []
        self.project_ids: list[str] = []
        self.notice = ""

    def compose(self) -> ComposeResult:
        yield Label("Writing Projects")
        with Horizontal(classes="writing-source-row"):
            yield Select(
                [("Local", "local"), ("Server", "server")],
                value=self.current_source,
                allow_blank=False,
                id="writing-source-select",
            )
            yield Button("Create Project", id="writing-create-project", variant="primary")
        yield Static(self.notice, id="writing-source-status")
        yield ListView(id="writing-project-list", classes="writing-project-list")

    def set_source(self, source: str) -> None:
        self.current_source = source if source in {"local", "server"} else "local"

    def clear_projects(self) -> None:
        self.projects = []
        self.project_titles = []
        self.project_ids = []

    def set_projects(self, projects: list[Any]) -> None:
        self.projects = list(projects or [])
        self.project_titles = [self._record_title(project) for project in self.projects]
        self.project_ids = [self._record_id(project) for project in self.projects]

    def set_notice(self, message: str) -> None:
        self.notice = message

    @staticmethod
    def _record_title(record: Any) -> str:
        if isinstance(record, dict):
            return str(record.get("title") or "Untitled Project")
        return str(getattr(record, "title", None) or "Untitled Project")

    @staticmethod
    def _record_id(record: Any) -> str:
        if isinstance(record, dict):
            return str(record.get("id") or "")
        return str(getattr(record, "id", "") or "")
