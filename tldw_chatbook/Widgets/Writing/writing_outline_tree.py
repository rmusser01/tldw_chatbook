"""Outline tree for Writing Suite project structures."""

from __future__ import annotations

from typing import Any, Mapping

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, Tree


class WritingOutlineTree(Vertical):
    """Source-honest project/manuscript/chapter/scene outline state."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.structure: Mapping[str, Any] | None = None
        self.labels: list[str] = []
        self.node_data: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Label("Outline")
        yield Tree("Writing Project", id="writing-outline-tree")

    def clear(self) -> None:
        self.structure = None
        self.labels = []
        self.node_data = []

    def set_structure(self, structure: Mapping[str, Any], *, source: str) -> None:
        self.structure = dict(structure or {})
        self.labels = []
        self.node_data = []
        project = self.structure.get("project")
        project_id = self._record_id(project) or str(self.structure.get("project_id") or "")
        if project is not None:
            self._append_node(source, "project", project, project_id=project_id)

        for manuscript_node in self.structure.get("manuscripts") or []:
            manuscript = self._mapping_get(manuscript_node, "manuscript")
            manuscript_id = self._record_id(manuscript)
            self._append_node(source, "manuscript", manuscript, project_id=project_id)
            for chapter_node in self._mapping_get(manuscript_node, "chapters") or []:
                chapter = self._mapping_get(chapter_node, "chapter")
                chapter_id = self._record_id(chapter)
                self._append_node(
                    source,
                    "chapter",
                    chapter,
                    project_id=project_id,
                    manuscript_id=manuscript_id,
                )
                for scene in self._mapping_get(chapter_node, "scenes") or []:
                    self._append_node(
                        source,
                        "scene",
                        scene,
                        project_id=project_id,
                        manuscript_id=manuscript_id,
                        chapter_id=chapter_id,
                    )
            for scene in self._mapping_get(manuscript_node, "direct_scenes") or []:
                self._append_node(
                    source,
                    "scene",
                    scene,
                    project_id=project_id,
                    manuscript_id=manuscript_id,
                    chapter_id=None,
                )

        unassigned_chapters = self.structure.get("unassigned_chapters") or []
        if unassigned_chapters:
            self.labels.append("Unassigned Chapters")
            self.node_data.append(
                {
                    "source": source,
                    "kind": "unassigned_chapters",
                    "id": None,
                    "project_id": project_id,
                    "title": "Unassigned Chapters",
                    "version": None,
                }
            )
            for chapter_node in unassigned_chapters:
                chapter = self._mapping_get(chapter_node, "chapter")
                chapter_id = self._record_id(chapter)
                self._append_node(
                    source,
                    "chapter",
                    chapter,
                    project_id=project_id,
                    manuscript_id=None,
                )
                for scene in self._mapping_get(chapter_node, "scenes") or []:
                    self._append_node(
                        source,
                        "scene",
                        scene,
                        project_id=project_id,
                        manuscript_id=None,
                        chapter_id=chapter_id,
                    )

    def _append_node(
        self,
        source: str,
        kind: str,
        record: Any,
        *,
        project_id: str,
        manuscript_id: str | None = None,
        chapter_id: str | None = None,
    ) -> None:
        title = self._record_title(record, fallback=f"Untitled {kind.title()}")
        self.labels.append(title)
        self.node_data.append(
            {
                "source": source,
                "kind": kind,
                "id": self._record_id(record),
                "project_id": project_id,
                "manuscript_id": manuscript_id,
                "chapter_id": chapter_id,
                "title": title,
                "version": self._record_get(record, "version"),
            }
        )

    @staticmethod
    def _mapping_get(record: Any, key: str, default: Any = None) -> Any:
        if isinstance(record, Mapping):
            return record.get(key, default)
        return getattr(record, key, default)

    @classmethod
    def _record_get(cls, record: Any, key: str, default: Any = None) -> Any:
        return cls._mapping_get(record, key, default)

    @classmethod
    def _record_id(cls, record: Any) -> str:
        value = cls._record_get(record, "id", "")
        return str(value or "")

    @classmethod
    def _record_title(cls, record: Any, *, fallback: str) -> str:
        value = cls._record_get(record, "title", None)
        return str(value or fallback)
