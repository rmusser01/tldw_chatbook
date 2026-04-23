"""Local/server normalization helpers for writing interop models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_chatbook.Writing_Interop.writing_markdown_adapter import (
    parse_server_content_json,
    server_content_to_markdown,
)
from tldw_chatbook.Writing_Interop.writing_models import (
    WritingChapter,
    WritingManuscript,
    WritingOutlineNode,
    WritingProject,
    WritingScene,
)


def _as_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dict(dumped)
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def normalize_local_project_row(row: Any) -> WritingProject:
    data = _as_mapping(row)
    return WritingProject(
        source="local",
        id=str(data.get("id")),
        title=str(data.get("title") or "Untitled Project"),
        subtitle=data.get("subtitle"),
        author=data.get("author"),
        genre=data.get("genre"),
        status=str(data.get("status") or "draft"),
        synopsis=data.get("synopsis"),
        target_word_count=data.get("target_word_count"),
        word_count=int(data.get("word_count") or 0),
        version=int(data.get("version") or 1),
        deleted=bool(data.get("deleted") or data.get("is_deleted")),
        metadata=dict(data.get("metadata") or {}),
    )


def normalize_local_manuscript_row(row: Any) -> WritingManuscript:
    data = _as_mapping(row)
    return WritingManuscript(
        source="local",
        id=str(data.get("id")),
        project_id=str(data.get("project_id")),
        title=str(data.get("title") or "Untitled Manuscript"),
        synopsis=data.get("synopsis"),
        status=str(data.get("status") or "draft"),
        word_count=int(data.get("word_count") or 0),
        sort_order=float(data.get("sort_order") or 0.0),
        version=int(data.get("version") or 1),
        deleted=bool(data.get("deleted") or data.get("is_deleted")),
        metadata=dict(data.get("metadata") or {}),
    )


def normalize_local_chapter_row(row: Any) -> WritingChapter:
    data = _as_mapping(row)
    return WritingChapter(
        source="local",
        id=str(data.get("id")),
        project_id=str(data.get("project_id")),
        manuscript_id=data.get("manuscript_id"),
        title=str(data.get("title") or "Untitled Chapter"),
        synopsis=data.get("synopsis"),
        status=str(data.get("status") or "draft"),
        word_count=int(data.get("word_count") or 0),
        sort_order=float(data.get("sort_order") or 0.0),
        version=int(data.get("version") or 1),
        deleted=bool(data.get("deleted") or data.get("is_deleted")),
        metadata=dict(data.get("metadata") or {}),
    )


def normalize_local_scene_row(row: Any) -> WritingScene:
    data = _as_mapping(row)
    chapter_id = data.get("chapter_id")
    return WritingScene(
        source="local",
        id=str(data.get("id")),
        project_id=str(data.get("project_id")),
        title=str(data.get("title") or "Untitled Scene"),
        chapter_id=chapter_id,
        manuscript_id=None if chapter_id is not None else data.get("manuscript_id"),
        body_markdown=str(data.get("body_markdown") or ""),
        synopsis=data.get("synopsis"),
        status=str(data.get("status") or "draft"),
        word_count=int(data.get("word_count") or 0),
        sort_order=float(data.get("sort_order") or 0.0),
        version=int(data.get("version") or 1),
        deleted=bool(data.get("deleted") or data.get("is_deleted")),
        metadata=dict(data.get("metadata") or {}),
    )


def normalize_server_project(value: Any) -> WritingProject:
    data = _as_mapping(value)
    return WritingProject(
        source="server",
        id=str(data.get("id")),
        title=str(data.get("title") or "Untitled Project"),
        subtitle=data.get("subtitle"),
        author=data.get("author"),
        genre=data.get("genre"),
        status=str(data.get("status") or "draft"),
        synopsis=data.get("synopsis"),
        target_word_count=data.get("target_word_count"),
        word_count=int(data.get("word_count") or 0),
        version=int(data.get("version") or 1),
        deleted=bool(data.get("deleted")),
        metadata=dict(data.get("settings") or {}),
    )


def normalize_server_part(value: Any) -> WritingManuscript:
    data = _as_mapping(value)
    return WritingManuscript(
        source="server",
        id=str(data.get("id")),
        project_id=str(data.get("project_id")),
        title=str(data.get("title") or "Untitled Manuscript"),
        synopsis=data.get("synopsis"),
        word_count=int(data.get("word_count") or 0),
        sort_order=float(data.get("sort_order") or 0.0),
        version=int(data.get("version") or 1),
        deleted=bool(data.get("deleted")),
    )


def normalize_server_chapter(value: Any) -> WritingChapter:
    data = _as_mapping(value)
    return WritingChapter(
        source="server",
        id=str(data.get("id")),
        project_id=str(data.get("project_id")),
        manuscript_id=data.get("part_id"),
        title=str(data.get("title") or "Untitled Chapter"),
        synopsis=data.get("synopsis"),
        status=str(data.get("status") or "draft"),
        word_count=int(data.get("word_count") or 0),
        sort_order=float(data.get("sort_order") or 0.0),
        version=int(data.get("version") or 1),
        deleted=bool(data.get("deleted")),
    )


def normalize_server_scene(value: Any) -> WritingScene:
    data = _as_mapping(value)
    content = data.get("content")
    content_json = data.get("content_json")
    parsed_content = content if isinstance(content, Mapping) else parse_server_content_json(content_json)
    body_markdown = server_content_to_markdown(parsed_content, data.get("content_plain"))
    chapter_id = data.get("chapter_id")
    return WritingScene(
        source="server",
        id=str(data.get("id")),
        project_id=str(data.get("project_id")),
        title=str(data.get("title") or "Untitled Scene"),
        chapter_id=chapter_id,
        manuscript_id=None if chapter_id is not None else data.get("part_id"),
        body_markdown=body_markdown,
        synopsis=data.get("synopsis"),
        status=str(data.get("status") or "draft"),
        word_count=int(data.get("word_count") or 0),
        sort_order=float(data.get("sort_order") or 0.0),
        version=int(data.get("version") or 1),
        deleted=bool(data.get("deleted")),
    )


def normalize_server_structure_outline(value: Any) -> list[WritingOutlineNode]:
    data = _as_mapping(value)
    project_id = str(data.get("project_id"))
    nodes: list[WritingOutlineNode] = []

    for part in data.get("parts") or []:
        part_data = _as_mapping(part)
        part_id = str(part_data.get("id"))
        nodes.append(
            WritingOutlineNode(
                source="server",
                kind="manuscript",
                id=part_id,
                project_id=project_id,
                parent_id=None,
                title=str(part_data.get("title") or "Untitled Manuscript"),
                entity_kind="manuscript",
                sort_order=float(part_data.get("sort_order") or 0.0),
            )
        )

        for chapter in part_data.get("chapters") or []:
            chapter_data = _as_mapping(chapter)
            chapter_id = str(chapter_data.get("id"))
            nodes.append(
                WritingOutlineNode(
                    source="server",
                    kind="chapter",
                    id=chapter_id,
                    project_id=project_id,
                    parent_id=part_id,
                    title=str(chapter_data.get("title") or "Untitled Chapter"),
                    entity_kind="chapter",
                    sort_order=float(chapter_data.get("sort_order") or 0.0),
                )
            )

            for scene in chapter_data.get("scenes") or []:
                scene_data = _as_mapping(scene)
                nodes.append(
                    WritingOutlineNode(
                        source="server",
                        kind="scene",
                        id=str(scene_data.get("id")),
                        project_id=project_id,
                        parent_id=chapter_id,
                        title=str(scene_data.get("title") or "Untitled Scene"),
                        entity_kind="scene",
                        sort_order=float(scene_data.get("sort_order") or 0.0),
                    )
                )

    unassigned = data.get("unassigned_chapters") or []
    if unassigned:
        bucket_id = f"{project_id}:unassigned_chapters"
        nodes.append(
            WritingOutlineNode(
                source="server",
                kind="unassigned_chapters",
                id=bucket_id,
                project_id=project_id,
                parent_id=None,
                title="Unassigned Chapters",
            )
        )
        for chapter in unassigned:
            chapter_data = _as_mapping(chapter)
            chapter_id = str(chapter_data.get("id"))
            nodes.append(
                WritingOutlineNode(
                    source="server",
                    kind="chapter",
                    id=chapter_id,
                    project_id=project_id,
                    parent_id=bucket_id,
                    title=str(chapter_data.get("title") or "Untitled Chapter"),
                    entity_kind="chapter",
                    sort_order=float(chapter_data.get("sort_order") or 0.0),
                )
            )

            for scene in chapter_data.get("scenes") or []:
                scene_data = _as_mapping(scene)
                nodes.append(
                    WritingOutlineNode(
                        source="server",
                        kind="scene",
                        id=str(scene_data.get("id")),
                        project_id=project_id,
                        parent_id=chapter_id,
                        title=str(scene_data.get("title") or "Untitled Scene"),
                        entity_kind="scene",
                        sort_order=float(scene_data.get("sort_order") or 0.0),
                    )
                )

    return nodes
