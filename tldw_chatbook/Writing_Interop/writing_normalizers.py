"""Local/server normalization helpers for writing interop models."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
from typing import Any

from tldw_chatbook.Writing_Interop.writing_markdown_adapter import (
    parse_server_content_json,
    server_content_to_markdown,
)
from tldw_chatbook.Writing_Interop.writing_models import (
    WritingCapability,
    WritingChapter,
    WritingDraft,
    WritingManuscript,
    WritingOutlineNode,
    WritingProject,
    WritingScene,
    WritingTrashEntry,
    WritingVersion,
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


def _required_str(value: Any, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def normalize_local_project_row(row: Any) -> WritingProject:
    data = _as_mapping(row)
    return WritingProject(
        source="local",
        id=_required_str(data.get("id"), field_name="id"),
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
        id=_required_str(data.get("id"), field_name="id"),
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
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
        id=_required_str(data.get("id"), field_name="id"),
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
        manuscript_id=_optional_str(data.get("manuscript_id")),
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
    chapter_id = _optional_str(data.get("chapter_id"))
    return WritingScene(
        source="local",
        id=_required_str(data.get("id"), field_name="id"),
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
        title=str(data.get("title") or "Untitled Scene"),
        chapter_id=chapter_id,
        manuscript_id=None if chapter_id is not None else _optional_str(data.get("manuscript_id")),
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
        id=_required_str(data.get("id"), field_name="id"),
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
        id=_required_str(data.get("id"), field_name="id"),
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
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
        id=_required_str(data.get("id"), field_name="id"),
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
        manuscript_id=_optional_str(data.get("part_id")),
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
    chapter_id = _optional_str(data.get("chapter_id"))
    return WritingScene(
        source="server",
        id=_required_str(data.get("id"), field_name="id"),
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
        title=str(data.get("title") or "Untitled Scene"),
        chapter_id=chapter_id,
        manuscript_id=None if chapter_id is not None else _optional_str(data.get("part_id")),
        body_markdown=body_markdown,
        synopsis=data.get("synopsis"),
        status=str(data.get("status") or "draft"),
        word_count=int(data.get("word_count") or 0),
        sort_order=float(data.get("sort_order") or 0.0),
        version=int(data.get("version") or 1),
        deleted=bool(data.get("deleted")),
    )


def normalize_local_draft_row(row: Any) -> WritingDraft:
    data = _as_mapping(row)
    return WritingDraft(
        source="local",
        entity_kind=_required_str(data.get("entity_kind"), field_name="entity_kind"),
        entity_id=_required_str(data.get("entity_id"), field_name="entity_id"),
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
        metadata=_json_mapping(data.get("metadata") or data.get("metadata_json")),
        body_markdown=data.get("body_markdown"),
        updated_at=_parse_datetime(data.get("updated_at")),
    )


def normalize_server_draft(value: Any) -> WritingDraft:
    data = _as_mapping(value)
    entity_kind = _required_str(data.get("entity_kind") or data.get("kind"), field_name="entity_kind")
    content = data.get("content")
    content_json = data.get("content_json")
    parsed_content = content if isinstance(content, Mapping) else parse_server_content_json(content_json)
    return WritingDraft(
        source="server",
        entity_kind=entity_kind,
        entity_id=_required_str(data.get("entity_id") or data.get("id"), field_name="entity_id"),
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
        metadata=_json_mapping(data.get("metadata") or data.get("metadata_json") or data.get("snapshot")),
        body_markdown=(
            server_content_to_markdown(parsed_content, data.get("content_plain"))
            if entity_kind == "scene"
            else data.get("body_markdown")
        ),
        updated_at=_parse_datetime(data.get("updated_at") or data.get("last_modified")),
    )


def normalize_local_version_row(row: Any) -> WritingVersion:
    data = _as_mapping(row)
    return WritingVersion(
        source="local",
        id=_required_str(data.get("id"), field_name="id"),
        entity_kind=_required_str(data.get("entity_kind"), field_name="entity_kind"),
        entity_id=_required_str(data.get("entity_id"), field_name="entity_id"),
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
        version_number=int(data.get("version_number") or 1),
        metadata=_json_mapping(data.get("snapshot_json") or data.get("metadata") or data.get("metadata_json")),
        body_markdown=data.get("body_markdown"),
        created_at=_parse_datetime(data.get("created_at")),
    )


def normalize_server_version(value: Any) -> WritingVersion:
    data = _as_mapping(value)
    entity_kind = _required_str(data.get("entity_kind") or data.get("kind"), field_name="entity_kind")
    content = data.get("content")
    content_json = data.get("content_json")
    parsed_content = content if isinstance(content, Mapping) else parse_server_content_json(content_json)
    body_markdown = server_content_to_markdown(parsed_content, data.get("content_plain"))
    return WritingVersion(
        source="server",
        id=_required_str(data.get("id"), field_name="id"),
        entity_kind=entity_kind,
        entity_id=_required_str(data.get("entity_id"), field_name="entity_id"),
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
        version_number=int(data.get("version_number") or data.get("version") or 1),
        metadata=_json_mapping(data.get("snapshot") or data.get("snapshot_json") or data.get("metadata")),
        body_markdown=body_markdown if entity_kind == "scene" else data.get("body_markdown"),
        created_at=_parse_datetime(data.get("created_at")),
    )


def normalize_local_trash_row(row: Any) -> WritingTrashEntry:
    data = _as_mapping(row)
    entity_kind = _required_str(data.get("entity_kind") or data.get("kind"), field_name="entity_kind")
    entity_id = _required_str(data.get("entity_id"), field_name="entity_id")
    return WritingTrashEntry(
        source="local",
        id=_required_str(data.get("id") or f"{entity_kind}:{entity_id}", field_name="id"),
        entity_kind=entity_kind,
        entity_id=entity_id,
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
        title=str(data.get("title") or "Untitled"),
        deleted_at=_parse_datetime(data.get("deleted_at")),
        metadata=_json_mapping(data.get("metadata") or data.get("metadata_json")),
    )


def normalize_server_trash(value: Any) -> WritingTrashEntry:
    data = _as_mapping(value)
    entity_kind = _required_str(data.get("entity_kind") or data.get("kind"), field_name="entity_kind")
    entity_id = _required_str(data.get("entity_id"), field_name="entity_id")
    return WritingTrashEntry(
        source="server",
        id=_required_str(data.get("id") or f"{entity_kind}:{entity_id}", field_name="id"),
        entity_kind=entity_kind,
        entity_id=entity_id,
        project_id=_required_str(data.get("project_id"), field_name="project_id"),
        title=str(data.get("title") or "Untitled"),
        deleted_at=_parse_datetime(data.get("deleted_at")),
        metadata=_json_mapping(data.get("metadata") or data.get("metadata_json")),
    )


def normalize_local_capability_row(row: Any) -> WritingCapability:
    data = _as_mapping(row)
    return WritingCapability(
        source="local",
        name=_required_str(data.get("name") or data.get("capability"), field_name="name"),
        supported=bool(data.get("supported", data.get("available", False))),
        reason=data.get("reason") or data.get("reason_code"),
        metadata=_json_mapping(data.get("metadata") or data.get("metadata_json")),
    )


def normalize_server_capability(value: Any) -> WritingCapability:
    data = _as_mapping(value)
    return WritingCapability(
        source="server",
        name=_required_str(data.get("name") or data.get("capability"), field_name="name"),
        supported=bool(data.get("supported", data.get("available", False))),
        reason=data.get("reason") or data.get("reason_code"),
        metadata=_json_mapping(data.get("metadata") or data.get("metadata_json")),
    )


def normalize_server_structure_outline(value: Any) -> list[WritingOutlineNode]:
    data = _as_mapping(value)
    project_id = _required_str(data.get("project_id"), field_name="project_id")
    nodes: list[WritingOutlineNode] = []

    for part in data.get("parts") or []:
        part_data = _as_mapping(part)
        part_id = _required_str(part_data.get("id"), field_name="id")
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
            chapter_id = _required_str(chapter_data.get("id"), field_name="id")
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
                        id=_required_str(scene_data.get("id"), field_name="id"),
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
            chapter_id = _required_str(chapter_data.get("id"), field_name="id")
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
                        id=_required_str(scene_data.get("id"), field_name="id"),
                        project_id=project_id,
                        parent_id=chapter_id,
                        title=str(scene_data.get("title") or "Untitled Scene"),
                        entity_kind="scene",
                        sort_order=float(scene_data.get("sort_order") or 0.0),
                    )
                )

    return nodes
