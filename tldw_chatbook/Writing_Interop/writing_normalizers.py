"""Normalization helpers for source-aware writing-suite records."""

from __future__ import annotations

from typing import Any, Mapping


_KIND_TO_RECORD_TYPE = {
    "project": "writing_project",
    "manuscript": "writing_manuscript",
    "chapter": "writing_chapter",
    "scene": "writing_scene",
}


def normalize_writing_record(source: str, kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a source-labeled Chatbook record without hiding backend-native IDs."""

    payload = dict(record)
    record_type = _KIND_TO_RECORD_TYPE[kind]
    record_id = str(payload.get("id"))
    payload["source"] = source
    payload["record_type"] = record_type
    payload["record_id"] = f"{source}:{record_type}:{record_id}"

    if kind == "chapter":
        payload["manuscript_id"] = payload.get("manuscript_id", payload.get("part_id"))
    if kind == "scene":
        payload["content_markdown"] = payload.get("content_markdown", payload.get("content_plain") or "")
    return payload


def normalize_writing_structure(source: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Map server `parts` and local manuscripts into Chatbook's manuscript vocabulary."""

    def normalize_scene_summary(scene: Mapping[str, Any]) -> dict[str, Any]:
        return normalize_writing_record(source, "scene", scene)

    def normalize_chapter_summary(chapter: Mapping[str, Any]) -> dict[str, Any]:
        normalized = normalize_writing_record(source, "chapter", chapter)
        normalized["scenes"] = [normalize_scene_summary(scene) for scene in chapter.get("scenes", [])]
        return normalized

    def normalize_manuscript_summary(manuscript: Mapping[str, Any]) -> dict[str, Any]:
        normalized = normalize_writing_record(source, "manuscript", manuscript)
        normalized["chapters"] = [
            normalize_chapter_summary(chapter)
            for chapter in manuscript.get("chapters", [])
        ]
        return normalized

    manuscripts = payload.get("manuscripts", payload.get("parts", []))
    return {
        "source": source,
        "project_id": payload.get("project_id"),
        "manuscripts": [normalize_manuscript_summary(item) for item in manuscripts],
        "unassigned_chapters": [
            normalize_chapter_summary(chapter)
            for chapter in payload.get("unassigned_chapters", [])
        ],
    }
