import json

from tldw_chatbook.Writing_Interop.writing_normalizers import (
    normalize_writing_record,
    normalize_writing_structure,
)


def test_scene_normalizer_prefers_chatbook_markdown_wrapper_over_plain_fallback():
    normalized = normalize_writing_record(
        "server",
        "scene",
        {
            "id": "scene-1",
            "chapter_id": "chapter-1",
            "project_id": "project-1",
            "title": "Scene",
            "content_json": json.dumps(
                {
                    "type": "chatbook-markdown",
                    "schema_version": 1,
                    "markdown": "# Opening\n\nBody text.",
                }
            ),
            "content_plain": "Opening Body text.",
            "version": 1,
        },
    )

    assert normalized["content_markdown"] == "# Opening\n\nBody text."
    assert normalized["content_markdown_fidelity"] == "chatbook_markdown"


def test_scene_normalizer_marks_plain_text_fallback_fidelity():
    normalized = normalize_writing_record(
        "server",
        "scene",
        {
            "id": "scene-1",
            "chapter_id": "chapter-1",
            "project_id": "project-1",
            "title": "Scene",
            "content_plain": "Opening line.",
            "version": 1,
        },
    )

    assert normalized["content_markdown"] == "Opening line."
    assert normalized["content_markdown_fidelity"] == "plain_text_fallback"


def test_structure_normalizer_keeps_unassigned_chapters_in_project_bucket():
    normalized = normalize_writing_structure(
        "server",
        {
            "project_id": "project-1",
            "parts": [],
            "unassigned_chapters": [
                {
                    "id": "chapter-loose",
                    "title": "Loose Chapter",
                    "sort_order": 0.0,
                    "part_id": None,
                    "version": 1,
                    "scenes": [],
                }
            ],
        },
    )

    chapter = normalized["unassigned_chapters"][0]

    assert normalized["manuscripts"] == []
    assert chapter["record_id"] == "server:writing_chapter:chapter-loose"
    assert chapter["manuscript_id"] is None
    assert chapter["outline_bucket"] == "unassigned_chapters"
    assert chapter["outline_parent_type"] == "project"
    assert chapter["outline_parent_id"] == "project-1"
