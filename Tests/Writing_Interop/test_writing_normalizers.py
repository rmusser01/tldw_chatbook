from datetime import datetime, timezone

from tldw_chatbook.Writing_Interop.writing_models import WritingOutlineNode
from tldw_chatbook.Writing_Interop.writing_normalizers import (
    normalize_local_draft_row,
    normalize_local_chapter_row,
    normalize_local_capability_row,
    normalize_local_manuscript_row,
    normalize_local_project_row,
    normalize_local_scene_row,
    normalize_local_trash_row,
    normalize_local_version_row,
    normalize_server_capability,
    normalize_server_part,
    normalize_server_project,
    normalize_server_scene,
    normalize_server_structure_outline,
    normalize_server_trash,
    normalize_server_version,
)
from tldw_chatbook.tldw_api.writing_manuscript_schemas import (
    ChapterSummary,
    ManuscriptPartResponse,
    ManuscriptProjectResponse,
    ManuscriptSceneResponse,
    ManuscriptStructureResponse,
    PartSummary,
)


def test_normalize_local_rows_to_models():
    project = normalize_local_project_row({"id": "p1", "title": "Project 1"})
    manuscript = normalize_local_manuscript_row({"id": "m1", "project_id": "p1", "title": "Manuscript 1"})
    chapter = normalize_local_chapter_row({"id": "c1", "project_id": "p1", "manuscript_id": "m1", "title": "Ch 1"})
    scene = normalize_local_scene_row(
        {
            "id": "s1",
            "project_id": "p1",
            "chapter_id": "c1",
            "manuscript_id": "m1",
            "title": "Scene 1",
            "body_markdown": "Hello",
        }
    )

    assert project.id == "p1"
    assert manuscript.project_id == "p1"
    assert chapter.manuscript_id == "m1"
    assert scene.chapter_id == "c1"
    assert scene.manuscript_id is None
    assert scene.body_markdown == "Hello"


def test_required_ids_do_not_normalize_to_literal_none():
    for normalizer, row in [
        (normalize_local_project_row, {"title": "Missing id"}),
        (normalize_local_manuscript_row, {"id": "m1", "title": "Missing project"}),
        (normalize_local_chapter_row, {"id": "c1", "title": "Missing project"}),
        (normalize_local_scene_row, {"id": "s1", "title": "Missing project", "chapter_id": "c1"}),
    ]:
        try:
            normalizer(row)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{normalizer.__name__} accepted missing required IDs")


def test_normalize_draft_version_trash_and_capability_shapes():
    draft = normalize_local_draft_row(
        {
            "entity_kind": "scene",
            "entity_id": "scene-1",
            "project_id": "project-1",
            "body_markdown": "# Draft",
            "updated_at": "2026-04-22T12:00:00+00:00",
        }
    )
    version = normalize_local_version_row(
        {
            "id": "version-1",
            "entity_kind": "scene",
            "entity_id": "scene-1",
            "project_id": "project-1",
            "version_number": 2,
            "body_markdown": "# Snapshot",
            "snapshot_json": {"title": "Scene 1"},
            "created_at": "2026-04-22T12:05:00+00:00",
        }
    )
    trash = normalize_local_trash_row(
        {
            "id": "trash-scene-1",
            "entity_kind": "scene",
            "entity_id": "scene-1",
            "project_id": "project-1",
            "title": "Scene 1",
            "deleted_at": "2026-04-22T12:10:00+00:00",
        }
    )
    capability = normalize_local_capability_row(
        {
            "name": "create_version",
            "supported": True,
            "metadata": {"reason_source": "local"},
        }
    )

    assert draft.body_markdown == "# Draft"
    assert draft.updated_at is not None
    assert version.version_number == 2
    assert version.metadata == {"title": "Scene 1"}
    assert trash.title == "Scene 1"
    assert trash.deleted_at is not None
    assert capability.source == "local"
    assert capability.name == "create_version"
    assert capability.supported is True


def test_normalize_server_version_trash_and_capability_shapes():
    version = normalize_server_version(
        {
            "id": "server-version-1",
            "entity_kind": "scene",
            "entity_id": "scene-1",
            "project_id": "project-1",
            "version_number": 1,
            "content_json": '{"type":"doc","content":[{"type":"paragraph","attrs":{"tldw_chatbook_markdown":true,"format":"markdown","version":1},"content":[{"type":"text","text":"# Server Snapshot"}]}]}',
            "snapshot": {"title": "Scene 1"},
        }
    )
    trash = normalize_server_trash(
        {
            "id": "server-trash-1",
            "kind": "chapter",
            "entity_id": "chapter-1",
            "project_id": "project-1",
            "title": "Chapter 1",
        }
    )
    capability = normalize_server_capability(
        {
            "name": "create_version",
            "supported": False,
            "reason": "server_version_history_unavailable",
        }
    )

    assert version.source == "server"
    assert version.body_markdown == "# Server Snapshot"
    assert version.metadata == {"title": "Scene 1"}
    assert trash.source == "server"
    assert trash.entity_kind == "chapter"
    assert capability.source == "server"
    assert capability.supported is False
    assert capability.reason == "server_version_history_unavailable"


def test_normalize_server_scene_prefers_markdown_wrapper_and_falls_back_to_plain():
    wrapped = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {"tldw_chatbook_markdown": True, "format": "markdown", "version": 1},
                "content": [{"type": "text", "text": "# Wrapped"}],
            }
        ],
    }
    scene = ManuscriptSceneResponse(
        id="s1",
        chapter_id="c1",
        project_id="p1",
        title="Scene 1",
        sort_order=1,
        content_json='{"type":"doc","content":[{"type":"paragraph","attrs":{"tldw_chatbook_markdown":true,"format":"markdown","version":1},"content":[{"type":"text","text":"# Wrapped"}]}]}',
        content_plain="fallback",
        created_at=datetime.now(timezone.utc),
        last_modified=datetime.now(timezone.utc),
        client_id="client",
        version=1,
    )
    normalized = normalize_server_scene(scene)
    assert normalized.body_markdown == "# Wrapped"

    non_wrapped = ManuscriptSceneResponse(
        id="s2",
        chapter_id="c1",
        project_id="p1",
        title="Scene 2",
        sort_order=2,
        content_json='{"type":"doc","content":[]}',
        content_plain="plain fallback",
        created_at=datetime.now(timezone.utc),
        last_modified=datetime.now(timezone.utc),
        client_id="client",
        version=1,
    )
    fallback_normalized = normalize_server_scene(non_wrapped)
    assert fallback_normalized.body_markdown == "plain fallback"


def test_normalize_server_project_and_part_to_models():
    project = ManuscriptProjectResponse(
        id="p1",
        title="Project 1",
        status="draft",
        created_at=datetime.now(timezone.utc),
        last_modified=datetime.now(timezone.utc),
        client_id="client",
        version=1,
    )
    part = ManuscriptPartResponse(
        id="m1",
        project_id="p1",
        title="Part 1",
        sort_order=1,
        created_at=datetime.now(timezone.utc),
        last_modified=datetime.now(timezone.utc),
        client_id="client",
        version=1,
    )

    normalized_project = normalize_server_project(project)
    normalized_part = normalize_server_part(part)
    assert normalized_project.id == "p1"
    assert normalized_part.project_id == "p1"


def test_server_unassigned_chapters_are_outline_bucket_not_fake_manuscript():
    structure = ManuscriptStructureResponse(
        project_id="project-1",
        parts=[
            PartSummary(
                id="part-1",
                title="Part 1",
                sort_order=1.0,
                chapters=[],
            )
        ],
        unassigned_chapters=[
            ChapterSummary(
                id="chapter-u1",
                title="Loose Chapter",
                sort_order=5.0,
                part_id=None,
                scenes=[],
            )
        ],
    )

    nodes = normalize_server_structure_outline(structure)
    assert all(isinstance(node, WritingOutlineNode) for node in nodes)

    unassigned_bucket = next(node for node in nodes if node.kind == "unassigned_chapters")
    assert unassigned_bucket.project_id == "project-1"
    assert unassigned_bucket.parent_id is None

    unassigned_chapter_node = next(node for node in nodes if node.id == "chapter-u1")
    assert unassigned_chapter_node.kind == "chapter"
    assert unassigned_chapter_node.parent_id == unassigned_bucket.id

    assert not any(node.kind == "manuscript" and node.id == "chapter-u1" for node in nodes)
