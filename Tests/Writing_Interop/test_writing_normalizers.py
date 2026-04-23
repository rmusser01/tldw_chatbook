from datetime import datetime, timezone

from tldw_chatbook.Writing_Interop.writing_models import WritingOutlineNode
from tldw_chatbook.Writing_Interop.writing_normalizers import (
    normalize_local_chapter_row,
    normalize_local_manuscript_row,
    normalize_local_project_row,
    normalize_local_scene_row,
    normalize_server_part,
    normalize_server_project,
    normalize_server_scene,
    normalize_server_structure_outline,
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
    assert scene.body_markdown == "Hello"


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
