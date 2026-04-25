import pytest

from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService


def test_local_writing_service_persists_project_hierarchy_and_unassigned_chapters(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")

    project = service.create_project(title="Novel", author="Ada", genre="sci-fi")
    manuscript = service.create_manuscript(project["id"], title="Book One", synopsis="Opening arc")
    assigned_chapter = service.create_chapter(
        project["id"],
        title="Chapter 1",
        manuscript_id=manuscript["id"],
        synopsis="Arrival",
    )
    unassigned_chapter = service.create_chapter(project["id"], title="Loose Chapter")
    scene = service.create_scene(
        assigned_chapter["id"],
        title="Scene 1",
        content_markdown="Opening line.",
        synopsis="Meet Ada",
    )

    structure = service.get_structure(project["id"])

    assert project["source"] == "local"
    assert manuscript["project_id"] == project["id"]
    assert assigned_chapter["manuscript_id"] == manuscript["id"]
    assert unassigned_chapter["manuscript_id"] is None
    assert scene["content_markdown"] == "Opening line."
    assert structure["manuscripts"][0]["id"] == manuscript["id"]
    assert structure["manuscripts"][0]["chapters"][0]["id"] == assigned_chapter["id"]
    assert structure["manuscripts"][0]["chapters"][0]["scenes"][0]["id"] == scene["id"]
    assert structure["unassigned_chapters"][0]["id"] == unassigned_chapter["id"]


def test_local_writing_service_updates_versions_and_soft_deletes(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(project["id"], title="Chapter 1", manuscript_id=manuscript["id"])
    scene = service.create_scene(chapter["id"], title="Scene 1", content_markdown="Draft")

    updated_scene = service.update_scene(
        scene["id"],
        expected_version=1,
        title="Scene 1 revised",
        content_markdown="Revised draft",
    )
    deleted = service.delete_scene(scene["id"], expected_version=2)

    assert updated_scene["version"] == 2
    assert updated_scene["content_markdown"] == "Revised draft"
    assert deleted is True
    assert service.get_scene(scene["id"]) is None
    assert service.list_scenes(chapter["id"]) == []


def test_local_writing_service_updates_and_deletes_manuscripts_and_chapters(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(project["id"], title="Chapter 1", manuscript_id=manuscript["id"])

    updated_manuscript = service.update_manuscript(
        manuscript["id"],
        expected_version=1,
        title="Book One Revised",
    )
    updated_chapter = service.update_chapter(
        chapter["id"],
        expected_version=1,
        title="Chapter 1 Revised",
        manuscript_id=None,
    )
    deleted_chapter = service.delete_chapter(chapter["id"], expected_version=2)
    deleted_manuscript = service.delete_manuscript(manuscript["id"], expected_version=2)

    assert updated_manuscript["version"] == 2
    assert updated_manuscript["title"] == "Book One Revised"
    assert updated_chapter["version"] == 2
    assert updated_chapter["manuscript_id"] is None
    assert deleted_chapter is True
    assert deleted_manuscript is True
    assert service.get_chapter(chapter["id"]) is None
    assert service.get_manuscript(manuscript["id"]) is None


def test_local_writing_service_preserves_chapter_manuscript_when_not_explicitly_changed(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(project["id"], title="Chapter 1", manuscript_id=manuscript["id"])

    updated_chapter = service.update_chapter(
        chapter["id"],
        expected_version=1,
        title="Chapter 1 Revised",
    )

    assert updated_chapter["manuscript_id"] == manuscript["id"]


def test_local_writing_service_rejects_stale_expected_versions(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")

    with pytest.raises(ValueError, match="version conflict"):
        service.update_project(project["id"], expected_version=2, title="Stale")


def test_local_writing_service_creates_manual_scene_versions_and_restores_them(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(project["id"], title="Chapter 1", manuscript_id=manuscript["id"])
    scene = service.create_scene(chapter["id"], title="Scene 1", content_markdown="# Draft\n\nOpening.")

    version = service.create_version("scene", scene["id"], label="First draft")
    service.update_scene(scene["id"], expected_version=1, title="Scene 1 revised", content_markdown="Changed.")
    restored = service.restore_version("scene", scene["id"], version["version_number"], expected_version=2)
    versions = service.list_versions("scene", scene["id"])

    assert version["source"] == "local"
    assert version["record_id"].startswith("local:writing_version:")
    assert version["entity_type"] == "scene"
    assert version["entity_id"] == scene["id"]
    assert version["version_number"] == 1
    assert version["label"] == "First draft"
    assert version["payload"]["content_markdown"] == "# Draft\n\nOpening."
    assert restored["title"] == "Scene 1"
    assert restored["content_markdown"] == "# Draft\n\nOpening."
    assert restored["version"] == 3
    assert versions[0]["version_number"] == 1
    assert versions[0]["payload"]["content_markdown"] == "# Draft\n\nOpening."


def test_local_writing_service_container_versions_snapshot_structure_not_body_drafts(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(project["id"], title="Chapter 1", manuscript_id=manuscript["id"])
    scene = service.create_scene(chapter["id"], title="Scene 1", content_markdown="Opening line.")

    manuscript_version = service.create_version("manuscript", manuscript["id"])
    chapter_version = service.create_version("chapter", chapter["id"])

    assert manuscript_version["payload"]["chapter_ids"] == [chapter["id"]]
    assert manuscript_version["payload"]["rendered_markdown"] == "Opening line."
    assert "content_markdown" not in manuscript_version["payload"]
    assert chapter_version["payload"]["scene_ids"] == [scene["id"]]
    assert chapter_version["payload"]["rendered_markdown"] == "Opening line."
    assert "content_markdown" not in chapter_version["payload"]


def test_local_writing_service_lists_and_restores_soft_deleted_records(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(project["id"], title="Chapter 1", manuscript_id=manuscript["id"])
    scene = service.create_scene(chapter["id"], title="Scene 1", content_markdown="Draft")

    assert service.delete_scene(scene["id"], expected_version=1) is True
    trash = service.list_trash(entity_type="scene")
    restored = service.restore_trash("scene", scene["id"], expected_version=2)

    assert trash[0]["record_id"] == f"local:writing_scene:{scene['id']}"
    assert trash[0]["deleted"] == 1
    assert restored["id"] == scene["id"]
    assert restored["deleted"] == 0
    assert restored["version"] == 3
    assert service.list_trash(entity_type="scene") == []
    assert service.get_scene(scene["id"])["content_markdown"] == "Draft"


def test_local_writing_service_reorders_and_moves_chapters_and_scenes(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript_a = service.create_manuscript(project["id"], title="Book One")
    manuscript_b = service.create_manuscript(project["id"], title="Book Two")
    chapter_a = service.create_chapter(project["id"], title="Chapter 1", manuscript_id=manuscript_a["id"])
    chapter_b = service.create_chapter(project["id"], title="Chapter 2", manuscript_id=manuscript_b["id"])
    scene = service.create_scene(chapter_a["id"], title="Scene 1", content_markdown="Draft")

    assert service.reorder_entities(
        project["id"],
        "chapters",
        [{"id": chapter_a["id"], "sort_order": 10.0, "version": 1, "new_parent_id": manuscript_b["id"]}],
    ) is True
    assert service.reorder_entities(
        project["id"],
        "scenes",
        [{"id": scene["id"], "sort_order": 5.0, "version": 1, "new_parent_id": chapter_b["id"]}],
    ) is True

    moved_chapter = service.get_chapter(chapter_a["id"])
    moved_scene = service.get_scene(scene["id"])

    assert moved_chapter["manuscript_id"] == manuscript_b["id"]
    assert moved_chapter["sort_order"] == 10.0
    assert moved_chapter["version"] == 2
    assert moved_scene["chapter_id"] == chapter_b["id"]
    assert moved_scene["sort_order"] == 5.0
    assert moved_scene["version"] == 2


def test_local_writing_service_supports_direct_manuscript_level_scenes(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")

    scene = service.create_scene(
        None,
        manuscript_id=manuscript["id"],
        title="Prologue",
        content_markdown="Direct scene.",
    )
    manuscript_scenes = service.list_scenes(manuscript_id=manuscript["id"])
    structure = service.get_structure(project["id"])

    assert scene["chapter_id"] is None
    assert scene["manuscript_id"] == manuscript["id"]
    assert manuscript_scenes[0]["id"] == scene["id"]
    assert structure["manuscripts"][0]["scenes"][0]["id"] == scene["id"]
