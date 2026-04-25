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
