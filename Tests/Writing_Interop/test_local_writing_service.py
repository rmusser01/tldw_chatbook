import pytest

from tldw_chatbook.DB.Writing_DB import WritingDatabase
from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService
from tldw_chatbook.Writing_Interop.writing_models import (
    WritingChapter,
    WritingManuscript,
    WritingOutlineNode,
    WritingProject,
    WritingScene,
    WritingTrashEntry,
    WritingVersion,
)


@pytest.fixture()
def db(tmp_path):
    database = WritingDatabase(tmp_path / "writing.db", client_id="test_client")
    yield database
    database.close()


@pytest.fixture()
def service(db):
    return LocalWritingService(db)


@pytest.mark.asyncio
async def test_unavailable_local_backend_fails_visibly():
    service = LocalWritingService(None)

    with pytest.raises(ValueError, match="Local writing backend is unavailable"):
        await service.list_projects()


@pytest.mark.asyncio
async def test_project_manuscript_chapter_scene_crud_returns_normalized_dataclasses(service):
    project = await service.create_project(
        title="First Novel",
        subtitle="A draft",
        author="Ada",
        genre="Speculative",
    )
    manuscript = await service.create_manuscript(
        project.id,
        title="Book One",
        sort_order=2,
    )
    chapter = await service.create_chapter(
        project.id,
        title="Chapter One",
        manuscript_id=manuscript.id,
        sort_order=1,
    )
    scene = await service.create_scene(
        project.id,
        title="Opening Scene",
        chapter_id=chapter.id,
        body_markdown="# Opening",
    )

    assert isinstance(project, WritingProject)
    assert isinstance(manuscript, WritingManuscript)
    assert isinstance(chapter, WritingChapter)
    assert isinstance(scene, WritingScene)
    assert project.source == manuscript.source == chapter.source == scene.source == "local"
    assert manuscript.project_id == project.id
    assert chapter.manuscript_id == manuscript.id
    assert scene.chapter_id == chapter.id
    assert scene.body_markdown == "# Opening"

    listed_projects = await service.list_projects()
    listed_manuscripts = await service.list_manuscripts(project.id)
    listed_chapters = await service.list_chapters(project.id, manuscript_id=manuscript.id)
    listed_scenes = await service.list_scenes(project.id, chapter_id=chapter.id)

    assert listed_projects == [project]
    assert listed_manuscripts == [manuscript]
    assert listed_chapters == [chapter]
    assert listed_scenes == [scene]

    updated_project = await service.update_project(
        project.id,
        {"title": "Second Novel"},
        expected_version=1,
    )
    updated_manuscript = await service.update_manuscript(
        manuscript.id,
        {"title": "Book One Revised"},
        expected_version=1,
    )
    updated_chapter = await service.update_chapter(
        chapter.id,
        {"title": "Chapter One Revised"},
        expected_version=1,
    )
    updated_scene = await service.update_scene(
        scene.id,
        {"title": "Opening Revised"},
        expected_version=1,
    )

    assert updated_project.title == "Second Novel"
    assert updated_manuscript.title == "Book One Revised"
    assert updated_chapter.title == "Chapter One Revised"
    assert updated_scene.title == "Opening Revised"
    assert updated_project.version == 2
    assert updated_manuscript.version == 2
    assert updated_chapter.version == 2
    assert updated_scene.version == 2

    assert await service.get_project(project.id) == updated_project
    assert await service.get_manuscript(manuscript.id) == updated_manuscript
    assert await service.get_chapter(chapter.id) == updated_chapter
    assert await service.get_scene(scene.id) == updated_scene

    deleted_scene = await service.delete_scene(scene.id, expected_version=2)
    restored_scene = await service.restore_scene(scene.id, expected_version=3)
    deleted_chapter = await service.delete_chapter(chapter.id, expected_version=2)
    restored_chapter = await service.restore_chapter(chapter.id, expected_version=3)
    deleted_manuscript = await service.delete_manuscript(manuscript.id, expected_version=2)
    restored_manuscript = await service.restore_manuscript(manuscript.id, expected_version=3)
    deleted_project = await service.delete_project(project.id, expected_version=2)
    restored_project = await service.restore_project(project.id, expected_version=3)

    assert deleted_scene.deleted is True
    assert restored_scene.deleted is False
    assert deleted_chapter.deleted is True
    assert restored_chapter.deleted is False
    assert deleted_manuscript.deleted is True
    assert restored_manuscript.deleted is False
    assert deleted_project.deleted is True
    assert restored_project.deleted is False


@pytest.mark.asyncio
async def test_unassigned_chapter_appears_in_outline(service):
    project = await service.create_project(title="Project")
    manuscript = await service.create_manuscript(project.id, title="Book")
    assigned = await service.create_chapter(
        project.id,
        manuscript_id=manuscript.id,
        title="Assigned Chapter",
    )
    unassigned = await service.create_chapter(
        project.id,
        manuscript_id=None,
        title="Loose Chapter",
    )

    outline = await service.get_outline(project.id)

    assert all(isinstance(node, WritingOutlineNode) for node in outline)
    bucket = next(node for node in outline if node.kind == "unassigned_chapters")
    assert bucket.title == "Unassigned Chapters"
    assert bucket.parent_id is None

    assigned_node = next(node for node in outline if node.id == assigned.id)
    unassigned_node = next(node for node in outline if node.id == unassigned.id)
    assert assigned_node.parent_id == manuscript.id
    assert unassigned_node.parent_id == bucket.id


@pytest.mark.asyncio
async def test_direct_manuscript_level_scene_works_locally(service):
    project = await service.create_project(title="Project")
    manuscript = await service.create_manuscript(project.id, title="Book")

    scene = await service.create_scene(
        project.id,
        title="Direct Scene",
        manuscript_id=manuscript.id,
        chapter_id=None,
        body_markdown="Direct prose",
    )

    assert scene.chapter_id is None
    assert scene.manuscript_id == manuscript.id

    listed = await service.list_scenes(
        project.id,
        manuscript_id=manuscript.id,
        chapter_id=None,
    )
    assert listed == [scene]

    outline = await service.get_outline(project.id)
    scene_node = next(node for node in outline if node.id == scene.id)
    assert scene_node.kind == "scene"
    assert scene_node.parent_id == manuscript.id


@pytest.mark.asyncio
async def test_scene_under_unassigned_chapter_works_locally(service):
    project = await service.create_project(title="Project")
    chapter = await service.create_chapter(
        project.id,
        manuscript_id=None,
        title="Loose Chapter",
    )

    scene = await service.create_scene(
        project.id,
        title="Loose Chapter Scene",
        chapter_id=chapter.id,
        body_markdown="Loose prose",
    )

    assert scene.chapter_id == chapter.id
    assert scene.manuscript_id is None

    listed = await service.list_scenes(project.id, chapter_id=chapter.id)
    assert listed == [scene]

    outline = await service.get_outline(project.id)
    scene_node = next(node for node in outline if node.id == scene.id)
    assert scene_node.parent_id == chapter.id


@pytest.mark.asyncio
async def test_autosave_manual_versions_and_restore_version_semantics(service):
    project = await service.create_project(title="Project")
    manuscript = await service.create_manuscript(project.id, title="Book")
    scene = await service.create_scene(
        project.id,
        title="Scene",
        manuscript_id=manuscript.id,
        body_markdown="Draft A",
    )

    autosaved = await service.autosave_scene(
        scene.id,
        body_markdown="Draft B",
        expected_version=1,
    )
    assert autosaved.body_markdown == "Draft B"
    assert autosaved.version == 2

    version_one = await service.create_version("scene", scene.id)
    assert isinstance(version_one, WritingVersion)
    assert version_one.version_number == 1
    assert version_one.body_markdown == "Draft B"
    assert (await service.get_scene(scene.id)).version == 2

    changed = await service.autosave_scene(
        scene.id,
        body_markdown="Draft C",
        expected_version=2,
    )
    version_two = await service.create_version("scene", scene.id)
    restored = await service.restore_version_to_working_state(
        version_one.id,
        expected_version=changed.version,
    )
    versions = await service.list_versions("scene", scene.id)

    assert version_two.version_number == 2
    assert restored.body_markdown == "Draft B"
    assert restored.version == changed.version + 1
    assert [version.version_number for version in versions] == [1, 2]


@pytest.mark.asyncio
async def test_container_versions_snapshot_child_membership_without_body_content(service):
    project = await service.create_project(title="Project")
    manuscript = await service.create_manuscript(project.id, title="Book")
    chapter = await service.create_chapter(
        project.id,
        manuscript_id=manuscript.id,
        title="Chapter",
    )
    scene = await service.create_scene(
        project.id,
        chapter_id=chapter.id,
        title="Scene",
        body_markdown="Scene body",
    )

    manuscript_version = await service.create_version("manuscript", manuscript.id)
    chapter_version = await service.create_version("chapter", chapter.id)

    assert manuscript_version.body_markdown is None
    assert chapter_version.body_markdown is None
    assert manuscript_version.metadata["child_chapter_ids"] == [chapter.id]
    assert manuscript_version.metadata["direct_scene_ids"] == []
    assert chapter_version.metadata["child_scene_ids"] == [scene.id]
    assert chapter_version.metadata["assembled_markdown"] == "Scene body"
    assert "Scene body" in manuscript_version.metadata["assembled_markdown"]


@pytest.mark.asyncio
async def test_trash_listing_and_restore_work_for_all_entity_kinds(service):
    project = await service.create_project(title="Trash Project")
    manuscript = await service.create_manuscript(project.id, title="Trash Book")
    chapter = await service.create_chapter(
        project.id,
        manuscript_id=manuscript.id,
        title="Trash Chapter",
    )
    scene = await service.create_scene(
        project.id,
        chapter_id=chapter.id,
        title="Trash Scene",
    )

    await service.delete_scene(scene.id, expected_version=1)
    await service.delete_chapter(chapter.id, expected_version=1)
    await service.delete_manuscript(manuscript.id, expected_version=1)
    await service.delete_project(project.id, expected_version=1)

    trash = await service.list_trash(project.id)
    assert all(isinstance(item, WritingTrashEntry) for item in trash)
    assert {(item.entity_kind, item.entity_id) for item in trash} == {
        ("project", project.id),
        ("manuscript", manuscript.id),
        ("chapter", chapter.id),
        ("scene", scene.id),
    }

    await service.restore_project(project.id)
    await service.restore_manuscript(manuscript.id)
    await service.restore_chapter(chapter.id)
    await service.restore_scene(scene.id)

    assert await service.list_trash(project.id) == []
    assert (await service.get_project(project.id)).deleted is False
    assert (await service.get_manuscript(manuscript.id)).deleted is False
    assert (await service.get_chapter(chapter.id)).deleted is False
    assert (await service.get_scene(scene.id)).deleted is False


@pytest.mark.asyncio
async def test_search_project_finds_local_scene_content(service):
    project = await service.create_project(title="Project")
    manuscript = await service.create_manuscript(project.id, title="Book")
    chapter = await service.create_chapter(
        project.id,
        manuscript_id=manuscript.id,
        title="Chapter",
    )
    scene = await service.create_scene(
        project.id,
        chapter_id=chapter.id,
        title="Opening Scene",
        body_markdown="A candle flickered in the observatory.",
        word_count=7,
    )

    results = await service.search_project(project.id, "flickered", limit=5)

    assert results == [
        {
            "source": "local",
            "entity_kind": "scene",
            "id": scene.id,
            "title": "Opening Scene",
            "chapter_id": chapter.id,
            "manuscript_id": None,
            "word_count": 7,
            "status": "draft",
            "snippet": "A candle flickered in the observatory.",
        }
    ]


@pytest.mark.asyncio
async def test_reorder_items_uses_local_database_with_server_shaped_items(service):
    project = await service.create_project(title="Project")
    first = await service.create_manuscript(project.id, title="First", sort_order=1)
    second = await service.create_manuscript(project.id, title="Second", sort_order=2)

    reordered = await service.reorder_items(
        project.id,
        "parts",
        [
            {"id": second.id, "sort_order": 10, "version": second.version},
            {"id": first.id, "sort_order": 20, "version": first.version},
        ],
    )

    assert [item.id for item in reordered] == [second.id, first.id]
    assert [item.sort_order for item in reordered] == [10.0, 20.0]
    assert [item.version for item in reordered] == [2, 2]


@pytest.mark.asyncio
async def test_local_chapter_scene_reorder_and_scene_moves(service):
    project = await service.create_project(title="Project")
    manuscript = await service.create_manuscript(project.id, title="Book")
    first_chapter = await service.create_chapter(
        project.id,
        manuscript_id=manuscript.id,
        title="First",
        sort_order=1,
    )
    second_chapter = await service.create_chapter(
        project.id,
        manuscript_id=manuscript.id,
        title="Second",
        sort_order=2,
    )
    first_scene = await service.create_scene(
        project.id,
        chapter_id=first_chapter.id,
        title="First Scene",
        sort_order=1,
    )
    second_scene = await service.create_scene(
        project.id,
        chapter_id=first_chapter.id,
        title="Second Scene",
        sort_order=2,
    )

    reordered_chapters = await service.reorder_items(
        project.id,
        "chapters",
        [
            {"id": second_chapter.id, "sort_order": 10, "version": second_chapter.version},
            {"id": first_chapter.id, "sort_order": 20, "version": first_chapter.version},
        ],
    )
    reordered_scenes = await service.reorder_items(
        project.id,
        "scenes",
        [
            {"id": second_scene.id, "sort_order": 10, "version": second_scene.version},
            {"id": first_scene.id, "sort_order": 20, "version": first_scene.version},
        ],
    )
    moved_to_manuscript = await service.move_scene(
        first_scene.id,
        manuscript.id,
        None,
        expected_version=reordered_scenes[1].version,
    )
    moved_to_chapter = await service.move_scene(
        first_scene.id,
        None,
        second_chapter.id,
        expected_version=moved_to_manuscript.version,
    )
    moved_between_chapters = await service.move_scene(
        first_scene.id,
        None,
        first_chapter.id,
        expected_version=moved_to_chapter.version,
    )

    assert [chapter.id for chapter in reordered_chapters] == [second_chapter.id, first_chapter.id]
    assert [scene.id for scene in reordered_scenes] == [second_scene.id, first_scene.id]
    assert moved_to_manuscript.manuscript_id == manuscript.id
    assert moved_to_manuscript.chapter_id is None
    assert moved_to_chapter.chapter_id == second_chapter.id
    assert moved_to_chapter.manuscript_id is None
    assert moved_between_chapters.chapter_id == first_chapter.id
