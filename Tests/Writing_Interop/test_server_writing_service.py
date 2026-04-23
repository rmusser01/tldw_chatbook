import pytest

from tldw_chatbook.Writing_Interop.server_writing_service import (
    ServerWritingService,
    WritingCapabilityError,
)
from tldw_chatbook.Writing_Interop.writing_markdown_adapter import markdown_to_server_content
from tldw_chatbook.Writing_Interop.writing_models import (
    WritingChapter,
    WritingManuscript,
    WritingOutlineNode,
    WritingProject,
    WritingScene,
)


class FakeProjectList:
    def __init__(self, projects):
        self.projects = projects


class FakeClient:
    def __init__(self):
        self.calls = []
        self.project = {
            "id": "project-1",
            "title": "Server Project",
            "status": "draft",
            "version": 3,
        }
        self.part = {
            "id": "part-1",
            "project_id": "project-1",
            "title": "Part One",
            "sort_order": 1.0,
            "version": 4,
        }
        self.chapter = {
            "id": "chapter-1",
            "project_id": "project-1",
            "part_id": "part-1",
            "title": "Chapter One",
            "sort_order": 2.0,
            "status": "draft",
            "version": 5,
        }
        self.scene = {
            "id": "scene-1",
            "project_id": "project-1",
            "chapter_id": "chapter-1",
            "title": "Scene One",
            "sort_order": 3.0,
            "content": markdown_to_server_content("Opening prose"),
            "content_plain": "Opening prose",
            "status": "draft",
            "version": 6,
        }
        self.structure = {
            "project_id": "project-1",
            "parts": [
                {
                    "id": "part-1",
                    "title": "Part One",
                    "sort_order": 1.0,
                    "version": 4,
                    "chapters": [
                        {
                            "id": "chapter-1",
                            "title": "Chapter One",
                            "sort_order": 2.0,
                            "part_id": "part-1",
                            "status": "draft",
                            "version": 5,
                            "scenes": [
                                {
                                    "id": "scene-1",
                                    "title": "Scene One",
                                    "sort_order": 3.0,
                                    "status": "draft",
                                    "version": 6,
                                }
                            ],
                        }
                    ],
                }
            ],
            "unassigned_chapters": [
                {
                    "id": "chapter-loose",
                    "title": "Loose Chapter",
                    "sort_order": 9.0,
                    "part_id": None,
                    "status": "draft",
                    "version": 1,
                    "scenes": [
                        {
                            "id": "scene-loose",
                            "title": "Loose Scene",
                            "sort_order": 1.0,
                            "status": "draft",
                            "version": 1,
                        }
                    ],
                }
            ],
        }

    async def list_manuscript_projects(self, **kwargs):
        self.calls.append(("list_manuscript_projects", kwargs))
        return FakeProjectList([self.project])

    async def list_manuscript_parts(self, project_id):
        self.calls.append(("list_manuscript_parts", project_id))
        return [self.part]

    async def get_manuscript_project_structure(self, project_id):
        self.calls.append(("get_manuscript_project_structure", project_id))
        return self.structure

    async def create_manuscript_scene(self, chapter_id, request_data):
        self.calls.append(("create_manuscript_scene", chapter_id, request_data))
        data = request_data.model_dump(exclude_none=True, mode="json")
        return {
            **self.scene,
            "chapter_id": chapter_id,
            "title": data["title"],
            "content": data["content"],
            "content_plain": data["content_plain"],
        }

    async def update_manuscript_scene(self, scene_id, request_data, expected_version):
        self.calls.append(("update_manuscript_scene", scene_id, request_data, expected_version))
        data = request_data.model_dump(exclude_unset=True, mode="json")
        return {
            **self.scene,
            "id": scene_id,
            "title": data.get("title", self.scene["title"]),
            "content": data.get("content", self.scene["content"]),
            "content_plain": data.get("content_plain", self.scene["content_plain"]),
            "version": expected_version + 1,
        }

    async def update_manuscript_project(self, project_id, request_data, expected_version):
        self.calls.append(("update_manuscript_project", project_id, request_data, expected_version))
        data = request_data.model_dump(exclude_unset=True, mode="json")
        return {**self.project, "id": project_id, **data, "version": expected_version + 1}

    async def update_manuscript_part(self, part_id, request_data, expected_version):
        self.calls.append(("update_manuscript_part", part_id, request_data, expected_version))
        data = request_data.model_dump(exclude_unset=True, mode="json")
        return {**self.part, "id": part_id, "title": data["title"], "version": expected_version + 1}

    async def update_manuscript_chapter(self, chapter_id, request_data, expected_version):
        self.calls.append(("update_manuscript_chapter", chapter_id, request_data, expected_version))
        data = request_data.model_dump(exclude_unset=True, mode="json")
        return {
            **self.chapter,
            "id": chapter_id,
            "part_id": data.get("part_id", self.chapter["part_id"]),
            "version": expected_version + 1,
        }

    async def delete_manuscript_part(self, part_id, expected_version):
        self.calls.append(("delete_manuscript_part", part_id, expected_version))
        return {"id": part_id, "deleted": True}

    async def reorder_manuscript_entities(self, project_id, request_data):
        self.calls.append(("reorder_manuscript_entities", project_id, request_data))
        return {"project_id": project_id, "ok": True}


@pytest.fixture()
def service():
    return ServerWritingService(FakeClient())


@pytest.mark.asyncio
async def test_server_projects_normalize_as_server_source(service):
    projects = await service.list_projects(status="draft", limit=10, offset=5)

    assert projects == [
        WritingProject(
            source="server",
            id="project-1",
            title="Server Project",
            status="draft",
            version=3,
        )
    ]
    assert service.client.calls[-1] == (
        "list_manuscript_projects",
        {"status": "draft", "limit": 10, "offset": 5},
    )


@pytest.mark.asyncio
async def test_server_parts_normalize_as_manuscripts(service):
    manuscripts = await service.list_manuscripts("project-1")

    assert manuscripts == [
        WritingManuscript(
            source="server",
            id="part-1",
            project_id="project-1",
            title="Part One",
            sort_order=1.0,
            version=4,
        )
    ]


@pytest.mark.asyncio
async def test_server_structure_maps_parts_and_unassigned_chapters(service):
    structure = await service.get_project_structure("project-1")

    assert structure["manuscripts"][0]["manuscript"].id == "part-1"
    assert structure["manuscripts"][0]["chapters"][0]["chapter"] == WritingChapter(
        source="server",
        id="chapter-1",
        project_id="project-1",
        manuscript_id="part-1",
        title="Chapter One",
        status="draft",
        sort_order=2.0,
        version=5,
    )
    assert structure["manuscripts"][0]["chapters"][0]["scenes"][0] == WritingScene(
        source="server",
        id="scene-1",
        project_id="project-1",
        chapter_id="chapter-1",
        title="Scene One",
        status="draft",
        sort_order=3.0,
        version=6,
    )
    assert structure["unassigned_chapters"][0]["chapter"].id == "chapter-loose"


@pytest.mark.asyncio
async def test_server_unassigned_chapters_are_preserved_under_explicit_bucket(service):
    outline = await service.get_outline("project-1")

    assert all(isinstance(node, WritingOutlineNode) for node in outline)
    bucket = next(node for node in outline if node.kind == "unassigned_chapters")
    loose_chapter = next(node for node in outline if node.id == "chapter-loose")
    loose_scene = next(node for node in outline if node.id == "scene-loose")

    assert bucket.title == "Unassigned Chapters"
    assert bucket.parent_id is None
    assert loose_chapter.parent_id == bucket.id
    assert loose_scene.parent_id == "chapter-loose"
    assert not any(node.kind == "manuscript" and node.id == "chapter-loose" for node in outline)


@pytest.mark.asyncio
async def test_creating_server_scene_requires_chapter_id(service):
    with pytest.raises(ValueError, match="chapter_id is required"):
        await service.create_scene("project-1", title="Scene", body_markdown="Body")


@pytest.mark.asyncio
async def test_direct_manuscript_level_server_scene_creation_raises_capability_error(service):
    with pytest.raises(WritingCapabilityError) as exc_info:
        await service.create_scene(
            "project-1",
            title="Direct Scene",
            manuscript_id="part-1",
            chapter_id=None,
            body_markdown="Body",
        )

    assert exc_info.value.capability == "server_direct_manuscript_scene"
    assert exc_info.value.source == "server"
    assert exc_info.value.reason == "server_direct_manuscript_scene_unavailable"


@pytest.mark.asyncio
async def test_server_scene_update_sends_markdown_wrapper_content_and_plain_text(service):
    scene = await service.update_scene(
        "scene-1",
        {"title": "Updated Scene", "body_markdown": "# Heading\n\n**Body** text"},
        expected_version=6,
    )

    _, scene_id, request_data, expected_version = service.client.calls[-1]
    payload = request_data.model_dump(exclude_unset=True, mode="json")

    assert scene_id == "scene-1"
    assert expected_version == 6
    assert payload["content"] == markdown_to_server_content("# Heading\n\n**Body** text")
    assert payload["content_plain"] == "Heading\nBody text"
    assert scene.body_markdown == "# Heading\n\n**Body** text"


@pytest.mark.asyncio
async def test_server_update_and_delete_pass_expected_version_to_client_methods(service):
    manuscript = await service.update_manuscript(
        "part-1",
        {"title": "Updated Part"},
        expected_version=4,
    )
    deleted = await service.delete_manuscript("part-1", expected_version=5)

    assert manuscript.title == "Updated Part"
    assert deleted == {"id": "part-1", "deleted": True}
    assert service.client.calls[-2][0:4] == (
        "update_manuscript_part",
        "part-1",
        service.client.calls[-2][2],
        4,
    )
    assert service.client.calls[-1] == ("delete_manuscript_part", "part-1", 5)


@pytest.mark.asyncio
async def test_server_keyword_updates_preserve_explicit_none_clears(service):
    project = await service.update_project(
        "project-1",
        expected_version=3,
        subtitle=None,
    )
    chapter = await service.update_chapter(
        "chapter-1",
        expected_version=5,
        manuscript_id=None,
    )

    project_payload = service.client.calls[-2][2].model_dump(exclude_unset=True, mode="json")
    chapter_payload = service.client.calls[-1][2].model_dump(exclude_unset=True, mode="json")

    assert project.subtitle is None
    assert chapter.manuscript_id is None
    assert project_payload == {"subtitle": None}
    assert chapter_payload == {"part_id": None}
    assert service.client.calls[-2][3] == 3
    assert service.client.calls[-1][3] == 5


@pytest.mark.asyncio
async def test_unsupported_server_capabilities_use_scope_gate_reason_codes(service):
    with pytest.raises(WritingCapabilityError) as direct_scene:
        await service.create_scene(
            "project-1",
            title="Direct Scene",
            manuscript_id="part-1",
            body_markdown="Body",
        )
    with pytest.raises(WritingCapabilityError) as reparent:
        await service.update_scene("scene-1", {"chapter_id": "chapter-2"}, expected_version=6)
    with pytest.raises(WritingCapabilityError) as version:
        await service.create_version("scene", "scene-1")
    with pytest.raises(WritingCapabilityError) as trash:
        await service.list_trash("project-1")

    assert direct_scene.value.reason == "server_direct_manuscript_scene_unavailable"
    assert reparent.value.reason == "server_scene_reparent_unavailable"
    assert version.value.reason == "server_version_history_unavailable"
    assert trash.value.reason == "server_trash_restore_unavailable"


@pytest.mark.asyncio
async def test_server_reorder_allows_chapter_parent_updates_and_scene_order_only(service):
    chapter_result = await service.reorder_items(
        "project-1",
        "chapters",
        [{"id": "chapter-1", "sort_order": 1, "version": 5, "new_parent_id": "part-2"}],
    )
    scene_result = await service.reorder_items(
        "project-1",
        "scenes",
        [{"id": "scene-1", "sort_order": 1, "version": 6}],
    )

    assert chapter_result == {"project_id": "project-1", "ok": True}
    assert scene_result == {"project_id": "project-1", "ok": True}
    assert service.client.calls[-2][0] == "reorder_manuscript_entities"
    assert service.client.calls[-1][0] == "reorder_manuscript_entities"


@pytest.mark.asyncio
async def test_server_reorder_blocks_scene_reparent_before_client_call(service):
    with pytest.raises(WritingCapabilityError) as exc_info:
        await service.reorder_items(
            "project-1",
            "scenes",
            [{"id": "scene-1", "sort_order": 1, "version": 6, "new_parent_id": "chapter-2"}],
        )

    assert exc_info.value.reason == "server_scene_reparent_unavailable"
    assert not any(call[0] == "reorder_manuscript_entities" for call in service.client.calls)
