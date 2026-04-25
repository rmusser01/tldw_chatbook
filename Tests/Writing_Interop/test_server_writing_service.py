import pytest

from tldw_chatbook.Writing_Interop.server_writing_service import ServerWritingService


class FakeWritingClient:
    def __init__(self):
        self.calls = []

    async def create_manuscript_project(self, request_data):
        self.calls.append(("create_manuscript_project", request_data.model_dump(mode="json", exclude_none=True)))
        return {"id": "project-1", "title": request_data.title, "version": 1}

    async def create_manuscript(self, project_id, request_data):
        self.calls.append(("create_manuscript", project_id, request_data.model_dump(mode="json", exclude_none=True)))
        return {"id": "manuscript-1", "project_id": project_id, "title": request_data.title, "version": 1}

    async def create_manuscript_chapter(self, project_id, request_data):
        self.calls.append(("create_manuscript_chapter", project_id, request_data.model_dump(mode="json", exclude_none=True)))
        return {
            "id": "chapter-1",
            "project_id": project_id,
            "part_id": request_data.part_id,
            "title": request_data.title,
            "version": 1,
        }

    async def create_manuscript_scene(self, chapter_id, request_data):
        self.calls.append(("create_manuscript_scene", chapter_id, request_data.model_dump(mode="json", exclude_none=True)))
        return {
            "id": "scene-1",
            "chapter_id": chapter_id,
            "project_id": "project-1",
            "title": request_data.title,
            "content_plain": request_data.content_plain,
            "version": 1,
        }

    async def get_manuscript_structure(self, project_id):
        self.calls.append(("get_manuscript_structure", project_id))
        return {
            "project_id": project_id,
            "parts": [
                {
                    "id": "manuscript-1",
                    "title": "Book One",
                    "sort_order": 0.0,
                    "chapters": [],
                }
            ],
            "unassigned_chapters": [],
        }


@pytest.mark.asyncio
async def test_server_writing_service_maps_chatbook_manuscripts_to_server_parts():
    client = FakeWritingClient()
    service = ServerWritingService(client=client)

    project = await service.create_project(title="Novel", author="Ada")
    manuscript = await service.create_manuscript(project["id"], title="Book One")
    chapter = await service.create_chapter(project["id"], title="Chapter 1", manuscript_id=manuscript["id"])
    scene = await service.create_scene(chapter["id"], title="Scene 1", content_markdown="Opening line.")
    structure = await service.get_structure(project["id"])

    assert project["record_id"] == "server:writing_project:project-1"
    assert manuscript["record_id"] == "server:writing_manuscript:manuscript-1"
    assert chapter["manuscript_id"] == "manuscript-1"
    assert scene["content_markdown"] == "Opening line."
    assert structure["manuscripts"][0]["record_id"] == "server:writing_manuscript:manuscript-1"
    assert client.calls == [
        ("create_manuscript_project", {"title": "Novel", "author": "Ada", "status": "draft", "settings": {}}),
        ("create_manuscript", "project-1", {"title": "Book One", "sort_order": 0.0}),
        (
            "create_manuscript_chapter",
            "project-1",
            {"title": "Chapter 1", "part_id": "manuscript-1", "sort_order": 0.0, "status": "draft"},
        ),
        (
            "create_manuscript_scene",
            "chapter-1",
            {"title": "Scene 1", "content_plain": "Opening line.", "sort_order": 0.0, "status": "draft"},
        ),
        ("get_manuscript_structure", "project-1"),
    ]
