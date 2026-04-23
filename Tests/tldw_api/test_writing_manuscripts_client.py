"""Tests for writing manuscript project endpoint wiring on the shared TLDW API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ManuscriptChapterCreateRequest,
    ManuscriptChapterResponse,
    ManuscriptChapterUpdateRequest,
    ManuscriptPartCreateRequest,
    ManuscriptPartResponse,
    ManuscriptPartUpdateRequest,
    ManuscriptProjectCreateRequest,
    ManuscriptProjectListResponse,
    ManuscriptProjectResponse,
    ManuscriptProjectUpdateRequest,
    ManuscriptSceneCreateRequest,
    ManuscriptSceneResponse,
    ManuscriptSceneUpdateRequest,
    ManuscriptSearchResponse,
    ManuscriptStructureResponse,
    ReorderItem,
    ReorderRequest,
    TLDWAPIClient,
)


def _project_payload(*, project_id: str = "proj-1", version: int = 1) -> dict:
    return {
        "id": project_id,
        "title": "Novel Draft",
        "subtitle": "A Beginning",
        "author": "Jane Doe",
        "genre": "Fantasy",
        "status": "draft",
        "synopsis": "An epic journey",
        "target_word_count": 90000,
        "settings": {"tone": "dark"},
        "word_count": 0,
        "created_at": "2026-04-22T00:00:00Z",
        "last_modified": "2026-04-22T00:00:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": version,
    }


def _part_payload(*, part_id: str = "part-1", project_id: str = "proj-1", version: int = 1) -> dict:
    return {
        "id": part_id,
        "project_id": project_id,
        "title": "Part One",
        "sort_order": 1.0,
        "synopsis": "Opening act",
        "word_count": 0,
        "created_at": "2026-04-22T00:00:00Z",
        "last_modified": "2026-04-22T00:00:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": version,
    }


def _chapter_payload(*, chapter_id: str = "chapter-1", project_id: str = "proj-1", version: int = 1) -> dict:
    return {
        "id": chapter_id,
        "project_id": project_id,
        "part_id": "part-1",
        "title": "Chapter One",
        "sort_order": 1.0,
        "synopsis": "Arrival",
        "pov_character_id": None,
        "word_count": 0,
        "status": "draft",
        "created_at": "2026-04-22T00:00:00Z",
        "last_modified": "2026-04-22T00:00:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": version,
    }


def _scene_payload(*, scene_id: str = "scene-1", chapter_id: str = "chapter-1", version: int = 1) -> dict:
    return {
        "id": scene_id,
        "chapter_id": chapter_id,
        "project_id": "proj-1",
        "title": "Scene One",
        "sort_order": 1.0,
        "content_json": None,
        "content_plain": "Opening line.",
        "synopsis": "A first beat",
        "word_count": 2,
        "pov_character_id": None,
        "status": "draft",
        "created_at": "2026-04-22T00:00:00Z",
        "last_modified": "2026-04-22T00:00:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": version,
    }


@pytest.mark.asyncio
async def test_list_manuscript_projects_uses_server_prefix():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(return_value={"projects": [], "total": 0})

    response = await client.list_manuscript_projects(status="draft", limit=25, offset=5)

    assert isinstance(response, ManuscriptProjectListResponse)
    assert response.total == 0
    client._request.assert_awaited_once_with(
        "GET",
        "/api/v1/writing/manuscripts/projects",
        params={"status": "draft", "limit": 25, "offset": 5},
    )


@pytest.mark.asyncio
async def test_create_manuscript_project_wires_payload_and_returns_typed_response():
    client = TLDWAPIClient("http://example.test", "token")
    payload = _project_payload()
    client._request = AsyncMock(return_value=payload)

    response = await client.create_manuscript_project(
        ManuscriptProjectCreateRequest(
            title="Novel Draft",
            subtitle="A Beginning",
            author="Jane Doe",
            genre="Fantasy",
            status="draft",
            synopsis="An epic journey",
            target_word_count=90000,
            settings={"tone": "dark"},
        )
    )

    assert isinstance(response, ManuscriptProjectResponse)
    client._request.assert_awaited_once_with(
        "POST",
        "/api/v1/writing/manuscripts/projects",
        json_data={
            "title": "Novel Draft",
            "subtitle": "A Beginning",
            "author": "Jane Doe",
            "genre": "Fantasy",
            "status": "draft",
            "synopsis": "An epic journey",
            "target_word_count": 90000,
            "settings": {"tone": "dark"},
        },
    )


@pytest.mark.asyncio
async def test_get_manuscript_project_returns_typed_response():
    client = TLDWAPIClient("http://example.test", "token")
    payload = _project_payload(project_id="proj-99", version=2)
    client._request = AsyncMock(return_value=payload)

    response = await client.get_manuscript_project("proj-99")

    assert isinstance(response, ManuscriptProjectResponse)
    assert response.id == "proj-99"
    client._request.assert_awaited_once_with(
        "GET",
        "/api/v1/writing/manuscripts/projects/proj-99",
    )


@pytest.mark.asyncio
async def test_update_manuscript_project_passes_expected_version_header():
    client = TLDWAPIClient("http://example.test", "token")
    payload = _project_payload(project_id="proj-2", version=4)
    client._request = AsyncMock(return_value=payload)

    response = await client.update_manuscript_project(
        "proj-2",
        ManuscriptProjectUpdateRequest(status="writing", synopsis="Updated synopsis"),
        expected_version=3,
    )

    assert isinstance(response, ManuscriptProjectResponse)
    assert response.version == 4
    client._request.assert_awaited_once_with(
        "PATCH",
        "/api/v1/writing/manuscripts/projects/proj-2",
        json_data={"status": "writing", "synopsis": "Updated synopsis"},
        headers={"expected-version": "3"},
    )


@pytest.mark.asyncio
async def test_update_manuscript_project_preserves_explicit_nulls_for_clearing():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(return_value=_project_payload(project_id="proj-2", version=4))

    await client.update_manuscript_project(
        "proj-2",
        ManuscriptProjectUpdateRequest(subtitle=None),
        expected_version=3,
    )

    client._request.assert_awaited_once_with(
        "PATCH",
        "/api/v1/writing/manuscripts/projects/proj-2",
        json_data={"subtitle": None},
        headers={"expected-version": "3"},
    )


@pytest.mark.asyncio
async def test_delete_manuscript_project_passes_expected_version_header():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(return_value={})

    response = await client.delete_manuscript_project("proj-2", expected_version=3)

    assert response == {}
    client._request.assert_awaited_once_with(
        "DELETE",
        "/api/v1/writing/manuscripts/projects/proj-2",
        headers={"expected-version": "3"},
    )


@pytest.mark.asyncio
async def test_reorder_manuscript_entities_omits_null_optional_fields():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(return_value={})

    response = await client.reorder_manuscript_entities(
        "proj-1",
        ReorderRequest(
            entity_type="parts",
            items=[ReorderItem(id="part-1", sort_order=1)],
        ),
    )

    assert response == {}
    client._request.assert_awaited_once_with(
        "POST",
        "/api/v1/writing/manuscripts/projects/proj-1/reorder",
        json_data={"entity_type": "parts", "items": [{"id": "part-1", "sort_order": 1.0}]},
    )


@pytest.mark.asyncio
async def test_part_chapter_scene_and_search_routes_wire_correctly():
    client = TLDWAPIClient("http://example.test", "token")
    mocked = AsyncMock(
        side_effect=[
            _part_payload(),
            [_part_payload()],
            _part_payload(version=2),
            {},
            _chapter_payload(),
            [_chapter_payload()],
            _chapter_payload(version=2),
            {},
            _scene_payload(),
            [_scene_payload()],
            _scene_payload(version=2),
            {},
            {
                "project_id": "proj-1",
                "parts": [{"id": "part-1", "title": "Part One", "sort_order": 1.0, "chapters": []}],
                "unassigned_chapters": [],
            },
            {"query": "Opening", "results": [{"id": "scene-1", "title": "Scene One", "chapter_id": "chapter-1"}]},
        ]
    )
    client._request = mocked

    created_part = await client.create_manuscript_part("proj-1", ManuscriptPartCreateRequest(title="Part One"))
    parts = await client.list_manuscript_parts("proj-1")
    updated_part = await client.update_manuscript_part(
        "part-1",
        ManuscriptPartUpdateRequest(synopsis=None),
        expected_version=1,
    )
    deleted_part = await client.delete_manuscript_part("part-1", expected_version=2)
    created_chapter = await client.create_manuscript_chapter(
        "proj-1",
        ManuscriptChapterCreateRequest(title="Chapter One", part_id="part-1"),
    )
    chapters = await client.list_manuscript_chapters("proj-1", part_id="part-1")
    updated_chapter = await client.update_manuscript_chapter(
        "chapter-1",
        ManuscriptChapterUpdateRequest(part_id=None),
        expected_version=1,
    )
    deleted_chapter = await client.delete_manuscript_chapter("chapter-1", expected_version=2)
    created_scene = await client.create_manuscript_scene(
        "chapter-1",
        ManuscriptSceneCreateRequest(title="Scene One", content_plain="Opening line."),
    )
    scenes = await client.list_manuscript_scenes("chapter-1")
    updated_scene = await client.update_manuscript_scene(
        "scene-1",
        ManuscriptSceneUpdateRequest(content=None, content_plain="Plain only"),
        expected_version=1,
    )
    deleted_scene = await client.delete_manuscript_scene("scene-1", expected_version=2)
    structure = await client.get_manuscript_project_structure("proj-1")
    search = await client.search_manuscript_project("proj-1", "Opening", limit=5)

    assert isinstance(created_part, ManuscriptPartResponse)
    assert isinstance(parts[0], ManuscriptPartResponse)
    assert isinstance(updated_part, ManuscriptPartResponse)
    assert deleted_part == {}
    assert isinstance(created_chapter, ManuscriptChapterResponse)
    assert isinstance(chapters[0], ManuscriptChapterResponse)
    assert isinstance(updated_chapter, ManuscriptChapterResponse)
    assert deleted_chapter == {}
    assert isinstance(created_scene, ManuscriptSceneResponse)
    assert isinstance(scenes[0], ManuscriptSceneResponse)
    assert isinstance(updated_scene, ManuscriptSceneResponse)
    assert deleted_scene == {}
    assert isinstance(structure, ManuscriptStructureResponse)
    assert isinstance(search, ManuscriptSearchResponse)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/writing/manuscripts/projects/proj-1/parts")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/writing/manuscripts/projects/proj-1/parts")
    assert mocked.await_args_list[2].args[:2] == ("PATCH", "/api/v1/writing/manuscripts/parts/part-1")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"synopsis": None}
    assert mocked.await_args_list[2].kwargs["headers"] == {"expected-version": "1"}
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/writing/manuscripts/parts/part-1")
    assert mocked.await_args_list[3].kwargs["headers"] == {"expected-version": "2"}
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/writing/manuscripts/projects/proj-1/chapters")
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/writing/manuscripts/projects/proj-1/chapters")
    assert mocked.await_args_list[5].kwargs["params"] == {"part_id": "part-1"}
    assert mocked.await_args_list[6].args[:2] == ("PATCH", "/api/v1/writing/manuscripts/chapters/chapter-1")
    assert mocked.await_args_list[6].kwargs["json_data"] == {"part_id": None}
    assert mocked.await_args_list[6].kwargs["headers"] == {"expected-version": "1"}
    assert mocked.await_args_list[7].args[:2] == ("DELETE", "/api/v1/writing/manuscripts/chapters/chapter-1")
    assert mocked.await_args_list[7].kwargs["headers"] == {"expected-version": "2"}
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/writing/manuscripts/chapters/chapter-1/scenes")
    assert mocked.await_args_list[9].args[:2] == ("GET", "/api/v1/writing/manuscripts/chapters/chapter-1/scenes")
    assert mocked.await_args_list[10].args[:2] == ("PATCH", "/api/v1/writing/manuscripts/scenes/scene-1")
    assert mocked.await_args_list[10].kwargs["json_data"] == {"content": None, "content_plain": "Plain only"}
    assert mocked.await_args_list[10].kwargs["headers"] == {"expected-version": "1"}
    assert mocked.await_args_list[11].args[:2] == ("DELETE", "/api/v1/writing/manuscripts/scenes/scene-1")
    assert mocked.await_args_list[11].kwargs["headers"] == {"expected-version": "2"}
    assert mocked.await_args_list[12].args[:2] == ("GET", "/api/v1/writing/manuscripts/projects/proj-1/structure")
    assert mocked.await_args_list[13].args[:2] == ("GET", "/api/v1/writing/manuscripts/projects/proj-1/search")
    assert mocked.await_args_list[13].kwargs["params"] == {"q": "Opening", "limit": 5}


def test_response_models_accept_forward_compatible_status_values():
    project_payload = _project_payload()
    project_payload["status"] = "paused"
    chapter_payload = _chapter_payload()
    chapter_payload["status"] = "blocked"
    scene_payload = _scene_payload()
    scene_payload["status"] = "queued"

    assert ManuscriptProjectResponse.model_validate(project_payload).status == "paused"
    assert ManuscriptChapterResponse.model_validate(chapter_payload).status == "blocked"
    assert ManuscriptSceneResponse.model_validate(scene_payload).status == "queued"
