"""Tests for writing manuscript endpoint wiring on the shared TLDW API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ManuscriptChapterCreate,
    ManuscriptChapterResponse,
    ManuscriptChapterUpdate,
    ManuscriptPartCreate,
    ManuscriptPartResponse,
    ManuscriptProjectCreate,
    ManuscriptProjectListResponse,
    ManuscriptProjectResponse,
    ManuscriptProjectUpdate,
    ManuscriptSceneCreate,
    ManuscriptSceneResponse,
    ManuscriptSceneUpdate,
    ManuscriptStructureResponse,
    TLDWAPIClient,
)


def _project_payload(**overrides):
    payload = {
        "id": "project-1",
        "title": "Novel",
        "subtitle": None,
        "author": "Ada",
        "genre": "sci-fi",
        "status": "draft",
        "synopsis": "Draft synopsis",
        "target_word_count": 90000,
        "settings": {},
        "word_count": 0,
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


def _part_payload(**overrides):
    payload = {
        "id": "manuscript-1",
        "project_id": "project-1",
        "title": "Manuscript",
        "sort_order": 0.0,
        "synopsis": "Book one",
        "word_count": 0,
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


def _chapter_payload(**overrides):
    payload = {
        "id": "chapter-1",
        "project_id": "project-1",
        "part_id": "manuscript-1",
        "title": "Chapter 1",
        "sort_order": 0.0,
        "synopsis": "Opening",
        "pov_character_id": None,
        "word_count": 0,
        "status": "draft",
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


def _scene_payload(**overrides):
    payload = {
        "id": "scene-1",
        "chapter_id": "chapter-1",
        "project_id": "project-1",
        "title": "Scene 1",
        "sort_order": 0.0,
        "content_json": None,
        "content_plain": "Opening line.",
        "synopsis": "Meet the protagonist",
        "word_count": 2,
        "pov_character_id": None,
        "status": "draft",
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_writing_project_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"projects": [_project_payload()], "total": 1},
            _project_payload(),
            _project_payload(),
            _project_payload(title="Novel v2", version=2),
            None,
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_manuscript_projects(status="draft", limit=10, offset=5)
    created = await client.create_manuscript_project(
        ManuscriptProjectCreate(title="Novel", author="Ada", genre="sci-fi")
    )
    fetched = await client.get_manuscript_project("project-1")
    updated = await client.update_manuscript_project(
        "project-1",
        ManuscriptProjectUpdate(title="Novel v2"),
        expected_version=1,
    )
    deleted = await client.delete_manuscript_project("project-1", expected_version=2)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/writing/manuscripts/projects")
    assert mocked.await_args_list[0].kwargs["params"] == {"status": "draft", "limit": 10, "offset": 5}
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/writing/manuscripts/projects")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/writing/manuscripts/projects/project-1")
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/writing/manuscripts/projects/project-1")
    assert mocked.await_args_list[3].kwargs["headers"] == {"expected-version": "1"}
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/writing/manuscripts/projects/project-1")
    assert mocked.await_args_list[4].kwargs["headers"] == {"expected-version": "2"}
    assert isinstance(listed, ManuscriptProjectListResponse)
    assert isinstance(created, ManuscriptProjectResponse)
    assert isinstance(fetched, ManuscriptProjectResponse)
    assert isinstance(updated, ManuscriptProjectResponse)
    assert deleted is True


@pytest.mark.asyncio
async def test_writing_hierarchy_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            _part_payload(),
            [_part_payload()],
            _chapter_payload(),
            [_chapter_payload()],
            _scene_payload(),
            [_scene_payload()],
            _scene_payload(title="Scene 1 v2", version=2),
            {
                "project_id": "project-1",
                "parts": [
                    {
                        "id": "manuscript-1",
                        "title": "Manuscript",
                        "sort_order": 0.0,
                        "word_count": 2,
                        "version": 1,
                        "chapters": [
                            {
                                "id": "chapter-1",
                                "title": "Chapter 1",
                                "sort_order": 0.0,
                                "part_id": "manuscript-1",
                                "word_count": 2,
                                "status": "draft",
                                "version": 1,
                                "scenes": [
                                    {
                                        "id": "scene-1",
                                        "title": "Scene 1",
                                        "sort_order": 0.0,
                                        "word_count": 2,
                                        "status": "draft",
                                        "version": 1,
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "unassigned_chapters": [],
            },
            None,
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    manuscript = await client.create_manuscript(
        "project-1",
        ManuscriptPartCreate(title="Manuscript", synopsis="Book one"),
    )
    manuscripts = await client.list_manuscripts("project-1")
    chapter = await client.create_manuscript_chapter(
        "project-1",
        ManuscriptChapterCreate(title="Chapter 1", part_id="manuscript-1"),
    )
    chapters = await client.list_manuscript_chapters("project-1", part_id="manuscript-1")
    scene = await client.create_manuscript_scene(
        "chapter-1",
        ManuscriptSceneCreate(title="Scene 1", content_plain="Opening line."),
    )
    scenes = await client.list_manuscript_scenes("chapter-1")
    updated_scene = await client.update_manuscript_scene(
        "scene-1",
        ManuscriptSceneUpdate(title="Scene 1 v2"),
        expected_version=1,
    )
    structure = await client.get_manuscript_structure("project-1")
    deleted_scene = await client.delete_manuscript_scene("scene-1", expected_version=2)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/writing/manuscripts/projects/project-1/parts")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/writing/manuscripts/projects/project-1/parts")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/writing/manuscripts/projects/project-1/chapters")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/writing/manuscripts/projects/project-1/chapters")
    assert mocked.await_args_list[3].kwargs["params"] == {"part_id": "manuscript-1"}
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/writing/manuscripts/chapters/chapter-1/scenes")
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/writing/manuscripts/chapters/chapter-1/scenes")
    assert mocked.await_args_list[6].args[:2] == ("PATCH", "/api/v1/writing/manuscripts/scenes/scene-1")
    assert mocked.await_args_list[6].kwargs["headers"] == {"expected-version": "1"}
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/writing/manuscripts/projects/project-1/structure")
    assert mocked.await_args_list[8].args[:2] == ("DELETE", "/api/v1/writing/manuscripts/scenes/scene-1")
    assert isinstance(manuscript, ManuscriptPartResponse)
    assert isinstance(manuscripts[0], ManuscriptPartResponse)
    assert isinstance(chapter, ManuscriptChapterResponse)
    assert isinstance(chapters[0], ManuscriptChapterResponse)
    assert isinstance(scene, ManuscriptSceneResponse)
    assert isinstance(scenes[0], ManuscriptSceneResponse)
    assert isinstance(updated_scene, ManuscriptSceneResponse)
    assert isinstance(structure, ManuscriptStructureResponse)
    assert deleted_scene is True
