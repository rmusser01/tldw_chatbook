"""Tests for writing manuscript endpoint wiring on the shared TLDW API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ManuscriptAnalysisListResponse,
    ManuscriptAnalysisRequest,
    ManuscriptAnalysisResponse,
    ManuscriptCharacterCreate,
    ManuscriptCharacterResponse,
    ManuscriptCharacterUpdate,
    ManuscriptChapterCreate,
    ManuscriptChapterResponse,
    ManuscriptChapterUpdate,
    ManuscriptCitationCreate,
    ManuscriptCitationResponse,
    ManuscriptPartCreate,
    ManuscriptPartResponse,
    ManuscriptPlotEventCreate,
    ManuscriptPlotEventResponse,
    ManuscriptPlotEventUpdate,
    ManuscriptPlotHoleCreate,
    ManuscriptPlotHoleResponse,
    ManuscriptPlotHoleUpdate,
    ManuscriptPlotLineCreate,
    ManuscriptPlotLineResponse,
    ManuscriptPlotLineUpdate,
    ManuscriptProjectCreate,
    ManuscriptProjectListResponse,
    ManuscriptProjectResponse,
    ManuscriptProjectUpdate,
    ManuscriptRelationshipCreate,
    ManuscriptRelationshipResponse,
    ManuscriptResearchRequest,
    ManuscriptResearchResponse,
    ManuscriptSceneCreate,
    ManuscriptSceneResponse,
    ManuscriptSceneUpdate,
    ManuscriptStructureResponse,
    ManuscriptTrashListResponse,
    ManuscriptVersionCreateRequest,
    ManuscriptVersionListResponse,
    ManuscriptVersionResponse,
    ManuscriptWorldInfoCreate,
    ManuscriptWorldInfoResponse,
    ManuscriptWorldInfoUpdate,
    ReorderItem,
    ReorderRequest,
    SceneCharacterLink,
    SceneCharacterLinkResponse,
    SceneWorldInfoLink,
    SceneWorldInfoLinkResponse,
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


def _version_payload(**overrides):
    payload = {
        "id": "version-1",
        "entity_type": "scene",
        "entity_id": "scene-1",
        "project_id": "project-1",
        "version_number": 1,
        "label": "First draft",
        "payload": {"title": "Scene 1", "content_plain": "Opening line."},
        "created_at": "2026-04-21T00:02:00Z",
        "client_id": "server-client",
    }
    payload.update(overrides)
    return payload


def _character_payload(**overrides):
    payload = {
        "id": "character-1",
        "project_id": "project-1",
        "name": "Ada",
        "role": "protagonist",
        "cast_group": "heroes",
        "full_name": "Ada Lovelace",
        "age": None,
        "gender": None,
        "appearance": None,
        "personality": None,
        "backstory": None,
        "motivation": None,
        "arc_summary": None,
        "notes": None,
        "custom_fields": {},
        "sort_order": 0.0,
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


def _world_info_payload(**overrides):
    payload = {
        "id": "world-1",
        "project_id": "project-1",
        "kind": "location",
        "name": "Capital",
        "description": "City",
        "parent_id": None,
        "properties": {},
        "tags": [],
        "sort_order": 0.0,
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


def _plot_line_payload(**overrides):
    payload = {
        "id": "plot-line-1",
        "project_id": "project-1",
        "title": "Main Plot",
        "description": None,
        "status": "active",
        "color": None,
        "sort_order": 0.0,
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


def _plot_event_payload(**overrides):
    payload = {
        "id": "plot-event-1",
        "project_id": "project-1",
        "plot_line_id": "plot-line-1",
        "title": "Inciting Incident",
        "description": None,
        "scene_id": "scene-1",
        "chapter_id": "chapter-1",
        "event_type": "plot",
        "sort_order": 0.0,
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


def _plot_hole_payload(**overrides):
    payload = {
        "id": "plot-hole-1",
        "project_id": "project-1",
        "title": "Continuity Issue",
        "description": None,
        "severity": "medium",
        "status": "open",
        "resolution": None,
        "scene_id": "scene-1",
        "chapter_id": "chapter-1",
        "plot_line_id": "plot-line-1",
        "detected_by": "manual",
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


def _relationship_payload(**overrides):
    payload = {
        "id": "relationship-1",
        "project_id": "project-1",
        "from_character_id": "character-1",
        "to_character_id": "character-2",
        "relationship_type": "mentor",
        "description": None,
        "bidirectional": True,
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


def _citation_payload(**overrides):
    payload = {
        "id": "citation-1",
        "project_id": "project-1",
        "scene_id": "scene-1",
        "source_type": "manual",
        "source_id": None,
        "source_title": "Reference",
        "excerpt": "Quote",
        "query_used": None,
        "anchor_offset": None,
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


def _analysis_payload(**overrides):
    payload = {
        "id": "analysis-1",
        "project_id": "project-1",
        "scope_type": "scene",
        "scope_id": "scene-1",
        "analysis_type": "pacing",
        "result": {"pacing": 0.8},
        "score": 0.8,
        "stale": False,
        "provider": None,
        "model": None,
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
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
async def test_writing_namespace_gateway_routes_server_only_writing_suite_surfaces(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    await client.call_server_writing_endpoint("GET", "capabilities")
    await client.call_server_writing_endpoint(
        "POST",
        "/api/v1/writing/snapshot/import",
        payload={"projects": []},
        headers={"Idempotency-Key": "writing-import-1"},
    )
    await client.call_server_writing_endpoint(
        "PATCH",
        "templates/three-act",
        payload={"description": "Updated template"},
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/writing/capabilities")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/writing/snapshot/import")
    assert mocked.await_args_list[1].kwargs["json_data"] == {"projects": []}
    assert mocked.await_args_list[1].kwargs["headers"] == {"Idempotency-Key": "writing-import-1"}
    assert mocked.await_args_list[2].args[:2] == ("PATCH", "/api/v1/writing/templates/three-act")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"description": "Updated template"}


@pytest.mark.asyncio
async def test_writing_namespace_gateway_rejects_cross_namespace_and_unsafe_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    unsafe_calls = [
        client.call_server_writing_endpoint("GET", "/api/v1/admin/users"),
        client.call_server_writing_endpoint("GET", "/api/v1/prompts/templates"),
        client.call_server_writing_endpoint("GET", "../admin/users"),
        client.call_server_writing_endpoint("OPTIONS", "capabilities"),
    ]
    for call in unsafe_calls:
        with pytest.raises(ValueError):
            await call

    mocked.assert_not_awaited()


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
    reordered = await client.reorder_manuscript_entities(
        "project-1",
        ReorderRequest(
            entity_type="chapters",
            items=[ReorderItem(id="chapter-1", sort_order=2.0, version=1, new_parent_id="manuscript-1")],
        ),
    )
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
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/writing/manuscripts/projects/project-1/reorder")
    assert mocked.await_args_list[8].kwargs["json_data"] == {
        "entity_type": "chapters",
        "items": [
            {
                "id": "chapter-1",
                "sort_order": 2.0,
                "version": 1,
                "new_parent_id": "manuscript-1",
            }
        ],
    }
    assert mocked.await_args_list[9].args[:2] == ("DELETE", "/api/v1/writing/manuscripts/scenes/scene-1")
    assert isinstance(manuscript, ManuscriptPartResponse)
    assert isinstance(manuscripts[0], ManuscriptPartResponse)
    assert isinstance(chapter, ManuscriptChapterResponse)
    assert isinstance(chapters[0], ManuscriptChapterResponse)
    assert isinstance(scene, ManuscriptSceneResponse)
    assert isinstance(scenes[0], ManuscriptSceneResponse)
    assert isinstance(updated_scene, ManuscriptSceneResponse)
    assert isinstance(structure, ManuscriptStructureResponse)
    assert reordered is True
    assert deleted_scene is True


@pytest.mark.asyncio
async def test_writing_version_and_trash_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            _version_payload(),
            {"versions": [_version_payload()], "total": 1},
            _version_payload(),
            _scene_payload(content_plain="Opening line.", version=3),
            {"items": [_scene_payload(deleted=True, version=4)], "total": 1},
            _scene_payload(deleted=False, version=5),
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_manuscript_version(
        "scene",
        "scene-1",
        ManuscriptVersionCreateRequest(label="First draft"),
    )
    versions = await client.list_manuscript_versions("scene", "scene-1")
    fetched = await client.get_manuscript_version("scene", "scene-1", 1)
    restored_version = await client.restore_manuscript_version(
        "scene",
        "scene-1",
        1,
        expected_version=2,
    )
    trash = await client.list_manuscript_trash(entity_type="scene")
    restored_trash = await client.restore_manuscript_trash("scene", "scene-1", expected_version=4)

    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("POST", "/api/v1/writing/manuscripts/scene/scene-1/versions"),
        ("GET", "/api/v1/writing/manuscripts/scene/scene-1/versions"),
        ("GET", "/api/v1/writing/manuscripts/scene/scene-1/versions/1"),
        ("POST", "/api/v1/writing/manuscripts/scene/scene-1/versions/1/restore"),
        ("GET", "/api/v1/writing/manuscripts/trash"),
        ("POST", "/api/v1/writing/manuscripts/trash/scene/scene-1/restore"),
    ]
    assert mocked.await_args_list[0].kwargs["json_data"] == {"label": "First draft"}
    assert mocked.await_args_list[3].kwargs["headers"] == {"expected-version": "2"}
    assert mocked.await_args_list[4].kwargs["params"] == {"entity_type": "scene"}
    assert mocked.await_args_list[5].kwargs["headers"] == {"expected-version": "4"}
    assert isinstance(created, ManuscriptVersionResponse)
    assert isinstance(versions, ManuscriptVersionListResponse)
    assert isinstance(fetched, ManuscriptVersionResponse)
    assert isinstance(restored_version, ManuscriptSceneResponse)
    assert isinstance(trash, ManuscriptTrashListResponse)
    assert isinstance(restored_trash, ManuscriptSceneResponse)


@pytest.mark.asyncio
async def test_writing_auxiliary_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            _character_payload(),
            [_character_payload()],
            _character_payload(),
            _character_payload(name="Ada v2", version=2),
            None,
            _relationship_payload(),
            [_relationship_payload()],
            None,
            _world_info_payload(),
            [_world_info_payload()],
            _world_info_payload(),
            _world_info_payload(name="Capital v2", version=2),
            None,
            _plot_line_payload(),
            [_plot_line_payload()],
            _plot_line_payload(title="Main Plot v2", version=2),
            None,
            _plot_event_payload(),
            [_plot_event_payload()],
            _plot_event_payload(title="Incident v2", version=2),
            None,
            _plot_hole_payload(),
            [_plot_hole_payload()],
            _plot_hole_payload(status="resolved", version=2),
            None,
            [{"scene_id": "scene-1", "character_id": "character-1", "is_pov": True, "name": "Ada", "role": "protagonist"}],
            [{"scene_id": "scene-1", "character_id": "character-1", "is_pov": True, "name": "Ada", "role": "protagonist"}],
            None,
            [{"scene_id": "scene-1", "world_info_id": "world-1", "name": "Capital", "kind": "location"}],
            [{"scene_id": "scene-1", "world_info_id": "world-1", "name": "Capital", "kind": "location"}],
            None,
            _citation_payload(),
            [_citation_payload()],
            None,
            {"query": "airships", "results": [{"source_id": "media-1", "title": "Source", "excerpt": "Text"}]},
            [_analysis_payload()],
            [_analysis_payload(scope_type="chapter", scope_id="chapter-1")],
            [_analysis_payload(scope_type="project", scope_id="project-1", analysis_type="plot_holes", score=None)],
            [_analysis_payload(scope_type="project", scope_id="project-1", analysis_type="consistency")],
            {"analyses": [_analysis_payload()], "total": 1},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    character = await client.create_manuscript_character(
        "project-1",
        ManuscriptCharacterCreate(name="Ada", role="protagonist", cast_group="heroes"),
    )
    characters = await client.list_manuscript_characters("project-1", role="protagonist", cast_group="heroes")
    fetched_character = await client.get_manuscript_character("character-1")
    updated_character = await client.update_manuscript_character(
        "character-1",
        ManuscriptCharacterUpdate(name="Ada v2"),
        expected_version=1,
    )
    deleted_character = await client.delete_manuscript_character("character-1", expected_version=2)
    relationship = await client.create_manuscript_relationship(
        "project-1",
        ManuscriptRelationshipCreate(
            from_character_id="character-1",
            to_character_id="character-2",
            relationship_type="mentor",
        ),
    )
    relationships = await client.list_manuscript_relationships("project-1")
    deleted_relationship = await client.delete_manuscript_relationship("relationship-1", expected_version=1)
    world_info = await client.create_manuscript_world_info(
        "project-1",
        ManuscriptWorldInfoCreate(kind="location", name="Capital"),
    )
    world_infos = await client.list_manuscript_world_info("project-1", kind="location")
    fetched_world_info = await client.get_manuscript_world_info("world-1")
    updated_world_info = await client.update_manuscript_world_info(
        "world-1",
        ManuscriptWorldInfoUpdate(name="Capital v2"),
        expected_version=1,
    )
    deleted_world_info = await client.delete_manuscript_world_info("world-1", expected_version=2)
    plot_line = await client.create_manuscript_plot_line(
        "project-1",
        ManuscriptPlotLineCreate(title="Main Plot"),
    )
    plot_lines = await client.list_manuscript_plot_lines("project-1")
    updated_plot_line = await client.update_manuscript_plot_line(
        "plot-line-1",
        ManuscriptPlotLineUpdate(title="Main Plot v2"),
        expected_version=1,
    )
    deleted_plot_line = await client.delete_manuscript_plot_line("plot-line-1", expected_version=2)
    plot_event = await client.create_manuscript_plot_event(
        "plot-line-1",
        ManuscriptPlotEventCreate(title="Inciting Incident", scene_id="scene-1", chapter_id="chapter-1"),
    )
    plot_events = await client.list_manuscript_plot_events("plot-line-1")
    updated_plot_event = await client.update_manuscript_plot_event(
        "plot-event-1",
        ManuscriptPlotEventUpdate(title="Incident v2"),
        expected_version=1,
    )
    deleted_plot_event = await client.delete_manuscript_plot_event("plot-event-1", expected_version=2)
    plot_hole = await client.create_manuscript_plot_hole(
        "project-1",
        ManuscriptPlotHoleCreate(title="Continuity Issue", scene_id="scene-1", chapter_id="chapter-1"),
    )
    plot_holes = await client.list_manuscript_plot_holes("project-1", status="open")
    updated_plot_hole = await client.update_manuscript_plot_hole(
        "plot-hole-1",
        ManuscriptPlotHoleUpdate(status="resolved"),
        expected_version=1,
    )
    deleted_plot_hole = await client.delete_manuscript_plot_hole("plot-hole-1", expected_version=2)
    linked_characters = await client.link_manuscript_scene_character(
        "scene-1",
        SceneCharacterLink(character_id="character-1", is_pov=True),
    )
    scene_characters = await client.list_manuscript_scene_characters("scene-1")
    unlinked_character = await client.unlink_manuscript_scene_character("scene-1", "character-1")
    linked_world_info = await client.link_manuscript_scene_world_info(
        "scene-1",
        SceneWorldInfoLink(world_info_id="world-1"),
    )
    scene_world_info = await client.list_manuscript_scene_world_info("scene-1")
    unlinked_world_info = await client.unlink_manuscript_scene_world_info("scene-1", "world-1")
    citation = await client.create_manuscript_citation(
        "scene-1",
        ManuscriptCitationCreate(source_type="manual", source_title="Reference", excerpt="Quote"),
    )
    citations = await client.list_manuscript_citations("scene-1")
    deleted_citation = await client.delete_manuscript_citation("citation-1", expected_version=1)
    research = await client.research_manuscript_scene(
        "scene-1",
        ManuscriptResearchRequest(query="airships", top_k=3),
    )
    scene_analyses = await client.analyze_manuscript_scene(
        "scene-1",
        ManuscriptAnalysisRequest(analysis_types=["pacing"]),
    )
    chapter_analyses = await client.analyze_manuscript_chapter(
        "chapter-1",
        ManuscriptAnalysisRequest(analysis_types=["pacing"]),
    )
    plot_hole_analyses = await client.analyze_manuscript_project_plot_holes(
        "project-1",
        ManuscriptAnalysisRequest(analysis_types=["plot_holes"]),
    )
    consistency_analyses = await client.analyze_manuscript_project_consistency(
        "project-1",
        ManuscriptAnalysisRequest(analysis_types=["consistency"]),
    )
    analyses = await client.list_manuscript_analyses(
        "project-1",
        scope_type="scene",
        analysis_type="pacing",
        include_stale=True,
    )

    expected_routes = [
        ("POST", "/api/v1/writing/manuscripts/projects/project-1/characters"),
        ("GET", "/api/v1/writing/manuscripts/projects/project-1/characters"),
        ("GET", "/api/v1/writing/manuscripts/characters/character-1"),
        ("PATCH", "/api/v1/writing/manuscripts/characters/character-1"),
        ("DELETE", "/api/v1/writing/manuscripts/characters/character-1"),
        ("POST", "/api/v1/writing/manuscripts/projects/project-1/characters/relationships"),
        ("GET", "/api/v1/writing/manuscripts/projects/project-1/characters/relationships"),
        ("DELETE", "/api/v1/writing/manuscripts/characters/relationships/relationship-1"),
        ("POST", "/api/v1/writing/manuscripts/projects/project-1/world-info"),
        ("GET", "/api/v1/writing/manuscripts/projects/project-1/world-info"),
        ("GET", "/api/v1/writing/manuscripts/world-info/world-1"),
        ("PATCH", "/api/v1/writing/manuscripts/world-info/world-1"),
        ("DELETE", "/api/v1/writing/manuscripts/world-info/world-1"),
        ("POST", "/api/v1/writing/manuscripts/projects/project-1/plot-lines"),
        ("GET", "/api/v1/writing/manuscripts/projects/project-1/plot-lines"),
        ("PATCH", "/api/v1/writing/manuscripts/plot-lines/plot-line-1"),
        ("DELETE", "/api/v1/writing/manuscripts/plot-lines/plot-line-1"),
        ("POST", "/api/v1/writing/manuscripts/plot-lines/plot-line-1/events"),
        ("GET", "/api/v1/writing/manuscripts/plot-lines/plot-line-1/events"),
        ("PATCH", "/api/v1/writing/manuscripts/plot-events/plot-event-1"),
        ("DELETE", "/api/v1/writing/manuscripts/plot-events/plot-event-1"),
        ("POST", "/api/v1/writing/manuscripts/projects/project-1/plot-holes"),
        ("GET", "/api/v1/writing/manuscripts/projects/project-1/plot-holes"),
        ("PATCH", "/api/v1/writing/manuscripts/plot-holes/plot-hole-1"),
        ("DELETE", "/api/v1/writing/manuscripts/plot-holes/plot-hole-1"),
        ("POST", "/api/v1/writing/manuscripts/scenes/scene-1/characters"),
        ("GET", "/api/v1/writing/manuscripts/scenes/scene-1/characters"),
        ("DELETE", "/api/v1/writing/manuscripts/scenes/scene-1/characters/character-1"),
        ("POST", "/api/v1/writing/manuscripts/scenes/scene-1/world-info"),
        ("GET", "/api/v1/writing/manuscripts/scenes/scene-1/world-info"),
        ("DELETE", "/api/v1/writing/manuscripts/scenes/scene-1/world-info/world-1"),
        ("POST", "/api/v1/writing/manuscripts/scenes/scene-1/citations"),
        ("GET", "/api/v1/writing/manuscripts/scenes/scene-1/citations"),
        ("DELETE", "/api/v1/writing/manuscripts/citations/citation-1"),
        ("POST", "/api/v1/writing/manuscripts/scenes/scene-1/research"),
        ("POST", "/api/v1/writing/manuscripts/scenes/scene-1/analyze"),
        ("POST", "/api/v1/writing/manuscripts/chapters/chapter-1/analyze"),
        ("POST", "/api/v1/writing/manuscripts/projects/project-1/analyze/plot-holes"),
        ("POST", "/api/v1/writing/manuscripts/projects/project-1/analyze/consistency"),
        ("GET", "/api/v1/writing/manuscripts/projects/project-1/analyses"),
    ]
    assert [call.args[:2] for call in mocked.await_args_list] == expected_routes
    assert mocked.await_args_list[1].kwargs["params"] == {"role": "protagonist", "cast_group": "heroes"}
    assert mocked.await_args_list[3].kwargs["headers"] == {"expected-version": "1"}
    assert mocked.await_args_list[9].kwargs["params"] == {"kind": "location"}
    assert mocked.await_args_list[22].kwargs["params"] == {"status": "open"}
    assert mocked.await_args_list[39].kwargs["params"] == {
        "scope_type": "scene",
        "analysis_type": "pacing",
        "include_stale": True,
    }
    assert isinstance(character, ManuscriptCharacterResponse)
    assert isinstance(characters[0], ManuscriptCharacterResponse)
    assert isinstance(fetched_character, ManuscriptCharacterResponse)
    assert isinstance(updated_character, ManuscriptCharacterResponse)
    assert deleted_character is True
    assert isinstance(relationship, ManuscriptRelationshipResponse)
    assert isinstance(relationships[0], ManuscriptRelationshipResponse)
    assert deleted_relationship is True
    assert isinstance(world_info, ManuscriptWorldInfoResponse)
    assert isinstance(world_infos[0], ManuscriptWorldInfoResponse)
    assert isinstance(fetched_world_info, ManuscriptWorldInfoResponse)
    assert isinstance(updated_world_info, ManuscriptWorldInfoResponse)
    assert deleted_world_info is True
    assert isinstance(plot_line, ManuscriptPlotLineResponse)
    assert isinstance(plot_lines[0], ManuscriptPlotLineResponse)
    assert isinstance(updated_plot_line, ManuscriptPlotLineResponse)
    assert deleted_plot_line is True
    assert isinstance(plot_event, ManuscriptPlotEventResponse)
    assert isinstance(plot_events[0], ManuscriptPlotEventResponse)
    assert isinstance(updated_plot_event, ManuscriptPlotEventResponse)
    assert deleted_plot_event is True
    assert isinstance(plot_hole, ManuscriptPlotHoleResponse)
    assert isinstance(plot_holes[0], ManuscriptPlotHoleResponse)
    assert isinstance(updated_plot_hole, ManuscriptPlotHoleResponse)
    assert deleted_plot_hole is True
    assert isinstance(linked_characters[0], SceneCharacterLinkResponse)
    assert isinstance(scene_characters[0], SceneCharacterLinkResponse)
    assert unlinked_character is True
    assert isinstance(linked_world_info[0], SceneWorldInfoLinkResponse)
    assert isinstance(scene_world_info[0], SceneWorldInfoLinkResponse)
    assert unlinked_world_info is True
    assert isinstance(citation, ManuscriptCitationResponse)
    assert isinstance(citations[0], ManuscriptCitationResponse)
    assert deleted_citation is True
    assert isinstance(research, ManuscriptResearchResponse)
    assert isinstance(scene_analyses[0], ManuscriptAnalysisResponse)
    assert isinstance(chapter_analyses[0], ManuscriptAnalysisResponse)
    assert isinstance(plot_hole_analyses[0], ManuscriptAnalysisResponse)
    assert isinstance(consistency_analyses[0], ManuscriptAnalysisResponse)
    assert isinstance(analyses, ManuscriptAnalysisListResponse)
