import json
from unittest.mock import Mock

import pytest

from tldw_chatbook.Writing_Interop.server_writing_service import ServerWritingService
from tldw_chatbook.runtime_policy import PolicyDecision, PolicyDeniedError


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
            "content_json": json.dumps(request_data.content) if request_data.content is not None else None,
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
            "unassigned_chapters": [
                {
                    "id": "chapter-loose",
                    "title": "Loose Chapter",
                    "sort_order": 99.0,
                    "part_id": None,
                    "version": 1,
                    "scenes": [],
                }
            ],
        }

    async def reorder_manuscript_entities(self, project_id, request_data):
        self.calls.append(("reorder_manuscript_entities", project_id, request_data.model_dump(mode="json", exclude_none=True)))
        return True

    async def create_manuscript_character(self, project_id, request_data):
        self.calls.append(("create_manuscript_character", project_id, request_data.model_dump(mode="json", exclude_none=True)))
        return {
            "id": "character-1",
            "project_id": project_id,
            "name": request_data.name,
            "role": request_data.role,
            "created_at": "2026-04-21T00:00:00Z",
            "last_modified": "2026-04-21T00:01:00Z",
            "client_id": "server-client",
            "version": 1,
        }

    async def list_manuscript_characters(self, project_id, **filters):
        self.calls.append(("list_manuscript_characters", project_id, filters))
        return [
            {
                "id": "character-1",
                "project_id": project_id,
                "name": "Ada",
                "role": "protagonist",
                "created_at": "2026-04-21T00:00:00Z",
                "last_modified": "2026-04-21T00:01:00Z",
                "client_id": "server-client",
                "version": 1,
            }
        ]

    async def create_manuscript_world_info(self, project_id, request_data):
        self.calls.append(("create_manuscript_world_info", project_id, request_data.model_dump(mode="json", exclude_none=True)))
        return {
            "id": "world-1",
            "project_id": project_id,
            "kind": request_data.kind,
            "name": request_data.name,
            "created_at": "2026-04-21T00:00:00Z",
            "last_modified": "2026-04-21T00:01:00Z",
            "client_id": "server-client",
            "version": 1,
        }

    async def create_manuscript_plot_line(self, project_id, request_data):
        self.calls.append(("create_manuscript_plot_line", project_id, request_data.model_dump(mode="json", exclude_none=True)))
        return {
            "id": "plot-line-1",
            "project_id": project_id,
            "title": request_data.title,
            "created_at": "2026-04-21T00:00:00Z",
            "last_modified": "2026-04-21T00:01:00Z",
            "client_id": "server-client",
            "version": 1,
        }

    async def create_manuscript_plot_event(self, plot_line_id, request_data):
        self.calls.append(("create_manuscript_plot_event", plot_line_id, request_data.model_dump(mode="json", exclude_none=True)))
        return {
            "id": "plot-event-1",
            "project_id": "project-1",
            "plot_line_id": plot_line_id,
            "title": request_data.title,
            "created_at": "2026-04-21T00:00:00Z",
            "last_modified": "2026-04-21T00:01:00Z",
            "client_id": "server-client",
            "version": 1,
        }

    async def create_manuscript_plot_hole(self, project_id, request_data):
        self.calls.append(("create_manuscript_plot_hole", project_id, request_data.model_dump(mode="json", exclude_none=True)))
        return {
            "id": "plot-hole-1",
            "project_id": project_id,
            "title": request_data.title,
            "created_at": "2026-04-21T00:00:00Z",
            "last_modified": "2026-04-21T00:01:00Z",
            "client_id": "server-client",
            "version": 1,
        }

    async def link_manuscript_scene_character(self, scene_id, request_data):
        self.calls.append(("link_manuscript_scene_character", scene_id, request_data.model_dump(mode="json", exclude_none=True)))
        return [{"scene_id": scene_id, "character_id": request_data.character_id, "is_pov": request_data.is_pov, "name": "Ada", "role": "protagonist"}]

    async def create_manuscript_citation(self, scene_id, request_data):
        self.calls.append(("create_manuscript_citation", scene_id, request_data.model_dump(mode="json", exclude_none=True)))
        return {
            "id": "citation-1",
            "project_id": "project-1",
            "scene_id": scene_id,
            "source_type": request_data.source_type,
            "created_at": "2026-04-21T00:00:00Z",
            "last_modified": "2026-04-21T00:01:00Z",
            "client_id": "server-client",
            "version": 1,
        }

    async def analyze_manuscript_scene(self, scene_id, request_data):
        self.calls.append(("analyze_manuscript_scene", scene_id, request_data.model_dump(mode="json", exclude_none=True)))
        return [
            {
                "id": "analysis-1",
                "project_id": "project-1",
                "scope_type": "scene",
                "scope_id": scene_id,
                "analysis_type": request_data.analysis_types[0],
                "result": {"pacing": 0.8},
                "created_at": "2026-04-21T00:00:00Z",
                "last_modified": "2026-04-21T00:01:00Z",
                "version": 1,
            }
        ]


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
    assert scene["content_markdown_fidelity"] == "chatbook_markdown"
    assert structure["manuscripts"][0]["record_id"] == "server:writing_manuscript:manuscript-1"
    assert structure["unassigned_chapters"][0]["record_id"] == "server:writing_chapter:chapter-loose"
    assert structure["unassigned_chapters"][0]["outline_bucket"] == "unassigned_chapters"
    assert structure["unassigned_chapters"][0]["outline_parent_type"] == "project"
    assert structure["unassigned_chapters"][0]["outline_parent_id"] == "project-1"
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
            {
                "title": "Scene 1",
                "content": {
                    "type": "doc",
                    "content": [
                        {
                            "type": "paragraph",
                            "attrs": {
                                "tldw_chatbook_markdown": True,
                                "format": "markdown",
                                "version": 1,
                            },
                            "content": [{"type": "text", "text": "Opening line."}],
                        }
                    ],
                },
                "content_plain": "Opening line.",
                "sort_order": 0.0,
                "status": "draft",
            },
        ),
        ("get_manuscript_structure", "project-1"),
    ]


@pytest.mark.asyncio
async def test_server_writing_service_maps_chatbook_reorder_to_server_parts():
    client = FakeWritingClient()
    service = ServerWritingService(client=client)

    reordered = await service.reorder_entities(
        "project-1",
        "manuscripts",
        [{"id": "manuscript-1", "sort_order": 2.0, "version": 1}],
    )

    assert reordered is True
    assert client.calls == [
        (
            "reorder_manuscript_entities",
            "project-1",
            {
                "entity_type": "parts",
                "items": [{"id": "manuscript-1", "sort_order": 2.0, "version": 1}],
            },
        )
    ]


@pytest.mark.asyncio
async def test_server_writing_service_routes_auxiliary_manuscript_surfaces():
    client = FakeWritingClient()
    service = ServerWritingService(client=client)

    character = await service.create_character("project-1", name="Ada", role="protagonist")
    characters = await service.list_characters("project-1", role="protagonist")
    world = await service.create_world_info("project-1", kind="location", name="Capital")
    plot_line = await service.create_plot_line("project-1", title="Main Plot")
    plot_event = await service.create_plot_event(plot_line["id"], title="Inciting Incident", scene_id="scene-1")
    plot_hole = await service.create_plot_hole("project-1", title="Continuity Issue", scene_id="scene-1")
    scene_characters = await service.link_scene_character("scene-1", character_id=character["id"], is_pov=True)
    citation = await service.create_citation("scene-1", source_type="manual", source_title="Reference")
    analyses = await service.analyze_scene("scene-1", analysis_types=["pacing"])

    assert character["record_id"] == "server:writing_character:character-1"
    assert characters[0]["record_type"] == "writing_character"
    assert world["record_id"] == "server:writing_world_info:world-1"
    assert plot_line["record_id"] == "server:writing_plot_line:plot-line-1"
    assert plot_event["record_id"] == "server:writing_plot_event:plot-event-1"
    assert plot_hole["record_id"] == "server:writing_plot_hole:plot-hole-1"
    assert scene_characters[0]["record_id"] == "server:writing_scene_character_link:scene-1:character-1"
    assert citation["record_id"] == "server:writing_citation:citation-1"
    assert analyses[0]["record_id"] == "server:writing_analysis:analysis-1"
    assert client.calls == [
        ("create_manuscript_character", "project-1", {"name": "Ada", "role": "protagonist", "custom_fields": {}, "sort_order": 0.0}),
        ("list_manuscript_characters", "project-1", {"role": "protagonist", "cast_group": None}),
        ("create_manuscript_world_info", "project-1", {"kind": "location", "name": "Capital", "properties": {}, "tags": [], "sort_order": 0.0}),
        ("create_manuscript_plot_line", "project-1", {"title": "Main Plot", "status": "active", "sort_order": 0.0}),
        ("create_manuscript_plot_event", "plot-line-1", {"title": "Inciting Incident", "scene_id": "scene-1", "event_type": "plot", "sort_order": 0.0}),
        ("create_manuscript_plot_hole", "project-1", {"title": "Continuity Issue", "scene_id": "scene-1", "severity": "medium", "detected_by": "manual"}),
        ("link_manuscript_scene_character", "scene-1", {"character_id": "character-1", "is_pov": True}),
        ("create_manuscript_citation", "scene-1", {"source_type": "manual", "source_title": "Reference"}),
        ("analyze_manuscript_scene", "scene-1", {"analysis_types": ["pacing"]}),
    ]


@pytest.mark.asyncio
async def test_server_writing_service_rejects_direct_manuscript_scene_creation():
    service = ServerWritingService(client=FakeWritingClient())

    with pytest.raises(NotImplementedError, match="Direct manuscript-level scenes"):
        await service.create_scene(
            None,
            manuscript_id="manuscript-1",
            title="Prologue",
            content_markdown="Direct scene.",
        )


@pytest.mark.asyncio
async def test_server_writing_service_enforces_policy_actions():
    client = FakeWritingClient()
    policy = Mock()
    service = ServerWritingService(client=client, policy_enforcer=policy)

    project = await service.create_project(title="Novel", author="Ada")
    manuscript = await service.create_manuscript(project["id"], title="Book One")
    chapter = await service.create_chapter(project["id"], title="Chapter 1", manuscript_id=manuscript["id"])
    await service.create_scene(chapter["id"], title="Scene 1", content_markdown="Opening line.")
    await service.get_structure(project["id"])
    await service.reorder_entities(
        project["id"],
        "manuscripts",
        [{"id": "manuscript-1", "sort_order": 2.0, "version": 1}],
    )
    with pytest.raises(NotImplementedError, match="Server writing version history"):
        await service.create_version("scene", "scene-1")

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "writing.projects.create.server",
        "writing.manuscripts.create.server",
        "writing.chapters.create.server",
        "writing.scenes.create.server",
        "writing.projects.structure.server",
        "writing.outline.reorder.server",
        "writing.versions.create.server",
    ]


@pytest.mark.asyncio
async def test_server_writing_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeWritingClient()
    service = ServerWritingService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.create_project(title="Novel")

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
