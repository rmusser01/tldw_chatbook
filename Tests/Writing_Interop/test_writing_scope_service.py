import pytest

from tldw_chatbook.Writing_Interop.writing_scope_service import WritingScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeWritingService:
    def __init__(self, source):
        self.source = source
        self.calls = []

    async def list_projects(self, *, limit=100, offset=0, status=None):
        self.calls.append(("list_projects", limit, offset, status))
        return [{"id": f"{self.source}-project-1", "title": "Novel", "version": 1}]

    async def create_project(self, **kwargs):
        self.calls.append(("create_project", kwargs))
        return {"id": f"{self.source}-project-1", "title": kwargs["title"], "version": 1}

    async def update_project(self, project_id, *, expected_version=None, **kwargs):
        self.calls.append(("update_project", project_id, expected_version, kwargs))
        return {"id": project_id, "title": kwargs["title"], "version": 2}

    async def delete_project(self, project_id, *, expected_version=None):
        self.calls.append(("delete_project", project_id, expected_version))
        return True

    async def create_manuscript(self, project_id, **kwargs):
        self.calls.append(("create_manuscript", project_id, kwargs))
        return {"id": f"{self.source}-manuscript-1", "project_id": project_id, "title": kwargs["title"], "version": 1}

    async def update_manuscript(self, manuscript_id, *, expected_version=None, **kwargs):
        self.calls.append(("update_manuscript", manuscript_id, expected_version, kwargs))
        return {"id": manuscript_id, "project_id": f"{self.source}-project-1", "title": kwargs["title"], "version": 2}

    async def delete_manuscript(self, manuscript_id, *, expected_version=None):
        self.calls.append(("delete_manuscript", manuscript_id, expected_version))
        return True

    async def create_chapter(self, project_id, **kwargs):
        self.calls.append(("create_chapter", project_id, kwargs))
        return {
            "id": f"{self.source}-chapter-1",
            "project_id": project_id,
            "manuscript_id": kwargs.get("manuscript_id"),
            "title": kwargs["title"],
            "version": 1,
        }

    async def update_chapter(self, chapter_id, *, expected_version=None, **kwargs):
        self.calls.append(("update_chapter", chapter_id, expected_version, kwargs))
        return {
            "id": chapter_id,
            "project_id": f"{self.source}-project-1",
            "manuscript_id": kwargs.get("manuscript_id"),
            "title": kwargs["title"],
            "version": 2,
        }

    async def delete_chapter(self, chapter_id, *, expected_version=None):
        self.calls.append(("delete_chapter", chapter_id, expected_version))
        return True

    async def create_scene(self, chapter_id, **kwargs):
        self.calls.append(("create_scene", chapter_id, kwargs))
        return {
            "id": f"{self.source}-scene-1",
            "chapter_id": chapter_id,
            "title": kwargs["title"],
            "content_markdown": kwargs.get("content_markdown", ""),
            "version": 1,
        }

    async def update_scene(self, scene_id, *, expected_version=None, **kwargs):
        self.calls.append(("update_scene", scene_id, expected_version, kwargs))
        return {
            "id": scene_id,
            "chapter_id": f"{self.source}-chapter-1",
            "title": kwargs["title"],
            "content_markdown": kwargs.get("content_markdown", ""),
            "version": 2,
        }

    async def delete_scene(self, scene_id, *, expected_version=None):
        self.calls.append(("delete_scene", scene_id, expected_version))
        return True

    async def create_version(self, entity_type, entity_id, *, label=None):
        self.calls.append(("create_version", entity_type, entity_id, label))
        return {
            "id": f"{self.source}-version-1",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "version_number": 1,
            "label": label,
            "payload": {"title": "Snapshot"},
        }

    async def list_versions(self, entity_type, entity_id):
        self.calls.append(("list_versions", entity_type, entity_id))
        return [
            {
                "id": f"{self.source}-version-1",
                "entity_type": entity_type,
                "entity_id": entity_id,
                "version_number": 1,
                "payload": {"title": "Snapshot"},
            }
        ]

    async def get_version(self, entity_type, entity_id, version_number):
        self.calls.append(("get_version", entity_type, entity_id, version_number))
        return {
            "id": f"{self.source}-version-1",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "version_number": version_number,
            "payload": {"title": "Snapshot"},
        }

    async def restore_version(self, entity_type, entity_id, version_number, *, expected_version=None):
        self.calls.append(("restore_version", entity_type, entity_id, version_number, expected_version))
        return {
            "id": entity_id,
            "title": "Snapshot",
            "version": 2,
        }

    async def list_trash(self, *, entity_type=None):
        self.calls.append(("list_trash", entity_type))
        return [
            {
                "id": f"{self.source}-scene-1",
                "title": "Deleted Scene",
                "content_markdown": "Draft",
                "deleted": 1,
            }
        ]

    async def restore_trash(self, entity_type, entity_id, *, expected_version=None):
        self.calls.append(("restore_trash", entity_type, entity_id, expected_version))
        return {
            "id": entity_id,
            "title": "Deleted Scene",
            "content_markdown": "Draft",
            "deleted": 0,
            "version": 2,
        }

    async def reorder_entities(self, project_id, entity_type, items):
        self.calls.append(("reorder_entities", project_id, entity_type, items))
        return True

    async def get_structure(self, project_id):
        self.calls.append(("get_structure", project_id))
        return {
            "project_id": project_id,
            "manuscripts": [
                {
                    "id": f"{self.source}-manuscript-1",
                    "title": "Book One",
                    "chapters": [],
                    "scenes": [],
                }
            ],
            "unassigned_chapters": [
                {
                    "id": f"{self.source}-chapter-loose",
                    "title": "Loose Chapter",
                    "part_id": None,
                    "version": 1,
                    "scenes": [],
                }
            ],
        }

    async def create_character(self, project_id, **kwargs):
        self.calls.append(("create_character", project_id, kwargs))
        return {"id": f"{self.source}-character-1", "project_id": project_id, "name": kwargs["name"], "version": 1}

    async def list_characters(self, project_id, **kwargs):
        self.calls.append(("list_characters", project_id, kwargs))
        return [{"id": f"{self.source}-character-1", "project_id": project_id, "name": "Ada", "version": 1}]

    async def create_world_info(self, project_id, **kwargs):
        self.calls.append(("create_world_info", project_id, kwargs))
        return {"id": f"{self.source}-world-1", "project_id": project_id, "kind": kwargs["kind"], "name": kwargs["name"], "version": 1}

    async def create_plot_line(self, project_id, **kwargs):
        self.calls.append(("create_plot_line", project_id, kwargs))
        return {"id": f"{self.source}-plot-line-1", "project_id": project_id, "title": kwargs["title"], "version": 1}

    async def create_plot_event(self, plot_line_id, **kwargs):
        self.calls.append(("create_plot_event", plot_line_id, kwargs))
        return {"id": f"{self.source}-plot-event-1", "plot_line_id": plot_line_id, "title": kwargs["title"], "version": 1}

    async def create_plot_hole(self, project_id, **kwargs):
        self.calls.append(("create_plot_hole", project_id, kwargs))
        return {"id": f"{self.source}-plot-hole-1", "project_id": project_id, "title": kwargs["title"], "version": 1}

    async def link_scene_character(self, scene_id, **kwargs):
        self.calls.append(("link_scene_character", scene_id, kwargs))
        return [{"scene_id": scene_id, "character_id": kwargs["character_id"], "is_pov": kwargs.get("is_pov", False)}]

    async def link_scene_world_info(self, scene_id, **kwargs):
        self.calls.append(("link_scene_world_info", scene_id, kwargs))
        return [{"scene_id": scene_id, "world_info_id": kwargs["world_info_id"]}]

    async def create_citation(self, scene_id, **kwargs):
        self.calls.append(("create_citation", scene_id, kwargs))
        return {"id": f"{self.source}-citation-1", "scene_id": scene_id, "source_type": kwargs["source_type"], "version": 1}

    async def research_scene(self, scene_id, **kwargs):
        self.calls.append(("research_scene", scene_id, kwargs))
        return {
            "scene_id": scene_id,
            "query": kwargs["query"],
            "results": [{"source_id": f"{self.source}-source-1", "title": "Reference"}],
        }

    async def analyze_scene(self, scene_id, **kwargs):
        self.calls.append(("analyze_scene", scene_id, kwargs))
        return [{"id": f"{self.source}-analysis-1", "scope_type": "scene", "scope_id": scene_id, "analysis_type": kwargs["analysis_types"][0]}]

    async def analyze_chapter(self, chapter_id, **kwargs):
        self.calls.append(("analyze_chapter", chapter_id, kwargs))
        return [{"id": f"{self.source}-analysis-2", "scope_type": "chapter", "scope_id": chapter_id, "analysis_type": kwargs["analysis_types"][0]}]

    async def analyze_project_plot_holes(self, project_id, **kwargs):
        self.calls.append(("analyze_project_plot_holes", project_id, kwargs))
        return [{"id": f"{self.source}-analysis-3", "scope_type": "project", "scope_id": project_id, "analysis_type": "plot_holes"}]

    async def analyze_project_consistency(self, project_id, **kwargs):
        self.calls.append(("analyze_project_consistency", project_id, kwargs))
        return [{"id": f"{self.source}-analysis-4", "scope_type": "project", "scope_id": project_id, "analysis_type": "consistency"}]

    async def list_analyses(self, project_id, **kwargs):
        self.calls.append(("list_analyses", project_id, kwargs))
        return {
            "analyses": [{"id": f"{self.source}-analysis-5", "scope_type": "scene", "scope_id": "scene-1", "analysis_type": "pacing"}],
            "total": 1,
        }


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="local",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_writing_scope_service_routes_by_backend_and_policy():
    local = FakeWritingService("local")
    server = FakeWritingService("server")
    policy = FakePolicyEnforcer()
    scope = WritingScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    local_projects = await scope.list_projects(mode="local")
    server_project = await scope.create_project(mode="server", title="Novel", author="Ada")
    manuscript = await scope.create_manuscript(mode="server", project_id=server_project["id"], title="Book One")
    chapter = await scope.create_chapter(
        mode="server",
        project_id=server_project["id"],
        manuscript_id=manuscript["id"],
        title="Chapter 1",
    )
    scene = await scope.create_scene(
        mode="server",
        chapter_id=chapter["id"],
        title="Scene 1",
        content_markdown="Opening line.",
    )

    assert local_projects[0]["record_id"] == "local:writing_project:local-project-1"
    assert server_project["record_id"] == "server:writing_project:server-project-1"
    assert manuscript["record_id"] == "server:writing_manuscript:server-manuscript-1"
    assert chapter["record_id"] == "server:writing_chapter:server-chapter-1"
    assert scene["record_id"] == "server:writing_scene:server-scene-1"
    assert policy.calls == [
        "writing.projects.list.local",
        "writing.projects.create.server",
        "writing.manuscripts.create.server",
        "writing.chapters.create.server",
        "writing.scenes.create.server",
    ]


@pytest.mark.asyncio
async def test_writing_scope_service_denies_blocked_actions_before_dispatch():
    server = FakeWritingService("server")
    scope = WritingScopeService(
        local_service=FakeWritingService("local"),
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("wrong_source"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.create_project(mode="server", title="Novel")

    assert exc.value.reason_code == "wrong_source"
    assert server.calls == []


@pytest.mark.asyncio
async def test_writing_scope_service_routes_update_and_delete_crud_actions():
    server = FakeWritingService("server")
    policy = FakePolicyEnforcer()
    scope = WritingScopeService(
        local_service=FakeWritingService("local"),
        server_service=server,
        policy_enforcer=policy,
    )

    project = await scope.update_project(mode="server", project_id="server-project-1", title="Novel v2", expected_version=1)
    manuscript = await scope.update_manuscript(
        mode="server",
        manuscript_id="server-manuscript-1",
        title="Book v2",
        expected_version=1,
    )
    chapter = await scope.update_chapter(
        mode="server",
        chapter_id="server-chapter-1",
        title="Chapter v2",
        manuscript_id=None,
        expected_version=1,
    )
    scene = await scope.update_scene(
        mode="server",
        scene_id="server-scene-1",
        title="Scene v2",
        content_markdown="Revised",
        expected_version=1,
    )
    deleted_scene = await scope.delete_scene(mode="server", scene_id="server-scene-1", expected_version=2)

    assert project["version"] == 2
    assert manuscript["record_id"] == "server:writing_manuscript:server-manuscript-1"
    assert chapter["manuscript_id"] is None
    assert scene["content_markdown"] == "Revised"
    assert deleted_scene is True
    assert policy.calls == [
        "writing.projects.update.server",
        "writing.manuscripts.update.server",
        "writing.chapters.update.server",
        "writing.scenes.update.server",
        "writing.scenes.delete.server",
    ]


@pytest.mark.asyncio
async def test_writing_scope_service_routes_manual_version_actions():
    server = FakeWritingService("server")
    policy = FakePolicyEnforcer()
    scope = WritingScopeService(
        local_service=FakeWritingService("local"),
        server_service=server,
        policy_enforcer=policy,
    )

    version = await scope.create_version(
        mode="server",
        entity_type="scene",
        entity_id="scene-1",
        label="First draft",
    )
    versions = await scope.list_versions(mode="server", entity_type="scene", entity_id="scene-1")
    fetched = await scope.get_version(mode="server", entity_type="scene", entity_id="scene-1", version_number=1)
    restored = await scope.restore_version(
        mode="server",
        entity_type="scene",
        entity_id="scene-1",
        version_number=1,
        expected_version=1,
    )

    assert version["record_id"] == "server:writing_version:server-version-1"
    assert versions[0]["record_id"] == "server:writing_version:server-version-1"
    assert fetched["version_number"] == 1
    assert restored["record_id"] == "server:writing_scene:scene-1"
    assert server.calls == [
        ("create_version", "scene", "scene-1", "First draft"),
        ("list_versions", "scene", "scene-1"),
        ("get_version", "scene", "scene-1", 1),
        ("restore_version", "scene", "scene-1", 1, 1),
    ]
    assert policy.calls == [
        "writing.versions.create.server",
        "writing.versions.list.server",
        "writing.versions.detail.server",
        "writing.versions.restore.server",
    ]


@pytest.mark.asyncio
async def test_writing_scope_service_routes_trash_actions():
    server = FakeWritingService("server")
    policy = FakePolicyEnforcer()
    scope = WritingScopeService(
        local_service=FakeWritingService("local"),
        server_service=server,
        policy_enforcer=policy,
    )

    trash = await scope.list_trash(mode="server", entity_type="scene")
    restored = await scope.restore_trash(
        mode="server",
        entity_type="scene",
        entity_id="server-scene-1",
        expected_version=1,
    )

    assert trash[0]["record_id"] == "server:writing_scene:server-scene-1"
    assert restored["record_id"] == "server:writing_scene:server-scene-1"
    assert server.calls == [
        ("list_trash", "scene"),
        ("restore_trash", "scene", "server-scene-1", 1),
    ]
    assert policy.calls == [
        "writing.trash.list.server",
        "writing.trash.restore.server",
    ]


@pytest.mark.asyncio
async def test_writing_scope_service_routes_reorder_actions():
    server = FakeWritingService("server")
    policy = FakePolicyEnforcer()
    scope = WritingScopeService(
        local_service=FakeWritingService("local"),
        server_service=server,
        policy_enforcer=policy,
    )

    reordered = await scope.reorder_entities(
        mode="server",
        project_id="project-1",
        entity_type="chapters",
        items=[{"id": "chapter-1", "sort_order": 2.0, "version": 1, "new_parent_id": "manuscript-1"}],
    )

    assert reordered is True
    assert server.calls == [
        (
            "reorder_entities",
            "project-1",
            "chapters",
            [{"id": "chapter-1", "sort_order": 2.0, "version": 1, "new_parent_id": "manuscript-1"}],
        )
    ]
    assert policy.calls == ["writing.outline.reorder.server"]


@pytest.mark.asyncio
async def test_writing_scope_service_routes_structure_actions():
    server = FakeWritingService("server")
    policy = FakePolicyEnforcer()
    scope = WritingScopeService(
        local_service=FakeWritingService("local"),
        server_service=server,
        policy_enforcer=policy,
    )

    structure = await scope.get_structure(mode="server", project_id="server-project-1")

    assert structure["source"] == "server"
    assert structure["manuscripts"][0]["record_id"] == "server:writing_manuscript:server-manuscript-1"
    assert structure["unassigned_chapters"][0]["record_id"] == "server:writing_chapter:server-chapter-loose"
    assert structure["unassigned_chapters"][0]["outline_bucket"] == "unassigned_chapters"
    assert server.calls == [("get_structure", "server-project-1")]
    assert policy.calls == ["writing.projects.structure.server"]


@pytest.mark.asyncio
async def test_writing_scope_service_routes_auxiliary_server_actions():
    server = FakeWritingService("server")
    policy = FakePolicyEnforcer()
    scope = WritingScopeService(
        local_service=FakeWritingService("local"),
        server_service=server,
        policy_enforcer=policy,
    )

    character = await scope.create_character(mode="server", project_id="project-1", name="Ada")
    characters = await scope.list_characters(mode="server", project_id="project-1", role="protagonist")
    world = await scope.create_world_info(mode="server", project_id="project-1", kind="location", name="Capital")
    plot_line = await scope.create_plot_line(mode="server", project_id="project-1", title="Main Plot")
    plot_event = await scope.create_plot_event(mode="server", plot_line_id=plot_line["id"], title="Inciting Incident")
    plot_hole = await scope.create_plot_hole(mode="server", project_id="project-1", title="Continuity Issue")
    scene_characters = await scope.link_scene_character(mode="server", scene_id="scene-1", character_id=character["id"], is_pov=True)
    scene_world = await scope.link_scene_world_info(mode="server", scene_id="scene-1", world_info_id=world["id"])
    citation = await scope.create_citation(mode="server", scene_id="scene-1", source_type="manual")
    research = await scope.research_scene(mode="server", scene_id="scene-1", query="context")
    scene_analyses = await scope.analyze_scene(mode="server", scene_id="scene-1", analysis_types=["pacing"])
    chapter_analyses = await scope.analyze_chapter(mode="server", chapter_id="chapter-1", analysis_types=["continuity"])
    plot_hole_analyses = await scope.analyze_project_plot_holes(mode="server", project_id="project-1")
    consistency_analyses = await scope.analyze_project_consistency(mode="server", project_id="project-1")
    listed_analyses = await scope.list_analyses(mode="server", project_id="project-1", scope_type="scene")

    assert character["record_id"] == "server:writing_character:server-character-1"
    assert characters[0]["record_type"] == "writing_character"
    assert world["record_id"] == "server:writing_world_info:server-world-1"
    assert plot_line["record_id"] == "server:writing_plot_line:server-plot-line-1"
    assert plot_event["record_id"] == "server:writing_plot_event:server-plot-event-1"
    assert plot_hole["record_id"] == "server:writing_plot_hole:server-plot-hole-1"
    assert scene_characters[0]["record_id"] == "server:writing_scene_character_link:scene-1:server-character-1"
    assert scene_world[0]["record_id"] == "server:writing_scene_world_info_link:scene-1:server-world-1"
    assert citation["record_id"] == "server:writing_citation:server-citation-1"
    assert research["results"][0]["record_id"] == "server:writing_research_result:server-source-1"
    assert scene_analyses[0]["record_id"] == "server:writing_analysis:server-analysis-1"
    assert chapter_analyses[0]["record_id"] == "server:writing_analysis:server-analysis-2"
    assert plot_hole_analyses[0]["record_id"] == "server:writing_analysis:server-analysis-3"
    assert consistency_analyses[0]["record_id"] == "server:writing_analysis:server-analysis-4"
    assert listed_analyses["analyses"][0]["record_id"] == "server:writing_analysis:server-analysis-5"
    assert server.calls == [
        ("create_character", "project-1", {"name": "Ada"}),
        ("list_characters", "project-1", {"role": "protagonist", "cast_group": None}),
        ("create_world_info", "project-1", {"kind": "location", "name": "Capital"}),
        ("create_plot_line", "project-1", {"title": "Main Plot"}),
        ("create_plot_event", "server-plot-line-1", {"title": "Inciting Incident"}),
        ("create_plot_hole", "project-1", {"title": "Continuity Issue"}),
        ("link_scene_character", "scene-1", {"character_id": "server-character-1", "is_pov": True}),
        ("link_scene_world_info", "scene-1", {"world_info_id": "server-world-1"}),
        ("create_citation", "scene-1", {"source_type": "manual"}),
        ("research_scene", "scene-1", {"query": "context", "top_k": 5}),
        ("analyze_scene", "scene-1", {"analysis_types": ["pacing"], "provider": None, "model": None}),
        ("analyze_chapter", "chapter-1", {"analysis_types": ["continuity"], "provider": None, "model": None}),
        ("analyze_project_plot_holes", "project-1", {"analysis_types": None, "provider": None, "model": None}),
        ("analyze_project_consistency", "project-1", {"analysis_types": None, "provider": None, "model": None}),
        ("list_analyses", "project-1", {"scope_type": "scene", "analysis_type": None, "include_stale": False}),
    ]
    assert policy.calls == [
        "writing.characters.create.server",
        "writing.characters.list.server",
        "writing.world_info.create.server",
        "writing.plot_lines.create.server",
        "writing.plot_events.create.server",
        "writing.plot_holes.create.server",
        "writing.scene_characters.create.server",
        "writing.scene_world_info.create.server",
        "writing.citations.create.server",
        "writing.research.launch.server",
        "writing.analysis.launch.server",
        "writing.analysis.launch.server",
        "writing.analysis.launch.server",
        "writing.analysis.launch.server",
        "writing.analysis.list.server",
    ]


def test_writing_scope_service_reports_known_unsupported_server_capabilities():
    scope = WritingScopeService(
        local_service=FakeWritingService("local"),
        server_service=FakeWritingService("server"),
    )

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == []
    assert [item["operation_id"] for item in server_report] == [
        "writing.scenes.direct_manuscript_level.server",
    ]
    assert server_report[0]["reason_code"] == "server_contract_missing"
    assert server_report[0]["affected_action_ids"] == [
        "writing.scenes.create.server",
        "writing.scenes.list.server",
    ]
