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
