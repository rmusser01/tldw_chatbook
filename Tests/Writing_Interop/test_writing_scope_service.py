import pytest

from tldw_chatbook.Writing_Interop.server_writing_service import (
    REASON_DIRECT_MANUSCRIPT_SCENE,
    REASON_SCENE_REPARENT,
    REASON_TRASH_RESTORE,
    REASON_VERSION_HISTORY,
    WritingCapabilityError,
)
from tldw_chatbook.Writing_Interop.writing_scope_service import (
    WritingBackend,
    WritingScopeService,
)


class FakeBackend:
    def __init__(self, label: str):
        self.label = label
        self.calls = []

    def list_projects(self, **kwargs):
        self.calls.append(("list_projects", kwargs))
        return [f"{self.label}:project"]

    async def create_project(self, **kwargs):
        self.calls.append(("create_project", kwargs))
        return {**kwargs, "source": self.label}

    async def update_manuscript(self, manuscript_id, update_data=None, expected_version=None, **kwargs):
        self.calls.append(
            ("update_manuscript", manuscript_id, update_data, expected_version, kwargs)
        )
        return {"id": manuscript_id, "source": self.label}

    async def delete_chapter(self, chapter_id, *, expected_version=None):
        self.calls.append(("delete_chapter", chapter_id, expected_version))
        return {"id": chapter_id, "source": self.label, "deleted": True}

    async def create_scene(self, project_id, **kwargs):
        self.calls.append(("create_scene", project_id, kwargs))
        return {"project_id": project_id, "source": self.label, **kwargs}

    async def move_scene(self, scene_id, manuscript_id, chapter_id, **kwargs):
        self.calls.append(("move_scene", scene_id, manuscript_id, chapter_id, kwargs))
        return {"id": scene_id, "source": self.label}

    async def create_version(self, entity_kind, entity_id, **kwargs):
        self.calls.append(("create_version", entity_kind, entity_id, kwargs))
        return {"entity_kind": entity_kind, "entity_id": entity_id, "source": self.label}

    async def list_versions(self, entity_kind, entity_id, **kwargs):
        self.calls.append(("list_versions", entity_kind, entity_id, kwargs))
        return [{"entity_kind": entity_kind, "entity_id": entity_id, "source": self.label}]

    async def get_version(self, version_id, **kwargs):
        self.calls.append(("get_version", version_id, kwargs))
        return {"id": version_id, "source": self.label}

    async def restore_version_to_working_state(self, version_id, **kwargs):
        self.calls.append(("restore_version_to_working_state", version_id, kwargs))
        return {"id": version_id, "source": self.label}

    async def restore_project(self, project_id, **kwargs):
        self.calls.append(("restore_project", project_id, kwargs))
        return {"id": project_id, "source": self.label}

    async def restore_scene(self, scene_id, **kwargs):
        self.calls.append(("restore_scene", scene_id, kwargs))
        return {"id": scene_id, "source": self.label}

    async def list_trash(self, project_id=None, **kwargs):
        self.calls.append(("list_trash", project_id, kwargs))
        return [{"project_id": project_id, "source": self.label}]


class FakePolicy:
    def __init__(self):
        self.calls = []

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)


@pytest.mark.asyncio
async def test_default_mode_is_local():
    local = FakeBackend("local")
    server = FakeBackend("server")
    service = WritingScopeService(local_service=local, server_service=server)

    result = await service.list_projects()

    assert result == ["local:project"]
    assert local.calls == [("list_projects", {})]
    assert server.calls == []


@pytest.mark.asyncio
async def test_invalid_mode_fails():
    service = WritingScopeService(
        local_service=FakeBackend("local"),
        server_service=FakeBackend("server"),
    )

    with pytest.raises(ValueError, match="Invalid writing backend"):
        await service.list_projects(mode="remote")


@pytest.mark.asyncio
async def test_local_routes_only_to_local_backend():
    local = FakeBackend("local")
    server = FakeBackend("server")
    service = WritingScopeService(local_service=local, server_service=server)

    result = await service.create_project(mode=WritingBackend.LOCAL, title="Local Project")

    assert result == {"source": "local", "title": "Local Project"}
    assert local.calls == [("create_project", {"title": "Local Project"})]
    assert server.calls == []


@pytest.mark.asyncio
async def test_server_routes_only_to_server_backend():
    local = FakeBackend("local")
    server = FakeBackend("server")
    service = WritingScopeService(local_service=local, server_service=server)

    result = await service.create_project(mode="server", title="Server Project")

    assert result == {"source": "server", "title": "Server Project"}
    assert server.calls == [("create_project", {"title": "Server Project"})]
    assert local.calls == []


@pytest.mark.asyncio
async def test_server_unavailable_fails_visibly():
    service = WritingScopeService(local_service=FakeBackend("local"), server_service=None)

    with pytest.raises(ValueError, match="Server writing backend is unavailable"):
        await service.list_projects(mode="server")


@pytest.mark.asyncio
async def test_server_mode_does_not_fallback_to_local_when_server_missing():
    local = FakeBackend("local")
    service = WritingScopeService(local_service=local, server_service=None)

    with pytest.raises(ValueError, match="Server writing backend is unavailable"):
        await service.list_projects(mode="server")

    assert local.calls == []


def test_capability_helper_reports_server_direct_scene_unsupported():
    service = WritingScopeService(local_service=object(), server_service=object())

    capability = service.get_capability(
        mode="server",
        action="create",
        entity_kind="scene",
        parent_kind="manuscript",
    )

    assert capability.source == "server"
    assert capability.name == "server_direct_manuscript_scene"
    assert capability.supported is False
    assert capability.reason == REASON_DIRECT_MANUSCRIPT_SCENE
    assert capability.metadata == {
        "action": "create",
        "entity_kind": "scene",
        "parent_kind": "manuscript",
    }


def test_capability_helper_reports_server_manual_versions_unsupported():
    service = WritingScopeService(local_service=object(), server_service=object())

    capability = service.get_capability(
        mode="server",
        action="create_version",
        entity_kind="scene",
    )

    assert capability.source == "server"
    assert capability.name == "server_version_history"
    assert capability.supported is False
    assert capability.reason == REASON_VERSION_HISTORY


def test_capability_helper_reports_server_trash_restore_unsupported():
    service = WritingScopeService(local_service=object(), server_service=object())

    capability = service.get_capability(
        mode="server",
        action="restore_deleted",
        entity_kind="scene",
    )

    assert capability.source == "server"
    assert capability.name == "server_trash_restore"
    assert capability.supported is False
    assert capability.reason == REASON_TRASH_RESTORE


def test_capability_helper_reports_server_scene_reparent_unsupported():
    service = WritingScopeService(local_service=object(), server_service=object())

    capability = service.get_capability(
        mode="server",
        action="reparent",
        entity_kind="scene",
    )

    assert capability.source == "server"
    assert capability.name == "server_scene_reparent"
    assert capability.supported is False
    assert capability.reason == REASON_SCENE_REPARENT


@pytest.mark.asyncio
async def test_server_direct_scene_create_is_blocked_before_backend_call():
    server = FakeBackend("server")
    service = WritingScopeService(local_service=FakeBackend("local"), server_service=server)

    with pytest.raises(WritingCapabilityError) as exc_info:
        await service.create_scene(
            "project-1",
            mode="server",
            title="Direct Scene",
            manuscript_id="manuscript-1",
            chapter_id=None,
        )

    assert exc_info.value.reason == REASON_DIRECT_MANUSCRIPT_SCENE
    assert server.calls == []


@pytest.mark.asyncio
async def test_server_manual_version_is_blocked_before_backend_call():
    server = FakeBackend("server")
    service = WritingScopeService(local_service=FakeBackend("local"), server_service=server)

    with pytest.raises(WritingCapabilityError) as exc_info:
        await service.create_version("scene", "scene-1", mode="server")

    assert exc_info.value.reason == REASON_VERSION_HISTORY
    assert server.calls == []


@pytest.mark.asyncio
async def test_server_trash_restore_is_blocked_before_backend_call():
    server = FakeBackend("server")
    service = WritingScopeService(local_service=FakeBackend("local"), server_service=server)

    with pytest.raises(WritingCapabilityError) as exc_info:
        await service.restore_scene("scene-1", mode="server", expected_version=2)

    assert exc_info.value.reason == REASON_TRASH_RESTORE
    assert server.calls == []


@pytest.mark.asyncio
async def test_server_scene_reparent_is_blocked_before_backend_call():
    server = FakeBackend("server")
    service = WritingScopeService(local_service=FakeBackend("local"), server_service=server)

    with pytest.raises(WritingCapabilityError) as exc_info:
        await service.move_scene(
            "scene-1",
            "manuscript-1",
            "chapter-2",
            mode="server",
            expected_version=2,
        )

    assert exc_info.value.reason == REASON_SCENE_REPARENT
    assert server.calls == []


@pytest.mark.asyncio
async def test_move_scene_local_server_mode_uses_server_reparent_gate():
    server = FakeBackend("server")
    service = WritingScopeService(local_service=FakeBackend("local"), server_service=server)

    with pytest.raises(WritingCapabilityError) as exc_info:
        await service.move_scene_local(
            "scene-1",
            None,
            "chapter-2",
            mode="server",
            expected_version=2,
        )

    assert exc_info.value.reason == REASON_SCENE_REPARENT
    assert server.calls == []


@pytest.mark.asyncio
async def test_policy_action_ids_are_checked_for_top_level_crud_actions():
    local = FakeBackend("local")
    server = FakeBackend("server")
    policy = FakePolicy()
    service = WritingScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    await service.list_projects(mode="local")
    await service.list_projects(mode="server")
    await service.create_project(mode="local", title="Local")
    await service.create_project(mode="server", title="Server")
    await service.update_manuscript(
        "manuscript-1",
        {"title": "Local"},
        mode="local",
        expected_version=1,
    )
    await service.update_manuscript(
        "manuscript-2",
        {"title": "Server"},
        mode="server",
        expected_version=1,
    )
    await service.delete_chapter("chapter-1", mode="local", expected_version=1)
    await service.delete_chapter("chapter-2", mode="server", expected_version=1)
    await service.create_scene("project-1", mode="local", title="Local", chapter_id="chapter-1")
    await service.create_scene("project-1", mode="server", title="Server", chapter_id="chapter-2")

    assert policy.calls == [
        "writing.projects.list.local",
        "writing.projects.list.server",
        "writing.projects.create.local",
        "writing.projects.create.server",
        "writing.manuscripts.update.local",
        "writing.manuscripts.update.server",
        "writing.chapters.delete.local",
        "writing.chapters.delete.server",
        "writing.scenes.create.local",
        "writing.scenes.create.server",
    ]


@pytest.mark.asyncio
async def test_policy_action_ids_cover_restore_version_and_trash_actions():
    local = FakeBackend("local")
    policy = FakePolicy()
    service = WritingScopeService(
        local_service=local,
        server_service=FakeBackend("server"),
        policy_enforcer=policy,
    )

    await service.restore_project("project-1", mode="local", expected_version=1)
    await service.restore_scene("scene-1", mode="local", expected_version=1)
    await service.create_version("scene", "scene-1", mode="local")
    await service.list_versions("scene", "scene-1", mode="local")
    await service.get_version("version-1", mode="local", entity_kind="scene")
    await service.restore_version_to_working_state(
        "version-1",
        mode="local",
        entity_kind="scene",
        expected_version=2,
    )
    await service.list_trash("project-1", mode="local")

    assert policy.calls == [
        "writing.projects.update.local",
        "writing.scenes.update.local",
        "writing.scenes.update.local",
        "writing.scenes.detail.local",
        "writing.scenes.detail.local",
        "writing.scenes.update.local",
        "writing.projects.list.local",
    ]
