"""Real local packages, authored independently from the parser."""

import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def native_package(tmp_path):
    count = 0

    def create(*, requires=None, extension=None):
        nonlocal count
        count += 1
        root = tmp_path / f"package-{count}"
        shutil.copytree(Path(__file__).parent / "fixtures" / "native", root)
        manifest = json.loads((root / "plugin.json").read_text())
        ext = {"version": 1} if extension is None else extension
        if requires is not None:
            ext["requires"] = requires
        manifest["extensions"]["io.github.rmusser01.chatbook"] = ext
        (root / "plugin.json").write_text(json.dumps(manifest))
        return root

    return create


class PluginStack:
    """Exercise the real stack on its one persistent storage-worker event loop."""

    def __init__(self, root):
        import asyncio
        import threading
        from concurrent.futures import Future

        ready = Future()

        def worker():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                from tldw_chatbook.Plugins.authority_store import (
                    FilePluginMarkerStore,
                    PluginAuthorityStore,
                )
                from tldw_chatbook.Plugins.coordinator import PluginCoordinator
                from tldw_chatbook.Plugins.registry import PluginRegistry
                from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

                self.owner = PluginRuntimeOwner(root / "plugins")
                assert self.owner.try_acquire()
                self.registry = PluginRegistry(
                    self.owner.root / "registry.sqlite3", owner=self.owner
                )
                self.authority = PluginAuthorityStore(
                    root / "trust" / "plugins",
                    FilePluginMarkerStore(root / "marker"),
                    accept_reduced_protection=True,
                )
                self.coordinator = PluginCoordinator(
                    self.registry, self.authority, self.owner
                )
                ready.set_result(True)
                self.loop.run_forever()
            except BaseException as error:
                if not ready.done():
                    ready.set_exception(error)
                else:
                    raise
            finally:
                if hasattr(self, "registry"):
                    self.registry.close()
                if hasattr(self, "owner"):
                    self.owner.close()
                self.loop.close()

        self.thread = threading.Thread(target=worker, name="plugin-storage-test")
        self.thread.start()
        ready.result(timeout=15)
        self.call(lambda: self.coordinator.bootstrap("test passphrase"))

    def call(self, callback):
        import asyncio
        import inspect

        async def invoke():
            result = callback()
            return await result if inspect.isawaitable(result) else result

        return asyncio.run_coroutine_threadsafe(invoke(), self.loop).result(timeout=30)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=15)
        assert not self.thread.is_alive()


@pytest.fixture
def plugin_stack(tmp_path):
    stack = PluginStack(tmp_path / "stack")
    try:
        yield stack
    finally:
        stack.close()


@pytest.fixture
async def native_console(tmp_path, native_package):
    """Actual services and persisted Console; only provider transport is a double."""
    from types import SimpleNamespace

    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.console_provider_doubles import persisted_console_store
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.service import PluginService
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="plugin-tests")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="workspace-a", name="A")
    registry.create_workspace(workspace_id="workspace-b", name="B")
    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=registry.get_workspace,
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    await service.bootstrap("test passphrase")
    local = LocalSkillsService(
        store_dir=tmp_path / "skills",
        plugin_service=service,
        allow_untrusted_without_trust_service=True,
    )
    skills = SkillsScopeService(local_service=local)
    gateway = _RecordingGateway()
    store = persisted_console_store(workspace_registry=registry)
    session = store.create_session(workspace_id="workspace-a")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="m",
        skills_service=skills,
    )
    controller.app = SimpleNamespace(
        local_skills_service=local,
        skills_scope_service=skills,
        workspace_registry_service=registry,
    )

    async def install(*, metadata="", tools=None, enable=True):
        package = native_package()
        file = package / "skills/review/SKILL.md"
        additions = ("metadata:\n" + metadata) if metadata else ""
        if tools is not None:
            additions += "allowed-tools: " + json.dumps(tools) + "\n"
        file.write_text(
            file.read_text().replace(
                "description: Review a change against its requirements.\n",
                "description: Review a change against its requirements.\n" + additions,
            )
        )
        review = await service.review_install(
            package, selection=("skill:review",), workspace_id="workspace-a"
        )
        await service.commit(review, review.operation_id)
        trust = await service.review_trust(review.installation_id)
        await service.commit(trust, trust.operation_id)
        if enable:
            active = await service.review_activation(
                review.installation_id, workspace_id="workspace-a", intent="enabled"
            )
            await service.commit(active, active.operation_id)
        return review

    try:
        yield SimpleNamespace(
            service=service,
            local=local,
            skills=skills,
            controller=controller,
            store=store,
            session=session,
            gateway=gateway,
            registry=registry,
            install=install,
        )
    finally:
        await controller.shutdown()
        await service.aclose()
        store.persistence.db.close()
        registry.db.close()
