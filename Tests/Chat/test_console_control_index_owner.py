"""The selected and App adapters retain one original lazy index owner."""

import asyncio
import sys

import pytest
from textual.app import App

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.project_index import runtime as index
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
from tldw_chatbook.config import get_workspaces_db_path

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("replacement", [None, "service", "registry"])
async def test_original_index_owner_is_cold_shared_and_never_readopts(replacement):
    database = WorkspaceDB(get_workspaces_db_path())
    app = App()
    registry = LocalWorkspaceRegistryService(database)
    app.workspace_registry_service = registry
    owner = index.get_project_index_owner(app)
    assert owner.service is None and not registry._project_index_listeners
    try:
        service = await asyncio.to_thread(owner.get)
        assert await asyncio.to_thread(index.get_project_index_service, app) is service
        assert owner.service is app._project_index_service is service
        assert service.binding_changed in registry._project_index_listeners
        if replacement == "service":
            app._project_index_service = object()
        elif replacement == "registry":
            app.workspace_registry_service = LocalWorkspaceRegistryService(database)
        if replacement is not None:
            with pytest.raises(RuntimeError, match="owner_changed"):
                await asyncio.to_thread(index.get_project_index_service, app)
            assert owner.service is service
        await asyncio.gather(index.close_project_index_service(app), owner.aclose())
        assert owner.closed and service._closed
        assert not registry._project_index_listeners
        service._factory.assert_retired()
        with pytest.raises(RuntimeError, match="owner_changed|closed"):
            await asyncio.to_thread(owner.get)
    finally:
        await owner.aclose()
        database.close()


async def test_subscription_failure_retains_unpublished_original_service_for_close():
    database = WorkspaceDB(get_workspaces_db_path())
    registry = LocalWorkspaceRegistryService(database)
    owner = index.ProjectIndexOwner(registry)
    captured = []

    def invoke():
        previous = sys.getprofile()

        def fail_subscription(frame, event, _result):
            if (
                event == "call"
                and frame.f_code
                is registry.subscribe_project_index_changes.__func__.__code__
            ):
                captured.append(frame.f_locals["listener"].__self__)
                raise OSError("task60 subscription failure")

        sys.setprofile(fail_subscription)
        try:
            owner.get()
        finally:
            sys.setprofile(previous)

    try:
        with pytest.raises(OSError, match="subscription failure"):
            await asyncio.to_thread(invoke)
        assert len(captured) == 1
        service = captured[0]
        assert owner.service is None and owner._unpublished == [service]
        assert not registry._project_index_listeners
        await owner.aclose()
        assert service._closed
        service._factory.assert_retired()
    finally:
        await owner.aclose()
        database.close()
