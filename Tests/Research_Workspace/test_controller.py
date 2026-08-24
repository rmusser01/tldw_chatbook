from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.Research_Workspace.contracts import (
    QualifiedWorkspaceRef,
    ResearchWorkspaceSummary,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.controller import ResearchWorkspaceController


class DeferredPort:
    def __init__(self) -> None:
        self.results: dict[QualifiedWorkspaceRef, asyncio.Future] = {}

    async def get_workspace(self, ref: QualifiedWorkspaceRef):
        future = asyncio.get_running_loop().create_future()
        self.results[ref] = future
        return await future


def local_ref(workspace_id: str) -> QualifiedWorkspaceRef:
    return QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, workspace_id)


def test_context_revision_increases_for_each_selection_and_capability_refresh() -> None:
    controller = ResearchWorkspaceController({})

    first = controller.select_workspace(local_ref("one"), capability_revision="a")
    second = controller.select_workspace(local_ref("two"), capability_revision="a")
    third = controller.set_capability_revision("b")

    assert (first, second, third) == (1, 2, 3)


def test_controller_rejects_result_for_a_different_captured_ref() -> None:
    controller = ResearchWorkspaceController({})
    ref = local_ref("one")
    controller.select_workspace(ref, capability_revision="a")
    capture = controller.capture_request()

    with pytest.raises(ValueError, match="mismatched workspace ref"):
        controller.accept_workspace_result(
            capture,
            ResearchWorkspaceSummary(ref=local_ref("two"), name="Wrong"),
        )


@pytest.mark.asyncio
async def test_stale_result_updates_owner_cache_but_not_visible_state() -> None:
    port = DeferredPort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    old_ref = local_ref("old")
    new_ref = local_ref("new")
    controller.select_workspace(old_ref, capability_revision="old-cap")

    old_request = asyncio.create_task(controller.refresh_selected_workspace())
    await asyncio.sleep(0)
    controller.select_workspace(new_ref, capability_revision="new-cap")
    new_request = asyncio.create_task(controller.refresh_selected_workspace())
    await asyncio.sleep(0)

    port.results[new_ref].set_result(
        ResearchWorkspaceSummary(ref=new_ref, name="New")
    )
    assert await new_request is True
    port.results[old_ref].set_result(
        ResearchWorkspaceSummary(ref=old_ref, name="Old")
    )
    assert await old_request is False

    assert controller.visible_workspace == ResearchWorkspaceSummary(
        ref=new_ref, name="New"
    )
    assert controller.canonical_workspace(old_ref) == ResearchWorkspaceSummary(
        ref=old_ref, name="Old"
    )
