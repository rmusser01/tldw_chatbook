"""Owner wiring for the Research Sources screen."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Research_Workspace import (
    QualifiedWorkspaceRef,
    ResearchWorkspaceController,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.layout_state import ResearchPanePreferences
from tldw_chatbook.Research_Workspace.overlay_store import (
    ResearchPresentationOverlayStore,
    ResearchSourceAnnotation,
    ResearchSourceFolder,
)
from tldw_chatbook.Research_Workspace.source_operations import (
    SourceOperationStatus,
)
from tldw_chatbook.UI.Research_Workspace_Modules import ResearchSourceIntakeRequest
from tldw_chatbook.UI.Screens.research_workspace_screen import (
    ResearchWorkspaceScreen,
)


class _RecordingOperationStore:
    def __init__(self, trace: list[tuple[object, ...]]) -> None:
        self.trace = trace
        self.operations = {}

    def create(self, operation):
        self.trace.append(("create", operation.operation_id, operation.workspace_id))
        self.operations[operation.operation_id] = operation
        return operation

    def advance_stage(self, operation_id, *, status, ingest_job_id="", **kwargs):
        self.trace.append(("advance", operation_id, status.value, ingest_job_id))
        operation = self.operations[operation_id]
        updated = replace(
            operation,
            catalog_status=status,
            ingest_job_id=ingest_job_id or operation.ingest_job_id,
            revision=operation.revision + 1,
            updated_at="2026-08-24T10:00:01Z",
        )
        self.operations[operation_id] = updated
        return updated


@pytest.mark.asyncio
async def test_url_intake_persists_one_qualified_operation_before_each_submit() -> None:
    trace: list[tuple[object, ...]] = []
    store = _RecordingOperationStore(trace)

    def submit(**kwargs):
        trace.append(
            (
                "submit",
                kwargs["research_source_operation_id"],
                kwargs["required_origin"],
                kwargs["source_path"],
            )
        )
        return SimpleNamespace(
            job_id=f"job-{len(trace)}", state=SimpleNamespace(value="queued")
        )

    app = SimpleNamespace(submit_library_ingest_job=submit)
    controller = ResearchWorkspaceController({})
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    controller.select_workspace(ref)
    screen = ResearchWorkspaceScreen(
        app,
        controller=controller,
        operation_store=store,
        operation_id_factory=iter(("operation-a", "operation-b")).__next__,
        now_factory=lambda: "2026-08-24T10:00:00Z",
    )

    await screen._submit_intake_request(
        ref,
        ResearchSourceIntakeRequest(
            "url",
            ("https://example.invalid/a", "https://example.invalid/b"),
        ),
    )

    assert [item[0] for item in trace] == [
        "create",
        "submit",
        "advance",
        "create",
        "submit",
        "advance",
    ]
    assert [item[3] for item in trace if item[0] == "submit"] == [
        "https://example.invalid/a",
        "https://example.invalid/b",
    ]
    assert all(item[2] == "local" for item in trace if item[0] == "submit")
    assert all(
        operation.catalog_status is SourceOperationStatus.IN_PROGRESS
        for operation in store.operations.values()
    )


@pytest.mark.asyncio
async def test_captured_server_ref_does_not_fall_back_after_navigation() -> None:
    trace: list[tuple[object, ...]] = []
    store = _RecordingOperationStore(trace)

    def submit(**kwargs):
        trace.append(("submit", kwargs["required_origin"]))
        return SimpleNamespace(
            job_id="job-server", state=SimpleNamespace(value="queued")
        )

    controller = ResearchWorkspaceController({})
    server_ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "server-workspace",
        server_profile_id="server-profile",
        principal_id="principal",
    )
    controller.select_workspace(server_ref)
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(submit_library_ingest_job=submit),
        controller=controller,
        operation_store=store,
        operation_id_factory=lambda: "operation-server",
        now_factory=lambda: "2026-08-24T10:00:00Z",
    )
    controller.select_workspace(
        QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "local-workspace")
    )

    await screen._submit_intake_request(
        server_ref,
        ResearchSourceIntakeRequest("url", ("https://example.invalid/paper",)),
    )

    assert ("submit", "server") in trace
    operation = store.operations["operation-server"]
    assert operation.workspace_id == "server-workspace"
    assert operation.server_profile_id == "server-profile"
    assert operation.principal_id == "principal"


@pytest.mark.asyncio
async def test_screen_overlay_keeps_device_source_data_qualified_across_switch(
    tmp_path,
) -> None:
    local_ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "local-workspace")
    server_ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "server-workspace",
        server_profile_id="profile",
        principal_id="principal",
    )
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(),
        overlay_store=ResearchPresentationOverlayStore(tmp_path / "overlay.json"),
    )
    screen._set_overlay_ref(local_ref)
    screen._source_folders = (
        ResearchSourceFolder("folder-local", "Evidence", ("source-local",)),
    )
    screen._source_annotations = (
        ResearchSourceAnnotation(
            "annotation-local",
            "source-local",
            "bounded quote",
            "device note",
            "2026-08-24T10:00:00Z",
            "2026-08-24T10:00:00Z",
        ),
    )
    await screen._save_overlay_preferences()

    screen._set_overlay_ref(server_ref)
    screen.pane_preferences = ResearchPanePreferences()
    screen._source_folders = ()
    screen._source_annotations = ()
    await screen._save_overlay_preferences()

    local = screen.overlay_store.load(local_ref)
    server = screen.overlay_store.load(server_ref)
    assert local is not None and server is not None
    assert [folder.folder_id for folder in local.source_folders] == ["folder-local"]
    assert [item.annotation_id for item in local.source_annotations] == [
        "annotation-local"
    ]
    assert server.source_folders == ()
    assert server.source_annotations == ()
