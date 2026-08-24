"""Owner wiring for the Research Sources screen."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Button, Input, Label, Select, Static, TextArea

from tldw_chatbook.Research_Workspace import (
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
    ResearchCapability,
    ResearchSourcePage,
    ResearchSourceSummary,
    ResearchWorkspaceSummary,
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
    CanonicalItemType,
    ResearchSourceOperation,
    SourceOperationStage,
    SourceOperationStatus,
)
from tldw_chatbook.UI.Research_Workspace_Modules import (
    ResearchSourceIntakeRequest,
    ResearchSourcesRegion,
)
from tldw_chatbook.UI.Research_Workspace_Modules.source_inspector import (
    ResearchSourceInspectorModal,
)
from tldw_chatbook.UI.Research_Workspace_Modules.source_receipt import (
    ResearchSourceReceiptList,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.tldw_api.exceptions import TLDWAPIError
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
            error_stage=(
                kwargs.get("stage")
                if status is SourceOperationStatus.FAILED
                else operation.error_stage
            ),
            error_code=kwargs.get("error_code", operation.error_code),
            error_message=kwargs.get("error_message", operation.error_message),
            revision=operation.revision + 1,
            updated_at="2026-08-24T10:00:01Z",
        )
        self.operations[operation_id] = updated
        return updated


class _MountedScreenApp(App[None]):
    def __init__(self, screen: ResearchWorkspaceScreen) -> None:
        super().__init__()
        self._screen = screen
        self.copied_text = ""

    def copy_to_clipboard(self, text: str) -> None:
        self.copied_text = text

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


@pytest.mark.asyncio
async def test_owner_failure_preserves_qualified_receipts_loaded_first() -> None:
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    operation = ResearchSourceOperation(
        operation_id="operation-offline-receipt",
        idempotency_key="offline-receipt",
        data_source=WorkspaceDataSource.LOCAL,
        workspace_id=ref.workspace_id,
        canonical_item_type=CanonicalItemType.LOCAL_LIBRARY,
        desired_selected=True,
        created_at="2026-08-24T10:00:00Z",
        updated_at="2026-08-24T10:00:00Z",
    )

    class FailingPort:
        async def list_workspaces(self, *, include_archived=False):
            return (ResearchWorkspaceSummary(ref, "Offline workspace"),)

        async def capabilities(self, owner_ref):
            raise CapabilityUnavailableError(
                ResearchCapability(
                    False,
                    "owner_offline",
                    "Local Library is unavailable.",
                    "local",
                    recovery_action="Restore Library access and refresh.",
                )
            )

    store = SimpleNamespace(list_recent=lambda **kwargs: (operation,))
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: FailingPort()})
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(), controller=controller, operation_store=store
    )
    app = _MountedScreenApp(screen)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.1)

        receipt_text = " ".join(
            str(widget.render())
            for widget in screen.query("_ResearchSourceReceiptSlot Static")
            if widget.display
        )
        assert "operation-offline-receipt" in receipt_text
        assert "Local Library is unavailable" in str(
            screen.query_one("#research-source-recovery", Static).render()
        )


@pytest.mark.asyncio
async def test_failing_selection_worker_recovers_without_workerfailed_or_app_exit() -> None:
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    available = ResearchCapability(True, "available", "Available.", "local")
    denied = ResearchCapability(
        False,
        "selection_denied",
        "Selection changed on the owner and must be reloaded.",
        "local",
        recovery_action="Refresh Sources.",
    )

    class FailingSelectionPort:
        async def list_workspaces(self, *, include_archived=False):
            return (ResearchWorkspaceSummary(ref, "Workspace"),)

        async def capabilities(self, owner_ref):
            return {
                "attach_existing": available,
                "set_selected_scope": available,
                "preview_source": available,
                "remove_source": available,
            }

        async def list_sources(self, owner_ref, *, limit=100, offset=0):
            source = ResearchSourceSummary(
                ref=ref,
                source_id="membership-1",
                catalog_item_id="1",
                title="Evidence",
                source_type="text",
                selected=True,
            )
            return ResearchSourcePage(
                items=(source,),
                limit=limit,
                offset=offset,
                total=1,
                desired_source_ids=("1",),
            )

        async def get_readiness(self, owner_ref, *, source_ids=()):
            return ()

        async def set_selected_scope(self, owner_ref, source_ids):
            raise CapabilityUnavailableError(denied)

    controller = ResearchWorkspaceController(
        {WorkspaceDataSource.LOCAL: FailingSelectionPort()}
    )
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _MountedScreenApp(screen)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.1)
        screen.query_one("#research-source-selection-clear").press()
        await pilot.pause(0.1)

        assert app.screen is screen
        assert "Selection changed on the owner" in str(
            screen.query_one("#research-source-recovery", Static).render()
        )


@pytest.mark.asyncio
async def test_expected_intake_reorder_preview_remove_folder_and_retry_failures_stay_mounted() -> None:
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-actions")
    available = ResearchCapability(True, "available", "Available.", "local")
    denied = ResearchCapability(
        False,
        "owner_conflict",
        "The source owner rejected the action.",
        "local",
        recovery_action="Refresh Sources.",
    )

    class Port:
        deny_attach = False

        async def list_workspaces(self, *, include_archived=False):
            return (ResearchWorkspaceSummary(ref, "Workspace"),)

        async def capabilities(self, owner_ref):
            return {
                "attach_existing": denied if self.deny_attach else available,
                "set_selected_scope": available,
                "preview_source": available,
                "remove_source": available,
                "reorder_sources": available,
            }

        async def list_sources(self, owner_ref, *, limit=100, offset=0):
            items = tuple(
                ResearchSourceSummary(
                    ref=ref,
                    source_id=f"membership-{index}",
                    catalog_item_id=str(index),
                    title=f"Evidence {index}",
                    source_type="text",
                    selected=True,
                    position=index - 1,
                )
                for index in (1, 2)
            )
            return ResearchSourcePage(
                items=items,
                limit=limit,
                offset=offset,
                total=2,
                desired_source_ids=("1", "2"),
            )

        async def get_readiness(self, owner_ref, *, source_ids=()):
            return ()

        async def reorder_sources(self, owner_ref, ordered_source_ids):
            raise CapabilityUnavailableError(denied)

        async def preview_source(self, owner_ref, source_id, **kwargs):
            raise CapabilityUnavailableError(denied)

        async def remove_source(self, owner_ref, source_id, *, expected_version=None):
            raise CapabilityUnavailableError(denied)

    class Scheduler:
        async def retry(self, operation_id, *, stage):
            raise TLDWAPIError("private transport detail")

    class FailingOverlay:
        def load(self, owner_ref):
            return None

        def save(self, *args, **kwargs):
            raise OSError("private device path")

    port = Port()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(),
        controller=controller,
        association_scheduler=Scheduler(),
        overlay_store=FailingOverlay(),
    )
    app = _MountedScreenApp(screen)

    async with app.run_test(size=(120, 38)) as pilot:
        await pilot.pause(0.1)

        port.deny_attach = True
        quick = screen.query_one("#research-source-quick-url", Input)
        quick.value = "https://example.invalid/source"
        screen.query_one("#research-source-quick-submit", Button).press()
        await pilot.pause()
        assert app.screen is screen

        screen.query_one("#research-source-row-down-0", Button).press()
        await pilot.pause()
        assert app.screen is screen

        screen.query_one("#research-source-row-preview-0", Button).press()
        await pilot.pause()
        assert app.screen is screen

        screen.query_one("#research-source-row-remove-0", Button).press()
        await pilot.pause()
        app.screen.query_one("#confirm-button", Button).press()
        await pilot.pause()
        assert app.screen is screen

        folder_name = screen.query_one("#research-source-folder-name", Input)
        folder_name.value = "Device folder"
        screen.query_one("#research-source-folder-new", Button).press()
        await pilot.pause()
        assert app.screen is screen

        screen.retry_source_stage(
            ResearchSourceReceiptList.RetryRequested(
                "operation-1", SourceOperationStage.CATALOG
            )
        )
        await pilot.pause()
        assert app.screen is screen
        assert "Source action could not be completed" in str(
            screen.query_one("#research-source-recovery", Static).render()
        )


@pytest.mark.asyncio
async def test_remove_association_requires_confirmation_and_escape_is_non_mutating() -> None:
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    available = ResearchCapability(True, "available", "Available.", "local")
    removed: list[str] = []

    class Port:
        async def list_workspaces(self, *, include_archived=False):
            return (ResearchWorkspaceSummary(ref, "Workspace"),)

        async def capabilities(self, owner_ref):
            return {
                "attach_existing": available,
                "set_selected_scope": available,
                "preview_source": available,
                "remove_source": available,
            }

        async def list_sources(self, owner_ref, *, limit=100, offset=0):
            return ResearchSourcePage(
                items=(
                    ResearchSourceSummary(
                        ref=ref,
                        source_id="membership-1",
                        catalog_item_id="1",
                        title="Evidence",
                        source_type="text",
                        selected=True,
                    ),
                ),
                limit=limit,
                offset=offset,
                total=1,
                desired_source_ids=("1",),
            )

        async def get_readiness(self, owner_ref, *, source_ids=()):
            return ()

        async def remove_source(self, owner_ref, source_id, *, expected_version=None):
            removed.append(source_id)
            return True

    screen = ResearchWorkspaceScreen(
        SimpleNamespace(),
        controller=ResearchWorkspaceController({WorkspaceDataSource.LOCAL: Port()}),
    )
    app = _MountedScreenApp(screen)
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause(0.1)
        screen.query_one("#research-source-row-remove-0", Button).press()
        await pilot.pause()

        dialog = app.screen
        assert isinstance(dialog, ConfirmationDialog)
        assert "removes from this workspace" in str(dialog.query_one(Label).render())
        assert "Library/Media item is retained" in str(dialog.query_one(Label).render())
        assert removed == []

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen
        assert removed == []

        screen.batch_source_action(
            ResearchSourcesRegion.BatchRequested("remove-selected")
        )
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationDialog)
        assert "removes from this workspace" in str(
            app.screen.query_one(Label).render()
        )
        assert "Library/Media item is retained" in str(
            app.screen.query_one(Label).render()
        )
        assert removed == []
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen
        assert removed == []

        screen.query_one("#research-source-row-remove-0", Button).press()
        await pilot.pause()
        app.screen.query_one("#confirm-button", Button).press()
        await pilot.pause(0.1)

        assert removed == ["membership-1"]


@pytest.mark.asyncio
async def test_intake_capability_denial_happens_before_operation_or_catalog_write() -> None:
    unavailable = ResearchCapability(
        False,
        "viewer_forbidden",
        "Viewers cannot add sources.",
        "server",
        recovery_action="Ask an owner or editor for access.",
    )

    class Port:
        async def capabilities(self, ref):
            return {"attach_existing": unavailable}

    trace: list[tuple[object, ...]] = []
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: Port()})
    controller.select_workspace(ref)
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(
            submit_library_ingest_job=lambda **kwargs: trace.append(("submit", kwargs))
        ),
        controller=controller,
        operation_store=_RecordingOperationStore(trace),
    )

    with pytest.raises(CapabilityUnavailableError):
        await screen._submit_intake_request(
            ref,
            ResearchSourceIntakeRequest(
                "url", ("https://example.invalid/paper",)
            ),
        )

    assert trace == []


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
    class Port:
        async def capabilities(self, ref):
            return {
                "attach_existing": ResearchCapability(
                    True, "available", "Available.", "local"
                )
            }

    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: Port()})
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
async def test_paste_staging_is_bound_after_operation_create_and_cleaned_if_submit_fails() -> None:
    trace: list[tuple[object, ...]] = []
    store = _RecordingOperationStore(trace)

    class Staging:
        def stage(self, operation_id, *, title, body):
            trace.append(("stage", operation_id, title, body))
            return f"/private/staging/{operation_id}.txt"

        def delete(self, operation_id):
            trace.append(("delete", operation_id))
            return True

    class Port:
        async def capabilities(self, ref):
            return {
                "attach_existing": ResearchCapability(
                    True, "available", "Available.", "local"
                )
            }

    def submit(**kwargs):
        trace.append(("submit", kwargs["research_source_operation_id"]))
        raise RuntimeError("submit failed")

    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: Port()})
    controller.select_workspace(ref)
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(submit_library_ingest_job=submit),
        controller=controller,
        operation_store=store,
        paste_staging_store=Staging(),
        operation_id_factory=lambda: "operation-paste",
        now_factory=lambda: "2026-08-24T10:00:00Z",
    )

    await screen._submit_intake_request(
        ref,
        ResearchSourceIntakeRequest("paste", ("PRIVATE BODY",), title="Paste"),
    )

    assert [item[0] for item in trace] == [
        "create",
        "stage",
        "submit",
        "advance",
        "delete",
    ]
    assert trace[1][1] == trace[0][1] == "operation-paste"


@pytest.mark.asyncio
async def test_captured_server_ref_does_not_fall_back_after_navigation() -> None:
    trace: list[tuple[object, ...]] = []
    store = _RecordingOperationStore(trace)

    def submit(**kwargs):
        trace.append(("submit", kwargs["required_origin"]))
        return SimpleNamespace(
            job_id="job-server", state=SimpleNamespace(value="queued")
        )

    class Port:
        async def capabilities(self, ref):
            return {
                "attach_existing": ResearchCapability(
                    True, "available", "Available.", "server"
                )
            }

    controller = ResearchWorkspaceController({WorkspaceDataSource.SERVER: Port()})
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


@pytest.mark.asyncio
async def test_annotation_edit_reopens_and_survives_overlay_store_restart(tmp_path) -> None:
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-annotation")
    available = ResearchCapability(True, "available", "Available.", "local")

    class Port:
        async def list_workspaces(self, *, include_archived=False):
            return (ResearchWorkspaceSummary(ref, "Workspace"),)

        async def capabilities(self, owner_ref):
            return {
                "attach_existing": available,
                "set_selected_scope": available,
                "preview_source": available,
                "remove_source": available,
            }

        async def list_sources(self, owner_ref, *, limit=100, offset=0):
            return ResearchSourcePage(
                items=(
                    ResearchSourceSummary(
                        ref=ref,
                        source_id="membership-1",
                        catalog_item_id="1",
                        title="Evidence",
                        source_type="text",
                    ),
                ),
                limit=limit,
                offset=offset,
                total=1,
            )

        async def get_readiness(self, owner_ref, *, source_ids=()):
            return ()

    path = tmp_path / "research" / "overlay.json"
    store = ResearchPresentationOverlayStore(path)
    store.save(
        ref,
        ResearchPanePreferences(),
        expected_revision=0,
        source_annotations=(
            ResearchSourceAnnotation(
                "annotation-stable",
                "membership-1",
                "Quote",
                "Original note",
                "2026-08-24T10:00:00Z",
                "2026-08-24T10:00:00Z",
            ),
        ),
    )
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(),
        controller=ResearchWorkspaceController({WorkspaceDataSource.LOCAL: Port()}),
        overlay_store=store,
        now_factory=lambda: "2026-08-24T10:01:00Z",
    )
    app = _MountedScreenApp(screen)

    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause(0.2)
        screen.query_one("#research-source-row-details-0", Button).press()
        await pilot.pause()
        inspector = app.screen
        assert isinstance(inspector, ResearchSourceInspectorModal)
        inspector.query_one("#research-source-annotation-list", Select).value = (
            "annotation-stable"
        )
        await pilot.pause()
        inspector.query_one("#research-source-annotation-note", TextArea).text = (
            "Edited note"
        )
        inspector.query_one("#research-source-annotation-save", Button).press()
        await pilot.pause(0.2)

        screen.query_one("#research-source-row-details-0", Button).press()
        await pilot.pause()
        reopened = app.screen
        assert isinstance(reopened, ResearchSourceInspectorModal)
        reopened.query_one("#research-source-annotation-list", Select).value = (
            "annotation-stable"
        )
        await pilot.pause()
        assert reopened.query_one("#research-source-annotation-note", TextArea).text == (
            "Edited note"
        )

    restarted = ResearchPresentationOverlayStore(path).load(ref)
    assert restarted is not None
    assert restarted.source_annotations[0].annotation_id == "annotation-stable"
    assert restarted.source_annotations[0].note == "Edited note"


@pytest.mark.asyncio
async def test_overlay_conflict_retains_draft_and_offers_private_recovery_actions(
    tmp_path,
) -> None:
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-conflict")
    path = tmp_path / "research" / "overlay.json"
    first_store = ResearchPresentationOverlayStore(path)
    initial = first_store.save(
        ref, ResearchPanePreferences(), expected_revision=0
    )
    screen = ResearchWorkspaceScreen(SimpleNamespace(), overlay_store=first_store)
    app = _MountedScreenApp(screen)

    async with app.run_test(size=(110, 32)) as pilot:
        await pilot.pause()
        screen._set_overlay_ref(ref)
        screen._overlay_revision = initial.revision
        screen._source_folders = (
            ResearchSourceFolder("folder-local", "Local draft", ("source-private",)),
        )
        screen._source_annotations = (
            ResearchSourceAnnotation(
                "annotation-local",
                "source-private",
                "PRIVATE QUOTE BODY",
                "PRIVATE NOTE BODY",
                "2026-08-24T10:00:00Z",
                "2026-08-24T10:00:00Z",
            ),
        )
        second_store = ResearchPresentationOverlayStore(path)
        second_store.save(
            ref,
            ResearchPanePreferences(sources_open=False),
            expected_revision=initial.revision,
            source_folders=(ResearchSourceFolder("folder-remote", "Remote layout"),),
        )
        await screen._save_overlay_preferences()
        await pilot.pause()

        dialog = app.screen
        assert dialog.__class__.__name__ == "ResearchOverlayConflictModal"
        for suffix in ("reload", "export", "fork"):
            assert dialog.query_one(f"#research-overlay-conflict-{suffix}", Button)
        assert [folder.folder_id for folder in screen._source_folders] == [
            "folder-local"
        ]
        painted = " ".join(str(widget.render()) for widget in dialog.query(Static))
        assert "PRIVATE QUOTE BODY" not in painted
        assert str(path) not in painted

        dialog.query_one("#research-overlay-conflict-export", Button).press()
        await pilot.pause()

        assert app.screen is screen
        assert app.copied_text
        assert "PRIVATE QUOTE BODY" not in app.copied_text
        assert "PRIVATE NOTE BODY" not in app.copied_text
        assert str(path) not in app.copied_text
        assert "http://" not in app.copied_text
        assert "https://" not in app.copied_text
        assert [folder.folder_id for folder in screen._source_folders] == [
            "folder-local"
        ]

        await screen._save_overlay_preferences()
        await pilot.pause()
        app.screen.query_one("#research-overlay-conflict-fork", Button).press()
        await pilot.pause()
        assert screen._overlay_fork_draft is not None
        assert [folder.folder_id for folder in screen._source_folders] == [
            "folder-local"
        ]

        await screen._save_overlay_preferences()
        await pilot.pause()
        app.screen.query_one("#research-overlay-conflict-reload", Button).press()
        await pilot.pause(0.1)
        assert [folder.folder_id for folder in screen._source_folders] == [
            "folder-remote"
        ]
