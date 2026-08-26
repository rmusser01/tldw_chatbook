"""Owner wiring for the Research Sources screen."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.app import App
from textual.widgets import Button, Input, Label, Select, Static, TextArea

from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJobRegistry,
)
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

    def get(self, operation_id):
        return self.operations.get(operation_id)

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
async def test_failing_selection_worker_recovers_without_workerfailed_or_app_exit() -> (
    None
):
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
async def test_expected_intake_reorder_preview_remove_folder_and_retry_failures_stay_mounted() -> (
    None
):
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
async def test_remove_association_requires_confirmation_and_escape_is_non_mutating() -> (
    None
):
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
async def test_filtered_preview_targets_the_one_displayed_selected_association(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global desired selection cannot disable or retarget filtered preview."""

    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-preview")
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
                items=tuple(
                    ResearchSourceSummary(
                        ref=ref,
                        source_id=f"membership-{name.casefold()}",
                        catalog_item_id=str(index),
                        title=f"{name} evidence",
                        source_type="text",
                        selected=True,
                    )
                    for index, name in ((1, "Alpha"), (2, "Beta"))
                ),
                limit=limit,
                offset=offset,
                total=2,
                desired_source_ids=("1", "2"),
            )

        async def get_readiness(self, owner_ref, *, source_ids=()):
            return ()

    screen = ResearchWorkspaceScreen(
        SimpleNamespace(),
        controller=ResearchWorkspaceController({WorkspaceDataSource.LOCAL: Port()}),
    )
    show = AsyncMock()
    monkeypatch.setattr(screen, "_show_source_inspector", show)
    app = _MountedScreenApp(screen)

    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause(0.1)
        screen.query_one("#research-source-search", Input).value = "alpha"
        await pilot.pause()
        preview = screen.query_one("#research-source-preview-selected", Button)
        assert not preview.disabled

        preview.press()
        await pilot.pause()

    show.assert_awaited_once_with("membership-alpha", load_preview=True)


@pytest.mark.asyncio
async def test_intake_capability_denial_happens_before_operation_or_catalog_write() -> (
    None
):
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
            ResearchSourceIntakeRequest("url", ("https://example.invalid/paper",)),
        )

    assert trace == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate",
    [
        "https://user:PRIVATE@example.invalid/paper",
        "file:///private/research.txt",
        "relative/private.txt",
        "https://example.invalid/control\x00value",
        "https://example.invalid/format\u200bvalue",
    ],
)
async def test_direct_url_intake_rejects_before_operation_job_or_server_write(
    candidate: str,
) -> None:
    """The owner boundary protects Quick URL and direct callers, not only the modal."""

    trace: list[tuple[object, ...]] = []

    class Port:
        async def capabilities(self, ref):
            return {
                "attach_existing": ResearchCapability(
                    True, "available", "Available.", "server"
                )
            }

    ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "workspace-server",
        server_profile_id="profile",
        principal_id="principal",
    )
    controller = ResearchWorkspaceController({WorkspaceDataSource.SERVER: Port()})
    controller.select_workspace(ref)
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(
            prepare_research_source_ingest_job=lambda **kwargs: trace.append(
                ("prepare", kwargs)
            ),
            _dispatch_research_source_catalog_job=lambda job_id: trace.append(
                ("dispatch", job_id)
            ),
        ),
        controller=controller,
        operation_store=_RecordingOperationStore(trace),
    )

    with pytest.raises(ValueError, match="valid HTTP or HTTPS URL"):
        await screen._submit_intake_request(
            ref,
            ResearchSourceIntakeRequest("url", (candidate,)),
        )

    assert trace == []


@pytest.mark.asyncio
async def test_url_intake_persists_one_qualified_operation_before_each_submit() -> None:
    trace: list[tuple[object, ...]] = []
    store = _RecordingOperationStore(trace)

    def prepare(**kwargs):
        trace.append(
            (
                "prepare",
                kwargs["research_source_operation_id"],
                kwargs["required_origin"],
                kwargs["source_path"],
            )
        )
        return SimpleNamespace(
            job_id=f"job-{len(trace)}", state=SimpleNamespace(value="queued")
        )

    def dispatch(job_id):
        linked = tuple(
            operation
            for operation in store.operations.values()
            if operation.ingest_job_id == job_id
            and operation.catalog_status is SourceOperationStatus.IN_PROGRESS
        )
        assert len(linked) == 1
        trace.append(("dispatch", job_id, linked[0].operation_id))

    app = SimpleNamespace(
        prepare_research_source_ingest_job=prepare,
        _dispatch_research_source_catalog_job=dispatch,
    )

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
        "prepare",
        "advance",
        "dispatch",
        "create",
        "prepare",
        "advance",
        "dispatch",
    ]
    assert [item[3] for item in trace if item[0] == "prepare"] == [
        "https://example.invalid/a",
        "https://example.invalid/b",
    ]
    assert all(item[2] == "local" for item in trace if item[0] == "prepare")
    assert all(
        operation.catalog_status is SourceOperationStatus.IN_PROGRESS
        for operation in store.operations.values()
    )


@pytest.mark.asyncio
async def test_transient_link_failure_retains_held_job_and_managed_paste() -> None:
    trace: list[tuple[object, ...]] = []

    class FailingLinkStore(_RecordingOperationStore):
        def advance_stage(self, operation_id, *, status, ingest_job_id="", **kwargs):
            if status is SourceOperationStatus.IN_PROGRESS:
                trace.append(("link-failed", operation_id, ingest_job_id))
                raise RuntimeError("operation store unavailable")
            return super().advance_stage(
                operation_id,
                status=status,
                ingest_job_id=ingest_job_id,
                **kwargs,
            )

    class Staging:
        def stage(self, operation_id, *, title, body):
            trace.append(("stage", operation_id, title, body))
            return f"/private/staging/{operation_id}.txt"

        def delete(self, operation_id):
            trace.append(("delete", operation_id))
            return True

    class IntakeApp:
        def prepare_research_source_ingest_job(self, **kwargs):
            trace.append(("prepare", kwargs["research_source_operation_id"]))
            return SimpleNamespace(job_id="job-prepared")

        def _cancel_research_source_prepared_job(self, job_id):
            trace.append(("cancel", job_id))

        def _dispatch_research_source_catalog_job(self, job_id):
            trace.append(("dispatch", job_id))

    class Port:
        async def capabilities(self, ref):
            return {
                "attach_existing": ResearchCapability(
                    True, "available", "Available.", "local"
                )
            }

    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: Port()})
    controller.select_workspace(ref)
    screen = ResearchWorkspaceScreen(
        IntakeApp(),
        controller=controller,
        operation_store=FailingLinkStore(trace),
        paste_staging_store=Staging(),
        operation_id_factory=lambda: "operation-paste-link-failure",
        now_factory=lambda: "2026-08-24T10:00:00Z",
    )

    await screen._submit_intake_request(
        ref,
        ResearchSourceIntakeRequest("paste", ("PRIVATE BODY",), title="Paste"),
    )

    assert [item[0] for item in trace] == [
        "create",
        "stage",
        "prepare",
        "link-failed",
    ]
    assert not any(item[0] == "dispatch" for item in trace)
    assert not any(item[0] == "cancel" for item in trace)
    assert not any(item[0] == "delete" for item in trace)


@pytest.mark.asyncio
async def test_link_exception_after_exact_commit_releases_and_dispatches() -> None:
    """An ambiguous write answer converges from the durable operation receipt."""

    trace: list[tuple[object, ...]] = []

    class CommittedThenRaisedStore(_RecordingOperationStore):
        def advance_stage(self, operation_id, *, status, ingest_job_id="", **kwargs):
            result = super().advance_stage(
                operation_id,
                status=status,
                ingest_job_id=ingest_job_id,
                **kwargs,
            )
            if status is SourceOperationStatus.IN_PROGRESS:
                raise OSError("commit answer lost")
            return result

    class IntakeApp:
        def prepare_research_source_ingest_job(self, **kwargs):
            trace.append(("prepare", kwargs["research_source_operation_id"]))
            return SimpleNamespace(job_id="job-prepared")

        def _cancel_research_source_prepared_job(self, job_id):
            trace.append(("cancel", job_id))

        def _dispatch_research_source_catalog_job(self, job_id):
            trace.append(("dispatch", job_id))

    class Port:
        async def capabilities(self, ref):
            return {
                "attach_existing": ResearchCapability(
                    True, "available", "Available.", "local"
                )
            }

    store = CommittedThenRaisedStore(trace)
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: Port()})
    controller.select_workspace(ref)
    screen = ResearchWorkspaceScreen(
        IntakeApp(),
        controller=controller,
        operation_store=store,
        operation_id_factory=lambda: "operation-commit-answer-lost",
        now_factory=lambda: "2026-08-24T10:00:00Z",
    )

    await screen._submit_intake_request(
        ref,
        ResearchSourceIntakeRequest("url", ("https://example.invalid/paper",)),
    )

    assert any(item[0] == "dispatch" for item in trace)
    assert not any(item[0] == "cancel" for item in trace)


@pytest.mark.asyncio
async def test_incompatible_link_keeps_paste_when_durable_cancel_fails() -> None:
    """Cleanup cannot outrun a cancellation write that did not commit."""

    trace: list[tuple[object, ...]] = []

    class IncompatibleStore(_RecordingOperationStore):
        def advance_stage(self, operation_id, *, status, ingest_job_id="", **kwargs):
            if status is SourceOperationStatus.IN_PROGRESS:
                operation = self.operations[operation_id]
                self.operations[operation_id] = replace(
                    operation,
                    catalog_status=SourceOperationStatus.IN_PROGRESS,
                    ingest_job_id="job-other",
                    revision=operation.revision + 1,
                )
                raise OSError("conflicting writer won")
            return super().advance_stage(
                operation_id,
                status=status,
                ingest_job_id=ingest_job_id,
                **kwargs,
            )

    class Staging:
        def stage(self, operation_id, *, title, body):
            trace.append(("stage", operation_id))
            return f"/private/staging/{operation_id}.txt"

        def delete(self, operation_id):
            trace.append(("delete", operation_id))
            return True

    class IntakeApp:
        def prepare_research_source_ingest_job(self, **kwargs):
            return SimpleNamespace(job_id="job-prepared")

        def _cancel_research_source_prepared_job(self, job_id):
            trace.append(("cancel", job_id))
            raise OSError("job store unavailable")

        def _dispatch_research_source_catalog_job(self, job_id):
            trace.append(("dispatch", job_id))

    class Port:
        async def capabilities(self, ref):
            return {
                "attach_existing": ResearchCapability(
                    True, "available", "Available.", "local"
                )
            }

    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: Port()})
    controller.select_workspace(ref)
    screen = ResearchWorkspaceScreen(
        IntakeApp(),
        controller=controller,
        operation_store=IncompatibleStore(trace),
        paste_staging_store=Staging(),
        operation_id_factory=lambda: "operation-incompatible",
        now_factory=lambda: "2026-08-24T10:00:00Z",
    )

    await screen._submit_intake_request(
        ref,
        ResearchSourceIntakeRequest("paste", ("PRIVATE BODY",), title="Paste"),
    )

    assert any(item[0] == "cancel" for item in trace)
    assert not any(item[0] == "delete" for item in trace)
    assert not any(item[0] == "dispatch" for item in trace)


@pytest.mark.asyncio
async def test_incompatible_link_cleans_paste_after_durable_cancel() -> None:
    """An incompatible receipt is cleaned only after terminal persistence."""

    trace: list[tuple[object, ...]] = []

    class IncompatibleStore(_RecordingOperationStore):
        def advance_stage(self, operation_id, *, status, ingest_job_id="", **kwargs):
            if status is SourceOperationStatus.IN_PROGRESS:
                current = self.operations[operation_id]
                self.operations[operation_id] = replace(
                    current,
                    catalog_status=SourceOperationStatus.IN_PROGRESS,
                    ingest_job_id="job-other",
                    revision=current.revision + 1,
                )
                raise OSError("conflicting writer won")
            return super().advance_stage(
                operation_id,
                status=status,
                ingest_job_id=ingest_job_id,
                **kwargs,
            )

    class Staging:
        def stage(self, operation_id, *, title, body):
            trace.append(("stage", operation_id))
            return f"/private/staging/{operation_id}.txt"

        def delete(self, operation_id):
            trace.append(("delete", operation_id))
            return True

    class IntakeApp:
        def prepare_research_source_ingest_job(self, **kwargs):
            return SimpleNamespace(job_id="job-prepared")

        def _cancel_research_source_prepared_job(self, job_id):
            trace.append(("cancel", job_id))
            return SimpleNamespace(state=IngestJobState.CANCELLED)

        def _dispatch_research_source_catalog_job(self, job_id):
            trace.append(("dispatch", job_id))

    class Port:
        async def capabilities(self, ref):
            return {
                "attach_existing": ResearchCapability(
                    True, "available", "Available.", "local"
                )
            }

    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: Port()})
    controller.select_workspace(ref)
    screen = ResearchWorkspaceScreen(
        IntakeApp(),
        controller=controller,
        operation_store=IncompatibleStore(trace),
        paste_staging_store=Staging(),
        operation_id_factory=lambda: "operation-incompatible-cancelled",
        now_factory=lambda: "2026-08-24T10:00:00Z",
    )

    await screen._submit_intake_request(
        ref,
        ResearchSourceIntakeRequest("paste", ("PRIVATE BODY",), title="Paste"),
    )

    assert [item[0] for item in trace][-2:] == ["cancel", "delete"]
    assert not any(item[0] == "dispatch" for item in trace)


@pytest.mark.asyncio
async def test_dispatch_failure_settles_linked_job_without_deleting_retryable_paste() -> (
    None
):
    trace: list[tuple[object, ...]] = []
    store = _RecordingOperationStore(trace)

    class Staging:
        def stage(self, operation_id, *, title, body):
            trace.append(("stage", operation_id, title, body))
            return f"/private/staging/{operation_id}.txt"

        def delete(self, operation_id):
            trace.append(("delete", operation_id))
            return True

    class IntakeApp:
        def prepare_research_source_ingest_job(self, **kwargs):
            trace.append(("prepare", kwargs["research_source_operation_id"]))
            return SimpleNamespace(job_id="job-prepared")

        def _dispatch_research_source_catalog_job(self, job_id):
            linked = store.operations["operation-paste-dispatch-failure"]
            assert linked.ingest_job_id == job_id
            trace.append(("dispatch", job_id))
            raise RuntimeError("dispatcher unavailable")

        def _fail_research_source_prepared_job(self, job_id):
            trace.append(("fail", job_id))

    class Port:
        async def capabilities(self, ref):
            return {
                "attach_existing": ResearchCapability(
                    True, "available", "Available.", "server"
                )
            }

    ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "workspace-1",
        server_profile_id="profile",
        principal_id="principal",
    )
    controller = ResearchWorkspaceController({WorkspaceDataSource.SERVER: Port()})
    controller.select_workspace(ref)
    screen = ResearchWorkspaceScreen(
        IntakeApp(),
        controller=controller,
        operation_store=store,
        paste_staging_store=Staging(),
        operation_id_factory=lambda: "operation-paste-dispatch-failure",
        now_factory=lambda: "2026-08-24T10:00:00Z",
    )

    await screen._submit_intake_request(
        ref,
        ResearchSourceIntakeRequest("paste", ("PRIVATE BODY",), title="Paste"),
    )

    assert [item[0] for item in trace] == [
        "create",
        "stage",
        "prepare",
        "advance",
        "dispatch",
        "fail",
    ]
    assert not any(item[0] == "delete" for item in trace)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data_source", "origin"),
    [
        (WorkspaceDataSource.LOCAL, "local"),
        (WorkspaceDataSource.SERVER, "server"),
    ],
)
async def test_immediate_terminal_listener_observes_link_once_before_prepare_returns(
    data_source: WorkspaceDataSource,
    origin: str,
) -> None:
    """An owner that settles synchronously cannot outrun operation lineage."""

    trace: list[tuple[object, ...]] = []
    store = _RecordingOperationStore(trace)
    registry = LibraryIngestJobRegistry()
    terminal_observations: list[tuple[str, str, str]] = []

    def observe_terminal() -> None:
        terminal = tuple(
            job
            for job in registry.jobs()
            if job.state
            in {
                IngestJobState.DONE,
                IngestJobState.FAILED,
                IngestJobState.CANCELLED,
                IngestJobState.SKIPPED,
            }
        )
        for job in terminal:
            operation = store.operations[job.research_source_operation_id]
            terminal_observations.append(
                (operation.operation_id, operation.ingest_job_id, job.job_id)
            )

    registry.add_listener(observe_terminal)

    class IntakeApp:
        def prepare_research_source_ingest_job(self, **kwargs):
            return registry.submit(
                source_path=kwargs["source_path"],
                origin=origin,
                research_source_operation_id=kwargs["research_source_operation_id"],
            )

        def _dispatch_research_source_catalog_job(self, job_id):
            registry.mark_failed(job_id, error="Immediate owner failure")

    class Port:
        async def capabilities(self, ref):
            return {
                "attach_existing": ResearchCapability(
                    True, "available", "Available.", origin
                )
            }

    ref = QualifiedWorkspaceRef(
        data_source,
        f"workspace-{origin}",
        server_profile_id="profile" if origin == "server" else "",
        principal_id="principal" if origin == "server" else "",
    )
    controller = ResearchWorkspaceController({data_source: Port()})
    controller.select_workspace(ref)
    screen = ResearchWorkspaceScreen(
        IntakeApp(),
        controller=controller,
        operation_store=store,
        operation_id_factory=lambda: f"operation-immediate-{origin}",
        now_factory=lambda: "2026-08-24T10:00:00Z",
    )

    await screen._submit_intake_request(
        ref,
        ResearchSourceIntakeRequest("url", (f"https://example.invalid/{origin}",)),
    )

    assert terminal_observations == [
        (
            f"operation-immediate-{origin}",
            "ingest-job-1",
            "ingest-job-1",
        )
    ]


@pytest.mark.asyncio
async def test_paste_staging_is_bound_after_operation_create_and_cleaned_if_submit_fails() -> (
    None
):
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

    def prepare(**kwargs):
        trace.append(("prepare", kwargs["research_source_operation_id"]))
        raise RuntimeError("submit failed")

    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: Port()})
    controller.select_workspace(ref)
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(prepare_research_source_ingest_job=prepare),
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
        "prepare",
        "advance",
        "delete",
    ]
    assert trace[1][1] == trace[0][1] == "operation-paste"


@pytest.mark.asyncio
async def test_captured_server_ref_does_not_fall_back_after_navigation() -> None:
    trace: list[tuple[object, ...]] = []
    store = _RecordingOperationStore(trace)

    def prepare(**kwargs):
        trace.append(("prepare", kwargs["required_origin"]))
        return SimpleNamespace(
            job_id="job-server", state=SimpleNamespace(value="queued")
        )

    def dispatch(job_id):
        trace.append(("dispatch", job_id))

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
        SimpleNamespace(
            prepare_research_source_ingest_job=prepare,
            _dispatch_research_source_catalog_job=dispatch,
        ),
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

    assert ("prepare", "server") in trace
    assert ("dispatch", "job-server") in trace
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
async def test_annotation_edit_reopens_and_survives_overlay_store_restart(
    tmp_path,
) -> None:
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
        inspector.query_one(
            "#research-source-annotation-list", Select
        ).value = "annotation-stable"
        await pilot.pause()
        inspector.query_one(
            "#research-source-annotation-note", TextArea
        ).text = "Edited note"
        inspector.query_one("#research-source-annotation-save", Button).press()
        await pilot.pause(0.2)

        screen.query_one("#research-source-row-details-0", Button).press()
        await pilot.pause()
        reopened = app.screen
        assert isinstance(reopened, ResearchSourceInspectorModal)
        reopened.query_one(
            "#research-source-annotation-list", Select
        ).value = "annotation-stable"
        await pilot.pause()
        assert reopened.query_one(
            "#research-source-annotation-note", TextArea
        ).text == ("Edited note")

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
    initial = first_store.save(ref, ResearchPanePreferences(), expected_revision=0)
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
        for _ in range(20):
            await pilot.pause()
            if (
                app.screen is screen
                and screen.query("#research-chat-pane")
                and [folder.folder_id for folder in screen._source_folders]
                == ["folder-remote"]
            ):
                break
        else:
            pytest.fail(
                "overlay reload did not remount the workspace with remote state"
            )
        assert [folder.folder_id for folder in screen._source_folders] == [
            "folder-remote"
        ]
