"""Mounted source inspector status and device-only annotation behavior."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select, Static, TextArea

from tldw_chatbook.Research_Workspace import (
    QualifiedWorkspaceRef,
    ResearchSourcePreview,
    ResearchSourceSummary,
    SourceReadiness,
    SourceReadinessState,
    WorkspaceDataSource,
)
from tldw_chatbook.UI.Research_Workspace_Modules.source_inspector import (
    ResearchSourceInspectorModal,
)
from tldw_chatbook.Research_Workspace.overlay_store import ResearchSourceAnnotation


@pytest.mark.asyncio
async def test_inspector_renders_owner_status_and_returns_device_annotation() -> None:
    app = App()
    saved = []
    ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "workspace-server",
        server_profile_id="profile",
        principal_id="principal",
    )
    source = ResearchSourceSummary(
        ref=ref,
        source_id="association-7",
        catalog_item_id="7",
        title="Paper",
        source_type="pdf",
    )
    readiness = SourceReadiness(
        ref=ref,
        source_id="association-7",
        catalog_item_id="7",
        state=SourceReadinessState.INDEXING,
        stale=True,
        retry_eligible=True,
        next_action="Refresh status",
        detail="Vector indexing is still processing.",
    )
    preview = ResearchSourcePreview(
        ref=ref,
        source_id="association-7",
        catalog_item_id="7",
        preview_mode="text",
        text="Grounded preview text.",
    )

    async with app.run_test(size=(80, 24)) as pilot:
        modal = ResearchSourceInspectorModal(
            source,
            readiness=readiness,
            preview=preview,
        )
        await app.push_screen(modal, callback=saved.append)
        await pilot.pause()

        painted = " ".join(str(widget.render()) for widget in modal.query(Static))
        assert "Lifecycle: Indexing" in painted
        assert "Vector indexing is still processing." in painted
        assert "Source of truth: Server owner" in painted
        assert "Progress: owner did not report a percentage" in painted
        assert "Retry eligible: Yes" in painted
        assert "Stale: Yes" in painted
        assert "Association ID: association-7" in painted
        assert "Next action: Refresh status" in painted
        assert "Grounded preview text." in painted
        assert "Annotation · Device-only" in painted

        modal.query_one("#research-source-annotation-quote", Input).value = "Evidence"
        modal.query_one(
            "#research-source-annotation-note", TextArea
        ).text = "Check this."
        modal.query_one("#research-source-annotation-save").press()
        await pilot.pause()

    assert len(saved) == 1
    assert saved[0].source_id == "association-7"
    assert saved[0].quote == "Evidence"
    assert saved[0].note == "Check this."


@pytest.mark.asyncio
async def test_inspector_missing_preview_is_honest() -> None:
    app = App()
    ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "workspace-server",
        server_profile_id="profile",
        principal_id="principal",
    )
    source = ResearchSourceSummary(
        ref=ref,
        source_id="association-missing",
        catalog_item_id="7",
        title="Missing media",
        source_type="pdf",
    )
    async with app.run_test(size=(80, 24)) as pilot:
        modal = ResearchSourceInspectorModal(source, readiness=None, preview=None)
        await app.push_screen(modal)
        await pilot.pause()

        assert "Preview unavailable" in str(
            modal.query_one("#research-source-preview-text", Static).render()
        )
        assert "Readiness: Unavailable" in str(
            modal.query_one("#research-source-status-readiness", Static).render()
        )


@pytest.mark.asyncio
async def test_inspector_has_real_recheck_action_and_escape_safe_cancel() -> None:
    app = App()
    results = []
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")
    source = ResearchSourceSummary(
        ref=ref,
        source_id="membership-1",
        catalog_item_id="1",
        title="Evidence",
        source_type="text",
    )
    async with app.run_test(size=(80, 24)) as pilot:
        modal = ResearchSourceInspectorModal(source, readiness=None, preview=None)
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()

        recheck = modal.query_one("#research-source-status-recheck", Button)
        assert not recheck.disabled
        recheck.press()
        await pilot.pause()

        reopened = ResearchSourceInspectorModal(source, readiness=None, preview=None)
        await app.push_screen(reopened, callback=results.append)
        await pilot.pause()
        reopened.query_one("#research-source-annotation-quote", Input).value = "draft"
        assert reopened.query_one("#research-source-annotation-quote", Input).value == "draft"
        await pilot.press("escape")
        await pilot.pause()

    assert results[0].action == "recheck"
    assert results[1] is None


@pytest.mark.asyncio
async def test_inspector_lists_edits_and_deletes_stable_device_annotations() -> None:
    app = App()
    results = []
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")
    source = ResearchSourceSummary(
        ref=ref,
        source_id="membership-1",
        catalog_item_id="1",
        title="Evidence",
        source_type="text",
    )
    existing = ResearchSourceAnnotation(
        "annotation-stable",
        source.source_id,
        "Quote",
        "Original note",
        "2026-08-24T10:00:00Z",
        "2026-08-24T10:00:00Z",
    )

    async with app.run_test(size=(80, 28)) as pilot:
        edit = ResearchSourceInspectorModal(
            source, readiness=None, preview=None, annotations=(existing,)
        )
        await app.push_screen(edit, callback=results.append)
        await pilot.pause()
        annotation_list = edit.query_one("#research-source-annotation-list", Select)
        annotation_list.value = existing.annotation_id
        await pilot.pause()
        assert edit.query_one("#research-source-annotation-note", TextArea).text == "Original note"
        edit.query_one("#research-source-annotation-note", TextArea).text = "Edited note"
        edit.query_one("#research-source-annotation-save", Button).press()
        await pilot.pause()

        delete = ResearchSourceInspectorModal(
            source, readiness=None, preview=None, annotations=(existing,)
        )
        await app.push_screen(delete, callback=results.append)
        await pilot.pause()
        delete.query_one("#research-source-annotation-list", Select).value = existing.annotation_id
        await pilot.pause()
        delete.query_one("#research-source-annotation-delete", Button).press()
        await pilot.pause()

    assert results[0].action == "update"
    assert results[0].annotation_id == "annotation-stable"
    assert results[0].note == "Edited note"
    assert results[1].action == "delete"
    assert results[1].annotation_id == "annotation-stable"
