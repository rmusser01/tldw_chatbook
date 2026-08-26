from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Markdown, Static, TextArea

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Research_Workspace import (
    BoundedPageResult,
    QualifiedWorkspaceRef,
    ResearchCapability,
    ResearchNoteConflictError,
    ResearchQuickNote,
    ResearchWorkspaceController,
    ResearchWorkspaceCatalogState,
    ResearchWorkspaceSummary,
    WorkspaceDataSource,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError
from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError


LOCAL_REF = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "notes-local")
SERVER_REF = QualifiedWorkspaceRef(
    WorkspaceDataSource.SERVER,
    "notes-server",
    server_profile_id="server-profile",
    principal_id="credential-fingerprint:test:0123456789abcdef01234567",
)


def note(ref=LOCAL_REF, *, note_id="note-1", version=3):
    return ResearchQuickNote(
        ref=ref,
        note_id=note_id,
        title="Grounded finding",
        content="**Evidence-backed** note",
        tags=("review",),
        version=version,
        source_ids=("source-1",),
    )


class _StudioHarness(ConsolidatedCSSApp):
    async def on_mount(self) -> None:
        from tldw_chatbook.UI.Research_Workspace_Modules.studio_region import (
            ResearchStudioRegion,
        )

        await self.push_screen(_WidgetScreen(ResearchStudioRegion()))


class _WidgetScreen(__import__("textual.screen", fromlist=["Screen"]).Screen):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def compose(self):
        yield self.widget


@pytest.mark.asyncio
async def test_studio_mounts_compose_once_quick_notes_parity_controls() -> None:
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )

    app = _StudioHarness()
    async with app.run_test(size=(72, 44)) as pilot:
        await pilot.pause()
        section = app.screen.query_one(ResearchQuickNotesSection)

        expected_labels = {
            "Load",
            "New",
            "Search",
            "Previous",
            "Next",
            "Edit",
            "Markdown Preview",
            "Save",
            "Delete",
            "Download .md",
            "Clear",
            "Undo",
            "Capture selected sources",
        }
        labels = {str(button.label) for button in section.query(Button)}
        assert expected_labels <= labels
        assert len(list(section.query("#research-quick-note-title"))) == 1
        assert len(list(section.query("#research-quick-note-body"))) == 1
        assert len(list(section.query("#research-quick-note-preview"))) == 1
        assert section.query_one(
            "#research-quick-note-capture-message", Button
        ).disabled
        assert "grounded Chat" in str(
            section.query_one("#research-quick-note-capture-message", Button).tooltip
        )


@pytest.mark.asyncio
async def test_quick_note_load_preview_clear_and_undo_patch_in_place() -> None:
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Research_Workspace_Modules.studio_region import (
        ResearchStudioRegion,
    )

    region = ResearchStudioRegion()
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(72, 44)) as pilot:
        await app.mount(region)
        await pilot.pause()
        section = region.query_one(ResearchQuickNotesSection)
        section.sync_workspace(LOCAL_REF)
        section.sync_note(note())
        title = section.query_one("#research-quick-note-title", Input)
        body = section.query_one("#research-quick-note-body", TextArea)

        assert title.value == "Grounded finding"
        assert body.text == "**Evidence-backed** note"
        section.query_one("#research-quick-note-preview-mode", Button).press()
        await pilot.pause()
        assert not body.display
        assert section.query_one("#research-quick-note-preview", Markdown).display

        section.query_one("#research-quick-note-edit-mode", Button).press()
        section.query_one("#research-quick-note-clear", Button).press()
        await pilot.pause()
        assert title.value == ""
        assert body.text == ""
        assert section.is_dirty
        assert section.has_nonempty_dirty_draft
        section.query_one("#research-quick-note-undo", Button).press()
        await pilot.pause()
        assert title.value == "Grounded finding"
        assert body.text == "**Evidence-backed** note"


@pytest.mark.asyncio
async def test_cleared_existing_note_saves_as_untitled_without_losing_dirty_retry() -> None:
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    port = _MountedNotePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _ResearchHarness(screen)
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
            if controller.selected_ref == LOCAL_REF:
                break
        section = screen.query_one(ResearchQuickNotesSection)
        section.sync_note(port.row)
        section.query_one("#research-quick-note-clear", Button).press()
        await pilot.pause()
        section.query_one("#research-quick-note-save", Button).press()
        for _ in range(30):
            await pilot.pause(0.02)
            if port.saved:
                break

        assert port.saved[-1][1].title == "Untitled Note"
        assert port.saved[-1][1].content == ""
        assert not section.is_dirty


class _MountedNotePort:
    def __init__(self, ref=LOCAL_REF) -> None:
        self.ref = ref
        self.saved = []
        self.deleted = []
        self.conflict = False
        self.fail_capabilities = False
        self.row = note(ref)

    async def list_workspaces(self, *, include_archived=False):
        return (ResearchWorkspaceSummary(ref=self.ref, name="Notes workspace"),)

    async def capabilities(self, ref):
        if self.fail_capabilities:
            raise WorkspaceRegistryServiceError("injected capability failure")
        available = ResearchCapability(True, "available", "Available.", "notes")
        return {
            "list_sources": available,
            "get_readiness": available,
            "list_notes": available,
            "get_note": available,
            "save_note": available,
            "delete_note": ResearchCapability(
                ref.data_source is WorkspaceDataSource.LOCAL,
                "available"
                if ref.data_source is WorkspaceDataSource.LOCAL
                else "version_precondition_unavailable",
                "Available."
                if ref.data_source is WorkspaceDataSource.LOCAL
                else "Server delete cannot enforce a version check.",
                "notes",
            ),
        }

    async def list_sources(self, ref, *, limit=100, offset=0):
        return BoundedPageResult(items=(), limit=limit, offset=offset, total=0)

    async def get_readiness(self, ref, *, source_ids=()):
        return ()

    async def list_notes(self, ref, page):
        return BoundedPageResult(items=(self.row,), limit=page.limit, total=1)

    async def get_note(self, ref, note_id):
        return self.row if self.row.note_id == note_id else None

    async def save_note(self, ref, request):
        if self.conflict:
            raise ResearchNoteConflictError(ref, request.note_id or "new")
        self.saved.append((ref, request))
        self.row = ResearchQuickNote(
            ref=ref,
            note_id=request.note_id or "note-created",
            title=request.title,
            content=request.content,
            tags=request.tags,
            version=(request.expected_version or 0) + 1,
            source_ids=request.source_ids,
        )
        return self.row

    async def delete_note(self, ref, note_id, expected_version):
        self.deleted.append((ref, note_id, expected_version))
        return True


class _ResearchHarness(ConsolidatedCSSApp):
    def __init__(self, screen):
        super().__init__()
        self._screen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


@pytest.mark.asyncio
async def test_mounted_save_uses_editor_owner_version_and_source_provenance() -> None:
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    port = _MountedNotePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _ResearchHarness(screen)
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
            if controller.visible_note_page is not None:
                break
        section = screen.query_one(ResearchQuickNotesSection)
        section.sync_note(port.row)
        section.query_one("#research-quick-note-body", TextArea).text = "Changed body"
        section.set_source_provenance(("source-captured",))
        section.query_one("#research-quick-note-save", Button).press()
        for _ in range(30):
            await pilot.pause(0.02)
            if port.saved:
                break

        saved_ref, request = port.saved[-1]
        assert saved_ref == LOCAL_REF
        assert request.note_id == "note-1"
        assert request.expected_version == 3
        assert request.content == "Changed body"
        assert request.source_ids == ("source-captured",)
        assert not section.is_dirty


@pytest.mark.asyncio
async def test_mounted_conflict_copy_as_new_preserves_draft_without_force_write() -> (
    None
):
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    port = _MountedNotePort()
    port.conflict = True
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _ResearchHarness(screen)
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
            if controller.selected_ref is not None:
                break
        section = screen.query_one(ResearchQuickNotesSection)
        section.sync_note(port.row)
        section.query_one(
            "#research-quick-note-body", TextArea
        ).text = "Conflict draft retained"
        section.query_one("#research-quick-note-save", Button).press()
        for _ in range(30):
            await pilot.pause(0.02)
            if app.screen.id == "research-note-conflict-modal":
                break

        assert app.screen.query_one("#research-note-conflict-copy", Button)
        assert section.query_one("#research-quick-note-body", TextArea).text == (
            "Conflict draft retained"
        )
        port.conflict = False
        app.screen.query_one("#research-note-conflict-copy", Button).press()
        for _ in range(30):
            await pilot.pause(0.02)
            if port.saved:
                break

        _, copied = port.saved[-1]
        assert copied.note_id is None
        assert copied.expected_version is None
        assert copied.content == "Conflict draft retained"
        assert section.query_one("#research-quick-note-body", TextArea).text == (
            "Conflict draft retained"
        )
        assert not section.is_dirty


@pytest.mark.asyncio
async def test_dirty_authority_switch_is_blocked_then_retries_exact_captured_owner() -> (
    None
):
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    local = _MountedNotePort(LOCAL_REF)
    local.conflict = True
    server = _MountedNotePort(SERVER_REF)
    controller = ResearchWorkspaceController(
        {
            WorkspaceDataSource.LOCAL: local,
            WorkspaceDataSource.SERVER: server,
        }
    )
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _ResearchHarness(screen)
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
            if controller.selected_ref == LOCAL_REF:
                break
        section = screen.query_one(ResearchQuickNotesSection)
        section.sync_note(local.row)
        section.query_one(
            "#research-quick-note-body", TextArea
        ).text = "Flush this exact local draft"
        screen.query_one("#research-data-source-server", Button).press()
        for _ in range(30):
            await pilot.pause(0.02)
            if app.screen.id == "research-note-switch-recovery-modal" and len(
                app.screen.query("#research-note-switch-cancel")
            ):
                break

        assert controller.selected_ref == LOCAL_REF
        assert section.query_one("#research-quick-note-body", TextArea).text == (
            "Flush this exact local draft"
        )
        app.screen.query_one("#research-note-switch-cancel", Button).press()
        await pilot.pause(0.05)
        assert controller.selected_ref == LOCAL_REF
        assert section.is_dirty

        screen.query_one("#research-data-source-server", Button).press()
        for _ in range(30):
            await pilot.pause(0.02)
            if app.screen.id == "research-note-switch-recovery-modal" and len(
                app.screen.query("#research-note-switch-retry")
            ):
                break
        local.conflict = False
        app.screen.query_one("#research-note-switch-retry", Button).press()
        for _ in range(60):
            await pilot.pause(0.02)
            if controller.selected_ref == SERVER_REF:
                break

        assert local.saved[-1][0] == LOCAL_REF
        assert local.saved[-1][1].content == "Flush this exact local draft"
        assert controller.selected_ref == SERVER_REF
        assert controller.selected_data_source is WorkspaceDataSource.SERVER


@pytest.mark.asyncio
async def test_server_delete_is_visible_disabled_with_exact_owner_reason() -> None:
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )

    app = _StudioHarness()
    async with app.run_test(size=(72, 44)) as pilot:
        await pilot.pause()
        section = app.screen.query_one(ResearchQuickNotesSection)
        capability = ResearchCapability(
            False,
            "version_precondition_unavailable",
            "This server cannot safely delete a Quick Note with a version check.",
            "server_workspace_notes",
        )
        section.sync_capabilities({"delete_note": capability})

        delete = section.query_one("#research-quick-note-delete", Button)
        assert delete.disabled
        assert "version check" in str(delete.tooltip)
        assert "version check" in str(
            section.query_one("#research-quick-note-owner-limits", Static).render()
        )


@pytest.mark.asyncio
async def test_unavailable_note_owner_operations_remain_visibly_fail_closed() -> None:
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )

    app = _StudioHarness()
    async with app.run_test(size=(72, 44)) as pilot:
        await pilot.pause()
        section = app.screen.query_one(ResearchQuickNotesSection)
        section.sync_workspace(LOCAL_REF)
        unavailable = ResearchCapability(
            False,
            "owner_unavailable",
            "Workspace notes are unavailable for this owner.",
            "workspace_notes",
        )
        section.sync_capabilities(
            {
                "list_notes": unavailable,
                "get_note": unavailable,
                "save_note": unavailable,
                "delete_note": unavailable,
            }
        )
        section.sync_page(
            BoundedPageResult(items=(note(),), limit=20, total=2, has_more=True)
        )
        selector = section.query_one("#research-quick-note-list")
        selector.value = "note-1"
        await pilot.pause()

        for button_id in (
            "research-quick-note-load",
            "research-quick-note-new",
            "research-quick-note-search-submit",
            "research-quick-note-prev",
            "research-quick-note-next",
            "research-quick-note-save",
            "research-quick-note-delete",
        ):
            assert section.query_one(f"#{button_id}", Button).disabled
        assert "unavailable for this owner" in str(
            section.query_one("#research-quick-note-owner-limits", Static).render()
        )


@pytest.mark.asyncio
async def test_capability_refresh_from_full_to_empty_resets_every_control_fail_closed() -> (
    None
):
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )

    app = _StudioHarness()
    async with app.run_test(size=(72, 44)) as pilot:
        await pilot.pause()
        section = app.screen.query_one(ResearchQuickNotesSection)
        section.sync_workspace(LOCAL_REF)
        available = ResearchCapability(True, "available", "Available.", "notes")
        section.sync_capabilities(
            {
                name: available
                for name in ("list_notes", "get_note", "save_note", "delete_note")
            }
        )
        section.sync_capabilities({})

        for widget_id in (
            "research-quick-note-search",
            "research-quick-note-search-submit",
            "research-quick-note-load",
            "research-quick-note-new",
            "research-quick-note-save",
            "research-quick-note-delete",
        ):
            assert section.query_one(f"#{widget_id}").disabled
        assert "unavailable" in str(
            section.query_one("#research-quick-note-owner-limits", Static).render()
        ).lower()


@pytest.mark.asyncio
async def test_capability_refresh_exception_immediately_disables_previous_actions() -> None:
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    port = _MountedNotePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _ResearchHarness(screen)
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
            if controller.visible_note_page is not None:
                break
        section = screen.query_one(ResearchQuickNotesSection)
        assert not section.query_one("#research-quick-note-save", Button).disabled

        port.fail_capabilities = True
        await screen._refresh_quick_notes(expected_ref=LOCAL_REF)

        for widget_id in (
            "research-quick-note-search",
            "research-quick-note-load",
            "research-quick-note-new",
            "research-quick-note-save",
            "research-quick-note-delete",
        ):
            assert section.query_one(f"#{widget_id}").disabled
        assert "retry" in str(
            section.query_one("#research-quick-note-status", Static).render()
        ).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner_error",
    [
        WorkspaceRegistryServiceError("PRIVATE OWNER DETAIL"),
        CharactersRAGDBError("PRIVATE OWNER DETAIL"),
    ],
)
async def test_expected_local_owner_errors_keep_editor_recoverable(owner_error) -> None:
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    port = _MountedNotePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _ResearchHarness(screen)

    async def fail():
        raise owner_error

    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
            if controller.selected_ref == LOCAL_REF:
                break
        section = screen.query_one(ResearchQuickNotesSection)
        section.sync_note(port.row)
        section.query_one("#research-quick-note-body", TextArea).text = "Retained draft"

        await screen._guard_quick_note_action(fail())

        status = str(section.query_one("#research-quick-note-status", Static).render())
        assert "PRIVATE OWNER DETAIL" not in status
        assert "retained" in status.lower()
        assert section.query_one("#research-quick-note-body", TextArea).text == (
            "Retained draft"
        )


@pytest.mark.asyncio
async def test_navigation_flush_retries_exact_original_ref_even_if_editor_ref_changes() -> (
    None
):
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    local = _MountedNotePort(LOCAL_REF)
    local.conflict = True
    server = _MountedNotePort(SERVER_REF)
    controller = ResearchWorkspaceController(
        {WorkspaceDataSource.LOCAL: local, WorkspaceDataSource.SERVER: server}
    )
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _ResearchHarness(screen)
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
            if controller.selected_ref == LOCAL_REF:
                break
        section = screen.query_one(ResearchQuickNotesSection)
        section.sync_note(local.row)
        section.query_one("#research-quick-note-body", TextArea).text = (
            "Exact original draft"
        )
        flush = asyncio.create_task(screen.flush_pending_work())
        for _ in range(30):
            await pilot.pause(0.02)
            if app.screen.id == "research-note-switch-recovery-modal":
                break

        section.editor_ref = SERVER_REF
        local.conflict = False
        app.screen.query_one("#research-note-switch-retry", Button).press()
        assert await flush is True
        assert local.saved[-1][0] == LOCAL_REF
        assert local.saved[-1][1].note_id == "note-1"
        assert server.saved == []


@pytest.mark.asyncio
async def test_catalog_fallback_cancel_preserves_dirty_editor_and_selected_workspace() -> (
    None
):
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    local = _MountedNotePort(LOCAL_REF)
    local.conflict = True
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: local})
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _ResearchHarness(screen)
    fallback_ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "fallback")
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
            if controller.selected_ref == LOCAL_REF:
                break
        section = screen.query_one(ResearchQuickNotesSection)
        section.sync_note(local.row)
        section.query_one("#research-quick-note-body", TextArea).text = (
            "Catalog must not destroy this"
        )
        state = ResearchWorkspaceCatalogState(
            data_source=WorkspaceDataSource.LOCAL,
            context_revision=controller.context_revision,
            catalog_generation=controller.catalog_generation,
            workspaces=(ResearchWorkspaceSummary(ref=fallback_ref, name="Fallback"),),
        )
        apply_state = asyncio.create_task(screen._apply_catalog_state(state))
        for _ in range(30):
            await pilot.pause(0.02)
            if app.screen.id == "research-note-switch-recovery-modal":
                break

        assert app.screen.id == "research-note-switch-recovery-modal"
        app.screen.query_one("#research-note-switch-cancel", Button).press()
        await apply_state

        assert controller.selected_ref == LOCAL_REF
        assert section.editor_ref == LOCAL_REF
        assert section.query_one("#research-quick-note-body", TextArea).text == (
            "Catalog must not destroy this"
        )


@pytest.mark.asyncio
async def test_stale_save_message_cannot_write_after_editor_owner_changes() -> None:
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    port = _MountedNotePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _ResearchHarness(screen)
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
            if controller.selected_ref == LOCAL_REF:
                break
        section = screen.query_one(ResearchQuickNotesSection)
        section.sync_note(port.row)
        stale_ref, stale_request = section.capture_save_request()
        section.sync_workspace(SERVER_REF)

        await screen._save_quick_note(stale_ref, stale_request)

        assert port.saved == []


@pytest.mark.asyncio
async def test_stale_delete_confirmation_cannot_mutate_after_editor_owner_changes() -> (
    None
):
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    port = _MountedNotePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _ResearchHarness(screen)
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
            if controller.selected_ref == LOCAL_REF:
                break
        section = screen.query_one(ResearchQuickNotesSection)
        section.sync_note(port.row)
        section.sync_workspace(SERVER_REF)

        await screen._delete_quick_note(LOCAL_REF, "note-1", 3)

        assert port.deleted == []


@pytest.mark.asyncio
async def test_dirty_quick_note_body_never_enters_navigation_or_overlay_state(
    tmp_path,
) -> None:
    from tldw_chatbook.Research_Workspace.overlay_store import (
        ResearchPresentationOverlayStore,
    )
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
        ResearchQuickNotesSection,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    overlay_path = tmp_path / "overlay.json"
    port = _MountedNotePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(),
        controller=controller,
        overlay_store=ResearchPresentationOverlayStore(overlay_path),
    )
    app = _ResearchHarness(screen)
    sentinel = "PRIVATE QUICK NOTE BODY MUST STAY CANONICAL"
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
            if controller.selected_ref == LOCAL_REF:
                break
        section = screen.query_one(ResearchQuickNotesSection)
        section.sync_note(port.row)
        section.query_one("#research-quick-note-body", TextArea).text = sentinel

        assert sentinel not in json.dumps(screen.save_state())
        await screen._save_overlay_preferences()
        assert sentinel not in overlay_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_switch_and_conflict_recovery_modals_offer_no_force_overwrite() -> None:
    from tldw_chatbook.UI.Research_Workspace_Modules.quick_note_modals import (
        ResearchNoteConflictModal,
        ResearchNoteSwitchRecoveryModal,
    )

    class ModalHarness(ConsolidatedCSSApp):
        async def on_mount(self):
            await self.push_screen(ResearchNoteSwitchRecoveryModal())

    app = ModalHarness()
    async with app.run_test(size=(80, 28)) as pilot:
        await pilot.pause()
        labels = {str(button.label) for button in app.screen.query(Button)}
        assert labels == {"Retry", "Discard editor changes", "Cancel"}
        assert not any("overwrite" in label.lower() for label in labels)

    class ConflictHarness(ConsolidatedCSSApp):
        async def on_mount(self):
            await self.push_screen(ResearchNoteConflictModal())

    app = ConflictHarness()
    async with app.run_test(size=(80, 28)) as pilot:
        await pilot.pause()
        labels = {str(button.label) for button in app.screen.query(Button)}
        assert labels == {"Reload", "Copy as new", "Cancel"}
        assert not any("overwrite" in label.lower() for label in labels)
