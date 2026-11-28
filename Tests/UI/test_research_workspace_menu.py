import pytest
from textual.widgets import Button, Input, Select, TextArea
from zipfile import ZipFile

from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from Tests.UI.app_factory import swap_app_aliases
from Tests.app_module_patches import patch_app_global
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Research_Workspace.quick_notes import ResearchNotePageRequest

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def _action(app, screen, pilot, name):
    menu = screen.query_one("#research-workspace-actions", Button)
    await _until(pilot, lambda: not menu.disabled, "workspace menu ready")
    menu.press()
    await _until(
        pilot,
        lambda: bool(list(app.screen.query(f"#research-action-{name}"))),
        f"{name} in workspace menu",
    )
    app.screen.query_one(f"#research-action-{name}", Button).press()


async def _name(app, pilot, name):
    await _until(
        pilot,
        lambda: bool(list(app.screen.query("#research-workspace-name"))),
        "workspace name",
    )
    app.screen.query_one("#research-workspace-name", Input).value = name
    app.screen.query_one("#research-workspace-name-confirm", Button).press()


async def test_local_workspace_menu_mutates_canonical_registry_and_restores_archive(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(60, 20)) as pilot:
            screen, _ = await open_chat(app, pilot)
            starting_ref = screen.controller.selected_ref

            await _action(app, screen, pilot, "new")
            await _name(app, pilot, "Field records")
            await _until(
                pilot,
                lambda: (
                    screen.controller.selected_ref != starting_ref
                    and screen.controller.selected_ref is not None
                ),
                "new workspace selected",
            )
            ref = screen.controller.selected_ref
            registry = app.workspace_registry_service
            assert registry.get_workspace(ref.workspace_id).name == "Field records"

            await _action(app, screen, pilot, "rename")
            await _name(app, pilot, "Field records reviewed")
            await _until(
                pilot,
                lambda: (
                    screen.controller.visible_workspace.name == "Field records reviewed"
                ),
                "renamed workspace",
            )
            assert (
                registry.get_workspace(ref.workspace_id).name
                == "Field records reviewed"
            )

            await _action(app, screen, pilot, "archive")
            await _until(
                pilot,
                lambda: bool(list(app.screen.query("#confirm-button"))),
                "archive confirmation",
            )
            app.screen.query_one("#confirm-button", Button).press()
            await _until(
                pilot,
                lambda: screen.controller.selected_ref == starting_ref,
                "other active workspace selected",
            )
            assert registry.get_workspace(ref.workspace_id).archived

            await _action(app, screen, pilot, "archived")
            await _until(
                pilot,
                lambda: bool(list(app.screen.query("#research-archived-picker"))),
                "archived owner list",
            )
            app.screen.query_one("#research-archived-picker", Select).value = ref
            restore = app.screen.query_one("#research-workspace-restore", Button)
            await _until(pilot, lambda: not restore.disabled, "restore enabled")
            restore.press()
            await _until(
                pilot,
                lambda: screen.controller.selected_ref == ref,
                "restored workspace selected",
            )
            assert not registry.get_workspace(ref.workspace_id).archived


async def test_template_creates_canonical_workspace_with_an_editable_starter_draft(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(84, 24)) as pilot:
            screen, _ = await open_chat(app, pilot)
            await _action(app, screen, pilot, "templates")
            await _until(
                pilot,
                lambda: bool(
                    list(app.screen.query("#research-template-literature-review"))
                ),
                "three workspace presets",
            )
            assert len(app.screen.query(".research-workspace-template")) == 3
            app.screen.query_one("#research-template-literature-review", Button).press()
            await _name(app, pilot, "Literature review")
            note = screen.query_one("#research-quick-note-title", Input)
            await _until(
                pilot, lambda: note.value == "Literature Review Plan", "review draft"
            )
            ref = screen.controller.selected_ref
            assert (
                app.workspace_registry_service.get_workspace(ref.workspace_id).name
                == "Literature review"
            )
            assert (
                "Source checklist"
                in screen.query_one("#research-quick-note-body", TextArea).text
            )


async def test_workspace_banner_and_pin_reopen_from_device_preferences(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, _ = await open_chat(app, pilot)
            ref = screen.controller.selected_ref
            await _action(app, screen, pilot, "banner")
            await _name(app, pilot, "Fieldwork round two")
            await _action(app, screen, pilot, "pin")
            await _action(app, screen, pilot, "pinned")
            await _until(
                pilot,
                lambda: bool(list(app.screen.query("#research-pinned-picker"))),
                "pinned workspace picker",
            )
            picker = app.screen.query_one("#research-pinned-picker", Select)
            picker.value = ref
            button = app.screen.query_one("#research-workspace-open-pinned", Button)
            await _until(
                pilot, lambda: not button.disabled, "pinned workspace can reopen"
            )
            button.press()
            await _until(
                pilot, lambda: app.screen is screen, "pinned workspace reopened"
            )
            saved = screen.overlay_store.load(ref).workspace_presentation
            assert saved.pinned is True and saved.banner == "Fieldwork round two"
            assert saved.last_opened_at is not None
            assert (
                app.workspace_registry_service.get_workspace(ref.workspace_id).name
                == "Study group"
            )


async def test_import_bundle_uses_canonical_notes_and_replays_into_same_workspace(
    tmp_path, monkeypatch
):
    bundle = tmp_path / "fieldwork.zip"
    with ZipFile(bundle, "w") as archive:
        archive.writestr(
            "manifest.md",
            "# Research workspace bundle\n\nLocal workspace `source`\n\n- Export content · note `1` · version 1 · `items/1-note.md`\n- Export content · artifact `2` · version 1 · `items/2-artifact.md`\n",
        )
        archive.writestr("items/1-note.md", "# Field note\n\nCollected samples.\n")
        archive.writestr(
            "items/2-artifact.md", "# Field artifact\n\nReviewed samples.\n"
        )
    app, db, _ = build_research_app(tmp_path, monkeypatch)
    notes = NotesInteropService(
        base_db_directory=tmp_path, api_client_id="test-client", global_db_to_use=db
    )
    swap_app_aliases(
        app,
        notes_scope_service=NotesScopeService(
            local_notes_service=notes, server_service=None
        ),
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(100, 30)) as pilot:
            screen, _ = await open_chat(app, pilot)
            original_ref = screen.controller.selected_ref
            imported = None
            for _ in range(2):
                await _action(app, screen, pilot, "import")
                await _until(
                    pilot,
                    lambda: bool(list(app.screen.query("#filename-input"))),
                    "bundle file picker",
                )
                app.screen.query_one("#filename-input", Input).value = str(bundle)
                app.screen.query_one("#select", Button).press()
                await _name(app, pilot, "Imported fieldwork")
                menu = screen.query_one("#research-workspace-actions", Button)
                await _until(
                    pilot,
                    lambda menu=menu: not menu.disabled,
                    "bundle owner write completed",
                )
                ref = screen.controller.selected_ref
                assert ref != original_ref and (imported is None or imported == ref), [
                    notification.message for notification in app._notifications
                ]
                imported = ref
                port = screen.controller.port_for_data_source(ref.data_source)
                page = await port.list_notes(ref, ResearchNotePageRequest())
                assert [(note.title, note.content) for note in page.items] == [
                    ("Field note", "Collected samples.")
                ]
                links = app.workspace_registry_service.list_workspace_memberships(
                    ref.workspace_id
                )
                (artifact_link,) = (
                    link for link in links if link.item_type == "chatbook"
                )
                artifact = await app.local_chatbook_service.get_chatbook(
                    artifact_link.item_id
                )
                assert artifact["metadata"]["content"] == "Reviewed samples."
