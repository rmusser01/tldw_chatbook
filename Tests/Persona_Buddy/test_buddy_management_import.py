"""Pasted pack paths reach real validation and publication without changing authority."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Persona_Visual.test_persona_visual_importer import (
    _archive_payloads,
    _write_archive,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
from tldw_chatbook.Persona_Buddy.library import BuddyLibrary
from tldw_chatbook.Persona_Buddy.preferences import (
    BuddySelection,
    PersonaBuddyPreferences,
)
from tldw_chatbook.UI.Navigation.buddy_management import BuddyManagementCoordinator
from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
    BuddyManagementChoice,
)


@pytest.fixture
def imports(tmp_path, monkeypatch):
    from tldw_chatbook import config

    profile = tmp_path / "profile"
    profile.mkdir(mode=0o700)
    db = CharactersRAGDB(tmp_path / "buddy.db", "buddy-import-input")
    library = BuddyLibrary(db, profile)
    previous = PersonaBuddyPreferences(selection=BuddySelection("previous"))
    controller = PersonaBuddyController(preferences=previous)
    manager = BuddyManagementCoordinator(
        SimpleNamespace(app_config={}, console_runtime=None),
        controller=controller,
        library=library,
    )
    archive = _write_archive(
        tmp_path / "my buddy.tldw-persona-vpack", _archive_payloads()
    )
    monkeypatch.setattr(config, "save_settings_to_cli_config", lambda _: True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    try:
        yield manager, archive, previous
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "style", ["absolute", "home", "home-double-slash", "single-quoted", "double-quoted"]
)
async def test_pasted_path_imports_independent_artwork(imports, style):
    manager, archive, _ = imports
    entered = {
        "absolute": str(archive),
        "home": "~/" + archive.name,
        "home-double-slash": "~//" + archive.name,
        "single-quoted": "'" + str(archive) + "'",
        "double-quoted": '"~/' + archive.name + '"',
    }[style]
    await manager.apply_choice(BuddyManagementChoice(import_path=entered))
    selected = manager.controller.current_preferences().selection.buddy_id
    assert selected != "previous"
    assert len(manager.library.list_buddies()) == 1
    assert manager.library.get_graph(selected).identity.persona_id is None
    assert manager.library.resolve_preview(selected).reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_style", ["absolute", "home", "double-quoted-home"])
async def test_equivalent_retry_paths_reuse_installed_buddy_after_save_failure(
    imports, monkeypatch, retry_style
):
    from tldw_chatbook import config

    manager, archive, previous = imports
    cached_imports = {}
    monkeypatch.setattr(config, "save_settings_to_cli_config", lambda _: False)
    with pytest.raises(ValueError, match="Could not save Buddy settings") as failed:
        await manager.apply_choice(
            BuddyManagementChoice(import_path="'" + str(archive) + "'"),
            expected_revision=0,
            imports=cached_imports,
        )
    installed = manager.library.list_buddies()
    assert len(installed) == 1
    assert manager.controller.current_preferences() == previous
    retry_path = {
        "absolute": str(archive),
        "home": "~/" + archive.name,
        "double-quoted-home": '"~/' + archive.name + '"',
    }[retry_style]
    monkeypatch.setattr(config, "save_settings_to_cli_config", lambda _: True)
    await manager.apply_choice(
        BuddyManagementChoice(import_path=retry_path),
        expected_revision=failed.value.buddy_retry_revision,
        imports=cached_imports,
    )
    assert manager.library.list_buddies() == installed
    assert manager.controller.current_preferences().selection == BuddySelection(
        installed[0].id
    )
    assert cached_imports == {str(archive): installed[0].id}


@pytest.mark.parametrize(
    "value", [None, 7, b"pack.zip", [], {"path": "private-input"}, "x" * 4097]
)
def test_shared_import_path_boundary_rejects_invalid_values_without_echoing_them(value):
    from tldw_chatbook.Utils.input_validation import validate_buddy_import_path

    with pytest.raises(ValueError) as failed:
        validate_buddy_import_path(value)
    assert str(failed.value) == "Check the path. Enter a local Buddy pack filename."


def test_shared_import_path_key_keeps_a_link_distinct_from_its_target(imports):
    from tldw_chatbook.Utils.input_validation import validate_buddy_import_path

    _, archive, _ = imports
    link = archive.with_name("linked.zip")
    link.symlink_to(archive)
    assert validate_buddy_import_path('"~/' + link.name + '"') == str(link)
    assert validate_buddy_import_path(str(link)) != validate_buddy_import_path(
        str(archive)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing", "not found"),
        ("directory", "regular file"),
        ("symlink", "regular file"),
        ("html", "Download raw file"),
        ("relative", "absolute path"),
        ("url", "Download the pack"),
        ("unsafe", "Check the path"),
    ],
)
async def test_bad_input_explains_recovery_and_retains_selection(
    imports, kind, message
):
    manager, archive, previous = imports
    if kind == "symlink":
        link = archive.with_name("linked.zip")
        link.symlink_to(archive)
        entered = str(link)
    elif kind == "html":
        archive.write_text("<!doctype html><title>GitHub file page</title>")
        entered = str(archive)
    else:
        entered = {
            "missing": str(archive.with_name("missing.zip")),
            "directory": str(archive.parent),
            "relative": archive.name,
            "url": "https://github.com/example/buddy.zip",
            "unsafe": str(archive) + "\x00",
        }[kind]
    with pytest.raises(ValueError, match=message) as failed:
        await manager.apply_choice(BuddyManagementChoice(import_path=entered))
    assert str(archive.parent) not in str(failed.value)
    assert manager.controller.current_preferences() == previous
    assert manager.library.list_buddies() == ()


@pytest.mark.asyncio
async def test_install_failure_identifies_storage_and_retains_selection(
    imports, monkeypatch
):
    manager, archive, previous = imports

    def fail_publish(_):
        raise OSError("private profile path must not reach the UI")

    monkeypatch.setattr(manager.library, "publish_review", fail_publish)
    with pytest.raises(ValueError, match="profile storage") as failed:
        await manager.apply_choice(BuddyManagementChoice(import_path=str(archive)))
    assert "private profile path" not in str(failed.value)
    assert manager.controller.current_preferences() == previous


@pytest.mark.asyncio
async def test_unreadable_file_is_not_reported_as_an_invalid_pack(imports, monkeypatch):
    from tldw_chatbook.Persona_Visual import importer

    manager, archive, previous = imports
    original_open = importer.os.open

    def deny_archive(path, *args, **kwargs):
        if path == archive:
            raise PermissionError("private filename must not reach the UI")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(importer.os, "open", deny_archive)
    with pytest.raises(ValueError, match="file access permissions") as failed:
        await manager.apply_choice(BuddyManagementChoice(import_path=str(archive)))
    assert "private filename" not in str(failed.value)
    assert manager.controller.current_preferences() == previous
    assert manager.library.list_buddies() == ()


@pytest.mark.asyncio
async def test_review_cannot_hide_a_concurrent_settings_change(imports, monkeypatch):
    manager, archive, _ = imports
    review_archive = manager.library.review_archive

    def change_while_reviewing(path):
        review = review_archive(path)
        manager.controller.apply_preferences_patch(selection=BuddySelection("newer"))
        return review

    monkeypatch.setattr(manager.library, "review_archive", change_while_reviewing)
    with pytest.raises(ValueError, match="changed elsewhere"):
        await manager.apply_choice(
            BuddyManagementChoice(import_path=str(archive)), expected_revision=0
        )
    assert manager.controller.current_preferences().selection == BuddySelection("newer")
    assert manager.library.list_buddies() == ()


@pytest.mark.asyncio
async def test_changed_archive_is_not_reported_as_storage_failure(imports, monkeypatch):
    manager, archive, previous = imports
    review_archive = manager.library.review_archive

    def replace_after_review(path):
        review = review_archive(path)
        archive.write_bytes(b"replacement content")
        return review

    monkeypatch.setattr(manager.library, "review_archive", replace_after_review)
    with pytest.raises(ValueError, match="changed during import"):
        await manager.apply_choice(BuddyManagementChoice(import_path=str(archive)))
    assert manager.controller.current_preferences() == previous
    assert manager.library.list_buddies() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "message"),
    [("unsupported", "unsupported format"), ("stale", "changed during import")],
)
async def test_review_category_provides_safe_recovery(
    imports, monkeypatch, category, message
):
    from tldw_chatbook.Persona_Visual.importer import PersonaVisualImportError

    manager, archive, previous = imports

    def fail_review(_):
        raise PersonaVisualImportError("persona_visual_import_" + category)

    monkeypatch.setattr(manager.library, "review_archive", fail_review)
    with pytest.raises(ValueError, match=message) as failed:
        await manager.apply_choice(BuddyManagementChoice(import_path=str(archive)))
    assert "persona_visual_import_" not in str(failed.value)
    assert manager.controller.current_preferences() == previous


@pytest.mark.asyncio
async def test_modal_can_recover_from_a_missing_file_and_import_the_pasted_path(
    imports,
):
    import asyncio

    from textual.app import App
    from textual.widgets import Button, Input

    from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
        BuddyManagementModal,
    )

    manager, archive, previous = imports
    app = App()
    async with app.run_test(size=(100, 40)) as pilot:
        modal = BuddyManagementModal(
            initial=BuddyManagementChoice(import_path="~/missing.zip"),
            apply=manager.apply_choice,
        )
        app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#buddy-apply", Button).press()
        async with asyncio.timeout(5):
            while "not found" not in str(
                modal.query_one("#buddy-import-error").render()
            ):
                await pilot.pause()
        assert app.screen is modal
        assert manager.controller.current_preferences() == previous
        modal.query_one("#buddy-import", Input).value = '"~/' + archive.name + '"'
        modal.query_one("#buddy-apply", Button).press()
        async with asyncio.timeout(5):
            while app.screen is modal:
                await pilot.pause()
        assert len(manager.library.list_buddies()) == 1
        assert manager.controller.current_preferences().selection != previous.selection
