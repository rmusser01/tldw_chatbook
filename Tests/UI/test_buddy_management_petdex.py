"""Management owns real Petdex staging and independent character publication."""

import json
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Checkbox, Input, Select

from Tests.Actor_Packs.test_actor_pack_attribution import services
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_navigation_decisions import _until
from Tests.UI.test_petdex_import_review import _source
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
from tldw_chatbook.Persona_Buddy.library import BuddyLibrary
from tldw_chatbook.UI.Navigation.buddy_management import BuddyManagementCoordinator
from tldw_chatbook.Widgets.Persona_Widgets.buddy_character_review import (
    BuddyCharacterReviewDialog,
)
from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
    BuddyManagementModal,
)
from tldw_chatbook.Widgets.Persona_Widgets.petdex_import_review import (
    PetdexImportReviewDialog,
)


@pytest.fixture
def journey(tmp_path, monkeypatch):
    from tldw_chatbook import config

    root = tmp_path / "profile"
    db, importer, _, _ = services(root)
    monkeypatch.setattr(config, "get_user_data_dir", lambda: root)
    monkeypatch.setattr(config, "save_settings_to_cli_config", lambda _: True)
    app = ConsolidatedCSSApp()
    app.app_config = {}
    app.chachanotes_db = db
    app.local_character_persona_service = importer._local_service
    app.character_persona_scope_service = SimpleNamespace(
        local_service=importer._local_service
    )
    app.console_runtime = None
    app.reconcile_persona_buddy_view = lambda: None
    manager = BuddyManagementCoordinator(
        app, controller=PersonaBuddyController(), library=BuddyLibrary(db, root)
    )
    package = tmp_path / "pet"
    package.mkdir()
    source = _source()
    (package / "pet.json").write_text(
        json.dumps(
            {
                "name": "Review pet",
                "spriteVersionNumber": 1,
                "creator": "Example artist",
                "notices": "Keep this notice",
            }
        )
    )
    (package / "spritesheet.png").write_bytes(source.image_bytes)
    try:
        yield app, manager, package, db
    finally:
        db.close_connection()


async def open_management(app, manager, pilot):
    manager.request_open()
    await _until(lambda: isinstance(app.screen, BuddyManagementModal))
    await pilot.pause()
    return app.screen


async def review_package(app, modal, package, pilot):
    modal.query_one("#buddy-petdex", Button).press()
    await _until(lambda: isinstance(app.screen, PetdexImportReviewDialog))
    await pilot.pause()
    dialog = app.screen
    dialog.query_one("#petdex-source", Input).value = str(package)
    dialog.query_one("#petdex-local", Button).press()
    await _until(lambda: dialog.source is not None and not dialog._busy)
    dialog.query_one("#petdex-prepare", Button).press()
    await _until(lambda: dialog._prepared is not None and not dialog._busy)
    dialog.query_one("#petdex-reviewed", Checkbox).value = True
    await pilot.pause()
    dialog.query_one("#petdex-accept", Button).press()
    await _until(lambda: modal.staged_review is not None)
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["cancel", "source", "profile", "publication", "settings", "apply"]
)
async def test_management_petdex_review_apply_and_safe_exits(
    journey, outcome, monkeypatch
):
    app, manager, package, _db = journey
    before = manager.controller.current_preferences()
    personas = app.local_character_persona_service.list_persona_profiles()
    async with app.run_test(size=(100, 42)) as pilot:
        modal = await open_management(app, manager, pilot)
        # Cancelling source review never installs anything.
        modal.query_one("#buddy-petdex", Button).press()
        await _until(lambda: isinstance(app.screen, PetdexImportReviewDialog))
        await pilot.pause()
        await pilot.press("escape")
        await _until(lambda: app.screen is modal)
        await review_package(app, modal, package, pilot)
        assert manager.library.list_buddies() == ()
        assert modal.staged_review.artwork["license"] is None
        if outcome == "cancel":
            await pilot.press("escape")
        else:
            if outcome == "source":
                (package / "pet.json").write_text("{}")
            elif outcome == "profile":
                app.chachanotes_db = object()
            elif outcome == "publication":
                import tldw_chatbook.Persona_Buddy.library as library_module

                def fail(*args, **kwargs):
                    raise OSError("disk full")

                monkeypatch.setattr(library_module, "publish_persona_visual", fail)
            elif outcome == "settings":
                from tldw_chatbook import config

                monkeypatch.setattr(
                    config, "save_settings_to_cli_config", lambda _: False
                )
            modal.query_one("#buddy-apply", Button).press()
            await pilot.pause()
            await _until(lambda: not modal._applying)
        if outcome == "apply":
            assert len(manager.library.list_buddies()) == 1
            record = manager.library.list_buddies()[0]
            assert manager.library.get_graph(record.id).identity.persona_id is None
            assert (
                manager.controller.current_preferences().selection.buddy_id == record.id
            )
        elif outcome == "settings":
            assert len(manager.library.list_buddies()) == 1
            assert manager.controller.current_preferences() == before
            recovery = str(modal.query_one("#buddy-form-error").render())
            assert "Buddy was installed" in recovery
            assert "previous settings" in recovery
            assert "Retry Apply" in recovery
            assert "reopen" in recovery
            await pilot.press("escape")
            assert len(manager.library.list_buddies()) == 1
        else:
            assert manager.library.list_buddies() == ()
            assert manager.controller.current_preferences() == before
        assert app.local_character_persona_service.list_persona_profiles() == personas


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [None, "source", "selection", "profile"])
async def test_selected_independent_buddy_creates_durable_character_without_applying(
    journey,
    change,
):
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
    from tldw_chatbook.Widgets.Console.character_expression_avatar import (
        CharacterExpressionAvatar,
    )

    app, manager, package, db = journey
    async with app.run_test(size=(100, 42)) as pilot:
        modal = await open_management(app, manager, pilot)
        await review_package(app, modal, package, pilot)
        modal.query_one("#buddy-apply", Button).press()
        await _until(lambda: app.screen is not modal)
        record = manager.library.list_buddies()[0]
        before = manager.controller.current_preferences()
        settings = dict(app.app_config)
        characters_before = db.list_character_cards()
        modal = await open_management(app, manager, pilot)
        modal.query_one("#buddy-motion", Select).value = "static"
        modal.query_one("#buddy-character", Button).press()
        await _until(lambda: isinstance(app.screen, BuddyCharacterReviewDialog))
        await pilot.pause()
        dialog = app.screen
        dialog.query_one("#buddy-name", Input).value = "Independent pet character"
        dialog.query_one("#buddy-prepare", Button).press()
        await _until(lambda: dialog._conversion is not None and not dialog._busy)
        for mode in ("static", "dynamic"):
            dialog.query_one("#buddy-preview-mode", Select).value = mode
            await pilot.pause()
            await _until(
                lambda: all(
                    not worker.is_running
                    for worker in dialog.workers
                    if worker.group == "buddy-preview"
                )
            )
            preview = dialog.query_one(
                "#buddy-expression-preview", CharacterExpressionAvatar
            )
            assert preview._animate is (mode == "dynamic")
        dialog.query_one("#buddy-warnings-accepted", Checkbox).value = True
        await pilot.pause()
        if change == "source":
            exported = manager.library.repository.get_active_buddy_pack_for_export(
                record.id
            )
            (manager.library.profile_root / exported.assets[0].storage_key).write_bytes(
                b"changed"
            )
        elif change == "selection":
            modal.query_one("#buddy-artwork", Select).value = "#none"
        elif change == "profile":
            app.character_persona_scope_service.local_service = object()
        dialog.query_one("#buddy-create", Button).press()
        if change is not None:
            await pilot.pause()
            await _until(lambda: not dialog._busy)
            assert dialog._result is None
            assert db.list_character_cards() == characters_before
            assert manager.controller.current_preferences() == before
            return
        await _until(lambda: dialog._result is not None)
        character_id = dialog._result.local_actor_id
        assert not dialog.query_one("#buddy-open", Button).display
        dialog.query_one("#buddy-cancel", Button).press()
        await _until(lambda: app.screen is modal)
        await pilot.press("escape")
        assert manager.controller.current_preferences() == before
        assert app.app_config == settings
        # Publication copied source bytes. Corrupting the original Buddy asset
        # leaves the independent character graph and its attribution unchanged.
        graph = VisualIdentityRepository(db).get_active_actor_pack(
            "character", character_id
        )
        assert graph
        contexts = [
            json.loads(asset["source_context_json"]) for asset in graph["assets"]
        ]
        assert all(
            context["tldw/artwork"]["creator"] == "Example artist"
            for context in contexts
        )
        assert all(context["tldw/artwork"]["license"] is None for context in contexts)
        assert all(
            "Keep this notice" in context["tldw/artwork"]["notices"]
            for context in contexts
        )
        copied = {
            asset["storage_relpath"]: (
                manager.library.profile_root
                / "visual_identities"
                / asset["storage_relpath"]
            ).read_bytes()
            for asset in graph["assets"]
        }
        card = db.get_character_card_by_id(int(character_id))
        assert card["name"] == "Independent pet character"
        assert len(db.list_character_cards()) == len(characters_before) + 1
        snapshot = manager.library.repository.get_active_buddy_pack_for_export(
            record.id
        )
        for asset in snapshot.assets:
            (manager.library.profile_root / asset.storage_key).write_bytes(b"changed")
        assert (
            VisualIdentityRepository(db).get_active_actor_pack(
                "character", character_id
            )
            == graph
        )
        assert all(
            (manager.library.profile_root / "visual_identities" / path).read_bytes()
            == data
            for path, data in copied.items()
        )
