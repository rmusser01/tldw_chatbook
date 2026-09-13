"""Petdex review retains explicit drafts and source authority."""

import pytest
from textual.widgets import Button, TextArea

from Tests.UI.consolidated_css import ConsolidatedCSSApp


@pytest.mark.asyncio
async def test_petdex_entry_requires_clean_saved_local_destination():
    from Tests.UI.test_personas_persona_visual_pack import PackApp, _inventory
    from tldw_chatbook.Widgets.Persona_Widgets.personas_persona_visual_pack_widget import (
        PersonasPersonaVisualPackWidget,
    )

    app = PackApp()
    async with app.run_test(size=(90, 45)):
        widget = app.query_one(PersonasPersonaVisualPackWidget)
        button = widget.query_one("#personas-persona-visual-petdex", Button)
        widget.set_availability("unsaved")
        assert button.disabled
        widget.show_inventory(_inventory(), dirty=False)
        assert not button.disabled
        widget.show_inventory(_inventory(), dirty=True)
        assert button.disabled
        widget.set_availability("server")
        assert button.disabled


@pytest.mark.asyncio
async def test_manual_review_errors_remain_editable_and_cancel_has_no_result():
    from tldw_chatbook.Widgets.Persona_Widgets.petdex_import_review import (
        PetdexImportReviewDialog,
    )

    dialog = PetdexImportReviewDialog(authority_guard=lambda: True, config={})
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(dialog)
        await pilot.pause()
        assert dialog.query_one("#petdex-accept", Button).disabled
        dialog.query_one("#petdex-states", TextArea).load_text("not json")
        with pytest.raises(ValueError):
            dialog.collect_states()
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not dialog


def _source(*, guard=lambda: True, version=1):
    import json
    from io import BytesIO

    from PIL import Image

    from tldw_chatbook.Petdex.sources import source_from_bytes

    output = BytesIO()
    image = Image.new(
        "RGBA", (8 * 24, (9 if version == 1 else 11) * 26), (35, 180, 120, 255)
    )
    image.save(output, format="PNG")
    return source_from_bytes(
        json.dumps(
            {
                "name": "Review pet",
                "spriteVersionNumber": version,
                "creator": "Example artist",
                "license": "CC BY 4.0",
                "notices": "Keep this notice",
            }
        ).encode(),
        output.getvalue(),
        "spritesheet.png",
        guard=guard,
    )


@pytest.mark.asyncio
async def test_local_review_prepares_real_native_archive_and_all_state_previews(
    monkeypatch, tmp_path
):
    from textual.widgets import Checkbox, Input, Select, Static

    from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive
    from tldw_chatbook.Petdex import sources
    from tldw_chatbook.Widgets.Persona_Widgets.petdex_import_review import (
        PetdexImportReviewDialog,
    )

    source = _source()
    monkeypatch.setattr(sources, "read_local_package", lambda path: source)
    dialog = PetdexImportReviewDialog(authority_guard=lambda: True, config={})
    app = ConsolidatedCSSApp()
    accepted = []
    async with app.run_test(size=(90, 35)) as pilot:
        await app.push_screen(dialog, accepted.append)
        dialog.query_one("#petdex-source", Input).value = "fixture"
        await dialog._load("petdex-local")
        assert "Example artist" in str(
            dialog.query_one("#petdex-credits", Static).render()
        )
        assert "Keep this notice" in str(
            dialog.query_one("#petdex-credits", Static).render()
        )
        await dialog._prepare()
        assert dialog._prepared is not None
        assert dialog.query_one("#petdex-accept", Button).disabled
        for state in dialog._prepared_states:
            dialog.query_one("#petdex-preview-state", Select).value = state.name
            await dialog._show_preview()
        dialog.query_one("#petdex-reviewed", Checkbox).value = True
        await pilot.pause()
        assert not dialog.query_one("#petdex-accept", Button).disabled
        path = tmp_path / "reviewed.tldw-persona-vpack"
        path.write_bytes(dialog._prepared)
        snapshot = read_buddy_archive(path)
        assert snapshot.artwork["creator"] == "Example artist"
        assert "Keep this notice" in snapshot.artwork["notices"]
        assert "idle fallback" in str(
            dialog.query_one("#petdex-warnings", Static).render()
        )
        await dialog._accept()
        await pilot.pause()
        assert app.screen is not dialog
        assert accepted and accepted[0].source is source


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", ["source", "destination"])
async def test_stale_review_cannot_accept(monkeypatch, stale):
    from textual.widgets import Checkbox

    from tldw_chatbook.Petdex import sources
    from tldw_chatbook.Widgets.Persona_Widgets.petdex_import_review import (
        PetdexImportReviewDialog,
    )

    current = {"source": True, "destination": True}
    source = _source(guard=lambda: current["source"])
    monkeypatch.setattr(sources, "read_local_package", lambda path: source)
    dialog = PetdexImportReviewDialog(
        authority_guard=lambda: current["destination"], config={}
    )
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(dialog)
        await dialog._load("petdex-local")
        await dialog._prepare()
        assert dialog._prepared
        dialog.query_one("#petdex-reviewed", Checkbox).value = True
        current[stale] = False
        await dialog._accept()
        assert app.screen is dialog
        assert not dialog._review_closed
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_manual_v2_requires_explicit_list_and_mapping(monkeypatch):
    import json

    from textual.widgets import Select, Static

    from tldw_chatbook.Petdex import sources
    from tldw_chatbook.Widgets.Persona_Widgets.petdex_import_review import (
        PetdexImportReviewDialog,
    )

    monkeypatch.setattr(sources, "read_local_package", lambda path: _source(version=2))
    dialog = PetdexImportReviewDialog(authority_guard=lambda: True, config={})
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(dialog)
        await dialog._load("petdex-local")
        assert dialog.inspection.states == ()
        await dialog._prepare()
        assert dialog._prepared is None
        dialog.query_one("#petdex-states", TextArea).load_text(
            json.dumps(
                [
                    {
                        "name": "idle",
                        "row": 10,
                        "frames": 2,
                        "duration_ms": 1001,
                        "loop": True,
                    }
                ]
            )
        )
        dialog._set_state_options(dialog.collect_states())
        dialog.query_one("#petdex-map-thinking", Select).value = Select.NULL
        await dialog._prepare()
        assert "Select every required" in str(
            dialog.query_one("#petdex-status", Static).render()
        )
        dialog.query_one("#petdex-map-thinking", Select).value = "idle"
        await dialog._prepare()
        assert dialog._prepared
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_replacement_failure_clears_old_source_but_picker_cancel_preserves_it(
    monkeypatch,
):
    from unittest.mock import AsyncMock

    from textual.widgets import Checkbox, Input, Select, Static

    from tldw_chatbook.Petdex import sources
    from tldw_chatbook.Widgets.Persona_Widgets.petdex_import_review import (
        PetdexImportReviewDialog,
    )

    source = _source()
    monkeypatch.setattr(sources, "read_local_package", lambda _path: source)
    dialog = PetdexImportReviewDialog(authority_guard=lambda: True, config={})
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(90, 35)) as pilot:
        await app.push_screen(dialog)
        dialog.query_one("#petdex-source", Input).value = "first"
        await dialog._load("petdex-local")
        await dialog._prepare()
        prepared = dialog._prepared
        credits = str(dialog.query_one("#petdex-credits", Static).render())
        assert dialog.query_one("#petdex-preview")

        monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value=None))
        await dialog._load("petdex-choose")
        assert dialog.source is source
        assert dialog._prepared is prepared
        assert str(dialog.query_one("#petdex-credits", Static).render()) == credits
        assert dialog.query_one("#petdex-preview")

        def fail_replacement(_path):
            raise ValueError("replacement is invalid")

        monkeypatch.setattr(sources, "read_local_package", fail_replacement)
        dialog.query_one("#petdex-source", Input).value = "replacement"
        await dialog._load("petdex-local")

        assert dialog.source is None
        assert dialog.inspection is None
        assert dialog._prepared is None
        assert dialog._prepared_key is None
        assert dialog._prepared_states == ()
        assert str(dialog.query_one("#petdex-credits", Static).render()) == (
            "No source loaded."
        )
        assert not str(dialog.query_one("#petdex-layout", Static).render())
        assert dialog.query_one("#petdex-states", TextArea).text == "[]"
        assert dialog.query_one("#petdex-preview-state", Select).value is Select.NULL
        assert not str(dialog.query_one("#petdex-warnings", Static).render())
        assert not dialog.query_one("#petdex-reviewed", Checkbox).value
        assert not dialog.query("#petdex-preview")
        assert "replacement is invalid" in str(
            dialog.query_one("#petdex-status", Static).render()
        )
        await pilot.press("escape")


from Tests.UI.test_personas_persona_visual_authoring import (
    _open_editor,
    _Repository,
    local_scope,  # noqa: F401 - imported pytest fixture
    stub_characters,  # noqa: F401 - imported pytest fixture
)


@pytest.mark.asyncio
@pytest.mark.usefixtures("stub_characters", "local_scope")
@pytest.mark.parametrize("invalidated", [None, "source", "destination", "cancel"])
async def test_mounted_petdex_handoff_uses_native_draft_and_preserves_saved_pack(
    monkeypatch, mock_app_instance, tmp_path, invalidated
):
    from unittest.mock import Mock

    import tldw_chatbook.UI.Screens.personas_screen as screen_module
    from Tests.UI.test_personas_workbench import PersonasTestApp
    from tldw_chatbook.Persona_Visual.authoring import inspect_persona_visual_draft
    from tldw_chatbook.Petdex.conversion import build_petdex_archive
    from tldw_chatbook.Petdex.review import PetdexReviewedArchive, review_petdex_import

    monkeypatch.setattr(screen_module, "PersonaVisualRepository", _Repository)
    publish = Mock()
    monkeypatch.setattr(screen_module, "publish_persona_visual", publish)
    current = [True]
    source = _source(guard=lambda: current[0])
    reviewed = PetdexReviewedArchive(build_petdex_archive(source), source)
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(110, 45)) as pilot:
        screen = await _open_editor(pilot)
        original = screen._persona_visual_authoring
        draft = original.draft
        imported_paths = []
        import_native = screen._import_persona_visual_from_path

        async def observe_import(path, **kwargs):
            imported_paths.append(path)
            return await import_native(path, **kwargs)

        monkeypatch.setattr(screen, "_import_persona_visual_from_path", observe_import)

        async def return_review(dialog):
            if invalidated == "source":
                current[0] = False
            if invalidated == "destination":
                screen._persona_visual_generation += 1
            return None if invalidated == "cancel" else reviewed

        monkeypatch.setattr(app, "push_screen_wait", return_review)
        await review_petdex_import(screen)
        assert not screen._io_dialog_active
        publish.assert_not_called()
        if invalidated is None:
            assert original.dirty
            assert original.draft is not draft
            assert inspect_persona_visual_draft(original.draft).activatable
            assert original.import_review is not None
            assert imported_paths
            from pathlib import Path

            assert not Path(imported_paths[0]).exists()
        else:
            assert original.draft is draft
            assert not original.dirty
            assert imported_paths == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("stub_characters", "local_scope")
async def test_source_change_at_final_native_import_boundary_discards_only_new_review(
    monkeypatch, mock_app_instance, tmp_path
):
    import tldw_chatbook.UI.Screens.personas_screen as screen_module
    from Tests.UI.test_personas_workbench import PersonasTestApp
    from tldw_chatbook.Petdex.conversion import build_petdex_archive

    monkeypatch.setattr(screen_module, "PersonaVisualRepository", _Repository)
    archive = tmp_path / "pet.tldw-persona-vpack"
    archive.write_bytes(build_petdex_archive(_source()))
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(110, 45)) as pilot:
        screen = await _open_editor(pilot)
        state = screen._persona_visual_authoring
        draft = state.draft
        assert not await screen._import_persona_visual_from_path(
            str(archive), source_guard=lambda: False
        )
        assert state.draft is draft
        assert not state.dirty
        assert state.import_review is None
        assert archive.exists()


@pytest.mark.parametrize("failure", ["replace", "authority", "changed-target"])
def test_native_export_failure_preserves_existing_file_and_cleans_temporary(
    monkeypatch, tmp_path, failure
):
    from tldw_chatbook.Petdex import review

    output_folder = tmp_path / "exports"
    output_folder.mkdir()
    target = output_folder / "buddy.tldw-persona-vpack"
    target.write_bytes(b"existing export")
    identity = review.export_target_identity(target)
    if failure == "replace":

        def fail_replace(*args):
            raise OSError("write failed")

        monkeypatch.setattr(review.os, "replace", fail_replace)
    if failure == "changed-target":
        target.write_bytes(b"newer output")
    with pytest.raises((OSError, ValueError)):
        review.write_native_export(
            target,
            b"new export",
            expected_identity=identity,
            authority_guard=lambda: failure != "authority",
        )
    assert target.read_bytes() == (
        b"newer output" if failure == "changed-target" else b"existing export"
    )
    assert list(output_folder.iterdir()) == [target]


def test_native_export_atomically_replaces_approved_target(tmp_path):
    from tldw_chatbook.Petdex import review

    output_folder = tmp_path / "exports"
    output_folder.mkdir()
    target = output_folder / "buddy.tldw-persona-vpack"
    target.write_bytes(b"existing export")
    review.write_native_export(
        target,
        b"reviewed export",
        expected_identity=review.export_target_identity(target),
        authority_guard=lambda: True,
    )
    assert target.read_bytes() == b"reviewed export"
    assert list(output_folder.iterdir()) == [target]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["review_petdex_import", "export_native_buddy"])
@pytest.mark.parametrize("failure", ["root", "service"])
async def test_destination_capture_failure_releases_dialog_slot(
    monkeypatch, operation, failure
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tldw_chatbook.Petdex import review
    from tldw_chatbook.Utils import paths

    def unavailable():
        raise OSError("destination unavailable")

    class UnavailableApp:
        @property
        def character_persona_scope_service(self):
            raise RuntimeError("service unavailable")

    if failure == "root":
        monkeypatch.setattr(paths, "get_user_data_dir", unavailable)
    screen = SimpleNamespace(
        _persona_visual_authoring=None,
        _io_dialog_active=True,
        app_instance=UnavailableApp(),
        _notify=Mock(),
    )
    await getattr(review, operation)(screen)
    assert not screen._io_dialog_active
    screen._notify.assert_called_once()


@pytest.mark.asyncio
async def test_missing_webp_encoder_previews_png_and_discloses_motion_fallback(
    monkeypatch,
):
    from PIL import features
    from textual.widgets import Static

    from tldw_chatbook.Petdex import sources
    from tldw_chatbook.Widgets.Persona_Widgets.petdex_import_review import (
        PetdexImportReviewDialog,
    )

    source = _source()
    monkeypatch.setattr(sources, "read_local_package", lambda path: source)
    original_check = features.check
    monkeypatch.setattr(
        features,
        "check",
        lambda name: False if name == "webp" else original_check(name),
    )
    app = ConsolidatedCSSApp()
    dialog = PetdexImportReviewDialog(authority_guard=lambda: True, config={})
    async with app.run_test(size=(90, 35)) as pilot:
        await app.push_screen(dialog)
        await dialog._load("petdex-local")
        await dialog._prepare()
        assert dialog._prepared is not None
        assert dialog.query_one("#petdex-preview")
        assert "first frame" in str(dialog.query_one("#petdex-status", Static).render())
        assert "WebP" in str(dialog.query_one("#petdex-status", Static).render())
        await pilot.press("escape")


def test_preview_closes_cropped_frames_when_encoding_fails(monkeypatch):
    from PIL import Image

    from tldw_chatbook.Petdex.conversion import inspect_petdex
    from tldw_chatbook.Petdex.review import preview_state

    source = _source()
    inspection = inspect_petdex(source)
    original_crop = Image.Image.crop
    original_convert = Image.Image.convert
    frames = []

    def observe_crop(image, box):
        result = original_crop(image, box)
        frames.append(result)
        return result

    def observe_convert(image, *args, **kwargs):
        result = original_convert(image, *args, **kwargs)
        frames.append(result)
        return result

    def fail_save(*args, **kwargs):
        raise OSError("encoder failure")

    monkeypatch.setattr(Image.Image, "crop", observe_crop)
    monkeypatch.setattr(Image.Image, "convert", observe_convert)
    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(OSError):
        preview_state(source, inspection, inspection.states[0])
    assert frames
    for frame in frames:
        with pytest.raises(ValueError, match="closed image"):
            frame.getpixel((0, 0))
