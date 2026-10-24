"""Mounted local-source authoring, actual event routing and atomic handoff."""

# ruff: noqa: F811 -- imported pytest fixtures are parameter names

from io import BytesIO

import pytest
from PIL import Image
from textual.pilot import Pilot
from textual.widgets import Button, Collapsible, Static

from Tests.UI.test_personas_expression_slots import (
    expr_db,  # noqa: F401 -- pytest fixture dependency
    personas_editor_with_bound_pack,  # noqa: F401 -- pytest fixture re-export
    personas_editor_with_saved_character,  # noqa: F401 -- pytest fixture re-export
)
from tldw_chatbook.Character_Chat.portrait_performance import FRAME_ROLES
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.portrait_performance_editor import (
    CharacterPerformanceRequested,
    PortraitPerformanceEditor,
)

pytestmark = pytest.mark.bootstrap_profile


def png(color="red", size=(12, 12)):
    with Image.new("RGB", size, color) as image, BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()


def saved_source(fixture):
    app, screen, db, character_id = fixture
    screen.app_instance.character_persona_scope_service = None
    record = db.get_character_card_by_id(character_id)
    assert db.update_character_card(character_id, {"image": png()}, record["version"])
    editor = screen.query_one(PersonasCharacterEditorWidget)
    editor.load_character(db.get_character_card_by_id(character_id))
    assert screen._local_character_actions_allowed()
    assert screen._character_editor_is_active()
    assert screen.state.selected_entity_kind == "character"
    assert str(screen.state.selected_entity_id) == str(character_id)
    assert screen.app_instance.chachanotes_db is db
    assert screen.app_instance.character_persona_scope_service is None
    return app, screen, db, character_id, editor


@pytest.mark.asyncio
async def test_keyboard_prepare_import_apply_remove_and_reopen(
    personas_editor_with_saved_character, tmp_path
):
    app, screen, db, character_id, editor = saved_source(
        personas_editor_with_saved_character
    )
    view = editor.query_one(PortraitPerformanceEditor)
    view.query_one(Collapsible).collapsed = False
    view.query_one("#portrait-performance-prepare", Button).focus()
    await Pilot(app).press("enter")
    await app.workers.wait_for_complete()
    controller = screen._portrait_performance()
    assert controller.draft is not None, str(
        view.query_one("#portrait-performance-status", Static).renderable
    )
    for index, role in enumerate(FRAME_ROLES):
        path = tmp_path / f"{role}.png"
        path.write_bytes(png((index * 30, 0, 0)))
        assert await controller.handle(
            CharacterPerformanceRequested("import", role=role), path=str(path)
        )
    assert not view.query_one("#portrait-performance-apply", Button).disabled
    assert await controller.handle(CharacterPerformanceRequested("apply"))
    record = db.get_character_portrait_performance(
        character_id=character_id, expression_key="neutral"
    )
    assert record.active and record.revision == 1
    controller.discard()
    assert await controller.handle(CharacterPerformanceRequested("prepare")), str(
        editor.query_one("#portrait-performance-status", Static).renderable
    )
    assert controller.draft.frames() == record.frames
    assert await controller.handle(CharacterPerformanceRequested("remove"))
    assert not db.get_character_portrait_performance(
        character_id=character_id, expression_key="neutral"
    ).active
    assert db.get_character_card_by_id(character_id)["image"] == png()


@pytest.mark.asyncio
async def test_source_gates_do_not_save_text_or_cross_server_identity(
    personas_editor_with_saved_character,
):
    _, screen, db, character_id, editor = saved_source(
        personas_editor_with_saved_character
    )
    controller = screen._portrait_performance()
    editor._area("description").text = "Unrelated unsaved text"
    editor.set_avatar_image(png("blue"))
    assert not await controller.handle(CharacterPerformanceRequested("prepare"))
    assert controller.draft is None
    editor.discard_unsaved_attachment()
    assert await controller.handle(CharacterPerformanceRequested("prepare")), str(
        editor.query_one("#portrait-performance-status", Static).renderable
    )
    assert (
        db.get_character_card_by_id(character_id)["description"]
        != "Unrelated unsaved text"
    )
    screen.state = __import__("dataclasses").replace(
        screen.state, runtime_source="server"
    )
    assert not await controller.handle(CharacterPerformanceRequested("apply"))
    controller._check_context()
    assert controller.draft is None
    assert (
        db.get_character_portrait_performance(
            character_id=character_id, expression_key="neutral"
        )
        is None
    )


@pytest.mark.asyncio
async def test_bad_import_and_failed_apply_keep_current_artwork(
    personas_editor_with_saved_character, tmp_path, monkeypatch
):
    _, screen, db, character_id, editor = saved_source(
        personas_editor_with_saved_character
    )
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare")), str(
        editor.query_one("#portrait-performance-status", Static).renderable
    )
    path = tmp_path / "pose.png"
    for role in FRAME_ROLES:
        path.write_bytes(png())
        assert await controller.handle(
            CharacterPerformanceRequested("import", role=role), path=str(path)
        )
    before = controller.draft.frames()
    path.write_bytes(png(size=(13, 12)))
    assert not await controller.handle(
        CharacterPerformanceRequested("import", role="small"), path=str(path)
    )
    assert controller.draft.frames() == before

    def fail(*args, **kwargs):
        raise RuntimeError("injected transaction failure")

    monkeypatch.setattr(db, "apply_character_portrait_performance", fail)
    assert not await controller.handle(CharacterPerformanceRequested("apply"))
    assert controller.draft.frames() == before
    assert not controller._busy
    assert (
        db.get_character_portrait_performance(
            character_id=character_id, expression_key="neutral"
        )
        is None
    )
    await controller.handle(CharacterPerformanceRequested("cancel"))
    assert controller.draft is None


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cancel", "profile", "source"])
async def test_held_import_never_retargets_or_stages_after_change(
    personas_editor_with_saved_character, tmp_path, monkeypatch, change
):
    import asyncio
    from threading import Event
    from types import SimpleNamespace

    import tldw_chatbook.UI.Persona_Modules.portrait_performance as authoring
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    _, screen, db, character_id, _ = saved_source(personas_editor_with_saved_character)
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    path = tmp_path / "held.png"
    path.write_bytes(png())
    entered, release = Event(), Event()
    original = authoring.normalize_performance_frame

    def held(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(authoring, "normalize_performance_frame", held)
    task = asyncio.create_task(
        controller.handle(CharacterPerformanceRequested("import"), path=str(path))
    )
    other = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert not await controller.handle(
            CharacterPerformanceRequested("import"), path=str(path)
        )
        if change == "cancel":
            await controller.handle(CharacterPerformanceRequested("cancel"))
        elif change == "profile":
            other = CharactersRAGDB(tmp_path / "replacement.db", "other-client")
            screen.app_instance.chachanotes_db = other
            screen.app_instance.character_persona_scope_service = SimpleNamespace(
                local_service=SimpleNamespace(db=other)
            )
            controller._check_context()
        else:
            record = db.get_character_card_by_id(character_id)
            assert db.update_character_card(
                character_id, {"image": png("blue")}, record["version"]
            )
        release.set()
        assert not await task
        assert controller.draft is None
        assert (
            db.get_character_portrait_performance(
                character_id=character_id, expression_key="neutral"
            )
            is None
        )
    finally:
        release.set()
        await task
        screen.app_instance.chachanotes_db = db
        screen.app_instance.character_persona_scope_service = None
        if other is not None:
            other.close_connection()


@pytest.mark.asyncio
async def test_staged_bound_pack_requires_explicit_apply(
    personas_editor_with_bound_pack,
):
    from tldw_chatbook.Widgets.Persona_Widgets.personas_visual_identity_pack_widget import (
        PersonasVisualIdentityPackWidget,
    )

    _, screen, _db, _id, _calls = personas_editor_with_bound_pack
    screen.app_instance.character_persona_scope_service = None
    editor = screen.query_one(PersonasCharacterEditorWidget)
    browser = editor.query_one(PersonasVisualIdentityPackWidget)
    browser.set_staged_change(browser.pack.assets[0].expression_key, "replace")
    controller = screen._portrait_performance()
    assert not await controller.handle(CharacterPerformanceRequested("prepare"))
    assert controller.draft is None
    assert "Apply or discard" in str(
        editor.query_one("#portrait-performance-status", Static).renderable
    )


@pytest.mark.asyncio
async def test_keyboard_cancel_import_picker_restores_focus(
    personas_editor_with_saved_character,
):
    app, screen, _db, _id, editor = saved_source(personas_editor_with_saved_character)
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    view = editor.query_one(PortraitPerformanceEditor)
    view.query_one(Collapsible).collapsed = False
    button = view.query_one("#portrait-performance-import", Button)
    button.focus()
    pilot = Pilot(app)
    await pilot.press("enter")
    await pilot.pause()
    assert app.screen is not screen
    await pilot.press("escape")
    await app.workers.wait_for_complete()
    await pilot.pause()
    assert app.screen is screen
    assert button.has_focus
    assert controller.draft.staged_roles == ()


@pytest.mark.asyncio
async def test_reduce_motion_allows_manual_still_review_without_a_timer(
    personas_editor_with_saved_character,
):
    from textual.widgets import Select

    from Tests.Chat.test_portrait_performance_state import performance_record
    from Tests.UI.test_character_expression_avatar import wait_ready
    from tldw_chatbook.Widgets.Console.character_expression_avatar import (
        CharacterExpressionAvatar,
    )

    app, screen, _db, _id, editor = saved_source(personas_editor_with_saved_character)
    app.app_config.setdefault("appearance", {})["reduce_motion"] = True
    editor.query_one(PortraitPerformanceEditor).query_one(Collapsible).collapsed = False
    await Pilot(app).pause()
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    for frame in performance_record().frames:
        controller.draft.stage(frame)
    editor.query_one("#portrait-performance-preview-mode", Select).value = "blink_open"
    await Pilot(app).pause()
    await app.workers.wait_for_complete()
    avatar = editor.query_one("#portrait-performance-staged", CharacterExpressionAvatar)
    await wait_ready(avatar, Pilot(app))
    assert avatar._timer is None
    assert avatar.current_image.getpixel((3, 3)) == (255, 0, 255, 255)


@pytest.mark.asyncio
async def test_preview_failure_after_commit_reports_the_saved_result(
    personas_editor_with_saved_character, monkeypatch
):
    from Tests.Chat.test_portrait_performance_state import performance_record

    _, screen, db, character_id, editor = saved_source(
        personas_editor_with_saved_character
    )
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    for frame in performance_record().frames:
        controller.draft.stage(frame)

    async def unavailable(*args, **kwargs):
        raise RuntimeError("preview failed after commit")

    monkeypatch.setattr(
        editor.query_one(PortraitPerformanceEditor), "show_draft", unavailable
    )
    assert await controller.handle(CharacterPerformanceRequested("apply"))
    assert db.get_character_portrait_performance(
        character_id=character_id, expression_key="neutral"
    ).active
    assert "saved; preview unavailable" in str(
        editor.query_one("#portrait-performance-status", Static).renderable
    )


@pytest.mark.asyncio
async def test_narrow_review_stacks_images_and_keeps_keyboard_prepare(
    personas_editor_with_saved_character,
):
    app, screen, _db, _id, editor = saved_source(personas_editor_with_saved_character)
    pilot = Pilot(app)
    await pilot.resize_terminal(80, 30)
    view = editor.query_one(PortraitPerformanceEditor)
    view.query_one(Collapsible).collapsed = False
    await pilot.pause()
    assert view.has_class("portrait-review-compact")
    view.query_one("#portrait-performance-prepare", Button).focus()
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    assert screen._portrait_performance().draft is not None


@pytest.mark.asyncio
async def test_large_source_keeps_manual_authoring_and_removal_available(
    personas_editor_with_saved_character, tmp_path
):
    from tldw_chatbook.Character_Chat.portrait_performance import (
        capture_performance_source,
        normalize_performance_frames,
    )

    _, screen, db, cid, editor = saved_source(personas_editor_with_saved_character)
    original = png(size=(4096, 4096))
    card = db.get_character_card_by_id(cid)
    assert db.update_character_card(cid, {"image": original}, card["version"])
    editor.load_character(db.get_character_card_by_id(cid))
    source, _ = capture_performance_source(db, character_id=cid, requested_state="idle")
    frames = normalize_performance_frames({role: png() for role in FRAME_ROLES})
    db.apply_character_portrait_performance(source, frames, expected_revision=0)
    controller = screen._portrait_performance()
    view = editor.query_one(PortraitPerformanceEditor)
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    assert controller.draft.frames() == frames
    assert not view.query_one("#portrait-performance-import", Button).disabled
    assert not view.query_one("#portrait-performance-remove", Button).disabled
    assert view.query_one("#portrait-performance-regions", Button).disabled
    assert view.query_one("#portrait-performance-generate", Button).disabled
    assert controller._reference is None
    assert not await controller.handle(CharacterPerformanceRequested("generate"))
    path = tmp_path / "manual.png"
    path.write_bytes(png("blue"))
    assert await controller.handle(
        CharacterPerformanceRequested("import", role="small"), path=str(path)
    )
    assert await controller.handle(CharacterPerformanceRequested("apply"))
    current = db.get_character_portrait_performance(
        character_id=cid, expression_key="neutral"
    )
    assert current.revision == 2 and current.frames[1].png != frames[1].png
    controller.discard()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    assert controller.draft.frames() == current.frames
    assert await controller.handle(CharacterPerformanceRequested("remove"))
    assert not db.get_character_portrait_performance(
        character_id=cid, expression_key="neutral"
    ).active
    assert db.get_character_card_by_id(cid)["image"] == original
