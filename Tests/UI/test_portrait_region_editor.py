"""Real mounted source-coordinate selection and independent region settings Save."""

# ruff: noqa: F811 -- shared pytest fixture re-exports
from dataclasses import replace

import pytest
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static

from Tests.Character_Chat.test_portrait_compositing import scene
from Tests.UI.test_personas_expression_slots import (
    expr_db,  # noqa: F401
    personas_editor_with_saved_character,  # noqa: F401
)
from Tests.UI.test_personas_portrait_performance import saved_source
from tldw_chatbook.Character_Chat.portrait_performance import capture_performance_source
from tldw_chatbook.Widgets.Persona_Widgets.portrait_performance_editor import (
    CharacterPerformanceRequested,
    PortraitPerformanceEditor,
)

pytestmark = pytest.mark.bootstrap_profile


def region_ui():
    from tldw_chatbook.Widgets.Persona_Widgets import portrait_region_editor

    return portrait_region_editor


def test_contain_mapping_refuses_padding_and_accounts_for_cell_aspect():
    ui = region_ui()
    transform = ui.RegionTransform.contain((100, 100), (80, 30))
    assert (transform.x, transform.y, transform.cols, transform.lines) == (
        10,
        0,
        60,
        30,
    )
    assert transform.source_point(9, 0) is None
    assert transform.source_point(70, 29) is None
    assert transform.source_point(10, 0) == (0, 0)
    assert transform.source_point(69, 29) == (99, 99)
    wide = ui.RegionTransform.contain((200, 100), (80, 30))
    assert (wide.x, wide.y, wide.cols, wide.lines) == (0, 5, 80, 20)
    assert wide.source_point(0, 4) is None


async def prepare(fixture):
    app, screen, db, cid, editor = saved_source(fixture)
    base, recipe, _components = scene()
    card = db.get_character_card_by_id(cid)
    assert db.update_character_card(cid, {"image": base}, card["version"])
    editor.load_character(db.get_character_card_by_id(cid))
    view = editor.query_one(PortraitPerformanceEditor)
    view.query_one(Collapsible).collapsed = False
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    return app, screen, db, cid, view, controller, recipe


@pytest.mark.asyncio
async def test_keyboard_confirmation_save_cancel_and_reopen(
    personas_editor_with_saved_character,
):
    import asyncio

    from textual.pilot import Pilot

    app, _, db, cid, _view, controller, recipe = await prepare(
        personas_editor_with_saved_character
    )
    task = asyncio.create_task(
        controller._screen()
        .run_worker(controller.handle(CharacterPerformanceRequested("regions")))
        .wait()
    )
    pilot = Pilot(app)
    try:
        async with asyncio.timeout(5):
            while not isinstance(app.screen, region_ui().PortraitRegionEditor):
                await pilot.pause()
        modal = app.screen
        assert modal.query_one("#portrait-regions-save", Button).disabled
        for role in ("left_eye", "right_eye", "mouth"):
            modal.query_one("#portrait-regions-role", Select).value = role
            await pilot.pause()
            rect = getattr(recipe, role)
            for key in ("x", "y", "width", "height"):
                field = modal.query_one(f"#portrait-region-{key}", Input)
                field.focus()
                await pilot.press("home", "shift+end")
                await pilot.press(*str(getattr(rect, key)))
            await pilot.pause()
        modal.query_one("#portrait-regions-neutral", Checkbox).value = True
        await pilot.pause()
        assert not modal.query_one("#portrait-regions-save", Button).disabled
        modal.query_one("#portrait-regions-save", Button).focus()
        await pilot.press("enter")
        assert await task
        settings = db.get_character_portrait_regions(
            character_id=cid, expression_key="neutral"
        )
        assert settings.recipe == recipe and settings.revision == 1
        assert controller.draft.composition.components == ()
        assert (
            controller.draft.staged_roles == ("rest",)
            and not controller.draft.can_apply
        )
        assert (
            db.get_character_portrait_performance(
                character_id=cid, expression_key="neutral"
            )
            is None
        )
        controller.discard()
        assert await controller.handle(CharacterPerformanceRequested("prepare"))
        assert controller._regions.recipe == recipe
        again = asyncio.create_task(
            controller._screen()
            .run_worker(controller.handle(CharacterPerformanceRequested("regions")))
            .wait()
        )
        async with asyncio.timeout(5):
            while not isinstance(app.screen, region_ui().PortraitRegionEditor):
                await pilot.pause()
        assert app.screen.query_one("#portrait-regions-neutral", Checkbox).value
        app.screen.action_cancel()
        assert not await again
        assert (
            db.get_character_portrait_regions(
                character_id=cid, expression_key="neutral"
            ).revision
            == 1
        )
    finally:
        if not task.done():
            if isinstance(app.screen, region_ui().PortraitRegionEditor):
                app.screen.action_cancel()
            await task


@pytest.mark.asyncio
async def test_changed_save_resets_only_draft_and_stops_exact_sample(
    personas_editor_with_saved_character,
):
    import asyncio

    from Tests.Character_Chat.test_portrait_compositing import core

    _, _, db, cid, _, controller, recipe = await prepare(
        personas_editor_with_saved_character
    )
    base, _, components = scene()
    source, _ = capture_performance_source(db, character_id=cid, requested_state="idle")
    rev = db.save_character_portrait_regions(source, recipe, expected_revision=0)
    frames = core().assemble_components(base, recipe, components)
    db.apply_character_portrait_performance(
        source,
        frames,
        expected_revision=0,
        composition_recipe=recipe,
        expected_regions_revision=rev,
    )
    controller.discard()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    before = db.get_character_portrait_performance(
        character_id=cid, expression_key="neutral"
    )
    task = controller._speech_task = asyncio.create_task(asyncio.sleep(30))
    await asyncio.sleep(0)
    authority, epoch = controller._authority, controller._epoch
    changed = replace(recipe, mouth=replace(recipe.mouth, width=19))
    assert await controller._save_regions(authority, controller.draft, epoch, changed)
    assert task.cancelled()
    assert controller.draft.composition.recipe == changed
    assert (
        controller.draft.composition.components == ()
        and controller.draft.staged_roles == ("rest",)
    )
    assert (
        db.get_character_portrait_performance(
            character_id=cid, expression_key="neutral"
        )
        == before
    )


@pytest.mark.asyncio
async def test_stale_region_save_cannot_write_replacement_editor(
    personas_editor_with_saved_character,
):
    _, _, db, cid, _, controller, recipe = await prepare(
        personas_editor_with_saved_character
    )
    authority, draft, epoch = controller._authority, controller.draft, controller._epoch
    controller.discard()
    assert not await controller._save_regions(authority, draft, epoch, recipe)
    assert (
        db.get_character_portrait_regions(character_id=cid, expression_key="neutral")
        is None
    )


@pytest.mark.asyncio
async def test_unchanged_save_keeps_revision_candidate_and_sample(
    personas_editor_with_saved_character,
):
    import asyncio

    _, _, db, cid, _, controller, recipe = await prepare(
        personas_editor_with_saved_character
    )
    assert await controller._save_regions(
        controller._authority, controller.draft, controller._epoch, recipe
    )
    before = controller.draft.composition
    task = controller._speech_task = asyncio.create_task(asyncio.sleep(30))
    try:
        assert await controller._save_regions(
            controller._authority, controller.draft, controller._epoch, recipe
        )
        assert controller.draft.composition is before and not task.cancelled()
        assert (
            db.get_character_portrait_regions(
                character_id=cid, expression_key="neutral"
            ).revision
            == 1
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cancel", "profile", "source", "settings"])
async def test_held_save_rejects_retired_authority_and_stale_revision(
    personas_editor_with_saved_character, monkeypatch, tmp_path, change
):
    import asyncio
    from threading import Event

    from tldw_chatbook.DB import portrait_region_settings
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Persona_Modules import portrait_regions as authoring

    _, screen, db, cid, _, controller, recipe = await prepare(
        personas_editor_with_saved_character
    )
    entered, release = Event(), Event()
    original = authoring.assemble_components

    def held(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(authoring, "assemble_components", held)
    task = asyncio.create_task(
        controller._save_regions(
            controller._authority, controller.draft, controller._epoch, recipe
        )
    )
    other = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if change == "cancel":
            controller.discard()
        elif change == "profile":
            other = CharactersRAGDB(tmp_path / "replacement.db", "other-profile")
            screen.app_instance.chachanotes_db = other
            controller._check_context()
        elif change == "source":
            card = db.get_character_card_by_id(cid)
            db.update_character_card(cid, {"image": b"changed"}, card["version"])
        else:
            portrait_region_settings.save_regions(
                db,
                controller.draft.source,
                replace(recipe, mouth=replace(recipe.mouth, width=19)),
                expected_revision=0,
            )
        release.set()
        assert not await task
        current = db.get_character_portrait_regions(
            character_id=cid, expression_key="neutral"
        )
        assert (
            current is None
            if change != "settings"
            else current.recipe.mouth.width == 19
        )
        if other:
            assert (
                other.execute_query(
                    "SELECT count(*) FROM character_portrait_regions"
                ).fetchone()[0]
                == 0
            )
    finally:
        release.set()
        await task
        screen.app_instance.chachanotes_db = db
        if other:
            other.close()


@pytest.mark.asyncio
async def test_pointer_padding_resize_and_narrow_keyboard_geometry(
    personas_editor_with_saved_character, tmp_path
):
    import asyncio

    from textual.pilot import Pilot

    app, _, _, _, _, controller, recipe = await prepare(
        personas_editor_with_saved_character
    )
    assert await controller._save_regions(
        controller._authority, controller.draft, controller._epoch, recipe
    )
    task = asyncio.create_task(
        controller._screen()
        .run_worker(controller.handle(CharacterPerformanceRequested("regions")))
        .wait()
    )
    pilot = Pilot(app)
    try:
        async with asyncio.timeout(5):
            while not isinstance(app.screen, region_ui().PortraitRegionEditor):
                await pilot.pause()
        modal = app.screen
        async with asyncio.timeout(5):
            while modal._transform is None:
                await pilot.pause()
        transform = modal._transform
        before = dict(modal._rectangles)
        assert transform.x > 0
        await pilot.click("#portrait-regions-source", offset=(0, 0))
        assert modal._corner is None and modal._rectangles == before
        await pilot.click("#portrait-regions-source", offset=(transform.x, transform.y))
        assert modal._corner == (0, 0)
        await pilot.click(
            "#portrait-regions-source",
            offset=(
                transform.x + transform.cols // 4,
                transform.y + transform.lines // 4,
            ),
        )
        assert modal._rectangles["left_eye"].x == modal._rectangles["left_eye"].y == 0
        selected = dict(modal._rectangles)
        await pilot.resize_terminal(48, 34)
        await pilot.pause()
        assert modal._rectangles == selected and modal.has_class("region-compact")
        for key in ("x", "y", "width", "height"):
            field = modal.query_one(f"#portrait-region-{key}", Input)
            field.focus()
            await pilot.pause()
            assert field.region.width >= 8
        confirmation = modal.query_one("#portrait-regions-neutral", Checkbox)
        assert len(confirmation.label.plain) + 8 <= modal.size.width
        save = modal.query_one("#portrait-regions-save", Button)
        save.focus()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert (
            app.focused is save
            and save.region.y >= 0
            and save.region.bottom <= app.size.height
        )
        modal.action_cancel()
        assert not await task
    finally:
        if not task.done():
            if isinstance(app.screen, region_ui().PortraitRegionEditor):
                app.screen.action_cancel()
            await task


def test_mask_preview_uses_original_crop_and_compositor_mask(monkeypatch):
    from tldw_chatbook.Character_Chat.portrait_compositing import _mask, _workspace

    base, recipe, _ = scene()
    ui = region_ui()
    seen = []
    original = ui.mosaic_from_image

    def capture(image, *args, **kwargs):
        seen.append((image.mode, image.size, image.tobytes()))
        return original(image, *args, **kwargs)

    monkeypatch.setattr(ui, "mosaic_from_image", capture)
    ui._render(
        base,
        recipe,
        {key: getattr(recipe, key) for _, key in ui.REGIONS},
        "mouth",
        (50, 12),
        False,
    )
    with _workspace(base, 24) as (stack, image):
        rect = recipe.mouth
        bounds = (rect.x, rect.y, rect.x + rect.width, rect.y + rect.height)
        crop = image.crop(bounds)
        mask = _mask(recipe, eyes=False, stack=stack).crop(bounds)
        try:
            assert seen[1] == ("RGBA", crop.size, crop.tobytes())
            assert seen[2] == ("L", mask.size, mask.tobytes())
        finally:
            crop.close()
            mask.close()


@pytest.mark.asyncio
async def test_retired_modal_drops_held_preview(
    personas_editor_with_saved_character, monkeypatch
):
    import asyncio
    from threading import Event

    from textual.pilot import Pilot

    app, _, _, _, _, controller, _ = await prepare(personas_editor_with_saved_character)
    entered, release = Event(), Event()
    ui = region_ui()
    original = ui._render

    def held(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(ui, "_render", held)
    task = asyncio.create_task(
        controller._screen()
        .run_worker(controller.handle(CharacterPerformanceRequested("regions")))
        .wait()
    )
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        modal = app.screen
        source = modal.query_one("#portrait-regions-source", Static)
        preview = modal._preview_task
        modal.action_cancel()
        before = source.renderable
        assert not await task
        release.set()
        await preview
        await Pilot(app).pause()
        assert (
            modal._retired and modal._transform is None and source.renderable is before
        )
    finally:
        release.set()
        if isinstance(app.screen, ui.PortraitRegionEditor):
            app.screen.action_cancel()
        await task


@pytest.mark.asyncio
async def test_unknown_saved_algorithm_allows_confirmed_reselection(
    personas_editor_with_saved_character,
):
    import json

    from tldw_chatbook.DB.portrait_region_settings import serialize_recipe

    _, _, db, cid, _, controller, recipe = await prepare(
        personas_editor_with_saved_character
    )
    assert await controller._save_regions(
        controller._authority, controller.draft, controller._epoch, recipe
    )
    raw = json.loads(serialize_recipe(recipe))
    raw["algorithm_version"] = 999
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE character_portrait_regions SET recipe_json=?", (json.dumps(raw),)
        )
    controller.discard()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    assert controller._regions is None and controller._regions_revision == 1
    assert await controller._save_regions(
        controller._authority, controller.draft, controller._epoch, recipe
    )
    assert (
        db.get_character_portrait_regions(
            character_id=cid, expression_key="neutral"
        ).revision
        == 2
    )


def test_preview_budget_refusal_retains_foreign_reservation():
    from tldw_chatbook.Chat import character_expression_playback as playback

    base, recipe, _ = scene()
    before = playback.preparation_bytes()
    held = playback.MAX_PREPARATION_BYTES - before
    playback._adjust_budget(held)
    try:
        with pytest.raises(ValueError, match="budget"):
            region_ui()._render(
                base,
                recipe,
                {key: getattr(recipe, key) for _, key in region_ui().REGIONS},
                "mouth",
                (50, 12),
                False,
            )
        assert playback.preparation_bytes() == before + held
    finally:
        playback._adjust_budget(-held)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,change,covered",
    [
        ("review", "regions", False),
        ("review", "source", False),
        ("regions", "regions", False),
        ("regions", "source", False),
        ("review", "regions", True),
    ],
)
async def test_open_dialog_retires_after_another_editor_changes_saved_artwork(
    personas_editor_with_saved_character, action, change, covered
):
    import asyncio

    from textual.pilot import Pilot
    from textual.screen import ModalScreen

    from Tests.Character_Chat.test_portrait_compositing import core, png
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Widgets.Persona_Widgets.portrait_composition_review import (
        PortraitCompositionReview,
    )

    app, screen, db, cid, _, controller, recipe = await prepare(
        personas_editor_with_saved_character
    )
    source = controller.draft.source
    assert await controller._save_regions(
        controller._authority, controller.draft, controller._epoch, recipe
    )
    base, _, components = scene()
    frames = core().assemble_components(base, recipe, components)
    db.apply_character_portrait_performance(
        source,
        frames,
        expected_revision=0,
        composition_recipe=recipe,
        expected_regions_revision=1,
    )
    controller.discard()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    before = db.get_character_portrait_performance(
        character_id=cid, expression_key="neutral"
    )
    kind = (
        PortraitCompositionReview
        if action == "review"
        else region_ui().PortraitRegionEditor
    )
    task = asyncio.create_task(
        screen.run_worker(
            controller.handle(CharacterPerformanceRequested(action))
        ).wait()
    )
    other = None
    cover = None
    try:
        async with asyncio.timeout(5):
            while not isinstance(app.screen, kind):
                await Pilot(app).pause()
        modal = app.screen
        await Pilot(app).pause()
        if action == "review":
            await modal.wait_for_render()
            assert modal._raster is not None
        canvas = modal.query_one(
            "#portrait-pairs-canvas"
            if action == "review"
            else "#portrait-regions-source",
            Static,
        )
        if covered:
            cover = ModalScreen()
            await app.push_screen(cover)
        other = CharactersRAGDB(db.db_path, "other-editor")
        if change == "regions":
            assert (
                other.save_character_portrait_regions(
                    source,
                    replace(
                        recipe,
                        mouth=replace(recipe.mouth, width=recipe.mouth.width - 1),
                    ),
                    expected_revision=1,
                )
                == 2
            )
        else:
            card = other.get_character_card_by_id(cid)
            assert other.update_character_card(
                cid, {"image": png((0, 0, 255, 255))}, card["version"]
            )
        async with asyncio.timeout(4):
            while not modal._retired:
                await Pilot(app).pause(0.05)
        assert controller.draft is None
        if action == "review":
            assert modal._raster is None
            assert str(canvas.renderable) == "Review retired."
        else:
            assert modal._transform is None
            assert str(canvas.renderable) == "Region editor retired."
        assert (
            db.get_character_portrait_performance(
                character_id=cid, expression_key="neutral"
            )
            == before
        )
        if cover:
            assert app.screen is cover
            cover.dismiss(None)
            await Pilot(app).pause()
        await task
        assert modal not in app.screen_stack
    finally:
        if cover is not None and app.screen is cover:
            cover.dismiss(None)
        controller.discard()
        await task
        if other is not None:
            other.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["replacement", "close", "read_error"])
async def test_dialog_validation_is_coalesced_owned_and_drained(
    personas_editor_with_saved_character, monkeypatch, change
):
    import asyncio
    from threading import Event

    from textual.pilot import Pilot

    app, screen, _, _, _, controller, _ = await prepare(
        personas_editor_with_saved_character
    )
    draft = controller.draft
    entered, release = Event(), Event()
    original = controller._verify_source
    calls = []

    def held(*args):
        calls.append(args)
        entered.set()
        assert release.wait(5)
        if change == "read_error":
            raise OSError("unavailable native read")
        return original(*args)

    monkeypatch.setattr(controller, "_verify_source", held)
    action = asyncio.create_task(
        screen.run_worker(
            controller.handle(CharacterPerformanceRequested("regions"))
        ).wait()
    )
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        for _ in range(20):
            controller._check_context()
        assert len(calls) == 1
        check = controller._review_check_task
        if change == "replacement":
            controller.discard()
            await action
            assert await controller.handle(CharacterPerformanceRequested("prepare"))
            replacement = controller.draft
            release.set()
            await check
            assert controller.draft is replacement and replacement is not draft
        elif change == "close":
            closing = asyncio.create_task(controller.close())
            await Pilot(app).pause(0.05)
            assert not closing.done()
            release.set()
            await closing
            assert check.done() and controller.draft is None
        else:
            modal = app.screen
            release.set()
            await check
            await action
            assert modal._retired and controller.draft is draft
    finally:
        release.set()
        await controller.close()
        await action


@pytest.mark.asyncio
async def test_queued_dialog_validation_cannot_overwrite_discard_status(
    personas_editor_with_saved_character,
):
    import asyncio

    from textual.pilot import Pilot

    app, screen, _, _, view, controller, _ = await prepare(
        personas_editor_with_saved_character
    )
    action = asyncio.create_task(
        screen.run_worker(
            controller.handle(CharacterPerformanceRequested("regions"))
        ).wait()
    )
    try:
        async with asyncio.timeout(3):
            while not isinstance(app.screen, region_ui().PortraitRegionEditor):
                await Pilot(app).pause()
        pending = controller._review_check_task
        if pending is not None:
            await pending
        controller._check_context()
        check = controller._review_check_task
        controller.discard()
        status = view.query_one("#portrait-performance-status", Static)
        before = status.renderable
        await check
        assert status.renderable == before
    finally:
        await controller.close()
        await action
