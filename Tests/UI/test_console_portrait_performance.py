"""Actual fitted pixels, widget stability and owned living-portrait playback."""

import time
from dataclasses import replace

import pytest
from textual.widgets import Static

from Tests.Chat.test_portrait_performance_state import performance_record
from Tests.UI.console_fixture_ownership import (
    owned_console_apps,  # noqa: F401 - pytest fixture
)
from Tests.UI.test_character_expression_avatar import AvatarApp, wait_ready
from Tests.UI.test_console_character_avatar import (
    _set_active_console_character,
    _set_chat_images_setting,
    avatar_db,  # noqa: F401 - reused pytest fixtures
    console_screen_with_db_and_pilot,
)
from tldw_chatbook.Audio.speech_presentation import (
    SpeechPresentationOwner,
    SpeechPresentationSnapshot,
)
from tldw_chatbook.Chat.character_expression_playback import preparation_bytes
from tldw_chatbook.Widgets.Console.character_expression_avatar import (
    CharacterExpressionAvatar,
)

# The reused fixture constructs in its defining module; register that exact
# namespace with the existing terminal-ownership fixture as well.
_character_screen = console_screen_with_db_and_pilot
pytestmark = pytest.mark.bootstrap_profile


def performance_app(*, animate=True):
    record = performance_record()
    owner = SpeechPresentationOwner("test", object(), "session", 1, 7, "reply", None)
    snapshot = [
        SpeechPresentationSnapshot(
            owner, 1, "playing", time.monotonic(), -12, "open", True
        )
    ]
    app = AvatarApp()
    app.avatar = CharacterExpressionAvatar(
        record.frames[0].png,
        box=(10, 6),
        animate=animate,
        is_current=lambda: True,
        monochrome=False,
        mode="pixels",
        id="avatar",
        performance=record,
        read_speech=lambda: snapshot[0],
    )
    return app, snapshot


@pytest.mark.asyncio
async def test_performance_changes_fitted_pixels_on_owned_silence_without_remount():
    before = preparation_bytes()
    app, snapshot = performance_app()
    async with app.run_test(size=(40, 16)) as pilot:
        await wait_ready(app.avatar, pilot)
        child = app.avatar.query_one(Static)
        app.avatar._last_tick = None
        snapshot[0] = replace(snapshot[0], observed_at=time.monotonic())
        await app.avatar._tick()
        loud = child.render().spans
        assert app.avatar.current_image.getpixel((3, 3)) == (0, 0, 255, 255)
        geometry = app.avatar.region
        snapshot[0] = replace(snapshot[0], mouth="rest", level_dbfs=None)
        await app.avatar._tick()
        assert child.render().spans != loud
        assert app.avatar.current_image.getpixel((3, 3)) == (255, 0, 0, 255)
        assert app.avatar.query_one(Static) is child
        assert app.avatar.region == geometry
    assert preparation_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("animate", [False, True])
async def test_static_or_stale_speech_keeps_poster_mouth_closed(animate):
    app, snapshot = performance_app(animate=animate)
    snapshot[0] = replace(snapshot[0], observed_at=time.monotonic() - 1)
    async with app.run_test(size=(40, 16)) as pilot:
        await wait_ready(app.avatar, pilot)
        await app.avatar._tick()
        assert app.avatar.current_image.getpixel((3, 3)) == (255, 0, 0, 255)
        if not animate:
            assert app.avatar._timer is None


@pytest.mark.asyncio
async def test_console_reads_committed_performance_revision_and_remove_tombstone(
    console_screen_with_db_and_pilot,
):
    from tldw_chatbook.Character_Chat.portrait_performance import (
        capture_performance_source,
    )

    _app, screen, db, pilot = console_screen_with_db_and_pilot
    frames = performance_record().frames
    character_id = db.add_character_card({"name": "Ada", "image": frames[0].png})
    _set_active_console_character(screen, character_id, "Ada")
    source, _ = capture_performance_source(
        db, character_id=character_id, requested_state="idle"
    )
    record = db.apply_character_portrait_performance(
        source, frames, expected_revision=0
    )
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    spec = screen._active_character_avatar
    assert spec["performance"].revision == record
    for _ in range(100):
        widget = screen.query_one(CharacterExpressionAvatar)
        if widget.current_image is not None:
            break
        await pilot.pause(0.02)
    assert widget.current_image is not None
    first_identity = spec["resolution_cache_identity"]
    removed = db.remove_character_portrait_performance(source, expected_revision=record)
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    assert not screen._active_character_avatar.get("performance")
    assert (
        screen._active_character_avatar["resolution_cache_identity"] != first_identity
    )
    reapplied = db.apply_character_portrait_performance(
        source, frames, expected_revision=removed
    )
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    assert screen._active_character_avatar["performance"].revision == reapplied
    assert (
        screen._active_character_avatar["resolution_cache_identity"] != first_identity
    )


@pytest.mark.asyncio
async def test_performance_render_failure_closes_to_selected_rest_poster(monkeypatch):
    app, snapshot = performance_app()
    async with app.run_test(size=(40, 16)) as pilot:
        await wait_ready(app.avatar, pilot)
        snapshot[0] = replace(snapshot[0], observed_at=time.monotonic())
        await app.avatar._tick()
        prepared = app.avatar._prepared

        def fail(*_args):
            raise ValueError("budget unavailable")

        monkeypatch.setattr(prepared, "render_pose", fail)
        snapshot[0] = replace(snapshot[0], mouth="rest")
        await app.avatar._tick()
        assert app.avatar.current_image.getpixel((3, 3)) == (255, 0, 0, 255)
        assert app.avatar._timer is None


@pytest.mark.asyncio
async def test_late_loud_render_cannot_reopen_after_new_silence(monkeypatch):
    import asyncio
    import threading

    import tldw_chatbook.UI.Console_Modules.character_avatar_layout as layout

    app, snapshot = performance_app()
    snapshot[0] = replace(snapshot[0], mouth="rest")
    async with app.run_test(size=(40, 16)) as pilot:
        await wait_ready(app.avatar, pilot)
        app.avatar._timer.stop()
        started, release = threading.Event(), threading.Event()
        original = layout.render_character_avatar_mosaic

        def held(image, *args, **kwargs):
            if image.getpixel((3, 3)) == (0, 0, 255, 255):
                started.set()
                assert release.wait(3)
            return original(image, *args, **kwargs)

        monkeypatch.setattr(layout, "render_character_avatar_mosaic", held)
        snapshot[0] = replace(snapshot[0], mouth="open", observed_at=time.monotonic())
        tick = asyncio.create_task(app.avatar._tick())
        assert await asyncio.to_thread(started.wait, 2)
        snapshot[0] = replace(
            snapshot[0], mouth="rest", level_dbfs=None, observed_at=time.monotonic()
        )
        release.set()
        await tick
        assert app.avatar.current_image.getpixel((3, 3)) == (255, 0, 0, 255)
        await app.avatar._tick()
        assert app.avatar.current_image.getpixel((3, 3)) == (255, 0, 0, 255)


@pytest.mark.asyncio
async def test_removed_same_source_old_decoder_cannot_restore_predecessor_pixels(
    console_screen_with_db_and_pilot,
    monkeypatch,
):
    import asyncio
    import threading

    import tldw_chatbook.Widgets.Console.character_expression_avatar as module
    from tldw_chatbook.Audio.speech_presentation import SpeechPresentation
    from tldw_chatbook.Character_Chat.portrait_performance import (
        capture_performance_source,
        normalize_performance_frames,
    )

    _app, screen, db, pilot = console_screen_with_db_and_pilot
    frames = performance_record().frames
    character_id = db.add_character_card({"name": "Ada", "image": frames[0].png})
    _set_active_console_character(screen, character_id, "Ada")
    source, _ = capture_performance_source(
        db, character_id=character_id, requested_state="idle"
    )
    first = db.apply_character_portrait_performance(source, frames, expected_revision=0)
    started, release = threading.Event(), threading.Event()
    original = module.prepare_performance

    def hold_old(record, box):
        if record.revision == first:
            started.set()
            assert release.wait(5)
        return original(record, box)

    monkeypatch.setattr(module, "prepare_performance", hold_old)
    before = preparation_bytes()
    initial = asyncio.create_task(
        screen._character._refresh_active_character_avatar_if_scope_changed(force=True)
    )
    assert await asyncio.to_thread(started.wait, 2)
    removed = db.remove_character_portrait_performance(source, expected_revision=first)
    replacement = {f.role: f.png for f in frames}
    replacement["open"] = frames[1].png
    newest = db.apply_character_portrait_performance(
        source,
        normalize_performance_frames(replacement),
        expected_revision=removed,
    )
    refresh = asyncio.create_task(
        screen._character._refresh_active_character_avatar_if_scope_changed(force=True)
    )
    for _ in range(100):
        if screen._active_character_avatar.get("performance").revision == newest:
            break
        await asyncio.sleep(0.01)
    release.set()
    assert screen._active_character_avatar["performance"].revision == newest
    await asyncio.gather(initial, refresh)
    for _ in range(100):
        widget = screen.query_one(CharacterExpressionAvatar)
        if widget.current_image is not None:
            break
        await pilot.pause(0.02)
    store = screen._ensure_console_chat_store()
    owner = SpeechPresentationOwner(
        "newest",
        db,
        store.active_session_id,
        store.active_session_epoch(),
        character_id,
        "reply",
        None,
    )
    speech = SpeechPresentation(owner)
    screen._message.bind_speech_presentation(speech, lambda: True)
    speech.publish(
        SpeechPresentationSnapshot(
            owner, 1, "playing", time.monotonic(), -12, "open", True
        )
    )
    await widget._tick()
    assert widget.current_image.getpixel((3, 3)) == (0, 128, 0, 255)
    await widget.remove()
    assert preparation_bytes() == before


async def current_console_avatar(screen, pilot):
    for _ in range(100):
        widget = screen.query_one(CharacterExpressionAvatar)
        if widget.current_image is not None and widget._current():
            return widget
        await pilot.pause(0.02)
    pytest.fail("current fitted portrait never painted")


@pytest.mark.asyncio
async def test_owned_older_replay_restores_latest_mood_and_manual_still_wins(
    console_screen_with_db_and_pilot,
    monkeypatch,
):
    from Tests.UI.test_console_character_avatar import _arm_live_emotes, _resolution
    from tldw_chatbook.Audio.speech_presentation import SpeechPresentation
    from tldw_chatbook.Character_Chat.portrait_performance import (
        PortraitPerformanceRecord,
        normalize_performance_frames,
        source_from_resolution,
    )
    from tldw_chatbook.Widgets.Console.console_image_viewer_modal import (
        AvatarViewRequested,
    )

    _app, screen, db, pilot = console_screen_with_db_and_pilot
    happy = performance_record().frames
    sad_bytes = {f.role: f.png for f in happy}
    sad_bytes["rest"], sad_bytes["open"] = happy[1].png, happy[3].png
    sad = normalize_performance_frames(sad_bytes)
    character_id = db.add_character_card({"name": "Ada", "image": happy[0].png})
    _set_active_console_character(screen, character_id, "Ada")
    store, older = _arm_live_emotes(screen, character_id, "happy")
    store.mark_message_complete(older.id)
    store, latest = _arm_live_emotes(screen, character_id, "sad")
    store.mark_message_complete(latest.id)
    frames = {"happy": happy, "sad": sad}

    def resolution(key, *, manual=None):
        return _resolution(
            character_id,
            requested=key,
            manual=manual,
            source="pack_manual" if manual else "history_immutable",
            identity_suffix=key,
            image=frames[key][0].png,
        )

    records = {
        key: PortraitPerformanceRecord(
            source_from_resolution(resolution(key)), 1, True, value
        )
        for key, value in frames.items()
    }
    monkeypatch.setattr(
        db,
        "get_character_portrait_performance",
        lambda **kwargs: records.get(kwargs["expression_key"]),
    )
    screen._character._resolve_historical_visual_identity = lambda _scope, identity: (
        resolution(identity.expression_key)
    )
    screen._character._resolve_visual_identity = lambda _scope, _state, manual: (
        resolution(manual or "sad", manual=manual)
    )
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    widget = await current_console_avatar(screen, pilot)
    assert widget.current_image.getpixel((3, 3)) == (0, 128, 0, 255)
    owner = SpeechPresentationOwner(
        "older-replay",
        db,
        store.active_session_id,
        store.active_session_epoch(),
        character_id,
        older.id,
        None,
    )
    speech = SpeechPresentation(owner)
    screen._message.bind_speech_presentation(speech, lambda: True)
    speech.set_transport("playing")
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    widget = await current_console_avatar(screen, pilot)

    async def loud_pixel(expected):
        for _ in range(30):
            speech.publish(
                SpeechPresentationSnapshot(
                    owner,
                    1,
                    "playing",
                    time.monotonic(),
                    -12,
                    "open",
                    True,
                )
            )
            await widget._tick()
            if widget.current_image.getpixel((3, 3)) == expected:
                return
            await pilot.pause(0.01)
        pytest.fail("owned mouth never committed the expected fitted pixels")

    await loud_pixel((0, 0, 255, 255))
    viewer = []
    monkeypatch.setattr(screen.app, "push_screen", viewer.append)
    screen._handle_avatar_view_requested(AvatarViewRequested())
    assert viewer[0]._image is not widget.current_image
    assert viewer[0]._image.tobytes() == widget.current_image.tobytes()
    viewer[0]._image.close()
    scope = screen._session._current_visual_identity_actor_scope()
    screen._session._set_manual_reaction(scope, "sad")
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    widget = await current_console_avatar(screen, pilot)
    await loud_pixel((255, 255, 0, 255))
    screen._session._clear_manual_reaction(scope)
    speech.set_transport("stopped")
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    widget = await current_console_avatar(screen, pilot)
    assert screen._active_character_avatar["state"] == "sad"
    assert widget.current_image.getpixel((3, 3)) == (0, 128, 0, 255)


@pytest.mark.asyncio
async def test_hidden_performance_resamples_silence_without_replaying_hidden_time():
    app, snapshot = performance_app()
    before = preparation_bytes()
    async with app.run_test(size=(40, 16)) as pilot:
        await wait_ready(app.avatar, pilot)
        snapshot[0] = replace(snapshot[0], observed_at=time.monotonic())
        await app.avatar._tick()
        app.avatar.display = False
        await app.avatar._tick()
        elapsed = app.avatar._elapsed_ms
        await pilot.pause(0.15)
        assert app.avatar._elapsed_ms == elapsed
        snapshot[0] = replace(
            snapshot[0], mouth="rest", level_dbfs=None, observed_at=time.monotonic()
        )
        app.avatar.display = True
        await pilot.pause(0.04)
        await app.avatar._tick()
        assert app.avatar._elapsed_ms - elapsed < 100
        assert app.avatar.current_image.getpixel((3, 3)) == (255, 0, 0, 255)
    assert preparation_bytes() == before


@pytest.mark.asyncio
async def test_unprepared_animation_uses_original_poster_during_owned_speech():
    app = AvatarApp()
    owner = SpeechPresentationOwner("test", object(), "session", 1, 7, "reply", None)
    app.avatar._read_speech = lambda: SpeechPresentationSnapshot(
        owner, 1, "playing", time.monotonic(), None, "rest", False
    )
    async with app.run_test() as pilot:
        await wait_ready(app.avatar, pilot)
        app.avatar._elapsed_ms = 150
        await app.avatar._tick()
        assert app.avatar.current_image.getpixel((3, 3)) == (255, 0, 0, 255)


@pytest.mark.asyncio
async def test_console_rejects_foreign_speech_and_obeys_each_motion_preference(
    console_screen_with_db_and_pilot,
):
    from tldw_chatbook.Audio.speech_presentation import SpeechPresentation
    from tldw_chatbook.Character_Chat.portrait_performance import (
        capture_performance_source,
    )

    app, screen, db, pilot = console_screen_with_db_and_pilot
    frames = performance_record().frames
    character_id = db.add_character_card({"name": "Ada", "image": frames[0].png})
    _set_active_console_character(screen, character_id, "Ada")
    source, _ = capture_performance_source(
        db, character_id=character_id, requested_state="idle"
    )
    db.apply_character_portrait_performance(source, frames, expected_revision=0)
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    widget = await current_console_avatar(screen, pilot)
    widget._timer.stop()
    store = screen._ensure_console_chat_store()
    owner = SpeechPresentationOwner(
        "valid",
        db,
        store.active_session_id,
        store.active_session_epoch(),
        character_id,
        "reply",
        None,
    )
    for foreign in (
        replace(owner, scope_token=object()),
        replace(owner, character_id=None),
        replace(owner, character_id=character_id + 1),
        replace(owner, session_id="other"),
        replace(owner, session_epoch=owner.session_epoch + 1),
    ):
        speech = SpeechPresentation(foreign)
        screen._message.bind_speech_presentation(speech, lambda: True)
        speech.publish(
            SpeechPresentationSnapshot(
                foreign, 1, "playing", time.monotonic(), -12, "open", True
            )
        )
        widget._last_tick = None
        await widget._tick()
        assert screen._character._read_character_speech() is None
        assert widget.current_image.getpixel((3, 3)) == (255, 0, 0, 255)
    speech = SpeechPresentation(owner)
    screen._message.bind_speech_presentation(speech, lambda: True)
    speech.set_transport("playing")
    for key, value in (
        ("character_expression_mode", "static"),
        ("reduce_motion", True),
        ("animations_enabled", False),
    ):
        appearance = app.app_config.setdefault("appearance", {})
        before = appearance.get(key)
        appearance[key] = value
        await screen._character._refresh_active_character_avatar_if_scope_changed(
            force=True
        )
        widget = await current_console_avatar(screen, pilot)
        assert not screen._active_character_avatar["animate"]
        assert widget._timer is None
        assert widget.current_image.getpixel((3, 3)) == (255, 0, 0, 255)
        if before is None:
            appearance.pop(key)
        else:
            appearance[key] = before
    _set_chat_images_setting(app, "react_character_expressions", False)
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    widget = await current_console_avatar(screen, pilot)
    assert not screen._active_character_avatar["animate"]
    assert widget._timer is None


@pytest.mark.asyncio
async def test_attachment_read_failure_keeps_selected_poster(
    console_screen_with_db_and_pilot, monkeypatch
):
    import sqlite3

    _app, screen, db, _pilot = console_screen_with_db_and_pilot
    png = performance_record().frames[0].png
    character_id = db.add_character_card({"name": "Ada", "image": png})
    _set_active_console_character(screen, character_id, "Ada")

    def unavailable(**_kwargs):
        raise sqlite3.OperationalError("injected read failure")

    monkeypatch.setattr(db, "get_character_portrait_performance", unavailable)
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    assert screen._active_character_avatar["pil"].getpixel((3, 3))[:3] == (255, 0, 0)
    assert not screen._active_character_avatar.get("performance")


@pytest.mark.asyncio
async def test_concurrent_sync_cannot_consume_live_expression_beats_twice(
    console_screen_with_db_and_pilot,
):
    import asyncio

    from Tests.UI.test_console_character_avatar import _arm_live_emotes, _resolution

    _app, screen, db, _pilot = console_screen_with_db_and_pilot
    png = performance_record().frames[0].png
    character_id = db.add_character_card({"name": "Ada", "image": png})
    _set_active_console_character(screen, character_id, "Ada")
    _arm_live_emotes(screen, character_id, "happy", "sad")
    screen._character._resolve_visual_identity = lambda _scope, state, _manual: (
        _resolution(
            character_id,
            requested=state,
            manual=None,
            source="pack_explicit",
            identity_suffix=state,
            image=png,
        )
    )
    entered, release = asyncio.Event(), asyncio.Event()
    painted = []

    async def render(**kwargs):
        state = kwargs["spec"]["state"]
        painted.append(state)
        if state == "happy":
            entered.set()
            await release.wait()

    screen._character._render_character_avatar = render
    first = asyncio.create_task(
        screen._character._refresh_active_character_avatar_if_scope_changed()
    )
    await asyncio.wait_for(entered.wait(), 2)
    second = asyncio.create_task(
        screen._character._refresh_active_character_avatar_if_scope_changed()
    )
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(second), 0.1)
    finally:
        release.set()
    await asyncio.wait_for(asyncio.gather(first, second), 2)
    assert painted == ["happy", "sad"]


@pytest.mark.asyncio
async def test_missing_historical_art_never_borrows_a_neutral_talking_set(
    console_screen_with_db_and_pilot,
):
    from Tests.UI.test_console_character_avatar import _arm_live_emotes
    from tldw_chatbook.Character_Chat.portrait_performance import (
        capture_performance_source,
    )

    _app, screen, db, _pilot = console_screen_with_db_and_pilot
    frames = performance_record().frames
    character_id = db.add_character_card({"name": "Ada", "image": frames[0].png})
    _set_active_console_character(screen, character_id, "Ada")
    source, _ = capture_performance_source(
        db, character_id=character_id, requested_state="idle"
    )
    db.apply_character_portrait_performance(source, frames, expected_revision=0)
    store, message = _arm_live_emotes(screen, character_id, "sad")
    store.mark_message_complete(message.id)
    screen._character._resolve_historical_visual_identity = lambda *_args: None
    screen._character._active_character_avatar = None
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    assert screen._active_character_avatar is not None
    assert not screen._active_character_avatar["animate"]
    assert screen._active_character_avatar.get("performance") is None


@pytest.mark.asyncio
async def test_offline_preview_cycles_exact_six_roles_and_idle_without_speech():
    from tldw_chatbook.Character_Chat.portrait_performance import FRAME_ROLES

    app, _snapshot = performance_app()
    app.avatar._read_speech = None
    app.avatar.set_preview_mode("cycle")
    before = preparation_bytes()
    async with app.run_test(size=(40, 16)) as pilot:
        await wait_ready(app.avatar, pilot)
        app.avatar._timer.stop()
        child = app.avatar.query_one(Static)
        for index, _role in enumerate(FRAME_ROLES):
            app.avatar._elapsed_ms = (
                (index + 0.25) * app.avatar._prepared.motion.blink_hold_s * 1000
            )
            app.avatar._last_tick = None
            await app.avatar._tick()
            assert app.avatar._frame_index == index
            assert app.avatar.current_image.getpixel(
                (3, 3)
            ) == app.avatar._prepared.frames[index].getpixel((3, 3))
            assert app.avatar.query_one(Static) is child
        app.avatar.set_preview_mode("idle")
        await app.avatar._tick()
        assert app.avatar._frame_index == 0
        app.avatar._elapsed_ms = 7100
        app.avatar._last_tick = None
        await app.avatar._tick()
        assert app.avatar._frame_index == 3
    assert preparation_bytes() == before


@pytest.mark.asyncio
async def test_personas_apply_and_remove_handoff_to_mounted_console(
    console_screen_with_db_and_pilot,
    monkeypatch,
):
    import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as handler
    from Tests.UI.test_personas_dictionaries import patch_character_paging
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
    from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
        EditCharacterRequested,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.portrait_performance_editor import (
        CharacterPerformanceRequested,
    )

    app, console, db, pilot = console_screen_with_db_and_pilot
    frames = performance_record().frames
    character_id = db.add_character_card(
        {"name": "Reviewed Ada", "image": frames[3].png}
    )
    _set_active_console_character(console, character_id, "Reviewed Ada")
    monkeypatch.setattr(app, "character_persona_scope_service", None)
    monkeypatch.setattr(
        handler,
        "fetch_all_characters",
        lambda: [db.get_character_card_by_id(character_id)],
    )
    monkeypatch.setattr(
        handler,
        "fetch_character_by_id",
        lambda _id: db.get_character_card_by_id(character_id),
    )
    patch_character_paging(monkeypatch)
    host = pilot.app
    from Tests.UI.consolidated_css import CSS_DIR

    host.stylesheet.read(CSS_DIR / "screen_feature_roleplay.tcss")
    host.stylesheet.reparse()
    host.stylesheet.update(host)
    personas = PersonasScreen(app)
    host.install_screen(personas, "portrait-authoring")
    await host.push_screen("portrait-authoring")
    await pilot.pause()
    await host.workers.wait_for_complete()
    await personas._select_character(str(character_id), "Reviewed Ada")
    await pilot.pause()
    await host.workers.wait_for_complete()
    personas.post_message(EditCharacterRequested(str(character_id)))
    await pilot.pause()
    await host.workers.wait_for_complete()
    controller = personas._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    for frame in frames:
        controller.draft.stage(frame)
    assert await controller.handle(CharacterPerformanceRequested("apply"))
    await host.pop_screen()
    await pilot.pause()
    await console._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    assert console._active_character_avatar["performance"].revision == 1
    widget = console.query_one(CharacterExpressionAvatar)
    await wait_ready(widget, pilot)
    assert widget.current_image.getpixel((3, 3)) == (255, 0, 0, 255)
    await host.push_screen("portrait-authoring")
    assert await controller.handle(CharacterPerformanceRequested("remove"))
    await host.pop_screen()
    await pilot.pause()
    await console._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    assert console._active_character_avatar.get("performance") is None
    assert console._active_character_avatar["pil"].getpixel((3, 3)) == (
        255,
        255,
        0,
        255,
    )


@pytest.mark.asyncio
async def test_graphics_mount_finishes_when_preview_pose_changes(monkeypatch):
    pytest.importorskip("textual_image.widget")
    app, _snapshot = performance_app()
    avatar = app.avatar
    avatar._mode = "graphics"
    avatar.set_preview_mode("rest")
    mount = avatar.mount

    async def change_pose_after_mount(*children, **kwargs):
        await mount(*children, **kwargs)
        if avatar._graphics is not None and avatar._graphics in children:
            avatar.set_preview_mode("small")

    monkeypatch.setattr(avatar, "mount", change_pose_after_mount)
    before = preparation_bytes()
    async with app.run_test(size=(40, 16)) as pilot:
        await wait_ready(avatar, pilot)
        avatar._timer.stop()
        graphics = avatar._graphics
        assert graphics is not None and not avatar._surface.display
        await avatar._tick()
        assert avatar._frame_index == 1
        assert avatar._graphics is graphics
        assert graphics.image is avatar.current_image
        assert avatar.current_image.getpixel((3, 3)) == avatar._prepared.frames[
            1
        ].getpixel((3, 3))
    assert preparation_bytes() == before
