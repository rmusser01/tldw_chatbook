"""Truthful radio presentation and stable production rail geometry."""

import time
from dataclasses import replace

import pytest
from textual.widgets import Static

from Tests.Chat.test_portrait_performance_state import performance_record
from Tests.UI.console_fixture_ownership import (
    owned_console_apps,  # noqa: F401 -- fixture
)
from Tests.UI.test_console_character_avatar import (
    _set_active_console_character,
    avatar_db,  # noqa: F401 -- fixture dependency
    console_screen_with_db_and_pilot,
)
from tldw_chatbook.Audio.speech_presentation import (
    SpeechPresentationOwner,
    SpeechPresentationSnapshot,
)
from tldw_chatbook.Chat.portrait_performance_state import portrait_radio_status

_character_screen = console_screen_with_db_and_pilot
pytestmark = pytest.mark.bootstrap_profile


def snapshot(state="playing", **changes):
    owner = SpeechPresentationOwner("radio", object(), "session", 1, 7, "reply", None)
    return replace(
        SpeechPresentationSnapshot(
            owner, 1, state, time.monotonic(), -12, "open", True
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "state,label",
    [
        ("preparing", "Preparing voice"),
        ("playing", "Transmitting"),
        ("paused", "Paused"),
        ("stopped", "Ready"),
        ("failed", "Voice unavailable"),
    ],
)
def test_status_follows_only_owned_transport(state, label):
    assert portrait_radio_status(snapshot(state)) == label
    assert portrait_radio_status(None) == "Ready"


@pytest.mark.asyncio
async def test_status_meter_and_portrait_bounds_stay_stable_during_timing_loss(
    console_screen_with_db_and_pilot, monkeypatch
):
    from tldw_chatbook.Character_Chat.portrait_performance import (
        capture_performance_source,
    )

    _app, screen, db, pilot = console_screen_with_db_and_pilot
    frames = performance_record().frames
    character_id = db.add_character_card({"name": "Radio Ada", "image": frames[0].png})
    _set_active_console_character(screen, character_id, "Radio Ada")
    source, _ = capture_performance_source(
        db, character_id=character_id, requested_state="idle"
    )
    db.apply_character_portrait_performance(source, frames, expected_revision=0)
    active = [snapshot()]
    monkeypatch.setattr(
        screen._character,
        "_read_character_speech",
        lambda: replace(active[0], observed_at=time.monotonic()) if active[0] else None,
    )
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    await pilot.pause()
    status = screen.query_one("#console-character-voice-status", Static)
    meter = screen.query_one("#console-character-voice-meter", Static)
    avatar = screen.query_one("#console-character-avatar-image")
    controls = screen.query_one("#console-character-reaction-open")
    before = (
        avatar.region,
        controls.region,
        screen.query_one("#console-character-radio").region,
        status.region,
    )
    active[0] = snapshot(timing_valid=False, mouth="rest", level_dbfs=None)
    await screen._character._refresh_active_character_avatar_if_scope_changed()
    await pilot.pause()
    assert str(status.renderable) == "Transmitting"
    assert meter.display and not meter.visible
    assert (
        avatar.region,
        controls.region,
        screen.query_one("#console-character-radio").region,
        status.region,
    ) == before
    assert screen.query_one("#console-character-avatar-image") is avatar


@pytest.mark.parametrize(
    "state,timing,level,age,expected",
    [
        ("playing", True, -12, 0, 0.8),
        ("playing", True, None, 0, 0),
        ("paused", True, -12, 0, None),
        ("playing", False, -12, 0, None),
        ("playing", True, -12, 0.2, None),
    ],
)
def test_meter_uses_only_fresh_output_measurement(state, timing, level, age, expected):
    from tldw_chatbook.Chat.portrait_performance_state import portrait_radio_level

    now = time.monotonic()
    value = snapshot(
        state, timing_valid=timing, level_dbfs=level, observed_at=now - age
    )
    assert portrait_radio_level(value, now=now) == expected


@pytest.mark.asyncio
async def test_voice_states_and_motion_keep_reserved_rows(
    console_screen_with_db_and_pilot, monkeypatch
):
    app, screen, db, pilot = console_screen_with_db_and_pilot
    character_id = db.add_character_card(
        {"name": "Radio Ada", "image": performance_record().frames[0].png}
    )
    _set_active_console_character(screen, character_id, "Radio Ada")
    active = [None]
    monkeypatch.setattr(
        screen._character,
        "_read_character_speech",
        lambda: replace(active[0], observed_at=time.monotonic()) if active[0] else None,
    )
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    await pilot.pause()
    status = screen.query_one("#console-character-voice-status", Static)
    meter = screen.query_one("#console-character-voice-meter", Static)
    before = (
        status.region,
        screen.query_one("#console-character-radio").region,
        screen.query_one("#console-character-reaction-open").region,
    )
    for state, label in [
        ("preparing", "Preparing voice"),
        ("playing", "Transmitting"),
        ("paused", "Paused"),
        ("failed", "Voice unavailable"),
        ("stopped", "Ready"),
    ]:
        active[0] = snapshot(state)
        screen._character.sync_radio()
        await pilot.pause()
        assert str(status.renderable) == label
        assert (
            status.region,
            screen.query_one("#console-character-radio").region,
            screen.query_one("#console-character-reaction-open").region,
        ) == before
    active[0] = snapshot(level_dbfs=None, mouth="rest")
    screen._character.sync_radio()
    assert str(meter.renderable) == ""
    app.app_config.setdefault("appearance", {})["reduce_motion"] = True
    active[0] = snapshot()
    screen._character.sync_radio()
    await pilot.pause()
    assert meter.display and not meter.visible
    assert (
        screen.query_one("#console-character-radio").region.height
        - status.region.height
        == 1
    )


@pytest.mark.asyncio
async def test_markup_shaped_character_name_is_literal_in_console_controls(
    console_screen_with_db_and_pilot,
):
    import asyncio

    _app, screen, db, pilot = console_screen_with_db_and_pilot
    name = "[/bold][reverse]Radio name[/reverse]"
    character_id = db.add_character_card(
        {"name": name, "image": performance_record().frames[0].png}
    )
    _set_active_console_character(screen, character_id, name)
    await screen._sync_native_console_chat_ui()
    await pilot.pause()
    label = screen.query_one("#console-assistant-label", Static)
    # A concurrent Console tick can coalesce the explicit sync; await painted state.
    async with asyncio.timeout(5):
        while name not in label.render().plain:
            await pilot.pause()
    assert name in label.render().plain


@pytest.mark.asyncio
async def test_compact_meter_follows_layout_not_speech_and_restores_after_resize(
    console_screen_with_db_and_pilot, monkeypatch
):
    _app, screen, db, pilot = console_screen_with_db_and_pilot
    character_id = db.add_character_card(
        {"name": "Radio Ada", "image": performance_record().frames[0].png}
    )
    _set_active_console_character(screen, character_id, "Radio Ada")
    active = [snapshot()]
    monkeypatch.setattr(
        screen._character,
        "_read_character_speech",
        lambda: replace(active[0], observed_at=time.monotonic()),
    )
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    await pilot.pause()
    meter = screen.query_one("#console-character-voice-meter", Static)
    assert meter.display
    await pilot.resize_terminal(120, 18)
    await pilot.pause()
    assert not meter.display
    for state in ("preparing", "playing", "paused", "stopped"):
        active[0] = snapshot(state)
        screen._character.sync_radio()
        await pilot.pause()
        assert not meter.display
    await pilot.resize_terminal(200, 72)
    await pilot.pause()
    assert meter.display
    assert not meter.visible
