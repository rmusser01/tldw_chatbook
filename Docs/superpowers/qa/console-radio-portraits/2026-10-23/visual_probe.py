"""One bounded production-CSS visual probe; snapshots are presentation fixtures."""
from pathlib import Path
import time
import pytest
from Tests.UI.console_fixture_ownership import owned_console_apps  # noqa: F401
from Tests.UI.test_console_character_avatar import (
    avatar_db, console_screen_with_db_and_pilot, _set_active_console_character,
)
from tldw_chatbook.Audio.speech_presentation import SpeechPresentationOwner, SpeechPresentationSnapshot

_character_screen = console_screen_with_db_and_pilot
pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_capture_radio_views(console_screen_with_db_and_pilot, monkeypatch):
    app, screen, db, pilot = console_screen_with_db_and_pilot
    pilot.app.no_color = False
    pilot.app.console.no_color = False
    # Headless harness inherits NO_COLOR; remove only its presentation filter.
    pilot.app._filters = [f for f in pilot.app._filters if type(f).__name__ not in {"Monochrome", "NoColor"}]
    pilot.app.refresh(repaint=True)
    data = Path('tldw_chatbook/assets/characters/samira/expressions/neutral.webp').read_bytes()
    char_id = db.add_character_card({'name':'Samira', 'image':data})
    _set_active_console_character(screen, char_id, 'Samira')
    owner = SpeechPresentationOwner('visual-only', object(), 'fixture', 1, char_id, None, None)
    monkeypatch.setattr(screen._character, '_read_character_speech', lambda: SpeechPresentationSnapshot(owner, 1, 'playing', time.monotonic(), -18, 'small', True))
    await screen._character._refresh_active_character_avatar_if_scope_changed(force=True)
    output = Path('Docs/superpowers/qa/console-radio-portraits/2026-10-23')
    for width, height in [(200,72),(120,36),(80,30)]:
        await pilot.resize_terminal(width,height)
        await pilot.pause()
        await pilot.pause()
        if width >= 100:
            screen.query_one("#console-character-avatar-frame").scroll_visible(animate=False, force=True)
        screen._character.sync_radio()
        await pilot.pause()
        svg = pilot.app.export_screenshot(title=f'Console radio · {width}x{height}')
        (output/f'radio-{width}x{height}.svg').write_text(svg)
