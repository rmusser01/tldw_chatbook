"""Opening Speech Recognition must not write to the user's config.

It did: arriving at the view persisted a `[dictation]` section, including
`[dictation.privacy]`, with nothing touched. `on_mount` does not save --
Textual posts `Changed` when a `Switch` or `Input` is created with a value,
and every one of those handlers called `_save_settings()`.

Two reasons that matters beyond tidiness. It records privacy preferences the
user never expressed, so a later change to the shipped defaults never
reaches anyone who once opened the view. And in the file, a value written on
mount is indistinguishable from one the user chose.

The assertion is on the config FILE, not on a mock. What matters is whether
bytes changed on disk.
"""

from __future__ import annotations

import pathlib

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen


@pytest.mark.asyncio
async def test_opening_the_view_writes_nothing(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('[general]\nusers_name = "probe"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    before = config.read_bytes()

    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = STTSScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        row = next(
            b for b in screen.query(Button)
            if getattr(b, "lab_view_key", None) == "dictation"
        )
        row.press()
        for _ in range(10):
            await pilot.pause()

    assert config.read_bytes() == before, (
        "opening Speech Recognition rewrote the config file:\n"
        f"{config.read_text()[:400]}"
    )


@pytest.mark.asyncio
async def test_a_real_change_still_persists(tmp_path, monkeypatch):
    """The guard must not silence genuine edits -- that would be the same
    bug pointing the other way, and harder to notice."""
    config = tmp_path / "config.toml"
    config.write_text('[general]\nusers_name = "probe"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))

    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = STTSScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        row = next(
            b for b in screen.query(Button)
            if getattr(b, "lab_view_key", None) == "dictation"
        )
        row.press()
        for _ in range(10):
            await pilot.pause()

        switch = screen.query_one("#punctuation-switch")
        switch.value = not switch.value
        for _ in range(6):
            await pilot.pause()

    assert "dictation" in config.read_text(), "a real toggle did not persist"
