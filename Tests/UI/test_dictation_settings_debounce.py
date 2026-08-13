"""task-15470: dictation-settings persistence must be one batched write, off
the event loop, and never fire synchronously on a parsing keystroke.

Before this task, `_save_settings` made 6-9 sequential
`save_setting_to_cli_config` calls -- each its own full config.toml
read+atomic-rewrite+cache-reload cycle -- and `_persist_settings` (the single
gate every switch/input handler calls) invoked it synchronously on the event
loop. The buffer-duration input fired this once per keystroke that happened
to parse as an in-range integer.

These tests pin the replacement: `_persist_settings` only marks settings
dirty and arms a debounce timer; the write itself is one call to the batch
API (`save_settings_to_cli_config`), dispatched off the loop via
`asyncio.to_thread`; and `on_unmount` (fired both when the Dictation view is
switched away from in `STTS_Window` and when the app quits while it is
mounted) force-flushes any pending write.
"""

from __future__ import annotations

import toml
from textual.app import App, ComposeResult
from textual.widgets import Input

from tldw_chatbook.config import _get_effective_config_path
from tldw_chatbook.UI.Dictation_Window_Improved import (
    DICTATION_SETTINGS_SAVE_DEBOUNCE_SECONDS,
    ImprovedDictationWindow,
)


class _DictationHarness(App[None]):
    """Mounts the widget alone -- it needs no TldwCli/Screen scaffolding."""

    def compose(self) -> ComposeResult:
        yield ImprovedDictationWindow()


async def _mounted_window(pilot):
    window = pilot.app.query_one(ImprovedDictationWindow)
    # Let the mount-time `call_after_refresh(self._finish_mounting)` clear
    # `_settings_are_mounting` -- until it does, `_persist_settings` is a
    # deliberate no-op (see its own module-mount-noise guard), and asserting
    # against it before that would test the wrong thing.
    for _ in range(20):
        if not getattr(window, "_settings_are_mounting", True):
            break
        await pilot.pause()
    return window


def _config_path():
    return _get_effective_config_path()


async def test_buffer_duration_keystroke_does_not_write_synchronously():
    """AC #1: no config rewrite fires on a parsing keystroke."""
    async with _DictationHarness().run_test() as pilot:
        window = await _mounted_window(pilot)

        buffer_input = window.query_one("#buffer-duration-input", Input)
        buffer_input.value = "300"
        await pilot.pause()

        assert window._settings_dirty is True
        assert window._settings_save_timer is not None

        config_path = _config_path()
        if config_path.exists():
            on_disk = toml.load(config_path)
            assert (
                on_disk.get("dictation", {}).get("buffer_duration_ms") != 300
            ), "the keystroke wrote to disk synchronously instead of debouncing"


async def test_burst_of_edits_collapses_into_one_batched_write():
    """AC #1: the eventual write is ONE atomic batched mutation.

    Edits three different settings (two switches, one input) in a burst,
    then lets the debounce fire once. `save_settings_to_cli_config` must be
    called exactly once, carrying every changed field in a single mapping --
    not the 6-9 sequential single-key calls this replaced.
    """
    async with _DictationHarness().run_test() as pilot:
        window = await _mounted_window(pilot)

        calls: list[dict] = []
        original = window._write_settings_snapshot

        def spy(snapshot):
            calls.append(snapshot)
            original(snapshot)

        window._write_settings_snapshot = spy

        window.settings["punctuation"] = False
        window._persist_settings()
        window.settings["commands"] = False
        window._persist_settings()
        buffer_input = window.query_one("#buffer-duration-input", Input)
        buffer_input.value = "250"
        await pilot.pause()

        assert calls == [], "a write landed before the debounce fired"

        await pilot.pause(DICTATION_SETTINGS_SAVE_DEBOUNCE_SECONDS + 0.3)
        for _ in range(20):
            if not window._settings_dirty:
                break
            await pilot.pause(0.05)

        assert len(calls) == 1, f"expected exactly one batched write, got {len(calls)}"
        snapshot = calls[0]
        assert snapshot["dictation"]["punctuation"] is False
        assert snapshot["dictation"]["commands"] is False
        assert snapshot["dictation"]["buffer_duration_ms"] == 250

        on_disk = toml.load(_config_path())
        assert on_disk["dictation"]["punctuation"] is False
        assert on_disk["dictation"]["commands"] is False
        assert on_disk["dictation"]["buffer_duration_ms"] == 250


async def test_switch_away_immediately_after_edit_flushes_the_pending_write():
    """AC #1/#3 flush test: unmount (matching STTS_Window's view switch,
    which calls `content_container.remove_children()`) before the debounce
    timer fires must not lose the edit.
    """
    async with _DictationHarness().run_test() as pilot:
        window = await _mounted_window(pilot)

        window.settings["commands"] = False
        window._persist_settings()
        assert window._settings_dirty is True

        # Remove the widget the same way STTS_Window does when switching
        # views -- deliberately with no pause long enough for the natural
        # debounce timer, so only the unmount flush can be responsible for
        # what lands on disk.
        await window.remove()

    on_disk = toml.load(_config_path())
    assert on_disk["dictation"]["commands"] is False


async def test_quit_immediately_after_edit_flushes_the_pending_write():
    """AC #1/#3 flush test: app quit while the Dictation view is mounted."""
    async with _DictationHarness().run_test() as pilot:
        window = await _mounted_window(pilot)

        window.settings["punctuation"] = False
        window._persist_settings()
        assert window._settings_dirty is True
        # Exiting the `async with` block below tears the app down (quit),
        # deliberately with no further pause.

    on_disk = toml.load(_config_path())
    assert on_disk["dictation"]["punctuation"] is False
