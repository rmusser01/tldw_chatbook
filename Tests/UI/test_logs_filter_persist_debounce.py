"""task-21124: Logs saved-filter persistence must be one debounced batched
write, off the event loop -- never two synchronous config rewrites per click.

Before this task, every level-chip click ran `save_filter_state()` inline on
the event loop: two sequential `save_setting_to_cli_config` calls, each a
full config.toml read-atomic-rewrite-reload cycle (four fsyncs per click,
all while holding the global config write lock). `on_unmount` additionally
rewrote the config file unconditionally on every exit from the Logs screen.

These tests pin the replacement (the task-15470 debounce shape): a chip
click only arms a debounce timer; the eventual write is ONE batched
`save_settings_to_cli_config` mutation dispatched off the loop; unmount
flushes a pending change and writes nothing when nothing changed.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import toml
from textual.app import ComposeResult
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.config import _get_effective_config_path
from tldw_chatbook.UI.Logs_Window import (
    LOGS_FILTER_SAVE_DEBOUNCE_SECONDS,
    LogsWindow,
)


class _LogsHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield LogsWindow(SimpleNamespace(_log_records=deque()))


def _saved_logs_section() -> dict:
    config_path = _get_effective_config_path()
    if not config_path.exists():
        return {}
    return toml.load(config_path).get("logs", {})


def _spy_writes(window: LogsWindow) -> list[dict]:
    calls: list[dict] = []
    original = window._write_filter_state

    def spy(snapshot):
        calls.append(dict(snapshot))
        original(snapshot)

    window._write_filter_state = spy
    return calls


async def test_chip_burst_debounces_into_one_batched_write():
    """A burst of chip clicks lands as exactly ONE write, after the debounce."""
    async with _LogsHarness().run_test(size=(120, 36)) as pilot:
        window = pilot.app.query_one(LogsWindow)
        window.load_from_app()
        await pilot.pause()
        calls = _spy_writes(window)

        window.query_one("#logs-filter-warning", Button).press()
        await pilot.pause()
        window.query_one("#logs-filter-error", Button).press()
        await pilot.pause()

        assert calls == [], "a write landed synchronously on the chip click"
        assert window._filter_save_timer is not None, "debounce timer not armed"
        assert _saved_logs_section().get("last_level_chip") != "error", (
            "the chip click wrote to disk before the debounce fired"
        )

        await pilot.pause(LOGS_FILTER_SAVE_DEBOUNCE_SECONDS + 0.3)
        for _ in range(40):
            if calls:
                break
            await pilot.pause(0.05)

        assert len(calls) == 1, f"expected one batched write, got {len(calls)}"
        assert calls[0]["last_level_chip"] == "error"
        assert "last_filter" in calls[0], "the write must batch both keys"
        assert _saved_logs_section().get("last_level_chip") == "error"


async def test_unmount_before_debounce_flushes_the_pending_change():
    """Leaving the screen right after a click must not lose the change."""
    async with _LogsHarness().run_test(size=(120, 36)) as pilot:
        window = pilot.app.query_one(LogsWindow)
        window.load_from_app()
        await pilot.pause()

        window.query_one("#logs-filter-error", Button).press()
        await pilot.pause()
        # Deliberately no pause long enough for the debounce timer: only
        # the unmount flush can be responsible for what lands on disk.
        await window.remove()

    assert _saved_logs_section().get("last_level_chip") == "error"


async def test_unmount_without_changes_writes_nothing():
    """Exiting the Logs screen untouched no longer rewrites the config."""
    async with _LogsHarness().run_test(size=(120, 36)) as pilot:
        window = pilot.app.query_one(LogsWindow)
        window.load_from_app()
        await pilot.pause()
        calls = _spy_writes(window)

        await window.remove()

        assert calls == [], "unmount rewrote the config file although nothing changed"


async def test_click_and_click_back_writes_nothing():
    """Returning to the persisted state cancels the pending write."""
    async with _LogsHarness().run_test(size=(120, 36)) as pilot:
        window = pilot.app.query_one(LogsWindow)
        window.load_from_app()
        await pilot.pause()
        baseline_chip = window._level_chip
        calls = _spy_writes(window)

        window.query_one("#logs-filter-error", Button).press()
        await pilot.pause()
        window.query_one(f"#logs-filter-{baseline_chip}", Button).press()
        await pilot.pause()

        await pilot.pause(LOGS_FILTER_SAVE_DEBOUNCE_SECONDS + 0.3)
        for _ in range(10):
            await pilot.pause(0.05)

        assert calls == [], (
            "click-away-and-back must be recognized as a no-op, not written"
        )


# --- TASK-21235: a change landing while a debounced write is in flight ---------


@private_profile_test
@pytest.mark.asyncio
async def test_revert_during_in_flight_write_is_persisted_without_unmount(
    request, monkeypatch,
):
    """TASK-21235: the swallow. A filter change landing while a debounced
    write is in flight used to compare against the STALE baseline
    (`_persisted_filter_state` only advances on write completion), so a
    revert to the on-disk state was judged a no-op and swallowed -- nothing
    reached disk until unmount repaired it. The dictation module's
    task-15470 dirty shape replaces the scheduling gate, so the revert
    always arms its own timer and lands without leaving the screen.
    """
    import threading

    import tldw_chatbook.config as cfg

    gate = threading.Event()
    real_save = cfg.save_settings_to_cli_config
    writes: list[dict] = []

    def gated_save(settings, *args, **kwargs):
        writes.append(settings)
        if len(writes) == 1:
            # First write blocks on the worker thread until released: the
            # event loop keeps running, so the revert below lands while
            # this write is genuinely in flight.
            gate.wait(timeout=10.0)
        return real_save(settings, *args, **kwargs)

    monkeypatch.setattr(cfg, "save_settings_to_cli_config", gated_save)

    async with _LogsHarness().run_test(size=(120, 36)) as pilot:
        window = pilot.app.query_one(LogsWindow)
        window.load_from_app()
        await pilot.pause()
        original_chip = window._level_chip

        # Change the filter; let the debounce fire so a write is in flight.
        window.query_one("#logs-filter-error", Button).press()
        await pilot.pause()
        await pilot.pause(LOGS_FILTER_SAVE_DEBOUNCE_SECONDS + 0.1)
        for _ in range(40):
            if writes:
                break
            await pilot.pause(0.05)
        assert len(writes) == 1, "the first (blocked) write did not start"
        assert writes[0]["logs"]["last_level_chip"] == "error"

        # Revert to the original chip WHILE write #1 is still blocked.
        window.query_one("#logs-filter-all", Button).press()
        await pilot.pause()
        assert window._filter_save_timer is not None, (
            "TASK-21235 regression: the in-flight-window revert was "
            "swallowed against the stale baseline (no timer armed)"
        )

        gate.set()
        # The revert's own debounce fires and its write lands -- with no
        # unmount anywhere in this test.
        for _ in range(60):
            if len(writes) >= 2:
                break
            await pilot.pause(0.05)

        assert len(writes) >= 2, (
            "the revert never reached disk without unmount (the swallow)"
        )
        assert writes[-1]["logs"]["last_level_chip"] == original_chip
        # The last save may still be mid-flight on its worker thread when
        # the list record lands; poll the disk to the settled state.
        for _ in range(40):
            if _saved_logs_section().get("last_level_chip") == original_chip:
                break
            await pilot.pause(0.05)
        assert _saved_logs_section().get("last_level_chip") == original_chip
