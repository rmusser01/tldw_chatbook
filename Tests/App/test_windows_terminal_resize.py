"""Windows resize admission at the app-selected Textual driver boundary."""

import sys
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.drivers.headless_driver import HeadlessDriver

from tldw_chatbook.Utils.text_selection_crash_guard import TextualAppGuards


class GuardedApp(TextualAppGuards, App):
    """Use the same compatibility boundary as TldwCli."""


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows console API")
@pytest.mark.asyncio
async def test_driver_enables_resize_on_each_start_and_restores_console(monkeypatch):
    from textual.drivers import win32
    from textual.drivers import windows_driver

    modes = {sys.__stdin__: 0x17, sys.__stdout__: 0x03}
    initial_modes = modes.copy()
    monkeypatch.setattr(win32, "get_console_mode", modes.__getitem__)
    monkeypatch.setattr(
        win32, "set_console_mode", lambda stream, mode: modes.__setitem__(stream, mode)
    )
    # Keep real driver start/stop/close and console-mode lifecycle; only replace
    # OS I/O threads, since pytest has no interactive console to read or paint.
    monkeypatch.setattr(
        windows_driver,
        "WriterThread",
        lambda *args: SimpleNamespace(start=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(
        win32,
        "EventMonitor",
        lambda *args: SimpleNamespace(start=lambda: None, join=lambda: None),
    )
    monkeypatch.setattr(windows_driver.WindowsDriver, "write", lambda *args: None)
    app = GuardedApp()
    driver = app.driver_class(app)
    try:
        for _ in range(2):
            driver.start_application_mode()
            assert modes[sys.__stdin__] & 0x0008, "Resize events must be enabled"
            assert modes[sys.__stdin__] & 0x0200, "VT input must stay enabled"
            driver.stop_application_mode()
            driver.close()
            assert modes == initial_modes
    finally:
        driver.close()


def test_explicit_constructor_driver_is_preserved():
    assert GuardedApp(driver_class=HeadlessDriver).driver_class is HeadlessDriver


def test_environment_driver_is_preserved(monkeypatch):
    from textual import constants

    monkeypatch.setattr(
        constants, "DRIVER", "textual.drivers.headless_driver:HeadlessDriver"
    )
    assert GuardedApp().driver_class is HeadlessDriver


@pytest.mark.skipif(sys.platform == "win32", reason="Default Unix driver")
def test_non_windows_driver_is_unchanged():
    assert GuardedApp().driver_class is App().driver_class
