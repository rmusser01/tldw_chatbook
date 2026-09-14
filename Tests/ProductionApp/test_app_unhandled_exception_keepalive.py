"""TASK-32533: a widget-handler exception keeps the screen alive and names its site.

Before this task ``TldwCli._handle_exception`` (TASK-1240) persisted only the
exception type and then let Textual exit the app, so the critique-3 crash left
``event=unhandled_exception exception_type=InvalidSelectValueError`` and
nothing else in the profile log -- the traceback went to the dead pane's
stderr. These tests run the REAL ``TldwCli`` under ``run_test`` and pin:

* a message-handler exception no longer exits the app (the screen stays, a
  notification names the failing handler, the message never leaks);
* the persisted record carries the raising frame as metadata (module,
  function, line -- never the message);
* a ``WorkerFailed`` still exits exactly as TASK-1240 specified;
* headless runs keep raising by default, so the suite never loses its
  exception signal.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from textual import work
from textual.widgets import Button
from textual.worker import WorkerFailed

from textual.message import Message

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import load_settings

_CANARY = "zqleakcanary-msg"
_DIAGNOSTICS_LOGGER = "tldw_chatbook.diagnostics.app"


class _RaisingButton(Button):
    """Raises from its own ``Button.Pressed`` handler -- a widget-handler crash."""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        raise RuntimeError(_CANARY)


class _WorkerCrashButton(Button):
    """Raises inside an ``exit_on_error`` worker -- the TASK-1240 path."""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self._crash()

    @work(exit_on_error=True)
    async def _crash(self) -> None:
        raise RuntimeError(_CANARY)


class _RecordSink(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def persisted_records() -> list[str]:
    logger = logging.getLogger(_DIAGNOSTICS_LOGGER)
    sink = _RecordSink()
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sink)
    try:
        yield sink.messages
    finally:
        logger.removeHandler(sink)
        logger.setLevel(previous_level)


class ZqBoom(Message):
    """A message the App itself handles -- dispatched on the App's own pump."""


class _AppLevelRaisingApp(TldwCli):
    """Raises from an App-level handler, like `@on(Worker.StateChanged)` would."""

    def on_zq_boom(self, message: ZqBoom) -> None:
        raise RuntimeError(_CANARY)


def _production_app(
    monkeypatch: pytest.MonkeyPatch, cls: type[TldwCli] = TldwCli
) -> TldwCli:
    """The real app from the sandbox config, splash off, wizard not offered."""

    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)
    app = cls()
    app.app_config = load_settings(force_reload=True)
    app.app_config["_first_run"] = False
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    return app


async def _close_production_app(app: TldwCli) -> None:
    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


async def _mount_button(app: TldwCli, pilot, button: Button) -> Button:
    for _ in range(500):
        if getattr(app, "_ui_ready", False):
            break
        await pilot.pause(0.01)
    else:
        raise AssertionError("production TldwCli never reached _ui_ready")
    await pilot.pause()
    await app.screen.mount(button)
    await pilot.pause()
    assert button.is_mounted
    return button


def _crash_records(messages: list[str]) -> list[str]:
    return [m for m in messages if "event=unhandled_exception" in m]


@pytest.mark.asyncio
async def test_a_handler_exception_keeps_the_screen_alive_and_notifies(
    monkeypatch: pytest.MonkeyPatch,
    persisted_records: list[str],
) -> None:
    app = _production_app(monkeypatch)
    app._keep_screen_alive_on_handler_error = True
    try:
        async with app.run_test(size=(140, 48)) as pilot:
            button = await _mount_button(
                app, pilot, _RaisingButton("boom", id="zq-raising-button")
            )
            screen_before = app.screen

            button.press()
            # Not `pilot.pause()`: it queues a callback on every widget of the
            # screen and waits for all of them, and the raising button's own
            # message loop breaks right after the handler -- a callback queued
            # there never runs. Poll the clock instead.
            for _ in range(100):
                await asyncio.sleep(0.02)
                if _crash_records(persisted_records) and button.parent is None:
                    break
            await asyncio.sleep(0.05)

            assert app.is_running, "a handler exception still exited the app"
            assert app._exception is None, "Textual recorded the crash as fatal"
            assert app.screen is screen_before
            # Textual's message loop exits after the handler and the widget
            # detaches itself from the DOM -- the "that panel may stop
            # responding" caveat in the notification copy.
            assert button.parent is None

            notifications = [
                notification.message for notification in app._notifications
            ]
            assert len(notifications) == 1, notifications
            # The raising frame is in this test module, not in `tldw_chatbook.`,
            # so the notification falls back to naming the pump -- exactly the
            # case the real P0 hit (a Textual Select failing at mount leaves no
            # Chatbook frame on the stack at all).
            assert "_RaisingButton#zq-raising-button" in notifications[0]
            assert _CANARY not in notifications[0]

            records = _crash_records(persisted_records)
            assert len(records) == 1, persisted_records
            record = records[0]
            assert "exception_type=RuntimeError" in record
            assert "site_function=on_button_pressed" in record
            assert "raise_function=on_button_pressed" in record
            assert "test_app_unhandled_exception_keepalive" in _field(
                record, "site_module"
            )
            assert _field(record, "raise_module") == _field(record, "site_module")
            assert _field(record, "raise_line").isdigit()
            assert _field(record, "site_line") == _field(record, "raise_line")
            assert _field(record, "widget_type") == "_RaisingButton"
            assert _field(record, "widget_id") == "zq-raising-button"
            assert _CANARY not in record
    finally:
        await _close_production_app(app)


def _field(record: str, name: str) -> str:
    for token in record.split():
        key, _, value = token.partition("=")
        if key == name:
            return value
    raise AssertionError(f"{name} missing from {record!r}")


@pytest.mark.asyncio
async def test_worker_failed_still_exits_like_before(
    monkeypatch: pytest.MonkeyPatch,
    persisted_records: list[str],
) -> None:
    """The TASK-1240 contract: a crashed exit_on_error worker still exits."""
    app = _production_app(monkeypatch)
    app._keep_screen_alive_on_handler_error = True
    try:
        with pytest.raises(WorkerFailed):
            async with app.run_test(size=(140, 48)) as pilot:
                button = await _mount_button(app, pilot, _WorkerCrashButton("boom"))
                button.press()
                for _ in range(100):
                    await pilot.pause(0.01)
                    if app._exception is not None:
                        break
                assert isinstance(app._exception, WorkerFailed)
        records = _crash_records(persisted_records)
        assert len(records) == 1, persisted_records
        assert "exception_type=RuntimeError" in records[0]
        assert "site_function=_crash" in records[0]
        # A worker runs outside any widget's dispatch, so no pump is named.
        assert "widget_type=" not in records[0]
        assert _CANARY not in records[0]
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_headless_runs_keep_raising_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under run_test the attribute defaults off, so the suite keeps its signal."""
    app = _production_app(monkeypatch)
    try:
        with pytest.raises(RuntimeError, match=_CANARY):
            async with app.run_test(size=(140, 48)) as pilot:
                button = await _mount_button(app, pilot, _RaisingButton("boom"))
                button.press()
                for _ in range(100):
                    await pilot.pause(0.01)
                    if app._exception is not None:
                        break
                assert isinstance(app._exception, RuntimeError)
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_an_app_level_handler_exception_still_reports_the_failure(
    monkeypatch: pytest.MonkeyPatch,
    persisted_records: list[str],
) -> None:
    """TASK-32533 review, Important #2: keep-alive must not swallow the App's own pump.

    `App._process_messages` runs the application loop. Skipping `super()` for a
    handler the App itself dispatches breaks out of that loop with no
    `_return_code` and no `panic()`: the app vanishes on exit 0 with no
    traceback and nothing in the log -- the P0's symptom with *less* evidence
    than before the fix. The `pump is not self` clause sends it to `super()`.
    """
    app = _production_app(monkeypatch, cls=_AppLevelRaisingApp)
    app._keep_screen_alive_on_handler_error = True
    try:
        with pytest.raises(RuntimeError, match=_CANARY):
            async with app.run_test(size=(140, 48)) as pilot:
                for _ in range(500):
                    if getattr(app, "_ui_ready", False):
                        break
                    await pilot.pause(0.01)
                app.post_message(ZqBoom())
                for _ in range(100):
                    await pilot.pause(0.01)
                    if app._exception is not None:
                        break
                # Textual recorded it as fatal, so the failure is reported
                # rather than unwinding the app loop in silence.
                assert isinstance(app._exception, RuntimeError)
                assert app.return_code == 1
        records = _crash_records(persisted_records)
        assert len(records) == 1, persisted_records
        assert "exception_type=RuntimeError" in records[0]
        assert _field(records[0], "widget_type") == "_AppLevelRaisingApp"
        assert _CANARY not in records[0]
        # No notification: the app is going down the normal way.
        assert [n.message for n in app._notifications] == []
    finally:
        await _close_production_app(app)
