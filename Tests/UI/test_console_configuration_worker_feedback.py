"""Natural headless frame/input readiness during original native configuration.

These controls never force a render. They report actual input/frame intervals;
500 ms held-reader liveness is not the separate 100 ms acceptance target or
physical terminal write/flush qualification.
"""

import asyncio
import contextlib
import sys
import threading
import time

import pytest
from textual._compositor import ChopsUpdate, LayoutUpdate
from textual.app import App
from textual.screen import Screen
from textual.widgets import Button

from Tests.Chat.test_console_configuration_worker_lifetime import (
    _OriginalConfigurationWorkspaceRead,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_console_hook_review_send_freeze import (
    NavigatedConsoleHarness,
    _key,
    _task_factory,
    _until,
)
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class _VisibleNavigatedConsoleHarness(NavigatedConsoleHarness):
    # The navigation harness omits app-tier CSS; actual Console geometry
    # requires the same boot bundle loaded by TldwCli.
    CSS_PATH = [str(BUNDLED_STYLESHEET)]


def _update_contains(update, region, marker):
    """Inspect only cells already supplied by the original compositor update."""
    if isinstance(update, LayoutUpdate):
        for y, strips in enumerate(update.strips, update.region.y):
            if not region.y <= y < region.bottom:
                continue
            x = update.region.x
            for strip in strips:
                start, end = max(x, region.x), min(x + strip.cell_length, region.right)
                if start < end and marker in strip.crop(start - x, end - x).text:
                    return True
                x += strip.cell_length
    elif isinstance(update, ChopsUpdate):
        for y, left, right in update.spans:
            if not region.y <= y < region.bottom:
                continue
            for x, strip in update.chops[y].items():
                if strip is None:
                    continue
                start = max(x, left, region.x)
                end = min(x + strip.cell_length, right, region.right)
                if start < end and marker in strip.crop(start - x, end - x).text:
                    return True
    return False


class _NaturalInputFrame:
    def __init__(self, host, console, composer, probe):
        self.host, self.console, self.composer, self.probe = (
            host,
            console,
            composer,
            probe,
        )
        self.edit_serial = composer.edit_serial
        self.insert_code = ConsoleComposerBar.insert_text.__code__
        self.display_code = App._display.__code__
        self.refresh_code = Screen._compositor_refresh.__code__
        self.sent_at = self.mutated_at = self.frame_at = None
        self.mutated_while_held = self.frame_while_held = False
        self.painted = threading.Event()

    def _returned(self, code, _offset, _value):
        if self.sent_at is None or code not in (self.insert_code, self.display_code):
            return
        frame = sys._getframe(1)
        if code is self.insert_code:
            if (
                self.mutated_at is None
                and frame.f_locals.get("self") is self.composer
                and self.composer.edit_serial != self.edit_serial
                and self.composer.draft_text().endswith("Z")
            ):
                self.mutated_at = time.perf_counter()
                self.mutated_while_held = not self.probe.release.is_set()
            return
        if (
            self.frame_at is not None
            or self.mutated_at is None
            or frame.f_locals.get("self") is not self.host
            or frame.f_locals.get("screen") is not self.console
            or self.host.screen is not self.console
        ):
            return
        parent = frame.f_back
        while parent is not None and parent.f_code is not self.refresh_code:
            parent = parent.f_back
        if parent is None:
            return
        region = self.console.query_one("#console-command-visible-text").region
        if _update_contains(frame.f_locals.get("renderable"), region, "Z"):
            self.frame_at = time.perf_counter()
            self.frame_while_held = not self.probe.release.is_set()
            self.painted.set()

    @contextlib.contextmanager
    def installed(self):
        monitoring = sys.monitoring
        tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
        monitoring.use_tool_id(tool, "configuration-natural-input-frame")
        try:
            monitoring.register_callback(
                tool, monitoring.events.PY_RETURN, self._returned
            )
            for code in (self.insert_code, self.display_code):
                monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
            yield self
        finally:
            for code in (self.insert_code, self.display_code):
                monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            monitoring.free_tool_id(tool)


@pytest.mark.parametrize("route", ["enter", "send-button"])
async def test_actual_send_composes_new_input_while_configuration_reader_is_held(
    route, monkeypatch, record_property
):
    from tldw_chatbook import config

    # Native source bindings require the same bootstrap profile as config.
    app = _build_test_app(user_data_dir=config.get_user_data_dir())
    _configure_native_ready_console(app)
    registry = app.workspace_registry_service
    workspace_id = f"feedback-owned-{route}"
    registry.create_workspace(workspace_id=workspace_id, name=f"Feedback owned {route}")
    # Only the eventual network boundary is immediate; capture, admission and
    # native readers retain their production bodies even on a failing baseline.
    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        lambda **_kwargs: "configuration feedback reply",
    )
    host = _VisibleNavigatedConsoleHarness(app)
    runtime = None
    probe = None
    releaser = None
    stop = threading.Event()
    try:
        with _task_factory("eager"):
            async with host.run_test(size=(120, 40)) as pilot:
                assert await _until(
                    lambda: isinstance(host.screen, ChatScreen)
                    and host.screen.is_mounted,
                    10,
                )
                console = host.screen
                runtime = console._console_runtime()
                controller = console._ensure_console_chat_controller()
                session = controller.store.ensure_session()
                session.workspace_id = workspace_id
                composer = console._console_composer_or_none()
                composer.load_draft("Native configuration")
                composer.focus()
                await pilot.pause()
                assert host.focused is composer
                draft_region = console.query_one("#console-command-visible-text").region
                assert (
                    draft_region and draft_region.overlaps(console.region)
                ), "the normal composer must be visible before measuring input/frame readiness"
                probe = _OriginalConfigurationWorkspaceRead(registry)
                feedback = _NaturalInputFrame(host, console, composer, probe)
                loop = asyncio.get_running_loop()
                caller_thread = threading.current_thread()

                def release_independently():
                    try:
                        while not probe.entered.wait(0.01):
                            if stop.is_set():
                                return
                        feedback.sent_at = time.perf_counter()
                        loop.call_soon_threadsafe(_key, host, "Z", "Z")
                        feedback.painted.wait(0.5)
                    finally:
                        probe.release.set()

                with probe.installed(), feedback.installed():
                    releaser = threading.Thread(target=release_independently)
                    releaser.start()
                    try:
                        if route == "enter":
                            _key(host, "enter", "\r")
                        else:
                            console.query_one("#console-send-message", Button).press()
                        assert await _until(
                            probe.entered.is_set, 10
                        ), "stock Send did not reach original configuration Workspace SQL"
                        assert await _until(probe.release.is_set, 2)
                        await asyncio.sleep(0)
                        for name, at in (
                            ("input_mutation", feedback.mutated_at),
                            ("natural_frame", feedback.frame_at),
                        ):
                            record_property(
                                name + "_seconds",
                                None if at is None else at - feedback.sent_at,
                            )
                        record_property("headless_frame_only", True)
                        record_property("held_reader_budget_seconds", 0.5)
                        assert probe.live_at_entry and not probe.release_timed_out
                        assert (
                            probe.thread is not caller_thread
                        ), "stock mounted capture executes native configuration on the input loop"
                        assert feedback.mutated_while_held, "driver input did not mutate the composer during native capture"
                        assert feedback.frame_while_held, "the normal compositor did not publish the changed input during capture"
                        assert composer.draft_text().endswith("Z")
                    finally:
                        # Original Send may have captured before a RED loop stall.
                        # Cancel its actual owner and retain all newer draft text.
                        controller.stop_active_run()
                        probe.release.set()
                        assert await _until(lambda: not console._hooks._busy, 15)
                    # Prove this issued connection/lease closed before host or
                    # runtime teardown can perform unrelated fixture cleanup.
                    assert await _until(
                        probe.retired, 5
                    ), "configuration completed without retiring its native reader"
    finally:
        stop.set()
        if probe is not None:
            probe.release.set()
        if releaser is not None:
            releaser.join(timeout=2)
            assert not releaser.is_alive()
        if runtime is not None:
            await runtime.dispose()
