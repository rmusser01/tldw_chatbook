"""Speech rail shortcuts and custom footer follow the active mounted view."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from pathlib import Path

from textual.widgets import Button, Static, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.background_signals import await_background_task, wait_for_signal
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen
from tldw_chatbook.UI.STTS_Window import STTSWindow
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Utils import platform_files

PLAYGROUND_HINTS = (
    "g generate",
    "r random text",
    "x clear",
    "p play",
    "s stop",
)


class SpeechHost(ConsolidatedCSSApp):
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    async def on_mount(self) -> None:
        self.screen_under_test = STTSScreen(self)
        await self.push_screen(self.screen_under_test)


async def _until(pilot, predicate) -> None:
    async with asyncio.timeout(10):
        while not predicate():
            await pilot.pause()


def _painted_hints(screen: STTSScreen) -> str:
    footer = screen.query_one(AppFooterStatus)
    return str(footer.query_one("#footer-key-quit", Static).content)


def _record_source_identity(request) -> None:
    root = Path(__file__).resolve().parents[2]
    sources = [
        STTSScreen,
        STTSWindow,
        SpeechPlaygroundPane,
        AppFooterStatus,
        platform_files,
    ]
    if type(platform_files.os).__module__.startswith("tldw_chatbook."):
        sources.append(type(platform_files.os))
    for cls in sources:
        source = Path(inspect.getfile(cls)).resolve()
        assert source.is_relative_to(root)
        request.node.user_properties.append(
            (
                cls.__name__,
                f"{source}:sha256={hashlib.sha256(source.read_bytes()).hexdigest()}",
            )
        )


@private_profile_test
async def test_footer_and_rail_actions_follow_playground_effects_return(
    request, monkeypatch
):
    """Stale registration or binding availability must fail after real rail Enter."""
    _record_source_identity(request)
    app = SpeechHost()
    mount_release = asyncio.Event()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = app.screen_under_test
        try:
            await _until(pilot, lambda: bool(screen.query(SpeechPlaygroundPane)))
            await app.workers.wait_for_complete()
            await pilot.pause()
            field = screen.query_one("#tts-text-input", TextArea)
            field.text = "ordinary rail Clear control"
            playground = screen.query_one("#lab-speech-row-playground", Button)
            playground.focus()
            await _until(pilot, lambda: app.focused is playground)
            await pilot.press("x")
            assert field.text == ""
            for hint in PLAYGROUND_HINTS:
                assert hint in _painted_hints(screen)

            effects = screen.query_one("#lab-speech-row-effects", Button)
            effects.focus()
            await _until(pilot, lambda: app.focused is effects)
            await pilot.press("enter")
            await _until(
                pilot,
                lambda: screen.stts_window.current_view == "effects"
                and not screen.query(SpeechPlaygroundPane)
                and bool(screen.query("#speech-effects-pane")),
            )
            await app.workers.wait_for_complete()
            await pilot.pause()
            effects.focus()
            await _until(pilot, lambda: app.focused is effects)
            painted = _painted_hints(screen)
            request.node.user_properties.append(("effects_footer_child", painted))
            for hint in PLAYGROUND_HINTS:
                assert hint not in painted, "Absent Playground still advertised"
            for hint in (
                "Move mode focus",
                "Enter Go",
                "F1 help",
                "F6 next pane",
                "Ctrl+P palette",
                "Ctrl+Q quit",
            ):
                assert hint in painted
            for key in ("g", "r", "x", "p", "s"):
                assert key not in screen.active_bindings
            await pilot.press("g", "r", "x", "p", "s")
            assert screen.stts_window.current_view == "effects"
            assert not screen.query(SpeechPlaygroundPane)

            # Mode focus remains functional while Playground shortcuts are absent.
            await pilot.press("]")
            assert app.focused.id.startswith("lab-mode-")
            # Park only the native target mount; keep the real view worker/state.
            content = screen.stts_window.query_one(".stts-content")
            native_mount = content.mount
            mount_started = asyncio.Event()

            async def gated_mount(*widgets, **kwargs):
                if any(isinstance(widget, SpeechPlaygroundPane) for widget in widgets):
                    mount_started.set()
                    await wait_for_signal(
                        mount_release, what="release Playground mount"
                    )
                return await native_mount(*widgets, **kwargs)

            monkeypatch.setattr(content, "mount", gated_mount)
            playground.focus()
            await _until(pilot, lambda: app.focused is playground)
            await pilot.press("enter")
            await wait_for_signal(mount_started, what="return Playground native mount")
            assert screen.stts_window.current_view == "playground"
            assert not screen.query(SpeechPlaygroundPane)
            for hint in PLAYGROUND_HINTS:
                assert hint not in _painted_hints(screen)
            for key in ("g", "r", "x", "p", "s"):
                assert key not in screen.active_bindings
            mount_release.set()
            await _until(
                pilot,
                lambda: screen.stts_window.current_view == "playground"
                and bool(screen.query(SpeechPlaygroundPane)),
            )
            await app.workers.wait_for_complete()
            await pilot.pause()
            playground.focus()
            await _until(pilot, lambda: app.focused is playground)
            for hint in PLAYGROUND_HINTS:
                assert hint in _painted_hints(screen)
            for key in ("g", "r", "x", "p", "s"):
                assert screen.active_bindings[key].node is screen
                assert screen.active_bindings[key].enabled
            field = screen.query_one("#tts-text-input", TextArea)
            field.text = "returned rail Clear control"
            await pilot.press("x")
            assert field.text == ""

            # Printable mirrors remain typing in the real editor, including x.
            field.focus()
            await _until(pilot, lambda: app.focused is field)
            await pilot.press("g", "r", "x", "p", "s")
            assert field.text == "grxps"
            request.node.user_properties.append(
                ("returned_footer_child", _painted_hints(screen))
            )
        finally:
            mount_release.set()
            await app.workers.wait_for_complete()


@private_profile_test
async def test_recompose_hides_cached_hints_and_rejects_old_window(
    request, monkeypatch
):
    """A retired owner's queued message cannot repopulate a replacement footer."""
    _record_source_identity(request)
    app = SpeechHost()
    native_started = asyncio.Event()
    native_release = asyncio.Event()
    load_started = asyncio.Event()
    load_release = asyncio.Event()
    task = None
    async with app.run_test(size=(200, 60)) as pilot:
        screen = app.screen_under_test
        try:
            await _until(pilot, lambda: bool(screen.query(SpeechPlaygroundPane)))
            await app.workers.wait_for_complete()
            await pilot.pause()
            old_window = screen.stts_window
            assert old_window is not None
            for hint in PLAYGROUND_HINTS:
                assert hint in _painted_hints(screen)

            native_recompose = BaseAppScreen.recompose
            native_load = STTSWindow._load_studio_preferences

            async def held_recompose(owner):
                native_started.set()
                await wait_for_signal(
                    native_release, what="release native screen removal"
                )
                await native_recompose(owner)

            async def held_load(window):
                load_started.set()
                await wait_for_signal(load_release, what="release native Studio load")
                await native_load(window)

            monkeypatch.setattr(BaseAppScreen, "recompose", held_recompose)
            monkeypatch.setattr(STTSWindow, "_load_studio_preferences", held_load)
            task = asyncio.create_task(screen.recompose())
            await wait_for_signal(native_started, what="native screen removal boundary")
            assert old_window.is_attached
            screen.post_message(STTSWindow.ContentChanged(old_window))
            await pilot.pause()
            for hint in PLAYGROUND_HINTS:
                assert hint not in _painted_hints(
                    screen
                ), "Recompose retained cached hint"
            assert screen._footer_shortcut_registration == (
                "stts",
                screen.LAB_FOOTER_SHORTCUTS,
            )
            native_release.set()
            await await_background_task(task, what="native screen recomposition")
            await wait_for_signal(load_started, what="fresh window Studio load")
            assert screen.stts_window is not old_window
            assert not old_window.is_attached
            assert not screen.query(SpeechPlaygroundPane)
            screen.post_message(STTSWindow.ContentChanged(old_window))
            await pilot.pause()
            for hint in PLAYGROUND_HINTS:
                assert hint not in _painted_hints(screen)
            for key in ("g", "r", "x", "p", "s"):
                assert key not in screen.active_bindings
            request.node.user_properties.append(
                ("recompose_absent_footer", _painted_hints(screen))
            )
            load_release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            for hint in PLAYGROUND_HINTS:
                assert hint in _painted_hints(screen)
            field = screen.query_one("#tts-text-input", TextArea)
            field.text = "recomposed Clear control"
            rail = screen.query_one("#lab-speech-row-playground", Button)
            rail.focus()
            await _until(pilot, lambda: app.focused is rail)
            await pilot.press("x")
            assert field.text == ""
        finally:
            native_release.set()
            load_release.set()
            if task is not None:
                await await_background_task(task, what="recomposition cleanup")
                await wait_for_signal(
                    load_started, what="fresh window cleanup readiness"
                )
            await app.workers.wait_for_complete()
