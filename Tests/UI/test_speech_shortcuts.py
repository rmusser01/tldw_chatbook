"""Speech Playground shortcuts preserve actions without shadowing global keys."""

from __future__ import annotations

import pytest
from textual.widgets import Button, TextArea

from tldw_chatbook.UI.Screens.stts_screen import STTSScreen
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from Tests.UI.app_factory import _build_test_app

#: Pane-local chords that do not conflict with ADR-031's global/terminal keys.
SUPPORTED_PANE_SHORTCUTS = {
    "ctrl+g": "generate_tts",
    "ctrl+l": "clear_text",
}
FORBIDDEN_PANE_SHORTCUTS = {"ctrl+r", "ctrl+p", "ctrl+s"}


def _bindings() -> dict[str, str]:
    return {
        b.key: b.action
        for b in getattr(SpeechPlaygroundPane, "BINDINGS", [])
        if hasattr(b, "key")
    }


@pytest.mark.unit
@pytest.mark.parametrize("key,action", sorted(SUPPORTED_PANE_SHORTCUTS.items()))
def test_each_supported_pane_shortcut_is_bound(key, action):
    """Every approved pane-local chord keeps its documented action."""

    bound = _bindings()
    assert key in bound, f"{key} ({action}) is no longer bound"
    assert bound[key] == action, f"{key} now runs {bound[key]!r}, not {action!r}"


@pytest.mark.unit
@pytest.mark.parametrize("action", sorted(set(SUPPORTED_PANE_SHORTCUTS.values())))
def test_every_bound_action_exists(action):
    """A binding naming a missing method fails only when the key is pressed."""
    assert callable(getattr(SpeechPlaygroundPane, f"action_{action}", None)), (
        f"action_{action} does not exist, so its binding is dead"
    )


def test_pane_does_not_shadow_reserved_or_terminal_shortcuts() -> None:
    """Pane-local bindings leave global and terminal-convention chords free."""

    assert FORBIDDEN_PANE_SHORTCUTS.isdisjoint(_bindings())


# --------------------------------------------------------------------------
# STTSScreen's own g/r/x/p/s mirror (TASK-2951)
#
# STTSScreen.BINDINGS re-declares g/r/x/p/s (no ctrl -- see stts_screen.py)
# so the shortcuts work from the screen's landed state, where the nav rail
# holds focus rather than the pane's text area. Each mirrored action() looks
# up the mounted playground via `_playground()` and forwards the call to it.
# `_playground()` used to query a retired legacy playground widget that was
# never mounted in production -- `STTSWindow._mount_view` only ever mounts
# `SpeechPlaygroundPane` for the playground view -- so every mirrored press
# silently found nothing and did nothing (TASK-2951).


def _make_harness(app_instance):
    from textual.app import App

    class _Harness(App):
        def __init__(self, inner):
            super().__init__()
            self._app_instance = inner

        async def on_mount(self) -> None:
            await self.push_screen(STTSScreen(self._app_instance))

    return _Harness(app_instance)


async def _wait_until(pilot, predicate, *, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition did not become true")


@pytest.mark.asyncio
async def test_screen_level_random_text_mirror_reaches_the_mounted_pane():
    """Press plain 'r' from the landed (nav-focused) state; the pane's text
    area must receive sample text. Chosen over 'g' (generate_tts) because it
    has an observable effect with no TTS/network service required."""
    app = _make_harness(_build_test_app())

    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        assert isinstance(screen, STTSScreen)
        await _wait_until(pilot, lambda: bool(screen.query(SpeechPlaygroundPane)))
        # Let the pane's own post-mount catalog-loading workers (and any
        # recomposes they trigger) settle before touching focus -- racing
        # them made this flaky: a mid-flight recompose could still be
        # rebuilding the tree when the key was pressed.
        await app.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one(SpeechPlaygroundPane)
        text_area = pane.query_one("#tts-text-input", TextArea)
        text_area.text = ""

        # Land focus on the rail row, not the pane's TextArea: a focused
        # TextArea consumes plain printable keys as typing (ADR-031), which
        # would make this test pass for the wrong reason.
        rail_row = screen.query_one("#lab-speech-row-playground", Button)
        rail_row.focus()
        await _wait_until(pilot, lambda: app.focused is rail_row)

        await pilot.press("r")
        await _wait_until(pilot, lambda: text_area.text != "")

        assert text_area.text != "", (
            "screen-level 'r' mirror did not reach the mounted "
            "SpeechPlaygroundPane -- _playground() is looking up the wrong "
            "type"
        )


@pytest.mark.asyncio
async def test_screen_level_clear_text_mirror_reaches_the_mounted_pane():
    """Press plain 'x' from the landed state; the pane's text area must
    clear. Covers the other direction from the random-text test above --
    starting non-empty and ending empty, so a no-op action cannot pass it
    by accident."""
    app = _make_harness(_build_test_app())

    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        assert isinstance(screen, STTSScreen)
        await _wait_until(pilot, lambda: bool(screen.query(SpeechPlaygroundPane)))
        await app.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one(SpeechPlaygroundPane)
        text_area = pane.query_one("#tts-text-input", TextArea)
        text_area.text = "text the mirror must clear"

        rail_row = screen.query_one("#lab-speech-row-playground", Button)
        rail_row.focus()
        await _wait_until(pilot, lambda: app.focused is rail_row)

        await pilot.press("x")
        await _wait_until(pilot, lambda: text_area.text == "")

        assert text_area.text == "", (
            "screen-level 'x' mirror did not reach the mounted "
            "SpeechPlaygroundPane -- _playground() is looking up the wrong "
            "type"
        )
