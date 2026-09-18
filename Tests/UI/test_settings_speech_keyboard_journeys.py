"""Global Speech configuration exposes visible keyboard recovery actions."""

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_overview_search_journeys import _painted
from Tests.UI.test_settings_provider_keyboard_journeys import _edit, _settle, _tab_to
from Tests.UI.test_settings_speech_tts_panel import (
    _build_test_app,
    _open_speech_tts,
    _StyledDestinationHarness,
)


async def _button(host, pilot, selector):
    button = await _tab_to(host, pilot, selector)
    assert str(button.label) in _painted(host, button)
    return button


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(190, 55), (80, 24)])
@private_profile_test
async def test_speech_keyboard_actions_validation_and_draft_recovery(
    request, theme, size
):
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = await _open_speech_tts(host, pilot)
        panel = screen.query_one("#settings-speech-tts-panel")
        for field in (
            "default-profile",
            "default-provider",
            "model-policy",
            "model-value",
            "voice-policy",
            "voice-value",
            "output-format",
            "speed",
            "configure-provider",
            "openai-base-url",
        ):
            await _tab_to(host, pilot, f"#settings-speech-{field}")
        for action in ("save", "revert", "restore-defaults", "open-lab-bottom"):
            await _button(host, pilot, f"#settings-speech-{action}")
        await _edit(host, pilot, "#settings-speech-speed", "invalid")
        await _button(host, pilot, "#settings-speech-save")
        await pilot.press("enter")
        await _settle(host, pilot)
        assert screen.focused is screen.query_one("#settings-speech-speed", Input)
        assert str(screen.query_one("#settings-speech-speed-error", Static).renderable)
        assert panel.has_unsaved_changes()
        await _button(host, pilot, "#settings-speech-revert")
        await pilot.press("enter")
        await _settle(host, pilot)
        assert not panel.has_unsaved_changes()

        voice = await _tab_to(host, pilot, "#settings-speech-voice-value")
        for resized in ((80, 24), (170, 48), size):
            await pilot.resize_terminal(*resized)
            await _settle(host, pilot)
            assert screen.focused is voice
            _assert_painted(screen, voice)
        await _button(host, pilot, "#settings-speech-browse-voices")
        await _tab_to(host, pilot, "#settings-speech-voice-value")
        await pilot.press("enter", "end", "enter")
        await _settle(host, pilot)
        await _edit(host, pilot, "#settings-speech-custom-id-value", "review-voice")
        await _button(host, pilot, "#settings-speech-custom-id-confirm")
        await pilot.press("enter")
        await _settle(host, pilot)
        assert (
            screen.query_one("#settings-speech-voice-value", Select).value
            == "review-voice"
        )
        assert panel.has_unsaved_changes()
        for choice in ("cancel", "discard"):
            await pilot.press("escape", "/", *"Overview", "enter")
            # The category worker awaits this modal; waiting for all workers
            # here would prevent the test from answering its own prompt.
            await pilot.pause()
            button = host.screen.query_one(f"#global-speech-tts-leave-{choice}", Button)
            for _ in range(3):
                action = host.screen.focused
                _assert_painted(host.screen, action)
                assert str(action.label) in _painted(host, action)
                await pilot.press("tab")
                await pilot.pause()
            for _ in range(5):
                if host.screen.focused is button:
                    break
                await pilot.press("tab")
                await pilot.pause()
            assert host.screen.focused is button
            assert str(button.label) in _painted(host, button)
            await pilot.press("enter")
            await _settle(host, pilot)
            if choice == "cancel":
                assert screen.active_category == "speech-tts"
                assert panel.has_unsaved_changes()
            else:
                assert screen.active_category == "overview"


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_compact_browse_keyboard_handoff_keeps_the_default_provider(
    request, theme
):
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    host.theme = theme
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        provider = screen.query_one("#settings-speech-default-provider", Select).value
        await _button(host, pilot, "#settings-speech-browse-voices")
        await pilot.resize_terminal(80, 24)
        await _settle(host, pilot)
        button = screen.query_one("#settings-speech-browse-voices", Button)
        assert screen.focused is button
        _assert_painted(screen, button)
        assert str(button.label) in _painted(host, button)
        await pilot.press("enter")
        await _settle(host, pilot)
        # Capture the real navigation message at the harness boundary. The
        # destination owns discovery, which this configuration review excludes.
        assert host.seen_routes == ["stts"]
        assert host.seen_contexts == [
            {"view": "playground", "provider": provider, "intent": "refresh-voices"}
        ]
