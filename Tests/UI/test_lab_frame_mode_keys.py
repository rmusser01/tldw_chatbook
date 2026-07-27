"""Focus-based mode switching on the Lab frame."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button

from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.LLM_Calls.huggingface_api import HuggingFaceAPI
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.lab_mode_strip import LAB_MODE_CHIP_IDS
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from Tests.UI.test_screen_navigation import _build_test_app


@pytest.fixture(autouse=True)
def _deterministic_models_mount(monkeypatch):
    """Neutralise two pre-existing, unrelated timing hazards this file's
    press/pause sequences are long enough to occasionally hit (found during
    Task 7 review: 3/5 runs failed before this fixture existed).

    1. Splash-screen race. ``SplashScreen.on_mount`` (``splash_screen.py``
       ~295-297) starts a REAL 1.5s wall-clock ``set_timer`` whose callback
       mounts the app's actual default-tab screen regardless of what a test
       has since pushed. Observed directly (via an instrumented, throwaway
       run): it fired mid-test and pushed a brand-new ``ChatScreen`` on top
       of this file's already-pushed ``LLMScreen``, whose own
       ``ConsoleSetupModal`` then auto-focused its action button and stole
       ``app.focused`` out from under an assertion. Forcing
       ``splash_screen.enabled`` False here makes ``TldwCli.compose()``
       (``app.py`` ~5934) skip the splash branch and mount the main UI
       immediately, before any test below ever pushes ``LLMScreen`` -- so
       there is no later auto-transition left to race against. Every other
       section/key still resolves through the real ``get_cli_setting``, so
       this does not change any other test-environment behaviour.
    2. Live network call. ``ModelSearchWidget.on_mount`` -> ``_initial_browse``
       -> ``perform_search()`` (``model_search_widget.py`` ~142-272) fires a
       real ``HuggingFaceAPI.search_models`` HTTP request to huggingface.co
       the moment ``LLMScreen``'s body mounts -- confirmed independently:
       the search widget lives inside ``llm-view-download-models``, which
       ``LLMManagementWindow.compose()`` builds eagerly. That request's
       variable real-world latency was the other half of the timing budget
       that let cause (1) surface at all. Patching it to an async no-op
       keeps this file's timing deterministic and independent of network
       reachability.

    Args:
        monkeypatch: pytest's monkeypatch fixture; reverts both patches
            automatically at the end of each test.
    """

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(HuggingFaceAPI, "search_models", AsyncMock(return_value=[]))


async def _models(app):
    screen = LLMScreen(app)
    await app.push_screen(screen)
    return screen


class _NavigationProbeLLMScreen(LLMScreen):
    """An ``LLMScreen`` that records ``NavigateToScreen`` instead of letting
    it reach the app's real handler.

    Used only by the test below: it needs a way to prove focus movement
    posts no navigation, and ``event.stop()`` here keeps that a self-
    contained check rather than triggering the app's actual navigation.
    """

    def __init__(self, app_instance):
        super().__init__(app_instance)
        self.navigated: list[str] = []

    def on_navigate_to_screen(self, message: NavigateToScreen) -> None:
        self.navigated.append(message.screen_name)
        message.stop()


@pytest.mark.asyncio
async def test_bracket_moves_focus_along_the_strip_without_navigating():
    """`[`/`]` must move focus only -- Enter on the focused chip is what
    navigates (see ``LabScreen.action_lab_mode_focus``'s docstring).

    ``navigated`` used to be a local list nothing ever appended to, so
    ``assert navigated == []`` passed vacuously regardless of whether
    bracket-key focus movement posted ``NavigateToScreen`` or not. This
    intercepts the message for real via ``_NavigationProbeLLMScreen``.
    """
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = _NavigationProbeLLMScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        screen.query_one(f"#{LAB_MODE_CHIP_IDS[0]}", Button).focus()
        await pilot.pause()

        await pilot.press("right_square_bracket")
        await pilot.pause()

        assert app.focused is not None
        assert app.focused.id == LAB_MODE_CHIP_IDS[1]
        assert screen.navigated == [], "moving focus must not navigate"


@pytest.mark.asyncio
async def test_bracket_wraps_at_both_ends():
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models(app)
        await pilot.pause()
        await pilot.pause()

        screen.query_one(f"#{LAB_MODE_CHIP_IDS[0]}", Button).focus()
        await pilot.pause()
        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert app.focused.id == LAB_MODE_CHIP_IDS[-1]

        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert app.focused.id == LAB_MODE_CHIP_IDS[0]


@pytest.mark.asyncio
async def test_bracket_starts_from_the_active_chip_when_nothing_is_focused():
    """With focus elsewhere, the first press should land beside the active
    mode rather than jumping to an arbitrary end of the strip."""
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models(app)
        await pilot.pause()
        await pilot.pause()
        screen.set_focus(None)
        await pilot.pause()

        await pilot.press("right_square_bracket")
        await pilot.pause()

        assert app.focused.id == LAB_MODE_CHIP_IDS[1]


@pytest.mark.asyncio
async def test_the_footer_advertises_the_mode_keys():
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models(app)
        await pilot.pause()
        await pilot.pause()
        footer = screen.query_one(AppFooterStatus)
        # `shortcut_text` is the assertable surface; AppFooterStatus has no
        # render() of its own. Existing hint tests use the same property.
        assert "[ / ]" in footer.shortcut_text
        # "Move mode focus", not "Switch mode": the action moves focus along
        # the strip and never navigates -- the adjacent `Enter  Go` hint is
        # the half that commits. The footer must not promise otherwise.
        assert "Move mode focus" in footer.shortcut_text
        assert "Switch mode" not in footer.shortcut_text
