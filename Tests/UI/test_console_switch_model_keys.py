"""Switch model from the composer: keystroke budgets and commit keys.

TASK-33004.5 AC#8, #9, #11, #12 and #18. Every path is real key presses on
the shipping Console screen, counted from the composer, against a scratch
config file the real default writer owns (``private_profile_test``). The
switcher fills readiness and catalogs in after Alt+M opens; each path waits
for that before typing, which costs no key.

Spec §2 budgets: Alex (switch to a Sonnet model, Temperature 0.9, Max tokens
8192, back to the composer) 14 keys; the A/B toggle 2; a model-only change 5;
a model change saved as the model default 16 or fewer.
"""

from __future__ import annotations

import asyncio
import tomllib

import pytest
from textual.widgets import Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app, attach_chachanotes_db
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_session_settings import (
    build_target_default_console_session_settings,
)
from tldw_chatbook.config import load_settings
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover

SON = ("s", "o", "n")
MODEL_ONLY = ("alt+m", *SON, "enter")
AB_TOGGLE = ("alt+m", "enter")
ALEX = ("alt+m", *SON, "tab", "0", ".", "9", "tab", "8", "1", "9", "2", "enter")
#: Model change plus one edit, saved as the model default: Shift+Tab from
#: Temperature reaches Find, and once more wraps to Save as model default.
SAVE_AS_MODEL_DEFAULT = (
    "alt+m",
    *SON,
    "tab",
    "0",
    ".",
    "9",
    "shift+tab",
    "shift+tab",
    "enter",
)
ANTHROPIC_MODELS = ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"]


class _Harness(ConsolidatedCSSApp):
    """The shipping Console screen under the production stylesheet."""

    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _console_app():
    """llama.cpp/model-a chat; Anthropic ready with a scratch key."""
    assert SettingsConfigAdapter().save_sections(
        {
            "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
            "api_settings.llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "model-a",
            },
            "api_settings.anthropic": {
                "api_key": "sk-ant-p4t5-scratch-not-a-real-key-0000",
                "model": "claude-sonnet-4-5",
                "streaming": True,
                "model_defaults": {
                    "claude-sonnet-4-5": {
                        "temperature": 1.0,
                        "max_tokens": 4096,
                        "unexposed": "kept",
                    },
                    "sibling/model": {"temperature": 0.4},
                },
            },
        }
    )
    app = _build_test_app()
    attach_chachanotes_db(app)
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.providers_models = {
        "llama_cpp": ["model-a"],
        "anthropic": list(ANTHROPIC_MODELS),
    }
    return app


class _Keys:
    """Real key presses, counted; waits for the switcher's fill cost none."""

    def __init__(self, harness: _Harness, pilot) -> None:
        self.harness = harness
        self.pilot = pilot

    async def run(self, keys: tuple[str, ...]) -> int:
        for key in keys:
            await self.pilot.press(key)
            await self.pilot.pause()
            if key == "alt+m":
                assert isinstance(self.harness.screen, ConsoleModelPopover)
                await self.harness.workers.wait_for_complete()
                await self.pilot.pause()
        return len(keys)


async def _drain(app) -> None:
    owner = app.console_settings_durability_owner
    for _ in range(50):
        if not owner.tasks:
            return
        await asyncio.gather(*tuple(owner.tasks))
    raise AssertionError(f"durability tasks still pending: {owner.tasks!r}")


def _composer(console: ChatScreen):
    return console.query_one("#console-native-composer")


async def _chips_show(console: ChatScreen, pilot, provider: str, model: str) -> None:
    """AC#9: the status chips name the applied pair (they refresh on the tick)."""

    def text(selector: str) -> str:
        return str(console.query_one(selector, Static).render())

    for _ in range(100):
        if provider in text("#console-provider-chip") and model in text(
            "#console-model-chip"
        ):
            return
        await pilot.pause(0.05)
    raise AssertionError(
        f"chips still show {text('#console-provider-chip')!r} "
        f"/ {text('#console-model-chip')!r}"
    )


@pytest.mark.asyncio
@private_profile_test
async def test_apply_paths_from_the_composer_meet_their_key_budgets(request) -> None:
    """AC#8, AC#9, AC#12, AC#18: model-only 5 keys, A/B 2, Alex 14.

    Enter applies to this chat through the live-commit path and writes no
    configuration; the switcher closes, focus returns to the composer and the
    chips show the new pair. Ctrl+O then carries an unapplied edit to Chat
    settings without applying it."""
    from tldw_chatbook.config import get_cli_config_path

    app = _console_app()
    harness = _Harness(app)
    config_path = get_cli_config_path()
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-provider-chip")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        _composer(console).focus()
        await pilot.pause()
        keys = _Keys(harness, pilot)
        before = config_path.read_bytes()

        def pair() -> tuple[str, str | None]:
            settings = store.session_settings(session_id)
            return (settings.provider, settings.model)

        def in_composer() -> bool:
            focused = harness.focused
            return focused is not None and _composer(console) in (
                focused,
                *focused.ancestors,
            )

        assert in_composer()
        assert await keys.run(MODEL_ONLY) == 5
        await _drain(app)
        assert pair() == ("anthropic", "claude-sonnet-4-5")
        assert harness.screen is console and in_composer()
        await _chips_show(console, pilot, "Anthropic", "claude-sonnet-4-5")

        assert await keys.run(AB_TOGGLE) == 2
        await _drain(app)
        assert pair() == ("llama_cpp", "model-a")
        assert harness.screen is console and in_composer()
        await _chips_show(console, pilot, "llama.cpp", "model-a")

        assert await keys.run(ALEX) == 14
        await _drain(app)
        applied = store.session_settings(session_id)
        assert (applied.provider, applied.model) == ("anthropic", "claude-sonnet-4-5")
        assert (applied.temperature, applied.max_tokens) == (0.9, 8192)
        assert harness.screen is console and in_composer()
        await _chips_show(console, pilot, "Anthropic", "claude-sonnet-4-5")
        assert config_path.read_bytes() == before

        # AC#12: Ctrl+O hands the highlighted pair and its edit to Chat
        # settings; the chat keeps its applied values until that applies.
        revision = store.session_settings_revision(session_id)
        await keys.run(("alt+m", "h", "a", "i", "tab", "0", ".", "3", "ctrl+o"))
        for _ in range(40):
            if not isinstance(
                harness.screen, ConsoleModelPopover
            ) and harness.screen.query("#console-settings-temperature"):
                break
            await pilot.pause(0.05)
        modal = harness.screen
        assert modal.query_one("#console-settings-temperature", Input).value == "0.3"
        assert store.session_settings(session_id).model == "claude-sonnet-4-5"
        assert store.session_settings_revision(session_id) == revision
    assert config_path.read_bytes() == before


@pytest.mark.asyncio
@private_profile_test
async def test_default_keys_write_the_three_values_through_the_real_writer(
    request,
) -> None:
    """AC#10, AC#11, AC#18: a model change saved as the model default takes
    11 keys (budget 16); Ctrl+N then makes another pair the default for new
    chats. Both write exactly Temperature, Max tokens and Streaming into the
    exact model profile, keep unexposed fields and sibling profiles, and Ctrl+N
    writes chat_defaults' provider and model."""
    from tldw_chatbook.config import get_cli_config_path

    app = _console_app()
    harness = _Harness(app)
    config_path = get_cli_config_path()

    def saved() -> dict:
        return tomllib.loads(config_path.read_text(encoding="utf-8"))

    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        _composer(console).focus()
        await pilot.pause()
        keys = _Keys(harness, pilot)

        count = await keys.run(SAVE_AS_MODEL_DEFAULT)
        assert count == 11 and count <= 16
        await _drain(app)
        assert harness.screen is console
        config = saved()
        profiles = config["api_settings"]["anthropic"]["model_defaults"]
        assert profiles["claude-sonnet-4-5"] == {
            "temperature": pytest.approx(0.9),
            "max_tokens": 4096,
            "streaming": True,
            "unexposed": "kept",
        }
        assert profiles["sibling/model"] == {"temperature": 0.4}
        assert config["chat_defaults"]["provider"] == "llama_cpp"
        settings = store.session_settings(store.active_session_id)
        assert (settings.provider, settings.model) == ("anthropic", "claude-sonnet-4-5")

        # AC#11: Ctrl+N, pressed while Max tokens holds a typed value. Haiku has
        # no profile yet: its Temperature is whatever the config chain
        # resolves, read from the saved config rather than from the field.
        resolved = build_target_default_console_session_settings(
            load_settings(force_reload=True), "anthropic", "claude-haiku-4-5"
        ).temperature
        assert resolved is not None
        await keys.run(("alt+m", "h", "a", "i", "tab", "tab", "8", "1", "9", "2"))
        shown = harness.screen.query_one("#console-popover-temperature", Input).value
        assert shown and float(shown) == pytest.approx(resolved), shown
        await keys.run(("ctrl+n",))
        await _drain(app)
        assert harness.screen is console
        config = saved()
        assert config["chat_defaults"]["provider"] == "anthropic"
        assert config["chat_defaults"]["model"] == "claude-haiku-4-5"
        profiles = config["api_settings"]["anthropic"]["model_defaults"]
        assert profiles["claude-haiku-4-5"] == {
            "temperature": pytest.approx(resolved),
            "max_tokens": 8192,
            "streaming": True,
        }
        assert profiles["claude-sonnet-4-5"]["temperature"] == pytest.approx(0.9)
        assert profiles["sibling/model"] == {"temperature": 0.4}

    assert app.console_default_durability_state.failure_phase is None
