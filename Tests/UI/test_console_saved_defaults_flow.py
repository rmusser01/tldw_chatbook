"""Use saved defaults on the real Console, config writer and Ctrl+T (TASK-33006.5).

ADR-095's 2026-09-26 amendment: a chat that holds work keeps its settings
when Settings saves new defaults, and adopts them only through Chat settings'
Use saved defaults, then Apply, which writes no configuration. Each test runs
in a private profile (a scratch ``TLDW_CONFIG_PATH``) and writes through
Settings' own config writer, then drives the shipping ``ChatScreen``, its
store and controller with real keys.
"""

from __future__ import annotations

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_provider_apply_defaults_flow import (  # noqa: F401
    _ConsoleFlowHarness,
    _drain_settings_tasks,
    _reset_default_intent_state,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.app_factory import _build_test_app, attach_chachanotes_db
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_provider_support import supported_generation_fields
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal
from tldw_chatbook.Widgets.Console.console_settings_saved_defaults import (
    USE_SAVED_DEFAULTS_ID,
)

LLAMA = {"api_url": "http://127.0.0.1:9099", "model": "model-a"}
#: A chat that holds work, with values no saved default gives.
WORK = ConsoleSessionSettings(
    provider="llama_cpp",
    model="model-a",
    base_url="http://127.0.0.1:9099",
    temperature=0.9,
    top_p=0.5,
    top_k=5,
    max_tokens=100,
    source="user",
)


def _console_app(**sections):
    """Write the profile through Settings' writer, then build the app from it."""
    adapter = SettingsConfigAdapter()
    assert adapter.save_sections(
        {
            "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
            "api_settings.llama_cpp": LLAMA,
            **sections,
        }
    )
    app = _build_test_app()
    attach_chachanotes_db(app)
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.providers_models = {"llama_cpp": ["model-a"]}
    return app


async def _console_with_work(app, harness, pilot):
    """Mount the Console and give its chat work: own settings and a message."""
    console = harness.screen_stack[-1]
    assert isinstance(console, ChatScreen)
    console._provider_readiness_app_config = lambda: app.app_config
    await _wait_for_selector(console, pilot, "#console-settings-summary")
    store = console._ensure_console_chat_store()
    session_id = store.active_session_id
    store.replace_session_settings(session_id, WORK)
    store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="Keep my settings."
    )
    return console, store, session_id


async def _use_saved_defaults(harness, pilot) -> ConsoleSettingsModal:
    """Ctrl+O, then Use saved defaults; returns the open Chat settings."""
    await pilot.press("ctrl+o")
    for _ in range(4):
        await pilot.pause()
    modal = harness.screen
    assert isinstance(modal, ConsoleSettingsModal)
    await harness.workers.wait_for_complete()
    await pilot.click(f"#{USE_SAVED_DEFAULTS_ID}")
    for _ in range(3):
        await pilot.pause()
    return modal


@pytest.mark.asyncio
@private_profile_test
async def test_use_saved_defaults_stages_what_a_new_blank_chat_resolves(request) -> None:
    """AC#5: Use saved defaults stages exactly Ctrl+T's values for the pair.

    The chain has a value at every level a new chat reads: the model profile
    (Max tokens), ``[console.provider_defaults.llama_cpp]`` (Temperature,
    Top K) and ``chat_defaults`` (Top P). Fields the provider does not accept
    are hidden and never sent, so the comparison is over the fields it does.
    """
    app = _console_app(
        **{
            "api_settings.llama_cpp": {
                **LLAMA,
                "model_defaults": {"model-a": {"max_tokens": 777}},
            },
            "chat_defaults": {
                "provider": "llama_cpp",
                "model": "model-a",
                "top_p": 0.81,
            },
            "console.provider_defaults.llama_cpp": {"temperature": 0.55, "top_k": 33},
        }
    )
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console, store, session_id = await _console_with_work(app, harness, pilot)
        modal = await _use_saved_defaults(harness, pilot)
        staged = modal._build_draft()
        assert (staged.provider, staged.model) == ("llama_cpp", "model-a")
        assert (staged.temperature, staged.top_k, staged.max_tokens, staged.top_p) == (
            pytest.approx(0.55),
            33,
            777,
            pytest.approx(0.81),
        )

        await pilot.press("escape")  # the unsaved prompt
        await pilot.pause()
        await pilot.press("d")  # discard: nothing reached the chat
        await pilot.pause()
        assert harness.screen is console
        assert store.session_settings(session_id) == WORK

        await pilot.press("ctrl+t")
        for _ in range(3):
            await pilot.pause()
        assert store.active_session_id != session_id
        blank = store.session_settings(store.active_session_id)
        accepted = supported_generation_fields("llama_cpp", "model-a", app.app_config)
        assert accepted >= {"temperature", "top_k", "max_tokens", "top_p"}
        assert (blank.provider, blank.model) == (staged.provider, staged.model)
        assert {name: getattr(staged, name) for name in accepted} == {
            name: getattr(blank, name) for name in accepted
        }


@pytest.mark.asyncio
@private_profile_test
async def test_saved_model_defaults_reach_a_chat_with_work_only_through_apply(
    request,
) -> None:
    """AC#8 (and AC#7): Settings saves, the chat keeps; Use saved defaults +
    Apply moves the new values into the conversation's snapshot, and Apply
    leaves the config file byte-identical."""
    from tldw_chatbook.config import get_cli_config_path

    app = _console_app()
    harness = _ConsoleFlowHarness(app)
    config_path = get_cli_config_path()
    async with harness.run_test(size=(211, 44)) as pilot:
        console, store, session_id = await _console_with_work(app, harness, pilot)

        # Settings saves new model defaults through its writer, then reloads.
        adapter = SettingsConfigAdapter()
        assert adapter.save_sections(
            {
                "api_settings.llama_cpp": {
                    **LLAMA,
                    "model_defaults": {
                        "model-a": {"temperature": 0.25, "max_tokens": 777}
                    },
                }
            }
        )
        app.app_config = adapter.load(force_reload=True)
        # Every Console redraw runs the D1 refresh; a chat with work keeps.
        assert console._session._active_console_session_settings() == WORK
        assert store.session_settings(session_id) == WORK

        modal = await _use_saved_defaults(harness, pilot)
        assert modal.query_one("#console-settings-temperature").value == "0.25"
        assert modal.query_one("#console-settings-max-tokens").value == "777"
        assert store.session_settings(session_id) == WORK  # staged only

        before_apply = config_path.read_bytes()
        await pilot.press("ctrl+enter")
        for _ in range(3):
            await pilot.pause()
        assert harness.screen is console
        await _drain_settings_tasks(app)
        assert config_path.read_bytes() == before_apply

        applied = store.session_settings(session_id)
        assert (applied.provider, applied.model) == ("llama_cpp", "model-a")
        assert (applied.temperature, applied.max_tokens) == (pytest.approx(0.25), 777)
        # The conversation's durable snapshot: persisted here on the main
        # thread, as the lifecycle test does, because the harness's in-memory
        # DB is per connection and the worker-thread flush cannot reach it.
        conversation_id = store.persist_session_if_needed(session_id)
        assert conversation_id is not None
        snapshot = store.persistence.get_conversation_generation_settings(
            conversation_id
        ).snapshot
        assert (snapshot.provider, snapshot.model) == ("llama_cpp", "model-a")
        assert (snapshot.temperature, snapshot.max_tokens) == (
            pytest.approx(0.25),
            777,
        )
