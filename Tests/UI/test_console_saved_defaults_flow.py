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
from textual.widgets import Button, Collapsible, Input
from textual.worker import WorkerCancelled

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app, attach_chachanotes_db
from Tests.UI.test_console_provider_apply_defaults_flow import (  # noqa: F401
    _ConsoleFlowHarness,
    _drain_settings_tasks,
    _reset_default_intent_state,
)
from Tests.UI.test_destination_shells import _wait_for_selector
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
    await _wait_for_selector(console, pilot, "#console-settings-summary")
    store = console._ensure_console_chat_store()
    session_id = store.active_session_id
    store.replace_session_settings(session_id, WORK)
    store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="Keep my settings."
    )
    return console, store, session_id


async def _settle(harness, pilot) -> None:
    """Wait out the workers; the Console's exclusive sync workers cancel
    each other while Settings sits on top, and a cancelled one raises."""
    for _ in range(5):
        try:
            await harness.workers.wait_for_complete()
            break
        except WorkerCancelled:
            continue
    await pilot.pause()


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
async def test_use_saved_defaults_stages_what_a_new_blank_chat_resolves(
    request,
) -> None:
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
    """AC#8 (and AC#7): the F4 Settings screen saves new model defaults and a
    persisted chat with work keeps its values; Use saved defaults + Apply
    rewrites that conversation's durable snapshot, and Apply leaves the config
    file byte-identical.

    Nothing between the save and the modal is patched: Settings' own Save
    writes the profile, and Chat settings reads it through the production
    seam (``_provider_readiness_app_config`` re-sources ``load_settings()``).
    The ChaChaNotes DB is a file, so the conversation is persisted BEFORE the
    save and Apply's worker-thread flush reaches it (``:memory:`` is per
    connection).
    """
    from tldw_chatbook.config import get_cli_config_path
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    app = _console_app()
    db_path = request.getfixturevalue("tmp_path") / "chachanotes.db"
    app.chachanotes_db = CharactersRAGDB(str(db_path), "test-client")
    request.addfinalizer(app.chachanotes_db.close)
    harness = _ConsoleFlowHarness(app)
    config_path = get_cli_config_path()
    async with harness.run_test(size=(211, 44)) as pilot:
        console, store, session_id = await _console_with_work(app, harness, pilot)
        conversation_id = store.persist_session_if_needed(session_id)
        assert conversation_id is not None

        def snapshot():
            return store.persistence.get_conversation_generation_settings(
                conversation_id
            ).snapshot

        assert (snapshot().temperature, snapshot().max_tokens) == (
            pytest.approx(0.9),
            100,
        )

        # F4: Settings ▸ Providers & Models saves new model defaults.
        await harness.push_screen(SettingsScreen(app))
        await _settle(harness, pilot)
        settings = harness.screen
        settings.query_one("#settings-category-providers-models", Button).press()
        await _settle(harness, pilot)
        settings.query_one(
            "#settings-generation-defaults", Collapsible
        ).collapsed = False
        await _settle(harness, pilot)
        for selector, text in (
            ("#settings-model-profile-temperature", "0.25"),
            ("#settings-model-profile-max-tokens", "777"),
        ):
            field = settings.query_one(selector, Input)
            field.focus()
            await pilot.press("home", "shift+end", "backspace", *text)
            await _settle(harness, pilot)
            assert field.value == text
        await pilot.press("escape", "s")  # Settings' own Save
        await _settle(harness, pilot)
        saved = config_path.read_text()
        assert "model_defaults" in saved and "777" in saved, saved
        harness.pop_screen()
        await pilot.pause()
        assert harness.screen is console
        # Every Console redraw runs the D1 refresh; a chat with work keeps.
        assert console._session._active_console_session_settings() == WORK
        assert store.session_settings(session_id) == WORK
        assert (snapshot().temperature, snapshot().max_tokens) == (
            pytest.approx(0.9),
            100,
        )

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
        await _settle(harness, pilot)
        assert config_path.read_bytes() == before_apply

        applied = store.session_settings(session_id)
        assert (applied.provider, applied.model) == ("llama_cpp", "model-a")
        assert (applied.temperature, applied.max_tokens) == (pytest.approx(0.25), 777)
        durable = snapshot()
        assert (durable.provider, durable.model) == ("llama_cpp", "model-a")
        assert (durable.temperature, durable.max_tokens) == (pytest.approx(0.25), 777)


@pytest.mark.asyncio
@private_profile_test
async def test_a_blank_top_p_applies_summarises_and_sends(
    request, tmp_path, monkeypatch
) -> None:
    """Qodo #2992: Custom OpenAI 2 does not accept Top P, so Use saved defaults
    stages it blank and Apply commits it blank. Apply used to crash in the
    settings summary's float(); the summary now shows the field rows' Source
    word for the blank, Chat settings reopens on it, and a send carries no
    Top P."""
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
    from tldw_chatbook.Chat.console_session_settings import (
        CONSOLE_VALUE_SOURCE_WORDS,
        ConsoleValueLayer,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Widgets.Console import ConsoleComposerBar

    pair = {"provider": "custom_2", "model": "model-a"}
    app = _console_app(
        chat_defaults={**pair, "temperature": 0.3},
        **{"api_settings.custom_2": {"api_url": "http://127.0.0.1:9101/v1"}},
    )
    app.chat_api_provider_value, app.chat_api_model_value = pair.values()
    app.providers_models = {"custom_2": ["model-a"]}
    # File-backed: the send commits its trace from a worker thread.
    app.chachanotes_db = CharactersRAGDB(tmp_path / "chachanotes.db", "test-client")
    request.addfinalizer(app.chachanotes_db.close)
    captured: list[dict] = []
    monkeypatch.setitem(
        API_CALL_HANDLERS,
        "custom-openai-api-2",
        lambda **kwargs: captured.append(kwargs) or "reply",
    )
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        work = ConsoleSessionSettings(**pair, top_p=0.5, source="user")
        store.replace_session_settings(session_id, work)
        assert "top_p" not in supported_generation_fields(
            *pair.values(), app.app_config
        )

        modal = await _use_saved_defaults(harness, pilot)
        assert modal._build_draft().top_p is None
        await pilot.press("ctrl+enter")
        for _ in range(3):
            await pilot.pause()
        assert harness.screen is console
        await _drain_settings_tasks(app)
        await _settle(harness, pilot)

        applied = store.session_settings(session_id)
        assert (applied.top_p, applied.temperature) == (None, pytest.approx(0.3))
        word = CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.PROVIDER_SCALARS]
        summary = console._context_spend._build_console_settings_summary_state()
        assert f"P {word}" in summary.sampling_row, summary.sampling_row

        await pilot.press("ctrl+o")  # Chat settings reopens on the blank
        for _ in range(4):
            await pilot.pause()
        reopened = harness.screen
        assert isinstance(reopened, ConsoleSettingsModal)
        assert reopened.query_one("#console-settings-top-p", Input).value == ""
        await pilot.press("escape")
        for _ in range(3):
            await pilot.pause()
        assert harness.screen is console

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")
        await pilot.pause(0.1)
        console.query_one("#console-send-message", Button).press()
        for _ in range(200):
            if captured:
                break
            await pilot.pause(0.05)

    assert captured, "the Custom OpenAI 2 handler was never called"
    assert not {"topp", "top_p"} & set(captured[-1]), sorted(captured[-1])
    assert captured[-1]["temp"] == pytest.approx(0.3)
