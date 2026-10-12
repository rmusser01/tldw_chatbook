"""The Get started card opens Switch model (TASK-33008.3).

The mounted tests press real keys on the shipping Console screen, against a
scratch config file the real writers own (``private_profile_test``):

- "Use detected <server>" lists a stub loopback server's two models, and
  picking the second saves ``chat_defaults`` and the endpoint (AC#8).
- Choose model and Choose provider open Switch model, not Chat settings; an
  endpoint recovery still opens Chat settings (AC#4).
- A pick becomes the new-chat default only while none works, and the scope
  line says which before Enter (AC#5, AC#6).
"""

from __future__ import annotations

import json
import threading
import tomllib
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app, attach_chachanotes_db
from Tests.UI.test_console_model_switcher import line_with, list_lines
from Tests.UI.test_console_switch_model_entry_points import _switcher_ready, _until
from Tests.UI.test_console_switch_model_keys import _drain, _Harness
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.local_server_discovery import (
    DiscoveredLocalServer,
    probe_models_endpoint,
)
from tldw_chatbook.UI.Console_Modules import connection_probe, model_switcher
from tldw_chatbook.UI.Console_Modules.connection_probe import (
    switcher_connection_prober,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal
from tldw_chatbook.Widgets.Console.console_setup_modal import (
    CONSOLE_SETUP_MODAL_ACTION_ID,
    CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID,
    ConsoleSetupModal,
)

STUB_MODELS = ("stub-model-a", "stub-model-b")
SAVED_LLAMA = "http://127.0.0.1:9099"


class _StubModels(BaseHTTPRequestHandler):
    """A llama.cpp-shaped server: it lists two models and nothing else."""

    def do_GET(self) -> None:
        self.server.paths.append(self.path)  # type: ignore[attr-defined]
        if self.path.rstrip("/") != "/v1/models":
            self.send_error(404)
            return
        body = json.dumps(
            {"object": "list", "data": [{"id": model} for model in STUB_MODELS]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        """Keep the test output quiet."""


@contextmanager
def _stub_llama_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubModels)
    server.paths = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def _console_app(sections: dict, providers_models: dict):
    """A factory app over a scratch config holding exactly ``sections``."""
    assert SettingsConfigAdapter().save_sections(sections)
    app = _build_test_app()
    attach_chachanotes_db(app)
    app.providers_models = providers_models
    return app


def _saved() -> dict:
    from tldw_chatbook.config import get_cli_config_path

    return tomllib.loads(get_cli_config_path().read_text(encoding="utf-8"))


def _config_bytes() -> bytes:
    from tldw_chatbook.config import get_cli_config_path

    return get_cli_config_path().read_bytes()


async def _blocking_card(
    harness: _Harness, pilot
) -> tuple[ChatScreen, ConsoleSetupModal]:
    console = harness.screen
    assert isinstance(console, ChatScreen)
    await _wait_for_selector(console, pilot, "#console-setup-modal")
    modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
    await _until(pilot, lambda: modal.is_blocking, "the Get started card")
    return console, modal


async def _focused_action(harness: _Harness, console, pilot, action_id: str) -> Button:
    action = console.query_one(f"#{action_id}", Button)
    await _until(
        pilot,
        lambda: action.display and harness.focused is action,
        f"focus on #{action_id}",
    )
    return action


def _pair(console: ChatScreen) -> tuple[str, str | None]:
    store = console._ensure_console_chat_store()
    settings = store.session_settings(store.active_session_id)
    return (settings.provider, settings.model)


def _scope(switcher: ConsoleModelPopover) -> str:
    return str(switcher.query_one("#console-popover-scope", Static).render())


async def _composer_ready(harness: _Harness, console: ChatScreen, pilot) -> None:
    """The card is gone and the next key types into the composer."""
    composer = console.query_one("#console-native-composer")
    await _until(
        pilot,
        lambda: (
            harness.focused is not None
            and composer in (harness.focused, *harness.focused.ancestors)
        ),
        f"focus in the composer (focused: {harness.focused!r})",
    )


@pytest.mark.asyncio
@pytest.mark.loopback_network
@pytest.mark.local_server_probe
@pytest.mark.parametrize(
    "llama_cpp",
    [
        # "" means none: a deleted key loads its shipped default.
        pytest.param({"api_url": "", "model": ""}, id="no-endpoint-saved"),
        # Every profile carries the template's http://localhost:8080: that is
        # no endpoint the user chose, so it is no task-16476 case either.
        pytest.param(None, id="untouched-shipped-profile"),
        # Editing another field leaves the endpoint the template's.
        pytest.param({"temperature": 0.5}, id="shipped-endpoint-other-field-edited"),
    ],
)
@private_profile_test
async def test_detected_stub_server_second_model_is_saved_by_the_real_writer(
    monkeypatch, request, llama_cpp
) -> None:
    """AC#8 (and R12, no endpoint saved): a stub loopback server lists two
    models; Enter on the card, Down, Enter picks the second. The real writer
    saves ``chat_defaults`` and the detected endpoint, and the chat holds the
    pair, with no "Keeping the saved endpoint" warning. The discovery probe
    and the switcher's open-time probe both run for real, against the stub
    only; nothing is written before the second Enter."""
    with _stub_llama_server() as server:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        sections: dict = {"chat_defaults": {"provider": "llama_cpp", "model": ""}}
        if llama_cpp is not None:
            sections["api_settings.llama_cpp"] = llama_cpp
        app = _console_app(sections, {"llama_cpp": []})
        if "api_url" not in (llama_cpp or {}):
            assert _saved()["api_settings"]["llama_cpp"]["api_url"] == (
                "http://localhost:8080"
            )
        notices: list[str] = []
        monkeypatch.setattr(
            app, "notify", lambda message, **_kwargs: notices.append(str(message))
        )

        async def discover(_config):
            found = await probe_models_endpoint(url, provider_key="llama_cpp")
            assert found.ok, found.detail
            return (
                DiscoveredLocalServer("llama_cpp", found.base_url, found.model_ids),
            )

        app.console_local_server_discovery = discover

        def stub_only(running_app, config):
            # Tests/conftest.py shuts this seam; open it for the detected
            # server alone, so no other local port on this machine is probed.
            probe = switcher_connection_prober(running_app, config)

            async def only_llama_cpp(targets, settled):
                await probe(
                    {
                        key: model
                        for key, model in targets.items()
                        if key == "llama_cpp"
                    },
                    settled,
                )

            return only_llama_cpp

        monkeypatch.setattr(connection_probe, "switcher_connection_prober", stub_only)
        harness = _Harness(app)
        async with harness.run_test(size=(211, 44)) as pilot:
            console, modal = await _blocking_card(harness, pilot)
            await _focused_action(
                harness, console, pilot, CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID
            )
            before = _config_bytes()

            await pilot.press("enter")
            switcher = await _switcher_ready(harness, pilot)
            await _until(
                pilot,
                lambda: server.paths.count("/v1/models") >= 2,
                "the switcher's probe of the detected endpoint",
            )
            await harness.workers.wait_for_complete()
            await pilot.pause()
            lines = list_lines(harness, switcher)
            for model in STUB_MODELS:
                assert "Ready" in line_with(lines, model, "llama.cpp")
            row = switcher.highlighted_row()
            assert (row.provider, row.model) == ("llama_cpp", STUB_MODELS[0])
            assert _scope(switcher) == model_switcher.SCOPE_THIS_CHAT_AND_NEW_CHATS

            await pilot.press("down")
            row = switcher.highlighted_row()
            assert (row.provider, row.model) == ("llama_cpp", STUB_MODELS[1])
            assert _config_bytes() == before

            await pilot.press("enter")
            await _until(
                pilot, lambda: harness.screen is console, "the switcher to close"
            )
            await _drain(app)
            await _until(pilot, lambda: not modal.is_blocking, "the card to unlock")
            await _composer_ready(harness, console, pilot)

            saved = _saved()
            assert saved["chat_defaults"]["provider"] == "llama_cpp"
            assert saved["chat_defaults"]["model"] == STUB_MODELS[1]
            assert saved["api_settings"]["llama_cpp"]["api_url"] == url
            assert saved["api_settings"]["llama_cpp"]["model"] == STUB_MODELS[1]
            assert _pair(console) == ("llama_cpp", STUB_MODELS[1])
            store = console._ensure_console_chat_store()
            assert store.session_settings(store.active_session_id).base_url == url
            assert not [notice for notice in notices if "Keeping" in notice]


@pytest.mark.asyncio
@private_profile_test
async def test_card_choose_model_opens_switch_model_and_sets_the_new_chat_default(
    monkeypatch, request
) -> None:
    """AC#4, AC#5, AC#6. Rewritten on purpose from
    ``test_console_missing_model_opens_console_settings_from_summary``, which
    pinned Choose model opening Chat settings.

    With no working new-chat default, the card's Choose model opens Switch
    model in apply mode; the scope line names the new-chat default; a pair
    that cannot send is refused, as Ctrl+N refuses it; Enter on a ready pair
    applies it to this chat, saves it as the default and unlocks the card."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _console_app(
        {
            "chat_defaults": {"provider": "llama_cpp", "model": ""},
            "api_settings.llama_cpp": {"api_url": SAVED_LLAMA, "model": ""},
            "api_settings.openai": {"api_key": ""},
        },
        {"llama_cpp": ["model-a", "model-b"], "openai": ["gpt-5.1"]},
    )
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console, modal = await _blocking_card(harness, pilot)
        action = await _focused_action(
            harness, console, pilot, CONSOLE_SETUP_MODAL_ACTION_ID
        )
        assert str(action.label) == "Choose model"
        assert not model_switcher.has_working_new_chat_default(console)
        before = _config_bytes()

        await pilot.press("enter")
        switcher = await _switcher_ready(harness, pilot)
        assert not switcher._pick_only
        find = switcher.query_one("#console-popover-find", Input)
        # On the chat's provider, as "… N more" opens one: unfiltered, another
        # provider's row could come first, and Enter would make it the default.
        assert find.value == "llama.cpp "
        assert _scope(switcher) == model_switcher.SCOPE_THIS_CHAT_AND_NEW_CHATS
        row = switcher.highlighted_row()
        assert (row.kind, row.provider, row.model) == ("pair", "llama_cpp", "model-a")

        # A typed id on a provider with no key: never the new-chat default.
        find.value = "OpenAI typed-model"
        await pilot.pause()
        row = switcher.highlighted_row()
        assert (row.kind, row.provider, row.model) == ("typed", "openai", "typed-model")
        await pilot.press("enter")
        await pilot.pause()
        error = switcher.query_one("#console-popover-error", Static)
        assert error.display and str(error.render()).startswith("Unavailable: ")
        assert harness.screen is switcher
        assert _pair(console) == ("llama_cpp", None)
        assert _config_bytes() == before

        find.value = ""
        await pilot.pause()
        row = switcher.highlighted_row()
        assert (row.kind, row.provider, row.model) == ("pair", "llama_cpp", "model-a")
        await pilot.press("enter")
        await _until(pilot, lambda: harness.screen is console, "the switcher to close")
        await _drain(app)
        await _until(pilot, lambda: not modal.is_blocking, "the card to unlock")
        await _composer_ready(harness, console, pilot)

        assert _pair(console) == ("llama_cpp", "model-a")
        saved = _saved()
        assert saved["chat_defaults"]["provider"] == "llama_cpp"
        assert saved["chat_defaults"]["model"] == "model-a"
        # The saved endpoint was never part of this pick.
        assert saved["api_settings"]["llama_cpp"]["api_url"] == SAVED_LLAMA
        assert model_switcher.has_working_new_chat_default(console)
    assert app.console_default_durability_state.failure_phase is None


@pytest.mark.asyncio
@private_profile_test
async def test_save_as_model_default_on_the_cards_list_also_sets_the_new_chat_default(
    monkeypatch, request
) -> None:
    """AC#5, AC#6. Rewritten on purpose from
    ``test_save_as_model_default_on_the_cards_list_saves_the_models_defaults``,
    which pinned the press leaving ``chat_defaults`` with no model under a
    scope line that said "also the default for new chats": the next new chat
    was blocked on the card again.

    While no new-chat default works, every pick on the card's list sets it,
    whichever action commits the pick. Ctrl+N's action writes the model's own
    defaults as well, so Save as model default loses nothing by running as
    it: both are saved."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _console_app(
        {
            "chat_defaults": {"provider": "llama_cpp", "model": ""},
            "api_settings.llama_cpp": {"api_url": SAVED_LLAMA, "model": ""},
        },
        {"llama_cpp": ["model-a", "model-b"]},
    )
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console, modal = await _blocking_card(harness, pilot)
        await _focused_action(harness, console, pilot, CONSOLE_SETUP_MODAL_ACTION_ID)
        await pilot.press("enter")
        switcher = await _switcher_ready(harness, pilot)
        assert _scope(switcher) == model_switcher.SCOPE_THIS_CHAT_AND_NEW_CHATS
        row = switcher.highlighted_row()
        assert (row.kind, row.provider, row.model) == ("pair", "llama_cpp", "model-a")
        switcher.query_one("#console-popover-temperature", Input).value = "0.31"
        await pilot.pause()

        switcher.query_one("#console-popover-save-model-default", Button).press()
        await _until(pilot, lambda: harness.screen is console, "the switcher to close")
        await _drain(app)
        await _until(pilot, lambda: not modal.is_blocking, "the card to unlock")

        assert _pair(console) == ("llama_cpp", "model-a")
        saved = _saved()
        profile = saved["api_settings"]["llama_cpp"]["model_defaults"]["model-a"]
        assert profile["temperature"] == pytest.approx(0.31)
        # What the scope line said before the press.
        assert saved["chat_defaults"]["provider"] == "llama_cpp"
        assert saved["chat_defaults"]["model"] == "model-a"
        assert model_switcher.has_working_new_chat_default(console)
    assert app.console_default_durability_state.failure_phase is None


@pytest.mark.asyncio
@private_profile_test
async def test_card_pick_applies_to_this_chat_only_while_a_new_chat_default_works(
    request,
) -> None:
    """AC#5, AC#6: new chats already start on a pair that can send, and only
    this chat's own pair lacks a model. The card's switcher says "this chat
    only", and Enter writes no configuration."""
    app = _console_app(
        {
            "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
            "api_settings.llama_cpp": {"api_url": SAVED_LLAMA, "model": "model-a"},
            "api_settings.ollama": {"model": ""},
        },
        {"llama_cpp": ["model-a", "model-b"], "ollama": []},
    )
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.replace_session_settings(
            session_id,
            replace(
                store.session_settings(session_id),
                provider="ollama",
                model=None,
                source="user",
            ),
        )
        console._sync_console_transcript_guidance()
        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        await _until(pilot, lambda: modal.is_blocking, "the Get started card")
        action = await _focused_action(
            harness, console, pilot, CONSOLE_SETUP_MODAL_ACTION_ID
        )
        assert str(action.label) == "Choose model"
        assert model_switcher.has_working_new_chat_default(console)
        before = _config_bytes()

        await pilot.press("enter")
        switcher = await _switcher_ready(harness, pilot)
        assert _scope(switcher) == model_switcher.SCOPE_THIS_CHAT
        # Choose model opens on the chat's provider (Ollama lists no model
        # here); clearing Find lists every provider again.
        find = switcher.query_one("#console-popover-find", Input)
        assert find.value == "Ollama "
        row = switcher.highlighted_row()
        assert (row.kind, row.provider) == ("setup", "ollama")
        find.value = ""
        await pilot.pause()
        row = switcher.highlighted_row()
        assert (row.kind, row.provider, row.model) == ("pair", "llama_cpp", "model-a")
        await pilot.press("down", "enter")
        await _until(pilot, lambda: harness.screen is console, "the switcher to close")
        await _drain(app)
        await _until(pilot, lambda: not modal.is_blocking, "the card to unlock")

        assert _pair(console) == ("llama_cpp", "model-b")
        assert _config_bytes() == before


@pytest.mark.asyncio
@private_profile_test
async def test_card_choose_provider_opens_switch_model_and_esc_returns(
    request,
) -> None:
    """AC#4, AC#3: Choose provider opens Switch model; Esc returns to the
    card, which still blocks, with nothing written."""
    app = _console_app(
        {"chat_defaults": {"provider": "not-a-provider", "model": ""}},
        {"llama_cpp": ["model-a"]},
    )
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console, modal = await _blocking_card(harness, pilot)
        action = await _focused_action(
            harness, console, pilot, CONSOLE_SETUP_MODAL_ACTION_ID
        )
        assert str(action.label) == "Choose provider"
        before = _config_bytes()

        await pilot.press("enter")
        switcher = await _switcher_ready(harness, pilot)
        assert not switcher._pick_only
        # Choosing a provider is choosing among them: Find stays empty.
        assert switcher.query_one("#console-popover-find", Input).value == ""
        assert _scope(switcher) == model_switcher.SCOPE_THIS_CHAT_AND_NEW_CHATS
        await pilot.press("escape")
        await _until(pilot, lambda: harness.screen is console, "Esc to close it")
        await pilot.pause()

        assert modal.is_blocking and harness.focused is action
        assert _config_bytes() == before


@pytest.mark.asyncio
@private_profile_test
async def test_card_endpoint_recovery_still_opens_chat_settings(request) -> None:
    """AC#4 (R16): an endpoint recovery on the card is not a pair choice; it
    still opens Chat settings."""
    app = _console_app(
        {
            "chat_defaults": {"provider": "databricks", "model": "some-model"},
            "api_settings.databricks": {"api_key": "dapi-scratch-not-a-real-key-0000"},
        },
        {"llama_cpp": ["model-a"]},
    )
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console, _modal = await _blocking_card(harness, pilot)
        action = await _focused_action(
            harness, console, pilot, CONSOLE_SETUP_MODAL_ACTION_ID
        )
        assert str(action.label) == "Configure endpoint"

        await pilot.press("enter")
        await _until(
            pilot,
            lambda: isinstance(harness.screen, ConsoleSettingsModal),
            "Chat settings to open",
        )


def _recovery_screen(action: str | None, *, blocking: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        _console_setup_modal_blocking=lambda: blocking,
        _active_console_settings_readiness=lambda: (
            SimpleNamespace(provider="ollama"),
            SimpleNamespace(recovery_action=action),
        ),
        _provider_readiness_app_config=dict,
    )


@pytest.mark.parametrize(
    "action, query",
    [
        ("select_provider", ""),
        ("select_supported_provider", ""),
        # Choose model: on the chat's provider (finding 6, round 1).
        ("select_model", "Ollama "),
        ("configure_endpoint", None),
        ("save_endpoint", None),
        ("review_provider_settings", None),
        ("configure_credential", None),
        ("retry_connection", None),
        (None, None),
    ],
)
def test_only_the_cards_pair_choices_open_switch_model(
    monkeypatch, action, query
) -> None:
    """AC#4 (R16): Choose provider and Choose model go to Switch model; every
    other recovery, and every recovery off the card, keeps its own route.
    ``query`` is the Find text it opens with (None: it does not open)."""
    opened: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        model_switcher,
        "_open_switcher",
        lambda screen, **kwargs: opened.append((screen, kwargs)),
    )

    screen = _recovery_screen(action)
    assert model_switcher.open_card_recovery_switcher(screen) is (query is not None)
    assert opened == (
        [] if query is None else [(screen, {"query": query, "card": True})]
    )

    # The same recovery from the transcript or the prompts hook: no card.
    assert not model_switcher.open_card_recovery_switcher(
        _recovery_screen(action, blocking=False)
    )
    assert len(opened) == (0 if query is None else 1)


@pytest.mark.parametrize(
    "defaults, operability, working",
    [
        ({"provider": "llama_cpp", "model": "model-a"}, "ready_to_send", True),
        ({"provider": "llama_cpp", "model": "model-a"}, "not_ready", False),
        ({"provider": "llama_cpp", "model": ""}, "ready_to_send", False),
        ({"provider": "", "model": "model-a"}, "ready_to_send", False),
        ({}, "ready_to_send", False),
    ],
)
def test_a_working_new_chat_default_needs_a_ready_provider_and_model(
    defaults, operability, working
) -> None:
    """AC#5 (R14): "no working new-chat default" means no saved provider, no
    saved model, or a saved pair that is not ready to send."""
    asked: list[tuple[str, str]] = []

    def readiness(provider, model):
        asked.append((provider, model))
        return SimpleNamespace(operability=operability)

    screen = SimpleNamespace(
        _persisted_chat_defaults=lambda: defaults,
        _console_default_readiness=readiness,
    )
    assert model_switcher.has_working_new_chat_default(screen) is working
    assert asked == (
        [("llama_cpp", "model-a")]
        if len(defaults) == 2 and all(defaults.values())
        else []
    )


@pytest.mark.asyncio
@pytest.mark.loopback_network
@pytest.mark.local_server_probe
@private_profile_test
async def test_a_plain_launch_runs_the_real_discovery_and_offers_the_server(
    monkeypatch, request
) -> None:
    """Owner ruling (Phase 5): a plain launch probes the localhost defaults,
    so "Use detected <server>" can appear for a real user. Nothing injects
    ``app.console_local_server_discovery`` here: the card's worker runs the
    real ``discover_local_servers``, with its well-known candidates pointed at
    a stub loopback server (and no configured candidate), and the stub must
    see the request."""
    from tldw_chatbook.Chat import local_server_discovery

    with _stub_llama_server() as server:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        for name in (
            "DEFAULT_LLAMACPP_DISCOVERY_URL",
            "DEFAULT_LLAMACPP_BASE_URL",
            "DEFAULT_OLLAMA_DISCOVERY_URL",
        ):
            monkeypatch.setattr(local_server_discovery, name, url)
        monkeypatch.setattr(local_server_discovery, "LOCAL_DISCOVERY_PROVIDER_KEYS", ())
        app = _console_app(
            {"chat_defaults": {"provider": "llama_cpp", "model": ""}},
            {"llama_cpp": []},
        )
        assert getattr(app, "console_local_server_discovery", None) is None
        harness = _Harness(app)
        async with harness.run_test(size=(211, 44)) as pilot:
            console, _modal = await _blocking_card(harness, pilot)
            action = await _focused_action(
                harness, console, pilot, CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID
            )
            assert str(action.label) == (
                f"Use detected llama.cpp (127.0.0.1:{server.server_address[1]})"
            )
            assert "/v1/models" in server.paths
