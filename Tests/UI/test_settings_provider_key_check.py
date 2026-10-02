"""Settings 't' checks a cloud key with one model listing (TASK-33005.4).

ADR-012 amendment of 2026-09-26 (owner decision D2): 't' on a cloud provider
runs one explicit, non-generating, authenticated model listing and reports
what it proved. These tests drive the real Settings screen through the real
``LocalLLMProviderCatalogService.discover_models`` and discovery client; only
the transport is an ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path

import httpx
import pytest
from loguru import logger
from textual.widgets import Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _active_destination_screen, _static_text
from Tests.UI.test_settings_configuration_hub import (
    StyledSettingsDestinationHarness,
    _open_settings_category,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_settings_readiness,
    readiness_words,
)
from tldw_chatbook.Chat.provider_test_evidence import provider_connection_evidence
from tldw_chatbook.LLM_Provider_Catalog import (
    openai_compatible_model_discovery as discovery_module,
)
from tldw_chatbook.UI.Screens.settings_screen import (
    PROVIDER_TEST_GUIDANCE,
    SettingsCategoryId,
)

PROVIDERS_MODELS = "#settings-category-providers-models"
LISTING = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}


class _Provider:
    """The provider's models route behind a MockTransport; records requests."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = (
            self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        )
        if isinstance(response, asyncio.Event):
            await response.wait()
            return httpx.Response(401)
        if isinstance(response, type) and issubclass(response, Exception):
            raise response("secret transport detail", request=request)
        return response


def _route_discovery_to(monkeypatch, provider: _Provider) -> None:
    """Give the real discovery client a MockTransport (nothing else changes)."""
    monkeypatch.setattr(
        discovery_module,
        "build_httpx_async_client",
        lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(provider), **kwargs
        ),
    )


def _config_digest() -> str:
    return hashlib.sha256(Path(os.environ["TLDW_CONFIG_PATH"]).read_bytes()).hexdigest()


def _result(screen) -> str:
    return _static_text(screen.query_one("#settings-provider-test-result", Static))


def _rows(screen) -> dict[str, str]:
    rows = {}
    for line in _result(screen).splitlines():
        label, _gap, text = line.partition("  ")
        rows[label] = text.strip()
    return rows


async def _test_and_settle(screen, pilot) -> dict[str, str]:
    screen.action_settings_test_category()
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if "checking the model listing" not in _result(screen):
            break
    await screen.workers.wait_for_complete()
    await pilot.pause()
    return _rows(screen)


async def _until(pilot, condition, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await pilot.pause(0.01)


def _cloud_app(provider: str, settings: dict, model: str = "gpt-4o"):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": model}
    app.app_config["api_settings"] = {provider: dict(settings)}
    app.providers_models = {
        "OpenAI": ["gpt-4o"],
        "Anthropic": ["claude-sonnet-5"],
        "OpenRouter": ["openai/gpt-4o"],
        "Google": ["gemini-2.5-flash"],
    }
    return app


def _console_word(app, host, provider: str, model: str) -> str:
    return readiness_words(
        build_console_settings_readiness(
            ConsoleSessionSettings(provider=provider, model=model),
            app_config=app.app_config,
            connection_evidence=provider_connection_evidence(host),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "readiness", "toast"),
    [
        (
            httpx.Response(200, json=LISTING),
            "Ready · verified",
            "key accepted (2 models listed) · generation not tested",
        ),
        (
            httpx.Response(401, text="secret body sk-saved-test-key"),
            "Not ready · key rejected",
            "OpenAI rejected the API key (unauthorized); enter a valid key",
        ),
        (
            httpx.Response(403),
            "Not ready · key rejected",
            "OpenAI rejected the API key (forbidden)",
        ),
        (
            httpx.ReadTimeout,
            "Not ready · timed out",
            "model listing failed (timeout) — check your network or the endpoint URL",
        ),
    ],
    ids=["200", "401", "403", "timeout"],
)
@private_profile_test
async def test_settings_t_checks_a_saved_cloud_key_with_one_listing(
    response, readiness, toast, request, monkeypatch
):
    """AC#1/#2/#3/#4/#5/#12/#14/#16: one GET to the endpoint a send uses,
    carrying the saved key; the outcome's word and phrase; nothing saved,
    cached or logged; the Console reads the same word for the same key."""
    provider = _Provider(response)
    _route_discovery_to(monkeypatch, provider)
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(str(message)), level="DEBUG")
    app = _cloud_app("openai", {"api_key": "sk-saved-test-key"})
    host = StyledSettingsDestinationHarness(app, "settings")
    try:
        async with host.run_test(size=(211, 44)) as pilot:
            await _open_settings_category(pilot, PROVIDERS_MODELS)
            screen = _active_destination_screen(host)
            toasts: list[str] = []
            host.notify = lambda message, **_kwargs: toasts.append(str(message))
            before = _config_digest()
            catalog = {key: list(value) for key, value in app.providers_models.items()}

            rows = await _test_and_settle(screen, pilot)

            assert [(r.method, str(r.url)) for r in provider.requests] == [
                ("GET", "https://api.openai.com/v1/models")
            ]
            assert provider.requests[0].headers["authorization"] == (
                "Bearer sk-saved-test-key"
            )
            assert rows["Readiness"].startswith(readiness), rows
            assert toast in toasts[-1]
            assert toasts[-1].startswith(rows["Readiness"])
            if response.__class__ is httpx.Response and response.status_code == 200:
                assert rows["Key"] == "saved in config · key accepted (2 models listed)"
                assert rows["Generation"] == "not tested"
            # AC#12: the shared owner gives the Console the same word.
            assert _console_word(app, host, "openai", "gpt-4o") == rows["Readiness"]
            # AC#5: no key, header or body reaches the UI or the logs.
            shown = "\n".join([_result(screen), *toasts, *logged])
            assert "sk-saved-test-key" not in shown
            assert "secret" not in shown
            # AC#14 / AC#16: nothing saved, cached or consented.
            assert _config_digest() == before
            assert app.providers_models == catalog
            assert app.local_llm_provider_catalog_service.discovery_cache.list() == ()
    finally:
        logger.remove(sink)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["env", "typed"])
@private_profile_test
async def test_settings_t_sends_the_env_or_typed_key_and_saves_nothing(
    source, request, monkeypatch
):
    """AC#2/#10: an env-var key and a typed, unsaved key are each the key the
    listing carries; a typed endpoint is where it goes; nothing is saved."""
    provider = _Provider(httpx.Response(200, json=LISTING))
    _route_discovery_to(monkeypatch, provider)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test-key")
    app = _cloud_app("openai", {"api_key_env_var": "OPENAI_API_KEY"})
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, PROVIDERS_MODELS)
        screen = _active_destination_screen(host)
        before = _config_digest()
        if source == "typed":
            screen.query_one(
                "#settings-provider-api-key", Input
            ).value = "sk-typed-test-key"
            screen.query_one(
                "#settings-provider-endpoint-value", Input
            ).value = "https://gateway.example.test/v1"
            await pilot.pause()

        rows = await _test_and_settle(screen, pilot)

        expected = {
            "env": ("https://api.openai.com/v1/models", "Bearer sk-env-test-key"),
            "typed": (
                "https://gateway.example.test/v1/models",
                "Bearer sk-typed-test-key",
            ),
        }[source]
        assert [
            (str(r.url), r.headers["authorization"]) for r in provider.requests
        ] == [expected]
        assert rows["Readiness"].startswith("Ready · verified"), rows
        assert rows["Key"].endswith("key accepted (2 models listed)")
        assert _config_digest() == before
        assert screen._category_has_unsaved_changes(
            SettingsCategoryId.PROVIDERS_MODELS
        ) is (source == "typed")


@pytest.mark.asyncio
@private_profile_test
async def test_settings_t_checks_anthropic_on_its_shipped_endpoint(
    request, monkeypatch
):
    """AC#9: Anthropic's built-in endpoint qualifies for a key check (the
    Settings field's placeholder, https://api.anthropic.com, would not) and
    the key travels in Anthropic's own header."""
    assert discovery_module.supports_openai_compatible_model_discovery(
        "anthropic", "https://api.anthropic.com/v1"
    )
    assert not discovery_module.supports_openai_compatible_model_discovery(
        "anthropic", "https://api.anthropic.com"
    )
    provider = _Provider(
        httpx.Response(
            200, json={"data": [{"id": "claude-sonnet-5"}], "has_more": False}
        )
    )
    _route_discovery_to(monkeypatch, provider)
    app = _cloud_app(
        "anthropic", {"api_key": "sk-ant-saved-test-key"}, model="claude-sonnet-5"
    )
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, PROVIDERS_MODELS)
        screen = _active_destination_screen(host)

        rows = await _test_and_settle(screen, pilot)

        (sent,) = provider.requests
        assert (sent.url.host, sent.url.path) == ("api.anthropic.com", "/v1/models")
        assert sent.headers["x-api-key"] == "sk-ant-saved-test-key"
        assert "authorization" not in sent.headers
        assert rows["Readiness"].startswith("Ready · verified"), rows
        assert rows["Model"] == "claude-sonnet-5 · listed by the server"


@pytest.mark.asyncio
@private_profile_test
async def test_settings_t_never_calls_a_public_listing_a_key_check(
    request, monkeypatch
):
    """AC#6: OpenRouter's catalog answers without a key (ADR-020), so its
    listing reads 'models listed; key not checked' and never 'accepted'."""
    provider = _Provider(httpx.Response(200, json={"data": [{"id": "openai/gpt-4o"}]}))
    _route_discovery_to(monkeypatch, provider)
    app = _cloud_app("openrouter", {"api_key": "sk-or-saved-test-key"}, "openai/gpt-4o")
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, PROVIDERS_MODELS)
        screen = _active_destination_screen(host)
        toasts: list[str] = []
        host.notify = lambda message, **_kwargs: toasts.append(str(message))

        rows = await _test_and_settle(screen, pilot)

        assert len(provider.requests) == 1
        assert rows["Readiness"] == "Ready · not tested"
        assert rows["Key"] == "saved in config · models listed; key not checked"
        assert "1 model listed; key not checked" in toasts[-1]
        assert "accepted" not in _result(screen) + toasts[-1]
        assert _console_word(app, host, "openrouter", "openai/gpt-4o") == (
            "Ready · not tested"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("typed", "env"),
    [("<API_KEY_HERE>", None), ("   ", None), (" your-api-key ", "sk-env-test-key")],
    ids=["placeholder", "whitespace", "padded-placeholder-over-env"],
)
@private_profile_test
async def test_settings_t_sends_nothing_without_a_usable_key(
    typed, env, request, monkeypatch
):
    """AC#7: TASK-32806.1's rule -- a missing, placeholder or blank key is no
    key, and no request is sent, even when an env var would fill in."""
    provider = _Provider(httpx.Response(200, json=LISTING))
    _route_discovery_to(monkeypatch, provider)
    if env is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", env)
    app = _cloud_app("openai", {"api_key_env_var": "OPENAI_API_KEY"})
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, PROVIDERS_MODELS)
        screen = _active_destination_screen(host)
        screen.query_one("#settings-provider-api-key", Input).value = typed
        await pilot.pause()
        toasts: list[str] = []
        host.notify = lambda message, **_kwargs: toasts.append(str(message))

        rows = await _test_and_settle(screen, pilot)

        assert provider.requests == []
        assert "missing" in (rows.get("Key check") or rows["Key"]), rows
        assert "accepted" not in _result(screen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "settings", "catalog", "note"),
    [
        (
            "google",
            {"api_key": "AIza-saved-test-key"},
            None,
            "No non-billable key check is available for Google",
        ),
        (
            "openai",
            {"api_key": "sk-saved-test-key"},
            {"Google": ["gemini-2.5-flash"]},
            "no model list is configured for OpenAI in [providers]",
        ),
        # AC#10 (fix round 1): a send would not go where the listing would.
        (
            "huggingface",
            {
                "api_key": "hf_saved_test_key",
                "api_base_url": "https://hf-gateway.example.test/v1",
            },
            {"HuggingFace": ["gemini-2.5-flash"]},
            "No non-billable key check is available for",
        ),
        (
            "openai",
            {
                "api_key": "sk-saved-test-key",
                "base_url": "https://gateway.example.test/v1",
            },
            None,
            "No non-billable key check is available for OpenAI",
        ),
    ],
    ids=[
        "no-supported-listing",
        "no-listing-configured",
        "send-ignores-huggingface-endpoint",
        "send-ignores-endpoint-alias",
    ],
)
@private_profile_test
async def test_settings_t_says_when_no_key_check_exists_and_sends_nothing(
    provider_name, settings, catalog, note, request, monkeypatch
):
    """AC#8 and the [providers] ruling: no request, no evidence, and the
    result says why -- never 'accepted'."""
    provider = _Provider(httpx.Response(200, json=LISTING))
    _route_discovery_to(monkeypatch, provider)
    app = _cloud_app(provider_name, settings, model="gemini-2.5-flash")
    if catalog is not None:
        app.providers_models = catalog
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, PROVIDERS_MODELS)
        screen = _active_destination_screen(host)
        toasts: list[str] = []
        host.notify = lambda message, **_kwargs: toasts.append(str(message))

        rows = await _test_and_settle(screen, pilot)

        assert provider.requests == []
        assert note in rows["Key check"]
        assert note in toasts[-1]
        assert rows["Readiness"] == "Ready · not tested"
        identity = screen._provider_current_draft_identity()
        assert provider_connection_evidence(host).evidence_for(identity) is None


@pytest.mark.asyncio
@private_profile_test
async def test_a_key_check_the_runtime_policy_refuses_records_nothing(
    request, monkeypatch
):
    """Fix round 1: in server mode the real runtime policy denies the local
    listing before any request (wrong_source). 't' says the key was not
    checked and publishes nothing -- never a 'connection error' the Console
    would treat as a send blocker (TASK-30011 AC#2)."""
    from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState

    provider = _Provider(httpx.Response(200, json=LISTING))
    _route_discovery_to(monkeypatch, provider)
    server_mode = RuntimeSourceState(
        active_source="server", server_configured=True, active_server_id="srv"
    )
    app = _cloud_app("openai", {"api_key": "sk-saved-test-key"})
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, PROVIDERS_MODELS)
        screen = _active_destination_screen(host)
        monkeypatch.setattr(
            app.llm_provider_catalog_scope_service,
            "policy_enforcer",
            ServicePolicyEnforcer(state_provider=lambda: server_mode),
        )
        toasts: list[str] = []
        host.notify = lambda message, **_kwargs: toasts.append(str(message))

        rows = await _test_and_settle(screen, pilot)

        assert provider.requests == []
        assert rows["Key check"].startswith("Key not checked"), rows
        assert toasts[-1] == rows["Key check"]
        assert "connection error" not in _result(screen)
        identity = screen._provider_current_draft_identity()
        assert provider_connection_evidence(host).evidence_for(identity) is None
        assert _console_word(app, host, "openai", "gpt-4o") == "Ready · not tested"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_key",
    ["openai", "cohere", "google", "groq", "openrouter", "deepseek", "huggingface"],
)
@private_profile_test
async def test_key_check_only_lists_where_a_send_would_go(
    provider_key, request, monkeypatch
):
    """AC#10 (fix round 1), proven on the real handlers and the real config
    loader: each is called as the Console gateway calls it (no pinned
    ``api_base_url``), so the saved table alone picks the URL. These six read
    ``api_base_url`` and no other spelling; Hugging Face reads neither (its
    legacy ``[API]`` table, TASK-2117). Settings refuses the key check for
    exactly the cases a send would skip. Gateway-pinned providers (Anthropic,
    Mistral, Moonshot, Z.AI, engine presets) send to the same resolver 't'
    lists through, so they need no row here."""
    import contextlib

    import requests
    import toml

    from tldw_chatbook import config as config_mod
    from tldw_chatbook.LLM_Calls import LLM_API_Calls, deepseek, groq, openrouter
    from tldw_chatbook.UI.Screens import settings_screen

    sent: list[str] = []

    class _Captured(Exception):
        pass

    def capture(url) -> None:
        sent.append(str(url))
        raise _Captured

    monkeypatch.setattr(
        requests.Session, "post", lambda _self, url, *_a, **_k: capture(url)
    )
    for module in (deepseek, groq, openrouter):
        monkeypatch.setattr(
            module,
            "hosted_chat_request",
            lambda *, config, **_k: capture(config.base_url),
        )
    handler = {
        "openai": LLM_API_Calls.chat_with_openai,
        "cohere": LLM_API_Calls.chat_with_cohere,
        "google": LLM_API_Calls.chat_with_google,
        "huggingface": LLM_API_Calls.chat_with_huggingface,
        "groq": groq.chat_with_groq,
        "openrouter": openrouter.chat_with_openrouter,
        "deepseek": deepseek.chat_with_deepseek,
    }[provider_key]

    def send_with(table: dict) -> str:
        document = toml.loads(config_mod.CONFIG_TOML_CONTENT)
        document["api_settings"][provider_key] = {"api_key": "sk-fake-test", **table}
        Path(os.environ["TLDW_CONFIG_PATH"]).write_text(toml.dumps(document))
        config_mod.load_settings(force_reload=True)
        config_mod.get_runtime_config_snapshot(force_reload=True)
        sent.clear()
        with contextlib.suppress(Exception):
            handler(
                [{"role": "user", "content": "hi"}],
                model="m",
                api_key="sk-fake-test",
                streaming=False,
            )
        assert len(sent) == 1, sent
        return sent[0]

    honours = provider_key in settings_screen._SEND_READS_ONLY_API_BASE_URL
    assert honours is (provider_key != "huggingface")
    saved = send_with({"api_base_url": "https://saved.example.test/v1"})
    assert saved.startswith("https://saved.example.test/v1") is honours, saved
    alias = send_with({"base_url": "https://alias.example.test/v1"})
    assert "alias.example.test" not in alias, alias


@pytest.mark.asyncio
@private_profile_test
async def test_a_key_check_outrun_by_an_edit_or_a_rerun_shows_only_the_latest(
    request, monkeypatch
):
    """AC#13: a reply for a draft that changed is discarded, and a repeated
    check shows only the latest result."""
    held = asyncio.Event()
    provider = _Provider(held, httpx.Response(200, json=LISTING))
    _route_discovery_to(monkeypatch, provider)
    app = _cloud_app("openai", {"api_key": "sk-saved-test-key"})
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, PROVIDERS_MODELS)
        screen = _active_destination_screen(host)
        tested = screen._provider_current_draft_identity()

        screen.action_settings_test_category()
        await _until(pilot, lambda: provider.requests)
        screen.query_one(
            "#settings-provider-api-key", Input
        ).value = "sk-other-test-key"
        await pilot.pause()
        held.set()  # The stale draft's reply (a 401) arrives now.
        await screen.workers.wait_for_complete()
        await pilot.pause()

        owner = provider_connection_evidence(host)
        assert owner.evidence_for(tested) is None
        assert "key rejected" not in _result(screen)

        # A rerun while one is in flight: only the second answer shows.
        held.clear()
        provider.responses = [held, httpx.Response(200, json=LISTING)]
        screen.action_settings_test_category()
        await _until(pilot, lambda: len(provider.requests) == 2)
        rows = await _test_and_settle(screen, pilot)
        held.set()
        await pilot.pause()

        assert rows["Readiness"].startswith("Ready · verified"), rows
        assert _rows(screen) == rows


@pytest.mark.asyncio
@private_profile_test
async def test_a_typed_key_check_reaches_the_console_after_save(request, monkeypatch):
    """AC#12: a typed key's check carries to the saved connection on Save, so
    the Console reads 'Ready · verified HH:MM' for that key."""
    provider = _Provider(httpx.Response(200, json=LISTING))
    _route_discovery_to(monkeypatch, provider)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _cloud_app("openai", {})
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, PROVIDERS_MODELS)
        screen = _active_destination_screen(host)
        screen.query_one(
            "#settings-provider-api-key", Input
        ).value = "sk-typed-test-key"
        await pilot.pause()
        rows = await _test_and_settle(screen, pilot)
        assert rows["Readiness"].startswith("Ready · verified"), rows

        screen.action_settings_save_category()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and screen._category_has_unsaved_changes(
            SettingsCategoryId.PROVIDERS_MODELS
        ):
            await pilot.pause(0.05)
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert app.app_config["api_settings"]["openai"]["api_key"] == (
            "sk-typed-test-key"
        )
        assert _console_word(app, host, "openai", "gpt-4o") == rows["Readiness"]
        assert len(provider.requests) == 1  # Save itself listed nothing.


def _count_cloud_requests(monkeypatch) -> list[str]:
    """Count every request to a non-loopback host; nothing leaves the test."""
    cloud: list[str] = []

    def refuse(request: httpx.Request) -> None:
        if request.url.host not in {"127.0.0.1", "localhost", "::1"}:
            cloud.append(request.url.host)
        raise httpx.ConnectError("blocked in test", request=request)

    async def async_send(self, request, *args, **kwargs):
        refuse(request)

    def sync_send(self, request, *args, **kwargs):
        refuse(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    monkeypatch.setattr(httpx.Client, "send", sync_send)
    return cloud


@pytest.mark.asyncio
@private_profile_test
async def test_no_cloud_request_while_opening_editing_and_saving_settings(
    request, monkeypatch
):
    """AC#15: only an explicit 't' contacts a cloud provider -- not opening
    Settings, switching categories, typing or saving."""
    cloud = _count_cloud_requests(monkeypatch)
    app = _cloud_app("openai", {"api_key": "sk-saved-test-key"})
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, PROVIDERS_MODELS)
        await _open_settings_category(pilot, "#settings-category-storage")
        await _open_settings_category(pilot, PROVIDERS_MODELS)
        screen = _active_destination_screen(host)
        screen.query_one("#settings-provider-api-key", Input).value = "sk-typed-key"
        await pilot.pause()
        screen.action_settings_save_category()
        await _until(
            pilot,
            lambda: (
                not screen._category_has_unsaved_changes(
                    SettingsCategoryId.PROVIDERS_MODELS
                )
            ),
        )
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert app.app_config["api_settings"]["openai"]["api_key"] == "sk-typed-key"
        assert cloud == []
        # Positive control: the counter does see the one explicit 't'.
        await _test_and_settle(screen, pilot)
        assert cloud == ["api.openai.com"]


@pytest.mark.asyncio
@private_profile_test
async def test_no_cloud_request_while_opening_console_and_the_switcher(
    request, monkeypatch
):
    """AC#15: opening the Console and Switch model reads readiness from
    config and evidence only; no cloud provider is contacted."""
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _open_provider_popover,
        _persisted_console_app,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector

    cloud = _count_cloud_requests(monkeypatch)
    app = _persisted_console_app()
    app.app_config["api_settings"]["openai"] = {"api_key": "sk-saved-test-key"}
    app.app_config["api_settings"]["anthropic"] = {"api_key": "sk-ant-saved-key"}
    app.providers_models.update(
        {"OpenAI": ["gpt-4o"], "Anthropic": ["claude-sonnet-5"]}
    )
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        modal = await _open_provider_popover(console, harness, pilot)

        assert modal.is_mounted
        assert cloud == []


@pytest.mark.asyncio
@private_profile_test
async def test_test_guidance_says_what_t_checks_and_never_claims_generation(
    request,
):
    """AC#17: the Test button's tooltip, its visible guidance and the F1
    notes say one thing -- what 't' checks for cloud and local providers --
    and the footer verb stays 'test provider'; none claims generation."""
    text = PROVIDER_TEST_GUIDANCE.casefold()
    assert "cloud" in text and "api key" in text
    assert "local server" in text
    assert "without generating" in text
    assert "generation succeeded" not in text and "verified" not in text

    app = _cloud_app("openai", {"api_key": "sk-saved-test-key"})
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, PROVIDERS_MODELS)
        screen = _active_destination_screen(host)
        button = screen.query_one("#settings-test-provider")
        guidance = screen.query_one("#settings-test-provider-guidance", Static)

        assert str(button.tooltip) == PROVIDER_TEST_GUIDANCE
        assert _static_text(guidance) == PROVIDER_TEST_GUIDANCE
        notes = screen._category_help_notes(SettingsCategoryId.PROVIDERS_MODELS)
        assert f"Test provider (t): {PROVIDER_TEST_GUIDANCE}" in notes
        assert ("t", "test provider") in screen._category_footer_shortcuts(
            SettingsCategoryId.PROVIDERS_MODELS
        )
