"""Synthetic native Provider review qualification; no real provider calls."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import toml
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.Wizards.provider_step_harness import (
    WizardHost,
    current_step,
    key_input,
    pinned_error,
    provider_list,
    resumed_on_provider,
    static_text,
    wait_for_step,
    wait_until,
)
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_MODEL,
    STEP_PROVIDER,
    STEP_VOICE,
)

pytestmark = pytest.mark.bootstrap_profile

_ACCOUNT_CASES = [
    ("azure", "https://review.openai.azure.com", "AZURE_OPENAI_API_KEY"),
    ("databricks", "https://review.azuredatabricks.net", "DATABRICKS_TOKEN"),
    (
        "cloudflare",
        "https://api.cloudflare.com/client/v4/accounts/review/ai/v1",
        "CLOUDFLARE_API_TOKEN",
    ),
]


async def _choose(pilot, wizard, provider):
    await wait_for_step(pilot, wizard, STEP_PROVIDER)
    container, step = current_step(wizard)
    choices = provider_list(step)
    choices.focus()
    await pilot.pause()
    # Ready providers move to their own section; select the actual visible row.
    ids = {str(choices.get_option_at_index(i).id) for i in range(choices.option_count)}
    option = (
        f"ready-{provider}" if f"ready-{provider}" in ids else f"provider-{provider}"
    )
    choices.highlighted = choices.get_option_index(option)
    await pilot.press("enter")
    await pilot.pause()
    assert step.selected_provider_key == provider
    return container, step


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, endpoint, env_var", _ACCOUNT_CASES)
@pytest.mark.parametrize("source", ["environment", "stored"])
@private_profile_test
async def test_account_url_keeps_selected_auth_through_discovery_and_real_save(
    provider,
    endpoint,
    env_var,
    source,
    request,
    monkeypatch,
):
    from tldw_chatbook import config
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        DiscoveredModel,
        ModelDiscoveryResult,
    )

    env_secret = "synthetic-exported-account-key"
    stored_secret = "synthetic-stored-account-key"
    monkeypatch.setenv(env_var, env_secret)
    settings = {"api_key": stored_secret} if source == "stored" else {}
    config.load_cli_config_and_ensure_existence()
    config.replace_cli_config_serialized(
        toml.dumps({"api_settings": {provider: settings}}), create_backup=False
    )
    wizard = resumed_on_provider(
        lambda c: c["api_settings"].__setitem__(provider, dict(settings))
    )
    service = LocalLLMProviderCatalogService(environ={env_var: env_secret})
    auth = []

    async def discover_models(**kwargs):
        # Exercise the production request-auth boundary against the exact staged
        # endpoint/source; the external provider transport alone is absent.
        selected = kwargs["provider"]
        staged = kwargs["staged_settings"]
        auth.append(
            (
                staged["api_settings"][selected]["credential_source"],
                service._resolve_api_key(
                    provider=selected,
                    provider_key=selected,
                    saved_settings=wizard.app_instance.app_config,
                    staged_settings=staged,
                ),
            )
        )
        return ModelDiscoveryResult(
            provider=selected,
            provider_list_key=selected,
            endpoint_fingerprint=None,
            status="success",
            models=(
                DiscoveredModel(
                    provider=selected,
                    provider_list_key=selected,
                    model_id="synthetic-account-model",
                    display_name="synthetic-account-model",
                    source="runtime_discovery",
                    endpoint_fingerprint="",
                    discovered_at="2026-10-31T00:00:00Z",
                ),
            ),
        )

    wizard.app_instance.llm_provider_catalog_scope_service = SimpleNamespace(
        discover_models=discover_models
    )
    app = WizardHost(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        container, step = await _choose(pilot, wizard, provider)
        assert step._credential_draft().source == source
        assert not step._current_provider_readiness().ready  # URL still missing
        step.query_one("#setup-provider-endpoint", Input).value = endpoint
        await pilot.pause()
        assert step._current_provider_readiness().ready
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        assert container.staged_provider_draft.credential.source == source
        await wait_until(pilot, lambda: bool(auth))
        assert auth[-1] == (
            source,
            env_secret if source == "environment" else stored_secret,
        )
        _, model = current_step(wizard)
        model.query_one("#setup-model-custom", Input).value = "synthetic-account-model"
        await pilot.pause()
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_VOICE)
        assert not pinned_error(wizard)

    saved = config.load_cli_config_and_ensure_existence(force_reload=True)
    table = saved["api_settings"][provider]
    assert table["credential_source"] == source
    if source == "environment":
        assert table["api_key_env_var"] == env_var
        assert not table.get("api_key")
    else:
        assert table["api_key"] == stored_secret
    assert env_secret not in Path(os.environ["TLDW_CONFIG_PATH"]).read_text(
        encoding="utf-8"
    )
    ready = get_provider_readiness(provider, saved)
    assert ready.ready and ready.api_key == (
        env_secret if source == "environment" else stored_secret
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:5001",
        "http://localhost:5001/v1",
        "http://localhost:5001/prefix",
        "http://localhost:5001/api/v1/generate",
    ],
)
@private_profile_test
async def test_kobold_wizard_save_and_dispatch_use_native_url_with_prompt(
    endpoint, request, monkeypatch
):
    from tldw_chatbook import config
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local

    config.load_cli_config_and_ensure_existence()
    requests = []

    class Session:
        def post(self, url, **kwargs):
            requests.append((url, kwargs["json"]))
            response = httpx.Response(
                200,
                json={"results": [{"text": "synthetic-native-reply"}]},
                request=httpx.Request("POST", url),
            )
            return response

    monkeypatch.setattr(local, "create_pooled_default_session", lambda **kw: Session())
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        container, step = await _choose(pilot, wizard, "koboldcpp")
        # The shipped native prefill remains usable without edits.
        assert step.query_one("#setup-provider-endpoint", Input).value.endswith(
            "/api/v1/generate"
        )
        step.query_one("#setup-provider-endpoint", Input).value = endpoint
        await pilot.pause()
        if not endpoint.endswith("/api/v1/generate"):
            assert "api/v1/generate" in static_text(
                step, "#setup-provider-endpoint-status"
            )
            assert "Custom" in static_text(step, "#setup-provider-endpoint-status")
            assert step.query_one("#setup-provider-test", Button).disabled
            config_path = Path(os.environ["TLDW_CONFIG_PATH"])
            await wait_until(
                pilot,
                lambda: toml.loads(config_path.read_text(encoding="utf-8"))
                .get("first_run", {})
                .get("setup_started")
                is True,
            )
            before = config_path.read_bytes()
            ok, refusal = await step.commit()
            assert not ok and "Custom" in refusal
            assert container.staged_provider_draft is None
            assert Path(os.environ["TLDW_CONFIG_PATH"]).read_bytes() == before
            assert requests == []
            return
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _, model = current_step(wizard)
        model.query_one("#setup-model-custom", Input).value = "synthetic-native-model"
        await pilot.pause()
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_VOICE)
    saved = config.load_cli_config_and_ensure_existence(force_reload=True)
    assert saved["api_settings"]["koboldcpp"]["api_url"] == endpoint
    reply = chat_api_call(
        "koboldcpp",
        [{"role": "user", "content": "synthetic native prompt"}],
        model="synthetic-native-model",
        streaming=False,
    )
    assert reply["choices"][0]["message"]["content"] == "synthetic-native-reply"
    assert len(requests) == 1
    assert requests[0][0] == endpoint
    assert requests[0][1]["prompt"] == "synthetic native prompt"
    assert "messages" not in requests[0][1]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_source_aware_ready_list_keeps_key_focus_and_geometry(size, monkeypatch):
    monkeypatch.setenv("REVIEW_PRIVATE_KEY", "synthetic-custom-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-conventional-key")

    def configure(c):
        c["api_settings"]["openai"] = {"credential_source": "none"}
        c["api_settings"]["groq"] = {
            "credential_source": "environment",
            "api_key_env_var": "REVIEW_PRIVATE_KEY",
        }

    wizard = resumed_on_provider(configure)
    app = WizardHost(wizard)
    async with app.run_test(size=size) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _, step = current_step(wizard)
        await wait_until(pilot, lambda: "groq" in step._ready_badges)
        assert "openai" not in step._ready_badges
        assert step._ready_badges["groq"] == "key in REVIEW_PRIVATE_KEY"
        # Source none remains unready until the user explicitly types a key.
        await _choose(pilot, wizard, "openai")
        field = key_input(step)
        assert app.focused is field
        assert field.region.height > 0 and app.screen.region.contains_region(
            field.region
        )
        field.value = "synthetic-explicit-replacement-key"
        await pilot.pause()
        assert step._current_provider_readiness().ready
        step._on_clear()
        await pilot.pause()
        assert not step._current_provider_readiness().ready


@pytest.mark.asyncio
@pytest.mark.local_server_probe
@pytest.mark.loopback_network
async def test_fresh_scan_finds_native_ipv6_only_localhost_listener(monkeypatch):
    from tldw_chatbook.Chat import local_server_discovery as discovery

    listener = None
    try:
        listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        listener.bind(("::1", 0))
    except OSError as error:
        if listener is not None:
            listener.close()
        pytest.skip(f"Native IPv6 loopback bind refused: {error.errno}")
    listener.listen()
    listener.setblocking(False)
    port = listener.getsockname()[1]
    asked = []

    async def respond(reader, writer):
        request = await reader.readuntil(b"\r\n\r\n")
        asked.append(request.split(b"\r\n", 1)[0])
        body = json.dumps({"data": [{"id": "synthetic-ipv6-model"}]}).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(respond, sock=listener)
    base = f"http://127.0.0.1:{port}"
    monkeypatch.setattr(discovery, "DEFAULT_LLAMACPP_DISCOVERY_URL", base)
    monkeypatch.setattr(discovery, "DEFAULT_LLAMACPP_BASE_URL", base)
    monkeypatch.setattr(discovery, "DEFAULT_OLLAMA_DISCOVERY_URL", base)
    monkeypatch.setattr(discovery, "ENGINE_DISCOVERY_DEFAULTS", ())
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=2.5) as client:
            with pytest.raises(httpx.ConnectError):
                await client.get(base + "/v1/models")
            results = await discovery.discover_local_servers({}, http_client=client)
        assert [r.base_url for r in results] == [f"http://localhost:{port}"]
        assert results[0].model_ids == ("synthetic-ipv6-model",)
        assert asked == [b"GET /v1/models HTTP/1.1"]
    finally:
        server.close()
        await server.wait_closed()
