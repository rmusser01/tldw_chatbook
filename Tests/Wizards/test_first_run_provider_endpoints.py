"""TASK-34100.6 slice C: endpoint fields on the Provider step.

Review findings provider-05 (Azure OpenAI, Databricks and Cloudflare had no
field for the URL they need and refused Next with 'API key required'),
new-provider-02 (KoboldCpp's shipped endpoint was rejected and any accepted
one was saved in a form ``chat_with_kobold`` can't use), provider-20 (Z.ai
previewed and saved ``…/paas/v4/v1/chat/completions``), provider-15 (endpoint
errors named the wrong cause) and TASK-34332 (keys typed into Endpoint right
after picking a local provider went to the provider list).
"""

from __future__ import annotations

import pytest
from textual import events
from textual.widgets import Button, Input, Label

from Tests.private_profile import private_profile_test
from Tests.Wizards.provider_step_harness import (
    SIZE,
    WizardHost,
    current_step,
    key_input,
    pick_provider,
    pinned_error,
    provider_list,
    resumed_on_provider,
    static_text,
    wait_for_step,
)
from tldw_chatbook.Chat.provider_endpoint_contract import resolve_provider_endpoint
from tldw_chatbook.Chat.provider_setup_persistence import (
    ProviderSetupDraft,
    build_provider_setup_mutation,
)
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_MODEL,
    STEP_PROVIDER,
)

pytestmark = pytest.mark.bootstrap_profile

#: provider -> (label, an example the field shows, a URL a user enters).
_BASE_URL_CASES = {
    "azure": (
        "Azure resource URL",
        "https://my-resource.openai.azure.com",
        "https://contoso-ai.openai.azure.com",
    ),
    "databricks": (
        "Databricks workspace URL",
        "azuredatabricks.net",
        "https://adb-1111111111111111.1.azuredatabricks.net",
    ),
    "cloudflare": (
        "Cloudflare account URL",
        "/accounts/<account-id>/ai/v1",
        "https://api.cloudflare.com/client/v4/accounts/0123456789abcdef/ai/v1",
    ),
}


async def _on_provider(pilot, wizard, provider_key):
    await wait_for_step(pilot, wizard, STEP_PROVIDER)
    _container, step = current_step(wizard)
    await pick_provider(pilot, step, provider_key)
    return step


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_key", sorted(_BASE_URL_CASES))
async def test_a_per_account_url_field_shows_above_the_key(provider_key, monkeypatch):
    """provider-05: the field the provider needs, labelled per provider."""
    from tldw_chatbook.Chat.provider_setup_facts import provider_setup_facts

    monkeypatch.delenv(provider_setup_facts(provider_key).env_var, raising=False)
    label, example, _entered = _BASE_URL_CASES[provider_key]
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard, provider_key)
        connection = step.query_one("#setup-provider-connection")
        assert connection.display
        assert str(step.query_one("#setup-provider-endpoint-label", Label).render()) == (
            label
        )
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        assert example in endpoint.placeholder
        assert endpoint.region.y < key_input(step).region.y
        # A cloud account URL is no local server: no detect, no dead Test.
        assert not step.query_one("#setup-provider-detect", Button).display
        assert not step.query_one("#setup-provider-test", Button).display
        # Nothing the user has not touched is an error yet.
        assert static_text(step, "#setup-provider-endpoint-status") == ""


@pytest.mark.asyncio
async def test_a_missing_url_is_named_as_a_missing_url(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard, "azure")
        key_input(step).value = "azure-key-0000000000000000"
        await pilot.pause()
        # Review round 1, live F8: the URL's own line sits under the URL field.
        url_line = static_text(step, "#setup-provider-endpoint-status")
        assert url_line == "Add your Azure OpenAI resource URL here to continue."
        help_line = static_text(step, "#setup-provider-key-status")
        assert "URL" not in help_line
        assert "API key is needed" not in help_line and "Needs a" not in help_line
        ok, error = await step.commit()
        assert not ok
        assert "Azure OpenAI resource URL" in error
        assert "API key required" not in error


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_key", sorted(_BASE_URL_CASES))
@private_profile_test
async def test_a_per_account_provider_reaches_a_saved_ready_config(
    provider_key, request, monkeypatch
):
    """provider-05: one wizard drive per provider, to a saved, ready config."""
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
    from tldw_chatbook.Chat.provider_setup_facts import provider_setup_facts
    from tldw_chatbook.config import load_cli_config_and_ensure_existence

    monkeypatch.delenv(provider_setup_facts(provider_key).env_var, raising=False)
    _label, _example, entered = _BASE_URL_CASES[provider_key]
    typed_key = f"{provider_key}-wizard-key-0000000000000000"
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard, provider_key)
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        endpoint.value = entered
        await pilot.pause()
        key_input(step).value = typed_key
        await pilot.pause()
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _container, model = current_step(wizard)
        custom = model.query_one("#setup-model-custom", Input)
        custom.value = f"{provider_key}-model"
        await pilot.pause()
        await pilot.press("ctrl+n")
        for _ in range(150):
            if current_step(wizard)[1] is not model:
                break
            await pilot.pause(0.05)
        assert current_step(wizard)[1] is not model, pinned_error(wizard)

    saved = load_cli_config_and_ensure_existence(force_reload=True)
    table = saved["api_settings"][provider_key]
    assert table["api_key"] == typed_key
    assert table["api_base_url"].startswith(entered.rstrip("/"))
    readiness = get_provider_readiness(provider_key, saved, environ={})
    assert readiness.ready, readiness.reason


def _template_endpoint(provider_key: str) -> str:
    from tldw_chatbook.Chat.console_provider_endpoints import first_configured_endpoint
    from tldw_chatbook.config import load_settings

    table = (load_settings().get("api_settings") or {}).get(provider_key) or {}
    return first_configured_endpoint(table) or ""


@pytest.mark.asyncio
async def test_every_local_rows_template_endpoint_resolves_in_the_wizard():
    """new-provider-02: a sweep that would have caught KoboldCpp."""
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = current_step(wizard)
        from tldw_chatbook.UI.Wizards.first_run_step_guard import (
            first_run_provider_catalog,
        )

        failures = {}
        for entry in first_run_provider_catalog():
            if entry.requires_api_key:
                continue
            endpoint = step._initial_endpoint_for(entry.readiness_key)
            if not endpoint:
                continue
            resolution = resolve_provider_endpoint(entry.readiness_key, endpoint)
            if resolution.errors or resolution.chat_url is None:
                failures[entry.readiness_key] = (endpoint, resolution.errors)
        assert failures == {}


def test_koboldcpp_saves_the_native_route_chat_with_kobold_posts_to():
    """new-provider-02: chat_with_kobold POSTs its native payload to api_url."""
    for entered in (
        "http://localhost:5001",
        "http://localhost:5001/api/v1/generate",
        "http://localhost:5001/v1",
    ):
        resolution = resolve_provider_endpoint("koboldcpp", entered)
        assert resolution.errors == (), entered
        assert resolution.chat_url == "http://localhost:5001/api/v1/generate"
        assert resolution.models_url == "http://localhost:5001/v1/models"
        mutation = build_provider_setup_mutation(
            ProviderSetupDraft(
                provider="koboldcpp",
                model="kobold-model",
                endpoint=entered,
                credential_source="none",
                credential_revision=0,
                draft_generation=1,
            ),
            {"api_settings": {}},
        )
        table = mutation.section_values["api_settings.koboldcpp"]
        assert table["api_url"] == "http://localhost:5001/api/v1/generate"


@pytest.mark.asyncio
async def test_koboldcpp_says_what_to_type(monkeypatch):
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard, "koboldcpp")
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        assert "http://127.0.0.1:5001" in endpoint.placeholder
        assert static_text(step, "#setup-provider-endpoint-status") == ""
        assert "Chat URL: http://localhost:5001/api/v1/generate" in static_text(
            step, "#setup-provider-effective-chat"
        )


def test_zai_previews_and_saves_its_own_route():
    """provider-20: the preset owns its route suffix."""
    base = "https://api.z.ai/api/paas/v4"
    resolution = resolve_provider_endpoint("zai", base)
    assert resolution.chat_url == f"{base}/chat/completions"
    mutation = build_provider_setup_mutation(
        ProviderSetupDraft(
            provider="zai",
            model="glm-4.6",
            endpoint=base,
            credential_source="draft",
            credential_value="zai-key-0000000000000000",
            credential_revision=0,
            draft_generation=1,
        ),
        {"api_settings": {}},
    )
    assert mutation.section_values["api_settings.zai"]["api_base_url"] == base


def _catalog_template_bases():
    from tldw_chatbook.Chat.console_session_settings import settings_provider_catalog

    cases = []
    for entry in settings_provider_catalog():
        if not entry.requires_api_key:
            continue
        template = _template_endpoint(entry.readiness_key)
        if template:
            cases.append((entry.readiness_key, template))
    return cases


@pytest.mark.parametrize(
    ("provider_key", "template"), _catalog_template_bases(), ids=lambda v: str(v)
)
def test_every_presets_saved_base_equals_its_template_unless_edited(
    provider_key, template
):
    """provider-20 sweep: no preset's untouched base is rewritten on save."""
    from tldw_chatbook.Chat.provider_setup_persistence import provider_endpoint_key

    mutation = build_provider_setup_mutation(
        ProviderSetupDraft(
            provider=provider_key,
            model="some-model",
            endpoint=template,
            credential_source="draft",
            credential_value="sk-sweep-0000000000000000",
            credential_revision=0,
            draft_generation=1,
        ),
        {"api_settings": {}},
    )
    section = next(
        values
        for name, values in mutation.section_values.items()
        if name.startswith("api_settings.")
    )
    endpoint_key = provider_endpoint_key(provider_key)
    assert section[endpoint_key] == template


@pytest.mark.parametrize("typed", ["htp:/foo", "http:/127.0.0.1:8080", "htps://x.y"])
def test_a_scheme_typo_asks_for_http_or_https(typed):
    """provider-15: 'htp:/foo' read as 'must not include an empty port'."""
    resolution = resolve_provider_endpoint("custom", typed)
    assert resolution.errors
    assert "http:// or https://" in resolution.errors[0]


@pytest.mark.asyncio
async def test_an_invalid_endpoint_names_the_field_and_why_test_is_off():
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard, "custom")
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        endpoint.value = "htp:/foo"
        await pilot.pause()
        status = static_text(step, "#setup-provider-endpoint-status")
        assert "http:// or https://" in status
        assert step.query_one("#setup-provider-test", Button).disabled
        assert "to enable Test connection" in status
        ok, error = await step.commit()
        assert not ok
        assert "endpoint" in error.lower() and "above" in error


@pytest.mark.asyncio
async def test_a_url_typed_in_one_burst_after_a_pick_lands_whole():
    """TASK-34332: pick llama.cpp, then Tab, End, Backspaces and the URL at once.

    ``pilot.press`` drains the loop between keys, so it cannot see this; the
    burst is posted the way the driver posts it (lessons-textual). On base the
    End reached the provider list before Tab's focus move, jumping it to the
    last row (a legacy alias) and replacing the endpoint.
    """
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    url = "http://127.0.0.1:9099"
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = current_step(wizard)
        choices = provider_list(step)
        choices.focus()
        choices.highlighted = choices.get_option_index("provider-llama_cpp")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert step.selected_provider_key == "llama_cpp"
        keys = ["tab", "end"] + ["backspace"] * 30 + list(url)
        for key in keys:
            character = key if len(key) == 1 else None
            event = events.Key(key, character)
            event.set_sender(app)
            app._driver.send_message(event)  # noqa: SLF001 - the driver path
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        # A fixed pause proves nothing under load: wait until every key of
        # the burst has been handled (the last one lands in some widget).
        for _ in range(200):
            if endpoint.value == url or step.selected_provider_key != "llama_cpp":
                break
            await pilot.pause(0.05)
        await pilot.pause(0.2)
        assert step.selected_provider_key == "llama_cpp"
        assert endpoint.value == url


@pytest.mark.asyncio
@pytest.mark.local_server_probe
async def test_koboldcpp_discovery_asks_its_openai_listing_so_a_failure_is_named():
    """Review round 1, live F11: KoboldCpp's discovery was handed the saved
    native /api/v1/generate route, which discovery refuses to list from, so no
    request was made and nothing answering read as a bare 'Couldn't discover
    models'. Discovery now asks the OpenAI listing KoboldCpp also serves, and a
    refused connection is named like Ollama's."""
    import httpx

    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        discover_openai_compatible_models,
    )
    from tldw_chatbook.UI.Wizards import first_run_setup_state as state
    from tldw_chatbook.UI.Wizards.first_run_model_discovery import (
        _first_run_discovery_staged_settings,
    )

    draft = state.resolve_first_run_provider_draft(
        state.FirstRunProviderDraft(
            provider="koboldcpp",
            endpoint="http://127.0.0.1:5001",
            credential=state.ProviderCredentialDraft("none", ""),
        ),
        {},
    )
    key = state.build_first_run_model_discovery_key(draft)
    table = _first_run_discovery_staged_settings(draft, key)["api_settings"]["koboldcpp"]
    endpoint = next(value for value in table.values() if str(value).startswith("http"))
    assert endpoint == "http://127.0.0.1:5001/v1/models"
    assert draft.endpoint.endswith("/api/v1/generate"), "what is saved is unchanged"

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request) from ConnectionRefusedError(
            61, "Connection refused"
        )

    result = await discover_openai_compatible_models(
        provider="koboldcpp",
        provider_list_key="koboldcpp",
        endpoint=endpoint,
        api_key=None,
        client=httpx.AsyncClient(transport=httpx.MockTransport(refused)),
    )
    assert result.error is not None and result.error.category == "connection_refused"


@pytest.mark.asyncio
async def test_fixing_the_endpoint_takes_its_refusal_down():
    """Review round 1, live F2: after Next was refused for an invalid
    endpoint, the pinned '✗ The endpoint isn't a valid address' stayed through
    edits and tests -- beside '✓ Connected'. An edit of the field it names
    takes it down (as a key edit does for a key refusal)."""
    from Tests.Wizards.provider_step_harness import pinned_error, wait_until

    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard, "custom")
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        endpoint.value = "htp://127.0.0.1:9099"
        await pilot.pause()
        await pilot.press("ctrl+n")
        await wait_until(pilot, lambda: "valid address" in pinned_error(wizard))
        endpoint.focus()
        await pilot.press("home", "right", "right", "t")
        await pilot.pause()
        assert endpoint.value == "http://127.0.0.1:9099"
        assert pinned_error(wizard) == ""
