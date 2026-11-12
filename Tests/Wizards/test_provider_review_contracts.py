"""Review regression controls for Provider credential and loopback contracts."""

import pytest

from tldw_chatbook.Chat.local_server_discovery import build_local_server_candidates
from tldw_chatbook.UI.Wizards.first_run_step_guard import first_run_provider_catalog
from tldw_chatbook.UI.Wizards.first_run_provider_list import ready_badges
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    FirstRunProviderDraft,
    ProviderCredentialDraft,
    build_first_run_provider_commit,
    resolve_first_run_provider_draft,
)


@pytest.mark.parametrize(
    "provider, endpoint, env_var",
    [
        ("azure", "https://review.openai.azure.com", "AZURE_OPENAI_API_KEY"),
        ("databricks", "https://review.azuredatabricks.net", "DATABRICKS_TOKEN"),
        (
            "cloudflare",
            "https://api.cloudflare.com/client/v4/accounts/review/ai/v1",
            "CLOUDFLARE_API_TOKEN",
        ),
    ],
)
@pytest.mark.parametrize("source", ["environment", "stored"])
def test_account_url_commit_retains_existing_credential(
    provider, endpoint, env_var, source, monkeypatch
):
    """A missing saved URL cannot turn a current credential into Clear."""
    monkeypatch.setenv(env_var, "synthetic-account-review-key")
    settings = {"api_key": "synthetic-stored-review-key"} if source == "stored" else {}
    draft = FirstRunProviderDraft(
        provider=provider,
        endpoint=endpoint,
        credential=ProviderCredentialDraft("none", "", 0),
    )
    mutation = build_first_run_provider_commit(
        draft, "review-model", {"api_settings": {provider: settings}}
    )
    saved = mutation.section_values[f"api_settings.{provider}"]
    assert saved["credential_source"] == source
    if source == "environment":
        assert saved["api_key_env_var"] == env_var
        assert "api_key" not in saved
    else:
        assert saved["api_key"] == "synthetic-stored-review-key"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:5001",
        "http://localhost:5001/v1",
        "http://localhost:5001/prefix",
    ],
)
def test_kobold_wizard_draft_refuses_non_native_routes(endpoint):
    """The native prompt sender cannot consume an OpenAI chat URL."""
    draft = FirstRunProviderDraft(
        provider="koboldcpp",
        endpoint=endpoint,
        credential=ProviderCredentialDraft("none", "", 0),
    )
    with pytest.raises(ValueError, match="api/v1/generate"):
        resolve_first_run_provider_draft(draft, {"api_settings": {}})


def test_fresh_discovery_probes_both_loopback_spellings_at_each_default_port():
    candidates = build_local_server_candidates({})
    urls = {c.base_url for c in candidates}
    for port in (8080, 9099, 11434, 1234, 1337, 5001):
        assert f"http://127.0.0.1:{port}" in urls
        assert f"http://localhost:{port}" in urls
    assert len(candidates) == len(urls) == 12


@pytest.mark.parametrize(
    "settings, env, expected",
    [
        (
            {"credential_source": "none", "api_key": "synthetic-saved-key"},
            {"OPENAI_API_KEY": "synthetic-exported-key"},
            {},
        ),
        (
            {
                "credential_source": "environment",
                "api_key_env_var": "REVIEW_PRIVATE_KEY",
            },
            {"REVIEW_PRIVATE_KEY": "synthetic-exported-key"},
            {"openai": "key in REVIEW_PRIVATE_KEY"},
        ),
        (
            {"api_key": "synthetic-saved-key", "api_key_env_var": "REVIEW_PRIVATE_KEY"},
            {"REVIEW_PRIVATE_KEY": "synthetic-exported-key"},
            {"openai": "key saved"},
        ),
        (
            {
                "credential_source": "environment",
                "api_key": "synthetic-saved-key",
                "api_key_env_var": "REVIEW_PRIVATE_KEY",
            },
            {"REVIEW_PRIVATE_KEY": "synthetic-exported-key"},
            {"openai": "key in REVIEW_PRIVATE_KEY"},
        ),
    ],
)
def test_ready_badge_respects_current_credential_source(settings, env, expected):
    entries = tuple(
        e for e in first_run_provider_catalog() if e.readiness_key == "openai"
    )
    assert (
        ready_badges(entries, {"api_settings": {"openai": settings}}, env) == expected
    )


@pytest.mark.asyncio
@pytest.mark.local_server_probe
@pytest.mark.parametrize(
    "answering_hosts, expected",
    [
        ({"127.0.0.1", "localhost"}, "http://127.0.0.1:1234"),
        ({"localhost"}, "http://localhost:1234"),
    ],
)
async def test_fresh_scan_probes_both_hosts_and_keeps_one_answering_address(
    answering_hosts, expected
):
    import httpx
    from tldw_chatbook.Chat.local_server_discovery import discover_local_servers

    seen = []

    async def reply(request):
        seen.append((request.url.host, request.url.port))
        if request.url.port == 1234 and request.url.host in answering_hosts:
            return httpx.Response(200, json={"data": [{"id": "synthetic-local-model"}]})
        raise httpx.ConnectError("refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        results = await discover_local_servers({}, http_client=client)
    assert ("127.0.0.1", 1234) in seen and ("localhost", 1234) in seen
    assert [r.base_url for r in results] == [expected]
    assert results[0].model_ids == ("synthetic-local-model",)


def test_configured_discovery_preserves_port_prefix_and_only_approved_hosts():
    from urllib.parse import urlsplit

    candidates = build_local_server_candidates(
        {
            "api_settings": {
                "custom": {"api_url": "http://localhost:4321/prefix/v1"},
                "custom_2": {"api_url": "https://remote.invalid:9999/v1"},
            }
        }
    )
    urls = {c.base_url for c in candidates}
    assert "http://localhost:4321/prefix" in urls
    assert "http://127.0.0.1:4321/prefix" in urls
    assert {urlsplit(url).hostname for url in urls} == {"127.0.0.1", "localhost"}
    assert len(candidates) == len(urls)


def test_borrowed_subscription_never_falls_back_to_an_old_api_key(monkeypatch):
    from types import SimpleNamespace
    from tldw_chatbook.LLM_Calls import anthropic_subscription
    from tldw_chatbook.UI.Wizards.first_run_provider_step import ProviderStep

    monkeypatch.setattr(
        anthropic_subscription, "subscription_credential_status", lambda **kw: "pending"
    )
    settings = {
        "auth_source": "claude_subscription",
        "api_key": "synthetic-old-api-key",
        "api_key_env_var": "REVIEW_ANTHROPIC_KEY",
    }
    config = {"api_settings": {"anthropic": settings}}
    owner = SimpleNamespace(
        selected_provider_key="anthropic",
        wizard=SimpleNamespace(app_instance=SimpleNamespace(app_config=config)),
        query_one=lambda *args: SimpleNamespace(value="", display=False),
        _provider_drafts={},
        _clear_requested=False,
        _environment=lambda: {"REVIEW_ANTHROPIC_KEY": "synthetic-old-env-key"},
        _provider_settings=lambda provider: settings,
    )
    source, value, readiness = ProviderStep._credential_at_request_boundary(owner)
    assert source == "none" and value is None
    assert not readiness.ready and readiness.api_key is None
    assert readiness.subscription_status == "pending"


def test_subscription_commit_preserves_inactive_api_credentials(monkeypatch):
    from tldw_chatbook.LLM_Calls import anthropic_subscription

    monkeypatch.setattr(
        anthropic_subscription, "subscription_credential_status", lambda **kw: "pending"
    )
    config = {
        "api_settings": {
            "anthropic": {
                "auth_source": "claude_subscription",
                "api_key": "synthetic-old-api-key",
                "api_key_env_var": "REVIEW_ANTHROPIC_KEY",
            }
        }
    }
    draft = FirstRunProviderDraft(
        provider="anthropic",
        endpoint="",
        credential=ProviderCredentialDraft("none", "", 0),
    )
    mutation = build_first_run_provider_commit(draft, "synthetic-model", config)
    values = mutation.section_values.get("api_settings.anthropic", {})
    deletes = mutation.delete_keys.get("api_settings.anthropic", ())
    assert "api_key" not in values and "api_key_env_var" not in values
    assert "api_key" not in deletes and "api_key_env_var" not in deletes


@pytest.mark.bootstrap_profile
def test_provider_refusal_status_contrast_reuses_every_shipped_theme_gate():
    # The assertion is static CSS/theme work; Wizard scope avoids the UI
    # conftest's unrelated app import under a per-test redirected config.
    from Tests.UI.test_theme_contrast import (
        ALL_THEMES,
        test_first_run_provider_status_styles_clear_aa_on_every_theme,
    )

    for theme in ALL_THEMES:
        test_first_run_provider_status_styles_clear_aa_on_every_theme(theme)
