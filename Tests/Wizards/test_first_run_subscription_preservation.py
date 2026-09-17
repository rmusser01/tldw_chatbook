"""Subscription setup preserves inactive API-key choices until an explicit edit."""

import tomllib
from dataclasses import replace
from types import MappingProxyType

import pytest
import toml

from Tests.private_profile import private_profile_test
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat import provider_setup_persistence as persistence
from tldw_chatbook.LLM_Calls import anthropic_subscription as subscription
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    FirstRunProviderDraft,
    ProviderCredentialDraft,
    build_first_run_provider_commit,
)


@pytest.fixture(autouse=True)
def ready_subscription(monkeypatch):
    monkeypatch.setattr(
        subscription, "subscription_credential_status", lambda **_kwargs: "ready"
    )


def _config(source="stored"):
    settings = {
        "auth_source": "claude_subscription",
        "api_base_url": "https://api.anthropic.com",
        "api_key": "inactive-stored-key",
        "api_key_env_var": "INACTIVE_ANTHROPIC_KEY",
        "model": "old-model",
    }
    if source is not None:
        settings["credential_source"] = source
    return {"api_settings": {"anthropic": settings}}


def _commit(config, credential=None):
    return build_first_run_provider_commit(
        FirstRunProviderDraft(
            provider="anthropic",
            endpoint="https://api.anthropic.com",
            credential=credential or ProviderCredentialDraft("none", ""),
        ),
        "selected-model",
        config,
    )


def _bind(mutation, snapshot):
    semantic = mutation.semantic_identity
    assert semantic is not None
    identity = persistence.ProviderSetupWriteIdentity(
        provider_key=semantic.provider_key,
        connection_identity=semantic.connection_identity,
        credential_source=semantic.credential_source,
        credential_revision=semantic.credential_revision,
        model_id="selected-model",
        model_provenance="manual",
    )
    guard = persistence.ProviderSetupWriteGuard()
    expected = persistence.capture_expected_provider_setup_state(
        snapshot, identity=identity
    )
    persistence.bind_provider_setup_write_expectation(
        mutation,
        guard=guard,
        expectation=guard.arm(identity),
        expected_state=expected,
    )
    return persistence.project_provider_setup_expected_state(
        snapshot, mutation=mutation, identity=identity
    )


@pytest.mark.parametrize("source", [None, "none", "stored", "environment"])
@pytest.mark.parametrize("status", ["pending", "ready", "missing", "expired"])
def test_unchanged_subscription_mutation_never_writes_inactive_credentials(
    source, status, monkeypatch
):
    monkeypatch.setattr(
        subscription, "subscription_credential_status", lambda **_kwargs: status
    )
    mutation = _commit(_config(source))
    credential_keys = {"api_key", "api_key_env_var", "credential_source"}

    assert credential_keys.isdisjoint(mutation.section_values["api_settings.anthropic"])
    assert credential_keys.isdisjoint(
        mutation.delete_keys.get("api_settings.anthropic", ())
    )
    assert mutation.semantic_identity.credential_source == "none"
    assert "inactive-stored-key" not in repr(mutation)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [None, "none", "stored", "environment"])
@private_profile_test
async def test_unchanged_subscription_preserves_credentials_after_atomic_save_and_reload(
    monkeypatch, source, request
):
    original = _config(source)
    path = config_module.get_cli_config_path()
    path.write_text(toml.dumps(original), encoding="utf-8")
    monkeypatch.setenv("INACTIVE_ANTHROPIC_KEY", "inactive-environment-key")
    snapshot = config_module.get_atomic_config_snapshot()
    mutation = _commit(snapshot.values)
    postcondition = _bind(mutation, snapshot)

    assert persistence.persist_provider_setup(mutation).fully_applied

    saved = tomllib.loads(path.read_text(encoding="utf-8"))
    settings = saved["api_settings"]["anthropic"]
    for key in ("api_key", "api_key_env_var", "credential_source", "auth_source"):
        assert settings.get(key) == original["api_settings"]["anthropic"].get(key)
    assert settings["model"] == "selected-model"
    assert saved["chat_defaults"] == {
        "provider": "anthropic",
        "model": "selected-model",
    }
    assert "inactive-environment-key" not in path.read_text(encoding="utf-8")
    assert postcondition._matches_snapshot(config_module.get_atomic_config_snapshot())


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["", "replacement-key"])
@private_profile_test
async def test_explicit_subscription_clear_or_replacement_remains_authoritative(
    replacement, request
):
    path = config_module.get_cli_config_path()
    path.write_text(toml.dumps(_config()), encoding="utf-8")
    snapshot = config_module.get_atomic_config_snapshot()
    mutation = _commit(snapshot.values, ProviderCredentialDraft("draft", replacement))
    _bind(mutation, snapshot)

    assert persistence.persist_provider_setup(mutation).fully_applied

    saved = tomllib.loads(path.read_text(encoding="utf-8"))["api_settings"]["anthropic"]
    assert saved["auth_source"] == "claude_subscription"
    assert "api_key_env_var" not in saved
    assert saved.get("api_key") == (replacement or None)
    assert saved["credential_source"] == ("stored" if replacement else "none")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [{"auth_source": "api_key"}, {"api_key": "concurrent-key"}],
    ids=["auth-source", "credential"],
)
@private_profile_test
async def test_preserving_subscription_setup_keeps_atomic_conflict_checks(
    change, request
):
    path = config_module.get_cli_config_path()
    path.write_text(toml.dumps(_config()), encoding="utf-8")
    snapshot = config_module.get_atomic_config_snapshot()
    mutation = _commit(snapshot.values)
    _bind(mutation, snapshot)
    assert config_module.apply_settings_mutation_to_cli_config(
        {"api_settings.anthropic": change}
    ).fully_applied

    result = persistence.persist_provider_setup(mutation)

    assert result.conflict
    saved = tomllib.loads(path.read_text(encoding="utf-8"))
    assert saved["api_settings"]["anthropic"]["model"] == "old-model"
    assert saved.get("chat_defaults", {}).get("model") != "selected-model"
    for key, value in change.items():
        assert saved["api_settings"]["anthropic"][key] == value


@pytest.mark.parametrize(
    ("provider", "auth_source", "credential_source"),
    [
        ("openai", "claude_subscription", "none"),
        ("anthropic", "api_key", "none"),
        ("anthropic", "claude_subscription", "stored"),
    ],
)
def test_credential_preservation_is_restricted_to_unchanged_subscription_setup(
    provider, auth_source, credential_source
):
    draft = persistence.ProviderSetupDraft(
        provider=provider,
        model="selected-model",
        endpoint="https://api.anthropic.com",
        credential_source=credential_source,
        credential_revision=0,
        draft_generation=0,
    )
    with pytest.raises(ValueError, match="Credential"):
        persistence.build_provider_setup_mutation(
            draft,
            {"api_settings": {provider: {"auth_source": auth_source}}},
            preserve_credentials=True,
        )


def test_preserving_mutation_still_rejects_unissued_copy():
    mutation = _commit(_config())
    forged = replace(mutation)

    with pytest.raises(ValueError, match="mutation"):
        persistence.persist_provider_setup(forged)
    with pytest.raises(TypeError):
        mutation.section_values["api_settings.anthropic"]["api_key"] = "forged"


def test_preserving_shape_rejects_partial_credential_mutation():
    mutation = _commit(_config())
    values = {
        **mutation.section_values,
        "api_settings.anthropic": MappingProxyType(
            {**mutation.section_values["api_settings.anthropic"], "api_key": "partial"}
        ),
    }
    with pytest.raises(ValueError, match="mutation"):
        replace(mutation, section_values=MappingProxyType(values))
