from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget
from tldw_chatbook.runtime_policy.bootstrap import RuntimePolicyContext
from tldw_chatbook.runtime_policy.server_credentials import (
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_API_KEY,
    SERVER_CREDENTIAL_BEARER_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
    InMemoryServerCredentialStore,
)
from tldw_chatbook.runtime_policy.server_context import (
    RuntimeServerContextProvider,
    ServerCredentialsUnavailable,
    ServerContextUnavailable,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


class SavingRuntimeStore:
    def __init__(self) -> None:
        self.saved_states: list[RuntimeSourceState] = []

    def save(self, state: RuntimeSourceState) -> None:
        self.saved_states.append(state)


class CountingTargetStore(ConfiguredServerTargetStore):
    def __init__(self, path, targets: list[ConfiguredServerTarget]) -> None:
        super().__init__(path)
        self.save_targets(targets)
        self.get_target_calls = 0

    def get_target(self, server_id: str) -> ConfiguredServerTarget | None:
        self.get_target_calls += 1
        return super().get_target(server_id)


class RaisingCredentialStore:
    def set_secret(self, server_id: str, purpose: str, secret: str) -> None:
        raise RuntimeError("keyring unavailable")

    def get_secret(self, server_id: str, purpose: str) -> str | None:
        raise RuntimeError("keyring unavailable")

    def delete_secret(self, server_id: str, purpose: str) -> None:
        raise RuntimeError("keyring unavailable")

    def clear_server(self, server_id: str) -> None:
        raise RuntimeError("keyring unavailable")

    def clear_all(self) -> None:
        raise RuntimeError("keyring unavailable")


def _runtime_context(
    *,
    active_source: str = "server",
    active_server_id: str | None = "https://server.example.com/api",
    server_configured: bool = True,
) -> RuntimePolicyContext:
    return RuntimePolicyContext(
        state=RuntimeSourceState(
            active_source=active_source,
            active_server_id=active_server_id,
            server_configured=server_configured,
            last_known_server_label="Server",
        ),
        store=SavingRuntimeStore(),
    )


def _target_store(tmp_path, targets: list[ConfiguredServerTarget] | None = None) -> ConfiguredServerTargetStore:
    store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    if targets is not None:
        store.save_targets(targets)
    return store


def _provider(
    tmp_path,
    *,
    runtime_context: RuntimePolicyContext | None = None,
    targets: list[ConfiguredServerTarget] | None = None,
    credential_store: InMemoryServerCredentialStore | None = None,
    app_config: dict | None = None,
) -> RuntimeServerContextProvider:
    return RuntimeServerContextProvider(
        runtime_context=runtime_context or _runtime_context(),
        target_store=_target_store(tmp_path, targets),
        credential_store=credential_store or InMemoryServerCredentialStore(),
        app_config=app_config or {},
    )


def test_resolves_matching_target_and_credential_store_secret(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN, "bearer-secret")
    target = ConfiguredServerTarget(
        server_id="https://server.example.com/api",
        label="Primary",
        base_url="https://server.example.com/api/",
        auth_mode="bearer",
        is_default=True,
    )

    provider = _provider(
        tmp_path,
        credential_store=credentials,
        targets=[target],
    )

    context = provider.get_active_context()

    assert context.active_server_id == "https://server.example.com/api"
    assert context.label == "Primary"
    assert context.base_url == "https://server.example.com/api"
    assert context.auth_method == "bearer"
    assert context.auth_token == "bearer-secret"
    assert context.credential_source == f"credential_store:{SERVER_CREDENTIAL_BEARER_TOKEN}"
    assert context.target == target


def test_rejects_server_mode_without_active_server(tmp_path):
    provider = _provider(
        tmp_path,
        runtime_context=_runtime_context(active_server_id=None, server_configured=True),
    )

    with pytest.raises(ServerContextUnavailable):
        provider.get_active_context()


def test_legacy_fallback_works_when_no_target_exists_and_app_config_matches_active_server(tmp_path):
    provider = _provider(
        tmp_path,
        app_config={
            "tldw_api": {
                "base_url": "https://Server.Example.com/api/",
                "bearer_token": "legacy-bearer",
                "auth_mode": "bearer",
            }
        },
    )

    context = provider.get_active_context()

    assert context.active_server_id == "https://server.example.com/api"
    assert context.label == "server.example.com"
    assert context.base_url == "https://server.example.com/api"
    assert context.auth_method == "bearer"
    assert context.auth_token == "legacy-bearer"
    assert context.credential_source == f"credential_store:{SERVER_CREDENTIAL_BEARER_TOKEN}"
    assert context.target.server_id == "https://server.example.com/api"
    assert context.target.auth_reference == "legacy:tldw_api"


def test_legacy_target_prefers_credential_store_token_over_legacy_config(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "stored-access")

    provider = _provider(
        tmp_path,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Legacy Profile",
                base_url="https://server.example.com/api",
                auth_mode="bearer",
                auth_reference="legacy:tldw_api",
                is_default=True,
            )
        ],
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "stale-legacy",
                "auth_mode": "bearer",
            }
        },
    )

    context = provider.get_active_context()

    assert context.auth_token == "stored-access"
    assert context.credential_source == f"credential_store:{SERVER_CREDENTIAL_ACCESS_TOKEN}"


def test_legacy_config_token_imports_only_for_active_server_profile(tmp_path):
    credentials = InMemoryServerCredentialStore()
    target_store = _target_store(
        tmp_path,
        [
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api",
                auth_mode="bearer",
                auth_reference="legacy:tldw_api",
                is_default=True,
            ),
            ConfiguredServerTarget(
                server_id="https://backup.example.com/api",
                label="Backup",
                base_url="https://backup.example.com/api",
                auth_mode="bearer",
                auth_reference="legacy:tldw_api",
            ),
        ],
    )
    provider = RuntimeServerContextProvider(
        runtime_context=_runtime_context(),
        target_store=target_store,
        credential_store=credentials,
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "legacy-bearer",
                "auth_mode": "bearer",
            }
        },
    )

    context = provider.get_active_context()

    assert context.active_server_id == "https://server.example.com/api"
    assert context.auth_token == "legacy-bearer"
    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
    ) == "legacy-bearer"
    assert credentials.get_secret(
        "https://backup.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
    ) is None


def test_legacy_config_token_does_not_apply_to_nonmatching_active_server_profile(tmp_path):
    credentials = InMemoryServerCredentialStore()
    provider = _provider(
        tmp_path,
        runtime_context=_runtime_context(active_server_id="https://backup.example.com/api"),
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://backup.example.com/api",
                label="Backup",
                base_url="https://backup.example.com/api",
                auth_mode="bearer",
                auth_reference="legacy:tldw_api",
                is_default=True,
            )
        ],
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "legacy-bearer",
                "auth_mode": "bearer",
            }
        },
    )

    context = provider.get_active_context()

    assert context.active_server_id == "https://backup.example.com/api"
    assert context.auth_token is None
    assert context.credential_source == "none"
    assert credentials.get_secret(
        "https://backup.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
    ) is None


def test_legacy_fallback_without_target_prefers_credential_store_token_over_legacy_config(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN, "stored-bearer")

    provider = _provider(
        tmp_path,
        credential_store=credentials,
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "stale-legacy",
                "auth_mode": "bearer",
            }
        },
    )

    context = provider.get_active_context()

    assert context.auth_token == "stored-bearer"
    assert context.credential_source == f"credential_store:{SERVER_CREDENTIAL_BEARER_TOKEN}"


def test_legacy_fallback_uses_config_token_when_credential_store_is_unavailable(tmp_path):
    provider = _provider(
        tmp_path,
        credential_store=RaisingCredentialStore(),
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "legacy-bearer",
                "auth_mode": "bearer",
            }
        },
    )

    context = provider.get_active_context()

    assert context.auth_token == "legacy-bearer"
    assert context.credential_source == "legacy:tldw_api"


def test_explicit_keyring_reference_raises_typed_error_when_credential_store_is_unavailable(tmp_path):
    provider = _provider(
        tmp_path,
        credential_store=RaisingCredentialStore(),
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api",
                auth_mode="bearer",
                auth_reference=f"keyring:{SERVER_CREDENTIAL_ACCESS_TOKEN}",
                is_default=True,
            )
        ],
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "legacy-bearer",
                "auth_mode": "bearer",
            }
        },
    )

    with pytest.raises(ServerCredentialsUnavailable) as exc:
        provider.get_active_context()

    assert isinstance(exc.value.__cause__, RuntimeError)


def test_bearer_auth_prefers_bearer_token_then_access_token_before_legacy_config(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "stored-access")
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN, "stored-bearer")

    provider = _provider(
        tmp_path,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Legacy Profile",
                base_url="https://server.example.com/api",
                auth_mode="bearer",
                auth_reference="legacy:tldw_api",
                is_default=True,
            )
        ],
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "stale-legacy",
                "auth_mode": "bearer",
            }
        },
    )

    context = provider.get_active_context()

    assert context.auth_token == "stored-bearer"
    assert context.credential_source == f"credential_store:{SERVER_CREDENTIAL_BEARER_TOKEN}"


def test_context_computes_bearer_headers_from_effective_auth_token(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN, "stored-bearer")
    provider = _provider(
        tmp_path,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api",
                auth_mode="bearer",
                is_default=True,
            )
        ],
    )

    context = provider.get_active_context()

    assert context.server_headers == {"Authorization": "Bearer stored-bearer"}


def test_api_key_auth_prefers_api_key_credential_before_legacy_config(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, "stored-api-key")

    provider = _provider(
        tmp_path,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Legacy API Key Profile",
                base_url="https://server.example.com/api",
                auth_mode="api_key",
                auth_reference="legacy:tldw_api",
                is_default=True,
            )
        ],
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "api_key": "stale-legacy-api-key",
                "auth_mode": "api_key",
            }
        },
    )

    context = provider.get_active_context()

    assert context.auth_token == "stored-api-key"
    assert context.credential_source == f"credential_store:{SERVER_CREDENTIAL_API_KEY}"


def test_context_computes_api_key_headers_from_effective_auth_token(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, "stored-api-key")
    provider = _provider(
        tmp_path,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api",
                auth_mode="api_key",
                is_default=True,
            )
        ],
    )

    context = provider.get_active_context()

    assert context.server_headers == {"X-API-KEY": "stored-api-key"}


def test_context_capabilities_reflect_runtime_state_and_target_status(tmp_path):
    runtime_context = _runtime_context()
    runtime_context.state = replace(
        runtime_context.state,
        server_reachability="reachable",
        server_auth_state="authenticated",
        last_known_server_label="Runtime Label",
    )
    provider = _provider(
        tmp_path,
        runtime_context=runtime_context,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api",
                auth_mode="api_key",
                is_default=True,
                last_known_reachability="unreachable",
                last_known_auth_state="auth_required",
                last_known_server_label="Target Label",
            )
        ],
    )

    context = provider.get_active_context()

    assert context.capabilities == {
        "server_configured": True,
        "reachability": "reachable",
        "auth_state": "authenticated",
        "last_known_server_label": "Runtime Label",
        "target_last_known_reachability": "unreachable",
        "target_last_known_auth_state": "auth_required",
        "target_last_known_server_label": "Target Label",
    }


def test_context_capabilities_update_when_active_server_runtime_state_changes(tmp_path):
    runtime_context = _runtime_context()
    target_store = _target_store(
        tmp_path,
        [
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api",
                auth_mode="api_key",
                is_default=True,
                last_known_reachability="reachable",
                last_known_auth_state="authenticated",
                last_known_server_label="Primary Target",
            ),
            ConfiguredServerTarget(
                server_id="https://backup.example.com/api",
                label="Backup",
                base_url="https://backup.example.com/api",
                auth_mode="api_key",
                last_known_reachability="unreachable",
                last_known_auth_state="session_invalid",
                last_known_server_label="Backup Target",
            ),
        ],
    )
    provider = RuntimeServerContextProvider(
        runtime_context=runtime_context,
        target_store=target_store,
        credential_store=InMemoryServerCredentialStore(),
        app_config={},
    )

    first_context = provider.get_active_context()
    runtime_context.state = RuntimeSourceState(
        active_source="server",
        active_server_id="https://backup.example.com/api",
        server_configured=True,
        server_reachability="unreachable",
        server_auth_state="session_invalid",
        last_known_server_label="Backup Runtime",
    )
    second_context = provider.get_active_context()

    assert first_context.target.server_id == "https://server.example.com/api"
    assert first_context.capabilities["reachability"] == "unknown"
    assert first_context.capabilities["target_last_known_server_label"] == "Primary Target"
    assert second_context.target.server_id == "https://backup.example.com/api"
    assert second_context.capabilities["reachability"] == "unreachable"
    assert second_context.capabilities["auth_state"] == "session_invalid"
    assert second_context.capabilities["last_known_server_label"] == "Backup Runtime"
    assert second_context.capabilities["target_last_known_server_label"] == "Backup Target"


def test_profile_target_auth_resolution_does_not_re_resolve_active_target(tmp_path):
    target_store = CountingTargetStore(
        tmp_path / "targets.json",
        [
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api",
                auth_mode="api_key",
                is_default=True,
            )
        ],
    )
    provider = RuntimeServerContextProvider(
        runtime_context=_runtime_context(),
        target_store=target_store,
        credential_store=InMemoryServerCredentialStore(),
        app_config={},
    )

    context = provider.get_active_context()

    assert context.auth_token is None
    assert target_store.get_target_calls == 1


def test_build_client_uses_active_context_base_url_and_bearer_token(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-secret")
    provider = _provider(
        tmp_path,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api/",
                auth_mode="bearer",
                is_default=True,
            )
        ],
    )

    client = provider.build_client()

    assert client.base_url == "https://server.example.com/api"
    assert client.bearer_token == "access-secret"
    assert client.token is None


def test_build_client_reuses_cached_client_for_same_active_context_and_token(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-secret")
    provider = _provider(
        tmp_path,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api/",
                auth_mode="bearer",
                is_default=True,
            )
        ],
    )

    first_client = provider.build_client()
    second_client = provider.build_client()

    assert second_client is first_client


@pytest.mark.asyncio
async def test_stored_token_change_replaces_cached_client_and_closes_opened_old_client(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-1")
    provider = _provider(
        tmp_path,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api/",
                auth_mode="bearer",
                is_default=True,
            )
        ],
    )

    first_client = provider.build_client()
    opened_http_client = await first_client._get_client()

    provider.store_auth_tokens(access_token="access-2")
    second_client = provider.build_client()
    cache_key_repr = repr(provider._cached_client_key)
    await provider.close_cached_client()

    assert second_client is not first_client
    assert opened_http_client.is_closed
    assert "access-1" not in cache_key_repr
    assert "access-2" not in cache_key_repr


@pytest.mark.asyncio
async def test_clear_active_server_auth_tokens_invalidates_cache_and_preserves_static_credentials(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-1")
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_REFRESH_TOKEN, "refresh-1")
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, "api-key-1")
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN, "bearer-1")
    provider = _provider(
        tmp_path,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api/",
                auth_mode="bearer",
                is_default=True,
            )
        ],
    )

    first_client = provider.build_client()
    opened_http_client = await first_client._get_client()

    provider.clear_active_server_auth_tokens()
    second_client = provider.build_client()
    await provider.close_cached_client()

    assert second_client is not first_client
    assert opened_http_client.is_closed
    assert second_client.bearer_token == "bearer-1"
    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
    ) is None
    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_REFRESH_TOKEN,
    ) is None
    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_API_KEY,
    ) == "api-key-1"
    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
    ) == "bearer-1"


@pytest.mark.asyncio
async def test_clear_all_credentials_invalidates_cached_client_and_removes_imported_credentials(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://backup.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "backup-access")
    provider = _provider(
        tmp_path,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api/",
                auth_mode="bearer",
                auth_reference="legacy:tldw_api",
                is_default=True,
            )
        ],
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "legacy-bearer",
                "auth_mode": "bearer",
            }
        },
    )

    first_client = provider.build_client()
    opened_http_client = await first_client._get_client()

    provider.clear_all_credentials()
    if provider._pending_client_close_tasks:
        await asyncio.gather(*provider._pending_client_close_tasks)

    assert provider._cached_client is None
    assert opened_http_client.is_closed
    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
    ) is None
    assert credentials.get_secret(
        "https://backup.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
    ) is None

    with pytest.raises(ServerCredentialsUnavailable):
        provider.get_active_context()
    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
    ) is None


def test_target_store_json_and_target_metadata_do_not_contain_stored_secret(tmp_path):
    secret = "literal-provider-token-must-not-leak"
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, secret)
    target_store = _target_store(
        tmp_path,
        [
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api/",
                auth_mode="api_key",
                is_default=True,
            )
        ],
    )
    reloaded_store = ConfiguredServerTargetStore(target_store.path)
    provider = RuntimeServerContextProvider(
        runtime_context=_runtime_context(),
        target_store=target_store,
        credential_store=credentials,
        app_config={},
    )

    context = provider.get_active_context()
    reloaded_target = reloaded_store.get_target("https://server.example.com/api")

    assert context.auth_token == secret
    assert provider.build_client().token == secret

    payload = json.loads(target_store.path.read_text(encoding="utf-8"))
    persisted_target_json = json.dumps(payload, sort_keys=True)
    target_metadata_json = json.dumps(
        {
            "context_capabilities": context.capabilities,
            "context_target": context.target.to_dict(),
            "reloaded_target": reloaded_target.to_dict() if reloaded_target else None,
        },
        sort_keys=True,
    )
    assert secret not in persisted_target_json
    assert secret not in target_metadata_json


def test_clear_active_server_credentials_and_clear_server_credentials_clear_per_server_secrets(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN, "active-secret")
    credentials.set_secret("server-b", SERVER_CREDENTIAL_BEARER_TOKEN, "other-secret")
    provider = _provider(tmp_path, credential_store=credentials)

    provider.clear_active_server_credentials()

    assert credentials.get_secret("https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN) is None
    assert credentials.get_secret("server-b", SERVER_CREDENTIAL_BEARER_TOKEN) == "other-secret"

    provider.clear_server_credentials("server-b")

    assert credentials.get_secret("server-b", SERVER_CREDENTIAL_BEARER_TOKEN) is None


@pytest.mark.asyncio
async def test_switching_active_server_rebuilds_client_with_new_profile_and_closes_old_client(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "shared-access")
    credentials.set_secret("https://backup.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "shared-access")
    runtime_context = _runtime_context()
    provider = RuntimeServerContextProvider(
        runtime_context=runtime_context,
        target_store=_target_store(
            tmp_path,
            [
                ConfiguredServerTarget(
                    server_id="https://server.example.com/api",
                    label="Primary",
                    base_url="https://shared.example.com/api",
                    auth_mode="bearer",
                    is_default=True,
                ),
                ConfiguredServerTarget(
                    server_id="https://backup.example.com/api",
                    label="Backup",
                    base_url="https://shared.example.com/api",
                    auth_mode="bearer",
                ),
            ],
        ),
        credential_store=credentials,
        app_config={},
    )

    first_client = provider.build_client()
    opened_http_client = await first_client._get_client()

    runtime_context.state = replace(
        runtime_context.state,
        active_server_id="https://backup.example.com/api",
    )
    second_client = provider.build_client()
    await provider.close_cached_client()

    assert second_client is not first_client
    assert opened_http_client.is_closed
    assert second_client.base_url == "https://shared.example.com/api"
    assert second_client.bearer_token == "shared-access"


def test_clear_active_server_auth_tokens_preserves_static_credentials(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-1")
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_REFRESH_TOKEN, "refresh-1")
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, "api-key-1")
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN, "bearer-1")
    credentials.set_secret("https://backup.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "other-access")
    provider = _provider(tmp_path, credential_store=credentials)

    provider.clear_active_server_auth_tokens()

    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
    ) is None
    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_REFRESH_TOKEN,
    ) is None
    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_API_KEY,
    ) == "api-key-1"
    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
    ) == "bearer-1"
    assert credentials.get_secret(
        "https://backup.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
    ) == "other-access"


def test_store_auth_tokens_scopes_tokens_to_active_server(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://backup.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "other-access")
    credentials.set_secret("https://backup.example.com/api", SERVER_CREDENTIAL_REFRESH_TOKEN, "other-refresh")
    target_store = _target_store(
        tmp_path,
        [
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api/",
                auth_mode="bearer",
                is_default=True,
            ),
            ConfiguredServerTarget(
                server_id="https://backup.example.com/api",
                label="Backup",
                base_url="https://backup.example.com/api/",
                auth_mode="bearer",
            ),
        ],
    )
    provider = RuntimeServerContextProvider(
        runtime_context=_runtime_context(),
        target_store=target_store,
        credential_store=credentials,
        app_config={},
    )

    provider.store_auth_tokens(access_token="access-1", refresh_token="refresh-1")

    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
    ) == "access-1"
    assert credentials.get_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_REFRESH_TOKEN,
    ) == "refresh-1"
    assert credentials.get_secret(
        "https://backup.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
    ) == "other-access"
    assert credentials.get_secret(
        "https://backup.example.com/api",
        SERVER_CREDENTIAL_REFRESH_TOKEN,
    ) == "other-refresh"

    payload = json.loads(target_store.path.read_text(encoding="utf-8"))
    assert "access-1" not in json.dumps(payload)
    assert "refresh-1" not in json.dumps(payload)


def test_mismatched_runtime_active_server_and_only_legacy_config_raises(tmp_path):
    provider = _provider(
        tmp_path,
        app_config={
            "tldw_api": {
                "base_url": "https://other.example.com/api",
                "api_key": "wrong-server-secret",
            }
        },
    )

    with pytest.raises(ServerContextUnavailable):
        provider.get_active_context()


def test_runtime_state_remains_authoritative_and_unmutated_during_context_resolution(tmp_path):
    runtime_context = _runtime_context()
    original_state = runtime_context.state
    provider = _provider(
        tmp_path,
        runtime_context=runtime_context,
        targets=[
            ConfiguredServerTarget(
                server_id="https://other.example.com/api",
                label="Other Default",
                base_url="https://other.example.com/api",
                auth_mode="api_key",
                is_default=True,
            )
        ],
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "api_key": "legacy-secret",
            }
        },
    )

    context = provider.get_active_context()

    assert context.active_server_id == original_state.active_server_id
    assert context.base_url == "https://server.example.com/api"
    assert runtime_context.state == original_state
    assert runtime_context.store.saved_states == []
