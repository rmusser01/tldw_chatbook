from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.state.app_state import AppState


def _make_app_like(*, base_url: str = "https://Example.COM:8443/api/") -> SimpleNamespace:
    return SimpleNamespace(
        app_config={
            "tldw_api": {
                "base_url": base_url,
            }
        },
        app_state=AppState(),
    )


def test_app_state_round_trips_runtime_source_state():
    original = AppState(
        runtime_source=RuntimeSourceState(
            active_source="server",
            active_server_id="server-alpha",
            server_configured=True,
            server_reachability="reachable",
            server_reachability_checked_at=datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc),
            server_auth_state="authenticated",
            server_auth_checked_at=datetime(2026, 4, 21, 12, 5, tzinfo=timezone.utc),
            last_known_server_label="Primary Server",
        )
    )

    restored = AppState.from_dict(original.to_dict())

    assert restored.runtime_source == original.runtime_source


def test_app_state_from_dict_ignores_malformed_runtime_source_payload():
    restored = AppState.from_dict(
        {
            "runtime_source": ["not", "a", "mapping"],
        }
    )

    assert restored.runtime_source == RuntimeSourceState()


def test_runtime_source_state_store_round_trips_json(tmp_path):
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    state = RuntimeSourceState(
        active_source="server",
        active_server_id="server-alpha",
        server_configured=True,
        server_reachability="reachable",
        server_auth_state="authenticated",
    )

    store.save(state)
    restored = store.load()

    assert restored == state


def test_runtime_source_state_store_loads_safe_default_on_malformed_json(tmp_path):
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    path = tmp_path / "runtime_policy.json"
    path.write_text("{not-json", encoding="utf-8")

    restored = RuntimeSourceStateStore(path).load()

    assert restored == RuntimeSourceState()


def test_build_runtime_api_client_uses_api_key_auth_from_config():
    from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client

    client = build_runtime_api_client(
        app_config={
            "tldw_api": {
                "base_url": "https://example.com/api/",
                "api_key": "secret-key",
            }
        }
    )

    assert client.base_url == "https://example.com/api"
    assert client.token == "secret-key"
    assert client.bearer_token is None


def test_build_runtime_api_client_supports_explicit_custom_token_overrides():
    from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client

    client = build_runtime_api_client(
        app_config={"tldw_api": {"base_url": "https://example.com/api/"}},
        endpoint_url="https://override.example.com/v1/",
        auth_method="custom_token",
        auth_token="bearer-secret",
    )

    assert client.base_url == "https://override.example.com/v1"
    assert client.token is None
    assert client.bearer_token == "bearer-secret"


def test_config_client_provider_builds_legacy_client_lazily():
    from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client_provider_from_config

    provider = build_runtime_api_client_provider_from_config(
        {"tldw_api": {"base_url": "https://example.test", "api_key": "secret"}}
    )

    assert provider._cached_client is None

    first = provider.build_client()
    second = provider.build_client()

    assert first is second
    assert first.base_url == "https://example.test"
    assert first.token == "secret"


def test_config_client_provider_preserves_legacy_config_alias_and_bearer_auth():
    from tldw_chatbook.runtime_policy.bootstrap import (
        build_runtime_api_client_from_config,
        build_runtime_api_client_provider_from_config,
    )

    app_config = {
        "tldw_api": {
            "url": "https://Alias.Example.COM:8443/api/",
            "auth_mode": "bearer",
            "bearer_token": "legacy-bearer",
        }
    }
    provider = build_runtime_api_client_provider_from_config(app_config)

    client = provider.build_client()
    expected_client = build_runtime_api_client_from_config(app_config)

    assert client.base_url == expected_client.base_url
    assert client.token == expected_client.token
    assert client.bearer_token == expected_client.bearer_token


def test_config_client_provider_repr_redacts_config_secrets():
    from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client_provider_from_config

    provider = build_runtime_api_client_provider_from_config(
        {"tldw_api": {"base_url": "https://example.test", "api_key": "secret"}}
    )

    assert "secret" not in repr(provider)
    assert "api_key" not in repr(provider)
    assert "redacted" in repr(provider)


def test_build_server_chatbook_service_wraps_authoritative_client_builder():
    from tldw_chatbook.runtime_policy.bootstrap import build_server_chatbook_service

    service = build_server_chatbook_service(
        app_config={
            "tldw_api": {
                "base_url": "https://example.com/api/",
                "api_key": "secret-key",
            }
        }
    )

    assert service.client is not None
    assert service.client.base_url == "https://example.com/api"
    assert service.client.token == "secret-key"


def test_build_server_chatbook_service_can_return_disconnected_service_when_unconfigured():
    from tldw_chatbook.runtime_policy.bootstrap import build_server_chatbook_service

    policy_enforcer = object()

    service = build_server_chatbook_service(
        app_config={},
        policy_enforcer=policy_enforcer,
        allow_unconfigured=True,
    )

    assert service.client is None
    assert service.policy_enforcer is policy_enforcer


def test_load_runtime_policy_for_app_derives_and_persists_authoritative_server_binding_from_app_config(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    app_like = _make_app_like()

    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "local"
    assert context.state.active_server_id == "https://example.com:8443/api"
    assert context.state.server_configured is True
    assert context.state.last_known_server_label == "example.com:8443"
    assert store.load() == context.state
    assert app_like.current_runtime_backend == "local"
    assert app_like.runtime_backend == "local"


def test_load_runtime_policy_for_app_supports_legacy_url_alias_and_provider_resolution(
    tmp_path,
):
    from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
    from tldw_chatbook.runtime_policy.server_context import RuntimeServerContextProvider
    from tldw_chatbook.runtime_policy.server_credentials import InMemoryServerCredentialStore
    from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    app_config = {
        "tldw_api": {
            "url": "https://Alias.Example.COM:8443/api/",
            "auth_mode": "bearer",
            "bearer_token": "legacy-bearer",
        }
    }
    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    store.save(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://old.example.com/api",
            server_configured=True,
            last_known_server_label="old.example.com",
        )
    )
    app_like = SimpleNamespace(app_config=app_config, app_state=AppState())

    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "server"
    assert context.state.active_server_id == "https://alias.example.com:8443/api"
    assert context.state.server_configured is True
    assert context.state.last_known_server_label == "alias.example.com:8443"
    assert store.load() == context.state

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target = target_store.upsert_legacy_config_target(app_config)
    assert target is not None
    assert target.server_id == context.state.active_server_id

    provider = RuntimeServerContextProvider(
        runtime_context=context,
        target_store=target_store,
        credential_store=InMemoryServerCredentialStore(),
        app_config=app_config,
    )

    active_context = provider.get_active_context()

    assert active_context.active_server_id == "https://alias.example.com:8443/api"
    assert active_context.base_url == "https://alias.example.com:8443/api"
    assert active_context.auth_method == "bearer"
    assert active_context.auth_token == "legacy-bearer"
    assert active_context.credential_source == "credential_store:bearer_token"


def test_wire_server_context_provider_exposes_provider_and_credential_store(tmp_path, monkeypatch):
    from tldw_chatbook.app import TldwCli

    class FakeKeyringServerCredentialStore:
        pass

    monkeypatch.setattr("tldw_chatbook.app.get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "tldw_chatbook.app.KeyringServerCredentialStore",
        FakeKeyringServerCredentialStore,
    )
    app_like = SimpleNamespace(
        app_config={"tldw_api": {"base_url": "https://example.com/api/"}},
        runtime_policy=SimpleNamespace(state=RuntimeSourceState()),
    )

    TldwCli._wire_server_context_provider(app_like)

    assert isinstance(app_like.server_credential_store, FakeKeyringServerCredentialStore)
    assert app_like.server_context_provider is not None
    assert app_like.server_context_provider.runtime_context is app_like.runtime_policy
    assert app_like.server_context_provider.target_store is app_like.unified_mcp_target_store
    assert app_like.server_context_provider.credential_store is app_like.server_credential_store


def test_auth_scope_updates_and_clears_legacy_imported_effective_bearer_token(tmp_path):
    from tldw_chatbook.Auth_Account_Interop.auth_account_scope_service import AuthAccountScopeService
    from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
    from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
    from tldw_chatbook.runtime_policy.server_context import RuntimeServerContextProvider
    from tldw_chatbook.runtime_policy.server_credentials import (
        SERVER_CREDENTIAL_BEARER_TOKEN,
        InMemoryServerCredentialStore,
    )
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    app_config = {
        "tldw_api": {
            "url": "https://Alias.Example.COM:8443/api/",
            "auth_mode": "bearer",
            "bearer_token": "legacy-bearer",
        }
    }
    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    store.save(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://old.example.com/api",
            server_configured=True,
            last_known_server_label="old.example.com",
        )
    )
    app_like = SimpleNamespace(app_config=app_config, app_state=AppState())
    context = load_runtime_policy_for_app(app_like, store=store)
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.upsert_legacy_config_target(app_config)
    credential_store = InMemoryServerCredentialStore()
    provider = RuntimeServerContextProvider(
        runtime_context=context,
        target_store=target_store,
        credential_store=credential_store,
        app_config=app_config,
    )
    scope = AuthAccountScopeService(server_context_provider=provider)

    assert provider.get_active_context().auth_token == "legacy-bearer"
    assert credential_store.get_secret(
        context.state.active_server_id,
        SERVER_CREDENTIAL_BEARER_TOKEN,
    ) == "legacy-bearer"

    scope.store_login_tokens(access_token="access-1", refresh_token="refresh-1")

    assert provider.get_active_context().auth_token == "access-1"
    assert credential_store.get_secret(
        context.state.active_server_id,
        SERVER_CREDENTIAL_BEARER_TOKEN,
    ) == "access-1"

    scope.clear_login_tokens()

    assert credential_store.get_secret(
        context.state.active_server_id,
        SERVER_CREDENTIAL_BEARER_TOKEN,
    ) is None


def test_load_runtime_policy_for_app_rebinds_persisted_runtime_state_to_configured_server_identity(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    store.save(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://old.example.com/api",
            server_configured=True,
            last_known_server_label="old.example.com",
        )
    )
    app_like = _make_app_like(base_url="https://new.example.com/v1/")

    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "server"
    assert context.state.active_server_id == "https://new.example.com/v1"
    assert context.state.server_configured is True
    assert context.state.last_known_server_label == "new.example.com"
    assert store.load() == context.state


def test_load_runtime_policy_for_app_clears_stale_capability_state_when_server_identity_changes(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    store.save(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://old.example.com/api",
            server_configured=True,
            server_reachability="reachable",
            server_reachability_checked_at=datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc),
            server_auth_state="authenticated",
            server_auth_checked_at=datetime(2026, 4, 21, 12, 5, tzinfo=timezone.utc),
            last_known_server_label="old.example.com",
        )
    )
    app_like = _make_app_like(base_url="https://new.example.com/v1/")

    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "server"
    assert context.state.active_server_id == "https://new.example.com/v1"
    assert context.state.server_reachability == "unknown"
    assert context.state.server_reachability_checked_at is None
    assert context.state.server_auth_state == "unknown"
    assert context.state.server_auth_checked_at is None
    assert store.load() == context.state


def test_set_authoritative_runtime_source_clears_probe_state_on_server_identity_change(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import (
        RuntimePolicyContext,
        set_authoritative_runtime_source,
    )
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    app_like = _make_app_like(base_url="https://new.example.com/v1/")
    old_state = RuntimeSourceState(
        active_source="server",
        active_server_id="https://old.example.com/api",
        server_configured=True,
        server_reachability="reachable",
        server_reachability_checked_at=datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc),
        server_auth_state="authenticated",
        server_auth_checked_at=datetime(2026, 4, 21, 12, 5, tzinfo=timezone.utc),
        last_known_server_label="old.example.com",
    )
    app_like.runtime_policy = RuntimePolicyContext(state=old_state, store=store)

    updated_state = set_authoritative_runtime_source(app_like, "server")

    assert updated_state.active_source == "server"
    assert updated_state.active_server_id == "https://new.example.com/v1"
    assert updated_state.server_configured is True
    assert updated_state.server_reachability == "unknown"
    assert updated_state.server_reachability_checked_at is None
    assert updated_state.server_auth_state == "unknown"
    assert updated_state.server_auth_checked_at is None
    assert store.load() == updated_state


def test_load_runtime_policy_for_app_downgrades_stale_server_mode_when_server_config_is_missing(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    store.save(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://server.example.com/api",
            server_configured=True,
            last_known_server_label="server.example.com",
        )
    )
    app_like = SimpleNamespace(
        app_config={},
        app_state=AppState(),
    )

    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "local"
    assert context.state.active_server_id is None
    assert context.state.server_configured is False
    assert store.load() == context.state


def test_reconcile_saved_screen_state_drops_wrong_server_snapshot_against_bootstrapped_authority(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import (
        load_runtime_policy_for_app,
        reconcile_saved_screen_state,
    )
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    app_like = _make_app_like(base_url="https://server-b.example.com/api/")
    context = load_runtime_policy_for_app(app_like, store=store)
    context.state = RuntimeSourceState(
        active_source="server",
        active_server_id=context.state.active_server_id,
        server_configured=context.state.server_configured,
        last_known_server_label=context.state.last_known_server_label,
    )
    saved_state = {
        "chat_state": {
            "tabs": [
                {
                    "tab_id": "tab-1",
                    "title": "Server Chat",
                    "runtime_backend": "server",
                }
            ]
        },
        "runtime_policy_snapshot": {
            "active_source": "server",
            "active_server_id": "https://server-a.example.com/api",
        },
    }

    assert reconcile_saved_screen_state(saved_state, context.state) is None


@pytest.mark.asyncio
async def test_handle_runtime_backend_changed_updates_authoritative_runtime_policy_and_persists(
    tmp_path,
):
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    forwarded = []

    async def screen_callback(runtime_backend: str) -> None:
        forwarded.append(runtime_backend)

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    app_like = _make_app_like(base_url="https://Server.EXAMPLE.com/api/")
    app_like.screen = SimpleNamespace(handle_runtime_backend_changed=screen_callback)
    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "local"
    assert context.state.active_server_id == "https://server.example.com/api"

    await TldwCli.handle_runtime_backend_changed(app_like, "server")
    await TldwCli.handle_runtime_backend_changed(app_like, "local")

    persisted = store.load()

    assert app_like.current_runtime_backend == "local"
    assert app_like.runtime_backend == "local"
    assert app_like.runtime_policy.state.active_source == "local"
    assert persisted.active_source == "local"
    assert persisted.active_server_id == "https://server.example.com/api"
    assert persisted.server_configured is True
    assert persisted.last_known_server_label == "server.example.com"
    assert forwarded == ["server", "local"]


@pytest.mark.asyncio
async def test_handle_runtime_backend_changed_forwards_resolved_authoritative_backend_to_screen(
    tmp_path,
):
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    forwarded = []

    async def screen_callback(runtime_backend: str) -> None:
        forwarded.append(runtime_backend)

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    app_like = SimpleNamespace(
        app_config={},
        app_state=AppState(),
        screen=SimpleNamespace(handle_runtime_backend_changed=screen_callback),
    )
    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "local"

    await TldwCli.handle_runtime_backend_changed(app_like, "server")

    assert app_like.runtime_policy.state.active_source == "local"
    assert forwarded == ["local"]
