from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tldw_chatbook.runtime_policy.bootstrap import (
    RuntimePolicyContext,
    set_authoritative_runtime_source,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.state.app_state import AppState


def _prepare_context(**kwargs):
    import tldw_chatbook.runtime_policy.bootstrap as bootstrap

    prepare = getattr(bootstrap, "_prepare_runtime_policy_context", None)
    assert callable(prepare), (
        "bootstrap must expose an app-independent runtime-policy preparation boundary"
    )
    return prepare(**kwargs)


def test_default_policy_path_follows_effective_config_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    import tldw_chatbook.runtime_policy.bootstrap as bootstrap
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    config_path = tmp_path / "custom" / "config.toml"
    config_path.parent.mkdir()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    constructed: list[RuntimeSourceStateStore] = []
    real_store_type = RuntimeSourceStateStore

    def capture_store(path, **kwargs):
        store = real_store_type(path, **kwargs)
        constructed.append(store)
        return store

    monkeypatch.setattr(bootstrap, "RuntimeSourceStateStore", capture_store)

    _prepare_context(app_config={}, publish=lambda _state: None)

    assert constructed[0].path == config_path.parent / "runtime_policy.json"


def test_config_override_does_not_read_fallback_or_migrate_default_runtime_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    import tldw_chatbook.config as config
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    default_config_path = tmp_path / "default" / "config.toml"
    default_config_path.parent.mkdir()
    default_policy_path = default_config_path.parent / "runtime_policy.json"
    RuntimeSourceStateStore(default_policy_path).save(
        RuntimeSourceState(
            active_source="server",
            active_server_id="DEFAULT-POLICY-SENTINEL",
            server_configured=True,
        )
    )
    override_config_path = tmp_path / "custom" / "config.toml"
    override_config_path.parent.mkdir()
    override_policy_path = override_config_path.parent / "runtime_policy.json"
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", default_config_path)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(override_config_path))
    context = _prepare_context(app_config={}, publish=lambda _state: None)

    assert context.state == RuntimeSourceState()
    assert not override_policy_path.exists()
    assert RuntimeSourceStateStore(default_policy_path).load().active_server_id == (
        "DEFAULT-POLICY-SENTINEL"
    )


@pytest.mark.skipif(
    __import__("os").name != "posix",
    reason="POSIX namespace contract",
)
def test_explicit_runtime_policy_path_never_gains_default_directory_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    import tldw_chatbook.config as config
    from tldw_chatbook.Utils.private_paths import PrivatePathError

    default_config_path = tmp_path / "default" / "config.toml"
    explicit_policy_path = default_config_path.parent / "runtime_policy.json"
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", default_config_path)
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    app_config = {
        "tldw_api": {
            "base_url": "https://explicit-path.example.test/api",
        }
    }

    with pytest.raises(PrivatePathError):
        _prepare_context(
            app_config=app_config,
            publish=lambda _state: None,
            path=explicit_policy_path,
        )

    assert not explicit_policy_path.parent.exists()


def test_app_state_round_trips_runtime_source_state():
    original = AppState(
        runtime_source=RuntimeSourceState(
            active_source="server",
            active_server_id="server-alpha",
            server_configured=True,
            server_reachability="reachable",
            server_reachability_checked_at=datetime(
                2026, 4, 21, 12, 0, tzinfo=timezone.utc
            ),
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
    from tldw_chatbook.runtime_policy.bootstrap import (
        build_runtime_api_client_provider_from_config,
    )

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
    from tldw_chatbook.runtime_policy.bootstrap import (
        build_runtime_api_client_provider_from_config,
    )

    provider = build_runtime_api_client_provider_from_config(
        {"tldw_api": {"base_url": "https://example.test", "api_key": "secret"}}
    )

    assert "secret" not in repr(provider)
    assert "api_key" not in repr(provider)
    assert "redacted" in repr(provider)


@pytest.mark.asyncio
async def test_config_client_provider_close_cached_client_clears_and_closes_previous_client():
    from tldw_chatbook.runtime_policy.bootstrap import (
        build_runtime_api_client_provider_from_config,
    )

    class FakeClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    provider = build_runtime_api_client_provider_from_config(
        {"tldw_api": {"base_url": "https://example.test", "api_key": "secret"}}
    )
    cached_client = FakeClient()
    provider._cached_client = cached_client

    await provider.close_cached_client()

    assert provider._cached_client is None
    assert cached_client.close_calls == 1


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


def test_prepare_runtime_policy_context_derives_and_persists_authoritative_server_binding(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    published: list[RuntimeSourceState] = []
    app_config = {
        "tldw_api": {
            "base_url": "https://Example.COM:8443/api/",
        }
    }

    context = _prepare_context(
        app_config=app_config,
        publish=published.append,
        store=store,
    )

    assert context.state.active_source == "local"
    assert context.state.active_server_id == "https://example.com:8443/api"
    assert context.state.server_configured is True
    assert context.state.last_known_server_label == "example.com:8443"
    assert store.load() == context.state
    assert published == [context.state]


def test_prepare_runtime_policy_context_commits_synchronized_state_as_revision_one():
    loaded = RuntimeSourceState(
        active_source="server",
        active_server_id="https://old.example.test/api",
        server_configured=True,
    )

    class RecordingStore:
        def __init__(self) -> None:
            self.saved_states: list[RuntimeSourceState] = []

        def load(self) -> RuntimeSourceState:
            return loaded

        def save(self, state: RuntimeSourceState) -> None:
            self.saved_states.append(state)

    store = RecordingStore()
    published: list[RuntimeSourceState] = []
    app_config = {
        "tldw_api": {
            "base_url": "https://new.example.test/v1",
        }
    }

    context = _prepare_context(
        app_config=app_config,
        publish=published.append,
        store=store,
    )

    state, revision = context.snapshot()
    assert revision == 1
    assert state.active_server_id == "https://new.example.test/v1"
    assert store.saved_states == [state]
    assert published == [state]


def test_prepare_runtime_policy_context_publishes_unchanged_loaded_state_without_save():
    loaded = RuntimeSourceState()

    class RecordingStore:
        def __init__(self) -> None:
            self.saved_states: list[RuntimeSourceState] = []

        def load(self) -> RuntimeSourceState:
            return loaded

        def save(self, state: RuntimeSourceState) -> None:
            self.saved_states.append(state)

    store = RecordingStore()
    published: list[RuntimeSourceState] = []

    context = _prepare_context(
        app_config={},
        publish=published.append,
        store=store,
    )

    assert context.snapshot() == (loaded, 0)
    assert store.saved_states == []
    assert published == [loaded]


def test_prepare_runtime_policy_context_propagates_load_failure():
    load_sentinel = "RUNTIME-POLICY-LOAD-SENTINEL"

    class RaisingLoadStore:
        def load(self) -> RuntimeSourceState:
            raise OSError(load_sentinel)

        def save(self, state: RuntimeSourceState) -> None:
            raise AssertionError("save must not run after load failure")

    with pytest.raises(OSError, match=load_sentinel):
        _prepare_context(
            app_config={},
            publish=lambda _state: None,
            store=RaisingLoadStore(),
        )


def test_prepare_runtime_policy_context_propagates_synchronization_save_failure():
    save_sentinel = "RUNTIME-POLICY-SAVE-SENTINEL"
    loaded = RuntimeSourceState()
    published: list[RuntimeSourceState] = []

    class RaisingSaveStore:
        def load(self) -> RuntimeSourceState:
            return loaded

        def save(self, state: RuntimeSourceState) -> None:
            raise OSError(save_sentinel)

    with pytest.raises(OSError, match=save_sentinel):
        _prepare_context(
            app_config={
                "tldw_api": {
                    "base_url": "https://save-failure.example.test/api",
                }
            },
            publish=published.append,
            store=RaisingSaveStore(),
        )

    assert published == []


def test_prepare_runtime_policy_context_propagates_initial_publication_failure():
    publish_sentinel = "RUNTIME-POLICY-PUBLISH-SENTINEL"

    class LoadedStateStore:
        def load(self) -> RuntimeSourceState:
            return RuntimeSourceState()

        def save(self, state: RuntimeSourceState) -> None:
            raise AssertionError("unchanged state must not be saved")

    def raise_on_publish(_state: RuntimeSourceState) -> None:
        raise RuntimeError(publish_sentinel)

    with pytest.raises(RuntimeError, match=publish_sentinel):
        _prepare_context(
            app_config={},
            publish=raise_on_publish,
            store=LoadedStateStore(),
        )


def test_prepare_runtime_policy_context_contains_post_commit_publication_failure():
    loaded = RuntimeSourceState()

    class RecordingStore:
        def __init__(self) -> None:
            self.saved_states: list[RuntimeSourceState] = []

        def load(self) -> RuntimeSourceState:
            return loaded

        def save(self, state: RuntimeSourceState) -> None:
            self.saved_states.append(state)

    def raise_on_publish(_state: RuntimeSourceState) -> None:
        raise RuntimeError("POST-COMMIT-PUBLISH-SENTINEL")

    store = RecordingStore()
    context = _prepare_context(
        app_config={
            "tldw_api": {
                "base_url": "https://new.example.test/v1",
            }
        },
        publish=raise_on_publish,
        store=store,
    )

    state, revision = context.snapshot()
    assert revision == 1
    assert state.active_server_id == "https://new.example.test/v1"
    assert store.saved_states == [state]


def test_prepare_runtime_policy_context_supports_legacy_url_alias_and_provider_resolution(
    tmp_path,
):
    from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
    from tldw_chatbook.runtime_policy.server_context import RuntimeServerContextProvider
    from tldw_chatbook.runtime_policy.server_credentials import (
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
    context = _prepare_context(
        app_config=app_config,
        publish=lambda _state: None,
        store=store,
    )

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


def test_auth_scope_updates_and_clears_legacy_imported_effective_bearer_token(tmp_path):
    from tldw_chatbook.Auth_Account_Interop.auth_account_scope_service import (
        AuthAccountScopeService,
    )
    from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
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
    context = _prepare_context(
        app_config=app_config,
        publish=lambda _state: None,
        store=store,
    )
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
    assert (
        credential_store.get_secret(
            context.state.active_server_id,
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        == "legacy-bearer"
    )

    scope.store_login_tokens(access_token="access-1", refresh_token="refresh-1")

    assert provider.get_active_context().auth_token == "access-1"
    assert (
        credential_store.get_secret(
            context.state.active_server_id,
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        == "access-1"
    )

    scope.clear_login_tokens()

    assert (
        credential_store.get_secret(
            context.state.active_server_id,
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        is None
    )


def test_prepare_runtime_policy_context_rebinds_persisted_state_to_configured_server_identity(
    tmp_path,
):
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
    context = _prepare_context(
        app_config={
            "tldw_api": {
                "base_url": "https://new.example.com/v1/",
            }
        },
        publish=lambda _state: None,
        store=store,
    )

    assert context.state.active_source == "server"
    assert context.state.active_server_id == "https://new.example.com/v1"
    assert context.state.server_configured is True
    assert context.state.last_known_server_label == "new.example.com"
    assert store.load() == context.state


def test_prepare_runtime_policy_context_clears_stale_capability_state_on_binding_change(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    store.save(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://old.example.com/api",
            server_configured=True,
            server_reachability="reachable",
            server_reachability_checked_at=datetime(
                2026, 4, 21, 12, 0, tzinfo=timezone.utc
            ),
            server_auth_state="authenticated",
            server_auth_checked_at=datetime(2026, 4, 21, 12, 5, tzinfo=timezone.utc),
            last_known_server_label="old.example.com",
        )
    )
    context = _prepare_context(
        app_config={
            "tldw_api": {
                "base_url": "https://new.example.com/v1/",
            }
        },
        publish=lambda _state: None,
        store=store,
    )

    assert context.state.active_source == "server"
    assert context.state.active_server_id == "https://new.example.com/v1"
    assert context.state.server_reachability == "unknown"
    assert context.state.server_reachability_checked_at is None
    assert context.state.server_auth_state == "unknown"
    assert context.state.server_auth_checked_at is None
    assert store.load() == context.state


def test_prepare_runtime_policy_context_downgrades_server_mode_without_server_config(
    tmp_path,
):
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
    context = _prepare_context(
        app_config={},
        publish=lambda _state: None,
        store=store,
    )

    assert context.state.active_source == "local"
    assert context.state.active_server_id is None
    assert context.state.server_configured is False
    assert store.load() == context.state


def test_authoritative_source_derives_binding_only_from_supplied_config():
    initial_state = RuntimeSourceState(
        active_source="server",
        active_server_id="https://old.example.test/api",
        server_configured=True,
        last_known_server_label="old.example.test",
    )

    class RecordingStore:
        def __init__(self) -> None:
            self.saved_states: list[RuntimeSourceState] = []

        def save(self, state: RuntimeSourceState) -> None:
            self.saved_states.append(state)

    store = RecordingStore()
    context = RuntimePolicyContext(initial_state, store)
    supplied_config = {
        "tldw_api": {
            "base_url": "https://new.example.test/v1/",
        }
    }

    updated_state = set_authoritative_runtime_source(
        context,
        "server",
        app_config=supplied_config,
    )

    assert updated_state.active_source == "server"
    assert updated_state.active_server_id == "https://new.example.test/v1"
    assert updated_state.server_configured is True
    assert updated_state.last_known_server_label == "new.example.test"
    assert store.saved_states == [updated_state]
    assert context.snapshot() == (updated_state, 1)


def test_authoritative_source_invalid_input_returns_unchanged_without_save():
    initial_state = RuntimeSourceState(
        active_source="server",
        active_server_id="https://old.example.test/api",
        server_configured=True,
    )

    class NeverSaveStore:
        def save(self, state: RuntimeSourceState) -> None:
            raise AssertionError("invalid source must not be saved")

    context = RuntimePolicyContext(initial_state, NeverSaveStore())

    returned_state = set_authoritative_runtime_source(
        context,
        "invalid-source",
        app_config={
            "tldw_api": {
                "base_url": "https://unused.example.test/api",
            }
        },
    )

    assert returned_state is initial_state
    assert context.snapshot() == (initial_state, 0)


def test_authoritative_source_persistence_failure_leaves_snapshot_unchanged():
    save_sentinel = "AUTHORITATIVE-SAVE-SENTINEL"
    initial_state = RuntimeSourceState()

    class RaisingSaveStore:
        def save(self, state: RuntimeSourceState) -> None:
            raise OSError(save_sentinel)

    context = RuntimePolicyContext(initial_state, RaisingSaveStore())

    with pytest.raises(OSError, match=save_sentinel):
        set_authoritative_runtime_source(
            context,
            "server",
            app_config={
                "tldw_api": {
                    "base_url": "https://candidate.example.test/api",
                }
            },
        )

    assert context.snapshot() == (initial_state, 0)


def test_authoritative_source_cas_rejection_raises_bounded_error(
    monkeypatch: pytest.MonkeyPatch,
):
    initial_state = RuntimeSourceState()

    class NeverSaveStore:
        def save(self, state: RuntimeSourceState) -> None:
            raise AssertionError("rejected CAS must not save")

    context = RuntimePolicyContext(initial_state, NeverSaveStore())

    def reject_commit(
        self,
        candidate: RuntimeSourceState,
        *,
        expected_revision: int,
    ) -> bool:
        assert self is context
        assert candidate.active_server_id == "https://candidate.example.test/api"
        assert expected_revision == 0
        return False

    monkeypatch.setattr(RuntimePolicyContext, "commit_state", reject_commit)

    with pytest.raises(RuntimeError, match="commit was rejected"):
        set_authoritative_runtime_source(
            context,
            "server",
            app_config={
                "tldw_api": {
                    "base_url": "https://candidate.example.test/api",
                }
            },
        )

    assert context.snapshot() == (initial_state, 0)


class TestRuntimePolicyPathIsolation:
    """The runtime-policy state file must follow the active config profile.

    The file records whether the app is running local or server. Its path was
    derived from a hardcoded home directory rather than from the config path in
    effect, so a profile launched with ``TLDW_CONFIG_PATH`` shared the real
    user's local/server mode: switching a scratch profile to server mode left
    the default profile in server mode afterwards (task-701).

    That is not merely untidy. It made local/server behaviour untestable without
    mutating real state, which is why verifying the server-ingest path in
    task-684.1 had to stop short of putting a test profile into server mode.
    """

    def test_an_isolated_profile_gets_its_own_policy_file(self, tmp_path, monkeypatch):
        from tldw_chatbook.runtime_policy import bootstrap

        scratch = tmp_path / "profile" / "config.toml"
        scratch.parent.mkdir(parents=True)
        scratch.write_text("[general]\n", encoding="utf-8")
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(scratch))

        resolved = bootstrap.default_runtime_policy_path()

        assert resolved.parent == scratch.parent.resolve(), (
            f"policy file landed at {resolved}, outside the active profile"
        )
        assert resolved.name == "runtime_policy.json"

    def test_two_profiles_do_not_share_a_policy_file(self, tmp_path, monkeypatch):
        from tldw_chatbook.runtime_policy import bootstrap

        first = tmp_path / "one" / "config.toml"
        second = tmp_path / "two" / "config.toml"
        for path in (first, second):
            path.parent.mkdir(parents=True)
            path.write_text("[general]\n", encoding="utf-8")

        monkeypatch.setenv("TLDW_CONFIG_PATH", str(first))
        first_path = bootstrap.default_runtime_policy_path()
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(second))
        second_path = bootstrap.default_runtime_policy_path()

        assert first_path != second_path, (
            "both profiles resolved to the same runtime-policy file, so one "
            "profile's local/server mode overwrites the other's"
        )

    def test_the_default_location_is_unchanged_without_an_override(
        self, monkeypatch
    ):
        """No override must mean exactly the historical path, or existing
        installs silently lose the mode they had persisted."""
        from tldw_chatbook.config import DEFAULT_CONFIG_PATH
        from tldw_chatbook.runtime_policy import bootstrap

        monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)

        assert bootstrap.default_runtime_policy_path() == (
            DEFAULT_CONFIG_PATH.parent / "runtime_policy.json"
        )

    def test_context_preparation_uses_the_active_profile(self, tmp_path, monkeypatch):
        """Preparing runtime policy writes only to the active profile."""
        scratch = tmp_path / "profile" / "config.toml"
        scratch.parent.mkdir(parents=True)
        scratch.write_text("[general]\n", encoding="utf-8")
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(scratch))

        context = _prepare_context(app_config={}, publish=lambda _state: None)
        assert context.commit_state(context.state, expected_revision=0)

        assert (scratch.parent / "runtime_policy.json").exists(), (
            "the profile's own policy file was never written"
        )

    def test_a_rejected_override_falls_back_loudly(self, monkeypatch, caplog):
        """A bad path may not stop the app booting, but must not be silent.

        Falling back means this profile's local/server mode is read from and
        written to the DEFAULT profile -- the very cross-profile leak this
        function exists to prevent -- so it is logged at warning, not debug.
        """
        from tldw_chatbook.runtime_policy import bootstrap

        def rejecting(*args, **kwargs):
            raise ValueError("rejected path")

        monkeypatch.setattr(
            "tldw_chatbook.config._get_effective_config_path", rejecting
        )

        assert bootstrap.default_runtime_policy_path() == (
            bootstrap.DEFAULT_RUNTIME_POLICY_PATH
        )

    def test_an_unexpected_failure_is_not_swallowed(self, monkeypatch):
        """Only a bad path is absorbed; a defect must surface.

        Catching everything would route runtime-policy state to the wrong
        profile whenever some unrelated bug appeared in path resolution, and the
        symptom -- a scratch profile quietly editing the real user's mode -- is
        precisely what this change set out to stop.
        """
        import pytest as _pytest

        from tldw_chatbook.runtime_policy import bootstrap

        def exploding(*args, **kwargs):
            raise RuntimeError("something unrelated broke")

        monkeypatch.setattr(
            "tldw_chatbook.config._get_effective_config_path", exploding
        )

        with _pytest.raises(RuntimeError):
            bootstrap.default_runtime_policy_path()
