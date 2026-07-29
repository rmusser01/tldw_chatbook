from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import get_args

import pytest

import tldw_chatbook.runtime_policy.server_context as server_context_module
from tldw_chatbook.MCP.server_target_store import (
    AuthorityScopeUnavailable,
    ConfiguredServerTargetStore,
)
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget
from tldw_chatbook.runtime_policy.bootstrap import RuntimePolicyContext
from tldw_chatbook.runtime_policy.server_credentials import (
    CredentialStoreUnavailable,
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_API_KEY,
    SERVER_CREDENTIAL_BEARER_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
    InMemoryServerCredentialStore,
    UnavailableServerCredentialStore,
)
from tldw_chatbook.runtime_policy.server_context import (
    RuntimeServerContextProvider,
    ServerCredentialsUnavailable,
    ServerContextUnavailable,
)
from tldw_chatbook.runtime_policy.types import (
    RuntimeSourceState,
    SERVER_CONTEXT_FAILURE_REASON_CODES,
    ServerContextFailureReason,
)
from tldw_chatbook.tldw_api.auth_user_schemas import UserProfileResponse


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


class LookupFailingCredentialStore:
    def set_secret(self, server_id: str, purpose: str, secret: str) -> None:
        return None

    def get_secret(self, server_id: str, purpose: str) -> str | None:
        raise RuntimeError("lookup unavailable")

    def delete_secret(self, server_id: str, purpose: str) -> None:
        return None

    def clear_server(self, server_id: str) -> None:
        return None

    def clear_all(self) -> None:
        return None


def _runtime_context(
    *,
    active_source: str = "server",
    active_server_id: str | None = "https://server.example.com/api",
    server_configured: bool = True,
    runtime_store: SavingRuntimeStore | None = None,
) -> RuntimePolicyContext:
    return RuntimePolicyContext(
        state=RuntimeSourceState(
            active_source=active_source,
            active_server_id=active_server_id,
            server_configured=server_configured,
            last_known_server_label="Server",
        ),
        store=runtime_store if runtime_store is not None else SavingRuntimeStore(),
    )


def _commit_runtime_state(
    runtime_context: RuntimePolicyContext,
    state: RuntimeSourceState,
) -> None:
    _, revision = runtime_context.snapshot()
    assert runtime_context.commit_state(state, expected_revision=revision)


def _target_store(
    tmp_path, targets: list[ConfiguredServerTarget] | None = None
) -> ConfiguredServerTargetStore:
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


_AUTHORITY_SCOPE = "123e4567-e89b-42d3-a456-426614174000"
_SECOND_AUTHORITY_SCOPE = "123e4567-e89b-42d3-b456-426614174001"
_AUTHORITY_SERVER_ID = "https://server.example.com/api"


def _identity_response(user_id) -> UserProfileResponse:
    return UserProfileResponse(
        profile_version="profile-v1",
        catalog_version="catalog-v1",
        user={"id": user_id},
    )


class IdentityClient:
    def __init__(self, response=None, *, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.profile_calls: list[dict] = []
        self.close_calls = 0

    async def get_current_user_profile(self, **kwargs):
        self.profile_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    async def close(self) -> None:
        self.close_calls += 1


class GatedIdentityClient(IdentityClient):
    def __init__(self, response) -> None:
        super().__init__(response)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_current_user_profile(self, **kwargs):
        self.profile_calls.append(kwargs)
        self.started.set()
        await self.release.wait()
        return self.response


def _authority_provider(
    tmp_path,
    monkeypatch,
    clients: list[IdentityClient],
    *,
    runtime_context: RuntimePolicyContext | None = None,
    credential_store: InMemoryServerCredentialStore | None = None,
    target: ConfiguredServerTarget | None = None,
):
    runtime_context = runtime_context or _runtime_context()
    credential_store = credential_store or InMemoryServerCredentialStore()
    active_server_id = runtime_context.state.active_server_id
    assert active_server_id is not None
    if (
        credential_store.get_secret(
            active_server_id,
            SERVER_CREDENTIAL_ACCESS_TOKEN,
        )
        is None
    ):
        credential_store.set_secret(
            active_server_id,
            SERVER_CREDENTIAL_ACCESS_TOKEN,
            "access-1",
        )
    target = target or ConfiguredServerTarget(
        server_id=active_server_id,
        label="Primary",
        base_url="https://server.example.com/api",
        auth_mode="bearer",
        is_default=True,
        authority_scope_id=_AUTHORITY_SCOPE,
    )
    target_store = _target_store(tmp_path, [target])
    remaining_clients = iter(clients)
    build_calls: list[dict] = []

    def build_client(**kwargs):
        build_calls.append(kwargs)
        return next(remaining_clients)

    monkeypatch.setattr(
        server_context_module,
        "build_runtime_api_client",
        build_client,
    )
    provider = RuntimeServerContextProvider(
        runtime_context=runtime_context,
        target_store=target_store,
        credential_store=credential_store,
        app_config={},
    )
    return provider, target_store, credential_store, build_calls


async def _resolve_authority(provider: RuntimeServerContextProvider) -> str:
    return await provider.resolve_character_authority_id(
        expected_server_id=_AUTHORITY_SERVER_ID
    )


def test_server_user_character_authority_encoding_matches_independent_vector():
    authority_id = server_context_module.encode_server_user_character_authority(
        _AUTHORITY_SCOPE,
        42,
    )

    assert (
        authority_id == "server-user-v1:"
        "24e98cf244c628ca3fa628cf0d6a46e79a90864ef5864d3022db18415d5710ed"
    )
    assert len(authority_id) == 79
    assert authority_id.isascii()


@pytest.mark.parametrize(
    "scope",
    [
        "",
        "123E4567-E89B-42D3-A456-426614174000",
        "123e4567e89b42d3a456426614174000",
        "123e4567-e89b-12d3-a456-426614174000",
        "123e4567-e89b-42d3-a456-42661417400é",
        f" {_AUTHORITY_SCOPE}",
        123,
    ],
)
def test_server_user_character_authority_encoding_rejects_noncanonical_scope(scope):
    with pytest.raises(ValueError):
        server_context_module.encode_server_user_character_authority(scope, 1)


@pytest.mark.parametrize("user_id", [True, 0, -1, 2**63, "1"])
def test_server_user_character_authority_encoding_rejects_invalid_user_id(user_id):
    with pytest.raises(ValueError):
        server_context_module.encode_server_user_character_authority(
            _AUTHORITY_SCOPE,
            user_id,
        )


@pytest.mark.parametrize("user_id", [1, 2**63 - 1])
def test_server_user_character_authority_encoding_accepts_user_id_boundaries(user_id):
    authority_id = server_context_module.encode_server_user_character_authority(
        _AUTHORITY_SCOPE,
        user_id,
    )

    assert authority_id.startswith("server-user-v1:")
    assert len(authority_id) == 79
    assert authority_id[15:] == authority_id[15:].lower()


@pytest.mark.asyncio
async def test_character_authority_resolver_requests_identity_section_and_caches_same_context(
    tmp_path,
    monkeypatch,
):
    client = IdentityClient(_identity_response(42))
    provider, _, _, build_calls = _authority_provider(
        tmp_path,
        monkeypatch,
        [client],
    )

    first = await _resolve_authority(provider)
    second = await _resolve_authority(provider)

    assert first == server_context_module.encode_server_user_character_authority(
        _AUTHORITY_SCOPE,
        42,
    )
    assert second == first
    assert client.profile_calls == [{"sections": "identity"}]
    assert len(build_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected_server_id",
    [
        "",
        " https://server.example.com/api ",
        "https://other.example.com/api",
    ],
)
async def test_character_authority_resolver_rejects_expected_target_before_scope_or_network(
    tmp_path,
    monkeypatch,
    expected_server_id,
):
    client = IdentityClient(_identity_response(42))
    provider, target_store, _, build_calls = _authority_provider(
        tmp_path,
        monkeypatch,
        [client],
    )
    ensure_calls = 0
    real_ensure = target_store.ensure_authority_scope_id

    def counted_ensure(server_id):
        nonlocal ensure_calls
        ensure_calls += 1
        return real_ensure(server_id)

    monkeypatch.setattr(target_store, "ensure_authority_scope_id", counted_ensure)

    with pytest.raises(server_context_module.ServerIdentityUnavailable) as exc_info:
        await provider.resolve_character_authority_id(
            expected_server_id=expected_server_id
        )

    assert str(exc_info.value) == "Server identity unavailable."
    assert exc_info.value.reason_code == "server_identity_unavailable"
    assert exc_info.value.recoverable is True
    assert ensure_calls == 0
    assert build_calls == []
    assert client.profile_calls == []


@pytest.mark.asyncio
async def test_character_authority_resolver_uses_only_nested_user_id_from_mapping_fake(
    tmp_path,
    monkeypatch,
):
    client = IdentityClient({"id": 999, "user": {"id": 42}})
    provider, _, _, _ = _authority_provider(tmp_path, monkeypatch, [client])

    authority_id = await _resolve_authority(provider)

    assert authority_id == server_context_module.encode_server_user_character_authority(
        _AUTHORITY_SCOPE,
        42,
    )


@pytest.mark.asyncio
async def test_character_authority_resolver_rejects_ambiguous_mapping_and_attribute_response(
    tmp_path,
    monkeypatch,
):
    class AmbiguousResponse(dict):
        def __init__(self) -> None:
            super().__init__(user={"id": 42})
            self.user = {"id": 43}

    client = IdentityClient(AmbiguousResponse())
    provider, _, _, _ = _authority_provider(tmp_path, monkeypatch, [client])

    with pytest.raises(server_context_module.ServerIdentityUnavailable):
        await _resolve_authority(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        None,
        UserProfileResponse(
            profile_version="profile-v1",
            catalog_version="catalog-v1",
            user=None,
        ),
        UserProfileResponse(
            profile_version="profile-v1",
            catalog_version="catalog-v1",
            user={},
        ),
        _identity_response(True),
        _identity_response(0),
        _identity_response(2**63),
        _identity_response("42"),
    ],
)
async def test_character_authority_resolver_rejects_invalid_identity_response(
    tmp_path,
    monkeypatch,
    response,
):
    client = IdentityClient(response)
    provider, _, _, _ = _authority_provider(tmp_path, monkeypatch, [client])

    with pytest.raises(server_context_module.ServerIdentityUnavailable) as exc_info:
        await _resolve_authority(provider)

    assert exc_info.value.to_contract() == {
        "reason_code": "server_identity_unavailable",
        "message": "Server identity unavailable.",
        "recoverable": True,
        "active_server_id": None,
    }


@pytest.mark.asyncio
async def test_character_authority_endpoint_failure_is_bounded_and_client_remains_usable(
    tmp_path,
    monkeypatch,
):
    sensitive_endpoint_error = "identity-response-sensitive-sentinel"
    client = IdentityClient(error=RuntimeError(sensitive_endpoint_error))
    provider, _, _, build_calls = _authority_provider(
        tmp_path,
        monkeypatch,
        [client],
    )

    with pytest.raises(server_context_module.ServerIdentityUnavailable) as exc_info:
        await _resolve_authority(provider)

    assert str(exc_info.value) == "Server identity unavailable."
    assert sensitive_endpoint_error not in repr(exc_info.value.to_contract())
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True
    assert provider.build_client() is client
    assert len(build_calls) == 1


@pytest.mark.asyncio
async def test_character_authority_resolver_does_not_swallow_cancellation(
    tmp_path,
    monkeypatch,
):
    client = IdentityClient(error=asyncio.CancelledError())
    provider, _, _, _ = _authority_provider(tmp_path, monkeypatch, [client])

    with pytest.raises(asyncio.CancelledError):
        await _resolve_authority(provider)


@pytest.mark.asyncio
async def test_character_authority_scope_failure_is_bounded_before_client_build(
    tmp_path,
    monkeypatch,
):
    client = IdentityClient(_identity_response(42))
    provider, target_store, _, build_calls = _authority_provider(
        tmp_path,
        monkeypatch,
        [client],
    )

    def unavailable_scope(server_id):
        raise AuthorityScopeUnavailable()

    monkeypatch.setattr(
        target_store,
        "ensure_authority_scope_id",
        unavailable_scope,
    )

    with pytest.raises(server_context_module.ServerIdentityUnavailable) as exc_info:
        await _resolve_authority(provider)

    assert str(exc_info.value) == "Server identity unavailable."
    assert exc_info.value.__cause__ is None
    assert build_calls == []
    assert provider.build_client() is client
    assert client.profile_calls == []


@pytest.mark.asyncio
async def test_character_authority_resolver_rejects_malformed_persisted_scope(
    tmp_path,
    monkeypatch,
):
    target = ConfiguredServerTarget(
        server_id="https://server.example.com/api",
        label="Primary",
        base_url="https://server.example.com/api",
        auth_mode="bearer",
        is_default=True,
        authority_scope_id="not-a-canonical-scope",
    )
    client = IdentityClient(_identity_response(42))
    provider, _, _, build_calls = _authority_provider(
        tmp_path,
        monkeypatch,
        [client],
        target=target,
    )

    with pytest.raises(server_context_module.ServerIdentityUnavailable):
        await _resolve_authority(provider)

    assert build_calls == []
    assert client.profile_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("context_change", ["target_metadata", "credential"])
async def test_character_authority_refetches_and_remains_stable_across_context_changes(
    tmp_path,
    monkeypatch,
    context_change,
):
    first_client = IdentityClient(_identity_response(42))
    second_client = IdentityClient(_identity_response(42))
    provider, target_store, _, build_calls = _authority_provider(
        tmp_path,
        monkeypatch,
        [first_client, second_client],
    )

    first = await _resolve_authority(provider)
    if context_change == "target_metadata":
        original_target = target_store.get_target("https://server.example.com/api")
        assert original_target is not None
        target_store.save_targets(
            [
                replace(
                    original_target,
                    label="Renamed",
                    base_url="https://moved.example.com/api",
                    last_known_server_label="Mutable display label",
                )
            ]
        )
    else:
        provider.store_auth_tokens(access_token="access-2")
    second = await _resolve_authority(provider)
    await asyncio.sleep(0)

    assert second == first
    assert len(build_calls) == 2
    assert first_client.profile_calls == [{"sections": "identity"}]
    assert second_client.profile_calls == [{"sections": "identity"}]
    assert first_client.close_calls == 1


@pytest.mark.asyncio
async def test_character_authority_separates_two_users_on_one_target(
    tmp_path,
    monkeypatch,
):
    first_client = IdentityClient(_identity_response(41))
    second_client = IdentityClient(_identity_response(42))
    provider, _, _, _ = _authority_provider(
        tmp_path,
        monkeypatch,
        [first_client, second_client],
    )

    first = await _resolve_authority(provider)
    provider.store_auth_tokens(access_token="access-2")
    second = await _resolve_authority(provider)

    assert first != second
    assert first == server_context_module.encode_server_user_character_authority(
        _AUTHORITY_SCOPE,
        41,
    )
    assert second == server_context_module.encode_server_user_character_authority(
        _AUTHORITY_SCOPE,
        42,
    )


@pytest.mark.asyncio
async def test_close_cached_client_clears_character_authority_cache(
    tmp_path,
    monkeypatch,
):
    client = IdentityClient(_identity_response(42))
    provider, _, _, _ = _authority_provider(tmp_path, monkeypatch, [client])
    await _resolve_authority(provider)

    await provider.close_cached_client()

    assert provider._cached_character_authority is None
    assert client.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stale_change",
    [
        "runtime_revision",
        "server_switch",
        "switch_away_and_back",
        "target",
        "scope",
        "credential",
        "client",
    ],
)
async def test_character_authority_rejects_stale_in_flight_capture(
    tmp_path,
    monkeypatch,
    stale_change,
):
    runtime_context = _runtime_context()
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
        "access-1",
    )
    stale_client = GatedIdentityClient(_identity_response(42))
    replacement_client = IdentityClient(_identity_response(42))
    clients = (
        [stale_client, replacement_client]
        if stale_change == "credential"
        else [stale_client]
    )
    provider, target_store, _, _ = _authority_provider(
        tmp_path,
        monkeypatch,
        clients,
        runtime_context=runtime_context,
        credential_store=credentials,
    )
    pending = asyncio.create_task(_resolve_authority(provider))
    await stale_client.started.wait()

    if stale_change == "runtime_revision":
        _commit_runtime_state(
            runtime_context,
            replace(runtime_context.state, last_known_server_label="Changed"),
        )
    elif stale_change in {"server_switch", "switch_away_and_back"}:
        original_state = runtime_context.state
        _commit_runtime_state(
            runtime_context,
            replace(
                original_state,
                active_server_id="https://other.example.com/api",
            ),
        )
        if stale_change == "switch_away_and_back":
            _commit_runtime_state(runtime_context, original_state)
    elif stale_change in {"target", "scope"}:
        original_target = target_store.get_target("https://server.example.com/api")
        assert original_target is not None
        replacement = (
            replace(original_target, label="Changed in flight")
            if stale_change == "target"
            else replace(
                original_target,
                authority_scope_id=_SECOND_AUTHORITY_SCOPE,
            )
        )
        target_store.save_targets([replacement])
    elif stale_change == "credential":
        credentials.set_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_ACCESS_TOKEN,
            "access-2",
        )
    else:
        provider._cached_client = replacement_client
    stale_client.release.set()

    with pytest.raises(server_context_module.ServerIdentityUnavailable):
        await pending
    assert provider._cached_character_authority is None
    if stale_change == "credential":
        assert provider.build_client() is replacement_client
        await asyncio.sleep(0)
        assert stale_client.close_calls == 1


@pytest.mark.asyncio
async def test_character_authority_rejects_older_conflicting_account_response(
    tmp_path,
    monkeypatch,
):
    old_started = asyncio.Event()
    release_old = asyncio.Event()

    class SwitchingAccountClient(IdentityClient):
        async def get_current_user_profile(self, **kwargs):
            self.profile_calls.append(kwargs)
            if len(self.profile_calls) == 1:
                old_started.set()
                await release_old.wait()
                return _identity_response(41)
            return _identity_response(42)

    client = SwitchingAccountClient()
    provider, _, _, _ = _authority_provider(tmp_path, monkeypatch, [client])
    old_request = asyncio.create_task(_resolve_authority(provider))
    await old_started.wait()

    current = await _resolve_authority(provider)
    release_old.set()

    with pytest.raises(server_context_module.ServerIdentityUnavailable):
        await old_request
    assert current == server_context_module.encode_server_user_character_authority(
        _AUTHORITY_SCOPE,
        42,
    )
    assert client.profile_calls == [
        {"sections": "identity"},
        {"sections": "identity"},
    ]


def test_resolves_matching_target_and_credential_store_secret(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
        "bearer-secret",
    )
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
    assert (
        context.credential_source
        == f"credential_store:{SERVER_CREDENTIAL_BEARER_TOKEN}"
    )
    assert context.target == target


def test_rejects_server_mode_without_active_server(tmp_path):
    provider = _provider(
        tmp_path,
        runtime_context=_runtime_context(active_server_id=None, server_configured=True),
    )

    with pytest.raises(ServerContextUnavailable):
        provider.get_active_context()


def test_server_context_failure_reason_codes_include_stable_contract_values():
    expected_reason_codes = {
        "server_not_configured",
        "server_profile_missing",
        "server_unavailable",
        "auth_required",
        "credential_store_unavailable",
        "server_credentials_unavailable",
        "stale_authorization",
        "profile_no_longer_authorized",
        "server_identity_unavailable",
    }

    assert set(get_args(ServerContextFailureReason)) == expected_reason_codes
    assert SERVER_CONTEXT_FAILURE_REASON_CODES == expected_reason_codes


def test_runtime_policy_package_exports_character_authority_apis():
    import tldw_chatbook.runtime_policy as runtime_policy

    assert (
        runtime_policy.ServerIdentityUnavailable
        is server_context_module.ServerIdentityUnavailable
    )
    assert (
        runtime_policy.encode_server_user_character_authority
        is server_context_module.encode_server_user_character_authority
    )
    assert {
        "ServerIdentityUnavailable",
        "encode_server_user_character_authority",
    } <= set(runtime_policy.__all__)


def test_context_unavailable_error_exposes_reason_code_and_safe_payload(tmp_path):
    provider = _provider(
        tmp_path, runtime_context=_runtime_context(active_server_id=None)
    )

    with pytest.raises(ServerContextUnavailable) as exc:
        provider.get_active_context()

    contract = exc.value.to_contract()
    assert exc.value.reason_code == "server_not_configured"
    assert contract["reason_code"] == "server_not_configured"
    assert contract["recoverable"] is True
    assert contract["active_server_id"] is None
    assert "token" not in repr(contract).lower()
    assert "authorization" not in repr(contract).lower()


def test_missing_active_server_profile_uses_profile_missing_contract(tmp_path):
    provider = _provider(tmp_path)

    with pytest.raises(ServerContextUnavailable) as exc:
        provider.get_active_context()

    contract = exc.value.to_contract()
    assert exc.value.reason_code == "server_profile_missing"
    assert contract["reason_code"] == "server_profile_missing"
    assert contract["active_server_id"] == "https://server.example.com/api"


def test_legacy_fallback_works_when_no_target_exists_and_app_config_matches_active_server(
    tmp_path,
):
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
    assert (
        context.credential_source
        == f"credential_store:{SERVER_CREDENTIAL_BEARER_TOKEN}"
    )
    assert context.target.server_id == "https://server.example.com/api"
    assert context.target.auth_reference == "legacy:tldw_api"


def test_legacy_target_prefers_credential_store_token_over_legacy_config(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
        "stored-access",
    )

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
    assert (
        context.credential_source
        == f"credential_store:{SERVER_CREDENTIAL_ACCESS_TOKEN}"
    )


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
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        == "legacy-bearer"
    )
    assert (
        credentials.get_secret(
            "https://backup.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        is None
    )


def test_legacy_config_token_does_not_apply_to_nonmatching_active_server_profile(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    provider = _provider(
        tmp_path,
        runtime_context=_runtime_context(
            active_server_id="https://backup.example.com/api"
        ),
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
    assert (
        credentials.get_secret(
            "https://backup.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        is None
    )


def test_legacy_fallback_without_target_prefers_credential_store_token_over_legacy_config(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
        "stored-bearer",
    )

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
    assert (
        context.credential_source
        == f"credential_store:{SERVER_CREDENTIAL_BEARER_TOKEN}"
    )


def test_legacy_fallback_uses_config_token_when_credential_store_is_unavailable(
    tmp_path,
):
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


def test_explicit_keyring_reference_raises_typed_error_when_credential_store_is_unavailable(
    tmp_path,
):
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


def test_explicit_keyring_reference_preserves_credential_store_unavailable_reason_code(
    tmp_path,
):
    provider = _provider(
        tmp_path,
        credential_store=UnavailableServerCredentialStore("no secure store"),
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
    )

    with pytest.raises(ServerCredentialsUnavailable) as exc:
        provider.get_active_context()

    assert exc.value.reason_code == CredentialStoreUnavailable.reason_code


def test_credential_store_unavailable_reason_is_preserved(tmp_path):
    provider = _provider(
        tmp_path,
        credential_store=UnavailableServerCredentialStore("secure store unavailable"),
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

    with pytest.raises(ServerCredentialsUnavailable) as exc:
        provider.get_active_context()

    contract = exc.value.to_contract()
    assert exc.value.reason_code == "credential_store_unavailable"
    assert contract["reason_code"] == "credential_store_unavailable"
    assert contract["recoverable"] is True
    assert contract["active_server_id"] == "https://server.example.com/api"
    assert "secure store unavailable" not in contract["message"]


def test_generic_credential_failure_contract_is_sanitized(tmp_path):
    provider = _provider(
        tmp_path,
        credential_store=LookupFailingCredentialStore(),
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

    with pytest.raises(ServerCredentialsUnavailable) as exc:
        provider.get_active_context()

    contract_repr = repr(exc.value.to_contract()).lower()
    assert exc.value.reason_code == "server_credentials_unavailable"
    assert "lookup unavailable" not in contract_repr
    assert "token" not in contract_repr
    assert "authorization" not in contract_repr


def test_bearer_auth_prefers_bearer_token_then_access_token_before_legacy_config(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
        "stored-access",
    )
    credentials.set_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
        "stored-bearer",
    )

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
    assert (
        context.credential_source
        == f"credential_store:{SERVER_CREDENTIAL_BEARER_TOKEN}"
    )


def test_context_computes_bearer_headers_from_effective_auth_token(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
        "stored-bearer",
    )
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
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, "stored-api-key"
    )

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
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, "stored-api-key"
    )
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
    _commit_runtime_state(
        runtime_context,
        replace(
            runtime_context.state,
            server_reachability="reachable",
            server_auth_state="authenticated",
            last_known_server_label="Runtime Label",
        ),
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
    _commit_runtime_state(
        runtime_context,
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://backup.example.com/api",
            server_configured=True,
            server_reachability="unreachable",
            server_auth_state="session_invalid",
            last_known_server_label="Backup Runtime",
        ),
    )
    second_context = provider.get_active_context()

    assert first_context.target.server_id == "https://server.example.com/api"
    assert first_context.capabilities["reachability"] == "unknown"
    assert (
        first_context.capabilities["target_last_known_server_label"] == "Primary Target"
    )
    assert second_context.target.server_id == "https://backup.example.com/api"
    assert second_context.capabilities["reachability"] == "unreachable"
    assert second_context.capabilities["auth_state"] == "session_invalid"
    assert second_context.capabilities["last_known_server_label"] == "Backup Runtime"
    assert (
        second_context.capabilities["target_last_known_server_label"] == "Backup Target"
    )


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


def test_build_client_without_required_credentials_raises_auth_required_contract(
    tmp_path,
):
    provider = _provider(
        tmp_path,
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

    with pytest.raises(ServerCredentialsUnavailable) as exc:
        provider.build_client()

    contract = exc.value.to_contract()
    assert exc.value.reason_code == "auth_required"
    assert contract["reason_code"] == "auth_required"
    assert contract["active_server_id"] == "https://server.example.com/api"
    assert "token" not in repr(contract).lower()
    assert "authorization" not in repr(contract).lower()


def test_build_client_raises_server_unavailable_when_runtime_state_is_unreachable(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, "stored-api-key"
    )
    runtime_context = _runtime_context()
    _commit_runtime_state(
        runtime_context,
        replace(runtime_context.state, server_reachability="unreachable"),
    )
    provider = _provider(
        tmp_path,
        runtime_context=runtime_context,
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

    with pytest.raises(ServerContextUnavailable) as exc:
        provider.build_client()

    assert exc.value.reason_code == "server_unavailable"
    assert exc.value.to_contract()["reason_code"] == "server_unavailable"


def test_build_client_raises_auth_required_when_runtime_state_requires_auth(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, "stored-api-key"
    )
    runtime_context = _runtime_context()
    _commit_runtime_state(
        runtime_context,
        replace(runtime_context.state, server_auth_state="auth_required"),
    )
    provider = _provider(
        tmp_path,
        runtime_context=runtime_context,
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

    with pytest.raises(ServerCredentialsUnavailable) as exc:
        provider.build_client()

    assert exc.value.reason_code == "auth_required"
    assert exc.value.to_contract()["reason_code"] == "auth_required"


def test_build_client_raises_stale_authorization_when_runtime_session_invalid(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, "stored-api-key"
    )
    runtime_context = _runtime_context()
    _commit_runtime_state(
        runtime_context,
        replace(runtime_context.state, server_auth_state="session_invalid"),
    )
    provider = _provider(
        tmp_path,
        runtime_context=runtime_context,
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

    with pytest.raises(ServerCredentialsUnavailable) as exc:
        provider.build_client()

    assert exc.value.reason_code == "stale_authorization"
    assert exc.value.to_contract()["reason_code"] == "stale_authorization"


def test_build_client_uses_active_context_base_url_and_bearer_token(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
        "access-secret",
    )
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
    credentials.set_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
        "access-secret",
    )
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
async def test_stored_token_change_replaces_cached_client_and_closes_opened_old_client(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-1"
    )
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
async def test_clear_active_server_auth_tokens_invalidates_cache_and_preserves_static_credentials(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-1"
    )
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_REFRESH_TOKEN, "refresh-1"
    )
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, "api-key-1"
    )
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN, "bearer-1"
    )
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
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_ACCESS_TOKEN,
        )
        is None
    )
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_REFRESH_TOKEN,
        )
        is None
    )
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_API_KEY,
        )
        == "api-key-1"
    )
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        == "bearer-1"
    )


@pytest.mark.asyncio
async def test_clear_all_credentials_invalidates_cached_client_and_removes_imported_credentials(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://backup.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
        "backup-access",
    )
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
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        is None
    )
    assert (
        credentials.get_secret(
            "https://backup.example.com/api",
            SERVER_CREDENTIAL_ACCESS_TOKEN,
        )
        is None
    )
    assert provider.app_config["tldw_api"]["bearer_token"] == "legacy-bearer"

    with pytest.raises(ServerCredentialsUnavailable):
        provider.get_active_context()
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        is None
    )


def test_clear_all_credentials_blocks_other_legacy_backed_profile_after_server_switch(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server-a.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
        "server-a-secret",
    )
    runtime_context = _runtime_context(
        active_server_id="https://server-a.example.com/api"
    )
    provider = _provider(
        tmp_path,
        runtime_context=runtime_context,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server-a.example.com/api",
                label="Server A",
                base_url="https://server-a.example.com/api/",
                auth_mode="bearer",
                auth_reference="legacy:tldw_api",
                is_default=True,
            ),
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Server B",
                base_url="https://server.example.com/api/",
                auth_mode="bearer",
                auth_reference="legacy:tldw_api",
            ),
        ],
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "legacy-bearer",
                "auth_mode": "bearer",
            }
        },
    )

    provider.clear_all_credentials()
    _commit_runtime_state(
        runtime_context,
        replace(
            runtime_context.state,
            active_server_id="https://server.example.com/api",
        ),
    )

    with pytest.raises(ServerCredentialsUnavailable):
        provider.get_active_context()
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        is None
    )


def test_clear_all_credentials_preserves_original_credential_store_error_for_legacy_profile(
    tmp_path,
):
    provider = _provider(
        tmp_path,
        credential_store=LookupFailingCredentialStore(),
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

    provider.clear_all_credentials()

    with pytest.raises(ServerCredentialsUnavailable) as exc:
        provider.get_active_context()

    assert isinstance(exc.value.__cause__, RuntimeError)


def test_target_store_json_and_target_metadata_do_not_contain_stored_secret(tmp_path):
    secret = "literal-provider-token-must-not-leak"
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, secret
    )
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


def test_clear_active_server_credentials_and_clear_server_credentials_clear_per_server_secrets(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_BEARER_TOKEN,
        "active-secret",
    )
    credentials.set_secret("server-b", SERVER_CREDENTIAL_BEARER_TOKEN, "other-secret")
    provider = _provider(tmp_path, credential_store=credentials)

    provider.clear_active_server_credentials()

    assert (
        credentials.get_secret(
            "https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN
        )
        is None
    )
    assert (
        credentials.get_secret("server-b", SERVER_CREDENTIAL_BEARER_TOKEN)
        == "other-secret"
    )

    provider.clear_server_credentials("server-b")

    assert credentials.get_secret("server-b", SERVER_CREDENTIAL_BEARER_TOKEN) is None


def test_clear_active_server_credentials_blocks_legacy_profile_reimport(tmp_path):
    credentials = InMemoryServerCredentialStore()
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

    provider.get_active_context()
    provider.clear_active_server_credentials()

    with pytest.raises(ServerCredentialsUnavailable):
        provider.get_active_context()
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        is None
    )


def test_cleared_legacy_profile_uses_no_longer_authorized_contract(tmp_path):
    credentials = InMemoryServerCredentialStore()
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

    provider.get_active_context()
    provider.clear_active_server_credentials()

    with pytest.raises(ServerCredentialsUnavailable) as exc:
        provider.get_active_context()

    contract = exc.value.to_contract()
    assert exc.value.reason_code == "profile_no_longer_authorized"
    assert contract["reason_code"] == "profile_no_longer_authorized"
    assert contract["active_server_id"] == "https://server.example.com/api"
    assert "legacy-bearer" not in repr(contract)


def test_clear_server_credentials_blocks_legacy_profile_reimport_after_activation(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    runtime_context = _runtime_context(
        active_server_id="https://server-a.example.com/api"
    )
    provider = _provider(
        tmp_path,
        runtime_context=runtime_context,
        credential_store=credentials,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server-a.example.com/api",
                label="Server A",
                base_url="https://server-a.example.com/api/",
                auth_mode="bearer",
                auth_reference="legacy:tldw_api",
                is_default=True,
            ),
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Server B",
                base_url="https://server.example.com/api/",
                auth_mode="bearer",
                auth_reference="legacy:tldw_api",
            ),
        ],
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "legacy-bearer",
                "auth_mode": "bearer",
            }
        },
    )

    _commit_runtime_state(
        runtime_context,
        replace(
            runtime_context.state,
            active_server_id="https://server.example.com/api",
        ),
    )
    context = provider.get_active_context()

    assert context.active_server_id == "https://server.example.com/api"
    assert context.auth_token == "legacy-bearer"
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        == "legacy-bearer"
    )

    _commit_runtime_state(
        runtime_context,
        replace(
            runtime_context.state,
            active_server_id="https://server-a.example.com/api",
        ),
    )
    provider.clear_server_credentials("https://server.example.com/api")

    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        is None
    )

    _commit_runtime_state(
        runtime_context,
        replace(
            runtime_context.state,
            active_server_id="https://server.example.com/api",
        ),
    )

    with pytest.raises(ServerCredentialsUnavailable):
        provider.get_active_context()
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        is None
    )


def test_rebind_app_config_installs_config_invalidates_cache_and_preserves_signout(
    tmp_path,
    monkeypatch,
):
    provider = _provider(
        tmp_path,
        targets=[
            ConfiguredServerTarget(
                server_id="https://old.example.com/api",
                label="Old",
                base_url="https://old.example.com/api",
                auth_mode="bearer",
                is_default=True,
            )
        ],
    )
    provider._legacy_cleared_server_ids.add("https://old.example.com/api")
    hook_calls: list[tuple[str, str | None, str | None]] = []
    monkeypatch.setattr(
        provider,
        "_invalidate_event_handles_for_server_switch",
        lambda previous, next_id: hook_calls.append(("event", previous, next_id)),
    )
    monkeypatch.setattr(
        provider,
        "_invalidate_sync_handles_for_server_switch",
        lambda previous, next_id: hook_calls.append(("sync", previous, next_id)),
    )

    class ClosingClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    cached_client = ClosingClient()
    provider._cached_client = cached_client
    provider._cached_client_key = object()
    refreshed_config = {
        "tldw_api": {
            "base_url": "https://new.example.com/api/",
            "bearer_token": "refreshed-token",
            "auth_mode": "bearer",
        }
    }

    provider.rebind_app_config(
        refreshed_config,
        previous_server_id=" https://old.example.com/api ",
        next_server_id="https://old.example.com/api",
    )

    assert provider.app_config is refreshed_config
    assert provider._cached_client is None
    assert provider._cached_client_key is None
    assert cached_client.close_calls == 1
    assert hook_calls == []
    assert provider._legacy_cleared_server_ids == {"https://old.example.com/api"}
    targets = provider.target_store.list_targets()
    assert [target.server_id for target in targets] == [
        "https://old.example.com/api",
        "https://new.example.com/api",
    ]
    assert [target.is_default for target in targets] == [False, True]


def test_rebind_app_config_invalidates_switch_hooks_once_with_normalized_ids(
    tmp_path,
    monkeypatch,
):
    provider = _provider(tmp_path)
    hook_calls: list[tuple[str, str | None, str | None]] = []
    monkeypatch.setattr(
        provider,
        "_invalidate_event_handles_for_server_switch",
        lambda previous, next_id: hook_calls.append(("event", previous, next_id)),
    )
    monkeypatch.setattr(
        provider,
        "_invalidate_sync_handles_for_server_switch",
        lambda previous, next_id: hook_calls.append(("sync", previous, next_id)),
    )

    provider.rebind_app_config(
        {},
        previous_server_id=" server-a ",
        next_server_id=" server-b ",
    )

    assert hook_calls == [
        ("event", "server-a", "server-b"),
        ("sync", "server-a", "server-b"),
    ]


def test_rebind_app_config_contains_target_write_failure_and_uses_refreshed_fallback(
    tmp_path,
    monkeypatch,
):
    endpoint_sentinel = "https://endpoint-sentinel.example.com/api"
    token_sentinel = "token-sentinel-value"
    path_sentinel = tmp_path / "path-sentinel-targets.json"
    exception_sentinel = "exception-message-sentinel"
    runtime_context = _runtime_context(active_server_id=endpoint_sentinel)
    target_store = ConfiguredServerTargetStore(path_sentinel)
    provider = RuntimeServerContextProvider(
        runtime_context=runtime_context,
        target_store=target_store,
        credential_store=InMemoryServerCredentialStore(),
        app_config={},
    )

    class ClosingClient:
        async def close(self) -> None:
            return None

    provider._cached_client = ClosingClient()
    provider._cached_client_key = object()
    refreshed_config = {
        "tldw_api": {
            "base_url": f"{endpoint_sentinel}/",
            "bearer_token": token_sentinel,
            "auth_mode": "bearer",
        }
    }

    def fail_after_install(app_config):
        assert app_config is refreshed_config
        assert provider.app_config is refreshed_config
        assert provider._cached_client is None
        assert provider._cached_client_key is None
        raise RuntimeError(exception_sentinel)

    monkeypatch.setattr(
        target_store,
        "upsert_legacy_config_target",
        fail_after_install,
    )
    warnings: list[str] = []
    sink = server_context_module.logger.add(
        warnings.append,
        level="WARNING",
        format="{message}",
    )
    try:
        provider.rebind_app_config(
            refreshed_config,
            previous_server_id=None,
            next_server_id=endpoint_sentinel,
        )
    finally:
        server_context_module.logger.remove(sink)

    context = provider.get_active_context()
    assert context.active_server_id == endpoint_sentinel
    assert context.base_url == endpoint_sentinel
    assert context.auth_token == token_sentinel
    assert len(warnings) == 1
    warning = warnings[0]
    assert "exception_category=RuntimeError" in warning
    assert endpoint_sentinel not in warning
    assert token_sentinel not in warning
    assert str(path_sentinel) not in warning
    assert exception_sentinel not in warning


def test_rebind_app_config_contains_synchronous_close_failure(tmp_path):
    close_sentinel = "synchronous-close-secret"
    provider = _provider(tmp_path)

    class FailingCloseClient:
        async def close(self) -> None:
            raise RuntimeError(close_sentinel)

    provider._cached_client = FailingCloseClient()
    provider._cached_client_key = object()
    warnings: list[str] = []
    sink = server_context_module.logger.add(
        warnings.append,
        level="WARNING",
        format="{message}",
    )
    try:
        provider.rebind_app_config(
            {},
            previous_server_id=None,
            next_server_id=None,
        )
    finally:
        server_context_module.logger.remove(sink)

    assert provider._cached_client is None
    assert provider._cached_client_key is None
    assert len(warnings) == 1
    assert "exception_category=RuntimeError" in warnings[0]
    assert close_sentinel not in warnings[0]


@pytest.mark.asyncio
async def test_rebind_app_config_contains_scheduled_close_failure(tmp_path):
    close_sentinel = "scheduled-close-secret"
    provider = _provider(tmp_path)

    class FailingCloseClient:
        async def close(self) -> None:
            raise RuntimeError(close_sentinel)

    provider._cached_client = FailingCloseClient()
    provider._cached_client_key = object()
    warnings: list[str] = []
    sink = server_context_module.logger.add(
        warnings.append,
        level="WARNING",
        format="{message}",
    )
    try:
        provider.rebind_app_config(
            {},
            previous_server_id=None,
            next_server_id=None,
        )
        pending_tasks = tuple(provider._pending_client_close_tasks)
        assert len(pending_tasks) == 1
        await asyncio.gather(*pending_tasks)
        await asyncio.sleep(0)
    finally:
        server_context_module.logger.remove(sink)

    assert provider._cached_client is None
    assert provider._cached_client_key is None
    assert provider._pending_client_close_tasks == set()
    assert len(warnings) == 1
    assert "exception_category=RuntimeError" in warnings[0]
    assert close_sentinel not in warnings[0]


@pytest.mark.asyncio
async def test_invalidate_for_server_switch_closes_cached_client_without_clearing_credentials(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server-a.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
        "server-a-access",
    )
    credentials.set_secret(
        "https://server-b.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
        "server-b-access",
    )
    runtime_context = _runtime_context(
        active_server_id="https://server-a.example.com/api"
    )
    provider = RuntimeServerContextProvider(
        runtime_context=runtime_context,
        target_store=_target_store(
            tmp_path,
            [
                ConfiguredServerTarget(
                    server_id="https://server-a.example.com/api",
                    label="Server A",
                    base_url="https://server-a.example.com/api/",
                    auth_mode="bearer",
                    is_default=True,
                ),
                ConfiguredServerTarget(
                    server_id="https://server-b.example.com/api",
                    label="Server B",
                    base_url="https://server-b.example.com/api/",
                    auth_mode="bearer",
                ),
            ],
        ),
        credential_store=credentials,
        app_config={},
    )

    first_client = provider.build_client()
    opened_http_client = await first_client._get_client()

    provider.invalidate_for_server_switch(
        "https://server-a.example.com/api",
        "https://server-a.example.com/api",
    )

    assert provider._cached_client is first_client
    assert not opened_http_client.is_closed

    provider._legacy_cleared_server_ids.add("https://server-a.example.com/api")
    provider.invalidate_for_server_switch(
        "https://server-a.example.com/api",
        "https://server-b.example.com/api",
    )
    if provider._pending_client_close_tasks:
        await asyncio.gather(*provider._pending_client_close_tasks)

    assert provider._cached_client is None
    assert opened_http_client.is_closed
    assert (
        credentials.get_secret(
            "https://server-a.example.com/api",
            SERVER_CREDENTIAL_ACCESS_TOKEN,
        )
        == "server-a-access"
    )
    assert (
        credentials.get_secret(
            "https://server-b.example.com/api",
            SERVER_CREDENTIAL_ACCESS_TOKEN,
        )
        == "server-b-access"
    )
    assert "https://server-a.example.com/api" in provider._legacy_cleared_server_ids


@pytest.mark.asyncio
async def test_switching_active_server_rebuilds_client_with_new_profile_and_closes_old_client(
    tmp_path,
):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
        "shared-access",
    )
    credentials.set_secret(
        "https://backup.example.com/api",
        SERVER_CREDENTIAL_ACCESS_TOKEN,
        "shared-access",
    )
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

    _commit_runtime_state(
        runtime_context,
        replace(
            runtime_context.state,
            active_server_id="https://backup.example.com/api",
        ),
    )
    second_client = provider.build_client()
    await provider.close_cached_client()

    assert second_client is not first_client
    assert opened_http_client.is_closed
    assert second_client.base_url == "https://shared.example.com/api"
    assert second_client.bearer_token == "shared-access"


def test_clear_active_server_auth_tokens_preserves_static_credentials(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-1"
    )
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_REFRESH_TOKEN, "refresh-1"
    )
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_API_KEY, "api-key-1"
    )
    credentials.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN, "bearer-1"
    )
    credentials.set_secret(
        "https://backup.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "other-access"
    )
    provider = _provider(tmp_path, credential_store=credentials)

    provider.clear_active_server_auth_tokens()

    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_ACCESS_TOKEN,
        )
        is None
    )
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_REFRESH_TOKEN,
        )
        is None
    )
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_API_KEY,
        )
        == "api-key-1"
    )
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )
        == "bearer-1"
    )
    assert (
        credentials.get_secret(
            "https://backup.example.com/api",
            SERVER_CREDENTIAL_ACCESS_TOKEN,
        )
        == "other-access"
    )


def test_store_auth_tokens_scopes_tokens_to_active_server(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret(
        "https://backup.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "other-access"
    )
    credentials.set_secret(
        "https://backup.example.com/api",
        SERVER_CREDENTIAL_REFRESH_TOKEN,
        "other-refresh",
    )
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

    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_ACCESS_TOKEN,
        )
        == "access-1"
    )
    assert (
        credentials.get_secret(
            "https://server.example.com/api",
            SERVER_CREDENTIAL_REFRESH_TOKEN,
        )
        == "refresh-1"
    )
    assert (
        credentials.get_secret(
            "https://backup.example.com/api",
            SERVER_CREDENTIAL_ACCESS_TOKEN,
        )
        == "other-access"
    )
    assert (
        credentials.get_secret(
            "https://backup.example.com/api",
            SERVER_CREDENTIAL_REFRESH_TOKEN,
        )
        == "other-refresh"
    )

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


def test_runtime_state_remains_authoritative_and_unmutated_during_context_resolution(
    tmp_path,
):
    runtime_store = SavingRuntimeStore()
    runtime_context = _runtime_context(runtime_store=runtime_store)
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
    assert runtime_store.saved_states == []
