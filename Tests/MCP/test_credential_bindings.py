"""Stable authority is asserted by the credential owner, never token contents."""

import asyncio
import threading

import pytest


class MemoryCredentialBackend:
    """Isolated protected-storage stand-in; never the user's keychain."""

    def __init__(self):
        self.records = {}
        self.lock = threading.RLock()

    def transaction(self):
        return self.lock

    def read(self, reference_id):
        return self.records.get(reference_id)

    def write(self, reference_id, value):
        self.records[reference_id] = value


@pytest.fixture
def binding_case():
    from tldw_chatbook.MCP.credential_bindings import (
        CredentialBindingService,
        HostCredential,
    )

    class Case:
        def __init__(self):
            self.backend = MemoryCredentialBackend()
            self.principal = "account-a"
            self.token = "secret-sentinel-initial"
            self.issuer = "https://issuer.example"
            self.scopes = ("read",)
            self.service = CredentialBindingService(
                self.backend, adapters={"test-host": self.callback}
            )
            self.reference = asyncio.run(self.service.create("test-host")).reference_id

        async def callback(self, reference_id):
            return HostCredential(
                method="bearer",
                endpoint_origin="https://mcp.example",
                headers={"Authorization": "Bearer " + self.token},
                issuer=self.issuer,
                audience="mcp",
                principal=self.principal,
                scopes=self.scopes,
            )

        def update(self):
            asyncio.run(self.service.authorize("test-host", self.reference))

        def binding(self):
            return self.service.binding(self.reference)

        def rotate_token_with_same_identity(self):
            self.token = "secret-sentinel-renewed"
            asyncio.run(self.service.renew(self.reference))

        def switch_account(self):
            self.principal = "account-b"
            self.update()

    return Case()


def test_token_rotation_does_not_change_binding_generation(binding_case):
    case = binding_case
    old = case.binding()
    case.rotate_token_with_same_identity()
    assert case.binding().authority_generation == old.authority_generation
    case.switch_account()
    assert case.binding().authority_generation > old.authority_generation


def test_profile_owner_retains_credential_reference_and_generation(tmp_path):
    """Existing real save entry must preserve the reviewed reference pair."""
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore

    owner = LocalMCPControlService(store=LocalMCPStore(tmp_path / "mcp.json"))
    control = owner.save_external_profile(
        {"profile_id": "control", "command": "python"}
    )
    assert control["command"] == "python"
    result = owner.save_external_profile(
        {
            "profile_id": "remote",
            "transport": "streamable_http",
            "url": "https://mcp.example/rpc",
            "credential_reference": "binding-1",
            "credential_generation": 1,
        }
    )
    assert result.get("credential_reference") == "binding-1"
    assert result.get("credential_generation") == 1


def test_current_secrets_resolved_only_for_exact_binding(binding_case):
    from tldw_chatbook.MCP.credential_bindings import CredentialError

    case = binding_case
    old = case.binding()
    case.rotate_token_with_same_identity()
    assert case.service.resolve(
        old.reference_id, old.authority_generation, old.endpoint_origin
    ) == {"authorization": "Bearer secret-sentinel-renewed"}
    assert "secret-sentinel" not in repr(old)
    case.switch_account()
    with pytest.raises(CredentialError, match="credential_changed"):
        case.service.resolve(
            old.reference_id, old.authority_generation, old.endpoint_origin
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("principal", "other"),
        ("issuer", "other"),
        ("scopes", ("read", "write")),
        ("principal", None),
        ("issuer", None),
    ],
)
def test_changed_or_unknown_identity_invalidates_authority(binding_case, field, value):
    case = binding_case
    old = case.binding()
    setattr(case, field, value)
    case.update()
    assert case.binding().authority_generation > old.authority_generation


def test_opaque_replacement_and_revocation_never_reuse_generation():
    from tldw_chatbook.MCP.credential_bindings import (
        CredentialBindingService,
        CredentialError,
    )

    service = CredentialBindingService(MemoryCredentialBackend())
    first = service.create_opaque(
        endpoint_origin="https://mcp.example", headers={"X-Key": "secret-sentinel"}
    )
    assert first.principal is first.issuer is first.scopes is None
    second = service.set_opaque(
        first.reference_id,
        endpoint_origin=first.endpoint_origin,
        headers={"X-Key": "secret-sentinel"},
    )
    assert second.authority_generation > first.authority_generation
    service.revoke(first.reference_id)
    with pytest.raises(CredentialError, match="credential_revoked"):
        service.resolve(
            first.reference_id, second.authority_generation, first.endpoint_origin
        )
    third = service.set_opaque(
        first.reference_id,
        endpoint_origin=first.endpoint_origin,
        headers={"X-Key": "secret-sentinel"},
    )
    assert third.authority_generation > second.authority_generation
    with pytest.raises(CredentialError, match="unsupported_authentication"):
        asyncio.run(service.renew(first.reference_id))


def test_unsupported_oauth_does_not_import_vendor_authority():
    from tldw_chatbook.MCP.credential_bindings import (
        CredentialBindingService,
        CredentialError,
    )

    service = CredentialBindingService(MemoryCredentialBackend())
    with pytest.raises(CredentialError, match="unsupported_authentication"):
        asyncio.run(service.create("oauth"))
    assert service.backend.records == {}


def test_expiry_failed_refresh_and_reopen_do_not_restore_old_authority(binding_case):
    from tldw_chatbook.MCP.credential_bindings import (
        CredentialBindingService,
        CredentialError,
    )

    case = binding_case
    original = case.callback

    async def expired(reference):
        from dataclasses import replace

        return replace(await original(reference), expires_at=1)

    case.service.adapters["test-host"] = expired
    asyncio.run(case.service.renew(case.reference))
    current = case.binding()
    with pytest.raises(CredentialError, match="credential_expired"):
        case.service.resolve(
            case.reference, current.authority_generation, current.endpoint_origin
        )

    async def fails(reference):
        raise RuntimeError("secret-sentinel-refresh-failure")

    case.service.adapters["test-host"] = fails
    with pytest.raises(CredentialError, match="credential_refresh_failed") as error:
        asyncio.run(case.service.renew(case.reference))
    assert "secret-sentinel" not in str(error.value)
    reopened = CredentialBindingService(case.backend)
    assert reopened.binding(case.reference) == current
    with pytest.raises(CredentialError, match="credential_expired"):
        reopened.resolve(
            case.reference, current.authority_generation, current.endpoint_origin
        )


def test_refresh_loses_race_to_revocation(binding_case):
    from tldw_chatbook.MCP.credential_bindings import CredentialError

    case = binding_case
    original = case.callback

    async def race(reference):
        credential = await original(reference)
        case.service.revoke(reference)
        return credential

    case.service.adapters["test-host"] = race
    with pytest.raises(CredentialError, match="credential_changed"):
        asyncio.run(case.service.renew(case.reference))
    with pytest.raises(CredentialError, match="credential_revoked"):
        case.service.resolve(
            case.reference, case.binding().authority_generation, "https://mcp.example"
        )


@pytest.mark.parametrize(
    "headers",
    [
        {"Mcp-Method": "secret-sentinel"},
        {"Host": "attacker"},
        {"Authorization": "secret\r\nsentinel"},
        {"X-Key": "secret-sentinel-😀"},
        {"authorization": "a", "Authorization": "b"},
    ],
)
def test_header_wire_validation_is_closed_and_sanitized(headers):
    from tldw_chatbook.MCP.credential_bindings import (
        CredentialBindingService,
        CredentialError,
    )

    backend = MemoryCredentialBackend()
    service = CredentialBindingService(backend)
    with pytest.raises(CredentialError, match="credential_headers_invalid") as error:
        service.create_opaque(endpoint_origin="https://mcp.example", headers=headers)
    assert "secret" not in str(error.value)
    assert not backend.records


def test_two_service_writers_failed_storage_and_ambiguous_commit(binding_case):
    from tldw_chatbook.MCP.credential_bindings import (
        CredentialBindingService,
        CredentialError,
    )

    case = binding_case
    other = CredentialBindingService(case.backend)
    old = case.binding()
    original_write = case.backend.write

    def fail(reference, value):
        raise OSError("secret-sentinel-write-failure")

    case.backend.write = fail
    with pytest.raises(CredentialError, match="credential_storage_unavailable"):
        other.set_opaque(
            case.reference,
            endpoint_origin=old.endpoint_origin,
            headers={"X-Key": "secret-sentinel"},
        )
    assert other.binding(case.reference) == old

    def ambiguous(reference, value):
        original_write(reference, value)
        raise OSError("secret-sentinel-response-lost")

    case.backend.write = ambiguous
    with pytest.raises(CredentialError, match="credential_storage_unavailable"):
        other.set_opaque(
            case.reference,
            endpoint_origin=old.endpoint_origin,
            headers={"X-Key": "secret-sentinel"},
        )
    assert case.binding().authority_generation > old.authority_generation
    with pytest.raises(CredentialError, match="credential_changed"):
        case.service.resolve(
            case.reference, old.authority_generation, old.endpoint_origin
        )


def test_keyring_owner_uses_profile_namespace_and_cross_instance_lock(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    import keyring

    from tldw_chatbook.MCP.credential_bindings import (
        CredentialBindingService,
        KeyringCredentialBackend,
    )

    values = {}
    monkeypatch.setattr(
        "tldw_chatbook.runtime_policy.server_credentials.is_secure_keyring_backend",
        lambda backend: True,
    )
    monkeypatch.setattr(
        keyring,
        "get_password",
        lambda namespace, reference: values.get((namespace, reference)),
    )
    monkeypatch.setattr(
        keyring,
        "set_password",
        lambda namespace, reference, value: values.__setitem__(
            (namespace, reference), value
        ),
    )
    services = [
        CredentialBindingService(KeyringCredentialBackend(tmp_path / "profile"))
        for _ in range(2)
    ]

    initial = services[0].create_opaque(
        endpoint_origin="https://mcp.example", headers={"X-Key": "secret-sentinel"}
    )

    def write(index):
        return services[index % 2].set_opaque(
            initial.reference_id,
            endpoint_origin="https://mcp.example",
            headers={"X-Key": "secret-sentinel"},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        bindings = list(pool.map(write, range(12)))
    assert len({binding.authority_generation for binding in bindings}) == 12
    assert services[0].binding(initial.reference_id).authority_generation == max(
        item.authority_generation for item in bindings
    )
    assert len(values) == 1
    namespace, _ = next(iter(values))
    assert namespace.startswith("tldw_chatbook.mcp_credentials.")
    assert KeyringCredentialBackend(tmp_path / "other").namespace != namespace
    assert not (tmp_path / "profile" / "mcp_credentials.lock").read_bytes()


def test_insecure_keyring_never_receives_secrets(tmp_path, monkeypatch):
    import keyring
    from keyring.backends.null import Keyring

    from tldw_chatbook.MCP.credential_bindings import (
        CredentialBindingService,
        CredentialError,
        KeyringCredentialBackend,
    )

    monkeypatch.setattr(keyring, "get_keyring", Keyring)

    def unexpected(*args):
        pytest.fail("insecure keyring must not receive secret material")

    monkeypatch.setattr(keyring, "set_password", unexpected)
    with pytest.raises(CredentialError, match="credential_storage_unavailable"):
        CredentialBindingService(KeyringCredentialBackend(tmp_path)).create_opaque(
            endpoint_origin="https://mcp.example", headers={"X-Key": "secret-sentinel"}
        )


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_store_migrates_without_fabricating_credentials(tmp_path, version):
    import json

    from tldw_chatbook.MCP.local_store import LocalMCPStore

    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": version,
                "profiles": [{"profile_id": "old", "command": "python"}],
            }
        )
    )
    profile = LocalMCPStore(path).get_profile("old")
    assert profile.credential_reference is profile.credential_generation is None
    migrated = path.read_bytes()
    assert json.loads(migrated)["schema_version"] == 3
    assert LocalMCPStore(path).get_profile("old") == profile
    assert path.read_bytes() == migrated


@pytest.mark.parametrize(
    "fields",
    [
        {"credential_reference": "ref"},
        {"credential_generation": 1},
        {"credential_reference": "ref", "credential_generation": True},
        {"credential_reference": "ref", "credential_generation": "1"},
    ],
)
def test_malformed_binding_storage_is_not_rewritten(tmp_path, fields):
    import json

    from tldw_chatbook.MCP.local_store import LocalMCPStore, LocalMCPStoreLoadError

    path = tmp_path / "mcp.json"
    raw = json.dumps(
        {
            "schema_version": 3,
            "profiles": [
                {
                    "profile_id": "bad",
                    "transport": "streamable_http",
                    "url": "https://mcp.example",
                    **fields,
                }
            ],
        }
    )
    path.write_text(raw)
    with pytest.raises(LocalMCPStoreLoadError):
        LocalMCPStore(path).load()
    assert path.read_text() == raw


def test_deleted_reference_cannot_be_recreated_or_inherit_old_review(binding_case):
    from tldw_chatbook.MCP.credential_bindings import CredentialError

    case = binding_case
    old = case.binding()
    case.backend.records.clear()
    with pytest.raises(CredentialError, match="credential_missing"):
        case.service.set_opaque(
            old.reference_id,
            endpoint_origin=old.endpoint_origin,
            headers={"X-Key": "new-token"},
        )
    with pytest.raises(CredentialError, match="credential_missing"):
        asyncio.run(case.service.authorize("test-host", old.reference_id))
    fresh = case.service.create_opaque(
        endpoint_origin=old.endpoint_origin, headers={"X-Key": "new-token"}
    )
    assert fresh.reference_id != old.reference_id
    assert fresh.authority_generation == 1


def test_generation_exhaustion_refuses_without_wrapping(binding_case):
    import json

    from tldw_chatbook.MCP.credential_bindings import CredentialError

    case = binding_case
    record = json.loads(case.backend.records[case.reference])
    record["binding"]["authority_generation"] = 2**63 - 1
    case.backend.records[case.reference] = json.dumps(record)
    before = case.backend.records[case.reference]
    with pytest.raises(CredentialError, match="credential_generation_exhausted"):
        case.service.set_opaque(
            case.reference,
            endpoint_origin="https://mcp.example",
            headers={"X-Key": "replacement"},
        )
    with pytest.raises(CredentialError, match="credential_generation_exhausted"):
        case.service.revoke(case.reference)
    assert case.backend.records[case.reference] == before


def test_config_factory_scopes_mcp_storage_without_provider_credentials(
    tmp_path, monkeypatch
):
    from keyring.backends.null import Keyring

    from tldw_chatbook.config import create_mcp_credential_service
    from tldw_chatbook.MCP.credential_bindings import CredentialError

    monkeypatch.setattr("keyring.get_keyring", Keyring)
    first = create_mcp_credential_service(tmp_path / "one")
    second = create_mcp_credential_service(tmp_path / "two")
    assert first.backend.namespace != second.backend.namespace
    with pytest.raises(CredentialError, match="credential_storage_unavailable"):
        first.create_opaque(
            endpoint_origin="https://mcp.example", headers={"X-Key": "secret-sentinel"}
        )


def test_credential_backend_errors_and_repr_never_expose_material(binding_case):
    from tldw_chatbook.MCP.credential_bindings import CredentialError

    case = binding_case
    assert "secret-sentinel" not in repr(asyncio.run(case.callback(case.reference)))
    case.backend.records[case.reference] = "secret-sentinel-malformed"
    with pytest.raises(CredentialError) as error:
        case.binding()
    assert str(error.value) == "credential_storage_invalid"
    assert "secret-sentinel" not in repr(error.value)


@pytest.mark.parametrize("version", [1, 2])
def test_old_schema_never_imports_credential_authority(tmp_path, version):
    import json

    from tldw_chatbook.MCP.local_store import LocalMCPStore, LocalMCPStoreLoadError

    path = tmp_path / "mcp.json"
    original = json.dumps(
        {
            "schema_version": version,
            "profiles": [
                {
                    "profile_id": "old",
                    "transport": "streamable_http",
                    "url": "https://mcp.example",
                    "credential_reference": "claimed-reference",
                    "credential_generation": 1,
                }
            ],
        }
    )
    path.write_text(original)
    with pytest.raises(LocalMCPStoreLoadError):
        LocalMCPStore(path).load()
    assert path.read_text() == original
