from __future__ import annotations

import base64
import hashlib
import logging
import re
import traceback

import pytest
from pydantic import ValidationError

from tldw_chatbook.Chat.citation_trace_identity import (
    CITATION_FINGERPRINT_KEYRING_SERVICE,
    MINIMUM_FINGERPRINT_SECRET_BYTES,
    CitationFingerprintCodec,
    CitationFingerprintDomain,
    CitationFingerprintKeyUnavailable,
    CitationIdentityNamespace,
    KeyringCitationFingerprintKeyProvider,
    LocalCitationIdentityContext,
    TraceNamespace,
    cache_owner_idempotency_key,
    imported_trace_namespace,
    imported_trace_idempotency_key,
    load_fingerprint_codec,
    local_retry_idempotency_key,
    local_trace_namespace,
    message_owner_idempotency_key,
    new_opaque_id,
    server_trace_namespace,
    server_wire_idempotency_key,
    sync_operation_idempotency_key,
)


class FakeProvider:
    def __init__(self, secret: bytes | None) -> None:
        self.secret = secret
        self.calls: list[str] = []

    def load_key(self, fingerprint_key_id: str) -> bytes:
        self.calls.append(fingerprint_key_id)
        if self.secret is None:
            raise CitationFingerprintKeyUnavailable("missing")
        return self.secret


class FakeKeyring:
    def __init__(self, value: str | None = None, *, error: Exception | None = None):
        self.value = value
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def get_password(self, service: str, account: str) -> str | None:
        self.calls.append((service, account))
        if self.error is not None:
            raise self.error
        return self.value


class SecureFakeKeyring(FakeKeyring):
    __module__ = "keyring.backends.macOS"
    priority = 5


class InsecureFakeKeyring(FakeKeyring):
    __module__ = "keyring.backends.file"
    priority = 1


class WritableSecureFakeKeyring(SecureFakeKeyring):
    __module__ = "keyring.backends.macOS"

    def __init__(self, value: str | None = None, *, error: Exception | None = None):
        super().__init__(value, error=error)
        self.set_calls: list[tuple[str, str, str]] = []

    def set_password(self, service: str, account: str, value: str) -> None:
        self.set_calls.append((service, account, value))
        self.value = value


class UnwritableSecureFakeKeyring(WritableSecureFakeKeyring):
    def set_password(self, service: str, account: str, value: str) -> None:
        raise RuntimeError("keyring is read-only")


def test_opaque_ids_use_128_random_bits_with_stable_bounded_prefixes() -> None:
    first = new_opaque_id("trace")
    second = new_opaque_id("trace")

    assert first != second
    assert re.fullmatch(r"trace_[0-9a-f]{32}", first)
    assert len(bytes.fromhex(first.removeprefix("trace_"))) == 16
    with pytest.raises(ValueError, match="prefix"):
        new_opaque_id("")
    with pytest.raises(ValueError, match="prefix"):
        new_opaque_id("UPPER")
    with pytest.raises(ValueError, match="prefix"):
        new_opaque_id("x" * 33)


def test_local_server_and_imported_namespaces_separate_authority_scopes() -> None:
    context_a = LocalCitationIdentityContext(
        profile_id="profile-a",
        local_authority_id="authority-local",
        fingerprint_key_id="key-a",
    )
    context_b = context_a.model_copy(update={"profile_id": "profile-b"})
    local_a = local_trace_namespace(context_a, trace_id="trace-a")
    local_a_other_trace = local_trace_namespace(context_a, trace_id="trace-b")
    local_b = local_trace_namespace(context_b, trace_id="trace-a")
    server_a = server_trace_namespace(
        profile_id="profile-a",
        connection_authority_id="server-authority",
        authenticated_tenant_id="tenant-a",
        wire_schema_version="grounding_trace/v1",
        server_trace_id="server-trace-a",
    )
    server_b = server_trace_namespace(
        profile_id="profile-a",
        connection_authority_id="server-authority",
        authenticated_tenant_id="tenant-b",
        wire_schema_version="grounding_trace/v1",
        server_trace_id="server-trace-a",
    )
    server_v2 = server_trace_namespace(
        profile_id="profile-a",
        connection_authority_id="server-authority",
        authenticated_tenant_id="tenant-a",
        wire_schema_version="grounding_trace/v2",
        server_trace_id="server-trace-a",
    )
    server_other_trace = server_trace_namespace(
        profile_id="profile-a",
        connection_authority_id="server-authority",
        authenticated_tenant_id="tenant-a",
        wire_schema_version="grounding_trace/v1",
        server_trace_id="server-trace-b",
    )
    imported = imported_trace_namespace(
        profile_id="profile-a",
        import_authority_id="import-authority",
        import_package_fingerprint="package-fingerprint",
        external_trace_id="external-trace",
        wire_schema_version="portable/v1",
        trace_id="imported-local-trace",
    )

    assert (
        len(
            {
                local_a,
                local_a_other_trace,
                local_b,
                server_a,
                server_b,
                server_v2,
                server_other_trace,
                imported,
            }
        )
        == 8
    )
    assert local_a.identity_namespace is CitationIdentityNamespace.LOCAL_TRACE
    assert server_a.identity_namespace is CitationIdentityNamespace.SERVER_TRACE
    assert imported.identity_namespace is CitationIdentityNamespace.IMPORTED_TRACE
    assert CitationIdentityNamespace.PAYLOAD.value == "payload_v1"
    assert CitationIdentityNamespace.OWNER.value == "owner_v1"

    with pytest.raises(ValidationError, match="UTF-8 bytes"):
        server_trace_namespace(
            profile_id="profile-a",
            connection_authority_id="é" * 129,
            authenticated_tenant_id="tenant",
            wire_schema_version="v1",
            server_trace_id="server-trace",
        )


def test_hostile_direct_namespace_construction_rejects_incoherent_origin_fields() -> (
    None
):
    context = LocalCitationIdentityContext(
        profile_id="profile",
        local_authority_id="local-authority",
        fingerprint_key_id="key",
    )
    local = local_trace_namespace(context, trace_id="trace")
    server = server_trace_namespace(
        profile_id="profile",
        connection_authority_id="server-authority",
        authenticated_tenant_id="tenant",
        wire_schema_version="grounding_trace/v1",
        server_trace_id="server-trace",
    )
    imported = imported_trace_namespace(
        profile_id="profile",
        import_authority_id="import-authority",
        import_package_fingerprint="package",
        external_trace_id="external",
        wire_schema_version="portable/v1",
        trace_id="local-import-trace",
    )

    hostile_updates = (
        (local, {"server_trace_id": "server-trace"}),
        (server, {"trace_id": "local-trace"}),
        (server, {"external_trace_id": "external"}),
        (imported, {"server_trace_id": "server-trace"}),
        (imported, {"trace_id": None}),
        (imported, {"origin_scope_id": "another-package"}),
    )
    for namespace, update in hostile_updates:
        with pytest.raises(ValidationError):
            TraceNamespace(**{**namespace.model_dump(), **update})


def test_fingerprint_framing_prevents_delimiter_ambiguity_and_separates_domains() -> (
    None
):
    codec = CitationFingerprintCodec(b"k" * MINIMUM_FINGERPRINT_SECRET_BYTES)

    assert codec.fingerprint(CitationFingerprintDomain.RAW_QUERY, "a|b", "c") != (
        codec.fingerprint(CitationFingerprintDomain.RAW_QUERY, "a", "b|c")
    )
    fingerprints = {
        codec.fingerprint(domain, "identical input")
        for domain in CitationFingerprintDomain
    }
    assert len(fingerprints) == len(CitationFingerprintDomain)


def test_fingerprint_codec_rejects_weak_secrets_and_never_exposes_sensitive_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = "private answer text"
    secret = b"s" * MINIMUM_FINGERPRINT_SECRET_BYTES
    with pytest.raises(ValueError, match="exactly"):
        CitationFingerprintCodec(b"")
    with pytest.raises(ValueError, match="exactly"):
        CitationFingerprintCodec(secret[:-1])
    with pytest.raises(ValueError, match="exactly"):
        CitationFingerprintCodec(secret + b"x")

    codec = CitationFingerprintCodec(secret)
    with caplog.at_level(logging.DEBUG):
        fingerprint = codec.fingerprint(CitationFingerprintDomain.MESSAGE_BODY, raw)

    assert raw not in fingerprint
    assert secret.hex() not in fingerprint
    assert hashlib.sha256(raw.encode()).hexdigest() not in fingerprint
    assert raw not in repr(codec)
    assert secret.hex() not in repr(codec)
    assert not caplog.records


def test_injected_key_provider_loads_lazily_and_fails_closed() -> None:
    secret = b"k" * MINIMUM_FINGERPRINT_SECRET_BYTES
    provider = FakeProvider(secret)

    assert provider.calls == []
    codec = load_fingerprint_codec(provider, "key-1")
    assert provider.calls == ["key-1"]
    assert codec.fingerprint(CitationFingerprintDomain.EXACT_PAYLOAD, b"value")

    missing = FakeProvider(None)
    with pytest.raises(
        CitationFingerprintKeyUnavailable,
        match="fingerprint_key_unavailable",
    ):
        load_fingerprint_codec(missing, "missing-key")


def test_keyring_adapter_uses_fixed_service_and_never_touches_real_keyring() -> None:
    secret = b"k" * MINIMUM_FINGERPRINT_SECRET_BYTES
    backend = SecureFakeKeyring(base64.b64encode(secret).decode("ascii"))
    provider = KeyringCitationFingerprintKeyProvider(keyring_backend=backend)

    assert backend.calls == []
    assert provider.load_key("key-1") == secret
    assert backend.calls == [(CITATION_FINGERPRINT_KEYRING_SERVICE, "key-1")]
    assert "redacted" in repr(provider).lower()


def test_keyring_adapter_provisions_one_missing_secret_and_reuses_it() -> None:
    backend = WritableSecureFakeKeyring()
    provider = KeyringCitationFingerprintKeyProvider(keyring_backend=backend)

    secret = provider.provision_key("key-1")

    assert len(secret) == MINIMUM_FINGERPRINT_SECRET_BYTES
    assert backend.set_calls == [
        (
            CITATION_FINGERPRINT_KEYRING_SERVICE,
            "key-1",
            base64.b64encode(secret).decode("ascii"),
        )
    ]
    assert provider.provision_key("key-1") == secret
    assert len(backend.set_calls) == 1


@pytest.mark.parametrize(
    "backend",
    (
        SecureFakeKeyring(None),
        WritableSecureFakeKeyring("not-base64"),
        WritableSecureFakeKeyring(error=RuntimeError("keyring unavailable")),
        UnwritableSecureFakeKeyring(),
        InsecureFakeKeyring(None),
    ),
)
def test_keyring_adapter_provisioning_failures_never_replace_existing_state(
    backend: FakeKeyring,
) -> None:
    provider = KeyringCitationFingerprintKeyProvider(keyring_backend=backend)

    with pytest.raises(
        CitationFingerprintKeyUnavailable,
        match="fingerprint_key_unavailable",
    ):
        provider.provision_key("key-1")

    assert getattr(backend, "set_calls", []) == []


@pytest.mark.parametrize(
    "backend",
    (
        SecureFakeKeyring(None),
        SecureFakeKeyring("not-base64"),
        SecureFakeKeyring(error=RuntimeError("keyring unavailable")),
        InsecureFakeKeyring(base64.b64encode(b"k" * 32).decode("ascii")),
        SecureFakeKeyring(base64.b64encode(b"k" * 31).decode("ascii")),
        SecureFakeKeyring(base64.b64encode(b"k" * 33).decode("ascii")),
    ),
)
def test_keyring_missing_invalid_or_unavailable_fails_closed(
    backend: FakeKeyring,
) -> None:
    provider = KeyringCitationFingerprintKeyProvider(keyring_backend=backend)

    with pytest.raises(
        CitationFingerprintKeyUnavailable,
        match="fingerprint_key_unavailable",
    ):
        provider.load_key("key-1")


def test_keyring_failure_traceback_never_includes_backend_message() -> None:
    sentinel = "backend-secret-sentinel"
    provider = KeyringCitationFingerprintKeyProvider(
        keyring_backend=SecureFakeKeyring(error=RuntimeError(sentinel))
    )

    with pytest.raises(CitationFingerprintKeyUnavailable) as captured:
        provider.load_key("key-1")

    assert str(captured.value) == CitationFingerprintKeyUnavailable.reason_code
    assert sentinel not in "".join(traceback.format_exception(captured.value))


def test_fingerprint_provider_failure_traceback_never_includes_provider_message() -> (
    None
):
    sentinel = "provider-secret-sentinel"

    class LeakyProvider:
        def load_key(self, fingerprint_key_id: str) -> bytes:
            raise RuntimeError(f"{sentinel}:{fingerprint_key_id}")

    with pytest.raises(CitationFingerprintKeyUnavailable) as captured:
        load_fingerprint_codec(LeakyProvider(), "key-1")

    assert str(captured.value) == CitationFingerprintKeyUnavailable.reason_code
    assert sentinel not in "".join(traceback.format_exception(captured.value))


def test_idempotency_keys_are_stable_domain_separated_and_opaque() -> None:
    codec = CitationFingerprintCodec(b"k" * MINIMUM_FINGERPRINT_SECRET_BYTES)
    context = LocalCitationIdentityContext(
        profile_id="profile",
        local_authority_id="local-authority",
        fingerprint_key_id="key",
    )
    local = local_trace_namespace(context, trace_id="trace")
    imported = imported_trace_namespace(
        profile_id="profile",
        import_authority_id="import-authority",
        import_package_fingerprint="secret-scoped-package",
        external_trace_id="external-trace",
        wire_schema_version="portable/v1",
        trace_id="imported-local-trace",
    )

    builders = (
        lambda: local_retry_idempotency_key(codec, local),
        lambda: message_owner_idempotency_key(
            codec,
            local,
            message_id="message",
            message_revision=1,
        ),
        lambda: cache_owner_idempotency_key(
            codec,
            local,
            message_id="message",
            message_revision=1,
        ),
        lambda: imported_trace_idempotency_key(codec, imported),
        lambda: sync_operation_idempotency_key(
            codec,
            local,
            sync_operation_id="sync-operation",
        ),
    )
    first = tuple(builder() for builder in builders)
    second = tuple(builder() for builder in builders)

    assert first == second
    assert len(set(first)) == len(first)
    assert all(value.startswith("hmac-sha256-v1:") for value in first)
    serialized = repr(first)
    for sensitive in (
        "profile",
        "local-authority",
        "trace",
        "message",
        "secret-scoped-package",
        "external-trace",
        "sync-operation",
    ):
        assert sensitive not in serialized


def test_server_wire_idempotency_binds_every_authenticated_scope_component() -> None:
    codec = CitationFingerprintCodec(b"k" * MINIMUM_FINGERPRINT_SECRET_BYTES)

    def key(
        *,
        profile_id: str = "profile-a",
        authority_id: str = "authority-a",
        tenant_id: str | None = "tenant-a",
        wire_version: str = "grounding_trace/v1",
    ) -> str:
        namespace = server_trace_namespace(
            profile_id=profile_id,
            connection_authority_id=authority_id,
            authenticated_tenant_id=tenant_id,
            wire_schema_version=wire_version,
            server_trace_id="shared-external-id",
        )
        return server_wire_idempotency_key(codec, namespace)

    keys = {
        key(),
        key(profile_id="profile-b"),
        key(authority_id="authority-b"),
        key(tenant_id="tenant-b"),
        key(tenant_id=None),
        key(wire_version="grounding_trace/v2"),
    }

    assert len(keys) == 6
    assert all("shared-external-id" not in value for value in keys)


def test_idempotency_external_ids_enforce_exact_utf8_boundaries() -> None:
    codec = CitationFingerprintCodec(b"k" * MINIMUM_FINGERPRINT_SECRET_BYTES)
    context = LocalCitationIdentityContext(
        profile_id="profile",
        local_authority_id="authority",
        fingerprint_key_id="key",
    )
    namespace = local_trace_namespace(context, trace_id="trace")

    assert sync_operation_idempotency_key(
        codec,
        namespace,
        sync_operation_id="é" * 128,
    )
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        sync_operation_idempotency_key(
            codec,
            namespace,
            sync_operation_id=("é" * 128) + "x",
        )
    with pytest.raises(TypeError, match="message_revision"):
        message_owner_idempotency_key(
            codec,
            namespace,
            message_id="message",
            message_revision="1",  # type: ignore[arg-type]
        )


def test_import_idempotency_ignores_retry_local_id_but_binds_external_origin() -> None:
    codec = CitationFingerprintCodec(b"k" * MINIMUM_FINGERPRINT_SECRET_BYTES)

    def key(
        *,
        local_trace_id: str,
        authority_id: str = "import-authority",
        external_trace_id: str = "external-trace",
    ) -> str:
        return imported_trace_idempotency_key(
            codec,
            imported_trace_namespace(
                profile_id="profile",
                import_authority_id=authority_id,
                import_package_fingerprint="secret-scoped-package",
                external_trace_id=external_trace_id,
                wire_schema_version="portable/v1",
                trace_id=local_trace_id,
            ),
        )

    assert key(local_trace_id="allocated-a") == key(local_trace_id="allocated-b")
    assert key(local_trace_id="allocated-a") != key(
        local_trace_id="allocated-b",
        authority_id="other-authority",
    )
    assert key(local_trace_id="allocated-a") != key(
        local_trace_id="allocated-b",
        external_trace_id="other-external-trace",
    )
