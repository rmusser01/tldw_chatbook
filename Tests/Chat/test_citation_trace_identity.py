from __future__ import annotations

import base64
import hashlib
import logging
import re

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
    imported_trace_namespace,
    load_fingerprint_codec,
    local_trace_namespace,
    new_opaque_id,
    server_trace_namespace,
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
    local_a = local_trace_namespace(context_a)
    local_b = local_trace_namespace(context_b)
    server_a = server_trace_namespace(
        profile_id="profile-a",
        connection_authority_id="server-authority",
        authenticated_tenant_id="tenant-a",
        wire_schema_version="grounding_trace/v1",
    )
    server_b = server_trace_namespace(
        profile_id="profile-a",
        connection_authority_id="server-authority",
        authenticated_tenant_id="tenant-b",
        wire_schema_version="grounding_trace/v1",
    )
    server_v2 = server_trace_namespace(
        profile_id="profile-a",
        connection_authority_id="server-authority",
        authenticated_tenant_id="tenant-a",
        wire_schema_version="grounding_trace/v2",
    )
    imported = imported_trace_namespace(
        profile_id="profile-a",
        import_authority_id="import-authority",
        import_package_fingerprint="package-fingerprint",
        external_trace_id="external-trace",
        wire_schema_version="portable/v1",
    )

    assert len({local_a, local_b, server_a, server_b, server_v2, imported}) == 6
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
        )


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
    with pytest.raises(ValueError, match="at least"):
        CitationFingerprintCodec(b"")
    with pytest.raises(ValueError, match="at least"):
        CitationFingerprintCodec(secret[:-1])

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
    backend = FakeKeyring(base64.b64encode(secret).decode("ascii"))
    provider = KeyringCitationFingerprintKeyProvider(keyring_backend=backend)

    assert backend.calls == []
    assert provider.load_key("key-1") == secret
    assert backend.calls == [(CITATION_FINGERPRINT_KEYRING_SERVICE, "key-1")]
    assert "redacted" in repr(provider).lower()


@pytest.mark.parametrize(
    "backend",
    (
        FakeKeyring(None),
        FakeKeyring("not-base64"),
        FakeKeyring(error=RuntimeError("keyring unavailable")),
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
