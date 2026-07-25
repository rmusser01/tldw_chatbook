"""Pure namespace, opaque identity, and keyed fingerprint contracts."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from enum import Enum
from typing import Annotated, Any, Literal, Protocol

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from .citation_trace_models import EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX
from tldw_chatbook.runtime_policy.server_credentials import (
    is_secure_keyring_backend,
)


CITATION_FINGERPRINT_KEYRING_SERVICE = "tldw_chatbook.citation-provenance.v1"
MINIMUM_FINGERPRINT_SECRET_BYTES = 32
_FINGERPRINT_PROTOCOL = b"tldw_chatbook.citation-provenance.fingerprint.v1"
_OPAQUE_PREFIX = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")


def _bounded_identifier(value: str) -> str:
    if not value:
        raise ValueError("identifier must not be empty")
    byte_count = len(value.encode("utf-8"))
    if byte_count > EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX:
        raise ValueError(
            f"identifier exceeds {EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX} UTF-8 bytes"
        )
    return value


BoundedIdentifier = Annotated[str, AfterValidator(_bounded_identifier)]


class CitationIdentityNamespace(str, Enum):
    """Disjoint identity namespaces used by provenance contracts."""

    LOCAL_TRACE = "local_trace_v1"
    SERVER_TRACE = "server_trace_v1"
    IMPORTED_TRACE = "imported_trace_v1"
    PAYLOAD = "payload_v1"
    OWNER = "owner_v1"


class CitationFingerprintDomain(str, Enum):
    """Domains that must never share an unqualified fingerprint."""

    MESSAGE_BODY = "message_body_v1"
    RAW_QUERY = "raw_query_v1"
    EXACT_PAYLOAD = "exact_payload_v1"
    OWNER_OPERATION = "owner_operation_v1"
    LOCAL_RETRY = "local_retry_v1"
    SERVER_WIRE = "server_wire_v1"
    MESSAGE_OWNER = "message_owner_v1"
    CACHE_OWNER = "cache_owner_v1"
    IMPORTED_TRACE = "imported_trace_v1"
    LEGACY_SOURCE = "legacy_source_v1"
    IMPORT_PACKAGE = "import_package_v1"
    SYNC_OPERATION = "sync_operation_v1"


class LocalCitationIdentityContext(BaseModel):
    """Stable local identities persisted later by the repository layer."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    profile_id: BoundedIdentifier
    local_authority_id: BoundedIdentifier
    fingerprint_key_id: BoundedIdentifier


class TraceNamespace(BaseModel):
    """Authority-scoped namespace for a local, server, or imported trace."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    identity_namespace: CitationIdentityNamespace
    profile_id: BoundedIdentifier
    origin_scope_id: BoundedIdentifier
    authority_id: BoundedIdentifier
    authenticated_tenant_id: BoundedIdentifier | None = None
    wire_schema_version: BoundedIdentifier
    trace_id: BoundedIdentifier | None = None
    server_trace_id: BoundedIdentifier | None = None
    import_package_fingerprint: BoundedIdentifier | None = None
    external_trace_id: BoundedIdentifier | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "TraceNamespace":
        if self.identity_namespace is CitationIdentityNamespace.LOCAL_TRACE:
            if self.origin_scope_id != self.profile_id:
                raise ValueError("local origin scope must equal profile_id")
            if self.trace_id is None:
                raise ValueError("local namespace requires trace_id")
            if any(
                value is not None
                for value in (
                    self.authenticated_tenant_id,
                    self.server_trace_id,
                    self.import_package_fingerprint,
                    self.external_trace_id,
                )
            ):
                raise ValueError("local namespace cannot carry external scope")
        elif self.identity_namespace is CitationIdentityNamespace.SERVER_TRACE:
            expected_scope = self.authenticated_tenant_id or "authority-root"
            if self.origin_scope_id != expected_scope:
                raise ValueError(
                    "server origin scope must match authenticated tenant scope"
                )
            if self.server_trace_id is None:
                raise ValueError("server namespace requires server_trace_id")
            if (
                self.trace_id is not None
                or self.import_package_fingerprint is not None
                or self.external_trace_id is not None
            ):
                raise ValueError("server namespace cannot carry local/import identity")
        elif self.identity_namespace is CitationIdentityNamespace.IMPORTED_TRACE:
            if self.trace_id is None:
                raise ValueError("imported namespace requires local trace_id")
            if self.authenticated_tenant_id is not None:
                raise ValueError("imported namespace cannot carry server tenant scope")
            if self.server_trace_id is not None:
                raise ValueError(
                    "imported namespace cannot carry server trace identity"
                )
            if self.origin_scope_id != self.import_package_fingerprint:
                raise ValueError("imported origin scope must match package fingerprint")
            if (
                self.import_package_fingerprint is not None
                and self.external_trace_id is not None
            ):
                return self
            raise ValueError(
                "imported namespace requires package and external trace IDs"
            )
        else:
            raise ValueError("TraceNamespace requires a trace identity namespace")
        return self


def new_opaque_id(prefix: str) -> str:
    """Create a prefixed opaque identifier backed by 128 random bits."""

    if _OPAQUE_PREFIX.fullmatch(prefix) is None:
        raise ValueError(
            "prefix must be 1-32 lowercase ASCII letters, digits, '_' or '-'"
        )
    return f"{prefix}_{secrets.token_hex(16)}"


def local_trace_namespace(
    context: LocalCitationIdentityContext,
    *,
    trace_id: str,
    wire_schema_version: str = "citation_trace/v1",
) -> TraceNamespace:
    """Construct the local authority namespace without persistence access."""

    return TraceNamespace(
        identity_namespace=CitationIdentityNamespace.LOCAL_TRACE,
        profile_id=context.profile_id,
        origin_scope_id=context.profile_id,
        authority_id=context.local_authority_id,
        wire_schema_version=wire_schema_version,
        trace_id=trace_id,
    )


def server_trace_namespace(
    *,
    profile_id: str,
    connection_authority_id: str,
    authenticated_tenant_id: str | None,
    wire_schema_version: str,
    server_trace_id: str,
) -> TraceNamespace:
    """Construct an authenticated server namespace without transport access."""

    return TraceNamespace(
        identity_namespace=CitationIdentityNamespace.SERVER_TRACE,
        profile_id=profile_id,
        origin_scope_id=authenticated_tenant_id or "authority-root",
        authority_id=connection_authority_id,
        authenticated_tenant_id=authenticated_tenant_id,
        wire_schema_version=wire_schema_version,
        server_trace_id=server_trace_id,
    )


def imported_trace_namespace(
    *,
    profile_id: str,
    import_authority_id: str,
    import_package_fingerprint: str,
    external_trace_id: str,
    wire_schema_version: str,
    trace_id: str,
) -> TraceNamespace:
    """Construct an inert imported namespace without rebinding its authority."""

    return TraceNamespace(
        identity_namespace=CitationIdentityNamespace.IMPORTED_TRACE,
        profile_id=profile_id,
        origin_scope_id=import_package_fingerprint,
        authority_id=import_authority_id,
        wire_schema_version=wire_schema_version,
        trace_id=trace_id,
        import_package_fingerprint=import_package_fingerprint,
        external_trace_id=external_trace_id,
    )


def _namespace_fingerprint_parts(namespace: TraceNamespace) -> tuple[str, ...]:
    """Return the complete, ordered namespace without serializing it publicly."""

    return (
        str(namespace.schema_version),
        namespace.identity_namespace.value,
        namespace.profile_id,
        namespace.origin_scope_id,
        namespace.authority_id,
        namespace.authenticated_tenant_id or "",
        namespace.wire_schema_version,
        namespace.trace_id or "",
        namespace.server_trace_id or "",
        namespace.import_package_fingerprint or "",
        namespace.external_trace_id or "",
    )


def _owner_parts(
    namespace: TraceNamespace,
    *,
    message_id: str,
    message_revision: int,
) -> tuple[str, ...]:
    if isinstance(message_revision, bool) or not isinstance(message_revision, int):
        raise TypeError("message_revision must be an integer")
    if message_revision < 0:
        raise ValueError("message_revision must be a non-negative integer")
    return (
        *_namespace_fingerprint_parts(namespace),
        _bounded_identifier(message_id),
        str(message_revision),
    )


def local_retry_idempotency_key(
    codec: CitationFingerprintCodec,
    namespace: TraceNamespace,
) -> str:
    """Derive an opaque retry key for one stable local trace identity."""

    if namespace.identity_namespace is not CitationIdentityNamespace.LOCAL_TRACE:
        raise ValueError("local retry requires a local trace namespace")
    return codec.fingerprint(
        CitationFingerprintDomain.LOCAL_RETRY,
        *_namespace_fingerprint_parts(namespace),
    )


def server_wire_idempotency_key(
    codec: CitationFingerprintCodec,
    namespace: TraceNamespace,
) -> str:
    """Derive an opaque key for one authenticated server wire identity."""

    if namespace.identity_namespace is not CitationIdentityNamespace.SERVER_TRACE:
        raise ValueError("server wire identity requires a server trace namespace")
    return codec.fingerprint(
        CitationFingerprintDomain.SERVER_WIRE,
        *_namespace_fingerprint_parts(namespace),
    )


def message_owner_idempotency_key(
    codec: CitationFingerprintCodec,
    namespace: TraceNamespace,
    *,
    message_id: str,
    message_revision: int,
) -> str:
    """Derive the normal message-owner operation key."""

    return codec.fingerprint(
        CitationFingerprintDomain.MESSAGE_OWNER,
        *_owner_parts(
            namespace,
            message_id=message_id,
            message_revision=message_revision,
        ),
    )


def cache_owner_idempotency_key(
    codec: CitationFingerprintCodec,
    namespace: TraceNamespace,
    *,
    message_id: str,
    message_revision: int,
) -> str:
    """Derive the disjoint cache-reuse owner operation key."""

    return codec.fingerprint(
        CitationFingerprintDomain.CACHE_OWNER,
        *_owner_parts(
            namespace,
            message_id=message_id,
            message_revision=message_revision,
        ),
    )


def imported_trace_idempotency_key(
    codec: CitationFingerprintCodec,
    namespace: TraceNamespace,
) -> str:
    """Derive the dormant imported-trace deduplication key."""

    if namespace.identity_namespace is not CitationIdentityNamespace.IMPORTED_TRACE:
        raise ValueError("import identity requires an imported trace namespace")
    return codec.fingerprint(
        CitationFingerprintDomain.IMPORTED_TRACE,
        str(namespace.schema_version),
        namespace.identity_namespace.value,
        namespace.profile_id,
        namespace.origin_scope_id,
        namespace.authority_id,
        namespace.wire_schema_version,
        namespace.import_package_fingerprint or "",
        namespace.external_trace_id or "",
    )


def sync_operation_idempotency_key(
    codec: CitationFingerprintCodec,
    namespace: TraceNamespace,
    *,
    sync_operation_id: str,
) -> str:
    """Derive the dormant Sync operation key without performing a write."""

    return codec.fingerprint(
        CitationFingerprintDomain.SYNC_OPERATION,
        *_namespace_fingerprint_parts(namespace),
        _bounded_identifier(sync_operation_id),
    )


class CitationFingerprintCodec:
    """HMAC-SHA-256 codec with explicit framing and domain separation."""

    __slots__ = ("_secret",)

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes):
            raise TypeError("fingerprint secret must be bytes")
        if len(secret) != MINIMUM_FINGERPRINT_SECRET_BYTES:
            raise ValueError(
                "fingerprint secret must be exactly "
                f"{MINIMUM_FINGERPRINT_SECRET_BYTES} bytes"
            )
        self._secret = secret

    def fingerprint(
        self,
        domain: CitationFingerprintDomain,
        *parts: str | bytes,
    ) -> str:
        """Return a versioned secret-scoped fingerprint for framed values."""

        if not isinstance(domain, CitationFingerprintDomain):
            raise TypeError("domain must be CitationFingerprintDomain")
        framed = bytearray()
        for part in (_FINGERPRINT_PROTOCOL, domain.value.encode("utf-8"), *parts):
            if isinstance(part, str):
                encoded = part.encode("utf-8")
            elif isinstance(part, bytes):
                encoded = part
            else:
                raise TypeError("fingerprint parts must be str or bytes")
            framed.extend(len(encoded).to_bytes(8, "big"))
            framed.extend(encoded)
        digest = hmac.new(self._secret, framed, hashlib.sha256).hexdigest()
        return f"hmac-sha256-v1:{digest}"

    def __repr__(self) -> str:
        return "CitationFingerprintCodec(secret=<redacted>)"


class CitationFingerprintKeyProvider(Protocol):
    """Injectable seam for loading an existing fingerprint secret."""

    def load_key(self, fingerprint_key_id: str) -> bytes:
        """Load existing secret bytes or fail closed."""


class CitationFingerprintKeyUnavailable(RuntimeError):
    """Raised when an existing fingerprint key cannot be loaded safely."""

    reason_code = "fingerprint_key_unavailable"


class KeyringCitationFingerprintKeyProvider:
    """Read-only production adapter for an existing base64 keyring secret."""

    def __init__(self, keyring_backend: Any | None = None) -> None:
        self._keyring_backend = keyring_backend

    def load_key(self, fingerprint_key_id: str) -> bytes:
        """Load an existing key; never generates or replaces one."""

        key_id = _bounded_identifier(fingerprint_key_id)
        backend = self._keyring_backend
        try:
            if backend is None:
                import keyring

                backend = keyring.get_keyring()
            get_keyring = getattr(backend, "get_keyring", None)
            if callable(get_keyring):
                backend = get_keyring()
            if not is_secure_keyring_backend(backend):
                raise CitationFingerprintKeyUnavailable(
                    "fingerprint_key_unavailable: insecure keyring backend"
                )
            encoded = backend.get_password(
                CITATION_FINGERPRINT_KEYRING_SERVICE,
                key_id,
            )
            if not encoded:
                raise CitationFingerprintKeyUnavailable(
                    "fingerprint_key_unavailable: key is missing"
                )
            secret = base64.b64decode(encoded, validate=True)
        except Exception:
            raise CitationFingerprintKeyUnavailable(
                CitationFingerprintKeyUnavailable.reason_code
            ) from None
        if len(secret) != MINIMUM_FINGERPRINT_SECRET_BYTES:
            raise CitationFingerprintKeyUnavailable(
                CitationFingerprintKeyUnavailable.reason_code
            ) from None
        return secret

    def __repr__(self) -> str:
        return "KeyringCitationFingerprintKeyProvider(keyring_backend=<redacted>)"


def load_fingerprint_codec(
    provider: CitationFingerprintKeyProvider,
    fingerprint_key_id: str,
) -> CitationFingerprintCodec:
    """Load an existing key through the provider and construct its codec."""

    try:
        secret = provider.load_key(_bounded_identifier(fingerprint_key_id))
        return CitationFingerprintCodec(secret)
    except Exception:
        raise CitationFingerprintKeyUnavailable(
            CitationFingerprintKeyUnavailable.reason_code
        ) from None


__all__ = [
    "CITATION_FINGERPRINT_KEYRING_SERVICE",
    "MINIMUM_FINGERPRINT_SECRET_BYTES",
    "CitationFingerprintCodec",
    "CitationFingerprintDomain",
    "CitationFingerprintKeyProvider",
    "CitationFingerprintKeyUnavailable",
    "CitationIdentityNamespace",
    "KeyringCitationFingerprintKeyProvider",
    "LocalCitationIdentityContext",
    "TraceNamespace",
    "cache_owner_idempotency_key",
    "imported_trace_namespace",
    "imported_trace_idempotency_key",
    "load_fingerprint_codec",
    "local_retry_idempotency_key",
    "local_trace_namespace",
    "message_owner_idempotency_key",
    "new_opaque_id",
    "server_trace_namespace",
    "server_wire_idempotency_key",
    "sync_operation_idempotency_key",
]
