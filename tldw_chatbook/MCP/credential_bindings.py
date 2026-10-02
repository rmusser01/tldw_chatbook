"""Host-owned MCP credential authority, separate from token storage revision.

Only injected host adapters can attest identity continuity. Production currently
supports explicit opaque header/bearer values; generic MCP OAuth is unavailable.
No token parsing, imported connector grants, or plaintext fallback is performed.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit
from uuid import uuid4


class CredentialError(RuntimeError):
    """Fixed-code failure, containing no backend exception or secret fragments."""


def endpoint_origin(url: str) -> str:
    """Canonicalize an already validated HTTP endpoint to its origin."""
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        port = parsed.port
        suffix = (
            f":{port}"
            if port and port != {"http": 80, "https": 443}[parsed.scheme]
            else ""
        )
        return f"{parsed.scheme}://{host}{suffix}"
    except (ValueError, TypeError, AttributeError):
        raise CredentialError("credential_origin_invalid") from None


@dataclass(frozen=True)
class CredentialBinding:
    reference_id: str
    authority_generation: int
    method: str
    issuer: str | None
    audience: str | None
    endpoint_origin: str
    principal: str | None
    scopes: tuple[str, ...] | None


def same_authority(old: CredentialBinding, new: CredentialBinding) -> bool:
    """Compare reviewed authority, excluding token bytes/expiry/storage revision."""
    return old == new


@dataclass(frozen=True, repr=False)
class HostCredential:
    """Result from a host adapter; never deserialize this from plugin claims."""

    method: str
    endpoint_origin: str
    headers: dict[str, str] = field(repr=False)
    issuer: str | None = None
    audience: str | None = None
    principal: str | None = None
    scopes: tuple[str, ...] | None = None
    expires_at: float | None = None

    def __repr__(self) -> str:
        return "HostCredential(<private>)"


_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_RESERVED = frozenset(
    {
        "host",
        "content-type",
        "content-length",
        "accept",
        "accept-encoding",
        "connection",
        "transfer-encoding",
        "cookie",
        "proxy-authorization",
        "te",
        "trailer",
        "upgrade",
        "last-event-id",
    }
)


def _headers(headers: dict) -> dict[str, str]:
    if type(headers) is not dict or not headers or len(headers) > 32:
        raise CredentialError("credential_headers_invalid")
    result = {}
    for key, value in headers.items():
        if (
            not isinstance(key, str)
            or not _HEADER_NAME.fullmatch(key)
            or key.lower() in _RESERVED
            or key.lower().startswith(("mcp-", "proxy-", "sec-"))
            or key.lower() in result
        ):
            raise CredentialError("credential_headers_invalid")
        if (
            not isinstance(value, str)
            or len(value) > 16384
            or value != value.strip(" \t")
            or any(
                (ord(c) < 32 and c != "\t") or ord(c) == 127 or ord(c) > 255
                for c in value
            )
        ):
            raise CredentialError("credential_headers_invalid")
        result[key.lower()] = value
    return result


class KeyringCredentialBackend:
    """One atomic protected record per reference, serialized across processes."""

    _locks: ClassVar[dict[str, threading.RLock]] = {}
    _locks_guard = threading.Lock()

    def __init__(self, data_root: Path):
        self.root = Path(data_root).resolve()
        self.namespace = (
            "tldw_chatbook.mcp_credentials."
            + sha256(str(self.root).encode()).hexdigest()
        )
        with self._locks_guard:
            self._lock = self._locks.setdefault(self.namespace, threading.RLock())

    @contextmanager
    def transaction(self):
        import keyring
        import portalocker

        from tldw_chatbook.runtime_policy.server_credentials import (
            is_secure_keyring_backend,
        )

        if not is_secure_keyring_backend(keyring.get_keyring()):
            raise CredentialError("credential_storage_unavailable")
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with portalocker.Lock(str(self.root / "mcp_credentials.lock"), timeout=10):
                yield

    def read(self, reference_id: str) -> str | None:
        import keyring

        return keyring.get_password(self.namespace, reference_id)

    def write(self, reference_id: str, value: str) -> None:
        import keyring

        keyring.set_password(self.namespace, reference_id, value)


class CredentialBindingService:
    """Resolve live secrets under reviewed authority through a single owner.

    Adapters are explicitly injected trusted host callbacks, not package-selected
    identities. No production generic OAuth adapter is registered.
    """

    def __init__(
        self,
        backend,
        *,
        adapters: dict[str, Callable[[str], Awaitable[HostCredential]]] | None = None,
    ):
        self.backend = backend
        self.adapters = dict(adapters or {})
        self._worker_lock = threading.Lock()
        self._worker_future = None

    async def _offloop(self, operation):
        """One retained OS-storage operation; async waiters queue no thread jobs."""
        while True:
            with self._worker_lock:
                future = self._worker_future
                own = future is None
                if own:
                    future = Future()
                    self._worker_future = future
            if own:

                def run(result_future=future):
                    try:
                        value, error = operation(), None
                    except Exception as caught:  # noqa: BLE001
                        # Never retain the backend's exception body or traceback.
                        error = CredentialError(
                            str(caught)
                            if isinstance(caught, CredentialError)
                            else "credential_storage_unavailable"
                        )
                        value = None
                    with self._worker_lock:
                        self._worker_future = None
                    if error is None:
                        result_future.set_result(value)
                    else:
                        result_future.set_exception(error)

                try:
                    threading.Thread(
                        target=run, name="mcp-credential-storage", daemon=True
                    ).start()
                except Exception:  # noqa: BLE001 -- no worker owns this capacity
                    with self._worker_lock:
                        self._worker_future = None
                    future.set_exception(
                        CredentialError("credential_storage_unavailable")
                    )
            wrapped = asyncio.wrap_future(future)
            # A timed-out caller leaves the real worker retained. Consume its
            # eventual failure without exposing backend details or warning noise.
            wrapped.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
            try:
                value = await asyncio.shield(wrapped)
            except Exception:
                if own:
                    raise
            else:
                if own:
                    return value
            # Another caller's result is never authority for this caller. After
            # real completion claim capacity and make a fresh protected lookup.

    async def resolve_async(
        self, reference_id: str, expected_generation: int, origin: str
    ) -> dict:
        """Resolve for async transport without blocking its event loop."""
        return await self._offloop(
            lambda: self.resolve(reference_id, expected_generation, origin)
        )

    def _renew_state(self, reference_id, *, allow_revoked=False):
        with self._transaction():
            record = self._read(reference_id)
            if record is None:
                raise CredentialError("credential_missing")
            if record["revoked"] and not allow_revoked:
                raise CredentialError("credential_revoked")
            return record["adapter"], record["storage_revision"]

    @contextmanager
    def _transaction(self):
        try:
            with self.backend.transaction():
                yield
        except CredentialError:
            raise
        except Exception:  # noqa: BLE001 -- backend failures must not disclose secrets
            raise CredentialError("credential_storage_unavailable") from None

    def _read(self, reference_id: str) -> dict | None:
        if not isinstance(reference_id, str) or not re.fullmatch(
            r"[A-Za-z0-9._-]{1,200}", reference_id
        ):
            raise CredentialError("credential_reference_invalid")
        raw = self.backend.read(reference_id)
        if raw is None:
            return None
        try:
            record = json.loads(raw)
            if (
                record["schema"] != 1
                or type(record["storage_revision"]) is not int
                or record["storage_revision"] < 1
                or type(record["revoked"]) is not bool
            ):
                raise ValueError
            binding = self._binding(record)
            if (
                binding.reference_id != reference_id
                or type(binding.authority_generation) is not int
                or not 1 <= binding.authority_generation < 2**63
            ):
                raise ValueError
            return record
        except (ValueError, TypeError, KeyError):
            raise CredentialError("credential_storage_invalid") from None

    @staticmethod
    def _binding(record: dict) -> CredentialBinding:
        data = dict(record["binding"])
        if data["scopes"] is not None:
            data["scopes"] = tuple(data["scopes"])
        return CredentialBinding(**data)

    def binding(self, reference_id: str) -> CredentialBinding:
        """Read current metadata, including a revocation tombstone's generation."""
        with self._transaction():
            record = self._read(reference_id)
            if record is None:
                raise CredentialError("credential_missing")
            return self._binding(record)

    def _publish(
        self, reference_id, credential, adapter, *, expected_revision=None, create=False
    ):
        if not isinstance(credential, HostCredential) or credential.method not in {
            "headers",
            "bearer",
        }:
            raise CredentialError("unsupported_authentication")
        headers = _headers(credential.headers)
        if credential.method == "bearer" and (
            set(headers) != {"authorization"}
            or not headers["authorization"].startswith("Bearer ")
            or len(headers["authorization"]) <= 7
        ):
            raise CredentialError("credential_headers_invalid")
        origin = endpoint_origin(credential.endpoint_origin)
        if credential.endpoint_origin != origin:
            raise CredentialError("credential_origin_invalid")
        for value in (credential.issuer, credential.audience, credential.principal):
            if value is not None and (
                not isinstance(value, str) or not value or len(value) > 2048
            ):
                raise CredentialError("credential_identity_invalid")
        scopes = credential.scopes
        if scopes is not None:
            if not isinstance(scopes, (tuple, list)) or any(
                not isinstance(scope, str) or not scope or len(scope) > 256
                for scope in scopes
            ):
                raise CredentialError("credential_identity_invalid")
            scopes = tuple(sorted(set(scopes)))
        expiry = credential.expires_at
        if expiry is not None and (
            type(expiry) not in (int, float) or not math.isfinite(expiry)
        ):
            raise CredentialError("credential_expiry_invalid")
        with self._transaction():
            previous = self._read(reference_id)
            if create and previous is not None:
                raise CredentialError("credential_changed")
            if not create and previous is None:
                raise CredentialError("credential_missing")
            revision = previous["storage_revision"] if previous else 0
            if expected_revision is not None and revision != expected_revision:
                raise CredentialError("credential_changed")
            generation = self._binding(previous).authority_generation if previous else 0
            candidate = CredentialBinding(
                reference_id,
                generation,
                credential.method,
                credential.issuer,
                credential.audience,
                origin,
                credential.principal,
                scopes,
            )
            verified = adapter is not None and all(
                value is not None
                for value in (
                    credential.issuer,
                    credential.audience,
                    credential.principal,
                    scopes,
                )
            )
            continuous = (
                previous
                and not previous["revoked"]
                and previous["adapter"] == adapter
                and verified
                and previous["verified"]
                and same_authority(self._binding(previous), candidate)
                and previous["header_names"] == sorted(headers)
            )
            if not continuous:
                if generation >= 2**63 - 1:
                    raise CredentialError("credential_generation_exhausted")
                candidate = replace(candidate, authority_generation=generation + 1)
            record = {
                "schema": 1,
                "binding": asdict(candidate),
                "headers": headers,
                "header_names": sorted(headers),
                "expires_at": expiry,
                "storage_revision": revision + 1,
                "adapter": adapter,
                "verified": verified,
                "revoked": False,
            }
            self.backend.write(reference_id, json.dumps(record))
            return candidate

    def set_opaque(
        self,
        reference_id: str,
        *,
        endpoint_origin: str,
        headers: dict[str, str],
        method: str = "headers",
        expires_at: float | None = None,
    ) -> CredentialBinding:
        """Explicit manual mapping/replacement always requires a new review."""
        return self._publish(
            reference_id,
            HostCredential(method, endpoint_origin, headers, expires_at=expires_at),
            None,
        )

    def create_opaque(
        self,
        *,
        endpoint_origin: str,
        headers: dict[str, str],
        method: str = "headers",
        expires_at: float | None = None,
    ) -> CredentialBinding:
        """Mint a fresh host reference; a lost reference can never be recreated."""
        return self._publish(
            uuid4().hex,
            HostCredential(method, endpoint_origin, headers, expires_at=expires_at),
            None,
            create=True,
        )

    async def create(self, adapter: str) -> CredentialBinding:
        """Mint a fresh reference from a registered host authentication adapter."""
        reference = uuid4().hex
        credential = await self._authenticate(adapter, reference)
        return await self._offloop(
            lambda: self._publish(
                reference, credential, adapter, expected_revision=0, create=True
            )
        )

    async def _authenticate(self, adapter, reference_id):
        callback = self.adapters.get(adapter)
        if callback is None:
            raise CredentialError("unsupported_authentication")
        try:
            return await callback(reference_id)
        except Exception:  # noqa: BLE001 -- sanitize host errors
            raise CredentialError("credential_refresh_failed") from None

    async def authorize(self, adapter: str, reference_id: str) -> CredentialBinding:
        """Explicitly reauthorize an existing reference through a host adapter."""
        _, revision = await self._offloop(
            lambda: self._renew_state(reference_id, allow_revoked=True)
        )
        credential = await self._authenticate(adapter, reference_id)
        return await self._offloop(
            lambda: self._publish(
                reference_id, credential, adapter, expected_revision=revision
            )
        )

    async def renew(self, reference_id: str) -> CredentialBinding:
        """Renew authentication only; never invoke or replay any MCP operation."""
        adapter, revision = await self._offloop(lambda: self._renew_state(reference_id))
        credential = await self._authenticate(adapter, reference_id)
        return await self._offloop(
            lambda: self._publish(
                reference_id, credential, adapter, expected_revision=revision
            )
        )

    def revoke(self, reference_id: str) -> None:
        """Retain monotonic authority/storage tombstones, without secret material."""
        with self._transaction():
            record = self._read(reference_id)
            if record is None:
                raise CredentialError("credential_missing")
            if record["binding"]["authority_generation"] >= 2**63 - 1:
                raise CredentialError("credential_generation_exhausted")
            record["binding"]["authority_generation"] += 1
            record["storage_revision"] += 1
            record.update(revoked=True, headers={})
            self.backend.write(reference_id, json.dumps(record))

    def validate_reference(
        self, reference_id: str, expected_generation: int, origin: str
    ) -> CredentialBinding:
        """Validate metadata without returning secrets to snapshot/recovery owners."""
        with self._transaction():
            record = self._validated(reference_id, expected_generation, origin)
            return self._binding(record)

    def _validated(self, reference_id, expected_generation, origin):
        record = self._read(reference_id)
        if record is None:
            raise CredentialError("credential_missing")
        if record["revoked"]:
            raise CredentialError("credential_revoked")
        binding = self._binding(record)
        if (
            type(expected_generation) is not int
            or binding.authority_generation != expected_generation
            or binding.endpoint_origin != origin
        ):
            raise CredentialError("credential_changed")
        if record["expires_at"] is not None and record["expires_at"] <= time.time():
            raise CredentialError("credential_expired")
        return record

    def resolve(
        self, reference_id: str, expected_generation: int, endpoint_origin: str
    ) -> dict:
        """Return current usable headers only at the selected transport boundary."""
        with self._transaction():
            return _headers(
                self._validated(reference_id, expected_generation, endpoint_origin)[
                    "headers"
                ]
            )

    def snapshot_binding(
        self, reference_id: str, expected_generation: int, origin: str
    ) -> dict:
        """Explicitly project MCP metadata into the existing plugin snapshot shape."""
        with self._transaction():
            record = self._validated(reference_id, expected_generation, origin)
            data = asdict(self._binding(record))
            data["authentication_method"] = data.pop("method")
            data["scopes"] = (
                list(data["scopes"]) if data["scopes"] is not None else None
            )
            data["identity_state"] = (
                "verified" if record["verified"] else "opaque_reviewed"
            )
            return data
