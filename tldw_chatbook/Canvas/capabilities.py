"""Short-lived, single-use capabilities for Canvas browser delivery."""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Literal, TypeAlias

from .limits import validate_opaque_identifier

CanvasCapabilityAction: TypeAlias = Literal[
    "shell_boot",
    "renderer_load",
    "render_plan",
    "source_read",
    "source_download",
    "bridge_prepare",
    "bridge_confirm",
]

_ACTIONS = frozenset(
    {
        "shell_boot",
        "renderer_load",
        "render_plan",
        "source_read",
        "source_download",
        "bridge_prepare",
        "bridge_confirm",
    }
)
_MAX_TTL_SECONDS = 300.0
_MAX_ACTIVE_CAPABILITIES = 512


class CanvasCapabilityError(ValueError):
    """A bounded capability refusal that never contains token material."""


@dataclass(frozen=True, slots=True)
class CanvasCapabilityScope:
    """Exact authority carried by one browser-delivery capability."""

    browser_session_id: str
    load_id: str
    conversation_session_id: str
    canvas_id: str
    revision_id: str
    action: CanvasCapabilityAction
    gateway_namespace: str = "gateway-default"
    shell_incarnation_id: str = "shell-default"

    def __post_init__(self) -> None:
        for field_name in (
            "browser_session_id",
            "load_id",
            "conversation_session_id",
            "canvas_id",
            "revision_id",
            "gateway_namespace",
            "shell_incarnation_id",
        ):
            validate_opaque_identifier(
                getattr(self, field_name), field_name=field_name.replace("_", " ")
            )
        if self.action not in _ACTIONS:
            raise CanvasCapabilityError("unsupported Canvas capability action")


@dataclass(frozen=True, slots=True, repr=False)
class CanvasCapabilityGrant:
    """A one-time bearer returned only to the trusted gateway shell."""

    token: str
    scope: CanvasCapabilityScope
    expires_in_seconds: float

    def __repr__(self) -> str:
        return (
            "CanvasCapabilityGrant(token=<redacted>, "
            f"action={self.scope.action!r}, expires_in_seconds={self.expires_in_seconds!r})"
        )


@dataclass(frozen=True, slots=True)
class _CapabilityRecord:
    digest: bytes
    scope: CanvasCapabilityScope
    expires_at: float


class CanvasCapabilityStore:
    """In-memory hash-only capability registry with exact revocation."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_active: int = _MAX_ACTIVE_CAPABILITIES,
    ) -> None:
        if (
            not isinstance(max_active, int)
            or isinstance(max_active, bool)
            or max_active < 1
        ):
            raise ValueError("max_active must be a positive integer")
        self._clock = clock
        self._max_active = max_active
        self._records: dict[bytes, _CapabilityRecord] = {}
        self._closed = False
        self._lock = RLock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def active_count(self) -> int:
        with self._lock:
            self._discard_expired()
            return len(self._records)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"CanvasCapabilityStore(active_count={len(self._records)}, "
                f"closed={self._closed})"
            )

    def issue(
        self, scope: CanvasCapabilityScope, *, ttl_seconds: float
    ) -> CanvasCapabilityGrant:
        """Mint a cryptographically random capability for one exact scope."""

        with self._lock:
            if self._closed:
                raise CanvasCapabilityError("Canvas capability store is closed")
            if not isinstance(scope, CanvasCapabilityScope):
                raise CanvasCapabilityError("invalid Canvas capability scope")
            if (
                isinstance(ttl_seconds, bool)
                or not isinstance(ttl_seconds, (int, float))
                or not math.isfinite(ttl_seconds)
                or ttl_seconds <= 0
                or ttl_seconds > _MAX_TTL_SECONDS
            ):
                raise CanvasCapabilityError("invalid Canvas capability lifetime")
            self._discard_expired()
            if len(self._records) >= self._max_active:
                raise CanvasCapabilityError("Canvas capability capacity reached")
            while True:
                token = secrets.token_urlsafe(32)
                digest = _token_digest(token)
                if digest not in self._records:
                    break
            lifetime = float(ttl_seconds)
            self._records[digest] = _CapabilityRecord(
                digest=digest,
                scope=scope,
                expires_at=self._clock() + lifetime,
            )
            return CanvasCapabilityGrant(
                token=token,
                scope=scope,
                expires_in_seconds=lifetime,
            )

    def consume(
        self,
        token: str,
        *,
        expected_scope: CanvasCapabilityScope | None = None,
        expected_action: CanvasCapabilityAction | None = None,
        expected_gateway_namespace: str | None = None,
        expected_shell_incarnation_id: str | None = None,
    ) -> CanvasCapabilityScope:
        """Consume one matching bearer exactly once."""

        with self._lock:
            if self._closed:
                raise CanvasCapabilityError("Canvas capability store is closed")
            digest = _token_digest(token)
            record = self._records.get(digest)
            if record is None or not hmac.compare_digest(record.digest, digest):
                raise CanvasCapabilityError("Canvas capability is unavailable")
            if self._clock() >= record.expires_at:
                self._records.pop(digest, None)
                raise CanvasCapabilityError("Canvas capability expired")
            if expected_scope is not None and record.scope != expected_scope:
                raise CanvasCapabilityError("Canvas capability scope mismatch")
            if expected_action is not None and record.scope.action != expected_action:
                raise CanvasCapabilityError("Canvas capability scope mismatch")
            if (
                expected_gateway_namespace is not None
                and record.scope.gateway_namespace != expected_gateway_namespace
            ):
                raise CanvasCapabilityError("Canvas capability scope mismatch")
            if (
                expected_shell_incarnation_id is not None
                and record.scope.shell_incarnation_id != expected_shell_incarnation_id
            ):
                raise CanvasCapabilityError("Canvas capability scope mismatch")
            self._records.pop(digest, None)
            return record.scope

    def revoke_load(self, browser_session_id: str, load_id: str) -> int:
        """Revoke every capability for one browser frame/load incarnation."""

        return self._revoke(
            lambda scope: (
                scope.browser_session_id == browser_session_id
                and scope.load_id == load_id
            )
        )

    def revoke_selection(
        self,
        *,
        browser_session_id: str,
        conversation_session_id: str,
        canvas_id: str,
        revision_id: str,
    ) -> int:
        """Revoke credentials for one exact selected revision."""

        return self._revoke(
            lambda scope: (
                scope.browser_session_id == browser_session_id
                and scope.conversation_session_id == conversation_session_id
                and scope.canvas_id == canvas_id
                and scope.revision_id == revision_id
            )
        )

    def revoke_browser_session(self, browser_session_id: str) -> int:
        """Revoke every credential delegated to one browser shell."""

        return self._revoke(
            lambda scope: scope.browser_session_id == browser_session_id
        )

    def close(self) -> None:
        """Revoke all credentials and permanently close the store."""

        with self._lock:
            self._records.clear()
            self._closed = True

    def _discard_expired(self) -> None:
        now = self._clock()
        expired = [
            digest
            for digest, record in self._records.items()
            if now >= record.expires_at
        ]
        for digest in expired:
            self._records.pop(digest, None)

    def _revoke(self, predicate: Callable[[CanvasCapabilityScope], bool]) -> int:
        with self._lock:
            matches = [
                digest
                for digest, record in self._records.items()
                if predicate(record.scope)
            ]
            for digest in matches:
                self._records.pop(digest, None)
            return len(matches)


def _token_digest(token: str) -> bytes:
    if not isinstance(token, str) or not token or len(token) > 256:
        raise CanvasCapabilityError("Canvas capability is unavailable")
    try:
        token_bytes = token.encode("ascii")
    except UnicodeEncodeError:
        raise CanvasCapabilityError("Canvas capability is unavailable") from None
    return hashlib.sha256(token_bytes).digest()


__all__ = [
    "CanvasCapabilityAction",
    "CanvasCapabilityError",
    "CanvasCapabilityGrant",
    "CanvasCapabilityScope",
    "CanvasCapabilityStore",
]
