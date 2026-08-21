"""Screen-independent Persona Buddy state leases and priority resolution."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from threading import RLock
from typing import Literal

from .preferences import PersonaBuddySelection


_BUILTIN_STATES = frozenset(
    {
        "idle",
        "listening",
        "thinking",
        "speaking",
        "approval_needed",
        "tool_running",
        "wake_armed",
        "offline",
        "error",
    }
)
_CUSTOM_STATE_PATTERN = re.compile(r"[a-z][a-z0-9_.:-]{0,95}\Z")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_UNSAFE_STATE_PREFIXES = (
    "env:",
    "file:",
    "ftp:",
    "http:",
    "https:",
    "proc:",
    "ssh:",
)
_UNSAFE_STATE_MARKERS = (
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
)
_LIVE_STATES = frozenset({"idle", "listening", "thinking", "speaking"})
_MAX_TTL_SECONDS = 3600.0

_LeaseKind = Literal[
    "error",
    "approval",
    "timed",
    "authored",
    "tool",
    "wake",
    "voice",
    "offline",
    "idle",
]


class PersonaBuddyStateError(ValueError):
    """A fixed-category error for untrusted controller state input."""

    __slots__ = ("category",)

    def __init__(self, category: str = "persona_buddy_state_invalid") -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class PersonaBuddyLeaseToken:
    """Exact capability for one source-and-owner lease revision."""

    source: str
    owner: str
    state: str
    lease_id: int
    expires_at: float | None = None


@dataclass(frozen=True, slots=True)
class PersonaBuddySnapshot:
    """Immutable controller state safe for app-owned consumers."""

    generation: int
    selection: PersonaBuddySelection | None
    state: str
    state_source: str | None = None
    state_owner: str | None = None


@dataclass(frozen=True, slots=True)
class _StateLease:
    token: PersonaBuddyLeaseToken
    kind: _LeaseKind


def _require_identifier(value: object) -> str:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise PersonaBuddyStateError()
    return value


def _require_state(value: object) -> str:
    if type(value) is not str or _CUSTOM_STATE_PATTERN.fullmatch(value) is None:
        raise PersonaBuddyStateError()
    if value in _BUILTIN_STATES:
        return value
    if value.startswith(_UNSAFE_STATE_PREFIXES):
        raise PersonaBuddyStateError()
    compact = re.sub(r"[._:-]+", "_", value)
    if any(marker in compact for marker in _UNSAFE_STATE_MARKERS):
        raise PersonaBuddyStateError()
    return value


def _require_expiration(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PersonaBuddyStateError()
    expiration = float(value)
    if not math.isfinite(expiration):
        raise PersonaBuddyStateError()
    return expiration


def _lease_kind(source: str, state: str, expires_at: float | None) -> _LeaseKind:
    if state == "error":
        return "error"
    if state == "approval_needed":
        return "approval"
    if source == "timed" or expires_at is not None:
        return "timed"
    if source == "authored":
        return "authored"
    if state == "tool_running":
        return "tool"
    if state == "wake_armed":
        return "wake"
    if state in _LIVE_STATES:
        return "voice" if state != "idle" else "idle"
    if state == "offline":
        return "offline"
    return "authored"


class PersonaBuddyController:
    """Own one explicit selection and source-scoped operational-state leases."""

    def __init__(
        self,
        *,
        selection: PersonaBuddySelection | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if selection is not None and type(selection) is not PersonaBuddySelection:
            raise PersonaBuddyStateError()
        self._lock = RLock()
        self._clock = clock
        self._selection = selection
        self._leases: dict[tuple[str, str], _StateLease] = {}
        self._generation = 0
        self._next_lease_id = 0

    def snapshot(self) -> PersonaBuddySnapshot:
        """Return the current immutable selection and resolved state."""

        with self._lock:
            self._discard_expired_locked()
            lease = self._resolve_locked()
            if lease is None:
                return PersonaBuddySnapshot(
                    generation=self._generation,
                    selection=self._selection,
                    state="idle",
                )
            token = lease.token
            return PersonaBuddySnapshot(
                generation=self._generation,
                selection=self._selection,
                state=token.state,
                state_source=token.source,
                state_owner=token.owner,
            )

    def select_local_persona(self, persona_id: str) -> int:
        """Explicitly replace the selected local Persona and return generation."""

        selection = PersonaBuddySelection("local", persona_id)
        with self._lock:
            if selection != self._selection:
                self._selection = selection
                self._generation += 1
            return self._generation

    def observe_persona(self, *, source: str, persona_id: object) -> int:
        """Observe UI/runtime selection without changing Buddy authority."""

        _require_identifier(source)
        del persona_id
        with self._lock:
            return self._generation

    def acquire_state(
        self,
        *,
        source: str,
        owner: str,
        state: str,
        expires_at: float | None = None,
    ) -> PersonaBuddyLeaseToken:
        """Acquire or replace one exact source-and-owner state lease."""

        normalized_source = _require_identifier(source)
        normalized_owner = _require_identifier(owner)
        normalized_state = _require_state(state)
        normalized_expiration = _require_expiration(expires_at)
        with self._lock:
            self._discard_expired_locked()
            self._next_lease_id += 1
            token = PersonaBuddyLeaseToken(
                source=normalized_source,
                owner=normalized_owner,
                state=normalized_state,
                lease_id=self._next_lease_id,
                expires_at=normalized_expiration,
            )
            self._leases[(normalized_source, normalized_owner)] = _StateLease(
                token=token,
                kind=_lease_kind(
                    normalized_source,
                    normalized_state,
                    normalized_expiration,
                ),
            )
            self._generation += 1
            return token

    def release_state(
        self,
        *,
        token: PersonaBuddyLeaseToken | None = None,
        source: str | None = None,
        owner: str | None = None,
    ) -> bool:
        """Release only an exact token or an exact source-and-owner pair."""

        with self._lock:
            self._discard_expired_locked()
            if token is not None:
                if type(token) is not PersonaBuddyLeaseToken:
                    return False
                key = (token.source, token.owner)
                current = self._leases.get(key)
                if current is None or current.token != token:
                    return False
            else:
                if source is None or owner is None:
                    return False
                try:
                    key = (_require_identifier(source), _require_identifier(owner))
                except PersonaBuddyStateError:
                    return False
                current = self._leases.get(key)
                if current is None:
                    return False
            del self._leases[key]
            self._generation += 1
            return True

    def set_timed_state(
        self,
        *,
        owner: str,
        state: str,
        ttl_seconds: float,
    ) -> PersonaBuddyLeaseToken:
        """Acquire a bounded explicit/custom state lease from the monotonic clock."""

        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, Real):
            raise PersonaBuddyStateError()
        ttl = float(ttl_seconds)
        if not math.isfinite(ttl) or not 0 < ttl <= _MAX_TTL_SECONDS:
            raise PersonaBuddyStateError()
        return self.acquire_state(
            source="timed",
            owner=owner,
            state=state,
            expires_at=self._clock() + ttl,
        )

    def set_authored_trigger(
        self,
        *,
        owner: str,
        state: str,
    ) -> PersonaBuddyLeaseToken:
        """Acquire or replace one authored Persona Visual trigger."""

        return self.acquire_state(source="authored", owner=owner, state=state)

    def _discard_expired_locked(self) -> None:
        now = self._clock()
        expired = [
            key
            for key, lease in self._leases.items()
            if lease.token.expires_at is not None and lease.token.expires_at <= now
        ]
        if not expired:
            return
        for key in expired:
            del self._leases[key]
        self._generation += 1

    def _resolve_locked(self) -> _StateLease | None:
        leases = tuple(self._leases.values())
        for kind in ("error", "approval", "timed", "authored", "tool"):
            if match := _newest(leases, kind):
                return match

        non_idle_voice = tuple(
            lease
            for lease in leases
            if lease.kind == "voice" and lease.token.state != "idle"
        )
        if non_idle_voice:
            return max(non_idle_voice, key=lambda lease: lease.token.lease_id)
        if wake := _newest(leases, "wake"):
            return wake
        for kind in ("voice", "idle", "offline"):
            if match := _newest(leases, kind):
                return match
        return None


def _newest(leases: tuple[_StateLease, ...], kind: str) -> _StateLease | None:
    candidates = (lease for lease in leases if lease.kind == kind)
    return max(candidates, key=lambda lease: lease.token.lease_id, default=None)
