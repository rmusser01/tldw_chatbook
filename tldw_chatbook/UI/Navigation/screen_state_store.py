"""Memory-only ownership for cross-visit screen snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import threading
from typing import Any

from ...runtime_policy.types import RuntimeSourceState


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Runtime scope used to decide whether a view snapshot is compatible."""

    active_source: str
    active_server_id: str | None = None

    def __post_init__(self) -> None:
        if self.active_source not in {"local", "server"}:
            raise ValueError("runtime source must be canonical")
        if self.active_server_id is not None and not isinstance(
            self.active_server_id,
            str,
        ):
            raise TypeError("active server ID must be a string or None")
        normalized_server_id = (
            self.active_server_id.strip()
            if self.active_source == "server" and self.active_server_id
            else None
        )
        object.__setattr__(self, "active_server_id", normalized_server_id or None)

    @classmethod
    def from_state(cls, state: RuntimeSourceState) -> RuntimeIdentity:
        """Derive the snapshot scope from authoritative runtime state."""
        source = "server" if state.active_source == "server" else "local"
        return cls(
            active_source=source,
            active_server_id=(state.active_server_id if source == "server" else None),
        )


@dataclass(slots=True)
class _SnapshotEnvelope:
    canonical_route: str
    snapshot: dict[str, Any]
    runtime_identity: RuntimeIdentity


class ScreenStateStore:
    """Own detached outer snapshot mappings for one application process."""

    def __init__(self) -> None:
        self._owner_thread_id = threading.get_ident()
        self._entries: dict[str, _SnapshotEnvelope] = {}

    def save(
        self,
        route: str,
        snapshot: Mapping[str, Any],
        runtime_identity: RuntimeIdentity,
    ) -> None:
        """Replace the snapshot for one canonical route."""
        self._assert_owner_thread()
        canonical_route = self._canonical_key(route)
        if not isinstance(snapshot, Mapping):
            raise TypeError("screen snapshot must be a mapping")
        self._require_runtime_identity(runtime_identity)
        detached_snapshot = dict(snapshot)
        self._entries[canonical_route] = _SnapshotEnvelope(
            canonical_route=canonical_route,
            snapshot=detached_snapshot,
            runtime_identity=runtime_identity,
        )

    def restore(
        self,
        route: str,
        runtime_identity: RuntimeIdentity,
    ) -> dict[str, Any] | None:
        """Return an outer copy when the stored runtime scope is compatible."""
        self._assert_owner_thread()
        canonical_route = self._canonical_key(route)
        self._require_runtime_identity(runtime_identity)
        envelope = self._entries.get(canonical_route)
        if not self._compatible(
            envelope,
            canonical_route=canonical_route,
            runtime_identity=runtime_identity,
        ):
            self._entries.pop(canonical_route, None)
            return None
        return dict(envelope.snapshot)

    def discard(self, route: str) -> None:
        """Forget one canonical route, if present."""
        self._assert_owner_thread()
        canonical_route = self._canonical_key(route)
        self._entries.pop(canonical_route, None)

    def has_snapshots(self, runtime_identity: RuntimeIdentity) -> bool:
        """Return whether any compatible snapshot remains."""
        self._assert_owner_thread()
        self._require_runtime_identity(runtime_identity)
        for canonical_route in tuple(self._entries):
            envelope = self._entries.get(canonical_route)
            if not self._compatible(
                envelope,
                canonical_route=canonical_route,
                runtime_identity=runtime_identity,
            ):
                self._entries.pop(canonical_route, None)
        return bool(self._entries)

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("screen state store access requires the owner thread")

    @staticmethod
    def _canonical_key(route: str) -> str:
        if not isinstance(route, str):
            raise TypeError("canonical route must be a string")
        canonical_route = route.strip()
        if not canonical_route:
            raise ValueError("canonical route must not be empty")
        return canonical_route

    @staticmethod
    def _require_runtime_identity(runtime_identity: RuntimeIdentity) -> None:
        if not isinstance(runtime_identity, RuntimeIdentity):
            raise TypeError("runtime identity must be a RuntimeIdentity")

    @staticmethod
    def _compatible(
        envelope: object,
        *,
        canonical_route: str,
        runtime_identity: RuntimeIdentity,
    ) -> bool:
        return (
            isinstance(envelope, _SnapshotEnvelope)
            and envelope.canonical_route == canonical_route
            and isinstance(envelope.snapshot, dict)
            and isinstance(envelope.runtime_identity, RuntimeIdentity)
            and envelope.runtime_identity == runtime_identity
        )
