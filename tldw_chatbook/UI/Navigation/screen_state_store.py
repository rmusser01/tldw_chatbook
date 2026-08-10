"""Memory-only ownership for cross-visit screen snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
import threading
from typing import Any

from ...runtime_policy.types import RuntimeSourceState


_SYSTEM_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ConsolePromptTargetProjection:
    """Expose only the sanitized Console target needed by Prompt insertion.

    Attributes:
        target_session_id: Exact current Console session identity.
        system_fingerprint: One-way fingerprint of exact current System text.
    """

    target_session_id: str
    system_fingerprint: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.target_session_id) is not str
            or not self.target_session_id.strip()
            or self.target_session_id != self.target_session_id.strip()
        ):
            raise ValueError("Console prompt target session is invalid")
        if (
            not isinstance(self.system_fingerprint, str)
            or _SYSTEM_FINGERPRINT.fullmatch(self.system_fingerprint) is None
        ):
            raise ValueError("Console prompt target System fingerprint is invalid")


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


@dataclass(slots=True)
class _ConsolePromptTargetEnvelope:
    canonical_route: str
    projection: ConsolePromptTargetProjection
    runtime_identity: RuntimeIdentity


class ScreenStateStore:
    """Own detached outer snapshot mappings for one application process."""

    def __init__(self) -> None:
        self._owner_thread_id = threading.get_ident()
        self._entries: dict[str, _SnapshotEnvelope] = {}
        self._console_prompt_targets: dict[str, _ConsolePromptTargetEnvelope] = {}

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
        self._console_prompt_targets.pop(canonical_route, None)

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
            self._discard_route(canonical_route)
            return None
        target = self._console_prompt_targets.get(canonical_route)
        if target is not None and not self._target_compatible(
            target,
            canonical_route=canonical_route,
            runtime_identity=runtime_identity,
        ):
            self._discard_route(canonical_route)
            return None
        return dict(envelope.snapshot)

    def publish_console_prompt_target(
        self,
        route: str,
        projection: ConsolePromptTargetProjection,
        runtime_identity: RuntimeIdentity,
    ) -> None:
        """Publish a detached sanitized Console target for one route/runtime.

        Args:
            route: Canonical Console route.
            projection: Sanitized live Console target projection.
            runtime_identity: Authoritative runtime scope at publication.

        Raises:
            RuntimeError: If called outside the owner thread.
            TypeError: If the projection or runtime identity has the wrong type.
            ValueError: If the route is blank or has no compatible snapshot.
        """
        self._assert_owner_thread()
        canonical_route = self._canonical_key(route)
        if not isinstance(projection, ConsolePromptTargetProjection):
            raise TypeError(
                "Console prompt target must be a ConsolePromptTargetProjection"
            )
        self._require_runtime_identity(runtime_identity)
        snapshot = self._entries.get(canonical_route)
        if not self._compatible(
            snapshot,
            canonical_route=canonical_route,
            runtime_identity=runtime_identity,
        ):
            self._discard_route(canonical_route)
            raise ValueError(
                "Console prompt target requires a compatible screen snapshot"
            )
        self._console_prompt_targets[canonical_route] = _ConsolePromptTargetEnvelope(
            canonical_route=canonical_route,
            projection=self._copy_projection(projection),
            runtime_identity=runtime_identity,
        )

    def restore_console_prompt_target(
        self,
        route: str,
        runtime_identity: RuntimeIdentity,
    ) -> ConsolePromptTargetProjection | None:
        """Restore a detached compatible Console target, if one was published.

        Args:
            route: Canonical Console route.
            runtime_identity: Authoritative runtime scope at restoration.

        Returns:
            A detached sanitized projection, or ``None`` when unavailable.
        """
        self._assert_owner_thread()
        canonical_route = self._canonical_key(route)
        self._require_runtime_identity(runtime_identity)
        envelope = self._console_prompt_targets.get(canonical_route)
        if envelope is None:
            return None
        if not self._target_compatible(
            envelope,
            canonical_route=canonical_route,
            runtime_identity=runtime_identity,
        ):
            self._discard_route(canonical_route)
            return None
        snapshot = self._entries.get(canonical_route)
        if not self._compatible(
            snapshot,
            canonical_route=canonical_route,
            runtime_identity=runtime_identity,
        ):
            self._discard_route(canonical_route)
            return None
        return self._copy_projection(envelope.projection)

    def discard(self, route: str) -> None:
        """Forget one canonical route, if present."""
        self._assert_owner_thread()
        canonical_route = self._canonical_key(route)
        self._discard_route(canonical_route)

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
                self._discard_route(canonical_route)
                continue
            target = self._console_prompt_targets.get(canonical_route)
            if target is not None and not self._target_compatible(
                target,
                canonical_route=canonical_route,
                runtime_identity=runtime_identity,
            ):
                self._discard_route(canonical_route)
        return bool(self._entries)

    def _discard_route(self, canonical_route: str) -> None:
        self._entries.pop(canonical_route, None)
        self._console_prompt_targets.pop(canonical_route, None)

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

    @staticmethod
    def _target_compatible(
        envelope: object,
        *,
        canonical_route: str,
        runtime_identity: RuntimeIdentity,
    ) -> bool:
        return (
            isinstance(envelope, _ConsolePromptTargetEnvelope)
            and envelope.canonical_route == canonical_route
            and isinstance(envelope.projection, ConsolePromptTargetProjection)
            and isinstance(envelope.runtime_identity, RuntimeIdentity)
            and envelope.runtime_identity == runtime_identity
        )

    @staticmethod
    def _copy_projection(
        projection: ConsolePromptTargetProjection,
    ) -> ConsolePromptTargetProjection:
        return ConsolePromptTargetProjection(
            target_session_id=projection.target_session_id,
            system_fingerprint=projection.system_fingerprint,
        )
