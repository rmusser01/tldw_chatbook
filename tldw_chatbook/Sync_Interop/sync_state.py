"""State shapes for sync dry-run/readiness flows.

This module intentionally models sync state only. It does not enqueue replay work
or dispatch remote mutations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ConflictStrategy(str, Enum):
    """Conflict strategies available to future sync execution paths."""

    PRESERVE_LOCAL = "preserve_local"
    REMOTE_WINS = "remote_wins"
    LOCAL_WINS = "local_wins"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True, slots=True)
class ConflictPolicy:
    """Default policy is conservative for the dry-run substrate."""

    strategy: ConflictStrategy = ConflictStrategy.PRESERVE_LOCAL
    allow_remote_overwrite: bool = False

    @classmethod
    def default(cls) -> "ConflictPolicy":
        return cls()


@dataclass(slots=True)
class SyncProfileState:
    """Per-server profile state for dry-run sync discovery."""

    server_profile_id: str
    enabled_domains: set[str] = field(default_factory=set)
    conflict_policy: ConflictPolicy = field(default_factory=ConflictPolicy.default)

    def __post_init__(self) -> None:
        if not self.server_profile_id:
            raise ValueError("server_profile_id is required")


class SyncProfileStateStore:
    """In-memory profile state keyed by server profile ID."""

    def __init__(self) -> None:
        self._states: dict[str, SyncProfileState] = {}

    def get_or_create(self, server_profile_id: str) -> SyncProfileState:
        if not server_profile_id:
            raise ValueError("server_profile_id is required")
        if server_profile_id not in self._states:
            self._states[server_profile_id] = SyncProfileState(server_profile_id=server_profile_id)
        return self._states[server_profile_id]

    def get(self, server_profile_id: str) -> SyncProfileState | None:
        return self._states.get(server_profile_id)


@dataclass(frozen=True, slots=True)
class RemotePullCursor:
    """Remote pull cursor scoped to server profile, domain, and collection."""

    server_profile_id: str
    domain: str
    remote_collection: str
    cursor: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("server_profile_id", "domain", "remote_collection"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} is required")

    def storage_key(self) -> str:
        return f"{self.server_profile_id}:{self.domain}:{self.remote_collection}"


@dataclass(frozen=True, slots=True)
class LocalOutboxEntry:
    """Local outbox entry shape only; no replay or dispatch behavior."""

    entry_id: str
    server_profile_id: str
    domain: str
    workspace_id: str | None
    local_entity_id: str
    operation: str
    payload_hash: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    remote_collection: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "entry_id",
            "server_profile_id",
            "domain",
            "local_entity_id",
            "operation",
            "payload_hash",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} is required")
        object.__setattr__(self, "payload", dict(self.payload))

    def storage_key(self) -> str:
        workspace_key = self.workspace_id or "none"
        return f"{self.server_profile_id}:{self.domain}:{workspace_key}:{self.entry_id}"

