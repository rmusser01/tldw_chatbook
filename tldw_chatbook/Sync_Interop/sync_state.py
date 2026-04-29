"""State shapes for sync dry-run/readiness flows.

This module intentionally models sync state only. It does not enqueue replay work
or dispatch remote mutations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from tldw_chatbook.runtime_policy.server_parity_models import FrozenJSONDict, SourceAuthority

_SOURCE_AUTHORITIES = {"local", "server"}


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
    workspace_id: str | None = None
    enabled_domains: set[str] = field(default_factory=set)
    conflict_policy: ConflictPolicy = field(default_factory=ConflictPolicy.default)

    def __post_init__(self) -> None:
        if not self.server_profile_id:
            raise ValueError("server_profile_id is required")


class SyncProfileStateStore:
    """In-memory profile state keyed by server profile and workspace IDs."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, str | None], SyncProfileState] = {}

    def get_or_create(
        self,
        server_profile_id: str,
        *,
        workspace_id: str | None = None,
    ) -> SyncProfileState:
        if not server_profile_id:
            raise ValueError("server_profile_id is required")
        key = (server_profile_id, workspace_id)
        if key not in self._states:
            self._states[key] = SyncProfileState(
                server_profile_id=server_profile_id,
                workspace_id=workspace_id,
            )
        return self._states[key]

    def get(
        self,
        server_profile_id: str,
        *,
        workspace_id: str | None = None,
    ) -> SyncProfileState | None:
        return self._states.get((server_profile_id, workspace_id))


@dataclass(frozen=True, slots=True)
class RemotePullCursor:
    """Remote pull cursor scoped to source, server profile, workspace, domain, and collection."""

    server_profile_id: str | None
    domain: str
    remote_collection: str
    cursor: str | None = None
    workspace_id: str | None = None
    source_authority: SourceAuthority = "server"

    def __post_init__(self) -> None:
        if self.source_authority not in _SOURCE_AUTHORITIES:
            raise ValueError("source_authority must be one of: local, server")
        if self.source_authority == "server" and not self.server_profile_id:
            raise ValueError("server_profile_id is required for server remote pull cursors")
        for field_name in ("domain", "remote_collection"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} is required")

    def storage_key(self) -> str:
        server_key = self.server_profile_id or "none"
        workspace_key = self.workspace_id or "none"
        return (
            f"{self.source_authority}:{server_key}:{workspace_key}:"
            f"{self.domain}:{self.remote_collection}"
        )


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
        object.__setattr__(self, "payload", FrozenJSONDict(self.payload))

    def storage_key(self) -> str:
        workspace_key = self.workspace_id or "none"
        return f"{self.server_profile_id}:{self.domain}:{workspace_key}:{self.entry_id}"
