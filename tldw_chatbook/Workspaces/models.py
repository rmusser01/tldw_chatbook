"""Workspace operating-context domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class WorkspaceAuthority(str, Enum):
    """Source of authority for a workspace record."""

    LOCAL_ONLY = "local-only"
    SERVER_BACKED = "server-backed"
    SYNCING_TO_SERVER = "syncing-to-server"
    SYNCING_FROM_SERVER = "syncing-from-server"
    CONFLICT = "conflict"
    DETACHED = "detached"
    REMOTE_ONLY = "remote-only"
    RUNTIME_MISSING = "runtime-missing"


class WorkspaceSyncStatus(str, Enum):
    """Current local/server sync state for a workspace."""

    NOT_CONFIGURED = "not-configured"
    READY = "ready"
    SYNCING = "syncing"
    BLOCKED = "blocked"
    CONFLICT = "conflict"


class WorkspaceTransferPolicy(str, Enum):
    """How an item should move during an explicit handoff."""

    COPY = "copy"
    REFERENCE = "reference"
    METADATA_ONLY = "metadata-only"
    LOCAL_ONLY = "local-only"


class RuntimeBindingKind(str, Enum):
    """Runtime resource type attached to a workspace."""

    LOCAL_FILESYSTEM = "local-filesystem"
    GIT_WORKTREE = "git-worktree"
    CONTAINER = "container"
    VM = "vm"
    REMOTE_RUNTIME = "remote-runtime"
    ACP_SESSION = "acp-session"


class RuntimeBindingStatus(str, Enum):
    """Readiness state for a runtime binding."""

    READY = "ready"
    MISSING = "missing"
    INSPECT_ONLY = "inspect-only"
    BLOCKED = "blocked"


class WorkspaceOperation(str, Enum):
    """Workspace-sensitive operations used by eligibility checks."""

    BROWSE = "browse"
    SEARCH = "search"
    OPEN = "open"
    EDIT = "edit"
    STAGE_IN_CONSOLE = "stage_in_console"
    RAG_GROUND = "rag_ground"
    AGENT_MANIPULATE = "agent_manipulate"
    TOOL_USE = "tool_use"


_SECRET_METADATA_PARTS = (
    "api_key",
    "apikey",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)

DEFAULT_WORKSPACE_ID = "workspace-default"
DEFAULT_WORKSPACE_NAME = "Default"
DEFAULT_WORKSPACE_DESCRIPTION = (
    "Built-in local chat workspace. Filesystem and runtime bindings are disabled "
    "until the user creates an explicit workspace."
)


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string for registry records."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WorkspaceRecord:
    """One user-visible workspace."""

    workspace_id: str
    name: str
    description: str = ""
    authority: WorkspaceAuthority | str = WorkspaceAuthority.LOCAL_ONLY
    sync_status: WorkspaceSyncStatus | str = WorkspaceSyncStatus.NOT_CONFIGURED
    active: bool = False
    archived: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_id", _required_text(self.workspace_id, "workspace_id"))
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        object.__setattr__(self, "description", _optional_text(self.description))
        object.__setattr__(
            self,
            "authority",
            _coerce_enum(self.authority, WorkspaceAuthority, "authority"),
        )
        object.__setattr__(
            self,
            "sync_status",
            _coerce_enum(self.sync_status, WorkspaceSyncStatus, "sync_status"),
        )


@dataclass(frozen=True)
class WorkspaceMembership:
    """Association between a visible item and a workspace."""

    workspace_id: str
    item_type: str
    item_id: str
    role: str = "source"
    transfer_policy: WorkspaceTransferPolicy | str = WorkspaceTransferPolicy.REFERENCE
    title: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    membership_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_id", _required_text(self.workspace_id, "workspace_id"))
        object.__setattr__(self, "item_type", _required_text(self.item_type, "item_type"))
        object.__setattr__(self, "item_id", _required_text(self.item_id, "item_id"))
        object.__setattr__(self, "role", _required_text(self.role, "role"))
        object.__setattr__(self, "title", _optional_text(self.title))
        object.__setattr__(
            self,
            "transfer_policy",
            _coerce_enum(self.transfer_policy, WorkspaceTransferPolicy, "transfer_policy"),
        )
        if self.membership_id is not None:
            object.__setattr__(
                self,
                "membership_id",
                _required_text(self.membership_id, "membership_id"),
            )


@dataclass(frozen=True)
class WorkspaceRuntimeBinding:
    """Runtime resource associated with a workspace."""

    workspace_id: str
    binding_id: str
    binding_kind: RuntimeBindingKind | str
    label: str
    locator: str
    status: RuntimeBindingStatus | str = RuntimeBindingStatus.INSPECT_ONLY
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_id", _required_text(self.workspace_id, "workspace_id"))
        object.__setattr__(self, "binding_id", _required_text(self.binding_id, "binding_id"))
        object.__setattr__(self, "label", _required_text(self.label, "label"))
        object.__setattr__(self, "locator", _required_text(self.locator, "locator"))
        object.__setattr__(
            self,
            "binding_kind",
            _coerce_enum(self.binding_kind, RuntimeBindingKind, "binding_kind"),
        )
        object.__setattr__(
            self,
            "status",
            _coerce_enum(self.status, RuntimeBindingStatus, "status"),
        )
        object.__setattr__(self, "metadata", scrub_secret_metadata(dict(self.metadata)))


@dataclass(frozen=True)
class WorkspaceEligibility:
    """Decision for a workspace-sensitive item operation."""

    visible: bool
    active_context_eligible: bool
    reason_code: str
    recovery_copy: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _required_text(self.reason_code, "reason_code"))
        object.__setattr__(self, "recovery_copy", _optional_text(self.recovery_copy))


def scrub_secret_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return metadata with secret-looking keys removed recursively."""

    scrubbed: dict[str, Any] = {}
    for key, item in value.items():
        if _is_secret_metadata_key(str(key)):
            continue
        scrubbed[str(key)] = _scrub_metadata_value(item)
    return scrubbed


def _scrub_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return scrub_secret_metadata(value)
    if isinstance(value, list):
        return [_scrub_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_metadata_value(item) for item in value)
    return value


def _is_secret_metadata_key(key: str) -> bool:
    key_normalized = _normalize_metadata_key(key)
    return any(
        _normalize_metadata_key(part) in key_normalized
        for part in _SECRET_METADATA_PARTS
    )


def _normalize_metadata_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _optional_text(value: str | None) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("text value must be text")
    return value.strip()


def _coerce_enum(value: Any, enum_type: type[Enum], field_name: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is invalid") from exc
