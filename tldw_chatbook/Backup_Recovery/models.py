"""Immutable local recovery inventory contracts (ADR-126)."""

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Mapping, Protocol


@dataclass(frozen=True)
class FileMetadata:
    """Versioned preview observations, never final capture or restore authority."""

    version: int
    root_id: str
    relative_path: str
    parent_id: str | None
    kind: str
    mode: int
    mtime_ns: int
    policy: str

    def __post_init__(self) -> None:
        from tldw_chatbook.Utils.path_validation import validate_recovery_relative_path

        validate_recovery_relative_path(self.relative_path)
        if (
            type(self.version) is not int
            or self.version != 1
            or self.kind not in {"file", "directory"}
            or self.policy not in {"private", "external"}
            or type(self.mode) is not int
            or not 0 <= self.mode <= 0o777
            or type(self.mtime_ns) is not int
            or type(self.root_id) is not str
            or not self.root_id
            or (self.parent_id is not None and type(self.parent_id) is not str)
        ):
            raise ValueError("invalid_file_metadata")


@dataclass(frozen=True)
class DiscoverySelections:
    """Explicit local options; operation exclusions are separately validated."""

    external_roots: tuple[Path, ...] = ()
    model_ids: tuple[str, ...] = ()
    temporary_media: bool = False
    diagnostics: bool = False
    planned_output_root: Path | None = None

    def __post_init__(self) -> None:
        if (
            type(self.external_roots) is not tuple
            or any(not isinstance(path, Path) for path in self.external_roots)
            or type(self.model_ids) is not tuple
            or any(type(value) is not str or not value for value in self.model_ids)
            or len(set(self.external_roots)) != len(self.external_roots)
            or len(set(self.model_ids)) != len(self.model_ids)
            or type(self.temporary_media) is not bool
            or type(self.diagnostics) is not bool
            or self.planned_output_root is not None
            and not isinstance(self.planned_output_root, Path)
        ):
            raise ValueError("invalid_discovery_selections")


@dataclass(frozen=True)
class StorageItem:
    owner: str
    logical_id: str
    path: Path | None
    status: str
    dependencies: tuple[str, ...]
    shared_group: str | None = None
    deletion_validated: bool = False
    metadata: FileMetadata | None = None

    def __post_init__(self) -> None:
        if type(self.dependencies) is not tuple or any(
            type(value) is not str for value in self.dependencies
        ):
            raise TypeError("immutable_dependencies_required")
        if self.metadata is not None and type(self.metadata) is not FileMetadata:
            raise TypeError("invalid_file_metadata")
        if type(self.deletion_validated) is not bool:
            raise TypeError("invalid_deletion_evidence")


@dataclass(frozen=True)
class Inventory:
    items: tuple[StorageItem, ...]
    complete: bool
    scope_digest: str
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.items) is not tuple or type(self.issues) is not tuple:
            raise TypeError("immutable_inventory_required")


@dataclass(frozen=True)
class SchemaPolicy:
    """Installed layouts; repeated version entries are exact alternatives.

    Validators match one complete ordered catalog, never a union of fragments.
    The tuple is installed code policy and cannot be populated from an archive.
    """

    owner: str
    versions: tuple[int, ...]
    schema_sql: tuple[tuple[int, tuple[str, ...]], ...]
    migration_steps: tuple[tuple[int, int, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        def immutable(value: object) -> bool:
            return type(value) in (str, int) or (
                type(value) is tuple and all(immutable(item) for item in value)
            )

        if not all(
            type(value) is tuple and immutable(value)
            for value in (self.versions, self.schema_sql, self.migration_steps)
        ):
            raise TypeError("immutable_schema_policy_required")


class OwnerAdapter(Protocol):
    owner_id: str
    activation_required: bool

    def discover(self, config: Mapping[str, object]) -> tuple[StorageItem, ...]: ...
    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None: ...
    def validate(self, candidate: Path) -> tuple[str, ...]: ...
    def relocate(self, candidate: Path, mapping: Mapping[str, Path]) -> None: ...
    def schema_policy(self) -> SchemaPolicy | None: ...


@dataclass(frozen=True)
class DiscoveryContext:
    """Locally constructed source selector, never serialized archive authority."""

    config_path: Path
    profile_id: str
    selections: DiscoverySelections = DiscoverySelections()


DISCOVERY_CONTEXT_KEY = "__chatbook_recovery_context__"


def discovery_context(config: Mapping[str, object]) -> DiscoveryContext:
    """Get the installed-created context; raw TOML cannot supply this object."""
    context = config.get(DISCOVERY_CONTEXT_KEY)
    if not isinstance(context, DiscoveryContext):
        raise ValueError("missing_discovery_context")
    return context


def storage_logical_id(
    context: DiscoveryContext, owner: str, local_id: str = ""
) -> str:
    """Format explicit profile/owner IDs, also used for cross-owner dependencies."""
    if not owner or ":" in owner or ":" in local_id:
        raise ValueError("invalid_logical_id")
    return f"profile:{context.profile_id}:{owner}" + (
        f":{local_id}" if local_id else ""
    )
