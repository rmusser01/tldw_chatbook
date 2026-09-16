"""Session-bound reviews of exact local-package installation inputs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .authority import PluginMarker
from .inspection import inspect_package
from .models import PackageInspection
from .package_files import canonical_json


@dataclass(frozen=True)
class PluginReview:
    """Immutable install intent; its token expires with the issuing session."""

    installation_id: str
    inspection: PackageInspection
    selection: tuple[str, ...]
    workspace_id: str | None
    authority_marker: PluginMarker
    authority_json: str
    token: str
    expires_at: float
    operation_id: str = ""
    alias: str | None = None
    kind: Literal[
        "install", "trust", "activate", "revoke", "uninstall", "update", "retain"
    ] = "install"
    intent: Literal["inherit", "enabled", "disabled"] | None = None
    previous_revision: str | None = None
    drain_token: str | None = None
    rollback: bool = False
    data_compatibility: str = "unknown"
    retired_revisions: tuple[str, ...] = ()
    retirement_json: str | None = None


@dataclass(frozen=True)
class OperationReceipt:
    """Non-secret outcome; commitment does not assert execution readiness."""

    operation_id: str | None
    phase: str
    committed: bool
    recovery_reason: str | None = None
    runtime_stopped: bool | None = None
    cleanup_pending: bool = False
    persistence_error: str | None = None
    cleanup_errors: tuple[str, ...] = ()
    request_id: str | None = None


def inspection_identity(inspection: PackageInspection) -> str:
    """Bind deterministic inspection inputs, excluding observational timestamps."""
    value = inspection.model_dump(mode="json")
    value.pop("diagnostics")
    for component in value["inventory"].values():
        component.pop("evidence")
    return canonical_json(value)


def reinspect(inspection: PackageInspection, root: Path) -> PackageInspection:
    """Verify materialized bytes while preserving authenticated source provenance."""
    observed = inspect_package(root, dialect=inspection.dialect)
    observed = observed.model_copy(
        update={
            "source_identity": inspection.source_identity,
            "source_digest": inspection.source_digest,
            "materialized_identity": inspection.materialized_identity,
            "link_targets": dict(inspection.link_targets),
        }
    )
    if inspection_identity(observed) != inspection_identity(inspection):
        raise ValueError("stale review or changed retained package")
    return observed
