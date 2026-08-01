"""Pure render state for managed-model browser surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import PreflightReport
    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactDiskUsage,
        ArtifactRef,
        InstalledArtifact,
        ProvenanceClass,
    )


@dataclass(frozen=True)
class PlanRow:
    """One dependency in an acquisition consent plan."""

    reference: ArtifactRef
    repository: str
    revision: str
    license_id: str
    license_url: str
    precision: str
    file_count: int
    total_bytes: int
    already_installed: bool
    provenance: str


@dataclass(frozen=True)
class PlanTotals:
    """Closure-wide disk requirements for an acquisition plan."""

    download_bytes: int
    already_staged_bytes: int
    staging_overhead_bytes: int
    retained_bytes: int
    destination: Path
    free_bytes: int
    required_bytes: int
    sufficient_space: bool
    gating_errors: tuple[str, ...]


@dataclass(frozen=True)
class UnmanagedRow:
    """One legacy model file outside the managed store."""

    path: Path
    size_bytes: int


@dataclass(frozen=True)
class InventoryRow:
    """One render-ready managed or legacy model inventory entry."""

    path: Path
    reference: ArtifactRef | None
    model_label: str
    revision: str | None
    precision: str | None
    dependencies: tuple[ArtifactRef, ...]
    ready: bool
    active: bool
    activation_allowed: bool
    is_broken: bool
    is_unmanaged: bool
    provenance: str
    action_hint: str
    error: str | None
    size_bytes: int | None
    installed_store_bytes: int | None
    staging_store_bytes: int | None
    free_bytes: int | None


def provenance_label(provenance: tuple[ProvenanceClass, ...]) -> str:
    """Describe the evidence recorded for a model without claiming safety.

    Args:
        provenance: Ordered provenance classes from a descriptor or report.

    Returns:
        Precise user-visible provenance text.
    """
    labels = {
        "chatbook_curated": "Curated by Chatbook",
        "integrity_verified": "Integrity verified",
        "local_integrity_recorded": "Local integrity recorded",
    }
    rendered = tuple(labels[item.value] for item in provenance)
    return " · ".join(rendered) if rendered else "Provenance unavailable"


def plan_rows(report: PreflightReport) -> tuple[PlanRow, ...]:
    """Map a preflight closure to render-ready dependency rows.

    Args:
        report: Immutable acquisition preflight report.

    Returns:
        One row for every model in the dependency closure.
    """
    return tuple(
        PlanRow(
            reference=entry.ref,
            repository=entry.repository,
            revision=entry.revision,
            license_id=entry.license_id,
            license_url=entry.license_url,
            precision=entry.precision,
            file_count=entry.file_count,
            total_bytes=entry.total_bytes,
            already_installed=entry.already_installed,
            provenance=provenance_label(entry.provenance),
        )
        for entry in report.entries
    )


def plan_totals(report: PreflightReport) -> PlanTotals:
    """Map closure-wide preflight values to render state.

    Args:
        report: Immutable acquisition preflight report.

    Returns:
        Render-ready disk, destination, and gating totals.
    """
    return PlanTotals(
        download_bytes=report.download_bytes,
        already_staged_bytes=report.already_staged_bytes,
        staging_overhead_bytes=report.staging_overhead_bytes,
        retained_bytes=report.retained_bytes,
        destination=report.destination,
        free_bytes=report.free_bytes,
        required_bytes=report.required_bytes,
        sufficient_space=report.sufficient_space,
        gating_errors=report.gating_errors,
    )


def inventory_rows(
    installed: Iterable[InstalledArtifact],
    usage: ArtifactDiskUsage | None,
    unmanaged: Iterable[UnmanagedRow],
) -> tuple[InventoryRow, ...]:
    """Map managed inventory and legacy files to visible rows.

    Args:
        installed: Entries returned by the managed-model service.
        usage: Current managed-store totals, if the scan succeeded.
        unmanaged: Legacy files discovered outside the managed store.

    Returns:
        Managed rows followed by visible unmanaged rows.
    """
    installed_bytes = usage.installed_bytes if usage is not None else None
    staging_bytes = usage.staging_bytes if usage is not None else None
    free_bytes = usage.free_bytes if usage is not None else None
    rows: list[InventoryRow] = []
    for item in installed:
        descriptor = item.descriptor
        if descriptor is None:
            rows.append(
                InventoryRow(
                    path=item.path,
                    reference=None,
                    model_label=item.path.name,
                    revision=None,
                    precision=None,
                    dependencies=(),
                    ready=False,
                    active=False,
                    activation_allowed=False,
                    is_broken=True,
                    is_unmanaged=False,
                    provenance="Integrity state unavailable",
                    action_hint="Unreadable manifest — Repair",
                    error=item.error,
                    size_bytes=None,
                    installed_store_bytes=installed_bytes,
                    staging_store_bytes=staging_bytes,
                    free_bytes=free_bytes,
                )
            )
            continue

        is_broken = item.error is not None
        activation_allowed = descriptor.consumer != "unassigned"
        if is_broken:
            action_hint = "Needs repair — Repair"
        elif not activation_allowed:
            action_hint = "Downloaded · runtime compatibility not verified"
        elif item.active:
            action_hint = "Active"
        elif item.ready:
            action_hint = "Ready"
        else:
            action_hint = "Unavailable until verified"
        rows.append(
            InventoryRow(
                path=item.path,
                reference=descriptor.reference,
                model_label=descriptor.model_id,
                revision=descriptor.reference.revision,
                precision=descriptor.precision,
                dependencies=descriptor.dependencies,
                ready=item.ready,
                active=item.active,
                activation_allowed=activation_allowed,
                is_broken=is_broken,
                is_unmanaged=False,
                provenance=provenance_label(descriptor.provenance),
                action_hint=action_hint,
                error=item.error,
                size_bytes=descriptor.expected_installed_bytes,
                installed_store_bytes=installed_bytes,
                staging_store_bytes=staging_bytes,
                free_bytes=free_bytes,
            )
        )

    rows.extend(
        InventoryRow(
            path=item.path,
            reference=None,
            model_label=item.path.name,
            revision=None,
            precision=None,
            dependencies=(),
            ready=False,
            active=False,
            activation_allowed=False,
            is_broken=False,
            is_unmanaged=True,
            provenance="Unmanaged — integrity unknown",
            action_hint="Import is not available yet",
            error=None,
            size_bytes=item.size_bytes,
            installed_store_bytes=installed_bytes,
            staging_store_bytes=staging_bytes,
            free_bytes=free_bytes,
        )
        for item in unmanaged
    )
    return tuple(rows)


def install_failure_message(exc: BaseException, *, model_label: str) -> str:
    """Map acquisition failures to stable, sanitized user-visible text.

    Args:
        exc: Failure raised by preflight or provisioning.
        model_label: Model name used where the message needs context.

    Returns:
        A message that does not expose raw exception details.
    """
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionBusyError,
        CatalogError,
        ConsentMismatchError,
        GatedRepositoryError,
        InsufficientSpaceError,
        PreflightNotGrantableError,
        TransferError,
    )

    if isinstance(exc, InsufficientSpaceError):
        return "Not enough free disk space for this install."
    if isinstance(exc, GatedRepositoryError):
        return (
            "This model's repository requires a credential. Configure "
            "HUGGINGFACE_API_KEY (or HF_TOKEN) and retry."
        )
    if isinstance(exc, AcquisitionBusyError):
        return f"Another {model_label} install is already in progress. Try again shortly."
    if isinstance(exc, ConsentMismatchError):
        return "The install plan changed. Retry Install to review the current plan."
    if isinstance(exc, PreflightNotGrantableError):
        return "This install plan cannot proceed. Retry Install to review the current plan."
    if isinstance(exc, CatalogError):
        return f"The {model_label} download source is misconfigured."
    if isinstance(exc, TransferError):
        if exc.retryable:
            return "The download was interrupted. Retry Install to resume."
        return "The download failed and cannot be retried automatically."
    return f"{model_label} install failed. See the application log for details."
