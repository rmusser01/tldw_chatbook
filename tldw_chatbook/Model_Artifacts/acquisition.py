"""TASK-595: managed model acquisition types and catalog resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from .service import ArtifactError, ArtifactRef

if TYPE_CHECKING:
    from .service import ArtifactDescriptor


# Constants per spec (Docs/superpowers/specs/2026-07-30-managed-model-acquisition-design.md)
ACQUISITION_SAFETY_MARGIN_BYTES = 256 * 1024 * 1024
MAX_FILE_REFETCHES = 1


# Error hierarchy: all subclass ArtifactError
class AcquisitionError(ArtifactError):
    """Base error for acquisition operations."""

    pass


class CatalogError(AcquisitionError):
    """Catalog lookup, cycle, or revision-conflict error."""

    pass


class ConsentMismatchError(AcquisitionError):
    """Closure fingerprint changed between preflight and provision."""

    pass


class PreflightNotGrantableError(AcquisitionError):
    """Preflight report cannot be granted due to gating or space errors."""

    pass


class AcquisitionBusyError(AcquisitionError):
    """Another acquisition session is already active."""

    pass


class InsufficientSpaceError(AcquisitionError):
    """Insufficient free space for acquisition."""

    pass


class GatedRepositoryError(AcquisitionError):
    """Authenticated repository requires credentials or fails access."""

    pass


class TransferError(AcquisitionError):
    """Network or transfer error with optional retry flag."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        """Initialize TransferError with retryable flag.

        Args:
            message: The error message.
            retryable: Whether this error is retryable.
        """
        super().__init__(message)
        self.retryable = retryable


# Protocol for catalog descriptors
class ArtifactCatalog(Protocol):
    """Protocol for artifact descriptor catalogs."""

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        """Retrieve a descriptor by reference.

        Args:
            ref: The artifact reference.

        Returns:
            The artifact descriptor.

        Raises:
            KeyError: If the reference is not found.
        """
        ...


# Frozen dataclasses per spec
@dataclass(frozen=True)
class ArtifactPreflightEntry:
    """One artifact entry in a preflight report."""

    ref: ArtifactRef
    source_url: str
    repository: str
    revision: str
    license_id: str
    license_url: str
    precision: str
    total_bytes: int
    file_count: int
    already_installed: bool


@dataclass(frozen=True)
class AcquisitionConsent:
    """Consent to acquire a resolved closure."""

    closure_fingerprint: str


@dataclass(frozen=True)
class AcquisitionProgress:
    """Progress during an active acquisition operation."""

    phase: Literal["fetch", "pre-verify", "verify-install", "activate"]
    ref: ArtifactRef
    file: str | None
    bytes_done: int
    bytes_total: int


@dataclass(frozen=True)
class PreflightReport:
    """Preflight report for a closure before acquisition.

    Fields marked as space/bytes are in bytes. The report aggregates
    requirements and fails early on gating or space constraints before
    any actual downloads.
    """

    root: ArtifactRef
    closure_fingerprint: str
    entries: tuple[ArtifactPreflightEntry, ...]
    download_bytes: int
    already_staged_bytes: int
    staging_overhead_bytes: int
    retained_bytes: int
    destination: Path
    free_bytes: int
    required_bytes: int
    sufficient_space: bool
    gating_errors: tuple[str, ...]

    def grant(self) -> AcquisitionConsent:
        """Grant acquisition consent from this preflight report.

        Raises PreflightNotGrantableError if gating_errors is non-empty
        or sufficient_space is False.

        Returns:
            AcquisitionConsent carrying the closure fingerprint.

        Raises:
            PreflightNotGrantableError: If gating errors exist or space is insufficient.
        """
        if self.gating_errors or not self.sufficient_space:
            raise PreflightNotGrantableError(
                f"preflight not grantable: gating_errors={self.gating_errors}, "
                f"sufficient_space={self.sufficient_space}"
            )
        return AcquisitionConsent(closure_fingerprint=self.closure_fingerprint)


def resolve_catalog_closure(
    root: ArtifactRef, catalog: ArtifactCatalog
) -> tuple[ArtifactDescriptor, ...]:
    """Resolve the full dependency closure from catalog descriptors.

    Deliberately not the core's _resolve_closure (which reads installed
    manifests): at preflight, dependencies may not be installed at all.
    Same rules: cycle and revision-conflict detection; stable sorted order.

    Args:
        root: The root artifact reference to resolve from.
        catalog: The catalog providing descriptors for references.

    Returns:
        A tuple of artifact descriptors in stable sorted order (by ref).

    Raises:
        CatalogError: If an unknown ref is encountered, a dependency cycle
            is detected, or two different revisions of the same artifact_id
            appear in the closure.
    """
    resolved: dict[ArtifactRef, ArtifactDescriptor] = {}
    revisions: dict[str, ArtifactRef] = {}
    visiting: set[ArtifactRef] = set()

    def visit(ref: ArtifactRef) -> None:
        if ref in resolved:
            return
        if ref in visiting:
            raise CatalogError(f"dependency cycle at {ref.artifact_id}")
        seen = revisions.get(ref.artifact_id)
        if seen is not None and seen != ref:
            raise CatalogError(
                f"conflicting revisions for {ref.artifact_id}: {seen.revision} vs {ref.revision}"
            )
        visiting.add(ref)
        try:
            descriptor = catalog.descriptor(ref)
        except Exception as exc:
            raise CatalogError(f"unknown artifact {ref.artifact_id}@{ref.revision}") from exc
        revisions[ref.artifact_id] = ref
        for dep in descriptor.dependencies:
            visit(dep)
        visiting.discard(ref)
        resolved[ref] = descriptor

    visit(root)
    return tuple(resolved[ref] for ref in sorted(resolved))
