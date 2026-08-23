"""Pure render state for managed-model browser surfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from tldw_chatbook.Model_Artifacts.service import ArtifactRole

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import PreflightReport
    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactDiskUsage,
        ArtifactRef,
        InstalledArtifact,
        ProvenanceClass,
    )


_GGUF_QUANTIZATION_TOKENS = (
    "IQ3_XXS",
    "IQ2_XXS",
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q3_K_S",
    "IQ4_XS",
    "IQ4_NL",
    "IQ3_XS",
    "IQ3_M",
    "IQ3_S",
    "IQ2_XS",
    "IQ2_M",
    "IQ2_S",
    "IQ1_M",
    "IQ1_S",
    "Q2_K_S",
    "Q2_K",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q6_K",
    "Q8_0",
    "BF16",
    "F16",
    "F32",
)
_GGUF_QUANTIZATION_RE = re.compile(
    rf"(?:^|[._-])({'|'.join(_GGUF_QUANTIZATION_TOKENS)})(?=[._-]|$)",
    re.IGNORECASE,
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


@dataclass(frozen=True)
class VariantGuidance:
    """Provider-neutral display facts for one selectable model variant."""

    filename: str
    total_bytes: int
    file_count: int
    source_index: int
    quantization: str | None
    summary: str
    filenames: tuple[str, ...] = ()


def variant_guidance(
    filename: str,
    *,
    total_bytes: int,
    file_count: int,
    source_index: int,
    filenames: Iterable[str] = (),
) -> VariantGuidance:
    """Build conservative filename-derived guidance for one variant.

    Args:
        filename: Exact provider-supplied candidate filename or display path.
        total_bytes: Exact total bytes across the selectable file set.
        file_count: Number of files in the selectable set.
        source_index: Stable position in the provider's source ordering.
        filenames: Additional exact filenames in the same selectable file set.

    Returns:
        Render-ready facts without a compatibility or machine-fit claim.
    """
    exact_filenames = (filename, *tuple(filenames))
    detected = {_filename_quantization(item) for item in exact_filenames}
    quantization = (
        detected.pop() if len(detected) == 1 and None not in detected else None
    )
    summary = _variant_quantization_summary(quantization)
    return VariantGuidance(
        filename=filename,
        total_bytes=total_bytes,
        file_count=file_count,
        source_index=source_index,
        quantization=quantization,
        summary=summary,
        filenames=exact_filenames,
    )


def filter_variant_guidance(
    rows: Iterable[VariantGuidance], query: str
) -> tuple[VariantGuidance, ...]:
    """Filter variant guidance by filename or recognized quantization."""
    normalized_query = query.strip().casefold()
    rows = tuple(rows)
    if not normalized_query:
        return rows

    return tuple(
        row
        for row in rows
        if any(
            normalized_query in filename.casefold()
            for filename in (row.filenames or (row.filename,))
        )
        or (
            row.quantization is not None
            and normalized_query in row.quantization.casefold()
        )
    )


def sort_variant_guidance(
    rows: Iterable[VariantGuidance], order: str
) -> tuple[VariantGuidance, ...]:
    """Sort variant guidance using one explicit user-facing order.

    Args:
        rows: Render-ready variants to sort.
        order: One of ``source``, ``size-asc``, ``size-desc``, or
            ``quantization``.

    Returns:
        A deterministically ordered tuple of variants.

    Raises:
        ValueError: If ``order`` is not supported.
    """
    rows = tuple(rows)
    if order == "source":
        return tuple(sorted(rows, key=lambda row: row.source_index))
    if order == "size-asc":
        return tuple(sorted(rows, key=lambda row: (row.total_bytes, row.source_index)))
    if order == "size-desc":
        return tuple(sorted(rows, key=lambda row: (-row.total_bytes, row.source_index)))
    if order == "quantization":
        return tuple(sorted(rows, key=_variant_quantization_sort_key))
    raise ValueError(f"Unsupported variant sort order: {order}")


def _filename_quantization(filename: str) -> str | None:
    """Return one exact token recognized in a provider path's basename."""
    basename = filename.rsplit("/", 1)[-1]
    match = _GGUF_QUANTIZATION_RE.search(basename)
    return match.group(1).upper() if match is not None else None


def _variant_quantization_sort_key(row: VariantGuidance) -> tuple[int, str, int]:
    """Place low-bit quantizations first and unknown filenames last."""
    quantization = row.quantization
    if quantization is None:
        return (99, "", row.source_index)
    bit_match = re.match(r"I?Q(\d)", quantization)
    if bit_match is not None:
        return (int(bit_match.group(1)), quantization, row.source_index)
    high_precision_rank = {"F16": 16, "BF16": 16, "F32": 32}
    return (
        high_precision_rank.get(quantization, 98),
        quantization,
        row.source_index,
    )


def _variant_quantization_summary(quantization: str | None) -> str:
    """Describe only the general compression class named by a filename token."""
    if quantization is None:
        return "No recognized quantization token in the filename."
    if quantization in {"F16", "BF16", "F32"}:
        return (
            "High-precision weights · typically larger than quantized variants "
            "of the same model."
        )
    bit_match = re.match(r"I?Q(\d)", quantization)
    bits = int(bit_match.group(1)) if bit_match is not None else 0
    importance_matrix = "importance-matrix " if quantization.startswith("IQ") else ""
    if bits <= 2:
        return (
            f"{bits}-bit {importance_matrix}quantization · for the same model, "
            "typically very compact, with a substantial fidelity trade-off."
        )
    if bits == 3:
        return (
            f"3-bit {importance_matrix}quantization · for the same model, "
            "typically compact, with a larger fidelity trade-off than 4-bit "
            "and higher variants."
        )
    if bits == 4:
        return (
            f"4-bit {importance_matrix}quantization · for the same model, "
            "typically smaller than higher-bit variants, with a greater "
            "fidelity trade-off."
        )
    if bits == 5:
        return (
            "5-bit quantization · for the same model, typically a middle ground "
            "between 4-bit size and higher-bit fidelity."
        )
    if bits == 6:
        return (
            "6-bit quantization · for the same model, typically larger than "
            "Q4/Q5 variants, with a smaller fidelity trade-off."
        )
    return (
        "8-bit quantization · for the same model, typically large, with a "
        "smaller fidelity trade-off than lower-bit variants."
    )


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


def format_mib(size_bytes: int) -> str:
    """Render a byte count as a complete MiB display string, unit included.

    This is the one place that turns a raw byte count (``PlanRow``/
    ``PlanTotals``/``InventoryRow`` fields, or a live
    ``AcquisitionProgress`` event) into display text, and it owns the unit
    as well as the number. Before this function existed, the plan panel,
    the install-progress widget, and the installed view each reimplemented
    this conversion independently and disagreed: the install-progress
    widget additionally switched to B/KiB for sub-MiB values while the
    other two always rendered MiB, so the same byte count could render as
    "512.0 KiB" in one view and "0.5 MiB" in another. The sub-MiB
    switching is deliberately dropped here -- always rendering MiB (even
    "0.0 MiB" for a tiny or zero count) is the one behaviour every caller
    now shares, so two screens showing the same report can never disagree
    about units or precision. Callers must not reimplement the conversion
    or append their own unit suffix.

    Args:
        size_bytes: A byte count (expected non-negative; formats without
            raising for any int).

    Returns:
        The value divided by 1024*1024, formatted to one decimal place
        and suffixed with the unit, e.g. ``"630.6 MiB"`` for
        ``661_191_781``, ``"0.0 MiB"`` for ``0``, and ``"1.0 MiB"`` for
        exactly ``1_048_576`` (1 MiB).
    """
    return f"{size_bytes / (1024 * 1024):.1f} MiB"


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
        activation_allowed = descriptor.role is ArtifactRole.ROOT
        if is_broken:
            action_hint = "Needs repair — Repair"
        elif descriptor.role is ArtifactRole.DEPENDENCY:
            action_hint = "Managed dependency"
        elif item.active:
            action_hint = "Active"
        elif item.ready:
            action_hint = "Ready"
        else:
            action_hint = "Installed · activation required"
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
            action_hint="Outside Chatbook · integrity unknown",
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
        TransferFailureCode,
    )

    if isinstance(exc, InsufficientSpaceError):
        return "Not enough free disk space for this install."
    if isinstance(exc, GatedRepositoryError):
        return (
            "This model's repository requires a credential. Configure "
            "HUGGINGFACE_API_KEY (or HF_TOKEN) and retry."
        )
    if isinstance(exc, AcquisitionBusyError):
        return (
            f"Another {model_label} install is already in progress. Try again shortly."
        )
    if isinstance(exc, ConsentMismatchError):
        return "The install plan changed. Retry Install to review the current plan."
    if isinstance(exc, PreflightNotGrantableError):
        return "This install plan cannot proceed. Retry Install to review the current plan."
    if isinstance(exc, CatalogError):
        return f"The {model_label} download source is misconfigured."
    if isinstance(exc, TransferError):
        if exc.code is TransferFailureCode.VERIFICATION_FAILED:
            return (
                "Package verification failed (size or SHA-256). No package was "
                "promoted. Select Retry install."
            )
        if exc.code is TransferFailureCode.SOURCE_UNAVAILABLE:
            return (
                "Pinned source unavailable — the app may be offline. Select Retry "
                "install when connectivity returns."
            )
        if exc.code is TransferFailureCode.LOCAL_STATE:
            if not exc.retryable:
                return (
                    "Package install found conflicting or invalid local state. "
                    "Review or Repair the local model store before installing again."
                )
            return "Package install could not access local state. Select Retry install."
        if exc.code is TransferFailureCode.SOURCE_BLOCKED:
            return (
                "Package install is blocked by local source-access policy. Review "
                "network policy, then select Retry install."
            )
        if exc.retryable:
            return "The download was interrupted. Retry Install to resume."
        return "The download failed and cannot be retried automatically."
    return f"{model_label} install failed. See the application log for details."
