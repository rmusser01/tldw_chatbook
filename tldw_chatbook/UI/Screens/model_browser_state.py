"""Pure render state for managed-model browser surfaces."""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable
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
    usage_notice: str = ""
    role: str = ""
    purpose: str = ""


@dataclass(frozen=True)
class ComponentCopy:
    """Plain-words install-review copy for one curated component.

    TASK-34100.14 ([voice-speech-18]). A descriptor's ``usage_notice`` is
    part of its identity: it is written into every installed manifest and
    hashed into the download stage's name, and Parakeet's managed-copy path
    compares installed descriptors by equality. Rewording it would orphan
    in-flight partial downloads and make an existing install look foreign.
    So the install review substitutes this copy at display time instead.

    Attributes:
        role: Short noun for the summary's licence line ("speech model").
        purpose: What the component does, in plain words.
        notice: Replacement for the descriptor's usage notice; ``""`` shows
            none (the panel's own closing line already says every file is
            checked against its recorded size and SHA-256).
    """

    role: str
    purpose: str
    notice: str = ""


# Keyed by artifact id; Tests/UI/test_model_artifact_widgets.py pins these
# keys to the constants in Local_Ingestion/parakeet_v2_artifact.py.
_CURATED_COMPONENT_COPY: dict[str, ComponentCopy] = {
    "parakeet-v2": ComponentCopy("speech model", "turns speech into text"),
    "parakeet-v3": ComponentCopy("speech model", "turns speech into text"),
    "silero-vad": ComponentCopy(
        "voice-activity detector", "splits long recordings into chunks"
    ),
}

# Canonical SPDX spellings for the ids Hugging Face model cards publish in
# lower case ("mit", "cc-by-4.0"). Unknown ids are shown as given.
_SPDX_CANONICAL = {
    spdx.casefold(): spdx
    for spdx in (
        "AGPL-3.0",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-4.0",
        "CC-BY-SA-4.0",
        "CC0-1.0",
        "GPL-2.0",
        "GPL-3.0",
        "ISC",
        "LGPL-2.1",
        "LGPL-3.0",
        "MIT",
        "MPL-2.0",
        "Unlicense",
    )
}


def component_copy(artifact_id: str) -> ComponentCopy | None:
    """Return the curated plain-words copy for one component, if any."""
    return _CURATED_COMPONENT_COPY.get(artifact_id)


def display_license(license_id: str) -> str:
    """Render a declared licence id for people, not parsers.

    Args:
        license_id: The descriptor's licence id (SPDX, ``other`` or the
            ``NOASSERTION`` sentinel).

    Returns:
        The canonical SPDX spelling (``MIT``, ``CC-BY-4.0``), ``Unknown /
        not declared`` for the sentinel, ``Other`` for ``other``, or the id
        unchanged.
    """
    if license_id == "NOASSERTION":
        return "Unknown / not declared"
    if license_id.casefold() == "other":
        return "Other"
    return _SPDX_CANONICAL.get(license_id.casefold(), license_id)


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
    """Filter variant guidance by filename or recognized quantization.

    Args:
        rows: Render-ready variants to filter without provider I/O.
        query: Case-insensitive filename or quantization substring.

    Returns:
        Matching variants in their original deterministic order.
    """
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
    rows = []
    for entry in report.entries:
        copy = component_copy(entry.ref.artifact_id)
        rows.append(
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
                usage_notice=(copy.notice if copy is not None else entry.usage_notice),
                role=copy.role if copy is not None else "",
                purpose=copy.purpose if copy is not None else "",
            )
        )
    return tuple(rows)


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


_MIB = 1024 * 1024
_GIB = 1024 * _MIB


def format_mib_progress(done_bytes: int, total_bytes: int) -> str:
    """Render transfer progress as ``"65.0 of 632.8 MiB"`` (one shared unit).

    Args:
        done_bytes: Bytes transferred so far.
        total_bytes: Bytes in the whole transfer.

    Returns:
        Both values in MiB to one decimal, the unit written once.
    """
    return f"{done_bytes / _MIB:.1f} of {total_bytes / _MIB:.1f} MiB"


def format_free_space(free_bytes: int) -> str:
    """Render free disk space: GiB from 1 GiB up, MiB below.

    Args:
        free_bytes: Free bytes on the destination's filesystem.

    Returns:
        For example ``"112.3 GiB free"`` or ``"512.0 MiB free"``.
    """
    if free_bytes >= _GIB:
        return f"{free_bytes / _GIB:.1f} GiB free"
    return f"{format_mib(free_bytes)} free"


def format_transfer_rate(bytes_per_second: float) -> str:
    """Render a transfer speed: MiB/s from 1 MiB/s up, whole KiB/s below."""
    if bytes_per_second >= _MIB:
        return f"{bytes_per_second / _MIB:.1f} MiB/s"
    return f"{max(bytes_per_second, 0) / 1024:.0f} KiB/s"


def format_time_left(seconds: float) -> str:
    """Render an estimate in plain words: ``"about 2 min left"``."""
    if seconds < 60:
        return "less than a minute left"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {minutes} min left"
    hours, minutes = divmod(minutes, 60)
    return f"about {hours} h {minutes} min left" if minutes else f"about {hours} h left"


class TransferRateEstimator:
    """Estimate speed and time left from successive cumulative byte counts.

    A sliding window (about the last eight seconds) smooths bursty chunk
    delivery without lagging behind a real change of speed. A new phase or
    file (``key``), or a count that goes backwards (a restarted transfer),
    starts a fresh window. No estimate is offered until the window spans at
    least a second.
    """

    WINDOW_SECONDS = 8.0
    MIN_SPAN_SECONDS = 1.0

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """Create an empty estimator.

        Args:
            clock: Monotonic seconds source (tests inject a fake).
        """
        self._clock = clock
        self._key: object = None
        self._samples: deque[tuple[float, int]] = deque()

    def reset(self) -> None:
        """Forget every sample."""
        self._key = None
        self._samples.clear()

    def observe(self, key: object, bytes_done: int) -> None:
        """Record one cumulative byte count.

        Args:
            key: Identifies one continuous transfer (e.g. the phase).
            bytes_done: Cumulative bytes transferred in that transfer.
        """
        now = self._clock()
        if key != self._key or (self._samples and bytes_done < self._samples[-1][1]):
            self._samples.clear()
            self._key = key
        self._samples.append((now, bytes_done))
        while (
            len(self._samples) > 2
            and now - self._samples[1][0] >= self.WINDOW_SECONDS
        ):
            self._samples.popleft()

    @property
    def bytes_per_second(self) -> float | None:
        """Current speed, or ``None`` until the window spans a second."""
        if len(self._samples) < 2:
            return None
        (first_time, first_bytes), (last_time, last_bytes) = (
            self._samples[0],
            self._samples[-1],
        )
        span = last_time - first_time
        if span < self.MIN_SPAN_SECONDS:
            return None
        return max(last_bytes - first_bytes, 0) / span

    def seconds_left(self, remaining_bytes: int) -> float | None:
        """Time to move ``remaining_bytes`` at the current speed, if known."""
        rate = self.bytes_per_second
        if not rate:
            return None
        return max(remaining_bytes, 0) / rate


def transfer_summary(
    done_bytes: int,
    total_bytes: int,
    estimator: TransferRateEstimator | None = None,
) -> str:
    """``"65.0 of 632.8 MiB · 4.1 MiB/s · about 2 min left"`` (shared copy).

    Args:
        done_bytes: Bytes transferred so far.
        total_bytes: Bytes in the whole transfer.
        estimator: Speed source; speed and time left are omitted until it
            has an estimate.

    Returns:
        The progress clause shared by the progress widget and the steps
        that echo it in their own status line.
    """
    parts = [format_mib_progress(done_bytes, total_bytes)]
    rate = estimator.bytes_per_second if estimator is not None else None
    if rate is not None:
        parts.append(format_transfer_rate(rate))
        left = estimator.seconds_left(total_bytes - done_bytes)
        if left is not None:
            parts.append(format_time_left(left))
    return " · ".join(parts)


def waiting_for_copy(waiting_for: str | None) -> str:
    """The queued-install line: name the download being waited for."""
    if waiting_for:
        return f"Waiting for the {waiting_for} download to finish…"
    return "Waiting for another model download to finish…"


def download_cancelled_message(
    installed_count: int = 0, *, staging_removed: bool = True
) -> str:
    """What a user-cancelled download left behind, stated plainly.

    Args:
        installed_count: Components that had already finished installing
            before the cancel landed (complete and verified, so kept).
        staging_removed: Whether the partial download was removed.

    Returns:
        ``"Download cancelled — nothing was saved."`` in the common case.
    """
    if not staging_removed:
        return (
            "Download cancelled. Its partial download could not be removed; "
            "installing again resumes from it."
        )
    if installed_count == 0:
        return "Download cancelled — nothing was saved."
    kept = (
        "1 component that had finished installing was kept"
        if installed_count == 1
        else f"{installed_count} components that had finished installing were kept"
    )
    return f"Download cancelled — the partial download was removed; {kept}."


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
        AcquisitionCancelledError,
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
    if isinstance(exc, AcquisitionCancelledError):
        return download_cancelled_message(
            len(exc.installed_refs), staging_removed=exc.staging_removed
        )
    if isinstance(exc, AcquisitionBusyError):
        # TASK-34100.14 ([voice-speech-08]): name the model whose download
        # holds the session, never the caller's own model_label -- starting
        # Parakeet while OmniVoice downloads used to blame "Parakeet".
        holder = getattr(exc, "holder_label", None)
        if holder:
            return (
                f"The {holder} download is still running. Try again when it "
                "finishes."
            )
        return (
            "Another model download is already running. Try again when it finishes."
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
