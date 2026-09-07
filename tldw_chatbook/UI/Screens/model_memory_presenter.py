"""Pure, bounded user-facing copy for Remote Models memory scenarios."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...Model_Artifacts.machine_memory import (
    AcceleratorMemoryObservation,
    AcceleratorState,
    CapacityState,
    CurrentPressure,
    GGUFMemoryProjection,
    MachineMemorySnapshot,
    MemoryKind,
    ProbeReason,
    SystemMemoryState,
    format_gib,
    ram_working_budget_bytes,
)


_OBSERVED_AT_PATTERN = re.compile(r"\d{2}:\d{2}\Z")
_SAFE_ACCELERATOR_REASONS = frozenset(
    {
        ProbeReason.MALFORMED_OUTPUT,
        ProbeReason.DUPLICATE_DEVICE,
        ProbeReason.TOO_MANY_DEVICES,
        ProbeReason.COMMAND_FAILED,
        ProbeReason.OUTPUT_TOO_LARGE,
        ProbeReason.SYSFS_UNTRUSTED_PATH,
        ProbeReason.SYSFS_MALFORMED,
        ProbeReason.SYSFS_PERMISSION_DENIED,
        ProbeReason.INVALID_MEMORY_VALUE,
        ProbeReason.MEMORY_UNAVAILABLE,
    }
)


@dataclass(frozen=True, slots=True)
class CandidateMemoryPresentation:
    """Immutable per-candidate memory scenario copy."""

    outcome: str
    details: str
    pressure: str | None
    failure_line: str | None


@dataclass(frozen=True, slots=True)
class MachineMemoryPresentation:
    """Immutable machine evidence panel copy."""

    headline: str
    evidence_lines: tuple[str, ...]
    limitation_lines: tuple[str, ...]
    action_label: str
    action_disabled: bool
    failure_line: str | None
    accelerator_detail_lines: tuple[str, ...]


def build_candidate_memory_presentation(
    projection: GGUFMemoryProjection | None,
    snapshot: MachineMemorySnapshot | None = None,
    *,
    active: bool = False,
    observed_at_label: str | None = None,
    failure: ProbeReason | None = None,
) -> CandidateMemoryPresentation:
    """Build non-blocking copy for one candidate's paired memory scenarios.

    Args:
        projection: Paired 32K/64K estimates, or None before estimation.
        snapshot: Optional machine evidence used to explain unavailable states.
        active: Whether a machine-memory observation is currently running.
        observed_at_label: Bounded display time for retained evidence.
        failure: Fixed reason for a failed refresh that retained prior evidence.

    Returns:
        Immutable candidate-row copy derived only from the supplied facts.
    """
    if active and projection is None:
        return CandidateMemoryPresentation(
            outcome="Memory scenario: Checking local memory…",
            details="",
            pressure=None,
            failure_line=None,
        )
    if projection is None or projection.primary_state is CapacityState.UNKNOWN:
        return CandidateMemoryPresentation(
            outcome=_unavailable_candidate_copy(snapshot),
            details="",
            pressure=None,
            failure_line=None,
        )

    estimate_32k = projection.context_32k
    estimate_64k = projection.context_64k
    budget = estimate_64k.ram_working_budget_bytes
    total = estimate_64k.total_physical_bytes
    estimated_32k = estimate_32k.estimated_bytes
    estimated_64k = estimate_64k.estimated_bytes
    if None in {budget, total, estimated_32k, estimated_64k}:
        return CandidateMemoryPresentation(
            outcome=_unavailable_candidate_copy(snapshot),
            details="",
            pressure=None,
            failure_line=None,
        )

    outcome = _candidate_outcome(
        projection,
        budget=budget,
        total=total,
        estimated_32k=estimated_32k,
        estimated_64k=estimated_64k,
    )
    details = (
        f"32K est. {format_gib(estimated_32k)} · "
        f"64K est. {format_gib(estimated_64k)} · "
        f"RAM budget {format_gib(budget)}"
    )
    return CandidateMemoryPresentation(
        outcome=outcome,
        details=details,
        pressure=_pressure_copy(projection.current_pressure),
        failure_line=_retained_failure_copy(failure, observed_at_label),
    )


def build_machine_memory_presentation(
    snapshot: MachineMemorySnapshot | None,
    *,
    active: bool = False,
    observed_at_label: str | None = None,
    failure: ProbeReason | None = None,
) -> MachineMemoryPresentation:
    """Build machine evidence, constraints, and refresh state without side effects.

    Args:
        snapshot: Bounded machine-memory evidence, or None before observation.
        active: Whether a machine-memory observation is currently running.
        observed_at_label: Bounded display time for retained evidence.
        failure: Fixed reason for a failed refresh that retained prior evidence.

    Returns:
        Immutable copy for the machine-memory evidence panel.
    """
    action_label = "Checking…" if active else "Recheck memory"
    if snapshot is None:
        return MachineMemoryPresentation(
            headline=(
                "Machine memory: Checking local memory…"
                if active
                else "Machine estimate unavailable · filename guidance still applies"
            ),
            evidence_lines=(),
            limitation_lines=(),
            action_label=action_label,
            action_disabled=active,
            failure_line=None,
            accelerator_detail_lines=(),
        )
    if snapshot.system_state not in {
        SystemMemoryState.OBSERVED,
        SystemMemoryState.PARTIAL,
    }:
        return MachineMemoryPresentation(
            headline=_unavailable_machine_copy(snapshot.system_state),
            evidence_lines=(),
            limitation_lines=(),
            action_label="Recheck memory",
            action_disabled=False,
            failure_line=None,
            accelerator_detail_lines=(),
        )

    total = snapshot.total_bytes
    if total is None:
        return MachineMemoryPresentation(
            headline="Machine estimate unavailable · filename guidance still applies",
            evidence_lines=(),
            limitation_lines=(),
            action_label=action_label,
            action_disabled=active,
            failure_line=None,
            accelerator_detail_lines=(),
        )
    budget = ram_working_budget_bytes(total)
    is_unified = snapshot.memory_kind is MemoryKind.UNIFIED
    memory_label = "unified" if is_unified else "RAM"
    evidence = [_available_memory_copy(snapshot, is_unified=is_unified)]
    accelerator_line = _accelerator_copy(snapshot)
    if accelerator_line is not None:
        evidence.append(accelerator_line)
    return MachineMemoryPresentation(
        headline=(
            f"Machine memory: {format_gib(total)} {memory_label} · "
            f"{format_gib(budget)} RAM working budget"
        ),
        evidence_lines=tuple(evidence),
        limitation_lines=(
            "Scenarios: 32,768 / 65,536 tokens · model support not checked",
            "Heuristic only · runtime, offload, and speed not checked",
            "One selected GGUF/model · heuristic runtime/context allowances · no unusual runtime options",
            "VRAM not used in this rating · model-context support, runtime compatibility, offload, and performance not verified",
        ),
        action_label=action_label,
        action_disabled=active,
        failure_line=_retained_failure_copy(failure, observed_at_label),
        accelerator_detail_lines=_accelerator_detail_lines(snapshot),
    )


def _candidate_outcome(
    projection: GGUFMemoryProjection,
    *,
    budget: int,
    total: int,
    estimated_32k: int,
    estimated_64k: int,
) -> str:
    state_32k = projection.context_32k.capacity_state
    state_64k = projection.context_64k.capacity_state
    if state_64k is CapacityState.WITHIN_BUDGET:
        return (
            "64K scenario within RAM budget · "
            f"{format_gib(budget - estimated_64k)} headroom"
        )
    if state_32k is CapacityState.WITHIN_BUDGET:
        if state_64k is CapacityState.OVER_RESERVE:
            return (
                "32K within budget · 64K crosses reserve · "
                f"{format_gib(estimated_64k - budget)} over reserve at 64K"
            )
        return (
            "32K within budget · 64K exceeds installed RAM · "
            f"{format_gib(estimated_64k - total)} over installed RAM at 64K"
        )
    if state_32k is CapacityState.OVER_RESERVE:
        return (
            "32K crosses reserve · "
            f"{format_gib(estimated_32k - budget)} over reserve at 32K"
        )
    return (
        "32K exceeds installed RAM · "
        f"{format_gib(estimated_32k - total)} over installed RAM at 32K"
    )


def _pressure_copy(pressure: CurrentPressure) -> str | None:
    if pressure is CurrentPressure.NEEDS_MORE_FOR_64K:
        return "64K may need more free RAM now"
    if pressure is CurrentPressure.NEEDS_MORE_FOR_BOTH:
        return "32K and 64K need more free RAM now"
    return None


def _unavailable_candidate_copy(snapshot: MachineMemorySnapshot | None) -> str:
    if snapshot is None or snapshot.system_state is SystemMemoryState.UNAVAILABLE:
        return "Memory estimate unavailable · machine memory not observed"
    if snapshot.system_state is SystemMemoryState.PERMISSION_DENIED:
        return "Memory estimate unavailable · memory access denied"
    if snapshot.system_state is SystemMemoryState.UNSUPPORTED:
        return "Memory estimate unavailable on this platform"
    return "Memory estimate unavailable · machine memory not observed"


def _unavailable_machine_copy(state: SystemMemoryState) -> str:
    if state is SystemMemoryState.PERMISSION_DENIED:
        return "Memory access was denied · filename guidance still applies"
    if state is SystemMemoryState.UNSUPPORTED:
        return "Machine estimate is not supported on this platform"
    return "Machine estimate unavailable · filename guidance still applies"


def _available_memory_copy(snapshot: MachineMemorySnapshot, *, is_unified: bool) -> str:
    if snapshot.available_bytes is None:
        return "Available now: Not observed · capacity estimate still available"
    suffix = " · GPU shares unified memory" if is_unified else ""
    return f"Available now: {format_gib(snapshot.available_bytes)}{suffix}"


def _accelerator_copy(snapshot: MachineMemorySnapshot) -> str | None:
    if snapshot.memory_kind is MemoryKind.UNIFIED:
        return None
    state = snapshot.accelerator_state
    reason = snapshot.accelerator_reason
    if state is AcceleratorState.OBSERVED:
        return _observed_accelerator_copy(snapshot.accelerators, partial=False)
    if (
        state is AcceleratorState.PARTIAL
        and not snapshot.accelerators
        and snapshot.platform == "darwin"
        and snapshot.architecture not in {"arm64", "aarch64"}
        and reason is ProbeReason.UNSUPPORTED_PLATFORM
    ):
        return "VRAM observation is unavailable on this platform · RAM estimate still available"
    if state is AcceleratorState.PARTIAL and snapshot.accelerators:
        return _observed_accelerator_copy(snapshot.accelerators, partial=True)
    if state is AcceleratorState.PERMISSION_DENIED:
        return "VRAM access denied · RAM estimate still available"
    if state is AcceleratorState.UNSUPPORTED:
        return "VRAM observation is unavailable on this platform · RAM estimate still available"
    if reason is ProbeReason.COMMAND_TIMEOUT:
        return "NVIDIA VRAM check timed out · RAM estimate still available"
    if reason is ProbeReason.UNTRUSTED_EXECUTABLE:
        return "NVIDIA VRAM tool was not used from an untrusted location"
    if reason in _SAFE_ACCELERATOR_REASONS:
        return "VRAM evidence could not be read safely · RAM estimate still available"
    return "VRAM not observed · not used in this rating"


def _accelerator_detail_lines(snapshot: MachineMemorySnapshot) -> tuple[str, ...]:
    if snapshot.memory_kind is MemoryKind.UNIFIED:
        return ()
    return tuple(
        _accelerator_fact(accelerator) for accelerator in snapshot.accelerators
    )


def _observed_accelerator_copy(
    accelerators: tuple[AcceleratorMemoryObservation, ...], *, partial: bool
) -> str:
    if len(accelerators) > 2:
        base = f"VRAM observed on {len(accelerators)} devices · show estimate details"
    else:
        facts = " · ".join(
            _accelerator_fact(accelerator) for accelerator in accelerators
        )
        base = f"VRAM observed: {facts}"
    incomplete = " · other accelerator evidence incomplete" if partial else ""
    return f"{base}{incomplete} · not used in this rating"


def _accelerator_fact(accelerator: AcceleratorMemoryObservation) -> str:
    vendor = {"nvidia": "NVIDIA", "amd": "AMD"}.get(
        accelerator.vendor.casefold(), accelerator.vendor.upper()
    )
    normalized_label = accelerator.label.casefold()
    normalized_vendor = vendor.casefold()
    label_has_vendor = normalized_label == normalized_vendor or (
        normalized_label.startswith(normalized_vendor)
        and len(accelerator.label) > len(vendor)
        and not accelerator.label[len(vendor)].isalnum()
    )
    fact = accelerator.label if label_has_vendor else f"{vendor} {accelerator.label}"
    if accelerator.total_bytes is None:
        return fact
    return f"{fact} {format_gib(accelerator.total_bytes)}"


def _retained_failure_copy(
    failure: ProbeReason | None, observed_at_label: str | None
) -> str | None:
    if failure is None:
        return None
    if observed_at_label is not None and _OBSERVED_AT_PATTERN.fullmatch(
        observed_at_label
    ):
        return f"Recheck failed · using memory observed at {observed_at_label}"
    return "Recheck failed · using previously observed memory"
