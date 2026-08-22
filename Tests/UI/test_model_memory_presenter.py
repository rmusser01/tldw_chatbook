"""Behavior tests for pure Remote Models memory-estimate presentation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Model_Artifacts.machine_memory import (
    AcceleratorMemoryObservation,
    AcceleratorSource,
    AcceleratorState,
    GIB,
    GGUFMemoryProjection,
    MachineMemorySnapshot,
    MemoryKind,
    ProbeReason,
    SystemMemoryState,
    project_gguf_memory,
)
from tldw_chatbook.UI.Screens.model_memory_presenter import (
    build_candidate_memory_presentation,
    build_machine_memory_presentation,
)


def _snapshot(
    *,
    total_gib: int = 32,
    available_gib: int | None = 21,
    system_state: SystemMemoryState = SystemMemoryState.OBSERVED,
    system_reason: ProbeReason | None = None,
    accelerator_state: AcceleratorState = AcceleratorState.NOT_OBSERVED,
    accelerator_reason: ProbeReason | None = None,
    accelerators: tuple[AcceleratorMemoryObservation, ...] = (),
    memory_kind: MemoryKind = MemoryKind.SYSTEM,
    platform: str = "linux",
    architecture: str = "x86_64",
) -> MachineMemorySnapshot:
    total_bytes = total_gib * GIB if system_state in {
        SystemMemoryState.OBSERVED,
        SystemMemoryState.PARTIAL,
    } else None
    resolved_memory_kind = (
        memory_kind
        if total_bytes is not None
        else MemoryKind.UNKNOWN
    )
    return MachineMemorySnapshot(
        platform=platform,
        architecture=architecture,
        system_state=system_state,
        accelerator_state=accelerator_state,
        total_bytes=total_bytes,
        available_bytes=(
            available_gib * GIB
            if total_bytes is not None and available_gib is not None
            else None
        ),
        memory_kind=resolved_memory_kind,
        accelerators=accelerators,
        system_reason=system_reason,
        accelerator_reason=accelerator_reason,
    )


def _projection(*, available_gib: int | None = 21) -> GGUFMemoryProjection:
    return project_gguf_memory(
        4 * GIB,
        _snapshot(
            available_gib=available_gib,
            system_state=(
                SystemMemoryState.OBSERVED
                if available_gib is not None
                else SystemMemoryState.PARTIAL
            ),
            system_reason=(
                None if available_gib is not None else ProbeReason.MEMORY_UNAVAILABLE
            ),
        ),
    )


def _device(
    *, label: str = "RTX 4090", vendor: str = "nvidia", gib: int = 24
) -> AcceleratorMemoryObservation:
    return AcceleratorMemoryObservation(
        vendor=vendor,
        label=label,
        total_bytes=gib * GIB,
        shared=False,
        source=AcceleratorSource.NVIDIA_SMI,
    )


def test_within_budget_copy_names_scenario_and_keeps_support_unverified() -> None:
    """Changing the primary outcome or adding a fit claim must break this copy."""
    presentation = build_candidate_memory_presentation(_projection())

    assert presentation.outcome == "64K scenario within RAM budget · 12.6 GiB headroom"
    assert presentation.details == (
        "32K est. 9.0 GiB · 64K est. 13.0 GiB · RAM budget 25.6 GiB"
    )
    assert "fits" not in presentation.outcome.lower()
    assert "support" not in presentation.outcome.lower()


def test_current_pressure_is_separate_from_stable_capacity_copy() -> None:
    """Using available RAM to relabel capacity would hide the stable outcome."""
    presentation = build_candidate_memory_presentation(_projection(available_gib=10))

    assert presentation.outcome.startswith("64K scenario within RAM budget")
    assert presentation.pressure == "64K may need more free RAM now"


@pytest.mark.parametrize(
    ("total_gib", "model_gib", "expected"),
    [
        (32, 4, "64K scenario within RAM budget · 12.6 GiB headroom"),
        (16, 4, "32K within budget · 64K crosses reserve · 0.2 GiB over reserve at 64K"),
        (12, 4, "32K within budget · 64K exceeds installed RAM · 1.0 GiB over installed RAM at 64K"),
        (12, 6, "32K crosses reserve · 1.4 GiB over reserve at 32K"),
        (12, 8, "32K exceeds installed RAM · 1.0 GiB over installed RAM at 32K"),
    ],
)
def test_candidate_capacity_outcomes_follow_exact_priority(
    total_gib: int, model_gib: int, expected: str
) -> None:
    """Changing scenario priority or overage arithmetic must change the row outcome."""
    snapshot = _snapshot(total_gib=total_gib, available_gib=total_gib)

    presentation = build_candidate_memory_presentation(
        project_gguf_memory(model_gib * GIB, snapshot), snapshot
    )

    assert presentation.outcome == expected


@pytest.mark.parametrize(
    ("available_gib", "expected"),
    [
        (21, None),
        (10, "64K may need more free RAM now"),
        (8, "32K and 64K need more free RAM now"),
        (None, None),
    ],
)
def test_candidate_pressure_matrix_never_changes_capacity(
    available_gib: int | None, expected: str | None
) -> None:
    """Changing the pressure branch must not affect the capacity result."""
    presentation = build_candidate_memory_presentation(_projection(available_gib=available_gib))

    assert presentation.outcome.startswith("64K scenario within RAM budget")
    assert presentation.pressure == expected


@pytest.mark.parametrize(
    ("system_state", "system_reason", "expected"),
    [
        (
            SystemMemoryState.UNAVAILABLE,
            ProbeReason.MEMORY_UNAVAILABLE,
            "Memory estimate unavailable · machine memory not observed",
        ),
        (
            SystemMemoryState.PERMISSION_DENIED,
            ProbeReason.PERMISSION_DENIED,
            "Memory estimate unavailable · memory access denied",
        ),
        (
            SystemMemoryState.UNSUPPORTED,
            ProbeReason.UNSUPPORTED_PLATFORM,
            "Memory estimate unavailable on this platform",
        ),
    ],
)
def test_candidate_unknown_copy_explains_only_fixed_system_state(
    system_state: SystemMemoryState, system_reason: ProbeReason, expected: str
) -> None:
    """A raw probe reason or a rating claim in unavailable copy is a privacy bug."""
    snapshot = _snapshot(system_state=system_state, system_reason=system_reason)

    presentation = build_candidate_memory_presentation(
        project_gguf_memory(4 * GIB, snapshot), snapshot
    )

    assert presentation.outcome == expected
    assert presentation.details == ""
    assert presentation.pressure is None


def test_candidate_active_copy_is_nonblocking_and_has_no_estimate() -> None:
    """Showing a capacity result before the first probe settles would mislead users."""
    presentation = build_candidate_memory_presentation(None, active=True)

    assert presentation.outcome == "Memory scenario: Checking local memory…"
    assert presentation.details == ""
    assert presentation.pressure is None


@pytest.mark.parametrize(
    ("system_state", "system_reason", "headline"),
    [
        (SystemMemoryState.UNAVAILABLE, ProbeReason.MEMORY_UNAVAILABLE, "Machine estimate unavailable · filename guidance still applies"),
        (SystemMemoryState.PERMISSION_DENIED, ProbeReason.PERMISSION_DENIED, "Memory access was denied · filename guidance still applies"),
        (SystemMemoryState.UNSUPPORTED, ProbeReason.UNSUPPORTED_PLATFORM, "Machine estimate is not supported on this platform"),
    ],
)
def test_machine_unavailable_system_states_have_fixed_recovery_copy(
    system_state: SystemMemoryState, system_reason: ProbeReason, headline: str
) -> None:
    """Collapsing unavailable states would erase an actionable distinction."""
    presentation = build_machine_memory_presentation(
        _snapshot(system_state=system_state, system_reason=system_reason)
    )

    assert presentation.headline == headline
    assert presentation.evidence_lines == ()
    assert presentation.limitation_lines == ()
    assert presentation.action_label == "Recheck memory"
    assert not presentation.action_disabled
    assert presentation.failure_line is None


def test_machine_observed_evidence_uses_binary_gib_budget_and_limitations() -> None:
    """Changing visible RAM facts, policy caveats, or retry availability must be caught."""
    presentation = build_machine_memory_presentation(_snapshot())

    assert presentation.headline == "Machine memory: 32.0 GiB RAM · 25.6 GiB RAM working budget"
    assert presentation.evidence_lines == (
        "Available now: 21.0 GiB",
        "VRAM not observed · not used in this rating",
    )
    assert presentation.limitation_lines == (
        "Scenarios: 32,768 / 65,536 tokens · model support not checked",
        "Heuristic only · runtime, offload, and speed not checked",
    )
    assert presentation.action_label == "Recheck memory"
    assert not presentation.action_disabled


def test_machine_partial_ram_keeps_capacity_but_omits_volatile_pressure() -> None:
    """Treating missing availability as absent capacity would discard usable guidance."""
    snapshot = _snapshot(
        available_gib=None,
        system_state=SystemMemoryState.PARTIAL,
        system_reason=ProbeReason.MEMORY_UNAVAILABLE,
    )

    presentation = build_machine_memory_presentation(snapshot)

    assert presentation.headline == "Machine memory: 32.0 GiB RAM · 25.6 GiB RAM working budget"
    assert presentation.evidence_lines[0] == (
        "Available now: Not observed · capacity estimate still available"
    )


def test_machine_active_first_probe_only_shows_checking_state() -> None:
    """A first-probe screen must not fabricate facts before observation finishes."""
    presentation = build_machine_memory_presentation(None, active=True)

    assert presentation.headline == "Machine memory: Checking local memory…"
    assert presentation.evidence_lines == ()
    assert presentation.action_label == "Checking…"
    assert presentation.action_disabled


def test_retained_refresh_failure_keeps_facts_and_uses_fixed_observed_label() -> None:
    """Replacing accepted RAM after a failed recheck would regress trustworthy guidance."""
    presentation = build_machine_memory_presentation(
        _snapshot(),
        observed_at_label="09:41",
        failure=ProbeReason.COMMAND_TIMEOUT,
    )

    assert presentation.headline == "Machine memory: 32.0 GiB RAM · 25.6 GiB RAM working budget"
    assert presentation.failure_line == "Recheck failed · using memory observed at 09:41"


def test_active_refresh_keeps_accepted_facts_but_disables_recheck() -> None:
    """A refresh must preserve the last evidence while preventing duplicate requests."""
    presentation = build_machine_memory_presentation(_snapshot(), active=True)

    assert presentation.headline.startswith("Machine memory: 32.0 GiB RAM")
    assert presentation.action_label == "Checking…"
    assert presentation.action_disabled


def test_apple_unified_memory_is_shown_once_without_vram_duplication() -> None:
    """A separate Apple VRAM line would double-count the shared physical pool."""
    snapshot = _snapshot(
        platform="darwin",
        architecture="arm64",
        memory_kind=MemoryKind.UNIFIED,
        accelerator_state=AcceleratorState.OBSERVED,
        accelerators=(
            AcceleratorMemoryObservation(
                vendor="apple",
                label="Apple unified memory",
                total_bytes=None,
                shared=True,
                source=AcceleratorSource.APPLE_UNIFIED,
            ),
        ),
    )

    presentation = build_machine_memory_presentation(snapshot)

    assert presentation.headline == "Machine memory: 32.0 GiB unified · 25.6 GiB RAM working budget"
    assert presentation.evidence_lines == ("Available now: 21.0 GiB · GPU shares unified memory",)


@pytest.mark.parametrize(
    ("state", "reason", "expected"),
    [
        (AcceleratorState.NOT_OBSERVED, ProbeReason.EXECUTABLE_NOT_FOUND, "VRAM not observed · not used in this rating"),
        (AcceleratorState.PERMISSION_DENIED, ProbeReason.PERMISSION_DENIED, "VRAM access denied · RAM estimate still available"),
        (AcceleratorState.UNSUPPORTED, ProbeReason.UNSUPPORTED_PLATFORM, "VRAM observation is unavailable on this platform · RAM estimate still available"),
        (AcceleratorState.NOT_OBSERVED, ProbeReason.COMMAND_TIMEOUT, "NVIDIA VRAM check timed out · RAM estimate still available"),
        (AcceleratorState.NOT_OBSERVED, ProbeReason.UNTRUSTED_EXECUTABLE, "NVIDIA VRAM tool was not used from an untrusted location"),
        (AcceleratorState.NOT_OBSERVED, ProbeReason.MALFORMED_OUTPUT, "VRAM evidence could not be read safely · RAM estimate still available"),
        (AcceleratorState.NOT_OBSERVED, ProbeReason.DUPLICATE_DEVICE, "VRAM evidence could not be read safely · RAM estimate still available"),
        (AcceleratorState.NOT_OBSERVED, ProbeReason.TOO_MANY_DEVICES, "VRAM evidence could not be read safely · RAM estimate still available"),
        (AcceleratorState.NOT_OBSERVED, ProbeReason.COMMAND_FAILED, "VRAM evidence could not be read safely · RAM estimate still available"),
        (AcceleratorState.NOT_OBSERVED, ProbeReason.SYSFS_UNTRUSTED_PATH, "VRAM evidence could not be read safely · RAM estimate still available"),
    ],
)
def test_accelerator_failure_copy_is_fixed_and_independent_from_ram(
    state: AcceleratorState, reason: ProbeReason, expected: str
) -> None:
    """Changing accelerator failure handling must not invalidate a known RAM estimate."""
    snapshot = _snapshot(accelerator_state=state, accelerator_reason=reason)

    presentation = build_machine_memory_presentation(snapshot)

    assert presentation.evidence_lines[-1] == expected


def test_observed_and_partial_vram_are_informational_only() -> None:
    """VRAM must never be folded into the v1 RAM estimate or omitted from caveats."""
    observed = build_machine_memory_presentation(
        _snapshot(accelerator_state=AcceleratorState.OBSERVED, accelerators=(_device(),))
    )
    partial = build_machine_memory_presentation(
        _snapshot(
            accelerator_state=AcceleratorState.PARTIAL,
            accelerator_reason=ProbeReason.COMMAND_TIMEOUT,
            accelerators=(_device(),),
        )
    )

    assert observed.evidence_lines[-1] == (
        "VRAM observed: NVIDIA RTX 4090 24.0 GiB · not used in this rating"
    )
    assert partial.evidence_lines[-1] == (
        "VRAM observed: NVIDIA RTX 4090 24.0 GiB · other accelerator evidence incomplete · not used in this rating"
    )


def test_overflow_accelerators_stay_bounded_behind_details_toggle() -> None:
    """Listing every device inline would make the panel unbounded and unusable."""
    devices = tuple(_device(label=f"RTX {index}") for index in range(3))
    snapshot = _snapshot(accelerator_state=AcceleratorState.OBSERVED, accelerators=devices)

    presentation = build_machine_memory_presentation(snapshot)

    assert presentation.evidence_lines[-1] == (
        "VRAM observed on 3 devices · show estimate details · not used in this rating"
    )


def test_presentations_are_immutable_values() -> None:
    """A later UI mutation must not alter the trusted copy object shared by rows."""
    candidate = build_candidate_memory_presentation(_projection())
    machine = build_machine_memory_presentation(_snapshot())

    with pytest.raises(FrozenInstanceError):
        candidate.outcome = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        machine.evidence_lines = ()  # type: ignore[misc]
