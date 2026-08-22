"""Tests for the bounded, pure GGUF memory-scenario policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Model_Artifacts.machine_memory import (
    AcceleratorMemoryObservation,
    AcceleratorSource,
    AcceleratorState,
    CapacityState,
    ContextMemoryEstimate,
    CurrentPressure,
    GGUFMemoryProjection,
    GIB,
    MIB,
    MAX_INPUT_BYTES,
    MAX_PROJECTED_BYTES,
    MachineMemorySnapshot,
    MemoryKind,
    ProbeReason,
    SystemMemoryState,
    format_gib,
    project_gguf_memory,
)


def _snapshot(
    *,
    total_bytes: int | None = 32 * GIB,
    available_bytes: int | None = 20 * GIB,
    system_state: SystemMemoryState = SystemMemoryState.OBSERVED,
    system_reason: ProbeReason | None = None,
    accelerators: tuple[AcceleratorMemoryObservation, ...] = (),
    accelerator_state: AcceleratorState = AcceleratorState.NOT_OBSERVED,
    accelerator_reason: ProbeReason | None = None,
    memory_kind: MemoryKind = MemoryKind.SYSTEM,
) -> MachineMemorySnapshot:
    return MachineMemorySnapshot(
        platform="linux",
        architecture="x86_64",
        system_state=system_state,
        accelerator_state=accelerator_state,
        total_bytes=total_bytes,
        available_bytes=available_bytes,
        memory_kind=memory_kind,
        accelerators=accelerators,
        system_reason=system_reason,
        accelerator_reason=accelerator_reason,
    )


def _estimate(
    *,
    context_tokens: int,
    model_bytes: int = 4 * GIB,
    runtime_allowance_bytes: int = GIB,
    context_allowance_bytes: int,
    ram_working_budget_bytes: int = 25 * GIB,
    total_physical_bytes: int = 32 * GIB,
) -> ContextMemoryEstimate:
    return ContextMemoryEstimate(
        context_tokens=context_tokens,
        model_bytes=model_bytes,
        runtime_allowance_bytes=runtime_allowance_bytes,
        context_allowance_bytes=context_allowance_bytes,
        estimated_bytes=(
            model_bytes + runtime_allowance_bytes + context_allowance_bytes
        ),
        ram_working_budget_bytes=ram_working_budget_bytes,
        total_physical_bytes=total_physical_bytes,
        capacity_state=CapacityState.WITHIN_BUDGET,
    )


def test_snapshot_is_frozen_and_rejects_boolean_memory() -> None:
    """Rejecting bool keeps Python's bool-as-int quirk out of capacity policy."""
    snapshot = _snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.total_bytes = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="total_bytes"):
        _snapshot(total_bytes=True)  # type: ignore[arg-type]


def test_projection_uses_exact_32768_and_65536_scenarios() -> None:
    """A wrong scenario token count or allowance must alter this visible estimate."""
    projection = project_gguf_memory(4 * GIB, _snapshot())

    assert projection.context_32k.context_tokens == 32_768
    assert projection.context_64k.context_tokens == 65_536
    assert projection.context_32k.runtime_allowance_bytes == GIB
    assert projection.context_32k.context_allowance_bytes == 4 * GIB
    assert projection.context_64k.context_allowance_bytes == 8 * GIB
    assert projection.context_64k.capacity_state is CapacityState.WITHIN_BUDGET


@pytest.mark.parametrize("model_bytes", [0, -1, True, 2**63])
def test_invalid_candidate_is_unknown_without_throwing(model_bytes: object) -> None:
    """Invalid provider candidate totals must remain non-blocking unknown estimates."""
    projection = project_gguf_memory(model_bytes, _snapshot())  # type: ignore[arg-type]

    assert projection.primary_state is CapacityState.UNKNOWN
    assert projection.current_pressure is CurrentPressure.UNKNOWN


@pytest.mark.parametrize(
    ("estimated_offset", "expected_state"),
    [
        (-1, CapacityState.WITHIN_BUDGET),
        (0, CapacityState.WITHIN_BUDGET),
        (1, CapacityState.OVER_RESERVE),
    ],
)
def test_capacity_classifies_one_byte_budget_boundaries(
    estimated_offset: int, expected_state: CapacityState
) -> None:
    """Changing <= budget to < budget would break the exact reserve boundary."""
    # 10 GiB total reserves the 2 GiB floor, leaving an 8 GiB working budget.
    # The fixed 1 GiB runtime plus 4 GiB context makes the estimate model + 5 GiB.
    total_bytes = 10 * GIB
    target_estimate = 8 * GIB + estimated_offset
    projection = project_gguf_memory(
        target_estimate - 5 * GIB,
        _snapshot(total_bytes=total_bytes, available_bytes=total_bytes),
    )

    assert projection.context_32k.estimated_bytes == target_estimate
    assert projection.context_32k.capacity_state is expected_state


@pytest.mark.parametrize(
    ("estimated_offset", "expected_state"),
    [
        (-1, CapacityState.OVER_RESERVE),
        (0, CapacityState.OVER_RESERVE),
        (1, CapacityState.OVER_TOTAL),
    ],
)
def test_capacity_classifies_one_byte_total_boundaries(
    estimated_offset: int, expected_state: CapacityState
) -> None:
    """Changing <= total to < total would misclassify a model at installed RAM."""
    total_bytes = 10 * GIB
    target_estimate = total_bytes + estimated_offset
    projection = project_gguf_memory(
        target_estimate - 5 * GIB,
        _snapshot(total_bytes=total_bytes, available_bytes=total_bytes),
    )

    assert projection.context_32k.estimated_bytes == target_estimate
    assert projection.context_32k.capacity_state is expected_state


def test_reserve_uses_floor_then_percentage_rounded_up_to_mib() -> None:
    """The reserve floor and percentage branch must remain independently exact."""
    floor_projection = project_gguf_memory(
        1 * GIB, _snapshot(total_bytes=10 * GIB, available_bytes=10 * GIB)
    )
    percentage_projection = project_gguf_memory(
        1 * GIB, _snapshot(total_bytes=11 * GIB, available_bytes=11 * GIB)
    )

    assert floor_projection.context_32k.ram_working_budget_bytes == 8 * GIB
    assert percentage_projection.context_32k.ram_working_budget_bytes == (
        11 * GIB - 2253 * MIB
    )


def test_percentage_allowances_round_one_byte_up_to_the_next_mib() -> None:
    """Dropping upward MiB rounding would understate a candidate just above a boundary."""
    model_bytes = 40 * GIB + 1
    projection = project_gguf_memory(model_bytes, _snapshot(total_bytes=128 * GIB))

    assert projection.context_32k.runtime_allowance_bytes == 4097 * MIB
    assert projection.context_32k.context_allowance_bytes == 10_241 * MIB
    assert projection.context_64k.context_allowance_bytes == 20_482 * MIB


def test_maximum_candidate_is_projected_without_python_overflow() -> None:
    """A valid 63-bit candidate retains its exact size even when capacity is unknown."""
    projection = project_gguf_memory(MAX_INPUT_BYTES, _snapshot())

    assert projection.context_32k.model_bytes == MAX_INPUT_BYTES
    assert projection.context_64k.model_bytes == MAX_INPUT_BYTES
    assert projection.context_32k.estimated_bytes is not None
    assert projection.context_32k.estimated_bytes <= MAX_PROJECTED_BYTES


def test_highest_valid_candidate_remains_bounded_without_wrapping() -> None:
    """The 64-bit bound must remain explicit even at the highest valid candidate."""
    projection = project_gguf_memory(
        MAX_INPUT_BYTES, _snapshot(total_bytes=MAX_INPUT_BYTES)
    )

    assert projection.primary_state is CapacityState.OVER_TOTAL
    assert projection.context_64k.estimated_bytes is not None
    assert projection.context_64k.estimated_bytes <= MAX_PROJECTED_BYTES


def test_out_of_bound_derived_estimate_is_rejected_without_wrapping() -> None:
    """A future formula cannot inject an estimate larger than the 64-bit output bound."""
    with pytest.raises(ValueError, match="estimated_bytes"):
        ContextMemoryEstimate(
            context_tokens=32_768,
            model_bytes=MAX_INPUT_BYTES,
            runtime_allowance_bytes=GIB,
            context_allowance_bytes=4 * GIB,
            estimated_bytes=MAX_PROJECTED_BYTES + 1,
            ram_working_budget_bytes=MAX_INPUT_BYTES,
            total_physical_bytes=MAX_INPUT_BYTES,
            capacity_state=CapacityState.OVER_TOTAL,
        )


def test_context_estimate_rejects_a_contradictory_derived_total() -> None:
    """A caller cannot label an undersized sum as a trustworthy memory estimate."""
    with pytest.raises(ValueError, match="estimated_bytes"):
        ContextMemoryEstimate(
            context_tokens=32_768,
            model_bytes=4 * GIB,
            runtime_allowance_bytes=GIB,
            context_allowance_bytes=4 * GIB,
            estimated_bytes=8 * GIB,
            ram_working_budget_bytes=25 * GIB,
            total_physical_bytes=32 * GIB,
            capacity_state=CapacityState.WITHIN_BUDGET,
        )


def test_context_estimate_rejects_float_context_tokens() -> None:
    """A float equal to 32768 is not a valid exact context-token count."""
    with pytest.raises(ValueError, match="context_tokens"):
        _estimate(
            context_tokens=32_768.0,  # type: ignore[arg-type]
            context_allowance_bytes=4 * GIB,
        )


@pytest.mark.parametrize(
    "context_64k",
    [
        _estimate(
            context_tokens=65_536,
            model_bytes=5 * GIB,
            context_allowance_bytes=8 * GIB,
        ),
        _estimate(
            context_tokens=65_536,
            runtime_allowance_bytes=2 * GIB,
            context_allowance_bytes=8 * GIB,
        ),
        _estimate(
            context_tokens=65_536,
            context_allowance_bytes=8 * GIB,
            ram_working_budget_bytes=24 * GIB,
        ),
        _estimate(
            context_tokens=65_536,
            context_allowance_bytes=8 * GIB,
            total_physical_bytes=31 * GIB,
        ),
        _estimate(context_tokens=65_536, context_allowance_bytes=6 * GIB),
    ],
)
def test_projection_rejects_mismatched_paired_scenario_facts(
    context_64k: ContextMemoryEstimate,
) -> None:
    """Paired scenarios must share model/machine facts and double the 32K allowance."""
    with pytest.raises(ValueError):
        GGUFMemoryProjection(
            context_32k=_estimate(
                context_tokens=32_768,
                context_allowance_bytes=4 * GIB,
            ),
            context_64k=context_64k,
            primary_state=CapacityState.WITHIN_BUDGET,
            current_pressure=CurrentPressure.NONE,
        )


def test_current_available_memory_sets_warning_without_changing_capacity() -> None:
    """Volatile free RAM must not change the stable capacity classification."""
    model_bytes = 4 * GIB
    roomy = project_gguf_memory(model_bytes, _snapshot(available_bytes=20 * GIB))
    only_32k_free = project_gguf_memory(
        model_bytes, _snapshot(available_bytes=10 * GIB)
    )
    too_little_free = project_gguf_memory(
        model_bytes, _snapshot(available_bytes=8 * GIB)
    )

    assert roomy.primary_state is CapacityState.WITHIN_BUDGET
    assert only_32k_free.primary_state is CapacityState.WITHIN_BUDGET
    assert too_little_free.primary_state is CapacityState.WITHIN_BUDGET
    assert roomy.current_pressure is CurrentPressure.NONE
    assert only_32k_free.current_pressure is CurrentPressure.NEEDS_MORE_FOR_64K
    assert too_little_free.current_pressure is CurrentPressure.NEEDS_MORE_FOR_BOTH


def test_partial_system_memory_keeps_stable_estimates_but_pressure_unknown() -> None:
    """Absent available RAM may not erase a trustworthy total-RAM estimate."""
    projection = project_gguf_memory(
        4 * GIB,
        _snapshot(
            available_bytes=None,
            system_state=SystemMemoryState.PARTIAL,
            system_reason=ProbeReason.INVALID_MEMORY_VALUE,
        ),
    )

    assert projection.primary_state is CapacityState.WITHIN_BUDGET
    assert projection.current_pressure is CurrentPressure.UNKNOWN


def test_unknown_system_total_fails_closed() -> None:
    """No rating may be emitted when physical RAM has not been observed."""
    projection = project_gguf_memory(
        4 * GIB,
        _snapshot(
            total_bytes=None,
            available_bytes=None,
            system_state=SystemMemoryState.UNAVAILABLE,
            system_reason=ProbeReason.MEMORY_UNAVAILABLE,
            memory_kind=MemoryKind.UNKNOWN,
        ),
    )

    assert projection.primary_state is CapacityState.UNKNOWN


def test_apple_unified_marker_is_shared_and_never_changes_ram_rating() -> None:
    """Adding Apple shared evidence must not double-count the system RAM pool."""
    apple_marker = AcceleratorMemoryObservation(
        vendor="Apple",
        label="Apple unified memory",
        total_bytes=None,
        shared=True,
        source=AcceleratorSource.APPLE_UNIFIED,
    )
    unified = MachineMemorySnapshot(
        platform="darwin",
        architecture="arm64",
        system_state=SystemMemoryState.OBSERVED,
        accelerator_state=AcceleratorState.OBSERVED,
        total_bytes=32 * GIB,
        available_bytes=20 * GIB,
        memory_kind=MemoryKind.UNIFIED,
        accelerators=(apple_marker,),
        system_reason=None,
        accelerator_reason=None,
    )

    assert (
        project_gguf_memory(4 * GIB, unified).primary_state
        is CapacityState.WITHIN_BUDGET
    )
    assert project_gguf_memory(4 * GIB, unified).context_32k.estimated_bytes == 9 * GIB


def test_duplicate_accelerator_labels_are_rejected() -> None:
    """Duplicate labels would make bounded per-device evidence ambiguous."""
    first = AcceleratorMemoryObservation(
        vendor="NVIDIA",
        label="NVIDIA GPU 0",
        total_bytes=24 * GIB,
        shared=False,
        source=AcceleratorSource.NVIDIA_SMI,
    )
    duplicate = AcceleratorMemoryObservation(
        vendor="NVIDIA",
        label="NVIDIA GPU 0",
        total_bytes=16 * GIB,
        shared=False,
        source=AcceleratorSource.NVIDIA_SMI,
    )

    with pytest.raises(ValueError, match="duplicate"):
        _snapshot(
            accelerators=(first, duplicate),
            accelerator_state=AcceleratorState.OBSERVED,
        )


@pytest.mark.parametrize(
    ("vendor", "label"),
    [
        ("NVIDIA\n", "NVIDIA GPU"),
        ("NVIDIA", "NVIDIA\x00GPU"),
        ("x" * 33, "NVIDIA GPU"),
        ("NVIDIA", "x" * 97),
    ],
)
def test_accelerator_text_bounds_reject_controls_and_overlong_values(
    vendor: str, label: str
) -> None:
    """Unbounded or control-bearing device text must not reach the UI model."""
    with pytest.raises(ValueError):
        AcceleratorMemoryObservation(
            vendor=vendor,
            label=label,
            total_bytes=24 * GIB,
            shared=False,
            source=AcceleratorSource.NVIDIA_SMI,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0.0 GiB"), (GIB, "1.0 GiB"), (3 * GIB // 2, "1.5 GiB")],
)
def test_format_gib_renders_one_decimal_binary_gib(value: int, expected: str) -> None:
    """The machine panel needs a stable binary-unit display for byte values."""
    assert format_gib(value) == expected
