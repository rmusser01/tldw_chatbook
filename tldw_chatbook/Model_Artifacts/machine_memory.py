"""Bounded, pure local-memory scenarios for remotely discovered GGUF files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


MIB = 1 << 20
GIB = 1 << 30
MAX_INPUT_BYTES = (1 << 63) - 1
MAX_PROJECTED_BYTES = (1 << 64) - 1
CONTEXT_32K = 32_768
CONTEXT_64K = 65_536
MAX_ACCELERATORS = 16

_PLATFORMS = frozenset({"darwin", "linux", "windows", "other"})
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,32}\Z")


class SystemMemoryState(StrEnum):
    """Trust state for total and available system-memory evidence."""

    OBSERVED = "observed"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED = "unsupported"


class AcceleratorState(StrEnum):
    """Trust state for optional accelerator-memory evidence."""

    OBSERVED = "observed"
    PARTIAL = "partial"
    NOT_OBSERVED = "not_observed"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED = "unsupported"


class MemoryKind(StrEnum):
    """Whether the system pool is unified, system RAM, or unavailable."""

    UNIFIED = "unified"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class AcceleratorSource(StrEnum):
    """Bounded source for one accelerator-memory observation."""

    APPLE_UNIFIED = "apple_unified"
    NVIDIA_SMI = "nvidia_smi"
    LINUX_DRM = "linux_drm"


class CurrentPressure(StrEnum):
    """Volatile available-RAM warning independent from stable capacity."""

    NONE = "none"
    NEEDS_MORE_FOR_64K = "needs_more_for_64k"
    NEEDS_MORE_FOR_BOTH = "needs_more_for_both"
    UNKNOWN = "unknown"


class CapacityState(StrEnum):
    """Stable system-RAM comparison outcome for a memory scenario."""

    WITHIN_BUDGET = "within_budget"
    OVER_RESERVE = "over_reserve"
    OVER_TOTAL = "over_total"
    UNKNOWN = "unknown"


class ProbeReason(StrEnum):
    """Fixed, privacy-safe observation failure reason."""

    MEMORY_UNAVAILABLE = "memory_unavailable"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    INVALID_MEMORY_VALUE = "invalid_memory_value"
    EXECUTABLE_NOT_FOUND = "executable_not_found"
    UNTRUSTED_EXECUTABLE = "untrusted_executable"
    COMMAND_TIMEOUT = "command_timeout"
    COMMAND_FAILED = "command_failed"
    OUTPUT_TOO_LARGE = "output_too_large"
    MALFORMED_OUTPUT = "malformed_output"
    TOO_MANY_DEVICES = "too_many_devices"
    DUPLICATE_DEVICE = "duplicate_device"
    SYSFS_PERMISSION_DENIED = "sysfs_permission_denied"
    SYSFS_UNTRUSTED_PATH = "sysfs_untrusted_path"
    SYSFS_MALFORMED = "sysfs_malformed"


def _require_enum(value: object, enum_type: type[StrEnum], name: str) -> None:
    if type(value) is not enum_type:
        raise ValueError(f"{name} must be a {enum_type.__name__}")


def _require_identifier(value: object, name: str) -> None:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be 1-32 ASCII identifier characters")


def _require_display_label(value: object, name: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > 96
        or not value.isprintable()
    ):
        raise ValueError(f"{name} must be 1-96 printable characters")


def _require_bytes(
    value: object,
    name: str,
    *,
    allow_none: bool = False,
    allow_zero: bool = False,
    maximum: int = MAX_INPUT_BYTES,
) -> None:
    if value is None and allow_none:
        return
    lower_bound = 0 if allow_zero else 1
    if type(value) is not int or not lower_bound <= value <= maximum:
        raise ValueError(
            f"{name} must be an integer between {lower_bound} and {maximum}"
        )


@dataclass(frozen=True, slots=True)
class AcceleratorMemoryObservation:
    """One bounded, non-identifying accelerator memory observation."""

    vendor: str
    label: str
    total_bytes: int | None
    shared: bool
    source: AcceleratorSource

    def __post_init__(self) -> None:
        _require_identifier(self.vendor, "vendor")
        _require_display_label(self.label, "label")
        if type(self.shared) is not bool:
            raise ValueError("shared must be a boolean")
        _require_enum(self.source, AcceleratorSource, "source")
        if self.shared:
            if self.total_bytes is not None:
                raise ValueError(
                    "shared observations may not have dedicated total_bytes"
                )
            if self.source is not AcceleratorSource.APPLE_UNIFIED:
                raise ValueError("shared observations must use apple_unified source")
            return
        _require_bytes(self.total_bytes, "total_bytes")
        if self.source is AcceleratorSource.APPLE_UNIFIED:
            raise ValueError("apple_unified observations must be shared")


@dataclass(frozen=True, slots=True)
class MachineMemorySnapshot:
    """Immutable local memory facts with independent RAM and accelerator states."""

    platform: str
    architecture: str
    system_state: SystemMemoryState
    accelerator_state: AcceleratorState
    total_bytes: int | None
    available_bytes: int | None
    memory_kind: MemoryKind
    accelerators: tuple[AcceleratorMemoryObservation, ...]
    system_reason: ProbeReason | None
    accelerator_reason: ProbeReason | None

    def __post_init__(self) -> None:
        if type(self.platform) is not str or self.platform not in _PLATFORMS:
            raise ValueError("platform must be darwin, linux, windows, or other")
        _require_identifier(self.architecture, "architecture")
        _require_enum(self.system_state, SystemMemoryState, "system_state")
        _require_enum(self.accelerator_state, AcceleratorState, "accelerator_state")
        _require_enum(self.memory_kind, MemoryKind, "memory_kind")
        if self.system_reason is not None:
            _require_enum(self.system_reason, ProbeReason, "system_reason")
        if self.accelerator_reason is not None:
            _require_enum(self.accelerator_reason, ProbeReason, "accelerator_reason")
        self._validate_system_memory()
        self._validate_accelerators()
        self._validate_memory_kind()

    def _validate_system_memory(self) -> None:
        if self.system_state is SystemMemoryState.OBSERVED:
            _require_bytes(self.total_bytes, "total_bytes")
            _require_bytes(self.available_bytes, "available_bytes", allow_zero=True)
            if self.available_bytes > self.total_bytes:  # type: ignore[operator]
                raise ValueError("available_bytes may not exceed total_bytes")
            if self.system_reason is not None:
                raise ValueError("observed system memory has no system_reason")
            return
        if self.system_state is SystemMemoryState.PARTIAL:
            _require_bytes(self.total_bytes, "total_bytes")
            if self.available_bytes is not None:
                raise ValueError("partial system memory omits available_bytes")
            if self.system_reason not in {
                None,
                ProbeReason.MEMORY_UNAVAILABLE,
                ProbeReason.INVALID_MEMORY_VALUE,
            }:
                raise ValueError("partial system memory needs a partial-memory reason")
            return
        if self.total_bytes is not None or self.available_bytes is not None:
            raise ValueError("unavailable system memory has no byte values")
        expected_reason = {
            SystemMemoryState.UNAVAILABLE: ProbeReason.MEMORY_UNAVAILABLE,
            SystemMemoryState.PERMISSION_DENIED: ProbeReason.PERMISSION_DENIED,
            SystemMemoryState.UNSUPPORTED: ProbeReason.UNSUPPORTED_PLATFORM,
        }[self.system_state]
        if self.system_reason is not expected_reason:
            raise ValueError("system_state requires its matching system_reason")

    def _validate_accelerators(self) -> None:
        if type(self.accelerators) is not tuple:
            raise ValueError("accelerators must be a tuple")
        if len(self.accelerators) > MAX_ACCELERATORS:
            raise ValueError("accelerators may contain at most 16 observations")
        labels: set[str] = set()
        for observation in self.accelerators:
            if type(observation) is not AcceleratorMemoryObservation:
                raise ValueError("accelerators must contain observations")
            label_key = observation.label.casefold()
            if label_key in labels:
                raise ValueError("duplicate accelerator label")
            labels.add(label_key)
        if self.accelerator_state is AcceleratorState.OBSERVED:
            if not self.accelerators or self.accelerator_reason is not None:
                raise ValueError("observed accelerators need facts and no reason")
            return
        if self.accelerator_state is AcceleratorState.PARTIAL:
            empty_darwin_fallback = (
                not self.accelerators
                and self.platform == "darwin"
                and self.architecture not in {"arm64", "aarch64"}
                and self.accelerator_reason is ProbeReason.UNSUPPORTED_PLATFORM
            )
            if empty_darwin_fallback:
                return
            if not self.accelerators or self.accelerator_reason is None:
                raise ValueError("partial accelerators need facts and a reason")
            return
        if self.accelerators:
            raise ValueError("unavailable accelerators may not have observations")
        if self.accelerator_state is AcceleratorState.NOT_OBSERVED:
            return
        if self.accelerator_state is AcceleratorState.PERMISSION_DENIED:
            if self.accelerator_reason not in {
                ProbeReason.PERMISSION_DENIED,
                ProbeReason.SYSFS_PERMISSION_DENIED,
            }:
                raise ValueError("permission-denied accelerators need an access reason")
            return
        if self.accelerator_reason is not ProbeReason.UNSUPPORTED_PLATFORM:
            raise ValueError("unsupported accelerators need unsupported_platform")

    def _validate_memory_kind(self) -> None:
        if self.memory_kind is MemoryKind.UNKNOWN:
            if self.total_bytes is not None:
                raise ValueError("unknown memory_kind has no total_bytes")
            return
        if self.total_bytes is None:
            raise ValueError("known memory_kind requires total_bytes")
        if self.memory_kind is MemoryKind.SYSTEM:
            return
        if self.platform != "darwin":
            raise ValueError("unified memory_kind is only supported on darwin")
        if not any(observation.shared for observation in self.accelerators):
            raise ValueError("unified memory_kind requires a shared observation")


@dataclass(frozen=True, slots=True)
class ContextMemoryEstimate:
    """One exact context-length memory comparison result."""

    context_tokens: int
    model_bytes: int | None
    runtime_allowance_bytes: int | None
    context_allowance_bytes: int | None
    estimated_bytes: int | None
    ram_working_budget_bytes: int | None
    total_physical_bytes: int | None
    capacity_state: CapacityState

    def __post_init__(self) -> None:
        if type(self.context_tokens) is not int or self.context_tokens not in {
            CONTEXT_32K,
            CONTEXT_64K,
        }:
            raise ValueError("context_tokens must be 32768 or 65536")
        _require_enum(self.capacity_state, CapacityState, "capacity_state")
        values = (
            (self.model_bytes, "model_bytes", MAX_INPUT_BYTES),
            (
                self.runtime_allowance_bytes,
                "runtime_allowance_bytes",
                MAX_PROJECTED_BYTES,
            ),
            (
                self.context_allowance_bytes,
                "context_allowance_bytes",
                MAX_PROJECTED_BYTES,
            ),
            (self.estimated_bytes, "estimated_bytes", MAX_PROJECTED_BYTES),
            (
                self.ram_working_budget_bytes,
                "ram_working_budget_bytes",
                MAX_INPUT_BYTES,
            ),
            (self.total_physical_bytes, "total_physical_bytes", MAX_INPUT_BYTES),
        )
        if self.capacity_state is CapacityState.UNKNOWN:
            if any(value is not None for value, _, _ in values):
                raise ValueError("unknown estimate has no numeric values")
            return
        for value, name, maximum in values:
            _require_bytes(
                value,
                name,
                allow_zero=name == "ram_working_budget_bytes",
                maximum=maximum,
            )
        if self.ram_working_budget_bytes > self.total_physical_bytes:  # type: ignore[operator]
            raise ValueError(
                "ram_working_budget_bytes may not exceed total_physical_bytes"
            )
        try:
            expected_estimated_bytes = _bounded_sum(
                self.model_bytes,  # type: ignore[arg-type]
                self.runtime_allowance_bytes,  # type: ignore[arg-type]
                self.context_allowance_bytes,  # type: ignore[arg-type]
            )
        except OverflowError as exc:
            raise ValueError("estimate components exceed projection bound") from exc
        if self.estimated_bytes != expected_estimated_bytes:
            raise ValueError("estimated_bytes must equal the estimate components")
        expected_state = _capacity_state(
            self.estimated_bytes,  # type: ignore[arg-type]
            self.ram_working_budget_bytes,  # type: ignore[arg-type]
            self.total_physical_bytes,  # type: ignore[arg-type]
        )
        if self.capacity_state is not expected_state:
            raise ValueError("capacity_state does not match estimate boundaries")


@dataclass(frozen=True, slots=True)
class GGUFMemoryProjection:
    """Paired 32K/64K scenarios and their stable plus volatile outcomes."""

    context_32k: ContextMemoryEstimate
    context_64k: ContextMemoryEstimate
    primary_state: CapacityState
    current_pressure: CurrentPressure

    def __post_init__(self) -> None:
        if type(self.context_32k) is not ContextMemoryEstimate:
            raise ValueError("context_32k must be a ContextMemoryEstimate")
        if type(self.context_64k) is not ContextMemoryEstimate:
            raise ValueError("context_64k must be a ContextMemoryEstimate")
        if self.context_32k.context_tokens != CONTEXT_32K:
            raise ValueError("context_32k must use 32768 tokens")
        if self.context_64k.context_tokens != CONTEXT_64K:
            raise ValueError("context_64k must use 65536 tokens")
        _require_enum(self.primary_state, CapacityState, "primary_state")
        _require_enum(self.current_pressure, CurrentPressure, "current_pressure")
        expected_primary = _primary_state(
            self.context_32k.capacity_state, self.context_64k.capacity_state
        )
        if self.primary_state is not expected_primary:
            raise ValueError("primary_state does not match context estimates")
        if self.primary_state is CapacityState.UNKNOWN:
            if (
                self.context_32k.capacity_state is not CapacityState.UNKNOWN
                or self.context_64k.capacity_state is not CapacityState.UNKNOWN
            ):
                raise ValueError(
                    "unknown projection requires unknown context estimates"
                )
            if self.current_pressure is not CurrentPressure.UNKNOWN:
                raise ValueError("unknown projection has unknown current_pressure")
            return
        if (
            self.context_32k.model_bytes != self.context_64k.model_bytes
            or self.context_32k.runtime_allowance_bytes
            != self.context_64k.runtime_allowance_bytes
            or self.context_32k.ram_working_budget_bytes
            != self.context_64k.ram_working_budget_bytes
            or self.context_32k.total_physical_bytes
            != self.context_64k.total_physical_bytes
        ):
            raise ValueError("context estimates must share model and machine facts")
        try:
            expected_64k_allowance = _bounded_sum(
                self.context_32k.context_allowance_bytes,  # type: ignore[arg-type]
                self.context_32k.context_allowance_bytes,  # type: ignore[arg-type]
            )
        except OverflowError as exc:
            raise ValueError("32K context allowance exceeds projection bound") from exc
        if self.context_64k.context_allowance_bytes != expected_64k_allowance:
            raise ValueError("64K context allowance must be twice the 32K allowance")


def _ceil_percent_mib(value: int, numerator: int, denominator: int = 100) -> int:
    """Return a percentage rounded upward to the next whole MiB."""
    units = (value * numerator + denominator * MIB - 1) // (denominator * MIB)
    result = units * MIB
    if result > MAX_PROJECTED_BYTES:
        raise OverflowError("percentage allowance exceeds projection bound")
    return result


def _bounded_sum(*values: int) -> int:
    """Return a checked projection sum without relying on unbounded Python ints."""
    total = sum(values)
    if total > MAX_PROJECTED_BYTES:
        raise OverflowError("projection exceeds 64-bit bound")
    return total


def _capacity_state(
    estimated_bytes: int, ram_working_budget_bytes: int, total_physical_bytes: int
) -> CapacityState:
    if estimated_bytes <= ram_working_budget_bytes:
        return CapacityState.WITHIN_BUDGET
    if estimated_bytes <= total_physical_bytes:
        return CapacityState.OVER_RESERVE
    return CapacityState.OVER_TOTAL


def _primary_state(state_32k: CapacityState, state_64k: CapacityState) -> CapacityState:
    if CapacityState.UNKNOWN in {state_32k, state_64k}:
        return CapacityState.UNKNOWN
    if state_64k is CapacityState.WITHIN_BUDGET:
        return CapacityState.WITHIN_BUDGET
    if state_32k is CapacityState.WITHIN_BUDGET:
        return state_64k
    return state_32k


def _unknown_projection() -> GGUFMemoryProjection:
    unknown_32k = ContextMemoryEstimate(
        context_tokens=CONTEXT_32K,
        model_bytes=None,
        runtime_allowance_bytes=None,
        context_allowance_bytes=None,
        estimated_bytes=None,
        ram_working_budget_bytes=None,
        total_physical_bytes=None,
        capacity_state=CapacityState.UNKNOWN,
    )
    unknown_64k = ContextMemoryEstimate(
        context_tokens=CONTEXT_64K,
        model_bytes=None,
        runtime_allowance_bytes=None,
        context_allowance_bytes=None,
        estimated_bytes=None,
        ram_working_budget_bytes=None,
        total_physical_bytes=None,
        capacity_state=CapacityState.UNKNOWN,
    )
    return GGUFMemoryProjection(
        context_32k=unknown_32k,
        context_64k=unknown_64k,
        primary_state=CapacityState.UNKNOWN,
        current_pressure=CurrentPressure.UNKNOWN,
    )


def _build_estimate(
    *,
    context_tokens: int,
    model_bytes: int,
    runtime_allowance_bytes: int,
    context_allowance_bytes: int,
    ram_working_budget_bytes: int,
    total_physical_bytes: int,
) -> ContextMemoryEstimate:
    estimated_bytes = _bounded_sum(
        model_bytes, runtime_allowance_bytes, context_allowance_bytes
    )
    return ContextMemoryEstimate(
        context_tokens=context_tokens,
        model_bytes=model_bytes,
        runtime_allowance_bytes=runtime_allowance_bytes,
        context_allowance_bytes=context_allowance_bytes,
        estimated_bytes=estimated_bytes,
        ram_working_budget_bytes=ram_working_budget_bytes,
        total_physical_bytes=total_physical_bytes,
        capacity_state=_capacity_state(
            estimated_bytes, ram_working_budget_bytes, total_physical_bytes
        ),
    )


def project_gguf_memory(
    model_bytes: int,
    snapshot: MachineMemorySnapshot,
) -> GGUFMemoryProjection:
    """Project independent 32K and 64K RAM scenarios for one GGUF candidate."""
    if (
        type(model_bytes) is not int
        or not 1 <= model_bytes <= MAX_INPUT_BYTES
        or type(snapshot) is not MachineMemorySnapshot
        or snapshot.total_bytes is None
        or snapshot.system_state
        not in {SystemMemoryState.OBSERVED, SystemMemoryState.PARTIAL}
    ):
        return _unknown_projection()
    try:
        runtime = max(GIB, _ceil_percent_mib(model_bytes, 10))
        allowance_32k = max(4 * GIB, _ceil_percent_mib(model_bytes, 25))
        allowance_64k = _bounded_sum(allowance_32k, allowance_32k)
        reserve = max(2 * GIB, _ceil_percent_mib(snapshot.total_bytes, 20))
        budget = max(0, snapshot.total_bytes - reserve)
        context_32k = _build_estimate(
            context_tokens=CONTEXT_32K,
            model_bytes=model_bytes,
            runtime_allowance_bytes=runtime,
            context_allowance_bytes=allowance_32k,
            ram_working_budget_bytes=budget,
            total_physical_bytes=snapshot.total_bytes,
        )
        context_64k = _build_estimate(
            context_tokens=CONTEXT_64K,
            model_bytes=model_bytes,
            runtime_allowance_bytes=runtime,
            context_allowance_bytes=allowance_64k,
            ram_working_budget_bytes=budget,
            total_physical_bytes=snapshot.total_bytes,
        )
    except OverflowError:
        return _unknown_projection()
    primary_state = _primary_state(
        context_32k.capacity_state, context_64k.capacity_state
    )
    if snapshot.available_bytes is None:
        current_pressure = CurrentPressure.UNKNOWN
    elif snapshot.available_bytes >= context_64k.estimated_bytes:
        current_pressure = CurrentPressure.NONE
    elif snapshot.available_bytes >= context_32k.estimated_bytes:
        current_pressure = CurrentPressure.NEEDS_MORE_FOR_64K
    else:
        current_pressure = CurrentPressure.NEEDS_MORE_FOR_BOTH
    return GGUFMemoryProjection(
        context_32k=context_32k,
        context_64k=context_64k,
        primary_state=primary_state,
        current_pressure=current_pressure,
    )


def format_gib(value: int) -> str:
    """Render a bounded byte count as a one-decimal binary GiB string."""
    _require_bytes(value, "value", allow_zero=True, maximum=MAX_PROJECTED_BYTES)
    return f"{value / GIB:.1f} GiB"
