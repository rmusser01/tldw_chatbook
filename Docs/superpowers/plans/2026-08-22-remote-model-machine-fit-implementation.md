# Remote Models Memory Scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add private, bounded machine-memory evidence and transparent 32,768-/65,536-token RAM scenarios to Remote Models without implying model-context support or runtime compatibility.

**Architecture:** A provider/runtime-neutral domain module validates facts and projects exact integer policy values; a separate local probe module observes bounded RAM and optional VRAM. The recomposition-stable `LLMScreen` owns the session snapshot, generation, timestamps, and worker, while `RemoteView` renders immutable presentation state and posts recheck intent. At less than 72 content cells Remote becomes a one-pane repository-to-detail drill-down.

**Tech Stack:** Python 3.11, frozen slot-backed dataclasses and `StrEnum`, psutil, bounded `subprocess.Popen`, Textual 8.x workers/messages/widgets, pytest/pytest-asyncio.

**Spec:** `Docs/superpowers/specs/2026-08-22-remote-model-machine-fit-design.md`

## Global Constraints

- ADR required: yes.
- ADR path: `backlog/decisions/080-model-machine-memory-fit-estimation.md`.
- Reason: the feature creates a long-lived local capability boundary, platform probe contract, privacy policy, and Models-screen lifecycle ownership rule.
- `32K` means exactly 32,768 tokens; `64K` means exactly 65,536 tokens.
- Outcomes are `within_budget`, `over_reserve`, `over_total`, or `unknown`; user copy never says bare “fits,” “compatible,” or “supported.”
- RAM working budget is total physical RAM minus `max(2 GiB, ceil_MiB(total × 20%))`.
- Current available RAM adds a warning only; it never changes classification, sorting, eligibility, or installation.
- VRAM is per-device observed evidence only, is never summed, never affects classification, and Apple unified memory appears once.
- No network/header requests, ML/runtime imports, configuration writes, persistence, permission prompts, PATH lookup, or raw exception/value logging.
- Candidate totals above `2**63 - 1` yield an unknown projection but remain selectable/installable.
- Probe execution stays off the Textual event loop and only the current screen-owned generation may publish.
- Below 72 measured `RemoteView` cells, use one-pane drill-down with Back and collapsed details; do not hide any core action.
- Run targeted tests only during implementation. Ask before any full-suite sweep.

---

### Task 1: Define the immutable memory domain and pure projection policy

**Files:**
- Create: `tldw_chatbook/Model_Artifacts/machine_memory.py`
- Create: `Tests/Model_Artifacts/test_machine_memory.py`

**Interfaces:**
- Produces: `SystemMemoryState`, `AcceleratorState`, `MemoryKind`, `AcceleratorSource`, `CapacityState`, `CurrentPressure`, `ProbeReason`.
- Produces: `AcceleratorMemoryObservation`, `MachineMemorySnapshot`, `ContextMemoryEstimate`, `GGUFMemoryProjection`.
- Produces: `project_gguf_memory(model_bytes: int, snapshot: MachineMemorySnapshot) -> GGUFMemoryProjection` and `format_gib(value: int) -> str`.
- Consumes: no provider, Textual, runtime, filesystem, subprocess, or network type.

- [ ] **Step 1: Write failing validation and exact-boundary tests**

```python
from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Model_Artifacts.machine_memory import (
    AcceleratorState,
    CapacityState,
    MachineMemorySnapshot,
    MemoryKind,
    SystemMemoryState,
    project_gguf_memory,
)


def _snapshot(
    *,
    total_gib: int = 32,
    available_gib: int = 20,
    total_bytes: object | None = None,
) -> MachineMemorySnapshot:
    total = total_gib * 1024**3 if total_bytes is None else total_bytes
    return MachineMemorySnapshot(
        platform="linux",
        architecture="x86_64",
        system_state=SystemMemoryState.OBSERVED,
        accelerator_state=AcceleratorState.NOT_OBSERVED,
        total_bytes=total,  # type: ignore[arg-type]
        available_bytes=available_gib * 1024**3,
        memory_kind=MemoryKind.SYSTEM,
        accelerators=(),
        system_reason=None,
        accelerator_reason=None,
    )


def test_snapshot_is_frozen_and_rejects_boolean_memory() -> None:
    snapshot = _snapshot(total_gib=32, available_gib=20)
    with pytest.raises(FrozenInstanceError):
        snapshot.total_bytes = 1  # type: ignore[misc]
    with pytest.raises(ValueError):
        _snapshot(total_bytes=True)  # type: ignore[arg-type]


def test_projection_uses_exact_32768_and_65536_scenarios() -> None:
    projection = project_gguf_memory(4 * 1024**3, _snapshot(total_gib=32))
    assert projection.context_32k.context_tokens == 32_768
    assert projection.context_64k.context_tokens == 65_536
    assert projection.context_32k.runtime_allowance_bytes == 1024**3
    assert projection.context_32k.context_allowance_bytes == 4 * 1024**3
    assert projection.context_64k.context_allowance_bytes == 8 * 1024**3
    assert projection.context_64k.capacity_state is CapacityState.WITHIN_BUDGET


@pytest.mark.parametrize("model_bytes", [0, -1, True, 2**63])
def test_invalid_candidate_is_unknown_without_throwing(model_bytes: object) -> None:
    projection = project_gguf_memory(model_bytes, _snapshot())  # type: ignore[arg-type]
    assert projection.primary_state is CapacityState.UNKNOWN
```

- [ ] **Step 2: Run the new policy tests and confirm they fail on the missing module**

Run: `pytest -q Tests/Model_Artifacts/test_machine_memory.py`

Expected: collection fails with `ModuleNotFoundError: ...machine_memory`.

- [ ] **Step 3: Implement bounded enums and frozen dataclasses**

```python
MIB = 1 << 20
GIB = 1 << 30
MAX_INPUT_BYTES = (1 << 63) - 1
MAX_PROJECTED_BYTES = (1 << 64) - 1
CONTEXT_32K = 32_768
CONTEXT_64K = 65_536
MAX_ACCELERATORS = 16


class SystemMemoryState(StrEnum):
    OBSERVED = "observed"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED = "unsupported"


class AcceleratorState(StrEnum):
    OBSERVED = "observed"
    PARTIAL = "partial"
    NOT_OBSERVED = "not_observed"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED = "unsupported"


class MemoryKind(StrEnum):
    UNIFIED = "unified"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class AcceleratorSource(StrEnum):
    APPLE_UNIFIED = "apple_unified"
    NVIDIA_SMI = "nvidia_smi"
    LINUX_DRM = "linux_drm"


class CurrentPressure(StrEnum):
    NONE = "none"
    NEEDS_MORE_FOR_64K = "needs_more_for_64k"
    NEEDS_MORE_FOR_BOTH = "needs_more_for_both"
    UNKNOWN = "unknown"


class CapacityState(StrEnum):
    WITHIN_BUDGET = "within_budget"
    OVER_RESERVE = "over_reserve"
    OVER_TOTAL = "over_total"
    UNKNOWN = "unknown"


class ProbeReason(StrEnum):
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


@dataclass(frozen=True, slots=True)
class AcceleratorMemoryObservation:
    vendor: str
    label: str
    total_bytes: int | None
    shared: bool
    source: AcceleratorSource


@dataclass(frozen=True, slots=True)
class MachineMemorySnapshot:
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


@dataclass(frozen=True, slots=True)
class ContextMemoryEstimate:
    context_tokens: int
    model_bytes: int | None
    runtime_allowance_bytes: int | None
    context_allowance_bytes: int | None
    estimated_bytes: int | None
    ram_working_budget_bytes: int | None
    total_physical_bytes: int | None
    capacity_state: CapacityState


@dataclass(frozen=True, slots=True)
class GGUFMemoryProjection:
    context_32k: ContextMemoryEstimate
    context_64k: ContextMemoryEstimate
    primary_state: CapacityState
    current_pressure: CurrentPressure
```

Implement strict `__post_init__` validation with `type(value) is int`, closed enums, exact character/length bounds, at most 16 accelerators, and state/value consistency.

- [ ] **Step 4: Implement exact integer projection and formatting**

```python
def _ceil_percent_mib(value: int, numerator: int, denominator: int = 100) -> int:
    units = (value * numerator + denominator * MIB - 1) // (denominator * MIB)
    return units * MIB


def project_gguf_memory(
    model_bytes: int,
    snapshot: MachineMemorySnapshot,
) -> GGUFMemoryProjection:
    if type(model_bytes) is not int or not 1 <= model_bytes <= MAX_INPUT_BYTES:
        return _unknown_projection()
    if snapshot.total_bytes is None:
        return _unknown_projection()
    runtime = max(GIB, _ceil_percent_mib(model_bytes, 10))
    allowance_32k = max(4 * GIB, _ceil_percent_mib(model_bytes, 25))
    allowance_64k = allowance_32k * 2
    reserve = max(2 * GIB, _ceil_percent_mib(snapshot.total_bytes, 20))
    budget = max(0, snapshot.total_bytes - reserve)
    return _build_projection(
        model_bytes, runtime, allowance_32k, allowance_64k, budget, snapshot
    )
```

Classify `estimated <= budget`, `estimated <= total`, and `estimated > total` exactly. Derive pressure independently from `available_bytes` against 32K/64K totals.

- [ ] **Step 5: Add mutation-sensitive boundary tests**

Cover one byte below/at/above RAM budget and total RAM; reserve floor/percentage branches; one-byte MiB rounding; `2**63 - 1`; derived overflow; available-memory pressure; Apple shared markers; duplicate accelerator labels; string controls; and `format_gib` output.

- [ ] **Step 6: Run targeted tests and static checks**

```bash
pytest -q Tests/Model_Artifacts/test_machine_memory.py
ruff check tldw_chatbook/Model_Artifacts/machine_memory.py Tests/Model_Artifacts/test_machine_memory.py
python -m compileall -q tldw_chatbook/Model_Artifacts/machine_memory.py
```

Expected: all pass with no Ruff or compilation output.

- [ ] **Step 7: Commit the pure domain**

```bash
git add tldw_chatbook/Model_Artifacts/machine_memory.py Tests/Model_Artifacts/test_machine_memory.py
git commit -m "feat(models): define machine memory scenarios"
```

---

### Task 2: Implement bounded local RAM and accelerator observation

**Files:**
- Create: `tldw_chatbook/Model_Artifacts/machine_memory_probe.py`
- Create: `Tests/Model_Artifacts/test_machine_memory_probe.py`
- Modify: `pyproject.toml` only if the existing required psutil declaration is absent; do not add a new dependency.

**Interfaces:**
- Consumes: domain types and constants from `machine_memory.py`.
- Produces: `CommandResult`, `MachineProbeSources`, and `observe_machine_memory(*, sources: MachineProbeSources | None = None) -> MachineMemorySnapshot`.
- Production sources wrap platform/architecture, `psutil.virtual_memory`, bounded `Path` operations, and `subprocess.Popen`; LLMScreen owns injected clocks separately.

```python
@dataclass(frozen=True, slots=True)
class CommandResult:
    return_code: int | None
    output: bytes
    reason: ProbeReason | None


@dataclass(frozen=True, slots=True)
class MachineProbeSources:
    platform_name: Callable[[], str]
    architecture: Callable[[], str]
    virtual_memory: Callable[[], object]
    lstat_path: Callable[[Path], os.stat_result]
    resolve_path: Callable[[Path], Path]
    read_bounded: Callable[[Path, int], bytes]
    drm_cards: Callable[[], tuple[Path, ...]]
    run_command: Callable[[Path, tuple[str, ...], float, int], CommandResult]
```

- [ ] **Step 1: Write failing deterministic platform tests**

```python
import stat
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock


def _sources(
    *,
    platform_name: str,
    architecture: str,
    total: int,
    available: int,
    run_command: Callable[..., CommandResult],
) -> MachineProbeSources:
    trusted_stat = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o755,
        st_uid=0,
        st_file_attributes=0,
    )
    return MachineProbeSources(
        platform_name=lambda: platform_name,
        architecture=lambda: architecture,
        virtual_memory=lambda: SimpleNamespace(total=total, available=available),
        lstat_path=lambda _path: trusted_stat,
        resolve_path=lambda path: path,
        read_bounded=lambda _path, _limit: b"",
        drm_cards=lambda: (),
        run_command=run_command,
    )


def test_darwin_arm64_reports_one_unified_pool_without_accelerator_command() -> None:
    runner = Mock()
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="darwin",
            architecture="arm64",
            total=32 * GIB,
            available=18 * GIB,
            run_command=runner,
        )
    )
    assert snapshot.memory_kind is MemoryKind.UNIFIED
    assert snapshot.total_bytes == 32 * GIB
    assert len(snapshot.accelerators) == 1
    assert snapshot.accelerators[0].shared is True
    runner.assert_not_called()


def test_linux_keeps_valid_ram_when_nvidia_output_is_malformed() -> None:
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(0, b"bad,row\n", None),
        )
    )
    assert snapshot.system_state is SystemMemoryState.OBSERVED
    assert snapshot.accelerator_state is AcceleratorState.NOT_OBSERVED
    assert snapshot.total_bytes == 64 * GIB
```

- [ ] **Step 2: Run probe tests and confirm the module is missing**

Run: `pytest -q Tests/Model_Artifacts/test_machine_memory_probe.py`

Expected: collection fails with `ModuleNotFoundError: ...machine_memory_probe`.

- [ ] **Step 3: Implement system-memory and Apple unified observation**

Normalize only `darwin`, `linux`, `windows`, or `other`; catch exceptions into `ProbeReason` without logging raw values. Validate total and available independently. Darwin arm64/aarch64 returns one shared Apple observation and never enters discrete probes.

```python
def observe_machine_memory(
    *,
    sources: MachineProbeSources | None = None,
) -> MachineMemorySnapshot:
    active = sources or production_probe_sources()
    platform_name = _normalize_platform(active.platform_name())
    architecture = _sanitize_identifier(active.architecture())
    if platform_name not in {"darwin", "linux", "windows"}:
        return _unsupported_snapshot(platform_name, architecture)
    total, available, state, reason = _observe_system_memory(active.virtual_memory)
    if total is None:
        return _snapshot_without_capacity(platform_name, architecture, state, reason)
    if platform_name == "darwin" and architecture in {"arm64", "aarch64"}:
        return _apple_unified_snapshot(total, available, state)
    accelerators, accelerator_state, accelerator_reason = _observe_accelerators(
        platform_name,
        active,
    )
    return MachineMemorySnapshot(
        platform=platform_name,
        architecture=architecture,
        system_state=state,
        accelerator_state=accelerator_state,
        total_bytes=total,
        available_bytes=available,
        memory_kind=MemoryKind.SYSTEM,
        accelerators=accelerators,
        system_reason=reason,
        accelerator_reason=accelerator_reason,
    )
```

- [ ] **Step 4: Implement exact trusted NVIDIA discovery and parsing**

```python
LINUX_NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
WINDOWS_NVIDIA_SMI = (
    Path(r"C:\Windows\System32\nvidia-smi.exe"),
    Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
)
NVIDIA_ARGV = (
    "--query-gpu=index,name,memory.total",
    "--format=csv,noheader,nounits",
)
COMMAND_TIMEOUT_SECONDS = 2.0
TERMINATE_GRACE_SECONDS = 0.25
MAX_COMMAND_OUTPUT_BYTES = 64 * 1024
```

Reject PATH lookup, symlinks/reparse points, non-regular files, and on Linux non-root-owned or group/world-writable executable/parent paths. Parse at most 16 unique integer indexes, bounded printable names, and positive MiB totals no greater than `MAX_INPUT_BYTES` after conversion.

- [ ] **Step 5: Implement bounded process cleanup and AMD DRM observation**

Use `stderr=subprocess.STDOUT`, bounded incremental reads, and always reap. Timeout/oversize must call `terminate()`, wait 250 ms, then `kill()` and wait when needed. Resolve at most 16 `cardN/device` paths below `/sys/devices`; read ASCII `vendor` and `mem_info_vram_total` with a 64-byte cap; accept only vendor `0x1002`. Describe results as `DRM-reported VRAM`.

- [ ] **Step 6: Add failure, trust, and privacy tests**

Use fakes for `Popen`, stat/path, and sysfs. Cover nonzero exit, timeout, oversized output before accumulation, terminate/kill/wait order, malformed CSV, duplicate/17 devices, symlink/reparse, Linux owner/mode, escaping sysfs resolution, unsupported/permission-denied psutil, invalid available memory, NVIDIA memory bounds, AMD-only DRM, and absence of hostnames/UUID/PCI addresses/raw exceptions/logs.

- [ ] **Step 7: Run targeted tests and static checks**

```bash
pytest -q Tests/Model_Artifacts/test_machine_memory.py Tests/Model_Artifacts/test_machine_memory_probe.py
ruff check tldw_chatbook/Model_Artifacts/machine_memory.py tldw_chatbook/Model_Artifacts/machine_memory_probe.py Tests/Model_Artifacts/test_machine_memory.py Tests/Model_Artifacts/test_machine_memory_probe.py
python -m compileall -q tldw_chatbook/Model_Artifacts/machine_memory.py tldw_chatbook/Model_Artifacts/machine_memory_probe.py
```

Expected: all pass; tests execute no real accelerator utility.

- [ ] **Step 8: Commit bounded observation**

```bash
git add tldw_chatbook/Model_Artifacts/machine_memory_probe.py Tests/Model_Artifacts/test_machine_memory_probe.py
git commit -m "feat(models): observe bounded local memory facts"
```

---

### Task 3: Define pure memory presentation copy

**Files:**
- Create: `tldw_chatbook/UI/Screens/model_memory_presenter.py`
- Create: `Tests/UI/test_model_memory_presenter.py`

**Interfaces:**
- Produces: `CandidateMemoryPresentation`, `MachineMemoryPresentation`, `build_candidate_memory_presentation`, and `build_machine_memory_presentation`.

- [ ] **Step 1: Write failing exact-copy tests**

```python
def _projection(*, available_gib: int = 21) -> GGUFMemoryProjection:
    snapshot = MachineMemorySnapshot(
        platform="linux",
        architecture="x86_64",
        system_state=SystemMemoryState.OBSERVED,
        accelerator_state=AcceleratorState.NOT_OBSERVED,
        total_bytes=32 * GIB,
        available_bytes=available_gib * GIB,
        memory_kind=MemoryKind.SYSTEM,
        accelerators=(),
        system_reason=None,
        accelerator_reason=None,
    )
    return project_gguf_memory(4 * GIB, snapshot)


def test_within_budget_copy_names_scenario_and_disclaims_context_support() -> None:
    presentation = build_candidate_memory_presentation(_projection())
    assert presentation.outcome == "64K scenario within RAM budget · 12.6 GiB headroom"
    assert presentation.details.startswith("32K est. 9.0 GiB · 64K est. 13.0 GiB")
    assert "fits" not in presentation.outcome.lower()


def test_current_pressure_is_separate_from_capacity_copy() -> None:
    presentation = build_candidate_memory_presentation(_projection(available_gib=10))
    assert presentation.outcome.startswith("64K scenario within RAM budget")
    assert presentation.pressure == "64K may need more free RAM now"
```

Add table-driven tests for every system/accelerator/probe/refresh state, observed-at retained copy, Apple unified evidence, VRAM-not-used copy, and overflow-device copy.

- [ ] **Step 2: Run presenter tests and confirm failure**

Run: `pytest -q Tests/UI/test_model_memory_presenter.py`

Expected: collection fails with `ModuleNotFoundError: ...model_memory_presenter`.

- [ ] **Step 3: Implement immutable presentation values and complete copy matrix**

```python
@dataclass(frozen=True, slots=True)
class CandidateMemoryPresentation:
    outcome: str
    details: str
    pressure: str | None


@dataclass(frozen=True, slots=True)
class MachineMemoryPresentation:
    headline: str
    evidence_lines: tuple[str, ...]
    limitation_lines: tuple[str, ...]
    action_label: str
    action_disabled: bool
    failure_line: str | None
```

Use complete, non-concatenated user messages where reordering matters. Every rendered remote value remains bounded and later uses `markup=False`.

- [ ] **Step 4: Run targeted presenter tests and static checks**

```bash
pytest -q Tests/UI/test_model_memory_presenter.py
ruff check tldw_chatbook/UI/Screens/model_memory_presenter.py Tests/UI/test_model_memory_presenter.py
python -m compileall -q tldw_chatbook/UI/Screens/model_memory_presenter.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the pure presenter**

```bash
git add tldw_chatbook/UI/Screens/model_memory_presenter.py Tests/UI/test_model_memory_presenter.py
git commit -m "feat(models): define machine memory presentation"
```

---

### Task 4: Render memory evidence and adaptive drill-down in RemoteView

**Files:**
- Modify: `tldw_chatbook/UI/Screens/model_remote_view.py`
- Modify: `tldw_chatbook/UI/Screens/llm_screen.py`
- Modify: `Tests/UI/test_model_remote_view.py`
- Modify: `Tests/UI/test_llm_screen_lab_adoption.py`

**Interfaces:**
- Consumes: domain projection and pure presentation values from Tasks 1 and 3.
- `LLMScreen` owns `_machine_memory_snapshot`, `_machine_memory_observed_label`, `_machine_memory_generation`, `_machine_memory_worker`, `_machine_memory_active`, and `_machine_memory_failure`.
- Produces: `LLMScreen._request_remote_machine_memory(*, force: bool)`, `_run_machine_memory_probe(generation: int)`, `_apply_machine_memory_result(generation, result)`, and `_hydrate_remote_machine_memory()`.
- Produces: `RemoteView.MachineMemoryRequested`, `apply_machine_memory_state`, `_show_repository_results`, `_show_repository_detail`, focus-locator restoration for one-pane mode, and the thin LLMScreen message handler that delegates to `_request_remote_machine_memory(force=event.force)`.
- Keeps acquisition ownership and existing exact candidate identity unchanged.

- [ ] **Step 1: Write failing LLMScreen generation and recomposition tests**

Add table-driven tests proving: first repository intent starts one screen worker; a forced recheck increments generation; stale completion is ignored; unavailable refresh retains accepted RAM; partial valid RAM replaces it; accelerator failure does not discard RAM; completion during the remount gap is retained; `DeferredViewsMounted` hydrates the replacement `RemoteView`; and recompose does not start a duplicate probe.

```python
def _machine_screen() -> LLMScreen:
    screen = LLMScreen.__new__(LLMScreen)
    screen._machine_memory_snapshot = None
    screen._machine_memory_observed_label = None
    screen._machine_memory_generation = 0
    screen._machine_memory_worker = None
    screen._machine_memory_active = False
    screen._machine_memory_failure = None
    screen._hydrate_remote_machine_memory = Mock(return_value=False)
    return screen


def test_stale_machine_result_cannot_replace_newer_snapshot() -> None:
    screen = _machine_screen()
    screen._machine_memory_generation = 2
    current = _machine_snapshot(total_gib=32)
    screen._machine_memory_snapshot = current

    screen._apply_machine_memory_result(1, _machine_snapshot(total_gib=64))

    assert screen._machine_memory_snapshot is current


def test_failed_recheck_retains_last_valid_ram() -> None:
    screen = _machine_screen()
    current = _machine_snapshot(total_gib=32)
    screen._machine_memory_snapshot = current
    screen._machine_memory_generation = 3
    screen._apply_machine_memory_result(3, _unavailable_snapshot())
    assert screen._machine_memory_snapshot is current
    assert screen._machine_memory_failure is ProbeReason.MEMORY_UNAVAILABLE
```

- [ ] **Step 2: Implement the screen-owned worker and transition table**

```python
def _request_remote_machine_memory(self, *, force: bool) -> None:
    if self._machine_memory_active and not force:
        return
    if self._machine_memory_snapshot is not None and not force:
        self._hydrate_remote_machine_memory()
        return
    self._machine_memory_generation += 1
    generation = self._machine_memory_generation
    self._machine_memory_active = True
    self._machine_memory_worker = self._run_machine_memory_probe(generation)


@work(
    thread=True,
    group="remote_machine_memory",
    exclusive=True,
    exit_on_error=False,
    description="Observe local model memory capacity",
)
def _run_machine_memory_probe(self, generation: int) -> None:
    result = observe_machine_memory()
    self.app.call_from_thread(self._apply_machine_memory_result, generation, result)
```

The thread returns only a bounded snapshot/fixed reason. Apply on the event loop, retain a previous valid RAM snapshot on failed refresh, capture a fixed local `HH:MM` acceptance label, then hydrate the current view. Invoke hydration from `DeferredViewsMounted` beside existing durable install/runtime hydration.

- [ ] **Step 3: Write failing mounted copy, recheck, and focus-identity tests**

Add tests that resolve a repository, capture the candidate `Button`, apply a machine snapshot, and assert the same button remains mounted/focused while `.remote-fit-outcome` and `.remote-fit-details` update. Assert exact disclaimer, pressure, Apple, VRAM-not-used, unavailable, retained failure, `Recheck memory` → `Checking…`, and details-toggle copy.

```python
@pytest.mark.asyncio
async def test_machine_update_preserves_focused_candidate_button() -> None:
    view = _view(adapter_factory=lambda: _Adapter(resolved=_resolved()))
    app = _RemoteApp(view)
    async with app.run_test(size=(100, 30)) as pilot:
        await _submit(app, pilot, "owner/repository")
        candidate = view.query_one(".remote-candidate", Button)
        candidate.focus()
        view.apply_machine_memory_state(
            _machine_presentation(active=False),
            _machine_snapshot(total_gib=32, available_gib=10),
        )
        await pilot.pause()
        assert view.query_one(".remote-candidate", Button) is candidate
        assert app.focused is candidate
        assert "64K scenario within RAM budget" in _text(view)
        assert "64K may need more free RAM now" in _text(view)
```

- [ ] **Step 4: Write failing narrow drill-down tests**

At a measured width of 71, assert results-only initial state; repository activation shows detail with `Back to repositories`; Back restores the exact repository button; a new search returns to results; details start collapsed; all controls are reachable by Tab. At 72, assert both panes remain present and details start expanded.

```python
@pytest.mark.asyncio
async def test_71_cell_drill_down_back_restores_repository_focus() -> None:
    view = _view(adapter_factory=lambda: _Adapter(results=(_summary(),), resolved=_resolved()))
    app = _RemoteApp(view)
    async with app.run_test(size=(71, 30)) as pilot:
        await _submit(app, pilot, "test model")
        repository = view.query_one(".remote-result", Button)
        repository.focus()
        repository.press()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert view.has_class("-single-pane")
        assert view.query_one(".remote-results-pane").display is False
        back = view.query_one("#remote-back-to-results", Button)
        back.press()
        await pilot.pause()
        assert view.query_one(".remote-results-pane").display is True
        assert app.focused is repository
```

- [ ] **Step 5: Run the focused screen and RemoteView tests and confirm behavioral failures**

Run:

```bash
pytest -q Tests/UI/test_llm_screen_lab_adoption.py -k "machine_memory or recompose"
pytest -q Tests/UI/test_model_remote_view.py -k "machine_memory or memory_scenario or drill_down"
```

Expected: tests fail because the new messages, panel, row statics, and one-pane behavior do not exist.

- [ ] **Step 6: Add presentation-only state and machine-memory intent**

```python
class MachineMemoryRequested(Message):
    def __init__(self, *, force: bool) -> None:
        super().__init__()
        self.force = force


@on(RemoteView.MachineMemoryRequested)
def _remote_machine_memory_requested(
    self,
    event: RemoteView.MachineMemoryRequested,
) -> None:
    self._request_remote_machine_memory(force=event.force)


def apply_machine_memory_state(
    self,
    presentation: MachineMemoryPresentation,
    snapshot: MachineMemorySnapshot | None,
) -> None:
    self._machine_presentation = presentation
    self._machine_snapshot = snapshot
    self._update_machine_panel_in_place()
    self._update_candidate_memory_statics_in_place()
```

After the first successful repository resolution, post `MachineMemoryRequested(force=False)`. Recheck posts `force=True`. Do not instantiate psutil, probe modules, or workers in RemoteView.

- [ ] **Step 7: Add the panel, candidate copy, and details toggle**

Place the machine panel before filename guidance. Add stable per-source-index IDs to outcome/details statics so updates never replace candidate buttons. Wide mode starts detail statics visible; narrow mode starts hidden. The shared text-labeled toggle changes only their display state. Render every dynamic value with `markup=False`.

- [ ] **Step 8: Implement the 72-cell one-pane navigation state**

Replace `_NARROW_WIDTH = 64` with `_SINGLE_PANE_WIDTH = 72`. Preserve both pane containers for stable query identities, but use display classes so only results or details occupies the workspace below the breakpoint. Save a repository locator from the bounded repository identity; show `Back to repositories` at the top of detail; restore the repository button or search input without rebuilding unrelated controls.

- [ ] **Step 9: Harden long names and accelerator overflow**

Exact filenames wrap inside the scroll container. Show at most two accelerator labels in the compact panel and place all bounded observations in expanded details. Test 160-character candidate labels, 96-character device labels, two/three/16 devices, CJK/RTL printable display labels after sanitization, and empty accelerator evidence.

- [ ] **Step 10: Run all screen and RemoteView tests and static checks**

```bash
pytest -q Tests/UI/test_model_remote_view.py
pytest -q Tests/UI/test_llm_screen_lab_adoption.py -k "machine_memory or recompose"
ruff check tldw_chatbook/UI/Screens/model_remote_view.py tldw_chatbook/UI/Screens/llm_screen.py Tests/UI/test_model_remote_view.py Tests/UI/test_llm_screen_lab_adoption.py
python -m compileall -q tldw_chatbook/UI/Screens/model_remote_view.py tldw_chatbook/UI/Screens/llm_screen.py
```

Expected: all RemoteView tests pass with no import-time I/O regression.

- [ ] **Step 11: Commit screen ownership and Remote presentation**

```bash
git add tldw_chatbook/UI/Screens/model_remote_view.py tldw_chatbook/UI/Screens/llm_screen.py Tests/UI/test_model_remote_view.py Tests/UI/test_llm_screen_lab_adoption.py
git commit -m "feat(models): show adaptive memory scenarios"
```

---

### Task 5: Prove production-width behavior and close the task evidence

**Files:**
- Modify: `Tests/UI/test_llm_screen_lab_adoption.py`
- Modify: `Tests/UI/test_ui_css_parse.py` only if the CSS reproduction test requires an explicit new selector assertion.
- Modify: `backlog/tasks/task-20938 - Add-hardware-aware-GGUF-machine-fit-estimates.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` only if implementation uncovers a genuinely reusable incident.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: production `TldwCli.CSS_PATH` compositor and keyboard evidence at 80×24.

- [ ] **Step 1: Replace the obsolete 80-column two-pane expectation**

Update `test_remote_two_pane_install_action_stays_inside_real_models_body_at_80_columns` so the real Models body expects one-pane drill-down at its measured content width. Keep a separate mounted 72-cell boundary test in `test_model_remote_view.py` for the two-pane threshold.

- [ ] **Step 2: Add one production-shaped end-to-end scenario**

In a real `TldwCli.CSS_PATH` app at 80×24, exercise:

1. Remote rail activation with the catalog rail in each supported state;
2. exact repository resolution;
3. machine Checking presentation;
4. screen-owned accepted RAM plus three-device overflow hydration;
5. result → detail → Back focus restoration;
6. detail toggle and current-pressure warning;
7. candidate selection and stable focus through in-place refresh;
8. Recheck and stale-generation rejection;
9. Install remaining painted, contained, enabled, and keyboard reachable;
10. `LabScreen.recompose()` replacing RemoteView while accepted facts survive.

Use `_assert_painted_inside` for Back, machine panel, details toggle, candidate, Recheck, selection summary, and Install. Assert no control region escapes its production parent after scrolling.

```python
@pytest.mark.asyncio
async def test_remote_memory_scenarios_survive_recompose_at_80_columns() -> None:
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _models_screen(app)
        remote = await _open_remote_view(screen, pilot)
        await _resolve_remote_for_test(remote, _resolved_remote_model(), pilot)
        screen._apply_machine_memory_result(
            screen._machine_memory_generation,
            _machine_snapshot(total_gib=32, available_gib=10, device_count=3),
        )
        await pilot.pause()
        candidate = remote.query_one(".remote-candidate", Button)
        candidate.focus()
        _assert_painted_inside(app, candidate, screen.query_one("#llm-view-remote"))

        old_remote = remote
        screen.recompose()
        assert await _wait_for(
            lambda: bool(screen.query(RemoteView))
            and screen.query_one(RemoteView) is not old_remote,
            pilot,
        )
        fresh_remote = screen.query_one(RemoteView)
        assert "64K scenario within RAM budget" in _text(fresh_remote)
        assert "VRAM observed on 3 devices" in _text(fresh_remote)
```

- [ ] **Step 3: Run the complete targeted feature suite**

```bash
pytest -q Tests/Model_Artifacts/test_machine_memory.py Tests/Model_Artifacts/test_machine_memory_probe.py Tests/UI/test_model_memory_presenter.py Tests/UI/test_model_remote_view.py Tests/UI/test_llm_screen_lab_adoption.py -k "machine_memory or memory_scenario or drill_down or remote_two_pane_install_action or remote_completion"
pytest -q Tests/UI/test_ui_css_parse.py
ruff check tldw_chatbook/Model_Artifacts/machine_memory.py tldw_chatbook/Model_Artifacts/machine_memory_probe.py tldw_chatbook/UI/Screens/model_memory_presenter.py tldw_chatbook/UI/Screens/model_remote_view.py tldw_chatbook/UI/Screens/llm_screen.py Tests/Model_Artifacts/test_machine_memory.py Tests/Model_Artifacts/test_machine_memory_probe.py Tests/UI/test_model_memory_presenter.py Tests/UI/test_model_remote_view.py Tests/UI/test_llm_screen_lab_adoption.py
python -m compileall -q tldw_chatbook/Model_Artifacts/machine_memory.py tldw_chatbook/Model_Artifacts/machine_memory_probe.py tldw_chatbook/UI/Screens/model_memory_presenter.py tldw_chatbook/UI/Screens/model_remote_view.py tldw_chatbook/UI/Screens/llm_screen.py
git diff --check
```

Expected: all selected tests pass; Ruff, compilation, and diff checks are clean. Do not run the full repository suite without asking the user.

- [ ] **Step 4: Perform one bounded local diagnostic**

On macOS Apple Silicon only, call `observe_machine_memory()` once from the project venv and verify: platform is Darwin, memory kind is unified, total is positive, the Apple shared marker appears once, and no discrete command was attempted. Report values only in the local terminal result; do not add them to logs, fixtures, snapshots, or the task file. Skip with a recorded reason on other hosts.

- [ ] **Step 5: Self-review against the specification and ADR**

Check every spec section against a test or implementation location; scan for bare user-visible `fits`, `compatible`, `supported`, `safe model budget`, Intel DRM, PATH lookup, raw exception logging, or VRAM arithmetic. Verify all added interfaces match the signatures in this plan.

- [ ] **Step 6: Complete Backlog evidence without overstating verification**

Check all acceptance criteria only after their evidence passes. Add concise Implementation Notes listing domain/probe/screen/view/test files, the no-header tradeoff, ADR-080, exact targeted commands/results, and the macOS diagnostic result or skip. Add a lessons entry only if a real reusable incident occurred. Set TASK-20938 to Done only after all Definition-of-Done requirements are satisfied.

- [ ] **Step 7: Commit final evidence and task hygiene**

```bash
git add Tests/UI/test_llm_screen_lab_adoption.py Tests/UI/test_ui_css_parse.py backlog/tasks/task-20938\ -\ Add-hardware-aware-GGUF-machine-fit-estimates.md backlog/docs/lessons-testing-evidence.md
git commit -m "test(models): prove memory scenarios in production layout"
```

If no lessons file changed, omit it from `git add`. If `test_ui_css_parse.py` did not require modification, omit it as well.
