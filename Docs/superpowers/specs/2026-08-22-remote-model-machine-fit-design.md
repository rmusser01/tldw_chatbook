# Remote Models machine-fit estimates

**Date:** 2026-08-22
**Task:** TASK-20938
**ADR:** `backlog/decisions/080-model-machine-memory-fit-estimation.md`
**Status:** approved in chat on 2026-08-22

## Outcome

Remote Models will help first-time and expert users judge whether an eligible
GGUF is plausible on the current machine before downloading it. The screen will
show conservative 32K and 64K context estimates, the exact machine and model
inputs used, and observed VRAM when a bounded platform probe can provide it.

The estimate is guidance, not admission. Users can still select, review, and
install every eligible candidate. The UI never turns a memory estimate into a
runtime-compatibility, GPU-offload, performance, or successful-inference claim.

## Users and interaction thesis

The surface remains an Operate-mode terminal workbench.

- A first-time user should see one answer first: `Likely fits at 64K`, `Likely
  at 32K · 64K is close`, `Close call at 32K`, `Unlikely at 32K`, or `Fit
  unknown`.
- An experienced user should be able to inspect exact RAM/unified-memory, safe
  budget, volatile available-memory, per-device VRAM, both estimated loads, and
  the policy assumptions without leaving the variant list.
- Both users keep the existing deterministic filename/quantization facts,
  filtering, sorting, exact selection, consent, verification, and runtime
  handoff.

## Scope

This slice includes:

1. immutable provider-neutral machine-memory observation contracts;
2. bounded macOS, Linux, and Windows system-memory and optional VRAM probes;
3. a pure GGUF memory-estimation policy for 32K and 64K context bands;
4. lazy refreshable observation in RemoteView with stale-generation fencing;
5. compact machine evidence and per-candidate estimate copy;
6. production-width keyboard and compositor evidence.

It does not include remote GGUF header range reads, runtime installation or
probing, GPU-layer recommendations, tensor-split planning, performance scores,
automatic model selection, installation blocking, persisted hardware profiles,
CPU/core-count or speed estimates, or support claims for a model architecture
or quantization.

## Architecture

### Provider-neutral capability module

A new module under `tldw_chatbook/Model_Artifacts/` owns the capability and
policy boundary. It must not import Textual, Hugging Face adapters, llama.cpp,
llamafile, CUDA, Metal, MLX, Torch, or other inference runtimes.

Its public values are frozen, slot-backed dataclasses with strict validation:

- `MachineMemorySnapshot`
  - observation state: `complete`, `partial`, `unavailable`,
    `permission_denied`, or `unsupported`;
  - normalized platform: `darwin`, `linux`, `windows`, or `other`;
  - bounded architecture label;
  - total and optional available physical-memory bytes;
  - memory kind: `unified`, `system`, or `unknown`;
  - zero or more accelerator-memory observations;
  - bounded reason codes, never raw exception text.
- `AcceleratorMemoryObservation`
  - bounded vendor and display label;
  - dedicated bytes, or a shared/unified marker;
  - evidence source: Apple unified memory, NVIDIA system tool, or Linux DRM;
  - no UUID, serial, PCI address, or driver inventory.
- `ContextMemoryEstimate`
  - context length, exact model bytes, runtime allowance, context allowance,
    total estimated bytes, safe budget, total physical memory, and one of
    `likely`, `close`, `unlikely`, or `unknown`.
- `GGUFMachineFitProjection`
  - the paired 32K and 64K estimates plus the primary user-facing outcome.

All byte fields are exact nonnegative integers within a defensive upper bound.
Strings have length and character bounds before they can reach UI or logs.

### Probe boundary

`observe_machine_memory()` accepts injected platform, physical-memory,
filesystem, executable, command-runner, and clock seams. Production defaults
use standard-library functions and the existing required `psutil` dependency.
Tests provide deterministic fakes and never inspect real CI hardware.

The operation is synchronous and side-effect-free except for bounded local
observation. RemoteView always calls it through a thread worker. It performs no
network request, permission prompt, configuration write, or native ML import.

System memory:

- Supported platforms are Darwin, Linux, and Windows.
- `psutil.virtual_memory()` supplies total and available physical memory.
- Total must be positive and within the defensive bound.
- Available memory is retained only when it is between zero and total; an
  invalid available value is omitted and makes the observation partial.
- Other platforms return `unsupported`, even if psutil happens to expose a
  value, so the supported contract stays explicit.

Apple unified memory:

- Darwin on `arm64` or `aarch64` is classified as unified memory.
- Total physical memory is the shared CPU/GPU pool and appears once.
- No NVIDIA, DRM, or native framework probe runs in this branch.
- Darwin on other architectures uses system memory and partial accelerator
  evidence.

NVIDIA VRAM:

- Linux and Windows may run `nvidia-smi` only from an explicit reviewed list of
  absolute system locations.
- The command uses a fixed argv, `shell=False`, a two-second timeout, and a
  reader that terminates the child if combined stdout/stderr exceeds 64 KiB.
- The query returns only index, bounded device name, and total memory in MiB.
- Nonzero exit, timeout, malformed rows, duplicate indexes, excessive device
  count, or oversized output yields unavailable accelerator evidence without
  invalidating known RAM.
- Tests inject the runner; automated tests never execute a host binary.

Linux DRM VRAM:

- At most 16 `cardN` entries are considered.
- Resolved targets must remain under the system-owned `/sys/devices` tree.
- Vendor and `mem_info_vram_total` reads are ASCII, digit-only, and capped at
  64 bytes.
- AMD (`0x1002`) and Intel (`0x8086`) entries with a positive bounded total may
  be reported. Other vendors and malformed data are ignored.
- NVIDIA observations from the system tool and DRM values are never merged or
  summed.

Windows has no AMD/Intel dedicated-memory fallback in v1. Adding DXGI or another
native API later requires evidence and an amendment; unreliable WMI
`AdapterRAM` values are not used.

### Observation states

- `complete`: total memory is valid and every supported observation branch for
  that platform settled successfully, as with Apple unified memory or valid
  results from the enabled system-memory and accelerator probes. This does not
  claim an exhaustive hardware or driver inventory.
- `partial`: total memory is valid, but available-memory or accelerator
  evidence is incomplete. System-memory fit estimates remain available.
- `unavailable`: a supported platform returned no valid total memory.
- `permission_denied`: observation was refused by the operating system.
- `unsupported`: the platform is outside Darwin, Linux, and Windows.

Raw exceptions are converted to fixed reason codes. The probe and UI do not
log exception strings or the snapshot values.

## Estimation policy

The policy uses binary units and integer arithmetic. Percentage allowances are
rounded upward to the next MiB so boundary outcomes are deterministic.

For exact candidate size `W`:

```text
runtime = max(1 GiB, ceil_MiB(W × 10%))
context_32k = max(4 GiB, ceil_MiB(W × 25%))
context_64k = context_32k × 2
estimated_32k = W + runtime + context_32k
estimated_64k = W + runtime + context_64k
```

For total physical memory `T`:

```text
machine_reserve = max(2 GiB, ceil_MiB(T × 20%))
safe_budget = max(0, T - machine_reserve)
```

Each context band is classified independently:

- `likely` when estimated bytes are at most the safe budget;
- `close` when estimated bytes exceed the safe budget but are at most total
  physical memory;
- `unlikely` when estimated bytes exceed total physical memory;
- `unknown` when no valid total physical memory exists.

The primary projection is:

1. 64K likely → `Likely fits at 64K`;
2. 32K likely, 64K close → `Likely at 32K · 64K is close`;
3. 32K likely, 64K unlikely → `Likely at 32K · 64K is unlikely`;
4. 32K close → `Close call at 32K`;
5. 32K unlikely → `Unlikely at 32K`;
6. missing total → `Fit unknown`.

Current available memory does not influence these states. Dedicated VRAM is
displayed but not included because an unselected runtime may use none, some, or
multiple devices, and Apple unified memory is already the system pool.

The panel states the assumptions verbatim: one model, 32K small context, 64K
standard context, heuristic runtime/context allowances, no unusual runtime
options, and no compatibility or performance verification.

## Remote Models interaction

### Machine evidence panel

After a repository resolves, the detail pane places a compact Machine estimate
panel before filename-derived guidance and variant controls.

Initial state:

```text
Machine estimate: Checking local memory…
```

Accepted observation example:

```text
Machine estimate: 32.0 GiB unified · 25.6 GiB safe model budget
Available now: 21.4 GiB · GPU shares unified memory
Assumes one model · 32K small · 64K standard · estimate only
[Refresh machine facts]
```

Partial Linux/Windows example:

```text
Machine estimate: 64.0 GiB RAM · 51.2 GiB safe model budget
VRAM: NVIDIA RTX 4090 24.0 GiB · other accelerators not assessed
Assumes one model · 32K small · 64K standard · estimate only
[Refresh machine facts]
```

Unavailable example:

```text
Machine estimate unavailable · filename guidance still applies
[Refresh machine facts]
```

Important states use text, not color alone. The Refresh action remains
keyboard reachable and is disabled only while its own refresh generation is
active. At narrow widths the facts and action stack vertically inside the
scrolling detail pane.

### Candidate estimates

Each candidate row adds one `.remote-fit-estimate` Static after deterministic
quantization guidance and before Select variant. Examples:

```text
Likely fits at 64K · 32K est. 9.0 GiB · 64K est. 13.0 GiB · safe 25.6 GiB
Likely at 32K · 64K is close · 9.0 / 13.0 GiB est. · safe 12.8 GiB
Fit unknown · machine memory unavailable
```

Long exact filenames remain primary. Fit copy never replaces exact size,
quantization, or shard facts. Filter and sort behavior remains local and
unchanged; fit sorting/filtering is outside this slice.

### Refresh and focus

RemoteView owns:

- the current accepted observation and observation time;
- the current probe generation;
- whether a probe is active;
- a bounded refresh failure code;
- a lazy probe factory injected for tests.

The first successful repository resolution starts a probe if none has been
accepted or started. Refresh increments the generation and starts an exclusive
thread worker. Completion publishes only when its generation is current and the
view remains mounted.

Publishing updates the machine panel and existing candidate estimate statics in
place. It must not rebuild a focused candidate button, filter, sort control, or
install action. Rows created later through filter or sort read the accepted
observation immediately.

While refreshing, the previous accepted observation and estimates remain
visible with `Refreshing…`. A failed refresh retains them, labels their age,
and exposes retry. A first-probe failure shows the fixed unavailable state.
Starting a new search does not discard a valid session observation; machine
facts are independent of provider and repository.

## Error, safety, and privacy behavior

- Probes never block discovery, selection, consent, download, or runtime
  handoff.
- Unsupported, unavailable, permission-denied, timed-out, malformed, and
  partial observations have distinct fixed copy and recovery.
- Candidate and machine values are validated before rendering with
  `markup=False`.
- Accelerator output is bounded before full accumulation, device count is
  capped, and names are stripped to a small printable subset.
- No shell is used. Executable selection cannot come from `PATH`, repository
  content, configuration, or a remote response.
- Snapshot values and raw errors are not written to logs. Diagnostics may use
  fixed event/reason codes only.
- Observation and estimates disappear when the process exits.

## Verification

### Pure policy tests

- exact integer rounding and GiB/MiB formatting;
- 32K/64K likely, close, unlikely, and unknown boundaries;
- safe reserve floor and percentage branch;
- corrupted, negative, boolean-as-integer, over-bound, and contradictory
  values fail closed;
- Apple unified memory is never double-counted;
- VRAM changes do not change the v1 system-memory rating.

### Probe tests

- deterministic Darwin arm64 unified observation with no subprocess;
- Darwin non-arm partial state;
- Linux RAM plus NVIDIA and AMD/Intel DRM observations;
- Windows RAM plus trusted NVIDIA observation;
- unsupported platform, psutil unavailable, permission denied, invalid
  available memory, and partial accelerator evidence;
- command timeout, nonzero exit, malformed CSV, duplicate/excess devices,
  excessive output, untrusted executable path, and invalid sysfs resolution;
- no hostname, serial, UUID, network, persistence, configuration write, or raw
  exception logging.

### Mounted and production UI tests

- lazy first observation and explicit refresh run off the event loop;
- stale generations and unmounted callbacks cannot publish;
- refresh failure retains previous accepted facts;
- candidate estimate statics update without replacing focused controls;
- filter and sort rebuilds use current facts and preserve existing selection
  rules;
- partial/unavailable facts leave selection and installation enabled;
- real `TldwCli.CSS_PATH` at 80×24 proves panel, row estimate, selection status,
  Refresh, and Install are painted, contained, and keyboard reachable;
- CSS bundle reproduction, Ruff, compilation, and diff checks pass.

Automated tests inject all accelerator and platform observations. A local macOS
diagnostic may exercise the real psutil/unified-memory branch, but it is
reported separately and cannot replace deterministic tests.

## Rollback

The feature is presentation-only. Rollback removes the machine probe factory,
panel, and candidate estimate line while leaving deterministic variant
guidance, provider discovery, artifact acquisition, managed storage, and
runtime configuration unchanged. No migration or persisted state requires
cleanup.
