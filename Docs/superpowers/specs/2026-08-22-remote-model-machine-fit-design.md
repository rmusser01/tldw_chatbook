# Remote Models machine-fit estimates

**Date:** 2026-08-22
**Task:** TASK-20938
**ADR:** `backlog/decisions/080-model-machine-memory-fit-estimation.md`
**Status:** revised after UX/architecture critique and approved in chat on 2026-08-22

## Outcome

Remote Models will help first-time and expert users compare an eligible GGUF's
heuristic memory allowance with the current machine before downloading it. The
screen will show transparent 32,768-token and 65,536-token memory scenarios,
the exact machine and model inputs used, and observed VRAM when a bounded
platform probe can provide it.

The estimate is guidance, not admission. Users can still select, review, and
install every eligible candidate. The UI never turns a memory estimate into a
runtime-compatibility, GPU-offload, performance, or successful-inference claim.

## Users and interaction thesis

The surface remains an Operate-mode terminal workbench.

- A first-time user should see one measured answer first: `64K scenario within
  RAM budget`, `32K within budget · 64K crosses reserve`, `32K crosses
  reserve`, `32K exceeds installed RAM`, or `Memory estimate unavailable`.
- An experienced user should be able to inspect exact RAM/unified-memory, RAM
  working budget, volatile available-memory, per-device VRAM, both estimated
  loads, and the policy assumptions without leaving the variant list.
- Both users keep the existing deterministic filename/quantization facts,
  filtering, sorting, exact selection, consent, verification, and runtime
  handoff.

## Scope

This slice includes:

1. immutable provider-neutral machine-memory observation contracts;
2. bounded macOS, Linux, and Windows system-memory and optional VRAM probes;
3. a pure GGUF memory-estimation policy for 32,768- and 65,536-token scenarios;
4. lazy refreshable session observation owned by LLMScreen with
   stale-generation fencing;
5. compact machine evidence and per-candidate estimate copy;
6. production-width keyboard and compositor evidence.

It does not include remote GGUF header range reads, model-context support
verification, runtime installation or
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
  - independent system-memory state: `observed`, `partial`, `unavailable`,
    `permission_denied`, or `unsupported`;
  - independent accelerator state: `observed`, `partial`, `not_observed`,
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
    total estimated bytes, RAM working budget, total physical memory, and one of
    `within_budget`, `over_reserve`, `over_total`, or `unknown`;
  - a current-pressure flag derived independently from volatile available RAM.
- `GGUFMemoryProjection`
  - the paired 32K and 64K estimates plus the primary user-facing outcome.

The exact public bounds are:

- candidate, total-RAM, available-RAM, and observed-VRAM inputs are integers,
  not booleans, from 1 through `2**63 - 1`, except available RAM may be zero;
- a candidate total above `2**63 - 1` remains eligible for download but its
  machine projection is `unknown` rather than truncated or rejected;
- derived estimates are checked against `2**64 - 1`; overflow or contradiction
  fails closed to `unknown` without affecting candidate eligibility;
- architecture and vendor identifiers are at most 32 ASCII characters from
  `[A-Za-z0-9_.-]`; display labels are at most 96 printable characters with
  control characters removed;
- `ProbeReason` is a closed enum containing only `memory_unavailable`,
  `permission_denied`, `unsupported_platform`, `invalid_memory_value`,
  `executable_not_found`, `untrusted_executable`, `command_timeout`,
  `command_failed`, `output_too_large`, `malformed_output`,
  `too_many_devices`, `duplicate_device`, `sysfs_permission_denied`,
  `sysfs_untrusted_path`, and `sysfs_malformed`;
- at most 16 accelerator observations may reach the snapshot.

### Probe boundary

`observe_machine_memory()` accepts injected platform, physical-memory,
filesystem, executable, and command-runner seams. Production defaults use
standard-library functions and the existing required `psutil` dependency.
The LLMScreen session controller separately accepts monotonic- and wall-clock
seams for generation/observed-at presentation. Tests provide deterministic
fakes and never inspect real CI hardware.

The operation is synchronous and side-effect-free except for bounded local
observation. LLMScreen always calls it through a thread worker. It performs no
network request, permission prompt, configuration write, or native ML import.

System memory:

- Supported platforms are Darwin, Linux, and Windows.
- `psutil.virtual_memory()` supplies total and available physical memory.
- Total must be positive and at most `2**63 - 1`.
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

- Linux may run only `/usr/bin/nvidia-smi`. Windows may run only
  `C:\\Windows\\System32\\nvidia-smi.exe` or
  `C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe`.
- The executable must be a regular file at the literal path. Symlinks and
  Windows reparse points are rejected. On Linux the file and its parent must be
  root-owned and not group- or world-writable. No `PATH` lookup occurs.
- The exact argv is `--query-gpu=index,name,memory.total` and
  `--format=csv,noheader,nounits`, with `shell=False`, stderr merged into
  stdout, a two-second timeout, and a 64-KiB combined-output cap.
- Timeout or oversize follows `terminate -> wait up to 250 ms -> kill -> wait`
  so the child is always reaped before returning a fixed reason code.
- The query returns only index, bounded device name, and total memory in MiB.
- Nonzero exit, timeout, malformed rows, duplicate indexes, more than 16
  devices, or oversized output yields unavailable accelerator evidence without
  invalidating known RAM.
- Reported framebuffer memory is observational evidence only. OS reservation,
  vGPU, MIG, and runtime policy can make usable memory different.
- Tests inject the runner; automated tests never execute a host binary.

Linux DRM VRAM:

- At most 16 `cardN` entries are considered.
- Resolved targets must remain under the system-owned `/sys/devices` tree.
- Vendor and `mem_info_vram_total` reads are ASCII, digit-only, and capped at
  64 bytes.
- AMD (`0x1002`) entries with a positive bounded total may be reported as
  `DRM-reported VRAM`. Other vendors and malformed data are ignored. Intel DRM
  support is deferred until a primary kernel contract is identified.
- NVIDIA observations from the system tool and DRM values are never merged or
  summed.

Windows has no AMD/Intel dedicated-memory fallback in v1. Adding DXGI or another
native API later requires evidence and an amendment; unreliable WMI
`AdapterRAM` values are not used.

### Observation states

System memory and accelerator evidence use independent states so incomplete GPU
evidence cannot make trustworthy RAM capacity appear uncertain:

- system `observed`: total and available memory are valid;
- system `partial`: total is valid but available memory is absent or invalid;
- system `unavailable`, `permission_denied`, or `unsupported`: no rating is
  produced;
- accelerator `observed`: at least one bounded device or Apple unified marker
  was observed and every attempted branch settled;
- accelerator `partial`: at least one observation is valid but another enabled
  branch failed;
- accelerator `not_observed`: supported branches settled without valid device
  evidence;
- accelerator `permission_denied` or `unsupported`: accelerator evidence is
  unavailable for the stated fixed reason.

`observed` never claims exhaustive hardware discovery. A valid system-memory
state is sufficient for memory scenarios regardless of accelerator state.

Raw exceptions are converted to fixed reason codes. The probe and UI do not
log exception strings or the snapshot values.

## Estimation policy

The policy uses binary units and integer arithmetic. Percentage allowances are
rounded upward to the next MiB so boundary outcomes are deterministic.

For exact candidate size `W`, the labels `32K` and `64K` mean exactly 32,768
and 65,536 tokens. They are comparison scenarios only; the estimator has not
read the model architecture or verified that the model supports either context.

The v1 allowance remains intentionally simple and inspectable. It is not an
architecture-derived KV-cache calculation and must not be described as
conservative:

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
ram_working_budget = max(0, T - machine_reserve)
```

Each context band is classified independently:

- `within_budget` when estimated bytes are at most the RAM working budget;
- `over_reserve` when estimated bytes exceed the working budget but are at most
  total physical memory;
- `over_total` when estimated bytes exceed total physical memory;
- `unknown` when no valid total physical memory exists.

The primary projection is:

1. 64K within budget → `64K scenario within RAM budget`;
2. 32K within budget, 64K over reserve →
   `32K within budget · 64K crosses reserve`;
3. 32K within budget, 64K over total →
   `32K within budget · 64K exceeds installed RAM`;
4. 32K over reserve → `32K crosses reserve`;
5. 32K over total → `32K exceeds installed RAM`;
6. missing total → `Memory estimate unavailable`.

Current available memory does not influence those stable capacity states. It
does produce a separate pressure warning:

- available at least estimated 64K: no pressure warning;
- available at least estimated 32K but below estimated 64K:
  `64K may need more free RAM now`;
- available below estimated 32K: `32K and 64K need more free RAM now`.

The warning is volatile evidence, never a blocker, and never changes sorting or
the capacity classification. Dedicated VRAM is displayed but not included
because an unselected runtime may use none, some, or multiple devices, and
Apple unified memory is already the system pool.

The panel states the assumptions verbatim: one model, 32,768-token and
65,536-token memory scenarios, heuristic runtime/context allowances, no unusual
runtime options, VRAM not used in the rating, and no model-context support,
runtime compatibility, offload, or performance verification.

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
Machine memory: 32.0 GiB unified · 25.6 GiB RAM working budget
Available now: 21.4 GiB · GPU shares unified memory
Scenarios: 32,768 / 65,536 tokens · model support not checked
Heuristic only · runtime, offload, and speed not checked
[Recheck memory] [Show estimate details]
```

Partial Linux/Windows example:

```text
Machine memory: 64.0 GiB RAM · 51.2 GiB RAM working budget
VRAM observed: NVIDIA RTX 4090 24.0 GiB · not used in this rating
Scenarios: 32,768 / 65,536 tokens · model support not checked
Heuristic only · runtime, offload, and speed not checked
[Recheck memory] [Show estimate details]
```

Unavailable example:

```text
Machine estimate unavailable · filename guidance still applies
[Recheck memory]
```

The presentation builder owns this normative copy matrix:

| Evidence state | Machine panel copy | Candidate copy | Recovery |
| --- | --- | --- | --- |
| first probe active | `Machine memory: Checking local memory…` | `Memory scenario: Checking local memory…` | `Checking…` disabled |
| system observed | exact RAM, available RAM, budget, and limitation lines above | capacity outcome, details, and optional pressure line | `Recheck memory` |
| system partial | `Available now: Not observed · capacity estimate still available` | normal stable outcome; no pressure line | `Recheck memory` |
| system unavailable | `Machine estimate unavailable · filename guidance still applies` | `Memory estimate unavailable · machine memory not observed` | `Recheck memory` |
| system permission denied | `Memory access was denied · filename guidance still applies` | `Memory estimate unavailable · memory access denied` | `Recheck memory` |
| system unsupported | `Machine estimate is not supported on this platform` | `Memory estimate unavailable on this platform` | no automatic retry; `Recheck memory` remains available |
| retained refresh failure | `Recheck failed · using memory observed at HH:MM` plus retained facts | retained outcome plus the same fixed failure line | `Recheck memory` |

Accelerator detail copy is independent:

| Accelerator state/reason | Expanded evidence copy |
| --- | --- |
| observed | `VRAM observed: <device facts> · not used in this rating` |
| partial | `VRAM observed: <device facts> · other accelerator evidence incomplete · not used in this rating` |
| not observed / executable missing | `VRAM not observed · not used in this rating` |
| permission denied / sysfs permission denied | `VRAM access denied · RAM estimate still available` |
| unsupported | `VRAM observation is unavailable on this platform · RAM estimate still available` |
| command timeout | `NVIDIA VRAM check timed out · RAM estimate still available` |
| untrusted executable | `NVIDIA VRAM tool was not used from an untrusted location` |
| malformed, duplicate, excessive, failed command, or unsafe sysfs | `VRAM evidence could not be read safely · RAM estimate still available` |

Important states use text, not color alone. `Recheck memory` remains keyboard
reachable and changes to `Checking…` while disabled for its active generation.
`Show estimate details` is a persistent text-labeled toggle: wide layouts start
expanded, while single-pane narrow layouts start collapsed. Toggling it updates
only estimate Static widgets and never rebuilds candidate controls.

### Candidate estimates

Each candidate row adds a `.remote-fit-outcome` Static followed by an optional
`.remote-fit-details` Static after deterministic quantization guidance and
before Select variant. Examples:

```text
64K scenario within RAM budget · 12.6 GiB headroom
32K est. 9.0 GiB · 64K est. 13.0 GiB · RAM budget 25.6 GiB

32K within budget · 64K crosses reserve · 0.2 GiB over reserve at 64K
32K est. 9.0 GiB · 64K est. 13.0 GiB · RAM budget 12.8 GiB
64K may need more free RAM now

Memory estimate unavailable · machine memory not observed
```

Long exact filenames remain primary. Memory copy never replaces exact size,
quantization, or shard facts. Filter and sort behavior remains local and
unchanged; memory sorting/filtering is outside this slice.

The existing filename-guidance limitation becomes: `Filename-derived general
guidance. Machine memory is estimated below; model-context support and runtime
compatibility remain unverified.`

### Adaptive layout

Layout is based on measured `RemoteView` content width, not the terminal
viewport. At 72 cells or wider, results and detail remain side by side and
estimate details start expanded. Below 72 cells, Remote uses one-pane drill-down:

1. repository results occupy the pane;
2. inspecting a repository saves the exact result focus locator and replaces
   the list with repository detail;
3. detail begins with keyboard-reachable `Back to repositories`;
4. Back restores the exact repository button when it still exists, otherwise
   the search control;
5. starting a new search always returns to the results pane;
6. the outcome remains visible, while exact estimate inputs begin collapsed
   behind the shared `Show estimate details` toggle.

Long filenames wrap within the scrolling detail pane and are never truncated
as source identity. Accelerator evidence shows at most two devices inline;
additional devices use `VRAM observed on N devices · show estimate details` and
remain inspectable in the expanded details. Layout changes preserve discovery,
selection, Recheck, and Install functionality.

### Refresh, ownership, and focus

The recomposition-stable `LLMScreen` owns the process-session capability state:

- the current accepted observation;
- accepted monotonic time and a fixed local observed-at label;
- the current probe generation and Worker handle;
- whether a probe is active;
- the last bounded probe failure code;
- the lazy probe factory injected for tests.

`RemoteView` owns presentation only. It posts a refresh request and accepts an
immutable presentation state from the screen. A first successful repository
resolution requests a probe if the screen has neither an accepted observation
nor an active probe. Recheck increments the generation and starts an exclusive
thread worker on the screen. Completion publishes only when its generation is
current; it may update the screen while no RemoteView is mounted.

Publishing updates the mounted machine panel and existing candidate estimate
statics in place. It must not rebuild a focused repository/candidate button,
filter, sort control, Back control, or install action. Rows created later
through filter or sort read the accepted observation immediately. After
`DeferredViewsMounted`, LLMScreen hydrates the new RemoteView with accepted
facts, refresh state, and failure presentation before accepting another probe.

While refreshing, the previous accepted observation and estimates remain
visible with `Checking…`. A failed, permission-denied, unsupported, or invalid
refresh retains the last valid system-memory snapshot and shows
`Recheck failed · using memory observed at HH:MM`; a successful partial
system-memory refresh may replace it only when its total RAM remains valid. A
first-probe failure shows the fixed unavailable state. Accelerator-only failure
updates its independent evidence state without discarding valid RAM.
Starting a new search does not discard a valid session observation; machine
facts are independent of provider and repository.

The normative transition table is:

| Current state | Probe outcome | Accepted capacity | Presentation |
| --- | --- | --- | --- |
| none | observed/partial system RAM | replace | estimates appear; accelerator state shown independently |
| none | unavailable/denied/unsupported | none | fixed unavailable reason and Recheck |
| accepted | refresh starts | retain | prior estimates plus Checking… |
| accepted | observed/partial system RAM | replace | new estimates and fixed observed-at label |
| accepted | unavailable/denied/unsupported/malformed | retain | prior estimates plus fixed failure/retry copy |
| any | stale generation | unchanged | no presentation mutation |
| any | body recompose | unchanged | hydrate the replacement RemoteView |

## Error, safety, and privacy behavior

- Probes never block discovery, selection, consent, download, or runtime
  handoff.
- Unsupported, unavailable, permission-denied, timed-out, malformed, partial,
  retained-stale, current-pressure, and accelerator-not-observed states have
  distinct fixed copy and recovery defined by the presentation builder.
- Candidate and machine values are validated before rendering with
  `markup=False`.
- Accelerator output is bounded before full accumulation, device count is
  capped at 16, and names are stripped to at most 96 printable characters.
- No shell is used. Executable selection cannot come from `PATH`, repository
  content, configuration, or a remote response.
- Snapshot values and raw errors are not written to logs. Diagnostics may use
  fixed event/reason codes only.
- Observation and estimates disappear when the process exits.

## Verification

### Pure policy tests

- exact integer rounding and GiB/MiB formatting;
- exact 32,768/65,536 scenario semantics and within-budget, over-reserve,
  over-total, and unknown boundaries;
- reserve floor and percentage branch;
- one-byte rounding boundaries and exact derived-value overflow behavior;
- corrupted, negative, boolean-as-integer, over-bound, and contradictory
  values fail closed;
- maximum single/sharded candidate totals remain eligible when projection is
  unknown;
- Apple unified memory is never double-counted;
- VRAM changes do not change the v1 system-memory rating.
- available-memory pressure warnings do not change the stable classification.

### Probe tests

- deterministic Darwin arm64 unified observation with no subprocess;
- Darwin non-arm partial state;
- Linux RAM plus NVIDIA and AMD DRM observations;
- Windows RAM plus trusted NVIDIA observation;
- unsupported platform, psutil unavailable, permission denied, invalid
  available memory, and partial accelerator evidence;
- command timeout, nonzero exit, malformed CSV, duplicate/excess devices,
  excessive output, symlink/reparse path, invalid owner/mode, terminate/kill/reap,
  and invalid sysfs resolution;
- no hostname, serial, UUID, network, persistence, configuration write, or raw
  exception logging.

### Mounted and production UI tests

- lazy first observation and explicit recheck run off the event loop;
- stale generations and unmounted callbacks cannot publish;
- refresh failure retains previous accepted facts;
- the full first-probe/refresh/recompose transition table is covered;
- screen-level ownership survives `LabScreen.recompose()` and hydrates the
  replacement RemoteView without a duplicate probe;
- current-pressure, model-context, runtime, offload, performance, and
  VRAM-not-used disclaimer copy is exact;
- candidate estimate statics update without replacing focused controls;
- filter and sort rebuilds use current facts and preserve existing selection
  rules;
- partial/unavailable facts leave selection and installation enabled;
- real `TldwCli.CSS_PATH` at 80×24 proves single-pane drill-down, Back, panel,
  compact row outcome, details toggle, selection status, Recheck, and Install
  are painted, contained, and keyboard reachable with both rail states;
- long filenames, multi-device overflow, focus restoration, and stable candidate
  widget identity are covered at production width;
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
